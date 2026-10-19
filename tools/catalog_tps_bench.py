#!/usr/bin/env python3
"""Measure every catalog LLM with one fixed decode protocol for the model cards.

  python3 tools/catalog_tps_bench.py run [model ...]   # starts each model, benchmarks it
  python3 tools/catalog_tps_bench.py summarize          # rewrites measured-performance.json

`run` takes over both Sparks: each model gets a fresh `up --node both --no-hermes`,
an idle settle, a discarded full-length warmup run, then the measured runs below.
Raw runs stay in the normal benchmark history; `summarize` keeps only the newest
completed pair per model and drops hosts, endpoints and container identity.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLI = str(ROOT / "spark-serve")
OUT = ROOT / "measured-performance.json"
SETTLE_SECONDS = 30
PROTOCOL = {
    "prompt_tokens": 512, "max_tokens": 256, "thinking": False,
    "warmup": {"concurrency": 1, "requests": 2},
    "single": {"concurrency": 1, "requests": 5},
    # ds4 caps max_num_seqs at 2, so 2 is the highest concurrency every model accepts.
    "concurrent": {"concurrency": 2, "requests": 6},
}


def log(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def bench(concurrency, requests):
    cmd = [CLI, "bench", "run", "--node", "head", "--kind", "decode",
           "--concurrency", str(concurrency), "--requests", str(requests),
           "--max-tokens", str(PROTOCOL["max_tokens"]),
           "--prompt-tokens", str(PROTOCOL["prompt_tokens"]), "--no-thinking", "--json"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    run = None
    for line in proc.stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("event") == "result":
            run = event["run"]
    if run is None:
        raise RuntimeError(f"bench produced no result (exit {proc.returncode}): {proc.stdout[-400:]} {proc.stderr[-400:]}")
    s = run["summary"]
    log(f"  {run['status']} c={concurrency} n={requests} p50={s['p50']['completion_tokens_per_second_end_to_end']} "
        f"agg={s['aggregate_completion_tokens_per_second_end_to_end']} id={run['id']}")
    return run


def run_model(model):
    log(f"{model}: starting")
    up = subprocess.run([CLI, "up", model, "--node", "both", "--no-hermes"], capture_output=True, text=True)
    if up.returncode:
        log(f"{model}: up failed ({up.returncode}): {(up.stdout + up.stderr)[-800:]}")
        return False
    log(f"{model}: ready; settling {SETTLE_SECONDS}s")
    time.sleep(SETTLE_SECONDS)
    try:
        bench(**PROTOCOL["warmup"])  # discarded
        ok = bench(**PROTOCOL["single"])["status"] == "completed"
        ok = bench(**PROTOCOL["concurrent"])["status"] == "completed" and ok
    except RuntimeError as exc:
        log(f"{model}: {exc}")
        return False
    return ok


def _matches(run, concurrency, requests):
    c = run.get("config", {})
    return (run.get("kind") == "decode" and run.get("status") == "completed"
            and c.get("concurrency") == concurrency and c.get("requests") == requests
            and c.get("max_tokens") == PROTOCOL["max_tokens"]
            and c.get("prompt_tokens") == PROTOCOL["prompt_tokens"]
            and c.get("thinking") is PROTOCOL["thinking"])


def summarize_runs(runs):
    """Newest-first runs -> {model: card summary}; needs a matching single and concurrent run."""
    models = {}
    for run in runs:
        entry = models.setdefault(run["model"], {})
        for key in ("single", "concurrent"):
            if key not in entry and _matches(run, **PROTOCOL[key]):
                entry[key] = run
    out = {}
    for model, entry in sorted(models.items()):
        if set(entry) != {"single", "concurrent"}:
            continue
        single, conc = entry["single"], entry["concurrent"]
        p50 = single["summary"]["p50"]
        out[model] = {
            "measured_at": single["started_at"][:10],
            "engine": single["runtime"]["engine"],
            "single_stream_tokens_per_second": p50["completion_tokens_per_second_end_to_end"],
            "single_stream_post_first_output_tokens_per_second": p50["post_first_output_tokens_per_second_estimate"],
            "single_stream_ttft_seconds": p50["first_model_output_seconds"],
            "concurrent_streams": conc["config"]["concurrency"],
            "concurrent_aggregate_tokens_per_second": conc["summary"]["aggregate_completion_tokens_per_second_end_to_end"],
            "reasoning_observed": any(r.get("reasoning_observed") for r in single["results"] + conc["results"]),
            "run_ids": [single["id"], conc["id"]],
        }
    return out


def write_summary():
    from spark_serve_benchmarks import list_benchmarks
    runs = list_benchmarks({})
    doc = {
        "schema": 1,
        "protocol": {**PROTOCOL, "placement": "both Sparks, fresh start per model",
                     "settle_seconds": SETTLE_SECONDS,
                     "headline": "p50 completion tokens/s per request, end to end (server usage tokens)"},
        "models": summarize_runs(runs),
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    log(f"wrote {OUT} ({len(doc['models'])} models)")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("models", nargs="*")
    sub.add_parser("summarize")
    args = p.parse_args()
    if args.cmd == "summarize":
        write_summary()
        return 0
    models = args.models or json.loads(subprocess.check_output([CLI, "list", "--json"]))
    models = [m if isinstance(m, str) else m["id"] for m in models]
    failed = [m for m in models if not run_model(m)]
    write_summary()
    log(f"done; failed: {failed or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
