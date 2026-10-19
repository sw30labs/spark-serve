"""AMD Strix Halo site: a thin client of halo-ctl on the box.

The box owns engine switching (Halogen, Lemonade, vLLM; one at a time) under
its own lock, shared with its GNOME widget. Nothing here touches the Spark
controller, its NCCL settings, fencing or allocation records.

Catalog: [strix] with host, ctl and [strix.models.<id>] entries carrying the
same card fields as Spark models plus engine, engine_model and port.
"""
from __future__ import annotations

import concurrent.futures
import datetime
import json
import subprocess
import time
import uuid

ENGINES = ("halogen", "lemonade", "vllm")
DEFAULT_CTL = "~/.local/bin/halo-ctl"
DEFAULT_KEY = "~/.config/halogen/api-key"
SSH_SECONDS = 30
UP_SECONDS = 1200


class StrixError(RuntimeError):
    pass


def site(cfg: dict) -> dict:
    strix = cfg.get("strix")
    if not strix or not strix.get("host"):
        raise StrixError("no [strix] host in the catalog")
    return strix


def models(cfg: dict) -> dict:
    return (cfg.get("strix") or {}).get("models") or {}


def resolve(cfg: dict, name: str) -> tuple[str, dict]:
    key = name.strip().lower()
    for mid, model in models(cfg).items():
        if key == mid or key in [a.lower() for a in model.get("aliases") or []]:
            return mid, model
    raise StrixError(f"unknown Strix model {name!r}; catalog: {', '.join(sorted(models(cfg)))}")


def ssh(cfg: dict, command: str, timeout: float = SSH_SECONDS) -> str:
    host = site(cfg)["host"]
    try:
        done = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, command],
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise StrixError(f"{host}: no answer within {int(timeout)}s") from exc
    if done.returncode:
        detail = (done.stderr or done.stdout).strip().splitlines()
        raise StrixError(f"{host}: {detail[-1] if detail else f'exit {done.returncode}'}")
    return done.stdout


def _ctl(cfg: dict) -> str:
    return site(cfg).get("ctl") or DEFAULT_CTL


def api_key(cfg: dict) -> str | None:
    """Read the box's API key for this process only; it is never stored on the Mac."""
    path = site(cfg).get("api_key_file") or DEFAULT_KEY
    out = ssh(cfg, f"cat {path} 2>/dev/null || true").strip()
    return out or None


def base_url(cfg: dict, model: dict) -> str:
    return f"http://{site(cfg)['host']}:{int(model['port'])}/v1"


def _match(cfg: dict, engine: str, served: str) -> str | None:
    for mid, model in models(cfg).items():
        if model.get("engine") == engine and model.get("served_name") == served:
            return mid
    return None


# Runs on the box in one SSH round trip: halo-ctl state and stats, then the
# serving engine's real context and token counters, read on localhost with the
# box's key. The key never leaves the box. Read-only.
_PROBE = r"""
import json, os, re, subprocess, urllib.request
CTL, KEY = os.path.expanduser(%(ctl)r), os.path.expanduser(%(key)r)
WANTED = {"vllm:generation_tokens_total", "vllm:prompt_tokens_total", "vllm:num_requests_running",
          "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc", "llamacpp:tokens_predicted_total",
          "llamacpp:prompt_tokens_total", "llamacpp:requests_processing", "llamacpp:requests_deferred",
          "llamacpp:kv_cache_usage_ratio", "llamacpp:n_decode_total"}
def sh(*args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return ""
key = open(KEY).read().strip() if os.path.exists(KEY) else ""
def get(url):
    request = urllib.request.Request(url, headers={"Authorization": "Bearer " + key} if key else {})
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.read(4 << 20).decode("utf-8", "replace")
    except Exception:
        return ""
state = json.loads(sh(CTL, "status", "--json"))
try:
    stats = json.loads(sh(CTL, "stats") or "{}")
except ValueError:
    stats = {}
engine, port, served, text = state.get("engine"), state.get("port"), None, ""
if engine in ("halogen", "vllm") and port:
    try:
        model = json.loads(get("http://127.0.0.1:%%s/v1/models" %% port))["data"][0]
        served = model.get("max_model_len") or model.get("context_length")
    except Exception:
        pass
    text = get("http://127.0.0.1:%%s/metrics" %% port)
elif engine == "lemonade" and port:
    try:
        loaded = (json.loads(get("http://127.0.0.1:%%s/api/v1/health" %% port)).get("all_models_loaded") or [{}])[0]
        served = loaded.get("max_context_window")
        if loaded.get("backend_url"):
            text = get(loaded["backend_url"].rsplit("/v1", 1)[0] + "/metrics")
    except Exception:
        pass
counters = {}
for line in text.splitlines():
    match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", line.strip())
    if match and match.group(1) in WANTED:
        name, value = match.group(1), float(match.group(2))
        counters[name] = max(counters.get(name, 0.0), value) if "kv_cache" in name else counters.get(name, 0.0) + value
print(json.dumps({"state": state, "stats": stats, "served_context": served, "counters": counters}))
"""

# Engine counters normalized to one shape; Halogen keeps halogen-stats' own fields.
_COUNTERS = {
    "vllm": {"active": "vllm:num_requests_running", "queued": "vllm:num_requests_waiting",
             "kv": "vllm:kv_cache_usage_perc", "generation": "vllm:generation_tokens_total",
             "prompt": "vllm:prompt_tokens_total"},
    "lemonade": {"active": "llamacpp:requests_processing", "queued": "llamacpp:requests_deferred",
                 # tokens_predicted_total moves only when a request finishes; decode steps
                 # move live and equal tokens with one slot and no drafts, as Lemonade runs here.
                 "kv": "llamacpp:kv_cache_usage_ratio", "generation": "llamacpp:n_decode_total",
                 "prompt": "llamacpp:prompt_tokens_total"},
}


def _metrics(engine, stats, counters):
    out = dict.fromkeys(("active", "queued", "kv_pct", "avg_gen_tps", "avg_prompt_tps", "last_gen_tps",
                         "draft_accept_pct", "generation_tokens_total", "prompt_tokens_total"))
    if engine == "halogen":
        out.update({key: stats.get(key) for key in (
            "active", "queued", "kv_pct", "avg_gen_tps", "avg_prompt_tps", "last_gen_tps", "draft_accept_pct")})
    names = _COUNTERS.get(engine)
    if names:
        kv = counters.get(names["kv"])
        out.update(active=counters.get(names["active"]), queued=counters.get(names["queued"]),
                   kv_pct=round(kv * 100, 2) if kv is not None else None,
                   generation_tokens_total=counters.get(names["generation"]),
                   prompt_tokens_total=counters.get(names["prompt"]))
    return out


def status(cfg: dict) -> dict:
    """halo-ctl status, resources and the engine's live context and counters, catalog id resolved."""
    script = _PROBE % {"ctl": _ctl(cfg), "key": site(cfg).get("api_key_file") or DEFAULT_KEY}
    raw = json.loads(ssh(cfg, "python3 - <<'SPARK_SERVE_PROBE'\n" + script + "\nSPARK_SERVE_PROBE").strip().splitlines()[-1])
    state, stats = raw.get("state") or {}, raw.get("stats") or {}
    engine = state.get("engine")
    model_id = _match(cfg, engine, state.get("model")) if engine in ENGINES else None
    url = base_url(cfg, models(cfg)[model_id]) if model_id else None
    return {"host": site(cfg)["host"], **state, "model_id": model_id, "url": url,
            "served_context": raw.get("served_context"),
            "resources": {key: stats.get(key) for key in (
                "ram_free_gib", "ram_total_gib", "gtt_used_gib", "gtt_total_gib",
                "busy_pct", "temp_c", "power_w", "last_gen_tps")},
            "metrics": _metrics(engine, stats, raw.get("counters") or {})}


def up(cfg: dict, model_id: str) -> dict:
    mid, model = resolve(cfg, model_id)
    engine = model.get("engine")
    if engine not in ENGINES:
        raise StrixError(f"{mid}: engine must be one of {', '.join(ENGINES)}")
    args = [engine]
    if engine == "lemonade":
        args.append(model["served_name"])
    elif engine == "vllm":
        args += [model["engine_model"], str(int(model.get("max_model_len") or 0) or "")]
    ssh(cfg, " ".join([_ctl(cfg), "up", *[a for a in args if a]]), timeout=UP_SECONDS)
    current = status(cfg)
    if current.get("model_id") != mid or current.get("phase") != "ready":
        raise StrixError(f"{mid} did not become ready (box reports {current.get('engine')}/{current.get('phase')})")
    return current


def stop(cfg: dict) -> dict:
    ssh(cfg, f"{_ctl(cfg)} stop", timeout=120)
    return status(cfg)


def catalog(cfg: dict, performance: dict) -> list[dict]:
    """Card entries in the same shape as `spark-serve list --json`, ranked within this site."""
    from spark_serve_ranking import modalities, score_models
    out = []
    for mid, m in models(cfg).items():
        out.append({
            "id": mid, "label": m.get("label") or mid, "aliases": list(m.get("aliases") or []),
            "served_name": m.get("served_name", ""), "ctx": int(m.get("max_model_len") or 0),
            "image": m.get("engine_model") or m.get("engine", ""), "notes": m.get("notes", ""),
            "backend": m.get("engine", ""), "topology": "strix", "wrapper": m.get("engine", ""),
            "hermes_provider": "", "performance": performance.get(mid),
            "quant": m.get("quant"), "quant_bits": m.get("quant_bits"),
            "quant_scope": m.get("quant_scope"), "quant_detail": m.get("quant_detail"),
            "modalities": modalities(m),
        })
    ranking = score_models(out)
    for item in out:
        item["ranking"] = ranking.get(item["id"])
    return out


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def bench(cfg: dict, *, concurrency=1, requests=3, max_tokens=128, prompt_tokens=512,
          thinking=None, emit=None) -> int:
    """Benchmark whatever catalog model is serving, under the Spark runner's protocol.

    Reuses its SSE accounting, prompts and summary so results are comparable and
    land in the same history (and `catalog_tps_bench.py summarize`).
    """
    from spark_serve_benchmarks import (DEADLINE_SECONDS, REQUEST_SECONDS, _prompt, _root, _save,
                                        _stream_request, _summary)
    emit = emit or (lambda event: print(json.dumps(event), flush=True))
    for name, value, low, high in (("concurrency", concurrency, 1, 4), ("requests", requests, 1, 12),
                                  ("max_tokens", max_tokens, 16, 512), ("prompt_tokens", prompt_tokens, 32, 32768)):
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise StrixError(f"benchmark {name} must be in {low}..{high}")
    if concurrency > requests:
        raise StrixError("benchmark concurrency cannot exceed request count")
    before = status(cfg)
    mid = before.get("model_id")
    if not mid or before.get("phase") != "ready":
        raise StrixError("no catalog model is ready on the Strix box")
    model = models(cfg)[mid]
    key = api_key(cfg)
    run_id = uuid.uuid4().hex
    run = {"id": run_id, "site": "strix", "kind": "decode", "status": "running", "started_at": _now(),
           "model": mid, "served_name": model["served_name"], "node": "strix", "hosts": [before["host"]],
           "endpoint": before["url"], "runtime": {"engine": model["engine"]},
           "served_context": before.get("served_context"),
           "config": {"concurrency": concurrency, "requests": requests, "max_tokens": max_tokens,
                      "prompt_tokens": prompt_tokens, "prompt_lengths": [prompt_tokens] * requests,
                      "thinking": thinking,
                      "reasoning_mode": "server-default" if thinking is None else "requested-on" if thinking else "requested-off",
                      "deadline_seconds": DEADLINE_SECONDS, "prompt_token_target_is_approximate": True},
           "warmup": None, "results": [],
           "limitations": ["Synthetic speed sample, not a quality evaluation.",
                           "Token counts are server usage, never streamed chunk counts.",
                           "Identity is checked before and after the run, not continuously."]}
    started = time.monotonic()
    deadline = started + DEADLINE_SECONDS
    completion_times = []

    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("Overall benchmark deadline exceeded")

    def request(index, warmup=False):
        record = {"index": index, "phase": "warmup" if warmup else "measurement",
                  "requested_prompt_tokens": prompt_tokens}
        try:
            payload = {"model": model["served_name"],
                       "messages": [{"role": "user", "content": _prompt(prompt_tokens, uuid.uuid4().hex)}],
                       "temperature": 0, "max_tokens": min(32, max_tokens) if warmup else max_tokens,
                       "stream": True, "stream_options": {"include_usage": True}}
            if thinking is not None:
                payload["chat_template_kwargs"] = {"enable_thinking": thinking}
            response = _stream_request(before["url"], payload,
                                       min(REQUEST_SECONDS, deadline - time.monotonic()), check, api_key=key)
            first = response["metrics"].get("first_model_output_seconds")
            response["metrics"]["prefill_tokens_per_second_estimate"] = (
                round(response["usage"]["prompt_tokens"] / first, 3) if first and first > 0 else None)
            record.update(passed=True, usage=response["usage"], metrics=response["metrics"],
                          finish_reason=response["finish_reason"], reasoning_observed=bool(response.get("reasoning")))
        except Exception as exc:
            record.update(passed=False, cancelled=False, error=f"{type(exc).__name__}: {exc}")
        if not warmup:
            completion_times.append(time.monotonic())
        return record

    measurement_started = None
    try:
        emit({"event": "started", "run": {k: v for k, v in run.items() if k != "results"}})
        run["warmup"] = request(-1, warmup=True)
        if run["warmup"]["passed"]:
            measurement_started = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                for future in concurrent.futures.as_completed([pool.submit(request, i) for i in range(requests)]):
                    run["results"].append(future.result())
                    emit({"event": "progress", "completed": len(run["results"]), "total": requests})
        after = status(cfg)
        rows = [run["warmup"], *run["results"]]
        if after.get("model_id") != mid or after.get("phase") != "ready":
            run.update(status="failed", error="Serving model changed during the benchmark")
        else:
            run["status"] = "completed" if len(run["results"]) == requests and all(r["passed"] for r in rows) else "failed"
    except Exception as exc:
        run.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        run["finished_at"] = _now()
        run["elapsed_seconds"] = round(time.monotonic() - started, 4)
        run["results"].sort(key=lambda r: r["index"])
        wall = max(completion_times) - measurement_started if completion_times and measurement_started else 0
        run["summary"] = _summary(run["results"], wall)
        (_root() / "benchmarks").mkdir(parents=True, exist_ok=True)
        _save(_root() / "benchmarks" / (run_id + ".json"), run)
        emit({"event": "result", "run": run})
    return 0 if run["status"] == "completed" else 1


def _emit_json(value):
    print(json.dumps(value), flush=True)


def main(cfg: dict, args, performance: dict) -> int:
    """`spark-serve strix …` entry point; JSON on stdout, errors as an error event."""
    try:
        if args.strix_cmd == "list":
            items = catalog(cfg, performance)
            if args.json:
                print(json.dumps(items, indent=2))
            else:
                for item in items:
                    star = " ★" if (item.get("ranking") or {}).get("best") else ""
                    print(f"{item['id']:<28} {item['backend']:<9} {item['served_name']:<40}{star}")
        elif args.strix_cmd == "status":
            current = status(cfg)
            if args.json:
                _emit_json(current)
            else:
                print(f"{current['host']}: {current['engine']} {current['phase']}"
                      + (f" · {current['model']} · {current['url']}" if current.get("url") else ""))
        elif args.strix_cmd == "up":
            _emit_json({"event": "starting", "model": args.model})
            _emit_json({"event": "ready", "status": up(cfg, args.model)})
        elif args.strix_cmd == "stop":
            _emit_json({"event": "stopped", "status": stop(cfg)})
        elif args.strix_cmd == "bench":
            return bench(cfg, concurrency=args.concurrency, requests=args.requests,
                         max_tokens=args.max_tokens, prompt_tokens=args.prompt_tokens, thinking=args.thinking)
        return 0
    except (StrixError, ValueError, KeyError) as exc:
        _emit_json({"event": "error", "detail": str(exc)})
        return 1


def add_parser(sub) -> None:
    strix = sub.add_parser("strix", help="AMD Strix Halo: one engine at a time through halo-ctl on the box")
    commands = strix.add_subparsers(dest="strix_cmd", required=True)
    lst = commands.add_parser("list", help="Strix catalog with measured speed and ranking")
    lst.add_argument("--json", action="store_true")
    st = commands.add_parser("status", help="serving engine, model and resources")
    st.add_argument("--json", action="store_true")
    upc = commands.add_parser("up", help="switch the box to a catalog model (stops the other engines)")
    upc.add_argument("model")
    upc.add_argument("--json", action="store_true", help="JSON events are the default output")
    stp = commands.add_parser("stop", help="stop every engine on the box")
    stp.add_argument("--json", action="store_true", help="JSON events are the default output")
    b = commands.add_parser("bench", help="decode benchmark of the serving model (same protocol as Sparks)")
    b.add_argument("--concurrency", type=int, default=1)
    b.add_argument("--requests", type=int, default=3)
    b.add_argument("--max-tokens", type=int, default=128)
    b.add_argument("--prompt-tokens", type=int, default=512)
    thinking = b.add_mutually_exclusive_group()
    thinking.add_argument("--thinking", dest="thinking", action="store_true")
    thinking.add_argument("--no-thinking", dest="thinking", action="store_false")
    b.set_defaults(thinking=None)
    b.add_argument("--json", action="store_true", help="JSON events are the default output")

