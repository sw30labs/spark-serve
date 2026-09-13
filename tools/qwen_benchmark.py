#!/usr/bin/env python3
"""Short synthetic throughput measurement for an already-running Qwen server.

Example:
  python3 tools/qwen_benchmark.py --model qwen3.8-flash-next --concurrency

Requires an explicit served model ID. Sends only synthetic prompts, executes no
generated code, and changes no model services. Uses SPARK_API_KEY if set; it does
not read OPENAI_API_KEY. The default is one warmup plus two samples of three
prompts, targeting roughly 1,500–2,000 output tokens (model-dependent). Optional
concurrency adds two substantive requests. The whole run is limited to 10 minutes.

Run the separate acceptance tool for correctness checks. This is a speed sample,
not a general model-quality evaluation or a sustained-load benchmark.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time

if __package__:
    from .qwen_acceptance import stream_request
else:
    from qwen_acceptance import stream_request


PROMPTS = {
    "prose": (
        "Write a clear 180-word explanation for a new software teammate of how a "
        "request travels from a browser through a web server to a database and back. "
        "Mention connection reuse, caching, and one practical source of latency. "
        "Use three short paragraphs and concrete examples. Finish naturally; do not "
        "include a title or a word count."
    ),
    "coding": (
        "Write a self-contained Python function moving_average(values, window). "
        "Return the average of every complete consecutive window using a running "
        "sum. Reject non-positive windows with ValueError and return an empty list "
        "when the window is longer than the input. Include a short docstring, three "
        "small example calls with expected results, and a 60-word explanation of "
        "time and space complexity. Keep the whole answer to about 180 words. "
        "No imports or unrelated discussion."
    ),
    "reasoning": (
        "A fictional workshop has two machines. Machine A makes 24 parts per hour "
        "for 7 hours; machine B makes 18 parts per hour for 6 hours. Five percent of "
        "the combined production is rejected. Each accepted part sells for 20 credits, "
        "and total production costs are 2,400 credits. Write about 180 words explaining "
        "the total output, expected accepted output, revenue, and profit. Treat the "
        "rejection percentage as an expectation so fractional expected parts are "
        "allowed. End with a compact table of the four quantities."
    ),
}
WARMUP = "In two short sentences, explain what a queue does in a computer system."
METRICS = (
    "ttft_seconds", "first_model_output_seconds", "elapsed_seconds",
    "completion_tokens_per_second_end_to_end",
    "post_first_output_tokens_per_second_estimate",
)


def summarize(records):
    """Median of successful individual requests; never count SSE deltas as tokens."""
    valid = [record for record in records if record["passed"]]
    p50 = {}
    for metric in METRICS:
        values = [record["metrics"][metric] for record in valid if record["metrics"].get(metric) is not None]
        p50[metric] = round(statistics.median(values), 4) if values else None
    return {
        "successful_requests": len(valid), "failed_requests": len(records) - len(valid),
        "total_prompt_tokens": sum(record["usage"]["prompt_tokens"] for record in valid),
        "total_completion_tokens": sum(record["usage"]["completion_tokens"] for record in valid),
        "p50": p50,
    }


class Benchmark:
    def __init__(self, args):
        self.args = args
        self.directory = Path(args.output)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.deadline = None
        self.print_lock = threading.Lock()

    def request(self, name, category, prompt, *, warmup=False, barrier=None):
        payload = {
            "model": self.args.model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": 128 if warmup else 768,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        record = {"name": name, "category": category}
        try:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Overall benchmark deadline reached before this request")
            (self.directory / (name + ".request.json")).write_text(json.dumps(payload, indent=2) + "\n")
            if barrier is not None:
                barrier.wait(timeout=min(10, remaining))
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Overall benchmark deadline reached before this request")
            response = stream_request(
                self.args.url, payload, min(self.args.timeout, remaining),
                api_key=os.getenv("SPARK_API_KEY"),
            )
            (self.directory / (name + ".response.json")).write_text(json.dumps(response, indent=2) + "\n")
            if response["finish_reason"] != "stop" or not response["content"].strip():
                raise ValueError("Expected a completed textual answer")
            record.update(passed=True, usage=response["usage"], metrics=response["metrics"])
            if not warmup and response["usage"]["completion_tokens"] < 100:
                record["note"] = "Shorter than the intended substantive output; interpret its speed cautiously."
        except Exception as exc:
            record.update(passed=False, error=f"{type(exc).__name__}: {exc}")
            (self.directory / (name + ".error.json")).write_text(json.dumps(record, indent=2) + "\n")
        with self.print_lock:
            print(json.dumps(record), flush=True)
        return record

    def run(self):
        started = time.monotonic()
        self.deadline = started + self.args.deadline_seconds
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        warmup = self.request("warmup", "warmup", WARMUP, warmup=True)
        sequential = []
        concurrent_records = []
        concurrent_wall_time = None
        if warmup["passed"]:
            # Interleave categories so one category is not always entirely cold.
            for sample in range(1, self.args.samples + 1):
                for category, prompt in PROMPTS.items():
                    sequential.append(self.request(f"{category}-{sample}", category, prompt))
            if self.args.concurrency:
                barrier = threading.Barrier(2)
                group_started = time.monotonic()
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [pool.submit(self.request, f"concurrent-{category}", category, PROMPTS[category], barrier=barrier)
                               for category in ("coding", "reasoning")]
                    concurrent_records = [future.result() for future in futures]
                concurrent_wall_time = time.monotonic() - group_started
        overall = summarize(sequential)
        concurrent_summary = summarize(concurrent_records)
        concurrent_summary["wall_seconds"] = round(concurrent_wall_time, 4) if concurrent_wall_time is not None else None
        concurrent_summary["aggregate_completion_tokens_per_second_end_to_end"] = (
            round(concurrent_summary["total_completion_tokens"] / concurrent_wall_time, 3)
            if concurrent_wall_time and concurrent_summary["successful_requests"] == 2 else None
        )
        report = {
            "model": self.args.model, "url": self.args.url, "started_at": timestamp,
            "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "elapsed_seconds": round(time.monotonic() - started, 4),
            "deadline_seconds": self.args.deadline_seconds, "thinking_enabled": False,
            "samples_per_prompt": self.args.samples,
            "passed": bool(warmup["passed"] and len(sequential) == self.args.samples * len(PROMPTS)
                           and all(record["passed"] for record in sequential + concurrent_records)),
            "warmup": warmup, "sequential_requests": sequential,
            "sequential_summary": overall,
            "by_prompt": {category: summarize([record for record in sequential if record["category"] == category]) for category in PROMPTS},
            "concurrent_requests": concurrent_records, "concurrent_summary": concurrent_summary,
            "limitations": [
                "Speed sample using synthetic prompts; no quality scores or external benchmark comparability implied.",
                "p50 is the median of individual successful requests; failures are reported separately.",
                "Token counts come from server usage, never the number of streamed deltas (MTP may pack multiple tokens).",
                "End-to-end throughput includes prefill and response overhead.",
                "Post-first-output throughput is an estimate: it includes the first delta's unknown token count.",
                "Warmup and concurrent requests are excluded from sequential medians.",
                "Repeated prompts may benefit from prefix caching; these are warm serving measurements.",
                "Generated code is saved for review and never executed.",
            ],
        }
        (self.directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"passed": report["passed"], "sequential_summary": overall,
                          "concurrent_summary": concurrent_summary, "report": str(self.directory / "report.json")}), flush=True)
        return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Exact served model ID; no automatic model selection")
    parser.add_argument("--url", default="http://sparkone.local:8000/v1", help="OpenAI API base URL including /v1")
    parser.add_argument("--output", default="diagnostics/qwen-benchmark-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--samples", type=int, default=2, help="Samples per prompt, 1–3 (default 2)")
    parser.add_argument("--timeout", type=float, default=120, help="Per-request wall deadline, seconds")
    parser.add_argument("--deadline-seconds", type=float, default=600, help="Whole-run deadline, at most 600 seconds")
    parser.add_argument("--concurrency", action="store_true", help="Also submit two substantive requests simultaneously")
    args = parser.parse_args()
    if not args.model.strip():
        parser.error("model must be a nonempty served ID")
    if not 1 <= args.samples <= 3 or not 1 <= args.timeout <= 600 or not 1 <= args.deadline_seconds <= 600:
        parser.error("samples must be 1–3 and deadlines 1–600 seconds")
    return Benchmark(args).run()


if __name__ == "__main__":
    sys.exit(main())
