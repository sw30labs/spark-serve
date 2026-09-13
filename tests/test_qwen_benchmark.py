"""Offline benchmark accounting and deadline checks; no network requests."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

from tools import qwen_benchmark as bench


def args(tmp_path, **overrides):
    values = dict(output=str(tmp_path), model="qwen-test", url="http://localhost:8000/v1",
                  samples=2, timeout=120, deadline_seconds=600, concurrency=False)
    return SimpleNamespace(**{**values, **overrides})


def response(tokens=250, rate=25):
    return {
        "content": "Synthetic complete answer", "finish_reason": "stop",
        "usage": {"prompt_tokens": 100, "completion_tokens": tokens, "total_tokens": 100 + tokens},
        "metrics": {"ttft_seconds": 1, "first_model_output_seconds": 1,
                    "elapsed_seconds": 10, "completion_tokens_per_second_end_to_end": rate,
                    "post_first_output_tokens_per_second_estimate": 27.778,
                    "nonempty_delta_count_not_tokens": 13},
    }


def test_median_uses_successful_requests_and_server_token_usage():
    rows = [{"passed": True, **response(rate=rate)} for rate in (10, 20, 60)]
    rows += [{"passed": False, "error": "truncated"}]
    summary = bench.summarize(rows)
    assert summary["p50"]["completion_tokens_per_second_end_to_end"] == 20
    assert summary["total_completion_tokens"] == 750
    assert summary["successful_requests"] == 3
    assert summary["failed_requests"] == 1


def test_default_suite_excludes_warmup_and_concurrency_from_sequential_medians(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-sent")
    monkeypatch.delenv("SPARK_API_KEY", raising=False)
    calls = []
    def request(url, payload, timeout, api_key):
        calls.append((payload, timeout, api_key))
        return response(tokens=20 if payload["max_tokens"] == 128 else 250)
    monkeypatch.setattr(bench, "stream_request", request)
    assert bench.Benchmark(args(tmp_path, concurrency=True)).run() == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert len(calls) == 9  # Warmup, six serial samples, and two simultaneous requests.
    assert all(api_key is None for _, _, api_key in calls)
    assert all(payload["model"] == "qwen-test" and payload["chat_template_kwargs"]["enable_thinking"] is False for payload, _, _ in calls)
    assert report["sequential_summary"]["total_completion_tokens"] == 1500
    assert report["concurrent_summary"]["total_completion_tokens"] == 500
    assert report["warmup"]["usage"]["completion_tokens"] == 20
    assert len(list(tmp_path.glob("*.request.json"))) == 9
    assert len(list(tmp_path.glob("*.response.json"))) == 9


def test_remaining_global_deadline_bounds_requests_and_exhaustion_never_calls_api(tmp_path, monkeypatch):
    monkeypatch.setattr(bench.time, "monotonic", lambda: 100)
    monkeypatch.setenv("SPARK_API_KEY", "local-test-key")
    request = Mock(return_value=response())
    monkeypatch.setattr(bench, "stream_request", request)
    runner = bench.Benchmark(args(tmp_path))
    runner.deadline = 105
    assert runner.request("bounded", "prose", "fixture")["passed"]
    assert request.call_args.args[2] == 5
    assert request.call_args.kwargs["api_key"] == "local-test-key"
    runner.deadline = 99
    failed = runner.request("expired", "prose", "fixture")
    assert not failed["passed"]
    assert "deadline" in failed["error"]
    assert request.call_count == 1
    assert (tmp_path / "expired.error.json").exists()


def test_warmup_failure_stops_benchmark_and_saves_error(tmp_path, monkeypatch):
    request = Mock(side_effect=RuntimeError("HTTP 503: unavailable"))
    monkeypatch.setattr(bench, "stream_request", request)
    assert bench.Benchmark(args(tmp_path)).run() == 1
    assert request.call_count == 1
    report = json.loads((tmp_path / "report.json").read_text())
    assert not report["passed"]
    assert report["sequential_requests"] == []
    assert "503" in report["warmup"]["error"]
