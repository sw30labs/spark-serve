"""Pinned TensorFold CUDA health/SSE contracts; no cluster or inference calls."""
import json
import threading
import time
from unittest.mock import Mock, patch

import pytest

from spark_serve_benchmarks import _identity
from spark_serve_monitor import (
    AllocationCollector, InferenceMetrics, MAX_BYTES, METRIC_KEYS, MetricsHTTPError,
    Monitor, _fetch_metrics_path, fetch_metrics, parse_tensorfold_health,
)
from tools.qwen_acceptance import StreamResult, sse_events


def health(prompt=100, completion=20, running=2, **extra):
    # Recipe patch 0001-cuda-live-token-counters.patch at a3aa89835022c55c.
    return json.dumps({"ok": True, "backend": "tensorfold", "busy": running > 0,
                       "requests_running": running, "prompt_tokens_total": prompt,
                       "completion_tokens_total": completion, "prefill_seconds_total": 4.2,
                       "context_length": 262144, **extra})


def update(accumulator, body, at):
    return accumulator.update(body, at, "qwen3.8-flash-next-tensorfold", "tensorfold")


def test_only_reported_counts_and_adjacent_scrape_rates_are_exposed():
    accumulator = InferenceMetrics()
    first, supported = update(accumulator, health(), 10)
    assert supported and first["requests_running"] == 2
    assert first["prompt_tokens_per_second"] is None
    assert first["generation_tokens_per_second"] is None
    second, _ = update(accumulator, health(140, 40, 1), 12)
    assert second["prompt_tokens_per_second"] == 20
    assert second["generation_tokens_per_second"] == 10
    assert second["requests_running"] == 1
    for key in ("requests_waiting", "kv_cache_percent", "requests_per_second", "ttft_seconds", "tpot_seconds"):
        assert second[key] is None
    json.dumps(second, allow_nan=False)


def test_idle_counters_report_zero_rate_but_missing_fields_stay_absent():
    accumulator = InferenceMetrics()
    update(accumulator, health(running=0), 10)
    idle, _ = update(accumulator, health(running=0), 12)
    assert idle["requests_running"] == idle["generation_tokens_per_second"] == 0
    missing = json.dumps({"ok": True, "backend": "tensorfold", "busy": False})
    for at in (14, 16):
        result, supported = update(accumulator, missing, at)
        assert not supported
        assert result == dict.fromkeys(METRIC_KEYS)


@pytest.mark.parametrize("value", [True, False, None, "123", -1, 1.5, 1.0,
                                  float("nan"), float("inf"), 2**53, 10**1000, {}, []])
def test_malformed_counters_are_absent_and_cannot_establish_a_rate(value):
    accumulator = InferenceMetrics()
    update(accumulator, health(), 10)
    body = health(prompt=value, completion=value, requests_running=value)
    result, supported = update(accumulator, body, 12)
    assert not supported and result == dict.fromkeys(METRIC_KEYS)
    recovered, _ = update(accumulator, health(140, 40), 14)
    assert recovered["generation_tokens_per_second"] is None


@pytest.mark.parametrize("condition", ["decrease", "gap", "reset", "missing_counter", "runtime"])
def test_baseline_resets_without_spurious_rates(condition):
    accumulator = InferenceMetrics(max_gap=6)
    update(accumulator, health(), 10)
    if condition == "reset":
        accumulator.reset()
    elif condition == "missing_counter":
        update(accumulator, health(prompt=None), 11)
    elif condition == "runtime":
        accumulator.update("vllm:generation_tokens_total 20", 11)
    result, _ = update(accumulator, health(140, 10 if condition == "decrease" else 40),
                       20 if condition == "gap" else 12)
    assert result["generation_tokens_per_second"] is None
    assert result["prompt_tokens_per_second"] is None
    assert result["requests_running"] == 2


@pytest.mark.parametrize("body", ["not-json", "[]", "null", '{"ok": true}',
                                  '{"backend": "tensorfold", "ok": 1}',
                                  '{"backend": "tensorfold", "ok": false}',
                                  '{"backend": "vllm", "ok": true}', "[" * 2000 + "]" * 2000])
def test_unready_malformed_and_foreign_health_schema_fail_closed(body):
    with pytest.raises(ValueError):
        parse_tensorfold_health(body)


def test_json_size_and_http_response_remain_bounded():
    with pytest.raises(ValueError, match="2 MiB"):
        parse_tensorfold_health(" " * (MAX_BYTES + 1))
    response = Mock(status=200)
    response.read.return_value = b" " * (MAX_BYTES + 1)
    connection = Mock()
    connection.getresponse.return_value = response
    with patch("spark_serve_monitor.http.client.HTTPConnection", return_value=connection):
        with pytest.raises(ValueError, match="2 MiB"):
            _fetch_metrics_path("http://worker:8000", "/health", 1)
    response.read.assert_called_once_with(MAX_BYTES + 1)
    connection.request.assert_called_once_with("GET", "/health", headers={"Accept": "application/json"})
    connection.close.assert_called_once()


def test_tensorfold_scrapes_health_under_existing_deadline_without_fallback():
    with patch("spark_serve_monitor._fetch_metrics_path", return_value=health()) as get:
        assert fetch_metrics("http://worker:8000", "tensorfold") == health()
        assert get.call_args.args[:2] == ("http://worker:8000", "/health")
        assert 0 < get.call_args.args[2] <= 3
    with patch("spark_serve_monitor._fetch_metrics_path", side_effect=MetricsHTTPError(404)) as get:
        with pytest.raises(MetricsHTTPError):
            fetch_metrics("http://worker:8000", "tensorfold")
        assert get.call_count == 1


def test_collector_marks_bad_health_stale_and_resets_before_recovery():
    samples = iter([health(), "[]", health(140, 40)])
    records = []
    allocation = {"id": "a", "endpoint": "http://worker:8000", "runtime": "tensorfold"}
    collector = AllocationCollector(allocation, 0.25, fetch=lambda *_: next(samples))

    def wait(_delay):
        records.append(collector.snapshot(time.monotonic()))
        if len(records) == 3:
            collector.done.set()

    with patch.object(collector.done, "wait", side_effect=wait):
        collector._run()
    assert [record["metrics_state"] for record in records] == ["live", "stale", "live"]
    assert records[-1]["metrics"]["generation_tokens_per_second"] is None


def test_tensorfold_allocation_uses_existing_container_identity_lifecycle():
    def host(_cfg, role, _interval):
        return Mock(done=threading.Event(), snapshot=lambda _now: {"node": role})

    collectors = []

    def allocation_factory(allocation, _interval):
        collector = Mock(done=threading.Event())
        collector.snapshot.return_value = {"metrics_state": "live", "sampled_at": 1,
                                           "metrics": dict.fromkeys(METRIC_KEYS), "error": None}
        collectors.append(collector)
        return collector

    monitor = Monitor({}, host_factory=host, allocation_factory=allocation_factory)
    monitor.status_at = time.monotonic()
    monitor.status = {"nodes": [{"node": "worker", "host": "spark-b", "model": "qwen38-tensorfold",
                                 "runtime": "tensorfold", "ready": True, "phase": "ready",
                                 "url": "http://worker:8000", "allocation_id": "a",
                                 "container_ids": ["container-1"]}]}
    assert monitor.snapshot()["allocations"][0]["metrics_state"] == "live"
    monitor.status["nodes"][0]["container_ids"] = ["container-2"]
    monitor.snapshot()
    collectors[0].stop.assert_called_once()
    assert len(collectors) == 2
    monitor.status["nodes"][0]["ready"] = False
    assert monitor.snapshot()["allocations"][0]["metrics_state"] == "unavailable"
    collectors[1].stop.assert_called_once()


def test_cuda_sse_final_usage_chunk_is_compatible_with_benchmark_accounting():
    # CUDA server v0.3.6.3 places usage alongside the final choice, then [DONE].
    packets = [
        {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"reasoning_content": "Think"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
         "tensorfold": {"prefill_s": 0.4, "decode_s": 0.8, "drafted": 6, "accepted": 4},
         "usage": {"prompt_tokens": 100, "completion_tokens": 8, "total_tokens": 108}},
    ]
    body = "".join("data: " + json.dumps(packet) + "\n\n" for packet in packets) + "data: [DONE]\n\n"
    result = StreamResult(0)
    for event in sse_events(body.encode().splitlines(keepends=True)):
        result.feed(event, 1)
    record = result.finish(2)
    assert record["usage"]["completion_tokens"] == 8
    assert record["metrics"]["completion_tokens_per_second_end_to_end"] == 4
    assert record["content"] == "Hello" and record["reasoning"] == "Think"


@pytest.mark.parametrize("explicit_runtime", [True, False])
def test_benchmark_identity_records_tensorfold_settings_and_fallback(explicit_runtime):
    cfg = {"cluster": {"head": "spark-a", "worker": "spark-b"}, "models": {"m": {
        "wrapper": "tensorfold", "served_name": "model", "image": "image@sha256:pinned",
        "max_model_len": 262144, "tensorfold": {"parallel": 4, "kv_dtype": "int8", "vision": True,
                                               "model_revision": "a" * 40, "HF_TOKEN": "private"}}}}
    owner = {"mode": "vllm", "phase": "ready", "allocation_id": "a", "allocation_hosts": ["spark-b"],
             "containers": [{"id": "container"}]}
    node = {"host": "spark-b", "model": "m", "served": "model", "ready": True, "ours_running": True,
            "phase": "ready", "url": "http://spark-b:8000", "allocation_hosts": ["spark-b"],
            "containers": [{"id": "container", "running": True, "image": "immutable-image",
                            "labels": {"ai.spark-serve.allocation": "a", "ai.spark-serve.model": "m"}}]}
    if explicit_runtime:
        node["runtime"] = "tensorfold"
    identity = _identity(cfg, {"nodes": [node]}, {"spark-b": owner}, "worker")
    assert identity["runtime"] == {
        "engine": "tensorfold", "image": "image@sha256:pinned", "max_model_len": 262144, "settings_source": "catalog",
        "settings": {"parallel": 4, "kv_dtype": "int8", "vision": True, "model_revision": "a" * 40}}
    assert identity["allocation"]["containers"][0]["image"] == "immutable-image"
