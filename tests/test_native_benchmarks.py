"""Offline benchmark isolation, protocol accounting, bounds and cancellation."""
import copy
import io
import json
import threading
import time
from unittest.mock import Mock

import pytest

import spark_serve_benchmarks as bench
from spark_serve_controller import Controller, ControllerError, atomic_json


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARK_SERVE_STATE_DIR", str(tmp_path))
    cfg = {"cluster": {"head": "spark-a", "worker": "spark-b", "lan_url": "http://spark-a:8000"},
           "models": {"m": {"served_name": "model", "max_model_len": 16384, "max_num_seqs": 4,
                              "image": "image@sha256:pinned"}}}
    owners, nodes = {}, []
    for role, host in (("head", "spark-a"), ("worker", "spark-b")):
        owners[host] = {"mode": "vllm", "model": "m", "phase": "ready", "allocation_id": host + "-allocation",
                        "generation": host + "-allocation", "allocation_hosts": [host],
                        "containers": [{"id": host + "-container"}]}
        nodes.append({"host": host, "node": role, "model": "m", "served": "model", "ready": True,
                      "ours_running": True, "phase": "ready", "runtime": "vllm", "url": f"http://{host}:8000",
                      "allocation_hosts": [host], "containers": [{"id": host + "-container", "running": True,
                      "image": "image@sha256:pinned", "labels": {"ai.spark-serve.model": "m",
                      "ai.spark-serve.allocation": host + "-allocation"}}]})
    atomic_json(tmp_path / "controller.json", {"version": 1, "nodes": owners})
    return cfg, {"nodes": nodes}, owners, tmp_path


def response(tokens=100):
    return {"finish_reason": "length", "content": "Synthetic text", "reasoning": "",
            "usage": {"prompt_tokens": 1000, "completion_tokens": tokens, "total_tokens": 1000 + tokens},
            "metrics": {"ttft_seconds": 2, "first_model_output_seconds": 1, "elapsed_seconds": 5,
                        "completion_tokens_per_second_end_to_end": tokens / 5,
                        "post_first_output_tokens_per_second_estimate": tokens / 4,
                        "nonempty_delta_count_not_tokens": 2}}


def run(setup, monkeypatch, **kwargs):
    cfg, status, _, root = setup
    calls, events = [], []
    def request(url, payload, timeout, check):
        check()
        calls.append((url, payload, timeout))
        return response(tokens=10 if len(calls) == 1 else 100)
    monkeypatch.setattr(bench, "_stream_request", request)
    code = bench.run_benchmark(cfg, lambda: copy.deepcopy(status), node="head",
                               emit=lambda e: events.append(copy.deepcopy(e)), **kwargs)
    return code, calls, events, bench.list_benchmarks(cfg)[0]


def test_history_uses_server_tokens_and_excludes_warmup(setup, monkeypatch):
    code, calls, events, report = run(setup, monkeypatch)
    assert code == 0 and len(calls) == 4
    assert report["summary"]["total_completion_tokens"] == 300
    assert report["summary"]["total_prompt_tokens"] == 3000
    assert report["summary"]["p50"]["prefill_tokens_per_second_estimate"] == 1000
    assert report["warmup"]["usage"]["completion_tokens"] == 10
    assert report["allocation"]["containers"][0]["id"] == "spark-a-container"
    assert report["config"]["reasoning_mode"] == "server-default"
    assert all("chat_template_kwargs" not in call[1] for call in calls)
    assert [e["event"] for e in events] == ["started", "progress", "progress", "progress", "result"]
    assert not list((setup[3] / "benchmarks" / "leases").glob("*.json"))


@pytest.mark.parametrize("thinking", [True, False])
def test_explicit_reasoning_mode_is_saved_and_sent(setup, monkeypatch, thinking):
    _, calls, _, report = run(setup, monkeypatch, thinking=thinking)
    assert all(call[1]["chat_template_kwargs"] == {"enable_thinking": thinking} for call in calls)
    assert report["config"]["thinking"] is thinking


def test_prefill_sweep_reports_approximate_target_and_actual_tokens(setup, monkeypatch):
    _, _, _, report = run(setup, monkeypatch, kind="prefill", prompt_tokens=2048)
    assert report["config"]["prompt_lengths"] == [683, 1365, 2048]
    assert [r["usage"]["prompt_tokens"] for r in report["results"]] == [1000] * 3
    assert report["config"]["prompt_token_target_is_approximate"]


@pytest.mark.parametrize("options", [{"concurrency": 5}, {"requests": 13}, {"max_tokens": 513},
                                      {"prompt_tokens": 32769}, {"kind": "prefill", "concurrency": 2},
                                      {"prompt_tokens": 8192}])
def test_bounds_reject_before_sending_any_request(setup, monkeypatch, options):
    cfg, status, _, _ = setup
    request = Mock()
    monkeypatch.setattr(bench, "_stream_request", request)
    with pytest.raises(ControllerError):
        bench.run_benchmark(cfg, lambda: status, node="head", **options)
    request.assert_not_called()


def test_changed_container_identity_rejects_admission(setup, monkeypatch):
    cfg, status, _, _ = setup
    status["nodes"][0]["containers"][0]["id"] = "replacement"
    with pytest.raises(ControllerError, match="container identity"):
        bench.run_benchmark(cfg, lambda: status, node="head")


def test_selected_allocation_replaced_before_admission_never_sends_request(setup, monkeypatch):
    cfg, status, _, _ = setup
    request = Mock()
    monkeypatch.setattr(bench, "_stream_request", request)
    with pytest.raises(ControllerError, match="Selected allocation changed"):
        bench.run_benchmark(cfg, lambda: status, node="head", expected_allocation_id="previous-allocation")
    request.assert_not_called()


def test_shared_allocation_benchmarks_head_once_and_leases_both_hosts(setup, monkeypatch):
    cfg, status, owners, root = setup
    for owner in owners.values():
        owner.update(allocation_hosts=["spark-a", "spark-b"], allocation_id="shared")
    for node in status["nodes"]:
        node["allocation_hosts"] = ["spark-a", "spark-b"]
        node["containers"][0]["labels"]["ai.spark-serve.allocation"] = "shared"
    atomic_json(root / "controller.json", {"version": 1, "nodes": owners})
    urls = []
    def request(url, payload, timeout, check):
        urls.append(url)
        lease = json.loads(next((root / "benchmarks" / "leases").glob("*.json")).read_text())
        assert lease["hosts"] == ["spark-a", "spark-b"]
        return response()
    monkeypatch.setattr(bench, "_stream_request", request)
    assert bench.run_benchmark(cfg, lambda: status, node="worker", emit=lambda e: None) == 0
    assert urls == ["http://spark-a:8000/v1"] * 4


def test_overlapping_lease_blocks_even_after_revocation_until_acknowledged(setup):
    cfg, status, _, root = setup
    atomic_json(root / "benchmarks" / "leases" / "other.json",
                {"hosts": ["spark-a"], "expires_at": time.time() + 30, "revoked": True})
    with pytest.raises(ControllerError, match="already owns"):
        bench.run_benchmark(cfg, lambda: status, node="head")


def test_cancellation_acknowledges_no_more_client_requests_before_return(setup, monkeypatch):
    cfg, status, _, root = setup
    entered = threading.Event()
    calls = []
    def request(url, payload, timeout, check):
        calls.append(payload)
        entered.set()
        while True:
            check()
            time.sleep(0.01)
    monkeypatch.setattr(bench, "_stream_request", request)
    worker = threading.Thread(target=lambda: bench.run_benchmark(cfg, lambda: status, node="head", emit=lambda e: None))
    worker.start()
    assert entered.wait(2)
    assert bench.cancel_benchmarks(cfg, ["spark-a"]) == 1
    worker.join(2)
    assert not worker.is_alive()
    assert len(calls) == 1
    assert bench.list_benchmarks(cfg)[0]["status"] == "cancelled"


def test_controller_identity_change_cancels_before_next_request(setup, monkeypatch):
    cfg, status, owners, root = setup
    calls = []
    def request(url, payload, timeout, check):
        check()
        calls.append(payload)
        owners["spark-a"]["allocation_id"] = "new-allocation"
        atomic_json(root / "controller.json", {"version": 1, "nodes": owners})
        return response()
    monkeypatch.setattr(bench, "_stream_request", request)
    assert bench.run_benchmark(cfg, lambda: status, node="head", emit=lambda e: None) == 1
    assert len(calls) == 1
    assert bench.list_benchmarks(cfg)[0]["status"] == "cancelled"


class FakeConnection:
    def __init__(self, body):
        self.body = io.BytesIO(body)
        self.sock = Mock()
        self.status = 200
    def connect(self): pass
    def request(self, *a, **k): pass
    def getresponse(self): return self
    def getheader(self, *a): return "text/event-stream"
    def readline(self, size): return self.body.readline(size)
    def close(self): pass


def stream_body(usage=True):
    packets = [{"choices": [{"delta": {"content": "many tokens in one chunk"}, "finish_reason": "length"}]}]
    if usage:
        packets.append({"choices": [], "usage": {"prompt_tokens": 99, "completion_tokens": 42, "total_tokens": 141}})
    return ("".join("data: " + json.dumps(p) + "\n\n" for p in packets) + "data: [DONE]\n\n").encode()


def test_stream_accepts_output_cap_but_requires_server_usage(monkeypatch):
    monkeypatch.setattr(bench.http.client, "HTTPConnection", lambda *a, **k: FakeConnection(stream_body()))
    result = bench._stream_request("http://spark-a:8000/v1", {}, 1, lambda: None)
    assert result["usage"]["completion_tokens"] == 42
    assert result["metrics"]["nonempty_delta_count_not_tokens"] == 1
    assert result["finish_reason"] == "length"
    monkeypatch.setattr(bench.http.client, "HTTPConnection", lambda *a, **k: FakeConnection(stream_body(False)))
    with pytest.raises(bench.ProtocolError, match="token usage"):
        bench._stream_request("http://spark-a:8000/v1", {}, 1, lambda: None)


def test_watcher_interrupts_a_blocked_sse_read(monkeypatch):
    entered, cancelled, closed = threading.Event(), threading.Event(), threading.Event()
    class Blocked(FakeConnection):
        def __init__(self):
            super().__init__(b"")
            self.sock.shutdown.side_effect = lambda *a: closed.set()
        def readline(self, size):
            entered.set()
            assert closed.wait(2)
            return b""
    monkeypatch.setattr(bench.http.client, "HTTPConnection", lambda *a, **k: Blocked())
    errors = []
    def check():
        if cancelled.is_set():
            raise bench.BenchmarkCancelled("cancelled")
    def run():
        try:
            bench._stream_request("http://spark-a:8000/v1", {}, 5, check)
        except Exception as exc:
            errors.append(exc)
    worker = threading.Thread(target=run)
    worker.start()
    assert entered.wait(2)
    cancelled.set()
    worker.join(1)
    assert not worker.is_alive()
    assert isinstance(errors[0], bench.BenchmarkCancelled)


def test_revocation_fails_closed_if_a_client_does_not_acknowledge(setup, monkeypatch):
    cfg, _, _, root = setup
    path = root / "benchmarks" / "leases" / "stuck.json"
    atomic_json(path, {"id": "stuck", "hosts": ["spark-a"], "expires_at": time.time() + 30})
    with path.with_suffix(".lock").open("a+") as handle:
        bench.fcntl.flock(handle, bench.fcntl.LOCK_EX)
        clock = iter([0, 6])
        monkeypatch.setattr(bench.time, "monotonic", lambda: next(clock))
        with pytest.raises(ControllerError, match="did not acknowledge"):
            bench.revoke_benchmarks(root, ["spark-a"])
        assert json.loads(path.read_text())["revoked"] is True
        bench.fcntl.flock(handle, bench.fcntl.LOCK_UN)


def test_failed_history_write_still_releases_client_lease(setup, monkeypatch):
    cfg, status, _, root = setup
    save = bench._save
    def fail_history(path, value):
        if path.parent.name == "benchmarks":
            raise OSError("disk full")
        return save(path, value)
    monkeypatch.setattr(bench, "_save", fail_history)
    monkeypatch.setattr(bench, "_stream_request", lambda *a: response())
    with pytest.raises(OSError, match="disk full"):
        bench.run_benchmark(cfg, lambda: status, node="head", emit=lambda e: None)
    assert not list((root / "benchmarks" / "leases").glob("*.json"))


def test_progress_output_delay_is_not_charged_to_measurement(setup, monkeypatch):
    cfg, status, _, _ = setup
    clock = [0.0]
    monkeypatch.setattr(bench.time, "monotonic", lambda: clock[0])
    def request(*args):
        clock[0] += 1
        return response()
    def emit(event):
        if event["event"] == "progress":
            clock[0] += 100  # Simulate a slow GUI/event consumer after completion.
    monkeypatch.setattr(bench, "_stream_request", request)
    assert bench.run_benchmark(cfg, lambda: status, node="head", requests=1, emit=emit) == 0
    report = bench.list_benchmarks(cfg)[0]
    assert report["summary"]["wall_seconds"] == 1
    assert report["summary"]["aggregate_completion_tokens_per_second_end_to_end"] == 100
    assert report["elapsed_seconds"] == 102


def test_final_output_failure_is_not_rethrown_after_history_and_cleanup(setup, monkeypatch):
    cfg, status, _, root = setup
    monkeypatch.setattr(bench, "_stream_request", lambda *a: response())
    def emit(event):
        if event["event"] == "result":
            raise BrokenPipeError("closed consumer")
    assert bench.run_benchmark(cfg, lambda: status, node="head", requests=1, emit=emit) == 1
    assert bench.list_benchmarks(cfg)[0]["status"] == "completed"
    assert not list((root / "benchmarks" / "leases").glob("*.json"))


@pytest.mark.parametrize("terminate", [False, True])
def test_unread_stdout_does_not_hold_lease_or_block_exit(setup, terminate):
    import os
    import subprocess
    import sys
    cfg, status, _, root = setup
    code = '''import json, signal, sys
import spark_serve_benchmarks as bench
cfg, status = json.loads(sys.argv[1])
cfg["models"]["m"]["image"] = "x" * 1000000
bench.EVENT_SECONDS = 0.1 if sys.argv[2] == "false" else 2

def interrupt(*args): raise KeyboardInterrupt()
signal.signal(signal.SIGTERM, interrupt)
def unexpected(*args): raise AssertionError("No inference should run with blocked initial output")
bench._stream_request = unexpected
raise SystemExit(bench.run_benchmark(cfg, lambda: status, node="head"))
'''
    process = subprocess.Popen([sys.executable, "-c", code, json.dumps([cfg, status]), str(terminate).lower()],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    try:
        if terminate:
            deadline = time.time() + 2
            while not list((root / "benchmarks" / "leases").glob("*.json")) and time.time() < deadline:
                time.sleep(.01)
            time.sleep(.05)
            process.terminate()
        assert process.wait(timeout=3) == 1
        assert not list((root / "benchmarks" / "leases").glob("*.json"))
        report = bench.list_benchmarks(cfg)[0]
        assert report["status"] == ("cancelled" if terminate else "failed")
        assert report["warmup"] is None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=1)
        process.stdout.close()
        process.stderr.close()


def test_late_connection_never_posts_after_request_deadline(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(bench.time, "monotonic", lambda: clock[0])
    connection = FakeConnection(stream_body())
    connection.connect = lambda: clock.__setitem__(0, 2.0)
    connection.request = Mock()
    monkeypatch.setattr(bench.http.client, "HTTPConnection", lambda *a, **k: connection)
    with pytest.raises(TimeoutError, match="Connection exhausted"):
        bench._stream_request("http://spark-a:8000/v1", {}, 1, lambda: None)
    connection.request.assert_not_called()


def test_default_event_writer_supports_in_memory_stream(monkeypatch):
    output = io.StringIO()
    monkeypatch.setattr(bench.sys, "stdout", output)
    bench._EventWriter()({"event": "test"})
    assert json.loads(output.getvalue()) == {"event": "test"}
