"""Offline monitoring contracts; no Spark connections or model requests."""
import json
import fcntl
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from spark_serve_monitor import (
    AllocationCollector, HostCollector, InferenceMetrics, METRIC_KEYS, Monitor,
    MetricsHTTPError, StatusCommand, _fetch_metrics_path, _remote_program,
    allocation_identity, allocations_from_status, fetch_metrics, parse_prometheus,
    resources_from_sample,
)


def metrics(prompt=100, generation=20, requests=2, latency_sum=1.0, latency_count=2):
    return f'''# TYPE vllm:prompt_tokens_total counter
vllm:num_requests_running{{model_name="test"}} 2
vllm:num_requests_waiting{{model_name="test"}} 1
vllm:kv_cache_usage_perc{{model_name="test"}} 0.42
vllm:prompt_tokens_total{{model_name="test"}} {prompt}
vllm:generation_tokens_total{{model_name="test"}} {generation}
vllm:request_success_total{{model_name="test",finished_reason="stop"}} {requests}
vllm:time_to_first_token_seconds_sum{{model_name="test"}} {latency_sum}
vllm:time_to_first_token_seconds_count{{model_name="test"}} {latency_count}
vllm:time_per_output_token_seconds_sum{{model_name="test"}} {latency_sum / 10}
vllm:time_per_output_token_seconds_count{{model_name="test"}} {latency_count}
'''


def node(role="head", identity="allocation-a", hosts=None, ready=True, runtime="vllm"):
    return {"node": role, "host": "spark-" + role, "allocation_id": identity,
            "allocation_hosts": hosts if hosts is not None else ["spark-" + role],
            "model": "test", "served": "test", "runtime": runtime, "ready": ready,
            "phase": "ready" if ready else "stopping", "url": "http://" + role + ":8000",
            "container_ids": ["container-" + role],
            "containers": [{"id": "unrelated", "running": True}]}


class MetricsTests(unittest.TestCase):
    def test_real_counter_rates_and_window_means(self):
        accumulator = InferenceMetrics()
        first, supported = accumulator.update(metrics(), 10, "test")
        self.assertTrue(supported)
        self.assertEqual(first["kv_cache_percent"], 42)
        self.assertIsNone(first["generation_tokens_per_second"])
        second, _ = accumulator.update(metrics(140, 40, 4, 3, 4), 12, "test")
        self.assertEqual(second["prompt_tokens_per_second"], 20)
        self.assertEqual(second["generation_tokens_per_second"], 10)
        self.assertEqual(second["requests_per_second"], 1)
        self.assertEqual(second["ttft_seconds"], 1)
        self.assertAlmostEqual(second["tpot_seconds"], 0.1)

    def test_reset_decrease_gap_and_explicit_failure_require_baseline(self):
        for reset in ("decrease", "gap", "failure"):
            with self.subTest(reset=reset):
                accumulator = InferenceMetrics(max_gap=6)
                accumulator.update(metrics(), 10, "test")
                if reset == "failure":
                    accumulator.reset()
                value, _ = accumulator.update(metrics(90 if reset == "decrease" else 140, 40, 4),
                                              20 if reset == "gap" else 12, "test")
                self.assertIsNone(value["generation_tokens_per_second"])
                self.assertEqual(value["requests_running"], 2)

    def test_series_disappearance_resets_even_when_aggregate_increases(self):
        accumulator = InferenceMetrics()
        accumulator.update('vllm:generation_tokens_total{engine="0"} 100\n'
                           'vllm:generation_tokens_total{engine="1"} 100', 10)
        value, _ = accumulator.update('vllm:generation_tokens_total{engine="0"} 500', 12)
        self.assertIsNone(value["generation_tokens_per_second"])

    def test_foreign_models_and_nonfinite_values_are_excluded(self):
        body = metrics() + '\nvllm:num_requests_running{model_name="foreign"} 100\n'
        body += 'vllm:num_requests_waiting{model_name="test",engine="1"} NaN\n'
        body += 'vllm:generation_tokens_total{model_name="test",engine="1"} +Inf\n'
        value, _ = InferenceMetrics().update(body, 10, "test")
        self.assertEqual(value["requests_running"], 2)
        self.assertEqual(value["requests_waiting"], 1)
        json.dumps(value, allow_nan=False)

    def test_nim_aliases_are_explicit_and_not_double_counted(self):
        native = metrics().replace("vllm:", "")
        self.assertFalse(parse_prometheus(native, "test"))
        value, supported = InferenceMetrics().update(native + metrics(), 10, "test", runtime="nim")
        self.assertTrue(supported)
        self.assertEqual(value["requests_running"], 2)
        reversed_value, _ = InferenceMetrics().update(metrics() + native, 10, "test", runtime="nim")
        self.assertEqual(value, reversed_value)

    def test_new_tpot_histogram_takes_precedence(self):
        accumulator = InferenceMetrics()
        suffix = '\nvllm:request_time_per_output_token_seconds_sum 1\nvllm:request_time_per_output_token_seconds_count 10'
        accumulator.update(metrics() + suffix, 10)
        suffix = '\nvllm:request_time_per_output_token_seconds_sum 3\nvllm:request_time_per_output_token_seconds_count 20'
        result, _ = accumulator.update(metrics(140, 40, 4, 3, 4) + suffix, 12)
        self.assertAlmostEqual(result["tpot_seconds"], 0.2)

    def test_no_completed_requests_has_no_latency_average(self):
        accumulator = InferenceMetrics()
        accumulator.update(metrics(), 10)
        value, _ = accumulator.update(metrics(), 12)
        self.assertEqual(value["requests_per_second"], 0)
        self.assertIsNone(value["ttft_seconds"])

    def test_unsupported_payload_is_not_zero(self):
        value, supported = InferenceMetrics().update("python_gc_objects_collected_total 100\n", 10)
        self.assertFalse(supported)
        self.assertEqual(value, dict.fromkeys(METRIC_KEYS))

    def test_nim_fallback_is_404_only(self):
        with patch("spark_serve_monitor._fetch_metrics_path", side_effect=[MetricsHTTPError(404), "ok"]) as get:
            self.assertEqual(fetch_metrics("http://worker:8000", "nim"), "ok")
            self.assertEqual([call.args[1] for call in get.call_args_list], ["/v1/metrics", "/metrics"])
            self.assertLessEqual(get.call_args_list[1].args[2], get.call_args_list[0].args[2])
        with patch("spark_serve_monitor._fetch_metrics_path", side_effect=MetricsHTTPError(401)) as get:
            with self.assertRaises(MetricsHTTPError):
                fetch_metrics("http://worker:8000", "nim")
            self.assertEqual(get.call_count, 1)

    def test_absolute_http_deadline_unblocks_connection_close_response(self):
        left, right = socket.socketpair()
        self.addCleanup(left.close)
        self.addCleanup(right.close)

        class Connection:
            sock = left

            def request(self, *_args, **_kwargs):
                pass

            def getresponse(self):
                self.sock = None  # http.client does this for Connection: close.
                return self

            status = 200

            def read(self, _size):
                return left.recv(4096)

            def close(self):
                pass

        with patch("spark_serve_monitor.http.client.HTTPConnection", return_value=Connection()):
            start = time.monotonic()
            with self.assertRaises(TimeoutError):
                _fetch_metrics_path("http://localhost:8000", "/metrics", 0.05)
            self.assertLess(time.monotonic() - start, 1)


class TopologyTests(unittest.TestCase):
    def test_tp2_uses_one_endpoint_and_solo_uses_two(self):
        hosts = ["spark-head", "spark-worker"]
        pair = allocations_from_status({"nodes": [node("worker", hosts=hosts), node(hosts=hosts)]})
        self.assertEqual(len(pair), 1)
        self.assertEqual(pair[0]["nodes"], ["head", "worker"])
        self.assertEqual(pair[0]["endpoint"], "http://head:8000")
        self.assertTrue(pair[0]["ready"])
        solo = allocations_from_status({"nodes": [node(), node("worker", "allocation-b")]})
        self.assertEqual(len(solo), 2)

    def test_yue_shared_generation_is_two_replicas(self):
        pair = allocations_from_status({"nodes": [node(runtime="yue", hosts=[]),
                                                  node("worker", runtime="yue", hosts=[])]})
        self.assertEqual(len(pair), 2)
        self.assertEqual({row["id"] for row in pair}, {"allocation-a:head", "allocation-a:worker"})
        self.assertTrue(all(row["ready"] for row in pair))

    def test_missing_tp2_member_is_not_ready(self):
        result = allocations_from_status({"nodes": [node(hosts=["spark-head", "spark-worker"])]})
        self.assertFalse(result[0]["ready"])

    def test_only_owned_container_changes_reset_identity(self):
        item = node()
        before = allocation_identity(allocations_from_status({"nodes": [item]})[0])
        item["containers"] = [{"id": "other-unrelated", "running": True}]
        same = allocation_identity(allocations_from_status({"nodes": [item]})[0])
        self.assertEqual(before, same)
        item["container_ids"] = ["replacement"]
        after = allocation_identity(allocations_from_status({"nodes": [item]})[0])
        self.assertNotEqual(before, after)

    def test_unified_memory_and_missing_gpu_fields_remain_distinct(self):
        result = resources_from_sample({"cpu": {"total_percent": 45},
                                        "memory": {"used_bytes": 30, "total_bytes": 128},
                                        "gpus": [{"memory_used_bytes": None, "utilization_gpu_percent": 75}]})
        self.assertEqual(result["memory_used_bytes"], 30)
        self.assertEqual(result["memory_total_bytes"], 128)
        self.assertIsNone(result["gpu_memory_used_bytes"])
        self.assertEqual(result["gpu_utilization_percent"], 75)
        self.assertIsNone(result["network_rx_bytes_per_second"])


class FakeHost:
    def __init__(self, cfg, node_id, interval):
        self.node, self.done = node_id, threading.Event()

    def start(self):
        pass

    def stop(self):
        self.done.set()

    def snapshot(self, now):
        return {"node": self.node, "state": "live" if self.node == "head" else "unavailable"}


class FakeAllocation(FakeHost):
    def __init__(self, allocation, interval):
        self.identity, self.done = allocation_identity(allocation), threading.Event()

    def snapshot(self, now):
        return {"metrics_state": "live", "sampled_at": 1000, "error": None,
                "metrics": dict.fromkeys(METRIC_KEYS)}


class MonitorLifecycleTests(unittest.TestCase):
    def monitor(self):
        monitor = Monitor({}, lambda: {}, host_factory=FakeHost, allocation_factory=FakeAllocation)
        monitor.status_at = time.monotonic()
        return monitor

    def test_partial_host_failure_does_not_block_other_host_or_inference(self):
        monitor = self.monitor()
        monitor.status = {"nodes": [node()]}
        snapshot = monitor.snapshot()
        self.assertEqual([host["state"] for host in snapshot["hosts"]], ["live", "unavailable"])
        self.assertEqual(snapshot["allocations"][0]["metrics_state"], "live")

    def test_identity_change_and_status_failure_stop_old_scrapers(self):
        monitor = self.monitor()
        monitor.status = {"nodes": [node()]}
        monitor.snapshot()
        old = next(iter(monitor.allocations.values()))
        monitor.status["nodes"][0]["container_ids"] = ["replacement"]
        monitor.snapshot()
        self.assertTrue(old.done.is_set())
        replacement = next(iter(monitor.allocations.values()))
        monitor.status_error = "status timed out"
        snapshot = monitor.snapshot()
        self.assertTrue(replacement.done.is_set())
        self.assertFalse(snapshot["allocations"][0]["ready"])
        self.assertEqual(snapshot["status_state"], "stale")
        self.assertEqual(snapshot["status_error"], "status timed out")

    def test_stop_cancels_status_reader_and_host_collectors(self):
        done = threading.Event()

        class Status:
            def __call__(self):
                done.wait(2)
                return {"nodes": []}

            def stop(self):
                done.set()

        monitor = Monitor({}, Status(), host_factory=FakeHost, allocation_factory=FakeAllocation)
        monitor.start()
        monitor.stop()
        self.assertTrue(done.is_set())
        self.assertFalse(monitor.thread.is_alive())
        self.assertTrue(all(host.done.is_set() for host in monitor.hosts))

    def test_host_snapshot_distinguishes_stale_from_unavailable(self):
        host = HostCollector({"cluster": {"head": "spark-head"}}, "head", 2)
        self.assertEqual(host.snapshot(10)["state"], "unavailable")
        host.latest = resources_from_sample({})
        host.received_at, host.sampled_at, host.connected = 10, 1000, True
        self.assertEqual(host.snapshot(12)["state"], "live")
        self.assertEqual(host.snapshot(17)["state"], "stale")
        host.connected = False
        self.assertEqual(host.snapshot(11)["state"], "stale")

    def test_remote_program_is_standalone_and_keeps_parent_watchdog(self):
        source = _remote_program(2)
        compile(source, "<remote>", "exec")
        self.assertIn('namespace["TelemetrySampler"]()', source)
        self.assertIn("sys.stdin.buffer.read(1)", source)

    def test_status_timeout_terminates_private_process_group(self):
        original_popen = subprocess.Popen

        def popen(_args, **kwargs):
            return original_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)

        reader = StatusCommand(timeout=0.05)
        with patch("spark_serve_monitor.subprocess.Popen", side_effect=popen):
            with self.assertRaisesRegex(TimeoutError, "Allocation status timed out"):
                reader()
        self.assertIsNone(reader.process)

    def test_status_cleanup_reaches_child_after_leader_has_exited(self):
        original_popen = subprocess.Popen
        with tempfile.TemporaryDirectory() as temp:
            pid_path = Path(temp) / "child.pid"
            child_code = ("import fcntl,os,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                          f"handle=open({str(pid_path)!r},'w'); fcntl.flock(handle,fcntl.LOCK_EX); "
                          "handle.write(str(os.getpid())); handle.flush(); time.sleep(30)")
            parent_code = f"import subprocess,sys; subprocess.Popen([sys.executable,'-c',{child_code!r}])"

            def popen(_args, **kwargs):
                return original_popen([sys.executable, "-c", parent_code], **kwargs)

            reader = StatusCommand(timeout=0.2)
            with patch("spark_serve_monitor.subprocess.Popen", side_effect=popen):
                with self.assertRaises(TimeoutError):
                    reader()
            self.assertTrue(pid_path.exists())
            child_pid = int(pid_path.read_text())
            # Releasing the child's lock proves its descriptors closed even
            # when a zombie briefly awaits the operating system's reaper.
            with pid_path.open() as handle:
                deadline = time.monotonic() + 1
                while True:
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            os.kill(child_pid, signal.SIGKILL)
                            self.fail("status child survived process-group cleanup")
                        time.sleep(0.01)

    def test_sigterm_unblocks_full_stdout_pipe_and_runs_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            ready, stopped = Path(temp) / "ready", Path(temp) / "stopped"
            code = f'''import threading
from pathlib import Path
import spark_serve_monitor as module
class FakeMonitor:
    def __init__(self, *args): self.done = threading.Event()
    def start(self): pass
    def snapshot(self):
        Path({str(ready)!r}).touch()
        return {{"large": "x" * (1024 * 1024)}}
    def stop(self): Path({str(stopped)!r}).touch()
module.Monitor = FakeMonitor
module.watch({{}}, interval=0.01)
'''
            process = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, cwd=Path(__file__).resolve().parents[1])
            try:
                deadline = time.monotonic() + 2
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(ready.exists())
                time.sleep(0.05)  # Fill stdout without consuming any bytes.
                process.terminate()
                self.assertEqual(process.wait(timeout=2), 0)
                self.assertTrue(stopped.exists())
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=2)
                process.stdout.close()
                process.stderr.close()

    def test_allocation_scrape_error_retains_stale_values_and_stops(self):
        calls = 0
        second = threading.Event()

        def fetch(_endpoint, _runtime):
            nonlocal calls
            calls += 1
            if calls == 1:
                return metrics()
            second.set()
            raise OSError("offline")

        allocation = allocations_from_status({"nodes": [node()]})[0]
        collector = AllocationCollector(allocation, 0.01, fetch=fetch)
        collector.start()
        self.assertTrue(second.wait(1))
        collector.stop()
        snapshot = collector.snapshot(time.monotonic())
        self.assertEqual(snapshot["metrics_state"], "stale")
        self.assertEqual(snapshot["metrics"]["requests_running"], 2)
        self.assertFalse(collector.thread.is_alive())


if __name__ == "__main__":
    unittest.main()
