"""Read-only, app-lifetime telemetry. No remote installation or lifecycle writes.

Host timestamps are local receipt times, so clock skew between Sparks cannot
hide stale data. Inference latency values are means over the adjacent scrape
window, never lifetime averages or percentiles. GPU memory is reported apart
from authoritative /proc node memory because Sparks use unified memory.
"""
from __future__ import annotations

import json
import http.client
import math
import os
import re
import selectors
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from collections import Counter
from urllib.parse import urlsplit


RESOURCE_KEYS = (
    "cpu_percent", "memory_used_bytes", "memory_total_bytes", "swap_used_bytes",
    "gpu_utilization_percent", "gpu_memory_used_bytes", "gpu_memory_total_bytes",
    "gpu_temperature_c", "gpu_power_watts", "network_rx_bytes_per_second",
    "network_tx_bytes_per_second", "disk_read_bytes_per_second", "disk_write_bytes_per_second",
)
METRIC_KEYS = (
    "requests_running", "requests_waiting", "kv_cache_percent",
    "prompt_tokens_per_second", "generation_tokens_per_second", "requests_per_second",
    "ttft_seconds", "tpot_seconds",
)
GAUGES = {
    "requests_running": ("vllm:num_requests_running",),
    "requests_waiting": ("vllm:num_requests_waiting",),
    "kv_cache_percent": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
}
COUNTERS = {
    "prompt_tokens_per_second": "vllm:prompt_tokens_total",
    "generation_tokens_per_second": "vllm:generation_tokens_total",
    "requests_per_second": "vllm:request_success_total",
}
HISTOGRAMS = {
    "ttft_seconds": ("vllm:time_to_first_token_seconds",),
    "tpot_seconds": ("vllm:request_time_per_output_token_seconds", "vllm:time_per_output_token_seconds"),
}
FAMILIES = {name for names in GAUGES.values() for name in names} | set(COUNTERS.values()) | {
    name + suffix for names in HISTOGRAMS.values() for name in names for suffix in ("_sum", "_count")
}
_SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+([^\s]+)(?:\s+[^\s]+)?$')
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"(?:,|$)')
MAX_BYTES = 2 * 1024 * 1024


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def resources_from_sample(sample):
    """Flatten a TelemetrySampler sample without inventing unsupported counters."""
    result = dict.fromkeys(RESOURCE_KEYS)
    result["cpu_percent"] = _number(sample.get("cpu", {}).get("total_percent"))
    memory = sample.get("memory", {})
    for dest, source in (("memory_used_bytes", "used_bytes"), ("memory_total_bytes", "total_bytes"),
                         ("swap_used_bytes", "swap_used_bytes")):
        result[dest] = _number(memory.get(source))
    # A Spark has one GPU. For multi-GPU hosts, utilization/temperature use the
    # busiest/hottest device while memory and power are summed only if complete.
    gpus = sample.get("gpus") or []
    for dest, source, sum_values in (
        ("gpu_utilization_percent", "utilization_gpu_percent", False),
        ("gpu_memory_used_bytes", "memory_used_bytes", True),
        ("gpu_memory_total_bytes", "memory_total_bytes", True),
        ("gpu_temperature_c", "temperature_gpu_c", False),
        ("gpu_power_watts", "power_draw_watts", True),
    ):
        values = [_number(gpu.get(source)) for gpu in gpus]
        if values and all(value is not None for value in values):
            result[dest] = sum(values) if sum_values else max(values)
    for domain, directions in (("network", ("rx", "tx")), ("disk", ("read", "write"))):
        for direction in directions:
            result[f"{domain}_{direction}_bytes_per_second"] = _number(
                sample.get(domain, {}).get(f"total_{direction}_bytes_per_second"))
    return result


def parse_prometheus(body, served_name=None, runtime="vllm"):
    """Read only explicitly supported vLLM families, including NIM exposing them.

    Model-labelled samples for a different model are excluded. Series identity
    includes all labels, so a worker disappearing cannot create a false rate.
    """
    if len(body.encode("utf-8")) > MAX_BYTES:
        raise ValueError("metrics response exceeds 2 MiB")
    values = {}
    for line in body.splitlines():
        match = _SAMPLE.fullmatch(line.strip())
        if not match:
            continue
        family = match[1]
        if runtime == "nim" and not family.startswith("vllm:"):
            family = "vllm:" + family
        if family not in FAMILIES:
            continue
        labels, offset = {}, 0
        raw_labels = match[2] or ""
        while offset < len(raw_labels):
            label = _LABEL.match(raw_labels, offset)
            if not label:
                break
            labels[label[1]] = re.sub(r'\\([\\"n])', lambda m: "\n" if m[1] == "n" else m[1], label[2])
            offset = label.end()
        if offset != len(raw_labels):
            continue
        if served_name and labels.get("model_name", served_name) != served_name:
            continue
        try:
            value = _number(float(match[3]))
        except ValueError:
            continue
        if value is not None:
            key = (family, tuple(sorted(labels.items())))
            # NIM can expose both names during engine/version transitions.
            # Prefer the canonical family instead of adding aliases together.
            if match[1].startswith("vllm:") or key not in values:
                values[key] = value
    return values


class InferenceMetrics:
    """Counter baselines live only for one allocation/container/endpoint identity."""

    def __init__(self, max_gap=15.0):
        self.max_gap = max_gap
        self.reset()

    def reset(self):
        self.previous = {}
        self.previous_at = None

    def update(self, body, now, served_name=None, runtime="vllm"):
        values = parse_prometheus(body, served_name, runtime)
        result = dict.fromkeys(METRIC_KEYS)
        for output, names in GAUGES.items():
            for name in names:
                series = [value for (family, _), value in values.items() if family == name]
                if series:
                    result[output] = min(100.0, max(series) * 100) if output == "kv_cache_percent" else sum(series)
                    break
        counters = {key: value for key, value in values.items()
                    if key[0] not in {name for names in GAUGES.values() for name in names}}
        elapsed = now - self.previous_at if self.previous_at is not None else 0
        continuous = (0 < elapsed <= self.max_gap and counters.keys() == self.previous.keys()
                      and all(value >= self.previous[key] for key, value in counters.items()))
        if continuous:
            deltas = {key: value - self.previous[key] for key, value in counters.items()}

            def total(family):
                rows = [value for (name, _), value in deltas.items() if name == family]
                return sum(rows) if rows else None

            for output, name in COUNTERS.items():
                value = total(name)
                result[output] = value / elapsed if value is not None else None
            for output, names in HISTOGRAMS.items():
                for name in names:
                    count, duration = total(name + "_count"), total(name + "_sum")
                    if count is not None and duration is not None:
                        result[output] = duration / count if count else None
                        break
        self.previous, self.previous_at = counters, now
        return result, bool(values)


def allocation_identity(allocation):
    return (allocation["id"], allocation["endpoint"], allocation["runtime"],
            tuple(allocation.get("_containers", ())), allocation.get("served_name"))


def allocations_from_status(status):
    """One logical allocation for TP2, independent allocations for solo models."""
    groups = {}
    for node in status.get("nodes", []):
        if not node.get("model"):
            continue
        hosts = sorted(node.get("allocation_hosts") or [node["host"]])
        identity = str(node.get("allocation_id") or "legacy:" + node["model"] + ":" + ",".join(hosts))
        groups.setdefault((identity, tuple(hosts)), []).append(node)
    allocations = []
    counts = Counter(identity for identity, _hosts in groups)
    for (identity, _hosts), members in groups.items():
        members.sort(key=lambda node: node["node"] != "head")
        leader = members[0]
        expected = set(leader.get("allocation_hosts") or [leader["host"]])
        ready = (all(member.get("ready") for member in members)
                 and {member["host"] for member in members} == expected)
        allocations.append({
            "id": identity if counts[identity] == 1 else identity + ":" + leader["node"],
            "model": leader["model"], "served_name": leader.get("served"),
            "runtime": leader.get("runtime") or leader.get("backend") or "unknown",
            "nodes": [member["node"] for member in members], "endpoint": leader.get("url", ""),
            "ready": ready, "phase": next((member.get("phase") or "unknown" for member in members
                                           if member.get("phase") != "ready"), "ready"),
            "_containers": sorted((member["host"], container_id)
                                  for member in members for container_id in member.get("container_ids", [])),
        })
    return allocations


def _remote_program(interval):
    source = (Path(__file__).parent / "spark_bench" / "telemetry.py").read_text()
    return f'''import sys, threading, time, signal, json
namespace = {{"__name__": "spark_serve_remote_telemetry"}}
exec(compile({source!r}, "<spark-serve-telemetry>", "exec"), namespace)
done = threading.Event()
for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, lambda *_: done.set())
def parent_closed():
    sys.stdin.buffer.read(1)
    done.set()
threading.Thread(target=parent_closed, daemon=True).start()
sampler = namespace["TelemetrySampler"]()
try:
    while not done.is_set():
        start = time.monotonic()
        sample = sampler.sample()
        if done.is_set():
            break
        print(json.dumps(sample, allow_nan=False, separators=(",", ":")), flush=True)
        done.wait(max(0, {interval!r} - (time.monotonic() - start)))
except (BrokenPipeError, KeyboardInterrupt):
    pass
'''


def _stop_process(process):
    if process is None:
        return
    if process.stdin:
        try:
            process.stdin.close()
        except OSError:
            pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    # Children can retain the private group after its leader exits. Always
    # finish cleanup of that group, including a stuck SSH descendant.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=1)
    if process.stdout:
        process.stdout.close()
    if process.stderr:
        process.stderr.close()


class StatusCommand:
    """Bound the existing status CLI and all its SSH subprocesses as one group."""

    def __init__(self, timeout=20.0):
        self.timeout, self.process = timeout, None
        self.lock, self.done = threading.Lock(), threading.Event()

    def __call__(self):
        with self.lock:
            if self.done.is_set():
                raise RuntimeError("monitor is stopping")
            process = subprocess.Popen([sys.executable, str(Path(__file__).with_name("spark-serve")),
                                        "status", "--json"], stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, start_new_session=True)
            self.process = process
        try:
            try:
                output, _errors = process.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError("Allocation status timed out") from exc
            if process.returncode:
                raise OSError(f"Allocation status exited with code {process.returncode}")
            if len(output) > MAX_BYTES:
                raise ValueError("Allocation status exceeds 2 MiB")
            result = json.loads(output)
            if not isinstance(result, dict) or not isinstance(result.get("nodes"), list):
                raise ValueError("Invalid allocation status response")
            return result
        finally:
            _stop_process(process)
            with self.lock:
                self.process = None

    def stop(self):
        self.done.set()
        with self.lock:
            process = self.process
            if process is not None:
                # Do not close pipes while communicate() is reading them. EOF
                # wakes it after every process in this private group is killed.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


class HostCollector:
    def __init__(self, cfg, node, interval):
        self.node, self.host, self.interval = node, cfg["cluster"][node], interval
        opts = list(cfg["cluster"].get("ssh_opts") or ["-o", "BatchMode=yes"])
        self.command = ["ssh", *opts, "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=5",
                        "-o", "ServerAliveCountMax=2", self.host,
                        "python3 -u -c " + shlex.quote(_remote_program(interval))]
        self.done, self.lock = threading.Event(), threading.Lock()
        self.latest, self.received_at, self.sampled_at, self.error = None, None, None, None
        self.connected = False
        self.thread = threading.Thread(target=self._run, name="telemetry-" + node, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.done.set()
        self.thread.join(timeout=3)

    def _run(self):
        backoff = 1.0
        while not self.done.is_set():
            process = None
            try:
                process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                           stderr=subprocess.DEVNULL, start_new_session=True)
                buffer, last_sample = b"", time.monotonic()
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    while not self.done.is_set():
                        if time.monotonic() - last_sample > max(30, self.interval * 5):
                            raise TimeoutError("remote telemetry sample timed out")
                        if not selector.select(timeout=0.25):
                            continue
                        chunk = os.read(process.stdout.fileno(), 65536)
                        if not chunk:
                            raise OSError("SSH telemetry connection closed")
                        buffer += chunk
                        if len(buffer) > MAX_BYTES:
                            raise ValueError("remote telemetry sample exceeds 2 MiB")
                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)
                            sample = json.loads(line)
                            if not isinstance(sample, dict) or sample.get("schema_version") != 1:
                                raise ValueError("unsupported remote telemetry sample")
                            resources = resources_from_sample(sample)
                            with self.lock:
                                self.latest, self.sampled_at = resources, time.time()
                                self.received_at = last_sample = time.monotonic()
                                self.connected = True
                                self.error = ("Some resource counters are unavailable" if sample.get("errors") else None)
                            backoff = 1.0
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                with self.lock:
                    self.error = str(exc)
                    self.connected = False
            finally:
                _stop_process(process)
            if self.done.wait(backoff):
                break
            backoff = min(30, backoff * 2)

    def snapshot(self, now):
        with self.lock:
            state = ("unavailable" if self.latest is None else "stale"
                     if not self.connected or now - self.received_at > max(6, self.interval * 3) else "live")
            return {"node": self.node, "host": self.host, "sampled_at": self.sampled_at,
                    "state": state, "error": self.error,
                    "resources": dict(self.latest) if self.latest else dict.fromkeys(RESOURCE_KEYS)}


class MetricsHTTPError(OSError):
    def __init__(self, status):
        self.status = status
        super().__init__(f"metrics endpoint returned HTTP {status}")


def _fetch_metrics_path(endpoint, path, timeout):
    origin = urlsplit(endpoint)
    if origin.scheme not in ("http", "https") or not origin.hostname or origin.username or origin.password:
        raise ValueError("metrics endpoint must be an HTTP(S) origin without credentials")
    if origin.path not in ("", "/") or origin.query or origin.fragment:
        raise ValueError("metrics endpoint must be an HTTP(S) origin")
    cls = http.client.HTTPSConnection if origin.scheme == "https" else http.client.HTTPConnection
    connection = cls(origin.hostname, origin.port, timeout=timeout)
    transport, expired = [None], threading.Event()

    def abort():
        expired.set()
        sock = transport[0] or connection.sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        connection.close()

    deadline = threading.Timer(timeout, abort)
    deadline.daemon = True
    deadline.start()
    try:
        connection.request("GET", path, headers={"Accept": "text/plain"})
        transport[0] = connection.sock
        if expired.is_set():
            abort()
            raise TimeoutError("metrics request timed out")
        response = connection.getresponse()
        if response.status != 200:
            raise MetricsHTTPError(response.status)
        body = response.read(MAX_BYTES + 1)
        if expired.is_set():
            raise TimeoutError("metrics request timed out")
        if len(body) > MAX_BYTES:
            raise ValueError("metrics response exceeds 2 MiB")
        return body.decode("utf-8")
    finally:
        deadline.cancel()
        connection.close()


def fetch_metrics(endpoint, runtime="vllm"):
    # NIM documents /v1/metrics and unprefixed metric names. A few vLLM-backed
    # images expose /metrics instead; only a 404 permits this fallback.
    paths = ("/v1/metrics", "/metrics") if runtime == "nim" else ("/metrics",)
    deadline = time.monotonic() + 3
    for index, path in enumerate(paths):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("metrics request timed out")
        try:
            return _fetch_metrics_path(endpoint, path, remaining)
        except MetricsHTTPError as exc:
            if exc.status != 404 or index == len(paths) - 1:
                raise


class AllocationCollector:
    def __init__(self, allocation, interval, fetch=fetch_metrics):
        self.allocation, self.interval, self.fetch = allocation, interval, fetch
        self.identity = allocation_identity(allocation)
        self.done, self.lock = threading.Event(), threading.Lock()
        self.metrics = dict.fromkeys(METRIC_KEYS)
        self.sampled_at, self.received_at, self.error = None, None, None
        self.state = "unavailable"
        self.thread = threading.Thread(target=self._run, name="inference-metrics", daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.done.set()
        self.thread.join(timeout=4)

    def _run(self):
        accumulator = InferenceMetrics(max_gap=max(6, self.interval * 3))
        backoff = self.interval
        while not self.done.is_set():
            started = time.monotonic()
            try:
                body = self.fetch(self.allocation["endpoint"], self.allocation["runtime"])
                metrics, supported = accumulator.update(body, time.monotonic(), self.allocation.get("served_name"),
                                                        self.allocation["runtime"])
                with self.lock:
                    self.metrics = metrics
                    self.sampled_at, self.received_at = time.time(), time.monotonic()
                    self.state = "live" if supported else "unsupported"
                    self.error = None if supported else "No supported vLLM metric families exposed"
                backoff = self.interval if supported else min(30, max(10, self.interval))
            except (OSError, ValueError, http.client.HTTPException) as exc:
                accumulator.reset()
                with self.lock:
                    self.state = "stale" if self.sampled_at is not None else "unavailable"
                    self.error = str(exc)
                backoff = min(30, max(self.interval, backoff * 2))
            self.done.wait(max(0, backoff - (time.monotonic() - started)))

    def snapshot(self, now):
        with self.lock:
            state = self.state
            if state == "live" and now - self.received_at > max(6, self.interval * 3):
                state = "stale"
            return {"metrics_state": state, "sampled_at": self.sampled_at,
                    "error": self.error, "metrics": dict(self.metrics)}


class Monitor:
    """Independent host collectors and one scrape worker per logical allocation."""

    def __init__(self, cfg, status_fn=None, interval=2.0, host_factory=HostCollector,
                 allocation_factory=AllocationCollector):
        if not math.isfinite(interval) or interval < 0.25:
            raise ValueError("monitor interval must be finite and at least 0.25 seconds")
        self.interval, self.status_fn = interval, status_fn or StatusCommand()
        self.hosts = [host_factory(cfg, node, interval) for node in ("head", "worker")]
        self.allocation_factory, self.allocations = allocation_factory, {}
        self.done, self.lock = threading.Event(), threading.Lock()
        self.status, self.status_at, self.status_error = {}, None, None
        self.thread = threading.Thread(target=self._status_loop, name="monitor-status", daemon=True)

    def start(self):
        for host in self.hosts:
            host.start()
        self.thread.start()

    def _status_loop(self):
        while not self.done.is_set():
            try:
                status = self.status_fn()
                with self.lock:
                    self.status, self.status_at, self.status_error = status, time.monotonic(), None
            except Exception as exc:
                with self.lock:
                    self.status_error = str(exc)
            self.done.wait(max(5, self.interval * 2))

    def snapshot(self):
        now = time.monotonic()
        with self.lock:
            status = self.status
            status_stale = (self.status_at is None or now - self.status_at > max(30, self.interval * 5)
                            or self.status_error is not None)
            status_error = self.status_error
        current = allocations_from_status(status)
        wanted = {allocation_identity(item): item for item in current
                  if item["ready"] and not status_stale and item["runtime"] in ("vllm", "nim")}
        for identity in list(self.allocations):
            if identity not in wanted:
                self.allocations.pop(identity).stop()
        for identity, allocation in wanted.items():
            if identity not in self.allocations:
                collector = self.allocation_factory(allocation, self.interval)
                self.allocations[identity] = collector
                collector.start()
        for allocation in current:
            collector = self.allocations.get(allocation_identity(allocation))
            if collector:
                allocation.update(collector.snapshot(now))
            else:
                supported = allocation["runtime"] in ("vllm", "nim")
                allocation.update(metrics_state="unavailable" if supported else "unsupported",
                                  sampled_at=None, error=status_error or ("Allocation status is stale" if status_stale
                                                                       else "Allocation is not ready" if supported
                                                                       else "Runtime metrics are not supported"),
                                  metrics=dict.fromkeys(METRIC_KEYS))
            if status_stale:
                allocation["ready"] = False
            allocation.pop("_containers", None)
        return {"event": "snapshot", "schema_version": 1, "timestamp": time.time(),
                "status_state": "unavailable" if self.status_at is None else "stale" if status_stale else "live",
                "status_error": status_error or ("Allocation status has not been read yet" if self.status_at is None
                                                  else "Allocation status is stale" if status_stale else None),
                "hosts": [host.snapshot(now) for host in self.hosts], "allocations": current}

    def stop(self):
        self.done.set()
        # Stop signals go out together so disconnected hosts do not serialize
        # their shutdown delays. Threads only perform bounded read operations.
        collectors = [*self.hosts, *self.allocations.values()]
        for collector in collectors:
            collector.done.set()
        if hasattr(self.status_fn, "stop"):
            self.status_fn.stop()
        for collector in collectors:
            collector.stop()
        self.thread.join(timeout=0.1)


def watch(cfg, status_fn=None, interval=2.0):
    """Write full NDJSON snapshots until a signal or the consumer closes stdout."""
    monitor = Monitor(cfg, status_fn, interval)

    def interrupted(*_args):
        monitor.done.set()
        # A consumer can pause with its stdout pipe still open. Raising also
        # interrupts a full pipe's blocking write, letting finally own cleanup.
        raise KeyboardInterrupt

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        monitor.start()
        while not monitor.done.is_set():
            start = time.monotonic()
            print(json.dumps(monitor.snapshot(), allow_nan=False, separators=(",", ":")), flush=True)
            monitor.done.wait(max(0, interval - (time.monotonic() - start)))
    except (BrokenPipeError, KeyboardInterrupt):
        # Redirect the actual descriptor: replacing only sys.stdout leaves its
        # old buffered stream able to block again when Python finalizes it.
        with open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), sys.stdout.fileno())
    finally:
        monitor.stop()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0
