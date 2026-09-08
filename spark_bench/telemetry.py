"""Read-only, dependency-free Linux/NVIDIA benchmark telemetry.

GPU ``utilization_memory_percent`` is the fraction of a sampling period when
device memory was busy, NOT measured bandwidth or SM occupancy. Unsupported
counters stay null. On unified-memory Sparks, /proc/meminfo is the authoritative
node memory view; nvidia-smi memory.used is often unavailable.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import signal
import socket
import subprocess
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

# nvidia-smi emits MiB, MHz, W, C with nounits. Aliases avoid dependence on
# whether a driver names clock-event counters "throttle" or "event" reasons.
GPU_FIELDS = {
    "index": (("index",), 1),
    "uuid": (("uuid",), None),
    "name": (("name",), None),
    "utilization_gpu_percent": (("utilization.gpu",), 1),
    "utilization_memory_percent": (("utilization.memory",), 1),
    "memory_total_bytes": (("memory.total",), 1024**2),
    "memory_used_bytes": (("memory.used",), 1024**2),
    "memory_free_bytes": (("memory.free",), 1024**2),
    "temperature_gpu_c": (("temperature.gpu",), 1),
    "power_draw_watts": (("power.draw",), 1),
    "power_limit_watts": (("power.limit",), 1),
    "clock_sm_mhz": (("clocks.current.sm", "clocks.sm"), 1),
    "clock_memory_mhz": (("clocks.current.memory", "clocks.mem"), 1),
    "clock_graphics_mhz": (("clocks.current.graphics", "clocks.gr"), 1),
    "clock_max_sm_mhz": (("clocks.max.sm",), 1),
    "clock_event_reasons_active": (("clocks_event_reasons.active", "clocks_throttle_reasons.active"), None),
    "clock_event_sw_power_cap": (("clocks_event_reasons.sw_power_cap", "clocks_throttle_reasons.sw_power_cap"), None),
    "clock_event_hw_thermal": (("clocks_event_reasons.hw_thermal_slowdown", "clocks_throttle_reasons.hw_thermal_slowdown"), None),
    "clock_event_sw_thermal": (("clocks_event_reasons.sw_thermal_slowdown", "clocks_throttle_reasons.sw_thermal_slowdown"), None),
    "clock_event_hw_slowdown": (("clocks_event_reasons.hw_slowdown", "clocks_throttle_reasons.hw_slowdown"), None),
}
ADVANCED_COUNTERS = ("sm_activity_percent", "sm_occupancy_percent", "tensor_activity_percent", "memory_bandwidth_bytes_per_second")


def _number(value: str, scale: float = 1) -> float | None:
    try:
        result = float(value.strip()) * scale
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _delta_rate(current: int, previous: int, elapsed: float) -> float | None:
    return (current - previous) / elapsed if elapsed > 0 and current >= previous else None


class TelemetrySampler:
    """Stateful sampler; first sample establishes baselines for rate counters.

    ``pids`` adds processes to the GPU process list. No command lines or
    environment variables are inspected. Inject roots/run/clocks for offline
    deterministic tests. One sampler should be called by one thread.
    """

    def __init__(self, proc_root="/proc", sys_root="/sys", run=None, pids=None,
                 monotonic: Callable[[], float] = time.monotonic,
                 wall_clock: Callable[[], float] = time.time):
        self.proc_root = Path(proc_root)
        self.sys_root = Path(sys_root)
        self.run = run or subprocess.run
        self.pids = set(int(pid) for pid in (pids or []))
        self.monotonic = monotonic
        self.wall_clock = wall_clock
        self._previous = {}
        self._gpu_fields = None
        self._gpu_error = None
        self._tick_hz = os.sysconf("SC_CLK_TCK")

    def _read(self, path, errors, optional=False):
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError) as exc:
            if not optional:
                errors.append({"source": str(path), "error": type(exc).__name__})
            return None

    def _command(self, args):
        # Never emit subprocess stderr or command lines into persisted logs.
        try:
            result = self.run(args, capture_output=True, text=True, timeout=5, check=False)
            if result.returncode != 0:
                return None, "exit_" + str(result.returncode)
            return result.stdout, None
        except (OSError, subprocess.SubprocessError) as exc:
            return None, type(exc).__name__

    def _discover_gpu(self):
        help_text, error = self._command(["nvidia-smi", "--help-query-gpu"])
        if help_text is None:
            self._gpu_fields = {}
            self._gpu_error = error
            return
        supported = set(re.findall(r'"([a-zA-Z0-9_.]+)"', help_text))
        # Field names appear quoted in NVIDIA's documented query help output.
        self._gpu_fields = {
            key: next((field for field in aliases if field in supported), None)
            for key, (aliases, _scale) in GPU_FIELDS.items()
        }
        self._gpu_fields = {key: field for key, field in self._gpu_fields.items() if field}
        if not self._gpu_fields:
            self._gpu_error = "no_supported_query_fields"

    def _gpu(self, errors):
        if self._gpu_fields is None:
            self._discover_gpu()
        if not self._gpu_fields:
            errors.append({"source": "nvidia-smi", "error": self._gpu_error})
            return [], [], False
        output, error = self._command([
            "nvidia-smi", "--query-gpu=" + ",".join(self._gpu_fields.values()),
            "--format=csv,noheader,nounits",
        ])
        gpus = []
        if error:
            errors.append({"source": "nvidia-smi/gpu", "error": error})
        else:
            for row in csv.reader(io.StringIO(output or ""), skipinitialspace=True):
                if len(row) != len(self._gpu_fields):
                    errors.append({"source": "nvidia-smi/gpu", "error": "column_count_mismatch"})
                    continue
                gpu = {key: None for key in (*GPU_FIELDS, *ADVANCED_COUNTERS)}
                availability = {key: "unsupported" for key in GPU_FIELDS}
                availability.update({key: "requires_profiler_or_dcgm" for key in ADVANCED_COUNTERS})
                for key, raw in zip(self._gpu_fields, row, strict=True):
                    raw = raw.strip()
                    scale = GPU_FIELDS[key][1]
                    value = (None if raw in ("N/A", "[N/A]", "[Not Supported]", "Not Supported", "") else raw) if scale is None else _number(raw, scale)
                    gpu[key] = value
                    availability[key] = "available" if value is not None else "unavailable_on_device"
                gpu["availability"] = availability
                gpus.append(gpu)
        app_output, app_error = self._command([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ])
        apps = []
        if app_error:
            errors.append({"source": "nvidia-smi/processes", "error": app_error})
        else:
            for row in csv.reader(io.StringIO(app_output or ""), skipinitialspace=True):
                if len(row) != 4:
                    continue
                try:
                    pid = int(row[1].strip())
                except ValueError:
                    continue
                apps.append({"gpu_uuid": row[0].strip(), "pid": pid,
                             "name": Path(row[2].strip()).name,
                             "gpu_memory_bytes": _number(row[3], 1024**2)})
        return gpus, apps, error is None and bool(gpus)

    def _cpu(self, errors):
        raw = self._read(self.proc_root / "stat", errors)
        now = {}
        percents = {}
        if raw:
            for line in raw.splitlines():
                parts = line.split()
                if not parts or not re.fullmatch(r"cpu\d*", parts[0]):
                    continue
                try:
                    # guest and guest_nice are already included in user/nice.
                    ticks = [int(part) for part in parts[1:9]]
                    total = sum(ticks)
                    idle = ticks[3] + (ticks[4] if len(ticks) > 4 else 0)
                except (ValueError, IndexError):
                    continue
                now[parts[0]] = (total, idle)
                old = self._previous.get("cpu", {}).get(parts[0])
                if old and total > old[0] and idle >= old[1]:
                    percents[parts[0]] = max(0.0, min(100.0, 100 * (1 - (idle - old[1]) / (total - old[0]))))
                else:
                    percents[parts[0]] = None
        self._previous["cpu"] = now
        load = self._read(self.proc_root / "loadavg", errors)
        try:
            load_values = [float(value) for value in load.split()[:3]] if load else None
        except ValueError:
            load_values = None
        pressure = {}
        for resource in ("cpu", "memory", "io"):
            pressure_raw = self._read(self.proc_root / "pressure" / resource, errors, optional=True)
            if pressure_raw:
                resource_values = {}
                for line in pressure_raw.splitlines():
                    fields = line.split()
                    resource_values[fields[0]] = {
                        key: _number(value) for key, value in
                        (part.split("=", 1) for part in fields[1:] if "=" in part)
                    }
                pressure[resource] = resource_values
        return {"total_percent": percents.get("cpu"),
                "per_core_percent": {key: val for key, val in percents.items() if key != "cpu"},
                "logical_cpu_count": max(0, len(now) - 1)}, load_values, pressure, bool(now)

    def _memory(self, errors):
        raw = self._read(self.proc_root / "meminfo", errors)
        values = {}
        if raw:
            for line in raw.splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    number = _number(parts[1], 1024 if len(parts) > 2 and parts[2] == "kB" else 1)
                    if number is not None:
                        values[parts[0].rstrip(":")] = int(number)
        total = values.get("MemTotal")
        available = values.get("MemAvailable")
        swap_total, swap_free = values.get("SwapTotal"), values.get("SwapFree")
        return {"total_bytes": total, "available_bytes": available,
                "free_bytes": values.get("MemFree"),
                "used_bytes": total - available if total is not None and available is not None else None,
                "swap_total_bytes": swap_total, "swap_free_bytes": swap_free,
                "swap_used_bytes": swap_total - swap_free if swap_total is not None and swap_free is not None else None,
                "source": "/proc/meminfo", "used_definition": "MemTotal - MemAvailable"}

    def _network(self, elapsed, errors):
        raw = self._read(self.proc_root / "net/dev", errors)
        now, result = {}, {}
        for line in (raw or "").splitlines():
            if ":" not in line:
                continue
            name, counters = line.rsplit(":", 1)
            fields = counters.split()
            try:
                values = {"rx_bytes": int(fields[0]), "tx_bytes": int(fields[8]),
                          "rx_errors": int(fields[2]), "tx_errors": int(fields[10]),
                          "rx_dropped": int(fields[3]), "tx_dropped": int(fields[11])}
            except (ValueError, IndexError):
                continue
            name = name.strip()
            now[name] = values
            old = self._previous.get("network", {}).get(name)
            result[name] = dict(values)
            for key in ("rx_bytes", "tx_bytes"):
                result[name][key + "_per_second"] = _delta_rate(values[key], old[key], elapsed) if old else None
        self._previous["network"] = now
        totals = {}
        for direction in ("rx", "tx"):
            rates = [row[direction + "_bytes_per_second"] for name, row in result.items() if name != "lo"]
            totals["total_" + direction + "_bytes_per_second"] = sum(rates) if rates and all(value is not None for value in rates) else None
        return {"interfaces": result, "totals_exclude": ["lo"], **totals}

    def _disk(self, elapsed, errors):
        raw = self._read(self.proc_root / "diskstats", errors)
        now, result = {}, {}
        for line in (raw or "").splitlines():
            fields = line.split()
            if len(fields) < 14:
                continue
            name = fields[2]
            # Exclude partitions to avoid double-counting whole devices, and
            # loop/RAM devices which don't represent physical disk traffic.
            if name.startswith(("loop", "ram")) or (self.sys_root / "class/block" / name / "partition").exists():
                continue
            try:
                values = {"read_bytes": int(fields[5]) * 512, "write_bytes": int(fields[9]) * 512,
                          "reads_completed": int(fields[3]), "writes_completed": int(fields[7]),
                          "io_time_ms": int(fields[12]), "io_in_flight": int(fields[11])}
            except ValueError:
                continue
            now[name] = values
            old = self._previous.get("disk", {}).get(name)
            result[name] = dict(values)
            for key in ("read_bytes", "write_bytes"):
                result[name][key + "_per_second"] = _delta_rate(values[key], old[key], elapsed) if old else None
            rate = _delta_rate(values["io_time_ms"], old["io_time_ms"], elapsed) if old else None
            result[name]["busy_percent"] = min(100.0, rate / 10) if rate is not None else None
        self._previous["disk"] = now
        totals = {}
        for direction in ("read", "write"):
            rates = [row[direction + "_bytes_per_second"] for row in result.values()]
            totals["total_" + direction + "_bytes_per_second"] = sum(rates) if rates and all(value is not None for value in rates) else None
        return {"devices": result, "totals_note": "whole block devices; stacked virtual devices can double count", **totals}

    def _processes(self, apps, elapsed, errors):
        by_pid = {}
        for app in apps:
            entry = by_pid.setdefault(app["pid"], {"pid": app["pid"], "gpu_allocations": []})
            entry["gpu_allocations"].append(app)
        for pid in self.pids:
            by_pid.setdefault(pid, {"pid": pid, "gpu_allocations": []})
        now, result = {}, []
        for pid, entry in sorted(by_pid.items()):
            raw = self._read(self.proc_root / str(pid) / "stat", errors, optional=True)
            entry.update(cpu_percent=None, resident_bytes=None, virtual_bytes=None, swap_bytes=None, proc_available=False)
            if raw:
                try:
                    close = raw.rindex(")")
                    fields = raw[close + 2:].split()
                    ticks, start = int(fields[11]) + int(fields[12]), int(fields[19])
                    now[pid] = (ticks, start)
                    old = self._previous.get("processes", {}).get(pid)
                    if old and old[1] == start:
                        rate = _delta_rate(ticks, old[0], elapsed)
                        entry["cpu_percent"] = 100 * rate / self._tick_hz if rate is not None else None
                    entry["name"] = raw[raw.index("(") + 1:close]
                    entry["resident_bytes"] = int(fields[21]) * os.sysconf("SC_PAGE_SIZE")
                    entry["virtual_bytes"] = int(fields[20])
                    entry["proc_available"] = True
                except (ValueError, IndexError):
                    pass
            status = self._read(self.proc_root / str(pid) / "status", errors, optional=True)
            for line in (status or "").splitlines():
                if line.startswith("VmSwap:"):
                    parts = line.split()
                    entry["swap_bytes"] = _number(parts[1], 1024) if len(parts) > 1 else None
            result.append(entry)
        self._previous["processes"] = now
        return result

    def _thermals(self, errors):
        temperatures = []
        for path in sorted((self.sys_root / "class/thermal").glob("thermal_zone*")):
            raw = self._read(path / "temp", errors, optional=True)
            value = _number(raw, 0.001) if raw else None
            kind = self._read(path / "type", errors, optional=True)
            if value is not None:
                temperatures.append({"sensor": path.name, "type": kind.strip() if kind else None, "temperature_c": value})
        for path in sorted((self.sys_root / "class/hwmon").glob("hwmon*/temp*_input")):
            raw = self._read(path, errors, optional=True)
            value = _number(raw, 0.001) if raw else None
            label = self._read(path.with_name(path.name.replace("_input", "_label")), errors, optional=True)
            if value is not None:
                temperatures.append({"sensor": path.parent.name + "/" + path.name,
                                     "type": label.strip() if label else None, "temperature_c": value})
        return temperatures

    def sample(self) -> dict:
        start = self.monotonic()
        wall_start = self.wall_clock()
        previous_time = self._previous.get("time")
        elapsed = start - previous_time if previous_time is not None else 0
        errors = []
        gpus, apps, gpu_ok = self._gpu(errors)
        cpu, load, pressure, cpu_ok = self._cpu(errors)
        memory = self._memory(errors)
        network = self._network(elapsed, errors)
        disk = self._disk(elapsed, errors)
        processes = self._processes(apps, elapsed, errors)
        thermals = self._thermals(errors)
        self._previous["time"] = start
        return {
            "schema_version": 1,
            "timestamp_wall": datetime.fromtimestamp(wall_start, UTC).isoformat(),
            "monotonic_s": start, "interval_s": elapsed, "hostname": socket.gethostname(),
            "collection_duration_s": self.monotonic() - start,
            "gpus": gpus, "cpu": cpu, "load_average": load, "memory": memory,
            "pressure": pressure, "network": network, "disk": disk,
            "processes": processes, "thermals": thermals,
            "availability": {"gpu": gpu_ok, "cpu": cpu_ok, "memory": memory["total_bytes"] is not None,
                             "network": bool(network["interfaces"]), "disk": bool(disk["devices"]),
                             "pressure": bool(pressure), "thermal_sensors": bool(thermals),
                             "sm_activity": False, "tensor_activity": False, "memory_bandwidth": False},
            "errors": errors,
        }


CSV_FIELDS = (
    "timestamp_wall", "monotonic_s", "hostname", "interval_s", "collection_duration_s",
    "gpu_index", "gpu_uuid", "utilization_gpu_percent", "utilization_memory_percent",
    "memory_used_bytes", "memory_total_bytes", "temperature_gpu_c", "power_draw_watts", "power_limit_watts",
    "clock_sm_mhz", "clock_memory_mhz", "clock_event_reasons_active",
    "sm_activity_percent", "sm_occupancy_percent", "tensor_activity_percent", "memory_bandwidth_bytes_per_second",
    "cpu_total_percent", "node_memory_total_bytes", "node_memory_available_bytes", "node_memory_used_bytes",
    "node_swap_used_bytes", "network_rx_bytes_per_second", "network_tx_bytes_per_second",
    "disk_read_bytes_per_second", "disk_write_bytes_per_second", "errors_json",
)


def flat_rows(sample):
    """One CSV row per GPU, or one blank-GPU row when NVIDIA is unavailable."""
    for gpu in sample.get("gpus") or [{}]:
        row = {key: sample.get(key) for key in CSV_FIELDS[:5]}
        row.update({key: gpu.get(key) for key in CSV_FIELDS[7:21]})
        row.update(gpu_index=gpu.get("index"), gpu_uuid=gpu.get("uuid"),
                   cpu_total_percent=sample.get("cpu", {}).get("total_percent"),
                   errors_json=json.dumps(sample.get("errors", []), separators=(",", ":")))
        for key in ("total", "available", "used"):
            row["node_memory_" + key + "_bytes"] = sample.get("memory", {}).get(key + "_bytes")
        row["node_swap_used_bytes"] = sample.get("memory", {}).get("swap_used_bytes")
        for direction in ("rx", "tx"):
            row["network_" + direction + "_bytes_per_second"] = sample.get("network", {}).get("total_" + direction + "_bytes_per_second")
        for direction in ("read", "write"):
            row["disk_" + direction + "_bytes_per_second"] = sample.get("disk", {}).get("total_" + direction + "_bytes_per_second")
        yield row


class TelemetryCollector:
    """Background append-only collection; JSONL is authoritative, CSV is flat.

    Collection is read-only and bounded by subprocess timeouts. ``stop`` raises
    if the collector failed, so a missing resource trace cannot look successful.
    """

    def __init__(self, output_dir, interval_s=2.0, sampler=None):
        if not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError("telemetry interval must be positive and finite")
        self.output_dir = Path(output_dir)
        self.interval_s = interval_s
        self.sampler = sampler or TelemetrySampler()
        self.latest = None
        self.samples_written = 0
        self._stop = threading.Event()
        self._thread = None
        self._error = None

    def _collect(self):
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = self.output_dir / "telemetry.csv"
            needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
            with (self.output_dir / "telemetry.jsonl").open("a", encoding="utf-8") as raw, csv_path.open("a", newline="", encoding="utf-8") as flat:
                writer = csv.DictWriter(flat, fieldnames=CSV_FIELDS)
                if needs_header:
                    writer.writeheader()
                while not self._stop.is_set():
                    start = time.monotonic()
                    sample = self.sampler.sample()
                    raw.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
                    writer.writerows(flat_rows(sample))
                    raw.flush()
                    flat.flush()
                    self.latest = sample
                    self.samples_written += 1
                    self._stop.wait(max(0, self.interval_s - (time.monotonic() - start)))
        except Exception as exc:
            self._error = exc

    def start(self):
        if self._thread is not None:
            raise RuntimeError("telemetry collector already started")
        self._thread = threading.Thread(target=self._collect, name="spark-bench-telemetry", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=20)
            if self._thread.is_alive():
                raise RuntimeError("telemetry collector did not stop within timeout")
        self.raise_if_failed()

    def raise_if_failed(self):
        """Nonblocking check for a controller to stop when raw capture fails."""
        if self._error is not None:
            raise RuntimeError("telemetry collection failed") from self._error

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--duration", type=float, help="seconds; otherwise collect until SIGINT/SIGTERM")
    parser.add_argument("--pid", type=int, action="append", default=[])
    args = parser.parse_args(argv)
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("duration must be positive and finite")
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("interval must be positive and finite")
    done = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_args: done.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with TelemetryCollector(args.output, args.interval, TelemetrySampler(pids=args.pid)):
            done.wait(args.duration)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
