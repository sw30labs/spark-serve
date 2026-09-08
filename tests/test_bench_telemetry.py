"""Deterministic telemetry regression tests; never invoke NVIDIA or read /proc."""

import csv
import json
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path

from spark_bench.telemetry import GPU_FIELDS, TelemetryCollector, TelemetrySampler, flat_rows


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class FakeNvidia:
    def __init__(self):
        self.calls = []
        self.values = {
            "index": "0", "uuid": "GPU-test", "name": "NVIDIA GB10",
            "utilization.gpu": "99", "utilization.memory": "65",
            "memory.total": "N/A", "memory.used": "N/A", "memory.free": "N/A",
            "temperature.gpu": "68", "power.draw": "115.5", "power.limit": "140",
            "clocks.current.sm": "2500", "clocks.current.memory": "4267",
            "clocks.current.graphics": "2500", "clocks.max.sm": "3000",
            "clocks_throttle_reasons.active": "0x0000000000000004",
        }

    def __call__(self, args, **kwargs):
        self.calls.append(args)
        assert kwargs["timeout"] == 5
        if args[1] == "--help-query-gpu":
            output = "\n".join('"' + key + '"' for key in self.values)
        elif args[1].startswith("--query-gpu="):
            fields = args[1].split("=", 1)[1].split(",")
            output = ", ".join(self.values[field] for field in fields) + "\n"
        else:
            output = "GPU-test, 123, /usr/bin/python3, N/A\n"
        return subprocess.CompletedProcess(args, 0, output, "")


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.proc = self.root / "proc"
        self.sys = self.root / "sys"
        self.clock = FakeClock()
        self.nvidia = FakeNvidia()
        self.sampler = TelemetrySampler(self.proc, self.sys, run=self.nvidia,
                                        monotonic=self.clock, wall_clock=lambda: 1700000000)
        self.write("proc/stat", "cpu 100 0 100 800 0 0 0 0 50 0\ncpu0 50 0 50 400 0 0 0 0 25 0\ncpu1 50 0 50 400 0 0 0 0 25 0\n")
        self.write("proc/loadavg", "1.2 0.8 0.5 2/200 123\n")
        self.write("proc/meminfo", "MemTotal: 131072000 kB\nMemAvailable: 90000000 kB\nMemFree: 10000000 kB\nSwapTotal: 1000 kB\nSwapFree: 750 kB\n")
        self.write("proc/net/dev", "Inter-| Receive | Transmit\n eth0: 1000 2 0 0 0 0 0 0 2000 2 0 0 0 0 0 0\n lo: 10000 0 0 0 0 0 0 0 10000 0 0 0 0 0 0 0\n")
        self.write("proc/diskstats", "259 0 nvme0n1 10 0 100 0 10 0 200 0 0 100 0\n259 1 nvme0n1p1 10 0 100 0 10 0 200 0 0 100 0\n7 0 loop0 10 0 100 0 10 0 200 0 0 100 0\n")
        self.write("sys/class/block/nvme0n1p1/partition", "1\n")
        self.write("sys/class/thermal/thermal_zone0/temp", "64000\n")
        self.write("sys/class/thermal/thermal_zone0/type", "cpu-thermal\n")
        self.write("sys/class/hwmon/hwmon0/temp1_input", "65000\n")
        self.write("sys/class/hwmon/hwmon0/temp1_label", "board\n")
        self.write("proc/pressure/memory", "some avg10=0.05 avg60=0.01 avg300=0.00 total=123\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n")
        self.process(ticks=100)

    def write(self, relative, content):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def process(self, ticks, start=20):
        fields = ["0"] * 22
        fields[0], fields[11], fields[12] = "S", str(ticks), "0"
        fields[19], fields[20], fields[21] = str(start), "100000", "10"
        self.write("proc/123/stat", "123 (python (worker)) " + " ".join(fields))
        self.write("proc/123/status", "Name:\tpython\nVmSwap:\t2 kB\n")

    def test_initial_sample_and_unified_memory_availability(self):
        result = self.sampler.sample()
        self.assertIsNone(result["cpu"]["total_percent"])
        self.assertEqual(result["memory"]["available_bytes"], 90000000 * 1024)
        self.assertEqual(result["memory"]["used_bytes"], (131072000 - 90000000) * 1024)
        self.assertEqual(result["memory"]["swap_used_bytes"], 250 * 1024)
        gpu = result["gpus"][0]
        self.assertEqual(gpu["utilization_gpu_percent"], 99)
        self.assertEqual(gpu["utilization_memory_percent"], 65)
        self.assertIsNone(gpu["memory_used_bytes"])
        self.assertEqual(gpu["availability"]["memory_used_bytes"], "unavailable_on_device")
        self.assertIsNone(gpu["sm_occupancy_percent"])
        self.assertIsNone(gpu["memory_bandwidth_bytes_per_second"])
        self.assertFalse(result["availability"]["memory_bandwidth"])
        self.assertEqual(gpu["clock_event_reasons_active"], "0x0000000000000004")
        self.assertEqual(gpu["availability"]["clock_event_hw_thermal"], "unsupported")
        self.assertEqual(result["load_average"], [1.2, 0.8, 0.5])
        self.assertEqual(result["thermals"][0]["temperature_c"], 64)
        self.assertEqual(result["thermals"][1]["temperature_c"], 65)
        self.assertEqual(result["pressure"]["memory"]["some"]["avg10"], 0.05)
        self.assertEqual(result["processes"][0]["name"], "python (worker)")
        self.assertEqual(result["processes"][0]["swap_bytes"], 2048)
        self.assertEqual(result["errors"], [])
        json.dumps(result, allow_nan=False)

    def test_rates_are_interval_deltas_and_guest_is_not_double_counted(self):
        self.sampler.sample()
        self.clock.now += 2
        self.write("proc/stat", "cpu 150 0 150 900 0 0 0 0 100 0\ncpu0 75 0 75 450 0 0 0 0 50 0\ncpu1 75 0 75 450 0 0 0 0 50 0\n")
        self.write("proc/net/dev", " eth0: 3000 2 0 0 0 0 0 0 6000 2 0 0 0 0 0 0\n lo: 90000 0 0 0 0 0 0 0 90000 0 0 0 0 0 0 0\n")
        self.write("proc/diskstats", "259 0 nvme0n1 10 0 300 0 10 0 600 0 0 1100 0\n")
        self.process(ticks=100 + self.sampler._tick_hz)
        result = self.sampler.sample()
        self.assertAlmostEqual(result["cpu"]["total_percent"], 50)
        self.assertAlmostEqual(result["cpu"]["per_core_percent"]["cpu0"], 50)
        self.assertEqual(result["network"]["total_rx_bytes_per_second"], 1000)
        self.assertEqual(result["network"]["total_tx_bytes_per_second"], 2000)
        self.assertEqual(result["disk"]["total_read_bytes_per_second"], 100 * 512)
        self.assertEqual(result["disk"]["total_write_bytes_per_second"], 200 * 512)
        self.assertEqual(result["disk"]["devices"]["nvme0n1"]["busy_percent"], 50)
        self.assertEqual(result["processes"][0]["cpu_percent"], 50)
        self.assertEqual(sum(call[1] == "--help-query-gpu" for call in self.nvidia.calls), 1)

    def test_counter_resets_and_reused_pid_do_not_create_false_rates(self):
        self.sampler.sample()
        self.clock.now += 1
        self.process(ticks=200, start=999)
        self.write("proc/net/dev", " eth0: 10 2 0 0 0 0 0 0 10 2 0 0 0 0 0 0\n")
        self.write("proc/diskstats", "259 0 nvme0n1 1 0 1 0 1 0 1 0 0 1 0\n")
        result = self.sampler.sample()
        self.assertIsNone(result["network"]["total_rx_bytes_per_second"])
        self.assertIsNone(result["disk"]["total_read_bytes_per_second"])
        self.assertIsNone(result["processes"][0]["cpu_percent"])

    def test_partition_and_loop_counters_excluded(self):
        result = self.sampler.sample()
        self.assertEqual(list(result["disk"]["devices"]), ["nvme0n1"])

    def test_missing_linux_and_nvidia_are_explicit_not_zero(self):
        def missing(*_args, **_kwargs):
            raise FileNotFoundError("nvidia-smi absent")
        sampler = TelemetrySampler(self.root / "missing", self.root / "missing", run=missing)
        result = sampler.sample()
        self.assertFalse(result["availability"]["gpu"])
        self.assertFalse(result["availability"]["memory"])
        self.assertFalse(result["availability"]["cpu"])
        self.assertEqual(result["gpus"], [])
        self.assertIsNone(result["memory"]["available_bytes"])
        self.assertIsNone(result["cpu"]["total_percent"])
        self.assertTrue(result["errors"])
        self.assertEqual(len(list(flat_rows(result))), 1)

    def test_subprocess_failure_does_not_record_stderr(self):
        def fail(args, **_kwargs):
            return subprocess.CompletedProcess(args, 1, "", "token=SHOULD_NOT_APPEAR")
        sampler = TelemetrySampler(self.proc, self.sys, run=fail)
        result = sampler.sample()
        self.assertNotIn("SHOULD_NOT_APPEAR", json.dumps(result))
        self.assertEqual(result["errors"][0]["error"], "exit_1")

    def test_malformed_gpu_rows_are_reported(self):
        original = self.nvidia
        def malformed(args, **kwargs):
            if args[1].startswith("--query-gpu="):
                return subprocess.CompletedProcess(args, 0, "0, corrupt", "")
            return original(args, **kwargs)
        self.sampler.run = malformed
        result = self.sampler.sample()
        self.assertEqual(result["gpus"], [])
        self.assertTrue(any(error["error"] == "column_count_mismatch" for error in result["errors"]))

    def test_csv_represents_gpu_and_node_memory_separately(self):
        sample = self.sampler.sample()
        rows = list(flat_rows(sample))
        self.assertIsNone(rows[0]["memory_used_bytes"])
        self.assertEqual(rows[0]["node_memory_used_bytes"], sample["memory"]["used_bytes"])
        self.assertEqual(rows[0]["temperature_gpu_c"], 68)
        self.assertIsNone(rows[0]["memory_bandwidth_bytes_per_second"])

    def test_collector_preserves_raw_samples_and_appends_single_csv_header(self):
        output = self.root / "output"
        for _index in range(2):
            collector = TelemetryCollector(output, interval_s=0.01, sampler=self.sampler).start()
            deadline = time.monotonic() + 2
            while collector.samples_written == 0 and time.monotonic() < deadline:
                threading.Event().wait(0.005)
            collector.stop()
            self.assertGreater(collector.samples_written, 0)
        raw_rows = [json.loads(line) for line in (output / "telemetry.jsonl").read_text().splitlines()]
        with (output / "telemetry.csv").open(newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        self.assertGreaterEqual(len(raw_rows), 2)
        self.assertEqual(len(raw_rows), len(csv_rows))
        self.assertTrue(all(row["hostname"] != "hostname" for row in csv_rows))
        self.assertEqual(csv_rows[0]["memory_used_bytes"], "")
        self.assertEqual(raw_rows[0]["gpus"][0]["availability"]["memory_used_bytes"], "unavailable_on_device")

    def test_collector_surfaces_failure_and_validates_interval(self):
        for invalid in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                TelemetryCollector(self.root, interval_s=invalid)
        class Broken:
            def sample(self):
                raise OSError("disk full")
        collector = TelemetryCollector(self.root / "broken", sampler=Broken()).start()
        collector._thread.join(timeout=2)
        with self.assertRaisesRegex(RuntimeError, "collection failed"):
            collector.raise_if_failed()
        with self.assertRaisesRegex(RuntimeError, "collection failed"):
            collector.stop()


if __name__ == "__main__":
    unittest.main()
