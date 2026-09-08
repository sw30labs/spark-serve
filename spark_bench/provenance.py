"""Read-only source and host fingerprints for reproducible benchmark records.

This is an allowlisted inventory, never an environment, process-command, or
configuration dump. Source hashes describe files at capture time, not an
assertion that an already-running interpreter loaded those same bytes.
"""

from __future__ import annotations

import csv
import hashlib
import io
import platform
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

SAMPLING_FIELDS = ("utilization.gpu", "utilization.memory", "temperature.gpu",
                   "temperature.gpu.tlimit", "power.draw", "clocks.current.sm",
                   "clocks_event_reasons_counters.sw_power_cap",
                   "clocks_event_reasons_counters.sw_thermal_slowdown",
                   "clocks_event_reasons_counters.hw_thermal_slowdown")


def _driver_version(run):
    query = ["nvidia-smi", "--query-gpu=index,uuid,driver_version", "--format=csv,noheader,nounits"]
    try:
        result = run(query, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": "unavailable", "versions": [], "gpus": [], "error": type(exc).__name__}
    if result.returncode:
        return {"status": "unavailable", "versions": [], "gpus": [], "error": "exit_" + str(result.returncode)}
    gpus, malformed = [], False
    for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True):
        if len(row) != 3:
            malformed = True
            continue
        index, uuid, version = (value.strip() for value in row)
        if not index.isdigit() or not re.fullmatch(r"\d+(?:\.[0-9A-Za-z]+)+", version):
            malformed = True
            continue
        gpus.append({"index": int(index), "uuid": uuid if uuid.startswith("GPU-") else None,
                     "driver_version": version})
    return {"status": "partial" if gpus and malformed else "available" if gpus else "unavailable",
            "versions": sorted({gpu["driver_version"] for gpu in gpus}), "gpus": gpus,
            "error": "invalid_or_unsupported_query_response" if malformed else None if gpus else "no_gpu_rows"}


def _sampling_support(run):
    try:
        result = run(["nvidia-smi", "--help-query-gpu"], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": "unavailable", "supported_query_fields": None, "error": type(exc).__name__}
    if result.returncode:
        return {"status": "unavailable", "supported_query_fields": None, "error": "exit_" + str(result.returncode)}
    supported = set(re.findall(r'"([a-zA-Z0-9_.]+)"', result.stdout))
    return {"status": "available" if supported else "unavailable",
            "supported_query_fields": [name for name in SAMPLING_FIELDS if name in supported] if supported else None,
            "error": None if supported else "no_supported_query_fields",
            "scope": "driver query syntax; supported fields can still return unavailable on this device"}


def collect_provenance(source_root=None, *, run=None, which=None) -> dict:
    """Fingerprint the deployed package/controller and their executing host.

    ``source_root`` is the directory containing spark_bench/ and the controller;
    it defaults to the package's parent directory. Optional injectable command
    runner permits offline tests. Missing GPU support or source files are
    explicit unavailable evidence and never silently reported as a known value.
    """
    root = Path(source_root).resolve() if source_root is not None else Path(__file__).resolve().parent.parent
    run = run or subprocess.run
    which = which or shutil.which
    files = sorted((root / "spark_bench").glob("*.py"))
    errors = []
    if not files:
        errors.append({"source": "spark_bench/*.py", "error": "no_source_files"})
    files.append(root / "spark_serve_controller.py")
    inventory = {}
    for path in files:
        name = path.relative_to(root).as_posix()
        record = {"sha256": None, "size_bytes": None, "status": "unavailable"}
        try:
            if not path.resolve().is_relative_to(root):
                raise ValueError("source symlink escapes inventory root")
            before = path.stat()
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError("source changed during fingerprint")
            record.update(sha256=digest, size_bytes=after.st_size, status="available")
        except (OSError, ValueError) as exc:
            record["error"] = type(exc).__name__
            errors.append({"source": name, "error": type(exc).__name__})
        inventory[name] = record
    return {
        "schema_version": 1, "captured_at": datetime.now(UTC).isoformat(),
        "source_hash_algorithm": "sha256", "source_files": inventory,
        "source_inventory_status": "partial" if errors else "complete",
        "source_capture_scope": "files present at capture time; no environment or configuration inventory",
        "host": {"system": platform.system(), "kernel_release": platform.release(),
                 "kernel_version": platform.version(), "machine": platform.machine(),
                 "python_implementation": platform.python_implementation(),
                 "python_version": platform.python_version()},
        "nvidia_driver": _driver_version(run),
        "nvidia_sampling": _sampling_support(run),
        "profiling_tools": {name: {"status": "present_capability_unverified" if which(name) else "not_found_on_path",
                                    "executed": False} for name in ("dcgmi", "ncu", "nsys")},
        "advanced_counter_scope": "SM activity/occupancy, tensor activity and actual memory bandwidth are not measured by this sampler. Tool presence requires a separate controlled profiling/capability run; no profiler is executed here.",
        "errors": errors,
    }
