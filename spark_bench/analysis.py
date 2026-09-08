"""Offline analysis of isolated YuE concurrency trials; standard library only."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean


def _number(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _stats(values):
    values = sorted(value for value in values if _number(value) is not None)
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}

    def percentile(fraction):
        index = (len(values) - 1) * fraction
        low = int(index)
        high = min(low + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (index - low)

    return {"count": len(values), "mean": mean(values), "p50": percentile(.5),
            "p95": percentile(.95), "min": values[0], "max": values[-1]}


def _json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected an object")
    return value


def _jsonl(path, issues):
    if not path.is_file():
        issues.append(f"missing {path.name}")
        return []
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError("expected an object")
            rows.append(value)
        except (ValueError, TypeError):
            issues.append(f"{path.name}:{index}: incomplete or invalid JSON record")
    return rows


def _failure(job):
    status = job.get("status")
    error = str(job.get("error") or "").lower()
    if job.get("cuda_oom") or job.get("oom_killed"):
        return "out_of_memory"
    if status == "failed_integrity" or "failed_integrity" in error:
        return "failed_integrity"
    if status == "succeeded":
        return None
    if status in {"timeout", "cancelled"}:
        return status
    if "out of memory" in error or "cuda oom" in error or "oomkill" in error:
        return "out_of_memory"
    if status == "failed":
        return "infrastructure_or_unknown"
    return "censored"


def _overlap(jobs, trial_end, start_key="started_at", end_key="finished_at"):
    events = []
    censored = 0
    for job in jobs:
        start, end = _number(job.get(start_key)), _number(job.get(end_key))
        if start is None:
            continue
        if end is None:
            end = trial_end
            censored += 1
        if end is not None and end > start:
            events.extend(((start, 1), (end, -1)))
    active = maximum = 0
    # Finishes precede starts at equal timestamps: half-open [start, finish).
    for _, delta in sorted(events):
        active += delta
        maximum = max(maximum, active)
    return maximum, censored


def _timestamp(record):
    for key in ("timestamp_wall", "timestamp", "timestamp_s", "time"):
        value = record.get(key)
        if _number(value) is not None:
            return float(value)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
            except ValueError:
                pass
    return None


_TELEMETRY_FIELDS = {
    "gpu_temperature_c": ("gpus", "temperature_gpu_c"),
    "gpu_temperature_tlimit_c": ("gpus", "temperature_tlimit_c"),
    "gpu_power_watts": ("gpus", "power_draw_watts"),
    "gpu_power_limit_watts": ("gpus", "power_limit_watts"),
    "gpu_utilization_percent": ("gpus", "utilization_gpu_percent"),
    "gpu_memory_utilization_percent": ("gpus", "utilization_memory_percent"),
    "gpu_memory_used_bytes": ("gpus", "memory_used_bytes"),
    "gpu_memory_total_bytes": ("gpus", "memory_total_bytes"),
    "gpu_sm_clock_mhz": ("gpus", "clock_sm_mhz"),
    "gpu_memory_clock_mhz": ("gpus", "clock_memory_mhz"),
    "cpu_percent": ("cpu", "total_percent"),
    "memory_available_bytes": ("memory", "available_bytes"),
    "memory_used_bytes": ("memory", "used_bytes"),
    "swap_used_bytes": ("memory", "swap_used_bytes"),
    "network_rx_bytes_per_second": ("network", "total_rx_bytes_per_second"),
    "network_tx_bytes_per_second": ("network", "total_tx_bytes_per_second"),
    "disk_read_bytes_per_second": ("disk", "total_read_bytes_per_second"),
    "disk_write_bytes_per_second": ("disk", "total_write_bytes_per_second"),
}

_REQUIRED_TELEMETRY = ("memory_available_bytes", "gpu_temperature_c", "cpu_percent", "gpu_utilization_percent")

_THROTTLE_COUNTERS = (
    "clock_event_sw_power_cap_us", "clock_event_sw_thermal_us", "clock_event_hw_thermal_us",
    "clock_event_hw_power_brake_us", "clock_event_sync_boost_us",
)


def _throttle_deltas(rows):
    """Count only adjacent observed increments, never driver-lifetime totals."""
    counters = {field: {"observed_delta_us": None, "observed_interval_s": 0.0,
        "valid_intervals": 0, "reset_intervals": 0, "unavailable_intervals": 0,
        "time_discontinuities": 0} for field in _THROTTLE_COUNTERS}
    previous = {}
    for row in rows:
        stamp = _number(row.get("monotonic_s"))
        basis = "monotonic" if stamp is not None else "epoch"
        stamp = stamp if stamp is not None else _timestamp(row)
        current = {}
        for gpu in row.get("gpus") or []:
            identity = gpu.get("uuid") if isinstance(gpu, dict) else None
            if not isinstance(identity, str) or not identity:
                for counter in counters.values():
                    counter["unavailable_intervals"] += 1
                continue
            key = (str(row.get("hostname") or ""), identity)
            values = {field: _number(gpu.get(field)) for field in _THROTTLE_COUNTERS}
            current[key] = (basis, stamp, values)
            old = previous.get(key)
            if old is None:
                continue
            old_basis, old_stamp, old_values = old
            for field, counter in counters.items():
                value, old_value = values[field], old_values[field]
                if value is None or old_value is None or value < 0 or old_value < 0:
                    counter["unavailable_intervals"] += 1
                elif stamp is None or old_stamp is None or basis != old_basis or stamp <= old_stamp:
                    counter["time_discontinuities"] += 1
                elif value < old_value:
                    counter["reset_intervals"] += 1
                else:
                    counter["observed_delta_us"] = (counter["observed_delta_us"] or 0) + value - old_value
                    counter["observed_interval_s"] += stamp - old_stamp
                    counter["valid_intervals"] += 1
        for _missing in previous.keys() - current.keys():
            for counter in counters.values():
                counter["unavailable_intervals"] += 1
        # A missing sample or changed GPU cannot bridge counter lifetimes.
        previous = current
    for counter in counters.values():
        delta = counter["observed_delta_us"]
        counter["observed_delta_s"] = delta / 1_000_000 if delta is not None else None
        incomplete = any(counter[key] for key in ("reset_intervals", "unavailable_intervals", "time_discontinuities"))
        counter["availability"] = "unavailable" if delta is None else "partial" if incomplete else "available"
    return counters


def _combine_throttle_deltas(trials):
    combined = {}
    for field in _THROTTLE_COUNTERS:
        samples = [trial[field] for trial in trials]
        deltas = [sample["observed_delta_us"] for sample in samples if sample["observed_delta_us"] is not None]
        total = sum(deltas) if deltas else None
        combined[field] = {"observed_delta_us": total, "observed_delta_s": total / 1_000_000 if total is not None else None,
            **{key: sum(sample[key] for sample in samples) for key in
               ("observed_interval_s", "valid_intervals", "reset_intervals", "unavailable_intervals", "time_discontinuities")},
            "trial_count": len(samples), "available_trials": len(deltas),
            "availability": "unavailable" if not deltas else
                "available" if all(sample["availability"] == "available" for sample in samples) else "partial"}
    return combined


def _telemetry(rows, *, counter_trials=None):
    metrics = {}
    for name, (section, field) in _TELEMETRY_FIELDS.items():
        values, available = [], 0
        for row in rows:
            parts = row.get(section) or ([] if section == "gpus" else {})
            if isinstance(parts, dict):
                parts = [parts]
            current = [_number(part.get(field)) for part in parts if isinstance(part, dict)]
            current = [value for value in current if value is not None]
            if current:
                available += 1
                values.extend(current)
        metrics[name] = {**_stats(values), "available_samples": available,
                         "unavailable_samples": len(rows) - available,
                         "availability": "unavailable" if not available else
                         "available" if available == len(rows) else "partial"}
    counters = defaultdict(set)
    for row in rows:
        sources = [("host", row.get("availability") or {})]
        sources += [(f"gpu{index}", gpu.get("availability") or {})
                    for index, gpu in enumerate(row.get("gpus") or []) if isinstance(gpu, dict)]
        for prefix, availability in sources:
            if isinstance(availability, dict):
                for key, value in availability.items():
                    counters[f"{prefix}.{key}"].add(json.dumps(value, sort_keys=True))
    return {"sample_count": len(rows), "metrics": metrics,
            "throttle_counters": _combine_throttle_deltas(counter_trials if counter_trials is not None else [_throttle_deltas(rows)]),
            "counter_availability": {key: [json.loads(value) for value in sorted(values)]
                                     for key, values in sorted(counters.items())},
            "notes": ["GPU memory utilization is not measured memory bandwidth; missing counters are unavailable.",
                "T.Limit is signed driver-reported thermal headroom; it is not an absolute temperature limit or a value to add to core temperature.",
                "Throttle counters are cumulative microseconds. Only adjacent increments for the same host/GPU UUID within a trial are counted; the first sample and cross-trial gaps are excluded.",
                "Counter decreases, time restarts and missing identity/values break continuity. Partial deltas are observed lower bounds; an unobserved driver reset with no visible decrease cannot be ruled out.",
                "Throttle categories can overlap. Their durations must not be summed as total throttled wall time."]}


def _trial(directory):
    issues = []
    timing_notes = []
    try:
        raw = _json(directory / "trial.json")
    except (OSError, ValueError, TypeError) as error:
        raw = {"trial_id": directory.name, "status": "incomplete"}
        issues.append(f"cannot read trial.json: {type(error).__name__}")
    if raw.get("measured") is False:
        return None
    rows = _jsonl(directory / "jobs.jsonl", issues)
    jobs, seen = [], set()
    for row in rows:
        if row.get("warmup"):
            continue
        identity = row.get("job_id")
        if not identity or identity in seen:
            issues.append("missing or duplicate job identity")
            continue
        seen.add(identity)
        jobs.append(row)
    start, end = _number(raw.get("started_at")), _number(raw.get("finished_at"))
    duration = _number(raw.get("duration_s"))
    if "duration_s" in raw and (duration is None or duration <= 0):
        issues.append("invalid monotonic trial duration")
    valid_epoch = start is not None and end is not None and end > start
    makespan = duration if duration is not None and duration > 0 else end - start if valid_epoch else None
    if makespan is None:
        issues.append("missing or invalid measured trial interval")
    if duration is not None and not valid_epoch:
        timing_notes.append("wall clock jumped or epoch timestamps are unavailable; monotonic trial duration used")
    if raw.get("status") not in {"completed", "complete", "succeeded", "ok"}:
        issues.append(f"trial status is {raw.get('status') or 'missing'}")
    if raw.get("stop_reason"):
        issues.append("trial stopped: " + str(raw["stop_reason"]))
    expected = raw.get("expected_jobs")
    if not isinstance(expected, int) or isinstance(expected, bool) or expected < 1:
        issues.append("missing or invalid expected_jobs")
    elif len(jobs) != expected:
        issues.append(f"expected {expected} measured jobs, found {len(jobs)}")
    concurrency = raw.get("concurrency")
    dispatchers = raw.get("dispatcher_count", 1)
    for name, value in (("concurrency", concurrency), ("dispatcher_count", dispatchers)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            issues.append(f"invalid {name}")
            if name == "concurrency":
                concurrency = None
            else:
                dispatchers = None
    failures = Counter()
    for job in jobs:
        category = _failure(job)
        if category:
            failures[category] += 1
            if category in {"censored", "timeout", "cancelled", "infrastructure_or_unknown"}:
                issues.append("incomplete outcome: " + category)
        if job.get("concurrency", concurrency) != concurrency or job.get("dispatcher_count", dispatchers) != dispatchers:
            issues.append("job/trial concurrency or dispatcher mismatch")
        a, b = _number(job.get("started_at")), _number(job.get("finished_at"))
        monotonic_latency = _number(job.get("latency_s")) if duration is not None else None
        if monotonic_latency is not None and monotonic_latency < 0:
            issues.append("job has negative latency")
        if (a is None or b is None or b <= a) and monotonic_latency is None:
            issues.append("job has missing or invalid timing")
        elif monotonic_latency is not None and (a is None or b is None or b <= a):
            timing_notes.append("job wall clock jumped or epoch timestamps are unavailable; monotonic latency used")
        elif duration is None and valid_epoch and (a < start or b > end):
            issues.append("job interval falls outside measured trial")
        if not job.get("workload_id") or job.get("seed") is None:
            issues.append("job lacks workload or seed identity")
        retry_count = job.get("retry_count", 0)
        if not isinstance(retry_count, int) or isinstance(retry_count, bool) or retry_count < 0:
            issues.append("job has invalid retry_count")
    request_maximum, request_censored = _overlap(jobs, end)
    mono_keys = ("inference_started_monotonic", "inference_finished_monotonic")
    epoch_keys = ("inference_started_at", "inference_finished_at")
    keys = mono_keys if any(_number(job.get(mono_keys[0])) is not None for job in jobs) else epoch_keys
    inference_jobs = [job for job in jobs if _number(job.get(keys[0])) is not None]
    inference_maximum, censored = _overlap(inference_jobs, None, *keys)
    missing_inference = any(_failure(job) is None and (_number(job.get(keys[0])) is None
                            or _number(job.get(keys[1])) is None) for job in jobs)
    maximum = None if not inference_jobs or censored or missing_inference else inference_maximum
    if maximum is not None and isinstance(concurrency, int) and maximum > concurrency:
        issues.append("actual overlap exceeds configured concurrency")
    telemetry_issues = []
    telemetry = _jsonl(directory / "telemetry.jsonl", telemetry_issues)
    telemetry = [row for row in telemetry if (stamp := _timestamp(row)) is None
                 or not valid_epoch or start <= stamp <= end]
    telemetry_metrics = _telemetry(telemetry)
    telemetry_unavailable = [name for name in _REQUIRED_TELEMETRY
                             if telemetry_metrics["metrics"][name]["availability"] == "unavailable"]
    if not telemetry or not any(_timestamp(row) is not None for row in telemetry):
        telemetry_unavailable.append("timestamped trial telemetry")
    if telemetry_issues:
        issues.extend("incomplete telemetry: " + issue for issue in telemetry_issues)
    corpus = sorted(Counter((str(job.get("workload_id") or "unknown"), str(job.get("seed")))
                            for job in jobs).items())
    corpus_id = hashlib.sha256(json.dumps(corpus).encode()).hexdigest()[:12]
    return {"trial_id": raw.get("trial_id") or directory.name, "concurrency": concurrency,
            "dispatcher_count": dispatchers, "started_at": start, "finished_at": end,
            "makespan_s": makespan, "expected_jobs": expected, "job_count": len(jobs),
            "successes": sum(_failure(job) is None for job in jobs),
            "failures_by_category": dict(failures), "max_active_observed": maximum,
            "inference_max_active_observed": maximum, "request_max_active_observed": request_maximum,
            "request_censored_intervals": request_censored,
            "overlap_basis": "inference_monotonic" if keys == mono_keys else "inference_epoch" if inference_jobs else "unavailable",
            "censored_intervals": censored, "status": raw.get("status"),
            "stop_reason": raw.get("stop_reason"), "complete": not issues,
            "issues": sorted(set(issues)), "corpus_id": corpus_id,
            "workload_counts": dict(Counter(str(job.get("workload_id") or "unknown") for job in jobs)),
            "timing_notes": sorted(set(timing_notes)), "telemetry_issues": telemetry_issues,
            "telemetry_sample_count": len(telemetry), "telemetry_required_unavailable": telemetry_unavailable,
            "thermal_headroom_min_c": telemetry_metrics["metrics"]["gpu_temperature_tlimit_c"]["min"],
            "throttle_counters": telemetry_metrics["throttle_counters"],
            "_jobs": jobs, "_telemetry": telemetry}


def _aggregate(trials):
    jobs = [job for trial in trials for job in trial["_jobs"]]
    successful = [job for job in jobs if _failure(job) is None]
    makespans = [trial["makespan_s"] for trial in trials]
    makespan = sum(makespans) if all(value is not None for value in makespans) else None

    def per_hour(value):
        return value * 3600 / makespan if makespan else None

    def latency(job):
        value = _number(job.get("latency_s"))
        if value is not None and value >= 0:
            return value
        start, finish = _number(job.get("started_at")), _number(job.get("finished_at"))
        return finish - start if start is not None and finish is not None and finish > start else None

    durations = [_number(job.get("audio_duration_s")) for job in successful]
    audio_duration = sum(durations) if all(value is not None and value >= 0 for value in durations) else None
    failures = Counter(_failure(job) for job in jobs if _failure(job))
    group = {"corpus_id": trials[0]["corpus_id"], "workload_counts": trials[0]["workload_counts"],
             "concurrency": trials[0]["concurrency"], "dispatcher_count": trials[0]["dispatcher_count"],
             "trial_ids": [trial["trial_id"] for trial in trials], "trial_count": len(trials),
             "complete_trial_count": sum(trial["complete"] for trial in trials),
             "job_count": len(jobs), "successes": len(successful), "makespan_s": makespan,
             "jobs_per_hour": per_hour(len(successful)), "audio_duration_s": audio_duration,
             "audio_seconds_per_hour": per_hour(audio_duration) if audio_duration is not None else None,
             "latency_s": _stats([latency(job) for job in successful]),
             "all_outcome_latency_s": _stats([latency(job) for job in jobs]),
             "inference_s": _stats([_number(job.get("inference_s")) for job in successful]),
             "retry_count": sum(job.get("retry_count", 0) for job in jobs
                                if isinstance(job.get("retry_count", 0), int) and job.get("retry_count", 0) > 0),
             "failures_by_category": dict(failures),
             "failure_rate": sum(failures.values()) / len(jobs) if jobs else None,
             "max_active_observed": max(trial["max_active_observed"] for trial in trials)
                 if all(trial["max_active_observed"] is not None for trial in trials) else None,
             "request_max_active_observed": max(trial["request_max_active_observed"] for trial in trials),
             "overlap_basis": sorted({trial["overlap_basis"] for trial in trials}),
             "reported_max_active_observed": max((value for job in jobs
                 if (value := _number(job.get("max_active_observed"))) is not None), default=None),
             "telemetry": _telemetry([row for trial in trials for row in trial["_telemetry"]],
                                     counter_trials=[_throttle_deltas(trial["_telemetry"]) for trial in trials]),
             "issues": sorted({issue for trial in trials for issue in trial["issues"]}),
             "timing_notes": sorted({note for trial in trials for note in trial["timing_notes"]})}
    group["workloads"] = {workload: {"job_count": sum(job.get("workload_id") == workload for job in jobs),
        "successes": sum(job.get("workload_id") == workload for job in successful),
        "jobs_per_hour": per_hour(sum(job.get("workload_id") == workload for job in successful)),
        "latency_s": _stats([latency(job) for job in successful if job.get("workload_id") == workload])}
        for workload in group["workload_counts"]}
    durations_by_input = defaultdict(list)
    for job in successful:
        value = _number(job.get("audio_duration_s"))
        if value is not None and value >= 0:
            durations_by_input[(str(job.get("workload_id")), str(job.get("seed")))].append(value)
    group["audio_duration_by_input"] = [{"workload_id": workload, "seed": seed, **_stats(values)}
                                       for (workload, seed), values in sorted(durations_by_input.items())]
    return group


def _display(value, digits=3):
    if value is None:
        return "unavailable"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def analyze(root: Path, *, min_speedup=1.10, max_latency_ratio=3.0, max_failure_rate=0.0) -> dict:
    """Write comparison.csv, report.md, summary.json without executing workloads."""
    root = Path(root)
    for name, value, minimum in (("min_speedup", min_speedup, 1), ("max_latency_ratio", max_latency_ratio, 0),
                                ("max_failure_rate", max_failure_rate, 0)):
        if _number(value) is None or value < minimum:
            raise ValueError(f"invalid {name}")
    if max_failure_rate > 1:
        raise ValueError("max_failure_rate must be at most 1")
    metadata = _json(root / "run-metadata.json")
    trials = [trial for directory in sorted((root / "trials").glob("*")) if directory.is_dir()
              if (trial := _trial(directory)) is not None]
    grouped = defaultdict(list)
    for trial in trials:
        grouped[(trial["corpus_id"], str(trial["dispatcher_count"]), str(trial["concurrency"]))].append(trial)
    comparisons = [_aggregate(items) for _, items in sorted(grouped.items())]
    levels = metadata.get("concurrency_levels", metadata.get("levels", []))
    levels = sorted({level for level in levels if isinstance(level, int) and not isinstance(level, bool) and level >= 1})
    issues = []
    if metadata.get("status") not in {"complete", "completed", "succeeded", "ok"}:
        issues.append(f"run metadata status is {metadata.get('status') or 'missing'}; run incomplete")
    if not levels:
        issues.append("requested concurrency levels are missing from metadata")
    if 1 not in levels:
        issues.append("requested levels do not include C1")
    if not any(level > 1 for level in levels):
        issues.append("no higher concurrency level requested for comparison")
    if not trials:
        issues.append("no measured trials")
    trial_ids = [trial["trial_id"] for trial in trials]
    if len(set(trial_ids)) != len(trial_ids):
        issues.append("duplicate measured trial identity")
    intervals = sorted((trial["started_at"], trial["finished_at"]) for trial in trials
                       if trial["started_at"] is not None and trial["finished_at"] is not None)
    consecutive = zip(intervals, intervals[1:], strict=False)  # noqa: RUF007 — adjacent intervals intentionally have unequal lengths
    if any(next_start < finish for (_, finish), (next_start, _) in consecutive) and not any(trial["timing_notes"] for trial in trials):
        issues.append("measured trials overlap on the same node")
    if any(not trial["complete"] for trial in trials):
        issues.append("incomplete measured trials or outcomes")
    synthetic = (metadata.get("synthetic") is True or str(metadata.get("synthetic")).lower() == "true"
                 or any(job.get("synthetic") is True for trial in trials for job in trial["_jobs"]))
    cohorts = defaultdict(list)
    for comparison in comparisons:
        cohorts[(comparison["corpus_id"], str(comparison["dispatcher_count"]))].append(comparison)
    recommendations = []
    for (corpus_id, dispatchers), groups in sorted(cohorts.items()):
        groups.sort(key=lambda item: item["concurrency"] if isinstance(item["concurrency"], int) else -1)
        baseline = next((item for item in groups if item["concurrency"] == 1), None)
        observed = {item["concurrency"] for item in groups}
        cohort_issues = []
        if not baseline:
            cohort_issues.append("missing matching C1 baseline")
        missing = sorted(set(levels) - observed)
        if missing:
            cohort_issues.append("missing requested levels: " + ", ".join(f"C{level}" for level in missing))
        if any(item["complete_trial_count"] < 2 for item in groups):
            cohort_issues.append("fewer than two complete independent trials per level")
        if any(item["max_active_observed"] is None for item in groups):
            cohort_issues.append("actual inference overlap unavailable or censored")
        elif any(isinstance(item["concurrency"], int) and item["max_active_observed"] < item["concurrency"] for item in groups):
            cohort_issues.append("configured concurrency was not exercised by actual overlap")
        if len(groups[0]["workload_counts"]) < 2:
            cohort_issues.append("fewer than two distinct measured workloads")
        iterations = metadata.get("iterations", metadata.get("repetitions"))
        if not isinstance(iterations, int) or iterations < 2:
            cohort_issues.append("metadata does not establish at least two iterations")
        elif any(item["trial_count"] < iterations for item in groups):
            cohort_issues.append("missing requested trial repetitions")
        expected_jobs = metadata.get("jobs_per_level")
        cohort_trials = [trial for item in groups
                         for trial in grouped[(item["corpus_id"], str(item["dispatcher_count"]), str(item["concurrency"]))]]
        if isinstance(expected_jobs, int) and any(trial["expected_jobs"] != expected_jobs for trial in cohort_trials):
            cohort_issues.append("incomplete corpus: expected_jobs differs from metadata jobs_per_level")
        unavailable_telemetry = sorted({name for trial in cohort_trials for name in trial["telemetry_required_unavailable"]})
        if unavailable_telemetry:
            cohort_issues.append("required resource telemetry unavailable in measured trials: " + ", ".join(unavailable_telemetry))
        if baseline and (not baseline["jobs_per_hour"] or baseline["latency_s"]["p95"] is None):
            cohort_issues.append("missing usable C1 success rate or latency")
        if baseline and baseline["failure_rate"] is not None and baseline["failure_rate"] > max_failure_rate:
            cohort_issues.append("C1 baseline exceeds acceptable failure rate")
        previous = None
        eligible = []
        for item in groups:
            rate = item["jobs_per_hour"]
            base_rate = baseline["jobs_per_hour"] if baseline else None
            base_latency = baseline["latency_s"]["p95"] if baseline else None
            latency = item["latency_s"]["p95"]
            item["speedup_vs_c1"] = rate / base_rate if rate is not None and base_rate else None
            item["p95_latency_ratio_vs_c1"] = latency / base_latency if latency is not None and base_latency else None
            item["marginal_jobs_per_hour"] = rate - previous["jobs_per_hour"] if previous and rate is not None and previous["jobs_per_hour"] is not None else None
            item["previous_concurrency"] = previous["concurrency"] if previous else None
            added_slots = (item["concurrency"] - previous["concurrency"] if previous and
                           isinstance(item["concurrency"], int) and isinstance(previous["concurrency"], int) else None)
            item["marginal_jobs_per_hour_per_slot"] = (item["marginal_jobs_per_hour"] / added_slots
                if added_slots and added_slots > 0 and item["marginal_jobs_per_hour"] is not None else None)
            item["marginal_speedup"] = rate / previous["jobs_per_hour"] if previous and rate is not None and previous["jobs_per_hour"] else None
            item["scaling_efficiency"] = item["speedup_vs_c1"] / item["concurrency"] if item["speedup_vs_c1"] is not None and item["concurrency"] else None
            baseline_audio = {(row["workload_id"], row["seed"]): row["mean"]
                              for row in baseline["audio_duration_by_input"]} if baseline else {}
            duration_comparison = []
            for row in item["audio_duration_by_input"]:
                base_duration = baseline_audio.get((row["workload_id"], row["seed"]))
                ratio = row["mean"] / base_duration if base_duration else None
                duration_comparison.append({"workload_id": row["workload_id"], "seed": row["seed"],
                                            "mean_duration_ratio_vs_c1": ratio,
                                            "material_shortening": ratio is not None and ratio < .8})
            item["output_duration_comparison"] = duration_comparison
            item["material_shortening"] = any(row["material_shortening"] for row in duration_comparison)
            if item["material_shortening"]:
                cohort_issues.append(f"C{item['concurrency']} output shortened by over 20% for matched inputs; manual quality review required")
            item["eligible"] = bool(item["speedup_vs_c1"] is not None and
                (item["concurrency"] == 1 or item["speedup_vs_c1"] >= min_speedup) and
                item["p95_latency_ratio_vs_c1"] is not None and item["p95_latency_ratio_vs_c1"] <= max_latency_ratio and
                item["failure_rate"] is not None and item["failure_rate"] <= max_failure_rate and not item["issues"] and
                not item["material_shortening"])
            if item["eligible"]:
                eligible.append(item)
            previous = item
        best = max(eligible, key=lambda item: (item["jobs_per_hour"], -item["concurrency"]), default=None)
        recommendations.append({"corpus_id": corpus_id, "dispatcher_count": groups[0]["dispatcher_count"],
            "candidate_concurrency": best["concurrency"] if best else None,
            "provisional": True, "issues": cohort_issues, "missing_levels": missing})
        issues.extend(f"corpus {corpus_id}, dispatcher {dispatchers}: {issue}" for issue in cohort_issues)
    incomplete = any("missing" in issue or "incomplete" in issue or "duplicate" in issue or "no measured" in issue or "overlap on" in issue for issue in issues)
    evidence = "synthetic" if synthetic else "incomplete" if incomplete else "underpowered" if issues else "complete"
    hypothesis = "not_assessed"
    if evidence == "complete":
        hypothesis = "h0_falsified" if any(item["candidate_concurrency"] and item["candidate_concurrency"] > 1 for item in recommendations) else "h0_not_rejected"
        for item in recommendations:
            item["provisional"] = False
    summary = {"schema_version": 1, "synthetic": synthetic, "metadata": metadata,
        "evidence_status": evidence, "hypothesis_result": hypothesis,
        "hypothesis": "C1 provides the best acceptable throughput within the tested workloads, node, runtime and dispatcher topology.",
        "thresholds": {"min_speedup": min_speedup, "max_latency_ratio": max_latency_ratio, "max_failure_rate": max_failure_rate},
        "requested_levels": levels, "issues": sorted(set(issues)), "comparisons": comparisons,
        "recommendations": recommendations,
        "trials": [{key: value for key, value in trial.items() if not key.startswith("_")} for trial in trials],
        "limitations": ["Synthetic runs validate the harness only; they provide no hardware capacity evidence.",
            "Two repetitions are a minimum evidence gate, not a statistical significance test.",
            "Throughput uses all measured trial wall time, including failed jobs, loading, validation and cleanup.",
            "Successful-job latency quantiles exclude failed and censored jobs; failure rates are reported separately.",
            "Corpus matching includes workload IDs, seeds and counts; dispatcher topologies are compared separately.",
            "No conclusion about saturation or untested concurrency levels follows from these results."]}
    summary["limitations"].append("Capacity evidence requires measured memory availability, temperature, CPU utilization and GPU utilization in every trial; advanced counters may remain explicitly unavailable.")
    summary["limitations"].append("A mean duration drop over 20% for a matching workload/seed triggers manual quality review, not an automatic capacity recommendation.")
    _write_outputs(root, summary)
    return summary


def _cost(item):
    if item is None:
        return "unavailable"
    metrics = item["telemetry"]["metrics"]

    def resource(name, statistic, unit, scale=1):
        metric = metrics[name]
        value = metric[statistic]
        return f"{_display(value / scale if value is not None else None)} {unit} ({metric['availability']})"

    return (f"C{item['concurrency']}: p50/p95 {_display(item['latency_s']['p50'])}/"
        f"{_display(item['latency_s']['p95'])} s; GPU/host memory peak "
        f"{resource('gpu_memory_used_bytes', 'max', 'GiB', 2**30)}/"
        f"{resource('memory_used_bytes', 'max', 'GiB', 2**30)}; host memory available minimum "
        f"{resource('memory_available_bytes', 'min', 'GiB', 2**30)}; GPU power mean/max "
        f"{resource('gpu_power_watts', 'mean', 'W')}/{resource('gpu_power_watts', 'max', 'W')}; "
        f"core temperature peak {resource('gpu_temperature_c', 'max', '°C')}; "
        f"T.Limit minimum {resource('gpu_temperature_tlimit_c', 'min', '°C')}; "
        f"failures {sum(item['failures_by_category'].values())}/{item['job_count']} "
        f"({json.dumps(item['failures_by_category'], sort_keys=True)}), retries {item['retry_count']}")


def _decision_report(summary):
    """Present observed rates and existing recommendations without reselecting capacity."""
    report = ["## Decision fields", ""]
    cohorts = summary["recommendations"] or [None]
    for recommendation in cohorts:
        groups = ([item for item in summary["comparisons"] if
                   (item["corpus_id"], item["dispatcher_count"]) ==
                   (recommendation["corpus_id"], recommendation["dispatcher_count"])] if recommendation else [])
        if recommendation:
            report += [f"Corpus {recommendation['corpus_id']}; {recommendation['dispatcher_count']} configured dispatcher(s).", ""]
        baseline = next((item for item in groups if item["concurrency"] == 1), None)
        best = max((item for item in groups if item["jobs_per_hour"] is not None),
                   key=lambda item: (item["jobs_per_hour"], -(item["concurrency"] or 0)), default=None)
        provisional = summary["evidence_status"] != "complete" or not recommendation or recommendation["provisional"]
        pending = "pending / not_assessed; " if provisional else ""
        candidate = recommendation["candidate_concurrency"] if recommendation else None
        baseline_text = (f"C1, {_display(baseline['jobs_per_hour'])} successful jobs/hour, "
                         f"{baseline['complete_trial_count']} complete trial(s)." if baseline else
                         "unavailable; no matching C1 baseline.")
        if best:
            best_text = (f"C{best['concurrency']}, {_display(best['jobs_per_hour'])} successful jobs/hour "
                         "(highest observed rate; " + ("meets" if best["eligible"] else "does not meet")
                         + " acceptance criteria" + ("; provisional" if provisional else "") + ").")
            speedup_text = (f"{_display(best['speedup_vs_c1'])}× matching C1; scaling efficiency "
                           f"{_display(best['scaling_efficiency'])}; ")
            speedup_text += (f"marginal speedup {_display(best['marginal_speedup'])}× "
                f"from C{best['previous_concurrency']}; "
                f"{_display(best['marginal_jobs_per_hour_per_slot'])} additional jobs/hour per added slot."
                if best["previous_concurrency"] is not None else "marginal comparison unavailable (no lower matching level).")
        else:
            best_text, speedup_text = "unavailable; no measured throughput.", "unavailable; comparison pending."
        cost = _cost(baseline)
        if best is not None and best is not baseline:
            cost += " → " + _cost(best)
        if baseline is None and best is None:
            cost = "unavailable; no measured comparison."
        limits = []
        if best:
            limits = [f"{name.removeprefix('clock_event_').removesuffix('_us')} "
                      f"{_display(counter['observed_delta_s'], 6)} s ({counter['availability']})"
                      for name, counter in best["telemetry"]["throttle_counters"].items()
                      if counter["observed_delta_s"] is not None and counter["observed_delta_s"] > 0]
        bottleneck = "unresolved; GPU utilization, even 99%, does not establish compute or memory-bandwidth saturation."
        if limits:
            bottleneck += " Observed limit-event counters: " + "; ".join(limits) + "; these do not identify the dominant bottleneck."
        else:
            bottleneck += " No supporting limit-event evidence establishes a cause; missing counters remain unavailable."
        boundary = bool(best and best["eligible"] and summary["requested_levels"] and
                        best["concurrency"] == max(summary["requested_levels"]) and
                        best["marginal_speedup"] is not None and
                        best["marginal_speedup"] >= summary["thresholds"]["min_speedup"])
        if provisional:
            conclusion = f"pending / not_assessed: {summary['evidence_status']} evidence; no hardware H0 verdict."
            capacity = "pending; no supported capacity."
            if candidate is not None:
                capacity += (f" C{candidate} with {recommendation['dispatcher_count']} dispatcher(s) is a provisional "
                             + ("harness candidate only." if summary["synthetic"] else "candidate only."))
        else:
            conclusion = (f"C{candidate} improves acceptable throughput over C1 within this measured corpus, node, runtime "
                          "and dispatcher topology." if candidate and candidate > 1 else
                          "H0 not rejected within this tested envelope; C1 is not established as universally optimal.")
            capacity = (f"{recommendation['dispatcher_count']} dispatcher(s), {candidate} concurrent job(s) per tested Spark "
                        "for this corpus/runtime; other nodes and dispatcher topologies are not established."
                        if candidate is not None else "none; no tested level meets the acceptance criteria.")
        if boundary and not summary["synthetic"]:
            conclusion += (" Ceiling not found: the highest requested level still shows useful marginal scaling"
                           + (" in provisional observations." if provisional else "."))
        if summary["synthetic"]:
            next_step = "Run the same full owned corpus and paired seeds in an isolated, guarded real sweep; hardware scaling and ceiling remain unassessed."
        elif provisional:
            next_step = "Resolve the listed evidence gaps and complete repeated levels with the same full corpus and paired seeds before recommending capacity."
            if boundary:
                next_step += " Then extend above the current highest level in a separate guarded sweep."
        elif boundary:
            next_step = (f"Extend above C{best['concurrency']} in a separate guarded, repeated sweep using the same full corpus "
                         "and paired seeds, with enough jobs to exercise each new level; do not call the current boundary an optimum.")
        else:
            next_step = "Repeat the candidate and neighboring levels with more jobs/trials; correlate stage timing with SM/tensor, measured bandwidth and limit counters to identify the bottleneck."
        report += [f"- **BASELINE:** {pending}{baseline_text}", f"- **BEST TESTED:** {pending}{best_text}",
                   f"- **SPEEDUP:** {pending}{speedup_text}", f"- **COST:** {cost}",
                   f"- **BOTTLENECK:** {bottleneck}", f"- **CONCLUSION:** {conclusion}",
                   f"- **RECOMMENDED CAPACITY:** {capacity}", f"- **NEXT EXPERIMENT:** {next_step}", ""]
    return report


def _write_outputs(root, summary):
    columns = ["corpus_id", "dispatcher_count", "concurrency", "trial_count", "complete_trial_count", "job_count",
        "successes", "makespan_s", "jobs_per_hour", "audio_seconds_per_hour", "failure_rate", "retry_count",
        "max_active_observed", "speedup_vs_c1", "scaling_efficiency", "p95_latency_ratio_vs_c1", "previous_concurrency",
        "marginal_speedup", "marginal_jobs_per_hour", "marginal_jobs_per_hour_per_slot", "eligible"]
    quantiles = ["mean", "p50", "p95", "min", "max"]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns + ["latency_s_" + key for key in quantiles] + ["failures_by_category"])
    writer.writeheader()
    for item in summary["comparisons"]:
        row = {key: item[key] for key in columns}
        row.update({"latency_s_" + key: item["latency_s"][key] for key in quantiles})
        row["failures_by_category"] = json.dumps(item["failures_by_category"], sort_keys=True)
        writer.writerow(row)
    report = ["# YuE concurrency comparison", "", f"Evidence: **{summary['evidence_status']}**. Hypothesis: **{summary['hypothesis_result']}**.", "",
        "H0 means C1 has the best acceptable throughput in the tested envelope. H0 not rejected is not proof that C1 is universally optimal.", ""]
    metadata = summary["metadata"]
    report += [(f"Node: {metadata.get('node', 'unavailable')}. Worker: {metadata.get('worker', 'unavailable')}. "
                f"Runtime manifest: {metadata.get('runtime_manifest', 'unavailable')}."),
               (f"Requested levels: {summary['requested_levels']}; iterations: {metadata.get('iterations', 'unavailable')}; "
                f"jobs per trial: {metadata.get('jobs_per_level', 'unavailable')}."),
               "Planned level order: " + json.dumps(metadata.get("order", "unavailable")) + ".", ""]
    if summary["synthetic"]:
        report += ["**SYNTHETIC HARNESS DATA — no hardware capacity conclusion.**", ""]
    report += _decision_report(summary)
    report += ["| Corpus | Dispatchers | C | Trials (complete) | Success/jobs | Jobs/hour | p50 / p95 latency (s) | Speedup | Failure rate | Actual overlap |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for item in summary["comparisons"]:
        report.append(f"| {item['corpus_id']} | {item['dispatcher_count']} | {item['concurrency']} | "
            f"{item['trial_count']} ({item['complete_trial_count']}) | {item['successes']}/{item['job_count']} | "
            f"{_display(item['jobs_per_hour'])} | {_display(item['latency_s']['p50'])} / {_display(item['latency_s']['p95'])} | "
            f"{_display(item['speedup_vs_c1'])} | {_display(item['failure_rate'])} | {_display(item['max_active_observed'])} |")
    report += ["", "## Capacity candidates", ""]
    for item in summary["recommendations"]:
        candidate = "none" if item["candidate_concurrency"] is None else f"C{item['candidate_concurrency']}"
        report.append(f"- Corpus {item['corpus_id']}, {item['dispatcher_count']} dispatcher(s): {candidate}; "
                      + ("provisional, insufficient evidence." if item["provisional"] else "supported within this tested envelope."))
    report += ["", "## Evidence gaps", ""] + ["- " + issue for issue in summary["issues"]]
    if not summary["issues"]:
        report.append("No completeness or minimum-repetition gaps detected.")
    report += ["", "## Resource telemetry", ""]
    for item in summary["comparisons"]:
        report.append(f"- {item['corpus_id']} / D{item['dispatcher_count']} / C{item['concurrency']}: " + "; ".join(
            f"{name}: {_display(item['telemetry']['metrics'][name]['max'])} ({item['telemetry']['metrics'][name]['availability']})"
            for name in ("gpu_temperature_c", "gpu_power_watts", "gpu_memory_used_bytes", "memory_used_bytes")))
        headroom = item["telemetry"]["metrics"]["gpu_temperature_tlimit_c"]
        report.append(f"  T.Limit headroom minimum: {_display(headroom['min'])} °C ({headroom['availability']}). "
            + "Observed throttle seconds within trials: " + "; ".join(
                f"{field.removeprefix('clock_event_').removesuffix('_us')}: {_display(counter['observed_delta_s'], 6)} "
                f"({counter['availability']}; resets={counter['reset_intervals']})"
                for field, counter in item["telemetry"]["throttle_counters"].items()) + ".")
    report += ["", "Missing telemetry is unavailable, never zero. Detailed counter availability and per-workload results are in summary.json.",
               "T.Limit is signed thermal headroom, not an absolute threshold. Throttle deltas exclude the initial lifetime totals, trial gaps and reset/gap intervals; partial values are lower bounds. Categories may overlap.",
               "", "## Interpretation limits", ""] + ["- " + note for note in summary["limitations"]]
    for name, content in (("comparison.csv", output.getvalue()), ("report.md", "\n".join(report) + "\n"),
                          ("summary.json", json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")):
        temporary = root / (name + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(root / name)
