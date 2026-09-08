import csv
import importlib.util
import json
from pathlib import Path

import pytest

# Keep the analysis tests independently runnable while the CLI is being staged.
_spec = importlib.util.spec_from_file_location(
    "bench_analysis", Path(__file__).resolve().parents[1] / "spark_bench/analysis.py")
_analysis = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_analysis)
analyze = _analysis.analyze


def metadata(root, **updates):
    value = {"concurrency_levels": [1, 2], "iterations": 2, "jobs_per_level": 4,
             "workload_sha256": "owned-corpus", "synthetic": False}
    value.update(updates)
    (root / "run-metadata.json").write_text(json.dumps(value))


def trial(root, concurrency, iteration=0, *, dispatchers=1, workloads=("short", "representative"),
          statuses=None, measured=True, duration=None, finished=True, telemetry=None, warmup=False):
    identity = f"d{dispatchers}-c{concurrency}-r{iteration}"
    directory = root / "trials" / identity
    directory.mkdir(parents=True)
    start = 1000 + iteration * 10000 + dispatchers * 1000 + concurrency * 100
    jobs = []
    for index in range(4):
        begin = start + index // concurrency * 10
        status = statuses[index] if statuses else "succeeded"
        jobs.append({"job_id": f"{identity}-job{index}", "workload_id": workloads[index % len(workloads)],
            "seed": 42 + index // len(workloads), "concurrency": concurrency, "dispatcher_count": dispatchers,
            "warmup": False, "status": status, "started_at": begin, "finished_at": begin + 10,
            "inference_started_at": begin + 1, "inference_finished_at": begin + 9,
            "latency_s": 10, "inference_s": 8, "audio_duration_s": 30,
            "retry_count": 0, "max_active_observed": concurrency})
    if warmup:
        jobs.append(dict(jobs[0], job_id=identity + "-warmup", warmup=True,
                         started_at=start - 100, finished_at=start - 10, latency_s=90))
    span = duration or ((4 + concurrency - 1) // concurrency) * 10
    record = {"trial_id": identity, "concurrency": concurrency, "dispatcher_count": dispatchers,
              "started_at": start, "finished_at": start + span if finished else None,
              "measured": measured, "expected_jobs": 4, "status": "completed", "stop_reason": ""}
    (directory / "trial.json").write_text(json.dumps(record))
    (directory / "jobs.jsonl").write_text("\n".join(json.dumps(job) for job in jobs) + "\n")
    (directory / "telemetry.jsonl").write_text("\n".join(json.dumps(row) for row in telemetry or []))
    return directory


def complete_matrix(root, **kwargs):
    metadata(root, **kwargs)
    for concurrency in (1, 2):
        for iteration in range(2):
            trial(root, concurrency, iteration)


def test_known_throughput_latency_scaling_and_artifacts(tmp_path):
    complete_matrix(tmp_path)
    result = analyze(tmp_path)
    assert result["evidence_status"] == "complete"
    assert result["hypothesis_result"] == "h0_falsified"
    baseline, parallel = sorted(result["comparisons"], key=lambda row: row["concurrency"])
    assert baseline["makespan_s"] == 80
    assert baseline["jobs_per_hour"] == 360
    assert parallel["jobs_per_hour"] == 720
    assert parallel["audio_seconds_per_hour"] == 21600
    assert parallel["latency_s"] == {"count": 8, "mean": 10, "p50": 10, "p95": 10, "min": 10, "max": 10}
    assert parallel["speedup_vs_c1"] == 2
    assert parallel["p95_latency_ratio_vs_c1"] == 1
    assert parallel["marginal_jobs_per_hour"] == 360
    assert parallel["max_active_observed"] == 2
    assert baseline["max_active_observed"] == 1  # Equal finish/start timestamps do not overlap.
    assert parallel["workloads"]["short"]["jobs_per_hour"] == 360
    assert result["recommendations"][0]["candidate_concurrency"] == 2
    assert result["recommendations"][0]["provisional"] is False
    with (tmp_path / "comparison.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 2
    assert json.loads((tmp_path / "summary.json").read_text()) == result
    assert "h0_falsified" in (tmp_path / "report.md").read_text()


def test_failures_consume_full_wall_time_and_never_contribute_audio(tmp_path):
    metadata(tmp_path)
    for iteration in range(2):
        trial(tmp_path, 1, iteration)
        directory = trial(tmp_path, 2, iteration, duration=40,
                          statuses=["failed_integrity", "succeeded", "succeeded", "succeeded"])
        rows = [json.loads(line) for line in (directory / "jobs.jsonl").read_text().splitlines()]
        rows[0]["audio_duration_s"] = 9999
        rows[1]["retry_count"] = 2
        (directory / "jobs.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    result = analyze(tmp_path)
    parallel = next(row for row in result["comparisons"] if row["concurrency"] == 2)
    assert parallel["makespan_s"] == 80
    assert parallel["jobs_per_hour"] == 270
    assert parallel["audio_duration_s"] == 180
    assert parallel["failure_rate"] == .25
    assert parallel["failures_by_category"] == {"failed_integrity": 2}
    assert parallel["retry_count"] == 4
    assert not parallel["eligible"]
    assert result["hypothesis_result"] == "h0_not_rejected"


def test_warmups_excluded_and_synthetic_never_concludes(tmp_path):
    metadata(tmp_path, synthetic=True)
    for concurrency in (1, 2):
        for iteration in range(2):
            trial(tmp_path, concurrency, iteration, warmup=True)
    trial(tmp_path, 4, 9, measured=False)
    result = analyze(tmp_path)
    assert result["evidence_status"] == "synthetic"
    assert result["hypothesis_result"] == "not_assessed"
    assert all(row["job_count"] == 8 for row in result["comparisons"])
    assert len(result["trials"]) == 4
    assert all(row["provisional"] for row in result["recommendations"])
    assert "SYNTHETIC HARNESS DATA" in (tmp_path / "report.md").read_text()


@pytest.mark.parametrize("mode", ["running", "timeout", "cancelled", "failed", "missing_finish"])
def test_incomplete_and_censored_experiments_cannot_decide_hypothesis(tmp_path, mode):
    metadata(tmp_path)
    for concurrency in (1, 2):
        for iteration in range(2):
            trial(tmp_path, concurrency, iteration, finished=mode != "missing_finish" or concurrency == 1,
                  statuses=([mode, "succeeded", "succeeded", "succeeded"]
                            if concurrency == 2 and mode != "missing_finish" else None))
    result = analyze(tmp_path)
    assert result["evidence_status"] == "incomplete"
    assert result["hypothesis_result"] == "not_assessed"
    parallel = next(row for row in result["comparisons"] if row["concurrency"] == 2)
    if mode == "missing_finish":
        assert parallel["jobs_per_hour"] is None
    else:
        assert parallel["jobs_per_hour"] == 540
    assert all(row["provisional"] for row in result["recommendations"])


def test_missing_baseline_and_requested_levels_are_explicit(tmp_path):
    metadata(tmp_path, concurrency_levels=[1, 2, 3])
    trial(tmp_path, 2)
    trial(tmp_path, 2, 1)
    result = analyze(tmp_path)
    assert result["evidence_status"] == "incomplete"
    assert result["comparisons"][0]["speedup_vs_c1"] is None
    assert result["recommendations"][0]["missing_levels"] == [1, 3]
    assert "missing matching C1" in " ".join(result["issues"])


def test_mixed_corpora_and_dispatchers_never_share_a_baseline(tmp_path):
    metadata(tmp_path)
    for iteration in range(2):
        trial(tmp_path, 1, iteration, workloads=("short",))
        trial(tmp_path, 2, iteration, workloads=("representative",))
        trial(tmp_path, 2, iteration, dispatchers=2, workloads=("short",))
    result = analyze(tmp_path)
    assert len(result["comparisons"]) == 3
    assert all(row["speedup_vs_c1"] is None for row in result["comparisons"] if row["concurrency"] == 2)
    assert result["hypothesis_result"] == "not_assessed"


def test_underpowered_corpus_or_repetitions_stay_provisional(tmp_path):
    metadata(tmp_path, iterations=1)
    trial(tmp_path, 1, workloads=("one_song",))
    trial(tmp_path, 2, workloads=("one_song",))
    result = analyze(tmp_path)
    assert result["evidence_status"] == "underpowered"
    assert result["hypothesis_result"] == "not_assessed"
    assert any("two distinct" in issue for issue in result["issues"])
    assert any("two complete independent" in issue for issue in result["issues"])


def test_telemetry_availability_is_not_fabricated_zero(tmp_path):
    metadata(tmp_path)
    trial(tmp_path, 1, telemetry=[
        {"gpus": [{"temperature_gpu_c": 60, "power_draw_watts": None,
                   "availability": {"tensor_utilization": "unavailable"}}],
         "memory": {"available_bytes": 64000}, "availability": {"memory_bandwidth": False}},
        {"gpus": [{"temperature_gpu_c": 64, "power_draw_watts": 85}],
         "memory": {"available_bytes": 62000}},
    ])
    telemetry = analyze(tmp_path)["comparisons"][0]["telemetry"]
    assert telemetry["sample_count"] == 2
    assert telemetry["metrics"]["gpu_temperature_c"]["mean"] == 62
    assert telemetry["metrics"]["gpu_power_watts"]["mean"] == 85
    assert telemetry["metrics"]["gpu_power_watts"]["availability"] == "partial"
    assert telemetry["metrics"]["gpu_memory_used_bytes"]["max"] is None
    assert telemetry["metrics"]["gpu_memory_used_bytes"]["availability"] == "unavailable"
    assert telemetry["counter_availability"]["gpu0.tensor_utilization"] == ["unavailable"]


def test_missing_job_and_partial_jsonl_do_not_disappear_from_evidence(tmp_path):
    complete_matrix(tmp_path)
    path = tmp_path / "trials/d1-c2-r0/jobs.jsonl"
    rows = path.read_text().splitlines()
    path.write_text("\n".join(rows[:-1]) + '\n{"job_id":')
    result = analyze(tmp_path)
    assert result["evidence_status"] == "incomplete"
    assert result["hypothesis_result"] == "not_assessed"
    broken = next(row for row in result["trials"] if row["trial_id"] == "d1-c2-r0")
    assert any("expected 4" in issue for issue in broken["issues"])
    assert any("incomplete or invalid JSON" in issue for issue in broken["issues"])


def test_monotonic_durations_and_inference_overlap_survive_wall_clock_jump(tmp_path):
    complete_matrix(tmp_path)
    for directory in (tmp_path / "trials").iterdir():
        raw = json.loads((directory / "trial.json").read_text())
        duration = raw["finished_at"] - raw["started_at"]
        raw.update(duration_s=duration, finished_at=raw["started_at"] - 100)
        (directory / "trial.json").write_text(json.dumps(raw))
        rows = [json.loads(line) for line in (directory / "jobs.jsonl").read_text().splitlines()]
        for row in rows:
            row.update(inference_started_monotonic=row["inference_started_at"],
                       inference_finished_monotonic=row["inference_finished_at"],
                       finished_at=row["started_at"] - 100)
        (directory / "jobs.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    result = analyze(tmp_path)
    assert result["evidence_status"] == "complete"
    parallel = next(row for row in result["comparisons"] if row["concurrency"] == 2)
    assert parallel["jobs_per_hour"] == 720
    assert parallel["max_active_observed"] == 2
    assert parallel["overlap_basis"] == ["inference_monotonic"]
    assert any("wall clock jumped" in note for note in parallel["timing_notes"])


def test_request_overlap_does_not_prove_inference_overlap(tmp_path):
    complete_matrix(tmp_path)
    for directory in (tmp_path / "trials").glob("*-c2-*"):
        rows = [json.loads(line) for line in (directory / "jobs.jsonl").read_text().splitlines()]
        for row in rows:
            row.pop("inference_started_at")
            row.pop("inference_finished_at")
        (directory / "jobs.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    result = analyze(tmp_path)
    parallel = next(row for row in result["comparisons"] if row["concurrency"] == 2)
    assert parallel["request_max_active_observed"] == 2
    assert parallel["max_active_observed"] is None
    assert result["hypothesis_result"] == "not_assessed"


def test_missing_requested_repetitions_and_shortened_output_need_more_evidence(tmp_path):
    complete_matrix(tmp_path, iterations=3)
    for directory in (tmp_path / "trials").glob("*-c2-*"):
        rows = [json.loads(line) for line in (directory / "jobs.jsonl").read_text().splitlines()]
        for row in rows:
            row["audio_duration_s"] = 15
        (directory / "jobs.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    result = analyze(tmp_path)
    assert result["evidence_status"] == "incomplete"
    assert result["hypothesis_result"] == "not_assessed"
    parallel = next(row for row in result["comparisons"] if row["concurrency"] == 2)
    assert parallel["material_shortening"]
    assert not parallel["eligible"]
