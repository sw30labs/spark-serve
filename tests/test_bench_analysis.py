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
             "workload_sha256": "owned-corpus", "synthetic": False, "status": "complete"}
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
    if telemetry is None:
        telemetry = [{"timestamp_s": start + 1, "cpu": {"total_percent": 30},
                      "memory": {"available_bytes": 64000000000},
                      "gpus": [{"temperature_gpu_c": 60, "utilization_gpu_percent": 75}]}]
    (directory / "telemetry.jsonl").write_text("\n".join(json.dumps(row) for row in telemetry))
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


@pytest.mark.parametrize("oom_flag", ["cuda_oom", "oom_killed"])
def test_explicit_oom_is_categorized_even_when_diagnostics_also_fail(tmp_path, oom_flag):
    complete_matrix(tmp_path)
    path = tmp_path / "trials/d1-c2-r0/jobs.jsonl"
    jobs = [json.loads(line) for line in path.read_text().splitlines()]
    jobs[0].update(status="failed_integrity", error="FAILED_INTEGRITY: partial diagnostics", **{oom_flag: True})
    path.write_text("\n".join(json.dumps(job) for job in jobs))
    result = analyze(tmp_path)
    parallel = next(row for row in result["comparisons"] if row["concurrency"] == 2)
    assert parallel["failures_by_category"] == {"out_of_memory": 1}
    assert parallel["successes"] == 7


@pytest.mark.parametrize("mode", ["missing_file", "empty", "invalid", "missing_required_counters", "failed_run"])
def test_run_and_resource_evidence_required_for_capacity_conclusion(tmp_path, mode):
    complete_matrix(tmp_path)
    telemetry = tmp_path / "trials/d1-c2-r0/telemetry.jsonl"
    if mode == "missing_file":
        telemetry.unlink()
    elif mode == "empty":
        telemetry.write_text("")
    elif mode == "invalid":
        telemetry.write_text('{"partial":')
    elif mode == "missing_required_counters":
        telemetry.write_text(json.dumps({"timestamp_s": 2201, "cpu": {"total_percent": 30}}))
    else:
        metadata(tmp_path, status="failed")
    result = analyze(tmp_path)
    assert result["evidence_status"] in {"incomplete", "underpowered"}
    assert result["hypothesis_result"] == "not_assessed"
    assert all(row["provisional"] for row in result["recommendations"])


def thermal_samples(values, *, start=2100, uuids=None, monotonic=None, headroom=None):
    rows = []
    for index, value in enumerate(values):
        gpu = {"uuid": uuids[index] if uuids is not None else "GPU-one",
               "temperature_gpu_c": 82 - index, "utilization_gpu_percent": 95,
               "temperature_tlimit_c": headroom[index] if headroom is not None else 10 - index,
               "clock_event_sw_thermal_us": value,
               "clock_event_sw_thermal_delta_us": 999999999,
               "clock_event_sw_thermal": "Not Active"}
        rows.append({"hostname": "spark-one", "timestamp_s": start + 1 + index,
                     "monotonic_s": monotonic[index] if monotonic is not None else 100 + index,
                     "cpu": {"total_percent": 30}, "memory": {"available_bytes": 64000000000},
                     "gpus": [gpu]})
    return rows


def test_throttle_deltas_exclude_lifetime_totals_and_between_trial_activity(tmp_path):
    metadata(tmp_path)
    trial(tmp_path, 1, 0, telemetry=thermal_samples([10_000_000, 11_000_000], headroom=[18, 7]))
    trial(tmp_path, 1, 1, telemetry=thermal_samples([90_000_000, 92_000_000], start=12100,
                                                 monotonic=[100, 102], headroom=[9, -2]))
    for iteration in range(2):
        trial(tmp_path, 2, iteration)
    result = analyze(tmp_path)
    assert result["evidence_status"] == "complete"  # Advanced counters stay optional.
    baseline = next(row for row in result["comparisons"] if row["concurrency"] == 1)
    counters = baseline["telemetry"]["throttle_counters"]
    observed = counters["clock_event_sw_thermal_us"]
    assert observed["observed_delta_us"] == 3_000_000
    assert observed["observed_delta_s"] == 3
    assert observed["valid_intervals"] == 2
    assert observed["availability"] == "available"
    assert counters["clock_event_sw_power_cap_us"]["observed_delta_us"] is None
    assert baseline["telemetry"]["metrics"]["gpu_temperature_tlimit_c"]["min"] == -2
    assert "T.Limit headroom minimum: -2.000" in (tmp_path / "report.md").read_text()
    # The sampled flags never report an active event, but cumulative deltas do.
    assert observed["observed_delta_us"] > 0


@pytest.mark.parametrize(("values", "options", "expected", "resets", "discontinuities"), [
    ([100, 105, 2, 7], {}, 10, 1, 0),
    ([100, None, 300, 305], {}, 5, 0, 0),
    ([100, 105, 700, 705], {"monotonic": [10, 11, 1, 2]}, 10, 0, 1),
    ([100, 500, 700, 705], {"uuids": ["GPU-a", "GPU-b", "GPU-a", "GPU-a"]}, 5, 0, 0),
    ([100, 500], {"uuids": [None, None]}, None, 0, 0),
    ([100], {}, None, 0, 0),
])
def test_counter_resets_missing_data_and_identity_changes_never_charge_lifetime(
    tmp_path, values, options, expected, resets, discontinuities,
):
    metadata(tmp_path)
    trial(tmp_path, 1, telemetry=thermal_samples(values, **options))
    telemetry = analyze(tmp_path)["comparisons"][0]["telemetry"]
    observed = telemetry["throttle_counters"]["clock_event_sw_thermal_us"]
    assert observed["observed_delta_us"] == expected
    assert observed["reset_intervals"] == resets
    assert observed["time_discontinuities"] == discontinuities
    assert observed["availability"] == ("unavailable" if expected is None else "partial")


def test_zero_counter_increment_is_measured_zero_not_unavailable(tmp_path):
    metadata(tmp_path)
    trial(tmp_path, 1, telemetry=thermal_samples([100, 100]))
    observed = analyze(tmp_path)["comparisons"][0]["telemetry"]["throttle_counters"]["clock_event_sw_thermal_us"]
    assert observed["observed_delta_us"] == 0
    assert observed["observed_delta_s"] == 0
    assert observed["availability"] == "available"


def test_decision_report_exposes_scoped_capacity_cost_and_unresolved_bottleneck(tmp_path):
    complete_matrix(tmp_path, node="spark-one", worker="spark-one-worker")
    for directory in (tmp_path / "trials").iterdir():
        record = json.loads((directory / "trial.json").read_text())
        telemetry = {"timestamp_s": record["started_at"] + 1, "cpu": {"total_percent": 30},
                     "memory": {"available_bytes": 64 * 2**30, "used_bytes": 32 * 2**30},
                     "gpus": [{"temperature_gpu_c": 80, "temperature_tlimit_c": 8,
                               "utilization_gpu_percent": 99, "power_draw_watts": 110,
                               "memory_used_bytes": 16 * 2**30}]}
        (directory / "telemetry.jsonl").write_text(json.dumps(telemetry))
    result = analyze(tmp_path)
    report = (tmp_path / "report.md").read_text()
    for label in ("BASELINE", "BEST TESTED", "SPEEDUP", "COST", "BOTTLENECK", "CONCLUSION",
                  "RECOMMENDED CAPACITY", "NEXT EXPERIMENT"):
        assert f"**{label}:**" in report
    assert "**BASELINE:** C1, 360.000 successful jobs/hour" in report
    assert "**BEST TESTED:** C2, 720.000 successful jobs/hour" in report
    assert "p50/p95 10.000/10.000 s" in report
    assert "16.000 GiB (available)/32.000 GiB (available)" in report
    assert "110.000 W (available)/110.000 W (available)" in report
    assert "core temperature peak 80.000 °C" in report
    assert "failures 0/8 ({})" in report
    assert "**BOTTLENECK:** unresolved" in report
    assert "even 99%, does not establish" in report
    assert "1 dispatcher(s), 2 concurrent job(s) per tested Spark" in report
    assert "Ceiling not found" in report
    assert "Extend above C2" in report
    assert result["recommendations"][0]["candidate_concurrency"] == 2


@pytest.mark.parametrize("metadata_updates", [{"synthetic": True}, {"status": "stopped"}, {"iterations": 3}])
def test_decision_report_withholds_supported_capacity_for_provisional_evidence(tmp_path, metadata_updates):
    complete_matrix(tmp_path, **metadata_updates)
    result = analyze(tmp_path)
    report = (tmp_path / "report.md").read_text()
    assert result["hypothesis_result"] == "not_assessed"
    assert "**CONCLUSION:** pending / not_assessed" in report
    assert "**RECOMMENDED CAPACITY:** pending; no supported capacity." in report
    assert "C2 with 1 dispatcher(s) is a provisional" in report
    assert "GPU power mean/max unavailable W (unavailable)" in report
    assert "1 dispatcher(s), 2 concurrent job(s) per tested Spark" not in report
    if metadata_updates.get("synthetic"):
        assert "harness candidate only" in report
        assert "hardware scaling and ceiling remain unassessed" in report
        assert "Ceiling not found" not in report


def test_csv_reports_marginal_throughput_per_added_slot_when_levels_skip(tmp_path):
    metadata(tmp_path, concurrency_levels=[1, 2, 4])
    for concurrency in (1, 2, 4):
        for iteration in range(2):
            trial(tmp_path, concurrency, iteration)
    result = analyze(tmp_path)
    highest = next(item for item in result["comparisons"] if item["concurrency"] == 4)
    assert highest["previous_concurrency"] == 2
    assert highest["marginal_jobs_per_hour"] == 720
    assert highest["marginal_jobs_per_hour_per_slot"] == 360
    with (tmp_path / "comparison.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    baseline = next(row for row in rows if row["concurrency"] == "1")
    upper = next(row for row in rows if row["concurrency"] == "4")
    assert float(upper["scaling_efficiency"]) == 1
    assert float(upper["marginal_speedup"]) == 2
    assert float(upper["marginal_jobs_per_hour_per_slot"]) == 360
    assert upper["previous_concurrency"] == "2"
    assert baseline["marginal_jobs_per_hour_per_slot"] == ""
    assert baseline["previous_concurrency"] == ""
    assert "Extend above C4" in (tmp_path / "report.md").read_text()


def test_fast_failing_level_is_observed_best_but_never_recommended(tmp_path):
    metadata(tmp_path)
    for iteration in range(2):
        trial(tmp_path, 1, iteration)
        trial(tmp_path, 2, iteration,
              statuses=["failed_integrity", "succeeded", "succeeded", "succeeded"])
    result = analyze(tmp_path)
    report = (tmp_path / "report.md").read_text()
    assert result["recommendations"][0]["candidate_concurrency"] == 1
    assert "**BEST TESTED:** C2, 540.000 successful jobs/hour (highest observed rate; does not meet acceptance criteria)" in report
    assert 'failures 2/8 ({"failed_integrity": 2})' in report
    assert "1 dispatcher(s), 1 concurrent job(s) per tested Spark" in report
    assert "Ceiling not found" not in report


def test_report_does_not_borrow_baselines_across_workloads_or_dispatchers(tmp_path):
    metadata(tmp_path)
    for iteration in range(2):
        trial(tmp_path, 1, iteration, workloads=("short",))
        trial(tmp_path, 2, iteration, workloads=("representative",))
        trial(tmp_path, 2, iteration, dispatchers=2, workloads=("short",))
    analyze(tmp_path)
    report = (tmp_path / "report.md").read_text()
    assert report.count("**BASELINE:**") == 3
    assert report.count("unavailable; no matching C1 baseline.") == 2
    assert report.count("**RECOMMENDED CAPACITY:** pending; no supported capacity.") == 3


def test_no_measured_trials_still_has_pending_decision_fields(tmp_path):
    metadata(tmp_path, status="stopped")
    analyze(tmp_path)
    report = (tmp_path / "report.md").read_text()
    assert "**BEST TESTED:** pending / not_assessed; unavailable; no measured throughput." in report
    assert "**RECOMMENDED CAPACITY:** pending; no supported capacity." in report


def test_positive_limit_counters_are_observations_not_a_bottleneck_verdict(tmp_path):
    metadata(tmp_path)
    for iteration in range(2):
        trial(tmp_path, 1, iteration)
        trial(tmp_path, 2, iteration,
              telemetry=thermal_samples([10_000_000, 11_000_000], start=2200 + iteration * 10000))
    analyze(tmp_path)
    report = (tmp_path / "report.md").read_text()
    assert "**BOTTLENECK:** unresolved" in report
    assert "Observed limit-event counters: sw_thermal 2.000000 s (available)" in report
    assert "these do not identify the dominant bottleneck" in report


def test_interior_best_does_not_claim_a_ceiling_or_request_automatic_extension(tmp_path):
    metadata(tmp_path, concurrency_levels=[1, 2, 4])
    for concurrency, duration in ((1, 40), (2, 20), (4, 25)):
        for iteration in range(2):
            trial(tmp_path, concurrency, iteration, duration=duration)
    result = analyze(tmp_path)
    report = (tmp_path / "report.md").read_text()
    assert result["recommendations"][0]["candidate_concurrency"] == 2
    assert "**BEST TESTED:** C2" in report
    assert "Ceiling not found" not in report
    assert "Extend above C4" not in report
    assert "Repeat the candidate and neighboring levels" in report


def occupancy(windows, concurrency):
    return _analysis._inference_occupancy(
        [{"start": start, "end": end} for start, end in windows], concurrency, "start", "end")


def test_occupancy_reveals_short_job_tail_despite_reaching_peak_concurrency():
    observed = occupancy([(0, 10), (0, 10), (0, 30), (0, 30)], 4)
    assert observed["availability"] == "available"
    assert observed["dwell_s_by_active_jobs"] == {"2": 20, "4": 10}
    assert observed["observed_window_s"] == 30
    assert observed["active_job_seconds"] == 80
    assert observed["mean_active_jobs"] == pytest.approx(8 / 3)
    assert observed["underfilled_fraction"] == pytest.approx(2 / 3)
    assert observed["full_fraction"] == pytest.approx(1 / 3)


def test_occupancy_handles_shared_boundaries_and_interior_idle_gaps():
    adjacent = occupancy([(0, 5), (5, 10)], 1)
    assert adjacent["dwell_s_by_active_jobs"] == {"1": 10}
    assert adjacent["underfilled_fraction"] == 0
    gaps = occupancy([(0, 5), (10, 15)], 2)
    assert gaps["dwell_s_by_active_jobs"] == {"0": 5, "1": 10}
    assert gaps["mean_active_jobs"] == pytest.approx(2 / 3)
    assert gaps["underfilled_fraction"] == 1


def test_occupancy_combines_time_weighted_windows_without_cross_trial_gaps():
    long = occupancy([(0, 10), (0, 10), (0, 30), (0, 30)], 4)
    short = occupancy([(1000, 1010)] * 4, 4)
    combined = _analysis._combine_occupancy([{"inference_occupancy": long}, {"inference_occupancy": short}], 4)
    assert combined["observed_window_s"] == 40
    assert combined["mean_active_jobs"] == 3  # Weighted by30s+10s, not a mean of two trial means.
    assert combined["underfilled_fraction"] == .5
    assert combined["available_trials"] == combined["trial_count"] == 2
    assert combined["dwell_s_by_active_jobs"] == {"2": 20, "4": 20}


def test_censored_occupancy_is_unavailable_and_partial_aggregate_labeled():
    incomplete = occupancy([(0, 10), (0, None)], 2)
    assert incomplete["availability"] == "unavailable"
    assert incomplete["mean_active_jobs"] is None
    assert occupancy([(5, 4)], 1)["observed_window_s"] is None
    combined = _analysis._combine_occupancy([
        {"inference_occupancy": occupancy([(0, 10)] * 2, 2)},
        {"inference_occupancy": incomplete}], 2)
    assert combined["availability"] == "partial"
    assert combined["available_trials"] == 1
    assert combined["observed_window_s"] == 10


def test_finite_batch_flag_and_dwell_metrics_preserve_original_rates(tmp_path):
    metadata(tmp_path, concurrency_levels=[1, 4])
    for concurrency in (1, 4):
        for iteration in range(2):
            trial(tmp_path, concurrency, iteration)
    result = analyze(tmp_path)
    baseline, high = sorted(result["comparisons"], key=lambda row: row["concurrency"])
    assert baseline["jobs_per_hour"] == 360
    assert high["jobs_per_hour"] == 1440
    assert high["speedup_vs_c1"] == 4
    assert high["inference_mean_active_jobs"] == 4
    assert high["inference_underfilled_fraction"] == 0
    assert high["inference_window_s"] == 16
    assert high["batch_waves_min"] == 1
    assert high["multiwave_followup_recommended"] is True
    assert baseline["multiwave_followup_recommended"] is False
    assert result["evidence_status"] == "complete"
    assert result["hypothesis_result"] == "h0_falsified"
    assert result["recommendations"][0]["candidate_concurrency"] == 4
    assert all("inference_occupancy" in row for row in result["trials"])
    report = (tmp_path / "report.md").read_text()
    assert "finite-batch throughput" in report
    assert "**LOAD SHAPE:** C4: mean active inference containers 4.000" in report
    assert "at least 16 same-mix jobs per trial" in report
    assert "Small-batch candidate; sustained queue capacity requires" in report
    with (tmp_path / "comparison.csv").open() as handle:
        row = next(row for row in csv.DictReader(handle) if row["concurrency"] == "4")
    assert float(row["inference_underfilled_fraction"]) == 0
    assert float(row["inference_mean_active_jobs"]) == 4


def test_c1_winner_still_requires_multiwave_check_when_higher_levels_had_short_batches(tmp_path):
    metadata(tmp_path, concurrency_levels=[1, 4])
    for iteration in range(2):
        trial(tmp_path, 1, iteration)
        trial(tmp_path, 4, iteration, duration=50)
    result = analyze(tmp_path)
    assert result["hypothesis_result"] == "h0_not_rejected"
    assert result["recommendations"][0]["candidate_concurrency"] == 1
    report = (tmp_path / "report.md").read_text()
    assert "Small batches at C=[4]" in report
    assert "at least 16 same-mix jobs per trial" in report
    assert "Small-batch candidate; sustained queue capacity requires" in report
