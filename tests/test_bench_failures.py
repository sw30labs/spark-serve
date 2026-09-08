"""Controller failure injection: cancel GPU work before waiting for threads."""

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from spark_bench import __main__, analysis, experiment, telemetry
from spark_bench.experiment import execute_trial, jobs_for_trial, load_workload


class NoTelemetry:
    def __init__(self, *_args, **_kwargs):
        pass
    def start(self):
        return self
    def stop(self):
        pass
    def raise_if_failed(self):
        pass


def jobs(count=3, concurrency=2):
    corpus = load_workload(Path(__file__).parent / 'workloads/yue-standard.json', synthetic=True)
    return jobs_for_trial(corpus, count=count, seed=42, run_id='test', trial_id='c2', concurrency=concurrency)


class BlockedRunner:
    def __init__(self, stop, *, first_completes=True):
        self.stop = stop
        self.first_completes = first_completes
        self.second_started = threading.Event()
        self.started = threading.Event()
        self.saw_stop = []

    def run(self, job, root):
        self.started.set()
        if job['seed'] == 42 and self.first_completes:
            assert self.second_started.wait(2), 'Second concurrent job never started'
            cancelled = False
        else:
            self.second_started.set()
            cancelled = self.stop.wait(1)
            self.saw_stop.append(cancelled)
        result = {key: job[key] for key in ('job_id', 'workload_id', 'seed', 'concurrency', 'dispatcher_count', 'warmup')}
        result.update(status='cancelled' if cancelled else 'succeeded', retry_count=0)
        root.mkdir(parents=True)
        (root / 'job.json').write_text(json.dumps(result))
        return result


def test_job_fsync_failure_cancels_before_executor_waits(tmp_path, monkeypatch):
    monkeypatch.setattr(telemetry, 'TelemetryCollector', NoTelemetry)
    stop = threading.Event()
    runner = BlockedRunner(stop)
    def fail_append(*_args, **_kwargs):
        raise OSError('fsync failed: no space left on device')
    monkeypatch.setattr(experiment, 'append_json', fail_append)
    with pytest.raises(OSError, match='fsync failed'):
        execute_trial(tmp_path / 'trial', jobs(), runner, concurrency=2, stop_event=stop,
                      synthetic=True, telemetry_interval=.01)
    assert stop.is_set()
    assert runner.saw_stop == [True], 'Controller waited for worker completion before signalling cancellation'
    trial = json.loads((tmp_path / 'trial' / 'trial.json').read_text())
    assert trial['status'] == 'stopped'
    saved = list((tmp_path / 'trial' / 'jobs').glob('*/job.json'))
    assert len(saved) == 2
    assert sorted(json.loads(path.read_text())['status'] for path in saved) == ['cancelled', 'succeeded']


def test_background_telemetry_failure_stops_inflight_generation(tmp_path, monkeypatch):
    stop = threading.Event()
    runner = BlockedRunner(stop, first_completes=False)
    class FailedTelemetry(NoTelemetry):
        def raise_if_failed(self):
            if runner.second_started.is_set():
                raise RuntimeError('telemetry collection failed')
    monkeypatch.setattr(telemetry, 'TelemetryCollector', FailedTelemetry)
    with pytest.raises(RuntimeError, match='telemetry collection failed'):
        execute_trial(tmp_path / 'trial', jobs(), runner, concurrency=2, stop_event=stop,
                      synthetic=True, telemetry_interval=.01)
    assert stop.is_set()
    assert runner.saw_stop and all(runner.saw_stop)
    assert json.loads((tmp_path / 'trial' / 'trial.json').read_text())['status'] == 'stopped'


def test_cuda_oom_halts_sweep_and_records_unlaunched_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(telemetry, 'TelemetryCollector', NoTelemetry)
    stop = threading.Event()
    launched = []
    class OOMRunner:
        def run(self, job, _root):
            launched.append(job['job_id'])
            result = {key: job[key] for key in ('job_id', 'workload_id', 'seed', 'concurrency', 'dispatcher_count', 'warmup')}
            return {**result, 'status': 'failed', 'cuda_oom': True, 'oom_killed': False,
                    'error': 'CUDA allocation failed', 'retry_count': 0}
    trial = execute_trial(tmp_path / 'trial', jobs(concurrency=1), OOMRunner(), concurrency=1,
                          stop_event=stop, synthetic=True, telemetry_interval=.01)
    assert stop.is_set()
    assert len(launched) == 1
    assert trial['status'] == 'stopped'
    assert 'OOM' in trial['stop_reason']
    results = [json.loads(line) for line in (tmp_path / 'trial' / 'jobs.jsonl').read_text().splitlines()]
    assert [row['status'] for row in results] == ['failed', 'cancelled', 'cancelled']


@pytest.mark.parametrize('headroom,should_stop', [(None, False), (6, False), (5, True), (-1, True)])
def test_thermal_headroom_stops_below_absolute_temperature_limit(tmp_path, monkeypatch, headroom, should_stop):
    monkeypatch.setattr(telemetry, 'TelemetryCollector', NoTelemetry)
    sample = {'memory': {'available_bytes': 40 * 1024**3, 'swap_used_bytes': 0},
              'gpus': [{'temperature_gpu_c': 76, 'temperature_tlimit_c': headroom}]}
    monkeypatch.setattr(telemetry, 'TelemetrySampler', lambda: SimpleNamespace(sample=lambda: sample))
    stop = threading.Event()
    launched = []
    class Runner:
        def run(self, job, _root):
            launched.append(job['job_id'])
            return {**job, 'status': 'succeeded', 'retry_count': 0}
    trial = execute_trial(tmp_path / 'trial', jobs(count=1, concurrency=1), Runner(), concurrency=1,
                          stop_event=stop, max_temperature_c=90, min_temperature_margin_c=5)
    assert stop.is_set() is should_stop
    assert bool(launched) is not should_stop
    assert trial['status'] == ('stopped' if should_stop else 'complete')
    receipts = [json.loads(line) for line in (tmp_path / 'trial' / 'guard-samples.jsonl').read_text().splitlines()]
    assert receipts[0]['sample'] == sample
    assert receipts[0]['baseline_swap_used_bytes'] == 0
    assert receipts[0]['thresholds'] == {'max_temperature_c': 90, 'min_temperature_margin_c': 5,
                                        'min_free_gb': 12, 'max_swap_growth_gb': 1}
    assert receipts[0]['decision'] == ('stop' if should_stop else 'continue')
    assert receipts[0]['evaluated_monotonic'] >= trial['started_monotonic']
    if should_stop:
        assert 'thermal headroom' in trial['stop_reason']
        assert trial['guard_trigger'] == receipts[0]
        assert trial['guard_trigger']['reason'] == trial['stop_reason']
    else:
        assert 'guard_trigger' not in trial


@pytest.mark.parametrize('triggering_headroom', [None, -1])
def test_guard_fsync_failure_stops_inflight_and_preserves_exact_sample(tmp_path, monkeypatch, triggering_headroom):
    monkeypatch.setattr(telemetry, 'TelemetryCollector', NoTelemetry)
    stop = threading.Event()
    runner = BlockedRunner(stop, first_completes=False)
    safe_sample = {'timestamp_wall': '2026-09-08T22:00:00+00:00', 'monotonic_s': 100,
                   'memory': {'available_bytes': 40 * 1024**3, 'swap_used_bytes': 256},
                   'gpus': [{'temperature_gpu_c': 76, 'temperature_tlimit_c': 6}]}
    next_sample = {**safe_sample, 'monotonic_s': 102,
                   'gpus': [{'temperature_gpu_c': 81, 'temperature_tlimit_c': triggering_headroom}]}
    samples = iter([safe_sample, next_sample])
    monkeypatch.setattr(telemetry, 'TelemetrySampler', lambda: SimpleNamespace(sample=lambda: next(samples)))
    original_append = experiment.append_json
    def fail_second_guard(path, value):
        if path.name == 'guard-samples.jsonl' and value['sequence'] == 2:
            assert runner.second_started.is_set()
            assert not stop.is_set(), 'Guard signalled cancellation before attempting to persist the decision'
            raise OSError('guard fsync failed')
        original_append(path, value)
    monkeypatch.setattr(experiment, 'append_json', fail_second_guard)
    with pytest.raises(OSError, match='guard fsync failed'):
        execute_trial(tmp_path / 'trial', jobs(count=1, concurrency=1), runner, concurrency=1,
                      stop_event=stop, telemetry_interval=.01, max_temperature_c=90,
                      min_temperature_margin_c=0)
    assert stop.is_set()
    assert runner.saw_stop == [True]
    trial = json.loads((tmp_path / 'trial' / 'trial.json').read_text())
    assert trial['status'] == 'stopped'
    assert trial['guard_persistence_error'] == 'OSError'
    assert trial['guard_persistence_failed_sample']['sample'] == next_sample
    receipts = [json.loads(line) for line in (tmp_path / 'trial' / 'guard-samples.jsonl').read_text().splitlines()]
    assert len(receipts) == 1
    assert receipts[0]['sample'] == safe_sample
    if triggering_headroom is not None:
        assert trial['guard_trigger']['sample'] == next_sample
        assert trial['guard_trigger']['decision'] == 'stop'
        assert trial['guard_trigger']['baseline_swap_used_bytes'] == 256
        assert 'thermal headroom' in trial['stop_reason']
    else:
        assert 'guard_trigger' not in trial
        assert trial['stop_reason'] == 'Guard evidence persistence failed'


def test_failed_production_restoration_exits_nonzero_and_preserves_metadata(tmp_path, monkeypatch):
    corpus_path = Path(__file__).parent / 'workloads/yue-standard.json'
    corpus = load_workload(corpus_path, synthetic=True)
    output = tmp_path / 'experiment'
    factory = SimpleNamespace(WORKER_ID='spark-test', SOURCE_REV='source', STAGE1_REV='stage1',
                              STAGE2_REV='stage2', CODEC_REV='codec')
    class FailedRestoration:
        runtime_manifest = 'manifest'
        manifest = {'image_id': 'sha256:test'}
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            raise RuntimeError('Factory did not confirm restored admission for the original generation')
    def freeze(workload, root, **_kwargs):
        (root / 'inputs').mkdir()
        (root / 'inputs' / 'workload.json').write_text(json.dumps(workload))
        return workload
    monkeypatch.setattr(experiment, 'load_workload', lambda *_args, **_kwargs: corpus)
    monkeypatch.setattr(experiment, 'freeze_workload', freeze)
    monkeypatch.setattr(experiment, 'load_factory', lambda *_args: factory)
    monkeypatch.setattr(experiment, 'NodeLease', lambda *_args, **_kwargs: FailedRestoration())
    monkeypatch.setattr(experiment, 'DockerRunner', lambda *_args, **_kwargs: object())
    monkeypatch.setattr(experiment, 'execute_trial', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiment, 'sys_platform', lambda: 'linux')
    monkeypatch.setattr(analysis, 'analyze', lambda *_args: None)
    monkeypatch.setattr('sys.argv', ['spark_bench', 'concurrency', '--workload', str(corpus_path),
                                   '--output', str(output), '--concurrency', '1', '--jobs-per-level', '1',
                                   '--iterations', '1', '--warmup', '0', '--cooldown', '0'])
    with pytest.raises(SystemExit) as stopped:
        __main__.main()
    assert stopped.value.code == 2
    metadata = json.loads((output / 'run-metadata.json').read_text())
    assert metadata['status'] == 'failed'
    assert 'original generation' in metadata['restoration_error']
    assert metadata['finished_at'] >= metadata['started_at']
