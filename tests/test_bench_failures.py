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
