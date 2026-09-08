import fcntl
import json
import subprocess
from types import SimpleNamespace

import pytest

from spark_bench import runtime
from spark_bench.runtime import BENCH_LABEL, DockerRunner, NodeLease


def runner_with_container(item):
    calls = []
    class Docker:
        def inspect(self, name):
            return item[0]
        def command(self, args):
            calls.append(args)
            item[0] = None
    lease = SimpleNamespace(factory=None, clean=True, worker=SimpleNamespace(docker=Docker()))
    return DockerRunner(lease), calls


def identity():
    return {'name': 'sparkbench-example', 'id': 'exact-id', 'run_id': 'run-one', 'job_id': 'job-one'}


def container():
    return {'Name': '/sparkbench-example', 'Id': 'exact-id',
            'Config': {'Labels': {BENCH_LABEL: 'run-one', BENCH_LABEL + '.job': 'job-one'}}}


def test_cleanup_removes_only_verified_exact_container():
    runner, calls = runner_with_container([container()])
    runner._remove(identity())
    assert calls == [['rm', '-f', 'exact-id']]
    assert runner.lease.clean


@pytest.mark.parametrize('field', ['id', 'name', 'run', 'job'])
def test_cleanup_never_removes_mismatched_owner(field):
    info = container()
    if field == 'id':
        info['Id'] = 'different'
    elif field == 'name':
        info['Name'] = '/production'
    else:
        info['Config']['Labels'][BENCH_LABEL if field == 'run' else BENCH_LABEL + '.job'] = 'other'
    runner, calls = runner_with_container([info])
    with pytest.raises(RuntimeError, match='ambiguous'):
        runner._remove(identity())
    assert not calls
    assert not runner.lease.clean


class LeaseWorker:
    def __init__(self):
        self.accepting = True
        self.calls = []
        self.docker = object()
        self.admission_override = {}

    def health(self):
        return {'runtime_ok': True, 'assets_ok': True, 'accepting': self.accepting,
                'generation': 'generation-one', 'probe': None, 'busy': False,
                'owned_containers': [], 'runtime_manifest': 'manifest-one'}

    def drain(self):
        self.calls.append('drain')
        self.accepting = False
        return self.health()

    def admit(self, generation):
        self.calls.append(('admit', generation))
        self.accepting = True
        return {**self.health(), **self.admission_override}


def make_lease(tmp_path, monkeypatch):
    state = tmp_path / 'node-state'
    state.mkdir()
    (state / 'node.json').write_text(json.dumps({'generation': 'generation-one'}))
    monkeypatch.setenv('SPARK_SERVE_NODE_STATE_DIR', str(state))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'image_id': 'sha256:test'}))
    worker = LeaseWorker()
    factory = SimpleNamespace(Worker=lambda _store: worker, JobStore=object,
                              MANIFEST_PATH=manifest,
                              assets_status=lambda *_args, **_kwargs: {'assets_ok': True, 'runtime_manifest': 'manifest-one'})
    return NodeLease(factory, tmp_path / 'results'), worker, state


@pytest.mark.parametrize('failure', ['container', 'process', 'transport'])
def test_gpu_audit_failure_leaves_production_drained(tmp_path, monkeypatch, failure):
    lease, worker, state = make_lease(tmp_path, monkeypatch)
    calls = []
    def command(args, **_kwargs):
        calls.append(args)
        if failure == 'transport':
            raise RuntimeError('docker inspection outcome unknown')
        if args[:2] == ['docker', 'ps']:
            output = 'foreign-container' if failure == 'container' else ''
        elif args[:2] == ['docker', 'inspect']:
            output = json.dumps([{'Id': 'foreign-container', 'HostConfig': {'DeviceRequests': [{'Count': -1}]}, 'Config': {}}])
        else:
            output = '123, python3, 12000\n'
        return subprocess.CompletedProcess(args, 0, output, '')
    monkeypatch.setattr(runtime, 'command', command)
    with pytest.raises(RuntimeError):
        lease.__enter__()
    assert worker.calls == ['drain']
    assert worker.accepting is False
    assert lease.clean is False
    assert lease.handle.closed
    assert json.loads((lease.root / 'lease-restored.json').read_text())['status'] == 'left_drained'
    assert all(args[1] not in ('stop', 'kill', 'rm', 'start') for args in calls)
    # The admission fence survives, but the failed benchmark releases its lock.
    with (state / 'node.lock').open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(handle, fcntl.LOCK_UN)


def test_renamed_immutable_container_id_is_not_treated_as_absent(tmp_path, monkeypatch):
    lease, worker, _state = make_lease(tmp_path, monkeypatch)
    monkeypatch.setattr(lease, '_audit_idle_gpu', lambda: None)
    lease.__enter__()
    original = container()
    original['Name'] = '/renamed-but-still-running'
    original['State'] = {'Running': True}
    calls = []
    class Docker:
        def inspect(self, name):
            return original if name == 'exact-id' else None
        def command(self, args, **_kwargs):
            calls.append(args)
            raise AssertionError('Must not mutate ambiguous container ownership')
    worker.docker = Docker()
    runner = DockerRunner(lease)
    with pytest.raises(RuntimeError, match='ambiguous'):
        runner._remove(identity())
    lease.__exit__(None, None, None)
    assert calls == []
    assert worker.calls == ['drain']
    assert lease.clean is False
    assert json.loads((lease.root / 'lease-restored.json').read_text())['status'] == 'left_drained'


@pytest.mark.parametrize('override', [{'accepting': False}, {'generation': 'new-generation'}, {'runtime_ok': False}])
def test_unhealthy_admission_receipt_is_not_reported_restored(tmp_path, monkeypatch, override):
    lease, worker, _state = make_lease(tmp_path, monkeypatch)
    monkeypatch.setattr(lease, '_audit_idle_gpu', lambda: None)
    lease.__enter__()
    worker.admission_override = override
    with pytest.raises(RuntimeError):
        lease.__exit__(None, None, None)
    assert lease.handle.closed
    receipt = lease.root / 'lease-restored.json'
    assert not receipt.exists() or json.loads(receipt.read_text())['status'] != 'restored'


class InferenceDocker:
    def __init__(self, *, log='', running=False):
        self.log = log
        self.running = running
        self.item = None
        self.calls = []

    def inspect(self, name):
        if self.item and name in (self.item['Id'], self.item['Name'].lstrip('/')):
            return self.item
        return None

    def command(self, args, **_kwargs):
        self.calls.append(args)
        output = ''
        if args[0] == 'create':
            labels = dict(value.split('=', 1) for i, value in enumerate(args) if i and args[i - 1] == '--label')
            self.item = {'Id': 'exact-runtime-id', 'Name': '/' + args[args.index('--name') + 1],
                         'Config': {'Labels': labels},
                         'State': {'Running': False, 'ExitCode': 1, 'OOMKilled': False}}
            output = self.item['Id']
        elif args[0] == 'start':
            self.item['State']['Running'] = self.running
        elif args[0] == 'logs':
            output = self.log
        elif args[0] == 'rm':
            assert args == ['rm', '-f', 'exact-runtime-id']
            self.item = None
        else:
            raise AssertionError(args)
        return subprocess.CompletedProcess(args, 0, output, '')


def fake_inference(tmp_path, *, log='', running=False, stop_event=None, timeout_s=10):
    vocal, instrumental = tmp_path / 'vocal.wav', tmp_path / 'instrumental.wav'
    vocal.write_bytes(b'vocal fixture; parser is not invoked')
    instrumental.write_bytes(b'instrumental fixture; parser is not invoked')
    docker = InferenceDocker(log=log, running=running)
    factory = SimpleNamespace(HF_HOME=tmp_path / 'hf', YUE_ROOT=tmp_path / 'yue', FACTORY_DIR=tmp_path,
                              _infer_argv=lambda *_args, **_kwargs: [],
                              _pick_audio=lambda *_args: pytest.fail('Invalid inference must not publish audio'))
    lease = SimpleNamespace(factory=factory, clean=True, runtime_manifest='manifest',
                            manifest={'image_id': 'sha256:test'}, worker=SimpleNamespace(docker=docker))
    job = {'job_id': 'test-job', 'run_id': 'run', 'workload_id': 'short', 'seed': 42,
           'concurrency': 1, 'dispatcher_count': 1, 'warmup': False,
           'vocal': str(vocal), 'instrumental': str(instrumental),
           'payload': {'lyrics': '[verse]\nCanción del niño', 'genre': 'folk'}}
    return DockerRunner(lease, stop_event=stop_event, timeout_s=timeout_s), docker, job


@pytest.mark.parametrize('message', ['torch.OutOfMemoryError: CUDA out of memory.',
                                    'RuntimeError: CUDA error: out of memory',
                                    'torch.OutOfMemoryError: allocation failed'])
def test_cuda_allocation_oom_is_detected_without_cgroup_oomkill(tmp_path, message):
    runner, docker, job = fake_inference(tmp_path, log=message)
    record = runner.run(job, tmp_path / 'job')
    assert record['cuda_oom'] is True
    assert record['oom_killed'] is False
    assert record['status'] != 'succeeded'
    assert record['exit_code'] == 1
    assert message in (tmp_path / 'job' / 'worker.log').read_text()
    assert not (tmp_path / 'job' / 'output.wav').exists()
    assert docker.item is None
    assert runner.lease.clean
    assert sum(args[0] == 'rm' for args in docker.calls) == 1


def test_timeout_uses_monotonic_clock_despite_wall_clock_rollback(tmp_path, monkeypatch):
    ticks = {'mono': 100.0, 'wall': 10000.0}
    class Stop:
        def is_set(self):
            return False
        def wait(self, _seconds):
            ticks['mono'] += 2
            ticks['wall'] -= 1000
            # Bound this test if the old wall-clock timeout bug is reintroduced.
            if ticks['mono'] > 120:
                raise AssertionError('Inference deadline did not follow monotonic time')
            return False
        def set(self):
            pass
    monkeypatch.setattr(runtime, 'time', SimpleNamespace(time=lambda: ticks['wall'], monotonic=lambda: ticks['mono']))
    runner, docker, job = fake_inference(tmp_path, running=True, stop_event=Stop(), timeout_s=3)
    record = runner.run(job, tmp_path / 'job')
    assert record['status'] == 'timeout'
    assert record['latency_s'] == 4
    assert record['finished_at'] < record['started_at']
    assert docker.item is None
