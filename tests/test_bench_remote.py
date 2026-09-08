import fcntl
import io
import json
import shlex
import subprocess
import tarfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from spark_bench import remote
from spark_serve_controller import Controller


def bundle(entries=None):
    entries = entries if entries is not None else {
        'run-metadata.json': json.dumps({'synthetic': False, 'status': 'complete', 'concurrency_levels': [1, 2, 3]}),
        'summary.json': json.dumps({'evidence_status': 'underpowered', 'hypothesis_result': 'not_assessed'}),
        'comparison.csv': 'concurrency,jobs_per_hour\n1,1\n',
        'report.md': 'Incomplete benchmark evidence.\n',
        'trials/c1/trial.json': json.dumps({'trial_id': 'c1', 'status': 'complete'}),
        'trials/c1/jobs.jsonl': json.dumps({'job_id': 'one', 'status': 'succeeded'}) + '\n',
        'trials/c1/telemetry.jsonl': json.dumps({'memory': {'available_bytes': 123}}) + '\n',
    }
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode='w:gz') as archive:
        for name, content in entries.items():
            member = tarfile.TarInfo(name)
            raw = content.encode()
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    return data.getvalue()


def lock_available(state):
    with (state / 'controller.lock').open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(handle, fcntl.LOCK_UN)
        return True


@pytest.fixture
def remote_case(tmp_path, monkeypatch):
    config = tmp_path / 'models.toml'
    config.write_text('[cluster]\nhead="spark-a"\nworker="spark-b"\nlan_url="http://spark-a.lan:8000"\n'
                      'ssh_opts=["-o","BatchMode=yes"]\n[yue]\nhead_url="http://spark-a.lan:8011"\n'
                      'worker_url="http://spark-b.lan:8011"\n')
    assets = {}
    for field in ('attestation', 'vocal', 'instrumental'):
        path = tmp_path / (field + '.asset')
        path.write_bytes((field + ': explicitly authorized fixture').encode())
        assets[field] = str(path)
    workload = {'version': 1, 'rights_basis': 'operator_owned', **assets,
                'workloads': [{'id': 'short', 'lyrics': '[verse]\nCanción del niño',
                               'genre': 'folk', 'run_n_segments': 2, 'max_new_tokens': 3000}]}
    monkeypatch.setattr(remote, 'load_workload', lambda _path: json.loads(json.dumps(workload)))
    repo = tmp_path / 'source'
    package = repo / 'spark_bench'
    package.mkdir(parents=True)
    for filename in ('__init__.py', 'remote.py', 'runtime.py'):
        (package / filename).write_text('# owned benchmark module\n')
    (repo / 'spark_serve_controller.py').write_text('# ownership controller\n')
    (repo / 'models.toml').write_text('private configuration, never deployed')
    (package / '.env').write_text('private environment, never deployed')
    monkeypatch.setattr(remote, '__file__', str(package / 'remote.py'))
    state = tmp_path / 'controller-state'
    state.mkdir()
    (state / 'controller.json').write_text(json.dumps({'version': 1, 'mode': 'yue', 'phase': 'ready'}))
    monkeypatch.setenv('SPARK_SERVE_STATE_DIR', str(state))
    args = SimpleNamespace(config=config, ssh_host='spark-a', workload=tmp_path / 'workload.json',
        output=tmp_path / 'output', endpoint='http://ignored.example:9999', concurrency='1,2,3',
        jobs_per_level=6, iterations=2, warmup=1, cooldown=5, timeout=90, wait_idle=120, seed=42,
        telemetry_interval=1, factory_root=Path('~/.local/share/artist-twin/yue-factory'),
        service='yue-icl.service', min_free_gb=12, max_temperature_c=85, max_swap_growth_gb=1,
        worker='worker-a', synthetic=False)
    return SimpleNamespace(args=args, state=state, package=package, assets=assets, workload=workload)


def fake_ssh(case, monkeypatch, *, launch_code=0, failure=None, archive=None, remote_path=None):
    calls = []

    def run(command, **kwargs):
        assert not lock_available(case.state), 'Real Mac controller lock must span every SSH operation'
        assert command[:4] == ['ssh', '-o', 'BatchMode=yes', 'spark-a']
        tokens = shlex.split(command[4])
        calls.append((tokens, kwargs))
        phase = 'deploy' if 'input' in kwargs else 'launch' if tokens[0] == 'env' else 'collect'
        if failure == phase:
            return subprocess.CompletedProcess(command, 9, b'', b'fixture transport failure')
        if phase == 'deploy':
            assert tokens[:2] == ['python3', '-c']
            path = remote_path or '/home/bench/' + tokens[-1]
            return subprocess.CompletedProcess(command, 0, (path + '\n').encode(), b'')
        if phase == 'launch':
            kwargs['stdout'].write(b'Fixture benchmark driver output\n')
            return subprocess.CompletedProcess(command, launch_code)
        return subprocess.CompletedProcess(command, 0, bundle() if archive is None else archive, b'')

    monkeypatch.setattr(remote.subprocess, 'run', run)
    return calls


def test_deploys_only_owned_source_and_explicit_references_under_real_lock(remote_case, monkeypatch):
    case = remote_case
    calls = fake_ssh(case, monkeypatch)
    output = remote.remote_run(case.args)
    assert output == case.args.output.resolve()
    assert len(calls) == 3
    assert lock_available(case.state)
    with tarfile.open(fileobj=io.BytesIO(calls[0][1]['input']), mode='r:gz') as archive:
        assert set(archive.getnames()) == {'src/spark_bench/__init__.py', 'src/spark_bench/remote.py',
            'src/spark_bench/runtime.py', 'src/spark_serve_controller.py', 'input/attestation.txt',
            'input/vocal.wav', 'input/instrumental.wav', 'input/workload.json'}
        assert all(member.isfile() for member in archive.getmembers())
        workload = json.load(archive.extractfile('input/workload.json'))
        for field, filename in (('attestation', 'attestation.txt'), ('vocal', 'vocal.wav'), ('instrumental', 'instrumental.wav')):
            assert workload[field] == filename
            assert archive.extractfile('input/' + filename).read() == Path(case.assets[field]).read_bytes()
        assert workload['workloads'] == case.workload['workloads']
    launch = calls[1][0]
    assert launch[0] == 'env'
    assert launch[2:6] == ['python3', '-m', 'spark_bench', 'concurrency']
    options = dict(zip(launch[6::2], launch[7::2], strict=True))
    assert options['--endpoint'] == 'http://spark-a.lan:8011'
    for name in ('concurrency', 'jobs_per_level', 'iterations', 'warmup', 'cooldown', 'timeout',
                 'wait_idle', 'seed', 'telemetry_interval', 'factory_root', 'service', 'min_free_gb',
                 'max_temperature_c', 'max_swap_growth_gb', 'worker'):
        assert options['--' + name.replace('_', '-')] == str(getattr(case.args, name))
    assert options['--output'].startswith('/home/bench/.local/share/spark-serve/benchmarks/bench-')
    assert options['--output'].endswith('/results')
    assert '--synthetic' not in launch
    assert not any(token in {'ip', 'nmcli', 'ethtool', 'systemctl', 'docker'} for tokens, _ in calls for token in tokens)
    receipt = json.loads((output / 'remote-run.json').read_text())
    assert receipt['status'] == 'complete'
    assert receipt['network_changed'] is False
    assert (output / 'results/trials/c1/jobs.jsonl').is_file()
    assert (output / 'results/trials/c1/telemetry.jsonl').is_file()


def test_non_source_entries_are_not_archived(remote_case, monkeypatch):
    case = remote_case
    directory = case.package / 'not-a-module.py'
    directory.mkdir()
    (directory / 'private.env').write_text('not source')
    (case.package / 'outside.py').symlink_to(case.package / '.env')
    calls = fake_ssh(case, monkeypatch)
    remote.remote_run(case.args)
    with tarfile.open(fileobj=io.BytesIO(calls[0][1]['input']), mode='r:gz') as archive:
        assert all(member.isfile() for member in archive.getmembers())
        assert not any('not-a-module' in member.name or 'outside.py' in member.name for member in archive.getmembers())


@pytest.mark.parametrize('host', ['unconfigured-host', '-oProxyCommand=unexpected'])
def test_rejects_unconfigured_host_without_ssh_or_output(remote_case, monkeypatch, host):
    remote_case.args.ssh_host = host
    monkeypatch.setattr(remote.subprocess, 'run', lambda *_args, **_kwargs: pytest.fail('No SSH allowed'))
    with pytest.raises(ValueError, match='configured'):
        remote.remote_run(remote_case.args)
    assert not remote_case.args.output.exists()


def test_rejects_credential_bearing_endpoint_before_deployment(remote_case, monkeypatch):
    config = remote_case.args.config
    config.write_text(config.read_text().replace('http://spark-a.lan:8011', 'http://user:credential@spark-a.lan:8011'))
    monkeypatch.setattr(remote.subprocess, 'run', lambda *_args, **_kwargs: pytest.fail('No SSH allowed'))
    with pytest.raises(RuntimeError, match='without credentials'):
        remote.remote_run(remote_case.args)
    assert not remote_case.args.output.exists()


@pytest.mark.parametrize('failure', ['deploy', 'launch', 'collect'])
def test_transport_or_remote_failure_releases_lock_and_cannot_return_success(remote_case, monkeypatch, failure):
    case = remote_case
    calls = fake_ssh(case, monkeypatch, failure=failure)
    with pytest.raises(RuntimeError):
        remote.remote_run(case.args)
    assert lock_available(case.state)
    receipt = json.loads((case.args.output / 'remote-run.json').read_text())
    assert receipt['status'] != 'complete'
    assert calls


@pytest.mark.parametrize('path', ['/etc', '../outside', '/home/bench/.local/share/spark-serve/benchmarks/other',
                                 '/home/bench\n/extra'])
def test_unexpected_deployment_path_is_rejected_before_launch(remote_case, monkeypatch, path):
    calls = fake_ssh(remote_case, monkeypatch, remote_path=path)
    with pytest.raises(RuntimeError):
        remote.remote_run(remote_case.args)
    assert len(calls) == 1
    assert lock_available(remote_case.state)


def test_empty_success_archive_does_not_become_successful_collection(remote_case, monkeypatch):
    fake_ssh(remote_case, monkeypatch, archive=bundle({}))
    with pytest.raises(RuntimeError):
        remote.remote_run(remote_case.args)
    assert lock_available(remote_case.state)
    assert json.loads((remote_case.args.output / 'remote-run.json').read_text())['status'] != 'complete'


def test_controller_context_closure_failure_is_not_reported_complete(remote_case, monkeypatch):
    class BrokenCloseController(Controller):
        @contextmanager
        def lock(self):
            with super().lock():
                yield
                raise RuntimeError('controller closure failed')

    monkeypatch.setattr(remote, 'Controller', BrokenCloseController)
    fake_ssh(remote_case, monkeypatch)
    with pytest.raises(RuntimeError, match='closure failed'):
        remote.remote_run(remote_case.args)
    assert lock_available(remote_case.state)
    assert json.loads((remote_case.args.output / 'remote-run.json').read_text())['status'] != 'complete'


@pytest.mark.parametrize('kind', ['traversal', 'absolute', 'symlink', 'hardlink', 'fifo'])
def test_raw_archive_rejects_unsafe_entries_before_any_extraction(tmp_path, kind):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode='w:gz') as archive:
        harmless = tarfile.TarInfo('harmless.json')
        harmless.size = 2
        archive.addfile(harmless, io.BytesIO(b'{}'))
        bad = tarfile.TarInfo('../outside.json' if kind == 'traversal' else
                              str(tmp_path / 'outside.json') if kind == 'absolute' else 'bad-entry')
        if kind in {'symlink', 'hardlink'}:
            bad.type = tarfile.SYMTYPE if kind == 'symlink' else tarfile.LNKTYPE
            bad.linkname = '../outside.json'
        elif kind == 'fifo':
            bad.type = tarfile.FIFOTYPE
        archive.addfile(bad)
    with tarfile.open(fileobj=io.BytesIO(data.getvalue()), mode='r:gz') as archive, pytest.raises(RuntimeError, match='Unsafe'):
        remote.safe_extract(archive, tmp_path / 'results')
    assert not (tmp_path / 'outside.json').exists()
    assert not (tmp_path / 'results/harmless.json').exists()
