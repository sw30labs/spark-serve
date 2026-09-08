"""Node-local experimental dispatch, fenced from production workload switching.

Each request executes the deployed YuE runtime in its own GPU container, exactly
the production model/process topology. No production job database is modified
except closing/restoring its existing admission gate.
"""
from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import shlex
import socket
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

from .common import digest_file, write_json

BENCH_LABEL = 'io.spark-serve.benchmark'


def command(argv, **kwargs):
    return subprocess.run(argv, text=True, capture_output=True, check=True, timeout=30, **kwargs)


def load_factory(root: Path, service: str):
    """Match the service's environment without recording it or its secrets."""
    result = command(['systemctl', '--user', 'show', service, '--property=Environment', '--value'])
    for item in shlex.split(result.stdout):
        key, separator, value = item.partition('=')
        if separator and (key.startswith('YUE_') or key == 'HF_HOME'):
            os.environ[key] = value
    root = root.expanduser().resolve()
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('yue_factory', root / 'yue_factory.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.API_VERSION != 2 or module.MOCK or module.RUNTIME != 'docker':
        raise RuntimeError('Benchmark requires the live protocol-v2 Docker factory')
    return module


class NodeLease:
    """Hold the real per-node fencing lock until all benchmark containers are gone.

    A killed controller cannot silently free an occupied GPU: uniquely labeled
    containers are visible to normal Spark Serve foreign-workload checks. A
    recovery command must inspect and clean only the saved benchmark identities.
    """
    def __init__(self, factory, root: Path, *, wait_idle_s=14400, stop_event=None):
        self.factory, self.root = factory, root
        self.wait_idle_s = wait_idle_s
        self.stop_event = stop_event or threading.Event()
        self.worker = factory.Worker(factory.JobStore())
        self.handle = None
        self.restore = False
        self.clean = True

    def __enter__(self):
        state = Path(os.environ.get('SPARK_SERVE_NODE_STATE_DIR', '~/.local/state/spark-serve')).expanduser()
        self.handle = (state / 'node.lock').open('a+')
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.handle.close()
            raise RuntimeError('A Spark Serve workload operation owns the node; retry later') from None
        try:
            self.generation = json.loads((state / 'node.json').read_text())['generation']
            before = self.worker.health()
            if not before.get('runtime_ok') or not before.get('assets_ok'):
                raise RuntimeError('Production YuE runtime is not admitted/validated')
            if before.get('generation') != self.generation:
                raise RuntimeError('Node fencing generation does not match the factory')
            if before.get('probe'):
                raise RuntimeError('A validation probe is active; retry after it finishes')
            if not before.get('accepting'):
                raise RuntimeError('Factory is already drained; reconcile ownership before benchmarking')
            self.restore = True
            write_json(self.root / 'lease.json', {'status': 'draining', 'node': socket.gethostname(),
                       'generation': self.generation, 'original_accepting': True,
                       'production_job': (before.get('active_job') or {}).get('job_id'),
                       'started_at': time.time(), 'pid': os.getpid()})
            self.worker.drain()
            deadline = time.monotonic() + self.wait_idle_s
            while True:
                health = self.worker.health()
                if not health.get('busy') and not health.get('owned_containers'):
                    break
                if self.stop_event.wait(2) or time.monotonic() >= deadline:
                    # Active production jobs cannot be re-admitted; keep their
                    # existing ownership and report the explicit drained state.
                    self.restore = False
                    raise RuntimeError('Production job did not drain within the wait; it was not cancelled')
            try:
                self._audit_idle_gpu()
            except BaseException:
                self.clean = False
                raise
            assets = self.factory.assets_status(self.worker.docker, verify_hashes=True)
            if not assets.get('assets_ok') or assets.get('runtime_manifest') != before.get('runtime_manifest'):
                raise RuntimeError('Pinned runtime assets changed during drain')
            self.manifest = json.loads(self.factory.MANIFEST_PATH.read_text())
            self.runtime_manifest = assets['runtime_manifest']
            write_json(self.root / 'lease.json', {'status': 'held', 'node': socket.gethostname(),
                       'generation': self.generation, 'original_accepting': True,
                       'runtime_manifest': self.runtime_manifest, 'started_at': time.time(), 'pid': os.getpid()})
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def _audit_idle_gpu(self):
        from spark_serve_controller import classify_gpu_processes
        ids = command(['docker', 'ps', '-q']).stdout.split()
        if ids:
            containers = json.loads(command(['docker', 'inspect', *ids]).stdout)
            for item in containers:
                host = item.get('HostConfig', {})
                env = item.get('Config', {}).get('Env') or []
                if (host.get('DeviceRequests') or host.get('Devices') or host.get('Runtime') == 'nvidia'
                    or host.get('Privileged') or any(x.startswith('NVIDIA_VISIBLE_DEVICES=')
                        and x.split('=', 1)[1] not in ('', 'none', 'void') for x in env)):
                    raise RuntimeError('Another GPU-capable container remains: ' + item['Id'])
        compute = command(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                           '--format=csv,noheader,nounits']).stdout
        audit = classify_gpu_processes(compute)
        write_json(self.root / 'gpu-idle-audit.json', audit)
        if audit['blocking_gpu_processes']:
            raise RuntimeError('Another compute process occupies the GPU; see gpu-idle-audit.json')

    def __exit__(self, *exc):
        if not self.handle or self.handle.closed:
            return
        try:
            if self.restore and self.clean:
                # Re-admit only the same fencing generation, after exact cleanup.
                try:
                    self._audit_idle_gpu()
                    restored = self.worker.admit(self.generation)
                    if (not restored.get('accepting') or not restored.get('runtime_ok')
                            or restored.get('generation') != self.generation):
                        raise RuntimeError('Factory did not confirm restored admission for the original generation')
                except BaseException:
                    self.clean = False
                    self.worker.drain()
                    write_json(self.root / 'lease-restored.json', {'status': 'left_drained',
                               'cleanup_verified': False, 'finished_at': time.time()})
                    raise
                write_json(self.root / 'lease-restored.json', {'status': 'restored',
                           'generation': self.generation, 'finished_at': time.time()})
            else:
                write_json(self.root / 'lease-restored.json', {'status': 'left_drained',
                           'cleanup_verified': self.clean, 'finished_at': time.time()})
        finally:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()


class DockerRunner:
    def __init__(self, lease: NodeLease, *, timeout_s=14400, stop_event=None):
        self.lease, self.factory = lease, lease.factory
        self.timeout_s = timeout_s
        self.stop_event = stop_event or threading.Event()
        self.lock = threading.Lock()
        self.active = {}
        self.max_active = 0

    def _verified(self, identity):
        item = self.lease.worker.docker.inspect(identity['id'])
        if item is None:
            if self.lease.worker.docker.inspect(identity['name']) is not None:
                self.lease.clean = False
                raise RuntimeError('Benchmark name was reused by another container; left drained')
            return None
        labels = item.get('Config', {}).get('Labels') or {}
        if (item.get('Id') != identity['id'] or item.get('Name', '').lstrip('/') != identity['name']
                or labels.get(BENCH_LABEL) != identity['run_id']
                or labels.get(BENCH_LABEL + '.job') != identity['job_id']):
            self.lease.clean = False
            raise RuntimeError('Benchmark container ownership is ambiguous; left drained')
        return item

    def _remove(self, identity):
        info = self._verified(identity)
        if info is None:
            return
        # Only benchmark-owned exact IDs can be cancelled or removed.
        self.lease.worker.docker.command(['rm', '-f', identity['id']])
        if self._verified(identity) is not None:
            self.lease.clean = False
            raise RuntimeError('Benchmark container removal not confirmed')

    def run(self, job: dict, root: Path):
        factory = self.factory
        root.mkdir(parents=True)
        (root / 'out').mkdir()
        started = time.time()
        started_mono = time.monotonic()
        record = {k: job[k] for k in ('job_id', 'workload_id', 'seed', 'concurrency', 'dispatcher_count', 'warmup')}
        record.update(started_at=started, started_monotonic=started_mono,
                      status='failed', retry_count=0, error='', artifact_dir=str(root))
        identity = None
        with self.lock:
            self.active[job['job_id']] = record
        try:
            payload = dict(job['payload'], seed=job['seed'])
            (root / 'lyrics.txt').write_text(payload['lyrics'], encoding='utf-8')
            (root / 'genre.txt').write_text(payload['genre'], encoding='utf-8')
            write_json(root / 'generation_request.json', {**payload, 'job_id': job['job_id'],
                       'runtime_manifest': self.lease.runtime_manifest, 'benchmark': True})
            args = factory._infer_argv(root, payload, Path(job['vocal']).read_bytes(),
                                       Path(job['instrumental']).read_bytes(), prefix='/work')
            name = 'sparkbench-' + job['job_id']
            argv = ['create', '--name', name, '--gpus', 'all', '--ipc', 'host',
                    '--ulimit', 'memlock=-1', '--ulimit', 'stack=67108864',
                    '-u', f'{os.getuid()}:{os.getgid()}', '--label', BENCH_LABEL + '=' + job['run_id'],
                    '--label', BENCH_LABEL + '.job=' + job['job_id']]
            for value in ('HOME=/work', 'PYTHONPATH=/factory/stubs:/opt/yue-stubs', 'HF_HOME=/hf',
                          'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'TORCHDYNAMO_DISABLE=1',
                          'PYTHONUNBUFFERED=1', 'YUE_WORK=/work',
                          'YUE_ATTN=' + os.environ.get('YUE_ATTN', 'sdpa'),
                          'YUE_INTEGRITY_CONFIG=' + json.dumps(payload.get('integrity_config', {}))):
                argv += ['-e', value]
            for value in (f'{factory.HF_HOME}:/hf:ro', f'{factory.YUE_ROOT}:/yue:ro',
                          f'{factory.FACTORY_DIR}:/factory:ro', f'{root}:/work'):
                argv += ['-v', value]
            argv += ['-w', '/yue/inference', self.lease.manifest['image_id'],
                     'python', '/factory/run_infer.py', *args]
            # Name and labels are recorded before creation to recover ambiguous create.
            identity = {'name': name, 'id': '', 'run_id': job['run_id'], 'job_id': job['job_id']}
            write_json(root / 'container.json', identity)
            try:
                identity['id'] = self.lease.worker.docker.command(argv).stdout.strip()
            except Exception:
                info = self.lease.worker.docker.inspect(name)
                if info:
                    identity['id'] = info['Id']
                raise
            finally:
                write_json(root / 'container.json', identity)
            self._verified(identity)
            if self.stop_event.is_set():
                record['status'] = 'cancelled'
                return record
            self.lease.worker.docker.command(['start', identity['id']])
            record['inference_started_at'] = time.time()
            record['inference_started_monotonic'] = time.monotonic()
            with self.lock:
                running = sum('inference_started_at' in value and 'inference_finished_at' not in value
                              for value in self.active.values())
                self.max_active = max(self.max_active, running)
            while True:
                info = self._verified(identity)
                if info is None:
                    raise RuntimeError('Inference container disappeared')
                if not info['State']['Running']:
                    break
                if self.stop_event.wait(1):
                    record['status'] = 'cancelled'
                    raise RuntimeError('Experiment stopped by resource guard or operator')
                if time.monotonic() - record['inference_started_monotonic'] > self.timeout_s:
                    record['status'] = 'timeout'
                    raise RuntimeError('Inference timeout; no automatic replacement/retry')
            record['inference_finished_at'] = time.time()
            record['inference_finished_monotonic'] = time.monotonic()
            record['inference_s'] = record['inference_finished_monotonic'] - record['inference_started_monotonic']
            record['exit_code'] = info['State'].get('ExitCode')
            record['oom_killed'] = info['State'].get('OOMKilled', False)
            logs = self.lease.worker.docker.command(['logs', identity['id']], timeout=120, check=False)
            (root / 'worker.log').write_text(logs.stdout + '\n' + logs.stderr, encoding='utf-8')
            record['cuda_oom'] = any(marker in (logs.stdout + logs.stderr).lower() for marker in
                                     ('cuda out of memory', 'cuda error: out of memory', 'torch.outofmemoryerror'))
            diagnostics = root / 'worker-diagnostics.json'
            report = json.loads(diagnostics.read_text()) if diagnostics.exists() else None
            if record['exit_code'] or not report or not report.get('ok'):
                if report and not report.get('ok'):
                    record['status'] = 'failed_integrity'
                raise RuntimeError('YuE inference/structural integrity failed; preserved worker.log and artifacts')
            body = factory._pick_audio(root / 'out', factory.IntegrityConfig(**payload.get('integrity_config', {})))
            (root / 'output.wav').write_bytes(body)
            with wave.open(str(root / 'output.wav')) as audio:
                record['audio_duration_s'] = audio.getnframes() / audio.getframerate()
            record.update(status='succeeded', output_sha256=digest_file(root / 'output.wav'))
        except Exception as exc:
            record['error'] = str(exc)[-1500:]
            if 'FAILED_INTEGRITY' in str(exc):
                record['status'] = 'failed_integrity'
        finally:
            if identity and identity.get('id'):
                try:
                    # Preserve even timeout/cancellation logs before cleanup.
                    logs = self.lease.worker.docker.command(['logs', '--tail', '2000', identity['id']], check=False)
                    if not (root / 'worker.log').exists():
                        (root / 'worker.log').write_text(logs.stdout + '\n' + logs.stderr, encoding='utf-8')
                except Exception as exc:
                    self.stop_event.set()
                    record.update(status='failed', error='Worker evidence capture failed: ' + str(exc))
                try:
                    self._remove(identity)
                except Exception as exc:
                    self.lease.clean = False
                    self.stop_event.set()
                    record.update(status='failed', error='Cleanup unverified: ' + str(exc))
            elif identity:
                self.lease.clean = False
                self.stop_event.set()
            if 'inference_started_monotonic' in record and 'inference_finished_monotonic' not in record:
                record['inference_finished_at'] = time.time()
                record['inference_finished_monotonic'] = time.monotonic()
                record['inference_s'] = record['inference_finished_monotonic'] - record['inference_started_monotonic']
            record['finished_at'] = time.time()
            record['finished_monotonic'] = time.monotonic()
            record['latency_s'] = record['finished_monotonic'] - started_mono
            record['max_active_observed'] = self.max_active
            write_json(root / 'job.json', record)
            with self.lock:
                self.active.pop(job['job_id'], None)
        return record


class SyntheticRunner:
    """Non-GPU harness validation; these timings cannot support a capacity claim."""
    def __init__(self, *, delay_s=0.02, stop_event=None, fail_seed=None):
        self.delay_s, self.fail_seed = delay_s, fail_seed
        self.stop_event = stop_event or threading.Event()

    def run(self, job, root):
        root.mkdir(parents=True)
        start = time.time()
        mono_start = time.monotonic()
        self.stop_event.wait(self.delay_s * job['payload']['run_n_segments'])
        record = {k: job[k] for k in ('job_id', 'workload_id', 'seed', 'concurrency', 'dispatcher_count', 'warmup')}
        record.update(started_at=start, finished_at=time.time(), inference_started_at=start,
                      inference_finished_at=time.time(), inference_started_monotonic=mono_start,
                      inference_finished_monotonic=time.monotonic(), latency_s=time.monotonic() - mono_start,
                      inference_s=time.monotonic() - mono_start, status='cancelled' if self.stop_event.is_set()
                      else 'failed_integrity' if job['seed'] == self.fail_seed else 'succeeded',
                      retry_count=0, audio_duration_s=job['payload']['run_n_segments'] * 30,
                      synthetic=True, error='')
        write_json(root / 'job.json', record)
        return record
