"""Mac-side deployment/run/collection while holding Spark Serve's real lock."""
from __future__ import annotations

import io
import json
import os
import shlex
import subprocess
import tarfile
import time
import tomllib
from pathlib import Path

from spark_serve_controller import Controller, yue_profile

from .common import write_json
from .experiment import load_workload


def safe_extract(archive: tarfile.TarFile, root: Path):
    """Only regular files/directories under the requested benchmark directory."""
    for member in archive.getmembers():
        candidate = (root / member.name).resolve()
        if not candidate.is_relative_to(root.resolve()) or not (member.isfile() or member.isdir()):
            raise RuntimeError('Unsafe benchmark archive entry')
    archive.extractall(root, filter='data')


def remote_run(args):
    config = tomllib.loads(args.config.read_text())
    workers = yue_profile(config)['workers']
    selected = next((worker for worker in workers if worker['host'] == args.ssh_host), None)
    if selected is None:
        raise ValueError('--ssh-host must be one of the configured Spark Serve hosts')
    workload = load_workload(args.workload)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    identifier = 'bench-' + time.strftime('%Y%m%d-%H%M%S') + '-' + os.urandom(3).hex()
    # Fixed, dedicated subtree; no shell home-variable interpolation.
    remote_rel = '.local/share/spark-serve/benchmarks/' + identifier
    ssh = ['ssh', *config['cluster'].get('ssh_opts', ['-o', 'BatchMode=yes']), args.ssh_host]
    repo = Path(__file__).resolve().parents[1]
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode='w:gz') as archive:
        for path in sorted((repo / 'spark_bench').glob('*.py')):
            archive.add(path, arcname='src/spark_bench/' + path.name)
        archive.add(repo / 'spark_serve_controller.py', arcname='src/spark_serve_controller.py')
        for field, name in (('attestation', 'attestation.txt'), ('vocal', 'vocal.wav'), ('instrumental', 'instrumental.wav')):
            archive.add(workload[field], arcname='input/' + name)
            workload[field] = name
        body = json.dumps(workload, ensure_ascii=False).encode()
        entry = tarfile.TarInfo('input/workload.json')
        entry.size = len(body)
        archive.addfile(entry, io.BytesIO(body))
    controller = Controller(config, None)
    receipt = {'version': 1, 'host': args.ssh_host, 'remote_path': remote_rel,
               'started_at': time.time(), 'status': 'preparing', 'network_changed': False}
    write_json(output / 'remote-run.json', receipt)
    with controller.lock():
        if controller.state.get('mode') != 'yue' or controller.state.get('phase') != 'ready':
            raise RuntimeError('Spark Serve must already be in admitted YuE mode')
        # Deployment contains our own regular files only; validate on receive too.
        deploy = '''import io,pathlib,sys,tarfile
root=pathlib.Path.home()/sys.argv[1]
root.mkdir(parents=True,exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as archive:
    for entry in archive.getmembers():
        if not (root/entry.name).resolve().is_relative_to(root.resolve()) or not (entry.isfile() or entry.isdir()):
            raise RuntimeError('unsafe archive')
    archive.extractall(root,filter='data')
print(root)
'''
        result = subprocess.run([*ssh, shlex.join(['python3', '-c', deploy, remote_rel])],
                                input=data.getvalue(), capture_output=True, timeout=120)
        if result.returncode:
            raise RuntimeError('Benchmark deployment failed: ' + result.stderr.decode(errors='replace')[-500:])
        remote_root = result.stdout.decode().strip()
        remote_args = ['concurrency', '--workload', remote_root + '/input/workload.json',
                       '--output', remote_root + '/results', '--endpoint', selected['url'],
                       '--concurrency', args.concurrency, '--jobs-per-level', str(args.jobs_per_level),
                       '--iterations', str(args.iterations), '--warmup', str(args.warmup),
                       '--cooldown', str(args.cooldown), '--timeout', str(args.timeout),
                       '--wait-idle', str(args.wait_idle), '--seed', str(args.seed),
                       '--telemetry-interval', str(args.telemetry_interval),
                       '--factory-root', str(args.factory_root), '--service', args.service,
                       '--min-free-gb', str(args.min_free_gb), '--max-temperature-c', str(args.max_temperature_c),
                       '--max-swap-growth-gb', str(args.max_swap_growth_gb)]
        if args.worker:
            remote_args += ['--worker', args.worker]
        # Remote child is attached to SSH. NodeLease is the final GPU fence even
        # if this Mac process disappears; never kill unrelated jobs on disconnect.
        launch = ['env', 'PYTHONPATH=' + remote_root + '/src', 'python3', '-m', 'spark_bench', *remote_args]
        receipt.update(status='running', remote_path=remote_root)
        write_json(output / 'remote-run.json', receipt)
        with (output / 'driver.log').open('wb') as log:
            result = subprocess.run([*ssh, shlex.join(launch)], stdout=log, stderr=subprocess.STDOUT)
        receipt.update(status='collecting', returncode=result.returncode, finished_at=time.time())
        write_json(output / 'remote-run.json', receipt)
        # Preserve full WAVs/tokens/logs remotely. Collect compact raw metrics and
        # reports locally; a separate explicit archive can retrieve large evidence.
        collect = '''import pathlib,sys,tarfile
root=pathlib.Path(sys.argv[1])/'results'
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
    for path in sorted(root.rglob('*')):
        if path.is_file() and not path.is_symlink() and path.suffix in ('.json','.jsonl','.csv','.md','.svg'):
            archive.add(path,arcname=str(path.relative_to(root)))
'''
        collected = subprocess.run([*ssh, shlex.join(['python3', '-c', collect, remote_root])],
                                   capture_output=True, timeout=180)
        if collected.returncode:
            raise RuntimeError('Raw results remain on the Spark; automatic collection failed')
        with tarfile.open(fileobj=io.BytesIO(collected.stdout), mode='r:gz') as archive:
            safe_extract(archive, output / 'results')
        receipt['status'] = 'complete' if result.returncode == 0 else 'stopped_or_failed'
        write_json(output / 'remote-run.json', receipt)
        if result.returncode:
            raise RuntimeError('Remote benchmark stopped; inspect collected results and driver.log')
    return output
