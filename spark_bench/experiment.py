"""Controlled same-corpus concurrency sweep with durable per-job outcomes."""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import signal
import socket
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

from .common import append_json, digest_file, write_json
from .provenance import collect_provenance
from .runtime import DockerRunner, NodeLease, SyntheticRunner, load_factory


@contextmanager
def stop_on_error(stop_event):
    try:
        yield
    except BaseException:
        stop_event.set()
        raise


def levels_value(text: str) -> list[int]:
    try:
        values = [int(value) for value in text.split(',')]
    except ValueError:
        raise ValueError('Concurrency must be comma-separated positive integers') from None
    if not values or len(set(values)) != len(values) or min(values) < 1:
        raise ValueError('Concurrency levels must be distinct positive integers')
    if 1 not in values:
        raise ValueError('Include concurrency=1 as the baseline')
    return values


def validate_endpoint(value):
    if value is None:
        return None
    parsed = urlsplit(value)
    if (parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.path not in ('', '/')
            or any(character.isspace() or ord(character) < 32 for character in value)):
        raise ValueError('Endpoint must be an HTTP(S) origin without credentials, query, or fragment')
    return value.rstrip('/')


def load_workload(path: Path, *, synthetic=False) -> dict:
    raw = json.loads(path.read_text(encoding='utf-8'))
    allowed = {'version', 'rights_basis', 'attestation', 'vocal', 'instrumental', 'workloads'}
    if not isinstance(raw, dict) or set(raw) - allowed or raw.get('version') != 1:
        raise ValueError('Workload must be a version=1 object with documented fields only')
    if not isinstance(raw.get('workloads'), list) or not raw['workloads']:
        raise ValueError('Provide at least one workload')
    if not synthetic:
        if raw.get('rights_basis') not in {'operator_owned', 'written_license', 'own_voice'}:
            raise ValueError('An authorized reference rights_basis is required')
        for key in ('attestation', 'vocal', 'instrumental'):
            value = str(raw.get(key, ''))
            if not value or '://' in value:
                raise ValueError(key + ' must be a local path')
            candidate = Path(value).expanduser()
            candidate = candidate if candidate.is_absolute() else path.parent / candidate
            if not candidate.is_file():
                raise ValueError('Missing local ' + key + ': ' + str(candidate))
            raw[key] = str(candidate.resolve())
        if Path(raw['attestation']).stat().st_size < 20:
            raise ValueError('Reference attestation must be nonempty')
        import wave
        for key in ('vocal', 'instrumental'):
            with wave.open(raw[key]) as audio:
                if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (44100, 1, 2):
                    raise ValueError(key + ' must be 44.1kHz mono PCM16')
                if abs(audio.getnframes() / 44100 - 30) > .05:
                    raise ValueError(key + ' must be an aligned 30-second reference')
        if digest_file(Path(raw['vocal'])) == digest_file(Path(raw['instrumental'])):
            raise ValueError('Reference tracks must contain distinct audio')
    identifiers = []
    for item in raw['workloads']:
        if not isinstance(item, dict) or set(item) - {'id', 'lyrics', 'genre', 'run_n_segments', 'max_new_tokens'}:
            raise ValueError('Unknown workload item fields')
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,48}', item.get('id', '')):
            raise ValueError('Workload IDs must be short alphanumeric identifiers')
        identifiers.append(item['id'])
        lyrics = item.get('lyrics', '')
        if not isinstance(lyrics, str) or not lyrics.strip() or '\x00' in lyrics:
            raise ValueError('Provide nonempty Unicode lyrics without NULL')
        sections = re.findall(r'\[(\w+)\](.*?)(?=\[|\Z)', lyrics, flags=re.DOTALL)
        # Refuse parser ambiguity instead of cleaning away punctuation or accents.
        if len(sections) != len(re.findall(r'\[[^\]]*\]', lyrics)) or any(not text.strip() for _, text in sections):
            raise ValueError('Workload lyrics contain malformed or empty YuE sections')
        segments = item.get('run_n_segments', max(2, len(sections)))
        tokens = item.get('max_new_tokens', 3000)
        if isinstance(segments, bool) or not isinstance(segments, int) or not 2 <= segments <= 12:
            raise ValueError('run_n_segments must be an integer in 2..12')
        if not sections or len(sections) > segments:
            raise ValueError('All lyric sections must fit the requested segment count')
        if isinstance(tokens, bool) or not isinstance(tokens, int) or not 3000 <= tokens <= 4000:
            raise ValueError('max_new_tokens must be an integer in 3000..4000')
        if not isinstance(item.get('genre', ''), str) or not item.get('genre', '').strip():
            raise ValueError('Each workload requires genre text')
        item.update(run_n_segments=segments, max_new_tokens=tokens)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('Workload IDs must be unique')
    return raw


def freeze_workload(workload: dict, root: Path, *, synthetic=False):
    frozen = json.loads(json.dumps(workload, ensure_ascii=False))
    folder = root / 'inputs'
    folder.mkdir()
    if not synthetic:
        hashes = {}
        for key, filename in (('vocal', 'vocal.wav'), ('instrumental', 'instrumental.wav'), ('attestation', 'attestation.txt')):
            source = Path(frozen[key])
            target = folder / filename
            shutil.copyfile(source, target)
            hashes[key] = digest_file(target)
            if hashes[key] != digest_file(source):
                raise ValueError('Input changed while freezing the corpus')
            frozen[key] = str(target.resolve())
        write_json(folder / 'reference-hashes.json', hashes)
    write_json(folder / 'workload.json', frozen)
    return frozen


def trial_order(levels, iterations):
    """Counterbalance odd/even rounds; rotate starts for >2 rounds."""
    for iteration in range(iterations):
        ordered = list(levels if iteration % 2 == 0 else reversed(levels))
        shift = (iteration // 2) % len(ordered)
        for concurrency in ordered[shift:] + ordered[:shift]:
            yield iteration, concurrency


def jobs_for_trial(workload, *, count, seed, run_id, trial_id, concurrency, warmup=False):
    jobs = []
    for index in range(count):
        item = workload['workloads'][index % len(workload['workloads'])]
        jobs.append({'job_id': f'{run_id}-{trial_id}-{index + 1:03d}', 'run_id': run_id,
                     'workload_id': item['id'], 'seed': (seed + index) % 2**32,
                     'concurrency': concurrency, 'dispatcher_count': 1, 'warmup': warmup,
                     'vocal': workload.get('vocal'), 'instrumental': workload.get('instrumental'),
                     'payload': {key: value for key, value in item.items() if key != 'id'}})
    return jobs


def execute_trial(root, jobs, runner, *, concurrency, stop_event, telemetry_interval=2, synthetic=False,
                  min_free_gb=12, max_temperature_c=85, max_swap_growth_gb=1, min_temperature_margin_c=5):
    from .telemetry import TelemetryCollector, TelemetrySampler
    root.mkdir(parents=True)
    start = time.time()
    start_mono = time.monotonic()
    trial = {'trial_id': root.name, 'concurrency': concurrency, 'dispatcher_count': 1,
             'started_at': start, 'finished_at': None, 'measured': not jobs[0]['warmup'],
             'expected_jobs': len(jobs), 'status': 'running', 'stop_reason': None}
    write_json(root / 'trial.json', trial)
    sampler = TelemetrySampler()
    outcomes = []
    collector = TelemetryCollector(root, interval_s=telemetry_interval)
    baseline_swap = None
    next_check = 0
    if hasattr(runner, 'max_active'):
        runner.max_active = 0
    try:
        collector.start()
        # The inner guard signals cancellation BEFORE executor shutdown waits.
        with ThreadPoolExecutor(max_workers=concurrency) as pool, stop_on_error(stop_event):
            pending = {}
            index = 0
            while index < len(jobs) or pending:
                collector.raise_if_failed()
                if not synthetic and time.monotonic() >= next_check:
                    sample = sampler.sample()
                    memory = sample.get('memory', {})
                    available = memory.get('available_bytes')
                    swap = memory.get('swap_used_bytes')
                    if baseline_swap is None:
                        baseline_swap = swap
                    temperatures = [gpu.get('temperature_gpu_c') for gpu in sample.get('gpus', [])]
                    temperatures = [value for value in temperatures if isinstance(value, (int, float))]
                    margins = [gpu.get('temperature_tlimit_c') for gpu in sample.get('gpus', [])]
                    margins = [value for value in margins if isinstance(value, (int, float)) and math.isfinite(value)]
                    reason = None
                    if available is None:
                        reason = 'Unified memory telemetry unavailable; cannot safely increase load'
                    elif not temperatures:
                        reason = 'GPU temperature telemetry unavailable; cannot monitor the thermal guard'
                    elif available < min_free_gb * 1024**3:
                        reason = 'Available unified memory below configured reserve'
                    elif temperatures and max(temperatures) >= max_temperature_c:
                        reason = 'GPU temperature exceeded configured guard'
                    elif margins and min(margins) <= min_temperature_margin_c:
                        reason = 'GPU thermal headroom reached configured minimum'
                    elif swap is not None and baseline_swap is not None and swap - baseline_swap > max_swap_growth_gb * 1024**3:
                        reason = 'Swap growth exceeded configured guard'
                    if reason:
                        trial['stop_reason'] = reason
                        stop_event.set()
                    next_check = time.monotonic() + telemetry_interval
                while index < len(jobs) and len(pending) < concurrency and not stop_event.is_set():
                    job = jobs[index]
                    pending[pool.submit(runner.run, job, root / 'jobs' / job['job_id'])] = job
                    index += 1
                if not pending:
                    break
                done, _ = wait(pending, timeout=min(telemetry_interval, 1), return_when=FIRST_COMPLETED)
                for future in done:
                    job = pending.pop(future)
                    try:
                        record = future.result()
                    except Exception as exc:
                        record = {key: job[key] for key in ('job_id', 'workload_id', 'seed', 'concurrency', 'dispatcher_count', 'warmup')}
                        record.update(status='failed', error=type(exc).__name__ + ': ' + str(exc), finished_at=time.time(), retry_count=0)
                        stop_event.set()
                    outcomes.append(record)
                    append_json(root / 'jobs.jsonl', record)
                    print(json.dumps({'trial': root.name, 'job': record['job_id'], 'status': record['status'],
                                      'latency_s': record.get('latency_s')}, ensure_ascii=False), flush=True)
                    if record.get('oom_killed') or record.get('cuda_oom') or record['status'] == 'timeout':
                        trial['stop_reason'] = 'OOM or inference timeout; higher levels require explicit review'
                        stop_event.set()
            for job in jobs[index:]:
                record = {key: job[key] for key in ('job_id', 'workload_id', 'seed', 'concurrency', 'dispatcher_count', 'warmup')}
                record.update(status='cancelled', error='Not launched: experiment stopped', finished_at=time.time(), retry_count=0)
                append_json(root / 'jobs.jsonl', record)
                outcomes.append(record)
    finally:
        try:
            collector.stop()
        finally:
            trial.update(finished_at=time.time(), duration_s=time.monotonic() - start_mono,
                         status='stopped' if stop_event.is_set() else 'complete',
                         completed_jobs=sum(item['status'] == 'succeeded' for item in outcomes))
            write_json(root / 'trial.json', trial)
    return trial


def run_experiment(args):
    from .analysis import analyze
    args.endpoint = validate_endpoint(args.endpoint)
    levels = levels_value(args.concurrency)
    if args.jobs_per_level < max(levels) or args.jobs_per_level < 1 or args.iterations < 1 or args.warmup < 0:
        raise ValueError('Use >= maximum concurrency jobs per level, positive iterations, and nonnegative warmup')
    if not 0 <= args.seed < 2**32:
        raise ValueError('Seed must be in 0..4294967295')
    for key in ('timeout', 'telemetry_interval', 'min_free_gb', 'max_temperature_c', 'wait_idle', 'max_swap_growth_gb'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(key + ' must be finite and positive')
    if not math.isfinite(args.cooldown) or args.cooldown < 0:
        raise ValueError('cooldown must be finite and nonnegative')
    if not math.isfinite(args.min_temperature_margin_c) or args.min_temperature_margin_c < 0:
        raise ValueError('min_temperature_margin_c must be finite and nonnegative')
    workload = load_workload(args.workload, synthetic=args.synthetic)
    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    workload = freeze_workload(workload, root, synthetic=args.synthetic)
    run_id = 'b' + time.strftime('%Y%m%d%H%M%S') + '-' + os.urandom(3).hex()
    metadata = {'version': 1, 'run_id': run_id, 'synthetic': args.synthetic, 'node': socket.gethostname(),
                'provenance': collect_provenance(),
                'worker': args.worker or socket.gethostname(), 'endpoint': args.endpoint,
                'architecture': 'one experimental dispatcher; independent model process/container per job',
                'concurrency_definition': 'active inference containers per physical Spark',
                'concurrency_levels': levels, 'iterations': args.iterations, 'jobs_per_level': args.jobs_per_level,
                'warmup_jobs': args.warmup, 'seed': args.seed, 'cooldown_s': args.cooldown,
                'timeout_s': args.timeout, 'telemetry_interval_s': args.telemetry_interval,
                'workload_sha256': digest_file(root / 'inputs' / 'workload.json'),
                'workloads': [item['id'] for item in workload['workloads']],
                'order': list(trial_order(levels, args.iterations)), 'started_at': time.time(),
                'status': 'preparing', 'network_topology_changed': False,
                'guards': {'min_free_gb': args.min_free_gb, 'max_temperature_c': args.max_temperature_c,
                           'min_temperature_margin_c': args.min_temperature_margin_c,
                           'max_swap_growth_gb': args.max_swap_growth_gb}}
    write_json(root / 'run-metadata.json', metadata)
    stop = threading.Event()
    previous = {}
    for number in (signal.SIGINT, signal.SIGTERM):
        previous[number] = signal.signal(number, lambda *_: stop.set())
    lease = None
    try:
        if args.synthetic:
            runner = SyntheticRunner(stop_event=stop)
        else:
            if sys_platform() != 'linux':
                raise ValueError('Live benchmark runs locally on the selected Spark (Linux); use SSH deployment')
            factory = load_factory(args.factory_root, args.service)
            lease = NodeLease(factory, root, wait_idle_s=args.wait_idle, stop_event=stop)
            lease.__enter__()
            if args.worker and factory.WORKER_ID != args.worker:
                raise ValueError('Selected worker identity differs from this physical node')
            metadata.update(runtime_manifest=lease.runtime_manifest, image_id=lease.manifest['image_id'],
                            source_revision=factory.SOURCE_REV, stage1_revision=factory.STAGE1_REV,
                            stage2_revision=factory.STAGE2_REV, codec_revision=factory.CODEC_REV,
                            attention=os.environ.get('YUE_ATTN', 'sdpa'))
            runner = DockerRunner(lease, timeout_s=args.timeout, stop_event=stop)
        metadata['status'] = 'running'
        write_json(root / 'run-metadata.json', metadata)
        schedule = ([(-1, 1)] if args.warmup else []) + list(trial_order(levels, args.iterations))
        for iteration, concurrency in schedule:
            if stop.is_set():
                break
            warmup = iteration < 0
            trial_id = 'warmup' if warmup else f'r{iteration + 1:02d}-c{concurrency:02d}'
            trial_jobs = jobs_for_trial(workload, count=args.warmup if warmup else args.jobs_per_level,
                                       seed=args.seed, run_id=run_id, trial_id=trial_id,
                                       concurrency=concurrency, warmup=warmup)
            execute_trial(root / 'trials' / trial_id, trial_jobs, runner, concurrency=concurrency,
                          stop_event=stop, synthetic=args.synthetic, telemetry_interval=args.telemetry_interval,
                          min_free_gb=args.min_free_gb, max_temperature_c=args.max_temperature_c,
                          max_swap_growth_gb=args.max_swap_growth_gb,
                          min_temperature_margin_c=args.min_temperature_margin_c)
            analyze(root)
            if args.cooldown and not stop.is_set():
                stop.wait(args.cooldown)
        metadata['status'] = 'stopped' if stop.is_set() else 'complete'
    except BaseException as exc:
        metadata.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        try:
            if lease:
                lease.__exit__(None, None, None)
        except Exception as exc:
            metadata.update(status='failed', restoration_error=str(exc))
        metadata['finished_at'] = time.time()
        write_json(root / 'run-metadata.json', metadata)
        analyze(root)
        for number, handler in previous.items():
            signal.signal(number, handler)
    if metadata['status'] != 'complete':
        raise RuntimeError(metadata.get('restoration_error') or metadata.get('error') or
                           'Experiment stopped before completion; inspect preserved results')
    return root


def sys_platform():
    import sys
    return sys.platform
