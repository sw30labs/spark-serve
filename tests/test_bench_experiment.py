import json
import threading
from pathlib import Path

import pytest

from spark_bench.experiment import (
    execute_trial,
    jobs_for_trial,
    levels_value,
    load_workload,
    trial_order,
    validate_endpoint,
)
from spark_bench.runtime import SyntheticRunner

CORPUS = Path(__file__).parent / 'workloads/yue-standard.json'


def test_same_seeded_corpus_at_every_concurrency():
    corpus = load_workload(CORPUS, synthetic=True)
    signatures = []
    for level in [1, 2, 3, 4]:
        jobs = jobs_for_trial(corpus, count=8, seed=42, run_id='one', trial_id=f'c{level}', concurrency=level)
        signatures.append([(job['workload_id'], job['seed'], job['payload']) for job in jobs])
    assert all(signature == signatures[0] for signature in signatures)
    assert [workload for workload, _, _ in signatures[0]].count('short') == 4
    assert list(trial_order([1, 2, 3, 4], 2)) == [(0, 1), (0, 2), (0, 3), (0, 4),
                                                        (1, 4), (1, 3), (1, 2), (1, 1)]


def test_workload_preserves_spanish_and_rejects_parser_loss(tmp_path):
    raw = json.loads(CORPUS.read_text())
    lyrics = raw['workloads'][1]['lyrics']
    parsed = load_workload(CORPUS, synthetic=True)
    assert parsed['workloads'][1]['lyrics'] == lyrics
    for word in ('canción', 'corazón', 'También', 'está', 'música', 'niño', '¡Qué', '¿Cómo', 'Sí', 'aún', 'vergüenza'):
        assert word in lyrics
    raw['workloads'][1]['lyrics'] = lyrics.replace('[chorus]', '[Pre-Chorus]')
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='malformed'):
        load_workload(path, synthetic=True)


@pytest.mark.parametrize('value', ['0,1', '2,3', '1,1', '', '1,2.5'])
def test_invalid_sweep(value):
    with pytest.raises(ValueError):
        levels_value(value)


@pytest.mark.parametrize('value', ['http://user:secret@node:8011', 'https://node/?token=secret',
                                  'http://node/path', 'http://node/#secret', 'http://node\n'])
def test_endpoint_never_records_credentials(value):
    with pytest.raises(ValueError, match='without credentials'):
        validate_endpoint(value)


def test_trial_really_overlaps_and_saves_failures(tmp_path):
    corpus = load_workload(CORPUS, synthetic=True)
    jobs = jobs_for_trial(corpus, count=4, seed=42, run_id='run', trial_id='c2', concurrency=2)
    stop = threading.Event()
    runner = SyntheticRunner(delay_s=.04, stop_event=stop, fail_seed=43)
    root = tmp_path / 'trial'
    trial = execute_trial(root, jobs, runner, concurrency=2, stop_event=stop, synthetic=True,
                          telemetry_interval=.02)
    records = [json.loads(line) for line in (root / 'jobs.jsonl').read_text().splitlines()]
    assert len(records) == 4
    assert sum(row['status'] == 'succeeded' for row in records) == 3
    assert sum(row['status'] == 'failed_integrity' for row in records) == 1
    first = sorted(records, key=lambda row: row['started_at'])[:2]
    assert first[1]['started_at'] < first[0]['finished_at']
    assert trial['finished_at'] >= max(row['finished_at'] for row in records)
    assert trial['completed_jobs'] == 3
    assert (root / 'telemetry.jsonl').is_file()


def test_cancelled_unlaunched_jobs_are_counted(tmp_path):
    corpus = load_workload(CORPUS, synthetic=True)
    jobs = jobs_for_trial(corpus, count=4, seed=42, run_id='run', trial_id='c2', concurrency=2)
    stop = threading.Event()
    stop.set()
    root = tmp_path / 'trial'
    result = execute_trial(root, jobs, SyntheticRunner(stop_event=stop), concurrency=2,
                           stop_event=stop, synthetic=True, telemetry_interval=.02)
    records = [json.loads(line) for line in (root / 'jobs.jsonl').read_text().splitlines()]
    assert len(records) == 4
    assert all(row['status'] == 'cancelled' for row in records)
    assert result['status'] == 'stopped'
