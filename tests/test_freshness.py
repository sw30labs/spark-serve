"""App and per-card "updated" dates: git HEAD, per-entry blame, fallbacks, cache."""
import datetime as dt
import importlib.machinery
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import spark_serve_freshness as fr

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("freshness_cli", str(ROOT / "spark-serve"))
spec = importlib.util.spec_from_loader(loader.name, loader)
cli = importlib.util.module_from_spec(spec)
loader.exec_module(cli)

EXAMPLE_V1 = """[cluster]
nnodes = 2

[models.a]
image = "a1"

[models.a.env]
X = "1"

[models.b]
image = "b1"
"""


def git(repo, *args, when=None):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t")
    if when:
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = when
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True, env=env).stdout


def repo(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "models.example.toml").write_text(EXAMPLE_V1)
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "one", when="2026-10-01T12:00:00")
    text = EXAMPLE_V1.replace('X = "1"', 'X = "2"')
    (tmp_path / "models.example.toml").write_text(text)
    git(tmp_path, "commit", "-qam", "two", when="2026-10-20T12:00:00")
    return tmp_path


def test_entry_line_ranges_cover_subtables_and_stop_at_other_tables():
    ranges = fr.entry_line_ranges(EXAMPLE_V1)
    assert ranges == {"a": [(4, 6), (7, 9)], "b": [(10, 11)]}


def test_app_info_reads_head_commit(tmp_path):
    info = fr.app_info(repo(tmp_path))
    assert info["source"] == "git" and info["updated"] == "2026-10-20"
    assert len(info["short"]) >= 7 and info["dirty"] is False


def test_app_info_outside_git_uses_script_mtime(tmp_path):
    script = tmp_path / "spark-serve"
    script.write_text("#")
    stamp = dt.datetime(2026, 9, 3, 12).timestamp()
    os.utime(script, (stamp, stamp))
    assert fr.app_info(tmp_path)["updated"] == "2026-09-03"
    assert fr.app_info(tmp_path)["source"] == "mtime"


def test_entry_dates_by_source(tmp_path):
    root = repo(tmp_path)
    example = root / "models.example.toml"
    catalog = root / "models.toml"
    catalog.write_text("")
    stamp = dt.datetime(2026, 10, 28, 12).timestamp()
    os.utime(catalog, (stamp, stamp))
    example_models = {"a": {"image": "a1", "env": {"X": "2"}}, "b": {"image": "b1"}}
    live = {
        "a": {"image": "a1", "env": {"X": "2"}},          # same as example -> blame of its lines
        "b": {"image": "b2"},                             # edited locally -> models.toml mtime
        "c": {"image": "c", "updated": dt.date(2026, 10, 30)},  # explicit field wins
        "d": {"image": "d"},                              # local only, re-measured later
    }
    perf = {"d": {"measured_at": "2026-10-29"}, "a": {"measured_at": "2026-10-05"}}
    dates = fr.entry_dates(root, live, example_models, perf, catalog, example)
    assert dates["a"]["updated"] == "2026-10-20" and dates["a"]["updated_source"].startswith("commit ")
    assert dates["b"] == {"updated": "2026-10-28", "updated_source": "models.toml saved"}
    assert dates["c"] == {"updated": "2026-10-30", "updated_source": "catalog field"}
    assert dates["d"] == {"updated": "2026-10-29", "updated_source": "measured"}


def test_untouched_entry_keeps_its_older_commit(tmp_path):
    root = repo(tmp_path)
    dates = fr.entry_dates(root, {"b": {"image": "b1"}}, {"b": {"image": "b1"}}, {},
                           root / "missing.toml", root / "models.example.toml")
    assert dates["b"]["updated"] == "2026-10-01"


def test_cache_recomputes_only_when_inputs_change(tmp_path):
    root = repo(tmp_path)
    calls = []
    path = root / "watched"
    path.write_text("1")
    cache = tmp_path / "state" / "c.json"
    compute = lambda: calls.append(1) or {"n": len(calls)}
    assert fr.cached(root, cache, [path], compute) == {"n": 1}
    assert fr.cached(root, cache, [path], compute) == {"n": 1}
    path.write_text("22")
    assert fr.cached(root, cache, [path], compute) == {"n": 2}
    cache.write_text("garbage")
    assert fr.cached(root, cache, [path], compute) == {"n": 3}


def test_list_json_carries_updated_and_never_fails(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("SPARK_SERVE_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "measured_performance", lambda: {})
    monkeypatch.setattr(cli, "catalog_freshness",
                        lambda cfg, perf: {"a": {"updated": "2026-10-30", "updated_source": "catalog field"}})
    cli.cmd_list_json({"cluster": {"nnodes": 2}, "models": {"a": {"image": "i"}, "b": {"image": "i"}}})
    items = {i["id"]: i for i in json.loads(capsys.readouterr().out)}
    assert items["a"]["updated"] == "2026-10-30" and items["b"]["updated"] is None



def test_catalog_freshness_swallows_errors(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("git exploded")
    monkeypatch.setattr(cli.freshness, "cached", boom)
    assert cli.catalog_freshness({"models": {}}, {}) == {}


def test_about_json(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli, "CATALOG", tmp_path / "none.toml")
    cli.main(["about", "--json"])
    info = json.loads(capsys.readouterr().out)
    assert info["source"] in ("git", "mtime") and "updated" in info and info["catalog_saved"] is None
