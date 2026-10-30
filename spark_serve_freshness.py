"""When the app checkout and each catalog entry last changed, for the GUI.

Dates come from sources that already exist, never from "now":

- App: the commit date of the checkout's HEAD (plus short hash and branch).
  Not a git checkout -> the mtime of the spark-serve script.
- Entry, first match wins:
  1. an explicit ``updated = 2026-10-30`` in the entry's models.toml table;
  2. the entry matches models.example.toml -> newest committer date among the
     tracked lines of that entry's tables (one ``git blame`` for the file);
  3. otherwise (local-only or locally edited entry) -> models.toml mtime.
  A newer ``measured_at`` from measured-performance.json then wins, because a
  re-measure changes the card too.

Everything is computed once per change of HEAD or of the three files and
cached in the state dir, so the GUI's 15 s poll does not re-run git.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

CACHE_SCHEMA = 1
_HEADER = re.compile(r"^\s*\[\s*models\.([A-Za-z0-9_-]+)\s*[.\]]")
_ANY_TABLE = re.compile(r"^\s*\[")


def _git(here: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(here), *args], capture_output=True, text=True,
                                timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _day(timestamp: float) -> str:
    return dt.datetime.fromtimestamp(timestamp).date().isoformat()


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def app_info(here: Path) -> dict:
    """HEAD commit of the checkout, or the script mtime outside git."""
    out = _git(here, "log", "-1", "--format=%H%x00%h%x00%ct%x00%cI")
    if out and out.count("\0") == 3:
        full, short, ct, iso = out.strip().split("\0")
        branch = (_git(here, "rev-parse", "--abbrev-ref", "HEAD") or "").strip() or None
        dirty = bool((_git(here, "status", "--porcelain", "--untracked-files=no") or "").strip())
        return {"source": "git", "commit": full, "short": short, "branch": branch, "dirty": dirty,
                "committed_at": iso, "updated": _day(int(ct))}
    stamp = _mtime(here / "spark-serve")
    return {"source": "mtime", "commit": None, "short": None, "branch": None, "dirty": False,
            "committed_at": None, "updated": _day(stamp) if stamp else None}


def entry_line_ranges(text: str) -> dict[str, list[tuple[int, int]]]:
    """1-based inclusive line ranges of each [models.<id>...] table."""
    ranges: dict[str, list[tuple[int, int]]] = {}
    current, start = None, 0
    lines = text.splitlines()
    for number, line in enumerate(lines, 1):
        if _ANY_TABLE.match(line):
            if current is not None:
                ranges.setdefault(current, []).append((start, number - 1))
            match = _HEADER.match(line)
            current, start = (match.group(1), number) if match else (None, 0)
    if current is not None:
        ranges.setdefault(current, []).append((start, len(lines)))
    return ranges


def blame_times(here: Path, relpath: str) -> dict[int, tuple[int, str]]:
    """Line number -> (committer time, short commit); uncommitted lines are skipped."""
    out = _git(here, "blame", "--line-porcelain", "--", relpath)
    times: dict[int, tuple[int, str]] = {}
    if not out:
        return times
    sha, line_no = None, None
    for line in out.splitlines():
        head = line.split(" ")
        if len(head) >= 3 and len(head[0]) == 40 and all(c in "0123456789abcdef" for c in head[0]):
            sha, line_no = head[0], int(head[2])
        elif line.startswith("committer-time ") and sha and line_no and set(sha) != {"0"}:
            times[line_no] = (int(line.split()[1]), sha[:7])
    return times


def _normalise(value):
    if isinstance(value, dict):
        return {k: _normalise(v) for k, v in value.items() if k != "updated"}
    if isinstance(value, list):
        return [_normalise(v) for v in value]
    return value


def _explicit(value) -> str | None:
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()):
        return value.strip()
    return None


def entry_dates(here: Path, models: dict, example_models: dict | None, performance: dict,
                catalog: Path, example: Path) -> dict[str, dict]:
    example_models = example_models or {}
    ranges: dict[str, list[tuple[int, int]]] = {}
    times: dict[int, tuple[int, str]] = {}
    if example.exists() and any(mid in example_models for mid in models):
        ranges = entry_line_ranges(example.read_text(encoding="utf-8", errors="replace"))
        times = blame_times(here, example.name)
    catalog_mtime = _mtime(catalog)
    out = {}
    for mid, entry in models.items():
        date, source = _explicit(entry.get("updated")), "catalog field"
        if date is None and mid in example_models and _normalise(example_models[mid]) == _normalise(entry):
            stamps = [times[n] for a, b in ranges.get(mid, []) for n in range(a, b + 1) if n in times]
            if stamps:
                stamp, short = max(stamps)
                date, source = _day(stamp), f"commit {short}"
        if date is None and catalog_mtime:
            date, source = _day(catalog_mtime), "models.toml saved"
        measured = (performance.get(mid) or {}).get("measured_at")
        if isinstance(measured, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", measured) and (date is None or measured > date):
            date, source = measured, "measured"
        out[mid] = {"updated": date, "updated_source": source if date else None}
    return out


def _fingerprint(here: Path, paths: list[Path]) -> str:
    head = (_git(here, "rev-parse", "HEAD") or "").strip()
    parts = [head]
    for path in paths:
        try:
            st = path.stat()
            parts.append(f"{path.name}:{st.st_mtime_ns}:{st.st_size}")
        except OSError:
            parts.append(f"{path.name}:-")
    return "|".join(parts)


def cached(here: Path, cache_path: Path, paths: list[Path], compute) -> dict:
    """compute() once per fingerprint; a broken cache is just recomputed."""
    key = _fingerprint(here, paths)
    try:
        data = json.loads(cache_path.read_text())
        if data.get("schema") == CACHE_SCHEMA and data.get("key") == key:
            return data["value"]
    except (OSError, ValueError, KeyError, AttributeError):
        pass
    value = compute()
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, tmp = tempfile.mkstemp(prefix=f".{cache_path.name}.", dir=cache_path.parent)
        with os.fdopen(fd, "w") as f:
            json.dump({"schema": CACHE_SCHEMA, "key": key, "value": value}, f)
        os.replace(tmp, cache_path)
    except OSError:
        pass
    return value
