#!/usr/bin/env python3
"""Allow resumable downloads while keeping room for one replacement shard."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from verify import checkpoint_path, manifest_files


def required_bytes(snapshot: Path, manifest: dict) -> int:
    missing = 0
    files = manifest_files(manifest)
    for item in files:
        path = checkpoint_path(snapshot, item["path"])
        # HF may have downloaded blobs before snapshot links were published.
        digest = (item.get("lfs") or {}).get("sha256") or item.get("git_blob_sha1")
        blob = snapshot.parent.parent / "blobs" / digest
        if not any(p.is_file() and p.stat().st_size == item["size"] for p in (path, blob)):
            missing += item["size"]
    return missing + max(item["size"] for item in files) + 2_000_000_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("model-source.json"))
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    needed = required_bytes(args.snapshot, json.loads(args.manifest.read_text()))
    free = shutil.disk_usage(args.cache_dir).free
    if free < needed:
        parser.exit(1, f"TensorFold checkpoint needs {needed / 1e9:.1f} GB free for remaining files and repair headroom; {free / 1e9:.1f} GB available.\n")


if __name__ == "__main__":
    main()
