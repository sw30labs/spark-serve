#!/usr/bin/env python3
"""Leave room for missing pinned files and one replacement shard during preparation."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from verify import checkpoint_path, checkpoints, load_manifest, snapshot_path


def required_bytes(root: Path, manifest: dict) -> int:
    missing = 0
    largest = 0
    for role, model in checkpoints(manifest).items():
        snapshot = snapshot_path(root, role, model)
        for item in model["files"]:
            largest = max(largest, item["size"])
            path = checkpoint_path(snapshot, item["path"])
            if path.is_file() and path.stat().st_size == item["size"]:
                continue
            partial = path.with_name(path.name + ".incomplete")
            if partial.is_symlink():
                raise ValueError("partial download must not be a symlink")
            held = partial.stat().st_size if partial.is_file() else 0
            missing += max(0, item["size"] - held)
    return missing + largest + 2_000_000_000


def check_space(root: Path, manifest: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    need = required_bytes(root, manifest)
    free = shutil.disk_usage(root).free
    if free < need:
        raise ValueError(f"GLM TensorFold checkpoint preparation needs {need:,} free bytes under {root}; {free:,} available")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("model-source.json"))
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()
    try:
        manifest, _ = load_manifest(args.manifest, args.expected_manifest_sha256)
        check_space(args.root, manifest)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"GLM TensorFold disk check failed: {exc}\n")


if __name__ == "__main__":
    main()
