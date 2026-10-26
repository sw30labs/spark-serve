#!/usr/bin/env python3
"""CPU-only preflight for the Flint NVFP4 checkpoint and its config/tokenizer snapshot.

Checks that both pinned Hub snapshots are complete. Small files are always
authenticated by Git blob SHA-1; with --full every LFS file is also SHA-256
checked (the 22.9 GB GGUF takes a few minutes). No network, no GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath

MANIFEST = Path(__file__).with_name("model-source.json")


def git_blob_digest(path: Path, size: int) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verify_snapshot(root: Path, section: dict, full: bool) -> int:
    if root.name != section["revision"]:
        raise ValueError(f"{section['model']}: expected snapshot {section['revision']}, got {root}")
    total = 0
    for item in section["files"]:
        rel = PurePosixPath(item["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"invalid manifest path: {item['path']!r}")
        path = root / rel
        if not path.is_file() or path.stat().st_size != item["size"]:
            raise ValueError(f"missing or incomplete: {path}")
        total += item["size"]
        lfs = item.get("lfs")
        if lfs:
            if full and sha256(path) != lfs["sha256"]:
                raise ValueError(f"SHA-256 mismatch: {path}")
        elif git_blob_digest(path, item["size"]) != item["git_blob_sha1"]:
            raise ValueError(f"Git blob checksum mismatch: {path}")
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="flint-nvfp4-gguf snapshot directory")
    parser.add_argument("config", type=Path, help="RecursiveSI/flint snapshot (config/tokenizer only)")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--full", action="store_true", help="SHA-256 every LFS file")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    try:
        size = verify_snapshot(args.checkpoint, manifest["checkpoint"], args.full)
        size += verify_snapshot(args.config, manifest["config_and_tokenizer"], args.full)
    except ValueError as error:
        print(f"preflight failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "bytes": size, "full_hash": args.full}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
