#!/usr/bin/env python3
"""CPU-only checkpoint preflight. Full SHA-256 verification is optional at setup."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath


class CheckpointFileError(ValueError):
    """A specific cached file is missing, incomplete, or corrupt."""

    def __init__(self, filename: str, message: str):
        self.filename = filename
        super().__init__(message)


def checkpoint_path(root: Path, filename: str) -> Path:
    # Snapshot files may symlink into the Hub blob cache, but manifest names
    # must remain relative repository paths.
    relative = PurePosixPath(filename)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"invalid checkpoint manifest path: {filename!r}")
    return root / relative


def git_blob_digest(path: Path, size: int) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path, manifest: dict, full_hash: bool = False) -> dict:
    if root.name != manifest["revision"]:
        raise ValueError(f"expected pinned snapshot {manifest['revision']}, got {root}")
    total = 0
    for number, item in enumerate(manifest["files"], start=1):
        if full_hash:
            print(f"Verifying {number}/{len(manifest['files'])}: {item['path']}",
                  file=sys.stderr, flush=True)
        path = checkpoint_path(root, item["path"])
        if not path.is_file() or path.stat().st_size != item["size"]:
            raise CheckpointFileError(item["path"], f"missing or incomplete checkpoint file: {path}")
        total += item["size"]
        digest = (item.get("lfs") or {}).get("sha256")
        if full_hash and digest:
            with path.open("rb") as source:
                actual = hashlib.file_digest(source, "sha256").hexdigest()
            if actual != digest:
                raise CheckpointFileError(item["path"], f"SHA-256 mismatch: {path.name}")
        elif not item.get("lfs"):
            # Git uses SHA-1 over its blob header plus bytes, rather than a
            # plain file checksum. These small serving files are cheap to
            # authenticate even during the ordinary startup preflight.
            expected = item.get("git_blob_sha1")
            if not expected:
                raise ValueError(f"manifest lacks Git blob digest: {item['path']}")
            if git_blob_digest(path, item["size"]) != expected:
                raise CheckpointFileError(item["path"], f"Git blob checksum mismatch: {path.name}")
        elif full_hash:
            raise ValueError(f"manifest lacks SHA-256 digest: {item['path']}")
    config = json.loads((root / "config.json").read_text())
    index = json.loads((root / "model.safetensors.index.json").read_text())
    files = {item["path"] for item in manifest["files"]}
    if not index.get("weight_map") or not set(index["weight_map"].values()) <= files:
        raise ValueError("checkpoint index references unverified shards")
    # Parse the serving metadata as well as checking its exact published size.
    for name in ("tokenizer_config.json", "preprocessor_config.json", "hf_quant_config.json"):
        json.loads((root / name).read_text())
    return {"model": manifest["model"], "revision": manifest["revision"],
            "files": len(files), "bytes": total, "sha256_verified": full_hash,
            "architectures": config.get("architectures")}


def verify_with_repair(root: Path, manifest: dict, cache_dir: Path, download=None) -> dict:
    """Force-download one failed file once, then verify the complete snapshot."""
    expected = cache_dir / ("models--" + manifest["model"].replace("/", "--")) / "snapshots" / manifest["revision"]
    if root.resolve() != expected.resolve():
        raise ValueError("repair cache does not contain the requested pinned snapshot")
    try:
        return verify(root, manifest, full_hash=True)
    except CheckpointFileError as exc:
        if download is None:
            from huggingface_hub import hf_hub_download
            download = hf_hub_download
        print(f"Repairing cached checkpoint file: {exc.filename}", file=sys.stderr, flush=True)
        download(repo_id=manifest["model"], filename=exc.filename,
                 revision=manifest["revision"], cache_dir=str(cache_dir),
                 force_download=True, token=False)
    # A second failure exits clearly instead of repeatedly redownloading or
    # accepting a repaired file without checking the remaining snapshot.
    return verify(root, manifest, full_hash=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--sha256", action="store_true")
    parser.add_argument("--repair-cache", type=Path,
                        help="repair one invalid cached file, then fully verify (setup only)")
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name("model-source.json").read_text())
    try:
        result = (verify_with_repair(args.snapshot, manifest, args.repair_cache)
                  if args.repair_cache else verify(args.snapshot, manifest, args.sha256))
        print(json.dumps(result), flush=True)
    except Exception as exc:
        parser.exit(1, f"Qwen checkpoint preflight failed: {exc}\nRun ./spark-serve pull qwen38 to prepare it.\n")


if __name__ == "__main__":
    main()
