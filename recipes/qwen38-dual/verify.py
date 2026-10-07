#!/usr/bin/env python3
"""Authenticate the reused NVIDIA snapshot without loading a model or CUDA."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import subprocess
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
    if (relative.is_absolute() or ".." in relative.parts or not relative.parts
            or str(relative) != filename or "\\" in filename):
        raise ValueError(f"invalid checkpoint manifest path: {filename!r}")
    path = root / relative
    allowed = [root.resolve()]
    # Reuse the existing HF cache: ordinary snapshot links point ../../blobs.
    # Other out-of-snapshot links are never accepted, even with matching bytes.
    if root.parent.name == "snapshots":
        allowed.append((root.parent.parent / "blobs").resolve())
    if not any(path.resolve().is_relative_to(base) for base in allowed):
        raise ValueError(f"checkpoint symlink escapes the snapshot and HF blobs: {filename!r}")
    return path


def manifest_files(manifest: dict) -> list[dict]:
    if not re.fullmatch(r"[0-9a-f]{40}", manifest.get("revision", "")):
        raise ValueError("checkpoint revision must be an immutable commit")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("checkpoint manifest requires files")
    names = set()
    for item in files:
        name, size = item.get("path"), item.get("size")
        if not isinstance(name, str) or name in names:
            raise ValueError("checkpoint manifest paths must be unique strings")
        names.add(name)
        checkpoint_path(Path("/manifest-validation"), name)
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("checkpoint file size must be a nonnegative integer")
        if item.get("lfs"):
            if not re.fullmatch(r"[0-9a-f]{64}", item["lfs"].get("sha256", "")):
                raise ValueError(f"missing LFS SHA-256: {name}")
        elif not re.fullmatch(r"[0-9a-f]{40}", item.get("git_blob_sha1", "")):
            raise ValueError(f"missing Git blob digest: {name}")
    return files


def git_blob_digest(path: Path, size: int) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path, manifest: dict, full_hash: bool = True) -> dict:
    if not full_hash:
        raise ValueError("Qwen dual preflight requires full checkpoint hashes")
    if root.name != manifest["revision"]:
        raise ValueError(f"expected pinned snapshot {manifest['revision']}, got {root}")
    total = 0
    for number, item in enumerate(manifest_files(manifest), start=1):
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
    actual = {str(path.relative_to(root)) for path in root.rglob("*")
              if path.is_file() and ".cache" not in path.relative_to(root).parts}
    if actual != files:
        raise ValueError(f"checkpoint inventory differs: extra={sorted(actual-files)}, missing={sorted(files-actual)}")
    weights = set(index.get("weight_map", {}).values())
    if not weights or not weights <= files or any(not name.endswith(".safetensors") for name in weights):
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
    parser.add_argument("--sha256", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("model-source.json"))
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--repair-cache", type=Path,
                        help="repair one invalid cached file, then fully verify (setup only)")
    parser.add_argument("--verify-runtime", action="store_true", help="authenticate installed runtime as well")
    args = parser.parse_args()
    try:
        raw = args.manifest.read_bytes()
        if (not re.fullmatch(r"[0-9a-f]{64}", args.expected_manifest_sha256)
                or hashlib.sha256(raw).hexdigest() != args.expected_manifest_sha256):
            raise ValueError("manifest SHA-256 differs from the controller pin")
        manifest = json.loads(raw)
        if args.verify_runtime:
            subprocess.run([sys.executable, str(Path(__file__).with_name("verify_runtime.py"))], check=True)
        result = (verify_with_repair(args.snapshot, manifest, args.repair_cache)
                  if args.repair_cache else verify(args.snapshot, manifest, True))
        result["manifest_sha256"] = args.expected_manifest_sha256
        print(json.dumps(result), flush=True)
    except Exception as exc:
        parser.exit(1, f"Qwen dual checkpoint preflight failed: {exc}\nRun ./spark-serve pull qwen38-dual --node both.\n")


if __name__ == "__main__":
    main()
