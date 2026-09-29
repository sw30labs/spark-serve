#!/usr/bin/env python3
"""Authenticate the pinned NGC checkpoint without importing a model or using CUDA."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath

SMALL_FILE_LIMIT = 32 * 1024 * 1024


class CheckpointFileError(ValueError):
    def __init__(self, filename: str, message: str):
        self.filename = filename
        super().__init__(message)


def checkpoint_path(root: Path, filename: str) -> Path:
    relative = PurePosixPath(filename)
    if (relative.is_absolute() or not relative.parts or ".." in relative.parts
            or str(relative) != filename):
        raise ValueError(f"invalid checkpoint manifest path: {filename!r}")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"checkpoint path escapes its pinned directory: {filename!r}")
    return path


def manifest_files(manifest: dict) -> list[dict]:
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("checkpoint manifest must contain files")
    names = set()
    for item in files:
        name, size = item.get("path"), item.get("size")
        if not isinstance(name, str) or name in names:
            raise ValueError("checkpoint manifest paths must be unique strings")
        # Validate the path even if no files have been downloaded yet.
        relative = PurePosixPath(name)
        if (relative.is_absolute() or not relative.parts or ".." in relative.parts
                or str(relative) != name):
            raise ValueError(f"invalid checkpoint manifest path: {name!r}")
        names.add(name)
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"invalid checkpoint size: {name}")
        digest = item.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"manifest lacks a valid authenticated digest: {name}")
    return files


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(root: Path, item: dict, *, full_hash: bool) -> None:
    path = checkpoint_path(root, item["path"])
    if not path.is_file() or path.stat().st_size != item["size"]:
        raise CheckpointFileError(item["path"], f"missing or incomplete checkpoint file: {path}")
    if full_hash or item["size"] <= SMALL_FILE_LIMIT:
        # NGC publishes SHA-256 for every file, including the large shards.
        # Fast startup hashes metadata and checks every shard's pinned length.
        if file_digest(path) != item["sha256"]:
            raise CheckpointFileError(item["path"], f"SHA-256 mismatch: {item['path']}")


def verify(root: Path, manifest: dict, full_hash: bool = False) -> dict:
    if root.name != manifest["revision"]:
        raise ValueError(f"expected pinned checkpoint {manifest['revision']}, got {root}")
    files = manifest_files(manifest)
    for number, item in enumerate(files, 1):
        if full_hash:
            print(f"Verifying {number}/{len(files)}: {item['path']}", file=sys.stderr, flush=True)
        verify_file(root, item, full_hash=full_hash)
    config = json.loads((root / "config.json").read_text())
    if config.get("architectures") != manifest["architectures"]:
        raise ValueError("checkpoint architecture does not match the pinned GLM recipe")
    names = {item["path"] for item in files}
    index = json.loads((root / "model.safetensors.index.json").read_text())
    weights = set(index.get("weight_map", {}).values())
    if not weights or not weights <= names or any(not name.endswith(".safetensors") for name in weights):
        raise ValueError("checkpoint index references unverified weight shards")
    for name in manifest.get("json_metadata", ["tokenizer_config.json", "generation_config.json", "tokenizer.json"]):
        if name not in names:
            raise ValueError(f"required serving metadata absent from manifest: {name}")
        json.loads(checkpoint_path(root, name).read_text())
    return {"model": manifest["model"], "revision": manifest["revision"],
            "files": len(files), "bytes": sum(item["size"] for item in files),
            "full_hash_verified": full_hash, "architectures": config["architectures"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--full-hash", action="store_true", help="authenticate all shards against published digests")
    parser.add_argument("--sha256", dest="full_hash", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("model-source.json"))
    parser.add_argument("--expected-manifest-sha256", help="require the controller's pinned manifest digest before inspecting assets")
    args = parser.parse_args()
    try:
        raw = args.manifest.read_bytes()
        manifest_sha256 = hashlib.sha256(raw).hexdigest()
        if args.expected_manifest_sha256 is not None:
            if not re.fullmatch(r"[0-9a-f]{64}", args.expected_manifest_sha256):
                raise ValueError("expected manifest SHA-256 must be 64 lowercase hexadecimal characters")
            if manifest_sha256 != args.expected_manifest_sha256:
                raise ValueError("checkpoint manifest SHA-256 does not match the controller pin")
        result = verify(args.snapshot, json.loads(raw), args.full_hash)
        result["manifest_sha256"] = manifest_sha256
        print(json.dumps(result), flush=True)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"GLM checkpoint preflight failed: {exc}\nRun ./spark-serve pull glm53 to prepare both Sparks.\n")


if __name__ == "__main__":
    main()
