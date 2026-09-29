#!/usr/bin/env python3
"""Authenticate both pinned EXL3 checkpoints; no model imports or GPU work."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath


class CheckpointFileError(ValueError):
    pass


def checkpoint_path(root: Path, filename: str) -> Path:
    relative = PurePosixPath(filename)
    if (not relative.parts or relative.is_absolute() or ".." in relative.parts
            or str(relative) != filename or "\\" in filename):
        raise ValueError(f"invalid checkpoint manifest path: {filename!r}")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"checkpoint path escapes its directory: {filename!r}")
    return path


def manifest_files(checkpoint: dict) -> list[dict]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", checkpoint.get("model", "")):
        raise ValueError("checkpoint requires a public Hugging Face model ID")
    if not re.fullmatch(r"[0-9a-f]{40}", checkpoint.get("revision", "")):
        raise ValueError("checkpoint revision must be an immutable commit")
    files = checkpoint.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("checkpoint manifest must contain files")
    names = set()
    for item in files:
        name, size = item.get("path"), item.get("size")
        if not isinstance(name, str) or name in names:
            raise ValueError("manifest paths must be unique strings")
        checkpoint_path(Path("/manifest-validation"), name)
        names.add(name)
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"invalid checkpoint size: {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", "")):
            raise ValueError(f"manifest lacks SHA-256: {name}")
    if "config.json" not in names or not any(name.endswith(".safetensors") for name in names):
        raise ValueError("manifest needs config.json and weight shards")
    return files


def checkpoints(manifest: dict) -> dict:
    models = manifest.get("checkpoints")
    if manifest.get("version") != 1 or not isinstance(models, dict) or set(models) != {"target", "draft"}:
        raise ValueError("manifest must pin target and draft checkpoints")
    for model in models.values():
        manifest_files(model)
    return models


def snapshot_path(root: Path, role: str, checkpoint: dict) -> Path:
    if role not in {"target", "draft"}:
        raise ValueError("invalid checkpoint role")
    manifest_files(checkpoint)
    return checkpoint_path(root, f"models/{role}/{checkpoint['revision']}")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(root: Path, item: dict) -> None:
    path = checkpoint_path(root, item["path"])
    if not path.is_file() or path.stat().st_size != item["size"]:
        raise CheckpointFileError(f"missing or incomplete checkpoint file: {path}")
    # Deliberately hash large shards at startup too. A same-size corrupt copy
    # must never be accepted merely because a previous preparation succeeded.
    if file_digest(path) != item["sha256"]:
        raise CheckpointFileError(f"SHA-256 mismatch: {path}")


def verify_metadata(snapshot: Path, checkpoint: dict) -> None:
    names = {item["path"] for item in manifest_files(checkpoint)}
    actual = {str(path.relative_to(snapshot)) for path in snapshot.rglob("*")
              if path.is_file() and ".cache" not in path.relative_to(snapshot).parts
              and not path.name.endswith(".incomplete")}
    if actual != names:
        raise ValueError(f"checkpoint inventory differs from manifest: extra={sorted(actual - names)}, missing={sorted(names - actual)}")
    config = json.loads((snapshot / "config.json").read_text())
    if config.get("architectures") != checkpoint["architectures"]:
        raise ValueError("checkpoint architecture differs from manifest")
    if "model.safetensors.index.json" in names:
        index = json.loads((snapshot / "model.safetensors.index.json").read_text())
        shards = set(index.get("weight_map", {}).values())
        if not shards or not shards <= names or any(not name.endswith(".safetensors") for name in shards):
            raise ValueError("weight index references unverified shards")
    elif "model.safetensors" not in names:
        raise ValueError("checkpoint requires an authenticated weight index")
    for name in names:
        if name.endswith(".json"):
            json.loads(checkpoint_path(snapshot, name).read_text())


def verify(root: Path, manifest: dict) -> dict:
    results = {}
    for role, checkpoint in checkpoints(manifest).items():
        snapshot = snapshot_path(root, role, checkpoint)
        files = checkpoint["files"]
        for number, item in enumerate(files, 1):
            print(f"Verifying {role} {number}/{len(files)}: {item['path']}", file=sys.stderr, flush=True)
            verify_file(snapshot, item)
        verify_metadata(snapshot, checkpoint)
        results[role] = {"model": checkpoint["model"], "revision": checkpoint["revision"],
                         "files": len(files), "bytes": sum(item["size"] for item in files)}
    return {"checkpoints": results, "full_hash_verified": True}


def load_manifest(path: Path, expected_sha256: str | None = None) -> tuple[dict, str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256) or digest != expected_sha256:
            raise ValueError("checkpoint manifest SHA-256 does not match controller pin")
    manifest = json.loads(raw)
    checkpoints(manifest)
    return manifest, digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("model-source.json"))
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()
    try:
        manifest, digest = load_manifest(args.manifest, args.expected_manifest_sha256)
        report = verify(args.root, manifest)
        report["manifest_sha256"] = digest
        print(json.dumps(report), flush=True)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"GLM EXL3 preflight failed: {exc}\nRun ./spark-serve pull glm53-exl3.\n")


if __name__ == "__main__":
    main()
