#!/usr/bin/env python3
"""Authenticate the installed TensorFold package and the exact upstream patch set."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path


def verify_files(root: Path, expected: dict[str, str]) -> None:
    actual = {str(p.relative_to(root)) for p in root.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts}
    if actual != set(expected):
        raise ValueError(f"runtime inventory differs: extra={sorted(actual-set(expected))}, missing={sorted(set(expected)-actual)}")
    for name, digest in expected.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"runtime path escapes package: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"installed runtime SHA-256 mismatch: {name}")


def main() -> None:
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / "installed-runtime.json").read_text())
    distribution = importlib.metadata.distribution("tensorfold")
    if distribution.version != manifest["tensorfold_version"]:
        raise ValueError("TensorFold version differs from the pinned runtime")
    # Every installed file was extracted from authenticated OCI image layers
    # and compared with TensorFold v0.6.0 plus the 82 ordered upstream patches.
    verify_files(Path(distribution.locate_file("tensorfold")), manifest["files"])
    upstream = json.loads((here / "upstream-files.json").read_text())
    verify_files(here / "upstream", upstream["files"])
    patches = sorted((here / "upstream").glob("*.patch"))
    payload = b"".join(p.read_bytes() for p in patches) + (manifest["image_extras"] + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()[:12]
    if len(patches) != manifest["patch_count"] or digest != manifest["patches_hash"]:
        raise ValueError("TensorFold recipe patch set differs")
    for name, version in manifest["dependencies"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"TensorFold {name} differs from the pinned runtime")
    print(f"Verified TensorFold {distribution.version}: {len(manifest['files'])} installed files, {len(patches)} patches")


if __name__ == "__main__":
    main()
