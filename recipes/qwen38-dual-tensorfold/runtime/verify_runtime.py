#!/usr/bin/env python3
"""Authenticate the installed native TensorFold tree and the bundled upstream notices."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


def verify_files(root: Path, expected: dict[str, str]) -> None:
    actual = {str(p.relative_to(root)) for p in root.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts}
    if actual != set(expected):
        raise ValueError(f"runtime inventory differs: extra={sorted(actual-set(expected))}, missing={sorted(set(expected)-actual)}")
    for name, digest in expected.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"runtime path escapes its root: {name}")
        with path.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != digest:
                raise ValueError(f"installed runtime SHA-256 mismatch: {name}")


def main() -> None:
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / "installed-runtime.json").read_text())
    root = Path(manifest["root"])
    # Every file was extracted from the authenticated TensorFold image layers.
    # The Python vision helper also matches TensorFold db28187 plus the recipe
    # patches; the Zig binary and kernel set are pinned by image digest only.
    verify_files(root, manifest["files"])
    for link, target in manifest["symlinks"].items():
        if not os.path.islink(link) or os.readlink(link) != target:
            raise ValueError(f"TensorFold link differs: {link}")
    for name in manifest["executables"]:
        if not os.access(root / name, os.X_OK):
            raise ValueError(f"TensorFold executable lost its mode: {name}")
    if len(manifest["patches_sha256"]) != manifest["patch_count"]:
        raise ValueError("TensorFold recipe patch inventory differs")
    upstream = json.loads((here / "upstream-files.json").read_text())
    verify_files(here / "upstream", upstream["files"])
    for name, version in manifest["dependencies"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"TensorFold {name} differs from the pinned runtime")
    print(f"Verified native TensorFold {manifest['tensorfold_revision'][:7]}-{manifest['patches_hash']}: "
          f"{len(manifest['files'])} installed files")


if __name__ == "__main__":
    main()
