#!/usr/bin/env python3
"""Authenticate the vendored runtime sources without CUDA or model loading."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


def main() -> None:
    manifest = json.loads(Path(__file__).with_name("upstream-files.json").read_text())
    root = Path("/opt/glm53")
    for name, item in manifest["files"].items():
        path = root / name
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != item["sha256"]:
            raise SystemExit(f"Runtime integrity failure: {path}")
        if path.suffix == ".py":
            ast.parse(source, filename=str(path))
    print(f"Verified {len(manifest['files'])} runtime files at {manifest['revision']}")


if __name__ == "__main__":
    main()
