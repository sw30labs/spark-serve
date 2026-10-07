#!/usr/bin/env python3
"""Authenticate the dual-Spark patch and draft vocabulary without GPU access."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path


def verify_upstream(here: Path) -> None:
    manifest = json.loads((here / "upstream-files.json").read_text())
    if manifest["revision"] != "cd839d0b62f6c737178a0c2037733420e0524ebf":
        raise ValueError("Unexpected dual-Spark upstream revision")
    for name, item in manifest["files"].items():
        path = here / "upstream" / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Vendored runtime integrity failure: {path}")
    vocab = [int(row) for row in (here / "upstream/draft_vocab_en_code_47k.txt").read_text().splitlines() if row.strip()]
    if len(vocab) != 47149 or len(set(vocab)) != 47149 or min(vocab) < 0 or max(vocab) >= 248320:
        raise ValueError("Invalid pinned 47,149-token dual-Spark draft vocabulary")


def check_installed(here: Path) -> None:
    verify_upstream(here)
    installed = json.loads((here / "installed-patches.json").read_text())
    if (len(installed["files"]) != 1 or installed["vllm_version"] != "0.30.0"
            or importlib.metadata.version("vllm") != "0.30.0"):
        raise ValueError("Invalid installed-patch receipt or vLLM version")
    for name, item in installed["files"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Installed runtime integrity failure: {name}")


def main() -> None:
    check_installed(Path(__file__).resolve().parent)
    print("Verified dual-Spark TP-aware MTP patch and 47,149 draft ids")


if __name__ == "__main__":
    main()
