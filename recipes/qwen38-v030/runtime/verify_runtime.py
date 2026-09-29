#!/usr/bin/env python3
"""Check vendored sources and installed patched modules without GPU access."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def verify_upstream(here: Path) -> None:
    manifest = json.loads((here / "upstream-files.json").read_text())
    for name, item in manifest["files"].items():
        path = here / "upstream" / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise SystemExit(f"Vendored runtime integrity failure: {path}")
    vocab = [int(row) for row in (here / "upstream/draft_vocab_en_code_47k.txt").read_text().splitlines() if row.strip()]
    if len(vocab) != 47172 or len(set(vocab)) != 47172 or min(vocab) < 0 or max(vocab) >= 248320:
        raise SystemExit("Invalid pinned 47,172-token draft vocabulary")


def main() -> None:
    here = Path(__file__).resolve().parent
    verify_upstream(here)
    installed = json.loads((here / "installed-patches.json").read_text())
    if len(installed["files"]) != 4 or installed["vllm_version"] != "0.30.0":
        raise SystemExit("Invalid installed-patch receipt")
    for name, item in installed["files"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != item["sha256"]:
            raise SystemExit(f"Installed runtime integrity failure: {name}")
    print("Verified six upstream files, 47,172 draft ids and four patched vLLM modules")


if __name__ == "__main__":
    main()
