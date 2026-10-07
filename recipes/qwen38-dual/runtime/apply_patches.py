#!/usr/bin/env python3
"""Install the pinned dual-Spark, TP-aware MTP patch without loading CUDA."""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from verify_runtime import verify_upstream


def main() -> None:
    here = Path(__file__).resolve().parent
    verify_upstream(here)
    version = importlib.metadata.version("vllm")
    if version != "0.30.0":
        raise SystemExit(f"Expected vLLM 0.30.0, found {version}")
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("Cannot locate the pinned vLLM package")
    site = Path(next(iter(spec.submodule_search_locations)))
    target = site / "models/qwen4_exp/nvidia/mtp.py"
    original = target.read_bytes()
    with tempfile.TemporaryDirectory(prefix="spark-qwen-dual-") as temporary:
        work = Path(temporary)
        for name in ("patch_mtp_draft_vocab.py", "patch_mtp_draft_vocab_v030.py"):
            shutil.copyfile(here / "upstream" / name, work / name)
        (work / "mtp_v030_patched.py.orig").write_bytes(original)
        subprocess.run([sys.executable, str(work / "patch_mtp_draft_vocab_v030.py")], check=True)
        source = (work / "mtp_v030_patched.py").read_bytes()
        ast.parse(source, filename=str(target))
        target.write_bytes(source)
        for cache in (target.parent / "__pycache__").glob(target.stem + ".*.pyc"):
            cache.unlink()
    receipt = {"vllm_version": version, "files": {str(target): {
        "original_sha256": hashlib.sha256(original).hexdigest(),
        "sha256": hashlib.sha256(source).hexdigest()}}}
    (here / "installed-patches.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("Installed the TP-aware dual-Spark MTP module")


if __name__ == "__main__":
    main()
