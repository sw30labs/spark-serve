#!/usr/bin/env python3
"""Build the exact pinned upstream patches, without importing CUDA/model code."""
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
    nvidia = site / "models/qwen4_exp/nvidia"
    targets = {"qsa.py": nvidia / "qsa.py", "ops/qsa.py": nvidia / "ops/qsa.py",
               "ngram_embedding.py": nvidia / "ngram_embedding.py", "mtp.py": nvidia / "mtp.py"}
    originals = {name: path.read_bytes() for name, path in targets.items()}
    with tempfile.TemporaryDirectory(prefix="spark-qwen-v030-") as temporary:
        work = Path(temporary)
        for name in ("patch_qsa_fp8_kv_v030.py", "patch_ple_mmap_v030.py",
                     "patch_mtp_draft_vocab_v030.py", "patch_mtp_draft_vocab.py"):
            shutil.copyfile(here / "upstream" / name, work / name)
        for name in ("qsa.py", "ops/qsa.py"):
            original = work / "v030_fp8kv/orig" / name
            original.parent.mkdir(parents=True, exist_ok=True)
            original.write_bytes(originals[name])
        (work / "v030_ple/orig").mkdir(parents=True)
        (work / "v030_ple/orig/ngram_embedding.py").write_bytes(originals["ngram_embedding.py"])
        (work / "mtp_v030_patched.py.orig").write_bytes(originals["mtp.py"])
        for name in ("patch_qsa_fp8_kv_v030.py", "patch_ple_mmap_v030.py", "patch_mtp_draft_vocab_v030.py"):
            subprocess.run([sys.executable, str(work / name)], check=True)
        generated = {
            "qsa.py": work / "v030_fp8kv/qsa.py",
            "ops/qsa.py": work / "v030_fp8kv/ops/qsa.py",
            "ngram_embedding.py": work / "v030_ple/ngram_embedding.py",
            "mtp.py": work / "mtp_v030_patched.py",
        }
        sources = {name: path.read_bytes() for name, path in generated.items()}
        for name, source in sources.items():
            ast.parse(source, filename=name)
        receipt = {"vllm_version": version, "files": {}}
        for name, source in sources.items():
            path = targets[name]
            path.write_bytes(source)
            # Do not leave a same-size/same-timestamp bytecode cache authoritative.
            for cache in (path.parent / "__pycache__").glob(path.stem + ".*.pyc"):
                cache.unlink()
            receipt["files"][str(path)] = {
                "original_sha256": hashlib.sha256(originals[name]).hexdigest(),
                "sha256": hashlib.sha256(source).hexdigest(),
            }
        (here / "installed-patches.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("Installed four vLLM modules from the three pinned patch generators")


if __name__ == "__main__":
    main()
