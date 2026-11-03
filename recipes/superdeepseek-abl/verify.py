#!/usr/bin/env python3
"""SuperDeepseek preflight: prepared snapshot and patchable vLLM. No GPU, no network.

spark-serve runs this inside the pinned image (python3, --network none, HF cache
read-only) on both Sparks before it may stop the current workload. It checks what
~/superdeepseek-abl/start.sh checked on the hosts, plus what entry.sh needs:
- the snapshot index (pinned by SHA-256) and every file it references exist;
- config.json, tokenizer.json and encoding/encoding_dsv4.py (pinned) exist;
- vLLM's DeepSeek V4 tokenizer holds the exact patch target once (or is patched);
- the vllm binary and the file entry.sh replaces exist.
Shards are not re-hashed (155 GiB); prepare.sh hard-linked them by LFS oid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath

DEFAULT_VLLM_PKG = "/usr/local/lib/python3.12/dist-packages/vllm"
DEFAULT_VLLM_BIN = "/usr/local/bin/vllm"
# Byte-identical to the strings in entry.sh (tests enforce it).
PATCH_OLD = '''elif reasoning_effort in ("max", "xhigh"):
                reasoning_effort = "max"
            else:
                reasoning_effort = "high'''
PATCH_NEW = '''elif reasoning_effort in ("max", "xhigh"):
                reasoning_effort = "max"
            elif reasoning_effort == "high":
                reasoning_effort = "high"
            else:
                reasoning_effort = "low'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_file(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if not relative.parts or relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"invalid snapshot path {name!r}")
    return root / relative


def check_snapshot(root: Path, index_sha: str | None, encoding_sha: str | None,
                   files: int | None) -> int:
    index = root / "model.safetensors.index.json"
    if not index.is_file():
        raise ValueError(f"model snapshot missing: {index} "
                         "(prepare it on sparkone: ~/superdeepseek-abl/prepare.sh --yes)")
    if index_sha and sha256(index) != index_sha:
        raise ValueError("model.safetensors.index.json does not match the pinned SHA-256")
    weight_map = json.loads(index.read_text()).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("index has no weight_map")
    names = sorted(set(weight_map.values()))
    missing = []
    for name in names:
        path = snapshot_file(root, name)
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(name)
    if missing:
        raise ValueError(f"index references {len(names)} files; missing or empty: {missing[:5]}")
    if files is not None and len(names) != files:
        raise ValueError(f"index references {len(names)} files, expected {files}")
    for name in ("config.json", "tokenizer.json", "encoding/encoding_dsv4.py"):
        if not snapshot_file(root, name).is_file():
            raise ValueError(f"snapshot is missing {name}")
    if encoding_sha and sha256(root / "encoding" / "encoding_dsv4.py") != encoding_sha:
        raise ValueError("encoding/encoding_dsv4.py does not match the pinned SHA-256")
    return len(names)


def check_vllm(pkg: Path, binary: Path) -> str:
    if not binary.is_file():
        raise ValueError(f"vllm binary missing: {binary}")
    replaced = pkg / "tokenizers" / "deepseek_v4_encoding.py"
    if not replaced.is_file():
        raise ValueError(f"vLLM file the tokenizer copy replaces is missing: {replaced}")
    tokenizer = pkg / "tokenizers" / "deepseek_v4.py"
    if not tokenizer.is_file():
        raise ValueError(f"vLLM DeepSeek V4 tokenizer missing: {tokenizer}")
    source = tokenizer.read_text()
    if PATCH_NEW in source:
        return "already patched"
    if source.count(PATCH_OLD) != 1:
        raise ValueError("DeepSeek V4 reasoning-effort patch target changed in this image")
    return "patch target found once"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SuperDeepseek preflight (no GPU, no network)")
    parser.add_argument("model_dir")
    parser.add_argument("--expected-index-sha256")
    parser.add_argument("--expected-encoding-sha256")
    parser.add_argument("--expected-files", type=int)
    parser.add_argument("--vllm-pkg", default=DEFAULT_VLLM_PKG)
    parser.add_argument("--vllm-bin", default=DEFAULT_VLLM_BIN)
    args = parser.parse_args(argv)
    try:
        count = check_snapshot(Path(args.model_dir), args.expected_index_sha256,
                               args.expected_encoding_sha256, args.expected_files)
        patch = check_vllm(Path(args.vllm_pkg), Path(args.vllm_bin))
    except (OSError, ValueError) as exc:
        print(f"superdeepseek preflight: {exc}", file=sys.stderr)
        return 2
    print(f"superdeepseek preflight ok: {count} indexed files present; tokenizer {patch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
