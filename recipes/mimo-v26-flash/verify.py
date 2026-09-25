#!/usr/bin/env python3
"""CPU-only check that a prepared MiMo snapshot matches its manifest."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def snapshot_files(root: Path):
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if ".cache" in relative.parts or not path.is_file() or path.is_symlink():
            continue
        yield path


def inventory(root: Path) -> dict:
    files = list(snapshot_files(root))
    return {
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "config_sha256": hashlib.sha256((root / "config.json").read_bytes()).hexdigest(),
        "dflash_config_sha256": hashlib.sha256((root / "dflash" / "config.json").read_bytes()).hexdigest(),
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify.py SNAPSHOT")
    root = Path(sys.argv[1])
    manifest_path = root.parent / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing {manifest_path}; run spark-serve pull mimo26")
    manifest = json.loads(manifest_path.read_text())
    config = json.loads((root / "config.json").read_text())
    dflash = json.loads((root / "dflash" / "config.json").read_text())
    if "MiMoV2ForCausalLM" not in config.get("architectures", []):
        raise SystemExit(f"unexpected architectures: {config.get('architectures')}")
    if dflash.get("dflash_config", {}).get("block_size") != 8:
        raise SystemExit("dflash block_size is not 8")
    if not (root / "dflash" / "dflash_draft_model.safetensors").is_file():
        raise SystemExit("dflash drafter weights are missing")
    if not list(root.glob("*.safetensors")) and not (root / "model.safetensors.index.json").is_file():
        raise SystemExit("model shards are missing")
    got = inventory(root)
    for key in ("files", "bytes", "config_sha256", "dflash_config_sha256"):
        if got[key] != manifest[key]:
            raise SystemExit(f"{key} mismatch: snapshot {got[key]} manifest {manifest[key]}")
    print(json.dumps({"ok": True, "revision": manifest.get("revision"), **got}))


if __name__ == "__main__":
    main()
