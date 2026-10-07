#!/usr/bin/env python3
"""List only the pinned snapshot and its referenced HF blobs for rsync."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from verify import checkpoint_path, manifest_files


def transfer_files(snapshot: Path, manifest: dict) -> list[str]:
    if snapshot.name != manifest["revision"] or snapshot.parent.name != "snapshots":
        raise ValueError("transfer must use the pinned Hugging Face snapshot")
    repo = snapshot.parent.parent
    names = set()
    for item in manifest_files(manifest):
        path = checkpoint_path(snapshot, item["path"])
        if not path.is_file():
            raise ValueError(f"missing transfer source: {path}")
        names.add(str(path.relative_to(repo)))
        if path.is_symlink():
            blob = path.resolve()
            blob_name = str(blob.relative_to((repo / "blobs").resolve()))
            # Preserve portable, ordinary HF links. Absolute head-only pointers
            # must never be copied to a different worker cache path.
            expected = Path("../../blobs") / blob_name
            if path.readlink() != expected:
                raise ValueError(f"nonportable Hugging Face snapshot link: {path}")
            names.add("blobs/" + blob_name)
    return sorted(names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()
    raw = Path(__file__).with_name("model-source.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.expected_manifest_sha256:
        raise ValueError("transfer manifest differs from the controller pin")
    args.output.write_text("\n".join(transfer_files(args.snapshot, json.loads(raw))) + "\n")


if __name__ == "__main__":
    main()
