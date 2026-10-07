#!/usr/bin/env python3
"""Resume pinned public Hugging Face files and write the slim Engram index."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import http.client
import json
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from prepare_engram_src import KEEP_SUFFIXES, slim_weight_map
from verify import (CheckpointFileError, checkpoint_path, checkpoints, file_digest,
                    load_manifest, snapshot_path, verify_file, verify_metadata)


def download_file(root: Path, item: dict, base_url: str, *, opener=urllib.request.urlopen,
                  pause=time.sleep, attempts=5) -> None:
    target = checkpoint_path(root, item["path"])
    if target.is_symlink():
        raise ValueError(f"checkpoint file must not be a symlink: {target}")
    if target.is_file():
        try:
            verify_file(root, item)
            print(f"Already verified: {item['path']}", flush=True)
            return
        except CheckpointFileError:
            pass
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".incomplete")
    if partial.is_symlink():
        raise ValueError(f"partial download must not be a symlink: {partial}")
    url = base_url.rstrip("/") + "/" + urllib.parse.quote(item["path"], safe="/")
    for attempt in range(attempts):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > item["size"]:
            partial.unlink()
            offset = 0
        try:
            if offset < item["size"]:
                headers = {"User-Agent": "spark-serve-dsv41-exl3/1", "Accept-Encoding": "identity"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                print(f"Downloading {item['path']} from byte {offset}/{item['size']}", flush=True)
                request = urllib.request.Request(url, headers=headers)
                with opener(request, timeout=90) as response:
                    status = response.status
                    if status == 206:
                        content_range = response.headers.get("Content-Range", "")
                        expected = f"bytes {offset}-"
                        if not content_range.startswith(expected) or not content_range.endswith(f"/{item['size']}"):
                            raise ValueError(f"unexpected resume range for {item['path']}: {content_range}")
                    elif status == 200:
                        offset = 0
                    else:
                        raise ValueError(f"unexpected download status {status}")
                    mode = "ab" if offset else "wb"
                    with partial.open(mode) as destination:
                        received = offset
                        last_progress = time.monotonic()
                        while chunk := response.read(4 * 1024 * 1024):
                            received += len(chunk)
                            if received > item["size"]:
                                raise ValueError(f"download exceeds pinned size: {item['path']}")
                            destination.write(chunk)
                            if time.monotonic() - last_progress >= 30:
                                print(f"Progress {item['path']}: {received}/{item['size']} bytes", flush=True)
                                last_progress = time.monotonic()
                        destination.flush()
                        os.fsync(destination.fileno())
                if partial.stat().st_size != item["size"]:
                    raise OSError(f"incomplete download: {item['path']}")
            partial_item = dict(item, path=str(partial.relative_to(root)))
            try:
                verify_file(root, partial_item)
            except CheckpointFileError:
                partial.unlink()
                raise
            os.replace(partial, target)
            parent = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
            return
        except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException) as exc:
            if attempt + 1 == attempts:
                raise
            print(f"Retry {attempt + 1}/{attempts - 1} for {item['path']}: {type(exc).__name__}: {exc}", flush=True)
            pause(min(2 ** attempt, 16))


def slim_index_bytes(raw: bytes, keep_suffixes: tuple[str, ...]) -> bytes:
    if tuple(keep_suffixes) != KEEP_SUFFIXES:
        raise ValueError("Engram index suffixes differ from the pinned slimming script")
    index = json.loads(raw)
    keep = slim_weight_map(index.get("weight_map") or {})
    if not keep:
        raise ValueError("native Engram index has no embed tables")
    unexpected = sorted({shard for shard in keep.values() if shard not in {
        "model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors"}})
    if unexpected:
        raise ValueError(f"Engram embed tables are outside shards 47 and 48: {unexpected}")
    slim = {
        "metadata": {
            "total_size": (index.get("metadata") or {}).get("total_size"),
            "dsv41_engram_src": "embed-only",
        },
        "weight_map": keep,
    }
    return (json.dumps(slim, indent=2) + "\n").encode()


def write_derived_index(snapshot: Path, checkpoint: dict, *, opener=urllib.request.urlopen) -> None:
    derived = checkpoint.get("derived_index")
    if not isinstance(derived, dict):
        raise ValueError("engram checkpoint requires a derived index")
    path_name = derived.get("path")
    matches = [entry for entry in checkpoint["files"] if entry["path"] == path_name]
    if len(matches) != 1:
        raise ValueError("derived Engram index is not a served file")
    item = matches[0]
    target = checkpoint_path(snapshot, path_name)
    if target.is_file() and not target.is_symlink():
        try:
            verify_file(snapshot, item)
            print(f"Already verified: {path_name}", flush=True)
            return
        except CheckpointFileError:
            pass
    upstream_size = derived.get("upstream_size")
    upstream_sha = derived.get("upstream_sha256")
    if isinstance(upstream_size, bool) or not isinstance(upstream_size, int) or upstream_size < 1:
        raise ValueError("derived Engram index lacks an upstream size")
    url = (f"https://huggingface.co/{checkpoint['model']}/resolve/{checkpoint['revision']}/"
           + urllib.parse.quote(path_name, safe="/"))
    print(f"Downloading native Engram index to slim ({upstream_size} bytes)", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "spark-serve-dsv41-exl3/1", "Accept-Encoding": "identity"})
    with opener(request, timeout=180) as response:
        if response.status != 200:
            raise ValueError(f"unexpected Engram index status {response.status}")
        raw = response.read()
    if len(raw) != upstream_size or hashlib.sha256(raw).hexdigest() != upstream_sha:
        raise ValueError("native Engram index does not match the pinned upstream hash")
    body = slim_index_bytes(raw, tuple(derived.get("keep_suffixes") or ()))
    if len(body) != item["size"] or hashlib.sha256(body).hexdigest() != item["sha256"]:
        raise ValueError("slim Engram index does not match the pinned served hash")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
        temporary.write(body)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_name = temporary.name
    os.replace(temporary_name, target)
    if file_digest(target) != item["sha256"]:
        raise CheckpointFileError(f"SHA-256 mismatch: {target}")


def download(root: Path, manifest: dict, workers: int = 4, *, opener=urllib.request.urlopen) -> dict:
    models = checkpoints(manifest)
    if workers < 1 or workers > 8:
        raise ValueError("download workers must be between 1 and 8")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".download.lock").open("a+") as lock:
        deadline = time.monotonic() + 8 * 3600
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise ValueError("another DeepSeek V4.1 EXL3 download still holds the preparation lock") from None
                print("Waiting for the existing DeepSeek V4.1 EXL3 download to finish", flush=True)
                time.sleep(30)
        result = {}
        for role, model in models.items():
            snapshot = snapshot_path(root, role, model)
            snapshot.mkdir(parents=True, exist_ok=True)
            derived_path = (model.get("derived_index") or {}).get("path") if role == "engram" else None
            base_url = f"https://huggingface.co/{model['model']}/resolve/{model['revision']}"
            pending = [item for item in model["files"] if item["path"] != derived_path]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(download_file, snapshot, item, base_url, opener=opener) for item in pending]
                for future in futures:
                    future.result()
            if role == "engram":
                write_derived_index(snapshot, model, opener=opener)
            verify_metadata(snapshot, model)
            result[role] = {
                "model": model["model"],
                "revision": model["revision"],
                "files": len(model["files"]),
                "bytes": sum(item["size"] for item in model["files"]),
            }
        return {"checkpoints": result, "full_hash_verified": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("model-source.json"))
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    try:
        manifest, digest = load_manifest(args.manifest, args.expected_manifest_sha256)
        result = download(args.root, manifest, args.workers)
        result["manifest_sha256"] = digest
        print(json.dumps(result), flush=True)
    except (OSError, ValueError, KeyError, TypeError, http.client.HTTPException) as exc:
        parser.exit(1, f"DeepSeek V4.1 EXL3 download failed: {exc}\nRerun preparation to resume partial downloads.\n")


if __name__ == "__main__":
    main()
