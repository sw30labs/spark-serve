#!/usr/bin/env python3
"""Resume pinned public Hugging Face files; authenticate before publication."""
from __future__ import annotations

import argparse
import fcntl
import http.client
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from verify import (CheckpointFileError, checkpoint_path, checkpoints, load_manifest,
                    snapshot_path, verify_file, verify_metadata)


def download_file(root: Path, item: dict, base_url: str, *, opener=urllib.request.urlopen,
                  pause=time.sleep, attempts=5) -> None:
    target = checkpoint_path(root, item["path"])
    if target.is_file():
        try:
            verify_file(root, item)
            print(f"Already verified: {item['path']}", flush=True)
            return
        except CheckpointFileError:
            # Keep the old file until the replacement has been authenticated.
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
                headers = {"User-Agent": "spark-serve-glm53-exl3/1", "Accept-Encoding": "identity"}
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
                        # A server may ignore Range. Replace the partial contents;
                        # never append a complete response to an existing prefix.
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


def download(root: Path, manifest: dict, workers: int = 4) -> dict:
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
                    raise ValueError("another EXL3 download still holds the preparation lock") from None
                print("Waiting for the existing EXL3 download to finish", flush=True)
                time.sleep(30)
        result = {}
        for role, model in models.items():
            snapshot = snapshot_path(root, role, model)
            snapshot.mkdir(parents=True, exist_ok=True)
            # URL is assembled only from validated model IDs and immutable revisions.
            base_url = f"https://huggingface.co/{model['model']}/resolve/{model['revision']}"
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(download_file, snapshot, item, base_url) for item in model["files"]]
                for future in futures:
                    future.result()
            # Each file was already SHA-256 verified, including reused files.
            verify_metadata(snapshot, model)
            result[role] = {"model": model["model"], "revision": model["revision"],
                            "files": len(model["files"]), "bytes": sum(item["size"] for item in model["files"])}
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
        parser.exit(1, f"GLM EXL3 download failed: {exc}\nRerun preparation to resume partial downloads.\n")


if __name__ == "__main__":
    main()
