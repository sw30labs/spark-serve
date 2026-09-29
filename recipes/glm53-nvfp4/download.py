#!/usr/bin/env python3
"""Resume public, version-pinned NGC files; authenticate each before publication."""
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
from pathlib import Path

from verify import CheckpointFileError, checkpoint_path, manifest_files, verify, verify_file


def download_file(root: Path, item: dict, base_url: str, *, opener=urllib.request.urlopen,
                  pause=time.sleep, attempts=5) -> None:
    target = checkpoint_path(root, item["path"])
    if target.is_file():
        try:
            verify_file(root, item, full_hash=True)
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
                headers = {"User-Agent": "spark-serve-glm53-preparation/1", "Accept-Encoding": "identity"}
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
                verify_file(root, partial_item, full_hash=True)
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


def download(root: Path, manifest: dict) -> dict:
    files = manifest_files(manifest)
    base_url = manifest["download_base_url"]
    parsed = urllib.parse.urlsplit(base_url)
    expected_path = f"/v2/models/nim/zai-org/glm-5.3-flash/versions/{manifest['revision']}/files"
    if (parsed.scheme != "https" or parsed.hostname != "api.ngc.nvidia.com"
            or parsed.path.rstrip("/") != expected_path
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("download URL must identify the pinned public NVIDIA NGC version")
    if root.name != manifest["revision"]:
        raise ValueError("download directory does not identify the pinned NGC version")
    root.mkdir(parents=True, exist_ok=True)
    with (root.parent / ".download.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("another GLM checkpoint preparation is running on this node") from None
        for item in files:
            download_file(root, item, base_url)
        # Every file was fully authenticated above; this final pass validates
        # the complete file set, serving metadata and index without rereading
        # hundreds of GB of already authenticated shards.
        result = verify(root, manifest)
        result["full_hash_verified"] = True
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("model-source.json"))
    args = parser.parse_args()
    try:
        result = download(args.snapshot, json.loads(args.manifest.read_text()))
        print(json.dumps(result), flush=True)
    except (OSError, ValueError, KeyError, TypeError, http.client.HTTPException) as exc:
        parser.exit(1, f"GLM checkpoint download failed: {exc}\nRerun preparation to resume partial downloads.\n")


if __name__ == "__main__":
    main()
