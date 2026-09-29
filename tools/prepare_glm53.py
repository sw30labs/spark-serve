#!/usr/bin/env python3
"""Prepare the pinned two-Spark GLM NIM assets without changing serving workloads."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes" / "glm53-nvfp4"
MODEL_KEY = "glm53"
RELATIVE_ROOT = Path("spark-serve/glm53-nvfp4")
DERIVATIVE_FILES = ("Dockerfile", "patch_runtime.py", "nvfp4_scale_reconcile.py")
BASE_IMAGE_PATTERN = r"nvcr\.io/nim/zai-org/glm-5\.3-flash@sha256:[0-9a-f]{64}"


def derivative_build_command(context: str = "image-context", tag: str = "spark-serve-glm53:marlin-gate-up-maxglobal-e4m3-v1",
                             iidfile: str = "image-build.iid") -> list[str]:
    """Shared by preparation and the initial two-node reproducibility check.

    The pinned base is already local. This creates no GPU workload, pulls no
    frontend, and leaves the original image and running containers untouched.
    The Dockerfile normalizes its single changed layer; SOURCE_DATE_EPOCH fixes
    the image config/history timestamps, including on the classic Docker store.
    """
    return ["docker", "buildx", "build", "--builder", "default", "--platform", "linux/arm64",
            "--network", "none", "--pull=false", "--provenance=false", "--sbom=false",
            "--build-arg", "SOURCE_DATE_EPOCH=0", "--tag", tag,
            "--iidfile", iidfile, context]


def validate_image_patch(pins: dict, recipe: Path = RECIPE) -> dict | None:
    if not pins.get("base_image"):
        return None
    patch = pins.get("image_patch")
    if (not isinstance(patch, dict) or not isinstance(patch.get("revision"), str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", patch["revision"])
            or not isinstance(patch.get("files"), dict)
            or set(patch["files"]) != set(DERIVATIVE_FILES)):
        raise ValueError("GLM derivative image requires a pinned revision and all build-file hashes")
    for name in DERIVATIVE_FILES:
        expected = patch["files"][name]
        if (not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)
                or hashlib.sha256((recipe / name).read_bytes()).hexdigest() != expected):
            raise ValueError(f"GLM derivative build file SHA-256 mismatch: {name}")
    if "FROM " + pins["base_image"] not in (recipe / "Dockerfile").read_text().splitlines():
        raise ValueError("GLM derivative Dockerfile must use its pinned NVIDIA base image")
    return patch


def write_payload(archive, pins: dict, recipe: Path = RECIPE) -> None:
    """A deterministic, small tar: no local ownership, mtimes or cache files."""
    entries = {name: recipe / name for name in ("verify.py", "download.py", "model-source.json", "source-pins.json")}
    if pins.get("base_image"):
        entries.update({"image-context/" + name: recipe / name for name in DERIVATIVE_FILES})
    with tarfile.open(fileobj=archive, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        if pins.get("base_image"):
            info = tarfile.TarInfo("image-context")
            info.type, info.mode = tarfile.DIRTYPE, 0o755
            tar.addfile(info)
        for name, path in sorted(entries.items()):
            data = path.read_bytes()
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o644
            tar.addfile(info, io.BytesIO(data))


def preparation_nodes(cfg: dict, selected: str) -> list[tuple[str, str]]:
    cluster = cfg["cluster"]
    roles = ("head", "worker") if selected == "both" else (selected,)
    if cluster["head"] == cluster["worker"]:
        raise ValueError("GLM requires two distinct configured Spark hosts")
    nodes = []
    for role in roles:
        host = str(cluster[role])
        cache = str(cluster.get("worker_hf_cache_host") or cluster["hf_cache_host"]) if role == "worker" else str(cluster["hf_cache_host"])
        if not host or host.startswith("-") or not Path(cache).is_absolute():
            raise ValueError("each Spark needs an SSH host and an absolute checkpoint cache path")
        nodes.append((host, cache))
    return nodes


def validate_recipe(cfg: dict, source: dict, pins: dict) -> dict:
    model = cfg["models"][MODEL_KEY]
    if model.get("recipe") != "glm53-nvfp4" or int(model.get("nnodes", 0)) != 2:
        raise ValueError("this preparation recipe requires glm53 with recipe=glm53-nvfp4 and nnodes=2")
    image = model.get("image", "")
    expected_format = r"sha256:[0-9a-f]{64}" if pins.get("base_image") else BASE_IMAGE_PATTERN
    if image != pins["image"] or not re.fullmatch(expected_format, image):
        raise ValueError("GLM image must match the recipe's pinned NVIDIA ARM64 image digest")
    if pins.get("base_image") and not re.fullmatch(BASE_IMAGE_PATTERN, pins["base_image"]):
        raise ValueError("GLM derivative base_image must be the pinned NVIDIA ARM64 image digest")
    if source["revision"] != pins["checkpoint_revision"]:
        raise ValueError("GLM source manifest and source pins disagree")
    declared_revision = model.get("nim", {}).get("model_revision")
    if declared_revision is not None and declared_revision != source["revision"]:
        raise ValueError("GLM catalog and recipe checkpoint revisions disagree")
    expected = str(Path(model.get("hf_mount") or "/cache/huggingface") / RELATIVE_ROOT / "models" / source["revision"])
    if model.get("serve_path") and model["serve_path"] != expected:
        raise ValueError(f"GLM serve_path must identify the prepared checkpoint: {expected}")
    return model


def preparation_script(remote: str, cache: str, model: dict, source: dict, *, pins: dict | None = None, skip_download=False) -> str:
    root = Path(cache) / RELATIVE_ROOT
    snapshot = root / "models" / source["revision"]
    image = model["image"]
    pins = pins or {"image": image}
    derivative = bool(pins.get("base_image"))
    # Image inspection is CPU-only. Pinning the reference is necessary but we
    # still reject a wrong-platform local image before publishing preparation.
    inspect_source = """import json, pathlib, subprocess, sys
expected, derivative, pins = sys.argv[1], sys.argv[2] == '1', json.loads(sys.argv[3])
image = pathlib.Path('image-build.iid').read_text().strip() if derivative else expected
if derivative and image != expected:
    raise SystemExit('GLM derivative image ID mismatch: expected ' + expected + ', built ' + image)
item = json.loads(subprocess.check_output(['docker', 'image', 'inspect', image], text=True))[0]
if item.get('Architecture') != 'arm64' or item.get('Os') != 'linux':
    raise SystemExit('GLM preparation requires the pinned Linux ARM64 NIM image')
if derivative and (item['Id'] != expected or item.get('Config', {}).get('User') != 'nvs:1000'):
    raise SystemExit('GLM derivative image must match the pinned ID and preserve NIM user nvs:1000')
report = {'image': expected, 'image_id': item['Id'], 'architecture': item['Architecture'], 'os': item['Os']}
if derivative:
    report.update(base_image=pins['base_image'], image_patch=pins['image_patch'])
pathlib.Path('image-inspection.json').write_text(json.dumps(report, sort_keys=True) + '\\n')
print(json.dumps(report))
"""
    stage_source = """import hashlib, json, os, pathlib, sys, tempfile, time
source, root, snapshot, image = map(str, sys.argv[1:])
source, root = pathlib.Path(source), pathlib.Path(root)
root.mkdir(parents=True, exist_ok=True)
(root / 'runtime-cache').mkdir(parents=True, exist_ok=True)
def publish(name, data):
    fd, temporary = tempfile.mkstemp(prefix='.' + name + '.', dir=root)
    try:
        with os.fdopen(fd, 'wb') as out:
            out.write(data); out.flush(); os.fsync(out.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, root / name)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
for name in ('verify.py', 'model-source.json', 'source-pins.json'):
    publish(name, (source / name).read_bytes())
manifest = (source / 'model-source.json').read_bytes()
metadata = json.loads(manifest)
receipt = {'version': 1, 'prepared_at': time.time(), 'model': metadata['model'],
           'revision': metadata['revision'], 'snapshot': snapshot, 'image': image,
           'files': len(metadata['files']), 'bytes': sum(item['size'] for item in metadata['files']),
           'manifest_sha256': hashlib.sha256(manifest).hexdigest(), 'full_hash_verified': True}
receipt.update(json.loads((source / 'image-inspection.json').read_text()))
publish('prepared.json', (json.dumps(receipt, sort_keys=True) + '\\n').encode())
parent = os.open(root, os.O_RDONLY)
try: os.fsync(parent)
finally: os.close(parent)
print(json.dumps(receipt), flush=True)
"""
    commands = ["set -euo pipefail", "cd " + shlex.quote(remote)]
    commands += [shlex.join(["python3", "verify.py" if skip_download else "download.py", str(snapshot)])
                 + (" --full-hash" if skip_download else "")]
    commands += [shlex.join(["docker", "pull", pins.get("base_image") or image])]
    if derivative:
        commands += [shlex.join(derivative_build_command())]
    commands += [shlex.join(["python3", "-c", inspect_source, image, "1" if derivative else "0", json.dumps(pins, sort_keys=True)]),
                 shlex.join(["python3", "-c", stage_source, remote, str(root), str(snapshot), image])]
    return "\n".join(commands) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "models.toml")
    parser.add_argument("--node", choices=("both", "head", "worker"), default="both")
    parser.add_argument("--skip-download", action="store_true", help="fully verify an already cached checkpoint")
    args = parser.parse_args()
    cfg = tomllib.loads(args.catalog.read_text())
    manifest_bytes = (RECIPE / "model-source.json").read_bytes()
    source = json.loads(manifest_bytes)
    pins = json.loads((RECIPE / "source-pins.json").read_text())
    if (hashlib.sha256(manifest_bytes).hexdigest() != pins["manifest_sha256"]
            or len(source["files"]) != pins["checkpoint_files"]
            or sum(item["size"] for item in source["files"]) != pins["checkpoint_bytes"]):
        raise ValueError("GLM checkpoint manifest does not match source-pins.json")
    model = validate_recipe(cfg, source, pins)
    validate_image_patch(pins)
    nodes = preparation_nodes(cfg, args.node)
    for host, cache in nodes:
        ssh = ["ssh", *cfg["cluster"].get("ssh_opts", ["-o", "BatchMode=yes"]),
               "-o", "ConnectTimeout=8", host]
        print(f"Preparing GLM on {host}; checkpoint {source['revision']}; existing workloads remain running", flush=True)
        proc = subprocess.run([*ssh, "mktemp -d /tmp/spark-serve-glm53.XXXXXXXX"],
                              capture_output=True, text=True, check=True, timeout=30)
        remote = proc.stdout.strip()
        if not re.fullmatch(r"/tmp/spark-serve-glm53\.[A-Za-z0-9]+", remote):
            raise RuntimeError("unexpected remote preparation directory")
        with tempfile.TemporaryFile() as archive:
            write_payload(archive, pins)
            archive.seek(0)
            subprocess.run([*ssh, "tar -xf - -C " + shlex.quote(remote)], stdin=archive, check=True, timeout=60)
        subprocess.run([*ssh, "bash -s"],
                       input=preparation_script(remote, cache, model, source, pins=pins, skip_download=args.skip_download),
                       text=True, check=True)
    print("GLM preparation complete on " + ", ".join(host for host, _ in nodes) + ". Start it with ./spark-serve up glm53.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as exc:
        sys.exit(f"GLM preparation failed: {exc}")
