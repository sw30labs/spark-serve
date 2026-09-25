#!/usr/bin/env python3
"""Prepare MiMo-V2.6-Flash-RL on both Sparks.

Weights download on the head only. The worker receives one copy over the QSFP
link. The serving image is built on the head and loaded on the worker.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shlex
import subprocess
import tarfile
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes" / "mimo-v26-flash"
MODEL_KEY = "mimo26"
RELATIVE_ROOT = Path("spark-serve/mimo-v26-flash")
SNAPSHOT_NAME = "XiaomiMiMo-MiMo-V2.6-Flash-RL"
LOCAL_IMAGE = "spark-serve-mimo-v26-flash:0.1.0"
PATCH_FILES = ("mimo_v2.py", "mimo_v2_omni.py", "triton_attn_diffkv.py")


def load_pins() -> dict:
    pins = json.loads((RECIPE / "source-pins.json").read_text())
    for name in PATCH_FILES:
        digest = hashlib.sha256((RECIPE / "patches" / name).read_bytes()).hexdigest()
        if digest != pins["patch_sha256"][name]:
            raise SystemExit(f"patch hash mismatch for {name}")
    return pins


def validate_catalog(cfg: dict, pins: dict) -> dict:
    model = cfg["models"][MODEL_KEY]
    if model.get("recipe") != "mimo-v26-flash" or int(model.get("nnodes") or 0) != 2:
        raise SystemExit("mimo26 must set recipe=mimo-v26-flash and nnodes=2")
    if model.get("image") != pins["local_image"] or model.get("hf_id") != pins["model"]:
        raise SystemExit("catalog image or hf_id does not match source-pins.json")
    expected = str(Path(model.get("hf_mount") or "/cache/huggingface") / RELATIVE_ROOT / SNAPSHOT_NAME)
    if model.get("serve_path") != expected:
        raise SystemExit(f"serve_path must be {expected}")
    if cfg["cluster"]["head"] == cfg["cluster"]["worker"]:
        raise SystemExit("MiMo needs two distinct Spark hosts")
    return model


def ssh(cfg: dict, host: str, script: str, timeout: int, *, stdin: bytes | None = None) -> subprocess.CompletedProcess:
    opts = list(cfg["cluster"].get("ssh_opts") or ["-o", "BatchMode=yes"])
    return subprocess.run(
        ["ssh", *opts, "-o", "ConnectTimeout=8", host, "bash", "-s"],
        input=stdin if stdin is not None else script.encode(),
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def ssh_text(cfg: dict, host: str, script: str, timeout: int = 120) -> str:
    proc = ssh(cfg, host, script, timeout)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or b"").decode(errors="replace").strip()
        raise SystemExit(f"{host}: {detail or f'exit {proc.returncode}'}")
    return proc.stdout.decode().strip()


def cache_root(cfg: dict, role: str) -> str:
    cluster = cfg["cluster"]
    cache = cluster["hf_cache_host"]
    if role == "worker" and cluster.get("worker_hf_cache_host"):
        cache = cluster["worker_hf_cache_host"]
    return str(Path(cache) / RELATIVE_ROOT)


def qsfp_ipv4(cfg: dict, host: str) -> tuple[str, str]:
    iface = str(cfg["cluster"]["nccl"]["NCCL_SOCKET_IFNAME"]).split(",")[0]
    address = ssh_text(
        cfg, host,
        "ip -4 -o addr show " + shlex.quote(iface) + " | awk '{print $4}' | cut -d/ -f1 | head -1",
    )
    if not address:
        raise SystemExit(f"{host}: no IPv4 on {iface}")
    return iface, address


def recipe_archive() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for path in sorted(RECIPE.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            data = path.read_bytes()
            info = tarfile.TarInfo(str(path.relative_to(RECIPE)))
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def upload_recipe(cfg: dict, host: str) -> str:
    remote = "/tmp/spark-serve-mimo-v26-flash-src"
    payload = recipe_archive()
    script = f"rm -rf {remote} && mkdir -p {remote} && tar -C {remote} -xf -\n"
    proc = subprocess.run(
        ["ssh", *cfg["cluster"].get("ssh_opts", ["-o", "BatchMode=yes"]),
         "-o", "ConnectTimeout=8", host, "bash", "-s"],
        input=script.encode() + payload,
        capture_output=True, timeout=120, check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"upload to {host} failed: {proc.stderr.decode(errors='replace')}")
    return remote


def image_probe(cfg: dict, host: str, name: str) -> str:
    return ssh_text(
        cfg, host,
        "docker image inspect --format '{{.Id}} {{.Architecture}}' "
        + shlex.quote(name) + " 2>/dev/null || true",
        timeout=60,
    )


def ensure_base_image(cfg: dict, host: str, pins: dict) -> None:
    probe = image_probe(cfg, host, pins["base_image"])
    if probe.startswith("sha256:") and probe.endswith(" arm64"):
        print(f"{host}: base image present", flush=True)
        return
    print(f"{host}: pulling {pins['base_image']}", flush=True)
    proc = ssh(cfg, host, "docker pull " + shlex.quote(pins["base_image"]), timeout=7200)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).decode(errors="replace")
        probe = image_probe(cfg, host, pins["base_image"])
        if not (probe.startswith("sha256:") and probe.endswith(" arm64")):
            raise SystemExit(detail)
    print(f"{host}: base image present", flush=True)


def build_image(cfg: dict, host: str, pins: dict, rebuild: bool) -> str:
    existing = image_probe(cfg, host, LOCAL_IMAGE)
    if existing.startswith("sha256:") and existing.endswith(" arm64") and not rebuild:
        image_id = existing.split()[0]
        print(f"{host}: using existing {LOCAL_IMAGE} {image_id}", flush=True)
        return image_id
    ensure_base_image(cfg, host, pins)
    remote = upload_recipe(cfg, host)
    print(f"{host}: building {LOCAL_IMAGE}", flush=True)
    script = "\n".join([
        "set -euo pipefail",
        "cd " + shlex.quote(remote),
        "docker build --network host --pull=false "
        "--build-arg " + shlex.quote("BASE_IMAGE=" + pins["base_image"]) + " "
        "-t " + shlex.quote(LOCAL_IMAGE) + " .",
        "docker image inspect --format '{{.Id}} {{.Architecture}} {{.Os}}' " + shlex.quote(LOCAL_IMAGE),
    ])
    proc = ssh(cfg, host, script, timeout=7200)
    output = (proc.stdout or b"").decode(errors="replace")
    if proc.returncode != 0:
        raise SystemExit((proc.stderr or b"").decode(errors="replace") + output)
    print(output[-2000:], flush=True)
    image_id = output.strip().splitlines()[-1].split()[0]
    if not image_id.startswith("sha256:"):
        raise SystemExit(f"could not read built image id from: {output[-500:]}")
    return image_id


def load_image_on_worker(cfg: dict, head: str, worker: str, image_id: str, head_ip: str, worker_ip: str) -> None:
    present = ssh_text(
        cfg, worker,
        "docker image inspect --format '{{.Id}}' " + shlex.quote(LOCAL_IMAGE) + " 2>/dev/null || true",
    )
    if present == image_id:
        print(f"{worker}: image already {image_id}", flush=True)
        return
    user = ssh_text(cfg, head, "id -un")
    print(f"{head}: sending image to {worker_ip}", flush=True)
    script = "\n".join([
        "set -euo pipefail",
        "docker save " + shlex.quote(LOCAL_IMAGE)
        + " | ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new "
        + f"-o BindAddress={shlex.quote(head_ip)} {shlex.quote(user + '@' + worker_ip)} "
        + "docker load",
    ])
    proc = ssh(cfg, head, script, timeout=7200)
    if proc.returncode != 0:
        raise SystemExit((proc.stderr or proc.stdout).decode(errors="replace"))
    print((proc.stdout or b"").decode()[-1000:], flush=True)
    loaded = ssh_text(cfg, worker, "docker image inspect --format '{{.Id}}' " + shlex.quote(LOCAL_IMAGE))
    if loaded != image_id:
        raise SystemExit(f"worker image id {loaded} != head {image_id}")


def download_script(root: str, revision: str) -> str:
    """Resume the public snapshot. Block brotli: httpx's decoder aborts long GETs."""
    snapshot = str(Path(root) / SNAPSHOT_NAME)
    return """#!/bin/bash
set -u
ROOT=__ROOT__
DIR=__DIR__
REV=__REV__
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
unset HF_TOKEN
export PYTHONUNBUFFERED=1
mkdir -p "$DIR"
PY="$HOME/miniconda3/bin/python3"
[ -x "$HOME/miniconda3/bin/python3.14" ] && PY="$HOME/miniconda3/bin/python3.14"
ok=0
attempt=0
while [ "$attempt" -lt 40 ]; do
  attempt=$((attempt + 1))
  echo "download attempt $attempt $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if MIMO_DIR="$DIR" MIMO_REV="$REV" "$PY" - << 'PY'
import os, sys
sys.modules["brotli"] = None
sys.modules["brotlicffi"] = None
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="XiaomiMiMo/MiMo-V2.6-Flash-RL",
    revision=os.environ["MIMO_REV"],
    local_dir=os.environ["MIMO_DIR"],
    token=False,
)
PY
  then
    ok=1
    break
  fi
  echo "attempt $attempt failed; resuming in 20s"
  sleep 20
done
if [ "$ok" -ne 1 ]; then
  echo "download failed after $attempt attempts"
  exit 1
fi
test -s "$DIR/config.json"
test -s "$DIR/dflash/config.json"
date -u +%Y-%m-%dT%H:%M:%SZ > "$ROOT/download.done"
""".replace("__ROOT__", shlex.quote(root)).replace("__DIR__", shlex.quote(snapshot)).replace("__REV__", shlex.quote(revision))


def ensure_download(cfg: dict, host: str, root: str, pins: dict) -> None:
    snapshot = str(Path(root) / SNAPSHOT_NAME)
    state = ssh_text(
        cfg, host,
        "\n".join([
            "set -euo pipefail",
            f"ROOT={shlex.quote(root)}",
            'if [ -f "$ROOT/download.done" ] && [ -s "$ROOT/' + SNAPSHOT_NAME + '/config.json" ]; then echo done; exit 0; fi',
            'if [ -f "$ROOT/download.pid" ] && kill -0 "$(cat "$ROOT/download.pid")" 2>/dev/null; then echo running; exit 0; fi',
            "echo missing",
        ]),
    )
    if state == "done":
        print(f"{host}: weights already downloaded", flush=True)
        return
    if state != "running":
        print(f"{host}: starting weight download (worker will get a copy)", flush=True)
        script = download_script(root, pins["revision"])
        remote = "\n".join([
            "set -euo pipefail",
            f"mkdir -p {shlex.quote(root)}",
            f"cat > {shlex.quote(root + '/download.sh')} << 'ENDDOWNLOAD'",
            script,
            "ENDDOWNLOAD",
            f"chmod 755 {shlex.quote(root + '/download.sh')}",
            f"rm -f {shlex.quote(root + '/download.done')}",
            f"nohup {shlex.quote(root + '/download.sh')} > {shlex.quote(root + '/download.log')} 2>&1 &",
            f"echo $! > {shlex.quote(root + '/download.pid')}",
        ])
        ssh_text(cfg, host, remote, timeout=60)
    deadline = time.monotonic() + 8 * 3600
    while time.monotonic() < deadline:
        time.sleep(30)
        probe = ssh_text(
            cfg, host,
            f"if [ -f {shlex.quote(root + '/download.done')} ]; then echo done; "
            f"elif [ -f {shlex.quote(root + '/download.pid')} ] && kill -0 \"$(cat {shlex.quote(root + '/download.pid')})\" 2>/dev/null; then "
            f"du -sb {shlex.quote(snapshot)} | awk '{{print $1}}'; else echo dead; fi",
            timeout=60,
        )
        if probe == "done":
            print(f"{host}: weight download finished", flush=True)
            return
        if probe == "dead":
            log = ssh_text(cfg, host, f"tail -c 2000 {shlex.quote(root + '/download.log')} || true")
            raise SystemExit(f"weight download exited early:\n{log}")
        print(f"{host}: downloading, snapshot bytes {probe}", flush=True)
    raise SystemExit("weight download exceeded 8 hours")


def finalize_snapshot(cfg: dict, host: str, root: str, pins: dict) -> dict:
    remote_src = upload_recipe(cfg, host)
    snapshot = str(Path(root) / SNAPSHOT_NAME)
    script = "\n".join([
        "set -euo pipefail",
        "python3 - " + shlex.quote(remote_src) + " " + shlex.quote(snapshot) + " " + shlex.quote(root) + " " + shlex.quote(pins["revision"]) + " << 'PY'",
        "import importlib.util, json, sys",
        "from pathlib import Path",
        "src, snapshot, root, revision = sys.argv[1:]",
        "def load(name):",
        "    spec = importlib.util.spec_from_file_location(name, Path(src) / (name + '.py'))",
        "    module = importlib.util.module_from_spec(spec)",
        "    spec.loader.exec_module(module)",
        "    return module",
        "jsonfix, verify = load('jsonfix'), load('verify')",
        "changed = jsonfix.repair_file(Path(snapshot) / 'dflash' / 'config.json')",
        "manifest = verify.inventory(Path(snapshot))",
        "manifest.update(version=1, model='XiaomiMiMo/MiMo-V2.6-Flash-RL', revision=revision, dflash_config_repaired=changed)",
        "target = Path(root) / 'manifest.json'",
        "target.write_text(json.dumps(manifest, sort_keys=True) + '\\n')",
        "Path(root, 'runtime-cache').mkdir(parents=True, exist_ok=True)",
        "print(json.dumps(manifest))",
        "PY",
    ])
    raw = ssh_text(cfg, host, script, timeout=600)
    manifest = json.loads(raw.splitlines()[-1])
    print(f"{host}: snapshot {manifest['files']} files, {manifest['bytes']} bytes", flush=True)
    return manifest


def copy_snapshot(cfg: dict, head: str, worker: str, root: str, head_ip: str, worker_ip: str, manifest: dict) -> None:
    user = ssh_text(cfg, head, "id -un")
    snapshot = str(Path(root) / SNAPSHOT_NAME)
    print(f"copying weights {head_ip} -> {worker_ip}", flush=True)
    # Run with bash -lc, not bash -s. rsync's ssh transport must not inherit a
    # script on stdin; that produced an empty copy that still exited 0.
    script = "\n".join([
        "set -euo pipefail",
        "command -v rsync >/dev/null",
        "ssh -n -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new "
        + f"-o BindAddress={shlex.quote(head_ip)} {shlex.quote(user + '@' + worker_ip)} "
        + "mkdir -p " + shlex.quote(root + "/runtime-cache"),
        "rsync -a --partial --info=stats2 --exclude '.cache/' --exclude 'download.log' --exclude 'download.pid' --exclude 'download.sh' "
        "-e " + shlex.quote(
            "ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new -o BindAddress=" + head_ip
        ) + " "
        + shlex.quote(root + "/") + " "
        + shlex.quote(f"{user}@{worker_ip}:{root}/"),
    ])
    opts = list(cfg["cluster"].get("ssh_opts") or ["-o", "BatchMode=yes"])
    proc = subprocess.run(
        ["ssh", *opts, "-o", "ConnectTimeout=8", head, "bash", "-lc", script],
        capture_output=True, timeout=6 * 3600, check=False,
    )
    if proc.returncode != 0:
        raise SystemExit("weight copy failed:\n" + (proc.stderr or proc.stdout).decode(errors="replace")[-4000:])
    print((proc.stderr or proc.stdout).decode(errors="replace")[-1500:], flush=True)
    checked = finalize_snapshot(cfg, worker, root, {"revision": manifest["revision"]})
    for key in ("files", "bytes", "config_sha256", "dflash_config_sha256"):
        if checked[key] != manifest[key]:
            raise SystemExit(f"worker {key} {checked[key]} != head {manifest[key]}")
    print(f"{worker}: copy matches the head snapshot", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "models.toml")
    parser.add_argument("--image-only", action="store_true", help="build and distribute the image; do not wait for weights")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the local image even if the tag exists")
    args = parser.parse_args()
    pins = load_pins()
    cfg = tomllib.loads(args.catalog.read_text())
    validate_catalog(cfg, pins)
    head, worker = cfg["cluster"]["head"], cfg["cluster"]["worker"]
    _, head_ip = qsfp_ipv4(cfg, head)
    _, worker_ip = qsfp_ipv4(cfg, worker)
    image_id = build_image(cfg, head, pins, args.rebuild)
    load_image_on_worker(cfg, head, worker, image_id, head_ip, worker_ip)
    if args.image_only:
        print(json.dumps({"image": LOCAL_IMAGE, "image_id": image_id}))
        return
    root = cache_root(cfg, "head")
    ensure_download(cfg, head, root, pins)
    manifest = finalize_snapshot(cfg, head, root, pins)
    copy_snapshot(cfg, head, worker, root, head_ip, worker_ip, manifest)
    print(json.dumps({"image": LOCAL_IMAGE, "image_id": image_id, "files": manifest["files"], "bytes": manifest["bytes"]}))


if __name__ == "__main__":
    main()
