#!/usr/bin/env python3
"""Prepare pinned GLM EXL3 assets on both Sparks without changing workloads."""
from __future__ import annotations

import argparse
import hashlib
import io
import ipaddress
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
RECIPE = ROOT / "recipes" / "glm53-exl3"
MODEL_KEY = "glm53-exl3"
RELATIVE_ROOT = Path("spark-serve/glm53-exl3")
UPSTREAM_REVISION = "b2c5986324a7bac7b02bf0a47ed7b92b25e56016"


def load_pins(recipe: Path = RECIPE) -> tuple[dict, dict]:
    pins = json.loads((recipe / "source-pins.json").read_text())
    raw = (recipe / "model-source.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != pins["manifest_sha256"]:
        raise ValueError("GLM EXL3 checkpoint manifest SHA-256 mismatch")
    if pins["upstream_revision"] != UPSTREAM_REVISION:
        raise ValueError("GLM EXL3 upstream source revision mismatch")
    if not re.fullmatch(r"ghcr\.io/miaai-lab/glm-5\.3-flash-2x-dgx-sparks@sha256:[0-9a-f]{64}", pins["base_image"]):
        raise ValueError("GLM EXL3 base image must be immutable")
    source = json.loads(raw)
    if set(source["checkpoints"]) != {"target", "draft"}:
        raise ValueError("both target and draft must be pinned")
    for role, checkpoint in source["checkpoints"].items():
        if checkpoint["revision"] != pins["checkpoint_revisions"][role]:
            raise ValueError(f"{role} checkpoint revisions disagree")
    expected = pins["runtime_sha256"]
    actual_names = {str(p.relative_to(recipe)) for p in (recipe / "runtime").rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts}
    if set(expected) != actual_names:
        raise ValueError("GLM EXL3 runtime inventory differs from source pins")
    for name, digest in {**expected, **pins["preparation_sha256"]}.items():
        path = recipe / name
        if not path.resolve().is_relative_to(recipe.resolve()):
            raise ValueError("runtime path escapes recipe")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"GLM EXL3 source SHA-256 mismatch: {name}")
    return pins, source


def node_root(cfg: dict, role: str) -> str:
    cluster = cfg["cluster"]
    cache = cluster.get("worker_hf_cache_host") if role == "worker" else None
    cache = cache or cluster["hf_cache_host"]
    if not Path(cache).is_absolute():
        raise ValueError("checkpoint cache paths must be absolute")
    return str(Path(cache) / RELATIVE_ROOT)


def validate_catalog(cfg: dict, pins: dict, source: dict) -> dict:
    model = cfg["models"][MODEL_KEY]
    if model.get("recipe") != MODEL_KEY or int(model.get("nnodes", 0)) != 2:
        raise ValueError("glm53-exl3 requires its own recipe and nnodes=2")
    if model.get("image") != pins["local_image"]:
        raise ValueError("catalog image differs from pinned EXL3 derivative")
    target = source["checkpoints"]["target"]
    if model.get("hf_id") != target["model"]:
        raise ValueError("catalog target model differs from pinned EXL3 checkpoint")
    root = Path(model.get("hf_mount") or "/cache/huggingface") / RELATIVE_ROOT
    expected = str(root / "models" / "target" / target["revision"])
    if model.get("serve_path") != expected:
        raise ValueError(f"EXL3 serve_path must be {expected}")
    preflight = model.get("preflight_args", [])
    required = ["/opt/spark-serve/glm53-exl3/verify.py", str(root),
                "--expected-manifest-sha256", pins["manifest_sha256"]]
    if preflight != required:
        raise ValueError("EXL3 preflight must authenticate both pinned snapshots")
    cluster = cfg["cluster"]
    if cluster["head"] == cluster["worker"]:
        raise ValueError("GLM EXL3 needs two distinct Spark hosts")
    for role in ("head", "worker"):
        host = cluster[role]
        if not isinstance(host, str) or not host or host.startswith("-") or any(c.isspace() for c in host):
            raise ValueError("invalid Spark SSH host")
        node_root(cfg, role)
    return model


def ssh_command(cfg: dict, role: str) -> list[str]:
    return ["ssh", *cfg["cluster"].get("ssh_opts", ["-o", "BatchMode=yes"]),
            "-o", "ConnectTimeout=8", cfg["cluster"][role]]


def remote_text(cfg: dict, role: str, script: str, timeout: int = 60) -> str:
    result = subprocess.run([*ssh_command(cfg, role), "bash -s"], input=script,
                            text=True, capture_output=True, check=True, timeout=timeout)
    return result.stdout.strip()


def run_remote(cfg: dict, role: str, script: str) -> None:
    subprocess.run([*ssh_command(cfg, role), "bash -s"], input=script,
                   text=True, check=True)


def write_payload(archive, recipe: Path = RECIPE) -> None:
    with tarfile.open(fileobj=archive, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for path in sorted(recipe.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            data = path.read_bytes()
            info = tarfile.TarInfo(str(path.relative_to(recipe)))
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            tar.addfile(info, io.BytesIO(data))


def upload_recipe(cfg: dict, role: str) -> str:
    remote = remote_text(cfg, role, "mktemp -d /tmp/spark-serve-glm53-exl3.XXXXXXXX")
    if not re.fullmatch(r"/tmp/spark-serve-glm53-exl3\.[A-Za-z0-9]+", remote):
        raise ValueError("unexpected remote preparation directory")
    with tempfile.TemporaryFile() as archive:
        write_payload(archive)
        archive.seek(0)
        subprocess.run([*ssh_command(cfg, role), "tar -xf - -C " + shlex.quote(remote)],
                       stdin=archive, check=True, timeout=120)
    return remote


def download_script(remote: str, root: str, pins: dict, skip_download: bool = False) -> str:
    tool = "verify.py" if skip_download else "download.py"
    return "\n".join(["set -euo pipefail", "cd " + shlex.quote(remote),
                       shlex.join(["python3", tool, root, "--expected-manifest-sha256", pins["manifest_sha256"]])]) + "\n"


def build_script(remote: str, pins: dict) -> str:
    return "\n".join(["set -euo pipefail", "cd " + shlex.quote(remote),
                       shlex.join(["docker", "pull", "--platform", "linux/arm64", pins["base_image"]]),
                       shlex.join(["docker", "build", "--network", "none", "--pull=false",
                                   "--file", "runtime/Dockerfile", "--tag", pins["local_image"], "."]),
                       shlex.join(["docker", "run", "--rm", "--network", "none",
                                   "--env", "NVIDIA_VISIBLE_DEVICES=void", "--env", "CUDA_VISIBLE_DEVICES=",
                                   pins["local_image"], "--patches-only"])]) + "\n"


def qsfp_address(cfg: dict, role: str) -> str:
    interface = str(cfg["cluster"]["nccl"]["NCCL_SOCKET_IFNAME"])
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", interface):
        raise ValueError("one explicit QSFP socket interface is required")
    raw = remote_text(cfg, role, "ip -j -4 addr show dev " + shlex.quote(interface))
    addresses = [x["local"] for device in json.loads(raw) for x in device.get("addr_info", []) if x.get("family") == "inet"]
    if len(addresses) != 1:
        raise ValueError(f"{role}: require one IPv4 address on {interface}")
    return str(ipaddress.IPv4Address(addresses[0]))


def copy_script(head_root: str, worker_root: str, head_ip: str, worker_ip: str,
                user: str, image: str | None) -> str:
    for address in (head_ip, worker_ip):
        ipaddress.IPv4Address(address)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", user):
        raise ValueError("invalid worker SSH username")
    transport = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                 "-o", "StrictHostKeyChecking=accept-new", "-o", "BindAddress=" + head_ip]
    peer = user + "@" + worker_ip
    commands = [
        "set -euo pipefail",
        shlex.join([*transport, "-n", peer, "mkdir -p " + shlex.quote(worker_root + "/models")]),
        shlex.join(["rsync", "-a", "--checksum", "--partial", "--protect-args", "--info=stats2",
                    "--exclude=.cache/", "--exclude=*.incomplete", "-e", shlex.join(transport),
                    head_root + "/models/", peer + ":" + worker_root + "/models/"]),
    ]
    if image is not None:
        commands.append(shlex.join(["docker", "save", image]) + " | " + shlex.join([*transport, peer, "docker load"]))
    return "\n".join(commands) + "\n"


def image_report(cfg: dict, role: str, pins: dict) -> dict:
    raw = remote_text(cfg, role, shlex.join(["docker", "image", "inspect", pins["local_image"]]))
    image = json.loads(raw)[0]
    if image.get("Architecture") != "arm64" or image.get("Os") != "linux":
        raise ValueError(f"{role}: wrong image platform")
    labels = image.get("Config", {}).get("Labels", {})
    if labels.get("ai.spark-serve.recipe.commit") != pins["upstream_revision"]:
        raise ValueError(f"{role}: image source revision does not match")
    return {"image": pins["local_image"], "image_id": image["Id"],
            "rootfs_layers": image["RootFS"]["Layers"], "base_image": pins["base_image"]}


def publish_script(remote: str, root: str, pins: dict, image: dict) -> str:
    receipt = {"manifest_sha256": pins["manifest_sha256"], "full_hash_verified": True, **image}
    return "\n".join([
        "set -euo pipefail", "mkdir -p " + shlex.quote(root + "/runtime-cache"),
        shlex.join(["cp", remote + "/verify.py", remote + "/model-source.json", remote + "/source-pins.json", root + "/"]),
        "printf '%s\\n' " + shlex.quote(json.dumps(receipt, sort_keys=True)) + " > " + shlex.quote(root + "/prepared.json.tmp"),
        shlex.join(["mv", root + "/prepared.json.tmp", root + "/prepared.json"]),
    ]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "models.toml")
    parser.add_argument("--node", choices=("both",), default="both", help="EXL3 TP2 is prepared as a two-node allocation")
    parser.add_argument("--download-only", action="store_true", help="download and authenticate target and draft on head only")
    parser.add_argument("--skip-download", action="store_true", help="authenticate existing head snapshots without Internet downloads")
    args = parser.parse_args()
    pins, source = load_pins()
    cfg = tomllib.loads(args.catalog.read_text())
    validate_catalog(cfg, pins, source)
    head_root, worker_root = node_root(cfg, "head"), node_root(cfg, "worker")
    head_source = upload_recipe(cfg, "head")
    print("Preparing GLM EXL3; serving containers remain running", flush=True)
    run_remote(cfg, "head", download_script(head_source, head_root, pins, args.skip_download))
    if args.download_only:
        print("Head target and draft downloads fully authenticated; runtime not published", flush=True)
        return
    run_remote(cfg, "head", build_script(head_source, pins))
    head_image = image_report(cfg, "head", pins)
    head_ip, worker_ip = qsfp_address(cfg, "head"), qsfp_address(cfg, "worker")
    worker_user = remote_text(cfg, "worker", "id -un")
    worker_image_id = remote_text(cfg, "worker", shlex.join([
        "docker", "image", "inspect", "--format", "{{.Id}}", pins["local_image"]]) + " 2>/dev/null || true")
    image_to_copy = None if worker_image_id == head_image["image_id"] else pins["local_image"]
    script = copy_script(head_root, worker_root, head_ip, worker_ip, worker_user, image_to_copy)
    # Keep rsync's transport stdin separate from shell source. SSH concatenates
    # remote argv; quote the bash -c payload as one remote command argument.
    subprocess.run([*ssh_command(cfg, "head"), "bash -c " + shlex.quote(script)], check=True)
    worker_source = upload_recipe(cfg, "worker")
    run_remote(cfg, "worker", download_script(worker_source, worker_root, pins, True))
    worker_image = image_report(cfg, "worker", pins)
    if worker_image["image_id"] != head_image["image_id"] or worker_image["rootfs_layers"] != head_image["rootfs_layers"]:
        raise ValueError("worker derivative image differs from head")
    for role, remote, root, image in (("head", head_source, head_root, head_image), ("worker", worker_source, worker_root, worker_image)):
        run_remote(cfg, role, publish_script(remote, root, pins, image))
    print("GLM EXL3 assets verified on both Sparks. Start with ./spark-serve up glm53-exl3.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as exc:
        sys.exit(f"GLM EXL3 preparation failed: {exc}")
