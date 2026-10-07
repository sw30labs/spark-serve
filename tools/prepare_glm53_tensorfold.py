#!/usr/bin/env python3
"""Prepare pinned GLM TensorFold assets on both Sparks without changing workloads."""
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
sys.path.insert(0, str(ROOT))
from spark_serve_tensorfold import validate_tensorfold_plan

RECIPE = ROOT / "recipes" / "glm53-tensorfold"
MODEL_KEY = "glm53-tensorfold"
RELATIVE_ROOT = Path("spark-serve/glm53-tensorfold")
UPSTREAM_REVISION = "33b50fde06fd7ea604cbc6a663880068ab1e2ee4"
RUNTIME_ROOT = "/opt/spark-serve/glm53-tensorfold"
CHECKPOINTS = {
    "target": ("Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold", "078455ffe6472f9a52fbc1139f58b9db2881b25c"),
    "draft": ("incoai/GLM-5.3-Flash-DFlash2", "bf582e4eacc1810f76656d1811693ff6c6737d2a"),
}


def load_pins(recipe: Path = RECIPE) -> tuple[dict, dict]:
    pins = json.loads((recipe / "source-pins.json").read_text())
    raw = (recipe / "model-source.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != pins["manifest_sha256"]:
        raise ValueError("GLM TensorFold checkpoint manifest SHA-256 mismatch")
    if pins["upstream_revision"] != UPSTREAM_REVISION:
        raise ValueError("GLM TensorFold upstream source revision mismatch")
    if not re.fullmatch(r"ghcr\.io/miaai-lab/glm-5\.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:[0-9a-f]{64}", pins["base_image"]):
        raise ValueError("GLM TensorFold base image must be immutable")
    source = json.loads(raw)
    if set(source["checkpoints"]) != {"target", "draft"}:
        raise ValueError("both target and draft must be pinned")
    for role, checkpoint in source["checkpoints"].items():
        if (checkpoint["model"], checkpoint["revision"]) != CHECKPOINTS[role]:
            raise ValueError(f"{role} must use the pinned GLM TensorFold checkpoint")
        if checkpoint["revision"] != pins["checkpoint_revisions"][role]:
            raise ValueError(f"{role} checkpoint revisions disagree")
    expected = pins["runtime_sha256"]
    actual_names = {str(p.relative_to(recipe)) for p in (recipe / "runtime").rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts}
    if set(expected) != actual_names:
        raise ValueError("GLM TensorFold runtime inventory differs from source pins")
    for name, digest in {**expected, **pins["preparation_sha256"]}.items():
        path = recipe / name
        if not path.resolve().is_relative_to(recipe.resolve()):
            raise ValueError("runtime path escapes recipe")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"GLM TensorFold source SHA-256 mismatch: {name}")
    return pins, source


def node_root(cfg: dict, role: str) -> str:
    cluster = cfg["cluster"]
    cache = cluster.get("worker_hf_cache_host") if role == "worker" else None
    cache = cache or cluster["hf_cache_host"]
    if not Path(cache).is_absolute():
        raise ValueError("checkpoint cache paths must be absolute")
    return str(Path(cache) / RELATIVE_ROOT)


def validate_catalog(cfg: dict, pins: dict, source: dict) -> dict:
    validate_tensorfold_plan(cfg, MODEL_KEY)
    validate_tensorfold_plan(cfg, MODEL_KEY, rank=1)
    model = cfg["models"][MODEL_KEY]
    if model.get("recipe") != MODEL_KEY or int(model.get("nnodes", 0)) != 2 or int(model.get("tensor_parallel", 0)) != 2:
        raise ValueError("glm53-tensorfold requires its own recipe and nnodes=2")
    if model.get("image") != pins["local_image"]:
        raise ValueError("catalog image differs from pinned TensorFold derivative")
    target = source["checkpoints"]["target"]
    if model.get("hf_id") != target["model"]:
        raise ValueError("catalog target model differs from pinned TensorFold checkpoint")
    root = Path(model.get("hf_mount") or "/cache/huggingface") / RELATIVE_ROOT
    expected = str(root / "models" / "target" / target["revision"])
    if model.get("serve_path") != expected:
        raise ValueError(f"TensorFold serve_path must be {expected}")
    preflight = model.get("preflight_args", [])
    required = [RUNTIME_ROOT + "/verify.py", str(root),
                "--expected-manifest-sha256", pins["manifest_sha256"], "--verify-runtime"]
    if preflight != required:
        raise ValueError("TensorFold preflight must authenticate both pinned snapshots")
    if model.get("wrapper") != "tensorfold" or model.get("hf_mount") != "/cache/huggingface":
        raise ValueError("TensorFold requires its wrapper and fixed read-only checkpoint mount")
    for role, setting in (("target", "model_revision"), ("draft", "draft_revision")):
        if model.get("tensorfold", {}).get(setting) != source["checkpoints"][role]["revision"]:
            raise ValueError(f"catalog {role} checkpoint revision differs from source pins")
    cluster = cfg["cluster"]
    if cluster["head"] == cluster["worker"]:
        raise ValueError("GLM TensorFold needs two distinct Spark hosts")
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
    remote = remote_text(cfg, role, "mktemp -d /tmp/spark-serve-glm53-tensorfold.XXXXXXXX")
    if not re.fullmatch(r"/tmp/spark-serve-glm53-tensorfold\.[A-Za-z0-9]+", remote):
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
    space_check = "import shutil,sys; free=shutil.disk_usage(sys.argv[1]).free; " \
                  "sys.exit('TensorFold image needs 35 GB free under Docker storage' if free < 35_000_000_000 else 0)"
    return "\n".join(["set -euo pipefail", "cd " + shlex.quote(remote),
                       "if ! " + shlex.join(["docker", "image", "inspect", pins["base_image"]]) + " >/dev/null 2>&1; then",
                       "  docker_root=$(docker info --format '{{.DockerRootDir}}')",
                       "  python3 -c " + shlex.quote(space_check) + ' "$docker_root"', "fi",
                       shlex.join(["docker", "pull", "--platform", "linux/arm64", pins["base_image"]]),
                       shlex.join(["docker", "build", "--network", "none", "--pull=false",
                                   "--file", "runtime/Dockerfile", "--tag", pins["local_image"], "."]),
                       shlex.join(["docker", "run", "--rm", "--network", "none",
                                   "--env", "NVIDIA_VISIBLE_DEVICES=void", "--env", "CUDA_VISIBLE_DEVICES=",
                                   pins["local_image"], "--verify-runtime"])]) + "\n"


def compatibility_script(cache: str, model: dict, pins: dict, source: dict, serve_argv: tuple[str, ...]) -> str:
    draft = str(Path(model["hf_mount"]) / RELATIVE_ROOT / "models" / "draft" / source["checkpoints"]["draft"]["revision"])
    return "\n".join(["set -euo pipefail", shlex.join([
        "docker", "run", "--rm", "--network", "none", "--env", "NVIDIA_VISIBLE_DEVICES=void",
        "--env", "CUDA_VISIBLE_DEVICES=", "--env", "HF_HUB_OFFLINE=1", "--env", "TRANSFORMERS_OFFLINE=1",
        "--volume", cache + ":" + model["hf_mount"] + ":ro", "--entrypoint", "python3", pins["local_image"],
        RUNTIME_ROOT + "/check_compatibility.py", model["serve_path"], draft, json.dumps(serve_argv),
    ])]) + "\n"


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
    if labels.get("ai.spark-serve.recipe.base") != pins["base_image"].split("@", 1)[1]:
        raise ValueError(f"{role}: derivative base digest differs")
    if labels.get("tf.patches") != pins["patches_hash"]:
        raise ValueError(f"{role}: upstream patch set differs")
    return {"image": pins["local_image"], "image_id": image["Id"],
            "rootfs_layers": image["RootFS"]["Layers"], "base_image": pins["base_image"]}


def publish_script(remote: str, root: str, pins: dict, image: dict) -> str:
    receipt = {"manifest_sha256": pins["manifest_sha256"], "full_hash_verified": True,
               "cpu_compatibility_verified": True, "checkpoint_revisions": pins["checkpoint_revisions"], **image}
    return "\n".join([
        "set -euo pipefail", shlex.join(["mkdir", "-p", *[root + "/runtime-cache/" + name for name in ("torch_extensions", "triton", "cuda", "xdg")]]),
        shlex.join(["cp", remote + "/verify.py", remote + "/model-source.json", remote + "/source-pins.json", root + "/"]),
        "printf '%s\\n' " + shlex.quote(json.dumps(receipt, sort_keys=True)) + " > " + shlex.quote(root + "/prepared.json.tmp"),
        shlex.join(["mv", root + "/prepared.json.tmp", root + "/prepared.json"]),
    ]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "models.toml")
    parser.add_argument("--node", choices=("both",), default="both", help="TensorFold TP2 is prepared as a two-node allocation")
    parser.add_argument("--download-only", action="store_true", help="download and authenticate target and draft on head only")
    parser.add_argument("--skip-download", action="store_true", help="authenticate existing head snapshots without Internet downloads")
    args = parser.parse_args()
    pins, source = load_pins()
    cfg = tomllib.loads(args.catalog.read_text())
    model = validate_catalog(cfg, pins, source)
    head_root, worker_root = node_root(cfg, "head"), node_root(cfg, "worker")
    head_source = upload_recipe(cfg, "head")
    print("Preparing GLM TensorFold; serving containers remain running", flush=True)
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
    worker_source = upload_recipe(cfg, "worker")
    run_remote(cfg, "worker", "\n".join(["set -euo pipefail", "cd " + shlex.quote(worker_source),
               shlex.join(["python3", "check_space.py", worker_root, "--expected-manifest-sha256", pins["manifest_sha256"]])]) + "\n")
    script = copy_script(head_root, worker_root, head_ip, worker_ip, worker_user, image_to_copy)
    # Keep rsync's transport stdin separate from shell source. SSH concatenates
    # remote argv; quote the bash -c payload as one remote command argument.
    subprocess.run([*ssh_command(cfg, "head"), "bash -c " + shlex.quote(script)], check=True)
    run_remote(cfg, "worker", download_script(worker_source, worker_root, pins, True))
    worker_image = image_report(cfg, "worker", pins)
    if worker_image["image_id"] != head_image["image_id"] or worker_image["rootfs_layers"] != head_image["rootfs_layers"]:
        raise ValueError("worker derivative image differs from head")
    allocation = (("head", head_source, head_root, head_image), ("worker", worker_source, worker_root, worker_image))
    for role, remote, root, image in allocation:
        cache = str(Path(root).parent.parent)
        rank = 0 if role == "head" else 1
        run_remote(cfg, role, compatibility_script(cache, model, {**pins, "local_image": image["image_id"]}, source,
                                                   validate_tensorfold_plan(cfg, MODEL_KEY, rank=rank).argv))
    for role, remote, root, image in allocation:
        run_remote(cfg, role, publish_script(remote, root, pins, image))
    print("GLM TensorFold assets verified on both Sparks. Start with ./spark-serve up glm53-tensorfold.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as exc:
        sys.exit(f"GLM TensorFold preparation failed: {exc}")
