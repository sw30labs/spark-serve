#!/usr/bin/env python3
"""Prepare Qwen dual vLLM 0.30 on both Sparks without changing serving workloads."""
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
from spark_serve_nodes import placement

MODEL_KEY = "qwen38-dual"
RECIPE = ROOT / "recipes" / MODEL_KEY
UPSTREAM_REVISION = "cd839d0b62f6c737178a0c2037733420e0524ebf"
CHECKPOINT_REVISION = "fc694b54fb0174e0913e6adf86691ef85a4ead47"
RUNTIME_ROOT = "/opt/spark-serve/qwen38-dual"


def node_cache(cfg: dict, role: str) -> str:
    cluster = cfg["cluster"]
    cache = cluster.get("worker_hf_cache_host") if role == "worker" else None
    cache = cache or cluster["hf_cache_host"]
    if not isinstance(cache, str) or not Path(cache).is_absolute():
        raise ValueError("checkpoint cache paths must be absolute")
    return cache


def load_pins(recipe: Path = RECIPE) -> tuple[dict, dict]:
    pins = json.loads((recipe / "source-pins.json").read_text())
    raw = (recipe / "model-source.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != pins["manifest_sha256"]:
        raise ValueError("Qwen dual checkpoint manifest SHA-256 mismatch")
    source = json.loads(raw)
    if source["revision"] != CHECKPOINT_REVISION or pins["checkpoint_revision"] != CHECKPOINT_REVISION:
        raise ValueError("Qwen dual must reuse the existing pinned NVIDIA checkpoint")
    if source["model"] != "nvidia/Qwen3.8-Flash-Next-NVFP4":
        raise ValueError("Qwen dual requires the NVIDIA NVFP4 checkpoint")
    if pins["upstream_revision"] != UPSTREAM_REVISION:
        raise ValueError("Qwen dual upstream source revision mismatch")
    if not re.fullmatch(r"docker\.io/vllm/vllm-openai@sha256:[0-9a-f]{64}", pins["base_image"]):
        raise ValueError("Qwen dual base image must be immutable")
    expected = pins["runtime_sha256"]
    actual = {str(p.relative_to(recipe)) for p in (recipe / "runtime").rglob("*")
              if p.is_file() and "__pycache__" not in p.parts}
    if set(expected) != actual:
        raise ValueError("Qwen dual runtime inventory differs from source pins")
    for name, digest in {**expected, **pins["preparation_sha256"]}.items():
        path = recipe / name
        if not path.resolve().is_relative_to(recipe.resolve()):
            raise ValueError("pinned source path escapes recipe")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Qwen dual source SHA-256 mismatch: {name}")
    return pins, source


def snapshot_path(cache: str, source: dict) -> str:
    if not Path(cache).is_absolute():
        raise ValueError("checkpoint cache path must be absolute")
    return str(Path(cache) / "hub" / ("models--" + source["model"].replace("/", "--"))
               / "snapshots" / source["revision"])


def validate_catalog(cfg: dict, pins: dict, source: dict) -> dict:
    model = cfg["models"][MODEL_KEY]
    if model.get("recipe") != MODEL_KEY or int(model.get("nnodes", 0)) != 2 or int(model.get("tensor_parallel", 0)) != 2:
        raise ValueError("qwen38-dual requires its own TP2 dual-Spark recipe")
    if model.get("image") != pins["local_image"] or model.get("hf_id") != source["model"]:
        raise ValueError("catalog image or checkpoint differs from the pinned Qwen recipe")
    if model.get("wrapper") != "vllm":
        raise ValueError("Qwen dual requires the existing vLLM wrapper")
    arguments = model.get("vllm", {}).get("args", [])
    for flag, expected in (("--tensor-parallel-size", "2"), ("--mm-encoder-tp-mode", "data"),
                           ("--kv-cache-dtype", "auto")):
        if arguments.count(flag) != 1 or arguments[arguments.index(flag) + 1:arguments.index(flag) + 2] != [expected]:
            raise ValueError(f"Qwen dual requires {flag} {expected}")
    if "--enable-expert-parallel" not in arguments:
        raise ValueError("Qwen dual requires expert parallelism")
    expected = snapshot_path(model.get("hf_mount") or "/cache/huggingface", source)
    if model.get("serve_path") != expected:
        raise ValueError("Qwen dual serve_path must reuse the pinned NVIDIA snapshot")
    required = [RUNTIME_ROOT + "/verify.py", expected,
                "--expected-manifest-sha256", pins["manifest_sha256"], "--verify-runtime"]
    if model.get("preflight_args") != required:
        raise ValueError("Qwen dual preflight must authenticate the pinned snapshot")
    cluster = cfg["cluster"]
    placement(cfg, model, "both")
    for role in ("head", "worker"):
        host = cluster[role]
        if not isinstance(host, str) or not host or host.startswith("-") or any(c.isspace() for c in host):
            raise ValueError("invalid Spark SSH host")
        snapshot_path(node_cache(cfg, role), source)
    return model


def ssh_command(cfg: dict, role: str) -> list[str]:
    return ["ssh", *cfg["cluster"].get("ssh_opts", ["-o", "BatchMode=yes"]),
            "-o", "ConnectTimeout=8", cfg["cluster"][role]]


def run_remote(cfg: dict, role: str, script: str) -> None:
    subprocess.run([*ssh_command(cfg, role), "bash -s"], input=script, text=True, check=True)


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
    result = subprocess.run([*ssh_command(cfg, role), "mktemp -d /tmp/spark-serve-qwen38-dual.XXXXXXXX"],
                            capture_output=True, text=True, check=True, timeout=30)
    remote = result.stdout.strip()
    if not re.fullmatch(r"/tmp/spark-serve-qwen38-dual\.[A-Za-z0-9]+", remote):
        raise ValueError("unexpected remote preparation directory")
    with tempfile.TemporaryFile() as archive:
        write_payload(archive)
        archive.seek(0)
        subprocess.run([*ssh_command(cfg, role), "tar -xf - -C " + shlex.quote(remote)],
                       stdin=archive, check=True, timeout=120)
    return remote


def build_script(remote: str, pins: dict) -> str:
    return "\n".join([
        "set -euo pipefail", "cd " + shlex.quote(remote),
        shlex.join(["docker", "pull", "--platform", "linux/arm64", pins["base_image"]]),
        shlex.join(["docker", "build", "--network", "none", "--pull=false", "--file", "runtime/Dockerfile",
                    "--tag", pins["local_image"], "."]),
        shlex.join(["docker", "run", "--rm", "--network", "none", "--env", "NVIDIA_VISIBLE_DEVICES=void",
                    "--env", "CUDA_VISIBLE_DEVICES=", pins["local_image"], "--verify-runtime"]),
    ]) + "\n"


def checkpoint_script(remote: str, cache: str, pins: dict, source: dict, skip_download: bool) -> str:
    snapshot = snapshot_path(cache, source)
    check = shlex.join(["python3", "verify.py", snapshot, "--expected-manifest-sha256", pins["manifest_sha256"]])
    commands = ["set -euo pipefail", "cd " + shlex.quote(remote)]
    if skip_download:
        return "\n".join([*commands, check]) + "\n"
    # The existing cached snapshot is checked first: a good cache never installs
    # download dependencies, reaches the Hub, or duplicates checkpoint bytes.
    download = "\n".join([
        "from huggingface_hub import snapshot_download",
        f"print(snapshot_download({source['model']!r}, revision={source['revision']!r}, "
        f"cache_dir={str(Path(cache) / 'hub')!r}, token=False, max_workers=2), flush=True)",
    ])
    commands += ["if " + check + "; then", "  echo 'Reusing fully authenticated NVIDIA checkpoint'", "else",
                 '  venv="$HOME/.local/share/spark-serve/recipes/qwen38-dual/download-env"',
                 '  python3 -m venv "$venv"',
                 '  "$venv/bin/pip" install --disable-pip-version-check huggingface_hub==1.31.0',
                 '  HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_XET_NUM_CONCURRENT_RANGE_GETS=4 "$venv/bin/python" -u -c ' + shlex.quote(download),
                 '  "$venv/bin/python" verify.py ' + shlex.join([snapshot, "--expected-manifest-sha256",
                     pins["manifest_sha256"], "--repair-cache", str(Path(cache) / "hub")]), "fi"]
    return "\n".join(commands) + "\n"


def compatibility_script(cache: str, model: dict, pins: dict) -> str:
    return "\n".join(["set -euo pipefail", shlex.join([
        "docker", "run", "--rm", "--network", "none", "--env", "NVIDIA_VISIBLE_DEVICES=void",
        "--env", "CUDA_VISIBLE_DEVICES=", "--env", "HF_HUB_OFFLINE=1", "--env", "TRANSFORMERS_OFFLINE=1",
        "--volume", cache + ":" + model.get("hf_mount", "/cache/huggingface") + ":ro",
        "--entrypoint", "python3", pins["local_image"], RUNTIME_ROOT + "/check_compatibility.py", model["serve_path"],
    ])]) + "\n"


def image_report(cfg: dict, role: str, pins: dict) -> dict:
    result = subprocess.run([*ssh_command(cfg, role), shlex.join(["docker", "image", "inspect", pins["local_image"]])],
                            capture_output=True, text=True, check=True, timeout=30)
    image = json.loads(result.stdout)[0]
    if image.get("Architecture") != "arm64" or image.get("Os") != "linux":
        raise ValueError("Qwen derivative image must be Linux ARM64")
    labels = image.get("Config", {}).get("Labels", {})
    if labels.get("ai.spark-serve.recipe.commit") != pins["upstream_revision"]:
        raise ValueError("Qwen derivative image source revision differs")
    if labels.get("ai.spark-serve.recipe.base") != pins["base_image"].split("@", 1)[1]:
        raise ValueError("Qwen derivative image base digest differs")
    return {"image": pins["local_image"], "image_id": image["Id"], "base_image": pins["base_image"],
            "rootfs_layers": image["RootFS"]["Layers"]}


def publish_script(remote: str, cache: str, pins: dict, report: dict) -> str:
    root = str(Path(cache) / "spark-serve" / MODEL_KEY)
    receipt = {"manifest_sha256": pins["manifest_sha256"], "checkpoint_revision": CHECKPOINT_REVISION,
               "full_hash_verified": True, "cpu_compatibility_verified": True, **report}
    return "\n".join([
        "set -euo pipefail", "mkdir -p " + shlex.quote(root + "/runtime-cache"),
        shlex.join(["cp", remote + "/verify.py", remote + "/model-source.json", remote + "/source-pins.json", root + "/"]),
        "printf '%s\\n' " + shlex.quote(json.dumps(receipt, sort_keys=True)) + " > " + shlex.quote(root + "/prepared.json.tmp"),
        shlex.join(["mv", root + "/prepared.json.tmp", root + "/prepared.json"]),
    ]) + "\n"


def remote_text(cfg: dict, role: str, script: str, timeout: int = 60) -> str:
    result = subprocess.run([*ssh_command(cfg, role), "bash -s"], input=script,
                            text=True, capture_output=True, check=True, timeout=timeout)
    return result.stdout.strip()


def qsfp_address(cfg: dict, role: str) -> str:
    interface = str(cfg["cluster"]["nccl"]["NCCL_SOCKET_IFNAME"])
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", interface):
        raise ValueError("one explicit QSFP socket interface is required")
    raw = remote_text(cfg, role, "ip -j -4 addr show dev " + shlex.quote(interface))
    addresses = [x["local"] for device in json.loads(raw) for x in device.get("addr_info", [])
                 if x.get("family") == "inet"]
    if len(addresses) != 1:
        raise ValueError(f"{role}: require one IPv4 address on {interface}")
    return str(ipaddress.IPv4Address(addresses[0]))


def checkpoint_probe_script(remote: str, cache: str, pins: dict, source: dict) -> str:
    check = shlex.join(["python3", "verify.py", snapshot_path(cache, source),
                        "--expected-manifest-sha256", pins["manifest_sha256"]])
    return "\n".join(["set -euo pipefail", "cd " + shlex.quote(remote),
                       "if " + check + "; then", "  echo SPARK_CHECKPOINT_AUTHENTICATED",
                       "else", "  echo SPARK_CHECKPOINT_NEEDS_COPY", "fi"]) + "\n"


def copy_script(remote: str, head_cache: str, worker_cache: str, source: dict,
                pins: dict, head_ip: str, worker_ip: str, user: str,
                copy_checkpoint: bool, image: str | None) -> str:
    for address in (head_ip, worker_ip):
        ipaddress.IPv4Address(address)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", user):
        raise ValueError("invalid worker SSH username")
    transport = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                 "-o", "StrictHostKeyChecking=accept-new", "-o", "BindAddress=" + head_ip]
    peer = user + "@" + worker_ip
    commands = ["set -euo pipefail"]
    if copy_checkpoint:
        repo = "models--" + source["model"].replace("/", "--")
        head_root, worker_root = str(Path(head_cache) / "hub" / repo), str(Path(worker_cache) / "hub" / repo)
        filelist = remote + "/snapshot-transfer.txt"
        commands += [
            shlex.join(["python3", remote + "/prepare_transfer.py", snapshot_path(head_cache, source), filelist,
                        "--expected-manifest-sha256", pins["manifest_sha256"]]),
            shlex.join([*transport, "-n", peer, "mkdir -p " + shlex.quote(worker_root)]),
            shlex.join(["rsync", "-a", "--checksum", "--partial", "--protect-args", "--info=stats2",
                        "--files-from", filelist, "-e", shlex.join(transport),
                        head_root + "/", peer + ":" + worker_root + "/"]),
        ]
    if image is not None:
        commands.append(shlex.join(["docker", "save", image]) + " | " + shlex.join([*transport, peer, "docker load"]))
    return "\n".join(commands) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "models.toml")
    parser.add_argument("--node", choices=("both",), default="both", help="TP2 requires both Sparks")
    parser.add_argument("--skip-download", action="store_true", help="require the pinned NVIDIA checkpoint on the head")
    parser.add_argument("--image-only", action="store_true", help="prepare the CPU-verified runtime on both nodes only")
    args = parser.parse_args()
    pins, source = load_pins()
    cfg = tomllib.loads(args.catalog.read_text())
    model = validate_catalog(cfg, pins, source)
    head_cache, worker_cache = node_cache(cfg, "head"), node_cache(cfg, "worker")
    head_source = upload_recipe(cfg, "head")
    print("Preparing Qwen dual; serving workloads remain running", flush=True)
    run_remote(cfg, "head", build_script(head_source, pins))
    head_image = image_report(cfg, "head", pins)
    worker_source = upload_recipe(cfg, "worker")
    copy_checkpoint = False
    if not args.image_only:
        run_remote(cfg, "head", checkpoint_script(head_source, head_cache, pins, source, args.skip_download))
        probe = remote_text(cfg, "worker", checkpoint_probe_script(worker_source, worker_cache, pins, source), timeout=3600)
        marker = probe.splitlines()[-1]
        if marker not in ("SPARK_CHECKPOINT_AUTHENTICATED", "SPARK_CHECKPOINT_NEEDS_COPY"):
            raise ValueError("worker checkpoint probe returned no authentication result")
        copy_checkpoint = marker == "SPARK_CHECKPOINT_NEEDS_COPY"
    worker_image_id = remote_text(cfg, "worker", shlex.join([
        "docker", "image", "inspect", "--format", "{{.Id}}", pins["local_image"]]) + " 2>/dev/null || true")
    image_to_copy = None if worker_image_id == head_image["image_id"] else pins["local_image"]
    if copy_checkpoint or image_to_copy:
        head_ip, worker_ip = qsfp_address(cfg, "head"), qsfp_address(cfg, "worker")
        worker_user = remote_text(cfg, "worker", "id -un")
        script = copy_script(head_source, head_cache, worker_cache, source, pins,
                             head_ip, worker_ip, worker_user, copy_checkpoint, image_to_copy)
        # Keep rsync/docker streaming stdin separate from the shell source.
        subprocess.run([*ssh_command(cfg, "head"), "bash -c " + shlex.quote(script)], check=True)
    worker_image = image_report(cfg, "worker", pins)
    if worker_image["image_id"] != head_image["image_id"] or worker_image["rootfs_layers"] != head_image["rootfs_layers"]:
        raise ValueError("worker derivative image differs from head")
    run_remote(cfg, "worker", shlex.join(["docker", "run", "--rm", "--network", "none",
                "--env", "NVIDIA_VISIBLE_DEVICES=void", "--env", "CUDA_VISIBLE_DEVICES=",
                pins["local_image"], "--verify-runtime"]) + "\n")
    if args.image_only:
        print(json.dumps({"head": head_image, "worker": worker_image}, sort_keys=True), flush=True)
        return
    if copy_checkpoint:
        # A good cache was already fully authenticated by the worker probe.
        # Only newly transferred bytes need a second complete hash pass.
        run_remote(cfg, "worker", checkpoint_script(worker_source, worker_cache, pins, source, True))
    for role, remote, cache, image in (("head", head_source, head_cache, head_image),
                                     ("worker", worker_source, worker_cache, worker_image)):
        run_remote(cfg, role, compatibility_script(cache, model, pins))
    for role, remote, cache, image in (("head", head_source, head_cache, head_image),
                                     ("worker", worker_source, worker_cache, worker_image)):
        run_remote(cfg, role, publish_script(remote, cache, pins, image))
    print("Qwen dual runtime and checkpoint ready on both Sparks; no model was launched.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as exc:
        sys.exit(f"Qwen dual preparation failed: {exc}")
