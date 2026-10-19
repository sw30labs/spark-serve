#!/usr/bin/env python3
"""Prepare native Qwen dual TensorFold on both Sparks without changing serving workloads."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spark_serve_tensorfold import validate_tensorfold_plan
from tools import prepare_qwen38_dual as dual

MODEL_KEY = "qwen38-dual-tensorfold"
RECIPE = ROOT / "recipes" / MODEL_KEY
UPSTREAM_REVISION = "be3ed857ab1055ca47b36134d22691f00f2bb94a"
CHECKPOINT_REVISION = dual.CHECKPOINT_REVISION
RUNTIME_ROOT = "/opt/spark-serve/" + MODEL_KEY
TEMP_PREFIX = "/tmp/spark-serve-" + MODEL_KEY


def load_pins(recipe: Path = RECIPE) -> tuple[dict, dict]:
    pins = json.loads((recipe / "source-pins.json").read_text())
    raw = (recipe / "model-source.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != pins["manifest_sha256"]:
        raise ValueError("Qwen dual TensorFold checkpoint manifest SHA-256 mismatch")
    source = json.loads(raw)
    if source["revision"] != CHECKPOINT_REVISION or pins["checkpoint_revision"] != CHECKPOINT_REVISION:
        raise ValueError("Qwen dual TensorFold must reuse the pinned NVIDIA checkpoint")
    if source["model"] != "nvidia/Qwen3.8-Flash-Next-NVFP4":
        raise ValueError("Qwen dual TensorFold requires the NVIDIA NVFP4 checkpoint")
    if pins["upstream_revision"] != UPSTREAM_REVISION:
        raise ValueError("Qwen dual TensorFold upstream source revision mismatch")
    if not re.fullmatch(r"ghcr\.io/miaai-lab/qwen3\.8-flash-dual-dgx-sparks-tensorfold@sha256:[0-9a-f]{64}",
                        pins["base_image"]):
        raise ValueError("Qwen dual TensorFold base image must be immutable")
    expected = pins["runtime_sha256"]
    actual = {str(p.relative_to(recipe)) for p in (recipe / "runtime").rglob("*")
              if p.is_file() and "__pycache__" not in p.parts}
    if set(expected) != actual:
        raise ValueError("Qwen dual TensorFold runtime inventory differs from source pins")
    for name, digest in {**expected, **pins["preparation_sha256"]}.items():
        path = recipe / name
        if not path.resolve().is_relative_to(recipe.resolve()):
            raise ValueError("pinned source path escapes recipe")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Qwen dual TensorFold source SHA-256 mismatch: {name}")
    return pins, source


def validate_catalog(cfg: dict, pins: dict, source: dict) -> dict:
    validate_tensorfold_plan(cfg, MODEL_KEY)
    validate_tensorfold_plan(cfg, MODEL_KEY, rank=1)
    model = cfg["models"][MODEL_KEY]
    if model.get("recipe") != MODEL_KEY:
        raise ValueError("qwen38-dual-tensorfold requires its own recipe")
    if model.get("image") != pins["local_image"]:
        raise ValueError("catalog image differs from the pinned TensorFold derivative")
    expected = dual.snapshot_path(model.get("hf_mount", "/cache/huggingface"), source)
    required = [RUNTIME_ROOT + "/verify.py", expected,
                "--expected-manifest-sha256", pins["manifest_sha256"], "--verify-runtime"]
    if model.get("serve_path") != expected or model.get("preflight_args") != required:
        raise ValueError("Qwen dual TensorFold must serve and authenticate the pinned snapshot")
    for role in ("head", "worker"):
        dual.snapshot_path(dual.node_cache(cfg, role), source)
    return model


def upload_recipe(cfg: dict, role: str) -> str:
    remote = dual.remote_text(cfg, role, f"mktemp -d {TEMP_PREFIX}.XXXXXXXX")
    if not re.fullmatch(re.escape(TEMP_PREFIX) + r"\.[A-Za-z0-9]+", remote):
        raise ValueError("unexpected remote preparation directory")
    with tempfile.TemporaryFile() as archive:
        dual.write_payload(archive, RECIPE)
        archive.seek(0)
        subprocess.run([*dual.ssh_command(cfg, role), "tar -xf - -C " + shlex.quote(remote)],
                       stdin=archive, check=True, timeout=120)
    return remote


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
                       verify_image_command(pins["local_image"])]) + "\n"


def verify_image_command(image: str) -> str:
    return shlex.join(["docker", "run", "--rm", "--network", "none", "--env", "NVIDIA_VISIBLE_DEVICES=void",
                       "--env", "CUDA_VISIBLE_DEVICES=", image, "--verify-runtime"])


def image_report(cfg: dict, role: str, pins: dict) -> dict:
    image = json.loads(dual.remote_text(cfg, role, shlex.join(["docker", "image", "inspect", pins["local_image"]])))[0]
    if image.get("Architecture") != "arm64" or image.get("Os") != "linux":
        raise ValueError(f"{role}: wrong image platform")
    labels = image.get("Config", {}).get("Labels", {})
    if labels.get("ai.spark-serve.recipe.commit") != pins["upstream_revision"]:
        raise ValueError(f"{role}: image source revision does not match")
    if labels.get("ai.spark-serve.recipe.base") != pins["base_image"].split("@", 1)[1]:
        raise ValueError(f"{role}: derivative base digest differs")
    if labels.get("tf.patches") != pins["patches_hash"] or labels.get("tf.kernels") != "present":
        raise ValueError(f"{role}: upstream patch set or kernel set differs")
    return {"image": pins["local_image"], "image_id": image["Id"],
            "rootfs_layers": image["RootFS"]["Layers"], "base_image": pins["base_image"]}


def compatibility_script(cache: str, model: dict, image_id: str, serve_argv: tuple[str, ...]) -> str:
    return "\n".join(["set -euo pipefail", shlex.join([
        "docker", "run", "--rm", "--network", "none", "--env", "NVIDIA_VISIBLE_DEVICES=void",
        "--env", "CUDA_VISIBLE_DEVICES=", "--env", "HF_HUB_OFFLINE=1",
        "--volume", cache + ":" + model.get("hf_mount", "/cache/huggingface") + ":ro", "--entrypoint", "python3", image_id,
        RUNTIME_ROOT + "/check_compatibility.py", model["serve_path"], json.dumps(serve_argv),
    ])]) + "\n"


def publish_script(remote: str, cache: str, pins: dict, report: dict) -> str:
    root = str(Path(cache) / "spark-serve" / MODEL_KEY)
    receipt = {"manifest_sha256": pins["manifest_sha256"], "checkpoint_revision": CHECKPOINT_REVISION,
               "full_hash_verified": True, "cpu_compatibility_verified": True, **report}
    return "\n".join([
        "set -euo pipefail",
        shlex.join(["mkdir", "-p", *[root + "/runtime-cache/" + name for name in ("torch_extensions", "triton", "cuda", "xdg")]]),
        shlex.join(["cp", remote + "/verify.py", remote + "/model-source.json", remote + "/source-pins.json", root + "/"]),
        "printf '%s\\n' " + shlex.quote(json.dumps(receipt, sort_keys=True)) + " > " + shlex.quote(root + "/prepared.json.tmp"),
        shlex.join(["mv", root + "/prepared.json.tmp", root + "/prepared.json"]),
    ]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "models.toml")
    parser.add_argument("--node", choices=("both",), default="both", help="TP2 is prepared as a two-node allocation")
    parser.add_argument("--skip-download", action="store_true", help="require the pinned NVIDIA checkpoint on the head")
    parser.add_argument("--image-only", action="store_true", help="prepare the CPU-verified runtime on both nodes only")
    args = parser.parse_args()
    pins, source = load_pins()
    cfg = tomllib.loads(args.catalog.read_text())
    model = validate_catalog(cfg, pins, source)
    head_cache, worker_cache = dual.node_cache(cfg, "head"), dual.node_cache(cfg, "worker")
    head_source = upload_recipe(cfg, "head")
    print("Preparing Qwen dual TensorFold; serving workloads remain running", flush=True)
    dual.run_remote(cfg, "head", build_script(head_source, pins))
    head_image = image_report(cfg, "head", pins)
    worker_source = upload_recipe(cfg, "worker")
    copy_checkpoint = False
    if not args.image_only:
        # The NVFP4 snapshot is shared with qwen38-dual: an authenticated cache
        # is reused as is, and only a missing or damaged one is fetched or copied.
        dual.run_remote(cfg, "head", dual.checkpoint_script(head_source, head_cache, pins, source, args.skip_download))
        probe = dual.remote_text(cfg, "worker", dual.checkpoint_probe_script(worker_source, worker_cache, pins, source),
                                 timeout=3600)
        marker = probe.splitlines()[-1]
        if marker not in ("SPARK_CHECKPOINT_AUTHENTICATED", "SPARK_CHECKPOINT_NEEDS_COPY"):
            raise ValueError("worker checkpoint probe returned no authentication result")
        copy_checkpoint = marker == "SPARK_CHECKPOINT_NEEDS_COPY"
    worker_image_id = dual.remote_text(cfg, "worker", shlex.join([
        "docker", "image", "inspect", "--format", "{{.Id}}", pins["local_image"]]) + " 2>/dev/null || true")
    image_to_copy = None if worker_image_id == head_image["image_id"] else pins["local_image"]
    if copy_checkpoint or image_to_copy:
        head_ip, worker_ip = dual.qsfp_address(cfg, "head"), dual.qsfp_address(cfg, "worker")
        worker_user = dual.remote_text(cfg, "worker", "id -un")
        script = dual.copy_script(head_source, head_cache, worker_cache, source, pins,
                                  head_ip, worker_ip, worker_user, copy_checkpoint, image_to_copy)
        # Keep rsync/docker streaming stdin separate from the shell source.
        subprocess.run([*dual.ssh_command(cfg, "head"), "bash -c " + shlex.quote(script)], check=True)
    worker_image = image_report(cfg, "worker", pins)
    if worker_image["image_id"] != head_image["image_id"] or worker_image["rootfs_layers"] != head_image["rootfs_layers"]:
        raise ValueError("worker derivative image differs from head")
    dual.run_remote(cfg, "worker", verify_image_command(pins["local_image"]) + "\n")
    if args.image_only:
        print(json.dumps({"head": head_image, "worker": worker_image}, sort_keys=True), flush=True)
        return
    if copy_checkpoint:
        dual.run_remote(cfg, "worker", dual.checkpoint_script(worker_source, worker_cache, pins, source, True))
    allocation = (("head", head_source, head_cache, head_image), ("worker", worker_source, worker_cache, worker_image))
    for rank, (role, remote, cache, image) in enumerate(allocation):
        argv = validate_tensorfold_plan(cfg, MODEL_KEY, rank=rank).argv
        dual.run_remote(cfg, role, compatibility_script(cache, model, image["image_id"], argv))
    for role, remote, cache, image in allocation:
        dual.run_remote(cfg, role, publish_script(remote, cache, pins, image))
    print("Qwen dual TensorFold runtime and checkpoint ready on both Sparks; no model was launched. "
          "Start with ./spark-serve up qwen38-dual-tensorfold --node both.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as exc:
        sys.exit(f"Qwen dual TensorFold preparation failed: {exc}")
