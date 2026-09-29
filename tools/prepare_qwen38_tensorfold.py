#!/usr/bin/env python3
"""Prepare Qwen TensorFold on one Spark without changing serving workloads."""
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
sys.path.insert(0, str(ROOT))
from spark_serve_nodes import launch_config
from spark_serve_tensorfold import validate_tensorfold_plan

MODEL_KEY = "qwen38-tensorfold"
RECIPE = ROOT / "recipes" / MODEL_KEY
UPSTREAM_REVISION = "a3aa89835022c55ca8e55008c37785954834e04f"
CHECKPOINT_REVISION = "dadefa8066e3be900a0d148d0f5a2f4eb1cf6534"
RUNTIME_ROOT = "/opt/spark-serve/qwen38-tensorfold"


def load_pins(recipe: Path = RECIPE) -> tuple[dict, dict]:
    pins = json.loads((recipe / "source-pins.json").read_text())
    raw = (recipe / "model-source.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != pins["manifest_sha256"]:
        raise ValueError("Qwen TensorFold checkpoint manifest SHA-256 mismatch")
    source = json.loads(raw)
    if source["revision"] != CHECKPOINT_REVISION or pins["checkpoint_revision"] != CHECKPOINT_REVISION:
        raise ValueError("Qwen TensorFold must use the pinned Vontra checkpoint")
    if source["model"] != "Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP":
        raise ValueError("Qwen TensorFold requires the Vontra MLX checkpoint")
    if pins["upstream_revision"] != UPSTREAM_REVISION:
        raise ValueError("Qwen TensorFold upstream source revision mismatch")
    if not re.fullmatch(r"ghcr\.io/miaai-lab/qwen3\.8-flash-next-single-dgx-spark-tensorfold@sha256:[0-9a-f]{64}", pins["base_image"]):
        raise ValueError("Qwen TensorFold base image must be immutable")
    expected = pins["runtime_sha256"]
    actual = {str(p.relative_to(recipe)) for p in (recipe / "runtime").rglob("*")
              if p.is_file() and "__pycache__" not in p.parts}
    if set(expected) != actual:
        raise ValueError("Qwen TensorFold runtime inventory differs from source pins")
    for name, digest in {**expected, **pins["preparation_sha256"]}.items():
        path = recipe / name
        if not path.resolve().is_relative_to(recipe.resolve()):
            raise ValueError("pinned source path escapes recipe")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Qwen TensorFold source SHA-256 mismatch: {name}")
    return pins, source


def snapshot_path(cache: str, source: dict) -> str:
    if not Path(cache).is_absolute():
        raise ValueError("checkpoint cache path must be absolute")
    return str(Path(cache) / "hub" / ("models--" + source["model"].replace("/", "--"))
               / "snapshots" / source["revision"])


def validate_catalog(cfg: dict, pins: dict, source: dict) -> dict:
    validate_tensorfold_plan(cfg, MODEL_KEY)
    model = cfg["models"][MODEL_KEY]
    if model.get("recipe") != MODEL_KEY or int(model.get("nnodes", 0)) != 1 or int(model.get("tensor_parallel", 0)) != 1:
        raise ValueError("qwen38-tensorfold requires its own TP1 single-Spark recipe")
    if model.get("image") != pins["local_image"] or model.get("hf_id") != source["model"]:
        raise ValueError("catalog image or checkpoint differs from the pinned Qwen recipe")
    expected = snapshot_path(model.get("hf_mount") or "/cache/huggingface", source)
    if model.get("serve_path") != expected:
        raise ValueError("Qwen TensorFold serve_path must use the pinned Vontra snapshot")
    required = [RUNTIME_ROOT + "/verify.py", expected,
                "--expected-manifest-sha256", pins["manifest_sha256"], "--verify-runtime"]
    if model.get("preflight_args") != required:
        raise ValueError("Qwen TensorFold preflight must authenticate the pinned snapshot")
    if model.get("wrapper") != "tensorfold" or model.get("hf_mount") != "/cache/huggingface":
        raise ValueError("TensorFold requires its wrapper and fixed read-only checkpoint mount")
    if model.get("tensorfold", {}).get("model_revision") != source["revision"]:
        raise ValueError("TensorFold catalog model revision differs from source pins")
    cluster = cfg["cluster"]
    host = cluster["head"]
    if not isinstance(host, str) or not host or host.startswith("-") or any(c.isspace() for c in host):
        raise ValueError("invalid selected Spark SSH host")
    snapshot_path(cluster["hf_cache_host"], source)
    return model


def ssh_command(cfg: dict) -> list[str]:
    return ["ssh", *cfg["cluster"].get("ssh_opts", ["-o", "BatchMode=yes"]),
            "-o", "ConnectTimeout=8", cfg["cluster"]["head"]]


def run_remote(cfg: dict, script: str) -> None:
    subprocess.run([*ssh_command(cfg), "bash -s"], input=script, text=True, check=True)


def write_payload(archive, recipe: Path = RECIPE) -> None:
    with tarfile.open(fileobj=archive, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for path in sorted(recipe.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            data = path.read_bytes()
            info = tarfile.TarInfo(str(path.relative_to(recipe)))
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            tar.addfile(info, io.BytesIO(data))


def upload_recipe(cfg: dict) -> str:
    result = subprocess.run([*ssh_command(cfg), "mktemp -d /tmp/spark-serve-qwen38-tensorfold.XXXXXXXX"],
                            capture_output=True, text=True, check=True, timeout=30)
    remote = result.stdout.strip()
    if not re.fullmatch(r"/tmp/spark-serve-qwen38-tensorfold\.[A-Za-z0-9]+", remote):
        raise ValueError("unexpected remote preparation directory")
    with tempfile.TemporaryFile() as archive:
        write_payload(archive)
        archive.seek(0)
        subprocess.run([*ssh_command(cfg), "tar -xf - -C " + shlex.quote(remote)],
                       stdin=archive, check=True, timeout=120)
    return remote


def build_script(remote: str, pins: dict) -> str:
    space_check = "import shutil,sys; free=shutil.disk_usage(sys.argv[1]).free; " \
                  "sys.exit('TensorFold image needs 35 GB free under Docker storage' if free < 35_000_000_000 else 0)"
    return "\n".join([
        "set -euo pipefail", "cd " + shlex.quote(remote),
        "if ! " + shlex.join(["docker", "image", "inspect", pins["base_image"]]) + " >/dev/null 2>&1; then",
        "  docker_root=$(docker info --format '{{.DockerRootDir}}')",
        "  python3 -c " + shlex.quote(space_check) + ' "$docker_root"', "fi",
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
    # A cached snapshot is checked first: a good cache never installs
    # download dependencies, reaches the Hub, or duplicates checkpoint bytes.
    download = "\n".join([
        "from huggingface_hub import snapshot_download",
        f"print(snapshot_download({source['model']!r}, revision={source['revision']!r}, "
        f"cache_dir={str(Path(cache) / 'hub')!r}, token=False, max_workers=2), flush=True)",
    ])
    commands += ["if " + check + "; then", "  echo 'Reusing fully authenticated Vontra checkpoint'", "else",
                 "  " + shlex.join(["python3", "check_space.py", snapshot, "--cache-dir", cache]),
                 '  venv="$HOME/.local/share/spark-serve/recipes/qwen38-tensorfold/download-env"',
                 '  python3 -m venv "$venv"',
                 '  "$venv/bin/pip" install --disable-pip-version-check huggingface_hub==1.31.0',
                 '  HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_XET_NUM_CONCURRENT_RANGE_GETS=4 "$venv/bin/python" -u -c ' + shlex.quote(download),
                 '  "$venv/bin/python" verify.py ' + shlex.join([snapshot, "--expected-manifest-sha256",
                     pins["manifest_sha256"], "--repair-cache", str(Path(cache) / "hub")]), "fi"]
    return "\n".join(commands) + "\n"


def compatibility_script(cache: str, model: dict, pins: dict, serve_argv: tuple[str, ...]) -> str:
    return "\n".join(["set -euo pipefail", shlex.join([
        "docker", "run", "--rm", "--network", "none", "--env", "NVIDIA_VISIBLE_DEVICES=void",
        "--env", "CUDA_VISIBLE_DEVICES=", "--env", "HF_HUB_OFFLINE=1", "--env", "TRANSFORMERS_OFFLINE=1",
        "--volume", cache + ":" + model.get("hf_mount", "/cache/huggingface") + ":ro",
        "--entrypoint", "python3", pins["local_image"], RUNTIME_ROOT + "/check_compatibility.py", model["serve_path"],
        json.dumps(serve_argv),
    ])]) + "\n"


def image_report(cfg: dict, pins: dict) -> dict:
    result = subprocess.run([*ssh_command(cfg), shlex.join(["docker", "image", "inspect", pins["local_image"]])],
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
        "set -euo pipefail", shlex.join(["mkdir", "-p", *[root + "/runtime-cache/" + name for name in ("torch_extensions", "triton", "cuda", "xdg")]]),
        shlex.join(["cp", remote + "/verify.py", remote + "/model-source.json", remote + "/source-pins.json", root + "/"]),
        "printf '%s\\n' " + shlex.quote(json.dumps(receipt, sort_keys=True)) + " > " + shlex.quote(root + "/prepared.json.tmp"),
        shlex.join(["mv", root + "/prepared.json.tmp", root + "/prepared.json"]),
    ]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "models.toml")
    parser.add_argument("--node", choices=("head", "worker"), default="head")
    parser.add_argument("--skip-download", action="store_true", help="require the pinned Vontra checkpoint already cached")
    parser.add_argument("--image-only", action="store_true", help="build and verify CPU runtime only; no checkpoint readiness receipt")
    args = parser.parse_args()
    pins, source = load_pins()
    cfg = launch_config(tomllib.loads(args.catalog.read_text()), MODEL_KEY, args.node)
    model = validate_catalog(cfg, pins, source)
    remote = upload_recipe(cfg)
    print(f"Preparing Qwen TensorFold on {cfg['cluster']['head']}; serving workloads remain running", flush=True)
    run_remote(cfg, build_script(remote, pins))
    report = image_report(cfg, pins)
    if args.image_only:
        print(json.dumps(report, sort_keys=True), flush=True)
        return
    cache = str(cfg["cluster"]["hf_cache_host"])
    run_remote(cfg, checkpoint_script(remote, cache, pins, source, args.skip_download))
    # The exact launch arguments are parsed inside the authenticated image before
    # publishing readiness; no CUDA or model loader is invoked by this check.
    run_remote(cfg, compatibility_script(cache, model, {**pins, "local_image": report["image_id"]},
                                        validate_tensorfold_plan(cfg, MODEL_KEY).argv))
    run_remote(cfg, publish_script(remote, cache, pins, report))
    print(f"Qwen TensorFold runtime and pinned checkpoint ready on {args.node}; no model was launched.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as exc:
        sys.exit(f"Qwen TensorFold preparation failed: {exc}")
