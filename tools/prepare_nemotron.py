#!/usr/bin/env python3
"""Prepare the pinned single-Spark Nemotron runtime without stopping a serving model."""
from __future__ import annotations

import argparse
import json
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

RECIPE = ROOT / "recipes" / "nemotron-super-nvfp4"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "models.toml")
    parser.add_argument("--node", choices=("head", "worker"), default="head")
    parser.add_argument("--skip-download", action="store_true", help="require the pinned snapshot already cached")
    args = parser.parse_args()
    cfg = launch_config(tomllib.loads(args.catalog.read_text()), "nemotron-super", args.node)
    cluster, model = cfg["cluster"], cfg["models"]["nemotron-super"]
    if int(model.get("nnodes", 0)) != 1 or model.get("recipe") != "nemotron-super-nvfp4":
        parser.error("this preparation recipe requires nemotron-super with nnodes=1")
    source = json.loads((RECIPE / "model-source.json").read_text())
    ssh = ["ssh", *cluster.get("ssh_opts", ["-o", "BatchMode=yes"]), cluster["head"]]
    # A new staging directory prevents an interrupted upload from altering a build in progress.
    proc = subprocess.run([*ssh, "mktemp -d /tmp/spark-serve-nemotron-super.XXXXXXXX"],
                          capture_output=True, text=True, check=True)
    remote = proc.stdout.strip()
    if not remote.startswith("/tmp/spark-serve-nemotron-super.") or "\n" in remote:
        raise RuntimeError("unexpected remote preparation directory")
    with tempfile.TemporaryFile() as archive:
        with tarfile.open(fileobj=archive, mode="w") as tar:
            for path in sorted(RECIPE.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    tar.add(path, arcname=str(path.relative_to(RECIPE)), recursive=False)
        archive.seek(0)
        subprocess.run([*ssh, "tar -xf - -C " + shlex.quote(remote)], stdin=archive, check=True)
    cache = str(cluster["hf_cache_host"])
    snapshot = str(Path(cache) / "hub" / ("models--" + source["model"].replace("/", "--")) / "snapshots" / source["revision"])
    download_code = "\n".join([
        "from huggingface_hub import snapshot_download",
        f"print(snapshot_download({source['model']!r}, revision={source['revision']!r}, cache_dir={str(Path(cache) / 'hub')!r}, token=False, max_workers=2), flush=True)",
    ])
    commands = ["set -euo pipefail", f"cd {shlex.quote(remote)}",
                'venv="$HOME/.local/share/spark-serve/recipes/nemotron-super-nvfp4/download-env"']
    if not args.skip_download:
        commands += ['python3 -m venv "$venv"',
                     '"$venv/bin/pip" install --disable-pip-version-check huggingface_hub==1.31.0',
                     'HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_XET_NUM_CONCURRENT_RANGE_GETS=4 "$venv/bin/python" -u -c ' + shlex.quote(download_code)]
    verifier = ("python3" if args.skip_download else '"$venv/bin/python"') + " verify.py " + shlex.quote(snapshot) + " --sha256"
    if not args.skip_download:
        verifier += " --repair-cache " + shlex.quote(str(Path(cache) / "hub"))
    commands += [verifier,
                 "docker build --tag " + shlex.quote(model["image"]) + " .",
                 "docker image inspect --format '{{.Id}}' " + shlex.quote(model["image"]),
                 "printf '%s\\n' " + shlex.quote("Prepared Nemotron on " + cluster["head"] + ". Start it with ./spark-serve up nemotron-super --node " + args.node + ".")]
    print(f"Preparing NVIDIA Nemotron on {cluster['head']}; checkpoint {source['revision']}", flush=True)
    subprocess.run([*ssh, "bash -s"], input="\n".join(commands) + "\n", text=True, check=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as exc:
        sys.exit(f"Nemotron preparation failed: {exc}")
