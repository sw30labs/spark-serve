#!/usr/bin/env python3
"""Check pinned NVFP4 metadata and the exact rank argv with the native parser; no CUDA load."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from verify_runtime import main as verify_runtime

NATIVE = "/opt/tensorfold/native/bin/tensorfold-native"
PROBE = "/nonexistent/spark-serve-argument-check"


def verify_metadata(model: Path) -> None:
    config = json.loads((model / "config.json").read_text())
    text = config.get("text_config", {})
    quant = config.get("quantization_config", {})
    if (config.get("architectures") != ["Qwen4ExpForConditionalGeneration"] or config.get("model_type") != "qwen4_exp"
            or quant.get("quant_method") != "modelopt" or quant.get("quant_algo") != "MIXED_PRECISION"):
        raise ValueError("TensorFold requires the pinned NVIDIA NVFP4 Qwen3.8 checkpoint")
    if (text.get("max_position_embeddings") != 262144 or text.get("num_experts_per_tok") != 10
            or not config.get("vision_config")):
        raise ValueError("checkpoint context, routing or vision metadata differs")
    names = json.loads((model / "model.safetensors.index.json").read_text()).get("weight_map", {})
    if not any(key.startswith("mtp.") for key in names) or not any(key.startswith("model.visual.") for key in names):
        raise ValueError("TensorFold checkpoint needs its MTP head and vision tower")


def verify_arguments(model: Path, serve_argv: list[str]) -> int:
    if not isinstance(serve_argv, list) or not all(isinstance(arg, str) for arg in serve_argv):
        raise ValueError("TensorFold serving arguments must be a string list")
    if not serve_argv or serve_argv[0] != str(model):
        raise ValueError("TensorFold argv checkpoint differs from the authenticated snapshot")

    def value(flag: str) -> str | None:
        found = [serve_argv[i + 1] for i, arg in enumerate(serve_argv[:-1]) if arg == flag]
        if len(found) > 1:
            raise ValueError(f"TensorFold argv repeats {flag}")
        return found[0] if found else None

    rank = value("--rank")
    if value("--tp") != "2" or rank not in ("0", "1") or not value("--master") or not value("--master-port"):
        raise ValueError("TensorFold argv must name TP2, one rank and the rendezvous")
    # tensorfold-native has no parse-only mode. With a model path that does not
    # exist, accepted arguments end at the model check (exit 1) and rejected
    # ones end in its usage error (exit 2); neither touches CUDA or the network.
    result = subprocess.run([NATIVE, "serve", PROBE, *serve_argv[1:]], capture_output=True, text=True, timeout=60)
    if result.returncode != 1 or "neither a directory" not in result.stderr:
        detail = (result.stderr.strip().splitlines() or ["no output"])[-1]
        raise ValueError(f"tensorfold-native rejects rank {rank} arguments: {detail}")
    return int(rank)


def verify_model(model: Path, serve_argv: list[str]) -> None:
    verify_metadata(model)
    rank = verify_arguments(model, serve_argv)
    print(f"Verified NVFP4 metadata, MTP, vision and native rank {rank} launch parser")


def main() -> None:
    verify_runtime()
    verify_model(Path(sys.argv[1]), json.loads(sys.argv[2]))


if __name__ == "__main__":
    main()
