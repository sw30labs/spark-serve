#!/usr/bin/env python3
"""Read the pinned model with TensorFold's CPU metadata parser; never load CUDA."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from verify_runtime import main as verify_runtime


def verify_model(model: Path, serve_argv: list[str] | None = None) -> None:
    from tensorfold import families

    config = families.read_config(model)
    family = families.detect(model)
    if family.model_type != "qwen4_exp" or families.quant_method(config) != "mlx":
        raise ValueError("TensorFold requires the pinned Qwen3.8 MLX affine checkpoint")
    family.package.check(model)
    if not family.package.has_mtp(model):
        raise ValueError("TensorFold checkpoint must retain its MTP draft head")
    text = config.get("text_config", {})
    if text.get("max_position_embeddings") != 262144 or not config.get("vision_config"):
        raise ValueError("TensorFold checkpoint context or vision metadata differs")
    if "cuda" not in families.backends_of(family) or "mlx" not in families.readable_quants(family, "cuda"):
        raise ValueError("TensorFold has no CUDA reader for this checkpoint")
    index = json.loads((model / "model.safetensors.index.json").read_text())
    names = index.get("weight_map", {})
    if not any(key.startswith("vision_tower.") for key in names) or not any("ple_embedding." in key for key in names):
        raise ValueError("TensorFold checkpoint needs both vision and SSD n-gram tensors")
    if serve_argv is not None:
        from tensorfold.cli import build_parser
        from tensorfold.serve_options import check

        if not isinstance(serve_argv, list) or not all(isinstance(arg, str) for arg in serve_argv):
            raise ValueError("TensorFold serving arguments must be a string list")
        args = build_parser().parse_args(["serve", *serve_argv])
        if args.model != str(model) or args.backend != "cuda":
            raise ValueError("TensorFold parser checkpoint or backend differs from preflight")
        check(args, family, "cuda", model)
    print("Verified CPU checkpoint metadata: Qwen3.8 MLX 4-bit/group-32, MTP, vision and 262144 context")


def main() -> None:
    verify_runtime()
    verify_model(Path(sys.argv[1]), json.loads(sys.argv[2]) if len(sys.argv) > 2 else None)


if __name__ == "__main__":
    main()
