#!/usr/bin/env python3
"""Check pinned GLM metadata and actual rank argv with the CPU parser; no CUDA load."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from verify_runtime import main as verify_runtime


def verify_model(model: Path, draft: Path, serve_argv: list[str]) -> None:
    from tensorfold import families
    from tensorfold.cli import build_parser
    from tensorfold.serve_options import check

    config = families.read_config(model)
    family = families.detect(model)
    if family.model_type != "glm5_next" or families.quant_method(config) != "exl3":
        raise ValueError("TensorFold requires the pinned GLM EXL3 routed-expert checkpoint")
    family.package.check(model)
    if not family.package.has_mtp(model):
        raise ValueError("GLM checkpoint must retain its MTP draft head")
    if config.get("text_config", {}).get("max_position_embeddings") != 1048576 or not config.get("vision_config"):
        raise ValueError("GLM checkpoint context or vision metadata differs")
    if "cuda" not in families.backends_of(family) or "exl3" not in families.readable_quants(family, "cuda"):
        raise ValueError("TensorFold has no CUDA EXL3 reader for this checkpoint")
    draft_config = json.loads((draft / "config.json").read_text())
    if (draft_config.get("architectures") != ["DFlash2DraftModel"]
            or draft_config.get("hidden_size") != 4096
            or draft_config.get("dflash_config", {}).get("target_layer_ids") != [5, 14, 24, 33, 42]
            or draft_config.get("vocab_size") != config.get("text_config", {}).get("vocab_size")):
        raise ValueError("DFlash2 metadata differs from the pinned GLM draft checkpoint")
    if not isinstance(serve_argv, list) or not all(isinstance(arg, str) for arg in serve_argv):
        raise ValueError("TensorFold serving arguments must be a string list")
    args = build_parser().parse_args(["serve", *serve_argv])
    if args.model != str(model) or args.backend != "cuda" or args.tp != 2 or args.rank not in (0, 1):
        raise ValueError("TensorFold parser checkpoint or TP2 rank differs from preparation")
    if args.drafter not in (None, "", "none", str(draft)):
        raise ValueError("TensorFold parser draft differs from the pinned DFlash2 checkpoint")
    check(args, family, "cuda", model)
    print(f"Verified CPU GLM EXL3 metadata, DFlash2, vision and rank {args.rank} launch parser")


def main() -> None:
    verify_runtime()
    verify_model(Path(sys.argv[1]), Path(sys.argv[2]), json.loads(sys.argv[3]))


if __name__ == "__main__":
    main()
