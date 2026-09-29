#!/usr/bin/env python3
"""CPU-only model metadata, native MTP remapping and patched-module checks.

Run after checkpoint authentication, with its snapshot mounted read-only and no
GPU or network access. This imports the real serving modules and native config
classes but never instantiates a model, allocates weights or evaluates kernels.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys


def check(snapshot: Path) -> dict:
    if snapshot.name != "fc694b54fb0174e0913e6adf86691ef85a4ead47":
        raise ValueError("Expected the pinned NVIDIA snapshot fc694b54")
    version = importlib.metadata.version("vllm")
    if version != "0.30.0":
        raise ValueError(f"Expected vLLM 0.30.0, found {version}")
    raw = json.loads((snapshot / "config.json").read_text())
    if raw.get("model_type") != "qwen4_exp":
        raise ValueError("Unexpected model type")
    quant_data = raw.get("quantization_config")
    if quant_data is None:
        quant_data = json.loads((snapshot / "hf_quant_config.json").read_text())

    from vllm.transformers_utils.config import get_config
    config = get_config(str(snapshot), trust_remote_code=False)
    text_config = config.get_text_config()
    if text_config.max_position_embeddings != 262144:
        raise ValueError("Unexpected native context length")
    if text_config.num_hidden_layers != 48:
        raise ValueError("Unexpected MTP starting layer")

    # Match native registry initialization: model imports qsa, which refers back
    # to model. Importing qsa first creates an artificial circular-import error.
    modules = ["vllm.models.qwen4_exp.nvidia." + name for name in
               ("model", "qsa", "ops.qsa", "ngram_embedding", "mtp")]
    imported = {name: importlib.import_module(name) for name in modules}
    mtp = imported[modules[-1]]
    from vllm.model_executor.layers.quantization import modelopt
    quant = modelopt.ModelOptMixedPrecisionConfig.from_config(quant_data)
    if not quant.quantized_layers:
        raise ValueError("Native ModelOpt config lost per-layer quantization")
    # The pinned checkpoint has 48 NVFP4 target expert blocks, one block-FP8
    # MTP expert block, and exactly one FP8 PLE table. FP8 here is existing
    # checkpoint storage, not an additional serving precision conversion.
    ple_prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    expected_algorithms = {
        f"model.language_model.layers.{layer}.mlp.experts": "NVFP4"
        for layer in range(48)
    }
    expected_algorithms.update({"mtp.layers.0.mlp.experts": "FP8_PB_WO",
                                ple_prefix: "FP8"})
    actual_algorithms = {prefix: str(item["quant_algo"]).upper()
                         for prefix, item in quant.quantized_layers.items()}
    mismatches = {
        prefix: {"expected": expected_algorithms.get(prefix),
                 "actual": actual_algorithms.get(prefix)}
        for prefix in sorted(set(expected_algorithms) | set(actual_algorithms))
        if expected_algorithms.get(prefix) != actual_algorithms.get(prefix)
    }
    if mismatches:
        raise ValueError("Unexpected per-layer quantization: " + json.dumps(mismatches))
    ple_group = quant_data.get("config_groups", {}).get("group_2", {})
    if (ple_group.get("targets") != [ple_prefix]
            or ple_group.get("weights") != {"dynamic": False, "num_bits": 8, "type": "float"}
            or ple_group.get("input_activations") is not None):
        raise ValueError("Unexpected pinned PLE quantization group: " + json.dumps(ple_group))
    declared = set(actual_algorithms.values())
    for prefix, item in quant.quantized_layers.items():
        expected = str(item["quant_algo"]).upper()
        if quant._resolve_quant_algo(prefix) != expected:
            raise ValueError(f"Native quantization lookup failed: {prefix}")
    if "FP8_PB_WO" not in modelopt._BLOCK_FP8_MOE_ALGOS:
        raise ValueError("Native MoE dispatcher lacks NVIDIA block-FP8")
    if list(quant.fp8_block_config.weight_block_size) != [128, 128]:
        raise ValueError("Unexpected native FP8 expert scale geometry")

    # v0.30 remaps quantized_layers in the draft model's configuration; the old
    # custom ModelOpt prefix bridge is neither needed nor copied into this image.
    draft = copy.copy(quant)
    draft.quantized_layers = mtp._remap_quantized_layers(
        quant.quantized_layers, text_config.num_hidden_layers)
    expected_prefix = "mtp.layers.48.mlp.experts"
    if draft._resolve_quant_algo(expected_prefix) != "FP8_PB_WO":
        raise ValueError("Native local-to-global MTP quantization remap failed")
    if quant._resolve_quant_algo("mtp.layers.0.mlp.experts") != "FP8_PB_WO":
        raise ValueError("MTP remapping mutated the target configuration")

    from vllm.config import SpeculativeConfig
    fields = {field.name for field in dataclasses.fields(SpeculativeConfig)}
    required = {"disable_eagle_block_drop", "index_share_for_mtp_iteration",
                "use_local_argmax_reduction"}
    if not required <= fields:
        raise ValueError(f"Missing native speculative fields: {required - fields}")
    # The native schema and template remain authoritative; no replacement chat
    # template or precision override is installed by this compatibility check.
    vocab_file = Path(__file__).with_name("upstream") / "draft_vocab_en_code_47k.txt"
    draft_ids = [int(value) for value in vocab_file.read_text().splitlines() if value.strip()]
    if max(draft_ids) >= text_config.vocab_size:
        raise ValueError("Pinned draft vocabulary exceeds the checkpoint vocabulary")
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU compatibility check unexpectedly initialized CUDA")
    # v0.30 CUDA uses the stable-libtorch extension. An unspecified CPU-only
    # platform may warn about legacy vllm._C, which is not the CUDA module.
    extension = importlib.util.find_spec("vllm._C_stable_libtorch")
    if extension is None or extension.origin is None or not Path(extension.origin).is_file():
        raise ValueError("Required v0.30 CUDA stable-libtorch extension is missing")
    return {"vllm_version": version, "snapshot": snapshot.name,
            "configuration": type(config).__name__, "native_context": 262144,
            "imported_modules": modules, "quantization_algorithms": sorted(declared),
            "quantized_layers": len(quant.quantized_layers),
            "fp8_ple_prefix": ple_prefix, "fp8_ple_weights_only": True,
            "mtp_expert_prefix": expected_prefix, "mtp_quant_algo": "FP8_PB_WO",
            "fp8_weight_block_size": [128, 128], "draft_vocabulary_ids": len(draft_ids),
            "native_speculative_fields": sorted(required), "cuda_initialized": False,
            "cuda_extension": extension.origin}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(check(args.snapshot), indent=2), flush=True)
    except Exception as exc:
        print(f"Qwen v0.30 CPU compatibility failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
