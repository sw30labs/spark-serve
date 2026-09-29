#!/usr/bin/env python3
"""Apply a source-hash-pinned local correction; does not load any model weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import py_compile

REVISION = "marlin-gate-up-maxglobal-e4m3-v1"
BASE_IMAGE = "nvcr.io/nim/zai-org/glm-5.3-flash@sha256:0bd2a1f4ffacf4ee61e1f8c1b48071a6713b51bad954ab0ca627c24be84f43f8"
SOURCE_SHA256 = '6ff8f28fbf1d567792375a333d936ca994962e27d9bbc9dcd3489016fce426de'
HELPER_SHA256 = '4357bba3403f0eabc7675d7916e4b5b0ec2c783ba4f0e83acd74e87e2aa37c17'
PATCHED_SHA256 = 'c551e00ea3e1c481b792bc396f58c1308288f81611aa415dc5b78185ef82040b'
SOURCE_RELATIVE = "srt/layers/quantization/modelopt_quant.py"
OLD_BLOCK = '            # Marlin supports only a single shared w1/w3 weight scale, so collapse\n            # the gate/up columns to the gate scale here. Other backends keep the\n            # raw scale and split the halves later (see _compute_gemm1_alphas).\n            if layer.moe_runner_config.is_gated:\n                if layer.w13_weight_scale_2.dim() == 1:\n                    # Some checkpoints store a shared scale for w1/w3.\n                    w13_weight_scale_2 = layer.w13_weight_scale_2\n                else:\n                    if layer.w13_weight_scale_2.shape[1] >= 2 and not torch.allclose(\n                        layer.w13_weight_scale_2[:, 0],\n                        layer.w13_weight_scale_2[:, 1],\n                    ):\n                        logger.warning_once(\n                            "w1_weight_scale_2 must match w3_weight_scale_2. "\n                            "Accuracy may be affected."\n                        )\n\n                    w13_weight_scale_2 = layer.w13_weight_scale_2[:, 0]\n            else:\n                w13_weight_scale_2 = layer.w13_weight_scale_2[:]'
NEW_BLOCK = '            # spark-serve local correction: reconcile separate gate/up global\n            # scales before the single-global Marlin encoding. This explicitly\n            # rounds block scales to E4M3; checkpoint FP4 payload is unchanged.\n            if layer.moe_runner_config.is_gated:\n                reconciled, w13_weight_scale_2, changed = (\n                    _spark_reconcile_nvfp4_gate_up(\n                        layer.w13_weight_scale, layer.w13_weight_scale_2\n                    )\n                )\n                copy_or_rebind_param(layer, "w13_weight_scale", reconciled)\n                if changed:\n                    logger.warning(\n                        "spark-serve NVFP4 reconciliation: %d experts use shared "\n                        "max gate/up global scales with additional E4M3 block-scale "\n                        "rounding; packed checkpoint weights are unchanged.", changed\n                    )\n            else:\n                w13_weight_scale_2 = layer.w13_weight_scale_2[:]'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def patch_source(source, helper):
    if digest(source) != SOURCE_SHA256:
        raise ValueError("Pinned modelopt_quant.py source SHA256 mismatch")
    if digest(helper) != HELPER_SHA256:
        raise ValueError("Pinned reconciliation helper SHA256 mismatch")
    text = source.decode("utf-8")
    helper_text = helper.decode("utf-8")
    marker = "def _compute_gemm1_alphas("
    if text.count(marker) != 1 or text.count(OLD_BLOCK) != 1:
        raise ValueError("Pinned patch locations do not match exactly once")
    function = helper_text[helper_text.index("def _spark_reconcile_nvfp4_gate_up"):].rstrip() + "\n\n\n"
    patched = text.replace(marker, function + marker).replace(OLD_BLOCK, NEW_BLOCK).encode()
    if digest(patched) != PATCHED_SHA256:
        raise ValueError("Patched source SHA256 differs from reviewed output")
    compile(patched, SOURCE_RELATIVE, "exec")
    return patched


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/sgl-workspace/sglang/python/sglang"))
    parser.add_argument("--receipt", type=Path, default=Path("/opt/spark-serve/glm53/patch-receipt.json"))
    args = parser.parse_args(argv)
    helper_path = Path(__file__).with_name("nvfp4_scale_reconcile.py")
    target = args.root / SOURCE_RELATIVE
    patched = patch_source(target.read_bytes(), helper_path.read_bytes())
    target.write_bytes(patched)
    # Deterministic hash-based caches cover every normal Python optimization
    # mode and cannot accidentally load the base image's timestamp-based pyc.
    caches = []
    for optimization in (0, 1, 2):
        cache = Path(py_compile.compile(str(target), doraise=True, optimize=optimization,
                    invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH))
        os.utime(cache, (0, 0))
        caches.append(cache)
    os.utime(target, (0, 0))
    for parent in {target.parent, *(cache.parent for cache in caches)}:
        os.utime(parent, (0, 0))
    receipt = {
        "revision": REVISION, "base_image": BASE_IMAGE,
        "source_relative": SOURCE_RELATIVE,
        "source_sha256": SOURCE_SHA256, "helper_sha256": HELPER_SHA256,
        "patched_sha256": PATCHED_SHA256,
        "checkpoint_files_modified": False,
        "method": "per-expert max global; explicit E4M3 block-scale rerounding before Marlin",
        "limitation": "Additional block-scale quantization; not exact weight preservation or an upstream/NVIDIA fix",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.utime(args.receipt, (0, 0))
    os.utime(args.receipt.parent, (0, 0))
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
