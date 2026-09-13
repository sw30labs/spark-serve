"""Exercise the actual ModelOpt resolver without importing CUDA dependencies.

The small metadata fixture reproduces NVIDIA checkpoint revision
fc694b54fb0174e0913e6adf86691ef85a4ead47's MTP entry. The compatibility issue
and correction are documented in the recipe's issue #2 and PR #1:
https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark/pull/1
Only CUDA-facing layer/config constructors are stubbed; resolution and dispatch
execute the four real methods from the shipped overlay.
"""
from __future__ import annotations

import ast
from pathlib import Path
import re
import sys
import types
import unittest
from unittest.mock import patch


OVERLAY = (Path(__file__).resolve().parents[1] / "recipes/qwen38-nvfp4"
           / "single-spark-vllm-tp1/patch/upstream-overlays/modelopt.py")
MTP_PREFIX = "mtp.layers.48.mlp.experts"
CURRENT_METADATA = {
    "mtp.layers.0.mlp.experts": {"quant_algo": "FP8_PB_WO", "group_size": 128},
}


class Attention:
    pass


class LinearBase:
    pass


class ParallelLMHead:
    pass


class RoutedExperts:
    pass


class Fp8Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Fp8MoEMethod:
    def __init__(self, config, layer):
        self.config = config
        self.layer = layer


def load_dispatch():
    tree = ast.parse(OVERLAY.read_text())
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                    and n.name == "ModelOptMixedPrecisionConfig")
    selected = {"_resolve_quant_algo", "_quantized_layer_group_size",
                "_quantized_layer_prefix_candidates", "get_quant_method"}
    methods = [n for n in original.body if isinstance(n, ast.FunctionDef)
               and n.name in selected]
    if len(methods) != len(selected):
        raise AssertionError("The shipped ModelOpt dispatch API changed")
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        ast.ClassDef(name="Dispatch", bases=[], keywords=[], body=methods, decorator_list=[]),
    ], type_ignores=[])
    namespace = dict(re=re, Attention=Attention, LinearBase=LinearBase,
                     ParallelLMHead=ParallelLMHead, RoutedExperts=RoutedExperts)
    exec(compile(ast.fix_missing_locations(module), str(OVERLAY), "exec"), namespace)
    obj = namespace["Dispatch"]()
    obj.is_layer_excluded = lambda prefix: False
    obj.packed_modules_mapping = {}
    obj.kv_cache_quant_method = None
    return obj


class ModelOptMTPTests(unittest.TestCase):
    def setUp(self):
        self.dispatch = load_dispatch()
        self.layer = RoutedExperts()
        fp8 = types.ModuleType("vllm.model_executor.layers.quantization.fp8")
        fp8.Fp8Config = Fp8Config
        fp8.Fp8MoEMethod = Fp8MoEMethod
        replacement = patch.dict(sys.modules, {fp8.__name__: fp8})
        replacement.start()
        self.addCleanup(replacement.stop)

    def assert_block_fp8(self, result, block_size=128):
        self.assertIsInstance(result, Fp8MoEMethod)
        self.assertIs(result.layer, self.layer)
        self.assertTrue(result.config.is_checkpoint_fp8_serialized)
        self.assertEqual(result.config.activation_scheme, "dynamic")
        self.assertEqual(result.config.weight_block_size, [block_size, block_size])

    def test_current_nvidia_metadata_loads_mtp_as_block_fp8(self):
        self.dispatch.quantized_layers = CURRENT_METADATA
        self.assertEqual(self.dispatch._resolve_quant_algo(MTP_PREFIX), "FP8_PB_WO")
        self.assert_block_fp8(self.dispatch.get_quant_method(self.layer, MTP_PREFIX))

    def test_legacy_metadata_still_loads_mtp_as_block_fp8(self):
        for algo in ("FP8_BLOCK_SCALES", "FP8_BLOCK"):
            with self.subTest(algo=algo):
                self.dispatch.quantized_layers = {
                    "mtp.layers.0.mlp.experts": {"quant_algo": algo, "group_size": 128},
                }
                self.assert_block_fp8(self.dispatch.get_quant_method(self.layer, MTP_PREFIX))

    def test_group_size_comes_from_resolved_draft_local_metadata(self):
        self.dispatch.quantized_layers = {
            "mtp.layers.0.mlp.experts": {"quant_algo": "FP8_PB_WO", "group_size": 64},
        }
        self.assert_block_fp8(self.dispatch.get_quant_method(self.layer, MTP_PREFIX), 64)

    def test_unknown_format_is_not_treated_as_block_fp8(self):
        self.dispatch.quantized_layers = {
            "mtp.layers.0.mlp.experts": {"quant_algo": "UNSUPPORTED_FORMAT", "group_size": 128},
        }
        self.assertIsNone(self.dispatch.get_quant_method(self.layer, MTP_PREFIX))

    def test_excluded_mtp_layer_remains_unquantized(self):
        self.dispatch.quantized_layers = CURRENT_METADATA
        self.dispatch.is_layer_excluded = lambda prefix: prefix == MTP_PREFIX
        self.assertIsNone(self.dispatch.get_quant_method(self.layer, MTP_PREFIX))


if __name__ == "__main__":
    unittest.main()
