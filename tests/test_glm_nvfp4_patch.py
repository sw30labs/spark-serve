"""CPU tests of the exact source-injected loader correction, with no GPU calls."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes/glm53-nvfp4"
CAPTURE = ROOT / "diagnostics/2026-09-19-glm53/quantization-review/exact-image-sources.json"


def load_patch():
    spec = importlib.util.spec_from_file_location("glm_scale_patch", RECIPE / "patch_runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PinnedPatchTests(unittest.TestCase):
    def test_rejects_unreviewed_source_or_helper(self):
        patch = load_patch()
        with self.assertRaisesRegex(ValueError, "source SHA256"):
            patch.patch_source(b"unreviewed", b"unreviewed")
        if CAPTURE.exists():
            source = json.loads(CAPTURE.read_text())[patch.SOURCE_RELATIVE].encode()
            with self.assertRaisesRegex(ValueError, "helper SHA256"):
                patch.patch_source(source, b"unreviewed")

    @unittest.skipUnless(CAPTURE.exists(), "exact image source capture unavailable")
    def test_pinned_transform_compiles_and_rejects_reapplication(self):
        patch = load_patch()
        source = json.loads(CAPTURE.read_text())[patch.SOURCE_RELATIVE].encode()
        helper = (RECIPE / "nvfp4_scale_reconcile.py").read_bytes()
        result = patch.patch_source(source, helper)
        self.assertEqual(patch.digest(result), patch.PATCHED_SHA256)
        self.assertNotIn(b"w1_weight_scale_2 must match w3_weight_scale_2", result)
        with self.assertRaises(ValueError):
            patch.patch_source(result, helper)


@unittest.skipIf(torch is None, "CPU torch required for numerical correction tests")
class ReconcileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Execute the function from the fully patched source when available.
        # No SGLang or CUDA modules are imported.
        patch = load_patch()
        if CAPTURE.exists():
            source = json.loads(CAPTURE.read_text())[patch.SOURCE_RELATIVE].encode()
            code = patch.patch_source(source, (RECIPE / "nvfp4_scale_reconcile.py").read_bytes()).decode()
        else:
            code = (RECIPE / "nvfp4_scale_reconcile.py").read_text()
        tree = ast.parse(code)
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_spark_reconcile_nvfp4_gate_up")
        env = {"torch": torch}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "reviewed_helper", "exec"), env)
        cls.reconcile = staticmethod(env[helper.name])
        cls.tree = tree

    def test_equal_and_zero_pairs_preserve_bytes_and_inputs(self):
        blocks = torch.tensor([[[0., .001953125, .015625, 448.]] * 2] * 3).to(torch.float8_e4m3fn)
        globals_ = torch.tensor([[1., 1.], [0., 0.], [2., 0.]])
        before = blocks.view(torch.uint8).clone()
        result, shared, count = self.reconcile(blocks, globals_)
        self.assertEqual(count, 1)
        self.assertTrue(torch.equal(result[:2].view(torch.uint8), before[:2]))
        self.assertTrue(torch.equal(blocks.view(torch.uint8), before))
        self.assertEqual(result[2, 1].float().count_nonzero().item(), 0)
        self.assertTrue(torch.equal(shared, torch.tensor([1., 0., 2.])))
        self.assertTrue(torch.isfinite(result.float()).all())

    def test_shared_input_and_equal_pairs_are_noop(self):
        blocks = torch.ones(2, 4, 4).to(torch.float8_e4m3fn)
        for global_ in [torch.tensor([1., 2.]), torch.tensor([[1.], [2.]]), torch.tensor([[1., 1.], [2., 2.]])]:
            result, _, count = self.reconcile(blocks, global_)
            self.assertIs(result, blocks)
            self.assertEqual(count, 0)

    def test_fail_closed_for_invalid_values_and_layout(self):
        good = torch.ones(1, 4, 4).to(torch.float8_e4m3fn)
        for globals_ in [torch.tensor([[float("nan"), 1.]]), torch.tensor([[float("inf"), 1.]]),
                         torch.tensor([[-1., 1.]]), torch.ones(1, 3), torch.ones(2, 2),
                         torch.ones(1, 2, dtype=torch.float16)]:
            with self.subTest(globals_=globals_), self.assertRaises(ValueError):
                self.reconcile(good, globals_)
        for blocks in [torch.ones(1, 3, 4).to(torch.float8_e4m3fn), good.float(),
                       torch.full((1, 4, 4), float("nan")).to(torch.float8_e4m3fn),
                       -torch.ones(1, 4, 4).to(torch.float8_e4m3fn).float()]:
            if blocks.shape == good.shape and blocks.dtype == torch.float32 and blocks.min() < 0:
                blocks = blocks.to(torch.float8_e4m3fn)
            with self.assertRaises(ValueError):
                self.reconcile(blocks, torch.ones(1, 2))

    def test_explicit_fp8_rounding_matches_independent_reference(self):
        # Includes subnormals; relative error is NOT uniformly 6.25% near zero.
        values = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn)
        blocks = values.repeat(4).reshape(1, 4, 127)
        globals_ = torch.tensor([[.875, 1.]])
        result, shared, _ = self.reconcile(blocks, globals_)
        expected = blocks.float().reshape(1, 2, 2, 127) * globals_[:, :, None, None]
        expected = expected.to(torch.float8_e4m3fn).reshape_as(blocks)
        self.assertTrue(torch.equal(result.view(torch.uint8), expected.view(torch.uint8)))
        self.assertTrue(torch.isfinite(result.float()).all())
        self.assertEqual(shared.item(), 1.)

    @unittest.skipUnless(CAPTURE.exists(), "exact image source capture unavailable")
    def test_actual_loader_calls_correction_before_marlin(self):
        candidates = [n for n in ast.walk(self.tree) if isinstance(n, ast.FunctionDef)
                      and n.name == "process_weights_after_loading"
                      and any(isinstance(x, ast.Name) and x.id == "_spark_reconcile_nvfp4_gate_up" for x in ast.walk(n))]
        self.assertEqual(len(candidates), 1)
        calls = []
        backend = SimpleNamespace(is_marlin=lambda: True)
        def bind(layer, name, value):
            setattr(layer, name, value)
        def prepare(layer):
            calls.append((layer.w13_weight_scale.clone(), layer.w13_weight_scale_2.clone()))
        env = {"torch": torch, "_spark_reconcile_nvfp4_gate_up": self.reconcile,
               "copy_or_rebind_param": bind, "prepare_moe_nvfp4_layer_for_marlin": prepare,
               "get_moe_runner_backend": lambda: backend,
               "logger": SimpleNamespace(warning=lambda *args: None)}
        exec(compile(ast.Module(body=candidates, type_ignores=[]), "patched_loader", "exec"), env)
        block = torch.full((1, 4, 4), 32.).to(torch.float8_e4m3fn)
        layer = SimpleNamespace(w13_weight_scale=block, w13_weight_scale_2=torch.tensor([[1., 1.5]]),
                                moe_runner_config=SimpleNamespace(is_gated=True))
        env["process_weights_after_loading"](SimpleNamespace(_moe_runner_backend=backend), layer)
        expected, shared, _ = self.reconcile(block, torch.tensor([[1., 1.5]]))
        self.assertEqual(len(calls), 1)
        self.assertTrue(torch.equal(calls[0][0].view(torch.uint8), expected.view(torch.uint8)))
        self.assertTrue(torch.equal(calls[0][1], shared))

    def test_clipped_swiglu_uses_corrected_weights_before_activation(self):
        blocks = torch.full((1, 2, 4), 32.).to(torch.float8_e4m3fn)
        globals_ = torch.tensor([[1., .7]])
        fixed, common, _ = self.reconcile(blocks, globals_)
        # A toy diagonal GEMM using the same packed FP4 values throughout.
        inputs = torch.tensor([[-.3, .1, .3, .6]])
        fp4 = torch.tensor([[1., -1., 2., .5]])
        exact = blocks.float() * globals_[:, :, None] * fp4 * inputs
        corrected = fixed.float() * common[:, None, None] * fp4 * inputs
        broken = blocks.float() * globals_[:, :1, None] * fp4 * inputs
        def clipped(x):
            gate = x[:, 0].clamp(max=10.)
            up = x[:, 1].clamp(-10., 10.)
            return torch.nn.functional.silu(gate) * up
        fixed_error = (clipped(corrected) - clipped(exact)).abs().max().item()
        broken_error = (clipped(broken) - clipped(exact)).abs().max().item()
        self.assertLess(fixed_error, broken_error / 3)
        self.assertTrue(torch.isfinite(clipped(corrected)).all())


if __name__ == "__main__":
    unittest.main()
