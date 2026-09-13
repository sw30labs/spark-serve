"""Regression coverage for switching providers with different server limits."""
import importlib.machinery
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import yaml

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("spark_serve_context_test", str(ROOT / "spark-serve"))
spec = importlib.util.spec_from_loader(loader.name, loader)
cli = importlib.util.module_from_spec(spec)
loader.exec_module(cli)


class HermesContextTest(unittest.TestCase):
    def test_server_limit_stays_per_model_and_existing_providers_survive(self):
        original = {
            "model": {"default": "old", "provider": "inferencer", "context_length": 1048576, "max_tokens": 32768},
            "providers": {
                "inferencer": {"base_url": "http://127.0.0.1:54323/v1", "models": {"deepseek-v41": {"context_length": 1048576}}},
                "spark": {"models": {"deepseek-text": {"context_length": 1048576}}},
            },
        }
        cfg = {"cluster": {"lan_url": "http://sparkone.local:8000"}}
        model = {"served_name": "deepseek-vision", "hermes_provider": "spark", "max_model_len": 131072}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(yaml.safe_dump(original))
            with patch.object(cli, "HERMES_CONFIG", path):
                message, ok = cli._hermes_patch(cfg, "ds4-vision", model, False)
            self.assertTrue(ok, message)
            actual = yaml.safe_load(path.read_text())
        self.assertNotIn("context_length", actual["model"])
        self.assertEqual(actual["model"]["provider"], "spark")
        self.assertEqual(actual["model"]["max_tokens"], 32768)
        self.assertEqual(actual["providers"]["inferencer"], original["providers"]["inferencer"])
        self.assertEqual(actual["providers"]["spark"]["models"]["deepseek-text"]["context_length"], 1048576)
        self.assertEqual(actual["providers"]["spark"]["models"]["deepseek-vision"]["context_length"], 131072)
        self.assertEqual(actual["providers"]["spark"]["base_url"], "http://sparkone.local:8000/v1")

if __name__ == "__main__":
    unittest.main()
