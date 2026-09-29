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
            path.chmod(0o600)
            with patch.object(cli, "HERMES_CONFIG", path):
                message, ok = cli._hermes_patch(cfg, "ds4-vision", model, False)
            self.assertTrue(ok, message)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(list(Path(tmp).iterdir()), [path])
            actual = yaml.safe_load(path.read_text())
        self.assertNotIn("context_length", actual["model"])
        self.assertEqual(actual["model"]["provider"], "spark")
        self.assertEqual(actual["model"]["max_tokens"], 32768)
        self.assertEqual(actual["providers"]["inferencer"], original["providers"]["inferencer"])
        self.assertEqual(actual["providers"]["spark"]["models"]["deepseek-text"]["context_length"], 1048576)
        self.assertEqual(actual["providers"]["spark"]["models"]["deepseek-vision"]["context_length"], 131072)
        self.assertEqual(actual["providers"]["spark"]["base_url"], "http://sparkone.local:8000/v1")

    def test_glm_vision_is_scoped_and_preserves_other_model_settings(self):
        original = {
            "model": {"default": "deepseek-v4-flash", "provider": "spark"},
            "providers": {"spark": {"models": {
                "deepseek-v4-flash": {"context_length": 1048576, "supports_vision": False},
                "glm-5.3-flash": {"context_length": 8192, "custom_setting": "preserve"},
            }}},
        }
        cfg = {"cluster": {"lan_url": "http://sparkone.local:8000"}}
        model = {"served_name": "glm-5.3-flash", "max_model_len": 131072,
                 "hermes_supports_vision": True}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(yaml.safe_dump(original))
            with patch.object(cli, "HERMES_CONFIG", path):
                message, ok = cli._hermes_patch(cfg, "glm53", model, False)
            self.assertTrue(ok, message)
            actual = yaml.safe_load(path.read_text())
        models = actual["providers"]["spark"]["models"]
        self.assertEqual(models["glm-5.3-flash"], {
            "context_length": 131072, "supports_vision": True, "custom_setting": "preserve"})
        self.assertEqual(models["deepseek-v4-flash"], original["providers"]["spark"]["models"]["deepseek-v4-flash"])
        self.assertNotIn("supports_vision", actual["model"])

    def test_same_endpoint_with_wrong_model_does_not_claim_hermes_selection(self):
        cfg = {"cluster": {"head": "sparkone", "worker": "sparktwo",
                           "lan_url": "http://sparkone.local:8000", "port": 8000}}
        selected = {"model": {"provider": "spark", "default": "deepseek-v4-flash"},
                    "providers": {"spark": {"base_url": "http://sparkone.local:8000/v1"}}}
        nodes = [{"node": "head", "served": "glm-5.3-flash", "can_use": True}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(yaml.safe_dump(selected))
            with patch.object(cli, "HERMES_CONFIG", path):
                self.assertIsNone(cli._active_client_node(cfg, nodes))
                selected["model"]["default"] = "glm-5.3-flash"
                path.write_text(yaml.safe_dump(selected))
                self.assertEqual(cli._active_client_node(cfg, nodes), "head")
                nodes[0]["can_use"] = False
                self.assertIsNone(cli._active_client_node(cfg, nodes))

    def test_failed_config_update_preserves_original_bytes(self):
        original = "model: {default: old, provider: spark}\nproviders:\n  spark:\n    models: {glm-5.3-flash: malformed}\n"
        cfg = {"cluster": {"lan_url": "http://sparkone.local:8000"}}
        model = {"served_name": "glm-5.3-flash", "max_model_len": 131072}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(original)
            with patch.object(cli, "HERMES_CONFIG", path):
                _, ok = cli._hermes_patch(cfg, "glm53", model, False)
            self.assertFalse(ok)
            self.assertEqual(path.read_text(), original)

    def test_shared_yaml_alias_does_not_change_the_unselected_model(self):
        original = """model: {default: deepseek-v4-flash, provider: spark}
providers:
  spark:
    models:
      deepseek-v4-flash: &shared
        context_length: 1048576
        supports_vision: false
        custom_setting: keep
      glm-5.3-flash: *shared
"""
        cfg = {"cluster": {"lan_url": "http://sparkone.local:8000"}}
        model = {"served_name": "glm-5.3-flash", "max_model_len": 131072,
                 "hermes_supports_vision": True}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(original)
            with patch.object(cli, "HERMES_CONFIG", path):
                message, ok = cli._hermes_patch(cfg, "glm53", model, False)
            self.assertTrue(ok, message)
            actual = yaml.safe_load(path.read_text())["providers"]["spark"]["models"]
        self.assertEqual(actual["deepseek-v4-flash"], {
            "context_length": 1048576, "supports_vision": False, "custom_setting": "keep"})
        self.assertEqual(actual["glm-5.3-flash"], {
            "context_length": 131072, "supports_vision": True, "custom_setting": "keep"})

if __name__ == "__main__":
    unittest.main()
