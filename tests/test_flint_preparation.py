"""CPU-only checks of the Flint NVFP4 preparation recipe (no weights needed)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

RECIPE = Path(__file__).resolve().parents[1] / "recipes/flint-nvfp4"
spec = importlib.util.spec_from_file_location("flint_verify", RECIPE / "verify.py")
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


class ManifestTest(unittest.TestCase):
    def test_pins_match_manifest(self):
        manifest = json.loads((RECIPE / "model-source.json").read_text())
        pins = json.loads((RECIPE / "source-pins.json").read_text())
        self.assertEqual(manifest["checkpoint"]["revision"], pins["checkpoint"]["revision"])
        self.assertEqual(manifest["config_and_tokenizer"]["revision"],
                         pins["config_and_tokenizer"]["revision"])
        gguf = next(f for f in manifest["checkpoint"]["files"] if f["path"] == "flint-nvfp4.gguf")
        self.assertEqual(gguf["lfs"]["sha256"], pins["checkpoint"]["gguf_sha256"])
        paths = {f["path"] for f in manifest["config_and_tokenizer"]["files"]}
        self.assertTrue({"config.json", "tokenizer.json", "chat_template.jinja"} <= paths)
        self.assertFalse(any(p.endswith(".safetensors") for p in paths))


class VerifyTest(unittest.TestCase):
    def snapshot(self, base: Path, rev: str, files: dict[str, bytes], lfs: set[str]):
        root = base / rev
        root.mkdir()
        entries = []
        for name, data in files.items():
            (root / name).write_bytes(data)
            entry = {"path": name, "size": len(data), "lfs": None}
            if name in lfs:
                entry["lfs"] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
            else:
                entry["git_blob_sha1"] = blob_sha1(data)
            entries.append(entry)
        return root, {"model": "x", "revision": rev, "files": entries}

    def test_accepts_complete_snapshot_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, section = self.snapshot(Path(tmp), "abc", {"config.json": b"{}", "w.gguf": b"weights"}, {"w.gguf"})
            self.assertEqual(verify.verify_snapshot(root, section, full=True), 9)
            (root / "w.gguf").write_bytes(b"weightz")
            verify.verify_snapshot(root, section, full=False)  # same length passes without --full
            with self.assertRaises(ValueError):
                verify.verify_snapshot(root, section, full=True)
            (root / "config.json").write_bytes(b"[]")
            with self.assertRaises(ValueError):
                verify.verify_snapshot(root, section, full=False)

    def test_rejects_wrong_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, section = self.snapshot(Path(tmp), "abc", {"a": b"1"}, set())
            section["revision"] = "def"
            with self.assertRaises(ValueError):
                verify.verify_snapshot(root, section, full=False)


if __name__ == "__main__":
    unittest.main()
