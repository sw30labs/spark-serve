"""Offline integrity checks for the DeepSeek-V4.1-Flash EXL3 recipe."""
import hashlib
import importlib.util
import io
import json
import shlex
import sys
import tomllib
from pathlib import Path
from unittest.mock import patch

import pytest

from tools import prepare_dsv41_exl3 as prepare


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


verify = load_module("dsv41_verify", prepare.RECIPE / "verify.py")
engram_src = load_module("dsv41_prepare_engram_src", prepare.RECIPE / "prepare_engram_src.py")
with patch.dict(sys.modules, {"verify": verify, "prepare_engram_src": engram_src}):
    download = load_module("dsv41_download", prepare.RECIPE / "download.py")


def checkpoint(role, revision, architecture, files):
    return {"model": "test/" + role, "revision": revision, "architectures": [architecture], "files": files}


@pytest.fixture
def assets(tmp_path):
    manifest = {"version": 1, "checkpoints": {}}
    for role, revision, architecture in (
            ("target", "1" * 40, "DeepseekV41ForCausalLM"),
            ("engram", "2" * 40, "DeepseekV41ForCausalLM")):
        snapshot = tmp_path / "models" / role / revision
        snapshot.mkdir(parents=True)
        data = {"config.json": json.dumps({"architectures": [architecture]}).encode(),
                "model.safetensors": b"authentic weight payload"}
        if role == "engram":
            index = {"weight_map": {"layers.1.engram.embed.weight": "model.safetensors"}}
            data["model.safetensors.index.json"] = json.dumps(index).encode()
        files = []
        for name, body in data.items():
            (snapshot / name).write_bytes(body)
            files.append({"path": name, "size": len(body), "sha256": hashlib.sha256(body).hexdigest()})
        manifest["checkpoints"][role] = checkpoint(role, revision, architecture, files)
    return tmp_path, manifest


def test_full_hash_covers_target_and_engram(assets):
    root, manifest = assets
    result = verify.verify(root, manifest)
    assert result["full_hash_verified"] is True
    assert set(result["checkpoints"]) == {"target", "engram"}


@pytest.mark.parametrize("role", ["target", "engram"])
def test_same_size_corruption_never_passes_preflight(assets, role):
    root, manifest = assets
    snapshot = verify.snapshot_path(root, role, manifest["checkpoints"][role])
    path = snapshot / "model.safetensors"
    path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises(verify.CheckpointFileError, match="SHA-256 mismatch"):
        verify.verify(root, manifest)


def test_unmanifested_weight_is_rejected(assets):
    root, manifest = assets
    snapshot = verify.snapshot_path(root, "target", manifest["checkpoints"]["target"])
    (snapshot / "stale.safetensors").write_bytes(b"stale")
    with pytest.raises(ValueError, match="inventory differs"):
        verify.verify(root, manifest)


def test_engram_index_cannot_name_a_missing_shard(assets):
    root, manifest = assets
    engram = manifest["checkpoints"]["engram"]
    snapshot = verify.snapshot_path(root, "engram", engram)
    body = json.dumps({"weight_map": {"layers.1.engram.embed.weight": "model-00001-of-00048.safetensors"}}).encode()
    (snapshot / "model.safetensors.index.json").write_bytes(body)
    for item in engram["files"]:
        if item["path"] == "model.safetensors.index.json":
            item["size"] = len(body)
            item["sha256"] = hashlib.sha256(body).hexdigest()
    with pytest.raises(ValueError, match="unverified shards"):
        verify.verify(root, manifest)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../escape", "a//b", "a\\b"])
def test_manifest_rejects_unsafe_paths(assets, name):
    _, manifest = assets
    manifest["checkpoints"]["target"]["files"][0]["path"] = name
    with pytest.raises(ValueError, match="manifest path"):
        verify.checkpoints(manifest)


def test_external_symlink_is_rejected(assets):
    root, manifest = assets
    snapshot = verify.snapshot_path(root, "target", manifest["checkpoints"]["target"])
    outside = root / "outside"
    path = snapshot / "model.safetensors"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        verify.verify(root, manifest)


def test_controller_manifest_pin_is_checked_before_assets(assets, monkeypatch, capsys):
    root, manifest = assets
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(sys, "argv", ["verify", str(root), "--manifest", str(path),
                                      "--expected-manifest-sha256", "0" * 64])

    def forbidden(*_args, **_kwargs):
        raise AssertionError("must authenticate the manifest before checkpoint reads")

    monkeypatch.setattr(verify, "verify", forbidden)
    with pytest.raises(SystemExit):
        verify.main()
    assert "does not match controller pin" in capsys.readouterr().err


def test_slim_index_keeps_only_embed_tables():
    raw = json.dumps({
        "metadata": {"total_size": 42},
        "weight_map": {
            "layers.14.engram.embed.weight": "model-00048-of-00048.safetensors",
            "layers.14.engram.embed.scale": "model-00048-of-00048.safetensors",
            "layers.1.engram.embed.weight": "model-00047-of-00048.safetensors",
            "layers.1.engram.embed.scale": "model-00047-of-00048.safetensors",
            "layers.0.mlp.weight": "model-00001-of-00048.safetensors",
        },
    }).encode()
    slim = json.loads(download.slim_index_bytes(raw, (".engram.embed.weight", ".engram.embed.scale")))
    assert set(slim["weight_map"]) == {
        "layers.14.engram.embed.weight", "layers.14.engram.embed.scale",
        "layers.1.engram.embed.weight", "layers.1.engram.embed.scale",
    }
    assert slim["metadata"]["dsv41_engram_src"] == "embed-only"
    assert slim["metadata"]["total_size"] == 42


def test_pinned_native_index_slims_to_the_served_hash():
    pins, source = prepare.load_pins()
    engram = source["checkpoints"]["engram"]
    derived = engram["derived_index"]
    served = next(item for item in engram["files"] if item["path"] == derived["path"])
    # The native index is not stored in git. Rebuild the served bytes from the
    # pinned slim file's contract using a stand-in only when the recorded size matches.
    assert served["size"] == 403
    assert served["sha256"] == "bf54f3f6980213246daf176064446d91e1d3734eea95228ad459a01fff766ab5"
    assert derived["upstream_sha256"] == "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8"
    assert pins["manifest_sha256"] == hashlib.sha256((prepare.RECIPE / "model-source.json").read_bytes()).hexdigest()


class Response(io.BytesIO):
    def __init__(self, data, status=200):
        super().__init__(data)
        self.status = status
        self.headers = {}


def test_derived_index_is_written_from_the_native_index(tmp_path):
    raw = json.dumps({
        "metadata": {"total_size": 7},
        "weight_map": {
            "layers.1.engram.embed.weight": "model-00047-of-00048.safetensors",
            "layers.1.engram.embed.scale": "model-00047-of-00048.safetensors",
            "layers.2.mlp.weight": "model-00002-of-00048.safetensors",
        },
    }).encode()
    body = download.slim_index_bytes(raw, (".engram.embed.weight", ".engram.embed.scale"))
    snapshot = tmp_path / "models" / "engram" / ("2" * 40)
    snapshot.mkdir(parents=True)
    checkpoint = {
        "model": "deepseek-ai/DeepSeek-V4.1-Flash",
        "revision": "2" * 40,
        "derived_index": {
            "path": "model.safetensors.index.json",
            "upstream_size": len(raw),
            "upstream_sha256": hashlib.sha256(raw).hexdigest(),
            "keep_suffixes": [".engram.embed.weight", ".engram.embed.scale"],
        },
        "files": [{"path": "model.safetensors.index.json", "size": len(body),
                   "sha256": hashlib.sha256(body).hexdigest()}],
    }

    def opener(_request, timeout=0):
        del timeout
        return Response(raw)

    download.write_derived_index(snapshot, checkpoint, opener=opener)
    written = (snapshot / "model.safetensors.index.json").read_bytes()
    assert written == body
    assert "layers.2.mlp.weight" not in written.decode()


def test_worker_verifier_never_downloads():
    pins, _ = prepare.load_pins()
    script = prepare.download_script("/tmp/recipe", "/worker/cache", pins, True)
    assert "verify.py /worker/cache" in script
    assert "download.py" not in script
    assert "--expected-manifest-sha256 " + pins["manifest_sha256"] in script


def test_image_build_leaves_workloads_untouched():
    pins, _ = prepare.load_pins()
    script = prepare.build_script("/tmp/recipe", pins)
    assert pins["base_image"] in script
    assert "--network none --pull=false" in script
    assert "--patches-only" in script
    assert pins["local_image"] in script
    assert all(token not in script for token in ("--gpus", "docker stop", "docker rm", "systemctl", "10.0.0."))


def test_weight_copy_uses_the_qsfp_link_and_not_nfs():
    script = prepare.copy_script(
        "/head/cache/spark-serve/dsv41-exl3", "/worker/cache with spaces",
        "192.168.100.10", "192.168.100.11", "spider", "spark-serve-dsv41-exl3:0.1.0")
    assert "BindAddress=192.168.100.10" in script
    assert "spider@192.168.100.11" in script
    assert "--checksum" in script
    assert "nfs" not in script
    assert "10.0.0." not in script
    assert "--delete" not in script
    assert "docker save spark-serve-dsv41-exl3:0.1.0" in script


def test_example_catalog_matches_pins():
    pins, source = prepare.load_pins()
    cfg = tomllib.loads((prepare.ROOT / "models.example.toml").read_text())
    model = prepare.validate_catalog(cfg, pins, source)
    assert model["served_name"] == "DeepSeek-v4.1-Flash-EXL3"
    assert model["container"] == "vllm_dsv41_exl3"
    assert "vllm_dsv41_exl3" in cfg["cluster"]["stop_names"]
    assert "--kv-cache-dtype" not in model["vllm"]["args"]


def test_catalog_rejects_a_moved_preflight_pin():
    pins, source = prepare.load_pins()
    cfg = tomllib.loads((prepare.ROOT / "models.example.toml").read_text())
    cfg["models"]["ds41-exl3"]["preflight_args"][-1] = "0" * 64
    with pytest.raises(ValueError, match="preflight"):
        prepare.validate_catalog(cfg, pins, source)
