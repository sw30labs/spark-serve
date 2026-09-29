"""Offline authentication, cache reuse and independent-node preparation checks."""
import hashlib
import importlib.util
import json
import sys

import pytest

from tools import prepare_qwen38_v030 as prepare

spec = importlib.util.spec_from_file_location("qwen_v030_verify", prepare.RECIPE / "verify.py")
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


@pytest.fixture
def assets(tmp_path):
    root = tmp_path / "models--test--model" / "snapshots" / ("1" * 40)
    root.mkdir(parents=True)
    data = {"config.json": b'{"architectures":["TestModel"]}',
            "model.safetensors.index.json": b'{"weight_map":{"weight":"model.safetensors"}}',
            "model.safetensors": b"authentic weight payload",
            "tokenizer_config.json": b"{}", "preprocessor_config.json": b"{}", "hf_quant_config.json": b"{}"}
    manifest = {"model": "test/model", "revision": root.name, "files": []}
    for name, body in data.items():
        (root / name).write_bytes(body)
        item = {"path": name, "size": len(body)}
        if name.endswith(".safetensors"):
            item["lfs"] = {"sha256": hashlib.sha256(body).hexdigest()}
        else:
            item["git_blob_sha1"] = hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()
        manifest["files"].append(item)
    return root, manifest


def test_full_hash_required_and_detects_same_size_corruption(assets):
    root, manifest = assets
    assert verify.verify(root, manifest)["sha256_verified"]
    with pytest.raises(ValueError, match="requires full"):
        verify.verify(root, manifest, full_hash=False)
    path = root / "model.safetensors"
    path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises(verify.CheckpointFileError, match="SHA-256 mismatch"):
        verify.verify(root, manifest)


def test_normal_hf_blob_symlink_can_reuse_cached_weights(assets):
    root, manifest = assets
    path = root / "model.safetensors"
    blob = root.parent.parent / "blobs" / "weight-digest"
    blob.parent.mkdir()
    path.rename(blob)
    path.symlink_to("../../blobs/weight-digest")
    assert verify.verify(root, manifest)["sha256_verified"]


def test_external_symlink_even_with_authentic_bytes_is_rejected(assets, tmp_path):
    root, manifest = assets
    path = root / "model.safetensors"
    outside = tmp_path / "outside"
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        verify.verify(root, manifest)


def test_stale_unmanifested_weight_is_rejected(assets):
    root, manifest = assets
    (root / "old.safetensors").write_bytes(b"stale")
    with pytest.raises(ValueError, match="inventory differs"):
        verify.verify(root, manifest)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../escape", "a//b", "a\\b"])
def test_manifest_rejects_unsafe_paths(assets, name):
    _, manifest = assets
    manifest["files"][0]["path"] = name
    with pytest.raises(ValueError, match="manifest path"):
        verify.manifest_files(manifest)


def test_manifest_pin_precedes_asset_reads(assets, monkeypatch, capsys):
    root, manifest = assets
    path = root.parent / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(sys, "argv", ["verify", str(root), "--manifest", str(path),
                                    "--expected-manifest-sha256", "0" * 64])
    def forbidden(*args):
        raise AssertionError("must authenticate manifest before reading assets")
    monkeypatch.setattr(verify, "verify", forbidden)
    with pytest.raises(SystemExit):
        verify.main()
    assert "differs from the controller pin" in capsys.readouterr().err


def test_pinned_manifest_reuses_existing_nvidia_bytes():
    pins, source = prepare.load_pins()
    assert (prepare.RECIPE / "model-source.json").read_bytes() == (prepare.ROOT / "recipes/qwen38-nvfp4/model-source.json").read_bytes()
    assert len(verify.manifest_files(source)) == 25
    assert pins["upstream_revision"] == prepare.UPSTREAM_REVISION
    assert source["revision"] == prepare.CHECKPOINT_REVISION


def catalog():
    pins, source = prepare.load_pins()
    model = {"recipe": prepare.MODEL_KEY, "image": pins["local_image"], "nnodes": 1,
             "tensor_parallel": 1, "hf_id": source["model"], "hf_mount": "/cache/huggingface",
             "serve_path": prepare.snapshot_path("/cache/huggingface", source)}
    model["preflight_args"] = [prepare.RUNTIME_ROOT + "/verify.py", model["serve_path"],
                              "--expected-manifest-sha256", pins["manifest_sha256"]]
    cfg = {"cluster": {"head": "sparkone", "worker": "sparktwo", "hf_cache_host": "/head/cache",
                       "worker_hf_cache_host": "/worker/cache", "lan_url": "http://one:8000",
                       "worker_lan_url": "http://two:8000", "port": 8000},
           "models": {prepare.MODEL_KEY: model}}
    return cfg, pins, source


def test_worker_preparation_uses_only_worker_host_and_cache():
    cfg, pins, source = catalog()
    selected = prepare.launch_config(cfg, prepare.MODEL_KEY, "worker")
    prepare.validate_catalog(selected, pins, source)
    assert prepare.ssh_command(selected)[-1] == "sparktwo"
    assert selected["cluster"]["hf_cache_host"] == "/worker/cache"
    assert cfg["cluster"]["head"] == "sparkone"
    script = prepare.checkpoint_script("/tmp/recipe", selected["cluster"]["hf_cache_host"], pins, source, True)
    assert "/worker/cache/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/" + prepare.CHECKPOINT_REVISION in script
    assert "download" not in script


def test_good_cache_branch_never_needs_hub_dependencies():
    _, pins, source = catalog()
    script = prepare.checkpoint_script("/tmp/recipe", "/head/cache", pins, source, False)
    before_fallback = script.split("\nelse\n", 1)[0]
    assert "verify.py" in before_fallback
    assert "bin/pip" not in before_fallback and "huggingface_hub" not in before_fallback
    assert prepare.CHECKPOINT_REVISION in script.split("\nelse\n", 1)[1]


def test_cpu_image_checks_are_offline_and_gpu_invisible():
    cfg, pins, source = catalog()
    build = prepare.build_script("/tmp/recipe", pins)
    compat = prepare.compatibility_script("/head/cache", cfg["models"][prepare.MODEL_KEY], pins)
    assert "--network none --pull=false" in build
    assert "--verify-runtime" in build
    assert ":/cache/huggingface:ro" in compat
    for script in (build, compat):
        assert "NVIDIA_VISIBLE_DEVICES=void" in script and "CUDA_VISIBLE_DEVICES=" in script
        assert all(word not in script for word in ("--gpus", "docker stop", "docker rm", "systemctl", "vllm serve"))


def test_catalog_requires_exact_checkpoint_and_controller_pin():
    cfg, pins, source = catalog()
    prepare.validate_catalog(cfg, pins, source)
    cfg["models"][prepare.MODEL_KEY]["preflight_args"][-1] = "0" * 64
    with pytest.raises(ValueError, match="preflight"):
        prepare.validate_catalog(cfg, pins, source)


def test_snapshot_revision_mismatch_is_rejected(assets):
    root, manifest = assets
    manifest["revision"] = "2" * 40
    with pytest.raises(ValueError, match="expected pinned snapshot"):
        verify.verify(root, manifest)
