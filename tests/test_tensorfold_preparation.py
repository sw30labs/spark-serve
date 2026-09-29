"""Offline authenticity, cache placement and preparation workload isolation."""
import hashlib
import importlib.util
import json
import sys
import tomllib

import pytest

from tools import prepare_qwen38_tensorfold as prepare
from spark_serve_controller import ControllerError


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verify = load_module("tensorfold_checkpoint_verify", prepare.RECIPE / "verify.py")
runtime = load_module("tensorfold_runtime_verify", prepare.RECIPE / "runtime/verify_runtime.py")
sys.path.insert(0, str(prepare.RECIPE))
try:
    space = load_module("tensorfold_checkpoint_space", prepare.RECIPE / "check_space.py")
finally:
    sys.path.pop(0)


@pytest.fixture
def assets(tmp_path):
    root = tmp_path / "models--test--model" / "snapshots" / ("1" * 40)
    root.mkdir(parents=True)
    files = {"config.json": b'{"architectures":["TestModel"]}',
             "model.safetensors.index.json": b'{"weight_map":{"weight":"model.safetensors"}}',
             "model.safetensors": b"authentic weight payload",
             "tokenizer_config.json": b"{}", "preprocessor_config.json": b"{}"}
    manifest = {"model": "test/model", "revision": root.name, "files": []}
    for name, body in files.items():
        (root / name).write_bytes(body)
        item = {"path": name, "size": len(body)}
        if name.endswith(".safetensors"):
            item["lfs"] = {"sha256": hashlib.sha256(body).hexdigest()}
        else:
            item["git_blob_sha1"] = hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()
        manifest["files"].append(item)
    return root, manifest


def catalog():
    pins, source = prepare.load_pins()
    cfg = tomllib.loads((prepare.ROOT / "models.example.toml").read_text())
    cfg["cluster"].update(head="sparkone", worker="sparktwo", hf_cache_host="/head/cache",
                          worker_hf_cache_host="/worker/cache")
    return cfg, pins, source


def test_published_snapshot_and_runtime_are_pinned():
    pins, source = prepare.load_pins()
    assert source["model"] == "Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP"
    assert source["revision"] == prepare.CHECKPOINT_REVISION
    files = verify.manifest_files(source)
    assert len(files) == 35
    assert sum(item["size"] for item in files) == 113233047784
    assert pins["base_platform"] == "linux/arm64"
    installed = json.loads((prepare.RECIPE / "runtime/installed-runtime.json").read_text())
    assert installed["tensorfold_revision"] == pins["tensorfold_revision"]
    assert installed["tensorfold_version"] == "0.3.6.3"
    assert len(installed["files"]) == 360
    assert "cuda/server.py" in installed["files"]
    assert "families/qwen4_exp/cuda/engine.py" in installed["files"]


def test_full_checkpoint_hashes_detect_same_size_corruption(assets):
    root, manifest = assets
    assert verify.verify(root, manifest)["sha256_verified"]
    with pytest.raises(ValueError, match="requires full"):
        verify.verify(root, manifest, full_hash=False)
    path = root / "model.safetensors"
    path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises(verify.CheckpointFileError, match="SHA-256 mismatch"):
        verify.verify(root, manifest)


def test_hub_blob_symlinks_are_supported_but_external_links_are_rejected(assets, tmp_path):
    root, manifest = assets
    weight = root / "model.safetensors"
    blob = root.parent.parent / "blobs" / "digest"
    blob.parent.mkdir()
    weight.rename(blob)
    weight.symlink_to("../../blobs/digest")
    assert verify.verify(root, manifest)["sha256_verified"]
    weight.unlink()
    outside = tmp_path / "outside"
    outside.write_bytes(blob.read_bytes())
    weight.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        verify.verify(root, manifest)


def test_stale_unmanifested_shard_is_rejected(assets):
    root, manifest = assets
    (root / "old.safetensors").write_bytes(b"stale")
    with pytest.raises(ValueError, match="inventory differs"):
        verify.verify(root, manifest)


def test_manifest_authentication_happens_before_runtime_or_weight_access(assets, monkeypatch, capsys):
    root, manifest = assets
    path = root.parent / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(sys, "argv", ["verify", str(root), "--manifest", str(path),
                                    "--expected-manifest-sha256", "0" * 64, "--verify-runtime"])
    def forbidden(*args, **kwargs):
        raise AssertionError("must authenticate manifest before runtime or asset access")
    monkeypatch.setattr(verify.subprocess, "run", forbidden)
    monkeypatch.setattr(verify, "verify", forbidden)
    with pytest.raises(SystemExit):
        verify.main()
    assert "differs from the controller pin" in capsys.readouterr().err


def test_installed_runtime_rejects_same_size_edits_and_extra_modules(tmp_path):
    path = tmp_path / "server.py"
    path.write_bytes(b"real code")
    expected = {"server.py": hashlib.sha256(path.read_bytes()).hexdigest()}
    runtime.verify_files(tmp_path, expected)
    path.write_bytes(b"fake code")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        runtime.verify_files(tmp_path, expected)
    path.write_bytes(b"real code")
    (tmp_path / "injected.py").write_bytes(b"extra")
    with pytest.raises(ValueError, match="inventory differs"):
        runtime.verify_files(tmp_path, expected)


def test_runtime_integrity_rejects_external_symlink_even_if_bytes_match(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"real code")
    (package / "server.py").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        runtime.verify_files(package, {"server.py": hashlib.sha256(outside.read_bytes()).hexdigest()})


def test_worker_preparation_uses_worker_host_and_cache_without_changing_head():
    cfg, pins, source = catalog()
    selected = prepare.launch_config(cfg, prepare.MODEL_KEY, "worker")
    prepare.validate_catalog(selected, pins, source)
    assert prepare.ssh_command(selected)[-1] == "sparktwo"
    assert selected["cluster"]["hf_cache_host"] == "/worker/cache"
    assert cfg["cluster"]["head"] == "sparkone"
    script = prepare.checkpoint_script("/tmp/recipe", "/worker/cache", pins, source, True)
    assert prepare.snapshot_path("/worker/cache", source) in script
    assert "download" not in script


def test_good_cache_needs_neither_hub_nor_download_dependencies():
    _, pins, source = catalog()
    script = prepare.checkpoint_script("/tmp/recipe", "/head/cache", pins, source, False)
    cached, fallback = script.split("\nelse\n", 1)
    assert "verify.py" in cached
    assert "bin/pip" not in cached and "huggingface_hub" not in cached
    assert prepare.CHECKPOINT_REVISION in fallback
    assert "token=False" in fallback


def test_preparation_is_gpu_invisible_and_never_touches_running_workloads():
    cfg, pins, source = catalog()
    build = prepare.build_script("/tmp/recipe", pins)
    plan = prepare.validate_tensorfold_plan(cfg, prepare.MODEL_KEY)
    compat = prepare.compatibility_script("/head/cache", cfg["models"][prepare.MODEL_KEY], pins, plan.argv)
    assert "--platform linux/arm64" in build
    assert "--network none --pull=false" in build
    assert "--verify-runtime" in build
    assert ":/cache/huggingface:ro" in compat
    for script in (build, compat):
        assert "NVIDIA_VISIBLE_DEVICES=void" in script and "CUDA_VISIBLE_DEVICES=" in script
        assert all(word not in script for word in ("--gpus", "docker stop", "docker rm", "systemctl", "tensorfold serve"))


@pytest.mark.parametrize("field,value", [
    ("serve_path", "/cache/huggingface/unpinned"),
    ("preflight_args", []), ("wrapper", "vllm"), ("hf_mount", "/different")])
def test_catalog_must_match_the_pinned_recipe(field, value):
    cfg, pins, source = catalog()
    prepare.validate_catalog(cfg, pins, source)
    cfg["models"][prepare.MODEL_KEY][field] = value
    with pytest.raises((ValueError, ControllerError)):
        prepare.validate_catalog(cfg, pins, source)


def test_publish_creates_scoped_runtime_cache_only_after_success():
    _, pins, _ = catalog()
    script = prepare.publish_script("/tmp/recipe", "/worker/cache", pins, {"image_id": "sha256:" + "a" * 64})
    assert "/worker/cache/spark-serve/qwen38-tensorfold/runtime-cache/torch_extensions" in script
    assert "prepared.json.tmp" in script and "mv " in script
    assert "full_hash_verified" in script and "cpu_compatibility_verified" in script


def test_unsupported_fifth_stream_fails_before_remote_preparation():
    cfg, pins, source = catalog()
    cfg["models"][prepare.MODEL_KEY]["tensorfold"]["parallel"] = 5
    with pytest.raises(ControllerError, match="parallel"):
        prepare.validate_catalog(cfg, pins, source)


def test_download_disk_check_deducts_cache_and_leaves_repair_headroom(assets):
    root, manifest = assets
    reserve = max(item["size"] for item in manifest["files"]) + 2_000_000_000
    assert space.required_bytes(root, manifest) == reserve
    path = root / "model.safetensors"
    size = path.stat().st_size
    path.unlink()
    assert space.required_bytes(root, manifest) == reserve + size
    # A completed HF blob is reusable even if interrupted before its link exists.
    digest = next(item["lfs"]["sha256"] for item in manifest["files"] if item.get("lfs"))
    blob = root.parent.parent / "blobs" / digest
    blob.parent.mkdir()
    blob.write_bytes(b"authentic weight payload")
    assert space.required_bytes(root, manifest) == reserve


def test_exact_catalog_serve_arguments_reach_cpu_compatibility_check():
    import shlex
    cfg, pins, _ = catalog()
    plan = prepare.validate_tensorfold_plan(cfg, prepare.MODEL_KEY)
    script = prepare.compatibility_script("/head/cache", cfg["models"][prepare.MODEL_KEY], pins, plan.argv)
    argv = shlex.split(script.splitlines()[-1])
    assert json.loads(argv[-1]) == list(plan.argv)
