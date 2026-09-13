"""Offline integrity and targeted cache-repair checks for the pinned recipe."""
import hashlib
import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from tools import prepare_qwen38 as prepare


SPEC = importlib.util.spec_from_file_location("qwen_checkpoint_verify", prepare.RECIPE / "verify.py")
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


def blob_sha(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


@pytest.fixture
def snapshot(tmp_path):
    cache = tmp_path / "cache with spaces" / "hub"
    manifest = {"model": "nvidia/test", "revision": "pinned-revision", "files": []}
    root = cache / "models--nvidia--test" / "snapshots" / manifest["revision"]
    root.mkdir(parents=True)
    contents = {
        "config.json": b'{"architectures":["Qwen"]}',
        "model.safetensors.index.json": b'{"weight_map":{"weight":"weights.safetensors"}}',
        "tokenizer_config.json": b"{}",
        "preprocessor_config.json": b"{}",
        "hf_quant_config.json": b"{}",
        "chat_template.jinja": b"original",
        "weights.safetensors": b"correct weights",
    }
    for name, data in contents.items():
        (root / name).write_bytes(data)
        item = {"path": name, "size": len(data), "lfs": None}
        if name.endswith("safetensors"):
            item["lfs"] = {"sha256": hashlib.sha256(data).hexdigest()}
        else:
            item["git_blob_sha1"] = blob_sha(data)
        manifest["files"].append(item)
    return root, manifest, cache, contents


def test_complete_snapshot_checks_both_git_blob_and_lfs_integrity(snapshot):
    root, manifest, _, contents = snapshot
    result = verify.verify(root, manifest, full_hash=True)
    assert result["sha256_verified"] is True
    assert result["bytes"] == sum(map(len, contents.values()))


def test_same_size_template_corruption_is_caught_during_fast_startup(snapshot):
    root, manifest, _, _ = snapshot
    (root / "chat_template.jinja").write_bytes(b"modified")
    with pytest.raises(verify.CheckpointFileError, match="Git blob") as failure:
        verify.verify(root, manifest)
    assert failure.value.filename == "chat_template.jinja"


def test_full_setup_hash_catches_corrupt_shard_beyond_fast_size_check(snapshot):
    root, manifest, _, contents = snapshot
    (root / "weights.safetensors").write_bytes(b"x" * len(contents["weights.safetensors"]))
    assert verify.verify(root, manifest)["sha256_verified"] is False
    with pytest.raises(verify.CheckpointFileError, match="SHA-256"):
        verify.verify(root, manifest, full_hash=True)


@pytest.mark.parametrize("filename", ["chat_template.jinja", "weights.safetensors"])
def test_repair_forces_only_bad_file_at_pinned_public_revision_then_rechecks_all(snapshot, filename):
    root, manifest, cache, contents = snapshot
    (root / filename).write_bytes(b"x" * len(contents[filename]))
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        (root / filename).write_bytes(contents[filename])
    result = verify.verify_with_repair(root, manifest, cache, download)
    assert result["sha256_verified"] is True
    assert calls == [{"repo_id": "nvidia/test", "filename": filename,
                      "revision": "pinned-revision", "cache_dir": str(cache),
                      "force_download": True, "token": False}]


def test_failed_repair_is_not_retried_forever(snapshot):
    root, manifest, cache, _ = snapshot
    (root / "weights.safetensors").unlink()
    calls = []
    with pytest.raises(verify.CheckpointFileError):
        verify.verify_with_repair(root, manifest, cache, lambda **kwargs: calls.append(kwargs))
    assert len(calls) == 1


def test_repair_rechecks_remaining_files_instead_of_accepting_first_repaired_file(snapshot):
    root, manifest, cache, contents = snapshot
    (root / "chat_template.jinja").unlink()
    (root / "weights.safetensors").unlink()
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        (root / kwargs["filename"]).write_bytes(contents[kwargs["filename"]])
    with pytest.raises(verify.CheckpointFileError) as failure:
        verify.verify_with_repair(root, manifest, cache, download)
    assert failure.value.filename == "weights.safetensors"
    assert len(calls) == 1


def test_repair_rejects_unrelated_cache_path_without_network(snapshot):
    root, manifest, cache, _ = snapshot
    with pytest.raises(ValueError, match="repair cache"):
        verify.verify_with_repair(root, manifest, cache / "other", lambda **_: pytest.fail("network"))


def test_hub_blob_symlinks_are_supported_but_manifest_traversal_is_rejected(snapshot):
    root, manifest, cache, contents = snapshot
    blob = cache / "blob"
    blob.write_bytes(contents["weights.safetensors"])
    (root / "weights.safetensors").unlink()
    (root / "weights.safetensors").symlink_to(blob)
    verify.verify(root, manifest, full_hash=True)
    manifest["files"][0]["path"] = "../../outside"
    with pytest.raises(ValueError, match="manifest path"):
        verify.verify(root, manifest)


def make_recipe(tmp_path):
    recipe = tmp_path / "recipe"
    patch = recipe / "single-spark-vllm-tp1" / "patch"
    patch.mkdir(parents=True)
    files = {"LICENSE": b"license", "NOTICE": b"notice", "single-spark-vllm-tp1/patch/overlay.py": b"print('overlay')"}
    for filename, data in files.items():
        (recipe / filename).write_bytes(data)
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    (recipe / "overlays.sha256.json").write_text(json.dumps(hashes))
    (recipe / "model-source.json").write_text(json.dumps({"model": "nvidia/test", "revision": "pinned-revision"}))
    return recipe


def test_vendored_overlay_checksums_and_complete_file_list_are_enforced(tmp_path):
    recipe = make_recipe(tmp_path)
    prepare.validate_overlays(recipe)
    patch = recipe / "single-spark-vllm-tp1/patch/overlay.py"
    original = patch.read_bytes()
    patch.write_bytes(b"altered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare.validate_overlays(recipe)
    patch.write_bytes(original)
    (patch.parent / "unlisted.py").write_text("unexpected")
    with pytest.raises(ValueError, match="do not match"):
        prepare.validate_overlays(recipe)


@pytest.mark.parametrize("skip_download", [False, True])
def test_preparation_paths_are_quoted_and_offline_mode_never_repairs(tmp_path, monkeypatch, skip_download):
    recipe = make_recipe(tmp_path)
    catalog = tmp_path / "catalog.toml"
    cache = "/cache/a space and ' quote"
    catalog.write_text('[cluster]\nhead="sparkone"\nhf_cache_host=' + json.dumps(cache) + '\n'
                       '[models.qwen38]\nnnodes=1\nrecipe="qwen38-nvfp4"\nimage="qwen:test"\n')
    monkeypatch.setattr(prepare, "RECIPE", recipe)
    argv = ["prepare", "--catalog", str(catalog)] + (["--skip-download"] if skip_download else [])
    monkeypatch.setattr(sys, "argv", argv)
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="/tmp/spark-serve-qwen38.12345678\n")
    monkeypatch.setattr(prepare.subprocess, "run", run)
    prepare.main()
    script = calls[-1][1]["input"]
    snapshot_path = cache + "/hub/models--nvidia--test/snapshots/pinned-revision"
    assert shlex.quote(snapshot_path) in script
    if skip_download:
        assert "snapshot_download" not in script
        assert "--repair-cache" not in script
        assert "python3 verify.py" in script
    else:
        assert "token=False" in script
        assert "--repair-cache " + shlex.quote(cache + "/hub") in script
        assert '"$venv/bin/python" verify.py' in script


def test_real_recipe_source_hashes_are_valid():
    prepare.validate_overlays(prepare.RECIPE)
