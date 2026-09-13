"""Offline checks for pinned Nemotron assets and physical-node preparation."""
import hashlib
import importlib.util
import json
import shlex
import subprocess
import sys

import pytest

from tools import prepare_nemotron as prepare


SPEC = importlib.util.spec_from_file_location("nemotron_checkpoint_verify", prepare.RECIPE / "verify.py")
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


def blob_sha(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


@pytest.fixture
def snapshot(tmp_path):
    cache = tmp_path / "cache with spaces" / "hub"
    manifest = {"model": "nvidia/test", "revision": "pinned-revision",
                "architectures": ["NemotronHForCausalLM"], "files": []}
    root = cache / "models--nvidia--test" / "snapshots" / manifest["revision"]
    root.mkdir(parents=True)
    contents = {
        "config.json": b'{"architectures":["NemotronHForCausalLM"]}',
        "model.safetensors.index.json": b'{"weight_map":{"weight":"weights.safetensors"}}',
        "tokenizer_config.json": b"{}", "generation_config.json": b"{}",
        "hf_quant_config.json": b"{}", "special_tokens_map.json": b"{}",
        "tokenizer.json": b'{"version":"1.0"}', "chat_template.jinja": b"original",
        "weights.safetensors": b"correct weights",
    }
    for name, data in contents.items():
        (root / name).write_bytes(data)
        item = {"path": name, "size": len(data), "lfs": None}
        if name in ("weights.safetensors", "model.safetensors.index.json", "tokenizer.json"):
            item["lfs"] = {"sha256": hashlib.sha256(data).hexdigest()}
        else:
            item["git_blob_sha1"] = blob_sha(data)
        manifest["files"].append(item)
    return root, manifest, cache, contents


def test_complete_snapshot_supports_nemotron_without_vision_metadata(snapshot):
    root, manifest, _, contents = snapshot
    result = verify.verify(root, manifest, full_hash=True)
    assert result["bytes"] == sum(map(len, contents.values()))
    assert result["sha256_verified"] is True
    assert result["architectures"] == ["NemotronHForCausalLM"]


@pytest.mark.parametrize("filename", ["chat_template.jinja", "tokenizer.json", "model.safetensors.index.json"])
def test_same_size_git_and_small_lfs_metadata_corruption_is_caught_at_startup(snapshot, filename):
    root, manifest, _, contents = snapshot
    (root / filename).write_bytes(b"x" * len(contents[filename]))
    with pytest.raises(verify.CheckpointFileError) as failure:
        verify.verify(root, manifest)
    assert failure.value.filename == filename


def test_setup_can_repair_only_one_bad_file_at_the_pinned_public_revision(snapshot):
    root, manifest, cache, contents = snapshot
    (root / "weights.safetensors").unlink()
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        (root / kwargs["filename"]).write_bytes(contents[kwargs["filename"]])
    assert verify.verify_with_repair(root, manifest, cache, download)["sha256_verified"]
    assert calls == [{"repo_id": "nvidia/test", "filename": "weights.safetensors",
                      "revision": "pinned-revision", "cache_dir": str(cache),
                      "force_download": True, "token": False}]


def test_missing_second_file_still_fails_after_one_repair(snapshot):
    root, manifest, cache, contents = snapshot
    (root / "tokenizer.json").unlink()
    (root / "weights.safetensors").unlink()
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        (root / kwargs["filename"]).write_bytes(contents[kwargs["filename"]])
    with pytest.raises(verify.CheckpointFileError):
        verify.verify_with_repair(root, manifest, cache, download)
    assert len(calls) == 1


def test_wrong_revision_or_architecture_fails(snapshot):
    root, manifest, _, _ = snapshot
    manifest["architectures"] = ["OtherModel"]
    with pytest.raises(ValueError, match="architecture"):
        verify.verify(root, manifest)
    manifest["revision"] = "unpinned"
    with pytest.raises(ValueError, match="pinned snapshot"):
        verify.verify(root, manifest)


@pytest.mark.parametrize("node", ["head", "worker"])
@pytest.mark.parametrize("skip_download", [False, True])
def test_preparation_uses_only_selected_physical_node_and_cache(tmp_path, monkeypatch, node, skip_download):
    catalog = tmp_path / "catalog.toml"
    worker_cache = "/worker/cache with ' quote"
    catalog.write_text('[cluster]\nhead="head-ssh"\nworker="worker-ssh"\n'
                       'lan_url="http://head:8000"\nworker_lan_url="http://worker:8000"\n'
                       'hf_cache_host="/head/cache"\nworker_hf_cache_host=' + json.dumps(worker_cache) + '\n'
                       '[models.nemotron-super]\nnnodes=1\nrecipe="nemotron-super-nvfp4"\nimage="nemotron:test"\n')
    monkeypatch.setattr(sys, "argv", ["prepare", "--catalog", str(catalog), "--node", node]
                        + (["--skip-download"] if skip_download else []))
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="/tmp/spark-serve-nemotron-super.12345678\n")
    monkeypatch.setattr(prepare.subprocess, "run", run)
    prepare.main()
    assert all(command[3] == node + "-ssh" for command, _ in calls)
    script = calls[-1][1]["input"]
    cache = "/head/cache" if node == "head" else worker_cache
    source = json.loads((prepare.RECIPE / "model-source.json").read_text())
    expected = cache + "/hub/models--" + source["model"].replace("/", "--") + "/snapshots/" + source["revision"]
    assert shlex.quote(expected) in script
    assert "docker build" in script
    assert "docker run" not in script and "docker stop" not in script
    if skip_download:
        assert "snapshot_download" not in script and "--repair-cache" not in script
        assert "python3 verify.py" in script
    else:
        assert "token=False" in script and source["revision"] in script
        assert "--repair-cache " + shlex.quote(cache + "/hub") in script


def test_published_manifest_has_complete_authentication():
    source = json.loads((prepare.RECIPE / "model-source.json").read_text())
    assert len(source["files"]) == 36
    assert sum(item["size"] for item in source["files"]) == 80365684262
    assert len({item["path"] for item in source["files"]}) == 36
    for item in source["files"]:
        if item["lfs"]:
            assert len(item["lfs"]["sha256"]) == 64
        else:
            assert len(item["git_blob_sha1"]) == 40
