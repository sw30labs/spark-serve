"""Offline integrity, resumed download and two-node placement checks."""
import copy
import hashlib
import importlib.util
import io
import json
import shlex
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from tools import prepare_glm53_exl3 as prepare


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


verify = module("exl3_verify", prepare.RECIPE / "verify.py")
with patch.dict(sys.modules, {"verify": verify}):
    download = module("exl3_download", prepare.RECIPE / "download.py")


@pytest.fixture
def assets(tmp_path):
    manifest = {"version": 1, "checkpoints": {}}
    for role, revision, architecture in (("target", "1" * 40, "Glm5NextForConditionalGeneration"),
                                          ("draft", "2" * 40, "DFlash2DraftModel")):
        snapshot = tmp_path / "models" / role / revision
        snapshot.mkdir(parents=True)
        data = {"config.json": json.dumps({"architectures": [architecture]}).encode(),
                "model.safetensors": b"authentic weight payload"}
        files = []
        for name, body in data.items():
            (snapshot / name).write_bytes(body)
            files.append({"path": name, "size": len(body), "sha256": hashlib.sha256(body).hexdigest()})
        manifest["checkpoints"][role] = {"model": "test/" + role, "revision": revision,
                                            "architectures": [architecture], "files": files}
    return tmp_path, manifest


def test_full_hash_covers_target_and_draft(assets):
    root, manifest = assets
    result = verify.verify(root, manifest)
    assert result["full_hash_verified"] is True
    assert set(result["checkpoints"]) == {"target", "draft"}


@pytest.mark.parametrize("role", ["target", "draft"])
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


def test_unverified_index_reference_is_rejected(assets):
    root, manifest = assets
    target = manifest["checkpoints"]["target"]
    snapshot = verify.snapshot_path(root, "target", target)
    body = json.dumps({"weight_map": {"weight": "other.safetensors"}}).encode()
    (snapshot / "model.safetensors.index.json").write_bytes(body)
    target["files"].append({"path": "model.safetensors.index.json", "size": len(body),
                            "sha256": hashlib.sha256(body).hexdigest()})
    with pytest.raises(ValueError, match="unverified shards"):
        verify.verify(root, manifest)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../escape", "a//b", "a\\b"])
def test_manifest_rejects_unsafe_paths(assets, name):
    _, manifest = assets
    manifest["checkpoints"]["target"]["files"][0]["path"] = name
    with pytest.raises(ValueError, match="manifest path"):
        verify.checkpoints(manifest)


def test_external_symlink_is_rejected(assets, tmp_path):
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
    def forbidden(*args):
        raise AssertionError("must authenticate manifest before checkpoint reads")
    monkeypatch.setattr(verify, "verify", forbidden)
    with pytest.raises(SystemExit):
        verify.main()
    assert "does not match controller pin" in capsys.readouterr().err


class Response(io.BytesIO):
    def __init__(self, data, status=200, content_range=""):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Range": content_range}


def file_case(tmp_path):
    data = b"abcdef"
    return {"path": "weights.bin", "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}, data, tmp_path / "weights.bin"


def test_resume_valid_range(tmp_path):
    item, data, target = file_case(tmp_path)
    target.with_name("weights.bin.incomplete").write_bytes(data[:3])
    requests = []
    def opener(request, **kwargs):
        requests.append(request)
        return Response(data[3:], 206, "bytes 3-5/6")
    download.download_file(tmp_path, item, "https://huggingface.co/test/model/resolve/" + "1" * 40,
                           opener=opener)
    assert requests[0].headers["Range"] == "bytes=3-"
    assert target.read_bytes() == data


def test_ignored_range_replaces_partial_instead_of_appending(tmp_path):
    item, data, target = file_case(tmp_path)
    target.with_name("weights.bin.incomplete").write_bytes(data[:3])
    download.download_file(tmp_path, item, "https://example.test", opener=lambda *a, **k: Response(data))
    assert target.read_bytes() == data


def test_corrupt_transfer_cannot_replace_prior_file(tmp_path):
    item, data, target = file_case(tmp_path)
    target.write_bytes(b"oldold")
    with pytest.raises(verify.CheckpointFileError):
        download.download_file(tmp_path, item, "https://example.test", attempts=1,
                               opener=lambda *a, **k: Response(b"badbad"))
    assert target.read_bytes() == b"oldold"


def test_verified_file_is_reused_without_network(tmp_path):
    item, data, target = file_case(tmp_path)
    target.write_bytes(data)
    def forbidden(*args, **kwargs):
        raise AssertionError("must reuse authenticated local file")
    download.download_file(tmp_path, item, "https://example.test", opener=forbidden)


def test_published_pins_and_complete_hf_manifests():
    pins, source = prepare.load_pins()
    models = verify.checkpoints(source)
    assert len(models["target"]["files"]) == 144
    assert len(models["draft"]["files"]) == 4
    assert sum(f["size"] for m in models.values() for f in m["files"]) == 178058030609
    assert pins["upstream_revision"] == prepare.UPSTREAM_REVISION
    assert models["target"]["revision"] == "25a44fdbf16862a46b7cc9921142c6c81350af2f"
    assert models["draft"]["revision"] == "dc77ff1c99eeb2df044ee3d4f0094eb033fee410"


def test_worker_copy_uses_qsfp_and_distinct_cache_paths():
    script = prepare.copy_script("/head/cache", "/worker/cache with spaces", "192.168.100.10", "192.168.100.11", "spider", "image:pin")
    assert "BindAddress=192.168.100.10" in script
    assert "spider@192.168.100.11" in script
    assert "--checksum" in script and "--protect-args" in script
    assert "'/worker/cache with spaces/models'" in shlex.split(script.splitlines()[1])[-1]
    assert "--delete" not in script
    assert "huggingface.co" not in script
    assert "docker save image:pin" in script and "docker load" in script


def test_worker_verifier_never_downloads():
    pins, _ = prepare.load_pins()
    script = prepare.download_script("/tmp/recipe", "/worker/cache", pins, True)
    assert "verify.py /worker/cache" in script
    assert "download.py" not in script
    assert "--expected-manifest-sha256 " + pins["manifest_sha256"] in script


def test_matching_worker_image_skips_transfer():
    script = prepare.copy_script("/head", "/worker", "192.168.100.10", "192.168.100.11", "spider", None)
    assert "rsync" in script
    assert "docker save" not in script and "docker load" not in script


def test_image_build_and_cpu_check_leave_workloads_untouched():
    pins, _ = prepare.load_pins()
    script = prepare.build_script("/tmp/recipe", pins)
    assert pins["base_image"] in script
    assert "--network none --pull=false" in script
    assert "--patches-only" in script
    assert all(token not in script for token in ("--gpus", "docker stop", "docker rm", "systemctl"))


def test_prepare_catalog_rejects_unpinned_verifier_and_wrong_target():
    pins, source = prepare.load_pins()
    root = "/cache/huggingface/spark-serve/glm53-exl3"
    model = {"recipe": "glm53-exl3", "image": pins["local_image"], "nnodes": 2,
             "hf_id": source["checkpoints"]["target"]["model"],
             "serve_path": root + "/models/target/" + source["checkpoints"]["target"]["revision"],
             "preflight_args": ["/opt/spark-serve/glm53-exl3/verify.py", root,
                                "--expected-manifest-sha256", pins["manifest_sha256"]]}
    cfg = {"cluster": {"head": "sparkone", "worker": "sparktwo", "hf_cache_host": "/head/cache",
                        "worker_hf_cache_host": "/worker/cache"}, "models": {"glm53-exl3": model}}
    assert prepare.validate_catalog(cfg, pins, source) == model
    assert prepare.node_root(cfg, "worker") == "/worker/cache/spark-serve/glm53-exl3"
    model["preflight_args"][-1] = "0" * 64
    with pytest.raises(ValueError, match="preflight"):
        prepare.validate_catalog(cfg, pins, source)
