"""Offline integrity, resumed download and two-node placement checks."""
import copy
import hashlib
import importlib.util
import io
import json
import shlex
import sys
import tomllib
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import pytest

from tools import prepare_glm53_tensorfold as prepare
from spark_serve_controller import ControllerError


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


verify = module("glm_tensorfold_verify", prepare.RECIPE / "verify.py")
with patch.dict(sys.modules, {"verify": verify}):
    space = module("glm_tensorfold_space", prepare.RECIPE / "check_space.py")
with patch.dict(sys.modules, {"verify": verify, "check_space": space}):
    download = module("glm_tensorfold_download", prepare.RECIPE / "download.py")
runtime = module("glm_tensorfold_runtime_verify", prepare.RECIPE / "runtime/verify_runtime.py")


def catalog():
    cfg = tomllib.loads((prepare.ROOT / "models.example.toml").read_text())
    cfg["cluster"].update(head="sparkone", worker="sparktwo", hf_cache_host="/head/cache",
                          worker_hf_cache_host="/worker/cache")
    return cfg


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
                                    "--expected-manifest-sha256", "0" * 64, "--verify-runtime"])
    def forbidden(*args, **kwargs):
        raise AssertionError("must authenticate manifest before checkpoint reads")
    monkeypatch.setattr(verify, "verify", forbidden)
    monkeypatch.setattr(verify.subprocess, "run", forbidden)
    with pytest.raises(SystemExit):
        verify.main()
    assert "does not match controller pin" in capsys.readouterr().err


@pytest.mark.parametrize("rank", [0, 1])
def test_preflight_accepts_exact_cli_serving_json_and_checks_both_snapshot_paths(assets, monkeypatch, capsys, rank):
    root, manifest = assets
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    planned = [str(verify.snapshot_path(root, "target", manifest["checkpoints"]["target"])),
               "--backend", "cuda", "--tp", "2", "--rank", str(rank)]
    calls = []
    monkeypatch.setattr(verify.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "check_compatibility", SimpleNamespace(
        verify_model=lambda target, draft, args: calls.append((target, draft, args))))
    monkeypatch.setattr(sys, "argv", ["verify", str(root), "--manifest", str(path),
                                    "--expected-manifest-sha256", digest, "--verify-runtime",
                                    "--serve-args-json", json.dumps(planned)])
    verify.main()
    assert calls == [(verify.snapshot_path(root, "target", manifest["checkpoints"]["target"]),
                      verify.snapshot_path(root, "draft", manifest["checkpoints"]["draft"]), planned)]
    assert json.loads(capsys.readouterr().out)["full_hash_verified"]


def test_serving_arguments_cannot_skip_runtime_authentication(assets, monkeypatch, capsys):
    root, manifest = assets
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(sys, "argv", ["verify", str(root), "--manifest", str(path),
                                    "--expected-manifest-sha256", digest, "--serve-args-json", "[]"])
    with pytest.raises(SystemExit):
        verify.main()
    assert "requires runtime authentication" in capsys.readouterr().err


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


def test_complete_authenticated_cache_needs_neither_download_nor_repair_space(assets, monkeypatch):
    root, manifest = assets
    def forbidden(*args, **kwargs):
        raise AssertionError("fully authenticated checkpoint must be reusable without download or repair space")
    monkeypatch.setattr(download, "check_space", forbidden)
    monkeypatch.setattr(download, "download_file", forbidden)
    assert download.download(root, manifest)["full_hash_verified"]


def test_published_pins_and_complete_hf_manifests():
    pins, source = prepare.load_pins()
    models = verify.checkpoints(source)
    assert len(models["target"]["files"]) == 97
    assert len(models["draft"]["files"]) == 5
    assert sum(f["size"] for m in models.values() for f in m["files"]) == 178058596393
    assert pins["upstream_revision"] == prepare.UPSTREAM_REVISION
    assert models["target"]["revision"] == "078455ffe6472f9a52fbc1139f58b9db2881b25c"
    assert models["draft"]["revision"] == "bf582e4eacc1810f76656d1811693ff6c6737d2a"
    installed = json.loads((prepare.RECIPE / "runtime/installed-runtime.json").read_text())
    assert installed["tensorfold_revision"] == pins["tensorfold_revision"]
    assert installed["tensorfold_version"] == "0.6.0"
    assert len(installed["files"]) == 426
    assert installed["patch_count"] == 82
    assert installed["patches_hash"] == pins["patches_hash"]
    assert installed["dependencies"] == {"av": "18.1.0", "xgrammar": "0.2.8", "transformers": "5.18.0"}


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
    assert "--verify-runtime" in script
    assert all(token not in script for token in ("--gpus", "docker stop", "docker rm", "systemctl"))


def test_prepare_catalog_rejects_unpinned_verifier_and_wrong_target():
    pins, source = prepare.load_pins()
    cfg = catalog()
    model = cfg["models"][prepare.MODEL_KEY]
    assert prepare.validate_catalog(cfg, pins, source) == model
    assert prepare.node_root(cfg, "worker") == "/worker/cache/spark-serve/glm53-tensorfold"
    model["preflight_args"][3] = "0" * 64
    with pytest.raises((ValueError, ControllerError)):
        prepare.validate_catalog(cfg, pins, source)


def test_runtime_rejects_same_size_corruption_and_extra_package_files(tmp_path):
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


def test_runtime_rejects_external_symlink_even_with_matching_bytes(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"real code")
    (package / "server.py").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        runtime.verify_files(package, {"server.py": hashlib.sha256(outside.read_bytes()).hexdigest()})


@pytest.mark.parametrize("rank", [0, 1])
def test_exact_rank_arguments_reach_offline_cpu_parser(rank):
    pins, source = prepare.load_pins()
    cfg = catalog()
    model = prepare.validate_catalog(cfg, pins, source)
    plan = prepare.validate_tensorfold_plan(cfg, prepare.MODEL_KEY, rank=rank)
    script = prepare.compatibility_script("/worker/cache", model, pins, source, plan.argv)
    argv = shlex.split(script.splitlines()[-1])
    assert json.loads(argv[-1]) == list(plan.argv)
    assert argv[-2].endswith("/models/draft/" + source["checkpoints"]["draft"]["revision"])
    assert "NVIDIA_VISIBLE_DEVICES=void" in script and "CUDA_VISIBLE_DEVICES=" in script
    assert "--network none" in script and ":/cache/huggingface:ro" in script
    assert all(word not in script for word in ("--gpus", "docker stop", "docker rm", "tensorfold serve"))


def test_mtp_launch_arguments_are_checked_without_using_dflash2():
    cfg = catalog()
    pins, source = prepare.load_pins()
    model = cfg["models"][prepare.MODEL_KEY]
    model["tensorfold"].update(drafter="mtp", parallel=1)
    model["max_num_seqs"] = 1
    plan = prepare.validate_tensorfold_plan(cfg, prepare.MODEL_KEY)
    args = list(plan.argv)
    assert args[args.index("--drafter") + 1] == "none"
    script = prepare.compatibility_script("/head/cache", model, pins, source, plan.argv)
    assert json.loads(shlex.split(script.splitlines()[-1])[-1]) == args


@pytest.mark.parametrize("field,value", [("serve_path", "/unpinned"), ("preflight_args", []),
                                         ("wrapper", "vllm"), ("hf_mount", "/different"), ("nnodes", 1)])
def test_catalog_changes_cannot_bypass_checkpoint_preflight(field, value):
    cfg = catalog()
    pins, source = prepare.load_pins()
    cfg["models"][prepare.MODEL_KEY][field] = value
    with pytest.raises((ValueError, ControllerError)):
        prepare.validate_catalog(cfg, pins, source)


def test_disk_check_accounts_for_both_models_and_resumable_downloads(assets):
    root, manifest = assets
    reserve = max(item["size"] for model in manifest["checkpoints"].values() for item in model["files"]) + 2_000_000_000
    assert space.required_bytes(root, manifest) == reserve
    snapshot = verify.snapshot_path(root, "target", manifest["checkpoints"]["target"])
    path = snapshot / "model.safetensors"
    data = path.read_bytes()
    path.unlink()
    assert space.required_bytes(root, manifest) == reserve + len(data)
    path.with_name(path.name + ".incomplete").write_bytes(data[:3])
    assert space.required_bytes(root, manifest) == reserve + len(data) - 3


def test_publication_records_both_checkpoints_and_cpu_parser_success():
    pins, _ = prepare.load_pins()
    script = prepare.publish_script("/tmp/recipe", "/worker/cache/spark-serve/glm53-tensorfold", pins, {"image_id": "sha256:" + "a" * 64})
    assert "prepared.json.tmp" in script and "mv " in script
    assert "full_hash_verified" in script and "cpu_compatibility_verified" in script
    assert all(revision in script for revision in pins["checkpoint_revisions"].values())
    assert "/runtime-cache/torch_extensions" in script


def test_invalid_worker_cache_fails_before_upload():
    cfg = catalog()
    pins, source = prepare.load_pins()
    cfg["cluster"]["worker_hf_cache_host"] = "/worker/../wrong"
    with pytest.raises((ValueError, ControllerError)):
        prepare.validate_catalog(cfg, pins, source)


@pytest.mark.parametrize("fail_worker_compatibility", [False, True])
def test_preparation_authenticates_entire_allocation_before_publishing(monkeypatch, fail_worker_compatibility):
    calls = []
    image = {"image_id": "sha256:" + "a" * 64, "rootfs_layers": ["sha256:" + "b" * 64]}
    monkeypatch.setattr(sys, "argv", ["prepare", "--catalog", str(prepare.ROOT / "models.example.toml"), "--skip-download"])
    monkeypatch.setattr(prepare, "upload_recipe", lambda cfg, role: "/tmp/recipe-" + role)
    monkeypatch.setattr(prepare, "image_report", lambda *args: image)
    monkeypatch.setattr(prepare, "qsfp_address", lambda cfg, role: "192.168.100.10" if role == "head" else "192.168.100.11")
    monkeypatch.setattr(prepare, "remote_text", lambda cfg, role, script: "spider" if script == "id -un" else image["image_id"])
    monkeypatch.setattr(prepare.subprocess, "run", lambda argv, **kwargs: calls.append(("copy", shlex.join(argv))))
    def remote(cfg, role, script):
        calls.append((role, script))
        if fail_worker_compatibility and role == "worker" and "check_compatibility.py" in script:
            raise RuntimeError("worker parser failed")
    monkeypatch.setattr(prepare, "run_remote", remote)
    if fail_worker_compatibility:
        with pytest.raises(RuntimeError, match="worker parser"):
            prepare.main()
        assert not any("prepared.json.tmp" in script for _, script in calls)
    else:
        prepare.main()
        assert [role for role, script in calls[-2:]] == ["head", "worker"]
        assert all("prepared.json.tmp" in script for _, script in calls[-2:])
    assert all(word not in script for _, script in calls for word in ("docker stop", "docker rm", "--gpus", "systemctl"))
