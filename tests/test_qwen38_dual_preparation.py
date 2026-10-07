"""Pinned TP2 runtime, cache preservation and non-disruptive preparation."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
from types import SimpleNamespace

import pytest

from tools import prepare_qwen38_dual as prepare


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verify = load_module("qwen_dual_verify", prepare.RECIPE / "verify.py")
runtime = load_module("qwen_dual_runtime", prepare.RECIPE / "runtime/verify_runtime.py")


def catalog():
    cfg = tomllib.loads((prepare.ROOT / "models.example.toml").read_text())
    cfg["cluster"]["hf_cache_host"] = "/head/cache"
    cfg["cluster"]["worker_hf_cache_host"] = "/worker/cache"
    return cfg


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


def test_pins_reuse_nvidia_checkpoint_and_distinct_tp_aware_sources():
    pins, source = prepare.load_pins()
    assert (prepare.RECIPE / "model-source.json").read_bytes() == (prepare.ROOT / "recipes/qwen38-nvfp4/model-source.json").read_bytes()
    assert len(verify.manifest_files(source)) == 25
    assert source["revision"] == prepare.CHECKPOINT_REVISION
    assert pins["upstream_revision"] == prepare.UPSTREAM_REVISION
    runtime.verify_upstream(prepare.RECIPE / "runtime")
    for filename in ("patch_mtp_draft_vocab.py", "draft_vocab_en_code_47k.txt"):
        assert (prepare.RECIPE / "runtime/upstream" / filename).read_bytes() != (prepare.ROOT / "recipes/qwen38-v030/runtime/upstream" / filename).read_bytes()


def test_same_size_checkpoint_corruption_and_extra_shard_fail(assets):
    root, manifest = assets
    assert verify.verify(root, manifest)["sha256_verified"]
    path = root / "model.safetensors"
    authentic = path.read_bytes()
    path.write_bytes(b"x" * len(authentic))
    with pytest.raises(verify.CheckpointFileError, match="SHA-256 mismatch"):
        verify.verify(root, manifest)
    path.write_bytes(authentic)
    (root / "stale.safetensors").write_bytes(b"stale")
    with pytest.raises(ValueError, match="inventory differs"):
        verify.verify(root, manifest)


def test_transfer_lists_pinned_snapshot_and_referenced_blob_only(assets, monkeypatch):
    root, manifest = assets
    path = root / "model.safetensors"
    repo = root.parent.parent
    blob = repo / "blobs" / "weight-digest"
    blob.parent.mkdir()
    path.rename(blob)
    path.symlink_to("../../blobs/weight-digest")
    (blob.parent / "unrelated-revision-blob").write_bytes(b"unrelated")
    monkeypatch.setitem(sys.modules, "verify", verify)
    transfer = load_module("qwen_dual_transfer", prepare.RECIPE / "prepare_transfer.py")
    files = transfer.transfer_files(root, manifest)
    assert files == sorted(["blobs/weight-digest", *[f"snapshots/{root.name}/{item['path']}" for item in manifest["files"]]])
    # Reconstruct the exact payload at a different cache root and authenticate
    # it there, proving the HF links still address the copied blob.
    destination = repo.parent / "different-worker-cache" / repo.name
    for name in files:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        source = repo / name
        if source.is_symlink():
            target.symlink_to(source.readlink())
        else:
            shutil.copyfile(source, target)
    assert verify.verify(destination / "snapshots" / root.name, manifest)["sha256_verified"]
    path.unlink()
    path.symlink_to(blob)
    with pytest.raises(ValueError, match="nonportable"):
        transfer.transfer_files(root, manifest)


def test_catalog_retains_shared_fabric_and_rejects_solo_placement():
    cfg = catalog()
    pins, source = prepare.load_pins()
    model = prepare.validate_catalog(cfg, pins, source)
    assert prepare.node_cache(cfg, "head") == "/head/cache"
    assert prepare.node_cache(cfg, "worker") == "/worker/cache"
    assert "NCCL_IB_DISABLE" not in model["env"]
    assert model["nnodes"] == model["tensor_parallel"] == 2
    for node in ("head", "worker"):
        with pytest.raises(RuntimeError, match="both Sparks"):
            prepare.placement(cfg, model, node)


@pytest.mark.parametrize("flag,value", [("--tensor-parallel-size", "1"),
                                       ("--mm-encoder-tp-mode", "weights"),
                                       ("--kv-cache-dtype", "fp8")])
def test_catalog_rejects_incompatible_tp_vision_or_unpatched_fp8(flag, value):
    cfg = catalog()
    args = cfg["models"][prepare.MODEL_KEY]["vllm"]["args"]
    args[args.index(flag) + 1] = value
    pins, source = prepare.load_pins()
    with pytest.raises(ValueError, match=flag):
        prepare.validate_catalog(cfg, pins, source)


def test_prepare_source_drift_fails_before_upload(tmp_path):
    recipe = tmp_path / "recipe"
    shutil.copytree(prepare.RECIPE, recipe)
    path = recipe / "runtime/upstream/patch_mtp_draft_vocab.py"
    path.write_text(path.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        prepare.load_pins(recipe)


def test_cpu_checks_and_copy_use_only_preparation_operations():
    cfg = catalog()
    pins, source = prepare.load_pins()
    scripts = [prepare.build_script("/tmp/recipe", pins),
               prepare.compatibility_script("/worker/cache", cfg["models"][prepare.MODEL_KEY], pins),
               prepare.copy_script("/tmp/recipe", "/head/cache", "/worker/cache", source, pins,
                                   "192.168.100.10", "192.168.100.11", "worker", True, pins["local_image"])]
    for script in scripts:
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)
        assert all(operation not in script for operation in ("--gpus", "docker stop", "docker rm", "systemctl", "vllm serve"))
    assert "--network none --pull=false" in scripts[0]
    assert "NVIDIA_VISIBLE_DEVICES=void" in scripts[1]
    assert "/worker/cache:/cache/huggingface:ro" in scripts[1]
    assert "BindAddress=192.168.100.10" in scripts[2]
    assert "worker@192.168.100.11" in scripts[2]
    assert "--files-from /tmp/recipe/snapshot-transfer.txt" in scripts[2]
    assert "--delete" not in scripts[2]
    assert "docker save" in scripts[2] and "docker load" in scripts[2]


def test_cached_head_never_downloads_or_installs_dependencies():
    pins, source = prepare.load_pins()
    cached = prepare.checkpoint_script("/tmp/recipe", "/head/cache", pins, source, False).split("\nelse\n")[0]
    assert "verify.py" in cached
    assert "bin/pip" not in cached and "huggingface_hub" not in cached
    skipped = prepare.checkpoint_script("/tmp/recipe", "/head/cache", pins, source, True)
    assert "snapshot_download" not in skipped and "bin/pip" not in skipped


def test_dual_generator_applies_tp_aware_reduction_and_rejects_drift(tmp_path):
    upstream = prepare.RECIPE / "runtime/upstream"
    for filename in ("patch_mtp_draft_vocab.py", "patch_mtp_draft_vocab_v030.py"):
        shutil.copyfile(upstream / filename, tmp_path / filename)
    original = tmp_path / "mtp_v030_patched.py.orig"
    original.write_text("""import torch
from torch import nn
from vllm.distributed import get_pp_group

class Qwen4ExpMultiTokenPredictor(nn.Module):
    pass

class Qwen4ExpMTP(nn.Module):
    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states)
    def load_weights(self):
        return loader.load_weights(remap_weight_names(), mapper=mapper)
""")
    patch = [sys.executable, str(tmp_path / "patch_mtp_draft_vocab_v030.py")]
    subprocess.run(patch, check=True, capture_output=True)
    generated = (tmp_path / "mtp_v030_patched.py").read_text()
    tree = ast.parse(generated)
    methods = {node.name: {child.name for child in node.body if isinstance(child, ast.FunctionDef)}
               for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "get_top_tokens" in methods["Qwen4ExpMTP"]
    assert "get_top_tokens" not in methods["Qwen4ExpMultiTokenPredictor"]
    assert "tensor_model_parallel_all_gather(local_pair, dim=-1)" in generated
    assert "org_vocab_start_index" in generated and "org_vocab_end_index" in generated
    assert "return self.logits_processor(self.lm_head, hidden_states)" in generated
    original.write_text(original.read_text().replace("from vllm.distributed import get_pp_group\n", ""))
    result = subprocess.run(patch, capture_output=True, text=True)
    assert result.returncode != 0 and "anchor 1 not unique/missing" in result.stderr


def test_cpu_checker_validates_the_class_that_owns_the_mtp_head():
    # The backbone Predictor and the full MTP model are different classes.
    # Validate that the GPU-free compatibility checker inspects the same owner
    # that the strict native patch generator modifies.
    tree = ast.parse((prepare.RECIPE / "runtime/check_compatibility.py").read_text())
    owners = [node.args[0].attr for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id == "getattr" and len(node.args) > 1
              and isinstance(node.args[0], ast.Attribute)
              and isinstance(node.args[0].value, ast.Name) and node.args[0].value.id == "mtp"
              and isinstance(node.args[1], ast.Constant) and node.args[1].value == "get_top_tokens"]
    assert owners == ["Qwen4ExpMTP"]


def test_installed_runtime_mutation_is_detected(tmp_path, monkeypatch):
    here = tmp_path / "runtime"
    shutil.copytree(prepare.RECIPE / "runtime", here)
    module = tmp_path / "mtp.py"
    module.write_bytes(b"authentic runtime")
    (here / "installed-patches.json").write_text(json.dumps({"vllm_version": "0.30.0", "files": {
        str(module): {"sha256": hashlib.sha256(module.read_bytes()).hexdigest()}}}))
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda package: "0.30.0")
    runtime.check_installed(here)
    module.write_bytes(b"changed runtime")
    with pytest.raises(ValueError, match="Installed runtime integrity failure"):
        runtime.check_installed(here)


@pytest.mark.parametrize("image_only", [False, True])
def test_main_reuses_authenticated_worker_and_publishes_after_both_compatibility_checks(tmp_path, monkeypatch, image_only):
    cfg = catalog()
    # main reads only a path; avoid leaking the private cluster catalog.
    path = tmp_path / "catalog.toml"
    path.write_text("placeholder")
    monkeypatch.setattr(prepare.tomllib, "loads", lambda text: cfg)
    monkeypatch.setattr(sys, "argv", ["prepare", "--catalog", str(path), *( ["--image-only"] if image_only else [])])
    calls = []
    monkeypatch.setattr(prepare, "upload_recipe", lambda cfg, role: "/tmp/recipe-" + role)
    monkeypatch.setattr(prepare, "run_remote", lambda cfg, role, script: calls.append((role, script)))
    image = {"image_id": "sha256:identical", "rootfs_layers": ["sha256:layer"]}
    monkeypatch.setattr(prepare, "image_report", lambda cfg, role, pins: image)
    def remote_text(cfg, role, script, **kwargs):
        if "SPARK_CHECKPOINT_AUTHENTICATED" in script:
            return "SPARK_CHECKPOINT_AUTHENTICATED"
        if "docker image inspect" in script:
            return image["image_id"]
        raise AssertionError("good worker cache must not need a transfer")
    monkeypatch.setattr(prepare, "remote_text", remote_text)
    monkeypatch.setattr(prepare.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no image or checkpoint copy expected")))
    prepare.main()
    compatibility = [i for i, (_, script) in enumerate(calls) if "check_compatibility.py" in script]
    publication = [i for i, (_, script) in enumerate(calls) if "prepared.json.tmp" in script]
    if image_only:
        assert not compatibility and not publication
        assert not any("verify.py /" in script for _, script in calls)
    else:
        assert len(compatibility) == len(publication) == 2
        assert max(compatibility) < min(publication)
        assert "/head/cache:/cache/huggingface:ro" in calls[compatibility[0]][1]
        assert "/worker/cache:/cache/huggingface:ro" in calls[compatibility[1]][1]
        assert not any(role == "worker" and "python3 verify.py /worker/cache" in script
                       for role, script in calls)
