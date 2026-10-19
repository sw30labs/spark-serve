"""Native Qwen dual TensorFold: pinned two-rank plan, readiness and preparation."""
import copy
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from spark_serve_controller import ControllerError
from spark_serve_nodes import launch_config
from spark_serve_tensorfold import (
    native_memory_need_gib, tensorfold_docker_argv, tensorfold_model_ready, tensorfold_native_health_ready,
    validate_tensorfold_plan,
)
from tools import prepare_qwen38_dual_tensorfold as prepare

ROOT = Path(__file__).resolve().parents[1]
MODEL = "qwen38-dual-tensorfold"
IMAGE = "sha256:" + "b" * 64
loader = importlib.machinery.SourceFileLoader("qwen_dual_tensorfold_cli", str(ROOT / "spark-serve"))
cli = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
loader.exec_module(cli)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


runtime = module("qwen_dual_tf_runtime", prepare.RECIPE / "runtime/verify_runtime.py")
with patch.dict(sys.modules, {"verify_runtime": runtime}):
    compatibility = module("qwen_dual_tf_compat", prepare.RECIPE / "runtime/check_compatibility.py")


def config(only=False):
    cfg = tomllib.loads((ROOT / "models.example.toml").read_text())
    if only:
        cfg["models"] = {MODEL: cfg["models"][MODEL]}
    cfg["cluster"].update(head="first", worker="second", lan_url="http://first:8000",
                          worker_lan_url="http://second:8000", hf_cache_host="/first/cache",
                          worker_hf_cache_host="/second/cache")
    return cfg


def completed(output="", code=0):
    return subprocess.CompletedProcess([], code, output, "")


def health(**changes):
    result = {"status": "ok", "model": "qwen3.8-flash-next-dual-tf", "model_ids": ["qwen3.8-flash-next-dual-tf"],
              "max_batch_size": 16, "warming": False, "memory": {}, "live": {"connections": 0, "waiting": 0}}
    result.update(changes)
    return json.dumps(result)


def test_ranks_share_command_and_fabric_but_use_their_own_cache():
    cfg = config()
    cfg["models"][MODEL]["image"] = IMAGE
    before = copy.deepcopy(cfg)
    plans = [validate_tensorfold_plan(cfg, MODEL, rank=rank) for rank in (0, 1)]
    for rank, (plan, host) in enumerate(zip(plans, ("first", "second"))):
        argv = tensorfold_docker_argv(plan, "generation")
        env = dict(plan.environment)
        assert plan.host == host and plan.rank == rank and plan.hosts == ("first", "second")
        assert f"/{host}/cache:/cache/huggingface:ro" in argv
        assert f"/{host}/cache/spark-serve/qwen38-dual-tensorfold/runtime-cache:/cache" in argv
        assert "/dev/infiniband" in argv and "IPC_LOCK" in argv and "--privileged" not in argv
        assert argv[argv.index("--entrypoint") + 1] == "/opt/spark-serve/qwen38-dual-tensorfold/entrypoint.sh"
        assert argv[argv.index(plan.image) + 1:] == list(plan.argv)
        assert plan.argv[0] == cfg["models"][MODEL]["serve_path"]
        assert plan.argv[plan.argv.index("--rank") + 1] == str(rank)
        assert plan.argv[plan.argv.index("--tp") + 1] == "2"
        assert plan.argv[plan.argv.index("--master") + 1] == "192.168.100.10"
        assert plan.argv[plan.argv.index("--kv-dtype") + 1] == "fp8"
        assert plan.argv[plan.argv.index("--context") + 1] == "1048576"
        assert plan.argv[plan.argv.index("--parallel") + 1] == "16"
        assert "--vision" in plan.argv and "--vision-urls" not in plan.argv and "--no-drafts" not in plan.argv
        assert env["TF_FLASHNEXT_YARN"] == "4" and env["TF_FLASHNEXT_ROCE"] == "1"
        assert env["SPARK_SERVE_MIN_AVAILABLE_GIB"] == "102"
        assert env["NCCL_IB_HCA"] == "rocep1s0f1,roceP2p1s0f1"
        # Python TensorFold's switches do not reach the native engine.
        assert not any(item.startswith(("TENSORFOLD_PREFILL_ROWS=", "TENSORFOLD_MTP_COPY=", "VLLM_", "TF_GLM_"))
                       for item in argv)
        assert "TENSORFOLD_CUDA_KERNELS=/opt/tensorfold/share/tensorfold/cuda/sm121" in argv
    rank0, rank1 = (list(plan.argv) for plan in plans)
    assert [a for a in rank0 if a != "0"] == [a for a in rank1 if a != "1"]
    assert cfg == before


def test_native_window_without_yarn_and_exact_cache_raise_memory_floor():
    cfg = config()
    model = cfg["models"][MODEL]
    model["max_model_len"] = 262144
    model["tensorfold"].update(kv_dtype="bf16", vision=False, drafts=False, roce=False)
    plan = validate_tensorfold_plan(cfg, MODEL)
    env = dict(plan.environment)
    assert env["TF_FLASHNEXT_YARN"] == "0" and env["TF_FLASHNEXT_ROCE"] == "0"
    assert env["SPARK_SERVE_MIN_AVAILABLE_GIB"] == str(native_memory_need_gib(262144, "bf16", False)) == "72"
    assert "--vision" not in plan.argv and "--no-drafts" in plan.argv


def test_memory_floor_matches_upstream_formula():
    # scripts/config.sh mem_need_gib: 48 + vision 2 + ceil((kv + 31) * ctx / 2^20) + floor 10 + slack 2
    assert native_memory_need_gib(1048576, "fp8", True) == 102
    assert native_memory_need_gib(1048576, "bf16", True) == 110
    assert native_memory_need_gib(1, "fp8", False) == 61


@pytest.mark.parametrize("changes", [
    {"nnodes": 1}, {"tensor_parallel": 1}, {"max_model_len": 1048577}, {"max_num_seqs": 8},
    {"preflight_args": ["-c", "pass"]}, {"serve_path": "/cache/huggingface/main"}, {"env": {"TF_FLASHNEXT_ROCE": "0"}},
    {"docker_extra": ["--privileged"]}, {"hf_id": "test/other"}, {"ready_path": "/v1/models"}, {"hf_mount": "/other"},
])
def test_model_overrides_rejected(changes):
    cfg = config()
    cfg["models"][MODEL].update(changes)
    with pytest.raises(ControllerError):
        validate_tensorfold_plan(cfg, MODEL)


@pytest.mark.parametrize("changes", [
    {"parallel": 17}, {"parallel": True}, {"kv_dtype": "int8"}, {"vision": 1}, {"drafts": "yes"},
    {"vision": False, "vision_urls": True}, {"model_revision": "main"}, {"master_port": 8000},
    {"max_images": 51}, {"max_videos": -1}, {"unknown": True}, {"max_tokens": 0},
])
def test_unsupported_settings_rejected(changes):
    cfg = config()
    cfg["models"][MODEL]["tensorfold"].update(changes)
    with pytest.raises(ControllerError):
        validate_tensorfold_plan(cfg, MODEL)


@pytest.mark.parametrize("rank", [-1, 2])
def test_only_two_ranks(rank):
    with pytest.raises(ControllerError):
        validate_tensorfold_plan(config(), MODEL, rank=rank)


def test_native_health_requires_served_model_and_finished_warming():
    model = config()["models"][MODEL]
    assert tensorfold_native_health_ready(health(), model["served_name"])
    assert tensorfold_model_ready(health(), model)
    for changes in ({"warming": True}, {"model": "other"}, {"status": "loading"}, {"max_batch_size": 0},
                    {"max_batch_size": True}):
        assert not tensorfold_model_ready(health(**changes), model)
    # The Python TensorFold schema never counts as native readiness, nor vice versa.
    python_health = json.dumps({"ok": True, "backend": "tensorfold", "busy": False, "requests_running": 0,
                                "prompt_tokens_total": 0, "completion_tokens_total": 0,
                                "prefill_seconds_total": 0.0, "context_length": 1048576})
    assert not tensorfold_model_ready(python_health, model)
    assert not tensorfold_model_ready(health(), config()["models"]["glm53-tensorfold"])
    assert not tensorfold_model_ready("{" * 100000, model)


def test_exact_rank_preflights_precede_two_node_switch(monkeypatch):
    cfg = config(only=True)
    order = []

    def ssh(scoped, host, script, **kwargs):
        if script.startswith("docker image inspect"):
            order.append(("resolve", host))
            return completed(IMAGE)
        rank = 0 if host == "first" else 1
        command = shlex.split(script.splitlines()[2])
        assert command[command.index("--network") + 1] == "none"
        assert not set(command) & {"--gpus", "--device", "--privileged"}
        args = json.loads(command[command.index("--serve-args-json") + 1])
        assert args == list(validate_tensorfold_plan(scoped, MODEL, rank=rank).argv)
        order.append(("preflight", host))
        return completed("verified")

    monkeypatch.setattr(cli, "ssh_cmd", ssh)
    monkeypatch.setattr(cli, "_bring_up", Mock())
    controller = Mock()
    controller.switch.side_effect = lambda model, start, **kwargs: (order.append(("switch", "both")),
                                                                   start("generation"))
    monkeypatch.setattr(cli, "Controller", lambda *args: controller)
    cli.cmd_up(cfg, MODEL, True, False, False, node="both")
    assert order == [("resolve", "first"), ("resolve", "second"),
                     ("preflight", "first"), ("preflight", "second"), ("switch", "both")]


def test_launch_starts_worker_then_head(monkeypatch):
    cfg = launch_config(config(only=True), MODEL, "both")
    cfg["models"][MODEL]["image"] = IMAGE
    calls = []
    ids = {"first": "c" * 64, "second": "d" * 64}
    monkeypatch.setattr(cli, "probe_v1", lambda *args, **kwargs: "")
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)

    def ssh(scoped, host, script, **kwargs):
        if "/proc/sys/vm/drop_caches" in script:
            return completed("dropped")
        calls.append(host)
        rank = 0 if host == "first" else 1
        assert "docker create --pull never" in script and IMAGE in script
        assert "/opt/spark-serve/qwen38-dual-tensorfold/entrypoint.sh" in script
        assert "SPARK_SERVE_MIN_AVAILABLE_GIB=102" in script
        return completed(f"created {host} rank={rank} id={ids[host]}\nstarted {host} rank={rank} id={ids[host]}\n")

    monkeypatch.setattr(cli, "ssh_cmd", ssh)
    cli._bring_up(cfg, MODEL, True, True, Mock(), generation="generation")
    assert calls == ["second", "first"]


@pytest.mark.parametrize("body,expected", [(health(), True), (health(warming=True), False)])
def test_head_probe_uses_native_health(monkeypatch, body, expected):
    cfg = config(only=True)
    monkeypatch.setattr(cli, "ssh_cmd", Mock(return_value=completed(body)))
    assert cli.probe_ready_path(cfg, cfg["models"][MODEL]) is expected


def test_pins_reuse_qwen_dual_checkpoint_manifest():
    pins, source = prepare.load_pins()
    shared = (ROOT / "recipes/qwen38-dual/model-source.json").read_bytes()
    assert (prepare.RECIPE / "model-source.json").read_bytes() == shared
    assert pins["manifest_sha256"] == hashlib.sha256(shared).hexdigest()
    assert source["revision"] == pins["checkpoint_revision"] == prepare.CHECKPOINT_REVISION
    installed = json.loads((prepare.RECIPE / "runtime/installed-runtime.json").read_text())
    assert len(installed["files"]) == 933 and installed["patch_count"] == 9 == len(installed["patches_sha256"])
    assert installed["patches_hash"] == pins["patches_hash"] == pins["upstream_image_tag"].rsplit("-", 1)[1]
    assert installed["tensorfold_revision"] == pins["tensorfold_revision"]
    assert "native/bin/tensorfold-native" in installed["executables"]
    dockerfile = (prepare.RECIPE / "runtime/Dockerfile").read_text()
    assert dockerfile.startswith("FROM " + pins["base_image"] + "\n")
    assert pins["upstream_revision"] in dockerfile


def test_catalog_validation_rejects_unpinned_verifier():
    pins, source = prepare.load_pins()
    cfg = config()
    assert prepare.validate_catalog(cfg, pins, source) == cfg["models"][MODEL]
    cfg["models"][MODEL]["preflight_args"][3] = "0" * 64
    with pytest.raises((ValueError, ControllerError)):
        prepare.validate_catalog(cfg, pins, source)


@pytest.mark.parametrize("field,value", [("serve_path", "/unpinned"), ("preflight_args", []), ("wrapper", "vllm"),
                                         ("image", "other:tag"), ("nnodes", 1)])
def test_catalog_changes_cannot_bypass_preflight(field, value):
    pins, source = prepare.load_pins()
    cfg = config()
    cfg["models"][MODEL][field] = value
    with pytest.raises((ValueError, ControllerError)):
        prepare.validate_catalog(cfg, pins, source)


def test_build_and_cpu_checks_leave_workloads_untouched():
    pins, _ = prepare.load_pins()
    cfg = config()
    model = cfg["models"][MODEL]
    scripts = [prepare.build_script("/tmp/recipe", pins)]
    for rank in (0, 1):
        plan = validate_tensorfold_plan(cfg, MODEL, rank=rank)
        script = prepare.compatibility_script("/second/cache", model, "sha256:" + "a" * 64, plan.argv)
        argv = shlex.split(script.splitlines()[-1])
        assert json.loads(argv[-1]) == list(plan.argv) and argv[-2] == model["serve_path"]
        assert "--network" in argv and "/second/cache:/cache/huggingface:ro" in argv
        scripts.append(script)
    assert pins["base_image"] in scripts[0] and "--network none --pull=false" in scripts[0]
    assert all(word not in script for script in scripts
               for word in ("--gpus", "docker stop", "docker rm", "systemctl"))


@pytest.mark.parametrize("fail_worker_compatibility", [False, True])
def test_whole_allocation_is_authenticated_before_publishing(monkeypatch, fail_worker_compatibility):
    calls = []
    image = {"image": "x", "image_id": "sha256:" + "a" * 64, "rootfs_layers": ["sha256:" + "b" * 64]}
    catalog = ROOT / "models.example.toml"
    monkeypatch.setattr(sys, "argv", ["prepare", "--catalog", str(catalog), "--skip-download"])
    monkeypatch.setattr(prepare, "load_pins", prepare.load_pins)
    monkeypatch.setattr(prepare, "upload_recipe", lambda cfg, role: "/tmp/recipe-" + role)
    monkeypatch.setattr(prepare, "image_report", lambda *args: image)

    def text(cfg, role, script, timeout=60):
        if script == "id -un":
            return "spider"
        return "SPARK_CHECKPOINT_AUTHENTICATED" if "verify.py" in script else image["image_id"]

    def remote(cfg, role, script):
        calls.append((role, script))
        if fail_worker_compatibility and role == "worker" and "check_compatibility.py" in script:
            raise RuntimeError("worker parser failed")

    monkeypatch.setattr(prepare.dual, "remote_text", text)
    monkeypatch.setattr(prepare.dual, "run_remote", remote)
    monkeypatch.setattr(prepare.subprocess, "run", lambda argv, **kwargs: calls.append(("copy", shlex.join(argv))))
    if fail_worker_compatibility:
        with pytest.raises(RuntimeError, match="worker parser"):
            prepare.main()
        assert not any("prepared.json.tmp" in script for _, script in calls)
    else:
        prepare.main()
        assert [role for role, _ in calls[-2:]] == ["head", "worker"]
        assert all("prepared.json.tmp" in script and "cpu_compatibility_verified" in script
                   for _, script in calls[-2:])
        assert sum("check_compatibility.py" in script for _, script in calls) == 2
    # A matching worker image and authenticated checkpoint need no transfer.
    assert not any(role == "copy" for role, _ in calls)
    assert all(word not in script for _, script in calls for word in ("docker stop", "docker rm", "--gpus"))


def test_runtime_rejects_corruption_extra_files_and_escaping_links(tmp_path):
    root = tmp_path / "opt"
    root.mkdir()
    (root / "native").write_bytes(b"real engine")
    expected = {"native": hashlib.sha256(b"real engine").hexdigest()}
    runtime.verify_files(root, expected)
    (root / "native").write_bytes(b"fake engine")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        runtime.verify_files(root, expected)
    (root / "native").write_bytes(b"real engine")
    (root / "injected").write_bytes(b"extra")
    with pytest.raises(ValueError, match="inventory differs"):
        runtime.verify_files(root, expected)
    (root / "injected").unlink()
    (root / "native").unlink()
    (tmp_path / "outside").write_bytes(b"real engine")
    (root / "native").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="escapes"):
        runtime.verify_files(root, expected)


@pytest.fixture
def snapshot(tmp_path):
    model = tmp_path / "fc694b54fb0174e0913e6adf86691ef85a4ead47"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": "qwen4_exp", "vision_config": {"depth": 1},
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION"},
        "text_config": {"max_position_embeddings": 262144, "num_experts_per_tok": 10}}))
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "mtp.fc_hidden.weight": "a.safetensors", "model.visual.patch.weight": "a.safetensors"}}))
    return model


def fake_native(tmp_path, code, message):
    script = tmp_path / "tensorfold-native"
    script.write_text(f"#!/bin/sh\necho {shlex.quote(message)} >&2\nexit {code}\n")
    script.chmod(0o755)
    return str(script)


@pytest.mark.parametrize("rank", [0, 1])
def test_native_parser_accepts_exact_rank_argv_without_the_model(monkeypatch, snapshot, tmp_path, rank):
    cfg = config()
    argv = list(validate_tensorfold_plan(cfg, MODEL, rank=rank).argv)
    argv[0] = str(snapshot)
    native = fake_native(tmp_path, 1, "/nonexistent is neither a directory nor a Hugging Face repo id")
    monkeypatch.setattr(compatibility, "NATIVE", native)
    calls = []
    real = subprocess.run
    monkeypatch.setattr(compatibility.subprocess, "run", lambda args, **kw: calls.append(args) or real(args, **kw))
    compatibility.verify_model(snapshot, argv)
    assert calls[0][:3] == [native, "serve", compatibility.PROBE] and calls[0][3:] == argv[1:]


@pytest.mark.parametrize("code,message", [(2, "usage: unknown flag --kv-dtype"), (0, "served"),
                                          (1, "CUDA initialization failed")])
def test_native_parser_rejection_fails_compatibility(monkeypatch, snapshot, tmp_path, code, message):
    argv = list(validate_tensorfold_plan(config(), MODEL).argv)
    argv[0] = str(snapshot)
    monkeypatch.setattr(compatibility, "NATIVE", fake_native(tmp_path, code, message))
    with pytest.raises(ValueError, match="rejects rank 0"):
        compatibility.verify_model(snapshot, argv)


@pytest.mark.parametrize("edit", ["model", "rank", "tp", "checkpoint", "mtp"])
def test_metadata_and_argv_identity_are_checked_before_the_parser(monkeypatch, snapshot, tmp_path, edit):
    argv = list(validate_tensorfold_plan(config(), MODEL).argv)
    argv[0] = str(snapshot)
    monkeypatch.setattr(compatibility, "NATIVE", fake_native(tmp_path, 1, "neither a directory"))
    if edit == "model":
        argv[0] = str(tmp_path / "other")
    elif edit == "rank":
        argv[argv.index("--rank") + 1] = "2"
    elif edit == "tp":
        argv[argv.index("--tp") + 1] = "1"
    elif edit == "checkpoint":
        config_path = snapshot / "config.json"
        data = json.loads(config_path.read_text())
        data["text_config"]["num_experts_per_tok"] = 5
        config_path.write_text(json.dumps(data))
    else:
        (snapshot / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
            "model.visual.patch.weight": "a.safetensors"}}))
    with pytest.raises(ValueError):
        compatibility.verify_model(snapshot, argv)


def test_entrypoint_refuses_low_memory_before_engine(tmp_path):
    entrypoint = (prepare.RECIPE / "runtime/entrypoint.sh").read_text()
    assert entrypoint.index("MemAvailable") < entrypoint.index("exec /opt/nvidia/nvidia_entrypoint.sh")
    assert "SPARK_SERVE_MIN_AVAILABLE_GIB:?" in entrypoint
    assert entrypoint.rstrip().endswith('/opt/tensorfold/native/bin/tensorfold-native serve "$@"')
