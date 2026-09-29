"""NIM launch plans must fail before a controller is allowed to stop anything."""
from dataclasses import FrozenInstanceError
import json

import pytest

from spark_serve_controller import ControllerError
from spark_serve_nim import nim_docker_argv, parse_nim_primary_node, validate_nim_plan


def config():
    return {
        "cluster": {
            "head": "first", "worker": "second", "port": 8000,
            "hf_cache_host": "/first/cache", "worker_hf_cache_host": "/second/cache",
            "nccl": {"NCCL_NET": "IB", "UCX_NET_DEVICES": "old-rail", "GLOO_SOCKET_IFNAME": "old-nic"},
        },
        "models": {"glm53": {
            "wrapper": "nim", "nnodes": 2, "tensor_parallel": 2,
            "image": "nvcr.io/nim/zai-org/glm-5.3-flash@sha256:" + "a" * 64,
            "container": "nim_glm53", "served_name": "glm-5.3-flash", "max_model_len": 131072,
            "ready_path": "/v1/health/ready",
            "hf_mount": "/cache/huggingface",
            "serve_path": "/cache/huggingface/spark-serve/glm53-nvfp4/models/nim-aa28e1f-nvfp4",
            "nim": {"model_source": "ngc", "model_revision": "nim-aa28e1f-nvfp4"},
            "env": {"NIM_LOG_LEVEL": "INFO"},
        }},
    }


def environment(argv):
    return dict(value.split("=", 1) for flag, value in zip(argv, argv[1:]) if flag == "-e")


def mounts(argv):
    return [value for flag, value in zip(argv, argv[1:]) if flag == "-v"]


def test_offline_plan_is_frozen_and_detached_from_mutable_catalog():
    cfg = config()
    plan = validate_nim_plan(cfg, "glm53")
    cfg["models"]["glm53"]["env"]["NIM_LOG_LEVEL"] = "DEBUG"
    cfg["cluster"]["head"] = "unexpected"
    assert plan.hosts == ("first", "second")
    assert dict(plan.env)["NIM_LOG_LEVEL"] == "INFO"
    with pytest.raises(FrozenInstanceError):
        plan.image = "mutable:latest"


@pytest.mark.parametrize("path", [None, "", "/v1/models", "/v1/health/live", "/health", "/v1/health/ready/"])
def test_model_listing_or_liveness_cannot_replace_engine_readiness(path):
    cfg = config()
    if path is None:
        cfg["models"]["glm53"].pop("ready_path")
    else:
        cfg["models"]["glm53"]["ready_path"] = path
    with pytest.raises(ControllerError, match="ready_path must be /v1/health/ready"):
        validate_nim_plan(cfg, "glm53")


def test_rank_zero_uses_official_hardware_contract_and_no_inherited_transport():
    plan = validate_nim_plan(config(), "glm53")
    argv = nim_docker_argv(plan, 0, "allocation-one")
    env = environment(argv)
    assert argv[:4] == ["docker", "create", "--pull", "never"]
    assert argv[-1] == plan.image
    assert "--privileged" not in argv and "--ipc" not in argv
    assert "/dev/infiniband:/dev/infiniband:rwm" in argv and "memlock=-1:-1" in argv
    assert env["NIM_SERVER_PORT"] == "8000" and env["NIM_NODE_MANAGER_PORT"] == "20000"
    assert env["NIM_MODEL_PATH"] == plan.serve_path
    assert env["NIM_CACHE_PATH"] == "/opt/nim/.cache"
    assert env["NIM_SERVED_MODEL_NAME"] == "glm-5.3-flash"
    assert env["NIM_MAX_MODEL_LEN"] == "131072"
    assert "NIM_PRIMARY_NODE" not in env
    assert not any(key.startswith(("NCCL_", "UCX_", "GLOO_", "TP_")) for key in env)
    assert mounts(argv) == ["/first/cache:/cache/huggingface:ro",
                            "/first/cache/spark-serve/glm53-nvfp4/runtime-cache:/opt/nim/.cache"]
    assert "ai.spark-serve.model=glm53" in argv
    assert "ai.spark-serve.allocation=allocation-one" in argv
    assert "ai.spark-serve.hosts=" + json.dumps(["first", "second"], separators=(",", ":")) in argv


def test_rank_one_requires_discovered_primary_and_uses_worker_storage():
    plan = validate_nim_plan(config(), "glm53")
    with pytest.raises(ControllerError, match="IP address"):
        nim_docker_argv(plan, 1, "allocation-one")
    primary = parse_nim_primary_node("INFO start worker node with NIM_PRIMARY_NODE=192.168.1.10 NIM_NODE_MANAGER_PORT=20000")
    argv = nim_docker_argv(plan, 1, "allocation-one", primary_node=primary)
    assert environment(argv)["NIM_PRIMARY_NODE"] == "192.168.1.10"
    assert environment(argv)["NIM_SERVER_PORT"] == "8002"
    assert "/second/cache:/cache/huggingface:ro" in mounts(argv)
    assert "/second/cache/spark-serve/glm53-nvfp4/runtime-cache:/opt/nim/.cache" in mounts(argv)
    with pytest.raises(ControllerError, match="discover its own"):
        nim_docker_argv(plan, 0, "allocation-one", primary_node=primary)


def test_hf_snapshots_keep_full_cache_mount_so_blob_symlinks_resolve():
    cfg = config(); model = cfg["models"]["glm53"]
    revision = "b" * 40
    model.update(hf_id="nvidia/GLM-5.3-Flash-NVFP4", serve_path="/cache/huggingface/hub/models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/" + revision)
    model["nim"].update(model_source="hf", model_revision=revision)
    plan = validate_nim_plan(cfg, "glm53")
    argv = nim_docker_argv(plan, 0, "allocation-one")
    assert environment(argv)["NIM_MODEL_PATH"] == model["serve_path"]
    assert "/first/cache:/cache/huggingface:ro" in mounts(argv)
    assert not any("snapshots" in mount for mount in mounts(argv))
    model["serve_path"] = model["serve_path"].replace(revision, "c" * 40)
    with pytest.raises(ControllerError, match="pinned HF snapshot"):
        validate_nim_plan(cfg, "glm53")


@pytest.mark.parametrize("key,value", [
    ("image", "nvcr.io/nim/zai-org/glm-5.3-flash:latest"),
    ("image", "invalid;repo@sha256:" + "a" * 64),
    ("nnodes", 1), ("nnodes", True), ("tensor_parallel", 1),
    ("serve_path", "/elsewhere/models/nim-aa28e1f-nvfp4"),
    ("serve_path", "/cache/huggingface/../models/nim-aa28e1f-nvfp4"),
    ("serve_path", "/cache/huggingface/models/unpinned"),
    ("hf_mount", "/"), ("max_model_len", 0), ("max_model_len", True),
    ("docker_extra", ["--privileged"]), ("vllm", {"args": ["--max-model-len", "999"]}),
    ("vllm", "invalid"), ("shm_size", "-1g"), ("container", "bad name"),
])
def test_invalid_static_launch_input_is_rejected(key, value):
    cfg = config(); cfg["models"]["glm53"][key] = value
    with pytest.raises(ControllerError):
        validate_nim_plan(cfg, "glm53")


@pytest.mark.parametrize("key,value", [
    ("model_revision", "latest"), ("model_source", "unknown"),
    ("manager_port", 8000), ("worker_port", 20000), ("worker_port", 65536),
    ("handshake_timeout", 0), ("handshake_timeout", True),
    ("runtime_cache_head", "/first/cache"),
    ("runtime_cache_head", "/first/cache/hub/writable"),
    ("runtime_cache_head", "/first/cache/spark-serve/glm53-nvfp4/models"),
    ("ffmpeg_path_head", "/opt/ffmpeg8"), ("unrecognised", "typo"),
])
def test_invalid_nim_settings_fail_closed(key, value):
    cfg = config(); cfg["models"]["glm53"]["nim"][key] = value
    with pytest.raises(ControllerError):
        validate_nim_plan(cfg, "glm53")


@pytest.mark.parametrize("key,value", [
    ("NCCL_NET", "Socket"), ("UCX_NET_DEVICES", "wrong"), ("GLOO_SOCKET_IFNAME", "wrong"),
    ("NIM_PRIMARY_NODE", "192.168.1.10"), ("NIM_SERVER_PORT", "8009"),
    ("NIM_MODEL_PATH", "/wrong"), ("NIM_CACHE_PATH", "/weights"),
    ("NIM_PASSTHROUGH_ARGS", "--tp-size 1 --context-length 4096"),
    ("NIM_TENSOR_PARALLEL_SIZE", 1), ("NIM_PIPELINE_PARALLEL_SIZE", 2),
    ("NIM_LOG_LEVEL", "WARNING"),
    ("NIM_KV_CACHE_PERCENT", "92"), ("NIM_KV_CACHE_PERCENT", "nan"),
    ("NIM_KV_CACHE_PERCENT", "inf"), ("NIM_KV_CACHE_PERCENT", "invalid"),
    ("NIM_KV_CACHE_PERCENT", 0), ("NIM_KVCACHE_PERCENT", 1),
    ("NGC_API_KEY", "do-not-put-secrets-in-catalog"),
    ("NIM_SERVED_MODEL_NAME", "wrong-model"), ("NIM_MAX_MODEL_LEN", 1048576),
    ("NIM_LOG_LEVEL", "INFO\nBAD"), ("NIM_LOG_LEVEL", {"bad": "table"}),
])
def test_environment_cannot_override_identity_transport_or_embed_credentials(key, value):
    cfg = config(); cfg["models"]["glm53"]["env"][key] = value
    with pytest.raises(ControllerError):
        validate_nim_plan(cfg, "glm53")


def test_optional_ffmpeg_is_read_only_and_must_exist_on_each_rank():
    cfg = config()
    cfg["models"]["glm53"]["nim"].update(ffmpeg_path_head="/first/ffmpeg8", ffmpeg_path_worker="/second/ffmpeg8")
    plan = validate_nim_plan(cfg, "glm53")
    assert "/first/ffmpeg8:/opt/ffmpeg8:ro" in mounts(nim_docker_argv(plan, 0, "allocation-one"))
    assert "/second/ffmpeg8:/opt/ffmpeg8:ro" in mounts(nim_docker_argv(plan, 1, "allocation-one", "192.168.1.10"))


def test_memory_budget_override_reaches_both_ranks_without_changing_context():
    cfg = config()
    cfg["models"]["glm53"]["env"]["NIM_KV_CACHE_PERCENT"] = "0.92"
    plan = validate_nim_plan(cfg, "glm53")
    for rank, primary in ((0, None), (1, "192.168.1.10")):
        env = environment(nim_docker_argv(plan, rank, "allocation", primary))
        assert env["NIM_KV_CACHE_PERCENT"] == "0.92"
        assert env["NIM_MAX_MODEL_LEN"] == "131072"


def test_primary_advertisement_accepts_repeat_lines_color_and_ipv6():
    assert parse_nim_primary_node("still starting") is None
    assert parse_nim_primary_node("\x1b[32mINFO NIM_PRIMARY_NODE=192.168.1.10\x1b[0m\nINFO NIM_PRIMARY_NODE=192.168.1.10") == "192.168.1.10"
    assert parse_nim_primary_node("INFO NIM_PRIMARY_NODE=fd00::10") == "fd00::10"


@pytest.mark.parametrize("value", ["localhost", "127.0.0.1", "0.0.0.0", "224.0.0.1", "169.254.1.1", "192.168.1.10;touch", "${HOST}"])
def test_primary_advertisement_rejects_non_remote_and_non_ip_values(value):
    with pytest.raises(ControllerError):
        parse_nim_primary_node("INFO NIM_PRIMARY_NODE=" + value)


def test_primary_advertisement_rejects_conflicting_addresses():
    with pytest.raises(ControllerError, match="conflicting"):
        parse_nim_primary_node("NIM_PRIMARY_NODE=192.168.1.10\nNIM_PRIMARY_NODE=192.168.1.11")


@pytest.mark.parametrize("rank", [-1, 2, True])
def test_invalid_rank_is_rejected(rank):
    with pytest.raises(ControllerError, match="rank"):
        nim_docker_argv(validate_nim_plan(config(), "glm53"), rank, "allocation-one")
