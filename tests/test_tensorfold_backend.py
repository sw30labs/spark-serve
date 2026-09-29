"""TensorFold's launch contract is separate from the vLLM cluster flags."""
import copy
import json

import pytest

from spark_serve_controller import ControllerError
from spark_serve_nodes import launch_config
from spark_serve_tensorfold import tensorfold_docker_argv, tensorfold_health_ready, validate_tensorfold_plan


def config():
    revision = "dadefa8066e3be900a0d148d0f5a2f4eb1cf6534"
    return {
        "cluster": {"head": "first", "worker": "second", "nnodes": 2, "port": 8000,
                    "lan_url": "http://first:8000", "worker_lan_url": "http://second:8000",
                    "hf_cache_host": "/first/cache", "worker_hf_cache_host": "/second/cache",
                    "nccl": {"NCCL_NET": "IB", "GLOO_SOCKET_IFNAME": "other-link"}},
        "models": {"qwen38-tensorfold": {
            "wrapper": "tensorfold", "recipe": "qwen38-tensorfold", "nnodes": 1, "tensor_parallel": 1,
            "image": "sha256:" + "b" * 64, "container": "tensorfold_qwen38",
            "hf_id": "Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP", "hf_mount": "/cache/huggingface",
            "serve_path": "/cache/huggingface/hub/models--Vontra--Qwen3.8-Flash-Next-MLX-4bit-MTP/snapshots/" + revision,
            "served_name": "qwen3.8-flash-next-tensorfold", "max_model_len": 262144, "max_num_seqs": 4,
            "ready_path": "/health", "drop_caches": False, "preflight_args": ["/opt/spark-serve/qwen38-tensorfold/verify.py",
                               "/cache/huggingface/hub/models--Vontra--Qwen3.8-Flash-Next-MLX-4bit-MTP/snapshots/" + revision,
                               "--expected-manifest-sha256", "583d1e6bd1992016e37d9c86c18bbd16a2f08ecca5da221f335e36a7e662f34a",
                               "--verify-runtime"],
            "tensorfold": {"model_revision": revision, "parallel": 4, "kv_dtype": "int8", "vision": True,
                           "ple_on_ssd": True, "mtp_drafts": 6, "mtp_confidence": 0.60},
        }},
    }


def health(**changes):
    result = {"ok": True, "backend": "tensorfold", "busy": False, "requests_running": 0,
              "prompt_tokens_total": 0, "completion_tokens_total": 0, "prefill_seconds_total": 0.0,
              "context_length": 262144}
    result.update(changes)
    return json.dumps(result)


def test_worker_launch_keeps_weights_readonly_and_identity_local():
    cfg = config()
    before = copy.deepcopy(cfg)
    plan = validate_tensorfold_plan(launch_config(cfg, "qwen38-tensorfold", "worker"), "qwen38-tensorfold")
    argv = tensorfold_docker_argv(plan, "generation")
    assert cfg == before
    assert argv[:4] == ["docker", "create", "--pull", "never"]
    assert "ai.spark-serve.hosts=[\"second\"]" in argv
    assert "ai.spark-serve.allocation=generation" in argv
    assert "/second/cache:/cache/huggingface:ro" in argv
    assert "/second/cache/spark-serve/qwen38-tensorfold/runtime-cache:/cache" in argv
    assert not any("/first/" in item for item in argv)
    assert argv[argv.index(plan.image) + 1:] == list(plan.argv)
    for flag, value in (("--parallel", "4"), ("--kv-dtype", "int8"), ("--context", "262144"),
                        ("--name", "qwen3.8-flash-next-tensorfold"), ("--port", "8000"), ("--backend", "cuda")):
        assert plan.argv[plan.argv.index(flag) + 1] == value
    assert "--ple-on-ssd" in plan.argv and "--vision" in plan.argv
    assert not set(argv) & {"--privileged", "--device", "--nnodes", "--tensor-parallel-size", "--master-addr"}
    assert not any(item.startswith(("NCCL_", "GLOO_", "VLLM_")) for item in argv)
    assert "HF_HUB_OFFLINE=1" in argv and "TENSORFOLD_NO_UPDATE_CHECK=1" in argv


def test_mutable_tag_can_be_previewed_but_not_started():
    cfg = config(); cfg["models"]["qwen38-tensorfold"]["image"] = "spark-serve-qwen38-tensorfold:0.1.0"
    plan = validate_tensorfold_plan(cfg, "qwen38-tensorfold")
    assert plan.image in tensorfold_docker_argv(plan)
    with pytest.raises(ControllerError, match="immutable image"):
        tensorfold_docker_argv(plan, "generation")


@pytest.mark.parametrize("change", [
    {"preflight_args": ["-c", "pass"]}, {"hf_id": "other/checkpoint"}, {"nnodes": 2}, {"nnodes": True}, {"tensor_parallel": 2}, {"ready_path": "/v1/models"},
    {"vllm": {"args": ["--parallel", "5"]}}, {"env": {"NCCL_NET": "Socket"}},
    {"docker_extra": ["--privileged"]}, {"mounts": [{"head": "/", "container": "/host"}]},
    {"max_model_len": 262145}, {"max_num_seqs": 5}, {"hf_mount": "/cache"},
    {"serve_path": "/cache/huggingface/hub/main"}, {"image": "bad;image"}, {"container": "bad name"},
])
def test_unsupported_launch_inputs_fail_before_execution(change):
    cfg = config(); cfg["models"]["qwen38-tensorfold"].update(change)
    with pytest.raises(ControllerError):
        validate_tensorfold_plan(cfg, "qwen38-tensorfold")


@pytest.mark.parametrize("change", [
    {"parallel": 5}, {"parallel": True}, {"kv_dtype": "bf16"}, {"model_revision": "main"},
    {"mtp_confidence": float("nan")}, {"mtp_confidence": 1.1}, {"vision": "yes"},
    {"ple_on_ssd": False}, {"args": ["--parallel", "5"]}, {"parallel": 3},
])
def test_unsupported_runtime_settings_are_not_silently_ignored(change):
    cfg = config(); cfg["models"]["qwen38-tensorfold"]["tensorfold"].update(change)
    with pytest.raises(ControllerError):
        validate_tensorfold_plan(cfg, "qwen38-tensorfold")


def test_busy_cuda_health_is_ready_and_context_must_match():
    assert tensorfold_health_ready(health(), 262144)
    assert tensorfold_health_ready(health(busy=True, requests_running=4), 262144)
    assert not tensorfold_health_ready(health(), 131072)


@pytest.mark.parametrize("body", ["garbage", "[]", "null", "{}", '{"ok": true}',
    health(ok=False), health(ok=1), health(backend="vllm"), health(busy="false"), health(busy=True),
    health(requests_running=-1), health(requests_running=True), health(prompt_tokens_total=None),
    health(completion_tokens_total=1.1), health(prefill_seconds_total=float("nan")),
    health(context_length=None), health(context_length=True), health(context_length=262144.0),
    health(prefill_seconds_total=10**400), " " * 65537, "[" * 2000 + "]" * 2000,
])
def test_malformed_or_unpatched_health_is_not_ready(body):
    assert not tensorfold_health_ready(body, 262144)
