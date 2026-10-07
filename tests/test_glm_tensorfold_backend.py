"""Validate GLM's two-rank launch contract before transport or model switching."""
import copy
from pathlib import Path
import tomllib

import pytest

from spark_serve_controller import ControllerError
from spark_serve_tensorfold import tensorfold_docker_argv, validate_tensorfold_plan


def config():
    cfg = tomllib.loads((Path(__file__).resolve().parents[1] / "models.example.toml").read_text())
    cfg["cluster"].update(head="first", worker="second", hf_cache_host="/first/cache",
                          worker_hf_cache_host="/second/cache")
    cfg["models"]["glm53-tensorfold"]["image"] = "sha256:" + "b" * 64
    return cfg


def test_rank_launches_share_allocation_but_use_local_cache_and_head_only_api():
    cfg = config()
    before = copy.deepcopy(cfg)
    for rank, host in enumerate(("first", "second")):
        plan = validate_tensorfold_plan(cfg, "glm53-tensorfold", rank=rank)
        argv = tensorfold_docker_argv(plan, "generation")
        assert plan.host == host and plan.rank == rank
        assert f"/{host}/cache:/cache/huggingface:ro" in argv
        assert f"/{host}/cache/spark-serve/glm53-tensorfold/runtime-cache:/cache" in argv
        assert 'ai.spark-serve.hosts=["first","second"]' in argv
        assert "ai.spark-serve.allocation=generation" in argv
        assert "--device" in argv and "/dev/infiniband" in argv
        assert "--privileged" not in argv
        assert plan.argv[plan.argv.index("--rank") + 1] == str(rank)
        assert plan.argv[plan.argv.index("--tp") + 1] == "2"
        assert ("--port" in plan.argv) == (rank == 0)
        assert ("--name" in plan.argv) == (rank == 0)
        assert "--vision" in plan.argv and "--vision-urls" not in plan.argv
        assert "NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1" in argv
        assert "TF_GLM_DENSE=q4" in argv and "TF_GLM_KV=fp8" in argv
        assert not any(item.startswith(("VLLM_", "GLOO_", "NCCL_NET=")) for item in argv)
        assert not any(item.startswith("TENSORFOLD_MTP_COPY=") for item in argv)
        assert argv[argv.index(plan.image) + 1:] == list(plan.argv)
    assert cfg == before


def test_eight_stream_reserve_and_verify_window_track_upstream_formula():
    cfg = config()
    cfg["models"]["glm53-tensorfold"]["tensorfold"]["parallel"] = 8
    cfg["models"]["glm53-tensorfold"]["max_num_seqs"] = 8
    env = dict(validate_tensorfold_plan(cfg, "glm53-tensorfold").environment)
    assert env["TF_GLM_MULTI_WINDOW"] == "64"
    assert env["TF_ROCE_MAX_KB"] == "1024"
    assert env["TENSORFOLD_MEMORY_RESERVE_GIB"] == "19.6"


def test_mtp_selects_target_head_and_single_stream():
    cfg = config()
    model = cfg["models"]["glm53-tensorfold"]
    model["tensorfold"].update(drafter="mtp", parallel=1)
    model["max_num_seqs"] = 1
    plan = validate_tensorfold_plan(cfg, "glm53-tensorfold")
    assert plan.argv[plan.argv.index("--drafter") + 1] == "none"
    assert plan.argv[plan.argv.index("--parallel") + 1] == "1"


@pytest.mark.parametrize("changes", [
    {"nnodes": 1}, {"tensor_parallel": 3}, {"max_model_len": 1048577},
    {"max_num_seqs": 8}, {"preflight_args": ["-c", "pass"]},
    {"serve_path": "/cache/huggingface/main"}, {"env": {"TF_GLM_KV": "bf16"}},
    {"docker_extra": ["--privileged"]}, {"hf_id": "test/other"},
])
def test_model_overrides_rejected(changes):
    cfg = config()
    cfg["models"]["glm53-tensorfold"].update(changes)
    with pytest.raises(ControllerError):
        validate_tensorfold_plan(cfg, "glm53-tensorfold")


@pytest.mark.parametrize("changes", [
    {"parallel": 9}, {"parallel": True}, {"kv_dtype": "bf16"}, {"dense": "bf16"},
    {"drafter": "mtp"}, {"drafter": "unknown"}, {"communication": "socket"},
    {"vision": 1}, {"vision": False, "vision_urls": True}, {"vision_urls": "true"},
    {"model_revision": "main"}, {"draft_revision": "main"}, {"master_port": 0}, {"master_port": 8000},
    {"unknown": True}, {"temperature": float("nan")},
])
def test_unprepared_or_unsupported_settings_rejected(changes):
    cfg = config()
    cfg["models"]["glm53-tensorfold"]["tensorfold"].update(changes)
    with pytest.raises(ControllerError):
        validate_tensorfold_plan(cfg, "glm53-tensorfold")


@pytest.mark.parametrize("master", ["127.0.0.1", "0.0.0.0", "224.0.0.1", "bad\naddress"])
def test_rendezvous_must_be_reachable(master):
    cfg = config()
    cfg["cluster"]["master_addr"] = master
    with pytest.raises(ControllerError):
        validate_tensorfold_plan(cfg, "glm53-tensorfold")
