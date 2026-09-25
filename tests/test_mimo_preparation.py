"""Offline checks for the MiMo-V2.6-Flash-RL catalog and snapshot helpers."""
import importlib.machinery
import importlib.util
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]
RECIPE = ROOT / "recipes" / "mimo-v26-flash"


def _load(name: str):
    path = RECIPE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"mimo_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


jsonfix = _load("jsonfix")
verify = _load("verify")


def _cli():
    loader = importlib.machinery.SourceFileLoader("spark_serve_mimo_test", str(ROOT / "spark-serve"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_repair_strips_a_trailing_comma_and_leaves_valid_json_untouched():
    broken = '{"block_size": 8,}\n'
    assert json.loads(jsonfix.repair_json_text(broken))["block_size"] == 8
    valid = '{"block_size": 8}\n'
    assert jsonfix.repair_json_text(valid) == valid


def test_inventory_ignores_hub_cache(tmp_path):
    root = tmp_path / "snapshot"
    (root / "dflash").mkdir(parents=True)
    (root / "config.json").write_text("{}\n")
    (root / "dflash" / "config.json").write_text("{}\n")
    (root / ".cache" / "huggingface").mkdir(parents=True)
    (root / ".cache" / "huggingface" / "download.json").write_bytes(b"x" * 1000)
    got = verify.inventory(root)
    assert got["files"] == 2
    assert got["bytes"] == len("{}\n") * 2


def test_example_catalog_keeps_the_measured_tp2_launch():
    cfg = tomllib.loads((ROOT / "models.example.toml").read_text())
    model = cfg["models"]["mimo26"]
    script = _cli().docker_run_script(cfg, model, 1, True)
    assert "vllm_mimo" in script
    assert "--moe-backend marlin" in script
    assert "--no-async-scheduling" in script
    assert "--kv-cache-dtype fp8" in script
    assert "--generation-config vllm" in script
    assert "repetition_penalty" in script
    assert '"enable_thinking":false' in script
    assert "max_new_tokens" not in script
    assert '"num_speculative_tokens":7' in script or '"num_speculative_tokens": 7' in script
    assert "--headless" in script
    assert "--node-rank 1" in script
    assert "--memory 112g" in script
    assert "NCCL_IB_HCA==rocep1s0f1,roceP2p1s0f1" in script or "NCCL_IB_HCA='=rocep1s0f1,roceP2p1s0f1'" in script or "NCCL_IB_HCA=\"=rocep1s0f1,roceP2p1s0f1\"" in script
    assert "VLLM_USE_DEEP_GEMM=0" in script
    assert model["max_model_len"] == 300000
    assert model["nnodes"] == 2
