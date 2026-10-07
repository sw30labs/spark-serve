"""Two-rank TensorFold startup preserves prepared assets and allocation identity."""
import copy
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import tomllib
from unittest.mock import Mock

import pytest

from spark_serve_nodes import launch_config
from spark_serve_tensorfold import validate_tensorfold_plan


ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("glm_tensorfold_startup_cli", str(ROOT / "spark-serve"))
spec = importlib.util.spec_from_loader(loader.name, loader)
cli = importlib.util.module_from_spec(spec)
loader.exec_module(cli)
MODEL = "glm53-tensorfold"
IMAGE = "sha256:" + "b" * 64


def config():
    cfg = tomllib.loads((ROOT / "models.example.toml").read_text())
    cfg["models"] = {MODEL: cfg["models"][MODEL]}
    cfg["cluster"].update(head="first", worker="second", lan_url="http://first:8000",
                          worker_lan_url="http://second:8000", hf_cache_host="/first/cache",
                          worker_hf_cache_host="/second/cache")
    return cfg


def completed(output="", code=0):
    return subprocess.CompletedProcess([], code, output, "")


def health(context, **changes):
    result = {"ok": True, "backend": "tensorfold", "busy": False, "requests_running": 0,
              "prompt_tokens_total": 0, "completion_tokens_total": 0, "prefill_seconds_total": 0.0,
              "context_length": context}
    result.update(changes)
    return json.dumps(result)


def test_matching_images_and_exact_rank_preflights_precede_two_node_switch(monkeypatch):
    cfg = config()
    original = copy.deepcopy(cfg)
    order = []

    def ssh(scoped, host, script, **kwargs):
        if script.startswith("docker image inspect"):
            order.append(("resolve", host))
            return completed(IMAGE)
        rank = 0 if host == "first" else 1
        assert scoped["models"][MODEL]["image"] == IMAGE
        command = shlex.split(script.splitlines()[2])
        assert "--network" in command and command[command.index("--network") + 1] == "none"
        assert not set(command) & {"--gpus", "--device", "--privileged"}
        assert f"/{host}/cache:/cache/huggingface:ro" in command
        args = json.loads(command[command.index("--serve-args-json") + 1])
        assert args == list(validate_tensorfold_plan(scoped, MODEL, rank=rank).argv)
        assert args[args.index("--rank") + 1] == str(rank)
        order.append(("preflight", host))
        return completed("verified")

    monkeypatch.setattr(cli, "ssh_cmd", ssh)
    launch = Mock()
    monkeypatch.setattr(cli, "_bring_up", launch)
    controller = Mock()

    def switch(model, start, **kwargs):
        assert model == MODEL and kwargs["hosts"] == ["first", "second"]
        order.append(("switch", "both"))
        start("generation")

    controller.switch.side_effect = switch
    monkeypatch.setattr(cli, "Controller", lambda *args: controller)
    cli.cmd_up(cfg, MODEL, True, False, False, node="both")
    assert order == [("resolve", "first"), ("resolve", "second"),
                     ("preflight", "first"), ("preflight", "second"), ("switch", "both")]
    assert launch.call_args.kwargs["generation"] == "generation"
    assert launch.call_args.args[0]["models"][MODEL]["image"] == IMAGE
    assert cfg == original


@pytest.mark.parametrize("failure", ["missing-worker-image", "different-worker-image", "worker-preflight"])
def test_worker_asset_failure_never_constructs_controller(monkeypatch, failure):
    cfg = config()
    calls = []

    def ssh(scoped, host, script, **kwargs):
        action = "resolve" if script.startswith("docker image inspect") else "preflight"
        calls.append((action, host))
        if action == "resolve":
            if host == "second" and failure == "missing-worker-image":
                return completed("", 1)
            if host == "second" and failure == "different-worker-image":
                return completed("sha256:" + "d" * 64)
            return completed(IMAGE)
        return completed("missing checkpoint", 1 if host == "second" else 0)

    monkeypatch.setattr(cli, "ssh_cmd", ssh)
    controller = Mock()
    launch = Mock()
    monkeypatch.setattr(cli, "Controller", controller)
    monkeypatch.setattr(cli, "_bring_up", launch)
    with pytest.raises(SystemExit):
        cli.cmd_up(cfg, MODEL, True, False, False, node="both")
    controller.assert_not_called()
    launch.assert_not_called()
    expected = [("resolve", "first"), ("resolve", "second")]
    if failure == "worker-preflight":
        expected += [("preflight", "first"), ("preflight", "second")]
    assert calls == expected


@pytest.mark.parametrize("node", ["head", "worker"])
def test_two_rank_recipe_rejects_single_node_placement_before_transport(monkeypatch, node):
    ssh = Mock()
    controller = Mock()
    monkeypatch.setattr(cli, "ssh_cmd", ssh)
    monkeypatch.setattr(cli, "Controller", controller)
    with pytest.raises(SystemExit):
        cli.cmd_up(config(), MODEL, True, False, False, node=node)
    ssh.assert_not_called()
    controller.assert_not_called()


def test_launch_starts_worker_then_head_with_guarded_rank_receipts(monkeypatch):
    cfg = launch_config(config(), MODEL, "both")
    cfg["models"][MODEL]["image"] = IMAGE
    calls = []
    ids = {"first": "c" * 64, "second": "d" * 64}
    monkeypatch.setattr(cli, "probe_v1", lambda *args, **kwargs: "")
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)

    def ssh(scoped, host, script, **kwargs):
        calls.append((host, script))
        rank = 0 if host == "first" else 1
        assert script.startswith("# spark-serve: generation-fence")
        assert "docker create --pull never" in script and IMAGE in script
        assert f"/{host}/cache:/cache/huggingface:ro" in script
        assert "/opt/spark-serve/glm53-tensorfold/entrypoint.sh" in script
        assert "ai.spark-serve.allocation=generation" in script
        return completed(f"created {host} rank={rank} id={ids[host]}\nstarted {host} rank={rank} id={ids[host]}\n")

    monkeypatch.setattr(cli, "ssh_cmd", ssh)
    result = cli._bring_up(cfg, MODEL, True, True, Mock(), generation="generation")
    assert [host for host, _ in calls] == ["second", "first"]
    assert result == {"containers": [{"host": "second", "rank": 1, "id": ids["second"]},
                                    {"host": "first", "rank": 0, "id": ids["first"]}], "served": None}


@pytest.mark.parametrize("healthy,worker_identity,expected", [(True, True, True), (False, True, False),
                                                              (True, False, False)])
def test_distributed_readiness_uses_head_health_and_both_container_identities(monkeypatch, healthy, worker_identity, expected):
    cfg = config()
    model = cfg["models"][MODEL]
    hosts = ["first", "second"]
    ids = {"first": "c" * 64, "second": "d" * 64}
    assignments = [{"host": host, "mode": "vllm", "model": MODEL, "phase": "ready",
                    "generation": "generation", "allocation_id": "generation", "allocation_hosts": hosts,
                    "legacy": False, "containers": [{"host": host, "id": ids[host]}]} for host in hosts]
    managed = {"mode": "vllm", "phase": "ready", "nodes": assignments, "yue_workers": [], "transition_error": None}
    monkeypatch.setattr(cli, "Controller", lambda *args: Mock(status=lambda: copy.deepcopy(managed)))

    def containers(scoped, host):
        cid = "e" * 64 if host == "second" and not worker_identity else ids[host]
        return [{"name": model["container"], "id": cid, "running": True,
                 "labels": {"ai.spark-serve.model": MODEL, "ai.spark-serve.hosts": json.dumps(hosts),
                            "ai.spark-serve.allocation": "generation"}}]

    monkeypatch.setattr(cli, "status_host_json", containers)
    monkeypatch.setattr(cli, "probe_v1", lambda scoped: json.dumps({"data": [{"id": model["served_name"]}]})
                        if scoped["cluster"]["head"] == "first" else "")
    readiness = Mock(return_value=completed(health(model["max_model_len"], ok=healthy)))
    monkeypatch.setattr(cli, "ssh_cmd", readiness)
    monkeypatch.setattr(cli, "_active_client_node", lambda *args: None)
    nodes = cli.collect_status(cfg)["nodes"]
    assert [node["ready"] for node in nodes] == [expected, expected]
    assert [node["can_use"] for node in nodes] == [expected, False]
    assert all(node["runtime"] == "tensorfold" for node in nodes)
    assert readiness.call_count == 1 and readiness.call_args.args[1] == "first"


def test_catalog_exposes_tensorfold_as_distributed(monkeypatch, capsys):
    cli.cmd_list_json(config())
    item = json.loads(capsys.readouterr().out)[0]
    assert (item["id"], item["backend"], item["wrapper"], item["topology"]) == (
        MODEL, "tensorfold", "tensorfold", "distributed")


@pytest.mark.parametrize("recipe,script", [("glm53-tensorfold", "prepare_glm53_tensorfold.py"),
                                           ("qwen38-dual", "prepare_qwen38_dual.py")])
def test_new_two_node_recipe_pull_dispatch(monkeypatch, recipe, script):
    cfg = {"cluster": {"head": "first", "worker": "second", "nnodes": 2},
           "models": {recipe: {"recipe": recipe, "nnodes": 2}}}
    run = Mock(return_value=completed())
    monkeypatch.setattr(cli.subprocess, "run", run)
    cli.cmd_pull(cfg, recipe, "both")
    assert str(ROOT / "tools" / script) in run.call_args.args[0]
    assert run.call_args.args[0][-2:] == ["--node", "both"]
