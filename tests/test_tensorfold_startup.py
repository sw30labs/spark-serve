"""TensorFold uses the controller's normal ownership and preflight guarantees."""
import copy
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import subprocess
from unittest.mock import Mock

import pytest

from spark_serve_nodes import launch_config
from test_tensorfold_backend import config, health


ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("tensorfold_startup_cli", str(ROOT / "spark-serve"))
spec = importlib.util.spec_from_loader(loader.name, loader)
cli = importlib.util.module_from_spec(spec)
loader.exec_module(cli)
MODEL = "qwen38-tensorfold"
IMAGE = "sha256:" + "b" * 64


def completed(output="", code=0):
    return subprocess.CompletedProcess([], code, output, "")


def test_image_is_pinned_before_offline_preflight_and_worker_only_switch(monkeypatch):
    cfg = config(); cfg["models"][MODEL]["image"] = "spark-serve-qwen38-tensorfold:0.1.0"
    order = []
    def ssh(scoped, host, script, **kwargs):
        assert host == "second"
        if script.startswith("docker image inspect"):
            order.append("resolve")
            return completed(IMAGE)
        assert "--network none" in script and "--gpus" not in script
        assert "/second/cache:/cache/huggingface:ro" in script and IMAGE in script
        assert "--serve-args-json" in script and '\"--parallel\",\"4\"' in script
        order.append("preflight")
        return completed("verified")
    monkeypatch.setattr(cli, "ssh_cmd", ssh)
    def launch(scoped, *args, **kwargs):
        assert scoped["models"][MODEL]["image"] == IMAGE
        assert scoped["cluster"]["head"] == "second"
        order.append("launch")
    monkeypatch.setattr(cli, "_bring_up", launch)
    controller = Mock()
    def switch(model, start, **kwargs):
        assert kwargs["hosts"] == ["second"]
        order.append("switch")
        start("generation")
    controller.switch.side_effect = switch
    monkeypatch.setattr(cli, "Controller", lambda *args: controller)
    cli.cmd_up(cfg, MODEL, True, False, False, node="worker")
    assert order == ["resolve", "preflight", "switch", "launch"]
    assert cfg["models"][MODEL]["image"] == "spark-serve-qwen38-tensorfold:0.1.0"


@pytest.mark.parametrize("failure", ["config", "image", "preflight"])
def test_failure_never_constructs_controller_or_stops_existing_model(monkeypatch, failure):
    cfg = config()
    if failure == "config":
        cfg["models"][MODEL]["tensorfold"]["parallel"] = 5
    calls = []
    def ssh(scoped, host, script, **kwargs):
        calls.append(host)
        if script.startswith("docker image inspect"):
            return completed(IMAGE if failure != "image" else "invalid", 0)
        return completed("missing weights", 1)
    monkeypatch.setattr(cli, "ssh_cmd", ssh)
    controller = Mock()
    monkeypatch.setattr(cli, "Controller", controller)
    with pytest.raises(SystemExit):
        cli.cmd_up(cfg, MODEL, True, False, False, node="worker")
    controller.assert_not_called()
    assert calls == ([] if failure == "config" else ["second"] if failure == "image" else ["second", "second"])


@pytest.mark.parametrize("body,served,expected", [
    (health(), "qwen3.8-flash-next-tensorfold", True),
    (health(busy=True, requests_running=1), "qwen3.8-flash-next-tensorfold", True),
    (health(ok=False), "qwen3.8-flash-next-tensorfold", False),
    ('{"ok":true}', "qwen3.8-flash-next-tensorfold", False),
    (health(), "different-model", False),
])
def test_startup_requires_valid_health_exact_model_and_container_checks(monkeypatch, body, served, expected):
    cfg = launch_config(config(), MODEL, "worker")
    monkeypatch.setattr(cli, "ssh_cmd", lambda *args, **kwargs: completed(body))
    monkeypatch.setattr(cli, "probe_v1", lambda *args, **kwargs: json.dumps({"data": [{"id": served}]}))
    ticks = iter([0, 0, 0, 2])
    monkeypatch.setattr(cli.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)
    checked = Mock()
    ready, found, _ = cli.wait_ready(cfg, timeout=1, expect_served="qwen3.8-flash-next-tensorfold",
                                    ready_path="/health", health_model=cfg["models"][MODEL],
                                    on_poll=Mock(), check_started=checked)
    assert ready is expected
    assert found == (served if expected else None)
    assert checked.call_count == (2 if expected else 1)


def test_worker_launch_is_generation_guarded_and_returns_immutable_receipt(monkeypatch):
    cfg = launch_config(config(), MODEL, "worker")
    commands = []
    cid = "c" * 64
    monkeypatch.setattr(cli, "probe_v1", lambda *args, **kwargs: "")
    def ssh(scoped, host, script, **kwargs):
        commands.append((host, script))
        return completed(f"created second rank=0 id={cid}\nstarted second rank=0 id={cid}\n")
    monkeypatch.setattr(cli, "ssh_cmd", ssh)
    result = cli._bring_up(cfg, MODEL, True, True, Mock(), generation="generation")
    assert result == {"containers": [{"host": "second", "rank": 0, "id": cid}], "served": None}
    assert len(commands) == 1 and commands[0][0] == "second"
    assert commands[0][1].startswith("# spark-serve: generation-fence")
    assert "ai.spark-serve.hosts=" in commands[0][1] and IMAGE in commands[0][1]
    assert "docker create --pull never" in commands[0][1]


@pytest.mark.parametrize("healthy,correct_id,expected", [(True, True, True), (False, True, False), (True, False, False)])
def test_status_and_hermes_eligibility_require_same_health_and_owned_identity(monkeypatch, healthy, correct_id, expected):
    cfg = config(); model = cfg["models"][MODEL]
    cid = "c" * 64
    assignments = [{"host": "second", "mode": "vllm", "model": MODEL, "phase": "ready",
                    "generation": "generation", "allocation_id": "generation", "allocation_hosts": ["second"],
                    "legacy": False, "containers": [{"host": "second", "id": cid}]}]
    managed = {"mode": "vllm", "phase": "ready", "nodes": assignments, "yue_workers": [], "transition_error": None}
    monkeypatch.setattr(cli, "Controller", lambda *args: Mock(status=lambda: copy.deepcopy(managed)))
    container = {"name": model["container"], "id": cid if correct_id else "d" * 64, "running": True,
                 "labels": {"ai.spark-serve.model": MODEL, "ai.spark-serve.hosts": '["second"]',
                            "ai.spark-serve.allocation": "generation"}}
    monkeypatch.setattr(cli, "status_host_json", lambda cfg, host: [container] if host == "second" else [])
    monkeypatch.setattr(cli, "probe_v1", lambda cfg: json.dumps({"data": [{"id": model["served_name"]}]})
                        if cfg["cluster"]["head"] == "second" else "")
    monkeypatch.setattr(cli, "ssh_cmd", lambda *args, **kwargs: completed(health(ok=healthy)))
    monkeypatch.setattr(cli, "_active_client_node", lambda cfg, nodes: None)
    observed = cli.collect_status(cfg)["nodes"][1]
    assert observed["ready"] is expected and observed["can_use"] is expected
    assert observed["runtime"] == "tensorfold"


def test_catalog_and_pull_dispatch_expose_tensorfold_as_single_spark(monkeypatch, capsys):
    cfg = config()
    cli.cmd_list_json(cfg)
    item = json.loads(capsys.readouterr().out)[0]
    assert (item["backend"], item["wrapper"], item["topology"]) == ("tensorfold", "tensorfold", "single")
    run = Mock(return_value=completed())
    monkeypatch.setattr(cli.subprocess, "run", run)
    cli.cmd_pull(cfg, MODEL, "worker")
    argv = run.call_args.args[0]
    assert str(ROOT / "tools/prepare_qwen38_tensorfold.py") in argv
    assert argv[-2:] == ["--node", "worker"]


@pytest.mark.parametrize("ready", [False, True])
def test_wait_command_model_listing_also_requires_tensorfold_health(monkeypatch, ready):
    cfg = config()
    monkeypatch.setattr(cli, "probe_v1", lambda *args, **kwargs: json.dumps({"data": [{"id": cfg["models"][MODEL]["served_name"]}]}))
    probe = Mock(return_value=ready)
    monkeypatch.setattr(cli, "probe_ready_path", probe)
    ticks = iter([0, 0, 0, 2])
    monkeypatch.setattr(cli.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)
    result = cli.wait_models(cfg, timeout=1, on_poll=Mock())
    assert result[0] is ready
    probe.assert_called_once_with(cfg, cfg["models"][MODEL])
