"""Preparation checks must fail before touching an existing serving workload."""
import importlib.machinery
import importlib.util
import shlex
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("spark_serve_preflight_test", str(ROOT / "spark-serve"))
spec = importlib.util.spec_from_loader(loader.name, loader)
cli = importlib.util.module_from_spec(spec)
loader.exec_module(cli)


def config(nnodes=1):
    return {
        "cluster": {"head": "head", "worker": "worker", "nnodes": 2,
                    "hf_cache_host": "/home/test/cache", "lan_url": "http://head:8000"},
        "models": {"qwen38": {"nnodes": nnodes, "image": "qwen38:local", "served_name": "qwen",
                              "hf_mount": "/cache/huggingface", "ready_timeout": 3600,
                              "drop_caches": False,
                              "preflight_args": ["/opt/spark-serve/qwen38/verify.py", "/cache/huggingface/qwen38"]}},
    }


def completed(code=0, output="verified", error=""):
    return subprocess.CompletedProcess([], code, output, error)


@pytest.mark.parametrize("nnodes,hosts", [(1, ["head"]), (2, ["head", "worker"])])
def test_preflight_is_cpu_only_readonly_offline_and_runs_only_required_hosts(monkeypatch, nnodes, hosts):
    cfg = config(nnodes)
    cfg["models"]["qwen38"]["docker_extra"] = ["--privileged", "--gpus", "all"]
    cfg["models"]["qwen38"]["mounts"] = [{"head": "/head/extra", "worker": "/worker/extra", "container": "/extra"}]
    ssh = Mock(return_value=completed())
    monkeypatch.setattr(cli, "ssh_cmd", ssh)
    sink = Mock()
    cli.model_preflight(cfg, "qwen38", cfg["models"]["qwen38"], sink)
    assert [call.args[1] for call in ssh.call_args_list] == hosts
    for call in ssh.call_args_list:
        script = call.args[2]
        assert "docker image inspect" in script
        assert '"$image_id"' in script
        command = shlex.split(script.splitlines()[2])
        assert command[command.index("--pull") + 1] == "never"
        assert command[command.index("--network") + 1] == "none"
        assert command[command.index("--entrypoint") + 1] == "python3"
        assert "NVIDIA_VISIBLE_DEVICES=void" in command
        assert "/home/test/cache:/cache/huggingface:ro" in command
        assert f"/{call.args[1]}/extra:/extra:ro" in command
        assert "--gpus" not in command and "--privileged" not in command and "--device" not in command
        assert call.kwargs["timeout"] == 145
    assert [call.kwargs["detail"] for call in sink.event.call_args_list] == ["checking local image and prepared weights", "passed"] * nnodes


@pytest.mark.parametrize("code,error", [(1, "No such image"), (2, "checkpoint manifest missing"), (124, "SSH timed out")])
def test_failed_preflight_never_constructs_controller_or_starts_gpu(monkeypatch, capsys, code, error):
    monkeypatch.setattr(cli, "ssh_cmd", Mock(return_value=completed(code, "", error)))
    controller = Mock()
    bring_up = Mock()
    monkeypatch.setattr(cli, "Controller", controller)
    monkeypatch.setattr(cli, "_bring_up", bring_up)
    with pytest.raises(SystemExit, match="1"):
        cli.cmd_up(config(), "qwen38", True, False, True)
    controller.assert_not_called()
    bring_up.assert_not_called()
    output = capsys.readouterr().out
    assert "current workload was not changed" in output
    assert error in output


def test_worker_failure_also_prevents_switch(monkeypatch):
    monkeypatch.setattr(cli, "ssh_cmd", Mock(side_effect=[completed(), completed(1, "", "worker missing assets")]))
    controller = Mock()
    monkeypatch.setattr(cli, "Controller", controller)
    with pytest.raises(SystemExit):
        cli.cmd_up(config(2), "qwen38", True, False, False)
    controller.assert_not_called()


def test_successful_preflight_precedes_controller_switch(monkeypatch):
    order = []
    monkeypatch.setattr(cli, "ssh_cmd", lambda *a, **k: order.append("preflight") or completed())
    controller = Mock()
    controller.switch.side_effect = lambda *a, **k: order.append("switch")
    monkeypatch.setattr(cli, "Controller", lambda *a: order.append("controller") or controller)
    cli.cmd_up(config(2), "qwen38", True, False, False)
    assert order == ["preflight", "preflight", "controller", "switch"]


def test_unconfigured_models_skip_preflight(monkeypatch):
    cfg = config()
    cfg["models"]["qwen38"].pop("preflight_args")
    ssh = Mock()
    monkeypatch.setattr(cli, "ssh_cmd", ssh)
    controller = Mock()
    monkeypatch.setattr(cli, "Controller", controller)
    cli.cmd_up(cfg, "qwen38", True, False, False)
    ssh.assert_not_called()
    controller.return_value.switch.assert_called_once()


@pytest.mark.parametrize("field,value", [("preflight_args", []), ("preflight_args", "verify.py"),
                                        ("ready_timeout", 0), ("ready_timeout", True), ("ready_timeout", "3600")])
def test_invalid_configuration_fails_before_switch(monkeypatch, field, value):
    cfg = config()
    cfg["models"]["qwen38"][field] = value
    controller = Mock()
    monkeypatch.setattr(cli, "Controller", controller)
    with pytest.raises(SystemExit):
        cli.cmd_up(cfg, "qwen38", True, False, False)
    controller.assert_not_called()


def test_recipe_pull_routes_to_prepare_tool_without_generic_ssh(monkeypatch):
    cfg = config()
    cfg["models"]["qwen38"]["recipe"] = "qwen38-nvfp4"
    run = Mock(return_value=completed())
    ssh = Mock()
    monkeypatch.setattr(cli.subprocess, "run", run)
    monkeypatch.setattr(cli, "ssh_cmd", ssh)
    cli.cmd_pull(cfg, "qwen38")
    run.assert_called_once_with([cli.sys.executable, str(ROOT / "tools/prepare_qwen38.py"), "--catalog", str(cli.CATALOG)], check=False)
    ssh.assert_not_called()


@pytest.mark.parametrize("recipe,code", [("unknown", 0), ("qwen38-nvfp4", 2)])
def test_bad_recipe_or_failed_preparation_fails_without_generic_pull(monkeypatch, recipe, code):
    cfg = config()
    cfg["models"]["qwen38"]["recipe"] = recipe
    run = Mock(return_value=completed(code))
    ssh = Mock()
    monkeypatch.setattr(cli.subprocess, "run", run)
    monkeypatch.setattr(cli, "ssh_cmd", ssh)
    with pytest.raises(SystemExit):
        cli.cmd_pull(cfg, "qwen38")
    ssh.assert_not_called()
    if recipe == "unknown":
        run.assert_not_called()


def test_ready_timeout_is_passed_to_actual_readiness_wait(monkeypatch):
    cfg = config()
    mid = "a" * 64
    monkeypatch.setattr(cli, "probe_v1", lambda cfg: "")
    monkeypatch.setattr(cli, "docker_run_script", lambda *a: "launch")
    monkeypatch.setattr(cli, "ssh_cmd", lambda *a, **k: completed(output=f"started head rank=0 id={mid}\n"))
    wait = Mock(return_value=(True, "qwen", "{}"))
    monkeypatch.setattr(cli, "wait_ready", wait)
    monkeypatch.setattr(cli, "maybe_hermes_json", lambda *a, **k: "skipped")
    cli._bring_up(cfg, "qwen38", True, False, Mock())
    assert wait.call_args.kwargs["timeout"] == 3600
    assert cli.model_ready_timeout({}) == 1500
