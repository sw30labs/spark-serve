"""Hermes config destination must stay independent of model placement."""
import contextlib
import copy
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest
import yaml

from spark_serve_nodes import launch_config

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("hermes_target_test", str(ROOT / "spark-serve"))
spec = importlib.util.spec_from_loader(loader.name, loader)
cli = importlib.util.module_from_spec(spec)
loader.exec_module(cli)


def config():
    return {
        "cluster": {
            "head": "sparkone", "worker": "sparktwo", "port": 8000,
            "lan_url": "http://sparkone.lan:8000",
            "worker_lan_url": "http://sparktwo.lan:8000",
            "hf_cache_host": "/cache", "nnodes": 2,
        },
        "models": {"qwen": {
            "served_name": "qwen-flash", "nnodes": 1,
            "hermes_provider": "spark", "hermes_context_length": 262144,
            "max_model_len": 131072, "hermes_supports_vision": True,
        }},
    }


def ready_controller(monkeypatch, *, can_use=True):
    controller = Mock()
    controller.lock.return_value = contextlib.nullcontext()
    monkeypatch.setattr(cli, "Controller", Mock(return_value=controller))
    monkeypatch.setattr(cli, "collect_status", lambda cfg: {"nodes": [{
        "node": "worker", "host": "sparktwo", "can_use": can_use,
        "model": "qwen", "served": "qwen-flash", "url": "http://sparktwo.lan:8000",
    }]})
    return controller


def selected_event(capsys):
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    return next(event for event in events if event["event"] == "selected")


def test_spark_target_uses_original_head_with_worker_endpoint(monkeypatch, capsys):
    cfg = config()
    original = copy.deepcopy(cfg)
    controller = ready_controller(monkeypatch)
    patcher = Mock(return_value=("patched", True))
    monkeypatch.setattr(cli, "_hermes_patch", patcher)

    cli.cmd_use(cfg, "worker", json_mode=True, hermes_target="spark")

    scoped, mid, model, skip = patcher.call_args.args
    assert scoped["cluster"]["head"] == "sparktwo"
    assert scoped["cluster"]["lan_url"] == "http://sparktwo.lan:8000"
    assert model["hermes_provider"] == "spark-worker"
    assert mid == "qwen" and not skip
    assert patcher.call_args.kwargs["remote_host"] == "sparkone"
    assert cfg == original
    controller.switch.assert_not_called()
    event = selected_event(capsys)
    assert event["hermes_target"] == "spark"
    assert event["hermes_host"] == "sparkone"
    assert event["node"] == "worker"


def test_default_target_keeps_local_selection(monkeypatch, capsys):
    controller = ready_controller(monkeypatch)
    patcher = Mock(return_value=("patched", True))
    monkeypatch.setattr(cli, "_hermes_patch", patcher)

    cli.cmd_use(config(), "worker", json_mode=True)

    assert patcher.call_args.kwargs.get("remote_host") is None
    controller.switch.assert_not_called()
    event = selected_event(capsys)
    assert event["hermes_target"] == "local"
    assert event["hermes_host"] is None


@pytest.mark.parametrize("target", ["local", "spark"])
def test_unverified_endpoint_never_patches_either_config(monkeypatch, capsys, target):
    controller = ready_controller(monkeypatch, can_use=False)
    patcher = Mock()
    monkeypatch.setattr(cli, "_hermes_patch", patcher)

    with pytest.raises(SystemExit):
        cli.cmd_use(config(), "worker", json_mode=True, hermes_target=target)

    patcher.assert_not_called()
    controller.switch.assert_not_called()
    assert '"event": "selected"' not in capsys.readouterr().out


def test_cli_target_is_explicit_and_defaults_to_local():
    parser = cli.build_parser()
    assert parser.parse_args(["use", "--node", "worker"]).hermes_target == "local"
    assert parser.parse_args([
        "use", "--node", "worker", "--hermes-target", "spark",
    ]).hermes_target == "spark"
    with pytest.raises(SystemExit):
        parser.parse_args(["use", "--node", "worker", "--hermes-target", "sparktwo"])


@pytest.fixture
def remote_home(monkeypatch, tmp_path):
    """Execute the exact SSH payload offline under a temporary remote HOME."""
    home = tmp_path / "spark-home"
    path = home / ".hermes" / "config.yaml"
    path.parent.mkdir(parents=True)
    local = tmp_path / "mac-config.yaml"
    local.write_text("model: {default: mac-model, provider: mac}\n")
    monkeypatch.setattr(cli, "HERMES_CONFIG", local)
    calls = []
    env = dict(os.environ, HOME=str(home))
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env.pop("HERMES_HOME", None)

    def ssh(cfg, host, remote, **kwargs):
        calls.append(host)
        return subprocess.run(
            ["bash", "-s"], input=remote, text=True, capture_output=True,
            env=env, timeout=15, check=False,
        )

    monkeypatch.setattr(cli, "ssh_cmd", ssh)
    yield path, local, calls
    assert local.read_text() == "model: {default: mac-model, provider: mac}\n"


def patch_remote():
    scoped = launch_config(config(), "qwen", "worker")
    return cli._hermes_patch(
        scoped, "qwen", scoped["models"]["qwen"], False, remote_host="sparkone",
    )


def test_remote_patch_preserves_config_and_file_permissions(remote_home):
    path, local, calls = remote_home
    shared = {"context_length": 8192, "supports_vision": False, "custom_setting": "keep"}
    original = {
        "model": {"default": "old", "provider": "other", "context_length": 1048576, "max_tokens": 4096},
        "providers": {
            "other": {"api_key": "PRIVATE-DO-NOT-LOG", "base_url": "http://other/v1"},
            "spark-ds4": {"base_url": "http://sparkone.lan:8000/v1"},
            "spark-worker": {
                "api_key": "PRIVATE-SPARK-KEY", "api_mode": "chat_completions",
                "models": {"old": shared, "qwen-flash": shared},
            },
        },
    }
    path.write_text(yaml.safe_dump(original, sort_keys=False))
    path.chmod(0o600)

    message, ok = patch_remote()

    assert ok, message
    assert calls == ["sparkone"]
    assert "PRIVATE-DO-NOT-LOG" not in message and "PRIVATE-SPARK-KEY" not in message
    assert path.stat().st_mode & 0o777 == 0o600
    assert set(path.parent.iterdir()) == {path}
    actual = yaml.safe_load(path.read_text())
    assert actual["model"] == {"default": "qwen-flash", "provider": "spark-worker", "max_tokens": 4096}
    assert actual["providers"]["other"] == original["providers"]["other"]
    assert actual["providers"]["spark-ds4"] == original["providers"]["spark-ds4"]
    provider = actual["providers"]["spark-worker"]
    assert provider["base_url"] == "http://sparktwo.lan:8000/v1"
    assert provider["api_key"] == "PRIVATE-SPARK-KEY"
    assert provider["models"]["old"] == shared
    assert provider["models"]["qwen-flash"] == {
        "context_length": 262144, "supports_vision": True, "custom_setting": "keep",
    }


@pytest.mark.parametrize("original", [
    "model: [PRIVATE-MALFORMED-SECRET\n",
    "model: {default: old, provider: spark-worker}\nproviders:\n  spark-worker:\n    models: {qwen-flash: PRIVATE-MALFORMED-SECRET}\n",
])
def test_remote_malformed_yaml_is_not_overwritten_or_logged(remote_home, original):
    path, local, calls = remote_home
    path.write_text(original)

    message, ok = patch_remote()

    assert not ok
    assert path.read_text() == original
    assert "PRIVATE-MALFORMED-SECRET" not in message
    assert set(path.parent.iterdir()) == {path}


def test_missing_remote_config_is_not_created(remote_home):
    path, local, calls = remote_home

    message, ok = patch_remote()

    assert not ok
    assert calls == ["sparkone"]
    assert not path.exists()


def test_ssh_failure_does_not_echo_remote_output(monkeypatch):
    ssh = Mock(return_value=subprocess.CompletedProcess(
        [], 255, "PRIVATE-REMOTE-STDOUT", "PRIVATE-REMOTE-STDERR",
    ))
    monkeypatch.setattr(cli, "ssh_cmd", ssh)

    message, ok = patch_remote()

    assert not ok
    assert "PRIVATE-REMOTE" not in message
    assert ssh.call_args.args[1] == "sparkone"


def test_failed_remote_patch_never_emits_selected(monkeypatch, capsys):
    controller = ready_controller(monkeypatch)
    monkeypatch.setattr(cli, "_hermes_patch", Mock(return_value=("not patched", False)))

    with pytest.raises(SystemExit):
        cli.cmd_use(config(), "worker", json_mode=True, hermes_target="spark")

    controller.switch.assert_not_called()
    assert '"event": "selected"' not in capsys.readouterr().out


def test_skipped_remote_patch_never_connects(monkeypatch):
    ssh = Mock()
    monkeypatch.setattr(cli, "ssh_cmd", ssh)

    _, ok = cli._hermes_patch(config(), "qwen", config()["models"]["qwen"], True, remote_host="sparkone")

    assert not ok
    ssh.assert_not_called()
