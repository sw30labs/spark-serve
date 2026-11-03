"""SuperDeepseek Start keeps Hermes on the spark-superds provider over IPv4.

sparkone.local resolves to IPv6 first and vLLM listens on IPv4 only, so the
catalog pins hermes_base_url to the head's IPv4 address (ADR 0016 amendment).
"""
import copy
import importlib.machinery
import importlib.util
import tomllib
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("spark_serve_superdeepseek_hermes_test", str(ROOT / "spark-serve"))
spec = importlib.util.spec_from_loader(loader.name, loader)
cli = importlib.util.module_from_spec(spec)
loader.exec_module(cli)

MID = "ds4-super-abl"
SERVED = "SuperDeepseek-V4-Flash-abliterated-MQ-2xDGX"
IPV4 = "http://192.168.86.57:8000/v1"


def catalog(url=IPV4):
    cfg = tomllib.loads((ROOT / "models.example.toml").read_text())
    cfg["cluster"]["lan_url"] = "http://sparkone.local:8000"
    cfg["cluster"]["port"] = 8000
    if url is None:
        cfg["models"][MID].pop("hermes_base_url", None)
    else:
        cfg["models"][MID]["hermes_base_url"] = url
    return cfg


def hermes_config(**superds):
    return {
        "model": {"default": "GLM-5.3-Flash-EXL3", "provider": "spark", "max_tokens": 32768},
        "providers": {
            "spark-superds": {"name": "Spark SuperDeepseek (abliterated)", "api_key": "test-key",
                              "api_mode": "chat_completions", "base_url": "http://192.168.86.57:8890/v1",
                              "default_model": SERVED, "models": {SERVED: {"context_length": 1048576}},
                              **superds},
            "spark": {"name": "Spark cluster", "api_key": "test-key", "api_mode": "chat_completions",
                      "base_url": "http://sparkone.local:8000/v1", "default_model": "GLM-5.3-Flash-EXL3",
                      "models": {"GLM-5.3-Flash-EXL3": {"context_length": 262144}}},
            "spark-ds4": {"base_url": "http://sparkone.local:8000/v1"},
        },
    }


def patch_config(monkeypatch, tmp_path, cfg, original, node="both"):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(original, sort_keys=False))
    path.chmod(0o600)
    monkeypatch.setattr(cli, "HERMES_CONFIG", path)
    scoped = cli.launch_config(cfg, MID, node)
    message, ok = cli._hermes_patch(scoped, MID, scoped["models"][MID], False)
    return path, message, ok


def test_example_catalog_selects_spark_superds_on_an_ipv4_url():
    model = tomllib.loads((ROOT / "models.example.toml").read_text())["models"][MID]
    assert model["hermes_provider"] == "spark-superds"
    assert model["hermes_base_url"].endswith(":8000/v1")
    assert "local" not in model["hermes_base_url"]


def test_start_hook_points_spark_superds_at_ipv4_and_leaves_spark_alone(monkeypatch, tmp_path):
    original = hermes_config()
    path, message, ok = patch_config(monkeypatch, tmp_path, catalog(), original)
    assert ok, message
    actual = yaml.safe_load(path.read_text())
    assert actual["model"]["provider"] == "spark-superds" and actual["model"]["default"] == SERVED
    assert actual["model"]["max_tokens"] == 32768
    block = actual["providers"]["spark-superds"]
    assert block["base_url"] == IPV4
    assert block["name"] == "Spark SuperDeepseek (abliterated)"  # display name kept
    assert block["api_key"] == "test-key" and block["default_model"] == SERVED
    assert block["models"][SERVED]["context_length"] == 1048576
    assert actual["providers"]["spark"] == original["providers"]["spark"]
    # The legacy spark-ds4 alias still follows the cluster URL, as before.
    assert actual["providers"]["spark-ds4"]["base_url"] == "http://sparkone.local:8000/v1"
    assert path.stat().st_mode & 0o777 == 0o600


def test_without_override_the_hook_keeps_the_cluster_url(monkeypatch, tmp_path):
    cfg = catalog(url=None)
    path, message, ok = patch_config(monkeypatch, tmp_path, cfg, hermes_config())
    assert ok, message
    assert yaml.safe_load(path.read_text())["providers"]["spark-superds"]["base_url"] == "http://sparkone.local:8000/v1"


def test_new_provider_block_still_gets_the_default_name(monkeypatch, tmp_path):
    original = hermes_config()
    del original["providers"]["spark-superds"]
    path, message, ok = patch_config(monkeypatch, tmp_path, catalog(), original)
    assert ok, message
    block = yaml.safe_load(path.read_text())["providers"]["spark-superds"]
    assert block["name"] == "Spark cluster" and block["base_url"] == IPV4 and block["api_key"] == "dummy"


def test_worker_placement_ignores_the_head_override():
    cfg = catalog()
    cfg["models"]["qwen"] = {"served_name": "qwen", "nnodes": 1, "hermes_base_url": IPV4}
    cfg["cluster"]["worker_lan_url"] = "http://spark-two.local:8000"
    scoped = cli.launch_config(cfg, "qwen", "worker")
    assert cli.hermes_base_url(scoped, "qwen", scoped["models"]["qwen"]) == "http://spark-two.local:8000/v1"
    head = cli.launch_config(cfg, "qwen", "head")
    assert cli.hermes_base_url(head, "qwen", head["models"]["qwen"]) == IPV4


@pytest.mark.parametrize("url", ["http://192.168.86.57:8890/v1", "http://192.168.86.57:8000",
                                 "http://user:pw@192.168.86.57:8000/v1", "ftp://192.168.86.57:8000/v1",
                                 "http://192.168.86.57:8000/v1?x=1", 8000])
def test_bad_override_refuses_before_any_switch(monkeypatch, capsys, url):
    ssh, controller = Mock(), Mock()
    monkeypatch.setattr(cli, "ssh_cmd", ssh)
    monkeypatch.setattr(cli, "Controller", controller)
    with pytest.raises(SystemExit):
        cli.cmd_up(catalog(url), MID, False, False, True)
    ssh.assert_not_called()
    controller.assert_not_called()
    assert "hermes_base_url" in capsys.readouterr().out


def test_bad_override_never_edits_hermes(monkeypatch, tmp_path):
    original = hermes_config()
    path, message, ok = patch_config(monkeypatch, tmp_path, catalog("http://192.168.86.57:8890/v1"), original)
    assert not ok and "cluster.port" in message
    assert yaml.safe_load(path.read_text()) == original


def test_status_recognizes_hermes_on_the_pinned_ipv4_url(monkeypatch, tmp_path):
    cfg = catalog()
    selected = hermes_config()
    selected["model"].update(default=SERVED, provider="spark-superds")
    selected["providers"]["spark-superds"]["base_url"] = IPV4
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(selected))
    monkeypatch.setattr(cli, "HERMES_CONFIG", path)
    nodes = [{"node": "head", "can_use": True, "served": SERVED, "model": MID},
             {"node": "worker", "can_use": True, "served": SERVED, "model": MID}]
    assert cli._active_client_node(cfg, nodes) == "head"
    other = copy.deepcopy(cfg)
    del other["models"][MID]["hermes_base_url"]
    assert cli._active_client_node(other, nodes) is None
