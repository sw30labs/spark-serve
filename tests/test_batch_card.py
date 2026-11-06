"""Card-only batch entries (ADR 0017): listed, never started, stopped or prepared."""

import importlib.machinery
import importlib.util
import json
import tomllib
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("spark_serve_batch_card_test", str(ROOT / "spark-serve"))
spec = importlib.util.spec_from_loader(loader.name, loader)
cli = importlib.util.module_from_spec(spec)
loader.exec_module(cli)


def example():
    with (ROOT / "models.example.toml").open("rb") as f:
        return tomllib.load(f)


def config():
    return {
        "cluster": {"head": "sparkone", "worker": "sparktwo", "nnodes": 2,
                    "lan_url": "http://head.lan:8000", "container": "vllm_cluster"},
        "models": {
            "served": {"image": "img", "served_name": "s"},
            "pika": {"wrapper": "batch", "launch": False, "home_node": "worker", "nnodes": 1,
                     "image": "", "served_name": "", "modalities": ["text", "image"],
                     "outputs": ["video", "audio"], "run_hint": "Run run-t2va.sh on sparktwo."},
        },
    }


def forbid_remote(monkeypatch):
    ssh = Mock(side_effect=AssertionError("card-only entries must not reach SSH"))
    controller = Mock(side_effect=AssertionError("card-only entries must not reach the controller"))
    run = Mock(side_effect=AssertionError("card-only entries must not run a preparation"))
    monkeypatch.setattr(cli, "ssh_cmd", ssh)
    monkeypatch.setattr(cli, "Controller", controller)
    monkeypatch.setattr(cli.subprocess, "run", run)
    return ssh, controller, run


def test_launch_refusal_only_for_card_entries():
    cfg = config()
    assert cli.launch_refusal("served", cfg["models"]["served"]) is None
    assert "card only" in cli.launch_refusal("pika", cfg["models"]["pika"])
    assert cli.launch_refusal("x", {"wrapper": "batch"})  # batch alone is card-only
    assert cli.launch_refusal("x", {"wrapper": "vllm", "launch": False})


@pytest.mark.parametrize("node", [None, "head", "worker", "both"])
def test_up_refuses_before_any_remote_step(monkeypatch, capsys, node):
    ssh, controller, run = forbid_remote(monkeypatch)
    with pytest.raises(SystemExit):
        cli.cmd_up(config(), "pika", no_hermes=True, no_wait=False, json_mode=False, node=node)
    assert "card only" in capsys.readouterr().err
    ssh.assert_not_called(); controller.assert_not_called(); run.assert_not_called()


def test_up_print_cmd_also_refuses(monkeypatch):
    forbid_remote(monkeypatch)
    with pytest.raises(SystemExit):
        cli.cmd_up(config(), "pika", no_hermes=True, no_wait=False, json_mode=False, print_cmd=True)


def test_up_json_mode_reports_refusal(monkeypatch, capsys):
    forbid_remote(monkeypatch)
    with pytest.raises(SystemExit):
        cli.cmd_up(config(), "pika", no_hermes=True, no_wait=False, json_mode=True, node="worker")
    assert "card only" in capsys.readouterr().out


def test_pull_refuses_without_downloading(monkeypatch, capsys):
    ssh, controller, run = forbid_remote(monkeypatch)
    with pytest.raises(SystemExit):
        cli.cmd_pull(config(), "pika", node="worker")
    assert "card only" in capsys.readouterr().err
    ssh.assert_not_called(); run.assert_not_called()


def test_list_json_marks_card(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli, "catalog_freshness", lambda *a, **k: {})
    monkeypatch.setattr(cli, "measured_performance", lambda: {})
    cli.cmd_list_json(config())
    items = {i["id"]: i for i in json.loads(capsys.readouterr().out)}
    pika, served = items["pika"], items["served"]
    assert pika["launch"] is False and pika["backend"] == "batch" and pika["wrapper"] == "batch"
    assert pika["topology"] == "single" and pika["home_host"] == "sparktwo"
    assert pika["outputs"] == ["video", "audio"] and pika["launch_note"]
    assert served["launch"] is True and served["outputs"] is None and served["launch_note"] is None


def test_text_list_handles_card_without_image(capsys):
    cli.cmd_list(config())
    line = next(l for l in capsys.readouterr().out.splitlines() if l.startswith("pika"))
    assert "card only: batch job on sparktwo, no endpoint" in line


def test_example_catalog_pika_entry_is_a_launchable_batch_job():
    cfg = example()
    model = cfg["models"]["pika-h3-flex"]
    assert cli.launch_refusal("pika-h3-flex", model) is None  # ADR 0018 superseded card-only for Pika
    assert model["home_node"] == "worker" and int(model["nnodes"]) == 1
    assert "container" not in model  # never joins the stop set
    assert cli.resolve_model(cfg, "pika") == ("pika-h3-flex", model)
    # Not added to any stop list or the protected set.
    assert "pika" not in " ".join(cfg["cluster"].get("stop_names") or [])
