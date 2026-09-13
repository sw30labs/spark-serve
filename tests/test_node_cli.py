"""Physical placement and client selection must never drift to the other GPU."""
import contextlib
import copy
import importlib.machinery
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from spark_serve_controller import ControllerError
from spark_serve_nodes import launch_config, node_url, placement

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("node_cli_test", str(ROOT / "spark-serve"))
spec = importlib.util.spec_from_loader(loader.name, loader)
cli = importlib.util.module_from_spec(spec)
loader.exec_module(cli)


def config():
    return {"cluster": {"head": "first", "worker": "second", "lan_url": "http://first.lan:8000",
                        "worker_lan_url": "http://second.lan:8000", "port": 8000,
                        "nnodes": 2, "tensor_parallel": 2, "master_addr": "192.0.2.1", "master_port": 29501,
                        "hf_cache_host": "/first/cache", "worker_hf_cache_host": "/second/cache",
                        "container": "vllm_cluster", "keep_containers": ["neo4j"]},
            "models": {name: {"served_name": name, "hf_id": name, "image": name + ":pinned", "nnodes": count,
                              "tensor_parallel": count, "hermes_provider": "spark", "max_model_len": 262144}
                       for name, count in (("qwen", 1), ("nemotron", 1), ("distributed", 2))}}


def test_worker_remaps_endpoint_cache_mounts_ip_and_rendezvous_without_mutating_catalog():
    cfg = config()
    cfg["models"]["nemotron"].update(host_ip_head="192.0.2.1", host_ip_worker="192.0.2.2",
                                         mounts=[{"head": "/first/extra", "worker": "/second/extra", "container": "/extra", "ro": True}])
    original = copy.deepcopy(cfg)
    scoped = launch_config(cfg, "nemotron", "worker")
    assert cfg == original
    assert scoped["cluster"]["head"] == "second"
    assert scoped["cluster"]["lan_url"] == "http://second.lan:8000"
    assert scoped["cluster"]["hf_cache_host"] == "/second/cache"
    assert scoped["models"]["nemotron"]["master_addr"] == "127.0.0.1"
    assert scoped["models"]["nemotron"]["host_ip_head"] == "192.0.2.2"
    assert scoped["models"]["nemotron"]["mounts"][0]["head"] == "/second/extra"
    assert scoped["models"]["nemotron"]["hermes_provider"] == "spark-worker"
    assert scoped["models"]["nemotron"]["tensor_parallel"] == 1


def test_missing_worker_mount_rejected():
    cfg = config()
    cfg["models"]["nemotron"]["mounts"] = [{"head": "/first/extra", "container": "/extra"}]
    with pytest.raises(ControllerError, match="worker path"):
        launch_config(cfg, "nemotron", "worker")


@pytest.mark.parametrize("url", ["http://user:password@second:8000", "http://second:8000/v1", "http://second:9000", "http://second:8000/?token=x"])
def test_worker_endpoint_is_validated(url):
    cfg = config(); cfg["cluster"]["worker_lan_url"] = url
    with pytest.raises(ControllerError):
        node_url(cfg, "worker")


def test_existing_yue_http_hostname_is_used_instead_of_ssh_alias():
    cfg = config(); cfg["cluster"].pop("worker_lan_url")
    cfg["yue"] = {"worker_url": "http://actual-worker.local:8011"}
    assert node_url(cfg, "worker") == "http://actual-worker.local:8000"


def test_duplicate_serving_address_rejected_before_mutation(monkeypatch):
    cfg = config(); cfg["cluster"]["worker_lan_url"] = "http://FIRST.LAN:8000/"
    controller = Mock(); monkeypatch.setattr(cli, "Controller", controller)
    with pytest.raises(SystemExit):
        cli.cmd_up(cfg, "nemotron", True, False, True, node="worker")
    controller.assert_not_called()


def test_distributed_placement_cannot_be_accidentally_split():
    cfg = config()
    assert placement(cfg, cfg["models"]["distributed"]) == ("both", ["first", "second"])
    with pytest.raises(ControllerError, match="both"):
        placement(cfg, cfg["models"]["distributed"], "worker")
    with pytest.raises(ControllerError, match="single-Spark"):
        placement(cfg, cfg["models"]["nemotron"], "both")


def test_worker_start_scopes_preflight_launch_and_controller_and_preserves_client(monkeypatch):
    cfg = config(); checked = []; started = []
    controller = Mock()
    controller.switch.side_effect = lambda target, start, **kw: start("fresh-worker-generation")
    monkeypatch.setattr(cli, "Controller", Mock(return_value=controller))
    monkeypatch.setattr(cli, "model_preflight", lambda cfg, *a: checked.append(cfg["cluster"]["head"]))
    monkeypatch.setattr(cli, "_bring_up", lambda cfg, mid, no_hermes, *a, **kw: started.append((cfg, mid, no_hermes)))
    cli.cmd_up(cfg, "nemotron", False, False, True, node="worker")
    assert checked == ["second"]
    assert controller.switch.call_args.kwargs["hosts"] == ["second"]
    assert started[0][0]["cluster"]["head"] == "second"
    assert started[0][1:] == ("nemotron", True)


def test_container_labels_capture_physical_allocation():
    cfg = launch_config(config(), "nemotron", "worker")
    cfg["_spark_serve_placement"]["generation"] = "generation-two"
    argv = cli.docker_base(cfg, cfg["models"]["nemotron"], "vllm_cluster")
    assert 'ai.spark-serve.hosts=["second"]' in argv
    assert "ai.spark-serve.model=nemotron" in argv
    assert "ai.spark-serve.allocation=generation-two" in argv
    assert "/second/cache:/cache/huggingface" in argv


def setup_status(monkeypatch, tmp_path, *, second_phase="ready", second_id="nemotron", distributed=False):
    cfg = config(); model_a = "distributed" if distributed else "qwen"; model_b = "distributed" if distributed else "nemotron"
    assignments = [{"host": host, "mode": "vllm", "model": model, "phase": "ready" if host == "first" else second_phase,
                    "generation": "group" if distributed else host,
                    "allocation_id": "group" if distributed else host,
                    "allocation_hosts": ["first", "second"] if distributed else [host], "error": None,
                    "legacy": False, "containers": [{"host": host, "id": host + "-immutable"}]}
                   for host, model in (("first", model_a), ("second", model_b))]
    containers = {host: [{"name": "vllm_cluster", "state": "running", "image": model + ":pinned", "running": True,
                          "id": host + "-immutable", "labels": {"ai.spark-serve.model": model,
                          "ai.spark-serve.allocation": "group" if distributed else host,
                          "ai.spark-serve.hosts": json.dumps(["first", "second"] if distributed else [host])}}]
                  for host, model in (("first", model_a), ("second", model_b))}
    managed = {"mode": "vllm", "phase": second_phase, "nodes": assignments, "yue_workers": [], "transition_error": None}
    monkeypatch.setattr(cli.Controller, "status", lambda self: copy.deepcopy(managed))
    monkeypatch.setattr(cli, "status_host_json", lambda cfg, host: containers[host])
    monkeypatch.setattr(cli, "probe_v1", lambda cfg: json.dumps({"data": [{"id": model_a if cfg["cluster"]["head"] == "first" else second_id}]}))
    monkeypatch.setattr(cli, "HERMES_CONFIG", tmp_path / "no-client-config")
    return cfg, containers


def test_two_model_endpoints_are_independently_ready(monkeypatch, tmp_path):
    cfg, _ = setup_status(monkeypatch, tmp_path)
    result = cli.collect_status(cfg)
    assert [(item["served"], item["ready"], item["can_use"]) for item in result["nodes"]] == [("qwen", True, True), ("nemotron", True, True)]
    assert result["nodes"][1]["url"] == "http://second.lan:8000"


@pytest.mark.parametrize("phase,served", [("failed", "nemotron"), ("ready", "wrong-model")])
def test_worker_failure_or_wrong_identity_does_not_hide_healthy_head(monkeypatch, tmp_path, phase, served):
    cfg, _ = setup_status(monkeypatch, tmp_path, second_phase=phase, second_id=served)
    result = cli.collect_status(cfg)
    assert result["ready"] and result["nodes"][0]["ready"]
    assert not result["nodes"][1]["ready"]


def test_wrong_allocation_label_cannot_claim_ready(monkeypatch, tmp_path):
    cfg, containers = setup_status(monkeypatch, tmp_path)
    containers["second"][0]["labels"]["ai.spark-serve.allocation"] = "stale"
    result = cli.collect_status(cfg)
    assert result["nodes"][0]["ready"] and not result["nodes"][1]["ready"]


def test_unlabelled_replacement_cannot_inherit_new_allocation_readiness(monkeypatch, tmp_path):
    cfg, containers = setup_status(monkeypatch, tmp_path)
    containers["second"][0].update(id="replacement-container", labels={})
    result = cli.collect_status(cfg)
    assert result["nodes"][0]["ready"] and not result["nodes"][1]["ready"]


def test_headless_distributed_worker_is_ready_only_as_part_of_healthy_group(monkeypatch, tmp_path):
    cfg, containers = setup_status(monkeypatch, tmp_path, distributed=True, second_id=None)
    result = cli.collect_status(cfg)
    assert result["nodes"][0]["ready"] and result["nodes"][1]["ready"]
    assert result["nodes"][0]["can_use"] and not result["nodes"][1]["can_use"]
    containers["second"] = []
    assert not any(item["ready"] for item in cli.collect_status(cfg)["nodes"])


def test_stale_distributed_rank_cannot_be_promoted_by_group_readiness(monkeypatch, tmp_path):
    cfg, containers = setup_status(monkeypatch, tmp_path, distributed=True, second_id=None)
    containers["second"][0]["labels"]["ai.spark-serve.allocation"] = "older-group"
    assert not any(item["ready"] or item["can_use"] for item in cli.collect_status(cfg)["nodes"])


def test_worker_client_selection_changes_only_hermes(monkeypatch):
    cfg = config(); controller = Mock()
    controller.lock.return_value = contextlib.nullcontext()
    monkeypatch.setattr(cli, "Controller", Mock(return_value=controller))
    monkeypatch.setattr(cli, "collect_status", lambda cfg: {"nodes": [{"node": "worker", "host": "second", "can_use": True, "model": "nemotron", "served": "nemotron", "url": "http://second.lan:8000"}]})
    patch = Mock(return_value=("selected", True)); monkeypatch.setattr(cli, "_hermes_patch", patch)
    cli.cmd_use(cfg, "worker", True)
    selected_cfg, _, model, _ = patch.call_args.args
    assert selected_cfg["cluster"]["lan_url"] == "http://second.lan:8000"
    assert model["hermes_provider"] == "spark-worker"
    controller.switch.assert_not_called()
