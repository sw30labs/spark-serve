"""NIM's asymmetric startup must retain the controller's failure guarantees."""
import copy
import importlib.machinery
import importlib.util
import json
import shlex
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from spark_serve_controller import Controller, ControllerError


ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("nim_startup_cli", str(ROOT / "spark-serve"))
spec = importlib.util.spec_from_loader(loader.name, loader)
cli = importlib.util.module_from_spec(spec)
loader.exec_module(cli)


def config():
    return {
        "cluster": {"head": "head", "worker": "worker", "nnodes": 2,
                    "port": 8000, "lan_url": "http://head:8000", "hf_cache_host": "/cache",
                    "master_addr": "192.0.2.1"},
        "models": {"glm53": {
            "image": "registry/nim@sha256:" + "a" * 64, "wrapper": "nim", "nnodes": 2,
            "served_name": "glm-5.3-flash", "max_model_len": 131072, "container": "nim_glm53",
            "serve_path": "/cache/huggingface/spark-serve/glm53-nvfp4/models/nim-aa28e1f-nvfp4",
            "ready_path": "/v1/health/ready", "ready_timeout": 1800, "drop_caches": False,
            "preflight_args": ["/verify.py", "/model"],
            "nim": {"model_source": "ngc", "model_revision": "nim-aa28e1f-nvfp4", "handshake_timeout": 10},
        }},
        "_spark_serve_placement": {"model": "glm53", "hosts": ["head", "worker"],
                                    "node": "both", "generation": "generation"},
    }


def install_fakes(monkeypatch, handshake="NIM_PRIMARY_NODE=192.0.2.55 NIM_NODE_MANAGER_PORT=20000", head_code=0):
    events, monitor = [], Mock()
    monitor.containers = []

    def remember(host, rank, output):
        monitor.containers.append({"host": host, "rank": rank, "id": str(rank) * 64})
        return True

    def fail(reason):
        raise ControllerError(reason)

    monitor.remember.side_effect = remember
    monitor.fail.side_effect = fail
    monkeypatch.setattr(cli, "_StartupMonitor", lambda *a: monitor)
    monkeypatch.setattr(cli, "probe_v1", lambda *a, **k: "")
    monkeypatch.setattr(cli, "guarded_remote_script", lambda generation, script: "guard:" + generation + ":" + script)

    def command(cfg, model, rank, headless):
        if rank == 1:
            assert cfg["_nim_primary_node"] == "192.0.2.55"
        return "launch-" + str(rank)

    monkeypatch.setattr(cli, "docker_run_script", command)

    def ssh(cfg, host, script, **kwargs):
        events.append((host, script))
        if "launch-0" in script:
            return subprocess.CompletedProcess([], head_code, "head launched", "head failure" if head_code else "")
        if "docker logs" in script:
            assert "0" * 64 in script  # Never read another/replacement container's handshake.
            return subprocess.CompletedProcess([], 0, handshake, "")
        if "launch-1" in script:
            return subprocess.CompletedProcess([], 0, "worker launched", "")
        pytest.fail("unexpected SSH: " + script)

    monkeypatch.setattr(cli, "ssh_cmd", ssh)
    monkeypatch.setattr(cli, "maybe_hermes_json", lambda *a, **k: "skipped")

    def ready(cfg, **kwargs):
        assert kwargs["ready_path"] == "/v1/health/ready"
        assert kwargs["expect_served"] == "glm-5.3-flash"
        assert len(monitor.containers) == 2
        assert kwargs["check_started"] == monitor.check
        return True, "glm-5.3-flash", json.dumps({"data": [{"id": "glm-5.3-flash"}]})

    monkeypatch.setattr(cli, "wait_ready", ready)
    return events, monitor


def test_head_handshake_worker_then_readiness_returns_both_receipts(monkeypatch):
    events, monitor = install_fakes(monkeypatch)
    result = cli._bring_up(config(), "glm53", True, False, Mock(), generation="generation")
    assert [host for host, _ in events] == ["head", "head", "worker"]
    assert events[0][1] == "guard:generation:launch-0"
    assert events[2][1] == "guard:generation:launch-1"
    assert result["served"] == "glm-5.3-flash"
    assert len(result["containers"]) == 2
    monitor.check.assert_called()


def test_failed_head_does_not_start_worker(monkeypatch):
    events, _ = install_fakes(monkeypatch, head_code=1)
    with pytest.raises(ControllerError, match="rank 0 launch failed"):
        cli._bring_up(config(), "glm53", True, False, Mock(), generation="generation")
    assert len(events) == 1


def test_conflicting_handshake_never_starts_worker(monkeypatch):
    events, _ = install_fakes(monkeypatch, handshake="NIM_PRIMARY_NODE=192.0.2.55\nNIM_PRIMARY_NODE=192.0.2.56")
    with pytest.raises(ControllerError, match="conflicting"):
        cli._bring_up(config(), "glm53", True, False, Mock(), generation="generation")
    assert all(host == "head" for host, _ in events)


def test_handshake_timeout_preserves_diagnostics_path(monkeypatch):
    events, monitor = install_fakes(monkeypatch, handshake="loading")
    values = iter([0, 0, 11])
    monkeypatch.setattr(cli.time, "monotonic", lambda: next(values))
    # No log poll is necessary once the deadline is exceeded.
    values = iter([0, 11])
    with pytest.raises(ControllerError, match="handshake timeout"):
        cli._bring_up(config(), "glm53", True, False, Mock(), generation="generation")
    monitor.fail.assert_called_once()
    assert len(events) == 1


def test_no_wait_still_observes_handshake_and_records_two_ranks(monkeypatch):
    events, _ = install_fakes(monkeypatch)
    monkeypatch.setattr(cli, "wait_ready", lambda *a, **k: pytest.fail("readiness should not block"))
    result = cli._bring_up(config(), "glm53", True, True, Mock(), generation="generation")
    assert len(events) == 3 and len(result["containers"]) == 2 and result["served"] is None


@pytest.mark.parametrize("change", [
    {"image": "registry/nim:latest"}, {"preflight_args": []}, {"wrapper": "nimm"},
    {"ready_path": None}, {"ready_path": "/v1/models"}, {"ready_path": "/v1/health/live"},
])
def test_invalid_recipe_never_enters_controller(monkeypatch, change):
    cfg = config()
    cfg["models"]["glm53"].update(change)
    controller = Mock()
    monkeypatch.setattr(cli, "Controller", controller)
    monkeypatch.setattr(cli, "model_preflight", lambda *a: pytest.fail("must validate before preflight"))
    with pytest.raises(SystemExit):
        cli.cmd_up(cfg, "glm53", True, False, False)
    controller.assert_not_called()


def test_catalog_exposes_nim_two_spark_topology(capsys):
    cli.cmd_list_json(config())
    item = json.loads(capsys.readouterr().out)[0]
    assert (item["backend"], item["topology"], item["wrapper"]) == ("nim", "distributed", "nim")


def test_logs_selects_catalog_nim_container(monkeypatch):
    monkeypatch.setattr(cli, "ssh_cmd", lambda *a, **k: subprocess.CompletedProcess([], 0, "nim_glm53\n", ""))
    assert cli.pick_log_container(config(), "head") == "nim_glm53"


@pytest.mark.parametrize("engine_ready,worker_present,expected", [
    (False, True, False), (True, True, True), (True, False, False),
])
def test_status_requires_engine_readiness_and_both_immutable_ranks(
    monkeypatch, engine_ready, worker_present, expected,
):
    cfg = config()
    hosts = ["head", "worker"]
    ids = {"head": "a" * 64, "worker": "b" * 64}
    assignments = [{
        "host": host, "mode": "vllm", "model": "glm53", "phase": "ready",
        "generation": "allocation", "allocation_id": "allocation", "allocation_hosts": hosts,
        "legacy": False, "error": None, "containers": [{"host": host, "id": ids[host]}],
    } for host in hosts]
    containers = {host: [{
        "name": "nim_glm53", "id": ids[host], "state": "running", "running": True,
        "image": cfg["models"]["glm53"]["image"],
        "labels": {"ai.spark-serve.model": "glm53", "ai.spark-serve.allocation": "allocation",
                   "ai.spark-serve.hosts": json.dumps(hosts)},
    }] for host in hosts}
    if not worker_present:
        containers["worker"] = []
    managed = {"mode": "vllm", "phase": "ready", "nodes": assignments,
               "yue_workers": [], "transition_error": None}
    monkeypatch.setattr(cli, "Controller", lambda *args: Mock(status=lambda: copy.deepcopy(managed)))
    monkeypatch.setattr(cli, "status_host_json", lambda cfg, host: containers[host])
    monkeypatch.setattr(cli, "probe_v1", lambda cfg: json.dumps({"data": [{"id": "glm-5.3-flash"}]})
                        if cfg["cluster"]["head"] == "head" else "")
    monkeypatch.setattr(cli, "_active_client_node", lambda cfg, nodes: None)
    readiness_calls = []

    def readiness(cfg, model):
        readiness_calls.append(cfg["cluster"]["head"])
        assert model["ready_path"] == "/v1/health/ready"
        return engine_ready

    monkeypatch.setattr(cli, "probe_ready_path", readiness)
    observed = cli.collect_status(cfg)
    assert [node["ready"] for node in observed["nodes"]] == [expected, expected]
    assert [node["can_use"] for node in observed["nodes"]] == [expected, False]
    assert readiness_calls == ["head"]  # The headless rank never presents a client model.


def test_rank_one_failure_saves_diagnostics_before_controller_cleans_both_ids(monkeypatch, tmp_path):
    cfg = config()
    ids = {"head": "a" * 64, "worker": "b" * 64}
    running = {host: [] for host in ids}
    removed = []
    controller = Controller(cfg, lambda *args, **kwargs: pytest.fail("unexpected controller SSH"), directory=tmp_path)
    monkeypatch.setattr(controller, "fence_nodes", lambda generation: [])
    monkeypatch.setattr(controller, "stop_yue", lambda **kwargs: None)
    monkeypatch.setattr(controller, "audit", lambda host: {
        "containers": copy.deepcopy(running[host]), "listening_ports": [], "gpu_processes": [],
    })

    def remove(host, script, **kwargs):
        # Keep the actual Controller.stop_vllm implementation and inspect the
        # immutable IDs it submits for cleanup of this failed allocation.
        argv = shlex.split(script.splitlines()[-1])
        assert argv[:3] == ["docker", "rm", "-f"]
        assert controller.operation_generation
        assert list((tmp_path / "diagnostics").glob("startup-*/failure.json"))
        removed.extend((host, cid) for cid in argv[3:])
        running[host] = [item for item in running[host] if item["id"] not in argv[3:]]
        return ""

    monkeypatch.setattr(controller, "remote", remove)
    monkeypatch.setattr(cli, "state_dir", lambda: tmp_path)
    monkeypatch.setattr(cli, "probe_v1", lambda *args, **kwargs: "")
    monkeypatch.setattr(cli, "docker_run_script", lambda cfg, model, rank, headless: "launch-" + str(rank))
    monkeypatch.setattr(cli, "wait_ready", lambda *args, **kwargs: pytest.fail("failed rank must not reach readiness"))

    def ssh(cfg, host, script, **kwargs):
        if "launch-" in script:
            rank = 0 if host == "head" else 1
            assert script.startswith("# spark-serve: generation-fence")
            running[host].append({"name": "nim_glm53", "id": ids[host], "running": rank == 0, "gpu": True})
            output = f"created {host} rank={rank} id={ids[host]}\n"
            if rank == 0:
                output += f"started {host} rank={rank} id={ids[host]}\n"
            return subprocess.CompletedProcess([], rank, output, "rank one failed" if rank else "")
        if "docker inspect" in script:
            assert ids[host] in script
            return subprocess.CompletedProcess([], 0, json.dumps({"Status": "running", "Running": True}), "")
        if "docker logs" in script:
            assert ids[host] in script
            return subprocess.CompletedProcess([], 0, "NIM_PRIMARY_NODE=192.0.2.55" if host == "head" else "rank one failed", "")
        pytest.fail("unexpected startup SSH: " + script)

    monkeypatch.setattr(cli, "ssh_cmd", ssh)
    with pytest.raises(ControllerError, match="rank 1 launch failed"):
        controller.switch("glm53", lambda generation: cli._bring_up(cfg, "glm53", True, False, Mock(), generation=generation))
    assert removed == [("head", ids["head"]), ("worker", ids["worker"])]
    assert not any(running.values())
    report_path, = (tmp_path / "diagnostics").glob("startup-*/failure.json")
    report = json.loads(report_path.read_text())
    assert {item["id"] for item in report["containers"]} == set(ids.values())
    persisted = json.loads((tmp_path / "controller.json").read_text())
    assert persisted["phase"] == "failed" and persisted["cleanup_errors"] == []
    assert "rank 1 launch failed" in persisted["error"]
