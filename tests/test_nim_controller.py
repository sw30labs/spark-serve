"""Recipe-specific port checks follow the selected and departing allocations."""
import json
import shlex
import subprocess
import sys
from unittest.mock import Mock

import pytest

from spark_serve_controller import Controller, ControllerError


def config():
    return {
        "cluster": {"head": "head", "worker": "worker", "nnodes": 2,
                    "port": 8000, "lan_url": "http://head:8000"},
        "models": {
            "glm53": {"wrapper": "nim", "nnodes": 2, "served_name": "glm-5.3-flash",
                      "container": "nim_glm53", "nim": {"manager_port": 20000, "worker_port": 8002}},
            "solo": {"wrapper": "vllm", "nnodes": 1, "served_name": "solo"},
        },
    }


def install_audit_transport(controller, monkeypatch, listeners=None):
    """Exercise the real audit script generator; never run its remote payload."""
    calls = []
    listeners = listeners or {}

    def remote(host, script, *, json_output=False, **kwargs):
        assert script.startswith("# spark-serve: audit\n")
        assert json_output
        argv = shlex.split(script.partition("\n")[2])
        assert argv[:2] == ["python3", "-c"]
        ports = json.loads(argv[-1])
        calls.append((host, ports))
        return {"containers": [], "gpu_processes": [],
                "listening_ports": sorted(set(ports) & set(listeners.get(host, [])))}

    monkeypatch.setattr(controller, "remote", remote)
    monkeypatch.setattr(controller, "fence_nodes", lambda generation: [])
    monkeypatch.setattr(controller, "stop_yue", lambda **kwargs: None)
    return calls


def make_controller(tmp_path, cfg=None):
    return Controller(cfg or config(), lambda *args, **kwargs: pytest.fail("unexpected SSH"), directory=tmp_path)


def test_audit_payload_includes_target_nim_ports_on_the_correct_hosts(monkeypatch, tmp_path):
    cfg = config()
    cfg["models"]["glm53"]["nim"].update(manager_port=22000, worker_port=8202)
    controller = make_controller(tmp_path, cfg)
    calls = install_audit_transport(controller, monkeypatch)
    controller.operation_target = "glm53"
    controller.audit("head")
    controller.audit("worker")
    assert calls == [("head", [8000, 8001, 8011, 22000]), ("worker", [8000, 8001, 8011, 8202, 22000])]


@pytest.mark.parametrize("host,port", [("head", 20000), ("worker", 20000), ("worker", 8002), ("head", 8001), ("worker", 8001)])
def test_nim_port_conflict_prevents_start_callback(monkeypatch, tmp_path, host, port):
    controller = make_controller(tmp_path)
    calls = install_audit_transport(controller, monkeypatch, {host: [port]})
    start = Mock()
    with pytest.raises(ControllerError, match="unmanaged listener remains"):
        controller.switch("glm53", start)
    start.assert_not_called()
    assert any(observed_host == host and port in ports for observed_host, ports in calls)
    assert controller.operation_target is None and controller.operation_hosts is None
    persisted = json.loads((tmp_path / "controller.json").read_text())
    assert persisted["phase"] == "failed"
    assert str(port) in persisted["error"]


def test_unused_catalog_nim_ports_do_not_block_unrelated_solo_start(monkeypatch, tmp_path):
    controller = make_controller(tmp_path)
    calls = install_audit_transport(controller, monkeypatch, {"head": [20000, 8002], "worker": [20000, 8002]})
    start = Mock(return_value={"containers": [{"host": "head", "rank": 0, "id": "a" * 64}], "served": "solo"})
    controller.switch("solo", start, hosts=["head"])
    start.assert_called_once()
    assert calls and all(host == "head" and ports == [8000, 8011] for host, ports in calls)
    assert controller.operation_target is None


def test_stop_retains_departing_nim_ports_until_idle_then_releases_them(monkeypatch, tmp_path):
    controller = make_controller(tmp_path)
    controller.save(mode="vllm", model="glm53", phase="ready", generation="old")
    calls = install_audit_transport(controller, monkeypatch)
    controller.switch("none")
    assert calls and all(20000 in ports for _, ports in calls)
    assert all((8002 in ports) == (host == "worker") for host, ports in calls)
    assert controller.operation_target is None
    calls.clear()
    controller.audit("head")
    controller.audit("worker")
    assert calls == [("head", [8000, 8011]), ("worker", [8000, 8011])]


def test_orphaned_worker_listener_prevents_successful_nim_stop(monkeypatch, tmp_path):
    controller = make_controller(tmp_path)
    controller.save(mode="vllm", model="glm53", phase="ready", generation="old")
    calls = install_audit_transport(controller, monkeypatch, {"worker": [8002]})
    with pytest.raises(ControllerError, match="8002"):
        controller.switch("none")
    assert ("worker", [8000, 8001, 8002, 8011, 20000]) in calls
    assert controller.node_states()["worker"]["model"] == "glm53"
    assert controller.node_states()["worker"]["phase"] == "failed"
    assert controller.operation_target is None


def test_replacing_nim_checks_old_ports_as_well_as_new_workload(monkeypatch, tmp_path):
    controller = make_controller(tmp_path)
    controller.save(mode="vllm", model="glm53", phase="ready", generation="old")
    calls = install_audit_transport(controller, monkeypatch, {"head": [20000]})
    start = Mock()
    with pytest.raises(ControllerError, match="20000"):
        controller.switch("solo", start, hosts=["head", "worker"])
    start.assert_not_called()
    assert all(20000 in ports for _, ports in calls)


def test_backend_port_override_is_checked_on_both_hosts(monkeypatch, tmp_path):
    cfg = config()
    cfg["models"]["glm53"]["env"] = {"NIM_BACKEND_PORT": "8101"}
    controller = make_controller(tmp_path, cfg)
    calls = install_audit_transport(controller, monkeypatch)
    controller.operation_target = "glm53"
    controller.audit("head")
    controller.audit("worker")
    assert all(8101 in ports and 8001 not in ports for _, ports in calls)


@pytest.mark.parametrize("address", ["192.168.100.10:20000", "[fd00::10]:20000"])
def test_actual_audit_payload_detects_non_loopback_bound_listeners(monkeypatch, tmp_path, capsys, address):
    controller = make_controller(tmp_path)
    controller.operation_target = "glm53"
    captured = []
    monkeypatch.setattr(controller, "remote", lambda host, script, **kwargs: captured.append(script) or {})
    controller.audit("head")
    argv = shlex.split(captured[0].partition("\n")[2])

    def run(args, **kwargs):
        if args == ["docker", "ps", "-aq"] or args[0] == "nvidia-smi":
            return subprocess.CompletedProcess(args, 0, "", "")
        assert args == ["ss", "-H", "-ltn"]
        return subprocess.CompletedProcess(args, 0,
            f"LISTEN 0 4096 {address} *:*\nLISTEN 0 128 0.0.0.0:22 0.0.0.0:*\n", "")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["audit", argv[-1]])
    exec(compile(argv[2], "<audit-payload>", "exec"), {})
    result = json.loads(capsys.readouterr().out)
    assert result["listening_ports"] == [20000]
