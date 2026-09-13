"""Offline ownership tests for independent Spark transitions."""

import copy
import hashlib
import json
from unittest.mock import Mock

import pytest

from spark_serve_controller import Controller, ControllerError, atomic_json, read_state
from test_controller import FakeController, config


class NodeController(FakeController):
    def __init__(self, directory):
        cfg = config()
        cfg["models"]["second"] = {**cfg["models"]["solo"], "served_name": "second"}
        super().__init__(directory, cfg)
        self.fences = {"head": "head-old", "worker": "worker-old"}

    def fence_nodes(self, generation):
        errors = Controller.fence_nodes(self, generation)
        if not errors:
            for worker in self._workers():
                self.fences[worker["host"]] = generation
        return errors

    def remote(self, host, script, **kwargs):
        if host in self.offline:
            self.calls.append(("remote", host, script))
            raise ControllerError(host + ": offline")
        return super().remote(host, script, **kwargs)

    def launch(self, host, *, fail=False):
        def start(generation):
            self.calls.append(("launch", host, generation))
            self.containers[host].append({
                "id": host + "-" + generation, "name": "vllm_cluster",
                "running": True, "gpu": True,
                "labels": {"ai.spark-serve.hosts": json.dumps([host]),
                           "ai.spark-serve.allocation": generation},
            })
            if fail:
                raise ControllerError("model startup failed")
        return start


@pytest.fixture
def controller(tmp_path):
    return NodeController(tmp_path)


def legacy_solo(c):
    c.save(mode="vllm", model="solo", phase="ready", generation="legacy-solo")
    c.containers["head"].append({
        "id": "original-head", "name": "vllm_cluster", "running": True, "gpu": True,
    })


def assert_only_host(c, host):
    assert c.calls
    assert all(call[1] == host for call in c.calls), c.calls


def test_legacy_solo_migration_is_read_only_and_keeps_actual_host_identity(controller):
    c = controller
    legacy_solo(c)
    before = (c.directory / "controller.json").read_bytes()
    nodes = c.node_states()
    assert nodes["head"]["model"] == "solo"
    assert nodes["head"]["generation"] == "legacy-solo"
    assert nodes["head"]["allocation_hosts"] == ["head"]
    assert nodes["worker"]["mode"] == "none"
    assert nodes["worker"]["phase"] == "stopped"
    assert (c.directory / "controller.json").read_bytes() == before
    assert not c.calls


def test_start_second_node_preserves_live_offline_peer_and_generation(controller):
    c = controller
    legacy_solo(c)
    original = c.node_states()["head"]
    containers = copy.deepcopy(c.containers["head"])
    c.offline.add("head")
    c.switch("second", c.launch("worker"), hosts=["worker"])
    assert_only_host(c, "worker")
    assert c.node_states()["head"] == original
    assert c.containers["head"] == containers
    assert c.fences["head"] == "head-old"
    assert c.node_states()["worker"]["generation"] == c.fences["worker"]
    assert c.node_states()["worker"]["phase"] == "ready"
    assert read_state(c.directory)["version"] == 1


def test_selected_restart_and_stop_preserve_peer_error_and_receipts(controller):
    c = controller
    legacy_solo(c)
    c.switch("second", c.launch("worker"), hosts=["worker"])
    nodes = c.node_states()
    nodes["head"].update(error="previous diagnostic", cleanup_errors=["retain me"], extra_receipt={"id": "head-proof"})
    c.save(nodes=nodes)
    peer = c.node_states()["head"]
    c.calls.clear()
    c.switch("second", c.launch("worker"), hosts=["worker"])
    assert_only_host(c, "worker")
    assert c.node_states()["head"] == peer
    c.calls.clear()
    c.switch("none", hosts=["worker"])
    assert_only_host(c, "worker")
    assert c.node_states()["head"] == peer
    assert c.node_states()["worker"]["mode"] == "none"
    assert c.node_states()["worker"]["allocation_hosts"] == []


def test_failed_start_cleanup_is_scoped_to_selected_host(controller):
    c = controller
    legacy_solo(c)
    peer = c.node_states()["head"]
    containers = copy.deepcopy(c.containers["head"])
    with pytest.raises(ControllerError, match="startup failed"):
        c.switch("second", c.launch("worker", fail=True), hosts=["worker"])
    assert_only_host(c, "worker")
    assert c.node_states()["head"] == peer
    assert c.containers["head"] == containers
    assert all(item["name"] == "neo4j" for item in c.containers["worker"])
    assert c.node_states()["worker"]["phase"] == "failed"
    assert "startup failed" in c.node_states()["worker"]["error"]
    assert c.operation_hosts is None and c.operation_generation is None


def test_selected_host_offline_fails_before_state_or_peer_mutation(controller):
    c = controller
    legacy_solo(c)
    before = (c.directory / "controller.json").read_bytes()
    c.offline.add("worker")
    with pytest.raises(ControllerError, match="offline"):
        c.switch("second", c.launch("worker"), hosts=["worker"])
    assert_only_host(c, "worker")
    assert (c.directory / "controller.json").read_bytes() == before
    assert all(call[0] == "audit" for call in c.calls)


@pytest.mark.parametrize("target", ["none", "solo"])
def test_persisted_distributed_allocation_rejects_single_scope_before_mutation(controller, target):
    c = controller
    c.save(mode="vllm", model="deepseek", phase="ready", generation="distributed")
    before = (c.directory / "controller.json").read_bytes()
    launch = Mock()
    with pytest.raises(ControllerError, match="both"):
        c.switch(target, launch, hosts=["worker"])
    assert not c.calls
    launch.assert_not_called()
    assert (c.directory / "controller.json").read_bytes() == before


@pytest.mark.parametrize("container_fields", [
    {},
    {"labels": {"ai.spark-serve.hosts": '["head", "worker"]'}},
    {"labels": {"ai.spark-serve.hosts": "invalid"}},
    {"command": ["vllm", "--nnodes", "2"]},
    {"command": ["vllm", "--headless"]},
])
def test_unknown_or_distributed_rank_is_not_stopped_with_single_scope(controller, container_fields):
    c = controller
    c.containers["worker"].append({
        "id": "orphan", "name": "vllm_cluster", "running": True, "gpu": True,
        **container_fields,
    })
    with pytest.raises(ControllerError, match="both"):
        c.switch("none", hosts=["worker"])
    assert c.calls == [("audit", "worker")]
    assert not (c.directory / "controller.json").exists()
    assert any(item["id"] == "orphan" for item in c.containers["worker"])


def test_solo_allocation_label_can_reconcile_lost_local_state(controller):
    c = controller
    c.containers["worker"].append({
        "id": "solo-orphan", "name": "vllm_cluster", "running": True, "gpu": True,
        "labels": {"ai.spark-serve.hosts": '["worker"]'},
    })
    c.switch("none", hosts=["worker"])
    assert_only_host(c, "worker")
    assert all(item["name"] == "neo4j" for item in c.containers["worker"])


def test_both_node_transition_replaces_split_layout_with_shared_allocation(controller):
    c = controller
    legacy_solo(c)
    c.switch("second", c.launch("worker"), hosts=["worker"])
    c.calls.clear()

    def distributed(generation):
        for host in ("head", "worker"):
            c.launch(host)(generation)

    c.switch("deepseek", distributed)
    nodes = c.node_states()
    assert {call[1] for call in c.calls} == {"head", "worker"}
    assert nodes["head"]["allocation_id"] == nodes["worker"]["allocation_id"]
    assert all(record["allocation_hosts"] == ["head", "worker"] for record in nodes.values())
    assert all(record["model"] == "deepseek" for record in nodes.values())
    assert all(record["phase"] == "ready" for record in nodes.values())
    c.switch("none")
    assert all(record["mode"] == "none" for record in c.node_states().values())


@pytest.mark.parametrize("target", ["none", "solo"])
def test_scoped_change_preserves_peer_yue_discovery_and_admission(controller, target):
    c = controller
    c.switch("yue")
    original = c.discovery()
    peer = c.node_states()["worker"]
    admission = copy.deepcopy(c.nodes["worker"])
    c.calls.clear()
    c.offline.add("worker")
    c.switch(target, c.launch("head"), hosts=["head"])
    assert_only_host(c, "head")
    assert c.node_states()["worker"] == peer
    assert c.nodes["worker"] == admission
    assert c.discovery() == {**original, "workers": [original["workers"][1]]}


def test_scoped_failed_start_preserves_peer_yue_discovery(controller):
    c = controller
    c.switch("yue")
    original = c.discovery()
    peer = c.node_states()["worker"]
    c.calls.clear()
    with pytest.raises(ControllerError, match="startup failed"):
        c.switch("solo", c.launch("head", fail=True), hosts=["head"])
    assert_only_host(c, "head")
    assert c.node_states()["worker"] == peer
    assert c.discovery() == {**original, "workers": [original["workers"][1]]}


def test_single_yue_start_and_distributed_target_reject_before_mutation(controller):
    for target in ("yue", "deepseek"):
        with pytest.raises(ControllerError, match="both"):
            controller.switch(target, Mock(), hosts=["head"])
    assert not controller.calls
    assert not (controller.directory / "controller.json").exists()


@pytest.mark.parametrize("hosts", [[], ["elsewhere"], ["head", "head"], "head"])
def test_invalid_host_scope_rejects_without_mutation(controller, hosts):
    with pytest.raises(ControllerError, match="Select"):
        controller.switch("none", hosts=hosts)
    assert not controller.calls
    assert not (controller.directory / "controller.json").exists()


@pytest.mark.parametrize("phase", ["starting", "failed"])
def test_interrupted_legacy_distributed_launch_requires_both_nodes(controller, phase):
    c = controller
    # Legacy launch state recorded mode=none until readiness, even after rank 1
    # had launched. Missing containers on one node cannot prove a solo workload.
    c.save(mode="none", target="deepseek", phase=phase, generation="interrupted")
    with pytest.raises(ControllerError, match="both"):
        c.switch("solo", c.launch("head"), hosts=["head"])
    assert not c.calls


def test_legacy_both_scope_solo_start_records_only_head_allocation(controller):
    c = controller
    c.switch("solo", c.launch("head"))
    nodes = c.node_states()
    assert nodes["head"]["allocation_hosts"] == ["head"]
    assert nodes["head"]["mode"] == "vllm"
    assert nodes["worker"]["mode"] == "none"
    assert nodes["worker"]["allocation_hosts"] == []
    assert {call[1] for call in c.calls} == {"head", "worker"}


def test_peer_yue_remains_ready_in_mixed_status_using_its_original_generation(controller):
    c = controller
    c.switch("yue")
    c.switch("solo", c.launch("head"), hosts=["head"])
    status = c.status()
    assert status["mode"] == "mixed"
    assert status["ready_workers"] == 1
    peer = next(worker for worker in status["yue_workers"] if worker["host"] == "worker")
    assert peer["ready"]
    c.nodes["worker"]["generation"] = "stale"
    assert c.status()["ready_workers"] == 0


def test_legacy_yue_receipts_migrate_without_changing_published_generation(controller):
    c = controller
    c.switch("yue")
    original = c.discovery()
    legacy = read_state(c.directory)
    legacy.pop("nodes")
    atomic_json(c.directory / "controller.json", legacy)
    c.calls.clear()
    c.switch("none", hosts=["head"])
    assert_only_host(c, "head")
    assert c.discovery() == {**original, "workers": [original["workers"][1]]}


def test_remote_scope_guard_rejects_peer_even_if_called_accidentally(controller):
    controller.operation_hosts = ["head"]
    controller.operation_generation = "new-head"
    controller.ssh = Mock()
    with pytest.raises(ControllerError, match="restricted"):
        Controller.remote(controller, "worker", "true")
    controller.ssh.assert_not_called()


def test_generations_change_only_for_selected_node_on_repeated_starts(controller):
    c = controller
    legacy_solo(c)
    c.switch("second", c.launch("worker"), hosts=["worker"])
    first = c.node_states()
    c.switch("second", c.launch("worker"), hosts=["worker"])
    second = c.node_states()
    assert first["head"] == second["head"]
    assert first["worker"]["generation"] != second["worker"]["generation"]
    assert c.fences["worker"] == second["worker"]["generation"]
    assert c.fences["head"] == "head-old"


def receipt_launch(c, hosts, served):
    def start(generation):
        records = []
        for rank, host in enumerate(hosts):
            c.launch(host)(generation)
            identity = hashlib.sha256((host + generation).encode()).hexdigest()
            c.containers[host][-1]["id"] = identity
            records.append({"host": host, "rank": rank, "id": identity})
        return {"containers": records, "served": served}
    return start


def test_new_assignment_persists_immutable_receipt_and_preserves_legacy_peer(controller):
    c = controller
    legacy_solo(c)
    nodes = c.node_states()
    assert nodes["head"]["legacy"] is True
    nodes["head"]["containers"] = [{"host": "head", "rank": 0, "id": "a" * 64}]
    c.save(nodes=nodes)
    peer = copy.deepcopy(nodes["head"])
    c.switch("second", receipt_launch(c, ["worker"], "second"), hosts=["worker"])
    current = c.node_states()
    assert current["head"] == peer
    assert current["worker"]["legacy"] is False
    assert current["worker"]["allocation_id"] == current["worker"]["generation"]
    assert current["worker"]["containers"] == [{
        "host": "worker", "rank": 0, "id": c.containers["worker"][-1]["id"],
    }]
    assert current["worker"]["served"] == "second"


def test_replacement_and_stop_remove_old_container_receipts(controller):
    c = controller
    legacy_solo(c)
    c.switch("second", receipt_launch(c, ["worker"], "second"), hosts=["worker"])
    first = c.node_states()["worker"]["containers"]
    c.switch("second", receipt_launch(c, ["worker"], "second"), hosts=["worker"])
    assert c.node_states()["worker"]["containers"] != first
    c.switch("none", hosts=["worker"])
    assert "containers" not in c.node_states()["worker"]
    assert "served" not in c.node_states()["worker"]


def test_distributed_receipts_are_partitioned_by_physical_host(controller):
    c = controller
    c.switch("deepseek", receipt_launch(c, ["head", "worker"], "deepseek"))
    nodes = c.node_states()
    for rank, host in enumerate(("head", "worker")):
        assert nodes[host]["containers"] == [{
            "host": host, "rank": rank, "id": c.containers[host][-1]["id"],
        }]
        assert nodes[host]["legacy"] is False
    assert nodes["head"]["allocation_id"] == nodes["worker"]["allocation_id"]


@pytest.mark.parametrize("bad", [
    {"containers": [], "served": "second"},
    {"containers": [{"host": "head", "rank": 0, "id": "a" * 64}], "served": "second"},
    {"containers": [{"host": "worker", "rank": 1, "id": "a" * 64}], "served": "second"},
    {"containers": [{"host": "worker", "rank": 0, "id": "short"}], "served": "second"},
    {"containers": [{"host": "worker", "rank": 0, "id": "a" * 64}], "served": "wrong"},
])
def test_invalid_launch_receipt_fails_and_cleans_only_selected_node(controller, bad):
    c = controller
    legacy_solo(c)
    peer = c.node_states()["head"]

    def start(generation):
        c.launch("worker")(generation)
        return bad

    with pytest.raises(ControllerError, match="receipt"):
        c.switch("second", start, hosts=["worker"])
    assert_only_host(c, "worker")
    assert c.node_states()["head"] == peer
    assert all(item["name"] == "neo4j" for item in c.containers["worker"])


def test_no_wait_launch_retains_container_receipt_without_claiming_readiness(controller):
    c = controller
    c.switch("solo", receipt_launch(c, ["head"], None), hosts=["head"], no_wait=True)
    record = c.node_states()["head"]
    assert record["phase"] == "starting"
    assert record["containers"][0]["id"] == c.containers["head"][-1]["id"]
    assert record["legacy"] is False


@pytest.mark.parametrize("contents", ["{broken", "[]", '{"version":99}'])
def test_corrupt_legacy_state_requires_both_even_when_selected_node_is_empty(controller, contents):
    c = controller
    path = c.directory / "controller.json"
    path.write_text(contents)
    with pytest.raises(ControllerError, match="both"):
        c.switch("solo", c.launch("head"), hosts=["head"])
    assert not c.calls
    assert path.read_text() == contents


@pytest.mark.parametrize("missing", [False, True])
def test_unknown_peer_assignment_requires_both_without_contacting_selected_node(controller, missing):
    c = controller
    legacy_solo(c)
    c.containers["head"] = [item for item in c.containers["head"] if item["name"] == "neo4j"]
    nodes = c.node_states()
    if missing:
        nodes.pop("worker")
    else:
        nodes["worker"].update(mode="unknown", phase="failed", error="ownership record could not be read", allocation_hosts=[])
    c.save(nodes=nodes)
    before = (c.directory / "controller.json").read_bytes()
    with pytest.raises(ControllerError, match="ownership is unknown.*both"):
        c.switch("solo", c.launch("head"), hosts=["head"])
    assert not c.calls
    assert (c.directory / "controller.json").read_bytes() == before


def test_fresh_install_can_start_each_node_independently(controller):
    c = controller
    assert not (c.directory / "controller.json").exists()
    c.switch("solo", receipt_launch(c, ["head"], "solo"), hosts=["head"])
    assert_only_host(c, "head")
    peer = c.node_states()["head"]
    c.calls.clear()
    c.switch("second", receipt_launch(c, ["worker"], "second"), hosts=["worker"])
    assert_only_host(c, "worker")
    assert c.node_states()["head"] == peer


def test_both_scope_can_reconcile_corrupt_legacy_state(controller):
    c = controller
    (c.directory / "controller.json").write_text("{broken")
    c.switch("none")
    assert {call[1] for call in c.calls} == {"head", "worker"}
    assert all(record["mode"] == "none" for record in c.node_states().values())
