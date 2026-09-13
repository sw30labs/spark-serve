"""Offline fault-injection tests. Never connects to real SSH, Docker, or HTTP."""

import copy
import importlib.machinery
import importlib.util
import json
import os
import shlex
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from spark_serve_controller import (
    Controller,
    ControllerError,
    guarded_remote_script,
    health_ready,
    read_state,
    yue_profile,
)


def config():
    return {
        "cluster": {
            "head": "head",
            "worker": "worker",
            "lan_url": "http://head.lan:8000",
            "port": 8000,
            "container": "vllm_cluster",
            "stop_names": ["vllm_cluster", "neo4j"],
            "keep_containers": ["neo4j"],
            "nnodes": 2,
            "master_addr": "192.168.100.10",
            "master_port": 29501,
            "tensor_parallel": 2,
            "hf_cache_host": "/cache",
            "nccl": {"NCCL_NET": "IB"},
        },
        "models": {
            "deepseek": {
                "image": "deepseek",
                "served_name": "deepseek",
                "hf_id": "deepseek",
                "wrapper": "dsv4",
            },
            "solo": {
                "image": "solo",
                "served_name": "solo",
                "hf_id": "solo",
                "nnodes": 1,
                "tensor_parallel": 1,
            },
        },
    }


def healthy(worker, generation="old", accepting=True):
    return {
        "service": "yue-icl-factory",
        "api_version": 2,
        "cuda": True,
        "runtime_ok": True,
        "assets_ok": True,
        "ownership_ok": True,
        "worker_id": worker,
        "runtime_manifest": "a" * 64,
        "generation": generation,
        "accepting": accepting,
        "mock": False,
        "busy": False,
        "active_job": None,
        "owned_containers": [],
    }


class FakeController(Controller):
    def __init__(self, directory, cfg=None):
        super().__init__(cfg or config(), lambda *a, **k: None, directory=directory)
        self.calls = []
        self.nodes = {host: healthy(host) for host in ("head", "worker")}
        self.containers = {
            host: [
                {
                    "name": "neo4j",
                    "id": "protected-" + host,
                    "running": True,
                    "gpu": False,
                }
            ]
            for host in self.nodes
        }
        self.compute = {host: [] for host in self.nodes}
        self.listeners = {host: [] for host in self.nodes}
        self.offline = set()
        self.bad_health = {}
        self.service_states = {host: "active" for host in self.nodes}
        self.started = []

    def fence_nodes(self, generation):
        self.calls.append(("fence", generation))
        return []

    def control(self, host, action, **options):
        self.calls.append((action, host, options))
        if host in self.offline:
            raise ControllerError(host + ": offline; ownership unknown")
        node = self.nodes[host]
        if action == "drain":
            node["accepting"] = False
        elif action == "admit":
            node.update(accepting=True, generation=options["generation"])
        elif action == "cancel":
            node.update(busy=False, active_job=None, owned_containers=[])
        return copy.deepcopy(node)

    def audit(self, host):
        self.calls.append(("audit", host))
        if host in self.offline:
            raise ControllerError(host + ": offline")
        return {
            "containers": copy.deepcopy(self.containers[host]),
            "listening_ports": self.listeners[host],
            "gpu_processes": self.compute[host],
        }

    def remote(self, host, script, **kwargs):
        self.calls.append(("remote", host, script))
        if "docker rm -f" in script:
            ids = script.split("docker rm -f ", 1)[1].split()
            self.containers[host] = [
                c for c in self.containers[host] if c["id"] not in ids
            ]
        if "ActiveState" in script:
            return self.service_states[host]
        return ""

    def service(self, host, action):
        self.calls.append(("service", host, action))
        self.service_states[host] = "inactive" if action == "stop" else "active"
        if action == "restart":
            self.nodes[host]["accepting"] = False

    def health(self, worker):
        node = copy.deepcopy(self.nodes[worker["host"]])
        # Startup must still acknowledge drained; fault only after admission.
        if node["accepting"]:
            node.update(self.bad_health.get(worker["host"], {}))
        return node

    def discovery(self):
        return json.loads((self.directory / "yue-workers.json").read_text())


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.c = FakeController(self.directory)

    def test_two_workers_published_only_after_validated_admission(self):
        self.c.switch("yue")
        discovery = self.c.discovery()
        self.assertEqual("yue", discovery["mode"])
        self.assertEqual(["head", "worker"], [w["id"] for w in discovery["workers"]])
        self.assertTrue(
            all(
                w["generation"] == discovery["generation"] for w in discovery["workers"]
            )
        )
        self.assertEqual("ready", read_state(self.directory)["phase"])
        self.assertEqual(2, self.c.status()["ready_workers"])
        self.assertTrue(
            all(
                c["name"] == "neo4j"
                for nodes in self.c.containers.values()
                for c in nodes
            )
        )

    def test_drain_pending_preserves_active_jobs_and_blocks_model_start(self):
        self.c.nodes["head"].update(busy=True, active_job={"job_id": "song-1"})
        started = []
        with self.assertRaisesRegex(ControllerError, "draining active jobs.*song-1"):
            self.c.switch("deepseek", lambda generation: started.append(True))
        self.assertFalse(started)
        self.assertEqual("song-1", self.c.nodes["head"]["active_job"]["job_id"])
        self.assertFalse(any(n["accepting"] for n in self.c.nodes.values()))
        self.assertFalse(any(call[0] == "cancel" for call in self.c.calls))
        self.assertEqual([], self.c.discovery()["workers"])

    def test_explicit_cancellation_then_switch(self):
        for node in self.c.nodes.values():
            node.update(busy=True, active_job={"job_id": "song-1"})
        self.c.switch(
            "deepseek", lambda generation: self.c.started.append(True), cancel_jobs=True
        )
        self.assertEqual([True], self.c.started)
        self.assertEqual(2, len([c for c in self.c.calls if c[0] == "cancel"]))
        self.assertEqual("vllm", read_state(self.directory)["mode"])
        self.assertEqual([], self.c.discovery()["workers"])

    def test_offline_node_blocks_transition_but_drains_other(self):
        self.c.offline.add("head")
        with self.assertRaisesRegex(ControllerError, "offline"):
            self.c.switch("deepseek", lambda generation: self.c.started.append(True))
        self.assertFalse(self.c.nodes["worker"]["accepting"])
        self.assertFalse(self.c.started)
        self.assertFalse(any(call[0] == "service" for call in self.c.calls))

    def test_unknown_busy_or_orphan_ownership_never_releases_gpu(self):
        for damage in (
            {"busy": True},
            {"ownership_ok": False},
            {"owned_containers": [{"running": True}]},
        ):
            with self.subTest(damage=damage):
                self.c.nodes["head"] = healthy("head")
                self.c.nodes["head"].update(damage)
                with self.assertRaisesRegex(ControllerError, "ownership is unresolved"):
                    self.c.switch(
                        "deepseek", lambda generation: self.c.started.append(True)
                    )
                self.assertFalse(self.c.started)

    def test_partial_yue_start_failure_drains_and_stops_first(self):
        self.c.bad_health["worker"] = {"runtime_ok": False}
        with self.assertRaisesRegex(ControllerError, "validation failed"):
            self.c.switch("yue")
        self.assertTrue(all(s == "inactive" for s in self.c.service_states.values()))
        self.assertTrue(all(not n["accepting"] for n in self.c.nodes.values()))
        self.assertEqual([], self.c.discovery()["workers"])
        self.assertEqual("failed", read_state(self.directory)["phase"])

    def test_stale_generation_never_published(self):
        self.c.bad_health["worker"] = {"generation": "previous-controller"}
        with self.assertRaises(ControllerError):
            self.c.switch("yue")
        self.assertEqual([], self.c.discovery()["workers"])

    def test_wrong_http_endpoint_never_published(self):
        self.c.bad_health["worker"] = {"worker_id": "a-different-server"}
        with self.assertRaisesRegex(ControllerError, "same runtime"):
            self.c.switch("yue")
        self.assertEqual([], self.c.discovery()["workers"])

    def test_two_endpoints_cannot_be_same_physical_worker(self):
        self.c.nodes["worker"]["worker_id"] = "head"
        with self.assertRaisesRegex(ControllerError, "same worker"):
            self.c.switch("yue")
        self.assertEqual([], self.c.discovery()["workers"])

    def test_foreign_gpu_container_is_not_killed_or_overlapped(self):
        self.c.containers["head"].append(
            {"id": "foreign", "name": "research", "running": True, "gpu": True}
        )
        with self.assertRaisesRegex(ControllerError, "GPU still in use.*research"):
            self.c.switch("yue")
        self.assertTrue(any(c["id"] == "foreign" for c in self.c.containers["head"]))
        self.assertFalse(any(call[0] == "admit" for call in self.c.calls))

    def test_idle_unmanaged_factory_blocks_switch_without_being_killed(self):
        self.c.listeners["worker"] = [8011]
        with self.assertRaisesRegex(ControllerError, "unmanaged listener.*8011"):
            self.c.switch("deepseek", lambda generation: self.c.started.append(True))
        self.assertFalse(self.c.started)
        self.assertEqual([8011], self.c.listeners["worker"])

    def test_host_compute_process_blocks_admission(self):
        self.c.compute["worker"] = ["123, python"]
        with self.assertRaisesRegex(ControllerError, "GPU still in use"):
            self.c.switch("yue")
        self.assertFalse(any(call[0] == "admit" for call in self.c.calls))

    def test_only_exact_catalog_container_ids_removed(self):
        self.c.containers["head"].extend(
            [
                {"id": "ours", "name": "vllm_cluster", "running": True, "gpu": True},
                {
                    "id": "unrelated",
                    "name": "vllm_experiment",
                    "running": False,
                    "gpu": True,
                },
            ]
        )
        self.c.switch("none")
        self.assertEqual(
            {"protected-head", "unrelated"},
            {c["id"] for c in self.c.containers["head"]},
        )
        self.assertEqual("none", read_state(self.directory)["mode"])

    def test_partial_vllm_start_failure_cleans_owned_worker(self):
        def launch(generation):
            self.c.containers["worker"].append(
                {"id": "rank1", "name": "vllm_cluster", "running": True, "gpu": True}
            )
            raise ControllerError("head failed")

        with self.assertRaisesRegex(ControllerError, "head failed"):
            self.c.switch("deepseek", launch)
        self.assertFalse(any(c["id"] == "rank1" for c in self.c.containers["worker"]))
        self.assertEqual("failed", read_state(self.directory)["phase"])

    def test_cli_and_gui_share_nonblocking_process_lock(self):
        another = FakeController(self.directory)
        with (
            self.c.lock(),
            self.assertRaisesRegex(ControllerError, "another Spark Serve"),
        ):
            another.switch("yue")
        self.assertFalse(another.calls)

    def test_interrupted_transition_is_reconciled_on_next_command(self):
        self.c.save(mode="yue", phase="starting", generation="interrupted")
        self.c.nodes["head"].update(accepting=True, generation="interrupted")
        self.c.switch("yue")
        self.assertNotEqual("interrupted", self.c.discovery()["generation"])
        self.assertEqual("fence", self.c.calls[0][0])
        self.assertEqual("drain", self.c.calls[1][0])

    def test_readiness_revoked_after_worker_reboot(self):
        self.c.switch("yue")
        self.c.nodes["head"].update(accepting=False, runtime_ok=False)
        status = self.c.status()
        self.assertEqual(1, status["ready_workers"])

    def test_after_idle_runs_under_lock_after_workloads_stop(self):
        ran = []

        def after_idle():
            ran.append(read_state(self.directory)["mode"])
            with self.assertRaisesRegex(ControllerError, "another Spark Serve"):
                FakeController(self.directory).switch("yue")

        self.c.switch("none", after_idle=after_idle)
        self.assertEqual(["none"], ran)
        self.assertEqual("stopped", read_state(self.directory)["phase"])

    def test_after_idle_failure_marks_transition_failed(self):
        def boom():
            raise ControllerError("reboot failed")

        with self.assertRaisesRegex(ControllerError, "reboot failed"):
            self.c.switch("none", after_idle=boom)
        self.assertEqual("failed", read_state(self.directory)["phase"])
        self.assertEqual("reboot failed", read_state(self.directory)["error"])

    def test_configuration_rejects_typos_and_credential_urls(self):
        for override in (
            {"typo": 1},
            {"port": 0},
            {"head_url": "http://user:secret@host"},
        ):
            cfg = config()
            cfg["yue"] = override
            with self.assertRaises(ControllerError):
                yue_profile(cfg)

    def test_unhealthy_200_and_mock_cannot_pass(self):
        node = healthy("worker", "generation")
        self.assertTrue(health_ready(node, "generation"))
        for damage in (
            {"mock": True},
            {"cuda": False},
            {"service": "vllm"},
            {"assets_ok": False},
            {"runtime_ok": False},
            {"generation": "stale"},
            {"runtime_manifest": ""},
            {"ownership_ok": False},
        ):
            self.assertFalse(health_ready({**node, **damage}, "generation"))

    def test_legacy_factory_is_never_executed_by_control(self):
        factory = self.directory / "yue_factory.py"
        effect = self.directory / "legacy-started"
        factory.write_text(f"from pathlib import Path\nPath({str(effect)!r}).touch()\n")
        cfg = config()
        cfg["yue"] = {"factory_root": str(self.directory)}

        def local_ssh(cfg, host, script, check=False):
            return subprocess.run(
                ["bash", "-s"],
                input=script,
                text=True,
                capture_output=True,
                check=False,
            )

        controller = Controller(cfg, local_ssh, directory=self.directory)
        with self.assertRaisesRegex(ControllerError, "legacy source was not executed"):
            controller.control("head", "status")
        self.assertFalse(effect.exists())


loader = importlib.machinery.SourceFileLoader(
    "spark_serve_cli_test", str(Path(__file__).parents[1] / "spark-serve")
)
spec = importlib.util.spec_from_loader(loader.name, loader)
cli = importlib.util.module_from_spec(spec)
loader.exec_module(cli)


class CLITests(unittest.TestCase):
    def status(self, served, worker=False):
        container = {"name": "vllm_cluster", "running": True}
        managed = {"mode": "vllm", "phase": "ready"}
        with (
            patch.object(
                cli,
                "status_host_json",
                side_effect=lambda cfg, host: (
                    [container] if host == "head" or worker else []
                ),
            ),
            patch.object(
                cli, "probe_v1", return_value=json.dumps({"data": [{"id": served}]})
            ),
            patch.object(cli.Controller, "status", return_value=managed),
        ):
            return cli.collect_status(config())

    def test_head_only_cannot_claim_distributed_model_ready(self):
        self.assertFalse(self.status("deepseek")["ready"])
        self.assertTrue(self.status("deepseek", worker=True)["ready"])

    def test_solo_recipe_remains_ready_with_only_head(self):
        self.assertTrue(self.status("solo")["ready"])

    def test_distributed_and_solo_recipe_flags_preserved(self):
        cfg = config()
        distributed = cli.docker_run_script(cfg, cfg["models"]["deepseek"], 1, True)
        solo = cli.docker_run_script(cfg, cfg["models"]["solo"], 0, False)
        self.assertIn("--nnodes 2 --node-rank 1", distributed)
        self.assertIn("--headless", distributed)
        self.assertIn("NCCL_NET=IB", distributed)
        self.assertIn("--tensor-parallel-size 1 --nnodes 1", solo)
        self.assertNotIn("--headless", solo)

    def test_invalid_model_is_rejected_before_any_transition(self):
        with (
            patch.object(cli.Controller, "switch") as switch,
            self.assertRaises(SystemExit),
        ):
            cli.cmd_up(config(), "typo", False, False, False)
        switch.assert_not_called()

    def test_yue_never_routes_to_vllm_or_hermes(self):
        with (
            patch.object(cli.Controller, "switch") as switch,
            patch.object(cli, "_bring_up") as vllm,
        ):
            cli.cmd_up(config(), "yue", False, False, False)
        self.assertEqual("yue", switch.call_args.args[0])
        vllm.assert_not_called()

    def test_ssh_timeout_is_reported_as_unknown(self):
        with patch.object(
            cli.subprocess, "run", side_effect=subprocess.TimeoutExpired("ssh", 660)
        ):
            result = cli.ssh_cmd(config(), "head", "true", check=False)
        self.assertEqual(124, result.returncode)
        self.assertIn("reconciliation", result.stderr)


class RemoteGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.env = {**os.environ, "SPARK_SERVE_NODE_STATE_DIR": self.temp.name}

    def run_script(self, script):
        return subprocess.run(
            ["bash", "-s"],
            input=script,
            text=True,
            capture_output=True,
            env=self.env,
            check=False,
            timeout=5,
        )

    def test_delayed_old_admit_and_model_start_are_fenced(self):
        self.assertEqual(
            0, self.run_script(guarded_remote_script("old", fence=True)).returncode
        )
        self.assertEqual(
            0, self.run_script(guarded_remote_script("new", fence=True)).returncode
        )
        # The payload represents ANY old SSH mutation: admit, service restart,
        # or docker run. It cannot execute after the new transition fence.
        effect = self.path / "obsolete-admit-started"
        command = shlex.join(
            [
                "python3",
                "-c",
                f"from pathlib import Path; Path({str(effect)!r}).touch()",
            ]
        )
        result = self.run_script(guarded_remote_script("old", command))
        self.assertEqual(75, result.returncode)
        self.assertIn("stale remote", result.stderr)
        self.assertFalse(effect.exists())
        self.assertEqual(
            0, self.run_script(guarded_remote_script("new", command)).returncode
        )
        self.assertTrue(effect.exists())

    def test_nested_factory_child_retains_lock_when_all_ancestors_die(self):
        self.run_script(guarded_remote_script("old", fence=True))
        entered, release = self.path / "factory-entered", self.path / "factory-release"
        factory_pid = self.path / "factory-pid"
        factory = self.path / "yue_factory.py"
        factory.write_text(f"""API_VERSION = 2
import json, os, pathlib, time
pathlib.Path({str(factory_pid)!r}).write_text(json.dumps({{"pid": os.getpid(), "parent": os.getppid()}}))
pathlib.Path({str(entered)!r}).touch()
while not pathlib.Path({str(release)!r}).exists(): time.sleep(0.01)
print(json.dumps({healthy("head", "old")!r}))
""")
        systemctl = self.path / "systemctl"
        systemctl.write_text("#!/bin/sh\nexit 0\n")
        systemctl.chmod(0o755)
        self.env["PATH"] = str(self.path) + os.pathsep + self.env["PATH"]
        scripts = []

        def capture(cfg, host, script, check=False):
            scripts.append(script)
            return subprocess.CompletedProcess(
                [], 0, json.dumps(healthy("head", "old")), ""
            )

        cfg = config()
        cfg["yue"] = {"factory_root": str(self.path)}
        controller = Controller(cfg, capture, directory=self.path)
        controller.operation_generation = "old"
        controller.control("head", "admit", generation="old")
        # Instrument only this test's source to record its own process IDs;
        # no system process inventory or elevated permissions are needed.
        guard_pid, shell_pid = self.path / "guard-pid", self.path / "shell-pid"
        guarded_argv = shlex.split(scripts[0].split("\n", 1)[1])
        guarded_argv[2] = guarded_argv[2].replace(
            "root.mkdir(parents=True, exist_ok=True, mode=0o700)",
            "root.mkdir(parents=True, exist_ok=True, mode=0o700)\n"
            + f"pathlib.Path({str(guard_pid)!r}).write_text(str(os.getpid()))",
        )
        guarded_argv[-1] = (
            "printf '%s' \"$$\" > "
            + shlex.quote(str(shell_pid))
            + "\n"
            + guarded_argv[-1]
        )
        current = subprocess.Popen(
            ["bash", "-s"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self.env,
        )
        current.stdin.write(shlex.join(guarded_argv))
        current.stdin.close()
        current.stdin = None
        newer = None
        try:
            deadline = time.monotonic() + 3
            while not entered.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(entered.exists())
            # Kill only our test child's ancestors, stopping at the subprocess
            # we launched. The factory child must retain the inherited FD itself.
            identity = json.loads(factory_pid.read_text())
            ancestors = {
                current.pid,
                int(guard_pid.read_text()),
                int(shell_pid.read_text()),
                identity["parent"],
            }
            self.assertNotIn(os.getpid(), ancestors)
            self.assertNotIn(identity["pid"], ancestors)
            for pid in ancestors:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            newer = subprocess.Popen(
                ["bash", "-s"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=self.env,
            )
            newer.stdin.write(guarded_remote_script("new", fence=True))
            newer.stdin.close()
            newer.stdin = None
            time.sleep(0.1)
            self.assertIsNone(newer.poll())
            self.assertEqual(
                "old", json.loads((self.path / "node.json").read_text())["generation"]
            )
        finally:
            release.touch()
            current.communicate(timeout=5)
            if newer is not None:
                newer.communicate(timeout=5)
        self.assertEqual(0, newer.returncode)

    def fake_docker(self):
        binary = self.path / "docker"
        binary.write_text("""#!/usr/bin/env python3
import json, os, pathlib, sys
root = pathlib.Path(os.environ["SPARK_SERVE_NODE_STATE_DIR"])
with (root / "docker-calls").open("a") as out: out.write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[1] == "create":
    if os.environ.get("SIMULATE_NEW_FENCE_AFTER_CREATE"):
        (root / "node.json").write_text(json.dumps({"generation": "new"}))
    print("a" * 64)
""")
        binary.chmod(0o755)
        self.env["PATH"] = str(self.path) + os.pathsep + self.env["PATH"]

    def test_vllm_creates_then_starts_by_immutable_id(self):
        self.fake_docker()
        self.run_script(guarded_remote_script("old", fence=True))
        cfg = config()
        launch = cli.docker_run_script(cfg, cfg["models"]["deepseek"], 0, False)
        result = self.run_script(guarded_remote_script("old", launch))
        self.assertEqual(0, result.returncode, result.stderr)
        calls = [
            json.loads(line)
            for line in (self.path / "docker-calls").read_text().splitlines()
        ]
        self.assertEqual("create", calls[0][0])
        self.assertEqual(["start", "a" * 64], calls[1])
        self.assertFalse(any(call[0] == "run" for call in calls))

    def test_delayed_create_cannot_start_after_new_generation(self):
        self.fake_docker()
        self.env["SIMULATE_NEW_FENCE_AFTER_CREATE"] = "1"
        self.run_script(guarded_remote_script("old", fence=True))
        cfg = config()
        launch = cli.docker_run_script(cfg, cfg["models"]["deepseek"], 0, False)
        result = self.run_script(guarded_remote_script("old", launch))
        self.assertEqual(75, result.returncode, result.stderr)
        self.assertIn("created but not started", result.stderr)
        calls = [
            json.loads(line)
            for line in (self.path / "docker-calls").read_text().splitlines()
        ]
        self.assertEqual(["create"], [call[0] for call in calls])

    def test_generation_cannot_change_between_check_and_mutation(self):
        self.check_remote_lock(False)

    def test_orphaned_command_retains_lock_after_guard_is_killed(self):
        self.check_remote_lock(True)

    def check_remote_lock(self, interrupt_guard):
        self.run_script(guarded_remote_script("old", fence=True))
        entered, release = self.path / "entered", self.path / "release"
        program = f"""from pathlib import Path
import time
Path({str(entered)!r}).touch()
while not Path({str(release)!r}).exists(): time.sleep(0.01)
"""
        guard_pid = self.path / "guard-pid"
        command = (
            "printf '%s' \"$PPID\" > "
            + shlex.quote(str(guard_pid))
            + "\n"
            + shlex.join(["python3", "-c", program])
            + "\ntrue\n"
        )
        current = subprocess.Popen(
            ["bash", "-s"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self.env,
        )
        current.stdin.write(guarded_remote_script("old", command))
        current.stdin.close()
        current.stdin = None
        newer = None
        try:
            deadline = time.monotonic() + 3
            while not entered.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(entered.exists())
            if interrupt_guard:
                os.kill(int(guard_pid.read_text()), signal.SIGKILL)
            newer = subprocess.Popen(
                ["bash", "-s"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=self.env,
            )
            newer.stdin.write(guarded_remote_script("new", fence=True))
            newer.stdin.close()
            newer.stdin = None
            time.sleep(0.1)
            self.assertIsNone(newer.poll())
            self.assertEqual(
                "old", json.loads((self.path / "node.json").read_text())["generation"]
            )
        finally:
            release.touch()
            current.communicate(timeout=5)
            if newer is not None:
                newer.communicate(timeout=5)
        if interrupt_guard:
            self.assertNotEqual(0, current.returncode)
        else:
            self.assertEqual(0, current.returncode)
        self.assertEqual(0, newer.returncode)
        self.assertEqual(
            "new", json.loads((self.path / "node.json").read_text())["generation"]
        )


if __name__ == "__main__":
    unittest.main()
