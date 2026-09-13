"""Offline startup failure checks; no real SSH or GPU workloads are started."""

import importlib.machinery
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from spark_serve_controller import Controller, ControllerError

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("spark_serve_startup_test", str(ROOT / "spark-serve"))
spec = importlib.util.spec_from_loader(loader.name, loader)
cli = importlib.util.module_from_spec(spec)
loader.exec_module(cli)

HEAD_ID, WORKER_ID = "a" * 64, "b" * 64
MODEL_BODY = json.dumps({"data": [{"id": "deepseek"}]})


def config(nnodes=2):
    return {
        "cluster": {
            "head": "head", "worker": "worker", "nnodes": nnodes,
            "lan_url": "http://head.lan:8000", "container": "vllm_cluster",
        },
        "models": {"ds4": {
            "image": "local-image", "served_name": "deepseek", "drop_caches": False,
        }},
    }


def result(stdout="", returncode=0, stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def state(status="running", oom=False):
    return json.dumps({
        "Status": status, "Running": status == "running", "OOMKilled": oom,
        "ExitCode": 0 if status == "running" else 1, "Error": "",
    })


class StartupMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.state_patch = patch.object(cli, "state_dir", return_value=self.directory)
        self.state_patch.start()
        self.addCleanup(self.state_patch.stop)
        self.sink = Mock()
        self.monitor = cli._StartupMonitor(config(), self.sink)
        self.monitor.remember("head", 0, f"started head rank=0 id={HEAD_ID}\n")
        self.monitor.remember("worker", 1, f"started worker rank=1 id={WORKER_ID}\n")

    def failure_report(self):
        reports = list(self.directory.glob("diagnostics/startup-*/failure.json"))
        self.assertEqual(1, len(reports))
        return json.loads(reports[0].read_text())

    def test_rank_exit_is_detected_and_both_logs_survive_guarded_cleanup(self):
        for failed_host in ("head", "worker"):
            with self.subTest(failed_host=failed_host):
                monitor = cli._StartupMonitor(config(), self.sink)
                monitor.containers = [dict(c) for c in self.monitor.containers]
                calls = []

                def ssh(cfg, host, remote, **kwargs):
                    calls.append((host, remote, kwargs))
                    cid = HEAD_ID if host == "head" else WORKER_ID
                    self.assertIn(cid, remote)
                    self.assertNotIn("vllm_cluster", remote)
                    self.assertLessEqual(kwargs["timeout"], 15)
                    if "docker inspect" in remote:
                        return result(state("exited" if host == failed_host else "running"))
                    return result(f"{host}: ibv_reg_mr_iova2: Cannot allocate memory\n")

                controller = Controller(config(), ssh, directory=self.directory)
                cleanup_reports = []

                def stop_vllm():
                    if controller.state.get("phase") == "starting":
                        cleanup_reports.extend(self.directory.glob("diagnostics/startup-*/failure.json"))

                with (
                    patch.object(cli, "ssh_cmd", side_effect=ssh),
                    patch.object(controller, "fence_nodes", return_value=[]),
                    patch.object(controller, "stop_yue"),
                    patch.object(controller, "stop_vllm", side_effect=stop_vllm),
                    patch.object(controller, "verify_idle"),
                    self.assertRaisesRegex(ControllerError, f"{failed_host} rank .*startup failed"),
                ):
                    controller.switch("ds4", lambda generation: monitor.check())
                self.assertTrue(cleanup_reports, "evidence must be on disk before automatic cleanup")
                report = json.loads(cleanup_reports[-1].read_text())
                self.assertEqual({HEAD_ID, WORKER_ID}, {c["id"] for c in report["containers"]})
                self.assertTrue(all("Cannot allocate memory" in c["log"] for c in report["containers"]))
                saved = json.loads((self.directory / "controller.json").read_text())
                self.assertEqual("failed", saved["phase"])
                self.assertIn("startup failed", saved["error"])
                self.assertIn("failure.json", saved["error"])
                self.assertEqual(4, len(calls))

    def test_missing_id_is_failure_even_if_replacement_name_could_be_running(self):
        with patch.object(cli, "ssh_cmd", return_value=result(
            returncode=1, stderr=f"Error: No such object: {HEAD_ID}",
        )) as ssh:
            reason = self.monitor._inspect(self.monitor.containers[0])
        self.assertIn("launched container is missing", reason)
        self.assertIn(HEAD_ID, ssh.call_args.args[2])
        self.assertNotIn("vllm_cluster", ssh.call_args.args[2])

    def test_unreachable_host_is_unknown_not_an_asserted_container_crash(self):
        for code in (124, 255):
            with self.subTest(code=code), patch.object(cli, "ssh_cmd", return_value=result(
                returncode=code, stderr="Connection timed out",
            )):
                reason = self.monitor._inspect(self.monitor.containers[0])
                self.assertIn("SSH unavailable", reason)
                self.assertIn("state is unknown", reason)
                self.assertNotIn("startup failed", reason)
                self.assertNotIn("missing", reason)

    def test_invalid_inspection_and_docker_errors_fail_closed(self):
        for response in (
            result("not JSON"), result("{}"), result("null"),
            result('{"Running":"true"}'),
            result(returncode=1, stderr="permission denied to Docker socket"),
        ):
            with self.subTest(response=response), patch.object(cli, "ssh_cmd", return_value=response):
                self.assertIn("state is unknown", self.monitor._inspect(self.monitor.containers[0]))

    def test_oom_dead_restarting_and_paused_states_are_failures(self):
        for response in (state("running", oom=True), state("dead"), state("restarting"), state("paused")):
            with self.subTest(response=response), patch.object(cli, "ssh_cmd", return_value=result(response)):
                self.assertIn("startup failed", self.monitor._inspect(self.monitor.containers[0]))

    def test_healthy_ranks_do_not_fetch_logs_or_emit_errors(self):
        with patch.object(cli, "ssh_cmd", return_value=result(state())) as ssh:
            self.monitor.check()
        self.assertEqual(2, ssh.call_count)
        self.sink.event.assert_not_called()
        self.assertFalse(list(self.directory.glob("diagnostics/**/*")))

    def test_diagnostics_are_bounded_and_written_before_error(self):
        with patch.object(cli, "ssh_cmd", return_value=result("x" * 100000)) as ssh:
            with self.assertRaisesRegex(ControllerError, "original failure"):
                self.monitor.fail("original failure")
        report = self.failure_report()
        self.assertEqual([32768, 32768], [len(c["log"]) for c in report["containers"]])
        self.assertTrue(all("tail -c 32768" in call.args[2] for call in ssh.call_args_list))
        self.assertEqual(2, self.sink.event.call_count)

    def test_diagnostic_disk_failure_does_not_hide_original_error_or_logs(self):
        with (
            patch.object(cli, "ssh_cmd", return_value=result("root cause")),
            patch.object(cli.tempfile, "mkdtemp", side_effect=OSError("disk full")),
            self.assertRaisesRegex(ControllerError, "original failure; could not save startup diagnostics"),
        ):
            self.monitor.fail("original failure")
        self.assertEqual(2, self.sink.event.call_count)
        self.assertEqual("root cause", self.sink.event.call_args.kwargs["output"])

    def test_only_full_unambiguous_ids_for_expected_rank_are_accepted(self):
        monitor = cli._StartupMonitor(config(), self.sink)
        for output in (
            "vllm_cluster", "started head rank=0 id=abc123", f"started head rank=1 id={HEAD_ID}",
            f"created head rank=0 id={HEAD_ID}\nstarted head rank=0 id={WORKER_ID}",
        ):
            self.assertFalse(monitor.remember("head", 0, output))
        self.assertTrue(monitor.remember("head", 0,
            f"created head rank=0 id={HEAD_ID}\n{HEAD_ID}\nstarted head rank=0 id={HEAD_ID}\n"))


class ReadinessTests(unittest.TestCase):
    def test_rank_failure_stops_wait_before_endpoint_poll_or_sleep(self):
        check = Mock(side_effect=ControllerError("worker exited"))
        with (
            patch.object(cli, "probe_v1") as probe,
            patch.object(cli.time, "sleep") as sleep,
            self.assertRaisesRegex(ControllerError, "worker exited"),
        ):
            cli.wait_ready(config(), expect_served="deepseek", check_started=check)
        probe.assert_not_called()
        sleep.assert_not_called()

    def test_healthy_wait_checks_ranks_before_and_after_matching_endpoint(self):
        check = Mock()
        with patch.object(cli, "probe_v1", return_value=MODEL_BODY):
            self.assertEqual((True, "deepseek", MODEL_BODY), cli.wait_ready(
                config(), expect_served="deepseek", check_started=check,
            ))
        self.assertEqual(2, check.call_count)

    def test_matching_endpoint_cannot_hide_rank_disappearing_during_probe(self):
        check = Mock(side_effect=[None, ControllerError("original ID missing")])
        with (
            patch.object(cli, "probe_v1", return_value=MODEL_BODY),
            self.assertRaisesRegex(ControllerError, "original ID missing"),
        ):
            cli.wait_ready(config(), expect_served="deepseek", check_started=check)

    def test_custom_readiness_path_also_checks_both_sides_of_success(self):
        check = Mock()
        with (
            patch.object(cli, "ssh_cmd", return_value=result()) as ssh,
            patch.object(cli, "probe_v1", return_value=MODEL_BODY),
        ):
            self.assertTrue(cli.wait_ready(
                config(), expect_served="deepseek", ready_path="/health", check_started=check,
            )[0])
        self.assertEqual(2, check.call_count)
        self.assertEqual(14, ssh.call_args.kwargs["timeout"])

    def test_endpoint_probe_has_bounded_ssh_timeout(self):
        with patch.object(cli, "ssh_cmd", return_value=result(MODEL_BODY)) as ssh:
            self.assertEqual(MODEL_BODY, cli.probe_v1(config(), timeout=4))
        self.assertEqual(14, ssh.call_args.kwargs["timeout"])


class BringUpTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.state_patch = patch.object(cli, "state_dir", return_value=Path(self.temporary.name))
        self.state_patch.start()
        self.addCleanup(self.state_patch.stop)

    def launch(self, cfg, wait_result):
        def ssh(cfg, host, remote, **kwargs):
            if "docker logs" in remote:
                return result("startup error details")
            rank, cid = (0, HEAD_ID) if host == "head" else (1, WORKER_ID)
            return result(f"created {host} rank={rank} id={cid}\nstarted {host} rank={rank} id={cid}\n")

        sink = Mock()
        with (
            patch.object(cli, "ssh_cmd", side_effect=ssh),
            patch.object(cli, "probe_v1", return_value=""),
            patch.object(cli, "docker_run_script", return_value="launch script"),
            patch.object(cli.time, "sleep"),
            patch.object(cli, "maybe_hermes_json", return_value="skipped"),
            patch.object(cli, "wait_ready", side_effect=wait_result),
        ):
            cli._bring_up(cfg, "ds4", True, False, sink)
        return sink

    def test_solo_recipe_monitors_only_launched_head(self):
        def ready(cfg, **kwargs):
            monitor = kwargs["check_started"].__self__
            self.assertEqual([("head", HEAD_ID)], [(c["host"], c["id"]) for c in monitor.containers])
            return True, "deepseek", MODEL_BODY

        sink = self.launch(config(nnodes=1), ready)
        self.assertIn("ready", [call.args[0] for call in sink.event.call_args_list])

    def test_timeout_preserves_both_rank_logs_with_a_meaningful_controller_error(self):
        with self.assertRaisesRegex(ControllerError, "startup timeout.*failure.json"):
            self.launch(config(), lambda *args, **kwargs: (False, None, ""))
        reports = list(Path(self.temporary.name).glob("diagnostics/startup-*/failure.json"))
        self.assertEqual(1, len(reports))
        self.assertEqual(2, len(json.loads(reports[0].read_text())["containers"]))


if __name__ == "__main__":
    unittest.main()
