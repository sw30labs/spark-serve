"""Offline tests for host reboot. Never connects to real SSH or reboots a Spark."""

import importlib.machinery
import importlib.util
import io
import unittest
from pathlib import Path
from unittest.mock import patch

from spark_serve_controller import ControllerError


def config():
    return {
        "cluster": {
            "head": "head",
            "worker": "worker",
            "lan_url": "http://head.lan:8000",
            "port": 8000,
            "container": "vllm_cluster",
            "stop_names": ["vllm_cluster"],
            "keep_containers": [],
            "nnodes": 2,
            "master_addr": "192.168.100.10",
            "master_port": 29501,
            "tensor_parallel": 2,
            "hf_cache_host": "/cache",
            "ssh_opts": ["-o", "BatchMode=yes"],
        },
        "models": {},
    }


ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("spark_serve_reboot_test", str(ROOT / "spark-serve"))
spec = importlib.util.spec_from_loader(loader.name, loader)
cli = importlib.util.module_from_spec(spec)
loader.exec_module(cli)


class Clock:
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


class RebootHelperTests(unittest.TestCase):
    def test_cluster_hosts_worker_then_head_without_duplicates(self):
        cfg = config()
        self.assertEqual(["worker", "head"], cli.cluster_hosts(cfg))
        cfg["cluster"]["worker"] = "head"
        self.assertEqual(["head"], cli.cluster_hosts(cfg))

    def test_redact_strips_password_and_prompt(self):
        self.assertEqual("******** ", cli._redact("secret SUDO_PASS", "secret"))

    def test_systemctl_nopasswd_requires_ok_marker(self):
        cfg = config()
        with patch.object(
            cli,
            "ssh_cmd",
            return_value=cli.subprocess.CompletedProcess([], 0, "ok\n", ""),
        ) as ssh:
            self.assertTrue(cli.systemctl_nopasswd(cfg, "head"))
        remote = ssh.call_args.args[2]
        self.assertIn("sudo -n /usr/bin/systemctl --version", remote)
        self.assertNotIn("|", remote)
        with patch.object(
            cli,
            "ssh_cmd",
            return_value=cli.subprocess.CompletedProcess([], 0, "", "password is required"),
        ):
            self.assertFalse(cli.systemctl_nopasswd(cfg, "worker"))

    def test_reboot_auth_plan_fails_closed_without_password(self):
        err = io.StringIO()
        with (
            patch.object(cli, "systemctl_nopasswd", side_effect=lambda cfg, host: host == "head"),
            patch.object(cli.sys, "stderr", err),
            self.assertRaises(SystemExit) as raised,
        ):
            cli.reboot_auth_plan(config(), None)
        self.assertEqual(1, raised.exception.code)
        self.assertIn("worker", err.getvalue())
        self.assertIn("sudo-password-stdin", err.getvalue())

    def test_reboot_auth_plan_rejects_bad_password_before_drain(self):
        err = io.StringIO()
        with (
            patch.object(cli, "systemctl_nopasswd", return_value=False),
            patch.object(cli, "sudo_password_ok", return_value=False),
            patch.object(cli.sys, "stderr", err),
            self.assertRaises(SystemExit),
        ):
            cli.reboot_auth_plan(config(), "wrong-pass")
        self.assertIn("rejected", err.getvalue())
        self.assertNotIn("wrong-pass", err.getvalue())

    def test_reboot_auth_plan_uses_nopasswd_and_password_per_host(self):
        with (
            patch.object(cli, "systemctl_nopasswd", side_effect=lambda cfg, host: host == "head"),
            patch.object(cli, "sudo_password_ok", return_value=True) as check,
        ):
            plan = cli.reboot_auth_plan(config(), "pw")
        self.assertEqual([("worker", "password"), ("head", "nopasswd")], plan)
        check.assert_called_once()
        self.assertEqual("worker", check.call_args.args[1])

    def test_issue_nopasswd_reboot_uses_systemctl_no_block(self):
        with patch.object(
            cli,
            "ssh_cmd",
            return_value=cli.subprocess.CompletedProcess([], 0, "issued\n", ""),
        ) as ssh:
            cli.issue_host_reboot(config(), "head", "nopasswd", None)
        remote = ssh.call_args.args[2]
        self.assertIn("sudo -n /usr/bin/systemctl reboot --no-block", remote)
        self.assertIn("echo issued", remote)

    def test_issue_nopasswd_reboot_fails_when_sudo_wants_a_password(self):
        with (
            patch.object(
                cli,
                "ssh_cmd",
                return_value=cli.subprocess.CompletedProcess(
                    [], 1, "", "sudo: a password is required"
                ),
            ),
            self.assertRaises(SystemExit),
        ):
            cli.issue_host_reboot(config(), "worker", "nopasswd", None)

    def test_ssh_sudo_puts_password_on_stdin_not_argv(self):
        captured = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            captured["input"] = kwargs.get("input")
            return cli.subprocess.CompletedProcess(argv, 0, "systemd 255", "")

        with patch.object(cli.subprocess, "run", side_effect=fake_run):
            proc = cli.ssh_sudo(config(), "worker", [cli.SYSTEMCTL, "--version"], "s3cret")
        self.assertEqual("s3cret\n", captured["input"])
        joined = " ".join(captured["argv"])
        self.assertNotIn("s3cret", joined)
        self.assertIn("python3 -c", captured["argv"][-1])
        self.assertIn("/usr/bin/systemctl", captured["argv"][-1])
        self.assertEqual(0, proc.returncode)
        self.assertNotIn("s3cret", proc.stdout)

    def test_wait_requires_down_then_up(self):
        clock = Clock()
        seen = []
        state = {"worker": True, "head": True}

        def reachable(host):
            return state[host]

        def sleep(seconds):
            clock.sleep(seconds)
            if clock.t >= 3:
                state["worker"] = False
                state["head"] = False
            if clock.t >= 9:
                state["worker"] = True
                state["head"] = True

        sink = cli._Sink(False)
        with patch.object(sink, "event", side_effect=lambda name, **fields: seen.append((name, fields))):
            cli.wait_for_hosts(
                config(),
                ["worker", "head"],
                sink,
                reachable=reachable,
                sleep=sleep,
                clock=clock.monotonic,
                down_timeout=30,
                up_timeout=30,
                interval=3,
            )
        kinds = [name for name, _ in seen]
        self.assertIn("host_down", kinds)
        self.assertIn("host_up", kinds)

    def test_wait_fails_if_ssh_never_drops(self):
        clock = Clock()
        sink = cli._Sink(False)
        with (
            patch.object(sink, "fail", side_effect=SystemExit(1)) as fail,
            self.assertRaises(SystemExit),
        ):
            cli.wait_for_hosts(
                config(),
                ["worker", "head"],
                sink,
                reachable=lambda host: True,
                sleep=clock.sleep,
                clock=clock.monotonic,
                down_timeout=9,
                up_timeout=9,
                interval=3,
            )
        self.assertIn("did not take effect", fail.call_args.args[0])

    def test_wait_fails_if_hosts_never_return(self):
        clock = Clock()
        sink = cli._Sink(False)
        with (
            patch.object(sink, "fail", side_effect=SystemExit(1)) as fail,
            self.assertRaises(SystemExit),
        ):
            cli.wait_for_hosts(
                config(),
                ["worker", "head"],
                sink,
                reachable=lambda host: False,
                sleep=clock.sleep,
                clock=clock.monotonic,
                down_timeout=3,
                up_timeout=9,
                interval=3,
            )
        self.assertIn("still unreachable", fail.call_args.args[0])


class RebootCommandTests(unittest.TestCase):
    def test_preflight_failure_does_not_drain_or_reboot(self):
        with (
            patch.object(cli, "reboot_auth_plan", side_effect=SystemExit(1)),
            patch.object(cli.Controller, "switch") as switch,
            patch.object(cli, "issue_host_reboot") as issue,
            self.assertRaises(SystemExit),
        ):
            cli.cmd_reboot(config(), password=None)
        switch.assert_not_called()
        issue.assert_not_called()

    def test_drain_then_reboot_worker_then_head(self):
        issued = []
        switched = []

        def fake_switch(self, target, *args, after_idle=None, **kwargs):
            switched.append((target, kwargs.get("cancel_jobs")))
            after_idle()

        with (
            patch.object(
                cli,
                "reboot_auth_plan",
                return_value=[("worker", "password"), ("head", "nopasswd")],
            ),
            patch.object(
                cli,
                "issue_host_reboot",
                side_effect=lambda cfg, host, method, password: issued.append((host, method, password)),
            ),
            patch.object(cli, "wait_for_hosts") as wait,
            patch.object(cli.Controller, "switch", fake_switch),
            patch.object(cli.Controller, "save"),
        ):
            cli.cmd_reboot(config(), password="pw", json_mode=True)

        self.assertEqual(
            [("worker", "password", "pw"), ("head", "nopasswd", "pw")],
            issued,
        )
        wait.assert_called_once()
        self.assertEqual(["worker", "head"], wait.call_args.args[1])
        self.assertEqual("none", switched[0][0])

    def test_active_jobs_block_reboot_unless_cancel_jobs(self):
        with (
            patch.object(cli, "reboot_auth_plan", return_value=[("worker", "nopasswd"), ("head", "nopasswd")]),
            patch.object(cli, "issue_host_reboot") as issue,
            patch.object(
                cli.Controller,
                "switch",
                side_effect=ControllerError("YuE is draining active jobs"),
            ),
            self.assertRaises(SystemExit),
        ):
            cli.cmd_reboot(config(), cancel_jobs=False)
        issue.assert_not_called()

    def test_no_wait_skips_ssh_watch(self):
        def fake_switch(self, target, *args, after_idle=None, **kwargs):
            after_idle()

        with (
            patch.object(cli, "reboot_auth_plan", return_value=[("head", "nopasswd")]),
            patch.object(cli, "issue_host_reboot"),
            patch.object(cli, "wait_for_hosts") as wait,
            patch.object(cli.Controller, "switch", fake_switch),
            patch.object(cli.Controller, "save"),
        ):
            cli.cmd_reboot(config(), no_wait=True)
        wait.assert_not_called()

    def test_parser_wires_reboot_flags(self):
        args = cli.build_parser().parse_args(
            ["reboot", "--cancel-jobs", "--no-wait", "--json", "--sudo-password-stdin"]
        )
        self.assertEqual("reboot", args.cmd)
        self.assertTrue(args.cancel_jobs)
        self.assertTrue(args.no_wait)
        self.assertTrue(args.json)
        self.assertTrue(args.sudo_password_stdin)

    def test_main_reads_password_from_stdin_only_when_flagged(self):
        with (
            patch.object(cli, "load", return_value=config()),
            patch.object(cli, "cmd_reboot") as reboot,
            patch.object(cli.sys, "stdin", io.StringIO("pw-from-stdin\n")),
        ):
            cli.main(["reboot", "--sudo-password-stdin"])
        self.assertEqual("pw-from-stdin", reboot.call_args.kwargs["password"])
        with (
            patch.object(cli, "load", return_value=config()),
            patch.object(cli, "cmd_reboot") as reboot,
            patch.object(cli.sys, "stdin", io.StringIO("should-not-read\n")),
        ):
            cli.main(["reboot"])
        self.assertIsNone(reboot.call_args.kwargs["password"])


if __name__ == "__main__":
    unittest.main()
