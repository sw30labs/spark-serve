"""Batch launch (ADR 0018): validate before stop, Spark-side restore, guards. Offline only."""

import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from unittest.mock import Mock

import pytest

import spark_serve_batch as batch
from spark_serve_controller import Controller, ControllerError

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("spark_serve_batch_launch_test", str(ROOT / "spark-serve"))
spec = importlib.util.spec_from_loader(loader.name, loader)
cli = importlib.util.module_from_spec(spec)
loader.exec_module(cli)

sv_loader = importlib.machinery.SourceFileLoader("batch_supervisor_test", str(ROOT / "recipes/batch/supervisor.py"))
sv_spec = importlib.util.spec_from_loader(sv_loader.name, sv_loader)
supervisor = importlib.util.module_from_spec(sv_spec)
sv_loader.exec_module(supervisor)

QWEN_STATE = {
    "mode": "vllm", "model": "qwen", "phase": "ready", "version": 1,
    "nodes": {
        "sparkone": {"mode": "vllm", "model": "qwen", "phase": "ready", "allocation_hosts": ["sparkone", "sparktwo"]},
        "sparktwo": {"mode": "vllm", "model": "qwen", "phase": "ready", "allocation_hosts": ["sparkone", "sparktwo"]},
    },
}


def example():
    with (ROOT / "models.example.toml").open("rb") as f:
        return tomllib.load(f)


def config():
    cfg = example()
    pika = cfg["models"]["pika-h3-flex"]
    return {
        "cluster": {"head": "sparkone", "worker": "sparktwo", "nnodes": 2, "port": 8000,
                    "lan_url": "http://head.lan:8000", "container": "vllm_cluster"},
        "models": {"qwen": {"image": "img", "served_name": "q", "wrapper": "tensorfold", "nnodes": 2},
                   "pika-h3-flex": pika},
    }


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARK_SERVE_STATE_DIR", str(tmp_path / "state"))
    for key in ("SPARK_SERVE_LOCAL_HOSTS", "SPARK_SERVE_NODE_SSH_CONFIG"):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / "state").mkdir()
    return tmp_path / "state"


def write_state(directory, state):
    (directory / "controller.json").write_text(json.dumps(state))


# -- spec and allocation ----------------------------------------------------

def test_example_pika_entry_is_launchable_batch():
    cfg = config()
    model = cfg["models"]["pika-h3-flex"]
    assert cli.launch_refusal("pika-h3-flex", model) is None
    form = batch.form("pika-h3-flex", model)
    assert form["resolutions"] == ["480p", "768p"] and form["duration_min"] == 5 and form["duration_max"] == 15
    assert "16:9" in form["aspects"] and form["default_seed"] == 42


def test_build_spec_fills_command_and_output():
    cfg = config()
    spec = batch.build_spec(cfg, "pika-h3-flex", cfg["models"]["pika-h3-flex"], prompt=" a cup ",
                            resolution="768p", duration=7, aspect="9:16", seed=3, now=0)
    jid = spec["id"]
    assert spec["host"] == "sparktwo" and spec["output"] == f"~/pika-h3-flex/outputs/{jid}.mp4"
    argv = spec["argv"]
    assert argv[argv.index("--resolution") + 1] == "768p" and argv[argv.index("--duration") + 1] == "7"
    assert argv[argv.index("--aspect") + 1] == "9:16" and argv[argv.index("--seed") + 1] == "3"
    assert argv[argv.index("--output") + 1] == spec["output"]
    assert argv[argv.index("--prompt-file") + 1] == f"~/.local/state/spark-serve-batch/{jid}/prompt.txt"
    assert spec["prompt"] == "a cup"


@pytest.mark.parametrize("kwargs", [
    {"prompt": ""}, {"prompt": "x", "resolution": "1080p"}, {"prompt": "x", "duration": 4},
    {"prompt": "x", "duration": 16}, {"prompt": "x", "aspect": "7:3"}, {"prompt": "x", "seed": -1},
    {"prompt": "x" * 8001},
])
def test_build_spec_rejects_bad_options(kwargs):
    cfg = config()
    with pytest.raises(ControllerError):
        batch.build_spec(cfg, "pika-h3-flex", cfg["models"]["pika-h3-flex"], **kwargs)


def test_previous_allocation():
    cfg = config()
    assert batch.previous_allocation(cfg, QWEN_STATE) == ("qwen", "both")
    assert batch.previous_allocation(cfg, {"mode": "none", "phase": "stopped"}) == (None, None)
    idle = {"nodes": {"sparkone": {"mode": "none"}, "sparktwo": {"mode": "none"}}}
    assert batch.previous_allocation(cfg, idle) == (None, None)
    solo = {"nodes": {"sparkone": {"mode": "none"},
                      "sparktwo": {"mode": "vllm", "model": "qwen", "phase": "ready", "allocation_hosts": ["sparktwo"]}}}
    assert batch.previous_allocation(cfg, solo) == ("qwen", "worker")
    for bad in ({"nodes": {"sparkone": {"mode": "yue", "phase": "ready"}}},
                {"nodes": {"sparkone": {"mode": "vllm", "model": "qwen", "phase": "starting", "allocation_hosts": ["sparkone"]}}},
                {"nodes": {"sparkone": {"mode": "vllm", "model": "gone", "phase": "ready", "allocation_hosts": ["sparkone"]}}}):
        with pytest.raises(ControllerError):
            batch.previous_allocation(cfg, bad)


def test_restore_spec_runs_snapshot_on_the_spark():
    cfg = config()
    spec = batch.build_spec(cfg, "pika-h3-flex", cfg["models"]["pika-h3-flex"], prompt="x", now=0)
    r = batch.restore_spec(spec, cfg, "qwen", "both")
    assert r["argv"][-5:] == ["up", "qwen", "--node", "both", "--no-hermes"]
    assert r["env"]["SPARK_SERVE_LOCAL_HOSTS"] == "sparktwo"
    assert r["env"]["SPARK_SERVE_STATE_DIR"].endswith("/controller")
    assert batch.restore_spec(spec, cfg, None, None) is None


def test_scripts_are_valid_bash():
    cfg = config()
    spec = batch.build_spec(cfg, "pika-h3-flex", cfg["models"]["pika-h3-flex"], prompt="it's \"quoted\" $HOME", now=0)
    scripts = [batch.upload_script(spec, b"tar", "Host sparkone\n  UserKnownHostsFile __JOB_DIR__/known_hosts\n",
                                   ["sparkone.local ssh-ed25519 AAAA"], b"print(1)"),
               batch.preflight_script(spec, ["sparkone"]), batch.launch_script(spec, "{}"),
               batch.read_script(spec["job_dir"]), batch.stop_script(spec["job_dir"]), batch.cleanup_script(spec["job_dir"])]
    for script in scripts:
        assert subprocess.run(["bash", "-n"], input=script, text=True).returncode == 0
    assert "it's" not in scripts[0]  # the prompt travels base64-encoded


def test_snapshot_contains_restore_code():
    import io, tarfile
    names = tarfile.open(fileobj=io.BytesIO(batch.snapshot(ROOT)), mode="r:gz").getnames()
    assert "app/spark-serve" in names and "app/spark_serve_controller.py" in names
    assert "app/recipes/batch/supervisor.py" in names and "app/spark_serve_batch.py" in names
    assert not any("__pycache__" in n for n in names)


def test_ssh_argv_hooks(monkeypatch):
    cfg = config()
    assert cli.ssh_argv(cfg, "sparkone")[0] == "ssh"
    monkeypatch.setenv("SPARK_SERVE_LOCAL_HOSTS", "sparktwo")
    monkeypatch.setenv("SPARK_SERVE_NODE_SSH_CONFIG", "/job/ssh_config")
    assert cli.ssh_argv(cfg, "sparktwo") == []
    argv = cli.ssh_argv(cfg, "sparkone")
    assert argv[:3] == ["ssh", "-F", "/job/ssh_config"] and argv[-1] == "sparkone"


# -- up ordering ------------------------------------------------------------

class FakeRemote:
    """Records each SSH script by its role and answers like the Spark would."""

    def __init__(self, fail=None):
        self.calls, self.fail = [], fail or {}

    def role(self, script):
        for key, marker in (("upload", "__SPARK_SERVE_B64__"), ("preflight", "preflight-ok"),
                            ("launch", "supervisor.py\" \"$D\""), ("stop", "touch \"$D/cancel\""),
                            ("read", "state.json\" 2>/dev/null"), ("cleanup", "rm -rf")):
            if marker in script:
                return key
        return "other"

    def __call__(self, cfg, host, script, check=True, timeout=660):
        role = self.role(script)
        self.calls.append((role, host))
        if role in self.fail:
            return subprocess.CompletedProcess([], 2, "", self.fail[role])
        out = {"launch": json.dumps({"phase": "starting", "supervisor_pid": 4242}),
               "read": json.dumps({"phase": "running", "restore": "pending"})}.get(role, "ok")
        return subprocess.CompletedProcess([], 0, out, "")


def wire(monkeypatch, remote, switch):
    monkeypatch.setattr(cli, "ssh_cmd", remote)
    monkeypatch.setattr(batch, "ssh_material", lambda cfg, home: ("Host sparkone\n", ["sparkone.local ssh-ed25519 AAAA"]))

    class FakeController(Controller):
        def switch(self, target, *a, **k):
            switch(target)
            self.save(mode="none", phase="stopped", nodes={})
    monkeypatch.setattr(cli, "Controller", FakeController)


def run_up(**options):
    cli.cmd_up(config(), "pika-h3-flex", no_hermes=True, no_wait=False, json_mode=True,
               batch_options={"prompt": "a cup", **options})


def test_up_validates_uploads_preflights_then_stops_then_launches(monkeypatch, isolated_state, capsys):
    write_state(isolated_state, QWEN_STATE)
    order = []
    remote = FakeRemote()
    wire(monkeypatch, remote, lambda target: order.append(("switch", target)))
    real = remote.__call__
    monkeypatch.setattr(cli, "ssh_cmd", lambda *a, **k: (order.append((remote.role(a[2]), a[1])), real(*a, **k))[1])
    run_up()
    assert [step for step in order if step[0] != "read"] == [
        ("upload", "sparktwo"), ("preflight", "sparktwo"), ("switch", "none"), ("launch", "sparktwo")]
    record = batch.load_record()
    assert record["previous"] == "qwen" and record["previous_node"] == "both" and record["host"] == "sparktwo"
    events = [json.loads(line)["event"] for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    assert "batch_started" in events


@pytest.mark.parametrize("stage", ["upload", "preflight"])
def test_failure_before_stop_never_stops(monkeypatch, isolated_state, stage):
    write_state(isolated_state, QWEN_STATE)
    switch = Mock()
    remote = FakeRemote(fail={stage: "missing ~/pika-h3-flex/weights/transformer"})
    wire(monkeypatch, remote, switch)
    with pytest.raises(SystemExit):
        run_up()
    switch.assert_not_called()
    assert batch.load_record() is None
    if stage == "preflight":
        assert ("cleanup", "sparktwo") in remote.calls


def test_bad_options_touch_nothing(monkeypatch, isolated_state):
    write_state(isolated_state, QWEN_STATE)
    switch, remote = Mock(), FakeRemote()
    wire(monkeypatch, remote, switch)
    with pytest.raises(SystemExit):
        run_up(resolution="4k")
    assert remote.calls == [] and switch.call_count == 0


def test_unrestorable_state_refuses_before_stop(monkeypatch, isolated_state):
    write_state(isolated_state, {"nodes": {"sparkone": {"mode": "yue", "phase": "ready"},
                                           "sparktwo": {"mode": "yue", "phase": "ready"}}})
    switch, remote = Mock(), FakeRemote()
    wire(monkeypatch, remote, switch)
    with pytest.raises(SystemExit):
        run_up()
    assert remote.calls == [] and switch.call_count == 0


def test_launch_failure_after_stop_restores_from_mac(monkeypatch, isolated_state, capsys):
    write_state(isolated_state, QWEN_STATE)
    remote = FakeRemote(fail={"launch": "setsid: not found"})
    wire(monkeypatch, remote, lambda target: None)
    restored = Mock(return_value=subprocess.CompletedProcess([], 0, "ready", ""))
    monkeypatch.setattr(cli.subprocess, "run", restored)
    with pytest.raises(SystemExit):
        run_up()
    argv = restored.call_args.args[0]
    assert argv[-5:] == ["up", "qwen", "--node", "both", "--no-hermes"]
    assert "restored qwen" in capsys.readouterr().out


def test_print_cmd_plans_without_remote(monkeypatch, isolated_state, capsys):
    write_state(isolated_state, QWEN_STATE)
    switch, remote = Mock(), FakeRemote()
    wire(monkeypatch, remote, switch)
    cli.cmd_up(config(), "pika-h3-flex", no_hermes=True, no_wait=False, json_mode=False, print_cmd=True,
               batch_options={"prompt": "secret prompt"})
    plan = json.loads(capsys.readouterr().out)
    assert plan["restore"]["model"] == "qwen" and "prompt" not in plan
    assert remote.calls == [] and switch.call_count == 0


# -- guards -----------------------------------------------------------------

def active_record():
    batch.save_record({"id": "20261105-210000", "model": "pika-h3-flex", "host": "sparktwo",
                       "job_dir": "~/.local/state/spark-serve-batch/20261105-210000", "output": "~/o.mp4",
                       "previous": "qwen", "previous_node": "both"})


def test_running_job_blocks_up_and_stop(monkeypatch, isolated_state):
    active_record()
    remote = FakeRemote()
    switch = Mock()
    wire(monkeypatch, remote, switch)
    with pytest.raises(SystemExit):
        cli.cmd_up(config(), "qwen", no_hermes=True, no_wait=False, json_mode=False, node="both")
    with pytest.raises(SystemExit):
        cli.cmd_stop(config(), node="both")
    switch.assert_not_called()


def test_unreachable_spark_still_blocks(monkeypatch, isolated_state):
    active_record()
    remote = FakeRemote(fail={"read": "ssh: connect timed out"})
    wire(monkeypatch, remote, Mock())
    with pytest.raises(ControllerError, match="unreachable"):
        cli.batch_guard(config())


def test_finished_job_adopts_spark_side_state(monkeypatch, isolated_state):
    write_state(isolated_state, {"mode": "none", "phase": "stopped", "nodes": {}})
    active_record()
    restored_state = {"mode": "vllm", "model": "qwen", "phase": "ready", "generation": "g2", "nodes": QWEN_STATE["nodes"]}

    def remote(cfg, host, script, check=True, timeout=660):
        if "controller/controller.json" in script:
            return subprocess.CompletedProcess([], 0, json.dumps(restored_state), "")
        return subprocess.CompletedProcess([], 0, json.dumps({"phase": "finished", "result": "done", "restore": "ok"}), "")
    monkeypatch.setattr(cli, "ssh_cmd", remote)
    view = cli.batch_view(config())
    assert view["closed"] is True
    saved = json.loads((isolated_state / "controller.json").read_text())
    assert saved["generation"] == "g2" and saved["model"] == "qwen"
    cli.batch_guard(config())  # no longer blocks


def test_batch_stop_signals_supervisor(monkeypatch, isolated_state, capsys):
    active_record()
    remote = FakeRemote()
    monkeypatch.setattr(cli, "ssh_cmd", remote)
    cli.cmd_batch(config(), "stop", json_mode=True)
    assert ("stop", "sparktwo") in remote.calls
    assert "qwen is restored on the Spark" in capsys.readouterr().out


def test_controller_switch_refuses_non_server_before_any_remote(tmp_path):
    ssh = Mock(side_effect=AssertionError("must not reach SSH"))
    cfg = config()
    with pytest.raises(ControllerError, match="not a server"):
        Controller(cfg, ssh, directory=tmp_path).switch("pika-h3-flex", start_vllm=lambda g: None)
    ssh.assert_not_called()


def test_pull_refuses_batch(monkeypatch):
    monkeypatch.setattr(cli, "ssh_cmd", Mock(side_effect=AssertionError("no SSH")))
    monkeypatch.setattr(cli.subprocess, "run", Mock(side_effect=AssertionError("no prep")))
    with pytest.raises(SystemExit):
        cli.cmd_pull(config(), "pika-h3-flex", node="worker")


# -- Spark-side supervisor --------------------------------------------------

def make_job(tmp_path, job_code="0", restore_code="0", out=True, require_gpu=False, timeout=None):
    d = tmp_path / "job"
    d.mkdir()
    output = tmp_path / "out.mp4"
    body = (f"open({str(output)!r}, 'wb').write(b'mp4');" if out else "") + f"raise SystemExit({job_code})"
    job = {"id": "j", "model": "pika-h3-flex", "host": "sparktwo", "output": str(output),
           "argv": [sys.executable, "-c", body], "workdir": str(tmp_path), "require_free_gpu": require_gpu,
           "timeout_minutes": timeout,
           "restore": {"model": "qwen", "argv": [sys.executable, "-c", f"raise SystemExit({restore_code})"],
                       "attempts": 2, "retry_seconds": 0}}
    (d / "job.json").write_text(json.dumps(job))
    return d


def state_of(d):
    return json.loads((d / "state.json").read_text())


def test_supervisor_success_then_restore(tmp_path):
    d = make_job(tmp_path)
    supervisor.Supervisor(str(d), sleep=lambda s: None).main()
    s = state_of(d)
    assert (s["result"], s["phase"], s["restore"], s["output_bytes"]) == ("done", "finished", "ok", 3)


def test_supervisor_failed_job_still_restores(tmp_path):
    d = make_job(tmp_path, job_code="3", out=False)
    supervisor.Supervisor(str(d), sleep=lambda s: None).main()
    s = state_of(d)
    assert s["result"] == "failed" and "exit 3" in s["error"] and s["restore"] == "ok"


def test_supervisor_restore_failure_is_reported(tmp_path):
    d = make_job(tmp_path, restore_code="1")
    supervisor.Supervisor(str(d), sleep=lambda s: None).main()
    s = state_of(d)
    assert s["restore"] == "failed" and s["restore_attempts"] == 2 and "qwen" in s["restore_error"]


def test_supervisor_cancel_kills_job_and_restores(tmp_path):
    d = make_job(tmp_path)
    job = json.loads((d / "job.json").read_text())
    job["argv"] = [sys.executable, "-c", "import time; time.sleep(60)"]
    (d / "job.json").write_text(json.dumps(job))
    import threading
    threading.Timer(0.5, (d / "cancel").touch).start()
    sup = supervisor.Supervisor(str(d))
    supervisor.GRACE_SECONDS, supervisor.POLL_SECONDS = 1, 0.05
    sup.main()
    s = state_of(d)
    assert s["result"] == "cancelled" and s["restore"] == "ok" and s["job_pid"]


def test_supervisor_busy_gpu_skips_job_and_restores(tmp_path):
    d = make_job(tmp_path, require_gpu=True)
    busy = subprocess.CompletedProcess([], 0, "123, python3, 60000 MiB\n", "")
    real_run = subprocess.run
    run = lambda argv, **k: busy if argv[0] == "nvidia-smi" else real_run(argv, **k)
    popen = Mock(side_effect=AssertionError("job must not start"))
    supervisor.Supervisor(str(d), sleep=lambda s: None, run=run, popen=popen).main()
    s = state_of(d)
    assert s["result"] == "failed" and "GPU is not free" in s["error"] and s["restore"] == "ok"
