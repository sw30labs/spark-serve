"""SuperDeepseek (ds4-super-abl): spark-serve Start equals ~/superdeepseek-abl/start.sh.

The in-container tokenizer patch runs from a reviewed, pinned recipes/ script,
and the offline preflight refuses before any switch can stop the current model.
"""
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes" / "superdeepseek-abl"
loader = importlib.machinery.SourceFileLoader("spark_serve_superdeepseek_test", str(ROOT / "spark-serve"))
spec = importlib.util.spec_from_loader(loader.name, loader)
cli = importlib.util.module_from_spec(spec)
loader.exec_module(cli)
vspec = importlib.util.spec_from_file_location("superdeepseek_verify", RECIPE / "verify.py")
verify = importlib.util.module_from_spec(vspec)
vspec.loader.exec_module(verify)

MID = "ds4-super-abl"
SNAPSHOT = ("/cache/huggingface/hub/models--Jiunsong--SuperDeepseek-V4-Flash-abliterated-MQ-2xDGX/"
            "snapshots/2a7dd6a12d46f752e290c6255140a42600897864")
UNPATCHED = '''class Tok:
    def effort(self, reasoning_effort):
        if True:
            if reasoning_effort is None:
                pass
            elif reasoning_effort in ("max", "xhigh"):
                reasoning_effort = "max"
            else:
                reasoning_effort = "high"
        return reasoning_effort
'''


def catalog():
    return tomllib.loads((ROOT / "models.example.toml").read_text())


def rendered(cfg=None):
    cfg = cfg or catalog()
    scoped = cli.launch_config(cfg, MID, "both")
    model = scoped["models"][MID]
    captured = {}
    original = cli.docker_create_start_script
    try:
        cli.docker_create_start_script = lambda full, rank: captured.setdefault(rank, full) and ""
        for rank, headless in ((0, False), (1, True)):
            cli.docker_run_script(scoped, model, rank, headless)
    finally:
        cli.docker_create_start_script = original
    return model, captured


def completed(code=0, output="ok", error=""):
    return subprocess.CompletedProcess([], code, output, error)


def test_example_entry_pins_the_committed_recipe_files():
    model = catalog()["models"][MID]
    assert model["wrapper"] == "vllm"
    assert model["entry_script_sha256"] == hashlib.sha256((RECIPE / "entry.sh").read_bytes()).hexdigest()
    assert model["preflight_script_sha256"] == hashlib.sha256((RECIPE / "verify.py").read_bytes()).hexdigest()
    assert model["image"].endswith("@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8")
    assert model["preflight_args"][0] == SNAPSHOT == model["serve_path"]


def test_start_runs_the_reviewed_entry_script_then_vllm_on_both_ranks():
    cfg = catalog()
    model, captured = rendered(cfg)
    entry = (RECIPE / "entry.sh").read_text()
    for rank in (0, 1):
        full = captured[rank]
        image_at = full.index(model["image"])
        assert full[image_at - 2:image_at] == ["--entrypoint", "bash"]
        assert full[image_at + 1:image_at + 4] == ["-c", entry, "spark-serve-entry"]
        argv = full[image_at + 4:]
        assert argv[0] == SNAPSHOT
        flags = dict(zip(argv[1::1], argv[2::1]))
        assert flags["--port"] == "8000" and flags["--tokenizer-mode"] == "deepseek_v4"
        assert flags["--master-addr"] == cfg["cluster"]["master_addr"] and flags["--master-port"] == "25001"
        assert flags["--node-rank"] == str(rank) and flags["--nnodes"] == "2"
        assert flags["--distributed-executor-backend"] == "mp"
        assert ("--headless" in argv) == (rank == 1)
        assert argv.count("--tensor-parallel-size") == 1
        cache = cfg["cluster"].get("worker_hf_cache_host") if rank else None
        assert f"{cache or cfg['cluster']['hf_cache_host']}:/cache/huggingface:ro" in full
        assert "/home/spider/superdeepseek-abl/work/flashinfer:/cache/flashinfer" in full
        ip = model["host_ip_head"] if rank == 0 else model["host_ip_worker"]
        assert f"VLLM_HOST_IP={ip}" in full
        assert "VLLM_USE_B12X_MOE=1" in full and "NCCL_IB_HCA==rocep1s0f1,roceP2p1s0f1" in full
        assert full[full.index("--shm-size") + 1] == "64g"
        assert "ai.spark-serve.model=ds4-super-abl" in full


def test_print_cmd_renders_without_ssh(monkeypatch, capsys):
    ssh = Mock(side_effect=AssertionError("no SSH for --print-cmd"))
    monkeypatch.setattr(cli, "ssh_cmd", ssh)
    cli.cmd_up(catalog(), "superdeepseek", True, False, False, print_cmd=True)
    out = capsys.readouterr().out
    assert out.count("docker create --name vllm_superdeepseek_abl") == 2
    assert "SuperDeepseek tokenizer patch applied" in out


def test_edited_entry_script_refuses_before_any_switch(monkeypatch, capsys):
    cfg = catalog()
    cfg["models"][MID]["entry_script_sha256"] = "0" * 64
    ssh, controller = Mock(return_value=completed()), Mock()
    monkeypatch.setattr(cli, "ssh_cmd", ssh)
    monkeypatch.setattr(cli, "Controller", controller)
    with pytest.raises(SystemExit):
        cli.cmd_up(cfg, MID, True, False, True)
    ssh.assert_not_called()
    controller.assert_not_called()
    assert "does not match entry_script_sha256; current workload was not changed" in capsys.readouterr().out


@pytest.mark.parametrize("path", ["../spark-serve", "/etc/passwd", "superdeepseek-abl/missing.sh", ""])
def test_entry_script_must_be_an_existing_file_under_recipes(path):
    with pytest.raises(cli.ControllerError):
        cli.recipe_text({"entry_script": path}, "entry_script", MID)


def test_entry_script_is_vllm_only():
    with pytest.raises(cli.ControllerError, match="only with wrapper"):
        cli.validate_entry_script(MID, {"wrapper": "dsv4", "entry_script": "superdeepseek-abl/entry.sh"})


def test_preflight_runs_the_pinned_verifier_offline_on_both_sparks(monkeypatch):
    cfg = catalog()
    scoped = cli.launch_config(cfg, MID, "both")
    ssh = Mock(return_value=completed(output="superdeepseek preflight ok"))
    monkeypatch.setattr(cli, "ssh_cmd", ssh)
    cli.model_preflight(scoped, MID, scoped["models"][MID], Mock())
    assert [call.args[1] for call in ssh.call_args_list] == [cfg["cluster"]["head"], cfg["cluster"]["worker"]]
    source = (RECIPE / "verify.py").read_text()
    for call in ssh.call_args_list:
        command = shlex.split(call.args[2].split("\n", 2)[2])
        assert command[command.index("--network") + 1] == "none"
        assert command[command.index("--entrypoint") + 1] == "python3"
        tail = command[command.index("$image_id") + 1:]
        assert tail[:2] == ["-c", source]
        assert tail[2:] == scoped["models"][MID]["preflight_args"]
        assert "--gpus" not in command and "--privileged" not in command


def test_failed_preflight_leaves_the_current_workload(monkeypatch, capsys):
    monkeypatch.setattr(cli, "ssh_cmd", Mock(return_value=completed(2, "", "superdeepseek preflight: model snapshot missing")))
    controller = Mock()
    monkeypatch.setattr(cli, "Controller", controller)
    with pytest.raises(SystemExit):
        cli.cmd_up(catalog(), MID, True, False, True)
    controller.assert_not_called()
    assert "current workload was not changed" in capsys.readouterr().out


def test_preflight_script_alone_needs_no_args(monkeypatch):
    model = {"image": "img", "nnodes": 1, "preflight_script": "superdeepseek-abl/verify.py"}
    cfg = {"cluster": {"head": "h", "worker": "w", "nnodes": 2, "hf_cache_host": "/c"}, "models": {"x": model}}
    ssh = Mock(return_value=completed())
    monkeypatch.setattr(cli, "ssh_cmd", ssh)
    cli.model_preflight(cfg, "x", model, Mock())
    assert ssh.call_count == 1


def test_pull_explains_external_preparation_without_downloading(monkeypatch, capsys):
    run = Mock()
    monkeypatch.setattr(cli.subprocess, "run", run)
    monkeypatch.setattr(cli, "ssh_cmd", Mock(side_effect=AssertionError("no SSH")))
    with pytest.raises(SystemExit):
        cli.cmd_pull(catalog(), MID)
    run.assert_not_called()
    assert "~/superdeepseek-abl/prepare.sh --yes" in capsys.readouterr().err


def test_cache_mount_stays_writable_unless_asked():
    cfg = {"cluster": {"hf_cache_host": "/c"}}
    assert "/c:/cache/huggingface" in cli.docker_base(cfg, {}, "n")
    assert "/c:/cache/huggingface:ro" in cli.docker_base(cfg, {"hf_cache_readonly": True}, "n")


def _entry_patch_strings():
    text = (RECIPE / "entry.sh").read_text()
    old = re.search(r'old = """(.*?)"""', text, re.S).group(1)
    new = re.search(r'new = """(.*?)"""', text, re.S).group(1)
    return old, new


def test_verifier_and_entry_patch_the_same_text():
    assert _entry_patch_strings() == (verify.PATCH_OLD, verify.PATCH_NEW)
    assert verify.PATCH_OLD in UNPATCHED


def fake_tree(tmp_path, tokenizer=UNPATCHED, shards=2):
    snap = tmp_path / "snap"
    (snap / "encoding").mkdir(parents=True)
    weight_map = {f"t{i}": f"model-{i}.safetensors" for i in range(shards)}
    for name in set(weight_map.values()):
        (snap / name).write_bytes(b"x")
    (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    for name in ("config.json", "tokenizer.json"):
        (snap / name).write_text("{}")
    (snap / "encoding" / "encoding_dsv4.py").write_text("# official encoding\n")
    pkg = tmp_path / "vllm"
    (pkg / "tokenizers").mkdir(parents=True)
    (pkg / "tokenizers" / "deepseek_v4.py").write_text(tokenizer)
    (pkg / "tokenizers" / "deepseek_v4_encoding.py").write_text("# bundled\n")
    binary = tmp_path / "vllm-bin"
    binary.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$(dirname \"$0\")/served-args\"\n")
    binary.chmod(0o755)
    return snap, pkg, binary


def run_verify(snap, pkg, binary, *extra):
    return verify.main([str(snap), "--vllm-pkg", str(pkg), "--vllm-bin", str(binary), *extra])


def test_verifier_accepts_a_prepared_snapshot_and_patchable_vllm(tmp_path, capsys):
    snap, pkg, binary = fake_tree(tmp_path)
    index_sha = hashlib.sha256((snap / "model.safetensors.index.json").read_bytes()).hexdigest()
    enc_sha = hashlib.sha256((snap / "encoding" / "encoding_dsv4.py").read_bytes()).hexdigest()
    assert run_verify(snap, pkg, binary, "--expected-files", "2", "--expected-index-sha256", index_sha,
                      "--expected-encoding-sha256", enc_sha) == 0
    assert "patch target found once" in capsys.readouterr().out


@pytest.mark.parametrize("breakage,message", [
    ("shard", "missing or empty"), ("count", "expected 3"), ("index_pin", "pinned SHA-256"),
    ("target", "patch target changed"), ("encoding", "missing encoding"), ("binary", "vllm binary missing"),
])
def test_verifier_fails_closed(tmp_path, capsys, breakage, message):
    snap, pkg, binary = fake_tree(tmp_path)
    extra = []
    if breakage == "shard":
        (snap / "model-1.safetensors").unlink()
    elif breakage == "count":
        extra = ["--expected-files", "3"]
    elif breakage == "index_pin":
        extra = ["--expected-index-sha256", "0" * 64]
    elif breakage == "target":
        (pkg / "tokenizers" / "deepseek_v4.py").write_text(UNPATCHED.replace('"high"', '"medium"'))
    elif breakage == "encoding":
        (snap / "encoding" / "encoding_dsv4.py").unlink()
        message = "encoding/encoding_dsv4.py"
    elif breakage == "binary":
        binary.unlink()
    assert run_verify(snap, pkg, binary, *extra) == 2
    assert message in capsys.readouterr().err


def run_entry(snap, pkg, binary, *args):
    env = {"PATH": f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}",
           "VLLM_PKG": str(pkg), "VLLM_BIN": str(binary)}
    script = (RECIPE / "entry.sh").read_text()
    return subprocess.run(["bash", "-c", script, "spark-serve-entry", str(snap), *args],
                          env=env, capture_output=True, text=True, timeout=60)


def test_entry_applies_the_tokenizer_patch_then_execs_vllm_serve(tmp_path):
    snap, pkg, binary = fake_tree(tmp_path)
    proc = run_entry(snap, pkg, binary, "--port", "8000", "--speculative-config", '{"method":"dspark"}')
    assert proc.returncode == 0, proc.stderr
    assert (pkg / "tokenizers" / "deepseek_v4_encoding.py").read_text() == "# official encoding\n"
    patched = (pkg / "tokenizers" / "deepseek_v4.py").read_text()
    assert verify.PATCH_NEW in patched and patched.count('reasoning_effort = "low"') == 1
    served = (tmp_path / "served-args").read_text().splitlines()
    assert served == ["serve", str(snap), "--port", "8000", "--speculative-config", '{"method":"dspark"}']
    # A restarted container keeps its patched layer: the second run is a no-op patch.
    assert run_entry(snap, pkg, binary).returncode == 0
    assert (pkg / "tokenizers" / "deepseek_v4.py").read_text() == patched
    assert run_verify(snap, pkg, binary) == 0


def test_entry_refuses_to_serve_when_the_patch_target_moved(tmp_path):
    snap, pkg, binary = fake_tree(tmp_path, tokenizer=UNPATCHED.replace('"high"', '"medium"'))
    proc = run_entry(snap, pkg, binary)
    assert proc.returncode != 0
    assert "patch target changed" in proc.stderr
    assert not (tmp_path / "served-args").exists()


def test_status_and_stop_see_the_container_but_never_neo4j(monkeypatch):
    cfg = catalog()
    cfg["cluster"]["keep_containers"] = ["singularity-atlas-neo4j"]
    listed = [{"name": "vllm_superdeepseek_abl", "id": "a", "state": "running", "image": "i", "running": True,
               "labels": {"ai.spark-serve.model": MID}},
              {"name": "singularity-atlas-neo4j", "id": "b", "state": "running", "image": "neo4j", "running": True,
               "labels": {}}]
    monkeypatch.setattr(cli, "ssh_cmd", Mock(return_value=completed(output=json.dumps(listed))))
    assert [item["name"] for item in cli.status_host_json(cfg, "spark-head")] == ["vllm_superdeepseek_abl"]
