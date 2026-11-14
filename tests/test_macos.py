"""MacOS Splash cards: local paths, documented serve flags, and a stop that cannot reach port 8000."""
import argparse
import importlib.machinery
import importlib.util
import json
import os
import stat
import sys
import time
import tomllib
from pathlib import Path

import pytest
import yaml

import spark_serve_macos as macos

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = tomllib.loads((ROOT / "models.example.toml").read_text())
MEASURED = json.loads((ROOT / "measured-performance.json").read_text())["models"]

SERVE_FLAGS = {"--model", "--model-dir", "--draft-model", "--host", "--port", "--mmproj"}


def card(port=28101, **over):
    model = {
        "label": "Qwen",
        "model": "mlx-community/Qwen3.8-27B-4bit",
        "format": "mlx",
        "draft_model": "incoai/Qwen3.8-27B-DFlash2",
        "port": port,
        "max_model_len": 262144,
        "aliases": ["qwen"],
    }
    model.update(over)
    return {"macos": {"host": "127.0.0.1", "splash": "splash", "reserved_ports": [8000],
                      "models": {"m": model}}}


def lay_down(tmp_path, cfg):
    model = cfg["macos"]["models"]["m"]
    target = tmp_path / "weights"
    draft = tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    model["model_dir"] = str(target)
    model["draft_dir"] = str(draft)
    return target, draft


def splash_script(tmp_path) -> str:
    path = tmp_path / "splash"
    path.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(120)\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


@pytest.fixture
def state(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARK_SERVE_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(macos, "_port_busy", lambda host, port: False)
    return tmp_path


def test_omitted_paths_default_to_the_splash_repo_directory():
    cfg = card()
    model = cfg["macos"]["models"]["m"]
    assert macos.model_dir(cfg, model) == (
        "/Volumes/DATA/local-models/splash/mlx-community/Qwen3.8-27B-4bit")
    assert macos.draft_dir(cfg, model) == (
        "/Volumes/DATA/local-models/splash/incoai/Qwen3.8-27B-DFlash2")
    variant = dict(model, model="ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF:Q2_0",
                   draft_model="incoai/Qwen3.8-Flash-Next-DFlash2")
    assert macos.model_dir(cfg, variant).endswith(
        "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF")
    assert ":Q2_0" not in macos.model_dir(cfg, variant)


def test_explicit_paths_win_and_disk_status_follows_them(state, tmp_path):
    cfg = card()
    model = cfg["macos"]["models"]["m"]
    model["model_dir"] = str(tmp_path / "weights")
    model["draft_dir"] = str(tmp_path / "draft")
    view = macos.server_view(cfg, "m", model, macos.reserved_ports(cfg))
    assert view["on_disk"] is False and view["model_on_disk"] is False and view["draft_on_disk"] is False
    (tmp_path / "weights").mkdir()
    view = macos.server_view(cfg, "m", model, macos.reserved_ports(cfg))
    assert view["model_on_disk"] is True and view["draft_on_disk"] is False and view["on_disk"] is False
    (tmp_path / "draft").mkdir()
    view = macos.server_view(cfg, "m", model, macos.reserved_ports(cfg))
    assert view["on_disk"] is True
    (tmp_path / "weights").rmdir()
    view = macos.server_view(cfg, "m", model, macos.reserved_ports(cfg))
    assert view["on_disk"] is False and view["model_on_disk"] is False


def test_port_8000_is_reserved_even_when_the_catalog_omits_it():
    cfg = card(port=8000)
    cfg["macos"]["reserved_ports"] = []
    assert 8000 in macos.reserved_ports(cfg)
    with pytest.raises(macos.MacOSError, match="8000"):
        macos.serve_argv(cfg, "m")
    cfg = card(port=28101)
    cfg["macos"]["models"]["n"] = dict(cfg["macos"]["models"]["m"], port=28101)
    with pytest.raises(macos.MacOSError, match="also used"):
        macos._check_ports(cfg)


def test_serve_argv_uses_only_documented_flags(state, tmp_path):
    cfg = card()
    lay_down(tmp_path, cfg)
    argv = macos.serve_argv(cfg, "m", "/opt/splash")
    assert argv[:2] == ["/opt/splash", "serve"]
    flags = [arg for arg in argv if arg.startswith("--")]
    assert flags == ["--model", "--model-dir", "--draft-model", "--host", "--port"]
    assert set(flags) <= SERVE_FLAGS
    assert "8000" not in argv
    assert argv[argv.index("--port") + 1] == "28101"
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--model") + 1] == "mlx-community/Qwen3.8-27B-4bit"
    cfg["macos"]["models"]["m"]["mmproj"] = str(tmp_path / "mmproj.gguf")
    with_vision = macos.serve_argv(cfg, "m", "/opt/splash")
    assert with_vision[with_vision.index("--mmproj") + 1].endswith("mmproj.gguf")
    assert [arg for arg in with_vision if arg.startswith("--")][-1] == "--mmproj"


def test_missing_weights_do_not_spawn_splash(state, tmp_path, monkeypatch):
    cfg = card()
    cfg["macos"]["splash"] = splash_script(tmp_path)
    spawned = []
    monkeypatch.setattr(macos.subprocess, "Popen", lambda *a, **k: spawned.append(a))
    with pytest.raises(macos.MacOSError, match="not on disk"):
        macos.up(cfg, "qwen")
    assert spawned == []


def test_start_and_stop_touch_only_that_cards_process(state, tmp_path, monkeypatch):
    cfg = card(port=28101)
    lay_down(tmp_path, cfg)
    other = dict(cfg["macos"]["models"]["m"], port=28102, label="Other")
    cfg["macos"]["models"]["n"] = other
    cfg["macos"]["splash"] = splash_script(tmp_path)
    monkeypatch.setattr(macos, "_wait_ready", lambda *args: True)
    signals = []
    real_signal = macos._signal_group

    def record(pid, sig):
        signals.append((pid, sig))
        real_signal(pid, sig)

    monkeypatch.setattr(macos, "_signal_group", record)
    first = macos.up(cfg, "m")
    second = macos.up(cfg, "n")
    try:
        assert first["pid"] != second["pid"]
        assert first["port"] == 28101 and second["port"] == 28102
        stopped = macos.stop(cfg, "m")
        assert stopped["stopped"] is True and stopped["pid"] == first["pid"]
        assert macos._alive(second["pid"])
        assert all(pid != second["pid"] for pid, _sig in signals)
        assert {pid for pid, _sig in signals} == {first["pid"]}
    finally:
        for name in ("m", "n"):
            try:
                macos.stop(cfg, name)
            except macos.MacOSError:
                pass


def test_a_process_on_port_8000_is_not_signalled(state, tmp_path, monkeypatch):
    cfg = card(port=28101)
    lay_down(tmp_path, cfg)
    script = splash_script(tmp_path)
    proc = macos.subprocess.Popen(
        [script, "serve", "--model", "omlx", "--model-dir", "/no", "--draft-model", "/no",
         "--host", "127.0.0.1", "--port", "8000"],
        start_new_session=True, stdout=macos.subprocess.DEVNULL, stderr=macos.subprocess.DEVNULL,
    )
    try:
        macos._remember("m", {
            "pid": proc.pid, "port": 28101,
            "argv": [script, "serve", "--model", "omlx", "--model-dir", "/no", "--draft-model", "/no",
                     "--host", "127.0.0.1", "--port", "8000"],
            "started": macos.process_started(proc.pid),
        })
        called = []
        monkeypatch.setattr(macos, "_signal_group", lambda pid, sig: called.append(pid))
        with pytest.raises(macos.MacOSError, match="8000"):
            macos.stop(cfg, "m")
        assert called == []
        assert macos._alive(proc.pid)
        view = macos.server_view(cfg, "m", cfg["macos"]["models"]["m"], macos.reserved_ports(cfg))
        assert view["pid"] is None
        assert macos._alive(proc.pid)
    finally:
        if macos._alive(proc.pid):
            os.killpg(proc.pid, 15)
            proc.wait(timeout=5)


def test_a_busy_card_port_is_left_alone(state, tmp_path, monkeypatch):
    cfg = card()
    lay_down(tmp_path, cfg)
    monkeypatch.setattr(macos, "_port_busy", lambda host, port: True)
    spawned = []
    monkeypatch.setattr(macos.subprocess, "Popen", lambda *a, **k: spawned.append(True))
    with pytest.raises(macos.MacOSError, match="left that process alone"):
        macos.up(cfg, "m")
    assert spawned == []


def test_ready_probe_never_requests_port_8000(monkeypatch):
    opened = []

    def urlopen(request, timeout=0):
        opened.append(request.full_url)
        raise OSError("should not be called")

    monkeypatch.setattr(macos.urllib.request, "urlopen", urlopen)
    assert macos._ready("127.0.0.1", 8000, "mlx-community/Qwen3.8-27B-4bit") is False
    assert opened == []


def test_catalog_marks_only_the_measured_card(state, tmp_path):
    cfg = card()
    cfg["macos"]["models"]["m"]["modalities"] = ["text", "image"]
    cfg["macos"]["models"]["n"] = dict(cfg["macos"]["models"]["m"], label="Other", port=28102,
                                       aliases=[])
    performance = {"m": {"measured_at": "2026-11-13", "single_stream_tokens_per_second": 171,
                         "concurrent_streams": 4, "concurrent_aggregate_tokens_per_second": 192}}
    items = {item["id"]: item for item in macos.catalog(cfg, performance)}
    assert items["m"]["performance"]["single_stream_tokens_per_second"] == 171
    assert items["n"]["performance"] is None
    assert items["m"]["ranking"]["best"] is True
    assert items["n"]["ranking"] is None
    assert items["m"]["backend"] == "splash" and items["m"]["topology"] == "macos"


def _cli():
    loader = importlib.machinery.SourceFileLoader("macos_hermes_cli", str(ROOT / "spark-serve"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    cli = importlib.util.module_from_spec(spec)
    loader.exec_module(cli)
    return cli


def test_use_selects_a_ready_card_without_moving_spark_blocks(state, tmp_path, monkeypatch):
    cfg = card(port=8101, model="mlx-community/Qwen3.8-27B-8bit", max_model_len=262144,
               hermes_supports_vision=True, hermes_provider="splash")
    lay_down(tmp_path, cfg)
    cfg["macos"]["splash"] = splash_script(tmp_path)
    monkeypatch.setattr(macos, "_wait_ready", lambda *args: True)
    macos.up(cfg, "m")
    try:
        monkeypatch.setattr(
            macos, "_ready",
            lambda host, port, served: port != 8000 and served == "mlx-community/Qwen3.8-27B-8bit")
        seen = {}

        def patch(*args, **kwargs):
            seen["args"] = args
            seen["kwargs"] = kwargs
            return ("Hermes (Mac) now uses Qwen", True)

        result = macos.use({**cfg, "cluster": {"head": "sparkone"}}, "m", "spark", patch)
        model = seen["args"][2]
        assert model["served_name"] == "mlx-community/Qwen3.8-27B-8bit"
        assert model["max_model_len"] == 262144
        assert model["hermes_supports_vision"] is True
        assert model["hermes_provider"] == "splash"
        assert seen["kwargs"]["move_ds4"] is False
        assert seen["kwargs"]["api_key"] is None
        assert seen["kwargs"]["provider_label"] == "Splash on this Mac"
        assert seen["kwargs"]["remote_host"] == "sparkone"
        assert seen["kwargs"]["base_url"] == "http://127.0.0.1:8101/v1"
        assert "8000" not in seen["kwargs"]["base_url"]
        assert result["provider"] == "splash" and result["port"] == 8101
        assert result["served"] == "mlx-community/Qwen3.8-27B-8bit"
        assert result["hermes_target"] == "spark" and result["hermes_host"] == "sparkone"
        called = []
        monkeypatch.setattr(macos, "_ready", lambda *args: False)
        with pytest.raises(macos.MacOSError, match="not ready"):
            macos.use(cfg, "m", "local", lambda *a, **k: called.append(True))
        assert called == []
    finally:
        macos.stop(cfg, "m")


def test_use_cli_reports_hermes_without_starting_another_process(state, tmp_path, monkeypatch, capsys):
    cfg = card(port=8104, model="unsloth/Qwen3.8-Flash-Next-GGUF:Q8_0", format="gguf",
               max_model_len=131072, label="Flash Next")
    lay_down(tmp_path, cfg)
    cfg["macos"]["splash"] = splash_script(tmp_path)
    monkeypatch.setattr(macos, "_wait_ready", lambda *args: True)
    macos.up(cfg, "m")
    spawned = []
    real_popen = macos.subprocess.Popen
    monkeypatch.setattr(macos.subprocess, "Popen", lambda *a, **k: spawned.append(a))
    try:
        monkeypatch.setattr(
            macos, "_ready",
            lambda host, port, served: port == 8104 and served == "unsloth/Qwen3.8-Flash-Next-GGUF:Q8_0")
        seen = {}

        def patch(cfg, mid, model, skip, **kwargs):
            seen.update(mid=mid, served=model["served_name"], ctx=model.get("max_model_len"),
                        provider=model.get("hermes_provider"), skip=skip, **kwargs)
            return ("patched", True)

        args = argparse.Namespace(macos_cmd="use", model="qwen", hermes_target="local", json=True)
        assert macos.main(cfg, args, {}, hermes_patch=patch) == 0
        assert spawned == []
        assert seen["provider"] == "splash" and seen["served"] == "unsloth/Qwen3.8-Flash-Next-GGUF:Q8_0"
        assert seen["ctx"] == 131072 and seen["base_url"] == "http://127.0.0.1:8104/v1"
        assert seen["api_key"] is None and seen["remote_host"] is None
        assert seen["provider_label"] == "Splash on this Mac" and seen["move_ds4"] is False
        assert seen["skip"] is False and seen["mid"] == "m"
        out = capsys.readouterr().out
        assert "8000" not in out
        events = [json.loads(line) for line in out.splitlines()]
        assert [event["event"] for event in events] == ["hermes", "selected"]
        selected = events[1]
        assert selected["url"] == "http://127.0.0.1:8104/v1"
        assert selected["served"] == "unsloth/Qwen3.8-Flash-Next-GGUF:Q8_0"
        assert selected["hermes_target"] == "local" and selected["hermes_host"] is None
        assert selected["host"] == "127.0.0.1" and selected["port"] == 8104
    finally:
        monkeypatch.setattr(macos.subprocess, "Popen", real_popen)
        macos.stop(cfg, "m")


def test_spark_hermes_target_uses_the_head_and_a_miss_never_patches(state, tmp_path, monkeypatch, capsys):
    cfg = card(port=8101)
    lay_down(tmp_path, cfg)
    cfg["macos"]["splash"] = splash_script(tmp_path)
    monkeypatch.setattr(macos, "_wait_ready", lambda *args: True)
    macos.up(cfg, "m")
    try:
        monkeypatch.setattr(macos, "_ready", lambda host, port, served: port == 8101)
        seen = {}

        def patch(*args, **kwargs):
            seen["remote_host"] = kwargs["remote_host"]
            return ("patched", True)

        headed = {**cfg, "cluster": {"head": "sparkone"}}
        args = argparse.Namespace(macos_cmd="use", model="m", hermes_target="spark", json=True)
        assert macos.main(headed, args, {}, hermes_patch=patch) == 0
        assert seen["remote_host"] == "sparkone"
        assert json.loads(capsys.readouterr().out.splitlines()[-1])["hermes_host"] == "sparkone"
        with pytest.raises(macos.MacOSError, match="cluster.head"):
            macos.use(cfg, "m", "spark", patch)
        called = []
        monkeypatch.setattr(macos, "_ready", lambda *args: False)
        with pytest.raises(macos.MacOSError, match="not ready"):
            macos.use(cfg, "m", "local", lambda *a, **k: called.append(True))
        assert called == []
        assert macos.main(cfg, argparse.Namespace(
            macos_cmd="use", model="m", hermes_target="local", json=True), {}) == 1
        assert "unavailable" in capsys.readouterr().out
    finally:
        macos.stop(cfg, "m")


def test_use_refuses_port_8000_without_probing_or_patching(monkeypatch):
    cfg = card(port=8000)
    opened = []
    monkeypatch.setattr(macos.urllib.request, "urlopen", lambda *a, **k: opened.append(True))
    called = []

    def patch(*args, **kwargs):
        called.append(True)
        return ("patched", True)

    with pytest.raises(macos.MacOSError, match="8000"):
        macos.use(cfg, "m", "local", patch)
    assert called == [] and opened == []
    args = argparse.Namespace(macos_cmd="use", model="m", hermes_target="local", json=True)
    assert macos.main(cfg, args, {}, hermes_patch=patch) == 1
    assert called == []


def test_use_writes_the_same_hermes_keys_as_strix(state, tmp_path, monkeypatch):
    cfg = card(port=8101, model="mlx-community/Qwen3.8-27B-8bit", max_model_len=262144,
               hermes_supports_vision=True, hermes_provider="splash", label="Qwen3.8-27B")
    lay_down(tmp_path, cfg)
    flash = dict(cfg["macos"]["models"]["m"], label="Flash Next", aliases=["flash"],
                 model="unsloth/Qwen3.8-Flash-Next-GGUF:Q8_0", format="gguf", port=8104,
                 max_model_len=131072, hermes_supports_vision=False)
    flash["model_dir"] = str(tmp_path / "flash")
    flash["draft_dir"] = str(tmp_path / "flash-draft")
    (tmp_path / "flash").mkdir()
    (tmp_path / "flash-draft").mkdir()
    cfg["macos"]["models"]["flash"] = flash
    cfg["macos"]["splash"] = splash_script(tmp_path)
    monkeypatch.setattr(macos, "_wait_ready", lambda *args: True)
    started = []
    try:
        macos.up(cfg, "m")
        started.append("m")
        macos.up(cfg, "flash")
        started.append("flash")
        monkeypatch.setattr(macos, "_ready", lambda host, port, served: port in {8101, 8104})
        cli = _cli()
        path = tmp_path / "config.yaml"
        original = {
            "model": {"default": "old", "provider": "spark", "max_tokens": 4096},
            "providers": {
                "spark": {"name": "Spark cluster", "api_key": "PRIVATE-SPARK-KEY",
                          "base_url": "http://sparkone.lan:8000/v1", "default_model": "old"},
                "spark-ds4": {"base_url": "http://sparkone.lan:8000/v1"},
            },
        }
        path.write_text(yaml.safe_dump(original, sort_keys=False))
        path.chmod(0o600)
        monkeypatch.setattr(cli, "HERMES_CONFIG", path)
        monkeypatch.delenv("SPARK_SERVE_HERMES_KEY", raising=False)

        first = macos.use(cfg, "m", "local", cli._hermes_patch)
        assert first["base_url"] == "http://127.0.0.1:8101/v1"
        assert first["served"] == "mlx-community/Qwen3.8-27B-8bit"
        actual = yaml.safe_load(path.read_text())
        assert actual["model"]["default"] == "mlx-community/Qwen3.8-27B-8bit"
        assert actual["model"]["provider"] == "splash"
        assert actual["model"]["max_tokens"] == 4096
        block = actual["providers"]["splash"]
        assert block["base_url"] == "http://127.0.0.1:8101/v1"
        assert block["default_model"] == "mlx-community/Qwen3.8-27B-8bit"
        assert block["api_key"] == "dummy"
        assert block["name"] == "Splash on this Mac"
        assert block["models"]["mlx-community/Qwen3.8-27B-8bit"] == {
            "context_length": 262144, "supports_vision": True,
        }
        assert actual["providers"]["spark"] == original["providers"]["spark"]
        assert actual["providers"]["spark-ds4"] == original["providers"]["spark-ds4"]
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert list(tmp_path.glob(".config.yaml.*")) == []

        second = macos.use(cfg, "flash", "local", cli._hermes_patch)
        assert second["served"] == "unsloth/Qwen3.8-Flash-Next-GGUF:Q8_0"
        assert second["base_url"] == "http://127.0.0.1:8104/v1"
        actual = yaml.safe_load(path.read_text())
        assert actual["model"]["default"] == "unsloth/Qwen3.8-Flash-Next-GGUF:Q8_0"
        block = actual["providers"]["splash"]
        assert block["base_url"] == "http://127.0.0.1:8104/v1"
        assert block["default_model"] == "unsloth/Qwen3.8-Flash-Next-GGUF:Q8_0"
        assert block["name"] == "Splash on this Mac"
        assert block["models"]["mlx-community/Qwen3.8-27B-8bit"]["context_length"] == 262144
        assert block["models"]["unsloth/Qwen3.8-Flash-Next-GGUF:Q8_0"] == {
            "context_length": 131072, "supports_vision": False,
        }
        assert "8000" not in block["base_url"]
        assert actual["providers"]["spark"]["base_url"] == "http://sparkone.lan:8000/v1"
        assert actual["providers"]["spark"]["api_key"] == "PRIVATE-SPARK-KEY"

        kept = path.read_text()
        monkeypatch.setattr(macos, "_ready", lambda *args: False)
        with pytest.raises(macos.MacOSError, match="not ready"):
            macos.use(cfg, "m", "local", cli._hermes_patch)
        assert path.read_text() == kept
    finally:
        for name in started:
            try:
                macos.stop(cfg, name)
            except macos.MacOSError:
                pass


def test_macos_use_parser_defaults_to_this_mac():
    cli = _cli()
    parser = cli.build_parser()
    args = parser.parse_args(["macos", "use", "macos-qwen38-27b"])
    assert args.macos_cmd == "use" and args.model == "macos-qwen38-27b"
    assert args.hermes_target == "local" and args.json is False
    spark = parser.parse_args(["macos", "use", "macos-flint", "--hermes-target", "spark", "--json"])
    assert spark.hermes_target == "spark" and spark.json is True


def test_shipped_catalog_matches_splash_and_keeps_port_8000():
    models = EXAMPLE["macos"]["models"]
    assert list(models) == [
        "macos-qwen38-27b", "macos-qwen36-35b", "macos-ornith-15-122b",
        "macos-qwen38-flash-next", "macos-flint",
    ]
    assert [models[mid]["port"] for mid in models] == [8101, 8102, 8103, 8104, 8105]
    assert 8000 in macos.reserved_ports(EXAMPLE)
    qwen = models["macos-qwen38-27b"]
    assert qwen["model"] == "mlx-community/Qwen3.8-27B-4bit"
    assert qwen["draft_model"] == "incoai/Qwen3.8-27B-DFlash2"
    assert qwen["model_dir"] == (
        "/Volumes/DATA/local-models/projects/TensorFold/models/mlx-community/Qwen3.8-27B-4bit")
    assert qwen["draft_dir"] == (
        "/Volumes/DATA/local-models/projects/TensorFold/models/incoai/Qwen3.8-27B-DFlash2")
    expected = {
        "macos-qwen36-35b": (
            "mlx-community/Qwen3.6-35B-A3B-4bit", "incoai/Qwen3.6-35B-A3B-DFlash2", "mlx"),
        "macos-ornith-15-122b": (
            "ornith-ai/Ornith-1.5-122B-MLX-4bit", "incoai/Ornith-1.5-122B-DFlash2", "mlx"),
        "macos-qwen38-flash-next": (
            "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF:Q2_0",
            "incoai/Qwen3.8-Flash-Next-DFlash2", "gguf"),
        "macos-flint": ("unsloth/flint-GGUF:UD-Q4_K_XL", "incoai/Flint-DFlash2", "gguf"),
    }
    root = "/Volumes/DATA/local-models/splash"
    for mid, (model_id, draft, fmt) in expected.items():
        row = models[mid]
        assert (row["model"], row["draft_model"], row["format"]) == (model_id, draft, fmt)
        assert row["model_dir"] == f"{root}/{macos.repo_name(model_id)}"
        assert row["draft_dir"] == f"{root}/{draft}"
    for mid in models:
        argv = macos.serve_argv(EXAMPLE, mid)
        assert [arg for arg in argv if arg.startswith("--")] == [
            "--model", "--model-dir", "--draft-model", "--host", "--port"]
        assert "8000" not in argv
    row = MEASURED["macos-qwen38-27b"]
    assert row["manual"] is True
    assert row["measured_at"] == "2026-11-13"
    assert row["splash_version"] == "1.6.1"
    assert row["single_stream_tokens_per_second"] == 171
    assert row["single_stream_ttft_seconds"] == 0.125
    assert row["ttft_kind"] == "cached"
    assert row["prefill_tokens_per_second"] == 544
    assert row["prefill_prompt_tokens"] == 19000
    assert row["prefill_approximate"] is True
    assert (row["concurrent_streams"], row["concurrent_aggregate_tokens_per_second"]) == (4, 192)
    items = {item["id"]: item for item in macos.catalog(EXAMPLE, MEASURED)}
    assert items["macos-qwen38-27b"]["performance"]["single_stream_tokens_per_second"] == 171
    assert items["macos-qwen38-27b"]["ranking"]["best"] is True
    for mid in ("macos-qwen36-35b", "macos-ornith-15-122b", "macos-qwen38-flash-next", "macos-flint"):
        assert items[mid]["performance"] is None and items[mid]["ranking"] is None


def test_manual_measurements_survive_summarize_until_a_protocol_row_exists():
    from tools import catalog_tps_bench as bench
    manual = {"macos-qwen38-27b": {"manual": True, "single_stream_tokens_per_second": 171}}
    protocol = {"ds4": {"single_stream_tokens_per_second": 39.6}}
    merged = bench.merge_measurements(protocol, manual)
    assert merged["macos-qwen38-27b"]["manual"] is True
    assert merged["ds4"]["single_stream_tokens_per_second"] == 39.6
    protocol["macos-qwen38-27b"] = {"single_stream_tokens_per_second": 180}
    replaced = bench.merge_measurements(protocol, manual)
    assert replaced["macos-qwen38-27b"] == {"single_stream_tokens_per_second": 180}


def test_cli_registers_macos(monkeypatch, tmp_path, capsys):
    import importlib.machinery
    import importlib.util
    loader = importlib.machinery.SourceFileLoader("macos_cli", str(ROOT / "spark-serve"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    cli = importlib.util.module_from_spec(spec)
    loader.exec_module(cli)
    parser = cli.build_parser()
    args = parser.parse_args(["macos", "list", "--json"])
    assert args.cmd == "macos" and args.macos_cmd == "list" and args.json
    monkeypatch.setattr(cli, "load", lambda: {"macos": {"models": {}}})
    with pytest.raises(SystemExit) as raised:
        cli.main(["macos", "status", "--json"])
    assert raised.value.code == 1
    assert "no [macos] models" in capsys.readouterr().out


def _kill_group(pid: int) -> None:
    if pid <= 0 or not macos._alive(pid):
        return
    try:
        os.killpg(pid, 15)
    except OSError:
        return
    for _ in range(50):
        if not macos._alive(pid):
            return
        time.sleep(0.05)
    try:
        os.killpg(pid, 9)
    except OSError:
        pass


def _wait_argv(pid: int, model: str) -> list[str]:
    for _ in range(50):
        argv = macos.process_argv(pid) or []
        if macos._flag(argv, "--model") == model and macos._alive(pid):
            return argv
        time.sleep(0.02)
    raise AssertionError(f"pid {pid} did not show --model {model}")


def detached_server(tmp_path, model: str, port: int, record_dir: Path, phase: str = "ready"):
    """A live process with the detached Splash argv shape: --model and --port, no --model-dir."""
    record_dir.mkdir(parents=True, exist_ok=True)
    script = tmp_path / f"detached-{port}.py"
    script.write_text("import time\ntime.sleep(300)\n")
    record = record_dir / f"serve-{port}.json"
    proc = macos.subprocess.Popen(
        [sys.executable, str(script), "-m", "server.server",
         "--model", model, "--binary", "splash", "--port", str(port),
         "--serve-record", str(record)],
        start_new_session=True,
        stdout=macos.subprocess.DEVNULL,
        stderr=macos.subprocess.DEVNULL,
    )
    record.write_text(json.dumps({"pid": proc.pid, "phase": phase}))
    _wait_argv(proc.pid, model)
    return proc


def handoff_splash(tmp_path, record_dir: Path) -> str:
    """A splash launcher that starts a detached server, writes the serve record, and exits."""
    record_dir.mkdir(parents=True, exist_ok=True)
    server = tmp_path / "detached_server.py"
    server.write_text("import time\ntime.sleep(300)\n")
    path = tmp_path / "splash"
    script = """#!/usr/bin/env python3
import json, os, subprocess, sys
port = sys.argv[sys.argv.index("--port") + 1]
model = sys.argv[sys.argv.index("--model") + 1]
record_dir = __RECORD_DIR__
os.makedirs(record_dir, exist_ok=True)
record = os.path.join(record_dir, "serve-" + port + ".json")
proc = subprocess.Popen(
    [__PYTHON__, __SERVER__, "-m", "server.server",
     "--model", model, "--binary", "splash", "--port", port, "--serve-record", record],
    start_new_session=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
temporary = record + ".tmp"
with open(temporary, "w") as out:
    json.dump({"pid": proc.pid, "phase": "ready"}, out)
os.replace(temporary, record)
raise SystemExit(0)
"""
    path.write_text(
        script.replace("__RECORD_DIR__", repr(str(record_dir)))
        .replace("__PYTHON__", repr(sys.executable))
        .replace("__SERVER__", repr(str(server)))
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def test_ready_matches_the_listed_model_id(monkeypatch):
    class Body:
        def __init__(self, payload):
            self.payload = payload

        def read(self, _limit):
            return self.payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    opened = []

    def urlopen(request, timeout=0):
        opened.append(request.full_url)
        body = b'{"data":[{"id":"mlx-community/Qwen3.8-27B-4bit-extra","max_model_len":262144}]}'
        return Body(body)

    monkeypatch.setattr(macos.urllib.request, "urlopen", urlopen)
    assert macos._listed_models("127.0.0.1", 8000, {8000}) is None
    assert macos._ready("127.0.0.1", 8000, "mlx-community/Qwen3.8-27B-4bit") is False
    assert opened == []
    assert macos._listed_models("127.0.0.1", 28101) == ["mlx-community/Qwen3.8-27B-4bit-extra"]
    assert macos._ready("127.0.0.1", 28101, "mlx-community/Qwen3.8-27B-4bit") is False
    assert macos._ready("127.0.0.1", 28101, "mlx-community/Qwen3.8-27B-4bit-extra") is True
    assert all(":8000" not in url for url in opened)


def test_dead_launcher_adopts_the_serve_record(state, tmp_path, monkeypatch):
    port = 28151
    cfg = card(port=port)
    lay_down(tmp_path, cfg)
    record_dir = tmp_path / "runtime"
    cfg["macos"]["serve_record_dir"] = str(record_dir)
    model = cfg["macos"]["models"]["m"]
    model_id = model["model"]
    proc = detached_server(tmp_path, model_id, port, record_dir)
    launcher_argv = macos.serve_argv(cfg, "m", "/opt/splash")
    live = macos.process_argv(proc.pid)
    assert macos._flag(live, "--model-dir") is None
    assert macos._flag(live, "--draft-model") is None
    assert macos._flag(live, "--host") is None
    assert macos._same_server(live, launcher_argv) is False
    dead = macos.subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    assert dead.pid != proc.pid and not macos._alive(dead.pid)
    macos._remember("m", {
        "pid": dead.pid, "port": port, "argv": launcher_argv, "started": "launcher-token",
    })
    try:
        reserved = macos.reserved_ports(cfg)
        starting = macos.server_view(cfg, "m", model, reserved)
        assert starting["phase"] == "starting" and starting["pid"] == proc.pid
        monkeypatch.setattr(
            macos, "_listed_models",
            lambda host, probed, reserved=None: None if probed == 8000 else [model_id])
        view = macos.server_view(cfg, "m", model, reserved)
        assert view["phase"] == "ready" and view["pid"] == proc.pid
        assert view["url"] == f"http://127.0.0.1:{port}/v1"
        saved = json.loads((tmp_path / "state" / "macos-servers.json").read_text())
        assert saved["models"]["m"]["pid"] == proc.pid
        assert saved["models"]["m"]["started"] == macos.process_started(proc.pid)
        assert saved["models"]["m"]["started"]
        assert "--model-dir" not in saved["models"]["m"]["argv"]
        (record_dir / f"serve-{port}.json").unlink()
        macos._remember("m", None)
        scanned = macos.server_view(cfg, "m", model, reserved)
        assert scanned["phase"] == "ready" and scanned["pid"] == proc.pid
        signals = []
        real_signal = macos._signal_group

        def record(pid, sig):
            signals.append(pid)
            real_signal(pid, sig)

        monkeypatch.setattr(macos, "_signal_group", record)
        stopped = macos.stop(cfg, "m")
        assert stopped["stopped"] is True and stopped["pid"] == proc.pid
        assert signals == [proc.pid]
        assert not macos._alive(proc.pid)
    finally:
        _kill_group(proc.pid)


def test_up_records_the_detached_server_not_the_launcher(state, tmp_path, monkeypatch):
    port = 28161
    cfg = card(port=port)
    lay_down(tmp_path, cfg)
    record_dir = tmp_path / "runtime"
    cfg["macos"]["serve_record_dir"] = str(record_dir)
    cfg["macos"]["splash"] = handoff_splash(tmp_path, record_dir)
    model_id = cfg["macos"]["models"]["m"]["model"]
    monkeypatch.setattr(macos, "POLL_SECONDS", 0.05)
    monkeypatch.setattr(macos, "HANDOFF_GRACE", 2)

    def listed(host, probed, reserved=None):
        if probed == 8000:
            return None
        record = record_dir / f"serve-{probed}.json"
        return [model_id] if record.is_file() else None

    monkeypatch.setattr(macos, "_listed_models", listed)
    view = macos.up(cfg, "m", ready_timeout=5)
    record = json.loads((record_dir / f"serve-{port}.json").read_text())
    try:
        assert view["phase"] == "ready"
        assert view["pid"] == record["pid"]
        live = macos.process_argv(view["pid"])
        assert "server.server" in live
        assert macos._same_server(live, macos.serve_argv(cfg, "m")) is False
        saved = json.loads((tmp_path / "state" / "macos-servers.json").read_text())
        assert saved["models"]["m"]["pid"] == record["pid"]
        assert saved["models"]["m"]["started"] == macos.process_started(record["pid"])
        assert "--model-dir" not in saved["models"]["m"]["argv"]
        stopped = macos.stop(cfg, "m")
        assert stopped["pid"] == record["pid"] and stopped["stopped"] is True
        assert not macos._alive(record["pid"])
    finally:
        _kill_group(record["pid"])


def test_up_stops_the_detached_server_if_it_never_serves(state, tmp_path, monkeypatch):
    port = 28162
    cfg = card(port=port)
    lay_down(tmp_path, cfg)
    record_dir = tmp_path / "runtime"
    cfg["macos"]["serve_record_dir"] = str(record_dir)
    cfg["macos"]["splash"] = handoff_splash(tmp_path, record_dir)
    monkeypatch.setattr(macos, "POLL_SECONDS", 0.05)
    monkeypatch.setattr(macos, "_listed_models", lambda *args, **kwargs: None)
    signals = []
    real_signal = macos._signal_group

    def record(pid, sig):
        signals.append(pid)
        real_signal(pid, sig)

    monkeypatch.setattr(macos, "_signal_group", record)
    try:
        with pytest.raises(macos.MacOSError, match="did not become ready"):
            macos.up(cfg, "m", ready_timeout=0.6)
        written = json.loads((record_dir / f"serve-{port}.json").read_text())
        assert written["pid"] in signals
        assert not macos._alive(written["pid"])
    finally:
        path = record_dir / f"serve-{port}.json"
        if path.is_file():
            _kill_group(json.loads(path.read_text()).get("pid") or 0)


def test_up_adopts_a_running_server_instead_of_spawning(state, tmp_path, monkeypatch):
    port = 28171
    cfg = card(port=port)
    lay_down(tmp_path, cfg)
    record_dir = tmp_path / "runtime"
    cfg["macos"]["serve_record_dir"] = str(record_dir)
    model_id = cfg["macos"]["models"]["m"]["model"]
    proc = detached_server(tmp_path, model_id, port, record_dir)
    monkeypatch.setattr(
        macos, "_listed_models",
        lambda host, probed, reserved=None: None if probed == 8000 else [model_id])
    spawned = []
    real_popen = macos.subprocess.Popen

    def track(*args, **kwargs):
        spawned.append(args)
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(macos.subprocess, "Popen", track)
    try:
        view = macos.up(cfg, "m")
        assert spawned == []
        assert view["phase"] == "ready" and view["pid"] == proc.pid
    finally:
        _kill_group(proc.pid)


def test_serving_port_without_a_verified_pid_enables_hermes(state, tmp_path, monkeypatch):
    port = 28181
    cfg = card(port=port)
    lay_down(tmp_path, cfg)
    cfg["macos"]["serve_record_dir"] = str(tmp_path / "no-records")
    model_id = cfg["macos"]["models"]["m"]["model"]
    monkeypatch.setattr(macos, "_command_lines", lambda: [])

    def listed(host, probed, reserved=None):
        if probed == 8000:
            raise AssertionError("probed 8000")
        return [model_id]

    monkeypatch.setattr(macos, "_listed_models", listed)
    view = macos.server_view(cfg, "m", cfg["macos"]["models"]["m"], macos.reserved_ports(cfg))
    assert view["phase"] == "ready" and view["pid"] is None
    assert view["url"] == f"http://127.0.0.1:{port}/v1"
    assert "Stop stays off" in view["error"]
    assert macos._record("m") is None
    result = macos.use(cfg, "m", "local", lambda *args, **kwargs: ("Hermes (Mac) now uses Qwen", True))
    assert result["base_url"] == f"http://127.0.0.1:{port}/v1"
    assert "8000" not in result["base_url"]
    signals = []
    monkeypatch.setattr(macos, "_signal_group", lambda pid, sig: signals.append(pid))
    assert macos.stop(cfg, "m")["stopped"] is False
    assert signals == []


def test_port_serving_another_model_keeps_hermes_off(state, tmp_path, monkeypatch):
    port = 28191
    cfg = card(port=port)
    lay_down(tmp_path, cfg)
    cfg["macos"]["serve_record_dir"] = str(tmp_path / "no-records")
    monkeypatch.setattr(macos, "_command_lines", lambda: [])

    def listed(host, probed, reserved=None):
        if probed == 8000:
            raise AssertionError("probed 8000")
        return ["other/model"]

    monkeypatch.setattr(macos, "_listed_models", listed)
    view = macos.server_view(cfg, "m", cfg["macos"]["models"]["m"], macos.reserved_ports(cfg))
    assert view["phase"] == "busy" and view["pid"] is None and view["url"] is None
    assert view["port_busy"] is True
    assert "other/model" in view["error"] and "busy with another model" in view["error"]
    called = []
    with pytest.raises(macos.MacOSError, match="not ready"):
        macos.use(cfg, "m", "local", lambda *args, **kwargs: called.append(True))
    assert called == []
    signals = []
    monkeypatch.setattr(macos, "_signal_group", lambda pid, sig: signals.append(pid))
    assert macos.stop(cfg, "m")["stopped"] is False
    assert signals == []


def test_port_8000_is_never_probed_or_signalled(state, tmp_path, monkeypatch):
    port = 28201
    cfg = card(port=port)
    lay_down(tmp_path, cfg)
    record_dir = tmp_path / "runtime"
    record_dir.mkdir()
    cfg["macos"]["serve_record_dir"] = str(record_dir)
    model_id = cfg["macos"]["models"]["m"]["model"]
    assert macos.serve_record_path(cfg, 8000) is None
    script = tmp_path / "sleeper.py"
    script.write_text("import time\ntime.sleep(300)\n")
    proc = macos.subprocess.Popen(
        [sys.executable, str(script), "-m", "server.server",
         "--model", model_id, "--port", str(port), "--port", "8000",
         "--serve-record", str(record_dir / "serve-8000.json")],
        start_new_session=True,
        stdout=macos.subprocess.DEVNULL,
        stderr=macos.subprocess.DEVNULL,
    )
    (record_dir / f"serve-{port}.json").write_text(json.dumps({"pid": proc.pid, "phase": "ready"}))
    (record_dir / "serve-8000.json").write_text("{this is not json")
    opened = []

    def urlopen(request, timeout=0):
        opened.append(request.full_url)
        raise OSError("closed")

    monkeypatch.setattr(macos.urllib.request, "urlopen", urlopen)
    binds = []

    def busy(host, probed):
        binds.append(probed)
        if probed == 8000:
            raise AssertionError("bound 8000")
        return False

    monkeypatch.setattr(macos, "_port_busy", busy)
    signals = []
    monkeypatch.setattr(macos, "_signal_group", lambda pid, sig: signals.append(pid))
    try:
        _wait_argv(proc.pid, model_id)
        assert macos._listed_models("127.0.0.1", 8000, {8000}) is None
        assert macos._ready("127.0.0.1", 8000, model_id) is False
        view = macos.server_view(cfg, "m", cfg["macos"]["models"]["m"], macos.reserved_ports(cfg))
        assert view["pid"] is None
        assert view["phase"] == "stopped"
        macos.status(cfg)
        stopped = macos.stop(cfg, "m")
        assert stopped["stopped"] is False
        assert signals == []
        assert macos._alive(proc.pid)
        assert all(":8000" not in url for url in opened)
        assert 8000 not in binds
    finally:
        _kill_group(proc.pid)
