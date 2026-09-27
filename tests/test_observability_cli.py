"""CLI integration checks without SSH, HTTP, or model requests."""

import importlib.machinery
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from spark_serve_controller import ControllerError
from test_node_cli import config
from test_node_controller import NodeController, legacy_solo

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("observability_cli_test", str(ROOT / "spark-serve"))
spec = importlib.util.spec_from_loader(loader.name, loader)
cli = importlib.util.module_from_spec(spec)
loader.exec_module(cli)


@pytest.fixture
def cfg(monkeypatch):
    value = config()
    monkeypatch.setattr(cli, "load", lambda: value)
    monkeypatch.setattr(cli, "_JSON_MODE", False)
    return value


def test_watch_uses_cancellable_status_process_and_interval(cfg, monkeypatch):
    import spark_serve_monitor

    watch = Mock()
    monkeypatch.setattr(spark_serve_monitor, "watch", watch)
    cli.main(["watch", "--json", "--interval", "3"])
    assert watch.call_args.args[0] is cfg
    assert len(watch.call_args.args) == 1
    assert watch.call_args.kwargs == {"interval": 3.0}


@pytest.mark.parametrize("flag,thinking", [([], None), (["--thinking"], True), (["--no-thinking"], False)])
def test_benchmark_forwards_explicit_target_and_server_default_thinking(cfg, monkeypatch, flag, thinking):
    import spark_serve_benchmarks

    run = Mock(return_value=0)
    monkeypatch.setattr(spark_serve_benchmarks, "run_benchmark", run)
    cli.main(["bench", "run", "--node", "worker", "--kind", "prefill", "--concurrency", "2", *flag])
    assert run.call_args.args[0] is cfg
    assert run.call_args.kwargs["node"] == "worker"
    assert run.call_args.kwargs["thinking"] is thinking
    assert run.call_args.kwargs["concurrency"] == 2
    assert run.call_args.kwargs["kind"] == "prefill"


def test_benchmark_exit_status_is_preserved(cfg, monkeypatch):
    import spark_serve_benchmarks

    monkeypatch.setattr(spark_serve_benchmarks, "run_benchmark", Mock(return_value=1))
    with pytest.raises(SystemExit) as result:
        cli.main(["bench", "run", "--node", "head", "--json"])
    assert result.value.code == 1


def test_benchmark_cancel_is_scoped_to_selected_physical_host(cfg, monkeypatch, capsys):
    import spark_serve_benchmarks

    cancel = Mock(return_value=1)
    monkeypatch.setattr(spark_serve_benchmarks, "cancel_benchmarks", cancel)
    cli.main(["bench", "cancel", "--node", "worker", "--json"])
    cancel.assert_called_once_with(cfg, hosts=["second"])
    assert json.loads(capsys.readouterr().out)["count"] == 1


def test_benchmark_validation_error_is_machine_readable(cfg, monkeypatch, capsys):
    import spark_serve_benchmarks

    monkeypatch.setattr(spark_serve_benchmarks, "run_benchmark", Mock(side_effect=ControllerError("target changed")))
    with pytest.raises(SystemExit):
        cli.main(["bench", "run", "--node", "head", "--json"])
    assert json.loads(capsys.readouterr().out)["detail"] == "target changed"


def test_switch_revokes_only_selected_benchmarks_before_remote_mutation(tmp_path, monkeypatch):
    import spark_serve_benchmarks

    controller = NodeController(tmp_path)
    legacy_solo(controller)
    revoked = []

    def revoke(directory, hosts):
        assert directory == tmp_path
        assert controller.node_states()["head"]["phase"] == "ready"
        assert all(call[0] == "audit" for call in controller.calls)
        revoked.extend(hosts)

    monkeypatch.setattr(spark_serve_benchmarks, "revoke_benchmarks", revoke)
    controller.switch("none", hosts=["head"])
    assert revoked == ["head"]


def test_rejected_split_does_not_cancel_a_distributed_benchmark(tmp_path, monkeypatch):
    import spark_serve_benchmarks

    controller = NodeController(tmp_path)
    controller.save(mode="vllm", model="deepseek", phase="ready", generation="distributed")
    revoke = Mock()
    monkeypatch.setattr(spark_serve_benchmarks, "revoke_benchmarks", revoke)
    with pytest.raises(ControllerError, match="both"):
        controller.switch("none", hosts=["head"])
    revoke.assert_not_called()
