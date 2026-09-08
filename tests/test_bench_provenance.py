"""Source fingerprint and missing-capability tests without GPU/system commands."""

import hashlib
import json
import subprocess

import pytest

from spark_bench import __main__, experiment
from spark_bench.provenance import collect_provenance


def sources(root):
    package = root / "spark_bench"
    package.mkdir()
    files = {"spark_bench/__init__.py": b"", "spark_bench/experiment.py": b"print('benchmark')\n",
             "spark_serve_controller.py": b"# production controller\n"}
    for name, content in files.items():
        (root / name).write_bytes(content)
    (package / "private.json").write_text('{"token":"DO_NOT_CAPTURE"}')
    (root / ".env").write_text("API_KEY=DO_NOT_CAPTURE\n")
    return files


def successful_driver(args, **kwargs):
    assert kwargs == {"capture_output": True, "text": True, "timeout": 5, "check": False}
    if args == ["nvidia-smi", "--help-query-gpu"]:
        return subprocess.CompletedProcess(args, 0, '"utilization.gpu"\n"temperature.gpu.tlimit"\n', '')
    assert args == ["nvidia-smi", "--query-gpu=index,uuid,driver_version", "--format=csv,noheader,nounits"]
    return subprocess.CompletedProcess(args, 0, "0, GPU-example, 580.173.02\n", "")


def test_inventory_hashes_only_benchmark_and_controller_sources(tmp_path, monkeypatch):
    expected = sources(tmp_path)
    monkeypatch.setenv("API_KEY", "DO_NOT_CAPTURE")
    result = collect_provenance(tmp_path, run=successful_driver,
                                which=lambda name: '/opt/profiler/bin/ncu' if name == 'ncu' else None)
    assert set(result["source_files"]) == set(expected)
    for name, content in expected.items():
        assert result["source_files"][name] == {
            "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content), "status": "available"}
    assert result["source_inventory_status"] == "complete"
    assert result["nvidia_driver"]["versions"] == ["580.173.02"]
    assert result["nvidia_driver"]["gpus"][0]["uuid"] == "GPU-example"
    assert result["nvidia_sampling"]["supported_query_fields"] == ["utilization.gpu", "temperature.gpu.tlimit"]
    assert result["profiling_tools"]["ncu"] == {"status": "present_capability_unverified", "executed": False}
    assert result["profiling_tools"]["dcgmi"] == {"status": "not_found_on_path", "executed": False}
    assert all(result["host"][key] for key in ["system", "kernel_release", "machine", "python_version"])
    assert "DO_NOT_CAPTURE" not in json.dumps(result)
    (tmp_path / "spark_bench/experiment.py").write_bytes(b"print('changed')\n")
    changed = collect_provenance(tmp_path, run=successful_driver)
    assert changed["source_files"]["spark_bench/experiment.py"]["sha256"] != result["source_files"]["spark_bench/experiment.py"]["sha256"]


@pytest.mark.parametrize("failure", ["missing", "timeout", "exit", "unsupported"])
def test_driver_failure_is_explicit_and_never_records_stderr(tmp_path, failure):
    sources(tmp_path)
    def run(args, **_kwargs):
        if failure == "missing":
            raise FileNotFoundError("DO_NOT_CAPTURE")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args, 5, stderr="DO_NOT_CAPTURE")
        return subprocess.CompletedProcess(args, 1 if failure == "exit" else 0,
                                           "0, GPU-example, N/A\n", "DO_NOT_CAPTURE")
    result = collect_provenance(tmp_path, run=run)
    assert result["nvidia_driver"]["status"] == "unavailable"
    assert result["nvidia_driver"]["versions"] == []
    assert result["nvidia_driver"]["error"]
    assert result["nvidia_sampling"]["status"] == "unavailable"
    assert "DO_NOT_CAPTURE" not in json.dumps(result)
    assert result["source_inventory_status"] == "complete"


def test_missing_controller_and_external_symlink_are_visible_inventory_gaps(tmp_path):
    sources(tmp_path)
    (tmp_path / "spark_serve_controller.py").unlink()
    # Point outside the allowlisted source root, without reading its contents.
    (tmp_path / "spark_bench/external.py").symlink_to(tmp_path.parent / "outside.py")
    result = collect_provenance(tmp_path, run=successful_driver)
    assert result["source_inventory_status"] == "partial"
    assert result["source_files"]["spark_serve_controller.py"]["sha256"] is None
    assert result["source_files"]["spark_bench/external.py"]["sha256"] is None
    assert {row["source"] for row in result["errors"]} == {"spark_serve_controller.py", "spark_bench/external.py"}


def test_experiment_persists_provenance_through_final_metadata_rewrite(tmp_path, monkeypatch):
    from pathlib import Path
    marker = {"schema_version": 1, "source_files": {"spark_bench/experiment.py": {"sha256": "test-receipt"}}}
    monkeypatch.setattr(experiment, "collect_provenance", lambda: marker)
    monkeypatch.setattr(experiment, "execute_trial", lambda *_args, **_kwargs: None)
    output = tmp_path / "run"
    workload = Path(__file__).parent / "workloads/yue-standard.json"
    monkeypatch.setattr("sys.argv", ["spark_bench", "concurrency", "--synthetic", "--workload", str(workload),
                                    "--output", str(output), "--concurrency", "1", "--jobs-per-level", "1",
                                    "--iterations", "1", "--warmup", "0", "--cooldown", "0"])
    __main__.main()
    metadata = json.loads((output / "run-metadata.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["provenance"] == marker
