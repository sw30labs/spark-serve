import importlib.machinery
import importlib.util
import json
from pathlib import Path

from tools import catalog_tps_bench as bench

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("catalog_tps_cli", str(ROOT / "spark-serve"))
spec = importlib.util.spec_from_loader(loader.name, loader)
cli = importlib.util.module_from_spec(spec)
loader.exec_module(cli)


def run(model, run_id, concurrency, requests, started, *, status="completed", thinking=False, tps=40.0):
    return {"id": run_id, "model": model, "kind": "decode", "status": status, "started_at": started,
            "runtime": {"engine": "vllm"}, "hosts": ["private-host"], "endpoint": "http://private/v1",
            "config": {"concurrency": concurrency, "requests": requests, "max_tokens": 256,
                       "prompt_tokens": 512, "thinking": thinking},
            "results": [{"reasoning_observed": False}],
            "summary": {"aggregate_completion_tokens_per_second_end_to_end": tps * concurrency,
                        "p50": {"completion_tokens_per_second_end_to_end": tps,
                                "post_first_output_tokens_per_second_estimate": tps + 1,
                                "first_model_output_seconds": 0.5}}}


def test_summary_uses_newest_matching_pair_and_drops_host_identity():
    runs = [  # newest first, as list_benchmarks returns them
        run("m", "new-single", 1, 5, "2026-10-19T02:00", tps=50.0),
        run("m", "new-conc", 2, 6, "2026-10-19T01:59", tps=45.0),
        run("m", "old-single", 1, 5, "2026-10-18T00:00", tps=10.0),
        run("m", "failed", 2, 6, "2026-10-19T03:00", status="failed"),
        run("m", "thinking", 1, 5, "2026-10-19T04:00", thinking=None),
        run("half", "only-single", 1, 5, "2026-10-19T00:00"),
    ]
    out = bench.summarize_runs(runs)
    assert set(out) == {"m"}
    assert out["m"]["run_ids"] == ["new-single", "new-conc"]
    assert out["m"]["single_stream_tokens_per_second"] == 50.0
    assert out["m"]["concurrent_aggregate_tokens_per_second"] == 90.0
    assert out["m"]["measured_at"] == "2026-10-19"
    assert "private" not in json.dumps(out)


def test_catalog_json_attaches_measured_performance(monkeypatch, capsys):
    monkeypatch.setattr(cli, "measured_performance", lambda: {"a": {"single_stream_tokens_per_second": 12.5}})
    cfg = {"cluster": {"nnodes": 2}, "models": {
        "a": {"image": "i", "quant": "EXL3 2.9bpw", "quant_bits": 2.9, "quant_scope": "all", "quant_detail": "d"},
        "b": {"image": "i"}}}
    cli.cmd_list_json(cfg)
    items = {item["id"]: item for item in json.loads(capsys.readouterr().out)}
    assert items["a"]["performance"] == {"single_stream_tokens_per_second": 12.5}
    assert (items["a"]["quant"], items["a"]["quant_bits"], items["a"]["quant_scope"]) == ("EXL3 2.9bpw", 2.9, "all")
    assert items["b"]["performance"] is None and items["b"]["quant"] is None


def test_missing_performance_file_is_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "HERE", tmp_path)
    assert cli.measured_performance() == {}
    (tmp_path / "measured-performance.json").write_text("not json")
    assert cli.measured_performance() == {}


def item(mid, tps, ctx=262144, bits=4, scope="experts", mods=("text",)):
    return {"id": mid, "ctx": ctx, "quant_bits": bits, "quant_scope": scope, "modalities": list(mods),
            "performance": {"single_stream_tokens_per_second": tps} if tps else None}


def test_ranking_stars_one_measured_model_and_ignores_unmeasured():
    from spark_serve_ranking import score_models
    scores = score_models([item("fast-lossy", 60, bits=2.9, scope="all"), item("balanced", 50),
                           item("unmeasured", None, ctx=1048576, mods=("text", "image", "video"))])
    assert "unmeasured" not in scores
    assert [m for m, s in scores.items() if s["best"]] == ["balanced"]
    assert scores["fast-lossy"]["parts"]["quant"] == 0.3


def test_eight_bit_weights_rank_with_the_mildest_four_bit_tier():
    from spark_serve_ranking import quant_tier, score_models
    assert quant_tier(8, "all") == quant_tier(8, "experts") == "eight_bit"
    assert quant_tier(4, "all") == "whole_model" and quant_tier(4, "experts") == "experts_only"
    scores = score_models([item("q8", 30, bits=8, scope="all"), item("q4", 30, bits=4, scope="all")])
    assert scores["q8"]["parts"]["quant"] == 1.0 and scores["q4"]["parts"]["quant"] == 0.6


def test_ranking_moves_when_catalog_changes_and_breaks_ties_on_speed():
    from spark_serve_ranking import score_models
    a, b = item("a", 40), item("b", 40)
    assert [m for m, s in score_models([a, b]).items() if s["best"]] == ["a"]
    assert score_models([a, b, item("c", 80)])["c"]["best"]
    assert score_models([]) == {}


def test_modalities_default_from_hermes_vision():
    from spark_serve_ranking import modalities
    assert modalities({}) == ["text"]
    assert modalities({"hermes_supports_vision": True}) == ["text", "image"]
    assert modalities({"hermes_supports_vision": True, "modalities": ["text", "image", "video"]})[-1] == "video"


def test_summary_keeps_engine_reported_context():
    single = run("m", "s", 1, 5, "2026-10-19T02:00")
    single["served_context"] = 131072
    out = bench.summarize_runs([single, run("m", "c", 2, 6, "2026-10-19T01:00")])
    assert out["m"]["served_context"] == 131072
    assert "served_context" not in bench.summarize_runs(
        [run("n", "s", 1, 5, "2026-10-19T02:00"), run("n", "c", 2, 6, "2026-10-19T01:00")])["n"]
