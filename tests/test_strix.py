import json

import pytest

import spark_serve_strix as strix

CFG = {"strix": {"host": "box", "ctl": "~/bin/halo-ctl", "models": {
    "halo-a": {"label": "A", "engine": "halogen", "served_name": "a", "port": 8731,
               "max_model_len": 262144, "quant": "Halogen 4-bit", "quant_bits": 4, "quant_scope": "all",
               "modalities": ["text"]},
    "halo-b": {"label": "B", "engine": "lemonade", "served_name": "b-Q4", "port": 13305,
               "max_model_len": 131072, "aliases": ["bee"]},
    "halo-c": {"label": "C", "engine": "vllm", "engine_model": "org/c", "served_name": "c", "port": 8000,
               "max_model_len": 131072},
}}}

STATUS = {"engine": "lemonade", "phase": "ready", "model": "b-Q4", "port": 13305,
          "engines": {"halogen": "stopped", "lemonade": "ready", "vllm": "stopped"}, "inference": "off",
          "api_key": True}
STATS = {"up": False, "ram_free_gib": 60.0, "ram_total_gib": 125.1, "busy_pct": 3.0, "temp_c": 41.0,
         "power_w": 12.5, "secret_ignored": 1}


def fake_ssh(calls, status=STATUS, stats=STATS):
    def run(cfg, command, timeout=strix.SSH_SECONDS):
        calls.append(command)
        if "status --json" in command:
            return json.dumps(status) + "\n" + json.dumps(stats) + "\n"
        if "cat " in command:
            return "hal-secret\n"
        return ""
    return run


def test_status_resolves_catalog_model_and_keeps_only_resource_fields(monkeypatch):
    monkeypatch.setattr(strix, "ssh", fake_ssh([]))
    current = strix.status(CFG)
    assert current["model_id"] == "halo-b"
    assert current["url"] == "http://box:13305/v1"
    assert current["resources"]["ram_free_gib"] == 60.0
    assert "secret_ignored" not in current["resources"]
    assert current["metrics"]["avg_gen_tps"] is None  # Lemonade: Halogen-only metrics stay empty


def test_halogen_status_carries_live_inference_metrics(monkeypatch):
    stats = {**STATS, "up": True, "active": 1, "avg_gen_tps": 52.0, "kv_pct": 8.6}
    monkeypatch.setattr(strix, "ssh", fake_ssh([], status={**STATUS, "engine": "halogen", "model": "a"}, stats=stats))
    current = strix.status(CFG)
    assert current["model_id"] == "halo-a"
    assert (current["metrics"]["active"], current["metrics"]["avg_gen_tps"], current["metrics"]["kv_pct"]) == (1, 52.0, 8.6)


def test_unknown_serving_model_has_no_catalog_id(monkeypatch):
    monkeypatch.setattr(strix, "ssh", fake_ssh([], status={**STATUS, "model": "something-else"}))
    current = strix.status(CFG)
    assert current["model_id"] is None and current["url"] is None


@pytest.mark.parametrize("model,expected", [
    ("halo-a", "~/bin/halo-ctl up halogen"),
    ("bee", "~/bin/halo-ctl up lemonade b-Q4"),
    ("halo-c", "~/bin/halo-ctl up vllm org/c 131072"),
])
def test_up_passes_engine_arguments(monkeypatch, model, expected):
    calls = []
    served = {"halo-a": ("halogen", "a"), "bee": ("lemonade", "b-Q4"), "halo-c": ("vllm", "c")}[model]
    monkeypatch.setattr(strix, "ssh", fake_ssh(calls, status={**STATUS, "engine": served[0], "model": served[1]}))
    strix.up(CFG, model)
    assert calls[0] == expected


def test_up_fails_when_box_serves_something_else(monkeypatch):
    monkeypatch.setattr(strix, "ssh", fake_ssh([]))  # box keeps reporting halo-b
    with pytest.raises(strix.StrixError, match="did not become ready"):
        strix.up(CFG, "halo-a")


def test_catalog_entries_match_card_shape_and_rank_within_site():
    items = strix.catalog(CFG, {"halo-a": {"single_stream_tokens_per_second": 50.0},
                                "halo-b": {"single_stream_tokens_per_second": 30.0}})
    by_id = {i["id"]: i for i in items}
    assert by_id["halo-a"]["topology"] == "strix" and by_id["halo-a"]["backend"] == "halogen"
    assert by_id["halo-a"]["ranking"]["best"] is True
    assert by_id["halo-c"]["ranking"] is None  # unmeasured never ranks
    assert by_id["halo-b"]["modalities"] == ["text"]


def test_missing_site_is_an_error():
    with pytest.raises(strix.StrixError):
        strix.site({})


def test_bench_uses_box_key_and_saves_comparable_run(monkeypatch, tmp_path):
    import spark_serve_benchmarks as benchmarks
    monkeypatch.setattr(strix, "ssh", fake_ssh([]))
    monkeypatch.setattr(benchmarks, "_root", lambda: tmp_path)
    keys = []

    def stream(url, payload, timeout, check, api_key=None):
        keys.append(api_key)
        assert url == "http://box:13305/v1" and payload["model"] == "b-Q4"
        return {"usage": {"prompt_tokens": 500, "completion_tokens": payload["max_tokens"]},
                "metrics": {"first_model_output_seconds": 0.5, "completion_tokens_per_second_end_to_end": 40.0},
                "finish_reason": "length"}

    monkeypatch.setattr(benchmarks, "_stream_request", stream)
    events = []
    assert strix.bench(CFG, concurrency=2, requests=4, max_tokens=256, thinking=False, emit=events.append) == 0
    run = events[-1]["run"]
    assert run["status"] == "completed" and run["model"] == "halo-b" and run["site"] == "strix"
    assert run["config"]["thinking"] is False and run["config"]["concurrency"] == 2
    assert set(keys) == {"hal-secret"}
    saved = json.loads(next((tmp_path / "benchmarks").glob("*.json")).read_text())
    assert "hal-secret" not in json.dumps(saved)


def test_bench_refuses_when_nothing_is_ready(monkeypatch):
    monkeypatch.setattr(strix, "ssh", fake_ssh([], status={**STATUS, "phase": "starting"}))
    with pytest.raises(strix.StrixError, match="no catalog model is ready"):
        strix.bench(CFG)
