import argparse
import json
import shlex

import pytest

import spark_serve_strix as strix

MODELS = {
    "halo-a": {"label": "A", "engine": "halogen", "served_name": "a", "port": 8731,
               "max_model_len": 262144, "quant": "Halogen 4-bit", "quant_bits": 4, "quant_scope": "all",
               "modalities": ["text"]},
    "halo-b": {"label": "B", "engine": "lemonade", "served_name": "b-Q4", "port": 13305,
               "max_model_len": 131072, "aliases": ["bee"]},
    "halo-c": {"label": "C", "engine": "vllm", "engine_model": "org/c", "served_name": "c", "port": 8000,
               "max_model_len": 131072},
    "halo-q": {"label": "Q cluster", "engine": "cluster", "served_name": "q8", "port": 8081,
               "max_model_len": 65536, "quant": "Q8_0", "quant_bits": 8, "quant_scope": "all"},
}
CFG = {"strix": {"hosts": ["one", "two"], "ctl": "~/bin/halo-ctl", "models": MODELS}}
ONE = {"strix": {"host": "box", "ctl": "~/bin/halo-ctl", "models": MODELS}}
DISK = {"halogen": [{"id": "a", "label": "A"}], "cluster": [{"id": "q8", "label": "Q8"}],
        "lemonade": [{"id": "b-Q4", "label": "B"}, {"id": "new-Q4", "label": "New"}],
        "vllm": [{"id": "org/c", "label": "C"}]}
PORTS = {"halogen": 8731, "cluster": 8081, "lemonade": 13305, "vllm": 8000}
STATS = {"up": False, "ram_free_gib": 60.0, "ram_total_gib": 125.1, "busy_pct": 3.0, "temp_c": 41.0,
         "power_w": 12.5, "secret_ignored": 1}


class Boxes:
    """Fake boxes: a probe that reports each one's halo-ctl state, and a halo-ctl that obeys up and stop."""

    def __init__(self, hosts=("one", "two"), load_polls=2, **serving):
        self.state = {h: ("none", "stopped", "") for h in hosts}
        self.state.update(serving)
        self.calls, self.down, self.rpc, self.disk = [], {}, {h: "active" for h in hosts}, {h: DISK for h in hosts}
        self.load_polls, self.loading, self.counters, self.exit_on_load = load_polls, {}, {}, False
        self.stats, self.served = dict(STATS), 131072

    def ssh(self, host, command, timeout=strix.SSH_SECONDS):
        self.calls.append((host, command))
        if host in self.down:
            raise strix.StrixError(f"{host}: {self.down[host]}")
        if "SPARK_SERVE_PROBE" in command:
            return json.dumps(self.probe(host)) + "\n"
        if command.startswith("cat "):
            return f"key-{host}\n"
        if command.startswith("journalctl"):
            return "older line\nload_model: out of memory\n"
        words = shlex.split(command)
        if words[1:2] == ["stop"]:
            self.state[host] = ("none", "stopped", "")
        elif words[1:2] == ["up"]:
            engine, name = words[2], (words[3] if len(words) > 3 else "a")
            if engine == "cluster":
                self.state[host], self.loading[host] = ("cluster", "starting", name), 0
            else:
                self.state[host] = (engine, "ready", name)
        return ""

    def probe(self, host):
        engine, phase, model = self.state[host]
        if engine == "cluster" and phase == "starting" and host in self.loading:
            self.loading[host] += 1
            if self.loading[host] > self.load_polls:
                self.state[host] = ("none", "stopped", "") if self.exit_on_load else ("cluster", "ready", model)
                engine, phase, model = self.state[host]
        clients = sum(1 for other, (e, p, _) in self.state.items()
                      if other != host and e == "cluster" and p in ("starting", "ready"))
        status = {"engine": engine, "phase": phase, "model": model, "port": PORTS.get(engine),
                  "engines": {}, "inference": "off", "api_key": True}
        return {"state": status, "stats": self.stats, "served_context": self.served,
                "counters": self.counters, "disk": self.disk[host], "rpc_worker": self.rpc[host],
                "rpc_clients": clients}


@pytest.fixture
def boxes(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARK_SERVE_STATE_DIR", str(tmp_path))

    def make(**kwargs):
        fake = Boxes(**kwargs)
        monkeypatch.setattr(strix, "ssh", fake.ssh)
        monkeypatch.setattr(strix.time, "sleep", lambda seconds: None)
        return fake
    return make


def commands(fake, host=None):
    """halo-ctl calls only, in order, as (host, command) pairs."""
    return [(h, c) for h, c in fake.calls if c.startswith("~/bin/halo-ctl") and (host is None or h == host)]


def by_host(current):
    return {b["host"]: b for b in current["boxes"]}


def test_hosts_take_the_one_box_form_and_reject_duplicates():
    assert strix.hosts(ONE) == ["box"] and strix.hosts(CFG) == ["one", "two"]
    with pytest.raises(strix.StrixError, match="distinct"):
        strix.hosts({"strix": {"hosts": ["one", "one"]}})
    with pytest.raises(strix.StrixError):
        strix.site({})


def test_a_box_must_be_named_once_there_are_two():
    assert strix.box(ONE, None) == "box"
    with pytest.raises(strix.StrixError, match="--box"):
        strix.box(CFG, None)
    with pytest.raises(strix.StrixError, match="unknown Strix box"):
        strix.box(CFG, "three")


def test_status_resolves_catalog_model_and_keeps_only_resource_fields(boxes):
    boxes(one=("lemonade", "ready", "b-Q4"))
    one = by_host(strix.status(CFG))["one"]
    assert one["model_id"] == "halo-b" and one["url"] == "http://one:13305/v1"
    assert one["resources"]["ram_free_gib"] == 60.0 and "secret_ignored" not in one["resources"]
    assert one["metrics"]["avg_gen_tps"] is None  # Halogen-only field
    assert one["served_context"] == 131072 and one["error"] is None
    assert [m["id"] for m in one["available"]["lemonade"]] == ["b-Q4", "new-Q4"]


def test_halogen_status_carries_live_inference_metrics(boxes):
    fake = boxes(one=("halogen", "ready", "a"))
    fake.stats = {**STATS, "up": True, "active": 1, "avg_gen_tps": 52.0, "kv_pct": 8.6}
    one = by_host(strix.status(CFG))["one"]
    assert one["model_id"] == "halo-a"
    assert (one["metrics"]["active"], one["metrics"]["avg_gen_tps"], one["metrics"]["kv_pct"]) == (1, 52.0, 8.6)


@pytest.mark.parametrize("reported", ["org/c", "c"])
def test_vllm_matches_the_checkpoint_halo_ctl_reports_and_the_older_served_name(boxes, reported):
    boxes(one=("vllm", "ready", reported))
    assert by_host(strix.status(CFG))["one"]["model_id"] == "halo-c"


def test_unknown_serving_model_has_no_catalog_id_but_keeps_its_endpoint(boxes):
    boxes(one=("lemonade", "ready", "new-Q4"))
    one = by_host(strix.status(CFG))["one"]
    assert one["model_id"] is None and one["url"] == "http://one:13305/v1"


def test_status_marks_the_cluster_master_and_its_worker(boxes):
    boxes(one=("cluster", "starting", "q8"))
    current = strix.status(CFG)
    one, two = by_host(current)["one"], by_host(current)["two"]
    assert (one["role"], one["peer"], two["role"], two["peer"]) == ("master", "two", "worker", "one")
    assert two["rpc_clients"] == 1 and two["engine"] == "none"
    assert current["cluster"] == {"master": "one", "worker": "two", "phase": "starting", "model_id": "halo-q",
                                  "url": "http://one:8081/v1", "conflict": False}


def test_an_unreachable_box_is_reported_beside_the_other(boxes):
    fake = boxes(two=("halogen", "ready", "a"))
    fake.down["one"] = "Permission denied (publickey,password)."
    current = strix.status(CFG)
    one, two = by_host(current)["one"], by_host(current)["two"]
    assert one["phase"] == "unreachable" and "Permission denied" in one["error"]
    assert two["model_id"] == "halo-a" and two["error"] is None


def test_a_worker_whose_master_is_out_of_sight_is_still_busy(boxes):
    fake = boxes(one=("cluster", "ready", "q8"))
    fake.down["one"] = "no route"
    two = by_host(strix.status(CFG))["two"]
    assert two["role"] == "worker" and two["engine"] == "none"


@pytest.mark.parametrize("model,expected", [
    ("halo-a", "~/bin/halo-ctl up halogen"),
    ("bee", "~/bin/halo-ctl up lemonade b-Q4"),
    ("halo-c", "~/bin/halo-ctl up vllm org/c 131072"),
])
def test_up_passes_engine_arguments(boxes, model, expected):
    fake = boxes()
    current = strix.up(CFG, model, "two")
    assert commands(fake) == [("two", expected)]
    assert by_host(current)["two"]["phase"] == "ready"


def test_up_needs_a_box_once_there_are_two(boxes):
    boxes()
    with pytest.raises(strix.StrixError, match="--box"):
        strix.up(CFG, "halo-a")


def test_a_model_on_disk_outside_the_catalog_starts_quoted(boxes):
    fake = boxes()
    fake.disk["one"] = {**DISK, "lemonade": [{"id": "odd $(name)", "label": "Odd"}]}
    current = strix.up(CFG, "lemonade:odd $(name)", "one")
    assert commands(fake) == [("one", "~/bin/halo-ctl up lemonade 'odd $(name)'")]
    assert by_host(current)["one"]["model_id"] is None


def test_up_refuses_a_model_missing_from_the_box_disk(boxes):
    fake = boxes()
    fake.disk["two"] = {**DISK, "vllm": []}
    with pytest.raises(strix.StrixError, match="two has no org/c on disk"):
        strix.up(CFG, "halo-c", "two")
    assert commands(fake) == []


def test_up_fails_when_box_serves_something_else(boxes, monkeypatch):
    fake = boxes()

    def lemonade_loads_another(host, command, timeout=strix.SSH_SECONDS):
        out = fake.ssh(host, command, timeout)
        if " up lemonade" in command:
            fake.state[host] = ("lemonade", "ready", "other-Q4")
        return out

    monkeypatch.setattr(strix, "ssh", lemonade_loads_another)
    with pytest.raises(strix.StrixError, match="did not become ready"):
        strix.up(CFG, "halo-b", "one")


def test_the_cluster_stops_the_other_box_first_and_waits_for_the_load(boxes):
    fake = boxes(two=("lemonade", "ready", "b-Q4"), load_polls=3)
    events = []
    current = strix.up(CFG, "halo-q", "one", emit=events.append)
    assert commands(fake) == [("two", "~/bin/halo-ctl stop"), ("one", "~/bin/halo-ctl up cluster q8")]
    assert current["cluster"]["phase"] == "ready" and current["cluster"]["worker"] == "two"
    assert "Stopping lemonade b-Q4 on two" in events[0]["detail"]


def test_moving_the_cluster_master_stops_the_old_master(boxes):
    fake = boxes(two=("cluster", "ready", "q8"))
    current = strix.up(CFG, "halo-q", "one")
    assert commands(fake) == [("two", "~/bin/halo-ctl stop"), ("one", "~/bin/halo-ctl up cluster q8")]
    assert by_host(current)["two"]["role"] == "worker"


def test_the_cluster_already_loading_on_that_master_is_only_awaited(boxes):
    fake = boxes(one=("cluster", "starting", "q8"))
    fake.loading["one"] = 0
    assert strix.up(CFG, "halo-q", "one")["cluster"]["phase"] == "ready"
    assert commands(fake) == []


def test_the_cluster_needs_the_workers_rpc_server(boxes):
    fake = boxes()
    fake.rpc["two"] = "inactive"
    with pytest.raises(strix.StrixError, match="ggml-rpc-server.service is inactive"):
        strix.up(CFG, "halo-q", "one")
    assert commands(fake) == []


def test_the_cluster_needs_a_reachable_worker(boxes):
    fake = boxes()
    fake.down["two"] = "Permission denied (publickey,password)."
    with pytest.raises(strix.StrixError, match="cannot check the cluster's worker"):
        strix.up(CFG, "halo-q", "one")


def test_a_cluster_that_exits_while_loading_reports_its_last_log_line(boxes):
    fake = boxes()
    fake.exit_on_load = True
    with pytest.raises(strix.StrixError, match="one: load_model: out of memory"):
        strix.up(CFG, "halo-q", "one")


def test_a_model_on_the_worker_stops_the_cluster_first(boxes):
    fake = boxes(one=("cluster", "ready", "q8"))
    current = strix.up(CFG, "halo-a", "two")
    assert commands(fake) == [("one", "~/bin/halo-ctl stop"), ("two", "~/bin/halo-ctl up halogen")]
    assert current["cluster"] is None and by_host(current)["two"]["model_id"] == "halo-a"


def test_a_worker_whose_master_cannot_be_reached_is_left_alone(boxes):
    fake = boxes(one=("cluster", "ready", "q8"))
    fake.down["one"] = "no route"
    with pytest.raises(strix.StrixError, match="busy as a cluster worker"):
        strix.up(CFG, "halo-a", "two")
    assert commands(fake) == []


def test_stopping_the_worker_stops_the_cluster(boxes):
    fake = boxes(one=("cluster", "ready", "q8"))
    current = strix.stop(CFG, "two")
    assert commands(fake) == [("one", "~/bin/halo-ctl stop"), ("two", "~/bin/halo-ctl stop")]
    assert current["cluster"] is None


def test_stopping_everything_stops_masters_first_and_reports_unreachable_boxes(boxes):
    fake = boxes(two=("cluster", "ready", "q8"))
    strix.stop(CFG)
    assert [h for h, _ in commands(fake)] == ["two", "one"]
    fake.down["one"] = "no route"
    with pytest.raises(strix.StrixError, match="no route"):
        strix.stop(CFG)


def test_catalog_entries_match_card_shape_and_rank_within_site():
    items = strix.catalog(CFG, {"halo-a": {"single_stream_tokens_per_second": 50.0},
                                "halo-b": {"single_stream_tokens_per_second": 30.0}})
    by_id = {i["id"]: i for i in items}
    assert by_id["halo-a"]["topology"] == "strix" and by_id["halo-a"]["backend"] == "halogen"
    assert by_id["halo-q"]["topology"] == "strix-cluster" and by_id["halo-q"]["ctl_id"] == "q8"
    assert by_id["halo-c"]["ctl_id"] == "org/c"
    assert by_id["halo-a"]["ranking"]["best"] is True
    assert by_id["halo-c"]["ranking"] is None  # unmeasured never ranks
    assert by_id["halo-b"]["modalities"] == ["text"]


def test_bench_uses_box_key_and_saves_comparable_run(boxes, monkeypatch, tmp_path):
    import spark_serve_benchmarks as benchmarks
    boxes(two=("lemonade", "ready", "b-Q4"))
    monkeypatch.setattr(benchmarks, "_root", lambda: tmp_path)
    keys = []

    def stream(url, payload, timeout, check, api_key=None):
        keys.append(api_key)
        assert url == "http://two:13305/v1" and payload["model"] == "b-Q4"
        return {"usage": {"prompt_tokens": 500, "completion_tokens": payload["max_tokens"]},
                "metrics": {"first_model_output_seconds": 0.5, "completion_tokens_per_second_end_to_end": 40.0},
                "finish_reason": "length"}

    monkeypatch.setattr(benchmarks, "_stream_request", stream)
    events = []
    assert strix.bench(CFG, "two", concurrency=2, requests=4, max_tokens=256, thinking=False, emit=events.append) == 0
    run = events[-1]["run"]
    assert run["status"] == "completed" and run["model"] == "halo-b" and run["site"] == "strix"
    assert run["hosts"] == ["two"] and run["config"]["thinking"] is False and run["config"]["concurrency"] == 2
    assert set(keys) == {"key-two"}
    saved = json.loads(next((tmp_path / "benchmarks").glob("*.json")).read_text())
    assert "key-two" not in json.dumps(saved)


def test_bench_on_the_worker_measures_the_cluster_on_its_master(boxes, monkeypatch, tmp_path):
    import spark_serve_benchmarks as benchmarks
    boxes(one=("cluster", "ready", "q8"))
    monkeypatch.setattr(benchmarks, "_root", lambda: tmp_path)
    urls = []

    def stream(url, payload, timeout, check, api_key=None):
        urls.append((url, api_key))
        return {"usage": {"prompt_tokens": 500, "completion_tokens": 16},
                "metrics": {"first_model_output_seconds": 1.0, "completion_tokens_per_second_end_to_end": 16.0},
                "finish_reason": "length"}

    monkeypatch.setattr(benchmarks, "_stream_request", stream)
    events = []
    assert strix.bench(CFG, "two", requests=1, emit=events.append) == 0
    assert set(urls) == {("http://one:8081/v1", "key-one")}
    assert events[-1]["run"]["hosts"] == ["one", "two"] and events[-1]["run"]["model"] == "halo-q"


def test_bench_refuses_when_nothing_is_ready(boxes):
    boxes(one=("lemonade", "starting", "b-Q4"))
    with pytest.raises(strix.StrixError, match="no catalog model is ready on one"):
        strix.bench(CFG, "one")


@pytest.mark.parametrize("engine,model,counters,expected", [
    ("lemonade", "b-Q4", {"llamacpp:requests_processing": 1.0, "llamacpp:requests_deferred": 2.0,
                          "llamacpp:kv_cache_usage_ratio": 0.125, "llamacpp:n_decode_total": 900.0,
                          "llamacpp:tokens_predicted_total": 0.0,
                          "llamacpp:prompt_tokens_total": 4000.0}, (1.0, 2.0, 12.5, 900.0, 4000.0)),
    ("vllm", "org/c", {"vllm:num_requests_running": 3.0, "vllm:num_requests_waiting": 0.0,
                       "vllm:kv_cache_usage_perc": 0.5, "vllm:generation_tokens_total": 1500.0,
                       "vllm:prompt_tokens_total": 6000.0}, (3.0, 0.0, 50.0, 1500.0, 6000.0)),
])
def test_engine_counters_are_normalized_for_the_app(boxes, engine, model, counters, expected):
    fake = boxes(one=(engine, "ready", model))
    fake.counters = counters
    m = by_host(strix.status(CFG))["one"]["metrics"]
    assert (m["active"], m["queued"], m["kv_pct"], m["generation_tokens_total"], m["prompt_tokens_total"]) == expected
    assert m["avg_gen_tps"] is None


def test_probe_keeps_the_key_on_the_box(boxes):
    fake = boxes()
    strix.status(CFG)
    probe = next(c for _, c in fake.calls if "SPARK_SERVE_PROBE" in c)
    assert "key-" not in probe and "~/.config/halogen/api-key" in probe and "( sport = :50053 )" in probe


def test_connection_follows_the_worker_to_its_master(boxes):
    boxes(one=("cluster", "ready", "q8"))
    details = strix.connection(CFG, "two")
    assert details == {"host": "one", "engine": "cluster", "base_url": "http://one:8081/v1",
                       "model": "q8", "api_key": "key-one"}


def test_connection_names_the_model_clients_send_to_vllm(boxes):
    boxes(two=("vllm", "ready", "org/c"))
    assert strix.connection(CFG, "two")["model"] == "c"
    with pytest.raises(strix.StrixError, match="nothing is serving on one"):
        strix.connection(CFG, "one")


@pytest.mark.parametrize("serving,host,units", [
    ({"one": ("cluster", "ready", "q8")}, "one", ["halo-cluster.service"]),
    ({"one": ("cluster", "ready", "q8")}, "two", ["ggml-rpc-server.service"]),
    ({}, "one", list(strix.UNITS.values())),
])
def test_logs_read_the_engine_the_rpc_worker_or_every_engine(boxes, serving, host, units):
    fake = boxes(**serving)
    out = strix.logs(CFG, host, lines=5)
    assert out["units"] == units and out["lines"][-1] == "load_model: out of memory"
    assert fake.calls[-1][1].endswith(" ".join(f"-u {u}" for u in units))


def test_cli_streams_progress_and_ends_with_the_ready_status(boxes, capsys):
    boxes(two=("halogen", "ready", "a"))
    parser = argparse.ArgumentParser()
    strix.add_parser(parser.add_subparsers(dest="cmd"))
    assert strix.main(CFG, parser.parse_args(["strix", "up", "halo-q", "--box", "one"]), {}) == 0
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [e["event"] for e in events][:2] == ["starting", "progress"]
    assert events[-1]["event"] == "ready" and events[-1]["status"]["cluster"]["master"] == "one"
    assert strix.main(CFG, parser.parse_args(["strix", "up", "halo-q"]), {}) == 1
    assert "--box" in json.loads(capsys.readouterr().out.splitlines()[-1])["detail"]


def test_polling_leaves_a_failing_box_alone_for_a_minute_and_actions_retry_it(boxes, monkeypatch):
    fake = boxes(two=("halogen", "ready", "a"))
    fake.down["one"] = "Permission denied (publickey,password)."
    clock = [1000.0]
    monkeypatch.setattr(strix.time, "time", lambda: clock[0])
    probes = lambda host: sum(1 for h, c in fake.calls if h == host and "SPARK_SERVE_PROBE" in c)
    strix.status(CFG, poll=True)
    clock[0] += 30
    held = by_host(strix.status(CFG, poll=True))["one"]
    assert probes("one") == 1 and probes("two") == 2
    assert "Permission denied" in held["error"] and "held" not in held
    strix.status(CFG)  # an action's status always tries the box
    assert probes("one") == 2
    del fake.down["one"]
    clock[0] += 61
    assert by_host(strix.status(CFG, poll=True))["one"]["error"] is None
    assert json.loads((strix._backoff_file()).read_text()) == {}
