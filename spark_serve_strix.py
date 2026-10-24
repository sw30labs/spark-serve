"""AMD Strix Halo site: a thin client of halo-ctl on each box.

Each box owns engine switching (Halogen, the llama.cpp cluster, Lemonade,
vLLM; one at a time) under its own lock, shared with its GNOME widget. The
cluster spans the two boxes: the one that runs it is the master, and the
other's always-on ggml-rpc-server holds its share of the weights. Only this
client sees both boxes, so it keeps the pair consistent: starting the cluster
stops the other box first, and starting a model on the worker stops the
cluster first. Nothing here touches the Spark controller, its NCCL settings,
fencing or allocation records.

Catalog: [strix] with hosts, ctl and [strix.models.<id>] entries carrying the
same card fields as Spark models plus engine, engine_model and port. What each
box can serve comes from `halo-ctl models`, the models on its disk.
"""
from __future__ import annotations

import concurrent.futures
import datetime
import json
import shlex
import subprocess
import time
import uuid

ENGINES = ("halogen", "cluster", "lemonade", "vllm")
UNITS = {"halogen": "halogen.service", "cluster": "halo-cluster.service",
         "lemonade": "halo-lemonade.service", "vllm": "halo-vllm.service"}
RPC_UNIT = "ggml-rpc-server.service"
DEFAULT_CTL = "~/.local/bin/halo-ctl"
DEFAULT_KEY = "~/.config/halogen/api-key"
SSH_SECONDS = 30
UP_SECONDS = 1200
# halo-ctl returns once the cluster's process stays up and leaves the load to
# us. A first load onto a fresh worker streams its share of ~190 GB.
CLUSTER_READY_SECONDS = 1800
READY_POLL_SECONDS = 5
# Polling leaves a box that refused this Mac's key, or did not answer, alone for
# a minute: sshd's per-source penalties refuse an address that keeps failing to
# log in, and would then refuse the ssh-copy-id that fixes it too. Commands that
# act on a box always try it afresh.
BACKOFF_SECONDS = 60


class StrixError(RuntimeError):
    pass


def site(cfg: dict) -> dict:
    strix = cfg.get("strix")
    if not strix or not (strix.get("hosts") or strix.get("host")):
        raise StrixError("no [strix] hosts in the catalog")
    return strix


def hosts(cfg: dict) -> list[str]:
    """SSH aliases, which are also the LAN hostnames clients use; `host` is the one-box form."""
    strix = site(cfg)
    names = strix.get("hosts") or [strix["host"]]
    if not isinstance(names, list) or not all(isinstance(n, str) and n for n in names) or len(set(names)) != len(names):
        raise StrixError("[strix] hosts must be distinct hostnames")
    return list(names)


def box(cfg: dict, name: str | None) -> str:
    """The box a command acts on; it must be named once there is more than one."""
    names = hosts(cfg)
    if name is None:
        if len(names) == 1:
            return names[0]
        raise StrixError(f"choose a box with --box ({', '.join(names)})")
    if name not in names:
        raise StrixError(f"unknown Strix box {name!r}; [strix] hosts: {', '.join(names)}")
    return name


def models(cfg: dict) -> dict:
    return (cfg.get("strix") or {}).get("models") or {}


def resolve(cfg: dict, name: str) -> tuple[str, dict]:
    key = name.strip().lower()
    for mid, model in models(cfg).items():
        if key == mid or key in [a.lower() for a in model.get("aliases") or []]:
            return mid, model
    raise StrixError(f"unknown Strix model {name!r}; catalog: {', '.join(sorted(models(cfg)))}")


def ctl_id(model: dict) -> str:
    """The name halo-ctl takes and lists: the checkpoint path for vLLM, the served name otherwise."""
    return model["engine_model"] if model.get("engine") == "vllm" else model["served_name"]


def target(cfg: dict, name: str) -> tuple[str | None, dict]:
    """A catalog id or alias, or `<engine>:<name>` for a model on a box's disk that the catalog lacks."""
    engine, sep, rest = name.partition(":")
    if not (sep and engine in ENGINES and rest):
        return resolve(cfg, name)
    for mid, model in models(cfg).items():
        if model.get("engine") == engine and ctl_id(model) == rest:
            return mid, model
    vllm = engine == "vllm"
    return None, {"label": rest, "engine": engine, "engine_model": rest if vllm else None,
                  "served_name": rest.rsplit("/", 1)[-1] if vllm else rest}


def _reports(model: dict, reported: str | None) -> bool:
    """halo-ctl names a running model by the id it was started with; older ones by the served name."""
    return model.get("engine") == "halogen" or reported in {model.get("served_name"), model.get("engine_model")}


def ssh(host: str, command: str, timeout: float = SSH_SECONDS) -> str:
    try:
        done = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, command],
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise StrixError(f"{host}: no answer within {int(timeout)}s") from exc
    if done.returncode:
        detail = (done.stderr or done.stdout).strip().splitlines()
        raise StrixError(f"{host}: {detail[-1] if detail else f'exit {done.returncode}'}")
    return done.stdout


def _ctl(cfg: dict) -> str:
    return site(cfg).get("ctl") or DEFAULT_CTL


def _command(cfg: dict, *args) -> str:
    # The ctl path keeps its ~ for the remote shell; model names from a box's disk are quoted.
    return " ".join([_ctl(cfg), *(shlex.quote(str(a)) for a in args)])


def api_key(cfg: dict, host: str) -> str | None:
    """Read a box's API key for this process only; it is never stored on the Mac."""
    path = site(cfg).get("api_key_file") or DEFAULT_KEY
    out = ssh(host, f"cat {path} 2>/dev/null || true").strip()
    return out or None


def _match(cfg: dict, engine: str, reported: str) -> str | None:
    for mid, model in models(cfg).items():
        if model.get("engine") == engine and _reports(model, reported):
            return mid
    return None


# Runs on the box in one SSH round trip: halo-ctl state, stats and the models
# on its disk, whether its RPC worker is up and in use, then the serving
# engine's real context and token counters, read on localhost with the box's
# key. The key never leaves the box. Read-only.
_PROBE = r"""
import json, os, re, subprocess, urllib.request
CTL, KEY = os.path.expanduser(%(ctl)r), os.path.expanduser(%(key)r)
WANTED = {"vllm:generation_tokens_total", "vllm:prompt_tokens_total", "vllm:num_requests_running",
          "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc", "llamacpp:tokens_predicted_total",
          "llamacpp:prompt_tokens_total", "llamacpp:requests_processing", "llamacpp:requests_deferred",
          "llamacpp:kv_cache_usage_ratio", "llamacpp:n_decode_total"}
def sh(*args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return ""
def load(text, default=None):
    try:
        return json.loads(text) if text.strip() else default
    except ValueError:
        return default
key = open(KEY).read().strip() if os.path.exists(KEY) else ""
def get(url):
    request = urllib.request.Request(url, headers={"Authorization": "Bearer " + key} if key else {})
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.read(4 << 20).decode("utf-8", "replace")
    except Exception:
        return ""
state = json.loads(sh(CTL, "status", "--json"))
stats = load(sh(CTL, "stats"), {})
disk = load(sh(CTL, "models"))
rpc = sh("systemctl", "--user", "is-active", "ggml-rpc-server.service").strip() or "unknown"
# A master loading or serving through this box's RPC worker holds a connection to it.
rpc_clients = len([l for l in sh("ss", "-Htn", "state", "established", "( sport = :50053 )").splitlines() if l.strip()])
engine, port, served, text = state.get("engine"), state.get("port"), None, ""
if engine in ("halogen", "vllm") and port:
    try:
        model = json.loads(get("http://127.0.0.1:%%s/v1/models" %% port))["data"][0]
        served = model.get("max_model_len") or model.get("context_length")
    except Exception:
        pass
    text = get("http://127.0.0.1:%%s/metrics" %% port)
elif engine == "cluster" and port:
    try:
        served = (json.loads(get("http://127.0.0.1:%%s/props" %% port)).get("default_generation_settings") or {}).get("n_ctx")
    except Exception:
        pass
elif engine == "lemonade" and port:
    try:
        loaded = (json.loads(get("http://127.0.0.1:%%s/api/v1/health" %% port)).get("all_models_loaded") or [{}])[0]
        served = loaded.get("max_context_window")
        if loaded.get("backend_url"):
            text = get(loaded["backend_url"].rsplit("/v1", 1)[0] + "/metrics")
    except Exception:
        pass
counters = {}
for line in text.splitlines():
    match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", line.strip())
    if match and match.group(1) in WANTED:
        name, value = match.group(1), float(match.group(2))
        counters[name] = max(counters.get(name, 0.0), value) if "kv_cache" in name else counters.get(name, 0.0) + value
print(json.dumps({"state": state, "stats": stats, "served_context": served, "counters": counters,
                  "disk": disk, "rpc_worker": rpc, "rpc_clients": rpc_clients}))
"""

# Engine counters normalized to one shape; Halogen keeps halogen-stats' own fields.
_COUNTERS = {
    "vllm": {"active": "vllm:num_requests_running", "queued": "vllm:num_requests_waiting",
             "kv": "vllm:kv_cache_usage_perc", "generation": "vllm:generation_tokens_total",
             "prompt": "vllm:prompt_tokens_total"},
    "lemonade": {"active": "llamacpp:requests_processing", "queued": "llamacpp:requests_deferred",
                 # tokens_predicted_total moves only when a request finishes; decode steps
                 # move live and equal tokens with one slot and no drafts, as Lemonade runs here.
                 "kv": "llamacpp:kv_cache_usage_ratio", "generation": "llamacpp:n_decode_total",
                 "prompt": "llamacpp:prompt_tokens_total"},
}


def _metrics(engine, stats, counters):
    out = dict.fromkeys(("active", "queued", "kv_pct", "avg_gen_tps", "avg_prompt_tps", "last_gen_tps",
                         "draft_accept_pct", "generation_tokens_total", "prompt_tokens_total"))
    if engine == "halogen":
        out.update({key: stats.get(key) for key in (
            "active", "queued", "kv_pct", "avg_gen_tps", "avg_prompt_tps", "last_gen_tps", "draft_accept_pct")})
    names = _COUNTERS.get(engine)
    if names:
        kv = counters.get(names["kv"])
        out.update(active=counters.get(names["active"]), queued=counters.get(names["queued"]),
                   kv_pct=round(kv * 100, 2) if kv is not None else None,
                   generation_tokens_total=counters.get(names["generation"]),
                   prompt_tokens_total=counters.get(names["prompt"]))
    return out


def _available(disk) -> dict | None:
    """halo-ctl's on-disk models per engine, or None when its halo-ctl cannot list them."""
    if not isinstance(disk, dict):
        return None
    return {engine: [{"id": str(m["id"]), "label": str(m.get("label") or m["id"])}
                     for m in disk.get(engine) or [] if isinstance(m, dict) and m.get("id")]
            for engine in ENGINES}


def _box(cfg: dict, host: str) -> dict:
    script = _PROBE % {"ctl": _ctl(cfg), "key": site(cfg).get("api_key_file") or DEFAULT_KEY}
    raw = json.loads(ssh(host, "python3 - <<'SPARK_SERVE_PROBE'\n" + script + "\nSPARK_SERVE_PROBE").strip().splitlines()[-1])
    state, stats = raw.get("state") or {}, raw.get("stats") or {}
    engine, port = state.get("engine"), state.get("port")
    return {"host": host, **state,
            "model_id": _match(cfg, engine, state.get("model")) if engine in ENGINES else None,
            "url": f"http://{host}:{int(port)}/v1" if engine in ENGINES and port else None,
            "served_context": raw.get("served_context"),
            "resources": {key: stats.get(key) for key in (
                "ram_free_gib", "ram_total_gib", "gtt_used_gib", "gtt_total_gib",
                "busy_pct", "temp_c", "power_w", "last_gen_tps")},
            "metrics": _metrics(engine, stats, raw.get("counters") or {}),
            "available": _available(raw.get("disk")),
            "rpc_worker": raw.get("rpc_worker"), "rpc_clients": raw.get("rpc_clients"),
            "role": None, "peer": None, "error": None}


def _unreachable(host: str, error: str) -> dict:
    return {"host": host, "engine": "unknown", "phase": "unreachable", "model": "", "port": None, "engines": {},
            "inference": None, "model_id": None, "url": None, "served_context": None, "resources": None,
            "metrics": None, "available": None, "rpc_worker": None, "rpc_clients": None,
            "role": None, "peer": None, "error": error}


def _pair(boxes: list[dict]) -> dict:
    """Mark the cluster's master and worker; the pair is the two configured boxes."""
    if len(boxes) == 2:
        boxes[0]["peer"], boxes[1]["peer"] = boxes[1]["host"], boxes[0]["host"]
    by_host = {b["host"]: b for b in boxes}
    masters = [b for b in boxes if b["engine"] == "cluster" and b["phase"] in ("starting", "ready")]
    for master in masters:
        master["role"] = "master"
    for master in masters:
        worker = by_host.get(master["peer"])
        if worker and worker["role"] is None:
            worker["role"] = "worker"
    for b in boxes:
        # Its master may be out of sight, but the connection to its RPC worker is not.
        if b["role"] is None and b.get("rpc_clients"):
            b["role"] = "worker"
    cluster = None
    if masters:
        m = masters[0]
        cluster = {"master": m["host"], "worker": m["peer"], "phase": m["phase"], "model_id": m["model_id"],
                   "url": m["url"], "conflict": len(masters) > 1}
    return {"boxes": boxes, "cluster": cluster}


def _backoff_file():
    from spark_serve_controller import state_dir
    return state_dir() / "strix-backoff.json"


def _backoff() -> dict:
    try:
        held = json.loads(_backoff_file().read_text())
    except (OSError, ValueError):
        return {}
    return held if isinstance(held, dict) else {}


def status(cfg: dict, poll: bool = False) -> dict:
    """Every box's halo-ctl state, resources and live counters, read in parallel, with cluster roles.

    `poll` reports a box that failed within the last minute without trying it again.
    """
    from spark_serve_controller import atomic_json
    names, held, now = hosts(cfg), _backoff(), time.time()

    def one(host):
        failed = held.get(host) or {}
        if poll and now - float(failed.get("at") or 0) < BACKOFF_SECONDS:
            return {**_unreachable(host, failed.get("error") or f"{host}: no answer"), "held": True}
        try:
            return _box(cfg, host)
        except (StrixError, ValueError, KeyError, IndexError) as exc:
            return _unreachable(host, str(exc))

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(names)) as pool:
        boxes = list(pool.map(one, names))
    failed = {b["host"]: held[b["host"]] if b.pop("held", False) else {"at": now, "error": b["error"]}
              for b in boxes if b["error"]}
    if failed != {h: v for h, v in held.items() if h in names}:
        atomic_json(_backoff_file(), failed)
    return _pair(boxes)


def _by_host(current: dict) -> dict:
    return {b["host"]: b for b in current["boxes"]}


def _describe(b: dict) -> str:
    return "the cluster" if b["engine"] == "cluster" else f"{b['engine']} {b['model']}".strip()


def _busy(b: dict) -> bool:
    return b["engine"] in ENGINES and b["phase"] != "stopped"


def _last_log(host: str, engine: str) -> str:
    try:
        out = ssh(host, f"journalctl --user -u {UNITS[engine]} -n 1 --no-pager -o cat")
    except StrixError:
        return ""
    return out.strip().splitlines()[-1] if out.strip() else ""


def _wait_ready(cfg: dict, host: str, model: dict, seconds: float, emit) -> dict:
    """Poll the box until it serves the model; fail as soon as its engine exits."""
    engine = model["engine"]
    started = noted = time.monotonic()
    while True:
        try:
            me = _box(cfg, host)
        except (StrixError, ValueError, KeyError, IndexError) as exc:
            me = _unreachable(host, str(exc))  # a dropped poll is not a failed start
        if not me["error"]:
            if me["engine"] == engine and me["phase"] == "ready":
                if not _reports(model, me["model"]):
                    raise StrixError(f"{model['label']} did not become ready on {host} (box serves {me['model']})")
                return status(cfg)
            if me["engine"] != engine or me["phase"] == "stopped":
                raise StrixError(f"{host}: {_last_log(host, engine) or f'{engine} exited during startup'}")
        elapsed = time.monotonic() - started
        if elapsed > seconds:
            raise StrixError(f"{model['label']} did not become ready on {host} within {int(seconds)}s "
                             f"(box reports {me['engine']}/{me['phase']})")
        if time.monotonic() - noted >= 30:
            emit({"event": "progress", "detail": f"{host}: {engine} {me['phase']}, {int(elapsed)}s"})
            noted = time.monotonic()
        time.sleep(READY_POLL_SECONDS)


def up(cfg: dict, name: str, host: str | None = None, emit=None) -> dict:
    """Serve a catalog model, or `<engine>:<name>` from the box's disk, on one box.

    The cluster starts with that box as master and stops whatever the other
    box serves first. A model on the cluster's worker stops the cluster first.
    halo-ctl itself stops the box's own engine.
    """
    emit = emit or (lambda event: None)
    host = box(cfg, host)
    mid, model = target(cfg, name)
    model = {**model, "label": model.get("label") or mid or name}
    engine, wanted = model.get("engine"), ctl_id(model) if model.get("engine") in ENGINES else None
    if engine not in ENGINES:
        raise StrixError(f"{mid or name}: engine must be one of {', '.join(ENGINES)}")
    boxes = _by_host(status(cfg))
    me = boxes[host]
    if me["error"]:
        raise StrixError(me["error"])
    disk = (me["available"] or {}).get(engine) if me["available"] is not None else None
    if disk is not None and wanted not in [m["id"] for m in disk]:
        raise StrixError(f"{host} has no {wanted} on disk for {engine}")
    if engine == "cluster":
        peer = boxes.get(me["peer"])
        if peer is None:
            raise StrixError("the cluster needs two [strix] hosts")
        if peer["error"]:
            raise StrixError(f"cannot check the cluster's worker: {peer['error']}")
        if peer["rpc_worker"] != "active":
            raise StrixError(f"{peer['host']}: {RPC_UNIT} is {peer['rpc_worker']}; the cluster needs it running")
        if not (me["role"] == "master" and _reports(model, me["model"])):
            if _busy(peer):
                emit({"event": "progress", "detail": f"Stopping {_describe(peer)} on {peer['host']}: the cluster needs its memory"})
                ssh(peer["host"], _command(cfg, "stop"), timeout=120)
            elif peer["rpc_clients"]:
                raise StrixError(f"{peer['host']}: its RPC worker is already in use by another master")
            emit({"event": "progress", "detail": f"Starting the cluster with {host} as master"})
            ssh(host, _command(cfg, "up", "cluster", wanted), timeout=UP_SECONDS)
        return _wait_ready(cfg, host, model, CLUSTER_READY_SECONDS, emit)
    if me["role"] == "worker":
        master = boxes.get(me["peer"])
        if master is None or master["error"] or master["role"] != "master":
            raise StrixError(f"{host} is busy as a cluster worker and its master cannot be stopped from here")
        emit({"event": "progress", "detail": f"Stopping the cluster on {master['host']}: {host} is its worker"})
        ssh(master["host"], _command(cfg, "stop"), timeout=120)
        if _box(cfg, host)["rpc_clients"]:
            raise StrixError(f"{host}: its RPC worker is still in use; stop the master there first")
    args = ["up", engine]
    if engine != "halogen":
        args.append(wanted)
    if engine == "vllm" and model.get("max_model_len"):
        args.append(int(model["max_model_len"]))
    emit({"event": "progress", "detail": f"Starting {model['label']} on {host}"})
    ssh(host, _command(cfg, *args), timeout=UP_SECONDS)
    return _wait_ready(cfg, host, model, 60, emit)


def stop(cfg: dict, host: str | None = None, emit=None) -> dict:
    """Stop one box, or all of them. Stopping the cluster's worker stops the cluster."""
    emit = emit or (lambda event: None)
    boxes = _by_host(status(cfg))
    if host is None:
        # Masters first, so their workers let go before anything else stops.
        targets = sorted(boxes, key=lambda h: boxes[h]["role"] != "master")
    else:
        host = box(cfg, host)
        targets = [host]
        if boxes[host]["role"] == "worker" and boxes[host]["peer"]:
            targets.insert(0, boxes[host]["peer"])
    failed = []
    for name in targets:
        if boxes[name]["error"]:
            failed.append(boxes[name]["error"])
            continue
        emit({"event": "progress", "detail": f"Stopping {name}"})
        ssh(name, _command(cfg, "stop"), timeout=120)
    if failed:
        raise StrixError("; ".join(failed))
    return status(cfg)


def connection(cfg: dict, host: str | None = None) -> dict:
    """What a client needs for the engine serving on a box: its master's, for the cluster's worker."""
    host = box(cfg, host)
    boxes = _by_host(status(cfg))
    me = boxes[host]
    if me["error"]:
        raise StrixError(me["error"])
    if me["role"] == "worker" and boxes.get(me["peer"], {}).get("role") == "master":
        me = boxes[me["peer"]]
    if not _busy(me):
        raise StrixError(f"nothing is serving on {host}")
    model = models(cfg).get(me["model_id"]) or {}
    served = model.get("served_name") or (me["model"].rsplit("/", 1)[-1] if me["engine"] == "vllm" else me["model"])
    return {"host": me["host"], "engine": me["engine"], "base_url": me["url"], "model": served,
            "api_key": api_key(cfg, me["host"])}


def logs(cfg: dict, host: str | None = None, lines: int = 60) -> dict:
    """Recent journal lines of the box's engine, of its RPC worker while it is one, else of every engine."""
    host = box(cfg, host)
    me = _by_host(status(cfg))[host]
    if me["error"]:
        raise StrixError(me["error"])
    if _busy(me):
        units = [UNITS[me["engine"]]]
    elif me["role"] == "worker":
        units = [RPC_UNIT]
    else:
        units = list(UNITS.values())
    lines = max(1, min(int(lines), 500))
    out = ssh(host, f"journalctl --user --no-pager -o short-iso -n {lines} " + " ".join(f"-u {u}" for u in units))
    return {"host": host, "units": units, "lines": out.splitlines()[-lines:]}


def catalog(cfg: dict, performance: dict) -> list[dict]:
    """Card entries in the same shape as `spark-serve list --json`, ranked within this site."""
    from spark_serve_ranking import modalities, score_models
    site(cfg)  # no [strix] hosts: the app hides the boxes
    out = []
    for mid, m in models(cfg).items():
        out.append({
            "id": mid, "label": m.get("label") or mid, "aliases": list(m.get("aliases") or []),
            "served_name": m.get("served_name", ""), "ctx": int(m.get("max_model_len") or 0),
            "image": m.get("engine_model") or m.get("engine", ""), "notes": m.get("notes", ""),
            "backend": m.get("engine", ""), "topology": "strix-cluster" if m.get("engine") == "cluster" else "strix",
            "wrapper": m.get("engine", ""), "hermes_provider": "", "performance": performance.get(mid),
            "quant": m.get("quant"), "quant_bits": m.get("quant_bits"),
            "quant_scope": m.get("quant_scope"), "quant_detail": m.get("quant_detail"),
            "modalities": modalities(m), "ctl_id": ctl_id(m) if m.get("engine") in ENGINES else None,
        })
    ranking = score_models(out)
    for item in out:
        item["ranking"] = ranking.get(item["id"])
    return out


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def bench(cfg: dict, host: str | None = None, *, concurrency=1, requests=3, max_tokens=128, prompt_tokens=512,
          thinking=None, emit=None) -> int:
    """Benchmark the catalog model serving on a box (the cluster, for its worker) under the Spark protocol.

    Reuses its SSE accounting, prompts and summary so results are comparable and
    land in the same history (and `catalog_tps_bench.py summarize`).
    """
    from spark_serve_benchmarks import (DEADLINE_SECONDS, REQUEST_SECONDS, _prompt, _root, _save,
                                        _stream_request, _summary)
    emit = emit or (lambda event: print(json.dumps(event), flush=True))
    for name, value, low, high in (("concurrency", concurrency, 1, 4), ("requests", requests, 1, 12),
                                  ("max_tokens", max_tokens, 16, 512), ("prompt_tokens", prompt_tokens, 32, 32768)):
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise StrixError(f"benchmark {name} must be in {low}..{high}")
    if concurrency > requests:
        raise StrixError("benchmark concurrency cannot exceed request count")
    host = box(cfg, host)
    boxes = _by_host(status(cfg))
    before = boxes[host]
    if before["role"] == "worker" and boxes.get(before["peer"], {}).get("role") == "master":
        before = boxes[before["peer"]]
    mid = before.get("model_id")
    if before["error"] or not mid or before.get("phase") != "ready":
        raise StrixError(f"no catalog model is ready on {host}")
    model = models(cfg)[mid]
    key = api_key(cfg, before["host"])
    run_id = uuid.uuid4().hex
    run = {"id": run_id, "site": "strix", "kind": "decode", "status": "running", "started_at": _now(),
           "model": mid, "served_name": model["served_name"], "node": before["host"],
           "hosts": [before["host"], before["peer"]] if before["role"] == "master" else [before["host"]],
           "endpoint": before["url"], "runtime": {"engine": model["engine"]},
           "served_context": before.get("served_context"),
           "config": {"concurrency": concurrency, "requests": requests, "max_tokens": max_tokens,
                      "prompt_tokens": prompt_tokens, "prompt_lengths": [prompt_tokens] * requests,
                      "thinking": thinking,
                      "reasoning_mode": "server-default" if thinking is None else "requested-on" if thinking else "requested-off",
                      "deadline_seconds": DEADLINE_SECONDS, "prompt_token_target_is_approximate": True},
           "warmup": None, "results": [],
           "limitations": ["Synthetic speed sample, not a quality evaluation.",
                           "Token counts are server usage, never streamed chunk counts.",
                           "Identity is checked before and after the run, not continuously."]}
    started = time.monotonic()
    deadline = started + DEADLINE_SECONDS
    completion_times = []

    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("Overall benchmark deadline exceeded")

    def request(index, warmup=False):
        record = {"index": index, "phase": "warmup" if warmup else "measurement",
                  "requested_prompt_tokens": prompt_tokens}
        try:
            payload = {"model": model["served_name"],
                       "messages": [{"role": "user", "content": _prompt(prompt_tokens, uuid.uuid4().hex)}],
                       "temperature": 0, "max_tokens": min(32, max_tokens) if warmup else max_tokens,
                       "stream": True, "stream_options": {"include_usage": True}}
            if thinking is not None:
                payload["chat_template_kwargs"] = {"enable_thinking": thinking}
            response = _stream_request(before["url"], payload,
                                       min(REQUEST_SECONDS, deadline - time.monotonic()), check, api_key=key)
            first = response["metrics"].get("first_model_output_seconds")
            response["metrics"]["prefill_tokens_per_second_estimate"] = (
                round(response["usage"]["prompt_tokens"] / first, 3) if first and first > 0 else None)
            record.update(passed=True, usage=response["usage"], metrics=response["metrics"],
                          finish_reason=response["finish_reason"], reasoning_observed=bool(response.get("reasoning")))
        except Exception as exc:
            record.update(passed=False, cancelled=False, error=f"{type(exc).__name__}: {exc}")
        if not warmup:
            completion_times.append(time.monotonic())
        return record

    measurement_started = None
    try:
        emit({"event": "started", "run": {k: v for k, v in run.items() if k != "results"}})
        run["warmup"] = request(-1, warmup=True)
        if run["warmup"]["passed"]:
            measurement_started = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                for future in concurrent.futures.as_completed([pool.submit(request, i) for i in range(requests)]):
                    run["results"].append(future.result())
                    emit({"event": "progress", "completed": len(run["results"]), "total": requests})
        after = _by_host(status(cfg))[before["host"]]
        rows = [run["warmup"], *run["results"]]
        if after.get("model_id") != mid or after.get("phase") != "ready":
            run.update(status="failed", error="Serving model changed during the benchmark")
        else:
            run["status"] = "completed" if len(run["results"]) == requests and all(r["passed"] for r in rows) else "failed"
    except Exception as exc:
        run.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        run["finished_at"] = _now()
        run["elapsed_seconds"] = round(time.monotonic() - started, 4)
        run["results"].sort(key=lambda r: r["index"])
        wall = max(completion_times) - measurement_started if completion_times and measurement_started else 0
        run["summary"] = _summary(run["results"], wall)
        (_root() / "benchmarks").mkdir(parents=True, exist_ok=True)
        _save(_root() / "benchmarks" / (run_id + ".json"), run)
        emit({"event": "result", "run": run})
    return 0 if run["status"] == "completed" else 1


def _emit_json(value):
    print(json.dumps(value), flush=True)


def _box_line(b: dict) -> str:
    if b["error"]:
        return b["error"] if b["error"].startswith(f"{b['host']}: ") else f"{b['host']}: {b['error']}"
    role = {"master": " · cluster master", "worker": " · cluster worker"}.get(b["role"], "")
    serving = f" · {b['model']} · {b['url']}" if _busy(b) and b.get("url") else ""
    return f"{b['host']}: {b['engine']} {b['phase']}{role}{serving}"


def main(cfg: dict, args, performance: dict) -> int:
    """`spark-serve strix …` entry point; JSON on stdout, errors as an error event."""
    try:
        if args.strix_cmd == "list":
            items = catalog(cfg, performance)
            if args.json:
                print(json.dumps(items, indent=2))
            else:
                for item in items:
                    star = " ★" if (item.get("ranking") or {}).get("best") else ""
                    print(f"{item['id']:<28} {item['backend']:<9} {item['served_name']:<40}{star}")
        elif args.strix_cmd == "status":
            current = status(cfg, poll=True)
            if args.json:
                _emit_json(current)
            else:
                for b in current["boxes"]:
                    print(_box_line(b))
        elif args.strix_cmd == "up":
            _emit_json({"event": "starting", "model": args.model, "box": args.box})
            _emit_json({"event": "ready", "status": up(cfg, args.model, args.box, emit=_emit_json)})
        elif args.strix_cmd == "stop":
            _emit_json({"event": "stopped", "status": stop(cfg, args.box, emit=_emit_json)})
        elif args.strix_cmd == "connection":
            details = connection(cfg, args.box)
            if args.json:
                _emit_json(details)
            else:
                print(f"Engine:   {details['engine']}\nBase URL: {details['base_url']}\n"
                      f"Model:    {details['model']}\nAPI key:  {details['api_key'] or '(no API key set)'}")
        elif args.strix_cmd == "logs":
            out = logs(cfg, args.box, args.lines)
            if args.json:
                _emit_json(out)
            else:
                print("\n".join(out["lines"]))
        elif args.strix_cmd == "bench":
            return bench(cfg, args.box, concurrency=args.concurrency, requests=args.requests,
                         max_tokens=args.max_tokens, prompt_tokens=args.prompt_tokens, thinking=args.thinking)
        return 0
    except (StrixError, ValueError, KeyError) as exc:
        _emit_json({"event": "error", "detail": str(exc)})
        return 1


def add_parser(sub) -> None:
    strix = sub.add_parser("strix", help="AMD Strix Halo boxes: one engine at a time each through halo-ctl, "
                                         "or the llama.cpp cluster across both")
    commands = strix.add_subparsers(dest="strix_cmd", required=True)
    lst = commands.add_parser("list", help="Strix catalog with measured speed and ranking")
    lst.add_argument("--json", action="store_true")
    st = commands.add_parser("status", help="every box: serving engine, model, cluster role, models on disk")
    st.add_argument("--json", action="store_true")
    upc = commands.add_parser("up", help="serve a model on a box (stops what conflicts; the cluster takes both boxes)")
    upc.add_argument("model", help="catalog id or alias, or <engine>:<name> for a model on the box's disk")
    upc.add_argument("--box", help="the box to serve on; for the cluster, its master")
    upc.add_argument("--json", action="store_true", help="JSON events are the default output")
    stp = commands.add_parser("stop", help="stop a box (its cluster, for the worker), or every box")
    stp.add_argument("--box", help="default: every box")
    stp.add_argument("--json", action="store_true", help="JSON events are the default output")
    con = commands.add_parser("connection", help="base URL, model and API key of what a box serves")
    con.add_argument("--box")
    con.add_argument("--json", action="store_true")
    lg = commands.add_parser("logs", help="recent journal lines of a box's engine")
    lg.add_argument("--box")
    lg.add_argument("-n", "--lines", type=int, default=60)
    lg.add_argument("--json", action="store_true")
    b = commands.add_parser("bench", help="decode benchmark of a box's serving model (same protocol as Sparks)")
    b.add_argument("--box", help="the box serving the model (either box, for the cluster)")
    b.add_argument("--concurrency", type=int, default=1)
    b.add_argument("--requests", type=int, default=3)
    b.add_argument("--max-tokens", type=int, default=128)
    b.add_argument("--prompt-tokens", type=int, default=512)
    thinking = b.add_mutually_exclusive_group()
    thinking.add_argument("--thinking", dest="thinking", action="store_true")
    thinking.add_argument("--no-thinking", dest="thinking", action="store_false")
    b.set_defaults(thinking=None)
    b.add_argument("--json", action="store_true", help="JSON events are the default output")
