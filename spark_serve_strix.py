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

Always-on services (ADR 0019): an entry with engine = "service" is a model a
systemd user unit on one box serves outside halo-ctl (box, unit, port). The
app shows it and reads its health, but never starts or stops it, and leaves
that box's halo-ctl alone: a halo-ctl up or stop there would kill whatever
holds its engine ports and take the box's GPU memory from the service.

`strix use` points Hermes at a model that is already ready (the same config
fields as Spark's `use`). It does not start or stop anything.
"""
from __future__ import annotations

import concurrent.futures
import datetime
import json
import shlex
import subprocess
import time
import uuid
from urllib.parse import urlsplit

ENGINES = ("halogen", "cluster", "lemonade", "vllm")
SERVICE = "service"   # an always-on systemd unit outside halo-ctl (ADR 0019)
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


def pinned_master(cfg: dict) -> str | None:
    """The box pinned as the cluster's master by `[strix] master`, or None when either may be."""
    pinned = site(cfg).get("master")
    if pinned is None:
        return None
    if pinned not in hosts(cfg):
        raise StrixError(f"[strix] master {pinned!r} is not one of the hosts: {', '.join(hosts(cfg))}")
    return pinned


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


def services(cfg: dict, host: str | None = None) -> dict:
    """Always-on service entries (ADR 0019) by id, of one box or all; each needs box, unit, port and served_name."""
    out = {}
    for mid, m in models(cfg).items():
        if m.get("engine") != SERVICE:
            continue
        missing = [k for k in ("box", "unit", "port", "served_name") if not m.get(k)]
        if missing:
            raise StrixError(f"[strix.models.{mid}] is a service but lacks {', '.join(missing)}")
        if m["box"] not in hosts(cfg):
            raise StrixError(f"[strix.models.{mid}] box {m['box']!r} is not one of the hosts: {', '.join(hosts(cfg))}")
        if host is None or m["box"] == host:
            out[mid] = m
    return out


def pinned(cfg: dict, host: str | None) -> list[str]:
    """The always-on units on a box; spark-serve runs no halo-ctl up or stop on a box that has any."""
    return [m["unit"] for m in services(cfg, host).values()] if host else []


def _pinned_error(cfg: dict, host: str) -> StrixError:
    return StrixError(f"{host} runs always-on {', '.join(pinned(cfg, host))} outside halo-ctl; "
                      "spark-serve starts and stops nothing there (ADR 0019)")


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


def needs_bearer(model: dict) -> bool:
    """Whether Hermes must send the box API key for this card.

    halo-ctl engines always do. Always-on services do not (ADR 0019: those
    llama-servers take none), except ``halogen.service`` — Halogen's :8731
    server requires its bearer — and any service card that sets ``api_key = true``.
    ``api_key = false`` turns the bearer off. The key itself is never in the catalog.
    """
    flag = model.get("api_key")
    if isinstance(flag, bool):
        return flag
    if model.get("engine") == SERVICE:
        return model.get("unit") == "halogen.service"
    return model.get("engine") in ENGINES


def _client_base_url(url: str | None) -> str | None:
    """http(s) URL ending in /v1 with no userinfo, query, or fragment."""
    if not isinstance(url, str) or not url:
        return None
    try:
        parsed = urlsplit(url)
    except ValueError:
        return None
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path.rstrip("/") != "/v1"):
        return None
    return url.rstrip("/")


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
SERVICES = %(services)r
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
def get(url, auth=True):
    request = urllib.request.Request(url, headers={"Authorization": "Bearer " + key} if key and auth else {})
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
def count(text):
    counters = {}
    for line in text.splitlines():
        match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", line.strip())
        if match and match.group(1) in WANTED:
            name, value = match.group(1), float(match.group(2))
            counters[name] = max(counters.get(name, 0.0), value) if "kv_cache" in name else counters.get(name, 0.0) + value
    return counters
# Always-on units outside halo-ctl (ADR 0019). The fourth field is whether this
# card's server requires the box bearer (halogen.service does). The key is read
# on the box and is not returned. Keyless llama-servers are still probed without one.
services = []
for sid, unit, sport, auth in SERVICES:
    item = {"id": sid, "unit": unit, "port": sport, "model": None, "served_context": None,
            "active": sh("systemctl", "--user", "is-active", unit).strip() or "unknown"}
    try:
        item["model"] = json.loads(get("http://127.0.0.1:%%s/v1/models" %% sport, auth))["data"][0].get("id")
    except Exception:
        pass
    try:
        item["served_context"] = (json.loads(get("http://127.0.0.1:%%s/props" %% sport, auth)).get("default_generation_settings") or {}).get("n_ctx")
    except Exception:
        pass
    item["counters"] = count(get("http://127.0.0.1:%%s/metrics" %% sport, auth))
    services.append(item)
print(json.dumps({"state": state, "stats": stats, "served_context": served, "counters": count(text),
                  "disk": disk, "rpc_worker": rpc, "rpc_clients": rpc_clients, "services": services}))
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
    # Always-on llama-servers run several slots, so decode steps undercount; count finished tokens.
    SERVICE: {"active": "llamacpp:requests_processing", "queued": "llamacpp:requests_deferred",
              "kv": "llamacpp:kv_cache_usage_ratio", "generation": "llamacpp:tokens_predicted_total",
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


def _service(cfg: dict, host: str, raw: dict) -> dict:
    """One always-on service as the box reports it: ready only when its port serves the catalog's name."""
    model = models(cfg).get(raw.get("id")) or {}
    active, served = raw.get("active"), raw.get("model")
    if active == "active" and served == model.get("served_name"):
        phase = "ready"
    elif active == "active" and served:
        phase = "mismatch"      # the port answers with another model
    elif active in ("active", "activating", "reloading"):
        phase = "starting"
    else:
        phase = "stopped"
    port = raw.get("port") or model.get("port")
    return {"id": raw.get("id"), "label": model.get("label") or raw.get("id"), "unit": raw.get("unit"),
            "active": active, "phase": phase, "model": served, "port": port,
            "url": f"http://{host}:{int(port)}/v1" if port else None,
            "served_context": raw.get("served_context"),
            "outputs": list(model.get("outputs") or []),
            "metrics": _metrics(SERVICE, {}, raw.get("counters") or {})}


def _box(cfg: dict, host: str) -> dict:
    specs = services(cfg, host)
    script = _PROBE % {"ctl": _ctl(cfg), "key": site(cfg).get("api_key_file") or DEFAULT_KEY,
                       "services": [[sid, m["unit"], int(m["port"]), needs_bearer(m)] for sid, m in specs.items()]}
    raw = json.loads(ssh(host, "python3 - <<'SPARK_SERVE_PROBE'\n" + script + "\nSPARK_SERVE_PROBE").strip().splitlines()[-1])
    state, stats = raw.get("state") or {}, raw.get("stats") or {}
    svc = [_service(cfg, host, s) for s in raw.get("services") or []]
    # halo-ctl finds its engines by port: an always-on service on :8081 reads as the cluster
    # being ready. A report that is one of this box's services is that service, not an engine.
    shadowed = None
    for s in svc:
        if state.get("engine") in ENGINES and state.get("port") == s["port"] and state.get("model") == s["model"]:
            shadowed = f"halo-ctl reads {s['unit']} on :{s['port']} as its {state['engine']} engine"
            state = {**state, "engine": "none", "phase": "stopped", "model": "", "port": None,
                     "engines": {**(state.get("engines") or {}), state["engine"]: "stopped"}}
            raw = {**raw, "served_context": None, "counters": {}}
            break
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
            "services": svc, "pinned": pinned(cfg, host), "shadowed": shadowed,
            "role": None, "peer": None, "error": None}


def _unreachable(host: str, error: str) -> dict:
    return {"host": host, "engine": "unknown", "phase": "unreachable", "model": "", "port": None, "engines": {},
            "inference": None, "model_id": None, "url": None, "served_context": None, "resources": None,
            "metrics": None, "available": None, "rpc_worker": None, "rpc_clients": None,
            "services": [], "pinned": [], "shadowed": None,
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
    return {**_pair(boxes), "master": pinned_master(cfg)}


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
    halo-ctl itself stops the box's own engine, including another cluster
    model on the same master.
    """
    emit = emit or (lambda event: None)
    mid, model = target(cfg, name)
    if model.get("engine") == SERVICE:
        raise StrixError(f"{model.get('label') or mid} is always on: {model.get('unit')} on {model.get('box')} "
                         "serves it outside halo-ctl, and spark-serve never starts or stops it (ADR 0019)")
    master = pinned_master(cfg) if model.get("engine") == "cluster" else None
    if master:
        if host not in (None, master):
            raise StrixError(f"the cluster's master is pinned to {master} ([strix] master); start it with --box {master}")
        host = master
    host = box(cfg, host)
    if pinned(cfg, host):
        raise _pinned_error(cfg, host)
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
        if pinned(cfg, peer["host"]):
            raise StrixError(f"the cluster needs {peer['host']} as its worker, but {_pinned_error(cfg, peer['host'])}")
        if peer["error"]:
            raise StrixError(f"cannot check the cluster's worker: {peer['error']}")
        if peer["rpc_worker"] != "active":
            raise StrixError(f"{peer['host']}: {RPC_UNIT} is {peer['rpc_worker']}; the cluster needs it running")
        if not (me["role"] == "master" and _reports(model, me["model"])):
            if _busy(peer):
                emit({"event": "progress", "detail": f"Stopping {_describe(peer)} on {peer['host']}: the cluster needs its memory"})
                ssh(peer["host"], _command(cfg, "stop"), timeout=120)
            elif me["role"] == "master":
                # The worker's connection is this master's; halo-ctl swaps the model here.
                emit({"event": "progress", "detail": f"Replacing the cluster's {me['model']} on {host}"})
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
        for name in targets:
            if pinned(cfg, name):
                raise _pinned_error(cfg, name)
    for name in [t for t in targets if pinned(cfg, t)]:
        emit({"event": "progress", "detail": f"Leaving {name} alone: {', '.join(pinned(cfg, name))} always on"})
        targets.remove(name)
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
        ready = [s for s in me.get("services") or [] if s["phase"] == "ready"]
        if not ready:
            raise StrixError(f"nothing is serving on {host}")
        # Always-on llama-servers take no key; the first is the catalog's first service on the box.
        return {"host": me["host"], "engine": SERVICE, "base_url": ready[0]["url"], "model": ready[0]["model"],
                "api_key": None, "unit": ready[0]["unit"],
                "services": [{"id": s["id"], "label": s["label"], "unit": s["unit"], "model": s["model"],
                              "base_url": s["url"], "outputs": s["outputs"]} for s in ready]}
    model = models(cfg).get(me["model_id"]) or {}
    served = model.get("served_name") or (me["model"].rsplit("/", 1)[-1] if me["engine"] == "vllm" else me["model"])
    return {"host": me["host"], "engine": me["engine"], "base_url": me["url"], "model": served,
            "api_key": api_key(cfg, me["host"])}


def _ready_for(box_status: dict, mid: str | None, model: dict) -> bool:
    """The box is serving this catalog card (or disk model) and has a base URL."""
    if (box_status.get("error") or box_status.get("phase") != "ready"
            or box_status.get("engine") != model.get("engine") or not box_status.get("url")):
        return False
    if model.get("engine") == "halogen":
        return True
    if mid and box_status.get("model_id"):
        return box_status["model_id"] == mid
    return _reports(model, box_status.get("model"))


def hermes_selection(cfg: dict, name: str, host: str | None = None) -> dict:
    """Live endpoint for a selected Strix model that is already ready.

    Does not start or stop anything. A service uses its own box. The cluster
    uses its master, including when ``--box`` names the worker. ``hermes_base_url``
    on the card replaces the live host URL (an IPv4 pin, for example). The API
    key is included only when the card requires a bearer, and is read from the
    box rather than the catalog.
    """
    mid, model = target(cfg, name)
    if model.get("engine") == SERVICE:
        if not mid:
            raise StrixError(f"unknown Strix model {name!r}")
        home = model["box"]
        if host is not None and box(cfg, host) != home:
            raise StrixError(f"{model.get('label') or mid} is always on {home}")
        current = _by_host(status(cfg))[home]
        if current["error"]:
            raise StrixError(current["error"])
        service = next((item for item in current.get("services") or [] if item.get("id") == mid), None)
        if not service or service.get("phase") != "ready" or not service.get("url"):
            raise StrixError(f"{model.get('label') or mid} is not ready on {home}")
        served = model.get("served_name") or service.get("model")
        live, where = service["url"], home
    else:
        if host is not None:
            host = box(cfg, host)
        current = status(cfg)
        by_host = _by_host(current)
        if host is not None and model.get("engine") == "cluster":
            me = by_host[host]
            peer = by_host.get(me.get("peer") or "")
            if me.get("role") == "worker" and peer and peer.get("role") == "master":
                host = me["peer"]
        matches = [item for item in current["boxes"]
                   if _ready_for(item, mid, model) and (host is None or item["host"] == host)]
        label = model.get("label") or name
        if host is not None and not matches:
            raise StrixError(f"{label} is not ready on {host}")
        if len(matches) != 1:
            raise StrixError("choose the box serving it with --box" if len(matches) > 1 else f"{label} is not ready")
        me = matches[0]
        served = model.get("served_name") or (
            me["model"].rsplit("/", 1)[-1] if me["engine"] == "vllm" else me["model"])
        live, where = me.get("url"), me["host"]
    if model.get("hermes_base_url"):
        url = _client_base_url(model.get("hermes_base_url"))
        if url is None:
            raise StrixError(f"{mid or name}: hermes_base_url must be an http(s) URL ending in /v1 without credentials")
    else:
        url = _client_base_url(live)
    if not url or not served:
        raise StrixError(f"{model.get('label') or name} has no usable base URL")
    key = None
    if needs_bearer(model):
        key = api_key(cfg, where)
        if not key:
            path = site(cfg).get("api_key_file") or DEFAULT_KEY
            raise StrixError(f"{where}: no API key in {path}; Hermes was not changed")
    return {"id": mid, "model": model, "host": where, "base_url": url, "served": served, "api_key": key}


def use(cfg: dict, name: str, host: str | None, hermes_target: str, patch) -> dict:
    """Write the selected ready Strix model into Hermes. ``patch`` is spark-serve._hermes_patch."""
    if hermes_target not in ("local", "spark"):
        raise StrixError("Hermes target must be local or spark")
    if patch is None:
        raise StrixError("Hermes update is unavailable")
    hermes_host = None
    if hermes_target == "spark":
        head = (cfg.get("cluster") or {}).get("head")
        if not isinstance(head, str) or not head:
            raise StrixError("Hermes (Spark) needs cluster.head")
        hermes_host = head
    chosen = hermes_selection(cfg, name, host)
    model = dict(chosen["model"])
    model["served_name"] = chosen["served"]
    if not model.get("hermes_provider"):
        model["hermes_provider"] = "amd-halo"
    message, ok = patch(
        cfg, chosen["id"] or name, model, False,
        remote_host=hermes_host, base_url=chosen["base_url"], api_key=chosen["api_key"],
        provider_label="AMD Strix Halo", move_ds4=False,
    )
    secret = chosen["api_key"]
    if secret and secret in (message or ""):
        message = message.replace(secret, "***")
    if not ok:
        raise StrixError(message)
    return {"message": message, "host": chosen["host"], "served": chosen["served"],
            "base_url": chosen["base_url"], "provider": model["hermes_provider"],
            "hermes_target": hermes_target, "hermes_host": hermes_host,
            "label": model.get("label") or chosen["served"]}


def logs(cfg: dict, host: str | None = None, lines: int = 60) -> dict:
    """Recent journal lines of the box's engine, of its RPC worker while it is one, else of every engine."""
    host = box(cfg, host)
    me = _by_host(status(cfg))[host]
    if me["error"]:
        raise StrixError(me["error"])
    if _busy(me):
        units = [UNITS[me["engine"]]]
    elif me.get("services"):
        units = [s["unit"] for s in me["services"]]
    elif me["role"] == "worker":
        units = [RPC_UNIT]
    else:
        units = list(UNITS.values())
    lines = max(1, min(int(lines), 500))
    out = ssh(host, f"journalctl --user --no-pager -o short-iso -n {lines} " + " ".join(f"-u {u}" for u in units))
    return {"host": host, "units": units, "lines": out.splitlines()[-lines:]}


def catalog(cfg: dict, performance: dict, dates: dict | None = None) -> list[dict]:
    """Card entries in the same shape as `spark-serve list --json`, ranked within this site.

    ``dates`` maps id -> {"updated", "updated_source"} (spark_serve_freshness).
    """
    dates = dates or {}
    from spark_serve_ranking import modalities, score_models
    site(cfg)  # no [strix] hosts: the app hides the boxes
    out = []
    for mid, m in models(cfg).items():
        out.append({
            "id": mid, "label": m.get("label") or mid, "aliases": list(m.get("aliases") or []),
            "served_name": m.get("served_name", ""), "ctx": int(m.get("max_model_len") or 0),
            "image": m.get("engine_model") or m.get("unit") or m.get("engine", ""), "notes": m.get("notes", ""),
            "backend": m.get("engine", ""),
            "topology": {"cluster": "strix-cluster", SERVICE: "strix-service"}.get(m.get("engine"), "strix"),
            "wrapper": m.get("engine", ""), "hermes_provider": "", "performance": performance.get(mid),
            "quant": m.get("quant"), "quant_bits": m.get("quant_bits"),
            "quant_scope": m.get("quant_scope"), "quant_detail": m.get("quant_detail"),
            "modalities": modalities(m), "ctl_id": ctl_id(m) if m.get("engine") in ENGINES else None,
            "outputs": list(m["outputs"]) if m.get("outputs") else None,
            "home_host": m.get("box"), "unit": m.get("unit"),
            # `<engine>:<halo-ctl id>` copies on a box's disk this card stands for (ADR 0019).
            "covers": list(m.get("covers") or []),
            "updated": (dates.get(mid) or {}).get("updated"),
            "updated_source": (dates.get(mid) or {}).get("updated_source"),
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
    always = "".join(f"\n  always on: {s['label']} ({s['unit']}) {s['phase']} · {s['model'] or '—'} · {s['url']}"
                     for s in b.get("services") or [])
    return f"{b['host']}: {b['engine']} {b['phase']}{role}{serving}{always}"


def main(cfg: dict, args, performance: dict, dates: dict | None = None, hermes_patch=None) -> int:
    """`spark-serve strix …` entry point; JSON on stdout, errors as an error event."""
    try:
        if args.strix_cmd == "list":
            items = catalog(cfg, performance, dates)
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
        elif args.strix_cmd == "use":
            result = use(cfg, args.model, args.box, args.hermes_target, hermes_patch)
            if args.json:
                _emit_json({"event": "hermes", "detail": result["message"]})
                _emit_json({"event": "selected", "host": result["host"], "served": result["served"],
                            "url": result["base_url"], "label": result["label"],
                            "hermes_target": result["hermes_target"], "hermes_host": result["hermes_host"]})
            else:
                target = "Hermes (Spark)" if result["hermes_target"] == "spark" else "Hermes (Mac)"
                where = f" on {result['hermes_host']}" if result["hermes_host"] else ""
                print(f"hermes: {result['message']}")
                print(f"{target}{where} uses {result['host']}: {result['label']}")
                print(f"  {result['base_url']}  model {result['served']}")
        elif args.strix_cmd == "connection":
            details = connection(cfg, args.box)
            if args.json:
                _emit_json(details)
            else:
                print(f"Engine:   {details['engine']}\nBase URL: {details['base_url']}\n"
                      f"Model:    {details['model']}\nAPI key:  {details['api_key'] or '(no API key set)'}")
                for extra in (details.get("services") or [])[1:]:
                    print(f"Also:     {extra['model']} at {extra['base_url']}")
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
    upc.add_argument("--box", help="the box to serve on; for the cluster, its master ([strix] master pins it)")
    upc.add_argument("--json", action="store_true", help="JSON events are the default output")
    stp = commands.add_parser("stop", help="stop a box (its cluster, for the worker), or every box")
    stp.add_argument("--box", help="default: every box")
    stp.add_argument("--json", action="store_true", help="JSON events are the default output")
    use_cmd = commands.add_parser("use", help="point Hermes at a Strix model that is already serving")
    use_cmd.add_argument("model", help="catalog id or alias, or <engine>:<name> for a model on the box's disk")
    use_cmd.add_argument("--box", help="the box serving it, when more than one is ready; a service uses its own box")
    use_cmd.add_argument("--hermes-target", choices=["local", "spark"], default="local",
                         help="Hermes config to update: this Mac (default) or the configured head Spark")
    use_cmd.add_argument("--json", action="store_true", help="emit JSON-line progress")
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
