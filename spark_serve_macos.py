"""Local Mac Studio models served by Splash, beside the Spark and Strix sites.

The machine is this Mac (Mac15,14, M3 Ultra, 512 GB unified memory). Splash
(`brew install incoai/tap/splash`) serves one model per process,
OpenAI-compatible, with `splash serve`. Its own default listener is
127.0.0.1:8000. Port 8000 on this Mac is a live oMLX server, so every card
binds another port (8101 and up) and this module never binds, probes, or
signals anything on 8000 or any other reserved port.

Catalog: [macos] plus [macos.models.<id>]. Each card names the Splash model
id, its DFlash2 draft, and the local directories (`model_dir`, `draft_dir`).
Omitted paths default to `<weights_root>/<hf-repo-name>/`, which is
`/Volumes/DATA/local-models/splash/<owner>/<repo>` unless the card or the
section sets another root. Whether those paths exist is the on-disk status;
nothing is marked permanently downloaded or missing.

Start runs `splash serve` for that card only, with flags documented by Splash
(README and docs/server.md): `--model`, `--model-dir`, `--draft-model`,
`--host`, `--port`, and `--mmproj` when the card sets one. Stop signals only
the process this module spawned for that card, and only after its command
line still matches that card and names no reserved port.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from spark_serve_controller import atomic_json, state_dir

# Splash's documented default. oMLX already owns it on this Mac.
OMLX_PORT = 8000
DEFAULT_HOST = "127.0.0.1"
DEFAULT_WEIGHTS_ROOT = "/Volumes/DATA/local-models/splash"
DEFAULT_MACHINE = {
    "name": "Mac Studio",
    "product": "Mac15,14",
    "chip": "M3 Ultra",
    "memory_gb": 512,
}
# Flags `splash serve` documents (docs/server.md). Nothing else is passed.
SERVE_FLAGS = ("--model", "--model-dir", "--draft-model", "--host", "--port", "--mmproj")
READY_SECONDS = 1800
POLL_SECONDS = 2
STOP_SECONDS = 5
PROBE_SECONDS = 2


class MacOSError(RuntimeError):
    pass


def site(cfg: dict) -> dict:
    macos = cfg.get("macos")
    if not isinstance(macos, dict) or not isinstance(macos.get("models"), dict) or not macos["models"]:
        raise MacOSError("no [macos] models in the catalog")
    return macos


def reserved_ports(cfg: dict) -> set[int]:
    """8000 is always reserved, even when the catalog omits or replaces the list."""
    ports = {OMLX_PORT}
    listed = (cfg.get("macos") or {}).get("reserved_ports") or []
    if isinstance(listed, list):
        for port in listed:
            ports.add(int(port))
    return ports


def models(cfg: dict) -> dict:
    return site(cfg)["models"]


def resolve(cfg: dict, name: str) -> tuple[str, dict]:
    key = name.strip().lower()
    for mid, model in models(cfg).items():
        if key == mid.lower() or key in [str(a).lower() for a in model.get("aliases") or []]:
            return mid, model
    raise MacOSError(f"unknown MacOS model {name!r}; catalog: {', '.join(sorted(models(cfg)))}")


def repo_name(model_id: str) -> str:
    """`OWNER/REPO` from `OWNER/REPO` or a GGUF `OWNER/REPO:VARIANT`."""
    if not isinstance(model_id, str) or model_id.count("/") != 1 or not all(model_id.split(":", 1)[0].split("/")):
        raise MacOSError(f"model id must be OWNER/REPO or OWNER/REPO:VARIANT, not {model_id!r}")
    return model_id.split(":", 1)[0]


def _root(cfg: dict, model: dict) -> str:
    raw = model.get("weights_root") or site(cfg).get("weights_root") or DEFAULT_WEIGHTS_ROOT
    return os.path.expanduser(str(raw))


def model_dir(cfg: dict, model: dict) -> str:
    """Local target. An explicit model_dir wins; otherwise `<weights_root>/<hf repo>/`."""
    if model.get("model_dir"):
        return os.path.expanduser(str(model["model_dir"]))
    return str(Path(_root(cfg, model)) / repo_name(model["model"]))


def draft_dir(cfg: dict, model: dict) -> str:
    """Local DFlash2 draft. An explicit draft_dir wins; otherwise `<weights_root>/<draft repo>/`."""
    draft = model.get("draft_model")
    if not isinstance(draft, str) or not draft:
        raise MacOSError("a card needs draft_model (the family's DFlash2 repository)")
    if model.get("draft_dir"):
        return os.path.expanduser(str(model["draft_dir"]))
    return str(Path(_root(cfg, model)) / repo_name(draft))


def card_port(model: dict, mid: str) -> int:
    port = model.get("port")
    if isinstance(port, bool) or not isinstance(port, int):
        raise MacOSError(f"[macos.models.{mid}] needs an integer port other than {OMLX_PORT}")
    return port


def card_host(cfg: dict, model: dict) -> str:
    host = model.get("host") or site(cfg).get("host") or DEFAULT_HOST
    if not isinstance(host, str) or not host:
        raise MacOSError("[macos] host must be an address")
    return host


def _check_ports(cfg: dict) -> None:
    reserved = reserved_ports(cfg)
    seen: dict[int, str] = {}
    for mid, model in models(cfg).items():
        port = card_port(model, mid)
        if port in reserved:
            raise MacOSError(
                f"[macos.models.{mid}] port {port} is reserved"
                f" ({OMLX_PORT} is the live oMLX server); splash is never bound there")
        if port in seen:
            raise MacOSError(f"[macos.models.{mid}] port {port} is also used by {seen[port]}")
        seen[port] = mid
        repo_name(model.get("model") or "")
        if not model.get("draft_model"):
            raise MacOSError(f"[macos.models.{mid}] needs draft_model")
        fmt = model.get("format")
        if fmt not in ("mlx", "gguf"):
            raise MacOSError(f"[macos.models.{mid}] format must be mlx or gguf")


def serve_argv(cfg: dict, mid: str, binary: str | None = None) -> list[str]:
    """The exact `splash serve` command for one card. Only documented flags."""
    _check_ports(cfg)
    model = models(cfg)[mid]
    binary = binary or str(site(cfg).get("splash") or "splash")
    argv = [
        binary, "serve",
        "--model", model["model"],
        "--model-dir", model_dir(cfg, model),
        "--draft-model", draft_dir(cfg, model),
        "--host", card_host(cfg, model),
        "--port", str(card_port(model, mid)),
    ]
    mmproj = model.get("mmproj")
    if mmproj:
        argv.extend(["--mmproj", os.path.expanduser(str(mmproj))])
    flags = [arg for arg in argv if arg.startswith("--")]
    if any(flag not in SERVE_FLAGS for flag in flags):
        raise MacOSError(f"refusing undocumented splash flags: {flags}")
    if OMLX_PORT in _flag_ports(argv):
        raise MacOSError(f"refusing to bind port {OMLX_PORT}")
    return argv


def _flag_ports(argv: list[str]) -> set[int]:
    ports = set()
    for index, arg in enumerate(argv):
        if arg == "--port" and index + 1 < len(argv):
            ports.add(int(argv[index + 1]))
        elif arg.startswith("--port="):
            ports.add(int(arg.split("=", 1)[1]))
    return ports


def _flag(argv: list[str], name: str) -> str | None:
    for index, arg in enumerate(argv):
        if arg == name and index + 1 < len(argv):
            return argv[index + 1]
    return None


def splash_binary(cfg: dict) -> str:
    raw = str(site(cfg).get("splash") or "splash")
    path = os.path.expanduser(raw)
    if os.path.sep in path:
        if not (os.path.isfile(path) and os.access(path, os.X_OK)):
            raise MacOSError(f"splash is not executable: {path}")
        return path
    found = shutil.which(path)
    if not found:
        raise MacOSError("splash is not on PATH; install it with: brew install incoai/tap/splash")
    return found


def _state_path() -> Path:
    return state_dir() / "macos-servers.json"


def _logs_dir() -> Path:
    return state_dir() / "macos-logs"


def _load_state() -> dict:
    try:
        data = json.loads(_state_path().read_text())
    except (OSError, ValueError):
        return {"models": {}}
    if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
        return {"models": {}}
    return data


def _save_state(data: dict) -> None:
    atomic_json(_state_path(), data)


def _record(mid: str) -> dict | None:
    row = _load_state()["models"].get(mid)
    return row if isinstance(row, dict) else None


def _remember(mid: str, row: dict | None) -> None:
    data = _load_state()
    if row is None:
        data["models"].pop(mid, None)
    else:
        data["models"][mid] = row
    _save_state(data)


def _alive(pid: int) -> bool:
    """False once the process has exited, including a zombie the parent has not reaped."""
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        try:
            text = stat.read_text()
        except OSError:
            return False
        end = text.rfind(")")
        fields = text[end + 2:].split() if end >= 0 else []
        if not fields:
            return False
        return fields[0] not in {"Z", "X"}
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _reap(pid: int) -> None:
    """Collect a child we started in this process. A later `stop` is not its parent."""
    try:
        os.waitpid(pid, os.WNOHANG)
    except (ChildProcessError, OSError):
        pass


def process_argv(pid: int) -> list[str] | None:
    """The process's own argument vector, from /proc on Linux or ps on macOS."""
    path = Path(f"/proc/{pid}/cmdline")
    if path.exists():
        try:
            data = path.read_bytes()
        except OSError:
            return None
        if not data:
            return None
        parts = data.split(b"\0")
        if parts and parts[-1] == b"":
            parts.pop()
        return [part.decode(errors="replace") for part in parts]
    try:
        done = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                              capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0 or not done.stdout.strip():
        return None
    try:
        return shlex.split(done.stdout.strip())
    except ValueError:
        return None


def process_started(pid: int) -> str | None:
    """A start token so a recycled pid is not treated as the server we launched."""
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        try:
            text = stat.read_text()
        except OSError:
            return None
        end = text.rfind(")")
        fields = text[end + 2:].split() if end >= 0 else []
        # field 22 (starttime) is index 19 of the post-comm fields
        return fields[19] if len(fields) > 19 else None
    try:
        done = subprocess.run(["ps", "-p", str(pid), "-o", "lstart="],
                              capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip() if done.returncode == 0 and done.stdout.strip() else None


def _same_server(live: list[str], expected: list[str]) -> bool:
    for flag in ("--model", "--model-dir", "--draft-model", "--host", "--port"):
        if _flag(live, flag) != _flag(expected, flag):
            return False
    return _flag(expected, "--port") is not None


def ownership(record: dict, reserved: set[int]) -> tuple[bool, str]:
    """True only when this pid is still the splash serve we spawned for the card.

    A reserved port anywhere in its command line is never owned, so oMLX on
    :8000 cannot be stopped through a stale or edited state file.
    """
    try:
        pid = int(record["pid"])
        port = int(record["port"])
        argv = list(record["argv"])
    except (KeyError, TypeError, ValueError):
        return False, "unreadable"
    if port in reserved or OMLX_PORT in _flag_ports(argv):
        return False, "reserved port"
    if not _alive(pid):
        return False, "dead"
    live = process_argv(pid)
    if not live:
        return False, "unreadable"
    live_ports = _flag_ports(live)
    if not live_ports or live_ports & reserved or OMLX_PORT in live_ports:
        return False, "reserved port"
    if port not in live_ports:
        return False, "different port"
    if not _same_server(live, argv):
        return False, "different command"
    started = record.get("started")
    current = process_started(pid)
    if started and current and started != current:
        return False, "different process"
    try:
        if os.getpgid(pid) != pid:
            return False, "not a group leader"
    except OSError:
        return False, "unreadable"
    return True, "ok"


def _port_busy(host: str, port: int) -> bool:
    if port == OMLX_PORT or port in {0}:
        raise MacOSError(f"refusing to touch port {port}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # SO_REUSEADDR ignores TIME_WAIT from a server that just exited. A
        # process still listening on the port still makes bind fail.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return True
    return False


def _on_disk(path: str) -> bool:
    return Path(path).exists()


def _machine(cfg: dict) -> dict:
    macos = cfg.get("macos") or {}
    out = dict(DEFAULT_MACHINE)
    for key in ("name", "product", "chip"):
        if isinstance(macos.get(key), str) and macos[key]:
            out[key] = macos[key]
    if isinstance(macos.get("memory_gb"), int):
        out["memory_gb"] = macos["memory_gb"]
    return out


def server_view(cfg: dict, mid: str, model: dict, reserved: set[int]) -> dict:
    target = model_dir(cfg, model)
    draft = draft_dir(cfg, model)
    port = card_port(model, mid)
    host = card_host(cfg, model)
    weights = _on_disk(target)
    draft_ok = _on_disk(draft)
    record = _record(mid)
    owned, why = ownership(record, reserved) if record else (False, "absent")
    phase = "stopped"
    error = None
    if why == "dead":
        _remember(mid, None)
        record, owned = None, False
    elif why == "reserved port":
        # Drop the pointer. Do not signal the process it names (oMLX owns :8000).
        _remember(mid, None)
        record, owned = None, False
        error = f"A recorded pid named port {OMLX_PORT}; it was not signalled."
    elif why in {"different port", "different command", "different process"}:
        _remember(mid, None)
        record, owned = None, False
        error = f"Recorded server no longer matches this card ({why}); it was not stopped."
    elif record and not owned and why != "absent":
        error = f"Cannot confirm the recorded server ({why}); it was not stopped."
    if owned:
        phase = "ready" if _ready(host, port, model["model"]) else "starting"
    url = f"http://{host}:{port}/v1"
    busy = phase == "stopped" and port not in reserved and _port_busy(host, port)
    return {
        "id": mid,
        "label": model.get("label") or mid,
        "phase": phase,
        "port": port,
        "host": host,
        "url": url if phase != "stopped" else None,
        "model": model["model"],
        "format": model.get("format"),
        "draft_model": model.get("draft_model"),
        "model_dir": target,
        "draft_dir": draft,
        "model_on_disk": weights,
        "draft_on_disk": draft_ok,
        "on_disk": weights and draft_ok,
        "pid": int(record["pid"]) if owned and record else None,
        "port_busy": busy,
        "error": error,
    }


def status(cfg: dict) -> dict:
    _check_ports(cfg)
    reserved = reserved_ports(cfg)
    if OMLX_PORT not in reserved:
        raise MacOSError("port 8000 must stay reserved")
    return {
        "machine": _machine(cfg),
        "host": site(cfg).get("host") or DEFAULT_HOST,
        "reserved_ports": sorted(reserved),
        "splash": site(cfg).get("splash") or "splash",
        "models": [server_view(cfg, mid, model, reserved) for mid, model in models(cfg).items()],
    }


def _ready(host: str, port: int, served: str) -> bool:
    """GET the card's own /v1/models. Never requests port 8000."""
    if port == OMLX_PORT or not served:
        return False
    url = f"http://{host}:{port}/v1/models"
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=PROBE_SECONDS) as response:
            body = response.read(1_000_000).decode(errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False
    return served in body


def _wait_ready(host: str, port: int, served: str, timeout: float, emit) -> bool:
    deadline = time.monotonic() + timeout
    next_note = 0.0
    while time.monotonic() < deadline:
        if _ready(host, port, served):
            return True
        now = time.monotonic()
        if emit and now >= next_note:
            emit({"event": "progress", "detail": f"waiting for splash on {host}:{port}"})
            next_note = now + 10
        time.sleep(POLL_SECONDS)
    return False


def _require_weights(cfg: dict, mid: str, model: dict) -> None:
    target, draft = model_dir(cfg, model), draft_dir(cfg, model)
    missing = [path for path in (target, draft) if not _on_disk(path)]
    if missing:
        raise MacOSError(
            f"{model.get('label') or mid} is not on disk yet: {', '.join(missing)}")


def up(cfg: dict, name: str, emit=None, *, ready_timeout: float = READY_SECONDS) -> dict:
    """Spawn splash serve for this card. Does not stop any other process."""
    emit = emit or (lambda event: None)
    _check_ports(cfg)
    mid, model = resolve(cfg, name)
    port = card_port(model, mid)
    host = card_host(cfg, model)
    if port in reserved_ports(cfg):
        raise MacOSError(f"refusing to start {mid} on reserved port {port}")
    _require_weights(cfg, mid, model)
    reserved = reserved_ports(cfg)
    record = _record(mid)
    if record and ownership(record, reserved)[0]:
        if _ready(host, port, model["model"]):
            emit({"event": "ready", "model": mid, "port": port, "detail": "already serving"})
            return server_view(cfg, mid, model, reserved)
        raise MacOSError(f"{model.get('label') or mid} is already starting on port {port}")
    if _port_busy(host, port):
        raise MacOSError(f"port {port} is already in use; left that process alone")
    binary = splash_binary(cfg)
    argv = serve_argv(cfg, mid, binary)
    if int(_flag(argv, "--port") or 0) != port or port == OMLX_PORT:
        raise MacOSError("refusing to start splash")
    log_path = _logs_dir() / f"{mid}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    emit({"event": "starting", "model": mid, "port": port, "argv": argv})
    log = log_path.open("ab", buffering=0)
    try:
        proc = subprocess.Popen(
            argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
        )
    except OSError as exc:
        log.close()
        raise MacOSError(f"could not start splash: {exc}") from exc
    log.close()
    _remember(mid, {
        "pid": proc.pid, "port": port, "argv": argv, "log": str(log_path),
        "started": process_started(proc.pid),
    })
    try:
        if not _wait_ready(host, port, model["model"], ready_timeout, emit):
            stop(cfg, mid, emit=emit)
            raise MacOSError(f"splash did not become ready on port {port}; stopped that process")
    except MacOSError:
        raise
    except Exception:
        stop(cfg, mid, emit=emit)
        raise
    view = server_view(cfg, mid, model, reserved)
    emit({"event": "ready", "model": mid, "port": port, "url": view["url"]})
    return view


def _signal_group(pid: int, sig: int) -> None:
    """Signal the process group only when pid is that group's leader."""
    if os.getpgid(pid) != pid:
        raise MacOSError(f"refusing to signal pid {pid}: it is not its own process group")
    os.killpg(pid, sig)


def stop(cfg: dict, name: str, emit=None) -> dict:
    """Stop the splash process this card started. No other pid is signalled."""
    emit = emit or (lambda event: None)
    _check_ports(cfg)
    mid, model = resolve(cfg, name)
    port = card_port(model, mid)
    if port in reserved_ports(cfg) or port == OMLX_PORT:
        raise MacOSError(f"refusing to stop anything on reserved port {port}")
    record = _record(mid)
    if not record:
        emit({"event": "stopped", "model": mid, "detail": "not running"})
        return {"id": mid, "stopped": False, "detail": "not running"}
    reserved = reserved_ports(cfg)
    owned, why = ownership(record, reserved)
    pid = int(record.get("pid") or 0)
    if not owned:
        if why == "dead":
            _remember(mid, None)
        elif why == "reserved port":
            _remember(mid, None)
            raise MacOSError(
                f"recorded pid {pid} names reserved port {OMLX_PORT}; it was not signalled")
        else:
            raise MacOSError(f"recorded pid {pid} is not this card's splash ({why}); it was not signalled")
        emit({"event": "stopped", "model": mid, "detail": "not running"})
        return {"id": mid, "stopped": False, "detail": "not running"}
    live = process_argv(pid) or []
    if OMLX_PORT in _flag_ports(live) or not _same_server(live, list(record["argv"])):
        raise MacOSError(f"refusing to signal pid {pid}: it is not this card's splash on port {port}")
    try:
        _signal_group(pid, signal.SIGTERM)
    except ProcessLookupError:
        _remember(mid, None)
        emit({"event": "stopped", "model": mid, "detail": "not running"})
        return {"id": mid, "stopped": False, "detail": "not running"}
    deadline = time.monotonic() + STOP_SECONDS
    while time.monotonic() < deadline and _alive(pid):
        _reap(pid)
        time.sleep(0.05)
    _reap(pid)
    if _alive(pid):
        if not ownership(record, reserved)[0]:
            raise MacOSError(f"pid {pid} changed while stopping; it was not killed")
        _signal_group(pid, signal.SIGKILL)
        deadline = time.monotonic() + STOP_SECONDS
        while time.monotonic() < deadline and _alive(pid):
            _reap(pid)
            time.sleep(0.05)
        _reap(pid)
    if _alive(pid):
        raise MacOSError(f"pid {pid} did not exit")
    _reap(pid)
    _remember(mid, None)
    emit({"event": "stopped", "model": mid, "pid": pid, "port": port})
    return {"id": mid, "stopped": True, "pid": pid, "port": port}


def logs(cfg: dict, name: str, lines: int = 80) -> dict:
    mid, _model = resolve(cfg, name)
    record = _record(mid)
    path = Path(record["log"]) if record and record.get("log") else _logs_dir() / f"{mid}.log"
    try:
        text = path.read_text(errors="replace")
    except OSError:
        text = ""
    count = max(1, min(int(lines), 500))
    return {"id": mid, "path": str(path), "lines": text.splitlines()[-count:]}


def use(cfg: dict, name: str, hermes_target: str, patch) -> dict:
    """Point Hermes at a Splash model that is already ready. Does not start or stop it."""
    if hermes_target not in ("local", "spark"):
        raise MacOSError("Hermes target must be local or spark")
    if patch is None:
        raise MacOSError("Hermes update is unavailable")
    mid, model = resolve(cfg, name)
    if card_port(model, mid) in reserved_ports(cfg):
        raise MacOSError(f"refusing to select a model on reserved port {card_port(model, mid)}")
    view = server_view(cfg, mid, model, reserved_ports(cfg))
    if view["phase"] != "ready" or not view.get("url"):
        raise MacOSError(f"{model.get('label') or mid} is not ready")
    hermes_host = None
    if hermes_target == "spark":
        head = (cfg.get("cluster") or {}).get("head")
        if not isinstance(head, str) or not head:
            raise MacOSError("Hermes (Spark) needs cluster.head")
        hermes_host = head
    chosen = dict(model)
    chosen["served_name"] = model["model"]
    if not chosen.get("hermes_provider"):
        chosen["hermes_provider"] = "splash"
    message, ok = patch(
        cfg, mid, chosen, False,
        remote_host=hermes_host, base_url=view["url"], api_key=None,
        provider_label="Splash on this Mac", move_ds4=False,
    )
    if not ok:
        raise MacOSError(message)
    return {"message": message, "host": view["host"], "served": model["model"],
            "base_url": view["url"], "provider": chosen["hermes_provider"],
            "hermes_target": hermes_target, "hermes_host": hermes_host,
            "label": model.get("label") or mid, "port": view["port"]}


def catalog(cfg: dict, performance: dict, dates: dict | None = None) -> list[dict]:
    """Card entries in the same shape as `spark-serve list --json`, ranked within this site."""
    dates = dates or {}
    from spark_serve_ranking import modalities, score_models
    _check_ports(cfg)
    out = []
    for mid, model in models(cfg).items():
        out.append({
            "id": mid,
            "label": model.get("label") or mid,
            "aliases": list(model.get("aliases") or []),
            "served_name": model.get("model") or "",
            "ctx": int(model.get("max_model_len") or 0),
            "image": model.get("model") or "",
            "notes": model.get("notes") or "",
            "backend": "splash",
            "topology": "macos",
            "wrapper": "splash",
            "hermes_provider": model.get("hermes_provider") or "splash",
            "performance": performance.get(mid),
            "quant": model.get("quant"),
            "quant_bits": model.get("quant_bits"),
            "quant_scope": model.get("quant_scope"),
            "quant_detail": model.get("quant_detail"),
            "modalities": modalities(model),
            "updated": (dates.get(mid) or {}).get("updated"),
            "updated_source": (dates.get(mid) or {}).get("updated_source"),
        })
    ranking = score_models(out)
    for item in out:
        item["ranking"] = ranking.get(item["id"])
    return out


def _emit_json(value) -> None:
    print(json.dumps(value), flush=True)


def main(cfg: dict, args, performance: dict, dates: dict | None = None, hermes_patch=None) -> int:
    """`spark-serve macos …` entry point; JSON on stdout, errors as an error event."""
    try:
        if args.macos_cmd == "list":
            items = catalog(cfg, performance, dates)
            if args.json:
                print(json.dumps(items, indent=2))
            else:
                for item in items:
                    star = " ★" if (item.get("ranking") or {}).get("best") else ""
                    print(f"{item['id']:<28} {item['served_name']}{star}")
        elif args.macos_cmd == "status":
            current = status(cfg)
            if args.json:
                _emit_json(current)
            else:
                machine = current["machine"]
                print(f"{machine['name']} {machine['product']} {machine['chip']} {machine['memory_gb']} GB"
                      f"  reserved ports: {', '.join(str(p) for p in current['reserved_ports'])}")
                for row in current["models"]:
                    disk = "on disk" if row["on_disk"] else "not on disk yet"
                    print(f"{row['id']:<28} :{row['port']:<5} {row['phase']:<9} {disk}  {row['model_dir']}")
        elif args.macos_cmd == "up":
            up(cfg, args.model, emit=_emit_json)
        elif args.macos_cmd == "stop":
            stop(cfg, args.model, emit=_emit_json)
        elif args.macos_cmd == "logs":
            out = logs(cfg, args.model, args.lines)
            if args.json:
                _emit_json(out)
            else:
                print("\n".join(out["lines"]))
        elif args.macos_cmd == "use":
            result = use(cfg, args.model, args.hermes_target, hermes_patch)
            if args.json:
                _emit_json({"event": "hermes", "detail": result["message"]})
                _emit_json({"event": "selected", "host": result["host"], "served": result["served"],
                            "url": result["base_url"], "label": result["label"], "port": result["port"],
                            "hermes_target": result["hermes_target"], "hermes_host": result["hermes_host"]})
            else:
                target = "Hermes (Spark)" if result["hermes_target"] == "spark" else "Hermes (Mac)"
                where = f" on {result['hermes_host']}" if result["hermes_host"] else ""
                print(f"hermes: {result['message']}")
                print(f"{target}{where} uses {result['label']}")
                print(f"  {result['base_url']}  model {result['served']}")
        return 0
    except (MacOSError, ValueError, KeyError) as exc:
        _emit_json({"event": "error", "detail": str(exc)})
        return 1


def add_parser(sub) -> None:
    macos = sub.add_parser(
        "macos",
        help="this Mac's Splash servers (one process per card; never port 8000, which oMLX owns)",
    )
    commands = macos.add_subparsers(dest="macos_cmd", required=True)
    listed = commands.add_parser("list", help="MacOS catalog with measured speed and ranking")
    listed.add_argument("--json", action="store_true")
    state = commands.add_parser("status", help="each card's port, phase, and whether its paths exist")
    state.add_argument("--json", action="store_true")
    start = commands.add_parser("up", help="splash serve for one card, on that card's port only")
    start.add_argument("model", help="catalog id or alias")
    start.add_argument("--json", action="store_true", help="JSON events are the default output")
    halt = commands.add_parser("stop", help="stop the splash process this card started, and no other")
    halt.add_argument("model", help="catalog id or alias")
    halt.add_argument("--json", action="store_true", help="JSON events are the default output")
    log = commands.add_parser("logs", help="the log of the splash process this card started")
    log.add_argument("model")
    log.add_argument("-n", "--lines", type=int, default=80)
    log.add_argument("--json", action="store_true")
    use_cmd = commands.add_parser("use", help="point Hermes at a Splash model that is already ready")
    use_cmd.add_argument("model")
    use_cmd.add_argument("--hermes-target", choices=["local", "spark"], default="local",
                         help="Hermes config to update: this Mac (default) or the configured head Spark")
    use_cmd.add_argument("--json", action="store_true", help="emit JSON-line progress")
