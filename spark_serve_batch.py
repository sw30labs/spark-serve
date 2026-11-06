"""Batch catalog entries (wrapper = "batch"), e.g. Pika H3 Flex. See ADR 0018.

`up <batch entry>` validates everything it can before anything stops, then:
stops the current model through the normal fenced Controller path, launches a
detached supervisor on the entry's Spark, and returns. The supervisor runs the
job and then restores the previous model with a spark-serve snapshot shipped in
the job directory, so the restore does not need the Mac. `batch status` reads
the job back and adopts the Spark-side controller state once it restored.
"""
from __future__ import annotations

import base64
import io
import json
import re
import shlex
import subprocess
import tarfile
import time
from pathlib import Path

from spark_serve_controller import ControllerError, atomic_json, read_state, state_dir

JOB_ROOT = "~/.local/state/spark-serve-batch"
RECORD = "batch.json"
SNAPSHOT_FILES = ("spark-serve", "models.toml", "models.example.toml", "measured-performance.json")
SNAPSHOT_GLOBS = ("spark_serve_*.py",)
SNAPSHOT_DIRS = ("recipes", "spark_bench")
FINAL_RESTORE = ("ok", "failed", "none")
SAFE_ID = re.compile(r"^[0-9]{8}-[0-9]{6}$")


def is_batch(model: dict) -> bool:
    return (model.get("wrapper") or "vllm") == "batch"


def batch_table(mid: str, model: dict) -> dict:
    table = model.get("batch")
    if not isinstance(table, dict) or not isinstance(table.get("command"), list) or not table["command"]:
        raise ControllerError(f"{mid}: batch entries need [models.{mid}.batch] with a command list")
    return table


def form(mid: str, model: dict) -> dict | None:
    """Options the app shows for a launchable batch entry (list --json)."""
    if not is_batch(model) or model.get("launch") is False:
        return None
    try:
        t = batch_table(mid, model)
    except ControllerError:
        return None
    return {
        "resolutions": list(t.get("resolutions") or []),
        "aspects": list(t.get("aspects") or []),
        "duration_min": int(t.get("duration_min") or 5), "duration_max": int(t.get("duration_max") or 5),
        "default_resolution": t.get("default_resolution") or (t.get("resolutions") or [None])[0],
        "default_aspect": t.get("default_aspect") or (t.get("aspects") or [None])[0],
        "default_duration": int(t.get("default_duration") or t.get("duration_min") or 5),
        "default_seed": int(t.get("default_seed") or 42),
        "expected_minutes": t.get("expected_minutes"),
    }


def job_id(now: float | None = None) -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime(now if now is not None else time.time()))


def build_spec(cfg: dict, mid: str, model: dict, *, prompt: str | None, resolution=None, duration=None,
               aspect=None, seed=None, now: float | None = None) -> dict:
    """Validate the request and build job.json. Local only: no SSH, nothing stops."""
    t = batch_table(mid, model)
    node = model.get("home_node")
    if node not in ("head", "worker"):
        raise ControllerError(f"{mid}: home_node must be head or worker")
    host = str(cfg["cluster"][node])
    prompt = (prompt or "").strip()
    if not prompt:
        raise ControllerError(f"{mid}: a prompt is required (--prompt or --prompt-file)")
    if len(prompt) > 8000:
        raise ControllerError(f"{mid}: prompt is longer than 8000 characters")
    resolution = resolution or t.get("default_resolution") or (t.get("resolutions") or [None])[0]
    if t.get("resolutions") and resolution not in t["resolutions"]:
        raise ControllerError(f"{mid}: resolution must be one of {', '.join(t['resolutions'])}")
    aspect = aspect or t.get("default_aspect") or (t.get("aspects") or [None])[0]
    if t.get("aspects") and aspect not in t["aspects"]:
        raise ControllerError(f"{mid}: aspect must be one of {', '.join(t['aspects'])}")
    try:
        duration = int(duration if duration is not None else t.get("default_duration") or t.get("duration_min") or 5)
        seed = int(seed if seed is not None else t.get("default_seed") or 42)
    except (TypeError, ValueError):
        raise ControllerError(f"{mid}: duration and seed must be whole numbers") from None
    low, high = int(t.get("duration_min") or duration), int(t.get("duration_max") or duration)
    if not low <= duration <= high:
        raise ControllerError(f"{mid}: duration must be {low}–{high} seconds")
    if not 0 <= seed < 2**32:
        raise ControllerError(f"{mid}: seed must be 0..4294967295")
    jid = job_id(now)
    job_dir = f"{JOB_ROOT}/{jid}"
    outputs = str(t.get("outputs") or "~").rstrip("/")
    output = f"{outputs}/{jid}.mp4"
    values = {"prompt_file": f"{job_dir}/prompt.txt", "output": output, "resolution": resolution,
              "aspect": aspect, "duration": str(duration), "seed": str(seed)}
    try:
        argv = [str(item).format(**values) for item in t["command"]]
    except (KeyError, IndexError, ValueError) as exc:
        raise ControllerError(f"{mid}: bad placeholder in batch command: {exc}") from None
    return {
        "id": jid, "model": mid, "label": model.get("label") or mid, "host": host, "node": node,
        "job_dir": job_dir, "argv": argv, "workdir": t.get("workdir") or "~",
        "env": dict(t.get("env") or {}), "unset_env": list(t.get("unset_env") or []),
        "path_prepend": list(t.get("path_prepend") or []), "require_paths": list(t.get("require_paths") or []),
        "require_free_gpu": True, "timeout_minutes": int(t.get("timeout_minutes") or 120),
        "expected_minutes": t.get("expected_minutes"), "output": output, "prompt": prompt,
        "options": {"resolution": resolution, "aspect": aspect, "duration": duration, "seed": seed},
        "restore": None,
    }


def previous_allocation(cfg: dict, state: dict) -> tuple[str | None, str | None]:
    """(model id, --node) serving now per controller state; (None, None) when idle.

    Refuses states the batch path cannot restore exactly (YuE, mixed, mid-transition).
    """
    cluster = cfg["cluster"]
    nodes = state.get("nodes") if isinstance(state.get("nodes"), dict) else None
    mode, phase = state.get("mode", "none"), state.get("phase")
    if nodes:
        active = {h: r for h, r in nodes.items() if isinstance(r, dict) and r.get("mode") not in (None, "none")}
        if not active:
            return None, None
        modes = {r.get("mode") for r in active.values()}
        models = {r.get("model") for r in active.values()}
        if modes != {"vllm"} or len(models) != 1:
            raise ControllerError("the Sparks are not running one spark-serve model (YuE, mixed or unknown); stop them first")
        if any(r.get("phase") != "ready" for r in active.values()):
            raise ControllerError("the current model is not ready (a transition or failure is pending); settle it first")
        mid = next(iter(models))
        hosts = sorted({h for r in active.values() for h in (r.get("allocation_hosts") or [])} | set(active))
    else:
        if mode in ("none",) and phase in ("stopped", None):
            return None, None
        if mode != "vllm" or phase != "ready" or not state.get("model"):
            raise ControllerError("the current model is not ready or not a spark-serve model; settle it first")
        mid, hosts = state["model"], [cluster["head"], cluster["worker"]]
    if mid not in (cfg.get("models") or {}):
        raise ControllerError(f"current model {mid!r} is not in the catalog; cannot promise to restore it")
    if set(hosts) == {cluster["head"], cluster["worker"]}:
        return mid, "both"
    if hosts == [cluster["head"]]:
        return mid, "head"
    if hosts == [cluster["worker"]]:
        return mid, "worker"
    raise ControllerError(f"cannot map the current allocation {hosts} to a node")


def snapshot(here: Path) -> bytes:
    """tar.gz of the code and catalog the Spark-side restore runs."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        names = [n for n in SNAPSHOT_FILES if (here / n).exists()]
        for pattern in SNAPSHOT_GLOBS:
            names += sorted(p.name for p in here.glob(pattern))
        for name in names:
            tar.add(here / name, arcname=f"app/{name}")
        for d in SNAPSHOT_DIRS:
            if (here / d).is_dir():
                tar.add(here / d, arcname=f"app/{d}",
                        filter=lambda info: None if "__pycache__" in info.name else info)
    return buf.getvalue()


def ssh_material(cfg: dict, home: str, run=subprocess.run) -> tuple[str, list[str]]:
    """Job-scoped ssh_config + known_hosts lines so the home Spark can reach its peers.

    Uses the Mac's own resolved HostName/User and the host keys the Mac already
    trusts; nothing in ~/.ssh is changed on either side.
    """
    lines, known = [], []
    for host in (cfg["cluster"]["head"], cfg["cluster"]["worker"]):
        if host == home:
            continue
        resolved = {}
        proc = run(["ssh", "-G", host], capture_output=True, text=True, timeout=15)
        for line in (proc.stdout or "").splitlines():
            key, _, value = line.partition(" ")
            if key in ("hostname", "user", "port") and value:
                resolved[key] = value.strip()
        hostname = resolved.get("hostname") or host
        keys = run(["ssh-keygen", "-F", hostname], capture_output=True, text=True, timeout=15)
        found = [l for l in (keys.stdout or "").splitlines() if l.strip() and not l.startswith("#")]
        if not found:
            raise ControllerError(f"no trusted host key for {hostname} on this Mac; cannot set up the Spark-side restore")
        known += found
        lines += [f"Host {host}", f"  HostName {hostname}", f"  User {resolved.get('user') or 'spider'}",
                  f"  Port {resolved.get('port') or 22}", "  BatchMode yes", "  StrictHostKeyChecking yes",
                  "  UserKnownHostsFile __JOB_DIR__/known_hosts", ""]
    return "\n".join(lines), known


def restore_spec(job: dict, cfg: dict, prev: str | None, node: str | None) -> dict | None:
    if not prev:
        return None
    d = job["job_dir"]
    return {
        "model": prev, "node": node, "cwd": f"{d}/app", "attempts": 3, "retry_seconds": 60, "timeout_seconds": 3600,
        "argv": ["python3", f"{d}/app/spark-serve", "up", prev, "--node", node, "--no-hermes"],
        "env": {"SPARK_SERVE_STATE_DIR": f"{d}/controller", "SPARK_SERVE_NODE_SSH_CONFIG": f"{d}/ssh_config",
                "SPARK_SERVE_LOCAL_HOSTS": job["host"]},
    }


def upload_script(job: dict, tarball: bytes, ssh_config: str, known: list[str], supervisor: bytes) -> str:
    d = job["job_dir"]
    public = {k: v for k, v in job.items() if k != "prompt"}
    files = {
        "job.json": json.dumps(public, indent=2).encode(),
        "prompt.txt": (job["prompt"] + "\n").encode(),
        "ssh_config": ssh_config.encode(),
        "known_hosts": ("\n".join(known) + "\n").encode(),
        "supervisor.py": supervisor,
        "app.tgz": tarball,
    }
    parts = ["set -euo pipefail", "umask 077", f'D="$HOME/{d[2:]}"',
             'test ! -e "$D" || { echo "job directory exists" >&2; exit 3; }', 'mkdir -p "$D/controller"']
    for name, data in files.items():
        parts.append(f"base64 -d > \"$D/{name}\" <<'__SPARK_SERVE_B64__'\n{base64.b64encode(data).decode()}\n__SPARK_SERVE_B64__")
    parts += ['sed -i "s|__JOB_DIR__|$D|g" "$D/ssh_config"', 'tar -xzf "$D/app.tgz" -C "$D" && rm "$D/app.tgz"',
              'chmod 700 "$D/app/spark-serve"', 'echo uploaded']
    return "\n".join(parts) + "\n"


def preflight_script(job: dict, peers: list[str]) -> str:
    """Read-only checks on the home Spark; exit non-zero with a reason. Nothing stops."""
    d = job["job_dir"]
    checks = ["set -uo pipefail", f'D="$HOME/{d[2:]}"', "fail(){ echo \"$1\" >&2; exit 2; }",
              "python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' || fail 'python3 3.11+ is needed on the Spark for the restore'",
              "pgrep -f 'spark-serve-batch/.*/supervisor.py' >/dev/null && fail 'another batch job is running on this Spark'",
              'command -v nvidia-smi >/dev/null || fail "nvidia-smi missing"',
              'command -v setsid >/dev/null || fail "setsid missing"']
    for path in job.get("require_paths") or []:
        p = shlex.quote(path[2:]) if path.startswith("~/") else shlex.quote(path)
        prefix = '"$HOME"/' if path.startswith("~/") else ""
        checks.append(f'test -e {prefix}{p} || fail {shlex.quote("missing " + path)}')
    out = job["output"]
    out_dir = out.rsplit("/", 1)[0]
    od = ('"$HOME"/' + shlex.quote(out_dir[2:])) if out_dir.startswith("~/") else shlex.quote(out_dir)
    checks += [f"mkdir -p {od} && test -w {od} || fail 'output folder not writable'",
               f"avail=$(df -Pk {od} | awk 'NR==2{{print $4}}'); [ \"$avail\" -gt 2097152 ] || fail 'less than 2 GB free for the output'"]
    for peer in peers:
        checks.append(f'timeout 20 ssh -F "$D/ssh_config" {shlex.quote(peer)} true </dev/null >/dev/null 2>&1 '
                      f'|| fail {shlex.quote("the Spark cannot reach " + peer + " over SSH for the restore")}')
    checks.append(f'python3 "$D/app/spark-serve" list >/dev/null || fail "the spark-serve snapshot does not run on the Spark"')
    checks.append("echo preflight-ok")
    return "\n".join(checks) + "\n"


def launch_script(job: dict, controller_json: str) -> str:
    d = job["job_dir"]
    return "\n".join([
        "set -euo pipefail", "umask 077", f'D="$HOME/{d[2:]}"',
        f"cat > \"$D/controller/controller.json\" <<'__SPARK_SERVE_STATE__'\n{controller_json}\n__SPARK_SERVE_STATE__",
        'setsid -f nohup python3 "$D/supervisor.py" "$D" >"$D/supervisor.log" 2>&1 </dev/null',
        'for i in $(seq 1 30); do test -s "$D/state.json" && break; sleep 1; done',
        'cat "$D/state.json"',
    ]) + "\n"


def read_script(job_dir: str) -> str:
    return f'D="$HOME/{job_dir[2:]}"; cat "$D/state.json" 2>/dev/null || echo "{{}}"\n'


def stop_script(job_dir: str) -> str:
    return "\n".join([
        "set -uo pipefail", f'D="$HOME/{job_dir[2:]}"', 'touch "$D/cancel"',
        'pid=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get(\'supervisor_pid\') or \'\')" "$D/state.json" 2>/dev/null)',
        'if [ -n "$pid" ] && grep -q supervisor.py "/proc/$pid/cmdline" 2>/dev/null; then kill -TERM "$pid"; echo signalled; else echo "no-supervisor"; fi',
    ]) + "\n"


def cleanup_script(job_dir: str) -> str:
    # Only a never-started job directory is removed (its state.json does not exist yet).
    return f'D="$HOME/{job_dir[2:]}"; test -e "$D/state.json" || rm -rf -- "$D"\n'


# -- local record ---------------------------------------------------------

def record_path(directory: Path | None = None) -> Path:
    return (directory or state_dir()) / RECORD


def load_record(directory: Path | None = None) -> dict | None:
    try:
        value = json.loads(record_path(directory).read_text())
        return value if isinstance(value, dict) and value.get("id") else None
    except (OSError, ValueError):
        return None


def save_record(record: dict, directory: Path | None = None) -> None:
    atomic_json(record_path(directory), record)


def finished(job_state: dict) -> bool:
    return job_state.get("phase") == "finished" and job_state.get("restore") in FINAL_RESTORE
