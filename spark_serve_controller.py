"""Serialized workload ownership for the existing Spark Serve CLI and Mac app.

The controller owns modes; YuE workers own durable rendering jobs.  No model,
audio, credentials, or private catalog contents are put in discovery records.
"""

from __future__ import annotations

import contextlib
import copy
import fcntl
import inspect
import json
import os
import re
import shlex
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit


def classify_gpu_processes(compute: str, proc_root="/proc") -> dict:
    """Keep all contexts visible; exempt only the verified, small desktop daemon.

    nvidia-smi reports GNOME remote desktop's encoder context as compute. It is
    not a heavy inference engine. Process names alone never establish identity:
    require the kernel executable link, the exact observed user-service cgroup,
    and a reported allocation at or below the fixed 1 GiB ceiling. Any missing,
    unsupported, malformed, or inconsistent evidence remains blocking.
    """
    import csv
    import math
    import os
    import re
    from pathlib import Path

    expected_executable = "/usr/libexec/gnome-remote-desktop-daemon"
    desktop_memory_limit_mib = 1024
    processes, desktop, blocking = [], [], []
    for line in compute.splitlines():
        if not line.strip() or line.strip().startswith("No running processes"):
            continue
        record = {"raw": line, "classification": "unverified_compute"}
        try:
            columns = next(csv.reader([line], skipinitialspace=True))
            if len(columns) != 3:
                raise ValueError("expected PID, executable, memory")
            pid = int(columns[0].strip())
            if pid < 1:
                raise ValueError("invalid PID")
            name = columns[1].strip()
            memory = float(columns[2].strip())
            if not math.isfinite(memory) or memory < 0:
                raise ValueError("invalid reported memory")
            record.update(pid=pid, reported_executable=name, used_gpu_memory_mib=memory)
            # Read /proc only for this exact reported system executable. An
            # arbitrary process that copies its short name cannot qualify.
            if name == expected_executable and memory <= desktop_memory_limit_mib:
                proc = Path(proc_root) / str(pid)
                executable = os.path.realpath(os.readlink(proc / "exe"))
                cgroups = (proc / "cgroup").read_text().splitlines()
                verified_group = next((group for group in cgroups if re.fullmatch(
                    r"0::/user\.slice/user-([0-9]+)\.slice/user@\1\.service/app\.slice/gnome-remote-desktop-handover\.service",
                    group)), None)
                record.update(executable=executable, cgroup=verified_group)
                if executable == expected_executable and verified_group:
                    record["classification"] = "verified_desktop_context"
                    record["memory_limit_mib"] = desktop_memory_limit_mib
        except (OSError, ValueError, StopIteration, csv.Error) as exc:
            record["verification_error"] = type(exc).__name__
        processes.append(record)
        if record["classification"] == "verified_desktop_context":
            desktop.append(record)
        else:
            blocking.append(record)
    return {"gpu_processes": processes, "desktop_gpu_contexts": desktop,
            "blocking_gpu_processes": blocking}


class ControllerError(RuntimeError):
    pass


def guarded_remote_script(generation: str, command: str = "", *, fence=False) -> str:
    """Fence delayed SSH commands under a persistent per-node generation lock.

    A newer transition either waits for an earlier command's effects or fences
    it before it runs. The child inherits the lock descriptor if its parent is
    interrupted; checking a token and then releasing the lock would race.
    """
    source = """import fcntl, json, os, pathlib, subprocess, sys, tempfile, time
generation, action, command = sys.argv[1:]
root = pathlib.Path(os.environ.get("SPARK_SERVE_NODE_STATE_DIR", "~/.local/state/spark-serve")).expanduser()
root.mkdir(parents=True, exist_ok=True, mode=0o700)
with (root / "node.lock").open("a+") as lock:
    deadline = time.monotonic() + 30
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() >= deadline:
                sys.stderr.write("another remote workload operation is running; retry after it completes")
                raise SystemExit(75)
            time.sleep(0.1)
    state_path = root / "node.json"
    if action == "fence":
        fd, temporary = tempfile.mkstemp(prefix=".node.", dir=root)
        with os.fdopen(fd, "w") as out:
            json.dump({"version": 1, "generation": generation, "updated_at": time.time()}, out)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, state_path)
        parent = os.open(root, os.O_RDONLY)
        try: os.fsync(parent)
        finally: os.close(parent)
        print(json.dumps({"generation": generation}))
    else:
        try: state = json.loads(state_path.read_text())
        except (OSError, ValueError): state = {}
        if state.get("generation") != generation:
            sys.stderr.write("stale remote controller generation; command was not executed")
            raise SystemExit(75)
        env = dict(os.environ, SPARK_SERVE_NODE_LOCK_FD=str(lock.fileno()), SPARK_SERVE_NODE_GENERATION=generation)
        result = subprocess.run(["bash", "-s"], input=command, text=True, env=env, pass_fds=(lock.fileno(),), check=False)
        raise SystemExit(result.returncode)
"""
    return "# spark-serve: generation-fence\n" + shlex.join(
        ["python3", "-c", source, generation, "fence" if fence else "run", command]
    )


def state_dir() -> Path:
    return Path(
        os.environ.get("SPARK_SERVE_STATE_DIR", "~/.local/state/spark-serve")
    ).expanduser()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(value, out, sort_keys=True)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_state(directory: Path | None = None) -> dict:
    path = (directory or state_dir()) / "controller.json"
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict) or value.get("version") != 1:
            raise ValueError("invalid controller state")
        return value
    except FileNotFoundError:
        return {"version": 1, "mode": "unknown", "phase": "unmanaged"}
    except (ValueError, OSError) as exc:
        return {"version": 1, "mode": "unknown", "phase": "failed", "error": str(exc)}


def yue_profile(cfg: dict) -> dict:
    raw = cfg.get("yue", {})
    if not isinstance(raw, dict):
        raise ControllerError("[yue] must be a table")
    allowed = {
        "service",
        "factory_root",
        "port",
        "head_url",
        "worker_url",
        "ready_timeout",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ControllerError(f"unknown [yue] keys: {', '.join(sorted(unknown))}")
    c = cfg["cluster"]
    port = int(raw.get("port", 8011))
    if not 1 <= port <= 65535:
        raise ControllerError("yue.port must be in 1..65535")
    head_default = urlsplit(c["lan_url"]).hostname or c["head"]
    workers = [
        {
            "host": c["head"],
            "url": raw.get("head_url", f"http://{head_default}:{port}"),
        },
        {
            "host": c["worker"],
            "url": raw.get("worker_url", f"http://{c['worker']}:{port}"),
        },
    ]
    if workers[0]["host"] == workers[1]["host"]:
        raise ControllerError("YuE replicas require two distinct configured SSH hosts")
    for worker in workers:
        parsed = urlsplit(str(worker["url"]))
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in ("", "/")
        ):
            raise ControllerError(
                "YuE worker URLs must be HTTP(S) origins without credentials"
            )
        worker["url"] = str(worker["url"]).rstrip("/")
    service = str(raw.get("service", "yue-icl.service"))
    if not service.endswith(".service"):
        service += ".service"
    if "/" in service or service.startswith("-"):
        raise ControllerError("yue.service must be a systemd service name")
    timeout = int(raw.get("ready_timeout", 180))
    if timeout < 1 or timeout > 1800:
        raise ControllerError("yue.ready_timeout must be in 1..1800 seconds")
    return {
        "workers": workers,
        "service": service,
        "factory_root": str(
            raw.get("factory_root", "~/.local/share/artist-twin/yue-factory")
        ),
        "port": port,
        "ready_timeout": timeout,
    }


def yue_catalog_entry() -> dict:
    return {
        "id": "yue",
        "label": "YuE · two workers",
        "aliases": ["yue-icl"],
        "served_name": "yue",
        "ctx": 0,
        "image": "yue-icl-spark",
        "notes": "Independent song/take workers. Drains active jobs before switching modes.",
        "wrapper": "yue",
        "backend": "yue",
        "topology": "replicas",
        "hermes_provider": "",
    }


def health_ready(
    health: dict, generation: str | None = None, *, admitted: bool = True
) -> bool:
    """HTTP success is insufficient: require validated assets and fenced admission."""
    if not isinstance(health, dict):
        return False
    good = (
        health.get("service") == "yue-icl-factory"
        and health.get("api_version") == 2
        and health.get("runtime_ok") is True
        and health.get("cuda") is True
        and health.get("assets_ok") is True
        and health.get("mock") is False
        and health.get("ownership_ok") is True
        and bool(health.get("worker_id"))
        and isinstance(health.get("runtime_manifest"), str)
        and bool(re.fullmatch(r"[0-9a-f]{64}", health["runtime_manifest"]))
    )
    if admitted:
        good = (
            good
            and health.get("accepting") is True
            and bool(generation)
            and health.get("generation") == generation
        )
    return bool(good)


class Controller:
    def __init__(self, cfg: dict, ssh, emit=None, directory: Path | None = None):
        self.cfg, self.ssh = cfg, ssh
        self.emit = emit or (lambda *args, **kwargs: None)
        self.directory = directory or state_dir()
        self.profile = yue_profile(cfg)
        self.state = read_state(self.directory)
        self.operation_generation = None
        self.operation_hosts = None

    def _workers(self):
        """Lifecycle work is restricted to the current transition's hosts."""
        return [worker for worker in self.profile["workers"]
                if self.operation_hosts is None or worker["host"] in self.operation_hosts]

    def _model(self, name):
        key = str(name or "").lower()
        if not key:
            return None, None
        for mid, model in self.cfg.get("models", {}).items():
            if key in {mid.lower(), str(model.get("served_name", "")).lower(),
                       *(str(alias).lower() for alias in model.get("aliases", []))}:
                return mid, model
        return None, None

    def node_states(self) -> dict:
        """Return ownership by actual SSH host, lazily adapting legacy state.

        Reading never mutates a live service or writes a migration. The next
        locked transition persists the map alongside the version-1 summary.
        """
        hosts = [worker["host"] for worker in self.profile["workers"]]
        saved = self.state.get("nodes")
        if isinstance(saved, dict):
            return {host: copy.deepcopy(saved.get(host) or {
                "mode": "unknown", "model": None, "phase": "unmanaged",
                "generation": None, "error": "Node ownership has not been recorded",
                "allocation_hosts": [], "allocation_id": None,
            }) for host in hosts}
        state = self.state
        mode, phase = state.get("mode", "unknown"), state.get("phase", "unmanaged")
        mid, model = self._model(state.get("model") or state.get("target"))
        participants = hosts
        if mode == "vllm" and model is not None and int(model.get("nnodes") or self.cfg["cluster"].get("nnodes") or 1) == 1:
            participants = hosts[:1]
        uncertain_group = (
            phase not in ("ready", "stopped", "unmanaged")
            and mode != "yue"
            and (model is None or int(model.get("nnodes") or self.cfg["cluster"].get("nnodes") or 1) > 1)
        )
        result = {}
        for worker in self.profile["workers"]:
            host = worker["host"]
            inactive = (mode == "none" and phase == "stopped") or (
                mode == "vllm" and phase == "ready" and host not in participants)
            record = {
                "mode": "none" if inactive else mode,
                "model": None if inactive else mid,
                "phase": "stopped" if inactive else phase,
                "generation": state.get("generation"),
                "error": None if inactive else state.get("error"),
                "allocation_hosts": hosts if uncertain_group else [] if inactive else ([host] if mode == "yue" else participants if mode == "vllm" else []),
                "allocation_id": None if inactive else state.get("generation"),
                "legacy": True,
            }
            if mode == "yue":
                receipt = next((item for item in state.get("workers", [])
                                if isinstance(item, dict) and item.get("url", "").rstrip("/") == worker["url"]), None)
                if receipt:
                    record["worker"] = copy.deepcopy(receipt)
            result[host] = record
        return result

    def _save_selected(self, *, _hosts=None, **fields):
        nodes = self.node_states()
        for worker in self._workers():
            if _hosts is not None and worker["host"] not in _hosts:
                continue
            record = nodes[worker["host"]]
            record.update(copy.deepcopy(fields))
            if (fields.get("phase") == "draining" or fields.get("mode") == "none"
                    or (fields.get("phase") == "starting" and "allocation_id" in fields)):
                record.pop("containers", None)
                record.pop("served", None)
            if fields.get("mode") in ("none", "vllm"):
                record.pop("worker", None)
        active = [record for record in nodes.values() if record.get("mode") != "none"]
        modes = {record.get("mode", "unknown") for record in active}
        mode = next(iter(modes)) if len(modes) == 1 else "mixed" if modes else "none"
        models = {record.get("model") for record in active if record.get("mode") == "vllm"}
        summary = {"mode": mode, "model": next(iter(models)) if len(models) == 1 else None}
        # Legacy fields describe the latest operation. Node records preserve
        # independent peer availability and errors even when that operation fails.
        summary.update({key: fields[key] for key in ("phase", "target", "generation", "error", "cleanup_errors") if key in fields})
        self.save(nodes=nodes, **summary)

    def _check_scope(self, hosts):
        """Reject splitting a distributed allocation before any remote mutation."""
        if len(hosts) == len(self.profile["workers"]):
            return
        host = hosts[0]
        nodes = self.node_states()
        for record in nodes.values():
            if (not isinstance(record, dict)
                    or (record.get("mode") == "unknown"
                        and (record.get("phase") != "unmanaged" or record.get("error")))):
                # A new installation has genuinely unmanaged nodes, without an
                # error. A failed read or incomplete persisted map instead lost
                # evidence about whether the peer owns a distributed allocation.
                raise ControllerError("Node ownership is unknown; select both nodes to reconcile it")
            members = record.get("allocation_hosts") or []
            if host in members and not set(members).issubset(hosts):
                raise ControllerError(f"{host}: workload uses both Sparks; select both nodes to stop or replace it")
        record = nodes[host]
        known_solo = (record.get("mode") == "vllm"
                      and record.get("allocation_hosts") == [host]
                      and self._model(record.get("model"))[1] is not None)
        # Inspect only the selected node. An offline unrelated peer must not
        # prevent a local change. Labels recover allocations after lost state.
        for container in self.audit(host)["containers"]:
            if container["name"] not in self._owned_vllm_names():
                continue
            labels = container.get("labels") or {}
            members = labels.get("ai.spark-serve.hosts")
            command = " ".join(container.get("command") or [])
            ranks = re.search(r"--nnodes(?:=|\s+)(\d+)", command)
            if (ranks and int(ranks.group(1)) > 1) or "--headless" in command:
                raise ControllerError(f"{host}: container is a distributed rank; select both nodes to reconcile it")
            if members is not None:
                try:
                    members = json.loads(members)
                    valid = (isinstance(members, list) and members and all(isinstance(item, str) for item in members)
                             and host in members and len(set(members)) == len(members))
                except (ValueError, TypeError):
                    valid = False
                if not valid or not set(members).issubset(hosts):
                    raise ControllerError(f"{host}: container allocation requires both nodes or is unknown; select both nodes to reconcile it")
            elif not known_solo:
                raise ControllerError(f"{host}: untracked container may be a distributed rank; select both nodes to reconcile it")

    @contextlib.contextmanager
    def lock(self):
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (self.directory / "controller.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ControllerError(
                    "another Spark Serve mode transition is running; wait for it to finish"
                ) from None
            try:
                self.state = read_state(self.directory)
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def save(self, **fields):
        self.state.update(fields, version=1, updated_at=time.time())
        atomic_json(self.directory / "controller.json", self.state)

    def revoke(self):
        selected = {worker["host"] for worker in self._workers()}
        remaining = [record["worker"] for host, record in self.node_states().items()
                     if host not in selected and record.get("mode") == "yue"
                     and record.get("phase") == "ready" and isinstance(record.get("worker"), dict)]
        generations = {worker.get("generation") for worker in remaining}
        # The version-1 Artist Twin contract requires one shared admission
        # generation. We only preserve old receipts; scoped YuE starts are not
        # supported until that consumer accepts independently admitted workers.
        if remaining and (len(generations) != 1 or not next(iter(generations))):
            raise ControllerError("YuE discovery generations disagree; select both nodes to reconcile admission")
        atomic_json(
            self.directory / "yue-workers.json",
            {
                "version": 1,
                "mode": "yue" if remaining else "unavailable",
                "generation": next(iter(generations)) if remaining else self.state.get("generation"),
                "workers": remaining,
            },
        )

    def remote(self, host: str, script: str, *, json_output=False, fenced=True):
        if self.operation_hosts is not None and host not in self.operation_hosts:
            raise ControllerError(f"{host}: operation is restricted to the selected nodes")
        if fenced and self.operation_generation:
            script = guarded_remote_script(self.operation_generation, script)
        proc = self.ssh(self.cfg, host, script, check=False)
        if proc.returncode:
            raise ControllerError(
                f"{host}: {(proc.stderr or proc.stdout or 'SSH command failed').strip()[:1200]}"
            )
        if not json_output:
            return proc.stdout.strip()
        try:
            value = json.loads(proc.stdout)
            if not isinstance(value, dict):
                raise TypeError("not an object")
            return value
        except (ValueError, TypeError):
            raise ControllerError(
                f"{host}: invalid control response; ownership is unknown"
            ) from None

    def control(self, host: str, action: str, **options) -> dict:
        # Inherit the service's explicit Environment, without printing it. This
        # keeps CLI controls on the same database/manifest as the HTTP process.
        script = """import json, os, pathlib, re, shlex, subprocess, sys
root, service, action, options = sys.argv[1:]
factory = pathlib.Path(root).expanduser() / "yue_factory.py"
if not factory.is_file():
    state = subprocess.run(["systemctl", "--user", "show", service, "--property=ActiveState", "--value"], text=True, capture_output=True, timeout=15)
    if state.returncode != 0 or state.stdout.strip() not in ("inactive", "failed"):
        sys.stderr.write("factory source is missing and service ownership cannot be verified; install protocol v2 first")
        raise SystemExit(2)
    print(json.dumps({"installed": False, "accepting": False, "active_job": None, "owned_containers": []}))
    raise SystemExit(0)
if not re.search(r"^API_VERSION = 2$", factory.read_text(), re.MULTILINE):
    sys.stderr.write("upgrade the YuE factory to protocol v2 before changing modes; legacy source was not executed")
    raise SystemExit(2)
env = os.environ.copy()
p = subprocess.run(["systemctl", "--user", "show", service, "--property=Environment", "--value"], text=True, capture_output=True, timeout=15)
if p.returncode == 0:
    for pair in shlex.split(p.stdout):
        key, sep, value = pair.partition("=")
        if sep: env[key] = value
argv = [sys.executable, str(factory), "control", action]
for key, value in json.loads(options).items(): argv += ["--" + key.replace("_", "-"), str(value)]
lock_fd = os.environ.get("SPARK_SERVE_NODE_LOCK_FD")
lock_fds = (int(lock_fd),) if lock_fd is not None else ()
for key in ("SPARK_SERVE_NODE_LOCK_FD", "SPARK_SERVE_NODE_GENERATION"):
    if key in os.environ: env[key] = os.environ[key]
p = subprocess.run(argv, env=env, text=True, capture_output=True, timeout=600, pass_fds=lock_fds)
sys.stdout.write(p.stdout)
sys.stderr.write(p.stderr)
raise SystemExit(p.returncode)
"""
        command = "# spark-serve: yue-control\n" + shlex.join(
            [
                "python3",
                "-c",
                script,
                self.profile["factory_root"],
                self.profile["service"],
                action,
                json.dumps(options),
            ]
        )
        status = self.remote(host, command, json_output=True)
        if status.get("installed") is not False and status.get("api_version") != 2:
            raise ControllerError(
                f"{host}: upgrade the YuE factory to protocol v2 before changing modes"
            )
        return status

    def fence_nodes(self, generation: str):
        errors = []
        for worker in self._workers():
            try:
                self.remote(
                    worker["host"],
                    guarded_remote_script(generation, fence=True),
                    fenced=False,
                )
            except ControllerError as exc:
                errors.append(str(exc))
        return errors

    def audit(self, host: str) -> dict:
        script = inspect.getsource(classify_gpu_processes) + "\n" + """import json, socket, subprocess, sys
def run(args):
    p = subprocess.run(args, text=True, capture_output=True, timeout=30)
    if p.returncode: raise RuntimeError(p.stderr.strip() or "command failed")
    return p.stdout.strip()
ids = run(["docker", "ps", "-aq"]).split()
containers = json.loads(run(["docker", "inspect", *ids])) if ids else []
compute = run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"])
listening = []
for port in json.loads(sys.argv[1]):
    with socket.socket() as probe:
        probe.settimeout(1)
        if probe.connect_ex(("127.0.0.1", port)) == 0: listening.append(port)
print(json.dumps({"listening_ports": listening, "containers": [{"id": c["Id"], "name": c["Name"].lstrip("/"), "running": c["State"]["Running"], "gpu": bool(c["HostConfig"].get("DeviceRequests")) or any("nvidia" in str(d) for d in c["HostConfig"].get("Devices") or []), "labels": c["Config"].get("Labels") or {}, "command": (c["Config"].get("Entrypoint") or []) + (c["Config"].get("Cmd") or [])} for c in containers], **classify_gpu_processes(compute) }))
"""
        return self.remote(
            host,
            "# spark-serve: audit\n"
            + shlex.join(
                [
                    "python3",
                    "-c",
                    script,
                    json.dumps(
                        [
                            int(self.cfg["cluster"].get("port") or 8000),
                            self.profile["port"],
                        ]
                    ),
                ]
            ),
            json_output=True,
        )

    def _owned_vllm_names(self):
        c = self.cfg["cluster"]
        names = {
            str(c.get("container") or "vllm_cluster"),
            *(str(n) for n in c.get("stop_names") or []),
        }
        names.update(
            str(m["container"])
            for m in (self.cfg.get("models") or {}).values()
            if m.get("container")
        )
        return names - set(self.cfg["cluster"].get("keep_containers") or [])

    def stop_vllm(self):
        """Only exact catalog-owned IDs; do not kill arbitrary :8000 occupants."""
        names = self._owned_vllm_names()
        for worker in self._workers():
            host = worker["host"]
            before = self.audit(host)
            ids = [c["id"] for c in before["containers"] if c["name"] in names]
            if ids:
                self.remote(
                    host,
                    "# spark-serve: stop-owned-vllm\n"
                    + shlex.join(["docker", "rm", "-f", *ids]),
                )
            after = self.audit(host)
            if any(c["name"] in names for c in after["containers"]):
                raise ControllerError(f"{host}: catalog containers remain after stop")
            self.emit(
                "stop",
                host=host,
                output="catalog containers stopped; protected containers retained",
            )

    def verify_idle(self):
        for worker in self._workers():
            host = worker["host"]
            audit = self.audit(host)
            if audit["listening_ports"]:
                raise ControllerError(
                    f"{host}: unmanaged listener remains on ports {audit['listening_ports']}; stop or reconcile it before switching"
                )
            gpu = [c["name"] for c in audit["containers"] if c["running"] and c["gpu"]]
            desktop = audit.get("desktop_gpu_contexts", [])
            if desktop:
                self.emit("desktop_gpu_context", host=host, contexts=desktop,
                          output="verified GNOME remote desktop contexts retained")
            if gpu or audit.get("blocking_gpu_processes", audit["gpu_processes"]):
                raise ControllerError(
                    f"{host}: GPU still in use ({', '.join(gpu) or 'compute processes'}); stop or reconcile that workload before switching"
                )

    def service(self, host: str, action: str):
        self.remote(
            host,
            "# spark-serve: service\n"
            + shlex.join(["systemctl", "--user", action, self.profile["service"]]),
        )

    def stop_yue(self, *, cancel_jobs: bool):
        statuses, errors = [], []
        # Drain every reachable node even if another is unknown. New dispatch
        # has already been revoked; in-flight submissions meet worker fencing.
        for worker in self._workers():
            try:
                statuses.append((worker, self.control(worker["host"], "drain")))
            except ControllerError as exc:
                errors.append(str(exc))
        if errors:
            raise ControllerError("; ".join(errors))
        active = []
        for worker, status in statuses:
            host = worker["host"]
            if status.get("installed") is False:
                continue
            if status.get("accepting") is not False:
                raise ControllerError(f"{host}: worker did not acknowledge draining")
            job = status.get("active_job")
            if job:
                job_id = (
                    (job.get("job_id") or job.get("id"))
                    if isinstance(job, dict)
                    else str(job)
                )
                if cancel_jobs and job_id:
                    self.control(host, "cancel", job_id=job_id)
                    status = self.control(host, "status")
                else:
                    active.append(f"{host}:{job_id}")
                    continue
            if (
                status.get("busy")
                or status.get("active_job")
                or status.get("ownership_ok") is not True
                or any(c.get("running") for c in status.get("owned_containers", []))
            ):
                raise ControllerError(
                    f"{host}: YuE ownership is unresolved; refusing to release its GPU"
                )
        if active:
            raise ControllerError(
                "YuE is draining active jobs "
                + ", ".join(active)
                + "; retry after completion, or use --cancel-jobs to explicitly cancel them"
            )
        for worker, status in statuses:
            if status.get("installed") is not False:
                self.service(worker["host"], "stop")
                # A service restart is always drained, but verify shutdown too.
                check = self.remote(
                    worker["host"],
                    shlex.join(
                        [
                            "systemctl",
                            "--user",
                            "show",
                            self.profile["service"],
                            "--property=ActiveState",
                            "--value",
                        ]
                    ),
                )
                if check not in ("inactive", "failed"):
                    raise ControllerError(
                        f"{worker['host']}: YuE service is still {check or 'unknown'}"
                    )

    def health(self, worker: dict) -> dict:
        # Probe from the Mac, the same network origin used by Artist Twin.
        import urllib.request

        try:
            with urllib.request.urlopen(
                worker["url"] + "/health", timeout=5
            ) as response:
                value = json.load(response)
            if not isinstance(value, dict):
                raise TypeError("invalid health")
            return value
        except (OSError, ValueError, TypeError) as exc:
            raise ControllerError(
                f"{worker['host']}: worker health unavailable ({type(exc).__name__})"
            ) from None

    def start_yue(self, generation: str):
        ready = []
        for worker in self._workers():
            host = worker["host"]
            self.service(host, "restart")
            deadline = time.monotonic() + self.profile["ready_timeout"]
            while True:
                try:
                    health = self.health(worker)
                    if (
                        health.get("api_version") == 2
                        and health.get("accepting") is False
                    ):
                        break
                except ControllerError:
                    pass
                if time.monotonic() >= deadline:
                    raise ControllerError(
                        f"{host}: factory did not start drained with protocol v2"
                    )
                time.sleep(1)
            self.emit("worker_start", host=host, output="YuE factory started drained")
            # admit runs a pinned-assets and CUDA validation before opening
            # admission. No worker list is published until BOTH acknowledge.
            control = self.control(host, "admit", generation=generation)
            health = self.health(worker)
            if not health_ready(control, generation) or not health_ready(
                health, generation
            ):
                raise ControllerError(
                    f"{host}: YuE runtime/assets/admission validation failed"
                )
            if (
                control["worker_id"] != health["worker_id"]
                or control["runtime_manifest"] != health["runtime_manifest"]
            ):
                raise ControllerError(
                    f"{host}: factory control and HTTP endpoint do not identify the same runtime"
                )
            ready.append(
                {
                    "id": health["worker_id"],
                    "url": worker["url"],
                    "generation": generation,
                    "runtime_manifest": health["runtime_manifest"],
                }
            )
            self.emit(
                "worker_ready",
                host=host,
                worker_id=health["worker_id"],
                ready_workers=len(ready),
                total_workers=len(self.profile["workers"]),
            )
        if len({w["id"] for w in ready}) != len(ready):
            raise ControllerError(
                "YuE endpoints identify the same worker; replicas require distinct physical workers"
            )
        self.save(
            mode="yue", phase="ready", generation=generation, workers=ready, error=None
        )
        nodes = self.node_states()
        for worker in self._workers():
            receipt = next(item for item in ready if item["url"] == worker["url"])
            nodes[worker["host"]].update(
                mode="yue", model="yue", phase="ready", generation=generation,
                error=None, worker=receipt, allocation_hosts=[worker["host"]],
                allocation_id=generation,
            )
        self.save(nodes=nodes)
        atomic_json(
            self.directory / "yue-workers.json",
            {"version": 1, "mode": "yue", "generation": generation, "workers": ready},
        )
        self.emit("ready", served="yue", ready_workers=len(ready), workers=ready)

    def switch(self, target: str, start_vllm=None, *, cancel_jobs=False, no_wait=False, after_idle=None, hosts=None):
        configured = [worker["host"] for worker in self.profile["workers"]]
        if hosts is None:
            selected = configured
        elif (not isinstance(hosts, list) or not hosts or len(set(hosts)) != len(hosts)
              or any(host not in configured for host in hosts)):
            raise ControllerError("Select one configured Spark host or both nodes")
        else:
            selected = [host for host in configured if host in hosts]
        if target == "yue" and selected != configured:
            raise ControllerError("Starting YuE currently requires both nodes; select both Sparks")
        mid, model = self._model(target)
        if target not in ("none", "yue") and model is None:
            raise ControllerError(f"Unknown workload {target!r}")
        nnodes = int((model or {}).get("nnodes") or self.cfg["cluster"].get("nnodes") or 1)
        if model is not None and nnodes > len(selected):
            raise ControllerError(f"{mid} requires both nodes; select both Sparks")
        launch_hosts = selected[:1] if model is not None and nnodes == 1 else selected
        with self.lock():
            self._check_scope(selected)
            self.operation_hosts = selected
            generation = str(uuid.uuid4())
            self.operation_generation = generation
            started = False
            try:
                self._save_selected(
                    target=target,
                    phase="draining",
                    generation=generation,
                    error=None,
                    cleanup_errors=[],
                )
                self.revoke()
                errors = self.fence_nodes(generation)
                try:
                    self.stop_yue(cancel_jobs=cancel_jobs)
                except ControllerError as exc:
                    errors.append(str(exc))
                if errors:
                    raise ControllerError("; ".join(errors))
                self._save_selected(phase="stopping")
                self.stop_vllm()
                self.verify_idle()
                self._save_selected(mode="none", model=None, phase="stopped", allocation_hosts=[], allocation_id=None, legacy=False)
                if after_idle is not None:
                    after_idle()
                if target == "none":
                    return
                self._save_selected(
                    _hosts=launch_hosts, phase="starting", model=mid if model is not None else "yue",
                    mode="vllm" if model is not None else "yue",
                    allocation_hosts=launch_hosts if model is not None else [], allocation_id=generation,
                    legacy=False,
                )
                started = True
                if target == "yue":
                    self.emit(
                        "start",
                        model="yue",
                        label="YuE · two workers",
                        image="yue-icl-spark",
                        served="yue",
                        ctx=0,
                        url="",
                        nnodes=2,
                        backend="yue",
                        topology="replicas",
                    )
                    self.start_yue(generation)
                else:
                    receipt = start_vllm(generation)
                    if receipt is not None:
                        containers = receipt.get("containers") if isinstance(receipt, dict) else None
                        if (not isinstance(containers, list) or len(containers) != len(launch_hosts)
                                or any(not isinstance(item, dict) or item.get("host") not in launch_hosts
                                       or not isinstance(item.get("id"), str)
                                       or not re.fullmatch(r"[0-9a-f]{64}", item["id"])
                                       or item.get("rank") != launch_hosts.index(item["host"])
                                       for item in containers)
                                or {item["host"] for item in containers} != set(launch_hosts)):
                            raise ControllerError("Model launch returned invalid immutable container receipts")
                        if not no_wait and receipt.get("served") != model["served_name"]:
                            raise ControllerError("Model launch receipt does not confirm the expected served model")
                        nodes = self.node_states()
                        for host in launch_hosts:
                            nodes[host]["containers"] = [copy.deepcopy(item) for item in containers if item["host"] == host]
                            nodes[host]["served"] = receipt.get("served")
                        self.save(nodes=nodes)
                    self._save_selected(
                        _hosts=launch_hosts,
                        mode="vllm",
                        model=mid,
                        phase="starting" if no_wait else "ready",
                        generation=generation,
                        error=None,
                    )
            except BaseException as exc:
                # Never turn a drain-pending render into an implicit cancel.
                # Once old workloads were stopped, clean up any partial start.
                cleanup = []
                try:
                    self.revoke()
                except Exception as err:
                    cleanup.append(str(err))
                if started:
                    try:
                        self.stop_yue(cancel_jobs=False)
                    except Exception as err:  # noqa: BLE001 — retain the original failure while reporting incomplete cleanup
                        cleanup.append(str(err))
                    try:
                        self.stop_vllm()
                    except Exception as err:  # noqa: BLE001 — retain the original failure while reporting incomplete cleanup
                        cleanup.append(str(err))
                self._save_selected(phase="failed", error=str(exc), cleanup_errors=cleanup)
                raise
            finally:
                self.operation_generation = None
                self.operation_hosts = None

    def status(self) -> dict:
        state = read_state(self.directory)
        self.state = state
        nodes = self.node_states()
        generation = state.get("generation")
        workers = []
        for worker in self.profile["workers"]:
            item = {
                "host": worker["host"],
                "url": worker["url"],
                "ready": False,
                "accepting": False,
                "busy": False,
            }
            try:
                control = self.control(worker["host"], "status")
                item.update(
                    id=control.get("worker_id"),
                    accepting=bool(control.get("accepting")),
                    busy=bool(control.get("busy")),
                    active_job=control.get("active_job"),
                    generation=control.get("generation"),
                )
                owner = nodes[worker["host"]]
                if owner.get("mode") == "yue" and owner.get("phase") == "ready":
                    health = self.health(worker)
                    item["ready"] = (
                        health_ready(health, owner.get("generation"))
                        and health_ready(control, owner.get("generation"))
                        and control.get("worker_id") == health.get("worker_id")
                        and control.get("runtime_manifest")
                        == health.get("runtime_manifest")
                    )
            except ControllerError as exc:
                item["error"] = str(exc)
            workers.append(item)
        return {
            "backend": "yue" if state.get("mode") == "yue" else "vllm",
            "mode": state.get("mode"),
            "phase": state.get("phase"),
            "generation": generation,
            "target": state.get("target"),
            "transition_error": state.get("error"),
            "yue_workers": workers,
            "ready_workers": sum(w["ready"] for w in workers),
            "total_workers": len(workers),
            "nodes": [dict(host=host, **record) for host, record in nodes.items()],
        }
