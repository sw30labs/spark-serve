"""Bounded synthetic benchmarks for verified, controller-owned LLM allocations.

A lease covers physical hosts, not API URLs. Lifecycle operations revoke leases
under the controller lock before changing those hosts. No generated code runs.
"""
from __future__ import annotations

import concurrent.futures
import datetime
import fcntl
import http.client
import json
import os
import select
from pathlib import Path
import socket
import statistics
import sys
import threading
import time
from urllib.parse import urlsplit, urlunsplit
import uuid

from tools.qwen_acceptance import MAX_LINE, MAX_RESPONSE, ProtocolError, StreamResult, sse_events

DEADLINE_SECONDS = 300
REQUEST_SECONDS = 120
POLL_SECONDS = 0.2
IDENTITY_SECONDS = 5
EVENT_SECONDS = 2
TENSORFOLD_SETTINGS = ("model_revision", "parallel", "kv_dtype", "vision", "ple_on_ssd",
                      "mtp_drafts", "mtp_confidence", "thinking", "temperature", "top_p", "top_k",
                      "draft_revision", "dense", "drafter", "vision_urls", "max_tokens", "communication")


class BenchmarkCancelled(RuntimeError):
    pass


class _EventWriter:
    """Bound output backpressure without a queue or a second writer thread."""

    def __init__(self):
        self.failed = False

    def __call__(self, event):
        if self.failed:
            return
        data = json.dumps(event) + "\n"
        try:
            try:
                fd = sys.stdout.fileno()
            except (AttributeError, OSError):
                # StringIO and other in-memory test streams have no descriptor.
                sys.stdout.write(data)
                sys.stdout.flush()
                return
            blocking = os.get_blocking(fd)
            try:
                os.set_blocking(fd, False)
                remaining = memoryview(data.encode("utf-8"))
                deadline = time.monotonic() + EVENT_SECONDS
                while remaining:
                    wait = deadline - time.monotonic()
                    if wait <= 0 or not select.select([], [fd], [], wait)[1]:
                        raise TimeoutError("Benchmark event consumer is not reading")
                    try:
                        written = os.write(fd, remaining)
                    except BlockingIOError:
                        continue
                    if not written:
                        raise BrokenPipeError("Benchmark event consumer closed")
                    remaining = remaining[written:]
            finally:
                try:
                    os.set_blocking(fd, blocking)
                except OSError:
                    pass
        except BaseException:
            self.failed = True
            raise


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _controller(cfg, ssh_run=None):
    from spark_serve_controller import Controller
    return Controller(cfg, ssh_run or (lambda *a, **k: None))


def _root():
    from spark_serve_controller import state_dir
    return state_dir()


def _save(path, value):
    from spark_serve_controller import atomic_json
    atomic_json(path, value)


def _leases(root):
    return Path(root) / "benchmarks" / "leases"


def revoke_benchmarks(root: Path, hosts: list[str]) -> int:
    """Revoke overlapping leases. Caller MUST hold Controller.lock()."""
    from spark_serve_controller import ControllerError
    count = 0
    pending = []
    for path in _leases(root).glob("*.json"):
        try:
            lease = json.loads(path.read_text())
            if not lease.get("revoked") and set(hosts).intersection(lease["hosts"]):
                lease.update(revoked=True, revoked_at=_now())
                _save(path, lease)
                count += 1
            if set(hosts).intersection(lease["hosts"]):
                pending.append(path.with_suffix(".lock"))
        except FileNotFoundError:
            continue  # The client completed and removed its unique marker.
        except (OSError, ValueError, KeyError, TypeError):
            # Unknown ownership cannot safely be excluded from this transition.
            _save(path, {"revoked": True, "expires_at": time.time() + DEADLINE_SECONDS + 5,
                         "hosts": list(hosts), "revoked_at": _now()})
            pending.append(path.with_suffix(".lock"))
            count += 1
    deadline = time.monotonic() + 5
    for path in pending:
        with path.open("a+") as handle:
            while True:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise ControllerError("benchmark did not acknowledge cancellation; retry after it exits")
                    time.sleep(0.05)
            fcntl.flock(handle, fcntl.LOCK_UN)
        # Admission remains serialized by the caller. A released lock proves
        # no client request can still launch from this unique lease.
        path.with_suffix(".json").unlink(missing_ok=True)
        path.unlink(missing_ok=True)
    return count


def cancel_benchmarks(cfg, hosts=None) -> int:
    controller = _controller(cfg)
    with controller.lock():
        return revoke_benchmarks(controller.directory, hosts or
                                 [cfg["cluster"]["head"], cfg["cluster"]["worker"]])


def list_benchmarks(cfg) -> list:
    """Completed run history, newest first; never expose credential environment."""
    runs = []
    for path in (_root() / "benchmarks").glob("*.json"):
        try:
            item = json.loads(path.read_text())
            if isinstance(item, dict) and item.get("id") and item.get("finished_at"):
                runs.append(item)
        except (OSError, ValueError):
            continue
    return sorted(runs, key=lambda r: r["started_at"], reverse=True)[:100]


def _identity(cfg, status, owners, node):
    from spark_serve_controller import ControllerError
    if node not in ("head", "worker"):
        raise ControllerError("benchmark node must be head or worker")
    nodes = {n["host"]: n for n in status.get("nodes", [])}
    host = cfg["cluster"][node]
    selected = nodes.get(host, {})
    hosts = selected.get("allocation_hosts") or []
    if not hosts or host not in hosts or set(hosts) - set(nodes):
        raise ControllerError("benchmark requires a known managed allocation")
    if len(hosts) > 1:
        node, host = "head", cfg["cluster"]["head"]
        selected = nodes.get(host, {})
    model_id = selected.get("model")
    model = cfg.get("models", {}).get(model_id, {})
    owner = owners.get(host, {})
    allocation_id = owner.get("allocation_id")
    if not allocation_id or not model or selected.get("served") != model.get("served_name"):
        raise ControllerError("benchmark requires a verified catalog model and allocation identity")
    containers = []
    for member in hosts:
        live, assigned = nodes.get(member, {}), owners.get(member, {})
        if (not live.get("ready") or not live.get("ours_running")
                or live.get("model") != model_id or live.get("phase") != "ready"
                or assigned.get("mode") != "vllm" or assigned.get("phase") != "ready"
                or assigned.get("allocation_id") != allocation_id
                or set(assigned.get("allocation_hosts") or []) != set(hosts)):
            raise ControllerError("benchmark requires every allocation member to be managed and ready")
        recorded = {c["id"] for c in assigned.get("containers", []) if c.get("id")}
        observed = {c["id"]: c for c in live.get("containers", []) if c.get("running") and c.get("id") in recorded}
        if not recorded or set(observed) != recorded:
            raise ControllerError("benchmark container identity changed or is unavailable")
        for cid, container in sorted(observed.items()):
            labels = container.get("labels") or {}
            if (labels.get("ai.spark-serve.allocation") != allocation_id
                    or labels.get("ai.spark-serve.model") != model_id):
                raise ControllerError("benchmark container ownership labels do not match")
            containers.append({"host": member, "id": cid, "image": container.get("image")})
    endpoint = selected.get("url", "").rstrip("/")
    parsed = urlsplit(endpoint)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/")):
        raise ControllerError("benchmark endpoint must be a configured HTTP(S) origin")
    engine = selected.get("runtime") or (model.get("wrapper") if model.get("wrapper") in ("nim", "tensorfold") else "vllm")
    runtime = {"engine": engine, "image": model.get("image") or cfg["cluster"].get("image")}
    if engine == "tensorfold":
        # Retain public launch settings for comparisons without copying runtime
        # environment or host paths into saved benchmark history.
        settings = model.get("tensorfold") or {}
        runtime["settings"] = {key: settings[key] for key in TENSORFOLD_SETTINGS if key in settings}
        runtime["settings_source"] = "catalog"
        runtime["max_model_len"] = model.get("max_model_len")
    return {"model": model_id, "served_name": model["served_name"], "node": node,
            "hosts": sorted(hosts), "endpoint": endpoint + "/v1",
            "runtime": runtime,
            "allocation": {"id": allocation_id, "containers": sorted(containers, key=lambda c: (c["host"], c["id"]))}}


def _owner_identity(root, identity):
    from spark_serve_controller import read_state
    nodes = read_state(root).get("nodes", {})
    for host in identity["hosts"]:
        owner = nodes.get(host, {})
        expected = {c["id"] for c in identity["allocation"]["containers"] if c["host"] == host}
        actual = {c.get("id") for c in owner.get("containers", [])}
        if (owner.get("phase") != "ready" or owner.get("mode") != "vllm"
                or owner.get("allocation_id") != identity["allocation"]["id"]
                or owner.get("model") != identity["model"] or expected != actual
                or set(owner.get("allocation_hosts") or []) != set(identity["hosts"])):
            return False
    return True


def _stream_request(url, payload, timeout, check):
    """Strict SSE accounting with a watcher that can interrupt blocked reads."""
    endpoint = urlsplit(url.rstrip("/") + "/chat/completions")
    cls = http.client.HTTPSConnection if endpoint.scheme == "https" else http.client.HTTPConnection
    conn = cls(endpoint.hostname, endpoint.port, timeout=min(timeout, 2))
    started = time.monotonic()
    stop = threading.Event()
    interrupted = []
    transport = None

    def watch():
        while not stop.wait(POLL_SECONDS):
            try:
                check()
                if time.monotonic() - started >= timeout:
                    raise TimeoutError("Request deadline exceeded")
            except Exception as exc:
                interrupted.append(exc)
                if transport:
                    try:
                        transport.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                return

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        check()
        conn.connect()
        transport = conn.sock
        if interrupted:
            raise interrupted[0]
        if time.monotonic() - started >= timeout:
            raise TimeoutError("Connection exhausted the request deadline")
        transport.settimeout(max(0.1, timeout - (time.monotonic() - started)))
        check()
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if os.environ.get("SPARK_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["SPARK_API_KEY"]
        if interrupted:
            raise interrupted[0]
        if time.monotonic() - started >= timeout:
            raise TimeoutError("Request deadline exceeded before submission")
        conn.request("POST", urlunsplit(("", "", endpoint.path, "", "")),
                     body=json.dumps(payload).encode(), headers=headers)
        response = conn.getresponse()
        if response.status != 200:
            raise ProtocolError(f"HTTP {response.status}")
        if "text/event-stream" not in response.getheader("Content-Type", ""):
            raise ProtocolError("Expected an SSE response")
        result = StreamResult(started)

        def lines():
            size = 0
            while True:
                check()
                line = response.readline(MAX_LINE + 1)
                if interrupted:
                    raise interrupted[0]
                if not line:
                    return
                size += len(line)
                if size > MAX_RESPONSE:
                    raise ProtocolError("Response exceeds 16 MiB bound")
                yield line

        for event in sse_events(lines()):
            result.feed(event, time.monotonic())
            if result.done:
                break
        check()
        # Bounded speed samples intentionally allow the requested output cap.
        finish_reason = result.finish_reason
        if finish_reason == "length":
            result.finish_reason = "stop"
        finished = result.finish(time.monotonic())
        finished["finish_reason"] = finish_reason
        return finished
    except Exception:
        if interrupted:
            raise interrupted[0]
        raise
    finally:
        stop.set()
        conn.close()


def _summary(records, wall):
    valid = [r for r in records if r.get("passed")]
    metrics = ("ttft_seconds", "first_model_output_seconds", "elapsed_seconds",
               "completion_tokens_per_second_end_to_end", "post_first_output_tokens_per_second_estimate",
               "prefill_tokens_per_second_estimate")
    medians = {}
    for name in metrics:
        values = [r["metrics"][name] for r in valid if r["metrics"].get(name) is not None]
        medians[name] = round(statistics.median(values), 4) if values else None
    total = sum(r["usage"]["completion_tokens"] for r in valid)
    return {"successful_requests": len(valid), "failed_requests": len(records) - len(valid),
            "total_prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in valid),
            "total_completion_tokens": total, "p50": medians, "wall_seconds": round(wall, 4),
            "aggregate_completion_tokens_per_second_end_to_end": round(total / wall, 3)
            if wall > 0 and valid and len(valid) == len(records) else None}


def _prompt(size, nonce):
    # Approximate token target, never used as a measured token count. Fresh leading
    # salt avoids intentionally reusing the long prefix in a prefill sweep.
    return f"Synthetic benchmark {nonce}. Read these entries:" + " x" * size + (
        "\nWrite a clear explanation of how a web request reaches a database, "
        "including connections, caching and latency. Continue until the output limit.")


def run_benchmark(cfg, status_fn, *, node, kind="decode", concurrency=1, requests=3,
                  max_tokens=128, prompt_tokens=512, thinking=None, ssh_run=None, emit=None,
                  expected_allocation_id=None) -> int:
    """Run one bounded suite; status_fn() returns fresh collect_status(cfg)."""
    from spark_serve_controller import ControllerError
    if kind not in ("decode", "prefill"):
        raise ControllerError("benchmark kind must be decode or prefill")
    for name, value, low, high in (("concurrency", concurrency, 1, 4), ("requests", requests, 1, 12),
                                  ("max_tokens", max_tokens, 16, 512), ("prompt_tokens", prompt_tokens, 32, 32768)):
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ControllerError(f"benchmark {name} must be in {low}..{high}")
    if concurrency > requests:
        raise ControllerError("benchmark concurrency cannot exceed request count")
    if kind == "prefill" and concurrency != 1:
        raise ControllerError("prefill sweeps run one request at a time")
    if thinking is not None and not isinstance(thinking, bool):
        raise ControllerError("thinking must be true, false, or server default")
    emit = emit or _EventWriter()
    controller = _controller(cfg, ssh_run)
    root = controller.directory
    run_id = uuid.uuid4().hex
    lease_path = _leases(root) / (run_id + ".json")
    lease_lock = None
    lengths = ([prompt_tokens] * requests if kind == "decode" else
               [max(32, round(prompt_tokens * (i + 1) / requests)) for i in range(requests)])
    with controller.lock():
        identity = _identity(cfg, status_fn(), controller.node_states(), node)
        if expected_allocation_id is not None and identity["allocation"]["id"] != expected_allocation_id:
            raise ControllerError("Selected allocation changed before benchmark admission; select it again")
        model = cfg["models"][identity["model"]]
        if concurrency > int(model.get("max_num_seqs") or 4):
            raise ControllerError("benchmark concurrency exceeds the model scheduler limit")
        context = int(model.get("max_model_len") or model.get("hermes_context_length") or 0)
        # UTF-8 bytes conservatively bound these ASCII prompts; reserve additional
        # tokens for chat wrappers and output. Actual server usage is retained.
        upper = max(len(_prompt(size, run_id).encode()) for size in lengths) + max_tokens + 1024
        if not context or upper > context:
            raise ControllerError("benchmark synthetic prompt plus output exceeds the conservative model context bound")
        for path in _leases(root).glob("*.json"):
            try:
                lease = json.loads(path.read_text())
                if (lease.get("expires_at", 0) > time.time()
                        and set(lease["hosts"]).intersection(identity["hosts"])):
                    raise ControllerError("a benchmark already owns this allocation; cancel it or wait")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise ControllerError("unreadable benchmark lease; cancel the previous run") from exc
        _leases(root).mkdir(parents=True, exist_ok=True, mode=0o700)
        lease_lock = lease_path.with_suffix(".lock").open("a+")
        fcntl.flock(lease_lock, fcntl.LOCK_EX)
        try:
            _save(lease_path, {"id": run_id, "hosts": identity["hosts"], "revoked": False,
                               "expires_at": time.time() + DEADLINE_SECONDS + 5, "allocation": identity["allocation"]})
        except BaseException:
            lease_lock.close()
            raise
    run = {"id": run_id, "kind": kind, "status": "running", "started_at": _now(), **identity,
           "config": {"concurrency": concurrency, "requests": requests, "max_tokens": max_tokens,
                      "prompt_tokens": prompt_tokens, "prompt_lengths": lengths,
                      "thinking": thinking, "reasoning_mode": "server-default" if thinking is None else "requested-on" if thinking else "requested-off",
                      "deadline_seconds": DEADLINE_SECONDS, "prompt_token_target_is_approximate": True},
           "warmup": None, "results": [],
           "limitations": ["Synthetic speed sample, not a quality evaluation.",
                           "Token counts are server usage, never streamed chunk counts.",
                           "TTFT is first visible content; first model output includes reasoning.",
                           "Prefill rate uses prompt tokens / first model output latency, including overhead.",
                           "Post-first-output rate includes the first delta's unknown token count.",
                           "Thinking is requested through chat_template_kwargs; runtime behavior may differ."]}
    started = time.monotonic()
    deadline = started + DEADLINE_SECONDS
    ended = threading.Event()
    invalid = []
    completion_times = []
    output_failed = False

    def check():
        if invalid:
            raise BenchmarkCancelled(invalid[0])
        try:
            lease = json.loads(lease_path.read_text())
            active = not lease.get("revoked") and lease.get("id") == run_id
        except (OSError, ValueError):
            active = False
        if not active or not _owner_identity(root, identity):
            raise BenchmarkCancelled("Benchmark cancelled or allocation identity changed")
        if time.monotonic() >= deadline:
            raise TimeoutError("Overall benchmark deadline exceeded")

    def observe():
        while not ended.wait(IDENTITY_SECONDS):
            try:
                fresh = _controller(cfg, ssh_run)
                observed = status_fn()
                if ended.is_set():
                    return
                current = _identity(cfg, observed, fresh.node_states(), identity["node"])
                if current != identity:
                    raise ValueError("serving identity changed")
            except Exception:
                if ended.is_set():
                    return
                invalid.append("Serving identity changed or could not be verified")
                return

    def request(index, size, warmup=False):
        record = {"index": index, "phase": "warmup" if warmup else "measurement",
                  "requested_prompt_tokens": size}
        try:
            check()
            payload = {
                "model": identity["served_name"], "messages": [{"role": "user", "content": _prompt(size, uuid.uuid4().hex)}],
                "temperature": 0, "max_tokens": min(32, max_tokens) if warmup else max_tokens,
                "stream": True, "stream_options": {"include_usage": True},
            }
            if thinking is not None:
                payload["chat_template_kwargs"] = {"enable_thinking": thinking}
            response = _stream_request(identity["endpoint"], payload,
                                       min(REQUEST_SECONDS, deadline - time.monotonic()), check)
            first = response["metrics"].get("first_model_output_seconds")
            response["metrics"]["prefill_tokens_per_second_estimate"] = (
                round(response["usage"]["prompt_tokens"] / first, 3) if first and first > 0 else None)
            record.update(passed=True, usage=response["usage"], metrics=response["metrics"],
                          finish_reason=response["finish_reason"], reasoning_observed=bool(response.get("reasoning")))
        except Exception as exc:
            record.update(passed=False, cancelled=isinstance(exc, BenchmarkCancelled),
                          error=f"{type(exc).__name__}: {exc}")
        if not warmup:
            completion_times.append(time.monotonic())
        return record

    measurement_started = None
    try:
        emit({"event": "started", "run": run})
        threading.Thread(target=observe, daemon=True).start()
        run["warmup"] = request(-1, 32, warmup=True)
        if run["warmup"]["passed"]:
            measurement_started = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(request, index, size) for index, size in enumerate(lengths)]
                try:
                    for future in concurrent.futures.as_completed(futures):
                        run["results"].append(future.result())
                        emit({"event": "progress", "completed": len(run["results"]), "total": requests,
                              "request": run["results"][-1]})
                except BaseException:
                    invalid.append("Benchmark interrupted")
                    raise
        rows = [run["warmup"], *run["results"]]
        run["status"] = ("cancelled" if any(r.get("cancelled") for r in rows) else
                         "completed" if len(run["results"]) == requests and all(r["passed"] for r in rows) else "failed")
    except (KeyboardInterrupt, BenchmarkCancelled) as exc:
        run.update(status="cancelled", error=str(exc) or "Interrupted")
    except Exception as exc:
        run.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        ended.set()
        run["finished_at"] = _now()
        run["elapsed_seconds"] = round(time.monotonic() - started, 4)
        run["results"].sort(key=lambda r: r["index"])
        measured_wall = max(completion_times) - measurement_started if completion_times and measurement_started is not None else 0
        run["summary"] = _summary(run["results"], measured_wall)
        try:
            _save(root / "benchmarks" / (run_id + ".json"), run)
        finally:
            # Unique run markers are never reused; no controller lock is needed
            # to acknowledge cancellation while lifecycle holds that lock.
            fcntl.flock(lease_lock, fcntl.LOCK_UN)
            lease_lock.close()
            lease_path.unlink(missing_ok=True)
            lease_path.with_suffix(".lock").unlink(missing_ok=True)
        try:
            emit({"event": "result", "run": run})
        except (OSError, KeyboardInterrupt):
            # History and lease cleanup already completed. Never bounce a broken
            # output channel into the CLI error path, which writes to stdout too.
            output_failed = True
    return 0 if run["status"] == "completed" and not output_failed else 1
