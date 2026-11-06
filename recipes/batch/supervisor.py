#!/usr/bin/env python3
"""spark-serve batch supervisor (ADR 0018). Runs on the batch job's Spark.

Started detached (setsid) by `spark-serve up <batch entry>` after spark-serve
stopped the previous model. It runs one job, then ALWAYS tries to bring the
previous model back with the job directory's own spark-serve snapshot, so the
restore never depends on the Mac being online.

Usage: supervisor.py JOB_DIR   (JOB_DIR holds job.json; state goes to state.json)
Stop:  touch JOB_DIR/cancel and SIGTERM the supervisor (spark-serve batch stop).
Stdlib only; Python 3.8+.
"""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

GRACE_SECONDS = 30
POLL_SECONDS = 2


class Supervisor:
    def __init__(self, job_dir, clock=time.time, sleep=time.sleep, popen=subprocess.Popen, run=subprocess.run):
        self.dir = os.path.abspath(os.path.expanduser(job_dir))
        with open(os.path.join(self.dir, "job.json")) as f:
            self.job = json.load(f)
        self.clock, self.sleep, self.popen, self.run_cmd = clock, sleep, popen, run
        self.state = {"id": self.job["id"], "model": self.job["model"], "host": self.job.get("host"),
                      "output": self.job["output"], "previous": (self.job.get("restore") or {}).get("model"),
                      "phase": "starting", "result": None, "restore": "pending" if self.job.get("restore") else "none",
                      "supervisor_pid": os.getpid(), "created_at": self.clock()}
        self.cancelled = False

    # -- state -------------------------------------------------------------
    def save(self, **fields):
        self.state.update(fields, updated_at=self.clock())
        fd, tmp = tempfile.mkstemp(prefix=".state.", dir=self.dir)
        with os.fdopen(fd, "w") as out:
            json.dump(self.state, out, indent=2)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, os.path.join(self.dir, "state.json"))

    def cancel_requested(self):
        return self.cancelled or os.path.exists(os.path.join(self.dir, "cancel"))

    def on_term(self, *_):
        self.cancelled = True

    # -- job ---------------------------------------------------------------
    def env(self):
        env = dict(os.environ)
        for key in self.job.get("unset_env") or []:
            env.pop(key, None)
        for key, value in (self.job.get("env") or {}).items():
            env[str(key)] = os.path.expanduser(str(value))
        prepend = [os.path.expanduser(p) for p in self.job.get("path_prepend") or []]
        if prepend:
            env["PATH"] = os.pathsep.join(prepend + [env.get("PATH", "")])
        return env

    def gpu_busy(self):
        if not self.job.get("require_free_gpu", True):
            return None
        try:
            proc = self.run_cmd(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                                 "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"cannot check the GPU: {exc}"
        if proc.returncode:
            return f"cannot check the GPU: nvidia-smi exit {proc.returncode}"
        apps = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        return ("GPU is not free: " + "; ".join(apps)[:400]) if apps else None

    def run_job(self):
        busy = self.gpu_busy()
        if busy:
            self.save(phase="restoring" if self.job.get("restore") else "finished", result="failed", error=busy, ended_at=self.clock())
            return
        if self.cancel_requested():
            self.save(result="cancelled", ended_at=self.clock())
            return
        argv = [os.path.expanduser(str(a)) for a in self.job["argv"]]
        workdir = os.path.expanduser(self.job.get("workdir") or "~")
        os.makedirs(os.path.dirname(os.path.expanduser(self.job["output"])), exist_ok=True)
        with open(os.path.join(self.dir, "job.log"), "ab") as log:
            proc = self.popen(argv, cwd=workdir, env=self.env(), stdout=log, stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL, start_new_session=True)
        self.save(phase="running", job_pid=proc.pid, started_at=self.clock())
        deadline = self.clock() + 60 * float(self.job.get("timeout_minutes") or 120)
        killed_at, reason = None, None
        while proc.poll() is None:
            if killed_at is None and (self.cancel_requested() or self.clock() >= deadline):
                reason = "cancelled" if self.cancel_requested() else "timeout"
                self.save(phase="stopping")
                self._signal(proc, signal.SIGTERM)
                killed_at = self.clock()
            elif killed_at is not None and self.clock() - killed_at >= GRACE_SECONDS:
                self._signal(proc, signal.SIGKILL)
            self.sleep(POLL_SECONDS)
        code = proc.returncode
        output = os.path.expanduser(self.job["output"])
        size = os.path.getsize(output) if os.path.isfile(output) else 0
        if reason == "cancelled":
            result, error = "cancelled", "stopped by request"
        elif reason == "timeout":
            result, error = "failed", f"timed out after {self.job.get('timeout_minutes') or 120} min"
        elif code == 0 and size > 0:
            result, error = "done", None
        else:
            result, error = "failed", f"job exit {code}" + ("" if size else ", no output file") + self._tail()
        self.save(result=result, job_exit=code, output_bytes=size, error=error, ended_at=self.clock())

    def _signal(self, proc, sig):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def _tail(self):
        try:
            with open(os.path.join(self.dir, "job.log"), "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 600))
                text = f.read().decode("utf-8", "replace").strip().splitlines()
            return (": " + text[-1][:300]) if text else ""
        except OSError:
            return ""

    # -- restore -----------------------------------------------------------
    def restore(self):
        spec = self.job.get("restore")
        if not spec:
            self.save(phase="finished", restore="none")
            return
        attempts = int(spec.get("attempts") or 3)
        env = dict(os.environ)
        env.update({str(k): os.path.expanduser(str(v)) for k, v in (spec.get("env") or {}).items()})
        argv = [os.path.expanduser(str(a)) for a in spec["argv"]]
        self.save(phase="restoring", restore="running", restore_started_at=self.clock())
        last = None
        for attempt in range(1, attempts + 1):
            with open(os.path.join(self.dir, "restore.log"), "ab") as log:
                log.write(f"\n== restore attempt {attempt} at {time.ctime(self.clock())}\n".encode())
                log.flush()
                try:
                    proc = self.run_cmd(argv, cwd=os.path.expanduser(spec.get("cwd") or self.dir), env=env,
                                        stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        timeout=int(spec.get("timeout_seconds") or 3600))
                    last = proc.returncode
                except (OSError, subprocess.TimeoutExpired) as exc:
                    last = f"{type(exc).__name__}: {exc}"
            self.save(restore_attempts=attempt, restore_exit=last)
            if last == 0:
                self.save(phase="finished", restore="ok", restore_ended_at=self.clock())
                return
            if attempt < attempts:
                self.sleep(float(spec.get("retry_seconds") or 60))
        self.save(phase="finished", restore="failed", restore_ended_at=self.clock(),
                  restore_error=f"restore of {spec.get('model')} failed after {attempts} attempts (last: {last}); see restore.log")

    def main(self):
        signal.signal(signal.SIGTERM, self.on_term)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        self.save()
        try:
            self.run_job()
        except Exception as exc:  # noqa: BLE001 - whatever happens, the restore still runs
            self.save(result="failed", error=f"supervisor error: {type(exc).__name__}: {exc}", ended_at=self.clock())
        finally:
            # Ignore further TERMs: a Stop during restore must not leave the Sparks empty.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            self.restore()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: supervisor.py JOB_DIR")
    Supervisor(sys.argv[1]).main()
