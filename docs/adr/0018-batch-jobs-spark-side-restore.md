# ADR 0018: Run batch entries with a Spark-side restore

Accepted · Recorded 2026-11-05 · Supersedes [ADR 0017](0017-batch-card-only-entries.md) for runnable batch entries

**Context.** ADR 0017 showed Pika H3 Flex as a card only. Pedro then asked for
Start to actually run it: stop Qwen, render on sparktwo, bring Qwen back
(2026-11-05). A render takes about 20 minutes with the model stopped on both
Sparks, and the Mac running the app drops offline at times, so a restore
driven from the Mac can be lost. The same evening a spark-serve stop took Qwen
down at 20:22 ET, and a Pika Start in the pre-0017 app then failed with
`unknown model wrapper 'batch'`. That failure came before any switch, but
nothing restored Qwen until 20:32 ET. Validate-before-stop is therefore part
of this decision, made explicit in the Controller as well as in `up`.

**Decision.**
- **Validate before stop, everywhere.** `Controller.switch` refuses any target
  whose wrapper it cannot start as a server (`SERVER_WRAPPERS`) before it
  takes the lock, fences or drains, and `up` checks the same before placement.
  A batch `up` runs all its checks before the stop: options against the
  entry's `[batch]` table; no other batch job; a current allocation it can restore
  exactly (one spark-serve model, ready; YuE or mixed modes are refused); the job upload; and a
  read-only preflight on the home Spark. The preflight checks Python 3.11+, the
  required paths, a writable output folder with 2 GB free, no running
  supervisor, SSH to the peer Spark with the job's own config, and that the
  spark-serve snapshot runs. A failure there removes the job directory and
  leaves the model serving.
- **Stop through the normal path.** The stop is `Controller.switch("none")`,
  so fencing, drain and the idle audit are unchanged. If the job cannot start
  after the stop, the Mac restores the previous model at once.
- **Detached supervisor on the Spark.** `recipes/batch/supervisor.py` starts
  with `setsid -f nohup` in `~/.local/state/spark-serve-batch/<id>/`. It
  checks that the GPU has no compute processes and runs the entry's command
  (placeholders filled, offline env, `HF_TOKEN` unset), writing
  `<outputs>/<id>.mp4`. It keeps `state.json` (running, stopping, restoring,
  finished; result done, failed or cancelled; restore ok, failed or none).
  `batch stop` touches `cancel` and sends SIGTERM. The job's process group
  gets TERM, then KILL after 30 s, and a 120-minute timeout does the same.
- **Restore on the Spark, always.** Whatever ends the job, the supervisor runs
  `python3 <job>/app/spark-serve up <previous> --node <node> --no-hermes`
  from a snapshot of the Mac's code and `models.toml`. It tries 3 times,
  60 s apart. The snapshot uses a job-scoped `ssh_config` and `known_hosts`
  built from the Mac's resolved HostName and the host keys the Mac already
  trusts, so no `~/.ssh` is changed. `SPARK_SERVE_LOCAL_HOSTS` runs the home
  Spark's own commands locally, because it cannot SSH to itself.
  `SPARK_SERVE_STATE_DIR` points at the job's controller copy, seeded with
  the Mac's state after the stop.
- **The Mac catches up.** `status --json` and `batch status` read the job
  back. Once it is finished, they copy the Spark-side controller state
  (new generation and container receipts) into the Mac's
  `controller.json`, so later switches stay fenced. Until then, `up`, `stop`,
  `reboot` and `shutdown` refuse, even if the Spark is unreachable.
- **App.** A batch card shows a Run form (prompt, resolution, aspect,
  duration, seed). Run asks for confirmation, naming what gets stopped. A
  status row shows the job phase, output path and errors, with a Stop job
  button. Per-Spark Start stays off for batch entries, and the menu bar
  leaves them out.
- `launch = false` still means card-only (ADR 0017). A batch entry without a
  `[batch]` table with a `command` is card-only too. `pull` refuses batch entries.

**Trade-off.** The Spark runs a copy of spark-serve and so needs Python 3.11+
and SSH to its peer. Both are checked before anything stops. While the job
runs, the Mac shows the Sparks stopped and relies on the job record.
The supervisor survives SSH loss and Mac sleep but not a Spark reboot or a
SIGKILL of itself; then `batch status` shows the last phase and Pedro
restores by hand. Restoring is limited to one ready spark-serve model; YuE is
refused rather than restored approximately.

**Evidence.** `tests/test_batch_launch.py` covers the order upload → preflight
→ stop → launch, and that nothing stops on bad options, an unrestorable state
or a failed upload or preflight. It also covers the Mac restore after a failed
launch, the guards (including an unreachable Spark), adoption, `batch stop`, the
Controller guard, and the supervisor's done, failed, cancelled, busy-GPU and
failed-restore paths. On 2026-11-05 a live check on sparktwo uploaded a real
Pika job and passed the preflight. It then ran the supervisor with a 3 s stand-in
job, and the "restore" step was the read-only `status --json` through the same
SSH setup. It saw Qwen ready on both Sparks, and nothing was stopped. No Pika
render was run as a test.
