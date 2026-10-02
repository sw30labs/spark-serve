# ADR 0012: Power off both Sparks through the fenced CLI

Accepted · Recorded 2026-10-02

**Context.** Stop both Sparks only stops managed workloads. Restart Sparks
drains, reboots both hosts with `systemctl reboot --no-block`, and waits for
SSH to return. Operators also need to power the machines off without a second
controller.

**Decision.** Add `spark-serve shutdown` and a confirmed Shutdown both Sparks
action in the app. Use the same serialized drain as reboot, including explicit
YuE cancellation and benchmark-lease revocation, then run
`/usr/bin/systemctl poweroff --no-block` on the worker and then the head.
Authenticate with the existing passwordless systemctl probe, or a sudo password
read from stdin for that command only. Wait until SSH drops. Do not wait for
the hosts to return and do not start serving again. Do not call a generic
`shutdown` or `reboot` binary, and do not change sudoers.

**Trade-off.** A powered-off pair stays off until someone presses the power
buttons. `--no-wait` returns after the power-off command is accepted, so a host
can still be up. Failed authentication happens before any drain.

**Evidence.** [CLI](../../spark-serve), [app](../../gui/SparkServeApp.swift),
[tests](../../tests/test_reboot.py), [lifecycle](0003-fenced-lifecycle.md),
[leases](0008-benchmark-leases.md).
