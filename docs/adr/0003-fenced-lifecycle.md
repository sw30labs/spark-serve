# ADR 0003: Serialize, fence, and drain transitions

Accepted · Recorded 2026-09-26 (retrospective)

**Context.** Concurrent commands, delayed SSH execution, and interrupted starts
can leave workloads competing for the same GPU.

**Decision.** Serialize transitions under the Mac controller lock and persist
state atomically. Remote generation locks fence stale commands and remain held
through their side effects. Revoke overlapping benchmark leases and discovery,
drain active work, stop exact owned container IDs, verify idle, then launch.
Protect unrelated containers. Require identity and readiness evidence; save
startup diagnostics before guarded cleanup. Retain failed state for explicit retry.

**Trade-off.** Unknown ownership or an unreachable participating host blocks the
transition. Active YuE jobs survive unless cancellation is explicit. Safety takes
precedence over best-effort switching or automatic rollback.

**Evidence.** [Controller](../../spark_serve_controller.py),
[controller tests](../../tests/test_controller.py), [startup tests](../../tests/test_startup.py),
[benchmark leases](0008-benchmark-leases.md).
