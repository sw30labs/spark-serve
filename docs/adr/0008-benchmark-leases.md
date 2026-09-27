# ADR 0008: Revocable benchmark leases over physical hosts

Accepted · Recorded 2026-09-26 (retrospective)

**Context.** A benchmark must target the selected allocation and cooperate with workload transitions, including models spanning both Sparks.

**Decision.** Admit runs under the existing controller lock after verifying readiness, allocation identity, and immutable container IDs. The app pins its selected allocation. Each run leases every participating physical host with a unique local marker and held file lock; requests run without holding the controller lock. Stop, switch, and reboot revoke overlapping leases and await client acknowledgement before changing workloads, failing closed if acknowledgement times out. Request watchers interrupt transport on revocation, ownership changes, or deadlines. Completion releases the lease and retains available results, including failures and cancellation.

**Trade-off.** Coordination applies to clients sharing the local controller state. Closing requests does not prove the engine has reclaimed GPU work; lifecycle idle checks remain authoritative.

**Evidence.** [Lease protocol](../../spark_serve_benchmarks.py), [controller transitions](../../spark_serve_controller.py), [tests](../../tests/test_native_benchmarks.py), [operator guide](../native-observability.md).
