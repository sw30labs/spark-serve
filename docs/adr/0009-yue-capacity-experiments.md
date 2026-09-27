# ADR 0009: Isolate YuE capacity experiments

Accepted · Recorded 2026-09-26 (retrospective)

**Context.** Production YuE admits one job per worker. Sending concurrent jobs
there measures admission rejection, not hardware capacity.

**Decision.** Use a separate dispatcher with isolated directories and benchmark
container identities. Drain production and hold the node lock throughout. Mac
`--ssh-host` mode also holds the controller lock through execution and restoration.
Reuse the deployed runtime and seeded workload. Count integrity-passing jobs and
audio duration per wall-clock hour, including failures and cleanup; apply resource guards.
Restore admission only after verified cleanup.

**Trade-off.** The experiment blocks lifecycle changes and can leave production
drained after ambiguous failure. Results describe independent inference containers,
not shared-model batching. This differs from the revocable leases used by
[live inference benchmarks](0008-benchmark-leases.md).

**Evidence.** [Remote orchestration](../../spark_bench/remote.py),
[node lease](../../spark_bench/runtime.py), [failure tests](../../tests/test_bench_failures.py),
[experiment guide](../worker-concurrency-benchmark.md).
