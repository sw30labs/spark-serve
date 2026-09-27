# ADR 0005: Admit YuE workers; leave jobs to Artist Twin

Accepted · Recorded 2026-09-26 (retrospective)

**Context.** Song generation needs durable job handling without duplicating that
system inside the workload controller.

**Decision.** Run independent protocol-v2 YuE workers. Spark Serve validates
worker identity, generation, pinned assets, CUDA, and admission, then publishes
an atomic public discovery file. Artist Twin owns jobs, dispatch, downloads,
rights checks, and provenance. Workers start drained; HTTP success alone never
establishes readiness. Active renders drain naturally unless cancellation is explicit.

**Trade-off.** Two workers increase parallel capacity, not the speed of one song.
The current shared-generation discovery contract requires starting both workers
together. A scoped stop can preserve the peer; independently admitting one
worker requires a contract change.

**Evidence.** [Admission and discovery](../../spark_serve_controller.py),
[ownership tests](../../tests/test_node_controller.py),
[worker setup](../../README.md#install-the-yue-workers).
