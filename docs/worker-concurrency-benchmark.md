# YuE worker concurrency experiment

## Architecture reconnaissance

Spark Serve is a Mac CLI/SwiftUI workload controller, not the YuE inference
server. `spark_serve_controller.py` admits one protocol-v2 Artist Twin factory
on each of two distinct physical Sparks and publishes their identities. Artist
Twin sends immutable jobs directly to those advertised HTTP endpoints.

Terminology in this implementation:

| Term | Actual implementation |
|---|---|
| Spark/node | One physical DGX Spark and its GB10 GPU/unified memory |
| Factory/worker | One stdlib HTTP process and durable job database per node |
| Request/job | One immutable song/take, identified by job and attempt IDs |
| Inference process | A fresh Docker container and Python process for each job |
| Queue slot | Artist Twin's dispatch reservation, separate from model batching |
| Stage-2 batch | Codec refinement within one song; not request concurrency |

The factory is deployed from Artist Twin's `scripts/spark/yue_factory.py`.
`JobStore.active()` rejects more than one nonterminal database row;
`JobStore.accept()` rejects another request while a job or probe is active;
`Worker.tick()` advances that single durable job. `Worker.ownership()` reconciles
all production-labeled YuE containers and fails closed on an untracked one.
Artist Twin additionally limits remote dispatch to two jobs and reserves each
advertised worker once. Spark Serve exposes no production concurrency setting.

`Worker._launch()` creates a fresh GPU container, mounts pinned weights/source
read-only, and executes `/factory/run_infer.py`. The wrapper invokes the pinned
upstream inference script once. `cuda_idx=0` and `stage2_batch_size=2` are fixed
in the deployed launch path. The HTTP process itself does not own model weights.
Concurrent inference containers therefore own independent model allocations,
CUDA contexts, and workspaces. Shared read-only weight files do not imply shared
GPU allocations. A persistent shared-model concurrent-call API does not exist.

These controls establish safe ownership and recovery, but they do not establish
the throughput optimum. Removing them would break reconciliation; simply
sending simultaneous requests to the production endpoint would measure its
admission rejections rather than GPU capacity.

## Experimental boundary

The experiment will use a separate dispatcher with isolated job directories and
benchmark-specific container identities. It will reuse the deployed runtime,
image digest, model settings, and integrity validation. The production factory
will be drained, active jobs allowed to finish, and node ownership held for the
experiment. No production capacity or network configuration will change.

Here concurrency means simultaneously running independent inference containers
on one physical Spark. This is the process topology that a future multi-job
factory would schedule. Multiple resident shared-model workers are not a
supported architecture to benchmark without a separate implementation.

Initial levels are 1, 2, 3, and 4, with a configurable ceiling. Identical seeded
short and representative workloads, equal per-level job counts, excluded
warmups, and rotated/reversed level order control input and cache/order effects.
The principal outcome is integrity-passing jobs per wall-clock hour; latency,
actual overlap, generated audio duration, failures, memory, resource telemetry,
temperature, and power are reported alongside it. GPU utilization alone is not
an acceptance criterion. Missing platform counters remain explicitly unavailable.

## Current evidence status

Architecture inspection is complete. No capacity conclusion or benchmark
throughput has been measured yet. Two previously requested VYRMA jobs are
currently using the nodes and must not be cancelled or overlapped by this test.

Detailed commands, raw-data schema, analysis, and measured findings follow with
the benchmark implementation and collected results.
