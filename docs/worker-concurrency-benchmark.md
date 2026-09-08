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

## Background and hypotheses

H0: one active render maximizes useful throughput within the tested workload
and hardware envelope. H1: two or more active renders materially improve
integrity-passing songs/hour without unacceptable reliability or resource cost.
Failure to reject H0 is not proof for every future model, workload or driver.
The current C1 ownership policy is the baseline, not an empirical conclusion.

## Experimental controls

`python -m spark_bench concurrency` runs a finite closed-loop workload. It fills
up to C slots and replaces completed work until the identical seeded corpus is
finished. Each level gets the same job count, workload order, token caps, lyric
text, references, seeds and pinned runtime. The default corpus includes a short
single-section request and a representative three-section continuation. The
runtime's minimum `run_n_segments=2` is a cap; it does not invent an extra section
for the short request. Default stage-2 batch size remains 2.

Every job includes its usual fresh process/model load. An excluded warmup heats
the shared file cache and exercises both inference stages before measurements.
No persistent model or CUDA cache is invented for one level only. Measured
rounds use ascending then descending order (rotating later rounds). Cooldowns
are explicit and excluded from each level's makespan. No network, NCCL, power,
clock, fan or CUDA MPS settings change.

The node-local dispatcher drains the factory, waits for current production jobs
without cancelling them, holds the real `node.lock`, audits idle GPU ownership,
and verifies pinned assets. Existing HTTP monitoring remains available but
production admission is closed. Benchmark containers have distinct labels;
they never enter the production job database. The Mac `--ssh-host` mode also
holds Spark Serve's `controller.lock` for deployment, execution, restoration and
collection, so model transitions cannot race the experiment. It tests one
configured physical node; the other node can finish its current work.

Each job has an isolated `/work` directory; model/cache/source mounts are
read-only. Cleanup verifies immutable Docker ID, exact name, run label and job
label. An ambiguous outcome leaves production drained. Normal completion
re-audits the GPU and re-admits the original generation, verifying its receipt.
The process returns nonzero for an incomplete sweep or failed restoration.

## Metrics and telemetry

The denominator is the full measured trial makespan, including load, failures,
validation and container cleanup. Success requires passing the existing YuE
worker diagnostics and final WAV validation; exit code zero alone is insufficient.
We report successes/hour, successful generated audio seconds/hour, counts by
failure class, retry count, and mean/median/p95/min/max successful request
latency. P95 with small samples is descriptive, not a precise tail estimate.

Elapsed durations and timeouts use monotonic clocks. Epoch times permit log
correlation. Actual inference overlap is calculated from container execution
intervals separately from request/future overlap, so queued futures cannot
pretend to prove C active generations.

`telemetry.jsonl` and `.csv` contain time-series GPU utilization, memory-busy
percentage, available GPU memory counters, temperature, clocks, power and
throttle flags; total/per-core CPU and process CPU/RSS; unified memory and swap;
Linux pressure metrics; per-interface network and disk rates; and available
system thermal sensors. First delta samples and unsupported counters are null.
On GB10, `nvidia-smi memory.used` may be unavailable: `/proc/meminfo` is the node
memory source. GPU memory-busy percentage is **not measured bandwidth**. SM
occupancy, tensor activity and achieved bytes/second need a profiler/DCGM counter
that this portable collector does not claim to expose. Network rates measure
whether I/O matters; they do not change the network topology.

Guards stop escalation on OOM (including CUDA allocation OOM without a Docker
OOMKilled flag), timeout, loss of durable outcomes/telemetry, low available
memory, swap growth or excessive GPU temperature. Defaults are 12 GiB free,
1 GiB swap growth, and 85°C maximum; configure them explicitly for a host.
These are experimental stop thresholds, not hardware specification claims.
The independent `--min-temperature-margin-c 5` guard also stops when available
driver-reported T.Limit headroom reaches 5°C or less. T.Limit is separate from
core temperature; it must not be added to core temperature to infer an absolute
limit. If the driver cannot report T.Limit, that optional guard is unavailable
and the absolute-temperature and other guards still apply. The planned live
sweep explicitly uses `--max-temperature-c 90 --min-temperature-margin-c 5`;
the committed absolute-temperature default remains 85°C. An early guard stop
leaves the sweep incomplete, with no H0 verdict and only provisional capacity
candidates; skipped higher levels are not evidence of an optimum.
Only benchmark-owned containers are cancelled. Failed WAVs, code arrays and
logs are retained. There is no automatic retry or silence trimming.

## How to run

Use Python 3.11+ (3.12 recommended) with Docker/NVIDIA runtime on the Spark. The
existing deployed Artist Twin factory, pinned assets and integrity code must
already be admitted by Spark Serve. No additional Python packages are required
by the benchmark; the existing inference container provides its own dependencies.

Copy `tests/workloads/yue-standard.json` to a private local file. Set `attestation`,
`vocal`, and `instrumental` to authorized local files. References must be distinct,
aligned 30-second 44.1 kHz mono PCM16 tracks. The provided lyric text is an
original synthetic workload; replace/add workloads when assessing another genre
or longer production songs. Paths and text are frozen before the experiment.
Keep the identical corpus/settings across every level; do not “repair” a failing
workload between C1 and C2.

From the Mac checkout, using a host in the private `models.toml`:

```sh
python3 -m spark_bench concurrency \
  --ssh-host spark-head --config models.toml \
  --workload /path/to/private/yue-workload.json \
  --concurrency 1,2,3,4 --jobs-per-level 8 --iterations 2 --warmup 1 \
  --telemetry-interval 2 --cooldown 60 --timeout 14400 --wait-idle 14400 \
  --output benchmark-results/spark-head-first-sweep
```

The chosen endpoint is derived from the configured node. `--worker` optionally
asserts the factory's durable identity. The endpoint is recorded for provenance;
the isolated benchmark executes the same container runtime locally on that node,
not the C1 HTTP admission path. Thus the benchmark measures inference-container
capacity including lifecycle overhead, excluding reference upload/download time.
This is the intended first single-node capacity experiment, not an HTTP load test.

For a node-local run, copy the `spark_bench/` package and
`spark_serve_controller.py` to the Spark; omit `--ssh-host` and run the same command
there. This mode holds the node lock but not the Mac controller lock, so prefer
the Mac wrapper during normal service operation. Run the other Spark as a
separate experiment/output root afterward; do not merge unlike nodes blindly.

Validate the harness locally without loading models:

```sh
python3 -m spark_bench concurrency --synthetic \
  --workload tests/workloads/yue-standard.json \
  --concurrency 1,2,3,4 --jobs-per-level 4 --iterations 2 \
  --warmup 1 --cooldown 0 --telemetry-interval 0.05 \
  --output benchmark-results/synthetic-check
python3 -m spark_bench analyze benchmark-results/synthetic-check
python3 -m pytest
```

Synthetic timing is labeled explicitly and can never support a hardware capacity
conclusion. Outputs must be new directories; overwriting evidence is refused.
After interruption, inspect `remote-run.json`, `lease.json`, container identity
receipts and exact Docker IDs before cleanup. Do not remove containers by a
name prefix or re-admit production while cleanup is unverified. A new experiment
uses a new output directory; partial trials are not silently resumed as warm data.

## Raw-data format

```text
run-metadata.json                 # corpus, levels, order, seed, runtime pins, guards
inputs/workload.json              # frozen lyrics/settings and local reference paths
inputs/reference-hashes.json      # byte identities; no base64 in logs
lease.json / lease-restored.json  # node ownership/restoration receipts
gpu-idle-audit.json
trials/r01-c01/
    trial.json                   # expected jobs, measured flag, monotonic makespan
    jobs.jsonl                   # one outcome per job, including failures/not-launched
    telemetry.jsonl / telemetry.csv
    jobs/<job-id>/
        job.json / container.json / generation_request.json
        worker.log / output.wav
        worker-diagnostics.json / timeline-events.jsonl / timeline-sections.json
        out/                     # all retained codec/reconstruction/vocoder evidence
comparison.csv / summary.json / report.md
```

The Mac wrapper stores `remote-run.json`, `driver.log` and collected compact
records in `results/`. Large WAV/token evidence remains on the Spark at the
recorded path. Logs contain no credentials, full process command lines or binary
reference data. Analysis can be rerun from the compact records at any time.

Resource summaries report the minimum signed `temperature_tlimit_c` headroom
separately from absolute core temperature. T.Limit is driver-reported remaining
thermal margin; adding it to the core temperature does not establish an absolute
safe temperature limit.

Where supported, cumulative clock-event counters expose microseconds spent under
software power caps, software/hardware thermal slowdown, hardware power braking,
and synchronization boost. Analysis recomputes increments within each measured
trial for consecutive observations of the same host/GPU UUID. It never charges
the first driver-lifetime total, a sampler-provided first delta, or activity in
the gap between trials. Counter decreases, clock restarts, missing observations,
and GPU identity changes break continuity; only observed valid intervals are
summed. Partial sums are lower bounds, and missing counters remain unavailable.
A driver reset that leaves no visible counter decrease cannot be inferred from
these samples. These event categories may overlap and must not be added together
as total throttled wall time. Counter increases can reveal throttling between
samples even when each sampled instantaneous event flag is inactive. These
advanced counters and headroom remain optional evidence, without changing the
required basic-resource telemetry gate.

## Analysis and capacity recommendation methodology

Comparisons match the exact workload-ID/seed multiset and dispatcher topology to
their C1 baseline. Speedup is throughput(C)/throughput(1); efficiency is
speedup/C; marginal gain is reported between tested levels. The report contains
per-workload results so aggregate mixes cannot hide a short-song substitution.
Matched-seed audio shortening greater than 20% requires review and prevents an
unqualified recommendation; it is a structural heuristic, not musical scoring.

At least two complete independent measured trials per requested level, two
workload types, matched C1, real inference overlap and nonsynthetic records are
required for a nonprovisional inference. Missing levels, censored outcomes,
infrastructure failures, lost telemetry and incomplete runs remain visible.
The conservative default candidate needs >=10% aggregate speedup, no failures,
and p95 latency no worse than 3x baseline. Missing resource evidence prevents a
strong bottleneck claim. Thresholds for analysis can be supplied to `analyze()`;
all chosen thresholds are saved in the report.

Treat the recommendation as candidate **concurrent model processes per Spark**.
One experimental dispatcher currently schedules these processes. A future
multi-job factory may expose `capacity`, `active_jobs`, and `available_slots`,
but its durable ownership/recovery and API must be updated and tested separately
before production can use that capacity. Running several copies of the current
factory is not a valid experiment: they conflict through global production
container ownership. There is no existing shared-model request server to test.

If C4 still scales materially without reliability/resource collapse, extend the
same sweep explicitly, for example `--concurrency 1,2,3,4,5,6 --jobs-per-level 12`.
The harness does not impose four as a ceiling. It also does not automatically
raise load after an OOM or guard stop. Preserve the first sweep, inspect the
telemetry, then launch the extended controlled experiment.

## Definition of Done and current evidence status

The engineering harness is complete when deterministic tests and a synthetic
full sweep pass, with durable failures and correct ownership handling. The
capacity experiment is complete only after real C1/C2/C3/C4 data supports the
throughput/latency/integrity/resource comparison and a scoped recommendation.
Architecture alone, free memory, or 99% utilization cannot settle H0. No real
throughput number or optimal capacity has yet been established in this document.
Measured runs and their final decision report belong in the output directory;
production remains at capacity 1 until separately authorized and implemented.

## First live attempt and thermal-guard adjustment

The first Sparkone attempt, `b20260908172750-309fa2` on 2026-09-08,
stopped its unmeasured C1 warmup after 581.4 seconds when the configured 5°C
minimum T.Limit margin was reached. Its 291 telemetry samples contained no
collection errors. The lowest observed margin was 4°C at a core temperature
of 81°C; maximum observed core temperature was 84°C. All five recorded
cumulative clock-event counters were unchanged across the attempt. Stage1
produced an aligned 17.38-second section and Stage2 had started when the guard
cancelled the job. No completed song or measured trial resulted: this attempt
is **not assessed**, with no baseline throughput or H0 conclusion. The exact
container was removed and both production factories were verified admitted
and idle afterward. Evidence is retained under the attempt's output directory
and `preflight/launch-verification.json` in the local live-run collection.

A separate follow-up attempt can change only the explicit experiment guard
from `--min-temperature-margin-c 5` to `--min-temperature-margin-c 0`, retaining
the same corpus, references, seeds, model settings, trial order, job counts,
warmup, 2-second telemetry interval, 90°C absolute-temperature guard, 12 GiB
available-memory reserve, and 1 GiB swap-growth guard. This adjustment tests
whether the positive-headroom stop prevented useful measurement before any
observed throttling. NVIDIA defines a T.Limit reading of zero or below as a
point where thermal conditions may cause clock optimization; it is not a
shutdown threshold. The guard adjustment follows that documented boundary,
without changing device clocks, power limits, cooling, or production admission
capacity. See the [NVIDIA temperature documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html#temperature).

The follow-up must use a new output directory and record its changed guard.
It must still abort at zero or negative headroom, preserve any stopped output,
and report measured throttle-counter increments. A zero-margin guard provides
less advance headroom, and sampling cannot prevent a brief boundary crossing.
If that attempt also stops, it remains incomplete; do not relax guards silently
or combine the two attempts into a completed throughput comparison.
