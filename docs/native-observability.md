# Native monitoring and benchmarks

Spark Serve keeps model control, monitoring, and benchmarks in the same macOS
app. The **Overview** shows physical Spark resources and logical model
allocations. **Models** retains placement, start/stop, and Hermes selection.
**Benchmarks** exercises an already-serving allocation and stores its results
locally. Selecting Hermes does not select or move the monitored workload.

Build and open the native app on an Apple Silicon Mac running **macOS 13 or
newer**, with Command Line Tools installed. Quit any running copy before
rebuilding, then run these commands from the repository root:

```sh
make -f gui/Makefile
open gui/SparkServeApp.app
```

Monitoring starts automatically in Overview. Use Models to manage workloads,
then choose a ready allocation in Benchmarks to run a decode test or prefill
sweep. The app uses `watch` and `bench` from the same Python CLI described below.
The [README screenshots](../README.md#screenshots) show the current interface
with synthetic demo data captured September 26, 2026, not hardware performance
measurements.

## Live metrics

```sh
./spark-serve watch --json
./spark-serve watch --json --interval 3
```

One app-owned CLI process streams versioned JSON-line snapshots. Each host has
an independent, temporary SSH sampler using the existing Python telemetry
collector. Nothing is installed on the Sparks. The sampler needs Python 3.11+
and reads Linux counters and `nvidia-smi`. Disconnects reconnect with backoff;
one offline host does not stop the other host's resource updates. Quitting the
app shuts down its monitoring processes.

Physical cards show CPU, GPU utilization, authoritative node memory, swap,
temperature, power, network rates, and disk rates. Spark has unified memory:
GPU memory is never added to node RAM. Unsupported counters display as
unavailable, and old samples are marked stale. Chart history is bounded and
belongs to the current app session.

Network and disk rates are Linux counter totals. Virtual network interfaces or
stacked block devices may count the same I/O more than once; these are not
measurements of a particular physical link or drive.

Allocations come from the controller's ownership records and verified
containers. A distributed model has one inference endpoint spanning two Spark
cards. Independent models have independent endpoints and metric streams.
Resource metrics still exist when a node is idle or running YuE.

Inference metrics are fetched from the allocation endpoint's `/metrics`
(NIM uses `/v1/metrics`, with a fallback when that path is absent):

- Running and waiting requests, and KV-cache utilization are current gauges.
- Prompt tokens/s, generated tokens/s, and completed requests/s are counter
  deltas over the adjacent scrape interval.
- TTFT and time per output token are histogram **means over that interval**,
  not lifetime averages or percentiles.

Allocation/container changes, counter resets, missing series, and gaps reset
the rate baseline. The first observation therefore has no rate yet. The
adapter supports explicit vLLM metric families, current and legacy TPOT names,
and the unprefixed families in NVIDIA's
[NIM observability contract](https://docs.nvidia.com/nim/large-language-models/1.14.0/observability.html).
OpenAI compatibility alone does not establish metric support. Resource
monitoring still works when inference metrics are unavailable.

## Benchmarks

Benchmarks send synthetic requests to an already-ready managed allocation.
They do not change model placement, launch a model, or retarget Hermes.

```sh
./spark-serve bench run --node head --kind decode --requests 3 --max-tokens 128 --json
./spark-serve bench run --node worker --kind decode --concurrency 2 --requests 4 --json
./spark-serve bench run --node head --kind prefill --prompt-tokens 4096 --json
./spark-serve bench list --json
./spark-serve bench cancel --node head --json
```

Thinking uses the server default unless `--thinking` or `--no-thinking` is
explicitly selected. Use the same setting when comparing runs. Prompt sizes
are synthetic targets; saved **server-reported token usage** is authoritative.
Streaming chunks are never counted as tokens. A server that omits usage cannot
produce a successful throughput result.

Warmup is recorded separately from measured requests. End-to-end throughput
includes prefill and client/network overhead. A post-first-output estimate,
where available, is labelled as such because one streamed delta can contain
multiple tokens. These are operational comparisons, not isolated GPU kernel
measurements or model-quality evaluations. Concurrent user traffic, prefix
caching, prompt size, reasoning mode, and model configuration affect results.

Admission uses the controller lock to verify physical scope and immutable
allocation identity. The app also pins the allocation selected when Run was
clicked (`--allocation-id` in the CLI), rejecting a replacement before sending
any requests. A benchmark leases all participating hosts, so a TP2 run
cannot overlap a second run aimed at its worker. Stop, switch, and reboot revoke
overlapping benchmark leases before changing workloads. Cancellation closes
client requests; it does not prove that the inference engine has already
reclaimed every request's GPU work. Lifecycle idle checks remain authoritative.

Results live under `~/.local/state/spark-serve/benchmarks` (or
`SPARK_SERVE_STATE_DIR/benchmarks`). The app reads the same history as the CLI
and can compare saved runs. Failed and cancelled runs retain their status and
partial evidence rather than appearing successful.

## Scope and provenance

Thanks to [sparkDash](https://github.com/MiaAI-Lab/sparkDash) by
[Mia'a AI Lab](https://x.com/MiaAI_lab) for the ideas behind live resource and
inference metrics, the decode/prefill benchmark workflow, and the topology
overview. Spark Serve implements these ideas with native SwiftUI views and
Python stdlib collectors, reusing its own telemetry and streaming protocol
code. Workload ownership remains with the existing Spark Serve controller.

## Development checks

```sh
python3 -m pytest tests -q
make -f gui/Makefile
make -f gui/Makefile smoke
```

The Python suite uses local fixtures and injected transports. The native smoke
test decodes synthetic backend-generated status, telemetry, and benchmark
records and checks bounded history, null counters, and stale state. Neither
test suite submits inference requests or changes a Spark workload.
