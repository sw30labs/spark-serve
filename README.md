# spark-serve

Mac CLI + SwiftUI helper that switches between catalogued vLLM models and
independent YuE song workers on a two-node [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
cluster is serving.

**This is not how you set up a cluster.** Cabling, ConnectX-7 / QSFP, pairing
the Sparks, SSH, and the fabric are NVIDIA's docs, not this repo. Start at the
[DGX Spark user guide](https://docs.nvidia.com/dgx/dgx-spark/)
([system configuration and clustering](https://docs.nvidia.com/dgx/dgx-spark/system-config-and-operation.html)).
This project assumes that cluster already exists and the Mac can SSH to both
nodes. It controls workload ownership and starts, drains, stops, and swaps the
serving backend.

The live OpenAI-compatible endpoint stays on the head node's LAN port 8000.
YuE uses one HTTP worker per Spark on port 8011. All SSH / Docker / NCCL work lives in the Python CLI. The GUI is a thin
`Process` wrapper around that CLI.

<p align="center">
  <img src="docs/gui.png" alt="spark-serve GUI: catalog cards, Start/Stop, foreign occupant on port 8000" width="720">
</p>

```
cp models.example.toml models.toml   # then edit [cluster]
./spark-serve list
./spark-serve status
./spark-serve up ds4
./spark-serve stop
./spark-serve logs -f
```

## Setup

1. Two Sparks with SSH aliases for head and worker (`BatchMode=yes`).
2. Python 3.11+ on the Mac (`tomllib`). The CLI re-execs `~/miniconda3/bin/python3`
   if `/usr/bin/python3` is too old.
3. Copy `models.example.toml` → `models.toml` and set:

   - `head` / `worker` — SSH hostnames
   - `master_addr` — QSFP / RoCE IP of the head (not the LAN NIC)
   - `lan_url` — URL clients use, e.g. `http://<head-lan-ip>:8000`
   - `hf_cache_host` — Hugging Face cache on the Sparks

`models.toml` is gitignored on purpose. Do not commit LAN IPs or SSH hostnames.

## Behaviour

`up ds4` / `up ds4-vision` drain YuE jobs, stop the previous workload, verify
both GPUs are free, start the worker (rank 1, `--headless`) and head (rank 0),
then retarget Hermes after readiness. Single-node recipes such as
`nemotron-super` preserve their `nnodes=1` and TP=1 settings.

`up yue` starts two **independent** workers. Each can render a different song or
take. This uses the existing SSH hosts and Mac app, while YuE keeps its own
inference pipeline and does not inject NCCL settings or change Hermes.

Every `up` and `stop` uses the same process lock and atomically persisted state
in `~/.local/state/spark-serve` (`SPARK_SERVE_STATE_DIR` overrides it). Discovery
is revoked before draining. Active renders keep running: the command reports
their job IDs and exits without starting a conflicting workload. Retry after
completion, or explicitly use `--cancel-jobs`. A failed/interrupted transition
stays visible and the next command reconciles both nodes before starting anything.
Each node also keeps a generation lock in `~/.local/state/spark-serve`; delayed SSH
commands for an older transition cannot admit a worker or launch a model after a
new transition starts. Commands hold that lock through their side effects,
including when a parent process is interrupted.

While a vLLM model starts, the CLI checks the exact containers it launched on
every required node. An exited container or an unverifiable node ends the wait
with its host and failure reason instead of leaving the app booting until the
readiness timeout. Before cleanup, bounded logs and container states are saved
under `~/.local/state/spark-serve/diagnostics/startup-*/failure.json` (or the
configured state directory) and shown in the app's boot log. The green model
check appears only after readiness; controls stay busy through cleanup.

See [DeepSeek startup recovery](docs/ds4-startup-recovery.md) for the verified
QSFP Socket transport workaround and validation results.

Only exact catalog-owned Docker IDs are stopped. `keep_containers` are retained,
including when accidentally listed in `stop_names`. Unreachable hosts, untracked
GPU containers, or remaining host compute processes block a mode switch. Stop
foreign workloads separately; Spark Serve does not claim ownership from a name
prefix or a port number.

`status --json` retains vLLM fields and adds `mode`, `phase`, `transition_error`,
`yue_workers`, and `ready_workers`. Distributed vLLM readiness requires the expected
container on **both** nodes plus the expected served ID; a head-only response is
sufficient only for an explicitly single-node recipe. YuE readiness requires the
same worker identity and generation from control and HTTP, validated pinned assets,
CUDA, and admission. HTTP 200 alone is insufficient.

### Install the YuE workers

Install Artist Twin's protocol-v2 factory and pinned runtime/assets on each Spark
using its `scripts/spark` deployment instructions. The service must start drained;
do not enable autonomous admission or run the old v1 factory. This CLI checks the
source protocol marker before invoking controls, so an old script cannot mistake
`control status` for a request to start another server.

After copying the final factory payload and creating its systemd service, run
`control manifest` on each Spark with that service's environment. This inventories
the pinned weights, tokenizers, codecs, source, factory files, and Docker image.
`up yue` checks those assets and performs a CUDA/import probe before admission.
The factory must be upgraded on both nodes before switching to or from YuE.

Optionally add this **top-level** section to the private `models.toml`:

```toml
[yue]
head_url = "http://spark-head.lan:8011"
worker_url = "http://spark-worker.lan:8011"
# service = "yue-icl.service"
# factory_root = "~/.local/share/artist-twin/yue-factory"
# port = 8011
# ready_timeout = 180
```

Use LAN names/addresses resolvable from the Mac; SSH aliases alone need not be DNS
names. The default head URL reuses the host from `cluster.lan_url`; the worker URL
defaults to the configured SSH worker name. Unknown `[yue]` keys fail closed.
Credentials belong in environment variables, never the catalog or discovery.

```sh
./spark-serve up yue
./spark-serve status --json
./spark-serve stop                 # drain; blocks if a render is active
./spark-serve stop --cancel-jobs   # explicit cancellation and verified cleanup
./spark-serve up ds4              # drain YuE, release GPUs, start DeepSeek
```

Artist Twin reads the atomic public file
`~/.local/state/spark-serve/yue-workers.json`, not this private model catalog.
Its schema is `{version:1, mode:"yue", generation, workers:[{id,url,generation,
runtime_manifest}]}`. An unavailable mode publishes an empty worker list.
Worker IDs come from each factory's durable identity, not the SSH alias. Artist
Twin owns durable job/take dispatch, rights checks, seeds, downloads, and provenance.
Rebooted workers start drained and require `up yue` to validate and re-admit them.

Two workers improve throughput under load; they do not split a single song across
the fabric. Each factory still loads the model pipeline for each render. Single-song
latency and aggregate speedup need measurement on the installed Sparks.

### Measure YuE concurrency capacity

`python3 -m spark_bench concurrency` runs a controlled C1…CN experiment with the
same seeded corpus, per-job audio-integrity gates, raw telemetry and reproducible
throughput/latency analysis. Its isolated dispatcher leaves production concurrency
unchanged. Use `--ssh-host` to drain one configured Spark and hold workload ownership
through the experiment. See [the benchmark guide](docs/worker-concurrency-benchmark.md)
for workload setup, guards, raw results and the criteria for a capacity recommendation.

The [September 8 handover](docs/handover-2026-09-08.md) records the cooling pause,
preserved evidence, active replay and commands for resuming both investigations.

Add a model by copying a `[models.<id>]` table. `wrapper = "vllm"` for a stock
`vllm/vllm-openai:*` image (ENTRYPOINT already `vllm serve`). `wrapper = "dsv4"`
for the Aiden GB10 DeepSeek image.

Do not bake a sudo password in here. `up` drops page caches only if
passwordless sudo already works on the Sparks.

NCCL / UCX in the example catalog are pinned to the right-port QSFP rails
(`enp1s0f1`). Do not switch them back to f0 unless the cable moves.

## GUI

Menu-bar + window app. It shells out to this CLI (`list` / `status` / `up` /
`stop`); it does not speak SSH itself. The YuE card and per-worker status show
readiness, draining, and active work. The regular Stop preserves active renders;
“Cancel jobs & stop” is a separate confirmed action. It never kills a running CLI
transition merely to launch another one.

```
make -C gui
open gui/SparkServeApp.app
```

CLI path is resolved from the app bundle (`repo/gui/SparkServeApp.app` → repo)
or `SPARK_SERVE_HOME`. The app's `PATH` includes `~/miniconda3/bin` so the
`python3` shebang works under a GUI environment.

Requires Command Line Tools (`swiftc`); full Xcode is not needed.

## Offline verification

```sh
python3 -m unittest discover -s tests -v
```

Fault-injection tests cover concurrent controllers, lost/offline ownership,
legacy migration, active-job draining, explicit cancellation, stale generations,
misrouted health, partial starts, exact-container cleanup, protected services,
and preservation of single-node/distributed vLLM recipes. They use no network or GPU.
