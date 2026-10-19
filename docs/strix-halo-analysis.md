# Strix Halo: new tab or separate app?

Analysis, 2026-10-19. Branch `analysis/strix-halo`. Nothing here is implemented.

## Recommendation

**Same app, new tab, separate backend.** Add a "Strix Halo" tab to the native
app, but do not fold the Strix box into the existing two-Spark controller.
Give it its own catalog section, state, lock and telemetry collector, and let
the box itself own engine switching through an extended `halogen-ctl` from
`halo-inference-control` (separate repo). The Mac app becomes a
client of that box-local authority, exactly as the GNOME widget already is.

```
 SparkServeApp (Mac)
 ├─ Overview / Models / Benchmarks   ── spark-serve CLI ── Controller (2× GB10, NCCL, fencing)
 └─ Strix Halo tab                   ── spark-serve strix … ──ssh──▶ halo-ctl (on the box)
                                                                     ├─ halogen  (podman, :8731)
                                         GNOME widget ──────────────▶├─ lemonade (lemond)
                                                                     └─ vllm     (ROCm)
```

## Why one app

The operating procedure is the same on both machines:

1. A catalog of model/engine recipes.
2. Pick one; it needs most of the unified memory, so stop whatever else runs.
3. Start, wait for readiness, point Hermes at it.
4. Watch resources and inference metrics.
5. Benchmark under a fixed protocol; compare on cards (context, quantization,
   input types, runtime, measured tok/s, best-model star).

Steps 4–5 are already engine-agnostic: benchmarks talk OpenAI-compatible HTTP
with server-reported usage, and the cards, ranking and `catalog_tps_bench.py`
only need catalog fields and an endpoint. A separate app would duplicate the
SwiftUI shell, monitor, benchmark runner, card design and ranking, and you
would lose the one place where Spark and Strix models can be compared side by
side.

## Why not the existing controller

The Spark controller is built around two NVIDIA nodes acting together, and
that is load-bearing, not incidental:

| Coupling in spark-serve | References |
|---|---:|
| `nnodes` / distributed placement | 30 |
| `allocation_hosts` (multi-host allocation identity) | 19 |
| NCCL settings | 16 |
| `cluster.head` / `cluster.worker` roles | 9 |
| Docker launch paths | 8 |
| `nvidia-smi` telemetry | 2 (+ `spark_bench/telemetry.py`) |

The Strix box differs on each axis: one node, AMD GPU (ROCm / amdgpu hwmon,
no `nvidia-smi`), Podman rather than Docker, systemd/polkit-managed services
(`lemond`), and engines with different lifecycles (Halogen container, Lemonade
daemon, vLLM). Teaching the fenced two-node controller about all of that
would risk the Spark path to gain nothing. A thin, separate backend is less
code and keeps the Spark controller's guarantees intact.

## Why the box owns switching

Halogen and Lemonade (and vLLM) each want most of the 128 GB, and there is no
swap, so the out-of-memory killer takes the largest process. Mutual exclusion
must therefore be enforced in one place. `halogen-ctl` already does this for
Halogen ↔ Lemonade and already has `status --json`, and the widget already
uses it. If the Mac app started engines directly, the widget and the app could
race. Extending the box-local tool keeps a single authority, and the widget
and the Mac app become two front ends of it.

## Proposed shape

**On the box (halo-inference-control):**

- Generalise `halogen-ctl` to `halo-ctl` with engines `halogen`, `lemonade`,
  `vllm`: `halo-ctl up <engine> [model]`, `stop`, `status --json`.
- One lock and an exclusive switch: starting one engine stops the others,
  then toggles `inference-mode` as Halogen needs.
- `status --json` reports engine, model, served name, port, phase
  (`stopped | starting | ready | failed`), and real free memory (the
  locked-weights correction `halogen-stats` already does).
- Keep `halogen-ctl` as a compatibility alias so the widget keeps working.

**In spark-serve:**

- `[strix]` catalog section: SSH alias, LAN base URL, and
  `[strix.models.<id>]` with `engine`, `served_name`, `port`, `ctx`, `quant*`,
  `modalities`, `notes` — the same card fields as Spark models.
- `spark_serve_strix.py`: calls `halo-ctl` over SSH; its own state file and
  lock under the state directory; no NCCL, fencing or Docker.
- CLI namespace: `./spark-serve strix list|status|up <id>|stop|watch|bench`.
- Telemetry: reuse `halogen-stats` JSON (memory, GPU load, temperature,
  power) instead of the `nvidia-smi` collector; inference metrics from each
  engine's endpoint (Halogen `/metrics`; Lemonade and vLLM to be confirmed).
- Benchmarks: the runner's admission (`_identity`) is tied to controller
  allocations. Add a Strix identity provider; the request/accounting code
  stays the same. `catalog_tps_bench.py` gains a `--site strix` option so
  Strix models are measured under the same protocol.
- GUI: a top-level "Strix Halo" tab reusing `ModelCard`, `PillFlow`,
  `BestBadge`, the benchmark views and the Hermes buttons. The star is
  computed per site; a lab-wide comparison can come later.

## Phases

1. Box: `halo-ctl` with exclusive switching and `status --json` for all three
   engines; widget keeps working.
2. spark-serve: `[strix]` catalog and `strix status/up/stop`.
3. Telemetry and the GUI tab (cards, start/stop, Hermes).
4. Benchmarks and measured speeds on the cards; quantization tags for GGUF and
   Halogen formats.
5. Point the Grok model scout at Strix-compatible releases too.

## Open questions (need you or the box)

- **The box was unreachable** on 2026-10-19 (SSH to its old
  address timed out). After the 10-18 SSD reinstall, what is its hostname?
  Nothing about the running system was verified for this analysis.
- How vLLM runs there today (container image, ROCm version, launch command)
  and which models Lemonade serves.
- Whether Lemonade and vLLM expose usage in streamed responses (needed for
  benchmarks) and Prometheus metrics.
- Quantization of Halogen's `.hgn` weights: not documented in the files read
  here; must be verified before tagging cards.
- Hermes: should "Use in Hermes" target the Strix endpoint too (it needs an
  API key; Halogen already has one)?
- Name: "spark-serve" will no longer describe the app; renaming is optional
  and can wait.

## Not recommended

- **A separate app:** duplicates monitor, benchmarks, cards and ranking, and
  splits the decision across two windows.
- **Strix as a third Spark node:** couples an AMD single-node box into NCCL,
  fencing and two-node allocation logic it does not use.
- **The Mac app driving Podman/systemd directly:** two controllers (widget and
  app) racing for 128 GB without swap.
