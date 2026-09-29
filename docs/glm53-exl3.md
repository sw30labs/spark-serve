# GLM-5.3-Flash EXL3 on two Sparks

`glm53-exl3` is a separate vLLM trial using EXL3/TR3 4-bit routed experts,
FP8 MLA KV and a DFlash2 speculative draft. It serves
`glm-5.3-flash-exl3` on the head's port 8000. Starting this tensor-parallel
model uses both Sparks and replaces their current serving workloads.

This trial is based on
[MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
at `b2c5986324a7bac7b02bf0a47ed7b92b25e56016`. Exact checkpoint and draft
pins belong in the [checkpoint manifest](../recipes/glm53-exl3/model-source.json),
with runtime sources in the
[runtime manifest](../recipes/glm53-exl3/runtime/upstream-files.json); serving settings
belong in `[models.glm53-exl3]` and its `env` / `vllm` sections in the catalog.
The existing `glm53` NIM recipe remains the comparison and rollback option.

| Asset | Pin |
|---|---|
| Target checkpoint | `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` at `25a44fdbf16862a46b7cc9921142c6c81350af2f` |
| DFlash2 checkpoint | `incoai/GLM-5.3-Flash-DFlash2` at `dc77ff1c99eeb2df044ee3d4f0094eb033fee410` |
| Target inventory | 144 files, 120 weight shards, 175,715,854,754 bytes |
| Draft inventory | 4 files, 2,342,175,855 bytes |
| Checkpoint manifest SHA-256 | `aede20b8ceb6cee9421f11e19d7a30b5aa99d3c96cd7f596fe564d785a975fa8` |

## What changes

| Component | Existing `glm53` pilot | EXL3 trial |
|---|---|---|
| Engine | Locally corrected NVIDIA NIM/SGLang image | Pinned Mia vLLM image |
| Routed expert weights | ModelOpt weight-only NVFP4 | EXL3/TR3 4 bits per weight |
| Attention path | Dense MLA fallback on the 11 attention layers | Sparse MLA SM121 adaptation |
| Target KV | BF16 | FP8 MLA |
| Speculation | Disabled | DFlash2, seven draft tokens |
| API model ID | `glm-5.3-flash` | `glm-5.3-flash-exl3` |
| Parsers | NIM profile | `glm45` reasoning, `glm47` tools |

The initial EXL3 catalog configures an 850,000-token ceiling, four sequences,
7,168-token prefill batches and a fixed 14 GiB KV pool per rank. These are
configuration limits, not four reserved full-size contexts. The current
InstantTensor loader needs the explicit KV budget at this geometry; the
upstream 0.85 utilization alone can profile too small a pool to boot.

The explicit KV reservation replaces automatic KV sizing from utilization;
`gpu_memory_utilization=0.85` is not a hard ceiling on total host RAM use.
Weights, activation/graph scratch, loader buffers and CPU processes need memory
beyond the 14 GiB pool. Upstream's fixed-14-GiB startup tests left approximately
5 GiB of idle host headroom. See the pinned
[InstantTensor memory measurements](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/blob/b2c5986324a7bac7b02bf0a47ed7b92b25e56016/README.md#instanttensor-and-kv-memory).

Native vision is enabled with a 2,048-token image processing limit and skipped
multimodal profiling. Initial acceptance covers one small still image.
Adaptive verification, additional dense FP8 conversion, experimental fast
EXL3 kernels and runtime abliteration are off. The recipe's compact DFlash
page geometry is enabled; full-context and sustained concurrency remain local
qualification work.

The SM121 sparse-attention adaptation also changes candidate selection: it
keeps 511 of the 512 selected four-token pools, then appends the three recent
tail tokens. This drops the lowest-ranked pool (four of 2,048 selected tokens)
to fit the kernel's fixed buffer. This approximation is part of the pinned
base image, even with the optional numerical experiments disabled.

The upstream EXL3 checkpoint is approximately 164 GiB, with 120 weight shards.
The separate [DFlash2 draft](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
adds roughly 2.3 GiB and carries CC BY-NC-ND 4.0 terms. These are different
quantization and runtime paths; passing API tests does not establish numerical
or model-quality equivalence to NIM or the original checkpoint.

Upstream implements native images, automatic tool calls, separated reasoning
and structured output. The local acceptance suite qualifies a generated still
image, text, JSON schema and tool history. Video is outside that suite.
Reasoning effort is kept at `low` in the initial correctness tests. The image's
chat template also accepts `enable_thinking: false`; the short speed benchmark
explicitly uses that mode, so its numbers must not be presented as comparable
to the always-thinking NIM measurements.

Upstream's August 28 high-acceptance code/counting benchmark reported 62.9
tokens/s C1 and 103.3 aggregate C2 with thinking off and 400 output tokens.
Those are historical results on the author's hardware. Its later 37.1 C1 prose
result used optional adaptive verification, dense FP8 conversion and cooperative
MoE; it does not describe an unmodified default recipe. Local results below
take precedence over these figures.

For long-context feasibility, the author's public image with stock settings
and compact DFlash KV enabled completed an **814,571-token cold prompt** with
**633.711 seconds to first token**, among 95 completed requests with zero
preemptions. The stock run's minimum sampled available RAM was **5.55 GiB**.
That run used an 850K limit, four sequences, 7,168-token batches and 0.85
utilization with automatic KV sizing; this local trial uses the explicit
14 GiB reservation. These are upstream results, not local qualification.
See the pinned [compact-cache qualification](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/blob/b2c5986324a7bac7b02bf0a47ed7b92b25e56016/README.md#experimental-compact-dflash2-cache-pages).

## Prepare and start

```sh
./spark-serve pull glm53-exl3
./spark-serve up glm53-exl3 --no-hermes --json
./spark-serve status --json
```

Preparation and startup are separate. Preparation stages and verifies the
pinned assets without replacing serving workloads. Keep `--no-hermes` during
initial acceptance so clients continue to use the prior selection until the
trial has been reviewed. Startup must pass both nodes' asset preflight before
stopping the old serving pair.

## Acceptance commands

These commands run from the Mac against an already-ready head. They use
synthetic fixtures and never execute generated code or external tools.
Replace the hostname if the private catalog uses another head.

```sh
python3 tools/glm53_acceptance.py \
  --url http://sparkone.local:8000/v1 --model glm-5.3-flash-exl3 \
  --reasoning-effort low --max-tokens 4096 --timeout 600 \
  --output diagnostics/glm53-exl3/acceptance-smoke.json

python3 tools/glm53_acceptance.py \
  --url http://sparkone.local:8000/v1 --model glm-5.3-flash-exl3 \
  --reasoning-effort low --max-tokens 4096 --timeout 600 \
  --tool-choice auto --only tools \
  --output diagnostics/glm53-exl3/acceptance-auto-tools.json

python3 tools/glm53_acceptance.py \
  --url http://sparkone.local:8000/v1 --model glm-5.3-flash-exl3 \
  --reasoning-effort low --max-tokens 4096 --timeout 900 \
  --only documents --long-context --context-targets 8192 32768 98304 \
  --output diagnostics/glm53-exl3/acceptance-long.json

python3 tools/qwen_benchmark.py \
  --url http://sparkone.local:8000/v1 --model glm-5.3-flash-exl3 \
  --samples 2 --timeout 180 --deadline-seconds 600 --concurrency \
  --output diagnostics/glm53-exl3/benchmark-fast
```

The GLM acceptance runner checks `/v1/models` and every stream's model ID.
It rejects truncation, missing finish markers and missing/invalid token usage.
Its named-tool test verifies the forced-call contract; the separate `auto`
round trip exercises model-emitted tool syntax and automatic parser handling.
Both preserve assistant reasoning and tool-call IDs in the replayed history,
then require the second turn to consume an unguessable synthetic result.
Vision and archive retrieval use strict JSON schemas whose values do not
contain the expected answers. Coding is checked with a bounded AST interpreter
on nine inputs, with no generated code execution.

The longer document run uses root `/tokenize` on the same vLLM endpoint;
no NIM tokenizer override is needed. The harness defaults to a conservative
131,072-token accounting limit even if the server advertises more. It sizes
prompts near 8K, 32K and 96K and records actual server counts. This does not
validate the server's entire configured context window.

After the shorter checks pass, exercise the configured 850K window with a
larger retrieval prompt. The fixture's proportional sizing may undershoot the
825K target; report the measured input count. The harness reserves space for
the response and refuses an oversized prompt before requesting completion.

```sh
python3 tools/glm53_acceptance.py \
  --url http://sparkone.local:8000/v1 --model glm-5.3-flash-exl3 \
  --reasoning-effort low --only documents --long-context \
  --context-limit 850000 --context-targets 825000 \
  --max-tokens 4096 --timeout 3600 \
  --output diagnostics/glm53-exl3/retrieval-near-800k.json
```

Despite its historical filename, `qwen_benchmark.py` accepts an explicit model
ID. It runs a warmup, two repetitions each of prose/coding/reasoning prompts,
then two simultaneous substantive requests. It disables thinking, allows 128
warmup / 768 sample output tokens, and uses server token usage. Its C2 aggregate
rate includes prefill and response overhead; its post-first-output metric is
an estimate because a speculative SSE delta can contain several tokens.
Repeated prompts may reuse prefix cache. This is a short speed sample, not
sustained-load qualification.

## Local results, 2026-09-27

Both nodes passed full checkpoint authentication and started the same image,
`sha256:92375877877fff12f71fdea0892812b1b21b21f6f7089a1edb21c9782b2f2815`.
The controller's readiness wait completed in **219 seconds after launch**,
excluding approximately **13.5 minutes of asset authentication** before
the ranks launched. The head rank started at 05:57:39 UTC and the API became
ready at 06:01:17 UTC; 219 seconds is not the wall time of the entire `up`
command. `/v1/models` advertised `glm-5.3-flash-exl3` with
`max_model_len=850000`. The initial smoke flagged a named-tool finish convention
that its checker did not support; the corrected checker passed the subsequent
named-tool round trip. The original failure receipt is preserved below.

The engine loaded **82.05 GiB per rank** in approximately 35.6 seconds;
graph capture took 101 seconds and allocated 1.93 GiB on the head and 1.88 GiB
on the worker. Effective serving settings matched the catalog: TP2, four
sequences, 7,168-token batches, FP8 target KV, 14 GiB KV per rank and DFlash2
k7 with a TP2 draft. The engine reported 1,572,073 token-equivalent KV capacity
(1.85 maximum-length requests). Its separate idle aligned prefix-cache figure
was 247,296 tokens; that measures a different cache property.
See the [startup receipt](../diagnostics/2026-09-26-glm53-exl3/startup-effective-config.json)
and [head](../diagnostics/2026-09-26-glm53-exl3/prepared-state-sparkone.json) /
[worker](../diagnostics/2026-09-26-glm53-exl3/prepared-state-sparktwo.json)
preparation receipts.

| Check | Local result |
|---|---|
| Pinned preparation and both-node preflight | Passed; full hashes and identical image |
| Startup and advertised context | Ready 219 seconds after rank launch, following approximately 13.5 minutes of asset authentication; 850,000 configured |
| Hermes selection and context configuration | Selected `glm-5.3-flash-exl3` through `spark`; model context 850,000, vision enabled |
| Streaming, strict schemas, coding and images | Passed initial smoke; coding checked on nine AST inputs |
| Named and automatic tool round trips | Both passed; named call uses the native `finish_reason=stop` convention |
| Measured 8K / 32K / 96K document retrieval | Passed at 7,756 / 31,431 / 95,061 input tokens |
| Near-800K document retrieval | Passed at 785,766 measured input tokens; 33.9-minute response under shared client traffic |
| Warm C1 / C2 speed sample, thinking off | 37.98 tokens/s median C1; 55.20 tokens/s aggregate C2 |
| Memory, OOM counters and container continuity | Sampled available RAM reached 1.25 GiB head / 3.39 GiB worker; host swap grew; no observed OOM kills or model-container restarts |

A [read-only Hermes configuration check](../diagnostics/2026-09-26-glm53-exl3/hermes-context.json)
confirmed the selected provider/model, the per-model 850,000-token context and
vision support, with no global context override. No configuration write was
needed. This verifies client settings, not an end-to-end Hermes conversation.

The [initial smoke receipt](../diagnostics/2026-09-26-glm53-exl3/acceptance-smoke.json)
passed five of six checks, including model identity, separated reasoning,
still-image OCR/colors, coding and exact recovery of three markers from a
1,395-token prompt. The named-tool response contained a valid `lookup_archive`
call and the expected arguments, but ended with `stop` instead of `tool_calls`.
The initial harness rejected that finish reason before replaying the tool
result. Inspection of the installed vLLM serving source confirmed that `stop`
is its intentional convention for a named tool choice; automatic and required
choices use `tool_calls`. The request-aware checker now accepts that named-call
convention while keeping automatic calls strict. The corrected
[named-tool replay](../diagnostics/2026-09-26-glm53-exl3/acceptance-named-tools.json)
passed: one valid call, correct arguments and exact consumption of the random
synthetic result on the second turn. The original failure receipt remains
intact, and
[source evidence](../diagnostics/2026-09-26-glm53-exl3/named-tool-finish-evidence.json)
records the installed branch and image identity. No serving-runtime change is
implied by this diagnosis.
The separate [automatic-tool receipt](../diagnostics/2026-09-26-glm53-exl3/acceptance-auto-tools.json)
passed the full round trip, with `finish_reason=tool_calls`, correct arguments,
and exact use of the random synthetic result on the second turn.

The [progressive retrieval run](../diagnostics/2026-09-26-glm53-exl3/acceptance-long.json)
recovered all three markers at **7,756**, **31,431** and **95,061** measured
input tokens, using low reasoning effort. Complete response times were
**13.21**, **21.63** and **76.98 seconds**, respectively.

The [larger retrieval probe](../diagnostics/2026-09-26-glm53-exl3/retrieval-near-800k.json)
also passed, recovering the early, middle and late markers exactly from
**785,766 measured input tokens**, with 59 completion tokens. The response took
**2,034.66 seconds (33.9 minutes)** while other clients used the same backend.
Retained engine logs show overlapping requests and repeated prefill deferrals
when the scheduler's learned step cost exceeded its decode-gap budget. This
elapsed time describes that shared workload, not isolated C1 throughput. Only
two nonempty streamed deltas arrived, so the receipt's post-first-output rate
estimate is not a useful decode-speed measurement. The server is configured
for 850,000 tokens; retrieval at the full 850K ceiling and sustained concurrent
full-context requests remain untested.
The [monitor summary](../diagnostics/2026-09-26-glm53-exl3/near800-monitor-summary.json)
records the observed scheduling pauses and sampled operating conditions.

The [fast-mode benchmark](../diagnostics/2026-09-26-glm53-exl3/benchmark-fast/report.json)
passed all six sequential samples and both simultaneous requests. The six
samples produced 1,720 output tokens, with a median end-to-end rate of
**37.98 tokens/s** and median first visible text of **0.488 seconds**. Prose
and coding medians were **19.99** and **38.89 tokens/s**, respectively. The
two-request pair produced 574 tokens in 10.398 seconds, or **55.20 aggregate
tokens/s**. These are short, warm measurements with thinking disabled and
possible prefix reuse, rather than comparisons to always-thinking NIM runs.

At the startup-ready capture, available RAM was **3.94 GiB on the head** and
**5.57 GiB on the worker**. During the shared near-800K run, the lowest available
RAM captured across the [root snapshot](../diagnostics/2026-09-26-glm53-exl3/operating-near800-01.json)
and [monitor samples](../diagnostics/2026-09-26-glm53-exl3/near800-monitor-summary.json)
was **1.25 GiB head / 3.39 GiB worker**. These are sampled minima, not guaranteed
absolute minima. Host swap usage increased from **1.74 to 3.75 GiB** on the head
and **0.58 to 1.53 GiB** on the worker between the
[ready](../diagnostics/2026-09-26-glm53-exl3/operating-ready.json) and
[after](../diagnostics/2026-09-26-glm53-exl3/operating-after.json) captures;
swap-out counters also increased. These counters cover the whole host and
cannot establish which process paged or whether UVM caused a stall.

The sampled host OOM-kill counters remained zero, and both model containers
retained their original IDs/start times with zero restarts and no OOM-kill
flag. The existing Neo4j container also retained its ID/start time and remained
running. After the checks, available RAM was **2.53 GiB head / 3.99 GiB worker**,
with no active or waiting requests on the head. The observed run completed,
but this tight memory margin and the long shared-load prefill do not qualify
sustained full-context concurrency or general model quality.

## Roll back

```sh
./spark-serve up glm53 --no-hermes --json
```

This restores the prepared NIM catalog entry on both Sparks. Confirm readiness
and model identity before changing any client selection. The NIM pilot's pins,
limitations and September 19 receipts are documented in
[glm53-nvfp4.md](glm53-nvfp4.md).
