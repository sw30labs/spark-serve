# GLM-5.3-Flash NVFP4 on both Sparks

`glm53` adds a locally corrected runtime derived from NVIDIA's two-Spark NIM
image to the catalog. It uses both GPUs for one model and exposes the client API on the head. Starting it
replaces the workloads on **both** Sparks.

**The corrected runtime passed bounded live acceptance on September 19, 2026,
and is selected in Hermes.** Both Sparks serve the same pinned image. The
configured limit is **131,072 tokens (128K)**; successful document retrieval was
tested through **95,144 prompt tokens**, not the full limit. This is an
operational pilot, not a sustained-load qualification or proof of superiority
over the existing models.

## Pinned runtime and checkpoint

| Component | Pin |
|---|---|
| Base runtime | NVIDIA NIM `nvcr.io/nim/zai-org/glm-5.3-flash:2.1.2-variant` |
| Linux ARM64 base manifest | `sha256:0bd2a1f4ffacf4ee61e1f8c1b48071a6713b51bad954ab0ca627c24be84f43f8` |
| Serving image ID | `sha256:161214462d6696699210c0aef16b2ba706e5a8ad692e2c9490aaa70d190e794c` |
| Local patch revision | `marlin-gate-up-maxglobal-e4m3-v1` |
| Explicit NIM profile | `d3c5ed87ba2f6ed5424d46a191fde9d1c53f424d3d454372fae8873205a04b8b` |
| NGC model/version | `nim/zai-org/glm-5.3-flash:nim-aa28e1f-nvfp4` |
| Checkpoint contents | 130 files, including 120 weight shards |
| Checkpoint size per Spark | **194,692,707,910 bytes (181.32 GiB)** |
| Manifest SHA-256 | `9d69f5c64b423a3da262dd64d35b3db30209aa8a5a60698cc819a45e86e17564` |

This is NIM's published NGC artifact: ModelOpt 0.45.0 weight-only NVFP4. It differs
from the approximately 204 GB, 33-shard Hugging Face artifact using ModelOpt
0.47.0. Preparation uses the NGC artifact and its publisher-provided SHA-256
digests for every file. The checkpoint remains unchanged. Preparation builds a
small corrected runtime layer on the pinned base. Exact metadata lives in the
[recipe directory](../recipes/glm53-nvfp4/).

The original image's Marlin loader used the gate projection's global scale for
both gate and up projections, even when the checkpoint supplied different
scales. This introduces a numerical error in the up projection. The local patch
uses the larger global scale for each expert and rescales both halves' block
scales before Marlin conversion. Rounding those block scales back to E4M3 adds
quantization error; this is neither exact weight preservation nor an official
NVIDIA fix. The runtime retains Marlin; it does not switch to a Cutlass workaround.
The upstream [SGLang scale-handling change](https://github.com/sgl-project/sglang/pull/27588)
explicitly leaves Marlin and Cutlass on a single global scale. An open
[vLLM reconciliation proposal](https://github.com/vllm-project/vllm/pull/55073)
uses the same maximum-scale policy; this local SGLang correction is independently
tested and is not an adoption of an upstream release.

[Checkpoint scale validation](../diagnostics/2026-09-19-glm53/quantization-review/checkpoint-scale-validation.json)
sampled 16 expert pairs and 32,768 blocks; nine pairs had unequal scales:

| Sampled relative scale error | Original loader | Local correction |
|---|---:|---:|
| Maximum | 42.857% | 5.455% |
| Mean | 5.628% | 0.673% |

These are sampled effective weight-scale errors, **not model-accuracy scores**.
They exclude original checkpoint quantization and Marlin activation/global BF16
rounding. Basic API and isolated Hermes probes passed on the unpatched image;
those passes were insufficient for adoption after this defect was found.

Independent CPU builds on both Sparks produced the same serving image ID, recorded
in the [build receipt](../diagnostics/2026-09-19-glm53/runtime-build-both.json).
The build checks original and patched source hashes, creates deterministic
hash-based Python bytecode, and preserves the NIM user `nvs:1000`.

CPU-only inspection of the pinned image confirmed its GB10 NVFP4 profile and
`gb10_throughput_nvfp4_1.yaml.j2` settings:

| Setting | Image profile value |
|---|---|
| Tensor parallel size / nodes | 2 / 2 |
| KV cache | BF16 |
| MoE backend | Marlin |
| Static memory fraction | 0.94 upstream; **0.92 catalog override** |
| Maximum running requests | 4 |
| Prefill chunk size | 4,096 tokens |
| Maximum decode CUDA graph batch size | 4 |

The catalog explicitly selects this profile through `NIM_MODEL_PROFILE` and
requests the initial 128K context limit. `NIM_KV_CACHE_PERCENT=0.92` reduces the
static memory reservation after a 95K-token trial at the upstream 0.94 setting
left less than 1 GiB available on the head. The four-request scheduler limit
does not reserve four simultaneous 128K contexts; requests share cache capacity.

The pinned GB10 profile sets `index_topk:null`: it disables GLM's sparse indexer
and runs **dense MLA on 11 attention layers**. The other 34 KDA layers remain
linear. This changes attention behavior relative to sparse-model inference and
increases long-document attention work. Published sparse-runtime throughput and
quality cannot be assumed equivalent here. `index_share_for_mtp_iteration:false`
separately disables draft-step index reuse; it is not the sparse-indexer switch.
MTP is disabled in this profile.

NVIDIA documents [two-Spark deployment](https://docs.nvidia.com/nim/vision-language-models/latest/deploy-on-dgx-spark.html)
and [local-model deployment](https://docs.nvidia.com/nim/vision-language-models/latest/get-started/advanced/get-started-glm-5-3-flash.html).
Thinking is always enabled; requests can choose `reasoning_effort` as `low`,
`high` or `max`, with `max` the upstream default. Text and images do not require
FFmpeg. Video needs FFmpeg 8 on both hosts and separate qualification.

## Local qualification — September 19, 2026

The following results use the pinned corrected image, memory fraction **0.92**,
BF16 KV cache, TP2 across two GB10 hosts, and `reasoning_effort=low`.
The final startup reached readiness **567 seconds after both ranks started**;
this excludes preparation and controller preflight. Runtime inspection confirmed
128K context, a four-request scheduler limit and **204,333 shared KV tokens**.
Startup allocated 26 Mamba cache slots.

| Check | Final result |
|---|---|
| Streaming and separate reasoning | Passed |
| Tool arguments and result round trip | Passed |
| Image text/shapes and strict JSON output | Passed |
| Coding fixture | Passed all nine bounded correctness cases |
| Actual Hermes agent | Tool and native image checks passed in an isolated Hermes home |
| Four concurrent requests | 12/12 passed across three batches; 10.25 seconds total |
| Desktop and selection | Fourth card shows 128K / 2 Sparks; both nodes serving; head selected in Hermes |
| Offline suite | 537 tests and 38 subtests passed |

Long-document requests retrieved independent markers near the beginning, middle
and end, using strict JSON schema without encoding the answers in that schema:

| Actual prompt tokens | End-to-end seconds | First content seconds | Result |
|---:|---:|---:|---|
| 7,774 | 12.58 | 9.22 | All three markers correct |
| 31,451 | 47.88 | 44.34 | All three markers correct |
| 95,144 | 310.84 | 306.89 | All three markers correct |

These are single synthetic retrieval requests, not document reasoning accuracy
or maximum-context benchmarks. The dense-attention fallback makes long prefill
expensive. Native video, the full 128K window, simultaneous long contexts,
`high`/`max` reasoning effort, sustained load and matched comparisons with the
other three models remain unqualified. The Hermes selection preserves existing
reasoning preferences; it does not impose the probes' `low` effort setting.

Memory snapshots before / during the largest request / after acceptance showed
**5.20 / 3.22 / 3.61 GiB available on the head** and **6.17 / 4.76 / 4.75 GiB on
the worker**. These are sampled values, not measured peaks. Both containers
retained their original identities with zero restarts, and host OOM-kill counters
did not increase. The protected database remained running. Existing swap
allocation remained approximately 4.76 / 2.58 GiB; swap-out counter deltas over the
acceptance window were only 20 / 16 KiB. This is not a claim of swap-free service.

Local receipts: [API smoke](../diagnostics/2026-09-19-glm53/acceptance-smoke.json),
[long documents](../diagnostics/2026-09-19-glm53/acceptance-long.json),
[Hermes](../diagnostics/2026-09-19-glm53/hermes-final.json),
[concurrency](../diagnostics/2026-09-19-glm53/concurrency.json),
[effective runtime](../diagnostics/2026-09-19-glm53/server-info-final.json),
[memory before](../diagnostics/2026-09-19-glm53/final-operating-before.json),
[during](../diagnostics/2026-09-19-glm53/final-operating-during-long.json) and
[after](../diagnostics/2026-09-19-glm53/final-operating-after.json).
Diagnostics are intentionally ignored by Git and may not exist in another clone.

## Prepare

For an existing installation, merge `[models.glm53]`, `[models.glm53.nim]` and
`[models.glm53.env]` from `models.example.toml` into the private catalog. Preserve
the existing cluster configuration and the image, manifest and NIM profile pins.

```sh
./spark-serve pull glm53
```

Preparation runs on both hosts, downloads public files without an API key,
authenticates all checkpoint bytes with SHA-256, pulls the pinned NIM base, and
builds the corrected runtime without GPU access or build-step network access.
It validates the three build-file hashes and rejects any resulting image ID
that differs from `source-pins.json` before publishing a preparation receipt.
It does not stop or start serving workloads. Reserve space on each Spark for
the checkpoint, container image, compilation cache and normal operating margin.
Interrupted transfers retain `.incomplete` files; rerun preparation to resume.

For direct control with a Python 3.11+ interpreter:

```sh
python3 tools/prepare_glm53.py --catalog models.toml --node both
python3 tools/prepare_glm53.py --catalog models.toml --node worker
python3 tools/prepare_glm53.py --catalog models.toml --skip-download
```

`--skip-download` fully verifies existing checkpoint files and still builds and
verifies the pinned derivative image. The direct preparation tool can prepare one host;
serving requires both.

Each host stores assets below
`<hf_cache_host>/spark-serve/glm53-nvfp4/`; `worker_hf_cache_host` can override the
worker's cache root. Model files are mounted read-only. The separate
`runtime-cache/` directory is writable at `/opt/nim/.cache`; its ownership must
allow the NIM image's numeric user to write. Preparation records a `prepared.json`
receipt after verification, including the actual image ID, base image and patch hashes.

Before changing a running workload, startup performs CPU-only preflight on both
hosts with no network and read-only checkpoint access. It verifies the pinned
manifest, hashes small serving files and checks every shard's size. Full shard
hashing happens during preparation. `--expected-manifest-sha256` in the catalog's
preflight arguments rejects a staged manifest that differs from the controller's
pin before checking checkpoint assets.

## Start and qualify

Keep the current Hermes selection during initial qualification:

```sh
./spark-serve up glm53 --no-hermes --json
./spark-serve status --json
./spark-serve logs -f
```

NIM starts rank zero first. The launcher reads `NIM_PRIMARY_NODE` from that
container's log, then starts rank one with the observed address. NIM discovers
the transport settings; the recipe does not inherit the DeepSeek NCCL/UCX
overrides. The default ports are 8000 for the head API, 8002 for the worker and
20000 for node management. Client requests go to the head.

Startup verifies both immutable container IDs, `/v1/health/ready` and the expected
served model ID. A failed rank or handshake retains bounded startup diagnostics
before cleanup. Use `./spark-serve logs worker` to inspect the other rank.

For broader qualification, retain receipts for text, reasoning, tool round trips,
structured output, image handling, coding correctness, long-context retrieval,
concurrency and a sustained run. Record actual server context, reasoning effort,
latency, throughput, available memory and swap. Compare the same workloads and
reasoning budgets against the existing models.

After the model is ready, run the bounded synthetic acceptance harness. Replace
`sparkone.local` with the configured head hostname if necessary:

```sh
python3 tools/glm53_acceptance.py --url http://sparkone.local:8000/v1 --model glm-5.3-flash --reasoning-effort low --output diagnostics/glm53-smoke.json
python3 tools/glm53_acceptance.py --url http://sparkone.local:8000/v1 --tokenizer-url http://sparkone.local:8001 --model glm-5.3-flash --only documents --long-context --context-targets 8192 32768 98304 --timeout 900 --output diagnostics/glm53-long.json
```

The smoke run checks streaming, a tool round trip, a generated image fixture,
bounded coding correctness and document retrieval. The longer run targets
measured prompt lengths below the configured limit. The pinned NIM front proxy
does not expose `/tokenize`; the harness uses the head's backend on port 8001 for
token counting. **All completion requests still go through port 8000.** These
synthetic checks produce JSON receipts and do not establish general model quality
or replace concurrency, soak and Hermes end-to-end checks.

After successful qualification, select the already-running head in Hermes:

```sh
./spark-serve use --node head
```

Subsequent `./spark-serve up glm53` calls retarget Hermes after readiness unless
`--no-hermes` is supplied. The initial 128K limit should only be raised after
qualification on the actual two-node runtime. The catalog sets
`hermes_supports_vision = true`; selection merges vision support and the 128K
limit into that model's Hermes metadata while preserving other settings.

## Recovery and offline checks

Restore the existing DeepSeek recipe through the normal controller:

```sh
./spark-serve up ds4
```

This replaces the GLM allocation on both nodes and retargets Hermes after
DeepSeek is ready. If a node is unreachable, reconcile it before retrying;
preserve the startup diagnostic path from the error. No reboot or manual
container removal is part of the ordinary recovery procedure.

Use Python 3.11+ with pytest. On the configured Mac, the following interpreter
is also used by the CLI's compatibility fallback:

```sh
~/miniconda3/bin/python3 -m pytest tests/test_glm_preparation.py tests/test_glm_nvfp4_patch.py tests/test_nim_backend.py tests/test_nim_startup.py tests/test_glm_acceptance.py -q
```

These offline checks cover integrity, interrupted downloads, pin rejection,
node/cache selection, deterministic image preparation, scale reconciliation,
NIM command construction and startup failure paths. They
do not contact the cluster or establish live model quality.

The final implementation passed **537 offline tests and 38 subtests**. Live
acceptance above establishes the tested behavior on the current two-Spark
deployment; it does not establish a general quality or performance advantage
over the existing models.
