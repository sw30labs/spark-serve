# Qwen3.8-Flash-Next NVIDIA NVFP4 on DGX Spark

This catalog entry runs NVIDIA's public checkpoint on the head Spark with TP1.
It exposes text, images and tools at the existing OpenAI-compatible endpoint,
using the served ID `qwen3.8-flash-next`. The second Spark has no Qwen rank.
Single-node placement reconciles only the selected Spark, preserving an independent
workload on the peer. Distributed allocations must be switched as a pair.
See [independent Spark control](independent-sparks.md).

## Install and run

Set the cluster's SSH hosts and `hf_cache_host` in the private `models.toml`,
then include the `models.qwen38` stanza from `models.example.toml`.

```sh
./spark-serve pull qwen38
./spark-serve up qwen38
```

`pull qwen38` prepares only the head node. It downloads the exact NVIDIA
snapshot into the configured Hugging Face cache, verifies it, and builds
`spark-serve-qwen38-nvfp4:0.1.0` from the vendored recipe. Downloads resume.
Preparation does not stop the current model. Every setup verifies all published
large-file SHA-256 hashes and small-file Git blob hashes. A corrupt cached file
gets one targeted retry; startup always rechecks small-file hashes and all sizes.
Allow space for approximately
132.73 GB of weights and metadata, the base container, and runtime caches.
The model files must remain on local NVMe: the lookup table is read during
inference. Moving it to network storage can make generation much slower.

Before `up qwen38` stops the current workload, a CPU-only container checks that
the built image and complete pinned snapshot exist. Missing files cause an
immediate error. First loading can take several minutes; the recipe permits
up to an hour while continuously checking the launched container for failure.
An exited process still fails promptly and preserves its logs.

The GUI reads this catalog dynamically. Select **Qwen3.8-Flash-Next NVFP4**,
shown as **1 Spark**, then choose the hostname's Start action. Use **Use in Hermes**
after readiness to select that endpoint, with its context limit scoped to the model.

## Runtime and provenance

| Component | Pinned version |
|---|---|
| NVIDIA checkpoint | `fc694b54fb0174e0913e6adf86691ef85a4ead47` |
| Community recipe | `6ad1c8f15cbab1ababd2048e8e5f94094dbfc4a0` |
| vLLM base source | `8a728663c1c3eeace834a95f5654fa653cc1998c` |
| ARM64 base image manifest | `sha256:a551e05307cd2e0092139d84db32af9c97e67d2eeeff072d21e429131d8c23f0` |

Sources: [NVIDIA checkpoint](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4),
[single-Spark recipe](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark/tree/6ad1c8f15cbab1ababd2048e8e5f94094dbfc4a0/single-spark-vllm-tp1).
The model weights are unchanged. Nine reviewed Python overlays are baked into
the derived image, with Apache-2.0 license, notices, original provenance and
an overlay hash manifest under `recipes/qwen38-nvfp4`.

The recipe's old ModelOpt dispatch did not recognize the refreshed NVIDIA
checkpoint's `FP8_PB_WO` name for MTP block-FP8 experts. Our one-line amendment
recognizes it alongside the older names, retaining the draft-local layer
mapping and 128×128 scales. See [recipe PR #1](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark/pull/1).
The regression exercises the actual dispatch methods without loading CUDA.

## Selected settings

- Native **262,144 tokens**, six sequences and 4,096-token prefill chunks.
- Official NVIDIA NVFP4 main experts; the checkpoint's other precisions remain intact.
- Staged disk PLE lookup; reduced MTP draft vocabulary of 65,536; three speculative tokens.
- FP8 e4m3 KV cache and 0.80 memory utilization.
- Decode CUDA graphs at sizes 4, 8, 12, 16, 20 and 24; compilation disabled.
- Prefix caching and FlashInfer autotuning disabled for this pinned stack.
- `qwen3` reasoning parser and `qwen3_xml` automatic tool parser.
- Thinking defaults on, matching NVIDIA's template. A client can request fast
  mode with `"chat_template_kwargs": {"enable_thinking": false}`.

This intentionally differs from the community speed profile's thinking-off
default. The fast profile gave one incorrect arithmetic answer in qualification;
thinking mode corrected it and passed the complete ten-case suite. NVIDIA's
checkpoint sampling defaults remain available; qualification requests explicitly
use greedy sampling for reproducibility and are not NVIDIA's benchmark settings.

The one-Spark entry overrides inherited NCCL transport to Socket and disables
IB. It does not change the cluster's cable or network configuration. A two-Spark
profile would need its own memory and transport qualification.
FlashInfer, Triton, Torch extension, CUDA and vLLM caches live under the configured
Hugging Face cache in Qwen-specific directories and survive container replacement.

The 262K server setting is the model's native limit, not evidence that every
workload fits at that size. Image tokens and concurrent requests also consume
memory. This recipe does not enable the 1M context extension.

## Qualification

Run the synthetic acceptance checks after the API is ready:

```sh
python3 tools/qwen_acceptance.py \
  --url http://YOUR-HEAD:8000/v1 --model qwen3.8-flash-next \
  --output diagnostics/qwen38-acceptance --concurrency
```

This exercises streaming text, strict JSON, automatic tool calling and its
result round trip, a generated image with OCR and colored shapes, coding
syntax, arithmetic, measured long-prompt retrieval and two simultaneous
requests. No model-supplied code or external tool action is executed.
An optional local API key comes from `SPARK_API_KEY`.

Each request and a combined report are saved in the chosen diagnostics
directory. Throughput uses server token usage: an MTP stream chunk may contain
several tokens. Visible TTFT and first model output are reported separately.
Short smoke-test rates are not a comprehensive performance benchmark.

For longer speed samples, `tools/qwen_benchmark.py` accepts the same URL/model
options and `--concurrency`. Its default run includes a warmup and six prose,
coding and numerical-explanation samples, with a ten-minute overall deadline.
It records per-request measurements and medians; concurrent aggregate throughput
is kept separate from single-request generation speed.

## Local qualification, 2026-09-13

The pinned image and all 25 checkpoint files passed integrity checks. The first
container reached API readiness in **12 minutes 42.6 seconds**; clicking Start
through the app, including preflight and controller steps, took about 13 minutes.
Model loading used **76.48 GiB** and took 583.4 seconds. The runtime allocated
19.1 GiB of KV cache and completed CUDA graph capture successfully.

The final reload with thinking enabled by default reached readiness in
**11 minutes 24.0 seconds** from container start. Model memory remained
76.48 GiB; this run allocated 18.7 GiB of KV cache. A fresh acceptance run
without any client thinking override passed **10/10** checks, with reasoning
tokens present in the responses. The app showed the model serving and Hermes
selected `qwen3.8-flash-next` with its 262,144-token limit.

The thinking-enabled suite passed **10/10** synthetic checks, including strict
JSON, automatic tool selection/result handling, image OCR and color recognition,
8,390 input-token retrieval, and two simultaneous requests. Coding checks syntax
and signature only; this is not an independent correctness evaluation of generated
programs. The fast suite passed 9/10: a two-step arithmetic question returned
198 instead of 204. The thinking retry returned 204 in 1.37 seconds. The original
failure receipt is retained.

The fast-mode benchmark comprised six longer responses and 1,828 output tokens:

| Measurement | Observed |
|---|---:|
| Median single-request throughput, including prefill | 37.13 tokens/s |
| Median first visible text | 0.305 seconds |
| Prose throughput | 26.77–27.84 tokens/s |
| Coding throughput | 37.19–38.65 tokens/s |
| Two-request combined throughput | 56.18 tokens/s |

These speed measurements used **thinking disabled**. Thinking adds reasoning
tokens and delays visible text; it is now the quality-oriented catalog default.
The numbers are local measurements, not the upstream author's benchmark results.
An additional **230,110-token** retrieval request passed with thinking enabled,
returning the exact random marker in 133.7 seconds. This exercises most of the
native context window; it does not assert perfect recall on arbitrary documents.
During ten minutes covering the substantive tests, minimum available memory was
approximately 10.54 GiB, swap use did not grow, and observed GPU temperature stayed at or below
73°C. The protected database remained healthy and the second Spark remained idle.
Receipts, exact requests and private host telemetry are retained under ignored
`diagnostics/2026-09-13-qwen38/`.
