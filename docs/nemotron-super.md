# Nemotron-3-Super NVFP4 on an independent Spark

`nemotron-super` runs NVIDIA's 120B/A12B NVFP4 checkpoint on one GB10 with
262,144-token context. Use the worker while Qwen remains on the head:

```bash
./spark-serve pull nemotron-super --node worker
./spark-serve up nemotron-super --node worker
```

Both models use port 8000 on their own host. Qwen is reached through
`cluster.lan_url`; Nemotron through `cluster.worker_lan_url`. Add `/v1` for an
OpenAI-compatible client. Nemotron's request model name is `nemotron-3-super`.
This placement is TP1 and does not use the inter-node GPU fabric.

Explicit node placement preserves the current Hermes selection. Once Nemotron
is ready, switch the client without restarting either model:

```bash
./spark-serve use --node worker  # select the worker's already-serving model
./spark-serve use --node head    # select the head's already-serving model
./spark-serve stop --node worker
```

An unscoped `stop` targets both nodes. DeepSeek profiles need both Sparks and
therefore replace both independent models when explicitly selected.

## Pinned sources

The configuration follows the DGX Spark section of the
[official vLLM Nemotron recipe](https://recipes.vllm.ai/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16),
pinned to [recipe commit 1fc9fdaae4bde39b8e192ddba9ba694885c98486](https://github.com/vllm-project/recipes/blob/1fc9fdaae4bde39b8e192ddba9ba694885c98486/models/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16.yaml).
This released-runtime recipe includes native reasoning and tool parsers.

| Component | Exact source |
|---|---|
| Checkpoint | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` |
| Checkpoint revision | `ff433f5493e25d631c9f12b5d55c674229923d02` |
| Upstream image tag | `vllm/vllm-openai:v0.28.0-ubuntu2404` |
| Linux ARM64 image digest | `sha256:41b54fb42c66a670a8b27e613ebef05898f24b9ab1bdab28bd00c877bd4935f4` |
| vLLM build commit | `2cf0a6915ce544dc493a0990f2ea38d81601128a` |
| Local image | `spark-serve-nemotron-super-nvfp4:0.1.0` |

The [NVIDIA model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/tree/ff433f5493e25d631c9f12b5d55c674229923d02)
provides the checkpoint and its [NVIDIA Nemotron Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/) terms. The stock
runtime retains its upstream licenses; the small derived image adds only this
repository's verifier and public source metadata. No vLLM source is patched,
model weights are not copied into the image, and no external reasoning plugin
is installed. See [recipe files](../recipes/nemotron-super-nvfp4/) for all pins
and the published file digests.

## Serving configuration

| Setting | Value |
|---|---|
| Nodes / tensor parallel size | 1 / 1 |
| Maximum context | 262,144 tokens |
| GPU memory utilization | 0.80 |
| Maximum concurrent sequences | 8 |
| Maximum batched tokens | 16,384 |
| Weight loader / attention | `fastsafetensors` / `flashinfer` |
| KV cache | FP8 |
| Mamba cache / SSM state | `align` / **float32** |
| FP32 matrix multiplication precision | `high` |
| Speculation | MTP, 3 tokens |
| Prefix caching | Enabled |
| Reasoning / tool parser | `nemotron_v3` / `qwen3_xml` |

The runtime selects the checkpoint's native ModelOpt quantization and the MoE
backend. FP32 Mamba state follows NVIDIA's numerical-stability guidance in the
[advanced deployment guide](https://docs.nvidia.com/nemotron/latest/usage-cookbook/Nemotron-3-Super/AdvancedDeploymentGuide/README.html).
The catalog keeps separate persistent compilation caches for Nemotron and
Qwen, enables the GB10 architecture target, and uses the socket backend for
single-node execution. Its 60-minute readiness allowance accommodates first-run
compilation; it is a timeout ceiling, not a measured startup estimate.

The 262K context limit is the current official Spark recipe's setting. Usable
concurrency and long-context performance depend on the runtime's measured KV
allocation. A larger context or batch should be qualified on the actual node.

## Preparation and integrity checks

Preparation runs only on the selected Spark. It downloads the exact public
checkpoint revision there, authenticates every published file, and builds the
image from the pinned ARM64 base. It does not start or stop a model. Weights
remain in that node's configured Hugging Face cache; `worker_hf_cache_host` can
override `hf_cache_host` when the nodes use different paths.

The initial worker preparation on September 13, 2026 reused a complete existing
head cache through a rate-limited transfer instead of downloading another copy
from the Hub. All **36 files / 80,365,684,262 bytes** passed full verification on
the worker. The prepared image was
`sha256:2d444d24394afa9c704ef4e598690311dca8f7851129c74c9986bd1ac6cb8987`.
CPU-only inspection confirmed vLLM 0.28.0, `fastsafetensors`, both native parsers,
and native Nemotron configuration loading with remote code disabled.

For an already-populated cache, skip checkpoint downloads while still doing
full verification and the image build:

```bash
python3 tools/prepare_nemotron.py --catalog models.toml --node worker --skip-download
```

A normal preparation may repair one failed cached file at the pinned public
revision, then reruns full verification. `--skip-download` never repairs files.
Every startup first uses the prepared image with no network, no GPU, and
read-only mounts to authenticate serving metadata and check shard lengths.
Missing or corrupt assets fail before any currently serving model is stopped.
Full SHA-256 hashing of large weight shards happens during preparation, so it
does not add an 80GB read to every startup.

## Live qualification — September 13, 2026

The prepared recipe served on sparktwo while the existing Qwen container on
sparkone continued running. Its first startup took about **7 minutes 30 seconds**
from container creation to a successful model-list response. Weight loading took
49.8 seconds; most remaining startup time was kernel tuning, compilation and
warmup. The runtime allocated 3.67 GiB for KV cache and reported 0.39 GiB of
captured CUDA graphs.

A worker-only stop and restart also passed, with Qwen's original container and
generation preserved. The cached restart took about **2 minutes 36 seconds**;
engine warmup fell from 323.8 to 34.8 seconds and compilation from 23.74 to 1.14
seconds. That second launch allocated 4.93 GiB of KV cache; allocations vary with
available memory. Both endpoints answered correctly after the restart.

All **9 functional checks passed** with default thinking and no retries: text,
strict JSON, automatic tool round trip, summary, coding syntax, numerical
reasoning, exact retrieval from **8,350 input tokens**, and two concurrent requests.
These synthetic checks qualify the exercised inputs, not the full 262K context
or maximum eight-sequence load.

In a separate simultaneous generation check, Qwen and Nemotron produced office
organization tips with 4.37 seconds of overlapping requests. Nemotron completed
144 output tokens (including reasoning) in 5.60 seconds, or **25.7 tokens/second
including prefill**. This is a single smoke measurement, not a sustained benchmark.
Detailed receipts are retained under
`diagnostics/2026-09-13-independent-sparks/` (ignored by Git).
