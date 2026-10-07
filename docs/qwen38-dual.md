# Qwen3.8-Flash-Next across two Sparks

`qwen38-dual` adds a separate TP2+EP+MTP3 runtime from
[MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks/tree/cd839d0b62f6c737178a0c2037733420e0524ebf).
It uses that repository's documented `start-v030.sh` lane: vLLM 0.30.0,
the original NVIDIA NVFP4 checkpoint, native 262,144-token context and
BF16 attention KV. Its API model ID is `qwen3.8-flash-next-dual`.
The existing single-Spark Qwen entries remain available.

Preparation completed on both Sparks on 2026-10-05. Both nodes have the same
ARM64 image, `sha256:310530cd660abf5d4f27dbc3a763462b2cd58dcec42a05f53aa686882d37ddc1`,
and the original 25-file NVIDIA checkpoint (132,734,506,847 bytes) passed full
hash verification on each node. Native model, QSA, PLE and MTP imports and
metadata compatibility checks passed in both images with networking disabled
and GPUs hidden; CUDA remained uninitialized. No inference server was started.
The [local preparation record](../diagnostics/qwen38-dual/preparation-2026-10-05.json)
contains the matching image and checkpoint receipts and CPU reports.

Distributed GPU serving, tools, vision, retrieval, model quality and performance
qualification remain pending. Upstream measurements describe the author's
hardware and configuration; they are not local qualification results.

| Asset | Pin |
|---|---|
| Upstream source | `cd839d0b62f6c737178a0c2037733420e0524ebf` |
| Model | `nvidia/Qwen3.8-Flash-Next-NVFP4` |
| Model revision | `fc694b54fb0174e0913e6adf86691ef85a4ead47` |
| Official vLLM base | `vllm/vllm-openai@sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56` |
| Local image | `spark-serve-qwen38-dual:0.1.0` |
| Checkpoint inventory | 25 files, approximately 132.7 GB on each Spark |
| Draft vocabulary | 47,149 IDs, from the TP-aware dual-Spark source |
| Provenance | [Source pins](../recipes/qwen38-dual/source-pins.json), [runtime sources](../recipes/qwen38-dual/runtime/upstream-files.json), [checkpoint manifest](../recipes/qwen38-dual/model-source.json) |

## Runtime settings

One rank runs on each Spark, with expert parallelism and
`allgather_reducescatter` communication over the catalog's existing fabric.
The shared cluster NCCL settings remain authoritative. The vision encoder is
replicated with `--mm-encoder-tp-mode data`: sharding its 4,304-wide intermediate
layer at TP2 would produce 2,152 features, incompatible with NVFP4 kernels.

The initial baseline follows the upstream v0.30 defaults: 262K context,
eight sequences, 8,192-token prefill batches, 0.80 GPU memory utilization,
BF16 KV, BF16 recurrent state, full decode CUDA graphs and breakable CUDA
graphs disabled. BF16 recurrent state differs from the existing single-Spark
v0.30 entry's FP32 setting and requires local quality qualification. Context
and sequence ceilings do not reserve a full context for every request.

The only installed runtime overlay is the upstream TP-aware reduced-vocabulary
MTP patch. Each rank slices its own draft `lm_head` shard and gathers the
winning value and token ID across ranks. The target still verifies with the
full vocabulary. The single-Spark vocabulary and TP1 patch are not reused.
MTP uses three proposals, local argmax reduction, index sharing and disabled
trailing prefix-cache block dropping, matching the upstream lane.

Stock vLLM 0.30 handles NVIDIA's mixed NVFP4/FP8 checkpoint metadata and MTP
quantization remapping. This entry does not add the older ModelOpt overlays,
single-Spark PLE mmap patch, FP8 KV backport, YaRN, FP8-dense conversion,
gated checkpoint variant or NFS weight sharing. Text, still images, reasoning
and automatic tools use the checkpoint's native template with `qwen3` and
`qwen3_coder` parsers. Thinking defaults on; requests can disable it with
`chat_template_kwargs: {"enable_thinking": false}`.

## Prepare and start

```sh
./spark-serve pull qwen38-dual --node both
./spark-serve up qwen38-dual --node both --no-hermes --json
./spark-serve status --json
```

Preparation builds the ARM64 derivative on the head from the immutable base,
authenticates the cached NVIDIA snapshot or downloads that exact revision,
then checks the worker cache. Existing authentic snapshots are reused without
another checkpoint copy. If the worker needs assets, only the pinned snapshot
and its referenced Hugging Face blobs are copied from the head over the
configured QSFP interface. Normal cache symlinks are preserved, and a differing
worker cache path is supported through `cluster.worker_hf_cache_host`.
The identical image is streamed directly to the worker when needed.

Both nodes receive full checkpoint hashing, runtime source verification and
a CPU-only metadata/import compatibility check with GPUs hidden and networking
disabled. Preparation leaves serving workloads running and launches no model.
Runtime caches use a separate `spark-serve/qwen38-dual/runtime-cache` namespace.
It requires approximately 133 GB of checkpoint storage per node if that
snapshot is not already cached, plus image and compilation cache space.

For a cached head snapshot, `--skip-download` prevents Hub downloads and still
authenticates/distributes assets. `--image-only` prepares and verifies the
runtime image on both Sparks without a checkpoint readiness receipt.

Startup authenticates every checkpoint file and installed patch on both nodes
before replacing their current allocations. It then starts the headless worker
and the head API using the existing vLLM controller. `--node head` and
`--node worker` are rejected because this recipe requires both Sparks.
Allow time to hash about 133 GB per node and load the distributed model;
the 3,600-second readiness deadline is a limit, not a measured boot time.

## Qualification commands

Run after both ranks are ready. Use the head's catalog HTTP hostname.

```sh
python3 tools/qwen_acceptance.py \
  --url http://sparkone.local:8000/v1 --model qwen3.8-flash-next-dual \
  --thinking --timeout 600 --concurrency \
  --output diagnostics/qwen38-dual/acceptance-thinking

python3 tools/qwen_acceptance.py \
  --url http://sparkone.local:8000/v1 --model qwen3.8-flash-next-dual \
  --no-thinking --timeout 600 --concurrency \
  --output diagnostics/qwen38-dual/acceptance-fast

python3 tools/qwen_benchmark.py \
  --url http://sparkone.local:8000/v1 --model qwen3.8-flash-next-dual \
  --samples 2 --timeout 180 --deadline-seconds 600 --concurrency \
  --output diagnostics/qwen38-dual/benchmark-fast
```

Record image IDs on both nodes, checkpoint revision, exact flags, BF16 SSM/KV
settings, actual retrieval prompt tokens, reasoning/tool/vision results,
speculative acceptance, TTFT, decode rate, memory headroom and concurrency.
A configured 262K ceiling does not establish a passed 262K retrieval test.

## Attribution

The TP-aware patch generators and draft vocabulary are vendored byte for byte
from Mia'a AI Lab's dual-Spark integration at the pinned revision. Their
[AGPLv3 license](../recipes/qwen38-dual/runtime/upstream/LICENSE) is retained.
The upstream integration credits
[getrefined's Qwen Spark recipe](https://github.com/getrefined/Qwen3.8-Flash-Next-NVFP4-vLLM-DGX-Spark).
Checkpoint licensing remains governed by the NVIDIA model card and its
upstream Qwen terms.
