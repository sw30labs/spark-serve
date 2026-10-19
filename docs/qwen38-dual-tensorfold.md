# Qwen3.8-Flash-Next on two Sparks with native TensorFold

This integrates [MiaAI-Lab's recipe](https://mia-ai.net/models/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold)
as `qwen38-dual-tensorfold`. It runs TensorFold's Zig engine, `tensorfold-native`, with no Python in the serving
path. Spark Serve starts and owns both containers through its normal fenced lifecycle, not upstream's `start.sh`.
The `qwen38-dual` vLLM recipe is separate. Both recipes serve the same NVIDIA checkpoint, so it is stored once on
each Spark.

| Setting | Catalog default |
|---|---|
| Sparks / tensor parallel | 2 / 2: each rank holds half of the attention and DeltaNet heads |
| Served model | `qwen3.8-flash-next-dual-tf` |
| Engine | `tensorfold-native`, TensorFold `zig-flashnext` at `db28187` with 9 MiaAI-Lab patches |
| Checkpoint | `nvidia/Qwen3.8-Flash-Next-NVFP4` at `fc694b54`, with all 10 routed experts per token |
| Context per request | 1,048,576 tokens, using Qwen's YaRN x4 past the native 262,144 |
| Concurrent streams | 16; configurable from 1 to 16 |
| KV cache | FP8 (lossy, about 1.8x the pool); `bf16` is exact |
| Drafts | The checkpoint's own MTP head; every drafted token is verified |
| Rank link | NCCL over RoCE, plus RoCE one-shot all-gathers |
| API | Cluster head, `cluster.port` (normally 8000) |
| Rendezvous | Head's fabric address, port 29551 |
| Inputs | Text, up to 50 images and 4 videos per request, and tools; remote media URLs disabled by default |

## Prepare and start

Merge `[models.qwen38-dual-tensorfold]` and its `tensorfold` section from `models.example.toml` into a private
catalog. The app lists it under Qwen after Refresh. Both Sparks need Docker with the NVIDIA runtime, key-based
SSH, a connected RoCE fabric, and the fabric settings in `[cluster.nccl]`.

```sh
./spark-serve pull qwen38-dual-tensorfold --node both
./spark-serve up qwen38-dual-tensorfold --node both --no-hermes --json
```

Preparation pulls the immutable upstream image, about 11.4 GB compressed, on the head. It builds a small local
derivative and copies the exact image to the worker over the QSFP link. The NVFP4 snapshot, 25 files and about
132.7 GB, is fully hashed on each node and reused when it is already authenticated. Only a missing or damaged
snapshot is downloaded on the head or copied to the worker. Each rank's exact argv then goes through the native
parser without a GPU or network, before either readiness receipt is published. Serving workloads keep running
throughout preparation.

Start checks both nodes' image identity, all 933 installed TensorFold files, full checkpoint hashes and the
native parser before stopping a workload. It starts rank 1 first, then rank 0. Readiness requires both owned
containers, the exact served model in `/health`, and graph warming to have finished.

## Memory

On GB10 unified memory, exhausting memory freezes the host instead of failing an allocation. Each rank
therefore refuses to start, with exit code 75, unless `MemAvailable` reaches upstream's measured floor. That
floor is 48 GiB fixed, plus KV cache for the window (9 GiB per million tokens in FP8, 17 in BF16), plus 31 GiB
per million prompt tokens while prefilling. Vision adds 2 GiB, the engine's 10 GiB reserve and 2 GiB of slack
complete the sum. The default 1M FP8 window needs **102 GiB per Spark**. A 262,144-token BF16 window without
vision needs 72 GiB. Stop other GPU work or lower `max_model_len` when a rank refuses.

## Settings

The `tensorfold` table supports `parallel`, `kv_dtype` (`fp8` or `bf16`), `drafts`, `vision`, `vision_urls`,
`max_images`, `max_videos`, `thinking`, `max_tokens`, `roce` and `master_port`. Keep `max_num_seqs` equal to
`parallel`. When `max_model_len` exceeds 262,144, YaRN is enabled automatically. Only the fabric interface,
HCAs, GID and optional debug level are taken from `[cluster.nccl]`; TensorFold uses upstream's four NCCL
channels. Direct Docker, mount, environment and vLLM overrides are rejected before a model switch. Upstream's
INT4-AutoRound quant, NFS worker weights and API-key mode are not part of this integration.

The native server's `/health` has no request counters. The live metrics monitor therefore reports this model's
inference metrics as unavailable, even though the native `/metrics` endpoint exists. Benchmarks work through
the OpenAI API.

## Provenance and qualification

The [source pins](../recipes/qwen38-dual-tensorfold/source-pins.json) record upstream commit
`be3ed857ab1055ca47b36134d22691f00f2bb94a` (v1.1) and image
`ghcr.io/miaai-lab/qwen3.8-flash-dual-dgx-sparks-tensorfold@sha256:e188128d…`. That image carries upstream's
patch hash, `f3e1300f4f62`, which was recomputed here from the pinned patches, Dockerfile and build inputs.
The [runtime inventory](../recipes/qwen38-dual-tensorfold/runtime/installed-runtime.json) was extracted from the
image's three authenticated TensorFold layers. Its 468 Python vision-helper files and 6 license files match
TensorFold `db28187` with the nine patches applied, byte for byte. The Zig binary, fatbins and sm121 kernel set
are pinned by the image digest; they were not rebuilt locally. The nine patches are pinned by SHA-256 rather than
vendored. Upstream's and TensorFold's license and notice files are retained in the
[recipe directory](../recipes/qwen38-dual-tensorfold/).

The recipe and TensorFold are Apache-2.0, and TensorFold code written before 0.6.0 keeps its MIT notice. The
NVFP4 checkpoint is governed by the NVIDIA Open Model License, with the Qwen Community License 1.0 as additional
information. NVIDIA's container terms apply to the base image.

Preparation completed on both Sparks on 2026-10-08. Both nodes have the identical ARM64 image
`sha256:b565e72f14bf828b4f537b1032f147cebd1a0bf8ed387aa76851929d506b03a1`. All 933 installed TensorFold files
were authenticated in the real image on each node. The existing NVFP4 snapshot passed full hash verification on
both nodes and was reused, so nothing was downloaded or copied. The native parser accepted the exact rank 0 and
rank 1 arguments. No model was launched, and existing services kept running.

Distributed GPU serving, tools, vision, long-context quality and performance qualification remain **pending**. Upstream reports 67.4 tok/s for one request and 390.1 tok/s combined at 16
requests with NVFP4; those are the author's measurements, not local results. FP8 KV and YaRN change arithmetic,
and exact draft verification does not make them lossless.
