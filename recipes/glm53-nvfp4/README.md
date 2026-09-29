# GLM-5.3-Flash NVFP4 preparation

This recipe stages the **exact NVIDIA NGC NVFP4 artifact** associated with the
official NIM deployment, not the separately published Hugging Face quantization.
The NGC artifact uses ModelOpt 0.45.0 weight-only NVFP4; the HF artifact uses a
different quantization configuration and shard layout. They are not substituted.

| Component | Pin |
|---|---|
| NIM base image | `nvcr.io/nim/zai-org/glm-5.3-flash:2.1.2-variant` |
| Linux ARM64 base manifest | `sha256:0bd2a1f4ffacf4ee61e1f8c1b48071a6713b51bad954ab0ca627c24be84f43f8` |
| Serving image ID | `sha256:161214462d6696699210c0aef16b2ba706e5a8ad692e2c9490aaa70d190e794c` |
| Local patch revision | `marlin-gate-up-maxglobal-e4m3-v1` |
| NGC model | `nim/zai-org/glm-5.3-flash` |
| NGC version | `nim-aa28e1f-nvfp4` |
| Files / weight shards | 130 / 120 |
| Total checkpoint size | 194,692,707,910 bytes (181.32 GiB) on each Spark |

The public NGC file listing supplies SHA-256 for every file. Both listing pages
were combined and checked against its total count and byte count. The manifest
contains those publisher digests, not checksums inferred from downloaded weights.
The checkpoint stays unchanged. Preparation pulls the immutable ARM64 base and
builds a small local runtime correction. This is not an official NVIDIA patch.

The base image's Marlin loader wrongly shared the gate global scale with the up
projection when those checkpoint scales differed. The correction uses each
expert's maximum global scale and rescales its gate/up block scales, explicitly
rounding them back to E4M3 before Marlin conversion. This adds block-scale
quantization error and does not preserve effective weights exactly. It retains
Marlin; there is no Cutlass workaround in this recipe.

Across 16 sampled expert pairs and 32,768 blocks, maximum relative scale error
fell from **42.857% to 5.455%**, and mean error from **5.628% to 0.673%**. These
are scale-error measurements, **not model-accuracy results**; they exclude
original quantization and Marlin activation/global BF16 rounding. See the
[numerical receipt](../../diagnostics/2026-09-19-glm53/quantization-review/checkpoint-scale-validation.json).
Earlier unpatched API and isolated Hermes smoke passes were insufficient for
adoption. The corrected image subsequently passed API, coding, vision, actual
Hermes, 95,144-token retrieval and four-request concurrency checks with the
catalog's 0.92 memory fraction. See the [qualification record](../../docs/glm53-nvfp4.md)
for measured latency, memory, the dense-attention fallback and remaining limits.

The [Dockerfile](Dockerfile) uses one CPU-only build step, no build-step network,
normalized file timestamps and `SOURCE_DATE_EPOCH=0`. The
[patcher](patch_runtime.py) checks original/helper/output hashes and replaces
Python caches with deterministic checked-hash bytecode. Independent builds with
Docker 29.2.1 and BuildKit 0.27.1 produced the same image ID on both Sparks; see
the [build receipt](../../diagnostics/2026-09-19-glm53/runtime-build-both.json).
Preparation authenticates all three build files against `source-pins.json` and
checks the resulting image ID, Linux ARM64 architecture and `nvs:1000` user before
publishing a receipt with image and patch provenance.

```bash
./spark-serve pull glm53
python3 tools/prepare_glm53.py --catalog models.toml
python3 tools/prepare_glm53.py --catalog models.toml --node worker
python3 tools/prepare_glm53.py --catalog models.toml --skip-download
```

Preparation preserves existing serving workloads. It uses each node's configured
Hugging Face cache root (`worker_hf_cache_host` overrides the worker location),
but keeps this NGC artifact in a separate directory:

```text
<cache>/spark-serve/glm53-nvfp4/
  verify.py
  model-source.json
  source-pins.json
  prepared.json
  runtime-cache/
  models/nim-aa28e1f-nvfp4/
```

Downloads are public and keyless. Interrupted downloads retain `.incomplete`
files and resume using HTTP Range. Completed files are authenticated before
atomic publication. Rerunning preparation hashes existing files and repairs a
corrupt file through a new authenticated download. `--skip-download` performs
full verification without repairing or downloading model files; it still builds
and verifies the pinned derivative image. The tool requires host Python 3.11+
and Docker with BuildKit;
checkpoint download and verification need only Python's standard library.

Startup preflight mounts the cache read-only and runs `verify.py` using the
prepared NIM image, with no GPU or network. It validates every file's exact size
and hashes all files up to 32 MiB, including configuration, tokenizer, processor,
chat template and shard index. Full shard SHA-256 verification occurs during
preparation. `--full-hash` or `--sha256` requests a complete recheck explicitly.
Controller preflight additionally passes `--expected-manifest-sha256` with the
repository's pinned digest. A mismatched staged manifest fails before any
checkpoint file is inspected.

Mount the cache read-only at `/cache/huggingface`, set `NIM_MODEL_PATH` to
`/cache/huggingface/spark-serve/glm53-nvfp4/models/nim-aa28e1f-nvfp4`, and mount
the separate `runtime-cache` directory writable at `/opt/nim/.cache`. Startup
must use the immutable derivative image ID and NVIDIA's rank-zero-first NIM protocol.
The official Spark guide says not to override hardware transport environment
variables; the NIM hardware helper discovers those settings.

Text and image serving do not require FFmpeg. Video support additionally needs
the FFmpeg 8 runtime specified by NVIDIA and separate qualification.

Sources:

- [Official two-Spark deployment](https://docs.nvidia.com/nim/vision-language-models/latest/deploy-on-dgx-spark.html)
- [Official model-specific air-gap deployment](https://docs.nvidia.com/nim/vision-language-models/latest/get-started/advanced/get-started-glm-5-3-flash.html)
- [Pinned NGC artifact](https://catalog.ngc.nvidia.com/orgs/nim/zai-org/models/glm-5.3-flash/nim-aa28e1f-nvfp4/file-browser)
- [NGC file manifest](https://api.ngc.nvidia.com/v2/models/nim/zai-org/glm-5.3-flash/versions/nim-aa28e1f-nvfp4/files)

Model files retain the publisher's licenses and governing terms. Preparation
downloads their LICENSE and README alongside the weights. No large artifacts or
private host information are bundled in this repository.
