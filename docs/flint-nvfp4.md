# Flint NVFP4 on one DGX Spark

Status 2026-10-25: **prepared, not served.** Weights, config/tokenizer, a llama.cpp
build and the vLLM Flint image are staged; no Flint process has touched a GPU.
Decision record: [ADR 0016](adr/0016-flint-nvfp4-single-spark.md).

## Sources

| What | Pin | Notes |
|---|---|---|
| NVFP4 weights | `RecursiveSI/flint-nvfp4-gguf@6ec43e19b565fc8d560e2d04985d7724f20673bf` | one 22,863,084,160-byte GGUF, SHA-256 `97cc8009…8bde`, Apache-2.0. Card: llama.cpp CUDA "tested on … RTX Spark, and DGX Spark", ≥48 GB recommended. |
| Config + tokenizer | `RecursiveSI/flint@2a247be0c0aec895ccb0ac7ca92229ec6df37288` | 7 small files only (no safetensors); vLLM reads them via `--hf-config-path` / `--tokenizer`. |
| llama.cpp | tag `b11889` (Flint since `b11803`, PR #30928) | built with `-DCMAKE_CUDA_ARCHITECTURES=121a-real`, CUDA 13.0. |
| vLLM | `vllm/vllm-openai:flint`, arm64 `sha256:5641d14b…c4bc` | day-0 image; Flint merged upstream in PR #62938 on 2026-10-22. |

Published speeds (GB200 ~485 tok/s batch 1, vLLM) come from SM100 kernels. PR #62938:
"other configurations use vLLM's standard layers: TP > 1, EP, and GPUs outside the
SM100 family such as sm_120". GB10 reports compute capability 12.1, so expect the
stock-layer path. Measure; do not assume.

## Where things are (both Sparks hold the same Hub cache copy)

```
/home/spider/.cache/huggingface/hub/models--RecursiveSI--flint-nvfp4-gguf/snapshots/6ec43e19b565fc8d560e2d04985d7724f20673bf/flint-nvfp4.gguf
/home/spider/.cache/huggingface/hub/models--RecursiveSI--flint/snapshots/2a247be0c0aec895ccb0ac7ca92229ec6df37288/
sparkone:~/flint-prep/llama.cpp-b11889/build/bin/llama-server
docker image spark-serve-flint-vllm:day0-5641d14b   (sparkone only)
```

Preflight (CPU only): `python3 recipes/flint-nvfp4/verify.py <gguf snapshot> <flint snapshot> [--full]`.

Staged 2026-10-25 (ET): downloaded once to sparkone, rsynced to sparktwo over
QSFP (192.168.100.11, 22.88 GB at ~610 MB/s). `verify.py --full` returned
`{"ok": true, "bytes": 22874444208}` on both Sparks; `flint-nvfp4.gguf` SHA-256
`97cc8009ff9b132d5cf3c488e6767962bb5a5dc158a90ed661bbbddf876b8bde` matches the
Hub LFS pointer. On sparkone `~/.cache/huggingface` is a symlink to
`~/models/huggingface`; same files either way. The vLLM image (25.7 GB
unpacked) and the llama.cpp build exist on sparkone only, so the first
qualification runs there. The llama.cpp web UI asset download failed during the
build; the HTTP API is unaffected.

## Daytime qualification (not before the Switchyard runs finish; Qwen back before 8 PM ET)

Flint needs one Spark; the dual-Spark Qwen needs ~102 GiB MemAvailable on each, so
stop it for the window and restore it afterwards.

1. `./spark-serve stop` (from the Mac). Confirm both Sparks idle.
2. llama.cpp lane on sparkone (card's flags, LAN-bound to the Spark only):
   ```
   ~/flint-prep/llama.cpp-b11889/build/bin/llama-server -m <gguf> --alias flint \
     -ngl all -c 262144 -np 1 -fa on -ctk f16 -ctv f16 --jinja \
     --temp 1 --top-p 0.95 --top-k 20 --min-p 0 --host 127.0.0.1 --port 8080
   ```
   Smoke test, then `llama-bench` and a served tok/s run per ADR 0007.
3. vLLM lane on sparkone (stop llama.cpp first):
   ```
   docker run --rm --gpus all --ipc=host -p 127.0.0.1:8001:8000 \
     -v /home/spider/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
     spark-serve-flint-vllm:day0-5641d14b \
     /root/.cache/huggingface/hub/models--RecursiveSI--flint-nvfp4-gguf/snapshots/6ec43e19b565fc8d560e2d04985d7724f20673bf/flint-nvfp4.gguf \
     --hf-config-path /root/.cache/huggingface/hub/models--RecursiveSI--flint/snapshots/2a247be0c0aec895ccb0ac7ca92229ec6df37288 \
     --tokenizer /root/.cache/huggingface/hub/models--RecursiveSI--flint/snapshots/2a247be0c0aec895ccb0ac7ca92229ec6df37288 \
     --served-model-name flint --max-model-len 262144 --gpu-memory-utilization 0.6 \
     --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm47 \
     --exclude-tools-when-tool-choice-none
   ```
   Local paths, because the cache was filled by revision and has no `refs/main`.
   Unverified on GB10: the sm_121 fallback, `gguf>=0.19.0` inside the image,
   and loading the GGUF by file path (the recipe says repo ID or local path).
4. Restore Qwen: `./spark-serve up qwen38-dual-tensorfold --node both`, wait for
   `/health`, confirm `qwen3.8-flash-next-dual-tf` answers.
5. Only then add a `[models.flint-nvfp4]` entry for the faster qualified lane.

Budget about 2 hours for steps 1–4, mostly Qwen's cold start (ready timeout
1,800 s) and the first vLLM start; not measured yet.
