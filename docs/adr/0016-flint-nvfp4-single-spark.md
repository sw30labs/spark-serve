# ADR 0016: Prepare Flint from the official NVFP4 GGUF on one Spark

Accepted · Recorded 2026-10-25

**Context.** Pedro asked for Recursive Flint (`RecursiveSI/flint@2a247be0`, 39.8B
total / ~2.5B active MoE) on the Sparks in NVFP4. Recursive publishes its own
NVFP4 checkpoint, `RecursiveSI/flint-nvfp4-gguf@6ec43e19` (22.9 GB, Apache-2.0),
which vLLM describes as quantization-aware trained and whose card lists DGX Spark
among the GPUs its llama.cpp runtime was tested on. Community NVFP4 builds exist
(`FutureMLS-Lab/flint-nvfp4-coldgptq`, ModelOpt W4A4 PTQ validated on B300 SGLang;
`HawkBearPig/flint-*`, custom mixed layouts the authors say are not drop-in for
any engine). vLLM's Flint kernels (PR #62938) target the SM100 family; GB10 is
sm_121, which the PR says falls back to vLLM's standard layers. The dual-Spark
Qwen serves Nightshift and needs ~102 GiB MemAvailable on each Spark, so Flint
cannot run beside it.

**Decision.** Use the official NVFP4 GGUF; do not run our own quantization.
Download it once to sparkone's Hub cache, copy it to sparktwo over QSFP and
verify it against the published SHA-256. Prepare two single-Spark (TP1) lanes:
llama.cpp `b11889` built for `121a-real`, and the pinned `vllm/vllm-openai:flint`
arm64 image. Qualify both by daytime measurement before adding a `models.toml`
entry; choose the lane from the measurements, not from published B300/GB200 numbers.

**Trade-off.** Flint on a Spark displaces the dual-Spark Qwen for the test window,
which must close with Qwen restored before 8 PM ET. The vLLM lane on sm_121 runs
stock layers, not the published Flint kernels, so GB200 speeds do not transfer.
llama.cpp's Flint path carries an activation-scaling workaround its PR says
costs 10–30%. The community W4A4 ModelOpt build stays a fallback only.

**Evidence.** [Runbook](../flint-nvfp4.md), [recipe](../../recipes/flint-nvfp4/),
[tests](../../tests/test_flint_preparation.py),
[NVFP4 card](https://huggingface.co/RecursiveSI/flint-nvfp4-gguf),
[vLLM PR #62938](https://github.com/vllm-project/vllm/pull/62938),
[llama.cpp PR #30928](https://github.com/ggml-org/llama.cpp/pull/30928).
