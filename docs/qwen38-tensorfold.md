# Qwen3.8 Flash Next on TensorFold

Experimental single-Spark recipe, managed by the existing Spark Serve app and
controller. Existing Qwen recipes remain available. This uses an MLX-format
checkpoint through TensorFold's **CUDA** backend on GB10.

| Setting | Default |
|---|---|
| Catalog key | `qwen38-tensorfold` |
| Served model | `qwen3.8-flash-next-tensorfold` |
| Runtime | TensorFold v0.3.6.3 with pinned MiaAI-Lab patches |
| Checkpoint | `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` |
| Context | 262,144 tokens per stream, including the reply |
| Concurrent streams | 4 |
| KV cache | int8; 1,048,576 tokens across the four streams |
| Drafting | MTP, up to 6 drafts, confidence 0.60 |
| Vision / thinking | Enabled |
| N-gram tables | Read from SSD |

## Prepare and start

Merge `[models.qwen38-tensorfold]` and its `tensorfold` section from
`models.example.toml` into your private catalog. The app lists the recipe after
Refresh. Preparation uses the selected Spark's configured cache path.

```sh
./spark-serve pull qwen38-tensorfold --node worker
./spark-serve up qwen38-tensorfold --node worker --no-hermes --json
```

Use `--node head` for Spark one. This is a separate approximately 106 GiB
checkpoint; the existing NVIDIA NVFP4 weights and PLE cache are not reusable.
Upstream budgets about 160 GB of free disk for a fresh checkpoint and image.
Leave additional room for the derivative image and compiled kernel cache.
The four-stream setting has an upstream startup estimate around 98 GiB, before
some process overhead; that is not a local memory measurement or guarantee.

Preparation authenticates the immutable image and checkpoint without starting
a model. Start repeats offline verification before disrupting the selected
workload. The other Spark's independent model stays running; an existing
two-Spark allocation still requires an explicit two-node stop before a solo
start. First launch compiles CUDA kernels; later launches reuse the recipe's
own cache. Spark Serve owns the container lifecycle.

Once ready, use the app's **Use in Hermes (Mac)** or **Use in Hermes (Spark)**:

```sh
./spark-serve use --node worker --hermes-target local
./spark-serve use --node worker --hermes-target spark
```

The second command configures Hermes on the cluster head to use the worker's
endpoint. Selecting a client does not move the model.

## Monitoring and benchmarks

Readiness requires the owned container, the exact served model, and a valid
TensorFold `/health` response. Live inference monitoring reads the reported
in-flight request count and prompt/completion counters from `/health`. Prompt
tokens are credited when the first output arrives, so their interval rate is
not instantaneous prefill speed. Waiting requests, KV usage,
request rate, TTFT and TPOT are unavailable in this pinned health schema.
Host resource charts still work; native benchmarks measure client-observed
timings and use the server's streamed token accounting.

```sh
./spark-serve bench run --node worker --kind decode --concurrency 1 --requests 3 --no-thinking --json
./spark-serve bench run --node worker --kind decode --concurrency 4 --requests 8 --no-thinking --json
./spark-serve bench run --node worker --kind prefill --prompt-tokens 4096 --no-thinking --json
```

## Qualification and limits

Local hardware qualification is **pending**. Offline tests cover preparation,
launch validation, readiness, metrics and native benchmark eligibility. They
do not establish speed, model quality, working vision or Hermes compatibility.
The generated serving arguments also passed the pinned engine's actual parser
and family checks against published checkpoint metadata without loading CUDA.
Before daily use, exercise thinking on/off, tools with typed arguments, long
file writes in Hermes, images, matched C1/C2/C4 benchmarks, and 200K+ retrieval
while recording RAM and swap. Compare identical prompts and settings against
the existing NVFP4 recipe; both runtime and quantization differ.

The existing synthetic acceptance runner can check the common chat API after
startup. Set the URL to the selected Spark's endpoint:

```sh
python3 tools/qwen_acceptance.py --url http://<spark>:8000/v1 \
  --model qwen3.8-flash-next-tensorfold --thinking --concurrency \
  --output diagnostics/qwen38-tensorfold/thinking
```

Repeat with `--no-thinking`. Those fixtures do not execute real Hermes tools.
TensorFold's pinned CUDA server buffers tool-call arguments until generation
finishes; large writes can cause client idle timeouts. The
[proposed streaming fix](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/pull/2)
is not included. Top-level `reasoning_effort`, generation penalties and
`logprobs` are not equivalent to vLLM support; use documented TensorFold request
fields and verify the client behavior you need. Image/video messages use data
URLs by default.

## Provenance

Recipe: [MiaAI-Lab, revision a3aa898](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/tree/a3aa89835022c55ca8e55008c37785954834e04f).
Engine: [Ash Hart's TensorFold v0.3.6.3](https://github.com/ashhart/TensorFold/tree/v0.3.6.3).
Checkpoint: [Vontra, revision dadefa8](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP/tree/dadefa8066e3be900a0d148d0f5a2f4eb1cf6534).

Immutable image identity, checkpoint revision, runtime hashes, and retained
license notices live in [the recipe directory](../recipes/qwen38-tensorfold/).
The [checkpoint manifest](../recipes/qwen38-tensorfold/model-source.json)
records publisher hashes. Upstream performance reports describe upstream
measurements; Spark Serve does not claim those results for this installation.
