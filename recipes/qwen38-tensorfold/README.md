# Qwen3.8 TensorFold recipe

Experimental, single Spark: four streams, int8 KV cache, 262,144 tokens each.
The pinned Vontra MLX checkpoint is separate from the NVIDIA NVFP4 recipes.

`source-pins.json` fixes the Linux ARM64 image digest, checkpoint revision and
recipe files. `model-source.json` records every publisher file size and LFS
SHA-256 or Git blob digest. Preparation and startup verify the full checkpoint.

The derivative image only adds verification and an entrypoint. Its 360
TensorFold runtime files are checked against `runtime/installed-runtime.json`
at build, preflight and launch. Those hashes were independently matched to
TensorFold commit `191188075bca56a7c71074a79375eb4c1cb22e1c`, with the nine ordered
patches from Mia's recipe commit `a3aa89835022c55ca8e55008c37785954834e04f`, and
the actual application layers of the pinned image. Preparation never starts or
stops a serving workload.

The nine patches and original MIT notices are retained in `runtime/upstream`.
`TENSORFOLD-LICENSE` and `TENSORFOLD-THIRD_PARTY_NOTICES.md` retain TensorFold's
notices; the remaining dependency notices stay in the upstream image. NVIDIA's
entrypoint and license banner are retained. Model weights carry their own Qwen
Community License 1.0 and are downloaded only during explicit preparation.

No language-draft patch or unmerged tool-streaming patch is included. Real
Hermes tools, long context, vision and matched benchmarks still need hardware
qualification; successful preparation establishes integrity, not model quality.
