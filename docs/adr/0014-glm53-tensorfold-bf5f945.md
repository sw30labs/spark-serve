# ADR 0014: Re-vendor GLM TensorFold to recipe bf5f945, keep its weights

Accepted · Recorded 2026-10-25

**Context.** `glm53-tensorfold` vendored MiaAI-Lab's recipe at `33b50fde`
(82 patches, published image `v0.6.0-31557ed1cef6`). Upstream `main` is now
[`bf5f9452`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/bf5f94523a8251029cedae86bf61f8ce37fac218),
152 commits later, with 123 patches (0084 to 0122, 0128, 0129) and the
published image `v0.6.0-789e8f3bdfa2` pinned by digest
`sha256:99fe7496d7002fd8f2ca3b1beb0aa07513835cd09989ca657b65875d60f69956`.
TensorFold itself is unchanged (v0.6.0, `c4646171`), but the image's pip
resolve moved `transformers` from 5.18.0 to 5.20.1. Hugging Face `main` of the
target moved from `078455ff` to `76c0b517`, and the scout proposed pinning it.

**Decision.** Vendor the 123 patches and refreshed notices, derive from the
new digest, and authenticate the 445 installed files, which were taken from
the three new image layers and matched against TensorFold `c4646171` with all
123 patches applied. Keep the target at `078455ff` and DFlash2 at `bf582e4e`.
Upstream `bf5f9452` still pins `078455ff`, and `76c0b517` changes only
`LICENSE` and `README.md` and adds `LICENSE-GLM-5.3-Flash`; every weight and
config file is byte-identical. Moving would only replace a verified 178 GB
manifest and both Sparks' snapshots. Mirror the rank environment upstream's
`scripts/config.sh` now exports:
- The picture budget becomes `MAX_IMAGES x IMAGE_TOKENS` = 102,400 tokens with
  a 1,440 MiB vision workspace, instead of TensorFold's shared 16,384, so
  pictures past the 8th no longer shrink earlier ones or invalidate the kept
  prompt.
- `NUCLEUS_UNION=1`, a 15 s SSE keep-alive, and `FILL_ROWS=2048`.
- Explicit off values for kept-state limits, effort tail, decode order and
  spill quota and turns.

Spill blocks (`SPILL_GIB=0`) and KV sharding (`TF_GLM_KV_SHARD` unset) stay
off, and broken-engine exit stays opt-in. The local derivative becomes
`spark-serve-glm53-tensorfold:v060-bf5f945`.

**Trade-off.** The picture budget costs about 1.7% of the KV pool on two
Sparks (upstream's measurement). The private `models.toml` must take the new
image name in the same change that lands this branch, or preparation refuses
it, and the existing `v060-33b50fd` derivative stops matching. The new image
is unqualified on this hardware until it has been prepared on both Sparks and
benchmarked. Upstream speed and quality claims (spill, KV sharding, picture
turns) are the author's. DFlash2 (`incoai/GLM-5.3-Flash-DFlash2`) remains
**CC BY-NC-ND 4.0, non-commercial only**; `drafter = "mtp"` with
`parallel = 1` avoids it. The target's licence notice now says Apache-2.0 for
the quantization, with Z.AI's MIT kept for the base model. Only the notice
changed, not the pinned files.

**Evidence.** [Source pins](../../recipes/glm53-tensorfold/source-pins.json),
[installed runtime](../../recipes/glm53-tensorfold/runtime/installed-runtime.json),
[rank environment](../../spark_serve_tensorfold.py), [runbook](../glm53-tensorfold.md),
[earlier TensorFold decision](0011-tensorfold-recipe.md),
[HF revision diff](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/commits/main).
