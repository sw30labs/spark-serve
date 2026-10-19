# Qwen3.8-Flash-Next with native TensorFold on two Sparks

See [the setup and qualification guide](../../docs/qwen38-dual-tensorfold.md).

This recipe follows [Mia's dual-Spark TensorFold recipe](https://mia-ai.net/models/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold)
at source commit `be3ed857ab1055ca47b36134d22691f00f2bb94a` (v1.1). It serves the original NVIDIA NVFP4
checkpoint, which `qwen38-dual` already uses, through TensorFold's Zig engine (`tensorfold-native`, branch
`zig-flashnext` at `db281878ddb836fd0df510d8771ecb7e0fe47d26`) with the recipe's nine patches.

`source-pins.json` identifies the immutable ARM64 image, upstream and TensorFold revisions, patch hashes and
every bundled runtime and preparation file. `model-source.json` is byte-identical to the `qwen38-dual` manifest.

`runtime/installed-runtime.json` lists all 933 files under `/opt/tensorfold`. They were extracted from the
three SHA-256-authenticated TensorFold layers of the pinned image. The 468 vision-helper Python files and 6
license files match TensorFold `db28187` with the nine patches applied, byte for byte. The Zig binary, fatbins
and sm121 kernel set cannot be rebuilt reproducibly here, so only the image digest pins them. The image's
`tf.patches` label, `f3e1300f4f62`, is upstream's hash of exactly these patches, its Dockerfile and build inputs.

The upstream patches (6.9 MB) are pinned by SHA-256 in `installed-runtime.json`, not vendored. Upstream's
Apache-2.0 `LICENSE`, `NOTICE` and `CREDITS.md`, and TensorFold's license and notices, are retained in
`runtime/upstream`. Changes from upstream: the local derivative image adds only verification files, an entrypoint
that checks free memory before `exec`ing upstream's NVIDIA entrypoint, and these notices. Startup is driven by
Spark Serve, not by upstream's `start.sh`.

Only the default NVFP4 quant is pinned. The INT4-AutoRound alternative is excluded.
