# Qwen dual-Spark recipe assets

See [the setup and qualification guide](../../docs/qwen38-dual.md).

`source-pins.json` authenticates the immutable ARM64 vLLM base, upstream
dual-Spark revision, local runtime inventory and checkpoint manifest.
`model-source.json` reuses the original NVIDIA snapshot used by the existing
single-Spark Qwen recipes. `runtime/upstream-files.json` records each vendored
TP-aware generator and vocabulary hash; the upstream license is retained.

Preparation and checkpoint/blob distribution are implemented by
`tools/prepare_qwen38_dual.py`. Startup uses the existing vLLM controller with
TP2, EP, MTP3 and vision encoder replication. Local GPU qualification is pending.
