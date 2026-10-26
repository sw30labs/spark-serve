# Flint NVFP4 preparation (single Spark)

Prepared only. Nothing in this directory has been served on a Spark yet; see
[the runbook](../../docs/flint-nvfp4.md) and [ADR 0016](../../docs/adr/0016-flint-nvfp4-single-spark.md).

- `model-source.json`: file lengths and Git/LFS digests published on the Hub for
  `RecursiveSI/flint-nvfp4-gguf@6ec43e19` (the 22.9 GB GGUF) and the config and
  tokenizer files of `RecursiveSI/flint@2a247be0` that vLLM needs beside the GGUF.
- `source-pins.json`: checkpoint revisions, llama.cpp tag, vLLM image digest.
- `verify.py`: CPU-only preflight, `--full` adds SHA-256 of the GGUF.

No `models.toml` entry yet: the controller would offer a Start button that
stops the dual-Spark Qwen, and neither lane has been qualified on GB10.
