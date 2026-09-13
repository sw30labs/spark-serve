# Nemotron Super NVFP4 preparation

This image adds a CPU-only checkpoint verifier to the pinned official vLLM ARM64
image. It does not patch vLLM, copy model weights, or install a reasoning plugin.
See [the deployment guide](../../docs/nemotron-super.md) for source attribution,
configuration, preparation, and independent-node operation.

`model-source.json` contains published file lengths and Git/LFS digests from
NVIDIA's public Hugging Face checkpoint at the exact pinned revision. Setup
authenticates every file; startup authenticates serving metadata and checks
large weight-shard lengths. `--skip-download` never attempts repairs or network
downloads of weights. The image build may still pull its pinned base image if absent.
