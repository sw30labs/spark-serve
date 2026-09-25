# MiMo-V2.6-Flash-RL image

`Dockerfile` starts from `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2`
and copies the three GB10 fixes in `patches/`. See `NOTICE` for the upstream
commit. `tools/prepare_mimo26.py` downloads the pinned Hugging Face revision on
the head, copies it to the worker over the QSFP link, builds this image, and
loads it on the worker.

```bash
./spark-serve pull mimo26
```

The runbook is [docs/mimo-v26-flash.md](../../docs/mimo-v26-flash.md).
