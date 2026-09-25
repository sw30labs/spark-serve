# MiMo-V2.6-Flash-RL

Official `XiaomiMiMo/MiMo-V2.6-Flash-RL` checkpoint on both Sparks, tensor-parallel 2.
Revision `5711b268169967567844e1e560e8a3966da959b1`. The architecture allows 1M tokens.
This catalog serves **300,000**, which is the context that has a published GB10 measurement.

One Spark cannot hold the checkpoint. Weights are about 166 GiB. At TP2 each rank
holds about 83 GiB. Preparation downloads once on the head and copies that tree
to the worker over the QSFP addresses on `enp1s0f1np1` (192.168.100.10 to .11).

```bash
./spark-serve pull mimo26
./spark-serve up mimo26 --no-hermes
```

`pull` builds `spark-serve-mimo-v26-flash:0.1.0` on the head from
`ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2` and loads that image on
the worker. The derivative copies three MIT-licensed GB10 fixes into vLLM
(fused FP8 QKV sharding, the omni class marked for DFlash, and FP8 KV in the
DiffKV backend) and installs `soundfile` and PyAV for audio input. The DFlash
value-scale patch is not applied; the upstream bench measured no change.
DeepGEMM stays off. MoE uses Marlin.

The checkpoint's `generation_config.json` sets `do_sample: false` and
`max_new_tokens: 2048`. The launch uses vLLM's own generation defaults and then
sets temperature 1.0, top_p 0.95, and repetition_penalty 1.05, so a client that
sends no sampling parameters still samples, and 2048 is not a server-wide cap.
Thinking is off unless a request passes `chat_template_kwargs.enable_thinking`.

NCCL stays on this pair's right-port QSFP rails from `[cluster.nccl]`. The
container is `vllm_mimo` on port 8000. Served name: `mimo-v2.6-flash`.

This pair uses a GPU memory fraction of 0.85. The published GB10 run used 0.90.
After the weight copy, CUDA still reported only about 1 GiB free because the
page cache held the checkpoint, and this account cannot drop that cache
without a sudo password. 0.85 still covers the 300K window: the same recipe's
notes put a 1M sequence at 14.3 GiB of FP8 KV, and 0.85 left about 13.8 GiB.

Upstream, on 2026-09-22, the 0.90 flag set (fp8 KV, DFlash 7, 300K) was
measured at about 45 tokens/s for one stream and about 156 aggregate at six,
with text, image, video, and audio, and with needle checks at 100K and 250K.
Those numbers are theirs, not a measurement of this pair. Cold start there was
about 11 minutes. A 1M request was not run; it needs about 3.3 times the
per-request KV blocks reserved at 300K.
