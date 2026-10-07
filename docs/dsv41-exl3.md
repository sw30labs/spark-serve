# DeepSeek-V4.1-Flash EXL3 on two Sparks

`ds41-exl3` serves `DeepSeek-v4.1-Flash-EXL3` on the head's port 8000.
It is a two-Spark vLLM trial of MiaAI-Lab's EXL3 2.9bpw recipe. Starting it
uses both Sparks and replaces their current serving workloads. The existing
`ds4` recipe remains the DeepSeek-V4-Flash path.

The launcher recipe is
[MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks)
at `6f7d1590ad49a2b8995188e45d7b9db31e677452`. Checkpoint pins are in
[model-source.json](../recipes/dsv41-exl3/model-source.json). Runtime pins are
in [source-pins.json](../recipes/dsv41-exl3/source-pins.json).

| Asset | Pin |
|---|---|
| Published image | `ghcr.io/miaai-lab/deepseek-v4.1-flash-exl3-2x-dgx-sparks@sha256:2f0cf3adc0f989c1d446be274df864eb799630175f604c3b22b71b7205971dce` |
| Image recipe stamp | `8c01a8543ff7e7b222ae1cd46b8be5051838dc7a7b075b4e647f3785c0429584` |
| Local image | `spark-serve-dsv41-exl3:0.1.0` |
| EXL3 checkpoint | `Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw` at `64ba41b6c916a587db06eae2e19b7845f7be6e6b` |
| Engram source | `deepseek-ai/DeepSeek-V4.1-Flash` at `2cba9e42aa026125f3ed06c6d98c1db82f7ca027` |
| EXL3 inventory | 49 files, 210,679,297,231 bytes |
| Served Engram | shards 47 and 48, native `config.json`, and a slim embed-only index |
| Served Engram bytes | 203,073,081,290 |
| Manifest SHA-256 | `fa38940c6e221778a6f4140b84a6f526fde9cdac4898f3b02835793a31914a9a` |

The published image already contains the EXL3 overlay and the SM121 kernels.
Its entrypoint is `vllm serve`. The local image is that digest plus the
Responses API content-type patch from the pinned recipe commit. Spark Serve
does not rebuild the CUDA overlay.

Engram tables are not inside the EXL3 tree. Preparation downloads shards 47
and 48 plus `config.json` from the native checkpoint, writes an index that
lists only the layer 1 and layer 14 embed tables, and mounts that directory
at `/engram-src` on both Sparks. Both trees are copied over the QSFP link.
The cluster's existing NCCL interfaces stay in `[cluster.nccl]`.

Serving follows the recipe's current defaults: 600,000 tokens, two sequences,
1,536-token prefill chunks, a 2.5 GiB KV pool, 64-token blocks, DSpark with
three speculative tokens, and vision on. Do not pass `--kv-cache-dtype`; vLLM
selects `fp8_ds_mla` for this checkpoint. The Engram row cache and resident
scales stay off. Official sampling is temperature 1.0 and top_p 0.95, with
thinking on. Those are client settings.

Mia's published decode and prefill rates are measurements on their hardware.
This cluster has not qualified the recipe yet.

## Prepare and start

Preparation downloads about 385 GiB on the head, copies it to the worker, and
pulls the published image. It does not stop a model that is already serving.

```sh
./spark-serve pull ds41-exl3
./spark-serve up ds41-exl3 --no-hermes --json
```

`up` hashes both trees before it stops the current workload. That preflight
can take a large part of its 3,600-second budget. Keep `--no-hermes` until
the trial has been reviewed.

## License

The published image and the vendored Responses patch are AGPL-3.0. The MIT
notice for earlier contributions is retained next to the patch. Model weights
remain under their publishers' licenses. The optional abliteration overlay
and the optional cooperative MoE path are not part of this recipe.
