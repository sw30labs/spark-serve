# Nemotron-3-Super (NVFP4, solo Spark)

Second catalog profile for the two-Spark LAN serve — **not** proven smarter than
DeepSeek-V4-Flash. Use it when you want NVIDIA's Nemotron-3-Super NVFP4 recipe on
a single GB10, or to A/B against `ds4` / `ds4-vision`.

## Shape (defaults — do not flip casually)

| Knob | Default | Why |
|------|---------|-----|
| `nnodes` | **1** | Solo on **sparkone** (head). sparktwo stays idle. |
| `tensor_parallel` | **1** | Matches NVIDIA single-Spark recipe. **TP=2 is not the default.** |
| `max_model_len` | **262144** | Safer KV on 128GB UMA than 1M. NVIDIA shows up to 1M with headroom tradeoffs. |
| Image | `vllm/vllm-openai:cu130-nightly` | Stock vLLM nightly (not Aiden DS4). |
| Weights | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | HF id / cache key. |
| Served name | `nemotron-3-super` | Hermes `-m` / OpenAI `model`. |
| Container | `vllm_cluster` (cluster default) | Same stop path as other recipes. |

Optional later: raise `nnodes`/`tensor_parallel` to 2 and rsync weights to
sparktwo — only as a documented override, never the sketch default.

## Why this profile

- Second recipe next to Flash / vision — same `:8000`, same Hermes provider slot.
- Official NVIDIA single-Spark NVFP4 path (marlin GEMM, fp4 quant, hermes tool parser).
- Leaves QSFP / GDR fabric unused: no multi-node NCCL for the default sketch.
- sparktwo is idle while this is up (power / heat win; also a limitation).

## Flip

```bash
./spark-serve up nemotron-super   # stops whatever owns :8000 (ds4 / ds4-vision)
./spark-serve up ds4              # back to text Flash (TP=2 both Sparks)
./spark-serve up ds4-vision       # back to FlyCockpit vision
./spark-serve stop                # leave Atlas neo4j alone
```

Only one recipe owns `:8000`. Atlas (`singularity-atlas-neo4j`) stays up
(`keep_containers`).

## Weights policy (hunt → pull once → optional rsync)

**Do not Hub-pull from the Mac.** On sparkone:

1. **Hunt** local caches first:
   - `~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`
   - any prior `HF_HOME` / shared volume copies
2. If missing, **pull once on sparkone** (HF CLI or `huggingface-cli download`), then leave `HF_HUB_OFFLINE=1` in the catalog env.
3. **rsync to sparktwo only if** you later switch the recipe to TP=2 / `nnodes=2`. Solo default never needs the worker cache.

Image: `docker pull vllm/vllm-openai:cu130-nightly` on sparkone when you are ready to boot (not part of this sketch commit).

## Env (catalog)

- `VLLM_NVFP4_GEMM_BACKEND=marlin`
- `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`
- `VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm`
- `VLLM_USE_FLASHINFER_MOE_FP4=0`
- `HF_HUB_OFFLINE=1` after the one-time pull

## Flags of note

- `--quantization fp4`, `--moe-backend marlin`, `--dtype auto`
- `--gpu-memory-utilization 0.90`, `--kv-cache-dtype fp8`
- `--enable-auto-tool-choice` + `--tool-call-parser hermes` (verify on first boot)
- MTP / speculative decoding: optional; omitted from the sketch args
- **Reasoning-parser plugin**: optional Phase 2 — do not block first boot on vendoring a `.py` plugin

## Hermes

```bash
./spark-serve up nemotron-super
# catalog sets hermes_provider=spark, hermes_context_length=262144
# OpenAI base: http://192.168.86.44:8000/v1   model: nemotron-3-super
```

Retarget Hermes to the spark provider / served name after flip (same pattern as ds4).

## Limitations

- No GDR / multi-node NCCL on the default path (solo).
- sparktwo idle while nemotron-super is up.
- Context 256k sketch vs NVIDIA's up-to-1M demo — raise `max_model_len` only after you measure KV headroom.
- Tool parser `hermes`: confirm tool calls on first boot; adjust if the nightly expects another parser.
- Reasoning-parser plugin is Phase 2 (optional).
- Not an Aiden / DS4 image — different entrypoint and flag surface.

## Dry checks (no live flip)

```bash
./spark-serve list
./spark-serve up nemotron-super --print-cmd   # rank0 only when nnodes=1
```

Do **not** run `up` against the live vision serve until weights + image are ready and you intend to stop `:8000`.
