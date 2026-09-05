# ds4-vision (FlyCockpit)

DeepSeek-V4-Flash-**0731** + FlyCockpit DeepEncoderV2 tower/adapter, served through
spark-serve on the same LAN URL as text DS4 (`http://192.168.86.44:8000`).

## Flip

```bash
./spark-serve up ds4-vision   # stops whatever was on :8000 (text ds4)
./spark-serve up ds4          # back to text-only Flash
./spark-serve stop            # leave Atlas neo4j alone
```

Only one recipe owns `:8000`. Flipping vision **stops** text `ds4`. Atlas
(`singularity-atlas-neo4j`) stays up.

## Prerequisites (both Sparks)

1. Hub cache: `deepseek-ai/DeepSeek-V4-Flash-0731` (48 shards, ~167 GB) — **not** the older `DeepSeek-V4-Flash` text weights.
2. Vision assets under `~/.cache/huggingface/vision/`:
   - `tower/deepencoder_v2_tower.safetensors` md5 `2d5dba626d816cc367d28b32e744830e`
   - `adapter/latest.pt` md5 `d9b3b3bda8f790ecf7cd5a98e6fb93a5`
3. Symlink tree: `python3 plugin/make_vision_model_dir.py` → `~/.cache/huggingface/dsv4-0731-vision`
4. Image: build FlyCockpit playbook → `dsv4-vision-vllm:0.1.1`
5. Playbook checkout at `~/REPOS/DeepSeek-V4-Vision-2x-DGX-Sparks` (plugin bind-mount)

## Agent contract

- Served name: `deepseek-v4-flash-0731-vision`
- Send images as OpenAI `image_url` content parts (data URL or http).
- On **image** turns set `"thinking": false` (or omit reasoning) — vision + thinking is unreliable.
- `--limit-mm-per-prompt` counts images in **replayed history**; default budget is 8.
- Good at screenshots / on-screen text. Not a click-agent on arbitrary live UIs.

## Smoke

```bash
curl -sS http://192.168.86.44:8000/v1/models
# text
curl -sS http://192.168.86.44:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash-0731-vision","messages":[{"role":"user","content":"Reply with exactly: VISION_OK"}],"max_tokens":16,"temperature":0,"thinking":false}'
# image: same endpoint with content array of text + image_url
```

Playbook upstream: https://github.com/FlyCockpit/DeepSeek-V4-Vision-2x-DGX-Sparks
