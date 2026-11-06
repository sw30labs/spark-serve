# Pika H3 Flex (card only)

`pika-h3-flex` in the catalog is a card only (ADR 0017). spark-serve shows it
but never starts, stops or prepares it; `up` and `pull` refuse it.

| | |
|---|---|
| Model | `Pika-Labs/Pika-H3-Flex` @`0ddf55ad7067ca1e76812253bd7b4600df87f759` |
| Base | `MiniMaxAI/MiniMax-H3` @`42ed227ee7df40d41602854ae760620d6eb651fe` |
| Host | sparktwo only, one GPU |
| venv / CLI | `~/pika-h3-flex/venv`, `pika-h3-pro` |
| Weights | `~/pika-h3-flex/weights` (137 GB: transformer 62, text encoder 63, VAE 9.8, audio VAE 0.6) |
| Adapter | `~/pika-h3-flex/weights/fl2va` (2.6 GB; the only adapter installed) |
| Wrapper | `~/pika-h3-flex/run-t2va.sh` (offline, logs to `~/pika-h3-flex/logs`) |
| Outputs | `~/pika-h3-flex/outputs` (`.mp4` plus a `.json` with timings) |

Tasks: `t2va`, `i2va`, `l2va`, `fl2va`, `ref2va`; 480p or 768p; 5–15 s.
Only t2va at 480p/5 s has been run here.

## Memory

~68 GiB GPU peak (69,597 MiB in `nvidia-smi`; 66.6 GiB allocated while
sampling) and ~52 GiB host RSS. `qwen38-dual-tensorfold` uses ~68 GB on
each Spark, so Pika cannot run while it serves. `--cpu-offload` lowers the
GPU need to ~30 GB but needs ~94 GB of host RAM, which a Spark serving Qwen
does not have either.

## Run (manual)

1. Stop the Sparks: `./spark-serve stop --node both` (or Stop both in the app).
2. On sparktwo: edit `~/pika-h3-flex/outputs/prompt.txt`, then
   `~/pika-h3-flex/run-t2va.sh`.
3. Bring the server back:
   `./spark-serve up qwen38-dual-tensorfold --node both --no-hermes`.


## Measured (2026-11-05, 480p 5 s t2va, seed 42)

18:58 wall: conditioning 433 s, transformer load 391 s, sampling 106 s,
decode/encode 195 s. 832×480, 124 frames, with audio.
