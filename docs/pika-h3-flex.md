# Pika H3 Flex (batch entry)

`pika-h3-flex` is a batch entry (ADR 0018). `up` stops the serving model,
runs one job on sparktwo, and the Spark brings the previous model back
itself when the job ends. `pull` refuses it: it was prepared by hand.

| | |
|---|---|
| Model | `Pika-Labs/Pika-H3-Flex` @`0ddf55ad7067ca1e76812253bd7b4600df87f759` |
| Base | `MiniMaxAI/MiniMax-H3` @`42ed227ee7df40d41602854ae760620d6eb651fe` |
| Host | sparktwo only, one GPU |
| venv / CLI | `~/pika-h3-flex/venv`, `pika-h3-pro` |
| Weights | `~/pika-h3-flex/weights` (137 GB: transformer 62, text encoder 63, VAE 9.8, audio VAE 0.6) |
| Adapter | `~/pika-h3-flex/weights/fl2va` (2.6 GB; the only adapter installed) |
| Outputs | `~/pika-h3-flex/outputs/<YYYYMMDD-HHMMSS>.mp4` (+ `.mp4.json` with timings) |
| Job dirs | `~/.local/state/spark-serve-batch/<id>/` (`state.json`, `job.log`, `restore.log`) |

The catalog command runs task `t2va`. The CLI also has i2va, l2va, fl2va and
ref2va, at 480p or 768p, for 5–15 s; only t2va at 480p/5 s has been run here.

## Run

App: select the Pika card, type a prompt, pick the options, then Run on
sparktwo… and confirm.

CLI:

```
./spark-serve up pika-h3-flex --prompt "…" [--resolution 480p] [--aspect 16:9] [--duration 5] [--seed 42]
./spark-serve up pika-h3-flex --prompt-file prompt.txt --print-cmd   # plan only; touches nothing
./spark-serve batch status
./spark-serve batch stop      # kills the job; the Spark still restores the previous model
```

While a job runs, `up`, `stop`, `reboot` and `shutdown` refuse. After it ends
and the restore finishes, `status`/`batch status` take over the Spark-side
state and they work again.

## If something goes wrong

- `batch status` shows `restore failed`: read
  `~/.local/state/spark-serve-batch/<id>/restore.log` on sparktwo, then
  `./spark-serve up qwen38-dual-tensorfold --node both --no-hermes` from the Mac.
- sparktwo rebooted mid-job: the supervisor is gone, so nothing restores.
  Bring the model back by hand as above.

## Memory

~68 GiB GPU peak (69,597 MiB in `nvidia-smi`; 66.6 GiB allocated while
sampling) and ~52 GiB host RSS. `qwen38-dual-tensorfold` uses ~68 GB on
each Spark, so Pika cannot run while it serves. `--cpu-offload` lowers the
GPU need to ~30 GB but needs ~94 GB of host RAM.

## Measured (2026-11-05, 480p 5 s t2va, seed 42)

18:58 wall: conditioning 433 s, transformer load 391 s, sampling 106 s,
decode/encode 195 s. 832×480, 124 frames, with audio. Bringing Qwen
back with `up` took about 6–7 minutes, most of it the weight-verifying preflight.
