# ADR 0017: Show batch jobs as card-only catalog entries

Accepted · Recorded 2026-11-05

**Context.** Pika H3 Flex (`Pika-Labs/Pika-H3-Flex` @`0ddf55ad`, a 33B
video + audio model on `MiniMaxAI/MiniMax-H3` @`42ed227e`) now runs on
sparktwo as a one-GPU batch CLI (`pika-h3-pro` in `~/pika-h3-flex/venv`).
It is not a server: no container, no port, no `/v1/models`. Its ~68 GiB GPU
peak (66.6 GiB allocated while sampling) does not fit beside
`qwen38-dual-tensorfold`, which holds ~68 GB on each Spark. Pedro wants it
shown as a card in the app. Every existing wrapper assumes a long-running
OpenAI-compatible server that the fenced controller swaps in and out
(ADR 0003), so a plain entry would reach `up`/`pull` and fail late or,
worse, look startable from the app.

**Decision.** A catalog entry with `wrapper = "batch"` or `launch = false`
is a card only:
- `up` (including `--print-cmd`) and `pull` refuse it right after the name
  resolves, before placement, preflight, SSH or the controller, so the
  current workload is never touched.
- `list --json` adds `launch`, `launch_note` (from `run_hint`), `home_host`
  (from `home_node`) and `outputs`; `backend` is `batch`. Older app builds
  ignore the new keys and still show the card under "Other".
- The app shows "batch job on <host> · no endpoint", a "Makes video + audio"
  pill and a neutral "batch job" pill instead of context, disables Start,
  explains how to run it, and leaves it out of the menu-bar start list.
- The entry has no `container`, so it never joins the controller's owned
  (stoppable) names; `keep_containers` and `stop_names` are unchanged.

Paths, timings and the run command live in the entry and in
`docs/pika-h3-flex.md`.

**Trade-off.** Running Pika still needs a manual stop of the Sparks, the
wrapper script, and an `up` to bring the server back; the app does not
fence or track it, so its status panels will show both Sparks idle while it
renders. Teaching the controller a batch lifecycle (lease the GPU, run,
restore the previous model) would remove that but is a much larger change;
revisit if Pika becomes routine.

**Evidence.** 480p 5 s t2va on sparktwo, 2026-11-05: 18:58 wall
(conditioning 433 s, transformer load 391 s, sampling 106 s, decode 195 s),
peak 69,597 MiB per `nvidia-smi`, 54.5 GB max RSS, Qwen stopped on both
Sparks and restored with `up` afterwards. `tests/test_batch_card.py` covers
the refusals and the list contract.
