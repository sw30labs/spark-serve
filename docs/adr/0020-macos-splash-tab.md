# ADR 0020: Serve this Mac Studio's models with Splash on their own ports

Accepted · Recorded 2026-11-14

**Context.** This Mac is a Mac Studio (Mac15,14, M3 Ultra, 512 GB unified
memory). Splash (`brew install incoai/tap/splash`, `splash serve`) serves
Qwen3.8-27B, Qwen3.6-35B-A3B, Ornith-1.5-122B, Qwen3.8-Flash-Next, and Flint
there, OpenAI-compatible, each with the DFlash2 draft named in Splash's
`install/families.py`. Splash's own default listener is 127.0.0.1:8000. Port
8000 on this Mac is already a live oMLX server.

Qwen3.8-27B's MLX 4-bit weights and `incoai/Qwen3.8-27B-DFlash2` draft are
under `/Volumes/DATA/local-models/projects/TensorFold/models/`. The other four
are downloading to `/Volumes/DATA/local-models/splash/<hf-repo-name>/`. On
2026-11-13, Splash 1.6.1 measured Qwen3.8-27B at 171 tok/s decode, about 544
tok/s prefill on a 19K prompt, 0.125 s cached TTFT, and 192 tok/s aggregate at
4 concurrent. That is not the 512-token catalog protocol. The other four are
unmeasured.

**Decision.**
- **A MacOS tab, separate from Spark and Strix.** `[macos]` and
  `[macos.models.<id>]` in the catalog, `spark_serve_macos.py`, and
  `MacOSView.swift`. The Spark and Strix tabs are unchanged.
- **One Splash process per card.** Start runs `splash serve` with only flags
  Splash documents (`--model`, `--model-dir`, `--draft-model`, `--host`,
  `--port`, and `--mmproj` when the card sets one). Cards default to ports
  8101–8105 on 127.0.0.1. Splash 1.6 exits that launcher and writes
  `serve-<port>.json`; status records that pid when it is alive, its `--port`
  and `--model` match the card, and it names no reserved port. Stop signals
  only that process group.
- **Port 8000 stays untouched.** It is reserved even if the catalog drops it.
  No card may use it. Status never requests it, and a state file that points
  at a process on it is forgotten without a signal.
- **Paths are configuration, disk status is the filesystem.** Each card has
  `model_dir` and `draft_dir`. Omitted, they default to
  `<weights_root>/<OWNER>/<REPO>/` (a GGUF `:VARIANT` is not part of the
  directory). A path that does not exist yet is "not on disk yet"; Start stays
  off until both exist. Nothing is marked permanently missing.
- **The one measurement is a manual row** in `measured-performance.json`
  (`manual: true`) so `catalog_tps_bench.py summarize` keeps it until a
  protocol run replaces that id. The card shows decode, the 4-stream
  aggregate, prefill, and cached TTFT, and says it is not the 512-token protocol.
- **Use in Hermes is the Strix write.** Bordered buttons beside Start, Stop,
  and Logs, and `spark-serve macos use <card>`, call the same `_hermes_patch`.
  A ready card is written as provider `splash` (or the card's
  `hermes_provider`) at `http://127.0.0.1:<port>/v1`. `model.default` is the
  card's `--model` id, and `context_length` is `max_model_len`. The control
  stays off until the card is ready. Port 8000 is never selected, and the
  `spark` provider block is left as it is. `--hermes-target spark` updates
  Hermes on the head Spark.

**Trade-off.** Splash 1.6 hands the serve off to a detached process and the
launcher exits. Status follows the process that is actually serving: Splash's
`serve-<port>.json`, or one process whose argv names that card's port and
model. Stop signals only that verified process group. A port that answers
with the card's model but whose process cannot be verified is shown as
serving so Hermes can use it, and Stop stays off. A port serving a different
model is shown as busy and is not selected. oMLX on port 8000 is never probed
or signalled. Qwen3.8-27B's directory names are the Hugging Face repo ids
under the TensorFold models root; rename `model_dir` and `draft_dir` if the
folders differ.

**Evidence.** [CLI](../../spark_serve_macos.py), [app](../../gui/MacOSView.swift),
[tests](../../tests/test_macos.py),
[Splash models](https://github.com/incoai/splash/blob/main/docs/models.md),
[Splash server options](https://github.com/incoai/splash/blob/main/docs/server.md).
