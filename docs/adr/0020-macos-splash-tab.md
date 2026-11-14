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
  8101–8105 on 127.0.0.1. Stop signals only the process this tool spawned for
  that card, and only after its command line still matches the card and names
  no reserved port.
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

**Trade-off.** Splash is not a service manager. A server keeps running after
the app quits until Stop. A process we did not start, including anything
already bound to a card's port, is left alone, so Start fails closed instead
of taking the port. Qwen3.8-27B's directory names are the Hugging Face repo
ids under the TensorFold models root; rename `model_dir` and `draft_dir` if
the folders differ.

**Evidence.** [CLI](../../spark_serve_macos.py), [app](../../gui/MacOSView.swift),
[tests](../../tests/test_macos.py),
[Splash models](https://github.com/incoai/splash/blob/main/docs/models.md),
[Splash server options](https://github.com/incoai/splash/blob/main/docs/server.md).
