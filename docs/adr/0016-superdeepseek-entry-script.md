# ADR 0016: Start SuperDeepseek through a pinned in-container entry script

Accepted · Recorded 2026-11-03

**Context.** `ds4-super-abl` (Jiunsong/SuperDeepseek-V4-Flash-abliterated-MQ-2xDGX
@`2a7dd6a1`) ran only through the standalone `~/superdeepseek-abl/start.sh` on
sparkone, on :8890. The catalog entry was a fail-closed placeholder, so the app's
Start button could not launch it. Two changes must be made inside the container
before `vllm serve`. Both come from the author's `run_superdeepseek_v4_vllm.sh`
@`2a7dd6a1` via the reviewed `container-entry.sh`:
- copy the snapshot's `encoding/encoding_dsv4.py` over vLLM's
  `tokenizers/deepseek_v4_encoding.py` (the file is byte-identical to the
  official DeepSeek-V4-Flash-0731 file);
- patch the exact string in `tokenizers/deepseek_v4.py`, so that an unknown or
  `low` reasoning effort maps to `low` instead of `high`.

Without them, the catalog's default `reasoning_effort: low` runs as `high`.
Status, Hermes and the app all probe `cluster.lan_url`, whose port must equal
`cluster.port` (8000).

**Decision.** Add three vLLM-wrapper catalog keys:
- `entry_script` + `entry_script_sha256`: a `recipes/` file run as
  `--entrypoint bash <image> -c <script> spark-serve-entry <vllm argv>`. It
  applies its patches, then runs `exec vllm serve "$@"`.
- `preflight_script` + `preflight_script_sha256`: a `recipes/` file run as
  `python3 -c <script> <preflight_args>` inside the existing offline preflight
  (no GPU, `--network none`, read-only caches) on every host.
- `hf_cache_readonly`: mount the HF cache `:ro`.

Paths cannot leave `recipes/`. A missing file or a stale pin refuses before
preflight, before the controller, and before `--print-cmd`. The script travels
in the container command, so the hosts need no launcher files, and
`docker inspect` shows exactly what ran. `recipes/superdeepseek-abl/entry.sh`
lifts the patch block verbatim. `verify.py` checks:
- the pinned index and its 50 files;
- config, tokenizer and the pinned encoding;
- the vllm binary;
- that the patch target occurs exactly once, or is already patched.

The vLLM flags live in the catalog. spark-serve serves on :8000, and `start.sh`
keeps :8890. Both use the container name `vllm_superdeepseek_abl` and the same
work-dir caches, so only one runs, and a switch stops either.
`keep_containers` is untouched. `pull` points to `prepare.sh` and does not
download anything.

**Trade-off.** A generic entry-script hook is a larger surface than a one-off
wrapper. The pins, the `recipes/`-only rule and the vLLM-only scope keep it
reviewable. Patching at start keeps the author's pinned image digest with no
derived build. The cost is that the target must be re-verified whenever the
image changes, and preflight does that before any stop. On ready, Hermes is
retargeted to :8000 instead of the manual :8890 (see the amendment below). Intentional differences from
`start.sh`:
- `docker create` with a fenced start;
- `--security-opt label=disable` and `-w /tmp`;
- spark-serve labels;
- `drop_caches` through `sudo -n`;
- no "no other containers" check (the controller swaps by design);
- no separate cross-node image-ID check (the digest pin and the
  `--pull never` preflight cover it).

**Evidence.** `--print-cmd` matches `docker inspect` of the 2026-11-03 start.sh
containers on both Sparks: env, binds (the launcher mount is now embedded), shm,
ulimits, devices, GPUs and image ID. The vLLM argv that `container-entry.sh`
builds matches flag for flag (36 on rank 0, 37 on rank 1), except `--port`. The
exited sparkone container holds the patched mapping.
[Tests](../../tests/test_superdeepseek_recipe.py), [runbook](../superdeepseek-abl.md),
[recipe](../../recipes/superdeepseek-abl/).

**Amendment 2026-11-03 (Hermes over IPv4).** `sparkone.local` resolves to IPv6
first and vLLM listens on IPv4 only, so the Start hook's
`cluster.lan_url` target (`spark` provider) failed for Hermes. Other models,
status and the app's endpoint checks still depend on `cluster.lan_url`, so it
stays unchanged. Instead, a model may pin `hermes_base_url` (http(s), ending in
`/v1`, no credentials, port equal to `cluster.port`). `ds4-super-abl` sets it to
`http://192.168.86.57:8000/v1` with `hermes_provider = "spark-superds"`.
The override names the head endpoint, so a worker placement keeps the worker
URL. A bad value refuses `up` before any switch and never edits Hermes. The
hook keeps an existing provider's display name, the legacy `spark-ds4` alias
still follows `cluster.lan_url`, and the status `active_node` check also accepts
the pinned URL, so the app still shows which node Hermes is using.
[Tests](../../tests/test_superdeepseek_hermes.py).
