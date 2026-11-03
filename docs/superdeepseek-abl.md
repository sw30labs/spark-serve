# SuperDeepseek V4 Flash abliterated MQ (`ds4-super-abl`)

Two-Spark TP2 over QSFP, DSpark K=1, 1M context. See ADR 0016 for why it starts
through a pinned entry script.

## Prepare (once, outside spark-serve)

On sparkone:

```bash
~/superdeepseek-abl/prepare.sh --yes          # hard-links parent shards, ~2.4 GiB download, rsync to spark-two
~/superdeepseek-abl/prepare.sh --yes --image  # pinned image on both Sparks
```

`spark-serve pull ds4-super-abl` prints this instead of downloading.

## Start, status, stop

- App: Start on "SuperDeepseek V4 Flash abliterated MQ", or
  `spark-serve up ds4-super-abl` (aliases `superdeepseek`, `superdeepseek-abl`).
- Order: pin checks → offline preflight in the pinned image on both Sparks
  (`recipes/superdeepseek-abl/verify.py`) → stop the current workload (never
  `keep_containers`) → rank 1, then rank 0. Readiness is `served_name` on
  `cluster.lan_url` (:8000). The measured load is about 8.5 minutes;
  `ready_timeout` is 3600 s.
- `spark-serve status` / `stop` treat `vllm_superdeepseek_abl` like any catalog
  container.
- Inside the container, `recipes/superdeepseek-abl/entry.sh` copies the
  model's `encoding_dsv4.py` into vLLM and patches the reasoning-effort mapping
  (unknown/`low` → `low`), then runs `exec vllm serve`. Its first stderr line is
  `spark-serve-entry: SuperDeepseek tokenizer patch applied`.
- `spark-serve up ds4-super-abl --print-cmd` shows both rank scripts without
  touching the Sparks.

## Hermes

On ready, Start (without `--no-hermes`) selects provider `spark-superds` at
`hermes_base_url` (`http://192.168.86.57:8000/v1`: IPv4 on purpose, since
`sparkone.local` resolves to IPv6 first). It keeps that provider's name, key and
other models, and leaves the `spark` provider alone.

## Standalone launcher

`~/superdeepseek-abl/start.sh` / `stop.sh` still work on :8890 with the same
container name and caches. Run one launcher at a time.

## Editing the recipe

After changing `entry.sh` or `verify.py`, update `entry_script_sha256` /
`preflight_script_sha256` in models.toml (`shasum -a 256 recipes/superdeepseek-abl/*`).
A stale pin refuses the start before anything is stopped.
