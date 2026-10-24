# Two Strix Halos and the llama.cpp cluster

Runbook, 2026-10-24. Decision: [ADR 0013](adr/0013-strix-halo-pair.md).
Single-box design notes: [Strix Halo analysis](strix-halo-analysis.md).

## The boxes

| | `amd-halo-2` | `amd-halo` |
|---|---|---|
| Use | Desktop with the GNOME widget | Headless (widget installed) |
| Cluster scripts | `~/Desktop/halo-cluster` | `~/halo-cluster` |
| `halo-ctl` | Engines `halogen`, `cluster`, `lemonade`, `vllm`; `models` | Same file |
| RPC worker | `ggml-rpc-server.service`, port 50053, enabled at boot | Same |
| Cluster weights | `~/models/Qwen3.8-Flash-Next` | Same, copied |

`halo-ctl models` lists the same models on both boxes: Halogen
`halogen-qwen3.8-flash-next`, cluster `qwen3.8-flash-next`, Lemonade
`gpt-oss-120b-Q4_K_M` and `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M`, and vLLM
`openai/gpt-oss-20b`. Read on 2026-10-24: `amd-halo-2` directly, `amd-halo`
through `amd-halo-2`'s key, read-only.

## The cluster card

`halo-qwen38-cluster` (alias `halo-cluster`): engine `cluster`, served name
`qwen3.8-flash-next`, port 8081, three MTP draft tokens on the master's GPU.
The card's 65,536-token context is `serve-qwen38.sh`'s `CTX`; the GGUF's native
context is 262,144.

Quantization, from the tensor headers of the six unsloth shards (175 GiB):
176.9B parameters, Q8_0 throughout. That covers 120.8B of experts, the 51.2B
per-layer token embedding, and 3.0B of attention and mixing. Routers and norms
are F32, and a few small query/key projections are BF16. The 3.9B MTP draft is
Q8_0. Cards show 8-bit weights in green, and the ranking scores them like
4-bit experts.

It is not yet measured under the catalog protocol. The halo-cluster README's
~28 tok/s (MTP 3, `amd-halo-2` as master) uses a different protocol. To measure:

```sh
python3 tools/catalog_tps_bench.py run halo-qwen38-cluster --site strix --box amd-halo-2
```

## Operating

```sh
./spark-serve strix status                                    # both boxes, roles, models on disk
./spark-serve strix up halo-qwen38-cluster --box amd-halo-2   # amd-halo-2 master, amd-halo worker
./spark-serve strix up halo-gpt-oss-120b --box amd-halo       # stops the cluster first if amd-halo is its worker
./spark-serve strix up lemonade:<name> --box amd-halo         # a model on disk without a catalog entry
./spark-serve strix stop --box amd-halo                       # the worker: stops the cluster
./spark-serve strix stop                                      # both boxes, masters first
./spark-serve strix connection --box amd-halo-2               # base URL, served model name, API key
./spark-serve strix logs --box amd-halo-2 --lines 120
```

`up` prints JSON progress events. For the cluster it waits up to 30 minutes for
the load. If the master's unit exits, `up` reports the unit's last journal line.
In the app, Stop abandons a start that is still loading.

The app's Strix Halo tab has one panel per box. Each shows the box's state,
cluster role, resources, inference mode and RPC worker, with Stop, Copy
connection details and Logs buttons. Below the panels are the model cards, one
column per engine, including models on disk without a catalog entry. Each card
has one Start button per box, labelled with what that start will stop. The
menu bar repeats the boxes' GNOME widget for both boxes: each engine's models
on disk, Stop, and Copy connection details.

Polling skips a box for a minute after it refuses this Mac's key or does not
answer. The boxes run OpenSSH 10.0, whose per-source penalties refuse an
address that keeps failing to log in, and that refusal would also block
`ssh-copy-id`. Commands that act on a box always try it again.

Inference mode switches system timers through polkit. Over SSH, `pkcheck`
answers "Authorization requires authentication", so the app shows the mode but
does not switch it. Use the widget on the box.

## Findings, 2026-10-24 (read-only)

1. `amd-halo` refuses this Mac's key, which `amd-halo-2` accepts. Run
   `ssh-copy-id amd-halo` once.
2. `amd-halo`'s 10 GbE port `enp97s0` is up at 10 Gb/s with its address but
   has no IPv4 route. Its default and LAN routes use the Wi-Fi interface
   `wlp99s0`, and its LAN hostname resolves to the Wi-Fi address. With
   `amd-halo` as master, the cluster's RPC stream to `amd-halo-2` therefore
   goes over Wi-Fi, as does the Mac's traffic to `amd-halo`. The halo-cluster
   README calls for the wired link.
3. A cluster load with `amd-halo` as master started at 13:59. The OOM killer
   stopped it at 14:18, after 18½ minutes, with a unit peak of 57.4 GB. Until
   then `halo-ctl` reported it as starting.
4. `halogen-stats` reports a 120.0 GiB GTT limit on `amd-halo` and 113.1 GiB on
   `amd-halo-2`. The halo-cluster README says both were set to 120.
5. On the worker, `halo-ctl` reports `stopped` while the RPC worker holds the
   cluster's share, so its widget would start another engine there. A guard in
   `halo-ctl` would close this gap: refuse while port 50053 has an established
   connection.
6. For vLLM, the widget's Copy connection details puts the checkpoint path
   (`openai/gpt-oss-20b`) on the Model line. Clients must send the served name
   (`gpt-oss-20b`), which spark-serve copies instead.
7. The cluster's `llama-server` runs without `--metrics`. spark-serve reads its
   served context from `/props` and shows — for its live rates; its counters
   are not wired up, because they could not be checked against a running
   server.
8. The boxes' newer `halo-ctl`, widget and cluster units are uncommitted files
   in `~/projects/halogen-widget` on `amd-halo-2`, a repository with no remote.
   The Mac's halo-inference-control checkout predates them.
