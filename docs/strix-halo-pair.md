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
| Network | 10 GbE only, no Wi-Fi profile | 10 GbE only, Wi-Fi profile with autoconnect off |
| Hardware | AMD RAH-001, the Ryzen AI Halo reference box built by Compal | Minisforum MS-S1 MAX |

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

Measured on 2026-10-24 under the catalog protocol: 512-token prompts, 256 output
tokens, thinking off. The server ran MTP 3, `CTX` 65,536, four slots and `-ts 7,3`
(finding 9):

| Master → worker | One request, end to end | After the first token | First token | Two at once, combined |
|---|---|---|---|---|
| `amd-halo` → `amd-halo-2` | 21.7 tok/s | 26.1 tok/s | 2.0 s | 26.4 tok/s |
| `amd-halo-2` → `amd-halo` | 21.4 tok/s | 25.7 tok/s | 2.0 s | 25.8 tok/s |

The difference is within the spread between requests (24.8–27.7 tok/s after the
first token), so either box can be master. Loads took 3 to 4 minutes. The card
shows the `amd-halo` master run. The halo-cluster README's ~28 tok/s uses its own
prompts, one slot and a 16,384-token context. To measure again:

```sh
python3 tools/catalog_tps_bench.py run halo-qwen38-cluster --site strix --box amd-halo
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

## Findings, 2026-10-24

1. `amd-halo` refused this Mac's key, which `amd-halo-2` accepted. Fixed the
   same day with `ssh-copy-id amd-halo`.
2. `amd-halo`'s 10 GbE port `enp97s0` was up at 10 Gb/s with its address but
   had no IPv4 route. Its default and LAN routes used the Wi-Fi interface
   `wlp99s0`, and its LAN hostname resolved to the Wi-Fi address. With
   `amd-halo` as master, the cluster's RPC stream to `amd-halo-2` therefore
   went over Wi-Fi, as did the Mac's traffic to `amd-halo`. The halo-cluster
   README calls for the wired link.

   That boot's log has one anomaly. Right after the wired port took its DHCP
   address, NetworkManager reported a conflict for that address with another
   host, whose MAC is the permanent address of `amd-halo`'s own Wi-Fi card.
   With two interfaces on one subnet and Linux's default `arp_ignore` and
   `arp_announce` of 0, either card can answer and send ARP for the other's
   address. The info-level log does not show what removed the routes. The
   next boot came up with routes on both interfaces.

   Fixed the same day: `amd-halo` is wired-only, like `amd-halo-2`.
   `~/wired-only.sh` on the box turned the Wi-Fi profile's autoconnect off,
   took Wi-Fi down and renewed the wired lease, and logged to
   `~/wired-only.log`. The router now answers `amd-halo` with the wired
   address. To bring Wi-Fi back, run
   `sudo nmcli connection modify <profile> connection.autoconnect yes`, then
   `sudo nmcli connection up <profile>`. Keep one interface per box on the
   LAN; a cable in the second 10 GbE port, `enp98s0`, would add a second one.
   The cluster's `scripts/peer.sh` names both boxes' wired addresses, and each
   RPC worker listens on its box's wired address only, so a changed DHCP lease
   breaks the cluster.
3. A cluster load with `amd-halo` as master started at 13:59. The OOM killer
   stopped it at 14:18, after 18½ minutes, with a unit peak of 57.4 GB. Until
   then `halo-ctl` reported it as starting. Finding 9 explains the kill.
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
9. The cluster's master keeps about 52 GiB in host memory on top of its GPU
   share. That is most likely the model's 51 GiB per-layer embedding. llama.cpp
   split the layers by GPU memory, about 51.5 : 48.5 from the 120 and 113.1 GiB
   limits, without counting it. Once loaded, `amd-halo` as master had 0.4 GiB
   available, and the OOM killer stopped it on its first request. `amd-halo-2`
   as master had 4.7 GiB, and the OOM killer stopped it at two concurrent
   requests. `serve-qwen38.sh` on both boxes now passes `-ts 7,3`, which puts 70%
   of the layers on the worker; RPC devices come first in llama.cpp's device list.
   Each box then kept 22–28 GiB available under load. The original script is
   kept beside it as `serve-qwen38.sh.bak-before-no-host-20261024`. Two other
   flags did not help. `--no-host` turned the 52 GiB into process memory.
   `--load-mode mmap` cut the load to 4 KB reads, about 20–25 MB/s.
10. After the 19:03 reboot, `amd-halo`'s RPC worker started 9 seconds into boot,
    before the GPU was ready. It logged "failed to initialize ROCm: no
    ROCm-capable device is detected" and served on the CPU (`Devices: CPU`).
    Every run with `amd-halo` as worker therefore used its CPU and reached 11 to
    13 tok/s, not 21. Restarting the unit put it on the GPU. Both boxes' units now
    pass `-d ROCm0`. With it, the worker exits while the GPU is missing (tested),
    and `Restart=always` retries every 5 seconds until it can start. The
    originals are kept beside the source copies as
    `ggml-rpc-server.service.bak-before-device-20261024`. To check which device a
    worker uses, run `journalctl --user -u ggml-rpc-server -b | grep -E "ROCm0|CPU:"`.
