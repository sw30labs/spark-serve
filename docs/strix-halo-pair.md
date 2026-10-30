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
| Cluster models | `models.json` beside the scripts | Same file |
| Cluster weights | `~/models/Qwen3.8-Flash-Next`, `~/models/Nemotron-3.5-Super-VL-120B-A12B` | Same, copied |
| Network | 10 GbE only, no Wi-Fi profile | 10 GbE only, Wi-Fi profile with autoconnect off |
| Hardware | AMD RAH-001, the Ryzen AI Halo reference box built by Compal | Minisforum MS-S1 MAX |

`halo-ctl models` lists the same models on both boxes: Halogen
`halogen-qwen3.8-flash-next`, cluster `qwen3.8-flash-next`,
`qwen3.8-flash-next-512k` and `nemotron-3.5-super`, Lemonade
`gpt-oss-120b-Q4_K_M` and `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M`, and vLLM
`openai/gpt-oss-20b`. Read on 2026-10-25.

## The cluster cards

Three catalog cards use engine `cluster`. Each one names an entry in
`models.json`, the registry beside the cluster scripts on both boxes:

| Card | Served name | Context | Serve script |
|---|---|---|---|
| `halo-qwen38-cluster` (alias `halo-cluster`) | `qwen3.8-flash-next` | 262,144 | `serve-qwen38.sh` |
| `halo-qwen38-cluster-512k` (alias `halo-cluster-512k`) | `qwen3.8-flash-next-512k` | 524,288 | `serve-qwen38.sh` with YaRN, factor 2 |
| `halo-nemotron35-cluster` (alias `halo-nemotron`) | `nemotron-3.5-super` | 1,048,576 | `serve-nemotron35.sh` |

All three serve on port 8081, one at a time. `halo-ctl up cluster <id>` records
the choice in `~/.config/halo/cluster.env`, which `halo-cluster.service` reads,
and `halo-ctl status` reports that id while the model loads. Starting another
cluster model on the master replaces the one it serves. `halo-ctl models` lists
an entry only where its weights are on disk. llama-server ignores the model name
in requests, so clients can keep sending `qwen3.8-flash-next` to the 512K model.

The registry arrived on 2026-10-25. Before it, `halo-ctl` and `halo-cluster-run`
knew only `qwen3.8-flash-next`. The originals are kept beside them in
`~/.local/bin` as `*.bak-before-registry-20261025`, and the halo-cluster README's
as `README.md.bak-before-registry-20261025`. To add a model, put its serve
script in `scripts/` and its entry in `models.json` on both boxes, then add its
card.

### Qwen3.8-Flash-Next Q8

`halo-qwen38-cluster` (alias `halo-cluster`): engine `cluster`, served name
`qwen3.8-flash-next`, port 8081, three MTP draft tokens on the master's GPU.
The card's context is `serve-qwen38.sh`'s `CTX`. Since 2026-10-24 it defaults to the
model's native maximum, 262,144 tokens; it was 65,536 before. Only 12 of the 48
layers keep a KV cache, so context is cheap. llama.cpp allocates the cache at
startup. Measured at ready with `amd-halo` as master:

| `CTX` | Master GPU | Worker GPU | Available, master / worker |
|---|---|---|---|
| 65,536 | 40.0 GiB | 93.6 GiB | 29.3 / 28.7 GiB |
| 131,072 | 41.5 GiB | 95.3 GiB | 27.8 / 26.9 GiB |
| 262,144 | 44.5 GiB | 98.8 GiB | 24.6 / 23.0 GiB |

That is about 52 KB per token, linear. Host memory does not change. Under load
the master's prompt cache can add up to 8 GiB of RAM (`--cache-ram`). During the
benchmark at 262,144, the lowest available memory was 18.6 GiB on the master and
22.8 GiB on the worker.

The model card allows up to 1,000,000 tokens with YaRN. llama.cpp's YaRN works for
this model; it was tested on 2026-10-24 at 524,288 tokens, the configuration the
`halo-qwen38-cluster-512k` card has served since 2026-10-25. The flags were
`--rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144`, set through
`LLAMA_ARG_ROPE_SCALING_TYPE`, `LLAMA_ARG_ROPE_SCALE` and `LLAMA_ARG_YARN_ORIG_CTX`.
At ready, the master had 50.4 GiB of GPU memory and 18.1 GiB available. The worker
had 105.8 GiB, 7.3 GiB under its limit, and 16.0 GiB available.

The test document was a 310,460-token synthetic ledger with three codes at token
positions 15,003, 154,623 and 288,574. The model recalled all three exactly, and
still answered a short question correctly. Reading the document took 33.7 minutes
(154 tok/s on average, slower as the context grows). Decode at that length ran at
14.9 tok/s. At 1M tokens, the Q8 weights would push the worker past its GPU limit.

Quantization, from the tensor headers of the six unsloth shards (175 GiB):
176.9B parameters, Q8_0 throughout. That covers 120.8B of experts, the 51.2B
per-layer token embedding, and 3.0B of attention and mixing. Routers and norms
are F32, and a few small query/key projections are BF16. The 3.9B MTP draft is
Q8_0. Cards show 8-bit weights in green, and the ranking scores them like
4-bit experts.

Measured on 2026-10-24 under the catalog protocol: 512-token prompts, 256 output
tokens, thinking off. The server ran MTP 3, four slots and `-ts 7,3` (finding 9):

| Master → worker | `CTX` | One request, end to end | After the first token | First token | Two at once, combined |
|---|---|---|---|---|---|
| `amd-halo` → `amd-halo-2` | 524,288, YaRN (512K card, 2026-10-25) | 20.9 tok/s | 25.0 tok/s | 2.0 s | 24.8 tok/s |
| `amd-halo` → `amd-halo-2` | 262,144 | 21.6 tok/s | 26.0 tok/s | 2.1 s | 25.9 tok/s |
| `amd-halo` → `amd-halo-2` | 65,536 | 21.7 tok/s | 26.1 tok/s | 2.0 s | 26.4 tok/s |
| `amd-halo-2` → `amd-halo` | 65,536 | 21.4 tok/s | 25.7 tok/s | 2.0 s | 25.8 tok/s |

All four are within the spread between requests (24.8–27.7 tok/s after the
first token), the 512K card at its low end. Either box can be master, and the
context size makes little difference at these prompt lengths. Loads took 3 to 4
minutes. The 262K card shows the 262,144 run and the 512K card its own. The
halo-cluster README's ~28 tok/s uses its own prompts, one slot and a
16,384-token context. To measure again (with Python 3.11 or newer, for
`tomllib`):

```sh
python3 tools/catalog_tps_bench.py run halo-qwen38-cluster --site strix --box amd-halo
```

### Nemotron 3.5 Super Q8: 1M tokens

NVIDIA's Nemotron 3.5 Super VL has a native 1M-token context and fits the pair
at 8-bit. It is a 120.7B-parameter mixture of experts with 12B active
(`nemotron_h_moe`: 40 Mamba, 40 MoE and 8 attention layers), Q8_0 throughout
except the routers and a few small F32 tensors. It was tested on the night of
2026-10-24, before it had a card. Since 2026-10-25 the card
`halo-nemotron35-cluster` serves it through `serve-nemotron35.sh`, with the same
flags on port 8081. The files, in `~/models/Nemotron-3.5-Super-VL-120B-A12B`,
come from `ggml-org/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF`: the model
(128,472,993,728 bytes) and its MTP head (4,276,991,712 bytes), both Q8_0. Both
match Hugging Face's SHA-256 (`SHA256SUMS` in the folder). They were downloaded
to `amd-halo` and on 2026-10-25 copied to `amd-halo-2`, where they were checked
again, so either box can be master. The vision projector was not downloaded, so
the model serves text only.

The test server, started by hand on `amd-halo` as master with `amd-halo-2`'s RPC
worker:

```sh
~/llama.cpp/rocm/bin/llama-server --rpc amd-halo-2.lan:50053 \
  -m NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q8_0.gguf -ngl 999 -fa on -c 1048576 \
  -md mtp-NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 -devd ROCm0 -ngld all \
  -a nemotron-3.5-super --api-key-file ~/.config/halogen/api-key --host 0.0.0.0 --port 8082
```

It used llama.cpp's default split by GPU memory, with no `-ts`, and was ready
after about 3½ minutes. Its four slots share one 1,048,576-token cache. Unlike
Qwen, the master needs little host memory: 2.3 GiB shared and 2.7 GiB for its
`llama-server`.

| | Master, `amd-halo` | Worker, `amd-halo-2` |
|---|---|---|
| GPU memory at ready | 75.3 GiB | 69.7 GiB |
| Available at ready | 44.2 GiB | 52.2 GiB |
| Available after the 1M-token request | 31.3 GiB | 50.9 GiB |

After that request the master's `llama-server` held 15.5 GiB of RAM; the journal
does not show what that memory is. The worker's RPC cache, `~/.cache/llama.cpp/rpc`
on `amd-halo-2`, wrote 59.0 GiB during the load. With Qwen's share it holds
148 GiB.

Speed under the catalog protocol (512-token prompts, 256 output tokens, thinking
off):

| Master → worker | One request, end to end | After the first token | First token | Two at once, combined |
|---|---|---|---|---|
| `amd-halo-2` → `amd-halo`, the card (2026-10-25) | 15.7 tok/s | 18.9 tok/s | 2.8 s | 18.5 tok/s |
| `amd-halo` → `amd-halo-2`, the test (not saved to the history) | 15.5 tok/s | 18.8 tok/s | 2.8 s | 18.4 tok/s |

That is about 70% of Qwen's speed in the cluster, either way round. About two
thirds of the MTP draft tokens were accepted in the test. With `amd-halo-2` as
master, the default split put 72.8 GiB on the master and 72.4 GiB on the
worker, and halogen-stats reported 46.0 and 49.6 GiB of RAM free.

The 1M-token recall test used the same synthetic ledger as Qwen's 512K test.
The document was 986,958 tokens, with the codes at positions 47,383, 488,049
and 937,013. The model recalled all three exactly and still answered the short
question correctly. Reading the document took 2 hours 34 minutes, 106.8 tok/s on
average. The rate fell as the context grew:

| Tokens read | Elapsed | Rate over the previous 50K tokens |
|---|---|---|
| 51,205 | 2.8 min | 307 tok/s |
| 251,909 | 18.7 min | 174 tok/s |
| 501,765 | 51.2 min | 109 tok/s |
| 751,621 | 97.5 min | 81 tok/s |
| 950,277 | 144.3 min | 66 tok/s |

At 301,061 tokens it had averaged 209 tok/s, against Qwen's 154 tok/s over 310K
tokens. Decode at 987K tokens ran at 15.1 tok/s, over 28 tokens with every draft
accepted.

## GLM-5.3-Flash 4-bit: a test, 2026-10-25

Z.ai's GLM-5.3-Flash (MIT licence) has a native 1M-token context. It has 320.8B
parameters with 18B active: 34 linear-attention (KDA) layers, 11 sparse-attention
(DSA, MLA) layers and one MTP layer (`glm5-next`). llama.cpp has supported it since
2026-09-30 and its MTP since 2026-10-07, both before b11849. The test used
unsloth's UD-Q4_K_XL GGUF: 186.0 GiB in six shards, Q4_K 63%, Q5_K 32%, Q8_0 3%,
Q6_K 2%. It was downloaded to `amd-halo` only, in `~/models/GLM-5.3-Flash/UD-Q4_K_XL`,
and checked against Hugging Face's SHA-256. A test server ran by hand on `amd-halo`
as master, on port 8082, with the model's own MTP layer drafting three tokens. It
has no card.

It does not fit at 1M tokens. With layers 0–24 on the worker (`-ts 25,22`), the
worker needed 95.3 GiB of weights, 10.8 GiB of cache and a 10.1 GiB compute buffer:
116.2 GiB against its 113.1 GiB GPU limit. The load stopped there, cleanly. A
256-token batch cut that buffer only to 9.1 GiB, and moving layers to the master
would leave the master about 5 GiB of RAM. `llama-fit-params`, built from the same
tree into `~/llama.cpp/rocm/bin` on `amd-halo`, had estimated the buffer at 4.3 GiB.

At 524,288 tokens it fits, just. llama.cpp turns on pipeline parallelism across
the two boxes, which keeps several copies of the compute buffers. Measured at
ready:

| | Master, `amd-halo` | Worker, `amd-halo-2` |
|---|---|---|
| GPU memory | 113.7 GiB | 110.9 GiB |
| Available | 6.4 GiB | 10.5 GiB |

The master's RAM is the limit, so the test ran with `--cache-ram 0
--ctx-checkpoints 1`: no prompt cache, and one context checkpoint per slot. Under
the benchmark, the master's available memory settled at 5.3 GiB.

Speed under the catalog protocol, measured from the Mac and not saved to the
history:

| One request, end to end | After the first token | First token | Two at once, combined |
|---|---|---|---|
| 11.6 tok/s | 14.3 tok/s | 4.2 s | 15.5 tok/s |

The model reasoned although the requests asked for thinking off, so part of those
tokens are reasoning. 55–64% of the MTP drafts were accepted. Prompt reading on the
580-token prompts ran at about 125 tok/s.

A recall test at 500K tokens was stopped after 39K tokens. Prompt reading slowed
about linearly with depth, from 235 tok/s near the start to 101 tok/s at 35K
tokens: each 6K-token window took about 7 seconds longer than the one before. At
that slope the whole prompt would have taken roughly 7 hours. The master's
`llama-server` also grew by about 7 KB per token read, which over 500K tokens
would have left the master about 1.4 GiB of RAM. Nemotron, for comparison, still
read 64 tok/s at 1M tokens. The recall test therefore ran at 131,072 tokens.

At 131,072 tokens memory was comfortable. At ready the master had 97.4 GiB on its
GPU and 22.9 GiB available, and the worker 100.8 GiB and 21.0 GiB. The lowest
during the test were 20.7 and 18.5 GiB. The document was 119,295 tokens, with the
codes at positions 5,762, 58,844 and 113,172. The model recalled all three
exactly, after a short reasoning passage, and answered the short question
correctly. Reading took 27.5 minutes, 72 tok/s on average:

| Tokens read | Elapsed | Rate over the previous ~20K tokens |
|---|---|---|
| 20,489 | 1.6 min | 211 tok/s |
| 61,449 | 8.7 min | 81 tok/s |
| 100,361 | 20.0 min | 52 tok/s |
| 119,303 | 27.3 min | 43 tok/s |

Decode at that length ran at 9.1 tok/s, with 81% of the drafts accepted.

At this quant GLM is slower than Nemotron Q8 and Qwen Q8 in the cluster, reads
long prompts far more slowly, and needs nearly all of both boxes' memory beyond
128K tokens. UD-IQ4_XS (146 GiB) is about 20 GiB smaller per box, more than the
shortfall at 1M. A smaller quant shrinks the weights, not the cache or the compute
buffers that grow with context. It was not tested. The UD-Q4_K_XL files stay on
`amd-halo`, and `amd-halo-2`'s RPC cache holds the worker's share, about 95 GiB.

## Operating

```sh
./spark-serve strix status                                    # both boxes, roles, models on disk
./spark-serve strix up halo-qwen38-cluster --box amd-halo-2   # amd-halo-2 master, amd-halo worker
./spark-serve strix up halo-cluster-512k --box amd-halo-2     # replaces the cluster model that master serves
./spark-serve strix up halo-nemotron --box amd-halo           # Nemotron 3.5 Super at 1,048,576 tokens
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

The app's AMD Strix Halo tab has one panel per box. Each shows the box's state,
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
11. `amd-halo` had the distribution's `lemond.service` enabled, listening on
    localhost only. While it ran, every start and stop from the Mac failed with
    "Could not stop lemond.service": `halo-ctl` cannot stop a system unit over
    SSH, because polkit asks for a password. It has been disabled since
    2026-10-24, as on `amd-halo-2`.

    Lemonade cannot use the cluster as installed. It runs the distribution's
    `llama-server`, build 10566, which has no `--rpc` option; the workers run
    b11849. Lemonade's `defaults.json` has `llamacpp.rocm_bin` for a custom
    binary, `llamacpp.rocm_args` for extra arguments, and `llamacpp.prefer_system`,
    true by default. Pointing them at `~/llama.cpp/rocm/bin/llama-server` with
    `--rpc amd-halo-2:50053` might work; it was not tested. The arguments would
    apply to every Lemonade model, so small models would be split across both
    boxes too, and Lemonade would share the worker with the cluster engine.
