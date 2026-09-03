# Spark QSFP / dual-HCA (this house)

Followable restore for the two DGX Sparks. NVIDIA's cluster guide is still
the pairing/OS baseline. This file is the cable, the two 200G rails, NCCL
pins, and the checks that proved it. Dated 2026-09-02. Re-verify live
`ip` / `show_gids` before you retune anything.

Do not stop `singularity-atlas-neo4j`. Do not point Mac SSH at fabric IPs.

## What is plugged in

One cable, **right QSFP cage on both boxes**. Left cage empty.

| | |
|---|---|
| Live cable | NVIDIA QSFP DAC **NJAAKK-N911** (400Gb Gen2, 32 AWG, LSZH). Label may read NJAAK-N911. |
| Replaced | Amphenol **NJAAKK-AU06** (ASUS Ascent GX10, Amazon B0GV4ZR8BT). Keep it unplugged. |
| Seating | Right cage = ConnectX `f1`. After plug, `f1` is UP and `f0` is DOWN. If `f0` came up, you used the left cage — move the cable; do not retune NCCL to `f0`. |
| Link | **200 Gb/s, 2X NDR**, DAC, MTU 9000, FEC RS. 200G is the spec on this split, not a miss vs 400G. It will not train to 400G on this pair. |

One N911 already presents **two** 200G netdevs on each Spark. That is not a
second cable. Do not plug AU06 into the left cage unless you want a third and
fourth 200G netdev. Mixing AU06+N911 is electrically fine and does **not**
change Nightshift ETA.

## Boxes and addresses

SSH from the Mac Studio stays on the LAN aliases. Never rewrite those to
`.100` / `.101`.

```
ssh -t sparkone "cd /home/spider/; bash -l"
ssh -t sparktwo "cd /home/spider/; bash -l"
```

| Role | hostname | LAN (`enP7s7`) | `.100` (`enp1s0f1np1` / `rocep1s0f1`) | `.101` (`enP2p1s0f1np1` / `roceP2p1s0f1`) |
|---|---|---|---|---|
| head | `sparkone` | 192.168.86.44/24 | 192.168.100.10/24 | 192.168.101.10/24 |
| worker | `spark-two` (alias `sparktwo`) | 192.168.86.38/24 | 192.168.100.11/24 | 192.168.101.11/24 |

vLLM master is the **head's `.100`**: `192.168.100.10:29501`.
Clients hit LAN: `http://192.168.86.44:8000`.

Firmware seen 2026-09-02: ConnectX-7 **MT4129** fw `28.45.4028` (`ibdev2netdev -v`).

## How the fabric IPs persist

Netplan, not DHCP. Files are root-only (`/etc/netplan/40-cx7.yaml`).
NetworkManager connections:

| box | connection | device | ipv4.method | address | mtu |
|---|---|---|---|---|---|
| sparkone | `netplan-enp1s0f1np1` | `enp1s0f1np1` | manual | 192.168.100.10/24 | 9000 |
| sparkone | `netplan-enP2p1s0f1np1` | `enP2p1s0f1np1` | manual | 192.168.101.10/24 | 9000 |
| spark-two | `netplan-enp1s0f1np1` | `enp1s0f1np1` | manual | 192.168.100.11/24 | 9000 |
| spark-two | `netplan-enP2p1s0f1np1` | `enP2p1s0f1np1` | manual | 192.168.101.11/24 | 9000 |

If a recable drops addresses:

```
# on sparkone; swap .10 -> .11 on sparktwo
sudo nmcli con mod netplan-enp1s0f1np1 ipv4.method manual ipv4.addresses 192.168.100.10/24 802-3-ethernet.mtu 9000
sudo nmcli con mod netplan-enP2p1s0f1np1 ipv4.method manual ipv4.addresses 192.168.101.10/24 802-3-ethernet.mtu 9000
sudo nmcli con up netplan-enp1s0f1np1
sudo nmcli con up netplan-enP2p1s0f1np1
```

Then confirm GIDs (next section). IPv4 RoCEv2 is index **3**. An empty
`::` at index 3 on either rail on either box used to kill TP=2.

## RoCE GID pin

`NCCL_IB_GID_INDEX=3` is IPv4 RoCEv2. Healthy `show_gids` (both boxes, both
cabled HCAs):

```
rocep1s0f1   index 3   192.168.100.10 or .11   v2
roceP2p1s0f1 index 3   192.168.101.10 or .11   v2
```

`f0` HCAs stay DOWN (no cable) with only link-local GIDs. That is expected.

## NCCL / spark-serve pins

Do not switch these to `f0` unless the cable moves to the left cage.

Live catalog is gitignored `models.toml`. The committed twin is
`models.example.toml` `[cluster.nccl]` plus `models.ds4.env`:

```
NCCL_IB_DISABLE=0
NCCL_IB_HCA==rocep1s0f1,roceP2p1s0f1
NCCL_IB_GID_INDEX=3
NCCL_IB_MERGE_NICS=1
NCCL_CROSS_NIC=1
NCCL_NET_PLUGIN=none
NCCL_NET=IB
NCCL_SOCKET_IFNAME=enp1s0f1np1
GLOO_SOCKET_IFNAME=enp1s0f1np1
TP_SOCKET_IFNAME=enp1s0f1np1
UCX_NET_DEVICES=rocep1s0f1:1,roceP2p1s0f1:1
```

The leading `=` on `NCCL_IB_HCA` is exact-match. Do not drop it.
`master_addr` is `192.168.100.10` (head `.100`, not LAN).
`keep_containers` must include `singularity-atlas-neo4j`.

House `models.toml` also has `head = "sparkone"`, `worker = "sparktwo"`,
`lan_url = "http://192.168.86.44:8000"`.

Bring DS4 back with `./spark-serve up ds4` from this repo on the Mac.
Atlas stays up. Local oMLX / GLM on the Mac is never this CLI's job.

## Recable checklist

1. Leave Atlas running. `spark-serve stop` is allowed (it must not touch
   `keep_containers`). Prefer not to yank power under vLLM.
2. Unplug the old DAC. Seat **N911 in the right cage** both ends until it
   clicks. Left cage stays empty.
3. From the Mac, SSH **LAN** aliases. Confirm:

```
ssh sparkone 'ip -br addr; ibdev2netdev; show_gids | grep -E "rocep1s0f1|roceP2p1s0f1"'
ssh sparktwo 'ip -br addr; ibdev2netdev; show_gids | grep -E "rocep1s0f1|roceP2p1s0f1"'
```

Expect `enp1s0f1np1` and `enP2p1s0f1np1` UP, both `f0` netdevs DOWN, GID
index 3 populated IPv4 RoCEv2 on both cabled HCAs, ethtool `200000Mb/s`.

4. If IPs are missing, replay the `nmcli` block above, then `show_gids`
   again. Empty GID 3 on sparktwo `.101` was the old TP killer.
5. `ping -c 5 -I 192.168.100.10 192.168.100.11` (and the `.101` pair).
6. `./spark-serve up ds4` and `curl -sS http://192.168.86.44:8000/v1/models`.
7. Optional: iperf + `ib_write_bw` + dual-HCA NCCL (below). Dual-HCA NCCL
   **stops DS4**; restore DS4 afterward. Atlas stays.

## Healthy numbers (under live TP=2 unless noted)

Same ~109 Gb/s RDMA ceiling on AU06 and N911. The cable EEPROM was not the
old ~13 Gbit/s wall. Nightshift ETA did not move with the cable.

TCP used `iperf` 2.1.9 (iperf3 not installed; sparktwo has no passwordless
sudo). RDMA: `ib_write_bw` / `ib_read_bw` `-d <hca> -x 3 -D 5 -f 1 -s 1048576 --report_gbits`.

| test | TCP Gbps (two→one / one→two) | RDMA write/read |
|---|---|---|
| AU06 `.100` 18:15 ET | 52.8 / 42.0 | ~109.3 |
| N911 `.100` 19:06 ET | 47.9 / 51.9 | ~109.3 |
| N911 `.101` 21:01 ET | 52.0 / 43.6 | ~109.3 |

### Dual-HCA NCCL all_gather (DS4 down, Atlas up)

Script on sparkone: `/home/spider/launch-roce.sh`.

**Override HOSTLIST.** The script default
`192.168.86.249:1,192.168.86.47:1` is stale.

```
# on sparkone; this STOPS the vLLM cluster. Atlas stays.
HOSTLIST=192.168.86.44:1,192.168.86.38:1 \
  NCCL_SOCKET_IFNAME=enP7s7 \
  /home/spider/launch-roce.sh
```

Pins inside the script (already correct): `NCCL_IB_HCA==rocep1s0f1,roceP2p1s0f1`,
`NCCL_IB_GID_INDEX=3`, bootstrap/OOB on LAN `enP7s7`.

16G all_gather n=2 (2026-09-02 21:09 ET):

- avg busbw **20.34 GB/s** (OOP 20.04 / in-place 20.65)
- both HCAs in NCCL INIT (`ndevs=2`)
- GDR disabled (GB10 expected)
- old wall ~2.94 GB/s; healthy no-GDR ~22–24 GB/s

Then `./spark-serve up ds4` from the Mac.

## Do not

- SSH to `.100` / `.101` from the Mac. Management is LAN.
- Switch `NCCL_IB_HCA` / `UCX_NET_DEVICES` to `f0` while the cable is in
  the right cage.
- Drop the leading `=` on `NCCL_IB_HCA`.
- Stop Atlas.
- Treat a 200G link as a failed 400G link.
- Expect Nightshift nights to get faster from this cable. They did not.
- Trust `/home/spider/launch-roce.sh` HOSTLIST without overriding it.

## Evidence

Mac logs (not in this repo):

- `~/REPOS/spark-qsfp-baseline-asus-2026-09-02.txt`
- `~/REPOS/spark-qsfp-after-n911-2026-09-02.txt`
- `~/REPOS/spark-qsfp-101-n911-2026-09-02.txt`
- `~/REPOS/spark-qsfp-dual-nccl-n911-2026-09-02.txt`
