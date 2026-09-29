# CLUSTER-DOUBLE-BIND — two physical QSFP cables

**Verified recovery: 2026-09-12.** DeepSeek vision served successfully on both Sparks, Hermes answered through the head hostname, and actual inference generated RDMA traffic on **both physical cables**. The working configuration is persistent. No host reboot was needed.

“Double bind” in this filename means using both directly connected QSFP cables through separate RoCE interfaces. There is **no Linux bond, LACP, bridge, or combined management interface**. Each physical QSFP port exposes two logical Ethernet/RoCE interfaces; two cables therefore provide four logical interfaces per node. This follows NVIDIA's [physical-to-logical mapping](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html).

This record describes the verified `ds4-vision` recipe. Its four-HCA overrides are model-specific: the shared `[cluster.nccl]` defaults and other model recipes were not converted or validated for four-HCA operation. Preserve those overrides when maintaining the vision recipe.

## Hosts, addresses, and files

The Mac repository is `/Users/spider/REPOS/spark-serve`. The spelling `/Users/spider/REPOs/spark-serve` resolves to the same directory on this machine.

| Role | SSH alias on Mac | Verified hostname | LAN IPv4 observed during recovery | Kernel during successful recovery |
| --- | --- | --- | --- | --- |
| Head / rank 0 | `sparkone` | `sparkone.local` | `192.168.86.36` | `6.17.0-1032-nvidia` |
| Worker / rank 1 | `sparktwo` | `spark-two.local` | `192.168.86.38` | `7.0.0-1019-nvidia` |

Both nodes used NVIDIA driver `580.173.02`. vLLM's PyNccl library reported NCCL `2.30.7`; PyTorch's separate bundled NCCL reported `2.28.9` in the diagnostic containers. Do not infer the active vLLM library from PyTorch's version alone.

Use **`http://sparkone.local:8000/v1`** for the model API, **`http://sparkone.local:8000/health`** for health, and **`http://sparkone.local:8000/v1/models`** for discovery. `sparktwo.local` was not a valid mDNS name. The LAN IPs above are observations, not addresses to hard-code into clients.

The original saved head address, `192.168.86.44`, had become stale while the head was reachable at `.36`. This explained client/launcher connection failures independently of the model's later RDMA startup error. DHCP renewal after cable changes was plausible but was not established from lease history.

| File/location | Purpose and saved state |
| --- | --- |
| [models.toml](models.toml) | Actual local catalog; `cluster.lan_url = "http://sparkone.local:8000"`; contains the working vision overrides below. This file is locally maintained/ignored by Git. |
| [spark-serve](spark-serve) | CLI reads the adjacent catalog; constructs containers, probes readiness, and updates Hermes. |
| [spark_serve_controller.py](spark_serve_controller.py) | Serializes workload transitions, drains managed jobs, and retains protected containers. |
| [gui/SparkServeApp.swift](gui/SparkServeApp.swift) | Thin app front end; invokes the CLI. Running app: `gui/SparkServeApp.app`, bundle ID `local.sparkserve.app`. |
| `/Users/spider/.hermes/config.yaml` | Selected provider `spark`, selected vision model, hostname API URL, and per-model contexts. |
| `/Users/spider/.ssh/config` | `sparktwo` now uses `Hostname spark-two.local`; existing user/key selection preserved. |
| `/Users/spider/.ssh/known_hosts` | Hostname alias added only after its Ed25519 key exactly matched the previously trusted `.38` key. |
| Each Spark: `/etc/netplan/40-cx7.yaml` | Preserved right-port `.100`/`.101` configuration. |
| Each Spark: `/etc/netplan/41-cx7-second-cable.yaml` | Added persistent left-port `.102`/`.103` configuration. |

Hermes' selected `spark` and legacy `spark-ds4` API URLs both use `sparkone.local`. Related saved YuE URLs are `http://sparkone.local:8011` and `http://spark-two.local:8011`. YuE was not the inference workload tested here.

The head must resolve its own `sparkone.local` name too: the CLI readiness probe runs `curl` through SSH on the head. Management remains on the separate `enP7s7` LAN interface and its existing default route. Do not remove SSH host-key checks to fix a hostname migration; compare a new name's key against an already trusted host identity.

## Physical cables and persistent network configuration

Viewed from the back of a Spark, `f0` corresponds to the left QSFP port near the RJ45 Ethernet port; `f1` is the right QSFP port. Keep left-to-left and right-to-right physical connections. Both cables remain connected during normal operation and software fallback testing.

| Physical cable | Ethernet interface | RoCE HCA | Spark One | Spark Two | Persistent file |
| --- | --- | --- | --- | --- | --- |
| Right, preserved | `enp1s0f1np1` | `rocep1s0f1` | `192.168.100.10/24` | `192.168.100.11/24` | `40-cx7.yaml` |
| Right, preserved | `enP2p1s0f1np1` | `roceP2p1s0f1` | `192.168.101.10/24` | `192.168.101.11/24` | `40-cx7.yaml` |
| Left, added | `enp1s0f0np0` | `rocep1s0f0` | `192.168.102.10/24` | `192.168.102.11/24` | `41-cx7-second-cable.yaml` |
| Left, added | `enP2p1s0f0np0` | `roceP2p1s0f0` | `192.168.103.10/24` | `192.168.103.11/24` | `41-cx7-second-cable.yaml` |

All four use **MTU 9000**, static IPv4, and `dhcp4: false`. No gateway was added to a QSFP interface. `.102` and `.103` are the local extension of the existing subnet scheme, not a mandatory NVIDIA addressing convention. Original management configuration, routes, and the right-port addresses were preserved.

The added file on **Spark One** is:

```yaml
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      addresses: [192.168.102.10/24]
      dhcp4: false
      mtu: 9000
    enP2p1s0f0np0:
      addresses: [192.168.103.10/24]
      dhcp4: false
      mtu: 9000
```

The added file on **Spark Two** is:

```yaml
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      addresses: [192.168.102.11/24]
      dhcp4: false
      mtu: 9000
    enP2p1s0f0np0:
      addresses: [192.168.103.11/24]
      dhcp4: false
      mtu: 9000
```

The preserved `40-cx7.yaml` files contain the two `f1` entries from the table, also with MTU 9000 and DHCP disabled. Do not replace the complete machine network configuration with these fragments. Recovery used the supported `netplan set --origin-hint 41-cx7-second-cable` interface to add each `f0` entry, followed by `netplan generate` and `netplan apply`. NetworkManager took a short time after apply to expose the new addresses.

For a future authorized network maintenance window, back up the existing files first, merge only the intended entries, then validate and apply on each affected host:

```sh
sudo netplan generate
sudo netplan apply
ip -br -4 address
ip -4 route
ibdev2netdev
show_gids
```

Verify management SSH remains reachable and every peer pair works. RoCEv2 IPv4 was verified at **GID index 3 on all four HCAs**, on both nodes. Recheck `show_gids` after address or driver changes instead of assuming that index is universally correct. NVIDIA's [two-Spark setup guide](https://build.nvidia.com/spark/connect-two-sparks/stacked-sparks) requires addressing all four interfaces when both cables are used.

## Saved DeepSeek vision recipe

Recipe `ds4-vision` serves **`deepseek-v4-flash-0731-vision`** from image **`dsv4-vision-vllm:0.1.1`**, with tensor parallel size 2. The container model tree is `/cache/huggingface/dsv4-0731-vision`; the host checkpoint cache is `/home/spider/.cache/huggingface`. Existing model assets, vision plugin/tower/adapter mounts, and the remaining inference flags are defined in [models.toml](models.toml) and [docs/ds4-vision.md](docs/ds4-vision.md).

The following are excerpts to merge into the existing tables, not a replacement catalog:

```toml
[cluster]
head = "sparkone"
worker = "sparktwo"
master_addr = "192.168.100.10"
master_port = 29501
lan_url = "http://sparkone.local:8000"

[models.ds4-vision]
docker_extra = ["--ulimit", "memlock=-1:-1"]
shm_size = "64g"
max_model_len = 131072
hermes_context_length = 131072
host_ip_head = "192.168.100.10"
host_ip_worker = "192.168.100.11"

[models.ds4-vision.env]
NCCL_BUFFSIZE = "1048576"
NCCL_MAX_NCHANNELS = "8"
NCCL_IB_HCA = "=rocep1s0f1,roceP2p1s0f1,rocep1s0f0,roceP2p1s0f0"
NCCL_CROSS_NIC = "0"
NCCL_DEBUG = "INFO"
NCCL_NVLS_ENABLE = "0"
```

The effective inherited settings relevant to transport are:

| Setting | Effective value | Origin |
| --- | --- | --- |
| `NCCL_NET` | `IB` | Shared cluster config |
| `NCCL_IB_DISABLE` | `0` | Shared cluster config |
| `NCCL_IB_GID_INDEX` | `3` | Shared cluster config; verified GIDs |
| `NCCL_IB_MERGE_NICS` | `1` | Shared cluster config |
| `NCCL_NET_PLUGIN` | `none` | Shared cluster config |
| `NCCL_CUMEM_ENABLE` | `0` | Shared cluster config |
| `NCCL_IGNORE_CPU_AFFINITY` | `1` | Shared cluster config |
| `NCCL_SOCKET_IFNAME` | `enp1s0f1np1` | Bootstrap/control on existing right rail |
| `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME` | `enp1s0f1np1` | Existing control interface |
| `NCCL_IB_HCA` | Four HCAs from the excerpt | Vision override of the shared two-HCA default |
| `NCCL_CROSS_NIC` | `0` | Vision override of shared value `1` |
| `NCCL_DEBUG` | `INFO` | Vision override of shared `WARN` |

The launcher appends model environment values after shared values; effective values were also verified in the created containers. The leading `=` in the HCA value requests exact names. `NCCL_CROSS_NIC=0` keeps corresponding network interfaces paired for the directly connected topology. No guessed bonding or routing override is involved. See the [NCCL 2.30.7 environment reference](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2307/user-guide/docs/env.html) for parameter semantics.

`IB` is NCCL's verbs transport name here: the physical fabric runs **Ethernet/RoCEv2**, not InfiniBand link mode. The successful service uses native RDMA. `NCCL_CUMEM_HOST_ENABLE=0` was ineffective in the full workload and was removed. The diagnostic `Socket` transport was not retained in production.

The CLI already creates these containers with host networking, host IPC, GPU access, and the RDMA devices. The added `--ulimit memlock=-1:-1` was verified as unlimited in Docker's actual configuration. Unlimited memlock alone did not resolve the failure. The successful configuration combines **1 MiB communication buffers and at most eight channels** with the four-HCA recipe. Treat these as a validated workaround for this deployment, not universal optimal tuning.

## Hermes and context length

The selected provider is `spark`, using the hostname API and the vision model above. Its saved per-model `context_length` is **131,072**. Inferencer's separate `inferencerlabs/DeepSeek-V4.1-MLX-Q4i` profile retains **1,048,576**, through `http://127.0.0.1:54323/v1`.

`_hermes_patch` in [spark-serve](spark-serve) now removes the global `model.context_length` override when publishing a recipe's context and stores the value under that provider/model. Previously the global override could incorrectly carry one model's limit into another provider. [tests/test_hermes_context.py](tests/test_hermes_context.py) checks that the Spark context is scoped correctly and existing Inferencer/other model entries survive.

**131.1K in Hermes is the selected Spark server's configured window.** It does not establish a model family's native maximum. Raising only Hermes' number cannot raise the server's capacity. The server flags, reported `/v1/models` limit, and Hermes value must agree. Neither a 1M-token Spark request nor maximum-context retrieval quality was tested in this recovery. Existing output limit and unrelated model settings were preserved.

## Verified results and their limits

These are the completed September 12 measurements, not a claim that health is monitored continuously afterward.

| Check | Result |
| --- | --- |
| Full startup | Ready after about **339 seconds of readiness waiting**, including loading, memory profiling, and warmup |
| Health and model discovery | HTTP **200** through the hostname; exact model ID; `max_model_len=131072` |
| Fresh real Hermes agent | Successful greeting in **12.26 s**, one API call, no reported failure |
| Native tool call | Structured `ping` request with synthetic test message; `finish_reason=tool_calls`; **8.45 s** |
| Vision | Correctly answered **Blue** for a generated solid-blue image; **0.47 s** |
| Actual inference traffic | Approximately **410 MB transmitted and received per physical cable, per node**, during the greeting/tool/vision test interval |
| Interface counters | No increment in RX/TX errors or dropped packets during that interval |
| Service and protected workload | Both `vllm_cluster` containers running; `singularity-atlas-neo4j` remained healthy |
| Visible app state | `serving deepseek-v4-flash-0731-vision`, hostname URL visible, vision card selected |
| Persistence | Actual local catalog and Hermes settings read back and checked |
| Context regression | Focused offline test passed; source diff whitespace check passed |

The inference counter interval was **2026-09-12 21:35:43–21:37:56 UTC**. Summed logical-HCA counter deltas for each physical cable, converted to bytes, were:

| Node | Cable | TX bytes | RX bytes |
| --- | --- | ---: | ---: |
| Spark One | Left | 410,072,880 | 410,071,708 |
| Spark One | Right | 410,106,260 | 410,102,296 |
| Spark Two | Left | 410,071,708 | 410,072,880 |
| Spark Two | Right | 410,102,296 | 410,106,260 |

Counters cover the interval containing the synthetic inference checks, not attribution to each individual request. The physical-cable grouping comes from the interface mapping above. These counters plus live NCCL four-HCA selection distinguish **link up**, **selected by NCCL**, and **actually carrying workload traffic**.

### Raw bandwidth comparison

With the model stopped, an eight-second-per-run `ib_write_bw` comparison used 64 KiB messages, one QP per process, GID 3, and concurrent processes on the selected HCAs. All paths passed. “One cable” selected only the two `f1` HCAs; both cables stayed physically connected.

| Selected paths | Aggregate measured bandwidth |
| --- | ---: |
| One physical cable, two logical HCAs | **196.031912 Gb/s** |
| Both physical cables, four logical HCAs | **206.869193 Gb/s** |

This is about **5.5% higher in one short raw RDMA comparison**. It is not a repeated statistical performance study and **not a measured model-token-throughput gain**. No doubling is claimed. NVIDIA says one cable can already provide full bandwidth in its [two-Spark guide](https://build.nvidia.com/spark/connect-two-sparks/stacked-sparks).

The [September 5 lab record](https://github.com/sw30labs/dgx-spark-roce-lab/blob/df9f3d8bac527920d45400814101ae0ca26941a1/docs/index.md) described one N911 cable on the right port and two logical rails; its left cages were unused. Historical “dual HCA” numbers therefore did not prove two-physical-cable use. The local September 2 file `spark-qsfp-dual-nccl-n911-2026-09-02.txt` likewise names only `rocep1s0f1,roceP2p1s0f1`. The current four-interface addresses and direct inference counters establish the September 12 two-cable result.

## Future verification and restart

Run these from the Mac; they inspect rather than restart:

```sh
cd /Users/spider/REPOS/spark-serve
./spark-serve status --json
curl --fail --silent --show-error http://sparkone.local:8000/health
curl --fail --silent --show-error http://sparkone.local:8000/v1/models
ssh -o BatchMode=yes sparkone hostname
ssh -o BatchMode=yes sparktwo hostname
ssh sparkone 'getent ahostsv4 sparkone.local; ip -br -4 address; ibdev2netdev; show_gids'
ssh sparktwo 'ip -br -4 address; ibdev2netdev; show_gids'
```

Expect the exact vision model, `ready: true`, `phase: ready`, the hostname URL, and four active RoCE devices with the intended peer addresses. If API authentication changes later, use the existing approved credential mechanism; do not put credentials into this document or command history.

Check actual container configuration without dumping unrelated environment/credentials:

```sh
ssh sparkone "docker inspect --format '{{json .HostConfig.Ulimits}}' vllm_cluster"
ssh sparktwo "docker inspect --format '{{json .HostConfig.Ulimits}}' vllm_cluster"
ssh sparkone 'docker exec vllm_cluster printenv NCCL_NET NCCL_IB_HCA NCCL_BUFFSIZE NCCL_MAX_NCHANNELS NCCL_CROSS_NIC'
ssh sparktwo 'docker exec vllm_cluster printenv NCCL_NET NCCL_IB_HCA NCCL_BUFFSIZE NCCL_MAX_NCHANNELS NCCL_CROSS_NIC'
```

For fresh traffic proof, snapshot each HCA's `/sys/class/infiniband/<HCA>/ports/1/counters/port_xmit_data` and `port_rcv_data` immediately before and after a bounded inference request. Subtract snapshots, multiply these standard data counters by four to convert their units to bytes, then sum the two `f0` HCAs for the left cable and the two `f1` HCAs for the right. Check Ethernet `rx_errors`, `tx_errors`, `rx_dropped`, and `tx_dropped` deltas under `/sys/class/net/<interface>/statistics/`. Avoid interpreting lifetime totals as evidence for a new request.

For example, on either Spark, run this snapshot command before and after the request:

```sh
for hca in rocep1s0f0 roceP2p1s0f0 rocep1s0f1 roceP2p1s0f1; do
  printf '%s TX=' "$hca"
  cat "/sys/class/infiniband/$hca/ports/1/counters/port_xmit_data"
  printf '%s RX=' "$hca"
  cat "/sys/class/infiniband/$hca/ports/1/counters/port_rcv_data"
done
```

Only during an authorized workload transition, coordinate other Spark users/jobs and use the normal controller:

```sh
cd /Users/spider/REPOS/spark-serve
./spark-serve up ds4-vision --json
```

`up` replaces the currently managed serving workload. It performs the stop/start sequence itself; a separate `stop` is not normally required. It protects the configured Atlas container and drains managed YuE work. Respect a controller conflict/drain error rather than manually removing arbitrary containers or locks. Use `--no-hermes` only when intentionally preserving the client's current selection; otherwise the repaired Hermes updater applies the selected recipe and per-model context. Run `hermes` afterward, or use Hermes' normal resume selector for an existing conversation.

The CLI loads its catalog once per invocation. An already running readiness loop can retain an obsolete address after a file edit. Check the real API and logs independently before classifying a long “booting” display as model loading. If only the app display is stale and no launch/transition is active, try Refresh, then quit/reopen **only the Spark Serve interface**. During this recovery that cleared a stale exit message without restarting the healthy service. Reselect the vision card if the reopened app defaults to a different card; do not press Start merely to refresh the display.

For an optional repeat of a single RDMA path test, with the model idle/stopped and no conflicting benchmark, start the server first and client in another session:

```sh
ssh sparkone 'timeout 25 ib_write_bw -d rocep1s0f1 -x 3 -p 19001 -s 65536 -q 1 -D 8 --report_gbits --output=bandwidth'
ssh sparktwo 'timeout 25 ib_write_bw -d rocep1s0f1 -x 3 -p 19001 -s 65536 -q 1 -D 8 --report_gbits --output=bandwidth 192.168.100.10'
```

To reproduce the recorded aggregate comparison, run matching server/client pairs **concurrently** with unique ports: `rocep1s0f1/.100` on 19001, `roceP2p1s0f1/.101` on 19002, then add `rocep1s0f0/.102` on 19003 and `roceP2p1s0f0/.103` on 19004 for the four-path case. Sum one side's four results, not server plus client. Do not run bandwidth or large NCCL benchmarks during full model loading or alongside latency-sensitive inference.

## Failure diagnosis and rollback boundaries

The full model originally failed with `ibv_reg_mr_iova2 ... Cannot allocate memory`. A shared-memory broadcast wait on the other rank was a downstream symptom, not evidence that `/dev/shm` was full. Raw RDMA tests and small NCCL collectives could succeed while full model initialization/profiling failed. Available RAM and unlimited memlock alone were insufficient explanations or fixes. NVIDIA's [networking troubleshooting guide](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2307/user-guide/docs/troubleshooting/networking_troubleshooting.html) explains memory-registration failures; inspect actual worker/container limits rather than only the login shell.

The kernel mismatch above is an **unproven possible contributor**. A [community report](https://forums.developer.nvidia.com/t/383023) describes the same newer-kernel error and recovery with an older kernel, but this recovery did not perform a kernel-only A/B test, and that discussion is not vendor confirmation. The saved smaller-buffer/eight-channel configuration succeeded on the existing kernels. **Booting an older kernel is not required for this working setup.** Normal sudo/systemd reboot attempts required interactive authentication; no privilege policy was bypassed, changed, or weakened.

If the failure returns, retain the first failing rank's logs, effective NCCL values, actual process locked-memory limit, host memory, and kernel/library versions before restarting. Verify the vision overrides survived catalog edits. Preserve the working settings while evaluating a specific change in a maintenance window; do not assume a cable fault from `ENOMEM` or blindly transplant discrete-GPU GPUDirect fixes.

Rollback means a scoped configuration change with a known purpose, not restoring every old file:

1. **Before any rollback**, preserve the current working catalog/network entries, coordinate jobs, and stop the managed model through `./spark-serve stop` only when a transition is intended. Retain `singularity-atlas-neo4j` and the management route.
2. **Optional one-cable software fallback:** keep both cables physically connected, select only the two `f1` HCAs in the vision override, and retain the hostname plus working buffer/channel limits initially. Recreate the service through the controller and verify it. This is no longer the documented two-cable workload configuration, and its full-model behavior with the current workaround was not separately measured.
3. **Network rollback:** if deliberately removing the added left-port addresses, remove only the two `f0` entries from `41-cx7-second-cable.yaml` on each host after backing it up. Remove the file only if it contains nothing else. Preserve `40-cx7.yaml`, management settings, and both physical cables; validate with `netplan generate`, apply, and verify SSH/peer connectivity. Do not remove addresses while an active NCCL job uses them.
4. **Client rollback:** restore individual fields only when the intended destination is verified. Old `.44` URLs are stale and old global Hermes context values can be wrong for another model; do not restore them wholesale from a backup.
5. **Runtime tuning rollback:** removing the small-buffer/channel workaround can reproduce the startup failure. Any such experiment needs a planned service restart and repeat of readiness plus real inference checks. Kernel or driver changes require their own justified maintenance procedure and existing administrator authentication.

## Evidence and backup locations

The document contains the topology, saved values, and principal measurements so it is usable without the task workspace. Detailed original receipts remain on this Mac; these paths are provenance, not portable dependencies:

`/Users/spider/Documents/Codex/2026-09-11/new-chat/work/spark-repair/`

| Evidence/backup | Contents |
| --- | --- |
| `persistence-verification.json` | Final saved transport, four HCAs, tuning, hostnames, and contexts |
| `final-status.json` | Final ready controller/model/container state |
| `verification-result.json` | Fresh Hermes result and effective context |
| `api-verification-result.json` | Health, model discovery, synthetic tool and vision results |
| `inference-cable-proof.json` | Exact inference interval and per-cable counter/error deltas |
| `network-counters-before-inference.json`, `network-counters-after-inference.json` | Raw counter snapshots supporting that calculation |
| `cable-bandwidth-results.json`, `test-cables.py` | Short raw RDMA comparison and its orchestration |
| `hostname-receipt.json`, `two-cable-network-receipt.json`, `service-settings-receipt.json` | Scoped changes and backup paths |
| `sparkone-netplan-ethernets-20260912T211418Z.json`, `sparktwo-netplan-ethernets-20260912T211418Z.json` | Preserved original right-port Ethernet entries |
| `models.toml.before-20260912T211644Z` | Catalog before four-HCA/tuning recovery; not a whole-file restore recommendation |
| `spark-serve-before-context-fix` | Prior CLI source for the scoped Hermes context correction |

Additional scoped Hermes field backups are in `/Users/spider/.hermes/backups/`. Historical local references are `/Users/spider/REPOS/ADR-001-dual-qsfp-interconnect.md`, `/Users/spider/REPOS/spark-qsfp-dual-nccl-n911-2026-09-02.txt`, and its `-full.log` companion. Historical address tables may predate the current port/subnet mapping; use the verified September 12 table above for this deployment. This document includes no credentials or private conversation/session content.
