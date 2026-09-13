# Independent workloads on two Sparks

Each Spark can serve its own single-node model. The initial pairing is
Qwen3.8-Flash-Next NVFP4 on the configured head and Nemotron-3-Super NVFP4 on the
configured worker. Both use port 8000 on their respective machines. They have
separate endpoints, model caches, containers, and ownership generations.

Set `cluster.lan_url` to the head's HTTP origin and `cluster.worker_lan_url` to
the worker's HTTP origin. These must be direct addresses of different physical
Sparks, reachable from the Mac and their respective Spark; an SSH alias alone
need not be a resolvable HTTP hostname. Both ports must match `cluster.port`.
Identical host/port pairs are rejected. Different DNS aliases pointing at the
same machine still require correct operator configuration.

If the worker cache path differs, set `cluster.worker_hf_cache_host`; otherwise
the existing `hf_cache_host` path is used on both machines. Worker placement also
uses each mount's `worker` path and worker host-IP override. Solo worker rendezvous
defaults to loopback, keeping the other Spark out of its process group.

## Start and select a model

With Qwen already running on the head:

```sh
./spark-serve pull nemotron-super --node worker
./spark-serve up nemotron-super --node worker
./spark-serve status
```

Preparation and startup target only the selected node. Starting Nemotron does
not stop Qwen or change Hermes's selected endpoint. To switch the client after
either endpoint is ready:

```sh
./spark-serve use --node worker   # select Nemotron in Hermes
./spark-serve use --node head     # return to Qwen
```

`use` verifies the selected endpoint and its recorded container allocation, then
changes Hermes configuration under the controller lock. It does not restart,
drain, or stop a model. The head keeps its catalog provider (normally `spark`);
the worker uses the corresponding `-worker` provider so both endpoints remain
configured. Running client sessions may need to reload their configuration.

For compatibility, `up MODEL` without `--node` still retargets Hermes after
readiness unless `--no-hermes` is supplied. It places a solo model on the head
and a distributed model on both nodes. Explicit `--node` placement leaves client
selection alone. The GUI always makes placement and client selection separate.

## Stop and inspect

```sh
./spark-serve stop --node worker     # leaves the head workload running
./spark-serve logs worker
./spark-serve wait --node worker
./spark-serve stop --node both       # drains and stops both managed workloads
```

The default `stop` remains a stop of both nodes. The GUI labels that action
explicitly and provides separate controls on each hostname panel. Its two model
checkmarks and health indicators remain independent while a peer starts or fails.

`status --json` retains the legacy head endpoint fields and adds `nodes` with
each node's observed model, endpoint, readiness, containers, phase and error.
`active_node` reports Hermes's configured selection. A distributed worker can be
healthy as a headless rank; `can_use` is false because its client API is on the head.

## Ownership and recovery

The existing Mac lock still serializes transitions. Serving is independent:
fencing, draining, stopping, cache clearing, launch and failure cleanup affect
only the requested nodes. An offline unrelated peer does not block a solo change.
Protected containers and unknown GPU workload checks retain their existing rules.

New launches record their immutable container IDs and label their model,
allocation generation and participant hosts. Readiness requires that identity
as well as the expected model response. Legacy single-node state is adapted on
the next locked transition without restarting the healthy server. Peer state and
generation are retained through successful and failed local operations.

A distributed model reserves both Sparks as one allocation. Replacing or stopping
one of its ranks is rejected before any remote mutation; choose both nodes.
Interrupted or ambiguous distributed ownership also requires both-node
reconciliation. A stopped or partially started allocation never becomes ready
merely because another process serves the same model name.

YuE starts still use both Sparks in this iteration. A scoped stop or solo-model
replacement can leave the other admitted YuE worker running; its discovery entry
and generation are preserved. Starting independently admitted YuE workers would
require an update to Artist Twin's discovery contract and is not enabled here.

## Qualification

Offline tests cover migration, independent success/failure and cleanup, offline
peers, protected containers, stale generations, distributed allocation rejection,
physical mount/cache mapping, two endpoints, wrong identities and client selection.
Live qualification receipts are kept under the ignored
`diagnostics/2026-09-13-independent-sparks/` directory. The separate recipe guides
record [Qwen](qwen38-nvfp4.md) and [Nemotron](nemotron-super.md) runtime details.

On September 13, 2026, Nemotron passed all nine text, reasoning, JSON, tool,
retrieval and concurrent-request checks on sparktwo. A separate request to each
Spark ran concurrently and both completed. The app selected Nemotron in Hermes
and returned to Qwen without changing either container ID. The worker-only stop
left the original Qwen container and its ownership generation unchanged, with
the head endpoint still ready. Nemotron then restarted successfully in about
**2 minutes 36 seconds** using its persistent caches. Both models answered a
simultaneous arithmetic check correctly after that restart. Hermes was left on
Qwen, with both model endpoints running. Offline validation passed **345 tests
and 32 subtests**; native GUI fixtures verified independent controls and readable
activity messages.

Continuity monitoring covered **56 samples over 18 minutes 20 seconds** with
zero Qwen availability failures or container restarts. All four intervening Qwen
arithmetic requests passed. The protected Neo4j container remained healthy and
unchanged in all ten checks. The temporary monitors were stopped after
qualification; both model services were left running.
