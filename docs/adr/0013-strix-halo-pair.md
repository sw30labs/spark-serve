# ADR 0013: Keep the Strix Halo pair consistent from the Mac

Accepted · Recorded 2026-10-24

**Context.** A second Strix Halo joined the first. Each box runs one engine at
a time through its own `halo-ctl` and lock, shared with its GNOME widget. A
fourth engine, the llama.cpp cluster, serves one model across both boxes: the
box that starts it is the master, and the other box's always-on
`ggml-rpc-server` holds its share of the weights. Neither box's `halo-ctl`
sees the other's engines, so a worker box reports itself stopped and would
start a second model into memory the cluster already holds.

**Decision.** `[strix]` lists both hosts. Every box is probed in parallel.
The box running the cluster is its master, and the other configured box is its
worker; a box with an established connection to its RPC port is also a worker.
spark-serve applies one rule across the pair: starting the cluster stops
whatever the other box serves, starting a model on the worker stops the
cluster first, and stopping the worker stops the cluster. Each box still
switches its own engines through `halo-ctl`; spark-serve never starts or stops
engines directly. What a box can serve is read from `halo-ctl models`, so the
app and menu bar offer what the widget offers, including models that are on
disk but not in the catalog.

**Trade-off.** The rule holds only for starts made from the Mac. The two
`halo-ctl` locks are separate, so a widget on one box can still start an
engine during a cross-box change, and the widget on the worker can start a
model while the cluster holds its memory. Box-side enforcement belongs in
`halo-ctl`. Polling skips a box that failed in the last minute, so a recovered
box can take up to a minute to reappear.

**Evidence.** [CLI](../../spark_serve_strix.py), [app](../../gui/StrixView.swift),
[tests](../../tests/test_strix.py), [runbook](../strix-halo-pair.md),
[earlier analysis](../strix-halo-analysis.md).
