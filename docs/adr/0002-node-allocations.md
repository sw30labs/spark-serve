# ADR 0002: Separate nodes, allocations, and client selection

Accepted · Recorded 2026-09-26 (retrospective)

**Context.** Two Sparks can serve independent models or host two ranks of one
model. An endpoint or model name alone cannot prove ownership.

**Decision.** Track each physical host's generation and immutable container
receipts. Treat distributed ranks as one allocation with one inference endpoint;
reject partial replacement. Scope solo changes to the selected host. Expose
allocation identity and engine runtime separately from lifecycle mode. Hermes
selection chooses a client endpoint; it does not move workloads or determine
monitoring scope.

**Trade-off.** Independent workloads remain independent, but ambiguous distributed
ownership requires both-node reconciliation. This is explicit two-node placement,
not an automatic scheduler. Legacy unscoped `up` retains its Hermes retargeting.

**Evidence.** [Placement](../../spark_serve_nodes.py),
[controller](../../spark_serve_controller.py), [node tests](../../tests/test_node_controller.py),
[compatibility rules](../independent-sparks.md).
