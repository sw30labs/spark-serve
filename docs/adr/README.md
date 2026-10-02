# Architecture decisions

The initial records describe shipped code at `585eeeb` and existing runbooks;
later records and amendments cover subsequent changes.
Dates below are record dates, not reconstructed approval dates. All are accepted.

| ADR | Decision |
|---|---|
| [0001](0001-native-app-cli.md) | One native app; Python owns remote operations |
| [0002](0002-node-allocations.md) | Physical nodes, logical allocations, separate client selection |
| [0003](0003-fenced-lifecycle.md) | Serialize and fence lifecycle transitions |
| [0004](0004-recipe-preparation.md) | Prepare recipes before disrupting workloads |
| [0005](0005-yue-worker-boundary.md) | Spark Serve admits YuE workers; Artist Twin owns jobs |
| [0006](0006-ephemeral-telemetry.md) | Temporary collectors and explicit metric semantics |
| [0007](0007-inference-benchmark-method.md) | Bounded inference tests with server token accounting |
| [0008](0008-benchmark-leases.md) | Revocable benchmark leases cover physical hosts |
| [0009](0009-yue-capacity-experiments.md) | Isolate YuE capacity experiments from production |
| [0010](0010-nim-backend.md) | Run NIM through the existing fenced controller |
| [0011](0011-tensorfold-recipe.md) | Add TensorFold as a separate managed recipe |
| [0012](0012-host-poweroff.md) | Power off both Sparks through the fenced CLI |

Keep ADRs short: context, decision, trade-off, evidence. Amend clarifications in
place. For a changed decision, add a new numbered ADR and link the superseded
record both ways. Keep commands, measurements, and model-specific fixes in runbooks.
