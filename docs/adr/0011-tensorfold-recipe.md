# ADR 0011: Add TensorFold as a separate managed recipe

Accepted · Recorded 2026-09-29

**Context.** TensorFold offers a different Qwen runtime and checkpoint. It needs
its own launch arguments and health metrics while sharing the same GPUs.

**Decision.** Add a pinned single-node `qwen38-tensorfold` recipe with four
streams and int8 KV. Reuse preparation-before-switch, container ownership,
generation fencing, node placement, benchmarks and both Hermes destinations.
Keep existing Qwen recipes available. Use typed TensorFold launch settings and
an explicit health adapter; preserve missing metrics as unavailable.

**Trade-off.** Separate weights consume disk. Runtime and quantization changes
require fresh hardware and Hermes qualification; upstream speed reports do not
qualify this installation.

**Evidence.** [Recipe](../qwen38-tensorfold.md),
[catalog](../../models.example.toml), [telemetry decision](0006-ephemeral-telemetry.md).
