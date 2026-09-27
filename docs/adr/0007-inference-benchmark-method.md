# ADR 0007: Bounded benchmarks with server token accounting

Accepted · Recorded 2026-09-26 (retrospective)

**Context.** Operational speed comparisons need reproducible settings and meaningful token counts without changing the serving workload.

**Decision.** Send synthetic decode requests or serial prefill sweeps to an already-ready allocation. Bound concurrency, context, request duration, and total duration. Require valid SSE completion and server-reported usage; intentional output-limit finishes are accepted. Record warmup separately and exclude it from measured results. Distinguish first visible content from first model output, including reasoning. Label prefill and post-first-output rates as estimates. Preserve configuration, requested reasoning mode, allocation identity, and results in local history. Thinking defaults to the server's behavior unless explicitly requested otherwise.

**Trade-off.** Prompt lengths are approximate targets. Caching, other traffic, and transport overhead affect these comparisons; they measure neither isolated GPU kernels nor model quality.

**Evidence.** [Runner](../../spark_serve_benchmarks.py), [SSE accounting](../../tools/qwen_acceptance.py), [tests](../../tests/test_native_benchmarks.py), [operator guide](../native-observability.md).
