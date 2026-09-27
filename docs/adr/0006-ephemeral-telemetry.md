# ADR 0006: Ephemeral telemetry with explicit metric semantics

Accepted · Recorded 2026-09-26 (retrospective)

**Context.** Readiness checks cannot explain resource contention. Rates need consecutive samples, while one unavailable Spark must not block the other's resource updates.

**Decision.** Run an independent, temporary SSH process per physical host, reusing the stateful Python sampler. Scrape inference metrics once per ready allocation through explicit vLLM/NIM adapters. Reconnect with backoff; report unavailable counters as null and old observations as stale. Reset rate baselines after identity changes, missing series, decreases, or gaps. Latencies are interval histogram means. Node memory is authoritative; GPU memory is never added to it. Native chart history stays bounded within the app session.

**Trade-off.** No remote daemon or telemetry database is required. Collection depends on SSH, Python, and runtime-specific metrics; OpenAI API compatibility does not imply observability support.

**Evidence.** [Monitor](../../spark_serve_monitor.py), [shared sampler](../../spark_bench/telemetry.py), [native history](../../gui/Observability.swift), [tests](../../tests/test_monitor.py), [operator guide](../native-observability.md).
