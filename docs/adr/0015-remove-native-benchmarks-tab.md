# ADR 0015: Remove the native Benchmarks tab; benchmarks stay in the CLI

Accepted · Recorded 2026-10-25 · Supersedes the Benchmarks part of [ADR 0001](0001-native-app-cli.md)

**Context.** [ADR 0001](0001-native-app-cli.md) kept Overview, Models, and
Benchmarks in SwiftUI. The operator has not used the Benchmarks tab and does
not intend to; measurements are taken with the CLI tools and recorded in the
catalog. The tab still loaded saved history at every app launch.

**Decision.** Remove the Benchmarks tab, its view, and its app-side run store
from the native app. The app keeps Overview, NVIDIA DGX Spark, and AMD Strix
Halo, and still invokes the Python CLI through subprocesses and JSON. The CLI
`bench` commands, the benchmark runner and leases
([ADR 0007](0007-inference-benchmark-method.md),
[ADR 0008](0008-benchmark-leases.md)), the measurement tools, and saved results
are unchanged.

**Trade-off.** Running, browsing, and comparing benchmark runs now needs the
terminal; the app no longer renders prefill sweep charts. The GUI protocol
smoke no longer checks benchmark history decoding; the Python suite still
covers the benchmark runner.

**Evidence.** [App](../../gui/SparkServeApp.swift),
[protocol smoke](../../gui/tests/README.md),
[operator guide](../native-observability.md), [runner](../../spark_serve_benchmarks.py).
