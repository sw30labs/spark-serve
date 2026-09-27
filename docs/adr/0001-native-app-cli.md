# ADR 0001: One native app, one CLI boundary

Accepted · Recorded 2026-09-26 (retrospective)

**Context.** Monitoring and benchmarks must fit the existing Mac app without
creating another workload controller.

**Decision.** Keep Overview, Models, and Benchmarks in SwiftUI. The app invokes
the Python CLI through subprocesses and JSON; Python owns SSH, Docker, and
lifecycle operations. Telemetry uses versioned JSON-line snapshots; benchmarks
emit progress events. The app owns and stops its streams. Reuse existing
telemetry and inference protocol code. Credit sparkDash for the UI ideas.

**Trade-off.** One installation and control path, with app/CLI version coupling.
The GUI remains macOS-specific; the CLI remains usable directly. A web dashboard
or service would introduce another deployment and process boundary.

**Evidence.** [App](../../gui/SparkServeApp.swift),
[streams](../../gui/Observability.swift), [protocol smoke](../../gui/tests/README.md),
[provenance](../native-observability.md#scope-and-provenance).
