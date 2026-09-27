import SwiftUI
import Charts

struct BenchmarkView: View {
    @ObservedObject var benchmarks: BenchmarkStore
    @ObservedObject var telemetry: TelemetryStore
    @EnvironmentObject var runner: CLIRunner
    @State private var targetID = ""
    @State private var kind = "decode"
    @State private var concurrency = 1
    @State private var requests = 3
    @State private var maxTokens = 128
    @State private var promptTokens = 1024
    @State private var reasoning = "default"
    @State private var selectedRunIDs: [String] = []

    private var selectedRuns: [BenchmarkRun] {
        selectedRunIDs.compactMap { id in benchmarks.history.first { $0.id == id } }
    }
    var body: some View {
        TimelineView(.periodic(from: .now, by: 2)) { context in
            let targets = telemetry.benchmarkTargets(now: context.date)
            let selectedTarget = targets.first { $0.id == targetID }
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    configuration(targets: targets, selectedTarget: selectedTarget)
                    if let error = benchmarks.error {
                        Label(error, systemImage: "exclamationmark.triangle")
                            .font(.caption).foregroundStyle(.orange).textSelection(.enabled)
                    }
                    if benchmarks.isRunning {
                        progress
                    } else if let current = benchmarks.current {
                        BenchmarkResultCard(run: current)
                    }
                    history
                    if selectedRuns.count == 2 {
                        BenchmarkComparison(first: selectedRuns[0], second: selectedRuns[1])
                    } else if let selected = selectedRuns.first {
                        BenchmarkResultCard(run: selected)
                    }
                }
                .padding(2)
            }
            .onChange(of: targets.map(\.id)) { ids in
                if !ids.contains(targetID) { targetID = ids.first ?? "" }
            }
            .onAppear {
                if !targets.contains(where: { $0.id == targetID }) { targetID = targets.first?.id ?? "" }
            }
        }
        .onChange(of: kind) { value in
            if value == "prefill" { concurrency = 1 }
        }
        .onChange(of: concurrency) { value in requests = max(requests, value) }
    }

    private func configuration(targets: [AllocationTelemetry], selectedTarget: AllocationTelemetry?) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Measure a running model").font(.title3.weight(.semibold))
            Text("Runs a warmup, then bounded test requests against the selected allocation. Hermes selection is unchanged.")
                .font(.caption).foregroundStyle(.secondary)
            if targets.isEmpty {
                Text("A ready managed vLLM or NIM allocation is required. Start a model in Models and wait for live status.")
                    .font(.callout).foregroundStyle(.secondary)
            } else {
                Picker("Target", selection: $targetID) {
                    ForEach(targets) { target in
                        Text("\(target.title) · \(target.nodes.map { runner.hostName($0) }.joined(separator: " + "))")
                            .tag(target.id)
                    }
                }
                .disabled(benchmarks.isRunning)
                if let selectedTarget {
                    Text(selectedTarget.endpoint ?? "Endpoint unavailable")
                        .font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary).textSelection(.enabled)
                }
            }
            HStack(alignment: .top, spacing: 24) {
                VStack(alignment: .leading, spacing: 12) {
                    Picker("Test", selection: $kind) {
                        Text("Decode").tag("decode")
                        Text("Prefill").tag("prefill")
                    }.pickerStyle(.segmented)
                    Stepper("Concurrency: \(concurrency)", value: $concurrency, in: 1...4)
                        .disabled(kind == "prefill")
                    Stepper("Requests: \(requests)", value: $requests, in: concurrency...12)
                }
                VStack(alignment: .leading, spacing: 12) {
                    Picker("Reasoning", selection: $reasoning) {
                        Text("Server default").tag("default")
                        Text("Request on").tag("on")
                        Text("Request off").tag("off")
                    }
                    Stepper("Maximum output: \(maxTokens) tokens", value: $maxTokens, in: 16...512, step: 16)
                    Picker("Approximate prompt", selection: $promptTokens) {
                        ForEach([128, 512, 1024, 4096, 8192, 16384, 32768], id: \.self) { value in
                            Text("\(value.formatted()) tokens").tag(value)
                        }
                    }
                }
            }
            .disabled(benchmarks.isRunning)
            HStack {
                Button("Run \(kind) benchmark") {
                    guard let target = telemetry.benchmarkTargets().first(where: { $0.id == targetID }) else { return }
                    benchmarks.start(target: target, kind: kind, concurrency: concurrency, requests: requests,
                                     maxTokens: maxTokens, promptTokens: promptTokens, reasoning: reasoning)
                }
                .buttonStyle(.borderedProminent)
                .disabled(selectedTarget == nil || benchmarks.isRunning || runner.isBusy)
                Text("Up to 5 minutes · \(requests) measured requests + warmup")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Text("Timing is measured at this Mac and includes transport. Prefill uses synthetic prompts; actual token counts come from the server. Reasoning settings are requests to the runtime, not proof of its behavior.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(14)
        .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 10))
    }

    private var progress: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(benchmarks.current?.title ?? "Checking target…").font(.headline)
                    if let current = benchmarks.current {
                        Text("\(current.hosts.joined(separator: " + ")) · \(current.endpoint)")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Text(benchmarks.completed == 0 ? "Warmup or first request in progress" : "\(benchmarks.completed) of \(benchmarks.total) requests finished")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Button(benchmarks.cancelling ? "Cancelling…" : "Cancel benchmark") { benchmarks.cancel() }
                    .disabled(benchmarks.cancelling)
            }
            ProgressView(value: Double(benchmarks.completed), total: Double(max(1, benchmarks.total)))
            if let request = benchmarks.latestRequest {
                HStack {
                    Text(request.passed ? "Latest request completed" : "Latest request failed").font(.caption)
                    Spacer()
                    Text("First visible token \(MetricFormat.latency(request.metrics?.ttft_seconds))")
                        .font(.caption).monospacedDigit()
                }
                if let error = request.error {
                    Text(error).font(.caption).foregroundStyle(.orange).lineLimit(3)
                }
            }
        }
        .padding(14)
        .background(Color.accentColor.opacity(0.06), in: RoundedRectangle(cornerRadius: 10))
    }

    private var history: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Saved runs").font(.title3.weight(.semibold))
                Text("Select up to two to compare").font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button("Refresh") { benchmarks.refreshHistory() }.disabled(benchmarks.loadingHistory)
            }
            if benchmarks.history.isEmpty {
                Text(benchmarks.loadingHistory ? "Loading saved runs…" : "Completed and interrupted runs will appear here.")
                    .font(.callout).foregroundStyle(.secondary).padding(.vertical, 12)
            } else {
                ForEach(benchmarks.history.prefix(20)) { run in
                    HStack(spacing: 10) {
                        Toggle("Compare run", isOn: Binding(
                            get: { selectedRunIDs.contains(run.id) },
                            set: { on in
                                if on {
                                    if selectedRunIDs.count == 2 { selectedRunIDs.removeFirst() }
                                    selectedRunIDs.append(run.id)
                                } else { selectedRunIDs.removeAll { $0 == run.id } }
                            }
                        ))
                        .labelsHidden()
                        .accessibilityLabel("Select \(run.title), \(run.kind), \(run.startedDate?.formatted() ?? "date unavailable")")
                        VStack(alignment: .leading, spacing: 3) {
                            Text(run.title).font(.subheadline.weight(.medium)).lineLimit(1)
                            Text("\(run.kind.capitalized) · \(run.hosts.joined(separator: " + ")) · C\(run.config.concurrency) · \(run.status)")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
                        Text(run.startedDate?.formatted(date: .abbreviated, time: .shortened) ?? "Date unavailable")
                            .font(.caption).foregroundStyle(.secondary)
                        Text(MetricFormat.number(run.summary?.aggregate_completion_tokens_per_second_end_to_end, suffix: " tok/s"))
                            .font(.caption.monospacedDigit()).frame(minWidth: 80, alignment: .trailing)
                    }
                    .padding(10)
                    .background(selectedRunIDs.contains(run.id) ? Color.accentColor.opacity(0.09) : Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 7))
                }
                Text("Rates in this list include time to first output. Showing the latest \(min(20, benchmarks.history.count)) saved runs.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}

private struct BenchmarkResultCard: View {
    let run: BenchmarkRun
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("\(run.kind.capitalized) · \(run.title)").font(.headline)
                Spacer()
                Text(run.status.capitalized).font(.caption).foregroundStyle(run.status == "completed" ? Color.green : .orange)
            }
            Text("Measured requests: \(run.summary?.successful_requests ?? 0) successful · \(run.summary?.failed_requests ?? 0) failed · \(MetricFormat.latency(run.summary?.wall_seconds)) wall time")
                .font(.caption).foregroundStyle(.secondary)
            if let warmup = run.warmup, !warmup.passed {
                Label("Warmup \(warmup.cancelled == true ? "cancelled" : "failed"): \(warmup.error ?? "No successful warmup result")", systemImage: "exclamationmark.triangle")
                    .font(.caption).foregroundStyle(.orange).textSelection(.enabled)
            }
            HStack {
                MetricTile(title: "Median first visible token", value: MetricFormat.latency(run.summary?.p50?.ttft_seconds))
                MetricTile(title: "Median first model output", value: MetricFormat.latency(run.summary?.p50?.first_model_output_seconds))
                MetricTile(title: "Aggregate completion rate", value: MetricFormat.number(run.summary?.aggregate_completion_tokens_per_second_end_to_end, suffix: " tok/s"))
            }
            HStack {
                MetricTile(title: "Median decode rate (estimate)", value: MetricFormat.number(run.summary?.p50?.post_first_output_tokens_per_second_estimate, suffix: " tok/s"))
                MetricTile(title: "Median prefill rate (estimate)", value: MetricFormat.number(run.summary?.p50?.prefill_tokens_per_second_estimate, suffix: " tok/s"))
                MetricTile(title: "Prompt / completion tokens", value: "\(run.summary?.total_prompt_tokens.map(String.init) ?? "—") / \(run.summary?.total_completion_tokens.map(String.init) ?? "—")")
            }
            Text("First model output includes reasoning. Aggregate rate includes time to first output; estimated decode excludes that interval. Server usage is required for token rates. Warmup is excluded.")
                .font(.caption).foregroundStyle(.secondary)
            if run.kind == "prefill" { PrefillSweepChart(run: run) }
            if let error = run.error { Text(error).font(.caption).foregroundStyle(.orange).textSelection(.enabled) }
            DisclosureGroup("Configuration and request details") {
                VStack(alignment: .leading, spacing: 8) {
                    BenchmarkMetadata(run: run)
                    ForEach(run.results) { result in
                        HStack {
                            Text("Request \(result.index + 1)")
                            Text(result.passed ? "OK" : "Failed").foregroundStyle(result.passed ? Color.green : .orange)
                            Spacer()
                            Text("\(result.usage?.prompt_tokens.map(String.init) ?? "—") prompt tokens · \(result.usage?.completion_tokens.map(String.init) ?? "—") output tokens")
                        }.font(.caption).monospacedDigit()
                        Text("First visible token \(MetricFormat.latency(result.metrics?.ttft_seconds)) · First model output \(MetricFormat.latency(result.metrics?.first_model_output_seconds)) · Total \(MetricFormat.latency(result.metrics?.elapsed_seconds))")
                            .font(.caption).foregroundStyle(.secondary).monospacedDigit()
                        if let error = result.error { Text(error).font(.caption).foregroundStyle(.orange).textSelection(.enabled) }
                    }
                }.padding(.top, 8)
            }
        }
        .padding(14)
        .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 10))
    }
}

private struct PrefillSweepChart: View {
    let run: BenchmarkRun
    private var measured: [BenchmarkRequest] {
        run.results.filter { $0.passed && $0.usage?.prompt_tokens != nil && $0.metrics?.first_model_output_seconds != nil }
            .sorted { ($0.usage?.prompt_tokens ?? 0) < ($1.usage?.prompt_tokens ?? 0) }
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Prefill context sweep").font(.subheadline.weight(.medium))
            if measured.isEmpty {
                Text("No successful measured requests with server token counts.").font(.caption).foregroundStyle(.secondary)
            } else {
                Chart(measured) { request in
                    if let tokens = request.usage?.prompt_tokens,
                       let seconds = request.metrics?.first_model_output_seconds {
                        LineMark(x: .value("Actual prompt tokens", tokens), y: .value("First output (s)", seconds))
                            .foregroundStyle(Color.accentColor.opacity(0.7))
                        PointMark(x: .value("Actual prompt tokens", tokens), y: .value("First output (s)", seconds))
                            .foregroundStyle(Color.accentColor)
                    }
                }
                .chartXAxisLabel("Actual server prompt tokens")
                .chartYAxisLabel("First model output (s)")
                .frame(height: 180)
                .accessibilityLabel("Prefill context sweep, actual prompt tokens versus time to first model output in seconds")
            }
            Text("Client-observed first output includes reasoning and transport. Only successful measured requests are plotted. Summary medians above combine context sizes.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(.vertical, 8)
    }
}

private struct BenchmarkMetadata: View {
    let run: BenchmarkRun
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("\(run.hosts.joined(separator: " + ")) · \(run.endpoint)")
            Text("\(run.runtime.engine ?? "Unknown runtime") · \(run.runtime.image ?? "Image not recorded")")
            Text("Concurrency \(run.config.concurrency) · \(run.config.requests) requests · max output \(run.config.max_tokens) · approximate prompt \(run.config.prompt_tokens)")
            if run.kind == "prefill", let lengths = run.config.prompt_lengths {
                Text("Requested context sizes: \(lengths.map(String.init).joined(separator: ", "))")
            }
            Text("Reasoning: \(run.config.reasoning_mode ?? "server-default")")
            Text("Allocation: \(run.allocation.id ?? "Not recorded")")
            ForEach(Array((run.allocation.containers ?? []).enumerated()), id: \.offset) { _, container in
                Text("\(container.host): \(container.id ?? "Container ID not recorded")")
            }
            Text("Run: \(run.id)")
        }
        .font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary).textSelection(.enabled)
    }
}

private struct BenchmarkComparison: View {
    let first: BenchmarkRun
    let second: BenchmarkRun
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Compare saved runs").font(.headline)
            Text("Compare matching configurations and runtime settings. These are observations from separate runs, not a controlled speedup claim.")
                .font(.caption).foregroundStyle(.secondary)
            Grid(alignment: .leading, horizontalSpacing: 24, verticalSpacing: 9) {
                GridRow {
                    Text("Metric").foregroundStyle(.secondary)
                    Text(first.title).fontWeight(.medium)
                    Text(second.title).fontWeight(.medium)
                }
                Divider()
                row("Test / state", "\(first.kind) / \(first.status)", "\(second.kind) / \(second.status)")
                row("Runtime", first.runtime.engine ?? "—", second.runtime.engine ?? "—")
                row("Hosts", first.hosts.joined(separator: " + "), second.hosts.joined(separator: " + "))
                row("Concurrency / requests", "\(first.config.concurrency) / \(first.config.requests)", "\(second.config.concurrency) / \(second.config.requests)")
                row("Prompt target / output cap", "\(first.config.prompt_tokens) / \(first.config.max_tokens)", "\(second.config.prompt_tokens) / \(second.config.max_tokens)")
                row("Reasoning request", first.config.reasoning_mode ?? "server-default", second.config.reasoning_mode ?? "server-default")
                row("Median first visible token", MetricFormat.latency(first.summary?.p50?.ttft_seconds), MetricFormat.latency(second.summary?.p50?.ttft_seconds))
                row("Median first model output", MetricFormat.latency(first.summary?.p50?.first_model_output_seconds), MetricFormat.latency(second.summary?.p50?.first_model_output_seconds))
                row("Median elapsed", MetricFormat.latency(first.summary?.p50?.elapsed_seconds), MetricFormat.latency(second.summary?.p50?.elapsed_seconds))
                row("Aggregate completion tok/s", MetricFormat.number(first.summary?.aggregate_completion_tokens_per_second_end_to_end), MetricFormat.number(second.summary?.aggregate_completion_tokens_per_second_end_to_end))
                row("Median decode tok/s (estimate)", MetricFormat.number(first.summary?.p50?.post_first_output_tokens_per_second_estimate), MetricFormat.number(second.summary?.p50?.post_first_output_tokens_per_second_estimate))
                row("Median prefill tok/s (estimate)", MetricFormat.number(first.summary?.p50?.prefill_tokens_per_second_estimate), MetricFormat.number(second.summary?.p50?.prefill_tokens_per_second_estimate))
            }
            .font(.caption).monospacedDigit().textSelection(.enabled)
            DisclosureGroup("Recorded runtime identities") {
                HStack(alignment: .top, spacing: 20) {
                    BenchmarkMetadata(run: first).frame(maxWidth: .infinity, alignment: .leading)
                    BenchmarkMetadata(run: second).frame(maxWidth: .infinity, alignment: .leading)
                }.padding(.top, 8)
            }
        }
        .padding(14)
        .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 10))
    }
    private func row(_ label: String, _ a: String, _ b: String) -> some View {
        GridRow { Text(label).foregroundStyle(.secondary); Text(a); Text(b) }
    }
}
