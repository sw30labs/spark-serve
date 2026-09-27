import SwiftUI
import Charts

enum MetricFormat {
    static func number(_ value: Double?, suffix: String = "", digits: Int = 1) -> String {
        guard let value, value.isFinite else { return "—" }
        return String(format: "%.*f", digits, value) + suffix
    }
    static func bytes(_ value: Double?) -> String {
        guard let value, value.isFinite, value >= 0 else { return "—" }
        let units = ["B", "KiB", "GiB"]
        if value >= 1_073_741_824 { return number(value / 1_073_741_824, suffix: " " + units[2]) }
        if value >= 1_048_576 { return number(value / 1_048_576, suffix: " MiB") }
        if value >= 1024 { return number(value / 1024, suffix: " " + units[1]) }
        return number(value, suffix: " B", digits: 0)
    }
    static func rate(_ value: Double?) -> String {
        value == nil ? "—" : bytes(value) + "/s"
    }
    static func latency(_ seconds: Double?) -> String {
        guard let seconds else { return "—" }
        return seconds < 1 ? number(seconds * 1000, suffix: " ms", digits: 0) : number(seconds, suffix: " s", digits: 2)
    }
    static func age(_ timestamp: Double?, now: Date) -> String {
        guard let timestamp else { return "No sample yet" }
        let seconds = max(0, Int(now.timeIntervalSince1970 - timestamp))
        return seconds < 60 ? "Sampled \(seconds)s ago" : "Sampled \(seconds / 60)m ago"
    }
}

struct ReadingState: View {
    let state: String
    private var color: Color {
        switch state {
        case "live", "ready": return .green
        case "stale", "starting", "draining": return .orange
        default: return .secondary
        }
    }
    var body: some View {
        HStack(spacing: 5) {
            Circle().fill(color).frame(width: 6, height: 6)
            Text(state.capitalized).font(.caption)
        }
        .foregroundStyle(color)
        .padding(.horizontal, 8).padding(.vertical, 4)
        .background(color.opacity(0.1), in: Capsule())
        .accessibilityElement(children: .combine)
    }
}

struct MetricTrend: View {
    let points: [MetricPoint]
    let color: Color
    let label: String
    let now: Date
    private var segments: [[MetricPoint]] {
        var result: [[MetricPoint]] = []
        for point in points {
            if let last = result.last?.last, point.timestamp - last.timestamp <= 10 {
                result[result.count - 1].append(point)
            } else {
                result.append([point])
            }
        }
        return result
    }
    var body: some View {
        Chart {
            ForEach(Array(segments.enumerated()), id: \.offset) { segment in
                ForEach(segment.element) { point in
                    LineMark(x: .value("Time", point.date), y: .value(label, point.value),
                             series: .value("Reading interval", segment.offset))
                        .foregroundStyle(color)
                        .lineStyle(StrokeStyle(lineWidth: 1.5))
                }
            }
        }
        .chartXAxis(.hidden).chartYAxis(.hidden)
        .chartXScale(domain: now.addingTimeInterval(-TelemetryStore.retentionSeconds)...now)
        .frame(height: 34)
        .accessibilityLabel("\(label), up to ten minutes of history")
        .accessibilityValue(points.last.map { MetricFormat.number($0.value) } ?? "No readings")
    }
}

struct MetricTile: View {
    let title: String
    let value: String
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.system(.title3, design: .rounded).weight(.medium)).monospacedDigit()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
    }
}

struct OverviewView: View {
    @ObservedObject var telemetry: TelemetryStore
    @EnvironmentObject var runner: CLIRunner

    var body: some View {
        TimelineView(.periodic(from: .now, by: 2)) { context in
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    HStack(alignment: .firstTextBaseline) {
                        Text("Your Sparks").font(.title3.weight(.semibold))
                        Spacer()
                        Text("Ten minutes of history while the app is open")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    if let error = telemetry.error {
                        Label(error, systemImage: "antenna.radiowaves.left.and.right.slash")
                            .font(.caption).foregroundStyle(.orange)
                    }
                    HStack(alignment: .top, spacing: 12) {
                        ForEach(["head", "worker"], id: \.self) { node in
                            HostResourceCard(
                                node: node, host: runner.hostName(node),
                                sample: telemetry.snapshot?.hosts.first { $0.node == node },
                                telemetry: telemetry, now: context.date
                            )
                        }
                    }
                    HStack {
                        Text("Serving workloads").font(.title3.weight(.semibold))
                        Spacer()
                        Text("One card per allocation").font(.caption).foregroundStyle(.secondary)
                    }
                    if let snapshot = telemetry.snapshot {
                        if !telemetry.topologyIsFresh(now: context.date) {
                            Label(snapshot.status_error ?? "Workload status is stale. Waiting for a fresh snapshot…", systemImage: "exclamationmark.circle")
                                .font(.caption).foregroundStyle(.orange)
                        }
                        if snapshot.allocations.isEmpty && telemetry.topologyIsFresh(now: context.date) {
                            Text("No managed workload is running. Open Models to start one.")
                                .foregroundStyle(.secondary).padding(.vertical, 14)
                        } else if !snapshot.allocations.isEmpty {
                            LazyVGrid(columns: [GridItem(.adaptive(minimum: 330), alignment: .top)], spacing: 12) {
                                ForEach(snapshot.allocations) { allocation in
                                    AllocationCard(allocation: allocation, telemetry: telemetry, now: context.date)
                                }
                            }
                        }
                    } else {
                        Text("Waiting for workload status…").foregroundStyle(.secondary).padding(.vertical, 14)
                    }
                    Text("— means the metric is not reported. Unified RAM is the host memory pool; GPU memory is a separate view of that shared pool.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                .padding(2)
            }
        }
    }
}

private struct HostResourceCard: View {
    let node: String
    let host: String
    let sample: HostTelemetry?
    @ObservedObject var telemetry: TelemetryStore
    let now: Date
    private var resources: HostResources? { sample?.resources }
    private var state: String {
        telemetry.currentState(sample?.state ?? "unavailable", sampledAt: sample?.sampled_at, now: now)
    }
    private var memoryFraction: Double? {
        guard let used = resources?.memory_used_bytes,
              let total = resources?.memory_total_bytes, total > 0 else { return nil }
        return min(1, max(0, used / total))
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label(sample?.host ?? host, systemImage: "desktopcomputer").font(.headline)
                Spacer()
                ReadingState(state: state)
            }
            Text("\(node.capitalized) · \(MetricFormat.age(sample?.sampled_at, now: now))")
                .font(.caption).foregroundStyle(.secondary)
            HStack(spacing: 18) {
                resourcePercent("GPU", value: resources?.gpu_utilization_percent, metric: "gpu", color: .green)
                resourcePercent("CPU", value: resources?.cpu_percent, metric: "cpu", color: .blue)
            }
            Divider()
            VStack(alignment: .leading, spacing: 5) {
                HStack {
                    Text("Unified RAM").font(.subheadline.weight(.medium))
                    Spacer()
                    Text("\(MetricFormat.bytes(resources?.memory_used_bytes)) / \(MetricFormat.bytes(resources?.memory_total_bytes))")
                        .font(.caption).monospacedDigit()
                }
                if let memoryFraction {
                    ProgressView(value: memoryFraction).tint(memoryFraction > 0.9 ? .orange : .accentColor)
                } else {
                    Capsule().fill(Color.secondary.opacity(0.1)).frame(height: 4)
                }
                HStack {
                    Text("Swap \(MetricFormat.bytes(resources?.swap_used_bytes))")
                    Spacer()
                    Text("GPU-reported \(MetricFormat.bytes(resources?.gpu_memory_used_bytes))")
                }
                .font(.caption2).foregroundStyle(.secondary)
            }
            HStack {
                MetricTile(title: "GPU temperature", value: MetricFormat.number(resources?.gpu_temperature_c, suffix: " °C", digits: 0))
                MetricTile(title: "GPU power", value: MetricFormat.number(resources?.gpu_power_watts, suffix: " W"))
            }
            Divider()
            ioRow("Network", first: "↓ " + MetricFormat.rate(resources?.network_rx_bytes_per_second),
                  second: "↑ " + MetricFormat.rate(resources?.network_tx_bytes_per_second))
            ioRow("Disk", first: "Read " + MetricFormat.rate(resources?.disk_read_bytes_per_second),
                  second: "Write " + MetricFormat.rate(resources?.disk_write_bytes_per_second))
            if let error = sample?.error, !error.isEmpty {
                Text(error).font(.caption).foregroundStyle(.orange).lineLimit(3).help(error)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .topLeading)
        .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.secondary.opacity(0.15)))
    }
    private func resourcePercent(_ title: String, value: Double?, metric: String, color: Color) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            MetricTile(title: "\(title) utilization", value: MetricFormat.number(value, suffix: "%", digits: 0))
            MetricTrend(points: telemetry.history["host:\(node):\(metric)"] ?? [], color: color, label: "\(title) utilization", now: now)
        }
    }
    private func ioRow(_ title: String, first: String, second: String) -> some View {
        HStack {
            Text(title).foregroundStyle(.secondary)
                .help(title == "Network" ? "Summed interface counters. Virtual interfaces may count the same traffic more than once." : "Summed block-device counters. Stacked devices may count the same I/O more than once.")
            Spacer()
            Text(first).monospacedDigit()
            Text(second).monospacedDigit()
        }.font(.caption)
    }
}

private struct AllocationCard: View {
    let allocation: AllocationTelemetry
    @ObservedObject var telemetry: TelemetryStore
    let now: Date
    @EnvironmentObject var runner: CLIRunner
    private var state: String {
        telemetry.currentState(allocation.metrics_state, sampledAt: allocation.sampled_at, now: now)
    }
    private var title: String {
        runner.models.first { $0.id == allocation.model }?.label ?? allocation.title
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(title).font(.headline).textSelection(.enabled)
                    Text("\(allocation.nodes.map { runner.hostName($0) }.joined(separator: " + ")) · \(allocation.runtime ?? "Unknown runtime")")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                ReadingState(state: telemetry.topologyIsFresh(now: now)
                             ? (allocation.ready ? "ready" : (allocation.phase ?? "unavailable")) : "stale")
            }
            if allocation.nodes.count > 1 {
                Label("One distributed allocation across both Sparks", systemImage: "point.3.connected.trianglepath.dotted")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if let endpoint = allocation.endpoint {
                Text(endpoint).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary).textSelection(.enabled)
            }
            HStack {
                ReadingState(state: state)
                Text(MetricFormat.age(allocation.sampled_at, now: now)).font(.caption).foregroundStyle(.secondary)
            }
            if state == "unsupported" {
                Text("Inference metrics are not exposed by this runtime.").font(.caption).foregroundStyle(.secondary)
            }
            HStack(spacing: 18) {
                tokenRate("Decode", value: allocation.metrics.generation_tokens_per_second, key: "decode", color: .green)
                tokenRate("Prefill", value: allocation.metrics.prompt_tokens_per_second, key: "prefill", color: .blue)
            }
            Divider()
            HStack {
                MetricTile(title: "Running / waiting", value: "\(MetricFormat.number(allocation.metrics.requests_running, digits: 0)) / \(MetricFormat.number(allocation.metrics.requests_waiting, digits: 0))")
                MetricTile(title: "KV cache", value: MetricFormat.number(allocation.metrics.kv_cache_percent, suffix: "%"))
                MetricTile(title: "Completed requests/s", value: MetricFormat.number(allocation.metrics.requests_per_second, digits: 2))
            }
            HStack {
                MetricTile(title: "Mean TTFT", value: MetricFormat.latency(allocation.metrics.ttft_seconds))
                MetricTile(title: "Mean time per output token", value: MetricFormat.latency(allocation.metrics.tpot_seconds))
            }
            .help("Server-reported histogram means over the latest collection interval; these are not percentiles.")
            if let error = allocation.error, !error.isEmpty {
                Text(error).font(.caption).foregroundStyle(.orange).lineLimit(3).help(error)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .topLeading)
        .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.secondary.opacity(0.15)))
    }
    private func tokenRate(_ label: String, value: Double?, key: String, color: Color) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            MetricTile(title: label, value: MetricFormat.number(value, suffix: " tok/s"))
            MetricTrend(points: telemetry.history["allocation:\(allocation.id):\(key)"] ?? [], color: color, label: "\(label) tokens per second", now: now)
        }
    }
}
