import Foundation
import Combine
import Darwin

// One bounded NDJSON reader per owned CLI operation. UI callbacks always run on
// the main queue; exiting drains stdout before reporting completion.
final class CLIStream {
    private var process: Process?
    private var generation = UUID()
    var isRunning: Bool { process?.isRunning == true }

    func start(home: URL, arguments: [String], onLine: @escaping (Data) -> Void,
               onExit: @escaping (Int32) -> Void) throws {
        stop()
        let token = UUID()
        generation = token
        let child = Process()
        child.executableURL = home.appendingPathComponent("spark-serve")
        child.arguments = arguments
        child.currentDirectoryURL = home
        child.environment = SparkServePaths.processEnvironment()
        let pipe = Pipe()
        child.standardOutput = pipe
        child.standardError = pipe
        try child.run()
        process = child
        DispatchQueue.global(qos: .utility).async { [weak self] in
            var buffer = Data()
            var droppingLine = false
            while true {
                let data = pipe.fileHandleForReading.availableData
                if data.isEmpty { break }
                buffer.append(data)
                while let newline = buffer.firstIndex(of: 10) {
                    let line = Data(buffer[..<newline])
                    buffer.removeSubrange(...newline)
                    if !droppingLine, !line.isEmpty, line.count <= 1_048_576 {
                        DispatchQueue.main.async { [weak self] in
                            guard self?.generation == token else { return }
                            onLine(line)
                        }
                    }
                    droppingLine = false
                }
                if buffer.count > 1_048_576 {
                    buffer.removeAll(keepingCapacity: false)
                    droppingLine = true
                }
            }
            if !buffer.isEmpty, !droppingLine {
                let line = buffer
                DispatchQueue.main.async { [weak self] in
                    guard self?.generation == token else { return }
                    onLine(line)
                }
            }
            child.waitUntilExit()
            try? pipe.fileHandleForReading.close()
            let code = child.terminationStatus
            DispatchQueue.main.async { [weak self] in
                guard let self, self.generation == token else { return }
                self.process = nil
                onExit(code)
            }
        }
    }

    func stop() {
        generation = UUID()
        guard let child = process else { return }
        process = nil
        guard child.isRunning else { return }
        child.terminate()
        // The CLI handles SIGTERM and closes collectors/requests. Bound cleanup
        // if a child cannot finish so quitting the app cannot leave a watcher.
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 3) {
            if child.isRunning { kill(child.processIdentifier, SIGKILL) }
        }
    }

    deinit { stop() }
}

struct TelemetrySnapshot: Decodable {
    let schema_version: Int
    let timestamp: Double
    let hosts: [HostTelemetry]
    let allocations: [AllocationTelemetry]
    let status_state: String?
    let status_error: String?
}

struct HostTelemetry: Decodable, Identifiable {
    var id: String { node }
    let node: String
    let host: String
    let sampled_at: Double?
    let state: String
    let error: String?
    let resources: HostResources
}

struct HostResources: Decodable {
    let cpu_percent: Double?
    let memory_used_bytes: Double?
    let memory_total_bytes: Double?
    let swap_used_bytes: Double?
    let gpu_utilization_percent: Double?
    let gpu_memory_used_bytes: Double?
    let gpu_memory_total_bytes: Double?
    let gpu_temperature_c: Double?
    let gpu_power_watts: Double?
    let network_rx_bytes_per_second: Double?
    let network_tx_bytes_per_second: Double?
    let disk_read_bytes_per_second: Double?
    let disk_write_bytes_per_second: Double?
}

struct AllocationTelemetry: Decodable, Identifiable {
    let id: String
    let model: String?
    let served_name: String?
    let runtime: String?
    let nodes: [String]
    let endpoint: String?
    let ready: Bool
    let phase: String?
    let metrics_state: String
    let sampled_at: Double?
    let error: String?
    let metrics: InferenceMetrics

    var title: String { served_name ?? model ?? "Managed workload" }
    var benchmarkNode: String { nodes.contains("head") ? "head" : (nodes.first ?? "head") }
    var supportsBenchmark: Bool { ready && (runtime == "vllm" || runtime == "nim") }
}

struct InferenceMetrics: Decodable {
    let requests_running: Double?
    let requests_waiting: Double?
    let kv_cache_percent: Double?
    let prompt_tokens_per_second: Double?
    let generation_tokens_per_second: Double?
    let requests_per_second: Double?
    let ttft_seconds: Double?
    let tpot_seconds: Double?
}

struct MetricPoint: Identifiable {
    var id: Double { timestamp }
    let timestamp: Double
    let value: Double
    var date: Date { Date(timeIntervalSince1970: timestamp) }
}

final class TelemetryStore: ObservableObject {
    @Published private(set) var snapshot: TelemetrySnapshot?
    @Published private(set) var streamRunning = false
    @Published private(set) var error: String?
    private(set) var history: [String: [MetricPoint]] = [:]
    private let home: URL?
    private let stream = CLIStream()
    private var retry: DispatchWorkItem?
    private var stopped = false
    static let retentionSeconds: Double = 600
    static let maximumPoints = 300

    init(home: URL?) { self.home = home }

    func start() {
        guard !stopped, !stream.isRunning else { return }
        retry?.cancel()
        guard let home else {
            error = "Set SPARK_SERVE_HOME to the repo root to connect."
            return
        }
        do {
            try stream.start(home: home, arguments: ["watch", "--json", "--interval", "2"],
                             onLine: { [weak self] in self?.receive($0) },
                             onExit: { [weak self] code in
                guard let self, !self.stopped else { return }
                self.streamRunning = false
                self.error = "Live metrics disconnected (exit \(code)). Reconnecting…"
                let retry = DispatchWorkItem { [weak self] in self?.start() }
                self.retry = retry
                DispatchQueue.main.asyncAfter(deadline: .now() + 5, execute: retry)
            })
            streamRunning = true
            error = nil
        } catch {
            self.error = "Cannot start live metrics: \(error.localizedDescription)"
        }
    }

    func stop() {
        stopped = true
        retry?.cancel()
        stream.stop()
        streamRunning = false
    }

    func receive(_ data: Data) {
        guard let value = try? JSONDecoder().decode(TelemetrySnapshot.self, from: data),
              value.schema_version == 1, value.timestamp.isFinite else { return }
        var activeKeys = Set<String>()
        for host in value.hosts {
            let fields: [(String, Double?)] = [
                ("cpu", host.resources.cpu_percent),
                ("gpu", host.resources.gpu_utilization_percent),
                ("memory", host.resources.memory_used_bytes),
            ]
            for (metric, reading) in fields {
                let key = "host:\(host.node):\(metric)"
                activeKeys.insert(key)
                if host.state == "live", let at = host.sampled_at {
                    append(reading, at: at, key: key)
                }
            }
        }
        for allocation in value.allocations {
            for (metric, reading) in [("decode", allocation.metrics.generation_tokens_per_second),
                                       ("prefill", allocation.metrics.prompt_tokens_per_second)] {
                let key = "allocation:\(allocation.id):\(metric)"
                activeKeys.insert(key)
                if allocation.metrics_state == "live", let at = allocation.sampled_at {
                    append(reading, at: at, key: key)
                }
            }
        }
        history = history.filter { activeKeys.contains($0.key) }
        snapshot = value
        error = nil
    }

    private func append(_ number: Double?, at: Double, key: String) {
        guard let number, number.isFinite, at.isFinite else { return }
        var points = history[key] ?? []
        guard points.last.map({ at > $0.timestamp }) ?? true else { return }
        points.append(MetricPoint(timestamp: at, value: number))
        points.removeAll { $0.timestamp < at - Self.retentionSeconds }
        if points.count > Self.maximumPoints { points.removeFirst(points.count - Self.maximumPoints) }
        history[key] = points
    }

    func currentState(_ state: String, sampledAt: Double?, now: Date) -> String {
        guard state == "live" else { return state }
        guard streamRunning, let sampledAt,
              now.timeIntervalSince1970 - sampledAt <= 10 else { return "stale" }
        return "live"
    }

    func benchmarkTargets(now: Date = Date()) -> [AllocationTelemetry] {
        guard topologyIsFresh(now: now), let snapshot else { return [] }
        return snapshot.allocations.filter { $0.supportsBenchmark }
    }

    func topologyIsFresh(now: Date) -> Bool {
        guard streamRunning, let snapshot, snapshot.status_state == "live" else { return false }
        return now.timeIntervalSince1970 - snapshot.timestamp <= 10
    }
}
