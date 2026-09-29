import Foundation
import Combine

struct BenchmarkRun: Decodable, Identifiable {
    let id: String
    let kind: String
    var status: String
    let started_at: String
    var finished_at: String?
    let model: String?
    let served_name: String?
    let node: String
    let hosts: [String]
    let endpoint: String
    let runtime: BenchmarkRuntime
    let allocation: BenchmarkAllocation
    let config: BenchmarkConfiguration
    let warmup: BenchmarkRequest?
    let results: [BenchmarkRequest]
    let summary: BenchmarkSummary?
    var error: String?
    var title: String { served_name ?? model ?? "Managed model" }
    var startedDate: Date? {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = formatter.date(from: started_at) { return date }
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.date(from: started_at)
    }
}

struct BenchmarkRuntime: Decodable {
    let engine: String?
    let image: String?
    let settings: BenchmarkRuntimeSettings?
    let settings_source: String?
    let max_model_len: Int?
    var streamsText: String { settings?.parallel.map(String.init) ?? "—" }
    var cacheText: String { settings?.kv_dtype ?? "—" }
    var contextText: String { max_model_len.map { "\($0.formatted()) tokens" } ?? "—" }
}
struct BenchmarkRuntimeSettings: Decodable {
    let parallel: Int?
    let kv_dtype: String?
}
struct BenchmarkAllocation: Decodable {
    let id: String?
    let containers: [BenchmarkContainer]?
}
struct BenchmarkContainer: Decodable {
    let host: String
    let id: String?
    let image: String?
}
struct BenchmarkConfiguration: Decodable {
    let concurrency: Int
    let requests: Int
    let max_tokens: Int
    let prompt_tokens: Int
    let thinking: Bool?
    let reasoning_mode: String?
    let deadline_seconds: Double?
    let prompt_lengths: [Int]?
}
struct BenchmarkRequest: Decodable, Identifiable {
    var id: Int { index }
    let index: Int
    let phase: String
    let requested_prompt_tokens: Int?
    let passed: Bool
    let cancelled: Bool?
    let error: String?
    let usage: BenchmarkUsage?
    let metrics: BenchmarkTimings?
}
struct BenchmarkUsage: Decodable {
    let prompt_tokens: Int?
    let completion_tokens: Int?
    let total_tokens: Int?
}
struct BenchmarkTimings: Decodable {
    let ttft_seconds: Double?
    let first_model_output_seconds: Double?
    let elapsed_seconds: Double?
    let completion_tokens_per_second_end_to_end: Double?
    let post_first_output_tokens_per_second_estimate: Double?
    let prefill_tokens_per_second_estimate: Double?
}
struct BenchmarkSummary: Decodable {
    let successful_requests: Int?
    let failed_requests: Int?
    let total_prompt_tokens: Int?
    let total_completion_tokens: Int?
    let p50: BenchmarkTimings?
    let wall_seconds: Double?
    let aggregate_completion_tokens_per_second_end_to_end: Double?
}
struct BenchmarkEvent: Decodable {
    let event: String
    let run: BenchmarkRun?
    let completed: Int?
    let total: Int?
    let request: BenchmarkRequest?
    let detail: String?
    let error: String?
}

final class BenchmarkStore: ObservableObject {
    @Published private(set) var current: BenchmarkRun?
    @Published private(set) var history: [BenchmarkRun] = []
    @Published private(set) var isRunning = false
    @Published private(set) var cancelling = false
    @Published private(set) var completed = 0
    @Published private(set) var total = 0
    @Published private(set) var latestRequest: BenchmarkRequest?
    @Published private(set) var error: String?
    @Published private(set) var loadingHistory = false
    private let home: URL?
    private let stream = CLIStream()
    private var listProcess: Process?
    private var cancelProcess: Process?
    private var activeNode: String?
    private var stopped = false

    init(home: URL?) { self.home = home }

    func start(target: AllocationTelemetry, kind: String, concurrency: Int,
               requests: Int, maxTokens: Int, promptTokens: Int, reasoning: String) {
        guard !stopped, !isRunning, let home, target.supportsBenchmark else { return }
        isRunning = true
        cancelling = false
        current = nil
        error = nil
        latestRequest = nil
        completed = 0
        total = requests
        activeNode = target.benchmarkNode
        var arguments = ["bench", "run", "--node", target.benchmarkNode, "--kind", kind,
                         "--allocation-id", target.id,
                         "--concurrency", String(concurrency), "--requests", String(requests),
                         "--max-tokens", String(maxTokens), "--prompt-tokens", String(promptTokens), "--json"]
        if reasoning == "on" { arguments.append("--thinking") }
        if reasoning == "off" { arguments.append("--no-thinking") }
        do {
            try stream.start(home: home, arguments: arguments,
                             onLine: { [weak self] in self?.receive($0) },
                             onExit: { [weak self] code in
                guard let self else { return }
                self.isRunning = false
                self.cancelling = false
                if self.current?.status == "running" {
                    self.current?.status = "failed"
                    self.current?.error = "Benchmark ended before a final result (exit \(code))."
                    self.error = self.current?.error
                } else if self.current == nil, self.error == nil {
                    self.error = "Benchmark did not return a result (exit \(code))."
                }
                self.refreshHistory()
            })
        } catch {
            isRunning = false
            self.error = "Cannot start benchmark: \(error.localizedDescription)"
        }
    }

    func receive(_ data: Data) {
        guard let event = try? JSONDecoder().decode(BenchmarkEvent.self, from: data) else {
            if let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               ["started", "progress", "result"].contains(object["event"] as? String ?? "") {
                error = "Could not read benchmark data. Update the app and CLI together."
            }
            return
        }
        switch event.event {
        case "started": current = event.run
        case "progress":
            completed = event.completed ?? completed
            total = event.total ?? total
            latestRequest = event.request
        case "result":
            current = event.run
            if let run = event.run {
                history.removeAll { $0.id == run.id }
                history.insert(run, at: 0)
                history = Array(history.prefix(100))
                error = run.error
            }
        case "error": error = event.detail ?? event.error ?? "Benchmark failed."
        default: break
        }
    }

    func cancel() {
        guard isRunning, !cancelling, let activeNode, let home else { return }
        cancelling = true
        let child = process(home: home, arguments: ["bench", "cancel", "--node", activeNode, "--json"])
        cancelProcess = child
        capture(child) { [weak self] _, code in
            guard let self else { return }
            self.cancelProcess = nil
            if code != 0 {
                self.cancelling = false
                self.error = "Could not revoke the benchmark. Try Cancel again."
            }
        }
    }

    func refreshHistory() {
        guard !stopped, !loadingHistory, let home else { return }
        loadingHistory = true
        let child = process(home: home, arguments: ["bench", "list", "--json"])
        listProcess = child
        capture(child) { [weak self] data, code in
            guard let self else { return }
            self.loadingHistory = false
            self.listProcess = nil
            guard code == 0, let runs = try? JSONDecoder().decode([BenchmarkRun].self, from: data) else {
                self.error = "Could not read saved benchmarks."
                return
            }
            self.history = Array(runs.sorted { $0.started_at > $1.started_at }.prefix(100))
            if self.error == "Could not read saved benchmarks." { self.error = nil }
        }
    }

    func stop() {
        stopped = true
        // SIGTERM is handled by bench run: it revokes admission, closes requests,
        // and persists the cancelled result. Never stop the inference workload.
        stream.stop()
        if listProcess?.isRunning == true { listProcess?.terminate() }
        if cancelProcess?.isRunning == true { cancelProcess?.terminate() }
    }

    private func process(home: URL, arguments: [String]) -> Process {
        let child = Process()
        child.executableURL = home.appendingPathComponent("spark-serve")
        child.arguments = arguments
        child.currentDirectoryURL = home
        child.environment = SparkServePaths.processEnvironment()
        return child
    }

    private func capture(_ child: Process, completion: @escaping (Data, Int32) -> Void) {
        let pipe = Pipe()
        child.standardOutput = pipe
        child.standardError = pipe
        do { try child.run() }
        catch { completion(Data(), -1); return }
        DispatchQueue.global(qos: .utility).async {
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            child.waitUntilExit()
            try? pipe.fileHandleForReading.close()
            let code = child.terminationStatus
            DispatchQueue.main.async { completion(data, code) }
        }
    }
}
