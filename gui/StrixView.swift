import SwiftUI

// MARK: - Strix Halo site
//
// A separate store and view for the AMD box. It drives `spark-serve strix …`,
// which talks to halo-ctl on the box; the Spark runner and controller are not
// involved. The box enforces one engine at a time, so the app only asks.

struct StrixResources: Codable {
    let ram_free_gib: Double?
    let ram_total_gib: Double?
    let gtt_used_gib: Double?
    let gtt_total_gib: Double?
    let busy_pct: Double?
    let temp_c: Double?
    let power_w: Double?
    let last_gen_tps: Double?
}

struct StrixStatus: Codable {
    let host: String
    let engine: String
    let phase: String
    let model: String
    let model_id: String?
    let url: String?
    let engines: [String: String]
    let inference: String?
    let resources: StrixResources?
}

final class StrixStore: ObservableObject {
    @Published var models: [ModelEntry] = []
    @Published var status: StrixStatus?
    @Published var busyAction: String?
    @Published var lastError: String?
    @Published var logLines: [String] = []

    private let home: URL?
    private let cliPath: String
    private var timer: Timer?
    private var refreshing = false

    init() {
        home = SparkServePaths.home()
        cliPath = home?.appendingPathComponent("spark-serve").path ?? ""
    }

    var isBusy: Bool { busyAction != nil }

    func startPolling() {
        guard timer == nil else { return }
        refresh()
        timer = Timer.scheduledTimer(withTimeInterval: 5, repeats: true) { [weak self] _ in self?.refresh() }
    }

    func stopPolling() {
        timer?.invalidate()
        timer = nil
    }

    func refresh() {
        guard !refreshing, !cliPath.isEmpty else { return }
        refreshing = true
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            let models = self.decode(["strix", "list", "--json"], as: [ModelEntry].self)
            let status = self.decode(["strix", "status", "--json"], as: StrixStatus.self)
            DispatchQueue.main.async {
                self.refreshing = false
                if let models { self.models = models }
                self.status = status
                if status == nil && self.lastError == nil {
                    self.lastError = "The Strix box did not answer; check SSH to it."
                } else if status != nil, !self.isBusy {
                    self.lastError = nil
                }
            }
        }
    }

    func start(_ model: ModelEntry) { perform("Starting \(model.label)", ["strix", "up", model.id]) }

    func stop() { perform("Stopping the engine", ["strix", "stop"]) }

    private func perform(_ title: String, _ args: [String]) {
        guard !isBusy else { return }
        busyAction = title
        lastError = nil
        log("$ spark-serve \(args.joined(separator: " "))")
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            let (out, code) = self.run(args)
            DispatchQueue.main.async {
                for line in out.split(whereSeparator: \.isNewline) {
                    let text = String(line)
                    if let data = text.data(using: .utf8),
                       let event = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                        if event["event"] as? String == "error" {
                            let detail = event["detail"] as? String ?? "failed"
                            self.lastError = detail
                            self.log("  error: \(detail)")
                        } else {
                            self.log("  \(event["event"] as? String ?? text)")
                        }
                    } else {
                        self.log(text)
                    }
                }
                if code != 0 && self.lastError == nil { self.lastError = "\(title) failed; see the log." }
                self.busyAction = nil
                self.refresh()
            }
        }
    }

    private func log(_ line: String) {
        logLines.append(line)
        if logLines.count > 200 { logLines.removeFirst(logLines.count - 200) }
    }

    private func decode<T: Decodable>(_ args: [String], as type: T.Type) -> T? {
        let (out, code) = run(args)
        guard code == 0, let data = out.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(type, from: data)
    }

    private func run(_ args: [String]) -> (String, Int32) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: cliPath)
        process.arguments = args
        process.currentDirectoryURL = home
        process.environment = SparkServePaths.processEnvironment()
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        do { try process.run() } catch { return ("error: \(error.localizedDescription)", -1) }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return (String(data: data, encoding: .utf8)?.trimmingCharacters(in: .newlines) ?? "", process.terminationStatus)
    }
}

enum StrixFacts {
    static let engines = [("halogen", "Halogen"), ("lemonade", "Lemonade"), ("vllm", "vLLM")]

    static func phaseColor(_ phase: String?) -> Color {
        switch phase {
        case "ready": return .green
        case "starting": return .orange
        case "failed": return .red
        default: return .gray
        }
    }

    static func gib(_ value: Double?) -> String {
        guard let value else { return "—" }
        return String(format: "%.1f GiB", value)
    }
}

struct StrixView: View {
    @ObservedObject var store: StrixStore
    @State private var selected: String?

    private var selectedModel: ModelEntry? { store.models.first { $0.id == selected } }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                statusPanel
                VStack(alignment: .leading, spacing: 8) {
                    Text("Choose a model")
                        .font(.headline)
                    Text("One engine runs at a time; starting a model stops the others on the box.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    grid
                }
                controls
                if !store.logLines.isEmpty { log }
            }
            .padding(2)
        }
        .onAppear {
            store.startPolling()
            if selected == nil { selected = store.status?.model_id ?? store.models.first?.id }
        }
        .onDisappear { store.stopPolling() }
        .onChange(of: store.models.map(\.id)) { ids in
            // Models arrive after the first SSH round trip; select the serving one then.
            if selected == nil || !ids.contains(selected!) { selected = store.status?.model_id ?? ids.first }
        }
    }

    private var statusPanel: some View {
        let status = store.status
        let color = StrixFacts.phaseColor(store.isBusy ? "starting" : status?.phase)
        return VStack(alignment: .leading, spacing: 10) {
            HStack {
                Circle().fill(color).frame(width: 10, height: 10)
                Text(status?.host ?? "Strix Halo")
                    .font(.headline)
                Spacer()
                if let action = store.busyAction {
                    ProgressView().controlSize(.small)
                    Text(action).font(.caption).foregroundStyle(.secondary)
                }
            }
            if let status {
                if status.engine == "none" {
                    Text("No engine running").foregroundStyle(.secondary)
                } else {
                    Text(store.models.first { $0.id == status.model_id }?.label ?? status.model)
                        .font(.subheadline.weight(.semibold))
                    Text("\(StrixFacts.engines.first { $0.0 == status.engine }?.1 ?? status.engine) · \(status.phase)"
                         + (status.url.map { " · \($0)" } ?? ""))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
                resources(status.resources)
            } else {
                Text("Waiting for the box…").foregroundStyle(.secondary)
            }
            if let error = store.lastError {
                Text(error).font(.caption).foregroundStyle(.red)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(color.opacity(0.6), lineWidth: 1))
    }

    private func resources(_ r: StrixResources?) -> some View {
        HStack(spacing: 24) {
            tile("Free memory", r.map { "\(StrixFacts.gib($0.ram_free_gib)) of \(StrixFacts.gib($0.ram_total_gib))" } ?? "—")
            tile("GPU", r?.busy_pct.map { String(format: "%.0f%%", $0) } ?? "—")
            tile("GPU memory (GTT)", r.map { "\(StrixFacts.gib($0.gtt_used_gib)) of \(StrixFacts.gib($0.gtt_total_gib))" } ?? "—")
            tile("Temperature", r?.temp_c.map { String(format: "%.0f °C", $0) } ?? "—")
            tile("Power", r?.power_w.map { String(format: "%.1f W", $0) } ?? "—")
        }
    }

    private func tile(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.callout.monospacedDigit())
        }
    }

    private var grid: some View {
        HStack(alignment: .top, spacing: 12) {
            ForEach(StrixFacts.engines, id: \.0) { engine, title in
                VStack(alignment: .leading, spacing: 8) {
                    Text(title)
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.secondary)
                        .accessibilityAddTraits(.isHeader)
                    ForEach(store.models.filter { $0.backend == engine }) { model in
                        Button { selected = model.id } label: {
                            ModelCard(model: model, isSelected: selected == model.id,
                                      isCurrent: store.status?.model_id == model.id, showNotes: true)
                        }
                        .buttonStyle(.plain)
                        .accessibilityLabel(model.label)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .topLeading)
            }
        }
        .padding(.horizontal, BestBadge.size / 2)
        .padding(.vertical, 2)
    }

    private var controls: some View {
        HStack(spacing: 12) {
            if let model = selectedModel {
                if store.status?.model_id == model.id && store.status?.phase == "ready" {
                    Label("Serving \(model.label)", systemImage: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                } else {
                    Button("Start \(model.label)") { store.start(model) }
                        .buttonStyle(.borderedProminent)
                        .disabled(store.isBusy)
                }
            }
            Button("Stop engine") { store.stop() }
                .disabled(store.isBusy || store.status?.engine == "none")
            Button("Refresh") { store.refresh() }
            Text("All three engines require the box's API key; vLLM takes about 2 minutes to start.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var log: some View {
        VStack(alignment: .leading, spacing: 2) {
            ForEach(Array(store.logLines.suffix(8).enumerated()), id: \.offset) { _, line in
                Text(line).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
            }
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(nsColor: .textBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }
}
