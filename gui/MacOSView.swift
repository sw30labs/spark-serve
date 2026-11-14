import SwiftUI
import AppKit

// MARK: - This Mac Studio
//
// Splash on the local Mac (ADR 0020). The app drives `spark-serve macos …`.
// Each card is its own splash server. Splash 1.6 exits the launcher, so status
// follows the detached process (or the port, when no process can be verified).
// Port 8000 is oMLX and is never bound, stopped, or selected for Hermes.
// On-disk status is whether the card's model_dir and draft_dir exist. Use in
// Hermes writes the same config as the Strix tab, and only once that card is ready.

struct MacOSMachine: Codable {
    var name: String? = nil
    var product: String? = nil
    var chip: String? = nil
    var memory_gb: Int? = nil
}

struct MacOSServer: Codable, Identifiable {
    let id: String
    let label: String
    let phase: String
    let port: Int
    let host: String
    var url: String? = nil
    let model: String
    var format: String? = nil
    var draft_model: String? = nil
    let model_dir: String
    let draft_dir: String
    let model_on_disk: Bool
    let draft_on_disk: Bool
    let on_disk: Bool
    var pid: Int? = nil
    var port_busy: Bool? = nil
    var error: String? = nil
}

struct MacOSSite: Codable {
    let machine: MacOSMachine
    let host: String
    let reserved_ports: [Int]
    let splash: String
    let models: [MacOSServer]
}

struct MacOSLogs: Codable, Identifiable {
    let id: String
    let path: String
    let lines: [String]
}

enum MacOSFacts {
    static let reservedPort = 8000

    static func diskNote(_ server: MacOSServer) -> String {
        if server.on_disk { return "On disk" }
        if !server.model_on_disk && !server.draft_on_disk { return "Not on disk yet" }
        if !server.model_on_disk { return "Weights not on disk yet" }
        return "Draft not on disk yet"
    }

    static func canStart(_ server: MacOSServer) -> Bool {
        server.port != reservedPort && server.on_disk && server.port_busy != true
            && server.phase != "ready" && server.phase != "starting" && server.phase != "busy"
    }

    static func canStop(_ server: MacOSServer) -> Bool {
        server.port != reservedPort && server.pid != nil
            && (server.phase == "ready" || server.phase == "starting")
    }

    /// Shown when the port is serving but Stop has no verified Splash process.
    static func stopNote(_ server: MacOSServer) -> String? {
        guard server.pid == nil else { return nil }
        guard server.phase == "ready" || server.phase == "starting" else { return nil }
        if let error = server.error, !error.isEmpty { return error }
        return "This port is serving this model, but no Splash process could be verified. Stop stays off."
    }

    static func phaseTitle(_ server: MacOSServer) -> String {
        switch server.phase {
        case "ready": return "serving"
        case "starting": return "starting"
        case "busy": return "busy with another model"
        case "failed": return "failed"
        default: return "stopped"
        }
    }

    /// Why Start is off, or nil when it can run. A serving card has no start blocker.
    static func blocker(_ server: MacOSServer) -> String? {
        if server.port == reservedPort { return "Port 8000 is oMLX; splash is never started there" }
        if server.phase == "ready" || server.phase == "starting" { return nil }
        if server.phase == "busy" {
            return server.error ?? "Port \(server.port) is busy with another model"
        }
        if !server.model_on_disk { return "Weights not on disk yet" }
        if !server.draft_on_disk { return "Draft not on disk yet" }
        if server.port_busy == true { return "Port \(server.port) is in use; it will not be stopped" }
        return nil
    }

    static func phaseColor(_ phase: String) -> Color {
        switch phase {
        case "ready": return .green
        case "starting", "busy": return .orange
        case "failed": return .red
        default: return .gray
        }
    }
}

final class MacOSStore: ObservableObject {
    @Published var models: [ModelEntry] = []
    @Published private(set) var site: MacOSSite?
    @Published private(set) var configured = false
    @Published private(set) var busyAction: String?
    @Published private(set) var busyIsStart = false
    @Published var lastError: String?
    @Published var logLines: [String] = []
    @Published var logEntries: [ActivityEntry] = []
    @Published var shownLogs: MacOSLogs?
    private var logSequence = 0

    private let home: URL?
    private let cliPath: String
    private var backgroundTimer: Timer?
    private var timer: Timer?
    private var refreshing = false
    private var refreshQueued = false
    private var viewers = 0
    private var running: Process?
    private var lineBuffer = Data()
    private var cancelled = false
    private var afterCancel: (() -> Void)?
    /// Set for the duration of a Hermes selection so the log can confirm it the way the Strix screen does.
    private var pendingHermes: PendingHermes?
    private var didReportHermes = false

    private struct PendingHermes {
        let title: String
        let host: String
    }

    init(poll: Bool = false) {
        home = SparkServePaths.home()
        cliPath = home?.appendingPathComponent("spark-serve").path ?? ""
        guard poll else { return }
        backgroundTimer = Timer.scheduledTimer(withTimeInterval: 15, repeats: true) { [weak self] _ in self?.refresh() }
        refresh()
    }

    var isBusy: Bool { busyAction != nil }
    var servers: [MacOSServer] { site?.models ?? [] }
    func server(_ id: String) -> MacOSServer? { servers.first { $0.id == id } }

    func attach() {
        viewers += 1
        guard timer == nil else { return }
        refresh()
        // While this tab is open, status is polled so a handoff can enable Hermes
        // without restarting the app. Start also refreshes when it reports ready.
        timer = Timer.scheduledTimer(withTimeInterval: 5, repeats: true) { [weak self] _ in self?.refresh() }
    }

    func detach() {
        viewers = max(0, viewers - 1)
        guard viewers == 0 else { return }
        timer?.invalidate()
        timer = nil
    }

    func refresh() {
        guard !cliPath.isEmpty else { return }
        if refreshing {
            refreshQueued = true
            return
        }
        refreshing = true
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            let models = self.decode(["macos", "list", "--json"], as: [ModelEntry].self)
            let site = models == nil ? nil : self.decode(["macos", "status", "--json"], as: MacOSSite.self)
            DispatchQueue.main.async {
                self.refreshing = false
                if let models { self.models = models }
                self.configured = models != nil || self.configured
                if let site { self.site = site }
                if site == nil && self.configured && self.lastError == nil {
                    self.lastError = "Splash status did not answer."
                } else if site != nil, !self.isBusy, self.lastError == "Splash status did not answer." {
                    self.lastError = nil
                }
                if self.refreshQueued {
                    self.refreshQueued = false
                    self.refresh()
                }
            }
        }
    }

    func apply(_ site: MacOSSite) { self.site = site }

    /// `spark-serve macos use` for a ready card. Nil until it is ready, and never for port 8000.
    func hermesArguments(_ id: String, target: HermesTarget) -> [String]? {
        guard let server = server(id), server.phase == "ready", server.port != MacOSFacts.reservedPort,
              let url = server.url, !url.isEmpty else { return nil }
        if let port = URL(string: url)?.port, port == MacOSFacts.reservedPort { return nil }
        return ["macos", "use", id, "--hermes-target", target.rawValue, "--json"]
    }

    func start(_ id: String) {
        guard let server = server(id), MacOSFacts.canStart(server) else { return }
        perform("Starting \(server.label)", ["macos", "up", id], isStart: true)
    }

    func stop(_ id: String) {
        guard let server = server(id) else { return }
        let title = "Stopping \(server.label)"
        let args = ["macos", "stop", id]
        if busyIsStart, let running, running.isRunning {
            log("  abandoning: \(busyAction ?? "start")")
            cancelled = true
            afterCancel = { [weak self] in self?.perform(title, args, isStart: false) }
            running.terminate()
            return
        }
        guard MacOSFacts.canStop(server) else { return }
        perform(title, args, isStart: false)
    }

    /// Same control as the Strix panels: write the selected ready model into Hermes on this Mac or the head Spark.
    func useInHermes(_ id: String, target: HermesTarget) {
        guard !isBusy, let server = server(id), hermesArguments(id, target: target) != nil else { return }
        didReportHermes = false
        pendingHermes = PendingHermes(title: target.title, host: server.host)
        log("  connecting \(target.title) to \(server.host)…")
        perform("Connecting \(target.title)", ["macos", "use", id, "--hermes-target", target.rawValue, "--json"], isStart: false)
    }

    func showLogs(_ id: String) {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            let (out, code) = self.run(["macos", "logs", id, "--lines", "120", "--json"])
            let last = out.split(whereSeparator: \.isNewline).last.map(String.init) ?? ""
            let logs = code == 0 ? try? JSONDecoder().decode(MacOSLogs.self, from: Data(last.utf8)) : nil
            DispatchQueue.main.async {
                if let logs { self.shownLogs = logs }
                else { self.lastError = MacOSStore.errorDetail(last) ?? "Could not read the log." }
            }
        }
    }

    private func perform(_ title: String, _ args: [String], isStart: Bool) {
        guard !isBusy else { return }
        busyAction = title
        busyIsStart = isStart
        lastError = nil
        cancelled = false
        lineBuffer = Data()
        log("$ spark-serve \(args.joined(separator: " "))")
        let process = makeProcess(args)
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        do { try process.run() } catch {
            lastError = "\(title) failed: \(error.localizedDescription)"
            busyAction = nil
            busyIsStart = false
            pendingHermes = nil
            return
        }
        running = process
        DispatchQueue.global(qos: .utility).async { [weak self] in
            while true {
                let data = pipe.fileHandleForReading.availableData
                if data.isEmpty { break }
                DispatchQueue.main.async { self?.consume(data) }
            }
            process.waitUntilExit()
            let code = process.terminationStatus
            DispatchQueue.main.async {
                guard let self else { return }
                self.consume(Data([0x0a]))
                if code == 0, let pending = self.pendingHermes, !self.didReportHermes {
                    self.log("  \(pending.title) now uses \(pending.host)")
                }
                self.pendingHermes = nil
                self.didReportHermes = false
                if code != 0 && !self.cancelled && self.lastError == nil { self.lastError = "\(title) failed; see the log." }
                self.running = nil
                self.busyAction = nil
                self.busyIsStart = false
                let next = self.afterCancel
                self.afterCancel = nil
                self.refresh()
                next?()
            }
        }
    }

    private func consume(_ chunk: Data) {
        lineBuffer.append(chunk)
        while let newline = lineBuffer.firstIndex(of: 0x0a) {
            let line = String(decoding: lineBuffer[lineBuffer.startIndex..<newline], as: UTF8.self)
            lineBuffer.removeSubrange(lineBuffer.startIndex...newline)
            handle(line.trimmingCharacters(in: .whitespaces))
        }
    }

    private func handle(_ line: String) {
        guard !line.isEmpty else { return }
        guard let data = line.data(using: .utf8),
              let event = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let name = event["event"] as? String else {
            log(line)
            return
        }
        switch name {
        case "error":
            let detail = event["detail"] as? String ?? "failed"
            if !cancelled { lastError = detail }
            log("  error: \(detail)")
        case "progress":
            if let detail = event["detail"] as? String { log("  \(detail)") }
        case "hermes":
            if let detail = event["detail"] as? String { log("  hermes: \(detail)") }
        case "selected":
            let target = HermesTarget(rawValue: event["hermes_target"] as? String ?? "local") ?? .local
            let host = event["host"] as? String ?? ""
            let destination = (event["hermes_host"] as? String).map { " on \($0)" } ?? ""
            let served = event["served"] as? String
            let label = (event["label"] as? String)
                ?? models.first { $0.served_name == served }?.label
                ?? served
            log("  \(target.title)\(destination) uses \(host)\(label.map { ": \($0)" } ?? "")")
            didReportHermes = true
        case "ready":
            log("  ready")
            refresh()
        case "stopped":
            log("  stopped")
            refresh()
        case "starting":
            if let port = event["port"] as? Int { log("  starting on port \(port)") }
            if let argv = event["argv"] as? [String] { log("  \(argv.joined(separator: " "))") }
            refresh()
        default:
            log("  \(name)")
        }
    }

    private func log(_ line: String) {
        logLines.append(line)
        if logLines.count > 200 { logLines.removeFirst(logLines.count - 200) }
        appendActivity(&logEntries, line, source: "MacOS", cap: 200, next: &logSequence)
    }

    static func errorDetail(_ line: String) -> String? {
        guard let data = line.data(using: .utf8),
              let event = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              event["event"] as? String == "error" else { return nil }
        return event["detail"] as? String
    }

    private func decode<T: Decodable>(_ args: [String], as type: T.Type) -> T? {
        let (out, code) = run(args)
        guard code == 0, let data = out.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(type, from: data)
    }

    private func makeProcess(_ args: [String]) -> Process {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: cliPath)
        process.arguments = args
        process.currentDirectoryURL = home
        process.environment = SparkServePaths.processEnvironment()
        return process
    }

    private func run(_ args: [String]) -> (String, Int32) {
        let process = makeProcess(args)
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        do { try process.run() } catch { return ("error: \(error.localizedDescription)", -1) }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return (String(data: data, encoding: .utf8)?.trimmingCharacters(in: .newlines) ?? "", process.terminationStatus)
    }
}

// MARK: - MacOS tab

struct MacOSView: View {
    @ObservedObject var store: MacOSStore
    @State private var selected: String?

    private var selectedServer: MacOSServer? { selected.flatMap { store.server($0) } }
    private var selectedModel: ModelEntry? { store.models.first { $0.id == selected } }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                machinePanel
                VStack(alignment: .leading, spacing: 8) {
                    Text("Choose a model")
                        .font(.headline)
                    Text("Each card runs its own splash serve on its own port. Start and Stop affect only that process. Port 8000 is oMLX and is left alone. A card can start once its model and draft directories both exist.")
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
            store.attach()
            if selected == nil { selected = store.servers.first { $0.phase == "ready" }?.id ?? store.models.first?.id ?? store.servers.first?.id }
        }
        .onDisappear { store.detach() }
        .onChange(of: store.servers.map(\.id)) { ids in
            if selected == nil || !ids.contains(selected!) {
                selected = store.servers.first { $0.phase == "ready" }?.id ?? ids.first
            }
        }
        .sheet(item: $store.shownLogs) { logs in
            MacOSLogSheet(logs: logs) { store.shownLogs = nil }
        }
    }

    private var machinePanel: some View {
        let machine = store.site?.machine
        let title = [machine?.name, machine?.product, machine?.chip,
                     machine?.memory_gb.map { "\($0) GB unified memory" }]
            .compactMap { $0 }.joined(separator: " · ")
        return VStack(alignment: .leading, spacing: 6) {
            HStack {
                Image(systemName: "desktopcomputer")
                Text(title.isEmpty ? "Mac Studio" : title).font(.headline)
                Spacer()
                if let host = store.site?.host {
                    Text("\(host) · splash").font(.caption).foregroundStyle(.secondary)
                }
            }
            Text("Reserved port \(MacOSFacts.reservedPort) (oMLX) is never started or stopped.")
                .font(.caption)
                .foregroundStyle(.secondary)
            if store.site == nil {
                Text(store.configured ? "Waiting for Splash…" : "No [macos] section in models.toml.")
                    .foregroundStyle(.secondary)
            }
            if let error = store.lastError, store.site == nil {
                Text(error).font(.caption).foregroundStyle(.red)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color(nsColor: .controlBackgroundColor)))
    }

    private var grid: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 210), spacing: 12, alignment: .top)],
                  alignment: .leading, spacing: 12) {
            ForEach(store.servers) { server in
                Button { selected = server.id } label: { card(server) }
                    .buttonStyle(.plain)
                    .accessibilityLabel(server.label)
            }
        }
        .padding(.horizontal, BestBadge.size / 2)
        .padding(.vertical, 2)
    }

    private func card(_ server: MacOSServer) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            if let model = store.models.first(where: { $0.id == server.id }) {
                ModelCard(model: model, isSelected: selected == server.id,
                          isCurrent: server.phase == "ready" || server.phase == "starting", showNotes: true)
            } else {
                Text(server.label).font(.headline)
            }
            Text(":\(server.port) · \(MacOSFacts.phaseTitle(server)) · \(MacOSFacts.diskNote(server))")
                .font(.caption2)
                .foregroundStyle(server.phase == "busy" ? Color.orange : (server.on_disk ? Color.secondary : Color.orange))
                .padding(.leading, 4)
        }
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 10) {
            if let server = selectedServer {
                HStack(alignment: .center, spacing: 12) {
                    startControl(server)
                    Button("Stop") { store.stop(server.id) }
                        .disabled(!MacOSFacts.canStop(server) || (store.isBusy && !store.busyIsStart))
                        .help(MacOSFacts.stopNote(server) ?? "Stop this card's Splash process")
                    Button("Logs") { store.showLogs(server.id) }
                        .disabled(store.isBusy)
                    hermesButtons(server)
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text(server.model_dir).font(.system(.caption2, design: .monospaced)).textSelection(.enabled)
                    Text(server.draft_dir).font(.system(.caption2, design: .monospaced)).textSelection(.enabled)
                }
                .foregroundStyle(.secondary)
                if let url = server.url {
                    Text(url).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                }
                if let note = MacOSFacts.stopNote(server) {
                    Text(note).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                } else if let error = server.error {
                    Text(error).font(.caption).foregroundStyle(.orange).textSelection(.enabled)
                }
            }
            HStack(spacing: 12) {
                Button("Refresh") { store.refresh() }
                if let action = store.busyAction {
                    ProgressView().controlSize(.small)
                    Text(action).font(.caption).foregroundStyle(.secondary)
                }
            }
            if let error = store.lastError {
                Text(error).font(.caption).foregroundStyle(.red).textSelection(.enabled)
            }
            if let model = selectedModel, model.performance == nil {
                Text("Unmeasured. The card shows a speed once measured-performance.json has a row for it.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func startControl(_ server: MacOSServer) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            if server.phase == "ready" || server.phase == "starting" {
                Label(server.phase == "ready" ? "Serving on :\(server.port)" : "Starting on :\(server.port)…",
                      systemImage: "checkmark.circle.fill")
                    .foregroundStyle(MacOSFacts.phaseColor(server.phase))
            } else if server.phase == "busy" {
                Label(MacOSFacts.blocker(server) ?? "Busy with another model",
                      systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(MacOSFacts.phaseColor(server.phase))
                    .font(.caption)
            } else {
                Button("Start") { store.start(server.id) }
                    .buttonStyle(.borderedProminent)
                    .disabled(store.isBusy || !MacOSFacts.canStart(server))
                Text(MacOSFacts.blocker(server) ?? " ")
                    .font(.caption2)
                    .foregroundColor(MacOSFacts.blocker(server) == nil ? .secondary : .orange)
                    .lineLimit(2)
            }
        }
        .frame(minWidth: 220, alignment: .leading)
    }

    /// Same labels, style, and help as the Strix panel. Disabled until the selected card is ready.
    private func hermesButtons(_ server: MacOSServer) -> some View {
        HStack(spacing: 8) {
            ForEach(HermesTarget.allCases, id: \.self) { target in
                Button("Use in \(target.title)") { store.useInHermes(server.id, target: target) }
                    .buttonStyle(.bordered)
                    .disabled(store.isBusy || store.hermesArguments(server.id, target: target) == nil)
                    .accessibilityLabel("Use \(server.label) in \(target.title)")
                    .help(server.phase == "busy"
                          ? (server.error ?? "This port is busy with another model")
                          : (target == .local ? "Update Hermes on this Mac" : "Update Hermes on the head Spark"))
            }
        }
    }

    private var log: some View {
        VStack(alignment: .leading, spacing: 2) {
            ForEach(Array(store.logLines.suffix(10).enumerated()), id: \.offset) { _, line in
                Text(line).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
            }
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(nsColor: .textBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }
}

struct MacOSLogSheet: View {
    let logs: MacOSLogs
    let close: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(logs.path).font(.headline).lineLimit(1)
                Spacer()
                Button("Done", action: close).keyboardShortcut(.defaultAction)
            }
            ScrollView {
                Text(logs.lines.isEmpty ? "No log lines." : logs.lines.joined(separator: "\n"))
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .background(Color(nsColor: .textBackgroundColor))
        }
        .padding(16)
        .frame(minWidth: 760, minHeight: 440)
    }
}

struct MacOSMenuSection: View {
    @ObservedObject var store: MacOSStore

    var body: some View {
        if store.configured {
            Divider()
            Menu(store.busyAction.map { "MacOS · \($0)…" } ?? "MacOS") {
                if store.servers.isEmpty { Text("No Splash cards") }
                ForEach(store.servers) { server in
                    Button(menuTitle(server)) { store.start(server.id) }
                        .disabled(store.isBusy || !MacOSFacts.canStart(server))
                    if MacOSFacts.canStop(server) {
                        Button("Stop \(server.label)") { store.stop(server.id) }
                            .disabled(store.isBusy && !store.busyIsStart)
                    }
                }
                if let error = store.lastError {
                    Divider()
                    Text(error)
                }
            }
        }
    }

    private func menuTitle(_ server: MacOSServer) -> String {
        if server.phase == "ready" {
            return server.pid == nil ? "✓ \(server.label) (Stop unavailable)" : "✓ \(server.label)"
        }
        if let blocker = MacOSFacts.blocker(server) { return "\(server.label) — \(blocker)" }
        return "\(server.label) · :\(server.port)"
    }
}
