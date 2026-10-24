import SwiftUI
import AppKit

// MARK: - Strix Halo site
//
// A separate store and views for the AMD boxes. They drive `spark-serve strix …`,
// which talks to halo-ctl on each box over SSH; the Spark runner and controller
// are not involved. Each box enforces one engine at a time and the CLI keeps the
// cluster pair consistent, so the app only asks. The menu bar offers what the
// boxes' GNOME widget offers: per box, each engine's models on disk, and Stop.

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

struct StrixMetrics: Codable {
    let active: Double?
    let queued: Double?
    let kv_pct: Double?
    let avg_gen_tps: Double?
    let avg_prompt_tps: Double?
    let last_gen_tps: Double?
    let draft_accept_pct: Double?
    var generation_tokens_total: Double? = nil
    var prompt_tokens_total: Double? = nil
}

/// A model halo-ctl found on a box's disk.
struct StrixDiskModel: Codable, Hashable {
    let id: String
    let label: String
}

/// One box. `role` is "master" or "worker" while the cluster runs; `error` means no reading.
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
    var metrics: StrixMetrics? = nil
    /// Context the running engine reports serving (not the catalog's value).
    var served_context: Int? = nil
    /// halo-ctl's models on disk per engine; nil when the box cannot say.
    var available: [String: [StrixDiskModel]]? = nil
    var rpc_worker: String? = nil
    var rpc_clients: Int? = nil
    var role: String? = nil
    var peer: String? = nil
    var error: String? = nil

    /// An engine is up or coming up on this box itself.
    var isServing: Bool { StrixFacts.engineIDs.contains(engine) && phase != "stopped" }
    var isWorker: Bool { role == "worker" }
}

struct StrixCluster: Codable {
    let master: String
    let worker: String?
    let phase: String
    let model_id: String?
    let url: String?
    var conflict: Bool? = nil
}

struct StrixSite: Codable {
    let boxes: [StrixStatus]
    let cluster: StrixCluster?
}

struct StrixConnection: Codable {
    let host: String
    let engine: String
    let base_url: String?
    let model: String
    let api_key: String?
}

struct StrixLogs: Codable, Identifiable {
    let host: String
    let units: [String]
    let lines: [String]
    var id: String { host }
}

/// Something to start: a catalog card, or a model on a box's disk that the catalog lacks.
struct StrixChoice: Identifiable, Hashable {
    /// Catalog id, or `<engine>:<halo-ctl id>` as `spark-serve strix up` takes it.
    let id: String
    let engine: String
    let ctlID: String
    let label: String
    let catalogued: Bool
}

final class StrixStore: ObservableObject {
    @Published var models: [ModelEntry] = []
    @Published private(set) var boxes: [StrixStatus] = []
    @Published private(set) var cluster: StrixCluster?
    @Published private(set) var busyAction: String?
    /// A start can be abandoned by Stop; a stop runs to the end.
    @Published private(set) var busyIsStart = false
    @Published var lastError: String?
    @Published var logLines: [String] = []
    @Published var shownLogs: StrixLogs?
    /// False when the catalog has no [strix] section; the Overview and menu then hide the boxes.
    @Published private(set) var configured = false
    @Published private(set) var sampledAt: Date?
    /// Recent samples keyed "<host>/gpu" and "<host>/decode".
    @Published private(set) var history: [String: [MetricPoint]] = [:]
    /// Rates from successive counter samples of the same engine and model (Lemonade, vLLM), by host.
    @Published private(set) var liveDecode: [String: Double] = [:]
    @Published private(set) var livePrefill: [String: Double] = [:]
    private var previousCounters: [String: (key: String, at: Double, generation: Double, prompt: Double)] = [:]

    private let home: URL?
    private let cliPath: String
    private var backgroundTimer: Timer?
    private var timer: Timer?
    private var refreshing = false
    private var viewers = 0
    private var running: Process?
    private var lineBuffer = Data()
    private var cancelled = false
    private var afterCancel: (() -> Void)?

    /// `poll` reads the boxes every 15 s for the menu bar, window or not; tests pass false.
    init(poll: Bool = false) {
        home = SparkServePaths.home()
        cliPath = home?.appendingPathComponent("spark-serve").path ?? ""
        guard poll else { return }
        backgroundTimer = Timer.scheduledTimer(withTimeInterval: 15, repeats: true) { [weak self] _ in self?.refresh() }
        refresh()
    }

    var isBusy: Bool { busyAction != nil }

    func box(_ host: String) -> StrixStatus? { boxes.first { $0.host == host } }

    /// Each visible view attaches; polling runs every 5 s while at least one is on screen,
    /// whatever order tab switches deliver onAppear and onDisappear in.
    func attach() {
        viewers += 1
        guard timer == nil else { return }
        refresh()
        timer = Timer.scheduledTimer(withTimeInterval: 5, repeats: true) { [weak self] _ in self?.refresh() }
    }

    func detach() {
        viewers = max(0, viewers - 1)
        guard viewers == 0 else { return }
        timer?.invalidate()
        timer = nil
    }

    func refresh() {
        guard !refreshing, !cliPath.isEmpty else { return }
        refreshing = true
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            let models = self.decode(["strix", "list", "--json"], as: [ModelEntry].self)
            let site = models == nil ? nil : self.decode(["strix", "status", "--json"], as: StrixSite.self)
            DispatchQueue.main.async {
                self.refreshing = false
                if let models { self.models = models }
                self.configured = models != nil || self.configured
                self.apply(site, at: Date())
                if site == nil && self.configured && self.lastError == nil {
                    self.lastError = "The Strix boxes did not answer; check the [strix] hosts."
                } else if site != nil, !self.isBusy, self.lastError?.hasPrefix("The Strix boxes") == true {
                    self.lastError = nil
                }
            }
        }
    }

    func apply(_ site: StrixSite?, at date: Date) {
        boxes = site?.boxes ?? []
        cluster = site?.cluster
        guard let site else { return }
        sampledAt = date
        let now = date.timeIntervalSince1970
        for status in site.boxes {
            updateRates(status, at: now)
            let decode = status.engine == "halogen" ? status.metrics?.avg_gen_tps : liveDecode[status.host]
            let values: [String: Double?] = ["gpu": status.resources?.busy_pct, "decode": decode]
            for (key, value) in values {
                let name = "\(status.host)/\(key)"
                var points = history[name] ?? []
                if let value { points.append(MetricPoint(timestamp: now, value: value)) }
                points.removeAll { $0.timestamp < now - TelemetryStore.retentionSeconds }
                history[name] = points
            }
        }
    }

    func updateRates(_ status: StrixStatus, at now: Double) {
        let host = status.host
        let key = "\(status.engine)/\(status.model)"
        guard let generation = status.metrics?.generation_tokens_total,
              let prompt = status.metrics?.prompt_tokens_total else {
            liveDecode[host] = nil; livePrefill[host] = nil; previousCounters[host] = nil
            return
        }
        if let previous = previousCounters[host], previous.key == key, now > previous.at,
           now - previous.at <= 30, generation >= previous.generation, prompt >= previous.prompt {
            liveDecode[host] = (generation - previous.generation) / (now - previous.at)
            livePrefill[host] = (prompt - previous.prompt) / (now - previous.at)
        } else {
            liveDecode[host] = nil; livePrefill[host] = nil  // first sample, a restart, or a counter reset
        }
        previousCounters[host] = (key, now, generation, prompt)
    }

    // MARK: Choices

    /// Catalog cards, then models on any box's disk that the catalog lacks, for one engine.
    func choices(engine: String) -> [StrixChoice] {
        let catalog = models.filter { $0.backend == engine }.map {
            StrixChoice(id: $0.id, engine: engine, ctlID: $0.ctl_id ?? $0.served_name, label: $0.label, catalogued: true)
        }
        var known = Set(catalog.map(\.ctlID))
        var extra: [StrixChoice] = []
        for box in boxes {
            for disk in box.available?[engine] ?? [] where !known.contains(disk.id) {
                known.insert(disk.id)
                extra.append(StrixChoice(id: "\(engine):\(disk.id)", engine: engine, ctlID: disk.id,
                                         label: disk.label, catalogued: false))
            }
        }
        return catalog + extra
    }

    /// What the widget lists for one box: the engine's models on its disk.
    func menuChoices(engine: String, on host: String) -> [StrixChoice] {
        choices(engine: engine).filter { onDisk($0, host) != false }
    }

    /// nil when the box cannot say (no reading, or a halo-ctl without `models`).
    func onDisk(_ choice: StrixChoice, _ host: String) -> Bool? {
        guard let available = box(host)?.available else { return nil }
        return (available[choice.engine] ?? []).contains { $0.id == choice.ctlID }
    }

    /// The box serves, or is starting, this choice itself.
    func serves(_ choice: StrixChoice, _ host: String) -> Bool {
        guard let b = box(host), b.engine == choice.engine, b.isServing else { return false }
        if choice.engine == "halogen" { return true }
        if let id = b.model_id { return id == choice.id }
        return b.model == choice.ctlID
    }

    func isServingAnywhere(_ choice: StrixChoice) -> Bool { boxes.contains { serves(choice, $0.host) } }

    /// Why a start on this box is not offered, or nil when it is.
    func blocker(_ choice: StrixChoice, on host: String) -> String? {
        guard let b = box(host) else { return "No reading from \(host) yet" }
        if b.error != nil { return "\(host) did not answer" }
        if onDisk(choice, host) == false { return "Not on \(host)'s disk" }
        if serves(choice, host) { return b.phase == "ready" ? "Serving on \(host)" : "Starting on \(host)" }
        if choice.engine == "cluster" {
            guard let peerName = b.peer, let peer = box(peerName) else { return "The cluster needs two boxes" }
            if peer.error != nil { return "\(peerName), its worker, did not answer" }
            if peer.rpc_worker != "active" { return "\(peerName)'s RPC worker is \(peer.rpc_worker ?? "unknown")" }
        } else if b.isWorker, let master = b.peer, box(master)?.role != "master" {
            return "Busy as a cluster worker; its master is out of reach"
        }
        return nil
    }

    /// What a start on this box stops first, said before the click.
    func consequence(_ choice: StrixChoice, on host: String) -> String? {
        guard let b = box(host) else { return nil }
        var stops: [String] = []
        if choice.engine == "cluster" {
            if let peerName = b.peer, let peer = box(peerName), peer.isServing {
                stops.append("\(peer.role == "master" ? "the cluster" : StrixFacts.engineTitle(peer.engine)) on \(peerName)")
            }
            if b.isServing && b.role != "master" { stops.append("\(StrixFacts.engineTitle(b.engine)) on \(host)") }
        } else {
            if b.isWorker, let master = b.peer { stops.append("the cluster (master \(master))") }
            if b.isServing { stops.append(b.role == "master" ? "the cluster" : "\(StrixFacts.engineTitle(b.engine)) on \(host)") }
        }
        return stops.isEmpty ? nil : "Stops " + stops.joined(separator: " and ")
    }

    /// The widget's status row: engine · phase · model, or the box's cluster role.
    func summary(_ b: StrixStatus) -> String {
        if b.error != nil { return "No answer" }
        if b.isWorker && !b.isServing { return "Cluster worker for \(b.peer ?? "the other box")" }
        guard b.isServing else { return "Stopped" }
        var bits = [StrixFacts.engineTitle(b.engine), b.phase]
        if let label = servingLabel(b) { bits.append(label) }
        return bits.joined(separator: " · ")
    }

    func servingLabel(_ b: StrixStatus) -> String? {
        guard b.isServing else { return nil }
        if let id = b.model_id, let model = models.first(where: { $0.id == id }) { return model.label }
        if let disk = b.available?[b.engine]?.first(where: { $0.id == b.model }) { return disk.label }
        return b.model.isEmpty ? nil : b.model
    }

    // MARK: Actions

    func start(_ choice: StrixChoice, on host: String) {
        perform("Starting \(choice.label) on \(host)", ["strix", "up", choice.id, "--box", host], isStart: true)
    }

    /// Stop one box (its cluster, for the worker) or both. A running start is abandoned first.
    func stop(_ host: String?) {
        let title = host.map { "Stopping \($0)" } ?? "Stopping both boxes"
        let args = ["strix", "stop"] + (host.map { ["--box", $0] } ?? [])
        if busyIsStart, let running, running.isRunning {
            log("  abandoning: \(busyAction ?? "start")")
            cancelled = true
            afterCancel = { [weak self] in self?.perform(title, args, isStart: false) }
            running.terminate()
            return
        }
        perform(title, args, isStart: false)
    }

    /// The widget's "Copy connection details": base URL, model and API key on the clipboard, never in the log.
    func copyConnection(_ host: String) {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            let (out, code) = self.run(["strix", "connection", "--box", host, "--json"])
            let last = out.split(whereSeparator: \.isNewline).last.map(String.init) ?? ""
            let details = code == 0 ? try? JSONDecoder().decode(StrixConnection.self, from: Data(last.utf8)) : nil
            let failure = StrixStore.errorDetail(last)
            DispatchQueue.main.async {
                guard let details else {
                    self.lastError = failure ?? "Could not read \(host)'s connection details."
                    return
                }
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(StrixFacts.connectionText(details), forType: .string)
                self.log("Copied \(details.host)'s connection details, API key included, to the clipboard")
            }
        }
    }

    func showLogs(_ host: String) {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            let (out, code) = self.run(["strix", "logs", "--box", host, "--lines", "120", "--json"])
            let last = out.split(whereSeparator: \.isNewline).last.map(String.init) ?? ""
            let logs = code == 0 ? try? JSONDecoder().decode(StrixLogs.self, from: Data(last.utf8)) : nil
            DispatchQueue.main.async {
                if let logs { self.shownLogs = logs }
                else { self.lastError = StrixStore.errorDetail(last) ?? "Could not read \(host)'s logs." }
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
            return
        }
        running = process
        // Drain through EOF before finishing, so the final event is never dropped.
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
        case "ready":
            log("  ready")
        default:
            log("  \(name)")
        }
    }

    private func log(_ line: String) {
        logLines.append(line)
        if logLines.count > 200 { logLines.removeFirst(logLines.count - 200) }
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

enum StrixFacts {
    static let engines = [("halogen", "Halogen"), ("cluster", "Cluster"), ("lemonade", "Lemonade"), ("vllm", "vLLM")]
    static let engineIDs = Set(engines.map(\.0))

    static func engineTitle(_ engine: String?) -> String {
        engines.first { $0.0 == engine }?.1 ?? engine ?? "—"
    }

    static func phaseColor(_ phase: String?) -> Color {
        switch phase {
        case "ready": return .green
        case "starting": return .orange
        case "worker": return .teal
        case "failed", "unreachable": return .red
        default: return .gray
        }
    }

    static func boxColor(_ b: StrixStatus?, busy: Bool = false) -> Color {
        guard let b else { return .gray }
        if b.error != nil { return phaseColor("unreachable") }
        if busy { return phaseColor("starting") }
        if b.isWorker && !b.isServing { return phaseColor("worker") }
        return phaseColor(b.phase)
    }

    static func gib(_ value: Double?) -> String {
        guard let value else { return "—" }
        return String(format: "%.1f GiB", value)
    }

    static func connectionText(_ c: StrixConnection) -> String {
        ["Engine:   \(engineTitle(c.engine)) on \(c.host)",
         "Base URL: \(c.base_url ?? "—")",
         "Model:    \(c.model)",
         "API key:  \(c.api_key ?? "(no API key set)")"].joined(separator: "\n")
    }

    /// The CLI prefixes errors with the box, which every view already shows.
    static func errorText(_ error: String, host: String) -> String {
        error.hasPrefix("\(host): ") ? String(error.dropFirst(host.count + 2)) : error
    }

    /// The fix for the commonest failure: a box that does not take this Mac's key.
    static func hint(_ error: String?, host: String) -> String? {
        guard let error, error.contains("Permission denied") else { return nil }
        return "Set up key login once from Terminal: ssh-copy-id \(host)"
    }
}

// MARK: - Strix Halo tab

struct StrixView: View {
    @ObservedObject var store: StrixStore
    @State private var selected: String?

    private var allChoices: [StrixChoice] { StrixFacts.engines.flatMap { store.choices(engine: $0.0) } }
    private var selectedChoice: StrixChoice? { allChoices.first { $0.id == selected } }
    private var servingChoice: String? { allChoices.first { store.isServingAnywhere($0) }?.id }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                boxPanels
                if let cluster = store.cluster { clusterBanner(cluster) }
                VStack(alignment: .leading, spacing: 8) {
                    Text("Choose a model")
                        .font(.headline)
                    Text("Each box runs one engine at a time and the cluster takes both boxes. Starting stops whatever is in the way; each button says what.")
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
            if selected == nil { selected = servingChoice ?? allChoices.first?.id }
        }
        .onDisappear { store.detach() }
        .onChange(of: allChoices.map(\.id)) { ids in
            // Models arrive after the first SSH round trip; select the serving one then.
            if selected == nil || !ids.contains(selected!) { selected = servingChoice ?? ids.first }
        }
        .sheet(item: $store.shownLogs) { logs in
            StrixLogSheet(logs: logs) { store.shownLogs = nil }
        }
    }

    private var boxPanels: some View {
        HStack(alignment: .top, spacing: 12) {
            if store.boxes.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Strix Halo").font(.headline)
                    Text(store.configured ? "Waiting for the boxes…" : "No [strix] section in models.toml.")
                        .foregroundStyle(.secondary)
                    if let error = store.lastError { Text(error).font(.caption).foregroundStyle(.red) }
                }
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(RoundedRectangle(cornerRadius: 8).fill(Color(nsColor: .controlBackgroundColor)))
            }
            ForEach(store.boxes, id: \.host) { box in
                StrixBoxPanel(store: store, box: box)
            }
        }
    }

    private func clusterBanner(_ cluster: StrixCluster) -> some View {
        let loading = cluster.phase == "starting"
        return HStack(spacing: 8) {
            Image(systemName: "point.3.connected.trianglepath.dotted")
            Text("Cluster: \(cluster.master) is the master, \(cluster.worker ?? "the other box") the worker · \(cluster.phase)"
                 + (loading ? " · about 2 minutes once the worker has cached its share; a first load takes much longer" : ""))
            if cluster.conflict == true {
                Text("Both boxes report being master; stop one.").foregroundStyle(.red)
            }
        }
        .font(.caption)
        .foregroundStyle(loading ? .orange : .secondary)
    }

    private var grid: some View {
        HStack(alignment: .top, spacing: 12) {
            ForEach(StrixFacts.engines, id: \.0) { engine, title in
                VStack(alignment: .leading, spacing: 8) {
                    Text(title)
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.secondary)
                        .accessibilityAddTraits(.isHeader)
                    let list = store.choices(engine: engine)
                    if list.isEmpty {
                        Text("No models").font(.caption).foregroundStyle(.tertiary)
                    }
                    ForEach(list) { choice in
                        Button { selected = choice.id } label: { card(choice) }
                            .buttonStyle(.plain)
                            .accessibilityLabel(choice.label)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .topLeading)
            }
        }
        .padding(.horizontal, BestBadge.size / 2)
        .padding(.vertical, 2)
    }

    private func card(_ choice: StrixChoice) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            if let model = store.models.first(where: { $0.id == choice.id }) {
                ModelCard(model: model, isSelected: selected == choice.id,
                          isCurrent: store.isServingAnywhere(choice), showNotes: true)
            } else {
                StrixDiskCard(choice: choice, isSelected: selected == choice.id, isCurrent: store.isServingAnywhere(choice))
            }
            Text(diskNote(choice))
                .font(.caption2)
                .foregroundStyle(.secondary)
                .padding(.leading, 4)
        }
    }

    private func diskNote(_ choice: StrixChoice) -> String {
        let known = store.boxes.filter { store.onDisk(choice, $0.host) != nil }
        guard !known.isEmpty else { return " " }
        let on = known.filter { store.onDisk(choice, $0.host) == true }.map(\.host)
        return on.isEmpty ? "Not on disk" : "On disk: " + on.joined(separator: ", ")
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 10) {
            if let choice = selectedChoice {
                HStack(alignment: .top, spacing: 16) {
                    ForEach(store.boxes, id: \.host) { box in
                        startButton(choice, box.host)
                    }
                }
            }
            HStack(spacing: 12) {
                Button("Stop both boxes") { store.stop(nil) }
                    .disabled((store.isBusy && !store.busyIsStart) || !store.boxes.contains { $0.isServing || $0.isWorker })
                Button("Refresh") { store.refresh() }
                if let action = store.busyAction {
                    ProgressView().controlSize(.small)
                    Text(action).font(.caption).foregroundStyle(.secondary)
                }
            }
            if let error = store.lastError {
                Text(error).font(.caption).foregroundStyle(.red).textSelection(.enabled)
            }
            Text("Every engine requires the box's API key. vLLM takes about 2 minutes to start; Stop abandons a start that is still loading.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private func startButton(_ choice: StrixChoice, _ host: String) -> some View {
        let blocker = store.blocker(choice, on: host)
        let consequence = store.consequence(choice, on: host)
        let title = choice.engine == "cluster" ? "Start with \(host) as master" : "Start on \(host)"
        return VStack(alignment: .leading, spacing: 4) {
            if store.serves(choice, host) {
                Label(store.box(host)?.phase == "ready" ? "Serving on \(host)" : "Starting on \(host)…",
                      systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)
            } else {
                Button(title) { store.start(choice, on: host) }
                    .buttonStyle(.borderedProminent)
                    .disabled(store.isBusy || blocker != nil)
                Text(blocker ?? consequence ?? " ")
                    .font(.caption2)
                    .foregroundColor(blocker == nil ? .secondary : .orange)
                    .lineLimit(2)
            }
        }
        .frame(minWidth: 200, alignment: .leading)
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

struct StrixBoxPanel: View {
    @ObservedObject var store: StrixStore
    let box: StrixStatus

    private var stoppable: Bool { box.isServing || box.isWorker }

    var body: some View {
        let color = StrixFacts.boxColor(box)
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Circle().fill(color).frame(width: 10, height: 10)
                Text(box.host).font(.headline)
                if let role = box.role {
                    Text(role == "master" ? "cluster master" : "cluster worker")
                        .font(.caption2.weight(.medium))
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .background(Capsule().fill(StrixFacts.phaseColor("worker").opacity(0.18)))
                }
                Spacer()
            }
            Text(store.summary(box))
                .font(.subheadline.weight(.semibold))
            if let error = box.error {
                Text(StrixFacts.errorText(error, host: box.host)).font(.caption).foregroundStyle(.red).textSelection(.enabled)
                if let hint = StrixFacts.hint(error, host: box.host) {
                    Text(hint).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                }
            } else {
                if box.isServing {
                    Text([box.url, box.served_context.map { "\(ModelCardFacts.contextTitle(ctx: $0, backend: box.engine)) served" }]
                            .compactMap { $0 }.joined(separator: " · "))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
                resources(box.resources)
                Text("Inference mode \(box.inference ?? "—") · RPC worker \(box.rpc_worker ?? "—")")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .help("Inference mode pauses background jobs while Halogen or the cluster runs. It needs the box's password, so switch it from the box's widget. The RPC worker lends this box's GPU to a cluster master.")
                HStack(spacing: 8) {
                    Button(box.isWorker && !box.isServing ? "Stop cluster" : "Stop") { store.stop(box.host) }
                        .disabled(!stoppable || (store.isBusy && !store.busyIsStart))
                    Button("Copy connection details") { store.copyConnection(box.host) }
                        .disabled(!stoppable)
                    Button("Logs") { store.showLogs(box.host) }
                }
                .controlSize(.small)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(color.opacity(0.6), lineWidth: 1))
    }

    private func resources(_ r: StrixResources?) -> some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 120), alignment: .topLeading)], alignment: .leading, spacing: 8) {
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
}

/// A model on a box's disk with no catalog entry: startable, but without card facts.
struct StrixDiskCard: View {
    let choice: StrixChoice
    let isSelected: Bool
    let isCurrent: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(choice.label).font(.headline).lineLimit(2).help(choice.label)
                if isCurrent {
                    Image(systemName: "checkmark.circle.fill").foregroundStyle(.green).font(.caption)
                }
            }
            Text("on disk: \(choice.ctlID)")
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(2)
            PillFlow(spacing: 6) {
                InfoPill(text: ModelCardFacts.runtimeTitle(wrapper: choice.engine),
                         color: ModelCardFacts.runtimeColor(wrapper: choice.engine))
                InfoPill(text: "Not in the catalog", color: ModelCardFacts.neutralColor)
            }
            Text("Context, quantization and speed are unknown until it gets a [strix.models] entry.")
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .lineLimit(2)
        }
        .padding(10)
        .frame(maxWidth: .infinity, minHeight: 150, alignment: .topLeading)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(isSelected ? Color.accentColor.opacity(0.12) : Color(nsColor: .controlBackgroundColor))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(isSelected ? Color.accentColor : Color.gray.opacity(0.2), lineWidth: isSelected ? 1.5 : 1)
        )
    }
}

struct StrixLogSheet: View {
    let logs: StrixLogs
    let close: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("\(logs.host) · \(logs.units.joined(separator: ", "))").font(.headline)
                Spacer()
                Button("Done", action: close).keyboardShortcut(.defaultAction)
            }
            ScrollView {
                Text(logs.lines.isEmpty ? "No journal lines." : logs.lines.joined(separator: "\n"))
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

// MARK: - Menu bar

/// The GNOME widget's menu, for every box: engines, their models on disk, Stop and the connection details.
struct StrixMenuSection: View {
    @ObservedObject var store: StrixStore

    var body: some View {
        if store.configured {
            Divider()
            Text(store.busyAction.map { "Strix Halo · \($0)…" } ?? "Strix Halo")
            ForEach(store.boxes, id: \.host) { box in
                Menu("\(box.host) · \(store.summary(box))") {
                    if let error = box.error {
                        Text(StrixFacts.errorText(error, host: box.host))
                        if let hint = StrixFacts.hint(error, host: box.host) { Text(hint) }
                    } else {
                        ForEach(StrixFacts.engines, id: \.0) { engine, title in
                            Menu(title) {
                                let list = store.menuChoices(engine: engine, on: box.host)
                                if list.isEmpty { Text("No models on disk") }
                                ForEach(list) { choice in
                                    Button(menuTitle(choice, box.host)) { store.start(choice, on: box.host) }
                                        .disabled(store.isBusy || store.blocker(choice, on: box.host) != nil)
                                }
                            }
                        }
                        Divider()
                        Button(box.isWorker && !box.isServing ? "Stop the cluster" : "Stop") { store.stop(box.host) }
                            .disabled(!(box.isServing || box.isWorker) || (store.isBusy && !store.busyIsStart))
                        Button("Copy connection details") { store.copyConnection(box.host) }
                            .disabled(!(box.isServing || box.isWorker))
                    }
                }
            }
            if let error = store.lastError {
                Text(error).lineLimit(2)
            }
        }
    }

    /// The widget marks the serving model; here a disabled entry also says why, and an enabled one what it stops.
    private func menuTitle(_ choice: StrixChoice, _ host: String) -> String {
        let role = choice.engine == "cluster" ? " (this box as master)" : ""
        if store.serves(choice, host) { return "✓ " + choice.label + role }
        let note = store.blocker(choice, on: host) ?? store.consequence(choice, on: host)
        return choice.label + role + (note.map { " — \($0)" } ?? "")
    }
}

// MARK: - Overview cards

extension StrixStore {
    /// live, stale (no fresh sample for 20 s) or unavailable, as the Spark cards use.
    func readingState(_ host: String, now: Date) -> String {
        guard let b = box(host), b.error == nil, let sampledAt else { return "unavailable" }
        return now.timeIntervalSince(sampledAt) > 20 ? "stale" : "live"
    }

    /// Boxes with a workload of their own; the cluster's worker shows on its master's card.
    var servingHosts: [String] { boxes.filter(\.isServing).map(\.host) }
}

struct StrixHostCard: View {
    @ObservedObject var store: StrixStore
    let host: String
    let now: Date
    private var b: StrixStatus? { store.box(host) }
    private var r: StrixResources? { b?.resources }
    private var usedFraction: Double? {
        guard let free = r?.ram_free_gib, let total = r?.ram_total_gib, total > 0 else { return nil }
        return min(1, max(0, (total - free) / total))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label(host, systemImage: "cpu").font(.headline)
                Spacer()
                ReadingState(state: store.readingState(host, now: now))
            }
            Text("Strix Halo · " + (b?.error != nil ? "no answer" : (b?.role.map { "cluster \($0) · " } ?? "")
                 + (store.sampledAt.map { MetricFormat.age($0.timeIntervalSince1970, now: now) } ?? "No sample yet")))
                .font(.caption).foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 3) {
                MetricTile(title: "GPU utilization", value: MetricFormat.number(r?.busy_pct, suffix: "%", digits: 0))
                MetricTrend(points: store.history["\(host)/gpu"] ?? [], color: .green, label: "GPU utilization", now: now)
            }
            Divider()
            VStack(alignment: .leading, spacing: 5) {
                HStack {
                    Text("Unified RAM").font(.subheadline.weight(.medium))
                    Spacer()
                    Text(r.map { "\(StrixFacts.gib(($0.ram_total_gib ?? 0) - ($0.ram_free_gib ?? 0))) / \(StrixFacts.gib($0.ram_total_gib))" } ?? "—")
                        .font(.caption).monospacedDigit()
                        .help("Used = total minus real free memory; locked model weights count as used, unlike `free`.")
                }
                if let usedFraction {
                    ProgressView(value: usedFraction).tint(usedFraction > 0.9 ? .orange : .accentColor)
                } else {
                    Capsule().fill(Color.secondary.opacity(0.1)).frame(height: 4)
                }
                HStack {
                    Text("No swap")
                    Spacer()
                    Text("GPU (GTT) \(StrixFacts.gib(r?.gtt_used_gib)) / \(StrixFacts.gib(r?.gtt_total_gib))")
                }
                .font(.caption2).foregroundStyle(.secondary)
            }
            HStack {
                MetricTile(title: "GPU temperature", value: MetricFormat.number(r?.temp_c, suffix: " °C", digits: 0))
                MetricTile(title: "GPU power", value: MetricFormat.number(r?.power_w, suffix: " W"))
            }
            if let error = b?.error {
                Text(StrixFacts.errorText(error, host: host)).font(.caption).foregroundStyle(.orange).lineLimit(3).help(error)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .topLeading)
        .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.secondary.opacity(0.15)))
    }
}

struct StrixWorkloadCard: View {
    @ObservedObject var store: StrixStore
    let host: String
    let now: Date
    private var b: StrixStatus? { store.box(host) }
    private var m: StrixMetrics? { b?.metrics }
    private var halogen: Bool { b?.engine == "halogen" }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(b.flatMap { store.servingLabel($0) } ?? "—").font(.headline).textSelection(.enabled)
                    Text("\(host)\(b?.role == "master" ? " + \(b?.peer ?? "worker")" : "") · \(StrixFacts.engineTitle(b?.engine))")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                ReadingState(state: store.readingState(host, now: now) == "stale" ? "stale" : (b?.phase ?? "unavailable"))
            }
            if let url = b?.url {
                Text(url).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary).textSelection(.enabled)
            }
            HStack(spacing: 18) {
                VStack(alignment: .leading, spacing: 3) {
                    MetricTile(title: halogen ? "Decode (recent average)" : "Decode (live)",
                               value: MetricFormat.number(halogen ? m?.avg_gen_tps : store.liveDecode[host], suffix: " tok/s"))
                    MetricTrend(points: store.history["\(host)/decode"] ?? [], color: .green, label: "Decode tokens per second", now: now)
                }
                MetricTile(title: halogen ? "Prefill (recent average)" : "Prefill (finished requests)",
                           value: MetricFormat.number(halogen ? m?.avg_prompt_tps : store.livePrefill[host], suffix: " tok/s", digits: 0))
                    .help(halogen ? "Halogen's average over recent requests." : "Prompt tokens are counted when a request finishes, so this moves in steps.")
            }
            Divider()
            HStack {
                MetricTile(title: "Context (served)", value: b?.served_context.map {
                    ModelCardFacts.contextTitle(ctx: $0, backend: b?.engine) } ?? "—")
                    .help("Reported by the running engine, not the catalog.")
                MetricTile(title: "Running / waiting", value: "\(MetricFormat.number(m?.active, digits: 0)) / \(MetricFormat.number(m?.queued, digits: 0))")
                MetricTile(title: halogen ? "KV pool" : "KV cache", value: MetricFormat.number(m?.kv_pct, suffix: "%"))
                if halogen {
                    MetricTile(title: "Draft acceptance", value: MetricFormat.number(m?.draft_accept_pct, suffix: "%", digits: 0))
                }
            }
            if b?.engine == "cluster" {
                Text("The cluster's llama-server publishes no counters unless it runs with --metrics; — means not reported.")
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .topLeading)
        .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.secondary.opacity(0.15)))
    }
}
