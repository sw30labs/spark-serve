import SwiftUI
import AppKit
import Foundation

// MARK: - Models

struct ModelEntry: Identifiable, Codable {
    let id: String
    let label: String
    let aliases: [String]
    let served_name: String
    let ctx: Int
    let image: String
    let notes: String
    let wrapper: String
    let hermes_provider: String
}

struct Container: Codable {
    let name: String
    let state: String
    let image: String
    let running: Bool
}

struct ClusterStatus: Codable {
    let head: String
    let worker: String
    let url: String
    let head_containers: [Container]
    let worker_containers: [Container]
    let ready: Bool
    let served: String?
    let foreign_served: String?
    let ours_running: Bool
    let vllm_running: Bool
    let port_busy: Bool
    let v1_models_raw: String
}

enum BootState {
    case idle
    case launching(model: String)
    case booting(model: String, elapsed: Int)
    case ready(model: String, served: String?)
    case failed(message: String)
}

// MARK: - Paths

enum SparkServePaths {
    static func home() -> URL? {
        if let env = ProcessInfo.processInfo.environment["SPARK_SERVE_HOME"], !env.isEmpty {
            return URL(fileURLWithPath: env)
        }
        if let plist = Bundle.main.object(forInfoDictionaryKey: "SparkServeHome") as? String, !plist.isEmpty {
            return URL(fileURLWithPath: plist)
        }
        var url = Bundle.main.bundleURL
        for _ in 0..<8 {
            let cli = url.appendingPathComponent("spark-serve")
            let toml = url.appendingPathComponent("models.toml")
            if FileManager.default.isExecutableFile(atPath: cli.path),
               FileManager.default.fileExists(atPath: toml.path) {
                return url
            }
            url.deleteLastPathComponent()
        }
        return nil
    }

    static func processEnvironment() -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let extras = [
            "\(home)/miniconda3/bin",
            "\(home)/miniforge3/bin",
            "/opt/homebrew/bin",
            "/usr/local/bin",
        ]
        let path = env["PATH"] ?? "/usr/bin:/bin:/usr/sbin:/sbin"
        env["PATH"] = (extras + [path]).joined(separator: ":")
        return env
    }
}

// MARK: - CLI Runner

final class CLIRunner: ObservableObject {
    @Published var status: ClusterStatus?
    @Published var models: [ModelEntry] = []
    @Published var bootState: BootState = .idle
    @Published var logLines: [String] = []

    private let home: URL?
    private let cliPath: String
    private var pollTimer: Timer?
    private var upProcess: Process?
    private var lineBuf = Data()
    private var lastDecodeError: String?

    var isBusy: Bool {
        switch bootState {
        case .launching, .booting: return true
        default: return false
        }
    }

    var badgeColor: Color {
        switch bootState {
        case .ready: return .green
        case .booting, .launching: return .orange
        case .failed: return .red
        case .idle:
            if status?.ready == true { return .green }
            if status?.foreign_served != nil { return .orange }
            return .gray
        }
    }

    var badgeText: String {
        switch bootState {
        case .ready(_, let served): return "serving \(served ?? "")"
        case .booting(_, let elapsed): return "booting (\(elapsed)s)"
        case .launching(let m): return "starting \(m)…"
        case .failed(let msg): return msg
        case .idle:
            if let s = status, s.ready {
                return "serving \(s.served ?? "")"
            }
            if let foreign = status?.foreign_served {
                return "foreign: \(foreign)"
            }
            return "cluster idle"
        }
    }

    init() {
        home = SparkServePaths.home()
        cliPath = home?.appendingPathComponent("spark-serve").path ?? ""
        startPolling()
    }

    deinit {
        stopPolling()
        upProcess?.terminate()
    }

    func refresh() {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            if self.cliPath.isEmpty || !FileManager.default.isExecutableFile(atPath: self.cliPath) {
                DispatchQueue.main.async {
                    self.bootState = .failed(message: "CLI not found. Set SPARK_SERVE_HOME to the repo root.")
                }
                return
            }
            let models = self.decode(["list", "--json"], as: [ModelEntry].self) ?? []
            let status = self.decode(["status", "--json"], as: ClusterStatus.self)
            DispatchQueue.main.async {
                self.models = models
                self.status = status
                self.syncBootState(with: status)
            }
        }
    }

    func startPolling(interval: TimeInterval = 15) {
        stopPolling()
        pollTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            self?.refresh()
        }
        refresh()
    }

    func stopPolling() {
        pollTimer?.invalidate()
        pollTimer = nil
    }

    func startUp(model: String, noHermes: Bool = false) {
        guard !isBusy else { return }
        bootState = .launching(model: model)
        logLines = []
        lineBuf = Data()
        lastDecodeError = nil

        var args: [String] = ["up", model, "--json"]
        if noHermes { args.append("--no-hermes") }

        let process = makeProcess(args)
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        upProcess = process

        do {
            try process.run()
        } catch {
            bootState = .failed(message: "Failed to launch: \(error.localizedDescription)")
            upProcess = nil
            return
        }

        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            DispatchQueue.main.async {
                self?.consumeStdout(data, model: model)
            }
        }

        process.terminationHandler = { [weak self] proc in
            pipe.fileHandleForReading.readabilityHandler = nil
            DispatchQueue.main.async {
                self?.flushStdout(model: model)
                self?.upProcess = nil
                self?.finishUp(statusCode: proc.terminationStatus)
                self?.refresh()
            }
        }
    }

    func stop() {
        appendLog("$ spark-serve stop")
        if let proc = upProcess, proc.isRunning {
            proc.terminate()
            upProcess = nil
        }
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            let out = self.runRaw(["stop"])
            DispatchQueue.main.async {
                for line in out.split(whereSeparator: \.isNewline) {
                    let s = String(line).trimmingCharacters(in: .whitespaces)
                    if !s.isEmpty { self.appendLog(s) }
                }
                self.bootState = .idle
            }
            self.refresh()
        }
    }

    func openMainWindow() {
        NSApp.activate(ignoringOtherApps: true)
        if let window = NSApplication.shared.windows.first(where: { $0.title == "spark-serve" }) {
            window.makeKeyAndOrderFront(nil)
        }
    }

    private func syncBootState(with status: ClusterStatus?) {
        guard let status else { return }
        switch bootState {
        case .launching, .booting:
            return
        case .ready:
            if !status.ready {
                bootState = .idle
            }
        case .failed:
            break
        case .idle:
            if status.ready, let served = status.served {
                let mid = models.first(where: { $0.served_name == served || $0.id == served })?.id ?? served
                bootState = .ready(model: mid, served: served)
            }
        }
    }

    private func finishUp(statusCode: Int32) {
        switch bootState {
        case .ready, .failed:
            return
        case .launching, .booting:
            if statusCode != 0 {
                bootState = .failed(message: "up exited \(statusCode)")
            } else {
                bootState = .idle
            }
        case .idle:
            if statusCode != 0 {
                bootState = .failed(message: "up exited \(statusCode)")
            }
        }
    }

    private func consumeStdout(_ chunk: Data, model: String) {
        if chunk.isEmpty { return }
        lineBuf.append(chunk)
        let nl = Data([0x0a])
        while let range = lineBuf.range(of: nl) {
            let lineData = lineBuf.subdata(in: lineBuf.startIndex..<range.lowerBound)
            lineBuf.removeSubrange(lineBuf.startIndex..<range.upperBound)
            if let line = String(data: lineData, encoding: .utf8) {
                let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    parseEvent(trimmed, model: model)
                }
            }
        }
    }

    private func flushStdout(model: String) {
        if lineBuf.isEmpty { return }
        if let line = String(data: lineBuf, encoding: .utf8) {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                parseEvent(trimmed, model: model)
            }
        }
        lineBuf = Data()
    }

    private func parseEvent(_ line: String, model: String) {
        guard let data = line.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let event = obj["event"] as? String else {
            appendLog(line)
            return
        }
        switch event {
        case "start":
            let label = obj["label"] as? String ?? model
            appendLog("── Starting \(label) (\(model)) ──")
        case "stop":
            if let host = obj["host"] as? String {
                appendLog("  stop \(host)")
            }
        case "drop_caches":
            if let host = obj["host"] as? String {
                appendLog("  drop_caches \(host): \(obj["output"] as? String ?? "")")
            }
        case "worker_start":
            appendLog("  worker started")
        case "head_start":
            appendLog("  head started")
        case "hermes":
            if let detail = obj["detail"] as? String {
                appendLog("  hermes: \(detail)")
            }
        case "waiting":
            bootState = .booting(model: model, elapsed: 0)
            appendLog("  waiting for /v1/models …")
        case "poll":
            if let elapsed = jsonInt(obj, "elapsed") {
                bootState = .booting(model: model, elapsed: elapsed)
            }
        case "ready":
            let served = obj["served"] as? String ?? model
            bootState = .ready(model: model, served: served)
            appendLog("  ready — serving \(served)")
        case "timeout":
            bootState = .failed(message: "Timed out waiting for /v1/models")
            appendLog("  timeout")
        case "error":
            let detail = obj["detail"] as? String ?? "error"
            bootState = .failed(message: detail)
            appendLog("  error: \(detail)")
        case "launched":
            appendLog("  launched (--no-wait)")
            bootState = .idle
        default:
            appendLog(line)
        }
    }

    private func jsonInt(_ obj: [String: Any], _ key: String) -> Int? {
        if let i = obj[key] as? Int { return i }
        if let n = obj[key] as? NSNumber { return n.intValue }
        return nil
    }

    private func appendLog(_ line: String) {
        logLines.append(line)
        if logLines.count > 400 {
            logLines.removeFirst(logLines.count - 400)
        }
    }

    private func decode<T: Decodable>(_ args: [String], as type: T.Type) -> T? {
        let out = runRaw(args)
        guard let data = out.data(using: .utf8), !data.isEmpty else { return nil }
        do {
            return try JSONDecoder().decode(type, from: data)
        } catch {
            let msg = "decode \(args.joined(separator: " ")): \(error.localizedDescription)"
            if lastDecodeError != msg {
                lastDecodeError = msg
                DispatchQueue.main.async { [weak self] in
                    self?.appendLog(msg)
                }
            }
            return nil
        }
    }

    private func makeProcess(_ args: [String]) -> Process {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: cliPath)
        process.arguments = args
        process.currentDirectoryURL = home
        process.environment = SparkServePaths.processEnvironment()
        return process
    }

    @discardableResult
    private func runRaw(_ args: [String]) -> String {
        let process = makeProcess(args)
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        do { try process.run() } catch { return "error: \(error.localizedDescription)" }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return String(data: data, encoding: .utf8)?.trimmingCharacters(in: .newlines) ?? ""
    }
}

// MARK: - App

struct SparkServeApp: App {
    @StateObject private var runner = CLIRunner()
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        Window("spark-serve", id: "main") {
            MainView()
                .environmentObject(runner)
                .frame(minWidth: 520, minHeight: 380)
        }
        .windowStyle(.titleBar)
        .defaultSize(width: 560, height: 420)

        MenuBarExtra {
            MenuBarView()
                .environmentObject(runner)
        } label: {
            MenuBarLabel()
                .environmentObject(runner)
        }
    }
}

class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        if !flag {
            sender.windows.first(where: { $0.title == "spark-serve" })?.makeKeyAndOrderFront(nil)
        }
        return true
    }
}

// MARK: - Menu Bar Label

struct MenuBarLabel: View {
    @EnvironmentObject var runner: CLIRunner

    var body: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(runner.badgeColor)
                .frame(width: 8, height: 8)
            Text("spark")
                .font(.system(size: 11, weight: .medium))
        }
    }
}

// MARK: - Menu Bar Content

struct MenuBarView: View {
    @EnvironmentObject var runner: CLIRunner

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Circle()
                    .fill(runner.badgeColor)
                    .frame(width: 10, height: 10)
                Text(runner.badgeText)
                    .font(.system(size: 12, weight: .medium))
                    .lineLimit(2)
            }
            .padding(.bottom, 4)

            if runner.status?.foreign_served != nil, runner.status?.ready != true {
                Text("Stop the foreign process on :8000 before up.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 8)
            }

            Divider()

            ForEach(runner.models) { m in
                Button(action: {
                    runner.openMainWindow()
                    runner.startUp(model: m.id)
                }) {
                    HStack {
                        Text("↑ \(m.label)")
                        Spacer()
                        Text(fmtCtx(m.ctx))
                            .foregroundStyle(.secondary)
                            .font(.caption)
                    }
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 8)
                .disabled(runner.isBusy)
            }

            Divider()

            Button("Stop cluster") {
                runner.stop()
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 8)

            Divider()

            Button("Open spark-serve window") {
                runner.openMainWindow()
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 8)

            Button("Quit") {
                NSApp.terminate(nil)
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 8)
        }
        .padding(.vertical, 6)
        .frame(width: 260)
    }

    private func fmtCtx(_ n: Int) -> String {
        if n >= 1_048_576 { return "1M ctx" }
        if n >= 262_144 { return "262k ctx" }
        if n >= 32_768 { return "\(n / 1024)k ctx" }
        return "\(n) ctx"
    }
}

// MARK: - Main Window

struct MainView: View {
    @EnvironmentObject var runner: CLIRunner
    @State private var selectedModel: String?
    @State private var showNotes = true

    var body: some View {
        VStack(spacing: 12) {
            header
            Divider()
            modelPicker
            controls
            Divider()
            logView
        }
        .padding(16)
        .onAppear {
            if selectedModel == nil {
                selectedModel = runner.models.first?.id
            }
        }
        .onChange(of: runner.models.count) { _ in
            if selectedModel == nil {
                selectedModel = runner.models.first?.id
            }
        }
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("spark-serve")
                    .font(.title2.bold())
                if let s = runner.status {
                    Text("\(s.head) + \(s.worker)  ·  \(s.url)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                if let foreign = runner.status?.foreign_served, runner.status?.ready != true {
                    Text("foreign on :8000: \(foreign)")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
            }
            Spacer()
            HStack(spacing: 6) {
                Circle()
                    .fill(runner.badgeColor)
                    .frame(width: 10, height: 10)
                Text(runner.badgeText)
                    .font(.caption.weight(.medium))
                    .lineLimit(2)
            }
            .padding(8)
            .background(runner.badgeColor.opacity(0.12))
            .clipShape(Capsule())
        }
    }

    private var isCurrentModel: String? {
        switch runner.bootState {
        case .ready(let m, _): return m
        case .booting(let m, _): return m
        case .launching(let m): return m
        default:
            return runner.status?.served.flatMap { served in
                runner.models.first(where: { $0.served_name == served || $0.id == served })?.id
            }
        }
    }

    private var modelPicker: some View {
        HStack(spacing: 12) {
            ForEach(runner.models) { m in
                ModelCard(
                    model: m,
                    isSelected: selectedModel == m.id,
                    isCurrent: isCurrentModel == m.id,
                    showNotes: showNotes
                )
                .onTapGesture { selectedModel = m.id }
            }
            Spacer()
        }
    }

    @ViewBuilder
    private var controls: some View {
        HStack(spacing: 12) {
            Button("Start \(selectedModel ?? "")") {
                if let m = selectedModel {
                    runner.startUp(model: m)
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(selectedModel == nil || runner.isBusy)

            Button("Stop") {
                runner.stop()
            }
            .buttonStyle(.bordered)
            .tint(.red)

            Button("Refresh") {
                runner.refresh()
            }
            .buttonStyle(.bordered)

            if case .booting(_, let elapsed) = runner.bootState {
                Text("elapsed \(elapsed)s")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var logView: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Boot log")
                .font(.caption.weight(.medium))
                .foregroundStyle(.secondary)
            ScrollView {
                VStack(alignment: .leading, spacing: 1) {
                    ForEach(Array(runner.logLines.enumerated()), id: \.offset) { _, line in
                        Text(line)
                            .font(.system(.caption, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                    }
                }
                .padding(8)
            }
            .frame(maxHeight: .infinity)
            .background(Color(nsColor: .textBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 6))
        }
    }
}

// MARK: - Model Card

struct ModelCard: View {
    let model: ModelEntry
    let isSelected: Bool
    let isCurrent: Bool
    let showNotes: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(model.label)
                    .font(.headline)
                if isCurrent {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                        .font(.caption)
                }
            }
            Text("served: \(model.served_name)")
                .font(.caption)
                .foregroundStyle(.secondary)
            HStack(spacing: 8) {
                Label(ctxLabel, systemImage: "text.line.inherit")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Label(model.wrapper, systemImage: "cube")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            if showNotes && !model.notes.isEmpty {
                Text(model.notes)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .lineLimit(2)
            }
        }
        .padding(10)
        .frame(width: 180)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(isSelected ? Color.accentColor.opacity(0.12) : Color(nsColor: .controlBackgroundColor))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(isSelected ? Color.accentColor : Color.gray.opacity(0.2), lineWidth: isSelected ? 1.5 : 1)
        )
    }

    private var ctxLabel: String {
        if model.ctx >= 1_048_576 { return "1M ctx" }
        if model.ctx >= 262_144 { return "262k ctx" }
        return "\(model.ctx / 1024)k ctx"
    }
}

// Top-level entry point (avoids -parse-as-library + @main, which trips the
// SwiftUICore "allowed client" link check for a module this size on
// CommandLineTools / Swift 6.3).
SparkServeApp.main()
