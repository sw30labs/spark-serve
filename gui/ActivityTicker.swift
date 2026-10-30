import SwiftUI
import AppKit

// MARK: - Activity ticker
//
// One-line strip at the bottom of the window, after Singularity Atlas's
// footer ticker (web/style.css #ticker-wrap): 26 pt tall, top hairline, dark
// strip, monospace, newest first, scrolling right to left at a steady pace.
// Atlas has no expand; here a click opens the full log in a popover so the
// history the old Activity log frame showed is still one click away.

struct ActivityEntry: Identifiable, Equatable {
    let id: Int
    let date: Date
    let text: String
    /// "Spark" or "Strix": which store logged it.
    let source: String
}

/// Pure ticker logic; covered by gui/tests.
enum TickerFacts {
    static let idleText = "No activity yet"
    static let limit = 12

    private static let clock: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = .current
        f.dateFormat = "HH:mm:ss"
        return f
    }()

    static func time(_ date: Date) -> String { clock.string(from: date) }

    /// Newest first across both stores, blank lines dropped, at most `limit`.
    static func items(_ lists: [ActivityEntry]..., limit: Int = TickerFacts.limit) -> [ActivityEntry] {
        let all: [ActivityEntry] = lists.flatMap { $0 }
        let shown: [ActivityEntry] = all.filter(isShown)
        let newest: [ActivityEntry] = shown.sorted { newer($0, than: $1) }
        return Array(newest.prefix(limit))
    }

    /// Oldest first, for the popover.
    static func history(_ lists: [ActivityEntry]...) -> [ActivityEntry] {
        let all: [ActivityEntry] = lists.flatMap { $0 }
        return all.sorted { newer($1, than: $0) }
    }

    private static func isShown(_ entry: ActivityEntry) -> Bool {
        !entry.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private static func newer(_ a: ActivityEntry, than b: ActivityEntry) -> Bool {
        if a.date != b.date { return a.date > b.date }
        return a.id > b.id
    }

    static func display(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespaces).replacingOccurrences(of: "\n", with: " ")
    }

    static func line(_ entry: ActivityEntry) -> String { "\(time(entry.date))  \(display(entry.text))" }

    /// Seconds for one pass at a steady 40 pt/s, whatever the content length.
    static func duration(contentWidth: CGFloat, gap: CGFloat, speed: CGFloat = 40) -> Double {
        Double(max(contentWidth + gap, 1) / speed)
    }

    /// Scroll only when the line does not fit and motion is allowed.
    static func scrolls(contentWidth: CGFloat, containerWidth: CGFloat, reduceMotion: Bool) -> Bool {
        !reduceMotion && containerWidth > 0 && contentWidth > containerWidth
    }
}

/// Bounded append shared by both stores.
func appendActivity(_ entries: inout [ActivityEntry], _ text: String, source: String, cap: Int, next id: inout Int) {
    id += 1
    entries.append(ActivityEntry(id: id, date: Date(), text: text, source: source))
    if entries.count > cap { entries.removeFirst(entries.count - cap) }
}

private struct TickerWidthKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = max(value, nextValue()) }
}

struct ActivityTicker: View {
    let items: [ActivityEntry]
    let history: [ActivityEntry]

    static let height: CGFloat = 26
    static let gap: CGFloat = 26

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var showLog = false
    @State private var contentWidth: CGFloat = 0
    @State private var containerWidth: CGFloat = 0
    @State private var offset: CGFloat = 0
    @State private var cycle = 0

    private var scrolls: Bool {
        TickerFacts.scrolls(contentWidth: contentWidth, containerWidth: containerWidth, reduceMotion: reduceMotion)
    }

    var body: some View {
        Button { showLog.toggle() } label: {
            GeometryReader { geo in
                Group {
                    if items.isEmpty {
                        Text(TickerFacts.idleText).foregroundStyle(.tertiary)
                    } else if reduceMotion {
                        entry(items[0]).lineLimit(1).truncationMode(.tail)
                    } else {
                        HStack(spacing: Self.gap) {
                            row.background(GeometryReader { g in
                                Color.clear.preference(key: TickerWidthKey.self, value: g.size.width)
                            })
                            if scrolls { row }
                        }
                        .fixedSize()
                        .offset(x: offset)
                        .id(cycle)
                    }
                }
                .font(.system(.caption, design: .monospaced))
                .padding(.horizontal, 12)
                .frame(width: geo.size.width, height: Self.height, alignment: .leading)
                .clipped()
                .onAppear { containerWidth = geo.size.width - 24; restart() }
                .onChange(of: geo.size.width) { width in containerWidth = width - 24; restart() }
            }
            .frame(height: Self.height)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .background(Color(nsColor: .underPageBackgroundColor))
        .overlay(alignment: .top) { Rectangle().fill(Color.gray.opacity(0.25)).frame(height: 1) }
        .onPreferenceChange(TickerWidthKey.self) { width in
            if abs(width - contentWidth) > 0.5 { contentWidth = width; restart() }
        }
        .onChange(of: items.map(\.id)) { _ in restart() }
        .help(items.first.map { "Latest: \(TickerFacts.line($0))\nClick for the full activity log." } ?? "Click for the full activity log.")
        .accessibilityLabel(items.first.map { "Activity: \(TickerFacts.display($0.text))" } ?? TickerFacts.idleText)
        .popover(isPresented: $showLog, arrowEdge: .top) {
            ActivityLogPopover(history: history)
        }
    }

    private var row: some View {
        HStack(spacing: Self.gap) {
            ForEach(items) { item in entry(item) }
        }
        .lineLimit(1)
        .fixedSize()
    }

    private func entry(_ item: ActivityEntry) -> some View {
        HStack(spacing: 6) {
            Text(TickerFacts.time(item.date)).foregroundStyle(Color.accentColor)
            Text(TickerFacts.display(item.text)).foregroundStyle(.secondary)
        }
    }

    private func restart() {
        var still = Transaction()
        still.disablesAnimations = true
        withTransaction(still) { offset = 0 }
        cycle += 1
        guard scrolls else { return }
        let distance = contentWidth + Self.gap
        let duration = TickerFacts.duration(contentWidth: contentWidth, gap: Self.gap)
        DispatchQueue.main.async {
            withAnimation(.linear(duration: duration).repeatForever(autoreverses: false)) { offset = -distance }
        }
    }
}

struct ActivityLogPopover: View {
    let history: [ActivityEntry]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Activity log").font(.headline)
                Text("\(history.count) lines").font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button("Copy") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(history.map(TickerFacts.line).joined(separator: "\n"), forType: .string)
                }
                .disabled(history.isEmpty)
            }
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 1) {
                        if history.isEmpty {
                            Text(TickerFacts.idleText).foregroundStyle(.tertiary)
                        }
                        ForEach(history) { item in
                            HStack(alignment: .firstTextBaseline, spacing: 8) {
                                Text(TickerFacts.time(item.date)).foregroundStyle(Color.accentColor)
                                Text(item.source).foregroundStyle(.tertiary)
                                Text(item.text).frame(maxWidth: .infinity, alignment: .leading)
                            }
                            .id(item.id)
                        }
                    }
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                    .padding(8)
                }
                .background(Color(nsColor: .textBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 6))
                .onAppear { if let last = history.last { proxy.scrollTo(last.id, anchor: .bottom) } }
            }
        }
        .padding(12)
        .frame(width: 720, height: 360)
    }
}
