import SwiftUI
import Charts
import DataScienceCore

struct UnsupervisedPanel: View {

    @Environment(AnalysisSession.self) private var session

    @State private var algorithm        = UnsupervisedAlgorithm.kMeans
    @State private var selectedFeatures = Set<String>()
    @State private var hp               = UnsupervisedHyperparams()
    @State private var isRunning        = false
    @State private var errorMessage: String? = nil

    // MARK: - Numeric columns from the active frame

    private var numericColumns: [String] {
        guard let df = session.activeFrame else { return [] }
        return df.columns.compactMap { col -> String? in
            let col = df[col.name]
            for i in 0..<min(col.count, 20) {
                if let v = col[i] {
                    let s = "\(v)"
                    if Double(s) != nil { return col.name }
                    return nil
                }
            }
            return nil
        }
    }

    // MARK: - Body

    var body: some View {
        Group {
            if session.activeFrame == nil {
                PlaceholderView(
                    title: "No Data Loaded",
                    subtitle: "Load a dataset in the Data tab first.",
                    icon: "circle.grid.3x3", phase: "Phase 5")
            } else {
                HSplitView {
                    configSidebar
                        .frame(minWidth: 240, idealWidth: 260, maxWidth: 300)
                    resultArea
                }
            }
        }
        .onAppear { seedFeatures() }
        .onChange(of: algorithm) { _, _ in seedFeatures() }
    }

    // MARK: - Config sidebar

    private var configSidebar: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {

                sectionHeader("Algorithm")
                Picker("", selection: $algorithm) {
                    ForEach(UnsupervisedAlgorithm.allCases) { alg in
                        Text(alg.displayName).tag(alg)
                    }
                }
                .pickerStyle(.radioGroup)
                .labelsHidden()

                Text(algorithm.subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                Divider()

                sectionHeader("Features")
                if numericColumns.isEmpty {
                    Text("No numeric columns found.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(numericColumns, id: \.self) { col in
                        Toggle(col, isOn: Binding(
                            get: { selectedFeatures.contains(col) },
                            set: { if $0 { selectedFeatures.insert(col) } else { selectedFeatures.remove(col) } }
                        ))
                        .toggleStyle(.checkbox)
                        .font(.system(size: 12))
                    }
                }

                Divider()

                sectionHeader("Hyperparameters")
                hyperpamsView

                Divider()

                if let err = errorMessage {
                    Text(err)
                        .foregroundStyle(.red)
                        .font(.caption)
                        .fixedSize(horizontal: false, vertical: true)
                }

                Button(action: runAlgorithm) {
                    HStack {
                        if isRunning {
                            ProgressView().scaleEffect(0.6).frame(width: 14, height: 14)
                        }
                        Text(isRunning ? "Running…" : "Run \(algorithm.displayName)")
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(isRunning || selectedFeatures.count < (algorithm == .pca ? 2 : 1))

                Spacer()
            }
            .padding(12)
        }
    }

    // MARK: - Hyperparameter controls (change per algorithm)

    @ViewBuilder
    private var hyperpamsView: some View {
        switch algorithm {
        case .kMeans:
            stepper("Clusters (k)", value: $hp.kClusters, range: 2...20)
            stepper("Max Iterations", value: $hp.maxIter, range: 10...500)
        case .dbscan:
            sliderRow("Epsilon (ε)", value: $hp.eps, in: 0.1...5.0, format: "%.2f")
            stepper("Min Points", value: $hp.minPts, range: 2...30)
        case .pca:
            stepper("Components", value: $hp.nComponents, range: 2...min(20, numericColumns.count))
        case .isolationForest:
            stepper("Trees", value: $hp.nTrees, range: 10...500)
            sliderRow("Contamination", value: $hp.contamination, in: 0.01...0.5, format: "%.2f")
        }
    }

    // MARK: - Result area

    @ViewBuilder
    private var resultArea: some View {
        if let result = session.unsupervisedResult {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    switch result {
                    case .clustering(let r):
                        clusteringResultView(r)
                    case .dimensionality(let r):
                        pcaResultView(r)
                    case .anomaly(let r):
                        anomalyResultView(r)
                    }
                }
                .padding(16)
            }
        } else {
            VStack(spacing: 8) {
                Image(systemName: "circle.grid.3x3")
                    .font(.system(size: 48))
                    .foregroundStyle(.quaternary)
                Text("Select features and run an algorithm")
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    // MARK: - Clustering results (KMeans / DBSCAN)

    private func clusteringResultView(_ r: ClusterResult) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            resultTitle(r.algorithm.displayName, count: r.labels.count)

            // Metrics row
            HStack(spacing: 16) {
                metricCard("Clusters", "\(r.nClusters)")
                if let inertia = r.inertia {
                    metricCard("Inertia", String(format: "%.2f", inertia))
                }
                if r.noiseCount > 0 {
                    metricCard("Noise", "\(r.noiseCount)")
                }
                if let sil = r.silhouette {
                    metricCard("Silhouette", String(format: "%.3f", sil))
                }
            }

            if let note = r.samplingNote {
                Label(note, systemImage: "info.circle")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            // Scatter plot
            if !r.points2D.isEmpty {
                GroupBox("Cluster Scatter — \(r.xLabel) vs \(r.yLabel)") {
                    clusterScatter(r.points2D, nClusters: r.nClusters)
                        .frame(height: 320)
                        .padding(8)
                }
            }

            // Cluster sizes table
            GroupBox("Cluster Sizes") {
                clusterSizeTable(labels: r.labels, nClusters: r.nClusters)
            }
        }
    }

    private func clusterScatter(_ pts: [ClusterPoint], nClusters: Int) -> some View {
        Chart(pts) { pt in
            PointMark(
                x: .value("X", pt.x),
                y: .value("Y", pt.y)
            )
            .foregroundStyle(by: .value("Cluster", pt.cluster == -1 ? "Noise" : "C\(pt.cluster)"))
            .symbolSize(30)
            .opacity(0.75)
        }
        .chartForegroundStyleScale(clusterPalette(n: nClusters))
    }

    private func clusterPalette(n: Int) -> KeyValuePairs<String, Color> {
        // Build a fixed palette — SwiftUI needs literal KeyValuePairs
        // We support up to 10 named clusters + noise
        // SwiftUI will only show entries that appear in the data
        return ["C0": Color("AccentColor"), "C1": .orange, "C2": .green, "C3": .red,
                "C4": .purple, "C5": .pink, "C6": .yellow, "C7": .cyan,
                "C8": .mint, "C9": .indigo, "Noise": .gray]
    }

    private func clusterSizeTable(labels: [Int], nClusters: Int) -> some View {
        let counts: [(label: String, count: Int)] = {
            var dict: [Int: Int] = [:]
            for l in labels { dict[l, default: 0] += 1 }
            let rows = dict.sorted { $0.key < $1.key }.map { k, v in
                (label: k == -1 ? "Noise" : "Cluster \(k)", count: v)
            }
            return rows
        }()
        return VStack(spacing: 0) {
            ForEach(counts, id: \.label) { row in
                HStack {
                    Text(row.label).font(.system(size: 12))
                    Spacer()
                    Text("\(row.count) rows")
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                .padding(.vertical, 4)
                .padding(.horizontal, 8)
                Divider()
            }
        }
    }

    // MARK: - PCA results

    private func pcaResultView(_ r: PCAResult) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            resultTitle("PCA", count: r.projectedPoints.count)

            // Metrics
            HStack(spacing: 16) {
                metricCard("Components", "\(r.nComponents)")
                if let first = r.components.first {
                    metricCard("PC1 Variance", String(format: "%.1f%%", first.explainedVariance * 100))
                }
                if r.components.count >= 2 {
                    let cum = r.components[1].cumulative
                    metricCard("PC1+PC2", String(format: "%.1f%%", cum * 100))
                }
            }

            // Scree plot
            if !r.components.isEmpty {
                GroupBox("Scree Plot — Explained Variance per Component") {
                    screePlot(r.components)
                        .frame(height: 200)
                        .padding(8)
                }
            }

            // PC1 vs PC2 scatter
            if !r.projectedPoints.isEmpty {
                GroupBox("Projection — PC1 vs PC2") {
                    Chart(r.projectedPoints) { pt in
                        PointMark(x: .value("PC1", pt.x), y: .value("PC2", pt.y))
                            .foregroundStyle(Color("AccentColor").opacity(0.6))
                            .symbolSize(25)
                    }
                    .frame(height: 280)
                    .padding(8)
                }
            }

            // Loadings table
            if !r.components.isEmpty {
                GroupBox("Component Loadings") {
                    loadingsTable(r.components)
                }
            }
        }
    }

    private func screePlot(_ components: [PCAComponent]) -> some View {
        Chart(components) { comp in
            BarMark(
                x: .value("PC", "PC\(comp.id + 1)"),
                y: .value("Variance", comp.explainedVariance * 100)
            )
            .foregroundStyle(Color("AccentColor").gradient)
            LineMark(
                x: .value("PC", "PC\(comp.id + 1)"),
                y: .value("Cumulative", comp.cumulative * 100)
            )
            .foregroundStyle(.orange)
            .lineStyle(StrokeStyle(lineWidth: 2))
            PointMark(
                x: .value("PC", "PC\(comp.id + 1)"),
                y: .value("Cumulative", comp.cumulative * 100)
            )
            .foregroundStyle(.orange)
        }
        .chartYAxisLabel("%")
    }

    private func loadingsTable(_ components: [PCAComponent]) -> some View {
        let features = components.first?.loadings.map { $0.feature } ?? []
        return VStack(spacing: 0) {
            // Header
            HStack {
                Text("Feature").font(.system(size: 11, weight: .semibold))
                    .frame(width: 120, alignment: .leading)
                ForEach(components.prefix(4)) { comp in
                    Text("PC\(comp.id + 1)")
                        .font(.system(size: 11, weight: .semibold))
                        .frame(maxWidth: .infinity)
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(.quaternary.opacity(0.5))
            Divider()
            ForEach(features, id: \.self) { feat in
                HStack {
                    Text(feat).font(.system(size: 11))
                        .frame(width: 120, alignment: .leading)
                        .lineLimit(1)
                    ForEach(components.prefix(4)) { comp in
                        if let loading = comp.loadings.first(where: { $0.feature == feat })?.value {
                            Text(String(format: "%.3f", loading))
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundStyle(loading > 0 ? Color("AccentColor") : Color.red)
                                .frame(maxWidth: .infinity)
                        }
                    }
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                Divider()
            }
        }
    }

    // MARK: - Anomaly detection results

    private func anomalyResultView(_ r: AnomalyResult) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            resultTitle("Isolation Forest", count: r.scores.count)

            HStack(spacing: 16) {
                metricCard("Anomalies", "\(r.anomalyCount)")
                metricCard("Normal", "\(r.scores.count - r.anomalyCount)")
                metricCard("Threshold", String(format: "%.3f", r.threshold))
                let pct = r.scores.isEmpty ? 0.0 : Double(r.anomalyCount) / Double(r.scores.count) * 100
                metricCard("Rate", String(format: "%.1f%%", pct))
            }

            if !r.points.isEmpty {
                GroupBox("Anomaly Scores (sorted by row index)") {
                    anomalyScoreChart(r.points, threshold: r.threshold)
                        .frame(height: 260)
                        .padding(8)
                }

                GroupBox("Top Anomalies") {
                    anomalyTable(r.points)
                }
            }
        }
    }

    private func anomalyScoreChart(_ pts: [AnomalyPoint], threshold: Double) -> some View {
        let displayPts = pts.count > 500 ? stride(from: 0, to: pts.count, by: pts.count / 500).map { pts[$0] } : pts
        return Chart {
            ForEach(displayPts) { pt in
                PointMark(x: .value("Row", pt.id), y: .value("Score", pt.score))
                    .foregroundStyle(pt.isAnomaly ? Color.red : Color("AccentColor").opacity(0.5))
                    .symbolSize(pt.isAnomaly ? 50 : 20)
            }
            RuleMark(y: .value("Threshold", threshold))
                .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [4]))
                .foregroundStyle(.orange)
                .annotation(position: .top, alignment: .leading) {
                    Text("threshold").font(.caption2).foregroundStyle(.orange)
                }
        }
        .chartXAxisLabel("Row Index")
        .chartYAxisLabel("Anomaly Score")
    }

    private func anomalyTable(_ pts: [AnomalyPoint]) -> some View {
        let top = pts.filter { $0.isAnomaly }.sorted { $0.score > $1.score }.prefix(20)
        return VStack(spacing: 0) {
            HStack {
                Text("Row").font(.system(size: 11, weight: .semibold)).frame(width: 60, alignment: .leading)
                Text("Score").font(.system(size: 11, weight: .semibold)).frame(maxWidth: .infinity, alignment: .trailing)
                Text("Status").font(.system(size: 11, weight: .semibold)).frame(width: 80, alignment: .trailing)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(.quaternary.opacity(0.5))
            Divider()
            ForEach(top) { pt in
                HStack {
                    Text("\(pt.id)").font(.system(size: 11, design: .monospaced))
                        .frame(width: 60, alignment: .leading)
                    Text(String(format: "%.4f", pt.score))
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(.red)
                        .frame(maxWidth: .infinity, alignment: .trailing)
                    Text("Anomaly")
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .background(.red)
                        .clipShape(Capsule())
                        .frame(width: 80, alignment: .trailing)
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                Divider()
            }
        }
    }

    // MARK: - Run

    private func runAlgorithm() {
        guard !isRunning, let df = session.activeFrame else { return }
        let features = selectedFeatures.filter { numericColumns.contains($0) }.sorted()
        guard !features.isEmpty else { errorMessage = "Select at least one numeric feature."; return }

        errorMessage = nil
        isRunning    = true

        let data: [[Double]] = {
            let cols = features.map { df[$0] }
            return (0..<df.rows.count).compactMap { i -> [Double]? in
                var row = [Double]()
                for col in cols {
                    guard let v = col[i] else { return nil }
                    switch v {
                    case let d as Double: row.append(d)
                    case let f as Float:  row.append(Double(f))
                    case let n as Int:    row.append(Double(n))
                    default:
                        if let d = Double("\(v)") { row.append(d) } else { return nil }
                    }
                }
                return row
            }
        }()

        let alg = algorithm
        let hpCopy = hp

        Task.detached(priority: .userInitiated) {
            let result = UnsupervisedLearner.run(
                data: data, features: features, algorithm: alg, hyperparams: hpCopy)
            await MainActor.run {
                session.unsupervisedResult = result
                isRunning = false
            }
        }
    }

    // MARK: - Seed feature defaults

    private func seedFeatures() {
        let cols = numericColumns
        selectedFeatures = Set(cols.prefix(min(algorithm == .pca ? 5 : 2, cols.count)))
    }

    // MARK: - Reusable sub-views

    private func sectionHeader(_ title: String) -> some View {
        Text(title.uppercased())
            .font(.system(size: 10, weight: .semibold))
            .foregroundStyle(.secondary)
    }

    private func resultTitle(_ name: String, count: Int) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(name).font(.title2.bold())
            Text("\(count) rows processed")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private func metricCard(_ label: String, _ value: String) -> some View {
        VStack(spacing: 2) {
            Text(value).font(.system(size: 18, weight: .semibold, design: .monospaced))
            Text(label).font(.caption).foregroundStyle(.secondary)
        }
        .frame(minWidth: 80)
        .padding(.vertical, 8)
        .padding(.horizontal, 12)
        .background(.quaternary.opacity(0.4))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private func stepper(_ label: String, value: Binding<Int>, range: ClosedRange<Int>) -> some View {
        HStack {
            Text(label).font(.system(size: 12))
            Spacer()
            Stepper("\(value.wrappedValue)", value: value, in: range)
                .labelsHidden()
            Text("\(value.wrappedValue)")
                .font(.system(size: 12, design: .monospaced))
                .frame(width: 28, alignment: .trailing)
        }
    }

    private func sliderRow(_ label: String, value: Binding<Double>, in range: ClosedRange<Double>, format: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label).font(.system(size: 12))
                Spacer()
                Text(String(format: format, value.wrappedValue))
                    .font(.system(size: 12, design: .monospaced))
            }
            Slider(value: value, in: range)
        }
    }
}

#Preview {
    UnsupervisedPanel()
        .environment(AnalysisSession())
        .frame(width: 1000, height: 700)
}
