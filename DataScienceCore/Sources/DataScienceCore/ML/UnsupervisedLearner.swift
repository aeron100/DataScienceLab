import Foundation
import Accelerate

/// Pure-Swift unsupervised learning algorithms.
/// All methods are synchronous and compute-heavy — call from a background task.
public enum UnsupervisedLearner {

    // MARK: - Public entry point

    public static func run(
        data: [[Double]],
        features: [String],
        algorithm: UnsupervisedAlgorithm,
        hyperparams: UnsupervisedHyperparams = UnsupervisedHyperparams()
    ) -> UnsupervisedResult {
        switch algorithm {
        case .kMeans:
            return .clustering(runKMeans(data: data, features: features, hp: hyperparams))
        case .dbscan:
            return .clustering(runDBSCAN(data: data, features: features, hp: hyperparams))
        case .pca:
            return .dimensionality(runPCA(data: data, features: features, hp: hyperparams))
        case .isolationForest:
            return .anomaly(runIsolationForest(data: data, features: features, hp: hyperparams))
        }
    }

    // MARK: - K-Means

    private static func runKMeans(
        data: [[Double]], features: [String], hp: UnsupervisedHyperparams
    ) -> ClusterResult {
        let n = data.count
        let d = data.first?.count ?? 0
        let k = min(max(1, hp.kClusters), n)
        guard n > 0, d > 0 else {
            return empty(.kMeans, features: features)
        }

        let (norm, _, _) = standardize(data)
        var centroids    = kMeansPlusPlus(data: norm, k: k)
        var labels       = [Int](repeating: 0, count: n)

        for _ in 0..<hp.maxIter {
            let old = labels
            // Assign each point to nearest centroid
            for i in 0..<n {
                var best = 0; var bestDist = Double.infinity
                for j in 0..<k {
                    let dist = squaredDist(norm[i], centroids[j])
                    if dist < bestDist { bestDist = dist; best = j }
                }
                labels[i] = best
            }
            if labels == old { break }
            // Update centroids
            for j in 0..<k {
                let members = (0..<n).filter { labels[$0] == j }.map { norm[$0] }
                guard !members.isEmpty else { continue }
                centroids[j] = (0..<d).map { dim in
                    members.map { $0[dim] }.reduce(0, +) / Double(members.count)
                }
            }
        }

        let inertia  = (0..<n).reduce(0.0) { $0 + squaredDist(norm[$1], centroids[labels[$1]]) }
        let silScore = n <= 2000 ? silhouette(data: norm, labels: labels, k: k) : nil
        let (pts, xl, yl) = project2D(data: norm, labels: labels, features: features)

        return ClusterResult(
            algorithm: .kMeans, labels: labels, nClusters: k,
            inertia: inertia, silhouette: silScore, noiseCount: 0,
            points2D: pts, xLabel: xl, yLabel: yl, features: features)
    }

    private static func kMeansPlusPlus(data: [[Double]], k: Int) -> [[Double]] {
        var rng = SystemRandomNumberGenerator()
        var centroids = [data[Int.random(in: 0..<data.count, using: &rng)]]
        while centroids.count < k {
            let dists = data.map { pt in centroids.map { squaredDist(pt, $0) }.min() ?? 0.0 }
            let total = dists.reduce(0, +)
            if total <= 0 {
                centroids.append(data[Int.random(in: 0..<data.count, using: &rng)])
                continue
            }
            var r = Double.random(in: 0..<total, using: &rng)
            var appended = false
            for (i, d) in dists.enumerated() {
                r -= d
                if r <= 0 { centroids.append(data[i]); appended = true; break }
            }
            if !appended { centroids.append(data.last!) }
        }
        return centroids
    }

    // MARK: - DBSCAN (O(n²) — subsampled to dbscanRowCap for large datasets)

    private static let dbscanRowCap = 50_000

    private static func runDBSCAN(
        data: [[Double]], features: [String], hp: UnsupervisedHyperparams
    ) -> ClusterResult {
        let n = data.count
        guard n > 0 else { return empty(.dbscan, features: features) }

        // Subsample for large datasets to keep O(n²) tractable
        let (workingData, wasSampled): ([[Double]], Bool)
        if n > Self.dbscanRowCap {
            var rng = SystemRandomNumberGenerator()
            let indices = (0..<n).shuffled(using: &rng).prefix(Self.dbscanRowCap).map { $0 }
            workingData = indices.map { data[$0] }
            wasSampled = true
        } else {
            workingData = data
            wasSampled = false
        }
        let wn = workingData.count

        let (norm, _, _) = standardize(workingData)
        let eps2    = hp.eps * hp.eps
        let minPts  = hp.minPts
        // -1 = unassigned, -2 = noise
        var labels  = [Int](repeating: -1, count: wn)
        var cluster = 0

        for i in 0..<wn {
            guard labels[i] == -1 else { continue }
            let nbrs = rangeQuery(data: norm, idx: i, eps2: eps2)
            if nbrs.count < minPts { labels[i] = -2; continue }

            labels[i] = cluster
            var seeds = Set(nbrs); seeds.remove(i)
            while !seeds.isEmpty {
                let q = seeds.removeFirst()
                if labels[q] == -2 { labels[q] = cluster }
                guard labels[q] == -1 else { continue }
                labels[q] = cluster
                let qNbrs = rangeQuery(data: norm, idx: q, eps2: eps2)
                if qNbrs.count >= minPts { seeds.formUnion(qNbrs) }
            }
            cluster += 1
        }

        // Remap noise marker -2 → -1
        for i in 0..<wn { if labels[i] == -2 { labels[i] = -1 } }

        let noise    = labels.filter { $0 == -1 }.count
        let silScore = wn <= 2000 && cluster > 1 ? silhouette(data: norm, labels: labels, k: cluster) : nil
        let (pts, xl, yl) = project2D(data: norm, labels: labels, features: features)

        let note = wasSampled
            ? "Ran on \(Self.dbscanRowCap.formatted()) of \(n.formatted()) rows (subsampled)"
            : nil

        return ClusterResult(
            algorithm: .dbscan, labels: labels, nClusters: cluster,
            inertia: nil, silhouette: silScore, noiseCount: noise,
            points2D: pts, xLabel: xl, yLabel: yl, features: features,
            samplingNote: note)
    }

    private static func rangeQuery(data: [[Double]], idx: Int, eps2: Double) -> [Int] {
        let pt = data[idx]
        return (0..<data.count).filter { squaredDist(pt, data[$0]) <= eps2 }
    }

    // MARK: - PCA (LAPACK dsyev_ symmetric eigendecomposition)

    private static func runPCA(
        data: [[Double]], features: [String], hp: UnsupervisedHyperparams
    ) -> PCAResult {
        let n = data.count
        let d = data.first?.count ?? 0
        guard n > 1, d > 0 else {
            return PCAResult(features: features, components: [], projectedPoints: [], nComponents: 0)
        }

        let (norm, _, _) = standardize(data)

        // Build covariance matrix in column-major order for LAPACK
        var cov = [Double](repeating: 0, count: d * d)
        for k in 0..<n {
            for i in 0..<d {
                for j in 0..<d {
                    cov[j * d + i] += norm[k][i] * norm[k][j]
                }
            }
        }
        let scale = 1.0 / Double(n - 1)
        for idx in 0..<cov.count { cov[idx] *= scale }

        // Eigendecomposition of symmetric matrix — eigenvalues returned ascending,
        // eigenvectors stored as columns in `cov` after the call.
        var jobz: Int8  = Int8(UInt8(ascii: "V"))
        var uplo: Int8  = Int8(UInt8(ascii: "U"))
        var nn          = Int32(d)
        var lda         = Int32(d)
        var info: Int32 = 0
        var lwork: Int32 = -1
        var eigenvalues  = [Double](repeating: 0, count: d)
        var wkopt        = 0.0

        // Query workspace size
        dsyev_(&jobz, &uplo, &nn, &cov, &lda, &eigenvalues, &wkopt, &lwork, &info)
        lwork = Int32(wkopt)
        var work = [Double](repeating: 0, count: Int(lwork))
        // Compute
        dsyev_(&jobz, &uplo, &nn, &cov, &lda, &eigenvalues, &work, &lwork, &info)

        guard info == 0 else {
            return PCAResult(features: features, components: [], projectedPoints: [], nComponents: 0)
        }

        let totalVar = max(eigenvalues.reduce(0, +), 1e-10)
        let nComp    = min(max(1, hp.nComponents), d)
        var components: [PCAComponent] = []
        var cumulative = 0.0

        for c in 0..<nComp {
            let evIdx = d - 1 - c                    // eigenvalues are ascending → reverse
            let ev    = max(eigenvalues[evIdx], 0.0)
            let expl  = ev / totalVar
            cumulative += expl
            // Eigenvector c: column evIdx of cov (column-major)
            let loadings: [(String, Double)] = features.enumerated().map { fi, fname in
                (fname, cov[evIdx * d + fi])
            }
            components.append(PCAComponent(
                id: c, explainedVariance: expl, cumulative: cumulative, loadings: loadings))
        }

        // Project all points onto PC1 and PC2
        let ev0 = evector(cov, col: d - 1, d: d)
        let ev1 = d >= 2 ? evector(cov, col: d - 2, d: d) : [Double](repeating: 0, count: d)
        let pts = norm.enumerated().map { (i, row) -> ClusterPoint in
            ClusterPoint(id: i, x: dot(row, ev0), y: nComp >= 2 ? dot(row, ev1) : 0, cluster: 0)
        }

        return PCAResult(features: features, components: components,
                         projectedPoints: pts, nComponents: nComp)
    }

    private static func evector(_ flat: [Double], col: Int, d: Int) -> [Double] {
        (0..<d).map { flat[col * d + $0] }
    }

    // MARK: - Isolation Forest

    private indirect enum INode: Sendable {
        case leaf(size: Int)
        case split(feature: Int, value: Double, left: INode, right: INode)
    }

    private static func runIsolationForest(
        data: [[Double]], features: [String], hp: UnsupervisedHyperparams
    ) -> AnomalyResult {
        let n = data.count
        guard n > 0 else {
            return AnomalyResult(algorithm: .isolationForest, scores: [], anomalyCount: 0,
                                 threshold: 0, points: [], features: features)
        }

        let (norm, _, _) = standardize(data)
        let subSize  = min(256, n)
        let maxDepth = Int(ceil(log2(Double(subSize))))
        var rng      = SystemRandomNumberGenerator()
        var pathSums = [Double](repeating: 0, count: n)

        for _ in 0..<hp.nTrees {
            let idxs   = (0..<n).shuffled(using: &rng).prefix(subSize).map { $0 }
            let sample = idxs.map { norm[$0] }
            let tree   = buildITree(data: sample, depth: 0, maxDepth: maxDepth, rng: &rng)
            for i in 0..<n {
                pathSums[i] += pathLength(point: norm[i], node: tree, depth: 0)
            }
        }

        let cn = avgPathLength(n: subSize)
        let scores = pathSums.map { pow(2.0, -($0 / Double(hp.nTrees)) / cn) }

        let sorted    = scores.sorted()
        let threshIdx = max(0, min(n - 1, Int(Double(n) * (1.0 - hp.contamination))))
        let threshold = sorted[threshIdx]

        let pts = scores.enumerated().map { i, s in
            AnomalyPoint(id: i, score: s, isAnomaly: s >= threshold)
        }
        let anomalyCount = pts.filter { $0.isAnomaly }.count

        return AnomalyResult(algorithm: .isolationForest, scores: scores,
                             anomalyCount: anomalyCount, threshold: threshold,
                             points: pts, features: features)
    }

    private static func buildITree(
        data: [[Double]], depth: Int, maxDepth: Int, rng: inout SystemRandomNumberGenerator
    ) -> INode {
        let n = data.count
        let d = data.first?.count ?? 0
        if n <= 1 || depth >= maxDepth || d == 0 { return .leaf(size: n) }

        let feat = Int.random(in: 0..<d, using: &rng)
        let vals = data.map { $0[feat] }
        guard let minV = vals.min(), let maxV = vals.max(), minV < maxV else {
            return .leaf(size: n)
        }
        let split = Double.random(in: minV..<maxV, using: &rng)
        let left  = data.filter { $0[feat] <  split }
        let right = data.filter { $0[feat] >= split }
        return .split(
            feature: feat, value: split,
            left:  buildITree(data: left,  depth: depth + 1, maxDepth: maxDepth, rng: &rng),
            right: buildITree(data: right, depth: depth + 1, maxDepth: maxDepth, rng: &rng))
    }

    private static func pathLength(point: [Double], node: INode, depth: Int) -> Double {
        switch node {
        case .leaf(let size):
            return Double(depth) + avgPathLength(n: size)
        case .split(let feat, let val, let left, let right):
            if point[feat] < val {
                return pathLength(point: point, node: left,  depth: depth + 1)
            } else {
                return pathLength(point: point, node: right, depth: depth + 1)
            }
        }
    }

    private static func avgPathLength(n: Int) -> Double {
        if n <= 1 { return 0 }
        if n == 2 { return 1 }
        let h = log(Double(n - 1)) + 0.5772156649   // + Euler-Mascheroni constant
        return 2.0 * h - 2.0 * Double(n - 1) / Double(n)
    }

    // MARK: - Silhouette score (O(n²))

    private static func silhouette(data: [[Double]], labels: [Int], k: Int) -> Double? {
        let n = data.count
        guard n > 1 else { return nil }
        var scores: [Double] = []
        for i in 0..<n {
            let ci = labels[i]
            guard ci >= 0 else { continue }
            let same  = (0..<n).filter { $0 != i && labels[$0] == ci }
            let a: Double = same.isEmpty ? 0 :
                same.map { sqrt(squaredDist(data[i], data[$0])) }.reduce(0, +) / Double(same.count)
            var minB = Double.infinity
            for c in 0..<k where c != ci {
                let other = (0..<n).filter { labels[$0] == c }
                guard !other.isEmpty else { continue }
                let b = other.map { sqrt(squaredDist(data[i], data[$0])) }.reduce(0, +) / Double(other.count)
                if b < minB { minB = b }
            }
            guard minB < Double.infinity else { continue }
            scores.append((minB - a) / max(a, minB))
        }
        return scores.isEmpty ? nil : scores.reduce(0, +) / Double(scores.count)
    }

    // MARK: - 2-D projection for scatter plots

    private static func project2D(
        data: [[Double]], labels: [Int], features: [String]
    ) -> ([ClusterPoint], String, String) {
        let d  = data.first?.count ?? 0
        guard d > 0 else { return ([], "", "") }
        let xl = features.count > 0 ? features[0] : "Dim 1"
        let yl = features.count > 1 ? features[1] : "Index"
        let pts = data.enumerated().map { (i, row) -> ClusterPoint in
            ClusterPoint(id: i, x: row[0], y: d > 1 ? row[1] : Double(i), cluster: labels[i])
        }
        return (pts, xl, yl)
    }

    // MARK: - Standardise (z-score, per column)

    public static func standardize(_ data: [[Double]]) -> ([[Double]], [Double], [Double]) {
        let n = data.count
        let d = data.first?.count ?? 0
        var means = [Double](repeating: 0, count: d)
        var stds  = [Double](repeating: 1, count: d)
        guard n > 0 else { return (data, means, stds) }
        for j in 0..<d {
            let vals = data.map { $0[j] }
            var m = 0.0
            vDSP_meanvD(vals, 1, &m, vDSP_Length(n))
            means[j] = m
            let centred = vals.map { $0 - m }
            var ms = 0.0
            vDSP_measqvD(centred, 1, &ms, vDSP_Length(n))
            stds[j] = max(1e-10, sqrt(ms))
        }
        let normed = data.map { row in (0..<d).map { j in (row[j] - means[j]) / stds[j] } }
        return (normed, means, stds)
    }

    // MARK: - Math helpers

    private static func squaredDist(_ a: [Double], _ b: [Double]) -> Double {
        zip(a, b).reduce(0) { acc, pair in acc + (pair.0 - pair.1) * (pair.0 - pair.1) }
    }

    private static func dot(_ a: [Double], _ b: [Double]) -> Double {
        zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
    }

    // MARK: - Empty result helpers

    private static func empty(_ alg: UnsupervisedAlgorithm, features: [String]) -> ClusterResult {
        ClusterResult(algorithm: alg, labels: [], nClusters: 0, inertia: nil,
                      silhouette: nil, noiseCount: 0, points2D: [],
                      xLabel: "", yLabel: "", features: features)
    }
}
