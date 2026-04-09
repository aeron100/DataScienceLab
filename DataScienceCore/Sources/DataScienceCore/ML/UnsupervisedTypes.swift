import Foundation

// MARK: - Algorithm

public enum UnsupervisedAlgorithm: String, CaseIterable, Sendable, Identifiable {
    case kMeans          = "kmeans"
    case dbscan          = "dbscan"
    case pca             = "pca"
    case isolationForest = "isolation_forest"

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .kMeans:          "K-Means"
        case .dbscan:          "DBSCAN"
        case .pca:             "PCA"
        case .isolationForest: "Isolation Forest"
        }
    }

    public var subtitle: String {
        switch self {
        case .kMeans:          "Partition data into k clusters"
        case .dbscan:          "Density-based clustering with noise detection"
        case .pca:             "Reduce dimensionality; find directions of maximum variance"
        case .isolationForest: "Detect anomalies by random isolation"
        }
    }

    public var isClustering:     Bool { self == .kMeans || self == .dbscan }
    public var isDimensionality: Bool { self == .pca }
    public var isAnomaly:        Bool { self == .isolationForest }
}

// MARK: - Hyperparameters

public struct UnsupervisedHyperparams: Sendable {
    // K-Means
    public var kClusters:     Int    = 3
    public var maxIter:       Int    = 100
    // DBSCAN
    public var eps:           Double = 0.5
    public var minPts:        Int    = 5
    // PCA
    public var nComponents:   Int    = 2
    // Isolation Forest
    public var nTrees:        Int    = 100
    public var contamination: Double = 0.1

    public init() {}
}

// MARK: - 2-D point (used for scatter plots in all algorithms)

public struct ClusterPoint: Identifiable, Sendable {
    public let id:      Int
    public let x:       Double
    public let y:       Double
    public let cluster: Int    // -1 = noise (DBSCAN)

    public init(id: Int, x: Double, y: Double, cluster: Int) {
        self.id = id; self.x = x; self.y = y; self.cluster = cluster
    }
}

// MARK: - Cluster result (KMeans / DBSCAN)

public struct ClusterResult: Sendable {
    public let algorithm:    UnsupervisedAlgorithm
    public let labels:       [Int]
    public let nClusters:    Int
    public let inertia:      Double?   // KMeans only
    public let silhouette:   Double?
    public let noiseCount:   Int       // DBSCAN only
    public let points2D:     [ClusterPoint]
    public let xLabel:       String
    public let yLabel:       String
    public let features:     [String]
    public let samplingNote: String?   // set when input was subsampled for performance

    public init(
        algorithm: UnsupervisedAlgorithm, labels: [Int], nClusters: Int,
        inertia: Double?, silhouette: Double?, noiseCount: Int,
        points2D: [ClusterPoint], xLabel: String, yLabel: String, features: [String],
        samplingNote: String? = nil
    ) {
        self.algorithm    = algorithm
        self.labels       = labels
        self.nClusters    = nClusters
        self.inertia      = inertia
        self.silhouette   = silhouette
        self.noiseCount   = noiseCount
        self.points2D     = points2D
        self.xLabel       = xLabel
        self.yLabel       = yLabel
        self.features     = features
        self.samplingNote = samplingNote
    }
}

// MARK: - PCA component

public struct PCAComponent: Identifiable, Sendable {
    public let id:                Int
    public let explainedVariance: Double
    public let cumulative:        Double
    public let loadings:          [(feature: String, value: Double)]

    public init(id: Int, explainedVariance: Double, cumulative: Double,
                loadings: [(feature: String, value: Double)]) {
        self.id                = id
        self.explainedVariance = explainedVariance
        self.cumulative        = cumulative
        self.loadings          = loadings
    }
}

// MARK: - PCA result

public struct PCAResult: Sendable {
    public let features:         [String]
    public let components:       [PCAComponent]
    public let projectedPoints:  [ClusterPoint]   // PC1 vs PC2 scatter
    public let nComponents:      Int

    public init(features: [String], components: [PCAComponent],
                projectedPoints: [ClusterPoint], nComponents: Int) {
        self.features        = features
        self.components      = components
        self.projectedPoints = projectedPoints
        self.nComponents     = nComponents
    }
}

// MARK: - Anomaly point

public struct AnomalyPoint: Identifiable, Sendable {
    public let id:        Int
    public let score:     Double
    public let isAnomaly: Bool

    public init(id: Int, score: Double, isAnomaly: Bool) {
        self.id = id; self.score = score; self.isAnomaly = isAnomaly
    }
}

// MARK: - Anomaly result

public struct AnomalyResult: Sendable {
    public let algorithm:    UnsupervisedAlgorithm
    public let scores:       [Double]
    public let anomalyCount: Int
    public let threshold:    Double
    public let points:       [AnomalyPoint]
    public let features:     [String]

    public init(algorithm: UnsupervisedAlgorithm, scores: [Double], anomalyCount: Int,
                threshold: Double, points: [AnomalyPoint], features: [String]) {
        self.algorithm    = algorithm
        self.scores       = scores
        self.anomalyCount = anomalyCount
        self.threshold    = threshold
        self.points       = points
        self.features     = features
    }
}

// MARK: - Unified result

public enum UnsupervisedResult: Sendable {
    case clustering(ClusterResult)
    case dimensionality(PCAResult)
    case anomaly(AnomalyResult)

    public var algorithm: UnsupervisedAlgorithm {
        switch self {
        case .clustering(let r):    r.algorithm
        case .dimensionality:       .pca
        case .anomaly(let r):       r.algorithm
        }
    }
}
