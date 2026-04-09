import Foundation
import TabularData
import Accelerate

/// Handles missing value imputation. Mirrors Python's _step_missing().
public enum MissingValueHandler {

    public enum Strategy: String, CaseIterable, Sendable {
        case none       = "none"
        case dropRows   = "drop_rows"
        case mean       = "mean"
        case median     = "median"
        case mode       = "mode"
        case forwardFill = "ffill"
        case knn        = "knn"

        public var displayName: String {
            switch self {
            case .none:        return "None"
            case .dropRows:    return "Drop rows"
            case .mean:        return "Fill mean"
            case .median:      return "Fill median"
            case .mode:        return "Fill mode"
            case .forwardFill: return "Forward fill"
            case .knn:         return "KNN impute"
            }
        }
    }

    /// Apply a global strategy (with optional per-column overrides) to a DataFrame.
    /// Returns the modified DataFrame and a human-readable description of what changed.
    public static func apply(
        to df: DataFrame,
        strategy: Strategy,
        columnOverrides: [String: Strategy] = [:]
    ) -> (DataFrame, String?) {
        // col.missingCount is unreliable; also col[i] == nil fails for boxed Optional<T>.none.
        let nullCols = df.columns.filter { col in (0..<col.count).contains { isColNil(col[$0]) } }.map { $0.name }
        if nullCols.isEmpty { return (df, "No null values found — nothing to impute") }
        if strategy == .none && columnOverrides.isEmpty { return (df, nil) }

        switch strategy {
        case .dropRows:
            return applyDropRows(df)
        case .forwardFill:
            return applyForwardFill(df, columns: nullCols)
        case .knn:
            return applyKNN(df, columns: nullCols)
        default:
            return applyStatFill(df, globalStrategy: strategy, overrides: columnOverrides, nullCols: nullCols)
        }
    }

    // Swift boxing quirk: Double?.none boxed as Any? becomes Any?.some(Double?.none),
    // NOT Any?.none. So == nil comparison always returns false. Use Mirror instead.
    private static func isColNil(_ val: Any?) -> Bool {
        guard let v = val else { return true }
        let m = Mirror(reflecting: v)
        return m.displayStyle == .optional && m.children.isEmpty
    }

    // MARK: - Drop rows

    private static func applyDropRows(_ df: DataFrame) -> (DataFrame, String?) {
        let before = df.rows.count
        var result = DataFrame()
        let columnNames = df.columns.map { $0.name }

        // Identify rows that have at least one nil
        var keepRows = [Int]()
        for rowIdx in 0..<df.rows.count {
            let hasNil = df.columns.contains { col in isColNil(col[rowIdx]) }
            if !hasNil { keepRows.append(rowIdx) }
        }

        for col in df.columns {
            let vals = keepRows.map { col[$0] }
            result.append(column: rebuildAnyColumn(named: col.name, values: vals, source: col))
        }
        _ = columnNames  // suppress unused warning

        let dropped = before - result.rows.count
        return (result, "Dropped \(dropped.formatted()) rows containing null values")
    }

    // MARK: - Forward fill

    private static func applyForwardFill(_ df: DataFrame, columns: [String]) -> (DataFrame, String?) {
        var result = DataFrame()
        for col in df.columns {
            guard columns.contains(col.name) else {
                result.append(column: col)
                continue
            }
            var lastNonNil: Any? = nil
            var vals: [Any?] = []
            // Forward pass — use isColNil because boxed Optional<T>.none satisfies `if let`
            for i in 0..<col.count {
                let raw = col[i]
                if isColNil(raw) {
                    vals.append(lastNonNil)
                } else {
                    lastNonNil = raw
                    vals.append(raw)
                }
            }
            // Backward pass to fill leading nulls
            var lastNonNilBack: Any? = nil
            for i in stride(from: col.count - 1, through: 0, by: -1) {
                if !isColNil(vals[i]) { lastNonNilBack = vals[i] }
                else if let v = lastNonNilBack { vals[i] = v }
            }
            result.append(column: rebuildAnyColumn(named: col.name, values: vals, source: col))
        }
        return (result, "Forward-filled nulls in \(columns.count) column(s)")
    }

    // MARK: - Stat fill (mean / median / mode)

    private static func applyStatFill(
        _ df: DataFrame,
        globalStrategy: Strategy,
        overrides: [String: Strategy],
        nullCols: [String]
    ) -> (DataFrame, String?) {
        var result = DataFrame()
        var counts: [Strategy: Int] = [:]

        for col in df.columns {
            guard nullCols.contains(col.name) else {
                result.append(column: col)
                continue
            }
            let strat = overrides[col.name] ?? globalStrategy
            if strat == .none { result.append(column: col); continue }

            let nums = numericDoubles(from: col)
            let isNumeric = !nums.isEmpty

            if isNumeric && (strat == .mean || strat == .median) {
                let fillVal = strat == .mean ? vDSPMean(nums) : percentile(nums.sorted(), 0.5)
                let filled = fillDoubleNils(col: col, with: fillVal)
                result.append(column: filled)
            } else {
                let modeVal = mostFrequent(col: col)
                let filled = fillWithAny(col: col, value: modeVal)
                result.append(column: filled)
            }
            counts[strat, default: 0] += 1
        }

        if counts.isEmpty { return (result, nil) }
        let parts = counts.map { "\($0.key.displayName) × \($0.value)" }
        return (result, parts.joined(separator: ", "))
    }

    // MARK: - KNN imputation (numeric columns; mode fill for categorical)

    // Maximum number of complete rows used as the neighbor reference pool.
    // Caps worst-case inner-loop work to O(referencePoolSize) regardless of dataset size,
    // making KNN practical for datasets up to 200k+ rows.
    private static let knnReferencePoolSize = 15_000

    private static func applyKNN(_ df: DataFrame, columns: [String], k: Int = 5) -> (DataFrame, String?) {
        // Identify numeric vs categorical null columns
        var numNullCols: [String] = []
        var catNullCols: [String] = []
        for col in df.columns where columns.contains(col.name) {
            if numericDoubles(from: col).count > 0 { numNullCols.append(col.name) }
            else { catNullCols.append(col.name) }
        }

        // Build a matrix of numeric columns for distance computation
        let numericColNames = df.columns
            .filter { numericDoubles(from: $0).count > 0 }
            .map { $0.name }

        let n = df.rows.count
        let d = numericColNames.count
        guard d > 0 else {
            // No numeric columns — fall back to mode fill for everything
            return applyStatFill(df, globalStrategy: .mode, overrides: [:], nullCols: columns)
        }

        // Build n × d matrix (nil → use column mean as placeholder for distance)
        var colMeans: [Double] = []
        var matrix: [[Double]] = Array(repeating: Array(repeating: 0.0, count: d), count: n)
        for (j, cName) in numericColNames.enumerated() {
            let col = df[cName]
            let nonNilVals = (0..<col.count).compactMap { col[$0].flatMap { $0 as? Double } }
            var mean = nonNilVals.isEmpty ? 0.0 : 0.0
            if !nonNilVals.isEmpty { vDSP_meanvD(nonNilVals, 1, &mean, vDSP_Length(nonNilVals.count)) }
            colMeans.append(mean)
            for i in 0..<n {
                matrix[i][j] = col[i].flatMap { $0 as? Double } ?? mean
            }
        }

        // For large datasets, build a sampled reference pool of complete rows so neighbor
        // search is O(referencePoolSize) per null cell instead of O(n).
        let completeRows = (0..<n).filter { i in
            numNullCols.allSatisfy { cName in df[cName][i].flatMap { $0 as? Double } != nil }
        }
        let referenceIndices: [Int]
        if completeRows.count > Self.knnReferencePoolSize {
            var rng = SystemRandomNumberGenerator()
            referenceIndices = completeRows.shuffled(using: &rng).prefix(Self.knnReferencePoolSize).map { $0 }
        } else {
            referenceIndices = completeRows
        }

        // For each null cell in a numeric null column, find k nearest rows in the reference pool
        var result = df
        for cName in numNullCols {
            let col = df[cName]
            var newVals: [Double?] = (0..<n).map { col[$0].flatMap { $0 as? Double } }
            for i in 0..<n where newVals[i] == nil {
                var distances: [(Int, Double)] = []
                for j in referenceIndices where j != i {
                    guard let _ = col[j].flatMap({ $0 as? Double }) else { continue }
                    var dist = 0.0
                    for d_idx in 0..<d {
                        let diff = matrix[i][d_idx] - matrix[j][d_idx]
                        dist += diff * diff
                    }
                    distances.append((j, dist))
                }
                distances.sort { $0.1 < $1.1 }
                let neighbors = distances.prefix(k).compactMap { col[$0.0].flatMap { $0 as? Double } }
                if !neighbors.isEmpty {
                    newVals[i] = neighbors.reduce(0, +) / Double(neighbors.count)
                }
            }
            result.removeColumn(cName)
            result.append(column: Column<Double?>(name: cName, contents: newVals))
        }

        // Categorical: mode fill
        for cName in catNullCols {
            let col = result[cName]
            let modeVal = mostFrequent(col: col)
            result.removeColumn(cName)
            result.append(column: fillWithAny(col: col, value: modeVal))
        }

        var parts: [String] = []
        if !numNullCols.isEmpty { parts.append("KNN-imputed \(numNullCols.count) numeric col(s)") }
        if !catNullCols.isEmpty { parts.append("mode-filled \(catNullCols.count) categorical col(s)") }
        return (result, parts.joined(separator: "; "))
    }

    // MARK: - Helpers

    static func numericDoubles(from col: AnyColumn) -> [Double] {
        var vals: [Double] = []
        for i in 0..<col.count {
            guard let v = col[i] else { continue }
            switch v {
            case let d as Double: vals.append(d)
            case let f as Float:  vals.append(Double(f))
            case let n as Int:    vals.append(Double(n))
            case let n as Int64:  vals.append(Double(n))
            default: break
            }
        }
        return vals
    }

    private static func vDSPMean(_ vals: [Double]) -> Double {
        var m = 0.0
        vDSP_meanvD(vals, 1, &m, vDSP_Length(vals.count))
        return m
    }

    private static func percentile(_ sorted: [Double], _ p: Double) -> Double {
        let n = sorted.count
        guard n > 0 else { return 0 }
        if n == 1 { return sorted[0] }
        let idx = p * Double(n - 1)
        let lo = Int(idx); let hi = min(lo + 1, n - 1)
        return sorted[lo] * (1 - (idx - Double(lo))) + sorted[hi] * (idx - Double(lo))
    }

    // Peel Optional wrapper for clean string representation (TabularData boxing quirk).
    private static func colString(_ val: Any) -> String {
        let m = Mirror(reflecting: val)
        if m.displayStyle == .optional {
            guard let child = m.children.first?.value else { return "" }
            return "\(child)"
        }
        return "\(val)"
    }

    private static func mostFrequent(col: AnyColumn) -> Any? {
        var counts: [String: (count: Int, value: Any)] = [:]
        for i in 0..<col.count {
            let raw = col[i]
            if isColNil(raw) { continue }
            guard let v = raw else { continue }
            let key = colString(v)
            counts[key] = ((counts[key]?.count ?? 0) + 1, v)
        }
        return counts.max(by: { $0.value.count < $1.value.count })?.value.value
    }

    private static func fillDoubleNils(col: AnyColumn, with fill: Double) -> AnyColumn {
        let vals: [Double?] = (0..<col.count).map { i in
            let raw = col[i]
            if isColNil(raw) { return fill }
            if let v = raw {
                switch v {
                case let d as Double: return d
                case let f as Float:  return Double(f)
                case let n as Int:    return Double(n)
                case let n as Int64:  return Double(n)
                default:              return fill
                }
            }
            return fill
        }
        return Column<Double?>(name: col.name, contents: vals).eraseToAnyColumn()
    }

    private static func fillWithAny(col: AnyColumn, value: Any?) -> AnyColumn {
        guard let value else { return col }
        let fillStr = colString(value)
        let vals: [String?] = (0..<col.count).map { i in
            let raw = col[i]
            if isColNil(raw) { return fillStr }
            if let v = raw { return colString(v) }
            return fillStr
        }
        return Column<String?>(name: col.name, contents: vals).eraseToAnyColumn()
    }

    private static func rebuildAnyColumn(named name: String, values: [Any?], source: AnyColumn) -> AnyColumn {
        // Try Double first, then String
        let doubles: [Double?] = values.map { v in
            guard let v else { return nil }
            switch v {
            case let d as Double: return d
            case let f as Float:  return Double(f)
            case let n as Int:    return Double(n)
            case let n as Int64:  return Double(n)
            default:              return nil
            }
        }
        let doubleCount = doubles.compactMap { $0 }.count
        if Double(doubleCount) / Double(max(values.count, 1)) > 0.8 {
            return Column<Double?>(name: name, contents: doubles).eraseToAnyColumn()
        }
        let strings: [String?] = values.map { v in v.map { "\($0)" } }
        return Column<String?>(name: name, contents: strings).eraseToAnyColumn()
    }
}
