//! Topological Entropy: Information Content of Persistence Diagrams
//!
//! Implements various entropy measures that quantify the complexity
//! of the topological structure encoded in persistence diagrams.

use crate::topology::PersistenceDiagram;

/// Topological entropy measures
#[derive(Debug, Clone, Copy)]
pub struct TopologicalEntropy {
    /// Persistent entropy (Shannon)
    pub persistent_entropy: f64,
    /// Normalized entropy [0, 1]
    pub normalized_entropy: f64,
    /// Dimension-weighted entropy
    pub weighted_entropy: f64,
    /// Total lifetime (normalization constant)
    pub total_lifetime: f64,
}

impl TopologicalEntropy {
    /// Compute all entropy measures from a persistence diagram
    pub fn from_diagram(pd: &PersistenceDiagram) -> Self {
        let (persistent_entropy, total_lifetime) = compute_persistent_entropy(pd);
        let n_intervals = pd.intervals.iter()
            .filter(|i| !i.is_essential())
            .count();

        let normalized_entropy = if n_intervals > 1 {
            persistent_entropy / (n_intervals as f64).ln()
        } else {
            0.0
        };

        let weighted_entropy = compute_weighted_entropy(pd);

        Self {
            persistent_entropy,
            normalized_entropy,
            weighted_entropy,
            total_lifetime,
        }
    }
}

/// Compute persistent (Shannon) entropy from diagram
///
/// H_P = -Σᵢ pᵢ log(pᵢ)
///
/// where pᵢ = lᵢ / L is the normalized lifetime
pub fn compute_persistent_entropy(pd: &PersistenceDiagram) -> (f64, f64) {
    let lifetimes: Vec<f64> = pd.intervals.iter()
        .filter(|i| !i.is_essential())
        .map(|i| i.persistence())
        .collect();

    if lifetimes.is_empty() {
        return (0.0, 0.0);
    }

    let total: f64 = lifetimes.iter().sum();
    if total <= 0.0 {
        return (0.0, 0.0);
    }

    let mut entropy = 0.0;
    for l in &lifetimes {
        let p = l / total;
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }

    (entropy, total)
}

/// Compute dimension-weighted entropy
///
/// Higher-dimensional features (loops, voids) contribute more
/// to the weighted entropy, reflecting their role in information
/// integration.
pub fn compute_weighted_entropy(pd: &PersistenceDiagram) -> f64 {
    let max_dim = pd.max_dimension.max(2);
    let mut weighted_sum = 0.0;

    for d in 0..=max_dim {
        let intervals: Vec<f64> = pd.intervals.iter()
            .filter(|i| i.dimension == d && !i.is_essential())
            .map(|i| i.persistence())
            .collect();

        if intervals.is_empty() {
            continue;
        }

        let total: f64 = intervals.iter().sum();
        if total <= 0.0 {
            continue;
        }

        let mut dim_entropy = 0.0;
        for l in &intervals {
            let p = l / total;
            if p > 0.0 {
                dim_entropy -= p * p.ln();
            }
        }

        // Weight by dimension + 1 (so β₀ has weight 1, β₁ has weight 2, etc.)
        weighted_sum += (d + 1) as f64 * dim_entropy;
    }

    weighted_sum
}

/// Convenience function to compute entropy from diagram
pub fn compute_entropy(pd: &PersistenceDiagram) -> TopologicalEntropy {
    TopologicalEntropy::from_diagram(pd)
}

/// Compute entropy from edge statistics (approximation)
///
/// This provides a faster approximation when full persistent
/// homology is too expensive.
#[allow(dead_code)]
pub fn entropy_from_edges(distances: &ndarray::Array2<f64>, max_eps: f64) -> f64 {
    let n = distances.nrows();
    if n < 2 {
        return 0.0;
    }

    // Collect edge lengths
    let mut edges: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in i+1..n {
            let d = distances[[i, j]];
            if d <= max_eps && d > 0.0 {
                edges.push(d);
            }
        }
    }

    if edges.is_empty() {
        return 0.0;
    }

    // Bin edges and compute histogram entropy
    let n_bins = (edges.len() as f64).sqrt().ceil() as usize;
    let n_bins = n_bins.max(5).min(50);

    let min_edge = edges.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_edge = edges.iter().cloned().fold(0.0, f64::max);
    let bin_width = (max_edge - min_edge) / n_bins as f64;

    if bin_width <= 0.0 {
        return 0.0;
    }

    let mut bins = vec![0usize; n_bins];
    for e in &edges {
        let bin = ((e - min_edge) / bin_width).floor() as usize;
        let bin = bin.min(n_bins - 1);
        bins[bin] += 1;
    }

    // Shannon entropy of histogram
    let total = edges.len() as f64;
    let mut entropy = 0.0;
    for count in bins {
        if count > 0 {
            let p = count as f64 / total;
            entropy -= p * p.ln();
        }
    }

    entropy
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::{PersistenceInterval, PersistenceDiagram};

    #[test]
    fn test_entropy_uniform() {
        // Three intervals with equal lifetime
        let mut pd = PersistenceDiagram::new();
        pd.add(PersistenceInterval::new(0.0, 1.0, 0));
        pd.add(PersistenceInterval::new(0.0, 1.0, 0));
        pd.add(PersistenceInterval::new(0.0, 1.0, 0));

        let entropy = TopologicalEntropy::from_diagram(&pd);

        // Uniform distribution has entropy ln(3)
        let expected = 3.0_f64.ln();
        assert!((entropy.persistent_entropy - expected).abs() < 0.01);
    }

    #[test]
    fn test_entropy_single() {
        // Single interval has zero entropy
        let mut pd = PersistenceDiagram::new();
        pd.add(PersistenceInterval::new(0.0, 1.0, 0));

        let entropy = TopologicalEntropy::from_diagram(&pd);
        assert_eq!(entropy.persistent_entropy, 0.0);
    }
}
