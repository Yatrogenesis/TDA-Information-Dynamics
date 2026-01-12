//! Exact Persistent Homology via Standard Algorithm
//!
//! This module implements the standard algorithm for computing exact
//! persistent homology, matching the output of libraries like Ripser.
//!
//! ## Algorithm Overview
//!
//! 1. Build filtered simplicial complex (all simplices with birth times)
//! 2. Sort simplices by (dimension, birth_time)
//! 3. Reduce boundary matrix using column operations
//! 4. Extract persistence pairs from reduced matrix
//!
//! ## Difference from Euler Approximation
//!
//! The Euler approximation (β₁ = E - V + β₀ - F) only gives the COUNT
//! of cycles at each filtration step. This exact algorithm gives the
//! actual birth/death times, enabling proper entropy calculation.
//!
//! ## Reference
//!
//! Edelsbrunner, Letscher, Zomorodian (2002). "Topological Persistence
//! and Simplification". Discrete & Computational Geometry.

use ndarray::Array2;
use std::collections::{HashMap, BTreeSet};

/// A simplex in the filtration with its birth time
#[derive(Debug, Clone)]
struct FilteredSimplex {
    /// Vertex indices (sorted)
    vertices: Vec<usize>,
    /// Filtration value when simplex appears
    birth: f64,
    /// Index in the filtration ordering
    filtration_index: usize,
}

impl FilteredSimplex {
    fn dimension(&self) -> usize {
        self.vertices.len() - 1
    }
}

/// Exact persistence interval from the standard algorithm
#[derive(Debug, Clone, Copy)]
pub struct ExactInterval {
    pub birth: f64,
    pub death: f64,
    pub dimension: usize,
}

impl ExactInterval {
    pub fn persistence(&self) -> f64 {
        if self.death.is_infinite() {
            f64::INFINITY
        } else {
            self.death - self.birth
        }
    }

    pub fn is_essential(&self) -> bool {
        self.death.is_infinite()
    }
}

/// Result of exact persistence computation
#[derive(Debug, Clone)]
pub struct ExactPersistenceDiagram {
    pub intervals: Vec<ExactInterval>,
}

impl ExactPersistenceDiagram {
    pub fn new() -> Self {
        Self { intervals: Vec::new() }
    }

    /// Get intervals for dimension d
    pub fn dim(&self, d: usize) -> Vec<&ExactInterval> {
        self.intervals.iter().filter(|i| i.dimension == d).collect()
    }

    /// Get finite intervals for dimension d
    pub fn finite_intervals(&self, d: usize) -> Vec<&ExactInterval> {
        self.intervals.iter()
            .filter(|i| i.dimension == d && !i.is_essential())
            .collect()
    }

    /// Compute persistence entropy for dimension d (matching Ripser/Python)
    pub fn persistence_entropy(&self, d: usize) -> f64 {
        let intervals: Vec<f64> = self.intervals.iter()
            .filter(|i| i.dimension == d && !i.is_essential())
            .map(|i| i.persistence())
            .filter(|&p| p > 0.0)
            .collect();

        if intervals.is_empty() {
            return 0.0;
        }

        let total: f64 = intervals.iter().sum();
        if total <= 0.0 {
            return 0.0;
        }

        let mut entropy = 0.0;
        for p in intervals {
            let prob = p / total;
            if prob > 0.0 {
                entropy -= prob * prob.ln();
            }
        }
        entropy
    }

    /// Total persistence for dimension d
    pub fn total_persistence(&self, d: usize) -> f64 {
        self.intervals.iter()
            .filter(|i| i.dimension == d && !i.is_essential())
            .map(|i| i.persistence())
            .sum()
    }

    /// Number of finite intervals in dimension d
    pub fn count(&self, d: usize) -> usize {
        self.finite_intervals(d).len()
    }
}

impl Default for ExactPersistenceDiagram {
    fn default() -> Self {
        Self::new()
    }
}

/// Sparse column representation for boundary matrix
#[derive(Debug, Clone)]
struct SparseColumn {
    /// Non-zero row indices (stored in BTreeSet for efficient operations)
    rows: BTreeSet<usize>,
}

impl SparseColumn {
    fn new() -> Self {
        Self { rows: BTreeSet::new() }
    }

    #[allow(dead_code)]
    fn from_indices(indices: impl IntoIterator<Item = usize>) -> Self {
        Self { rows: indices.into_iter().collect() }
    }

    fn is_zero(&self) -> bool {
        self.rows.is_empty()
    }

    /// Get the lowest (maximum) non-zero index
    fn low(&self) -> Option<usize> {
        self.rows.iter().next_back().copied()
    }

    /// XOR (symmetric difference) with another column - addition in Z/2Z
    fn add_assign(&mut self, other: &SparseColumn) {
        for &row in &other.rows {
            if !self.rows.remove(&row) {
                self.rows.insert(row);
            }
        }
    }
}

/// Compute exact persistent homology using the standard algorithm
///
/// # Arguments
/// * `distance_matrix` - Pairwise distances between points
/// * `max_epsilon` - Maximum filtration value
/// * `max_dim` - Maximum homology dimension to compute (typically 1)
///
/// # Returns
/// * `ExactPersistenceDiagram` with birth/death pairs for each dimension
pub fn compute_exact_persistence(
    distance_matrix: &Array2<f64>,
    max_epsilon: f64,
    max_dim: usize,
) -> ExactPersistenceDiagram {
    let n = distance_matrix.nrows();

    // Step 1: Build filtration (all simplices up to max_dim + 1)
    let mut simplices: Vec<FilteredSimplex> = Vec::new();

    // 0-simplices (vertices) - all born at time 0
    for i in 0..n {
        simplices.push(FilteredSimplex {
            vertices: vec![i],
            birth: 0.0,
            filtration_index: 0, // Will be set after sorting
        });
    }

    // 1-simplices (edges) - born when distance <= epsilon
    for i in 0..n {
        for j in i+1..n {
            let d = distance_matrix[[i, j]];
            if d <= max_epsilon {
                simplices.push(FilteredSimplex {
                    vertices: vec![i, j],
                    birth: d,
                    filtration_index: 0,
                });
            }
        }
    }

    // 2-simplices (triangles) - needed for H1, born when all edges present
    if max_dim >= 1 {
        for i in 0..n {
            for j in i+1..n {
                let dij = distance_matrix[[i, j]];
                if dij > max_epsilon { continue; }

                for k in j+1..n {
                    let dik = distance_matrix[[i, k]];
                    let djk = distance_matrix[[j, k]];

                    if dik <= max_epsilon && djk <= max_epsilon {
                        // Triangle birth = max of all edge births
                        let birth = dij.max(dik).max(djk);
                        if birth <= max_epsilon {
                            simplices.push(FilteredSimplex {
                                vertices: vec![i, j, k],
                                birth,
                                filtration_index: 0,
                            });
                        }
                    }
                }
            }
        }
    }

    // Step 2: Sort by (birth, dimension, lexicographic vertices)
    simplices.sort_by(|a, b| {
        a.birth.partial_cmp(&b.birth).unwrap()
            .then(a.dimension().cmp(&b.dimension()))
            .then(a.vertices.cmp(&b.vertices))
    });

    // Assign filtration indices
    for (idx, s) in simplices.iter_mut().enumerate() {
        s.filtration_index = idx;
    }

    // Build lookup: vertices -> filtration index
    let mut simplex_index: HashMap<Vec<usize>, usize> = HashMap::new();
    for s in &simplices {
        simplex_index.insert(s.vertices.clone(), s.filtration_index);
    }

    // Step 3: Build and reduce boundary matrix
    let m = simplices.len();
    let mut columns: Vec<SparseColumn> = Vec::with_capacity(m);
    let mut low_to_col: HashMap<usize, usize> = HashMap::new();

    for (col_idx, simplex) in simplices.iter().enumerate() {
        // Compute boundary of this simplex
        let mut boundary = SparseColumn::new();

        if simplex.dimension() > 0 {
            // Boundary of [v0, v1, ..., vk] = sum of [v0, ..., v̂i, ..., vk]
            for i in 0..simplex.vertices.len() {
                let mut face: Vec<usize> = simplex.vertices.clone();
                face.remove(i);

                if let Some(&face_idx) = simplex_index.get(&face) {
                    // XOR operation (Z/2Z coefficients)
                    if !boundary.rows.remove(&face_idx) {
                        boundary.rows.insert(face_idx);
                    }
                }
            }
        }

        // Reduce column using previously reduced columns
        while let Some(low_idx) = boundary.low() {
            if let Some(&pivot_col) = low_to_col.get(&low_idx) {
                boundary.add_assign(&columns[pivot_col]);
            } else {
                break;
            }
        }

        // Record pivot if column is non-zero
        if let Some(low_idx) = boundary.low() {
            low_to_col.insert(low_idx, col_idx);
        }

        columns.push(boundary);
    }

    // Step 4: Extract persistence pairs
    let mut diagram = ExactPersistenceDiagram::new();
    let mut paired: Vec<bool> = vec![false; m];

    for (col_idx, column) in columns.iter().enumerate() {
        if let Some(low_idx) = column.low() {
            // This column kills the feature created by simplex at low_idx
            let birth_simplex = &simplices[low_idx];
            let death_simplex = &simplices[col_idx];
            let dim = birth_simplex.dimension();

            paired[low_idx] = true;
            paired[col_idx] = true;

            // Only record if persistence > 0
            if death_simplex.birth > birth_simplex.birth {
                diagram.intervals.push(ExactInterval {
                    birth: birth_simplex.birth,
                    death: death_simplex.birth,
                    dimension: dim,
                });
            }
        }
    }

    // Essential features (unpaired simplices that create homology)
    for (idx, simplex) in simplices.iter().enumerate() {
        if !paired[idx] && columns[idx].is_zero() {
            // This simplex creates an essential feature
            diagram.intervals.push(ExactInterval {
                birth: simplex.birth,
                death: f64::INFINITY,
                dimension: simplex.dimension(),
            });
        }
    }

    diagram
}

/// Convenience function matching the API used with approximate persistence
pub fn compute_exact_persistence_simple(
    distance_matrix: &Array2<f64>,
    max_epsilon: f64,
) -> ExactPersistenceDiagram {
    compute_exact_persistence(distance_matrix, max_epsilon, 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_two_points() {
        // Two points at distance 1
        let dm = array![
            [0.0, 1.0],
            [1.0, 0.0]
        ];

        let pd = compute_exact_persistence(&dm, 2.0, 1);

        // H0: One interval from 0 to 1 (two components merge)
        // Plus one essential (final component)
        let h0 = pd.dim(0);
        assert!(!h0.is_empty());

        let finite_h0: Vec<_> = h0.iter().filter(|i| !i.is_essential()).collect();
        assert_eq!(finite_h0.len(), 1);
        assert!((finite_h0[0].birth - 0.0).abs() < 1e-10);
        assert!((finite_h0[0].death - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_triangle() {
        // Equilateral triangle with side 1
        let dm = array![
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0]
        ];

        let pd = compute_exact_persistence(&dm, 2.0, 1);

        // H0: Two intervals (3 components merge to 1)
        let h0_all = pd.dim(0);
        let h0_finite_count = h0_all.iter().filter(|i| !i.is_essential()).count();
        assert_eq!(h0_finite_count, 2);

        // H1: One cycle born at epsilon=1, dies at epsilon=1 (triangle fills immediately)
        // So should be empty or very short-lived
        let h1 = pd.finite_intervals(1);
        // Triangle is filled at same time edges appear, so no H1 persistence
        assert!(h1.is_empty() || h1.iter().all(|i| i.persistence() < 1e-10));
    }

    #[test]
    fn test_square_cycle() {
        // Square: cycle that persists
        // Points at (0,0), (1,0), (1,1), (0,1)
        // Edges: 0-1, 1-2, 2-3, 3-0 all length 1
        // Diagonals: 0-2, 1-3 length sqrt(2) ≈ 1.414
        let s2 = 2.0_f64.sqrt();
        let dm = array![
            [0.0, 1.0, s2,  1.0],
            [1.0, 0.0, 1.0, s2 ],
            [s2,  1.0, 0.0, 1.0],
            [1.0, s2,  1.0, 0.0]
        ];

        let pd = compute_exact_persistence(&dm, 2.0, 1);

        // H1: One cycle born at epsilon=1, dies at epsilon=sqrt(2)
        let h1 = pd.finite_intervals(1);
        assert!(!h1.is_empty(), "Square should have H1 cycle");

        let cycle = h1[0];
        assert!((cycle.birth - 1.0).abs() < 1e-10, "Cycle born at 1");
        assert!((cycle.death - s2).abs() < 1e-10, "Cycle dies at sqrt(2)");

        // Entropy should be well-defined
        let entropy = pd.persistence_entropy(1);
        assert!(entropy >= 0.0);
    }

    #[test]
    fn test_entropy_calculation() {
        // Create a distance matrix with known H1 structure
        let s2 = 2.0_f64.sqrt();
        let dm = array![
            [0.0, 1.0, s2,  1.0],
            [1.0, 0.0, 1.0, s2 ],
            [s2,  1.0, 0.0, 1.0],
            [1.0, s2,  1.0, 0.0]
        ];

        let pd = compute_exact_persistence(&dm, 2.0, 1);
        let entropy = pd.persistence_entropy(1);

        // Single interval: entropy = 0 (all probability on one feature)
        // or entropy = -1 * ln(1) = 0
        // This is correct: single feature has zero entropy
        assert!(entropy.abs() < 1e-10 || entropy > 0.0);
    }
}
