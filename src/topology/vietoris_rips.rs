//! Vietoris-Rips Complex Construction
//!
//! The Vietoris-Rips complex VR_ε(X) is a simplicial complex where:
//! - 0-simplices are the points in X
//! - A k-simplex [v₀, ..., vₖ] exists iff d(vᵢ, vⱼ) ≤ ε for all i,j
//!
//! This provides a computationally tractable approximation to the
//! underlying topological structure of the point cloud.

use ndarray::Array2;
use std::collections::HashSet;

/// Vietoris-Rips filtration builder
pub struct VietorisRips {
    /// Distance matrix (precomputed)
    distances: Array2<f64>,
    /// Maximum filtration value
    max_epsilon: f64,
    /// Number of filtration steps
    n_steps: usize,
}

/// A simplex represented by its vertex indices
#[allow(dead_code)]
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct Simplex {
    pub vertices: Vec<usize>,
    pub birth: usize,  // Filtration index where simplex appears
}

impl VietorisRips {
    /// Create a new VR filtration from a distance matrix
    pub fn new(distances: Array2<f64>, max_epsilon: f64, n_steps: usize) -> Self {
        Self {
            distances,
            max_epsilon,
            n_steps,
        }
    }

    /// Alias for new - create from precomputed distance matrix
    pub fn from_distance_matrix(distances: &Array2<f64>, max_epsilon: f64, n_steps: usize) -> Self {
        Self {
            distances: distances.clone(),
            max_epsilon,
            n_steps,
        }
    }

    /// Create from point cloud (computes distance matrix)
    pub fn from_points(points: &Array2<f64>, max_epsilon: f64, n_steps: usize) -> Self {
        let distances = Self::compute_distance_matrix(points);
        Self {
            distances,
            max_epsilon,
            n_steps,
        }
    }

    /// Compute Euclidean distance matrix
    fn compute_distance_matrix(points: &Array2<f64>) -> Array2<f64> {
        let n = points.nrows();
        let dim = points.ncols();

        let mut dm = Array2::<f64>::zeros((n, n));

        for i in 0..n {
            for j in i+1..n {
                let mut dist_sq = 0.0;
                for d in 0..dim {
                    let diff = points[[i, d]] - points[[j, d]];
                    dist_sq += diff * diff;
                }
                let dist = dist_sq.sqrt();
                dm[[i, j]] = dist;
                dm[[j, i]] = dist;
            }
        }

        dm
    }

    /// Get filtration value for a given step
    pub fn epsilon_at(&self, step: usize) -> f64 {
        self.max_epsilon * (step as f64) / (self.n_steps as f64)
    }

    /// Get all edges at a given filtration step
    pub fn edges_at(&self, step: usize) -> Vec<(usize, usize)> {
        let epsilon = self.epsilon_at(step);
        let n = self.distances.nrows();
        let mut edges = Vec::new();

        for i in 0..n {
            for j in i+1..n {
                if self.distances[[i, j]] <= epsilon {
                    edges.push((i, j));
                }
            }
        }

        edges
    }

    /// Get all triangles at a given filtration step
    pub fn triangles_at(&self, step: usize) -> Vec<(usize, usize, usize)> {
        let epsilon = self.epsilon_at(step);
        let n = self.distances.nrows();
        let mut triangles = Vec::new();

        for i in 0..n {
            for j in i+1..n {
                if self.distances[[i, j]] > epsilon {
                    continue;
                }
                for k in j+1..n {
                    if self.distances[[i, k]] <= epsilon &&
                       self.distances[[j, k]] <= epsilon {
                        triangles.push((i, j, k));
                    }
                }
            }
        }

        triangles
    }

    /// Count connected components using Union-Find
    pub fn count_components_at(&self, step: usize) -> usize {
        let n = self.distances.nrows();
        let mut parent: Vec<usize> = (0..n).collect();
        let mut rank = vec![0usize; n];

        fn find(parent: &mut [usize], i: usize) -> usize {
            if parent[i] != i {
                parent[i] = find(parent, parent[i]);
            }
            parent[i]
        }

        fn union(parent: &mut [usize], rank: &mut [usize], x: usize, y: usize) {
            let rx = find(parent, x);
            let ry = find(parent, y);
            if rx != ry {
                if rank[rx] < rank[ry] {
                    parent[rx] = ry;
                } else if rank[rx] > rank[ry] {
                    parent[ry] = rx;
                } else {
                    parent[ry] = rx;
                    rank[rx] += 1;
                }
            }
        }

        let edges = self.edges_at(step);
        for (i, j) in edges {
            union(&mut parent, &mut rank, i, j);
        }

        // Count unique roots
        let mut roots = HashSet::new();
        for i in 0..n {
            roots.insert(find(&mut parent, i));
        }
        roots.len()
    }

    /// Estimate number of 1-cycles (loops) at filtration step
    /// Using Euler characteristic: χ = V - E + F
    /// For connected: β₁ = E - V + β₀ (when no 2-simplices filled)
    pub fn estimate_cycles_at(&self, step: usize) -> usize {
        let n = self.distances.nrows();  // V
        let edges = self.edges_at(step);
        let e = edges.len();
        let triangles = self.triangles_at(step);
        let f = triangles.len();
        let beta0 = self.count_components_at(step);

        // β₁ = E - V + β₀ - F (approximate)
        // This is a lower bound when triangles don't fill all cycles
        if e + beta0 > n + f {
            e + beta0 - n - f
        } else {
            0
        }
    }

    /// Get the filtration step where two points become connected
    pub fn connection_time(&self, i: usize, j: usize) -> usize {
        let d = self.distances[[i, j]];
        let step = (d / self.max_epsilon * self.n_steps as f64).ceil() as usize;
        step.min(self.n_steps)
    }

    /// Number of points
    pub fn n_points(&self) -> usize {
        self.distances.nrows()
    }

    /// Number of filtration steps
    pub fn n_steps(&self) -> usize {
        self.n_steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_triangle() {
        // Equilateral triangle with side 1
        let points = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.866]
        ];

        let vr = VietorisRips::from_points(&points, 2.0, 100);

        // At ε < 1, three components
        assert_eq!(vr.count_components_at(40), 3);

        // At ε ≥ 1, one component
        assert_eq!(vr.count_components_at(60), 1);
    }
}
