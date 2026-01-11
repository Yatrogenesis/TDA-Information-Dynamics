//! Persistent Homology Computation
//!
//! Computes persistence diagrams from Vietoris-Rips filtrations.
//! A persistence interval [b, d) represents a topological feature
//! that is "born" at filtration value b and "dies" at value d.
//!
//! ## Interpretation
//!
//! - Long-lived features (large d-b) represent robust topological structure
//! - Short-lived features may be noise or transient phenomena
//! - The persistence diagram encodes the "topological fingerprint" of the data

use super::VietorisRips;

/// A persistence interval [birth, death)
#[derive(Debug, Clone, Copy)]
pub struct PersistenceInterval {
    pub birth: f64,
    pub death: f64,
    pub dimension: usize,
}

impl PersistenceInterval {
    pub fn new(birth: f64, death: f64, dimension: usize) -> Self {
        Self { birth, death, dimension }
    }

    /// Lifetime of the feature
    pub fn persistence(&self) -> f64 {
        self.death - self.birth
    }

    /// Is this an essential feature (infinite persistence)?
    pub fn is_essential(&self) -> bool {
        self.death.is_infinite()
    }
}

/// Persistence diagram: collection of intervals for each dimension
#[derive(Debug, Clone)]
pub struct PersistenceDiagram {
    pub intervals: Vec<PersistenceInterval>,
    pub max_dimension: usize,
}

impl PersistenceDiagram {
    pub fn new() -> Self {
        Self {
            intervals: Vec::new(),
            max_dimension: 0,
        }
    }

    pub fn add(&mut self, interval: PersistenceInterval) {
        if interval.dimension > self.max_dimension {
            self.max_dimension = interval.dimension;
        }
        self.intervals.push(interval);
    }

    /// Get all intervals for a given dimension
    pub fn dim(&self, d: usize) -> Vec<&PersistenceInterval> {
        self.intervals.iter().filter(|i| i.dimension == d).collect()
    }

    /// Number of finite intervals in dimension d
    pub fn betti(&self, d: usize) -> usize {
        self.intervals.iter()
            .filter(|i| i.dimension == d && !i.is_essential())
            .count()
    }

    /// Total persistence in dimension d
    pub fn total_persistence(&self, d: usize) -> f64 {
        self.intervals.iter()
            .filter(|i| i.dimension == d && !i.is_essential())
            .map(|i| i.persistence())
            .sum()
    }

    /// Mean persistence in dimension d
    pub fn mean_persistence(&self, d: usize) -> f64 {
        let intervals: Vec<_> = self.intervals.iter()
            .filter(|i| i.dimension == d && !i.is_essential())
            .collect();
        if intervals.is_empty() {
            return 0.0;
        }
        intervals.iter().map(|i| i.persistence()).sum::<f64>() / intervals.len() as f64
    }

    /// Maximum persistence in dimension d
    pub fn max_persistence(&self, d: usize) -> f64 {
        self.intervals.iter()
            .filter(|i| i.dimension == d && !i.is_essential())
            .map(|i| i.persistence())
            .fold(0.0, f64::max)
    }
}

impl Default for PersistenceDiagram {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute persistence diagram from Vietoris-Rips complex
///
/// This implements a simplified algorithm suitable for small to medium
/// point clouds. For production use, consider interfacing with Ripser.
pub fn compute_persistence(vr: &VietorisRips) -> PersistenceDiagram {
    let mut diagram = PersistenceDiagram::new();
    let n = vr.n_points();
    let n_steps = vr.n_steps();

    // H0: Track component births and deaths via union-find
    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank = vec![0usize; n];
    let mut birth_time = vec![0usize; n];  // When each component was born
    let mut alive = vec![true; n];  // Which components are still independent

    fn find(parent: &mut [usize], i: usize) -> usize {
        if parent[i] != i {
            parent[i] = find(parent, parent[i]);
        }
        parent[i]
    }

    // Collect all edges with their filtration times
    let mut edges: Vec<(usize, usize, usize)> = Vec::new();
    for step in 1..=n_steps {
        let epsilon = vr.epsilon_at(step);
        let prev_epsilon = vr.epsilon_at(step - 1);

        for i in 0..n {
            for j in i+1..n {
                // Check if edge appears at this step
                let d = vr.epsilon_at(vr.connection_time(i, j));
                if d > prev_epsilon && d <= epsilon {
                    edges.push((i, j, step));
                }
            }
        }
    }

    // Sort edges by filtration time
    edges.sort_by_key(|e| e.2);

    // Process edges to track H0
    for (i, j, step) in edges {
        let ri = find(&mut parent, i);
        let rj = find(&mut parent, j);

        if ri != rj {
            // Merge components - younger one dies
            let (survivor, dying) = if birth_time[ri] <= birth_time[rj] {
                (ri, rj)
            } else {
                (rj, ri)
            };

            // Union
            if rank[survivor] < rank[dying] {
                parent[survivor] = dying;
                // Actually dying should be the one with later birth
                let death_eps = vr.epsilon_at(step);
                let birth_eps = vr.epsilon_at(birth_time[dying]);
                if death_eps > birth_eps {
                    diagram.add(PersistenceInterval::new(birth_eps, death_eps, 0));
                }
            } else {
                parent[dying] = survivor;
                let death_eps = vr.epsilon_at(step);
                let birth_eps = vr.epsilon_at(birth_time[dying]);
                if death_eps > birth_eps {
                    diagram.add(PersistenceInterval::new(birth_eps, death_eps, 0));
                }
                if rank[survivor] == rank[dying] {
                    rank[survivor] += 1;
                }
            }
        }
    }

    // Add essential H0 classes (surviving components)
    let mut roots = std::collections::HashSet::new();
    for i in 0..n {
        roots.insert(find(&mut parent, i));
    }
    for root in roots {
        diagram.add(PersistenceInterval::new(0.0, f64::INFINITY, 0));
    }

    // H1: Approximate cycle detection
    // Count cycles at each filtration step and track changes
    let mut prev_cycles = 0;
    for step in 1..=n_steps {
        let cycles = vr.estimate_cycles_at(step);
        let epsilon = vr.epsilon_at(step);

        // New cycles born
        if cycles > prev_cycles {
            for _ in 0..(cycles - prev_cycles) {
                // Cycle born at this step, death unknown (set to max)
                diagram.add(PersistenceInterval::new(epsilon, vr.epsilon_at(n_steps), 1));
            }
        }
        prev_cycles = cycles;
    }

    diagram
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_persistence_components() {
        // Two points far apart
        let points = array![
            [0.0, 0.0],
            [10.0, 0.0]
        ];

        let vr = VietorisRips::from_points(&points, 15.0, 100);
        let pd = compute_persistence(&vr);

        // Should have H0 interval from 0 to ~10
        let dim0 = pd.dim(0);
        let h0: Vec<_> = dim0.iter().filter(|i| !i.is_essential()).collect();
        assert!(!h0.is_empty());
    }
}
