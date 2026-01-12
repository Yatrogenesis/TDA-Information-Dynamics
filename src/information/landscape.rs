//! Persistence Landscapes: Functional Summaries of Persistence Diagrams
//!
//! Persistence landscapes provide a stable, vectorizable representation
//! of persistence diagrams suitable for statistical analysis.

use crate::topology::PersistenceDiagram;

/// Persistence landscape representation
#[derive(Debug, Clone)]
pub struct PersistenceLandscape {
    /// Landscape functions λₖ(t) for k = 1, 2, ...
    pub functions: Vec<Vec<(f64, f64)>>,
    /// Number of landscape functions
    pub k_max: usize,
    /// Grid resolution
    pub resolution: usize,
}

impl PersistenceLandscape {
    /// Compute persistence landscape from diagram
    pub fn from_diagram(pd: &PersistenceDiagram, dimension: usize, k_max: usize, resolution: usize) -> Self {
        let intervals: Vec<_> = pd.intervals.iter()
            .filter(|i| i.dimension == dimension && !i.is_essential())
            .collect();

        if intervals.is_empty() {
            return Self {
                functions: vec![vec![(0.0, 0.0)]; k_max],
                k_max,
                resolution,
            };
        }

        // Determine grid range
        let min_t = intervals.iter().map(|i| i.birth).fold(f64::INFINITY, f64::min);
        let max_t = intervals.iter().map(|i| i.death).fold(0.0, f64::max);

        let dt = (max_t - min_t) / resolution as f64;

        // Compute landscape values at each grid point
        let mut functions = Vec::with_capacity(k_max);

        for k in 0..k_max {
            let mut func = Vec::with_capacity(resolution);

            for i in 0..=resolution {
                let t = min_t + i as f64 * dt;

                // Compute tent function values for all intervals
                let mut values: Vec<f64> = intervals.iter()
                    .map(|interval| tent_function(t, interval.birth, interval.death))
                    .collect();

                // Sort descending
                values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

                // k-th largest value
                let lambda_k = if k < values.len() { values[k] } else { 0.0 };
                func.push((t, lambda_k));
            }

            functions.push(func);
        }

        Self {
            functions,
            k_max,
            resolution,
        }
    }

    /// L^p norm of the k-th landscape function
    pub fn lp_norm(&self, k: usize, p: f64) -> f64 {
        if k >= self.functions.len() {
            return 0.0;
        }

        let func = &self.functions[k];
        if func.len() < 2 {
            return 0.0;
        }

        let mut integral = 0.0;
        for i in 1..func.len() {
            let dt = func[i].0 - func[i-1].0;
            let avg = (func[i].1.abs().powf(p) + func[i-1].1.abs().powf(p)) / 2.0;
            integral += dt * avg;
        }

        integral.powf(1.0 / p)
    }

    /// L^2 norm of all landscape functions combined
    pub fn total_l2_norm(&self) -> f64 {
        let mut total = 0.0;
        for k in 0..self.k_max {
            total += self.lp_norm(k, 2.0).powi(2);
        }
        total.sqrt()
    }

    /// Inner product with another landscape
    pub fn inner_product(&self, other: &PersistenceLandscape) -> f64 {
        if self.k_max != other.k_max || self.resolution != other.resolution {
            return 0.0;
        }

        let mut product = 0.0;
        for k in 0..self.k_max {
            let f1 = &self.functions[k];
            let f2 = &other.functions[k];

            if f1.len() != f2.len() {
                continue;
            }

            for i in 1..f1.len() {
                let dt = f1[i].0 - f1[i-1].0;
                let prod = (f1[i].1 * f2[i].1 + f1[i-1].1 * f2[i-1].1) / 2.0;
                product += dt * prod;
            }
        }

        product
    }
}

/// Tent function for persistence interval
fn tent_function(t: f64, birth: f64, death: f64) -> f64 {
    let mid = (birth + death) / 2.0;

    if t < birth || t > death {
        0.0
    } else if t <= mid {
        t - birth
    } else {
        death - t
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::{PersistenceInterval, PersistenceDiagram};

    #[test]
    fn test_landscape_single() {
        let mut pd = PersistenceDiagram::new();
        pd.add(PersistenceInterval::new(0.0, 2.0, 0));

        let landscape = PersistenceLandscape::from_diagram(&pd, 0, 1, 100);

        // Maximum should be at midpoint with value 1.0
        let max_val = landscape.functions[0].iter()
            .map(|(_, v)| *v)
            .fold(0.0, f64::max);

        assert!((max_val - 1.0).abs() < 0.1);
    }
}
