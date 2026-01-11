//! Betti Numbers: Topological Invariants
//!
//! The k-th Betti number βₖ counts the number of k-dimensional
//! "holes" in a topological space:
//!
//! - β₀: Number of connected components
//! - β₁: Number of 1-dimensional loops/cycles
//! - β₂: Number of 2-dimensional voids/cavities
//!
//! These invariants provide a compact summary of the topological
//! structure at a given filtration scale.

use super::VietorisRips;

/// Betti numbers at a specific filtration value
#[derive(Debug, Clone, Copy)]
pub struct BettiNumbers {
    pub beta_0: usize,  // Connected components
    pub beta_1: usize,  // Loops
    pub beta_2: usize,  // Voids
    pub epsilon: f64,   // Filtration scale
}

impl BettiNumbers {
    pub fn new(beta_0: usize, beta_1: usize, beta_2: usize, epsilon: f64) -> Self {
        Self { beta_0, beta_1, beta_2, epsilon }
    }

    /// Compute Betti numbers at a given filtration step
    pub fn at_step(vr: &VietorisRips, step: usize) -> Self {
        let epsilon = vr.epsilon_at(step);
        let beta_0 = vr.count_components_at(step);
        let beta_1 = vr.estimate_cycles_at(step);
        let beta_2 = 0;  // Requires tetrahedra counting - approximation

        Self::new(beta_0, beta_1, beta_2, epsilon)
    }

    /// Total topological complexity
    pub fn total(&self) -> usize {
        self.beta_0 + self.beta_1 + self.beta_2
    }

    /// Euler characteristic χ = β₀ - β₁ + β₂
    pub fn euler_characteristic(&self) -> i64 {
        self.beta_0 as i64 - self.beta_1 as i64 + self.beta_2 as i64
    }
}

/// Betti curve: sequence of Betti numbers across filtration
#[derive(Debug, Clone)]
pub struct BettiCurve {
    pub values: Vec<BettiNumbers>,
}

impl BettiCurve {
    /// Compute full Betti curve
    pub fn compute(vr: &VietorisRips) -> Self {
        let values: Vec<BettiNumbers> = (0..=vr.n_steps())
            .map(|step| BettiNumbers::at_step(vr, step))
            .collect();

        Self { values }
    }

    /// Get β₀ curve
    pub fn beta_0_curve(&self) -> Vec<(f64, usize)> {
        self.values.iter().map(|b| (b.epsilon, b.beta_0)).collect()
    }

    /// Get β₁ curve
    pub fn beta_1_curve(&self) -> Vec<(f64, usize)> {
        self.values.iter().map(|b| (b.epsilon, b.beta_1)).collect()
    }

    /// Integrated β₁ (area under curve)
    pub fn integrated_beta_1(&self) -> f64 {
        if self.values.len() < 2 {
            return 0.0;
        }

        let mut integral = 0.0;
        for i in 1..self.values.len() {
            let de = self.values[i].epsilon - self.values[i-1].epsilon;
            let avg = (self.values[i].beta_1 + self.values[i-1].beta_1) as f64 / 2.0;
            integral += de * avg;
        }
        integral
    }
}
