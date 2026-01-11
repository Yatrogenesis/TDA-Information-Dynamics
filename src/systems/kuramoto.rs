//! Kuramoto Model: Coupled Oscillator Synchronization
//!
//! The Kuramoto model describes N coupled oscillators with phases θᵢ:
//!
//!   dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)
//!
//! where ωᵢ are natural frequencies and K is coupling strength.
//!
//! ## Phase Transition
//!
//! At critical coupling Kc, the system transitions from:
//! - K < Kc: Incoherent phase (random phases)
//! - K > Kc: Synchronized phase (coherent oscillation)
//!
//! ## Topological Signatures
//!
//! The phases θᵢ can be mapped to points on S¹ or ℝ² (via cos/sin).
//! - **Incoherent**: Uniform distribution → high β₀, low β₁
//! - **Synchronized**: Clustered distribution → low β₀, emerging β₁

use ndarray::{Array1, Array2};
use rand_distr::{Normal, Uniform, Distribution};
use std::f64::consts::PI;

/// State of Kuramoto system
#[derive(Debug, Clone)]
pub struct KuramotoState {
    /// Oscillator phases θᵢ ∈ [0, 2π)
    pub phases: Array1<f64>,
    /// Natural frequencies ωᵢ
    pub frequencies: Array1<f64>,
    /// Order parameter r (synchronization measure)
    pub order_parameter: f64,
    /// Mean phase ψ
    pub mean_phase: f64,
    /// Coupling strength K
    pub coupling: f64,
    /// Simulation time
    pub time: f64,
}

/// Kuramoto oscillator system
pub struct KuramotoSystem {
    /// Number of oscillators
    n_oscillators: usize,
    /// Natural frequencies
    frequencies: Array1<f64>,
    /// Current phases
    phases: Array1<f64>,
    /// Coupling strength
    coupling: f64,
    /// Integration timestep
    dt: f64,
    /// Current time
    time: f64,
}

impl KuramotoSystem {
    /// Create new Kuramoto system
    ///
    /// # Arguments
    /// * `n` - Number of oscillators
    /// * `coupling` - Coupling strength K
    /// * `freq_std` - Standard deviation of frequency distribution
    pub fn new(n: usize, coupling: f64, freq_std: f64) -> Self {
        let mut rng = rand::rng();

        // Natural frequencies from Lorentzian (or Gaussian for simplicity)
        let normal = Normal::new(0.0, freq_std).unwrap();
        let frequencies = Array1::from_iter((0..n).map(|_| normal.sample(&mut rng)));

        // Random initial phases
        let uniform = Uniform::new(0.0, 2.0 * PI).unwrap();
        let phases = Array1::from_iter((0..n).map(|_| uniform.sample(&mut rng)));

        Self {
            n_oscillators: n,
            frequencies,
            phases,
            coupling,
            dt: 0.01,
            time: 0.0,
        }
    }

    /// Create system with uniform frequencies (mean-field limit)
    pub fn uniform(n: usize, coupling: f64) -> Self {
        let mut rng = rand::rng();

        let frequencies = Array1::zeros(n);

        // Random initial phases
        let uniform = Uniform::new(0.0, 2.0 * PI).unwrap();
        let phases = Array1::from_iter((0..n).map(|_| uniform.sample(&mut rng)));

        Self {
            n_oscillators: n,
            frequencies,
            phases,
            coupling,
            dt: 0.01,
            time: 0.0,
        }
    }

    /// Compute order parameter r·e^(iψ) = (1/N) Σⱼ e^(iθⱼ)
    fn compute_order_parameter(&self) -> (f64, f64) {
        let n = self.n_oscillators as f64;

        let sum_cos: f64 = self.phases.iter().map(|&theta| theta.cos()).sum();
        let sum_sin: f64 = self.phases.iter().map(|&theta| theta.sin()).sum();

        let r = ((sum_cos / n).powi(2) + (sum_sin / n).powi(2)).sqrt();
        let psi = (sum_sin / n).atan2(sum_cos / n);

        (r, psi)
    }

    /// RK4 integration step
    pub fn step(&mut self) {
        let n = self.n_oscillators;
        let dt = self.dt;
        let k = self.coupling;

        // Compute derivative dθ/dt
        let deriv = |phases: &Array1<f64>| -> Array1<f64> {
            let (r, psi) = {
                let n_f = n as f64;
                let sum_cos: f64 = phases.iter().map(|&theta| theta.cos()).sum();
                let sum_sin: f64 = phases.iter().map(|&theta| theta.sin()).sum();
                let r = ((sum_cos / n_f).powi(2) + (sum_sin / n_f).powi(2)).sqrt();
                let psi = (sum_sin / n_f).atan2(sum_cos / n_f);
                (r, psi)
            };

            // dθᵢ/dt = ωᵢ + K·r·sin(ψ - θᵢ)
            Array1::from_iter(
                self.frequencies.iter().zip(phases.iter())
                    .map(|(&omega, &theta)| omega + k * r * (psi - theta).sin())
            )
        };

        // RK4
        let k1 = deriv(&self.phases);
        let k2 = deriv(&(&self.phases + &(&k1 * (dt / 2.0))));
        let k3 = deriv(&(&self.phases + &(&k2 * (dt / 2.0))));
        let k4 = deriv(&(&self.phases + &(&k3 * dt)));

        // Update phases
        self.phases = &self.phases + &((&k1 + &(&k2 * 2.0) + &(&k3 * 2.0) + &k4) * (dt / 6.0));

        // Wrap to [0, 2π)
        self.phases.mapv_inplace(|theta| theta.rem_euclid(2.0 * PI));

        self.time += dt;
    }

    /// Run multiple steps
    pub fn run(&mut self, n_steps: usize) {
        for _ in 0..n_steps {
            self.step();
        }
    }

    /// Set coupling strength
    pub fn set_coupling(&mut self, coupling: f64) {
        self.coupling = coupling;
    }

    /// Gradually increase coupling (for transition studies)
    pub fn ramp_coupling(&mut self, target: f64, rate: f64, steps_per_increment: usize) {
        while self.coupling < target {
            self.run(steps_per_increment);
            self.coupling += rate;
        }
        self.coupling = target;
    }

    /// Get current state
    pub fn state(&self) -> KuramotoState {
        let (r, psi) = self.compute_order_parameter();

        KuramotoState {
            phases: self.phases.clone(),
            frequencies: self.frequencies.clone(),
            order_parameter: r,
            mean_phase: psi,
            coupling: self.coupling,
            time: self.time,
        }
    }

    /// Get phases
    pub fn phases(&self) -> &Array1<f64> {
        &self.phases
    }

    /// Map phases to 2D points on unit circle (for TDA)
    pub fn to_circle_points(&self) -> Array2<f64> {
        let n = self.n_oscillators;
        let mut points = Array2::zeros((n, 2));

        for i in 0..n {
            let theta = self.phases[i];
            points[[i, 0]] = theta.cos();
            points[[i, 1]] = theta.sin();
        }

        points
    }

    /// Map phases to 3D torus embedding (for richer TDA)
    /// Uses (cos θ, sin θ, θ/2π) embedding
    pub fn to_cylinder_points(&self) -> Array2<f64> {
        let n = self.n_oscillators;
        let mut points = Array2::zeros((n, 3));

        for i in 0..n {
            let theta = self.phases[i];
            points[[i, 0]] = theta.cos();
            points[[i, 1]] = theta.sin();
            points[[i, 2]] = theta / (2.0 * PI);
        }

        points
    }

    /// Compute distance matrix on circle (geodesic distance)
    pub fn circle_distance_matrix(&self) -> Array2<f64> {
        let n = self.n_oscillators;
        let mut dist = Array2::zeros((n, n));

        for i in 0..n {
            for j in i + 1..n {
                // Geodesic distance on S¹
                let d_theta = (self.phases[i] - self.phases[j]).abs();
                let d = d_theta.min(2.0 * PI - d_theta);
                dist[[i, j]] = d;
                dist[[j, i]] = d;
            }
        }

        dist
    }

    /// Compute Euclidean distance matrix for circle embedding
    pub fn euclidean_distance_matrix(&self) -> Array2<f64> {
        let points = self.to_circle_points();
        let n = self.n_oscillators;
        let mut dist = Array2::zeros((n, n));

        for i in 0..n {
            for j in i + 1..n {
                let dx = points[[i, 0]] - points[[j, 0]];
                let dy = points[[i, 1]] - points[[j, 1]];
                let d = (dx * dx + dy * dy).sqrt();
                dist[[i, j]] = d;
                dist[[j, i]] = d;
            }
        }

        dist
    }

    /// Order parameter (synchronization measure)
    pub fn order_parameter(&self) -> f64 {
        self.compute_order_parameter().0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kuramoto_initialization() {
        let system = KuramotoSystem::new(100, 1.0, 0.5);
        assert_eq!(system.n_oscillators, 100);
        assert!(system.order_parameter() >= 0.0 && system.order_parameter() <= 1.0);
    }

    #[test]
    fn test_kuramoto_synchronization() {
        // Strong coupling should lead to synchronization
        let mut system = KuramotoSystem::uniform(50, 5.0);

        // Run for a while
        system.run(10000);

        // With strong coupling and uniform frequencies, should synchronize
        let r = system.order_parameter();
        assert!(r > 0.9, "Expected synchronization, got r = {}", r);
    }

    #[test]
    fn test_kuramoto_incoherent() {
        // Weak coupling should remain incoherent
        let mut system = KuramotoSystem::new(100, 0.1, 1.0);

        system.run(1000);

        // With weak coupling, should remain relatively incoherent
        let r = system.order_parameter();
        assert!(r < 0.5, "Expected incoherence, got r = {}", r);
    }
}
