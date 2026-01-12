//! Brusselator: Chemical Oscillations
//!
//! The Brusselator is a theoretical model for autocatalytic chemical reactions:
//!
//!   dX/dt = A + X²Y - (B+1)X
//!   dY/dt = BX - X²Y
//!
//! where:
//! - X, Y: Chemical concentrations
//! - A: Input concentration (fixed)
//! - B: Control parameter (bifurcation parameter)
//!
//! ## Hopf Bifurcation
//!
//! The system has a fixed point at (X*, Y*) = (A, B/A).
//! A Hopf bifurcation occurs at:
//!
//!   B_c = 1 + A²
//!
//! For A = 1: B_c = 2
//!
//! - B < B_c: Stable fixed point (steady state)
//! - B > B_c: Limit cycle (oscillations)
//!
//! ## Topological Signatures
//!
//! Similar to FitzHugh-Nagumo:
//! - **Steady**: Trajectory concentrated at point → small diameter
//! - **Oscillating**: Trajectory forms limit cycle → β₁ = 1
//!
//! The Brusselator is important for:
//! - Belousov-Zhabotinsky reactions
//! - Pattern formation in chemistry/biology
//! - Trinity-like chemical systems
//!
//! ## References
//!
//! - Prigogine, I. & Lefever, R. (1968). Symmetry breaking instabilities
//!   in dissipative systems II. J. Chem. Phys.
//! - Nicolis, G. & Prigogine, I. (1977). Self-Organization in
//!   Nonequilibrium Systems. Wiley.

use ndarray::Array2;
use rand_distr::{Normal, Distribution};

use super::traits::{DynamicalSystem, Controllable, Bifurcating};

/// State of the Brusselator system
#[derive(Debug, Clone)]
pub struct BrusselatorState {
    /// Chemical concentration X
    pub x: f64,
    /// Chemical concentration Y
    pub y: f64,
    /// Parameter A
    pub a: f64,
    /// Parameter B (control parameter)
    pub b: f64,
    /// Simulation time
    pub time: f64,
    /// Oscillation amplitude
    pub amplitude: f64,
}

/// Brusselator oscillator (single or network of coupled cells)
pub struct Brusselator {
    /// Number of cells
    n_cells: usize,
    /// X concentrations
    x: Vec<f64>,
    /// Y concentrations
    y: Vec<f64>,
    /// Parameter A (fixed)
    a: f64,
    /// Parameter B (control parameter)
    b: f64,
    /// Coupling strength (for network)
    coupling: f64,
    /// Integration timestep
    dt: f64,
    /// Current time
    time: f64,
    /// Trajectory history for TDA
    trajectory: Vec<(f64, f64)>,
    /// Maximum trajectory length
    max_trajectory: usize,
}

impl Brusselator {
    /// Create new Brusselator with specified parameters
    pub fn new(n_cells: usize, a: f64, b: f64, coupling: f64) -> Self {
        // Initialize near the fixed point
        let x_fp = a;
        let y_fp = b / a;

        let x = vec![x_fp; n_cells];
        let y = vec![y_fp; n_cells];

        Self {
            n_cells,
            x,
            y,
            a,
            b,
            coupling,
            dt: 0.01,
            time: 0.0,
            trajectory: Vec::new(),
            max_trajectory: 500,
        }
    }

    /// Create single Brusselator oscillator
    pub fn single(a: f64, b: f64) -> Self {
        Self::new(1, a, b, 0.0)
    }

    /// Standard Brusselator with A = 1
    pub fn standard(b: f64) -> Self {
        Self::single(1.0, b)
    }

    /// Add noise to initial conditions
    pub fn perturb(&mut self, noise_std: f64) {
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, noise_std).unwrap();

        for i in 0..self.n_cells {
            self.x[i] += normal.sample(&mut rng).abs();  // Keep positive
            self.y[i] += normal.sample(&mut rng).abs();
        }
    }

    /// Set parameter B
    pub fn set_b(&mut self, b: f64) {
        self.b = b.max(0.0);
    }

    /// Get parameter B
    pub fn get_b(&self) -> f64 {
        self.b
    }

    /// Brusselator dynamics
    fn derivatives(&self, x: f64, y: f64, coupling_term: f64) -> (f64, f64) {
        let a = self.a;
        let b = self.b;

        // dX/dt = A + X²Y - (B+1)X + coupling
        let dx = a + x * x * y - (b + 1.0) * x + coupling_term;
        // dY/dt = BX - X²Y
        let dy = b * x - x * x * y;

        (dx, dy)
    }

    /// RK4 integration step
    fn rk4_step(&mut self) {
        let dt = self.dt;
        let n = self.n_cells;

        // Store old values
        let x_old = self.x.clone();
        let y_old = self.y.clone();

        // Coupling term (diffusive)
        let coupling_term = |x_arr: &[f64], i: usize| -> f64 {
            if n == 1 || self.coupling == 0.0 {
                return 0.0;
            }
            let x_mean: f64 = x_arr.iter().sum::<f64>() / n as f64;
            self.coupling * (x_mean - x_arr[i])
        };

        // k1
        let mut k1_x = vec![0.0; n];
        let mut k1_y = vec![0.0; n];
        for i in 0..n {
            let (dx, dy) = self.derivatives(x_old[i], y_old[i], coupling_term(&x_old, i));
            k1_x[i] = dx;
            k1_y[i] = dy;
        }

        // k2
        let x_mid1: Vec<f64> = x_old.iter().zip(&k1_x).map(|(x, k)| x + k * dt / 2.0).collect();
        let y_mid1: Vec<f64> = y_old.iter().zip(&k1_y).map(|(y, k)| y + k * dt / 2.0).collect();
        let mut k2_x = vec![0.0; n];
        let mut k2_y = vec![0.0; n];
        for i in 0..n {
            let (dx, dy) = self.derivatives(x_mid1[i], y_mid1[i], coupling_term(&x_mid1, i));
            k2_x[i] = dx;
            k2_y[i] = dy;
        }

        // k3
        let x_mid2: Vec<f64> = x_old.iter().zip(&k2_x).map(|(x, k)| x + k * dt / 2.0).collect();
        let y_mid2: Vec<f64> = y_old.iter().zip(&k2_y).map(|(y, k)| y + k * dt / 2.0).collect();
        let mut k3_x = vec![0.0; n];
        let mut k3_y = vec![0.0; n];
        for i in 0..n {
            let (dx, dy) = self.derivatives(x_mid2[i], y_mid2[i], coupling_term(&x_mid2, i));
            k3_x[i] = dx;
            k3_y[i] = dy;
        }

        // k4
        let x_end: Vec<f64> = x_old.iter().zip(&k3_x).map(|(x, k)| x + k * dt).collect();
        let y_end: Vec<f64> = y_old.iter().zip(&k3_y).map(|(y, k)| y + k * dt).collect();
        let mut k4_x = vec![0.0; n];
        let mut k4_y = vec![0.0; n];
        for i in 0..n {
            let (dx, dy) = self.derivatives(x_end[i], y_end[i], coupling_term(&x_end, i));
            k4_x[i] = dx;
            k4_y[i] = dy;
        }

        // Update
        for i in 0..n {
            self.x[i] = x_old[i] + (k1_x[i] + 2.0 * k2_x[i] + 2.0 * k3_x[i] + k4_x[i]) * dt / 6.0;
            self.y[i] = y_old[i] + (k1_y[i] + 2.0 * k2_y[i] + 2.0 * k3_y[i] + k4_y[i]) * dt / 6.0;

            // Keep concentrations positive
            self.x[i] = self.x[i].max(0.0);
            self.y[i] = self.y[i].max(0.0);
        }

        self.time += dt;

        // Store trajectory (mean X, Y)
        let x_mean: f64 = self.x.iter().sum::<f64>() / n as f64;
        let y_mean: f64 = self.y.iter().sum::<f64>() / n as f64;
        self.trajectory.push((x_mean, y_mean));

        if self.trajectory.len() > self.max_trajectory {
            self.trajectory.remove(0);
        }
    }

    /// Compute oscillation amplitude
    pub fn compute_amplitude(&self) -> f64 {
        if self.trajectory.len() < 10 {
            return 0.0;
        }

        let recent: Vec<f64> = self.trajectory.iter()
            .rev()
            .take(100)
            .map(|(x, _)| *x)
            .collect();

        let x_max = recent.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let x_min = recent.iter().cloned().fold(f64::INFINITY, f64::min);

        x_max - x_min
    }

    /// Distance matrix for trajectory
    pub fn trajectory_distance_matrix(&self) -> Array2<f64> {
        let n = self.trajectory.len();
        let mut dist = Array2::zeros((n, n));

        for i in 0..n {
            for j in i + 1..n {
                let (x_i, y_i) = self.trajectory[i];
                let (x_j, y_j) = self.trajectory[j];
                let d = ((x_i - x_j).powi(2) + (y_i - y_j).powi(2)).sqrt();
                dist[[i, j]] = d;
                dist[[j, i]] = d;
            }
        }

        dist
    }

    /// Clear trajectory
    pub fn clear_trajectory(&mut self) {
        self.trajectory.clear();
    }

    /// Theoretical critical B for Hopf bifurcation
    ///
    /// B_c = 1 + A²
    pub fn theoretical_b_critical(&self) -> f64 {
        1.0 + self.a * self.a
    }
}

impl DynamicalSystem for Brusselator {
    type State = BrusselatorState;

    fn step(&mut self) {
        self.rk4_step();
    }

    fn state(&self) -> BrusselatorState {
        BrusselatorState {
            x: self.x[0],
            y: self.y[0],
            a: self.a,
            b: self.b,
            time: self.time,
            amplitude: self.compute_amplitude(),
        }
    }

    fn state_dimension(&self) -> usize {
        2  // (X, Y) plane
    }

    fn n_elements(&self) -> usize {
        self.trajectory.len()
    }

    fn distance_matrix(&self) -> Array2<f64> {
        self.trajectory_distance_matrix()
    }

    fn order_parameter(&self) -> f64 {
        self.compute_amplitude()
    }

    fn suggested_epsilon(&self, k: usize) -> f64 {
        let dist = self.distance_matrix();
        let n = dist.nrows();

        if n < k + 1 {
            return 0.5;
        }

        let mut knn_distances = Vec::with_capacity(n);
        for i in 0..n {
            let mut distances: Vec<f64> = (0..n)
                .filter(|&j| j != i)
                .map(|j| dist[[i, j]])
                .collect();
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if distances.len() >= k {
                knn_distances.push(distances[k - 1]);
            }
        }

        if knn_distances.is_empty() {
            return 0.5;
        }

        knn_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        knn_distances[knn_distances.len() / 2]
    }
}

impl Controllable for Brusselator {
    type Parameter = f64;

    fn set_parameter(&mut self, param: f64) {
        self.set_b(param);
    }

    fn get_parameter(&self) -> f64 {
        self.get_b()
    }

    fn ramp_parameter(&mut self, target: f64, rate: f64, steps_per_increment: usize) {
        while (self.b - target).abs() > rate {
            self.run(steps_per_increment);
            if self.b < target {
                self.b += rate;
            } else {
                self.b -= rate;
            }
        }
        self.b = target;
    }
}

impl Bifurcating for Brusselator {
    fn critical_parameter(&self) -> Option<f64> {
        Some(self.theoretical_b_critical())
    }

    fn bifurcation_type(&self) -> &'static str {
        "Hopf (supercritical)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brusselator_initialization() {
        let system = Brusselator::standard(1.5);
        assert!((system.a - 1.0).abs() < 1e-10);
        assert!((system.b - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_brusselator_fixed_point() {
        // Below bifurcation, should stay near fixed point
        let mut system = Brusselator::standard(1.5);  // B < B_c = 2
        system.perturb(0.1);
        system.run(2000);

        let amp = system.compute_amplitude();
        assert!(amp < 0.5, "Expected steady state, got amplitude {}", amp);
    }

    #[test]
    fn test_brusselator_oscillation() {
        // Above bifurcation, should oscillate
        let mut system = Brusselator::standard(3.0);  // B > B_c = 2, further into oscillation
        system.perturb(0.1);
        system.run(20000);  // More time for limit cycle to develop

        let amp = system.compute_amplitude();
        // Brusselator has smaller amplitude oscillations than FHN
        assert!(amp > 0.1, "Expected oscillation, got amplitude {}", amp);
    }

    #[test]
    fn test_brusselator_critical_b() {
        let system = Brusselator::standard(2.0);
        let b_c = system.theoretical_b_critical();
        assert!((b_c - 2.0).abs() < 0.01, "Expected B_c = 2, got {}", b_c);
    }

    #[test]
    fn test_brusselator_hopf_transition() {
        let mut system = Brusselator::standard(1.5);
        system.perturb(0.1);

        // Below critical
        system.run(2000);
        let amp_low = system.compute_amplitude();

        // Above critical
        system.set_b(2.5);
        system.clear_trajectory();
        system.run(5000);
        let amp_high = system.compute_amplitude();

        assert!(amp_high > amp_low * 2.0,
            "Expected Hopf transition: amp_low={}, amp_high={}", amp_low, amp_high);
    }
}
