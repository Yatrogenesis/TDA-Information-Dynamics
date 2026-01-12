//! XY Model: Classical 2D Spin System
//!
//! The XY model consists of planar spins on a 2D lattice:
//!   H = -J Σ<ij> cos(θᵢ - θⱼ)
//!
//! where θᵢ ∈ [0, 2π) is the spin angle at site i.
//!
//! ## Berezinskii-Kosterlitz-Thouless (BKT) Transition
//!
//! The XY model exhibits a unique topological phase transition:
//! - T < T_BKT: Vortex-antivortex pairs are bound, quasi-long-range order
//! - T > T_BKT: Free vortices proliferate, exponential decay of correlations
//!
//! For a square lattice with J=1: T_BKT ≈ 0.893
//!
//! ## Topological Signatures
//!
//! Vortices are topological defects where the phase winds by ±2π
//! around a closed path. They can be detected via:
//! - Winding number calculation
//! - TDA: H1 features in the phase field
//!
//! ## References
//!
//! - Kosterlitz, J. M., & Thouless, D. J. (1973). Ordering, metastability
//!   and phase transitions in two-dimensional systems. J. Phys. C.
//! - Berezinskii, V. L. (1972). Destruction of long-range order in
//!   one-dimensional and two-dimensional systems. Sov. Phys. JETP.

use ndarray::Array2;
use rand::Rng;
use std::f64::consts::PI;

use super::traits::{DynamicalSystem, Controllable, Bifurcating};

/// State of the XY model
#[derive(Debug, Clone)]
pub struct XYState {
    /// Spin angles θᵢ ∈ [0, 2π)
    pub spins: Array2<f64>,
    /// Temperature
    pub temperature: f64,
    /// Order parameter |<e^{iθ}>|
    pub magnetization: f64,
    /// Vortex density
    pub vortex_density: f64,
    /// Energy per spin
    pub energy: f64,
    /// Monte Carlo step count
    pub mc_steps: usize,
}

/// XY Model on a square lattice
pub struct XYModel {
    /// Lattice size (L x L)
    size: usize,
    /// Spin angles
    spins: Array2<f64>,
    /// Temperature (control parameter)
    temperature: f64,
    /// Coupling constant
    coupling: f64,
    /// Monte Carlo step count
    mc_steps: usize,
    /// History for TDA (stores magnetization vectors)
    history: Vec<(f64, f64)>,
    /// Maximum history length
    max_history: usize,
}

impl XYModel {
    /// Create new XY model with random initial spins
    pub fn new(size: usize, temperature: f64) -> Self {
        let mut rng = rand::rng();
        let mut spins = Array2::zeros((size, size));

        // Random initial configuration
        for i in 0..size {
            for j in 0..size {
                spins[[i, j]] = rng.random::<f64>() * 2.0 * PI;
            }
        }

        Self {
            size,
            spins,
            temperature,
            coupling: 1.0,
            mc_steps: 0,
            history: Vec::new(),
            max_history: 500,
        }
    }

    /// Create with aligned initial spins (ordered state)
    pub fn aligned(size: usize, temperature: f64) -> Self {
        let spins = Array2::zeros((size, size));

        Self {
            size,
            spins,
            temperature,
            coupling: 1.0,
            mc_steps: 0,
            history: Vec::new(),
            max_history: 500,
        }
    }

    /// Set temperature
    pub fn set_temperature(&mut self, t: f64) {
        self.temperature = t.max(0.001);  // Avoid division by zero
    }

    /// Get lattice size
    pub fn lattice_size(&self) -> usize {
        self.size
    }

    /// Get spin angles
    pub fn spins(&self) -> &Array2<f64> {
        &self.spins
    }

    /// Compute energy
    pub fn compute_energy(&self) -> f64 {
        let l = self.size;
        let mut energy = 0.0;

        for i in 0..l {
            for j in 0..l {
                let theta = self.spins[[i, j]];
                // Right neighbor (periodic)
                let right = self.spins[[i, (j + 1) % l]];
                // Down neighbor (periodic)
                let down = self.spins[[(i + 1) % l, j]];

                energy -= self.coupling * (theta - right).cos();
                energy -= self.coupling * (theta - down).cos();
            }
        }

        energy / (l * l) as f64
    }

    /// Compute magnetization |<e^{iθ}>|
    pub fn compute_magnetization(&self) -> f64 {
        let l = self.size;
        let n = (l * l) as f64;

        let mut mx = 0.0;
        let mut my = 0.0;

        for i in 0..l {
            for j in 0..l {
                let theta = self.spins[[i, j]];
                mx += theta.cos();
                my += theta.sin();
            }
        }

        (mx * mx + my * my).sqrt() / n
    }

    /// Count vortices in the configuration
    ///
    /// A vortex has winding number ±1 around a plaquette.
    /// Winding = Σ dθ / 2π around the plaquette.
    pub fn count_vortices(&self) -> (usize, usize) {
        let l = self.size;
        let mut n_plus = 0;
        let mut n_minus = 0;

        for i in 0..l {
            for j in 0..l {
                // Four corners of plaquette (i,j), (i,j+1), (i+1,j+1), (i+1,j)
                let t1 = self.spins[[i, j]];
                let t2 = self.spins[[i, (j + 1) % l]];
                let t3 = self.spins[[(i + 1) % l, (j + 1) % l]];
                let t4 = self.spins[[(i + 1) % l, j]];

                // Compute winding
                let w1 = Self::angle_diff(t2 - t1);
                let w2 = Self::angle_diff(t3 - t2);
                let w3 = Self::angle_diff(t4 - t3);
                let w4 = Self::angle_diff(t1 - t4);

                let winding = (w1 + w2 + w3 + w4) / (2.0 * PI);

                if winding > 0.5 {
                    n_plus += 1;
                } else if winding < -0.5 {
                    n_minus += 1;
                }
            }
        }

        (n_plus, n_minus)
    }

    /// Wrap angle difference to [-π, π]
    fn angle_diff(mut dtheta: f64) -> f64 {
        while dtheta > PI {
            dtheta -= 2.0 * PI;
        }
        while dtheta < -PI {
            dtheta += 2.0 * PI;
        }
        dtheta
    }

    /// Single Metropolis sweep (one MC step per spin)
    fn metropolis_sweep(&mut self) {
        let l = self.size;
        let mut rng = rand::rng();
        let beta = 1.0 / self.temperature;

        for _ in 0..(l * l) {
            // Random site
            let i = rng.random_range(0..l);
            let j = rng.random_range(0..l);

            // Current angle
            let old_theta = self.spins[[i, j]];

            // Propose new angle (small perturbation or random)
            let new_theta = if rng.random::<f64>() < 0.9 {
                // Small change
                old_theta + (rng.random::<f64>() - 0.5) * PI * 0.5
            } else {
                // Random
                rng.random::<f64>() * 2.0 * PI
            };

            // Compute energy change
            let delta_e = self.local_energy_change(i, j, old_theta, new_theta);

            // Metropolis criterion
            if delta_e <= 0.0 || rng.random::<f64>() < (-beta * delta_e).exp() {
                self.spins[[i, j]] = Self::wrap_angle(new_theta);
            }
        }

        self.mc_steps += 1;

        // Store magnetization for TDA
        let l = self.size;
        let n = (l * l) as f64;
        let mut mx = 0.0;
        let mut my = 0.0;
        for i in 0..l {
            for j in 0..l {
                let theta = self.spins[[i, j]];
                mx += theta.cos();
                my += theta.sin();
            }
        }
        self.history.push((mx / n, my / n));

        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    /// Compute local energy change for a spin flip
    fn local_energy_change(&self, i: usize, j: usize, old: f64, new: f64) -> f64 {
        let l = self.size;

        // Neighbors (periodic boundary)
        let neighbors = [
            self.spins[[(i + l - 1) % l, j]],  // up
            self.spins[[(i + 1) % l, j]],      // down
            self.spins[[i, (j + l - 1) % l]],  // left
            self.spins[[i, (j + 1) % l]],      // right
        ];

        let mut old_energy = 0.0;
        let mut new_energy = 0.0;

        for &n_theta in &neighbors {
            old_energy -= self.coupling * (old - n_theta).cos();
            new_energy -= self.coupling * (new - n_theta).cos();
        }

        new_energy - old_energy
    }

    /// Wrap angle to [0, 2π)
    fn wrap_angle(mut theta: f64) -> f64 {
        while theta < 0.0 {
            theta += 2.0 * PI;
        }
        while theta >= 2.0 * PI {
            theta -= 2.0 * PI;
        }
        theta
    }

    /// Geodesic distance on S¹
    pub fn circle_distance(theta1: f64, theta2: f64) -> f64 {
        let diff = (theta1 - theta2).abs();
        diff.min(2.0 * PI - diff)
    }

    /// Distance matrix for spins (geodesic on S¹)
    pub fn spin_distance_matrix(&self) -> Array2<f64> {
        let n = self.size * self.size;
        let mut dist = Array2::zeros((n, n));

        for i in 0..n {
            let theta_i = self.spins[[i / self.size, i % self.size]];
            for j in i + 1..n {
                let theta_j = self.spins[[j / self.size, j % self.size]];
                let d = Self::circle_distance(theta_i, theta_j);
                dist[[i, j]] = d;
                dist[[j, i]] = d;
            }
        }

        dist
    }

    /// Distance matrix for magnetization history (Euclidean in R²)
    pub fn history_distance_matrix(&self) -> Array2<f64> {
        let n = self.history.len();
        let mut dist = Array2::zeros((n, n));

        for i in 0..n {
            for j in i + 1..n {
                let (mx_i, my_i) = self.history[i];
                let (mx_j, my_j) = self.history[j];
                let d = ((mx_i - mx_j).powi(2) + (my_i - my_j).powi(2)).sqrt();
                dist[[i, j]] = d;
                dist[[j, i]] = d;
            }
        }

        dist
    }

    /// Clear history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Theoretical BKT transition temperature
    pub fn theoretical_t_bkt() -> f64 {
        // For square lattice with J=1
        // T_BKT ≈ 2 / (π * ln(1 + sqrt(2))) ≈ 0.893
        0.893
    }
}

impl DynamicalSystem for XYModel {
    type State = XYState;

    fn step(&mut self) {
        self.metropolis_sweep();
    }

    fn state(&self) -> XYState {
        let (n_plus, n_minus) = self.count_vortices();
        let n_vortices = (n_plus + n_minus) as f64;
        let n_sites = (self.size * self.size) as f64;

        XYState {
            spins: self.spins.clone(),
            temperature: self.temperature,
            magnetization: self.compute_magnetization(),
            vortex_density: n_vortices / n_sites,
            energy: self.compute_energy(),
            mc_steps: self.mc_steps,
        }
    }

    fn state_dimension(&self) -> usize {
        2  // Each spin lives on S¹, which is 1D, but we use R² embedding
    }

    fn n_elements(&self) -> usize {
        self.history.len()
    }

    fn distance_matrix(&self) -> Array2<f64> {
        self.history_distance_matrix()
    }

    fn order_parameter(&self) -> f64 {
        self.compute_magnetization()
    }

    fn suggested_epsilon(&self, k: usize) -> f64 {
        // For XY model, magnetization lives in unit disk
        // Typical scale is ~0.3-0.5
        let dist = self.history_distance_matrix();
        let n = dist.nrows();

        if n < k + 1 {
            return 0.2;
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
            return 0.2;
        }

        knn_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        knn_distances[knn_distances.len() / 2]
    }
}

impl Controllable for XYModel {
    type Parameter = f64;

    fn set_parameter(&mut self, param: f64) {
        self.set_temperature(param);
    }

    fn get_parameter(&self) -> f64 {
        self.temperature
    }

    fn ramp_parameter(&mut self, target: f64, rate: f64, steps_per_increment: usize) {
        while (self.temperature - target).abs() > rate {
            self.run(steps_per_increment);
            if self.temperature < target {
                self.temperature += rate;
            } else {
                self.temperature -= rate;
            }
        }
        self.temperature = target;
    }
}

impl Bifurcating for XYModel {
    fn critical_parameter(&self) -> Option<f64> {
        Some(Self::theoretical_t_bkt())
    }

    fn bifurcation_type(&self) -> &'static str {
        "BKT (Kosterlitz-Thouless)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xy_initialization() {
        let model = XYModel::new(10, 1.0);
        assert_eq!(model.lattice_size(), 10);
        assert!((model.temperature - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_xy_aligned() {
        let model = XYModel::aligned(10, 0.5);
        // All spins at 0, magnetization should be 1
        let m = model.compute_magnetization();
        assert!((m - 1.0).abs() < 1e-10, "Expected m=1, got {}", m);
    }

    #[test]
    fn test_xy_energy_aligned() {
        let model = XYModel::aligned(10, 0.5);
        let e = model.compute_energy();
        // All aligned: E = -2J per spin (each bond contributes -J, 2 bonds/spin in 2D)
        assert!((e - (-2.0)).abs() < 0.01, "Expected E=-2, got {}", e);
    }

    #[test]
    fn test_xy_mc_step() {
        let mut model = XYModel::new(10, 1.0);
        model.run(100);
        assert_eq!(model.mc_steps, 100);
    }

    #[test]
    fn test_xy_high_temp() {
        // At high temperature, magnetization should be small
        let mut model = XYModel::new(20, 5.0);
        model.run(1000);
        let m = model.compute_magnetization();
        assert!(m < 0.3, "At high T, expected low m, got {}", m);
    }

    #[test]
    fn test_xy_low_temp() {
        // At low temperature, magnetization should be larger
        let mut model = XYModel::aligned(20, 0.3);
        model.run(500);
        let m = model.compute_magnetization();
        assert!(m > 0.5, "At low T, expected high m, got {}", m);
    }

    #[test]
    fn test_xy_vortex_count() {
        let model = XYModel::aligned(10, 1.0);
        let (n_plus, n_minus) = model.count_vortices();
        // Aligned state has no vortices
        assert_eq!(n_plus, 0);
        assert_eq!(n_minus, 0);
    }
}
