//! FitzHugh-Nagumo Model: Excitable Neuron Dynamics
//!
//! The FitzHugh-Nagumo model is a 2D simplification of Hodgkin-Huxley,
//! capturing the essential dynamics of neuronal excitability:
//!
//!   dv/dt = v - v³/3 - w + I_ext
//!   dw/dt = ε(v + a - b·w)
//!
//! where:
//! - v: membrane potential (fast variable)
//! - w: recovery variable (slow variable)
//! - ε: timescale separation (small, ~0.08)
//! - a, b: shape parameters
//! - I_ext: external current (control parameter)
//!
//! ## Hopf Bifurcation
//!
//! At critical current I_c, the system transitions from:
//! - I < I_c: Stable fixed point (resting state)
//! - I > I_c: Stable limit cycle (periodic spiking)
//!
//! ## Topological Signatures
//!
//! - **Resting**: Trajectory collapses to point → β₀ = 1, β₁ = 0
//! - **Spiking**: Trajectory forms loop → β₀ = 1, β₁ = 1 (cycle born!)
//!
//! The CUSUM should detect the β₁ increase BEFORE the limit cycle
//! is fully established.
//!
//! ## References
//!
//! - FitzHugh, R. (1961). Impulses and physiological states in theoretical
//!   models of nerve membrane. Biophysical Journal, 1(6), 445-466.
//! - Nagumo, J., et al. (1962). An active pulse transmission line
//!   simulating nerve axon. Proceedings of the IRE, 50(10), 2061-2070.

use ndarray::{Array1, Array2};
use rand_distr::{Normal, Distribution};
use super::traits::{DynamicalSystem, Controllable, Bifurcating};

/// State of a single FHN neuron
#[derive(Debug, Clone)]
pub struct FHNNeuronState {
    /// Membrane potential
    pub v: f64,
    /// Recovery variable
    pub w: f64,
}

/// State of FHN network
#[derive(Debug, Clone)]
pub struct FHNState {
    /// All neuron states (v, w pairs)
    pub neurons: Vec<FHNNeuronState>,
    /// External current
    pub i_ext: f64,
    /// Coupling strength
    pub coupling: f64,
    /// Simulation time
    pub time: f64,
    /// Oscillation amplitude (order parameter)
    pub amplitude: f64,
}

/// FitzHugh-Nagumo system (network of coupled neurons)
pub struct FitzHughNagumoSystem {
    /// Number of neurons
    n_neurons: usize,
    /// Membrane potentials
    v: Array1<f64>,
    /// Recovery variables
    w: Array1<f64>,
    /// Timescale separation ε
    epsilon: f64,
    /// Parameter a
    a: f64,
    /// Parameter b
    b: f64,
    /// External current (bifurcation parameter)
    i_ext: f64,
    /// Coupling strength (for network)
    coupling: f64,
    /// Integration timestep
    dt: f64,
    /// Current time
    time: f64,
    /// Trajectory history for TDA (last N points)
    trajectory: Vec<(f64, f64)>,
    /// Maximum trajectory length
    max_trajectory: usize,
}

impl FitzHughNagumoSystem {
    /// Create new FHN system with standard parameters
    ///
    /// # Arguments
    /// * `n_neurons` - Number of coupled neurons (1 for single neuron)
    /// * `i_ext` - External current (control parameter)
    /// * `coupling` - Coupling strength between neurons
    pub fn new(n_neurons: usize, i_ext: f64, coupling: f64) -> Self {
        // Standard FHN parameters (Hopf bifurcation around I ≈ 0.35)
        let epsilon = 0.08;
        let a = 0.7;
        let b = 0.8;

        // Initialize near the resting state
        let v = Array1::from_elem(n_neurons, -1.0);
        let w = Array1::from_elem(n_neurons, -0.5);

        Self {
            n_neurons,
            v,
            w,
            epsilon,
            a,
            b,
            i_ext,
            coupling,
            dt: 0.05,
            time: 0.0,
            trajectory: Vec::new(),
            max_trajectory: 500,  // Store last 500 points for TDA
        }
    }

    /// Create single neuron
    pub fn single(i_ext: f64) -> Self {
        Self::new(1, i_ext, 0.0)
    }

    /// Create network with specified parameters
    pub fn with_params(
        n_neurons: usize,
        i_ext: f64,
        coupling: f64,
        epsilon: f64,
        a: f64,
        b: f64,
    ) -> Self {
        let v = Array1::from_elem(n_neurons, -1.0);
        let w = Array1::from_elem(n_neurons, -0.5);

        Self {
            n_neurons,
            v,
            w,
            epsilon,
            a,
            b,
            i_ext,
            coupling,
            dt: 0.05,
            time: 0.0,
            trajectory: Vec::new(),
            max_trajectory: 500,
        }
    }

    /// Add noise to initial conditions
    pub fn perturb(&mut self, noise_std: f64) {
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, noise_std).unwrap();

        for i in 0..self.n_neurons {
            self.v[i] += normal.sample(&mut rng);
            self.w[i] += normal.sample(&mut rng);
        }
    }

    /// FHN dynamics for single neuron
    fn derivatives(&self, v: f64, w: f64, coupling_term: f64) -> (f64, f64) {
        // dv/dt = v - v³/3 - w + I_ext + coupling
        let dv = v - v.powi(3) / 3.0 - w + self.i_ext + coupling_term;
        // dw/dt = ε(v + a - b·w)
        let dw = self.epsilon * (v + self.a - self.b * w);
        (dv, dw)
    }

    /// RK4 integration step
    fn rk4_step(&mut self) {
        let dt = self.dt;
        let n = self.n_neurons;

        // Store old values
        let v_old = self.v.clone();
        let w_old = self.w.clone();

        // Compute coupling terms (diffusive coupling)
        let coupling_term = |v_arr: &Array1<f64>, i: usize| -> f64 {
            if n == 1 || self.coupling == 0.0 {
                return 0.0;
            }
            let v_mean: f64 = v_arr.iter().sum::<f64>() / n as f64;
            self.coupling * (v_mean - v_arr[i])
        };

        // k1
        let mut k1_v = Array1::zeros(n);
        let mut k1_w = Array1::zeros(n);
        for i in 0..n {
            let (dv, dw) = self.derivatives(v_old[i], w_old[i], coupling_term(&v_old, i));
            k1_v[i] = dv;
            k1_w[i] = dw;
        }

        // k2
        let v_mid1 = &v_old + &(&k1_v * (dt / 2.0));
        let w_mid1 = &w_old + &(&k1_w * (dt / 2.0));
        let mut k2_v = Array1::zeros(n);
        let mut k2_w = Array1::zeros(n);
        for i in 0..n {
            let (dv, dw) = self.derivatives(v_mid1[i], w_mid1[i], coupling_term(&v_mid1, i));
            k2_v[i] = dv;
            k2_w[i] = dw;
        }

        // k3
        let v_mid2 = &v_old + &(&k2_v * (dt / 2.0));
        let w_mid2 = &w_old + &(&k2_w * (dt / 2.0));
        let mut k3_v = Array1::zeros(n);
        let mut k3_w = Array1::zeros(n);
        for i in 0..n {
            let (dv, dw) = self.derivatives(v_mid2[i], w_mid2[i], coupling_term(&v_mid2, i));
            k3_v[i] = dv;
            k3_w[i] = dw;
        }

        // k4
        let v_end = &v_old + &(&k3_v * dt);
        let w_end = &w_old + &(&k3_w * dt);
        let mut k4_v = Array1::zeros(n);
        let mut k4_w = Array1::zeros(n);
        for i in 0..n {
            let (dv, dw) = self.derivatives(v_end[i], w_end[i], coupling_term(&v_end, i));
            k4_v[i] = dv;
            k4_w[i] = dw;
        }

        // Update
        self.v = &v_old + &((&k1_v + &(&k2_v * 2.0) + &(&k3_v * 2.0) + &k4_v) * (dt / 6.0));
        self.w = &w_old + &((&k1_w + &(&k2_w * 2.0) + &(&k3_w * 2.0) + &k4_w) * (dt / 6.0));

        self.time += dt;

        // Store trajectory point (mean v, mean w)
        let v_mean = self.v.iter().sum::<f64>() / n as f64;
        let w_mean = self.w.iter().sum::<f64>() / n as f64;
        self.trajectory.push((v_mean, w_mean));

        // Trim trajectory if too long
        if self.trajectory.len() > self.max_trajectory {
            self.trajectory.remove(0);
        }
    }

    /// Set external current
    pub fn set_current(&mut self, i_ext: f64) {
        self.i_ext = i_ext;
    }

    /// Get external current
    pub fn current(&self) -> f64 {
        self.i_ext
    }

    /// Get membrane potentials
    pub fn potentials(&self) -> &Array1<f64> {
        &self.v
    }

    /// Get recovery variables
    pub fn recovery(&self) -> &Array1<f64> {
        &self.w
    }

    /// Compute oscillation amplitude (peak-to-peak voltage)
    pub fn compute_amplitude(&self) -> f64 {
        if self.trajectory.len() < 10 {
            return 0.0;
        }

        let recent: Vec<f64> = self.trajectory.iter()
            .rev()
            .take(100)
            .map(|(v, _)| *v)
            .collect();

        let v_max = recent.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let v_min = recent.iter().cloned().fold(f64::INFINITY, f64::min);

        v_max - v_min
    }

    /// Get trajectory as point cloud for TDA
    pub fn trajectory_points(&self) -> Array2<f64> {
        let n = self.trajectory.len();
        let mut points = Array2::zeros((n, 2));

        for (i, (v, w)) in self.trajectory.iter().enumerate() {
            points[[i, 0]] = *v;
            points[[i, 1]] = *w;
        }

        points
    }

    /// Get network state as point cloud (v_i, w_i) for each neuron
    pub fn network_points(&self) -> Array2<f64> {
        let n = self.n_neurons;
        let mut points = Array2::zeros((n, 2));

        for i in 0..n {
            points[[i, 0]] = self.v[i];
            points[[i, 1]] = self.w[i];
        }

        points
    }

    /// Euclidean distance matrix for trajectory
    pub fn trajectory_distance_matrix(&self) -> Array2<f64> {
        let points = self.trajectory_points();
        let n = points.nrows();
        let mut dist = Array2::zeros((n, n));

        for i in 0..n {
            for j in i + 1..n {
                let dv = points[[i, 0]] - points[[j, 0]];
                let dw = points[[i, 1]] - points[[j, 1]];
                let d = (dv * dv + dw * dw).sqrt();
                dist[[i, j]] = d;
                dist[[j, i]] = d;
            }
        }

        dist
    }

    /// Clear trajectory history
    pub fn clear_trajectory(&mut self) {
        self.trajectory.clear();
    }

    /// Theoretical critical current for Hopf bifurcation
    ///
    /// For standard parameters (a=0.7, b=0.8, ε=0.08):
    /// I_c ≈ 1 - 2a/3 + a³/27 ≈ 0.3315 (approximate)
    pub fn theoretical_i_critical(&self) -> f64 {
        // This is an approximation - exact value depends on parameters
        let a = self.a;
        1.0 - 2.0 * a / 3.0 + a.powi(3) / 27.0
    }
}

impl DynamicalSystem for FitzHughNagumoSystem {
    type State = FHNState;

    fn step(&mut self) {
        self.rk4_step();
    }

    fn state(&self) -> FHNState {
        let neurons = (0..self.n_neurons)
            .map(|i| FHNNeuronState {
                v: self.v[i],
                w: self.w[i],
            })
            .collect();

        FHNState {
            neurons,
            i_ext: self.i_ext,
            coupling: self.coupling,
            time: self.time,
            amplitude: self.compute_amplitude(),
        }
    }

    fn state_dimension(&self) -> usize {
        2  // (v, w) plane
    }

    fn n_elements(&self) -> usize {
        // For TDA, we use trajectory points, not neurons
        self.trajectory.len()
    }

    fn distance_matrix(&self) -> Array2<f64> {
        self.trajectory_distance_matrix()
    }

    fn order_parameter(&self) -> f64 {
        self.compute_amplitude()
    }

    fn suggested_epsilon(&self, k: usize) -> f64 {
        // For FHN, the attractor lives in a bounded region
        // v ∈ [-2, 2], w ∈ [-1, 2] roughly
        // A good epsilon is ~0.1 to 0.3
        let dist = self.distance_matrix();
        let n = dist.nrows();

        if n < k + 1 {
            return 0.2;  // Default for FHN
        }

        // k-NN heuristic
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
        let mid = knn_distances.len() / 2;
        knn_distances[mid]
    }
}

impl Controllable for FitzHughNagumoSystem {
    type Parameter = f64;

    fn set_parameter(&mut self, param: f64) {
        self.i_ext = param;
    }

    fn get_parameter(&self) -> f64 {
        self.i_ext
    }

    fn ramp_parameter(&mut self, target: f64, rate: f64, steps_per_increment: usize) {
        while (self.i_ext - target).abs() > rate {
            self.run(steps_per_increment);
            if self.i_ext < target {
                self.i_ext += rate;
            } else {
                self.i_ext -= rate;
            }
        }
        self.i_ext = target;
    }
}

impl Bifurcating for FitzHughNagumoSystem {
    fn critical_parameter(&self) -> Option<f64> {
        Some(self.theoretical_i_critical())
    }

    fn bifurcation_type(&self) -> &'static str {
        "Hopf (supercritical)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fhn_initialization() {
        let system = FitzHughNagumoSystem::single(0.0);
        assert_eq!(system.n_neurons, 1);
        assert_eq!(system.i_ext, 0.0);
    }

    #[test]
    fn test_fhn_resting_state() {
        // Below bifurcation, should stay near fixed point
        let mut system = FitzHughNagumoSystem::single(0.0);
        system.run(1000);

        let amp = system.compute_amplitude();
        assert!(amp < 0.5, "Expected resting state, got amplitude {}", amp);
    }

    #[test]
    fn test_fhn_oscillation() {
        // Above bifurcation, should oscillate
        let mut system = FitzHughNagumoSystem::single(0.5);
        system.run(2000);

        let amp = system.compute_amplitude();
        assert!(amp > 1.0, "Expected oscillation, got amplitude {}", amp);
    }

    #[test]
    fn test_fhn_hopf_transition() {
        // Test transition from resting to oscillating
        let mut system = FitzHughNagumoSystem::single(0.0);

        // Below critical
        system.set_current(0.2);
        system.run(1000);
        let amp_low = system.compute_amplitude();

        // Above critical
        system.set_current(0.5);
        system.clear_trajectory();
        system.run(2000);
        let amp_high = system.compute_amplitude();

        assert!(amp_high > amp_low * 2.0,
            "Expected Hopf transition: amp_low={}, amp_high={}", amp_low, amp_high);
    }

    #[test]
    fn test_fhn_network() {
        // Test coupled network
        let mut system = FitzHughNagumoSystem::new(10, 0.5, 0.5);
        system.perturb(0.1);
        system.run(2000);

        // Should still oscillate with coupling
        let amp = system.compute_amplitude();
        assert!(amp > 0.5, "Network should oscillate, got amplitude {}", amp);
    }

    #[test]
    fn test_fhn_suggested_epsilon() {
        let mut system = FitzHughNagumoSystem::single(0.5);
        system.run(500);

        let eps = system.suggested_epsilon(5);
        assert!(eps > 0.01 && eps < 1.0, "Unexpected epsilon: {}", eps);
    }
}
