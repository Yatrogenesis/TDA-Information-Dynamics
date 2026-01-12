//! Dynamical System Traits: Standardized API for TDA Analysis
//!
//! This module defines traits that all dynamical systems must implement
//! to work with the TDA-CUSUM framework. This enables:
//!
//! - Automatic VR epsilon calibration
//! - Unified interface for analysis pipelines
//! - Easy addition of new systems
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    DynamicalSystem Trait                     │
//! ├─────────────────────────────────────────────────────────────┤
//! │  + step()              - Advance system by dt               │
//! │  + run(n)              - Run n steps                        │
//! │  + state_dimension()   - Dimension of state space           │
//! │  + distance_matrix()   - Pairwise distances for TDA         │
//! │  + order_parameter()   - System-specific transition metric  │
//! │  + suggested_epsilon() - Auto-calibrated VR epsilon         │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use ndarray::Array2;

/// Trait for dynamical systems compatible with TDA-CUSUM analysis
pub trait DynamicalSystem {
    /// State type for this system
    type State;

    /// Advance system by one timestep
    fn step(&mut self);

    /// Run multiple timesteps
    fn run(&mut self, n_steps: usize) {
        for _ in 0..n_steps {
            self.step();
        }
    }

    /// Get current state snapshot
    fn state(&self) -> Self::State;

    /// Dimension of state space (for embedding)
    fn state_dimension(&self) -> usize;

    /// Number of elements/particles/oscillators
    fn n_elements(&self) -> usize;

    /// Compute pairwise distance matrix for TDA
    ///
    /// This is the key method for VR complex construction.
    /// Different systems may use different metrics:
    /// - Euclidean for point clouds
    /// - Geodesic for circular/toroidal systems
    fn distance_matrix(&self) -> Array2<f64>;

    /// System-specific order parameter for transition detection
    ///
    /// Examples:
    /// - Kuramoto: r (synchronization)
    /// - LJ: Potential energy or Q6
    /// - FHN: Amplitude of oscillation
    fn order_parameter(&self) -> f64;

    /// Suggested VR epsilon based on current state
    ///
    /// Auto-calibration using k-th nearest neighbor heuristic:
    /// epsilon ≈ median of k-th NN distances
    fn suggested_epsilon(&self, k: usize) -> f64 {
        let dist = self.distance_matrix();
        let n = dist.nrows();

        if n < k + 1 {
            // Fallback for small systems
            return dist.iter().cloned().fold(0.0, f64::max) / 2.0;
        }

        // For each point, find k-th nearest neighbor distance
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

        // Return median of k-NN distances
        knn_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = knn_distances.len() / 2;

        if knn_distances.len() % 2 == 0 && mid > 0 {
            (knn_distances[mid - 1] + knn_distances[mid]) / 2.0
        } else {
            knn_distances[mid]
        }
    }

    /// Suggested number of VR filtration steps
    fn suggested_vr_steps(&self) -> usize {
        // Heuristic: 20-50 steps usually sufficient
        30
    }
}

/// Trait for systems with controllable parameters
pub trait Controllable {
    /// Parameter type (coupling, temperature, current, etc.)
    type Parameter;

    /// Set control parameter
    fn set_parameter(&mut self, param: Self::Parameter);

    /// Get current parameter value
    fn get_parameter(&self) -> Self::Parameter;

    /// Ramp parameter gradually
    fn ramp_parameter(&mut self, target: Self::Parameter, rate: f64, steps_per_increment: usize);
}

/// Trait for systems exhibiting bifurcations
pub trait Bifurcating: DynamicalSystem + Controllable {
    /// Theoretical critical parameter value (if known)
    fn critical_parameter(&self) -> Option<f64> {
        None
    }

    /// Name of the bifurcation type
    fn bifurcation_type(&self) -> &'static str;
}

/// Configuration for auto-calibration
#[derive(Debug, Clone)]
pub struct TDAConfig {
    /// Maximum VR epsilon
    pub max_epsilon: f64,
    /// Number of filtration steps
    pub n_steps: usize,
    /// Scale index for Betti measurement
    pub measurement_scale: usize,
    /// k for k-NN epsilon estimation
    pub knn_k: usize,
}

impl Default for TDAConfig {
    fn default() -> Self {
        Self {
            max_epsilon: 1.0,
            n_steps: 30,
            measurement_scale: 15,
            knn_k: 5,
        }
    }
}

impl TDAConfig {
    /// Auto-configure based on system
    pub fn auto_configure<S: DynamicalSystem>(system: &S) -> Self {
        let suggested_eps = system.suggested_epsilon(5);
        let n_steps = system.suggested_vr_steps();

        Self {
            max_epsilon: suggested_eps * 2.0,  // Allow headroom
            n_steps,
            measurement_scale: n_steps / 2,
            knn_k: 5,
        }
    }
}

