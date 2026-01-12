//! Lennard-Jones System: Classical Liquid-Solid Model
//!
//! The Lennard-Jones potential is:
//!
//!   V(r) = 4ε[(σ/r)¹² - (σ/r)⁶]
//!
//! This system exhibits a first-order phase transition from
//! liquid to solid (crystallization) upon cooling.
//!
//! ## Topological Signatures
//!
//! - **Liquid**: β₀ fluctuates, low β₁, high entropy
//! - **Solid (FCC)**: β₀ ≈ 1, structured β₁, low entropy
//! - **Glass**: Intermediate, frustrated β₁

use ndarray::Array2;
use rand_distr::{Normal, Distribution};

/// State of Lennard-Jones system
#[derive(Debug, Clone)]
pub struct LJState {
    /// Particle positions [N, 3]
    pub positions: Array2<f64>,
    /// Particle velocities [N, 3]
    pub velocities: Array2<f64>,
    /// Box size (cubic)
    pub box_size: f64,
    /// Current temperature
    pub temperature: f64,
    /// Total energy
    pub total_energy: f64,
    /// Potential energy
    pub potential_energy: f64,
    /// Kinetic energy
    pub kinetic_energy: f64,
    /// Simulation time
    pub time: f64,
}

/// Lennard-Jones simulation system
pub struct LennardJonesSystem {
    /// Number of particles
    n_particles: usize,
    /// LJ sigma parameter (used in physics equations)
    #[allow(dead_code)]
    sigma: f64,
    /// LJ epsilon parameter
    epsilon: f64,
    /// Cutoff radius
    cutoff: f64,
    /// Integration timestep
    dt: f64,
    /// Box size
    box_size: f64,
    /// Current state
    state: LJState,
    /// Forces array
    forces: Array2<f64>,
}

impl LennardJonesSystem {
    /// Create new LJ system
    ///
    /// # Arguments
    /// * `n_particles` - Number of particles (should be N³ for FCC)
    /// * `density` - Number density ρ* = N/V
    /// * `temperature` - Initial temperature T*
    pub fn new(n_particles: usize, density: f64, temperature: f64) -> Self {
        let sigma = 1.0;
        let epsilon = 1.0;

        // Box size from density
        let volume = n_particles as f64 / density;
        let box_size = volume.powf(1.0 / 3.0);

        // Cutoff at 2.5σ
        let cutoff = 2.5 * sigma;

        // Initialize on FCC lattice
        let positions = Self::fcc_lattice(n_particles, box_size);

        // Initialize velocities from Maxwell-Boltzmann
        let velocities = Self::maxwell_boltzmann(n_particles, temperature);

        let state = LJState {
            positions: positions.clone(),
            velocities,
            box_size,
            temperature,
            total_energy: 0.0,
            potential_energy: 0.0,
            kinetic_energy: 0.0,
            time: 0.0,
        };

        let forces = Array2::zeros((n_particles, 3));

        let mut system = Self {
            n_particles,
            sigma,
            epsilon,
            cutoff,
            dt: 0.001,
            box_size,
            state,
            forces,
        };

        system.compute_forces();
        system.compute_energies();

        system
    }

    /// Initialize FCC lattice
    fn fcc_lattice(n: usize, box_size: f64) -> Array2<f64> {
        let n_cells = ((n as f64 / 4.0).powf(1.0 / 3.0).ceil() as usize).max(1);
        let cell_size = box_size / n_cells as f64;

        let mut positions = Array2::zeros((n, 3));

        // FCC basis
        let basis = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ];

        let mut idx = 0;
        'outer: for ix in 0..n_cells {
            for iy in 0..n_cells {
                for iz in 0..n_cells {
                    for b in &basis {
                        if idx >= n {
                            break 'outer;
                        }
                        positions[[idx, 0]] = (ix as f64 + b[0]) * cell_size;
                        positions[[idx, 1]] = (iy as f64 + b[1]) * cell_size;
                        positions[[idx, 2]] = (iz as f64 + b[2]) * cell_size;
                        idx += 1;
                    }
                }
            }
        }

        positions
    }

    /// Initialize Maxwell-Boltzmann velocities
    fn maxwell_boltzmann(n: usize, temperature: f64) -> Array2<f64> {
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, temperature.sqrt()).unwrap();

        let mut velocities = Array2::zeros((n, 3));

        // Random velocities
        for i in 0..n {
            for d in 0..3 {
                velocities[[i, d]] = normal.sample(&mut rng);
            }
        }

        // Remove center of mass motion
        let mut com = [0.0; 3];
        for i in 0..n {
            for d in 0..3 {
                com[d] += velocities[[i, d]];
            }
        }
        for d in 0..3 {
            com[d] /= n as f64;
        }
        for i in 0..n {
            for d in 0..3 {
                velocities[[i, d]] -= com[d];
            }
        }

        velocities
    }

    /// Minimum image distance (periodic BC)
    fn min_image(&self, dr: f64) -> f64 {
        let mut d = dr;
        if d > self.box_size / 2.0 {
            d -= self.box_size;
        } else if d < -self.box_size / 2.0 {
            d += self.box_size;
        }
        d
    }

    /// Compute forces
    fn compute_forces(&mut self) {
        self.forces.fill(0.0);
        let n = self.n_particles;
        let cutoff2 = self.cutoff * self.cutoff;

        for i in 0..n {
            for j in i + 1..n {
                // Distance vector with minimum image
                let mut dr = [0.0; 3];
                let mut r2 = 0.0;

                for d in 0..3 {
                    dr[d] = self.min_image(
                        self.state.positions[[i, d]] - self.state.positions[[j, d]],
                    );
                    r2 += dr[d] * dr[d];
                }

                if r2 < cutoff2 && r2 > 0.0 {
                    let r2_inv = 1.0 / r2;
                    let r6_inv = r2_inv * r2_inv * r2_inv;
                    let r12_inv = r6_inv * r6_inv;

                    // Force magnitude: f = 24ε/r * [2(σ/r)¹² - (σ/r)⁶]
                    let f_mag = 24.0 * self.epsilon * r2_inv * (2.0 * r12_inv - r6_inv);

                    for d in 0..3 {
                        let f = f_mag * dr[d];
                        self.forces[[i, d]] += f;
                        self.forces[[j, d]] -= f;
                    }
                }
            }
        }
    }

    /// Compute energies
    fn compute_energies(&mut self) {
        let n = self.n_particles;
        let cutoff2 = self.cutoff * self.cutoff;

        // Potential energy
        let mut pe = 0.0;
        for i in 0..n {
            for j in i + 1..n {
                let mut r2 = 0.0;
                for d in 0..3 {
                    let dr = self.min_image(
                        self.state.positions[[i, d]] - self.state.positions[[j, d]],
                    );
                    r2 += dr * dr;
                }

                if r2 < cutoff2 && r2 > 0.0 {
                    let r6_inv = 1.0 / (r2 * r2 * r2);
                    let r12_inv = r6_inv * r6_inv;
                    pe += 4.0 * self.epsilon * (r12_inv - r6_inv);
                }
            }
        }

        // Kinetic energy
        let mut ke = 0.0;
        for i in 0..n {
            for d in 0..3 {
                ke += 0.5 * self.state.velocities[[i, d]].powi(2);
            }
        }

        self.state.potential_energy = pe;
        self.state.kinetic_energy = ke;
        self.state.total_energy = pe + ke;
        self.state.temperature = 2.0 * ke / (3.0 * (n - 1) as f64);
    }

    /// Velocity Verlet integration step
    pub fn step(&mut self) {
        let dt = self.dt;
        let n = self.n_particles;

        // First half-kick
        for i in 0..n {
            for d in 0..3 {
                self.state.velocities[[i, d]] += 0.5 * dt * self.forces[[i, d]];
            }
        }

        // Drift
        for i in 0..n {
            for d in 0..3 {
                self.state.positions[[i, d]] += dt * self.state.velocities[[i, d]];
                // Periodic BC
                self.state.positions[[i, d]] =
                    self.state.positions[[i, d]].rem_euclid(self.box_size);
            }
        }

        // Recompute forces
        self.compute_forces();

        // Second half-kick
        for i in 0..n {
            for d in 0..3 {
                self.state.velocities[[i, d]] += 0.5 * dt * self.forces[[i, d]];
            }
        }

        self.compute_energies();
        self.state.time += dt;
    }

    /// Run multiple steps
    pub fn run(&mut self, n_steps: usize) {
        for _ in 0..n_steps {
            self.step();
        }
    }

    /// Apply velocity rescaling thermostat
    pub fn thermostat(&mut self, target_temp: f64) {
        let current_temp = self.state.temperature;
        if current_temp > 0.0 {
            let scale = (target_temp / current_temp).sqrt();
            self.state.velocities *= scale;
            self.compute_energies();
        }
    }

    /// Cool the system (for crystallization)
    pub fn cool(&mut self, target_temp: f64, cooling_rate: f64, steps_per_temp: usize) {
        let mut temp = self.state.temperature;
        while temp > target_temp {
            self.thermostat(temp);
            self.run(steps_per_temp);
            temp *= 1.0 - cooling_rate;
        }
        self.thermostat(target_temp);
    }

    /// Get current state
    pub fn state(&self) -> &LJState {
        &self.state
    }

    /// Get positions for TDA analysis
    pub fn positions(&self) -> &Array2<f64> {
        &self.state.positions
    }

    /// Compute pairwise distance matrix
    pub fn distance_matrix(&self) -> Array2<f64> {
        let n = self.n_particles;
        let mut dist = Array2::zeros((n, n));

        for i in 0..n {
            for j in i + 1..n {
                let mut r2 = 0.0;
                for d in 0..3 {
                    let dr = self.min_image(
                        self.state.positions[[i, d]] - self.state.positions[[j, d]],
                    );
                    r2 += dr * dr;
                }
                let r = r2.sqrt();
                dist[[i, j]] = r;
                dist[[j, i]] = r;
            }
        }

        dist
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lj_initialization() {
        let system = LennardJonesSystem::new(32, 0.8, 1.0);
        assert!(system.state.positions.nrows() == 32);
        assert!(system.state.temperature > 0.0);
    }

    #[test]
    fn test_lj_energy_conservation() {
        let mut system = LennardJonesSystem::new(32, 0.8, 1.0);
        let e0 = system.state.total_energy;

        system.run(100);
        let e1 = system.state.total_energy;

        // Energy should be roughly conserved (within 1%)
        let rel_error = ((e1 - e0) / e0.abs()).abs();
        assert!(rel_error < 0.01, "Energy drift: {:.2}%", rel_error * 100.0);
    }
}
