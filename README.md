# TDA-Information-Dynamics

**Topological Dynamics of Information: Critical Transition Detection via Persistent Homology and Sequential Analysis (TDA-CUSUM)**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18408154.svg)](https://doi.org/10.5281/zenodo.18408154)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)
[![Tests](https://img.shields.io/badge/tests-35%20passing-brightgreen.svg)]()

## Overview

A Rust framework implementing TDA-CUSUM methodology for early warning detection of phase transitions in complex dynamical systems. This crate provides both fast Euler-approximated and exact persistent homology algorithms, enabling detection of topological precursors before traditional order parameters signal the transition.

### Central Hypothesis

> Phase transitions are preceded by fundamental changes in the **topology of the phase space manifold**, detectable BEFORE conventional order parameters show the transition.

### Key Result

For all validated systems:

```
t_CUSUM < t_physical
```

The topological warning precedes the physical manifestation of the phase transition.

## Validated Systems

| System | Transition Type | CUSUM Detection | Traditional | Early Warning |
|--------|----------------|-----------------|-------------|---------------|
| **Brusselator** | Hopf bifurcation | B = 1.36 | B = 3.08 | +1.72 |
| **FitzHugh-Nagumo** | Hopf (excitable) | I = 0.15 | I = 0.56 | +0.41 |
| **XY Model** | BKT transition | T = 0.73 | T = 0.94 | +0.21 |
| **Kuramoto** | Synchronization | K = 0.60 | K = 1.10 | +0.50 |
| **Lennard-Jones** | Crystallization | 73% rate | - | Variable |
## Theoretical Framework

### 1. Topological Characterization

For a point cloud X(t) representing system microstate, we construct the **Vietoris-Rips filtration** VR_e(X) and compute **persistent homology**:

- **beta_0**: Connected components (clustering)
- **beta_1**: 1-dimensional cycles (ring structures)
- **beta_2**: 2-dimensional voids (cavities)

### 2. Persistence Methods

This crate provides two complementary approaches:

**Euler Approximation (Fast)**
```
beta_1 = E - V + beta_0 - F
```
- O(n^2) complexity
- Counts features at each scale
- Suitable for real-time monitoring

**Exact Algorithm (Standard)**
- Boundary matrix reduction
- True birth/death pairs
- Matches Python/Ripser output
- Required for entropy-based detection

### 3. Topological Entropy

Given persistence diagram D = {(b_i, d_i)}, the **persistent entropy** is:

```
H_P = -sum_i p_i log(p_i)
```

where p_i = l_i / L is the normalized lifetime.

### 4. CUSUM Detection

The Cumulative Sum algorithm monitors topological statistics:

```
C(t) = max(0, C(t-1) + (S(t) - mu_0) - k)
```

Detection occurs when C(t) > h (threshold).

**sigma_min enhancement**: Prevents infinite sensitivity when baseline variance approaches zero.

## Implemented Systems

### 1. Brusselator (Chemical Oscillator)

```
dX/dt = A + X^2*Y - (B+1)*X
dY/dt = B*X - X^2*Y
```

- **Transition**: Hopf bifurcation at B_c = 1 + A^2
- **Detection**: Diameter-based CUSUM

### 2. FitzHugh-Nagumo (Excitable Neuron)

```
dv/dt = v - v^3/3 - w + I_ext
dw/dt = epsilon*(v + a - b*w)
```

- **Transition**: Hopf at I_c ~ 0.546
- **Detection**: Network diameter CUSUM

### 3. XY Model (Planar Spins)

```
H = -J * sum cos(theta_i - theta_j)
```

- **Transition**: BKT at T_BKT ~ 0.893
- **Detection**: Magnetization + Vortex CUSUM

### 4. Kuramoto (Coupled Oscillators)

```
d(theta_i)/dt = omega_i + (K/N) * sum sin(theta_j - theta_i)
```

- **Transition**: Synchronization at K_c = 2*sigma_omega
- **Detection**: beta_0 + Entropy CUSUM

### 5. Lennard-Jones (Particles)

```
V(r) = 4*epsilon * [(sigma/r)^12 - (sigma/r)^6]
```

- **Transition**: Crystallization (or vitrification)
- **Detection**: Exact S_H1 CUSUM (73% rate at N=144)

## Installation

```bash
git clone https://github.com/Yatrogenesis/TDA-Information-Dynamics
cd TDA-Information-Dynamics
cargo build --release
```

## Usage

### Run Individual Systems

```bash
# Chemical oscillator (Hopf bifurcation)
cargo run --release --bin brusselator_tda_cusum

# Excitable neuron network
cargo run --release --bin fhn_tda_cusum

# Planar spin model (BKT transition)
cargo run --release --bin xy_tda_cusum

# Coupled oscillators (synchronization)
cargo run --release --bin kuramoto_tda

# Particle crystallization (Euler approximation)
cargo run --release --bin lj_tda_cusum

# Particle crystallization (exact persistence, matches Python/Ripser)
cargo run --release --bin lj_exact_tda_cusum

# Compare exact vs Euler methods
cargo run --release --bin lj_exact_vs_euler
```

### Run Tests

```bash
cargo test --lib --release
# 35 tests passing
```

### Library Usage

```rust
use tda_info_dynamics::{
    // Persistence computation
    VietorisRips,
    compute_persistence,           // Euler (fast)
    compute_exact_persistence_simple,  // Exact (matches Ripser)
    compute_entropy,

    // Detection
    CusumDetector,

    // Systems
    Brusselator,
    FitzHughNagumoSystem,
    XYModel,
    KuramotoSystem,
    LennardJonesSystem,
};

// Example: Exact persistence for crystallization detection
let dist = system.distance_matrix();
let pd = compute_exact_persistence_simple(&dist, max_epsilon);
let entropy = pd.persistence_entropy(1);  // H1 entropy

// CUSUM detection with sigma_min
let mut detector = CusumDetector::with_sigma_min(0.5, 4.0, 0.1);
detector.calibrate(&baseline_data);

for value in &monitoring_data {
    if detector.update(*value) {
        println!("Topological precursor detected!");
    }
}
```

## Module Structure

```
src/
+-- lib.rs                        # Re-exports
+-- topology/
|   +-- vietoris_rips.rs          # VR complex construction
|   +-- persistence.rs            # Euler approximation (fast)
|   +-- persistence_exact.rs      # Standard algorithm (exact)
|   +-- betti.rs                  # Betti curves
+-- information/
|   +-- entropy.rs                # Topological entropy
|   +-- landscape.rs              # Persistence landscapes
+-- cusum/
|   +-- detector.rs               # CUSUM + sigma_min
+-- systems/
|   +-- traits.rs                 # DynamicalSystem, Controllable, Bifurcating
|   +-- brusselator.rs            # Chemical oscillator (RK4)
|   +-- fitzhugh_nagumo.rs        # Excitable neuron (RK4)
|   +-- xy_model.rs               # Planar spins (Monte Carlo)
|   +-- kuramoto.rs               # Coupled oscillators (RK4)
|   +-- lennard_jones.rs          # Particles (Velocity Verlet)

src/bin/
+-- brusselator_tda_cusum.rs      # Hopf bifurcation detection
+-- fhn_tda_cusum.rs              # FHN network transition
+-- xy_tda_cusum.rs               # BKT transition detection
+-- kuramoto_tda.rs               # Synchronization detection
+-- lj_tda_cusum.rs               # Crystallization (Euler)
+-- lj_exact_tda_cusum.rs         # Crystallization (Exact)
+-- lj_exact_vs_euler.rs          # Method comparison
```

## Related Projects

- **[TDA-Phase-Transitions](https://github.com/Yatrogenesis/TDA-Phase-Transitions)** (Python)
  DOI: [10.5281/zenodo.18220298](https://doi.org/10.5281/zenodo.18220298)
  Comprehensive paper on TDA approach to 2D Lennard-Jones crystallization.

- **[TDA-Complex-Systems](https://github.com/Yatrogenesis/TDA-Complex-Systems)** (Rust)
  Extensions: LJ 3D, TIP4P water, glass transition with Steinhardt Q6.

## References

1. Edelsbrunner, H., & Harer, J. (2010). *Computational Topology: An Introduction*. AMS.

2. Steinhardt, P. J., Nelson, D. R., & Ronchetti, M. (1983). Bond-orientational order in liquids and glasses. *Physical Review B*, 28(2), 784.

3. Tononi, G. (2008). Consciousness as integrated information. *Biological Bulletin*, 215(3), 216-242.

4. Page, E. S. (1954). Continuous inspection schemes. *Biometrika*, 41(1/2), 100-115.

5. Carlsson, G. (2009). Topology and data. *Bulletin of the AMS*, 46(2), 255-308.

6. Kuramoto, Y. (1984). *Chemical Oscillations, Waves, and Turbulence*. Springer.

7. Kosterlitz, J. M., & Thouless, D. J. (1973). Ordering, metastability and phase transitions in two-dimensional systems. *Journal of Physics C*, 6(7), 1181.

## Author

**Francisco Molina-Burgos**
Avermex Research Division
Merida, Yucatan, Mexico
2026

## License

MIT License - See [LICENSE](LICENSE) for details.
