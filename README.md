# TDA-Information-Dynamics

**Topological Dynamics of Information: Critical Transition Detection via Persistent Homology and Sequential Analysis (TDA-CUSUM)**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)

## Overview

This crate implements a unified theoretical-computational framework for quantifying and predicting the emergence of macroscopic order in complex stochastic systems. By combining Topological Data Analysis (TDA) with sequential detection methods (CUSUM), we achieve early warning detection of phase transitions.

### Central Hypothesis

Phase transitions (from thermodynamic equilibrium to ordered states or dynamic attractors) are preceded by fundamental changes in the **topology of the phase space manifold**, which are not detectable by conventional local order parameters.

### Key Result

For properly calibrated systems:

```
t_CUSUM < t_physical
```

The topological warning precedes the physical manifestation of the phase transition.

## Theoretical Framework

### 1. Topological Characterization

For a point cloud X(t) representing system microstate at time t, we construct the **Vietoris-Rips filtration** VR_ε(X) and compute **persistent homology**.

The k-th Betti number βₖ counts topological features:
- **β₀**: Connected components (particles in same cluster)
- **β₁**: 1-dimensional loops/cycles (ring structures)
- **β₂**: 2-dimensional voids (cavities)

### 2. Topological Entropy

Given a persistence diagram D = {(bᵢ, dᵢ)}, the **persistent (Shannon) entropy** is:

```
H_P = -Σᵢ pᵢ log(pᵢ)
```

where pᵢ = lᵢ / L is the normalized lifetime, with lᵢ = dᵢ - bᵢ and L = Σⱼ lⱼ.

### 3. Connection to Integrated Information (Φ)

High persistent entropy with long-lived high-dimensional features (β₁, β₂) indicates a system with high **information integration (Φ)** in the sense of Integrated Information Theory (IIT).

The system distinguishes itself from:
- **Pure randomness** (white noise): maximal trivial entropy
- **Perfect regularity** (crystal): zero entropy

Complex systems occupy an intermediate regime of **structured complexity**.

### 4. CUSUM Detection

The Cumulative Sum (CUSUM) algorithm monitors topological statistics S(t):

```
C(t) = max(0, C(t-1) + (S(t) - μ₀) - k)
```

Testing:
- **H₀**: System in metastable disorder (reference regime)
- **H₁**: Trajectory toward new attractor (phase transition)

Detection occurs when C(t) > h (threshold).

## Implemented Systems

### Lennard-Jones Fluid

Classical liquid-solid (crystallization) transition:

```
V(r) = 4ε[(σ/r)¹² - (σ/r)⁶]
```

**Topological signatures:**
- Liquid: β₀ fluctuates, low β₁, high entropy
- Solid (FCC): β₀ ≈ 1, structured β₁, low entropy, Q6 ≈ 0.574
- Glass: Intermediate, frustrated β₁, Q6 ≈ 0.29

### Kuramoto Oscillators

Synchronization transition in coupled oscillators:

```
dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)
```

**Topological signatures** (on circle embedding):
- Incoherent (K < Kc): Uniform phase distribution → high β₀, low β₁
- Synchronized (K > Kc): Clustered phases → low β₀, emerging β₁

## Installation

```bash
git clone https://github.com/Yatrogenesis/TDA-Information-Dynamics
cd TDA-Information-Dynamics
cargo build --release
```

## Usage

### Lennard-Jones Crystallization Detection

```bash
cargo run --release --bin lj_tda_cusum
```

### Kuramoto Synchronization Detection

```bash
cargo run --release --bin kuramoto_tda
```

### Library Usage

```rust
use tda_info_dynamics::{
    VietorisRips,
    compute_persistence,
    compute_entropy,
    CusumDetector,
    LennardJonesSystem,
    KuramotoSystem,
};

// Build Vietoris-Rips filtration from points
let vr = VietorisRips::from_points(&points, max_epsilon, n_steps);

// Compute persistence diagram
let pd = compute_persistence(&vr);

// Compute topological entropy
let entropy = compute_entropy(&pd);
println!("Persistent entropy: {}", entropy.persistent_entropy);

// CUSUM detection
let mut detector = CusumDetector::with_params(0.5, 4.0);
detector.calibrate(&reference_data);

for value in &monitoring_data {
    if detector.update(*value) {
        println!("Phase transition precursor detected!");
    }
}
```

## Module Structure

```
src/
├── lib.rs           # Main library
├── topology/        # TDA core
│   ├── mod.rs
│   ├── vietoris_rips.rs  # VR filtration construction
│   ├── persistence.rs    # Persistent homology
│   └── betti.rs          # Betti numbers & curves
├── information/     # Information-theoretic measures
│   ├── mod.rs
│   ├── entropy.rs        # Topological entropy
│   └── landscape.rs      # Persistence landscapes
├── cusum/          # Sequential detection
│   ├── mod.rs
│   └── detector.rs       # CUSUM algorithms
├── systems/        # Physical models
│   ├── mod.rs
│   ├── lennard_jones.rs  # LJ fluid simulation
│   └── kuramoto.rs       # Kuramoto oscillators
└── bin/            # Executables
    ├── lj_tda_cusum.rs
    └── kuramoto_tda.rs
```

## Related Projects

- [TDA-Complex-Systems](https://github.com/Yatrogenesis/TDA-Complex-Systems): Detailed analysis of LJ 3D, LJ Glass, and TIP4P water with Steinhardt Q6
- [TDA-Phase-Transitions](https://github.com/Yatrogenesis/TDA-Phase-Transitions): Comprehensive paper on TDA approach to phase transitions

## References

1. Edelsbrunner, H., & Harer, J. (2010). *Computational Topology: An Introduction*. AMS.

2. Steinhardt, P. J., Nelson, D. R., & Ronchetti, M. (1983). Bond-orientational order in liquids and glasses. *Physical Review B*, 28(2), 784.

3. Tononi, G. (2008). Consciousness as integrated information: A provisional manifesto. *Biological Bulletin*, 215(3), 216-242.

4. Page, E. S. (1954). Continuous inspection schemes. *Biometrika*, 41(1/2), 100-115.

5. Carlsson, G. (2009). Topology and data. *Bulletin of the American Mathematical Society*, 46(2), 255-308.

## Author

**Francisco Molina Burgos**
Independent Researcher
2026

## License

MIT License - See [LICENSE](LICENSE) for details.
