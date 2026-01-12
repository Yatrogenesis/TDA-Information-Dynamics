//! # TDA-Information-Dynamics
//!
//! Topological Dynamics of Information: Critical Transition Detection
//! via Persistent Homology and Sequential Analysis (TDA-CUSUM)
//!
//! ## Theoretical Framework
//!
//! This crate implements a unified theoretical-computational framework for
//! quantifying and predicting the emergence of macroscopic order in complex
//! stochastic systems.
//!
//! ### Central Hypothesis
//!
//! Phase transitions (from thermodynamic equilibrium to ordered states or
//! dynamic attractors) are preceded by fundamental changes in the topology
//! of the phase space manifold, which are not detectable by conventional
//! local order parameters.
//!
//! ### Methodology
//!
//! 1. **Persistent Homology (TDA)**: Characterizes the geometric structure
//!    of system information via Betti numbers (β₀, β₁, β₂)
//!
//! 2. **Topological Entropy**: Quantifies structural complexity using
//!    Shannon entropy over persistence diagrams
//!
//! 3. **CUSUM Detection**: Identifies precursors of symmetry breaking
//!    (crystallization, synchronization, dynamic arrest) before classical
//!    thermodynamic singularities are observable
//!
//! ## Key Result
//!
//! For properly calibrated systems:
//!
//!   t_CUSUM < t_physical
//!
//! The topological warning precedes the physical manifestation of the
//! phase transition, enabling early detection and prediction.
//!
//! ## References
//!
//! - Edelsbrunner & Harer, "Computational Topology" (2010)
//! - Steinhardt et al., PRB 28, 784 (1983) - Order parameters
//! - Tononi, "Integrated Information Theory" - Φ metric connection
//! - Tegmark, "Mathematical Universe Hypothesis" - Structural realism
//!
//! ## Author
//!
//! Francisco Molina-Burgos
//! Avermex Research Division
//! Mérida, Yucatán, México
//! 2026

pub mod topology;
pub mod information;
pub mod cusum;
pub mod systems;

// Re-exports from topology
pub use topology::{
    // Approximate persistence (Euler-based, fast)
    PersistenceDiagram,
    PersistenceInterval,
    compute_persistence,
    // Exact persistence (standard algorithm, matches Ripser)
    ExactPersistenceDiagram,
    ExactInterval,
    compute_exact_persistence,
    compute_exact_persistence_simple,
    // Complex construction
    VietorisRips,
    BettiNumbers,
    BettiCurve,
};

// Re-exports from information
pub use information::{
    TopologicalEntropy,
    PersistenceLandscape,
    compute_entropy,
};

// Re-exports from cusum
pub use cusum::{
    CusumDetector,
    CusumResult,
    DetectionEvent,
    TwoSidedCusum,
    AdaptiveCusum,
};

// Re-exports from systems
pub use systems::{
    // Traits
    DynamicalSystem,
    Controllable,
    Bifurcating,
    TDAConfig,
    // Lennard-Jones
    LennardJonesSystem,
    LJState,
    // Kuramoto
    KuramotoSystem,
    KuramotoState,
    // FitzHugh-Nagumo
    FitzHughNagumoSystem,
    FHNState,
    FHNNeuronState,
    // XY Model
    XYModel,
    XYState,
    // Brusselator
    Brusselator,
    BrusselatorState,
};
