//! Topology Module: Persistent Homology and Simplicial Complexes
//!
//! Implements the mathematical structures for topological data analysis:
//! - Vietoris-Rips filtration
//! - Persistent homology computation (approximate and exact)
//! - Betti numbers extraction
//!
//! ## Mathematical Background
//!
//! For a point cloud X(t) representing the system microstate at time t,
//! we construct a filtration of simplicial complexes VR_ε(X) indexed by
//! the scale parameter ε. The persistent homology tracks the birth and
//! death of topological features (connected components, loops, voids)
//! across this filtration.
//!
//! ## Two Implementations
//!
//! - `persistence.rs`: Fast approximate algorithm using Euler characteristic
//!   (β₁ = E - V + β₀ - F). Good for Betti number counts.
//!
//! - `persistence_exact.rs`: Exact standard algorithm via boundary matrix
//!   reduction. Provides true birth/death pairs for entropy computation.
//!   Matches output of Ripser/Python ripser library.

mod vietoris_rips;
mod persistence;
mod persistence_exact;
mod betti;

pub use vietoris_rips::VietorisRips;
pub use persistence::{PersistenceDiagram, PersistenceInterval, compute_persistence};
pub use persistence_exact::{
    ExactPersistenceDiagram,
    ExactInterval,
    compute_exact_persistence,
    compute_exact_persistence_simple,
};
pub use betti::{BettiNumbers, BettiCurve};
