//! Physical Systems Module: Dynamical Models for TDA Analysis
//!
//! Implements prototype dynamical systems for studying topological
//! signatures of phase transitions:
//!
//! - **Lennard-Jones**: Classical liquid-solid transition
//! - **Kuramoto**: Synchronization transition in coupled oscillators
//!
//! These systems serve as test beds for the CUSUM detection framework.

mod lennard_jones;
mod kuramoto;

pub use lennard_jones::{LennardJonesSystem, LJState};
pub use kuramoto::{KuramotoSystem, KuramotoState};
