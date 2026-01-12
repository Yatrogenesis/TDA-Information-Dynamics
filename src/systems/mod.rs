//! Physical Systems Module: Dynamical Models for TDA Analysis
//!
//! Implements dynamical systems for studying topological signatures
//! of phase transitions and bifurcations:
//!
//! - **Lennard-Jones**: Classical liquid-solid transition
//! - **Kuramoto**: Synchronization transition in coupled oscillators
//! - **FitzHugh-Nagumo**: Hopf bifurcation in excitable neurons
//!
//! ## Architecture
//!
//! All systems implement the `DynamicalSystem` trait for standardized
//! interaction with the TDA-CUSUM framework. This enables:
//!
//! - Automatic VR epsilon calibration via k-NN heuristic
//! - Unified interface for analysis pipelines
//! - Easy addition of new systems
//!
//! ## Future Systems (Planned)
//!
//! - **Rössler**: Chaotic attractor analysis
//! - **Lorenz**: Strange attractor

pub mod traits;
mod lennard_jones;
mod kuramoto;
mod fitzhugh_nagumo;
mod xy_model;
mod brusselator;

pub use traits::{DynamicalSystem, Controllable, Bifurcating, TDAConfig};
pub use lennard_jones::{LennardJonesSystem, LJState};
pub use kuramoto::{KuramotoSystem, KuramotoState};
pub use fitzhugh_nagumo::{FitzHughNagumoSystem, FHNState, FHNNeuronState};
pub use xy_model::{XYModel, XYState};
pub use brusselator::{Brusselator, BrusselatorState};
