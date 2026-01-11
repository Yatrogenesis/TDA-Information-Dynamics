//! Information Module: Topological Entropy and Information Integration
//!
//! Implements information-theoretic measures on persistence diagrams,
//! connecting TDA to Integrated Information Theory (IIT) concepts.
//!
//! ## Core Concepts
//!
//! ### Persistent Entropy (Shannon)
//!
//! Given a persistence diagram D = {(bᵢ, dᵢ)}, we define the
//! persistent entropy as:
//!
//!   H_P = -Σᵢ pᵢ log(pᵢ)
//!
//! where pᵢ = lᵢ / L, lᵢ = dᵢ - bᵢ is the lifetime of generator i,
//! and L = Σⱼ lⱼ is the total lifetime.
//!
//! ### Connection to Φ (IIT)
//!
//! High persistent entropy with long-lived high-dimensional features
//! (β₁, β₂) indicates a system with high information integration (Φ).
//! The system distinguishes itself from both:
//! - Pure randomness (white noise): maximal trivial entropy
//! - Perfect regularity (crystal): zero entropy
//!
//! This places complex systems in an intermediate regime of
//! structured complexity.

mod entropy;
mod landscape;

pub use entropy::{TopologicalEntropy, compute_entropy};
pub use landscape::PersistenceLandscape;
