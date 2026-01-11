//! CUSUM Module: Sequential Detection of Topological Phase Transitions
//!
//! Implements Cumulative Sum (CUSUM) algorithms for detecting structural
//! stability loss in dynamical systems through topological signatures.
//!
//! ## Theoretical Framework
//!
//! The CUSUM detector monitors a topological statistic S(t) (e.g., persistent
//! entropy, Betti numbers) and tests:
//!
//! - H₀: System in metastable disorder (reference regime)
//! - H₁: Trajectory toward new attractor (phase transition)
//!
//! The cumulative sum is defined as:
//!
//!   C(t) = max(0, C(t-1) + (S(t) - μ₀) - k)
//!
//! where μ₀ is the reference mean and k is the allowance parameter.
//!
//! ## Key Result
//!
//! For properly calibrated systems:
//!
//!   t_CUSUM < t_physical
//!
//! The topological warning precedes the physical manifestation of the
//! phase transition.

mod detector;

pub use detector::{
    CusumDetector,
    CusumResult,
    DetectionEvent,
    TwoSidedCusum,
    AdaptiveCusum,
};
