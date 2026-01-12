//! Brusselator TDA-CUSUM: Chemical Oscillation Detection
//!
//! This binary demonstrates the TDA-CUSUM framework for detecting
//! the Hopf bifurcation in chemical oscillators.
//!
//! ## Key Insight
//!
//! The Brusselator undergoes a supercritical Hopf bifurcation:
//! - B < B_c = 1 + A²: Stable steady state
//! - B > B_c: Limit cycle oscillations (Belousov-Zhabotinsky type)
//!
//! For A = 1: B_c = 2
//!
//! This model is relevant for Trinity-like chemical systems and
//! pattern formation in reaction-diffusion systems.

use tda_info_dynamics::{
    Brusselator,
    DynamicalSystem,
    VietorisRips,
    compute_persistence,
    compute_entropy,
    CusumDetector,
};

/// Compute trajectory diameter
fn trajectory_diameter(dist: &ndarray::Array2<f64>) -> f64 {
    let n = dist.nrows();
    let mut max_dist = 0.0f64;
    for i in 0..n {
        for j in i+1..n {
            let d = dist[[i, j]];
            if d > max_dist {
                max_dist = d;
            }
        }
    }
    max_dist
}

/// Subsample distance matrix
fn subsample_distance_matrix(dist: &ndarray::Array2<f64>, target: usize) -> ndarray::Array2<f64> {
    let n = dist.nrows();
    if n <= target {
        return dist.clone();
    }

    let step = n / target;
    let indices: Vec<usize> = (0..target).map(|i| i * step).collect();

    let mut sub = ndarray::Array2::zeros((target, target));
    for (i, &idx_i) in indices.iter().enumerate() {
        for (j, &idx_j) in indices.iter().enumerate() {
            sub[[i, j]] = dist[[idx_i, idx_j]];
        }
    }
    sub
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  TDA-CUSUM: Brusselator Hopf Bifurcation Detection");
    println!("  Francisco Molina Burgos - 2026");
    println!("═══════════════════════════════════════════════════════════════\n");

    // System parameters
    let a = 1.0;
    let b_initial = 1.0;  // Start below bifurcation
    let b_final = 3.5;    // End above bifurcation
    let amp_threshold = 0.5;

    let mut system = Brusselator::standard(b_initial);
    let b_critical = system.theoretical_b_critical();

    println!("System Parameters:");
    println!("  Brusselator (A = {})", a);
    println!("  B: {:.2} → {:.2}", b_initial, b_final);
    println!("  Theoretical B_c = 1 + A² = {:.4}", b_critical);
    println!("  Amplitude threshold = {:.2}", amp_threshold);

    // TDA parameters
    let vr_epsilon = 0.3;
    let vr_steps = 25;
    let target_points = 50;

    println!("\nTDA Parameters:");
    println!("  VR epsilon = {:.2}", vr_epsilon);
    println!("  VR steps = {}", vr_steps);
    println!("  Subsampled points = {}", target_points);
    println!("  Metrics: trajectory diameter, entropy");

    // Thermalize
    println!("\nThermalizing at B = {:.2}...", b_initial);
    system.perturb(0.1);
    system.run(2000);
    system.clear_trajectory();

    // Calibration phase
    println!("\nCalibration Phase (steady state baseline)...");
    let mut cal_diameter: Vec<f64> = Vec::new();
    let mut cal_entropy: Vec<f64> = Vec::new();

    for i in 0..20 {
        system.run(200);

        if system.n_elements() < 30 {
            continue;
        }

        let dist = system.distance_matrix();
        let sub_dist = subsample_distance_matrix(&dist, target_points);
        let diameter = trajectory_diameter(&sub_dist);

        let vr = VietorisRips::from_distance_matrix(&sub_dist, vr_epsilon, vr_steps);
        let pd = compute_persistence(&vr);
        let entropy = compute_entropy(&pd);

        cal_diameter.push(diameter);
        cal_entropy.push(entropy.persistent_entropy);

        if i % 5 == 0 {
            let state = system.state();
            println!(
                "  Sample {:2}: diam = {:.4}, H_P = {:.3}, X = {:.3}, Y = {:.3}, amp = {:.4}",
                i + 1, diameter, entropy.persistent_entropy, state.x, state.y, state.amplitude
            );
        }
    }

    // Initialize CUSUM with sigma_min to prevent infinite sensitivity
    // sigma_min = 0.1: requires ~0.25 diameter change for detection (k=0.3, h=2.5)
    let mut cusum_diameter = CusumDetector::with_sigma_min(0.3, 2.5, 0.1);
    let mut cusum_entropy = CusumDetector::with_sigma_min(0.3, 3.0, 0.1);

    cusum_diameter.calibrate(&cal_diameter);
    cusum_entropy.calibrate(&cal_entropy);

    println!("\nCUSUM Calibrated:");
    println!(
        "  Diameter: μ₀ = {:.4}, σ₀ = {:.4}",
        cusum_diameter.result().reference_mean,
        cusum_diameter.result().reference_std
    );
    println!(
        "  Entropy:  μ₀ = {:.4}, σ₀ = {:.4}",
        cusum_entropy.result().reference_mean,
        cusum_entropy.result().reference_std
    );

    // Parameter ramp
    println!("\n══════════════════════════════════════════════════════════════");
    println!("  Parameter Ramp: Monitoring for Hopf Bifurcation");
    println!("══════════════════════════════════════════════════════════════\n");

    let param_rate = 0.02;
    let steps_per_sample = 200;
    let mut current_b = b_initial;

    let mut diameter_detected = false;
    let mut entropy_detected = false;
    let mut amp_crossed = false;

    let mut b_cusum_diameter = None;
    let mut b_cusum_entropy = None;
    let mut b_amplitude = None;

    let mut step = 0;
    system.clear_trajectory();

    while current_b < b_final {
        current_b += param_rate;
        system.set_b(current_b);
        system.run(steps_per_sample);

        let state = system.state();

        if !amp_crossed && state.amplitude > amp_threshold {
            amp_crossed = true;
            b_amplitude = Some(current_b);
            println!(
                ">>> TRADITIONAL (amp > {}) at B = {:.4} (step {})",
                amp_threshold, current_b, step
            );
        }

        if system.n_elements() < 30 {
            step += 1;
            continue;
        }

        let dist = system.distance_matrix();
        let sub_dist = subsample_distance_matrix(&dist, target_points);
        let diameter = trajectory_diameter(&sub_dist);

        let vr = VietorisRips::from_distance_matrix(&sub_dist, vr_epsilon, vr_steps);
        let pd = compute_persistence(&vr);
        let entropy = compute_entropy(&pd);

        if !diameter_detected && cusum_diameter.update(diameter) {
            diameter_detected = true;
            b_cusum_diameter = Some(current_b);
            println!(
                ">>> DIAMETER DETECTION at B = {:.4} (step {})",
                current_b, step
            );
        }

        if !entropy_detected && cusum_entropy.update(entropy.persistent_entropy) {
            entropy_detected = true;
            b_cusum_entropy = Some(current_b);
            println!(
                ">>> ENTROPY DETECTION at B = {:.4} (step {})",
                current_b, step
            );
        }

        if step % 10 == 0 {
            println!(
                "Step {:3}: B = {:.3}, amp = {:.4}, diam = {:.4}, H_P = {:.3}",
                step, current_b, state.amplitude, diameter, entropy.persistent_entropy
            );
        }

        step += 1;
    }

    // Results
    println!("\n══════════════════════════════════════════════════════════════");
    println!("  Results");
    println!("══════════════════════════════════════════════════════════════\n");

    println!("Detection Comparison:");
    println!("─────────────────────────────────────────────────────────────");

    if let Some(b) = b_cusum_diameter {
        println!("  Diameter CUSUM detection: B = {:.4}", b);
    } else {
        println!("  Diameter CUSUM: No detection");
    }

    if let Some(b) = b_cusum_entropy {
        println!("  Entropy CUSUM detection:  B = {:.4}", b);
    } else {
        println!("  Entropy CUSUM: No detection");
    }

    if let Some(b) = b_amplitude {
        println!("  Traditional amplitude:    B = {:.4}", b);
    } else {
        println!("  Traditional: No oscillation detected");
    }

    println!("  Theoretical B_c:          B = {:.4}", b_critical);

    // Early warning
    let first_topo = b_cusum_diameter.or(b_cusum_entropy);
    if let (Some(b_topo), Some(b_trad)) = (first_topo, b_amplitude) {
        let delta_b = b_trad - b_topo;
        if delta_b > 0.0 {
            println!("\n  ✓ EARLY WARNING: ΔB = {:.4}", delta_b);
            println!("    Topological precursor detected BEFORE oscillation!");
        } else if delta_b < 0.0 {
            println!("\n  Detection after threshold: ΔB = {:.4}", delta_b);
        }
    }

    println!("\n─────────────────────────────────────────────────────────────");
    println!("Theoretical Note:");
    println!("  The Brusselator undergoes a supercritical Hopf bifurcation.");
    println!("  At B = B_c = 1 + A² = {:.4}, the steady state becomes unstable", b_critical);
    println!("  and a limit cycle emerges (chemical oscillations).");
    println!("  This model is relevant for Belousov-Zhabotinsky reactions");
    println!("  and Trinity-like autocatalytic systems.");

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Analysis Complete");
    println!("═══════════════════════════════════════════════════════════════");
}
