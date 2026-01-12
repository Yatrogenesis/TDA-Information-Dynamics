//! FitzHugh-Nagumo TDA-CUSUM: Hopf Bifurcation Detection
//!
//! This binary demonstrates the TDA-CUSUM framework for detecting
//! the Hopf bifurcation in excitable neuron models.
//!
//! ## Key Insight
//!
//! As the system approaches the Hopf bifurcation:
//! - Resting: Trajectory concentrated at fixed point → small diameter
//! - Transition: Trajectory spirals outward → diameter increases
//! - Oscillating: Trajectory forms limit cycle → large diameter, stable β₀
//!
//! We monitor the **trajectory diameter** as a robust topological precursor.

use tda_info_dynamics::{
    FitzHughNagumoSystem,
    DynamicalSystem,
    VietorisRips,
    compute_persistence,
    compute_entropy,
    CusumDetector,
    PersistenceDiagram,
};

/// Compute statistics from distance matrix
fn trajectory_statistics(dist: &ndarray::Array2<f64>) -> (f64, f64, f64) {
    let n = dist.nrows();
    if n < 2 {
        return (0.0, 0.0, 0.0);
    }

    // Maximum distance (diameter)
    let mut max_dist = 0.0f64;
    let mut total_dist = 0.0f64;
    let mut count = 0;

    for i in 0..n {
        for j in i+1..n {
            let d = dist[[i, j]];
            if d > max_dist {
                max_dist = d;
            }
            total_dist += d;
            count += 1;
        }
    }

    let mean_dist = if count > 0 { total_dist / count as f64 } else { 0.0 };

    // Approximate "spread" - variance of distances
    let mut var = 0.0;
    for i in 0..n {
        for j in i+1..n {
            let d = dist[[i, j]];
            var += (d - mean_dist).powi(2);
        }
    }
    let std_dist = if count > 1 { (var / (count - 1) as f64).sqrt() } else { 0.0 };

    (max_dist, mean_dist, std_dist)
}

/// Extract H1 statistics from persistence diagram
fn h1_statistics(pd: &PersistenceDiagram) -> (usize, f64, f64) {
    let h1_intervals = pd.dim(1);
    let count = h1_intervals.len();
    let total_lifetime = pd.total_persistence(1);
    let max_lifetime: f64 = h1_intervals.iter()
        .filter(|i| !i.is_essential())
        .map(|i| i.persistence())
        .fold(0.0, f64::max);

    (count, total_lifetime, max_lifetime)
}

/// Subsample a distance matrix uniformly
fn subsample_distance_matrix(dist: &ndarray::Array2<f64>, target_points: usize) -> ndarray::Array2<f64> {
    let n = dist.nrows();
    if n <= target_points {
        return dist.clone();
    }

    let step = n / target_points;
    let indices: Vec<usize> = (0..target_points).map(|i| i * step).collect();

    let mut sub = ndarray::Array2::zeros((target_points, target_points));
    for (i, &idx_i) in indices.iter().enumerate() {
        for (j, &idx_j) in indices.iter().enumerate() {
            sub[[i, j]] = dist[[idx_i, idx_j]];
        }
    }
    sub
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  TDA-CUSUM: FitzHugh-Nagumo Hopf Bifurcation Detection");
    println!("  Francisco Molina Burgos - 2026");
    println!("═══════════════════════════════════════════════════════════════\n");

    // System parameters
    let i_initial = 0.0;
    let i_final = 0.8;
    let amp_threshold = 1.5;

    let mut system = FitzHughNagumoSystem::single(i_initial);
    let i_critical = system.theoretical_i_critical();

    println!("System Parameters:");
    println!("  FitzHugh-Nagumo single neuron");
    println!("  I_ext: {:.2} → {:.2}", i_initial, i_final);
    println!("  Theoretical I_c ≈ {:.4}", i_critical);
    println!("  Amplitude threshold = {:.2}", amp_threshold);

    // TDA parameters - small epsilon for fine structure
    let vr_epsilon = 0.5;
    let vr_steps = 30;
    let target_points = 60;

    println!("\nTDA Parameters:");
    println!("  VR epsilon = {:.2}", vr_epsilon);
    println!("  VR steps = {}", vr_steps);
    println!("  Subsampled points = {}", target_points);
    println!("  Metrics: trajectory diameter, H0 components, H1 cycles");

    // Warm-up
    println!("\nWarming up system...");
    system.run(500);
    system.clear_trajectory();

    // Calibration phase
    println!("\nCalibration Phase (resting state baseline)...");
    let mut cal_entropy: Vec<f64> = Vec::new();
    let mut cal_diameter: Vec<f64> = Vec::new();
    let mut cal_h0: Vec<f64> = Vec::new();

    for i in 0..20 {
        system.run(100);

        if system.n_elements() < 30 {
            continue;
        }

        let dist = system.distance_matrix();
        let sub_dist = subsample_distance_matrix(&dist, target_points);
        let (diameter, _mean_d, _std_d) = trajectory_statistics(&sub_dist);

        let vr = VietorisRips::from_distance_matrix(&sub_dist, vr_epsilon, vr_steps);
        let pd = compute_persistence(&vr);
        let entropy = compute_entropy(&pd);
        let h0 = vr.count_components_at(vr_steps / 2);
        let (h1_count, _h1_total, _) = h1_statistics(&pd);

        cal_entropy.push(entropy.persistent_entropy);
        cal_diameter.push(diameter);
        cal_h0.push(-(h0 as f64));  // Negative: looking for DECREASE in components

        if i % 5 == 0 {
            println!(
                "  Sample {:2}: diam = {:.4}, β₀ = {:2}, H1 = {:2}, H_P = {:.3}, amp = {:.4}",
                i + 1, diameter, h0, h1_count, entropy.persistent_entropy, system.order_parameter()
            );
        }
    }

    // Initialize CUSUM
    // sigma_min prevents infinite sensitivity when baseline has zero variance
    let mut cusum_diameter = CusumDetector::with_sigma_min(0.3, 2.5, 0.1);
    let mut cusum_entropy = CusumDetector::with_sigma_min(0.3, 3.0, 0.1);
    let mut cusum_h0 = CusumDetector::with_sigma_min(0.2, 2.5, 0.5);

    cusum_diameter.calibrate(&cal_diameter);
    cusum_entropy.calibrate(&cal_entropy);
    cusum_h0.calibrate(&cal_h0);

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
    println!(
        "  β₀:       μ₀ = {:.4}, σ₀ = {:.4}",
        -cusum_h0.result().reference_mean,
        cusum_h0.result().reference_std
    );

    // Current ramp
    println!("\n══════════════════════════════════════════════════════════════");
    println!("  Current Ramp: Monitoring for Hopf Bifurcation Precursors");
    println!("══════════════════════════════════════════════════════════════\n");

    let current_rate = 0.01;
    let steps_per_sample = 100;
    let mut current_i = i_initial;

    let mut diameter_detected = false;
    let mut entropy_detected = false;
    let mut h0_detected = false;
    let mut amp_crossed = false;

    let mut i_cusum_diameter = None;
    let mut i_cusum_entropy = None;
    let mut i_cusum_h0 = None;
    let mut i_amplitude = None;

    let mut step = 0;
    system.clear_trajectory();

    while current_i < i_final {
        current_i += current_rate;
        system.set_current(current_i);
        system.run(steps_per_sample);

        let amp = system.order_parameter();

        if !amp_crossed && amp > amp_threshold {
            amp_crossed = true;
            i_amplitude = Some(current_i);
            println!(
                ">>> TRADITIONAL (amp > {}) at I = {:.4} (step {})",
                amp_threshold, current_i, step
            );
        }

        if system.n_elements() < 30 {
            step += 1;
            continue;
        }

        let dist = system.distance_matrix();
        let sub_dist = subsample_distance_matrix(&dist, target_points);
        let (diameter, _, _) = trajectory_statistics(&sub_dist);

        let vr = VietorisRips::from_distance_matrix(&sub_dist, vr_epsilon, vr_steps);
        let pd = compute_persistence(&vr);
        let entropy = compute_entropy(&pd);
        let h0 = vr.count_components_at(vr_steps / 2);
        let (h1_count, _h1_total, _) = h1_statistics(&pd);

        // Update CUSUM - diameter should INCREASE
        if !diameter_detected && cusum_diameter.update(diameter) {
            diameter_detected = true;
            i_cusum_diameter = Some(current_i);
            println!(
                ">>> DIAMETER DETECTION at I = {:.4} (step {})",
                current_i, step
            );
        }

        if !entropy_detected && cusum_entropy.update(entropy.persistent_entropy) {
            entropy_detected = true;
            i_cusum_entropy = Some(current_i);
            println!(
                ">>> ENTROPY DETECTION at I = {:.4} (step {})",
                current_i, step
            );
        }

        // β₀ should decrease (more connectivity)
        if !h0_detected && cusum_h0.update(-(h0 as f64)) {
            h0_detected = true;
            i_cusum_h0 = Some(current_i);
            println!(
                ">>> β₀ DETECTION at I = {:.4} (step {})",
                current_i, step
            );
        }

        if step % 10 == 0 {
            println!(
                "Step {:3}: I = {:.3}, amp = {:.3}, diam = {:.3}, β₀ = {:2}, H1 = {:2}",
                step, current_i, amp, diameter, h0, h1_count
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

    if let Some(i) = i_cusum_diameter {
        println!("  Diameter CUSUM detection: I = {:.4}", i);
    } else {
        println!("  Diameter CUSUM: No detection");
    }

    if let Some(i) = i_cusum_entropy {
        println!("  Entropy CUSUM detection:  I = {:.4}", i);
    } else {
        println!("  Entropy CUSUM: No detection");
    }

    if let Some(i) = i_cusum_h0 {
        println!("  β₀ CUSUM detection:       I = {:.4}", i);
    } else {
        println!("  β₀ CUSUM: No detection");
    }

    if let Some(i) = i_amplitude {
        println!("  Traditional amplitude:    I = {:.4}", i);
    } else {
        println!("  Traditional: No oscillation detected");
    }

    println!("  Theoretical I_c:          I ≈ {:.4}", i_critical);

    // Early warning analysis
    let first_topo = i_cusum_diameter.or(i_cusum_h0).or(i_cusum_entropy);
    if let (Some(i_topo), Some(i_amp)) = (first_topo, i_amplitude) {
        let delta_i = i_amp - i_topo;
        if delta_i > 0.0 {
            println!("\n  ✓ EARLY WARNING: ΔI = {:.4}", delta_i);
            println!("    Topological precursor detected BEFORE oscillation!");
        } else if delta_i < 0.0 {
            println!("\n  Detection after threshold: ΔI = {:.4}", delta_i);
        } else {
            println!("\n  Detection coincident with threshold");
        }
    }

    println!("\n─────────────────────────────────────────────────────────────");
    println!("Theoretical Note:");
    println!("  FitzHugh-Nagumo undergoes a supercritical Hopf bifurcation.");
    println!("  At I = I_c ≈ {:.4}, the stable fixed point becomes unstable", i_critical);
    println!("  and a stable limit cycle is born.");
    println!("  The trajectory diameter should increase as the system");
    println!("  spirals toward the limit cycle - this is our primary indicator.");

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Analysis Complete");
    println!("═══════════════════════════════════════════════════════════════");
}
