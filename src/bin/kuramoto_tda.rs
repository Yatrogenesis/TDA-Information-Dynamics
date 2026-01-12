//! Kuramoto TDA-CUSUM: Synchronization Transition Detection
//!
//! This binary demonstrates the TDA-CUSUM framework for detecting
//! the onset of synchronization in coupled oscillator systems.
//!
//! ## Protocol
//!
//! 1. Initialize Kuramoto system at weak coupling (incoherent)
//! 2. Collect baseline topological statistics using geodesic distance on S¹
//! 3. Ramp coupling while monitoring CUSUM
//! 4. Detect topological precursors of synchronization
//! 5. Compare t_CUSUM with r(t) order parameter crossing
//!
//! ## Key Insight
//!
//! Using geodesic distance on S¹ with small epsilon reveals:
//! - Incoherent: Many connected components (high β₀)
//! - Synchronized: Few components, phases clustered (low β₀)

use tda_info_dynamics::{
    KuramotoSystem,
    VietorisRips,
    compute_persistence,
    compute_entropy,
    CusumDetector,
    BettiCurve,
};

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  TDA-CUSUM: Kuramoto Synchronization Detection");
    println!("  Francisco Molina Burgos - 2026");
    println!("═══════════════════════════════════════════════════════════════\n");

    // System parameters
    let n_oscillators = 100;
    let freq_std = 0.5;     // Small frequency spread
    let k_initial = 0.0;    // Start with NO coupling for true incoherence
    let k_final = 3.0;      // Final coupling (synchronized)
    let r_threshold = 0.7;  // Traditional sync threshold

    // TDA parameters - KEY CHANGES
    // On S¹, geodesic distance ranges from 0 to π ≈ 3.14
    // With N=100 uniformly distributed, avg spacing ≈ 2π/100 ≈ 0.063
    // Use epsilon that shows fragmentation vs clustering
    let vr_epsilon = 0.4;   // Very small - reveals fragmentation
    let vr_steps = 30;
    let beta0_scale = 15;   // Scale index for β₀ measurement (~0.2 geodesic)

    println!("System Parameters:");
    println!("  N = {} oscillators", n_oscillators);
    println!("  Frequency std = {:.2}", freq_std);
    println!("  K: {:.2} → {:.2}", k_initial, k_final);
    println!("  Sync threshold r = {:.2}", r_threshold);
    println!("\nTDA Parameters (geodesic on S¹):");
    println!("  VR epsilon = {:.2} (max geodesic = π)", vr_epsilon);
    println!("  β₀ scale index = {}", beta0_scale);
    println!();

    // Initialize system with small frequency spread
    println!("Initializing Kuramoto system (small freq spread)...");
    let mut system = KuramotoSystem::new(n_oscillators, k_initial, freq_std);

    // Equilibrate
    println!("Equilibrating at K = {:.2}...", k_initial);
    system.run(5000);

    // Calibration phase - using GEODESIC distance on circle
    println!("\nCalibration Phase (incoherent baseline)...");
    let mut calibration_entropy: Vec<f64> = Vec::new();
    let mut calibration_beta0: Vec<f64> = Vec::new();
    let mut calibration_beta1: Vec<f64> = Vec::new();

    for i in 0..20 {
        system.run(200);

        // Get geodesic distance matrix on S¹
        let dist = system.circle_distance_matrix();

        // Build Vietoris-Rips using geodesic distances
        let vr = VietorisRips::from_distance_matrix(&dist, vr_epsilon, vr_steps);

        // Compute persistence
        let pd = compute_persistence(&vr);

        // Compute entropy
        let entropy = compute_entropy(&pd);
        // Use NEGATIVE values for CUSUM calibration (looking for entropy DECREASE)
        calibration_entropy.push(-entropy.persistent_entropy);

        // Count β₀ at specified scale - use negative (looking for DECREASE)
        let beta0 = vr.count_components_at(beta0_scale) as f64;
        calibration_beta0.push(-beta0);

        // Estimate β₁ (cycles) - use positive (looking for INCREASE)
        let beta1 = vr.estimate_cycles_at(beta0_scale) as f64;
        calibration_beta1.push(beta1);

        if i % 5 == 0 {
            println!(
                "  Sample {:2}: H_P = {:.4}, β₀ = {:2.0}, β₁ = {:2.0}, r = {:.4}",
                i + 1,
                entropy.persistent_entropy,
                beta0,
                beta1,
                system.order_parameter()
            );
        }
    }

    // Initialize CUSUM detectors
    // Very sensitive: low allowance (0.1σ), low threshold (2.0σ)
    let mut cusum_entropy = CusumDetector::with_params(0.1, 2.0);
    let mut cusum_beta0 = CusumDetector::with_params(0.1, 2.0);
    let mut cusum_beta1 = CusumDetector::with_params(0.1, 2.0);

    cusum_entropy.calibrate(&calibration_entropy);
    cusum_beta0.calibrate(&calibration_beta0);
    cusum_beta1.calibrate(&calibration_beta1);

    println!("\nCUSUM Calibrated:");
    println!(
        "  Entropy: μ₀ = {:.4}, σ₀ = {:.4}",
        cusum_entropy.result().reference_mean,
        cusum_entropy.result().reference_std
    );
    println!(
        "  β₀:      μ₀ = {:.4}, σ₀ = {:.4}",
        cusum_beta0.result().reference_mean,
        cusum_beta0.result().reference_std
    );
    println!(
        "  β₁:      μ₀ = {:.4}, σ₀ = {:.4}",
        cusum_beta1.result().reference_mean,
        cusum_beta1.result().reference_std
    );

    // Coupling ramp with CUSUM monitoring
    println!("\n══════════════════════════════════════════════════════════════");
    println!("  Coupling Ramp: Monitoring for Synchronization Precursors");
    println!("══════════════════════════════════════════════════════════════\n");

    let coupling_rate = 0.02;  // Slower ramp for better resolution
    let steps_per_sample = 300;
    let mut current_k = k_initial;

    let mut entropy_detected = false;
    let mut beta0_detected = false;
    let mut beta1_detected = false;
    let mut r_crossed = false;

    let mut t_cusum_entropy = None;
    let mut t_cusum_beta0 = None;
    let mut t_cusum_beta1 = None;
    let mut k_physical = None;

    let mut step = 0;

    while current_k < k_final {
        // Increase coupling
        current_k += coupling_rate;
        system.set_coupling(current_k);
        system.run(steps_per_sample);

        let r = system.order_parameter();

        // Check traditional threshold
        if !r_crossed && r > r_threshold {
            r_crossed = true;
            k_physical = Some(current_k);
            println!(
                ">>> TRADITIONAL SYNC (r > {}) at K = {:.4} (step {})",
                r_threshold, current_k, step
            );
        }

        // Compute topological statistics using geodesic distance
        let dist = system.circle_distance_matrix();
        let vr = VietorisRips::from_distance_matrix(&dist, vr_epsilon, vr_steps);
        let pd = compute_persistence(&vr);
        let entropy = compute_entropy(&pd);
        let beta0 = vr.count_components_at(beta0_scale) as f64;
        let beta1 = vr.estimate_cycles_at(beta0_scale) as f64;

        // Update CUSUM
        // Entropy: expect decrease during sync (use negative)
        if !entropy_detected && cusum_entropy.update(-entropy.persistent_entropy) {
            entropy_detected = true;
            t_cusum_entropy = Some(current_k);
            println!(
                ">>> ENTROPY DETECTION at K = {:.4} (step {})",
                current_k, step
            );
        }

        // β₀: expect decrease as phases cluster (use negative)
        if !beta0_detected && cusum_beta0.update(-beta0) {
            beta0_detected = true;
            t_cusum_beta0 = Some(current_k);
            println!(
                ">>> β₀ DETECTION at K = {:.4} (step {})",
                current_k, step
            );
        }

        // β₁: expect increase as synchronized cluster forms ring structure
        if !beta1_detected && cusum_beta1.update(beta1) {
            beta1_detected = true;
            t_cusum_beta1 = Some(current_k);
            println!(
                ">>> β₁ DETECTION at K = {:.4} (step {})",
                current_k, step
            );
        }

        if step % 10 == 0 {
            println!(
                "Step {:3}: K = {:.3}, r = {:.4}, H_P = {:.4}, β₀ = {:2.0}, β₁ = {:2.0}",
                step,
                current_k,
                r,
                entropy.persistent_entropy,
                beta0,
                beta1
            );
        }

        step += 1;
    }

    // Final analysis
    println!("\n══════════════════════════════════════════════════════════════");
    println!("  Results");
    println!("══════════════════════════════════════════════════════════════\n");

    println!("Detection Comparison:");
    println!("─────────────────────────────────────────────────────────────");

    if let Some(k) = t_cusum_entropy {
        println!("  Entropy CUSUM detection: K = {:.4}", k);
    } else {
        println!("  Entropy CUSUM: No detection");
    }

    if let Some(k) = t_cusum_beta0 {
        println!("  β₀ CUSUM detection:      K = {:.4}", k);
    } else {
        println!("  β₀ CUSUM: No detection");
    }

    if let Some(k) = t_cusum_beta1 {
        println!("  β₁ CUSUM detection:      K = {:.4}", k);
    } else {
        println!("  β₁ CUSUM: No detection");
    }

    if let Some(k) = k_physical {
        println!("  Traditional r-threshold: K = {:.4}", k);
    } else {
        println!("  Traditional: No synchronization achieved");
    }

    // Early warning analysis
    let first_topo = t_cusum_entropy.or(t_cusum_beta0).or(t_cusum_beta1);
    if let (Some(k_topo), Some(k_phys)) = (first_topo, k_physical) {
        let delta_k = k_phys - k_topo;
        if delta_k > 0.0 {
            println!("\n  ✓ EARLY WARNING: ΔK = {:.4}", delta_k);
            println!("    Topological precursor detected BEFORE synchronization!");
        } else {
            println!("\n  Detection after threshold: ΔK = {:.4}", delta_k);
        }
    }

    // Theoretical context
    println!("\n─────────────────────────────────────────────────────────────");
    println!("Theoretical Note:");
    println!("  For Gaussian frequency distribution (σ={:.2}):", freq_std);
    println!("  Kc ≈ 2σ = {:.2} (critical coupling)", 2.0 * freq_std);
    println!("  Topological precursors should emerge before Kc.");

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Analysis Complete");
    println!("═══════════════════════════════════════════════════════════════");
}
