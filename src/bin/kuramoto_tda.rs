//! Kuramoto TDA-CUSUM: Synchronization Transition Detection
//!
//! This binary demonstrates the TDA-CUSUM framework for detecting
//! the onset of synchronization in coupled oscillator systems.
//!
//! ## Protocol
//!
//! 1. Initialize Kuramoto system at weak coupling (incoherent)
//! 2. Collect baseline topological statistics on circle embedding
//! 3. Ramp coupling while monitoring CUSUM
//! 4. Detect topological precursors of synchronization
//! 5. Compare t_CUSUM with r(t) order parameter crossing

use tda_info_dynamics::{
    KuramotoSystem,
    VietorisRips,
    compute_persistence,
    compute_entropy,
    CusumDetector,
    PersistenceLandscape,
};

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  TDA-CUSUM: Kuramoto Synchronization Detection");
    println!("  Francisco Molina Burgos - 2026");
    println!("═══════════════════════════════════════════════════════════════\n");

    // System parameters
    let n_oscillators = 100;
    let freq_std = 1.0;     // Frequency spread
    let k_initial = 0.5;    // Initial coupling (incoherent)
    let k_final = 4.0;      // Final coupling (synchronized)
    let r_threshold = 0.7;  // Traditional sync threshold

    println!("System Parameters:");
    println!("  N = {} oscillators", n_oscillators);
    println!("  Frequency std = {:.2}", freq_std);
    println!("  K: {:.2} → {:.2}", k_initial, k_final);
    println!("  Sync threshold r = {:.2}", r_threshold);
    println!();

    // Initialize system
    println!("Initializing Kuramoto system...");
    let mut system = KuramotoSystem::new(n_oscillators, k_initial, freq_std);

    // Equilibrate
    println!("Equilibrating at K = {:.2}...", k_initial);
    system.run(5000);

    // Calibration phase
    println!("\nCalibration Phase (incoherent baseline)...");
    let mut calibration_entropy: Vec<f64> = Vec::new();
    let mut calibration_beta0: Vec<f64> = Vec::new();

    for i in 0..20 {
        system.run(200);

        // Get circle embedding
        let points = system.to_circle_points();

        // Build Vietoris-Rips on 2D circle points
        let vr = VietorisRips::from_points(&points, 2.5, 30);

        // Compute persistence
        let pd = compute_persistence(&vr);

        // Compute entropy
        let entropy = compute_entropy(&pd);
        calibration_entropy.push(entropy.persistent_entropy);

        // Count β₀ at intermediate scale
        let beta0_mid = vr.count_components_at(15) as f64;
        calibration_beta0.push(beta0_mid);

        if i % 5 == 0 {
            println!(
                "  Sample {}: H_P = {:.4}, β₀ = {:.0}, r = {:.4}",
                i + 1,
                entropy.persistent_entropy,
                beta0_mid,
                system.order_parameter()
            );
        }
    }

    // Initialize CUSUM detectors
    let mut cusum_entropy = CusumDetector::with_params(0.5, 4.0);
    let mut cusum_beta0 = CusumDetector::with_params(0.5, 4.0);

    cusum_entropy.calibrate(&calibration_entropy);
    cusum_beta0.calibrate(&calibration_beta0);

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

    // Coupling ramp with CUSUM monitoring
    println!("\n══════════════════════════════════════════════════════════════");
    println!("  Coupling Ramp: Monitoring for Synchronization Precursors");
    println!("══════════════════════════════════════════════════════════════\n");

    let coupling_rate = 0.05;
    let steps_per_sample = 200;
    let mut current_k = k_initial;

    let mut entropy_detected = false;
    let mut beta0_detected = false;
    let mut r_crossed = false;

    let mut t_cusum_entropy = None;
    let mut t_cusum_beta0 = None;
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

        // Compute topological statistics
        let points = system.to_circle_points();
        let vr = VietorisRips::from_points(&points, 2.5, 30);
        let pd = compute_persistence(&vr);
        let entropy = compute_entropy(&pd);
        let beta0_mid = vr.count_components_at(15) as f64;

        // Update CUSUM (inverted for entropy - we expect decrease)
        // Using negative values since entropy decreases during sync
        if !entropy_detected && cusum_entropy.update(-entropy.persistent_entropy) {
            entropy_detected = true;
            t_cusum_entropy = Some(current_k);
            println!(
                ">>> ENTROPY DETECTION at K = {:.4} (step {})",
                current_k, step
            );
        }

        // β₀ should decrease as system synchronizes
        if !beta0_detected && cusum_beta0.update(-beta0_mid) {
            beta0_detected = true;
            t_cusum_beta0 = Some(current_k);
            println!(
                ">>> β₀ DETECTION at K = {:.4} (step {})",
                current_k, step
            );
        }

        if step % 5 == 0 {
            println!(
                "Step {:3}: K = {:.3}, r = {:.4}, H_P = {:.4}, β₀ = {:2.0}, C_H = {:.2}, C_β = {:.2}",
                step,
                current_k,
                r,
                entropy.persistent_entropy,
                beta0_mid,
                cusum_entropy.current_value(),
                cusum_beta0.current_value()
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

    if let Some(k) = k_physical {
        println!("  Traditional r-threshold: K = {:.4}", k);
    } else {
        println!("  Traditional: No synchronization achieved");
    }

    // Early warning analysis
    if let (Some(k_topo), Some(k_phys)) = (t_cusum_entropy.or(t_cusum_beta0), k_physical) {
        let delta_k = k_phys - k_topo;
        if delta_k > 0.0 {
            println!("\n  ✓ EARLY WARNING: ΔK = {:.4}", delta_k);
            println!("    Topological precursor detected BEFORE synchronization!");
        } else {
            println!("\n  × Late detection: ΔK = {:.4}", delta_k);
        }
    }

    // Theoretical context
    println!("\n─────────────────────────────────────────────────────────────");
    println!("Theoretical Note:");
    println!("  For Lorentzian frequency distribution:");
    println!("  Kc = 2σ = {:.2} (critical coupling)", 2.0 * freq_std);
    println!("  Topological precursors should emerge near Kc.");

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Analysis Complete");
    println!("═══════════════════════════════════════════════════════════════");
}
