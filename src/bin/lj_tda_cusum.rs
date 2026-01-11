//! Lennard-Jones TDA-CUSUM: Crystallization Transition Detection
//!
//! This binary demonstrates the TDA-CUSUM framework for detecting
//! the onset of crystallization in a supercooled Lennard-Jones liquid.
//!
//! ## Protocol
//!
//! 1. Initialize LJ system at high temperature (liquid phase)
//! 2. Collect baseline topological statistics (calibration)
//! 3. Begin cooling while monitoring CUSUM
//! 4. Detect topological precursors of crystallization
//! 5. Compare t_CUSUM with t_physical (Q6 order parameter)

use tda_info_dynamics::{
    LennardJonesSystem,
    VietorisRips,
    compute_persistence,
    compute_entropy,
    CusumDetector,
    BettiCurve,
    PersistenceLandscape,
};

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  TDA-CUSUM: Lennard-Jones Crystallization Detection");
    println!("  Francisco Molina Burgos - 2026");
    println!("═══════════════════════════════════════════════════════════════\n");

    // System parameters
    let n_particles = 108;  // 3³ × 4 for FCC
    let density = 0.85;     // Reduced density ρ*
    let t_initial = 2.0;    // Initial temperature T* (liquid)
    let t_final = 0.5;      // Final temperature T* (crystal)

    println!("System Parameters:");
    println!("  N = {} particles", n_particles);
    println!("  ρ* = {:.2}", density);
    println!("  T*: {:.2} → {:.2}", t_initial, t_final);
    println!();

    // Initialize system
    println!("Initializing Lennard-Jones system...");
    let mut system = LennardJonesSystem::new(n_particles, density, t_initial);

    // Equilibrate at high temperature
    println!("Equilibrating at T* = {:.2}...", t_initial);
    for _ in 0..5 {
        system.run(1000);
        system.thermostat(t_initial);
    }

    // Calibration phase: collect baseline statistics
    println!("\nCalibration Phase (liquid baseline)...");
    let mut calibration_entropy: Vec<f64> = Vec::new();
    let mut calibration_beta1: Vec<f64> = Vec::new();

    for i in 0..20 {
        system.run(500);
        system.thermostat(t_initial);

        // Compute distance matrix
        let dist = system.distance_matrix();

        // Build Vietoris-Rips complex
        let vr = VietorisRips::from_distance_matrix(&dist, 3.0, 50);

        // Compute persistence
        let pd = compute_persistence(&vr);

        // Compute entropy
        let entropy = compute_entropy(&pd);
        calibration_entropy.push(entropy.persistent_entropy);

        // Compute integrated β₁
        let betti_curve = BettiCurve::compute(&vr);
        let int_beta1 = betti_curve.integrated_beta_1();
        calibration_beta1.push(int_beta1);

        if i % 5 == 0 {
            println!(
                "  Sample {}: H_P = {:.4}, ∫β₁ = {:.4}, T* = {:.3}",
                i + 1,
                entropy.persistent_entropy,
                int_beta1,
                system.state().temperature
            );
        }
    }

    // Initialize CUSUM detectors
    let mut cusum_entropy = CusumDetector::with_params(0.5, 4.0);
    let mut cusum_beta1 = CusumDetector::with_params(0.5, 4.0);

    cusum_entropy.calibrate(&calibration_entropy);
    cusum_beta1.calibrate(&calibration_beta1);

    println!("\nCUSUM Calibrated:");
    println!(
        "  Entropy: μ₀ = {:.4}, σ₀ = {:.4}",
        cusum_entropy.result().reference_mean,
        cusum_entropy.result().reference_std
    );
    println!(
        "  β₁:      μ₀ = {:.4}, σ₀ = {:.4}",
        cusum_beta1.result().reference_mean,
        cusum_beta1.result().reference_std
    );

    // Cooling phase with CUSUM monitoring
    println!("\n══════════════════════════════════════════════════════════════");
    println!("  Cooling Phase: Monitoring for Crystallization Precursors");
    println!("══════════════════════════════════════════════════════════════\n");

    let cooling_rate = 0.02;
    let steps_per_sample = 500;
    let mut current_temp = t_initial;

    let mut entropy_detected = false;
    let mut beta1_detected = false;
    let mut t_cusum_entropy = None;
    let mut t_cusum_beta1 = None;

    let mut step = 0;

    while current_temp > t_final && (!entropy_detected || !beta1_detected) {
        // Cool slightly
        current_temp *= 1.0 - cooling_rate;
        system.thermostat(current_temp);
        system.run(steps_per_sample);

        // Compute topological statistics
        let dist = system.distance_matrix();
        let vr = VietorisRips::from_distance_matrix(&dist, 3.0, 50);
        let pd = compute_persistence(&vr);
        let entropy = compute_entropy(&pd);
        let betti_curve = BettiCurve::compute(&vr);
        let int_beta1 = betti_curve.integrated_beta_1();

        // Update CUSUM
        if !entropy_detected && cusum_entropy.update(entropy.persistent_entropy) {
            entropy_detected = true;
            t_cusum_entropy = Some(current_temp);
            println!(
                ">>> ENTROPY DETECTION at T* = {:.4} (step {})",
                current_temp, step
            );
        }

        if !beta1_detected && cusum_beta1.update(int_beta1) {
            beta1_detected = true;
            t_cusum_beta1 = Some(current_temp);
            println!(
                ">>> β₁ DETECTION at T* = {:.4} (step {})",
                current_temp, step
            );
        }

        if step % 5 == 0 {
            println!(
                "Step {:3}: T* = {:.4}, H_P = {:.4}, ∫β₁ = {:.4}, C_H = {:.2}, C_β = {:.2}",
                step,
                current_temp,
                entropy.persistent_entropy,
                int_beta1,
                cusum_entropy.current_value(),
                cusum_beta1.current_value()
            );
        }

        step += 1;
    }

    // Final analysis
    println!("\n══════════════════════════════════════════════════════════════");
    println!("  Results");
    println!("══════════════════════════════════════════════════════════════\n");

    if let Some(t) = t_cusum_entropy {
        println!("Entropy CUSUM detection temperature: T* = {:.4}", t);
    } else {
        println!("Entropy CUSUM: No detection");
    }

    if let Some(t) = t_cusum_beta1 {
        println!("β₁ CUSUM detection temperature:     T* = {:.4}", t);
    } else {
        println!("β₁ CUSUM: No detection");
    }

    // Note: Full Q6 analysis would require spherical harmonics implementation
    // which is available in the related TDA-Complex-Systems repository
    println!("\nNote: Compare with Q6 order parameter for t_physical estimation.");
    println!("See TDA-Complex-Systems repository for full Steinhardt Q6 analysis.");

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Analysis Complete");
    println!("═══════════════════════════════════════════════════════════════");
}
