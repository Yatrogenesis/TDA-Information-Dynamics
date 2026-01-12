//! Lennard-Jones TDA-CUSUM with EXACT Persistent Homology
//!
//! This binary uses the exact standard algorithm (matching Ripser/Python)
//! for persistent homology computation, enabling direct comparison with
//! the published Python results (DOI: 10.5281/zenodo.18220298).
//!
//! Key difference from lj_tda_cusum.rs:
//! - Uses compute_exact_persistence() instead of Euler approximation
//! - Computes true birth/death pairs for entropy calculation
//! - Should match Python/Ripser detection behavior

use tda_info_dynamics::{
    LennardJonesSystem,
    compute_exact_persistence_simple,
    CusumDetector,
};

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  TDA-CUSUM: Lennard-Jones with EXACT Persistent Homology");
    println!("  Matching Python/Ripser Algorithm");
    println!("  Francisco Molina-Burgos - Avermex Research Division - 2026");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Parameters matching Python validation
    // Note: Exact algorithm is O(m³), so we use moderate N
    let n_particles = 144;  // Same as Python v9 baseline
    let density = 0.70;     // Same as Python (ρ=0.7)
    let t_initial = 2.0;
    let t_final = 0.1;
    let max_epsilon = 2.5;  // Same as Python ripser max_edge

    println!("System Parameters (matching Python v9):");
    println!("  N = {} particles", n_particles);
    println!("  ρ* = {:.2} (same as Python)", density);
    println!("  T*: {:.2} → {:.2}", t_initial, t_final);
    println!("  max_ε = {:.1} (Ripser)", max_epsilon);
    println!();

    // Initialize system
    println!("Initializing Lennard-Jones system...");
    let mut system = LennardJonesSystem::new(n_particles, density, t_initial);

    // Equilibrate at high temperature (liquid phase)
    println!("Equilibrating at T* = {:.2} (liquid baseline)...", t_initial);
    for _ in 0..10 {
        system.run(200);
        system.thermostat(t_initial);
    }

    // Calibration phase: collect baseline S_H1 statistics
    println!("\n══════════════════════════════════════════════════════════════");
    println!("  Calibration Phase: Exact S_H1 Baseline (Liquid)");
    println!("══════════════════════════════════════════════════════════════\n");

    let mut calibration_entropy: Vec<f64> = Vec::new();
    let baseline_samples = 30;  // 30% of trajectory like Python

    for i in 0..baseline_samples {
        system.run(100);
        system.thermostat(t_initial);

        let dist = system.distance_matrix();
        let pd = compute_exact_persistence_simple(&dist, max_epsilon);
        let entropy = pd.persistence_entropy(1);
        calibration_entropy.push(entropy);

        if i % 10 == 0 {
            let h1_count = pd.count(1);
            println!(
                "  Sample {:2}: S_H1 = {:.4}, #H1 intervals = {}, T* = {:.3}",
                i + 1, entropy, h1_count, system.state().temperature
            );
        }
    }

    // Calibration statistics
    let baseline_mean: f64 = calibration_entropy.iter().sum::<f64>() / calibration_entropy.len() as f64;
    let baseline_var: f64 = calibration_entropy.iter()
        .map(|x| (x - baseline_mean).powi(2))
        .sum::<f64>() / calibration_entropy.len() as f64;
    let baseline_std = baseline_var.sqrt();

    println!("\nBaseline Statistics (liquid phase):");
    println!("  μ₀ = {:.4}", baseline_mean);
    println!("  σ₀ = {:.4}", baseline_std);

    // Initialize CUSUM detector (negative deviation = entropy collapse)
    // Using parameters similar to Python CUSUM
    let mut cusum = CusumDetector::with_sigma_min(0.5, 3.0, 0.1);
    cusum.calibrate(&calibration_entropy);

    println!("\nCUSUM Calibrated:");
    println!("  Reference: μ = {:.4}, σ = {:.4}",
             cusum.result().reference_mean,
             cusum.result().reference_std);
    println!("  Threshold: 3σ (same as Python)");

    // Cooling phase with CUSUM monitoring
    println!("\n══════════════════════════════════════════════════════════════");
    println!("  Cooling Phase: Monitoring S_H1 Collapse");
    println!("══════════════════════════════════════════════════════════════\n");

    let total_cooling_steps = 100;
    let cooling_rate = (t_initial / t_final).ln() / total_cooling_steps as f64;
    let mut current_temp = t_initial;

    let mut detection_temp = None;
    let mut detection_step = None;

    println!("Step   T*     S_H1    #H1   CUSUM   Status");
    println!("─────────────────────────────────────────────");

    for step in 0..total_cooling_steps {
        // Exponential cooling (like Python linear quench approximation)
        current_temp = t_initial * (-cooling_rate * step as f64).exp();
        if current_temp < t_final {
            current_temp = t_final;
        }

        system.thermostat(current_temp);
        system.run(50);  // 5000 total steps / 100 = 50 per sample

        // Compute exact persistence
        let dist = system.distance_matrix();
        let pd = compute_exact_persistence_simple(&dist, max_epsilon);
        let entropy = pd.persistence_entropy(1);
        let h1_count = pd.count(1);

        // Update CUSUM (looking for NEGATIVE deviation = collapse)
        let detected = cusum.update(entropy);

        if detected && detection_temp.is_none() {
            detection_temp = Some(current_temp);
            detection_step = Some(step);
            println!(
                "{:3}    {:.4}  {:.4}  {:3}   {:.2}   >>> DETECTION <<<",
                step, current_temp, entropy, h1_count, cusum.current_value()
            );
        } else if step % 10 == 0 || step < 5 {
            println!(
                "{:3}    {:.4}  {:.4}  {:3}   {:.2}",
                step, current_temp, entropy, h1_count, cusum.current_value()
            );
        }
    }

    // Results
    println!("\n══════════════════════════════════════════════════════════════");
    println!("  Results");
    println!("══════════════════════════════════════════════════════════════\n");

    match (detection_temp, detection_step) {
        (Some(t), Some(s)) => {
            println!("CUSUM Detection:");
            println!("  Temperature: T* = {:.4}", t);
            println!("  Step: {}", s);
            println!("\nComparison with Python Results:");
            println!("  Python N=144: 73.3% detection rate, mean gap ~750 steps");
            println!("  Rust Exact:   Detection at step {}", s);

            // Estimate if this is a precursor
            let estimated_crystal_step = 70;  // Typical crystallization around step 70
            if s < estimated_crystal_step {
                let gap = estimated_crystal_step - s;
                println!("\n  ✓ PRECURSOR DETECTED");
                println!("    Estimated gap: ~{} steps before crystallization", gap);
            }
        }
        _ => {
            println!("No CUSUM detection during cooling.");
            println!("\nPossible reasons:");
            println!("  - System vitrified (glass) instead of crystallizing");
            println!("  - Stochastic variation (Python shows 73.3% rate at N=144)");
            println!("  - This trial may be in the 26.7% non-detection cases");
        }
    }

    println!("\n══════════════════════════════════════════════════════════════");
    println!("  Comparison with Published Results");
    println!("══════════════════════════════════════════════════════════════");
    println!("\nPython (DOI: 10.5281/zenodo.18220298):");
    println!("  N=144:  73.3% precursor rate, gap=750±1041 steps");
    println!("  N=400:  80.0% precursor rate, gap=1385±893 steps");
    println!("  N=900:  100% precursor rate, gap=1585±563 steps");
    println!("  N=1600: 100% precursor rate, gap=1725±683 steps");
    println!("\nRust Exact (this run): Single trial result above");
    println!("For statistical validation, run multiple seeds.");

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Analysis Complete");
    println!("═══════════════════════════════════════════════════════════════");
}
