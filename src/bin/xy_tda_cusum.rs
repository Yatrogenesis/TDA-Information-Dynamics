//! XY Model TDA-CUSUM: BKT Transition Detection
//!
//! This binary demonstrates the TDA-CUSUM framework for detecting
//! the Berezinskii-Kosterlitz-Thouless (BKT) transition.
//!
//! ## Key Insight
//!
//! The BKT transition is characterized by vortex unbinding:
//! - T < T_BKT: Vortex-antivortex pairs bound, quasi-long-range order
//! - T > T_BKT: Free vortices proliferate, correlations decay exponentially
//!
//! Topological signatures:
//! - Vortex density increases sharply at T_BKT
//! - Magnetization drops (though it's zero at all T in infinite system)
//! - TDA on spin configuration can detect changes in topological structure

use tda_info_dynamics::{
    XYModel,
    DynamicalSystem,
    Controllable,
    VietorisRips,
    compute_persistence,
    compute_entropy,
    CusumDetector,
};

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  TDA-CUSUM: XY Model BKT Transition Detection");
    println!("  Francisco Molina Burgos - 2026");
    println!("═══════════════════════════════════════════════════════════════\n");

    // System parameters
    let lattice_size = 16;  // L x L lattice
    let t_initial = 0.3;    // Start in ordered phase
    let t_final = 1.5;      // End in disordered phase
    let t_bkt = XYModel::theoretical_t_bkt();

    println!("System Parameters:");
    println!("  XY Model on {}x{} square lattice", lattice_size, lattice_size);
    println!("  T: {:.2} → {:.2}", t_initial, t_final);
    println!("  Theoretical T_BKT ≈ {:.4}", t_bkt);

    // TDA parameters
    let vr_epsilon = 0.5;
    let vr_steps = 25;
    let target_points = 50;

    println!("\nTDA Parameters:");
    println!("  VR epsilon = {:.2} (for magnetization history)", vr_epsilon);
    println!("  VR steps = {}", vr_steps);
    println!("  Subsampled points = {}", target_points);
    println!("  Metrics: vortex density, magnetization, entropy");

    // Create system and thermalize at low T
    println!("\nThermalizing at T = {:.2}...", t_initial);
    let mut system = XYModel::aligned(lattice_size, t_initial);
    system.run(2000);  // Thermalize
    system.clear_history();

    // Calibration phase
    println!("\nCalibration Phase (ordered state baseline)...");
    let mut cal_vortex: Vec<f64> = Vec::new();
    let mut cal_mag: Vec<f64> = Vec::new();
    let mut cal_entropy: Vec<f64> = Vec::new();

    for i in 0..20 {
        system.run(50);

        let state = system.state();
        cal_vortex.push(state.vortex_density);
        cal_mag.push(-state.magnetization);  // Negative: looking for DECREASE

        if system.n_elements() >= 30 {
            let dist = system.distance_matrix();
            let sub_dist = subsample_distance_matrix(&dist, target_points);
            let vr = VietorisRips::from_distance_matrix(&sub_dist, vr_epsilon, vr_steps);
            let pd = compute_persistence(&vr);
            let entropy = compute_entropy(&pd);
            cal_entropy.push(entropy.persistent_entropy);
        }

        if i % 5 == 0 {
            println!(
                "  Sample {:2}: T = {:.3}, m = {:.4}, ρ_v = {:.4}, E = {:.4}",
                i + 1, system.get_parameter(), state.magnetization, state.vortex_density, state.energy
            );
        }
    }

    // Initialize CUSUM with sigma_min to prevent infinite sensitivity
    let mut cusum_vortex = CusumDetector::with_sigma_min(0.3, 2.5, 0.005);  // vortex density scale ~0.1
    let mut cusum_mag = CusumDetector::with_sigma_min(0.3, 2.5, 0.05);      // magnetization scale 0-1
    let mut cusum_entropy = CusumDetector::with_sigma_min(0.3, 3.0, 0.1);

    cusum_vortex.calibrate(&cal_vortex);
    cusum_mag.calibrate(&cal_mag);
    if !cal_entropy.is_empty() {
        cusum_entropy.calibrate(&cal_entropy);
    }

    println!("\nCUSUM Calibrated:");
    println!(
        "  Vortex: μ₀ = {:.4}, σ₀ = {:.4}",
        cusum_vortex.result().reference_mean,
        cusum_vortex.result().reference_std
    );
    println!(
        "  Mag:    μ₀ = {:.4}, σ₀ = {:.4}",
        -cusum_mag.result().reference_mean,
        cusum_mag.result().reference_std
    );

    // Temperature ramp
    println!("\n══════════════════════════════════════════════════════════════");
    println!("  Temperature Ramp: Monitoring for BKT Transition");
    println!("══════════════════════════════════════════════════════════════\n");

    let temp_rate = 0.01;
    let mc_per_sample = 50;
    let vortex_threshold = 0.02;  // Traditional: vortex density > 2%

    let mut current_t = t_initial;

    let mut vortex_detected = false;
    let mut mag_detected = false;
    let mut entropy_detected = false;
    let mut traditional_crossed = false;

    let mut t_cusum_vortex = None;
    let mut t_cusum_mag = None;
    let mut t_cusum_entropy = None;
    let mut t_traditional = None;

    let mut step = 0;
    system.clear_history();

    while current_t < t_final {
        current_t += temp_rate;
        system.set_temperature(current_t);
        system.run(mc_per_sample);

        let state = system.state();

        // Traditional threshold
        if !traditional_crossed && state.vortex_density > vortex_threshold {
            traditional_crossed = true;
            t_traditional = Some(current_t);
            println!(
                ">>> TRADITIONAL (ρ_v > {}) at T = {:.4} (step {})",
                vortex_threshold, current_t, step
            );
        }

        // CUSUM updates
        if !vortex_detected && cusum_vortex.update(state.vortex_density) {
            vortex_detected = true;
            t_cusum_vortex = Some(current_t);
            println!(
                ">>> VORTEX DETECTION at T = {:.4} (step {})",
                current_t, step
            );
        }

        if !mag_detected && cusum_mag.update(-state.magnetization) {
            mag_detected = true;
            t_cusum_mag = Some(current_t);
            println!(
                ">>> MAGNETIZATION DETECTION at T = {:.4} (step {})",
                current_t, step
            );
        }

        // Entropy from TDA
        if system.n_elements() >= 30 {
            let dist = system.distance_matrix();
            let sub_dist = subsample_distance_matrix(&dist, target_points);
            let vr = VietorisRips::from_distance_matrix(&sub_dist, vr_epsilon, vr_steps);
            let pd = compute_persistence(&vr);
            let entropy = compute_entropy(&pd);

            if !entropy_detected && cusum_entropy.update(entropy.persistent_entropy) {
                entropy_detected = true;
                t_cusum_entropy = Some(current_t);
                println!(
                    ">>> ENTROPY DETECTION at T = {:.4} (step {})",
                    current_t, step
                );
            }
        }

        if step % 10 == 0 {
            println!(
                "Step {:3}: T = {:.3}, m = {:.4}, ρ_v = {:.4}, E = {:.3}",
                step, current_t, state.magnetization, state.vortex_density, state.energy
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

    if let Some(t) = t_cusum_vortex {
        println!("  Vortex CUSUM detection: T = {:.4}", t);
    } else {
        println!("  Vortex CUSUM: No detection");
    }

    if let Some(t) = t_cusum_mag {
        println!("  Magnetization CUSUM:    T = {:.4}", t);
    } else {
        println!("  Magnetization CUSUM: No detection");
    }

    if let Some(t) = t_cusum_entropy {
        println!("  Entropy CUSUM:          T = {:.4}", t);
    } else {
        println!("  Entropy CUSUM: No detection");
    }

    if let Some(t) = t_traditional {
        println!("  Traditional (ρ_v):      T = {:.4}", t);
    } else {
        println!("  Traditional: Threshold not crossed");
    }

    println!("  Theoretical T_BKT:      T ≈ {:.4}", t_bkt);

    // Early warning
    let first_topo = t_cusum_vortex.or(t_cusum_mag).or(t_cusum_entropy);
    if let (Some(t_topo), Some(t_trad)) = (first_topo, t_traditional) {
        let delta_t = t_trad - t_topo;
        if delta_t > 0.0 {
            println!("\n  ✓ EARLY WARNING: ΔT = {:.4}", delta_t);
            println!("    Topological precursor detected BEFORE threshold!");
        } else if delta_t < 0.0 {
            println!("\n  Detection after threshold: ΔT = {:.4}", delta_t);
        }
    }

    println!("\n─────────────────────────────────────────────────────────────");
    println!("Theoretical Note:");
    println!("  The BKT transition is a topological phase transition.");
    println!("  Below T_BKT ≈ {:.3}, vortex-antivortex pairs are bound.", t_bkt);
    println!("  Above T_BKT, free vortices proliferate.");
    println!("  This is captured by the vortex density ρ_v increasing");
    println!("  and the magnetization decreasing.");

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Analysis Complete");
    println!("═══════════════════════════════════════════════════════════════");
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
