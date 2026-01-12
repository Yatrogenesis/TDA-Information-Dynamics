//! Lennard-Jones: Exact Persistence vs Euler Approximation
//!
//! This binary compares the exact persistent homology (standard algorithm)
//! with the Euler approximation to validate that both detect the same
//! physical phenomenon during crystallization.
//!
//! The exact algorithm matches Python/Ripser output.
//! The Euler approximation is faster but only provides Betti number counts.

use tda_info_dynamics::{
    LennardJonesSystem,
    compute_exact_persistence_simple,
    VietorisRips,
    compute_persistence,
    compute_entropy,
    CusumDetector,
};

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Exact Persistence vs Euler Approximation");
    println!("  Lennard-Jones Crystallization Detection");
    println!("  Francisco Molina-Burgos - Avermex Research Division - 2026");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Use smaller system for comparison (exact algorithm is O(n³))
    let n_particles = 64;  // 4³ FCC
    let density = 0.85;
    let t_initial = 2.0;
    let t_final = 0.5;

    println!("System Parameters:");
    println!("  N = {} particles (small for exact comparison)", n_particles);
    println!("  ρ* = {:.2}", density);
    println!("  T*: {:.2} → {:.2}\n", t_initial, t_final);

    // Initialize system
    println!("Initializing Lennard-Jones system...");
    let mut system = LennardJonesSystem::new(n_particles, density, t_initial);

    // Equilibrate
    println!("Equilibrating at T* = {:.2}...\n", t_initial);
    for _ in 0..5 {
        system.run(500);
        system.thermostat(t_initial);
    }

    // Calibration phase
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Calibration Phase: Comparing Methods at Baseline");
    println!("═══════════════════════════════════════════════════════════════\n");

    let mut exact_entropies: Vec<f64> = Vec::new();
    let mut euler_entropies: Vec<f64> = Vec::new();

    for i in 0..10 {
        system.run(200);
        system.thermostat(t_initial);

        let dist = system.distance_matrix();
        let max_eps = 3.0;

        // Exact persistence
        let exact_pd = compute_exact_persistence_simple(&dist, max_eps);
        let exact_entropy = exact_pd.persistence_entropy(1);
        exact_entropies.push(exact_entropy);

        // Euler approximation
        let vr = VietorisRips::from_distance_matrix(&dist, max_eps, 50);
        let euler_pd = compute_persistence(&vr);
        let euler_entropy = compute_entropy(&euler_pd);
        euler_entropies.push(euler_entropy.persistent_entropy);

        if i % 3 == 0 {
            println!(
                "Sample {:2}: Exact S_H1 = {:.4}, Euler S_H1 = {:.4}, Δ = {:.4}",
                i + 1,
                exact_entropy,
                euler_entropy.persistent_entropy,
                (exact_entropy - euler_entropy.persistent_entropy).abs()
            );
        }
    }

    // Statistics
    let exact_mean: f64 = exact_entropies.iter().sum::<f64>() / exact_entropies.len() as f64;
    let euler_mean: f64 = euler_entropies.iter().sum::<f64>() / euler_entropies.len() as f64;

    let exact_var: f64 = exact_entropies.iter().map(|x| (x - exact_mean).powi(2)).sum::<f64>()
        / exact_entropies.len() as f64;
    let euler_var: f64 = euler_entropies.iter().map(|x| (x - euler_mean).powi(2)).sum::<f64>()
        / euler_entropies.len() as f64;

    println!("\nBaseline Statistics:");
    println!("  Exact:  μ = {:.4}, σ = {:.4}", exact_mean, exact_var.sqrt());
    println!("  Euler:  μ = {:.4}, σ = {:.4}", euler_mean, euler_var.sqrt());
    println!("  Δμ = {:.4}", (exact_mean - euler_mean).abs());

    // Initialize CUSUM detectors for both
    let mut cusum_exact = CusumDetector::with_sigma_min(0.5, 4.0, 0.1);
    let mut cusum_euler = CusumDetector::with_sigma_min(0.5, 4.0, 0.1);

    cusum_exact.calibrate(&exact_entropies);
    cusum_euler.calibrate(&euler_entropies);

    // Cooling phase
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Cooling Phase: Monitoring with Both Methods");
    println!("═══════════════════════════════════════════════════════════════\n");

    let cooling_rate = 0.02;
    let mut current_temp = t_initial;
    let mut exact_detected = false;
    let mut euler_detected = false;
    let mut t_exact = None;
    let mut t_euler = None;
    let mut step = 0;

    println!("Step   T*     | Exact S_H1  CUSUM | Euler S_H1  CUSUM");
    println!("-------|------|-------------------|------------------");

    while current_temp > t_final && step < 100 {
        current_temp *= 1.0 - cooling_rate;
        system.thermostat(current_temp);
        system.run(500);

        let dist = system.distance_matrix();
        let max_eps = 3.0;

        // Exact
        let exact_pd = compute_exact_persistence_simple(&dist, max_eps);
        let exact_entropy = exact_pd.persistence_entropy(1);

        // Euler
        let vr = VietorisRips::from_distance_matrix(&dist, max_eps, 50);
        let euler_pd = compute_persistence(&vr);
        let euler_entropy = compute_entropy(&euler_pd).persistent_entropy;

        // Update CUSUM
        if !exact_detected && cusum_exact.update(exact_entropy) {
            exact_detected = true;
            t_exact = Some(current_temp);
            println!(">>> EXACT DETECTION at T* = {:.4}", current_temp);
        }

        if !euler_detected && cusum_euler.update(euler_entropy) {
            euler_detected = true;
            t_euler = Some(current_temp);
            println!(">>> EULER DETECTION at T* = {:.4}", current_temp);
        }

        if step % 5 == 0 {
            println!(
                "{:3}    {:.4} | {:.4}     {:.2}    | {:.4}     {:.2}",
                step,
                current_temp,
                exact_entropy,
                cusum_exact.current_value(),
                euler_entropy,
                cusum_euler.current_value()
            );
        }

        step += 1;
    }

    // Results
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Results Comparison");
    println!("═══════════════════════════════════════════════════════════════\n");

    match (t_exact, t_euler) {
        (Some(te), Some(tu)) => {
            println!("Exact algorithm detection:  T* = {:.4}", te);
            println!("Euler approximation detection: T* = {:.4}", tu);
            println!("Temperature difference: ΔT* = {:.4}", (te - tu).abs());

            if (te - tu).abs() < 0.1 {
                println!("\n✓ Both methods detect transition at similar temperature");
                println!("  The Euler approximation is valid for this system.");
            } else {
                println!("\n⚠ Methods differ significantly");
                println!("  Consider using exact algorithm for precise results.");
            }
        }
        (Some(te), None) => {
            println!("Exact algorithm detection:  T* = {:.4}", te);
            println!("Euler approximation: No detection");
            println!("\n⚠ Euler approximation missed the transition");
        }
        (None, Some(tu)) => {
            println!("Exact algorithm: No detection");
            println!("Euler approximation detection: T* = {:.4}", tu);
            println!("\n⚠ Possible false positive from Euler approximation");
        }
        (None, None) => {
            println!("Neither method detected a transition");
            println!("System may have vitrified (glass) instead of crystallizing");
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Analysis Complete");
    println!("═══════════════════════════════════════════════════════════════");
    println!("\nNote: Exact algorithm matches Python/Ripser output.");
    println!("      Euler approximation is faster but loses birth/death info.");
}
