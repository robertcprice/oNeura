//! Paper Demo: Stress-Resilient Ecosystem Design
//!
//! Runs a full NSGA-II Pareto evolution with stress resilience across
//! all 7 fitness objectives (biomass, biodiversity, stability, carbon,
//! fruit, microbial, fly_metabolism), exports telemetry
//! and Pareto front, and prints a comprehensive summary.
//!
//! Usage:
//!   paper_demo [options]
//!   paper_demo --population 16 --generations 10 --frames 200 --lite

use oneura_core::terrarium::{
    ecosystem_dashboard, evolve_pareto_stressed, run_and_export, sparkline,
    telemetry_from_pareto_result, EvolutionConfig, FitnessConfig, FitnessObjective,
    GenomeConstraints, SearchStrategy,
};
use std::time::Instant;

fn main() {
    let mut population: usize = 16;
    let mut generations: usize = 10;
    let mut frames: usize = 200;
    let mut lite = true;
    let mut seed: u64 = 42;
    let mut telemetry_path = String::from("experiments/paper_telemetry.json");
    let mut pareto_path = String::from("experiments/paper_pareto.json");
    let mut export_best = true;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--population" => {
                population = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(population);
                i += 1;
            }
            "--generations" => {
                generations = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(generations);
                i += 1;
            }
            "--frames" => {
                frames = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(frames);
                i += 1;
            }
            "--lite" => {
                lite = true;
            }
            "--full" => {
                lite = false;
            }
            "--seed" => {
                seed = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(seed);
                i += 1;
            }
            "--telemetry" => {
                telemetry_path = args.get(i + 1).cloned().unwrap_or(telemetry_path);
                i += 1;
            }
            "--pareto" => {
                pareto_path = args.get(i + 1).cloned().unwrap_or(pareto_path);
                i += 1;
            }
            "--no-export" => {
                export_best = false;
            }
            "--help" | "-h" => {
                eprintln!("Paper Demo: Stress-Resilient Ecosystem Design");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  --population <N>     Population size (default: 16)");
                eprintln!("  --generations <N>    Generations to evolve (default: 10)");
                eprintln!("  --frames <N>         Frames per world (default: 200)");
                eprintln!("  --lite               Use lite 10x8 worlds (default)");
                eprintln!("  --full               Use full 20x16 worlds");
                eprintln!("  --seed <N>           Random seed (default: 42)");
                eprintln!("  --telemetry <PATH>   Telemetry output (default: experiments/paper_telemetry.json)");
                eprintln!("  --pareto <PATH>      Pareto front output (default: experiments/paper_pareto.json)");
                eprintln!("  --no-export          Skip exporting best world");
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown arg: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    eprintln!("╔══════════════════════════════════════════════════════╗");
    eprintln!("║  Paper Demo: Stress-Resilient Ecosystem Design      ║");
    eprintln!("╚══════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!("Configuration:");
    eprintln!("  Population:   {}", population);
    eprintln!("  Generations:  {}", generations);
    eprintln!("  Frames/world: {}", frames);
    eprintln!(
        "  Grid:         {}",
        if lite { "10x8 (lite)" } else { "20x16 (full)" }
    );
    eprintln!("  Seed:         {}", seed);
    eprintln!();

    let config = EvolutionConfig {
        population_size: population,
        generations,
        frames_per_world: frames,
        master_seed: seed,
        lite,
        constraints: GenomeConstraints {
            max_fly_count: None,
            max_microbe_count: None,
            force_fly_count: None,
        },
        fitness: FitnessConfig {
            primary: FitnessObjective::MaxBiomass,
            snapshot_interval: 10,
        },
        strategy: SearchStrategy::Evolutionary {
            tournament_size: 3,
            mutation_rate: 0.15,
            crossover_rate: 0.7,
            elitism: 2,
        },
        thread_count: None,
    };

    // Phase 1: Pareto + Stress evolution
    eprintln!("Phase 1: NSGA-II Pareto + Stress Resilience...");
    let t0 = Instant::now();

    match evolve_pareto_stressed(config) {
        Ok(pareto_result) => {
            let t_evolve = t0.elapsed().as_secs_f64();

            eprintln!();
            eprintln!("=== Evolution Results ===");
            eprintln!("Time:           {:.1}s", t_evolve);
            eprintln!(
                "Pareto front:   {} solutions",
                pareto_result.pareto_front.len()
            );
            eprintln!("Worlds eval'd:  {}", pareto_result.total_worlds_evaluated);

            // Print trade-off summary
            if !pareto_result.pareto_front.is_empty() {
                let obj_names = [
                    "biomass",
                    "biodiversity",
                    "stability",
                    "carbon",
                    "fruit",
                    "microbial",
                    "fly_metab",
                ];

                // Extract objective values as vectors per objective
                let obj_vecs: Vec<Vec<f32>> = (0..obj_names.len())
                    .map(|j| {
                        pareto_result
                            .pareto_front
                            .iter()
                            .map(|p| {
                                let o = &p.objectives;
                                match j {
                                    0 => o.biomass,
                                    1 => o.biodiversity,
                                    2 => o.stability,
                                    3 => o.carbon,
                                    4 => o.fruit,
                                    5 => o.microbial,
                                    6 => o.fly_metabolism,
                                    _ => 0.0,
                                }
                            })
                            .collect()
                    })
                    .collect();

                eprintln!();
                eprintln!("--- Trade-off Analysis (Pareto Front) ---");

                for (j, vals) in obj_vecs.iter().enumerate() {
                    let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mean = vals.iter().sum::<f32>() / vals.len().max(1) as f32;
                    let spark = sparkline(vals, 20);
                    eprintln!(
                        "  {:12} min={:8.3} mean={:8.3} max={:8.3}  {}",
                        obj_names[j], min, mean, max, spark
                    );
                }

                // Stress metrics from telemetry (generated from pareto result)
                let telemetry = telemetry_from_pareto_result(&pareto_result);
                if let Some(ref sm) = telemetry.stress_metrics {
                    let resistance = if sm.pre_stress_biomass > 1e-6 {
                        sm.min_stress_biomass / sm.pre_stress_biomass
                    } else {
                        0.0
                    };
                    let recovery = if sm.pre_stress_biomass > 1e-6 {
                        sm.post_recovery_biomass / sm.pre_stress_biomass
                    } else {
                        0.0
                    };
                    eprintln!();
                    eprintln!("--- Resilience Metrics (Final Gen) ---");
                    eprintln!("  Pre-stress biomass:  {:.3}", sm.pre_stress_biomass);
                    eprintln!("  Min stress biomass:  {:.3}", sm.min_stress_biomass);
                    eprintln!("  Post-recovery:       {:.3}", sm.post_recovery_biomass);
                    eprintln!("  Recovery ratio:      {:.3}", recovery);
                    eprintln!("  Resistance:          {:.3}", resistance);
                }

                // Parameter sensitivity — show variance of genome params across Pareto front
                eprintln!();
                eprintln!("--- Parameter Sensitivity ---");
                let param_names = [
                    "proton_scale",
                    "temperature",
                    "water_count",
                    "water_vol",
                    "moisture",
                    "plant_count",
                    "fruit_count",
                    "fly_count",
                    "microbes",
                    "fly_psc",
                    "fly_neural",
                    "fly_scale",
                    "resp_vmax",
                    "nitr_vmax",
                    "photo_vmax",
                    "miner_vmax",
                    "enzyme_x",
                    "enzyme_y",
                    "seed",
                    "time_warp",
                ];
                let norm_vecs: Vec<Vec<f32>> = pareto_result
                    .pareto_front
                    .iter()
                    .map(|p| p.genome.normalized_params())
                    .collect();
                if !norm_vecs.is_empty() {
                    let n_params = norm_vecs[0].len().min(param_names.len());
                    for j in 0..n_params {
                        let vals: Vec<f32> = norm_vecs
                            .iter()
                            .map(|v| *v.get(j).unwrap_or(&0.0))
                            .collect();
                        let mean = vals.iter().sum::<f32>() / vals.len().max(1) as f32;
                        let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>()
                            / vals.len().max(1) as f32;
                        let marker = if var > 0.05 { " ** HIGH" } else { "" };
                        eprintln!(
                            "  {:12} mean={:.3} var={:.4}{}",
                            param_names[j], mean, var, marker
                        );
                    }
                }
            }

            // Phase 2: Export telemetry
            eprintln!();
            eprintln!("Phase 2: Exporting results...");

            // Telemetry JSON
            let telemetry = telemetry_from_pareto_result(&pareto_result);
            let json = serde_json::to_string_pretty(&vec![telemetry]).unwrap_or_default();
            match std::fs::write(&telemetry_path, &json) {
                Ok(_) => eprintln!("  Telemetry:  {}", telemetry_path),
                Err(e) => eprintln!("  Error writing telemetry: {}", e),
            }

            // Pareto front JSON
            let pareto_json =
                serde_json::to_string_pretty(&pareto_result.pareto_front).unwrap_or_default();
            match std::fs::write(&pareto_path, &pareto_json) {
                Ok(_) => eprintln!("  Pareto:     {}", pareto_path),
                Err(e) => eprintln!("  Error writing pareto: {}", e),
            }

            // Phase 3: Run best world through ecosystem health
            if export_best && !pareto_result.pareto_front.is_empty() {
                eprintln!();
                eprintln!("Phase 3: Ecosystem health analysis of best solution...");
                let best = &pareto_result.pareto_front[0];
                match run_and_export(best.genome.clone(), frames, lite, 10, None) {
                    Ok(export) => {
                        if let Some(ref health) = export.health_report {
                            eprintln!("  Shannon diversity:   {:.3}", health.shannon_diversity);
                            eprintln!("  Simpson diversity:   {:.3}", health.simpson_diversity);
                            eprintln!("  Resilience score:    {:.3}", health.resilience_score);
                            eprintln!("  Overall health:      {:.3}", health.overall_health);
                            eprintln!("  Trophic levels:      {}", health.trophic_levels);
                        }

                        // Dashboard from snapshots
                        if !export.snapshots.is_empty() {
                            eprintln!();
                            eprintln!("{}", ecosystem_dashboard(&export.snapshots, 60));
                        }
                    }
                    Err(e) => eprintln!("  Export failed: {}", e),
                }
            }

            let total_time = t0.elapsed().as_secs_f64();
            eprintln!();
            eprintln!("╔══════════════════════════════════════════════════════╗");
            eprintln!(
                "║  Demo complete in {:.1}s                              ║",
                total_time
            );
            eprintln!("╚══════════════════════════════════════════════════════╝");
            eprintln!();
            eprintln!("Next steps:");
            eprintln!("  python experiments/evolve_dashboard.py \\");
            eprintln!("    --telemetry {} \\", telemetry_path);
            eprintln!("    --pareto {} \\", pareto_path);
            eprintln!("    --output experiments/figures/");
        }
        Err(e) => {
            eprintln!("Evolution failed: {}", e);
            std::process::exit(1);
        }
    }
}
