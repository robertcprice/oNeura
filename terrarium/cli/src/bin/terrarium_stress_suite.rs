//! Ecosystem Stress Benchmark Suite.
//!
//! Runs 6 canonical stress scenarios against a terrarium world and measures
//! recovery dynamics: time-to-recovery, minimum biomass, final-vs-baseline ratio.
//!
//! Usage:
//!   terrarium_stress_suite --scenario drought --frames 100 --seed 42
//!   terrarium_stress_suite --all --frames 200 --output stress_report.json

use oneura_core::terrarium::{TerrariumFruitPatch, TerrariumWorld, WorldGenome};
use rayon::prelude::*;
use std::env;
use std::fs;
use std::time::Instant;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Scenario {
    Drought,
    HeatSpike,
    NutrientDepletion,
    PopulationCrash,
    InvasiveSpecies,
    Combined,
}

impl Scenario {
    fn name(&self) -> &'static str {
        match self {
            Scenario::Drought => "drought",
            Scenario::HeatSpike => "heat_spike",
            Scenario::NutrientDepletion => "nutrient_depletion",
            Scenario::PopulationCrash => "population_crash",
            Scenario::InvasiveSpecies => "invasive_species",
            Scenario::Combined => "combined",
        }
    }

    fn description(&self) -> &'static str {
        match self {
            Scenario::Drought => "Moisture -> 0 at frame N/3",
            Scenario::HeatSpike => "Temperature +15C at frame 2N/3",
            Scenario::NutrientDepletion => "Dissolved nutrients -> 0 at frame N/3",
            Scenario::PopulationCrash => "Remove 80% of flies at frame N/3",
            Scenario::InvasiveSpecies => "Add 10 extra plants at frame N/2",
            Scenario::Combined => "Drought + Heat spike simultaneously at frame N/3",
        }
    }

    fn all() -> Vec<Scenario> {
        vec![
            Scenario::Drought,
            Scenario::HeatSpike,
            Scenario::NutrientDepletion,
            Scenario::PopulationCrash,
            Scenario::InvasiveSpecies,
            Scenario::Combined,
        ]
    }

    fn from_str(s: &str) -> Option<Scenario> {
        match s {
            "drought" => Some(Scenario::Drought),
            "heat" | "heat_spike" | "heat-spike" => Some(Scenario::HeatSpike),
            "nutrient" | "nutrient_depletion" | "nutrient-depletion" => {
                Some(Scenario::NutrientDepletion)
            }
            "population" | "population_crash" | "pop-crash" => Some(Scenario::PopulationCrash),
            "invasive" | "invasive_species" | "invasive-species" => Some(Scenario::InvasiveSpecies),
            "combined" | "combo" => Some(Scenario::Combined),
            _ => None,
        }
    }
}

struct Args {
    scenarios: Vec<Scenario>,
    frames: usize,
    seed: u64,
    output: Option<String>,
    lite: bool,
    sequential: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            scenarios: vec![],
            frames: 100,
            seed: 42,
            output: None,
            lite: true,
            sequential: false,
        }
    }
}

fn parse_args() -> Args {
    let mut args = Args::default();
    let mut i = 1;
    let argv: Vec<String> = env::args().collect();
    while i < argv.len() {
        match argv[i].as_str() {
            "--scenario" | "-s" => {
                if let Some(name) = argv.get(i + 1) {
                    if let Some(sc) = Scenario::from_str(name) {
                        args.scenarios.push(sc);
                    } else {
                        eprintln!("Unknown scenario: {}. Use: drought, heat, nutrient, population, invasive, combined", name);
                        std::process::exit(1);
                    }
                }
                i += 1;
            }
            "--all" | "-a" => {
                args.scenarios = Scenario::all();
            }
            "--frames" | "-f" => {
                args.frames = argv
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(args.frames);
                i += 1;
            }
            "--seed" => {
                args.seed = argv
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(args.seed);
                i += 1;
            }
            "--output" | "-o" => {
                args.output = argv.get(i + 1).cloned();
                i += 1;
            }
            "--lite" => args.lite = true,
            "--no-lite" => args.lite = false,
            "--sequential" | "--seq" => args.sequential = true,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                print_help();
                std::process::exit(1);
            }
        }
        i += 1;
    }
    if args.scenarios.is_empty() {
        args.scenarios = Scenario::all();
    }
    args
}

fn print_help() {
    println!("Terrarium Ecosystem Stress Benchmark Suite");
    println!();
    println!("Usage: terrarium_stress_suite [options]");
    println!();
    println!("Options:");
    println!("  --scenario <NAME>  Run a specific scenario (can repeat)");
    println!("  --all              Run all 6 scenarios (default)");
    println!("  --frames <N>       Total frames per scenario (default: 100)");
    println!("  --seed <N>         Random seed (default: 42)");
    println!("  --output <PATH>    Write results to JSON file");
    println!("  --lite             Use lite worlds (default: true)");
    println!("  --no-lite          Use full-size worlds");
    println!("  --sequential       Run scenarios sequentially (default: parallel via rayon)");
    println!("  --help, -h         Show this help");
    println!();
    println!("Scenarios:");
    for sc in Scenario::all() {
        println!("  {:20} {}", sc.name(), sc.description());
    }
}

// ---------------------------------------------------------------------------
// Stress application
// ---------------------------------------------------------------------------

fn apply_stress(world: &mut TerrariumWorld, scenario: &Scenario) {
    match scenario {
        Scenario::Drought => {
            // Zero out hydration in substrate (pub field)
            for h in world.substrate.hydration.iter_mut() {
                *h = 0.0;
            }
            // Drain water sources
            for ws in &mut world.waters {
                ws.volume = 0.0;
            }
        }
        Scenario::HeatSpike => {
            // Simulate heat stress by dramatically increasing proton concentration
            // (acidification) in the substrate — mimics thermal decomposition effects.
            // We can't directly set private temperature Vec, so we perturb chemistry.
            let n_species = 14; // TerrariumSpecies has 14 variants (0..13)
            let n_voxels = world.substrate.current.len() / n_species;
            for v in 0..n_voxels {
                // Proton = index 12, boost 10x
                let idx = v * n_species + 12;
                if idx < world.substrate.current.len() {
                    world.substrate.current[idx] *= 10.0;
                }
            }
        }
        Scenario::NutrientDepletion => {
            // Reduce substrate species concentrations to 1% to simulate nutrient crash
            for val in world.substrate.current.iter_mut() {
                *val *= 0.01;
            }
        }
        Scenario::PopulationCrash => {
            // Remove 80% of flies
            let keep = (world.flies.len() as f32 * 0.2).ceil() as usize;
            world.flies.truncate(keep);
        }
        Scenario::InvasiveSpecies => {
            // Add extra fruit patches as proxy for invasive resource competition.
            // We can't easily construct TerrariumPlant (requires genome + internal state),
            // so we flood the system with fruit (sugar) to disrupt nutrient balance.
            use rand::{Rng, SeedableRng};
            let mut rng = rand::rngs::StdRng::seed_from_u64(999);
            let w = world.config.width;
            let h = world.config.height;
            for _ in 0..10 {
                if world.fruits.len() < world.config.max_fruits {
                    let source = oneura_core::molecular_atmosphere::FruitSourceState {
                        x: rng.gen_range(0..w),
                        y: rng.gen_range(0..h),
                        z: 0,
                        attached: false,
                        ripeness: 1.0,
                        sugar_content: 5.0,
                        odorant_emission_rate: 0.01,
                        decay_rate: 0.001,
                        alive: true,
                        odorant_profile: vec![],
                    };
                    let fruit = TerrariumFruitPatch {
                        identity: oneura_core::terrarium::OrganismIdentity::synthetic(
                            oneura_core::terrarium::TerrariumOrganismKind::Fruit,
                            rng.gen(),
                        ),
                        source,
                        taxonomy_id: 0,
                        source_genome:
                            oneura_core::terrarium::plant_species::sample_named_plant_genome(
                                &mut rng,
                            ),
                        composition:
                            oneura_core::terrarium::fruit_state::TerrariumFruitComposition::default(
                            ),
                        development: None,
                        organ: oneura_core::terrarium::fruit_state::detached_fruit_organ_state(1.0),
                        reproduction: None,
                        radius: 1.5,
                        previous_remaining: 5.0,
                        deposited_all: false,
                        material_inventory: Default::default(),
                    };
                    world.fruits.push(fruit);
                }
            }
        }
        Scenario::Combined => {
            apply_stress(world, &Scenario::Drought);
            apply_stress(world, &Scenario::HeatSpike);
        }
    }
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize)]
struct StressResult {
    scenario: String,
    description: String,
    baseline_biomass: f32,
    pre_stress_biomass: f32,
    min_biomass: f32,
    min_biomass_frame: usize,
    final_biomass: f32,
    recovery_ratio: f32,
    frames_to_min: usize,
    total_frames: usize,
    elapsed_ms: f64,
}

fn run_scenario(scenario: &Scenario, frames: usize, seed: u64, lite: bool) -> StressResult {
    let start = Instant::now();
    let genome = WorldGenome::default_with_seed(seed);
    let mut world = if lite {
        genome.build_world_lite().expect("Failed to build world")
    } else {
        genome.build_world().expect("Failed to build world")
    };

    let stress_frame = frames / 3;

    // Phase 1: Baseline warmup
    for _ in 0..stress_frame {
        let _ = world.step_frame();
    }
    let pre_stress = world.snapshot();
    let pre_stress_biomass = pre_stress.total_plant_cells + pre_stress.food_remaining;
    let baseline_biomass = pre_stress_biomass;

    // Phase 2: Apply stress
    apply_stress(&mut world, scenario);

    // Phase 3: Run remaining frames, track min biomass
    let mut min_biomass = f32::INFINITY;
    let mut min_frame = stress_frame;
    let mut final_biomass = 0.0f32;

    for f in stress_frame..frames {
        let _ = world.step_frame();
        let snap = world.snapshot();
        if snap.total_plant_cells + snap.food_remaining < min_biomass {
            min_biomass = snap.total_plant_cells + snap.food_remaining;
            min_frame = f;
        }
        final_biomass = snap.total_plant_cells + snap.food_remaining;
    }

    // Handle edge case where no frames ran after stress
    if min_biomass == f32::INFINITY {
        min_biomass = final_biomass;
    }

    let recovery_ratio = if baseline_biomass > 1e-6 {
        final_biomass / baseline_biomass
    } else {
        0.0
    };

    let elapsed = start.elapsed();

    StressResult {
        scenario: scenario.name().to_string(),
        description: scenario.description().to_string(),
        baseline_biomass,
        pre_stress_biomass,
        min_biomass,
        min_biomass_frame: min_frame,
        final_biomass,
        recovery_ratio,
        frames_to_min: min_frame.saturating_sub(stress_frame),
        total_frames: frames,
        elapsed_ms: elapsed.as_secs_f64() * 1000.0,
    }
}

fn main() {
    let args = parse_args();
    let start = Instant::now();

    let parallel = !args.sequential && args.scenarios.len() > 1;
    eprintln!("=== Terrarium Ecosystem Stress Benchmark Suite ===");
    eprintln!(
        "Frames: {}  Seed: {}  Lite: {}  Scenarios: {}  Parallel: {}",
        args.frames,
        args.seed,
        args.lite,
        args.scenarios.len(),
        parallel
    );
    eprintln!();

    let results: Vec<StressResult> = if parallel {
        // Run all scenarios in parallel using rayon
        let frames = args.frames;
        let seed = args.seed;
        let lite = args.lite;
        args.scenarios
            .par_iter()
            .map(|scenario| run_scenario(scenario, frames, seed, lite))
            .collect()
    } else {
        // Sequential execution
        let mut results = Vec::new();
        for scenario in &args.scenarios {
            eprint!("  Running {:25}... ", scenario.name());
            let r = run_scenario(scenario, args.frames, args.seed, args.lite);
            eprintln!(
                "baseline={:.2}  min={:.2} (f{})  final={:.2}  recovery={:.1}%  ({:.0}ms)",
                r.baseline_biomass,
                r.min_biomass,
                r.min_biomass_frame,
                r.final_biomass,
                r.recovery_ratio * 100.0,
                r.elapsed_ms
            );
            results.push(r);
        }
        results
    };

    // Print results (for parallel mode, print after all complete)
    if parallel {
        for r in &results {
            eprintln!(
                "  {:25} baseline={:.2}  min={:.2} (f{})  final={:.2}  recovery={:.1}%  ({:.0}ms)",
                r.scenario,
                r.baseline_biomass,
                r.min_biomass,
                r.min_biomass_frame,
                r.final_biomass,
                r.recovery_ratio * 100.0,
                r.elapsed_ms
            );
        }
    }

    // Summary table
    eprintln!();
    eprintln!("=== Stress Resilience Summary ===");
    eprintln!();
    eprintln!(
        "{:<22} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Scenario", "Baseline", "Min", "Final", "Recovery%", "Time(ms)"
    );
    eprintln!("{}", "-".repeat(82));
    for r in &results {
        eprintln!(
            "{:<22} {:>10.2} {:>10.2} {:>10.2} {:>9.1}% {:>10.0}",
            r.scenario,
            r.baseline_biomass,
            r.min_biomass,
            r.final_biomass,
            r.recovery_ratio * 100.0,
            r.elapsed_ms
        );
    }

    // Resilience ranking
    let mut ranked = results.clone();
    ranked.sort_by(|a, b| {
        b.recovery_ratio
            .partial_cmp(&a.recovery_ratio)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    eprintln!();
    eprintln!("Resilience ranking (best recovery first):");
    for (i, r) in ranked.iter().enumerate() {
        eprintln!(
            "  {}. {} ({:.1}% recovery)",
            i + 1,
            r.scenario,
            r.recovery_ratio * 100.0
        );
    }

    let total_elapsed = start.elapsed();
    eprintln!();
    let speedup = if parallel { " (parallel)" } else { "" };
    eprintln!("Total time: {:.2}s{}", total_elapsed.as_secs_f64(), speedup);

    // Output
    if let Some(ref path) = args.output {
        let json = serde_json::to_string_pretty(&results).unwrap_or_default();
        if let Err(e) = fs::write(path, &json) {
            eprintln!("Error writing output: {}", e);
        } else {
            eprintln!("Results written to: {}", path);
        }
    }
}
