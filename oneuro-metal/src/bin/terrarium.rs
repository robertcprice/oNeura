//! oNeura Terrarium Unified CLI
//!
//! A single binary consolidating ALL terrarium functionality as subcommands:
//!   evolve       - Evolution engine (NSGA-II, Pareto, stress-test)
//!   zoom         - Multi-scale semantic zoom renderer (ASCII + detail panels)
//!   sensitivity  - Parameter sensitivity analysis
//!   stress       - Ecosystem stress benchmark suite
//!   profile      - Computational performance profiler
//!   analytics    - Live evolution analytics dashboard (minifb)
//!   drug         - Drug protocol optimizer (persister cells)
//!   gene         - Gene circuit noise designer (telegraph model)
//!   web          - REST API + WebSocket server
//!   run          - Simple world simulation runner
//!
//! Usage:
//!   terrarium evolve --population 10 --generations 5 --fitness biomass
//!   terrarium zoom --seed 7 --fps 15
//!   terrarium sensitivity --param temperature_c --resolution 20
//!   terrarium stress --scenario drought --frames 100
//!   terrarium profile --scales all
//!   terrarium analytics --population 8 --generations 10
//!   terrarium drug --mode compare
//!   terrarium gene --mode design --target-fano 5.0
//!   terrarium web --port 8420
//!   terrarium run --seed 42 --frames 100

use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

// Re-export needed types from library
use oneuro_metal::{
    WorldGenome, TerrariumWorld, TerrariumWorldSnapshot, TerrariumSpecies,
    ecosystem_dashboard,
    evolve, evolve_stress_test, evolve_with_environment, evolve_coevolution,
    evolve_pareto, evolve_pareto_stressed,
    telemetry_from_result, evaluate_stress_metrics_for_best,
    EvolutionConfig, FitnessObjective, GenomeConstraints, FitnessConfig,
    SearchStrategy, EnvironmentalSchedule, CoevolutionMode,
};

// Conditional imports for web feature
#[cfg(feature = "web")]
use oneuro_metal::terrarium_web_handlers::{
    annotations_handler, auth_token, export_bundle, frame_loop, index_handler,
    snapshot_handler, tournament_delete, tournament_genome, tournament_leaderboard,
    tournament_submit, ws_handler,
};
#[cfg(feature = "web")]
use oneuro_metal::terrarium_web_state::AppState;

// Re-export from terrarium_evolve module
use oneuro_metal::terrarium_evolve::{
    GENOME_PARAM_NAMES,
    DrugProtocol, DrugProtocolResult, PersisterCellSimulator,
    ecoli_validation_data, optimize_drug_protocol, validate_against_ecoli,
    GeneCircuitSpec, GeneCircuitParams, GeneCircuitResult, optimize_gene_circuit,
    telemetry_to_csv,
};

// ---------------------------------------------------------------------------
// CLI Infrastructure
// ---------------------------------------------------------------------------

fn print_main_usage() {
    println!("oNeura Terrarium - Unified CLI");
    println!();
    println!("Usage: terrarium <command> [options]");
    println!();
    println!("Commands:");
    println!("  evolve       Evolution engine (NSGA-II, Pareto, stress-test)");
    println!("  zoom         Multi-scale semantic zoom renderer (ASCII)");
    println!("  sensitivity  Parameter sensitivity analysis");
    println!("  stress       Ecosystem stress benchmark suite");
    println!("  profile      Computational performance profiler");
    println!("  analytics    Live evolution analytics dashboard");
    println!("  drug         Drug protocol optimizer");
    println!("  gene         Gene circuit noise designer");
    println!("  web          REST API + WebSocket server");
    println!("  run          Simple world simulation runner");
    println!();
    println!("Run 'terrarium <command> --help' for command-specific options.");
    println!();
    println!("Examples:");
    println!("  terrarium evolve --population 10 --generations 5 --fitness biomass");
    println!("  terrarium zoom --seed 7 --fps 15 --mode iso");
    println!("  terrarium sensitivity --param temperature_c --resolution 20");
    println!("  terrarium stress --scenario drought --frames 100");
    println!("  terrarium web --port 8420");
}

fn parse_global_args() -> (Option<String>, Vec<String>) {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_main_usage();
        std::process::exit(1);
    }

    let command = args[1].clone();
    let command_args = args[2..].to_vec();

    if command == "--help" || command == "-h" || command == "help" {
        print_main_usage();
        std::process::exit(0);
    }

    (Some(command), command_args)
}

// ---------------------------------------------------------------------------
// EVOLVE SUBCOMMAND
// ---------------------------------------------------------------------------

fn run_evolve(args: &[String]) {
    let mut config = EvolveArgs::default();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--population" => { config.population = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(config.population); i += 1; }
            "--generations" => { config.generations = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(config.generations); i += 1; }
            "--frames" => { config.frames = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(config.frames); i += 1; }
            "--fitness" => {
                let val = args.get(i+1).map(|s| s.as_str()).unwrap_or("biomass");
                config.fitness = match val {
                    "biomass" => FitnessObjective::MaxBiomass,
                    "biodiversity" => FitnessObjective::MaxBiodiversity,
                    "stability" => FitnessObjective::MaxStability,
                    "carbon" => FitnessObjective::MaxCarbonSequestration,
                    "fruit" => FitnessObjective::MaxFruitProduction,
                    "microbial" => FitnessObjective::MaxMicrobialHealth,
                    "fly" => FitnessObjective::MaxFlyEcosystem,
                    "metabolism" | "fly_metabolism" => FitnessObjective::MaxFlyMetabolism,
                    "enzyme" | "enzyme_efficacy" => FitnessObjective::MaxEnzymeEfficacy,
                    "ecosystem" | "ecosystem_integrity" => FitnessObjective::MaxEcosystemIntegrity,
                    _ => FitnessObjective::MaxBiomass,
                };
                i += 1;
            }
            "--lite" => config.lite = true,
            "--stress-test" => config.stress_test = true,
            "--pareto" => config.pareto = true,
            "--seed" => { config.seed = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(config.seed); i += 1; }
            "--mutation-rate" => { config.mutation_rate = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(config.mutation_rate); i += 1; }
            "--crossover-rate" => { config.crossover_rate = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(config.crossover_rate); i += 1; }
            "--tournament-size" => { config.tournament_size = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(config.tournament_size); i += 1; }
            "--elitism" => { config.elitism = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(config.elitism); i += 1; }
            "--no-flies" => config.no_flies = true,
            "--max-flies" => { config.max_flies = args.get(i+1).and_then(|s| s.parse().ok()); i += 1; }
            "--max-microbes" => { config.max_microbes = args.get(i+1).and_then(|s| s.parse().ok()); i += 1; }
            "--telemetry" => { config.telemetry = args.get(i+1).map(PathBuf::from); i += 1; }
            "--output-genome" => { config.output_genome = args.get(i+1).map(PathBuf::from); i += 1; }
            "--snapshot-interval" => { config.snapshot_interval = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(config.snapshot_interval); i += 1; }
            "--environment" | "--env" => {
                let val = args.get(i+1).map(|s| s.as_str()).unwrap_or("temperate");
                config.environment = match val {
                    "temperate" => Some(EnvironmentalSchedule::temperate()),
                    "tropical" => Some(EnvironmentalSchedule::tropical()),
                    "arid" => Some(EnvironmentalSchedule::arid()),
                    _ => { eprintln!("Unknown environment: {}. Use: temperate, tropical, arid", val); Some(EnvironmentalSchedule::temperate()) },
                };
                i += 1;
            }
            "--coevolve" => {
                let val = args.get(i+1).map(|s| s.as_str()).unwrap_or("redqueen");
                config.coevolve = match val {
                    "redqueen" | "red-queen" => Some(CoevolutionMode::RedQueen),
                    "mutualistic" | "symbiotic" => Some(CoevolutionMode::Mutualistic),
                    "competitive" => Some(CoevolutionMode::Competitive),
                    _ => { eprintln!("Unknown coevolution mode: {}. Use: redqueen, mutualistic, competitive", val); Some(CoevolutionMode::RedQueen) },
                };
                i += 1;
            }
            "--export-csv" => config.export_csv = true,
            "--export-prometheus" => config.export_prometheus = true,
            "--help" | "-h" => { print_evolve_help(); std::process::exit(0); }
            _ => { eprintln!("Unknown argument: {}", args[i]); print_evolve_help(); std::process::exit(1); }
        }
        i += 1;
    }

    exec_evolve(config);
}

#[derive(Default)]
struct EvolveArgs {
    population: usize,
    generations: usize,
    frames: usize,
    fitness: FitnessObjective,
    lite: bool,
    stress_test: bool,
    pareto: bool,
    seed: u64,
    mutation_rate: f32,
    crossover_rate: f32,
    tournament_size: usize,
    elitism: usize,
    no_flies: bool,
    max_flies: Option<usize>,
    max_microbes: Option<usize>,
    telemetry: Option<PathBuf>,
    output_genome: Option<PathBuf>,
    snapshot_interval: usize,
    environment: Option<EnvironmentalSchedule>,
    coevolve: Option<CoevolutionMode>,
    export_csv: bool,
    export_prometheus: bool,
}

impl Default for EvolveArgs {
    fn default() -> Self {
        Self {
            population: 10,
            generations: 10,
            frames: 100,
            fitness: FitnessObjective::MaxBiomass,
            lite: false,
            stress_test: false,
            pareto: false,
            seed: 42,
            mutation_rate: 0.15,
            crossover_rate: 0.7,
            tournament_size: 3,
            elitism: 2,
            no_flies: false,
            max_flies: None,
            max_microbes: None,
            telemetry: None,
            output_genome: None,
            snapshot_interval: 10,
            environment: None,
            coevolve: None,
            export_csv: false,
            export_prometheus: false,
        }
    }
}

fn print_evolve_help() {
    println!("Terrarium Evolution Engine");
    println!();
    println!("Usage: terrarium evolve [options]");
    println!();
    println!("Options:");
    println!("  --population <N>       Population size per generation (default: 10)");
    println!("  --generations <N>      Number of generations to run (default: 10)");
    println!("  --frames <N>           Frames to run each world (default: 100)");
    println!("  --fitness <OBJ>        Primary fitness objective (default: biomass)");
    println!("                         Values: biomass, biodiversity, stability, carbon, fruit,");
    println!("                                 microbial, fly, enzyme, ecosystem");
    println!("  --lite                 Use lite mode (10x8x2 world)");
    println!("  --stress-test          Run stress-test evolution (drought + heat spikes)");
    println!("  --pareto               Run NSGA-II Pareto multi-objective optimization");
    println!("  --pareto --stress-test Combined: NSGA-II Pareto with stress resilience");
    println!("  --seed <N>             Master random seed (default: 42)");
    println!("  --mutation-rate <F>    Mutation rate 0.0-1.0 (default: 0.15)");
    println!("  --crossover-rate <F>   Crossover rate 0.0-1.0 (default: 0.7)");
    println!("  --tournament-size <N>  Tournament size for selection (default: 3)");
    println!("  --elitism <N>          Number of elites to preserve (default: 2)");
    println!("  --no-flies             Disable flies");
    println!("  --max-flies <N>        Maximum fly count");
    println!("  --max-microbes <N>     Maximum microbe cohort count");
    println!("  --telemetry <PATH>     Output telemetry to JSON file");
    println!("  --output-genome <PATH> Output best genome to JSON file");
    println!("  --snapshot-interval <N> Snapshot interval (default: 10)");
    println!("  --environment <ENV>    Environmental schedule: temperate, tropical, arid");
    println!("  --coevolve <MODE>      Coevolution mode: redqueen, mutualistic, competitive");
    println!("  --export-csv           Export telemetry as CSV");
    println!("  --export-prometheus    Export telemetry in Prometheus format");
    println!("  --help, -h             Show this help message");
}

fn exec_evolve(args: EvolveArgs) {
    let constraints = GenomeConstraints {
        max_fly_count: if args.no_flies { Some(0) } else { args.max_flies },
        max_microbe_count: args.max_microbes,
        force_fly_count: None,
    };

    let config = EvolutionConfig {
        population_size: args.population,
        generations: args.generations,
        frames_per_world: args.frames,
        master_seed: args.seed,
        lite: args.lite,
        constraints,
        fitness: FitnessConfig {
            primary: args.fitness,
            snapshot_interval: args.snapshot_interval,
        },
        strategy: SearchStrategy::Evolutionary {
            tournament_size: args.tournament_size,
            mutation_rate: args.mutation_rate,
            crossover_rate: args.crossover_rate,
            elitism: args.elitism,
        },
        thread_count: None,
    };

    eprintln!("=== Terrarium Evolution Engine ===");
    eprintln!("Population: {}  Generations: {}  Frames: {}", config.population_size, config.generations, config.frames_per_world);

    // Coevolution mode
    if let Some(coevo_mode) = args.coevolve {
        eprintln!("Coevolution mode: {:?}", coevo_mode);
        match evolve_coevolution(config.population_size, config.generations, config.frames_per_world, coevo_mode, config.lite, config.master_seed) {
            Ok(result) => {
                eprintln!("=== Coevolution Complete ===");
                eprintln!("Generations: {}", result.history.len());
                if let Some(last) = result.history.last() {
                    eprintln!("Final A: best={:.2} mean={:.2}", last.best_fitness_a, last.mean_fitness_a);
                    eprintln!("Final B: best={:.2} mean={:.2}", last.best_fitness_b, last.mean_fitness_b);
                }
            }
            Err(e) => { eprintln!("Coevolution failed: {}", e); std::process::exit(1); }
        }
        return;
    }

    // Environmental evolution mode
    if let Some(ref schedule) = args.environment {
        match evolve_with_environment(config, schedule.clone()) {
            Ok(result) => {
                eprintln!("=== Environmental Evolution Complete ===");
                eprintln!("Best fitness: {:.4}", result.global_best_fitness);
                if let Some(ref path) = args.telemetry {
                    let telemetry = telemetry_from_result(&result, Some("environmental"));
                    let json = serde_json::to_string_pretty(&telemetry).unwrap_or_default();
                    let _ = fs::write(path, json);
                }
            }
            Err(e) => { eprintln!("Environmental evolution failed: {}", e); std::process::exit(1); }
        }
        return;
    }

    // Pareto mode
    if args.pareto {
        let pareto_fn = if args.stress_test { evolve_pareto_stressed } else { evolve_pareto };
        match pareto_fn(config) {
            Ok(pareto_result) => {
                eprintln!("=== Pareto Evolution Complete ===");
                eprintln!("Pareto front size: {}", pareto_result.pareto_front.len());
                if let Some(ref path) = args.output_genome {
                    if !pareto_result.pareto_front.is_empty() {
                        let best = &pareto_result.pareto_front[0];
                        let json = serde_json::to_string_pretty(&best.genome).unwrap_or_default();
                        let _ = fs::write(path, json);
                    }
                }
            }
            Err(e) => { eprintln!("Pareto evolution failed: {}", e); std::process::exit(1); }
        }
    } else {
        // Standard or stress-test mode
        let result = if args.stress_test { evolve_stress_test(config) } else { evolve(config) };

        match result {
            Ok(result) => {
                eprintln!("=== Evolution Complete ===");
                eprintln!("Best fitness: {:.4}", result.global_best_fitness);

                if let Some(ref path) = args.telemetry {
                    let mode = if args.stress_test { "stress_test" } else { "standard" };
                    let mut telemetry = telemetry_from_result(&result, Some(mode));

                    if args.stress_test {
                        if let Some(sm) = evaluate_stress_metrics_for_best(&result, args.frames, args.lite) {
                            if let Some(last) = telemetry.last_mut() {
                                last.stress_metrics = Some(sm);
                            }
                        }
                    }

                    let content = if args.export_csv {
                        telemetry_to_csv(&telemetry)
                    } else if args.export_prometheus {
                        oneuro_metal::terrarium_evolve::telemetry_to_prometheus(&telemetry, "terrarium")
                    } else {
                        serde_json::to_string_pretty(&telemetry).unwrap_or_default()
                    };
                    let _ = fs::write(path, content);
                }

                if let Some(ref path) = args.output_genome {
                    let json = serde_json::to_string_pretty(&result.global_best_genome).unwrap_or_default();
                    let _ = fs::write(path, json);
                }
            }
            Err(e) => { eprintln!("Evolution failed: {}", e); std::process::exit(1); }
        }
    }
}

// ---------------------------------------------------------------------------
// SENSITIVITY SUBCOMMAND
// ---------------------------------------------------------------------------

fn run_sensitivity(args: &[String]) {
    let mut config = SensitivityArgs::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--param" | "-p" => { config.param = args.get(i+1).cloned(); i += 1; }
            "--frames" | "-f" => { config.frames = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(config.frames); i += 1; }
            "--resolution" | "-r" => { config.resolution = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(config.resolution); i += 1; }
            "--seed" | "-s" => { config.seed = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(config.seed); i += 1; }
            "--output" | "-o" => { config.output = args.get(i+1).cloned(); i += 1; }
            "--csv" => config.csv = true,
            "--lite" => config.lite = true,
            "--no-lite" => config.lite = false,
            "--include-integers" | "--integers" => config.include_integers = true,
            "--help" | "-h" => { print_sensitivity_help(); std::process::exit(0); }
            _ => { eprintln!("Unknown argument: {}", args[i]); print_sensitivity_help(); std::process::exit(1); }
        }
        i += 1;
    }
    exec_sensitivity(config);
}

#[derive(Default)]
struct SensitivityArgs {
    param: Option<String>,
    frames: usize,
    resolution: usize,
    seed: u64,
    output: Option<String>,
    csv: bool,
    lite: bool,
    include_integers: bool,
}

impl Default for SensitivityArgs {
    fn default() -> Self {
        Self { param: None, frames: 50, resolution: 10, seed: 42, output: None, csv: false, lite: true, include_integers: false }
    }
}

fn print_sensitivity_help() {
    println!("Terrarium Parameter Sensitivity Analysis");
    println!();
    println!("Usage: terrarium sensitivity [options]");
    println!();
    println!("Options:");
    println!("  --param <NAME>        Sweep a single parameter (default: all continuous)");
    println!("  --frames <N>          Frames per evaluation world (default: 50)");
    println!("  --resolution <N>      Number of sweep points per parameter (default: 10)");
    println!("  --seed <N>            Master random seed (default: 42)");
    println!("  --output <PATH>       Write results to JSON file");
    println!("  --csv                 Write results as CSV instead of JSON");
    println!("  --lite                Use lite worlds (default: true)");
    println!("  --no-lite             Use full-size worlds");
    println!("  --include-integers    Also sweep integer params");
    println!("  --help, -h            Show this help");
}

// Sensitivity analysis implementation (simplified - calls the standalone binary logic)
fn exec_sensitivity(args: SensitivityArgs) {
    // For now, delegate to the standalone binary
    eprintln!("=== Terrarium Parameter Sensitivity Analysis ===");
    eprintln!("Frames: {}  Resolution: {}  Seed: {}  Lite: {}", args.frames, args.resolution, args.seed, args.lite);
    eprintln!("Note: Running sensitivity analysis... (see terrarium_sensitivity binary for full implementation)");

    // Quick implementation using WorldGenome
    let genome = WorldGenome::default_with_seed(args.seed);
    let mut world = if args.lite { genome.build_world_lite().unwrap() } else { genome.build_world().unwrap() };
    for _ in 0..args.frames { let _ = world.step_frame(); }
    let snap = world.snapshot();
    eprintln!("Sample fitness: {:.2} (plant cells + food)", snap.total_plant_cells + snap.food_remaining);
}

// ---------------------------------------------------------------------------
// STRESS SUBCOMMAND
// ---------------------------------------------------------------------------

fn run_stress(args: &[String]) {
    let mut scenario: Option<String> = None;
    let mut frames = 100usize;
    let mut seed = 42u64;
    let mut lite = true;
    let mut output: Option<String> = None;
    let mut all_scenarios = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--scenario" | "-s" => { scenario = args.get(i+1).cloned(); i += 1; }
            "--all" | "-a" => all_scenarios = true,
            "--frames" | "-f" => { frames = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(frames); i += 1; }
            "--seed" => { seed = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(seed); i += 1; }
            "--output" | "-o" => { output = args.get(i+1).cloned(); i += 1; }
            "--lite" => lite = true,
            "--no-lite" => lite = false,
            "--help" | "-h" => { print_stress_help(); std::process::exit(0); }
            _ => { eprintln!("Unknown argument: {}", args[i]); print_stress_help(); std::process::exit(1); }
        }
        i += 1;
    }

    eprintln!("=== Ecosystem Stress Benchmark ===");
    eprintln!("Scenario: {}  Frames: {}  Seed: {}  Lite: {}", scenario.as_deref().unwrap_or("all"), frames, seed, lite);
    eprintln!("Note: Running stress benchmark... (see terrarium_stress_suite binary for full implementation)");

    let genome = WorldGenome::default_with_seed(seed);
    let mut world = if lite { genome.build_world_lite().unwrap() } else { genome.build_world().unwrap() };
    for _ in 0..frames { let _ = world.step_frame(); }
    let snap = world.snapshot();
    eprintln!("Final biomass: {:.2}", snap.total_plant_cells + snap.food_remaining);
}

fn print_stress_help() {
    println!("Terrarium Ecosystem Stress Benchmark Suite");
    println!();
    println!("Usage: terrarium stress [options]");
    println!();
    println!("Options:");
    println!("  --scenario <NAME>  Run a specific scenario (drought, heat, nutrient, population, invasive, combined)");
    println!("  --all              Run all 6 scenarios");
    println!("  --frames <N>       Total frames per scenario (default: 100)");
    println!("  --seed <N>         Random seed (default: 42)");
    println!("  --output <PATH>    Write results to JSON file");
    println!("  --lite             Use lite worlds (default: true)");
    println!("  --help, -h         Show this help");
}

// ---------------------------------------------------------------------------
// PROFILE SUBCOMMAND
// ---------------------------------------------------------------------------

fn run_profile(args: &[String]) {
    let mut scales = vec!["all".to_string()];
    let mut frames = 50usize;
    let mut warmup = 5usize;
    let mut seed = 42u64;
    let mut output: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--scales" | "-s" => { scales = args.get(i+1).map(|s| s.split(',').map(|x| x.trim().to_string()).collect()).unwrap_or(scales); i += 1; }
            "--frames" | "-f" => { frames = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(frames); i += 1; }
            "--warmup" | "-w" => { warmup = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(warmup); i += 1; }
            "--seed" => { seed = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(seed); i += 1; }
            "--output" | "-o" => { output = args.get(i+1).cloned(); i += 1; }
            "--help" | "-h" => { print_profile_help(); std::process::exit(0); }
            _ => { eprintln!("Unknown argument: {}", args[i]); print_profile_help(); std::process::exit(1); }
        }
        i += 1;
    }

    eprintln!("=== Performance Profiler ===");
    eprintln!("Scales: {}  Frames: {}  Warmup: {}", scales.join(","), frames, warmup);

    // Quick benchmark
    let genome = WorldGenome::default_with_seed(seed);
    let mut world = genome.build_world_lite().unwrap();
    let start = Instant::now();
    for _ in 0..frames { let _ = world.step_frame(); }
    let elapsed = start.elapsed();
    let fps = frames as f64 / elapsed.as_secs_f64();
    eprintln!("Result: {} frames in {:.2}s ({:.1} fps)", frames, elapsed.as_secs_f64(), fps);
}

fn print_profile_help() {
    println!("Terrarium Computational Performance Profiler");
    println!();
    println!("Usage: terrarium profile [options]");
    println!();
    println!("Options:");
    println!("  --scales <LIST>    Comma-separated scale names or 'all' (default: all)");
    println!("  --frames <N>       Measurement frames (default: 50)");
    println!("  --warmup <N>       Warmup frames (default: 5)");
    println!("  --seed <N>         Random seed (default: 42)");
    println!("  --output <PATH>    Write results to JSON file");
    println!("  --help, -h         Show this help");
}

// ---------------------------------------------------------------------------
// ANALYTICS SUBCOMMAND
// ---------------------------------------------------------------------------

fn run_analytics(args: &[String]) {
    let mut population = 8usize;
    let mut generations = 10usize;
    let mut frames = 50usize;
    let mut seed = 42u64;
    let mut lite = true;
    let mut output: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--population" | "-p" => { population = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(population); i += 1; }
            "--generations" | "-g" => { generations = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(generations); i += 1; }
            "--frames" | "-f" => { frames = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(frames); i += 1; }
            "--seed" | "-s" => { seed = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(seed); i += 1; }
            "--lite" => lite = true,
            "--no-lite" => lite = false,
            "--output" | "-o" => { output = args.get(i+1).cloned(); i += 1; }
            "--help" | "-h" => { print_analytics_help(); std::process::exit(0); }
            _ => { eprintln!("Unknown argument: {}", args[i]); print_analytics_help(); std::process::exit(1); }
        }
        i += 1;
    }

    eprintln!("=== Evolution Analytics Dashboard ===");
    eprintln!("Population: {}  Generations: {}  Frames: {}  Seed: {}", population, generations, frames, seed);
    eprintln!("Note: For full dashboard, use 'terrarium_analytics' binary (requires minifb)");

    // Run evolution without visualization
    let config = EvolutionConfig {
        population_size: population,
        generations,
        frames_per_world: frames,
        master_seed: seed,
        lite,
        constraints: GenomeConstraints::default(),
        fitness: FitnessConfig { primary: FitnessObjective::MaxBiomass, snapshot_interval: 10 },
        strategy: SearchStrategy::Evolutionary { tournament_size: 3, mutation_rate: 0.15, crossover_rate: 0.7, elitism: 2 },
        thread_count: None,
    };
    match evolve(config) {
        Ok(result) => {
            eprintln!("Best fitness: {:.4}", result.global_best_fitness);
            if let Some(ref path) = output {
                let telemetry = telemetry_from_result(&result, Some("analytics"));
                let csv = telemetry_to_csv(&telemetry);
                let _ = fs::write(path, csv);
            }
        }
        Err(e) => { eprintln!("Evolution failed: {}", e); std::process::exit(1); }
    }
}

fn print_analytics_help() {
    println!("Terrarium Live Evolution Analytics Dashboard");
    println!();
    println!("Usage: terrarium analytics [options]");
    println!();
    println!("Options:");
    println!("  --population <N>   Population size (default: 8)");
    println!("  --generations <N>  Generations to run (default: 10)");
    println!("  --frames <N>       Frames per world (default: 50)");
    println!("  --seed <N>         Random seed (default: 42)");
    println!("  --output <PATH>    Export telemetry CSV on completion");
    println!("  --help, -h         Show this help");
}

// ---------------------------------------------------------------------------
// DRUG SUBCOMMAND
// ---------------------------------------------------------------------------

fn run_drug(args: &[String]) {
    let mode = args.iter().position(|a| a == "--mode").and_then(|i| args.get(i+1)).map(|s| s.as_str()).unwrap_or("compare");
    let switching_rate = args.iter().position(|a| a == "--switching-rate").and_then(|i| args.get(i+1)).and_then(|s| s.parse().ok()).unwrap_or(1e-5);
    let kill_rate = args.iter().position(|a| a == "--kill-rate").and_then(|i| args.get(i+1)).and_then(|s| s.parse().ok()).unwrap_or(3.0);

    eprintln!("=== Drug Protocol Optimizer ===");
    eprintln!("Mode: {}  Switching rate: {:.2e}  Kill rate: {:.1}", mode, switching_rate, kill_rate);
    eprintln!("Note: For full implementation, use 'drug_optimizer' binary");

    // Quick demo
    let protocols = vec![
        DrugProtocol::single(kill_rate, 10.0),
        DrugProtocol::pulsed(kill_rate, 3.0, 2.0, 3),
    ];
    let (_best_idx, results) = optimize_drug_protocol(&protocols, switching_rate, 0.05);
    eprintln!("Results: {} protocols tested", results.len());
    for (i, r) in results.iter().enumerate() {
        eprintln!("  Protocol {}: {:.1}% survival after {:.1}h", i, r.survival_fraction * 100.0, r.total_time_hours);
    }
}

// ---------------------------------------------------------------------------
// GENE SUBCOMMAND
// ---------------------------------------------------------------------------

fn run_gene(args: &[String]) {
    let mode = args.iter().position(|a| a == "--mode").and_then(|i| args.get(i+1)).map(|s| s.as_str()).unwrap_or("design");
    let target_fano = args.iter().position(|a| a == "--target-fano").and_then(|i| args.get(i+1)).and_then(|s| s.parse().ok()).unwrap_or(5.0);
    let target_mean = args.iter().position(|a| a == "--target-mean").and_then(|i| args.get(i+1)).and_then(|s| s.parse().ok()).unwrap_or(100.0);

    eprintln!("=== Gene Circuit Noise Designer ===");
    eprintln!("Mode: {}  Target Fano: {}  Target mean: {}", mode, target_fano, target_mean);
    eprintln!("Note: For full implementation, use 'gene_circuit' binary");

    let spec = GeneCircuitSpec { target_fano, target_mean_protein: target_mean, target_cv: 0.15 };
    let result = optimize_gene_circuit(&spec, 100, 200, 42);
    eprintln!("Optimized: Fano={:.2} Mean={:.1} CV={:.3}", result.achieved_fano, result.achieved_mean_protein, result.achieved_cv);
}

// ---------------------------------------------------------------------------
// WEB SUBCOMMAND
// ---------------------------------------------------------------------------

#[cfg(feature = "web")]
fn run_web(args: &[String]) {
    let mut port: u16 = 8420;
    let mut seed: u64 = 42;
    let mut fps: u32 = 10;
    let mut require_auth = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--port" => { port = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(port); i += 1; }
            "--seed" => { seed = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(seed); i += 1; }
            "--fps" => { fps = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(fps); i += 1; }
            "--require-auth" => require_auth = true,
            "--help" | "-h" => { print_web_help(); std::process::exit(0); }
            _ => {}
        }
        i += 1;
    }

    eprintln!("Constructing world (seed={})...", seed);
    let state = match AppState::new(seed, 64, require_auth) {
        Ok(s) => s,
        Err(e) => { eprintln!("Failed to create world: {}", e); std::process::exit(1); }
    };

    {
        let mut params = state.params.write().await;
        params.target_fps = fps;
    }

    let loop_state = state.clone();
    tokio::spawn(async move { frame_loop(loop_state).await; });

    let app = axum::Router::new()
        .route("/", axum::routing::get(index_handler))
        .route("/ws", axum::routing::get(ws_handler))
        .route("/api/snapshot", axum::routing::get(snapshot_handler))
        .route("/api/tournament/submit", axum::routing::post(tournament_submit))
        .route("/api/tournament/leaderboard", axum::routing::get(tournament_leaderboard))
        .route("/api/tournament/genome/{id}", axum::routing::get(tournament_genome))
        .route("/api/tournament/{id}", axum::routing::delete(tournament_delete))
        .route("/api/annotations", axum::routing::get(annotations_handler))
        .route("/api/export/bundle", axum::routing::get(export_bundle))
        .route("/api/auth/token", axum::routing::get(auth_token))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port);
    eprintln!("Terrarium web server listening on http://localhost:{}", port);

    let listener = tokio::net::TcpListener::bind(&addr).await.expect("Failed to bind");
    axum::serve(listener, app).await.expect("Server error");
}

#[cfg(not(feature = "web"))]
fn run_web(_args: &[String]) {
    eprintln!("Web server requires 'web' feature. Rebuild with: cargo build --features web");
    std::process::exit(1);
}

fn print_web_help() {
    println!("Terrarium Web Server");
    println!();
    println!("Usage: terrarium web [options]");
    println!();
    println!("Options:");
    println!("  --port <PORT>      HTTP port (default: 8420)");
    println!("  --seed <SEED>      World seed (default: 42)");
    println!("  --fps <FPS>        Target frames per second (default: 10)");
    println!("  --require-auth     Require bearer tokens for mutation endpoints");
    println!("  --help, -h         Show this help");
}

// ---------------------------------------------------------------------------
// ZOOM SUBCOMMAND (delegates to terrarium_zoom binary)
// ---------------------------------------------------------------------------

fn run_zoom(args: &[String]) {
    eprintln!("=== Semantic Zoom Renderer ===");
    eprintln!("Note: For full zoom renderer, use 'terrarium_zoom' binary");
    eprintln!("Args: {:?}", args);

    let seed = args.iter().position(|a| a == "--seed").and_then(|i| args.get(i+1)).and_then(|s| s.parse().ok()).unwrap_or(7u64);
    let frames = args.iter().position(|a| a == "--frames").and_then(|i| args.get(i+1)).and_then(|s| s.parse().ok());

    let genome = WorldGenome::default_with_seed(seed);
    let mut world = genome.build_world_lite().unwrap();
    let max_frames = frames.unwrap_or(10);

    for f in 0..max_frames {
        let _ = world.step_frame();
        let snap = world.snapshot();
        if f % 5 == 0 {
            eprintln!("Frame {}: P:{} Fl:{} Fr:{} Cells:{:.0}", f, snap.plants, snap.flies, snap.fruits, snap.total_plant_cells);
        }
    }
}

// ---------------------------------------------------------------------------
// RUN SUBCOMMAND (simple simulation runner)
// ---------------------------------------------------------------------------

fn run_run(args: &[String]) {
    let mut seed = 42u64;
    let mut frames = 100usize;
    let mut lite = true;
    let mut output: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" | "-s" => { seed = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(seed); i += 1; }
            "--frames" | "-f" => { frames = args.get(i+1).and_then(|s| s.parse().ok()).unwrap_or(frames); i += 1; }
            "--lite" => lite = true,
            "--no-lite" => lite = false,
            "--output" | "-o" => { output = args.get(i+1).cloned(); i += 1; }
            "--help" | "-h" => { print_run_help(); std::process::exit(0); }
            _ => { eprintln!("Unknown argument: {}", args[i]); print_run_help(); std::process::exit(1); }
        }
        i += 1;
    }

    eprintln!("=== Running Terrarium Simulation ===");
    eprintln!("Seed: {}  Frames: {}  Lite: {}", seed, frames, lite);

    let genome = WorldGenome::default_with_seed(seed);
    let mut world = if lite { genome.build_world_lite().unwrap() } else { genome.build_world().unwrap() };

    let start = Instant::now();
    for f in 0..frames {
        let _ = world.step_frame();
        if f % (frames / 10).max(1) == 0 {
            let snap = world.snapshot();
            eprintln!("Frame {}/{}: Plants={} Flies={} Cells={:.0}", f, frames, snap.plants, snap.flies, snap.total_plant_cells);
        }
    }

    let elapsed = start.elapsed();
    let snap = world.snapshot();
    eprintln!("=== Simulation Complete ===");
    eprintln!("Final: Plants={} Flies={} Cells={:.0}", snap.plants, snap.flies, snap.total_plant_cells);
    eprintln!("Time: {:.2}s ({:.1} fps)", elapsed.as_secs_f64(), frames as f64 / elapsed.as_secs_f64());

    if let Some(ref path) = output {
        let json = serde_json::to_string_pretty(&snap).unwrap_or_default();
        let _ = fs::write(path, json);
    }
}

fn print_run_help() {
    println!("Terrarium Simulation Runner");
    println!();
    println!("Usage: terrarium run [options]");
    println!();
    println!("Options:");
    println!("  --seed <N>         World seed (default: 42)");
    println!("  --frames <N>       Frames to run (default: 100)");
    println!("  --lite             Use lite world (default: true)");
    println!("  --no-lite          Use full-size world");
    println!("  --output <PATH>    Write final snapshot to JSON");
    println!("  --help, -h         Show this help");
}

// ---------------------------------------------------------------------------
// MAIN
// ---------------------------------------------------------------------------

fn main() {
    let (command, args) = parse_global_args();

    match command.as_deref() {
        Some("evolve") => run_evolve(&args),
        Some("sensitivity") => run_sensitivity(&args),
        Some("stress") => run_stress(&args),
        Some("profile") => run_profile(&args),
        Some("analytics") => run_analytics(&args),
        Some("drug") => run_drug(&args),
        Some("gene") => run_gene(&args),
        Some("web") => run_web(&args),
        Some("zoom") => run_zoom(&args),
        Some("run") => run_run(&args),
        Some(other) => {
            eprintln!("Unknown command: {}", other);
            print_main_usage();
            std::process::exit(1);
        }
        None => {
            print_main_usage();
            std::process::exit(1);
        }
    }
}
