//! CLI for Terrarium Evolution Engine.
//!
//! Usage:
//!   terrarium_evolve --population 10 --generations 5 --frames 100 --fitness biomass --lite
//!   terrarium_evolve --stress-test --population 8 --generations 3 --frames 50
//!   terrarium_evolve --telemetry telemetry.json --output-genome best_genome.json
//!   terrarium_evolve --telemetry out.csv --export-csv
//!   terrarium_evolve --telemetry out.prom --export-prometheus
//!   terrarium_evolve --scan --x-param temperature_c --y-param moisture_scale --resolution 15 --output landscape.csv

use oneura_core::terrarium::{
    evaluate_stress_metrics_for_best, evolve, evolve_coevolution, evolve_pareto_stressed,
    evolve_stress_test, evolve_with_environment, scan_fitness_landscape, telemetry_from_result,
    telemetry_to_csv, telemetry_to_prometheus, CoevolutionMode, EnvironmentalSchedule,
    EvolutionConfig, FitnessConfig, FitnessObjective, GenomeConstraints, LandscapeAxis,
    SearchStrategy, WorldGenome, GENOME_PARAM_NAMES,
};
use std::env;
use std::fs;
use std::path::PathBuf;

fn parse_args() -> Args {
    let mut args = Args::default();
    let mut i = 1;

    while i < env::args().len() {
        let arg = env::args().nth(i).unwrap();
        match arg.as_str() {
            "--population" => {
                args.population = env::args()
                    .nth(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(args.population);
                i += 1;
            }
            "--generations" => {
                args.generations = env::args()
                    .nth(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(args.generations);
                i += 1;
            }
            "--frames" => {
                args.frames = env::args()
                    .nth(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(args.frames);
                i += 1;
            }
            "--fitness" => {
                let val = env::args().nth(i + 1).unwrap_or_default();
                args.fitness = match val.as_str() {
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
            "--lite" => args.lite = true,
            "--stress-test" => args.stress_test = true,
            "--pareto" => args.pareto = true,
            "--seed" => {
                args.seed = env::args()
                    .nth(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(args.seed);
                i += 1;
            }
            "--mutation-rate" => {
                args.mutation_rate = env::args()
                    .nth(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(args.mutation_rate);
                i += 1;
            }
            "--crossover-rate" => {
                args.crossover_rate = env::args()
                    .nth(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(args.crossover_rate);
                i += 1;
            }
            "--tournament-size" => {
                args.tournament_size = env::args()
                    .nth(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(args.tournament_size);
                i += 1;
            }
            "--elitism" => {
                args.elitism = env::args()
                    .nth(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(args.elitism);
                i += 1;
            }
            "--no-flies" => args.no_flies = true,
            "--max-flies" => {
                args.max_flies = env::args().nth(i + 1).and_then(|s| s.parse().ok());
                i += 1;
            }
            "--max-microbes" => {
                args.max_microbes = env::args().nth(i + 1).and_then(|s| s.parse().ok());
                i += 1;
            }
            "--telemetry" => {
                args.telemetry = env::args().nth(i + 1).map(PathBuf::from);
                i += 1;
            }
            "--output-genome" => {
                args.output_genome = env::args().nth(i + 1).map(PathBuf::from);
                i += 1;
            }
            "--snapshot-interval" => {
                args.snapshot_interval = env::args()
                    .nth(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(args.snapshot_interval);
                i += 1;
            }
            "--environment" | "--env" => {
                let val = env::args().nth(i + 1).unwrap_or_default();
                args.environment = match val.as_str() {
                    "temperate" => Some(EnvironmentalSchedule::temperate()),
                    "tropical" => Some(EnvironmentalSchedule::tropical()),
                    "arid" => Some(EnvironmentalSchedule::arid()),
                    _ => {
                        eprintln!(
                            "Unknown environment: {}. Use: temperate, tropical, arid",
                            val
                        );
                        Some(EnvironmentalSchedule::temperate())
                    }
                };
                i += 1;
            }
            "--coevolve" => {
                let val = env::args().nth(i + 1).unwrap_or_default();
                args.coevolve = match val.as_str() {
                    "redqueen" | "red-queen" => Some(CoevolutionMode::RedQueen),
                    "mutualistic" | "symbiotic" => Some(CoevolutionMode::Mutualistic),
                    "competitive" => Some(CoevolutionMode::Competitive),
                    _ => {
                        eprintln!(
                            "Unknown coevolution mode: {}. Use: redqueen, mutualistic, competitive",
                            val
                        );
                        Some(CoevolutionMode::RedQueen)
                    }
                };
                i += 1;
            }
            "--scan" => args.scan = true,
            "--x-param" => {
                args.x_param = env::args()
                    .nth(i + 1)
                    .unwrap_or_else(|| args.x_param.clone());
                i += 1;
            }
            "--y-param" => {
                args.y_param = env::args()
                    .nth(i + 1)
                    .unwrap_or_else(|| args.y_param.clone());
                i += 1;
            }
            "--resolution" => {
                args.resolution = env::args()
                    .nth(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(args.resolution);
                i += 1;
            }
            "--output" => {
                args.output = env::args().nth(i + 1).map(PathBuf::from);
                i += 1;
            }
            "--export-csv" => args.export_csv = true,
            "--export-prometheus" => args.export_prometheus = true,
            "--prometheus-job" => {
                args.prometheus_job = env::args().nth(i + 1).unwrap_or_else(|| "terrarium".into());
                i += 1;
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown argument: {}", arg);
                print_help();
                std::process::exit(1);
            }
        }
        i += 1;
    }
    args
}

fn print_help() {
    println!("Terrarium Evolution Engine");
    println!();
    println!("Usage: terrarium_evolve [options]");
    println!();
    println!("Options:");
    println!("  --population <N>       Population size per generation (default: 10)");
    println!("  --generations <N>      Number of generations to run (default: 10)");
    println!("  --frames <N>           Frames to run each world (default: 100)");
    println!("  --fitness <OBJ>        Primary fitness objective (default: biomass)");
    println!("                         Values: biomass, biodiversity, stability, carbon, fruit, microbial, fly, enzyme, ecosystem");
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
    println!("  --telemetry <PATH>     Output telemetry to file");
    println!("  --output-genome <PATH> Output best genome to JSON file");
    println!("  --snapshot-interval <N> Snapshot interval (default: 10)");
    println!("  --environment <ENV>    Environmental schedule: temperate, tropical, arid");
    println!("  --coevolve <MODE>      Coevolution mode: redqueen, mutualistic, competitive");
    println!("  --scan                 Scan 2D fitness landscape (use with --x-param, --y-param)");
    println!("  --x-param <NAME>       X axis parameter name (default: temperature_c)");
    println!("  --y-param <NAME>       Y axis parameter name (default: moisture_scale)");
    println!("  --resolution <N>       Grid resolution for landscape scan (default: 10)");
    println!("  --output <PATH>        Output CSV file for landscape scan");
    println!("  --export-csv           Export telemetry as CSV (use with --telemetry)");
    println!("  --export-prometheus    Export telemetry as Prometheus text format (use with --telemetry)");
    println!("  --prometheus-job <STR> Prometheus job label (default: terrarium)");
    println!("  --help, -h             Show this help message");
    println!();
    println!("Scan mode parameters:");
    for (i, name) in GENOME_PARAM_NAMES.iter().enumerate() {
        println!("  {:2}: {}", i, name);
    }
}

struct Args {
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
    scan: bool,
    x_param: String,
    y_param: String,
    resolution: usize,
    output: Option<PathBuf>,
    export_csv: bool,
    export_prometheus: bool,
    prometheus_job: String,
}

impl Default for Args {
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
            scan: false,
            x_param: "temperature_c".into(),
            y_param: "moisture_scale".into(),
            resolution: 10,
            output: None,
            export_csv: false,
            export_prometheus: false,
            prometheus_job: "terrarium".into(),
        }
    }
}

/// Write telemetry in the appropriate format (JSON, CSV, or Prometheus).
fn write_telemetry(
    path: &std::path::Path,
    telemetry: &[oneura_core::terrarium::GenerationTelemetry],
    export_csv: bool,
    export_prometheus: bool,
    prometheus_job: &str,
) {
    let content = if export_csv {
        telemetry_to_csv(telemetry)
    } else if export_prometheus {
        telemetry_to_prometheus(telemetry, prometheus_job)
    } else {
        serde_json::to_string_pretty(&telemetry).unwrap_or_default()
    };

    let format_name = if export_csv {
        "CSV"
    } else if export_prometheus {
        "Prometheus"
    } else {
        "JSON"
    };
    if let Err(e) = fs::write(path, &content) {
        eprintln!("Error writing telemetry: {}", e);
    } else {
        eprintln!("Telemetry ({}) written to: {}", format_name, path.display());
    }
}

fn main() {
    let args = parse_args();

    let constraints = GenomeConstraints {
        max_fly_count: if args.no_flies {
            Some(0)
        } else {
            args.max_flies
        },
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
    eprintln!("Population: {}", config.population_size);
    eprintln!("Generations: {}", config.generations);
    eprintln!("Frames/world: {}", config.frames_per_world);
    let mode_str = if args.scan {
        "scan"
    } else if args.coevolve.is_some() {
        "coevolution"
    } else if args.environment.is_some() {
        "environmental"
    } else if args.pareto && args.stress_test {
        "pareto-stressed"
    } else if args.pareto {
        "pareto"
    } else if args.stress_test {
        "stress-test"
    } else {
        "standard"
    };
    eprintln!("Mode: {}", mode_str);
    eprintln!("Lite: {}", config.lite);
    if args.export_csv {
        eprintln!("Export: CSV");
    }
    if args.export_prometheus {
        eprintln!("Export: Prometheus (job={})", args.prometheus_job);
    }
    eprintln!("");

    // Scan mode — 2D fitness landscape
    if args.scan {
        let x_idx = GENOME_PARAM_NAMES.iter().position(|&n| n == args.x_param);
        let y_idx = GENOME_PARAM_NAMES.iter().position(|&n| n == args.y_param);

        let x_idx = match x_idx {
            Some(i) => i,
            None => {
                eprintln!("Unknown x-param '{}'. Available:", args.x_param);
                for (i, name) in GENOME_PARAM_NAMES.iter().enumerate() {
                    eprintln!("  {:2}: {}", i, name);
                }
                std::process::exit(1);
            }
        };
        let y_idx = match y_idx {
            Some(i) => i,
            None => {
                eprintln!("Unknown y-param '{}'. Available:", args.y_param);
                for (i, name) in GENOME_PARAM_NAMES.iter().enumerate() {
                    eprintln!("  {:2}: {}", i, name);
                }
                std::process::exit(1);
            }
        };

        let x_axis = LandscapeAxis::new(x_idx);
        let y_axis = LandscapeAxis::new(y_idx);
        let base = WorldGenome::default_with_seed(args.seed);

        eprintln!(
            "Scanning {} vs {} ({}x{} grid, {} frames)...",
            args.x_param, args.y_param, args.resolution, args.resolution, args.frames
        );

        let landscape = scan_fitness_landscape(
            &base,
            x_axis,
            y_axis,
            args.fitness,
            args.resolution,
            args.frames,
        );

        // Print ASCII heatmap
        println!("{}", landscape.to_ascii(args.resolution.min(60)));

        // Print peak
        eprintln!(
            "Peak: {}={:.4}, {}={:.4} -> fitness={:.6}",
            landscape.x_param,
            landscape.peak.0,
            landscape.y_param,
            landscape.peak.1,
            landscape.peak.2
        );

        // Write CSV if requested
        if let Some(ref path) = args.output {
            let csv = landscape.to_csv();
            if let Err(e) = fs::write(path, &csv) {
                eprintln!("Error writing CSV: {}", e);
            } else {
                eprintln!("Landscape CSV written to: {}", path.display());
            }
        }

        return;
    }

    // Coevolution mode
    if let Some(coevo_mode) = args.coevolve {
        eprintln!("Coevolution mode: {:?}", coevo_mode);
        match evolve_coevolution(
            config.population_size,
            config.generations,
            config.frames_per_world,
            coevo_mode,
            config.lite,
            config.master_seed,
        ) {
            Ok(result) => {
                eprintln!("");
                eprintln!("=== Coevolution Complete ===");
                eprintln!("Generations: {}", result.history.len());
                if let Some(last) = result.history.last() {
                    eprintln!(
                        "Final A: best={:.2} mean={:.2}",
                        last.best_fitness_a, last.mean_fitness_a
                    );
                    eprintln!(
                        "Final B: best={:.2} mean={:.2}",
                        last.best_fitness_b, last.mean_fitness_b
                    );
                }
                eprintln!("Total time: {:.2}s", result.total_wall_time_ms / 1000.0);
            }
            Err(e) => {
                eprintln!("Coevolution failed: {}", e);
                std::process::exit(1);
            }
        }
        return;
    }

    // Environmental evolution mode
    if let Some(ref schedule) = args.environment {
        match evolve_with_environment(config, schedule.clone()) {
            Ok(result) => {
                eprintln!("");
                eprintln!("=== Environmental Evolution Complete ===");
                eprintln!("Total worlds: {}", result.total_worlds_evaluated);
                eprintln!("Best fitness: {:.4}", result.global_best_fitness);
                eprintln!("Total time: {:.2}s", result.total_wall_time_ms / 1000.0);

                if let Some(ref path) = args.telemetry {
                    let telemetry = telemetry_from_result(&result, Some("environmental"));
                    write_telemetry(
                        path,
                        &telemetry,
                        args.export_csv,
                        args.export_prometheus,
                        &args.prometheus_job,
                    );
                }
                if let Some(ref path) = args.output_genome {
                    let json = serde_json::to_string_pretty(&result.global_best_genome)
                        .unwrap_or_default();
                    if let Err(e) = fs::write(path, json) {
                        eprintln!("Error writing genome: {}", e);
                    }
                }
            }
            Err(e) => {
                eprintln!("Environmental evolution failed: {}", e);
                std::process::exit(1);
            }
        }
        return;
    }

    if args.pareto {
        // Pareto mode (with or without stress) uses ParetoEvolutionResult
        let pareto_fn = if args.stress_test {
            evolve_pareto_stressed
        } else {
            oneura_core::terrarium::evolve_pareto
        };
        match pareto_fn(config) {
            Ok(pareto_result) => {
                eprintln!("");
                eprintln!("=== Pareto Evolution Complete ===");
                eprintln!("Pareto front size: {}", pareto_result.pareto_front.len());
                eprintln!(
                    "Total worlds evaluated: {}",
                    pareto_result.total_worlds_evaluated
                );
                eprintln!(
                    "Total time: {:.2}s",
                    pareto_result.total_wall_time_ms / 1000.0
                );

                // Output telemetry
                if let Some(ref path) = args.telemetry {
                    let telemetry =
                        oneura_core::terrarium::telemetry_from_pareto_result(&pareto_result);
                    write_telemetry(
                        path,
                        &[telemetry],
                        args.export_csv,
                        args.export_prometheus,
                        &args.prometheus_job,
                    );
                }

                // Output best genomes from Pareto front
                if let Some(ref path) = args.output_genome {
                    if !pareto_result.pareto_front.is_empty() {
                        let best = &pareto_result.pareto_front[0];
                        let json = serde_json::to_string_pretty(&best.genome).unwrap_or_default();
                        if let Err(e) = fs::write(path, json) {
                            eprintln!("Error writing genome: {}", e);
                        } else {
                            eprintln!("Best genome written to: {}", path.display());
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("Pareto evolution failed: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        // Standard or stress-test mode
        let result = if args.stress_test {
            evolve_stress_test(config)
        } else {
            evolve(config)
        };

        match result {
            Ok(result) => {
                eprintln!("");
                eprintln!("=== Evolution Complete ===");
                eprintln!("Total worlds evaluated: {}", result.total_worlds_evaluated);
                eprintln!("Total time: {:.2}s", result.total_wall_time_ms / 1000.0);
                eprintln!("Best fitness: {:.4}", result.global_best_fitness);
                eprintln!("Best genome: {:?}", result.global_best_genome);

                // Output telemetry
                if let Some(ref path) = args.telemetry {
                    let mode = if args.stress_test {
                        "stress_test"
                    } else {
                        "standard"
                    };
                    let mut telemetry = telemetry_from_result(&result, Some(mode));

                    // For stress-test mode, attach stress metrics to the last generation
                    if args.stress_test {
                        if let Some(sm) =
                            evaluate_stress_metrics_for_best(&result, args.frames, args.lite)
                        {
                            if let Some(last) = telemetry.last_mut() {
                                last.stress_metrics = Some(sm);
                            }
                        }
                    }

                    write_telemetry(
                        path,
                        &telemetry,
                        args.export_csv,
                        args.export_prometheus,
                        &args.prometheus_job,
                    );
                }

                // Output best genome
                if let Some(ref path) = args.output_genome {
                    let json = serde_json::to_string_pretty(&result.global_best_genome)
                        .unwrap_or_default();
                    if let Err(e) = fs::write(path, json) {
                        eprintln!("Error writing genome: {}", e);
                    } else {
                        eprintln!("Best genome written to: {}", path.display());
                    }
                }
            }
            Err(e) => {
                eprintln!("Evolution failed: {}", e);
                std::process::exit(1);
            }
        }
    }
}
