//! CLI for Terrarium Evolution Engine.
//!
//! Usage:
//!   terrarium_evolve --population 10 --generations 5 --frames 100 --fitness biomass --lite
//!   terrarium_evolve --stress-test --population 8 --generations 3 --frames 50
//!   terrarium_evolve --telemetry telemetry.json --output-genome best_genome.json

use oneuro_metal::{
    evolve, evolve_stress_test, telemetry_from_result, evaluate_stress_metrics_for_best,
    EvolutionConfig, FitnessObjective, GenomeConstraints, FitnessConfig,
    SearchStrategy,
    evolve_pareto_stressed,
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
                args.population = env::args().nth(i + 1).and_then(|s| s.parse().ok()).unwrap_or(args.population);
                i += 1;
            }
            "--generations" => {
                args.generations = env::args().nth(i + 1).and_then(|s| s.parse().ok()).unwrap_or(args.generations);
                i += 1;
            }
            "--frames" => {
                args.frames = env::args().nth(i + 1).and_then(|s| s.parse().ok()).unwrap_or(args.frames);
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
                    _ => FitnessObjective::MaxBiomass,
                };
                i += 1;
            }
            "--lite" => args.lite = true,
            "--stress-test" => args.stress_test = true,
            "--pareto" => args.pareto = true,
            "--seed" => {
                args.seed = env::args().nth(i + 1).and_then(|s| s.parse().ok()).unwrap_or(args.seed);
                i += 1;
            }
            "--mutation-rate" => {
                args.mutation_rate = env::args().nth(i + 1).and_then(|s| s.parse().ok()).unwrap_or(args.mutation_rate);
                i += 1;
            }
            "--crossover-rate" => {
                args.crossover_rate = env::args().nth(i + 1).and_then(|s| s.parse().ok()).unwrap_or(args.crossover_rate);
                i += 1;
            }
            "--tournament-size" => {
                args.tournament_size = env::args().nth(i + 1).and_then(|s| s.parse().ok()).unwrap_or(args.tournament_size);
                i += 1;
            }
            "--elitism" => {
                args.elitism = env::args().nth(i + 1).and_then(|s| s.parse().ok()).unwrap_or(args.elitism);
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
                args.snapshot_interval = env::args().nth(i + 1).and_then(|s| s.parse().ok()).unwrap_or(args.snapshot_interval);
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
    println!("                         Values: biomass, biodiversity, stability, carbon, fruit, microbial, fly");
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
    println!("  --help, -h             Show this help message");
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
        }
    }
}

fn main() {
    let args = parse_args();

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
    eprintln!("Population: {}", config.population_size);
    eprintln!("Generations: {}", config.generations);
    eprintln!("Frames/world: {}", config.frames_per_world);
    let mode_str = if args.pareto && args.stress_test {
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
    eprintln!("");

    if args.pareto {
        // Pareto mode (with or without stress) uses ParetoEvolutionResult
        let pareto_fn = if args.stress_test {
            evolve_pareto_stressed
        } else {
            oneuro_metal::evolve_pareto
        };
        match pareto_fn(config) {
            Ok(pareto_result) => {
                eprintln!("");
                eprintln!("=== Pareto Evolution Complete ===");
                eprintln!("Pareto front size: {}", pareto_result.pareto_front.len());
                eprintln!("Total worlds evaluated: {}", pareto_result.total_worlds_evaluated);
                eprintln!("Total time: {:.2}s", pareto_result.total_wall_time_ms / 1000.0);

                // Output telemetry
                if let Some(ref path) = args.telemetry {
                    let telemetry = oneuro_metal::telemetry_from_pareto_result(&pareto_result);
                    let json = serde_json::to_string_pretty(&vec![telemetry]).unwrap_or_default();
                    if let Err(e) = fs::write(path, json) {
                        eprintln!("Error writing telemetry: {}", e);
                    } else {
                        eprintln!("Telemetry written to: {}", path.display());
                    }
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
                    let mode = if args.stress_test { "stress_test" } else { "standard" };
                    let mut telemetry = telemetry_from_result(&result, Some(mode));

                    // For stress-test mode, attach stress metrics to the last generation
                    if args.stress_test {
                        if let Some(sm) = evaluate_stress_metrics_for_best(
                            &result,
                            args.frames,
                            args.lite,
                        ) {
                            if let Some(last) = telemetry.last_mut() {
                                last.stress_metrics = Some(sm);
                            }
                        }
                    }

                    let json = serde_json::to_string_pretty(&telemetry).unwrap_or_default();
                    if let Err(e) = fs::write(path, json) {
                        eprintln!("Error writing telemetry: {}", e);
                    } else {
                        eprintln!("Telemetry written to: {}", path.display());
                    }
                }

                // Output best genome
                if let Some(ref path) = args.output_genome {
                    let json = serde_json::to_string_pretty(&result.global_best_genome).unwrap_or_default();
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
