//! 2D Fitness Landscape Scanner.
//!
//! Scans a 2D grid of WorldGenome parameter pairs, evaluating ecosystem fitness
//! at each point to produce a fitness landscape heatmap.
//!
//! Usage:
//!   terrarium_landscape --x temperature_c --y moisture_scale --resolution 15
//!   terrarium_landscape --x respiration_vmax --y photosynthesis_vmax --resolution 20 --output landscape.csv
//!   terrarium_landscape --list-params

use oneura_core::terrarium::{
    scan_fitness_landscape, FitnessObjective, LandscapeAxis, WorldGenome, GENOME_PARAM_NAMES,
};
use std::env;
use std::fs;
use std::time::Instant;

struct Args {
    x_param: String,
    y_param: String,
    resolution: usize,
    frames: usize,
    seed: u64,
    fitness: FitnessObjective,
    output: Option<String>,
    list_params: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            x_param: "temperature_c".into(),
            y_param: "moisture_scale".into(),
            resolution: 10,
            frames: 50,
            seed: 42,
            fitness: FitnessObjective::MaxBiomass,
            output: None,
            list_params: false,
        }
    }
}

fn parse_args() -> Args {
    let mut args = Args::default();
    let mut i = 1;
    let argv: Vec<String> = env::args().collect();
    while i < argv.len() {
        match argv[i].as_str() {
            "--x" | "--x-param" | "-x" => {
                args.x_param = argv.get(i + 1).cloned().unwrap_or(args.x_param);
                i += 1;
            }
            "--y" | "--y-param" | "-y" => {
                args.y_param = argv.get(i + 1).cloned().unwrap_or(args.y_param);
                i += 1;
            }
            "--resolution" | "-r" => {
                args.resolution = argv
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(args.resolution);
                i += 1;
            }
            "--frames" | "-f" => {
                args.frames = argv
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(args.frames);
                i += 1;
            }
            "--seed" | "-s" => {
                args.seed = argv
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(args.seed);
                i += 1;
            }
            "--fitness" => {
                let val = argv.get(i + 1).map(|s| s.as_str()).unwrap_or("biomass");
                args.fitness = match val {
                    "biomass" => FitnessObjective::MaxBiomass,
                    "biodiversity" => FitnessObjective::MaxBiodiversity,
                    "stability" => FitnessObjective::MaxStability,
                    "carbon" => FitnessObjective::MaxCarbonSequestration,
                    "fruit" => FitnessObjective::MaxFruitProduction,
                    "microbial" => FitnessObjective::MaxMicrobialHealth,
                    "fly" => FitnessObjective::MaxFlyEcosystem,
                    "enzyme" => FitnessObjective::MaxEnzymeEfficacy,
                    "ecosystem" => FitnessObjective::MaxEcosystemIntegrity,
                    _ => FitnessObjective::MaxBiomass,
                };
                i += 1;
            }
            "--output" | "-o" => {
                args.output = argv.get(i + 1).cloned();
                i += 1;
            }
            "--list-params" | "--list" | "-l" => args.list_params = true,
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
    args
}

fn print_help() {
    println!("Terrarium 2D Fitness Landscape Scanner");
    println!();
    println!("Scans a 2D grid of genome parameter values to map the fitness landscape.");
    println!("Outputs an ASCII heatmap to stderr and optional CSV data to a file.");
    println!();
    println!("Usage: terrarium_landscape [options]");
    println!();
    println!("Options:");
    println!("  --x <PARAM>         X-axis parameter name (default: temperature_c)");
    println!("  --y <PARAM>         Y-axis parameter name (default: moisture_scale)");
    println!("  --resolution <N>    Grid points per axis (default: 10)");
    println!("  --frames <N>        Simulation frames per evaluation (default: 50)");
    println!("  --seed <N>          Random seed for base genome (default: 42)");
    println!("  --fitness <OBJ>     Fitness objective (default: biomass)");
    println!("                      Values: biomass, biodiversity, stability, carbon,");
    println!("                              fruit, microbial, fly, enzyme, ecosystem");
    println!("  --output <PATH>     Write landscape CSV to file");
    println!("  --list-params       List all scannable parameter names and exit");
    println!("  --help, -h          Show this help");
    println!();
    println!("Examples:");
    println!("  terrarium_landscape --x temperature_c --y moisture_scale -r 15");
    println!(
        "  terrarium_landscape --x respiration_vmax --y photosynthesis_vmax -r 10 -o landscape.csv"
    );
    println!("  terrarium_landscape --fitness enzyme --x enzyme_probe_x --y enzyme_probe_y");
}

fn main() {
    let args = parse_args();

    if args.list_params {
        println!("Scannable genome parameters:");
        for (i, name) in GENOME_PARAM_NAMES.iter().enumerate() {
            println!("  {:2}: {}", i, name);
        }
        return;
    }

    // Resolve parameter indices
    let x_idx = match GENOME_PARAM_NAMES.iter().position(|&n| n == args.x_param) {
        Some(i) => i,
        None => {
            eprintln!(
                "Unknown x parameter '{}'. Use --list-params to see available names.",
                args.x_param
            );
            std::process::exit(1);
        }
    };
    let y_idx = match GENOME_PARAM_NAMES.iter().position(|&n| n == args.y_param) {
        Some(i) => i,
        None => {
            eprintln!(
                "Unknown y parameter '{}'. Use --list-params to see available names.",
                args.y_param
            );
            std::process::exit(1);
        }
    };

    if x_idx == y_idx {
        eprintln!("X and Y parameters must be different.");
        std::process::exit(1);
    }

    let x_axis = LandscapeAxis::new(x_idx);
    let y_axis = LandscapeAxis::new(y_idx);
    let base = WorldGenome::default_with_seed(args.seed);

    eprintln!("=== Terrarium Fitness Landscape Scanner ===");
    eprintln!(
        "X: {}  Y: {}  Resolution: {}x{}  Frames: {}  Seed: {}  Fitness: {:?}",
        args.x_param,
        args.y_param,
        args.resolution,
        args.resolution,
        args.frames,
        args.seed,
        args.fitness
    );
    eprintln!("Total evaluations: {}", args.resolution * args.resolution);
    eprintln!();

    let start = Instant::now();

    let landscape = scan_fitness_landscape(
        &base,
        x_axis,
        y_axis,
        args.fitness,
        args.resolution,
        args.frames,
    );

    let elapsed = start.elapsed();

    // Print ASCII heatmap to stdout
    println!("{}", landscape.to_ascii(args.resolution.min(60)));

    // Print summary to stderr
    eprintln!(
        "Peak: {}={:.4}, {}={:.4} -> fitness={:.6}",
        landscape.x_param, landscape.peak.0, landscape.y_param, landscape.peak.1, landscape.peak.2
    );
    eprintln!(
        "Scan time: {:.2}s ({:.1} evals/sec)",
        elapsed.as_secs_f64(),
        (args.resolution * args.resolution) as f64 / elapsed.as_secs_f64()
    );

    // Write CSV if requested
    if let Some(ref path) = args.output {
        let csv = landscape.to_csv();
        if let Err(e) = fs::write(path, &csv) {
            eprintln!("Error writing CSV: {}", e);
        } else {
            eprintln!("Landscape CSV written to: {}", path);
        }
    }
}
