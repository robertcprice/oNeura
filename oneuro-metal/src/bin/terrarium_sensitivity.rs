//! Parameter Sensitivity Analysis for Terrarium Evolution.
//!
//! Sweeps each WorldGenome parameter one-at-a-time (OAT) across a range of values,
//! measuring fitness response to quantify which parameters matter most.
//!
//! Usage:
//!   terrarium_sensitivity --frames 50 --resolution 10
//!   terrarium_sensitivity --param temperature_c --resolution 20 --output sensitivity.json
//!   terrarium_sensitivity --include-integers --resolution 5

use oneuro_metal::WorldGenome;
use oneuro_metal::terrarium_evolve::GENOME_PARAM_NAMES;
use std::env;
use std::fs;
use std::time::Instant;

struct Args {
    param: Option<String>,
    frames: usize,
    resolution: usize,
    seed: u64,
    output: Option<String>,
    csv: bool,
    lite: bool,
    include_integers: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            param: None,
            frames: 50,
            resolution: 10,
            seed: 42,
            output: None,
            csv: false,
            lite: true,
            include_integers: false,
        }
    }
}

fn parse_args() -> Args {
    let mut args = Args::default();
    let mut i = 1;
    let argv: Vec<String> = env::args().collect();
    while i < argv.len() {
        match argv[i].as_str() {
            "--param" | "-p" => { args.param = argv.get(i + 1).cloned(); i += 1; }
            "--frames" | "-f" => { args.frames = argv.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(args.frames); i += 1; }
            "--resolution" | "-r" => { args.resolution = argv.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(args.resolution); i += 1; }
            "--seed" | "-s" => { args.seed = argv.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(args.seed); i += 1; }
            "--output" | "-o" => { args.output = argv.get(i + 1).cloned(); i += 1; }
            "--csv" => args.csv = true,
            "--lite" => args.lite = true,
            "--no-lite" => args.lite = false,
            "--include-integers" | "--integers" => args.include_integers = true,
            "--help" | "-h" => { print_help(); std::process::exit(0); }
            other => { eprintln!("Unknown argument: {}", other); print_help(); std::process::exit(1); }
        }
        i += 1;
    }
    args
}

fn print_help() {
    println!("Terrarium Parameter Sensitivity Analysis");
    println!();
    println!("Usage: terrarium_sensitivity [options]");
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
    println!("  --include-integers    Also sweep integer params (plant_count, fly_count, etc.)");
    println!("  --help, -h            Show this help");
    println!();
    println!("Continuous: {}", CONTINUOUS_PARAMS.iter().map(|(n,_)| *n).collect::<Vec<_>>().join(", "));
    println!("Integer:    {}", INTEGER_PARAMS.iter().map(|(n,_,_)| *n).collect::<Vec<_>>().join(", "));
    println!("All genome: {}", GENOME_PARAM_NAMES.join(", "));
}

/// Continuous sweepable parameters (name, genome_index).
const CONTINUOUS_PARAMS: &[(&str, usize)] = &[
    ("proton_scale",          0),
    ("temperature_c",         1),
    ("water_volume",          3),
    ("moisture_scale",        4),
    ("respiration_vmax",     11),
    ("nitrification_vmax",   12),
    ("photosynthesis_vmax",  13),
    ("mineralization_vmax",  14),
    ("time_warp",            15),
    ("psc_scale",             9),
];

/// Integer sweepable parameters (name, genome_index, (lo, hi)).
const INTEGER_PARAMS: &[(&str, usize, (usize, usize))] = &[
    ("plant_count",   5,  (2, 16)),
    ("fruit_count",   6,  (0, 8)),
    ("fly_count",     7,  (0, 6)),
    ("water_count",   2,  (1, 6)),
    ("microbe_count", 8,  (0, 6)),
];

/// Ranges for each continuous sweepable parameter (lo, hi).
fn param_range(name: &str) -> (f32, f32) {
    match name {
        "proton_scale"        => (0.3, 3.0),
        "temperature_c"       => (10.0, 40.0),
        "water_volume"        => (50.0, 300.0),
        "moisture_scale"      => (0.5, 2.0),
        "respiration_vmax"    => (0.3, 3.0),
        "nitrification_vmax"  => (0.3, 3.0),
        "photosynthesis_vmax" => (0.3, 3.0),
        "mineralization_vmax" => (0.3, 3.0),
        "time_warp"           => (100.0, 2000.0),
        "psc_scale"           => (5.0, 100.0),
        _ => (0.0, 1.0),
    }
}

fn set_genome_param(genome: &mut WorldGenome, name: &str, val: f32) {
    match name {
        "proton_scale"        => genome.initial_proton_scale = val,
        "temperature_c"       => genome.soil_temperature_c = val,
        "water_volume"        => genome.water_volume = val,
        "moisture_scale"      => genome.initial_moisture_scale = val,
        "respiration_vmax"    => genome.respiration_vmax_scale = val,
        "nitrification_vmax"  => genome.nitrification_vmax_scale = val,
        "photosynthesis_vmax" => genome.photosynthesis_vmax_scale = val,
        "mineralization_vmax" => genome.mineralization_vmax_scale = val,
        "time_warp"           => genome.time_warp = val,
        "psc_scale"           => genome.fly_psc_scale = val,
        _ => {}
    }
}

fn set_genome_integer_param(genome: &mut WorldGenome, name: &str, val: usize) {
    match name {
        "plant_count"   => genome.plant_count = val,
        "fruit_count"   => genome.fruit_count = val,
        "fly_count"     => genome.fly_count = val,
        "water_count"   => genome.water_source_count = val,
        "microbe_count" => genome.microbe_cohort_count = val,
        _ => {}
    }
}

#[derive(Debug, Clone, serde::Serialize)]
struct SensitivityResult {
    parameter: String,
    sensitivity_index: f32,
    min_fitness: f32,
    max_fitness: f32,
    mean_fitness: f32,
    sweep_values: Vec<f32>,
    fitness_values: Vec<f32>,
    is_integer: bool,
}

fn evaluate_fitness(genome: &WorldGenome, frames: usize, lite: bool) -> f32 {
    let world_result = if lite {
        genome.build_world_lite()
    } else {
        genome.build_world()
    };
    match world_result {
        Ok(mut world) => {
            for _ in 0..frames {
                let _ = world.step_frame();
            }
            let snap = world.snapshot();
            snap.total_plant_cells + snap.food_remaining
        }
        Err(_) => 0.0,
    }
}

fn sweep_param(name: &str, resolution: usize, frames: usize, seed: u64, lite: bool) -> SensitivityResult {
    let (lo, hi) = param_range(name);
    let mut sweep_values = Vec::with_capacity(resolution);
    let mut fitness_values = Vec::with_capacity(resolution);

    for i in 0..resolution {
        let t = if resolution > 1 { i as f32 / (resolution - 1) as f32 } else { 0.5 };
        let val = lo + t * (hi - lo);
        sweep_values.push(val);

        let mut genome = WorldGenome::default_with_seed(seed);
        set_genome_param(&mut genome, name, val);
        let fitness = evaluate_fitness(&genome, frames, lite);
        fitness_values.push(fitness);
    }

    compute_sensitivity(name, sweep_values, fitness_values, false)
}

fn sweep_integer_param(name: &str, lo: usize, hi: usize, frames: usize, seed: u64, lite: bool) -> SensitivityResult {
    // For integer params, sweep each discrete value from lo to hi inclusive
    let steps = hi - lo + 1;
    let mut sweep_values = Vec::with_capacity(steps);
    let mut fitness_values = Vec::with_capacity(steps);

    for val in lo..=hi {
        sweep_values.push(val as f32);

        let mut genome = WorldGenome::default_with_seed(seed);
        set_genome_integer_param(&mut genome, name, val);
        let fitness = evaluate_fitness(&genome, frames, lite);
        fitness_values.push(fitness);
    }

    compute_sensitivity(name, sweep_values, fitness_values, true)
}

fn compute_sensitivity(name: &str, sweep_values: Vec<f32>, fitness_values: Vec<f32>, is_integer: bool) -> SensitivityResult {
    let min_f = fitness_values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_f = fitness_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean_f: f32 = fitness_values.iter().sum::<f32>() / fitness_values.len().max(1) as f32;
    let sensitivity = if mean_f.abs() > 1e-10 {
        (max_f - min_f) / mean_f
    } else {
        0.0
    };

    SensitivityResult {
        parameter: name.to_string(),
        sensitivity_index: sensitivity,
        min_fitness: min_f,
        max_fitness: max_f,
        mean_fitness: mean_f,
        sweep_values,
        fitness_values,
        is_integer,
    }
}

fn ascii_bar(val: f32, max_val: f32, width: usize) -> String {
    let filled = if max_val > 0.0 {
        ((val / max_val) * width as f32).round() as usize
    } else {
        0
    };
    let filled = filled.min(width);
    format!("{}{}", "\u{2588}".repeat(filled), "\u{2591}".repeat(width - filled))
}

fn main() {
    let args = parse_args();
    let start = Instant::now();

    eprintln!("=== Terrarium Parameter Sensitivity Analysis ===");
    eprintln!("Frames/eval: {}  Resolution: {}  Seed: {}  Lite: {}  Integers: {}",
        args.frames, args.resolution, args.seed, args.lite, args.include_integers);
    eprintln!();

    let mut results: Vec<SensitivityResult> = Vec::new();

    // Build list of params to sweep
    let sweep_specific = args.param.is_some();
    let target_param = args.param.as_deref();

    // Continuous params
    let continuous: Vec<(&str, usize)> = if let Some(p) = target_param {
        CONTINUOUS_PARAMS.iter().filter(|(name, _)| *name == p).cloned().collect()
    } else {
        CONTINUOUS_PARAMS.to_vec()
    };

    for (name, _idx) in &continuous {
        eprint!("  Sweeping {:25}... ", name);
        let r = sweep_param(name, args.resolution, args.frames, args.seed, args.lite);
        eprintln!("SI={:.4}  range=[{:.2}, {:.2}]  mean={:.2}", r.sensitivity_index, r.min_fitness, r.max_fitness, r.mean_fitness);
        results.push(r);
    }

    // Integer params (if requested, or if --param matches an integer param)
    let integers: Vec<(&str, usize, (usize, usize))> = if let Some(p) = target_param {
        INTEGER_PARAMS.iter().filter(|(name, _, _)| *name == p).cloned().collect()
    } else if args.include_integers {
        INTEGER_PARAMS.to_vec()
    } else {
        vec![]
    };

    for (name, _idx, (lo, hi)) in &integers {
        let steps = hi - lo + 1;
        eprint!("  Sweeping {:25} ({} discrete values)... ", name, steps);
        let r = sweep_integer_param(name, *lo, *hi, args.frames, args.seed, args.lite);
        eprintln!("SI={:.4}  range=[{:.2}, {:.2}]  mean={:.2}", r.sensitivity_index, r.min_fitness, r.max_fitness, r.mean_fitness);
        results.push(r);
    }

    if results.is_empty() {
        if sweep_specific {
            let all_names: Vec<&str> = CONTINUOUS_PARAMS.iter().map(|(n,_)| *n)
                .chain(INTEGER_PARAMS.iter().map(|(n,_,_)| *n))
                .collect();
            eprintln!("No sweepable parameter found. Available: {}", all_names.join(", "));
        }
        std::process::exit(1);
    }

    // Sort by sensitivity index (descending)
    results.sort_by(|a, b| b.sensitivity_index.partial_cmp(&a.sensitivity_index).unwrap_or(std::cmp::Ordering::Equal));

    eprintln!();
    eprintln!("=== Sensitivity Ranking (descending) ===");
    eprintln!();
    let max_si = results.first().map(|r| r.sensitivity_index).unwrap_or(1.0).max(0.01);
    eprintln!("{:<25} {:>8} {:>5}  {}", "Parameter", "SI", "Type", "Bar");
    eprintln!("{}", "-".repeat(75));
    for r in &results {
        let bar = ascii_bar(r.sensitivity_index, max_si, 30);
        let kind = if r.is_integer { "int" } else { "cont" };
        eprintln!("{:<25} {:>8.4} {:>5}  {}", r.parameter, r.sensitivity_index, kind, bar);
    }

    let elapsed = start.elapsed();
    eprintln!();
    eprintln!("Total time: {:.2}s", elapsed.as_secs_f64());

    // Output
    if let Some(ref path) = args.output {
        let content = if args.csv {
            let mut csv = String::from("parameter,sensitivity_index,min_fitness,max_fitness,mean_fitness,is_integer\n");
            for r in &results {
                csv.push_str(&format!("{},{:.6},{:.6},{:.6},{:.6},{}\n", r.parameter, r.sensitivity_index, r.min_fitness, r.max_fitness, r.mean_fitness, r.is_integer));
            }
            csv
        } else {
            serde_json::to_string_pretty(&results).unwrap_or_default()
        };
        if let Err(e) = fs::write(path, &content) {
            eprintln!("Error writing output: {}", e);
        } else {
            eprintln!("Results written to: {}", path);
        }
    }
}
