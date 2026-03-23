//! Parameter Sensitivity Analysis for Terrarium Evolution.
//!
//! Sweeps each WorldGenome parameter one-at-a-time (OAT) across a range of values,
//! measuring fitness response to quantify which parameters matter most.
//!
//! Usage:
//!   terrarium_sensitivity --frames 50 --resolution 10
//!   terrarium_sensitivity --param temperature_c --resolution 20 --output sensitivity.json
//!   terrarium_sensitivity --include-integers --resolution 5
//!   terrarium_sensitivity --multi-objective --resolution 10
//!   terrarium_sensitivity --high-res

use oneura_core::terrarium::WorldGenome;
use oneura_core::terrarium::{evaluate_fitness, FitnessObjective, GENOME_PARAM_NAMES};
use rayon::prelude::*;
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
    multi_objective: bool,
    sequential: bool,
    high_res: bool,
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
            multi_objective: false,
            sequential: false,
            high_res: false,
        }
    }
}

fn parse_args() -> Args {
    let mut args = Args::default();
    let mut i = 1;
    let argv: Vec<String> = env::args().collect();
    while i < argv.len() {
        match argv[i].as_str() {
            "--param" | "-p" => {
                args.param = argv.get(i + 1).cloned();
                i += 1;
            }
            "--frames" | "-f" => {
                args.frames = argv
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(args.frames);
                i += 1;
            }
            "--resolution" | "-r" => {
                args.resolution = argv
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(args.resolution);
                i += 1;
            }
            "--seed" | "-s" => {
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
            "--csv" => args.csv = true,
            "--lite" => args.lite = true,
            "--no-lite" => args.lite = false,
            "--include-integers" | "--integers" => args.include_integers = true,
            "--multi-objective" | "--multi" => args.multi_objective = true,
            "--sequential" => args.sequential = true,
            "--high-res" => {
                args.high_res = true;
                args.frames = 150;
                args.resolution = 25;
                args.include_integers = true;
                args.multi_objective = true;
            }
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
    println!("Terrarium Parameter Sensitivity Analysis");
    println!();
    println!("Usage: terrarium_sensitivity [options]");
    println!();
    println!("Options:");
    println!("  --param <NAME>        Sweep a single parameter (default: all continuous)");
    println!("  --frames <N>          Frames per evaluation world (default: 50)");
    println!("  --resolution <N>      Number of sweep points per parameter (default: 10)");
    println!("  --seed <N>            Master random seed (default: 42)");
    println!("  --output <PATH>       Write results to file");
    println!("  --csv                 Write results as CSV instead of JSON");
    println!("  --lite                Use lite worlds (default: true)");
    println!("  --no-lite             Use full-size worlds");
    println!("  --include-integers    Also sweep integer params (plant_count, fly_count, etc.)");
    println!("  --multi-objective     Evaluate all 6 fitness objectives per sweep point");
    println!("  --sequential          Disable parallel sweep (single-threaded)");
    println!("  --high-res            High-resolution preset: 150 frames, 25 points, all params, multi-objective");
    println!("  --help, -h            Show this help");
    println!();
    println!(
        "Continuous: {}",
        CONTINUOUS_PARAMS
            .iter()
            .map(|(n, _)| *n)
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "Integer:    {}",
        INTEGER_PARAMS
            .iter()
            .map(|(n, _, _)| *n)
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("All genome: {}", GENOME_PARAM_NAMES.join(", "));
}

/// Continuous sweepable parameters (name, genome_index).
const CONTINUOUS_PARAMS: &[(&str, usize)] = &[
    ("proton_scale", 0),
    ("temperature_c", 1),
    ("water_volume", 3),
    ("moisture_scale", 4),
    ("respiration_vmax", 11),
    ("nitrification_vmax", 12),
    ("photosynthesis_vmax", 13),
    ("mineralization_vmax", 14),
    ("time_warp", 15),
    ("psc_scale", 9),
];

/// Integer sweepable parameters (name, genome_index, (lo, hi)).
const INTEGER_PARAMS: &[(&str, usize, (usize, usize))] = &[
    ("plant_count", 5, (2, 16)),
    ("fruit_count", 6, (0, 8)),
    ("fly_count", 7, (0, 6)),
    ("water_count", 2, (1, 6)),
    ("microbe_count", 8, (0, 6)),
];

/// Fitness objectives for multi-objective analysis.
const OBJECTIVES: &[(&str, FitnessObjective)] = &[
    ("biomass", FitnessObjective::MaxBiomass),
    ("biodiversity", FitnessObjective::MaxBiodiversity),
    ("stability", FitnessObjective::MaxStability),
    ("carbon", FitnessObjective::MaxCarbonSequestration),
    ("fruit", FitnessObjective::MaxFruitProduction),
    ("microbial", FitnessObjective::MaxMicrobialHealth),
];

/// Ranges for each continuous sweepable parameter (lo, hi).
fn param_range(name: &str) -> (f32, f32) {
    match name {
        "proton_scale" => (0.3, 3.0),
        "temperature_c" => (10.0, 40.0),
        "water_volume" => (50.0, 300.0),
        "moisture_scale" => (0.5, 2.0),
        "respiration_vmax" => (0.3, 3.0),
        "nitrification_vmax" => (0.3, 3.0),
        "photosynthesis_vmax" => (0.3, 3.0),
        "mineralization_vmax" => (0.3, 3.0),
        "time_warp" => (100.0, 2000.0),
        "psc_scale" => (5.0, 100.0),
        _ => (0.0, 1.0),
    }
}

fn set_genome_param(genome: &mut WorldGenome, name: &str, val: f32) {
    match name {
        "proton_scale" => genome.initial_proton_scale = val,
        "temperature_c" => genome.soil_temperature_c = val,
        "water_volume" => genome.water_volume = val,
        "moisture_scale" => genome.initial_moisture_scale = val,
        "respiration_vmax" => genome.respiration_vmax_scale = val,
        "nitrification_vmax" => genome.nitrification_vmax_scale = val,
        "photosynthesis_vmax" => genome.photosynthesis_vmax_scale = val,
        "mineralization_vmax" => genome.mineralization_vmax_scale = val,
        "time_warp" => genome.time_warp = val,
        "psc_scale" => genome.fly_psc_scale = val,
        _ => {}
    }
}

fn set_genome_integer_param(genome: &mut WorldGenome, name: &str, val: usize) {
    match name {
        "plant_count" => genome.plant_count = val,
        "fruit_count" => genome.fruit_count = val,
        "fly_count" => genome.fly_count = val,
        "water_count" => genome.water_source_count = val,
        "microbe_count" => genome.microbe_cohort_count = val,
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Single-objective result
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Multi-objective result
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize)]
struct MultiObjectiveResult {
    parameter: String,
    is_integer: bool,
    sweep_values: Vec<f32>,
    /// Per-objective SI and fitness curves.
    objectives: Vec<ObjectiveSensitivity>,
    /// Composite SI = mean of all per-objective SIs.
    composite_si: f32,
}

#[derive(Debug, Clone, serde::Serialize)]
struct ObjectiveSensitivity {
    objective: String,
    sensitivity_index: f32,
    min_fitness: f32,
    max_fitness: f32,
    mean_fitness: f32,
    fitness_values: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Evaluation helpers
// ---------------------------------------------------------------------------

fn evaluate_single_fitness(genome: &WorldGenome, frames: usize, lite: bool) -> f32 {
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

/// Run a world and evaluate ALL objectives at once (avoids rebuilding per objective).
fn evaluate_multi_fitness(genome: &WorldGenome, frames: usize, lite: bool) -> Vec<f32> {
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
            OBJECTIVES
                .iter()
                .map(|(_, obj)| evaluate_fitness(*obj, &snap, &[]))
                .collect()
        }
        Err(_) => vec![0.0; OBJECTIVES.len()],
    }
}

// ---------------------------------------------------------------------------
// Single-objective sweeps
// ---------------------------------------------------------------------------

/// A sweep task: either continuous or integer.
enum SweepTask {
    Continuous {
        name: &'static str,
    },
    Integer {
        name: &'static str,
        lo: usize,
        hi: usize,
    },
}

fn sweep_continuous(
    name: &str,
    resolution: usize,
    frames: usize,
    seed: u64,
    lite: bool,
) -> SensitivityResult {
    let (lo, hi) = param_range(name);
    let mut sweep_values = Vec::with_capacity(resolution);
    let mut fitness_values = Vec::with_capacity(resolution);

    for i in 0..resolution {
        let t = if resolution > 1 {
            i as f32 / (resolution - 1) as f32
        } else {
            0.5
        };
        let val = lo + t * (hi - lo);
        sweep_values.push(val);

        let mut genome = WorldGenome::default_with_seed(seed);
        set_genome_param(&mut genome, name, val);
        let fitness = evaluate_single_fitness(&genome, frames, lite);
        fitness_values.push(fitness);
    }

    compute_sensitivity(name, sweep_values, fitness_values, false)
}

fn sweep_integer(
    name: &str,
    lo: usize,
    hi: usize,
    frames: usize,
    seed: u64,
    lite: bool,
) -> SensitivityResult {
    let steps = hi - lo + 1;
    let mut sweep_values = Vec::with_capacity(steps);
    let mut fitness_values = Vec::with_capacity(steps);

    for val in lo..=hi {
        sweep_values.push(val as f32);
        let mut genome = WorldGenome::default_with_seed(seed);
        set_genome_integer_param(&mut genome, name, val);
        let fitness = evaluate_single_fitness(&genome, frames, lite);
        fitness_values.push(fitness);
    }

    compute_sensitivity(name, sweep_values, fitness_values, true)
}

fn compute_sensitivity(
    name: &str,
    sweep_values: Vec<f32>,
    fitness_values: Vec<f32>,
    is_integer: bool,
) -> SensitivityResult {
    let min_f = fitness_values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_f = fitness_values
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
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

// ---------------------------------------------------------------------------
// Multi-objective sweeps
// ---------------------------------------------------------------------------

fn sweep_continuous_multi(
    name: &str,
    resolution: usize,
    frames: usize,
    seed: u64,
    lite: bool,
) -> MultiObjectiveResult {
    let (lo, hi) = param_range(name);
    let mut sweep_values = Vec::with_capacity(resolution);
    // fitness_matrix[sweep_point][objective]
    let mut fitness_matrix: Vec<Vec<f32>> = Vec::with_capacity(resolution);

    for i in 0..resolution {
        let t = if resolution > 1 {
            i as f32 / (resolution - 1) as f32
        } else {
            0.5
        };
        let val = lo + t * (hi - lo);
        sweep_values.push(val);

        let mut genome = WorldGenome::default_with_seed(seed);
        set_genome_param(&mut genome, name, val);
        fitness_matrix.push(evaluate_multi_fitness(&genome, frames, lite));
    }

    build_multi_result(name, false, sweep_values, &fitness_matrix)
}

fn sweep_integer_multi(
    name: &str,
    lo: usize,
    hi: usize,
    frames: usize,
    seed: u64,
    lite: bool,
) -> MultiObjectiveResult {
    let steps = hi - lo + 1;
    let mut sweep_values = Vec::with_capacity(steps);
    let mut fitness_matrix: Vec<Vec<f32>> = Vec::with_capacity(steps);

    for val in lo..=hi {
        sweep_values.push(val as f32);
        let mut genome = WorldGenome::default_with_seed(seed);
        set_genome_integer_param(&mut genome, name, val);
        fitness_matrix.push(evaluate_multi_fitness(&genome, frames, lite));
    }

    build_multi_result(name, true, sweep_values, &fitness_matrix)
}

fn build_multi_result(
    name: &str,
    is_integer: bool,
    sweep_values: Vec<f32>,
    fitness_matrix: &[Vec<f32>],
) -> MultiObjectiveResult {
    let n_obj = OBJECTIVES.len();
    let mut objectives = Vec::with_capacity(n_obj);
    let mut si_sum = 0.0f32;

    for obj_idx in 0..n_obj {
        let values: Vec<f32> = fitness_matrix.iter().map(|row| row[obj_idx]).collect();
        let min_f = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_f = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean_f = values.iter().sum::<f32>() / values.len().max(1) as f32;
        let si = if mean_f.abs() > 1e-10 {
            (max_f - min_f) / mean_f
        } else {
            0.0
        };
        si_sum += si;
        objectives.push(ObjectiveSensitivity {
            objective: OBJECTIVES[obj_idx].0.to_string(),
            sensitivity_index: si,
            min_fitness: min_f,
            max_fitness: max_f,
            mean_fitness: mean_f,
            fitness_values: values,
        });
    }

    MultiObjectiveResult {
        parameter: name.to_string(),
        is_integer,
        sweep_values,
        composite_si: si_sum / n_obj as f32,
        objectives,
    }
}

// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------

fn ascii_bar(val: f32, max_val: f32, width: usize) -> String {
    let filled = if max_val > 0.0 {
        ((val / max_val) * width as f32).round() as usize
    } else {
        0
    };
    let filled = filled.min(width);
    format!(
        "{}{}",
        "\u{2588}".repeat(filled),
        "\u{2591}".repeat(width - filled)
    )
}

fn print_single_results(results: &[SensitivityResult]) {
    eprintln!();
    eprintln!("=== Sensitivity Ranking (descending) ===");
    eprintln!();
    let max_si = results
        .first()
        .map(|r| r.sensitivity_index)
        .unwrap_or(1.0)
        .max(0.01);
    eprintln!("{:<25} {:>8} {:>5}  {}", "Parameter", "SI", "Type", "Bar");
    eprintln!("{}", "-".repeat(75));
    for r in results {
        let bar = ascii_bar(r.sensitivity_index, max_si, 30);
        let kind = if r.is_integer { "int" } else { "cont" };
        eprintln!(
            "{:<25} {:>8.4} {:>5}  {}",
            r.parameter, r.sensitivity_index, kind, bar
        );
    }
}

fn print_multi_results(results: &[MultiObjectiveResult]) {
    eprintln!();
    eprintln!("=== Multi-Objective Sensitivity Ranking (by composite SI) ===");
    eprintln!();

    let max_si = results
        .first()
        .map(|r| r.composite_si)
        .unwrap_or(1.0)
        .max(0.01);

    // Header
    let obj_names: Vec<&str> = OBJECTIVES.iter().map(|(n, _)| *n).collect();
    eprint!("{:<22} {:>8} {:>5}", "Parameter", "CompSI", "Type");
    for name in &obj_names {
        // Truncate objective name to 8 chars for column
        let short: String = name.chars().take(8).collect();
        eprint!(" {:>8}", short);
    }
    eprintln!("  Bar");
    eprintln!("{}", "-".repeat(22 + 8 + 5 + obj_names.len() * 9 + 5 + 30));

    for r in results {
        let bar = ascii_bar(r.composite_si, max_si, 20);
        let kind = if r.is_integer { "int" } else { "cont" };
        eprint!("{:<22} {:>8.4} {:>5}", r.parameter, r.composite_si, kind);
        for obj in &r.objectives {
            eprint!(" {:>8.4}", obj.sensitivity_index);
        }
        eprintln!("  {}", bar);
    }

    // Per-objective rankings
    eprintln!();
    eprintln!("=== Per-Objective Top 3 Most Sensitive Parameters ===");
    eprintln!();
    for (obj_idx, (obj_name, _)) in OBJECTIVES.iter().enumerate() {
        let mut ranked: Vec<(&str, f32)> = results
            .iter()
            .map(|r| {
                (
                    r.parameter.as_str(),
                    r.objectives[obj_idx].sensitivity_index,
                )
            })
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top3: Vec<String> = ranked
            .iter()
            .take(3)
            .map(|(name, si)| format!("{}({:.3})", name, si))
            .collect();
        eprintln!("  {:<14}: {}", obj_name, top3.join(", "));
    }
}

// ---------------------------------------------------------------------------
// Output serialization
// ---------------------------------------------------------------------------

#[derive(serde::Serialize)]
#[serde(untagged)]
enum OutputResults {
    Single(Vec<SensitivityResult>),
    Multi(Vec<MultiObjectiveResult>),
}

fn write_output(path: &str, results: &OutputResults, csv: bool) {
    let content = if csv {
        match results {
            OutputResults::Single(r) => {
                let mut csv = String::from(
                    "parameter,sensitivity_index,min_fitness,max_fitness,mean_fitness,is_integer\n",
                );
                for row in r {
                    csv.push_str(&format!(
                        "{},{:.6},{:.6},{:.6},{:.6},{}\n",
                        row.parameter,
                        row.sensitivity_index,
                        row.min_fitness,
                        row.max_fitness,
                        row.mean_fitness,
                        row.is_integer
                    ));
                }
                csv
            }
            OutputResults::Multi(r) => {
                let obj_names: Vec<&str> = OBJECTIVES.iter().map(|(n, _)| *n).collect();
                let mut csv = String::from("parameter,composite_si,is_integer");
                for name in &obj_names {
                    csv.push_str(&format!(",{}_si,{}_mean", name, name));
                }
                csv.push('\n');
                for row in r {
                    csv.push_str(&format!(
                        "{},{:.6},{}",
                        row.parameter, row.composite_si, row.is_integer
                    ));
                    for obj in &row.objectives {
                        csv.push_str(&format!(
                            ",{:.6},{:.6}",
                            obj.sensitivity_index, obj.mean_fitness
                        ));
                    }
                    csv.push('\n');
                }
                csv
            }
        }
    } else {
        serde_json::to_string_pretty(&results).unwrap_or_default()
    };
    if let Err(e) = fs::write(path, &content) {
        eprintln!("Error writing output: {}", e);
    } else {
        eprintln!("Results written to: {}", path);
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let args = parse_args();
    let start = Instant::now();

    let mode_str = if args.high_res {
        "HIGH-RES"
    } else if args.multi_objective {
        "MULTI-OBJECTIVE"
    } else {
        "STANDARD"
    };
    let par_str = if args.sequential {
        "sequential"
    } else {
        "parallel"
    };

    eprintln!(
        "=== Terrarium Parameter Sensitivity Analysis ({}) ===",
        mode_str
    );
    eprintln!(
        "Frames/eval: {}  Resolution: {}  Seed: {}  Lite: {}  Integers: {}  Mode: {}",
        args.frames, args.resolution, args.seed, args.lite, args.include_integers, par_str
    );
    if args.multi_objective {
        let obj_names: Vec<&str> = OBJECTIVES.iter().map(|(n, _)| *n).collect();
        eprintln!("Objectives: {}", obj_names.join(", "));
    }
    eprintln!();

    // Build task list
    let target_param = args.param.as_deref();
    let mut tasks: Vec<SweepTask> = Vec::new();

    // Continuous params
    let continuous: Vec<(&str, usize)> = if let Some(p) = target_param {
        CONTINUOUS_PARAMS
            .iter()
            .filter(|(name, _)| *name == p)
            .cloned()
            .collect()
    } else {
        CONTINUOUS_PARAMS.to_vec()
    };
    for (name, _) in &continuous {
        tasks.push(SweepTask::Continuous { name });
    }

    // Integer params
    let integers: Vec<(&str, usize, (usize, usize))> = if let Some(p) = target_param {
        INTEGER_PARAMS
            .iter()
            .filter(|(name, _, _)| *name == p)
            .cloned()
            .collect()
    } else if args.include_integers {
        INTEGER_PARAMS.to_vec()
    } else {
        vec![]
    };
    for (name, _, (lo, hi)) in &integers {
        tasks.push(SweepTask::Integer {
            name,
            lo: *lo,
            hi: *hi,
        });
    }

    if tasks.is_empty() {
        if target_param.is_some() {
            let all_names: Vec<&str> = CONTINUOUS_PARAMS
                .iter()
                .map(|(n, _)| *n)
                .chain(INTEGER_PARAMS.iter().map(|(n, _, _)| *n))
                .collect();
            eprintln!(
                "No sweepable parameter found. Available: {}",
                all_names.join(", ")
            );
        }
        std::process::exit(1);
    }

    let n_tasks = tasks.len();
    eprintln!("Sweeping {} parameters...", n_tasks);
    eprintln!();

    if args.multi_objective {
        // ---- Multi-objective mode ----
        let resolution = args.resolution;
        let frames = args.frames;
        let seed = args.seed;
        let lite = args.lite;

        let mut results: Vec<MultiObjectiveResult> = if args.sequential {
            tasks
                .iter()
                .enumerate()
                .map(|(i, task)| {
                    let r = match task {
                        SweepTask::Continuous { name } => {
                            eprint!("  [{}/{}] Sweeping {:25}... ", i + 1, n_tasks, name);
                            let r = sweep_continuous_multi(name, resolution, frames, seed, lite);
                            eprintln!("CompSI={:.4}", r.composite_si);
                            r
                        }
                        SweepTask::Integer { name, lo, hi } => {
                            eprint!(
                                "  [{}/{}] Sweeping {:25} ({} values)... ",
                                i + 1,
                                n_tasks,
                                name,
                                hi - lo + 1
                            );
                            let r = sweep_integer_multi(name, *lo, *hi, frames, seed, lite);
                            eprintln!("CompSI={:.4}", r.composite_si);
                            r
                        }
                    };
                    r
                })
                .collect()
        } else {
            tasks
                .par_iter()
                .map(|task| match task {
                    SweepTask::Continuous { name } => {
                        sweep_continuous_multi(name, resolution, frames, seed, lite)
                    }
                    SweepTask::Integer { name, lo, hi } => {
                        sweep_integer_multi(name, *lo, *hi, frames, seed, lite)
                    }
                })
                .collect()
        };

        // Sort by composite SI descending
        results.sort_by(|a, b| {
            b.composite_si
                .partial_cmp(&a.composite_si)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        print_multi_results(&results);

        let elapsed = start.elapsed();
        eprintln!();
        eprintln!(
            "Total time: {:.2}s ({} params x {} objectives)",
            elapsed.as_secs_f64(),
            n_tasks,
            OBJECTIVES.len()
        );

        if let Some(ref path) = args.output {
            write_output(path, &OutputResults::Multi(results), args.csv);
        }
    } else {
        // ---- Single-objective mode ----
        let resolution = args.resolution;
        let frames = args.frames;
        let seed = args.seed;
        let lite = args.lite;

        let mut results: Vec<SensitivityResult> = if args.sequential {
            tasks
                .iter()
                .enumerate()
                .map(|(i, task)| {
                    let r = match task {
                        SweepTask::Continuous { name } => {
                            eprint!("  [{}/{}] Sweeping {:25}... ", i + 1, n_tasks, name);
                            let r = sweep_continuous(name, resolution, frames, seed, lite);
                            eprintln!(
                                "SI={:.4}  range=[{:.2}, {:.2}]  mean={:.2}",
                                r.sensitivity_index, r.min_fitness, r.max_fitness, r.mean_fitness
                            );
                            r
                        }
                        SweepTask::Integer { name, lo, hi } => {
                            eprint!(
                                "  [{}/{}] Sweeping {:25} ({} values)... ",
                                i + 1,
                                n_tasks,
                                name,
                                hi - lo + 1
                            );
                            let r = sweep_integer(name, *lo, *hi, frames, seed, lite);
                            eprintln!(
                                "SI={:.4}  range=[{:.2}, {:.2}]  mean={:.2}",
                                r.sensitivity_index, r.min_fitness, r.max_fitness, r.mean_fitness
                            );
                            r
                        }
                    };
                    r
                })
                .collect()
        } else {
            tasks
                .par_iter()
                .map(|task| match task {
                    SweepTask::Continuous { name } => {
                        sweep_continuous(name, resolution, frames, seed, lite)
                    }
                    SweepTask::Integer { name, lo, hi } => {
                        sweep_integer(name, *lo, *hi, frames, seed, lite)
                    }
                })
                .collect()
        };

        // Sort by SI descending
        results.sort_by(|a, b| {
            b.sensitivity_index
                .partial_cmp(&a.sensitivity_index)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        print_single_results(&results);

        let elapsed = start.elapsed();
        eprintln!();
        eprintln!(
            "Total time: {:.2}s ({} params)",
            elapsed.as_secs_f64(),
            n_tasks
        );

        if let Some(ref path) = args.output {
            write_output(path, &OutputResults::Single(results), args.csv);
        }
    }
}
