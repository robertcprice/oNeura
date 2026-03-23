//! Computational Performance Profiler for Terrarium.
//!
//! Measures throughput (frames/sec) and step_frame() latency at different
//! world scales: grid sizes from 8x6 to 128x96, organism counts from 2 to 32.
//!
//! Usage:
//!   terrarium_profile --scales small --frames 50
//!   terrarium_profile --scales all --output perf_report.json

use oneura_core::terrarium::WorldGenome;
use std::env;
use std::fs;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Scale definitions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ScaleConfig {
    name: &'static str,
    plants: usize,
    flies: usize,
    water: usize,
    lite: bool,
}

const SCALES: &[ScaleConfig] = &[
    ScaleConfig {
        name: "tiny",
        plants: 2,
        flies: 0,
        water: 1,
        lite: true,
    },
    ScaleConfig {
        name: "small",
        plants: 4,
        flies: 1,
        water: 2,
        lite: true,
    },
    ScaleConfig {
        name: "medium",
        plants: 8,
        flies: 2,
        water: 3,
        lite: false,
    },
    ScaleConfig {
        name: "large",
        plants: 16,
        flies: 4,
        water: 4,
        lite: false,
    },
    ScaleConfig {
        name: "dense",
        plants: 16,
        flies: 6,
        water: 5,
        lite: false,
    },
];

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Args {
    scales: Vec<String>,
    frames: usize,
    warmup: usize,
    seed: u64,
    output: Option<String>,
    csv: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            scales: vec!["all".into()],
            frames: 50,
            warmup: 5,
            seed: 42,
            output: None,
            csv: false,
        }
    }
}

fn parse_args() -> Args {
    let mut args = Args::default();
    let mut i = 1;
    let argv: Vec<String> = env::args().collect();
    while i < argv.len() {
        match argv[i].as_str() {
            "--scales" | "-s" => {
                if let Some(val) = argv.get(i + 1) {
                    args.scales = val.split(',').map(|s| s.trim().to_string()).collect();
                }
                i += 1;
            }
            "--frames" | "-f" => {
                args.frames = argv
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(args.frames);
                i += 1;
            }
            "--warmup" | "-w" => {
                args.warmup = argv
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(args.warmup);
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
            "--csv" => args.csv = true,
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
    println!("Terrarium Computational Performance Profiler");
    println!();
    println!("Usage: terrarium_profile [options]");
    println!();
    println!("Options:");
    println!("  --scales <LIST>    Comma-separated scale names or 'all' (default: all)");
    println!("  --frames <N>       Measurement frames after warmup (default: 50)");
    println!("  --warmup <N>       Warmup frames before measurement (default: 5)");
    println!("  --seed <N>         Random seed (default: 42)");
    println!("  --output <PATH>    Write results to JSON/CSV file");
    println!("  --csv              Output CSV instead of JSON");
    println!("  --help, -h         Show this help");
    println!();
    println!("Available scales:");
    for sc in SCALES {
        let grid = if sc.lite { "10x8x2" } else { "20x16x2" };
        println!(
            "  {:8} {} grid, {} plants, {} flies",
            sc.name, grid, sc.plants, sc.flies
        );
    }
}

// ---------------------------------------------------------------------------
// Profiling
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize)]
struct ProfileResult {
    scale: String,
    grid_size: String,
    cells: usize,
    plants: usize,
    flies: usize,
    warmup_frames: usize,
    measurement_frames: usize,
    total_ms: f64,
    mean_frame_ms: f64,
    min_frame_ms: f64,
    max_frame_ms: f64,
    p50_frame_ms: f64,
    p99_frame_ms: f64,
    fps: f64,
    throughput_cells_per_sec: f64,
}

fn profile_scale(scale: &ScaleConfig, frames: usize, warmup: usize, seed: u64) -> ProfileResult {
    let mut genome = WorldGenome::default_with_seed(seed);
    genome.plant_count = scale.plants;
    genome.fly_count = scale.flies;
    genome.water_source_count = scale.water;

    let mut world = if scale.lite {
        genome.build_world_lite().expect("Failed to build world")
    } else {
        genome.build_world().expect("Failed to build world")
    };

    let grid_w = world.config.width;
    let grid_h = world.config.height;
    let grid_d = world.config.depth;
    let cells = grid_w * grid_h * grid_d;

    // Warmup
    for _ in 0..warmup {
        let _ = world.step_frame();
    }

    // Measure
    let mut frame_times: Vec<f64> = Vec::with_capacity(frames);
    let total_start = Instant::now();

    for _ in 0..frames {
        let frame_start = Instant::now();
        let _ = world.step_frame();
        frame_times.push(frame_start.elapsed().as_secs_f64() * 1000.0);
    }

    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
    frame_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mean = frame_times.iter().sum::<f64>() / frame_times.len().max(1) as f64;
    let min = frame_times.first().copied().unwrap_or(0.0);
    let max = frame_times.last().copied().unwrap_or(0.0);
    let p50 = frame_times
        .get(frame_times.len() / 2)
        .copied()
        .unwrap_or(0.0);
    let p99_idx =
        ((frame_times.len() as f64 * 0.99) as usize).min(frame_times.len().saturating_sub(1));
    let p99 = frame_times.get(p99_idx).copied().unwrap_or(0.0);

    let fps = if mean > 0.0 { 1000.0 / mean } else { 0.0 };
    let throughput = fps * cells as f64;

    ProfileResult {
        scale: scale.name.to_string(),
        grid_size: format!("{}x{}x{}", grid_w, grid_h, grid_d),
        cells,
        plants: world.plants.len(),
        flies: world.flies.len(),
        warmup_frames: warmup,
        measurement_frames: frames,
        total_ms,
        mean_frame_ms: mean,
        min_frame_ms: min,
        max_frame_ms: max,
        p50_frame_ms: p50,
        p99_frame_ms: p99,
        fps,
        throughput_cells_per_sec: throughput,
    }
}

fn throughput_bar(fps: f64, max_fps: f64, width: usize) -> String {
    let filled = if max_fps > 0.0 {
        ((fps / max_fps) * width as f64).round() as usize
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

fn main() {
    let args = parse_args();
    let start = Instant::now();

    eprintln!("=== Terrarium Computational Performance Profiler ===");
    eprintln!(
        "Warmup: {} frames  Measurement: {} frames  Seed: {}",
        args.warmup, args.frames, args.seed
    );
    eprintln!();

    let selected: Vec<&ScaleConfig> = if args.scales.iter().any(|s| s == "all") {
        SCALES.iter().collect()
    } else {
        SCALES
            .iter()
            .filter(|sc| args.scales.contains(&sc.name.to_string()))
            .collect()
    };

    if selected.is_empty() {
        eprintln!(
            "No scales matched. Available: {}",
            SCALES.iter().map(|s| s.name).collect::<Vec<_>>().join(", ")
        );
        std::process::exit(1);
    }

    let mut results: Vec<ProfileResult> = Vec::new();

    for scale in &selected {
        let grid = if scale.lite { "10x8" } else { "20x16" };
        eprint!(
            "  Profiling {:8} ({}, {:>2} plants, {:>2} flies) ... ",
            scale.name, grid, scale.plants, scale.flies
        );
        let r = profile_scale(scale, args.frames, args.warmup, args.seed);
        eprintln!(
            "{:.1} fps  mean={:.2}ms  p99={:.2}ms  {:.0} cells/s",
            r.fps, r.mean_frame_ms, r.p99_frame_ms, r.throughput_cells_per_sec
        );
        results.push(r);
    }

    // Summary
    eprintln!();
    eprintln!("=== Performance Summary ===");
    eprintln!();
    eprintln!(
        "{:<10} {:>12} {:>8} {:>8} {:>10} {:>10} {:>12}",
        "Scale", "Grid", "Plants", "Flies", "FPS", "Mean(ms)", "Cells/s"
    );
    eprintln!("{}", "-".repeat(80));
    let max_fps = results.iter().map(|r| r.fps).fold(0.0f64, f64::max);
    for r in &results {
        eprintln!(
            "{:<10} {:>12} {:>8} {:>8} {:>10.1} {:>10.2} {:>12.0}",
            r.scale,
            r.grid_size,
            r.plants,
            r.flies,
            r.fps,
            r.mean_frame_ms,
            r.throughput_cells_per_sec
        );
    }

    // Scaling curve
    eprintln!();
    eprintln!("FPS scaling (relative to peak):");
    for r in &results {
        let bar = throughput_bar(r.fps, max_fps, 40);
        eprintln!("  {:8} {:>7.1} fps  {}", r.scale, r.fps, bar);
    }

    // Bottleneck analysis
    if results.len() >= 2 {
        let r0 = &results[0];
        let r1 = &results[results.len() - 1];
        let organisms_ratio = (r1.plants + r1.flies) as f64 / (r0.plants + r0.flies).max(1) as f64;
        let time_ratio = r1.mean_frame_ms / r0.mean_frame_ms.max(0.001);
        let scaling_exponent = if organisms_ratio > 1.0 {
            time_ratio.ln() / organisms_ratio.ln()
        } else {
            0.0
        };
        eprintln!();
        eprintln!(
            "Scaling analysis: {:.1}x organisms -> {:.1}x time (exponent: {:.2})",
            organisms_ratio, time_ratio, scaling_exponent
        );
        if scaling_exponent < 1.1 {
            eprintln!("  -> Near-linear scaling (good)");
        } else if scaling_exponent < 1.5 {
            eprintln!("  -> Slightly super-linear — watch for quadratic bottlenecks");
        } else {
            eprintln!("  -> Super-linear scaling — potential O(n^2) bottleneck");
        }
    }

    let total = start.elapsed();
    eprintln!();
    eprintln!("Total profiling time: {:.2}s", total.as_secs_f64());

    // Output
    if let Some(ref path) = args.output {
        let content = if args.csv {
            let mut csv = String::from("scale,grid,cells,plants,flies,fps,mean_ms,min_ms,max_ms,p50_ms,p99_ms,cells_per_sec\n");
            for r in &results {
                csv.push_str(&format!(
                    "{},{},{},{},{},{:.2},{:.3},{:.3},{:.3},{:.3},{:.3},{:.0}\n",
                    r.scale,
                    r.grid_size,
                    r.cells,
                    r.plants,
                    r.flies,
                    r.fps,
                    r.mean_frame_ms,
                    r.min_frame_ms,
                    r.max_frame_ms,
                    r.p50_frame_ms,
                    r.p99_frame_ms,
                    r.throughput_cells_per_sec
                ));
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
