use oneuro_metal::{DishBrainPongSim, PongScale};
use std::env;
use std::fs;
use std::process::ExitCode;

fn print_usage() {
    eprintln!(
        "Usage: cargo run --release --bin dishbrain_pong -- --scale <small|medium|large|mega> --rallies <n> --seed <n> [--benchmark-mode|--no-interval-biology] [--json <path>]"
    );
}

fn main() -> ExitCode {
    let mut scale = PongScale::Large;
    let mut rallies: usize = 80;
    let mut seed: u64 = 42;
    let mut benchmark_mode = false;
    let mut json_path: Option<String> = None;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--scale" => match args.next() {
                Some(value) => match PongScale::from_str(&value) {
                    Ok(parsed) => scale = parsed,
                    Err(err) => {
                        eprintln!("{}", err);
                        print_usage();
                        return ExitCode::FAILURE;
                    }
                },
                None => {
                    print_usage();
                    return ExitCode::FAILURE;
                }
            },
            "--rallies" => match args.next().and_then(|value| value.parse::<usize>().ok()) {
                Some(value) => rallies = value,
                None => {
                    print_usage();
                    return ExitCode::FAILURE;
                }
            },
            "--seed" => match args.next().and_then(|value| value.parse::<u64>().ok()) {
                Some(value) => seed = value,
                None => {
                    print_usage();
                    return ExitCode::FAILURE;
                }
            },
            "--json" => match args.next() {
                Some(value) => json_path = Some(value),
                None => {
                    print_usage();
                    return ExitCode::FAILURE;
                }
            },
            "--benchmark-mode" | "--no-interval-biology" => {
                benchmark_mode = true;
            }
            "--help" | "-h" => {
                print_usage();
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                print_usage();
                return ExitCode::FAILURE;
            }
        }
    }

    let mut sim = DishBrainPongSim::new(scale, seed);
    if benchmark_mode {
        sim.enable_latency_benchmark_mode();
    }
    let result = sim.run_replication(rallies);

    println!(
        "DishBrain Pong | scale={} | benchmark_mode={} | no_interval_biology={} | neurons={} | synapses={} | wall={:.3}s | bio={:.1}ms | rt_factor={:.6}x | hits={}/{} | gpu_dispatch_active={}",
        result.scale,
        result.benchmark_mode,
        result.no_interval_biology,
        result.neurons,
        result.synapses,
        result.wall_time_s,
        result.bio_time_ms,
        result.realtime_factor,
        result.total_hits,
        result.rallies,
        result.gpu_dispatch_active,
    );

    if let Some(path) = json_path {
        match serde_json::to_string_pretty(&result) {
            Ok(payload) => {
                if let Err(err) = fs::write(&path, payload) {
                    eprintln!("Failed to write {}: {}", path, err);
                    return ExitCode::FAILURE;
                }
            }
            Err(err) => {
                eprintln!("Failed to serialize result: {}", err);
                return ExitCode::FAILURE;
            }
        }
    }

    ExitCode::SUCCESS
}
