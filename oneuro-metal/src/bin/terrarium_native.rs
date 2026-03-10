use oneuro_metal::{TerrariumWorld, TerrariumWorldSnapshot};
use std::env;
use std::io::{self, Write};
use std::process::ExitCode;
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
struct Cli {
    seed: u64,
    fps: u64,
    frames: Option<usize>,
    no_render: bool,
    cpu_substrate: bool,
    summary_every: usize,
}

impl Default for Cli {
    fn default() -> Self {
        Self {
            seed: 7,
            fps: 12,
            frames: None,
            no_render: false,
            cpu_substrate: false,
            summary_every: 30,
        }
    }
}

fn print_usage() {
    eprintln!(
        "Usage: cargo run --release --bin terrarium_native -- [--seed <n>] [--fps <n>] [--frames <n>] [--summary-every <n>] [--no-render] [--cpu-substrate]"
    );
}

fn parse_args() -> Result<Cli, String> {
    let mut cli = Cli::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--seed" => {
                cli.seed = args
                    .next()
                    .ok_or_else(|| "missing value for --seed".to_string())?
                    .parse::<u64>()
                    .map_err(|_| "invalid integer for --seed".to_string())?;
            }
            "--fps" => {
                cli.fps = args
                    .next()
                    .ok_or_else(|| "missing value for --fps".to_string())?
                    .parse::<u64>()
                    .map_err(|_| "invalid integer for --fps".to_string())?;
            }
            "--frames" => {
                cli.frames = Some(
                    args.next()
                        .ok_or_else(|| "missing value for --frames".to_string())?
                        .parse::<usize>()
                        .map_err(|_| "invalid integer for --frames".to_string())?,
                );
            }
            "--summary-every" => {
                cli.summary_every = args
                    .next()
                    .ok_or_else(|| "missing value for --summary-every".to_string())?
                    .parse::<usize>()
                    .map_err(|_| "invalid integer for --summary-every".to_string())?
                    .max(1);
            }
            "--no-render" => cli.no_render = true,
            "--cpu-substrate" => cli.cpu_substrate = true,
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(cli)
}

fn render_terminal(
    world: &TerrariumWorld,
    snapshot: &TerrariumWorldSnapshot,
    frame_idx: usize,
) -> io::Result<()> {
    let width = world.config.width;
    let height = world.config.height;
    let mut grid = vec![vec!['.'; width]; height];

    for water in &world.waters {
        if water.alive {
            grid[water.y.min(height - 1)][water.x.min(width - 1)] = '~';
        }
    }
    for plant in &world.plants {
        grid[plant.y.min(height - 1)][plant.x.min(width - 1)] = 'T';
    }
    for fruit in &world.fruits {
        if fruit.source.alive && fruit.source.sugar_content > 0.01 {
            grid[fruit.source.y.min(height - 1)][fruit.source.x.min(width - 1)] = 'o';
        }
    }
    for fly in &world.flies {
        let body = fly.body_state();
        let x = body.x.round().clamp(0.0, (width - 1) as f32) as usize;
        let y = body.y.round().clamp(0.0, (height - 1) as f32) as usize;
        grid[y][x] = if body.is_flying { '*' } else { 'f' };
    }

    print!("\x1b[2J\x1b[H");
    println!(
        "Native Terrarium | frame={} | time={} | light={:.2} | temp={:.2}C | humidity={:.2}",
        frame_idx,
        world.time_label(),
        snapshot.light,
        snapshot.temperature,
        snapshot.humidity,
    );
    println!(
        "plants={} fruits={} seeds={} flies={} food={:.3} fly_food={:.3}",
        snapshot.plants,
        snapshot.fruits,
        snapshot.seeds,
        snapshot.flies,
        snapshot.food_remaining,
        snapshot.fly_food_total,
    );
    println!(
        "soil_moisture={:.3} deep={:.3} microbes={:.3} symbionts={:.3} atm_co2={:.4}",
        snapshot.mean_soil_moisture,
        snapshot.mean_deep_moisture,
        snapshot.mean_microbes,
        snapshot.mean_symbionts,
        snapshot.mean_atmospheric_co2,
    );
    println!(
        "cells={:.1} vitality={:.3} energy={:.3} division={:.3} substrate={} steps={}",
        snapshot.total_plant_cells,
        snapshot.mean_cell_vitality,
        snapshot.mean_cell_energy,
        snapshot.mean_division_pressure,
        snapshot.substrate_backend,
        snapshot.substrate_steps,
    );
    println!("legend: ~ water  T plant  o fruit  f grounded fly  * flying fly");
    println!();
    for row in grid {
        let line: String = row.into_iter().collect();
        println!("{line}");
    }
    io::stdout().flush()
}

fn print_summary(frame_idx: usize, world: &TerrariumWorld, snapshot: &TerrariumWorldSnapshot) {
    println!(
        "frame={} time={} plants={} fruits={} seeds={} flies={} food={:.3} fly_food={:.3} light={:.2} temp={:.2} humidity={:.2} atm_co2={:.4} substrate={} cells={:.1}",
        frame_idx,
        world.time_label(),
        snapshot.plants,
        snapshot.fruits,
        snapshot.seeds,
        snapshot.flies,
        snapshot.food_remaining,
        snapshot.fly_food_total,
        snapshot.light,
        snapshot.temperature,
        snapshot.humidity,
        snapshot.mean_atmospheric_co2,
        snapshot.substrate_backend,
        snapshot.total_plant_cells,
    );
}

fn main() -> ExitCode {
    let cli = match parse_args() {
        Ok(parsed) => parsed,
        Err(err) => {
            eprintln!("{err}");
            print_usage();
            return ExitCode::FAILURE;
        }
    };

    let mut world = match TerrariumWorld::demo(cli.seed, !cli.cpu_substrate) {
        Ok(world) => world,
        Err(err) => {
            eprintln!("failed to build terrarium: {err}");
            return ExitCode::FAILURE;
        }
    };

    let frame_budget = if cli.fps > 0 {
        Some(Duration::from_secs_f64(1.0 / cli.fps as f64))
    } else {
        None
    };
    let mut frame_idx = 0usize;

    loop {
        let started = Instant::now();
        if let Err(err) = world.step_frame() {
            eprintln!("terrarium step failed: {err}");
            return ExitCode::FAILURE;
        }
        frame_idx += 1;
        let snapshot = world.snapshot();

        if cli.no_render {
            if frame_idx == 1 || frame_idx % cli.summary_every == 0 {
                print_summary(frame_idx, &world, &snapshot);
            }
        } else if let Err(err) = render_terminal(&world, &snapshot, frame_idx) {
            eprintln!("render failed: {err}");
            return ExitCode::FAILURE;
        }

        if let Some(max_frames) = cli.frames {
            if frame_idx >= max_frames {
                break;
            }
        }

        if let Some(target) = frame_budget {
            let elapsed = started.elapsed();
            if elapsed < target {
                thread::sleep(target - elapsed);
            }
        }
    }

    ExitCode::SUCCESS
}
