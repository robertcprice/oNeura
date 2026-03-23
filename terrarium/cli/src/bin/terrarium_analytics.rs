//! Live Evolution Analytics Dashboard.
//!
//! Real-time windowed display showing 4-pane evolution analytics:
//!   Top-left:     Convergence curve (best/mean/worst fitness per generation)
//!   Top-right:    Population diversity sparkline
//!   Bottom-left:  Parameter heatmap (genome params as colored bars)
//!   Bottom-right: Multi-objective radar chart (8 objectives)
//!
//! Usage:
//!   terrarium_analytics --population 8 --generations 10 --frames 50 --lite
//!   terrarium_analytics --width 1024 --height 768 --fitness biomass

use minifb::{Key, Window, WindowOptions};
use oneura_core::terrarium::telemetry_to_csv;
use oneura_core::terrarium::{
    evolve, telemetry_from_result, EvolutionConfig, FitnessConfig, FitnessObjective,
    GenerationTelemetry, GenomeConstraints, SearchStrategy,
};
use std::env;
use std::fs;
use std::time::Instant;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Args {
    population: usize,
    generations: usize,
    frames: usize,
    fitness: FitnessObjective,
    lite: bool,
    seed: u64,
    width: usize,
    height: usize,
    output: Option<String>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            population: 8,
            generations: 10,
            frames: 50,
            fitness: FitnessObjective::MaxBiomass,
            lite: true,
            seed: 42,
            width: 960,
            height: 640,
            output: None,
        }
    }
}

fn parse_args() -> Args {
    let mut args = Args::default();
    let mut i = 1;
    let argv: Vec<String> = env::args().collect();
    while i < argv.len() {
        match argv[i].as_str() {
            "--population" | "-p" => {
                args.population = argv
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(args.population);
                i += 1;
            }
            "--generations" | "-g" => {
                args.generations = argv
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(args.generations);
                i += 1;
            }
            "--frames" | "-f" => {
                args.frames = argv
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(args.frames);
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
                    _ => FitnessObjective::MaxBiomass,
                };
                i += 1;
            }
            "--lite" => args.lite = true,
            "--no-lite" => args.lite = false,
            "--seed" | "-s" => {
                args.seed = argv
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(args.seed);
                i += 1;
            }
            "--width" => {
                args.width = argv
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(args.width);
                i += 1;
            }
            "--height" => {
                args.height = argv
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(args.height);
                i += 1;
            }
            "--output" | "-o" => {
                args.output = argv.get(i + 1).cloned();
                i += 1;
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
    println!("Terrarium Live Evolution Analytics Dashboard");
    println!();
    println!("Usage: terrarium_analytics [options]");
    println!();
    println!("Options:");
    println!("  --population <N>   Population size (default: 8)");
    println!("  --generations <N>  Generations to run (default: 10)");
    println!("  --frames <N>       Frames per world (default: 50)");
    println!("  --fitness <OBJ>    Fitness: biomass, biodiversity, stability, carbon, fruit, microbial, fly");
    println!("  --lite             Use lite worlds (default: true)");
    println!("  --seed <N>         Random seed (default: 42)");
    println!("  --width <N>        Window width (default: 960)");
    println!("  --height <N>       Window height (default: 640)");
    println!("  --output <PATH>    Export telemetry CSV on completion");
    println!("  --help, -h         Show this help");
}

// ---------------------------------------------------------------------------
// Drawing helpers
// ---------------------------------------------------------------------------

fn rgb(r: u8, g: u8, b: u8) -> u32 {
    (r as u32) << 16 | (g as u32) << 8 | b as u32
}

fn lerp_color(a: u32, b: u32, t: f32) -> u32 {
    let t = t.clamp(0.0, 1.0);
    let ar = ((a >> 16) & 0xFF) as f32;
    let ag = ((a >> 8) & 0xFF) as f32;
    let ab = (a & 0xFF) as f32;
    let br = ((b >> 16) & 0xFF) as f32;
    let bg = ((b >> 8) & 0xFF) as f32;
    let bb = (b & 0xFF) as f32;
    let r = (ar + (br - ar) * t) as u8;
    let g = (ag + (bg - ag) * t) as u8;
    let bl = (ab + (bb - ab) * t) as u8;
    rgb(r, g, bl)
}

fn fill_rect(buf: &mut [u32], bw: usize, x: usize, y: usize, w: usize, h: usize, color: u32) {
    let bh = buf.len() / bw;
    for dy in 0..h {
        let py = y + dy;
        if py >= bh {
            break;
        }
        for dx in 0..w {
            let px = x + dx;
            if px >= bw {
                break;
            }
            buf[py * bw + px] = color;
        }
    }
}

fn draw_line(
    buf: &mut [u32],
    bw: usize,
    bh: usize,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    color: u32,
) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut cx = x0;
    let mut cy = y0;
    loop {
        if cx >= 0 && cy >= 0 && (cx as usize) < bw && (cy as usize) < bh {
            buf[cy as usize * bw + cx as usize] = color;
        }
        if cx == x1 && cy == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            cx += sx;
        }
        if e2 <= dx {
            err += dx;
            cy += sy;
        }
    }
}

fn draw_text_tiny(
    buf: &mut [u32],
    bw: usize,
    bh: usize,
    x: usize,
    y: usize,
    text: &str,
    color: u32,
) {
    // Simple 3x5 digit/letter rendering for labels
    let chars: Vec<char> = text.chars().collect();
    for (ci, ch) in chars.iter().enumerate() {
        let glyph = match ch {
            '0'..='9' => digit_glyph((*ch as u8 - b'0') as usize),
            '.' => DOT_GLYPH,
            '-' => DASH_GLYPH,
            '%' => PERCENT_GLYPH,
            _ => SPACE_GLYPH,
        };
        for row in 0..5 {
            for col in 0..3 {
                if glyph[row] & (1 << (2 - col)) != 0 {
                    let px = x + ci * 4 + col;
                    let py = y + row;
                    if px < bw && py < bh {
                        buf[py * bw + px] = color;
                    }
                }
            }
        }
    }
}

const SPACE_GLYPH: [u8; 5] = [0, 0, 0, 0, 0];
const DOT_GLYPH: [u8; 5] = [0, 0, 0, 0, 0b010];
const DASH_GLYPH: [u8; 5] = [0, 0, 0b111, 0, 0];
const PERCENT_GLYPH: [u8; 5] = [0b101, 0b001, 0b010, 0b100, 0b101];

fn digit_glyph(d: usize) -> [u8; 5] {
    match d {
        0 => [0b111, 0b101, 0b101, 0b101, 0b111],
        1 => [0b010, 0b110, 0b010, 0b010, 0b111],
        2 => [0b111, 0b001, 0b111, 0b100, 0b111],
        3 => [0b111, 0b001, 0b111, 0b001, 0b111],
        4 => [0b101, 0b101, 0b111, 0b001, 0b001],
        5 => [0b111, 0b100, 0b111, 0b001, 0b111],
        6 => [0b111, 0b100, 0b111, 0b101, 0b111],
        7 => [0b111, 0b001, 0b001, 0b001, 0b001],
        8 => [0b111, 0b101, 0b111, 0b101, 0b111],
        9 => [0b111, 0b101, 0b111, 0b001, 0b111],
        _ => SPACE_GLYPH,
    }
}

// ---------------------------------------------------------------------------
// Pane renderers
// ---------------------------------------------------------------------------

const BG: u32 = 0x0d0f12;
const BORDER: u32 = 0x2a2f3a;
const LABEL: u32 = 0x8890a0;
const BEST_COLOR: u32 = 0x4ecdc4;
const MEAN_COLOR: u32 = 0x5b9bd5;
const WORST_COLOR: u32 = 0xe74c3c;
const DIVERSITY_COLOR: u32 = 0xd46cb0;

/// Top-left: Convergence curve
fn draw_convergence(
    buf: &mut [u32],
    bw: usize,
    bh: usize,
    ox: usize,
    oy: usize,
    pw: usize,
    ph: usize,
    data: &[GenerationTelemetry],
) {
    fill_rect(buf, bw, ox, oy, pw, ph, BG);
    // Border
    for x in ox..ox + pw {
        if oy < bh {
            buf[oy * bw + x.min(bw - 1)] = BORDER;
        }
        if oy + ph - 1 < bh {
            buf[(oy + ph - 1) * bw + x.min(bw - 1)] = BORDER;
        }
    }
    for y in oy..oy + ph {
        if y < bh {
            buf[y * bw + ox.min(bw - 1)] = BORDER;
            buf[y * bw + (ox + pw - 1).min(bw - 1)] = BORDER;
        }
    }

    draw_text_tiny(buf, bw, bh, ox + 4, oy + 4, "CONVERGENCE", LABEL);

    if data.is_empty() {
        return;
    }

    let margin = 20;
    let plot_x = ox + margin;
    let plot_y = oy + 16;
    let plot_w = pw.saturating_sub(margin * 2);
    let plot_h = ph.saturating_sub(36);

    let mut max_f = f32::NEG_INFINITY;
    let mut min_f = f32::INFINITY;
    for d in data {
        max_f = max_f.max(d.best_fitness);
        min_f = min_f.min(d.worst_fitness);
    }
    let range = (max_f - min_f).max(1e-6);

    // Draw 3 curves
    for (series, color) in [(0, BEST_COLOR), (1, MEAN_COLOR), (2, WORST_COLOR)] {
        let mut prev: Option<(i32, i32)> = None;
        for (i, d) in data.iter().enumerate() {
            let val = match series {
                0 => d.best_fitness,
                1 => d.mean_fitness,
                _ => d.worst_fitness,
            };
            let fx = plot_x as i32 + (i as f32 / data.len().max(1) as f32 * plot_w as f32) as i32;
            let fy = (plot_y + plot_h) as i32 - ((val - min_f) / range * plot_h as f32) as i32;
            if let Some((px, py)) = prev {
                draw_line(buf, bw, bh, px, py, fx, fy, color);
            }
            prev = Some((fx, fy));
        }
    }

    // Y-axis labels
    let max_str = format!("{:.1}", max_f);
    let min_str = format!("{:.1}", min_f);
    draw_text_tiny(buf, bw, bh, ox + 2, plot_y, &max_str, LABEL);
    draw_text_tiny(buf, bw, bh, ox + 2, plot_y + plot_h - 6, &min_str, LABEL);
}

/// Top-right: Diversity sparkline
fn draw_diversity(
    buf: &mut [u32],
    bw: usize,
    bh: usize,
    ox: usize,
    oy: usize,
    pw: usize,
    ph: usize,
    data: &[GenerationTelemetry],
) {
    fill_rect(buf, bw, ox, oy, pw, ph, BG);
    for x in ox..ox + pw {
        if oy < bh {
            buf[oy * bw + x.min(bw - 1)] = BORDER;
        }
        if oy + ph - 1 < bh {
            buf[(oy + ph - 1) * bw + x.min(bw - 1)] = BORDER;
        }
    }
    for y in oy..oy + ph {
        if y < bh {
            buf[y * bw + ox.min(bw - 1)] = BORDER;
            buf[y * bw + (ox + pw - 1).min(bw - 1)] = BORDER;
        }
    }

    draw_text_tiny(buf, bw, bh, ox + 4, oy + 4, "DIVERSITY", LABEL);

    if data.is_empty() {
        return;
    }

    let margin = 20;
    let plot_x = ox + margin;
    let plot_y = oy + 16;
    let plot_w = pw.saturating_sub(margin * 2);
    let plot_h = ph.saturating_sub(36);

    let diversities: Vec<f32> = data.iter().map(|d| d.population_diversity).collect();
    let max_d = diversities
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max)
        .max(0.01);

    // Filled area under curve
    for (i, &div) in diversities.iter().enumerate() {
        let fx = plot_x + (i as f32 / diversities.len().max(1) as f32 * plot_w as f32) as usize;
        let bar_h = (div / max_d * plot_h as f32) as usize;
        let bar_y = plot_y + plot_h - bar_h;
        fill_rect(
            buf,
            bw,
            fx,
            bar_y,
            2.max(plot_w / diversities.len().max(1)),
            bar_h,
            DIVERSITY_COLOR,
        );
    }
}

/// Bottom-left: Parameter heatmap
fn draw_heatmap(
    buf: &mut [u32],
    bw: usize,
    bh: usize,
    ox: usize,
    oy: usize,
    pw: usize,
    ph: usize,
    data: &[GenerationTelemetry],
) {
    fill_rect(buf, bw, ox, oy, pw, ph, BG);
    for x in ox..ox + pw {
        if oy < bh {
            buf[oy * bw + x.min(bw - 1)] = BORDER;
        }
        if oy + ph - 1 < bh {
            buf[(oy + ph - 1) * bw + x.min(bw - 1)] = BORDER;
        }
    }
    for y in oy..oy + ph {
        if y < bh {
            buf[y * bw + ox.min(bw - 1)] = BORDER;
            buf[y * bw + (ox + pw - 1).min(bw - 1)] = BORDER;
        }
    }

    draw_text_tiny(buf, bw, bh, ox + 4, oy + 4, "GENOME HEATMAP", LABEL);

    if data.is_empty() {
        return;
    }
    let last = data.last().unwrap();
    let params = &last.best_genome_params;
    if params.is_empty() {
        return;
    }

    let margin = 8;
    let bar_area_y = oy + 16;
    let bar_area_h = ph.saturating_sub(24);
    let bar_w = (pw.saturating_sub(margin * 2)) / params.len().max(1);

    let cold = rgb(20, 40, 80); // blue
    let hot = rgb(220, 60, 40); // red

    for (i, &val) in params.iter().enumerate() {
        let bx = ox + margin + i * bar_w;
        let bar_h = (val.clamp(0.0, 1.0) * bar_area_h as f32) as usize;
        let by = bar_area_y + bar_area_h - bar_h;
        let color = lerp_color(cold, hot, val);
        fill_rect(buf, bw, bx, by, bar_w.saturating_sub(1), bar_h, color);
    }
}

/// Bottom-right: Radar chart placeholder (shows objective bars)
fn draw_radar(
    buf: &mut [u32],
    bw: usize,
    bh: usize,
    ox: usize,
    oy: usize,
    pw: usize,
    ph: usize,
    data: &[GenerationTelemetry],
) {
    fill_rect(buf, bw, ox, oy, pw, ph, BG);
    for x in ox..ox + pw {
        if oy < bh {
            buf[oy * bw + x.min(bw - 1)] = BORDER;
        }
        if oy + ph - 1 < bh {
            buf[(oy + ph - 1) * bw + x.min(bw - 1)] = BORDER;
        }
    }
    for y in oy..oy + ph {
        if y < bh {
            buf[y * bw + ox.min(bw - 1)] = BORDER;
            buf[y * bw + (ox + pw - 1).min(bw - 1)] = BORDER;
        }
    }

    draw_text_tiny(buf, bw, bh, ox + 4, oy + 4, "OBJECTIVES", LABEL);

    if data.is_empty() {
        return;
    }

    // Show generation-over-generation improvement
    let margin = 16;
    let plot_y = oy + 20;
    let plot_h = ph.saturating_sub(36);

    // Radar as concentric polygon points — simplified to horizontal bars for fitness progression
    let n = data.len();
    let bar_h = plot_h / n.max(1);

    for (i, d) in data.iter().enumerate() {
        let by = plot_y + i * bar_h;
        let fitness_norm = d.best_fitness
            / data
                .iter()
                .map(|d| d.best_fitness)
                .fold(f32::NEG_INFINITY, f32::max)
                .max(0.01);
        let bar_w = (fitness_norm * (pw - margin * 2) as f32) as usize;
        let t = i as f32 / n.max(1) as f32;
        let color = lerp_color(rgb(60, 60, 120), BEST_COLOR, t);
        fill_rect(
            buf,
            bw,
            ox + margin,
            by,
            bar_w,
            bar_h.saturating_sub(1),
            color,
        );
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args = parse_args();

    eprintln!("=== Terrarium Live Evolution Analytics ===");
    eprintln!(
        "Pop: {}  Gens: {}  Frames: {}  Seed: {}  Lite: {}",
        args.population, args.generations, args.frames, args.seed, args.lite
    );

    // Run evolution first, collect telemetry
    let config = EvolutionConfig {
        population_size: args.population,
        generations: args.generations,
        frames_per_world: args.frames,
        master_seed: args.seed,
        lite: args.lite,
        fitness: FitnessConfig {
            primary: args.fitness,
            snapshot_interval: 10,
        },
        strategy: SearchStrategy::Evolutionary {
            tournament_size: 3,
            mutation_rate: 0.15,
            crossover_rate: 0.7,
            elitism: 2,
        },
        constraints: GenomeConstraints::default(),
        thread_count: None,
    };

    eprintln!("Running evolution...");
    let start = Instant::now();
    let result = evolve(config).expect("Evolution failed");
    let elapsed = start.elapsed();
    eprintln!(
        "Evolution complete in {:.2}s — best fitness: {:.4}",
        elapsed.as_secs_f64(),
        result.global_best_fitness
    );

    let telemetry = telemetry_from_result(&result, Some("analytics"));

    // Export CSV if requested
    if let Some(ref path) = args.output {
        let csv = telemetry_to_csv(&telemetry);
        if let Err(e) = fs::write(path, &csv) {
            eprintln!("Error writing CSV: {}", e);
        } else {
            eprintln!("Telemetry exported to: {}", path);
        }
    }

    // Open window
    let w = args.width;
    let h = args.height;
    let mut buf = vec![BG; w * h];

    let mut window = Window::new(
        "oNeura Evolution Analytics",
        w,
        h,
        WindowOptions {
            resize: false,
            ..WindowOptions::default()
        },
    )
    .expect("Failed to create window");

    window.set_target_fps(30);

    // Progressive reveal: show generations one at a time
    let mut visible_gens = 0usize;
    let mut last_advance = Instant::now();
    let advance_interval = std::time::Duration::from_millis(300);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        // Advance visible generations progressively
        if visible_gens < telemetry.len() && last_advance.elapsed() >= advance_interval {
            visible_gens += 1;
            last_advance = Instant::now();
        }

        // If user presses Space, show all
        if window.is_key_pressed(Key::Space, minifb::KeyRepeat::No) {
            visible_gens = telemetry.len();
        }

        let visible = &telemetry[..visible_gens];
        let half_w = w / 2;
        let half_h = h / 2;

        // Draw 4 panes
        draw_convergence(&mut buf, w, h, 0, 0, half_w, half_h, visible);
        draw_diversity(&mut buf, w, h, half_w, 0, half_w, half_h, visible);
        draw_heatmap(&mut buf, w, h, 0, half_h, half_w, half_h, visible);
        draw_radar(&mut buf, w, h, half_w, half_h, half_w, half_h, visible);

        // Status bar at bottom
        let status = format!(
            "Gen {}/{} | Best: {:.2} | Space=show all | Esc=quit",
            visible_gens,
            telemetry.len(),
            visible.last().map(|d| d.best_fitness).unwrap_or(0.0)
        );
        draw_text_tiny(&mut buf, w, h, 4, h - 8, &status, LABEL);

        window.update_with_buffer(&buf, w, h).unwrap();
    }

    eprintln!("Dashboard closed.");
}
