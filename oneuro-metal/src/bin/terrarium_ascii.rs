//! 3D Isometric ASCII Terrarium Renderer
//!
//! Real-time 3D visualization of the terrarium using Unicode block characters
//! and ANSI truecolor. Renders terrain height, plants, water, flies, and soil
//! chemistry as a pseudo-isometric heightmap at 15-30 FPS in the terminal.
//!
//! Usage:
//!   cargo run --profile fast --no-default-features --bin terrarium_ascii -- [OPTIONS]
//!
//! Options:
//!   --seed <n>       World seed (default: 7)
//!   --fps <n>        Target framerate (default: 15)
//!   --frames <n>     Quit after N frames (default: infinite)
//!   --mode <name>    View mode: iso, top, split (default: iso)
//!   --no-color       Disable ANSI colors

use oneuro_metal::{TerrariumWorld, TerrariumWorldConfig, TerrariumWorldSnapshot};
use std::env;
use std::io::{self, Write};
use std::process::ExitCode;
use std::thread;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// ANSI Color Helpers
// ---------------------------------------------------------------------------

fn fg(r: u8, g: u8, b: u8) -> String {
    format!("\x1b[38;2;{r};{g};{b}m")
}

fn bg(r: u8, g: u8, b: u8) -> String {
    format!("\x1b[48;2;{r};{g};{b}m")
}

const RESET: &str = "\x1b[0m";

fn lerp_color(a: (u8, u8, u8), b: (u8, u8, u8), t: f32) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0);
    (
        (a.0 as f32 + (b.0 as f32 - a.0 as f32) * t) as u8,
        (a.1 as f32 + (b.1 as f32 - a.1 as f32) * t) as u8,
        (a.2 as f32 + (b.2 as f32 - a.2 as f32) * t) as u8,
    )
}

/// Map moisture to soil color (brown gradient).
fn soil_color(moisture: f32, organic: f32) -> (u8, u8, u8) {
    let dry = (180, 140, 80);     // sandy
    let wet = (80, 60, 30);       // dark earth
    let rich = (60, 90, 40);      // organic-rich
    let base = lerp_color(dry, wet, moisture.clamp(0.0, 1.0) * 2.0);
    lerp_color(base, rich, organic.clamp(0.0, 1.0) * 3.0)
}

/// Map canopy density to plant color.
fn plant_color(canopy: f32, vitality: f32) -> (u8, u8, u8) {
    let sparse = (100, 160, 60);
    let dense = (30, 120, 30);
    let stressed = (160, 150, 50);
    let base = lerp_color(sparse, dense, canopy.clamp(0.0, 1.0));
    lerp_color(base, stressed, (1.0 - vitality).clamp(0.0, 1.0))
}

// ---------------------------------------------------------------------------
// Screen Buffer
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Cell {
    ch: char,
    fg: (u8, u8, u8),
    bg: (u8, u8, u8),
}

impl Default for Cell {
    fn default() -> Self {
        Self { ch: ' ', fg: (200, 200, 200), bg: (15, 15, 25) }
    }
}

struct ScreenBuffer {
    width: usize,
    height: usize,
    cells: Vec<Cell>,
}

impl ScreenBuffer {
    fn new(width: usize, height: usize) -> Self {
        Self { width, height, cells: vec![Cell::default(); width * height] }
    }

    fn clear(&mut self) {
        for cell in &mut self.cells {
            *cell = Cell::default();
        }
    }

    fn set(&mut self, x: usize, y: usize, ch: char, fg_c: (u8, u8, u8), bg_c: (u8, u8, u8)) {
        if x < self.width && y < self.height {
            let idx = y * self.width + x;
            self.cells[idx] = Cell { ch, fg: fg_c, bg: bg_c };
        }
    }

    fn render(&self, use_color: bool) -> String {
        let mut out = String::with_capacity(self.width * self.height * 20);
        out.push_str("\x1b[H"); // cursor home
        for y in 0..self.height {
            for x in 0..self.width {
                let cell = &self.cells[y * self.width + x];
                if use_color {
                    out.push_str(&fg(cell.fg.0, cell.fg.1, cell.fg.2));
                    out.push_str(&bg(cell.bg.0, cell.bg.1, cell.bg.2));
                }
                out.push(cell.ch);
            }
            if use_color {
                out.push_str(RESET);
            }
            out.push('\n');
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Isometric Renderer
// ---------------------------------------------------------------------------

fn render_isometric(
    buf: &mut ScreenBuffer,
    world: &TerrariumWorld,
    snapshot: &TerrariumWorldSnapshot,
    frame_idx: usize,
) {
    buf.clear();
    let gw = world.config.width;
    let gh = world.config.height;
    let moisture = world.moisture_field();

    // Screen center offset
    let ox = buf.width / 2;
    let oy = 3; // top margin for header

    // Header
    let header = format!(
        " oNeuro Terrarium 3D | F:{} | {} | {:.0}C | plants:{} flies:{} fruit:{}",
        frame_idx, world.time_label(), snapshot.temperature,
        snapshot.plants, snapshot.flies, snapshot.fruits,
    );
    for (i, ch) in header.chars().enumerate() {
        if i < buf.width {
            buf.set(i, 0, ch, (0, 220, 255), (20, 20, 40));
        }
    }
    let sub = format!(
        " moisture:{:.2} microbes:{:.3} CO2:{:.4} O2:{:.2} cells:{:.0}",
        snapshot.mean_soil_moisture, snapshot.mean_microbes,
        snapshot.mean_atmospheric_co2, snapshot.mean_atmospheric_o2,
        snapshot.total_plant_cells,
    );
    for (i, ch) in sub.chars().enumerate() {
        if i < buf.width {
            buf.set(i, 1, ch, (180, 180, 200), (20, 20, 40));
        }
    }

    // Draw terrain from back to front (painter's algorithm)
    for gy in 0..gh {
        for gx in 0..gw {
            // Isometric projection
            let iso_x = (gx as isize - gy as isize) * 2 + ox as isize;
            let iso_y = (gx as isize + gy as isize) / 2 + oy as isize;

            // Terrain height from moisture + organic matter
            let m_idx = gy * gw + gx;
            let m = if m_idx < moisture.len() { moisture[m_idx] } else { 0.3 };
            let height = (m * 3.0).clamp(0.0, 4.0) as isize;

            let color = soil_color(m, m * 0.5);

            // Draw terrain column
            for h in 0..=height {
                let sy = iso_y - h;
                let sx = iso_x;
                if sx >= 0 && sy >= 0 {
                    let sxu = sx as usize;
                    let syu = sy as usize;
                    // Top face uses lighter shade
                    let face_color = if h == height {
                        lerp_color(color, (255, 255, 255), 0.15)
                    } else {
                        lerp_color(color, (0, 0, 0), 0.2)
                    };
                    let ch = if h == height { '\u{2593}' } else { '\u{2588}' }; // ▓ or █
                    for dx in 0..3 {
                        buf.set(sxu + dx, syu, ch, face_color, (10, 10, 20));
                    }
                }
            }
        }
    }

    // Draw water sources
    for water in &world.waters {
        if water.alive {
            let iso_x = (water.x as isize - water.y as isize) * 2 + ox as isize;
            let iso_y = (water.x as isize + water.y as isize) / 2 + oy as isize;
            if iso_x >= 0 && iso_y >= 0 {
                let sx = iso_x as usize;
                let sy = iso_y as usize;
                let wave = if frame_idx % 4 < 2 { '\u{2248}' } else { '~' }; // ≈ or ~
                for dx in 0..4 {
                    buf.set(sx + dx, sy, wave, (60, 140, 255), (20, 60, 120));
                }
                // Water depth indication
                if sy + 1 < buf.height {
                    buf.set(sx + 1, sy + 1, '\u{2592}', (30, 80, 180), (15, 40, 80)); // ▒
                }
            }
        }
    }

    // Draw plants as vertical columns
    for plant in &world.plants {
        let iso_x = (plant.x as isize - plant.y as isize) * 2 + ox as isize;
        let iso_y = (plant.x as isize + plant.y as isize) / 2 + oy as isize;
        if iso_x >= 0 && iso_y >= 0 {
            let sx = iso_x as usize;
            let sy = iso_y as usize;
            let cells = plant.cellular.total_cells();
            let plant_h = (cells * 0.02).clamp(1.0, 6.0) as usize;
            let vitality = plant.cellular.vitality();
            let pc = plant_color(cells * 0.01, vitality);

            // Trunk
            for h in 0..plant_h {
                if sy >= h + 1 {
                    buf.set(sx + 1, sy - h - 1, '\u{2503}', (120, 80, 40), (10, 10, 20)); // ┃
                }
            }
            // Canopy
            if sy >= plant_h + 1 {
                let canopy_y = sy - plant_h - 1;
                buf.set(sx, canopy_y, '\u{2663}', pc, (10, 10, 20));     // ♣
                buf.set(sx + 1, canopy_y, '\u{2663}', pc, (10, 10, 20));
                buf.set(sx + 2, canopy_y, '\u{2663}', pc, (10, 10, 20));
                if canopy_y > 0 {
                    buf.set(sx + 1, canopy_y - 1, '\u{25B2}', // ▲
                        lerp_color(pc, (200, 255, 100), 0.3), (10, 10, 20));
                }
            }
        }
    }

    // Draw fruits
    for fruit in &world.fruits {
        if fruit.source.alive && fruit.source.sugar_content > 0.01 {
            let iso_x = (fruit.source.x as isize - fruit.source.y as isize) * 2 + ox as isize;
            let iso_y = (fruit.source.x as isize + fruit.source.y as isize) / 2 + oy as isize;
            if iso_x >= 0 && iso_y >= 0 {
                let ripe = fruit.source.ripeness.clamp(0.0, 1.0);
                let c = lerp_color((100, 200, 50), (255, 80, 30), ripe);
                buf.set(iso_x as usize + 1, iso_y as usize, '\u{25CF}', c, (10, 10, 20)); // ●
            }
        }
    }

    // Draw flies
    for fly in &world.flies {
        let body = fly.body_state();
        let gx = body.x.round().clamp(0.0, (gw - 1) as f32) as isize;
        let gy = body.y.round().clamp(0.0, (gh - 1) as f32) as isize;
        let iso_x = (gx - gy) * 2 + ox as isize;
        let iso_y = (gx + gy) / 2 + oy as isize;
        let altitude = body.z.clamp(0.0, 5.0) as isize;

        if iso_x >= 0 && iso_y >= altitude {
            let sy = (iso_y - altitude) as usize;
            let sx = iso_x as usize;
            let (ch, color) = if body.is_flying {
                // Animated wing flap
                let wing = if frame_idx % 6 < 3 { '\u{2736}' } else { '\u{2734}' }; // ✶ or ✴
                (wing, (255, 230, 50))
            } else {
                ('\u{25C6}', (255, 200, 80)) // ◆
            };
            buf.set(sx + 1, sy, ch, color, (10, 10, 20));
        }
    }

    // Legend bar at bottom
    let legend_y = buf.height - 2;
    let legend = " \u{2663}=plant  \u{25CF}=fruit  ~=water  \u{25C6}=fly  \u{2593}=terrain";
    for (i, ch) in legend.chars().enumerate() {
        if i < buf.width {
            buf.set(i, legend_y, ch, (140, 140, 160), (20, 20, 35));
        }
    }
}

// ---------------------------------------------------------------------------
// Enhanced Top-Down Renderer
// ---------------------------------------------------------------------------

fn render_topdown_color(
    buf: &mut ScreenBuffer,
    world: &TerrariumWorld,
    snapshot: &TerrariumWorldSnapshot,
    frame_idx: usize,
) {
    buf.clear();
    let gw = world.config.width;
    let gh = world.config.height;
    let moisture = world.moisture_field();

    // Header
    let header = format!(
        " Terrarium Top-Down | F:{} | {} | {:.0}C | P:{} Fl:{} Fr:{}",
        frame_idx, world.time_label(), snapshot.temperature,
        snapshot.plants, snapshot.flies, snapshot.fruits,
    );
    for (i, ch) in header.chars().enumerate() {
        if i < buf.width {
            buf.set(i, 0, ch, (0, 220, 255), (20, 20, 40));
        }
    }

    let ox = (buf.width.saturating_sub(gw * 2)) / 2;
    let oy = 2;

    // Terrain base layer
    for gy in 0..gh {
        for gx in 0..gw {
            let m_idx = gy * gw + gx;
            let m = if m_idx < moisture.len() { moisture[m_idx] } else { 0.3 };
            let color = soil_color(m, m * 0.5);
            let ch = if m > 0.5 { '\u{2593}' } else if m > 0.3 { '\u{2592}' } else { '\u{2591}' };
            buf.set(ox + gx * 2, oy + gy, ch, color, (10, 10, 20));
            buf.set(ox + gx * 2 + 1, oy + gy, ch, color, (10, 10, 20));
        }
    }

    // Water overlay
    for water in &world.waters {
        if water.alive {
            let wave = if frame_idx % 4 < 2 { '\u{2248}' } else { '~' };
            for dy in 0..3_usize {
                for dx in 0..3_usize {
                    let wx = water.x.saturating_add(dx).saturating_sub(1);
                    let wy = water.y.saturating_add(dy).saturating_sub(1);
                    if wx < gw && wy < gh {
                        buf.set(ox + wx * 2, oy + wy, wave, (80, 160, 255), (20, 50, 100));
                        buf.set(ox + wx * 2 + 1, oy + wy, wave, (80, 160, 255), (20, 50, 100));
                    }
                }
            }
        }
    }

    // Plants
    for plant in &world.plants {
        let cells = plant.cellular.total_cells();
        let vitality = plant.cellular.vitality();
        let pc = plant_color(cells * 0.01, vitality);
        let ch = if cells > 50.0 { '\u{2663}' } else { '\u{2022}' }; // ♣ or •
        buf.set(ox + plant.x * 2, oy + plant.y, ch, pc, (10, 30, 10));
        buf.set(ox + plant.x * 2 + 1, oy + plant.y, ch, pc, (10, 30, 10));
    }

    // Fruits
    for fruit in &world.fruits {
        if fruit.source.alive && fruit.source.sugar_content > 0.01 {
            let ripe = fruit.source.ripeness.clamp(0.0, 1.0);
            let c = lerp_color((100, 200, 50), (255, 80, 30), ripe);
            buf.set(ox + fruit.source.x * 2, oy + fruit.source.y, '\u{25CF}', c, (10, 10, 20));
        }
    }

    // Flies
    for fly in &world.flies {
        let body = fly.body_state();
        let gx = body.x.round().clamp(0.0, (gw - 1) as f32) as usize;
        let gy = body.y.round().clamp(0.0, (gh - 1) as f32) as usize;
        let (ch, color) = if body.is_flying {
            (if frame_idx % 4 < 2 { '\u{2736}' } else { '\u{2734}' }, (255, 230, 50))
        } else {
            ('\u{25C6}', (255, 200, 80))
        };
        buf.set(ox + gx * 2, oy + gy, ch, color, (10, 10, 20));
    }

    // Stats panel below
    let stats_y = oy + gh + 1;
    let stats = [
        format!(" Moisture:{:.3} Deep:{:.3} Glucose:{:.3}", snapshot.mean_soil_moisture, snapshot.mean_deep_moisture, snapshot.mean_soil_glucose),
        format!(" Microbes:{:.3} Symbionts:{:.3} ATP:{:.3}", snapshot.mean_microbes, snapshot.mean_symbionts, snapshot.mean_soil_atp_flux),
        format!(" CO2:{:.4} O2:{:.4} Cells:{:.0} Energy:{:.2}", snapshot.mean_atmospheric_co2, snapshot.mean_atmospheric_o2, snapshot.total_plant_cells, snapshot.mean_cell_energy),
        format!(" FlyEnergy:{:.1} FlyEC:{:.2} Seeds:{} Events:{}", snapshot.avg_fly_energy, snapshot.avg_fly_energy_charge, snapshot.seeds, snapshot.ecology_event_count),
    ];
    for (i, line) in stats.iter().enumerate() {
        for (j, ch) in line.chars().enumerate() {
            if j < buf.width && stats_y + i < buf.height {
                buf.set(j, stats_y + i, ch, (160, 200, 160), (15, 25, 15));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Cli {
    seed: u64,
    fps: u64,
    frames: Option<usize>,
    mode: ViewMode,
    use_color: bool,
    cpu_substrate: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ViewMode {
    Isometric,
    TopDown,
    Split,
}

impl Default for Cli {
    fn default() -> Self {
        Self {
            seed: 7, fps: 15, frames: None,
            mode: ViewMode::Isometric, use_color: true, cpu_substrate: false,
        }
    }
}

fn print_usage() {
    eprintln!(
        "Usage: terrarium_ascii [--seed N] [--fps N] [--frames N] [--mode iso|top|split] [--no-color] [--cpu-substrate]"
    );
}

fn parse_args() -> Result<Cli, String> {
    let mut cli = Cli::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--seed" => cli.seed = args.next().ok_or("missing --seed")?.parse().map_err(|_| "bad --seed")?,
            "--fps" => cli.fps = args.next().ok_or("missing --fps")?.parse().map_err(|_| "bad --fps")?,
            "--frames" => cli.frames = Some(args.next().ok_or("missing --frames")?.parse().map_err(|_| "bad --frames")?),
            "--mode" => {
                cli.mode = match args.next().ok_or("missing --mode")?.as_str() {
                    "iso" | "isometric" => ViewMode::Isometric,
                    "top" | "topdown" => ViewMode::TopDown,
                    "split" => ViewMode::Split,
                    other => return Err(format!("unknown mode: {other}")),
                };
            }
            "--no-color" => cli.use_color = false,
            "--cpu-substrate" => cli.cpu_substrate = true,
            "--help" | "-h" => { print_usage(); std::process::exit(0); }
            other => return Err(format!("unknown arg: {other}")),
        }
    }
    Ok(cli)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> ExitCode {
    let cli = match parse_args() {
        Ok(c) => c,
        Err(e) => { eprintln!("{e}"); print_usage(); return ExitCode::FAILURE; }
    };

    let mut world = match TerrariumWorld::demo(cli.seed, !cli.cpu_substrate) {
        Ok(w) => w,
        Err(e) => { eprintln!("failed to build terrarium: {e}"); return ExitCode::FAILURE; }
    };

    // Determine terminal size (fallback to 120x40)
    let term_width = 120;
    let term_height = 45;

    let mut buf = ScreenBuffer::new(term_width, term_height);
    let _buf2 = if cli.mode == ViewMode::Split {
        Some(ScreenBuffer::new(term_width, term_height))
    } else {
        None
    };

    let frame_budget = if cli.fps > 0 {
        Some(Duration::from_secs_f64(1.0 / cli.fps as f64))
    } else {
        None
    };

    // Hide cursor, clear screen
    print!("\x1b[?25l\x1b[2J");
    let _ = io::stdout().flush();

    let mut frame_idx = 0usize;
    let mut last_fps = 0.0f32;
    let mut fps_timer = Instant::now();
    let mut fps_frames = 0usize;

    loop {
        let started = Instant::now();
        if let Err(e) = world.step_frame() {
            eprintln!("step failed: {e}");
            break;
        }
        frame_idx += 1;
        let snapshot = world.snapshot();

        match cli.mode {
            ViewMode::Isometric => {
                render_isometric(&mut buf, &world, &snapshot, frame_idx);
            }
            ViewMode::TopDown => {
                render_topdown_color(&mut buf, &world, &snapshot, frame_idx);
            }
            ViewMode::Split => {
                // Left half: isometric, right half: top-down
                let half_w = term_width / 2;
                let mut left = ScreenBuffer::new(half_w, term_height);
                let mut right = ScreenBuffer::new(half_w, term_height);
                render_isometric(&mut left, &world, &snapshot, frame_idx);
                render_topdown_color(&mut right, &world, &snapshot, frame_idx);
                // Merge into main buffer
                buf.clear();
                for y in 0..term_height {
                    for x in 0..half_w {
                        let lc = &left.cells[y * half_w + x];
                        buf.set(x, y, lc.ch, lc.fg, lc.bg);
                        if x < right.width {
                            let rc = &right.cells[y * right.width + x];
                            buf.set(x + half_w, y, rc.ch, rc.fg, rc.bg);
                        }
                    }
                }
            }
        }

        // FPS counter
        fps_frames += 1;
        if fps_timer.elapsed().as_secs_f32() >= 1.0 {
            last_fps = fps_frames as f32 / fps_timer.elapsed().as_secs_f32();
            fps_frames = 0;
            fps_timer = Instant::now();
        }
        let fps_str = format!(" FPS: {last_fps:.1} ");
        let fps_x = buf.width.saturating_sub(fps_str.len() + 1);
        for (i, ch) in fps_str.chars().enumerate() {
            buf.set(fps_x + i, 0, ch, (255, 255, 100), (20, 20, 40));
        }

        let rendered = buf.render(cli.use_color);
        print!("{rendered}");
        let _ = io::stdout().flush();

        if let Some(max) = cli.frames {
            if frame_idx >= max { break; }
        }

        if let Some(target) = frame_budget {
            let elapsed = started.elapsed();
            if elapsed < target {
                thread::sleep(target - elapsed);
            }
        }
    }

    // Show cursor
    print!("\x1b[?25h");
    let _ = io::stdout().flush();
    ExitCode::SUCCESS
}
