//! Multi-Scale Semantic Zoom Terrarium Renderer
//!
//! Extends terrarium_ascii with multi-scale semantic zoom: zooming in progressively
//! reveals organism physiology, cellular metabolism, soil chemistry, and molecular
//! dynamics — all emergent from real simulation state.
//!
//! Zoom Levels:
//!   Ecosystem (0.4-1.5x): Grid overview with plants/flies/water
//!   Organism  (1.5-2.5x): Split view with physiology detail panel
//!   Cellular  (2.5-3.5x): Tissue-level metabolic data
//!   Molecular (3.5x+):    Substrate chemistry / particle density
//!
//! Usage:
//!   cargo run --profile fast --no-default-features --bin terrarium_zoom -- [OPTIONS]
//!
//! Options:
//!   --seed <n>       World seed (default: 7)
//!   --fps <n>        Target framerate (default: 15)
//!   --frames <n>     Quit after N frames (default: infinite)
//!   --mode <name>    View mode: iso, top, split, heat, dash (default: iso)
//!   --no-color       Disable ANSI colors

use oneuro_metal::{TerrariumWorld, TerrariumWorldSnapshot, TerrariumSpecies, PlantTissue, ecosystem_dashboard};
// Note: fly_metabolisms, earthworm_population, nematode_guilds are private fields
// on TerrariumWorld. Detail panels use public fields (substrate, plants, flies, waters)
// and snapshot means where per-cell private data isn't accessible.
use std::env;
use std::io::{self, Read, Write};
use std::process::ExitCode;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};
use std::fs::File;

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
    let dry = (180, 140, 80);
    let wet = (80, 60, 30);
    let rich = (60, 90, 40);
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

/// Heatmap: blue(cold) -> cyan -> green -> yellow -> red(hot)
fn heatmap_color(value: f32) -> (u8, u8, u8) {
    let v = value.clamp(0.0, 1.0);
    if v < 0.25 {
        lerp_color((0, 0, 200), (0, 200, 255), v / 0.25)
    } else if v < 0.5 {
        lerp_color((0, 200, 255), (0, 220, 0), (v - 0.25) / 0.25)
    } else if v < 0.75 {
        lerp_color((0, 220, 0), (255, 255, 0), (v - 0.5) / 0.25)
    } else {
        lerp_color((255, 255, 0), (255, 30, 0), (v - 0.75) / 0.25)
    }
}

// ---------------------------------------------------------------------------
// Scenario Presets
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum ScenarioPreset { Normal, Drought, Heat, Competition, Dormancy, Feast, Climate, Amr, Soil }

impl ScenarioPreset {
    fn label(&self) -> &'static str {
        match self {
            Self::Normal => "Normal",
            Self::Drought => "DROUGHT",
            Self::Heat => "HEAT WAVE",
            Self::Competition => "COMPETITION",
            Self::Dormancy => "DORMANCY STRESS",
            Self::Feast => "FEAST",
            Self::Climate => "CLIMATE IMPACT",
            Self::Amr => "AMR EMERGENCE",
            Self::Soil => "SOIL HEALTH",
        }
    }
    fn color(&self) -> (u8, u8, u8) {
        match self {
            Self::Normal => (180, 180, 180),
            Self::Drought => (220, 160, 40),
            Self::Heat => (255, 60, 30),
            Self::Competition => (40, 200, 40),
            Self::Dormancy => (120, 80, 200),
            Self::Feast => (60, 200, 255),
            Self::Climate => (200, 120, 40),
            Self::Amr => (255, 80, 80),
            Self::Soil => (140, 200, 80),
        }
    }
}

/// Apply initial scenario conditions to the world (called once after creation).
fn apply_scenario(world: &mut TerrariumWorld, preset: ScenarioPreset) {
    match preset {
        ScenarioPreset::Normal => {},
        ScenarioPreset::Drought => {
            for v in world.moisture_field_mut().iter_mut() { *v *= 0.3; }
            if world.waters.len() > 1 { world.waters.truncate(1); }
        },
        ScenarioPreset::Heat => {
            for v in world.temperature_field_mut().iter_mut() { *v += 15.0; }
        },
        ScenarioPreset::Competition => {
            // Double plant density by cloning with slight offsets
            let existing: Vec<_> = world.plants.clone();
            for p in &existing {
                let mut p2 = p.clone();
                p2.x = (p2.x + 1).min(world.config.width.saturating_sub(1));
                p2.y = (p2.y + 1).min(world.config.height.saturating_sub(1));
                world.plants.push(p2);
            }
        },
        ScenarioPreset::Dormancy => {
            for v in world.moisture_field_mut().iter_mut() { *v *= 0.2; }
            for v in world.temperature_field_mut().iter_mut() { *v += 8.0; }
            world.fruits.clear();
        },
        ScenarioPreset::Feast => {
            for v in world.moisture_field_mut().iter_mut() { *v = (*v * 2.0).min(1.0); }
            // Double fruit patches by cloning existing ones with larger radius
            let existing: Vec<_> = world.fruits.clone();
            for mut fp in existing {
                fp.radius *= 1.5;
                world.fruits.push(fp);
            }
        },
        ScenarioPreset::Climate => {
            // Gradual warming — start with slightly elevated temperatures
            for v in world.temperature_field_mut().iter_mut() { *v += 3.0; }
            // Boost moisture for active nutrient cycling
            for v in world.moisture_field_mut().iter_mut() { *v = (*v * 1.3).min(0.95); }
        },
        ScenarioPreset::Amr => {
            // High moisture for biofilm-promoting conditions
            for v in world.moisture_field_mut().iter_mut() { *v = (*v * 1.8).min(0.95); }
        },
        ScenarioPreset::Soil => {
            // Rich soil conditions — high moisture, moderate temperature
            for v in world.moisture_field_mut().iter_mut() { *v = (*v * 1.5).min(0.92); }
        },
    }
}

/// Per-frame progressive scenario stress (called each step).
fn apply_scenario_step(world: &mut TerrariumWorld, preset: ScenarioPreset, frame: usize) {
    match preset {
        ScenarioPreset::Normal | ScenarioPreset::Competition | ScenarioPreset::Feast
        | ScenarioPreset::Climate | ScenarioPreset::Amr | ScenarioPreset::Soil => {},
        ScenarioPreset::Drought => {
            if frame % 10 == 0 {
                for v in world.moisture_field_mut().iter_mut() { *v = (*v - 0.005).max(0.0); }
            }
        },
        ScenarioPreset::Heat => {
            if frame % 20 == 0 {
                for v in world.temperature_field_mut().iter_mut() { *v += 0.1; }
            }
        },
        ScenarioPreset::Dormancy => {
            if frame % 15 == 0 {
                for v in world.moisture_field_mut().iter_mut() { *v = (*v - 0.003).max(0.0); }
            }
        },
    }
}

fn terminal_size() -> (usize, usize) {
    if let Ok(output) = std::process::Command::new("stty")
        .arg("size").stdin(std::process::Stdio::inherit()).output()
    {
        if let Ok(s) = String::from_utf8(output.stdout) {
            let parts: Vec<&str> = s.trim().split_whitespace().collect();
            if parts.len() == 2 {
                if let (Ok(rows), Ok(cols)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>()) {
                    if cols > 20 && rows > 10 { return (cols, rows); }
                }
            }
        }
    }
    (120, 45)
}

// ---------------------------------------------------------------------------
// Keyboard input
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum KeyInput { Char(char), Up, Down, Left, Right }

fn open_tty() -> Option<File> { File::open("/dev/tty").ok() }

fn spawn_key_reader(tty: File) -> mpsc::Receiver<KeyInput> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let mut tty = tty;
        let mut buf = [0u8; 8];
        loop {
            match tty.read(&mut buf[..1]) {
                Ok(0) => break,
                Ok(_) => {
                    let key = match buf[0] {
                        b'q' | b'Q' => Some(KeyInput::Char('q')),
                        b'h' | b'H' => Some(KeyInput::Char('h')),
                        b'\t' => Some(KeyInput::Char('\t')),
                        b'm' | b'M' => Some(KeyInput::Char('m')),
                        b'w' | b'W' => Some(KeyInput::Char('w')),
                        b'a' | b'A' => Some(KeyInput::Char('a')),
                        b's' | b'S' => Some(KeyInput::Char('s')),
                        b'd' | b'D' => Some(KeyInput::Char('d')),
                        b'+' | b'=' => Some(KeyInput::Char('+')),
                        b'-' | b'_' => Some(KeyInput::Char('-')),
                        b' ' => Some(KeyInput::Char(' ')),
                        b'r' | b'R' => Some(KeyInput::Char('r')),
                        b'\n' | b'\r' => Some(KeyInput::Char('\n')),
                        b'1' => Some(KeyInput::Char('1')),
                        b'2' => Some(KeyInput::Char('2')),
                        b'3' => Some(KeyInput::Char('3')),
                        b'4' => Some(KeyInput::Char('4')),
                        b'5' => Some(KeyInput::Char('5')),
                        27 => {
                            match tty.read(&mut buf[1..3]) {
                                Ok(2) if buf[1] == b'[' => match buf[2] {
                                    b'A' => Some(KeyInput::Up),
                                    b'B' => Some(KeyInput::Down),
                                    b'C' => Some(KeyInput::Right),
                                    b'D' => Some(KeyInput::Left),
                                    _ => None,
                                },
                                _ => Some(KeyInput::Char('\x1b')),
                            }
                        }
                        _ => None,
                    };
                    if let Some(k) = key { if tx.send(k).is_err() { break; } }
                }
                Err(_) => break,
            }
        }
    });
    rx
}

fn set_raw_mode() -> bool {
    std::process::Command::new("sh").args(["-c", "stty raw -echo </dev/tty"])
        .status().map(|s| s.success()).unwrap_or(false)
}

fn restore_terminal() {
    let _ = std::process::Command::new("sh").args(["-c", "stty sane </dev/tty"]).status();
}

// ---------------------------------------------------------------------------
// Screen Buffer
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Cell { ch: char, fg: (u8, u8, u8), bg: (u8, u8, u8) }

impl Default for Cell {
    fn default() -> Self { Self { ch: ' ', fg: (200, 200, 200), bg: (15, 15, 25) } }
}

struct ScreenBuffer { width: usize, height: usize, cells: Vec<Cell> }

impl ScreenBuffer {
    fn new(w: usize, h: usize) -> Self { Self { width: w, height: h, cells: vec![Cell::default(); w * h] } }
    fn clear(&mut self) { for c in &mut self.cells { *c = Cell::default(); } }

    /// Clear with sky color based on daylight + moonlight (night sky gets blue tint from moon).
    fn clear_with_sky(&mut self, light: f32, moonlight: f32) {
        let day = light.clamp(0.0, 1.0);
        let night = (1.0 - day * 2.0).clamp(0.0, 1.0);
        // Day: warm gray (15,15,25), night: dark blue-black, moon: slight blue tint
        let r = (15.0 + day * 10.0 + moonlight * night * 5.0) as u8;
        let g = (15.0 + day * 12.0 + moonlight * night * 8.0) as u8;
        let b = (25.0 + day * 5.0 + moonlight * night * 20.0) as u8;
        let bg = (r, g, b);
        for c in &mut self.cells { *c = Cell { ch: ' ', fg: (200, 200, 200), bg }; }
    }

    fn set(&mut self, x: usize, y: usize, ch: char, fg_c: (u8, u8, u8), bg_c: (u8, u8, u8)) {
        if x < self.width && y < self.height {
            self.cells[y * self.width + x] = Cell { ch, fg: fg_c, bg: bg_c };
        }
    }

    fn write_str(&mut self, x: usize, y: usize, s: &str, fg_c: (u8, u8, u8), bg_c: (u8, u8, u8)) {
        for (i, ch) in s.chars().enumerate() { self.set(x + i, y, ch, fg_c, bg_c); }
    }

    fn render(&self, use_color: bool) -> String {
        let mut out = String::with_capacity(self.width * self.height * 20);
        out.push_str("\x1b[H");
        for y in 0..self.height {
            for x in 0..self.width {
                let c = &self.cells[y * self.width + x];
                if use_color {
                    out.push_str(&fg(c.fg.0, c.fg.1, c.fg.2));
                    out.push_str(&bg(c.bg.0, c.bg.1, c.bg.2));
                }
                out.push(c.ch);
            }
            if use_color { out.push_str(RESET); }
            out.push_str("\r\n");
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Semantic Zoom Levels & Cursor
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum ZoomLevel { Ecosystem, Organism, Cellular, Molecular }

impl ZoomLevel {
    fn from_zoom(z: f32) -> Self {
        if z < 1.5 { Self::Ecosystem }
        else if z < 2.5 { Self::Organism }
        else if z < 3.5 { Self::Cellular }
        else { Self::Molecular }
    }
    fn label(self) -> &'static str {
        match self {
            Self::Ecosystem => "Ecosystem",
            Self::Organism => "Organism",
            Self::Cellular => "Cellular",
            Self::Molecular => "Molecular",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum SelectedEntity { None, Plant(usize), Fly(usize), Water(usize), SoilCell }

// ---------------------------------------------------------------------------
// View State
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum ViewMode { Isometric, TopDown, Split, Heatmap, Dashboard }

struct ViewState {
    mode: ViewMode,
    camera_x: isize, camera_y: isize,
    zoom: f32,
    paused: bool, show_help: bool, show_minimap: bool, show_legend: bool,
    cursor_x: usize, cursor_y: usize,
    selected: SelectedEntity,
}

impl ViewState {
    fn new(mode: ViewMode, show_minimap: bool) -> Self {
        Self {
            mode, camera_x: 0, camera_y: 0, zoom: 1.0,
            paused: false, show_help: false, show_minimap, show_legend: true,
            cursor_x: 22, cursor_y: 16, selected: SelectedEntity::None,
        }
    }
    fn zoom_level(&self) -> ZoomLevel { ZoomLevel::from_zoom(self.zoom) }
    fn cycle_mode(&mut self) {
        self.mode = match self.mode {
            ViewMode::Isometric => ViewMode::TopDown, ViewMode::TopDown => ViewMode::Heatmap,
            ViewMode::Heatmap => ViewMode::Dashboard, ViewMode::Dashboard => ViewMode::Split,
            ViewMode::Split => ViewMode::Isometric,
        };
    }
    fn mode_name(&self) -> &'static str {
        match self.mode {
            ViewMode::Isometric => "Isometric 3D", ViewMode::TopDown => "Top-Down Map",
            ViewMode::Split => "Split View", ViewMode::Heatmap => "Moisture Heatmap",
            ViewMode::Dashboard => "Ecosystem Dashboard",
        }
    }
}

// ---------------------------------------------------------------------------
// Progress bar helper
// ---------------------------------------------------------------------------

fn bar_str(value: f32, width: usize) -> String {
    let filled = ((value.clamp(0.0, 1.0)) * width as f32).round() as usize;
    format!("{}{}", "\u{2588}".repeat(filled), "\u{2591}".repeat(width.saturating_sub(filled)))
}

// ---------------------------------------------------------------------------
// Legend Bar
// ---------------------------------------------------------------------------

fn draw_legend_bar(buf: &mut ScreenBuffer, vs: &ViewState) {
    let bar_h = 4;
    let y = buf.height.saturating_sub(bar_h);
    let bar_bg = (18, 22, 38);
    let sep_fg = (50, 55, 80);
    let lbl_fg = (170, 180, 210);
    let dim_fg = (110, 115, 140);
    let key_fg = (80, 200, 140);
    let zoom_fg = (255, 200, 80);

    for row in y..buf.height { for x in 0..buf.width { buf.set(x, row, ' ', lbl_fg, bar_bg); } }
    for sx in 0..buf.width { buf.set(sx, y, '\u{2500}', sep_fg, bar_bg); }

    // Row 2: Symbols using actual render colors
    let r2 = y + 1;
    let mut x = 1;
    let typical_plant = plant_color(0.5, 0.8);
    buf.set(x, r2, '\u{2663}', typical_plant, bar_bg); x += 1;
    buf.write_str(x, r2, "=Plant ", lbl_fg, bar_bg); x += 7;
    buf.set(x, r2, '\u{25CF}', (100, 200, 50), bar_bg); x += 1;
    buf.write_str(x, r2, "\u{2192}", dim_fg, bar_bg); x += 1;
    buf.set(x, r2, '\u{25CF}', (255, 80, 30), bar_bg); x += 1;
    buf.write_str(x, r2, "=Fruit ", lbl_fg, bar_bg); x += 7;
    buf.set(x, r2, '\u{2248}', (60, 140, 255), bar_bg); x += 1;
    buf.write_str(x, r2, "=Water ", lbl_fg, bar_bg); x += 7;
    buf.set(x, r2, '\u{2736}', (255, 230, 50), bar_bg); x += 1;
    buf.write_str(x, r2, "=Fly(air) ", lbl_fg, bar_bg); x += 10;
    buf.set(x, r2, '\u{25C6}', (255, 200, 80), bar_bg); x += 1;
    buf.write_str(x, r2, "=Fly(land) ", lbl_fg, bar_bg); x += 11;
    // Terrain samples from actual soil_color()
    let top_c = lerp_color(soil_color(0.4, 0.2), (255, 255, 255), 0.15);
    let side_c = lerp_color(soil_color(0.4, 0.2), (0, 0, 0), 0.2);
    buf.set(x, r2, '\u{2593}', top_c, bar_bg); x += 1;
    buf.write_str(x, r2, "=Top ", lbl_fg, bar_bg); x += 5;
    buf.set(x, r2, '\u{2588}', side_c, bar_bg); x += 1;
    buf.write_str(x, r2, "=Side", lbl_fg, bar_bg);

    // Row 3: Gradients using actual color functions
    let r3 = y + 2;
    x = 1;
    buf.write_str(x, r3, "Soil: ", dim_fg, bar_bg); x += 6;
    for i in 0..10 { let t = i as f32 / 9.0; buf.set(x + i, r3, '\u{2588}', soil_color(t, t * 0.4), bar_bg); }
    x += 10;
    buf.write_str(x, r3, "(dry", soil_color(0.0, 0.0), bar_bg); x += 4;
    buf.set(x, r3, '\u{2192}', dim_fg, bar_bg); x += 1;
    buf.write_str(x, r3, "wet) ", soil_color(1.0, 0.4), bar_bg); x += 5;
    buf.write_str(x, r3, "Plants: ", dim_fg, bar_bg); x += 8;
    let ph = plant_color(1.0, 1.0);
    let ps = plant_color(0.8, 0.2);
    buf.set(x, r3, '\u{2588}', ph, bar_bg); x += 1;
    buf.set(x, r3, '\u{2588}', ph, bar_bg); x += 1;
    buf.write_str(x, r3, "healthy ", ph, bar_bg); x += 8;
    buf.set(x, r3, '\u{2588}', ps, bar_bg); x += 1;
    buf.set(x, r3, '\u{2588}', ps, bar_bg); x += 1;
    buf.write_str(x, r3, "stressed", ps, bar_bg);

    // Row 4: Controls + zoom
    let r4 = y + 3;
    let zl = vs.zoom_level();
    let controls = if zl == ZoomLevel::Ecosystem {
        format!(" [H]elp [Tab]mode [WASD]pan [+/-]zoom [Space]{} [M]inimap [R]eset [Q]uit",
            if vs.paused { "resume" } else { "pause" })
    } else {
        format!(" [H]elp [WASD]cursor [+/-]zoom [Enter]select [Esc]deselect [Space]{} [Q]uit",
            if vs.paused { "resume" } else { "pause" })
    };
    buf.write_str(0, r4, &controls[..controls.len().min(buf.width)], key_fg, bar_bg);

    let right_info = if zl != ZoomLevel::Ecosystem {
        let sel_name = match vs.selected {
            SelectedEntity::None => "", SelectedEntity::Plant(_) => "Plant",
            SelectedEntity::Fly(_) => "Fly", SelectedEntity::Water(_) => "Water",
            SelectedEntity::SoilCell => "Soil",
        };
        format!(" {} | [{},{}] {} ", zl.label(), vs.cursor_x, vs.cursor_y, sel_name)
    } else {
        format!(" {} | {} ", vs.mode_name(), zl.label())
    };
    let mx = buf.width.saturating_sub(right_info.len() + 1);
    buf.write_str(mx, r4, &right_info, zoom_fg, bar_bg);
    if vs.paused {
        let px = buf.width.saturating_sub(right_info.len() + 12);
        buf.write_str(px, r4, " PAUSED ", (255, 80, 80), (80, 20, 20));
    }
}

// ---------------------------------------------------------------------------
// Help Overlay
// ---------------------------------------------------------------------------

fn draw_help_overlay(buf: &mut ScreenBuffer) {
    let pw = 62; let ph = 40;
    let px = buf.width.saturating_sub(pw) / 2;
    let py = buf.height.saturating_sub(ph) / 2;
    let pbg = (20, 25, 45);
    let bfg = (80, 160, 255);
    let tfg = (0, 230, 255);
    let hfg = (255, 200, 80);
    let txt = (200, 210, 220);
    let kfg = (100, 255, 150);
    let dfg = (130, 130, 160);

    for y in py..py + ph { for x in px..px + pw { buf.set(x, y, ' ', txt, pbg); } }
    for x in px..px + pw { buf.set(x, py, '\u{2550}', bfg, pbg); buf.set(x, py + ph - 1, '\u{2550}', bfg, pbg); }
    for y in py..py + ph { buf.set(px, y, '\u{2551}', bfg, pbg); buf.set(px + pw - 1, y, '\u{2551}', bfg, pbg); }
    buf.set(px, py, '\u{2554}', bfg, pbg); buf.set(px + pw - 1, py, '\u{2557}', bfg, pbg);
    buf.set(px, py + ph - 1, '\u{255A}', bfg, pbg); buf.set(px + pw - 1, py + ph - 1, '\u{255D}', bfg, pbg);

    let cx = px + 2;
    let mut r = py + 1;
    buf.write_str(cx, r, "   oNeura Terrarium - Semantic Zoom Viewer", tfg, pbg); r += 2;

    let keys = [
        ("W/A/S/D or Arrows", "Pan camera / move cursor"),
        ("+  /  -", "Zoom in/out (semantic zoom)"),
        ("Enter", "Select entity at cursor"),
        ("Esc", "Deselect / quit"),
        ("Tab", "Cycle view mode"),
        ("1-5", "Jump to mode"),
        ("Space", "Pause / resume"),
        ("H", "Toggle help"), ("M", "Toggle minimap"), ("R", "Reset camera"), ("Q", "Quit"),
    ];
    buf.write_str(cx, r, "\u{2500}\u{2500} CONTROLS ", hfg, pbg); r += 1;
    for (k, desc) in &keys {
        buf.write_str(cx + 2, r, k, kfg, pbg);
        buf.write_str(cx + 22, r, desc, txt, pbg);
        r += 1;
    }
    r += 1;
    buf.write_str(cx, r, "\u{2500}\u{2500} SEMANTIC ZOOM ", hfg, pbg); r += 1;
    buf.write_str(cx + 2, r, "0.4-1.5x  Ecosystem  Grid overview", txt, pbg); r += 1;
    buf.write_str(cx + 2, r, "1.5-2.5x  Organism   Physiology panel", txt, pbg); r += 1;
    buf.write_str(cx + 2, r, "2.5-3.5x  Cellular   Tissue metabolites", txt, pbg); r += 1;
    buf.write_str(cx + 2, r, "3.5x+     Molecular  Substrate chemistry", txt, pbg); r += 2;

    // Color guide using actual functions
    buf.write_str(cx, r, "\u{2500}\u{2500} COLORS ", hfg, pbg); r += 1;
    buf.write_str(cx + 2, r, "Soil: ", (255, 180, 60), pbg);
    for i in 0..5 { let t = i as f32 / 4.0; let c = soil_color(t, t * 0.5);
        buf.set(cx + 8 + i * 3, r, '\u{2588}', c, pbg); buf.set(cx + 9 + i * 3, r, '\u{2588}', c, pbg); }
    r += 1;
    buf.write_str(cx + 2, r, "Plant:", (255, 180, 60), pbg);
    for (i, &(can, vit)) in [(0.1,1.0),(0.5,0.8),(1.0,1.0),(0.8,0.3)].iter().enumerate() {
        let c = plant_color(can, vit);
        buf.set(cx + 8 + i * 3, r, '\u{2663}', c, pbg); buf.set(cx + 9 + i * 3, r, '\u{2663}', c, pbg); }
    r += 2;
    buf.write_str(cx, r, "    Press H to close   |   Tab to change view", dfg, pbg);
}

// ---------------------------------------------------------------------------
// Isometric Renderer
// ---------------------------------------------------------------------------

fn render_isometric(buf: &mut ScreenBuffer, world: &TerrariumWorld, snap: &TerrariumWorldSnapshot, fi: usize, vs: &ViewState) {
    buf.clear_with_sky(snap.light, snap.moonlight);
    let gw = world.config.width;
    let gh = world.config.height;
    let moisture = world.moisture_field();
    let z = vs.zoom;
    let cw = (2.0 * z).round() as isize;
    let ch = (z * 0.5).round().max(1.0) as isize;
    let ox = (buf.width as isize) / 2 + vs.camera_x * cw;
    let oy = 3 + vs.camera_y;

    let moon_ascii = match () {
        _ if snap.lunar_phase < 0.0625  => "(  )",
        _ if snap.lunar_phase < 0.1875  => "( |)",
        _ if snap.lunar_phase < 0.3125  => "( ))",
        _ if snap.lunar_phase < 0.4375  => "(|))",
        _ if snap.lunar_phase < 0.5625  => "(())",
        _ if snap.lunar_phase < 0.6875  => "((|)",
        _ if snap.lunar_phase < 0.8125  => "(( )",
        _ if snap.lunar_phase < 0.9375  => "(| )",
        _                                => "(  )",
    };
    let header = format!(" oNeura Terrarium 3D | F:{} | {} {} | {:.0}C | P:{} Fl:{} Fr:{}", fi, world.time_label(), moon_ascii, snap.temperature, snap.plants, snap.flies, snap.fruits);
    buf.write_str(0, 0, &header[..header.len().min(buf.width)], (0, 220, 255), (20, 20, 40));
    let sub = format!(" moisture:{:.2} microbes:{:.3} CO2:{:.4} O2:{:.2} tide:{:.2} zoom:{:.1}x [{}]",
        snap.mean_soil_moisture, snap.mean_microbes, snap.mean_atmospheric_co2, snap.mean_atmospheric_o2, snap.tidal_moisture_factor, vs.zoom, vs.zoom_level().label());
    buf.write_str(0, 1, &sub[..sub.len().min(buf.width)], (180, 180, 200), (20, 20, 40));
    // Energy budget line: photosynthesis vs respiration
    let photo = snap.mean_atmospheric_o2 * snap.light;
    let resp = snap.mean_atmospheric_co2 * (1.0 + snap.flies as f32 * 0.01);
    let net = photo - resp;
    let (net_label, net_color) = if net >= 0.0 { ("+", (80, 220, 80)) } else { ("", (220, 80, 80)) };
    let energy_line = format!(" Energy: photo={:.3} resp={:.3} net={}{:.3}", photo, resp, net_label, net);
    buf.write_str(0, 2, &energy_line[..energy_line.len().min(buf.width)], net_color, (20, 20, 40));
    if vs.paused { let ps = " PAUSED "; buf.write_str((buf.width.saturating_sub(ps.len()))/2, 3, ps, (255,80,80), (60,20,20)); }

    for gy in 0..gh { for gx in 0..gw {
        let ix = (gx as isize - gy as isize) * cw + ox;
        let iy = (gx as isize + gy as isize) * ch + oy;
        let mi = gy * gw + gx;
        let m = if mi < moisture.len() { moisture[mi] } else { 0.3 };
        let height = (m * 3.0 * z).clamp(0.0, 6.0) as isize;
        let color = soil_color(m, m * 0.5);
        let tw = (cw as usize).max(2).min(5);
        for h in 0..=height {
            let sy = iy - h; let sx = ix;
            if sx >= 0 && sy >= 0 {
                let fc = if h == height { lerp_color(color, (255,255,255), 0.15) } else { lerp_color(color, (0,0,0), 0.2) };
                let cc = if h == height { '\u{2593}' } else { '\u{2588}' };
                for dx in 0..tw { buf.set(sx as usize + dx, sy as usize, cc, fc, (10,10,20)); }
            }
        }
    }}

    for water in &world.waters { if water.alive {
        let ix = (water.x as isize - water.y as isize) * cw + ox;
        let iy = (water.x as isize + water.y as isize) * ch + oy;
        if ix >= 0 && iy >= 0 {
            let wave = if fi % 4 < 2 { '\u{2248}' } else { '~' };
            let tw = (cw as usize + 1).max(3).min(6);
            for dx in 0..tw { buf.set(ix as usize + dx, iy as usize, wave, (60,140,255), (20,60,120)); }
        }
    }}

    for plant in &world.plants {
        let ix = (plant.x as isize - plant.y as isize) * cw + ox;
        let iy = (plant.x as isize + plant.y as isize) * ch + oy;
        if ix >= 0 && iy >= 0 {
            let sx = ix as usize; let sy = iy as usize;
            let cells = plant.cellular.total_cells();
            let ph = (cells * 0.02 * z).clamp(1.0, 6.0) as usize;
            let vit = plant.cellular.vitality();
            let pc = plant_color(cells * 0.01, vit);
            for h in 0..ph { if sy >= h + 1 { buf.set(sx + 1, sy - h - 1, '\u{2503}', (120,80,40), (10,10,20)); } }
            if sy >= ph + 1 {
                let cy = sy - ph - 1;
                buf.set(sx, cy, '\u{2663}', pc, (10,10,20));
                buf.set(sx+1, cy, '\u{2663}', pc, (10,10,20));
                buf.set(sx+2, cy, '\u{2663}', pc, (10,10,20));
                if cy > 0 { buf.set(sx+1, cy-1, '\u{25B2}', lerp_color(pc, (200,255,100), 0.3), (10,10,20)); }
            }
        }
    }

    for fruit in &world.fruits { if fruit.source.alive && fruit.source.sugar_content > 0.01 {
        let ix = (fruit.source.x as isize - fruit.source.y as isize) * cw + ox;
        let iy = (fruit.source.x as isize + fruit.source.y as isize) * ch + oy;
        if ix >= 0 && iy >= 0 {
            let ripe = fruit.source.ripeness.clamp(0.0, 1.0);
            buf.set(ix as usize + 1, iy as usize, '\u{25CF}', lerp_color((100,200,50), (255,80,30), ripe), (10,10,20));
        }
    }}

    for fly in &world.flies {
        let b = fly.body_state();
        let gx = b.x.round().clamp(0.0, (gw-1) as f32) as isize;
        let gy = b.y.round().clamp(0.0, (gh-1) as f32) as isize;
        let ix = (gx - gy) * cw + ox; let iy = (gx + gy) * ch + oy;
        let alt = (b.z * z).clamp(0.0, 5.0) as isize;
        if ix >= 0 && iy >= alt {
            let (cc, col) = if b.is_flying {
                (if fi % 6 < 3 { '\u{2736}' } else { '\u{2734}' }, (255,230,50))
            } else { ('\u{25C6}', (255,200,80)) };
            buf.set(ix as usize + 1, (iy - alt) as usize, cc, col, (10,10,20));
        }
    }
}

// ---------------------------------------------------------------------------
// Top-Down Renderer (region-aware for split view)
// ---------------------------------------------------------------------------

fn render_topdown_region(buf: &mut ScreenBuffer, world: &TerrariumWorld, snap: &TerrariumWorldSnapshot, fi: usize, vs: &ViewState, rx: usize, rw: usize) {
    let gw = world.config.width; let gh = world.config.height;
    let moisture = world.moisture_field();
    let td_moon = match () {
        _ if snap.lunar_phase < 0.0625  => "(  )",
        _ if snap.lunar_phase < 0.1875  => "( |)",
        _ if snap.lunar_phase < 0.3125  => "( ))",
        _ if snap.lunar_phase < 0.4375  => "(|))",
        _ if snap.lunar_phase < 0.5625  => "(())",
        _ if snap.lunar_phase < 0.6875  => "((|)",
        _ if snap.lunar_phase < 0.8125  => "(( )",
        _ if snap.lunar_phase < 0.9375  => "(| )",
        _                                => "(  )",
    };
    let header = format!(" Terrarium | F:{} | {} {} | {:.0}C | P:{} Fl:{} tide:{:.2} zoom:{:.1}x [{}]",
        fi, world.time_label(), td_moon, snap.temperature, snap.plants, snap.flies, snap.tidal_moisture_factor, vs.zoom, vs.zoom_level().label());
    buf.write_str(rx, 0, &header[..header.len().min(rw)], (0, 220, 255), (20, 20, 40));

    let scale = (2.0 * vs.zoom).round().max(1.0).min(6.0) as usize;
    let ox = (rx as isize + (rw as isize - (gw * scale) as isize) / 2) + vs.camera_x * scale as isize;
    let oy = 2 + vs.camera_y;

    for gy in 0..gh { for gx in 0..gw {
        let mi = gy * gw + gx;
        let m = if mi < moisture.len() { moisture[mi] } else { 0.3 };
        let color = soil_color(m, m * 0.5);
        let cc = if m > 0.5 { '\u{2593}' } else if m > 0.3 { '\u{2592}' } else { '\u{2591}' };
        let px = ox + (gx * scale) as isize; let py = oy + gy as isize;
        if py >= 0 { for dx in 0..scale {
            let sx = px + dx as isize;
            if sx >= rx as isize && (sx as usize) < rx + rw { buf.set(sx as usize, py as usize, cc, color, (10,10,20)); }
        }}
    }}

    for water in &world.waters { if water.alive {
        let wave = if fi % 4 < 2 { '\u{2248}' } else { '~' };
        for dy in 0..3_usize { for dx in 0..3_usize {
            let wx = water.x.saturating_add(dx).saturating_sub(1);
            let wy = water.y.saturating_add(dy).saturating_sub(1);
            if wx < gw && wy < gh {
                let px = ox + (wx * scale) as isize; let py = oy + wy as isize;
                if px >= 0 && py >= 0 { for s in 0..scale {
                    let sx = (px + s as isize) as usize;
                    if sx >= rx && sx < rx + rw { buf.set(sx, py as usize, wave, (80,160,255), (20,50,100)); }
                }}
            }
        }}
    }}

    for plant in &world.plants {
        let cells = plant.cellular.total_cells();
        let pc = plant_color(cells * 0.01, plant.cellular.vitality());
        let cc = if cells > 50.0 { '\u{2663}' } else { '\u{2022}' };
        let px = ox + (plant.x * scale) as isize; let py = oy + plant.y as isize;
        if px >= rx as isize && py >= 0 && (px as usize) < rx + rw {
            buf.set(px as usize, py as usize, cc, pc, (10,30,10));
            if scale > 1 && (px as usize + 1) < rx + rw { buf.set(px as usize + 1, py as usize, cc, pc, (10,30,10)); }
        }
    }

    for fruit in &world.fruits { if fruit.source.alive && fruit.source.sugar_content > 0.01 {
        let ripe = fruit.source.ripeness.clamp(0.0, 1.0);
        let c = lerp_color((100,200,50), (255,80,30), ripe);
        let px = ox + (fruit.source.x * scale) as isize; let py = oy + fruit.source.y as isize;
        if px >= rx as isize && py >= 0 && (px as usize) < rx + rw { buf.set(px as usize, py as usize, '\u{25CF}', c, (10,10,20)); }
    }}

    for fly in &world.flies {
        let b = fly.body_state();
        let gx = b.x.round().clamp(0.0, (gw-1) as f32) as usize;
        let gy = b.y.round().clamp(0.0, (gh-1) as f32) as usize;
        let (cc, col) = if b.is_flying { (if fi % 4 < 2 { '\u{2736}' } else { '\u{2734}' }, (255,230,50)) } else { ('\u{25C6}', (255,200,80)) };
        let px = ox + (gx * scale) as isize; let py = oy + gy as isize;
        if px >= rx as isize && py >= 0 && (px as usize) < rx + rw { buf.set(px as usize, py as usize, cc, col, (10,10,20)); }
    }
}

fn render_topdown(buf: &mut ScreenBuffer, world: &TerrariumWorld, snap: &TerrariumWorldSnapshot, fi: usize, vs: &ViewState) {
    buf.clear();
    render_topdown_region(buf, world, snap, fi, vs, 0, buf.width);
}

fn render_heatmap(buf: &mut ScreenBuffer, world: &TerrariumWorld, snap: &TerrariumWorldSnapshot, fi: usize, vs: &ViewState) {
    buf.clear();
    let gw = world.config.width; let gh = world.config.height;
    let moisture = world.moisture_field();
    let header = format!(" Heatmap | F:{} | {:.0}C | Avg:{:.3} | P:{} Fl:{} z:{:.1}x", fi, snap.temperature, snap.mean_soil_moisture, snap.plants, snap.flies, vs.zoom);
    buf.write_str(0, 0, &header[..header.len().min(buf.width)], (255, 200, 0), (30, 10, 10));
    let scale = (3.0 * vs.zoom).round().max(1.0).min(8.0) as usize;
    let ox = ((buf.width as isize - (gw * scale) as isize) / 2) + vs.camera_x * scale as isize;
    let oy = 2 + vs.camera_y;
    for gy in 0..gh { for gx in 0..gw {
        let mi = gy * gw + gx;
        let m = if mi < moisture.len() { moisture[mi] } else { 0.0 };
        let color = heatmap_color(m.clamp(0.0, 1.0));
        let cc = if m > 0.7 { '\u{2588}' } else if m > 0.5 { '\u{2593}' } else if m > 0.3 { '\u{2592}' } else { '\u{2591}' };
        let px = ox + (gx * scale) as isize; let py = oy + gy as isize;
        if py >= 0 { for dx in 0..scale { if px + dx as isize >= 0 { buf.set((px + dx as isize) as usize, py as usize, cc, color, (5,5,15)); } } }
    }}
}

fn render_dashboard(buf: &mut ScreenBuffer, history: &[TerrariumWorldSnapshot], fi: usize) {
    buf.clear();
    let dash = ecosystem_dashboard(history, buf.width);
    for (y, line) in dash.lines().enumerate() {
        if y >= buf.height.saturating_sub(3) { break; }
        let is_h = line.starts_with('=') || line.starts_with('-');
        buf.write_str(0, y, &line[..line.len().min(buf.width)], if is_h { (0,220,255) } else { (200,220,200) }, if is_h { (20,40,60) } else { (15,15,25) });
    }
    let fs = format!(" F:{fi} ");
    buf.write_str(buf.width.saturating_sub(fs.len()+1), buf.height.saturating_sub(4), &fs, (255,200,100), (30,30,50));
}

fn draw_minimap(buf: &mut ScreenBuffer, world: &TerrariumWorld, mx: usize, my: usize) {
    let gw = world.config.width; let gh = world.config.height;
    let moisture = world.moisture_field();
    buf.set(mx, my, '\u{250C}', (80,80,120), (10,10,20));
    for x in 1..=gw { buf.set(mx+x, my, '\u{2500}', (80,80,120), (10,10,20)); }
    buf.set(mx+gw+1, my, '\u{2510}', (80,80,120), (10,10,20));
    for gy in 0..gh {
        buf.set(mx, my+gy+1, '\u{2502}', (80,80,120), (10,10,20));
        for gx in 0..gw { let mi = gy*gw+gx; let m = if mi < moisture.len() { moisture[mi] } else { 0.3 };
            buf.set(mx+gx+1, my+gy+1, '\u{2588}', soil_color(m, m*0.3), (5,5,10)); }
        buf.set(mx+gw+1, my+gy+1, '\u{2502}', (80,80,120), (10,10,20));
    }
    buf.set(mx, my+gh+1, '\u{2514}', (80,80,120), (10,10,20));
    for x in 1..=gw { buf.set(mx+x, my+gh+1, '\u{2500}', (80,80,120), (10,10,20)); }
    buf.set(mx+gw+1, my+gh+1, '\u{2518}', (80,80,120), (10,10,20));
    for p in &world.plants { if p.x < gw && p.y < gh { buf.set(mx+p.x+1, my+p.y+1, '\u{2022}', (0,200,0), (5,5,10)); } }
    for fly in &world.flies { let b = fly.body_state();
        let fx = b.x.round().clamp(0.0,(gw-1) as f32) as usize; let fy = b.y.round().clamp(0.0,(gh-1) as f32) as usize;
        buf.set(mx+fx+1, my+fy+1, '\u{00B7}', (255,230,50), (5,5,10)); }
}

// ---------------------------------------------------------------------------
// Detail Panel (Organism / Cellular / Molecular)
// ---------------------------------------------------------------------------

fn draw_detail_panel(buf: &mut ScreenBuffer, world: &TerrariumWorld, snap: &TerrariumWorldSnapshot, vs: &ViewState, panel_x: usize, panel_w: usize, fi: usize) {
    let pbg = (12, 14, 28);
    let tfg = (0, 220, 255); let lfg = (170, 180, 210); let vfg = (220, 230, 200);
    let bfg = (80, 200, 140); let hfg = (255, 200, 80); let dfg = (110, 115, 140);
    let bw = panel_w.saturating_sub(6).min(16);

    for y in 0..buf.height.saturating_sub(4) { for x in panel_x..panel_x+panel_w { buf.set(x, y, ' ', lfg, pbg); }
        buf.set(panel_x, y, '\u{2502}', (50,55,80), pbg); }

    let cx = panel_x + 2;
    let zl = vs.zoom_level();
    let gw = world.config.width; let gh = world.config.height;
    let depth = world.config.depth.max(1);

    match (zl, vs.selected) {
        // ---- Plant detail (Organism level) ----
        (ZoomLevel::Organism, SelectedEntity::Plant(idx)) if idx < world.plants.len() => {
            let plant = &world.plants[idx];
            let phys = &plant.physiology; let cell = &plant.cellular;
            let mut r = 1;
            buf.write_str(cx, r, &format!("Plant #{} [{},{}] age:{:.0}s", idx, plant.x, plant.y, phys.age_s()), tfg, pbg); r += 2;
            buf.write_str(cx, r, "\u{2500}\u{2500}\u{2500} Physiology \u{2500}\u{2500}\u{2500}", hfg, pbg); r += 1;
            buf.write_str(cx, r, "Health ", lfg, pbg);
            buf.write_str(cx+9, r, &bar_str(phys.health(), bw), bfg, pbg);
            buf.write_str(cx+10+bw, r, &format!(" {:.2}", phys.health()), vfg, pbg); r += 1;
            for (name, val) in [("Leaf   ", phys.leaf_biomass()), ("Stem   ", phys.stem_biomass()),
                ("Root   ", phys.root_biomass()), ("Carbon ", phys.storage_carbon()),
                ("Water  ", phys.water_buffer()), ("Nitro  ", phys.nitrogen_buffer())] {
                buf.write_str(cx, r, name, lfg, pbg);
                buf.write_str(cx+9, r, &format!("{:.3}g", val), vfg, pbg); r += 1;
            }
            buf.write_str(cx, r, &format!("Height  {:.1}mm", phys.height_mm()), vfg, pbg); r += 2;
            buf.write_str(cx, r, "\u{2500}\u{2500}\u{2500} Vitality \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}", hfg, pbg); r += 1;
            for (name, tissue) in [("Leaf  ", PlantTissue::Leaf), ("Stem  ", PlantTissue::Stem),
                ("Root  ", PlantTissue::Root), ("Merist", PlantTissue::Meristem)] {
                let s = cell.cluster_snapshot(tissue);
                buf.write_str(cx, r, name, lfg, pbg);
                buf.write_str(cx+7, r, &bar_str(s.vitality, bw), bfg, pbg);
                buf.write_str(cx+8+bw, r, &format!(" {:.2}", s.vitality), vfg, pbg); r += 1;
            }
            r += 1;
            let ec = cell.energy_charge();
            buf.write_str(cx, r, "EC: ", lfg, pbg); buf.write_str(cx+4, r, &bar_str(ec, bw), bfg, pbg);
            buf.write_str(cx+5+bw, r, &format!(" {:.2}  Cells:{:.0}", ec, cell.total_cells()), vfg, pbg);
        }

        // ---- Plant detail (Cellular level) ----
        (ZoomLevel::Cellular, SelectedEntity::Plant(idx)) if idx < world.plants.len() => {
            let plant = &world.plants[idx]; let cell = &plant.cellular;
            let mut r = 1;
            buf.write_str(cx, r, &format!("Plant #{} Cellular", idx), tfg, pbg); r += 2;
            for (name, tissue) in [("LEAF", PlantTissue::Leaf), ("STEM", PlantTissue::Stem),
                ("ROOT", PlantTissue::Root), ("MERI", PlantTissue::Meristem)] {
                let s = cell.cluster_snapshot(tissue);
                buf.write_str(cx, r, &format!("\u{250C}\u{2500} {} ({:.0} cells)", name, s.cell_count), hfg, pbg); r += 1;
                for (mname, mval) in [("Vital", s.vitality), ("ATP  ", s.state_atp.clamp(0.0,1.0)),
                    ("Gluc ", s.state_glucose.clamp(0.0,1.0)), ("Strch", s.state_starch.clamp(0.0,1.0)),
                    ("Water", s.state_water.clamp(0.0,1.0)), ("NO3  ", s.state_nitrate.clamp(0.0,1.0))] {
                    buf.write_str(cx, r, "\u{2502} ", dfg, pbg);
                    buf.write_str(cx+2, r, mname, lfg, pbg);
                    buf.write_str(cx+8, r, &bar_str(mval, bw), bfg, pbg);
                    buf.write_str(cx+9+bw, r, &format!(" {:.2}", mval), vfg, pbg); r += 1;
                }
                buf.write_str(cx, r, "\u{2514}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}", dfg, pbg); r += 1;
                if r >= buf.height.saturating_sub(6) { break; }
            }
        }

        // ---- Fly detail ----
        (ZoomLevel::Organism, SelectedEntity::Fly(idx)) |
        (ZoomLevel::Cellular, SelectedEntity::Fly(idx)) if idx < world.flies.len() => {
            let fly = &world.flies[idx]; let b = fly.body_state();
            let mut r = 1;
            buf.write_str(cx, r, &format!("Fly #{} ({:.1},{:.1},{:.1})", idx, b.x, b.y, b.z), tfg, pbg); r += 1;
            buf.write_str(cx, r, &format!("State: {} E:{:.1}uJ", if b.is_flying {"Flying"} else {"Landed"}, b.energy), vfg, pbg); r += 2;
            buf.write_str(cx, r, "\u{2500}\u{2500}\u{2500} Body State \u{2500}\u{2500}\u{2500}", hfg, pbg); r += 1;
            let ef = (b.energy / 100.0).clamp(0.0, 1.0);
            buf.write_str(cx, r, "Energy  ", lfg, pbg);
            buf.write_str(cx+9, r, &bar_str(ef, bw), bfg, pbg);
            buf.write_str(cx+10+bw, r, &format!(" {:.1}", b.energy), vfg, pbg); r += 1;
            for (name, val) in [("Speed   ", b.speed), ("Heading ", b.heading),
                ("Pitch   ", b.pitch), ("Altitude", b.z), ("Temp    ", b.temperature)] {
                buf.write_str(cx, r, name, lfg, pbg);
                buf.write_str(cx+9, r, &format!("{:.2}", val), vfg, pbg); r += 1;
            }
            r += 1;
            // Local substrate at fly position
            let fx = b.x.round().clamp(0.0, (gw-1) as f32) as usize;
            let fy = b.y.round().clamp(0.0, (gh-1) as f32) as usize;
            let idx_2d = fy * gw + fx;
            buf.write_str(cx, r, "\u{2500}\u{2500}\u{2500} Local Chem \u{2500}\u{2500}\u{2500}", hfg, pbg); r += 1;
            for (name, sp) in [("O2     ", TerrariumSpecies::OxygenGas), ("Glucose", TerrariumSpecies::Glucose),
                ("CO2    ", TerrariumSpecies::CarbonDioxide)] {
                let field = world.substrate.species_field(sp);
                let val = if idx_2d < field.len() / depth { field[idx_2d] } else { 0.0 };
                buf.write_str(cx, r, name, lfg, pbg);
                buf.write_str(cx+8, r, &bar_str(val.clamp(0.0,1.0), bw), bfg, pbg);
                buf.write_str(cx+9+bw, r, &format!(" {:.3}", val), vfg, pbg); r += 1;
            }
        }

        // ---- Water detail ----
        (ZoomLevel::Organism, SelectedEntity::Water(idx)) |
        (ZoomLevel::Cellular, SelectedEntity::Water(idx)) if idx < world.waters.len() => {
            let water = &world.waters[idx];
            let mut r = 1;
            buf.write_str(cx, r, &format!("Water [{},{}]", water.x, water.y), tfg, pbg); r += 1;
            buf.write_str(cx, r, &format!("Volume: {:.1}  Evap: {:.3}/s", water.volume, water.evaporation_rate), vfg, pbg); r += 2;
            if water.x < gw && water.y < gh {
                let idx_2d = water.y * gw + water.x;
                buf.write_str(cx, r, "\u{2500}\u{2500}\u{2500} Dissolved \u{2500}\u{2500}\u{2500}\u{2500}", hfg, pbg); r += 1;
                for (name, sp) in [("Glucose", TerrariumSpecies::Glucose), ("O2     ", TerrariumSpecies::OxygenGas),
                    ("NO3    ", TerrariumSpecies::Nitrate), ("NH4    ", TerrariumSpecies::Ammonium),
                    ("CO2    ", TerrariumSpecies::CarbonDioxide)] {
                    let field = world.substrate.species_field(sp);
                    let val = if idx_2d < field.len() / depth { field[idx_2d] } else { 0.0 };
                    buf.write_str(cx, r, name, lfg, pbg);
                    buf.write_str(cx+8, r, &bar_str(val.clamp(0.0,1.0), bw), bfg, pbg);
                    buf.write_str(cx+9+bw, r, &format!(" {:.3}", val), vfg, pbg); r += 1;
                }
                r += 1;
                buf.write_str(cx, r, "\u{2500}\u{2500}\u{2500} Ecology (mean) \u{2500}", hfg, pbg); r += 1;
                buf.write_str(cx, r, &format!("Microbes  {:.4} gC", snap.mean_microbes), vfg, pbg); r += 1;
                buf.write_str(cx, r, &format!("Symbionts {:.4} gC", snap.mean_symbionts), vfg, pbg);
            }
        }

        // ---- Soil cell detail ----
        (ZoomLevel::Organism, SelectedEntity::SoilCell) |
        (ZoomLevel::Cellular, SelectedEntity::SoilCell) => {
            let gx = vs.cursor_x.min(gw.saturating_sub(1));
            let gy = vs.cursor_y.min(gh.saturating_sub(1));
            let idx_2d = gy * gw + gx;
            let mut r = 1;
            buf.write_str(cx, r, &format!("Soil Cell [{},{}]", gx, gy), tfg, pbg); r += 2;
            buf.write_str(cx, r, "\u{2500}\u{2500}\u{2500} Chemistry \u{2500}\u{2500}\u{2500}\u{2500}", hfg, pbg); r += 1;
            let moisture = world.moisture_field();
            let m_val = if idx_2d < moisture.len() { moisture[idx_2d] } else { 0.0 };
            buf.write_str(cx, r, "Moisture", lfg, pbg);
            buf.write_str(cx+9, r, &bar_str(m_val, bw), bfg, pbg);
            buf.write_str(cx+10+bw, r, &format!(" {:.3}", m_val), vfg, pbg); r += 1;
            // Mean ecological data (per-cell data is private)
            buf.write_str(cx, r, "Microbes", lfg, pbg);
            buf.write_str(cx+9, r, &bar_str(snap.mean_microbes.clamp(0.0,1.0), bw), bfg, pbg);
            buf.write_str(cx+10+bw, r, &format!(" {:.3}", snap.mean_microbes), vfg, pbg); r += 1;
            buf.write_str(cx, r, "Symbiont", lfg, pbg);
            buf.write_str(cx+9, r, &bar_str(snap.mean_symbionts.clamp(0.0,1.0), bw), bfg, pbg);
            buf.write_str(cx+10+bw, r, &format!(" {:.3}", snap.mean_symbionts), vfg, pbg); r += 2;

            buf.write_str(cx, r, "\u{2500}\u{2500}\u{2500} Substrate \u{2500}\u{2500}\u{2500}\u{2500}", hfg, pbg); r += 1;
            let substrate_species = [
                ("Glucose", TerrariumSpecies::Glucose), ("O2     ", TerrariumSpecies::OxygenGas),
                ("NH4    ", TerrariumSpecies::Ammonium), ("NO3    ", TerrariumSpecies::Nitrate),
                ("CO2    ", TerrariumSpecies::CarbonDioxide), ("ATP flx", TerrariumSpecies::AtpFlux),
                ("H+ (pH)", TerrariumSpecies::Proton),
            ];
            for (name, sp) in &substrate_species {
                let field = world.substrate.species_field(*sp);
                let val = if idx_2d < field.len() / depth { field[idx_2d] } else { 0.0 };
                buf.write_str(cx, r, name, lfg, pbg);
                buf.write_str(cx+8, r, &bar_str(val.clamp(0.0,1.0), bw), bfg, pbg);
                buf.write_str(cx+9+bw, r, &format!(" {:.3}", val), vfg, pbg); r += 1;
                if r >= buf.height.saturating_sub(6) { break; }
            }
        }

        // ---- Molecular view ----
        (ZoomLevel::Molecular, _) => {
            let gx = vs.cursor_x.min(gw.saturating_sub(1));
            let gy = vs.cursor_y.min(gh.saturating_sub(1));
            let idx_2d = gy * gw + gx;
            let mut r = 1;
            buf.write_str(cx, r, &format!("Molecular [{},{}]", gx, gy), tfg, pbg); r += 2;

            // Element legend
            let elems: [(&str, (u8,u8,u8)); 6] = [("C",(139,90,43)),("H",(200,200,200)),("O",(255,60,60)),("N",(60,60,255)),("P",(255,165,0)),("S",(255,255,0))];
            let mut ex = cx;
            for (sym, col) in &elems { buf.set(ex, r, '\u{2588}', *col, pbg); buf.write_str(ex+1, r, sym, *col, pbg); ex += sym.len() + 2; }
            r += 2;

            buf.write_str(cx, r, "\u{2500}\u{2500}\u{2500} All 14 Species \u{2500}\u{2500}", hfg, pbg); r += 1;
            let all_sp = [
                ("Carbon  ", TerrariumSpecies::Carbon, (139,90,43)), ("Hydrogen", TerrariumSpecies::Hydrogen, (200,200,200)),
                ("Oxygen  ", TerrariumSpecies::Oxygen, (255,60,60)), ("Nitrogen", TerrariumSpecies::Nitrogen, (60,60,255)),
                ("Phosphor", TerrariumSpecies::Phosphorus, (255,165,0)), ("Sulfur  ", TerrariumSpecies::Sulfur, (255,255,0)),
                ("Water   ", TerrariumSpecies::Water, (60,140,255)), ("Glucose ", TerrariumSpecies::Glucose, (200,140,60)),
                ("O2 gas  ", TerrariumSpecies::OxygenGas, (180,60,60)), ("Ammonium", TerrariumSpecies::Ammonium, (120,180,60)),
                ("Nitrate ", TerrariumSpecies::Nitrate, (60,160,120)), ("CO2     ", TerrariumSpecies::CarbonDioxide, (160,160,160)),
                ("H+ (pH) ", TerrariumSpecies::Proton, (200,100,200)), ("ATP flux", TerrariumSpecies::AtpFlux, (255,200,80)),
            ];
            for (name, sp, col) in &all_sp {
                let field = world.substrate.species_field(*sp);
                let val = if idx_2d < field.len() / depth { field[idx_2d] } else { 0.0 };
                buf.write_str(cx, r, name, *col, pbg);
                buf.write_str(cx+9, r, &bar_str(val.clamp(0.0,1.0), bw), *col, pbg);
                buf.write_str(cx+10+bw, r, &format!(" {:.4}", val), vfg, pbg); r += 1;
                if r >= buf.height.saturating_sub(6) { break; }
            }

            // Particle density map
            if r + 8 < buf.height.saturating_sub(6) {
                r += 1;
                buf.write_str(cx, r, "\u{2500}\u{2500}\u{2500} Particles \u{2500}\u{2500}\u{2500}\u{2500}", hfg, pbg); r += 1;
                let mw = panel_w.saturating_sub(6).min(28);
                let mh = 6;
                let c_val = { let f = world.substrate.species_field(TerrariumSpecies::Carbon); if idx_2d < f.len()/depth { f[idx_2d] } else { 0.0 } };
                let h_val = { let f = world.substrate.species_field(TerrariumSpecies::Hydrogen); if idx_2d < f.len()/depth { f[idx_2d] } else { 0.0 } };
                let o_val = { let f = world.substrate.species_field(TerrariumSpecies::Oxygen); if idx_2d < f.len()/depth { f[idx_2d] } else { 0.0 } };
                let n_val = { let f = world.substrate.species_field(TerrariumSpecies::Nitrogen); if idx_2d < f.len()/depth { f[idx_2d] } else { 0.0 } };

                buf.set(cx, r, '\u{250C}', dfg, pbg);
                for dx in 0..mw { buf.set(cx+1+dx, r, '\u{2500}', dfg, pbg); }
                buf.set(cx+1+mw, r, '\u{2510}', dfg, pbg); r += 1;
                for dy in 0..mh {
                    buf.set(cx, r, '\u{2502}', dfg, pbg); buf.set(cx+1+mw, r, '\u{2502}', dfg, pbg);
                    for dx in 0..mw {
                        let hash = ((dx * 7 + dy * 13 + fi * 3 + idx_2d * 11) % 37) as f32 / 37.0;
                        let thresh_c = c_val * 0.8;
                        let thresh_h = thresh_c + h_val * 0.5;
                        let thresh_o = thresh_h + o_val * 0.6;
                        let thresh_n = thresh_o + n_val * 0.7;
                        let (ch, col) = if hash < thresh_c { ('\u{2588}', (139,90,43)) }
                            else if hash < thresh_h { ('\u{2593}', (200,200,200)) }
                            else if hash < thresh_o { ('\u{2592}', (255,60,60)) }
                            else if hash < thresh_n { ('\u{2588}', (60,60,255)) }
                            else { (' ', pbg) };
                        buf.set(cx+1+dx, r, ch, col, pbg);
                    }
                    r += 1;
                }
                buf.set(cx, r, '\u{2514}', dfg, pbg);
                for dx in 0..mw { buf.set(cx+1+dx, r, '\u{2500}', dfg, pbg); }
                buf.set(cx+1+mw, r, '\u{2518}', dfg, pbg);
            }
        }

        // ---- No selection: cursor info ----
        _ => {
            let gx = vs.cursor_x.min(gw.saturating_sub(1));
            let gy = vs.cursor_y.min(gh.saturating_sub(1));
            let idx_2d = gy * gw + gx;
            let mut r = 1;
            buf.write_str(cx, r, &format!("Cursor [{},{}]", gx, gy), tfg, pbg); r += 2;
            let moisture = world.moisture_field();
            let m = if idx_2d < moisture.len() { moisture[idx_2d] } else { 0.0 };
            buf.write_str(cx, r, "Moisture", lfg, pbg);
            buf.write_str(cx+9, r, &bar_str(m, bw), bfg, pbg);
            buf.write_str(cx+10+bw, r, &format!(" {:.2}", m), vfg, pbg); r += 2;
            buf.write_str(cx, r, "Press Enter to select", dfg, pbg); r += 1;
            buf.write_str(cx, r, "nearest entity, or", dfg, pbg); r += 1;
            buf.write_str(cx, r, "soil cell if none.", dfg, pbg);
        }
    }
}

// ---------------------------------------------------------------------------
// Cursor Crosshair
// ---------------------------------------------------------------------------

fn draw_cursor(buf: &mut ScreenBuffer, world: &TerrariumWorld, vs: &ViewState, fi: usize) {
    let gw = world.config.width; let gh = world.config.height;
    let scale = (2.0 * vs.zoom).round().max(1.0).min(6.0) as isize;
    let spatial_w = buf.width * 3 / 5;
    let ox = (spatial_w as isize - gw as isize * scale) / 2 + vs.camera_x * scale;
    let oy = 2 + vs.camera_y;
    let cx = vs.cursor_x.min(gw.saturating_sub(1));
    let cy = vs.cursor_y.min(gh.saturating_sub(1));
    let sx = ox + cx as isize * scale; let sy = oy + cy as isize;
    if sx >= 0 && sy >= 0 {
        let sxu = sx as usize; let syu = sy as usize;
        let blink = fi % 8 < 5;
        let cfig = if blink { (255,255,0) } else { (200,180,0) };
        if syu > 0 { buf.set(sxu, syu-1, '\u{2502}', cfig, (10,10,20)); }
        if syu+1 < buf.height { buf.set(sxu, syu+1, '\u{2502}', cfig, (10,10,20)); }
        if sxu > 0 { buf.set(sxu-1, syu, '\u{2500}', cfig, (10,10,20)); }
        if sxu+1 < buf.width { buf.set(sxu+1, syu, '\u{2500}', cfig, (10,10,20)); }
        buf.set(sxu, syu, '\u{253C}', cfig, (10,10,20));
    }
}

// ---------------------------------------------------------------------------
// Entity Selection
// ---------------------------------------------------------------------------

fn select_nearest(world: &TerrariumWorld, vs: &ViewState) -> SelectedEntity {
    let cx = vs.cursor_x as f32; let cy = vs.cursor_y as f32;
    let mut best_d = f32::MAX; let mut best = SelectedEntity::SoilCell;
    for (i, p) in world.plants.iter().enumerate() {
        let d = ((p.x as f32 - cx).powi(2) + (p.y as f32 - cy).powi(2)).sqrt();
        if d < best_d && d < 3.0 { best_d = d; best = SelectedEntity::Plant(i); }
    }
    for (i, f) in world.flies.iter().enumerate() {
        let b = f.body_state();
        let d = ((b.x - cx).powi(2) + (b.y - cy).powi(2)).sqrt();
        if d < best_d && d < 3.0 { best_d = d; best = SelectedEntity::Fly(i); }
    }
    for (i, w) in world.waters.iter().enumerate() { if w.alive {
        let d = ((w.x as f32 - cx).powi(2) + (w.y as f32 - cy).powi(2)).sqrt();
        if d < best_d && d < 2.0 { best_d = d; best = SelectedEntity::Water(i); }
    }}
    best
}

// ---------------------------------------------------------------------------
// Main render dispatch
// ---------------------------------------------------------------------------

fn render_scene(buf: &mut ScreenBuffer, world: &TerrariumWorld, snap: &TerrariumWorldSnapshot, fi: usize, vs: &ViewState, history: &[TerrariumWorldSnapshot], tw: usize, th: usize) {
    let zl = vs.zoom_level();
    if zl != ZoomLevel::Ecosystem && vs.mode != ViewMode::Dashboard {
        // Split: spatial context (left ~60%) + detail panel (right ~40%)
        let pw = (buf.width * 2 / 5).max(30).min(45);
        let sw = buf.width.saturating_sub(pw);
        buf.clear();
        render_topdown_region(buf, world, snap, fi, vs, 0, sw);
        draw_cursor(buf, world, vs, fi);
        draw_detail_panel(buf, world, snap, vs, sw, pw, fi);
    } else {
        match vs.mode {
            ViewMode::Isometric => {
                render_isometric(buf, world, snap, fi, vs);
                if vs.show_minimap { draw_minimap(buf, world, buf.width.saturating_sub(world.config.width + 4), 3); }
            }
            ViewMode::TopDown => {
                render_topdown(buf, world, snap, fi, vs);
                if vs.show_minimap { draw_minimap(buf, world, buf.width.saturating_sub(world.config.width + 4), 3); }
            }
            ViewMode::Split => {
                let hw = tw / 2;
                let mut l = ScreenBuffer::new(hw, th); let mut r = ScreenBuffer::new(hw, th);
                render_isometric(&mut l, world, snap, fi, vs);
                r.clear(); render_topdown_region(&mut r, world, snap, fi, vs, 0, hw);
                buf.clear();
                for y in 0..th { for x in 0..hw {
                    let lc = &l.cells[y * hw + x]; buf.set(x, y, lc.ch, lc.fg, lc.bg);
                    if x < r.width { let rc = &r.cells[y * r.width + x]; buf.set(x+hw, y, rc.ch, rc.fg, rc.bg); }
                }}
            }
            ViewMode::Heatmap => render_heatmap(buf, world, snap, fi, vs),
            ViewMode::Dashboard => render_dashboard(buf, history, fi),
        }
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Cli { seed: u64, fps: u64, frames: Option<usize>, mode: ViewMode, use_color: bool, cpu_substrate: bool, show_minimap: bool, scenario: ScenarioPreset }

impl Default for Cli {
    fn default() -> Self { Self { seed: 7, fps: 15, frames: None, mode: ViewMode::Isometric, use_color: true, cpu_substrate: false, show_minimap: false, scenario: ScenarioPreset::Normal } }
}

fn print_usage() {
    eprintln!("oNeura Terrarium Semantic Zoom Viewer\n");
    eprintln!("Usage: terrarium_zoom [OPTIONS]\n");
    eprintln!("Options:");
    eprintln!("  --seed <N>       World seed (default: 7)");
    eprintln!("  --fps <N>        Target framerate (default: 15)");
    eprintln!("  --frames <N>     Quit after N frames");
    eprintln!("  --mode <MODE>    iso/top/split/heat/dash (default: iso)");
    eprintln!("  --minimap        Show minimap");
    eprintln!("  --no-color       Disable ANSI colors");
    eprintln!("  --scenario <S>   Scenario preset: normal/drought/heat/competition/dormancy/feast/climate/amr/soil");
    eprintln!("  --cpu-substrate  Use CPU substrate\n");
    eprintln!("Semantic Zoom: 0.4-1.5x Ecosystem | 1.5-2.5x Organism | 2.5-3.5x Cellular | 3.5x+ Molecular");
    eprintln!("Scenario keys: D=drought H=heat N=normal C=competition F=feast");
}

fn parse_args() -> Result<Cli, String> {
    let mut cli = Cli::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--seed" => cli.seed = args.next().ok_or("missing --seed")?.parse().map_err(|_| "bad --seed")?,
            "--fps" => cli.fps = args.next().ok_or("missing --fps")?.parse().map_err(|_| "bad --fps")?,
            "--frames" => cli.frames = Some(args.next().ok_or("missing --frames")?.parse().map_err(|_| "bad --frames")?),
            "--mode" => cli.mode = match args.next().ok_or("missing --mode")?.as_str() {
                "iso"|"isometric" => ViewMode::Isometric, "top"|"topdown" => ViewMode::TopDown,
                "split" => ViewMode::Split, "heat"|"heatmap" => ViewMode::Heatmap,
                "dash"|"dashboard" => ViewMode::Dashboard, o => return Err(format!("unknown mode: {o}")),
            },
            "--scenario" => cli.scenario = match args.next().ok_or("missing --scenario")?.as_str() {
                "normal" => ScenarioPreset::Normal, "drought" => ScenarioPreset::Drought,
                "heat" => ScenarioPreset::Heat, "competition" => ScenarioPreset::Competition,
                "dormancy" => ScenarioPreset::Dormancy, "feast" => ScenarioPreset::Feast,
                "climate" | "climate_impact" => ScenarioPreset::Climate,
                "amr" | "amr_emergence" => ScenarioPreset::Amr,
                "soil" | "soil_health" => ScenarioPreset::Soil,
                o => return Err(format!("unknown scenario: {o}")),
            },
            "--minimap" => cli.show_minimap = true,
            "--no-color" => cli.use_color = false,
            "--cpu-substrate" => cli.cpu_substrate = true,
            "--help"|"-h" => { print_usage(); std::process::exit(0); }
            o => return Err(format!("unknown arg: {o}")),
        }
    }
    Ok(cli)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> ExitCode {
    let cli = match parse_args() { Ok(c) => c, Err(e) => { eprintln!("{e}"); print_usage(); return ExitCode::FAILURE; } };
    let mut world = match TerrariumWorld::demo(cli.seed, !cli.cpu_substrate) {
        Ok(w) => w, Err(e) => { eprintln!("failed: {e}"); return ExitCode::FAILURE; }
    };
    let mut scenario = cli.scenario;
    if scenario != ScenarioPreset::Normal { apply_scenario(&mut world, scenario); }
    let (tw, th) = terminal_size();
    let mut buf = ScreenBuffer::new(tw, th);
    let budget = if cli.fps > 0 { Some(Duration::from_secs_f64(1.0 / cli.fps as f64)) } else { None };
    let mut history: Vec<TerrariumWorldSnapshot> = Vec::with_capacity(256);
    let mut vs = ViewState::new(cli.mode, cli.show_minimap);

    let tty = open_tty(); let raw_ok = set_raw_mode();
    let key_rx = match (raw_ok, tty) {
        (true, Some(f)) => Some(spawn_key_reader(f)),
        _ => { eprintln!("Warning: no tty for interactive controls"); None }
    };

    print!("\x1b[?25l\x1b[2J"); let _ = io::stdout().flush();
    let mut fi = 0usize; let mut last_fps = 0.0f32;
    let mut fps_timer = Instant::now(); let mut fps_frames = 0usize;
    let mut quit = false;

    loop {
        let prev_zl = vs.zoom_level();

        if let Some(ref rx) = key_rx { while let Ok(key) = rx.try_recv() {
            let zl = vs.zoom_level();
            match key {
                KeyInput::Char('q') => quit = true,
                KeyInput::Char('\x1b') => { if vs.selected != SelectedEntity::None { vs.selected = SelectedEntity::None; } else { quit = true; } }
                KeyInput::Char('h') => vs.show_help = !vs.show_help,
                KeyInput::Char('\t') => vs.cycle_mode(),
                KeyInput::Char('m') => vs.show_minimap = !vs.show_minimap,
                KeyInput::Char(' ') => vs.paused = !vs.paused,
                KeyInput::Char('r') => { vs.camera_x = 0; vs.camera_y = 0; vs.zoom = 1.0; vs.selected = SelectedEntity::None; }
                KeyInput::Char('\n') => { if zl != ZoomLevel::Ecosystem { vs.selected = select_nearest(&world, &vs); } }
                KeyInput::Char('w') | KeyInput::Up => { if zl != ZoomLevel::Ecosystem { vs.cursor_y = vs.cursor_y.saturating_sub(1); } else { vs.camera_y += 2; } }
                KeyInput::Char('s') | KeyInput::Down => { if zl != ZoomLevel::Ecosystem { vs.cursor_y = (vs.cursor_y+1).min(world.config.height.saturating_sub(1)); } else { vs.camera_y -= 2; } }
                KeyInput::Char('a') | KeyInput::Left => { if zl != ZoomLevel::Ecosystem { vs.cursor_x = vs.cursor_x.saturating_sub(1); } else { vs.camera_x += 2; } }
                KeyInput::Char('d') | KeyInput::Right => { if zl != ZoomLevel::Ecosystem { vs.cursor_x = (vs.cursor_x+1).min(world.config.width.saturating_sub(1)); } else { vs.camera_x -= 2; } }
                KeyInput::Char('+') => vs.zoom = (vs.zoom + 0.2).min(5.0),
                KeyInput::Char('-') => vs.zoom = (vs.zoom - 0.2).max(0.4),
                KeyInput::Char('1') => vs.mode = ViewMode::Isometric,
                KeyInput::Char('2') => vs.mode = ViewMode::TopDown,
                KeyInput::Char('3') => vs.mode = ViewMode::Heatmap,
                KeyInput::Char('4') => vs.mode = ViewMode::Dashboard,
                KeyInput::Char('5') => vs.mode = ViewMode::Split,
                // Scenario shortcuts (uppercase = Shift)
                KeyInput::Char('D') => { scenario = ScenarioPreset::Drought; apply_scenario(&mut world, scenario); }
                KeyInput::Char('H') => { scenario = ScenarioPreset::Heat; apply_scenario(&mut world, scenario); }
                KeyInput::Char('N') => { scenario = ScenarioPreset::Normal; }
                KeyInput::Char('C') => { scenario = ScenarioPreset::Competition; apply_scenario(&mut world, scenario); }
                KeyInput::Char('F') => { scenario = ScenarioPreset::Feast; apply_scenario(&mut world, scenario); }
                _ => {}
            }
        }}

        let new_zl = vs.zoom_level();
        if prev_zl == ZoomLevel::Ecosystem && new_zl != ZoomLevel::Ecosystem && vs.selected == SelectedEntity::None {
            vs.selected = select_nearest(&world, &vs);
        }
        if new_zl == ZoomLevel::Ecosystem && prev_zl != ZoomLevel::Ecosystem { vs.selected = SelectedEntity::None; }

        if quit { break; }

        if !vs.paused {
            if let Err(e) = world.step_frame() { eprintln!("\x1b[?25h\x1b[0m\nstep failed: {e}"); restore_terminal(); return ExitCode::FAILURE; }
            apply_scenario_step(&mut world, scenario, fi);
            fi += 1;
            let snap = world.snapshot();
            if history.len() >= 200 { history.remove(0); }
            history.push(snap.clone());
            render_scene(&mut buf, &world, &snap, fi, &vs, &history, tw, th);
            fps_frames += 1;
            if fps_timer.elapsed().as_secs_f32() >= 1.0 { last_fps = fps_frames as f32 / fps_timer.elapsed().as_secs_f32(); fps_frames = 0; fps_timer = Instant::now(); }
        } else {
            if let Some(snap) = history.last() { render_scene(&mut buf, &world, snap, fi, &vs, &history, tw, th); }
        }

        let fps_str = format!(" FPS: {last_fps:.1} ");
        buf.write_str(buf.width.saturating_sub(fps_str.len()+1), 0, &fps_str, (255,255,100), (20,20,40));
        if scenario != ScenarioPreset::Normal {
            let sc_str = format!(" {} ", scenario.label());
            buf.write_str(buf.width.saturating_sub(sc_str.len()+1), 1, &sc_str, scenario.color(), (40,20,20));
        }
        if vs.show_legend { draw_legend_bar(&mut buf, &vs); }
        if vs.show_help { draw_help_overlay(&mut buf); }

        print!("{}", buf.render(cli.use_color)); let _ = io::stdout().flush();
        if let Some(max) = cli.frames { if fi >= max { break; } }
        if let Some(t) = budget { thread::sleep(if vs.paused { Duration::from_millis(50) } else { t }); }
    }

    restore_terminal();
    print!("\x1b[?25h\x1b[0m\n"); let _ = io::stdout().flush();
    println!("Terrarium zoom exited after {fi} frames.");
    ExitCode::SUCCESS
}
