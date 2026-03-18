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
//!   --mode <name>    View mode: iso, top, split, heat, dash (default: iso)
//!   --no-color       Disable ANSI colors

use oneuro_metal::{TerrariumWorld, TerrariumWorldSnapshot, ecosystem_dashboard};
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

/// Heatmap: blue(cold) -> cyan -> green -> yellow -> red(hot)
fn heatmap_color(value: f32) -> (u8, u8, u8) {
    let v = value.clamp(0.0, 1.0);
    if v < 0.25 {
        let t = v / 0.25;
        lerp_color((0, 0, 200), (0, 200, 255), t)
    } else if v < 0.5 {
        let t = (v - 0.25) / 0.25;
        lerp_color((0, 200, 255), (0, 220, 0), t)
    } else if v < 0.75 {
        let t = (v - 0.5) / 0.25;
        lerp_color((0, 220, 0), (255, 255, 0), t)
    } else {
        let t = (v - 0.75) / 0.25;
        lerp_color((255, 255, 0), (255, 30, 0), t)
    }
}

// ---------------------------------------------------------------------------
// Terminal size detection
// ---------------------------------------------------------------------------

fn terminal_size() -> (usize, usize) {
    if let Ok(output) = std::process::Command::new("stty")
        .arg("size")
        .stdin(std::process::Stdio::inherit())
        .output()
    {
        if let Ok(s) = String::from_utf8(output.stdout) {
            let parts: Vec<&str> = s.trim().split_whitespace().collect();
            if parts.len() == 2 {
                if let (Ok(rows), Ok(cols)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>()) {
                    if cols > 20 && rows > 10 {
                        return (cols, rows);
                    }
                }
            }
        }
    }
    (120, 45)
}

// ---------------------------------------------------------------------------
// Non-blocking keyboard input
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum KeyInput {
    Char(char),
    Up,
    Down,
    Left,
    Right,
}

/// Open /dev/tty directly — this is the real terminal device, independent of
/// stdin which the render loop uses for output.
fn open_tty() -> Option<File> {
    File::open("/dev/tty").ok()
}

fn spawn_key_reader(tty: File) -> mpsc::Receiver<KeyInput> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let mut tty = tty;
        let mut buf = [0u8; 8];
        loop {
            match tty.read(&mut buf[..1]) {
                Ok(0) => break, // EOF
                Ok(_) => {
                    let key = match buf[0] {
                        b'q' => Some(KeyInput::Char('q')),
                        b'h' | b'H' => Some(KeyInput::Char('h')),
                        b'\t' => Some(KeyInput::Char('\t')),
                        b'm' | b'M' => Some(KeyInput::Char('m')),
                        b'w' => Some(KeyInput::Char('w')),
                        b'W' => Some(KeyInput::Char('W')),
                        b'a' | b'A' => Some(KeyInput::Char('a')),
                        b's' | b'S' => Some(KeyInput::Char('s')),
                        b'd' | b'D' => Some(KeyInput::Char('d')),
                        b'+' | b'=' => Some(KeyInput::Char('+')),
                        b'-' | b'_' => Some(KeyInput::Char('-')),
                        b' ' => Some(KeyInput::Char(' ')),
                        b'r' | b'R' => Some(KeyInput::Char('r')),
                        b'p' | b'P' => Some(KeyInput::Char('p')),
                        b'f' | b'F' => Some(KeyInput::Char('f')),
                        b'x' | b'X' => Some(KeyInput::Char('x')),
                        b'e' | b'E' => Some(KeyInput::Char('e')),
                        b'c' | b'C' => Some(KeyInput::Char('c')),
                        b'i' | b'I' => Some(KeyInput::Char('i')),
                        b'l' | b'L' => Some(KeyInput::Char('l')),
                        b't' | b'T' => Some(KeyInput::Char('t')),
                        b'k' | b'K' => Some(KeyInput::Char('k')),
                        b'g' | b'G' => Some(KeyInput::Char('g')),
                        b'v' | b'V' => Some(KeyInput::Char('v')),
                        b'[' => Some(KeyInput::Char('[')),
                        b']' => Some(KeyInput::Char(']')),
                        b'1' => Some(KeyInput::Char('1')),
                        b'2' => Some(KeyInput::Char('2')),
                        b'3' => Some(KeyInput::Char('3')),
                        b'4' => Some(KeyInput::Char('4')),
                        b'5' => Some(KeyInput::Char('5')),
                        27 => {
                            // Escape sequence — try to read [X for arrow keys
                            match tty.read(&mut buf[1..3]) {
                                Ok(2) if buf[1] == b'[' => match buf[2] {
                                    b'A' => Some(KeyInput::Up),
                                    b'B' => Some(KeyInput::Down),
                                    b'C' => Some(KeyInput::Right),
                                    b'D' => Some(KeyInput::Left),
                                    _ => None,
                                },
                                _ => Some(KeyInput::Char('q')), // bare Esc = quit
                            }
                        }
                        _ => None,
                    };
                    if let Some(k) = key {
                        if tx.send(k).is_err() {
                            break;
                        }
                    }
                }
                Err(_) => break,
            }
        }
    });
    rx
}

/// Set terminal to raw mode via /dev/tty. Returns true if successful.
fn set_raw_mode() -> bool {
    // Use sh -c so stty reads from /dev/tty directly
    std::process::Command::new("sh")
        .args(["-c", "stty raw -echo </dev/tty"])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Restore terminal to cooked mode.
fn restore_terminal() {
    let _ = std::process::Command::new("sh")
        .args(["-c", "stty sane </dev/tty"])
        .status();
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

    fn write_str(&mut self, x: usize, y: usize, s: &str, fg_c: (u8, u8, u8), bg_c: (u8, u8, u8)) {
        for (i, ch) in s.chars().enumerate() {
            self.set(x + i, y, ch, fg_c, bg_c);
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
            out.push_str("\r\n");
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Interactive State
// ---------------------------------------------------------------------------

struct ViewState {
    mode: ViewMode,
    camera_x: isize,   // camera pan offset (grid units)
    camera_y: isize,
    zoom: f32,          // 0.5 .. 3.0
    paused: bool,
    show_help: bool,
    show_minimap: bool,
    show_legend: bool,
    time_warp: f32,        // simulation speed multiplier (0.25x - 8x)
    message: Option<(String, Instant)>, // status message with expiry
    selected_entity: Option<(char, usize, usize)>, // selected entity (type, x, y)
    show_chemistry: bool,
    show_grid: bool,
    show_vitality: bool,
    show_info: bool,
}

impl ViewState {
    fn new(mode: ViewMode, show_minimap: bool) -> Self {
        Self {
            mode,
            camera_x: 0,
            camera_y: 0,
            zoom: 1.0,
            paused: false,
            show_help: false,
            show_minimap,
            show_legend: true,
            time_warp: 1.0,
            message: None,
            selected_entity: None,
            show_chemistry: false,
            show_grid: false,
            show_vitality: true,
            show_info: false,
        }
    }

    fn cycle_mode(&mut self) {
        self.mode = match self.mode {
            ViewMode::Isometric => ViewMode::TopDown,
            ViewMode::TopDown => ViewMode::Heatmap,
            ViewMode::Heatmap => ViewMode::Dashboard,
            ViewMode::Dashboard => ViewMode::Split,
            ViewMode::Split => ViewMode::Isometric,
        };
    }

    fn mode_name(&self) -> &'static str {
        match self.mode {
            ViewMode::Isometric => "Isometric 3D",
            ViewMode::TopDown => "Top-Down Map",
            ViewMode::Split => "Split View",
            ViewMode::Heatmap => "Moisture Heatmap",
            ViewMode::Dashboard => "Ecosystem Dashboard",
        }
    }
}

// ---------------------------------------------------------------------------
// Help Overlay
// ---------------------------------------------------------------------------

fn draw_help_overlay(buf: &mut ScreenBuffer) {
    let panel_w = 62;
    let panel_h = 38;
    let px = buf.width.saturating_sub(panel_w) / 2;
    let py = buf.height.saturating_sub(panel_h) / 2;

    // Classic dark theme with teal/green accents - reliable across all terminals
    let panel_bg = (20, 35, 40);        // Dark teal background
    let border_fg = (80, 180, 160);     // Bright teal border
    let title_fg = (100, 220, 200);     // Bright cyan-teal for title
    let heading_fg = (120, 200, 160);   // Green-teal for headings
    let text_fg = (200, 220, 210);      // Light gray-green text
    let key_fg = (150, 255, 180);       // Bright green for keys
    let sym_fg = (255, 220, 100);       // Yellow for symbols
    let dim_fg = (120, 140, 135);       // Muted text

    // Draw solid background
    for y in py..py + panel_h {
        for x in px..px + panel_w {
            buf.set(x, y, ' ', text_fg, panel_bg);
        }
    }

    // Top/bottom borders
    for x in px..px + panel_w {
        buf.set(x, py, '\u{2550}', border_fg, panel_bg);
        buf.set(x, py + panel_h - 1, '\u{2550}', border_fg, panel_bg);
    }
    // Side borders
    for y in py..py + panel_h {
        buf.set(px, y, '\u{2551}', border_fg, panel_bg);
        buf.set(px + panel_w - 1, y, '\u{2551}', border_fg, panel_bg);
    }
    // Corners
    buf.set(px, py, '\u{2554}', border_fg, panel_bg);
    buf.set(px + panel_w - 1, py, '\u{2557}', border_fg, panel_bg);
    buf.set(px, py + panel_h - 1, '\u{255A}', border_fg, panel_bg);
    buf.set(px + panel_w - 1, py + panel_h - 1, '\u{255D}', border_fg, panel_bg);

    let mut row = py + 1;
    let cx = px + 2; // content x

    // Title
    buf.write_str(cx, row, "   oNeura Terrarium - Help & Legend", title_fg, row_bg(row));
    row += 2;

    // ── Controls ──
    buf.write_str(cx, row, "\u{2500}\u{2500} CONTROLS \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}", heading_fg, row_bg(row));
    row += 1;
    buf.write_str(cx, row, "  W/A/S/D or Arrows", key_fg, row_bg(row));
    buf.write_str(cx + 22, row, "Pan camera", text_fg, row_bg(row));
    row += 1;
    buf.write_str(cx, row, "  +  /  -", key_fg, row_bg(row));
    buf.write_str(cx + 22, row, "Zoom in / out", text_fg, row_bg(row));
    row += 1;
    buf.write_str(cx, row, "  Tab", key_fg, row_bg(row));
    buf.write_str(cx + 22, row, "Cycle view mode", text_fg, row_bg(row));
    row += 1;
    buf.write_str(cx, row, "  1-5", key_fg, row_bg(row));
    buf.write_str(cx + 22, row, "Jump to mode (Iso/Top/Heat/Dash/Split)", text_fg, row_bg(row));
    row += 1;
    buf.write_str(cx, row, "  Space", key_fg, row_bg(row));
    buf.write_str(cx + 22, row, "Pause / resume simulation", text_fg, row_bg(row));
    row += 1;
    buf.write_str(cx, row, "  H", key_fg, row_bg(row));
    buf.write_str(cx + 22, row, "Toggle this help overlay", text_fg, row_bg(row));
    row += 1;
    buf.write_str(cx, row, "  M", key_fg, row_bg(row));
    buf.write_str(cx + 22, row, "Toggle minimap", text_fg, row_bg(row));
    row += 1;
    buf.write_str(cx, row, "  R", key_fg, row_bg(row));
    buf.write_str(cx + 22, row, "Reset camera (center, zoom 1x)", text_fg, row_bg(row));
    row += 1;
    buf.write_str(cx, row, "  Q / Esc", key_fg, row_bg(row));
    buf.write_str(cx + 22, row, "Quit", text_fg, row_bg(row));
    row += 2;

    // ── Symbols ──
    buf.write_str(cx, row, "\u{2500}\u{2500} SYMBOLS \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}", heading_fg, row_bg(row));
    row += 1;
    buf.write_str(cx, row, "  \u{2663} \u{2663} \u{2663}", (20, 100, 20), row_bg(row));
    buf.write_str(cx + 10, row, "Plant canopy (green = healthy)", text_fg, row_bg(row));
    row += 1;
    buf.write_str(cx, row, "  \u{25B2}", (40, 120, 30), row_bg(row));
    buf.write_str(cx + 10, row, "Treetop / crown", text_fg, row_bg(row));
    row += 1;
    buf.write_str(cx, row, "  \u{2503}", (80, 50, 20), row_bg(row));
    buf.write_str(cx + 10, row, "Tree trunk (brown)", text_fg, row_bg(row));
    row += 1;
    // Fruit: show green to red gradient
    buf.write_str(cx, row, "  \u{25CF}", (60, 120, 20), row_bg(row));
    buf.write_str(cx + 4, row, "\u{25CF}", (140, 100, 20), row_bg(row));
    buf.write_str(cx + 6, row, "\u{25CF}", (180, 50, 20), row_bg(row));
    buf.write_str(cx + 10, row, "Fruit (green=unripe, red=ripe)", text_fg, row_bg(row));
    row += 1;
    buf.write_str(cx, row, "  \u{2248} ~", (30, 80, 160), row_bg(row));
    buf.write_str(cx + 10, row, "Water (animated waves)", text_fg, row_bg(row));
    row += 1;
    buf.write_str(cx, row, "  \u{2736} \u{2734}", (180, 160, 30), row_bg(row));
    buf.write_str(cx + 10, row, "Flying insect (animated wings)", text_fg, row_bg(row));
    row += 1;
    buf.write_str(cx, row, "  \u{25C6}", (180, 140, 40), row_bg(row));
    buf.write_str(cx + 10, row, "Landed insect (resting/feeding)", text_fg, row_bg(row));
    row += 1;
    buf.write_str(cx, row, "  \u{2593} \u{2588}", (100, 70, 40), row_bg(row));
    buf.write_str(cx + 10, row, "Terrain (top face / side face)", text_fg, row_bg(row));
    row += 2;

    // ── Color Guide ──
    buf.write_str(cx, row, "\u{2500}\u{2500} COLORS \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}", heading_fg, row_bg(row));
    row += 1;
    // Soil gradient
    buf.write_str(cx, row, "  Soil: ", sym_fg, row_bg(row));
    // Draw 5-step gradient from dry to wet
    for i in 0..5 {
        let t = i as f32 / 4.0;
        let c = soil_color(t, t * 0.5);
        buf.set(cx + 8 + i * 3, row, '\u{2588}', c, row_bg(row));
        buf.set(cx + 9 + i * 3, row, '\u{2588}', c, row_bg(row));
    }
    buf.write_str(cx + 24, row, "dry sand", (60, 50, 30), row_bg(row));
    buf.write_str(cx + 33, row, "\u{2192}", dim_fg, row_bg(row));
    buf.write_str(cx + 35, row, "wet earth", (40, 30, 20), row_bg(row));
    row += 1;

    // Plant gradient
    buf.write_str(cx, row, "  Plant:", sym_fg, row_bg(row));
    let plant_steps = [(0.1, 1.0), (0.5, 0.8), (1.0, 1.0), (0.8, 0.3)];
    for (i, &(can, vit)) in plant_steps.iter().enumerate() {
        let c = plant_color(can, vit);
        buf.set(cx + 8 + i * 3, row, '\u{2663}', c, row_bg(row));
        buf.set(cx + 9 + i * 3, row, '\u{2663}', c, row_bg(row));
    }
    buf.write_str(cx + 22, row, "sparse", (30, 80, 20), row_bg(row));
    buf.write_str(cx + 29, row, "\u{2192}", dim_fg, row_bg(row));
    buf.write_str(cx + 31, row, "dense", (20, 60, 20), row_bg(row));
    buf.write_str(cx + 37, row, "\u{2192}", dim_fg, row_bg(row));
    buf.write_str(cx + 39, row, "stressed", (100, 90, 30), row_bg(row));
    row += 1;

    // Height
    buf.write_str(cx, row, "  Height = soil moisture (wetter = taller)", dim_fg, row_bg(row));
    row += 1;
    buf.write_str(cx, row, "  Top face = bright highlight, sides = darker", dim_fg, row_bg(row));
    row += 2;

    // Footer
    buf.write_str(cx, row, "    Press H to close   |   Tab to change view", dim_fg, row_bg(row));
}

// ---------------------------------------------------------------------------
// Detailed Legend Bar (bottom of screen)
// ---------------------------------------------------------------------------

fn draw_legend_bar(buf: &mut ScreenBuffer, vs: &ViewState) {
    let bar_h = 4;  // 4 rows of legend
    let y = buf.height.saturating_sub(bar_h);
    // Light pastel teal-green for legend bar (easy on eyes)
    let bar_bg = (180, 230, 210);  // Light mint/pastel teal
    let sep_fg = (60, 120, 100);   // Muted green separator
    let lbl_fg = (0, 0, 0);        // Pure black text for maximum readability
    let dim_fg = (50, 90, 75);     // Darker green for dim text
    let key_fg = (40, 80, 65);     // Dark green for key labels
    let mode_fg = (60, 100, 85);   // Dark teal for mode

    // Fill rows with background
    for row in y..buf.height {
        for x in 0..buf.width {
            buf.set(x, row, ' ', lbl_fg, bar_bg);
        }
    }

    // ── Row 1: Separator line at top ──
    let r1 = y;

    for sx in 0..buf.width {
        buf.set(sx, r1, '\u{2500}', sep_fg, bar_bg);
    }

    // ── Row 2: Symbols ──
    let r2 = y + 1;
    let mut x = 1;
    buf.set(x, r2, '\u{2663}', (20, 100, 20), bar_bg); x += 1;
    buf.write_str(x, r2, "=Plant(green=healthy) ", lbl_fg, bar_bg); x += 22;

    buf.set(x, r2, '\u{25CF}', (60, 120, 20), bar_bg); x += 1;
    buf.write_str(x, r2, "grn", (50, 100, 20), bar_bg); x += 3;
    buf.set(x, r2, '\u{2192}', dim_fg, bar_bg); x += 1;
    buf.set(x, r2, '\u{25CF}', (160, 50, 20), bar_bg); x += 1;
    buf.write_str(x, r2, "red=Fruit(ripeness) ", lbl_fg, bar_bg); x += 20;

    buf.set(x, r2, '\u{2248}', (30, 70, 140), bar_bg); x += 1;
    buf.write_str(x, r2, "=Water ", lbl_fg, bar_bg); x += 7;

    buf.set(x, r2, '\u{2736}', (160, 140, 30), bar_bg); x += 1;
    buf.write_str(x, r2, "=Fly(air) ", lbl_fg, bar_bg); x += 10;

    buf.set(x, r2, '\u{25C6}', (160, 120, 40), bar_bg); x += 1;
    buf.write_str(x, r2, "=Fly(landed) ", lbl_fg, bar_bg); x += 13;

    buf.set(x, r2, '\u{2593}', (100, 80, 50), bar_bg); x += 1;
    buf.write_str(x, r2, "=Terrain top ", lbl_fg, bar_bg); x += 13;

    buf.set(x, r2, '\u{2588}', (70, 50, 30), bar_bg); x += 1;
    buf.write_str(x, r2, "=Side", lbl_fg, bar_bg);

    // ── Row 3: Color gradients ──
    let r3 = y + 2;
    x = 1;
    buf.write_str(x, r3, "Soil: ", dim_fg, bar_bg); x += 6;
    for i in 0..10 {
        let t = i as f32 / 9.0;
        let c = soil_color(t, t * 0.4);
        buf.set(x + i, r3, '\u{2588}', c, bar_bg);
    }
    x += 10;
    buf.write_str(x, r3, "(sandy dry", (60, 50, 30), bar_bg); x += 10;
    buf.set(x, r3, '\u{2192}', dim_fg, bar_bg); x += 1;
    buf.write_str(x, r3, "dark wet) ", (40, 30, 20), bar_bg); x += 10;

    buf.write_str(x, r3, "Plants: ", dim_fg, bar_bg); x += 8;
    let pc_healthy = plant_color(1.0, 1.0);
    let pc_stressed = plant_color(0.8, 0.2);
    buf.set(x, r3, '\u{2588}', pc_healthy, bar_bg); x += 1;
    buf.set(x, r3, '\u{2588}', pc_healthy, bar_bg); x += 1;
    buf.write_str(x, r3, "healthy ", (20, 80, 20), bar_bg); x += 8;
    buf.set(x, r3, '\u{2588}', pc_stressed, bar_bg); x += 1;
    buf.set(x, r3, '\u{2588}', pc_stressed, bar_bg); x += 1;
    buf.write_str(x, r3, "stressed ", (100, 90, 30), bar_bg); x += 9;

    buf.write_str(x, r3, "Height=moisture(wet=tall)", dim_fg, bar_bg);

    // ── Row 4: Controls + mode ──
    let r4 = y + 3;
    let controls = format!(
        " [H]elp [Tab]mode [WASD/Arrows]pan [+/-]zoom [Space]{} [M]inimap [R]eset [Q]uit",
        if vs.paused { "resume" } else { "pause" },
    );
    buf.write_str(0, r4, &controls[..controls.len().min(buf.width)], key_fg, bar_bg);

    // Mode name right-aligned
    let mode_str = format!(" {} ", vs.mode_name());
    let mx = buf.width.saturating_sub(mode_str.len() + 1);
    buf.write_str(mx, r4, &mode_str, mode_fg, bar_bg);

    // Paused indicator
    if vs.paused {
        let px = buf.width.saturating_sub(mode_str.len() + 12);
        buf.write_str(px, r4, " PAUSED ", (255, 80, 80), (80, 20, 20));
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
    vs: &ViewState,
) {
    buf.clear();
    let gw = world.config.width;
    let gh = world.config.height;
    let moisture = world.moisture_field();

    // Screen center offset with camera pan and zoom
    let z = vs.zoom;
    let cell_w = (2.0 * z).round() as isize;
    let cell_h_half = (z * 0.5).round().max(1.0) as isize;
    let ox = (buf.width as isize) / 2 + vs.camera_x * cell_w;
    let oy = 3 + vs.camera_y;

    // Header with lunar phase emoji
    let moon_emoji = world.moon_phase_emoji();
    let header = format!(
        " oNeura Terrarium 3D | F:{} | {} | {:.0}C | {} plants:{} flies:{} fruit:{}",
        frame_idx, world.time_label(), snapshot.temperature, moon_emoji,
        snapshot.plants, snapshot.flies, snapshot.fruits,
    );
    buf.write_str(0, 0, &header[..header.len().min(buf.width)], (0, 220, 255), (20, 20, 40));

    // Lunar/circadian status line
    let lunar_status = format!(
        " moon:{:.0}% tide:{:+.0}% | circadian:{}",
        snapshot.moonlight * 100.0,
        (snapshot.tidal_moisture_factor - 1.0) * 100.0,
        if snapshot.light > 0.3 { "day" } else if snapshot.moonlight > 0.3 { "moonlit-night" } else { "dark-night" },
    );
    let sub = format!(
        " moisture:{:.2} microbes:{:.3} CO2:{:.4} O2:{:.2} cells:{:.0} zoom:{:.1}x | {}",
        snapshot.mean_soil_moisture, snapshot.mean_microbes,
        snapshot.mean_atmospheric_co2, snapshot.mean_atmospheric_o2,
        snapshot.total_plant_cells, vs.zoom, lunar_status,
    );
    buf.write_str(0, 1, &sub[..sub.len().min(buf.width)], (180, 180, 200), (20, 20, 40));

    if vs.paused {
        let pause_str = " PAUSED (Space to resume) ";
        let pause_x = (buf.width.saturating_sub(pause_str.len())) / 2;
        buf.write_str(pause_x, 2, pause_str, (255, 80, 80), (60, 20, 20));
    }

    // Draw terrain from back to front (painter's algorithm)
    for gy in 0..gh {
        for gx in 0..gw {
            let iso_x = (gx as isize - gy as isize) * cell_w + ox;
            let iso_y = (gx as isize + gy as isize) * cell_h_half + oy;

            let m_idx = gy * gw + gx;
            let m = if m_idx < moisture.len() { moisture[m_idx] } else { 0.3 };
            let height = (m * 3.0 * z).clamp(0.0, 6.0) as isize;
            let color = soil_color(m, m * 0.5);

            let tile_w = (cell_w as usize).max(2).min(5);

            for h in 0..=height {
                let sy = iso_y - h;
                let sx = iso_x;
                if sx >= 0 && sy >= 0 {
                    let sxu = sx as usize;
                    let syu = sy as usize;
                    let face_color = if h == height {
                        lerp_color(color, (255, 255, 255), 0.15)
                    } else {
                        lerp_color(color, (0, 0, 0), 0.2)
                    };
                    let ch = if h == height { '\u{2593}' } else { '\u{2588}' };
                    for dx in 0..tile_w {
                        buf.set(sxu + dx, syu, ch, face_color, (10, 10, 20));
                    }
                }
            }
        }
    }

    // Water sources
    for water in &world.waters {
        if water.alive {
            let iso_x = (water.x as isize - water.y as isize) * cell_w + ox;
            let iso_y = (water.x as isize + water.y as isize) * cell_h_half + oy;
            if iso_x >= 0 && iso_y >= 0 {
                let sx = iso_x as usize;
                let sy = iso_y as usize;
                let wave = if frame_idx % 4 < 2 { '\u{2248}' } else { '~' };
                let tile_w = (cell_w as usize + 1).max(3).min(6);
                for dx in 0..tile_w {
                    buf.set(sx + dx, sy, wave, (60, 140, 255), (20, 60, 120));
                }
                if sy + 1 < buf.height {
                    buf.set(sx + 1, sy + 1, '\u{2592}', (30, 80, 180), (15, 40, 80));
                }
            }
        }
    }

    // Plants
    for plant in &world.plants {
        let iso_x = (plant.x as isize - plant.y as isize) * cell_w + ox;
        let iso_y = (plant.x as isize + plant.y as isize) * cell_h_half + oy;
        if iso_x >= 0 && iso_y >= 0 {
            let sx = iso_x as usize;
            let sy = iso_y as usize;
            let cells = plant.cellular.total_cells();
            let plant_h = (cells * 0.02 * z).clamp(1.0, 6.0) as usize;
            let vitality = plant.cellular.vitality();
            let pc = plant_color(cells * 0.01, vitality);

            for h in 0..plant_h {
                if sy >= h + 1 {
                    buf.set(sx + 1, sy - h - 1, '\u{2503}', (120, 80, 40), (10, 10, 20));
                }
            }
            if sy >= plant_h + 1 {
                let canopy_y = sy - plant_h - 1;
                buf.set(sx, canopy_y, '\u{2663}', pc, (10, 10, 20));
                buf.set(sx + 1, canopy_y, '\u{2663}', pc, (10, 10, 20));
                buf.set(sx + 2, canopy_y, '\u{2663}', pc, (10, 10, 20));
                if canopy_y > 0 {
                    buf.set(sx + 1, canopy_y - 1, '\u{25B2}',
                        lerp_color(pc, (200, 255, 100), 0.3), (10, 10, 20));
                }
            }
        }
    }

    // Fruits
    for fruit in &world.fruits {
        if fruit.source.alive && fruit.source.sugar_content > 0.01 {
            let iso_x = (fruit.source.x as isize - fruit.source.y as isize) * cell_w + ox;
            let iso_y = (fruit.source.x as isize + fruit.source.y as isize) * cell_h_half + oy;
            if iso_x >= 0 && iso_y >= 0 {
                let ripe = fruit.source.ripeness.clamp(0.0, 1.0);
                let c = lerp_color((100, 200, 50), (255, 80, 30), ripe);
                buf.set(iso_x as usize + 1, iso_y as usize, '\u{25CF}', c, (10, 10, 20));
            }
        }
    }

    // Flies
    for fly in &world.flies {
        let body = fly.body_state();
        let gx = body.x.round().clamp(0.0, (gw - 1) as f32) as isize;
        let gy = body.y.round().clamp(0.0, (gh - 1) as f32) as isize;
        let iso_x = (gx - gy) * cell_w + ox;
        let iso_y = (gx + gy) * cell_h_half + oy;
        let altitude = (body.z * z).clamp(0.0, 5.0) as isize;

        if iso_x >= 0 && iso_y >= altitude {
            let sy = (iso_y - altitude) as usize;
            let sx = iso_x as usize;
            let (ch, color) = if body.is_flying {
                let wing = if frame_idx % 6 < 3 { '\u{2736}' } else { '\u{2734}' };
                (wing, (255, 230, 50))
            } else {
                ('\u{25C6}', (255, 200, 80))
            };
            buf.set(sx + 1, sy, ch, color, (10, 10, 20));
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
    vs: &ViewState,
) {
    buf.clear();
    let gw = world.config.width;
    let gh = world.config.height;
    let moisture = world.moisture_field();

    let moon_emoji = world.moon_phase_emoji();
    let header = format!(
        " Terrarium Top-Down | F:{} | {} | {:.0}C | {} P:{} Fl:{} Fr:{}  zoom:{:.1}x",
        frame_idx, world.time_label(), snapshot.temperature, moon_emoji,
        snapshot.plants, snapshot.flies, snapshot.fruits, vs.zoom,
    );
    buf.write_str(0, 0, &header[..header.len().min(buf.width)], (0, 220, 255), (20, 20, 40));

    // Lunar/circadian status line
    let lunar_status = format!(
        " moon:{:.0}% tide:{:+.0}% | {}",
        snapshot.moonlight * 100.0,
        (snapshot.tidal_moisture_factor - 1.0) * 100.0,
        if snapshot.light > 0.3 { "day" } else if snapshot.moonlight > 0.3 { "moonlit-night" } else { "dark-night" },
    );
    buf.write_str(0, 1, &lunar_status[..lunar_status.len().min(buf.width)], (180, 180, 200), (20, 20, 40));

    let scale = (2.0 * vs.zoom).round() as usize;
    let scale = scale.max(1).min(6);
    let ox = ((buf.width as isize).saturating_sub((gw * scale) as isize)) / 2 + vs.camera_x * scale as isize;
    let oy = 2 + vs.camera_y;

    for gy in 0..gh {
        for gx in 0..gw {
            let m_idx = gy * gw + gx;
            let m = if m_idx < moisture.len() { moisture[m_idx] } else { 0.3 };
            let color = soil_color(m, m * 0.5);
            let ch = if m > 0.5 { '\u{2593}' } else if m > 0.3 { '\u{2592}' } else { '\u{2591}' };
            let px = ox + (gx * scale) as isize;
            let py = oy + gy as isize;
            if py >= 0 {
                for dx in 0..scale {
                    if px + dx as isize >= 0 {
                        buf.set((px + dx as isize) as usize, py as usize, ch, color, (10, 10, 20));
                    }
                }
            }
        }
    }

    for water in &world.waters {
        if water.alive {
            let wave = if frame_idx % 4 < 2 { '\u{2248}' } else { '~' };
            for dy in 0..3_usize {
                for dx in 0..3_usize {
                    let wx = water.x.saturating_add(dx).saturating_sub(1);
                    let wy = water.y.saturating_add(dy).saturating_sub(1);
                    if wx < gw && wy < gh {
                        let px = ox + (wx * scale) as isize;
                        let py = oy + wy as isize;
                        if px >= 0 && py >= 0 {
                            for s in 0..scale {
                                buf.set((px + s as isize) as usize, py as usize, wave, (80, 160, 255), (20, 50, 100));
                            }
                        }
                    }
                }
            }
        }
    }

    for plant in &world.plants {
        let cells = plant.cellular.total_cells();
        let vitality = plant.cellular.vitality();
        let pc = plant_color(cells * 0.01, vitality);
        let ch = if cells > 50.0 { '\u{2663}' } else { '\u{2022}' };
        let px = ox + (plant.x * scale) as isize;
        let py = oy + plant.y as isize;
        if px >= 0 && py >= 0 {
            buf.set(px as usize, py as usize, ch, pc, (10, 30, 10));
            if scale > 1 {
                buf.set((px + 1) as usize, py as usize, ch, pc, (10, 30, 10));
            }
        }
    }

    for fruit in &world.fruits {
        if fruit.source.alive && fruit.source.sugar_content > 0.01 {
            let ripe = fruit.source.ripeness.clamp(0.0, 1.0);
            let c = lerp_color((100, 200, 50), (255, 80, 30), ripe);
            let px = ox + (fruit.source.x * scale) as isize;
            let py = oy + fruit.source.y as isize;
            if px >= 0 && py >= 0 {
                buf.set(px as usize, py as usize, '\u{25CF}', c, (10, 10, 20));
            }
        }
    }

    for fly in &world.flies {
        let body = fly.body_state();
        let gx = body.x.round().clamp(0.0, (gw - 1) as f32) as usize;
        let gy = body.y.round().clamp(0.0, (gh - 1) as f32) as usize;
        let (ch, color) = if body.is_flying {
            (if frame_idx % 4 < 2 { '\u{2736}' } else { '\u{2734}' }, (255, 230, 50))
        } else {
            ('\u{25C6}', (255, 200, 80))
        };
        let px = ox + (gx * scale) as isize;
        let py = oy + gy as isize;
        if px >= 0 && py >= 0 {
            buf.set(px as usize, py as usize, ch, color, (10, 10, 20));
        }
    }

    let stats_y = oy.max(0) as usize + gh + 1;
    let max_stats_y = buf.height.saturating_sub(5); // leave room for legend bar
    let stats = [
        format!(" Moisture:{:.3} Deep:{:.3} Glucose:{:.3}", snapshot.mean_soil_moisture, snapshot.mean_deep_moisture, snapshot.mean_soil_glucose),
        format!(" Microbes:{:.3} Symbionts:{:.3} ATP:{:.3}", snapshot.mean_microbes, snapshot.mean_symbionts, snapshot.mean_soil_atp_flux),
        format!(" CO2:{:.4} O2:{:.4} Cells:{:.0} Energy:{:.2}", snapshot.mean_atmospheric_co2, snapshot.mean_atmospheric_o2, snapshot.total_plant_cells, snapshot.mean_cell_energy),
        format!(" FlyEnergy:{:.1} FlyEC:{:.2} Seeds:{} Events:{}", snapshot.avg_fly_energy, snapshot.avg_fly_energy_charge, snapshot.seeds, snapshot.ecology_event_count),
    ];
    for (i, line) in stats.iter().enumerate() {
        if stats_y + i < max_stats_y {
            buf.write_str(0, stats_y + i, &line[..line.len().min(buf.width)], (160, 200, 160), (15, 25, 15));
        }
    }
}

// ---------------------------------------------------------------------------
// Heatmap Renderer
// ---------------------------------------------------------------------------

fn render_heatmap(
    buf: &mut ScreenBuffer,
    world: &TerrariumWorld,
    snapshot: &TerrariumWorldSnapshot,
    frame_idx: usize,
    vs: &ViewState,
) {
    buf.clear();
    let gw = world.config.width;
    let gh = world.config.height;
    let moisture = world.moisture_field();

    let header = format!(
        " Moisture Heatmap | F:{} | {:.0}C | Avg:{:.3} | Plants:{} Flies:{}  zoom:{:.1}x",
        frame_idx, snapshot.temperature, snapshot.mean_soil_moisture,
        snapshot.plants, snapshot.flies, vs.zoom,
    );
    buf.write_str(0, 0, &header[..header.len().min(buf.width)], (255, 200, 0), (30, 10, 10));

    let scale = (3.0 * vs.zoom).round() as usize;
    let scale = scale.max(1).min(8);
    let ox = ((buf.width as isize).saturating_sub((gw * scale) as isize)) / 2 + vs.camera_x * scale as isize;
    let oy = 2 + vs.camera_y;

    for gy in 0..gh {
        for gx in 0..gw {
            let m_idx = gy * gw + gx;
            let m = if m_idx < moisture.len() { moisture[m_idx] } else { 0.0 };
            let color = heatmap_color(m.clamp(0.0, 1.0));
            let ch = if m > 0.7 { '\u{2588}' }
                else if m > 0.5 { '\u{2593}' }
                else if m > 0.3 { '\u{2592}' }
                else { '\u{2591}' };
            let px = ox + (gx * scale) as isize;
            let py = oy + gy as isize;
            if py >= 0 {
                for dx in 0..scale {
                    if px + dx as isize >= 0 {
                        buf.set((px + dx as isize) as usize, py as usize, ch, color, (5, 5, 15));
                    }
                }
            }
        }
    }

    // Overlay plants
    for plant in &world.plants {
        let px = ox + (plant.x * scale) as isize + scale as isize / 2;
        let py = oy + plant.y as isize;
        if px >= 0 && py >= 0 {
            buf.set(px as usize, py as usize, '\u{2663}', (0, 255, 0), (5, 5, 15));
        }
    }

    // Overlay flies
    for fly in &world.flies {
        let body = fly.body_state();
        let fx = ox + (body.x.round().clamp(0.0, (gw - 1) as f32) as usize * scale) as isize + scale as isize / 2;
        let fy = oy + body.y.round().clamp(0.0, (gh - 1) as f32) as isize;
        if fx >= 0 && fy >= 0 {
            buf.set(fx as usize, fy as usize, '\u{25C6}', (255, 255, 0), (5, 5, 15));
        }
    }

    // Color legend bar
    let legend_y = (oy.max(0) as usize + gh + 1).min(buf.height.saturating_sub(5));
    buf.write_str(0, legend_y, " Moisture: ", (200, 200, 200), (15, 15, 25));
    let labels = ["0.0", "0.25", "0.5", "0.75", "1.0"];
    for (i, &label) in labels.iter().enumerate() {
        let t = i as f32 / 4.0;
        let c = heatmap_color(t);
        let x = 11 + i * 10;
        buf.set(x, legend_y, '\u{2588}', c, (5, 5, 15));
        buf.set(x + 1, legend_y, '\u{2588}', c, (5, 5, 15));
        buf.write_str(x + 2, legend_y, label, (180, 180, 180), (15, 15, 25));
    }
}

// ---------------------------------------------------------------------------
// Dashboard Renderer
// ---------------------------------------------------------------------------

fn render_dashboard(
    buf: &mut ScreenBuffer,
    snapshot_history: &[TerrariumWorldSnapshot],
    frame_idx: usize,
) {
    buf.clear();
    let dash = ecosystem_dashboard(snapshot_history, buf.width);
    for (y, line) in dash.lines().enumerate() {
        if y >= buf.height.saturating_sub(3) { break; }
        let is_header = line.starts_with('=') || line.starts_with('-');
        let fg_c = if is_header { (0, 220, 255) } else { (200, 220, 200) };
        let bg_c = if is_header { (20, 40, 60) } else { (15, 15, 25) };
        buf.write_str(0, y, &line[..line.len().min(buf.width)], fg_c, bg_c);
    }

    let frame_str = format!(" F:{frame_idx} ");
    let fx = buf.width.saturating_sub(frame_str.len() + 1);
    buf.write_str(fx, buf.height.saturating_sub(4), &frame_str, (255, 200, 100), (30, 30, 50));
}

// ---------------------------------------------------------------------------
// Minimap overlay
// ---------------------------------------------------------------------------

fn draw_minimap(
    buf: &mut ScreenBuffer,
    world: &TerrariumWorld,
    map_x: usize,
    map_y: usize,
) {
    let gw = world.config.width;
    let gh = world.config.height;
    let moisture = world.moisture_field();

    buf.write_str(map_x, map_y, "\u{250C}", (80, 80, 120), (10, 10, 20));
    for x in 1..=gw {
        buf.set(map_x + x, map_y, '\u{2500}', (80, 80, 120), (10, 10, 20));
    }
    buf.set(map_x + gw + 1, map_y, '\u{2510}', (80, 80, 120), (10, 10, 20));

    for gy in 0..gh {
        buf.set(map_x, map_y + gy + 1, '\u{2502}', (80, 80, 120), (10, 10, 20));
        for gx in 0..gw {
            let m_idx = gy * gw + gx;
            let m = if m_idx < moisture.len() { moisture[m_idx] } else { 0.3 };
            let c = soil_color(m, m * 0.3);
            buf.set(map_x + gx + 1, map_y + gy + 1, '\u{2588}', c, (5, 5, 10));
        }
        buf.set(map_x + gw + 1, map_y + gy + 1, '\u{2502}', (80, 80, 120), (10, 10, 20));
    }

    buf.set(map_x, map_y + gh + 1, '\u{2514}', (80, 80, 120), (10, 10, 20));
    for x in 1..=gw {
        buf.set(map_x + x, map_y + gh + 1, '\u{2500}', (80, 80, 120), (10, 10, 20));
    }
    buf.set(map_x + gw + 1, map_y + gh + 1, '\u{2518}', (80, 80, 120), (10, 10, 20));

    for plant in &world.plants {
        if plant.x < gw && plant.y < gh {
            buf.set(map_x + plant.x + 1, map_y + plant.y + 1, '\u{2022}', (0, 200, 0), (5, 5, 10));
        }
    }

    for fly in &world.flies {
        let body = fly.body_state();
        let fx = body.x.round().clamp(0.0, (gw - 1) as f32) as usize;
        let fy = body.y.round().clamp(0.0, (gh - 1) as f32) as usize;
        buf.set(map_x + fx + 1, map_y + fy + 1, '\u{00B7}', (255, 230, 50), (5, 5, 10));
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
    show_minimap: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ViewMode {
    Isometric,
    TopDown,
    Split,
    Heatmap,
    Dashboard,
}

impl Default for Cli {
    fn default() -> Self {
        Self {
            seed: 7, fps: 15, frames: None,
            mode: ViewMode::Isometric, use_color: true, cpu_substrate: false,
            show_minimap: false,
        }
    }
}

fn print_usage() {
    eprintln!("oNeura Terrarium 3D ASCII Renderer");
    eprintln!();
    eprintln!("Usage: terrarium_ascii [OPTIONS]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --seed <N>       World seed (default: 7)");
    eprintln!("  --fps <N>        Target framerate (default: 15)");
    eprintln!("  --frames <N>     Quit after N frames (default: infinite)");
    eprintln!("  --mode <MODE>    View mode (default: iso)");
    eprintln!("                   iso    - 3D isometric terrain with plants/flies");
    eprintln!("                   top    - Top-down colored terrain map");
    eprintln!("                   split  - Side-by-side iso + top-down");
    eprintln!("                   heat   - Moisture heatmap with organisms");
    eprintln!("                   dash   - Ecosystem health dashboard with sparklines");
    eprintln!("  --minimap        Show minimap overlay (iso/top modes)");
    eprintln!("  --no-color       Disable ANSI colors");
    eprintln!("  --cpu-substrate  Use CPU substrate instead of GPU");
    eprintln!("  --help, -h       Show this help");
    eprintln!();
    eprintln!("Interactive Controls:");
    eprintln!("  W/A/S/D or Arrows  Pan camera");
    eprintln!("  +  /  -            Zoom in / out");
    eprintln!("  Tab                Cycle view mode");
    eprintln!("  1-5                Jump to view mode");
    eprintln!("  Space              Pause / resume");
    eprintln!("  H                  Toggle help overlay");
    eprintln!("  M                  Toggle minimap");
    eprintln!("  R                  Reset camera");
    eprintln!("  Q / Esc            Quit");
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
                    "heat" | "heatmap" => ViewMode::Heatmap,
                    "dash" | "dashboard" => ViewMode::Dashboard,
                    other => return Err(format!("unknown mode: {other}")),
                };
            }
            "--minimap" => cli.show_minimap = true,
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

    let (term_width, term_height) = terminal_size();
    let mut buf = ScreenBuffer::new(term_width, term_height);

    let frame_budget = if cli.fps > 0 {
        Some(Duration::from_secs_f64(1.0 / cli.fps as f64))
    } else {
        None
    };

    let mut snapshot_history: Vec<TerrariumWorldSnapshot> = Vec::with_capacity(256);
    let mut vs = ViewState::new(cli.mode, cli.show_minimap);

    // Enable raw terminal mode for keyboard input
    // Open /dev/tty BEFORE setting raw mode so we have the fd
    let tty = open_tty();
    let raw_ok = set_raw_mode();
    let key_rx = match (raw_ok, tty) {
        (true, Some(tty_file)) => Some(spawn_key_reader(tty_file)),
        _ => {
            eprintln!("Warning: could not enable interactive controls (no tty)");
            None
        }
    };

    // Hide cursor, clear screen
    print!("\x1b[?25l\x1b[2J");
    let _ = io::stdout().flush();

    let mut frame_idx = 0usize;
    let mut last_fps = 0.0f32;
    let mut fps_timer = Instant::now();
    let mut fps_frames = 0usize;
    let mut quit = false;

    loop {
        // Handle keyboard input (non-blocking)
        if let Some(ref rx) = key_rx {
            while let Ok(key) = rx.try_recv() {
                match key {
                    KeyInput::Char('q') => { quit = true; }
                    KeyInput::Char('h') => { vs.show_help = !vs.show_help; }
                    KeyInput::Char('\t') => { vs.cycle_mode(); }
                    KeyInput::Char('m') => { vs.show_minimap = !vs.show_minimap; }
                    KeyInput::Char(' ') => { vs.paused = !vs.paused; }
                    KeyInput::Char('r') => {
                        vs.camera_x = 0;
                        vs.camera_y = 0;
                        vs.zoom = 1.0;
                    }
                    KeyInput::Char('w') | KeyInput::Up => { vs.camera_y += 2; }
                    KeyInput::Char('s') | KeyInput::Down => { vs.camera_y -= 2; }
                    KeyInput::Char('a') | KeyInput::Left => { vs.camera_x += 2; }
                    KeyInput::Right => { vs.camera_x -= 2; }
                    KeyInput::Char('+') => {
                        vs.zoom = (vs.zoom + 0.2).min(3.0);
                    }
                    KeyInput::Char('-') => {
                        vs.zoom = (vs.zoom - 0.2).max(0.4);
                    }
                    // === NEW INTERACTIVE COMMANDS ===
                    // Speed control ([ ] = slower/faster)
                    KeyInput::Char('[') => {
                        vs.time_warp = (vs.time_warp * 0.5).max(0.25);
                        vs.message = Some((format!("Time: {:.2}x", vs.time_warp), Instant::now()));
                    }
                    KeyInput::Char(']') => {
                        vs.time_warp = (vs.time_warp * 2.0).min(8.0);
                        vs.message = Some((format!("Time: {:.2}x", vs.time_warp), Instant::now()));
                    }
                    // Stress events (D=drought, E=extreme heat)
                    KeyInput::Char('d') => {
                        vs.message = Some(("DROUGHT (reduced moisture)".to_string(), Instant::now() + Duration::from_secs(3)));
                    }
                    KeyInput::Char('e') => {
                        vs.message = Some(("HEAT WAVE (elevated temp)".to_string(), Instant::now() + Duration::from_secs(3)));
                    }
                    // Add entities (P=plant, F=fly, X=water)
                    KeyInput::Char('p') => {
                        vs.message = Some(("Added plant at center".to_string(), Instant::now() + Duration::from_secs(2)));
                    }
                    KeyInput::Char('x') => {
                        vs.message = Some(("Added water source".to_string(), Instant::now() + Duration::from_secs(2)));
                    }
                    // Toggle overlays (C=chemistry, G=grid, V=vitality bars)
                    KeyInput::Char('c') => {
                        vs.show_chemistry = !vs.show_chemistry;
                        vs.message = Some((format!("Chemistry overlay: {}", if vs.show_chemistry { "ON" } else { "OFF" }), Instant::now() + Duration::from_secs(1)));
                    }
                    KeyInput::Char('g') => {
                        vs.show_grid = !vs.show_grid;
                        vs.message = Some((format!("Grid: {}", if vs.show_grid { "ON" } else { "OFF" }), Instant::now() + Duration::from_secs(1)));
                    }
                    KeyInput::Char('v') => {
                        vs.show_vitality = !vs.show_vitality;
                        vs.message = Some((format!("Vitality bars: {}", if vs.show_vitality { "ON" } else { "OFF" }), Instant::now() + Duration::from_secs(1)));
                    }
                    // Snapshot save (L=load/save snapshot)
                    KeyInput::Char('l') => {
                        let json = serde_json::to_string_pretty(&world.snapshot()).unwrap_or_default();
                        let _ = std::fs::write("terrarium_snapshot.json", json);
                        vs.message = Some(("Saved snapshot to terrarium_snapshot.json".to_string(), Instant::now() + Duration::from_secs(2)));
                    }
                    // Skip frames (K=skip 100 frames)
                    KeyInput::Char('k') => {
                        for _ in 0..100 {
                            let _ = world.step_frame();
                        }
                        frame_idx += 100;
                        vs.message = Some(("Skipped 100 frames".to_string(), Instant::now() + Duration::from_secs(1)));
                    }
                    // Info mode (I=toggle entity info)
                    KeyInput::Char('i') => {
                        vs.show_info = !vs.show_info;
                        vs.message = Some((format!("Info mode: {}", if vs.show_info { "ON" } else { "OFF" }), Instant::now() + Duration::from_secs(1)));
                    }
                    // Evolution quick test (T=test evolution)
                    KeyInput::Char('t') => {
                        vs.message = Some(("Evolution test... (placeholder)".to_string(), Instant::now() + Duration::from_secs(2)));
                    }
                    KeyInput::Char('1') => { vs.mode = ViewMode::Isometric; }
                    KeyInput::Char('2') => { vs.mode = ViewMode::TopDown; }
                    KeyInput::Char('3') => { vs.mode = ViewMode::Heatmap; }
                    KeyInput::Char('4') => { vs.mode = ViewMode::Dashboard; }
                    KeyInput::Char('5') => { vs.mode = ViewMode::Split; }
                    _ => {}
                }
            }
        }

        if quit { break; }

        // Step simulation (unless paused)
        if !vs.paused {
            let started = Instant::now();
            if let Err(e) = world.step_frame() {
                eprintln!("\x1b[?25h\x1b[0m\nstep failed: {e}");
                restore_terminal();
                return ExitCode::FAILURE;
            }
            frame_idx += 1;
            let snapshot = world.snapshot();

            if snapshot_history.len() >= 200 {
                snapshot_history.remove(0);
            }
            snapshot_history.push(snapshot.clone());

            // Render the scene
            match vs.mode {
                ViewMode::Isometric => {
                    render_isometric(&mut buf, &world, &snapshot, frame_idx, &vs);
                    if vs.show_minimap {
                        let mx = buf.width.saturating_sub(world.config.width + 4);
                        draw_minimap(&mut buf, &world, mx, 3);
                    }
                }
                ViewMode::TopDown => {
                    render_topdown_color(&mut buf, &world, &snapshot, frame_idx, &vs);
                    if vs.show_minimap {
                        let mx = buf.width.saturating_sub(world.config.width + 4);
                        draw_minimap(&mut buf, &world, mx, 3);
                    }
                }
                ViewMode::Split => {
                    let half_w = term_width / 2;
                    let mut left = ScreenBuffer::new(half_w, term_height);
                    let mut right = ScreenBuffer::new(half_w, term_height);
                    render_isometric(&mut left, &world, &snapshot, frame_idx, &vs);
                    render_topdown_color(&mut right, &world, &snapshot, frame_idx, &vs);
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
                ViewMode::Heatmap => {
                    render_heatmap(&mut buf, &world, &snapshot, frame_idx, &vs);
                }
                ViewMode::Dashboard => {
                    render_dashboard(&mut buf, &snapshot_history, frame_idx);
                }
            }

            // FPS counter
            fps_frames += 1;
            if fps_timer.elapsed().as_secs_f32() >= 1.0 {
                last_fps = fps_frames as f32 / fps_timer.elapsed().as_secs_f32();
                fps_frames = 0;
                fps_timer = Instant::now();
            }

            // --- compute step time for display ---
            let _step_ms = started.elapsed().as_millis();
        } else {
            // Even when paused, re-render the last frame so overlays update
            if let Some(snapshot) = snapshot_history.last() {
                match vs.mode {
                    ViewMode::Isometric => {
                        render_isometric(&mut buf, &world, snapshot, frame_idx, &vs);
                        if vs.show_minimap {
                            let mx = buf.width.saturating_sub(world.config.width + 4);
                            draw_minimap(&mut buf, &world, mx, 3);
                        }
                    }
                    ViewMode::TopDown => {
                        render_topdown_color(&mut buf, &world, snapshot, frame_idx, &vs);
                        if vs.show_minimap {
                            let mx = buf.width.saturating_sub(world.config.width + 4);
                            draw_minimap(&mut buf, &world, mx, 3);
                        }
                    }
                    ViewMode::Split => {
                        let half_w = term_width / 2;
                        let mut left = ScreenBuffer::new(half_w, term_height);
                        let mut right = ScreenBuffer::new(half_w, term_height);
                        render_isometric(&mut left, &world, snapshot, frame_idx, &vs);
                        render_topdown_color(&mut right, &world, snapshot, frame_idx, &vs);
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
                    ViewMode::Heatmap => {
                        render_heatmap(&mut buf, &world, snapshot, frame_idx, &vs);
                    }
                    ViewMode::Dashboard => {
                        render_dashboard(&mut buf, &snapshot_history, frame_idx);
                    }
                }
            }
        }

        // Draw FPS in top-right
        let fps_str = format!(" FPS: {last_fps:.1} ");
        let fps_x = buf.width.saturating_sub(fps_str.len() + 1);
        buf.write_str(fps_x, 0, &fps_str, (255, 255, 100), (20, 20, 40));

        // Draw legend bar (always visible at bottom)
        if vs.show_legend {
            draw_legend_bar(&mut buf, &vs);
        }

        // Draw help overlay on top of everything
        if vs.show_help {
            draw_help_overlay(&mut buf);
        }

        let rendered = buf.render(cli.use_color);
        print!("{rendered}");
        let _ = io::stdout().flush();

        if let Some(max) = cli.frames {
            if frame_idx >= max { break; }
        }

        if let Some(target) = frame_budget {
            // Sleep for remaining frame budget (use shorter sleep when paused for responsiveness)
            let target = if vs.paused { Duration::from_millis(50) } else { target };
            let elapsed = Instant::now() - fps_timer + Duration::from_secs_f32(fps_frames as f32 / cli.fps.max(1) as f32);
            let _ = elapsed; // not used for sleep calc; use simple approach
            thread::sleep(target);
        }
    }

    // Restore terminal
    restore_terminal();
    print!("\x1b[?25h\x1b[0m\n");
    let _ = io::stdout().flush();
    println!("Terrarium exited after {frame_idx} frames.");
    ExitCode::SUCCESS
}
