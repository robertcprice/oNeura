use font8x8::{UnicodeFonts, BASIC_FONTS};
use minifb::{Key, KeyRepeat, Window, WindowOptions};
use oneuro_metal::{TerrariumTopdownView, TerrariumWorld, TerrariumWorldSnapshot};
use std::env;
use std::process::ExitCode;
use std::thread;
use std::time::{Duration, Instant};

const PANEL_W: usize = 280;

#[derive(Debug, Clone)]
struct Cli {
    seed: u64,
    fps: u64,
    frames: Option<usize>,
    cpu_substrate: bool,
    cell_px: usize,
}

impl Default for Cli {
    fn default() -> Self {
        Self {
            seed: 7,
            fps: 30,
            frames: None,
            cpu_substrate: false,
            cell_px: 16,
        }
    }
}

fn print_usage() {
    eprintln!(
        "Usage: cargo run --release --bin terrarium_viewer -- [--seed <n>] [--fps <n>] [--frames <n>] [--cell <px>] [--cpu-substrate]"
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
                    .parse()
                    .map_err(|_| "invalid integer for --seed".to_string())?;
            }
            "--fps" => {
                cli.fps = args
                    .next()
                    .ok_or_else(|| "missing value for --fps".to_string())?
                    .parse()
                    .map_err(|_| "invalid integer for --fps".to_string())?;
            }
            "--frames" => {
                cli.frames = Some(
                    args.next()
                        .ok_or_else(|| "missing value for --frames".to_string())?
                        .parse()
                        .map_err(|_| "invalid integer for --frames".to_string())?,
                );
            }
            "--cell" => {
                cli.cell_px = args
                    .next()
                    .ok_or_else(|| "missing value for --cell".to_string())?
                    .parse()
                    .map_err(|_| "invalid integer for --cell".to_string())?;
                cli.cell_px = cli.cell_px.clamp(6, 32);
            }
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

fn rgb(r: u8, g: u8, b: u8) -> u32 {
    ((r as u32) << 16) | ((g as u32) << 8) | b as u32
}

fn blend(dst: u32, src: u32, alpha: f32) -> u32 {
    let alpha = alpha.clamp(0.0, 1.0);
    let inv = 1.0 - alpha;
    let dr = ((dst >> 16) & 0xff) as f32;
    let dg = ((dst >> 8) & 0xff) as f32;
    let db = (dst & 0xff) as f32;
    let sr = ((src >> 16) & 0xff) as f32;
    let sg = ((src >> 8) & 0xff) as f32;
    let sb = (src & 0xff) as f32;
    rgb(
        (dr * inv + sr * alpha).round() as u8,
        (dg * inv + sg * alpha).round() as u8,
        (db * inv + sb * alpha).round() as u8,
    )
}

fn lerp_color(a: (u8, u8, u8), b: (u8, u8, u8), t: f32) -> u32 {
    let t = t.clamp(0.0, 1.0);
    rgb(
        (a.0 as f32 + (b.0 as f32 - a.0 as f32) * t).round() as u8,
        (a.1 as f32 + (b.1 as f32 - a.1 as f32) * t).round() as u8,
        (a.2 as f32 + (b.2 as f32 - a.2 as f32) * t).round() as u8,
    )
}

fn field_color(view: TerrariumTopdownView, value: f32, daylight: f32) -> u32 {
    let v = value.clamp(0.0, 1.0).sqrt();
    let day = daylight.clamp(0.2, 1.0);
    match view {
        TerrariumTopdownView::Terrain => blend(
            lerp_color((46, 35, 28), (128, 104, 72), v),
            rgb(210, 198, 160),
            0.18 * day,
        ),
        TerrariumTopdownView::SoilMoisture => lerp_color((68, 42, 26), (55, 130, 178), v),
        TerrariumTopdownView::Canopy => lerp_color((18, 44, 24), (116, 185, 90), v),
        TerrariumTopdownView::Chemistry => lerp_color((18, 24, 32), (228, 164, 56), v),
        TerrariumTopdownView::Odor => lerp_color((10, 12, 14), (220, 82, 42), v),
        TerrariumTopdownView::GasExchange => lerp_color((8, 18, 26), (110, 214, 196), v),
    }
}

fn draw_rect(
    buffer: &mut [u32],
    buffer_w: usize,
    buffer_h: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    color: u32,
) {
    let x1 = (x + w).min(buffer_w);
    let y1 = (y + h).min(buffer_h);
    for yy in y..y1 {
        let row = yy * buffer_w;
        for xx in x..x1 {
            buffer[row + xx] = color;
        }
    }
}

fn draw_text(
    buffer: &mut [u32],
    buffer_w: usize,
    buffer_h: usize,
    x: usize,
    y: usize,
    text: &str,
    color: u32,
) {
    let mut cursor_x = x;
    for ch in text.chars() {
        if ch == '\n' {
            cursor_x = x;
            continue;
        }
        if let Some(glyph) = BASIC_FONTS.get(ch) {
            for (row, bits) in glyph.iter().enumerate() {
                let yy = y + row;
                if yy >= buffer_h {
                    continue;
                }
                let row_start = yy * buffer_w;
                for col in 0..8usize {
                    let xx = cursor_x + col;
                    if xx >= buffer_w {
                        continue;
                    }
                    if (bits >> col) & 1 == 1 {
                        buffer[row_start + xx] = color;
                    }
                }
            }
        }
        cursor_x += 8;
    }
}

fn draw_bar(
    buffer: &mut [u32],
    buffer_w: usize,
    buffer_h: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    value: f32,
    color: u32,
) {
    draw_rect(buffer, buffer_w, buffer_h, x, y, w, h, rgb(28, 32, 36));
    let fill = (w as f32 * value.clamp(0.0, 1.0)).round() as usize;
    draw_rect(buffer, buffer_w, buffer_h, x, y, fill.min(w), h, color);
}

fn normalize_field(values: &[f32]) -> Vec<f32> {
    let peak = values.iter().copied().fold(0.0f32, f32::max);
    if peak <= 1.0e-9 {
        vec![0.0; values.len()]
    } else {
        values.iter().map(|value| *value / peak).collect()
    }
}

fn draw_world(
    buffer: &mut [u32],
    world: &TerrariumWorld,
    snapshot: &TerrariumWorldSnapshot,
    view: TerrariumTopdownView,
    cell_px: usize,
) {
    let world_w = world.width();
    let world_h = world.height();
    let screen_w = world_w * cell_px + PANEL_W;
    let screen_h = world_h * cell_px;
    buffer.fill(rgb(14, 16, 18));

    let field = normalize_field(&world.topdown_field(view));
    for y in 0..world_h {
        for x in 0..world_w {
            let idx = y * world_w + x;
            let color = field_color(view, field[idx], snapshot.light);
            draw_rect(
                buffer,
                screen_w,
                screen_h,
                x * cell_px,
                y * cell_px,
                cell_px,
                cell_px,
                color,
            );
        }
    }

    for water in &world.waters {
        if water.alive {
            let x = water.x * cell_px;
            let y = water.y * cell_px;
            draw_rect(
                buffer,
                screen_w,
                screen_h,
                x + cell_px / 4,
                y + cell_px / 4,
                (cell_px / 2).max(2),
                (cell_px / 2).max(2),
                rgb(72, 168, 220),
            );
        }
    }
    for plant in &world.plants {
        let x = plant.x * cell_px;
        let y = plant.y * cell_px;
        let canopy = plant.canopy_radius_cells().min(3) * (cell_px / 3).max(1);
        let accent = blend(
            rgb(18, 64, 26),
            rgb(98, 182, 92),
            snapshot.light * 0.6 + 0.25,
        );
        draw_rect(
            buffer,
            screen_w,
            screen_h,
            x.saturating_sub(canopy / 2),
            y.saturating_sub(canopy / 2),
            (cell_px + canopy).min(screen_w.saturating_sub(x.saturating_sub(canopy / 2))),
            (cell_px + canopy).min(screen_h.saturating_sub(y.saturating_sub(canopy / 2))),
            blend(accent, rgb(14, 18, 14), 0.18),
        );
        draw_rect(
            buffer,
            screen_w,
            screen_h,
            x + cell_px / 4,
            y + cell_px / 4,
            (cell_px / 2).max(2),
            (cell_px / 2).max(2),
            accent,
        );
    }
    for fruit in &world.fruits {
        if fruit.source.alive && fruit.source.sugar_content > 0.01 {
            let x = fruit.source.x * cell_px;
            let y = fruit.source.y * cell_px;
            let color = if fruit.source.sugar_content > 0.4 {
                rgb(236, 146, 40)
            } else {
                rgb(168, 98, 36)
            };
            draw_rect(
                buffer,
                screen_w,
                screen_h,
                x + cell_px / 4,
                y + cell_px / 4,
                (cell_px / 2).max(2),
                (cell_px / 2).max(2),
                color,
            );
        }
    }
    for fly in &world.flies {
        let body = fly.body_state();
        let x = body.x.round().clamp(0.0, (world_w - 1) as f32) as usize * cell_px;
        let y = body.y.round().clamp(0.0, (world_h - 1) as f32) as usize * cell_px;
        let color = if body.is_flying {
            rgb(242, 248, 232)
        } else {
            rgb(22, 22, 24)
        };
        draw_rect(
            buffer,
            screen_w,
            screen_h,
            x + cell_px / 3,
            y + cell_px / 3,
            (cell_px / 3).max(2),
            (cell_px / 3).max(2),
            color,
        );
    }
}

fn draw_panel(
    buffer: &mut [u32],
    world: &TerrariumWorld,
    snapshot: &TerrariumWorldSnapshot,
    view: TerrariumTopdownView,
    paused: bool,
    fps: u64,
    cell_px: usize,
) {
    let world_px_w = world.width() * cell_px;
    let screen_w = world_px_w + PANEL_W;
    let screen_h = world.height() * cell_px;
    draw_rect(
        buffer,
        screen_w,
        screen_h,
        world_px_w,
        0,
        PANEL_W,
        screen_h,
        rgb(20, 22, 26),
    );
    draw_rect(
        buffer,
        screen_w,
        screen_h,
        world_px_w,
        0,
        2,
        screen_h,
        rgb(40, 46, 54),
    );

    let x = world_px_w + 14;
    let mut y = 14usize;
    draw_text(
        buffer,
        screen_w,
        screen_h,
        x,
        y,
        "NATIVE TERRARIUM",
        rgb(232, 236, 240),
    );
    y += 20;
    draw_text(
        buffer,
        screen_w,
        screen_h,
        x,
        y,
        &format!("time  {}", world.time_label()),
        rgb(190, 198, 208),
    );
    y += 12;
    draw_text(
        buffer,
        screen_w,
        screen_h,
        x,
        y,
        &format!("view  {}", view.label()),
        rgb(190, 198, 208),
    );
    y += 12;
    draw_text(
        buffer,
        screen_w,
        screen_h,
        x,
        y,
        if paused { "state paused" } else { "state live" },
        if paused {
            rgb(234, 186, 78)
        } else {
            rgb(112, 196, 122)
        },
    );
    y += 12;
    draw_text(
        buffer,
        screen_w,
        screen_h,
        x,
        y,
        &format!("fps   {}", fps),
        rgb(190, 198, 208),
    );
    y += 20;

    let lines = [
        format!("plants {}", snapshot.plants),
        format!("fruits {}", snapshot.fruits),
        format!("seeds  {}", snapshot.seeds),
        format!("flies  {}", snapshot.flies),
        format!("food   {:.2}", snapshot.food_remaining),
        format!("flyeat {:.2}", snapshot.fly_food_total),
    ];
    for line in lines {
        draw_text(buffer, screen_w, screen_h, x, y, &line, rgb(216, 220, 226));
        y += 12;
    }
    y += 8;

    draw_text(
        buffer,
        screen_w,
        screen_h,
        x,
        y,
        "LIGHT",
        rgb(210, 214, 220),
    );
    y += 10;
    draw_bar(
        buffer,
        screen_w,
        screen_h,
        x,
        y,
        PANEL_W - 28,
        8,
        snapshot.light,
        rgb(230, 200, 88),
    );
    y += 18;
    draw_text(
        buffer,
        screen_w,
        screen_h,
        x,
        y,
        "HUMIDITY",
        rgb(210, 214, 220),
    );
    y += 10;
    draw_bar(
        buffer,
        screen_w,
        screen_h,
        x,
        y,
        PANEL_W - 28,
        8,
        snapshot.humidity,
        rgb(80, 156, 228),
    );
    y += 18;
    draw_text(
        buffer,
        screen_w,
        screen_h,
        x,
        y,
        "CELL VITALITY",
        rgb(210, 214, 220),
    );
    y += 10;
    draw_bar(
        buffer,
        screen_w,
        screen_h,
        x,
        y,
        PANEL_W - 28,
        8,
        snapshot.mean_cell_vitality,
        rgb(94, 188, 108),
    );
    y += 18;
    draw_text(
        buffer,
        screen_w,
        screen_h,
        x,
        y,
        "CELL ENERGY",
        rgb(210, 214, 220),
    );
    y += 10;
    draw_bar(
        buffer,
        screen_w,
        screen_h,
        x,
        y,
        PANEL_W - 28,
        8,
        snapshot.mean_cell_energy,
        rgb(238, 168, 70),
    );
    y += 20;

    let info = [
        format!("temp {:.2} C", snapshot.temperature),
        format!("soil {:.3}", snapshot.mean_soil_moisture),
        format!("deep {:.3}", snapshot.mean_deep_moisture),
        format!("micro {:.3}", snapshot.mean_microbes),
        format!("symb {:.3}", snapshot.mean_symbionts),
        format!("cells {:.0}", snapshot.total_plant_cells),
        format!("substr {}", snapshot.substrate_backend),
        format!("steps {}", snapshot.substrate_steps),
    ];
    for line in info {
        draw_text(buffer, screen_w, screen_h, x, y, &line, rgb(184, 190, 198));
        y += 12;
    }

    y = screen_h.saturating_sub(90);
    for line in [
        "1 terrain  2 soil",
        "3 canopy   4 chemistry",
        "5 odor     6 gas",
        "space pause",
        "right step r reset",
        "up/down fps esc quit",
    ] {
        draw_text(buffer, screen_w, screen_h, x, y, line, rgb(148, 154, 162));
        y += 12;
    }
}

fn update_title(
    window: &mut Window,
    snapshot: &TerrariumWorldSnapshot,
    world: &TerrariumWorld,
    view: TerrariumTopdownView,
) {
    window.set_title(&format!(
        "oNeura Terrarium | time={} | view={} | plants={} fruits={} flies={} | food={:.2} | cells={:.0} | substrate={}",
        world.time_label(),
        view.label(),
        snapshot.plants,
        snapshot.fruits,
        snapshot.flies,
        snapshot.food_remaining,
        snapshot.total_plant_cells,
        snapshot.substrate_backend,
    ));
}

fn main() -> ExitCode {
    let cli = match parse_args() {
        Ok(cli) => cli,
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
    let mut view = TerrariumTopdownView::Terrain;
    let mut paused = false;
    let mut fps = cli.fps.max(1);

    let width = world.width() * cli.cell_px + PANEL_W;
    let height = world.height() * cli.cell_px;
    let mut window = match Window::new(
        "oNeura Terrarium",
        width,
        height,
        WindowOptions {
            resize: true,
            ..WindowOptions::default()
        },
    ) {
        Ok(window) => window,
        Err(err) => {
            eprintln!("failed to open window: {err}");
            return ExitCode::FAILURE;
        }
    };
    let mut buffer = vec![0u32; width * height];
    let mut frame_idx = 0usize;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let started = Instant::now();
        if window.is_key_pressed(Key::Space, KeyRepeat::No) {
            paused = !paused;
        }
        if window.is_key_pressed(Key::Key1, KeyRepeat::No) {
            view = TerrariumTopdownView::Terrain;
        }
        if window.is_key_pressed(Key::Key2, KeyRepeat::No) {
            view = TerrariumTopdownView::SoilMoisture;
        }
        if window.is_key_pressed(Key::Key3, KeyRepeat::No) {
            view = TerrariumTopdownView::Canopy;
        }
        if window.is_key_pressed(Key::Key4, KeyRepeat::No) {
            view = TerrariumTopdownView::Chemistry;
        }
        if window.is_key_pressed(Key::Key5, KeyRepeat::No) {
            view = TerrariumTopdownView::Odor;
        }
        if window.is_key_pressed(Key::Key6, KeyRepeat::No) {
            view = TerrariumTopdownView::GasExchange;
        }
        if window.is_key_pressed(Key::R, KeyRepeat::No) {
            match TerrariumWorld::demo(cli.seed, !cli.cpu_substrate) {
                Ok(new_world) => world = new_world,
                Err(err) => {
                    eprintln!("reset failed: {err}");
                    return ExitCode::FAILURE;
                }
            }
            frame_idx = 0;
        }
        if window.is_key_pressed(Key::Up, KeyRepeat::No) {
            fps = (fps + 5).min(120);
        }
        if window.is_key_pressed(Key::Down, KeyRepeat::No) {
            fps = fps.saturating_sub(5).max(1);
        }

        let stepped = if !paused || window.is_key_pressed(Key::Right, KeyRepeat::No) {
            if let Err(err) = world.step_frame() {
                eprintln!("terrarium step failed: {err}");
                return ExitCode::FAILURE;
            }
            frame_idx += 1;
            true
        } else {
            false
        };
        let snapshot = world.snapshot();
        let _ = stepped;
        draw_world(&mut buffer, &world, &snapshot, view, cli.cell_px);
        draw_panel(
            &mut buffer,
            &world,
            &snapshot,
            view,
            paused,
            fps,
            cli.cell_px,
        );
        update_title(&mut window, &snapshot, &world, view);

        if let Err(err) = window.update_with_buffer(&buffer, width, height) {
            eprintln!("buffer update failed: {err}");
            return ExitCode::FAILURE;
        }

        if let Some(max_frames) = cli.frames {
            if frame_idx >= max_frames {
                break;
            }
        }

        let target = Duration::from_secs_f64(1.0 / fps as f64);
        let elapsed = started.elapsed();
        if elapsed < target {
            thread::sleep(target - elapsed);
        }
    }

    ExitCode::SUCCESS
}
