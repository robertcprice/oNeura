//! Stats panel, HUD overlay, minimap, and entity detail drawing.

use font8x8::{UnicodeFonts, BASIC_FONTS};
use oneuro_metal::{TerrariumWorld, TerrariumWorldSnapshot};
use super::camera::Camera;
use super::color::{rgb, OverlayMode};
use super::mesh::EntityTag;
use super::selection::Selection;
use super::{VIEWPORT_W, PANEL_W, TOTAL_W, TOTAL_H, CELL_SIZE};

pub fn draw_rect(buffer: &mut [u32], bw: usize, bh: usize, x: usize, y: usize, w: usize, h: usize, color: u32) {
    let x1 = (x + w).min(bw);
    let y1 = (y + h).min(bh);
    for yy in y..y1 {
        let row = yy * bw;
        for xx in x..x1 { buffer[row + xx] = color; }
    }
}

pub fn draw_text(buffer: &mut [u32], bw: usize, bh: usize, x: usize, y: usize, text: &str, color: u32) {
    let mut cx = x;
    for ch in text.chars() {
        if let Some(glyph) = BASIC_FONTS.get(ch) {
            for (row, bits) in glyph.iter().enumerate() {
                let yy = y + row;
                if yy >= bh { continue; }
                let rs = yy * bw;
                for col in 0..8usize {
                    let xx = cx + col;
                    if xx >= bw { continue; }
                    if (bits >> col) & 1 == 1 { buffer[rs + xx] = color; }
                }
            }
        }
        cx += 8;
    }
}

fn draw_bar(buffer: &mut [u32], bw: usize, bh: usize, x: usize, y: usize, w: usize, h: usize, value: f32, color: u32) {
    draw_rect(buffer, bw, bh, x, y, w, h, rgb(28, 32, 36));
    let fill = (w as f32 * value.clamp(0.0, 1.0)).round() as usize;
    draw_rect(buffer, bw, bh, x, y, fill.min(w), h, color);
}

pub fn draw_panel(
    buffer: &mut [u32], world: &TerrariumWorld, snapshot: &TerrariumWorldSnapshot,
    paused: bool, realistic: bool, actual_fps: f32, cam: &Camera, selection: &Selection,
    pop_history: &std::collections::VecDeque<(usize, usize)>,
) {
    let px = VIEWPORT_W;
    draw_rect(buffer, TOTAL_W, TOTAL_H, px, 0, PANEL_W, TOTAL_H, rgb(20, 22, 26));
    draw_rect(buffer, TOTAL_W, TOTAL_H, px, 0, 2, TOTAL_H, rgb(40, 46, 54));
    let x = px + 14;
    let mut y = 14usize;
    draw_text(buffer, TOTAL_W, TOTAL_H, x, y, "oNeura TERRARIUM 3D", rgb(232, 236, 240)); y += 20;
    draw_text(buffer, TOTAL_W, TOTAL_H, x, y, &format!("time  {}", world.time_label()), rgb(190, 198, 208)); y += 12;
    draw_text(buffer, TOTAL_W, TOTAL_H, x, y, if paused { "state PAUSED" } else { "state LIVE" }, if paused { rgb(234, 186, 78) } else { rgb(112, 196, 122) }); y += 12;
    draw_text(buffer, TOTAL_W, TOTAL_H, x, y, &format!("fps   {:.1}", actual_fps), rgb(190, 198, 208)); y += 12;
    draw_text(buffer, TOTAL_W, TOTAL_H, x, y, &format!("mode  {}", if realistic { "REALISTIC" } else { "FLAT" }), if realistic { rgb(230, 200, 88) } else { rgb(160, 160, 170) }); y += 20;

    for line in [
        format!("plants {}", snapshot.plants), format!("fruits {}", snapshot.fruits),
        format!("seeds  {}", snapshot.seeds), format!("flies  {}", snapshot.flies),
        format!("food   {:.2}", snapshot.food_remaining), format!("flyeat {:.2}", snapshot.fly_food_total),
    ] {
        draw_text(buffer, TOTAL_W, TOTAL_H, x, y, &line, rgb(216, 220, 226)); y += 12;
    }
    y += 8;

    draw_text(buffer, TOTAL_W, TOTAL_H, x, y, "LIGHT", rgb(210, 214, 220)); y += 10;
    draw_bar(buffer, TOTAL_W, TOTAL_H, x, y, PANEL_W - 28, 8, snapshot.light, rgb(230, 200, 88)); y += 18;
    draw_text(buffer, TOTAL_W, TOTAL_H, x, y, "HUMIDITY", rgb(210, 214, 220)); y += 10;
    draw_bar(buffer, TOTAL_W, TOTAL_H, x, y, PANEL_W - 28, 8, snapshot.humidity, rgb(80, 156, 228)); y += 18;
    draw_text(buffer, TOTAL_W, TOTAL_H, x, y, "VITALITY", rgb(210, 214, 220)); y += 10;
    draw_bar(buffer, TOTAL_W, TOTAL_H, x, y, PANEL_W - 28, 8, snapshot.mean_cell_vitality, rgb(94, 188, 108)); y += 18;
    draw_text(buffer, TOTAL_W, TOTAL_H, x, y, "ENERGY", rgb(210, 214, 220)); y += 10;
    draw_bar(buffer, TOTAL_W, TOTAL_H, x, y, PANEL_W - 28, 8, snapshot.mean_cell_energy, rgb(238, 168, 70)); y += 20;

    for line in [
        format!("temp {:.2} C", snapshot.temperature), format!("soil {:.3}", snapshot.mean_soil_moisture),
        format!("deep {:.3}", snapshot.mean_deep_moisture), format!("micro {:.3}", snapshot.mean_microbes),
        format!("symb {:.3}", snapshot.mean_symbionts), format!("cells {:.0}", snapshot.total_plant_cells),
        format!("CO2  {:.4}", snapshot.mean_atmospheric_co2), format!("O2   {:.2}", snapshot.mean_atmospheric_o2),
    ] {
        draw_text(buffer, TOTAL_W, TOTAL_H, x, y, &line, rgb(184, 190, 198)); y += 12;
    }
    y += 6;

    // Selection detail
    if selection.is_selected() {
        draw_rect(buffer, TOTAL_W, TOTAL_H, px + 4, y, PANEL_W - 8, 1, rgb(60, 66, 74));
        y += 4;
        draw_text(buffer, TOTAL_W, TOTAL_H, x, y, &format!("SEL {}", selection.label()), rgb(255, 220, 100)); y += 12;
        y = draw_entity_detail(buffer, world, selection, x, y);
    }
    y += 4;

    // Population sparkline
    if pop_history.len() > 2 {
        draw_rect(buffer, TOTAL_W, TOTAL_H, px + 4, y, PANEL_W - 8, 1, rgb(40, 46, 54));
        y += 4;
        draw_text(buffer, TOTAL_W, TOTAL_H, x, y, "POPULATION", rgb(210, 214, 220)); y += 10;
        let graph_w = (PANEL_W - 28).min(200);
        let graph_h: usize = 32;
        draw_rect(buffer, TOTAL_W, TOTAL_H, x, y, graph_w, graph_h, rgb(16, 18, 22));
        // Draw border
        draw_rect(buffer, TOTAL_W, TOTAL_H, x, y, graph_w, 1, rgb(40, 46, 54));
        draw_rect(buffer, TOTAL_W, TOTAL_H, x, y + graph_h - 1, graph_w, 1, rgb(40, 46, 54));

        let max_pop = pop_history.iter().map(|(p, f)| (*p).max(*f)).max().unwrap_or(1).max(1);
        let n = pop_history.len();
        for (i, (plants, flies)) in pop_history.iter().enumerate() {
            let gx = x + (i * graph_w / n).min(graph_w - 1);
            // Plant bar (green, from bottom)
            let ph = (*plants * (graph_h - 2) / max_pop).min(graph_h - 2);
            for dy in 0..ph {
                let py = y + graph_h - 2 - dy;
                if gx < TOTAL_W && py < TOTAL_H { buffer[py * TOTAL_W + gx] = rgb(40, 180, 60); }
            }
            // Fly bar (yellow, drawn on top)
            let fh = (*flies * (graph_h - 2) / max_pop).min(graph_h - 2);
            for dy in 0..fh {
                let py = y + graph_h - 2 - dy;
                if gx < TOTAL_W && py < TOTAL_H { buffer[py * TOTAL_W + gx] = rgb(230, 210, 40); }
            }
        }
        y += graph_h + 2;
        draw_text(buffer, TOTAL_W, TOTAL_H, x, y, "plants", rgb(40, 180, 60));
        draw_text(buffer, TOTAL_W, TOTAL_H, x + 56, y, "flies", rgb(230, 210, 40));
        y += 12;
    }

    // Minimap
    draw_minimap(buffer, world, x, y);
    y += 82;

    // Camera info
    draw_text(buffer, TOTAL_W, TOTAL_H, x, y, "CAMERA", rgb(210, 214, 220)); y += 10;
    draw_text(buffer, TOTAL_W, TOTAL_H, x, y, &format!("yaw {:.1} pitch {:.1}", cam.yaw.to_degrees(), cam.pitch.to_degrees()), rgb(160, 166, 174)); y += 12;
    draw_text(buffer, TOTAL_W, TOTAL_H, x, y, &format!("dist {:.1}", cam.distance), rgb(160, 166, 174));

    // Controls legend at bottom
    let cy = TOTAL_H.saturating_sub(110);
    for (i, line) in [
        "L-drag  rotate", "R-drag  pan", "scroll  zoom", "click   select",
        "Tab     cycle", "WASD    pan", "L       lighting", "F       follow",
        "T       orbit", "[/]     speed", "1-6     overlay", "E       export CSV",
        "V       record", "space   pause", "F12     screenshot",
        "R       reset cam", "esc     quit",
    ].iter().enumerate() {
        draw_text(buffer, TOTAL_W, TOTAL_H, x, cy + i * 10, line, rgb(128, 134, 142));
    }
}

fn draw_entity_detail(buffer: &mut [u32], world: &TerrariumWorld, sel: &Selection, x: usize, mut y: usize) -> usize {
    match sel.tag {
        EntityTag::Plant(i) => {
            if let Some(plant) = world.plants.get(i) {
                let vit = plant.cellular.vitality();
                let cells = plant.cellular.total_cells();
                let h = plant.physiology.height_mm();
                draw_text(buffer, TOTAL_W, TOTAL_H, x, y, &format!("ht {:.1}mm  cells {:.0}", h, cells), rgb(200, 210, 200)); y += 11;
                draw_text(buffer, TOTAL_W, TOTAL_H, x, y, "VITALITY", rgb(180, 190, 180)); y += 9;
                draw_bar(buffer, TOTAL_W, TOTAL_H, x, y, PANEL_W - 28, 6, vit, rgb(80, 180, 90)); y += 12;
                let ec = plant.cellular.energy_charge();
                draw_text(buffer, TOTAL_W, TOTAL_H, x, y, "ENERGY", rgb(180, 190, 180)); y += 9;
                draw_bar(buffer, TOTAL_W, TOTAL_H, x, y, PANEL_W - 28, 6, ec, rgb(220, 180, 60)); y += 12;
            }
        }
        EntityTag::Fly(i) => {
            if let Some(fly) = world.flies.get(i) {
                let b = fly.body_state();
                draw_text(buffer, TOTAL_W, TOTAL_H, x, y, &format!("E {:.2} spd {:.2}", b.energy, b.speed), rgb(220, 218, 180)); y += 11;
                draw_text(buffer, TOTAL_W, TOTAL_H, x, y, &format!("alt {:.2} {}", b.z, if b.is_flying { "FLYING" } else { "LANDED" }), rgb(200, 200, 180)); y += 11;
                // Fly metabolism details from snapshot
                draw_text(buffer, TOTAL_W, TOTAL_H, x, y, &format!("hdg {:.1}", b.heading), rgb(190, 190, 170)); y += 11;
            }
        }
        EntityTag::Water(i) => {
            if let Some(water) = world.waters.get(i) {
                draw_text(buffer, TOTAL_W, TOTAL_H, x, y, &format!("vol {:.3} {}", water.volume, if water.alive { "ALIVE" } else { "DRY" }), rgb(160, 200, 230)); y += 11;
                draw_text(buffer, TOTAL_W, TOTAL_H, x, y, &format!("pos ({}, {})", water.x, water.y), rgb(150, 190, 220)); y += 11;
            }
        }
        EntityTag::Fruit(i) => {
            if let Some(fruit) = world.fruits.get(i) {
                draw_text(buffer, TOTAL_W, TOTAL_H, x, y, &format!("sugar {:.3} ripe {:.2}", fruit.source.sugar_content, fruit.source.ripeness), rgb(220, 180, 150)); y += 11;
                draw_text(buffer, TOTAL_W, TOTAL_H, x, y, &format!("{}", if fruit.source.alive { "ALIVE" } else { "CONSUMED" }), rgb(200, 160, 130)); y += 11;
            }
        }
        _ => {}
    }
    y
}

/// Draw an 100x72 minimap in the panel showing terrain and entity positions.
fn draw_minimap(buffer: &mut [u32], world: &TerrariumWorld, x: usize, y: usize) {
    let map_w: usize = 100;
    let map_h: usize = 72;
    let gw = world.config.width;
    let gh = world.config.height;
    let moisture = world.moisture_field();

    // Background
    draw_rect(buffer, TOTAL_W, TOTAL_H, x, y, map_w, map_h, rgb(16, 18, 22));
    // Border
    draw_rect(buffer, TOTAL_W, TOTAL_H, x, y, map_w, 1, rgb(50, 56, 64));
    draw_rect(buffer, TOTAL_W, TOTAL_H, x, y + map_h - 1, map_w, 1, rgb(50, 56, 64));
    draw_rect(buffer, TOTAL_W, TOTAL_H, x, y, 1, map_h, rgb(50, 56, 64));
    draw_rect(buffer, TOTAL_W, TOTAL_H, x + map_w - 1, y, 1, map_h, rgb(50, 56, 64));

    // Terrain color
    for my in 1..(map_h - 1) {
        for mx in 1..(map_w - 1) {
            let gx = ((mx - 1) as f32 / (map_w - 2) as f32 * (gw - 1) as f32) as usize;
            let gy = ((my - 1) as f32 / (map_h - 2) as f32 * (gh - 1) as f32) as usize;
            let mi = gy.min(gh - 1) * gw + gx.min(gw - 1);
            let m = if mi < moisture.len() { moisture[mi] } else { 0.3 };
            let g = (m * 120.0 + 30.0).min(255.0) as u8;
            let bx = x + mx;
            let by = y + my;
            if bx < TOTAL_W && by < TOTAL_H {
                buffer[by * TOTAL_W + bx] = rgb(g / 3, g, g / 4);
            }
        }
    }

    let to_map = |ex: f32, ey: f32| -> (usize, usize) {
        let mx = (ex / (gw as f32) * (map_w - 2) as f32) as usize + 1;
        let my = (ey / (gh as f32) * (map_h - 2) as f32) as usize + 1;
        (x + mx.min(map_w - 2), y + my.min(map_h - 2))
    };

    // Water dots (blue)
    for water in &world.waters {
        if water.alive {
            let (px, py) = to_map(water.x as f32, water.y as f32);
            if px < TOTAL_W && py < TOTAL_H { buffer[py * TOTAL_W + px] = rgb(60, 120, 220); }
        }
    }
    // Plant dots (green)
    for plant in &world.plants {
        let (px, py) = to_map(plant.x as f32, plant.y as f32);
        if px < TOTAL_W && py < TOTAL_H { buffer[py * TOTAL_W + px] = rgb(40, 200, 60); }
    }
    // Fly dots (yellow)
    for fly in &world.flies {
        let b = fly.body_state();
        let (px, py) = to_map(b.x, b.y);
        if px < TOTAL_W && py < TOTAL_H { buffer[py * TOTAL_W + px] = rgb(240, 230, 40); }
    }
    // Fruit dots (orange)
    for fruit in &world.fruits {
        if fruit.source.alive && fruit.source.sugar_content > 0.01 {
            let (px, py) = to_map(fruit.source.x as f32, fruit.source.y as f32);
            if px < TOTAL_W && py < TOTAL_H { buffer[py * TOTAL_W + px] = rgb(240, 140, 40); }
        }
    }

    // Label
    draw_text(buffer, TOTAL_W, TOTAL_H, x, y + map_h + 2, "MINIMAP", rgb(130, 136, 144));
}

pub fn draw_hud(buffer: &mut [u32], paused: bool, realistic: bool, screenshot_msg: &str, zoom: &super::camera::ZoomLevel, following: bool, sim_speed: u32, auto_orbit: bool, overlay: &OverlayMode, recording: bool) {
    let label = if realistic { "3D REALISTIC" } else { "3D FLAT" };
    draw_rect(buffer, TOTAL_W, TOTAL_H, 4, 4, label.len() * 8 + 8, 14, rgb(10, 12, 16));
    draw_text(buffer, TOTAL_W, TOTAL_H, 8, 7, label, if realistic { rgb(230, 200, 88) } else { rgb(160, 160, 170) });
    // Zoom level indicator
    let zoom_label = zoom.label();
    let zoom_color = match zoom {
        super::camera::ZoomLevel::Ecosystem => rgb(100, 200, 100),
        super::camera::ZoomLevel::Organism => rgb(100, 180, 230),
        super::camera::ZoomLevel::Cellular => rgb(200, 160, 230),
        super::camera::ZoomLevel::Molecular => rgb(230, 120, 230),
    };
    let zw = zoom_label.len() * 8 + 8;
    draw_rect(buffer, TOTAL_W, TOTAL_H, 4, 22, zw, 14, rgb(10, 12, 16));
    draw_text(buffer, TOTAL_W, TOTAL_H, 8, 25, zoom_label, zoom_color);
    // Status indicators row
    let mut ix = 4usize;
    if following {
        let fmsg = "FOLLOW";
        let fw = fmsg.len() * 8 + 8;
        draw_rect(buffer, TOTAL_W, TOTAL_H, ix, 40, fw, 14, rgb(10, 40, 60));
        draw_text(buffer, TOTAL_W, TOTAL_H, ix + 4, 43, fmsg, rgb(80, 200, 255));
        ix += fw + 4;
    }
    if sim_speed > 1 {
        let smsg = format!("{}x", sim_speed);
        let sw = smsg.len() * 8 + 8;
        draw_rect(buffer, TOTAL_W, TOTAL_H, ix, 40, sw, 14, rgb(50, 30, 10));
        draw_text(buffer, TOTAL_W, TOTAL_H, ix + 4, 43, &smsg, rgb(255, 180, 60));
        ix += sw + 4;
    }
    if auto_orbit {
        let omsg = "ORBIT";
        let ow = omsg.len() * 8 + 8;
        draw_rect(buffer, TOTAL_W, TOTAL_H, ix, 40, ow, 14, rgb(30, 10, 50));
        draw_text(buffer, TOTAL_W, TOTAL_H, ix + 4, 43, omsg, rgb(180, 120, 255));
        ix += ow + 4;
    }
    if *overlay != OverlayMode::Default {
        let olabel = overlay.label();
        let olen = olabel.len() * 8 + 8;
        draw_rect(buffer, TOTAL_W, TOTAL_H, ix, 40, olen, 14, rgb(10, 40, 10));
        draw_text(buffer, TOTAL_W, TOTAL_H, ix + 4, 43, olabel, rgb(100, 230, 140));
        ix += olen + 4;
    }
    if recording {
        let rmsg = "REC";
        let rw = rmsg.len() * 8 + 8;
        draw_rect(buffer, TOTAL_W, TOTAL_H, ix, 40, rw, 14, rgb(80, 10, 10));
        draw_text(buffer, TOTAL_W, TOTAL_H, ix + 4, 43, rmsg, rgb(255, 60, 60));
    }
    if paused {
        let msg = "PAUSED";
        let mx = (VIEWPORT_W - msg.len() * 8) / 2;
        draw_rect(buffer, TOTAL_W, TOTAL_H, mx - 4, 4, msg.len() * 8 + 8, 14, rgb(80, 20, 20));
        draw_text(buffer, TOTAL_W, TOTAL_H, mx, 7, msg, rgb(255, 80, 80));
    }
    // Screenshot notification
    if !screenshot_msg.is_empty() {
        let sw = screenshot_msg.len() * 8 + 8;
        let sx = (VIEWPORT_W - sw) / 2;
        draw_rect(buffer, TOTAL_W, TOTAL_H, sx, TOTAL_H - 24, sw, 16, rgb(20, 60, 20));
        draw_text(buffer, TOTAL_W, TOTAL_H, sx + 4, TOTAL_H - 21, screenshot_msg, rgb(120, 240, 120));
    }
}
