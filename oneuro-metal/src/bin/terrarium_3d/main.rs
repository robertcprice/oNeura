//! 3D Terrarium Viewer — Software Rasterizer with Orbit Camera, Sunlight Shadows,
//! SSAO, Entity Selection, Particles, Minimap, Screenshot Export, and Infinite Zoom.
//!
//! Zoom levels: Ecosystem → Organism → Cellular → Molecular
//! Scroll wheel zooms continuously. At molecular level, renders individual atoms.
//!
//! Usage: cargo build --profile fast --no-default-features --bin terrarium_3d
//!        ./target/fast/terrarium_3d --seed 7 --fps 30

mod math;
mod color;
mod camera;
mod mesh;
mod lighting;
mod terrain;
mod plants;
mod flies;
mod water;
mod fruits;
mod rasterizer;
mod hud;
mod input;
mod selection;
mod particles;
mod screenshot;
mod zoom;

use minifb::{Key, KeyRepeat, Window, WindowOptions};
use oneuro_metal::TerrariumWorld;
use std::env;
use std::process::ExitCode;
use std::thread;
use std::time::{Duration, Instant};

use camera::{Camera, ZoomLevel};
use color::rgb;
use hud::{draw_panel, draw_hud};
use input::InputState;
use lighting::{sun_direction, sun_color};
use particles::ParticleSystem;
use rasterizer::Rasterizer;
use selection::Selection;
use terrain::build_terrain_mesh;
use plants::build_plant_meshes;
use flies::build_fly_meshes;
use water::build_water_meshes;
use fruits::build_fruit_meshes;
use zoom::{build_organism_detail, build_cellular_detail, build_molecular_detail};

const VIEWPORT_W: usize = 960;
const VIEWPORT_H: usize = 640;
const PANEL_W: usize = 280;
const TOTAL_W: usize = VIEWPORT_W + PANEL_W;
const TOTAL_H: usize = VIEWPORT_H;
const CELL_SIZE: f32 = 0.5;
const HEIGHT_SCALE: f32 = 3.0;
const FOV_Y: f32 = std::f32::consts::PI / 3.0;
const NEAR: f32 = 0.1;
const FAR: f32 = 100.0;
const FOG_NEAR: f32 = 15.0;
const FOG_FAR: f32 = 60.0;
const SHADOW_STEPS: usize = 30;
const SHADOW_STEP_SIZE: f32 = 0.5;
const SHADOW_DARKEN: f32 = 0.4;

fn print_usage() {
    eprintln!("oNeura Terrarium 3D Viewer (Software Rasterizer)
");
    eprintln!("Usage: terrarium_3d [OPTIONS]
");
    eprintln!("Options:");
    eprintln!("  --seed <N>        World seed (default: 7)");
    eprintln!("  --fps <N>         Target framerate (default: 30)");
    eprintln!("  --frames <N>      Quit after N frames");
    eprintln!("  --cpu-substrate   Use CPU substrate backend");
}

fn main() -> ExitCode {
    let (seed, fps, frames, cpu_sub) = {
        let mut seed = 7u64; let mut fps = 30u64; let mut frames: Option<usize> = None; let mut cpu = false;
        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--seed" => seed = args.next().unwrap_or_default().parse().unwrap_or(7),
                "--fps" => fps = args.next().unwrap_or_default().parse().unwrap_or(30),
                "--frames" => frames = args.next().and_then(|s| s.parse().ok()),
                "--cpu-substrate" => cpu = true,
                "--help" | "-h" => { print_usage(); std::process::exit(0); }
                o => { eprintln!("unknown arg: {o}"); print_usage(); return ExitCode::FAILURE; }
            }
        }
        (seed, fps, frames, cpu)
    };

    let mut world = match TerrariumWorld::demo(seed, !cpu_sub) {
        Ok(w) => w,
        Err(e) => { eprintln!("failed to build terrarium: {e}"); return ExitCode::FAILURE; }
    };

    let terrain_seed = world.config.width as u64 * 31 + world.config.height as u64;

    let mut window = match Window::new("oNeura Terrarium 3D", TOTAL_W, TOTAL_H, WindowOptions { resize: false, ..WindowOptions::default() }) {
        Ok(w) => w,
        Err(e) => { eprintln!("failed to open window: {e}"); return ExitCode::FAILURE; }
    };

    let mut buffer = vec![0u32; TOTAL_W * TOTAL_H];
    let mut raster = Rasterizer::new(VIEWPORT_W, VIEWPORT_H);
    let mut cam = Camera::new();
    let mut input_state = InputState::new();
    let mut sel = Selection::new();
    let mut particle_sys = ParticleSystem::new();
    let mut paused = false;
    let mut realistic = true;
    let mut frame_idx = 0usize;
    let mut actual_fps = 0.0f32;
    let mut fps_timer = Instant::now();
    let mut fps_frames = 0usize;
    let mut screenshot_msg = String::new();
    let mut screenshot_timer = 0u32;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let started = Instant::now();

        // Toggle keys
        if window.is_key_pressed(Key::Space, KeyRepeat::No) { paused = !paused; }
        if window.is_key_pressed(Key::L, KeyRepeat::No) { realistic = !realistic; }
        if window.is_key_pressed(Key::R, KeyRepeat::No) { cam.reset(); }

        // Tab: cycle entity selection
        if window.is_key_pressed(Key::Tab, KeyRepeat::No) {
            sel.update_cycle_list(world.plants.len(), world.flies.len(), world.waters.len(), world.fruits.len());
            sel.cycle_next();
        }

        // F12: screenshot
        if window.is_key_pressed(Key::F12, KeyRepeat::No) {
            let mut vp_buf = vec![0u32; VIEWPORT_W * VIEWPORT_H];
            for y in 0..VIEWPORT_H {
                vp_buf[y * VIEWPORT_W..(y + 1) * VIEWPORT_W]
                    .copy_from_slice(&buffer[y * TOTAL_W..y * TOTAL_W + VIEWPORT_W]);
            }
            match screenshot::save_screenshot(&vp_buf, VIEWPORT_W, VIEWPORT_H) {
                Ok(name) => { screenshot_msg = format!("Saved: {}", name); screenshot_timer = 90; }
                Err(e) => { screenshot_msg = format!("Error: {}", e); screenshot_timer = 90; }
            }
        }

        // Camera keyboard controls — speed scales with zoom level
        let zoom = cam.zoom_level();
        let zoom_scale = match zoom {
            ZoomLevel::Ecosystem => 1.0,
            ZoomLevel::Organism => 0.3,
            ZoomLevel::Cellular => 0.08,
            ZoomLevel::Molecular => 0.01,
        };
        let rot_speed = 0.03;
        if window.is_key_down(Key::Left)  { cam.yaw += rot_speed; }
        if window.is_key_down(Key::Right) { cam.yaw -= rot_speed; }
        if window.is_key_down(Key::Up)    { cam.pitch = (cam.pitch + rot_speed).min(85.0_f32.to_radians()); }
        if window.is_key_down(Key::Down)  { cam.pitch = (cam.pitch - rot_speed).max(-85.0_f32.to_radians()); }
        let pan_speed = 0.15 * zoom_scale;
        let r = cam.right();
        let f = cam.forward_xz();
        if window.is_key_down(Key::W) { cam.target = math::add3(cam.target, math::scale3(f, pan_speed)); }
        if window.is_key_down(Key::S) { cam.target = math::add3(cam.target, math::scale3(f, -pan_speed)); }
        if window.is_key_down(Key::A) { cam.target = math::add3(cam.target, math::scale3(r, -pan_speed)); }
        if window.is_key_down(Key::D) { cam.target = math::add3(cam.target, math::scale3(r, pan_speed)); }

        // Zoom: +/- keys with proportional speed, NO LIMIT — infinite zoom
        let zoom_step = cam.distance * 0.08;  // proportional to distance
        if window.is_key_pressed(Key::Equal, KeyRepeat::Yes) { cam.distance = (cam.distance - zoom_step).max(0.005); }
        if window.is_key_pressed(Key::Minus, KeyRepeat::Yes) { cam.distance = cam.distance + zoom_step; }

        // Mouse input — returns click event if user clicked without dragging
        if let Some(click) = input_state.handle(&window, &mut cam) {
            let tag = raster.tag_at(click.x, click.y);
            sel.select(tag);
        }

        // Simulation step
        if !paused {
            if let Err(e) = world.step_frame() { eprintln!("step failed: {e}"); return ExitCode::FAILURE; }
            frame_idx += 1;

            // Spawn particles from entities
            let gw = world.config.width;
            let gh = world.config.height;
            let moisture = world.moisture_field();
            for (i, water_ent) in world.waters.iter().enumerate() {
                if water_ent.alive {
                    let gx = water_ent.x.min(gw - 1);
                    let gy = water_ent.y.min(gh - 1);
                    let base_y = terrain::terrain_height(gx, gy, gw, gh, &moisture, terrain_seed);
                    particle_sys.spawn_water_evap(gx as f32 * CELL_SIZE, base_y + 0.03, gy as f32 * CELL_SIZE, frame_idx, i);
                }
            }
            for (i, plant) in world.plants.iter().enumerate() {
                let gx = plant.x.min(gw - 1);
                let gy = plant.y.min(gh - 1);
                let base_y = terrain::terrain_height(gx, gy, gw, gh, &moisture, terrain_seed);
                let h = (plant.physiology.height_mm() * 0.15).clamp(0.3, 2.5);
                particle_sys.spawn_pollen(gx as f32 * CELL_SIZE, base_y + h, gy as f32 * CELL_SIZE, frame_idx, i);
            }
            for (i, fruit) in world.fruits.iter().enumerate() {
                if fruit.source.alive && fruit.source.ripeness > 0.7 {
                    let gx = fruit.source.x.min(gw - 1);
                    let gy = fruit.source.y.min(gh - 1);
                    let base_y = terrain::terrain_height(gx, gy, gw, gh, &moisture, terrain_seed);
                    particle_sys.spawn_fruit_sparkle(gx as f32 * CELL_SIZE, base_y + 0.1, gy as f32 * CELL_SIZE, frame_idx, i);
                }
            }
            particle_sys.update(1.0 / fps.max(1) as f32);
        }

        // Build scene geometry (varies by zoom level)
        let snapshot = world.snapshot();
        let gw = world.config.width;
        let gh = world.config.height;
        let moisture = world.moisture_field();

        let mut terrain_tris = Vec::new();
        let mut entity_tris = Vec::new();

        match zoom {
            ZoomLevel::Ecosystem | ZoomLevel::Organism => {
                terrain_tris = build_terrain_mesh(&world);
                entity_tris = build_plant_meshes(&world.plants, gw, gh, &moisture, terrain_seed);
                entity_tris.extend(build_fly_meshes(&world.flies, gw, gh, &moisture, terrain_seed, frame_idx));
                entity_tris.extend(build_water_meshes(&world, terrain_seed, frame_idx));
                entity_tris.extend(build_fruit_meshes(&world.fruits, gw, gh, &moisture, terrain_seed));

                // At organism level, add detail meshes for selected entity
                if zoom == ZoomLevel::Organism && sel.is_selected() {
                    entity_tris.extend(build_organism_detail(&world, &sel, cam.target));
                }
            }
            ZoomLevel::Cellular => {
                // Show terrain as context + cellular detail for selected entity
                terrain_tris = build_terrain_mesh(&world);
                entity_tris = build_plant_meshes(&world.plants, gw, gh, &moisture, terrain_seed);
                entity_tris.extend(build_fly_meshes(&world.flies, gw, gh, &moisture, terrain_seed, frame_idx));
                if sel.is_selected() {
                    entity_tris.extend(build_cellular_detail(&world, &sel, cam.target));
                }
            }
            ZoomLevel::Molecular => {
                // Only detail meshes at molecular scale — no terrain
                if sel.is_selected() {
                    entity_tris = build_molecular_detail(&world, &sel, cam.target);
                } else {
                    // If nothing selected, show substrate species cloud at camera target
                    entity_tris = build_molecular_detail(&world, &sel, cam.target);
                }
            }
        }

        // Render
        let sun_d = sun_direction(snapshot.light);
        let sun_c = sun_color(snapshot.light);
        let cam_eye = cam.eye();
        let mvp = cam.mvp();
        raster.clear(snapshot.light);
        raster.rasterize(&terrain_tris, &mvp, cam_eye, sun_d, sun_c, realistic);
        raster.rasterize(&entity_tris, &mvp, cam_eye, sun_d, sun_c, realistic);
        if realistic && zoom != ZoomLevel::Molecular {
            raster.shadow_pass(&world, sun_d);
            raster.ssao_pass();
        }
        if zoom != ZoomLevel::Molecular {
            raster.fog_pass(cam_eye, snapshot.light);
        }

        // Selection outline
        if sel.is_selected() {
            raster.draw_selection_outline(sel.tag);
        }

        // Composite viewport into display buffer
        buffer.fill(rgb(14, 16, 18));
        for y in 0..VIEWPORT_H {
            buffer[y * TOTAL_W..y * TOTAL_W + VIEWPORT_W]
                .copy_from_slice(&raster.color_buf[y * VIEWPORT_W..y * VIEWPORT_W + VIEWPORT_W]);
        }

        // Render particles on top of viewport (skip at molecular level)
        if zoom != ZoomLevel::Molecular {
            particle_sys.render(&mut buffer, VIEWPORT_W, VIEWPORT_H, &mvp);
        }

        // Draw panel and HUD
        draw_panel(&mut buffer, &world, &snapshot, paused, realistic, actual_fps, &cam, &sel);
        let msg = if screenshot_timer > 0 { &screenshot_msg } else { "" };
        draw_hud(&mut buffer, paused, realistic, msg, &zoom);
        if screenshot_timer > 0 { screenshot_timer -= 1; }

        // FPS counter
        fps_frames += 1;
        let fps_elapsed = fps_timer.elapsed().as_secs_f32();
        if fps_elapsed >= 1.0 { actual_fps = fps_frames as f32 / fps_elapsed; fps_frames = 0; fps_timer = Instant::now(); }

        // Window title
        window.set_title(&format!(
            "oNeura Terrarium 3D | {} | {} | P:{} Fl:{} | {:.1} FPS | {}{}",
            world.time_label(), zoom.label(), snapshot.plants, snapshot.flies, actual_fps,
            if realistic { "Realistic" } else { "Flat" },
            if sel.is_selected() { format!(" | {}", sel.label()) } else { String::new() },
        ));

        if let Err(e) = window.update_with_buffer(&buffer, TOTAL_W, TOTAL_H) {
            eprintln!("buffer update failed: {e}"); return ExitCode::FAILURE;
        }
        if let Some(max) = frames { if frame_idx >= max { break; } }
        let target = Duration::from_secs_f64(1.0 / fps.max(1) as f64);
        let elapsed = started.elapsed();
        if elapsed < target { thread::sleep(target - elapsed); }
    }
    ExitCode::SUCCESS
}
