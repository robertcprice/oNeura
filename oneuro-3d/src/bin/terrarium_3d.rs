//! Terrarium 3D Viewer — Bevy 0.15 real-time visualization of the oNeura terrarium.
//!
//! Architecture:
//! - `TerrariumWorld` runs on a background thread, produces snapshots via channel
//! - Bevy consumes snapshots and spawns/updates/despawns entities
//! - Orbital camera with WASD pan, scroll zoom, auto-follow
//!
//! Usage:
//!   cargo run -p oneuro-3d --profile fast --bin terrarium_3d -- --seed 42 --fps 30

use bevy::prelude::*;
use bevy::render::mesh::Indices;
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::render_resource::PrimitiveTopology;
use bevy::window::PresentMode;
use oneuro_metal::{TerrariumTopdownView, TerrariumWorld, TerrariumWorldSnapshot};
use std::collections::HashMap;
use std::sync::{mpsc, Mutex};

// ============================================================================
// Constants
// ============================================================================

const VOXEL_SIZE: f32 = 0.5;
const FLY_BODY_SCALE: f32 = 0.15;
const PLANT_SCALE: f32 = 0.3;
const FRUIT_SCALE: f32 = 0.08;
const DEFAULT_SEED: u64 = 7;
const DEFAULT_FPS: f32 = 30.0;

// ============================================================================
// CLI Args
// ============================================================================

#[derive(Debug, Clone, Resource)]
struct ViewerConfig {
    seed: u64,
    fps: f32,
    coupling: bool,
}

impl Default for ViewerConfig {
    fn default() -> Self {
        Self {
            seed: DEFAULT_SEED,
            fps: DEFAULT_FPS,
            coupling: true,
        }
    }
}

fn parse_cli() -> ViewerConfig {
    let mut config = ViewerConfig::default();
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" if i + 1 < args.len() => {
                config.seed = args[i + 1].parse().unwrap_or(DEFAULT_SEED);
                i += 2;
            }
            "--fps" if i + 1 < args.len() => {
                config.fps = args[i + 1].parse().unwrap_or(DEFAULT_FPS);
                i += 2;
            }
            "--no-coupling" => {
                config.coupling = false;
                i += 1;
            }
            _ => { i += 1; }
        }
    }
    config
}

// ============================================================================
// Simulation Resources
// ============================================================================

#[derive(Resource)]
struct SimChannel {
    rx: Mutex<mpsc::Receiver<SimFrame>>,
}

struct SimFrame {
    snapshot: TerrariumWorldSnapshot,
    terrain_field: Vec<f32>,
    width: usize,
    height: usize,
    fly_positions: Vec<(f32, f32, f32, f32)>, // x, y, z, energy
    plant_positions: Vec<(f32, f32, f32)>,     // x, y, height_mm
    fruit_positions: Vec<(f32, f32)>,          // x, y
    water_cells: Vec<(f32, f32, f32)>,         // x, y, volume
}

#[derive(Resource, Default)]
struct EntityMap {
    flies: HashMap<usize, Entity>,
    plants: HashMap<usize, Entity>,
    fruits: HashMap<usize, Entity>,
}

// ============================================================================
// Components
// ============================================================================

#[derive(Component)]
struct TerrainMesh;

#[derive(Component)]
struct FlyEntity(usize);

#[derive(Component)]
struct PlantEntity(usize);

#[derive(Component)]
struct FruitEntity(usize);

#[derive(Component)]
struct OrbitCamera {
    focus: Vec3,
    distance: f32,
    yaw: f32,
    pitch: f32,
    auto_follow: Option<usize>,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            focus: Vec3::ZERO,
            distance: 15.0,
            yaw: -0.4,
            pitch: 0.8,
            auto_follow: None,
        }
    }
}

#[derive(Component)]
struct HudText;

// ============================================================================
// Mesh Helpers
// ============================================================================

fn create_terrain_mesh(field: &[f32], width: usize, height: usize) -> Mesh {
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(width * height);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(width * height);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(width * height);
    let mut indices: Vec<u32> = Vec::new();

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let h = field.get(idx).copied().unwrap_or(0.0) * 2.0;
            positions.push([x as f32 * VOXEL_SIZE, h, y as f32 * VOXEL_SIZE]);
            normals.push([0.0, 1.0, 0.0]);
            uvs.push([x as f32 / width as f32, y as f32 / height as f32]);
        }
    }

    for y in 0..(height.saturating_sub(1)) {
        for x in 0..(width.saturating_sub(1)) {
            let tl = (y * width + x) as u32;
            let tr = tl + 1;
            let bl = tl + width as u32;
            let br = bl + 1;
            indices.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
        }
    }

    // Recompute normals from triangles
    let mut normal_accum = vec![[0.0f32; 3]; positions.len()];
    for tri in indices.chunks(3) {
        if tri.len() < 3 { continue; }
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let v0 = Vec3::from(positions[i0]);
        let v1 = Vec3::from(positions[i1]);
        let v2 = Vec3::from(positions[i2]);
        let n = (v1 - v0).cross(v2 - v0);
        for idx in [i0, i1, i2] {
            normal_accum[idx][0] += n.x;
            normal_accum[idx][1] += n.y;
            normal_accum[idx][2] += n.z;
        }
    }
    for (i, n) in normal_accum.iter().enumerate() {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-8);
        normals[i] = [n[0] / len, n[1] / len, n[2] / len];
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

fn terrain_color(height: f32) -> Color {
    let t = height.clamp(0.0, 1.0);
    Color::srgb(
        0.35 + 0.15 * (1.0 - t),
        0.25 + 0.45 * t,
        0.10 + 0.10 * t,
    )
}

// ============================================================================
// Setup Systems
// ============================================================================

fn setup_scene(
    mut commands: Commands,
    config: Res<ViewerConfig>,
) {
    // Spawn orbit camera
    let cam = OrbitCamera::default();
    let cam_pos = orbit_position(&cam);
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(cam_pos).looking_at(cam.focus, Vec3::Y),
        cam,
    ));

    // Directional light (sun)
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(1.0, 0.95, 0.85),
            illuminance: 10_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.8, 0.3, 0.0)),
    ));

    // Ambient light
    commands.insert_resource(AmbientLight {
        color: Color::srgb(0.6, 0.7, 0.9),
        brightness: 200.0,
    });

    // Start simulation on background thread
    let seed = config.seed;
    let fps = config.fps;
    let (tx, rx) = mpsc::channel();

    std::thread::spawn(move || {
        let mut world = match TerrariumWorld::demo(seed, false) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("Failed to create terrarium world: {e}");
                return;
            }
        };
        loop {
            if world.step_frame().is_err() {
                break;
            }

            let snapshot = world.snapshot();
            let terrain = world.topdown_field(TerrariumTopdownView::Terrain);
            let w = world.width();
            let h = world.height();

            let fly_positions: Vec<(f32, f32, f32, f32)> = world
                .flies
                .iter()
                .map(|f| {
                    let bs = f.body_state();
                    (bs.x, bs.y, bs.z, bs.energy)
                })
                .collect();

            let plant_positions: Vec<(f32, f32, f32)> = world
                .plants
                .iter()
                .map(|p| (p.x as f32, p.y as f32, p.physiology.height_mm()))
                .collect();

            let fruit_positions: Vec<(f32, f32)> = world
                .fruits
                .iter()
                .filter(|f| f.source.alive)
                .map(|f| (f.source.x as f32, f.source.y as f32))
                .collect();

            let water_cells: Vec<(f32, f32, f32)> = world
                .waters
                .iter()
                .filter(|w| w.alive)
                .map(|w| (w.x as f32, w.y as f32, w.volume))
                .collect();

            let frame = SimFrame {
                snapshot,
                terrain_field: terrain,
                width: w,
                height: h,
                fly_positions,
                plant_positions,
                fruit_positions,
                water_cells,
            };

            if tx.send(frame).is_err() {
                break;
            }

            std::thread::sleep(std::time::Duration::from_secs_f32(1.0 / fps));
        }
    });

    commands.insert_resource(SimChannel { rx: Mutex::new(rx) });
    commands.insert_resource(EntityMap::default());
}

fn setup_hud(mut commands: Commands) {
    commands.spawn((
        Text::new("oNeura Terrarium 3D"),
        TextFont {
            font_size: 16.0,
            ..default()
        },
        TextColor(Color::WHITE),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        },
        HudText,
    ));
}

fn orbit_position(cam: &OrbitCamera) -> Vec3 {
    let x = cam.distance * cam.pitch.cos() * cam.yaw.sin();
    let y = cam.distance * cam.pitch.sin();
    let z = cam.distance * cam.pitch.cos() * cam.yaw.cos();
    cam.focus + Vec3::new(x, y, z)
}

// ============================================================================
// Update Systems
// ============================================================================

fn consume_sim_frames(
    mut commands: Commands,
    channel: Res<SimChannel>,
    mut entity_map: ResMut<EntityMap>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    terrain_query: Query<Entity, With<TerrainMesh>>,
    mut hud_query: Query<&mut Text, With<HudText>>,
) {
    // Drain to latest frame
    let mut latest_frame = None;
    let rx = channel.rx.lock().unwrap();
    while let Ok(frame) = rx.try_recv() {
        latest_frame = Some(frame);
    }
    drop(rx);

    let Some(frame) = latest_frame else { return };

    // Update terrain mesh
    if frame.width > 1 && frame.height > 1 {
        let terrain_mesh = create_terrain_mesh(&frame.terrain_field, frame.width, frame.height);
        let terrain_handle = meshes.add(terrain_mesh);
        let terrain_mat = materials.add(StandardMaterial {
            base_color: terrain_color(0.5),
            perceptual_roughness: 0.9,
            ..default()
        });

        for entity in terrain_query.iter() {
            commands.entity(entity).despawn();
        }
        commands.spawn((
            Mesh3d(terrain_handle),
            MeshMaterial3d(terrain_mat),
            Transform::default(),
            TerrainMesh,
        ));
    }

    // ---- Update flies ----
    let current_fly_ids: std::collections::HashSet<usize> =
        (0..frame.fly_positions.len()).collect();

    let stale: Vec<usize> = entity_map
        .flies
        .keys()
        .filter(|k| !current_fly_ids.contains(k))
        .copied()
        .collect();
    for id in stale {
        if let Some(entity) = entity_map.flies.remove(&id) {
            commands.entity(entity).despawn();
        }
    }

    for (i, &(x, y, z, energy)) in frame.fly_positions.iter().enumerate() {
        let pos = Vec3::new(x * VOXEL_SIZE, z * VOXEL_SIZE + 0.2, y * VOXEL_SIZE);
        let energy_t: f32 = (energy / 100.0).clamp(0.0, 1.0);
        let fly_color = Color::srgb(0.2 + 0.6 * (1.0 - energy_t), 0.2, 0.2 + 0.6 * energy_t);

        if let Some(&entity) = entity_map.flies.get(&i) {
            commands
                .entity(entity)
                .insert(Transform::from_translation(pos).with_scale(Vec3::splat(FLY_BODY_SCALE)));
        } else {
            let mesh = meshes.add(Sphere::new(1.0).mesh().ico(3).unwrap());
            let mat = materials.add(StandardMaterial {
                base_color: fly_color,
                ..default()
            });
            let entity = commands
                .spawn((
                    Mesh3d(mesh),
                    MeshMaterial3d(mat),
                    Transform::from_translation(pos).with_scale(Vec3::splat(FLY_BODY_SCALE)),
                    FlyEntity(i),
                ))
                .id();
            entity_map.flies.insert(i, entity);
        }
    }

    // ---- Update plants ----
    let current_plant_ids: std::collections::HashSet<usize> =
        (0..frame.plant_positions.len()).collect();
    let stale: Vec<usize> = entity_map
        .plants
        .keys()
        .filter(|k| !current_plant_ids.contains(k))
        .copied()
        .collect();
    for id in stale {
        if let Some(entity) = entity_map.plants.remove(&id) {
            commands.entity(entity).despawn();
        }
    }

    for (i, &(x, y, h_mm)) in frame.plant_positions.iter().enumerate() {
        let h: f32 = h_mm * 0.001; // mm to meters
        let pos = Vec3::new(x * VOXEL_SIZE, h * 0.5, y * VOXEL_SIZE);
        let scale = Vec3::new(PLANT_SCALE, h.max(0.05), PLANT_SCALE);

        if let Some(&entity) = entity_map.plants.get(&i) {
            commands
                .entity(entity)
                .insert(Transform::from_translation(pos).with_scale(scale));
        } else {
            let mesh = meshes.add(Cylinder::new(0.5, 1.0));
            let greenness = (h / 0.2).clamp(0.0, 1.0);
            let mat = materials.add(StandardMaterial {
                base_color: Color::srgb(0.1, 0.35 + 0.4 * greenness, 0.1),
                ..default()
            });
            let entity = commands
                .spawn((
                    Mesh3d(mesh),
                    MeshMaterial3d(mat),
                    Transform::from_translation(pos).with_scale(scale),
                    PlantEntity(i),
                ))
                .id();
            entity_map.plants.insert(i, entity);
        }
    }

    // ---- Update fruits ----
    let current_fruit_ids: std::collections::HashSet<usize> =
        (0..frame.fruit_positions.len()).collect();
    let stale: Vec<usize> = entity_map
        .fruits
        .keys()
        .filter(|k| !current_fruit_ids.contains(k))
        .copied()
        .collect();
    for id in stale {
        if let Some(entity) = entity_map.fruits.remove(&id) {
            commands.entity(entity).despawn();
        }
    }

    for (i, &(x, y)) in frame.fruit_positions.iter().enumerate() {
        let pos = Vec3::new(x * VOXEL_SIZE, 0.05, y * VOXEL_SIZE);

        if !entity_map.fruits.contains_key(&i) {
            let mesh = meshes.add(Sphere::new(1.0).mesh().ico(2).unwrap());
            let mat = materials.add(StandardMaterial {
                base_color: Color::srgb(0.9, 0.3, 0.1),
                ..default()
            });
            let entity = commands
                .spawn((
                    Mesh3d(mesh),
                    MeshMaterial3d(mat),
                    Transform::from_translation(pos).with_scale(Vec3::splat(FRUIT_SCALE)),
                    FruitEntity(i),
                ))
                .id();
            entity_map.fruits.insert(i, entity);
        }
    }

    // ---- Update HUD ----
    let snap = &frame.snapshot;
    for mut text in hud_query.iter_mut() {
        **text = format!(
            "oNeura 3D | t={:.1}s | flies={} | plants={} | fruits={} | food={:.0} | energy={:.0} | cells={:.0}",
            snap.time_s,
            frame.fly_positions.len(),
            frame.plant_positions.len(),
            frame.fruit_positions.len(),
            snap.food_remaining,
            snap.avg_fly_energy,
            snap.total_plant_cells,
        );
    }
}

fn camera_control(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut mouse_wheel: EventReader<bevy::input::mouse::MouseWheel>,
    mut mouse_motion: EventReader<bevy::input::mouse::MouseMotion>,
    mouse_button: Res<ButtonInput<MouseButton>>,
    time: Res<Time>,
    mut query: Query<(&mut OrbitCamera, &mut Transform)>,
) {
    let dt = time.delta_secs();

    for (mut orbit, mut transform) in query.iter_mut() {
        let pan_speed = 5.0 * dt;
        let forward = Vec3::new(orbit.yaw.sin(), 0.0, orbit.yaw.cos()).normalize();
        let right = Vec3::new(orbit.yaw.cos(), 0.0, -orbit.yaw.sin()).normalize();

        if keyboard.pressed(KeyCode::KeyW) {
            orbit.focus += forward * pan_speed;
        }
        if keyboard.pressed(KeyCode::KeyS) {
            orbit.focus -= forward * pan_speed;
        }
        if keyboard.pressed(KeyCode::KeyA) {
            orbit.focus -= right * pan_speed;
        }
        if keyboard.pressed(KeyCode::KeyD) {
            orbit.focus += right * pan_speed;
        }
        if keyboard.pressed(KeyCode::KeyQ) {
            orbit.focus.y -= pan_speed;
        }
        if keyboard.pressed(KeyCode::KeyE) {
            orbit.focus.y += pan_speed;
        }

        // Mouse drag orbit (right button)
        if mouse_button.pressed(MouseButton::Right) {
            for motion in mouse_motion.read() {
                orbit.yaw -= motion.delta.x * 0.005;
                orbit.pitch = (orbit.pitch + motion.delta.y * 0.005).clamp(0.1, 1.5);
            }
        } else {
            mouse_motion.read().for_each(drop);
        }

        // Scroll zoom
        for scroll in mouse_wheel.read() {
            orbit.distance = (orbit.distance - scroll.y * 0.5).clamp(2.0, 50.0);
        }

        // Toggle auto-follow with F key
        if keyboard.just_pressed(KeyCode::KeyF) {
            orbit.auto_follow = match orbit.auto_follow {
                Some(_) => None,
                None => Some(0),
            };
        }

        let pos = orbit_position(&orbit);
        transform.translation = pos;
        transform.look_at(orbit.focus, Vec3::Y);
    }
}

fn auto_follow_system(
    mut query: Query<&mut OrbitCamera>,
    fly_query: Query<(&FlyEntity, &Transform)>,
) {
    for mut orbit in query.iter_mut() {
        if let Some(fly_idx) = orbit.auto_follow {
            for (fly, transform) in fly_query.iter() {
                if fly.0 == fly_idx {
                    let target = transform.translation;
                    orbit.focus = orbit.focus.lerp(target, 0.1);
                    break;
                }
            }
        }
    }
}

// ============================================================================
// App
// ============================================================================

fn main() {
    let config = parse_cli();

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "oNeura Terrarium 3D".into(),
                resolution: (1280.0, 720.0).into(),
                present_mode: PresentMode::AutoVsync,
                ..default()
            }),
            ..default()
        }))
        .insert_resource(config)
        .add_systems(Startup, (setup_scene, setup_hud))
        .add_systems(Update, (
            consume_sim_frames,
            camera_control,
            auto_follow_system,
        ))
        .run();
}
