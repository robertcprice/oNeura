//! Terrarium 3D Viewer — Bevy 0.15 real-time visualization of the oNeura terrarium.
//!
//! Architecture:
//! - `TerrariumWorld` runs on a background thread, produces snapshots via channel
//! - Bevy consumes snapshots and spawns/updates/despawns entities
//! - Orbital camera with WASD pan, scroll zoom, auto-follow
//! - Bio-pulse custom material for living ecosystem visual effects
//! - Shared mesh library with LOD for performance
//! - Water rendering with translucent animated material
//!
//! Usage:
//!   cargo run -p oneuro-3d --profile fast --bin terrarium_3d -- --seed 42 --fps 30

use bevy::hierarchy::DespawnRecursiveExt;
use bevy::prelude::*;
use bevy::render::mesh::Indices;
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::render_resource::PrimitiveTopology;
use bevy::render::view::screenshot::{save_to_disk, Screenshot};
use bevy::window::PresentMode;
use oneura_core::botany::{BotanicalGrowthForm, MorphNode, NodeType};
use oneura_core::organism_metabolism::OrganismMetabolism;
use oneura_core::terrarium::archive::TerrariumWorldArchive;
use oneura_core::terrarium::reporting::{
    apply_archive_name_assignments, apply_world_name_assignments, format_archive_lineage,
    format_archive_organism_listing, format_archive_summary, format_world_lineage,
    format_world_organism_listing, format_world_summary, parse_name_assignment,
    OrganismNameAssignment,
};
use oneura_core::terrarium::visual_projection::{
    fly_visual_response, mean_field as mean_visual_field,
    mean_wind_speed as mean_visual_wind_speed, plant_visual_response, quantize_rgb,
    quantized_surface_height, sample_visual_air, sample_visual_chemistry, terrain_voxel_batches,
    water_visual_response_with_chemistry, TerrariumEarthwormVisualResponse,
    TerrariumFlyVisualResponse, TerrariumFruitVisualResponse, TerrariumNematodeVisualResponse,
    TerrariumPlantVisualResponse, TerrariumSeedVisualResponse, TerrariumSoilSurfaceClass,
    TerrariumSoilSurfaceVisualResponse, TerrariumTerrainVoxelBatch, TerrariumVoxelMaterialClass,
    TerrariumWaterCycleInputs, TerrariumWaterVisualResponse,
};
use oneura_core::terrarium::{
    resolve_seed_provenance, TerrariumAtmosphereFrame, TerrariumDemoPreset, TerrariumPlantSnapshot,
    TerrariumWorld, TerrariumWorldSnapshot,
};
use oneura_core::terrarium_web_handlers::{
    annotations_handler, archive_handler, auth_token, checkpoint_handler, ecosystem_run_handler,
    ecosystem_snapshot_handler, ecosystem_start_handler, ecosystem_step_handler, export_bundle,
    frame_loop, import_checkpoint_handler, index_handler, inspect_handler,
    organism_lineage_handler, organism_rename_handler, organisms_handler, snapshot_handler,
    snapshot_history_handler, tournament_delete, tournament_genome, tournament_leaderboard,
    tournament_submit, ws_handler,
};
use oneura_core::terrarium_web_state::AppState;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::net::TcpListener;
use std::sync::{mpsc, Mutex};
use std::time::{Duration, Instant};
use tower_http::cors::{Any, CorsLayer};

// ============================================================================
// Constants
// ============================================================================

const VOXEL_SIZE: f32 = 0.5;
const FLY_BODY_SCALE: f32 = 0.22;
const PLANT_SCALE: f32 = 0.42;
const FRUIT_SCALE: f32 = 0.14;
const PIXEL_COLOR_LEVELS: usize = 5;
const PIXEL_WORLD_STEP: f32 = VOXEL_SIZE / 8.0;
const PIXEL_SCALE_STEP: f32 = 0.02;
const DEFAULT_FPS: f32 = 30.0;
const FLAG_SUBSTRATE: f32 = 1.0;
const FLAG_PLANT: f32 = 4.0;
const FLAG_FLY: f32 = 8.0;
const FLAG_FRUIT: f32 = 16.0;

type BioMaterial = StandardMaterial;

// ============================================================================
// CLI Args
// ============================================================================

#[derive(Debug, Clone, Resource)]
struct ViewerConfig {
    seed: Option<u64>,
    fps: f32,
    preset: TerrariumDemoPreset,
    coupling: bool,
    use_bio_shader: bool,
    native_scene: bool,
    port: Option<u16>,
    open_browser: bool,
    screenshot_out: Option<String>,
    screenshot_after_s: f32,
    checkpoint_in: Option<String>,
    checkpoint_out: Option<String>,
    archive_in: Option<String>,
    archive_out: Option<String>,
    query_only: bool,
    list_organisms: bool,
    lineage_ids: Vec<u64>,
    name_assignments: Vec<OrganismNameAssignment>,
}

impl Default for ViewerConfig {
    fn default() -> Self {
        Self {
            seed: None,
            fps: DEFAULT_FPS,
            preset: TerrariumDemoPreset::Demo,
            coupling: true,
            use_bio_shader: true,
            native_scene: false,
            port: None,
            open_browser: true,
            screenshot_out: None,
            screenshot_after_s: 2.0,
            checkpoint_in: None,
            checkpoint_out: None,
            archive_in: None,
            archive_out: None,
            query_only: false,
            list_organisms: false,
            lineage_ids: Vec::new(),
            name_assignments: Vec::new(),
        }
    }
}

fn print_usage(program_name: &str) {
    eprintln!("oNeura Terrarium 3D Desktop Viewer");
    eprintln!();
    eprintln!("Usage: {program_name} [OPTIONS]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --seed <N>               World seed (default: natural entropy)");
    eprintln!("  --fps <N>                Target framerate (default: 30)");
    eprintln!("  --preset <NAME>          Demo preset: demo | terrarium | aquarium");
    eprintln!("  --load-checkpoint <PATH> Start from a saved terrarium checkpoint");
    eprintln!("  --save-checkpoint <PATH> Save a terrarium checkpoint on exit");
    eprintln!("  --load-archive <PATH>    Inspect a terrarium archive and exit");
    eprintln!("  --save-archive <PATH>    Save a terrarium archive on exit");
    eprintln!("  --name-organism <ID=NAME> Assign a display name to an organism");
    eprintln!("  --list-organisms         Print tracked organism identities");
    eprintln!("  --show-lineage <ID>      Print the lineage chain for one organism");
    eprintln!("  --query-only             Apply query/save actions and exit");
    eprintln!("  --native-scene           Force native Bevy rendering instead of the web shell");
    eprintln!("  --screenshot-out <PATH>  Save a screenshot and exit");
    eprintln!("  --screenshot-after <S>   Delay screenshot capture (default: 2.0)");
    eprintln!("  --port <PORT>            Preferred localhost port for desktop web shell");
    eprintln!("  --no-open                Do not open the browser for desktop web shell");
    eprintln!("  --no-coupling            Disable ecosystem coupling");
    eprintln!("  --no-bio-shader          Disable bio material shader");
    eprintln!("  --help, -h               Show this help");
}

fn parse_cli(program_name: &str) -> Result<ViewerConfig, String> {
    let mut config = ViewerConfig::default();
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" if i + 1 < args.len() => {
                config.seed = args[i + 1].parse().ok();
                i += 2;
            }
            "--load-checkpoint" if i + 1 < args.len() => {
                config.checkpoint_in = Some(args[i + 1].clone());
                i += 2;
            }
            "--save-checkpoint" if i + 1 < args.len() => {
                config.checkpoint_out = Some(args[i + 1].clone());
                i += 2;
            }
            "--load-archive" if i + 1 < args.len() => {
                config.archive_in = Some(args[i + 1].clone());
                i += 2;
            }
            "--save-archive" if i + 1 < args.len() => {
                config.archive_out = Some(args[i + 1].clone());
                i += 2;
            }
            "--name-organism" if i + 1 < args.len() => {
                config
                    .name_assignments
                    .push(parse_name_assignment(&args[i + 1])?);
                i += 2;
            }
            "--list-organisms" => {
                config.list_organisms = true;
                i += 1;
            }
            "--show-lineage" if i + 1 < args.len() => {
                config
                    .lineage_ids
                    .push(args[i + 1].parse().map_err(|_| "bad --show-lineage")?);
                i += 2;
            }
            "--query-only" => {
                config.query_only = true;
                i += 1;
            }
            "--fps" if i + 1 < args.len() => {
                config.fps = args[i + 1].parse().unwrap_or(DEFAULT_FPS);
                i += 2;
            }
            "--preset" if i + 1 < args.len() => {
                config.preset =
                    TerrariumDemoPreset::parse(&args[i + 1]).unwrap_or(TerrariumDemoPreset::Demo);
                i += 2;
            }
            "--no-coupling" => {
                config.coupling = false;
                i += 1;
            }
            "--no-bio-shader" => {
                config.use_bio_shader = false;
                i += 1;
            }
            "--native-scene" => {
                config.native_scene = true;
                i += 1;
            }
            "--port" if i + 1 < args.len() => {
                config.port = args[i + 1].parse().ok();
                i += 2;
            }
            "--no-open" => {
                config.open_browser = false;
                i += 1;
            }
            "--screenshot-out" if i + 1 < args.len() => {
                config.screenshot_out = Some(args[i + 1].clone());
                i += 2;
            }
            "--screenshot-after" if i + 1 < args.len() => {
                config.screenshot_after_s = args[i + 1].parse().unwrap_or(2.0);
                i += 2;
            }
            "--help" | "-h" => {
                print_usage(program_name);
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }
    if config.checkpoint_in.is_some() && config.archive_in.is_some() {
        return Err("use either --load-checkpoint or --load-archive, not both".into());
    }
    if config.archive_in.is_some() && config.checkpoint_out.is_some() {
        return Err("cannot emit a checkpoint when inspecting an archive".into());
    }
    Ok(config)
}

fn load_or_create_world(
    config: &ViewerConfig,
    source_label: &str,
    use_gpu_substrate: bool,
) -> Result<(TerrariumWorld, TerrariumDemoPreset), String> {
    let mut world = if let Some(path) = config.checkpoint_in.as_deref() {
        TerrariumWorld::load_checkpoint(path)
            .map_err(|e| format!("failed to load terrarium checkpoint: {e}"))?
    } else {
        let seed_provenance = resolve_seed_provenance(config.seed, source_label);
        let mut world =
            TerrariumWorld::demo_preset(seed_provenance.seed, use_gpu_substrate, config.preset)
                .map_err(|e| format!("failed to create terrarium world: {e}"))?;
        world.set_seed_provenance(seed_provenance);
        world
    };
    apply_world_name_assignments(&mut world, &config.name_assignments)?;
    let present_preset =
        TerrariumDemoPreset::infer_from_config(&world.config).unwrap_or(config.preset);
    Ok((world, present_preset))
}

fn print_world_reports(world: &TerrariumWorld, config: &ViewerConfig) -> Result<(), String> {
    println!("{}", format_world_summary(world));
    if config.list_organisms {
        println!();
        println!("{}", format_world_organism_listing(world));
    }
    for lineage_id in &config.lineage_ids {
        println!();
        println!("{}", format_world_lineage(world, *lineage_id)?);
    }
    Ok(())
}

fn print_archive_reports(
    archive: &TerrariumWorldArchive,
    config: &ViewerConfig,
) -> Result<(), String> {
    println!("{}", format_archive_summary(archive));
    if config.list_organisms {
        println!();
        println!("{}", format_archive_organism_listing(archive));
    }
    for lineage_id in &config.lineage_ids {
        println!();
        println!("{}", format_archive_lineage(archive, *lineage_id)?);
    }
    Ok(())
}

fn save_world_outputs(world: &mut TerrariumWorld, config: &ViewerConfig) -> Result<(), String> {
    if let Some(path) = config.checkpoint_out.as_deref() {
        world
            .save_checkpoint(path)
            .map_err(|error| format!("failed to save terrarium checkpoint: {error}"))?;
        println!("saved terrarium checkpoint to {path}");
    }
    if let Some(path) = config.archive_out.as_deref() {
        world
            .save_archive(path)
            .map_err(|error| format!("failed to save terrarium archive: {error}"))?;
        println!("saved terrarium archive to {path}");
    }
    Ok(())
}

fn inspect_archive(config: &ViewerConfig) -> Result<(), String> {
    let path = config
        .archive_in
        .as_deref()
        .ok_or_else(|| "archive inspection requires --load-archive".to_string())?;
    let mut archive = TerrariumWorldArchive::load_from_path(path)?;
    apply_archive_name_assignments(&mut archive, &config.name_assignments)?;
    print_archive_reports(&archive, config)?;
    if let Some(path) = config.archive_out.as_deref() {
        archive.save_to_path(path)?;
        println!("saved terrarium archive to {path}");
    }
    Ok(())
}

// ============================================================================
// Simulation Resources
// ============================================================================

#[derive(Resource)]
struct SimChannel {
    rx: Mutex<mpsc::Receiver<SimFrame>>,
}

#[derive(Resource)]
struct InitialSimState {
    world: Option<TerrariumWorld>,
    checkpoint_out: Option<String>,
    archive_out: Option<String>,
}

struct SimFrame {
    snapshot: TerrariumWorldSnapshot,
    surface_relief: Vec<f32>,
    terrain_voxels: Vec<TerrariumTerrainVoxelBatch>,
    atmosphere: TerrariumAtmosphereFrame,
    width: usize,
    height: usize,
    fly_states: Vec<(f32, f32, f32, f32, f32, bool, f32)>,
    fly_visuals: Vec<TerrariumFlyVisualResponse>,
    plant_snapshots: Vec<TerrariumPlantSnapshot>,
    plant_visuals: Vec<TerrariumPlantVisualResponse>,
    fruit_positions: Vec<(f32, f32)>,
    fruit_visuals: Vec<TerrariumFruitVisualResponse>,
    seed_states: Vec<(f32, f32, u32)>,
    seed_visuals: Vec<TerrariumSeedVisualResponse>,
    water_cells: Vec<(f32, f32, f32)>,
    water_visuals: Vec<TerrariumWaterVisualResponse>,
    earthworms: Vec<(usize, usize, TerrariumEarthwormVisualResponse)>,
    nematodes: Vec<(usize, usize, TerrariumNematodeVisualResponse)>,
    soil_surface: Vec<(usize, usize, TerrariumSoilSurfaceVisualResponse)>,
}

#[derive(Resource, Default)]
struct EntityMap {
    terrain_batches: HashMap<TerrariumVoxelMaterialClass, Entity>,
    terrain_batch_fingerprints: HashMap<TerrariumVoxelMaterialClass, u64>,
    flies: HashMap<usize, Entity>,
    plants: HashMap<usize, Entity>,
    fruits: HashMap<usize, Entity>,
    seeds: HashMap<usize, Entity>,
    waters: HashMap<usize, Entity>,
    earthworms: HashMap<usize, Entity>,
    nematodes: HashMap<usize, Entity>,
    soil_surface: HashMap<usize, Entity>,
}

#[derive(Resource)]
struct MeshLibrary {
    fly_body: Handle<Mesh>,
    fly_head: Handle<Mesh>,
    fly_wing: Handle<Mesh>,
    plant_trunk: Handle<Mesh>,
    plant_leaf: Handle<Mesh>,
    fruit: Handle<Mesh>,
    seed: Handle<Mesh>,
    soil_marker: Handle<Mesh>,
    water_quad: Handle<Mesh>,
}

#[derive(Resource)]
struct TerrainMaterialLibrary {
    bedrock: Handle<BioMaterial>,
    subsoil: Handle<BioMaterial>,
    surface: Handle<BioMaterial>,
    water: Handle<BioMaterial>,
}

#[derive(Resource, Default)]
struct SimTime(f32);

#[derive(Resource)]
struct ScreenshotCapture {
    path: Option<String>,
    capture_after_s: f32,
    requested: bool,
    completed: bool,
}

// ============================================================================
// Components
// ============================================================================

#[derive(Component, Clone, Copy, PartialEq, Eq)]
struct TerrainBatchEntity(TerrariumVoxelMaterialClass);

#[derive(Component)]
struct FlyEntity(usize);

#[derive(Component, Clone, Copy, PartialEq, Eq)]
struct FlyPart(FlyPartRole);

#[derive(Clone, Copy, PartialEq, Eq)]
enum FlyPartRole {
    Thorax,
    Abdomen,
    Head,
    LeftWing,
    RightWing,
    LeftLeg,
    RightLeg,
    Proboscis,
}

#[derive(Component, Clone, Copy, PartialEq, Eq)]
struct FlyRigLayout {
    has_proboscis: bool,
}

#[derive(Component)]
struct PlantEntity;

#[derive(Component, Clone, Copy, PartialEq)]
struct PlantPart {
    node_index: usize,
    node_type: NodeType,
}

#[derive(Component, Clone, Copy, PartialEq, Eq)]
struct PlantRigFingerprint(u64);

#[derive(Component)]
struct FruitEntity;

#[derive(Component)]
struct SeedEntity;

#[derive(Component)]
struct WaterEntity;

#[derive(Component)]
struct EarthwormEntity;

#[derive(Component, Clone, Copy, PartialEq, Eq)]
struct EarthwormPart {
    segment_index: usize,
}

#[derive(Component, Clone, Copy, PartialEq, Eq)]
struct EarthwormRigLayout {
    segment_count: usize,
}

#[derive(Component)]
struct NematodeEntity;

#[derive(Component, Clone, Copy, PartialEq, Eq)]
struct NematodePart {
    segment_index: Option<usize>,
}

#[derive(Component, Clone, Copy, PartialEq, Eq)]
struct NematodeRigLayout {
    segment_count: usize,
    has_stylet: bool,
}

#[derive(Component)]
struct SoilSurfaceEntity;

#[derive(Component)]
struct OrbitCamera {
    focus: Vec3,
    distance: f32,
    yaw: f32,
    pitch: f32,
    auto_follow: Option<usize>,
    needs_framing: bool,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            focus: Vec3::new(11.0, 0.6, 8.0),
            distance: 34.0,
            yaw: -0.55,
            pitch: 0.95,
            auto_follow: None,
            needs_framing: true,
        }
    }
}

#[derive(Component)]
struct HudText;

#[derive(Component)]
struct HudControls;

// ============================================================================
// Mesh Helpers
// ============================================================================

fn push_box_faces(
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
    center: Vec3,
    half: Vec3,
    color: [f32; 4],
) {
    let corners = [
        center + Vec3::new(-half.x, -half.y, -half.z),
        center + Vec3::new(half.x, -half.y, -half.z),
        center + Vec3::new(half.x, half.y, -half.z),
        center + Vec3::new(-half.x, half.y, -half.z),
        center + Vec3::new(-half.x, -half.y, half.z),
        center + Vec3::new(half.x, -half.y, half.z),
        center + Vec3::new(half.x, half.y, half.z),
        center + Vec3::new(-half.x, half.y, half.z),
    ];
    let faces = [
        ([1, 0, 3, 2], [0.0, 0.0, -1.0]),
        ([4, 5, 6, 7], [0.0, 0.0, 1.0]),
        ([4, 0, 3, 7], [-1.0, 0.0, 0.0]),
        ([1, 5, 6, 2], [1.0, 0.0, 0.0]),
        ([3, 7, 6, 2], [0.0, 1.0, 0.0]),
        ([0, 1, 5, 4], [0.0, -1.0, 0.0]),
    ];
    for (face_indices, normal) in faces {
        let base = positions.len() as u32;
        for corner_index in face_indices {
            let p = corners[corner_index];
            positions.push([p.x, p.y, p.z]);
            normals.push(normal);
            colors.push(color);
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

fn create_voxel_batch_mesh(batch: &TerrariumTerrainVoxelBatch) -> Mesh {
    let mut positions = Vec::with_capacity(batch.instances.len() * 24);
    let mut normals = Vec::with_capacity(batch.instances.len() * 24);
    let mut uvs = Vec::with_capacity(batch.instances.len() * 24);
    let mut colors = Vec::with_capacity(batch.instances.len() * 24);
    let mut indices = Vec::with_capacity(batch.instances.len() * 36);
    let alpha = if batch.class == TerrariumVoxelMaterialClass::Water {
        0.86
    } else {
        1.0
    };

    for instance in &batch.instances {
        let center = Vec3::new(instance.x * VOXEL_SIZE, instance.y, instance.z * VOXEL_SIZE);
        let half = Vec3::new(VOXEL_SIZE * 0.5, instance.scale_y * 0.5, VOXEL_SIZE * 0.5);
        let base_len = positions.len();
        push_box_faces(
            &mut positions,
            &mut normals,
            &mut colors,
            &mut indices,
            center,
            half,
            [instance.rgb[0], instance.rgb[1], instance.rgb[2], alpha],
        );
        for _ in base_len..positions.len() {
            uvs.push([0.0, 0.0]); // Dummy UVs to prevent Metal crashing
        }
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

fn create_smooth_surface_mesh(
    width: usize,
    height: usize,
    surface_relief: &[f32],
    batch: &TerrariumTerrainVoxelBatch,
) -> Mesh {
    let mut positions = Vec::with_capacity(width * height);
    let mut normals = Vec::with_capacity(width * height);
    let mut uvs = Vec::with_capacity(width * height);
    let mut colors = Vec::with_capacity(width * height);
    let mut indices = Vec::with_capacity((width - 1) * (height - 1) * 6);

    for z in 0..height {
        for x in 0..width {
            let idx = (z * width + x).clamp(0, surface_relief.len().saturating_sub(1));
            let y = surface_relief.get(idx).copied().unwrap_or(0.0);
            positions.push([x as f32 * VOXEL_SIZE, y, z as f32 * VOXEL_SIZE]);
            normals.push([0.0, 1.0, 0.0]);
            uvs.push([x as f32 / width as f32, z as f32 / height as f32]);
            
            // Procedural detail noise based on coordinates
            let noise = ((x * 12345 + z * 67890) % 100) as f32 / 100.0;
            
            // Give it a nice natural grass/dirt color based on height
            if y < 0.2 {
                let dirt = 0.3 + noise * 0.1;
                colors.push([dirt, dirt * 0.8, dirt * 0.6, 1.0]); // Varied Dirt
            } else {
                let green_base = 0.3 + (y * 0.1).min(0.2);
                let detail = noise * 0.15;
                colors.push([0.15 + detail, green_base + detail, 0.1 + detail * 0.5, 1.0]); // Varied Grass
            }
        }
    }

    for z in 0..height.saturating_sub(1) {
        for x in 0..width.saturating_sub(1) {
            let top_left = (z * width + x) as u32;
            let top_right = top_left + 1;
            let bottom_left = ((z + 1) * width + x) as u32;
            let bottom_right = bottom_left + 1;

            indices.extend_from_slice(&[
                top_left, bottom_left, top_right,
                top_right, bottom_left, bottom_right,
            ]);
        }
    }

    for z in 0..height.saturating_sub(1) {
        for x in 0..width.saturating_sub(1) {
            let i0 = z * width + x;
            let i1 = (z + 1) * width + x;
            let i2 = z * width + x + 1;

            let p0 = Vec3::from(positions[i0]);
            let p1 = Vec3::from(positions[i1]);
            let p2 = Vec3::from(positions[i2]);
            let normal = (p1 - p0).cross(p2 - p0).normalize_or_zero();
            if normal.length_squared() > 0.0 {
                normals[i0] = normal.into();
                normals[i1] = normal.into();
                normals[i2] = normal.into();
            }
        }
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

fn terrain_surface_height(
    surface_relief: &[f32],
    width: usize,
    height: usize,
    x: f32,
    y: f32,
) -> f32 {
    if surface_relief.is_empty() || width == 0 || height == 0 {
        return 0.0;
    }
    let ix = x.floor().clamp(0.0, width.saturating_sub(1) as f32) as usize;
    let iy = y.floor().clamp(0.0, height.saturating_sub(1) as f32) as usize;
    quantized_surface_height(surface_relief[iy * width + ix])
}

fn voxel_water_surface_height(
    terrain_voxels: &[TerrariumTerrainVoxelBatch],
    surface_y: f32,
    x: usize,
    y: usize,
) -> f32 {
    let target_x = x as f32 + 0.5;
    let target_y = y as f32 + 0.5;
    let mut best = surface_y;
    for batch in terrain_voxels {
        if batch.class != TerrariumVoxelMaterialClass::Water {
            continue;
        }
        for instance in &batch.instances {
            if (instance.x - target_x).abs() <= 0.01 && (instance.z - target_y).abs() <= 0.01 {
                best = best.max(instance.y + instance.scale_y * 0.5);
            }
        }
    }
    best
}

fn mean_field(field: &[f32], fallback: f32) -> f32 {
    mean_visual_field(field, fallback)
}

fn mean_wind_speed(atmosphere: &TerrariumAtmosphereFrame) -> f32 {
    mean_visual_wind_speed(atmosphere)
}

fn upsert_simple_entity<F>(
    commands: &mut Commands,
    map: &mut HashMap<usize, Entity>,
    materials: &mut Assets<BioMaterial>,
    query: &mut Query<(&mut Transform, &mut MeshMaterial3d<BioMaterial>), F>,
    index: usize,
    mesh: Handle<Mesh>,
    material_base: StandardMaterial,
    transform: Transform,
    spawn_components: impl Bundle,
) where
    F: bevy::ecs::query::QueryFilter,
{
    if let Some(entity) = map.get(&index).copied() {
        if let Ok((mut existing_transform, mut existing_material)) = query.get_mut(entity) {
            *existing_transform = transform;
            existing_material.0 =
                sync_bio_material(materials, Some(&existing_material.0), material_base);
            return;
        }
        commands.entity(entity).try_despawn_recursive();
        map.remove(&index);
    }

    let material = sync_bio_material(materials, None, material_base);
    let entity = commands
        .spawn((
            Mesh3d(mesh),
            MeshMaterial3d(material),
            transform,
            spawn_components,
        ))
        .id();
    map.insert(index, entity);
}

fn add_bio_material(
    materials: &mut Assets<BioMaterial>,
    base: StandardMaterial,
    _humidity: f32,
    _activity: f32,
    _daylight: f32,
    _time_s: f32,
    _entity_flag: f32,
    _energy: f32,
    _stress: f32,
    _enabled: bool,
) -> Handle<BioMaterial> {
    materials.add(base)
}

fn sync_bio_material(
    materials: &mut Assets<BioMaterial>,
    existing: Option<&Handle<BioMaterial>>,
    base: StandardMaterial,
) -> Handle<BioMaterial> {
    if let Some(handle) = existing {
        if let Some(material) = materials.get_mut(handle) {
            *material = base.clone();
            return handle.clone();
        }
    }
    materials.add(base)
}

fn snap_scalar(value: f32, step: f32) -> f32 {
    if step <= 0.0 {
        value
    } else {
        (value / step).round() * step
    }
}

fn snap_translation(value: Vec3) -> Vec3 {
    Vec3::new(
        snap_scalar(value.x, PIXEL_WORLD_STEP),
        snap_scalar(value.y, PIXEL_WORLD_STEP),
        snap_scalar(value.z, PIXEL_WORLD_STEP),
    )
}

fn snap_scale(value: Vec3) -> Vec3 {
    Vec3::new(
        snap_scalar(value.x.max(PIXEL_SCALE_STEP), PIXEL_SCALE_STEP).max(PIXEL_SCALE_STEP),
        snap_scalar(value.y.max(PIXEL_SCALE_STEP), PIXEL_SCALE_STEP).max(PIXEL_SCALE_STEP),
        snap_scalar(value.z.max(PIXEL_SCALE_STEP), PIXEL_SCALE_STEP).max(PIXEL_SCALE_STEP),
    )
}

fn snap_transform(mut transform: Transform) -> Transform {
    transform.translation = snap_translation(transform.translation);
    transform.scale = snap_scale(transform.scale);
    transform
}

fn pixel_rgb(rgb: [f32; 3]) -> [f32; 3] {
    // Stop quantizing, just pass through so we don't get black trees.
    rgb
}

fn pixel_color(rgb: [f32; 3]) -> Color {
    let [r, g, b] = pixel_rgb(rgb);
    Color::srgb(r, g, b)
}

fn pixel_alpha_color(rgb: [f32; 3], alpha: f32) -> Color {
    let [r, g, b] = pixel_rgb(rgb);
    Color::srgba(r, g, b, alpha)
}

fn terrain_material_base(class: TerrariumVoxelMaterialClass) -> StandardMaterial {
    match class {
        TerrariumVoxelMaterialClass::Bedrock => StandardMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 0.96,
            depth_bias: 0.0,
            ..default()
        },
        TerrariumVoxelMaterialClass::Subsoil => StandardMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 0.92,
            depth_bias: 0.0,
            ..default()
        },
        TerrariumVoxelMaterialClass::Surface => StandardMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 0.88,
            depth_bias: 0.0,
            ..default()
        },
        TerrariumVoxelMaterialClass::Water => StandardMaterial {
            base_color: Color::WHITE,
            alpha_mode: AlphaMode::Opaque,
            perceptual_roughness: 0.08,
            metallic: 0.22,
            depth_bias: 0.0,
            ..default()
        },
    }
}

fn terrain_batch_fingerprint(batch: &TerrariumTerrainVoxelBatch) -> u64 {
    let mut hasher = DefaultHasher::new();
    batch.class.hash(&mut hasher);
    batch.instances.len().hash(&mut hasher);
    for instance in &batch.instances {
        instance.x.to_bits().hash(&mut hasher);
        instance.y.to_bits().hash(&mut hasher);
        instance.z.to_bits().hash(&mut hasher);
        instance.scale_y.to_bits().hash(&mut hasher);
        for channel in instance.rgb {
            channel.to_bits().hash(&mut hasher);
        }
    }
    hasher.finish()
}

fn preset_camera_pose(
    preset: TerrariumDemoPreset,
    width: usize,
    height: usize,
    surface_relief: &[f32],
) -> (Vec3, f32, f32, f32) {
    let center_x = width as f32 * VOXEL_SIZE * 0.5;
    let center_z = height as f32 * VOXEL_SIZE * 0.5;
    let center_surface = terrain_surface_height(
        surface_relief,
        width,
        height,
        width as f32 * 0.5,
        height as f32 * 0.5,
    );
    match preset {
        TerrariumDemoPreset::Demo => (
            Vec3::new(center_x, center_surface + 0.8, center_z),
            34.0,
            -0.55,
            0.95,
        ),
        TerrariumDemoPreset::MicroTerrarium => (
            Vec3::new(center_x, center_surface + 0.45, center_z),
            10.5,
            -0.78,
            0.78,
        ),
        TerrariumDemoPreset::MicroAquarium => (
            Vec3::new(center_x, center_surface + 0.20, center_z),
            8.4,
            -0.72,
            0.56,
        ),
    }
}

fn sync_mesh_asset(
    meshes: &mut Assets<Mesh>,
    existing: Option<&Handle<Mesh>>,
    mesh: Mesh,
) -> Handle<Mesh> {
    if let Some(handle) = existing {
        if let Some(existing_mesh) = meshes.get_mut(handle) {
            *existing_mesh = mesh;
            return handle.clone();
        }
    }
    meshes.add(mesh)
}

fn soil_surface_shape_scale(class: TerrariumSoilSurfaceClass) -> Vec3 {
    match class {
        TerrariumSoilSurfaceClass::EarthwormCast => Vec3::new(0.18, 0.05, 0.09),
        TerrariumSoilSurfaceClass::NematodeBloom => Vec3::new(0.08, 0.03, 0.16),
        TerrariumSoilSurfaceClass::NitrifierCrust => Vec3::new(0.16, 0.02, 0.10),
        TerrariumSoilSurfaceClass::DenitrifierFilm => Vec3::new(0.15, 0.01, 0.15),
        TerrariumSoilSurfaceClass::MycorrhizalPatch => Vec3::new(0.14, 0.04, 0.14),
        TerrariumSoilSurfaceClass::MicrobialMat => Vec3::new(0.10, 0.05, 0.10),
        TerrariumSoilSurfaceClass::WetDetritus => Vec3::new(0.16, 0.03, 0.12),
        TerrariumSoilSurfaceClass::Humus => Vec3::new(0.12, 0.04, 0.12),
        TerrariumSoilSurfaceClass::Mineral => Vec3::splat(0.08),
    }
}

fn plant_node_type_code(node_type: NodeType) -> u8 {
    match node_type {
        NodeType::Trunk => 0,
        NodeType::Branch => 1,
        NodeType::Leaf => 2,
        NodeType::Fruit => 3,
        NodeType::Bud => 4,
    }
}

fn plant_morphology_fingerprint(plant: &TerrariumPlantSnapshot) -> u64 {
    let mut hasher = DefaultHasher::new();
    plant.taxonomy_id.hash(&mut hasher);
    plant_node_type_code(match plant.growth_form {
        BotanicalGrowthForm::RosetteHerb => NodeType::Leaf,
        BotanicalGrowthForm::OrchardTree => NodeType::Trunk,
        BotanicalGrowthForm::StoneFruitTree => NodeType::Branch,
        BotanicalGrowthForm::CitrusTree => NodeType::Fruit,
        BotanicalGrowthForm::GrassClump => NodeType::Bud,
        BotanicalGrowthForm::FloatingAquatic => NodeType::Leaf,
        BotanicalGrowthForm::SubmergedAquatic => NodeType::Leaf,
    })
    .hash(&mut hasher);
    plant.morphology.len().hash(&mut hasher);
    for node in &plant.morphology {
        for value in node.position {
            value.to_bits().hash(&mut hasher);
        }
        for value in node.rotation {
            value.to_bits().hash(&mut hasher);
        }
        node.radius.to_bits().hash(&mut hasher);
        plant_node_type_code(node.node_type).hash(&mut hasher);
    }
    hasher.finish()
}

fn fly_root_transform(
    x: f32,
    y: f32,
    z: f32,
    heading: f32,
    visual: TerrariumFlyVisualResponse,
) -> Transform {
    snap_transform(
        Transform::from_translation(Vec3::new(
            x * VOXEL_SIZE,
            z * VOXEL_SIZE + 0.2,
            y * VOXEL_SIZE,
        ))
        .with_rotation(
            Quat::from_rotation_y(heading)
                * Quat::from_euler(EulerRot::XYZ, visual.pitch, 0.0, visual.roll),
        )
        .with_scale(Vec3::splat(FLY_BODY_SCALE * visual.sprite_scale)),
    )
}

fn fly_part_transform(role: FlyPartRole, visual: TerrariumFlyVisualResponse) -> Transform {
    let transform = match role {
        FlyPartRole::Thorax => {
            Transform::from_translation(Vec3::new(0.0, 0.0, 0.12)).with_scale(Vec3::new(
                0.92 * visual.thorax_scale,
                0.72 * visual.thorax_scale,
                1.02 * visual.thorax_scale,
            ))
        }
        FlyPartRole::Abdomen => {
            Transform::from_translation(Vec3::new(0.0, -0.03, -0.68 * visual.abdomen_scale))
                .with_scale(Vec3::new(
                    0.72 * visual.abdomen_scale,
                    0.58 * visual.abdomen_scale,
                    1.16 * visual.abdomen_scale,
                ))
        }
        FlyPartRole::Head => {
            Transform::from_translation(Vec3::new(0.0, 0.04, 0.94 * visual.head_scale))
                .with_scale(Vec3::splat(visual.head_scale))
        }
        FlyPartRole::LeftWing => {
            Transform::from_translation(Vec3::new(0.52 * visual.wing_span, 0.26, 0.06))
                .with_rotation(Quat::from_euler(
                    EulerRot::XYZ,
                    0.18,
                    0.0,
                    visual.wing_angle,
                ))
                .with_scale(Vec3::new(visual.wing_width, visual.wing_span, 1.0))
        }
        FlyPartRole::RightWing => {
            Transform::from_translation(Vec3::new(-0.52 * visual.wing_span, 0.26, 0.06))
                .with_rotation(Quat::from_euler(
                    EulerRot::XYZ,
                    0.18,
                    0.0,
                    -visual.wing_angle,
                ))
                .with_scale(Vec3::new(visual.wing_width, visual.wing_span, 1.0))
        }
        FlyPartRole::LeftLeg => {
            let leg_scale = 0.10 * visual.leg_span;
            Transform::from_translation(Vec3::new(-0.34 * visual.leg_span, -0.18, -0.08))
                .with_scale(Vec3::new(leg_scale, 0.03, 0.26))
        }
        FlyPartRole::RightLeg => {
            let leg_scale = 0.10 * visual.leg_span;
            Transform::from_translation(Vec3::new(0.34 * visual.leg_span, -0.18, -0.08))
                .with_scale(Vec3::new(leg_scale, 0.03, 0.26))
        }
        FlyPartRole::Proboscis => Transform::from_translation(Vec3::new(
            0.0,
            -0.04,
            1.20 * visual.head_scale + visual.proboscis_extension * 0.24,
        ))
        .with_scale(Vec3::new(0.06, 0.04, visual.proboscis_extension)),
    };
    snap_transform(transform)
}

fn fly_part_mesh(mesh_lib: &MeshLibrary, role: FlyPartRole) -> Handle<Mesh> {
    match role {
        FlyPartRole::Thorax | FlyPartRole::Abdomen => mesh_lib.fly_body.clone(),
        FlyPartRole::Head => mesh_lib.fly_head.clone(),
        FlyPartRole::LeftWing | FlyPartRole::RightWing => mesh_lib.fly_wing.clone(),
        FlyPartRole::LeftLeg | FlyPartRole::RightLeg | FlyPartRole::Proboscis => {
            mesh_lib.soil_marker.clone()
        }
    }
}

fn fly_part_material(
    role: FlyPartRole,
    body_mat: &Handle<BioMaterial>,
    wing_mat: &Handle<BioMaterial>,
) -> Handle<BioMaterial> {
    match role {
        FlyPartRole::LeftWing | FlyPartRole::RightWing => wing_mat.clone(),
        _ => body_mat.clone(),
    }
}

fn spawn_fly_parts(
    commands: &mut Commands,
    root: Entity,
    mesh_lib: &MeshLibrary,
    body_mat: &Handle<BioMaterial>,
    wing_mat: &Handle<BioMaterial>,
    visual: TerrariumFlyVisualResponse,
) {
    let mut roles = vec![
        FlyPartRole::Thorax,
        FlyPartRole::Abdomen,
        FlyPartRole::Head,
        FlyPartRole::LeftWing,
        FlyPartRole::RightWing,
        FlyPartRole::LeftLeg,
        FlyPartRole::RightLeg,
    ];
    if visual.proboscis_extension > 0.0 {
        roles.push(FlyPartRole::Proboscis);
    }
    commands.entity(root).with_children(|parent| {
        for role in roles {
            parent.spawn((
                Mesh3d(fly_part_mesh(mesh_lib, role)),
                MeshMaterial3d(fly_part_material(role, body_mat, wing_mat)),
                fly_part_transform(role, visual),
                FlyPart(role),
            ));
        }
    });
}

fn update_fly_parts<F>(
    child_entities: &[Entity],
    query: &mut Query<(&FlyPart, &mut Transform, &mut MeshMaterial3d<BioMaterial>), F>,
    materials: &mut Assets<BioMaterial>,
    body_base: &StandardMaterial,
    wing_base: &StandardMaterial,
    visual: TerrariumFlyVisualResponse,
) -> bool
where
    F: bevy::ecs::query::QueryFilter,
{
    let expected_count = if visual.proboscis_extension > 0.0 {
        8
    } else {
        7
    };
    if child_entities.len() != expected_count {
        return false;
    }
    let mut seen = [false; 8];
    for child in child_entities {
        let Ok((part, mut transform, mut material)) = query.get_mut(*child) else {
            return false;
        };
        let slot = match part.0 {
            FlyPartRole::Thorax => 0,
            FlyPartRole::Abdomen => 1,
            FlyPartRole::Head => 2,
            FlyPartRole::LeftWing => 3,
            FlyPartRole::RightWing => 4,
            FlyPartRole::LeftLeg => 5,
            FlyPartRole::RightLeg => 6,
            FlyPartRole::Proboscis => {
                if visual.proboscis_extension <= 0.0 {
                    return false;
                }
                7
            }
        };
        if seen[slot] {
            return false;
        }
        *transform = fly_part_transform(part.0, visual);
        let material_base = match part.0 {
            FlyPartRole::LeftWing | FlyPartRole::RightWing => wing_base.clone(),
            _ => body_base.clone(),
        };
        material.0 = sync_bio_material(materials, Some(&material.0), material_base);
        seen[slot] = true;
    }
    seen[..7].iter().all(|present| *present) && (visual.proboscis_extension <= 0.0 || seen[7])
}

fn plant_root_transform(
    plant: &TerrariumPlantSnapshot,
    visual: TerrariumPlantVisualResponse,
    base_y: f32,
) -> Transform {
    snap_transform(
        Transform::from_translation(Vec3::new(
            (plant.x as f32 + 0.5) * VOXEL_SIZE,
            base_y + visual.vertical_offset,
            (plant.y as f32 + 0.5) * VOXEL_SIZE,
        ))
        .with_rotation(Quat::from_euler(
            EulerRot::XYZ,
            -visual.lean_z,
            0.0,
            visual.lean_x,
        )),
    )
}

fn plant_node_position(node: &MorphNode, visual: TerrariumPlantVisualResponse) -> Vec3 {
    snap_translation(Vec3::new(
        node.position[0] * VOXEL_SIZE + visual.sway_x * (0.14 + node.position[1].max(0.0) * 0.05),
        node.position[1] * VOXEL_SIZE * 0.9,
        node.position[2] * VOXEL_SIZE + visual.sway_z * (0.14 + node.position[1].max(0.0) * 0.05),
    ))
}

fn plant_node_positions(
    plant: &TerrariumPlantSnapshot,
    visual: TerrariumPlantVisualResponse,
) -> Vec<Vec3> {
    plant
        .morphology
        .iter()
        .map(|node| plant_node_position(node, visual))
        .collect()
}

fn stem_like(node_type: NodeType) -> bool {
    matches!(
        node_type,
        NodeType::Trunk | NodeType::Branch | NodeType::Bud
    )
}

fn plant_stem_parent(morphology: &[MorphNode], node_positions: &[Vec3], index: usize) -> Vec3 {
    let current = node_positions[index];
    if morphology[index].node_type == NodeType::Trunk {
        return Vec3::ZERO;
    }
    let mut best = None;
    let mut best_score = f32::INFINITY;
    for j in (0..index).rev() {
        if !stem_like(morphology[j].node_type) {
            continue;
        }
        let candidate = node_positions[j];
        if candidate.y > current.y + 0.04 {
            continue;
        }
        let dist = candidate.distance(current);
        let score = dist + (current.y - candidate.y).abs() * 0.35;
        if score < best_score {
            best_score = score;
            best = Some(candidate);
        }
    }
    best.unwrap_or(Vec3::ZERO)
}

fn plant_part_transform(
    plant: &TerrariumPlantSnapshot,
    node_index: usize,
    node_positions: &[Vec3],
) -> Transform {
    let node = &plant.morphology[node_index];
    let node_translation = node_positions[node_index];
    let node_rotation = Quat::from_euler(
        EulerRot::XYZ,
        node.rotation[0],
        node.rotation[1],
        node.rotation[2],
    );
    let transform = match node.node_type {
        NodeType::Trunk | NodeType::Branch => {
            let start = plant_stem_parent(&plant.morphology, node_positions, node_index);
            let delta = node_translation - start;
            let length = delta.length().max(0.03);
            let radius = match plant.growth_form {
                BotanicalGrowthForm::GrassClump => {
                    (node.radius * PLANT_SCALE * 0.18).clamp(0.012, 0.040)
                }
                BotanicalGrowthForm::FloatingAquatic => {
                    (node.radius * PLANT_SCALE * 0.16).clamp(0.010, 0.032)
                }
                BotanicalGrowthForm::SubmergedAquatic => {
                    (node.radius * PLANT_SCALE * 0.20).clamp(0.012, 0.038)
                }
                _ => (node.radius * PLANT_SCALE * 0.70).clamp(0.05, 0.16),
            };
            let rotation = if delta.length_squared() > 1e-6 {
                Quat::from_rotation_arc(Vec3::Y, delta.normalize())
            } else {
                Quat::IDENTITY
            };
            Transform::from_translation((start + node_translation) * 0.5)
                .with_rotation(rotation)
                .with_scale(Vec3::new(radius, length, radius))
        }
        NodeType::Bud => {
            let scale = match plant.growth_form {
                BotanicalGrowthForm::GrassClump => 0.03,
                BotanicalGrowthForm::FloatingAquatic | BotanicalGrowthForm::SubmergedAquatic => {
                    0.04
                }
                _ => 0.05,
            };
            Transform::from_translation(node_translation).with_scale(Vec3::splat(scale))
        }
        NodeType::Leaf => {
            let canopy_scale = (plant.vitality * 0.35 + 0.65).clamp(0.55, 1.15);
            let scale = (node.radius * PLANT_SCALE * 3.0 * canopy_scale).clamp(0.12, 0.82);
            match plant.growth_form {
                BotanicalGrowthForm::GrassClump => {
                    let blade_start =
                        plant_stem_parent(&plant.morphology, node_positions, node_index);
                    let blade_delta = node_translation - blade_start;
                    let blade_len = blade_delta.length().max(0.04);
                    let blade_rotation = if blade_delta.length_squared() > 1e-6 {
                        Quat::from_rotation_arc(Vec3::Y, blade_delta.normalize())
                    } else {
                        Quat::IDENTITY
                    };
                    Transform::from_translation((blade_start + node_translation) * 0.5)
                        .with_rotation(blade_rotation)
                        .with_scale(Vec3::new(scale * 0.08, blade_len, scale * 0.08))
                }
                BotanicalGrowthForm::FloatingAquatic => {
                    Transform::from_translation(node_translation)
                        .with_rotation(Quat::from_euler(
                            EulerRot::XYZ,
                            std::f32::consts::FRAC_PI_2,
                            node.rotation[1],
                            0.0,
                        ))
                        .with_scale(Vec3::new(scale * 0.95, scale * 0.82, 1.0))
                }
                BotanicalGrowthForm::SubmergedAquatic => {
                    Transform::from_translation(node_translation)
                        .with_rotation(
                            node_rotation * Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
                        )
                        .with_scale(Vec3::new(scale * 0.24, scale * 0.88, 1.0))
                }
                BotanicalGrowthForm::OrchardTree
                | BotanicalGrowthForm::StoneFruitTree
                | BotanicalGrowthForm::CitrusTree => Transform::from_translation(node_translation)
                    .with_rotation(node_rotation)
                    .with_scale(Vec3::new(scale * 1.10, scale * 0.74, scale)),
                BotanicalGrowthForm::RosetteHerb => Transform::from_translation(node_translation)
                    .with_rotation(
                        node_rotation * Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
                    )
                    .with_scale(Vec3::new(scale * 0.52, scale * 0.96, 1.0)),
            }
        }
        NodeType::Fruit => {
            let scale = (node.radius * FRUIT_SCALE * 5.0).clamp(0.08, 0.22);
            Transform::from_translation(node_translation).with_scale(Vec3::splat(scale))
        }
    };
    snap_transform(transform)
}

fn plant_part_mesh(
    mesh_lib: &MeshLibrary,
    plant: &TerrariumPlantSnapshot,
    node_type: NodeType,
) -> Handle<Mesh> {
    match node_type {
        NodeType::Trunk | NodeType::Branch => mesh_lib.plant_trunk.clone(),
        NodeType::Bud => mesh_lib.soil_marker.clone(),
        NodeType::Leaf => match plant.growth_form {
            BotanicalGrowthForm::GrassClump
            | BotanicalGrowthForm::FloatingAquatic
            | BotanicalGrowthForm::SubmergedAquatic
            | BotanicalGrowthForm::RosetteHerb => mesh_lib.fly_wing.clone(),
            BotanicalGrowthForm::OrchardTree
            | BotanicalGrowthForm::StoneFruitTree
            | BotanicalGrowthForm::CitrusTree => mesh_lib.plant_leaf.clone(),
        },
        NodeType::Fruit => mesh_lib.fruit.clone(),
    }
}

fn plant_part_material(
    node_type: NodeType,
    trunk_mat: &Handle<BioMaterial>,
    leaf_mat: &Handle<BioMaterial>,
    fruit_mat: &Handle<BioMaterial>,
) -> Handle<BioMaterial> {
    match node_type {
        NodeType::Trunk | NodeType::Branch | NodeType::Bud => trunk_mat.clone(),
        NodeType::Leaf => leaf_mat.clone(),
        NodeType::Fruit => fruit_mat.clone(),
    }
}

fn spawn_plant_parts(
    commands: &mut Commands,
    root: Entity,
    mesh_lib: &MeshLibrary,
    plant: &TerrariumPlantSnapshot,
    node_positions: &[Vec3],
    trunk_mat: &Handle<BioMaterial>,
    leaf_mat: &Handle<BioMaterial>,
    fruit_mat: &Handle<BioMaterial>,
) {
    commands.entity(root).with_children(|parent| {
        for (node_index, node) in plant.morphology.iter().enumerate() {
            parent.spawn((
                Mesh3d(plant_part_mesh(mesh_lib, plant, node.node_type)),
                MeshMaterial3d(plant_part_material(
                    node.node_type,
                    trunk_mat,
                    leaf_mat,
                    fruit_mat,
                )),
                plant_part_transform(plant, node_index, node_positions),
                PlantPart {
                    node_index,
                    node_type: node.node_type,
                },
            ));
        }
    });
}

fn update_plant_parts<F>(
    child_entities: &[Entity],
    query: &mut Query<(&PlantPart, &mut Transform, &mut MeshMaterial3d<BioMaterial>), F>,
    materials: &mut Assets<BioMaterial>,
    plant: &TerrariumPlantSnapshot,
    node_positions: &[Vec3],
    trunk_base: &StandardMaterial,
    leaf_base: &StandardMaterial,
    fruit_base: &StandardMaterial,
) -> bool
where
    F: bevy::ecs::query::QueryFilter,
{
    if child_entities.len() != plant.morphology.len() {
        return false;
    }
    let mut seen = vec![false; plant.morphology.len()];
    for child in child_entities {
        let Ok((part, mut transform, mut material)) = query.get_mut(*child) else {
            return false;
        };
        let Some(node) = plant.morphology.get(part.node_index) else {
            return false;
        };
        if seen[part.node_index] || node.node_type != part.node_type {
            return false;
        }
        *transform = plant_part_transform(plant, part.node_index, node_positions);
        let material_base = match node.node_type {
            NodeType::Trunk | NodeType::Branch | NodeType::Bud => trunk_base.clone(),
            NodeType::Leaf => leaf_base.clone(),
            NodeType::Fruit => fruit_base.clone(),
        };
        material.0 = sync_bio_material(materials, Some(&material.0), material_base);
        seen[part.node_index] = true;
    }
    seen.into_iter().all(|present| present)
}

fn earthworm_root_transform(
    x: usize,
    y: usize,
    visual: TerrariumEarthwormVisualResponse,
    base_y: f32,
) -> Transform {
    snap_transform(
        Transform::from_translation(Vec3::new(
            (x as f32 + 0.5) * VOXEL_SIZE,
            base_y + 0.05 + visual.height_offset,
            (y as f32 + 0.5) * VOXEL_SIZE,
        ))
        .with_rotation(Quat::from_rotation_y(visual.yaw_rad)),
    )
}

fn earthworm_segment_transform(
    segment_index: usize,
    segment_count: usize,
    visual: TerrariumEarthwormVisualResponse,
) -> Transform {
    let t = if segment_count > 1 {
        segment_index as f32 / (segment_count - 1) as f32
    } else {
        0.0
    };
    let x_off = (t - 0.5) * visual.length_scale * 0.9;
    let z_off = visual.curl * (1.0 - ((t - 0.5) * 2.0).abs()) * 0.42;
    let y_off = (std::f32::consts::PI * t).sin() * 0.03 * visual.activity;
    let taper = 0.72 + (1.0 - ((t - 0.5) * 2.0).abs()) * 0.28;
    snap_transform(
        Transform::from_translation(Vec3::new(x_off, y_off, z_off)).with_scale(Vec3::new(
            0.10 * visual.thickness_scale * taper,
            0.07 * visual.thickness_scale * taper,
            0.14 * visual.thickness_scale * taper,
        )),
    )
}

fn earthworm_segment_material(
    segment_index: usize,
    segment_count: usize,
    body_mat: &Handle<BioMaterial>,
    clitellum_mat: &Handle<BioMaterial>,
) -> Handle<BioMaterial> {
    let t = if segment_count > 1 {
        segment_index as f32 / (segment_count - 1) as f32
    } else {
        0.0
    };
    if (0.38..=0.58).contains(&t) {
        clitellum_mat.clone()
    } else {
        body_mat.clone()
    }
}

fn spawn_earthworm_parts(
    commands: &mut Commands,
    root: Entity,
    mesh_lib: &MeshLibrary,
    segment_count: usize,
    visual: TerrariumEarthwormVisualResponse,
    body_mat: &Handle<BioMaterial>,
    clitellum_mat: &Handle<BioMaterial>,
) {
    commands.entity(root).with_children(|parent| {
        for segment_index in 0..segment_count {
            parent.spawn((
                Mesh3d(mesh_lib.soil_marker.clone()),
                MeshMaterial3d(earthworm_segment_material(
                    segment_index,
                    segment_count,
                    body_mat,
                    clitellum_mat,
                )),
                earthworm_segment_transform(segment_index, segment_count, visual),
                EarthwormPart { segment_index },
            ));
        }
    });
}

fn update_earthworm_parts<F>(
    child_entities: &[Entity],
    query: &mut Query<
        (
            &EarthwormPart,
            &mut Transform,
            &mut MeshMaterial3d<BioMaterial>,
        ),
        F,
    >,
    materials: &mut Assets<BioMaterial>,
    segment_count: usize,
    visual: TerrariumEarthwormVisualResponse,
    body_base: &StandardMaterial,
    clitellum_base: &StandardMaterial,
) -> bool
where
    F: bevy::ecs::query::QueryFilter,
{
    if child_entities.len() != segment_count {
        return false;
    }
    let mut seen = vec![false; segment_count];
    for child in child_entities {
        let Ok((part, mut transform, mut material)) = query.get_mut(*child) else {
            return false;
        };
        if part.segment_index >= segment_count || seen[part.segment_index] {
            return false;
        }
        *transform = earthworm_segment_transform(part.segment_index, segment_count, visual);
        let material_base = if (0.38..=0.58).contains(&if segment_count > 1 {
            part.segment_index as f32 / (segment_count - 1) as f32
        } else {
            0.0
        }) {
            clitellum_base.clone()
        } else {
            body_base.clone()
        };
        material.0 = sync_bio_material(materials, Some(&material.0), material_base);
        seen[part.segment_index] = true;
    }
    seen.into_iter().all(|present| present)
}

fn nematode_root_transform(
    x: usize,
    y: usize,
    visual: TerrariumNematodeVisualResponse,
    base_y: f32,
) -> Transform {
    snap_transform(
        Transform::from_translation(Vec3::new(
            (x as f32 + 0.5) * VOXEL_SIZE,
            base_y + 0.04 + visual.height_offset,
            (y as f32 + 0.5) * VOXEL_SIZE,
        ))
        .with_rotation(Quat::from_rotation_y(visual.yaw_rad)),
    )
}

fn nematode_segment_transform(
    segment_index: usize,
    segment_count: usize,
    visual: TerrariumNematodeVisualResponse,
) -> Transform {
    let t = if segment_count > 1 {
        segment_index as f32 / (segment_count - 1) as f32
    } else {
        0.0
    };
    let x_off = (t - 0.5) * visual.length_scale * 0.52;
    let z_off = visual.curl * (std::f32::consts::PI * t).sin() * 0.28;
    let taper = 1.0 - t * 0.45;
    snap_transform(
        Transform::from_translation(Vec3::new(x_off, 0.0, z_off)).with_scale(Vec3::new(
            0.05 * visual.thickness_scale * taper,
            0.04 * visual.thickness_scale * taper,
            0.08 * visual.thickness_scale * taper,
        )),
    )
}

fn nematode_stylet_transform(visual: TerrariumNematodeVisualResponse) -> Transform {
    snap_transform(
        Transform::from_translation(Vec3::new(visual.length_scale * 0.30, 0.0, 0.0))
            .with_scale(Vec3::new(0.02, 0.02, 0.10 * visual.stylet_length_scale)),
    )
}

fn spawn_nematode_parts(
    commands: &mut Commands,
    root: Entity,
    mesh_lib: &MeshLibrary,
    segment_count: usize,
    visual: TerrariumNematodeVisualResponse,
    body_mat: &Handle<BioMaterial>,
    head_mat: &Handle<BioMaterial>,
) {
    commands.entity(root).with_children(|parent| {
        for segment_index in 0..segment_count {
            let mat = if segment_index + 1 == segment_count {
                head_mat.clone()
            } else {
                body_mat.clone()
            };
            parent.spawn((
                Mesh3d(mesh_lib.soil_marker.clone()),
                MeshMaterial3d(mat),
                nematode_segment_transform(segment_index, segment_count, visual),
                NematodePart {
                    segment_index: Some(segment_index),
                },
            ));
        }
        if visual.stylet_length_scale > 0.0 {
            parent.spawn((
                Mesh3d(mesh_lib.soil_marker.clone()),
                MeshMaterial3d(head_mat.clone()),
                nematode_stylet_transform(visual),
                NematodePart {
                    segment_index: None,
                },
            ));
        }
    });
}

fn update_nematode_parts<F>(
    child_entities: &[Entity],
    query: &mut Query<
        (
            &NematodePart,
            &mut Transform,
            &mut MeshMaterial3d<BioMaterial>,
        ),
        F,
    >,
    materials: &mut Assets<BioMaterial>,
    segment_count: usize,
    visual: TerrariumNematodeVisualResponse,
    body_base: &StandardMaterial,
    head_base: &StandardMaterial,
) -> bool
where
    F: bevy::ecs::query::QueryFilter,
{
    let has_stylet = visual.stylet_length_scale > 0.0;
    let expected_count = segment_count + usize::from(has_stylet);
    if child_entities.len() != expected_count {
        return false;
    }
    let mut seen = vec![false; segment_count];
    let mut saw_stylet = false;
    for child in child_entities {
        let Ok((part, mut transform, mut material)) = query.get_mut(*child) else {
            return false;
        };
        match part.segment_index {
            Some(segment_index) => {
                if segment_index >= segment_count || seen[segment_index] {
                    return false;
                }
                *transform = nematode_segment_transform(segment_index, segment_count, visual);
                material.0 = sync_bio_material(
                    materials,
                    Some(&material.0),
                    if segment_index + 1 == segment_count {
                        head_base.clone()
                    } else {
                        body_base.clone()
                    },
                );
                seen[segment_index] = true;
            }
            None => {
                if !has_stylet || saw_stylet {
                    return false;
                }
                *transform = nematode_stylet_transform(visual);
                material.0 = sync_bio_material(materials, Some(&material.0), head_base.clone());
                saw_stylet = true;
            }
        }
    }
    seen.into_iter().all(|present| present) && (!has_stylet || saw_stylet)
}

fn plant_species_line(snapshot: &TerrariumWorldSnapshot) -> String {
    let mut names = Vec::new();
    for species in &snapshot.species_presence {
        if names
            .iter()
            .any(|seen: &String| seen == &species.common_name)
        {
            continue;
        }
        names.push(species.common_name.clone());
        if names.len() >= 4 {
            break;
        }
    }
    if names.is_empty() {
        "Species: none".to_string()
    } else {
        format!("Species: {}", names.join(", "))
    }
}

fn species_model_line(snapshot: &TerrariumWorldSnapshot) -> String {
    let mut models = Vec::new();
    if snapshot.species_presence.iter().any(|species| {
        matches!(
            species.domain,
            oneura_core::terrarium::TerrariumSpeciesDomain::Plant
        )
    }) {
        models.push("plants=species");
    }
    if snapshot.species_presence.iter().any(|species| {
        matches!(
            species.domain,
            oneura_core::terrarium::TerrariumSpeciesDomain::Insect
        ) && species.reference_neuron_count.is_some()
    }) {
        models.push("flies=neural");
    }
    if snapshot.species_presence.iter().any(|species| {
        matches!(
            species.authority,
            oneura_core::terrarium::TerrariumSpeciesAuthority::GuildReference
        )
    }) {
        models.push("soil=guild-ref");
    }
    if snapshot.species_presence.iter().any(|species| {
        matches!(
            species.domain,
            oneura_core::terrarium::TerrariumSpeciesDomain::Annelid
        )
    }) {
        models.push("worms=population");
    }
    if models.is_empty() {
        "Models:none".to_string()
    } else {
        format!("Models:{}", models.join(" | "))
    }
}

// ============================================================================
// Setup Systems
// ============================================================================

fn setup_meshes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<BioMaterial>>,
) {
    let library = MeshLibrary {
        fly_body: meshes.add(Capsule3d::new(0.5, 1.5).mesh().build()),
        fly_head: meshes.add(Sphere::new(0.6).mesh().ico(2).unwrap()),
        fly_wing: meshes.add(Ellipse::new(0.8, 1.5).mesh()),
        plant_trunk: meshes.add(Cylinder::new(0.3, 1.0)),
        plant_leaf: meshes.add(Cone::new(1.0, 2.0).mesh().build()),
        fruit: meshes.add(Sphere::new(0.8).mesh().ico(2).unwrap()),
        seed: meshes.add(Sphere::new(0.4).mesh().ico(1).unwrap()),
        soil_marker: meshes.add(Sphere::new(0.5).mesh().ico(1).unwrap()),
        water_quad: meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(0.5))),
    };
    let terrain_materials = TerrainMaterialLibrary {
        bedrock: materials.add(terrain_material_base(TerrariumVoxelMaterialClass::Bedrock)),
        subsoil: materials.add(terrain_material_base(TerrariumVoxelMaterialClass::Subsoil)),
        surface: materials.add(terrain_material_base(TerrariumVoxelMaterialClass::Surface)),
        water: materials.add(terrain_material_base(TerrariumVoxelMaterialClass::Water)),
    };
    commands.insert_resource(library);
    commands.insert_resource(terrain_materials);
}

fn setup_scene(
    mut commands: Commands,
    config: Res<ViewerConfig>,
    mut initial_state: ResMut<InitialSimState>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut cam = OrbitCamera::default();
    match config.preset {
        TerrariumDemoPreset::Demo => {}
        TerrariumDemoPreset::MicroTerrarium => {
            cam.focus = Vec3::new(3.0, 0.45, 3.0);
            cam.distance = 10.5;
            cam.yaw = -0.78;
            cam.pitch = 0.78;
        }
        TerrariumDemoPreset::MicroAquarium => {
            cam.focus = Vec3::new(3.5, 0.20, 2.5);
            cam.distance = 8.4;
            cam.yaw = -0.72;
            cam.pitch = 0.56;
        }
    }
    
    let cam_pos = orbit_position(&cam);
    commands.spawn((
        Camera3d::default(),
        Camera {
            hdr: true,
            clear_color: ClearColorConfig::Custom(Color::srgb(0.2, 0.3, 0.5)),
            ..default()
        },
        Transform::from_translation(cam_pos).looking_at(cam.focus, Vec3::Y),
        DistanceFog {
            color: Color::srgb(0.2, 0.3, 0.5),
            directional_light_color: Color::srgb(1.0, 0.95, 0.85),
            directional_light_exponent: 30.0,
            falloff: FogFalloff::Linear {
                start: 10.0,
                end: 50.0,
            },
        },
        cam,
    ));

    commands.spawn((
        DirectionalLight {
            color: Color::srgb(1.0, 0.95, 0.85),
            illuminance: 10_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.8, 0.3, 0.0)),
    ));

    commands.insert_resource(AmbientLight {
        color: Color::srgb(0.6, 0.7, 0.9),
        brightness: 200.0,
    });

    // Molecular Setup at 1000, 1000, 1000
    let atom_mesh = meshes.add(Sphere::new(0.5).mesh().build());
    
    for i in 0..20 {
        let angle = (i as f32) * std::f32::consts::PI * 2.0 / 20.0;
        let x = angle.cos() * 2.5;
        let y = (i as f32 * 0.2) - 1.0;
        let z = angle.sin() * 2.5;
        
        let color = if i % 2 == 0 { Color::srgb(0.8, 0.2, 0.2) } else { Color::srgb(0.2, 0.2, 0.8) };
        let mat = materials.add(StandardMaterial {
            base_color: color,
            ..default()
        });

        commands.spawn((
            Mesh3d(atom_mesh.clone()),
            MeshMaterial3d(mat),
            Transform::from_xyz(1000.0 + x, 1000.0 + y, 1000.0 + z),
        ));
    }


    let fps = config.fps;
    let checkpoint_out = initial_state.checkpoint_out.clone();
    let archive_out = initial_state.archive_out.clone();
    let mut world = initial_state
        .world
        .take()
        .expect("initial terrarium world should be provided before scene setup");
    let (tx, rx) = mpsc::sync_channel(1);

    std::thread::spawn(move || {
        let step_budget = if fps > 0.0 {
            Some(Duration::from_secs_f32(1.0 / fps.max(1.0)))
        } else {
            None
        };
        let mut next_tick = Instant::now();
        loop {
            if world.step_frame().is_err() {
                break;
            }

            let snapshot = world.snapshot();
            let atmosphere = world.atmosphere_frame();
            let (surface_relief, terrain_voxels) = terrain_voxel_batches(&world, &atmosphere);
            let w = world.width();
            let h = world.height();
            let energy_charges: Vec<f32> = world
                .fly_metabolisms
                .iter()
                .map(|m| m.energy_charge())
                .collect();

            let fly_states: Vec<(f32, f32, f32, f32, f32, bool, f32)> = world
                .flies
                .iter()
                .enumerate()
                .map(|(i, f)| {
                    let bs = f.body_state();
                    (
                        bs.x,
                        bs.y,
                        bs.z,
                        bs.heading,
                        energy_charges
                            .get(i)
                            .copied()
                            .unwrap_or((bs.energy / 100.0).clamp(0.0, 1.0)),
                        bs.is_flying,
                        bs.wing_beat_freq,
                    )
                })
                .collect();

            let fly_visuals: Vec<TerrariumFlyVisualResponse> = world
                .flies
                .iter()
                .enumerate()
                .map(|(i, f)| {
                    let bs = f.body_state();
                    let local_air = sample_visual_air(&atmosphere, w, h, bs.x, bs.y);
                    fly_visual_response(
                        local_air,
                        bs,
                        energy_charges
                            .get(i)
                            .copied()
                            .unwrap_or((bs.energy / 100.0).clamp(0.0, 1.0)),
                        world.time_s,
                        i as f32,
                    )
                })
                .collect();

            let plant_visuals: Vec<TerrariumPlantVisualResponse> = world
                .plants
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let local_air =
                        sample_visual_air(&atmosphere, w, h, p.x as f32 + 0.5, p.y as f32 + 0.5);
                    plant_visual_response(
                        p.genome.taxonomy_id,
                        local_air,
                        p.cellular.vitality(),
                        (p.cellular.total_cells() * 0.01).clamp(0.0, 1.0),
                        world.time_s,
                        i as f32,
                    )
                })
                .collect();
            let plant_snapshots = snapshot.full_plants.clone();

            let all_fruit_visuals = world.fruit_visuals();
            let mut fruit_positions = Vec::new();
            let mut fruit_visuals = Vec::new();
            for (i, fruit) in world.fruits.iter().enumerate() {
                if !fruit.source.alive {
                    continue;
                }
                fruit_positions.push((fruit.source.x as f32, fruit.source.y as f32));
                fruit_visuals.push(all_fruit_visuals.get(i).copied().unwrap_or_default());
            }

            let seed_states: Vec<(f32, f32, u32)> = world
                .seeds
                .iter()
                .map(|seed| (seed.x, seed.y, seed.genome.taxonomy_id))
                .collect();
            let seed_visuals = world.seed_visuals();

            let water_cells: Vec<(f32, f32, f32)> = world
                .waters
                .iter()
                .filter(|w| w.alive)
                .map(|w| (w.x as f32, w.y as f32, w.volume))
                .collect();
            let water_cycle = TerrariumWaterCycleInputs {
                lunar_phase: world.lunar_phase(),
                moonlight: world.moonlight(),
                tidal_moisture_factor: world.tidal_moisture_factor(),
            };
            let water_visuals: Vec<TerrariumWaterVisualResponse> = world
                .waters
                .iter()
                .filter(|w| w.alive)
                .enumerate()
                .map(|(i, water)| {
                    let local_air = sample_visual_air(
                        &atmosphere,
                        w,
                        h,
                        water.x as f32 + 0.5,
                        water.y as f32 + 0.5,
                    );
                    let chemistry = sample_visual_chemistry(&world, water.x, water.y);
                    water_visual_response_with_chemistry(
                        local_air,
                        water.volume,
                        world.time_s,
                        i as f32,
                        water_cycle,
                        chemistry,
                    )
                })
                .collect();
            let earthworms = world.earthworm_visual_markers();
            let nematodes = world.nematode_visual_markers();
            let soil_surface = world.soil_surface_markers();

            let frame = SimFrame {
                snapshot,
                surface_relief,
                terrain_voxels,
                atmosphere,
                width: w,
                height: h,
                fly_states,
                fly_visuals,
                plant_snapshots,
                plant_visuals,
                fruit_positions,
                fruit_visuals,
                seed_states,
                seed_visuals,
                water_cells,
                water_visuals,
                earthworms,
                nematodes,
                soil_surface,
            };

            if tx.send(frame).is_err() {
                break;
            }

            if let Some(step_budget) = step_budget {
                next_tick += step_budget;
                let now = Instant::now();
                if next_tick > now {
                    std::thread::sleep(next_tick - now);
                } else {
                    next_tick = now;
                }
            }
        }

        if let Some(path) = checkpoint_out.as_deref() {
            if let Err(error) = world.save_checkpoint(path) {
                eprintln!("failed to save terrarium checkpoint: {error}");
            }
        }
        if let Some(path) = archive_out.as_deref() {
            if let Err(error) = world.save_archive(path) {
                eprintln!("failed to save terrarium archive: {error}");
            }
        }
    });

    commands.insert_resource(SimChannel { rx: Mutex::new(rx) });
    commands.insert_resource(EntityMap::default());
    commands.insert_resource(SimTime(0.0));
}

fn setup_hud(mut commands: Commands, config: Res<ViewerConfig>) {
    commands.spawn((
        Text::new(format!("oNeura 3D | {}", config.preset.label())),
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

    commands.spawn((
        Text::new("WASD: pan | QE: up/down | RMB: orbit | Scroll: zoom | F: follow | Tab: cycle"),
        TextFont {
            font_size: 12.0,
            ..default()
        },
        TextColor(Color::srgba(1.0, 1.0, 1.0, 0.6)),
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        },
        HudControls,
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
    mesh_lib: Res<MeshLibrary>,
    terrain_materials: Res<TerrainMaterialLibrary>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<BioMaterial>>,
    config: Res<ViewerConfig>,
    mut terrain_query: Query<(
        Entity,
        &TerrainBatchEntity,
        &mut Mesh3d,
        &mut MeshMaterial3d<BioMaterial>,
    )>,
    mut simple_entity_queries: ParamSet<(
        Query<
            (&mut Transform, &mut MeshMaterial3d<BioMaterial>),
            (
                With<FruitEntity>,
                Without<OrbitCamera>,
                Without<TerrainBatchEntity>,
            ),
        >,
        Query<
            (&mut Transform, &mut MeshMaterial3d<BioMaterial>),
            (
                With<SeedEntity>,
                Without<OrbitCamera>,
                Without<TerrainBatchEntity>,
            ),
        >,
        Query<
            (&mut Transform, &mut MeshMaterial3d<BioMaterial>),
            (
                With<WaterEntity>,
                Without<OrbitCamera>,
                Without<TerrainBatchEntity>,
            ),
        >,
        Query<
            (&mut Transform, &mut MeshMaterial3d<BioMaterial>),
            (
                With<SoilSurfaceEntity>,
                Without<OrbitCamera>,
                Without<TerrainBatchEntity>,
            ),
        >,
    )>,
    mut composite_queries: ParamSet<(
        Query<
            (&mut Transform, &Children, &mut FlyRigLayout),
            (
                With<FlyEntity>,
                Without<FruitEntity>,
                Without<SeedEntity>,
                Without<WaterEntity>,
                Without<SoilSurfaceEntity>,
                Without<OrbitCamera>,
                Without<TerrainBatchEntity>,
            ),
        >,
        Query<
            (&FlyPart, &mut Transform, &mut MeshMaterial3d<BioMaterial>),
            (
                Without<FruitEntity>,
                Without<SeedEntity>,
                Without<WaterEntity>,
                Without<SoilSurfaceEntity>,
                Without<TerrainBatchEntity>,
            ),
        >,
        Query<
            (&mut Transform, &Children, &mut PlantRigFingerprint),
            (
                With<PlantEntity>,
                Without<FruitEntity>,
                Without<SeedEntity>,
                Without<WaterEntity>,
                Without<SoilSurfaceEntity>,
                Without<OrbitCamera>,
                Without<TerrainBatchEntity>,
            ),
        >,
        Query<
            (&PlantPart, &mut Transform, &mut MeshMaterial3d<BioMaterial>),
            (
                Without<FruitEntity>,
                Without<SeedEntity>,
                Without<WaterEntity>,
                Without<SoilSurfaceEntity>,
                Without<TerrainBatchEntity>,
            ),
        >,
        Query<
            (&mut Transform, &Children, &mut EarthwormRigLayout),
            (
                With<EarthwormEntity>,
                Without<FruitEntity>,
                Without<SeedEntity>,
                Without<WaterEntity>,
                Without<SoilSurfaceEntity>,
                Without<OrbitCamera>,
                Without<TerrainBatchEntity>,
            ),
        >,
        Query<
            (
                &EarthwormPart,
                &mut Transform,
                &mut MeshMaterial3d<BioMaterial>,
            ),
            (
                Without<FruitEntity>,
                Without<SeedEntity>,
                Without<WaterEntity>,
                Without<SoilSurfaceEntity>,
                Without<TerrainBatchEntity>,
            ),
        >,
        Query<
            (&mut Transform, &Children, &mut NematodeRigLayout),
            (
                With<NematodeEntity>,
                Without<FruitEntity>,
                Without<SeedEntity>,
                Without<WaterEntity>,
                Without<SoilSurfaceEntity>,
                Without<OrbitCamera>,
                Without<TerrainBatchEntity>,
            ),
        >,
        Query<
            (
                &NematodePart,
                &mut Transform,
                &mut MeshMaterial3d<BioMaterial>,
            ),
            (
                Without<FruitEntity>,
                Without<SeedEntity>,
                Without<WaterEntity>,
                Without<SoilSurfaceEntity>,
                Without<TerrainBatchEntity>,
            ),
        >,
    )>,
    mut hud_query: Query<&mut Text, With<HudText>>,
    mut light_query: Query<&mut DirectionalLight>,
    mut camera_query: Query<&mut OrbitCamera, With<OrbitCamera>>,
    mut ambient_light: ResMut<AmbientLight>,
    mut sim_time: ResMut<SimTime>,
) {
    let mut latest_frame = None;
    let rx = channel.rx.lock().unwrap();
    while let Ok(frame) = rx.try_recv() {
        latest_frame = Some(frame);
    }
    drop(rx);

    let Some(frame) = latest_frame else { return };

    sim_time.0 = frame.snapshot.time_s;

    if let Some(mut orbit) = camera_query.iter_mut().next() {
        if orbit.auto_follow.is_none() && orbit.needs_framing {
            let (focus, distance, yaw, pitch) = preset_camera_pose(
                config.preset,
                frame.width,
                frame.height,
                &frame.surface_relief,
            );
            orbit.focus = focus;
            orbit.distance = distance;
            orbit.yaw = yaw;
            orbit.pitch = pitch;
            orbit.needs_framing = false;
        }
    }

    let mean_humidity = mean_field(&frame.atmosphere.humidity, frame.snapshot.humidity);
    let mean_pressure = mean_field(
        &frame.atmosphere.pressure_kpa,
        frame.snapshot.mean_air_pressure_kpa,
    );
    let mean_wind = mean_wind_speed(&frame.atmosphere);

    ambient_light.brightness = 170.0 + mean_humidity.clamp(0.0, 1.0) * 90.0;
    for mut light in light_query.iter_mut() {
        light.illuminance = 7_500.0
            + frame.snapshot.light.clamp(0.0, 1.0) * 4_000.0
            + ((frame.snapshot.temperature - 10.0) / 20.0).clamp(0.0, 1.0) * 1_000.0
            + mean_wind * 600.0;
    }

    // ---- Terrain ----
    if frame.width > 1 && frame.height > 1 {
        let active_classes: HashSet<TerrariumVoxelMaterialClass> = frame
            .terrain_voxels
            .iter()
            .filter(|batch| !batch.instances.is_empty())
            .map(|batch| batch.class)
            .collect();
        let stale_batches: Vec<TerrariumVoxelMaterialClass> = entity_map
            .terrain_batches
            .keys()
            .copied()
            .filter(|class| !active_classes.contains(class))
            .collect();
        for class in stale_batches {
            if let Some(entity) = entity_map.terrain_batches.remove(&class) {
                commands.entity(entity).try_despawn_recursive();
            }
            entity_map.terrain_batch_fingerprints.remove(&class);
        }

        for batch in frame
            .terrain_voxels
            .iter()
            .filter(|batch| !batch.instances.is_empty())
        {
            let fingerprint = terrain_batch_fingerprint(batch);
            let batch_material = match batch.class {
                TerrariumVoxelMaterialClass::Bedrock => terrain_materials.bedrock.clone(),
                TerrariumVoxelMaterialClass::Subsoil => terrain_materials.subsoil.clone(),
                TerrariumVoxelMaterialClass::Surface => terrain_materials.surface.clone(),
                TerrariumVoxelMaterialClass::Water => terrain_materials.water.clone(),
            };

            if let Some(entity) = entity_map.terrain_batches.get(&batch.class).copied() {
                if let Ok((_entity, _marker, mut mesh3d, mut terrain_material)) =
                    terrain_query.get_mut(entity)
                {
                    let previous = entity_map
                        .terrain_batch_fingerprints
                        .get(&batch.class)
                        .copied()
                        .unwrap_or_default();
                    if previous != fingerprint {
                        let batch_mesh = if batch.class == TerrariumVoxelMaterialClass::Surface {
                            create_smooth_surface_mesh(frame.width, frame.height, &frame.surface_relief, batch)
                        } else {
                            create_voxel_batch_mesh(batch)
                        };
                        mesh3d.0 = sync_mesh_asset(&mut meshes, Some(&mesh3d.0), batch_mesh);
                        entity_map
                            .terrain_batch_fingerprints
                            .insert(batch.class, fingerprint);
                    }
                    terrain_material.0 = batch_material;
                    continue;
                }
                commands.entity(entity).try_despawn_recursive();
                entity_map.terrain_batches.remove(&batch.class);
                entity_map.terrain_batch_fingerprints.remove(&batch.class);
            }

            let batch_mesh = if batch.class == TerrariumVoxelMaterialClass::Surface {
                create_smooth_surface_mesh(frame.width, frame.height, &frame.surface_relief, batch)
            } else {
                create_voxel_batch_mesh(batch)
            };
            let terrain_handle = sync_mesh_asset(&mut meshes, None, batch_mesh);
            let entity = commands
                .spawn((
                    Mesh3d(terrain_handle),
                    MeshMaterial3d(batch_material),
                    Transform::default(),
                    TerrainBatchEntity(batch.class),
                ))
                .id();
            entity_map.terrain_batches.insert(batch.class, entity);
            entity_map
                .terrain_batch_fingerprints
                .insert(batch.class, fingerprint);
        }
    }

    // ---- Flies (Shared visual response) ----
    despawn_stale(&mut commands, &mut entity_map.flies, frame.fly_states.len());

    for (i, &(x, y, z, heading, _energy_frac, is_flying, _wing_beat_freq)) in
        frame.fly_states.iter().enumerate()
    {
        let visual = frame.fly_visuals.get(i).copied().unwrap_or_default();
        let root_transform = fly_root_transform(x, y, z, heading, visual);
        let body_base = StandardMaterial {
            base_color: pixel_color(visual.body_rgb),
            metallic: 0.18,
            perceptual_roughness: 0.65,
            ..default()
        };
        let wing_base = StandardMaterial {
            base_color: pixel_alpha_color(visual.wing_rgb, if is_flying { 0.65 } else { 0.45 }),
            alpha_mode: AlphaMode::Blend,
            perceptual_roughness: 0.4,
            ..default()
        };
        let has_proboscis = visual.proboscis_extension > 0.0;
        let mut fly_root = entity_map.flies.get(&i).copied();
        let mut child_entities = Vec::new();
        let mut needs_rebuild = false;

        if let Some(entity) = fly_root {
            let layout_matches = {
                let mut fly_roots = composite_queries.p0();
                if let Ok((mut transform, children, mut layout)) = fly_roots.get_mut(entity) {
                    *transform = root_transform;
                    child_entities = children.iter().copied().collect();
                    let matches = layout.has_proboscis == has_proboscis;
                    if !matches {
                        layout.has_proboscis = has_proboscis;
                    }
                    matches
                } else {
                    false
                }
            };
            if child_entities.is_empty() && !layout_matches {
                commands.entity(entity).try_despawn_recursive();
                entity_map.flies.remove(&i);
                fly_root = None;
            } else if layout_matches {
                let updated = {
                    let mut fly_parts = composite_queries.p1();
                    update_fly_parts(
                        &child_entities,
                        &mut fly_parts,
                        &mut materials,
                        &body_base,
                        &wing_base,
                        visual,
                    )
                };
                needs_rebuild = !updated;
            } else {
                needs_rebuild = true;
            }
        }

        let fly_root = if let Some(entity) = fly_root {
            entity
        } else {
            let entity = commands
                .spawn((
                    root_transform,
                    Visibility::default(),
                    FlyEntity(i),
                    FlyRigLayout { has_proboscis },
                ))
                .id();
            entity_map.flies.insert(i, entity);
            needs_rebuild = true;
            entity
        };

        if needs_rebuild {
            commands.entity(fly_root).despawn_descendants();
            let body_mat = add_bio_material(
                &mut materials,
                body_base.clone(),
                mean_humidity,
                if is_flying { 0.85 } else { 0.55 },
                frame.snapshot.light.clamp(0.0, 1.0),
                frame.snapshot.time_s,
                FLAG_FLY,
                0.76,
                if is_flying { 0.12 } else { 0.22 },
                config.use_bio_shader,
            );
            let wing_mat = add_bio_material(
                &mut materials,
                wing_base.clone(),
                mean_humidity,
                if is_flying { 0.95 } else { 0.35 },
                frame.snapshot.light.clamp(0.0, 1.0),
                frame.snapshot.time_s,
                FLAG_FLY,
                0.72,
                0.10,
                config.use_bio_shader,
            );
            spawn_fly_parts(
                &mut commands,
                fly_root,
                &mesh_lib,
                &body_mat,
                &wing_mat,
                visual,
            );
        }
    }

    // ---- Plants (Morphology-driven) ----
    despawn_stale(
        &mut commands,
        &mut entity_map.plants,
        frame.plant_snapshots.len(),
    );

    for (i, plant) in frame.plant_snapshots.iter().enumerate() {
        let visual = frame.plant_visuals.get(i).copied().unwrap_or_default();
        let surface_y = terrain_surface_height(
            &frame.surface_relief,
            frame.width,
            frame.height,
            plant.x as f32 + 0.5,
            plant.y as f32 + 0.5,
        );
        let base_y = match plant.growth_form {
            BotanicalGrowthForm::FloatingAquatic => {
                voxel_water_surface_height(&frame.terrain_voxels, surface_y, plant.x, plant.y)
            }
            BotanicalGrowthForm::SubmergedAquatic => surface_y,
            _ => surface_y,
        };
        let root_transform = plant_root_transform(plant, visual, base_y);
        let node_positions = plant_node_positions(plant, visual);
        let fingerprint = PlantRigFingerprint(plant_morphology_fingerprint(plant));
        let trunk_base = StandardMaterial {
            base_color: pixel_color(visual.stem_rgb),
            perceptual_roughness: 0.92,
            ..default()
        };
        let leaf_base = StandardMaterial {
            base_color: pixel_color(visual.leaf_rgb),
            perceptual_roughness: 0.84,
            ..default()
        };
        let fruit_base = StandardMaterial {
            base_color: pixel_color(visual.fruit_rgb),
            emissive: bevy::color::LinearRgba::new(0.12, 0.03, 0.01, 1.0),
            ..default()
        };
        let mut plant_root = entity_map.plants.get(&i).copied();
        let mut child_entities = Vec::new();
        let mut needs_rebuild = false;

        if let Some(entity) = plant_root {
            let fingerprint_matches = {
                let mut plant_roots = composite_queries.p2();
                if let Ok((mut transform, children, mut rig_fingerprint)) =
                    plant_roots.get_mut(entity)
                {
                    *transform = root_transform;
                    child_entities = children.iter().copied().collect();
                    let matches = *rig_fingerprint == fingerprint;
                    if !matches {
                        *rig_fingerprint = fingerprint;
                    }
                    matches
                } else {
                    false
                }
            };
            if child_entities.is_empty() && !fingerprint_matches {
                commands.entity(entity).try_despawn_recursive();
                entity_map.plants.remove(&i);
                plant_root = None;
            } else if fingerprint_matches {
                let updated = {
                    let mut plant_parts = composite_queries.p3();
                    update_plant_parts(
                        &child_entities,
                        &mut plant_parts,
                        &mut materials,
                        plant,
                        &node_positions,
                        &trunk_base,
                        &leaf_base,
                        &fruit_base,
                    )
                };
                needs_rebuild = !updated;
            } else {
                needs_rebuild = true;
            }
        }

        let plant_root = if let Some(entity) = plant_root {
            entity
        } else {
            let entity = commands
                .spawn((
                    root_transform,
                    Visibility::default(),
                    PlantEntity,
                    fingerprint,
                ))
                .id();
            entity_map.plants.insert(i, entity);
            needs_rebuild = true;
            entity
        };

        if needs_rebuild {
            commands.entity(plant_root).despawn_descendants();
            let trunk_mat = add_bio_material(
                &mut materials,
                trunk_base.clone(),
                mean_humidity,
                0.42,
                frame.snapshot.light.clamp(0.0, 1.0),
                frame.snapshot.time_s,
                FLAG_PLANT,
                0.48,
                0.16,
                config.use_bio_shader,
            );
            let leaf_mat = add_bio_material(
                &mut materials,
                leaf_base.clone(),
                mean_humidity,
                0.78,
                frame.snapshot.light.clamp(0.0, 1.0),
                frame.snapshot.time_s,
                FLAG_PLANT,
                visual.canopy_scale.clamp(0.0, 1.0),
                0.12,
                config.use_bio_shader,
            );
            let fruit_mat = add_bio_material(
                &mut materials,
                fruit_base.clone(),
                mean_humidity,
                0.64,
                frame.snapshot.light.clamp(0.0, 1.0),
                frame.snapshot.time_s,
                FLAG_FRUIT,
                0.68,
                0.08,
                config.use_bio_shader,
            );
            spawn_plant_parts(
                &mut commands,
                plant_root,
                &mesh_lib,
                plant,
                &node_positions,
                &trunk_mat,
                &leaf_mat,
                &fruit_mat,
            );
        }
    }

    // ---- Fruits ----
    despawn_stale(
        &mut commands,
        &mut entity_map.fruits,
        frame.fruit_positions.len(),
    );

    for (i, &(x, y)) in frame.fruit_positions.iter().enumerate() {
        let visual = frame.fruit_visuals.get(i).copied().unwrap_or_default();
        let fruit_meta = frame.snapshot.full_fruits.get(i);
        let shape = fruit_meta.map(|fruit| fruit.shape).unwrap_or_default();
        let radius = fruit_meta.map(|fruit| fruit.radius).unwrap_or(1.0);
        let surface_y = terrain_surface_height(
            &frame.surface_relief,
            frame.width,
            frame.height,
            x + 0.5,
            y + 0.5,
        );
        let transform = snap_transform(
            Transform::from_translation(Vec3::new(
                (x + 0.5) * VOXEL_SIZE,
                surface_y + 0.08 + visual.vertical_offset * 0.8,
                (y + 0.5) * VOXEL_SIZE,
            ))
            .with_rotation(Quat::from_euler(
                EulerRot::XYZ,
                visual.tilt_x,
                0.0,
                visual.tilt_z,
            ))
            .with_scale(Vec3::new(
                FRUIT_SCALE * visual.sprite_scale * shape.width_scale * (0.86 + radius * 0.10),
                FRUIT_SCALE * visual.sprite_scale * shape.height_scale * (0.82 + radius * 0.12),
                FRUIT_SCALE * visual.sprite_scale * shape.depth_scale * (0.86 + radius * 0.10),
            )),
        );
        upsert_simple_entity(
            &mut commands,
            &mut entity_map.fruits,
            &mut materials,
            &mut simple_entity_queries.p0(),
            i,
            mesh_lib.fruit.clone(),
            StandardMaterial {
                base_color: pixel_color(visual.skin_rgb),
                emissive: bevy::color::LinearRgba::new(0.3, 0.1, 0.02, 1.0),
                ..default()
            },
            transform,
            FruitEntity,
        );
    }

    // ---- Seeds ----
    despawn_stale(
        &mut commands,
        &mut entity_map.seeds,
        frame.seed_states.len(),
    );

    for (i, &(x, y, _taxonomy_id)) in frame.seed_states.iter().enumerate() {
        let visual = frame.seed_visuals.get(i).copied().unwrap_or_default();
        let seed_meta = frame.snapshot.full_seeds.get(i);
        let shape = seed_meta.map(|seed| seed.shape).unwrap_or_default();
        let surface_y =
            terrain_surface_height(&frame.surface_relief, frame.width, frame.height, x, y);
        let transform = snap_transform(
            Transform::from_translation(Vec3::new(
                x * VOXEL_SIZE,
                surface_y + 0.06 + visual.vertical_offset * 0.5,
                y * VOXEL_SIZE,
            ))
            .with_rotation(Quat::from_euler(
                EulerRot::XYZ,
                visual.tilt_x,
                0.0,
                visual.tilt_z,
            ))
            .with_scale(Vec3::new(
                0.05 * visual.sprite_scale * shape.width_scale.max(0.2),
                0.03 * visual.sprite_scale * shape.height_scale.max(0.2),
                0.07 * visual.sprite_scale * shape.depth_scale.max(0.2),
            )),
        );
        upsert_simple_entity(
            &mut commands,
            &mut entity_map.seeds,
            &mut materials,
            &mut simple_entity_queries.p1(),
            i,
            mesh_lib.seed.clone(),
            StandardMaterial {
                base_color: pixel_color(visual.shell_rgb),
                emissive: bevy::color::LinearRgba::new(
                    visual.accent_rgb[0] * 0.05,
                    visual.accent_rgb[1] * 0.04,
                    visual.accent_rgb[2] * 0.03,
                    1.0,
                ),
                ..default()
            },
            transform,
            SeedEntity,
        );
    }

    // ---- Water ----
    despawn_stale(
        &mut commands,
        &mut entity_map.waters,
        frame.water_cells.len(),
    );

    let has_voxel_water = frame.terrain_voxels.iter().any(|batch| {
        batch.class == TerrariumVoxelMaterialClass::Water && !batch.instances.is_empty()
    });
    if has_voxel_water {
        for entity in entity_map
            .waters
            .drain()
            .map(|(_, entity)| entity)
            .collect::<Vec<_>>()
        {
            commands.entity(entity).try_despawn_recursive();
        }
    } else {
        for (i, &(x, y, volume)) in frame.water_cells.iter().enumerate() {
            let water_height = volume.clamp(0.01, 1.0) * 0.3;
            let visual = frame.water_visuals.get(i).copied().unwrap_or_default();
            let surface_y = terrain_surface_height(
                &frame.surface_relief,
                frame.width,
                frame.height,
                x + 0.5,
                y + 0.5,
            );
            let pos = Vec3::new(
                (x + 0.5) * VOXEL_SIZE,
                surface_y + water_height * 0.5 + 0.02 + visual.vertical_offset * 0.4,
                (y + 0.5) * VOXEL_SIZE,
            );
            let alpha = (0.3 + volume * 0.4).min(0.7);
            let water_transform = snap_transform(
                Transform::from_translation(pos)
                    .with_rotation(Quat::from_euler(
                        EulerRot::XYZ,
                        visual.tilt_x * 0.6,
                        0.0,
                        visual.tilt_z * 0.6,
                    ))
                    .with_scale(Vec3::new(
                        visual.radius_x_cells.max(0.8) * VOXEL_SIZE,
                        (water_height.max(0.06) * visual.thickness_scale).max(0.06),
                        visual.radius_y_cells.max(0.8) * VOXEL_SIZE,
                    )),
            );
            upsert_simple_entity(
                &mut commands,
                &mut entity_map.waters,
                &mut materials,
                &mut simple_entity_queries.p2(),
                i,
                mesh_lib.water_quad.clone(),
                StandardMaterial {
                    base_color: pixel_alpha_color(visual.rgb, alpha),
                    alpha_mode: AlphaMode::Blend,
                    perceptual_roughness: 0.1,
                    metallic: 0.3,
                    ..default()
                },
                water_transform,
                WaterEntity,
            );
        }
    }

    // ---- Earthworms (field-derived fauna markers) ----
    despawn_stale(
        &mut commands,
        &mut entity_map.earthworms,
        frame.earthworms.len(),
    );

    for (i, &(x, y, visual)) in frame.earthworms.iter().enumerate() {
        let segment_count = visual.segment_count.max(4) as usize;
        let surface_y = terrain_surface_height(
            &frame.surface_relief,
            frame.width,
            frame.height,
            x as f32 + 0.5,
            y as f32 + 0.5,
        );
        let root_transform = earthworm_root_transform(x, y, visual, surface_y);
        let body_base = StandardMaterial {
            base_color: pixel_color(visual.body_rgb),
            ..default()
        };
        let clitellum_base = StandardMaterial {
            base_color: pixel_color(visual.clitellum_rgb),
            ..default()
        };
        let mut earthworm_root = entity_map.earthworms.get(&i).copied();
        let mut child_entities = Vec::new();
        let mut needs_rebuild = false;

        if let Some(entity) = earthworm_root {
            let layout_matches = {
                let mut earthworm_roots = composite_queries.p4();
                if let Ok((mut transform, children, mut layout)) = earthworm_roots.get_mut(entity) {
                    *transform = root_transform;
                    child_entities = children.iter().copied().collect();
                    let matches = layout.segment_count == segment_count;
                    if !matches {
                        layout.segment_count = segment_count;
                    }
                    matches
                } else {
                    false
                }
            };
            if child_entities.is_empty() && !layout_matches {
                commands.entity(entity).try_despawn_recursive();
                entity_map.earthworms.remove(&i);
                earthworm_root = None;
            } else if layout_matches {
                let updated = {
                    let mut earthworm_parts = composite_queries.p5();
                    update_earthworm_parts(
                        &child_entities,
                        &mut earthworm_parts,
                        &mut materials,
                        segment_count,
                        visual,
                        &body_base,
                        &clitellum_base,
                    )
                };
                needs_rebuild = !updated;
            } else {
                needs_rebuild = true;
            }
        }

        let earthworm_root = if let Some(entity) = earthworm_root {
            entity
        } else {
            let entity = commands
                .spawn((
                    root_transform,
                    Visibility::default(),
                    EarthwormEntity,
                    EarthwormRigLayout { segment_count },
                ))
                .id();
            entity_map.earthworms.insert(i, entity);
            needs_rebuild = true;
            entity
        };

        if needs_rebuild {
            commands.entity(earthworm_root).despawn_descendants();
            let body_mat = add_bio_material(
                &mut materials,
                body_base.clone(),
                mean_humidity,
                visual.activity,
                frame.snapshot.light.clamp(0.0, 1.0),
                frame.snapshot.time_s,
                FLAG_SUBSTRATE,
                visual.activity,
                0.14,
                config.use_bio_shader,
            );
            let clitellum_mat = add_bio_material(
                &mut materials,
                clitellum_base.clone(),
                mean_humidity,
                visual.activity,
                frame.snapshot.light.clamp(0.0, 1.0),
                frame.snapshot.time_s,
                FLAG_SUBSTRATE,
                visual.activity,
                0.18,
                config.use_bio_shader,
            );
            spawn_earthworm_parts(
                &mut commands,
                earthworm_root,
                &mesh_lib,
                segment_count,
                visual,
                &body_mat,
                &clitellum_mat,
            );
        }
    }

    // ---- Nematodes (guild hotspot projections) ----
    despawn_stale(
        &mut commands,
        &mut entity_map.nematodes,
        frame.nematodes.len(),
    );

    for (i, &(x, y, visual)) in frame.nematodes.iter().enumerate() {
        let segment_count = 6usize;
        let has_stylet = visual.stylet_length_scale > 0.0;
        let surface_y = terrain_surface_height(
            &frame.surface_relief,
            frame.width,
            frame.height,
            x as f32 + 0.5,
            y as f32 + 0.5,
        );
        let root_transform = nematode_root_transform(x, y, visual, surface_y);
        let body_base = StandardMaterial {
            base_color: pixel_color(visual.body_rgb),
            ..default()
        };
        let head_base = StandardMaterial {
            base_color: pixel_color(visual.head_rgb),
            ..default()
        };
        let mut nematode_root = entity_map.nematodes.get(&i).copied();
        let mut child_entities = Vec::new();
        let mut needs_rebuild = false;

        if let Some(entity) = nematode_root {
            let layout_matches = {
                let mut nematode_roots = composite_queries.p6();
                if let Ok((mut transform, children, mut layout)) = nematode_roots.get_mut(entity) {
                    *transform = root_transform;
                    child_entities = children.iter().copied().collect();
                    let matches =
                        layout.segment_count == segment_count && layout.has_stylet == has_stylet;
                    if !matches {
                        layout.segment_count = segment_count;
                        layout.has_stylet = has_stylet;
                    }
                    matches
                } else {
                    false
                }
            };
            if child_entities.is_empty() && !layout_matches {
                commands.entity(entity).try_despawn_recursive();
                entity_map.nematodes.remove(&i);
                nematode_root = None;
            } else if layout_matches {
                let updated = {
                    let mut nematode_parts = composite_queries.p7();
                    update_nematode_parts(
                        &child_entities,
                        &mut nematode_parts,
                        &mut materials,
                        segment_count,
                        visual,
                        &body_base,
                        &head_base,
                    )
                };
                needs_rebuild = !updated;
            } else {
                needs_rebuild = true;
            }
        }

        let nematode_root = if let Some(entity) = nematode_root {
            entity
        } else {
            let entity = commands
                .spawn((
                    root_transform,
                    Visibility::default(),
                    NematodeEntity,
                    NematodeRigLayout {
                        segment_count,
                        has_stylet,
                    },
                ))
                .id();
            entity_map.nematodes.insert(i, entity);
            needs_rebuild = true;
            entity
        };

        if needs_rebuild {
            commands.entity(nematode_root).despawn_descendants();
            let body_mat = add_bio_material(
                &mut materials,
                body_base.clone(),
                mean_humidity,
                visual.activity,
                frame.snapshot.light.clamp(0.0, 1.0),
                frame.snapshot.time_s,
                FLAG_SUBSTRATE,
                visual.activity,
                0.08,
                config.use_bio_shader,
            );
            let head_mat = add_bio_material(
                &mut materials,
                head_base.clone(),
                mean_humidity,
                visual.activity,
                frame.snapshot.light.clamp(0.0, 1.0),
                frame.snapshot.time_s,
                FLAG_SUBSTRATE,
                visual.activity,
                0.08,
                config.use_bio_shader,
            );
            spawn_nematode_parts(
                &mut commands,
                nematode_root,
                &mesh_lib,
                segment_count,
                visual,
                &body_mat,
                &head_mat,
            );
        }
    }

    // ---- Soil surface ----
    despawn_stale(
        &mut commands,
        &mut entity_map.soil_surface,
        frame.soil_surface.len(),
    );

    for (i, &(x, y, visual)) in frame.soil_surface.iter().enumerate() {
        let shape_scale = soil_surface_shape_scale(visual.class);
        let surface_y = terrain_surface_height(
            &frame.surface_relief,
            frame.width,
            frame.height,
            x as f32 + 0.5,
            y as f32 + 0.5,
        );
        let transform = snap_transform(
            Transform::from_translation(Vec3::new(
                (x as f32 + 0.5) * VOXEL_SIZE,
                surface_y + 0.03 + visual.height_offset,
                (y as f32 + 0.5) * VOXEL_SIZE,
            ))
            .with_rotation(Quat::from_rotation_y(visual.yaw_rad))
            .with_scale(
                Vec3::new(
                    shape_scale.x * visual.width_scale,
                    shape_scale.y * visual.thickness_scale,
                    shape_scale.z * visual.depth_scale,
                ) * (0.55 + visual.sprite_scale * 0.55),
            ),
        );
        upsert_simple_entity(
            &mut commands,
            &mut entity_map.soil_surface,
            &mut materials,
            &mut simple_entity_queries.p3(),
            i,
            mesh_lib.soil_marker.clone(),
            StandardMaterial {
                base_color: pixel_color(visual.rgb),
                emissive: bevy::color::LinearRgba::new(
                    visual.accent_rgb[0] * 0.04,
                    visual.accent_rgb[1] * 0.04,
                    visual.accent_rgb[2] * 0.04,
                    1.0,
                ),
                ..default()
            },
            transform,
            SoilSurfaceEntity,
        );
    }

    // ---- HUD ----
    let snap = &frame.snapshot;
    for mut text in hud_query.iter_mut() {
        **text = format!(
            "oNeura 3D | {} | t={:.1}s\nFlies: {} | Energy: {:.0}\nPlants: {} | Cells: {:.0}\nFruits: {} | Water: {}\nAir: RH {:.0}% | Wind {:.3} | p {:.2}kPa\n{}\n{}\nFood: {:.0}",
            config.preset.label(),
            snap.time_s,
            frame.fly_states.len(),
            snap.avg_fly_energy,
            frame.plant_snapshots.len(),
            snap.total_plant_cells,
            frame.fruit_positions.len(),
            frame.water_cells.len(),
            mean_humidity * 100.0,
            mean_wind,
            mean_pressure,
            plant_species_line(snap),
            species_model_line(snap),
            snap.food_remaining,
        );
    }
}

fn despawn_stale(commands: &mut Commands, map: &mut HashMap<usize, Entity>, current_count: usize) {
    let stale: Vec<usize> = map
        .keys()
        .filter(|&&k| k >= current_count)
        .copied()
        .collect();
    for id in stale {
        if let Some(entity) = map.remove(&id) {
            commands.entity(entity).try_despawn_recursive();
        }
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
        if keyboard.pressed(KeyCode::Space) {
            orbit.focus.y += pan_speed;
        }
        if keyboard.pressed(KeyCode::ShiftLeft) {
            orbit.focus.y -= pan_speed;
        }
        if keyboard.pressed(KeyCode::KeyQ) {
            orbit.yaw -= pan_speed * 0.5;
        }
        if keyboard.pressed(KeyCode::KeyE) {
            orbit.yaw += pan_speed * 0.5;
        }

        if mouse_button.pressed(MouseButton::Right) {
            for motion in mouse_motion.read() {
                orbit.yaw -= motion.delta.x * 0.005;
                // Allow looking up to the sky (0.0) and down from the top (PI/2)
                orbit.pitch = (orbit.pitch + motion.delta.y * 0.005).clamp(0.01, std::f32::consts::PI - 0.01);
            }
        } else {
            mouse_motion.read().for_each(drop);
        }

        for scroll in mouse_wheel.read() {
            orbit.distance = (orbit.distance - scroll.y * 0.7).clamp(4.0, 80.0);
        }
        
        // Molecular zoom check
        if orbit.distance <= 4.1 && orbit.focus.x < 500.0 {
            orbit.focus = Vec3::new(1000.0, 1000.0, 1000.0);
            orbit.distance = 15.0;
            println!("Zoomed into molecular map!");
        } else if orbit.distance >= 20.0 && orbit.focus.x > 500.0 {
            orbit.focus = Vec3::new(11.0, 0.6, 8.0);
            orbit.distance = 5.0;
            println!("Zoomed out to macro ecosystem!");
        }

        if keyboard.just_pressed(KeyCode::KeyF) {
            orbit.auto_follow = match orbit.auto_follow {
                Some(_) => None,
                None => Some(0),
            };
        }

        if keyboard.just_pressed(KeyCode::Tab) {
            if let Some(ref mut idx) = orbit.auto_follow {
                *idx += 1;
            }
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
            let mut found = false;
            for (fly, transform) in fly_query.iter() {
                if fly.0 == fly_idx {
                    orbit.focus = orbit.focus.lerp(transform.translation, 0.1);
                    found = true;
                    break;
                }
            }
            if !found && fly_idx > 0 {
                orbit.auto_follow = Some(0);
            }
        }
    }
}

fn request_screenshot(
    mut commands: Commands,
    sim_time: Res<SimTime>,
    mut capture: ResMut<ScreenshotCapture>,
) {
    if capture.requested {
        return;
    }
    let Some(path) = capture.path.clone() else {
        return;
    };
    if sim_time.0 < capture.capture_after_s {
        return;
    }
    capture.requested = true;
    commands
        .spawn(Screenshot::primary_window())
        .observe(save_to_disk(path));
}

fn exit_after_screenshot(capture: Res<ScreenshotCapture>, mut exit_events: EventWriter<AppExit>) {
    if capture.completed {
        return;
    }
    let Some(path) = capture.path.as_ref() else {
        return;
    };
    if std::path::Path::new(path).exists() {
        exit_events.send(AppExit::Success);
    }
}

fn use_desktop_web_shell(config: &ViewerConfig) -> bool {
    !config.native_scene && config.screenshot_out.is_none()
}

fn pick_shell_port(preferred: Option<u16>) -> Result<u16, String> {
    if let Some(port) = preferred {
        return Ok(port);
    }
    for port in 8420..8450 {
        if TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return Ok(port);
        }
    }
    Err("failed to find a free localhost port in 8420-8449".to_string())
}

fn launch_browser(url: String) {
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(450));
        let _ = webbrowser::open(&url);
    });
}

fn run_desktop_web_shell(config: ViewerConfig, world: TerrariumWorld) -> Result<(), String> {
    let port = pick_shell_port(config.port)?;
    let url = format!("http://127.0.0.1:{port}/");
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|err| format!("failed to build tokio runtime: {err}"))?;

    runtime.block_on(async move {
        let state = AppState::new_from_world(world, 64, false, config.preset);

        {
            let mut params = state.params.write().await;
            params.target_fps = config.fps.round().clamp(1.0, 120.0) as u32;
        }

        let loop_state = state.clone();
        tokio::spawn(async move {
            frame_loop(loop_state).await;
        });

        let app = axum::Router::new()
            .route("/", axum::routing::get(index_handler))
            .route("/healthz", axum::routing::get(|| async { "ok" }))
            .route("/ws", axum::routing::get(ws_handler))
            .route("/api/snapshot", axum::routing::get(snapshot_handler))
            .route(
                "/api/snapshot_history",
                axum::routing::get(snapshot_history_handler),
            )
            .route("/api/inspect", axum::routing::get(inspect_handler))
            .route("/api/organisms", axum::routing::get(organisms_handler))
            .route(
                "/api/organisms/{id}/lineage",
                axum::routing::get(organism_lineage_handler),
            )
            .route(
                "/api/organisms/{id}/name",
                axum::routing::post(organism_rename_handler),
            )
            .route("/api/archive", axum::routing::get(archive_handler))
            .route("/api/checkpoint", axum::routing::get(checkpoint_handler))
            .route(
                "/api/import/checkpoint",
                axum::routing::post(import_checkpoint_handler)
                    .layer(axum::extract::DefaultBodyLimit::max(128 * 1024 * 1024)),
            )
            .route(
                "/api/tournament/submit",
                axum::routing::post(tournament_submit),
            )
            .route(
                "/api/tournament/leaderboard",
                axum::routing::get(tournament_leaderboard),
            )
            .route(
                "/api/tournament/genome/{id}",
                axum::routing::get(tournament_genome),
            )
            .route(
                "/api/tournament/{id}",
                axum::routing::delete(tournament_delete),
            )
            .route("/api/annotations", axum::routing::get(annotations_handler))
            .route("/api/export/bundle", axum::routing::get(export_bundle))
            .route("/api/auth/token", axum::routing::get(auth_token))
            .route(
                "/api/ecosystem/snapshot",
                axum::routing::get(ecosystem_snapshot_handler),
            )
            .route(
                "/api/ecosystem/start",
                axum::routing::post(ecosystem_start_handler),
            )
            .route(
                "/api/ecosystem/step",
                axum::routing::post(ecosystem_step_handler),
            )
            .route(
                "/api/ecosystem/run",
                axum::routing::post(ecosystem_run_handler),
            )
            .layer(
                CorsLayer::new()
                    .allow_origin(Any)
                    .allow_methods(Any)
                    .allow_headers(Any),
            )
            .with_state(state);

        let listener = tokio::net::TcpListener::bind(("127.0.0.1", port))
            .await
            .map_err(|err| format!("failed to bind desktop terrarium shell: {err}"))?;

        eprintln!(
            "Launching desktop terrarium shell at {} ({})",
            url,
            config.preset.cli_name()
        );
        if config.open_browser {
            launch_browser(url.clone());
        }

        axum::serve(listener, app)
            .await
            .map_err(|err| format!("desktop terrarium shell server error: {err}"))
    })
}

fn run_native_scene(config: ViewerConfig, world: TerrariumWorld) {
    let window_title = format!("oNeura Terrarium 3D | {}", config.preset.label());
    let screenshot_capture = ScreenshotCapture {
        path: config.screenshot_out.clone(),
        capture_after_s: config.screenshot_after_s,
        requested: false,
        completed: false,
    };
    let initial_state = InitialSimState {
        world: Some(world),
        checkpoint_out: config.checkpoint_out.clone(),
        archive_out: config.archive_out.clone(),
    };

    let mut app = App::new();

    app.add_plugins(
        DefaultPlugins
            .set(AssetPlugin {
                file_path: format!("{}/src", env!("CARGO_MANIFEST_DIR")),
                ..default()
            })
            .set(ImagePlugin::default_nearest())
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: window_title,
                    resolution: (1280.0, 720.0).into(),
                    present_mode: PresentMode::AutoVsync,
                    ..default()
                }),
                ..default()
            }),
    );
    app.insert_resource(config)
        .insert_resource(initial_state)
        .insert_resource(screenshot_capture)
        .add_systems(Startup, (setup_meshes, setup_scene, setup_hud).chain())
        .add_systems(
            Update,
            (
                consume_sim_frames,
                camera_control,
                auto_follow_system,
                request_screenshot,
                exit_after_screenshot,
            ),
        )
        .run();
}

// ============================================================================
// App
// ============================================================================

fn main() {
    let program_name = std::env::args()
        .next()
        .unwrap_or_else(|| "terrarium_3d".to_string());
    let mut config = match parse_cli(&program_name) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("{error}");
            print_usage(&program_name);
            std::process::exit(1);
        }
    };

    if config.archive_in.is_some() {
        if let Err(error) = inspect_archive(&config) {
            eprintln!("{error}");
            std::process::exit(1);
        }
        return;
    }

    let web_shell = use_desktop_web_shell(&config);
    if web_shell
        && !config.query_only
        && (config.checkpoint_out.is_some() || config.archive_out.is_some())
    {
        eprintln!(
            "--save-checkpoint/--save-archive require --native-scene, --screenshot-out, or --query-only in desktop mode"
        );
        std::process::exit(1);
    }

    let source_label = if web_shell {
        "terrarium_3d_web_shell"
    } else {
        "terrarium_3d"
    };
    let (mut world, present_preset) = match load_or_create_world(&config, source_label, web_shell) {
        Ok(result) => result,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    };
    config.preset = present_preset;

    if config.query_only || config.list_organisms || !config.lineage_ids.is_empty() {
        if let Err(error) = print_world_reports(&world, &config) {
            eprintln!("{error}");
            std::process::exit(1);
        }
        if config.query_only {
            if let Err(error) = save_world_outputs(&mut world, &config) {
                eprintln!("{error}");
                std::process::exit(1);
            }
            return;
        }
    }

    if web_shell {
        if let Err(err) = run_desktop_web_shell(config, world) {
            eprintln!("{err}");
            std::process::exit(1);
        }
        return;
    }

    run_native_scene(config, world);
}
