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
//!   --seed <n>       World seed (default: natural entropy)
//!   --fps <n>        Target framerate (default: 15)
//!   --frames <n>     Quit after N frames (default: infinite)
//!   --mode <name>    View mode: iso, top, split, heat, dash (default: iso)
//!   --no-color       Disable ANSI colors

use oneura_core::botany::NodeType;
use oneura_core::drosophila::BodyState;
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
    quantized_surface_height, rgb_to_u8, sample_visual_air, terrain_visual_response,
    water_visual_response, TerrariumEarthwormVisualResponse, TerrariumFlyVisualResponse,
    TerrariumNematodeVisualResponse, TerrariumPlantVisualResponse, TerrariumSoilSurfaceClass,
    TerrariumSoilSurfaceVisualResponse, TerrariumVisualAirSample, TerrariumWaterCycleInputs,
};
use oneura_core::terrarium::{
    resolve_seed_provenance, TerrariumAtmosphereFrame, TerrariumDemoPreset, TerrariumPlantSnapshot,
    TerrariumTopdownView, TerrariumWorld, TerrariumWorldSnapshot,
};

fn row_bg(y: usize) -> (u8, u8, u8) {
    if y % 2 == 0 {
        (10, 10, 20)
    } else {
        (15, 15, 25)
    }
}

use oneura_core::terrarium::evolve::ecosystem_dashboard;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet, VecDeque};
use std::env;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{self, Read, Write};
use std::process::ExitCode;
use std::sync::mpsc;
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

// ---------------------------------------------------------------------------
// 3D Math Primitives for True3D Rendering
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}
impl Vec3 {
    fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    fn add(self, v: Vec3) -> Vec3 {
        Vec3::new(self.x + v.x, self.y + v.y, self.z + v.z)
    }
    fn sub(self, v: Vec3) -> Vec3 {
        Vec3::new(self.x - v.x, self.y - v.y, self.z - v.z)
    }
    fn mul(self, s: f32) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
    fn dot(self, v: Vec3) -> f32 {
        self.x * v.x + self.y * v.y + self.z * v.z
    }
    fn cross(self, v: Vec3) -> Vec3 {
        Vec3::new(
            self.y * v.z - self.z * v.y,
            self.z * v.x - self.x * v.z,
            self.x * v.y - self.y * v.x,
        )
    }
    fn length(self) -> f32 {
        self.dot(self).sqrt()
    }
    fn normalize(self) -> Vec3 {
        let l = self.length();
        if l > 1e-6 {
            self.mul(1.0 / l)
        } else {
            self
        }
    }
}

struct Mat4 {
    m: [f32; 16],
}
impl Mat4 {
    fn identity() -> Self {
        let mut m = [0.0; 16];
        m[0] = 1.0;
        m[5] = 1.0;
        m[10] = 1.0;
        m[15] = 1.0;
        Self { m }
    }
    fn multiply(&self, other: &Mat4) -> Mat4 {
        let mut res = [0.0; 16];
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    res[i * 4 + j] += self.m[i * 4 + k] * other.m[k * 4 + j];
                }
            }
        }
        Mat4 { m: res }
    }
    fn transform(&self, v: Vec3) -> [f32; 4] {
        let x = v.x * self.m[0] + v.y * self.m[4] + v.z * self.m[8] + self.m[12];
        let y = v.x * self.m[1] + v.y * self.m[5] + v.z * self.m[9] + self.m[13];
        let z = v.x * self.m[2] + v.y * self.m[6] + v.z * self.m[10] + self.m[14];
        let w = v.x * self.m[3] + v.y * self.m[7] + v.z * self.m[11] + self.m[15];
        [x, y, z, w]
    }
}

fn look_at(eye: Vec3, center: Vec3, up: Vec3) -> Mat4 {
    let f = center.sub(eye).normalize();
    let s = f.cross(up).normalize();
    let u = s.cross(f);
    let mut m = Mat4::identity();
    m.m[0] = s.x;
    m.m[4] = s.y;
    m.m[8] = s.z;
    m.m[1] = u.x;
    m.m[5] = u.y;
    m.m[9] = u.z;
    m.m[2] = -f.x;
    m.m[6] = -f.y;
    m.m[10] = -f.z;
    m.m[12] = -s.dot(eye);
    m.m[13] = -u.dot(eye);
    m.m[14] = f.dot(eye);
    m
}

fn perspective(fovy: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let tan_half_fovy = (fovy / 2.0).tan();
    let mut m = Mat4::identity();
    m.m[0] = 1.0 / (aspect * tan_half_fovy);
    m.m[5] = 1.0 / tan_half_fovy;
    m.m[10] = -(far + near) / (far - near);
    m.m[11] = -1.0;
    m.m[14] = -(2.0 * far * near) / (far - near);
    m.m[15] = 0.0;
    m
}

// ---------------------------------------------------------------------------
// True3D Rendering Core
// ---------------------------------------------------------------------------

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq)]
enum EntityTag {
    None,
    Terrain,
    Plant(usize),
    Fly(usize),
    Water(usize),
    Fruit(usize),
}

#[derive(Clone)]
struct Vertex {
    pos: Vec3,
    #[allow(dead_code)]
    normal: Vec3,
    color: (u8, u8, u8),
}

#[derive(Clone)]
struct Triangle {
    v: [Vertex; 3],
    #[allow(dead_code)]
    tag: EntityTag,
}

fn lit_triangle_color(tri: &Triangle, light_dir: Vec3) -> (u8, u8, u8) {
    let edge_a = tri.v[1].pos.sub(tri.v[0].pos);
    let edge_b = tri.v[2].pos.sub(tri.v[0].pos);
    let normal = edge_a.cross(edge_b).normalize();
    let diffuse = normal.dot(light_dir).abs().max(0.2);
    let base = tri.v[0].color;
    (
        (base.0 as f32 * diffuse).min(255.0) as u8,
        (base.1 as f32 * diffuse).min(255.0) as u8,
        (base.2 as f32 * diffuse).min(255.0) as u8,
    )
}

const TRUE3D_TERRAIN_BASE_Y: f32 = -0.82;
const TRUE3D_COLUMN_HALF_XZ: f32 = 0.46;
const TRUE3D_WATER_HALF_XZ: f32 = 0.40;
const TRUE3D_STATIC_COLOR_LEVELS: usize = 5;
const TRUE3D_DYNAMIC_COLOR_LEVELS: usize = 5;
const TRUE3D_WORLD_STEP: f32 = 0.0625;
const TRUE3D_ANIMATION_HZ: f32 = 6.0;
const TRUE3D_STATE_MISSING: u64 = u64::MAX;
const TRUE3D_RENDER_KIND_TERRAIN_CELL: u64 = 1 << 56;
const TRUE3D_RENDER_KIND_SOIL_SURFACE: u64 = 2 << 56;
const TRUE3D_RENDER_KIND_PLANT: u64 = 3 << 56;
const TRUE3D_RENDER_KIND_FLY: u64 = 4 << 56;
const TRUE3D_RENDER_KIND_WATER_SOURCE: u64 = 5 << 56;
const TRUE3D_RENDER_KIND_FRUIT: u64 = 6 << 56;
const TRUE3D_RENDER_KIND_SEED: u64 = 7 << 56;
const TRUE3D_RENDER_KIND_EARTHWORM: u64 = 8 << 56;
const TRUE3D_RENDER_KIND_NEMATODE: u64 = 9 << 56;
const TRUE3D_RENDER_KIND_WATER_TILE: u64 = 10 << 56;

fn snap_true3d_scalar(value: f32) -> f32 {
    (value / TRUE3D_WORLD_STEP).round() * TRUE3D_WORLD_STEP
}

fn true3d_animation_time(time_s: f32) -> f32 {
    (time_s * TRUE3D_ANIMATION_HZ).floor() / TRUE3D_ANIMATION_HZ
}

fn snap_true3d_vec3(value: Vec3) -> Vec3 {
    Vec3::new(
        snap_true3d_scalar(value.x),
        snap_true3d_scalar(value.y),
        snap_true3d_scalar(value.z),
    )
}

fn true3d_render_id(kind: u64, index: usize) -> u64 {
    kind | index as u64
}

fn hash_true3d_scalar(value: f32, hasher: &mut DefaultHasher) {
    snap_true3d_scalar(value).to_bits().hash(hasher);
}

fn hash_true3d_vec3(value: Vec3, hasher: &mut DefaultHasher) {
    let value = snap_true3d_vec3(value);
    value.x.to_bits().hash(hasher);
    value.y.to_bits().hash(hasher);
    value.z.to_bits().hash(hasher);
}

fn hash_true3d_rgb(value: (u8, u8, u8), hasher: &mut DefaultHasher) {
    value.0.hash(hasher);
    value.1.hash(hasher);
    value.2.hash(hasher);
}

fn true3d_node_type_code(node_type: NodeType) -> u8 {
    match node_type {
        NodeType::Trunk => 0,
        NodeType::Branch => 1,
        NodeType::Leaf => 2,
        NodeType::Fruit => 3,
        NodeType::Bud => 4,
    }
}

fn pixel_rgb(rgb: [f32; 3], levels: usize) -> (u8, u8, u8) {
    rgb_to_u8(quantize_rgb(rgb, levels))
}

fn shade_rgb(color: (u8, u8, u8), factor: f32) -> (u8, u8, u8) {
    (
        (color.0 as f32 * factor).clamp(0.0, 255.0) as u8,
        (color.1 as f32 * factor).clamp(0.0, 255.0) as u8,
        (color.2 as f32 * factor).clamp(0.0, 255.0) as u8,
    )
}

fn push_box(
    tris: &mut Vec<Triangle>,
    center: Vec3,
    half: Vec3,
    color: (u8, u8, u8),
    tag: EntityTag,
) {
    let center = snap_true3d_vec3(center);
    let half = Vec3::new(
        snap_true3d_scalar(half.x).max(TRUE3D_WORLD_STEP),
        snap_true3d_scalar(half.y).max(TRUE3D_WORLD_STEP),
        snap_true3d_scalar(half.z).max(TRUE3D_WORLD_STEP),
    );
    let corners = [
        snap_true3d_vec3(Vec3::new(
            center.x - half.x,
            center.y - half.y,
            center.z - half.z,
        )),
        snap_true3d_vec3(Vec3::new(
            center.x + half.x,
            center.y - half.y,
            center.z - half.z,
        )),
        snap_true3d_vec3(Vec3::new(
            center.x + half.x,
            center.y + half.y,
            center.z - half.z,
        )),
        snap_true3d_vec3(Vec3::new(
            center.x - half.x,
            center.y + half.y,
            center.z - half.z,
        )),
        snap_true3d_vec3(Vec3::new(
            center.x - half.x,
            center.y - half.y,
            center.z + half.z,
        )),
        snap_true3d_vec3(Vec3::new(
            center.x + half.x,
            center.y - half.y,
            center.z + half.z,
        )),
        snap_true3d_vec3(Vec3::new(
            center.x + half.x,
            center.y + half.y,
            center.z + half.z,
        )),
        snap_true3d_vec3(Vec3::new(
            center.x - half.x,
            center.y + half.y,
            center.z + half.z,
        )),
    ];
    let side_color = shade_rgb(color, 0.82);
    let bottom_color = shade_rgb(color, 0.66);
    let faces = [
        ([3, 2, 6, 7], color),
        ([0, 1, 2, 3], side_color),
        ([5, 4, 7, 6], side_color),
        ([4, 0, 3, 7], side_color),
        ([1, 5, 6, 2], side_color),
        ([4, 5, 1, 0], bottom_color),
    ];
    for (face_indices, face_color) in faces {
        let quad = [
            corners[face_indices[0]],
            corners[face_indices[1]],
            corners[face_indices[2]],
            corners[face_indices[3]],
        ];
        for (v0, v1, v2) in [(quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])] {
            tris.push(Triangle {
                v: [
                    Vertex {
                        pos: v0,
                        normal: Vec3::new(0.0, 1.0, 0.0),
                        color: face_color,
                    },
                    Vertex {
                        pos: v1,
                        normal: Vec3::new(0.0, 1.0, 0.0),
                        color: face_color,
                    },
                    Vertex {
                        pos: v2,
                        normal: Vec3::new(0.0, 1.0, 0.0),
                        color: face_color,
                    },
                ],
                tag,
            });
        }
    }
}

fn soil_surface_class_code(class: TerrariumSoilSurfaceClass) -> u8 {
    match class {
        TerrariumSoilSurfaceClass::Mineral => 0,
        TerrariumSoilSurfaceClass::Humus => 1,
        TerrariumSoilSurfaceClass::WetDetritus => 2,
        TerrariumSoilSurfaceClass::MicrobialMat => 3,
        TerrariumSoilSurfaceClass::NitrifierCrust => 4,
        TerrariumSoilSurfaceClass::DenitrifierFilm => 5,
        TerrariumSoilSurfaceClass::MycorrhizalPatch => 6,
        TerrariumSoilSurfaceClass::EarthwormCast => 7,
        TerrariumSoilSurfaceClass::NematodeBloom => 8,
    }
}

fn hash_quantized_unit(value: f32, levels: u16, hasher: &mut DefaultHasher) {
    let bucket = (value.clamp(0.0, 1.0) * levels as f32).round() as u16;
    bucket.hash(hasher);
}

fn mean_visual_air_sample(atmosphere: &TerrariumAtmosphereFrame) -> TerrariumVisualAirSample {
    TerrariumVisualAirSample {
        temperature_c: mean_visual_field(&atmosphere.temperature_c, 20.0),
        humidity: mean_visual_field(&atmosphere.humidity, 0.5),
        pressure_kpa: mean_visual_field(&atmosphere.pressure_kpa, 101.325),
        pressure_delta_kpa: 0.0,
        wind_x: 0.0,
        wind_y: 0.0,
        wind_z: 0.0,
        wind_speed: mean_visual_wind_speed(atmosphere),
    }
}

fn true3d_mean_air_state_key(mean_air: TerrariumVisualAirSample) -> u64 {
    let mut hasher = DefaultHasher::new();
    hash_quantized_unit(mean_air.humidity, 16, &mut hasher);
    hash_quantized_unit(
        (mean_air.temperature_c / 40.0).clamp(0.0, 1.0),
        16,
        &mut hasher,
    );
    hash_quantized_unit((mean_air.wind_speed / 4.0).clamp(0.0, 1.0), 16, &mut hasher);
    hasher.finish()
}

fn true3d_plant_air_state_key(local_air: TerrariumVisualAirSample) -> u64 {
    let mut hasher = DefaultHasher::new();
    hash_quantized_unit(local_air.humidity.clamp(0.0, 1.0), 16, &mut hasher);
    hash_quantized_unit(
        ((local_air.temperature_c - 4.0) / 32.0).clamp(0.0, 1.0),
        16,
        &mut hasher,
    );
    hash_quantized_unit(
        ((local_air.pressure_delta_kpa / 2.0) * 0.5 + 0.5).clamp(0.0, 1.0),
        16,
        &mut hasher,
    );
    hash_quantized_unit(
        (local_air.wind_speed / 4.0).clamp(0.0, 1.0),
        16,
        &mut hasher,
    );
    hash_quantized_unit(
        ((local_air.wind_x / 2.0) * 0.5 + 0.5).clamp(0.0, 1.0),
        12,
        &mut hasher,
    );
    hash_quantized_unit(
        ((local_air.wind_y / 2.0) * 0.5 + 0.5).clamp(0.0, 1.0),
        12,
        &mut hasher,
    );
    hasher.finish()
}

fn plant_true3d_shape_state_key(plant: &TerrariumPlantSnapshot) -> u64 {
    let mut hasher = DefaultHasher::new();
    plant.morphology.len().hash(&mut hasher);
    for node in &plant.morphology {
        true3d_node_type_code(node.node_type).hash(&mut hasher);
        hash_true3d_scalar(node.position[0], &mut hasher);
        hash_true3d_scalar(node.position[1], &mut hasher);
        hash_true3d_scalar(node.position[2], &mut hasher);
        hash_true3d_scalar(node.radius, &mut hasher);
    }
    hasher.finish()
}

fn plant_true3d_state_key(
    plant: &TerrariumPlantSnapshot,
    ground_y: f32,
    local_air_key: u64,
    vitality: f32,
    canopy_density: f32,
    time_s: f32,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    plant.taxonomy_id.hash(&mut hasher);
    plant.x.hash(&mut hasher);
    plant.y.hash(&mut hasher);
    plant_true3d_shape_state_key(plant).hash(&mut hasher);
    hash_true3d_scalar(ground_y, &mut hasher);
    local_air_key.hash(&mut hasher);
    hash_quantized_unit(vitality.clamp(0.0, 1.0), 16, &mut hasher);
    hash_quantized_unit(canopy_density.clamp(0.0, 1.0), 16, &mut hasher);
    time_s.to_bits().hash(&mut hasher);
    hasher.finish()
}

fn true3d_fly_air_state_key(local_air: TerrariumVisualAirSample) -> u64 {
    let mut hasher = DefaultHasher::new();
    hash_quantized_unit(local_air.humidity.clamp(0.0, 1.0), 16, &mut hasher);
    hash_quantized_unit(
        (local_air.wind_speed / 4.0).clamp(0.0, 1.0),
        16,
        &mut hasher,
    );
    hash_quantized_unit(
        ((local_air.wind_x / 2.0) * 0.5 + 0.5).clamp(0.0, 1.0),
        16,
        &mut hasher,
    );
    hash_quantized_unit(
        ((local_air.wind_y / 2.0) * 0.5 + 0.5).clamp(0.0, 1.0),
        16,
        &mut hasher,
    );
    hasher.finish()
}

fn fly_true3d_state_key(
    body: &BodyState,
    local_air_key: u64,
    energy_frac: f32,
    time_s: f32,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    hash_true3d_scalar(body.x, &mut hasher);
    hash_true3d_scalar(body.y, &mut hasher);
    hash_true3d_scalar(body.z, &mut hasher);
    hash_true3d_scalar(body.heading, &mut hasher);
    hash_true3d_scalar(body.pitch, &mut hasher);
    hash_true3d_scalar(body.roll, &mut hasher);
    hash_quantized_unit((body.speed / 1.2).clamp(0.0, 1.0), 16, &mut hasher);
    hash_quantized_unit(
        (body.vertical_velocity.abs() / 12.0).clamp(0.0, 1.0),
        16,
        &mut hasher,
    );
    hash_quantized_unit(energy_frac.clamp(0.0, 1.0), 16, &mut hasher);
    hash_quantized_unit(
        ((body.temperature - 12.0) / 18.0).clamp(0.0, 1.0),
        16,
        &mut hasher,
    );
    body.is_flying.hash(&mut hasher);
    body.proboscis_extended.hash(&mut hasher);
    hash_quantized_unit(
        ((body.wing_beat_freq - 20.0) / 280.0).clamp(0.0, 1.0),
        16,
        &mut hasher,
    );
    hash_quantized_unit(
        ((body.wing_stroke / 2.0) + 0.5).clamp(0.0, 1.0),
        16,
        &mut hasher,
    );
    hash_quantized_unit(
        ((body.wing_dihedral / 1.2) * 0.5 + 0.5).clamp(0.0, 1.0),
        16,
        &mut hasher,
    );
    hash_quantized_unit(
        ((body.wing_sweep / 1.4) * 0.5 + 0.5).clamp(0.0, 1.0),
        16,
        &mut hasher,
    );
    local_air_key.hash(&mut hasher);
    time_s.to_bits().hash(&mut hasher);
    hasher.finish()
}

fn terrain_cell_true3d_state_key(
    ground_y: f32,
    moisture_t: f32,
    mean_air_key: u64,
    water_cycle: TerrariumWaterCycleInputs,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    hash_true3d_scalar(ground_y, &mut hasher);
    hash_quantized_unit(moisture_t, 12, &mut hasher);
    mean_air_key.hash(&mut hasher);
    let pooled_water_t = ((moisture_t - 0.82) / 0.18).clamp(0.0, 1.0);
    if pooled_water_t > 0.08 {
        hash_quantized_unit(water_cycle.lunar_phase.clamp(0.0, 1.0), 32, &mut hasher);
        hash_quantized_unit(water_cycle.moonlight.clamp(0.0, 1.0), 16, &mut hasher);
        hash_quantized_unit(
            ((water_cycle.tidal_moisture_factor - 0.85) / 0.30).clamp(0.0, 1.0),
            16,
            &mut hasher,
        );
    }
    hasher.finish()
}

fn terrain_cell_true3d_fingerprint(
    ground_y: f32,
    moisture_t: f32,
    mean_air: TerrariumVisualAirSample,
    water_cycle: TerrariumWaterCycleInputs,
    center_x: f32,
    center_z: f32,
    cell_size: f32,
    cell_idx: usize,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    let terrain_rgb = pixel_rgb(
        terrain_visual_response(moisture_t, 0.5, mean_air).rgb,
        TRUE3D_STATIC_COLOR_LEVELS,
    );
    let column_height = (ground_y - TRUE3D_TERRAIN_BASE_Y).max(0.18);
    let column_center = Vec3::new(
        center_x,
        TRUE3D_TERRAIN_BASE_Y + column_height * 0.5,
        center_z,
    );
    hash_true3d_vec3(column_center, &mut hasher);
    hash_true3d_scalar(cell_size * TRUE3D_COLUMN_HALF_XZ, &mut hasher);
    hash_true3d_scalar(column_height * 0.5, &mut hasher);
    hash_true3d_rgb(terrain_rgb, &mut hasher);

    let pooled_water_t = ((moisture_t - 0.82) / 0.18).clamp(0.0, 1.0);
    (pooled_water_t > 0.08).hash(&mut hasher);
    if pooled_water_t > 0.08 {
        let water_visual = water_visual_response(
            mean_air,
            pooled_water_t * 120.0,
            0.0,
            cell_idx as f32,
            water_cycle,
        );
        let water_height =
            (0.10 + pooled_water_t * 0.18).clamp(0.06, 0.22) * water_visual.thickness_scale;
        let water_center = Vec3::new(center_x, ground_y + water_height * 0.5 + 0.03, center_z);
        hash_true3d_vec3(water_center, &mut hasher);
        hash_true3d_scalar(cell_size * TRUE3D_WATER_HALF_XZ, &mut hasher);
        hash_true3d_scalar(water_height * 0.5, &mut hasher);
        hash_true3d_rgb(
            pixel_rgb(water_visual.rgb, TRUE3D_STATIC_COLOR_LEVELS),
            &mut hasher,
        );
    }
    hasher.finish()
}

fn push_true3d_terrain_cell(
    tris: &mut Vec<Triangle>,
    ground_y: f32,
    moisture_t: f32,
    mean_air: TerrariumVisualAirSample,
    water_cycle: TerrariumWaterCycleInputs,
    center_x: f32,
    center_z: f32,
    cell_size: f32,
    cell_idx: usize,
) {
    let terrain_rgb = pixel_rgb(
        terrain_visual_response(moisture_t, 0.5, mean_air).rgb,
        TRUE3D_STATIC_COLOR_LEVELS,
    );
    let column_height = (ground_y - TRUE3D_TERRAIN_BASE_Y).max(0.18);
    let column_center = Vec3::new(
        center_x,
        TRUE3D_TERRAIN_BASE_Y + column_height * 0.5,
        center_z,
    );
    push_box(
        tris,
        column_center,
        Vec3::new(
            cell_size * TRUE3D_COLUMN_HALF_XZ,
            column_height * 0.5,
            cell_size * TRUE3D_COLUMN_HALF_XZ,
        ),
        terrain_rgb,
        EntityTag::Terrain,
    );

    let pooled_water_t = ((moisture_t - 0.82) / 0.18).clamp(0.0, 1.0);
    if pooled_water_t > 0.08 {
        let water_visual = water_visual_response(
            mean_air,
            pooled_water_t * 120.0,
            0.0,
            cell_idx as f32,
            water_cycle,
        );
        let water_height =
            (0.10 + pooled_water_t * 0.18).clamp(0.06, 0.22) * water_visual.thickness_scale;
        let water_center = Vec3::new(center_x, ground_y + water_height * 0.5 + 0.03, center_z);
        push_box(
            tris,
            water_center,
            Vec3::new(
                cell_size * TRUE3D_WATER_HALF_XZ,
                water_height * 0.5,
                cell_size * TRUE3D_WATER_HALF_XZ,
            ),
            pixel_rgb(water_visual.rgb, TRUE3D_STATIC_COLOR_LEVELS),
            EntityTag::Terrain,
        );
    }
}

fn water_true3d_state_key(
    local_air: TerrariumVisualAirSample,
    volume: f32,
    time_s: f32,
    phase_seed: f32,
    water_cycle: TerrariumWaterCycleInputs,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    hash_quantized_unit(
        ((local_air.temperature_c - 4.0) / 26.0).clamp(0.0, 1.0),
        16,
        &mut hasher,
    );
    hash_quantized_unit(local_air.humidity.clamp(0.0, 1.0), 16, &mut hasher);
    hash_quantized_unit(
        ((local_air.pressure_delta_kpa / 2.0) * 0.5 + 0.5).clamp(0.0, 1.0),
        16,
        &mut hasher,
    );
    hash_quantized_unit(
        (local_air.wind_speed / 4.0).clamp(0.0, 1.0),
        16,
        &mut hasher,
    );
    hash_quantized_unit(
        ((local_air.wind_x / 2.0) * 0.5 + 0.5).clamp(0.0, 1.0),
        12,
        &mut hasher,
    );
    hash_quantized_unit(
        ((local_air.wind_y / 2.0) * 0.5 + 0.5).clamp(0.0, 1.0),
        12,
        &mut hasher,
    );
    hash_quantized_unit((volume / 120.0).clamp(0.0, 1.0), 16, &mut hasher);
    hash_quantized_unit(water_cycle.lunar_phase.clamp(0.0, 1.0), 32, &mut hasher);
    hash_quantized_unit(water_cycle.moonlight.clamp(0.0, 1.0), 16, &mut hasher);
    hash_quantized_unit(
        ((water_cycle.tidal_moisture_factor - 0.85) / 0.30).clamp(0.0, 1.0),
        16,
        &mut hasher,
    );
    time_s.to_bits().hash(&mut hasher);
    phase_seed.to_bits().hash(&mut hasher);
    hasher.finish()
}

fn water_source_true3d_state_key(
    local_air: TerrariumVisualAirSample,
    volume: f32,
    time_s: f32,
    phase_seed: f32,
    water_cycle: TerrariumWaterCycleInputs,
    wx: usize,
    wy: usize,
    ground_y: f32,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    water_true3d_state_key(local_air, volume, time_s, phase_seed, water_cycle).hash(&mut hasher);
    wx.hash(&mut hasher);
    wy.hash(&mut hasher);
    hash_true3d_scalar(ground_y, &mut hasher);
    hasher.finish()
}

fn soil_surface_true3d_fingerprint(
    gx: usize,
    gy: usize,
    visual: TerrariumSoilSurfaceVisualResponse,
    ground_y: f32,
    ox: f32,
    oz: f32,
    cell_size: f32,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    gx.hash(&mut hasher);
    gy.hash(&mut hasher);
    soil_surface_class_code(visual.class).hash(&mut hasher);
    let center = Vec3::new(
        ox + gx as f32 * cell_size,
        ground_y + 0.04 + visual.height_offset * 5.0,
        oz + gy as f32 * cell_size,
    );
    if visual.thickness_scale > 0.10 {
        let box_center = center.add(Vec3::new(0.0, visual.thickness_scale * 0.04, 0.0));
        hash_true3d_vec3(box_center, &mut hasher);
        hash_true3d_scalar((visual.width_scale * 0.18).clamp(0.08, 0.26), &mut hasher);
        hash_true3d_scalar(
            (visual.thickness_scale * 0.05).clamp(0.02, 0.08),
            &mut hasher,
        );
        hash_true3d_scalar((visual.depth_scale * 0.18).clamp(0.08, 0.26), &mut hasher);
        hash_true3d_rgb(
            pixel_rgb(visual.accent_rgb, TRUE3D_STATIC_COLOR_LEVELS),
            &mut hasher,
        );
    } else {
        hash_true3d_vec3(center, &mut hasher);
        hash_true3d_scalar((visual.width_scale * 0.34).clamp(0.14, 0.62), &mut hasher);
        hash_true3d_scalar((visual.depth_scale * 0.34).clamp(0.14, 0.62), &mut hasher);
        hash_true3d_rgb(
            pixel_rgb(visual.rgb, TRUE3D_STATIC_COLOR_LEVELS),
            &mut hasher,
        );
    }
    hasher.finish()
}

fn push_soil_surface_true3d(
    tris: &mut Vec<Triangle>,
    gx: usize,
    gy: usize,
    visual: TerrariumSoilSurfaceVisualResponse,
    ground_y: f32,
    ox: f32,
    oz: f32,
    cell_size: f32,
) {
    let center = Vec3::new(
        ox + gx as f32 * cell_size,
        ground_y + 0.04 + visual.height_offset * 5.0,
        oz + gy as f32 * cell_size,
    );
    if visual.thickness_scale > 0.10 {
        push_box(
            tris,
            center.add(Vec3::new(0.0, visual.thickness_scale * 0.04, 0.0)),
            Vec3::new(
                (visual.width_scale * 0.18).clamp(0.08, 0.26),
                (visual.thickness_scale * 0.05).clamp(0.02, 0.08),
                (visual.depth_scale * 0.18).clamp(0.08, 0.26),
            ),
            pixel_rgb(visual.accent_rgb, TRUE3D_STATIC_COLOR_LEVELS),
            EntityTag::Terrain,
        );
    } else {
        push_quad_xz(
            tris,
            center,
            (visual.width_scale * 0.34).clamp(0.14, 0.62),
            (visual.depth_scale * 0.34).clamp(0.14, 0.62),
            pixel_rgb(visual.rgb, TRUE3D_STATIC_COLOR_LEVELS),
            EntityTag::Terrain,
        );
    }
}

fn push_diamond(
    tris: &mut Vec<Triangle>,
    center: Vec3,
    rx: f32,
    ry: f32,
    rz: f32,
    color: (u8, u8, u8),
    tag: EntityTag,
) {
    let center = snap_true3d_vec3(center);
    let rx = snap_true3d_scalar(rx).max(TRUE3D_WORLD_STEP);
    let ry = snap_true3d_scalar(ry).max(TRUE3D_WORLD_STEP);
    let rz = snap_true3d_scalar(rz).max(TRUE3D_WORLD_STEP);
    let top = snap_true3d_vec3(Vec3::new(center.x, center.y + ry, center.z));
    let bot = snap_true3d_vec3(Vec3::new(center.x, center.y - ry, center.z));
    let f = snap_true3d_vec3(Vec3::new(center.x, center.y, center.z + rz));
    let b = snap_true3d_vec3(Vec3::new(center.x, center.y, center.z - rz));
    let l = snap_true3d_vec3(Vec3::new(center.x - rx, center.y, center.z));
    let r = snap_true3d_vec3(Vec3::new(center.x + rx, center.y, center.z));
    let faces = [
        (top, r, f),
        (top, f, l),
        (top, l, b),
        (top, b, r),
        (bot, f, r),
        (bot, l, f),
        (bot, b, l),
        (bot, r, b),
    ];
    for (v0, v1, v2) in faces {
        tris.push(Triangle {
            v: [
                Vertex {
                    pos: v0,
                    normal: Vec3::new(0.0, 1.0, 0.0),
                    color,
                },
                Vertex {
                    pos: v1,
                    normal: Vec3::new(0.0, 1.0, 0.0),
                    color,
                },
                Vertex {
                    pos: v2,
                    normal: Vec3::new(0.0, 1.0, 0.0),
                    color,
                },
            ],
            tag,
        });
    }
}

fn push_quad_xz(
    tris: &mut Vec<Triangle>,
    center: Vec3,
    half_x: f32,
    half_z: f32,
    color: (u8, u8, u8),
    tag: EntityTag,
) {
    let center = snap_true3d_vec3(center);
    let half_x = snap_true3d_scalar(half_x).max(TRUE3D_WORLD_STEP);
    let half_z = snap_true3d_scalar(half_z).max(TRUE3D_WORLD_STEP);
    let y = center.y;
    let p00 = snap_true3d_vec3(Vec3::new(center.x - half_x, y, center.z - half_z));
    let p10 = snap_true3d_vec3(Vec3::new(center.x + half_x, y, center.z - half_z));
    let p01 = snap_true3d_vec3(Vec3::new(center.x - half_x, y, center.z + half_z));
    let p11 = snap_true3d_vec3(Vec3::new(center.x + half_x, y, center.z + half_z));
    let normal = Vec3::new(0.0, 1.0, 0.0);
    for face in [(p00, p10, p01), (p10, p11, p01)] {
        tris.push(Triangle {
            v: [
                Vertex {
                    pos: face.0,
                    normal,
                    color,
                },
                Vertex {
                    pos: face.1,
                    normal,
                    color,
                },
                Vertex {
                    pos: face.2,
                    normal,
                    color,
                },
            ],
            tag,
        });
    }
}

fn push_earthworm_true3d(
    tris: &mut Vec<Triangle>,
    ox: f32,
    oz: f32,
    cell_size: f32,
    gx: f32,
    gy: f32,
    visual: TerrariumEarthwormVisualResponse,
    _tag_idx: usize,
) {
    let segments = visual.segment_count.max(4) as usize;
    for seg in 0..segments {
        let t = if segments > 1 {
            seg as f32 / (segments - 1) as f32
        } else {
            0.0
        };
        let x = ox + gx * cell_size + (t - 0.5) * visual.length_scale * 0.8;
        let z = oz
            + gy * cell_size
            + visual.curl * (std::f32::consts::PI * t).sin() * 0.30
            + visual.yaw_rad.sin() * 0.08;
        let y =
            0.10 + visual.height_offset + (std::f32::consts::PI * t).sin() * 0.04 * visual.activity;
        let taper = 0.72 + (1.0 - ((t - 0.5) * 2.0).abs()) * 0.28;
        let color = if (0.38..=0.58).contains(&t) {
            pixel_rgb(visual.clitellum_rgb, TRUE3D_DYNAMIC_COLOR_LEVELS)
        } else {
            pixel_rgb(visual.body_rgb, TRUE3D_DYNAMIC_COLOR_LEVELS)
        };
        push_diamond(
            tris,
            Vec3::new(x, y, z),
            0.07 * visual.thickness_scale * taper,
            0.05 * visual.thickness_scale * taper,
            0.10 * visual.thickness_scale * taper,
            color,
            EntityTag::Terrain,
        );
    }
}

fn push_nematode_true3d(
    tris: &mut Vec<Triangle>,
    ox: f32,
    oz: f32,
    cell_size: f32,
    gx: f32,
    gy: f32,
    visual: TerrariumNematodeVisualResponse,
    _tag_idx: usize,
) {
    let segments = 6usize;
    for seg in 0..segments {
        let t = if segments > 1 {
            seg as f32 / (segments - 1) as f32
        } else {
            0.0
        };
        let x = ox + gx * cell_size + (t - 0.5) * visual.length_scale * 0.54;
        let z = oz + gy * cell_size + visual.curl * (std::f32::consts::PI * t).sin() * 0.18;
        let taper = 1.0 - t * 0.42;
        let color = if seg == segments - 1 {
            pixel_rgb(visual.head_rgb, TRUE3D_DYNAMIC_COLOR_LEVELS)
        } else {
            pixel_rgb(visual.body_rgb, TRUE3D_DYNAMIC_COLOR_LEVELS)
        };
        push_diamond(
            tris,
            Vec3::new(x, 0.08 + visual.height_offset, z),
            0.04 * visual.thickness_scale * taper,
            0.03 * visual.thickness_scale * taper,
            0.06 * visual.thickness_scale * taper,
            color,
            EntityTag::Terrain,
        );
    }

    if visual.stylet_length_scale > 0.0 {
        push_diamond(
            tris,
            Vec3::new(
                ox + gx * cell_size + visual.length_scale * 0.18,
                0.08 + visual.height_offset,
                oz + gy * cell_size,
            ),
            0.01,
            0.01,
            0.05 + visual.stylet_length_scale * 0.05,
            pixel_rgb(visual.head_rgb, TRUE3D_DYNAMIC_COLOR_LEVELS),
            EntityTag::Terrain,
        );
    }
}

fn plant_true3d_fingerprint(
    plant: &TerrariumPlantSnapshot,
    visual: TerrariumPlantVisualResponse,
    ground_y: f32,
    ox: f32,
    oz: f32,
    cell_size: f32,
) -> u64 {
    let base_x = ox + plant.x as f32 * cell_size;
    let base_z = oz + plant.y as f32 * cell_size;
    let mut hasher = DefaultHasher::new();
    plant.taxonomy_id.hash(&mut hasher);
    plant.x.hash(&mut hasher);
    plant.y.hash(&mut hasher);
    for node in &plant.morphology {
        let pos = Vec3::new(
            base_x
                + node.position[0] * 0.8
                + visual.sway_x * (0.14 + node.position[1].max(0.0) * 0.05),
            ground_y + node.position[1] * 0.7 + visual.vertical_offset * 6.0,
            base_z
                + node.position[2] * 0.8
                + visual.sway_z * (0.14 + node.position[1].max(0.0) * 0.05),
        );
        let color = match node.node_type {
            NodeType::Trunk | NodeType::Branch | NodeType::Bud => {
                pixel_rgb(visual.stem_rgb, TRUE3D_DYNAMIC_COLOR_LEVELS)
            }
            NodeType::Leaf => pixel_rgb(visual.leaf_rgb, TRUE3D_DYNAMIC_COLOR_LEVELS),
            NodeType::Fruit => pixel_rgb(visual.fruit_rgb, TRUE3D_DYNAMIC_COLOR_LEVELS),
        };
        let (rx, ry, rz) = match node.node_type {
            NodeType::Leaf => {
                let s = (node.radius * visual.canopy_scale).clamp(0.12, 0.42);
                (s * 1.1, s * 0.7, s * 1.1)
            }
            NodeType::Fruit => (0.18, 0.18, 0.18),
            _ => {
                let s = node.radius.clamp(0.08, 0.22);
                (s * 0.8, s * 1.2, s * 0.8)
            }
        };
        true3d_node_type_code(node.node_type).hash(&mut hasher);
        hash_true3d_vec3(pos, &mut hasher);
        hash_true3d_scalar(rx, &mut hasher);
        hash_true3d_scalar(ry, &mut hasher);
        hash_true3d_scalar(rz, &mut hasher);
        hash_true3d_rgb(color, &mut hasher);
    }
    hasher.finish()
}

fn push_plant_true3d(
    tris: &mut Vec<Triangle>,
    plant: &TerrariumPlantSnapshot,
    visual: TerrariumPlantVisualResponse,
    ground_y: f32,
    ox: f32,
    oz: f32,
    cell_size: f32,
    tag_idx: usize,
) {
    let base_x = ox + plant.x as f32 * cell_size;
    let base_z = oz + plant.y as f32 * cell_size;
    let tag = EntityTag::Plant(tag_idx);

    // If the server computed a ribbon mesh for this plant (trees), use it.
    if let Some(ref mesh) = plant.branch_mesh {
        let positions = &mesh.positions;
        let normals = &mesh.normals;
        let indices = &mesh.indices;
        let colors = &mesh.colors;
        let num_tris = indices.len() / 3;
        for t in 0..num_tris {
            let i0 = indices[t * 3] as usize;
            let i1 = indices[t * 3 + 1] as usize;
            let i2 = indices[t * 3 + 2] as usize;
            let v = |idx: usize| -> Vertex {
                let px = base_x + positions[idx * 3] * 0.7
                    + visual.sway_x * positions[idx * 3 + 1].max(0.0) * 0.04;
                let py = ground_y + positions[idx * 3 + 1] * 0.7
                    + visual.vertical_offset * 6.0;
                let pz = base_z + positions[idx * 3 + 2] * 0.7
                    + visual.sway_z * positions[idx * 3 + 1].max(0.0) * 0.04;
                let cr = (colors[idx * 3] * 255.0) as u8;
                let cg = (colors[idx * 3 + 1] * 255.0) as u8;
                let cb = (colors[idx * 3 + 2] * 255.0) as u8;
                Vertex {
                    pos: Vec3::new(px, py, pz),
                    normal: Vec3::new(
                        normals[idx * 3],
                        normals[idx * 3 + 1],
                        normals[idx * 3 + 2],
                    ),
                    color: (cr, cg, cb),
                }
            };
            tris.push(Triangle {
                v: [v(i0), v(i1), v(i2)],
                tag,
            });
        }
    }

    // Leaf and fruit nodes still rendered as diamonds (canopy + fruits)
    for node in &plant.morphology {
        if !matches!(node.node_type, NodeType::Leaf | NodeType::Fruit) {
            // Trunk/Branch/Bud: skip if ribbon mesh handled them
            if plant.branch_mesh.is_some() {
                continue;
            }
        }
        let pos = Vec3::new(
            base_x
                + node.position[0] * 0.8
                + visual.sway_x * (0.14 + node.position[1].max(0.0) * 0.05),
            ground_y + node.position[1] * 0.7 + visual.vertical_offset * 6.0,
            base_z
                + node.position[2] * 0.8
                + visual.sway_z * (0.14 + node.position[1].max(0.0) * 0.05),
        );
        let color = match node.node_type {
            NodeType::Trunk | NodeType::Branch | NodeType::Bud => {
                pixel_rgb(visual.stem_rgb, TRUE3D_DYNAMIC_COLOR_LEVELS)
            }
            NodeType::Leaf => pixel_rgb(visual.leaf_rgb, TRUE3D_DYNAMIC_COLOR_LEVELS),
            NodeType::Fruit => pixel_rgb(visual.fruit_rgb, TRUE3D_DYNAMIC_COLOR_LEVELS),
        };
        let (rx, ry, rz) = match node.node_type {
            NodeType::Leaf => {
                let s = (node.radius * visual.canopy_scale).clamp(0.12, 0.42);
                (s * 1.1, s * 0.7, s * 1.1)
            }
            NodeType::Fruit => (0.18, 0.18, 0.18),
            _ => {
                let s = node.radius.clamp(0.08, 0.22);
                (s * 0.8, s * 1.2, s * 0.8)
            }
        };
        push_diamond(tris, pos, rx, ry, rz, color, tag);
    }
}

fn fly_true3d_fingerprint(
    body: &BodyState,
    visual: TerrariumFlyVisualResponse,
    ox: f32,
    oz: f32,
    cell_size: f32,
) -> u64 {
    let body_color = pixel_rgb(visual.body_rgb, TRUE3D_DYNAMIC_COLOR_LEVELS);
    let wing_color = pixel_rgb(visual.wing_rgb, TRUE3D_DYNAMIC_COLOR_LEVELS);
    let fx = ox + body.x * cell_size;
    let fz = oz + body.y * cell_size;
    let fy = body.z * 2.0 + 2.0 + visual.pitch.abs() * 0.3;
    let body_center = Vec3::new(fx, fy, fz + 0.08);
    let body_scale = 0.20 * visual.sprite_scale * visual.thorax_scale;
    let abdomen_center = body_center.sub(Vec3::new(0.0, 0.02, 0.32 * visual.abdomen_scale));
    let head_center = body_center.add(Vec3::new(0.0, 0.02, 0.38 * visual.head_scale));
    let forward = Vec3::new(body.heading.sin(), 0.0, body.heading.cos()).normalize();
    let right = Vec3::new(forward.z, 0.0, -forward.x);
    let wing_root_l = body_center
        .add(right.mul(0.12 * visual.sprite_scale * visual.leg_span))
        .add(Vec3::new(0.0, 0.08 * visual.sprite_scale, 0.0));
    let wing_root_r = body_center
        .sub(right.mul(0.12 * visual.sprite_scale * visual.leg_span))
        .add(Vec3::new(0.0, 0.08 * visual.sprite_scale, 0.0));
    let wing_lift = if body.is_flying {
        0.16 + visual.wing_angle.abs() * 0.18
    } else {
        0.05 + visual.wing_angle.abs() * 0.05
    };
    let wing_span = 0.38 * visual.sprite_scale * visual.wing_span;
    let wing_sweep = forward.mul(visual.wing_angle * 0.10);
    let wing_l_a = wing_root_l.add(right.mul(wing_span)).add(wing_sweep);
    let wing_l_b = wing_root_l
        .add(right.mul(wing_span * (0.45 + visual.wing_width * 0.30)))
        .sub(wing_sweep)
        .add(Vec3::new(0.0, wing_lift, 0.0));
    let wing_r_a = wing_root_r.sub(right.mul(wing_span)).add(wing_sweep);
    let wing_r_b = wing_root_r
        .sub(right.mul(wing_span * (0.45 + visual.wing_width * 0.30)))
        .sub(wing_sweep)
        .add(Vec3::new(0.0, wing_lift, 0.0));

    let mut hasher = DefaultHasher::new();
    hash_true3d_vec3(body_center, &mut hasher);
    hash_true3d_vec3(abdomen_center, &mut hasher);
    hash_true3d_vec3(head_center, &mut hasher);
    hash_true3d_scalar(body_scale, &mut hasher);
    hash_true3d_scalar(body_scale * 0.78 * visual.abdomen_scale, &mut hasher);
    hash_true3d_scalar(body_scale * 0.62 * visual.head_scale, &mut hasher);
    hash_true3d_vec3(wing_root_l, &mut hasher);
    hash_true3d_vec3(wing_l_a, &mut hasher);
    hash_true3d_vec3(wing_l_b, &mut hasher);
    hash_true3d_vec3(wing_root_r, &mut hasher);
    hash_true3d_vec3(wing_r_a, &mut hasher);
    hash_true3d_vec3(wing_r_b, &mut hasher);
    hash_true3d_rgb(body_color, &mut hasher);
    hash_true3d_rgb(wing_color, &mut hasher);
    if visual.proboscis_extension > 0.0 {
        hash_true3d_vec3(
            head_center.add(forward.mul(0.18 + visual.proboscis_extension * 0.24)),
            &mut hasher,
        );
        hash_true3d_scalar(0.10 + visual.proboscis_extension * 0.12, &mut hasher);
    }
    hasher.finish()
}

fn push_fly_true3d(
    tris: &mut Vec<Triangle>,
    body: &BodyState,
    visual: TerrariumFlyVisualResponse,
    ox: f32,
    oz: f32,
    cell_size: f32,
    tag_idx: usize,
) {
    let body_color = pixel_rgb(visual.body_rgb, TRUE3D_DYNAMIC_COLOR_LEVELS);
    let wing_color = pixel_rgb(visual.wing_rgb, TRUE3D_DYNAMIC_COLOR_LEVELS);
    let fx = ox + body.x * cell_size;
    let fz = oz + body.y * cell_size;
    let fy = body.z * 2.0 + 2.0 + visual.pitch.abs() * 0.3;
    let body_center = Vec3::new(fx, fy, fz + 0.08);
    let body_scale = 0.20 * visual.sprite_scale * visual.thorax_scale;
    push_diamond(
        tris,
        body_center,
        body_scale,
        body_scale * 0.72,
        body_scale * 1.08,
        body_color,
        EntityTag::Fly(tag_idx),
    );
    let abdomen_center = body_center.sub(Vec3::new(0.0, 0.02, 0.32 * visual.abdomen_scale));
    push_diamond(
        tris,
        abdomen_center,
        body_scale * 0.78 * visual.abdomen_scale,
        body_scale * 0.60 * visual.abdomen_scale,
        body_scale * 1.14 * visual.abdomen_scale,
        body_color,
        EntityTag::Fly(tag_idx),
    );
    let head_center = body_center.add(Vec3::new(0.0, 0.02, 0.38 * visual.head_scale));
    push_diamond(
        tris,
        head_center,
        body_scale * 0.62 * visual.head_scale,
        body_scale * 0.48 * visual.head_scale,
        body_scale * 0.68 * visual.head_scale,
        body_color,
        EntityTag::Fly(tag_idx),
    );

    let forward = Vec3::new(body.heading.sin(), 0.0, body.heading.cos()).normalize();
    let right = Vec3::new(forward.z, 0.0, -forward.x);
    let wing_root_l = body_center
        .add(right.mul(0.12 * visual.sprite_scale * visual.leg_span))
        .add(Vec3::new(0.0, 0.08 * visual.sprite_scale, 0.0));
    let wing_root_r = body_center
        .sub(right.mul(0.12 * visual.sprite_scale * visual.leg_span))
        .add(Vec3::new(0.0, 0.08 * visual.sprite_scale, 0.0));
    let wing_lift = if body.is_flying {
        0.16 + visual.wing_angle.abs() * 0.18
    } else {
        0.05 + visual.wing_angle.abs() * 0.05
    };
    let wing_span = 0.38 * visual.sprite_scale * visual.wing_span;
    let wing_sweep = forward.mul(visual.wing_angle * 0.10);
    let wing_l_a = wing_root_l.add(right.mul(wing_span)).add(wing_sweep);
    let wing_l_b = wing_root_l
        .add(right.mul(wing_span * (0.45 + visual.wing_width * 0.30)))
        .sub(wing_sweep)
        .add(Vec3::new(0.0, wing_lift, 0.0));
    let wing_r_a = wing_root_r.sub(right.mul(wing_span)).add(wing_sweep);
    let wing_r_b = wing_root_r
        .sub(right.mul(wing_span * (0.45 + visual.wing_width * 0.30)))
        .sub(wing_sweep)
        .add(Vec3::new(0.0, wing_lift, 0.0));
    for (a, b, c) in [
        (wing_root_l, wing_l_a, wing_l_b),
        (wing_root_r, wing_r_a, wing_r_b),
    ] {
        tris.push(Triangle {
            v: [
                Vertex {
                    pos: a,
                    normal: Vec3::new(0.0, 1.0, 0.0),
                    color: wing_color,
                },
                Vertex {
                    pos: b,
                    normal: Vec3::new(0.0, 1.0, 0.0),
                    color: wing_color,
                },
                Vertex {
                    pos: c,
                    normal: Vec3::new(0.0, 1.0, 0.0),
                    color: wing_color,
                },
            ],
            tag: EntityTag::Fly(tag_idx),
        });
    }

    if visual.proboscis_extension > 0.0 {
        push_diamond(
            tris,
            head_center.add(forward.mul(0.18 + visual.proboscis_extension * 0.24)),
            0.02,
            0.02,
            0.10 + visual.proboscis_extension * 0.12,
            body_color,
            EntityTag::Fly(tag_idx),
        );
    }
}

#[derive(Default)]
struct True3DRenderGroup {
    fingerprint: u64,
    tris: Vec<Triangle>,
}

#[derive(Default)]
struct True3DRenderSceneCache {
    groups: HashMap<u64, True3DRenderGroup>,
    active_render_ids: Vec<u64>,
    active_lookup: HashSet<u64>,
}

impl True3DRenderSceneCache {
    fn begin_frame(&mut self) {
        self.active_render_ids.clear();
        self.active_lookup.clear();
    }

    fn mark_active(&mut self, render_id: u64) -> bool {
        if self.active_lookup.insert(render_id) {
            self.active_render_ids.push(render_id);
        }
        self.groups.contains_key(&render_id)
    }

    fn mark_active_if_fingerprint_matches(&mut self, render_id: u64, fingerprint: u64) -> bool {
        if !self.mark_active(render_id) {
            return false;
        }
        self.groups
            .get(&render_id)
            .map(|group| group.fingerprint == fingerprint && !group.tris.is_empty())
            .unwrap_or(false)
    }

    fn reuse_or_rebuild<F>(&mut self, render_id: u64, fingerprint: u64, build: F)
    where
        F: FnOnce(&mut Vec<Triangle>),
    {
        self.mark_active(render_id);

        let group = self.groups.entry(render_id).or_default();
        if group.fingerprint != fingerprint || group.tris.is_empty() {
            group.tris.clear();
            build(&mut group.tris);
            group.fingerprint = fingerprint;
        }
    }

    fn finish_frame(&mut self) {
        let active = &self.active_lookup;
        self.groups
            .retain(|render_id, _| active.contains(render_id));
    }

    fn active_triangles(&self) -> impl Iterator<Item = &Triangle> {
        self.active_render_ids
            .iter()
            .filter_map(|render_id| self.groups.get(render_id))
            .flat_map(|group| group.tris.iter())
    }
}

fn render_true3d(
    buf: &mut ScreenBuffer,
    world: &oneura_core::terrarium::TerrariumWorld,
    snapshot: &oneura_core::terrarium::TerrariumWorldSnapshot,
    frame_idx: usize,
    vs: &ViewState,
    scratch: &mut RenderScratch,
) {
    // We use a higher resolution buffer for rasterization and then map to half-blocks.
    // Half-blocks give 2x vertical resolution.
    let w = buf.width;
    let h_low = buf.height;
    let h = h_low * 2;

    // Camera setup
    let angle = (frame_idx as f32 * 0.02) + (vs.camera_x as f32 * 0.05);
    let cam_dist = 40.0 / vs.zoom;
    let cam_height = 25.0 - (vs.camera_y as f32 * 0.5);
    let eye = Vec3::new(angle.cos() * cam_dist, cam_height, angle.sin() * cam_dist);
    let center = Vec3::new(0.0, 0.0, 0.0);
    let up = Vec3::new(0.0, 1.0, 0.0);

    let view = look_at(eye, center, up);
    let proj = perspective(1.0, w as f32 / h as f32, 0.1, 100.0);
    let mvp = proj.multiply(&view);

    // 1. Terrain
    let gw = world.config.width;
    let gh = world.config.height;
    let moisture = world.moisture_field();
    let atmosphere = world.atmosphere_frame();
    let surface_relief = world.topdown_field(TerrariumTopdownView::Terrain);
    let cell_size = 1.0;
    let ox = -(gw as f32 * cell_size) / 2.0;
    let oz = -(gh as f32 * cell_size) / 2.0;
    let mean_air = mean_visual_air_sample(&atmosphere);
    let mean_air_state_key = true3d_mean_air_state_key(mean_air);
    let anim_time = true3d_animation_time(world.time_s);
    let fruit_visuals = world.fruit_visuals_at_time(anim_time);
    let seed_visuals = world.seed_visuals_at_time(anim_time);
    let earthworms = world.earthworm_visual_markers_at_time(anim_time);
    let nematodes = world.nematode_visual_markers_at_time(anim_time);
    let soil_surface = world.soil_surface_markers();
    let water_cycle = TerrariumWaterCycleInputs {
        lunar_phase: snapshot.lunar_phase,
        moonlight: snapshot.moonlight,
        tidal_moisture_factor: snapshot.tidal_moisture_factor,
    };
    let (
        terrain_scene,
        dynamic_scene,
        terrain_cell_state,
        water_tile_state,
        water_source_state,
        plant_state,
        tris,
        z_buf,
        color_buf,
    ) = scratch.true3d_working_set(
        w,
        h_low,
        gw * gh,
        world.waters.len(),
        snapshot.full_plants.len(),
    );
    let mut ground_heights = vec![0.0f32; gw * gh];
    for (idx, ground_y) in ground_heights.iter_mut().enumerate() {
        *ground_y = surface_relief
            .get(idx)
            .copied()
            .map(quantized_surface_height)
            .unwrap_or(0.0);
    }

    // 1. Terrain cells and soil-surface hotspots.
    for gy in 0..gh {
        for gx in 0..gw {
            let idx = gy * gw + gx;
            let ground_y = ground_heights[idx];
            let moisture_t = moisture.get(idx).copied().unwrap_or(0.0).clamp(0.0, 1.0);
            let center_x = ox + gx as f32 * cell_size;
            let center_z = oz + gy as f32 * cell_size;
            let render_id = true3d_render_id(TRUE3D_RENDER_KIND_TERRAIN_CELL, idx);
            let state_key = terrain_cell_true3d_state_key(
                ground_y,
                moisture_t,
                mean_air_state_key,
                water_cycle,
            );
            if terrain_cell_state[idx] == state_key && terrain_scene.mark_active(render_id) {
                continue;
            }
            let fingerprint = terrain_cell_true3d_fingerprint(
                ground_y,
                moisture_t,
                mean_air,
                water_cycle,
                center_x,
                center_z,
                cell_size,
                idx,
            );
            terrain_scene.reuse_or_rebuild(render_id, fingerprint, |group_tris| {
                push_true3d_terrain_cell(
                    group_tris,
                    ground_y,
                    moisture_t,
                    mean_air,
                    water_cycle,
                    center_x,
                    center_z,
                    cell_size,
                    idx,
                );
            });
            terrain_cell_state[idx] = state_key;
        }
    }
    for &(gx, gy, visual) in &soil_surface {
        let idx = gy * gw + gx;
        let ground_y = ground_heights[idx];
        let render_id = true3d_render_id(TRUE3D_RENDER_KIND_SOIL_SURFACE, idx);
        let fingerprint =
            soil_surface_true3d_fingerprint(gx, gy, visual, ground_y, ox, oz, cell_size);
        terrain_scene.reuse_or_rebuild(render_id, fingerprint, |group_tris| {
            push_soil_surface_true3d(group_tris, gx, gy, visual, ground_y, ox, oz, cell_size);
        });
    }
    terrain_scene.finish_frame();

    // 2. Plants
    for (i, plant) in snapshot.full_plants.iter().enumerate() {
        let local_air = sample_local_atmosphere(
            &atmosphere,
            gw,
            gh,
            plant.x as f32 + 0.5,
            plant.y as f32 + 0.5,
        );
        let local_air_state_key = true3d_plant_air_state_key(local_air);
        let (vitality, canopy_density) = plant_metrics(world, i);
        let ground_y = ground_heights[plant.y * gw + plant.x];
        let render_id = true3d_render_id(TRUE3D_RENDER_KIND_PLANT, i);
        let state_key = plant_true3d_state_key(
            plant,
            ground_y,
            local_air_state_key,
            vitality,
            canopy_density,
            anim_time,
        );
        if plant_state[i] == state_key && dynamic_scene.mark_active(render_id) {
            continue;
        }
        let visual = plant_visual_response(
            plant.taxonomy_id,
            local_air,
            vitality,
            canopy_density,
            anim_time,
            i as f32,
        );
        let fingerprint = plant_true3d_fingerprint(plant, visual, ground_y, ox, oz, cell_size);
        dynamic_scene.reuse_or_rebuild(render_id, fingerprint, |group_tris| {
            push_plant_true3d(group_tris, plant, visual, ground_y, ox, oz, cell_size, i);
        });
        plant_state[i] = state_key;
    }

    // 3. Flies
    for (i, fly) in world.flies.iter().enumerate() {
        let b = fly.body_state();
        let visual = sample_fly_visual(world, &atmosphere, gw, gh, i, &b, anim_time);
        let render_id = true3d_render_id(TRUE3D_RENDER_KIND_FLY, i);
        let fingerprint = fly_true3d_fingerprint(&b, visual, ox, oz, cell_size);
        dynamic_scene.reuse_or_rebuild(render_id, fingerprint, |group_tris| {
            push_fly_true3d(group_tris, &b, visual, ox, oz, cell_size, i);
        });
    }

    // 4. Water bodies and soil-surface chemistry markers.
    for gy in 0..gh {
        for gx in 0..gw {
            let idx = gy * gw + gx;
            let water_t: f32 = ((moisture[idx] - 0.82) / 0.18).clamp(0.0, 1.0);
            if water_t <= 0.08 {
                continue;
            }
            let local_air =
                sample_local_atmosphere(&atmosphere, gw, gh, gx as f32 + 0.5, gy as f32 + 0.5);
            let volume = water_t * 120.0;
            let render_id = true3d_render_id(TRUE3D_RENDER_KIND_WATER_TILE, idx);
            let state_key =
                water_true3d_state_key(local_air, volume, anim_time, idx as f32, water_cycle);
            if water_tile_state[idx] == state_key && dynamic_scene.mark_active(render_id) {
                continue;
            }
            let visual =
                water_visual_response(local_air, volume, anim_time, idx as f32, water_cycle);
            let water_height: f32 = (0.22_f32 + water_t * 0.34 + water_t * water_t * 0.12)
                .clamp(0.18, 0.62)
                * visual.thickness_scale.clamp(0.82, 1.12);
            let surface_center = Vec3::new(
                ox + gx as f32 * cell_size,
                ground_heights[idx] + 0.02 + water_height * 0.5 + visual.vertical_offset * 2.0,
                oz + gy as f32 * cell_size,
            );
            let half_x = (visual.radius_x_cells * 0.50).clamp(0.28, 0.88);
            let half_z = (visual.radius_y_cells * 0.50).clamp(0.28, 0.88);
            let color = pixel_rgb(visual.rgb, TRUE3D_STATIC_COLOR_LEVELS);
            let mut hasher = DefaultHasher::new();
            hash_true3d_vec3(surface_center, &mut hasher);
            hash_true3d_scalar(half_x, &mut hasher);
            hash_true3d_scalar(half_z, &mut hasher);
            hash_true3d_rgb(color, &mut hasher);
            dynamic_scene.reuse_or_rebuild(render_id, hasher.finish(), |group_tris| {
                push_quad_xz(
                    group_tris,
                    surface_center,
                    half_x,
                    half_z,
                    color,
                    EntityTag::Terrain,
                );
            });
            water_tile_state[idx] = state_key;
        }
    }

    for (i, water) in world.waters.iter().enumerate() {
        if !water.alive {
            continue;
        }
        let render_id = true3d_render_id(TRUE3D_RENDER_KIND_WATER_SOURCE, i);
        let wx = water.x.min(gw.saturating_sub(1));
        let wy = water.y.min(gh.saturating_sub(1));
        if moisture[wy * gw + wx] > 0.94 {
            water_source_state[i] = TRUE3D_STATE_MISSING;
            continue;
        }
        let ground_y = ground_heights[wy * gw + wx];
        let local_air = sample_local_atmosphere(
            &atmosphere,
            gw,
            gh,
            water.x as f32 + 0.5,
            water.y as f32 + 0.5,
        );
        let state_key = water_source_true3d_state_key(
            local_air,
            water.volume,
            anim_time,
            i as f32,
            water_cycle,
            wx,
            wy,
            ground_y,
        );
        if water_source_state[i] == state_key && dynamic_scene.mark_active(render_id) {
            continue;
        }
        let visual =
            water_visual_response(local_air, water.volume, anim_time, i as f32, water_cycle);
        let surface_center = Vec3::new(
            ox + wx as f32 * cell_size,
            ground_y + 0.10 + visual.vertical_offset * 4.0,
            oz + wy as f32 * cell_size,
        );
        let half_x = (visual.radius_x_cells * 0.55).clamp(0.24, 0.85);
        let half_z = (visual.radius_y_cells * 0.55).clamp(0.24, 0.85);
        let color = pixel_rgb(visual.rgb, TRUE3D_DYNAMIC_COLOR_LEVELS);
        let mut hasher = DefaultHasher::new();
        hash_true3d_vec3(surface_center, &mut hasher);
        hash_true3d_scalar(half_x, &mut hasher);
        hash_true3d_scalar(half_z, &mut hasher);
        hash_true3d_rgb(color, &mut hasher);
        dynamic_scene.reuse_or_rebuild(render_id, hasher.finish(), |group_tris| {
            push_quad_xz(
                group_tris,
                surface_center,
                half_x,
                half_z,
                color,
                EntityTag::Water(i),
            );
        });
        water_source_state[i] = state_key;
    }

    // 5. Ground fruit and seed layer.
    for (i, fruit) in world.fruits.iter().enumerate() {
        if !fruit.source.alive || fruit.source.sugar_content <= 0.01 {
            continue;
        }
        let visual = fruit_visuals.get(i).copied().unwrap_or_default();
        let fx = fruit.source.x.min(gw.saturating_sub(1));
        let fy = fruit.source.y.min(gh.saturating_sub(1));
        let center = Vec3::new(
            ox + fx as f32 * cell_size,
            ground_heights[fy * gw + fx] + 0.12 + visual.vertical_offset * 5.0,
            oz + fy as f32 * cell_size,
        );
        let radius = (visual.sprite_scale * 0.18).clamp(0.10, 0.28);
        let skin_color = pixel_rgb(visual.skin_rgb, TRUE3D_DYNAMIC_COLOR_LEVELS);
        let stem_color = pixel_rgb(visual.stem_rgb, TRUE3D_DYNAMIC_COLOR_LEVELS);
        let stem_center = center.add(Vec3::new(0.0, radius * 1.15, 0.0));
        let mut hasher = DefaultHasher::new();
        hash_true3d_vec3(center, &mut hasher);
        hash_true3d_vec3(stem_center, &mut hasher);
        hash_true3d_scalar(radius, &mut hasher);
        hash_true3d_rgb(skin_color, &mut hasher);
        hash_true3d_rgb(stem_color, &mut hasher);
        let fingerprint = hasher.finish();
        let render_id = true3d_render_id(TRUE3D_RENDER_KIND_FRUIT, i);
        if dynamic_scene.mark_active_if_fingerprint_matches(render_id, fingerprint) {
            continue;
        }
        dynamic_scene.reuse_or_rebuild(render_id, fingerprint, |group_tris| {
            push_diamond(
                group_tris,
                center,
                radius,
                radius * 0.92,
                radius,
                skin_color,
                EntityTag::Fruit(i),
            );
            push_diamond(
                group_tris,
                stem_center,
                radius * 0.16,
                radius * 0.22,
                radius * 0.16,
                stem_color,
                EntityTag::Fruit(i),
            );
        });
    }

    for (i, seed) in world.seeds.iter().enumerate() {
        let visual = seed_visuals.get(i).copied().unwrap_or_default();
        let sx = seed.x.round().clamp(0.0, gw.saturating_sub(1) as f32) as usize;
        let sy = seed.y.round().clamp(0.0, gh.saturating_sub(1) as f32) as usize;
        let center = Vec3::new(
            ox + sx as f32 * cell_size,
            ground_heights[sy * gw + sx] + 0.08 + visual.vertical_offset * 4.0,
            oz + sy as f32 * cell_size,
        );
        let radius = (visual.sprite_scale * 0.11).clamp(0.05, 0.14);
        let shell_color = pixel_rgb(visual.shell_rgb, TRUE3D_DYNAMIC_COLOR_LEVELS);
        let accent_color = pixel_rgb(visual.accent_rgb, TRUE3D_DYNAMIC_COLOR_LEVELS);
        let accent_center = center.add(Vec3::new(radius * 0.5, radius * 0.35, 0.0));
        let mut hasher = DefaultHasher::new();
        hash_true3d_vec3(center, &mut hasher);
        hash_true3d_vec3(accent_center, &mut hasher);
        hash_true3d_scalar(radius, &mut hasher);
        hash_true3d_rgb(shell_color, &mut hasher);
        hash_true3d_rgb(accent_color, &mut hasher);
        let fingerprint = hasher.finish();
        let render_id = true3d_render_id(TRUE3D_RENDER_KIND_SEED, i);
        if dynamic_scene.mark_active_if_fingerprint_matches(render_id, fingerprint) {
            continue;
        }
        dynamic_scene.reuse_or_rebuild(render_id, fingerprint, |group_tris| {
            push_diamond(
                group_tris,
                center,
                radius,
                radius * 0.72,
                radius * 1.1,
                shell_color,
                EntityTag::Terrain,
            );
            push_diamond(
                group_tris,
                accent_center,
                radius * 0.22,
                radius * 0.14,
                radius * 0.30,
                accent_color,
                EntityTag::Terrain,
            );
        });
    }

    // 6. Soil fauna projections.
    for (i, &(gx, gy, visual)) in earthworms.iter().enumerate() {
        let mut hasher = DefaultHasher::new();
        gx.hash(&mut hasher);
        gy.hash(&mut hasher);
        (visual.segment_count.max(4) as usize).hash(&mut hasher);
        hash_true3d_scalar(visual.length_scale, &mut hasher);
        hash_true3d_scalar(visual.thickness_scale, &mut hasher);
        hash_true3d_scalar(visual.curl, &mut hasher);
        hash_true3d_scalar(visual.activity, &mut hasher);
        hash_true3d_scalar(visual.height_offset, &mut hasher);
        hash_true3d_scalar(visual.yaw_rad, &mut hasher);
        hash_true3d_rgb(
            pixel_rgb(visual.body_rgb, TRUE3D_DYNAMIC_COLOR_LEVELS),
            &mut hasher,
        );
        hash_true3d_rgb(
            pixel_rgb(visual.clitellum_rgb, TRUE3D_DYNAMIC_COLOR_LEVELS),
            &mut hasher,
        );
        let fingerprint = hasher.finish();
        let render_id = true3d_render_id(TRUE3D_RENDER_KIND_EARTHWORM, i);
        if dynamic_scene.mark_active_if_fingerprint_matches(render_id, fingerprint) {
            continue;
        }
        dynamic_scene.reuse_or_rebuild(render_id, fingerprint, |group_tris| {
            push_earthworm_true3d(
                group_tris, ox, oz, cell_size, gx as f32, gy as f32, visual, i,
            );
        });
    }
    for (i, &(gx, gy, visual)) in nematodes.iter().enumerate() {
        let mut hasher = DefaultHasher::new();
        gx.hash(&mut hasher);
        gy.hash(&mut hasher);
        hash_true3d_scalar(visual.length_scale, &mut hasher);
        hash_true3d_scalar(visual.thickness_scale, &mut hasher);
        hash_true3d_scalar(visual.curl, &mut hasher);
        hash_true3d_scalar(visual.height_offset, &mut hasher);
        hash_true3d_scalar(visual.yaw_rad, &mut hasher);
        hash_true3d_scalar(visual.stylet_length_scale, &mut hasher);
        hash_true3d_rgb(
            pixel_rgb(visual.body_rgb, TRUE3D_DYNAMIC_COLOR_LEVELS),
            &mut hasher,
        );
        hash_true3d_rgb(
            pixel_rgb(visual.head_rgb, TRUE3D_DYNAMIC_COLOR_LEVELS),
            &mut hasher,
        );
        let fingerprint = hasher.finish();
        let render_id = true3d_render_id(TRUE3D_RENDER_KIND_NEMATODE, i);
        if dynamic_scene.mark_active_if_fingerprint_matches(render_id, fingerprint) {
            continue;
        }
        dynamic_scene.reuse_or_rebuild(render_id, fingerprint, |group_tris| {
            push_nematode_true3d(
                group_tris,
                ox,
                oz,
                cell_size,
                gx as f32,
                gy as f32,
                visual,
                i + earthworms.len(),
            );
        });
    }

    dynamic_scene.finish_frame();

    // Rasterization
    let light_dir = Vec3::new(0.5, 1.0, 0.3).normalize();

    for tri in terrain_scene
        .active_triangles()
        .chain(dynamic_scene.active_triangles())
        .chain(tris.iter())
    {
        let lit_color = lit_triangle_color(tri, light_dir);
        let mut clip = [[0.0f32; 4]; 3];
        let mut screen = [[0.0f32; 3]; 3];
        let mut culled = false;
        for i in 0..3 {
            clip[i] = mvp.transform(tri.v[i].pos);
            if clip[i][3] < 0.1 {
                culled = true;
                break;
            }
            let inv_w = 1.0 / clip[i][3];
            screen[i][0] = (clip[i][0] * inv_w * 0.5 + 0.5) * w as f32;
            screen[i][1] = (1.0 - (clip[i][1] * inv_w * 0.5 + 0.5)) * h as f32;
            screen[i][2] = clip[i][2] * inv_w;
        }
        if culled {
            continue;
        }

        // Bounding box
        let min_x = screen
            .iter()
            .map(|s| s[0])
            .fold(f32::MAX, f32::min)
            .max(0.0) as usize;
        let max_x = screen
            .iter()
            .map(|s| s[0])
            .fold(f32::MIN, f32::max)
            .min(w as f32 - 1.0) as usize;
        let min_y = screen
            .iter()
            .map(|s| s[1])
            .fold(f32::MAX, f32::min)
            .max(0.0) as usize;
        let max_y = screen
            .iter()
            .map(|s| s[1])
            .fold(f32::MIN, f32::max)
            .min(h as f32 - 1.0) as usize;

        let edge = |a: [f32; 3], b: [f32; 3], c: [f32; 2]| -> f32 {
            (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])
        };

        let area = edge(screen[0], screen[1], [screen[2][0], screen[2][1]]);
        if area.abs() < 1e-6 {
            continue;
        }
        let inv_area = 1.0 / area;

        for py in min_y..=max_y {
            for px in min_x..=max_x {
                let p = [px as f32 + 0.5, py as f32 + 0.5];
                let w0 = edge(screen[1], screen[2], p) * inv_area;
                let w1 = edge(screen[2], screen[0], p) * inv_area;
                let w2 = edge(screen[0], screen[1], p) * inv_area;

                if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                    let z = w0 * screen[0][2] + w1 * screen[1][2] + w2 * screen[2][2];
                    let idx = py * w + px;
                    if z < z_buf[idx] {
                        z_buf[idx] = z;
                        color_buf[idx] = lit_color;
                    }
                }
            }
        }
    }

    // Map to half-blocks in ScreenBuffer
    buf.clear();
    for y_low in 0..h_low {
        for x in 0..w {
            let top_color = color_buf[(y_low * 2) * w + x];
            let bot_color = color_buf[(y_low * 2 + 1) * w + x];
            // '▀' uses fg for top half, bg for bottom half
            buf.set(x, y_low, '▀', top_color, bot_color);
        }
    }

    // Header
    let header = format!(
        " {} | True 3D CLI | Frame:{} | {} | P:{} Fl:{} Fr:{} Sd:{} W:{} Ew:{} Nm:{} | {:.0}Hz",
        vs.preset.label(),
        frame_idx,
        world.time_label(),
        snapshot.plants,
        snapshot.flies,
        snapshot.fruits,
        snapshot.seeds,
        world.waters.iter().filter(|water| water.alive).count(),
        earthworms.len(),
        nematodes.len(),
        snapshot.avg_wing_beat_freq
    );
    buf.write_str(
        0,
        0,
        &header[..header.len().min(buf.width)],
        (0, 255, 255),
        (20, 20, 40),
    );

    if vs.show_info {
        let species = plant_species_line(snapshot);
        let models = species_model_line(snapshot);
        buf.write_str(
            0,
            1,
            &species[..species.len().min(buf.width)],
            (170, 235, 180),
            (12, 22, 18),
        );
        buf.write_str(
            0,
            2,
            &models[..models.len().min(buf.width)],
            (190, 215, 240),
            (12, 18, 28),
        );
    }
}

/// Map moisture to soil color (brown gradient).
fn soil_color(moisture: f32, organic: f32) -> (u8, u8, u8) {
    let dry = (180, 140, 80); // sandy
    let wet = (80, 60, 30); // dark earth
    let rich = (60, 90, 40); // organic-rich
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

type LocalAtmosphereSample = TerrariumVisualAirSample;

fn sample_local_atmosphere(
    atmosphere: &TerrariumAtmosphereFrame,
    width: usize,
    height: usize,
    x: f32,
    y: f32,
) -> LocalAtmosphereSample {
    sample_visual_air(atmosphere, width, height, x, y)
}

fn mean_field(field: &[f32], fallback: f32) -> f32 {
    mean_visual_field(field, fallback)
}

fn mean_wind_speed(atmosphere: &TerrariumAtmosphereFrame) -> f32 {
    mean_visual_wind_speed(atmosphere)
}

fn soil_color_with_atmosphere(
    moisture: f32,
    organic: f32,
    local_air: LocalAtmosphereSample,
) -> (u8, u8, u8) {
    rgb_to_u8(terrain_visual_response(moisture, organic, local_air).rgb)
}

#[allow(dead_code)]
fn plant_color_with_atmosphere(
    taxonomy_id: u32,
    canopy: f32,
    vitality: f32,
    local_air: LocalAtmosphereSample,
) -> (u8, u8, u8) {
    rgb_to_u8(plant_visual_response(taxonomy_id, local_air, vitality, canopy, 0.0, 0.0).leaf_rgb)
}

fn water_color(local_air: LocalAtmosphereSample) -> (u8, u8, u8) {
    rgb_to_u8(
        water_visual_response(
            local_air,
            1.0,
            0.0,
            0.0,
            TerrariumWaterCycleInputs {
                lunar_phase: 0.0,
                moonlight: 0.0,
                tidal_moisture_factor: 1.0,
            },
        )
        .rgb,
    )
}

fn fly_energy_fraction(world: &TerrariumWorld, idx: usize, body: &BodyState) -> f32 {
    world
        .fly_metabolisms
        .get(idx)
        .map(|m| m.energy_charge())
        .unwrap_or((body.energy / 100.0).clamp(0.0, 1.0))
}

fn sample_fly_visual(
    world: &TerrariumWorld,
    atmosphere: &TerrariumAtmosphereFrame,
    gw: usize,
    gh: usize,
    idx: usize,
    body: &BodyState,
    time_s: f32,
) -> oneura_core::terrarium::visual_projection::TerrariumFlyVisualResponse {
    let local_air = sample_local_atmosphere(atmosphere, gw, gh, body.x, body.y);
    fly_visual_response(
        local_air,
        body,
        fly_energy_fraction(world, idx, body),
        time_s,
        idx as f32,
    )
}

fn heading_octant(heading: f32) -> usize {
    let tau = std::f32::consts::TAU;
    let turn = heading.rem_euclid(tau) / tau;
    ((turn * 8.0).round() as isize).rem_euclid(8) as usize
}

fn heading_cell_offset(heading: f32) -> (isize, isize) {
    match heading_octant(heading) {
        0 => (1, 0),
        1 => (1, -1),
        2 => (0, -1),
        3 => (-1, -1),
        4 => (-1, 0),
        5 => (-1, 1),
        6 => (0, 1),
        _ => (1, 1),
    }
}

fn fly_body_glyph(body: &BodyState, wing_angle: f32, proboscis_extension: f32) -> char {
    if proboscis_extension > 0.08 {
        return '\u{271B}';
    }
    if body.is_flying {
        if wing_angle.abs() > 0.45 {
            '\u{2736}'
        } else {
            '\u{2734}'
        }
    } else {
        match heading_octant(body.heading) {
            0 | 4 => '\u{25C6}',
            1 | 5 => '\u{25E2}',
            2 | 6 => '\u{25C7}',
            _ => '\u{25E3}',
        }
    }
}

fn fly_wing_glyph(heading: f32) -> char {
    match heading_octant(heading) {
        0 | 4 => '\u{2500}',
        2 | 6 => '\u{2502}',
        1 | 5 => '\u{2571}',
        _ => '\u{2572}',
    }
}

fn earthworm_glyph(visual: TerrariumEarthwormVisualResponse) -> char {
    if visual.curl.abs() > 0.22 {
        '\u{223F}'
    } else {
        '~'
    }
}

fn nematode_glyph(visual: TerrariumNematodeVisualResponse) -> char {
    if visual.stylet_length_scale > 0.05 {
        '>'
    } else {
        '-'
    }
}

fn draw_isometric_fauna_markers(
    buf: &mut ScreenBuffer,
    ox: isize,
    oy: isize,
    cell_w: isize,
    cell_h_half: isize,
    earthworms: &[(usize, usize, TerrariumEarthwormVisualResponse)],
    nematodes: &[(usize, usize, TerrariumNematodeVisualResponse)],
) {
    for (gx, gy, visual) in earthworms {
        let iso_x = (*gx as isize - *gy as isize) * cell_w + ox;
        let iso_y = (*gx as isize + *gy as isize) * cell_h_half + oy
            - (visual.height_offset * 8.0).round() as isize;
        let segments = (visual.segment_count.max(4) as isize / 2).max(3);
        for seg in 0..segments {
            let t = if segments > 1 {
                seg as f32 / (segments - 1) as f32
            } else {
                0.0
            };
            let px = iso_x + ((t - 0.5) * visual.length_scale * 3.0).round() as isize;
            let py =
                iso_y - (visual.curl * (std::f32::consts::PI * t).sin() * 2.0).round() as isize;
            let color = if (0.38..=0.58).contains(&t) {
                rgb_to_u8(visual.clitellum_rgb)
            } else {
                rgb_to_u8(visual.body_rgb)
            };
            set_if_visible(
                buf,
                px + 1,
                py,
                earthworm_glyph(*visual),
                color,
                (10, 10, 20),
            );
        }
    }

    for (gx, gy, visual) in nematodes {
        let iso_x = (*gx as isize - *gy as isize) * cell_w + ox;
        let iso_y = (*gx as isize + *gy as isize) * cell_h_half + oy
            - (visual.height_offset * 6.0).round() as isize;
        let segments = 4isize;
        for seg in 0..segments {
            let t = if segments > 1 {
                seg as f32 / (segments - 1) as f32
            } else {
                0.0
            };
            let px = iso_x + ((t - 0.5) * visual.length_scale * 2.0).round() as isize;
            let py =
                iso_y - (visual.curl * (std::f32::consts::PI * t).sin() * 1.5).round() as isize;
            let color = if seg == segments - 1 {
                rgb_to_u8(visual.head_rgb)
            } else {
                rgb_to_u8(visual.body_rgb)
            };
            set_if_visible(
                buf,
                px + 1,
                py,
                nematode_glyph(*visual),
                color,
                (10, 10, 20),
            );
        }
    }
}

fn draw_topdown_fauna_markers(
    buf: &mut ScreenBuffer,
    ox: isize,
    oy: isize,
    scale: usize,
    earthworms: &[(usize, usize, TerrariumEarthwormVisualResponse)],
    nematodes: &[(usize, usize, TerrariumNematodeVisualResponse)],
) {
    for (gx, gy, visual) in earthworms {
        let px = ox + (*gx * scale) as isize;
        let py = oy + *gy as isize - (visual.height_offset * 4.0).round() as isize;
        let segments = (visual.segment_count.max(4) as isize / 2).max(3);
        for seg in 0..segments {
            let t = if segments > 1 {
                seg as f32 / (segments - 1) as f32
            } else {
                0.0
            };
            let dx = ((t - 0.5) * visual.length_scale * 2.0).round() as isize;
            let dy = (visual.curl * (std::f32::consts::PI * t).sin() * 1.4).round() as isize;
            let color = if (0.38..=0.58).contains(&t) {
                rgb_to_u8(visual.clitellum_rgb)
            } else {
                rgb_to_u8(visual.body_rgb)
            };
            set_if_visible(
                buf,
                px + dx,
                py + dy,
                earthworm_glyph(*visual),
                color,
                (10, 10, 20),
            );
        }
    }

    for (gx, gy, visual) in nematodes {
        let px = ox + (*gx * scale) as isize;
        let py = oy + *gy as isize - (visual.height_offset * 3.0).round() as isize;
        let segments = 4isize;
        for seg in 0..segments {
            let t = if segments > 1 {
                seg as f32 / (segments - 1) as f32
            } else {
                0.0
            };
            let dx = ((t - 0.5) * visual.length_scale * 1.5).round() as isize;
            let dy = (visual.curl * (std::f32::consts::PI * t).sin()).round() as isize;
            let color = if seg == segments - 1 {
                rgb_to_u8(visual.head_rgb)
            } else {
                rgb_to_u8(visual.body_rgb)
            };
            set_if_visible(
                buf,
                px + dx,
                py + dy,
                nematode_glyph(*visual),
                color,
                (10, 10, 20),
            );
        }
    }
}

#[allow(dead_code)]
fn wind_lean_offset(local_air: LocalAtmosphereSample, frame_idx: usize, max_lean: f32) -> isize {
    let gust_phase = frame_idx as f32 * (0.12 + local_air.wind_speed * 0.35)
        + local_air.pressure_delta_kpa * 5.0;
    let gust = gust_phase.sin() * 0.65 + (gust_phase * 0.61).cos() * 0.35;
    (local_air.wind_x * max_lean + gust * local_air.wind_speed * max_lean)
        .round()
        .clamp(-max_lean, max_lean) as isize
}

#[allow(dead_code)]
fn plant_canopy_glyph(cells: f32, local_air: LocalAtmosphereSample, frame_idx: usize) -> char {
    if local_air.wind_speed > 0.05 {
        if (frame_idx / 2) % 2 == 0 {
            if local_air.wind_x >= 0.0 {
                '\u{2571}'
            } else {
                '\u{2572}'
            }
        } else if local_air.wind_y >= 0.0 {
            '\u{2572}'
        } else {
            '\u{2571}'
        }
    } else if cells > 150.0 {
        '\u{2663}'
    } else if cells > 80.0 {
        '\u{273F}'
    } else {
        '\u{2731}'
    }
}

#[allow(dead_code)]
fn plant_topdown_glyph(cells: f32, local_air: LocalAtmosphereSample, frame_idx: usize) -> char {
    if local_air.wind_speed > 0.05 {
        if (frame_idx / 2) % 2 == 0 {
            if local_air.wind_x >= 0.0 {
                '/'
            } else {
                '\\'
            }
        } else if local_air.wind_y >= 0.0 {
            '\\'
        } else {
            '/'
        }
    } else if cells > 50.0 {
        '\u{2663}'
    } else {
        '\u{2022}'
    }
}

fn water_surface_glyph(local_air: LocalAtmosphereSample, frame_idx: usize) -> char {
    if local_air.wind_speed > 0.05 {
        if (frame_idx / 2) % 2 == 0 {
            if local_air.wind_x.abs() >= local_air.wind_y.abs() {
                '~'
            } else {
                '\u{2248}'
            }
        } else if local_air.wind_x >= 0.0 {
            '\u{2571}'
        } else {
            '\u{2572}'
        }
    } else if frame_idx % 4 < 2 {
        '\u{2248}'
    } else {
        '~'
    }
}

fn set_if_visible(
    buf: &mut ScreenBuffer,
    x: isize,
    y: isize,
    ch: char,
    fg_c: (u8, u8, u8),
    bg_c: (u8, u8, u8),
) {
    if x >= 0 && y >= 0 {
        buf.set(x as usize, y as usize, ch, fg_c, bg_c);
    }
}

fn plant_metrics(world: &TerrariumWorld, idx: usize) -> (f32, f32) {
    world
        .plants
        .get(idx)
        .map(|plant| {
            (
                plant.cellular.vitality(),
                (plant.cellular.total_cells() * 0.01).clamp(0.0, 1.0),
            )
        })
        .unwrap_or((0.75, 0.45))
}

fn plant_node_char(node_type: NodeType) -> char {
    match node_type {
        NodeType::Trunk | NodeType::Branch | NodeType::Bud => '\u{2588}',
        NodeType::Leaf => '\u{2593}',
        NodeType::Fruit => '\u{25CF}',
    }
}

fn plant_node_color(
    node_type: NodeType,
    visual: oneura_core::terrarium::visual_projection::TerrariumPlantVisualResponse,
) -> (u8, u8, u8) {
    match node_type {
        NodeType::Trunk | NodeType::Branch | NodeType::Bud => rgb_to_u8(visual.stem_rgb),
        NodeType::Leaf => rgb_to_u8(visual.leaf_rgb),
        NodeType::Fruit => rgb_to_u8(visual.fruit_rgb),
    }
}

fn fruit_ground_glyph(taxonomy_id: u32, ripeness: f32) -> char {
    match taxonomy_id {
        3750 => {
            if ripeness > 0.55 {
                '\u{1F34E}'
            } else {
                '\u{25CF}'
            }
        }
        23211 => {
            if ripeness > 0.55 {
                '\u{1F350}'
            } else {
                '\u{25CF}'
            }
        }
        3760 => {
            if ripeness > 0.55 {
                '\u{1F351}'
            } else {
                '\u{25CF}'
            }
        }
        42229 => {
            if ripeness > 0.55 {
                '\u{1F352}'
            } else {
                '\u{25CF}'
            }
        }
        2711 => {
            if ripeness > 0.55 {
                '\u{1F34A}'
            } else {
                '\u{25CF}'
            }
        }
        2708 => {
            if ripeness > 0.55 {
                '\u{1F34B}'
            } else {
                '\u{25CF}'
            }
        }
        15368 => '\u{2731}',
        3702 => '\u{25E6}',
        _ => '\u{25CF}',
    }
}

fn seed_glyph(taxonomy_id: u32) -> char {
    match taxonomy_id {
        23211 | 3750 | 3760 | 42229 | 2711 | 2708 => ',',
        15368 => '\'',
        3702 => '\u{00B7}',
        _ => '\u{25CF}',
    }
}

fn soil_surface_glyph(class: TerrariumSoilSurfaceClass) -> char {
    match class {
        TerrariumSoilSurfaceClass::Mineral => '\u{00B7}',
        TerrariumSoilSurfaceClass::Humus => '\u{2591}',
        TerrariumSoilSurfaceClass::WetDetritus => '~',
        TerrariumSoilSurfaceClass::MicrobialMat => '\u{2731}',
        TerrariumSoilSurfaceClass::NitrifierCrust => ':',
        TerrariumSoilSurfaceClass::DenitrifierFilm => '=',
        TerrariumSoilSurfaceClass::MycorrhizalPatch => '\u{273F}',
        TerrariumSoilSurfaceClass::EarthwormCast => '\u{25D8}',
        TerrariumSoilSurfaceClass::NematodeBloom => '\u{2218}',
    }
}

fn fruit_shape_spread(fruit: &oneura_core::terrarium::TerrariumFruitSnapshot) -> isize {
    if fruit.shape.width_scale.max(fruit.shape.depth_scale) > 1.08 {
        1
    } else {
        0
    }
}

fn fruit_stem_glyph(fruit: &oneura_core::terrarium::TerrariumFruitSnapshot) -> Option<char> {
    if fruit.shape.stem_length > 0.18 {
        Some('\'')
    } else {
        None
    }
}

fn seed_awn_glyph(seed: &oneura_core::terrarium::TerrariumSeedSnapshot) -> Option<char> {
    if seed.shape.awn_length > 0.18 {
        Some('/')
    } else {
        None
    }
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
        " Species:none".to_string()
    } else {
        format!(" Species:{}", names.join(","))
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
        models.push("plants:species");
    }
    if snapshot.species_presence.iter().any(|species| {
        matches!(
            species.domain,
            oneura_core::terrarium::TerrariumSpeciesDomain::Insect
        ) && species.reference_neuron_count.is_some()
    }) {
        models.push("flies:neurons");
    }
    if snapshot.species_presence.iter().any(|species| {
        matches!(
            species.authority,
            oneura_core::terrarium::TerrariumSpeciesAuthority::GuildReference
        )
    }) {
        models.push("soil:guild refs");
    }
    if snapshot.species_presence.iter().any(|species| {
        matches!(
            species.domain,
            oneura_core::terrarium::TerrariumSpeciesDomain::Annelid
        )
    }) {
        models.push("worms:population");
    }
    if models.is_empty() {
        " Models:none".to_string()
    } else {
        format!(" Models:{}", models.join(" | "))
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
                if let (Ok(rows), Ok(cols)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>())
                {
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
        Self {
            ch: ' ',
            fg: (200, 200, 200),
            bg: (15, 15, 25),
        }
    }
}

struct ScreenBuffer {
    width: usize,
    height: usize,
    cells: Vec<Cell>,
}

impl ScreenBuffer {
    fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            cells: vec![Cell::default(); width * height],
        }
    }

    fn clear(&mut self) {
        for cell in &mut self.cells {
            *cell = Cell::default();
        }
    }

    fn set(&mut self, x: usize, y: usize, ch: char, fg_c: (u8, u8, u8), bg_c: (u8, u8, u8)) {
        if x < self.width && y < self.height {
            let idx = y * self.width + x;
            self.cells[idx] = Cell {
                ch,
                fg: fg_c,
                bg: bg_c,
            };
        }
    }

    fn write_str(&mut self, x: usize, y: usize, s: &str, fg_c: (u8, u8, u8), bg_c: (u8, u8, u8)) {
        for (i, ch) in s.chars().enumerate() {
            self.set(x + i, y, ch, fg_c, bg_c);
        }
    }

    fn render_into(&self, use_color: bool, out: &mut String) {
        out.clear();
        out.reserve(self.width * self.height * 12);
        // Use CSI H (cursor position) to start at top-left
        out.push_str("\x1b[H");

        let mut last_fg = None;
        let mut last_bg = None;

        for y in 0..self.height {
            for x in 0..self.width {
                let cell = &self.cells[y * self.width + x];
                if use_color {
                    if last_fg != Some(cell.fg) {
                        out.push_str(&fg(cell.fg.0, cell.fg.1, cell.fg.2));
                        last_fg = Some(cell.fg);
                    }
                    if last_bg != Some(cell.bg) {
                        out.push_str(&bg(cell.bg.0, cell.bg.1, cell.bg.2));
                        last_bg = Some(cell.bg);
                    }
                }
                out.push(cell.ch);
            }

            // At the end of every line, reset colors and clear to end of line (CSI K)
            // to prevent artifacts if the terminal is wider than our buffer.
            if use_color {
                out.push_str(RESET);
                last_fg = None;
                last_bg = None;
            }
            out.push_str("\x1b[K\r\n");
        }

        // Final clear-to-bottom just in case
        out.push_str("\x1b[J");
    }
}

struct RenderScratch {
    split_left: ScreenBuffer,
    split_right: ScreenBuffer,
    true3d_terrain_scene: True3DRenderSceneCache,
    true3d_dynamic_scene: True3DRenderSceneCache,
    true3d_terrain_cell_state: Vec<u64>,
    true3d_water_tile_state: Vec<u64>,
    true3d_water_source_state: Vec<u64>,
    true3d_plant_state: Vec<u64>,
    true3d_tris: Vec<Triangle>,
    true3d_z: Vec<f32>,
    true3d_color: Vec<(u8, u8, u8)>,
    terminal_out: String,
}

impl RenderScratch {
    fn new(width: usize, height: usize) -> Self {
        let half_width = (width / 2).max(1);
        Self {
            split_left: ScreenBuffer::new(half_width, height),
            split_right: ScreenBuffer::new(half_width, height),
            true3d_terrain_scene: True3DRenderSceneCache::default(),
            true3d_dynamic_scene: True3DRenderSceneCache::default(),
            true3d_terrain_cell_state: Vec::new(),
            true3d_water_tile_state: Vec::new(),
            true3d_water_source_state: Vec::new(),
            true3d_plant_state: Vec::new(),
            true3d_tris: Vec::new(),
            true3d_z: Vec::new(),
            true3d_color: Vec::new(),
            terminal_out: String::with_capacity(width * height * 12),
        }
    }

    fn true3d_working_set(
        &mut self,
        width: usize,
        height: usize,
        terrain_cell_count: usize,
        water_source_count: usize,
        plant_count: usize,
    ) -> (
        &mut True3DRenderSceneCache,
        &mut True3DRenderSceneCache,
        &mut Vec<u64>,
        &mut Vec<u64>,
        &mut Vec<u64>,
        &mut Vec<u64>,
        &mut Vec<Triangle>,
        &mut [f32],
        &mut [(u8, u8, u8)],
    ) {
        let Self {
            true3d_terrain_scene,
            true3d_dynamic_scene,
            true3d_terrain_cell_state,
            true3d_water_tile_state,
            true3d_water_source_state,
            true3d_plant_state,
            true3d_tris,
            true3d_z,
            true3d_color,
            ..
        } = self;
        let pixel_count = width * height * 2;
        true3d_terrain_scene.begin_frame();
        true3d_dynamic_scene.begin_frame();
        if true3d_terrain_cell_state.len() != terrain_cell_count {
            true3d_terrain_cell_state.resize(terrain_cell_count, TRUE3D_STATE_MISSING);
        }
        if true3d_water_tile_state.len() != terrain_cell_count {
            true3d_water_tile_state.resize(terrain_cell_count, TRUE3D_STATE_MISSING);
        }
        if true3d_water_source_state.len() != water_source_count {
            true3d_water_source_state.resize(water_source_count, TRUE3D_STATE_MISSING);
        }
        if true3d_plant_state.len() != plant_count {
            true3d_plant_state.resize(plant_count, TRUE3D_STATE_MISSING);
        }
        true3d_tris.clear();
        if true3d_z.len() != pixel_count {
            *true3d_z = vec![f32::MAX; pixel_count];
        } else {
            true3d_z.fill(f32::MAX);
        }
        if true3d_color.len() != pixel_count {
            *true3d_color = vec![(15, 15, 25); pixel_count];
        } else {
            true3d_color.fill((15, 15, 25));
        }
        (
            true3d_terrain_scene,
            true3d_dynamic_scene,
            true3d_terrain_cell_state,
            true3d_water_tile_state,
            true3d_water_source_state,
            true3d_plant_state,
            true3d_tris,
            true3d_z,
            true3d_color,
        )
    }
}

// ---------------------------------------------------------------------------
// Interactive State
// ---------------------------------------------------------------------------

struct ViewState {
    mode: ViewMode,
    preset: TerrariumDemoPreset,
    camera_x: isize, // camera pan offset (grid units)
    camera_y: isize,
    zoom: f32, // 0.5 .. 3.0
    paused: bool,
    show_help: bool,
    show_minimap: bool,
    show_legend: bool,
    time_warp: f32,                     // simulation speed multiplier (0.25x - 8x)
    message: Option<(String, Instant)>, // status message with expiry
    #[allow(dead_code)]
    selected_entity: Option<(char, usize, usize)>, // selected entity (type, x, y)
    show_chemistry: bool,
    show_grid: bool,
    show_vitality: bool,
    show_info: bool,
}

impl ViewState {
    fn new(mode: ViewMode, show_minimap: bool, preset: TerrariumDemoPreset) -> Self {
        Self {
            mode,
            preset,
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
            ViewMode::Split => ViewMode::True3D,
            ViewMode::True3D => ViewMode::Isometric,
        };
    }

    fn mode_name(&self) -> &'static str {
        match self.mode {
            ViewMode::Isometric => "Isometric 3D",
            ViewMode::TopDown => "Top-Down Map",
            ViewMode::Split => "Split View",
            ViewMode::Heatmap => "Moisture Heatmap",
            ViewMode::Dashboard => "Ecosystem Dashboard",
            ViewMode::True3D => "True 3D Software Rasterizer",
        }
    }

    fn set_message<S: Into<String>>(&mut self, text: S, duration: Duration) {
        self.message = Some((text.into(), Instant::now() + duration));
    }

    fn clear_expired_message(&mut self) {
        if self
            .message
            .as_ref()
            .is_some_and(|(_, expires_at)| *expires_at <= Instant::now())
        {
            self.message = None;
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
    let panel_bg = (20, 35, 40); // Dark teal background
    let border_fg = (80, 180, 160); // Bright teal border
    let title_fg = (100, 220, 200); // Bright cyan-teal for title
    let heading_fg = (120, 200, 160); // Green-teal for headings
    let text_fg = (200, 220, 210); // Light gray-green text
    let key_fg = (150, 255, 180); // Bright green for keys
    let sym_fg = (255, 220, 100); // Yellow for symbols
    let dim_fg = (120, 140, 135); // Muted text

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
    buf.set(
        px + panel_w - 1,
        py + panel_h - 1,
        '\u{255D}',
        border_fg,
        panel_bg,
    );

    let mut row = py + 1;
    let cx = px + 2; // content x

    // Title
    buf.write_str(
        cx,
        row,
        "   oNeura Terrarium - Help & Legend",
        title_fg,
        row_bg(row),
    );
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
    buf.write_str(cx, row, "  1-6", key_fg, row_bg(row));
    buf.write_str(
        cx + 22,
        row,
        "Jump to mode (Iso/Top/Heat/Dash/Split/True3D)",
        text_fg,
        row_bg(row),
    );
    row += 1;
    buf.write_str(cx, row, "  Space", key_fg, row_bg(row));
    buf.write_str(
        cx + 22,
        row,
        "Pause / resume simulation",
        text_fg,
        row_bg(row),
    );
    row += 1;
    buf.write_str(cx, row, "  W / S", key_fg, row_bg(row));
    buf.write_str(
        cx + 22,
        row,
        "Orbit vertical / Height",
        text_fg,
        row_bg(row),
    );
    row += 1;
    buf.write_str(cx, row, "  A / D", key_fg, row_bg(row));
    buf.write_str(
        cx + 22,
        row,
        "Orbit horizontal / Rotate",
        text_fg,
        row_bg(row),
    );
    row += 1;
    buf.write_str(cx, row, "  H", key_fg, row_bg(row));
    buf.write_str(
        cx + 22,
        row,
        "Toggle this help overlay",
        text_fg,
        row_bg(row),
    );
    row += 1;
    buf.write_str(cx, row, "  M", key_fg, row_bg(row));
    buf.write_str(cx + 22, row, "Toggle minimap", text_fg, row_bg(row));
    row += 1;
    buf.write_str(cx, row, "  R", key_fg, row_bg(row));
    buf.write_str(
        cx + 22,
        row,
        "Reset camera (center, zoom 1x)",
        text_fg,
        row_bg(row),
    );
    row += 1;
    buf.write_str(cx, row, "  Q / Esc", key_fg, row_bg(row));
    buf.write_str(cx + 22, row, "Quit", text_fg, row_bg(row));
    row += 2;

    // ── Symbols ──
    buf.write_str(cx, row, "\u{2500}\u{2500} SYMBOLS \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}", heading_fg, row_bg(row));
    row += 1;
    buf.write_str(
        cx,
        row,
        "  \u{1F33F} \u{2663} \u{1F332}",
        (20, 100, 20),
        row_bg(row),
    );
    buf.write_str(
        cx + 10,
        row,
        "Plant canopy (green = healthy)",
        text_fg,
        row_bg(row),
    );
    row += 1;
    buf.write_str(cx, row, "  \u{25B2}", (40, 120, 30), row_bg(row));
    buf.write_str(cx + 10, row, "Treetop / crown", text_fg, row_bg(row));
    row += 1;
    buf.write_str(cx, row, "  \u{2503}", (80, 50, 20), row_bg(row));
    buf.write_str(cx + 10, row, "Tree trunk (brown)", text_fg, row_bg(row));
    row += 1;
    // Fruit: show green to red gradient
    buf.write_str(
        cx,
        row,
        "  \u{1F34E} \u{1F347} \u{1F34C}",
        (180, 50, 20),
        row_bg(row),
    );
    buf.write_str(cx + 10, row, "Fruit / Food sources", text_fg, row_bg(row));
    row += 1;
    buf.write_str(cx, row, "  \u{2248} ~", (30, 80, 160), row_bg(row));
    buf.write_str(cx + 10, row, "Water (animated waves)", text_fg, row_bg(row));
    row += 1;
    buf.write_str(
        cx,
        row,
        "  \u{1FE0} \u{2736} \u{2734}",
        (180, 160, 30),
        row_bg(row),
    );
    buf.write_str(
        cx + 10,
        row,
        "Flying fly (animated wings)",
        text_fg,
        row_bg(row),
    );
    row += 1;
    buf.write_str(cx, row, "  \u{1F41E} \u{25C6}", (180, 140, 40), row_bg(row));
    buf.write_str(
        cx + 10,
        row,
        "Landed fly (resting/feeding)",
        text_fg,
        row_bg(row),
    );
    row += 1;
    buf.write_str(cx, row, "  \u{2593} \u{2588}", (100, 70, 40), row_bg(row));
    buf.write_str(
        cx + 10,
        row,
        "Terrain (top face / side face)",
        text_fg,
        row_bg(row),
    );
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
    buf.write_str(
        cx,
        row,
        "  Height = soil moisture (wetter = taller)",
        dim_fg,
        row_bg(row),
    );
    row += 1;
    buf.write_str(
        cx,
        row,
        "  Top face = bright highlight, sides = darker",
        dim_fg,
        row_bg(row),
    );
    row += 2;

    // Footer
    buf.write_str(
        cx,
        row,
        "    Press H to close   |   Tab to change view",
        dim_fg,
        row_bg(row),
    );
}

// ---------------------------------------------------------------------------
// Detailed Legend Bar (bottom of screen)
// ---------------------------------------------------------------------------

fn draw_legend_bar(buf: &mut ScreenBuffer, vs: &ViewState) {
    let bar_h = 4; // 4 rows of legend
    let y = buf.height.saturating_sub(bar_h);
    // Light pastel teal-green for legend bar (easy on eyes)
    let bar_bg = (180, 230, 210); // Light mint/pastel teal
    let sep_fg = (60, 120, 100); // Muted green separator
    let lbl_fg = (0, 0, 0); // Pure black text for maximum readability
    let dim_fg = (50, 90, 75); // Darker green for dim text
    let key_fg = (40, 80, 65); // Dark green for key labels
    let mode_fg = (60, 100, 85); // Dark teal for mode

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
    buf.set(x, r2, '\u{2663}', (20, 100, 20), bar_bg);
    x += 1;
    buf.write_str(x, r2, "=Plant(green=healthy) ", lbl_fg, bar_bg);
    x += 22;

    buf.set(x, r2, '\u{25CF}', (60, 120, 20), bar_bg);
    x += 1;
    buf.write_str(x, r2, "grn", (50, 100, 20), bar_bg);
    x += 3;
    buf.set(x, r2, '\u{2192}', dim_fg, bar_bg);
    x += 1;
    buf.set(x, r2, '\u{25CF}', (160, 50, 20), bar_bg);
    x += 1;
    buf.write_str(x, r2, "red=Fruit(ripeness) ", lbl_fg, bar_bg);
    x += 20;

    buf.set(x, r2, '\u{2248}', (30, 70, 140), bar_bg);
    x += 1;
    buf.write_str(x, r2, "=Water ", lbl_fg, bar_bg);
    x += 7;

    buf.set(x, r2, '\u{2736}', (160, 140, 30), bar_bg);
    x += 1;
    buf.write_str(x, r2, "=Fly(air) ", lbl_fg, bar_bg);
    x += 10;

    buf.set(x, r2, '\u{25C6}', (160, 120, 40), bar_bg);
    x += 1;
    buf.write_str(x, r2, "=Fly(landed) ", lbl_fg, bar_bg);
    x += 13;

    buf.set(x, r2, '\u{2593}', (100, 80, 50), bar_bg);
    x += 1;
    buf.write_str(x, r2, "=Terrain top ", lbl_fg, bar_bg);
    x += 13;

    buf.set(x, r2, '\u{2588}', (70, 50, 30), bar_bg);
    x += 1;
    buf.write_str(x, r2, "=Side", lbl_fg, bar_bg);

    // ── Row 3: Color gradients ──
    let r3 = y + 2;
    x = 1;
    buf.write_str(x, r3, "Soil: ", dim_fg, bar_bg);
    x += 6;
    for i in 0..10 {
        let t = i as f32 / 9.0;
        let c = soil_color(t, t * 0.4);
        buf.set(x + i, r3, '\u{2588}', c, bar_bg);
    }
    x += 10;
    buf.write_str(x, r3, "(sandy dry", (60, 50, 30), bar_bg);
    x += 10;
    buf.set(x, r3, '\u{2192}', dim_fg, bar_bg);
    x += 1;
    buf.write_str(x, r3, "dark wet) ", (40, 30, 20), bar_bg);
    x += 10;

    buf.write_str(x, r3, "Plants: ", dim_fg, bar_bg);
    x += 8;
    let pc_healthy = plant_color(1.0, 1.0);
    let pc_stressed = plant_color(0.8, 0.2);
    buf.set(x, r3, '\u{2588}', pc_healthy, bar_bg);
    x += 1;
    buf.set(x, r3, '\u{2588}', pc_healthy, bar_bg);
    x += 1;
    buf.write_str(x, r3, "healthy ", (20, 80, 20), bar_bg);
    x += 8;
    buf.set(x, r3, '\u{2588}', pc_stressed, bar_bg);
    x += 1;
    buf.set(x, r3, '\u{2588}', pc_stressed, bar_bg);
    x += 1;
    buf.write_str(x, r3, "stressed ", (100, 90, 30), bar_bg);
    x += 9;

    buf.write_str(x, r3, "Height=moisture(wet=tall)", dim_fg, bar_bg);

    // ── Row 4: Controls + mode ──
    let r4 = y + 3;
    let controls = format!(
        " [H]elp [Tab]mode [1-6]jump [WASD/Arrows]pan/orbit [+/-]zoom [Space]{} [M]inimap [R]eset [Q]uit",
        if vs.paused { "resume" } else { "pause" },
    );
    buf.write_str(
        0,
        r4,
        &controls[..controls.len().min(buf.width)],
        key_fg,
        bar_bg,
    );

    // Mode name right-aligned
    let mode_str = format!(" {} | {} ", vs.preset.label(), vs.mode_name());
    let mx = buf.width.saturating_sub(mode_str.len() + 1);
    buf.write_str(mx, r4, &mode_str, mode_fg, bar_bg);

    // Paused indicator
    if vs.paused {
        let px = buf.width.saturating_sub(mode_str.len() + 12);
        buf.write_str(px, r4, " PAUSED ", (255, 80, 80), (80, 20, 20));
    }
}

fn draw_status_banner(buf: &mut ScreenBuffer, vs: &ViewState) {
    let Some((message, _)) = vs.message.as_ref() else {
        return;
    };
    let banner_bg = (28, 42, 58);
    let banner_fg = (245, 248, 255);
    let text = format!(" {message} ");
    let x = buf.width.saturating_sub(text.len()) / 2;
    let y = 1;
    buf.write_str(
        x,
        y,
        &text[..text.len().min(buf.width)],
        banner_fg,
        banner_bg,
    );
}

fn push_snapshot_history(
    snapshot_history: &mut VecDeque<TerrariumWorldSnapshot>,
    snapshot: TerrariumWorldSnapshot,
) {
    if snapshot_history.len() >= 200 {
        snapshot_history.pop_front();
    }
    snapshot_history.push_back(snapshot);
}

fn render_active_view(
    buf: &mut ScreenBuffer,
    scratch: &mut RenderScratch,
    world: &TerrariumWorld,
    snapshot: &TerrariumWorldSnapshot,
    snapshot_history: &VecDeque<TerrariumWorldSnapshot>,
    frame_idx: usize,
    vs: &ViewState,
) {
    match vs.mode {
        ViewMode::Isometric => {
            render_isometric(buf, world, snapshot, frame_idx, vs);
            if vs.show_minimap {
                let mx = buf.width.saturating_sub(world.config.width + 4);
                draw_minimap(buf, world, mx, 3);
            }
        }
        ViewMode::TopDown => {
            render_topdown_color(buf, world, snapshot, frame_idx, vs);
            if vs.show_minimap {
                let mx = buf.width.saturating_sub(world.config.width + 4);
                draw_minimap(buf, world, mx, 3);
            }
        }
        ViewMode::Split => {
            let half_w = buf.width / 2;
            render_isometric(&mut scratch.split_left, world, snapshot, frame_idx, vs);
            render_topdown_color(&mut scratch.split_right, world, snapshot, frame_idx, vs);
            buf.clear();
            for y in 0..buf.height {
                for x in 0..half_w {
                    let lc = &scratch.split_left.cells[y * scratch.split_left.width + x];
                    buf.set(x, y, lc.ch, lc.fg, lc.bg);
                    if x < scratch.split_right.width {
                        let rc = &scratch.split_right.cells[y * scratch.split_right.width + x];
                        buf.set(x + half_w, y, rc.ch, rc.fg, rc.bg);
                    }
                }
            }
        }
        ViewMode::Heatmap => render_heatmap(buf, world, snapshot, frame_idx, vs),
        ViewMode::Dashboard => {
            let history: Vec<_> = snapshot_history.iter().cloned().collect();
            render_dashboard(buf, &history, frame_idx);
        }
        ViewMode::True3D => render_true3d(buf, world, snapshot, frame_idx, vs, scratch),
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
    let atmosphere = world.atmosphere_frame();
    let fruit_visuals = world.fruit_visuals();
    let seed_visuals = world.seed_visuals();
    let earthworms = world.earthworm_visual_markers();
    let nematodes = world.nematode_visual_markers();
    let soil_surface = world.soil_surface_markers();
    let mean_humidity = mean_field(&atmosphere.humidity, snapshot.humidity);
    let mean_pressure = mean_field(&atmosphere.pressure_kpa, snapshot.mean_air_pressure_kpa);
    let mean_wind = mean_wind_speed(&atmosphere);

    // Screen center offset with camera pan and zoom
    let z = vs.zoom;
    let cell_w = (2.0 * z).round() as isize;
    let cell_h_half = (z * 0.5).round().max(1.0) as isize;
    let ox = (buf.width as isize) / 2 + vs.camera_x * cell_w;
    let oy = 3 + vs.camera_y;

    // Header with lunar phase emoji
    let moon_emoji = world.moon_phase_name();
    let header = format!(
        " oNeura Terrarium 3D | F:{} | {} | {:.0}C | {} plants:{} flies:{} fruit:{}",
        frame_idx,
        world.time_label(),
        snapshot.temperature,
        moon_emoji,
        snapshot.plants,
        snapshot.flies,
        snapshot.fruits,
    );
    buf.write_str(
        0,
        0,
        &header[..header.len().min(buf.width)],
        (0, 220, 255),
        (20, 20, 40),
    );

    // Lunar/circadian status line
    let lunar_status = format!(
        " moon:{:.0}% tide:{:+.0}% | wing:{:.0}Hz",
        snapshot.moonlight * 100.0,
        (snapshot.tidal_moisture_factor - 1.0) * 100.0,
        snapshot.avg_wing_beat_freq,
    );
    let sub = format!(
        " moisture:{:.2} microbes:{:.3} CO2:{:.4} O2:{:.2} RH:{:.0}% wind:{:.3} p:{:.2}kPa zoom:{:.1}x | {}",
        snapshot.mean_soil_moisture, snapshot.mean_microbes,
        snapshot.mean_atmospheric_co2, snapshot.mean_atmospheric_o2,
        mean_humidity * 100.0, mean_wind, mean_pressure, vs.zoom, lunar_status,
    );
    buf.write_str(
        0,
        1,
        &sub[..sub.len().min(buf.width)],
        (180, 180, 200),
        (20, 20, 40),
    );

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
            let m = if m_idx < moisture.len() {
                moisture[m_idx]
            } else {
                0.3
            };
            let height = (m * 3.0 * z).clamp(0.0, 6.0) as isize;
            let local_air =
                sample_local_atmosphere(&atmosphere, gw, gh, gx as f32 + 0.5, gy as f32 + 0.5);
            let color = soil_color_with_atmosphere(m, m * 0.5, local_air);

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

    for (gx, gy, visual) in &soil_surface {
        let iso_x = (*gx as isize - *gy as isize) * cell_w + ox;
        let iso_y = (*gx as isize + *gy as isize) * cell_h_half + oy;
        if iso_x >= 0 && iso_y >= 0 {
            let fg = rgb_to_u8(visual.accent_rgb);
            let glyph = soil_surface_glyph(visual.class);
            set_if_visible(
                buf,
                iso_x + 1,
                iso_y - 1 - (visual.height_offset * 8.0).round() as isize,
                glyph,
                fg,
                (12, 12, 18),
            );
            if visual.width_scale > 1.15 {
                set_if_visible(buf, iso_x + 2, iso_y - 1, glyph, fg, (12, 12, 18));
            }
            if visual.depth_scale > 1.15 {
                set_if_visible(buf, iso_x, iso_y - 2, glyph, fg, (12, 12, 18));
            }
        }
    }

    // Water sources
    for water in &world.waters {
        if water.alive {
            let iso_x = (water.x as isize - water.y as isize) * cell_w + ox;
            let iso_y = (water.x as isize + water.y as isize) * cell_h_half + oy;
            let local_air = sample_local_atmosphere(
                &atmosphere,
                gw,
                gh,
                water.x as f32 + 0.5,
                water.y as f32 + 0.5,
            );
            if iso_x >= 0 && iso_y >= 0 {
                let sx = iso_x as usize;
                let sy =
                    (iso_y - (local_air.pressure_delta_kpa * 4.0).round() as isize).max(0) as usize;
                let wave = water_surface_glyph(local_air, frame_idx);
                let water_fg = water_color(local_air);
                let tile_w = (cell_w as usize + 1).max(3).min(6);
                for dx in 0..tile_w {
                    buf.set(sx + dx, sy, wave, water_fg, (20, 60, 120));
                }
                if sy + 1 < buf.height {
                    buf.set(
                        sx + 1,
                        sy + 1,
                        '\u{2592}',
                        lerp_color(water_fg, (220, 240, 255), 0.18),
                        (15, 40, 80),
                    );
                }
            }
        }
    }

    // Plants
    for (i, plant) in snapshot.full_plants.iter().enumerate() {
        let local_air = sample_local_atmosphere(
            &atmosphere,
            gw,
            gh,
            plant.x as f32 + 0.5,
            plant.y as f32 + 0.5,
        );
        let (vitality, canopy_density) = plant_metrics(world, i);
        let visual = plant_visual_response(
            plant.taxonomy_id,
            local_air,
            vitality,
            canopy_density,
            world.time_s,
            i as f32,
        );
        let base_ground = (moisture
            [plant.y.min(gh.saturating_sub(1)) * gw + plant.x.min(gw.saturating_sub(1))]
            * 3.0
            * z)
            .clamp(0.0, 6.0);

        for node in &plant.morphology {
            let world_x = plant.x as f32
                + node.position[0]
                + visual.sway_x * (0.16 + node.position[1].max(0.0) * 0.05);
            let world_y = plant.y as f32
                + node.position[2]
                + visual.sway_z * (0.16 + node.position[1].max(0.0) * 0.05);
            let screen_x = (world_x - world_y) * cell_w as f32 + ox as f32;
            let screen_y = (world_x + world_y) * cell_h_half as f32 + oy as f32
                - base_ground
                - node.position[1] * z * 0.95
                - visual.vertical_offset * 8.0;
            let px = screen_x.round() as isize + 1;
            let py = screen_y.round() as isize - 1;
            let ch = plant_node_char(node.node_type);
            let color = plant_node_color(node.node_type, visual);
            let spread = match node.node_type {
                NodeType::Leaf => {
                    ((node.radius * visual.canopy_scale * z).round() as isize).clamp(1, 2)
                }
                NodeType::Fruit => 1,
                _ => 0,
            };

            if spread == 0 {
                set_if_visible(buf, px, py, ch, color, (10, 10, 20));
            } else {
                for dy in -spread..=spread {
                    for dx in -spread..=spread {
                        if dx.abs() + dy.abs() > spread + 1 {
                            continue;
                        }
                        set_if_visible(buf, px + dx, py + dy, ch, color, (10, 10, 20));
                    }
                }
            }
        }
    }

    // Fruits
    for (i, fruit) in world.fruits.iter().enumerate() {
        if fruit.source.alive && fruit.source.sugar_content > 0.01 {
            let fruit_meta = snapshot.full_fruits.get(i);
            let iso_x = (fruit.source.x as isize - fruit.source.y as isize) * cell_w + ox;
            let iso_y = (fruit.source.x as isize + fruit.source.y as isize) * cell_h_half + oy;
            if iso_x >= 0 && iso_y >= 0 {
                let visual = fruit_visuals.get(i).copied().unwrap_or_default();
                let taxonomy_id = fruit_meta
                    .map(|f| f.taxonomy_id)
                    .unwrap_or(fruit.taxonomy_id);
                let fruit_sym = fruit_ground_glyph(
                    taxonomy_id,
                    fruit_meta
                        .map(|f| f.ripeness)
                        .unwrap_or(fruit.source.ripeness)
                        .clamp(0.0, 1.0),
                );
                let px = iso_x as usize + 1;
                let py = (iso_y - (visual.vertical_offset * 6.0).round() as isize).max(0) as usize;
                buf.set(px, py, fruit_sym, rgb_to_u8(visual.skin_rgb), (10, 10, 20));
                if let Some(fruit_meta) = fruit_meta {
                    if fruit_shape_spread(fruit_meta) > 0 {
                        set_if_visible(
                            buf,
                            px as isize + 1,
                            py as isize,
                            fruit_sym,
                            rgb_to_u8(visual.skin_rgb),
                            (10, 10, 20),
                        );
                    }
                    if let Some(stem_glyph) = fruit_stem_glyph(fruit_meta) {
                        set_if_visible(
                            buf,
                            px as isize,
                            py as isize - 1,
                            stem_glyph,
                            rgb_to_u8(visual.stem_rgb),
                            (10, 10, 20),
                        );
                    }
                }
            }
        }
    }

    for (i, seed) in world.seeds.iter().enumerate() {
        let seed_meta = snapshot.full_seeds.get(i);
        let visual = seed_visuals.get(i).copied().unwrap_or_default();
        let iso_x = (seed.x.round() as isize - seed.y.round() as isize) * cell_w + ox;
        let iso_y = (seed.x.round() as isize + seed.y.round() as isize) * cell_h_half + oy;
        if iso_x >= 0 && iso_y >= 0 {
            let px = iso_x as usize + 1;
            let py = (iso_y - (visual.vertical_offset * 5.0).round() as isize).max(0) as usize;
            let taxonomy_id = seed_meta
                .map(|s| s.taxonomy_id)
                .unwrap_or(seed.genome.taxonomy_id);
            buf.set(
                px,
                py,
                seed_glyph(taxonomy_id),
                rgb_to_u8(visual.shell_rgb),
                (10, 10, 20),
            );
            if let Some(seed_meta) = seed_meta {
                if let Some(awn_glyph) = seed_awn_glyph(seed_meta) {
                    set_if_visible(
                        buf,
                        px as isize + 1,
                        py as isize,
                        awn_glyph,
                        rgb_to_u8(visual.accent_rgb),
                        (10, 10, 20),
                    );
                }
            }
        }
    }

    // Flies
    for (i, fly) in world.flies.iter().enumerate() {
        let body: BodyState = fly.body_state().clone();
        let visual = sample_fly_visual(world, &atmosphere, gw, gh, i, &body, world.time_s);
        let body_color = rgb_to_u8(visual.body_rgb);
        let wing_color = rgb_to_u8(visual.wing_rgb);
        let gx = body.x.round().clamp(0.0, (gw - 1) as f32) as isize;
        let gy = body.y.round().clamp(0.0, (gh - 1) as f32) as isize;
        let iso_x = (gx - gy) * cell_w + ox;
        let iso_y = (gx + gy) * cell_h_half + oy;
        let altitude = (body.z * z).clamp(0.0, 5.0) as isize;

        if iso_x >= 0 && iso_y >= altitude {
            let sy = (iso_y - altitude) as usize;
            let sx = iso_x as usize;
            buf.set(
                sx + 1,
                sy,
                fly_body_glyph(&body, visual.wing_angle, visual.proboscis_extension),
                body_color,
                (10, 10, 20),
            );
            let (wx, wy) = heading_cell_offset(body.heading);
            set_if_visible(
                buf,
                sx as isize + 1 + wx,
                sy as isize + wy,
                fly_wing_glyph(body.heading),
                wing_color,
                (10, 10, 20),
            );
        }
    }

    draw_isometric_fauna_markers(buf, ox, oy, cell_w, cell_h_half, &earthworms, &nematodes);
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
    let atmosphere = world.atmosphere_frame();
    let fruit_visuals = world.fruit_visuals();
    let seed_visuals = world.seed_visuals();
    let earthworms = world.earthworm_visual_markers();
    let nematodes = world.nematode_visual_markers();
    let soil_surface = world.soil_surface_markers();
    let mean_humidity = mean_field(&atmosphere.humidity, snapshot.humidity);
    let mean_pressure = mean_field(&atmosphere.pressure_kpa, snapshot.mean_air_pressure_kpa);
    let mean_wind = mean_wind_speed(&atmosphere);

    let moon_emoji = world.moon_phase_name();
    let header = format!(
        " Terrarium Top-Down | F:{} | {} | {:.0}C | {} P:{} Fl:{} Fr:{}  zoom:{:.1}x",
        frame_idx,
        world.time_label(),
        snapshot.temperature,
        moon_emoji,
        snapshot.plants,
        snapshot.flies,
        snapshot.fruits,
        vs.zoom,
    );
    buf.write_str(
        0,
        0,
        &header[..header.len().min(buf.width)],
        (0, 220, 255),
        (20, 20, 40),
    );

    // Lunar/circadian status line
    let lunar_status = format!(
        " moon:{:.0}% tide:{:+.0}% | RH:{:.0}% wind:{:.3} p:{:.2}kPa | {}",
        snapshot.moonlight * 100.0,
        (snapshot.tidal_moisture_factor - 1.0) * 100.0,
        mean_humidity * 100.0,
        mean_wind,
        mean_pressure,
        if snapshot.light > 0.3 {
            "day"
        } else if snapshot.moonlight > 0.3 {
            "moonlit-night"
        } else {
            "dark-night"
        },
    );
    buf.write_str(
        0,
        1,
        &lunar_status[..lunar_status.len().min(buf.width)],
        (180, 180, 200),
        (20, 20, 40),
    );

    let scale = (2.0 * vs.zoom).round() as usize;
    let scale = scale.max(1).min(6);
    let ox = ((buf.width as isize).saturating_sub((gw * scale) as isize)) / 2
        + vs.camera_x * scale as isize;
    let oy = 2 + vs.camera_y;

    for gy in 0..gh {
        for gx in 0..gw {
            let m_idx = gy * gw + gx;
            let m = if m_idx < moisture.len() {
                moisture[m_idx]
            } else {
                0.3
            };
            let local_air =
                sample_local_atmosphere(&atmosphere, gw, gh, gx as f32 + 0.5, gy as f32 + 0.5);
            let color = soil_color_with_atmosphere(m, m * 0.5, local_air);
            let ch = if m > 0.5 {
                '\u{2593}'
            } else if m > 0.3 {
                '\u{2592}'
            } else {
                '\u{2591}'
            };
            let px = ox + (gx * scale) as isize;
            let py = oy + gy as isize;
            if py >= 0 {
                for dx in 0..scale {
                    if px + dx as isize >= 0 {
                        buf.set(
                            (px + dx as isize) as usize,
                            py as usize,
                            ch,
                            color,
                            (10, 10, 20),
                        );
                    }
                }
            }
        }
    }

    for (gx, gy, visual) in &soil_surface {
        let px = ox + (*gx * scale) as isize;
        let py = oy + *gy as isize - (visual.height_offset * 4.0).round() as isize;
        if px >= 0 && py >= 0 {
            let glyph = soil_surface_glyph(visual.class);
            buf.set(
                px as usize,
                py as usize,
                glyph,
                rgb_to_u8(visual.accent_rgb),
                (12, 12, 18),
            );
            if visual.width_scale > 1.15 {
                set_if_visible(
                    buf,
                    px + 1,
                    py,
                    glyph,
                    rgb_to_u8(visual.accent_rgb),
                    (12, 12, 18),
                );
            }
            if visual.depth_scale > 1.15 {
                set_if_visible(
                    buf,
                    px,
                    py + 1,
                    glyph,
                    rgb_to_u8(visual.accent_rgb),
                    (12, 12, 18),
                );
            }
        }
    }

    for water in &world.waters {
        if water.alive {
            let local_air = sample_local_atmosphere(
                &atmosphere,
                gw,
                gh,
                water.x as f32 + 0.5,
                water.y as f32 + 0.5,
            );
            let wave = water_surface_glyph(local_air, frame_idx);
            let water_fg = water_color(local_air);
            for dy in 0..3_usize {
                for dx in 0..3_usize {
                    let wx = water.x.saturating_add(dx).saturating_sub(1);
                    let wy = water.y.saturating_add(dy).saturating_sub(1);
                    if wx < gw && wy < gh {
                        let px = ox + (wx * scale) as isize;
                        let py = oy + wy as isize
                            - (local_air.pressure_delta_kpa * 3.0).round() as isize;
                        if px >= 0 && py >= 0 {
                            for s in 0..scale {
                                buf.set(
                                    (px + s as isize) as usize,
                                    py as usize,
                                    wave,
                                    water_fg,
                                    (20, 50, 100),
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    for (i, plant) in snapshot.full_plants.iter().enumerate() {
        let local_air = sample_local_atmosphere(
            &atmosphere,
            gw,
            gh,
            plant.x as f32 + 0.5,
            plant.y as f32 + 0.5,
        );
        let (vitality, canopy_density) = plant_metrics(world, i);
        let visual = plant_visual_response(
            plant.taxonomy_id,
            local_air,
            vitality,
            canopy_density,
            world.time_s,
            i as f32,
        );

        for node in &plant.morphology {
            let px = ox
                + ((plant.x as f32
                    + node.position[0]
                    + visual.sway_x * (0.14 + node.position[1].max(0.0) * 0.04))
                    * scale as f32)
                    .round() as isize;
            let py = oy
                + (plant.y as f32
                    + node.position[2]
                    + visual.sway_z * (0.14 + node.position[1].max(0.0) * 0.04))
                    .round() as isize
                - (visual.vertical_offset * 5.0).round() as isize;
            let ch = match node.node_type {
                NodeType::Leaf if scale > 1 => '\u{2588}',
                NodeType::Leaf => '\u{2593}',
                _ => plant_node_char(node.node_type),
            };
            let pc = plant_node_color(node.node_type, visual);
            let spread = match node.node_type {
                NodeType::Leaf => ((node.radius * visual.canopy_scale * scale as f32).round()
                    as isize)
                    .clamp(1, 2),
                NodeType::Fruit => 1,
                _ => 0,
            };
            if spread == 0 {
                set_if_visible(buf, px, py, ch, pc, (10, 30, 10));
            } else {
                for dy in -spread..=spread {
                    for dx in -spread..=spread {
                        if dx.abs() + dy.abs() > spread + 1 {
                            continue;
                        }
                        set_if_visible(buf, px + dx, py + dy, ch, pc, (10, 30, 10));
                    }
                }
            }
        }
    }

    for (i, fruit) in world.fruits.iter().enumerate() {
        if fruit.source.alive && fruit.source.sugar_content > 0.01 {
            let fruit_meta = snapshot.full_fruits.get(i);
            let visual = fruit_visuals.get(i).copied().unwrap_or_default();
            let px = ox + (fruit.source.x * scale) as isize;
            let py = oy + fruit.source.y as isize;
            if px >= 0 && py >= 0 {
                let taxonomy_id = fruit_meta
                    .map(|f| f.taxonomy_id)
                    .unwrap_or(fruit.taxonomy_id);
                let glyph = fruit_ground_glyph(
                    taxonomy_id,
                    fruit_meta
                        .map(|f| f.ripeness)
                        .unwrap_or(fruit.source.ripeness),
                );
                buf.set(
                    px as usize,
                    py as usize,
                    glyph,
                    rgb_to_u8(visual.skin_rgb),
                    (10, 10, 20),
                );
                if let Some(fruit_meta) = fruit_meta {
                    if fruit_shape_spread(fruit_meta) > 0 {
                        set_if_visible(
                            buf,
                            px + 1,
                            py,
                            glyph,
                            rgb_to_u8(visual.skin_rgb),
                            (10, 10, 20),
                        );
                    }
                }
            }
        }
    }

    for (i, seed) in world.seeds.iter().enumerate() {
        let seed_meta = snapshot.full_seeds.get(i);
        let visual = seed_visuals.get(i).copied().unwrap_or_default();
        let px = ox + (seed.x.round() as usize * scale) as isize;
        let py = oy + seed.y.round() as isize;
        if px >= 0 && py >= 0 {
            let taxonomy_id = seed_meta
                .map(|s| s.taxonomy_id)
                .unwrap_or(seed.genome.taxonomy_id);
            buf.set(
                px as usize,
                py as usize,
                seed_glyph(taxonomy_id),
                rgb_to_u8(visual.shell_rgb),
                (10, 10, 20),
            );
            if let Some(seed_meta) = seed_meta {
                if let Some(awn_glyph) = seed_awn_glyph(seed_meta) {
                    set_if_visible(
                        buf,
                        px + 1,
                        py,
                        awn_glyph,
                        rgb_to_u8(visual.accent_rgb),
                        (10, 10, 20),
                    );
                }
            }
        }
    }

    for (i, fly) in world.flies.iter().enumerate() {
        let body: BodyState = fly.body_state().clone();
        let visual = sample_fly_visual(world, &atmosphere, gw, gh, i, &body, world.time_s);
        let gx = body.x.round().clamp(0.0, (gw - 1) as f32) as usize;
        let gy = body.y.round().clamp(0.0, (gh - 1) as f32) as usize;
        let px = ox + (gx * scale) as isize;
        let py = oy + gy as isize;
        if px >= 0 && py >= 0 {
            buf.set(
                px as usize,
                py as usize,
                fly_body_glyph(&body, visual.wing_angle, visual.proboscis_extension),
                rgb_to_u8(visual.body_rgb),
                (10, 10, 20),
            );
            let (wx, wy) = heading_cell_offset(body.heading);
            set_if_visible(
                buf,
                px + wx,
                py + wy,
                fly_wing_glyph(body.heading),
                rgb_to_u8(visual.wing_rgb),
                (10, 10, 20),
            );
        }
    }

    draw_topdown_fauna_markers(buf, ox, oy, scale, &earthworms, &nematodes);

    let stats_y = oy.max(0) as usize + gh + 1;
    let max_stats_y = buf.height.saturating_sub(5); // leave room for legend bar
    let stats = [
        format!(
            " Moisture:{:.3} Deep:{:.3} Glucose:{:.3}",
            snapshot.mean_soil_moisture, snapshot.mean_deep_moisture, snapshot.mean_soil_glucose
        ),
        format!(
            " Microbes:{:.3} Symbionts:{:.3} ATP:{:.3}",
            snapshot.mean_microbes, snapshot.mean_symbionts, snapshot.mean_soil_atp_flux
        ),
        format!(
            " CO2:{:.4} O2:{:.4} Cells:{:.0} Energy:{:.2}",
            snapshot.mean_atmospheric_co2,
            snapshot.mean_atmospheric_o2,
            snapshot.total_plant_cells,
            snapshot.mean_cell_energy
        ),
        format!(
            " RH:{:.0}% Wind:{:.3} Pressure:{:.2}kPa",
            mean_humidity * 100.0,
            mean_wind,
            mean_pressure
        ),
        format!(
            " FlyEnergy:{:.1} FlyEC:{:.2} Seeds:{} Events:{}",
            snapshot.avg_fly_energy,
            snapshot.avg_fly_energy_charge,
            snapshot.seeds,
            snapshot.ecology_event_count
        ),
        plant_species_line(snapshot),
        species_model_line(snapshot),
    ];
    for (i, line) in stats.iter().enumerate() {
        if stats_y + i < max_stats_y {
            buf.write_str(
                0,
                stats_y + i,
                &line[..line.len().min(buf.width)],
                (160, 200, 160),
                (15, 25, 15),
            );
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
        frame_idx,
        snapshot.temperature,
        snapshot.mean_soil_moisture,
        snapshot.plants,
        snapshot.flies,
        vs.zoom,
    );
    buf.write_str(
        0,
        0,
        &header[..header.len().min(buf.width)],
        (255, 200, 0),
        (30, 10, 10),
    );

    let scale = (3.0 * vs.zoom).round() as usize;
    let scale = scale.max(1).min(8);
    let ox = ((buf.width as isize).saturating_sub((gw * scale) as isize)) / 2
        + vs.camera_x * scale as isize;
    let oy = 2 + vs.camera_y;

    for gy in 0..gh {
        for gx in 0..gw {
            let m_idx = gy * gw + gx;
            let m = if m_idx < moisture.len() {
                moisture[m_idx]
            } else {
                0.0
            };
            let color = heatmap_color(m.clamp(0.0, 1.0));
            let ch = if m > 0.7 {
                '\u{2588}'
            } else if m > 0.5 {
                '\u{2593}'
            } else if m > 0.3 {
                '\u{2592}'
            } else {
                '\u{2591}'
            };
            let px = ox + (gx * scale) as isize;
            let py = oy + gy as isize;
            if py >= 0 {
                for dx in 0..scale {
                    if px + dx as isize >= 0 {
                        buf.set(
                            (px + dx as isize) as usize,
                            py as usize,
                            ch,
                            color,
                            (5, 5, 15),
                        );
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
            buf.set(
                px as usize,
                py as usize,
                '\u{2663}',
                (0, 255, 0),
                (5, 5, 15),
            );
        }
    }

    // Overlay flies
    for fly in &world.flies {
        let body: BodyState = fly.body_state().clone();
        let fx = ox
            + (body.x.round().clamp(0.0, (gw - 1) as f32) as usize * scale) as isize
            + scale as isize / 2;
        let fy = oy + body.y.round().clamp(0.0, (gh - 1) as f32) as isize;
        if fx >= 0 && fy >= 0 {
            buf.set(
                fx as usize,
                fy as usize,
                '\u{25C6}',
                (255, 255, 0),
                (5, 5, 15),
            );
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
    for (y, line_obj) in dash.lines().enumerate() {
        let line: &str = line_obj;
        if y >= buf.height.saturating_sub(3) {
            break;
        }
        let is_header = line.starts_with('=') || line.starts_with('-');
        let fg_c = if is_header {
            (0, 220, 255)
        } else {
            (200, 220, 200)
        };
        let bg_c = if is_header {
            (20, 40, 60)
        } else {
            (15, 15, 25)
        };
        buf.write_str(0, y, &line[..line.len().min(buf.width)], fg_c, bg_c);
    }

    let frame_str = format!(" F:{frame_idx} ");
    let fx = buf.width.saturating_sub(frame_str.len() + 1);
    buf.write_str(
        fx,
        buf.height.saturating_sub(4),
        &frame_str,
        (255, 200, 100),
        (30, 30, 50),
    );
}

// ---------------------------------------------------------------------------
// Minimap overlay
// ---------------------------------------------------------------------------

fn draw_minimap(buf: &mut ScreenBuffer, world: &TerrariumWorld, map_x: usize, map_y: usize) {
    let gw = world.config.width;
    let gh = world.config.height;
    let moisture = world.moisture_field();

    buf.write_str(map_x, map_y, "\u{250C}", (80, 80, 120), (10, 10, 20));
    for x in 1..=gw {
        buf.set(map_x + x, map_y, '\u{2500}', (80, 80, 120), (10, 10, 20));
    }
    buf.set(
        map_x + gw + 1,
        map_y,
        '\u{2510}',
        (80, 80, 120),
        (10, 10, 20),
    );

    for gy in 0..gh {
        buf.set(
            map_x,
            map_y + gy + 1,
            '\u{2502}',
            (80, 80, 120),
            (10, 10, 20),
        );
        for gx in 0..gw {
            let m_idx = gy * gw + gx;
            let m = if m_idx < moisture.len() {
                moisture[m_idx]
            } else {
                0.3
            };
            let c = soil_color(m, m * 0.3);
            buf.set(map_x + gx + 1, map_y + gy + 1, '\u{2588}', c, (5, 5, 10));
        }
        buf.set(
            map_x + gw + 1,
            map_y + gy + 1,
            '\u{2502}',
            (80, 80, 120),
            (10, 10, 20),
        );
    }

    buf.set(
        map_x,
        map_y + gh + 1,
        '\u{2514}',
        (80, 80, 120),
        (10, 10, 20),
    );
    for x in 1..=gw {
        buf.set(
            map_x + x,
            map_y + gh + 1,
            '\u{2500}',
            (80, 80, 120),
            (10, 10, 20),
        );
    }
    buf.set(
        map_x + gw + 1,
        map_y + gh + 1,
        '\u{2518}',
        (80, 80, 120),
        (10, 10, 20),
    );

    for plant in &world.plants {
        if plant.x < gw && plant.y < gh {
            buf.set(
                map_x + plant.x + 1,
                map_y + plant.y + 1,
                '\u{2022}',
                (0, 200, 0),
                (5, 5, 10),
            );
        }
    }

    for fly in &world.flies {
        let body: BodyState = fly.body_state().clone();
        let fx = body.x.round().clamp(0.0, (gw - 1) as f32) as usize;
        let fy = body.y.round().clamp(0.0, (gh - 1) as f32) as usize;
        buf.set(
            map_x + fx + 1,
            map_y + fy + 1,
            '\u{00B7}',
            (255, 230, 50),
            (5, 5, 10),
        );
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Cli {
    seed: Option<u64>,
    fps: u64,
    frames: Option<usize>,
    preset: TerrariumDemoPreset,
    mode: ViewMode,
    use_color: bool,
    cpu_substrate: bool,
    show_minimap: bool,
    checkpoint_in: Option<String>,
    checkpoint_out: Option<String>,
    archive_in: Option<String>,
    archive_out: Option<String>,
    query_only: bool,
    list_organisms: bool,
    lineage_ids: Vec<u64>,
    name_assignments: Vec<OrganismNameAssignment>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ViewMode {
    Isometric,
    TopDown,
    Split,
    Heatmap,
    Dashboard,
    True3D,
}

impl Default for Cli {
    fn default() -> Self {
        Self {
            seed: None,
            fps: 15,
            frames: None,
            preset: TerrariumDemoPreset::Demo,
            mode: ViewMode::True3D,
            use_color: true,
            cpu_substrate: false,
            show_minimap: false,
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
    eprintln!("oNeura Terrarium 3D CLI Renderer");
    eprintln!();
    eprintln!("Usage: {program_name} [OPTIONS]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --seed <N>       World seed (default: natural entropy)");
    eprintln!("  --fps <N>        Target framerate (default: 15)");
    eprintln!("  --frames <N>     Quit after N frames (default: infinite)");
    eprintln!("  --preset <NAME>  Demo preset: demo | terrarium | aquarium");
    eprintln!("  --mode <MODE>    View mode (default: true3d)");
    eprintln!("                   true3d - Software rasterized terminal 3D");
    eprintln!("                   iso    - 3D isometric overview");
    eprintln!("                   split  - Isometric + analysis map");
    eprintln!("                   top    - Top-down analysis map");
    eprintln!("                   heat   - Moisture heatmap");
    eprintln!("                   dash   - Ecosystem dashboard");
    eprintln!("  --minimap        Show minimap overlay (iso/top modes)");
    eprintln!("  --load-checkpoint <PATH>  Start from a saved terrarium checkpoint");
    eprintln!("  --save-checkpoint <PATH>  Save a terrarium checkpoint on exit");
    eprintln!("  --load-archive <PATH>     Inspect a terrarium archive and exit");
    eprintln!("  --save-archive <PATH>     Save a terrarium archive on exit");
    eprintln!("  --name-organism <ID=NAME> Assign a display name to an organism");
    eprintln!("  --list-organisms          Print tracked organism identities");
    eprintln!("  --show-lineage <ID>       Print the lineage chain for one organism");
    eprintln!("  --query-only              Apply query/save actions and exit");
    eprintln!("  --no-color       Disable ANSI colors");
    eprintln!("  --cpu-substrate  Use CPU substrate instead of GPU");
    eprintln!("  --help, -h       Show this help");
    eprintln!();
    eprintln!("Interactive Controls:");
    eprintln!("  W/A/S/D or Arrows  Pan camera");
    eprintln!("  +  /  -            Zoom in / out");
    eprintln!("  Tab                Cycle view mode");
    eprintln!("  1-6                Jump to view mode");
    eprintln!("  Space              Pause / resume");
    eprintln!("  H                  Toggle help overlay");
    eprintln!("  M                  Toggle minimap");
    eprintln!("  R                  Reset camera");
    eprintln!("  Q / Esc            Quit");
}

fn parse_args(program_name: &str) -> Result<Cli, String> {
    let mut cli = Cli::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--seed" => {
                cli.seed = Some(
                    args.next()
                        .ok_or("missing --seed")?
                        .parse()
                        .map_err(|_| "bad --seed")?,
                )
            }
            "--load-checkpoint" => {
                cli.checkpoint_in = Some(args.next().ok_or("missing --load-checkpoint")?)
            }
            "--save-checkpoint" => {
                cli.checkpoint_out = Some(args.next().ok_or("missing --save-checkpoint")?)
            }
            "--load-archive" => cli.archive_in = Some(args.next().ok_or("missing --load-archive")?),
            "--save-archive" => {
                cli.archive_out = Some(args.next().ok_or("missing --save-archive")?)
            }
            "--name-organism" => {
                let spec = args.next().ok_or("missing --name-organism")?;
                cli.name_assignments.push(parse_name_assignment(&spec)?);
            }
            "--list-organisms" => cli.list_organisms = true,
            "--show-lineage" => {
                cli.lineage_ids.push(
                    args.next()
                        .ok_or("missing --show-lineage")?
                        .parse()
                        .map_err(|_| "bad --show-lineage")?,
                );
            }
            "--query-only" => cli.query_only = true,
            "--fps" => {
                cli.fps = args
                    .next()
                    .ok_or("missing --fps")?
                    .parse()
                    .map_err(|_| "bad --fps")?
            }
            "--frames" => {
                cli.frames = Some(
                    args.next()
                        .ok_or("missing --frames")?
                        .parse()
                        .map_err(|_| "bad --frames")?,
                )
            }
            "--preset" => {
                let name = args.next().ok_or("missing --preset")?;
                cli.preset = TerrariumDemoPreset::parse(&name)
                    .ok_or_else(|| format!("unknown preset: {name}"))?;
            }
            "--mode" => {
                cli.mode = match args.next().ok_or("missing --mode")?.as_str() {
                    "3d" | "true3d" | "raster" => ViewMode::True3D,
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
            "--help" | "-h" => {
                print_usage(program_name);
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }
    if cli.checkpoint_in.is_some() && cli.archive_in.is_some() {
        return Err("use either --load-checkpoint or --load-archive, not both".into());
    }
    if cli.archive_in.is_some() && cli.checkpoint_out.is_some() {
        return Err("cannot emit a checkpoint when inspecting an archive".into());
    }
    Ok(cli)
}

fn load_or_create_world(cli: &Cli) -> Result<(TerrariumWorld, TerrariumDemoPreset), String> {
    let mut world = if let Some(path) = cli.checkpoint_in.as_deref() {
        TerrariumWorld::load_checkpoint(path)
            .map_err(|e| format!("failed to load terrarium checkpoint: {e}"))?
    } else {
        let seed_provenance = resolve_seed_provenance(cli.seed, "terrarium_ascii");
        let mut world =
            TerrariumWorld::demo_preset(seed_provenance.seed, !cli.cpu_substrate, cli.preset)
                .map_err(|e| format!("failed to create world: {e}"))?;
        world.set_seed_provenance(seed_provenance);
        world
    };
    apply_world_name_assignments(&mut world, &cli.name_assignments)?;
    let present_preset =
        TerrariumDemoPreset::infer_from_config(&world.config).unwrap_or(cli.preset);
    Ok((world, present_preset))
}

fn print_world_reports(world: &TerrariumWorld, cli: &Cli) -> Result<(), String> {
    println!("{}", format_world_summary(world));
    if cli.list_organisms {
        println!();
        println!("{}", format_world_organism_listing(world));
    }
    for lineage_id in &cli.lineage_ids {
        println!();
        println!("{}", format_world_lineage(world, *lineage_id)?);
    }
    Ok(())
}

fn print_archive_reports(archive: &TerrariumWorldArchive, cli: &Cli) -> Result<(), String> {
    println!("{}", format_archive_summary(archive));
    if cli.list_organisms {
        println!();
        println!("{}", format_archive_organism_listing(archive));
    }
    for lineage_id in &cli.lineage_ids {
        println!();
        println!("{}", format_archive_lineage(archive, *lineage_id)?);
    }
    Ok(())
}

fn save_world_outputs(world: &mut TerrariumWorld, cli: &Cli) -> Result<(), String> {
    if let Some(path) = cli.checkpoint_out.as_deref() {
        world
            .save_checkpoint(path)
            .map_err(|error| format!("failed to save terrarium checkpoint: {error}"))?;
        println!("saved terrarium checkpoint to {path}");
    }
    if let Some(path) = cli.archive_out.as_deref() {
        world
            .save_archive(path)
            .map_err(|error| format!("failed to save terrarium archive: {error}"))?;
        println!("saved terrarium archive to {path}");
    }
    Ok(())
}

fn inspect_archive(cli: &Cli) -> Result<ExitCode, String> {
    let path = cli
        .archive_in
        .as_deref()
        .ok_or_else(|| "archive inspection requires --load-archive".to_string())?;
    let mut archive = TerrariumWorldArchive::load_from_path(path)?;
    apply_archive_name_assignments(&mut archive, &cli.name_assignments)?;
    print_archive_reports(&archive, cli)?;
    if let Some(path) = cli.archive_out.as_deref() {
        archive.save_to_path(path)?;
        println!("saved terrarium archive to {path}");
    }
    Ok(ExitCode::SUCCESS)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

pub fn run_terminal_3d(program_name: &str) -> ExitCode {
    let cli = match parse_args(program_name) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            print_usage(program_name);
            return ExitCode::FAILURE;
        }
    };
    if cli.archive_in.is_some() {
        return match inspect_archive(&cli) {
            Ok(code) => code,
            Err(error) => {
                eprintln!("{error}");
                ExitCode::FAILURE
            }
        };
    }
    let (mut world, present_preset) = match load_or_create_world(&cli) {
        Ok(result) => result,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    if cli.query_only || cli.list_organisms || !cli.lineage_ids.is_empty() {
        if let Err(error) = print_world_reports(&world, &cli) {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
        if cli.query_only {
            if let Err(error) = save_world_outputs(&mut world, &cli) {
                eprintln!("{error}");
                return ExitCode::FAILURE;
            }
            return ExitCode::SUCCESS;
        }
    }
    let (term_width, term_height) = terminal_size();
    let mut buf = ScreenBuffer::new(term_width, term_height);
    let mut scratch = RenderScratch::new(term_width, term_height);

    let frame_budget = if cli.fps > 0 {
        Some(Duration::from_secs_f64(1.0 / cli.fps as f64))
    } else {
        None
    };

    let mut snapshot_history: VecDeque<TerrariumWorldSnapshot> = VecDeque::with_capacity(256);
    let mut vs = ViewState::new(cli.mode, cli.show_minimap, present_preset);
    let mut last_snapshot = world.snapshot();
    push_snapshot_history(&mut snapshot_history, last_snapshot.clone());

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
    let mut sim_step_credit = 0.0f32;
    let mut next_frame_deadline = Instant::now();

    loop {
        vs.clear_expired_message();

        // Handle keyboard input (non-blocking)
        if let Some(ref rx) = key_rx {
            while let Ok(key) = rx.try_recv() {
                match key {
                    KeyInput::Char('q') => {
                        quit = true;
                    }
                    KeyInput::Char('h') => {
                        vs.show_help = !vs.show_help;
                    }
                    KeyInput::Char('\t') => {
                        vs.cycle_mode();
                    }
                    KeyInput::Char('m') => {
                        vs.show_minimap = !vs.show_minimap;
                    }
                    KeyInput::Char(' ') => {
                        vs.paused = !vs.paused;
                    }
                    KeyInput::Char('r') => {
                        vs.camera_x = 0;
                        vs.camera_y = 0;
                        vs.zoom = 1.0;
                    }
                    KeyInput::Char('w') | KeyInput::Up => {
                        vs.camera_y += 2;
                    }
                    KeyInput::Char('s') | KeyInput::Down => {
                        vs.camera_y -= 2;
                    }
                    KeyInput::Char('a') | KeyInput::Left => {
                        vs.camera_x += 2;
                    }
                    KeyInput::Char('d') | KeyInput::Right => {
                        vs.camera_x -= 2;
                    }
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
                        vs.set_message(
                            format!("Time warp {:.2}x", vs.time_warp),
                            Duration::from_secs(2),
                        );
                    }
                    KeyInput::Char(']') => {
                        vs.time_warp = (vs.time_warp * 2.0).min(8.0);
                        vs.set_message(
                            format!("Time warp {:.2}x", vs.time_warp),
                            Duration::from_secs(2),
                        );
                    }
                    KeyInput::Char('e') => vs.set_message(
                        "Scenario injection is not wired in the terminal renderer yet",
                        Duration::from_secs(2),
                    ),
                    KeyInput::Char('t') => vs.set_message(
                        "Scenario injection is not wired in the terminal renderer yet",
                        Duration::from_secs(2),
                    ),
                    KeyInput::Char('p') => vs.set_message(
                        "Live spawning is not wired in the terminal renderer yet",
                        Duration::from_secs(2),
                    ),
                    KeyInput::Char('x') => vs.set_message(
                        "Live spawning is not wired in the terminal renderer yet",
                        Duration::from_secs(2),
                    ),
                    // Toggle overlays (C=chemistry, G=grid, V=vitality bars)
                    KeyInput::Char('c') => {
                        vs.show_chemistry = !vs.show_chemistry;
                        vs.set_message(
                            format!(
                                "Chemistry overlay {}",
                                if vs.show_chemistry { "ON" } else { "OFF" }
                            ),
                            Duration::from_secs(1),
                        );
                    }
                    KeyInput::Char('g') => {
                        vs.show_grid = !vs.show_grid;
                        vs.set_message(
                            format!("Grid {}", if vs.show_grid { "ON" } else { "OFF" }),
                            Duration::from_secs(1),
                        );
                    }
                    KeyInput::Char('v') => {
                        vs.show_vitality = !vs.show_vitality;
                        vs.set_message(
                            format!(
                                "Vitality overlay {}",
                                if vs.show_vitality { "ON" } else { "OFF" }
                            ),
                            Duration::from_secs(1),
                        );
                    }
                    // Snapshot save (L=load/save snapshot)
                    KeyInput::Char('l') => {
                        let json = serde_json::to_string_pretty(&last_snapshot).unwrap_or_default();
                        let _ = std::fs::write("terrarium_snapshot.json", json);
                        vs.set_message("Saved terrarium_snapshot.json", Duration::from_secs(2));
                    }
                    // Skip frames (K=skip 100 frames)
                    KeyInput::Char('k') => {
                        for _ in 0..100 {
                            if world.step_frame().is_err() {
                                break;
                            }
                            frame_idx += 1;
                        }
                        last_snapshot = world.snapshot();
                        push_snapshot_history(&mut snapshot_history, last_snapshot.clone());
                        vs.set_message("Skipped 100 simulation frames", Duration::from_secs(1));
                    }
                    // Info mode (I=toggle entity info)
                    KeyInput::Char('i') => {
                        vs.show_info = !vs.show_info;
                        vs.set_message(
                            format!("Info {}", if vs.show_info { "ON" } else { "OFF" }),
                            Duration::from_secs(1),
                        );
                    }
                    KeyInput::Char('1') => {
                        vs.mode = ViewMode::Isometric;
                    }
                    KeyInput::Char('2') => {
                        vs.mode = ViewMode::TopDown;
                    }
                    KeyInput::Char('3') => {
                        vs.mode = ViewMode::Heatmap;
                    }
                    KeyInput::Char('4') => {
                        vs.mode = ViewMode::Dashboard;
                    }
                    KeyInput::Char('5') => {
                        vs.mode = ViewMode::Split;
                    }
                    KeyInput::Char('6') => {
                        vs.mode = ViewMode::True3D;
                    }
                    _ => {}
                }
            }
        }

        if quit {
            break;
        }

        // Step simulation (unless paused). Time warp spends simulation credits
        // without forcing extra terminal redraws.
        if !vs.paused {
            sim_step_credit += vs.time_warp.max(0.0);
            let mut steps_this_render = 0usize;
            while sim_step_credit >= 1.0 && steps_this_render < 16 {
                if let Err(e) = world.step_frame() {
                    eprintln!("\x1b[?25h\x1b[0m\nstep failed: {e}");
                    restore_terminal();
                    return ExitCode::FAILURE;
                }
                sim_step_credit -= 1.0;
                steps_this_render += 1;
                frame_idx += 1;
            }
            if steps_this_render > 0 {
                last_snapshot = world.snapshot();
                push_snapshot_history(&mut snapshot_history, last_snapshot.clone());
            }
        }

        render_active_view(
            &mut buf,
            &mut scratch,
            &world,
            &last_snapshot,
            &snapshot_history,
            frame_idx,
            &vs,
        );

        fps_frames += 1;
        if fps_timer.elapsed().as_secs_f32() >= 1.0 {
            last_fps = fps_frames as f32 / fps_timer.elapsed().as_secs_f32();
            fps_frames = 0;
            fps_timer = Instant::now();
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
        draw_status_banner(&mut buf, &vs);

        buf.render_into(cli.use_color, &mut scratch.terminal_out);
        print!("{}", scratch.terminal_out);
        let _ = io::stdout().flush();

        if let Some(max) = cli.frames {
            if frame_idx >= max {
                break;
            }
        }

        if let Some(target) = frame_budget {
            let sleep_target = if vs.paused {
                target.min(Duration::from_millis(50))
            } else {
                target
            };
            next_frame_deadline += sleep_target;
            let now = Instant::now();
            if next_frame_deadline > now {
                thread::sleep(next_frame_deadline - now);
            } else {
                next_frame_deadline = now;
            }
        } else if vs.paused {
            thread::sleep(Duration::from_millis(20));
        }
    }

    // Restore terminal
    restore_terminal();
    print!("\x1b[?25h\x1b[0m\n");
    let _ = io::stdout().flush();
    if let Err(error) = save_world_outputs(&mut world, &cli) {
        eprintln!("{error}");
        return ExitCode::FAILURE;
    }

    println!("Terrarium exited after {frame_idx} frames.");
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    run_terminal_3d("terrarium_ascii")
}
