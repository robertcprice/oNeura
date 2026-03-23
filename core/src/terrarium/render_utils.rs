use rand::{rngs::StdRng, Rng};
use rand_distr::StandardNormal;
use std::hash::{Hash, Hasher};

use crate::constants::clamp;
use crate::drosophila::BodyState;
use crate::plant_cellular::PlantTissue;
use crate::terrarium_render::{
    TerrariumDynamicBatchKind, TerrariumFlyPartKind, TerrariumFruitPartKind,
    TerrariumPbrMaterialRender, TerrariumSeedPartKind,
};
use crate::WholeCellChemistrySite;

use super::TerrariumWorldConfig;

pub(crate) const RENDER_CELL_WORLD_SIZE: f32 = 0.34;
pub(crate) const RENDER_TILE_BASE_HEIGHT: f32 = 0.08;
pub(crate) const RENDER_TILE_HEIGHT_SCALE: f32 = 0.28;
pub(crate) const RENDER_ALTITUDE_SCALE: f32 = 0.22;
pub(crate) const RENDER_OVERLAY_MAX_PUFFS: usize = 42;
pub(crate) const RENDER_OVERLAY_MIN_NORM: f32 = 0.62;
pub(crate) const RENDER_SUBSTRATE_SURFACE_EPSILON: f32 = 0.006;
pub(crate) const SUBSTRATE_BATCH_CHUNK_XY: usize = 8;
pub(crate) const SUBSTRATE_BATCH_CHUNK_Z: usize = 4;
pub(crate) const DYNAMIC_BATCH_CHUNK_WORLD_XZ: f32 = RENDER_CELL_WORLD_SIZE * 5.0;
pub(crate) const DYNAMIC_BATCH_CHUNK_WORLD_Y: f32 = RENDER_ALTITUDE_SCALE * 5.0;

pub(crate) fn idx2(width: usize, x: usize, y: usize) -> usize {
    y * width + x
}

pub(crate) fn idx3(width: usize, height: usize, x: usize, y: usize, z: usize) -> usize {
    (z * height + y) * width + x
}

pub(crate) fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().copied().sum::<f32>() / values.len() as f32
    }
}

pub(crate) fn sample_normal(rng: &mut StdRng, sigma: f32) -> f32 {
    let z: f32 = rng.sample(StandardNormal);
    z * sigma
}

pub(crate) fn integrate_displacement(
    offset: &mut f32,
    velocity: &mut f32,
    force: f32,
    stiffness: f32,
    damping: f32,
    mass: f32,
    dt: f32,
    limit: f32,
) {
    let effective_mass = mass.max(1.0e-4);
    let restoring = -*offset * stiffness;
    let drag = -*velocity * damping;
    let accel = (force + restoring + drag) / effective_mass;
    *velocity += accel * dt;
    *offset = clamp(*offset + *velocity * dt, -limit, limit);
    if offset.abs() >= limit * 0.999 {
        *velocity *= 0.4;
    }
}

pub(crate) fn render_cell_center_world(width: usize, height: usize, x: f32, y: f32) -> [f32; 3] {
    [
        (x - width as f32 * 0.5) * RENDER_CELL_WORLD_SIZE,
        0.0,
        (y - height as f32 * 0.5) * RENDER_CELL_WORLD_SIZE,
    ]
}

pub(crate) fn render_world_to_cell_xy(width: usize, height: usize, world_x: f32, world_z: f32) -> [f32; 2] {
    [
        world_x / RENDER_CELL_WORLD_SIZE + width as f32 * 0.5,
        world_z / RENDER_CELL_WORLD_SIZE + height as f32 * 0.5,
    ]
}

pub(crate) fn sample_topdown_field_clamped(field: &[f32], width: usize, height: usize, x: f32, y: f32) -> f32 {
    if field.is_empty() || width == 0 || height == 0 {
        return 0.0;
    }
    let x = x.clamp(0.0, width as f32 - 1.0);
    let y = y.clamp(0.0, height as f32 - 1.0);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;
    let v00 = field[idx2(width, x0, y0)];
    let v10 = field[idx2(width, x1, y0)];
    let v01 = field[idx2(width, x0, y1)];
    let v11 = field[idx2(width, x1, y1)];
    let vx0 = v00 + (v10 - v00) * tx;
    let vx1 = v01 + (v11 - v01) * tx;
    vx0 + (vx1 - vx0) * ty
}

pub(crate) fn terrain_surface_y_at_cell_xy(
    width: usize,
    height: usize,
    terrain: &[f32],
    terrain_min: f32,
    terrain_inv: f32,
    x: f32,
    y: f32,
) -> f32 {
    render_top_surface_y(normalize_unit(
        sample_topdown_field_clamped(terrain, width, height, x, y),
        terrain_min,
        terrain_inv,
    ))
}

pub(crate) fn fly_translation_world_from_body(
    config: &TerrariumWorldConfig,
    terrain: &[f32],
    terrain_min: f32,
    terrain_inv: f32,
    body: &BodyState,
) -> ([f32; 3], f32) {
    let x = body.x.clamp(0.0, config.width as f32 - 0.01);
    let y = body.y.clamp(0.0, config.height as f32 - 0.01);
    let ground_y = terrain_surface_y_at_cell_xy(
        config.width,
        config.height,
        terrain,
        terrain_min,
        terrain_inv,
        x,
        y,
    );
    let mut translation = render_cell_center_world(config.width, config.height, x, y);
    let base_clearance = if body.is_flying { 0.18 } else { 0.08 };
    translation[1] = ground_y + base_clearance + body.z.max(0.0) * RENDER_ALTITUDE_SCALE;
    (translation, ground_y)
}

pub(crate) fn fly_body_state_from_world_translation(
    config: &TerrariumWorldConfig,
    terrain: &[f32],
    terrain_min: f32,
    terrain_inv: f32,
    translation_world: [f32; 3],
    is_flying: bool,
) -> (f32, f32, f32, f32) {
    let [x, y] = render_world_to_cell_xy(
        config.width,
        config.height,
        translation_world[0],
        translation_world[2],
    );
    let x = x.clamp(0.0, config.width as f32 - 0.01);
    let y = y.clamp(0.0, config.height as f32 - 0.01);
    let ground_y = terrain_surface_y_at_cell_xy(
        config.width,
        config.height,
        terrain,
        terrain_min,
        terrain_inv,
        x,
        y,
    );
    let base_clearance = if is_flying { 0.18 } else { 0.08 };
    let z = ((translation_world[1] - ground_y - base_clearance) / RENDER_ALTITUDE_SCALE).max(0.0);
    (x, y, z, ground_y)
}

pub(crate) fn render_top_surface_y(terrain_norm: f32) -> f32 {
    RENDER_TILE_BASE_HEIGHT + terrain_norm * RENDER_TILE_HEIGHT_SCALE - 0.02
}

pub(crate) fn render_substrate_voxel_world_size(voxel_size_mm: f32, cell_size_mm: f32) -> [f32; 3] {
    let size_ratio = (voxel_size_mm / cell_size_mm.max(1.0e-3)).clamp(0.5, 1.25);
    [
        RENDER_CELL_WORLD_SIZE * 0.62 * size_ratio,
        (RENDER_ALTITUDE_SCALE * 0.72 * size_ratio).clamp(0.06, 0.22),
        RENDER_CELL_WORLD_SIZE * 0.62 * size_ratio,
    ]
}

pub(crate) fn render_substrate_voxel_translation_world(
    width: usize,
    height: usize,
    x: f32,
    y: f32,
    z: usize,
    ground_y: f32,
    voxel_height_world: f32,
) -> [f32; 3] {
    let mut translation = render_cell_center_world(width, height, x, y);
    translation[1] =
        ground_y + RENDER_SUBSTRATE_SURFACE_EPSILON - voxel_height_world * (z as f32 + 0.5);
    translation
}

pub(crate) fn render_atmosphere_cell_translation_world(
    width: usize,
    height: usize,
    x: f32,
    y: f32,
    z: usize,
    ground_y: f32,
) -> [f32; 3] {
    let mut translation = render_cell_center_world(width, height, x, y);
    translation[1] = ground_y + 0.10 + (z as f32 + 0.5) * RENDER_ALTITUDE_SCALE;
    translation
}

pub(crate) fn decode_idx3(width: usize, height: usize, idx: usize) -> (usize, usize, usize) {
    let plane = width * height;
    let z = idx / plane;
    let rem = idx % plane;
    let y = rem / width;
    let x = rem % width;
    (x, y, z)
}

pub(crate) fn field_min_inv(values: &[f32]) -> (f32, f32) {
    let mut min_value = f32::INFINITY;
    let mut max_value = f32::NEG_INFINITY;
    for value in values {
        min_value = min_value.min(*value);
        max_value = max_value.max(*value);
    }
    let span = (max_value - min_value).max(1.0e-6);
    (min_value, 1.0 / span)
}

pub(crate) fn normalize_unit(value: f32, min_value: f32, inv_span: f32) -> f32 {
    ((value - min_value) * inv_span).clamp(0.0, 1.0)
}

pub(crate) fn strongest_render_cells(values: &[f32], max_points: usize, min_norm: f32) -> Vec<(usize, f32)> {
    let (min_value, inv_span) = field_min_inv(values);
    let mut ranked = values
        .iter()
        .enumerate()
        .map(|(idx, value)| (idx, normalize_unit(*value, min_value, inv_span)))
        .filter(|(_, norm)| *norm >= min_norm)
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked.truncate(max_points);
    ranked
}

pub(crate) fn render_pressure_response(
    idx: usize,
    air_pressure: &[f32],
    pressure_mean: f32,
    pressure_min: f32,
    pressure_inv: f32,
) -> (f32, f32) {
    let local_t = normalize_unit(air_pressure[idx], pressure_min, pressure_inv);
    let mean_t = normalize_unit(pressure_mean, pressure_min, pressure_inv);
    let pressure_bias = clamp(local_t - mean_t, -1.0, 1.0);
    (pressure_bias.abs(), pressure_bias)
}

pub(crate) fn rgb(r: f32, g: f32, b: f32) -> [f32; 3] {
    [r, g, b]
}

pub(crate) fn rgba(r: f32, g: f32, b: f32, a: f32) -> [f32; 4] {
    [r, g, b, a]
}

pub(crate) fn lerp_rgb(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

pub(crate) fn with_alpha(color: [f32; 3], alpha: f32) -> [f32; 4] {
    [color[0], color[1], color[2], alpha]
}

pub(crate) fn pbr_material(
    base_color_rgba: [f32; 4],
    emissive_rgb: [f32; 3],
    metallic: f32,
    perceptual_roughness: f32,
    reflectance: f32,
    alpha_blend: bool,
    double_sided: bool,
) -> TerrariumPbrMaterialRender {
    TerrariumPbrMaterialRender {
        base_color_rgba,
        emissive_rgb,
        metallic,
        perceptual_roughness,
        reflectance,
        alpha_blend,
        double_sided,
        shader_atmosphere_rgba: [0.40, 0.50, 0.50, 0.50],
        shader_dynamics_rgba: [0.35, 0.35, 0.10, 0.0],
        shader_flags: 0,
    }
}

pub(crate) const TERRARIUM_SHADER_FLAG_SUBSTRATE: u32 = 1 << 0;
pub(crate) const TERRARIUM_SHADER_FLAG_PLUME: u32 = 1 << 1;
pub(crate) const TERRARIUM_SHADER_FLAG_FLUID: u32 = 1 << 2;
pub(crate) const TERRARIUM_SHADER_FLAG_MICROBE: u32 = 1 << 3;
pub(crate) const TERRARIUM_SHADER_FLAG_PLANT: u32 = 1 << 4;
pub(crate) const TERRARIUM_SHADER_FLAG_SEED: u32 = 1 << 5;
pub(crate) const TERRARIUM_SHADER_FLAG_FRUIT: u32 = 1 << 6;
pub(crate) const TERRARIUM_SHADER_FLAG_FLY: u32 = 1 << 7;
pub(crate) const TERRARIUM_SHADER_FLAG_DEBUG_OVERLAY: u32 = 1 << 8;
pub(crate) const TERRARIUM_RENDER_ID_SUBSTRATE_VOXEL: u8 = 1;
pub(crate) const TERRARIUM_RENDER_ID_MICROBE: u8 = 2;
pub(crate) const TERRARIUM_RENDER_ID_MICROBE_PACKET: u8 = 3;
pub(crate) const TERRARIUM_RENDER_ID_MICROBE_SITE: u8 = 4;
pub(crate) const TERRARIUM_RENDER_ID_WATER: u8 = 5;
pub(crate) const TERRARIUM_RENDER_ID_PLANT: u8 = 6;
pub(crate) const TERRARIUM_RENDER_ID_SEED: u8 = 7;
pub(crate) const TERRARIUM_RENDER_ID_FRUIT: u8 = 8;
pub(crate) const TERRARIUM_RENDER_ID_PLUME: u8 = 9;
pub(crate) const TERRARIUM_RENDER_ID_FLY: u8 = 10;
pub(crate) const TERRARIUM_RENDER_ID_SUBSTRATE_BATCH: u8 = 11;
pub(crate) const TERRARIUM_RENDER_ID_DYNAMIC_BATCH: u8 = 12;
pub(crate) const TERRARIUM_RENDER_SLOT_MICROBE_BODY: u16 = 0xFF01;
pub(crate) const TERRARIUM_RENDER_SLOT_MICROBE_PACKET: u16 = 0xFF02;
pub(crate) const TERRARIUM_RENDER_SLOT_PLANT_STEM: u16 = 0xFF11;
pub(crate) const TERRARIUM_RENDER_SLOT_PLANT_CANOPY: u16 = 0xFF12;
pub(crate) const TERRARIUM_RENDER_SLOT_FLY_LIGHT: u16 = 0xFFF0;

pub(crate) fn with_shader_response(
    mut material: TerrariumPbrMaterialRender,
    shader_atmosphere_rgba: [f32; 4],
    shader_dynamics_rgba: [f32; 4],
    shader_flags: u32,
) -> TerrariumPbrMaterialRender {
    material.shader_atmosphere_rgba = shader_atmosphere_rgba;
    material.shader_dynamics_rgba = shader_dynamics_rgba;
    material.shader_flags = shader_flags;
    material
}

pub(crate) fn terrarium_render_id(class: u8, primary: u64, secondary: u16) -> u64 {
    ((class as u64) << 56) | ((primary & 0x00ff_ffff_ffff) << 16) | secondary as u64
}

pub(crate) fn terrarium_render_id_class(render_id: u64) -> u8 {
    ((render_id >> 56) & 0xff) as u8
}

pub(crate) fn terrarium_grid_render_primary(x: usize, y: usize, z: usize) -> u64 {
    ((x as u64) << 32) | ((y as u64) << 16) | z as u64
}

pub(crate) fn terrarium_render_child_id(render_id: u64, child_slot: u16) -> u64 {
    (render_id & !0xffff) | child_slot as u64
}

pub(crate) fn terrarium_substrate_batch_render_id(
    chunk_x: usize,
    chunk_y: usize,
    chunk_z: usize,
    material_state_key: u64,
    hide_on_cutaway: bool,
) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    chunk_x.hash(&mut hasher);
    chunk_y.hash(&mut hasher);
    chunk_z.hash(&mut hasher);
    material_state_key.hash(&mut hasher);
    hide_on_cutaway.hash(&mut hasher);
    terrarium_render_id(TERRARIUM_RENDER_ID_SUBSTRATE_BATCH, hasher.finish(), 0)
}

pub(crate) fn terrarium_dynamic_batch_render_id(
    kind: TerrariumDynamicBatchKind,
    chunk_x: i32,
    chunk_y: i32,
    chunk_z: i32,
    material_state_key: u64,
) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    kind.hash(&mut hasher);
    chunk_x.hash(&mut hasher);
    chunk_y.hash(&mut hasher);
    chunk_z.hash(&mut hasher);
    material_state_key.hash(&mut hasher);
    terrarium_render_id(TERRARIUM_RENDER_ID_DYNAMIC_BATCH, hasher.finish(), 0)
}

pub(crate) fn terrarium_tissue_render_slot(tissue: PlantTissue) -> u16 {
    match tissue {
        PlantTissue::Leaf => 1,
        PlantTissue::Stem => 2,
        PlantTissue::Root => 3,
        PlantTissue::Meristem => 4,
    }
}

pub(crate) fn terrarium_site_render_slot(site: WholeCellChemistrySite) -> u16 {
    match site {
        WholeCellChemistrySite::Cytosol => 1,
        WholeCellChemistrySite::AtpSynthaseBand => 2,
        WholeCellChemistrySite::RibosomeCluster => 3,
        WholeCellChemistrySite::SeptumRing => 4,
        WholeCellChemistrySite::ChromosomeTrack => 5,
    }
}

pub(crate) fn terrarium_seed_part_render_slot(kind: TerrariumSeedPartKind) -> u16 {
    match kind {
        TerrariumSeedPartKind::Coat => 1,
        TerrariumSeedPartKind::Endosperm => 2,
        TerrariumSeedPartKind::Radicle => 3,
        TerrariumSeedPartKind::CotyledonLeft => 4,
        TerrariumSeedPartKind::CotyledonRight => 5,
    }
}

pub(crate) fn terrarium_fruit_part_render_slot(kind: TerrariumFruitPartKind) -> u16 {
    match kind {
        TerrariumFruitPartKind::Skin => 1,
        TerrariumFruitPartKind::Pulp => 2,
        TerrariumFruitPartKind::Core => 3,
        TerrariumFruitPartKind::Stem => 4,
    }
}

pub(crate) fn terrarium_fly_part_render_slot(kind: TerrariumFlyPartKind) -> u16 {
    match kind {
        TerrariumFlyPartKind::Thorax => 1,
        TerrariumFlyPartKind::Head => 2,
        TerrariumFlyPartKind::Abdomen => 3,
        TerrariumFlyPartKind::Proboscis => 4,
        TerrariumFlyPartKind::WingLeft => 5,
        TerrariumFlyPartKind::WingRight => 6,
    }
}
