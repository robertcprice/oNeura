// ── Heuristic audit (2026-03-20, updated Phase 4) ─────────────────────────
//
// TRANSITIVELY_DESCRIPTOR_DERIVED (render inputs now carry quantum-
// descriptor-enriched geochemistry; chemistry probe values flow from
// descriptor-informed species profiles):
//   874-907   Chemistry probe aggregation multipliers — the underlying
//             species concentrations are driven by descriptor-derived
//             reactivity, thermodynamics, and activity scales
//
// DERIVED_FROM_ORGANISM_BIOLOGY (anatomy emerges from organism model):
//   1255-1325 Fly visual scaling — thorax/abdomen/wing proportions from
//             Drosophila body model (segment counts, wing area)
//   1335-1377 Earthworm visual scaling — segment count from body model
//
// CALIBRATION_WEIGHTS (landscape-level heuristics on derived inputs):
//   365-397   Basin scoring / cut formula (depth exponents, coverage weights)
//   433-471   Grass cover falloff, flood waterline quantile
//   561       Water budget formula (base + raw + raw²)
//   1462-1555 Soil surface class thresholds and RGB tuples
//
// IRREDUCIBLE_PRESENTATION (art/color/geometry decisions):
//   10-15     Physical constants + voxel geometry
//   240-272   Taxonomy color palettes
//   651-663   Voxel color blending thresholds
//   939-1002  Terrain RGB formula (dry/wet/rich base colors + chemistry tint)
//   1012-1203 Plant/fruit atmospheric presentation over explicit pigment state
//   1258-1332 Water visual response (lunar, tidal, chemistry tinting)
//
// ──────────────────────────────────────────────────────────────────────────

use crate::botany::BotanicalGrowthForm;
use crate::botany::visual_phenotype::MolecularVisualState;
use crate::constants::clamp;
use crate::drosophila::BodyState;
use crate::soil_fauna::{EarthwormBody, NematodeKind};
use crate::terrarium::packet::GenotypePacketEcology;
use std::collections::VecDeque;

use super::emergent_color;
use super::{TerrariumAtmosphereFrame, TerrariumSpecies, TerrariumWorld};

const ATMOS_PRESSURE_BASELINE_KPA: f32 = 101.325;
pub const TERRAIN_VOXEL_HEIGHT: f32 = 0.6;
pub const TERRAIN_VOXEL_BASE_Y: f32 = -0.2;
pub const TERRAIN_SURFACE_RELIEF_MIN: f32 = 0.0;
pub const TERRAIN_SURFACE_RELIEF_MAX: f32 = 1.1;
pub const TERRAIN_SURFACE_LEVELS: usize = 7;

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct TerrariumVisualAirSample {
    pub temperature_c: f32,
    pub humidity: f32,
    pub pressure_kpa: f32,
    pub pressure_delta_kpa: f32,
    pub wind_x: f32,
    pub wind_y: f32,
    pub wind_z: f32,
    pub wind_speed: f32,
}

#[derive(
    Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash,
)]
#[serde(rename_all = "snake_case")]
pub enum TerrariumVisualProvenance {
    RuntimeProjection,
    CoarseFieldProjection,
    OwnedRegionProjection,
    #[default]
    PresentationFallback,
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct TerrariumTerrainVisualResponse {
    pub rgb: [f32; 3],
    #[serde(default)]
    pub provenance: TerrariumVisualProvenance,
    #[serde(default = "default_visual_uncertainty")]
    pub uncertainty: f32,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct TerrariumVisualChemistrySample {
    pub acidity: f32,
    pub alkalinity: f32,
    pub silicate: f32,
    pub carbonate: f32,
    pub ferric: f32,
    pub ferrous: f32,
    pub salinity: f32,
    pub clay: f32,
    pub redox: f32,
    pub clarity: f32,
}

impl Default for TerrariumVisualChemistrySample {
    fn default() -> Self {
        Self {
            acidity: 0.0,
            alkalinity: 0.0,
            silicate: 0.0,
            carbonate: 0.0,
            ferric: 0.0,
            ferrous: 0.0,
            salinity: 0.0,
            clay: 0.0,
            redox: 0.0,
            clarity: 1.0,
        }
    }
}

#[derive(
    Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash,
)]
pub enum TerrariumVoxelMaterialClass {
    #[default]
    Bedrock,
    Subsoil,
    Surface,
    Water,
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct TerrariumTerrainVoxelInstance {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub scale_y: f32,
    pub rgb: [f32; 3],
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TerrariumTerrainVoxelBatch {
    pub class: TerrariumVoxelMaterialClass,
    pub instances: Vec<TerrariumTerrainVoxelInstance>,
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct TerrariumPlantVisualResponse {
    pub stem_rgb: [f32; 3],
    pub leaf_rgb: [f32; 3],
    pub fruit_rgb: [f32; 3],
    pub sway_x: f32,
    pub sway_z: f32,
    pub lean_x: f32,
    pub lean_z: f32,
    pub vertical_offset: f32,
    pub canopy_scale: f32,
    #[serde(default)]
    pub provenance: TerrariumVisualProvenance,
    #[serde(default = "default_visual_uncertainty")]
    pub uncertainty: f32,
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct TerrariumFruitVisualResponse {
    pub skin_rgb: [f32; 3],
    pub stem_rgb: [f32; 3],
    pub tilt_x: f32,
    pub tilt_z: f32,
    pub vertical_offset: f32,
    pub sprite_scale: f32,
    #[serde(default)]
    pub provenance: TerrariumVisualProvenance,
    #[serde(default = "default_visual_uncertainty")]
    pub uncertainty: f32,
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct TerrariumWaterVisualResponse {
    pub rgb: [f32; 3],
    pub tilt_x: f32,
    pub tilt_z: f32,
    pub vertical_offset: f32,
    pub thickness_scale: f32,
    pub radius_x_cells: f32,
    pub radius_y_cells: f32,
    pub clarity: f32,
    #[serde(default)]
    pub provenance: TerrariumVisualProvenance,
    #[serde(default = "default_visual_uncertainty")]
    pub uncertainty: f32,
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct TerrariumWaterCycleInputs {
    pub lunar_phase: f32,
    pub moonlight: f32,
    pub tidal_moisture_factor: f32,
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct TerrariumSeedVisualResponse {
    pub shell_rgb: [f32; 3],
    pub accent_rgb: [f32; 3],
    pub tilt_x: f32,
    pub tilt_z: f32,
    pub vertical_offset: f32,
    pub sprite_scale: f32,
    #[serde(default)]
    pub provenance: TerrariumVisualProvenance,
    #[serde(default = "default_visual_uncertainty")]
    pub uncertainty: f32,
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct TerrariumFlyVisualResponse {
    pub body_rgb: [f32; 3],
    pub wing_rgb: [f32; 3],
    pub pitch: f32,
    pub roll: f32,
    pub wing_angle: f32,
    pub sprite_scale: f32,
    pub thorax_scale: f32,
    pub abdomen_scale: f32,
    pub head_scale: f32,
    pub wing_span: f32,
    pub wing_width: f32,
    pub leg_span: f32,
    pub proboscis_extension: f32,
    #[serde(default)]
    pub provenance: TerrariumVisualProvenance,
    #[serde(default = "default_visual_uncertainty")]
    pub uncertainty: f32,
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct TerrariumEarthwormVisualResponse {
    pub body_rgb: [f32; 3],
    pub clitellum_rgb: [f32; 3],
    pub segment_count: u8,
    pub length_scale: f32,
    pub thickness_scale: f32,
    pub curl: f32,
    pub yaw_rad: f32,
    pub height_offset: f32,
    pub activity: f32,
    #[serde(default)]
    pub provenance: TerrariumVisualProvenance,
    #[serde(default = "default_visual_uncertainty")]
    pub uncertainty: f32,
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct TerrariumNematodeVisualResponse {
    pub body_rgb: [f32; 3],
    pub head_rgb: [f32; 3],
    pub length_scale: f32,
    pub thickness_scale: f32,
    pub curl: f32,
    pub yaw_rad: f32,
    pub height_offset: f32,
    pub stylet_length_scale: f32,
    pub activity: f32,
    #[serde(default)]
    pub provenance: TerrariumVisualProvenance,
    #[serde(default = "default_visual_uncertainty")]
    pub uncertainty: f32,
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum TerrariumSoilSurfaceClass {
    #[default]
    Mineral,
    Humus,
    WetDetritus,
    MicrobialMat,
    NitrifierCrust,
    DenitrifierFilm,
    MycorrhizalPatch,
    EarthwormCast,
    NematodeBloom,
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct TerrariumSoilSurfaceVisualResponse {
    pub class: TerrariumSoilSurfaceClass,
    pub rgb: [f32; 3],
    pub accent_rgb: [f32; 3],
    pub density: f32,
    pub height_offset: f32,
    pub sprite_scale: f32,
    pub width_scale: f32,
    pub depth_scale: f32,
    pub thickness_scale: f32,
    pub yaw_rad: f32,
    #[serde(default)]
    pub provenance: TerrariumVisualProvenance,
    #[serde(default = "default_visual_uncertainty")]
    pub uncertainty: f32,
}

const fn default_visual_uncertainty() -> f32 {
    1.0
}

fn visual_uncertainty(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

fn lerp_rgb(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
    ]
}

fn quantize_channel(value: f32, levels: usize) -> f32 {
    let levels = levels.max(2);
    let step_count = (levels - 1) as f32;
    ((value.clamp(0.0, 1.0) * step_count).round() / step_count).clamp(0.0, 1.0)
}

pub fn quantize_rgb(rgb: [f32; 3], levels: usize) -> [f32; 3] {
    [
        quantize_channel(rgb[0], levels),
        quantize_channel(rgb[1], levels),
        quantize_channel(rgb[2], levels),
    ]
}

fn taxonomy_palette_accent(taxonomy_id: u32) -> [f32; 3] {
    // Emergent: base green from chlorophyll molecular optics (Mg-porphyrin CPK)
    // with species-specific variation from taxonomy_id hash → pigment ratio
    let base_green = emergent_color::emergent_leaf_color(0.75, 0.15, 0.10);

    // Species-specific pigment variation: hash taxonomy_id to get chlorophyll/
    // carotenoid ratio shift (different species have slightly different
    // pigment profiles — Sims & Gamon 2002).
    let hash = taxonomy_id.wrapping_mul(2654435761); // Knuth multiplicative hash
    let variation = (hash & 0xFF) as f32 / 255.0; // 0.0–1.0

    // Shift chlorophyll/carotenoid ratio based on species
    let chl = 0.65 + variation * 0.15; // 0.65–0.80
    let car = 0.10 + (1.0 - variation) * 0.10; // 0.10–0.20
    let ant = 0.05 + ((hash >> 8) & 0x7F) as f32 / 127.0 * 0.08; // 0.05–0.13

    let emergent = emergent_color::emergent_leaf_color(chl, car, ant);

    // Legacy palette for reference (kept for blend factor support)
    let legacy = match taxonomy_id {
        3750 => [0.34, 0.58, 0.18],
        23211 => [0.40, 0.58, 0.16],
        3760 => [0.22, 0.54, 0.18],
        42229 => [0.24, 0.50, 0.16],
        2711 => [0.20, 0.48, 0.24],
        2708 => [0.28, 0.52, 0.20],
        15368 => [0.44, 0.58, 0.18],
        3702 => [0.26, 0.52, 0.22],
        _ => base_green,
    };

    // Always return emergent (blend controlled at call site)
    let _ = legacy; // retained for future blend support
    emergent
}

fn taxonomy_fruit_palette(taxonomy_id: u32) -> [f32; 3] {
    // Emergent: ripe fruit color from carotenoid/anthocyanin CPK composition
    // Species-specific anthocyanin expression determines red vs yellow fruit
    let hash = taxonomy_id.wrapping_mul(2654435761);
    let anthocyanin_expr = ((hash >> 4) & 0xFF) as f32 / 255.0; // 0.0–1.0
    emergent_color::emergent_fruit_color(0.85, anthocyanin_expr)
}

pub fn rgb_to_u8(rgb: [f32; 3]) -> (u8, u8, u8) {
    (
        (rgb[0].clamp(0.0, 1.0) * 255.0).round() as u8,
        (rgb[1].clamp(0.0, 1.0) * 255.0).round() as u8,
        (rgb[2].clamp(0.0, 1.0) * 255.0).round() as u8,
    )
}

pub fn mean_field(field: &[f32], fallback: f32) -> f32 {
    if field.is_empty() {
        fallback
    } else {
        field.iter().sum::<f32>() / field.len() as f32
    }
}

pub fn mean_wind_speed(atmosphere: &TerrariumAtmosphereFrame) -> f32 {
    let len = atmosphere
        .wind_x
        .len()
        .min(atmosphere.wind_y.len())
        .min(atmosphere.wind_z.len());
    if len == 0 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..len {
        let wx = atmosphere.wind_x[i];
        let wy = atmosphere.wind_y[i];
        let wz = atmosphere.wind_z[i];
        sum += (wx * wx + wy * wy + wz * wz).sqrt();
    }
    sum / len as f32
}

fn neighborhood_water_stats(
    width: usize,
    height: usize,
    field: &[f32],
    x: usize,
    y: usize,
) -> (f32, f32, f32) {
    let mut weighted_sum = 0.0f32;
    let mut weighted_total = 0.0f32;
    let mut local_max = 0.0f32;
    let mut occupied_weight = 0.0f32;
    for dy in -1isize..=1 {
        for dx in -1isize..=1 {
            let nx = x as isize + dx;
            let ny = y as isize + dy;
            if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                continue;
            }
            let weight = if dx == 0 && dy == 0 {
                1.0
            } else if dx == 0 || dy == 0 {
                0.65
            } else {
                0.40
            };
            let sample = field[ny as usize * width + nx as usize].clamp(0.0, 1.0);
            weighted_sum += sample * weight;
            weighted_total += weight;
            local_max = local_max.max(sample);
            if sample > 0.18 {
                occupied_weight += weight;
            }
        }
    }
    if weighted_total <= 0.0 {
        (0.0, 0.0, 0.0)
    } else {
        (
            weighted_sum / weighted_total,
            local_max,
            occupied_weight / weighted_total,
        )
    }
}

pub fn open_water_basin_mask(width: usize, height: usize, water_mask: &[f32]) -> Vec<f32> {
    if width == 0 || height == 0 || water_mask.len() != width * height {
        return Vec::new();
    }
    let mut basins = vec![0.0f32; water_mask.len()];
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let raw = water_mask[idx].clamp(0.0, 1.0);
            let (local_mean, local_max, support) =
                neighborhood_water_stats(width, height, water_mask, x, y);
            let core = ((raw - 0.18) / 0.60).clamp(0.0, 1.0);
            let pooled = ((local_mean - 0.18) / 0.36).clamp(0.0, 1.0);
            let edge = ((local_max - 0.28) / 0.52).clamp(0.0, 1.0) * support.sqrt();
            basins[idx] = clamp(core * 0.62 + pooled * 0.32 + edge * 0.20, 0.0, 1.0);
        }
    }
    basins
}

pub fn projected_surface_relief(
    width: usize,
    height: usize,
    soil_structure: &[f32],
    water_mask: &[f32],
) -> Vec<f32> {
    let open_water = open_water_basin_mask(width, height, water_mask);
    let mut projected = Vec::with_capacity(soil_structure.len());
    for (idx, relief) in soil_structure.iter().enumerate() {
        let water = water_mask.get(idx).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        let basin = open_water.get(idx).copied().unwrap_or(0.0);
        let shoreline = if width > 0 && height > 0 && idx < width * height {
            let x = idx % width;
            let y = idx / width;
            let (_, local_max, support) =
                neighborhood_water_stats(width, height, &open_water, x, y);
            (local_max * 0.55 + support * 0.45).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let basin_cut = 0.40 * water.powf(0.72)
            + 0.10 * water * water
            + 0.08 * basin.powf(0.86)
            + 0.05 * shoreline.powf(1.08) * (0.35 + water * 0.65) * (1.0 - basin * 0.35);
        projected.push((relief - basin_cut).max(0.0));
    }
    projected
}

pub fn quantized_surface_level(relief: f32) -> u8 {
    let normalized = ((relief - TERRAIN_SURFACE_RELIEF_MIN)
        / (TERRAIN_SURFACE_RELIEF_MAX - TERRAIN_SURFACE_RELIEF_MIN))
        .clamp(0.0, 1.0);
    (normalized * (TERRAIN_SURFACE_LEVELS.saturating_sub(1) as f32)).round() as u8
}

pub fn quantized_surface_height(relief: f32) -> f32 {
    TERRAIN_VOXEL_BASE_Y + (quantized_surface_level(relief) as f32 + 1.0) * TERRAIN_VOXEL_HEIGHT
}

fn grass_cover_projection(world: &TerrariumWorld) -> Vec<f32> {
    let width = world.config.width;
    let height = world.config.height;
    let mut cover = vec![0.0f32; width * height];

    for plant in &world.plants {
        if plant.morphology.growth_form != BotanicalGrowthForm::GrassClump {
            continue;
        }
        let radius = plant.canopy_radius_cells().clamp(1, 5) as isize;
        let vitality = plant.cellular.vitality().clamp(0.0, 1.0);
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let x = plant.x as isize + dx;
                let y = plant.y as isize + dy;
                if x < 0 || y < 0 || x >= width as isize || y >= height as isize {
                    continue;
                }
                let dist = ((dx * dx + dy * dy) as f32).sqrt();
                let falloff = (1.0 - dist / (radius as f32 + 0.5)).clamp(0.0, 1.0);
                let idx = y as usize * width + x as usize;
                cover[idx] = cover[idx].max(falloff * (0.45 + vitality * 0.55));
            }
        }
    }

    cover
}

fn flood_waterline(surface_relief: &[f32], water_mask: &[f32]) -> Option<f32> {
    if surface_relief.is_empty() || surface_relief.len() != water_mask.len() {
        return None;
    }
    let mut wet_heights = Vec::new();
    let mut wet_sum = 0.0f32;
    let mut wet_count = 0usize;
    for (idx, water) in water_mask.iter().enumerate() {
        let water = water.clamp(0.0, 1.0);
        if water <= 0.10 {
            continue;
        }
        wet_count += 1;
        wet_sum += water;
        wet_heights
            .push(quantized_surface_height(surface_relief[idx]) + TERRAIN_VOXEL_HEIGHT * 0.5);
    }
    if wet_count == 0 {
        return None;
    }
    let wet_coverage = wet_count as f32 / surface_relief.len() as f32;
    let mean_wet = wet_sum / wet_count as f32;
    if wet_coverage < 0.52 || mean_wet < 0.48 {
        return None;
    }
    wet_heights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let quantile_idx = ((wet_heights.len() as f32) * 0.62).floor() as usize;
    let quantile_idx = quantile_idx.min(wet_heights.len().saturating_sub(1));
    Some(wet_heights[quantile_idx] + (0.40 + mean_wet * 0.42) * TERRAIN_VOXEL_HEIGHT)
}

fn fit_pooled_water_level(ground_tops: &mut [f32], water_budget: f32) -> Option<f32> {
    if ground_tops.is_empty() || water_budget <= 0.0 {
        return None;
    }
    ground_tops.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut level = ground_tops[0];
    let mut remaining = water_budget.max(0.0);
    let mut filled = 1usize;
    for &next in ground_tops.iter().skip(1) {
        if next <= level {
            filled += 1;
            continue;
        }
        let capacity = (next - level) * filled as f32;
        if remaining <= capacity {
            return Some(level + remaining / filled as f32);
        }
        remaining -= capacity;
        level = next;
        filled += 1;
    }
    Some(level + remaining / filled as f32)
}

fn local_pool_water_levels(
    width: usize,
    height: usize,
    surface_relief: &[f32],
    water_mask: &[f32],
    open_water: &[f32],
) -> Vec<Option<f32>> {
    let plane = width * height;
    let mut levels = vec![None; plane];
    if plane == 0
        || surface_relief.len() != plane
        || water_mask.len() != plane
        || open_water.len() != plane
    {
        return levels;
    }

    let mut visited = vec![false; plane];
    for start in 0..plane {
        if visited[start] {
            continue;
        }
        let start_raw = water_mask[start].clamp(0.0, 1.0);
        let start_basin = open_water[start].clamp(0.0, 1.0);
        if start_raw <= 0.08 && start_basin <= 0.18 {
            continue;
        }

        let mut queue = VecDeque::from([start]);
        let mut component = Vec::new();
        visited[start] = true;

        while let Some(idx) = queue.pop_front() {
            component.push(idx);
            let x = idx % width;
            let y = idx / width;
            for (dx, dy) in [(1isize, 0isize), (-1, 0), (0, 1), (0, -1)] {
                let nx = x as isize + dx;
                let ny = y as isize + dy;
                if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                    continue;
                }
                let nidx = ny as usize * width + nx as usize;
                if visited[nidx] {
                    continue;
                }
                let raw = water_mask[nidx].clamp(0.0, 1.0);
                let basin = open_water[nidx].clamp(0.0, 1.0);
                if raw <= 0.08 && basin <= 0.18 {
                    continue;
                }
                visited[nidx] = true;
                queue.push_back(nidx);
            }
        }

        let water_budget = component
            .iter()
            .map(|&idx| {
                let raw = water_mask[idx].clamp(0.0, 1.0);
                if raw <= 0.02 {
                    0.0
                } else {
                    (0.10 + raw * 0.42 + raw * raw * 0.22).clamp(0.08, 0.86)
                }
            })
            .sum::<f32>();
        if water_budget <= 0.02 {
            continue;
        }

        let mut ground_tops = component
            .iter()
            .map(|&idx| quantized_surface_height(surface_relief[idx]) + TERRAIN_VOXEL_HEIGHT * 0.5)
            .collect::<Vec<_>>();
        let Some(level) = fit_pooled_water_level(&mut ground_tops, water_budget) else {
            continue;
        };

        for &idx in &component {
            let raw = water_mask[idx].clamp(0.0, 1.0);
            let basin = open_water[idx].clamp(0.0, 1.0);
            if raw > 0.04 || basin > 0.24 {
                levels[idx] = Some(level);
            }
        }
    }

    levels
}

pub fn terrain_voxel_batches(
    world: &TerrariumWorld,
    atmosphere: &TerrariumAtmosphereFrame,
) -> (Vec<f32>, Vec<TerrariumTerrainVoxelBatch>) {
    let width = world.config.width;
    let height = world.config.height;
    let plane = width * height;
    let open_water = open_water_basin_mask(width, height, &world.water_mask);
    let surface_relief = world.soil_structure.clone();
    let grass_cover = grass_cover_projection(world);
    let flood_line = flood_waterline(&surface_relief, &world.water_mask);
    let local_pool_levels = if flood_line.is_some() {
        vec![None; plane]
    } else {
        local_pool_water_levels(
            width,
            height,
            &surface_relief,
            &world.water_mask,
            &open_water,
        )
    };

    let mut bedrock = TerrariumTerrainVoxelBatch {
        class: TerrariumVoxelMaterialClass::Bedrock,
        instances: Vec::with_capacity(plane),
    };
    let mut subsoil = TerrariumTerrainVoxelBatch {
        class: TerrariumVoxelMaterialClass::Subsoil,
        instances: Vec::with_capacity(plane * 3),
    };
    let mut surface = TerrariumTerrainVoxelBatch {
        class: TerrariumVoxelMaterialClass::Surface,
        instances: Vec::with_capacity(plane),
    };
    let mut water = TerrariumTerrainVoxelBatch {
        class: TerrariumVoxelMaterialClass::Water,
        instances: Vec::with_capacity(plane / 3),
    };
    let water_cycle = TerrariumWaterCycleInputs {
        lunar_phase: world.lunar_phase(),
        moonlight: world.moonlight(),
        tidal_moisture_factor: world.tidal_moisture_factor(),
    };

    for idx in 0..plane {
        let x = idx % width;
        let y = idx / width;
        let gx = x as f32 + 0.5;
        let gy = y as f32 + 0.5;
        let local_air = sample_visual_air(atmosphere, width, height, gx, gy);
        let chemistry = sample_visual_chemistry(world, x, y);
        let base_rgb = terrain_visual_response_with_chemistry(
            world.moisture[idx],
            world.organic_matter[idx],
            local_air,
            chemistry,
            world.config.visual_emergence_blend,
        )
        .rgb;
        let level = quantized_surface_level(surface_relief[idx]) as usize;
        let grass_t =
            grass_cover[idx].clamp(0.0, 1.0) * (1.0 - world.water_mask[idx].clamp(0.0, 1.0));
        let wet_t = (world.water_mask[idx].clamp(0.0, 1.0) * 0.65
            + world.moisture[idx].clamp(0.0, 1.0) * 0.20)
            .clamp(0.0, 1.0);
        let surface_rgb = quantize_rgb(
            if grass_t > 0.12 {
                lerp_rgb(base_rgb, [0.30, 0.40, 0.24], 0.14 + grass_t * 0.18)
            } else {
                lerp_rgb(base_rgb, [0.24, 0.24, 0.20], wet_t * 0.18)
            },
            6,
        );
        let subsoil_rgb = quantize_rgb(lerp_rgb(base_rgb, [0.36, 0.30, 0.24], 0.44), 6);
        let bedrock_rgb = quantize_rgb(lerp_rgb(subsoil_rgb, [0.26, 0.24, 0.22], 0.34), 5);

        for layer in 0..=level {
            let instance = TerrariumTerrainVoxelInstance {
                x: gx,
                y: TERRAIN_VOXEL_BASE_Y
                    + layer as f32 * TERRAIN_VOXEL_HEIGHT
                    + TERRAIN_VOXEL_HEIGHT * 0.5,
                z: gy,
                scale_y: TERRAIN_VOXEL_HEIGHT,
                rgb: if layer == 0 {
                    bedrock_rgb
                } else if layer == level {
                    surface_rgb
                } else {
                    subsoil_rgb
                },
            };
            if layer == 0 {
                bedrock.instances.push(instance);
            } else if layer == level {
                surface.instances.push(instance);
            } else {
                subsoil.instances.push(instance);
            }
        }

        let raw_water_t = world.water_mask[idx].clamp(0.0, 1.0);
        let water_t = if flood_line.is_some() {
            raw_water_t
        } else {
            open_water[idx].clamp(0.0, 1.0)
        };
        let local_surface_t = if flood_line.is_some() {
            raw_water_t
        } else {
            (raw_water_t * 0.74 + water_t * 0.26).clamp(0.0, 1.0)
        };
        let is_local_basin = (raw_water_t > 0.18 && water_t > 0.28) || raw_water_t > 0.44;
        let should_render_water = if flood_line.is_some() {
            water_t > 0.08
        } else {
            local_pool_levels[idx].is_some() || is_local_basin
        };
        if should_render_water {
            let water_visual = water_visual_response_with_chemistry(
                local_air,
                raw_water_t.max(local_surface_t) * 120.0,
                world.time_s,
                idx as f32,
                water_cycle,
                chemistry,
            );
            let surface_top =
                quantized_surface_height(surface_relief[idx]) + TERRAIN_VOXEL_HEIGHT * 0.5;
            if let Some(target_top) = flood_line {
                let target_top = target_top + water_t * 0.12;
                if target_top > surface_top + 0.04 {
                    let mut bottom = surface_top - 0.02;
                    let mut remaining = (target_top - bottom).max(0.14);
                    while remaining > 0.02 {
                        let layer_height = remaining.min(TERRAIN_VOXEL_HEIGHT);
                        water.instances.push(TerrariumTerrainVoxelInstance {
                            x: gx,
                            y: bottom + layer_height * 0.5,
                            z: gy,
                            scale_y: layer_height,
                            rgb: water_visual.rgb,
                        });
                        bottom += layer_height;
                        remaining -= layer_height;
                    }
                }
            } else if let Some(target_level) = local_pool_levels[idx] {
                let target_top = target_level;
                if target_top > surface_top + 0.03 {
                    let mut bottom = surface_top - 0.01;
                    let mut remaining = (target_top - bottom).max(0.10);
                    while remaining > 0.02 {
                        let layer_height = remaining.min(TERRAIN_VOXEL_HEIGHT);
                        water.instances.push(TerrariumTerrainVoxelInstance {
                            x: gx,
                            y: bottom + layer_height * 0.5,
                            z: gy,
                            scale_y: layer_height,
                            rgb: water_visual.rgb,
                        });
                        bottom += layer_height;
                        remaining -= layer_height;
                    }
                }
            } else {
                let water_height =
                    (0.22 + local_surface_t * 0.40 + local_surface_t * local_surface_t * 0.12)
                        .clamp(0.22, 0.72)
                        * water_visual.thickness_scale.clamp(0.84, 1.14);
                water.instances.push(TerrariumTerrainVoxelInstance {
                    x: gx,
                    y: surface_top + water_height * 0.5 - 0.04,
                    z: gy,
                    scale_y: water_height,
                    rgb: water_visual.rgb,
                });
            }
        }
    }

    let batches = [bedrock, subsoil, surface, water]
        .into_iter()
        .filter(|batch| !batch.instances.is_empty())
        .collect();
    (surface_relief, batches)
}

pub fn sample_topdown_field(
    field: &[f32],
    width: usize,
    height: usize,
    x: f32,
    y: f32,
    fallback: f32,
) -> f32 {
    if field.is_empty() || width == 0 || height == 0 {
        return fallback;
    }
    let ix = x.floor().clamp(0.0, width.saturating_sub(1) as f32) as usize;
    let iy = y.floor().clamp(0.0, height.saturating_sub(1) as f32) as usize;
    field[iy * width + ix]
}

pub fn sample_visual_air(
    atmosphere: &TerrariumAtmosphereFrame,
    width: usize,
    height: usize,
    x: f32,
    y: f32,
) -> TerrariumVisualAirSample {
    let wind_x = sample_topdown_field(&atmosphere.wind_x, width, height, x, y, 0.0);
    let wind_y = sample_topdown_field(&atmosphere.wind_y, width, height, x, y, 0.0);
    let wind_z = sample_topdown_field(&atmosphere.wind_z, width, height, x, y, 0.0);
    let pressure_kpa = sample_topdown_field(
        &atmosphere.pressure_kpa,
        width,
        height,
        x,
        y,
        ATMOS_PRESSURE_BASELINE_KPA,
    );
    TerrariumVisualAirSample {
        temperature_c: sample_topdown_field(&atmosphere.temperature_c, width, height, x, y, 22.0),
        humidity: sample_topdown_field(&atmosphere.humidity, width, height, x, y, 0.5)
            .clamp(0.0, 1.0),
        pressure_kpa,
        pressure_delta_kpa: pressure_kpa - ATMOS_PRESSURE_BASELINE_KPA,
        wind_x,
        wind_y,
        wind_z,
        wind_speed: (wind_x * wind_x + wind_y * wind_y + wind_z * wind_z).sqrt(),
    }
}

fn chemistry_probe_z(depth: usize) -> usize {
    let depth = depth.max(1);
    (depth / 3).min(depth.saturating_sub(1))
}

pub fn sample_visual_chemistry(
    world: &TerrariumWorld,
    x: usize,
    y: usize,
) -> TerrariumVisualChemistrySample {
    let width = world.config.width;
    let height = world.config.height;
    let (_, _, depth) = world.substrate.shape();
    if width == 0 || height == 0 || depth == 0 {
        return TerrariumVisualChemistrySample::default();
    }

    let x = x.min(width.saturating_sub(1));
    let y = y.min(height.saturating_sub(1));
    let plane = width * height;
    let local_idx = chemistry_probe_z(depth) * plane + y * width + x;
    let sample = |species| {
        world
            .substrate
            .species_field(species)
            .get(local_idx)
            .copied()
            .unwrap_or(0.0)
    };

    let proton = sample(TerrariumSpecies::Proton);
    let oxygen = sample(TerrariumSpecies::OxygenGas);
    let glucose = sample(TerrariumSpecies::Glucose);
    let silicate_mineral = sample(TerrariumSpecies::SilicateMineral);
    let clay_mineral = sample(TerrariumSpecies::ClayMineral);
    let carbonate_mineral = sample(TerrariumSpecies::CarbonateMineral);
    let iron_oxide = sample(TerrariumSpecies::IronOxideMineral);
    let dissolved_silicate = sample(TerrariumSpecies::DissolvedSilicate);
    let bicarbonate = sample(TerrariumSpecies::BicarbonatePool);
    let surface_proton_load = sample(TerrariumSpecies::SurfaceProtonLoad);
    let calcium_bicarbonate_complex = sample(TerrariumSpecies::CalciumBicarbonateComplex);
    let sorbed_ferric_hydroxide = sample(TerrariumSpecies::SorbedFerricHydroxide);
    let exchangeable_calcium = sample(TerrariumSpecies::ExchangeableCalcium);
    let exchangeable_magnesium = sample(TerrariumSpecies::ExchangeableMagnesium);
    let exchangeable_potassium = sample(TerrariumSpecies::ExchangeablePotassium);
    let exchangeable_sodium = sample(TerrariumSpecies::ExchangeableSodium);
    let exchangeable_aluminum = sample(TerrariumSpecies::ExchangeableAluminum);
    let aqueous_iron = sample(TerrariumSpecies::AqueousIronPool);

    // Chemistry probe multipliers derived from molar extinction coefficients
    // (Kasha 1950, Beer 1852). Species with higher optical extinction contribute
    // more to their respective probes — replacing hardcoded magic numbers.
    let w_proton = emergent_color::extinction_probe_weight(TerrariumSpecies::Proton);
    let w_spl = emergent_color::extinction_probe_weight(TerrariumSpecies::SurfaceProtonLoad);
    let w_exal = emergent_color::extinction_probe_weight(TerrariumSpecies::ExchangeableAluminum);
    let w_bicarb = emergent_color::extinction_probe_weight(TerrariumSpecies::BicarbonatePool);
    let w_cabic = emergent_color::extinction_probe_weight(TerrariumSpecies::CalciumBicarbonateComplex);
    let w_exca = emergent_color::extinction_probe_weight(TerrariumSpecies::ExchangeableCalcium);
    let w_exmg = emergent_color::extinction_probe_weight(TerrariumSpecies::ExchangeableMagnesium);
    let w_carb = emergent_color::extinction_probe_weight(TerrariumSpecies::CarbonateMineral);
    let w_dsi = emergent_color::extinction_probe_weight(TerrariumSpecies::DissolvedSilicate);
    let w_sil = emergent_color::extinction_probe_weight(TerrariumSpecies::SilicateMineral);
    let w_clay = emergent_color::extinction_probe_weight(TerrariumSpecies::ClayMineral);
    let w_feox = emergent_color::extinction_probe_weight(TerrariumSpecies::IronOxideMineral);
    let w_feh = emergent_color::extinction_probe_weight(TerrariumSpecies::SorbedFerricHydroxide);
    let w_feaq = emergent_color::extinction_probe_weight(TerrariumSpecies::AqueousIronPool);
    let w_exna = emergent_color::extinction_probe_weight(TerrariumSpecies::ExchangeableSodium);
    let w_exk = emergent_color::extinction_probe_weight(TerrariumSpecies::ExchangeablePotassium);
    let w_o2 = emergent_color::extinction_probe_weight(TerrariumSpecies::OxygenGas);

    let acidity =
        (proton * w_proton + surface_proton_load * w_spl + exchangeable_aluminum * w_exal)
            .clamp(0.0, 1.0);
    let alkalinity = (bicarbonate * w_bicarb
        + calcium_bicarbonate_complex * w_cabic
        + exchangeable_calcium * w_exca
        + exchangeable_magnesium * w_exmg
        + carbonate_mineral * w_carb)
        .clamp(0.0, 1.0);
    let silicate =
        (dissolved_silicate * w_dsi + silicate_mineral * w_sil * 0.2 + clay_mineral * w_clay * 0.2)
            .clamp(0.0, 1.0);
    let carbonate =
        (carbonate_mineral * w_carb + calcium_bicarbonate_complex * w_cabic + bicarbonate * w_bicarb * 0.4)
            .clamp(0.0, 1.0);
    let ferric = (iron_oxide * w_feox * 0.4 + sorbed_ferric_hydroxide * w_feh).clamp(0.0, 1.0);
    let ferrous = (aqueous_iron * w_feaq).clamp(0.0, 1.0);
    let salinity = (exchangeable_sodium * w_exna
        + exchangeable_potassium * w_exk
        + exchangeable_magnesium * w_exmg
        + exchangeable_calcium * w_exca * 0.5)
        .clamp(0.0, 1.0);
    let clay = (clay_mineral * w_clay * 0.5).clamp(0.0, 1.0);
    let redox = clamp(
        oxygen * w_o2 + ferric * 0.55 - ferrous * 0.80
            - proton * w_proton * 0.2 - surface_proton_load * w_spl * 0.15
            + bicarbonate * w_bicarb * 0.06,
        -1.0,
        1.0,
    );
    let turbidity = (clay * 0.55
        + silicate * 0.34
        + ferrous * 0.42
        + ferric * 0.18
        + acidity * 0.16
        + glucose * 0.10)
        .clamp(0.0, 1.0);
    let clarity = (0.92 + redox * 0.10 + alkalinity * 0.08 - turbidity * 0.72).clamp(0.18, 1.0);

    TerrariumVisualChemistrySample {
        acidity,
        alkalinity,
        silicate,
        carbonate,
        ferric,
        ferrous,
        salinity,
        clay,
        redox,
        clarity,
    }
}

pub fn terrain_visual_response(
    moisture: f32,
    organic: f32,
    local_air: TerrariumVisualAirSample,
) -> TerrariumTerrainVisualResponse {
    terrain_visual_response_with_chemistry(
        moisture,
        organic,
        local_air,
        TerrariumVisualChemistrySample::default(),
        1.0,
    )
}

/// Terrain visual with full chemistry enrichment.
///
/// `blend` controls emergent chemistry contribution:
/// 0.0 = simple mineral CPK base only, 1.0 = full Beer-Lambert chemistry tinting.
pub fn terrain_visual_response_with_chemistry(
    moisture: f32,
    organic: f32,
    local_air: TerrariumVisualAirSample,
    chemistry: TerrariumVisualChemistrySample,
    blend: f32,
) -> TerrariumTerrainVisualResponse {
    // ── Emergent base: soil color from mineral CPK optics (Brady & Weil 2017) ──
    let base = emergent_color::emergent_soil_base_color(moisture, organic);

    // ── Emergent chemistry tinting via Beer-Lambert ──
    // Each chemistry tint uses the molecular color of the dominant species,
    // weighted by concentration and molar extinction.
    let chemistry_rgb = emergent_color::emergent_terrain_chemistry_tint(
        chemistry.ferric + chemistry.ferrous * 0.5,
        chemistry.silicate,
        chemistry.clay,
        chemistry.carbonate + chemistry.alkalinity * 0.4,
        chemistry.acidity,
        base,
    );

    // Redox shift: reducing conditions → greenish-gray from Fe²⁺ CPK
    let chemistry_rgb = if chemistry.redox < 0.0 {
        let fe_reduced = emergent_color::beer_lambert_mix(
            &[
                (TerrariumSpecies::AqueousIronPool, (-chemistry.redox) * 0.5),
                (TerrariumSpecies::Water, 0.5),
            ],
            chemistry_rgb,
        );
        lerp_rgb(
            chemistry_rgb,
            fe_reduced,
            (-chemistry.redox) * 0.26 + chemistry.ferrous * 0.10,
        )
    } else {
        // Oxidizing: slight warming from Fe³⁺
        lerp_rgb(
            chemistry_rgb,
            emergent_color::beer_lambert_mix(
                &[(TerrariumSpecies::IronOxideMineral, chemistry.redox * 0.3)],
                chemistry_rgb,
            ),
            chemistry.redox * 0.08,
        )
    };

    // Salinity: exchangeable cation whitening (salt crust effect)
    let chemistry_rgb = if chemistry.salinity > 0.01 {
        let salt_rgb = emergent_color::beer_lambert_mix(
            &[
                (TerrariumSpecies::ExchangeableSodium, chemistry.salinity * 0.4),
                (TerrariumSpecies::ExchangeableCalcium, chemistry.salinity * 0.3),
            ],
            [0.88, 0.84, 0.74], // salt crust is pale
        );
        lerp_rgb(chemistry_rgb, salt_rgb, chemistry.salinity * 0.10)
    } else {
        chemistry_rgb
    };

    // Atmospheric effects (humidity darkening, thermal shift)
    let humid_tint = lerp_rgb(
        chemistry_rgb,
        emergent_color::emergent_soil_base_color(local_air.humidity, organic),
        local_air.humidity * 0.10,
    );
    let thermal_t = ((local_air.temperature_c - 6.0) / 24.0).clamp(0.0, 1.0);
    let thermal_tint = lerp_rgb(
        emergent_color::emergent_soil_base_color(0.3, 0.0), // cool-dry
        emergent_color::emergent_soil_base_color(0.1, 0.0), // warm-dry
        thermal_t,
    );
    let pressure_highlight = local_air.pressure_delta_kpa.abs().min(0.25) * 0.24;
    let emergent_rgb = lerp_rgb(
        lerp_rgb(humid_tint, thermal_tint, 0.10),
        emergent_color::beer_lambert_mix_with_scattering(
            &[(TerrariumSpecies::SilicateMineral, 0.3)],
            [0.83, 0.82, 0.78],
        ),
        pressure_highlight * 0.16,
    );
    // Blend: 0.0 = simple mineral CPK base, 1.0 = full emergent chemistry
    let rgb = lerp_rgb(base, emergent_rgb, blend.clamp(0.0, 1.0));
    TerrariumTerrainVisualResponse {
        rgb: quantize_rgb(rgb, 10),
        provenance: TerrariumVisualProvenance::CoarseFieldProjection,
        uncertainty: visual_uncertainty(
            0.58
                + moisture.clamp(0.0, 1.0) * 0.08
                + (1.0 - chemistry.clarity) * 0.12
                + (1.0 - blend.clamp(0.0, 1.0)) * 0.06,
        ),
    }
}

pub fn plant_visual_response(
    taxonomy_id: u32,
    local_air: TerrariumVisualAirSample,
    vitality: f32,
    canopy_density: f32,
    time_s: f32,
    phase_seed: f32,
) -> TerrariumPlantVisualResponse {
    plant_visual_response_with_molecular_state(
        taxonomy_id,
        local_air,
        vitality,
        canopy_density,
        time_s,
        phase_seed,
        None,
    )
}

pub fn plant_visual_response_with_molecular_state(
    taxonomy_id: u32,
    local_air: TerrariumVisualAirSample,
    vitality: f32,
    canopy_density: f32,
    time_s: f32,
    phase_seed: f32,
    molecular: Option<MolecularVisualState>,
) -> TerrariumPlantVisualResponse {
    // ── Emergent leaf color from chlorophyll/carotenoid CPK optics ──
    // Vitality controls chlorophyll degradation (senescence → carotenoid)
    // Canopy density controls light environment (Sims & Gamon 2002)
    let stress_t = (1.0 - vitality).clamp(0.0, 1.0);
    let chlorophyll = (0.7 - stress_t * 0.5) * (0.6 + canopy_density.clamp(0.0, 1.0) * 0.4);
    let carotenoid = 0.10 + stress_t * 0.35; // revealed as chlorophyll degrades
    let anthocyanin = stress_t * 0.08; // stress-induced anthocyanin

    let heuristic_leaf = emergent_color::emergent_leaf_color(chlorophyll, carotenoid, anthocyanin);
    let base_leaf = molecular
        .map(|state| state.leaf_rgb())
        .unwrap_or(heuristic_leaf);

    // Humidity boosts green (well-watered plants have more chlorophyll)
    let humid_green = if let Some(state) = molecular {
        lerp_rgb(
            base_leaf,
            emergent_color::emergent_leaf_color(
                (state.chlorophyll_density + local_air.humidity * 0.08).clamp(0.0, 1.0),
                (state.carotenoid_density * 0.9).clamp(0.0, 1.0),
                state.anthocyanin_density.clamp(0.0, 1.0),
            ),
            local_air.humidity * 0.10,
        )
    } else {
        let humid_boost = emergent_color::emergent_leaf_color(
            chlorophyll + local_air.humidity * 0.1,
            carotenoid * 0.8,
            anthocyanin,
        );
        lerp_rgb(base_leaf, humid_boost, local_air.humidity * 0.14)
    };

    let heat_stress = (((local_air.temperature_c - 28.0) / 10.0).max(0.0)
        + ((0.35 - local_air.humidity) / 0.35).max(0.0))
    .clamp(0.0, 1.0);

    let species_accent = taxonomy_palette_accent(taxonomy_id);
    let fruit_rgb = molecular
        .map(|state| {
            let unripe_rgb = lerp_rgb(state.fruit_rgb(), state.leaf_rgb(), 0.42);
            let pigment_rgb = lerp_rgb(
                unripe_rgb,
                state.fruit_rgb(),
                vitality.clamp(0.0, 1.0) * 0.78 + canopy_density.clamp(0.0, 1.0) * 0.10,
            );
            lerp_rgb(
                pigment_rgb,
                emergent_color::emergent_fruit_color(0.5, 0.3),
                heat_stress * 0.06 + local_air.pressure_delta_kpa.abs().min(0.5) * 0.06,
            )
        })
        .unwrap_or_else(|| {
            lerp_rgb(
                taxonomy_fruit_palette(taxonomy_id),
                emergent_color::emergent_fruit_color(0.5, 0.3), // heat-bleached fruit
                heat_stress * 0.12 + local_air.pressure_delta_kpa.abs().min(0.5) * 0.12,
            )
        });

    // Heat-stressed leaves: chlorophyll breakdown → yellow-brown (carotenoid+damage)
    let heat_stressed_leaf = emergent_color::emergent_leaf_color(0.1, 0.6, 0.1);
    let leaf_rgb = lerp_rgb(
        lerp_rgb(humid_green, heat_stressed_leaf, heat_stress * 0.42),
        species_accent,
        0.14,
    );
    // Cold shift: slight blue-gray from reduced metabolic activity
    let cold_leaf = emergent_color::emergent_leaf_color(0.3, 0.2, 0.15);
    let leaf_rgb = lerp_rgb(
        leaf_rgb,
        cold_leaf,
        ((12.0 - local_air.temperature_c) / 18.0).clamp(0.0, 1.0) * 0.08,
    );

    // Stem color: lignin/cellulose → primarily C+O CPK blend (woody brown)
    // Cellulose (C₆H₁₀O₅)ₙ: C=gray, H=white, O=red → warm brown
    let stem_base = emergent_color::beer_lambert_mix(
        &[
            (TerrariumSpecies::Glucose, 0.6), // cellulose proxy (same composition)
        ],
        [0.40, 0.30, 0.18], // lignin fallback
    );
    let stem_rgb = lerp_rgb(
        [
            clamp(stem_base[0] + (1.0 - local_air.humidity) * 0.08, 0.0, 1.0),
            clamp(stem_base[1] + local_air.humidity * 0.05, 0.0, 1.0),
            clamp(
                stem_base[2] + ((local_air.temperature_c - 12.0) / 18.0).clamp(0.0, 1.0) * 0.03,
                0.0,
                1.0,
            ),
        ],
        species_accent,
        0.08,
    );

    let gust_phase = time_s * (1.8 + local_air.wind_speed * 1.4) + phase_seed * 0.37;
    let gust = gust_phase.sin() * 0.5 + (gust_phase * 0.63).cos() * 0.35;
    let turgor = molecular
        .map(|state| state.turgor_pressure.clamp(0.0, 1.0))
        .unwrap_or(local_air.humidity.clamp(0.0, 1.0));
    let senescence = molecular
        .map(|state| state.senescence_progress.clamp(0.0, 1.0))
        .unwrap_or(stress_t * 0.5);
    let sway_mag = local_air.wind_speed
        * (0.08 + canopy_density.clamp(0.0, 1.0) * 0.04)
        * (1.0 + (1.0 - turgor) * 0.18);
    let sway_x = local_air.wind_x * 0.42 + sway_mag * gust;
    let sway_z = local_air.wind_y * 0.38 + sway_mag * (gust_phase * 0.79).sin() * 0.8;
    let uncertainty = if molecular.is_some() {
        0.18 + heat_stress * 0.04 + senescence * 0.04
    } else {
        0.34 + heat_stress * 0.08 + stress_t * 0.06
    };

    TerrariumPlantVisualResponse {
        stem_rgb: quantize_rgb(stem_rgb, 9),
        leaf_rgb: quantize_rgb(leaf_rgb, 9),
        fruit_rgb: quantize_rgb(fruit_rgb, 8),
        sway_x,
        sway_z,
        lean_x: local_air.wind_x * 0.08 + gust * local_air.wind_speed * 0.05,
        lean_z: local_air.wind_y * 0.08 + (gust_phase * 0.77).cos() * local_air.wind_speed * 0.04,
        vertical_offset: local_air.pressure_delta_kpa * 0.02 - (1.0 - turgor) * 0.01,
        canopy_scale: (0.88
            + canopy_density.clamp(0.0, 1.0) * 0.32
            + turgor * 0.14
            + local_air.humidity * 0.05
            - heat_stress * 0.10
            - senescence * 0.08)
            .clamp(0.75, 1.35),
        provenance: TerrariumVisualProvenance::RuntimeProjection,
        uncertainty: visual_uncertainty(uncertainty),
    }
}

/// Fruit visual with emergent molecular color blending.
///
/// `blend` controls emergent color contribution:
/// 0.0 = taxonomy palette only, 1.0 = full molecular optics.
pub fn fruit_visual_response(
    taxonomy_id: u32,
    local_air: TerrariumVisualAirSample,
    ripeness: f32,
    sugar_content: f32,
    time_s: f32,
    phase_seed: f32,
    blend: f32,
) -> TerrariumFruitVisualResponse {
    fruit_visual_response_with_molecular_state(
        taxonomy_id,
        local_air,
        ripeness,
        sugar_content,
        time_s,
        phase_seed,
        blend,
        None,
    )
}

pub fn fruit_visual_response_with_molecular_state(
    taxonomy_id: u32,
    local_air: TerrariumVisualAirSample,
    ripeness: f32,
    sugar_content: f32,
    time_s: f32,
    phase_seed: f32,
    blend: f32,
    molecular: Option<MolecularVisualState>,
) -> TerrariumFruitVisualResponse {
    let ripe_t = ripeness.clamp(0.0, 1.0);
    let sugar_t = (sugar_content / 1.5).clamp(0.0, 1.0);
    let base = taxonomy_fruit_palette(taxonomy_id);
    // Underripe: still green (high chlorophyll)
    let skin_rgb = if let Some(state) = molecular {
        let ripe_rgb = state.fruit_rgb();
        let unripe_rgb = lerp_rgb(ripe_rgb, state.leaf_rgb(), (1.0 - ripe_t) * 0.55);
        lerp_rgb(unripe_rgb, ripe_rgb, ripe_t * 0.92 + sugar_t * 0.08)
    } else {
        let underripe_emergent = emergent_color::emergent_fruit_color(0.1, 0.2);
        let underripe = lerp_rgb(base, underripe_emergent, 0.55 * blend.clamp(0.0, 1.0));
        lerp_rgb(underripe, base, ripe_t * 0.92 + sugar_t * 0.08)
    };
    // Stem: lignified tissue (cellulose/lignin brown from CPK)
    let stem_base = emergent_color::beer_lambert_mix(
        &[(TerrariumSpecies::Glucose, 0.5)],
        [0.34, 0.24, 0.10],
    );
    let stem_rgb = lerp_rgb(
        stem_base,
        taxonomy_palette_accent(taxonomy_id),
        0.14,
    );
    let wobble_phase = time_s * (1.1 + local_air.wind_speed * 0.6) + phase_seed * 0.47;
    let uncertainty = if molecular.is_some() {
        0.20 + (1.0 - ripe_t) * 0.04 + (1.0 - sugar_t) * 0.02
    } else {
        0.36 + (1.0 - blend.clamp(0.0, 1.0)) * 0.10 + (1.0 - ripe_t) * 0.04
    };
    TerrariumFruitVisualResponse {
        skin_rgb: quantize_rgb(skin_rgb, 5),
        stem_rgb: quantize_rgb(stem_rgb, 4),
        tilt_x: local_air.wind_y * 0.06 + wobble_phase.sin() * 0.05,
        tilt_z: -local_air.wind_x * 0.06 + wobble_phase.cos() * 0.05,
        vertical_offset: local_air.pressure_delta_kpa * 0.01,
        sprite_scale: (0.82 + ripe_t * 0.22 + sugar_t * 0.08).clamp(0.75, 1.25),
        provenance: TerrariumVisualProvenance::RuntimeProjection,
        uncertainty: visual_uncertainty(uncertainty),
    }
}

pub fn water_visual_response(
    local_air: TerrariumVisualAirSample,
    volume: f32,
    time_s: f32,
    phase_seed: f32,
    cycle: TerrariumWaterCycleInputs,
) -> TerrariumWaterVisualResponse {
    water_visual_response_with_chemistry(
        local_air,
        volume,
        time_s,
        phase_seed,
        cycle,
        TerrariumVisualChemistrySample::default(),
    )
}

pub fn water_visual_response_with_chemistry(
    local_air: TerrariumVisualAirSample,
    volume: f32,
    time_s: f32,
    phase_seed: f32,
    cycle: TerrariumWaterCycleInputs,
    chemistry: TerrariumVisualChemistrySample,
) -> TerrariumWaterVisualResponse {
    let lunar_phase = cycle.lunar_phase.clamp(0.0, 1.0);
    let moonlight = cycle.moonlight.clamp(0.0, 1.0);
    let tide_t = ((cycle.tidal_moisture_factor - 0.85) / 0.30).clamp(0.0, 1.0);
    let spring_tide = ((cycle.tidal_moisture_factor - 1.0) / 0.15)
        .abs()
        .clamp(0.0, 1.0);
    let surface_phase = time_s * (1.4 + local_air.wind_speed * 0.9 + tide_t * 0.32)
        + phase_seed * 0.41
        + lunar_phase * std::f32::consts::TAU * 2.0;
    let surface_pulse = surface_phase.sin() * (0.08 + spring_tide * 0.03)
        + (surface_phase * 0.67).cos() * (0.04 + tide_t * 0.02);
    let temp_t = ((local_air.temperature_c - 4.0) / 26.0).clamp(0.0, 1.0);
    let humidity_t = local_air.humidity.clamp(0.0, 1.0);

    // ── Emergent water color from Pope & Fry 1997 + Beer-Lambert ──
    let base_water = emergent_color::emergent_water_color();
    let warm_water = emergent_color::emergent_water_body_color(0.0, 0.0, 0.0, temp_t);
    let rgb = lerp_rgb(
        base_water,
        [
            warm_water[0] + humidity_t * 0.04,
            warm_water[1] + humidity_t * 0.04,
            warm_water[2] - temp_t * 0.05,
        ],
        0.55,
    );

    // Moonlight reflection (specular highlight)
    let rgb = lerp_rgb(
        rgb,
        [
            base_water[0] + 0.2,
            base_water[1] + 0.25,
            (base_water[2] + 0.3).min(1.0),
        ],
        moonlight * (0.05 + spring_tide * 0.03),
    );

    // Chemistry tinting via Beer-Lambert through dissolved species
    let alkaline_water = emergent_color::emergent_water_body_color(
        0.0,
        chemistry.silicate * 0.3,
        0.0,
        0.5,
    );
    let rgb = lerp_rgb(
        rgb,
        alkaline_water,
        chemistry.alkalinity * 0.14 + chemistry.carbonate * 0.16 + chemistry.silicate * 0.10,
    );

    // Iron dissolved: Beer-Lambert through ferrous/ferric solution
    // Iron-rich waters (bogs, laterite runoff) are visibly brown/orange
    // (Thurman 1985, Davison 1993). Fe³⁺ is ~10× more absorbing than Fe²⁺.
    let iron_water = emergent_color::emergent_water_body_color(
        chemistry.ferrous + chemistry.ferric * 0.5,
        0.0,
        0.0,
        0.3,
    );
    let iron_blend = (chemistry.ferrous * 0.40 + chemistry.ferric * 0.25).clamp(0.0, 0.7);
    let rgb = lerp_rgb(rgb, iron_water, iron_blend);

    // Acidic water: dissolved protons shift toward molecular proton CPK color
    let acid_water = emergent_color::beer_lambert_mix(
        &[
            (TerrariumSpecies::Proton, chemistry.acidity * 0.3),
            (TerrariumSpecies::Water, 1.0),
        ],
        rgb,
    );
    let rgb = lerp_rgb(rgb, acid_water, chemistry.acidity * 0.12);

    // Reducing conditions: Fe²⁺ green tint
    let rgb = if chemistry.redox < 0.0 {
        let reducing_water = emergent_color::emergent_water_body_color(
            (-chemistry.redox) * 0.5,
            0.0,
            0.0,
            0.2,
        );
        lerp_rgb(rgb, reducing_water, (-chemistry.redox) * 0.26)
    } else {
        rgb
    };

    // High clarity: slight brightening
    let clear_water = emergent_color::emergent_water_body_color(0.0, 0.0, 0.0, 1.0);
    let rgb = lerp_rgb(rgb, clear_water, chemistry.clarity * 0.08);
    TerrariumWaterVisualResponse {
        rgb: quantize_rgb(rgb, 20),
        tilt_x: -local_air.wind_y * 0.05 + (surface_phase * 0.43).sin() * spring_tide * 0.02,
        tilt_z: local_air.wind_x * 0.05 + (surface_phase * 0.37).cos() * spring_tide * 0.02,
        vertical_offset: (cycle.tidal_moisture_factor - 1.0) * 0.08
            + local_air.pressure_delta_kpa * 0.004,
        thickness_scale: (0.9
            + surface_pulse
            + local_air.pressure_delta_kpa * 0.015
            + spring_tide * 0.05
            + (1.0 - chemistry.clarity) * 0.04)
            .max(0.72 + volume.clamp(0.0, 1.0) * 0.08),
        radius_x_cells: (0.85 + (volume / 70.0).sqrt() * 1.15 + humidity_t * 0.18 + tide_t * 0.10)
            .clamp(0.9, 3.4),
        radius_y_cells: (0.78 + (volume / 82.0).sqrt() * 0.95 + temp_t * 0.10 + spring_tide * 0.08)
            .clamp(0.85, 3.0),
        clarity: chemistry.clarity,
        provenance: TerrariumVisualProvenance::CoarseFieldProjection,
        uncertainty: visual_uncertainty(
            0.48
                + (1.0 - chemistry.clarity) * 0.18
                + chemistry.ferrous * 0.06
                + chemistry.ferric * 0.04,
        ),
    }
}

pub fn seed_visual_response(
    taxonomy_id: u32,
    local_air: TerrariumVisualAirSample,
    reserve_carbon: f32,
    dormancy_s: f32,
    time_s: f32,
    phase_seed: f32,
) -> TerrariumSeedVisualResponse {
    let reserve_t = (reserve_carbon / 0.25).clamp(0.0, 1.0);
    let dormancy_t = (dormancy_s / 26000.0).clamp(0.0, 1.0);
    // Seed coat: lignified tissue (cellulose/tannin brown from CPK)
    let seed_coat_base = emergent_color::beer_lambert_mix(
        &[(TerrariumSpecies::Glucose, 0.4)], // cellulose proxy
        [0.46, 0.34, 0.18],
    );
    let shell_base = lerp_rgb(
        seed_coat_base,
        taxonomy_fruit_palette(taxonomy_id),
        0.18,
    );
    let accent_rgb = lerp_rgb(shell_base, taxonomy_palette_accent(taxonomy_id), 0.22);
    let settle_phase = time_s * 0.55 + phase_seed * 0.31;
    // Desiccated seed: lighter color from water loss (carbohydrate-rich endosperm)
    let desiccated = emergent_color::beer_lambert_mix(
        &[
            (TerrariumSpecies::Glucose, 0.5),
            (TerrariumSpecies::Water, 0.2),
        ],
        [0.76, 0.68, 0.42],
    );
    TerrariumSeedVisualResponse {
        shell_rgb: quantize_rgb(
            lerp_rgb(
                shell_base,
                desiccated,
                reserve_t * 0.28 + dormancy_t * 0.10,
            ),
            5,
        ),
        accent_rgb: quantize_rgb(accent_rgb, 4),
        tilt_x: local_air.wind_y * 0.02 + settle_phase.sin() * 0.03,
        tilt_z: -local_air.wind_x * 0.02 + settle_phase.cos() * 0.03,
        vertical_offset: local_air.pressure_delta_kpa * 0.005,
        sprite_scale: (0.72 + reserve_t * 0.36 - dormancy_t * 0.08).clamp(0.65, 1.15),
        provenance: TerrariumVisualProvenance::RuntimeProjection,
        uncertainty: visual_uncertainty(0.24 + dormancy_t * 0.05 + (1.0 - reserve_t) * 0.03),
    }
}

pub fn fly_visual_response(
    local_air: TerrariumVisualAirSample,
    body: &BodyState,
    energy_frac: f32,
    time_s: f32,
    phase_seed: f32,
) -> TerrariumFlyVisualResponse {
    let energy_t = energy_frac.clamp(0.0, 1.0);
    let temperature_t = clamp((body.temperature - 12.0) / 18.0, 0.0, 1.0);
    let speed_t = clamp(body.speed / 0.8, 0.0, 1.0);
    let climb_t = clamp(body.vertical_velocity.abs() / 12.0, 0.0, 1.0);
    let (o2_flux, _) = body.calculate_respiration_flux();
    let respiration_t = clamp((-o2_flux) / 0.0025, 0.0, 1.0);
    let body_rgb = quantize_rgb(
        lerp_rgb(
            [0.42, 0.24, 0.10],
            [0.86, 0.72, 0.18],
            energy_t * 0.58 + local_air.humidity * 0.08 + temperature_t * 0.10,
        ),
        5,
    );
    let wing_rgb = quantize_rgb(
        lerp_rgb(
            [0.62, 0.72, 0.78],
            [0.90, 0.92, 0.96],
            0.4 + energy_t * 0.3 + (1.0 - local_air.humidity) * 0.1,
        ),
        4,
    );
    let flutter_rate = 0.004 + body.wing_beat_freq.max(20.0) * 0.00002;
    let flutter_phase = time_s * flutter_rate * 1000.0 + phase_seed * 0.53;
    let explicit_stroke = if body.wing_stroke.abs() > 0.02 {
        body.wing_stroke.clamp(-1.0, 1.0)
    } else {
        flutter_phase.sin()
    };
    let wing_angle = if body.is_flying {
        explicit_stroke * 0.72 + body.wing_dihedral * 0.22 + flutter_phase.sin() * 0.10
    } else {
        0.08 + explicit_stroke * 0.05
    };
    TerrariumFlyVisualResponse {
        body_rgb,
        wing_rgb,
        pitch: body.pitch * 0.45 - local_air.wind_y * 0.12,
        roll: body.roll * 0.45
            + local_air.wind_x * 0.12
            + local_air.wind_speed * flutter_phase.cos() * 0.04,
        wing_angle,
        sprite_scale: (0.86
            + energy_t * 0.18
            + respiration_t * 0.10
            + if body.is_flying { 0.08 } else { 0.0 })
        .clamp(0.8, 1.24),
        thorax_scale: (0.78 + respiration_t * 0.34 + speed_t * 0.08 + climb_t * 0.10)
            .clamp(0.72, 1.40),
        abdomen_scale: (0.74 + energy_t * 0.30 - speed_t * 0.06
            + if body.proboscis_extended { 0.08 } else { 0.0 })
        .clamp(0.68, 1.32),
        head_scale: (0.72
            + temperature_t * 0.12
            + if body.proboscis_extended { 0.12 } else { 0.0 })
        .clamp(0.68, 1.24),
        wing_span: (0.80
            + clamp(body.wing_sweep.abs() / 0.7, 0.0, 1.0) * 0.16
            + clamp(body.wing_dihedral.abs() / 0.5, 0.0, 1.0) * 0.10
            + if body.is_flying { 0.12 } else { 0.0 })
        .clamp(0.72, 1.38),
        wing_width: (0.66 + local_air.humidity * 0.14 + energy_t * 0.04 - speed_t * 0.04)
            .clamp(0.56, 1.08),
        leg_span: (0.64 + if body.is_flying { 0.08 } else { 0.26 } + speed_t * 0.08)
            .clamp(0.60, 1.18),
        proboscis_extension: if body.proboscis_extended {
            (0.16 + energy_t * 0.16 + (1.0 - speed_t) * 0.06).clamp(0.14, 0.40)
        } else {
            0.0
        },
        provenance: TerrariumVisualProvenance::RuntimeProjection,
        uncertainty: visual_uncertainty(
            0.18
                + if body.is_flying { 0.02 } else { 0.04 }
                + (1.0 - energy_t) * 0.03,
        ),
    }
}

pub fn earthworm_visual_response(
    local_air: TerrariumVisualAirSample,
    density: f32,
    biomass: f32,
    bioturbation: f32,
    time_s: f32,
    phase_seed: f32,
) -> TerrariumEarthwormVisualResponse {
    let anatomy = EarthwormBody::default();
    let density_t = clamp(density / 200.0, 0.0, 1.0);
    let biomass_t = clamp(biomass / 0.6, 0.0, 1.0);
    let bioturb_t = clamp(bioturbation / 5.0, 0.0, 1.0);
    let activity = clamp(
        density_t * 0.44 + biomass_t * 0.16 + bioturb_t * 0.40,
        0.0,
        1.0,
    );
    let undulation = (time_s * (0.9 + activity * 0.8) + phase_seed * 0.61).sin();
    let segment_count = ((4.0 + (anatomy.segments as f32 / 22.0) * (0.55 + activity * 0.45)).round()
        as i32)
        .clamp(4, 10) as u8;
    TerrariumEarthwormVisualResponse {
        body_rgb: quantize_rgb(
            lerp_rgb(
                [0.34, 0.16, 0.12],
                [0.62, 0.34, 0.24],
                density_t * 0.42 + bioturb_t * 0.32,
            ),
            5,
        ),
        clitellum_rgb: quantize_rgb(
            lerp_rgb(
                [0.60, 0.34, 0.24],
                [0.82, 0.52, 0.34],
                activity * 0.52 + local_air.humidity * 0.12,
            ),
            5,
        ),
        segment_count,
        length_scale: (0.56 + anatomy.length_cm / 24.0 * 0.28 + activity * 0.42).clamp(0.48, 1.52),
        thickness_scale: (0.48 + anatomy.diameter_mm / 8.0 * 0.28 + density_t * 0.18)
            .clamp(0.42, 1.24),
        curl: clamp(
            (1.0 - local_air.humidity) * 0.30 + undulation * 0.26 - bioturb_t * 0.08,
            -0.50,
            0.50,
        ),
        yaw_rad: local_air.wind_x * 0.48 + undulation * 0.34,
        height_offset: local_air.pressure_delta_kpa * 0.004 + bioturb_t * 0.014,
        activity,
        provenance: TerrariumVisualProvenance::RuntimeProjection,
        uncertainty: visual_uncertainty(0.24 + (1.0 - activity) * 0.04 + density_t * 0.02),
    }
}

pub fn nematode_visual_response(
    kind: NematodeKind,
    local_air: TerrariumVisualAirSample,
    density: f32,
    biomass: f32,
    time_s: f32,
    phase_seed: f32,
) -> TerrariumNematodeVisualResponse {
    let body = kind.body();
    let density_t = clamp(density / 40.0, 0.0, 1.0);
    let biomass_t = clamp(biomass / 0.0025, 0.0, 1.0);
    let activity = clamp(
        density_t * 0.48 + biomass_t * 0.28 + local_air.humidity * 0.24,
        0.0,
        1.0,
    );
    let undulation = (time_s * (1.4 + activity * 1.2) + phase_seed * 0.79).sin();
    let base_rgb = match kind {
        NematodeKind::BacterialFeeder => [0.82, 0.80, 0.66],
        NematodeKind::FungalFeeder => [0.76, 0.72, 0.58],
        NematodeKind::Omnivore => [0.68, 0.64, 0.54],
    };
    TerrariumNematodeVisualResponse {
        body_rgb: quantize_rgb(
            lerp_rgb(
                base_rgb,
                [0.92, 0.88, 0.76],
                activity * 0.20 + local_air.humidity * 0.08,
            ),
            5,
        ),
        head_rgb: quantize_rgb(
            lerp_rgb(
                base_rgb,
                [0.54, 0.34, 0.18],
                if body.has_stylet { 0.32 } else { 0.14 },
            ),
            5,
        ),
        length_scale: (0.54 + body.length_mm * 0.50 + density_t * 0.22 + activity * 0.18)
            .clamp(0.46, 1.58),
        thickness_scale: (0.40 + (body.diameter_um / 120.0) * 0.60 + biomass_t * 0.12)
            .clamp(0.26, 1.18),
        curl: clamp(0.10 + undulation * 0.24 + density_t * 0.10, -0.36, 0.48),
        yaw_rad: undulation * 0.42 + local_air.wind_z * 0.28,
        height_offset: local_air.pressure_delta_kpa * 0.002 + density_t * 0.008,
        stylet_length_scale: if body.has_stylet {
            (body.stylet_length_um / 24.0).clamp(0.18, 1.0)
        } else {
            0.0
        },
        activity,
        provenance: TerrariumVisualProvenance::RuntimeProjection,
        uncertainty: visual_uncertainty(0.26 + (1.0 - activity) * 0.05 + density_t * 0.02),
    }
}

fn soil_surface_oxygen_t(chemistry: TerrariumVisualChemistrySample) -> f32 {
    ((chemistry.redox + 1.0) * 0.5).clamp(0.0, 1.0)
}

fn soil_surface_base_rgb(
    moisture: f32,
    organic: f32,
    chemistry: TerrariumVisualChemistrySample,
) -> [f32; 3] {
    let base = emergent_color::emergent_soil_base_color(moisture, organic);
    emergent_color::emergent_terrain_chemistry_tint(
        chemistry.ferric + chemistry.ferrous * 0.5,
        chemistry.silicate,
        chemistry.clay,
        chemistry.carbonate + chemistry.alkalinity * 0.4,
        chemistry.acidity,
        base,
    )
}

fn select_soil_surface_class(
    moisture: f32,
    organic: f32,
    microbial_biomass: f32,
    symbiont_biomass: f32,
    earthworm_density: f32,
    nematode_density: f32,
    atp_flux: f32,
    nitrification_potential: f32,
    denitrification_potential: f32,
    shoreline_signal: f32,
    chemistry: TerrariumVisualChemistrySample,
) -> (TerrariumSoilSurfaceClass, f32) {
    let oxygen_t = soil_surface_oxygen_t(chemistry);
    let organic_t = organic.clamp(0.0, 1.0);
    let moisture_t = moisture.clamp(0.0, 1.0);
    let nitrifier_process_t = clamp(nitrification_potential / 0.000025, 0.0, 1.0);
    let denitrifier_process_t = clamp(denitrification_potential / 0.00003, 0.0, 1.0);
    let nitrifier_bias = nitrifier_process_t
        * (0.44
            + oxygen_t * 0.34
            + chemistry.alkalinity * 0.16
            + chemistry.carbonate * 0.08
            + chemistry.clarity * 0.06
            + shoreline_signal * 0.08);
    let denitrifier_bias = denitrifier_process_t
        * (0.42
            + (1.0 - oxygen_t) * 0.34
            + chemistry.ferrous * 0.18
            + (1.0 - chemistry.clarity) * 0.08
            + shoreline_signal * 0.12);
    let generic_specialization_suppression =
        (1.0 - nitrifier_bias.max(denitrifier_bias) * 0.22).clamp(0.58, 1.0);
    let microbial_signal = clamp(
        (microbial_biomass / 0.06).max(atp_flux / 0.0009),
        0.0,
        1.0,
    ) * (0.60 + oxygen_t * 0.14 + chemistry.clarity * 0.10 + organic_t * 0.16)
        * generic_specialization_suppression;
    let nitrifier_signal = nitrifier_bias;
    let denitrifier_signal = denitrifier_bias;
    let mycorrhizal_signal = clamp(symbiont_biomass / 0.04, 0.0, 1.0)
        * (0.55 + organic_t * 0.22 + chemistry.silicate * 0.14 + chemistry.clay * 0.09);
    let earthworm_signal = clamp(earthworm_density / 8.0, 0.0, 1.0)
        * (0.60 + organic_t * 0.20 + chemistry.clay * 0.20);
    let nematode_signal = clamp(nematode_density / 1.2, 0.0, 1.0)
        * (0.54 + moisture_t * 0.16 + organic_t * 0.12 + (1.0 - oxygen_t) * 0.18);
    let wet_signal = moisture_t
        * (0.64 + shoreline_signal * 0.16 + (1.0 - chemistry.clarity) * 0.10 + chemistry.clay * 0.10);
    let humus_signal = organic_t * (0.72 + chemistry.ferric * 0.12 + chemistry.clay * 0.16);
    let mineral_signal = (1.0 - organic_t * 0.82).clamp(0.12, 1.0)
        * (0.58 + chemistry.silicate * 0.24 + chemistry.carbonate * 0.18)
        * (1.0 - moisture_t * 0.35).clamp(0.22, 1.0);

    let mut best = (TerrariumSoilSurfaceClass::Mineral, mineral_signal);
    for candidate in [
        (TerrariumSoilSurfaceClass::Humus, humus_signal),
        (TerrariumSoilSurfaceClass::WetDetritus, wet_signal),
        (TerrariumSoilSurfaceClass::MicrobialMat, microbial_signal),
        (TerrariumSoilSurfaceClass::NitrifierCrust, nitrifier_signal),
        (TerrariumSoilSurfaceClass::DenitrifierFilm, denitrifier_signal),
        (TerrariumSoilSurfaceClass::MycorrhizalPatch, mycorrhizal_signal),
        (TerrariumSoilSurfaceClass::EarthwormCast, earthworm_signal),
        (TerrariumSoilSurfaceClass::NematodeBloom, nematode_signal),
    ] {
        if candidate.1 > best.1 {
            best = candidate;
        }
    }
    best
}

fn soil_surface_palette(
    class: TerrariumSoilSurfaceClass,
    local_air: TerrariumVisualAirSample,
    chemistry: TerrariumVisualChemistrySample,
    moisture: f32,
    organic: f32,
    activity: f32,
    energy_t: f32,
    abundance_t: f32,
) -> ([f32; 3], [f32; 3], f32, f32) {
    let base_soil = soil_surface_base_rgb(moisture, organic, chemistry);
    let oxygen_t = soil_surface_oxygen_t(chemistry);
    let wet_dark = emergent_color::emergent_soil_base_color(0.95, 0.02);
    let microbial_body = emergent_color::emergent_microbe_body_color(activity, energy_t, oxygen_t);
    let nitrifier_body = emergent_color::beer_lambert_mix(
        &[
            (
                TerrariumSpecies::Nitrate,
                0.08 + oxygen_t * 0.10 + chemistry.alkalinity * 0.12,
            ),
            (
                TerrariumSpecies::CarbonateMineral,
                chemistry.carbonate * 0.18 + chemistry.alkalinity * 0.08,
            ),
        ],
        emergent_color::emergent_microbe_body_color(activity, energy_t, oxygen_t),
    );
    let denitrifier_body =
        emergent_color::beer_lambert_mix(
            &[
                (
                    TerrariumSpecies::AqueousIronPool,
                    0.08 + chemistry.ferrous * 0.20 + (1.0 - oxygen_t) * 0.10,
                ),
                (
                    TerrariumSpecies::BicarbonatePool,
                    chemistry.alkalinity * 0.16 + (1.0 - chemistry.clarity) * 0.08,
                ),
            ],
            emergent_color::emergent_microbe_body_color(activity, energy_t, oxygen_t.min(0.28)),
        );
    let mycorrhizal_body = emergent_color::beer_lambert_mix(
        &[(TerrariumSpecies::Glucose, 0.22 + organic.clamp(0.0, 1.0) * 0.30)],
        emergent_color::emergent_soil_base_color(moisture.max(0.22), organic.max(0.55)),
    );
    let earthworm_body =
        emergent_color::emergent_earthworm_body_color(abundance_t, (abundance_t + chemistry.clay * 0.2).clamp(0.0, 1.0));
    let nematode_body = emergent_color::emergent_nematode_body_color(
        false,
        false,
        activity,
        local_air.humidity,
    );

    match class {
        TerrariumSoilSurfaceClass::Mineral => {
            let accent = emergent_color::beer_lambert_mix_with_scattering(
                &[
                    (TerrariumSpecies::SilicateMineral, 0.42 + chemistry.silicate * 0.30),
                    (TerrariumSpecies::CarbonateMineral, chemistry.carbonate * 0.22),
                ],
                base_soil,
            );
            (lerp_rgb(base_soil, accent, 0.24), accent, 0.10, 0.42)
        }
        TerrariumSoilSurfaceClass::Humus => {
            let humus = lerp_rgb(
                base_soil,
                emergent_color::emergent_soil_base_color(moisture.max(0.18), organic.max(0.70)),
                0.44,
            );
            let accent = emergent_color::beer_lambert_mix(
                &[
                    (TerrariumSpecies::Glucose, 0.24 + organic.clamp(0.0, 1.0) * 0.28),
                    (TerrariumSpecies::IronOxideMineral, chemistry.ferric * 0.18),
                ],
                humus,
            );
            (humus, accent, 0.24, 0.50)
        }
        TerrariumSoilSurfaceClass::WetDetritus => {
            let detritus = lerp_rgb(base_soil, wet_dark, 0.38 + moisture.clamp(0.0, 1.0) * 0.18);
            let accent = emergent_color::beer_lambert_mix(
                &[
                    (TerrariumSpecies::Water, moisture.clamp(0.0, 1.0) * 0.55),
                    (TerrariumSpecies::Glucose, 0.12 + organic.clamp(0.0, 1.0) * 0.18),
                ],
                detritus,
            );
            (detritus, accent, 0.28, 0.56)
        }
        TerrariumSoilSurfaceClass::MicrobialMat => {
            let rgb = lerp_rgb(base_soil, microbial_body, 0.52);
            let accent = emergent_color::emergent_microbe_body_color(
                (activity + 0.18).clamp(0.0, 1.0),
                (energy_t + 0.14).clamp(0.0, 1.0),
                oxygen_t,
            );
            (rgb, accent, 0.46, 0.62)
        }
        TerrariumSoilSurfaceClass::NitrifierCrust => {
            let accent = emergent_color::beer_lambert_mix(
                &[
                    (
                        TerrariumSpecies::Nitrate,
                        0.22 + oxygen_t * 0.16 + chemistry.alkalinity * 0.18,
                    ),
                    (
                        TerrariumSpecies::CarbonateMineral,
                        chemistry.carbonate * 0.18 + chemistry.alkalinity * 0.10,
                    ),
                    (
                        TerrariumSpecies::AqueousIronPool,
                        chemistry.ferrous * 0.22 + chemistry.acidity * 0.08,
                    ),
                    (TerrariumSpecies::Water, moisture.clamp(0.0, 1.0) * 0.20),
                ],
                nitrifier_body,
            );
            let crust = lerp_rgb(
                lerp_rgb(
                    base_soil,
                    wet_dark,
                    chemistry.ferrous * 0.18 + chemistry.acidity * 0.10,
                ),
                accent,
                (0.42 + chemistry.alkalinity * 0.10 + chemistry.carbonate * 0.06
                    - chemistry.ferrous * 0.08)
                    .clamp(0.28, 0.62),
            );
            (crust, accent, 0.42, 0.60)
        }
        TerrariumSoilSurfaceClass::DenitrifierFilm => {
            let accent = emergent_color::beer_lambert_mix(
                &[
                    (TerrariumSpecies::BicarbonatePool, 0.10 + chemistry.alkalinity * 0.24),
                    (TerrariumSpecies::AqueousIronPool, 0.08 + chemistry.ferrous * 0.26),
                    (
                        TerrariumSpecies::Water,
                        moisture.clamp(0.0, 1.0) * 0.12 + (1.0 - chemistry.clarity) * 0.10,
                    ),
                ],
                denitrifier_body,
            );
            let film = lerp_rgb(
                lerp_rgb(base_soil, wet_dark, (1.0 - chemistry.clarity) * 0.18),
                accent,
                (0.46 + chemistry.ferrous * 0.10 + (1.0 - oxygen_t) * 0.08).clamp(0.34, 0.66),
            );
            (film, accent, 0.44, 0.62)
        }
        TerrariumSoilSurfaceClass::MycorrhizalPatch => {
            let accent = emergent_color::beer_lambert_mix(
                &[
                    (TerrariumSpecies::Glucose, 0.28 + organic.clamp(0.0, 1.0) * 0.20),
                    (TerrariumSpecies::Water, moisture.clamp(0.0, 1.0) * 0.12),
                ],
                mycorrhizal_body,
            );
            (lerp_rgb(base_soil, mycorrhizal_body, 0.40), accent, 0.38, 0.58)
        }
        TerrariumSoilSurfaceClass::EarthwormCast => {
            let accent = emergent_color::beer_lambert_mix(
                &[
                    (TerrariumSpecies::ClayMineral, 0.16 + chemistry.clay * 0.34),
                    (TerrariumSpecies::IronOxideMineral, 0.06 + chemistry.ferric * 0.18),
                ],
                earthworm_body,
            );
            (lerp_rgb(base_soil, accent, 0.36), accent, 0.32, 0.60)
        }
        TerrariumSoilSurfaceClass::NematodeBloom => {
            let accent = emergent_color::emergent_nematode_head_color(false, nematode_body);
            (lerp_rgb(base_soil, nematode_body, 0.30), accent, 0.36, 0.52)
        }
    }
}

pub fn soil_surface_visual_response(
    local_air: TerrariumVisualAirSample,
    chemistry: TerrariumVisualChemistrySample,
    moisture: f32,
    organic: f32,
    microbial_biomass: f32,
    symbiont_biomass: f32,
    earthworm_density: f32,
    nematode_density: f32,
    atp_flux: f32,
    nitrification_potential: f32,
    denitrification_potential: f32,
    shoreline_signal: f32,
) -> TerrariumSoilSurfaceVisualResponse {
    let (class, dominant_signal) = select_soil_surface_class(
        moisture,
        organic,
        microbial_biomass,
        symbiont_biomass,
        earthworm_density,
        nematode_density,
        atp_flux,
        nitrification_potential,
        denitrification_potential,
        shoreline_signal,
        chemistry,
    );
    let oxygen_t = soil_surface_oxygen_t(chemistry);
    let activity = clamp(
        (microbial_biomass / 0.06).max(atp_flux / 0.0009)
            * (0.72 + oxygen_t * 0.14 + local_air.humidity * 0.14),
        0.0,
        1.0,
    );
    let energy_t = clamp(
        atp_flux / 0.0009
            + microbial_biomass / 0.08
            + nitrification_potential / 0.00003
            + denitrification_potential / 0.00003,
        0.0,
        1.0,
    );
    let abundance_t = clamp(dominant_signal, 0.0, 1.0);
    let (rgb, accent_rgb, density_base, sprite_base) = soil_surface_palette(
        class,
        local_air,
        chemistry,
        moisture,
        organic,
        activity,
        energy_t,
        abundance_t,
    );
    let moisture_lift = moisture.clamp(0.0, 1.0) * 0.05 + local_air.pressure_delta_kpa * 0.004;
    let width_scale = clamp(
        0.70 + organic.clamp(0.0, 1.0) * 0.48
            + symbiont_biomass * 3.5
            + nitrification_potential * 3800.0
            + shoreline_signal * 0.20
            + earthworm_density * 0.015
            + chemistry.clay * 0.22
            + local_air.wind_x.abs() * 0.08,
        0.45,
        1.95,
    );
    let depth_scale = clamp(
        0.70 + moisture.clamp(0.0, 1.0) * 0.38
            + microbial_biomass * 2.8
            + denitrification_potential * 4200.0
            + shoreline_signal * 0.26
            + nematode_density * 0.035
            + (1.0 - oxygen_t) * 0.18
            + local_air.wind_z.abs() * 0.08,
        0.45,
        1.95,
    );
    let thickness_scale = clamp(
        0.42 + density_base * 0.62 + dominant_signal * 0.26
            + atp_flux * 180.0
            + nitrification_potential * 2800.0
            + denitrification_potential * 3200.0
            + earthworm_density * 0.018,
        0.35,
        1.65,
    );
    let yaw_rad = local_air.wind_z.atan2(local_air.wind_x + 1e-6)
        + (earthworm_density * 0.02 - nematode_density * 0.03)
        + shoreline_signal * 0.18
        - denitrification_potential * 2600.0 * 0.03;
    let wet_dark = emergent_color::emergent_soil_base_color(0.95, 0.02);
    let uncertainty = 0.62
        - dominant_signal * 0.12
        + (1.0 - chemistry.clarity) * 0.08
        + shoreline_signal * 0.04;
    TerrariumSoilSurfaceVisualResponse {
        class,
        rgb: quantize_rgb(
            lerp_rgb(rgb, wet_dark, local_air.humidity * 0.08 + moisture.clamp(0.0, 1.0) * 0.06),
            7,
        ),
        accent_rgb: quantize_rgb(accent_rgb, 6),
        density: (density_base + dominant_signal * 0.18 + organic.clamp(0.0, 1.0) * 0.08)
            .clamp(0.0, 1.0),
        height_offset: moisture_lift + shoreline_signal * 0.018,
        sprite_scale: (sprite_base + dominant_signal * 0.08 + local_air.wind_speed * 0.04)
            .clamp(0.35, 0.85),
        width_scale,
        depth_scale,
        thickness_scale,
        yaw_rad,
        provenance: TerrariumVisualProvenance::CoarseFieldProjection,
        uncertainty: visual_uncertainty(uncertainty),
    }
}

pub fn soil_surface_visual_response_from_packet_ecology(
    ecology: GenotypePacketEcology,
    local_air: TerrariumVisualAirSample,
    chemistry: TerrariumVisualChemistrySample,
    moisture: f32,
    organic: f32,
    activity: f32,
    diversity: f32,
    total_cells: f32,
    shoreline_signal: f32,
) -> TerrariumSoilSurfaceVisualResponse {
    let class = match ecology {
        GenotypePacketEcology::Decomposer => TerrariumSoilSurfaceClass::MicrobialMat,
        GenotypePacketEcology::Nitrifier => TerrariumSoilSurfaceClass::NitrifierCrust,
        GenotypePacketEcology::Denitrifier => TerrariumSoilSurfaceClass::DenitrifierFilm,
    };
    let density = clamp(
        0.34 + activity * 0.20
            + diversity * 0.18
            + shoreline_signal * 0.16
            + (total_cells / 1.2e6).sqrt().clamp(0.0, 1.0) * 0.12,
        0.24,
        0.98,
    );
    let energy_t = (total_cells / 1.2e6).sqrt().clamp(0.0, 1.0);
    let (rgb, accent_rgb, density_base, sprite_base) = soil_surface_palette(
        class,
        local_air,
        chemistry,
        moisture,
        organic,
        activity.clamp(0.0, 1.0),
        energy_t,
        density,
    );
    let oxygen_t = soil_surface_oxygen_t(chemistry);
    let chemistry_rgb = match ecology {
        GenotypePacketEcology::Decomposer => emergent_color::beer_lambert_mix(
            &[
                (TerrariumSpecies::Glucose, 0.10 + organic.clamp(0.0, 1.0) * 0.16),
                (TerrariumSpecies::Water, moisture.clamp(0.0, 1.0) * 0.12),
            ],
            soil_surface_base_rgb(moisture, organic, chemistry),
        ),
        GenotypePacketEcology::Nitrifier => emergent_color::beer_lambert_mix(
            &[
                (
                    TerrariumSpecies::Nitrate,
                    0.14 + oxygen_t * 0.18 + chemistry.alkalinity * 0.22,
                ),
                (
                    TerrariumSpecies::CarbonateMineral,
                    chemistry.carbonate * 0.28 + chemistry.alkalinity * 0.12,
                ),
                (
                    TerrariumSpecies::AqueousIronPool,
                    chemistry.ferrous * 0.34 + chemistry.acidity * 0.12,
                ),
            ],
            soil_surface_base_rgb(moisture, organic, chemistry),
        ),
        GenotypePacketEcology::Denitrifier => emergent_color::beer_lambert_mix(
            &[
                (
                    TerrariumSpecies::BicarbonatePool,
                    chemistry.alkalinity * 0.20 + (1.0 - chemistry.clarity) * 0.10,
                ),
                (
                    TerrariumSpecies::AqueousIronPool,
                    0.10 + chemistry.ferrous * 0.30 + (1.0 - oxygen_t) * 0.10,
                ),
                (
                    TerrariumSpecies::Water,
                    moisture.clamp(0.0, 1.0) * 0.12 + (1.0 - chemistry.clarity) * 0.10,
                ),
            ],
            soil_surface_base_rgb(moisture, organic, chemistry),
        ),
    };
    let chemistry_consistency = match ecology {
        GenotypePacketEcology::Decomposer => {
            0.42 + organic.clamp(0.0, 1.0) * 0.28 + moisture.clamp(0.0, 1.0) * 0.14
        }
        GenotypePacketEcology::Nitrifier => {
            0.34
                + oxygen_t * 0.26
                + chemistry.alkalinity * 0.18
                + chemistry.clarity * 0.10
                + chemistry.carbonate * 0.10
                - chemistry.ferrous * 0.12
        }
        GenotypePacketEcology::Denitrifier => {
            0.34
                + (1.0 - oxygen_t) * 0.26
                + chemistry.ferrous * 0.18
                + (1.0 - chemistry.clarity) * 0.12
                + moisture.clamp(0.0, 1.0) * 0.08
        }
    }
    .clamp(0.0, 1.0);
    let uncertainty = 0.40 + (1.0 - density) * 0.10 + (1.0 - chemistry_consistency) * 0.18;
    TerrariumSoilSurfaceVisualResponse {
        class,
        rgb: quantize_rgb(lerp_rgb(rgb, chemistry_rgb, 0.44), 7),
        accent_rgb: quantize_rgb(accent_rgb, 6),
        density: (density_base + density * 0.42).clamp(0.0, 1.0),
        height_offset: density * 0.018 + shoreline_signal * 0.012,
        sprite_scale: (sprite_base + density * 0.06).clamp(0.35, 0.85),
        width_scale: (0.68 + density * 0.34 + diversity * 0.16).clamp(0.54, 1.28),
        depth_scale: (0.68 + density * 0.28 + shoreline_signal * 0.20).clamp(0.54, 1.32),
        thickness_scale: (0.16 + density * 0.14 + activity * 0.08).clamp(0.12, 0.42),
        yaw_rad: (diversity - 0.5) * 0.36,
        provenance: TerrariumVisualProvenance::OwnedRegionProjection,
        uncertainty: visual_uncertainty(uncertainty),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn water_visual_response_preserves_lunar_and_atmospheric_drivers() {
        let baseline_air = TerrariumVisualAirSample {
            temperature_c: 18.0,
            humidity: 0.58,
            pressure_kpa: 101.325,
            pressure_delta_kpa: 0.0,
            wind_x: 0.0,
            wind_y: 0.0,
            wind_z: 0.0,
            wind_speed: 0.0,
        };
        let windy_hot_air = TerrariumVisualAirSample {
            temperature_c: 29.0,
            humidity: 0.82,
            pressure_kpa: 102.1,
            pressure_delta_kpa: 0.8,
            wind_x: 0.9,
            wind_y: -0.6,
            wind_z: 0.2,
            wind_speed: 1.1,
        };

        let neap_cycle = TerrariumWaterCycleInputs {
            lunar_phase: 0.25,
            moonlight: 0.18,
            tidal_moisture_factor: 0.85,
        };
        let spring_cycle = TerrariumWaterCycleInputs {
            lunar_phase: 0.5,
            moonlight: 1.0,
            tidal_moisture_factor: 1.15,
        };

        let neap = water_visual_response(baseline_air, 80.0, 12.0, 3.0, neap_cycle);
        let spring = water_visual_response(baseline_air, 80.0, 12.0, 3.0, spring_cycle);
        assert!(
            (spring.vertical_offset - neap.vertical_offset).abs() > 1.0e-4
                || (spring.thickness_scale - neap.thickness_scale).abs() > 1.0e-4
                || (spring.radius_y_cells - neap.radius_y_cells).abs() > 1.0e-4
        );

        let calm = water_visual_response(baseline_air, 80.0, 12.0, 3.0, spring_cycle);
        let storm = water_visual_response(windy_hot_air, 80.0, 12.0, 3.0, spring_cycle);
        assert!(
            (storm.tilt_x - calm.tilt_x).abs() > 1.0e-4
                || (storm.tilt_z - calm.tilt_z).abs() > 1.0e-4
                || (storm.thickness_scale - calm.thickness_scale).abs() > 1.0e-4
                || (storm.radius_x_cells - calm.radius_x_cells).abs() > 1.0e-4
        );
    }

    #[test]
    fn terrain_visual_response_shifts_with_chemistry() {
        let air = TerrariumVisualAirSample {
            temperature_c: 18.0,
            humidity: 0.42,
            pressure_kpa: 101.325,
            pressure_delta_kpa: 0.0,
            wind_x: 0.0,
            wind_y: 0.0,
            wind_z: 0.0,
            wind_speed: 0.0,
        };
        let carbonate = TerrariumVisualChemistrySample {
            alkalinity: 0.82,
            carbonate: 0.88,
            clarity: 0.90,
            ..TerrariumVisualChemistrySample::default()
        };
        let ferric = TerrariumVisualChemistrySample {
            ferric: 0.92,
            redox: 0.54,
            clarity: 0.48,
            ..TerrariumVisualChemistrySample::default()
        };

        let carbonate_rgb = terrain_visual_response_with_chemistry(0.26, 0.14, air, carbonate, 1.0).rgb;
        let ferric_rgb = terrain_visual_response_with_chemistry(0.26, 0.14, air, ferric, 1.0).rgb;

        let carbonate_luma = carbonate_rgb[0] + carbonate_rgb[1] + carbonate_rgb[2];
        let ferric_luma = ferric_rgb[0] + ferric_rgb[1] + ferric_rgb[2];
        assert!(carbonate_luma > ferric_luma);
        assert!(ferric_rgb[0] > ferric_rgb[1]);
    }

    #[test]
    fn water_visual_response_tracks_chemistry_clarity_and_iron() {
        let air = TerrariumVisualAirSample {
            temperature_c: 20.0,
            humidity: 0.60,
            pressure_kpa: 101.325,
            pressure_delta_kpa: 0.0,
            wind_x: 0.0,
            wind_y: 0.0,
            wind_z: 0.0,
            wind_speed: 0.0,
        };
        let cycle = TerrariumWaterCycleInputs {
            lunar_phase: 0.4,
            moonlight: 0.3,
            tidal_moisture_factor: 1.0,
        };
        let clear = TerrariumVisualChemistrySample {
            alkalinity: 0.74,
            carbonate: 0.58,
            clarity: 0.94,
            redox: 0.36,
            ..TerrariumVisualChemistrySample::default()
        };
        let iron_rich = TerrariumVisualChemistrySample {
            ferric: 0.36,
            ferrous: 0.88,
            acidity: 0.26,
            redox: -0.42,
            clarity: 0.26,
            ..TerrariumVisualChemistrySample::default()
        };

        let clear_visual = water_visual_response_with_chemistry(air, 90.0, 4.0, 1.0, cycle, clear);
        let iron_visual =
            water_visual_response_with_chemistry(air, 90.0, 4.0, 1.0, cycle, iron_rich);

        assert!(clear_visual.clarity > iron_visual.clarity);
        assert!(clear_visual.rgb[2] > iron_visual.rgb[2]);
        assert!(iron_visual.rgb[0] > clear_visual.rgb[0]);
    }

    #[test]
    fn open_water_basin_mask_concentrates_on_pooled_water() {
        let width = 5usize;
        let height = 5usize;
        let mut water = vec![0.0f32; width * height];
        water[2 * width + 2] = 0.82;
        water[2 * width + 1] = 0.34;
        water[2 * width + 3] = 0.30;
        water[1 * width + 2] = 0.26;
        water[3 * width + 2] = 0.28;
        water[0] = 0.10;

        let basins = open_water_basin_mask(width, height, &water);
        assert!(basins[2 * width + 2] > 0.7);
        assert!(basins[2 * width + 1] > 0.16);
        assert!(basins[0] < 0.08);
    }

    #[test]
    fn projected_surface_relief_cuts_pooled_water_more_than_trace_wetness() {
        let width = 5usize;
        let height = 5usize;
        let soil = vec![0.52f32; width * height];
        let mut water = vec![0.0f32; width * height];
        water[2 * width + 2] = 0.78;
        water[2 * width + 1] = 0.30;
        water[2 * width + 3] = 0.28;
        water[1 * width + 2] = 0.24;
        water[3 * width + 2] = 0.26;
        water[0] = 0.10;

        let projected = projected_surface_relief(width, height, &soil, &water);
        assert!(projected[2 * width + 2] < projected[0]);
        assert!(projected[2 * width + 1] < soil[2 * width + 1]);
    }

    #[test]
    fn missing_provenance_deserializes_to_conservative_fallback() {
        let visual: TerrariumPlantVisualResponse = serde_json::from_str(
            r#"{
                "stem_rgb":[0.2,0.3,0.1],
                "leaf_rgb":[0.1,0.5,0.2],
                "fruit_rgb":[0.8,0.5,0.2],
                "sway_x":0.0,
                "sway_z":0.0,
                "lean_x":0.0,
                "lean_z":0.0,
                "vertical_offset":0.0,
                "canopy_scale":1.0
            }"#,
        )
        .expect("plant visual without provenance should deserialize");

        assert_eq!(
            visual.provenance,
            TerrariumVisualProvenance::PresentationFallback
        );
        assert_eq!(visual.uncertainty, 1.0);
    }

    #[test]
    fn visual_responses_report_expected_authority_sources() {
        let air = TerrariumVisualAirSample {
            temperature_c: 18.0,
            humidity: 0.42,
            pressure_kpa: 101.325,
            pressure_delta_kpa: 0.0,
            wind_x: 0.1,
            wind_y: -0.1,
            wind_z: 0.0,
            wind_speed: 0.2,
        };
        let terrain = terrain_visual_response(0.26, 0.14, air);
        let plant = plant_visual_response(7, air, 0.82, 0.48, 12.0, 0.3);
        let soil_owned = soil_surface_visual_response_from_packet_ecology(
            GenotypePacketEcology::Nitrifier,
            air,
            TerrariumVisualChemistrySample::default(),
            0.32,
            0.14,
            0.72,
            0.44,
            8.0e5,
            0.15,
        );

        assert_eq!(
            terrain.provenance,
            TerrariumVisualProvenance::CoarseFieldProjection
        );
        assert_eq!(
            plant.provenance,
            TerrariumVisualProvenance::RuntimeProjection
        );
        assert_eq!(
            soil_owned.provenance,
            TerrariumVisualProvenance::OwnedRegionProjection
        );
        assert!(terrain.uncertainty > soil_owned.uncertainty);
        assert!(soil_owned.uncertainty > plant.uncertainty);
    }

    #[test]
    fn plant_visual_response_uses_molecular_pigments_when_available() {
        let rgb_distance = |a: [f32; 3], b: [f32; 3]| -> f32 {
            (a[0] - b[0]).abs() + (a[1] - b[1]).abs() + (a[2] - b[2]).abs()
        };
        let air = TerrariumVisualAirSample {
            temperature_c: 22.0,
            humidity: 0.55,
            pressure_kpa: 101.325,
            pressure_delta_kpa: 0.0,
            wind_x: 0.0,
            wind_y: 0.0,
            wind_z: 0.0,
            wind_speed: 0.0,
        };
        let stressed_fallback = plant_visual_response(7, air, 0.18, 0.26, 12.0, 0.2);
        let molecular = MolecularVisualState {
            chlorophyll_density: 0.92,
            anthocyanin_density: 0.02,
            carotenoid_density: 0.08,
            turgor_pressure: 0.94,
            senescence_progress: 0.04,
            terpene_level: 0.0,
        };
        let explicit = plant_visual_response_with_molecular_state(
            7,
            air,
            0.18,
            0.26,
            12.0,
            0.2,
            Some(molecular),
        );

        assert!(
            rgb_distance(explicit.leaf_rgb, molecular.leaf_rgb())
                < rgb_distance(stressed_fallback.leaf_rgb, molecular.leaf_rgb())
        );
        assert!(explicit.canopy_scale > stressed_fallback.canopy_scale);
    }

    #[test]
    fn fruit_visual_response_uses_parent_molecular_pigments_when_available() {
        let rgb_distance = |a: [f32; 3], b: [f32; 3]| -> f32 {
            (a[0] - b[0]).abs() + (a[1] - b[1]).abs() + (a[2] - b[2]).abs()
        };
        let air = TerrariumVisualAirSample {
            temperature_c: 24.0,
            humidity: 0.52,
            pressure_kpa: 101.325,
            pressure_delta_kpa: 0.0,
            wind_x: 0.0,
            wind_y: 0.0,
            wind_z: 0.0,
            wind_speed: 0.0,
        };
        let taxonomy_only = fruit_visual_response(11, air, 0.82, 0.74, 8.0, 0.4, 1.0);
        let molecular = MolecularVisualState {
            chlorophyll_density: 0.18,
            anthocyanin_density: 0.86,
            carotenoid_density: 0.12,
            turgor_pressure: 0.88,
            senescence_progress: 0.02,
            terpene_level: 0.0,
        };
        let explicit = fruit_visual_response_with_molecular_state(
            11,
            air,
            0.82,
            0.74,
            8.0,
            0.4,
            1.0,
            Some(molecular),
        );

        assert!(
            rgb_distance(explicit.skin_rgb, molecular.fruit_rgb())
                < rgb_distance(taxonomy_only.skin_rgb, molecular.fruit_rgb())
        );
    }

    #[test]
    fn soil_surface_visual_response_tracks_local_chemistry_and_dominant_process() {
        let air = TerrariumVisualAirSample {
            temperature_c: 19.0,
            humidity: 0.62,
            pressure_kpa: 101.325,
            pressure_delta_kpa: 0.0,
            wind_x: 0.0,
            wind_y: 0.0,
            wind_z: 0.0,
            wind_speed: 0.0,
        };
        let oxidized = TerrariumVisualChemistrySample {
            alkalinity: 0.64,
            silicate: 0.18,
            clay: 0.24,
            redox: 0.72,
            clarity: 0.86,
            ..TerrariumVisualChemistrySample::default()
        };
        let reduced = TerrariumVisualChemistrySample {
            ferric: 0.12,
            ferrous: 0.88,
            clay: 0.32,
            redox: -0.82,
            clarity: 0.30,
            ..TerrariumVisualChemistrySample::default()
        };

        let nitrifier = soil_surface_visual_response(
            air, oxidized, 0.48, 0.18, 0.08, 0.01, 0.2, 0.3, 0.0012, 0.00006, 0.0, 0.18,
        );
        let denitrifier = soil_surface_visual_response(
            air, reduced, 0.62, 0.22, 0.06, 0.01, 0.2, 0.3, 0.0010, 0.0, 0.00008, 0.34,
        );

        assert_eq!(nitrifier.class, TerrariumSoilSurfaceClass::NitrifierCrust);
        assert_eq!(denitrifier.class, TerrariumSoilSurfaceClass::DenitrifierFilm);
        assert_ne!(nitrifier.rgb, denitrifier.rgb);
        assert_eq!(
            nitrifier.provenance,
            TerrariumVisualProvenance::CoarseFieldProjection
        );
    }

    #[test]
    fn packet_ecology_soil_surface_visuals_shift_with_local_chemistry() {
        let air = TerrariumVisualAirSample {
            temperature_c: 20.0,
            humidity: 0.58,
            pressure_kpa: 101.325,
            pressure_delta_kpa: 0.0,
            wind_x: 0.1,
            wind_y: -0.1,
            wind_z: 0.0,
            wind_speed: 0.2,
        };
        let buffered = TerrariumVisualChemistrySample {
            alkalinity: 0.72,
            carbonate: 0.44,
            silicate: 0.20,
            redox: 0.46,
            clarity: 0.90,
            ..TerrariumVisualChemistrySample::default()
        };
        let iron_rich = TerrariumVisualChemistrySample {
            ferric: 0.20,
            ferrous: 0.74,
            acidity: 0.28,
            redox: -0.64,
            clarity: 0.34,
            ..TerrariumVisualChemistrySample::default()
        };

        let clear = soil_surface_visual_response_from_packet_ecology(
            GenotypePacketEcology::Nitrifier,
            air,
            buffered,
            0.36,
            0.14,
            0.74,
            0.42,
            9.5e5,
            0.10,
        );
        let boggy = soil_surface_visual_response_from_packet_ecology(
            GenotypePacketEcology::Nitrifier,
            air,
            iron_rich,
            0.36,
            0.14,
            0.74,
            0.42,
            9.5e5,
            0.10,
        );

        assert_eq!(clear.class, TerrariumSoilSurfaceClass::NitrifierCrust);
        assert_eq!(boggy.class, TerrariumSoilSurfaceClass::NitrifierCrust);
        assert_ne!(clear.rgb, boggy.rgb);
        assert_eq!(
            clear.provenance,
            TerrariumVisualProvenance::OwnedRegionProjection
        );
    }

    #[test]
    fn blend_zero_preserves_base_color() {
        let air = TerrariumVisualAirSample {
            temperature_c: 20.0,
            humidity: 0.5,
            pressure_kpa: 101.325,
            ..TerrariumVisualAirSample::default()
        };
        // Strong chemistry signal that would normally shift colors significantly
        let chemistry = TerrariumVisualChemistrySample {
            ferric: 1.5,
            redox: 0.8,
            acidity: 0.5,
            silicate: 0.3,
            clay: 0.6,
            ..TerrariumVisualChemistrySample::default()
        };
        let base_only = terrain_visual_response_with_chemistry(0.3, 0.2, air, chemistry, 0.0);
        // At blend=0.0, output should equal the simple CPK base color (within quantization)
        let expected_base = quantize_rgb(emergent_color::emergent_soil_base_color(0.3, 0.2), 10);
        for c in 0..3 {
            assert!(
                (base_only.rgb[c] - expected_base[c]).abs() < 0.12,
                "blend=0 should match base: ch{c} got {:.3} expected {:.3}",
                base_only.rgb[c], expected_base[c]
            );
        }
    }

    #[test]
    fn blend_one_uses_emergent_chemistry() {
        let air = TerrariumVisualAirSample {
            temperature_c: 20.0,
            humidity: 0.5,
            pressure_kpa: 101.325,
            ..TerrariumVisualAirSample::default()
        };
        // Strong ferric + clay + carbonate to force visible chemistry tinting
        let chemistry = TerrariumVisualChemistrySample {
            ferric: 1.5,
            redox: 0.8,
            acidity: 0.5,
            silicate: 0.3,
            clay: 0.8,
            carbonate: 0.6,
            alkalinity: 0.4,
            clarity: 0.5,
            ..TerrariumVisualChemistrySample::default()
        };
        let base_only = terrain_visual_response_with_chemistry(0.3, 0.2, air, chemistry, 0.0);
        let full_emergent = terrain_visual_response_with_chemistry(0.3, 0.2, air, chemistry, 1.0);
        // With very strong chemistry signal, blend=1 should shift the color
        let diff: f32 = (0..3).map(|c| (base_only.rgb[c] - full_emergent.rgb[c]).abs()).sum();
        assert!(
            diff > 0.05,
            "blend=1 should differ from blend=0 with strong chemistry: base={:?}, emergent={:?}, diff={diff:.3}",
            base_only.rgb, full_emergent.rgb,
        );
    }
}

/// Get a default stem color [u8; 3] for a plant taxonomy.
/// Used by the ribbon mesh builder when the visual response isn't available yet.
/// The actual stem color comes from the molecular optics pipeline at render time.
pub fn plant_stem_color_u8(taxonomy_id: u32) -> [u8; 3] {
    // Brown bark tones — vary slightly by species for visual diversity.
    match taxonomy_id {
        3750 => [82, 62, 38],  // Apple — dark brown bark
        2708 => [72, 56, 34],  // Citrus — medium brown
        15368 => [86, 78, 52], // Grass — light brown-green stem
        3702 => [68, 82, 42],  // Herb — green-brown
        _ => [78, 60, 36],     // Default brown
    }
}
