// Render and visual methods for TerrariumWorld.
//
// Extracted from terrarium_world.rs for modularity.

use super::*;

use crate::constants::clamp;
use crate::drosophila::{BodyState, DrosophilaSim};
use crate::molecular_atmosphere::{
    ATMOS_DENSITY_BASELINE_KG_M3, ATMOS_PRESSURE_BASELINE_KPA,
};
use crate::plant_cellular::{PlantClusterSnapshot, PlantTissue};
use crate::seed_cellular::{SeedClusterSnapshot, SeedTissue};
use crate::terrarium::TerrariumSpecies;
use crate::terrarium_render::*;
use crate::terrarium_render_pipeline::{
    build_dynamic_batch_renders, build_dynamic_render_delta, build_substrate_batch_renders,
    stamp_dynamic_render_fingerprints,
};
use crate::terrarium_scene_query::{
    TerrariumRaycastBvhNode, TerrariumRaycastSurface, TerrariumSceneRaycastHit,
};
use crate::LocalChemistrySiteReport;

use mesh::{render_cuboid_mesh, render_cylinder_mesh, render_ellipsoid_mesh};

impl super::TerrariumWorld {
pub fn render_scene_center_world(&self) -> [f32; 3] {
    render_cell_center_world(
        self.config.width,
        self.config.height,
        self.config.width as f32 * 0.5,
        self.config.height as f32 * 0.5,
    )
}

pub fn render_scene_half_extents_world(&self) -> [f32; 2] {
    [
        self.config.width as f32 * RENDER_CELL_WORLD_SIZE * 0.55,
        self.config.height as f32 * RENDER_CELL_WORLD_SIZE * 0.55,
    ]
}

pub fn render_lighting(&self) -> TerrariumLightingRender {
    let daylight = self.daylight().clamp(0.0, 1.0);
    let hours = self.time_of_day_hours();
    let humidity_mean = mean(&self.topdown_humidity());
    let pressure_mean = mean(&self.topdown_air_pressure_kpa());
    let temperature_mean = mean(&self.temperature);
    let co2_mean = mean(&self.topdown_atmospheric_co2());
    let o2_mean = mean(&self.topdown_atmospheric_o2());
    let humidity_t = humidity_mean.clamp(0.0, 1.0);
    let pressure_t = clamp(
        (pressure_mean - ATMOS_PRESSURE_BASELINE_KPA) / 4.0 * 0.5 + 0.5,
        0.0,
        1.0,
    );
    let temperature_t = ((temperature_mean - 8.0) / 28.0).clamp(0.0, 1.0);
    let co2_stress = clamp(
        (co2_mean - ATMOS_CO2_BASELINE) / ATMOS_CO2_BASELINE.max(1.0e-6),
        0.0,
        1.5,
    );
    let o2_depletion = clamp(
        (ATMOS_O2_BASELINE - o2_mean) / ATMOS_O2_BASELINE.max(1.0e-6),
        0.0,
        1.0,
    );
    let haze = clamp(
        humidity_t * 0.62
            + co2_stress * 0.14
            + o2_depletion * 0.12
            + (0.5 - pressure_t).abs() * 0.28,
        0.0,
        1.0,
    );
    let clear_day = lerp_rgb(rgb(0.68, 0.88, 0.98), rgb(0.52, 0.66, 0.78), haze);
    let clear_night = lerp_rgb(rgb(0.03, 0.05, 0.11), rgb(0.08, 0.08, 0.12), haze);
    let ambient_day = lerp_rgb(rgb(0.72, 0.78, 0.84), rgb(0.46, 0.50, 0.56), haze);
    let ambient_night = lerp_rgb(rgb(0.08, 0.10, 0.16), rgb(0.06, 0.07, 0.10), haze);
    let sun_day = lerp_rgb(rgb(1.0, 0.97, 0.88), rgb(0.86, 0.80, 0.72), haze);
    let sun_night = lerp_rgb(rgb(0.28, 0.34, 0.52), rgb(0.18, 0.22, 0.32), haze);
    let clear_color_rgb = lerp_rgb(clear_night, clear_day, daylight);
    let ambient_color_rgb = lerp_rgb(ambient_night, ambient_day, daylight);
    let sun_color_rgb = lerp_rgb(sun_night, sun_day, daylight);
    let [half_w, half_h] = self.render_scene_half_extents_world();
    let center = self.render_scene_center_world();
    let sun_angle = (hours / 24.0) * std::f32::consts::TAU;
    let sun_radius_x = half_w.max(1.0) * 1.75;
    let sun_radius_z = half_h.max(1.0) * 1.75;
    let sun_translation_world = [
        center[0] + sun_angle.cos() * sun_radius_x,
        center[1] + 1.8 + daylight * (half_w.max(half_h) * 2.1 + 6.0),
        center[2] + sun_angle.sin() * sun_radius_z,
    ];
    TerrariumLightingRender {
        clear_color_rgb,
        ambient_color_rgb,
        ambient_brightness: (0.10 + daylight * 0.40) * (1.0 - haze * 0.28),
        sun_color_rgb,
        sun_illuminance: (900.0 + daylight * 24_000.0) * (1.0 - haze * 0.18),
        sun_translation_world,
        sun_focus_world: center,
        humidity_t,
        pressure_t,
        temperature_t,
        daylight,
        time_phase: (self.time_s / 86_400.0).fract(),
    }
}

fn render_ground_material(
    &self,
    view: TerrariumTopdownView,
    field_norm: f32,
    terrain_norm: f32,
) -> TerrariumPbrMaterialRender {
    let daylight = self.daylight().clamp(0.0, 1.0);
    let (base_color, emissive_rgb, roughness) = match view {
        TerrariumTopdownView::Terrain => {
            let soil = lerp_rgb(rgb(0.18, 0.11, 0.07), rgb(0.48, 0.34, 0.18), terrain_norm);
            let moss = lerp_rgb(rgb(0.12, 0.19, 0.10), rgb(0.34, 0.56, 0.19), field_norm);
            (
                lerp_rgb(soil, moss, field_norm * 0.55),
                rgb(0.0, 0.0, 0.0),
                0.96,
            )
        }
        TerrariumTopdownView::SoilMoisture => (
            lerp_rgb(rgb(0.20, 0.12, 0.08), rgb(0.12, 0.48, 0.66), field_norm),
            rgb(0.0, field_norm * 0.01, field_norm * 0.02),
            0.78,
        ),
        TerrariumTopdownView::Canopy => (
            lerp_rgb(rgb(0.07, 0.14, 0.07), rgb(0.48, 0.74, 0.26), field_norm),
            rgb(field_norm * 0.01, field_norm * 0.03, 0.0),
            0.84,
        ),
        TerrariumTopdownView::Chemistry => (
            lerp_rgb(rgb(0.05, 0.07, 0.12), rgb(0.85, 0.56, 0.18), field_norm),
            rgb(field_norm * 0.04, field_norm * 0.02, 0.0),
            0.72,
        ),
        TerrariumTopdownView::Odor => (
            lerp_rgb(rgb(0.04, 0.04, 0.05), rgb(0.88, 0.36, 0.18), field_norm),
            rgb(field_norm * 0.08, field_norm * 0.02, 0.0),
            0.54,
        ),
        TerrariumTopdownView::GasExchange => (
            lerp_rgb(rgb(0.03, 0.07, 0.10), rgb(0.42, 0.88, 0.72), field_norm),
            rgb(0.0, field_norm * 0.05, field_norm * 0.03),
            0.58,
        ),
    };
    with_shader_response(
        pbr_material(
            rgba(base_color[0], base_color[1], base_color[2], 1.0),
            emissive_rgb,
            0.0,
            roughness,
            0.06,
            false,
            false,
        ),
        [field_norm, terrain_norm, 0.18 + field_norm * 0.42, daylight],
        [
            field_norm,
            terrain_norm,
            (field_norm - terrain_norm).abs(),
            view as u32 as f32 * 0.07,
        ],
        TERRARIUM_SHADER_FLAG_DEBUG_OVERLAY,
    )
}

fn render_water_material(&self, radius: f32) -> TerrariumPbrMaterialRender {
    let fullness = clamp((radius - 0.18) / 0.36, 0.0, 1.0);
    let base = lerp_rgb(rgb(0.07, 0.32, 0.62), rgb(0.14, 0.58, 0.92), fullness);
    with_shader_response(
        pbr_material(
            with_alpha(base, 0.58 + fullness * 0.18),
            rgb(0.01, 0.03 + fullness * 0.03, 0.06 + fullness * 0.06),
            0.02,
            0.18,
            0.18,
            true,
            false,
        ),
        [
            0.78 + fullness * 0.18,
            0.52 + fullness * 0.10,
            0.66 + fullness * 0.16,
            self.daylight().clamp(0.0, 1.0),
        ],
        [fullness, 0.42 + fullness * 0.24, 0.02, radius],
        TERRARIUM_SHADER_FLAG_FLUID,
    )
}

fn render_substrate_voxel_material(
    &self,
    hydro_signal: f32,
    carbon_signal: f32,
    oxic_signal: f32,
    nitrogen_signal: f32,
    biotic_signal: f32,
    energy_signal: f32,
    signal: f32,
) -> TerrariumPbrMaterialRender {
    let base = rgb(
        0.08 + carbon_signal * 0.44 + nitrogen_signal * 0.14,
        0.09 + oxic_signal * 0.34 + biotic_signal * 0.20,
        0.12 + hydro_signal * 0.44 + energy_signal * 0.16,
    );
    let emissive = rgb(
        carbon_signal * 0.02 + energy_signal * 0.06,
        oxic_signal * 0.03 + biotic_signal * 0.05,
        hydro_signal * 0.03 + energy_signal * 0.08,
    );
    with_shader_response(
        pbr_material(
            with_alpha(base, 0.18 + signal * 0.40),
            emissive,
            0.0,
            0.28,
            0.08,
            true,
            false,
        ),
        [
            hydro_signal,
            oxic_signal,
            energy_signal,
            self.daylight().clamp(0.0, 1.0),
        ],
        [biotic_signal, carbon_signal, nitrogen_signal, signal],
        TERRARIUM_SHADER_FLAG_SUBSTRATE,
    )
}

fn render_explicit_microbe_body_material(
    &self,
    activity: f32,
    energy_state: f32,
    stress_state: f32,
    oxygen_t: f32,
    translation_support_t: f32,
) -> TerrariumPbrMaterialRender {
    let base = rgb(
        0.18 + activity * 0.18 + stress_state * 0.12,
        0.24 + translation_support_t * 0.28 + energy_state * 0.08,
        0.22 + oxygen_t * 0.24 + energy_state * 0.10,
    );
    let emissive = rgb(
        0.03 + activity * 0.05,
        0.04 + translation_support_t * 0.08,
        0.05 + energy_state * 0.10,
    );
    with_shader_response(
        pbr_material(
            with_alpha(base, 0.42 + activity * 0.24),
            emissive,
            0.0,
            0.24,
            0.08,
            true,
            true,
        ),
        [
            oxygen_t,
            translation_support_t,
            0.36 + activity * 0.44,
            self.daylight().clamp(0.0, 1.0),
        ],
        [activity, energy_state, stress_state, translation_support_t],
        TERRARIUM_SHADER_FLAG_MICROBE,
    )
}

fn render_explicit_microbe_packet_material(
    &self,
    packet: &GenotypePacket,
) -> TerrariumPbrMaterialRender {
    let activity_t = packet.activity.clamp(0.0, 1.0);
    let dormancy_t = packet.dormancy.clamp(0.0, 1.0);
    let reserve_t = packet.reserve.clamp(0.0, 1.0);
    let damage_t = packet.damage.clamp(0.0, 1.0);
    let promotion_t = if packet.qualifies_for_promotion() {
        1.0
    } else {
        (activity_t * 0.45 + reserve_t * 0.35 - dormancy_t * 0.20 - damage_t * 0.25)
            .clamp(0.0, 1.0)
    };
    let base = rgb(
        0.56 + promotion_t * 0.24 - damage_t * 0.10,
        0.18 + activity_t * 0.24,
        0.24 + (1.0 - dormancy_t) * 0.26 + reserve_t * 0.08,
    );
    let emissive = rgb(
        0.08 + promotion_t * 0.12,
        0.02 + activity_t * 0.05,
        0.04 + reserve_t * 0.05 + (1.0 - dormancy_t) * 0.04,
    );
    with_shader_response(
        pbr_material(
            with_alpha(base, 0.52 + activity_t * 0.20),
            emissive,
            0.0,
            0.18,
            0.08,
            true,
            true,
        ),
        [
            reserve_t,
            1.0 - dormancy_t,
            promotion_t,
            self.daylight().clamp(0.0, 1.0),
        ],
        [activity_t, reserve_t, damage_t, promotion_t],
        TERRARIUM_SHADER_FLAG_MICROBE,
    )
}

fn render_explicit_microbe_packet_population_material(
    &self,
    mean_activity: f32,
    mean_dormancy: f32,
    promotion_t: f32,
    damage_t: f32,
) -> TerrariumPbrMaterialRender {
    let base = rgb(
        0.60 + promotion_t * 0.22 - damage_t * 0.08,
        0.18 + mean_activity * 0.24,
        0.26 + (1.0 - mean_dormancy) * 0.28,
    );
    let emissive = rgb(
        0.08 + promotion_t * 0.12,
        0.02 + mean_activity * 0.05,
        0.04 + (1.0 - mean_dormancy) * 0.06,
    );
    with_shader_response(
        pbr_material(
            with_alpha(base, 0.54 + mean_activity * 0.18),
            emissive,
            0.0,
            0.18,
            0.08,
            true,
            true,
        ),
        [
            1.0 - mean_dormancy,
            0.42 + promotion_t * 0.38,
            0.28 + mean_activity * 0.48,
            self.daylight().clamp(0.0, 1.0),
        ],
        [mean_activity, 1.0 - mean_dormancy, damage_t, promotion_t],
        TERRARIUM_SHADER_FLAG_MICROBE,
    )
}

fn render_explicit_microbe_site_material(
    &self,
    report: LocalChemistrySiteReport,
) -> TerrariumPbrMaterialRender {
    let occupancy_t = report.assembly_occupancy.clamp(0.0, 1.0);
    let stability_t = report.assembly_stability.clamp(0.0, 1.0);
    let atp_t = report.atp_support.clamp(0.0, 1.4) / 1.4;
    let translation_t = report.translation_support.clamp(0.0, 1.4) / 1.4;
    let oxygen_t = report.mean_oxygen.clamp(0.0, 1.2) / 1.2;
    let byproduct_t = report.byproduct_load.clamp(0.0, 1.0);
    let demand_t = report.demand_satisfaction.clamp(0.0, 1.0);
    let flux_t = report.mean_atp_flux.clamp(0.0, 0.2) / 0.2;
    let base = rgb(
        0.18 + occupancy_t * 0.18 + byproduct_t * 0.12,
        0.22 + translation_t * 0.24 + demand_t * 0.10,
        0.24 + oxygen_t * 0.20 + atp_t * 0.16,
    );
    let emissive = rgb(
        0.02 + flux_t * 0.10,
        0.03 + translation_t * 0.07,
        0.04 + atp_t * 0.08 + stability_t * 0.03,
    );
    with_shader_response(
        pbr_material(
            with_alpha(base, 0.46 + occupancy_t * 0.22),
            emissive,
            0.0,
            0.16 + (1.0 - stability_t) * 0.12,
            0.08,
            true,
            true,
        ),
        [
            oxygen_t,
            translation_t,
            flux_t,
            self.daylight().clamp(0.0, 1.0),
        ],
        [occupancy_t, atp_t, byproduct_t, stability_t],
        TERRARIUM_SHADER_FLAG_MICROBE,
    )
}

fn render_seed_part_material(
    &self,
    kind: TerrariumSeedPartKind,
    snapshot: &SeedClusterSnapshot,
    dormancy_t: f32,
) -> TerrariumPbrMaterialRender {
    let hydration_t = snapshot.hydration.min(1.2) / 1.2;
    let energy_t = snapshot.energy_charge.clamp(0.0, 1.0);
    let stress_t = snapshot.transcript_stress_response.clamp(0.0, 1.0);
    let germination_t = snapshot.transcript_germination_program.clamp(0.0, 1.0);
    let nitrogen_t = (snapshot.nitrogen_pool / (snapshot.nitrogen_pool + 0.4)).clamp(0.0, 1.0);
    let sugar_t = (snapshot.sugar_pool / (snapshot.sugar_pool + 0.6)).clamp(0.0, 1.0);
    let (base, emissive, roughness, alpha) = match kind {
        TerrariumSeedPartKind::Coat => (
            lerp_rgb(
                rgb(0.36, 0.24, 0.10),
                rgb(0.66, 0.54, 0.24),
                sugar_t * 0.35 + dormancy_t * 0.40 + hydration_t * 0.25,
            ),
            rgb(0.0, 0.0, 0.0),
            0.88 - hydration_t * 0.08,
            1.0,
        ),
        TerrariumSeedPartKind::Endosperm => (
            lerp_rgb(
                rgb(0.62, 0.56, 0.24),
                rgb(0.90, 0.84, 0.46),
                sugar_t * 0.74 + nitrogen_t * 0.26,
            ),
            rgb(0.01 + sugar_t * 0.03, 0.01 + germination_t * 0.02, 0.0),
            0.66 - sugar_t * 0.10,
            0.92,
        ),
        TerrariumSeedPartKind::Radicle
        | TerrariumSeedPartKind::CotyledonLeft
        | TerrariumSeedPartKind::CotyledonRight => (
            rgb(
                0.24 + hydration_t * 0.12,
                0.30 + nitrogen_t * 0.24,
                0.14 + energy_t * 0.12,
            ),
            rgb(0.01, 0.01 + germination_t * 0.04, 0.0),
            0.76 - hydration_t * 0.10 + stress_t * 0.06,
            0.94,
        ),
    };
    with_shader_response(
        pbr_material(
            rgba(base[0], base[1], base[2], alpha),
            emissive,
            0.02,
            roughness.clamp(0.18, 0.94),
            0.08,
            alpha < 1.0,
            false,
        ),
        [
            hydration_t,
            nitrogen_t,
            sugar_t,
            self.daylight().clamp(0.0, 1.0),
        ],
        [germination_t, energy_t, stress_t, dormancy_t],
        TERRARIUM_SHADER_FLAG_SEED,
    )
}

fn render_plant_stem_material(
    &self,
    health: f32,
    leaf_energy: f32,
) -> TerrariumPbrMaterialRender {
    let vigor = (health * 0.6 + leaf_energy * 0.4).clamp(0.0, 1.0);
    let base = lerp_rgb(rgb(0.24, 0.15, 0.09), rgb(0.40, 0.29, 0.15), vigor);
    with_shader_response(
        pbr_material(
            rgba(base[0], base[1], base[2], 1.0),
            rgb(0.0, 0.0, 0.0),
            0.01,
            0.92,
            0.06,
            false,
            false,
        ),
        [
            0.34 + vigor * 0.26,
            0.46 + health * 0.18,
            0.30 + leaf_energy * 0.22,
            self.daylight().clamp(0.0, 1.0),
        ],
        [vigor, health, 0.04, leaf_energy],
        TERRARIUM_SHADER_FLAG_PLANT,
    )
}

fn render_plant_canopy_material(
    &self,
    health: f32,
    leaf_energy: f32,
) -> TerrariumPbrMaterialRender {
    let base = rgb(
        0.16 + health * 0.22,
        0.28 + leaf_energy * 0.44,
        0.12 + health * 0.16,
    );
    with_shader_response(
        pbr_material(
            rgba(base[0], base[1], base[2], 1.0),
            rgb(0.0, leaf_energy * 0.04, 0.0),
            0.01,
            0.88,
            0.06,
            false,
            false,
        ),
        [
            0.48 + health * 0.22,
            0.44 + leaf_energy * 0.24,
            0.28 + leaf_energy * 0.18,
            self.daylight().clamp(0.0, 1.0),
        ],
        [health, leaf_energy, (1.0 - health).max(0.0), 0.0],
        TERRARIUM_SHADER_FLAG_PLANT,
    )
}

fn render_plant_tissue_material(
    &self,
    tissue: PlantTissue,
    snapshot: &PlantClusterSnapshot,
    plant_health: f32,
) -> TerrariumPbrMaterialRender {
    let vitality_t = snapshot.vitality.clamp(0.0, 1.0);
    let energy_t = plant_cluster_energy_charge(snapshot);
    let hydration_t = plant_cluster_hydration(snapshot);
    let division_t = snapshot.division_buffer.clamp(0.0, 1.0);
    let stress_t = snapshot.transcript_stress_response.clamp(0.0, 1.0);
    let transport_t = snapshot.transcript_transport_program.clamp(0.0, 1.0);
    let cell_cycle_t = snapshot.transcript_cell_cycle.clamp(0.0, 1.0);
    let (base, emissive, roughness) = match tissue {
        PlantTissue::Leaf => (
            rgb(
                0.16 + plant_health * 0.14 + stress_t * 0.06,
                0.26 + vitality_t * 0.34 + hydration_t * 0.10,
                0.12 + energy_t * 0.10,
            ),
            rgb(0.0, 0.02 + energy_t * 0.06, 0.0),
            0.84 - hydration_t * 0.08,
        ),
        PlantTissue::Stem => (
            rgb(
                0.22 + transport_t * 0.10,
                0.16 + plant_health * 0.10,
                0.10 + energy_t * 0.06,
            ),
            rgb(0.01, 0.01 + transport_t * 0.03, 0.0),
            0.90 - transport_t * 0.08,
        ),
        PlantTissue::Root => (
            rgb(
                0.20 + hydration_t * 0.08,
                0.16 + transport_t * 0.08,
                0.10 + vitality_t * 0.05,
            ),
            rgb(0.01 + transport_t * 0.02, 0.01, 0.0),
            0.92 - hydration_t * 0.06,
        ),
        PlantTissue::Meristem => (
            rgb(
                0.30 + division_t * 0.14,
                0.34 + cell_cycle_t * 0.26,
                0.16 + energy_t * 0.08,
            ),
            rgb(0.02 + division_t * 0.06, 0.04 + energy_t * 0.05, 0.01),
            0.76 - division_t * 0.08,
        ),
    };
    with_shader_response(
        pbr_material(
            rgba(base[0], base[1], base[2], 0.92),
            emissive,
            0.02,
            roughness.clamp(0.18, 0.96),
            0.08,
            false,
            false,
        ),
        [
            hydration_t,
            vitality_t,
            transport_t.max(energy_t),
            self.daylight().clamp(0.0, 1.0),
        ],
        [
            energy_t,
            division_t.max(cell_cycle_t),
            stress_t,
            plant_health,
        ],
        TERRARIUM_SHADER_FLAG_PLANT,
    )
}

fn render_fruit_part_material(
    &self,
    kind: TerrariumFruitPartKind,
    ripeness: f32,
    sugar_content: f32,
    odor_t: f32,
    microbial_t: f32,
    humidity_t: f32,
) -> TerrariumPbrMaterialRender {
    let (base, emissive, roughness, alpha, alpha_blend) = match kind {
        TerrariumFruitPartKind::Skin => (
            rgb(
                0.56 + ripeness * 0.34,
                0.18 + sugar_content * 0.22,
                0.10 + odor_t * 0.05,
            ),
            rgb(ripeness * 0.05, sugar_content * 0.03, 0.0),
            0.36 - humidity_t * 0.08,
            1.0,
            false,
        ),
        TerrariumFruitPartKind::Pulp => (
            rgb(
                0.74 + sugar_content * 0.18,
                0.28 + ripeness * 0.20,
                0.14 + odor_t * 0.05,
            ),
            rgb(0.03 + sugar_content * 0.05, 0.01 + odor_t * 0.02, 0.0),
            0.46 - sugar_content * 0.10,
            0.78,
            true,
        ),
        TerrariumFruitPartKind::Core => (
            rgb(
                0.34 + microbial_t * 0.14,
                0.22 + (1.0 - ripeness) * 0.10,
                0.10 + sugar_content * 0.05,
            ),
            rgb(0.01 + microbial_t * 0.03, 0.0, 0.0),
            0.74 - microbial_t * 0.08,
            0.94,
            true,
        ),
        TerrariumFruitPartKind::Stem => (
            rgb(
                0.24 + humidity_t * 0.08,
                0.32 + (1.0 - ripeness) * 0.14,
                0.12 + odor_t * 0.04,
            ),
            rgb(0.01, 0.01 + humidity_t * 0.02, 0.0),
            0.82,
            1.0,
            false,
        ),
    };
    with_shader_response(
        pbr_material(
            rgba(base[0], base[1], base[2], alpha),
            emissive,
            0.02,
            roughness.clamp(0.16, 0.94),
            0.08,
            alpha_blend,
            false,
        ),
        [
            humidity_t,
            odor_t,
            microbial_t,
            self.daylight().clamp(0.0, 1.0),
        ],
        [ripeness, sugar_content, microbial_t, odor_t],
        TERRARIUM_SHADER_FLAG_FRUIT,
    )
}

fn render_plume_material(
    &self,
    kind: TerrariumRenderPlumeKind,
    intensity: f32,
    humidity_t: f32,
    pressure_t: f32,
    density_t: f32,
    wind_t: f32,
) -> TerrariumPbrMaterialRender {
    let intensity = intensity.clamp(0.0, 1.0);
    let (base, emissive) = match kind {
        TerrariumRenderPlumeKind::Odor => (
            rgb(0.86 + intensity * 0.10, 0.34 + intensity * 0.10, 0.16),
            rgb(0.10 + intensity * 0.10, 0.03 + intensity * 0.03, 0.01),
        ),
        TerrariumRenderPlumeKind::GasExchange => (
            rgb(
                0.18 + intensity * 0.16,
                0.68 + intensity * 0.18,
                0.62 + intensity * 0.14,
            ),
            rgb(0.01, 0.06 + intensity * 0.08, 0.05 + intensity * 0.06),
        ),
        TerrariumRenderPlumeKind::Microbe => (
            rgb(
                0.70 + intensity * 0.16,
                0.16 + intensity * 0.06,
                0.28 + intensity * 0.10,
            ),
            rgb(0.06 + intensity * 0.06, 0.01, 0.02 + intensity * 0.03),
        ),
    };
    with_shader_response(
        pbr_material(
            with_alpha(base, 0.12 + intensity * 0.22),
            emissive,
            0.0,
            0.08,
            0.04,
            true,
            true,
        ),
        [
            humidity_t.clamp(0.0, 1.0),
            pressure_t.clamp(0.0, 1.0),
            density_t.clamp(0.0, 1.0),
            self.daylight().clamp(0.0, 1.0),
        ],
        [
            intensity,
            wind_t.clamp(0.0, 1.0),
            pressure_t.clamp(0.0, 1.0),
            kind as u32 as f32 * 0.11,
        ],
        TERRARIUM_SHADER_FLAG_PLUME,
    )
}

fn render_fly_body_material(&self, energy_t: f32, air_load: f32) -> TerrariumPbrMaterialRender {
    let base = rgb(
        0.08 + energy_t * 0.08,
        0.07 + energy_t * 0.06,
        0.06 + energy_t * 0.04 + air_load * 0.03,
    );
    with_shader_response(
        pbr_material(
            rgba(base[0], base[1], base[2], 1.0),
            rgb(energy_t * 0.02, energy_t * 0.015, 0.0),
            0.01,
            0.72,
            0.06,
            false,
            false,
        ),
        [
            0.24 + air_load * 0.32,
            0.30 + energy_t * 0.18,
            0.26 + air_load * 0.16,
            self.daylight().clamp(0.0, 1.0),
        ],
        [energy_t, air_load, (1.0 - energy_t).max(0.0) * 0.35, 0.0],
        TERRARIUM_SHADER_FLAG_FLY,
    )
}

fn render_fly_wing_material(&self, air_load: f32) -> TerrariumPbrMaterialRender {
    let alpha = 0.22 + air_load.clamp(0.0, 1.0) * 0.20;
    with_shader_response(
        pbr_material(
            rgba(0.90, 0.94, 1.0, alpha),
            rgb(0.03, 0.04, 0.05),
            0.0,
            0.22,
            0.12,
            true,
            true,
        ),
        [
            0.20 + air_load * 0.28,
            0.46 + air_load * 0.18,
            0.60 + air_load * 0.20,
            self.daylight().clamp(0.0, 1.0),
        ],
        [0.38 + air_load * 0.30, 0.22 + air_load * 0.18, 0.02, 0.17],
        TERRARIUM_SHADER_FLAG_FLY | TERRARIUM_SHADER_FLAG_FLUID,
    )
}

fn build_ground_tiles(&self, view: TerrariumTopdownView) -> Vec<TerrariumGroundTileRender> {
    let terrain = self.topdown_field(TerrariumTopdownView::Terrain);
    let field = self.topdown_field(view);
    let (terrain_min, terrain_inv) = field_min_inv(&terrain);
    let (field_min, field_inv) = field_min_inv(&field);
    let mut tiles = Vec::with_capacity(self.config.width * self.config.height);
    for idx in 0..self.config.width * self.config.height {
        let (x, y) = (idx % self.config.width, idx / self.config.width);
        let terrain_norm = normalize_unit(terrain[idx], terrain_min, terrain_inv);
        let field_norm = normalize_unit(field[idx], field_min, field_inv);
        let tile_height = RENDER_TILE_BASE_HEIGHT + terrain_norm * RENDER_TILE_HEIGHT_SCALE;
        let size_world = [
            RENDER_CELL_WORLD_SIZE * 0.94,
            tile_height,
            RENDER_CELL_WORLD_SIZE * 0.94,
        ];
        let mut translation =
            render_cell_center_world(self.config.width, self.config.height, x as f32, y as f32);
        translation[1] = tile_height * 0.5 - 0.02;
        tiles.push(TerrariumGroundTileRender {
            idx,
            primitive: TerrariumMeshPrimitive::Cube,
            mesh: render_cuboid_mesh(size_world),
            translation_world: translation,
            scale_world: size_world,
            terrain_norm,
            field_norm,
            material: self.render_ground_material(view, field_norm, terrain_norm),
        });
    }
    tiles
}

pub fn render_ground_tiles(
    &self,
    view: TerrariumTopdownView,
) -> Vec<TerrariumGroundTileRender> {
    self.build_ground_tiles(view)
}

pub fn render_ground_tiles_cached(
    &mut self,
    view: TerrariumTopdownView,
    revision: u64,
) -> &[TerrariumGroundTileRender] {
    let cache_index = view.cache_index();
    if self.cached_ground_tiles[cache_index].revision != revision {
        self.cached_ground_tiles[cache_index].tiles = self.build_ground_tiles(view);
        self.cached_ground_tiles[cache_index].revision = revision;
    }
    &self.cached_ground_tiles[cache_index].tiles
}

fn render_substrate_voxels(
    &self,
    terrain: &[f32],
    terrain_min: f32,
    terrain_inv: f32,
) -> Vec<TerrariumSubstrateVoxelRender> {
    let (x_dim, y_dim, z_dim) = self.substrate.shape();
    let total_voxels = self.substrate.total_voxels();
    if total_voxels == 0 || x_dim == 0 || y_dim == 0 || z_dim == 0 {
        return Vec::new();
    }

    let water = self.substrate.species_field(TerrariumSpecies::Water);
    let glucose = self.substrate.species_field(TerrariumSpecies::Glucose);
    let oxygen = self.substrate.species_field(TerrariumSpecies::OxygenGas);
    let ammonium = self.substrate.species_field(TerrariumSpecies::Ammonium);
    let nitrate = self.substrate.species_field(TerrariumSpecies::Nitrate);
    let carbon_dioxide = self
        .substrate
        .species_field(TerrariumSpecies::CarbonDioxide);
    let atp_flux = self.substrate.species_field(TerrariumSpecies::AtpFlux);
    let hydration = &self.substrate.hydration;
    let microbial = &self.substrate.microbial_activity;
    let nitrifier = &self.substrate.nitrifier_activity;
    let denitrifier = &self.substrate.denitrifier_activity;
    let plant_drive = &self.substrate.plant_drive;

    let (water_min, water_inv) = field_min_inv(water);
    let (glucose_min, glucose_inv) = field_min_inv(glucose);
    let (oxygen_min, oxygen_inv) = field_min_inv(oxygen);
    let (ammonium_min, ammonium_inv) = field_min_inv(ammonium);
    let (nitrate_min, nitrate_inv) = field_min_inv(nitrate);
    let (carbon_dioxide_min, carbon_dioxide_inv) = field_min_inv(carbon_dioxide);
    let (atp_flux_min, atp_flux_inv) = field_min_inv(atp_flux);
    let (hydration_min, hydration_inv) = field_min_inv(hydration);
    let (microbial_min, microbial_inv) = field_min_inv(microbial);
    let (nitrifier_min, nitrifier_inv) = field_min_inv(nitrifier);
    let (denitrifier_min, denitrifier_inv) = field_min_inv(denitrifier);
    let (plant_drive_min, plant_drive_inv) = field_min_inv(plant_drive);

    let plane = x_dim * y_dim;
    let voxel_size_world = render_substrate_voxel_world_size(
        self.substrate.voxel_size_mm,
        self.config.cell_size_mm,
    );
    let voxel_mesh = render_cuboid_mesh(voxel_size_world);
    let exposed_perimeter = (x_dim * 2 + y_dim * 2).saturating_sub(4);
    let estimated_visible =
        x_dim * y_dim + exposed_perimeter.saturating_mul(z_dim.saturating_sub(1));
    let mut renders = Vec::with_capacity(estimated_visible);
    for z in 0..z_dim {
        for y in 0..y_dim {
            for x in 0..x_dim {
                let exposed = z == 0 || x == 0 || y == 0 || x + 1 == x_dim || y + 1 == y_dim;
                if !exposed {
                    continue;
                }

                let idx = z * plane + y * x_dim + x;
                let world_x = ((x as f32 + 0.5) / x_dim as f32) * self.config.width as f32;
                let world_y = ((y as f32 + 0.5) / y_dim as f32) * self.config.height as f32;
                let cell_x =
                    world_x.floor().clamp(0.0, self.config.width as f32 - 1.0) as usize;
                let cell_y =
                    world_y.floor().clamp(0.0, self.config.height as f32 - 1.0) as usize;
                let terrain_idx = idx2(self.config.width, cell_x, cell_y);
                let ground_y = render_top_surface_y(normalize_unit(
                    terrain[terrain_idx],
                    terrain_min,
                    terrain_inv,
                ));

                let water_t = normalize_unit(water[idx], water_min, water_inv);
                let glucose_t = normalize_unit(glucose[idx], glucose_min, glucose_inv);
                let oxygen_t = normalize_unit(oxygen[idx], oxygen_min, oxygen_inv);
                let ammonium_t = normalize_unit(ammonium[idx], ammonium_min, ammonium_inv);
                let nitrate_t = normalize_unit(nitrate[idx], nitrate_min, nitrate_inv);
                let carbon_dioxide_t =
                    normalize_unit(carbon_dioxide[idx], carbon_dioxide_min, carbon_dioxide_inv);
                let atp_flux_t = normalize_unit(atp_flux[idx], atp_flux_min, atp_flux_inv);
                let hydration_t = normalize_unit(hydration[idx], hydration_min, hydration_inv);
                let microbial_t = normalize_unit(microbial[idx], microbial_min, microbial_inv);
                let nitrifier_t = normalize_unit(nitrifier[idx], nitrifier_min, nitrifier_inv);
                let denitrifier_t =
                    normalize_unit(denitrifier[idx], denitrifier_min, denitrifier_inv);
                let plant_drive_t =
                    normalize_unit(plant_drive[idx], plant_drive_min, plant_drive_inv);
                let components = [
                    water_t,
                    glucose_t,
                    oxygen_t,
                    ammonium_t,
                    nitrate_t,
                    carbon_dioxide_t,
                    atp_flux_t,
                    hydration_t,
                    microbial_t,
                    nitrifier_t,
                    denitrifier_t,
                    plant_drive_t,
                ];
                let signal = (components.iter().map(|value| value * value).sum::<f32>()
                    / components.len() as f32)
                    .sqrt();

                let hydro_signal = (water_t + hydration_t) * 0.5;
                let carbon_signal = (glucose_t + carbon_dioxide_t) * 0.5;
                let oxic_signal = (oxygen_t + nitrifier_t + plant_drive_t) / 3.0;
                let nitrogen_signal = (ammonium_t + nitrate_t + denitrifier_t) / 3.0;
                let biotic_signal =
                    (microbial_t + nitrifier_t + denitrifier_t + plant_drive_t) * 0.25;
                let translation = render_substrate_voxel_translation_world(
                    self.config.width,
                    self.config.height,
                    world_x,
                    world_y,
                    z,
                    ground_y,
                    voxel_size_world[1],
                );
                renders.push(TerrariumSubstrateVoxelRender {
                    render_id: terrarium_render_id(
                        TERRARIUM_RENDER_ID_SUBSTRATE_VOXEL,
                        terrarium_grid_render_primary(x, y, z),
                        0,
                    ),
                    render_fingerprint: 0,
                    voxel: [x, y, z],
                    primitive: TerrariumMeshPrimitive::Cube,
                    mesh: voxel_mesh.clone(),
                    translation_world: translation,
                    signal,
                    material: self.render_substrate_voxel_material(
                        hydro_signal,
                        carbon_signal,
                        oxic_signal,
                        nitrogen_signal,
                        biotic_signal,
                        atp_flux_t,
                        signal,
                    ),
                });
            }
        }
    }

    renders
}

fn render_explicit_microbes(
    &self,
    terrain: &[f32],
    terrain_min: f32,
    terrain_inv: f32,
) -> Vec<TerrariumExplicitMicrobeRender> {
    if self.explicit_microbes.is_empty() {
        return Vec::new();
    }

    let voxel_size_world = render_substrate_voxel_world_size(
        self.substrate.voxel_size_mm,
        self.config.cell_size_mm,
    );
    let mut renders = Vec::with_capacity(self.explicit_microbes.len());
    for (cohort_idx, cohort) in self.explicit_microbes.iter().enumerate() {
        let flat = idx2(self.config.width, cohort.x, cohort.y);
        let chemistry = cohort.last_snapshot.local_chemistry.unwrap_or_default();
        let microbe_render_id =
            terrarium_render_id(TERRARIUM_RENDER_ID_MICROBE, cohort_idx as u64, 0);
        let ground_y =
            render_top_surface_y(normalize_unit(terrain[flat], terrain_min, terrain_inv));
        let translation = render_substrate_voxel_translation_world(
            self.config.width,
            self.config.height,
            cohort.x as f32 + 0.5,
            cohort.y as f32 + 0.5,
            cohort.z,
            ground_y,
            voxel_size_world[1],
        );
        let activity = self.explicit_microbe_activity[flat].clamp(0.0, 1.25);
        let activity_t = (activity / 1.25).clamp(0.0, 1.0);
        let energy_state = (cohort.smoothed_energy / 1.2).clamp(0.0, 1.0);
        let stress_state = (cohort.smoothed_stress / 1.2).clamp(0.0, 1.0);
        let oxygen_t = (cohort.last_snapshot.oxygen_mm / 1.2).clamp(0.0, 1.0);
        let division_t = cohort.last_snapshot.division_progress.clamp(0.0, 1.0);
        let translation_support_t =
            ((chemistry.translation_support - 0.6) / 0.8).clamp(0.0, 1.0);
        let body_radius = (voxel_size_world[0]
            * (0.24
                + cohort.radius as f32 * 0.08
                + (cohort.represented_cells / EXPLICIT_MICROBE_COHORT_CELLS.max(1.0e-3))
                    .sqrt()
                    * 0.10
                + cohort.represented_packets * 0.04))
            .clamp(voxel_size_world[0] * 0.24, voxel_size_world[0] * 0.62);
        let site_reports = &cohort.last_snapshot.local_chemistry_sites;
        let site_frame = explicit_microbe_site_local_frame(site_reports, body_radius);
        let sites = if let Some(frame) = site_frame {
            site_reports
                .iter()
                .copied()
                .map(|report| {
                    let occupancy_t = report.assembly_occupancy.clamp(0.0, 1.0);
                    let stability_t = report.assembly_stability.clamp(0.0, 1.0);
                    let draw_t = (report.substrate_draw + report.energy_draw).clamp(0.0, 1.0);
                    let size_base = frame.pitch * (0.85 + report.patch_radius as f32 * 0.55);
                    let mesh = render_cuboid_mesh([
                        size_base * (0.78 + occupancy_t * 0.24),
                        size_base * (0.62 + stability_t * 0.28),
                        size_base * (0.78 + draw_t * 0.24),
                    ]);
                    TerrariumExplicitMicrobeSiteRender {
                        render_id: terrarium_render_id(
                            TERRARIUM_RENDER_ID_MICROBE_SITE,
                            cohort_idx as u64,
                            terrarium_site_render_slot(report.site),
                        ),
                        render_fingerprint: 0,
                        site: report.site,
                        translation_local: explicit_microbe_site_local_translation(
                            frame, report,
                        ),
                        mesh,
                        material: self.render_explicit_microbe_site_material(report),
                    }
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let body_mesh = if sites.is_empty() {
            render_stateful_microbe_body_mesh(
                body_radius,
                cohort.represented_cells,
                division_t,
                energy_state,
                stress_state,
                cohort.identity.gene_catabolic,
                cohort.identity.gene_extracellular_scavenging,
                translation_support_t,
            )
        } else {
            TerrariumTriangleMeshRender::default()
        };

        let matching_population = self
            .packet_populations
            .iter()
            .find(|pop| pop.x == cohort.x && pop.y == cohort.y && pop.z == cohort.z);
        let (
            packet_mesh,
            packets,
            packet_mean_activity,
            packet_mean_dormancy,
            promotion_t,
            damage_t,
        ) = if let Some(pop) = matching_population {
            let mean_damage = if pop.packets.is_empty() {
                0.0
            } else {
                pop.packets.iter().map(|packet| packet.damage).sum::<f32>()
                    / pop.packets.len() as f32
            };
            let packets = if let Some(frame) = site_frame {
                pop.packets
                    .iter()
                    .enumerate()
                    .map(
                        |(packet_idx, packet)| TerrariumExplicitMicrobePacketRender {
                            render_id: terrarium_render_id(
                                TERRARIUM_RENDER_ID_MICROBE_PACKET,
                                cohort_idx as u64,
                                (packet_idx.min(u16::MAX as usize)) as u16,
                            ),
                            render_fingerprint: 0,
                            translation_local: render_explicit_microbe_packet_local_translation(
                                frame,
                                site_reports,
                                packet,
                                packet_idx,
                            ),
                            mesh: render_explicit_microbe_packet_mesh(body_radius, packet),
                            material: self.render_explicit_microbe_packet_material(packet),
                        },
                    )
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            (
                if packets.is_empty() {
                    render_stateful_microbe_packet_mesh(body_radius, &pop.packets)
                } else {
                    TerrariumTriangleMeshRender::default()
                },
                packets,
                pop.mean_activity().clamp(0.0, 1.0),
                pop.mean_dormancy().clamp(0.0, 1.0),
                pop.promotion_candidates() as f32 / pop.packets.len().max(1) as f32,
                mean_damage.clamp(0.0, 1.0),
            )
        } else {
            (
                TerrariumTriangleMeshRender::default(),
                Vec::new(),
                0.0,
                0.0,
                0.0,
                0.0,
            )
        };

        renders.push(TerrariumExplicitMicrobeRender {
            render_id: microbe_render_id,
            body_render_fingerprint: 0,
            packet_population_render_fingerprint: 0,
            translation_world: translation,
            body_mesh,
            packet_mesh,
            packets,
            sites,
            represented_cells: cohort.represented_cells,
            represented_packets: cohort.represented_packets,
            activity,
            stress: cohort.smoothed_stress,
            body_material: self.render_explicit_microbe_body_material(
                activity_t,
                energy_state,
                stress_state,
                oxygen_t,
                translation_support_t,
            ),
            packet_material: self.render_explicit_microbe_packet_population_material(
                packet_mean_activity,
                packet_mean_dormancy,
                promotion_t,
                damage_t,
            ),
        });
    }

    renders
}

fn render_explicit_plumes(
    &self,
    terrain: &[f32],
    terrain_min: f32,
    terrain_inv: f32,
    explicit_microbes: &[TerrariumExplicitMicrobeRender],
) -> Vec<TerrariumPlumeRender> {
    let width = self.config.width;
    let height = self.config.height;
    let depth = self.config.depth.max(1);
    let total = width * height * depth;
    let Some(odor) = self.odorants.get(ETHYL_ACETATE_IDX) else {
        return Vec::new();
    };
    let Some(co2) = self.odorants.get(ATMOS_CO2_IDX) else {
        return Vec::new();
    };
    let Some(o2) = self.odorants.get(ATMOS_O2_IDX) else {
        return Vec::new();
    };
    if odor.len() != total
        || co2.len() != total
        || o2.len() != total
        || self.humidity.len() != total
        || self.air_density.len() != total
        || self.air_pressure_kpa.len() != total
        || self.wind_x.len() != total
        || self.wind_y.len() != total
        || self.wind_z.len() != total
    {
        return Vec::new();
    }

    let pressure_mean = mean(&self.air_pressure_kpa);
    let (pressure_min, pressure_inv) = field_min_inv(&self.air_pressure_kpa);
    let (density_min, density_inv) = field_min_inv(&self.air_density);
    let gas_signal = co2
        .iter()
        .zip(o2.iter())
        .zip(self.humidity.iter())
        .zip(self.air_density.iter())
        .map(|(((co2, o2), humidity), density)| {
            let co2_dev = ((*co2 - ATMOS_CO2_BASELINE) / ATMOS_CO2_BASELINE.max(1.0e-6)).abs();
            let o2_dev = ((*o2 - ATMOS_O2_BASELINE) / ATMOS_O2_BASELINE.max(1.0e-6)).abs();
            let humidity_dev = (*humidity - 0.40).abs();
            let density_dev = ((*density - ATMOS_DENSITY_BASELINE_KG_M3)
                / ATMOS_DENSITY_BASELINE_KG_M3.max(1.0e-6))
            .abs();
            (co2_dev * 0.44 + o2_dev * 0.38 + humidity_dev * 0.14 + density_dev * 0.24)
                .clamp(0.0, 2.0)
        })
        .collect::<Vec<_>>();

    let mut plumes =
        Vec::with_capacity(RENDER_OVERLAY_MAX_PUFFS * 2 + explicit_microbes.len().min(16));
    for (idx, norm) in
        strongest_render_cells(odor, RENDER_OVERLAY_MAX_PUFFS, RENDER_OVERLAY_MIN_NORM)
    {
        let (x, y, z) = decode_idx3(width, height, idx);
        let ground_idx = idx2(width, x, y);
        let ground_y = render_top_surface_y(normalize_unit(
            terrain[ground_idx],
            terrain_min,
            terrain_inv,
        ));
        let base_translation = render_atmosphere_cell_translation_world(
            width,
            height,
            x as f32 + 0.5,
            y as f32 + 0.5,
            z,
            ground_y,
        );
        let (pressure_activity, pressure_bias) = render_pressure_response(
            idx,
            &self.air_pressure_kpa,
            pressure_mean,
            pressure_min,
            pressure_inv,
        );
        let humidity_t = self.humidity[idx].clamp(0.0, 1.0);
        let density_t = normalize_unit(self.air_density[idx], density_min, density_inv);
        let drift_x = clamp(self.wind_x[idx] * 0.12, -0.18, 0.18);
        let drift_z = clamp(self.wind_y[idx] * 0.12, -0.18, 0.18);
        let wind_t = clamp(
            (self.wind_x[idx].abs() + self.wind_y[idx].abs() + self.wind_z[idx].abs()) * 0.16,
            0.0,
            1.0,
        );
        let lift = clamp(
            self.wind_z[idx] * 0.18 - pressure_bias * 0.08 + pressure_activity * 0.05,
            -0.04,
            0.18,
        );
        let scale = [
            0.08 + norm * 0.16 + humidity_t * 0.04,
            0.12 + norm * 0.30 + density_t * 0.05,
            0.08 + norm * 0.16 + humidity_t * 0.04,
        ];
        plumes.push(TerrariumPlumeRender {
            render_id: terrarium_render_id(TERRARIUM_RENDER_ID_PLUME, idx as u64, 1),
            render_fingerprint: 0,
            kind: TerrariumRenderPlumeKind::Odor,
            primitive: TerrariumMeshPrimitive::UvSphere {
                sectors: 16,
                stacks: 10,
            },
            mesh: render_stateful_plume_mesh(
                scale,
                norm,
                [self.wind_x[idx], self.wind_z[idx], self.wind_y[idx]],
                pressure_bias,
                humidity_t,
            ),
            translation_world: [
                base_translation[0] + drift_x,
                base_translation[1] + lift,
                base_translation[2] + drift_z,
            ],
            rotation_xyz_rad: [
                clamp(-self.wind_y[idx] * 0.22, -0.45, 0.45),
                0.0,
                clamp(self.wind_x[idx] * 0.22, -0.45, 0.45),
            ],
            scale_world: scale,
            intensity: norm,
            material: self.render_plume_material(
                TerrariumRenderPlumeKind::Odor,
                norm,
                humidity_t,
                pressure_activity,
                density_t,
                wind_t,
            ),
        });
    }

    for (idx, norm) in strongest_render_cells(
        &gas_signal,
        RENDER_OVERLAY_MAX_PUFFS,
        RENDER_OVERLAY_MIN_NORM,
    ) {
        let (x, y, z) = decode_idx3(width, height, idx);
        let ground_idx = idx2(width, x, y);
        let ground_y = render_top_surface_y(normalize_unit(
            terrain[ground_idx],
            terrain_min,
            terrain_inv,
        ));
        let base_translation = render_atmosphere_cell_translation_world(
            width,
            height,
            x as f32 + 0.5,
            y as f32 + 0.5,
            z,
            ground_y,
        );
        let (pressure_activity, pressure_bias) = render_pressure_response(
            idx,
            &self.air_pressure_kpa,
            pressure_mean,
            pressure_min,
            pressure_inv,
        );
        let humidity_t = self.humidity[idx].clamp(0.0, 1.0);
        let density_t = normalize_unit(self.air_density[idx], density_min, density_inv);
        let drift_x = clamp(self.wind_x[idx] * 0.14, -0.22, 0.22);
        let drift_z = clamp(self.wind_y[idx] * 0.14, -0.22, 0.22);
        let wind_t = clamp(
            (self.wind_x[idx].abs() + self.wind_y[idx].abs() + self.wind_z[idx].abs()) * 0.16,
            0.0,
            1.0,
        );
        let lift = clamp(
            self.wind_z[idx] * 0.22 - pressure_bias * 0.11 + pressure_activity * 0.06,
            -0.02,
            0.24,
        );
        let scale = [
            0.10 + norm * 0.18 + humidity_t * 0.03,
            0.16 + norm * 0.34 + density_t * 0.06,
            0.10 + norm * 0.18 + humidity_t * 0.03,
        ];
        plumes.push(TerrariumPlumeRender {
            render_id: terrarium_render_id(TERRARIUM_RENDER_ID_PLUME, idx as u64, 2),
            render_fingerprint: 0,
            kind: TerrariumRenderPlumeKind::GasExchange,
            primitive: TerrariumMeshPrimitive::UvSphere {
                sectors: 16,
                stacks: 10,
            },
            mesh: render_stateful_plume_mesh(
                scale,
                norm,
                [self.wind_x[idx], self.wind_z[idx], self.wind_y[idx]],
                pressure_bias,
                humidity_t,
            ),
            translation_world: [
                base_translation[0] + drift_x,
                base_translation[1] + lift,
                base_translation[2] + drift_z,
            ],
            rotation_xyz_rad: [
                clamp(-self.wind_y[idx] * 0.26, -0.48, 0.48),
                0.0,
                clamp(self.wind_x[idx] * 0.26, -0.48, 0.48),
            ],
            scale_world: scale,
            intensity: norm,
            material: self.render_plume_material(
                TerrariumRenderPlumeKind::GasExchange,
                norm,
                humidity_t,
                pressure_activity,
                density_t,
                wind_t,
            ),
        });
    }

    for (source_idx, (source, render)) in self
        .explicit_microbes
        .iter()
        .zip(explicit_microbes.iter())
        .enumerate()
    {
        let represented_cells_t = (source.represented_cells
            / EXPLICIT_MICROBE_COHORT_CELLS.max(1.0e-3))
        .clamp(0.0, 1.5);
        let activity_t = (render.activity.max(0.0) / 1.25 * 0.65
            + represented_cells_t * 0.25
            + render.stress.clamp(0.0, 1.0) * 0.10)
            .clamp(0.0, 1.0);
        if activity_t < 0.03 {
            continue;
        }
        let air_idx = idx3(width, height, source.x, source.y, source.z.min(depth - 1));
        let (pressure_activity, pressure_bias) = render_pressure_response(
            air_idx,
            &self.air_pressure_kpa,
            pressure_mean,
            pressure_min,
            pressure_inv,
        );
        let stress_t = render.stress.clamp(0.0, 1.0);
        let density_t = normalize_unit(self.air_density[air_idx], density_min, density_inv);
        let drift_x = clamp(self.wind_x[air_idx] * 0.08, -0.10, 0.10);
        let drift_z = clamp(self.wind_y[air_idx] * 0.08, -0.10, 0.10);
        let humidity_t = self.humidity[air_idx].clamp(0.0, 1.0);
        let wind_t = clamp(
            (self.wind_x[air_idx].abs()
                + self.wind_y[air_idx].abs()
                + self.wind_z[air_idx].abs())
                * 0.16,
            0.0,
            1.0,
        );
        let lift = clamp(
            self.wind_z[air_idx] * 0.08 - pressure_bias * 0.04 + pressure_activity * 0.02,
            0.0,
            0.10,
        );
        let scale = [
            0.06 + activity_t * 0.10 + density_t * 0.02,
            0.09 + activity_t * 0.18 + stress_t * 0.05,
            0.06 + activity_t * 0.10 + density_t * 0.02,
        ];
        plumes.push(TerrariumPlumeRender {
            render_id: terrarium_render_id(TERRARIUM_RENDER_ID_PLUME, source_idx as u64, 3),
            render_fingerprint: 0,
            kind: TerrariumRenderPlumeKind::Microbe,
            primitive: TerrariumMeshPrimitive::UvSphere {
                sectors: 16,
                stacks: 10,
            },
            mesh: render_stateful_plume_mesh(
                scale,
                activity_t,
                [
                    self.wind_x[air_idx],
                    self.wind_z[air_idx],
                    self.wind_y[air_idx],
                ],
                pressure_bias,
                humidity_t,
            ),
            translation_world: [
                render.translation_world[0] + drift_x,
                render.translation_world[1] + scale[1] * 0.65 + lift,
                render.translation_world[2] + drift_z,
            ],
            rotation_xyz_rad: [
                clamp(-self.wind_y[air_idx] * 0.18, -0.30, 0.30),
                0.0,
                clamp(self.wind_x[air_idx] * 0.18, -0.30, 0.30),
            ],
            scale_world: scale,
            intensity: activity_t,
            material: self.render_plume_material(
                TerrariumRenderPlumeKind::Microbe,
                activity_t,
                humidity_t,
                pressure_activity,
                density_t,
                wind_t,
            ),
        });
    }

    plumes
}

pub(super) fn build_dynamic_snapshot(&self) -> TerrariumDynamicRenderSnapshot {
    let terrain = self.topdown_field(TerrariumTopdownView::Terrain);
    let (terrain_min, terrain_inv) = field_min_inv(&terrain);
    let mut snapshot = TerrariumDynamicRenderSnapshot::default();
    snapshot.substrate_voxels =
        self.render_substrate_voxels(&terrain, terrain_min, terrain_inv);
    snapshot.explicit_microbes =
        self.render_explicit_microbes(&terrain, terrain_min, terrain_inv);
    let depth = self.config.depth.max(1);

    for (water_idx, water) in self.waters.iter().enumerate() {
        if !water.alive {
            continue;
        }
        let idx = idx2(self.config.width, water.x, water.y);
        let ground_y =
            render_top_surface_y(normalize_unit(terrain[idx], terrain_min, terrain_inv));
        let mut translation = render_cell_center_world(
            self.config.width,
            self.config.height,
            water.x as f32 + 0.5,
            water.y as f32 + 0.5,
        );
        let radius = (0.18 + water.volume * 0.0012).clamp(0.18, 0.54);
        let scale_world = [radius, 0.06 + radius * 0.22, radius];
        let air_idx = idx3(
            self.config.width,
            self.config.height,
            water.x,
            water.y,
            water.z.min(depth - 1),
        );
        let humidity_t = self.humidity[air_idx].clamp(0.0, 1.0);
        let pressure_t = clamp(
            (self.air_pressure_kpa[air_idx] - ATMOS_PRESSURE_BASELINE_KPA) / 4.0 * 0.5 + 0.5,
            0.0,
            1.0,
        );
        let volume_t = (water.volume / 180.0).clamp(0.0, 1.0);
        translation[1] = ground_y + 0.06 + water.z as f32 * 0.04;
        snapshot.waters.push(TerrariumWaterRender {
            render_id: terrarium_render_id(TERRARIUM_RENDER_ID_WATER, water_idx as u64, 0),
            render_fingerprint: 0,
            primitive: TerrariumMeshPrimitive::UvSphere {
                sectors: 24,
                stacks: 16,
            },
            mesh: render_stateful_water_mesh(
                scale_world,
                [
                    self.wind_x[air_idx] * 0.04,
                    self.wind_z[air_idx] * 0.04,
                    self.wind_y[air_idx] * 0.04,
                ],
                pressure_t,
                humidity_t,
                volume_t,
            ),
            translation_world: translation,
            scale_world,
            material: self.render_water_material(radius),
        });
    }

    for (plant_idx, plant) in self.plants.iter().enumerate() {
        let idx = idx2(self.config.width, plant.x, plant.y);
        let plant_render_id =
            terrarium_render_id(TERRARIUM_RENDER_ID_PLANT, plant_idx as u64, 0);
        let ground_y =
            render_top_surface_y(normalize_unit(terrain[idx], terrain_min, terrain_inv));
        let translation = render_cell_center_world(
            self.config.width,
            self.config.height,
            plant.x as f32 + 0.5,
            plant.y as f32 + 0.5,
        );
        let leaf_cluster = plant.cellular.cluster_snapshot(PlantTissue::Leaf);
        let stem_cluster = plant.cellular.cluster_snapshot(PlantTissue::Stem);
        let root_cluster = plant.cellular.cluster_snapshot(PlantTissue::Root);
        let meristem_cluster = plant.cellular.cluster_snapshot(PlantTissue::Meristem);
        let height = (plant.physiology.height_mm() * 0.055).clamp(0.22, 1.55);
        let canopy_radius = (plant.canopy_radius_cells() as f32 * 0.16).clamp(0.22, 1.25);
        let health = (plant.physiology.health() * 0.55
            + leaf_cluster.vitality.clamp(0.0, 1.0) * 0.45)
            .clamp(0.0, 1.0);
        let leaf_energy = plant_cluster_energy_charge(&leaf_cluster);
        let stem_energy = plant_cluster_energy_charge(&stem_cluster);
        let pose_scale = height / plant.physiology.height_mm().max(1.0);
        let canopy_offset_world = [
            plant.pose.canopy_offset_mm[0] * pose_scale,
            plant.pose.canopy_offset_mm[1] * pose_scale,
            plant.pose.canopy_offset_mm[2] * pose_scale,
        ];
        let canopy_velocity_world = [
            plant.pose.canopy_velocity_mm_s[0] * pose_scale * 0.12,
            plant.pose.canopy_velocity_mm_s[1] * pose_scale * 0.12,
            plant.pose.canopy_velocity_mm_s[2] * pose_scale * 0.12,
        ];
        let stem_radius = (0.06
            + (stem_cluster.cell_count / 140.0).sqrt() * 0.02
            + stem_cluster.transcript_transport_program.clamp(0.0, 1.0) * 0.03)
            .clamp(0.06, 0.16);
        let stem_scale_world = [stem_radius, height, stem_radius];
        let canopy_scale_world = [canopy_radius, 0.55, canopy_radius];
        let root_radius_world = (plant.root_radius_cells() as f32 * 0.14).clamp(0.20, 1.10);
        let root_depth_world = (0.22
            + (root_cluster.cell_count / 180.0).sqrt() * 0.16
            + plant_cluster_hydration(&root_cluster) * 0.18)
            .clamp(0.22, 0.96);
        let canopy_translation_world = [
            translation[0] + canopy_offset_world[0],
            ground_y + height + canopy_offset_world[1],
            translation[2] + canopy_offset_world[2],
        ];
        let stem_translation_world = [translation[0], ground_y + height * 0.5, translation[2]];
        let stem_rotation_xyz_rad = [
            plant.pose.stem_tilt_x_rad * 0.35,
            0.0,
            plant.pose.stem_tilt_z_rad * 0.35,
        ];
        let canopy_rotation_xyz_rad = [
            plant.pose.stem_tilt_x_rad * 0.30 + canopy_velocity_world[2] * 0.08,
            0.0,
            plant.pose.stem_tilt_z_rad * 0.30 - canopy_velocity_world[0] * 0.08,
        ];
        let tissues = vec![
            TerrariumPlantTissueRender {
                render_id: terrarium_render_id(
                    TERRARIUM_RENDER_ID_PLANT,
                    plant_idx as u64,
                    terrarium_tissue_render_slot(PlantTissue::Leaf),
                ),
                render_fingerprint: 0,
                tissue: PlantTissue::Leaf,
                mesh: render_stateful_plant_tissue_mesh(
                    PlantTissue::Leaf,
                    [canopy_radius * 0.70, 0.24, canopy_radius * 0.70],
                    &leaf_cluster,
                ),
                translation_world: canopy_translation_world,
                rotation_xyz_rad: canopy_rotation_xyz_rad,
                material: self.render_plant_tissue_material(
                    PlantTissue::Leaf,
                    &leaf_cluster,
                    health,
                ),
            },
            TerrariumPlantTissueRender {
                render_id: terrarium_render_id(
                    TERRARIUM_RENDER_ID_PLANT,
                    plant_idx as u64,
                    terrarium_tissue_render_slot(PlantTissue::Stem),
                ),
                render_fingerprint: 0,
                tissue: PlantTissue::Stem,
                mesh: render_stateful_plant_tissue_mesh(
                    PlantTissue::Stem,
                    [stem_radius * 1.18, height * 0.84, stem_radius * 1.18],
                    &stem_cluster,
                ),
                translation_world: stem_translation_world,
                rotation_xyz_rad: stem_rotation_xyz_rad,
                material: self.render_plant_tissue_material(
                    PlantTissue::Stem,
                    &stem_cluster,
                    health,
                ),
            },
            TerrariumPlantTissueRender {
                render_id: terrarium_render_id(
                    TERRARIUM_RENDER_ID_PLANT,
                    plant_idx as u64,
                    terrarium_tissue_render_slot(PlantTissue::Root),
                ),
                render_fingerprint: 0,
                tissue: PlantTissue::Root,
                mesh: render_stateful_plant_tissue_mesh(
                    PlantTissue::Root,
                    [root_radius_world, root_depth_world, root_radius_world],
                    &root_cluster,
                ),
                translation_world: [
                    translation[0],
                    ground_y - root_depth_world * 0.42,
                    translation[2],
                ],
                rotation_xyz_rad: [
                    -plant.pose.stem_tilt_x_rad * 0.18,
                    0.0,
                    -plant.pose.stem_tilt_z_rad * 0.18,
                ],
                material: self.render_plant_tissue_material(
                    PlantTissue::Root,
                    &root_cluster,
                    health,
                ),
            },
            TerrariumPlantTissueRender {
                render_id: terrarium_render_id(
                    TERRARIUM_RENDER_ID_PLANT,
                    plant_idx as u64,
                    terrarium_tissue_render_slot(PlantTissue::Meristem),
                ),
                render_fingerprint: 0,
                tissue: PlantTissue::Meristem,
                mesh: render_stateful_plant_tissue_mesh(
                    PlantTissue::Meristem,
                    [
                        0.10 + meristem_cluster.division_buffer.clamp(0.0, 1.0) * 0.08,
                        0.09 + meristem_cluster.transcript_cell_cycle.clamp(0.0, 1.0) * 0.08,
                        0.10 + meristem_cluster.vitality.clamp(0.0, 1.0) * 0.06,
                    ],
                    &meristem_cluster,
                ),
                translation_world: [
                    translation[0] + canopy_offset_world[0] * 0.28,
                    ground_y + height * 0.76 + canopy_offset_world[1] * 0.30,
                    translation[2] + canopy_offset_world[2] * 0.28,
                ],
                rotation_xyz_rad: canopy_rotation_xyz_rad,
                material: self.render_plant_tissue_material(
                    PlantTissue::Meristem,
                    &meristem_cluster,
                    health,
                ),
            },
        ];
        let stem_mesh = if tissues.is_empty() {
            render_stateful_stem_mesh(
                stem_scale_world[1],
                stem_scale_world[0] * 0.5,
                canopy_offset_world,
                &stem_cluster,
            )
        } else {
            TerrariumTriangleMeshRender::default()
        };
        let canopy_mesh = if tissues.is_empty() {
            render_stateful_canopy_mesh(
                canopy_scale_world,
                &plant.genome,
                health,
                leaf_energy,
                canopy_velocity_world,
                &leaf_cluster,
            )
        } else {
            TerrariumTriangleMeshRender::default()
        };
        snapshot.plants.push(TerrariumPlantRender {
            render_id: plant_render_id,
            stem_render_fingerprint: 0,
            canopy_render_fingerprint: 0,
            stem_primitive: TerrariumMeshPrimitive::Cylinder { resolution: 18 },
            stem_mesh,
            stem_translation_world,
            stem_rotation_xyz_rad,
            stem_scale_world,
            canopy_primitive: TerrariumMeshPrimitive::UvSphere {
                sectors: 18,
                stacks: 12,
            },
            canopy_mesh,
            canopy_translation_world,
            canopy_rotation_xyz_rad,
            canopy_scale_world,
            health,
            leaf_energy,
            tissues,
            stem_material: self.render_plant_stem_material(health, stem_energy),
            canopy_material: self.render_plant_canopy_material(health, leaf_energy),
        });
    }

    for (seed_idx, seed) in self.seeds.iter().enumerate() {
        let sx = seed.x.clamp(0.0, self.config.width as f32 - 0.01);
        let sy = seed.y.clamp(0.0, self.config.height as f32 - 0.01);
        let seed_render_id = terrarium_render_id(TERRARIUM_RENDER_ID_SEED, seed_idx as u64, 0);
        let cell_x = sx.floor() as usize;
        let cell_y = sy.floor() as usize;
        let idx = idx2(self.config.width, cell_x, cell_y);
        let ground_y =
            render_top_surface_y(normalize_unit(terrain[idx], terrain_min, terrain_inv));
        let mut translation =
            render_cell_center_world(self.config.width, self.config.height, sx, sy);
        let coat_cluster = seed.cellular.cluster_snapshot(SeedTissue::Coat);
        let endosperm_cluster = seed.cellular.cluster_snapshot(SeedTissue::Endosperm);
        let radicle_cluster = seed.cellular.cluster_snapshot(SeedTissue::Radicle);
        let cotyledon_cluster = seed.cellular.cluster_snapshot(SeedTissue::Cotyledon);
        let total_cells = coat_cluster.cell_count
            + endosperm_cluster.cell_count
            + radicle_cluster.cell_count
            + cotyledon_cluster.cell_count;
        let hydration_t = seed.cellular.hydration().min(1.2) / 1.2;
        let vitality_t = seed.cellular.vitality().clamp(0.0, 1.0);
        let energy_t = seed.cellular.energy_charge().clamp(0.0, 1.0);
        let uniform_scale_world = (0.028
            + total_cells.sqrt() * 0.0045
            + hydration_t * 0.010
            + vitality_t * 0.006
            + energy_t * 0.004)
            .clamp(0.040, 0.12);
        let pose_scale = uniform_scale_world / self.config.cell_size_mm.max(1.0e-3);
        let seed_offset_world = [
            seed.pose.offset_mm[0] * pose_scale,
            seed.pose.offset_mm[1] * pose_scale,
            seed.pose.offset_mm[2] * pose_scale,
        ];
        translation[0] += seed_offset_world[0];
        translation[1] = ground_y + 0.04 + seed_offset_world[1];
        translation[2] += seed_offset_world[2];
        let dormancy_t = clamp(seed.dormancy_s / 26_000.0, 0.0, 1.0);
        let coat_material = self.render_seed_part_material(
            TerrariumSeedPartKind::Coat,
            &coat_cluster,
            dormancy_t,
        );
        let endosperm_material = self.render_seed_part_material(
            TerrariumSeedPartKind::Endosperm,
            &endosperm_cluster,
            dormancy_t,
        );
        let radicle_material = self.render_seed_part_material(
            TerrariumSeedPartKind::Radicle,
            &radicle_cluster,
            dormancy_t,
        );
        let cotyledon_material = self.render_seed_part_material(
            TerrariumSeedPartKind::CotyledonLeft,
            &cotyledon_cluster,
            dormancy_t,
        );
        let parts = render_stateful_seed_parts(
            seed_idx as u64,
            seed,
            uniform_scale_world,
            &coat_cluster,
            &endosperm_cluster,
            &radicle_cluster,
            &cotyledon_cluster,
            coat_material,
            endosperm_material,
            radicle_material,
            cotyledon_material,
        );
        snapshot.seeds.push(TerrariumSeedRender {
            render_id: seed_render_id,
            translation_world: translation,
            rotation_xyz_rad: seed.pose.rotation_xyz_rad,
            parts,
        });
    }

    for (fruit_idx, fruit) in self.fruits.iter().enumerate() {
        if !fruit.source.alive || fruit.source.sugar_content <= 0.01 {
            continue;
        }
        let fruit_render_id =
            terrarium_render_id(TERRARIUM_RENDER_ID_FRUIT, fruit_idx as u64, 0);
        let idx = idx2(self.config.width, fruit.source.x, fruit.source.y);
        let ground_y =
            render_top_surface_y(normalize_unit(terrain[idx], terrain_min, terrain_inv));
        let translation = render_cell_center_world(
            self.config.width,
            self.config.height,
            fruit.source.x as f32 + 0.5,
            fruit.source.y as f32 + 0.5,
        );
        let radius = (0.12 + fruit.radius * 0.10).clamp(0.12, 0.34);
        let pose_scale =
            radius / (fruit.radius * self.config.cell_size_mm.max(1.0e-3)).max(1.0e-3);
        let fruit_offset_world = [
            fruit.pose.offset_mm[0] * pose_scale,
            fruit.pose.offset_mm[1] * pose_scale,
            fruit.pose.offset_mm[2] * pose_scale,
        ];
        let fruit_velocity_world = [
            fruit.pose.velocity_mm_s[0] * pose_scale * 0.10,
            fruit.pose.velocity_mm_s[1] * pose_scale * 0.10,
            fruit.pose.velocity_mm_s[2] * pose_scale * 0.10,
        ];
        let odor_t = clamp(fruit.source.odorant_emission_rate / 0.18, 0.0, 1.0);
        let microbial_t = clamp(
            (self.microbial_biomass[idx]
                + self.nitrifier_biomass[idx] * 0.55
                + self.denitrifier_biomass[idx] * 0.70
                + self.explicit_microbe_activity[idx] * 0.45)
                / 1.6,
            0.0,
            1.0,
        );
        let humidity_t = clamp(
            self.sample_humidity_at(
                fruit.source.x,
                fruit.source.y,
                fruit.source.z.min(self.config.depth.max(1) - 1),
            ),
            0.0,
            1.0,
        );
        let ripeness = fruit.source.ripeness.clamp(0.0, 1.0);
        let sugar_content = fruit.source.sugar_content.clamp(0.0, 1.0);
        let skin_material = self.render_fruit_part_material(
            TerrariumFruitPartKind::Skin,
            ripeness,
            sugar_content,
            odor_t,
            microbial_t,
            humidity_t,
        );
        let pulp_material = self.render_fruit_part_material(
            TerrariumFruitPartKind::Pulp,
            ripeness,
            sugar_content,
            odor_t,
            microbial_t,
            humidity_t,
        );
        let core_material = self.render_fruit_part_material(
            TerrariumFruitPartKind::Core,
            ripeness,
            sugar_content,
            odor_t,
            microbial_t,
            humidity_t,
        );
        let stem_material = self.render_fruit_part_material(
            TerrariumFruitPartKind::Stem,
            ripeness,
            sugar_content,
            odor_t,
            microbial_t,
            humidity_t,
        );
        let parts = render_stateful_fruit_parts(
            fruit_idx as u64,
            radius,
            ripeness,
            sugar_content,
            odor_t,
            microbial_t,
            humidity_t,
            fruit_offset_world,
            fruit_velocity_world,
            skin_material,
            pulp_material,
            core_material,
            stem_material,
        );
        snapshot.fruits.push(TerrariumFruitRender {
            render_id: fruit_render_id,
            translation_world: [
                translation[0] + fruit_offset_world[0],
                ground_y
                    + 0.12
                    + fruit.source.z as f32 * RENDER_ALTITUDE_SCALE * 0.6
                    + fruit_offset_world[1],
                translation[2] + fruit_offset_world[2],
            ],
            ripeness,
            sugar_content,
            parts,
        });
    }
    snapshot.plumes = self.render_explicit_plumes(
        &terrain,
        terrain_min,
        terrain_inv,
        &snapshot.explicit_microbes,
    );

    for (fly_idx, fly) in self.flies.iter().enumerate() {
        let body = fly.body_state();
        let fly_render_id = terrarium_render_id(TERRARIUM_RENDER_ID_FLY, fly_idx as u64, 0);
        let (translation_world, _) = fly_translation_world_from_body(
            &self.config,
            &terrain,
            terrain_min,
            terrain_inv,
            body,
        );
        let energy_t = (body.energy / 3700.0).clamp(0.0, 1.0);
        let air_load = body.air_load.clamp(0.0, 1.0);
        let body_scale_world = [0.16, 0.08, 0.28];
        let wing_scale_world = [0.34, 0.16, 1.0];
        let body_material = self.render_fly_body_material(energy_t, air_load);
        let wing_material = self.render_fly_wing_material(air_load);
        let parts = render_stateful_fly_parts(
            fly_idx as u64,
            body,
            body_scale_world,
            wing_scale_world,
            energy_t,
            air_load,
            body_material.clone(),
            wing_material.clone(),
        );
        snapshot.flies.push(TerrariumFlyRender {
            render_id: fly_render_id,
            point_light_render_fingerprint: 0,
            translation_world,
            body_rotation_yxz_rad: [
                -body.heading + std::f32::consts::PI * 0.5 + body.yaw_slip.clamp(-0.2, 0.2),
                body.pitch.clamp(-0.7, 0.7),
                body.roll.clamp(-0.7, 0.7),
            ],
            parts,
            point_light_intensity: if body.is_flying {
                28.0 + air_load * 42.0
            } else {
                14.0 + air_load * 14.0
            },
            point_light_color_rgb: lerp_rgb(
                rgb(1.0, 0.72, 0.42),
                rgb(1.0, 0.86, 0.58),
                energy_t * 0.65 + air_load * 0.35,
            ),
            point_light_range: if body.is_flying {
                1.3 + air_load * 0.4
            } else {
                1.0
            },
            point_light_translation_local: [0.0, 0.22, 0.0],
        });
    }

    stamp_dynamic_render_fingerprints(&mut snapshot);
    snapshot
}

pub(super) fn build_fly_collision_scene(
    &self,
    terrain: &[f32],
    terrain_min: f32,
    terrain_inv: f32,
) -> (
    Vec<TerrariumRaycastSurface>,
    Vec<TerrariumRaycastBvhNode>,
    Option<usize>,
) {
    let mut snapshot = TerrariumDynamicRenderSnapshot::default();
    let depth = self.config.depth.max(1);

    for (water_idx, water) in self.waters.iter().enumerate() {
        if !water.alive {
            continue;
        }
        let idx = idx2(self.config.width, water.x, water.y);
        let ground_y =
            render_top_surface_y(normalize_unit(terrain[idx], terrain_min, terrain_inv));
        let mut translation = render_cell_center_world(
            self.config.width,
            self.config.height,
            water.x as f32 + 0.5,
            water.y as f32 + 0.5,
        );
        let radius = (0.18 + water.volume * 0.0012).clamp(0.18, 0.54);
        let scale_world = [radius, 0.06 + radius * 0.22, radius];
        let air_idx = idx3(
            self.config.width,
            self.config.height,
            water.x,
            water.y,
            water.z.min(depth - 1),
        );
        translation[1] = ground_y + 0.06 + water.z as f32 * 0.04;
        snapshot.waters.push(TerrariumWaterRender {
            render_id: terrarium_render_id(TERRARIUM_RENDER_ID_WATER, water_idx as u64, 0),
            render_fingerprint: 0,
            primitive: TerrariumMeshPrimitive::UvSphere {
                sectors: 14,
                stacks: 9,
            },
            mesh: render_stateful_water_mesh(
                scale_world,
                [
                    self.wind_x[air_idx] * 0.04,
                    self.wind_z[air_idx] * 0.04,
                    self.wind_y[air_idx] * 0.04,
                ],
                clamp(
                    (self.air_pressure_kpa[air_idx] - ATMOS_PRESSURE_BASELINE_KPA) / 4.0 * 0.5
                        + 0.5,
                    0.0,
                    1.0,
                ),
                self.humidity[air_idx].clamp(0.0, 1.0),
                (water.volume / 180.0).clamp(0.0, 1.0),
            ),
            translation_world: translation,
            scale_world,
            material: self.render_water_material(radius),
        });
    }

    for (plant_idx, plant) in self.plants.iter().enumerate() {
        let idx = idx2(self.config.width, plant.x, plant.y);
        let ground_y =
            render_top_surface_y(normalize_unit(terrain[idx], terrain_min, terrain_inv));
        let translation = render_cell_center_world(
            self.config.width,
            self.config.height,
            plant.x as f32 + 0.5,
            plant.y as f32 + 0.5,
        );
        let leaf_cluster = plant.cellular.cluster_snapshot(PlantTissue::Leaf);
        let stem_cluster = plant.cellular.cluster_snapshot(PlantTissue::Stem);
        let height = (plant.physiology.height_mm() * 0.055).clamp(0.22, 1.55);
        let canopy_radius = (plant.canopy_radius_cells() as f32 * 0.16).clamp(0.22, 1.25);
        let health = (plant.physiology.health() * 0.55
            + leaf_cluster.vitality.clamp(0.0, 1.0) * 0.45)
            .clamp(0.0, 1.0);
        let leaf_energy = plant_cluster_energy_charge(&leaf_cluster);
        let stem_energy = plant_cluster_energy_charge(&stem_cluster);
        let pose_scale = height / plant.physiology.height_mm().max(1.0);
        let canopy_offset_world = [
            plant.pose.canopy_offset_mm[0] * pose_scale,
            plant.pose.canopy_offset_mm[1] * pose_scale,
            plant.pose.canopy_offset_mm[2] * pose_scale,
        ];
        let canopy_translation_world = [
            translation[0] + canopy_offset_world[0],
            ground_y + height + canopy_offset_world[1],
            translation[2] + canopy_offset_world[2],
        ];
        let stem_translation_world = [translation[0], ground_y + height * 0.5, translation[2]];
        let stem_rotation_xyz_rad = [
            plant.pose.stem_tilt_x_rad * 0.35,
            0.0,
            plant.pose.stem_tilt_z_rad * 0.35,
        ];
        let canopy_rotation_xyz_rad = [
            plant.pose.stem_tilt_x_rad * 0.30,
            0.0,
            plant.pose.stem_tilt_z_rad * 0.30,
        ];
        let stem_radius = (0.06
            + (stem_cluster.cell_count / 140.0).sqrt() * 0.02
            + stem_cluster.transcript_transport_program.clamp(0.0, 1.0) * 0.03)
            .clamp(0.06, 0.16);
        let stem_scale_world = [stem_radius, height, stem_radius];
        let canopy_scale_world = [canopy_radius, 0.44 + leaf_energy * 0.10, canopy_radius];
        snapshot.plants.push(TerrariumPlantRender {
            render_id: terrarium_render_id(TERRARIUM_RENDER_ID_PLANT, plant_idx as u64, 0),
            stem_render_fingerprint: 0,
            canopy_render_fingerprint: 0,
            stem_primitive: TerrariumMeshPrimitive::Cylinder { resolution: 12 },
            stem_mesh: render_stateful_stem_mesh(
                stem_scale_world[1],
                stem_scale_world[0] * 0.5,
                canopy_offset_world,
                &stem_cluster,
            ),
            stem_translation_world,
            stem_rotation_xyz_rad,
            stem_scale_world,
            canopy_primitive: TerrariumMeshPrimitive::UvSphere {
                sectors: 12,
                stacks: 8,
            },
            canopy_mesh: render_ellipsoid_mesh(
                [
                    canopy_scale_world[0] * 0.58,
                    canopy_scale_world[1] * 0.42,
                    canopy_scale_world[2] * 0.58,
                ],
                12,
                8,
            ),
            canopy_translation_world,
            canopy_rotation_xyz_rad,
            canopy_scale_world,
            health,
            leaf_energy,
            tissues: Vec::new(),
            stem_material: self.render_plant_stem_material(health, stem_energy),
            canopy_material: self.render_plant_canopy_material(health, leaf_energy),
        });
    }

    for (seed_idx, seed) in self.seeds.iter().enumerate() {
        let sx = seed.x.clamp(0.0, self.config.width as f32 - 0.01);
        let sy = seed.y.clamp(0.0, self.config.height as f32 - 0.01);
        let cell_x = sx.floor() as usize;
        let cell_y = sy.floor() as usize;
        let idx = idx2(self.config.width, cell_x, cell_y);
        let ground_y =
            render_top_surface_y(normalize_unit(terrain[idx], terrain_min, terrain_inv));
        let mut translation =
            render_cell_center_world(self.config.width, self.config.height, sx, sy);
        let coat_cluster = seed.cellular.cluster_snapshot(SeedTissue::Coat);
        let endosperm_cluster = seed.cellular.cluster_snapshot(SeedTissue::Endosperm);
        let radicle_cluster = seed.cellular.cluster_snapshot(SeedTissue::Radicle);
        let cotyledon_cluster = seed.cellular.cluster_snapshot(SeedTissue::Cotyledon);
        let total_cells = coat_cluster.cell_count
            + endosperm_cluster.cell_count
            + radicle_cluster.cell_count
            + cotyledon_cluster.cell_count;
        let hydration_t = seed.cellular.hydration().min(1.2) / 1.2;
        let vitality_t = seed.cellular.vitality().clamp(0.0, 1.0);
        let energy_t = seed.cellular.energy_charge().clamp(0.0, 1.0);
        let uniform_scale_world = (0.028
            + total_cells.sqrt() * 0.0045
            + hydration_t * 0.010
            + vitality_t * 0.006
            + energy_t * 0.004)
            .clamp(0.040, 0.12);
        let pose_scale = uniform_scale_world / self.config.cell_size_mm.max(1.0e-3);
        translation[0] += seed.pose.offset_mm[0] * pose_scale;
        translation[1] = ground_y + 0.04 + seed.pose.offset_mm[1] * pose_scale;
        translation[2] += seed.pose.offset_mm[2] * pose_scale;
        let dormancy_t = clamp(seed.dormancy_s / 26_000.0, 0.0, 1.0);
        snapshot.seeds.push(TerrariumSeedRender {
            render_id: terrarium_render_id(TERRARIUM_RENDER_ID_SEED, seed_idx as u64, 0),
            translation_world: translation,
            rotation_xyz_rad: seed.pose.rotation_xyz_rad,
            parts: vec![TerrariumSeedPartRender {
                render_id: terrarium_render_id(
                    TERRARIUM_RENDER_ID_SEED,
                    seed_idx as u64,
                    terrarium_seed_part_render_slot(TerrariumSeedPartKind::Coat),
                ),
                render_fingerprint: 0,
                kind: TerrariumSeedPartKind::Coat,
                mesh: render_ellipsoid_mesh(
                    [
                        uniform_scale_world * 0.42,
                        uniform_scale_world * (0.54 + dormancy_t * 0.06),
                        uniform_scale_world * 0.40,
                    ],
                    10,
                    7,
                ),
                translation_local: [0.0, 0.0, 0.0],
                rotation_xyz_rad: [0.0, 0.0, 0.0],
                material: self.render_seed_part_material(
                    TerrariumSeedPartKind::Coat,
                    &coat_cluster,
                    dormancy_t,
                ),
            }],
        });
    }

    for (fruit_idx, fruit) in self.fruits.iter().enumerate() {
        if !fruit.source.alive || fruit.source.sugar_content <= 0.01 {
            continue;
        }
        let idx = idx2(self.config.width, fruit.source.x, fruit.source.y);
        let ground_y =
            render_top_surface_y(normalize_unit(terrain[idx], terrain_min, terrain_inv));
        let translation = render_cell_center_world(
            self.config.width,
            self.config.height,
            fruit.source.x as f32 + 0.5,
            fruit.source.y as f32 + 0.5,
        );
        let radius = (0.12 + fruit.radius * 0.10).clamp(0.12, 0.34);
        let pose_scale =
            radius / (fruit.radius * self.config.cell_size_mm.max(1.0e-3)).max(1.0e-3);
        let fruit_offset_world = [
            fruit.pose.offset_mm[0] * pose_scale,
            fruit.pose.offset_mm[1] * pose_scale,
            fruit.pose.offset_mm[2] * pose_scale,
        ];
        let fruit_velocity_world = [
            fruit.pose.velocity_mm_s[0] * pose_scale * 0.10,
            fruit.pose.velocity_mm_s[1] * pose_scale * 0.10,
            fruit.pose.velocity_mm_s[2] * pose_scale * 0.10,
        ];
        let odor_t = clamp(fruit.source.odorant_emission_rate / 0.18, 0.0, 1.0);
        let microbial_t = clamp(
            (self.microbial_biomass[idx]
                + self.nitrifier_biomass[idx] * 0.55
                + self.denitrifier_biomass[idx] * 0.70
                + self.explicit_microbe_activity[idx] * 0.45)
                / 1.6,
            0.0,
            1.0,
        );
        let humidity_t = clamp(
            self.sample_humidity_at(
                fruit.source.x,
                fruit.source.y,
                fruit.source.z.min(self.config.depth.max(1) - 1),
            ),
            0.0,
            1.0,
        );
        let ripeness = fruit.source.ripeness.clamp(0.0, 1.0);
        let sugar_content = fruit.source.sugar_content.clamp(0.0, 1.0);
        let stem_height = radius * (0.34 + (1.0 - ripeness) * 0.10 + humidity_t * 0.06);
        let stem_radius = radius * (0.06 + odor_t * 0.02);
        let sway_x = clamp(fruit_velocity_world[2] * 0.05, -0.18, 0.18);
        let sway_z = clamp(-fruit_velocity_world[0] * 0.05, -0.18, 0.18);
        snapshot.fruits.push(TerrariumFruitRender {
            render_id: terrarium_render_id(TERRARIUM_RENDER_ID_FRUIT, fruit_idx as u64, 0),
            translation_world: [
                translation[0] + fruit_offset_world[0],
                ground_y
                    + 0.12
                    + fruit.source.z as f32 * RENDER_ALTITUDE_SCALE * 0.6
                    + fruit_offset_world[1],
                translation[2] + fruit_offset_world[2],
            ],
            ripeness,
            sugar_content,
            parts: vec![
                TerrariumFruitPartRender {
                    render_id: terrarium_render_id(
                        TERRARIUM_RENDER_ID_FRUIT,
                        fruit_idx as u64,
                        terrarium_fruit_part_render_slot(TerrariumFruitPartKind::Skin),
                    ),
                    render_fingerprint: 0,
                    kind: TerrariumFruitPartKind::Skin,
                    mesh: render_stateful_fruit_mesh(
                        radius,
                        ripeness,
                        sugar_content,
                        fruit_offset_world,
                        fruit_velocity_world,
                    ),
                    translation_local: [0.0, 0.0, 0.0],
                    rotation_xyz_rad: [0.0, 0.0, 0.0],
                    material: self.render_fruit_part_material(
                        TerrariumFruitPartKind::Skin,
                        ripeness,
                        sugar_content,
                        odor_t,
                        microbial_t,
                        humidity_t,
                    ),
                },
                TerrariumFruitPartRender {
                    render_id: terrarium_render_id(
                        TERRARIUM_RENDER_ID_FRUIT,
                        fruit_idx as u64,
                        terrarium_fruit_part_render_slot(TerrariumFruitPartKind::Stem),
                    ),
                    render_fingerprint: 0,
                    kind: TerrariumFruitPartKind::Stem,
                    mesh: render_cylinder_mesh(stem_radius, stem_height, 10),
                    translation_local: [0.0, radius * 0.56 + stem_height * 0.40, 0.0],
                    rotation_xyz_rad: [sway_x, 0.0, sway_z],
                    material: self.render_fruit_part_material(
                        TerrariumFruitPartKind::Stem,
                        ripeness,
                        sugar_content,
                        odor_t,
                        microbial_t,
                        humidity_t,
                    ),
                },
            ],
        });
    }

    build_dynamic_raycast_scene(&snapshot)
}

pub fn render_dynamic_snapshot(&self) -> TerrariumDynamicRenderSnapshot {
    self.build_dynamic_snapshot()
}

pub fn render_dynamic_snapshot_cached(
    &mut self,
    revision: u64,
) -> &TerrariumDynamicRenderSnapshot {
    if self.cached_dynamic_snapshot.revision != revision {
        let snapshot = self.build_dynamic_snapshot();
        let (delta, fingerprints) =
            build_dynamic_render_delta(&snapshot, &self.cached_dynamic_snapshot.fingerprints);
        self.cached_raycast_scene
            .sync_with_snapshot(revision, &snapshot);
        self.cached_dynamic_snapshot.snapshot = snapshot;
        self.cached_dynamic_snapshot.delta = delta;
        self.cached_dynamic_snapshot.fingerprints = fingerprints;
        self.cached_dynamic_snapshot.revision = revision;
    }
    &self.cached_dynamic_snapshot.snapshot
}

pub fn render_substrate_batches_cached(
    &mut self,
    revision: u64,
) -> &[TerrariumSubstrateBatchRender] {
    let _ = self.render_dynamic_snapshot_cached(revision);
    if self.cached_substrate_batches.revision != revision {
        self.cached_substrate_batches.batches =
            build_substrate_batch_renders(&self.cached_dynamic_snapshot.snapshot);
        self.cached_substrate_batches.revision = revision;
    }
    &self.cached_substrate_batches.batches
}

pub fn render_dynamic_batches_cached(
    &mut self,
    revision: u64,
) -> &[TerrariumDynamicBatchRender] {
    let _ = self.render_dynamic_snapshot_cached(revision);
    if self.cached_dynamic_batches.revision != revision {
        self.cached_dynamic_batches.batches =
            build_dynamic_batch_renders(&self.cached_dynamic_snapshot.snapshot);
        self.cached_dynamic_batches.revision = revision;
    }
    &self.cached_dynamic_batches.batches
}

pub fn render_dynamic_delta_cached(&mut self, revision: u64) -> &TerrariumDynamicRenderDelta {
    let _ = self.render_dynamic_snapshot_cached(revision);
    &self.cached_dynamic_snapshot.delta
}

fn render_scene_raycast_internal(
    &mut self,
    revision: u64,
    origin_world: [f32; 3],
    direction_world: [f32; 3],
    cutaway: bool,
    solid_only: bool,
) -> Option<TerrariumSceneRaycastHit> {
    let _ = self.render_dynamic_snapshot_cached(revision);
    self.cached_raycast_scene
        .raycast(origin_world, direction_world, cutaway, solid_only, 0)
}

pub fn render_scene_raycast_cached(
    &mut self,
    revision: u64,
    origin_world: [f32; 3],
    direction_world: [f32; 3],
    cutaway: bool,
) -> Option<TerrariumSceneRaycastHit> {
    self.render_scene_raycast_internal(revision, origin_world, direction_world, cutaway, false)
}

pub fn render_scene_solid_raycast_cached(
    &mut self,
    revision: u64,
    origin_world: [f32; 3],
    direction_world: [f32; 3],
    cutaway: bool,
) -> Option<TerrariumSceneRaycastHit> {
    self.render_scene_raycast_internal(revision, origin_world, direction_world, cutaway, true)
}

pub(super) fn resolve_fly_contacts_with_scene(
    config: &TerrariumWorldConfig,
    fly: &mut DrosophilaSim,
    previous_body: &BodyState,
    terrain: &[f32],
    terrain_min: f32,
    terrain_inv: f32,
    collision_surfaces: &[TerrariumRaycastSurface],
    collision_nodes: &[TerrariumRaycastBvhNode],
    collision_root: Option<usize>,
) {
    let excluded_class_mask = 1u16 << TERRARIUM_RENDER_ID_FLY;
    crate::terrarium_contact::resolve_fly_contacts_with_scene(
        config,
        fly,
        previous_body,
        terrain,
        terrain_min,
        terrain_inv,
        collision_surfaces,
        collision_nodes,
        collision_root,
        excluded_class_mask,
    );
}

#[allow(dead_code)]
pub(super) fn pairwise_fly_contact_radius_world(body: &BodyState) -> f32 {
    crate::terrarium_contact::pairwise_fly_contact_radius_world(body)
}

pub(super) fn resolve_pairwise_fly_contacts(
    config: &TerrariumWorldConfig,
    flies: &mut [DrosophilaSim],
    terrain: &[f32],
    terrain_min: f32,
    terrain_inv: f32,
    collision_surfaces: &[TerrariumRaycastSurface],
    collision_nodes: &[TerrariumRaycastBvhNode],
    collision_root: Option<usize>,
) {
    let excluded_class_mask = 1u16 << TERRARIUM_RENDER_ID_FLY;
    crate::terrarium_contact::resolve_pairwise_fly_contacts(
        config,
        flies,
        terrain,
        terrain_min,
        terrain_inv,
        collision_surfaces,
        collision_nodes,
        collision_root,
        excluded_class_mask,
    )
}

pub(super) fn scalar_field_topdown(&self, field: &[f32]) -> Vec<f32> {
    let plane = self.config.width * self.config.height;
    let depth = self.config.depth.max(1);
    if field.len() != plane * depth {
        return vec![0.0; plane];
    }
    let mut reduced = vec![0.0f32; plane];
    for z in 0..depth {
        let start = z * plane;
        let end = start + plane;
        for (dst, value) in reduced.iter_mut().zip(field[start..end].iter()) {
            *dst += *value;
        }
    }
    for value in &mut reduced {
        *value /= depth as f32;
    }
    reduced
}
}
