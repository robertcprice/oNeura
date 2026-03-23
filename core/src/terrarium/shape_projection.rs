use crate::constants::clamp;
use crate::plant_cellular::{PlantCellularStateSim, PlantTissue};
use crate::plant_organism::PlantOrganismSim;

use super::TerrariumPlantGenome;

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct TerrariumPlantStructureDescriptor {
    pub internode_length: f32,
    pub branch_angle_rad: f32,
    pub thickness_base: f32,
    pub fruit_radius_scale: f32,
    pub leaf_radius_scale: f32,
    pub lateral_bias: f32,
    pub droop_factor: f32,
    pub branch_twist_rad: f32,
    pub branch_depth_attenuation: f32,
    pub canopy_depth_scale: f32,
    pub leaf_cluster_density: f32,
}

pub fn structure_descriptor_from_morphology(
    morphology: &crate::botany::PlantMorphology,
) -> TerrariumPlantStructureDescriptor {
    TerrariumPlantStructureDescriptor {
        internode_length: morphology.internode_length,
        branch_angle_rad: morphology.branch_angle_rad,
        thickness_base: morphology.thickness_base,
        fruit_radius_scale: morphology.fruit_radius_scale,
        leaf_radius_scale: morphology.leaf_radius_scale,
        lateral_bias: morphology.lateral_bias,
        droop_factor: morphology.droop_factor,
        branch_twist_rad: morphology.branch_twist_rad,
        branch_depth_attenuation: morphology.branch_depth_attenuation,
        canopy_depth_scale: morphology.canopy_depth_scale,
        leaf_cluster_density: morphology.leaf_cluster_density,
    }
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct TerrariumFruitShapeDescriptor {
    pub width_scale: f32,
    pub height_scale: f32,
    pub depth_scale: f32,
    pub top_taper: f32,
    pub stem_length: f32,
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct TerrariumSeedShapeDescriptor {
    pub width_scale: f32,
    pub height_scale: f32,
    pub depth_scale: f32,
    pub awn_length: f32,
}

pub fn plant_structure_descriptor(
    genome: &TerrariumPlantGenome,
    physiology: &PlantOrganismSim,
    cellular: &PlantCellularStateSim,
    moisture: f32,
    light: f32,
    temperature_c: f32,
) -> TerrariumPlantStructureDescriptor {
    let leaf = cellular.cluster_snapshot(PlantTissue::Leaf);
    let stem = cellular.cluster_snapshot(PlantTissue::Stem);
    let root = cellular.cluster_snapshot(PlantTissue::Root);
    let meristem = cellular.cluster_snapshot(PlantTissue::Meristem);
    let total_cells = cellular.total_cells().max(1.0);

    let leaf_frac = leaf.cell_count / total_cells;
    let stem_frac = stem.cell_count / total_cells;
    let root_frac = root.cell_count / total_cells;
    let meristem_frac = meristem.cell_count / total_cells;

    let vitality = cellular.vitality().clamp(0.0, 1.0);
    let energy = cellular.energy_charge().clamp(0.0, 1.0);
    let division_pressure = clamp(
        cellular.division_signal() * 0.55
            + (cellular.last_new_cells() / total_cells * 28.0).clamp(0.0, 1.0) * 0.45,
        0.0,
        1.0,
    );
    let maturity = clamp(
        physiology.age_s() / 18_000.0 + physiology.height_mm() / genome.max_height_mm.max(1.0),
        0.0,
        1.4,
    );
    let hydration = clamp(
        moisture * 0.52 + (physiology.water_buffer() / 1.2).clamp(0.0, 1.0) * 0.48,
        0.0,
        1.0,
    );
    let nutrient_status = clamp(
        (physiology.nitrogen_buffer() / 0.7).clamp(0.0, 1.0) * 0.55
            + (root.state_nitrate / 1.6).clamp(0.0, 1.0) * 0.45,
        0.0,
        1.0,
    );
    let temperature_factor = clamp((temperature_c - 6.0) / 24.0, 0.0, 1.0);
    let drought = 1.0 - hydration;
    let reproductive_load = clamp(
        physiology.fruit_count() as f32 * 0.12
            + (physiology.storage_carbon() / 0.45).clamp(0.0, 1.0) * 0.52,
        0.0,
        1.2,
    );
    let woodiness = clamp(
        stem_frac * 1.35
            + (physiology.stem_biomass() / (physiology.total_biomass() + 1e-6)).clamp(0.0, 1.0)
                * 0.55
            + maturity * 0.18,
        0.0,
        1.2,
    );
    let slenderness = clamp(
        genome.max_height_mm / (genome.canopy_radius_mm * 2.0 + 1.0),
        0.45,
        2.25,
    );
    let root_to_canopy = clamp(
        genome.root_radius_mm / genome.canopy_radius_mm.max(1.0),
        0.35,
        1.8,
    );

    let internode_length = clamp(
        0.18 + genome.max_height_mm * 0.020
            + division_pressure * 0.28
            + temperature_factor * 0.10
            + light * 0.10
            + hydration * 0.12
            - drought * 0.14,
        0.16,
        1.30,
    );
    let branch_angle_rad = clamp(
        0.16 + (1.0 - light) * 0.28
            + meristem_frac * 0.30
            + root_frac * 0.12
            + genome.shade_tolerance * 0.10
            - woodiness * 0.08,
        0.10,
        1.18,
    );
    let thickness_base = clamp(
        0.02 + woodiness * 0.16
            + vitality * 0.04
            + genome.root_depth_bias * 0.02
            + nutrient_status * 0.03,
        0.02,
        0.28,
    );
    let fruit_radius_scale = clamp(
        0.30 + reproductive_load * 0.42 + nutrient_status * 0.12 + temperature_factor * 0.08
            - drought * 0.10,
        0.25,
        1.70,
    );
    let leaf_radius_scale = clamp(
        0.48 + leaf_frac * 1.90
            + (leaf.transcript_transport_program * 0.20)
            + hydration * 0.18
            + light * 0.12
            - drought * 0.16,
        0.50,
        2.30,
    );
    let lateral_bias = clamp(
        0.48 + meristem_frac * 0.34
            + division_pressure * 0.22
            + light * 0.10
            + genome.shade_tolerance * 0.12
            - woodiness * 0.05,
        0.42,
        1.25,
    );
    let droop_factor = clamp(
        drought * 0.42 + leaf_frac * 0.18 + reproductive_load * 0.12
            - stem_frac * 0.16
            - energy * 0.08,
        0.0,
        0.82,
    );
    let branch_twist_rad = clamp(
        0.06 + ((slenderness - 0.45) / 1.8) * 0.28
            + genome.shade_tolerance * 0.10
            + meristem_frac * 0.16
            + division_pressure * 0.12
            - woodiness * 0.06,
        0.04,
        0.72,
    );
    let branch_depth_attenuation = clamp(
        0.42 + woodiness * 0.24 + root_to_canopy * 0.10 + vitality * 0.08
            - division_pressure * 0.06,
        0.38,
        1.08,
    );
    let canopy_depth_scale = clamp(
        0.44 + leaf_frac * 0.52 + root_frac * 0.10 + hydration * 0.18 + (1.0 / slenderness) * 0.08
            - drought * 0.18,
        0.30,
        1.35,
    );
    let leaf_cluster_density = clamp(
        0.28 + leaf_frac * 1.25 + division_pressure * 0.16 + light * 0.10 + hydration * 0.14
            - drought * 0.12,
        0.18,
        1.45,
    );

    TerrariumPlantStructureDescriptor {
        internode_length,
        branch_angle_rad,
        thickness_base,
        fruit_radius_scale,
        leaf_radius_scale,
        lateral_bias,
        droop_factor,
        branch_twist_rad,
        branch_depth_attenuation,
        canopy_depth_scale,
        leaf_cluster_density,
    }
}

pub fn fruit_shape_descriptor(
    genome: &TerrariumPlantGenome,
    radius: f32,
    ripeness: f32,
    sugar_content: f32,
) -> TerrariumFruitShapeDescriptor {
    let ripe_t = ripeness.clamp(0.0, 1.0);
    let sugar_t = clamp(sugar_content / 1.6, 0.0, 1.0);
    let size_t = clamp(radius / 1.8, 0.3, 1.4);
    let slenderness = clamp(
        genome.max_height_mm / (genome.canopy_radius_mm * 2.0 + 1.0),
        0.45,
        2.25,
    );
    let root_to_canopy = clamp(
        genome.root_radius_mm / genome.canopy_radius_mm.max(1.0),
        0.35,
        1.8,
    );
    let seed_mass_t = clamp(genome.seed_mass / 0.20, 0.0, 1.0);
    let volatile_t = clamp((genome.volatile_scale - 0.45) / 1.4, 0.0, 1.0);

    let width_scale = clamp(
        0.42 + (1.0 / slenderness) * 0.34 + size_t * 0.18 + sugar_t * 0.12 + root_to_canopy * 0.08,
        0.28,
        1.42,
    );
    let height_scale = clamp(
        0.48 + slenderness * 0.24 + genome.water_use_efficiency * 0.12 + ripe_t * 0.08
            - sugar_t * 0.04,
        0.34,
        1.52,
    );
    let depth_scale = clamp(
        0.40 + root_to_canopy * 0.18 + size_t * 0.16 + seed_mass_t * 0.18 + sugar_t * 0.06,
        0.28,
        1.42,
    );
    let top_taper = clamp(
        (slenderness - 0.9).max(0.0) * 0.16 + volatile_t * 0.12 + (1.0 - ripe_t) * 0.10,
        0.0,
        0.40,
    );
    let stem_length = clamp(
        0.05 + root_to_canopy * 0.08 + (1.0 - sugar_t) * 0.06 + volatile_t * 0.04,
        0.04,
        0.32,
    );

    TerrariumFruitShapeDescriptor {
        width_scale,
        height_scale,
        depth_scale,
        top_taper,
        stem_length,
    }
}

pub fn seed_shape_descriptor(
    genome: &TerrariumPlantGenome,
    reserve_carbon: f32,
    dormancy_s: f32,
) -> TerrariumSeedShapeDescriptor {
    let reserve_t = clamp(reserve_carbon / 0.25, 0.0, 1.0);
    let dormancy_t = clamp(dormancy_s / 26_000.0, 0.0, 1.0);
    let mass_t = clamp(genome.seed_mass / 0.20, 0.0, 1.0);
    let slenderness = clamp(
        genome.max_height_mm / (genome.canopy_radius_mm * 2.0 + 1.0),
        0.45,
        2.25,
    );
    let dry_bias = clamp(genome.water_use_efficiency / 1.5, 0.0, 1.0);
    let root_bias = clamp(genome.root_depth_bias / 1.1, 0.0, 1.0);

    let width_scale = clamp(
        0.26 + mass_t * 0.62 + reserve_t * 0.10 + (1.0 / slenderness) * 0.10,
        0.18,
        1.10,
    );
    let height_scale = clamp(
        0.22 + reserve_t * 0.26 + genome.leaf_efficiency * 0.10 - dormancy_t * 0.04,
        0.18,
        1.00,
    );
    let depth_scale = clamp(
        0.42 + mass_t * 0.78 + root_bias * 0.22 + slenderness * 0.06,
        0.32,
        1.80,
    );
    let awn_length = clamp(
        dormancy_t * 0.20 + dry_bias * 0.18 + (1.0 - mass_t) * 0.10 + slenderness * 0.04,
        0.0,
        0.70,
    );

    TerrariumSeedShapeDescriptor {
        width_scale,
        height_scale,
        depth_scale,
        awn_length,
    }
}
