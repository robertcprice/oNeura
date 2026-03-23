use super::{TerrariumPlant, TerrariumPlantGenome};
use crate::constants::clamp;
use crate::seed_cellular::SeedCellularStateSim;
use crate::terrarium::material_exchange::{
    deposit_species_to_inventory, inventory_component_amount, withdraw_species_from_inventory,
};
use crate::terrarium::{RegionalMaterialInventory, TerrariumSpecies};
use rand::Rng;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TerrariumFruitReproductionState {
    pub paternal_genome: Option<TerrariumPlantGenome>,
    pub paternal_taxonomy_id: Option<u32>,
    pub paternal_organism_id: Option<u64>,
    pub embryo_genome: Option<TerrariumPlantGenome>,
    pub embryo_cellular: Option<SeedCellularStateSim>,
    pub reserve_carbon: f32,
    pub dormancy_s: f32,
    pub pollination_score: f32,
    pub self_pollinated: bool,
    pub seed_released: bool,
    pub material_inventory: RegionalMaterialInventory,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ReleasedSeedState {
    pub embryo_genome: TerrariumPlantGenome,
    pub embryo_cellular: SeedCellularStateSim,
    pub reserve_carbon: f32,
    pub dormancy_s: f32,
    pub material_inventory: RegionalMaterialInventory,
}

pub fn initialize_fruit_reproduction<R: Rng + ?Sized>(
    plants: &[TerrariumPlant],
    parent_x: usize,
    parent_y: usize,
    maternal_genome: &TerrariumPlantGenome,
    rng: &mut R,
) -> TerrariumFruitReproductionState {
    let (paternal_genome, paternal_organism_id, pollination_score, self_pollinated) =
        select_pollen_donor(plants, parent_x, parent_y, maternal_genome, 0.0, 0.0)
            .unwrap_or_else(|| (maternal_genome.clone(), None, 0.42, true));
    let embryo_genome = maternal_genome.recombine_with(&paternal_genome, rng);
    let dormancy_s = clamp(
        12_000.0 + embryo_genome.seed_mass * 28_000.0,
        4_800.0,
        24_000.0,
    );
    let reserve_carbon = clamp(
        embryo_genome.seed_mass * (0.30 + pollination_score * 0.18),
        0.012,
        0.08,
    );
    let embryo_cellular =
        SeedCellularStateSim::new(embryo_genome.seed_mass, reserve_carbon, dormancy_s);
    let mut material_inventory = RegionalMaterialInventory::new("fruit-embryo:init".into());
    deposit_species_to_inventory(
        &mut material_inventory,
        TerrariumSpecies::Water,
        0.14 + embryo_genome.seed_mass * 1.1,
    );
    deposit_species_to_inventory(
        &mut material_inventory,
        TerrariumSpecies::Glucose,
        reserve_carbon * 0.42,
    );
    deposit_species_to_inventory(
        &mut material_inventory,
        TerrariumSpecies::AminoAcidPool,
        reserve_carbon * 0.20,
    );
    deposit_species_to_inventory(
        &mut material_inventory,
        TerrariumSpecies::NucleotidePool,
        reserve_carbon * 0.12,
    );
    deposit_species_to_inventory(
        &mut material_inventory,
        TerrariumSpecies::MembranePrecursorPool,
        reserve_carbon * 0.10,
    );
    TerrariumFruitReproductionState {
        paternal_taxonomy_id: Some(paternal_genome.taxonomy_id),
        paternal_organism_id,
        paternal_genome: Some(paternal_genome),
        embryo_genome: Some(embryo_genome),
        embryo_cellular: Some(embryo_cellular),
        reserve_carbon,
        dormancy_s,
        pollination_score,
        self_pollinated,
        seed_released: false,
        material_inventory,
    }
}

pub fn step_fruit_reproduction(
    reproduction: &mut TerrariumFruitReproductionState,
    seed_mass: f32,
    ripeness: f32,
    sugar_content: f32,
    moisture: f32,
    deep_moisture: f32,
    nutrients: f32,
    fruit_surface_inventory: &RegionalMaterialInventory,
    dt: f32,
) {
    if reproduction.seed_released {
        return;
    }
    let Some(embryo_cellular) = reproduction.embryo_cellular.as_mut() else {
        return;
    };

    let ripeness_t = ripeness.clamp(0.0, 1.0);
    let sugar_t = sugar_content.clamp(0.0, 1.5);
    let seed_mass_t = clamp(seed_mass / 0.20, 0.0, 1.0);
    let target_reserve = clamp(
        seed_mass * (0.26 + ripeness_t * 0.52 + reproduction.pollination_score * 0.18)
            + sugar_t * 0.02
            + seed_mass_t * 0.01,
        0.012,
        0.20,
    );
    let reserve_approach = clamp(dt / 2400.0, 0.02, 0.20);
    reproduction.reserve_carbon = clamp(
        reproduction.reserve_carbon
            + (target_reserve - reproduction.reserve_carbon) * reserve_approach,
        0.0,
        0.20,
    );
    reproduction.dormancy_s = clamp(
        reproduction.dormancy_s
            - dt * (0.04 + ripeness_t * 0.12 + reproduction.pollination_score * 0.04),
        2_400.0,
        24_000.0,
    );
    let surface_water =
        inventory_component_amount(fruit_surface_inventory, TerrariumSpecies::Water);
    let surface_glucose =
        inventory_component_amount(fruit_surface_inventory, TerrariumSpecies::Glucose);
    let surface_amino =
        inventory_component_amount(fruit_surface_inventory, TerrariumSpecies::AminoAcidPool);
    let surface_nucleotide =
        inventory_component_amount(fruit_surface_inventory, TerrariumSpecies::NucleotidePool);
    let surface_membrane = inventory_component_amount(
        fruit_surface_inventory,
        TerrariumSpecies::MembranePrecursorPool,
    );
    let surface_oxygen =
        inventory_component_amount(fruit_surface_inventory, TerrariumSpecies::OxygenGas);
    let surface_co2 =
        inventory_component_amount(fruit_surface_inventory, TerrariumSpecies::CarbonDioxide);
    let mut embryo_target = RegionalMaterialInventory::new("fruit-embryo:target".into());
    deposit_species_to_inventory(
        &mut embryo_target,
        TerrariumSpecies::Water,
        clamp(
            0.10 + moisture * 0.08 + deep_moisture * 0.05 + surface_water * 0.42,
            0.02,
            2.0,
        ),
    );
    deposit_species_to_inventory(
        &mut embryo_target,
        TerrariumSpecies::Glucose,
        clamp(
            reproduction.reserve_carbon * 0.44 + sugar_t * 0.18 + surface_glucose * 0.28,
            0.0,
            1.8,
        ),
    );
    deposit_species_to_inventory(
        &mut embryo_target,
        TerrariumSpecies::AminoAcidPool,
        clamp(
            reproduction.reserve_carbon * 0.18 + surface_amino * 0.36 + nutrients * 0.08,
            0.0,
            1.2,
        ),
    );
    deposit_species_to_inventory(
        &mut embryo_target,
        TerrariumSpecies::NucleotidePool,
        clamp(
            reproduction.reserve_carbon * 0.12 + surface_nucleotide * 0.34,
            0.0,
            1.0,
        ),
    );
    deposit_species_to_inventory(
        &mut embryo_target,
        TerrariumSpecies::MembranePrecursorPool,
        clamp(
            reproduction.reserve_carbon * 0.10 + surface_membrane * 0.28,
            0.0,
            1.0,
        ),
    );
    deposit_species_to_inventory(
        &mut embryo_target,
        TerrariumSpecies::OxygenGas,
        clamp(0.04 + surface_oxygen * 0.72, 0.0, 0.8),
    );
    deposit_species_to_inventory(
        &mut embryo_target,
        TerrariumSpecies::CarbonDioxide,
        clamp(0.02 + surface_co2 * 0.60 + ripeness_t * 0.04, 0.0, 0.8),
    );
    if reproduction.material_inventory.is_empty() {
        reproduction.material_inventory = embryo_target.scaled(0.88);
    } else {
        let _ = reproduction
            .material_inventory
            .relax_toward(&embryo_target, 0.24);
    }
    let water_take = withdraw_species_from_inventory(
        &mut reproduction.material_inventory,
        TerrariumSpecies::Water,
        (0.004 + ripeness_t * 0.006 + surface_water * 0.001) * (dt / 120.0),
    );
    let glucose_take = withdraw_species_from_inventory(
        &mut reproduction.material_inventory,
        TerrariumSpecies::Glucose,
        (0.003 + ripeness_t * 0.007 + reproduction.reserve_carbon * 0.010) * (dt / 120.0),
    );
    let amino_take = withdraw_species_from_inventory(
        &mut reproduction.material_inventory,
        TerrariumSpecies::AminoAcidPool,
        (0.0015 + ripeness_t * 0.003 + reproduction.pollination_score * 0.002) * (dt / 120.0),
    );
    let nucleotide_take = withdraw_species_from_inventory(
        &mut reproduction.material_inventory,
        TerrariumSpecies::NucleotidePool,
        (0.0012 + ripeness_t * 0.0026 + reproduction.pollination_score * 0.0018) * (dt / 120.0),
    );
    let membrane_take = withdraw_species_from_inventory(
        &mut reproduction.material_inventory,
        TerrariumSpecies::MembranePrecursorPool,
        (0.0010 + ripeness_t * 0.0020 + seed_mass_t * 0.0014) * (dt / 120.0),
    );
    let fruit_moisture = clamp(
        moisture * 0.32 + deep_moisture * 0.18 + ripeness_t * 0.34 + sugar_t * 0.08,
        0.0,
        1.2,
    );
    let fruit_nutrients = clamp(
        nutrients * 0.26 + sugar_t * 0.10 + seed_mass_t * 0.06,
        0.0,
        1.0,
    );
    let embryo_moisture = clamp(fruit_moisture + water_take * 0.48, 0.0, 1.3);
    let embryo_nutrients = clamp(
        fruit_nutrients
            + glucose_take * 0.18
            + amino_take * 0.72
            + nucleotide_take * 0.58
            + membrane_take * 0.28,
        0.0,
        1.2,
    );
    let embryo_oxygen = clamp(0.04 + surface_oxygen * 0.42, 0.0, 0.6);
    let embryo_co2 = clamp(0.12 + surface_co2 * 0.36, 0.0, 0.8);
    let _feedback = embryo_cellular.step(
        dt,
        embryo_moisture,
        embryo_moisture * 0.82,
        embryo_nutrients,
        0.0,
        1.0,
        0.0,
        0.08,
        embryo_oxygen,
        embryo_co2,
        0.06,
        0.02,
        reproduction.dormancy_s,
        reproduction.reserve_carbon,
    );
    reproduction.reserve_carbon = reproduction
        .reserve_carbon
        .max(embryo_cellular.reserve_carbon_equivalent());
}

pub fn fruit_ready_for_seed_release(
    reproduction: &TerrariumFruitReproductionState,
    ripeness: f32,
) -> bool {
    if reproduction.seed_released {
        return false;
    }
    let Some(embryo_cellular) = reproduction.embryo_cellular.as_ref() else {
        return false;
    };
    ripeness >= 0.82
        && reproduction.reserve_carbon >= 0.028
        && embryo_cellular.vitality() >= 0.34
        && embryo_cellular.energy_charge() >= 0.16
}

pub fn release_seed(
    reproduction: &mut TerrariumFruitReproductionState,
) -> Option<ReleasedSeedState> {
    if reproduction.seed_released {
        return None;
    }
    let embryo_genome = reproduction.embryo_genome.clone()?;
    let embryo_cellular = reproduction.embryo_cellular.clone()?;
    reproduction.seed_released = true;
    Some(ReleasedSeedState {
        embryo_genome,
        embryo_cellular: embryo_cellular.clone(),
        reserve_carbon: reproduction
            .reserve_carbon
            .max(embryo_cellular.reserve_carbon_equivalent()),
        dormancy_s: reproduction.dormancy_s,
        material_inventory: reproduction.material_inventory.clone(),
    })
}

fn select_pollen_donor(
    plants: &[TerrariumPlant],
    parent_x: usize,
    parent_y: usize,
    maternal_genome: &TerrariumPlantGenome,
    wind_x: f32,
    wind_y: f32,
) -> Option<(TerrariumPlantGenome, Option<u64>, f32, bool)> {
    let maternal_growth_form =
        crate::terrarium::plant_species::plant_growth_form(maternal_genome.taxonomy_id);
    plants
        .iter()
        .filter_map(|plant| {
            if plant.x == parent_x && plant.y == parent_y {
                return None;
            }
            let donor_genome = &plant.genome;
            let donor_growth_form =
                crate::terrarium::plant_species::plant_growth_form(donor_genome.taxonomy_id);
            let compatibility = if donor_genome.taxonomy_id == maternal_genome.taxonomy_id {
                1.0
            } else if donor_growth_form == maternal_growth_form {
                0.28
            } else {
                0.0
            };
            if compatibility <= 0.0 {
                return None;
            }
            let dx = plant.x as f32 - parent_x as f32;
            let dy = plant.y as f32 - parent_y as f32;
            let dist = (dx * dx + dy * dy).sqrt();
            let wind_speed = (wind_x * wind_x + wind_y * wind_y).sqrt();
            let wind_boost = if wind_speed > 0.01 && dist > 0.01 {
                let cos_angle = (dx * wind_x + dy * wind_y) / (dist * wind_speed);
                cos_angle.max(0.0) * wind_speed * 2.0
            } else {
                0.0
            };
            let effective_range = 10.0 + wind_boost;
            let distance_t = (1.0 - dist / effective_range).clamp(0.0, 1.0);
            if distance_t <= 0.0 {
                return None;
            }
            let score =
                compatibility * distance_t * (0.28 + plant.physiology.health().max(0.0) * 0.72);
            (score > 0.10).then_some((
                donor_genome.clone(),
                Some(plant.identity.organism_id),
                score,
                false,
            ))
        })
        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrarium::{OrganismIdentity, TerrariumOrganismKind};
    use rand::SeedableRng;

    #[test]
    fn nearby_compatible_donor_creates_distinct_embryo_genome() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(17);
        let maternal = crate::terrarium::plant_species::genome_for_taxonomy(3750, &mut rng);
        let mut donor = maternal.clone();
        donor.leaf_efficiency = (donor.leaf_efficiency + 0.22).min(1.6);
        donor.root_depth_bias = (donor.root_depth_bias + 0.18).min(1.05);
        let plants = vec![
            TerrariumPlant::new(
                8,
                8,
                maternal.clone(),
                1.0,
                OrganismIdentity::synthetic(TerrariumOrganismKind::Plant, 1),
                &mut rng,
            ),
            TerrariumPlant::new(
                9,
                8,
                donor.clone(),
                1.0,
                OrganismIdentity::synthetic(TerrariumOrganismKind::Plant, 2),
                &mut rng,
            ),
        ];

        let repro = initialize_fruit_reproduction(&plants, 8, 8, &maternal, &mut rng);
        let embryo = repro
            .embryo_genome
            .expect("fruit should carry an embryo genome");

        assert_eq!(repro.paternal_taxonomy_id, Some(donor.taxonomy_id));
        assert!(
            (embryo.leaf_efficiency - maternal.leaf_efficiency).abs() > 1.0e-4
                || (embryo.root_depth_bias - maternal.root_depth_bias).abs() > 1.0e-4
        );
    }
}
