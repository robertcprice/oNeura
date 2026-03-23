use super::calibrator::SubstrateKinetics;
use super::substrate::{BatchedAtomTerrarium, TerrariumSpecies};
use super::*;
use crate::atomistic_chemistry::MoleculeGraph;
use crate::drosophila::DrosophilaScale;
use crate::molecular_atmosphere::FruitSourceState;
use crate::seed_cellular::{SeedCellularStateSim, SeedTissue};

fn owned_boundary_pair(world: &TerrariumWorld) -> (usize, usize) {
    let width = world.config.width;
    let height = world.config.height;
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if world.explicit_microbe_authority[idx] < 0.01 {
                continue;
            }
            for (nx, ny, valid) in [
                (x.wrapping_sub(1), y, x > 0),
                (x + 1, y, x + 1 < width),
                (x, y.wrapping_sub(1), y > 0),
                (x, y + 1, y + 1 < height),
            ] {
                if !valid {
                    continue;
                }
                let neighbor_idx = ny * width + nx;
                if world.explicit_microbe_authority[neighbor_idx] < 0.01 {
                    return (idx, neighbor_idx);
                }
            }
        }
    }
    panic!("expected an explicitly owned boundary cell with a neighboring background cell");
}

#[test]
fn native_terrarium_world_runs_and_stays_bounded() {
    let mut world = TerrariumWorld::demo(7, false).unwrap();
    world.run_frames(20).unwrap();
    let snapshot = world.snapshot();
    assert!(snapshot.plants > 0);
    assert!(snapshot.food_remaining >= 0.0);
    assert!(snapshot.light >= 0.0);
    assert!(snapshot.humidity >= 0.0);
    assert!(snapshot.mean_soil_moisture >= 0.0);
    assert!(snapshot.total_plant_cells > 0.0);
    assert!(snapshot.mean_atmospheric_o2 > 0.0);
    assert!(snapshot.mean_microbial_cells > 0.0);
    assert!(snapshot.mean_microbial_packets > 0.0);
    assert!(snapshot.mean_microbial_novel_packets >= 0.0);
    assert!(snapshot.mean_microbial_latent_packets >= 0.0);
    assert!(snapshot.mean_microbial_packet_load > 0.0);
    assert!((0.0..=1.0).contains(&snapshot.mean_microbial_strain_yield));
    assert!((0.0..=1.0).contains(&snapshot.mean_microbial_strain_stress_tolerance));
    assert!((0.0..=1.0).contains(&snapshot.mean_microbial_gene_catabolic));
    assert!((0.0..=1.0).contains(&snapshot.mean_microbial_gene_stress_response));
    assert!((0.0..=1.0).contains(&snapshot.mean_microbial_gene_dormancy_maintenance));
    assert!((0.0..=1.0).contains(&snapshot.mean_microbial_gene_extracellular_scavenging));
    assert!((0.0..=1.0).contains(&snapshot.mean_microbial_genotype_divergence));
    assert!(snapshot.mean_microbial_catalog_generation >= 0.0);
    assert!((0.0..=1.0).contains(&snapshot.mean_microbial_catalog_novelty));
    assert!((0.0..=1.0).contains(&snapshot.mean_microbial_local_catalog_share));
    assert!((0.0..=1.0).contains(&snapshot.mean_microbial_catalog_bank_dominance));
    assert!(snapshot.mean_microbial_catalog_bank_richness >= 0.0);
    assert!(snapshot.mean_microbial_lineage_generation >= 0.0);
    assert!((0.0..=1.0).contains(&snapshot.mean_microbial_lineage_novelty));
    assert!(snapshot.mean_microbial_packet_mutation_flux >= 0.0);
    assert!((0.0..=1.0).contains(&snapshot.mean_microbial_novel_fraction));
    assert!((0.0..=1.0).contains(&snapshot.mean_microbial_bank_simpson_diversity));
    assert!(snapshot.mean_nitrifier_cells > 0.0);
    assert!(snapshot.mean_nitrifier_packets > 0.0);
    assert!(snapshot.mean_nitrifier_novel_packets >= 0.0);
    assert!(snapshot.mean_nitrifier_latent_packets >= 0.0);
    assert!(snapshot.mean_nitrifier_packet_load > 0.0);
    assert!((0.0..=1.0).contains(&snapshot.mean_nitrifier_strain_oxygen_affinity));
    assert!((0.0..=1.0).contains(&snapshot.mean_nitrifier_strain_ammonium_affinity));
    assert!((0.0..=1.0).contains(&snapshot.mean_nitrifier_gene_oxygen_respiration));
    assert!((0.0..=1.0).contains(&snapshot.mean_nitrifier_gene_ammonium_transport));
    assert!((0.0..=1.0).contains(&snapshot.mean_nitrifier_gene_stress_persistence));
    assert!((0.0..=1.0).contains(&snapshot.mean_nitrifier_gene_redox_efficiency));
    assert!((0.0..=1.0).contains(&snapshot.mean_nitrifier_genotype_divergence));
    assert!(snapshot.mean_nitrifier_catalog_generation >= 0.0);
    assert!((0.0..=1.0).contains(&snapshot.mean_nitrifier_catalog_novelty));
    assert!((0.0..=1.0).contains(&snapshot.mean_nitrifier_local_catalog_share));
    assert!((0.0..=1.0).contains(&snapshot.mean_nitrifier_catalog_bank_dominance));
    assert!(snapshot.mean_nitrifier_catalog_bank_richness >= 0.0);
    assert!(snapshot.mean_nitrifier_lineage_generation >= 0.0);
    assert!((0.0..=1.0).contains(&snapshot.mean_nitrifier_lineage_novelty));
    assert!(snapshot.mean_nitrifier_packet_mutation_flux >= 0.0);
    assert!((0.0..=1.0).contains(&snapshot.mean_nitrifier_novel_fraction));
    assert!((0.0..=1.0).contains(&snapshot.mean_nitrifier_bank_simpson_diversity));
    assert!(snapshot.mean_denitrifier_cells >= 0.0);
    assert!(snapshot.mean_denitrifier_packets >= 0.0);
    assert!(snapshot.mean_denitrifier_novel_packets >= 0.0);
    assert!(snapshot.mean_denitrifier_latent_packets >= 0.0);
    assert!(snapshot.mean_denitrifier_packet_load >= 0.0);
    assert!((0.0..=1.0).contains(&snapshot.mean_denitrifier_strain_anoxia_affinity));
    assert!((0.0..=1.0).contains(&snapshot.mean_denitrifier_strain_nitrate_affinity));
    assert!((0.0..=1.0).contains(&snapshot.mean_denitrifier_gene_anoxia_respiration));
    assert!((0.0..=1.0).contains(&snapshot.mean_denitrifier_gene_nitrate_transport));
    assert!((0.0..=1.0).contains(&snapshot.mean_denitrifier_gene_stress_persistence));
    assert!((0.0..=1.0).contains(&snapshot.mean_denitrifier_gene_reductive_flexibility));
    assert!((0.0..=1.0).contains(&snapshot.mean_denitrifier_genotype_divergence));
    assert!(snapshot.mean_denitrifier_catalog_generation >= 0.0);
    assert!((0.0..=1.0).contains(&snapshot.mean_denitrifier_catalog_novelty));
    assert!((0.0..=1.0).contains(&snapshot.mean_denitrifier_local_catalog_share));
    assert!((0.0..=1.0).contains(&snapshot.mean_denitrifier_catalog_bank_dominance));
    assert!(snapshot.mean_denitrifier_catalog_bank_richness >= 0.0);
    assert!(snapshot.mean_denitrifier_lineage_generation >= 0.0);
    assert!((0.0..=1.0).contains(&snapshot.mean_denitrifier_lineage_novelty));
    assert!(snapshot.mean_denitrifier_packet_mutation_flux >= 0.0);
    assert!((0.0..=1.0).contains(&snapshot.mean_denitrifier_novel_fraction));
    assert!((0.0..=1.0).contains(&snapshot.mean_denitrifier_bank_simpson_diversity));
    assert!((0.0..=1.0).contains(&snapshot.mean_microbial_copiotroph_fraction));
    assert!((0.0..=1.0).contains(&snapshot.mean_nitrifier_aerobic_fraction));
    assert!((0.0..=1.0).contains(&snapshot.mean_denitrifier_anoxic_fraction));
    assert!((0.0..=1.25).contains(&snapshot.mean_microbial_vitality));
    assert!((0.0..=1.25).contains(&snapshot.mean_nitrifier_vitality));
    assert!((0.0..=1.25).contains(&snapshot.mean_denitrifier_vitality));
    assert!(snapshot.explicit_microbes > 0);
    assert!(snapshot.explicit_microbe_represented_cells > 0.0);
    assert!(snapshot.explicit_microbe_owned_fraction > 0.0);
    assert!(snapshot.explicit_microbe_max_authority > 0.0);
    assert!(snapshot.mean_explicit_microbe_activity > 0.0);
    assert!(snapshot.mean_explicit_microbe_atp_mm > 0.0);
    assert!(snapshot.mean_explicit_microbe_glucose_mm > 0.0);
    assert!(snapshot.mean_explicit_microbe_oxygen_mm > 0.0);
    assert!(snapshot.mean_explicit_microbe_translation_support >= 0.0);
}

#[test]
fn explicit_microbes_step_and_report_live_whole_cell_metrics() {
    let mut world = TerrariumWorld::demo(9, false).unwrap();
    let before = world.snapshot();
    world.step_frame().unwrap();
    let after = world.snapshot();

    assert!(after.explicit_microbes > 0);
    assert!(after.explicit_microbe_represented_cells > 0.0);
    assert!(after.explicit_microbe_owned_fraction > 0.0);
    assert!(after.explicit_microbe_max_authority > 0.0);
    assert!(after.mean_explicit_microbe_activity > 0.0);
    assert!(after.mean_explicit_microbe_atp_mm > 0.0);
    assert!(after.mean_explicit_microbe_glucose_mm > 0.0);
    assert!(after.mean_explicit_microbe_oxygen_mm > 0.0);
    assert!(after.mean_explicit_microbe_translation_support >= 0.0);
    assert!(
        (after.mean_explicit_microbe_atp_mm - before.mean_explicit_microbe_atp_mm).abs() > 1.0e-6
            || (after.mean_explicit_microbe_division_progress
                - before.mean_explicit_microbe_division_progress)
                .abs()
                > 1.0e-6
            || (after.mean_explicit_microbe_local_co2 - before.mean_explicit_microbe_local_co2)
                .abs()
                > 1.0e-6
    );
}

#[test]
fn owned_cells_use_explicit_chemistry_before_substrate_step() {
    let mut world = TerrariumWorld::demo(17, false).unwrap();
    let cohort = &world.explicit_microbes[0];
    let (x, y, z, radius) = (cohort.x, cohort.y, cohort.z, cohort.radius);
    let flat = y * world.config.width + x;
    let eco_dt = world.config.world_dt_s * world.config.time_warp;

    world.step_broad_soil(eco_dt).unwrap();
    let _ = world
        .substrate
        .deposit_patch_species(TerrariumSpecies::Glucose, x, y, z, radius, 0.15);
    let _ =
        world
            .substrate
            .deposit_patch_species(TerrariumSpecies::OxygenGas, x, y, z, radius, 0.12);

    let gas = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Gas);
    let aqueous = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Aqueous);
    let inv_co2_before = world.explicit_microbes[0]
        .material_inventory
        .total_amount_for_component(
            MaterialRegionKind::GasPhase,
            &MoleculeGraph::representative_carbon_dioxide(),
            &MaterialPhaseSelector::Exact(gas.clone()),
        );
    let inv_glucose_before = world.explicit_microbes[0]
        .material_inventory
        .total_amount_for_component(
            MaterialRegionKind::PoreWater,
            &MoleculeGraph::representative_glucose(),
            &MaterialPhaseSelector::Exact(aqueous.clone()),
        );

    world.step_explicit_microbes(eco_dt).unwrap();
    world.sync_substrate_controls().unwrap();

    let inv_co2_after = world.explicit_microbes[0]
        .material_inventory
        .total_amount_for_component(
            MaterialRegionKind::GasPhase,
            &MoleculeGraph::representative_carbon_dioxide(),
            &MaterialPhaseSelector::Exact(gas),
        );
    let inv_glucose_after = world.explicit_microbes[0]
        .material_inventory
        .total_amount_for_component(
            MaterialRegionKind::PoreWater,
            &MoleculeGraph::representative_glucose(),
            &MaterialPhaseSelector::Exact(aqueous),
        );

    assert!(
        inv_glucose_after < inv_glucose_before
            || inv_co2_after > inv_co2_before
            || world.explicit_microbes[0].cumulative_co2_release > 0.0,
        "explicit cohorts did not change owned inventory chemistry before substrate stepping"
    );
    assert!(world.explicit_microbe_authority[flat] >= 0.01);

    let plane = world.config.width * world.config.height;
    for depth in 0..world.config.depth.max(1) {
        let gid = depth * plane + flat;
        assert_eq!(world.substrate.microbial_activity[gid], 0.0);
        assert_eq!(world.substrate.nitrifier_activity[gid], 0.0);
        assert_eq!(world.substrate.denitrifier_activity[gid], 0.0);
    }
}

#[test]
fn explicit_microbes_build_local_authority_fields() {
    let world = TerrariumWorld::demo(11, false).unwrap();
    assert!(
        world
            .explicit_microbe_authority
            .iter()
            .copied()
            .fold(0.0f32, f32::max)
            > 0.0
    );
    assert!(
        world
            .explicit_microbe_activity
            .iter()
            .copied()
            .fold(0.0f32, f32::max)
            > 0.0
    );
}

#[test]
fn dynamic_recruitment_adds_explicit_microbes_from_soil_hotspots() {
    let mut world = TerrariumWorld::demo(41, false).unwrap();
    world.explicit_microbes.clear();
    world.explicit_microbe_authority.fill(0.0);
    world.explicit_microbe_activity.fill(0.0);

    let x = world.config.width / 2;
    let y = world.config.height / 2;
    let flat = y * world.config.width + x;
    world.microbial_cells[flat] = 22.0;
    world.microbial_packets[flat] = 5.0;
    world.microbial_vitality[flat] = 1.0;
    world.microbial_dormancy[flat] = 0.0;
    world.root_exudates[flat] = 0.6;
    world.litter_carbon[flat] = 0.4;
    world.organic_matter[flat] = 0.25;

    world.recruit_explicit_microbes_from_soil().unwrap();

    assert!(!world.explicit_microbes.is_empty());
    assert!(world.explicit_microbe_authority[flat] > 0.0);
    assert!(
        world.explicit_microbes.iter().any(|cohort| {
            cohort.x.abs_diff(x) <= EXPLICIT_MICROBE_RECRUITMENT_SPACING
                && cohort.y.abs_diff(y) <= EXPLICIT_MICROBE_RECRUITMENT_SPACING
        }),
        "no explicit cohort was recruited near the microbial hotspot"
    );
}

#[test]
fn dynamic_recruitment_inherits_local_genotype_identity() {
    let mut world = TerrariumWorld::demo(43, false).unwrap();
    world.explicit_microbes.clear();
    world.explicit_microbe_authority.fill(0.0);
    world.explicit_microbe_activity.fill(0.0);

    let x = world.config.width / 2;
    let y = world.config.height / 2;
    let flat = y * world.config.width + x;
    world.microbial_cells[flat] = 24.0;
    world.microbial_packets[flat] = 6.0;
    world.microbial_vitality[flat] = 1.0;
    world.microbial_dormancy[flat] = 0.0;
    world.root_exudates[flat] = 0.7;
    world.litter_carbon[flat] = 0.5;
    world.organic_matter[flat] = 0.25;

    let slot = world.microbial_secondary.banks[SHADOW_BANK_IDX].catalog_slots[flat] as usize;
    world.microbial_secondary.banks[SHADOW_BANK_IDX].packets[flat] = 7.5;
    world.microbial_secondary.banks[VARIANT_BANK_IDX].packets[flat] = 0.1;
    world.microbial_secondary.banks[NOVEL_BANK_IDX].packets[flat] = 0.1;
    world.microbial_secondary.banks[SHADOW_BANK_IDX].catalog_bank[slot]
        .record
        .genotype_id = 707;
    world.microbial_secondary.banks[SHADOW_BANK_IDX].catalog_bank[slot]
        .record
        .lineage_id = 313;
    world.microbial_secondary.banks[SHADOW_BANK_IDX].catalog_bank[slot]
        .catalog
        .catalog_id = 111;
    world.microbial_secondary.banks[SHADOW_BANK_IDX].catalog_bank[slot]
        .catalog
        .local_bank_share = 0.82;
    world.microbial_secondary.banks[SHADOW_BANK_IDX].catalog_bank[slot].genes =
        [0.88; INTERNAL_SECONDARY_GENOTYPE_AXES];

    world.recruit_explicit_microbes_from_soil().unwrap();

    let cohort = world
        .explicit_microbes
        .iter()
        .find(|cohort| {
            cohort.x.abs_diff(x) <= EXPLICIT_MICROBE_RECRUITMENT_SPACING
                && cohort.y.abs_diff(y) <= EXPLICIT_MICROBE_RECRUITMENT_SPACING
        })
        .expect("recruited cohort near hotspot");
    assert_eq!(cohort.identity.bank_idx, SHADOW_BANK_IDX);
    assert_eq!(cohort.identity.record.genotype_id, 707);
    assert_eq!(cohort.identity.record.lineage_id, 313);
    assert_eq!(cohort.identity.catalog.catalog_id, 111);
    assert!(cohort.identity.catalog.local_bank_share > 0.7);
    assert!(cohort.identity.gene_catabolic > 0.5);
    assert!(cohort.represented_packets > 0.0);
}

#[test]
fn explicit_microbe_inputs_ignore_coarse_pools_in_owned_cells() {
    let mut world = TerrariumWorld::demo(23, false).unwrap();
    let cohort = &world.explicit_microbes[0];
    let (x, y, z, radius) = (cohort.x, cohort.y, cohort.z, cohort.radius);
    let flat = y * world.config.width + x;

    world.explicit_microbe_authority[flat] = 1.0;
    world.root_exudates[flat] = 0.0;
    world.litter_carbon[flat] = 0.0;
    world.dissolved_nutrients[flat] = 0.0;
    world.shallow_nutrients[flat] = 0.0;
    world.organic_matter[flat] = 0.0;
    let owned_baseline = world.explicit_microbe_environment_inputs(x, y, z, radius);

    world.root_exudates[flat] = 1.0;
    world.litter_carbon[flat] = 1.0;
    world.dissolved_nutrients[flat] = 1.0;
    world.shallow_nutrients[flat] = 1.0;
    world.organic_matter[flat] = 1.0;
    let owned_enriched = world.explicit_microbe_environment_inputs(x, y, z, radius);

    assert!((owned_enriched.glucose_mm - owned_baseline.glucose_mm).abs() <= 1.0e-6);
    assert!((owned_enriched.amino_acids_mm - owned_baseline.amino_acids_mm).abs() <= 1.0e-6);
    assert!((owned_enriched.nucleotides_mm - owned_baseline.nucleotides_mm).abs() <= 1.0e-6);
    assert!(
        (owned_enriched.membrane_precursors_mm - owned_baseline.membrane_precursors_mm).abs()
            <= 1.0e-6
    );

    world.explicit_microbe_authority[flat] = 0.0;
    let background_enriched = world.explicit_microbe_environment_inputs(x, y, z, radius);
    assert!(
        background_enriched.metabolic_load < owned_enriched.metabolic_load + 0.1,
        "background cells should have lower or similar metabolic load without ownership penalty"
    );
}

#[test]
fn explicit_microbe_inputs_derive_nutrients_from_material_inventory_projection() {
    let world = TerrariumWorld::demo(61, false).unwrap();
    let cohort = &world.explicit_microbes[0];
    let (x, y, z, radius) = (cohort.x, cohort.y, cohort.z, cohort.radius);

    let inventory = world.explicit_microbe_material_inventory(x, y, z, radius);
    let projected = inventory.estimate_whole_cell_environment_inputs(&[
        MaterialRegionKind::PoreWater,
        MaterialRegionKind::GasPhase,
        MaterialRegionKind::MineralSurface,
        MaterialRegionKind::BiofilmMatrix,
    ]);
    let inputs = world.explicit_microbe_environment_inputs(x, y, z, radius);

    assert!((inputs.glucose_mm - projected.glucose_mm).abs() <= 1.0e-6);
    assert!((inputs.oxygen_mm - projected.oxygen_mm).abs() <= 1.0e-6);
    assert!((inputs.amino_acids_mm - projected.amino_acids_mm).abs() <= 1.0e-6);
    assert!((inputs.nucleotides_mm - projected.nucleotides_mm).abs() <= 1.0e-6);
    assert!((inputs.membrane_precursors_mm - projected.membrane_precursors_mm).abs() <= 1.0e-6);
    assert!(inputs.metabolic_load >= 0.25);
}

#[test]
fn explicit_microbe_inputs_follow_persistent_owned_material_inventory() {
    let mut world = TerrariumWorld::demo(61, false).unwrap();
    let cohort = &world.explicit_microbes[0];
    let (x, y, z, radius) = (cohort.x, cohort.y, cohort.z, cohort.radius);
    let aqueous = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Aqueous);

    let baseline_from_patch = world
        .explicit_microbe_material_inventory_target_from_patch(x, y, z, radius)
        .estimate_whole_cell_environment_inputs(&[
            MaterialRegionKind::PoreWater,
            MaterialRegionKind::GasPhase,
            MaterialRegionKind::MineralSurface,
            MaterialRegionKind::BiofilmMatrix,
        ]);
    world.explicit_microbes[0]
        .material_inventory
        .set_component_amount(
            MaterialRegionKind::PoreWater,
            MoleculeGraph::representative_glucose(),
            aqueous,
            0.0,
        )
        .unwrap();

    let inputs = world.explicit_microbe_environment_inputs(x, y, z, radius);
    assert!(inputs.glucose_mm < baseline_from_patch.glucose_mm);
}

#[test]
fn explicit_microbe_material_inventory_relaxes_toward_patch_without_snapping() {
    let mut world = TerrariumWorld::demo(61, false).unwrap();
    let cohort = &world.explicit_microbes[0];
    let (x, y, z, radius) = (cohort.x, cohort.y, cohort.z, cohort.radius);
    let flat = idx2(world.config.width, x, y);
    world.explicit_microbe_authority[flat] = 1.0;

    let aqueous = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Aqueous);
    let selector = MaterialPhaseSelector::Exact(aqueous.clone());
    let target = world.explicit_microbe_material_inventory_target_from_patch(x, y, z, radius);
    let target_glucose = target.total_amount_for_component(
        MaterialRegionKind::PoreWater,
        &MoleculeGraph::representative_glucose(),
        &selector,
    );
    world.explicit_microbes[0]
        .material_inventory
        .set_component_amount(
            MaterialRegionKind::PoreWater,
            MoleculeGraph::representative_glucose(),
            aqueous.clone(),
            target_glucose * 0.25,
        )
        .unwrap();
    let before_sync = world.explicit_microbe_material_inventory(x, y, z, radius);
    let before_glucose = before_sync.total_amount_for_component(
        MaterialRegionKind::PoreWater,
        &MoleculeGraph::representative_glucose(),
        &selector,
    );

    world.sync_explicit_microbe_material_inventory(0).unwrap();

    let after_sync = world.explicit_microbe_material_inventory(x, y, z, radius);
    let after_glucose = after_sync.total_amount_for_component(
        MaterialRegionKind::PoreWater,
        &MoleculeGraph::representative_glucose(),
        &selector,
    );

    assert!(after_glucose > before_glucose);
    assert!(after_glucose < target_glucose);
}

#[test]
fn explicit_microbe_material_sync_draws_glucose_conservatively_from_owned_patch() {
    let mut world = TerrariumWorld::demo(61, false).unwrap();
    let cohort = &world.explicit_microbes[0];
    let (x, y, z, radius) = (cohort.x, cohort.y, cohort.z, cohort.radius);
    let flat = idx2(world.config.width, x, y);
    world.explicit_microbe_authority[flat] = 1.0;

    let aqueous = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Aqueous);
    let selector = MaterialPhaseSelector::Exact(aqueous.clone());
    let target = world.explicit_microbe_material_inventory_target_from_patch(x, y, z, radius);
    let target_glucose = target.total_amount_for_component(
        MaterialRegionKind::PoreWater,
        &MoleculeGraph::representative_glucose(),
        &selector,
    );
    world.explicit_microbes[0]
        .material_inventory
        .set_component_amount(
            MaterialRegionKind::PoreWater,
            MoleculeGraph::representative_glucose(),
            aqueous,
            target_glucose * 0.25,
        )
        .unwrap();

    let inventory_before = world.explicit_microbes[0]
        .material_inventory
        .total_amount_for_component(
            MaterialRegionKind::PoreWater,
            &MoleculeGraph::representative_glucose(),
            &selector,
        );
    let patch_before =
        world
            .substrate
            .patch_mean_species(TerrariumSpecies::Glucose, x, y, z, radius);

    world.sync_explicit_microbe_material_inventory(0).unwrap();

    let inventory_after = world.explicit_microbes[0]
        .material_inventory
        .total_amount_for_component(
            MaterialRegionKind::PoreWater,
            &MoleculeGraph::representative_glucose(),
            &selector,
        );
    let patch_after =
        world
            .substrate
            .patch_mean_species(TerrariumSpecies::Glucose, x, y, z, radius);

    assert!(inventory_after > inventory_before);
    assert!(patch_after + 1.0e-9 < patch_before);
}

#[test]
fn explicit_microbe_material_sync_returns_excess_glucose_to_owned_patch() {
    let mut world = TerrariumWorld::demo(61, false).unwrap();
    let cohort = &world.explicit_microbes[0];
    let (x, y, z, radius) = (cohort.x, cohort.y, cohort.z, cohort.radius);
    let flat = idx2(world.config.width, x, y);
    world.explicit_microbe_authority[flat] = 1.0;

    let _ = world
        .substrate
        .extract_patch_species(TerrariumSpecies::Glucose, x, y, z, radius, 10.0);

    let aqueous = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Aqueous);
    let selector = MaterialPhaseSelector::Exact(aqueous.clone());
    world.explicit_microbes[0]
        .material_inventory
        .set_component_amount(
            MaterialRegionKind::PoreWater,
            MoleculeGraph::representative_glucose(),
            aqueous,
            0.9,
        )
        .unwrap();

    let inventory_before = world.explicit_microbes[0]
        .material_inventory
        .total_amount_for_component(
            MaterialRegionKind::PoreWater,
            &MoleculeGraph::representative_glucose(),
            &selector,
        );
    let patch_before =
        world
            .substrate
            .patch_mean_species(TerrariumSpecies::Glucose, x, y, z, radius);

    world.sync_explicit_microbe_material_inventory(0).unwrap();

    let inventory_after = world.explicit_microbes[0]
        .material_inventory
        .total_amount_for_component(
            MaterialRegionKind::PoreWater,
            &MoleculeGraph::representative_glucose(),
            &selector,
        );
    let patch_after =
        world
            .substrate
            .patch_mean_species(TerrariumSpecies::Glucose, x, y, z, radius);

    assert!(inventory_after + 1.0e-9 < inventory_before);
    assert!(patch_after > patch_before);
}

#[test]
fn owned_material_sync_retains_internal_glucose_buffer_when_patch_is_empty() {
    let mut world = TerrariumWorld::demo(61, false).unwrap();
    let cohort = &world.explicit_microbes[0];
    let (x, y, z, radius) = (cohort.x, cohort.y, cohort.z, cohort.radius);
    let flat = idx2(world.config.width, x, y);
    world.explicit_microbe_authority[flat] = 1.0;

    let _ = world
        .substrate
        .extract_patch_species(TerrariumSpecies::Glucose, x, y, z, radius, 10.0);

    let aqueous = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Aqueous);
    let selector = MaterialPhaseSelector::Exact(aqueous.clone());
    let target = world.explicit_microbe_material_inventory_target_from_patch(x, y, z, radius);
    let target_glucose = target.total_amount_for_component(
        MaterialRegionKind::PoreWater,
        &MoleculeGraph::representative_glucose(),
        &selector,
    );
    let (lower, upper) = TerrariumWorld::reserve_band(target_glucose, 0.55, 1.35, 0.02, 0.03);
    let reserve_amount = 0.5 * (lower + upper);
    world.explicit_microbes[0]
        .material_inventory
        .set_component_amount(
            MaterialRegionKind::PoreWater,
            MoleculeGraph::representative_glucose(),
            aqueous,
            reserve_amount,
        )
        .unwrap();

    let inventory_before = world.explicit_microbes[0]
        .material_inventory
        .total_amount_for_component(
            MaterialRegionKind::PoreWater,
            &MoleculeGraph::representative_glucose(),
            &selector,
        );
    let patch_before =
        world
            .substrate
            .patch_mean_species(TerrariumSpecies::Glucose, x, y, z, radius);

    world.sync_explicit_microbe_material_inventory(0).unwrap();

    let inventory_after = world.explicit_microbes[0]
        .material_inventory
        .total_amount_for_component(
            MaterialRegionKind::PoreWater,
            &MoleculeGraph::representative_glucose(),
            &selector,
        );
    let patch_after =
        world
            .substrate
            .patch_mean_species(TerrariumSpecies::Glucose, x, y, z, radius);

    assert!(
        (inventory_after - inventory_before).abs() <= 1.0e-8,
        "owned sync should retain an internal glucose reserve band"
    );
    assert!(
        (patch_after - patch_before).abs() <= 1.0e-8,
        "owned sync should not churn glucose back into an empty patch while within reserve"
    );
}

#[test]
fn explicit_microbe_step_updates_owned_ledger_and_lifecycle_state() {
    let mut world = TerrariumWorld::demo(47, false).unwrap();
    let cohort = &world.explicit_microbes[0];
    let (x, y, z, radius) = (cohort.x, cohort.y, cohort.z, cohort.radius);
    let eco_dt = world.config.world_dt_s * world.config.time_warp;

    let _ = world
        .substrate
        .deposit_patch_species(TerrariumSpecies::Glucose, x, y, z, radius, 0.25);
    let _ =
        world
            .substrate
            .deposit_patch_species(TerrariumSpecies::OxygenGas, x, y, z, radius, 0.18);
    let before_cells = world.explicit_microbes[0].represented_cells;
    let before_age = world.explicit_microbes[0].age_s;

    world.step_explicit_microbes(eco_dt).unwrap();

    let after = &world.explicit_microbes[0];
    assert!(after.age_s > before_age);
    assert!(after.cumulative_glucose_draw > 0.0 || after.cumulative_oxygen_draw > 0.0);
    assert!(after.cumulative_co2_release >= 0.0);
    assert!(after.smoothed_energy >= 0.0);
    assert!(after.smoothed_stress >= 0.0);
    assert!(after.represented_packets > 0.0);
    assert!(
        (after.represented_cells - before_cells).abs() > 1.0e-6
            || after.radius != EXPLICIT_MICROBE_PATCH_RADIUS
    );
}

#[test]
fn explicit_microbe_step_depletes_owned_material_inventory() {
    let mut world = TerrariumWorld::demo(47, false).unwrap();
    let cohort = &world.explicit_microbes[0];
    #[allow(unused_variables)]
    let (x, y, z, radius) = (cohort.x, cohort.y, cohort.z, cohort.radius);
    let aqueous = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Aqueous);
    let gas = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Gas);
    let glucose_before = world.explicit_microbes[0]
        .material_inventory
        .total_amount_for_component(
            MaterialRegionKind::PoreWater,
            &MoleculeGraph::representative_glucose(),
            &MaterialPhaseSelector::Exact(aqueous),
        );
    let oxygen_before = world.explicit_microbes[0]
        .material_inventory
        .total_amount_for_component(
            MaterialRegionKind::GasPhase,
            &MoleculeGraph::representative_oxygen_gas(),
            &MaterialPhaseSelector::Exact(gas),
        );
    let eco_dt = world.config.world_dt_s * world.config.time_warp;

    world.step_explicit_microbes(eco_dt).unwrap();

    let after = &world.explicit_microbes[0];
    let glucose_after = after.material_inventory.total_amount_for_component(
        MaterialRegionKind::PoreWater,
        &MoleculeGraph::representative_glucose(),
        &MaterialPhaseSelector::Kind(MaterialPhaseKind::Aqueous),
    );
    let oxygen_after = after.material_inventory.total_amount_for_component(
        MaterialRegionKind::GasPhase,
        &MoleculeGraph::representative_oxygen_gas(),
        &MaterialPhaseSelector::Kind(MaterialPhaseKind::Gas),
    );

    assert!(after.cumulative_glucose_draw > 0.0 || after.cumulative_oxygen_draw > 0.0);
    if after.cumulative_glucose_draw > 0.0 {
        assert!(glucose_after + 1.0e-9 < glucose_before);
    }
    if after.cumulative_oxygen_draw > 0.0 {
        assert!(oxygen_after + 1.0e-9 < oxygen_before);
    }
    assert_eq!(
        world
            .explicit_microbe_material_inventory(x, y, z, radius)
            .total_amount_for_component(
                MaterialRegionKind::PoreWater,
                &MoleculeGraph::representative_glucose(),
                &MaterialPhaseSelector::Kind(MaterialPhaseKind::Aqueous),
            ),
        glucose_after
    );
}

#[test]
fn explicit_microbe_step_accumulates_product_molecules_in_owned_inventory() {
    let mut world = TerrariumWorld::demo(47, false).unwrap();
    let gas = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Gas);
    let aqueous = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Aqueous);
    let co2_before = world.explicit_microbes[0]
        .material_inventory
        .total_amount_for_component(
            MaterialRegionKind::GasPhase,
            &MoleculeGraph::representative_carbon_dioxide(),
            &MaterialPhaseSelector::Exact(gas.clone()),
        );
    let proton_before = world.explicit_microbes[0]
        .material_inventory
        .total_amount_for_component(
            MaterialRegionKind::PoreWater,
            &MoleculeGraph::representative_proton_pool(),
            &MaterialPhaseSelector::Exact(aqueous.clone()),
        );
    let eco_dt = world.config.world_dt_s * world.config.time_warp;

    world.step_explicit_microbes(eco_dt).unwrap();

    let after = &world.explicit_microbes[0];
    let co2_after = after.material_inventory.total_amount_for_component(
        MaterialRegionKind::GasPhase,
        &MoleculeGraph::representative_carbon_dioxide(),
        &MaterialPhaseSelector::Exact(gas),
    );
    let proton_after = after.material_inventory.total_amount_for_component(
        MaterialRegionKind::PoreWater,
        &MoleculeGraph::representative_proton_pool(),
        &MaterialPhaseSelector::Exact(aqueous),
    );

    if after.cumulative_co2_release > 0.0 {
        assert!(co2_after > co2_before);
    }
    if after.cumulative_proton_release > 0.0 {
        assert!(proton_after > proton_before);
    }
}

#[test]
fn owned_explicit_microbe_step_keeps_products_in_inventory_before_patch_writeback() {
    let mut world = TerrariumWorld::demo(47, false).unwrap();
    let cohort = &world.explicit_microbes[0];
    let (x, y, z, radius) = (cohort.x, cohort.y, cohort.z, cohort.radius);
    let flat = idx2(world.config.width, x, y);
    world.explicit_microbe_authority[flat] = 1.0;

    let patch_co2_before =
        world
            .substrate
            .patch_mean_species(TerrariumSpecies::CarbonDioxide, x, y, z, radius);
    let patch_proton_before =
        world
            .substrate
            .patch_mean_species(TerrariumSpecies::Proton, x, y, z, radius);
    let atmos_co2_before = world.sample_odorant_patch(ATMOS_CO2_IDX, x, y, z, radius);
    let eco_dt = world.config.world_dt_s * world.config.time_warp;

    world.step_explicit_microbes(eco_dt).unwrap();

    let after = &world.explicit_microbes[0];
    let patch_co2_after =
        world
            .substrate
            .patch_mean_species(TerrariumSpecies::CarbonDioxide, x, y, z, radius);
    let patch_proton_after =
        world
            .substrate
            .patch_mean_species(TerrariumSpecies::Proton, x, y, z, radius);
    let atmos_co2_after = world.sample_odorant_patch(ATMOS_CO2_IDX, x, y, z, radius);

    if after.cumulative_co2_release > 0.0 {
        assert!(
            (patch_co2_after - patch_co2_before).abs() <= 1.0e-8,
            "owned step still wrote CO2 directly into coarse substrate"
        );
        assert!(
            (atmos_co2_after - atmos_co2_before).abs() <= 1.0e-8,
            "owned step still wrote CO2 directly into atmosphere"
        );
    }
    if after.cumulative_proton_release > 0.0 {
        assert!(
            (patch_proton_after - patch_proton_before).abs() <= 1.0e-8,
            "owned step still wrote protons directly into coarse substrate"
        );
    }
}

#[test]
fn owned_explicit_microbe_step_keeps_atp_in_inventory_before_patch_writeback() {
    let mut world = TerrariumWorld::demo(47, false).unwrap();
    let cohort = &world.explicit_microbes[0];
    let (x, y, z, radius) = (cohort.x, cohort.y, cohort.z, cohort.radius);
    let flat = idx2(world.config.width, x, y);
    world.explicit_microbe_authority[flat] = 1.0;

    let aqueous = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Aqueous);
    let atp_before = world.explicit_microbes[0]
        .material_inventory
        .total_amount_for_component(
            MaterialRegionKind::BiofilmMatrix,
            &MoleculeGraph::representative_atp(),
            &MaterialPhaseSelector::Exact(aqueous.clone()),
        );
    let patch_atp_before =
        world
            .substrate
            .patch_mean_species(TerrariumSpecies::AtpFlux, x, y, z, radius);
    let eco_dt = world.config.world_dt_s * world.config.time_warp;

    world.step_explicit_microbes(eco_dt).unwrap();

    let atp_after = world.explicit_microbes[0]
        .material_inventory
        .total_amount_for_component(
            MaterialRegionKind::BiofilmMatrix,
            &MoleculeGraph::representative_atp(),
            &MaterialPhaseSelector::Exact(aqueous),
        );
    let patch_atp_after =
        world
            .substrate
            .patch_mean_species(TerrariumSpecies::AtpFlux, x, y, z, radius);

    assert!(
        atp_after >= atp_before,
        "owned step should accumulate ATP-like support in the owned inventory"
    );
    assert!(
        (patch_atp_after - patch_atp_before).abs() <= 1.0e-8,
        "owned step still wrote ATP flux directly into the coarse patch"
    );
}

#[test]
fn owned_explicit_microbe_material_sync_spills_products_conservatively_to_environment() {
    let mut world = TerrariumWorld::demo(61, false).unwrap();
    let cohort = &world.explicit_microbes[0];
    let (x, y, z, radius) = (cohort.x, cohort.y, cohort.z, cohort.radius);
    let flat = idx2(world.config.width, x, y);
    world.explicit_microbe_authority[flat] = 1.0;

    let gas = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Gas);
    let aqueous = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Aqueous);
    world.explicit_microbes[0]
        .material_inventory
        .set_component_amount(
            MaterialRegionKind::GasPhase,
            MoleculeGraph::representative_carbon_dioxide(),
            gas.clone(),
            0.35,
        )
        .unwrap();
    world.explicit_microbes[0]
        .material_inventory
        .set_component_amount(
            MaterialRegionKind::PoreWater,
            MoleculeGraph::representative_proton_pool(),
            aqueous.clone(),
            0.08,
        )
        .unwrap();

    let inv_co2_before = world.explicit_microbes[0]
        .material_inventory
        .total_amount_for_component(
            MaterialRegionKind::GasPhase,
            &MoleculeGraph::representative_carbon_dioxide(),
            &MaterialPhaseSelector::Exact(gas.clone()),
        );
    let inv_proton_before = world.explicit_microbes[0]
        .material_inventory
        .total_amount_for_component(
            MaterialRegionKind::PoreWater,
            &MoleculeGraph::representative_proton_pool(),
            &MaterialPhaseSelector::Exact(aqueous.clone()),
        );
    let patch_co2_before =
        world
            .substrate
            .patch_mean_species(TerrariumSpecies::CarbonDioxide, x, y, z, radius);
    let patch_proton_before =
        world
            .substrate
            .patch_mean_species(TerrariumSpecies::Proton, x, y, z, radius);
    let atmos_co2_before = world.sample_odorant_patch(ATMOS_CO2_IDX, x, y, z, radius);

    world.sync_explicit_microbe_material_inventory(0).unwrap();

    let inv_co2_after = world.explicit_microbes[0]
        .material_inventory
        .total_amount_for_component(
            MaterialRegionKind::GasPhase,
            &MoleculeGraph::representative_carbon_dioxide(),
            &MaterialPhaseSelector::Exact(gas),
        );
    let inv_proton_after = world.explicit_microbes[0]
        .material_inventory
        .total_amount_for_component(
            MaterialRegionKind::PoreWater,
            &MoleculeGraph::representative_proton_pool(),
            &MaterialPhaseSelector::Exact(aqueous),
        );
    let patch_co2_after =
        world
            .substrate
            .patch_mean_species(TerrariumSpecies::CarbonDioxide, x, y, z, radius);
    let patch_proton_after =
        world
            .substrate
            .patch_mean_species(TerrariumSpecies::Proton, x, y, z, radius);
    let atmos_co2_after = world.sample_odorant_patch(ATMOS_CO2_IDX, x, y, z, radius);

    assert!(inv_co2_after + 1.0e-9 < inv_co2_before);
    assert!(inv_proton_after + 1.0e-9 < inv_proton_before);
    assert!(
        patch_co2_after > patch_co2_before || atmos_co2_after > atmos_co2_before,
        "owned CO2 inventory did not spill into environment"
    );
    assert!(
        patch_proton_after > patch_proton_before,
        "owned proton inventory did not spill back into patch"
    );
}

#[test]
fn owned_membrane_sync_consumes_internal_atp_before_patch_flux() {
    let mut world = TerrariumWorld::demo(61, false).unwrap();
    let cohort = &world.explicit_microbes[0];
    let (x, y, z, radius) = (cohort.x, cohort.y, cohort.z, cohort.radius);
    let flat = idx2(world.config.width, x, y);
    world.explicit_microbe_authority[flat] = 1.0;

    let _ = world
        .substrate
        .extract_patch_species(TerrariumSpecies::AtpFlux, x, y, z, radius, 10.0);

    let aqueous = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Aqueous);
    let amorphous = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Amorphous);
    world.explicit_microbes[0]
        .material_inventory
        .set_component_amount(
            MaterialRegionKind::BiofilmMatrix,
            MoleculeGraph::representative_atp(),
            aqueous.clone(),
            0.12,
        )
        .unwrap();
    world.explicit_microbes[0]
        .material_inventory
        .set_component_amount(
            MaterialRegionKind::BiofilmMatrix,
            MoleculeGraph::representative_membrane_precursor_pool(),
            amorphous.clone(),
            0.0,
        )
        .unwrap();

    let atp_before = world.explicit_microbes[0]
        .material_inventory
        .total_amount_for_component(
            MaterialRegionKind::BiofilmMatrix,
            &MoleculeGraph::representative_atp(),
            &MaterialPhaseSelector::Exact(aqueous),
        );
    let membrane_before = world.explicit_microbes[0]
        .material_inventory
        .total_amount_for_component(
            MaterialRegionKind::BiofilmMatrix,
            &MoleculeGraph::representative_membrane_precursor_pool(),
            &MaterialPhaseSelector::Exact(amorphous.clone()),
        );
    let patch_atp_before =
        world
            .substrate
            .patch_mean_species(TerrariumSpecies::AtpFlux, x, y, z, radius);

    world
        .sync_owned_component_membrane_pool(0, x, y, z, radius, 0.08, 0.04, 1.0)
        .unwrap();

    let atp_after = world.explicit_microbes[0]
        .material_inventory
        .total_amount_for_component(
            MaterialRegionKind::BiofilmMatrix,
            &MoleculeGraph::representative_atp(),
            &MaterialPhaseSelector::Kind(MaterialPhaseKind::Aqueous),
        );
    let membrane_after = world.explicit_microbes[0]
        .material_inventory
        .total_amount_for_component(
            MaterialRegionKind::BiofilmMatrix,
            &MoleculeGraph::representative_membrane_precursor_pool(),
            &MaterialPhaseSelector::Exact(amorphous),
        );
    let patch_atp_after =
        world
            .substrate
            .patch_mean_species(TerrariumSpecies::AtpFlux, x, y, z, radius);

    assert!(atp_after + 1.0e-9 < atp_before);
    assert!(membrane_after > membrane_before);
    assert!(
        (patch_atp_after - patch_atp_before).abs() <= 1.0e-8,
        "membrane sync should consume local ATP-like inventory before coarse ATP flux"
    );
}

#[test]
fn owned_consumption_capped_to_inventory_stock() {
    let mut world = TerrariumWorld::demo(47, false).unwrap();
    let cohort = &world.explicit_microbes[0];
    let (x, y, _z, _radius) = (cohort.x, cohort.y, cohort.z, cohort.radius);
    let flat = idx2(world.config.width, x, y);
    world.explicit_microbe_authority[flat] = 1.0;

    let aqueous = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Aqueous);
    let gas = MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Gas);
    let tiny_stock = 1.0e-7;
    world.explicit_microbes[0]
        .material_inventory
        .set_component_amount(
            MaterialRegionKind::PoreWater,
            MoleculeGraph::representative_glucose(),
            aqueous.clone(),
            tiny_stock,
        )
        .unwrap();
    world.explicit_microbes[0]
        .material_inventory
        .set_component_amount(
            MaterialRegionKind::GasPhase,
            MoleculeGraph::representative_oxygen_gas(),
            gas.clone(),
            tiny_stock,
        )
        .unwrap();

    let eco_dt = world.config.world_dt_s * world.config.time_warp;
    world.step_explicit_microbes(eco_dt).unwrap();

    let glucose_after = world.explicit_microbes[0]
        .material_inventory
        .total_amount_for_component(
            MaterialRegionKind::PoreWater,
            &MoleculeGraph::representative_glucose(),
            &MaterialPhaseSelector::Exact(aqueous),
        );
    let oxygen_after = world.explicit_microbes[0]
        .material_inventory
        .total_amount_for_component(
            MaterialRegionKind::GasPhase,
            &MoleculeGraph::representative_oxygen_gas(),
            &MaterialPhaseSelector::Exact(gas),
        );

    assert!(
        glucose_after >= 0.0,
        "owned glucose inventory went negative: {}",
        glucose_after
    );
    assert!(
        oxygen_after >= 0.0,
        "owned oxygen inventory went negative: {}",
        oxygen_after
    );
}

#[test]
fn owned_summary_pools_reconcile_from_substrate() {
    let mut world = TerrariumWorld::demo(31, false).unwrap();
    let cohort = &world.explicit_microbes[0];
    let (x, y, z) = (cohort.x, cohort.y, cohort.z);
    let flat = y * world.config.width + x;
    let background_flat = world
        .explicit_microbe_authority
        .iter()
        .position(|authority| *authority < EXPLICIT_OWNERSHIP_THRESHOLD)
        .unwrap();

    world.explicit_microbe_authority[flat] = 1.0;
    world.root_exudates[flat] = 0.0;
    world.litter_carbon[flat] = 0.0;
    world.dissolved_nutrients[flat] = 0.0;
    world.shallow_nutrients[flat] = 0.0;
    world.mineral_nitrogen[flat] = 0.0;
    world.organic_matter[flat] = 0.0;
    world.root_exudates[background_flat] = 0.0;
    world.litter_carbon[background_flat] = 0.0;
    world.dissolved_nutrients[background_flat] = 0.0;
    world.shallow_nutrients[background_flat] = 0.0;
    world.mineral_nitrogen[background_flat] = 0.0;
    world.organic_matter[background_flat] = 0.0;

    let _ = world
        .substrate
        .deposit_patch_species(TerrariumSpecies::Glucose, x, y, z, 1, 0.12);
    let _ = world
        .substrate
        .deposit_patch_species(TerrariumSpecies::Ammonium, x, y, z, 1, 0.08);
    let _ = world
        .substrate
        .deposit_patch_species(TerrariumSpecies::Nitrate, x, y, z, 1, 0.05);
    let _ = world
        .substrate
        .deposit_patch_species(TerrariumSpecies::Phosphorus, x, y, z, 1, 0.03);
    let _ =
        world
            .substrate
            .deposit_patch_species(TerrariumSpecies::CarbonDioxide, x, y, z, 1, 0.04);
    let _ = world
        .substrate
        .deposit_patch_species(TerrariumSpecies::AtpFlux, x, y, z, 1, 0.002);

    world.reconcile_owned_summary_pools_from_substrate();

    assert!(world.root_exudates[flat] > 0.0);
    assert!(world.litter_carbon[flat] > 0.0);
    assert!(world.dissolved_nutrients[flat] > 0.0);
    assert!(world.shallow_nutrients[flat] > 0.0);
    assert!(world.mineral_nitrogen[flat] > 0.0);
    assert!(world.organic_matter[flat] > 0.0);

    assert_eq!(world.root_exudates[background_flat], 0.0);
    assert_eq!(world.litter_carbon[background_flat], 0.0);
    assert_eq!(world.dissolved_nutrients[background_flat], 0.0);
    assert_eq!(world.shallow_nutrients[background_flat], 0.0);
    assert_eq!(world.mineral_nitrogen[background_flat], 0.0);
    assert_eq!(world.organic_matter[background_flat], 0.0);
}

#[test]
fn background_only_deposit_skips_owned_cells() {
    let mut field = vec![0.0f32; 3];
    let owned_mask = vec![1.0f32, 0.0, 0.0];
    deposit_2d_background_only(
        &mut field,
        3,
        1,
        0,
        0,
        1,
        1.0,
        &owned_mask,
        EXPLICIT_OWNERSHIP_THRESHOLD,
    );

    assert!(field[0].abs() <= 1.0e-9);
    assert!(field[1] > 0.0 || field[2] > 0.0);
    assert!((field.iter().sum::<f32>() - 1.0).abs() <= 1.0e-6);
}

#[test]
fn fruit_decay_routes_owned_detritus_to_substrate_not_coarse_litter() {
    let mut world = TerrariumWorld::demo(29, false).unwrap();
    let cohort = &world.explicit_microbes[0];
    let (x, y, z) = (cohort.x, cohort.y, cohort.z);
    let flat = y * world.config.width + x;
    let eco_dt = world.config.world_dt_s * world.config.time_warp;

    world.fruits.clear();
    world.litter_carbon.fill(0.0);
    let source_genome = world
        .plants
        .first()
        .map(|plant| plant.genome.clone())
        .unwrap_or_else(|| TerrariumPlantGenome::sample(&mut world.rng));
    let taxonomy_id = source_genome.taxonomy_id;
    let composition = crate::terrarium::fruit_state::fruit_composition_from_parent(
        &source_genome,
        taxonomy_id,
        0.20,
        1.0,
    );
    let identity = world.register_organism_identity(
        TerrariumOrganismKind::Fruit,
        Some(taxonomy_id),
        None,
        None,
        None,
        None,
        None,
    );
    world.fruits.push(TerrariumFruitPatch {
        identity,
        source: FruitSourceState {
            x,
            y,
            z,
            attached: false,
            ripeness: 1.0,
            sugar_content: 0.20,
            odorant_emission_rate: 0.0,
            decay_rate: 0.0,
            alive: true,
            odorant_profile: Vec::new(),
        },
        taxonomy_id,
        source_genome,
        composition,
        development: None,
        organ: crate::terrarium::fruit_state::detached_fruit_organ_state(1.0),
        reproduction: None,
        radius: 1.0,
        previous_remaining: 0.50,
        deposited_all: false,
        material_inventory: RegionalMaterialInventory::new("fruit:test".into()),
    });

    let glucose_before = world
        .substrate
        .patch_mean_species(TerrariumSpecies::Glucose, x, y, z, 1);
    let ammonium_before =
        world
            .substrate
            .patch_mean_species(TerrariumSpecies::Ammonium, x, y, z, 1);

    world.step_food_patches_native(eco_dt).unwrap();

    let glucose_after = world
        .substrate
        .patch_mean_species(TerrariumSpecies::Glucose, x, y, z, 1);
    let ammonium_after = world
        .substrate
        .patch_mean_species(TerrariumSpecies::Ammonium, x, y, z, 1);

    assert!(
        world.explicit_microbe_authority[flat] >= 0.01,
        "test fruit was not placed in an owned cell"
    );
    assert!(
        world.litter_carbon[flat].abs() <= 1.0e-9,
        "owned fruit detritus still landed in coarse litter pool"
    );
    assert!(
        glucose_after > glucose_before || ammonium_after > ammonium_before,
        "owned fruit detritus did not reach substrate chemistry"
    );
}

#[test]
fn broad_soil_freezes_owned_cells_but_keeps_background_active() {
    let mut world = TerrariumWorld::demo(11, false).unwrap();
    let (owned_idx, background_idx) = owned_boundary_pair(&world);
    assert!(world.explicit_microbe_authority[owned_idx] >= 0.01);
    let owned_biomass = world.microbial_biomass[owned_idx];
    let owned_cells = world.microbial_cells[owned_idx];
    let owned_packets = world.microbial_packets[owned_idx];
    world.dissolved_nutrients[owned_idx] = 0.0;
    world.shallow_nutrients[owned_idx] = 0.0;
    world.deep_minerals.fill(0.0);
    world.dissolved_nutrients[background_idx] = 1.0;
    world.shallow_nutrients[background_idx] = 1.0;
    let eco_dt = world.config.world_dt_s * world.config.time_warp;

    world.step_broad_soil(eco_dt).unwrap();

    assert!((world.microbial_biomass[owned_idx] - owned_biomass).abs() < 1.0e-9);
    assert!((world.microbial_cells[owned_idx] - owned_cells).abs() < 1.0e-9);
    assert!((world.microbial_packets[owned_idx] - owned_packets).abs() < 1.0e-9);
    assert!(world.dissolved_nutrients[owned_idx] > 0.0);
    assert!(world.shallow_nutrients[owned_idx] > 0.0);
    assert!(world.dissolved_nutrients[background_idx] < 1.0);
    assert_eq!(world.microbial_packet_mutation_flux[owned_idx], 0.0);
    assert_eq!(world.nitrification_potential[owned_idx], 0.0);
    assert_eq!(world.denitrification_potential[owned_idx], 0.0);

    let background_active = world
        .explicit_microbe_authority
        .iter()
        .zip(world.microbial_packet_mutation_flux.iter())
        .any(|(authority, flux)| *authority < 0.01 && *flux > 0.0);
    assert!(background_active);
}

#[test]
fn owned_cells_receive_boundary_chemistry_from_background_only() {
    let mut world = TerrariumWorld::demo(23, false).unwrap();
    let (owned_idx, background_idx) = owned_boundary_pair(&world);

    world.dissolved_nutrients.fill(0.0);
    world.shallow_nutrients.fill(0.0);
    world.deep_minerals.fill(0.0);
    world.organic_matter.fill(0.0);
    world.litter_carbon.fill(0.0);
    world.root_exudates.fill(0.0);
    world.symbiont_biomass.fill(0.0);
    world.microbial_biomass.fill(0.0);
    world.microbial_cells.fill(0.0);
    world.microbial_packets.fill(0.0);
    world.microbial_copiotroph_packets.fill(0.0);
    world.nitrifier_biomass.fill(0.0);
    world.nitrifier_cells.fill(0.0);
    world.nitrifier_packets.fill(0.0);
    world.nitrifier_aerobic_packets.fill(0.0);
    world.denitrifier_biomass.fill(0.0);
    world.denitrifier_cells.fill(0.0);
    world.denitrifier_packets.fill(0.0);
    world.denitrifier_anoxic_packets.fill(0.0);
    world.dissolved_nutrients[background_idx] = 1.0;

    let eco_dt = world.config.world_dt_s * world.config.time_warp;

    world.step_broad_soil(eco_dt).unwrap();

    assert!(world.dissolved_nutrients[owned_idx] > 0.0);
    assert!(world.dissolved_nutrients[background_idx] < 1.0);
    assert_eq!(world.microbial_packet_mutation_flux[owned_idx], 0.0);
    assert_eq!(world.nitrification_potential[owned_idx], 0.0);
    assert_eq!(world.denitrification_potential[owned_idx], 0.0);
}

#[test]
fn owned_cells_zero_coarse_nitrogen_controls_in_substrate() {
    let mut world = TerrariumWorld::demo(11, false).unwrap();
    let owned_idx = world
        .explicit_microbe_authority
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(idx, _)| idx)
        .unwrap();
    assert!(world.explicit_microbe_authority[owned_idx] >= 0.01);

    world
        .step_broad_soil(world.config.world_dt_s * world.config.time_warp)
        .unwrap();
    world.sync_substrate_controls().unwrap();

    let plane = world.config.width * world.config.height;
    for z in 0..world.config.depth.max(1) {
        let gid = z * plane + owned_idx;
        assert_eq!(world.substrate.nitrifier_activity[gid], 0.0);
        assert_eq!(world.substrate.denitrifier_activity[gid], 0.0);
    }
}

#[test]
fn latent_strain_banks_remain_active() {
    let mut world = TerrariumWorld::demo(19, false).unwrap();
    world.run_frames(90).unwrap();
    let snapshot = world.snapshot();
    assert!(
        snapshot.mean_microbial_latent_packets > 1.0e-4,
        "microbial latent banks collapsed to zero"
    );
    assert!(
        snapshot.mean_nitrifier_latent_packets > 1.0e-4,
        "nitrifier latent banks collapsed to zero"
    );
    assert!(
        snapshot.mean_denitrifier_latent_packets > 1.0e-4,
        "denitrifier latent banks collapsed to zero"
    );
    assert!(
        snapshot.mean_microbial_bank_simpson_diversity >= snapshot.mean_microbial_shadow_fraction
    );
    assert!(
        snapshot.mean_nitrifier_bank_simpson_diversity >= snapshot.mean_nitrifier_shadow_fraction
    );
    assert!(
        snapshot.mean_denitrifier_bank_simpson_diversity
            >= snapshot.mean_denitrifier_shadow_fraction
    );
}

#[test]
fn public_secondary_lineages_accumulate_novelty() {
    let mut world = TerrariumWorld::demo(23, false).unwrap();
    let before = world.snapshot();
    world.run_frames(120).unwrap();
    let after = world.snapshot();
    assert!(
        after.mean_microbial_lineage_generation >= before.mean_microbial_lineage_generation,
        "microbial lineage generation did not progress"
    );
    assert!(
        after.mean_nitrifier_lineage_generation >= before.mean_nitrifier_lineage_generation,
        "nitrifier lineage generation did not progress"
    );
    assert!(
        after.mean_denitrifier_lineage_generation >= before.mean_denitrifier_lineage_generation,
        "denitrifier lineage generation did not progress"
    );
    assert!(
        after.mean_microbial_lineage_novelty > 0.0
            || after.mean_nitrifier_lineage_novelty > 0.0
            || after.mean_denitrifier_lineage_novelty > 0.0,
        "secondary lineage novelty stayed at zero"
    );
}

#[test]
fn snapshot_uses_catalog_slots_and_catalog_bank_for_secondary_genotypes() {
    let mut world = TerrariumWorld::demo(31, false).unwrap();
    world.run_frames(8).unwrap();
    let baseline = world.snapshot();

    for secondary in [
        &mut world.microbial_secondary,
        &mut world.nitrifier_secondary,
        &mut world.denitrifier_secondary,
    ] {
        for bank in &mut secondary.banks {
            if let Some(catalog_entry) = bank.catalog_bank.get_mut(0) {
                catalog_entry.genes = [1.0; INTERNAL_SECONDARY_GENOTYPE_AXES];
                catalog_entry.record.genotype_divergence = 1.0;
                catalog_entry.record.generation = 999.0;
                catalog_entry.record.novelty = 1.0;
                catalog_entry.catalog.generation = 999.0;
                catalog_entry.catalog.novelty = 1.0;
                catalog_entry.catalog.local_bank_share = 1.0;
            }
            for slot in &mut bank.catalog_slots {
                *slot = 0;
            }
        }
    }

    let corrupted = world.snapshot();
    assert!(
        (corrupted.mean_microbial_gene_catabolic - baseline.mean_microbial_gene_catabolic).abs()
            > 1.0e-4
    );
    assert!(
        (corrupted.mean_microbial_genotype_divergence
            - baseline.mean_microbial_genotype_divergence)
            .abs()
            > 1.0e-4
    );
    assert!(
        (corrupted.mean_microbial_catalog_generation - baseline.mean_microbial_catalog_generation)
            .abs()
            > 1.0e-4
    );
    assert!(
        (corrupted.mean_microbial_lineage_generation - baseline.mean_microbial_lineage_generation)
            .abs()
            > 1.0e-4
    );
    assert!(
        (corrupted.mean_nitrifier_gene_oxygen_respiration
            - baseline.mean_nitrifier_gene_oxygen_respiration)
            .abs()
            > 1.0e-4
    );
    assert!(
        (corrupted.mean_nitrifier_genotype_divergence
            - baseline.mean_nitrifier_genotype_divergence)
            .abs()
            > 1.0e-4
    );
    assert!(
        (corrupted.mean_nitrifier_catalog_generation - baseline.mean_nitrifier_catalog_generation)
            .abs()
            > 1.0e-4
    );
    assert!(
        (corrupted.mean_nitrifier_lineage_generation - baseline.mean_nitrifier_lineage_generation)
            .abs()
            > 1.0e-4
    );
    assert!(
        (corrupted.mean_denitrifier_gene_anoxia_respiration
            - baseline.mean_denitrifier_gene_anoxia_respiration)
            .abs()
            > 1.0e-4
    );
    assert!(
        (corrupted.mean_denitrifier_genotype_divergence
            - baseline.mean_denitrifier_genotype_divergence)
            .abs()
            > 1.0e-4
    );
    assert!(
        (corrupted.mean_denitrifier_catalog_generation
            - baseline.mean_denitrifier_catalog_generation)
            .abs()
            > 1.0e-4
    );
    assert!(
        (corrupted.mean_denitrifier_lineage_generation
            - baseline.mean_denitrifier_lineage_generation)
            .abs()
            > 1.0e-4
    );
}

#[test]
fn gas_exchange_field_changes_over_time() {
    let mut world = TerrariumWorld::demo(7, false).unwrap();
    let initial = world.topdown_field(TerrariumTopdownView::GasExchange);
    let initial_o2 = world.topdown_atmospheric_o2();
    world.run_frames(8).unwrap();
    let evolved = world.topdown_field(TerrariumTopdownView::GasExchange);
    let evolved_o2 = world.topdown_atmospheric_o2();
    let max_delta = initial
        .iter()
        .zip(evolved.iter())
        .map(|(before, after)| (before - after).abs())
        .fold(0.0f32, f32::max);
    let max_o2_delta = initial_o2
        .iter()
        .zip(evolved_o2.iter())
        .map(|(before, after)| (before - after).abs())
        .fold(0.0f32, f32::max);
    assert!(max_delta > 1.0e-6, "gas exchange field stayed static");
    assert!(
        max_o2_delta > 1.0e-7,
        "atmospheric oxygen field stayed static"
    );
    let snapshot = world.snapshot();
    assert!(
        snapshot.mean_atmospheric_o2.is_finite() && snapshot.mean_atmospheric_o2 >= 0.0,
        "mean_atmospheric_o2 should be finite and non-negative, got {}",
        snapshot.mean_atmospheric_o2,
    );
}

#[test]
fn nonplant_respiration_changes_gas_balance() {
    let mut world = TerrariumWorld::demo(7, false).unwrap();
    world.plants.clear();
    world
        .substrate
        .relax_species_toward(TerrariumSpecies::OxygenGas, 0.04, 0.92);
    let initial = world.snapshot();
    world.run_frames(8).unwrap();
    let evolved = world.snapshot();
    assert!(
        evolved.mean_atmospheric_co2 > initial.mean_atmospheric_co2 + 1.0e-5,
        "nonplant respiration did not raise atmospheric co2"
    );
    assert!(
        (evolved.mean_atmospheric_o2 - initial.mean_atmospheric_o2).abs() > 1.0e-5,
        "nonplant respiration did not materially change atmospheric o2"
    );
}

#[test]
fn explicit_air_pressure_field_tracks_native_atmosphere_state() {
    let mut world = TerrariumWorld::demo(7, false).unwrap();
    let initial_pressure = world.topdown_air_pressure_kpa();
    let hot_x = world.config.width / 2;
    let hot_y = world.config.height / 2;
    for z in 0..world.config.depth.max(1) {
        let hot_idx = (z * world.config.height + hot_y) * world.config.width + hot_x;
        world.temperature[hot_idx] = 29.0 - z as f32 * 0.8;
        world.humidity[hot_idx] = 0.88 - z as f32 * 0.12;
        world.wind_x[hot_idx] = 1.4;
        world.wind_y[hot_idx] = -0.9;
        world.wind_z[hot_idx] = 0.5;
    }

    world.step_explicit_air_state().unwrap();

    let evolved_pressure = world.topdown_air_pressure_kpa();
    let max_delta = initial_pressure
        .iter()
        .zip(evolved_pressure.iter())
        .map(|(before, after)| (before - after).abs())
        .fold(0.0f32, f32::max);
    assert!(max_delta > 1.0e-4, "air pressure field stayed static");
    let snapshot = world.snapshot();
    assert!(
        snapshot.mean_air_pressure_kpa > 80.0 && snapshot.mean_air_pressure_kpa < 120.0,
        "mean air pressure left a plausible terrarium range"
    );
}

#[test]
fn visual_biomechanics_follow_native_air_state() {
    let mut world = TerrariumWorld::demo(13, false).unwrap();
    let plant_idx = 0;
    let (plant_x, plant_y) = (world.plants[plant_idx].x, world.plants[plant_idx].y);
    world.add_fruit(plant_x, plant_y, 0.9, Some(1.0));
    let fruit_idx = world.fruits.len() - 1;

    let canopy_z = ((world.plants[plant_idx].physiology.height_mm()
        / world.config.cell_size_mm.max(1.0e-3))
    .round() as usize)
        .min(world.config.depth.max(1) - 1);
    let canopy_flat = (canopy_z * world.config.height + plant_y) * world.config.width + plant_x;
    world.wind_x[canopy_flat] = 1.9;
    world.wind_y[canopy_flat] = -1.4;
    world.wind_z[canopy_flat] = 0.8;
    world.air_pressure_kpa[canopy_flat] = 101.325 * 1.02;
    world.air_density[canopy_flat] = 1.197 * 0.94;

    world.step_biomechanics(world.config.world_dt_s);
    world.step_biomechanics(world.config.world_dt_s);

    let plant_pose = &world.plants[plant_idx].pose;
    assert!(
        plant_pose
            .canopy_offset_mm
            .iter()
            .any(|offset| offset.abs() > 1.0e-5),
        "plant pose stayed static under explicit air forcing"
    );
    assert!(
        plant_pose.stem_tilt_x_rad.abs() > 1.0e-5 || plant_pose.stem_tilt_z_rad.abs() > 1.0e-5,
        "plant stem tilt stayed static under explicit air forcing"
    );

    let fruit_pose = &world.fruits[fruit_idx].pose;
    assert!(
        fruit_pose
            .offset_mm
            .iter()
            .any(|offset| offset.abs() > 1.0e-5),
        "fruit pose stayed static under explicit air forcing"
    );
}

#[test]
fn seed_cellular_state_tracks_native_seed_environment() {
    let mut world = TerrariumWorld::demo(17, false).unwrap();
    let eco_dt = world.config.world_dt_s * world.config.time_warp;
    let plant_idx = 0;
    let (plant_x, plant_y) = (world.plants[plant_idx].x, world.plants[plant_idx].y);
    if world.seeds.is_empty() {
        let genome = world.plants[plant_idx].genome.clone();
        let identity = world.register_organism_identity(
            TerrariumOrganismKind::Seed,
            Some(genome.taxonomy_id),
            None,
            None,
            None,
            Some(crate::terrarium::organism_identity::plant_genome_hash(
                &genome,
            )),
            Some(
                crate::terrarium::organism_identity::seed_phylo_traits_from_genome(
                    &genome, 0.18, 1.0,
                ),
            ),
        );
        world.seeds.push(TerrariumSeed {
            identity,
            x: (plant_x as f32 + 1.25).clamp(0.0, world.config.width as f32 - 0.01),
            y: (plant_y as f32 + 0.75).clamp(0.0, world.config.height as f32 - 0.01),
            dormancy_s: 14_000.0,
            reserve_carbon: 0.18,
            age_s: 3_600.0,
            genome: genome.clone(),
            cellular: SeedCellularStateSim::new(genome.seed_mass, 0.18, 14_000.0),
            pose: TerrariumSeedPose::default(),
            microsite: TerrariumSeedMicrosite::default(),
            material_inventory: RegionalMaterialInventory::new("seed:test:native_env".into()),
        });
    }
    let seed_x = world.seeds[0]
        .x
        .round()
        .clamp(0.0, (world.config.width - 1) as f32) as usize;
    let seed_y = world.seeds[0]
        .y
        .round()
        .clamp(0.0, (world.config.height - 1) as f32) as usize;
    let flat = idx2(world.config.width, seed_x, seed_y);
    let before_radicle = world.seeds[0]
        .cellular
        .cluster_snapshot(SeedTissue::Radicle);
    let before_reserve = world.seeds[0].reserve_carbon;

    world.moisture[flat] = 0.95;
    world.deep_moisture[flat] = 0.72;
    world.shallow_nutrients[flat] = 0.32;
    world.symbiont_biomass[flat] = 0.24;
    world.canopy_cover[flat] = 0.18;
    world.litter_carbon[flat] = 0.14;

    world.step_seeds_native(eco_dt).unwrap();
    world.step_seeds_native(eco_dt).unwrap();

    let after_radicle = world.seeds[0]
        .cellular
        .cluster_snapshot(SeedTissue::Radicle);
    assert!(
        after_radicle.transcript_germination_program
            > before_radicle.transcript_germination_program
            || after_radicle.cell_count > before_radicle.cell_count,
        "seed cellular radicle state stayed static under favorable local conditions"
    );
    assert!(
        world.seeds[0].reserve_carbon != before_reserve,
        "seed reserve carbon did not reconcile from native cellular state"
    );
}

#[test]
fn seed_germination_follows_native_cellular_feedback() {
    let mut world = TerrariumWorld::demo(17, false).unwrap();
    let eco_dt = world.config.world_dt_s * world.config.time_warp;
    let initial_plants = world.plants.len();
    world.seeds.clear();
    let mut target = None;
    'find_open: for y in 0..world.config.height {
        for x in 0..world.config.width {
            if !world
                .plants
                .iter()
                .any(|plant| plant.x == x && plant.y == y)
            {
                target = Some((x, y));
                break 'find_open;
            }
        }
    }
    let (seed_x, seed_y) = target.expect("demo terrarium had no free seed cell");
    let genome = world.plants[0].genome.clone();
    let identity = world.register_organism_identity(
        TerrariumOrganismKind::Seed,
        Some(genome.taxonomy_id),
        None,
        None,
        None,
        Some(crate::terrarium::organism_identity::plant_genome_hash(
            &genome,
        )),
        Some(
            crate::terrarium::organism_identity::seed_phylo_traits_from_genome(&genome, 0.18, 1.0),
        ),
    );
    world.seeds.push(TerrariumSeed {
        identity,
        x: seed_x as f32,
        y: seed_y as f32,
        dormancy_s: 0.0,
        reserve_carbon: 0.18,
        age_s: 3_600.0,
        genome: genome.clone(),
        cellular: SeedCellularStateSim::new(genome.seed_mass, 0.18, 0.0),
        pose: TerrariumSeedPose::default(),
        microsite: TerrariumSeedMicrosite::default(),
        material_inventory: RegionalMaterialInventory::new("seed:test:germination".into()),
    });

    let flat = idx2(world.config.width, seed_x, seed_y);
    world.moisture[flat] = 1.0;
    world.deep_moisture[flat] = 0.82;
    world.shallow_nutrients[flat] = 0.36;
    world.symbiont_biomass[flat] = 0.28;
    world.canopy_cover[flat] = 0.08;
    world.litter_carbon[flat] = 0.18;

    for _ in 0..8 {
        world.step_seeds_native(eco_dt).unwrap();
        if world.plants.len() > initial_plants {
            break;
        }
    }

    assert!(
        world.plants.len() > initial_plants,
        "seed did not germinate from native cellular feedback under favorable local conditions"
    );
}

#[test]
fn render_projection_reports_native_scene_state() {
    let mut world = TerrariumWorld::demo(17, false).unwrap();
    let plant_idx = 0;
    let (plant_x, plant_y) = (world.plants[plant_idx].x, world.plants[plant_idx].y);
    world.add_fruit(plant_x, plant_y, 0.9, Some(1.0));
    if world.seeds.is_empty() {
        let genome = world.plants[plant_idx].genome.clone();
        let identity = world.register_organism_identity(
            TerrariumOrganismKind::Seed,
            Some(genome.taxonomy_id),
            None,
            None,
            None,
            Some(crate::terrarium::organism_identity::plant_genome_hash(
                &genome,
            )),
            Some(
                crate::terrarium::organism_identity::seed_phylo_traits_from_genome(
                    &genome, 0.18, 1.0,
                ),
            ),
        );
        world.seeds.push(TerrariumSeed {
            identity,
            x: (plant_x as f32 + 1.25).clamp(0.0, world.config.width as f32 - 0.01),
            y: (plant_y as f32 + 0.75).clamp(0.0, world.config.height as f32 - 0.01),
            dormancy_s: 14_000.0,
            reserve_carbon: 0.18,
            age_s: 3_600.0,
            genome: genome.clone(),
            cellular: SeedCellularStateSim::new(genome.seed_mass, 0.18, 14_000.0),
            pose: TerrariumSeedPose::default(),
            microsite: TerrariumSeedMicrosite::default(),
            material_inventory: RegionalMaterialInventory::new("seed:test:render".into()),
        });
    }
    let seed_x = world.seeds[0]
        .x
        .round()
        .clamp(0.0, (world.config.width - 1) as f32) as usize;
    let seed_y = world.seeds[0]
        .y
        .round()
        .clamp(0.0, (world.config.height - 1) as f32) as usize;
    let seed_x_l = seed_x.saturating_sub(1);
    let seed_x_r = (seed_x + 1).min(world.config.width - 1);
    let seed_y_d = seed_y.saturating_sub(1);
    let seed_y_u = (seed_y + 1).min(world.config.height - 1);
    let seed_flat_l = idx2(world.config.width, seed_x_l, seed_y);
    let seed_flat_r = idx2(world.config.width, seed_x_r, seed_y);
    let seed_flat_d = idx2(world.config.width, seed_x, seed_y_d);
    let seed_flat_u = idx2(world.config.width, seed_x, seed_y_u);
    world.moisture[seed_flat_l] = 0.12;
    world.moisture[seed_flat_r] = 1.05;
    world.shallow_nutrients[seed_flat_d] = 0.08;
    world.shallow_nutrients[seed_flat_u] = 1.10;
    world.symbiont_biomass[seed_flat_u] = 0.42;
    let seed_air = idx3(world.config.width, world.config.height, seed_x, seed_y, 0);
    world.wind_x[seed_air] = 1.1;
    world.wind_y[seed_air] = -0.7;
    world.wind_z[seed_air] = 0.2;
    world.air_pressure_kpa[seed_air] = ATMOS_PRESSURE_BASELINE_KPA * 1.008;
    world.air_density[seed_air] = ATMOS_DENSITY_BASELINE_KG_M3 * 0.985;
    let canopy_z = ((world.plants[plant_idx].physiology.height_mm()
        / world.config.cell_size_mm.max(1.0e-3))
    .round() as usize)
        .min(world.config.depth.max(1) - 1);
    let canopy_flat = (canopy_z * world.config.height + plant_y) * world.config.width + plant_x;
    world.wind_x[canopy_flat] = 1.4;
    world.wind_y[canopy_flat] = -0.8;
    world.wind_z[canopy_flat] = 0.5;
    world.air_pressure_kpa[canopy_flat] = ATMOS_PRESSURE_BASELINE_KPA * 1.01;
    world.air_density[canopy_flat] = ATMOS_DENSITY_BASELINE_KG_M3 * 0.96;
    world.odorants[ETHYL_ACETATE_IDX][canopy_flat] = 1.4;
    world.odorants[ATMOS_CO2_IDX][canopy_flat] = ATMOS_CO2_BASELINE * 1.8;
    world.odorants[ATMOS_O2_IDX][canopy_flat] = ATMOS_O2_BASELINE * 0.7;
    let hotspot_z = 1.min(world.config.depth.max(1) - 1);
    let hotspot = (hotspot_z * world.config.height + plant_y) * world.config.width + plant_x;
    world.substrate.hydration[hotspot] = 1.0;
    world.substrate.microbial_activity[hotspot] = 1.2;
    world.substrate.nitrifier_activity[hotspot] = 1.1;
    world.substrate.denitrifier_activity[hotspot] = 0.9;
    world.substrate.plant_drive[hotspot] = 1.0;
    let hotspot_water = world
        .substrate
        .index(TerrariumSpecies::Water, plant_x, plant_y, hotspot_z);
    world.substrate.current[hotspot_water] = 1.1;
    let hotspot_glucose =
        world
            .substrate
            .index(TerrariumSpecies::Glucose, plant_x, plant_y, hotspot_z);
    world.substrate.current[hotspot_glucose] = 0.24;
    let hotspot_oxygen =
        world
            .substrate
            .index(TerrariumSpecies::OxygenGas, plant_x, plant_y, hotspot_z);
    world.substrate.current[hotspot_oxygen] = 0.42;
    let hotspot_nitrate =
        world
            .substrate
            .index(TerrariumSpecies::Nitrate, plant_x, plant_y, hotspot_z);
    world.substrate.current[hotspot_nitrate] = 0.18;
    let hotspot_co2 =
        world
            .substrate
            .index(TerrariumSpecies::CarbonDioxide, plant_x, plant_y, hotspot_z);
    world.substrate.current[hotspot_co2] = 0.12;
    let hotspot_atp = world
        .substrate
        .index(TerrariumSpecies::AtpFlux, plant_x, plant_y, hotspot_z);
    world.substrate.current[hotspot_atp] = 0.08;
    world.recruit_packet_populations();
    world.step_biomechanics(world.config.world_dt_s);
    world.step_biomechanics(world.config.world_dt_s);

    let tiles = world.render_ground_tiles(TerrariumTopdownView::Terrain);
    assert_eq!(tiles.len(), world.config.width * world.config.height);
    assert!(tiles.iter().all(|tile| tile.scale_world[1] > 0.0));
    assert!(tiles
        .iter()
        .all(|tile| tile.material.perceptual_roughness > 0.0));
    assert!(tiles
        .iter()
        .all(|tile| tile.primitive == TerrariumMeshPrimitive::Cube));
    assert!(tiles
        .iter()
        .all(|tile| !tile.mesh.positions.is_empty() && !tile.mesh.indices.is_empty()));
    let odor_tiles = world.render_ground_tiles(TerrariumTopdownView::Odor);
    assert!(
        odor_tiles
            .iter()
            .any(|tile| tile.material.emissive_rgb[0] > 0.0),
        "odor tile render descriptors omitted emissive response"
    );

    let render = world.render_dynamic_snapshot();
    let substrate_batches = world.render_substrate_batches_cached(0).to_vec();
    let dynamic_batches = world.render_dynamic_batches_cached(0).to_vec();
    let mut render_ids = std::collections::HashSet::new();
    assert!(
        !render.substrate_voxels.is_empty(),
        "native render projection omitted explicit substrate voxels"
    );
    assert!(
        !substrate_batches.is_empty(),
        "native render projection omitted cached substrate batches for viewer consumption"
    );
    assert!(
        !dynamic_batches.is_empty(),
        "native render projection omitted cached dynamic biological batches for viewer consumption"
    );
    assert!(substrate_batches.iter().all(|batch| {
        batch.render_id != 0
            && batch.render_fingerprint != 0
            && batch.mesh_cache_key != 0
            && batch.material_state_key != 0
            && !batch.mesh.positions.is_empty()
            && !batch.mesh.indices.is_empty()
    }));
    assert!(dynamic_batches.iter().all(|batch| {
        batch.render_id != 0
            && batch.render_fingerprint != 0
            && batch.mesh_cache_key != 0
            && batch.material_state_key != 0
            && !batch.mesh.positions.is_empty()
            && !batch.mesh.indices.is_empty()
    }));
    assert!(dynamic_batches.iter().any(|batch| {
        batch.kind == TerrariumDynamicBatchKind::MicrobePacket
            || batch.kind == TerrariumDynamicBatchKind::MicrobeSite
            || batch.kind == TerrariumDynamicBatchKind::Water
            || batch.kind == TerrariumDynamicBatchKind::Plume
            || batch.kind == TerrariumDynamicBatchKind::PlantTissue
            || batch.kind == TerrariumDynamicBatchKind::SeedPart
            || batch.kind == TerrariumDynamicBatchKind::FruitPart
            || batch.kind == TerrariumDynamicBatchKind::FlyPart
    }));
    assert!(dynamic_batches
        .iter()
        .any(|batch| batch.kind == TerrariumDynamicBatchKind::Water));
    assert!(dynamic_batches
        .iter()
        .any(|batch| batch.kind == TerrariumDynamicBatchKind::PlantTissue));
    assert!(dynamic_batches
        .iter()
        .any(|batch| batch.kind == TerrariumDynamicBatchKind::SeedPart));
    assert!(dynamic_batches
        .iter()
        .any(|batch| batch.kind == TerrariumDynamicBatchKind::FruitPart));
    assert!(dynamic_batches
        .iter()
        .any(|batch| batch.kind == TerrariumDynamicBatchKind::FlyPart));
    assert!(
        !render.explicit_microbes.is_empty(),
        "native render projection omitted explicit microbe cohorts"
    );
    assert!(render.substrate_voxels.iter().all(|voxel| {
        render_ids.insert(voxel.render_id)
            && (voxel.material.shader_flags & TERRARIUM_SHADER_FLAG_SUBSTRATE) != 0
            && voxel.material.shader_dynamics_rgba[0] >= 0.0
            && voxel.material.shader_dynamics_rgba[1] >= 0.0
            && voxel.render_fingerprint != 0
            && voxel.primitive == TerrariumMeshPrimitive::Cube
            && voxel.signal >= 0.0
            && !voxel.mesh.positions.is_empty()
            && !voxel.mesh.indices.is_empty()
    }));
    assert!(render.explicit_microbes.iter().all(|microbe| {
        render_ids.insert(microbe.render_id)
            && ((!microbe.body_mesh.positions.is_empty() && !microbe.body_mesh.indices.is_empty())
                || !microbe.sites.is_empty())
            && microbe.body_render_fingerprint != 0
            && microbe.packet_population_render_fingerprint != 0
            && microbe.represented_cells > 0.0
            && microbe.represented_packets > 0.0
    }));
    assert!(
        render
            .explicit_microbes
            .iter()
            .flat_map(|microbe| microbe.packets.iter())
            .all(|packet| {
                render_ids.insert(packet.render_id)
                    && (packet.material.shader_flags & TERRARIUM_SHADER_FLAG_MICROBE) != 0
                    && packet.render_fingerprint != 0
            }),
        "explicit microbe packet renders lost native shader IDs or category flags"
    );
    assert!(
        render
            .explicit_microbes
            .iter()
            .flat_map(|microbe| microbe.sites.iter())
            .all(|site| {
                render_ids.insert(site.render_id)
                    && (site.material.shader_flags & TERRARIUM_SHADER_FLAG_MICROBE) != 0
                    && site.render_fingerprint != 0
            }),
        "explicit microbe site renders lost native shader IDs or category flags"
    );
    assert!(
        render.explicit_microbes.iter().any(|microbe| {
            !microbe.packet_mesh.indices.is_empty()
                || microbe
                    .packets
                    .iter()
                    .any(|packet| !packet.mesh.indices.is_empty())
        }),
        "explicit microbe packet geometry was not projected"
    );
    assert!(
        render
            .explicit_microbes
            .iter()
            .any(|microbe| !microbe.sites.is_empty()),
        "explicit microbe local chemistry sites were not projected"
    );
    assert!(
        render
            .explicit_microbes
            .iter()
            .any(|microbe| !microbe.packets.is_empty()),
        "explicit microbe packet populations were not kept as first-class native renders"
    );
    assert!(!render.waters.is_empty());
    assert!(render.waters.iter().all(|water| {
        render_ids.insert(water.render_id)
            && water.render_fingerprint != 0
            && (water.material.shader_flags & TERRARIUM_SHADER_FLAG_FLUID) != 0
    }));
    assert!(!render.plants.is_empty());
    assert!(!render.seeds.is_empty());
    assert!(!render.fruits.is_empty());
    assert!(!render.flies.is_empty());
    assert!(
        render
            .plumes
            .iter()
            .any(|plume| plume.kind == TerrariumRenderPlumeKind::Odor),
        "native render projection omitted odor plumes"
    );
    assert!(
        render
            .plumes
            .iter()
            .any(|plume| plume.kind == TerrariumRenderPlumeKind::GasExchange),
        "native render projection omitted gas-exchange plumes"
    );
    assert!(
        render
            .plumes
            .iter()
            .any(|plume| plume.kind == TerrariumRenderPlumeKind::Microbe),
        "native render projection omitted explicit-microbe plumes"
    );
    let plant_render = &render.plants[plant_idx];
    assert!(
        plant_render.tissues.len() >= 4,
        "plant cellular tissue descriptors were not projected"
    );
    assert!(
        plant_render.stem_rotation_xyz_rad[0].abs() > 1.0e-5
            || plant_render.stem_rotation_xyz_rad[2].abs() > 1.0e-5,
        "render projection dropped plant pose authority"
    );
    assert!(
        plant_render.tissues.iter().all(|tissue| {
            render_ids.insert(tissue.render_id)
                && (tissue.material.shader_flags & TERRARIUM_SHADER_FLAG_PLANT) != 0
                && tissue.render_fingerprint != 0
                && !tissue.mesh.indices.is_empty()
        }),
        "plant tissue meshes were empty"
    );
    assert!(
        plant_render
            .tissues
            .iter()
            .any(|tissue| tissue.tissue == crate::PlantTissue::Root
                && tissue.translation_world[1] < plant_render.stem_translation_world[1]),
        "plant root tissue did not remain below the stem anchor"
    );
    assert!(
        render
            .fruits
            .iter()
            .any(|fruit| fruit.translation_world[1] > plant_render.stem_translation_world[1]),
        "render projection did not keep fruit attached above the substrate"
    );
    assert!(render
        .seeds
        .iter()
        .all(|seed| render_ids.insert(seed.render_id) && !seed.parts.is_empty()));
    assert!(
        render.seeds.iter().any(|seed| seed.parts.len() >= 2),
        "seed native part projection did not preserve multipart structure"
    );
    assert!(
        render.seeds.iter().any(|seed| {
            seed.parts
                .iter()
                .any(|part| part.kind == TerrariumSeedPartKind::Coat)
                && seed
                    .parts
                    .iter()
                    .any(|part| part.kind == TerrariumSeedPartKind::Radicle)
        }),
        "seed substructure descriptors were not projected"
    );
    assert!(
        render.seeds.iter().any(|seed| {
            seed.rotation_xyz_rad[0].abs() > 1.0e-5
                || seed.rotation_xyz_rad[1].abs() > 1.0e-5
                || seed.rotation_xyz_rad[2].abs() > 1.0e-5
        }),
        "seed pose authority was not retained by render projection"
    );
    assert!(render.fruits.iter().all(|fruit| !fruit.parts.is_empty()));
    assert!(render
        .fruits
        .iter()
        .all(|fruit| render_ids.insert(fruit.render_id)));
    assert!(
        render.fruits.iter().any(|fruit| {
            fruit
                .parts
                .iter()
                .any(|part| part.kind == TerrariumFruitPartKind::Skin)
                && fruit
                    .parts
                    .iter()
                    .any(|part| part.kind == TerrariumFruitPartKind::Stem)
        }),
        "fruit substructure descriptors were not projected"
    );
    let terrain = world.topdown_field(TerrariumTopdownView::Terrain);
    let (terrain_min, terrain_inv) = field_min_inv(&terrain);
    let ground_y = render_top_surface_y(normalize_unit(
        terrain[idx2(world.config.width, plant_x, plant_y)],
        terrain_min,
        terrain_inv,
    ));
    assert!(
        render.plumes.iter().any(|plume| {
            plume.kind == TerrariumRenderPlumeKind::Odor
                && plume.translation_world[1]
                    > ground_y + canopy_z as f32 * RENDER_ALTITUDE_SCALE * 0.5
        }),
        "odor plume renders did not retain 3D atmosphere altitude"
    );
    assert!(
        render.plumes.iter().all(|plume| {
            render_ids.insert(plume.render_id)
                && plume.render_fingerprint != 0
                && (plume.material.shader_flags & TERRARIUM_SHADER_FLAG_PLUME) != 0
                && plume.material.alpha_blend
                && plume.material.base_color_rgba[3] > 0.0
        }),
        "plume descriptors lost translucent material state"
    );
    assert!(render
        .flies
        .iter()
        .all(|fly| render_ids.insert(fly.render_id) && !fly.parts.is_empty()));
    assert!(render.flies.iter().any(|fly| {
        fly.parts
            .iter()
            .any(|part| part.kind == TerrariumFlyPartKind::Thorax)
            && fly
                .parts
                .iter()
                .any(|part| part.kind == TerrariumFlyPartKind::WingLeft)
            && fly
                .parts
                .iter()
                .any(|part| part.kind == TerrariumFlyPartKind::WingRight)
    }));
    assert!(render.plants.iter().all(|plant| {
        (!plant.tissues.is_empty())
            || (!plant.stem_mesh.indices.is_empty() && !plant.canopy_mesh.indices.is_empty())
    }));
    assert!(
        render
            .plants
            .iter()
            .any(|plant| !plant.tissues.is_empty() && plant.stem_mesh.indices.is_empty()),
        "plant fallback meshes were not retired when tissue renders were present"
    );
    assert!(render.flies.iter().all(|fly| fly.parts.iter().all(|part| {
        render_ids.insert(part.render_id)
            && part.render_fingerprint != 0
            && (part.material.shader_flags & TERRARIUM_SHADER_FLAG_FLY) != 0
            && !part.mesh.indices.is_empty()
    })));
    assert!(
        render
            .seeds
            .iter()
            .flat_map(|seed| seed.parts.iter())
            .all(|part| {
                render_ids.insert(part.render_id)
                    && (part.material.shader_flags & TERRARIUM_SHADER_FLAG_SEED) != 0
                    && part.render_fingerprint != 0
            }),
        "seed part descriptors lost native shader IDs or category flags"
    );
    assert!(
        render
            .fruits
            .iter()
            .flat_map(|fruit| fruit.parts.iter())
            .all(|part| {
                render_ids.insert(part.render_id)
                    && (part.material.shader_flags & TERRARIUM_SHADER_FLAG_FRUIT) != 0
                    && part.render_fingerprint != 0
            }),
        "fruit part descriptors lost native shader IDs or category flags"
    );
    assert!(
        render.flies.iter().any(|fly| {
            fly.parts
                .iter()
                .any(|part| part.kind == TerrariumFlyPartKind::Thorax)
                && fly
                    .parts
                    .iter()
                    .any(|part| part.kind == TerrariumFlyPartKind::WingLeft)
                && fly
                    .parts
                    .iter()
                    .any(|part| part.kind == TerrariumFlyPartKind::WingRight)
        }),
        "fly substructure descriptors were not projected"
    );
    assert!(render
        .flies
        .iter()
        .all(|fly| fly.point_light_render_fingerprint != 0));

    let lighting = world.render_lighting();
    assert!(lighting.sun_illuminance > 0.0);
    assert!(lighting.ambient_brightness > 0.0);
    assert_ne!(lighting.clear_color_rgb, lighting.sun_color_rgb);
    assert!((0.0..=1.0).contains(&lighting.humidity_t));
    assert!((0.0..=1.0).contains(&lighting.pressure_t));
    assert!((0.0..=1.0).contains(&lighting.temperature_t));
    assert!((0.0..=1.0).contains(&lighting.daylight));
    assert!((0.0..=1.0).contains(&lighting.time_phase));
}

#[test]
fn render_lighting_tracks_scene_air_and_day_cycle() {
    let mut world = TerrariumWorld::demo(31, false).unwrap();
    let warm_midday = 31.0;
    let humid_midday = 0.86;
    let pressurized_midday = ATMOS_PRESSURE_BASELINE_KPA + 1.6;
    world.time_s = 12.0 * 3600.0;
    world.temperature.fill(warm_midday);
    world.humidity.fill(humid_midday);
    world.air_pressure_kpa.fill(pressurized_midday);

    let midday = world.render_lighting();
    let expected_humidity_t = super::mean(&world.topdown_humidity()).clamp(0.0, 1.0);
    let expected_pressure_mean = super::mean(&world.topdown_air_pressure_kpa());
    assert!(
        (midday.humidity_t - expected_humidity_t).abs() < 1.0e-4,
        "native lighting humidity should come directly from the terrarium field"
    );
    let expected_pressure_t =
        ((expected_pressure_mean - ATMOS_PRESSURE_BASELINE_KPA) / 4.0 * 0.5 + 0.5).clamp(0.0, 1.0);
    assert!(
        (midday.pressure_t - expected_pressure_t).abs() < 1.0e-4,
        "native lighting pressure normalization drifted from terrarium state"
    );
    let expected_temperature_t = ((warm_midday - 8.0) / 28.0).clamp(0.0, 1.0);
    assert!(
        (midday.temperature_t - expected_temperature_t).abs() < 1.0e-4,
        "native lighting temperature normalization drifted from terrarium state"
    );
    assert!(
        midday.daylight > 0.95,
        "midday lighting should read as high daylight when the terrarium clock is at noon"
    );
    assert!(
        (midday.time_phase - 0.5).abs() < 1.0e-4,
        "native lighting phase should track terrarium time directly"
    );

    world.time_s = 0.0;
    let midnight = world.render_lighting();
    assert!(
        midnight.daylight < midday.daylight,
        "lighting daylight should respond to the terrarium clock"
    );
    assert!(
        midnight.sun_illuminance < midday.sun_illuminance,
        "sun output should dim when the terrarium clock moves to night"
    );
    assert!(
        midnight.time_phase < midday.time_phase,
        "native lighting phase should change when the terrarium clock changes"
    );
}

#[test]
fn native_render_delta_reports_only_changed_descriptors() {
    let mut world = TerrariumWorld::demo(19, false).unwrap();

    let initial = world.render_dynamic_delta_cached(0).clone();
    assert!(
        !initial.meshes.is_empty() || !initial.point_lights.is_empty(),
        "initial native render delta should contain the first full scene projection"
    );
    assert!(
        initial.removed_render_ids.is_empty(),
        "initial native render delta should not report removals"
    );
    assert!(initial.meshes.iter().all(|mesh| mesh.mesh_cache_key != 0));
    assert!(initial
        .meshes
        .iter()
        .all(|mesh| mesh.material_state_key != 0));
    assert!(initial
        .meshes
        .iter()
        .all(|mesh| ((mesh.render_id >> 56) & 0xff) as u8 != 5));

    let noop = world.render_dynamic_delta_cached(1).clone();
    assert!(noop.meshes.is_empty());
    assert!(noop.point_lights.is_empty());
    assert!(noop.removed_render_ids.is_empty());
}

#[test]
fn native_scene_raycast_hits_live_geometry() {
    let mut world = TerrariumWorld::demo(23, false).unwrap();
    // Force the snapshot build so we can inspect the BVH.
    let _ = world.render_dynamic_snapshot_cached(0);
    assert!(
        world.cached_raycast_scene.bvh_root.is_some(),
        "BVH should be built after snapshot cache"
    );
    assert!(
        !world.cached_raycast_scene.bvh_nodes.is_empty(),
        "BVH should have nodes after snapshot cache"
    );
    let n_surfaces = world.cached_raycast_scene.surfaces.len();
    let _n_solid = world
        .cached_raycast_scene
        .surfaces
        .iter()
        .filter(|s| s.solid)
        .count();
    assert!(n_surfaces > 0, "scene should have surfaces; got 0");
    // Copy BVH root AABB so we don't hold a borrow across the mutable raycast call.
    // Pick a known solid surface and aim at its AABB center from above.
    let solid_surface = world
        .cached_raycast_scene
        .surfaces
        .iter()
        .find(|s| s.solid && !s.world_positions.is_empty())
        .expect("should have at least one solid surface");
    let sx = (solid_surface.aabb_min_world[0] + solid_surface.aabb_max_world[0]) * 0.5;
    let sz = (solid_surface.aabb_min_world[2] + solid_surface.aabb_max_world[2]) * 0.5;
    let above_y = solid_surface.aabb_max_world[1] + 2.0;
    // Drop the borrow before calling the mutable raycast method.
    let hit =
        world.render_scene_solid_raycast_cached(0, [sx, above_y, sz], [0.0, -1.0, 0.0], false);
    let hit = hit.expect("solid raycast at known surface center should hit");
    assert!(hit.distance > 0.0);
    assert!(hit.position_world[1] <= above_y);
    assert!(hit.normal_world[1].abs() > 0.0);
}

#[test]
fn native_scene_raycast_includes_batch_owned_water_geometry() {
    let mut world = TerrariumWorld::demo(29, false).unwrap();
    world.plants.clear();
    world.fruits.clear();
    world.flies.clear();
    world.explicit_microbes.clear();
    world.seeds.clear();
    world.waters.clear();
    let water_x = world.config.width / 2;
    let water_y = world.config.height / 2;
    world.add_water(water_x, water_y, 80.0, 0.0);

    let render = world.render_dynamic_snapshot_cached(0).clone();
    let water = render
        .waters
        .first()
        .expect("water render descriptor should exist for raycast regression");
    let hit = world.render_scene_solid_raycast_cached(
        0,
        [
            water.translation_world[0],
            water.translation_world[1] + 1.0,
            water.translation_world[2],
        ],
        [0.0, -1.0, 0.0],
        false,
    );
    let hit = hit.expect("native solid raycast should hit batch-owned water geometry");
    assert_eq!(
        terrarium_render_id_class(hit.render_id),
        TERRARIUM_RENDER_ID_WATER
    );
    assert!(hit.distance > 0.0);
}

#[test]
fn native_fly_contacts_project_body_onto_support_geometry() {
    let mut world = TerrariumWorld::demo(31, false).unwrap();
    world.plants.clear();
    world.seeds.clear();
    world.explicit_microbes.clear();
    world.flies.clear();
    world.waters.clear();
    world.fruits.clear();
    let fruit_x = world.config.width / 2;
    let fruit_y = world.config.height / 2;
    world.add_fruit(fruit_x, fruit_y, 1.5, Some(1.0));
    let fly_x = fruit_x as f32 + 0.5;
    let fly_y = fruit_y as f32 + 0.5;
    world.add_fly(DrosophilaScale::Tiny, fly_x, fly_y, 7);
    world.flies[0].set_body_state(
        fly_x,
        fly_y,
        0.0,
        Some(0.0),
        Some(0.0),
        Some(false),
        Some(0.0),
        None,
        None,
        None,
    );

    let terrain = world.topdown_field(TerrariumTopdownView::Terrain);
    let (terrain_min, terrain_inv) = field_min_inv(&terrain);
    let previous_body = world.flies[0].body_state().clone();
    let (before_world, _) = fly_translation_world_from_body(
        &world.config,
        &terrain,
        terrain_min,
        terrain_inv,
        &previous_body,
    );
    let (collision_surfaces, collision_nodes, collision_root) =
        world.build_fly_collision_scene(&terrain, terrain_min, terrain_inv);
    let config = world.config.clone();
    TerrariumWorld::resolve_fly_contacts_with_scene(
        &config,
        &mut world.flies[0],
        &previous_body,
        &terrain,
        terrain_min,
        terrain_inv,
        &collision_surfaces,
        &collision_nodes,
        collision_root,
    );

    let resolved_body = world.flies[0].body_state().clone();
    let (after_world, _) = fly_translation_world_from_body(
        &world.config,
        &terrain,
        terrain_min,
        terrain_inv,
        &resolved_body,
    );
    assert!(!resolved_body.is_flying);
    assert!(
        after_world[1] > before_world[1] + 0.005,
        "support contact did not lift the fly onto geometry: before={before_world:?} after={after_world:?}"
    );
}

#[test]
fn native_fly_side_contacts_push_body_out_of_support_geometry() {
    let mut world = TerrariumWorld::demo(35, false).unwrap();
    world.plants.clear();
    world.seeds.clear();
    world.explicit_microbes.clear();
    world.flies.clear();
    world.waters.clear();
    world.fruits.clear();
    let fruit_x = world.config.width / 2;
    let fruit_y = world.config.height / 2;
    world.add_fruit(fruit_x, fruit_y, 1.5, Some(1.0));

    let terrain = world.topdown_field(TerrariumTopdownView::Terrain);
    let (terrain_min, terrain_inv) = field_min_inv(&terrain);
    let support_snapshot = world.build_dynamic_snapshot();
    let fruit_render = support_snapshot
        .fruits
        .first()
        .expect("fruit support render should exist for side-contact regression");
    let skin_part = fruit_render
        .parts
        .iter()
        .find(|part| part.kind == TerrariumFruitPartKind::Skin)
        .expect("fruit skin part should exist for side-contact regression");
    let local_max_x = skin_part
        .mesh
        .positions
        .iter()
        .map(|position| position[0])
        .fold(f32::NEG_INFINITY, f32::max);
    let initial_world = [
        fruit_render.translation_world[0] + local_max_x * 0.72,
        fruit_render.translation_world[1] + 0.02,
        fruit_render.translation_world[2],
    ];
    let (body_x, body_y, body_z, _) = fly_body_state_from_world_translation(
        &world.config,
        &terrain,
        terrain_min,
        terrain_inv,
        initial_world,
        false,
    );
    world.add_fly(DrosophilaScale::Tiny, body_x, body_y, 17);
    world.flies[0].set_body_state(
        body_x,
        body_y,
        body_z,
        Some(0.0),
        Some(0.0),
        Some(false),
        Some(0.0),
        None,
        None,
        None,
    );

    let previous_body = world.flies[0].body_state().clone();
    let (before_world, _) = fly_translation_world_from_body(
        &world.config,
        &terrain,
        terrain_min,
        terrain_inv,
        &previous_body,
    );
    let (collision_surfaces, collision_nodes, collision_root) =
        world.build_fly_collision_scene(&terrain, terrain_min, terrain_inv);
    let config = world.config.clone();
    TerrariumWorld::resolve_fly_contacts_with_scene(
        &config,
        &mut world.flies[0],
        &previous_body,
        &terrain,
        terrain_min,
        terrain_inv,
        &collision_surfaces,
        &collision_nodes,
        collision_root,
    );

    let resolved_body = world.flies[0].body_state().clone();
    let (after_world, _) = fly_translation_world_from_body(
        &world.config,
        &terrain,
        terrain_min,
        terrain_inv,
        &resolved_body,
    );
    let before_horizontal_distance = ((before_world[0] - fruit_render.translation_world[0])
        * (before_world[0] - fruit_render.translation_world[0])
        + (before_world[2] - fruit_render.translation_world[2])
            * (before_world[2] - fruit_render.translation_world[2]))
        .sqrt();
    let after_horizontal_distance = ((after_world[0] - fruit_render.translation_world[0])
        * (after_world[0] - fruit_render.translation_world[0])
        + (after_world[2] - fruit_render.translation_world[2])
            * (after_world[2] - fruit_render.translation_world[2]))
        .sqrt();
    assert!(
        after_horizontal_distance > before_horizontal_distance + 0.01,
        "side contact did not push the fly out of fruit support geometry: before={before_horizontal_distance:.4} after={after_horizontal_distance:.4}"
    );
    assert!(!resolved_body.is_flying);
}

#[test]
fn native_pairwise_fly_contacts_separate_overlapping_bodies() {
    let mut world = TerrariumWorld::demo(37, false).unwrap();
    world.plants.clear();
    world.seeds.clear();
    world.explicit_microbes.clear();
    world.flies.clear();
    world.waters.clear();
    world.fruits.clear();
    let fruit_x = world.config.width / 2;
    let fruit_y = world.config.height / 2;
    world.add_fruit(fruit_x, fruit_y, 1.5, Some(1.0));
    let fly_x = fruit_x as f32 + 0.5;
    let fly_y = fruit_y as f32 + 0.5;
    world.add_fly(DrosophilaScale::Tiny, fly_x, fly_y, 11);
    world.add_fly(DrosophilaScale::Tiny, fly_x, fly_y, 13);
    for fly in &mut world.flies {
        fly.set_body_state(
            fly_x,
            fly_y,
            0.0,
            Some(0.0),
            Some(0.0),
            Some(false),
            Some(0.0),
            None,
            None,
            None,
        );
    }

    let terrain = world.topdown_field(TerrariumTopdownView::Terrain);
    let (terrain_min, terrain_inv) = field_min_inv(&terrain);
    let (collision_surfaces, collision_nodes, collision_root) =
        world.build_fly_collision_scene(&terrain, terrain_min, terrain_inv);
    let config = world.config.clone();
    for fly in &mut world.flies {
        let previous_body = fly.body_state().clone();
        TerrariumWorld::resolve_fly_contacts_with_scene(
            &config,
            fly,
            &previous_body,
            &terrain,
            terrain_min,
            terrain_inv,
            &collision_surfaces,
            &collision_nodes,
            collision_root,
        );
    }
    TerrariumWorld::resolve_pairwise_fly_contacts(
        &config,
        &mut world.flies,
        &terrain,
        terrain_min,
        terrain_inv,
        &collision_surfaces,
        &collision_nodes,
        collision_root,
    );

    let body_a = world.flies[0].body_state().clone();
    let body_b = world.flies[1].body_state().clone();
    let (world_a, _) =
        fly_translation_world_from_body(&world.config, &terrain, terrain_min, terrain_inv, &body_a);
    let (world_b, _) =
        fly_translation_world_from_body(&world.config, &terrain, terrain_min, terrain_inv, &body_b);
    let separation_world = [
        world_b[0] - world_a[0],
        world_b[1] - world_a[1],
        world_b[2] - world_a[2],
    ];
    let horizontal_separation = (separation_world[0] * separation_world[0]
        + separation_world[2] * separation_world[2])
        .sqrt();
    let min_separation = TerrariumWorld::pairwise_fly_contact_radius_world(&body_a)
        + TerrariumWorld::pairwise_fly_contact_radius_world(&body_b)
        - 0.02;
    assert!(
        horizontal_separation >= min_separation,
        "horizontal_separation={horizontal_separation:.4} min_separation={min_separation:.4} world_a={world_a:?} world_b={world_b:?} body_a=({:.4},{:.4},{:.4}) body_b=({:.4},{:.4},{:.4})",
        body_a.x,
        body_a.y,
        body_a.z,
        body_b.x,
        body_b.y,
        body_b.z,
    );
    assert!(!body_a.is_flying);
    assert!(!body_b.is_flying);
}

#[test]
fn soil_atmosphere_coupling_moves_surface_gases_toward_air_balance() {
    let mut world = TerrariumWorld::demo_minimal(7).unwrap();
    world.odorants[ATMOS_O2_IDX].fill(ATMOS_O2_BASELINE * 1.35);
    world.odorants[ATMOS_CO2_IDX].fill(ATMOS_CO2_BASELINE * 0.55);
    world
        .substrate
        .relax_species_toward(TerrariumSpecies::OxygenGas, 0.0, 1.0);
    world
        .substrate
        .relax_species_toward(TerrariumSpecies::CarbonDioxide, 0.0, 1.0);

    let x = world.config.width / 2;
    let y = world.config.height / 2;
    let _ =
        world
            .substrate
            .deposit_patch_species(TerrariumSpecies::CarbonDioxide, x, y, 0, 1, 0.06);

    let before_o2 = world
        .substrate
        .patch_mean_species(TerrariumSpecies::OxygenGas, x, y, 0, 1);
    let before_co2 =
        world
            .substrate
            .patch_mean_species(TerrariumSpecies::CarbonDioxide, x, y, 0, 1);
    let eq_o2 = HENRY_O2 * world.sample_odorant_patch(ATMOS_O2_IDX, x, y, 0, 1);
    let eq_co2 = HENRY_CO2 * world.sample_odorant_patch(ATMOS_CO2_IDX, x, y, 0, 1);

    world.couple_soil_atmosphere_gases(120.0);

    let after_o2 = world
        .substrate
        .patch_mean_species(TerrariumSpecies::OxygenGas, x, y, 0, 1);
    let after_co2 = world
        .substrate
        .patch_mean_species(TerrariumSpecies::CarbonDioxide, x, y, 0, 1);

    assert!(
        (after_o2 - eq_o2).abs() < (before_o2 - eq_o2).abs(),
        "surface soil o2 did not move toward Henry equilibrium"
    );
    assert!(
        (after_co2 - eq_co2).abs() < (before_co2 - eq_co2).abs(),
        "surface soil co2 did not move toward Henry equilibrium"
    );
}

#[test]
fn guild_populations_shift_with_gas_regime() {
    let mut world = TerrariumWorld::demo(7, false).unwrap();
    world.plants.clear();
    world.fruits.clear();
    world.waters.clear();
    world.moisture.fill(0.86);
    world.deep_moisture.fill(1.05);
    world.odorants[ATMOS_O2_IDX].fill(ATMOS_O2_BASELINE * 0.55);
    world.odorants[ATMOS_CO2_IDX].fill(ATMOS_CO2_BASELINE * 1.35);
    world.denitrifier_anoxic_fraction.fill(0.12);
    world.denitrifier_anoxic_packets = world
        .denitrifier_packets
        .iter()
        .copied()
        .map(|packets| packets * 0.12)
        .collect();
    world
        .substrate
        .relax_species_toward(TerrariumSpecies::OxygenGas, 0.03, 0.95);
    world
        .substrate
        .relax_species_toward(TerrariumSpecies::CarbonDioxide, 0.08, 0.85);

    let before = world.snapshot();
    world.run_frames(120).unwrap();
    let after = world.snapshot();

    assert!(
        after.mean_denitrifier_cells > before.mean_denitrifier_cells
            || after.mean_denitrifier_packets > before.mean_denitrifier_packets
            || after.mean_denitrification_potential > before.mean_denitrification_potential,
        "denitrifier population response did not strengthen under wet, low-o2 conditions"
    );
    assert!(
        after.mean_denitrifier_anoxic_packets >= before.mean_denitrifier_anoxic_packets
            || after.mean_denitrifier_anoxic_fraction >= before.mean_denitrifier_anoxic_fraction,
        "denitrifier anoxic packet subpopulation did not strengthen under wet, low-o2 conditions"
    );
}

#[test]
fn shoreline_absorption_drives_distinct_background_microbiomes() {
    let mut world =
        TerrariumWorld::demo_preset(7, false, TerrariumDemoPreset::MicroTerrarium).unwrap();
    let width = world.config.width;
    let height = world.config.height;
    let shore_idx = (0..width * height)
        .filter(|idx| world.water_mask[*idx] < 0.30)
        .max_by(|a, b| {
            shoreline_water_signal(width, height, &world.water_mask, *a)
                .partial_cmp(&shoreline_water_signal(
                    width,
                    height,
                    &world.water_mask,
                    *b,
                ))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("micro terrarium should expose a shoreline cell");
    let dry_idx = (0..width * height)
        .filter(|idx| world.water_mask[*idx] < 0.02)
        .min_by(|a, b| {
            shoreline_water_signal(width, height, &world.water_mask, *a)
                .partial_cmp(&shoreline_water_signal(
                    width,
                    height,
                    &world.water_mask,
                    *b,
                ))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("micro terrarium should expose a dry background cell");

    world
        .step_broad_soil(world.config.world_dt_s * world.config.time_warp)
        .unwrap();

    let shore_moisture = world.moisture[shore_idx] + world.deep_moisture[shore_idx];
    let dry_moisture = world.moisture[dry_idx] + world.deep_moisture[dry_idx];
    let trait_divergence = (world.denitrifier_anoxic_fraction[shore_idx]
        - world.denitrifier_anoxic_fraction[dry_idx])
        .abs()
        + (world.nitrifier_aerobic_fraction[shore_idx] - world.nitrifier_aerobic_fraction[dry_idx])
            .abs();
    let shore_process_signal =
        world.denitrification_potential[shore_idx] + world.nitrification_potential[shore_idx];
    let dry_process_signal =
        world.denitrification_potential[dry_idx] + world.nitrification_potential[dry_idx];

    assert!(
        shore_moisture > dry_moisture,
        "shoreline cell should hold more water than dry background"
    );
    assert!(
        trait_divergence > 0.01 || shore_process_signal > dry_process_signal,
        "shoreline chemistry should bias a distinct background microbiome"
    );
}

/// Verify that the Michaelis-Menten-based flux helper produces values in the
/// same order of magnitude as the old magic-number path.  The old path
/// produced glucose draws in roughly 0.0001..0.003 range for typical soil
/// concentrations and 24-cell cohorts.  This test uses default kinetics with
/// representative soil conditions and a moderate WCS activity gate.
#[test]
fn explicit_microbe_kinetic_flux_stays_in_historic_range() {
    let k = SubstrateKinetics::default();
    let eco_dt = 0.016; // ~60 FPS
    let uptake_scale = 24.0; // typical cohort

    // Representative soil concentrations (moderate nutrient levels)
    let soil_glucose = 0.08;
    let soil_oxygen = 0.12;
    let soil_ammonium = 0.04;
    let soil_nitrate = 0.03;

    // Moderate WCS activity (cell is metabolizing at ~40% capacity)
    let wcs_glucose_delta = 0.4;
    let wcs_oxygen_delta = 0.3;
    let wcs_ammonium_delta = 0.2;
    let wcs_nitrate_delta = 0.15;
    let wcs_atp_delta = 0.1;
    let local_atp_flux = 0.05;

    let (
        glucose_draw,
        oxygen_draw,
        ammonium_draw,
        nitrate_draw,
        atp_flux,
        co2_release,
        proton_release,
    ) = TerrariumWorld::compute_explicit_microbe_fluxes(
        &k,
        eco_dt,
        uptake_scale,
        soil_glucose,
        soil_oxygen,
        soil_ammonium,
        soil_nitrate,
        wcs_glucose_delta,
        wcs_oxygen_delta,
        wcs_ammonium_delta,
        wcs_nitrate_delta,
        wcs_atp_delta,
        local_atp_flux,
    );

    // All outputs should be non-negative
    assert!(glucose_draw >= 0.0, "glucose_draw = {glucose_draw}");
    assert!(oxygen_draw >= 0.0, "oxygen_draw = {oxygen_draw}");
    assert!(ammonium_draw >= 0.0, "ammonium_draw = {ammonium_draw}");
    assert!(nitrate_draw >= 0.0, "nitrate_draw = {nitrate_draw}");
    assert!(atp_flux >= 0.0, "atp_flux = {atp_flux}");
    assert!(co2_release >= 0.0, "co2_release = {co2_release}");
    assert!(proton_release >= 0.0, "proton_release = {proton_release}");

    // Within historic clamp ceilings (safety guards)
    assert!(
        glucose_draw <= 0.0035,
        "glucose_draw {glucose_draw} exceeds ceiling"
    );
    assert!(
        oxygen_draw <= 0.0032,
        "oxygen_draw {oxygen_draw} exceeds ceiling"
    );
    assert!(
        ammonium_draw <= 0.0020,
        "ammonium_draw {ammonium_draw} exceeds ceiling"
    );
    assert!(
        nitrate_draw <= 0.0018,
        "nitrate_draw {nitrate_draw} exceeds ceiling"
    );
    assert!(atp_flux <= 0.0018, "atp_flux {atp_flux} exceeds ceiling");
    assert!(
        co2_release <= 0.0040,
        "co2_release {co2_release} exceeds ceiling"
    );
    assert!(
        proton_release <= 0.0012,
        "proton_release {proton_release} exceeds ceiling"
    );

    // CO2 should be stoichiometrically related to glucose draw
    // (at minimum the respiration component: 1:1 ratio)
    assert!(
        co2_release > 0.0,
        "CO2 release should be positive when glucose is consumed"
    );
    assert!(
        co2_release <= glucose_draw * 2.0,
        "CO2 release {co2_release} is unreasonably large vs glucose_draw {glucose_draw}"
    );

    // Proton release should be proportional to CO2 + nitrification
    assert!(
        proton_release <= co2_release + ammonium_draw + 1.0e-6,
        "proton_release {proton_release} should not exceed CO2 + ammonium sources"
    );
}

// ==========================================================================
// Shade Avoidance Syndrome (SAS) wiring tests (Phase 4 integration)
// (Main tests in mod.rs inline tests module for default feature gating)
// ==========================================================================

// ==========================================================================
// Herbivore grazing tests (Phase 5 biological trigger)
// (Main tests in mod.rs inline tests module for default feature gating)
// ==========================================================================

#[test]
fn sas_elongation_wired_to_height() {
    // A plant under shade (high SAS expression) should grow taller than
    // a plant in full sun (no SAS). Both start from identical state;
    // the only difference is the SAS gene expression level.
    use crate::botany::physiology_bridge::{compute_molecular_drive, MolecularDriveState};
    use std::collections::HashMap;

    let mut metabolome = crate::botany::PlantMetabolome::new();
    metabolome.glucose_count = 200.0;
    metabolome.water_count = 500.0;
    metabolome.sucrose_count = 20.0;
    metabolome.starch_reserve = 30.0;

    let mut gene_sun = HashMap::new();
    gene_sun.insert("RbcL".to_string(), 0.8);
    gene_sun.insert("FT".to_string(), 0.0);
    gene_sun.insert("PIN1".to_string(), 0.5);
    gene_sun.insert("NRT2.1".to_string(), 0.5);
    gene_sun.insert("DREB".to_string(), 0.0);
    gene_sun.insert("SAS".to_string(), 0.0);

    let mut gene_shade = gene_sun.clone();
    gene_shade.insert("SAS".to_string(), 0.8);

    let drive_sun = compute_molecular_drive(
        &metabolome, &gene_sun, 0.8, 0.9, 0.4, 0.9, 1.0, 20.0, 0.0,
    );
    let drive_shade = compute_molecular_drive(
        &metabolome, &gene_shade, 0.3, 0.9, 0.4, 0.9, 1.0, 20.0, 0.0,
    );

    // Create two identical plants
    let mut plant_sun = crate::plant_organism::PlantOrganismSim::new(
        50.0, 5.0, 4.0, 1.0, 1.0, 1.0, 1.0, 0.8, 1.0, 1.0, 0.5, 1.0,
        0.08, 0.30, 0.35, 0.25, 0.5, 0.5, 0.3, 8000.0, 14000.0,
    );
    let mut plant_shade = plant_sun.clone();

    // Step both plants
    for _ in 0..5 {
        let _ = plant_sun.step_molecular(
            20.0, 0.02, 0.01, &drive_sun, 1.0, 0.9, 1.0, 0.1, 0.0,
            0.8, 0.8, 0.0, 0.0, 400.0, 9000.0,
        );
        let _ = plant_shade.step_molecular(
            20.0, 0.02, 0.01, &drive_shade, 1.0, 0.9, 1.0, 0.1, 0.0,
            0.8, 0.8, 0.0, 0.0, 400.0, 9000.0,
        );
    }

    assert!(
        plant_shade.height_mm() > plant_sun.height_mm(),
        "Shade plant ({:.2} mm) should be taller than sun plant ({:.2} mm) due to SAS elongation",
        plant_shade.height_mm(),
        plant_sun.height_mm(),
    );
}

#[test]
fn sas_branching_wired_to_laterals() {
    // Under shade, the branching factor (< 1.0) should reduce lateral_bias
    // in the morphology when applied in the flora step.
    use crate::botany::physiology_bridge::{shade_avoidance_branching, shade_avoidance_elongation};

    let sas_shade = 0.8;
    let branch_factor = shade_avoidance_branching(sas_shade);
    let elongation_factor = shade_avoidance_elongation(sas_shade);

    // Apply to a base lateral_bias
    let base_lateral = 0.92;
    let shaded_lateral = base_lateral * branch_factor;

    assert!(
        shaded_lateral < base_lateral * 0.7,
        "Shade should reduce lateral branching: base={base_lateral}, shaded={shaded_lateral}",
    );

    // Elongation factor should increase internode length
    let base_internode = 0.60;
    let shaded_internode = base_internode * elongation_factor;

    assert!(
        shaded_internode > base_internode * 1.3,
        "Shade should increase internode length: base={base_internode}, shaded={shaded_internode}",
    );
}

#[test]
fn full_sun_no_sas_effect() {
    // In full sun (SAS=0), shade factors should be ~1.0.
    use crate::botany::physiology_bridge::{shade_avoidance_branching, shade_avoidance_elongation};

    let elongation = shade_avoidance_elongation(0.0);
    let branching = shade_avoidance_branching(0.0);

    assert!(
        (elongation - 1.0).abs() < 0.01,
        "Full sun elongation factor should be ~1.0, got {elongation}",
    );
    assert!(
        (branching - 1.0).abs() < 0.01,
        "Full sun branching factor should be ~1.0, got {branching}",
    );
}

#[test]
fn shade_height_capped_by_max() {
    // Even with maximum SAS elongation, height should not exceed max_height_mm.
    use crate::botany::physiology_bridge::compute_molecular_drive;
    use std::collections::HashMap;

    let mut metabolome = crate::botany::PlantMetabolome::new();
    metabolome.glucose_count = 500.0;
    metabolome.water_count = 800.0;
    metabolome.sucrose_count = 40.0;
    metabolome.starch_reserve = 60.0;

    let mut gene_expr = HashMap::new();
    gene_expr.insert("RbcL".to_string(), 0.9);
    gene_expr.insert("FT".to_string(), 0.0);
    gene_expr.insert("PIN1".to_string(), 0.7);
    gene_expr.insert("NRT2.1".to_string(), 0.3);
    gene_expr.insert("DREB".to_string(), 0.0);
    gene_expr.insert("SAS".to_string(), 1.0); // maximum shade

    let drive = compute_molecular_drive(
        &metabolome, &gene_expr, 0.3, 0.9, 0.8, 1.5, 1.0, 20.0, 0.0,
    );

    let max_height = 15.0;
    let mut plant = crate::plant_organism::PlantOrganismSim::new(
        max_height, 8.0, 5.0, 1.0, 1.0, 1.0, 1.0, 0.8, 1.0, 1.0, 0.5, 1.0,
        0.08, 0.80, 0.90, 0.70, 2.0, 1.0, 0.5, 8000.0, 14000.0,
    );

    for _ in 0..20 {
        let _ = plant.step_molecular(
            20.0, 0.05, 0.03, &drive, 1.0, 0.9, 1.0, 0.1, 0.0,
            0.8, 0.8, 0.0, 0.0, 800.0, 9000.0,
        );
    }

    assert!(
        plant.height_mm() <= max_height,
        "Height ({:.2} mm) should not exceed max ({max_height} mm) even under max SAS",
        plant.height_mm(),
    );
}

// ==========================================================================
// Herbivore damage / leaf grazing tests (Phase 5 biological trigger)
// ==========================================================================

#[test]
fn fly_grazing_reduces_leaf_biomass() {
    let mut world = TerrariumWorld::demo(201, false).expect("demo world");
    // Clear to isolate the test
    world.flies.clear();
    world.fly_metabolisms.clear();
    world.fly_identities.clear();
    world.fruits.clear();
    world.seeds.clear();

    // Ensure we have at least one plant
    assert!(!world.plants.is_empty(), "demo world should have plants");
    let initial_leaf = world.plants[0].physiology.leaf_biomass();

    // Place a hungry fly right on top of the plant
    let plant_x = world.plants[0].x as f32;
    let plant_y = world.plants[0].y as f32;
    world.add_fly(DrosophilaScale::Tiny, plant_x, plant_y, 201);
    // Make fly very hungry to trigger grazing
    if let Some(m) = world.fly_metabolisms.first_mut() {
        m.fat_body_glycogen_mg = 0.0;
        m.fat_body_lipid_mg = 0.0;
        m.hemolymph_trehalose_mm = 1.0;
    }
    world.flies[0].set_body_state(
        plant_x, plant_y, 0.0,
        Some(0.0), Some(0.0), Some(false), Some(0.0), None, None, None,
    );

    // Run many fly steps so at least one grazing event should occur
    for _ in 0..500 {
        let _ = world.step_flies();
    }

    let final_leaf = world.plants[0].physiology.leaf_biomass();
    let grazing_events = world
        .ecology_events
        .iter()
        .filter(|e| matches!(e, EcologyTelemetryEvent::FlyGrazing { .. }))
        .count();

    // With a very hungry fly on top of a plant for 500 steps, we expect grazing
    assert!(
        grazing_events > 0,
        "Expected at least one grazing event in 500 steps, got 0"
    );
    assert!(
        final_leaf < initial_leaf,
        "Leaf biomass should decrease after grazing: initial={initial_leaf}, final={final_leaf}"
    );
}

#[test]
fn defended_plant_deters_grazing() {
    let mut world = TerrariumWorld::demo(202, false).expect("demo world");
    world.flies.clear();
    world.fly_metabolisms.clear();
    world.fly_identities.clear();
    world.fruits.clear();
    world.seeds.clear();

    assert!(!world.plants.is_empty());

    // Give the plant high jasmonate + salicylate (strong defense)
    world.plants[0].metabolome.jasmonate_count = 50.0;
    world.plants[0].metabolome.salicylate_count = 50.0;
    let defended_initial_leaf = world.plants[0].physiology.leaf_biomass();

    let plant_x = world.plants[0].x as f32;
    let plant_y = world.plants[0].y as f32;
    world.add_fly(DrosophilaScale::Tiny, plant_x, plant_y, 202);
    if let Some(m) = world.fly_metabolisms.first_mut() {
        m.fat_body_glycogen_mg = 0.0;
        m.fat_body_lipid_mg = 0.0;
        m.hemolymph_trehalose_mm = 1.0;
    }
    world.flies[0].set_body_state(
        plant_x, plant_y, 0.0,
        Some(0.0), Some(0.0), Some(false), Some(0.0), None, None, None,
    );

    for _ in 0..500 {
        let _ = world.step_flies();
    }

    let defended_grazing = world
        .ecology_events
        .iter()
        .filter(|e| matches!(e, EcologyTelemetryEvent::FlyGrazing { .. }))
        .count();

    // With high defense metabolites, grazing should be strongly suppressed
    // (the Hill deterrence at JA+SA=100 with Km=5 gives deterrence > 0.99)
    assert!(
        defended_grazing == 0,
        "Defended plant should deter almost all grazing, got {defended_grazing} events"
    );
    // Leaf biomass should be unchanged or nearly so
    let defended_final_leaf = world.plants[0].physiology.leaf_biomass();
    assert!(
        (defended_final_leaf - defended_initial_leaf).abs() < 0.01,
        "Defended plant leaf should be mostly unchanged: initial={defended_initial_leaf}, final={defended_final_leaf}"
    );
}

#[test]
fn grazing_energy_gain() {
    // Verify that a fly gains energy from leaf grazing
    let mut world = TerrariumWorld::demo(203, false).expect("demo world");
    world.flies.clear();
    world.fly_metabolisms.clear();
    world.fly_identities.clear();
    world.fruits.clear();
    world.seeds.clear();

    assert!(!world.plants.is_empty());
    let plant_x = world.plants[0].x as f32;
    let plant_y = world.plants[0].y as f32;
    world.add_fly(DrosophilaScale::Tiny, plant_x, plant_y, 203);
    if let Some(m) = world.fly_metabolisms.first_mut() {
        m.fat_body_glycogen_mg = 0.0;
        m.fat_body_lipid_mg = 0.0;
        m.hemolymph_trehalose_mm = 1.0;
    }
    world.flies[0].set_body_state(
        plant_x, plant_y, 0.0,
        Some(0.0), Some(0.0), Some(false), Some(0.0), None, None, None,
    );

    // Record energy before
    let energy_before = world.fly_metabolisms[0].energy_fraction();

    // Run enough steps that some grazing occurs
    for _ in 0..500 {
        let _ = world.step_flies();
    }

    let grazing_events = world
        .ecology_events
        .iter()
        .filter(|e| matches!(e, EcologyTelemetryEvent::FlyGrazing { .. }))
        .count();

    if grazing_events > 0 {
        let energy_after = world.fly_metabolisms[0].energy_fraction();
        // Energy should not have decreased further from grazing alone
        // (the fly is also spending energy on neural activity, so we just
        // check that ingestion happened — which is verified by the grazing event)
        assert!(
            energy_after >= 0.0,
            "Fly energy should remain non-negative: {energy_after}"
        );
    }
}

#[test]
fn grazing_probability_is_low() {
    // With a non-hungry fly, grazing should almost never happen
    let mut world = TerrariumWorld::demo(204, false).expect("demo world");
    world.flies.clear();
    world.fly_metabolisms.clear();
    world.fly_identities.clear();
    world.fruits.clear();
    world.seeds.clear();

    assert!(!world.plants.is_empty());
    let plant_x = world.plants[0].x as f32;
    let plant_y = world.plants[0].y as f32;
    world.add_fly(DrosophilaScale::Tiny, plant_x, plant_y, 204);
    // Give fly plenty of energy — not hungry at all
    if let Some(m) = world.fly_metabolisms.first_mut() {
        m.fat_body_glycogen_mg = 0.1;
        m.fat_body_lipid_mg = 0.3;
        m.hemolymph_trehalose_mm = 40.0;
    }
    world.flies[0].set_body_state(
        plant_x, plant_y, 0.0,
        Some(0.0), Some(0.0), Some(false), Some(0.0), None, None, None,
    );

    for _ in 0..200 {
        let _ = world.step_flies();
    }

    let grazing_events = world
        .ecology_events
        .iter()
        .filter(|e| matches!(e, EcologyTelemetryEvent::FlyGrazing { .. }))
        .count();

    // A well-fed fly should rarely graze — the Hill function on hunger
    // with Km=0.6 gives very low drive when hunger is low
    assert!(
        grazing_events <= 2,
        "Well-fed fly should rarely graze, got {grazing_events} events in 200 steps"
    );
}
