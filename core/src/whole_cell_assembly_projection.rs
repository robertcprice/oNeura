//! Pure assembly-inventory projection helpers extracted from `whole_cell.rs`.
//!
//! These reducers convert named-complex and runtime process signals into
//! aggregate assembly inventory surfaces without depending on simulator state.

use crate::whole_cell_data::{
    WholeCellComplexAssemblyState, WholeCellNamedComplexState, WholeCellProcessWeights,
    WholeCellReactionRuntimeState, WholeCellSpeciesClass, WholeCellSpeciesRuntimeState,
};
use crate::whole_cell_inventory_authority::assembly_inventory_projection_available;
use crate::whole_cell_process_weights::runtime_projection_weights;

pub(crate) fn named_complex_effective_projection(
    state: &WholeCellNamedComplexState,
) -> (f32, f32, f32, f32) {
    let effective_abundance = state.abundance
        + 0.35 * state.elongation_intermediate
        + 0.15 * state.nucleation_intermediate;
    let effective_target = state.target_abundance
        + 0.25 * state.elongation_intermediate
        + 0.10 * state.nucleation_intermediate;
    let effective_assembly_rate = state.assembly_rate
        + 0.55 * state.maturation_rate
        + 0.30 * state.elongation_rate
        + 0.15 * state.nucleation_rate;
    let effective_degradation_rate = state.degradation_rate
        + 0.20 * state.nucleation_intermediate
        + 0.15 * state.elongation_intermediate;
    (
        effective_abundance.max(0.0),
        effective_target.max(0.0),
        effective_assembly_rate.max(0.0),
        effective_degradation_rate.max(0.0),
    )
}

pub(crate) fn accumulate_assembly_inventory_projection(
    inventory: &mut WholeCellComplexAssemblyState,
    weights: WholeCellProcessWeights,
    current: f32,
    target: f32,
    assembly_rate: f32,
    degradation_rate: f32,
) {
    let replisome_weight = (weights.replication + 0.35 * weights.segregation).max(0.0);
    let dnaa_weight =
        (0.72 * weights.replication + 0.18 * weights.transcription + 0.10 * weights.segregation)
            .max(0.0);

    inventory.atp_band_complexes += current * weights.energy;
    inventory.ribosome_complexes += current * weights.translation;
    inventory.rnap_complexes += current * weights.transcription;
    inventory.replisome_complexes += current * replisome_weight;
    inventory.membrane_complexes += current * weights.membrane;
    inventory.ftsz_polymer += current * weights.constriction;
    inventory.dnaa_activity += current * dnaa_weight;

    inventory.atp_band_target += target * weights.energy;
    inventory.ribosome_target += target * weights.translation;
    inventory.rnap_target += target * weights.transcription;
    inventory.replisome_target += target * replisome_weight;
    inventory.membrane_target += target * weights.membrane;
    inventory.ftsz_target += target * weights.constriction;
    inventory.dnaa_target += target * dnaa_weight;

    inventory.atp_band_assembly_rate += assembly_rate * weights.energy;
    inventory.ribosome_assembly_rate += assembly_rate * weights.translation;
    inventory.rnap_assembly_rate += assembly_rate * weights.transcription;
    inventory.replisome_assembly_rate += assembly_rate * replisome_weight;
    inventory.membrane_assembly_rate += assembly_rate * weights.membrane;
    inventory.ftsz_assembly_rate += assembly_rate * weights.constriction;
    inventory.dnaa_assembly_rate += assembly_rate * dnaa_weight;

    inventory.atp_band_degradation_rate += degradation_rate * weights.energy;
    inventory.ribosome_degradation_rate += degradation_rate * weights.translation;
    inventory.rnap_degradation_rate += degradation_rate * weights.transcription;
    inventory.replisome_degradation_rate += degradation_rate * replisome_weight;
    inventory.membrane_degradation_rate += degradation_rate * weights.membrane;
    inventory.ftsz_degradation_rate += degradation_rate * weights.constriction;
    inventory.dnaa_degradation_rate += degradation_rate * dnaa_weight;
}

pub(crate) fn runtime_species_projection_class_weight(species_class: WholeCellSpeciesClass) -> f32 {
    match species_class {
        WholeCellSpeciesClass::Pool => 0.08,
        WholeCellSpeciesClass::Rna => 0.24,
        WholeCellSpeciesClass::Protein => 0.28,
        WholeCellSpeciesClass::ComplexSubunitPool => 0.20,
        WholeCellSpeciesClass::ComplexNucleationIntermediate => 0.22,
        WholeCellSpeciesClass::ComplexElongationIntermediate => 0.24,
        WholeCellSpeciesClass::ComplexMature => 0.26,
    }
}

pub(crate) fn explicit_runtime_assembly_inventory(
    species: &[WholeCellSpeciesRuntimeState],
    reactions: &[WholeCellReactionRuntimeState],
) -> Option<WholeCellComplexAssemblyState> {
    let mut inventory = WholeCellComplexAssemblyState::default();

    for species in species {
        let class_weight = runtime_species_projection_class_weight(species.species_class);
        let weights =
            runtime_projection_weights(species.process_weights, &species.subsystem_targets);
        let current = class_weight * species.count.max(0.0).sqrt();
        let target = (class_weight * species.anchor_count.max(0.0).sqrt()).max(current);
        accumulate_assembly_inventory_projection(
            &mut inventory,
            weights,
            current,
            target,
            species.synthesis_rate.max(0.0),
            species.turnover_rate.max(0.0),
        );
    }

    for reaction in reactions {
        let weights =
            runtime_projection_weights(reaction.process_weights, &reaction.subsystem_targets);
        let flux = reaction.current_flux.max(0.0);
        let supported_flux =
            flux * reaction.reactant_satisfaction.clamp(0.0, 1.5) * reaction.catalyst_support;
        accumulate_assembly_inventory_projection(
            &mut inventory,
            weights,
            0.35 * supported_flux,
            reaction.nominal_rate.max(supported_flux),
            supported_flux,
            0.10 * supported_flux,
        );
    }

    assembly_inventory_projection_available(inventory).then_some(inventory)
}

#[cfg(all(test, feature = "satellite_tests"))]
mod tests {
    use super::*;
    use crate::whole_cell_data::{
        WholeCellAssemblyFamily, WholeCellAssetClass, WholeCellReactionClass,
    };

    #[test]
    fn named_complex_effective_projection_includes_intermediates() {
        let state = WholeCellNamedComplexState {
            id: "test_complex".to_string(),
            operon: "test_operon".to_string(),
            asset_class: WholeCellAssetClass::Generic,
            family: WholeCellAssemblyFamily::Generic,
            subsystem_targets: Vec::new(),
            abundance: 10.0,
            target_abundance: 12.0,
            subunit_pool: 0.0,
            nucleation_intermediate: 4.0,
            elongation_intermediate: 6.0,
            assembly_rate: 2.0,
            degradation_rate: 1.0,
            nucleation_rate: 3.0,
            elongation_rate: 5.0,
            maturation_rate: 7.0,
            component_satisfaction: 0.0,
            structural_support: 0.0,
            assembly_progress: 0.0,
            stalled_intermediate: 0.0,
            damaged_abundance: 0.0,
            limiting_component_signal: 0.0,
            shared_component_pressure: 0.0,
            insertion_progress: 0.0,
            failure_count: 0.0,
        };

        let (abundance, target, assembly_rate, degradation_rate) =
            named_complex_effective_projection(&state);

        assert!((abundance - 12.7).abs() < 1.0e-6);
        assert!((target - 13.9).abs() < 1.0e-6);
        assert!((assembly_rate - 7.8).abs() < 1.0e-6);
        assert!((degradation_rate - 2.7).abs() < 1.0e-6);
    }

    #[test]
    fn accumulate_assembly_inventory_projection_routes_weights_to_expected_channels() {
        let mut inventory = WholeCellComplexAssemblyState::default();
        let weights = WholeCellProcessWeights {
            energy: 0.5,
            transcription: 0.25,
            translation: 0.75,
            replication: 0.8,
            segregation: 0.2,
            membrane: 0.6,
            constriction: 0.4,
        };

        accumulate_assembly_inventory_projection(&mut inventory, weights, 2.0, 3.0, 4.0, 5.0);

        assert!((inventory.atp_band_complexes - 1.0).abs() < 1.0e-6);
        assert!((inventory.ribosome_target - 2.25).abs() < 1.0e-6);
        assert!((inventory.membrane_assembly_rate - 2.4).abs() < 1.0e-6);
        assert!((inventory.ftsz_degradation_rate - 2.0).abs() < 1.0e-6);
        assert!((inventory.replisome_complexes - 1.74).abs() < 1.0e-6);
        assert!((inventory.dnaa_target - 1.923).abs() < 1.0e-6);
    }

    #[test]
    fn explicit_runtime_assembly_inventory_projects_species_and_reactions() {
        let species = vec![
            WholeCellSpeciesRuntimeState {
                id: "rna".to_string(),
                name: "rna".to_string(),
                species_class: WholeCellSpeciesClass::Rna,
                compartment: "cytosol".to_string(),
                asset_class: WholeCellAssetClass::Generic,
                basal_abundance: 0.0,
                process_weights: WholeCellProcessWeights {
                    transcription: 1.0,
                    ..WholeCellProcessWeights::default()
                },
                bulk_field: None,
                operon: None,
                parent_complex: None,
                subsystem_targets: Vec::new(),
                chromosome_domain: None,
                patch_domain: Default::default(),
                spatial_scope: Default::default(),
                count: 16.0,
                anchor_count: 25.0,
                synthesis_rate: 3.0,
                turnover_rate: 1.0,
            },
            WholeCellSpeciesRuntimeState {
                id: "protein".to_string(),
                name: "protein".to_string(),
                species_class: WholeCellSpeciesClass::Protein,
                compartment: "cytosol".to_string(),
                asset_class: WholeCellAssetClass::Generic,
                basal_abundance: 0.0,
                process_weights: WholeCellProcessWeights {
                    translation: 1.0,
                    ..WholeCellProcessWeights::default()
                },
                bulk_field: None,
                operon: None,
                parent_complex: None,
                subsystem_targets: Vec::new(),
                chromosome_domain: None,
                patch_domain: Default::default(),
                spatial_scope: Default::default(),
                count: 9.0,
                anchor_count: 16.0,
                synthesis_rate: 2.0,
                turnover_rate: 0.5,
            },
        ];
        let reactions = vec![WholeCellReactionRuntimeState {
            id: "translation".to_string(),
            name: "translation".to_string(),
            reaction_class: WholeCellReactionClass::Translation,
            asset_class: WholeCellAssetClass::Translation,
            nominal_rate: 4.0,
            process_weights: WholeCellProcessWeights {
                translation: 1.0,
                ..WholeCellProcessWeights::default()
            },
            operon: None,
            catalyst: None,
            reactants: Vec::new(),
            products: Vec::new(),
            subsystem_targets: Vec::new(),
            chromosome_domain: None,
            patch_domain: Default::default(),
            spatial_scope: Default::default(),
            current_flux: 2.5,
            cumulative_extent: 0.0,
            reactant_satisfaction: 1.0,
            catalyst_support: 1.0,
        }];

        let inventory =
            explicit_runtime_assembly_inventory(&species, &reactions).expect("projected inventory");

        assert!(inventory.rnap_complexes > 0.0);
        assert!(inventory.ribosome_complexes > 0.0);
        assert!(inventory.ribosome_target >= inventory.ribosome_complexes);
        assert!(inventory.ribosome_assembly_rate > 0.0);
    }
}
