//! Pure compiled-asset and non-runtime fallback reducers extracted from
//! `whole_cell.rs`.
//!
//! These helpers keep registry-drive and operon-asset fallback weighting out of
//! the main simulator implementation.

use crate::whole_cell_data::{
    WholeCellGenomeAssetPackage, WholeCellGenomeFeature, WholeCellGenomeProcessRegistry,
    WholeCellOperonSpec, WholeCellProcessWeights, WholeCellReactionRuntimeState,
    WholeCellSpeciesRuntimeState,
};
use crate::whole_cell_process_weights::{explicit_projection_weights, runtime_projection_weights};

const REGISTRY_SPECIES_DRIVE_SCALE: f32 = 0.28;
const REGISTRY_REACTION_DRIVE_SCALE: f32 = 0.80;

pub(crate) fn registry_process_drive_from_runtime(
    species: &[WholeCellSpeciesRuntimeState],
    reactions: &[WholeCellReactionRuntimeState],
) -> WholeCellProcessWeights {
    let mut drive = WholeCellProcessWeights::default();

    for species in species {
        let abundance_weight = REGISTRY_SPECIES_DRIVE_SCALE * species.count.max(0.0).sqrt();
        drive.add_weighted(
            runtime_projection_weights(species.process_weights, &species.subsystem_targets),
            abundance_weight,
        );
    }

    for reaction in reactions {
        let reaction_weight = REGISTRY_REACTION_DRIVE_SCALE * reaction.current_flux.max(0.0);
        drive.add_weighted(
            runtime_projection_weights(reaction.process_weights, &reaction.subsystem_targets),
            reaction_weight,
        );
    }

    drive
}

pub(crate) fn registry_process_drive_from_registry(
    registry: &WholeCellGenomeProcessRegistry,
) -> WholeCellProcessWeights {
    let mut drive = WholeCellProcessWeights::default();

    for species in &registry.species {
        let abundance_weight =
            REGISTRY_SPECIES_DRIVE_SCALE * species.basal_abundance.max(0.0).sqrt();
        drive.add_weighted(
            explicit_projection_weights(species.process_weights, &species.subsystem_targets),
            abundance_weight,
        );
    }

    for reaction in &registry.reactions {
        let reaction_weight = REGISTRY_REACTION_DRIVE_SCALE * reaction.nominal_rate.max(0.0);
        drive.add_weighted(
            explicit_projection_weights(reaction.process_weights, &reaction.subsystem_targets),
            reaction_weight,
        );
    }

    drive
}

pub(crate) fn operon_process_weights_from_assets(
    operon: &WholeCellOperonSpec,
    organism_genes: Option<&[WholeCellGenomeFeature]>,
    assets: Option<&WholeCellGenomeAssetPackage>,
) -> WholeCellProcessWeights {
    let mut weights = operon.process_weights.clamped();
    if weights.total() > 1.0e-6 {
        return weights;
    }
    if let Some(genes) = organism_genes {
        for gene in &operon.genes {
            if let Some(feature) = genes.iter().find(|feature| feature.gene == *gene) {
                weights.add_weighted(
                    feature.process_weights,
                    0.35 * feature.basal_expression.max(0.1),
                );
            }
        }
    }
    if weights.total() <= 1.0e-6 {
        if let Some(assets) = assets {
            for rna in assets.rnas.iter().filter(|rna| rna.operon == operon.name) {
                weights.add_weighted(
                    explicit_projection_weights(rna.process_weights, &[]),
                    rna.basal_abundance.max(0.1),
                );
            }
            for protein in assets
                .proteins
                .iter()
                .filter(|protein| protein.operon == operon.name)
            {
                weights.add_weighted(
                    explicit_projection_weights(
                        protein.process_weights,
                        &protein.subsystem_targets,
                    ),
                    protein.basal_abundance.max(0.1),
                );
            }
            for complex in assets
                .complexes
                .iter()
                .filter(|complex| complex.operon == operon.name)
            {
                weights.add_weighted(
                    explicit_projection_weights(
                        complex.process_weights,
                        &complex.subsystem_targets,
                    ),
                    complex.basal_abundance.max(0.1),
                );
            }
        }
    }
    weights.clamped()
}

#[cfg(all(test, feature = "satellite_tests"))]
mod tests {
    use super::*;
    use crate::whole_cell_data::{
        WholeCellAssetClass, WholeCellReactionClass, WholeCellSpeciesClass,
    };
    use crate::whole_cell_submodels::Syn3ASubsystemPreset;

    #[test]
    fn registry_process_drive_from_runtime_uses_explicit_runtime_channels() {
        let species = vec![WholeCellSpeciesRuntimeState {
            id: "s".to_string(),
            name: "s".to_string(),
            species_class: WholeCellSpeciesClass::Protein,
            compartment: "cyto".to_string(),
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
            count: 16.0,
            anchor_count: 0.0,
            synthesis_rate: 0.0,
            turnover_rate: 0.0,
        }];
        let reactions = vec![WholeCellReactionRuntimeState {
            id: "r".to_string(),
            name: "r".to_string(),
            reaction_class: WholeCellReactionClass::Transcription,
            asset_class: WholeCellAssetClass::Generic,
            nominal_rate: 0.0,
            process_weights: WholeCellProcessWeights {
                transcription: 1.0,
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
            current_flux: 2.0,
            cumulative_extent: 0.0,
            reactant_satisfaction: 1.0,
            catalyst_support: 1.0,
        }];

        let drive = registry_process_drive_from_runtime(&species, &reactions);

        assert!(drive.translation > 0.0);
        assert!(drive.transcription > 0.0);
        assert!(drive.energy.abs() < 1.0e-6);
    }

    #[test]
    fn operon_process_weights_from_assets_uses_subsystem_targets_without_asset_class_fallback() {
        let operon = WholeCellOperonSpec {
            name: "op".to_string(),
            genes: vec!["g".to_string()],
            promoter_bp: 0,
            terminator_bp: 1,
            basal_activity: 1.0,
            polycistronic: false,
            process_weights: WholeCellProcessWeights::default(),
            subsystem_targets: Vec::new(),
            asset_class: None,
            complex_family: None,
        };
        let assets = WholeCellGenomeAssetPackage {
            organism: "test".to_string(),
            chromosome_length_bp: 1,
            origin_bp: 0,
            terminus_bp: 0,
            chromosome_domains: Vec::new(),
            operons: vec![operon.clone()],
            operon_semantics: Vec::new(),
            rnas: Vec::new(),
            proteins: vec![crate::whole_cell_data::WholeCellProteinProductSpec {
                id: "p".to_string(),
                gene: "g".to_string(),
                operon: "op".to_string(),
                rna_id: "r".to_string(),
                aa_length: 10,
                basal_abundance: 2.0,
                translation_cost: 1.0,
                nucleotide_cost: 1.0,
                asset_class: WholeCellAssetClass::Membrane,
                process_weights: WholeCellProcessWeights::default(),
                subsystem_targets: vec![Syn3ASubsystemPreset::RibosomePolysomeCluster],
            }],
            protein_semantics: Vec::new(),
            complex_semantics: Vec::new(),
            complexes: Vec::new(),
            pools: Vec::new(),
        };

        let weights = operon_process_weights_from_assets(&operon, None, Some(&assets));

        assert!(weights.translation > 0.0);
        assert!(weights.membrane.abs() < 1.0e-6);
    }
}
