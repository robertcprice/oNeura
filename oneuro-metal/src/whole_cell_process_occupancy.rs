//! Pure process-occupancy projection helpers extracted from `whole_cell.rs`.
//!
//! These reducers keep live-vs-fallback occupancy merging and occupancy/inventory
//! projection logic out of the main simulator implementation.

use crate::whole_cell_assembly_projection::runtime_species_projection_class_weight;
use crate::whole_cell_data::{
    WholeCellComplexAssemblyState, WholeCellProcessOccupancyState, WholeCellProcessWeights,
    WholeCellReactionRuntimeState, WholeCellSpeciesRuntimeState,
};
use crate::whole_cell_process_weights::runtime_projection_weights;

pub(crate) fn process_occupancy_from_legacy_inventory(
    inventory: WholeCellComplexAssemblyState,
) -> WholeCellProcessOccupancyState {
    WholeCellProcessOccupancyState {
        current: WholeCellProcessWeights {
            energy: inventory.atp_band_complexes.max(0.0),
            transcription: inventory.rnap_complexes.max(0.0),
            translation: inventory.ribosome_complexes.max(0.0),
            replication: (0.78 * inventory.replisome_complexes + 0.22 * inventory.dnaa_activity)
                .max(0.0),
            segregation: (0.42 * inventory.replisome_complexes + 0.10 * inventory.dnaa_activity)
                .max(0.0),
            membrane: inventory.membrane_complexes.max(0.0),
            constriction: inventory.ftsz_polymer.max(0.0),
        },
        target: WholeCellProcessWeights {
            energy: inventory.atp_band_target.max(0.0),
            transcription: inventory.rnap_target.max(0.0),
            translation: inventory.ribosome_target.max(0.0),
            replication: (0.78 * inventory.replisome_target + 0.22 * inventory.dnaa_target)
                .max(0.0),
            segregation: (0.42 * inventory.replisome_target + 0.10 * inventory.dnaa_target)
                .max(0.0),
            membrane: inventory.membrane_target.max(0.0),
            constriction: inventory.ftsz_target.max(0.0),
        },
        assembly_rate: WholeCellProcessWeights {
            energy: inventory.atp_band_assembly_rate.max(0.0),
            transcription: inventory.rnap_assembly_rate.max(0.0),
            translation: inventory.ribosome_assembly_rate.max(0.0),
            replication: (0.78 * inventory.replisome_assembly_rate
                + 0.22 * inventory.dnaa_assembly_rate)
                .max(0.0),
            segregation: (0.42 * inventory.replisome_assembly_rate
                + 0.10 * inventory.dnaa_assembly_rate)
                .max(0.0),
            membrane: inventory.membrane_assembly_rate.max(0.0),
            constriction: inventory.ftsz_assembly_rate.max(0.0),
        },
        degradation_rate: WholeCellProcessWeights {
            energy: inventory.atp_band_degradation_rate.max(0.0),
            transcription: inventory.rnap_degradation_rate.max(0.0),
            translation: inventory.ribosome_degradation_rate.max(0.0),
            replication: (0.78 * inventory.replisome_degradation_rate
                + 0.22 * inventory.dnaa_degradation_rate)
                .max(0.0),
            segregation: (0.42 * inventory.replisome_degradation_rate
                + 0.10 * inventory.dnaa_degradation_rate)
                .max(0.0),
            membrane: inventory.membrane_degradation_rate.max(0.0),
            constriction: inventory.ftsz_degradation_rate.max(0.0),
        },
    }
}

pub(crate) fn prefer_live_process_weights(
    fallback: WholeCellProcessWeights,
    live: WholeCellProcessWeights,
) -> WholeCellProcessWeights {
    let fallback = fallback.clamped();
    let live = live.clamped();
    WholeCellProcessWeights {
        energy: if live.energy > 1.0e-6 {
            live.energy
        } else {
            fallback.energy
        },
        transcription: if live.transcription > 1.0e-6 {
            live.transcription
        } else {
            fallback.transcription
        },
        translation: if live.translation > 1.0e-6 {
            live.translation
        } else {
            fallback.translation
        },
        replication: if live.replication > 1.0e-6 {
            live.replication
        } else {
            fallback.replication
        },
        segregation: if live.segregation > 1.0e-6 {
            live.segregation
        } else {
            fallback.segregation
        },
        membrane: if live.membrane > 1.0e-6 {
            live.membrane
        } else {
            fallback.membrane
        },
        constriction: if live.constriction > 1.0e-6 {
            live.constriction
        } else {
            fallback.constriction
        },
    }
}

pub(crate) fn prefer_live_process_occupancy(
    fallback: WholeCellProcessOccupancyState,
    live: WholeCellProcessOccupancyState,
) -> WholeCellProcessOccupancyState {
    WholeCellProcessOccupancyState {
        current: prefer_live_process_weights(fallback.current, live.current),
        target: prefer_live_process_weights(fallback.target, live.target),
        assembly_rate: prefer_live_process_weights(fallback.assembly_rate, live.assembly_rate),
        degradation_rate: prefer_live_process_weights(
            fallback.degradation_rate,
            live.degradation_rate,
        ),
    }
}

#[cfg(test)]
pub(crate) fn legacy_inventory_from_process_occupancy(
    occupancy: WholeCellProcessOccupancyState,
) -> WholeCellComplexAssemblyState {
    WholeCellComplexAssemblyState {
        atp_band_complexes: (occupancy.current.energy
            + 0.15 * occupancy.current.membrane
            + 0.20 * occupancy.assembly_rate.energy)
            .max(0.0),
        ribosome_complexes: (occupancy.current.translation
            + 0.10 * occupancy.current.transcription
            + 0.25 * occupancy.assembly_rate.translation)
            .max(0.0),
        rnap_complexes: (occupancy.current.transcription
            + 0.10 * occupancy.current.translation
            + 0.25 * occupancy.assembly_rate.transcription)
            .max(0.0),
        replisome_complexes: (occupancy.current.replication
            + 0.70 * occupancy.current.segregation
            + 0.25 * occupancy.assembly_rate.replication
            + 0.15 * occupancy.assembly_rate.segregation)
            .max(0.0),
        membrane_complexes: (occupancy.current.membrane
            + 0.15 * occupancy.current.energy
            + 0.25 * occupancy.assembly_rate.membrane)
            .max(0.0),
        ftsz_polymer: (occupancy.current.constriction
            + 0.20 * occupancy.current.membrane
            + 0.35 * occupancy.assembly_rate.constriction
            + 0.20 * occupancy.assembly_rate.membrane)
            .max(0.0),
        dnaa_activity: (0.78 * occupancy.current.replication
            + 0.22 * occupancy.current.transcription
            + 0.20 * occupancy.assembly_rate.replication)
            .max(0.0),
        atp_band_target: (occupancy.target.energy + 0.15 * occupancy.target.membrane).max(0.0),
        ribosome_target: (occupancy.target.translation + 0.10 * occupancy.target.transcription)
            .max(0.0),
        rnap_target: (occupancy.target.transcription + 0.10 * occupancy.target.translation)
            .max(0.0),
        replisome_target: (occupancy.target.replication + 0.70 * occupancy.target.segregation)
            .max(0.0),
        membrane_target: (occupancy.target.membrane + 0.15 * occupancy.target.energy).max(0.0),
        ftsz_target: (occupancy.target.constriction + 0.20 * occupancy.target.membrane).max(0.0),
        dnaa_target: (0.78 * occupancy.target.replication + 0.22 * occupancy.target.transcription)
            .max(0.0),
        atp_band_assembly_rate: (occupancy.assembly_rate.energy
            + 0.15 * occupancy.assembly_rate.membrane)
            .max(0.0),
        ribosome_assembly_rate: (occupancy.assembly_rate.translation
            + 0.10 * occupancy.assembly_rate.transcription)
            .max(0.0),
        rnap_assembly_rate: (occupancy.assembly_rate.transcription
            + 0.10 * occupancy.assembly_rate.translation)
            .max(0.0),
        replisome_assembly_rate: (occupancy.assembly_rate.replication
            + 0.70 * occupancy.assembly_rate.segregation)
            .max(0.0),
        membrane_assembly_rate: (occupancy.assembly_rate.membrane
            + 0.15 * occupancy.assembly_rate.energy)
            .max(0.0),
        ftsz_assembly_rate: (occupancy.assembly_rate.constriction
            + 0.20 * occupancy.assembly_rate.membrane)
            .max(0.0),
        dnaa_assembly_rate: (0.78 * occupancy.assembly_rate.replication
            + 0.22 * occupancy.assembly_rate.transcription)
            .max(0.0),
        atp_band_degradation_rate: (occupancy.degradation_rate.energy
            + 0.15 * occupancy.degradation_rate.membrane)
            .max(0.0),
        ribosome_degradation_rate: (occupancy.degradation_rate.translation
            + 0.10 * occupancy.degradation_rate.transcription)
            .max(0.0),
        rnap_degradation_rate: (occupancy.degradation_rate.transcription
            + 0.10 * occupancy.degradation_rate.translation)
            .max(0.0),
        replisome_degradation_rate: (occupancy.degradation_rate.replication
            + 0.70 * occupancy.degradation_rate.segregation)
            .max(0.0),
        membrane_degradation_rate: (occupancy.degradation_rate.membrane
            + 0.15 * occupancy.degradation_rate.energy)
            .max(0.0),
        ftsz_degradation_rate: (occupancy.degradation_rate.constriction
            + 0.20 * occupancy.degradation_rate.membrane)
            .max(0.0),
        dnaa_degradation_rate: (0.78 * occupancy.degradation_rate.replication
            + 0.22 * occupancy.degradation_rate.transcription)
            .max(0.0),
    }
}

pub(crate) fn direct_capacity_inventory_projection(
    occupancy: WholeCellProcessOccupancyState,
) -> WholeCellComplexAssemblyState {
    let current = occupancy.current.clamped();
    let target = occupancy.target.clamped();
    let assembly_rate = occupancy.assembly_rate.clamped();
    let degradation_rate = occupancy.degradation_rate.clamped();
    WholeCellComplexAssemblyState {
        atp_band_complexes: current.energy,
        ribosome_complexes: current.translation,
        rnap_complexes: current.transcription,
        replisome_complexes: current.replication,
        membrane_complexes: current.membrane,
        ftsz_polymer: current.constriction,
        dnaa_activity: current.replication,
        atp_band_target: target.energy,
        ribosome_target: target.translation,
        rnap_target: target.transcription,
        replisome_target: target.replication,
        membrane_target: target.membrane,
        ftsz_target: target.constriction,
        dnaa_target: target.replication,
        atp_band_assembly_rate: assembly_rate.energy,
        ribosome_assembly_rate: assembly_rate.translation,
        rnap_assembly_rate: assembly_rate.transcription,
        replisome_assembly_rate: assembly_rate.replication,
        membrane_assembly_rate: assembly_rate.membrane,
        ftsz_assembly_rate: assembly_rate.constriction,
        dnaa_assembly_rate: assembly_rate.replication,
        atp_band_degradation_rate: degradation_rate.energy,
        ribosome_degradation_rate: degradation_rate.translation,
        rnap_degradation_rate: degradation_rate.transcription,
        replisome_degradation_rate: degradation_rate.replication,
        membrane_degradation_rate: degradation_rate.membrane,
        ftsz_degradation_rate: degradation_rate.constriction,
        dnaa_degradation_rate: degradation_rate.replication,
        ..WholeCellComplexAssemblyState::default()
    }
}

pub(crate) fn explicit_runtime_process_occupancy(
    species: &[WholeCellSpeciesRuntimeState],
    reactions: &[WholeCellReactionRuntimeState],
) -> WholeCellProcessOccupancyState {
    let mut occupancy = WholeCellProcessOccupancyState::default();

    for species in species {
        let class_weight = runtime_species_projection_class_weight(species.species_class);
        let weights =
            runtime_projection_weights(species.process_weights, &species.subsystem_targets);
        let current = class_weight * species.count.max(0.0).sqrt();
        let target = (class_weight * species.anchor_count.max(0.0).sqrt()).max(current);
        occupancy.add_projection(
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
        occupancy.add_projection(
            weights,
            0.35 * supported_flux,
            reaction.nominal_rate.max(supported_flux),
            supported_flux,
            0.10 * supported_flux,
        );
    }

    occupancy
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::whole_cell_data::{
        WholeCellAssetClass, WholeCellReactionClass, WholeCellSpeciesClass,
    };

    #[test]
    fn prefer_live_process_occupancy_uses_live_channels_selectively() {
        let fallback = WholeCellProcessOccupancyState {
            current: WholeCellProcessWeights {
                energy: 1.0,
                transcription: 2.0,
                translation: 3.0,
                replication: 4.0,
                segregation: 5.0,
                membrane: 6.0,
                constriction: 7.0,
            },
            ..WholeCellProcessOccupancyState::default()
        };
        let live = WholeCellProcessOccupancyState {
            current: WholeCellProcessWeights {
                energy: 0.0,
                transcription: 8.0,
                translation: 0.0,
                replication: 9.0,
                segregation: 0.0,
                membrane: 10.0,
                constriction: 0.0,
            },
            ..WholeCellProcessOccupancyState::default()
        };

        let merged = prefer_live_process_occupancy(fallback, live);

        assert!((merged.current.energy - 1.0).abs() < 1.0e-6);
        assert!((merged.current.transcription - 8.0).abs() < 1.0e-6);
        assert!((merged.current.translation - 3.0).abs() < 1.0e-6);
        assert!((merged.current.replication - 9.0).abs() < 1.0e-6);
        assert!((merged.current.membrane - 10.0).abs() < 1.0e-6);
    }

    #[test]
    fn direct_capacity_inventory_projection_keeps_channels_separate() {
        let occupancy = WholeCellProcessOccupancyState {
            current: WholeCellProcessWeights {
                energy: 4.0,
                transcription: 5.0,
                translation: 6.0,
                replication: 7.0,
                segregation: 8.0,
                membrane: 9.0,
                constriction: 10.0,
            },
            target: WholeCellProcessWeights {
                energy: 1.0,
                transcription: 1.5,
                translation: 2.0,
                replication: 2.5,
                segregation: 3.0,
                membrane: 3.5,
                constriction: 4.0,
            },
            assembly_rate: WholeCellProcessWeights {
                energy: 0.4,
                transcription: 0.5,
                translation: 0.6,
                replication: 0.7,
                segregation: 0.8,
                membrane: 0.9,
                constriction: 1.0,
            },
            degradation_rate: WholeCellProcessWeights {
                energy: 0.04,
                transcription: 0.05,
                translation: 0.06,
                replication: 0.07,
                segregation: 0.08,
                membrane: 0.09,
                constriction: 0.10,
            },
        };

        let projected = direct_capacity_inventory_projection(occupancy);

        assert!((projected.atp_band_complexes - 4.0).abs() < 1.0e-6);
        assert!((projected.rnap_complexes - 5.0).abs() < 1.0e-6);
        assert!((projected.ribosome_complexes - 6.0).abs() < 1.0e-6);
        assert!((projected.replisome_complexes - 7.0).abs() < 1.0e-6);
        assert!((projected.membrane_complexes - 9.0).abs() < 1.0e-6);
        assert!((projected.ftsz_polymer - 10.0).abs() < 1.0e-6);
        assert!((projected.dnaa_activity - 7.0).abs() < 1.0e-6);
        assert!((projected.atp_band_target - 1.0).abs() < 1.0e-6);
        assert!((projected.ribosome_target - 2.0).abs() < 1.0e-6);
        assert!((projected.replisome_target - 2.5).abs() < 1.0e-6);
        assert!((projected.ftsz_target - 4.0).abs() < 1.0e-6);
        assert!((projected.ribosome_assembly_rate - 0.6).abs() < 1.0e-6);
        assert!((projected.replisome_assembly_rate - 0.7).abs() < 1.0e-6);
        assert!((projected.membrane_degradation_rate - 0.09).abs() < 1.0e-6);
        assert!((projected.dnaa_degradation_rate - 0.07).abs() < 1.0e-6);
    }

    #[test]
    fn explicit_runtime_process_occupancy_projects_species_and_reactions() {
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
            current_flux: 2.5,
            cumulative_extent: 0.0,
            reactant_satisfaction: 1.0,
            catalyst_support: 1.0,
        }];

        let occupancy = explicit_runtime_process_occupancy(&species, &reactions);

        assert!(occupancy.current.transcription > 0.0);
        assert!(occupancy.current.translation > 0.0);
        assert!(occupancy.target.translation >= occupancy.current.translation);
        assert!(occupancy.assembly_rate.translation > 0.0);
        assert!(occupancy.degradation_rate.transcription > 0.0);
    }
}
