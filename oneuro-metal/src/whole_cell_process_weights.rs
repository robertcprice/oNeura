//! Process-weight and scale projection helpers extracted from `whole_cell.rs`.
//!
//! These are pure reducers used by the whole-cell runtime. Keeping them in a
//! dedicated module helps limit growth of the main simulator file.

use crate::whole_cell::WholeCellOrganismProcessScales;
use crate::whole_cell_data::{WholeCellAssetClass, WholeCellComplexSpec, WholeCellProcessWeights};
use crate::whole_cell_submodels::Syn3ASubsystemPreset;

pub(crate) fn named_complex_projection_weights(
    complex: &WholeCellComplexSpec,
) -> WholeCellProcessWeights {
    projection_weights(
        complex.process_weights,
        complex.asset_class,
        &complex.subsystem_targets,
    )
}

pub(crate) fn projection_weights(
    process_weights: WholeCellProcessWeights,
    asset_class: WholeCellAssetClass,
    subsystem_targets: &[Syn3ASubsystemPreset],
) -> WholeCellProcessWeights {
    let mut weights =
        subsystem_augmented_process_weights(process_weights.clamped(), subsystem_targets);
    if weights.total() <= 1.0e-6 {
        weights = asset_class_process_template(asset_class);
    }
    weights
}

pub(crate) fn runtime_projection_weights(
    process_weights: WholeCellProcessWeights,
    subsystem_targets: &[Syn3ASubsystemPreset],
) -> WholeCellProcessWeights {
    explicit_projection_weights(process_weights, subsystem_targets)
}

pub(crate) fn explicit_projection_weights(
    process_weights: WholeCellProcessWeights,
    subsystem_targets: &[Syn3ASubsystemPreset],
) -> WholeCellProcessWeights {
    subsystem_augmented_process_weights(process_weights.clamped(), subsystem_targets)
}

pub(crate) fn subsystem_augmented_process_weights(
    mut weights: WholeCellProcessWeights,
    subsystem_targets: &[Syn3ASubsystemPreset],
) -> WholeCellProcessWeights {
    if subsystem_targets.contains(&Syn3ASubsystemPreset::AtpSynthaseMembraneBand) {
        weights.energy = weights.energy.max(1.0);
    }
    if subsystem_targets.contains(&Syn3ASubsystemPreset::RibosomePolysomeCluster) {
        weights.translation = weights.translation.max(1.0);
    }
    if subsystem_targets.contains(&Syn3ASubsystemPreset::ReplisomeTrack) {
        weights.replication = weights.replication.max(1.0);
    }
    if subsystem_targets.contains(&Syn3ASubsystemPreset::FtsZSeptumRing) {
        weights.constriction = weights.constriction.max(1.0);
    }
    weights
}

pub(crate) fn process_scale_from_weights(
    scales: WholeCellOrganismProcessScales,
    process_weights: WholeCellProcessWeights,
    asset_class: WholeCellAssetClass,
    subsystem_targets: &[Syn3ASubsystemPreset],
) -> f32 {
    let weights = projection_weights(process_weights, asset_class, subsystem_targets);
    let total = weights.total();
    if total <= 1.0e-6 {
        1.0
    } else {
        (weights.energy * scales.energy_scale
            + weights.transcription * scales.transcription_scale
            + weights.translation * scales.translation_scale
            + weights.replication * scales.replication_scale
            + weights.segregation * scales.segregation_scale
            + weights.membrane * scales.membrane_scale
            + weights.constriction * scales.constriction_scale)
            / total
    }
}

pub(crate) fn asset_class_process_template(
    asset_class: WholeCellAssetClass,
) -> WholeCellProcessWeights {
    match asset_class {
        WholeCellAssetClass::Energy => WholeCellProcessWeights {
            energy: 1.0,
            ..WholeCellProcessWeights::default()
        },
        WholeCellAssetClass::Translation => WholeCellProcessWeights {
            translation: 1.0,
            ..WholeCellProcessWeights::default()
        },
        WholeCellAssetClass::Replication => WholeCellProcessWeights {
            replication: 1.0,
            ..WholeCellProcessWeights::default()
        },
        WholeCellAssetClass::Segregation => WholeCellProcessWeights {
            segregation: 1.0,
            ..WholeCellProcessWeights::default()
        },
        WholeCellAssetClass::Membrane => WholeCellProcessWeights {
            membrane: 1.0,
            ..WholeCellProcessWeights::default()
        },
        WholeCellAssetClass::Constriction => WholeCellProcessWeights {
            constriction: 1.0,
            ..WholeCellProcessWeights::default()
        },
        WholeCellAssetClass::QualityControl => WholeCellProcessWeights {
            translation: 1.0,
            ..WholeCellProcessWeights::default()
        },
        WholeCellAssetClass::Homeostasis => WholeCellProcessWeights {
            transcription: 1.0,
            ..WholeCellProcessWeights::default()
        },
        WholeCellAssetClass::Generic => WholeCellProcessWeights::default(),
    }
}

pub(crate) fn process_weights_mean_signal(weights: WholeCellProcessWeights) -> f32 {
    (weights.energy
        + weights.transcription
        + weights.translation
        + weights.replication
        + weights.segregation
        + weights.membrane
        + weights.constriction)
        / 7.0
}

#[cfg(all(test, feature = "satellite_tests"))]
mod tests {
    use super::*;

    #[test]
    fn test_process_scale_projection_prefers_explicit_process_weights_over_asset_class_scale() {
        let scales = WholeCellOrganismProcessScales {
            energy_scale: 0.72,
            transcription_scale: 0.84,
            translation_scale: 1.82,
            replication_scale: 1.16,
            segregation_scale: 0.93,
            membrane_scale: 0.58,
            constriction_scale: 0.76,
            amino_cost_scale: 1.0,
            nucleotide_cost_scale: 1.0,
        };

        let explicit = process_scale_from_weights(
            scales,
            WholeCellProcessWeights {
                translation: 1.0,
                ..WholeCellProcessWeights::default()
            },
            WholeCellAssetClass::Membrane,
            &[],
        );
        let fallback = process_scale_from_weights(
            scales,
            WholeCellProcessWeights::default(),
            WholeCellAssetClass::Membrane,
            &[],
        );

        assert!((explicit - scales.translation_scale).abs() < 1.0e-6);
        assert!((fallback - scales.membrane_scale).abs() < 1.0e-6);
        assert!((explicit - fallback).abs() > 1.0e-3);
    }

    #[test]
    fn test_process_scale_projection_uses_subsystem_targets_before_asset_class_fallback() {
        let scales = WholeCellOrganismProcessScales {
            energy_scale: 0.74,
            transcription_scale: 0.88,
            translation_scale: 1.76,
            replication_scale: 1.10,
            segregation_scale: 0.95,
            membrane_scale: 0.62,
            constriction_scale: 0.80,
            amino_cost_scale: 1.0,
            nucleotide_cost_scale: 1.0,
        };

        let targeted = process_scale_from_weights(
            scales,
            WholeCellProcessWeights::default(),
            WholeCellAssetClass::Generic,
            &[Syn3ASubsystemPreset::RibosomePolysomeCluster],
        );
        let untargeted = process_scale_from_weights(
            scales,
            WholeCellProcessWeights::default(),
            WholeCellAssetClass::Generic,
            &[],
        );

        assert!((targeted - scales.translation_scale).abs() < 1.0e-6);
        assert!((untargeted - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn test_process_scale_projection_uses_subsystem_targets_before_conflicting_asset_class() {
        let scales = WholeCellOrganismProcessScales {
            energy_scale: 0.74,
            transcription_scale: 0.88,
            translation_scale: 1.76,
            replication_scale: 1.10,
            segregation_scale: 0.95,
            membrane_scale: 0.62,
            constriction_scale: 0.80,
            amino_cost_scale: 1.0,
            nucleotide_cost_scale: 1.0,
        };

        let targeted = process_scale_from_weights(
            scales,
            WholeCellProcessWeights::default(),
            WholeCellAssetClass::Membrane,
            &[Syn3ASubsystemPreset::RibosomePolysomeCluster],
        );
        let fallback = process_scale_from_weights(
            scales,
            WholeCellProcessWeights::default(),
            WholeCellAssetClass::Membrane,
            &[],
        );

        assert!((targeted - scales.translation_scale).abs() < 1.0e-6);
        assert!((fallback - scales.membrane_scale).abs() < 1.0e-6);
    }

    #[test]
    fn extracted_projection_helpers_return_expected_values() {
        let weights = WholeCellProcessWeights {
            membrane: 1.0,
            ..WholeCellProcessWeights::default()
        };

        assert_eq!(
            projection_weights(weights, WholeCellAssetClass::Generic, &[]),
            weights
        );
    }
}
