//! Pure scale and operon-signal reducer helpers extracted from `whole_cell.rs`.
//!
//! These helpers are formula-only and do not depend on simulator state.

use crate::whole_cell_data::WholeCellOperonState;

pub(crate) fn operon_inventory_projection_available(operon: &WholeCellOperonState) -> bool {
    (operon.basal_activity - 1.0).abs() > 1.0e-6
        || (operon.effective_activity - 1.0).abs() > 1.0e-6
        || (operon.support_level - 1.0).abs() > 1.0e-6
        || (operon.stress_penalty - 1.0).abs() > 1.0e-6
        || (operon.fork_support - 1.0).abs() > 1.0e-6
        || operon.fork_pressure.abs() > 1.0e-6
        || operon.transcript_abundance.abs() > 1.0e-6
        || operon.protein_abundance.abs() > 1.0e-6
        || operon.transcript_synthesis_rate.abs() > 1.0e-6
        || operon.protein_synthesis_rate.abs() > 1.0e-6
        || operon.transcript_turnover_rate.abs() > 1.0e-6
        || operon.protein_turnover_rate.abs() > 1.0e-6
}

pub(crate) fn operon_direct_state_available(
    runtime_has_signal: bool,
    reaction_has_signal: bool,
) -> bool {
    runtime_has_signal || reaction_has_signal
}

pub(crate) fn operon_direct_transcription_available(
    runtime_has_transcript_state: bool,
    reaction_transcription_projection: f32,
) -> bool {
    runtime_has_transcript_state || reaction_transcription_projection > 1.0e-6
}

pub(crate) fn operon_direct_elongation_available(
    runtime_has_protein_state: bool,
    runtime_assembly_projection: f32,
    reaction_translation_projection: f32,
    reaction_assembly_projection: f32,
) -> bool {
    runtime_has_protein_state
        || runtime_assembly_projection > 1.0e-6
        || reaction_translation_projection > 1.0e-6
        || reaction_assembly_projection > 1.0e-6
}

pub(crate) fn finite_scale(value: f32, fallback: f32, min_value: f32, max_value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(min_value, max_value)
    } else {
        fallback.clamp(min_value, max_value)
    }
}

pub(crate) fn process_scale(signal: f32, mean_signal: f32) -> f32 {
    if mean_signal <= 1.0e-6 {
        1.0
    } else {
        (0.82 + 0.28 * (signal / mean_signal)).clamp(0.68, 1.45)
    }
}

pub(crate) fn runtime_process_scale_projection(
    prior: f32,
    capacity: f32,
    live_signal: f32,
    support: f32,
    localized_supply: f32,
    global_activity: f32,
    degradation_penalty: f32,
    bias: f32,
    capacity_mean: f32,
    live_signal_mean: f32,
    min_value: f32,
    max_value: f32,
) -> f32 {
    let capacity_scale = process_scale(capacity, capacity_mean);
    let live_scale = process_scale(live_signal, live_signal_mean);
    let support = support.clamp(0.55, 1.80);
    let localized_supply = localized_supply.clamp(0.55, 1.10);
    let global_activity = global_activity.clamp(0.50, 1.80);
    let bias = bias.clamp(0.60, 1.40);
    let degradation_penalty = degradation_penalty.clamp(0.0, 1.35);
    finite_scale(
        prior
            * (0.36
                + 0.16 * support
                + 0.08 * localized_supply
                + 0.06 * global_activity
                + 0.10 * bias)
            + 0.16 * capacity_scale
            + 0.12 * live_scale
            - 0.05 * degradation_penalty,
        prior,
        min_value,
        max_value,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_scale_uses_fallback_for_non_finite_input() {
        let scaled = finite_scale(f32::NAN, 1.2, 0.5, 2.0);
        assert!((scaled - 1.2).abs() < 1.0e-6);
    }

    #[test]
    fn process_scale_defaults_to_one_without_mean_signal() {
        assert!((process_scale(5.0, 0.0) - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn runtime_process_scale_projection_responds_to_capacity_and_live_signal() {
        let projected = runtime_process_scale_projection(
            1.0, 3.0, 4.0, 1.1, 0.9, 1.2, 0.1, 1.0, 2.0, 2.0, 0.5, 2.0,
        );
        assert!(projected > 1.0);
        assert!(projected <= 2.0);
    }

    #[test]
    fn operon_direct_elongation_available_detects_assembly_only_signal() {
        assert!(operon_direct_elongation_available(false, 0.0, 0.0, 0.2));
        assert!(!operon_direct_elongation_available(false, 0.0, 0.0, 0.0));
    }

    #[test]
    fn operon_inventory_projection_available_detects_non_default_projection() {
        let operon = WholeCellOperonState {
            name: "test".to_string(),
            promoter_bp: 0,
            terminator_bp: 10,
            gene_count: 1,
            copy_gain: 1.0,
            basal_activity: 1.0,
            effective_activity: 1.0,
            support_level: 1.0,
            stress_penalty: 1.0,
            fork_support: 1.0,
            fork_pressure: 0.0,
            promoter_accessibility: 1.0,
            rnap_occupancy: 0.0,
            elongation_occupancy: 0.0,
            blockage_pressure: 0.0,
            transcript_abundance: 3.0,
            protein_abundance: 0.0,
            transcript_synthesis_rate: 0.0,
            protein_synthesis_rate: 0.0,
            transcript_turnover_rate: 0.0,
            protein_turnover_rate: 0.0,
            strand_alignment: 0.5,
        };

        assert!(operon_inventory_projection_available(&operon));
    }
}
