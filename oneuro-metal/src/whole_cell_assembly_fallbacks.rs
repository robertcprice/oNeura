//! Pure fallback assembly inventory, target, seeding, and channel-advance helpers extracted from
//! `whole_cell.rs`.
//!
//! These reducers keep non-runtime assembly scaffolding and target projection
//! out of the main simulator implementation.

use crate::whole_cell::{WholeCellOrganismProcessScales, WholeCellProcessFluxes};
use crate::whole_cell_complex_channels::complex_channel_step;
use crate::whole_cell_data::{WholeCellComplexAssemblyState, WholeCellProcessWeights};
use crate::whole_cell_scale_reducers::finite_scale;

#[derive(Debug, Clone, Copy)]
pub(crate) struct BareAssemblyTargetInputs {
    pub(crate) replicated_fraction: f32,
    pub(crate) quantum_energy: f32,
    pub(crate) quantum_translation: f32,
    pub(crate) quantum_nucleotide: f32,
    pub(crate) quantum_membrane: f32,
    pub(crate) quantum_segregation: f32,
    pub(crate) energy_signal: f32,
    pub(crate) transcription_signal: f32,
    pub(crate) translation_signal: f32,
    pub(crate) membrane_signal: f32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct BareLowerScaleAssemblyInventoryInputs {
    pub(crate) capacities: WholeCellProcessFluxes,
    pub(crate) localized_supply: f32,
    pub(crate) effective_metabolic_load: f32,
    pub(crate) chemistry_atp_support: f32,
    pub(crate) chemistry_translation_support: f32,
    pub(crate) chemistry_nucleotide_support: f32,
    pub(crate) chemistry_membrane_support: f32,
    pub(crate) quantum_oxphos_efficiency: f32,
    pub(crate) quantum_translation_efficiency: f32,
    pub(crate) quantum_nucleotide_efficiency: f32,
    pub(crate) quantum_membrane_efficiency: f32,
    pub(crate) quantum_segregation_efficiency: f32,
    pub(crate) replicated_fraction: f32,
    pub(crate) initiation_signal: f32,
    pub(crate) separation_signal: f32,
    pub(crate) explicit_division_state_available: bool,
    pub(crate) division_progress: f32,
    pub(crate) atp_mm: f32,
    pub(crate) glucose_mm: f32,
    pub(crate) nucleotides_mm: f32,
    pub(crate) amino_acids_mm: f32,
    pub(crate) membrane_precursors_mm: f32,
    pub(crate) oxygen_mm: f32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DerivedAssemblyTargetInputs {
    pub(crate) assembly_drive: WholeCellProcessWeights,
    pub(crate) mean_signal: f32,
    pub(crate) protein_signal: f32,
    pub(crate) transcript_signal: f32,
    pub(crate) localized_supply: f32,
    pub(crate) crowding: f32,
    pub(crate) replicated_fraction: f32,
    pub(crate) organism_scales: WholeCellOrganismProcessScales,
    pub(crate) expression_energy_support: f32,
    pub(crate) expression_translation_support: f32,
    pub(crate) expression_nucleotide_support: f32,
    pub(crate) expression_membrane_support: f32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ComplexAssemblyAdvanceInputs {
    pub(crate) energy_support: f32,
    pub(crate) transcription_support: f32,
    pub(crate) translation_support: f32,
    pub(crate) replication_support: f32,
    pub(crate) membrane_support: f32,
    pub(crate) constriction_support: f32,
    pub(crate) degradation_pressure: f32,
    pub(crate) dt_scale: f32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ComplexAssemblySupportInputs {
    pub(crate) dt: f32,
    pub(crate) reference_dt: f32,
    pub(crate) crowding_penalty: f32,
    pub(crate) effective_load: f32,
    pub(crate) expression_energy_support: f32,
    pub(crate) expression_translation_support: f32,
    pub(crate) expression_nucleotide_support: f32,
    pub(crate) expression_membrane_support: f32,
    pub(crate) chemistry_atp_support: f32,
    pub(crate) organism_scales: WholeCellOrganismProcessScales,
    pub(crate) replicated_fraction: f32,
}

pub(crate) fn complex_assembly_signal_scale(signal: f32, mean_signal: f32, fallback: f32) -> f32 {
    if mean_signal <= 1.0e-6 {
        fallback
    } else {
        finite_scale(0.84 + 0.32 * (signal / mean_signal), fallback, 0.60, 1.65)
    }
}

fn saturating_signal(value: f32, half_saturation: f32) -> f32 {
    let value = value.max(0.0);
    let half_saturation = half_saturation.max(1.0e-6);
    (value / (value + half_saturation)).clamp(0.0, 1.0)
}

pub(crate) fn bare_lower_scale_assembly_inventory(
    inputs: BareLowerScaleAssemblyInventoryInputs,
) -> WholeCellComplexAssemblyState {
    let load_gate = 1.0 / inputs.effective_metabolic_load.max(1.0);
    let energy_support = finite_scale(inputs.chemistry_atp_support, 1.0, 0.70, 1.50);
    let translation_support = finite_scale(inputs.chemistry_translation_support, 1.0, 0.70, 1.50);
    let nucleotide_support = finite_scale(inputs.chemistry_nucleotide_support, 1.0, 0.70, 1.50);
    let membrane_support = finite_scale(inputs.chemistry_membrane_support, 1.0, 0.70, 1.50);
    let quantum_energy = finite_scale(inputs.quantum_oxphos_efficiency, 1.0, 0.70, 1.50);
    let quantum_translation = finite_scale(inputs.quantum_translation_efficiency, 1.0, 0.70, 1.50);
    let quantum_nucleotide = finite_scale(inputs.quantum_nucleotide_efficiency, 1.0, 0.70, 1.50);
    let quantum_membrane = finite_scale(inputs.quantum_membrane_efficiency, 1.0, 0.70, 1.50);
    let quantum_segregation = finite_scale(inputs.quantum_segregation_efficiency, 1.0, 0.70, 1.50);
    let division_progress_signal = if inputs.explicit_division_state_available {
        0.0
    } else {
        inputs.division_progress.max(0.0)
    };
    let division_signal = saturating_signal(
        0.42 * inputs.replicated_fraction
            + 0.22 * inputs.separation_signal
            + 0.20 * division_progress_signal
            + 0.16 * inputs.initiation_signal,
        0.8,
    );
    let energy_signal = saturating_signal(
        0.68 * inputs.atp_mm.max(0.0) + 0.16 * inputs.glucose_mm.max(0.0) + 0.16 * energy_support,
        1.2,
    );
    let transcription_signal = saturating_signal(
        0.64 * inputs.nucleotides_mm.max(0.0)
            + 0.20 * inputs.localized_supply
            + 0.16 * nucleotide_support,
        1.0,
    );
    let translation_signal = saturating_signal(
        0.68 * inputs.amino_acids_mm.max(0.0) + 0.18 * translation_support + 0.14 * energy_support,
        1.1,
    );
    let membrane_signal = saturating_signal(
        0.70 * inputs.membrane_precursors_mm.max(0.0)
            + 0.18 * membrane_support
            + 0.12 * inputs.oxygen_mm.max(0.0),
        0.8,
    );

    WholeCellComplexAssemblyState {
        atp_band_complexes: (18.0
            * inputs.capacities.energy_capacity
            * (0.72 + 0.28 * energy_signal)
            * (0.84 + 0.16 * energy_support)
            * (0.84 + 0.16 * quantum_energy)
            * (0.86 + 0.14 * inputs.localized_supply))
            .clamp(4.0, 512.0),
        ribosome_complexes: (24.0
            * inputs.capacities.translation_capacity
            * (0.70 + 0.30 * translation_signal)
            * (0.84 + 0.16 * translation_support)
            * (0.84 + 0.16 * quantum_translation)
            * (0.88 + 0.12 * load_gate))
            .clamp(8.0, 640.0),
        rnap_complexes: (16.0
            * inputs.capacities.transcription_capacity
            * (0.72 + 0.28 * transcription_signal)
            * (0.84 + 0.16 * nucleotide_support)
            * (0.84 + 0.16 * quantum_nucleotide)
            * (0.88 + 0.12 * inputs.localized_supply))
            .clamp(6.0, 384.0),
        replisome_complexes: (8.0
            * inputs.capacities.replication_capacity
            * (0.74 + 0.26 * transcription_signal)
            * (0.80 + 0.20 * (1.0 - inputs.replicated_fraction))
            * (0.82 + 0.18 * inputs.initiation_signal))
            * (0.84 + 0.16 * quantum_nucleotide).clamp(2.0, 192.0),
        membrane_complexes: (16.0
            * inputs.capacities.membrane_capacity
            * (0.70 + 0.30 * membrane_signal)
            * (0.84 + 0.16 * membrane_support)
            * (0.86 + 0.14 * energy_support))
            * (0.84 + 0.16 * quantum_membrane).clamp(4.0, 384.0),
        ftsz_polymer: (28.0
            * inputs.capacities.constriction_capacity
            * (0.68 + 0.32 * membrane_signal)
            * (0.72 + 0.28 * division_signal)
            * (0.84 + 0.16 * inputs.separation_signal))
            * (0.82 + 0.18 * quantum_membrane)
            * (0.84 + 0.16 * quantum_segregation).clamp(8.0, 768.0),
        dnaa_activity: (12.0
            * inputs.capacities.replication_capacity
            * (0.70 + 0.30 * transcription_signal)
            * (0.82 + 0.18 * inputs.initiation_signal)
            * (0.82 + 0.18 * (1.0 - inputs.replicated_fraction)))
            * (0.84 + 0.16 * quantum_nucleotide).clamp(4.0, 256.0),
        ..WholeCellComplexAssemblyState::default()
    }
}

pub(crate) fn bare_assembly_target(
    prior: WholeCellComplexAssemblyState,
    inputs: BareAssemblyTargetInputs,
) -> WholeCellComplexAssemblyState {
    WholeCellComplexAssemblyState {
        atp_band_target: (prior.atp_band_complexes
            * (0.76 + 0.44 * inputs.energy_signal)
            * (0.84 + 0.16 * inputs.quantum_energy))
            .clamp(4.0, 512.0),
        ribosome_target: (prior.ribosome_complexes
            * (0.74 + 0.42 * inputs.translation_signal)
            * (0.84 + 0.16 * inputs.quantum_translation))
            .clamp(8.0, 640.0),
        rnap_target: (prior.rnap_complexes
            * (0.76 + 0.38 * inputs.transcription_signal)
            * (0.84 + 0.16 * inputs.quantum_nucleotide))
            .clamp(6.0, 384.0),
        replisome_target: (prior.replisome_complexes
            * (0.74 + 0.42 * inputs.transcription_signal)
            * (0.82 + 0.24 * (1.0 - inputs.replicated_fraction))
            * (0.84 + 0.16 * inputs.quantum_nucleotide))
            .clamp(2.0, 192.0),
        membrane_target: (prior.membrane_complexes
            * (0.74 + 0.40 * inputs.membrane_signal)
            * (0.84 + 0.16 * inputs.quantum_membrane))
            .clamp(4.0, 384.0),
        ftsz_target: (prior.ftsz_polymer
            * (0.72 + 0.36 * inputs.membrane_signal + 0.36 * inputs.replicated_fraction))
            * (0.82 + 0.18 * inputs.quantum_membrane)
            * (0.84 + 0.16 * inputs.quantum_segregation).clamp(8.0, 768.0),
        dnaa_target: (prior.dnaa_activity
            * (0.74 + 0.34 * inputs.transcription_signal)
            * (0.84 + 0.20 * (1.0 - inputs.replicated_fraction))
            * (0.84 + 0.16 * inputs.quantum_nucleotide))
            .clamp(4.0, 256.0),
        ..prior
    }
}

pub(crate) fn derived_assembly_target(
    prior: WholeCellComplexAssemblyState,
    inputs: DerivedAssemblyTargetInputs,
) -> WholeCellComplexAssemblyState {
    let energy_scale = finite_scale(
        0.45 * inputs.organism_scales.energy_scale
            + 0.30
                * complex_assembly_signal_scale(
                    inputs.assembly_drive.energy,
                    inputs.mean_signal,
                    1.0,
                )
            + 0.15 * inputs.expression_energy_support
            + 0.10 * inputs.localized_supply,
        1.0,
        0.55,
        1.85,
    );
    let transcription_scale = finite_scale(
        0.42 * inputs.organism_scales.transcription_scale
            + 0.30
                * complex_assembly_signal_scale(
                    inputs.assembly_drive.transcription,
                    inputs.mean_signal,
                    1.0,
                )
            + 0.18 * inputs.expression_translation_support
            + 0.10 * inputs.transcript_signal,
        1.0,
        0.55,
        1.85,
    );
    let translation_scale = finite_scale(
        0.40 * inputs.organism_scales.translation_scale
            + 0.34
                * complex_assembly_signal_scale(
                    inputs.assembly_drive.translation,
                    inputs.mean_signal,
                    1.0,
                )
            + 0.16 * inputs.expression_translation_support
            + 0.10 * inputs.protein_signal,
        1.0,
        0.55,
        1.95,
    );
    let replication_signal =
        0.5 * (inputs.assembly_drive.replication + inputs.assembly_drive.segregation);
    let replication_scale = finite_scale(
        0.38 * inputs.organism_scales.replication_scale
            + 0.26 * complex_assembly_signal_scale(replication_signal, inputs.mean_signal, 1.0)
            + 0.22 * inputs.expression_nucleotide_support
            + 0.14 * (1.0 - 0.35 * inputs.replicated_fraction).clamp(0.65, 1.15),
        1.0,
        0.55,
        1.95,
    );
    let membrane_scale = finite_scale(
        0.44 * inputs.organism_scales.membrane_scale
            + 0.30
                * complex_assembly_signal_scale(
                    inputs.assembly_drive.membrane,
                    inputs.mean_signal,
                    1.0,
                )
            + 0.16 * inputs.expression_membrane_support
            + 0.10 * inputs.protein_signal,
        1.0,
        0.55,
        1.95,
    );
    let constriction_scale = finite_scale(
        0.36 * inputs.organism_scales.constriction_scale
            + 0.26
                * complex_assembly_signal_scale(
                    inputs.assembly_drive.constriction,
                    inputs.mean_signal,
                    1.0,
                )
            + 0.18 * inputs.expression_membrane_support
            + 0.20 * (0.70 + 0.60 * inputs.replicated_fraction).clamp(0.70, 1.30),
        1.0,
        0.55,
        2.10,
    );
    let replication_window = (0.78 + 0.30 * (1.0 - inputs.replicated_fraction)).clamp(0.60, 1.20);
    let constriction_window = (0.65 + 0.60 * inputs.replicated_fraction).clamp(0.65, 1.35);

    WholeCellComplexAssemblyState {
        atp_band_complexes: prior.atp_band_complexes,
        ribosome_complexes: prior.ribosome_complexes,
        rnap_complexes: prior.rnap_complexes,
        replisome_complexes: prior.replisome_complexes,
        membrane_complexes: prior.membrane_complexes,
        ftsz_polymer: prior.ftsz_polymer,
        dnaa_activity: prior.dnaa_activity,
        atp_band_target: (prior.atp_band_complexes
            * energy_scale
            * inputs.crowding
            * (0.72 + 0.28 * inputs.protein_signal))
            .clamp(4.0, 512.0),
        ribosome_target: (prior.ribosome_complexes
            * translation_scale
            * inputs.crowding
            * (0.68 + 0.32 * inputs.protein_signal))
            .clamp(8.0, 640.0),
        rnap_target: (prior.rnap_complexes
            * transcription_scale
            * inputs.crowding
            * (0.72 + 0.28 * inputs.transcript_signal))
            .clamp(6.0, 384.0),
        replisome_target: (prior.replisome_complexes
            * replication_scale
            * replication_window
            * (0.72 + 0.28 * inputs.transcript_signal))
            .clamp(2.0, 192.0),
        membrane_target: (prior.membrane_complexes
            * membrane_scale
            * inputs.crowding
            * (0.72 + 0.28 * inputs.protein_signal))
            .clamp(4.0, 384.0),
        ftsz_target: (prior.ftsz_polymer
            * constriction_scale
            * constriction_window
            * (0.66 + 0.34 * inputs.protein_signal))
            .clamp(8.0, 768.0),
        dnaa_target: (prior.dnaa_activity
            * replication_scale
            * replication_window
            * (0.70 + 0.30 * inputs.transcript_signal))
            .clamp(4.0, 256.0),
        ..WholeCellComplexAssemblyState::default()
    }
}

pub(crate) fn seed_complex_assembly_from_target(
    target: WholeCellComplexAssemblyState,
) -> WholeCellComplexAssemblyState {
    WholeCellComplexAssemblyState {
        atp_band_complexes: target.atp_band_target,
        ribosome_complexes: target.ribosome_target,
        rnap_complexes: target.rnap_target,
        replisome_complexes: target.replisome_target,
        membrane_complexes: target.membrane_target,
        ftsz_polymer: target.ftsz_target,
        dnaa_activity: target.dnaa_target,
        ..target
    }
}

pub(crate) fn complex_assembly_advance_inputs(
    inputs: ComplexAssemblySupportInputs,
) -> ComplexAssemblyAdvanceInputs {
    let dt_scale = (inputs.dt / inputs.reference_dt.max(0.05)).clamp(0.5, 6.0);
    let crowding = inputs.crowding_penalty.clamp(0.65, 1.10);
    let degradation_pressure =
        (0.70 + 0.24 * (inputs.effective_load - 1.0).max(0.0) + 0.18 * (1.0 - crowding).max(0.0))
            .clamp(0.65, 1.80);

    ComplexAssemblyAdvanceInputs {
        energy_support: finite_scale(
            0.55 * inputs.expression_energy_support + 0.45 * inputs.chemistry_atp_support,
            1.0,
            0.55,
            1.55,
        ),
        transcription_support: finite_scale(
            0.55 * inputs.expression_translation_support
                + 0.45 * inputs.organism_scales.transcription_scale,
            1.0,
            0.55,
            1.60,
        ),
        translation_support: finite_scale(
            0.60 * inputs.expression_translation_support
                + 0.40 * inputs.organism_scales.translation_scale,
            1.0,
            0.55,
            1.60,
        ),
        replication_support: finite_scale(
            0.55 * inputs.expression_nucleotide_support
                + 0.45
                    * (0.5
                        * (inputs.organism_scales.replication_scale
                            + inputs.organism_scales.segregation_scale)),
            1.0,
            0.55,
            1.60,
        ),
        membrane_support: finite_scale(
            0.60 * inputs.expression_membrane_support
                + 0.40 * inputs.organism_scales.membrane_scale,
            1.0,
            0.55,
            1.60,
        ),
        constriction_support: finite_scale(
            0.50 * inputs.expression_membrane_support
                + 0.30 * inputs.organism_scales.constriction_scale
                + 0.20 * (0.70 + 0.60 * inputs.replicated_fraction),
            1.0,
            0.55,
            1.70,
        ),
        degradation_pressure,
        dt_scale,
    }
}

pub(crate) fn advance_complex_assembly_state(
    current: WholeCellComplexAssemblyState,
    target: WholeCellComplexAssemblyState,
    inputs: ComplexAssemblyAdvanceInputs,
) -> WholeCellComplexAssemblyState {
    let (atp_band_complexes, atp_band_assembly_rate, atp_band_degradation_rate) =
        complex_channel_step(
            current.atp_band_complexes,
            target.atp_band_target,
            inputs.energy_support,
            inputs.degradation_pressure,
            inputs.dt_scale,
            512.0,
        );
    let (ribosome_complexes, ribosome_assembly_rate, ribosome_degradation_rate) =
        complex_channel_step(
            current.ribosome_complexes,
            target.ribosome_target,
            inputs.translation_support,
            inputs.degradation_pressure,
            inputs.dt_scale,
            640.0,
        );
    let (rnap_complexes, rnap_assembly_rate, rnap_degradation_rate) = complex_channel_step(
        current.rnap_complexes,
        target.rnap_target,
        inputs.transcription_support,
        inputs.degradation_pressure,
        inputs.dt_scale,
        384.0,
    );
    let (replisome_complexes, replisome_assembly_rate, replisome_degradation_rate) =
        complex_channel_step(
            current.replisome_complexes,
            target.replisome_target,
            inputs.replication_support,
            inputs.degradation_pressure,
            inputs.dt_scale,
            192.0,
        );
    let (membrane_complexes, membrane_assembly_rate, membrane_degradation_rate) =
        complex_channel_step(
            current.membrane_complexes,
            target.membrane_target,
            inputs.membrane_support,
            inputs.degradation_pressure,
            inputs.dt_scale,
            384.0,
        );
    let (ftsz_polymer, ftsz_assembly_rate, ftsz_degradation_rate) = complex_channel_step(
        current.ftsz_polymer,
        target.ftsz_target,
        inputs.constriction_support,
        inputs.degradation_pressure,
        inputs.dt_scale,
        768.0,
    );
    let (dnaa_activity, dnaa_assembly_rate, dnaa_degradation_rate) = complex_channel_step(
        current.dnaa_activity,
        target.dnaa_target,
        inputs.replication_support,
        inputs.degradation_pressure,
        inputs.dt_scale,
        256.0,
    );

    WholeCellComplexAssemblyState {
        atp_band_complexes,
        ribosome_complexes,
        rnap_complexes,
        replisome_complexes,
        membrane_complexes,
        ftsz_polymer,
        dnaa_activity,
        atp_band_target: target.atp_band_target,
        ribosome_target: target.ribosome_target,
        rnap_target: target.rnap_target,
        replisome_target: target.replisome_target,
        membrane_target: target.membrane_target,
        ftsz_target: target.ftsz_target,
        dnaa_target: target.dnaa_target,
        atp_band_assembly_rate,
        ribosome_assembly_rate,
        rnap_assembly_rate,
        replisome_assembly_rate,
        membrane_assembly_rate,
        ftsz_assembly_rate,
        dnaa_assembly_rate,
        atp_band_degradation_rate,
        ribosome_degradation_rate,
        rnap_degradation_rate,
        replisome_degradation_rate,
        membrane_degradation_rate,
        ftsz_degradation_rate,
        dnaa_degradation_rate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_assembly_target_responds_to_membrane_signal() {
        let prior = WholeCellComplexAssemblyState {
            membrane_complexes: 10.0,
            ftsz_polymer: 12.0,
            ..WholeCellComplexAssemblyState::default()
        };
        let low = bare_assembly_target(
            prior,
            BareAssemblyTargetInputs {
                replicated_fraction: 0.2,
                quantum_energy: 1.0,
                quantum_translation: 1.0,
                quantum_nucleotide: 1.0,
                quantum_membrane: 1.0,
                quantum_segregation: 1.0,
                energy_signal: 0.5,
                transcription_signal: 0.5,
                translation_signal: 0.5,
                membrane_signal: 0.2,
            },
        );
        let high = bare_assembly_target(
            prior,
            BareAssemblyTargetInputs {
                membrane_signal: 0.9,
                ..BareAssemblyTargetInputs {
                    replicated_fraction: 0.2,
                    quantum_energy: 1.0,
                    quantum_translation: 1.0,
                    quantum_nucleotide: 1.0,
                    quantum_membrane: 1.0,
                    quantum_segregation: 1.0,
                    energy_signal: 0.5,
                    transcription_signal: 0.5,
                    translation_signal: 0.5,
                    membrane_signal: 0.2,
                }
            },
        );

        assert!(high.membrane_target > low.membrane_target);
        assert!(high.ftsz_target > low.ftsz_target);
    }

    #[test]
    fn seed_complex_assembly_from_target_maps_targets_into_current_channels() {
        let target = WholeCellComplexAssemblyState {
            atp_band_target: 4.0,
            ribosome_target: 5.0,
            rnap_target: 6.0,
            replisome_target: 7.0,
            membrane_target: 8.0,
            ftsz_target: 9.0,
            dnaa_target: 10.0,
            ..WholeCellComplexAssemblyState::default()
        };

        let seeded = seed_complex_assembly_from_target(target);

        assert!((seeded.atp_band_complexes - 4.0).abs() < 1.0e-6);
        assert!((seeded.membrane_complexes - 8.0).abs() < 1.0e-6);
        assert!((seeded.dnaa_activity - 10.0).abs() < 1.0e-6);
    }

    #[test]
    fn derived_assembly_target_preserves_prior_current_channels() {
        let prior = WholeCellComplexAssemblyState {
            atp_band_complexes: 3.0,
            ribosome_complexes: 4.0,
            rnap_complexes: 5.0,
            replisome_complexes: 6.0,
            membrane_complexes: 7.0,
            ftsz_polymer: 8.0,
            dnaa_activity: 9.0,
            ..WholeCellComplexAssemblyState::default()
        };
        let target = derived_assembly_target(
            prior,
            DerivedAssemblyTargetInputs {
                assembly_drive: WholeCellProcessWeights {
                    energy: 1.0,
                    transcription: 1.0,
                    translation: 1.0,
                    replication: 1.0,
                    segregation: 1.0,
                    membrane: 1.0,
                    constriction: 1.0,
                },
                mean_signal: 1.0,
                protein_signal: 0.6,
                transcript_signal: 0.5,
                localized_supply: 1.0,
                crowding: 1.0,
                replicated_fraction: 0.3,
                organism_scales: WholeCellOrganismProcessScales::default(),
                expression_energy_support: 1.0,
                expression_translation_support: 1.0,
                expression_nucleotide_support: 1.0,
                expression_membrane_support: 1.0,
            },
        );

        assert!((target.atp_band_complexes - prior.atp_band_complexes).abs() < 1.0e-6);
        assert!(target.ribosome_target > 0.0);
        assert!(target.ftsz_target > 0.0);
    }

    #[test]
    fn advance_complex_assembly_state_preserves_targets_and_updates_channels() {
        let current = WholeCellComplexAssemblyState {
            atp_band_complexes: 8.0,
            ribosome_complexes: 10.0,
            rnap_complexes: 12.0,
            replisome_complexes: 6.0,
            membrane_complexes: 9.0,
            ftsz_polymer: 7.0,
            dnaa_activity: 5.0,
            ..WholeCellComplexAssemblyState::default()
        };
        let target = WholeCellComplexAssemblyState {
            atp_band_target: 18.0,
            ribosome_target: 22.0,
            rnap_target: 20.0,
            replisome_target: 11.0,
            membrane_target: 16.0,
            ftsz_target: 13.0,
            dnaa_target: 9.0,
            ..WholeCellComplexAssemblyState::default()
        };

        let next = advance_complex_assembly_state(
            current,
            target,
            ComplexAssemblyAdvanceInputs {
                energy_support: 1.1,
                transcription_support: 1.0,
                translation_support: 1.2,
                replication_support: 1.0,
                membrane_support: 1.15,
                constriction_support: 1.05,
                degradation_pressure: 0.9,
                dt_scale: 1.0,
            },
        );

        assert!(next.atp_band_complexes > current.atp_band_complexes);
        assert!(next.ribosome_complexes > current.ribosome_complexes);
        assert!(next.membrane_complexes > current.membrane_complexes);
        assert!(next.ribosome_assembly_rate > 0.0);
        assert!(next.dnaa_degradation_rate > 0.0);
        assert!((next.ribosome_target - target.ribosome_target).abs() < 1.0e-6);
        assert!((next.ftsz_target - target.ftsz_target).abs() < 1.0e-6);
    }

    #[test]
    fn complex_assembly_advance_inputs_raise_energy_support_and_degradation_with_load() {
        let base = ComplexAssemblySupportInputs {
            dt: 0.5,
            reference_dt: 0.25,
            crowding_penalty: 0.95,
            effective_load: 1.0,
            expression_energy_support: 0.8,
            expression_translation_support: 0.9,
            expression_nucleotide_support: 0.85,
            expression_membrane_support: 0.88,
            chemistry_atp_support: 0.7,
            organism_scales: WholeCellOrganismProcessScales::default(),
            replicated_fraction: 0.4,
        };
        let low = complex_assembly_advance_inputs(base);
        let high = complex_assembly_advance_inputs(ComplexAssemblySupportInputs {
            chemistry_atp_support: 1.2,
            effective_load: 1.8,
            ..base
        });

        assert!(high.energy_support > low.energy_support);
        assert!(high.degradation_pressure > low.degradation_pressure);
        assert!((low.dt_scale - 2.0).abs() < 1.0e-6);
    }

    #[test]
    fn bare_lower_scale_assembly_inventory_ignores_division_progress_when_direct_state_exists() {
        let base = BareLowerScaleAssemblyInventoryInputs {
            capacities: WholeCellProcessFluxes {
                energy_capacity: 1.0,
                transcription_capacity: 1.0,
                translation_capacity: 1.0,
                replication_capacity: 1.0,
                segregation_capacity: 1.0,
                membrane_capacity: 1.0,
                constriction_capacity: 1.0,
            },
            localized_supply: 1.0,
            effective_metabolic_load: 1.0,
            chemistry_atp_support: 1.0,
            chemistry_translation_support: 1.0,
            chemistry_nucleotide_support: 1.0,
            chemistry_membrane_support: 1.0,
            quantum_oxphos_efficiency: 1.0,
            quantum_translation_efficiency: 1.0,
            quantum_nucleotide_efficiency: 1.0,
            quantum_membrane_efficiency: 1.0,
            quantum_segregation_efficiency: 1.0,
            replicated_fraction: 0.5,
            initiation_signal: 0.8,
            separation_signal: 0.6,
            explicit_division_state_available: false,
            division_progress: 1.0,
            atp_mm: 1.0,
            glucose_mm: 0.8,
            nucleotides_mm: 1.0,
            amino_acids_mm: 1.0,
            membrane_precursors_mm: 1.0,
            oxygen_mm: 1.0,
        };

        let heuristic = bare_lower_scale_assembly_inventory(base);
        let direct = bare_lower_scale_assembly_inventory(BareLowerScaleAssemblyInventoryInputs {
            explicit_division_state_available: true,
            ..base
        });

        assert!(heuristic.ftsz_polymer > direct.ftsz_polymer);
        assert!((heuristic.dnaa_activity - direct.dnaa_activity).abs() < 1.0e-6);
    }

    #[test]
    fn bare_lower_scale_assembly_inventory_responds_to_membrane_supply() {
        let low = bare_lower_scale_assembly_inventory(BareLowerScaleAssemblyInventoryInputs {
            capacities: WholeCellProcessFluxes {
                energy_capacity: 1.0,
                transcription_capacity: 1.0,
                translation_capacity: 1.0,
                replication_capacity: 1.0,
                segregation_capacity: 1.0,
                membrane_capacity: 1.0,
                constriction_capacity: 1.0,
            },
            localized_supply: 1.0,
            effective_metabolic_load: 1.0,
            chemistry_atp_support: 1.0,
            chemistry_translation_support: 1.0,
            chemistry_nucleotide_support: 1.0,
            chemistry_membrane_support: 0.3,
            quantum_oxphos_efficiency: 1.0,
            quantum_translation_efficiency: 1.0,
            quantum_nucleotide_efficiency: 1.0,
            quantum_membrane_efficiency: 1.0,
            quantum_segregation_efficiency: 1.0,
            replicated_fraction: 0.4,
            initiation_signal: 0.5,
            separation_signal: 0.3,
            explicit_division_state_available: true,
            division_progress: 0.0,
            atp_mm: 1.0,
            glucose_mm: 0.8,
            nucleotides_mm: 1.0,
            amino_acids_mm: 1.0,
            membrane_precursors_mm: 0.2,
            oxygen_mm: 0.1,
        });
        let high = bare_lower_scale_assembly_inventory(BareLowerScaleAssemblyInventoryInputs {
            chemistry_membrane_support: 1.2,
            membrane_precursors_mm: 1.5,
            oxygen_mm: 1.0,
            ..BareLowerScaleAssemblyInventoryInputs {
                capacities: WholeCellProcessFluxes {
                    energy_capacity: 1.0,
                    transcription_capacity: 1.0,
                    translation_capacity: 1.0,
                    replication_capacity: 1.0,
                    segregation_capacity: 1.0,
                    membrane_capacity: 1.0,
                    constriction_capacity: 1.0,
                },
                localized_supply: 1.0,
                effective_metabolic_load: 1.0,
                chemistry_atp_support: 1.0,
                chemistry_translation_support: 1.0,
                chemistry_nucleotide_support: 1.0,
                chemistry_membrane_support: 0.3,
                quantum_oxphos_efficiency: 1.0,
                quantum_translation_efficiency: 1.0,
                quantum_nucleotide_efficiency: 1.0,
                quantum_membrane_efficiency: 1.0,
                quantum_segregation_efficiency: 1.0,
                replicated_fraction: 0.4,
                initiation_signal: 0.5,
                separation_signal: 0.3,
                explicit_division_state_available: true,
                division_progress: 0.0,
                atp_mm: 1.0,
                glucose_mm: 0.8,
                nucleotides_mm: 1.0,
                amino_acids_mm: 1.0,
                membrane_precursors_mm: 0.2,
                oxygen_mm: 0.1,
            }
        });

        assert!(high.membrane_complexes > low.membrane_complexes);
        assert!(high.ftsz_polymer > low.ftsz_polymer);
    }
}
