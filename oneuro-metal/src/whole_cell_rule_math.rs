//! Pure rule-signal and blend reducers extracted from `whole_cell.rs`.
//!
//! These helpers keep occupancy-to-rule-context math and generic fallback
//! blending out of the main simulator implementation.
//!
//! This module defines its own extended signal enum and context types because
//! the canonical `WholeCellRuleSignal` in `whole_cell.rs` only exposes 36
//! base signals.  The rule-math layer needs ~30 additional derived signals
//! (process occupancy blends, capacity slots, inventory channels, etc.) that
//! are computed here and consumed by `ScalarRule` evaluation.  Keeping these
//! extra slots local avoids bloating the core enum while giving the rule
//! evaluator full access to every signal dimension it needs.

use crate::substrate_ir::{ScalarContext, ScalarRule, EMPTY_SCALAR_BRANCH};
use crate::whole_cell_data::{WholeCellComplexAssemblyState, WholeCellProcessWeights};

// ---------------------------------------------------------------------------
// Local process-fluxes type (mirrors the private WholeCellProcessFluxes in
// whole_cell.rs so that this module compiles independently).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub(crate) struct WholeCellProcessFluxes {
    pub(crate) energy_capacity: f32,
    pub(crate) transcription_capacity: f32,
    pub(crate) translation_capacity: f32,
    pub(crate) replication_capacity: f32,
    pub(crate) segregation_capacity: f32,
    pub(crate) membrane_capacity: f32,
    pub(crate) constriction_capacity: f32,
}

impl Default for WholeCellProcessFluxes {
    fn default() -> Self {
        Self {
            energy_capacity: 0.0,
            transcription_capacity: 0.0,
            translation_capacity: 0.0,
            replication_capacity: 0.0,
            segregation_capacity: 0.0,
            membrane_capacity: 0.0,
            constriction_capacity: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Local occupancy state (the canonical type was removed from whole_cell_data).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub(crate) struct WholeCellProcessOccupancyState {
    pub(crate) current: WholeCellProcessWeights,
    pub(crate) target: WholeCellProcessWeights,
    pub(crate) assembly_rate: WholeCellProcessWeights,
    pub(crate) degradation_rate: WholeCellProcessWeights,
}

impl Default for WholeCellProcessOccupancyState {
    fn default() -> Self {
        Self {
            current: WholeCellProcessWeights::default(),
            target: WholeCellProcessWeights::default(),
            assembly_rate: WholeCellProcessWeights::default(),
            degradation_rate: WholeCellProcessWeights::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Extended rule-signal enum -- superset of the 36 base signals from
// `whole_cell::WholeCellRuleSignal` plus ~30 derived signals used only by
// the rule-math layer.
//
// The first 37 entries (Dt .. MembranePrecursorFloor) are index-compatible
// with the canonical enum so that `ScalarFactor` indices produced elsewhere
// still evaluate correctly.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub(crate) enum RuleSignal {
    // ---- base signals (indices 0-35, matching whole_cell::WholeCellRuleSignal) ----
    Dt = 0,
    AtpBandSignal,
    RibosomeSignal,
    ReplisomeSignal,
    SeptumSignal,
    WeightedRibosomeReplisomeSignal,
    WeightedAtpSeptumSignal,
    GlucoseSignal,
    OxygenSignal,
    AminoSignal,
    NucleotideSignal,
    MembraneSignal,
    EnergySignal,
    ReplicatedFraction,
    InverseReplicatedFraction,
    DivisionReadiness,
    LocalizedSupplyScale,
    CrowdingPenalty,
    AtpSupport,
    TranslationSupport,
    NucleotideSupport,
    MembraneSupport,
    AtpBandScale,
    RibosomeTranslationScale,
    ReplisomeReplicationScale,
    ReplisomeSegregationScale,
    MembraneAssemblyScale,
    FtszConstrictionScale,
    MdTranslationScale,
    MdMembraneScale,
    QuantumOxphosEfficiency,
    QuantumTranslationEfficiency,
    QuantumNucleotideEfficiency,
    QuantumMembraneEfficiency,
    QuantumSegregationEfficiency,
    EffectiveMetabolicLoad,
    MembranePrecursorFloor, // = 36

    // ---- extended signals (process occupancy blends) ----
    EnergyProcessSignal,
    TranscriptionProcessSignal,
    TranslationProcessSignal,
    ReplicationProcessSignal,
    SegregationProcessSignal,
    MembraneProcessSignal,
    ConstrictionProcessSignal,

    // ---- extended signals (complex inventory) ----
    AtpBandComplexes,
    RibosomeComplexes,
    RnapComplexes,
    ReplisomeComplexes,
    MembraneComplexes,
    FtszPolymer,
    DnaaActivity,

    // ---- extended signals (capacity / stage) ----
    EnergyCapacity,
    EnergyCapacityCapped16,
    EnergyCapacityCapped18,
    TranscriptionCapacity,
    TranscriptionCapacityCapped16,
    TranslationCapacity,
    ReplicationCapacity,
    SegregationCapacity,
    MembraneCapacity,
    ConstrictionCapacity,

    // ---- extended signals (stage drive) ----
    DnaaSignal,
    ReplisomeAssemblySignal,
    ConstrictionSignal,
    TranscriptionDriveMix,
    TranslationDriveMix,
    BiosyntheticLoadMix,
}

impl RuleSignal {
    pub(crate) const COUNT: usize = Self::BiosyntheticLoadMix as usize + 1;
}

// ---------------------------------------------------------------------------
// Rule context -- a fixed-size signal array sized to hold every extended slot.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub(crate) struct RuleContext {
    pub(crate) signals: [f32; RuleSignal::COUNT],
}

impl Default for RuleContext {
    fn default() -> Self {
        Self {
            signals: [0.0; RuleSignal::COUNT],
        }
    }
}

impl RuleContext {
    /// Write a signal value, clamping to >=0 and replacing non-finite with 0.
    fn set(&mut self, signal: RuleSignal, value: f32) {
        self.signals[signal as usize] = if value.is_finite() {
            value.max(0.0)
        } else {
            0.0
        };
    }

    /// Read a signal value.
    fn get(self, signal: RuleSignal) -> f32 {
        self.signals[signal as usize]
    }

    /// Convert to a `ScalarContext` for rule evaluation.
    fn scalar(self) -> ScalarContext<{ RuleSignal::COUNT }> {
        ScalarContext {
            signals: self.signals,
        }
    }
}

// ---------------------------------------------------------------------------
// Input bundles
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub(crate) struct BaseRuleContextInputs {
    pub(crate) dt: f32,
    pub(crate) septum_signal: f32,
    pub(crate) glucose_signal: f32,
    pub(crate) oxygen_signal: f32,
    pub(crate) amino_signal: f32,
    pub(crate) nucleotide_signal: f32,
    pub(crate) membrane_signal: f32,
    pub(crate) energy_signal: f32,
    pub(crate) replicated_fraction: f32,
    pub(crate) division_readiness: f32,
    pub(crate) localized_supply_scale: f32,
    pub(crate) crowding_penalty: f32,
    pub(crate) atp_support: f32,
    pub(crate) translation_support: f32,
    pub(crate) nucleotide_support: f32,
    pub(crate) membrane_support: f32,
    pub(crate) atp_band_scale: f32,
    pub(crate) ribosome_translation_scale: f32,
    pub(crate) replisome_replication_scale: f32,
    pub(crate) replisome_segregation_scale: f32,
    pub(crate) membrane_assembly_scale: f32,
    pub(crate) ftsz_constriction_scale: f32,
    pub(crate) md_translation_scale: f32,
    pub(crate) md_membrane_scale: f32,
    pub(crate) quantum_oxphos_efficiency: f32,
    pub(crate) quantum_translation_efficiency: f32,
    pub(crate) quantum_nucleotide_efficiency: f32,
    pub(crate) quantum_membrane_efficiency: f32,
    pub(crate) quantum_segregation_efficiency: f32,
    pub(crate) effective_metabolic_load: f32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct StageRuleContextInputs {
    pub(crate) fluxes: WholeCellProcessFluxes,
    pub(crate) dnaa_signal: f32,
    pub(crate) replisome_assembly_signal: f32,
    pub(crate) constriction_signal: f32,
    pub(crate) transcription_drive_mix: f32,
    pub(crate) translation_drive_mix: f32,
    pub(crate) biosynthetic_load_mix: f32,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct DirectStageRuleMixInputs {
    pub(crate) fluxes: WholeCellProcessFluxes,
    pub(crate) runtime_occupancy: WholeCellProcessOccupancyState,
    pub(crate) operon_drive: WholeCellProcessWeights,
    pub(crate) registry_drive: WholeCellProcessWeights,
    pub(crate) transcription_flux: f32,
    pub(crate) transcription_extent: f32,
    pub(crate) translation_flux: f32,
    pub(crate) translation_extent: f32,
    pub(crate) transcript_species_signal: f32,
    pub(crate) protein_species_signal: f32,
    pub(crate) assembly_species_signal: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DirectStageRuleMixes {
    pub(crate) transcription_drive_mix: Option<f32>,
    pub(crate) translation_drive_mix: Option<f32>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn saturating_signal(value: f32, half_saturation: f32) -> f32 {
    let value = value.max(0.0);
    let half_saturation = half_saturation.max(1.0e-6);
    (value / (value + half_saturation)).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Population functions
// ---------------------------------------------------------------------------

pub(crate) fn populate_base_rule_context(
    ctx: &mut RuleContext,
    inputs: BaseRuleContextInputs,
) {
    ctx.set(RuleSignal::Dt, inputs.dt);
    ctx.set(RuleSignal::SeptumSignal, inputs.septum_signal);
    ctx.set(RuleSignal::GlucoseSignal, inputs.glucose_signal);
    ctx.set(RuleSignal::OxygenSignal, inputs.oxygen_signal);
    ctx.set(RuleSignal::AminoSignal, inputs.amino_signal);
    ctx.set(RuleSignal::NucleotideSignal, inputs.nucleotide_signal);
    ctx.set(RuleSignal::MembraneSignal, inputs.membrane_signal);
    ctx.set(RuleSignal::EnergySignal, inputs.energy_signal);
    ctx.set(
        RuleSignal::ReplicatedFraction,
        inputs.replicated_fraction,
    );
    ctx.set(
        RuleSignal::InverseReplicatedFraction,
        (1.0 - inputs.replicated_fraction).clamp(0.0, 1.0),
    );
    ctx.set(RuleSignal::DivisionReadiness, inputs.division_readiness);
    ctx.set(
        RuleSignal::LocalizedSupplyScale,
        inputs.localized_supply_scale,
    );
    ctx.set(RuleSignal::CrowdingPenalty, inputs.crowding_penalty);
    ctx.set(RuleSignal::AtpSupport, inputs.atp_support);
    ctx.set(
        RuleSignal::TranslationSupport,
        inputs.translation_support,
    );
    ctx.set(
        RuleSignal::NucleotideSupport,
        inputs.nucleotide_support,
    );
    ctx.set(RuleSignal::MembraneSupport, inputs.membrane_support);
    ctx.set(RuleSignal::AtpBandScale, inputs.atp_band_scale);
    ctx.set(
        RuleSignal::RibosomeTranslationScale,
        inputs.ribosome_translation_scale,
    );
    ctx.set(
        RuleSignal::ReplisomeReplicationScale,
        inputs.replisome_replication_scale,
    );
    ctx.set(
        RuleSignal::ReplisomeSegregationScale,
        inputs.replisome_segregation_scale,
    );
    ctx.set(
        RuleSignal::MembraneAssemblyScale,
        inputs.membrane_assembly_scale,
    );
    ctx.set(
        RuleSignal::FtszConstrictionScale,
        inputs.ftsz_constriction_scale,
    );
    ctx.set(RuleSignal::MdTranslationScale, inputs.md_translation_scale);
    ctx.set(RuleSignal::MdMembraneScale, inputs.md_membrane_scale);
    ctx.set(
        RuleSignal::QuantumOxphosEfficiency,
        inputs.quantum_oxphos_efficiency,
    );
    ctx.set(
        RuleSignal::QuantumTranslationEfficiency,
        inputs.quantum_translation_efficiency,
    );
    ctx.set(
        RuleSignal::QuantumNucleotideEfficiency,
        inputs.quantum_nucleotide_efficiency,
    );
    ctx.set(
        RuleSignal::QuantumMembraneEfficiency,
        inputs.quantum_membrane_efficiency,
    );
    ctx.set(
        RuleSignal::QuantumSegregationEfficiency,
        inputs.quantum_segregation_efficiency,
    );
    ctx.set(
        RuleSignal::EffectiveMetabolicLoad,
        inputs.effective_metabolic_load,
    );
}

pub(crate) fn populate_process_rule_context(
    ctx: &mut RuleContext,
    occupancy: WholeCellProcessOccupancyState,
    inventory: WholeCellComplexAssemblyState,
) {
    ctx.set(
        RuleSignal::EnergyProcessSignal,
        (occupancy.current.energy
            + 0.34 * occupancy.assembly_rate.energy
            + 0.10 * occupancy.target.energy
            + 0.08 * occupancy.current.membrane)
            .max(0.0),
    );
    ctx.set(
        RuleSignal::TranscriptionProcessSignal,
        (occupancy.current.transcription
            + 0.34 * occupancy.assembly_rate.transcription
            + 0.10 * occupancy.target.transcription
            + 0.06 * occupancy.current.translation)
            .max(0.0),
    );
    ctx.set(
        RuleSignal::TranslationProcessSignal,
        (occupancy.current.translation
            + 0.36 * occupancy.assembly_rate.translation
            + 0.10 * occupancy.target.translation
            + 0.06 * occupancy.current.transcription)
            .max(0.0),
    );
    ctx.set(
        RuleSignal::ReplicationProcessSignal,
        (occupancy.current.replication
            + 0.24 * occupancy.current.segregation
            + 0.34 * occupancy.assembly_rate.replication
            + 0.18 * occupancy.assembly_rate.segregation
            + 0.10 * occupancy.target.replication)
            .max(0.0),
    );
    ctx.set(
        RuleSignal::SegregationProcessSignal,
        (occupancy.current.segregation
            + 0.30 * occupancy.assembly_rate.segregation
            + 0.10 * occupancy.target.segregation
            + 0.12 * occupancy.current.replication)
            .max(0.0),
    );
    ctx.set(
        RuleSignal::MembraneProcessSignal,
        (occupancy.current.membrane
            + 0.30 * occupancy.assembly_rate.membrane
            + 0.10 * occupancy.target.membrane
            + 0.10 * occupancy.current.energy)
            .max(0.0),
    );
    ctx.set(
        RuleSignal::ConstrictionProcessSignal,
        (occupancy.current.constriction
            + 0.34 * occupancy.assembly_rate.constriction
            + 0.12 * occupancy.target.constriction
            + 0.08 * occupancy.current.membrane)
            .max(0.0),
    );
    ctx.set(
        RuleSignal::AtpBandComplexes,
        inventory.atp_band_complexes,
    );
    ctx.set(
        RuleSignal::RibosomeComplexes,
        inventory.ribosome_complexes,
    );
    ctx.set(RuleSignal::RnapComplexes, inventory.rnap_complexes);
    ctx.set(
        RuleSignal::ReplisomeComplexes,
        inventory.replisome_complexes,
    );
    ctx.set(
        RuleSignal::MembraneComplexes,
        inventory.membrane_complexes,
    );
    ctx.set(RuleSignal::FtszPolymer, inventory.ftsz_polymer);
    ctx.set(RuleSignal::DnaaActivity, inventory.dnaa_activity);
}

pub(crate) fn populate_stage_rule_context(
    ctx: &mut RuleContext,
    inputs: StageRuleContextInputs,
) {
    ctx.set(
        RuleSignal::EnergyCapacity,
        inputs.fluxes.energy_capacity,
    );
    ctx.set(
        RuleSignal::EnergyCapacityCapped16,
        inputs.fluxes.energy_capacity.min(1.6),
    );
    ctx.set(
        RuleSignal::EnergyCapacityCapped18,
        inputs.fluxes.energy_capacity.min(1.8),
    );
    ctx.set(
        RuleSignal::TranscriptionCapacity,
        inputs.fluxes.transcription_capacity,
    );
    ctx.set(
        RuleSignal::TranscriptionCapacityCapped16,
        inputs.fluxes.transcription_capacity.min(1.6),
    );
    ctx.set(
        RuleSignal::TranslationCapacity,
        inputs.fluxes.translation_capacity,
    );
    ctx.set(
        RuleSignal::ReplicationCapacity,
        inputs.fluxes.replication_capacity,
    );
    ctx.set(
        RuleSignal::SegregationCapacity,
        inputs.fluxes.segregation_capacity,
    );
    ctx.set(
        RuleSignal::MembraneCapacity,
        inputs.fluxes.membrane_capacity,
    );
    ctx.set(
        RuleSignal::ConstrictionCapacity,
        inputs.fluxes.constriction_capacity,
    );
    ctx.set(RuleSignal::DnaaSignal, inputs.dnaa_signal);
    ctx.set(
        RuleSignal::ReplisomeAssemblySignal,
        inputs.replisome_assembly_signal,
    );
    ctx.set(
        RuleSignal::ConstrictionSignal,
        inputs.constriction_signal,
    );
    ctx.set(
        RuleSignal::TranscriptionDriveMix,
        inputs.transcription_drive_mix,
    );
    ctx.set(
        RuleSignal::TranslationDriveMix,
        inputs.translation_drive_mix,
    );
    ctx.set(
        RuleSignal::BiosyntheticLoadMix,
        inputs.biosynthetic_load_mix,
    );
}

pub(crate) fn direct_stage_rule_mixes(inputs: DirectStageRuleMixInputs) -> DirectStageRuleMixes {
    let transcription_available = inputs.runtime_occupancy.current.transcription > 1.0e-6
        || inputs.runtime_occupancy.assembly_rate.transcription > 1.0e-6
        || inputs.transcription_flux > 1.0e-6
        || inputs.transcription_extent > 1.0e-6
        || inputs.transcript_species_signal > 1.0e-6;
    let translation_available = inputs.runtime_occupancy.current.translation > 1.0e-6
        || inputs.runtime_occupancy.assembly_rate.translation > 1.0e-6
        || inputs.translation_flux > 1.0e-6
        || inputs.translation_extent > 1.0e-6
        || inputs.protein_species_signal > 1.0e-6
        || inputs.assembly_species_signal > 1.0e-6;
    let transcription_drive_mix = if transcription_available {
        Some(saturating_signal(
            0.40 * inputs.fluxes.transcription_capacity
                + 0.20
                    * saturating_signal(
                        inputs.transcription_flux + 0.35 * inputs.transcription_extent,
                        18.0,
                    )
                + 0.14 * saturating_signal(inputs.transcript_species_signal, 256.0)
                + 0.14 * inputs.operon_drive.transcription
                + 0.12 * inputs.runtime_occupancy.current.transcription
                + 0.08 * inputs.registry_drive.transcription,
            2.4,
        ))
    } else {
        None
    };
    let translation_drive_mix = if translation_available {
        Some(saturating_signal(
            0.40 * inputs.fluxes.translation_capacity
                + 0.20
                    * saturating_signal(
                        inputs.translation_flux + 0.30 * inputs.translation_extent,
                        20.0,
                    )
                + 0.14 * saturating_signal(inputs.protein_species_signal, 256.0)
                + 0.12 * saturating_signal(inputs.assembly_species_signal, 192.0)
                + 0.12 * inputs.operon_drive.translation
                + 0.08 * inputs.runtime_occupancy.current.translation
                + 0.06 * inputs.registry_drive.translation,
            2.6,
        ))
    } else {
        None
    };

    DirectStageRuleMixes {
        transcription_drive_mix,
        translation_drive_mix,
    }
}

pub(crate) fn evaluate_process_preferred_capacity_rule(
    rule: ScalarRule,
    process_signal: RuleSignal,
    scalar: ScalarContext<{ RuleSignal::COUNT }>,
) -> f32 {
    let branch = if scalar.signal(process_signal as usize) > 1.0e-6 && rule.branch_count >= 2 {
        rule.branches[1]
    } else if rule.branch_count >= 1 {
        rule.branches[0]
    } else {
        EMPTY_SCALAR_BRANCH
    };
    let value = rule.bias + branch.evaluate(scalar);
    let min_value = rule.min_value.min(rule.max_value);
    let max_value = rule.max_value.max(rule.min_value);
    if value.is_finite() {
        value.clamp(min_value, max_value)
    } else {
        min_value
    }
}

pub(crate) fn capacity_rule_fluxes_from_context(
    ctx: RuleContext,
    energy_rule: ScalarRule,
    transcription_rule: ScalarRule,
    translation_rule: ScalarRule,
    replication_rule: ScalarRule,
    segregation_rule: ScalarRule,
    membrane_rule: ScalarRule,
    constriction_rule: ScalarRule,
) -> WholeCellProcessFluxes {
    let scalar = ctx.scalar();
    WholeCellProcessFluxes {
        energy_capacity: evaluate_process_preferred_capacity_rule(
            energy_rule,
            RuleSignal::EnergyProcessSignal,
            scalar,
        ),
        transcription_capacity: evaluate_process_preferred_capacity_rule(
            transcription_rule,
            RuleSignal::TranscriptionProcessSignal,
            scalar,
        ),
        translation_capacity: evaluate_process_preferred_capacity_rule(
            translation_rule,
            RuleSignal::TranslationProcessSignal,
            scalar,
        ),
        replication_capacity: evaluate_process_preferred_capacity_rule(
            replication_rule,
            RuleSignal::ReplicationProcessSignal,
            scalar,
        ),
        segregation_capacity: evaluate_process_preferred_capacity_rule(
            segregation_rule,
            RuleSignal::SegregationProcessSignal,
            scalar,
        ),
        membrane_capacity: evaluate_process_preferred_capacity_rule(
            membrane_rule,
            RuleSignal::MembraneProcessSignal,
            scalar,
        ),
        constriction_capacity: evaluate_process_preferred_capacity_rule(
            constriction_rule,
            RuleSignal::ConstrictionProcessSignal,
            scalar,
        ),
    }
}

pub(crate) fn blend_with_substrate_preference(
    rule_flux: f32,
    substrate_flux: f32,
    max_flux: f32,
) -> f32 {
    let rule_flux = rule_flux.max(0.0);
    let substrate_flux = substrate_flux.max(0.0);
    let selected = if substrate_flux > 1.0e-6 {
        substrate_flux
    } else if rule_flux > 1.0e-6 {
        rule_flux
    } else {
        0.0
    };
    if selected.is_finite() {
        selected.clamp(0.0, max_flux)
    } else {
        rule_flux.clamp(0.0, max_flux)
    }
}

pub(crate) fn blend_stage_flux_auto(rule_flux: f32, substrate_flux: f32) -> f32 {
    let max_flux = rule_flux.max(substrate_flux).max(1.0) * 2.0;
    if substrate_flux.abs() > 1.0e-6 {
        substrate_flux.clamp(0.0, max_flux)
    } else {
        rule_flux.clamp(0.0, max_flux)
    }
}

pub(crate) fn blend_runtime_bulk_delta(runtime_delta: f32, fallback_delta: f32) -> f32 {
    if runtime_delta.abs() > 1.0e-6 {
        runtime_delta
    } else {
        fallback_delta
    }
}

#[cfg(all(test, feature = "satellite_tests"))]
mod tests {
    use super::*;
    use crate::substrate_ir::{ScalarBranch, ScalarFactor};

    fn test_scalar_factor(signal: RuleSignal) -> ScalarFactor {
        ScalarFactor::new(signal as usize, 0.0, 1.0, 1.0)
    }

    #[test]
    fn populate_process_rule_context_sets_live_and_inventory_channels() {
        let occupancy = WholeCellProcessOccupancyState {
            current: WholeCellProcessWeights {
                energy: 2.0,
                transcription: 3.0,
                translation: 4.0,
                replication: 5.0,
                segregation: 6.0,
                membrane: 7.0,
                constriction: 8.0,
            },
            target: WholeCellProcessWeights {
                energy: 1.0,
                transcription: 1.0,
                translation: 1.0,
                replication: 1.0,
                segregation: 1.0,
                membrane: 1.0,
                constriction: 1.0,
            },
            assembly_rate: WholeCellProcessWeights {
                energy: 0.5,
                transcription: 0.5,
                translation: 0.5,
                replication: 0.5,
                segregation: 0.5,
                membrane: 0.5,
                constriction: 0.5,
            },
            degradation_rate: WholeCellProcessWeights::default(),
        };
        let inventory = WholeCellComplexAssemblyState {
            atp_band_complexes: 11.0,
            ribosome_complexes: 12.0,
            rnap_complexes: 13.0,
            replisome_complexes: 14.0,
            membrane_complexes: 15.0,
            ftsz_polymer: 16.0,
            dnaa_activity: 17.0,
            ..WholeCellComplexAssemblyState::default()
        };
        let mut ctx = RuleContext::default();

        populate_process_rule_context(&mut ctx, occupancy, inventory);

        assert!(ctx.get(RuleSignal::EnergyProcessSignal) > 2.0);
        assert!((ctx.get(RuleSignal::FtszPolymer) - 16.0).abs() < 1.0e-6);
        assert!((ctx.get(RuleSignal::DnaaActivity) - 17.0).abs() < 1.0e-6);
    }

    #[test]
    fn populate_base_and_stage_rule_context_sets_expected_surfaces() {
        let mut ctx = RuleContext::default();
        populate_base_rule_context(
            &mut ctx,
            BaseRuleContextInputs {
                dt: 0.25,
                septum_signal: 0.8,
                glucose_signal: 0.7,
                oxygen_signal: 0.6,
                amino_signal: 0.5,
                nucleotide_signal: 0.4,
                membrane_signal: 0.3,
                energy_signal: 1.2,
                replicated_fraction: 0.25,
                division_readiness: 0.5,
                localized_supply_scale: 1.1,
                crowding_penalty: 0.9,
                atp_support: 1.0,
                translation_support: 1.1,
                nucleotide_support: 1.2,
                membrane_support: 1.3,
                atp_band_scale: 1.4,
                ribosome_translation_scale: 1.5,
                replisome_replication_scale: 1.6,
                replisome_segregation_scale: 1.7,
                membrane_assembly_scale: 1.8,
                ftsz_constriction_scale: 1.9,
                md_translation_scale: 1.05,
                md_membrane_scale: 0.95,
                quantum_oxphos_efficiency: 1.1,
                quantum_translation_efficiency: 1.2,
                quantum_nucleotide_efficiency: 1.3,
                quantum_membrane_efficiency: 1.4,
                quantum_segregation_efficiency: 1.5,
                effective_metabolic_load: 1.25,
            },
        );
        populate_stage_rule_context(
            &mut ctx,
            StageRuleContextInputs {
                fluxes: WholeCellProcessFluxes {
                    energy_capacity: 2.0,
                    transcription_capacity: 1.5,
                    translation_capacity: 1.4,
                    replication_capacity: 1.3,
                    segregation_capacity: 1.2,
                    membrane_capacity: 1.1,
                    constriction_capacity: 1.0,
                },
                dnaa_signal: 0.8,
                replisome_assembly_signal: 0.7,
                constriction_signal: 0.6,
                transcription_drive_mix: 0.5,
                translation_drive_mix: 0.4,
                biosynthetic_load_mix: 0.3,
            },
        );

        assert!((ctx.get(RuleSignal::Dt) - 0.25).abs() < 1.0e-6);
        assert!((ctx.get(RuleSignal::InverseReplicatedFraction) - 0.75).abs() < 1.0e-6);
        assert!((ctx.get(RuleSignal::EnergyCapacityCapped16) - 1.6).abs() < 1.0e-6);
        assert!((ctx.get(RuleSignal::TranslationDriveMix) - 0.4).abs() < 1.0e-6);
    }

    #[test]
    fn capacity_rule_fluxes_prefer_process_branch_when_live_signal_exists() {
        let mut ctx = RuleContext::default();
        ctx.set(RuleSignal::EnergyProcessSignal, 2.0);
        ctx.set(RuleSignal::AtpBandComplexes, 10.0);
        let rule = ScalarRule::new(
            0.0,
            2,
            [
                ScalarBranch::new(
                    1.0,
                    1,
                    [
                        test_scalar_factor(RuleSignal::AtpBandComplexes),
                        crate::substrate_ir::EMPTY_SCALAR_FACTOR,
                        crate::substrate_ir::EMPTY_SCALAR_FACTOR,
                        crate::substrate_ir::EMPTY_SCALAR_FACTOR,
                        crate::substrate_ir::EMPTY_SCALAR_FACTOR,
                        crate::substrate_ir::EMPTY_SCALAR_FACTOR,
                        crate::substrate_ir::EMPTY_SCALAR_FACTOR,
                        crate::substrate_ir::EMPTY_SCALAR_FACTOR,
                    ],
                ),
                ScalarBranch::new(
                    1.0,
                    1,
                    [
                        test_scalar_factor(RuleSignal::EnergyProcessSignal),
                        crate::substrate_ir::EMPTY_SCALAR_FACTOR,
                        crate::substrate_ir::EMPTY_SCALAR_FACTOR,
                        crate::substrate_ir::EMPTY_SCALAR_FACTOR,
                        crate::substrate_ir::EMPTY_SCALAR_FACTOR,
                        crate::substrate_ir::EMPTY_SCALAR_FACTOR,
                        crate::substrate_ir::EMPTY_SCALAR_FACTOR,
                        crate::substrate_ir::EMPTY_SCALAR_FACTOR,
                    ],
                ),
                EMPTY_SCALAR_BRANCH,
                EMPTY_SCALAR_BRANCH,
            ],
            0.0,
            100.0,
        );

        let fluxes =
            capacity_rule_fluxes_from_context(ctx, rule, rule, rule, rule, rule, rule, rule);

        assert!((fluxes.energy_capacity - 2.0).abs() < 1.0e-6);
    }

    #[test]
    fn blend_helpers_prefer_live_runtime_values() {
        assert!((blend_with_substrate_preference(1.0, 2.0, 3.0) - 2.0).abs() < 1.0e-6);
        assert!((blend_stage_flux_auto(1.0, 2.0) - 2.0).abs() < 1.0e-6);
        assert!((blend_runtime_bulk_delta(0.5, -1.0) - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn direct_stage_rule_mixes_return_none_without_runtime_signal() {
        let mixes = direct_stage_rule_mixes(DirectStageRuleMixInputs {
            fluxes: WholeCellProcessFluxes::default(),
            runtime_occupancy: WholeCellProcessOccupancyState::default(),
            operon_drive: WholeCellProcessWeights::default(),
            registry_drive: WholeCellProcessWeights::default(),
            transcription_flux: 0.0,
            transcription_extent: 0.0,
            translation_flux: 0.0,
            translation_extent: 0.0,
            transcript_species_signal: 0.0,
            protein_species_signal: 0.0,
            assembly_species_signal: 0.0,
        });

        assert!(mixes.transcription_drive_mix.is_none());
        assert!(mixes.translation_drive_mix.is_none());
    }

    #[test]
    fn direct_stage_rule_mixes_respond_to_runtime_signals() {
        let mixes = direct_stage_rule_mixes(DirectStageRuleMixInputs {
            fluxes: WholeCellProcessFluxes {
                transcription_capacity: 1.4,
                translation_capacity: 1.6,
                ..WholeCellProcessFluxes::default()
            },
            runtime_occupancy: WholeCellProcessOccupancyState {
                current: WholeCellProcessWeights {
                    transcription: 0.8,
                    translation: 0.9,
                    ..WholeCellProcessWeights::default()
                },
                assembly_rate: WholeCellProcessWeights {
                    transcription: 0.3,
                    translation: 0.4,
                    ..WholeCellProcessWeights::default()
                },
                ..WholeCellProcessOccupancyState::default()
            },
            operon_drive: WholeCellProcessWeights {
                transcription: 0.5,
                translation: 0.6,
                ..WholeCellProcessWeights::default()
            },
            registry_drive: WholeCellProcessWeights {
                transcription: 0.2,
                translation: 0.3,
                ..WholeCellProcessWeights::default()
            },
            transcription_flux: 3.0,
            transcription_extent: 0.5,
            translation_flux: 4.0,
            translation_extent: 0.7,
            transcript_species_signal: 48.0,
            protein_species_signal: 64.0,
            assembly_species_signal: 24.0,
        });

        assert!(mixes.transcription_drive_mix.expect("transcription mix") > 0.0);
        assert!(mixes.translation_drive_mix.expect("translation mix") > 0.0);
    }
}
