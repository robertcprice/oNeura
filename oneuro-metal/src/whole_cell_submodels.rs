//! Local chemistry submodels that can feed the native whole-cell runtime.
//!
//! The main whole-cell simulator stays coarse and fast. This module hosts
//! optional submodels that can be sampled periodically:
//! - a batched chemistry lattice for microdomain substrate support
//! - localized molecular dynamics probes for short-range structural signals

use crate::molecular_dynamics::GPUMolecularDynamics;
use crate::substrate_ir::{
    evaluate_patch_assembly, execute_patch_reaction, localize_patch, AssemblyComponent,
    AssemblyContext, AssemblyRule, AssemblyState, FluxChannel, LocalizationCue, LocalizationRule,
    LocalizedPatch, ReactionContext, ReactionLaw, ReactionRule, ReactionTerm, SpatialChannel,
    EMPTY_LOCALIZATION_CUE, EMPTY_REACTION_TERM,
};
use crate::terrarium::{BatchedAtomTerrarium, TerrariumSpecies};
use crate::whole_cell::WholeCellSnapshot;

fn finite_clamped(value: f32, fallback: f32, min_value: f32, max_value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(min_value, max_value)
    } else {
        fallback.clamp(min_value, max_value)
    }
}

fn saturating_signal(value: f32, half_saturation: f32) -> f32 {
    let value = value.max(0.0);
    let half_saturation = half_saturation.max(1.0e-6);
    (value / (value + half_saturation)).clamp(0.0, 1.0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WholeCellChemistrySite {
    Cytosol,
    AtpSynthaseBand,
    RibosomeCluster,
    SeptumRing,
    ChromosomeTrack,
}

impl WholeCellChemistrySite {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cytosol => "cytosol",
            Self::AtpSynthaseBand => "atp_synthase_band",
            Self::RibosomeCluster => "ribosome_cluster",
            Self::SeptumRing => "septum_ring",
            Self::ChromosomeTrack => "chromosome_track",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "cytosol" | "cytoplasm" => Some(Self::Cytosol),
            "atp_synthase_band" | "atp_band" | "respiratory_band" => Some(Self::AtpSynthaseBand),
            "ribosome_cluster" | "ribosome" | "translation" => Some(Self::RibosomeCluster),
            "septum_ring" | "septum" | "division_ring" => Some(Self::SeptumRing),
            "chromosome_track" | "chromosome" | "dna_track" => Some(Self::ChromosomeTrack),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Syn3ASubsystemPreset {
    AtpSynthaseMembraneBand,
    RibosomePolysomeCluster,
    ReplisomeTrack,
    FtsZSeptumRing,
}

impl Syn3ASubsystemPreset {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AtpSynthaseMembraneBand => "atp_synthase_membrane_band",
            Self::RibosomePolysomeCluster => "ribosome_polysome_cluster",
            Self::ReplisomeTrack => "replisome_track",
            Self::FtsZSeptumRing => "ftsz_septum_ring",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "atp_synthase_membrane_band" | "atp_band" | "atp_synthase" => {
                Some(Self::AtpSynthaseMembraneBand)
            }
            "ribosome_polysome_cluster" | "ribosome_cluster" | "polysome" => {
                Some(Self::RibosomePolysomeCluster)
            }
            "replisome_track" | "replisome" | "dna_track" => Some(Self::ReplisomeTrack),
            "ftsz_septum_ring" | "septum_ring" | "ftsz" | "septum" => Some(Self::FtsZSeptumRing),
            _ => None,
        }
    }

    pub fn chemistry_site(self) -> WholeCellChemistrySite {
        match self {
            Self::AtpSynthaseMembraneBand => WholeCellChemistrySite::AtpSynthaseBand,
            Self::RibosomePolysomeCluster => WholeCellChemistrySite::RibosomeCluster,
            Self::ReplisomeTrack => WholeCellChemistrySite::ChromosomeTrack,
            Self::FtsZSeptumRing => WholeCellChemistrySite::SeptumRing,
        }
    }

    pub fn default_interval_steps(self) -> u64 {
        match self {
            Self::AtpSynthaseMembraneBand => 12,
            Self::RibosomePolysomeCluster => 10,
            Self::ReplisomeTrack => 14,
            Self::FtsZSeptumRing => 8,
        }
    }

    pub fn default_probe_request(self) -> LocalMDProbeRequest {
        match self {
            Self::AtpSynthaseMembraneBand => LocalMDProbeRequest {
                site: WholeCellChemistrySite::AtpSynthaseBand,
                n_atoms: 24,
                steps: 12,
                dt_ps: 0.001,
                box_size_angstrom: 16.0,
                temperature_k: 310.0,
            },
            Self::RibosomePolysomeCluster => LocalMDProbeRequest {
                site: WholeCellChemistrySite::RibosomeCluster,
                n_atoms: 36,
                steps: 14,
                dt_ps: 0.001,
                box_size_angstrom: 18.0,
                temperature_k: 310.0,
            },
            Self::ReplisomeTrack => LocalMDProbeRequest {
                site: WholeCellChemistrySite::ChromosomeTrack,
                n_atoms: 28,
                steps: 12,
                dt_ps: 0.001,
                box_size_angstrom: 16.0,
                temperature_k: 310.0,
            },
            Self::FtsZSeptumRing => LocalMDProbeRequest {
                site: WholeCellChemistrySite::SeptumRing,
                n_atoms: 32,
                steps: 14,
                dt_ps: 0.001,
                box_size_angstrom: 17.0,
                temperature_k: 310.0,
            },
        }
    }

    pub fn all() -> &'static [Self] {
        const PRESETS: [Syn3ASubsystemPreset; 4] = [
            Syn3ASubsystemPreset::AtpSynthaseMembraneBand,
            Syn3ASubsystemPreset::RibosomePolysomeCluster,
            Syn3ASubsystemPreset::ReplisomeTrack,
            Syn3ASubsystemPreset::FtsZSeptumRing,
        ];
        &PRESETS
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduledSubsystemProbe {
    pub preset: Syn3ASubsystemPreset,
    pub interval_steps: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct PatchSpeciesMetrics {
    mean_glucose: f32,
    mean_oxygen: f32,
    mean_atp_flux: f32,
    mean_carbon_dioxide: f32,
    mean_nitrate: f32,
    mean_ammonium: f32,
    mean_proton: f32,
    mean_phosphorus: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct LocalPatchSignals {
    oxygen_signal: f32,
    carbon_signal: f32,
    energy_signal: f32,
    nitrogen_signal: f32,
    phosphorus_signal: f32,
    biosynthesis_signal: f32,
    structural_signal: f32,
    stress_signal: f32,
}

impl LocalPatchSignals {
    fn feature(self, idx: usize) -> f32 {
        match idx {
            0 => self.oxygen_signal,
            1 => self.carbon_signal,
            2 => self.energy_signal,
            3 => self.nitrogen_signal,
            4 => self.phosphorus_signal,
            5 => self.biosynthesis_signal,
            6 => self.structural_signal,
            7 => self.stress_signal,
            _ => 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct LocalActivityProfile {
    activity_bias: f32,
    activity_weights: [f32; 8],
    activity_state_weights: [f32; 6],
    activity_min: f32,
    activity_max: f32,
    catalyst_bias: f32,
    catalyst_weights: [f32; 8],
    catalyst_state_weights: [f32; 6],
    catalyst_min: f32,
    catalyst_max: f32,
}

impl LocalActivityProfile {
    fn evaluate_signal(
        bias: f32,
        weights: [f32; 8],
        state_weights: [f32; 6],
        min: f32,
        max: f32,
        signals: LocalPatchSignals,
        state: WholeCellSubsystemState,
    ) -> f32 {
        let state_features = [
            state.atp_scale,
            state.translation_scale,
            state.replication_scale,
            state.segregation_scale,
            state.membrane_scale,
            state.constriction_scale,
        ];
        let mut value = bias;
        for (idx, weight) in weights.iter().enumerate() {
            value += weight * signals.feature(idx);
        }
        for (idx, weight) in state_weights.iter().enumerate() {
            value += weight * state_features[idx];
        }
        value.clamp(min, max)
    }

    fn activity(self, signals: LocalPatchSignals, state: WholeCellSubsystemState) -> f32 {
        Self::evaluate_signal(
            self.activity_bias,
            self.activity_weights,
            self.activity_state_weights,
            self.activity_min,
            self.activity_max,
            signals,
            state,
        )
    }

    fn catalyst(self, signals: LocalPatchSignals, state: WholeCellSubsystemState) -> f32 {
        Self::evaluate_signal(
            self.catalyst_bias,
            self.catalyst_weights,
            self.catalyst_state_weights,
            self.catalyst_min,
            self.catalyst_max,
            signals,
            state,
        )
    }
}

/// Weight order: glucose, oxygen, atp_flux, nitrate, ammonium, phosphorus,
/// assembly_occupancy_delta, assembly_stability_delta.
#[derive(Debug, Clone, Copy, PartialEq)]
struct SupportProfile {
    bias: f32,
    weights: [f32; 8],
    turnover_penalty: f32,
    min: f32,
    max: f32,
}

impl SupportProfile {
    fn evaluate(self, metrics: PatchSpeciesMetrics, assembly: AssemblyState) -> f32 {
        let weighted = self.bias
            + self.weights[0] * metrics.mean_glucose
            + self.weights[1] * metrics.mean_oxygen
            + self.weights[2] * metrics.mean_atp_flux
            + self.weights[3] * metrics.mean_nitrate
            + self.weights[4] * metrics.mean_ammonium
            + self.weights[5] * metrics.mean_phosphorus
            + self.weights[6] * (assembly.occupancy - 1.0)
            + self.weights[7] * (assembly.stability - 1.0)
            - self.turnover_penalty * assembly.turnover;
        weighted.clamp(self.min, self.max)
    }
}

/// Positive weight order: atp, translation, nucleotide, membrane, assembly
/// occupancy delta, assembly stability delta, crowding delta, demand delta.
/// Penalty order: substrate draw, energy draw, biosynthetic draw, byproduct
/// load, assembly turnover.
#[derive(Debug, Clone, Copy, PartialEq)]
struct ScaleProfile {
    baseline: f32,
    weights: [f32; 8],
    penalties: [f32; 5],
    min: f32,
    max: f32,
}

impl ScaleProfile {
    fn evaluate(self, report: LocalChemistrySiteReport) -> f32 {
        let drive = self.baseline
            + self.weights[0] * (report.atp_support - 1.0)
            + self.weights[1] * (report.translation_support - 1.0)
            + self.weights[2] * (report.nucleotide_support - 1.0)
            + self.weights[3] * (report.membrane_support - 1.0)
            + self.weights[4] * (report.assembly_occupancy - 1.0)
            + self.weights[5] * (report.assembly_stability - 1.0)
            + self.weights[6] * (report.crowding_penalty - 1.0)
            + self.weights[7] * (report.demand_satisfaction - 1.0)
            - self.penalties[0] * report.substrate_draw
            - self.penalties[1] * report.energy_draw
            - self.penalties[2] * report.biosynthetic_draw
            - self.penalties[3] * report.byproduct_load
            - self.penalties[4] * report.assembly_turnover;
        drive.clamp(self.min, self.max)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SubsystemCouplingProfile {
    localization_rule: LocalizationRule,
    assembly_rule: AssemblyRule,
    activity_profile: LocalActivityProfile,
    atp_support: SupportProfile,
    translation_support: SupportProfile,
    nucleotide_support: SupportProfile,
    membrane_support: SupportProfile,
    atp_scale: ScaleProfile,
    translation_scale: ScaleProfile,
    replication_scale: ScaleProfile,
    segregation_scale: ScaleProfile,
    membrane_scale: ScaleProfile,
    constriction_scale: ScaleProfile,
}

const NEUTRAL_SCALE_PROFILE: ScaleProfile = ScaleProfile {
    baseline: 1.0,
    weights: [0.0; 8],
    penalties: [0.0; 5],
    min: 0.75,
    max: 1.35,
};

const ATP_BAND_ACTIVITY_PROFILE: LocalActivityProfile = LocalActivityProfile {
    activity_bias: 0.12,
    activity_weights: [0.34, 0.06, 0.38, 0.04, 0.06, 0.10, 0.22, -0.22],
    activity_state_weights: [0.16, 0.00, 0.00, 0.00, 0.10, 0.00],
    activity_min: 0.0,
    activity_max: 3.0,
    catalyst_bias: 0.22,
    catalyst_weights: [0.22, 0.04, 0.26, 0.02, 0.06, 0.06, 0.18, -0.18],
    catalyst_state_weights: [0.20, 0.00, 0.00, 0.00, 0.14, 0.00],
    catalyst_min: 0.4,
    catalyst_max: 1.8,
};

const RIBOSOME_ACTIVITY_PROFILE: LocalActivityProfile = LocalActivityProfile {
    activity_bias: 0.10,
    activity_weights: [0.06, 0.28, 0.24, 0.26, 0.02, 0.24, 0.22, -0.20],
    activity_state_weights: [0.06, 0.18, 0.00, 0.00, 0.00, 0.00],
    activity_min: 0.0,
    activity_max: 3.0,
    catalyst_bias: 0.18,
    catalyst_weights: [0.04, 0.20, 0.18, 0.18, 0.02, 0.16, 0.18, -0.16],
    catalyst_state_weights: [0.04, 0.20, 0.00, 0.00, 0.00, 0.00],
    catalyst_min: 0.4,
    catalyst_max: 1.8,
};

const REPLISOME_ACTIVITY_PROFILE: LocalActivityProfile = LocalActivityProfile {
    activity_bias: 0.08,
    activity_weights: [0.04, 0.10, 0.22, 0.30, 0.04, 0.26, 0.24, -0.18],
    activity_state_weights: [0.04, 0.00, 0.22, 0.08, 0.00, 0.00],
    activity_min: 0.0,
    activity_max: 3.0,
    catalyst_bias: 0.16,
    catalyst_weights: [0.04, 0.08, 0.18, 0.24, 0.04, 0.20, 0.20, -0.16],
    catalyst_state_weights: [0.04, 0.00, 0.24, 0.10, 0.00, 0.00],
    catalyst_min: 0.4,
    catalyst_max: 1.8,
};

const SEPTUM_ACTIVITY_PROFILE: LocalActivityProfile = LocalActivityProfile {
    activity_bias: 0.08,
    activity_weights: [0.08, 0.12, 0.22, 0.04, 0.22, 0.18, 0.26, -0.20],
    activity_state_weights: [0.04, 0.04, 0.00, 0.00, 0.14, 0.18],
    activity_min: 0.0,
    activity_max: 3.0,
    catalyst_bias: 0.16,
    catalyst_weights: [0.08, 0.10, 0.18, 0.02, 0.20, 0.16, 0.22, -0.18],
    catalyst_state_weights: [0.04, 0.04, 0.00, 0.00, 0.16, 0.22],
    catalyst_min: 0.4,
    catalyst_max: 1.8,
};

const ATP_BAND_LOCALIZATION: LocalizationRule = LocalizationRule::new(
    "atp_membrane_localization",
    2,
    5,
    [
        LocalizationCue::new(SpatialChannel::BoundaryProximity, 0.72, 1.0),
        LocalizationCue::new(SpatialChannel::RadialCenterProximity, 0.46, 1.0),
        LocalizationCue::new(
            SpatialChannel::Species(TerrariumSpecies::OxygenGas),
            0.30,
            0.05,
        ),
        LocalizationCue::new(
            SpatialChannel::Species(TerrariumSpecies::AtpFlux),
            0.18,
            0.04,
        ),
        LocalizationCue::new(SpatialChannel::Hydration, -0.12, 1.0),
        EMPTY_LOCALIZATION_CUE,
        EMPTY_LOCALIZATION_CUE,
        EMPTY_LOCALIZATION_CUE,
    ],
    0.16,
    1.0,
    0.95,
);

const RIBOSOME_LOCALIZATION: LocalizationRule = LocalizationRule::new(
    "ribosome_cluster_localization",
    2,
    5,
    [
        LocalizationCue::new(SpatialChannel::CenterProximity, 0.62, 1.0),
        LocalizationCue::new(
            SpatialChannel::Species(TerrariumSpecies::Glucose),
            0.26,
            0.05,
        ),
        LocalizationCue::new(
            SpatialChannel::Species(TerrariumSpecies::Ammonium),
            0.24,
            0.04,
        ),
        LocalizationCue::new(
            SpatialChannel::Species(TerrariumSpecies::AtpFlux),
            0.18,
            0.04,
        ),
        LocalizationCue::new(SpatialChannel::BoundaryProximity, -0.16, 1.0),
        EMPTY_LOCALIZATION_CUE,
        EMPTY_LOCALIZATION_CUE,
        EMPTY_LOCALIZATION_CUE,
    ],
    0.20,
    1.0,
    0.90,
);

const REPLISOME_LOCALIZATION: LocalizationRule = LocalizationRule::new(
    "replisome_track_localization",
    2,
    5,
    [
        LocalizationCue::new(SpatialChannel::CenterProximity, 0.54, 1.0),
        LocalizationCue::new(
            SpatialChannel::Species(TerrariumSpecies::Nitrate),
            0.26,
            0.04,
        ),
        LocalizationCue::new(
            SpatialChannel::Species(TerrariumSpecies::AtpFlux),
            0.20,
            0.04,
        ),
        LocalizationCue::new(
            SpatialChannel::Species(TerrariumSpecies::Glucose),
            0.12,
            0.05,
        ),
        LocalizationCue::new(SpatialChannel::BoundaryProximity, -0.14, 1.0),
        EMPTY_LOCALIZATION_CUE,
        EMPTY_LOCALIZATION_CUE,
        EMPTY_LOCALIZATION_CUE,
    ],
    0.22,
    1.0,
    0.96,
);

const SEPTUM_LOCALIZATION: LocalizationRule = LocalizationRule::new(
    "septum_ring_localization",
    3,
    6,
    [
        LocalizationCue::new(SpatialChannel::VerticalMidplaneProximity, 0.64, 1.0),
        LocalizationCue::new(SpatialChannel::RadialCenterProximity, 0.46, 1.0),
        LocalizationCue::new(
            SpatialChannel::Species(TerrariumSpecies::Phosphorus),
            0.18,
            0.03,
        ),
        LocalizationCue::new(
            SpatialChannel::Species(TerrariumSpecies::AtpFlux),
            0.12,
            0.04,
        ),
        LocalizationCue::new(
            SpatialChannel::Species(TerrariumSpecies::Glucose),
            0.10,
            0.05,
        ),
        LocalizationCue::new(SpatialChannel::BoundaryProximity, -0.12, 1.0),
        EMPTY_LOCALIZATION_CUE,
        EMPTY_LOCALIZATION_CUE,
    ],
    0.24,
    1.4,
    1.05,
);

const ATP_BAND_ASSEMBLY: AssemblyRule = AssemblyRule::new(
    "atp_membrane_assembly",
    4,
    [
        AssemblyComponent::new(TerrariumSpecies::OxygenGas, 1.0, 0.05),
        AssemblyComponent::new(TerrariumSpecies::AtpFlux, 0.9, 0.04),
        AssemblyComponent::new(TerrariumSpecies::Phosphorus, 0.6, 0.03),
        AssemblyComponent::new(TerrariumSpecies::Glucose, 0.3, 0.08),
    ],
    1.08,
    0.46,
    0.08,
);

const RIBOSOME_ASSEMBLY: AssemblyRule = AssemblyRule::new(
    "ribosome_translation_assembly",
    4,
    [
        AssemblyComponent::new(TerrariumSpecies::Ammonium, 1.0, 0.05),
        AssemblyComponent::new(TerrariumSpecies::AtpFlux, 0.9, 0.04),
        AssemblyComponent::new(TerrariumSpecies::Glucose, 0.7, 0.06),
        AssemblyComponent::new(TerrariumSpecies::Nitrate, 0.6, 0.05),
    ],
    1.05,
    0.44,
    0.10,
);

const REPLISOME_ASSEMBLY: AssemblyRule = AssemblyRule::new(
    "replisome_polymer_assembly",
    4,
    [
        AssemblyComponent::new(TerrariumSpecies::Nitrate, 1.0, 0.05),
        AssemblyComponent::new(TerrariumSpecies::AtpFlux, 0.8, 0.04),
        AssemblyComponent::new(TerrariumSpecies::Glucose, 0.5, 0.08),
        AssemblyComponent::new(TerrariumSpecies::Ammonium, 0.3, 0.06),
    ],
    1.04,
    0.45,
    0.08,
);

const SEPTUM_ASSEMBLY: AssemblyRule = AssemblyRule::new(
    "septum_membrane_assembly",
    4,
    [
        AssemblyComponent::new(TerrariumSpecies::Phosphorus, 1.0, 0.03),
        AssemblyComponent::new(TerrariumSpecies::AtpFlux, 0.8, 0.04),
        AssemblyComponent::new(TerrariumSpecies::Glucose, 0.6, 0.06),
        AssemblyComponent::new(TerrariumSpecies::OxygenGas, 0.4, 0.06),
    ],
    1.10,
    0.47,
    0.09,
);

const ATP_BAND_PROFILE: SubsystemCouplingProfile = SubsystemCouplingProfile {
    localization_rule: ATP_BAND_LOCALIZATION,
    assembly_rule: ATP_BAND_ASSEMBLY,
    activity_profile: ATP_BAND_ACTIVITY_PROFILE,
    atp_support: SupportProfile {
        bias: 0.95,
        weights: [0.00, 0.24, 0.08, 0.00, 0.00, 0.00, 0.10, 0.08],
        turnover_penalty: 0.05,
        min: 0.85,
        max: 1.45,
    },
    translation_support: SupportProfile {
        bias: 0.92,
        weights: [0.04, 0.00, 0.03, 0.00, 0.00, 0.00, 0.04, 0.04],
        turnover_penalty: 0.04,
        min: 0.82,
        max: 1.20,
    },
    nucleotide_support: SupportProfile {
        bias: 0.92,
        weights: [0.00, 0.00, 0.04, 0.05, 0.00, 0.00, 0.04, 0.04],
        turnover_penalty: 0.04,
        min: 0.82,
        max: 1.20,
    },
    membrane_support: SupportProfile {
        bias: 0.94,
        weights: [0.00, 0.08, 0.16, 0.00, 0.00, 0.04, 0.12, 0.10],
        turnover_penalty: 0.05,
        min: 0.85,
        max: 1.40,
    },
    atp_scale: ScaleProfile {
        baseline: 1.0,
        weights: [0.88, 0.06, 0.04, 0.12, 0.12, 0.14, 0.18, 0.16],
        penalties: [0.05, 0.14, 0.03, 0.08, 0.06],
        min: 0.75,
        max: 1.45,
    },
    translation_scale: NEUTRAL_SCALE_PROFILE,
    replication_scale: NEUTRAL_SCALE_PROFILE,
    segregation_scale: NEUTRAL_SCALE_PROFILE,
    membrane_scale: ScaleProfile {
        baseline: 1.0,
        weights: [0.12, 0.04, 0.05, 0.82, 0.14, 0.16, 0.18, 0.16],
        penalties: [0.04, 0.03, 0.12, 0.08, 0.06],
        min: 0.75,
        max: 1.40,
    },
    constriction_scale: NEUTRAL_SCALE_PROFILE,
};

const RIBOSOME_PROFILE: SubsystemCouplingProfile = SubsystemCouplingProfile {
    localization_rule: RIBOSOME_LOCALIZATION,
    assembly_rule: RIBOSOME_ASSEMBLY,
    activity_profile: RIBOSOME_ACTIVITY_PROFILE,
    atp_support: SupportProfile {
        bias: 0.92,
        weights: [0.00, 0.05, 0.10, 0.00, 0.00, 0.00, 0.08, 0.08],
        turnover_penalty: 0.05,
        min: 0.82,
        max: 1.30,
    },
    translation_support: SupportProfile {
        bias: 0.95,
        weights: [0.20, 0.00, 0.08, 0.00, 0.10, 0.00, 0.12, 0.10],
        turnover_penalty: 0.06,
        min: 0.85,
        max: 1.45,
    },
    nucleotide_support: SupportProfile {
        bias: 0.92,
        weights: [0.05, 0.00, 0.00, 0.08, 0.00, 0.00, 0.08, 0.06],
        turnover_penalty: 0.05,
        min: 0.82,
        max: 1.25,
    },
    membrane_support: SupportProfile {
        bias: 0.92,
        weights: [0.00, 0.00, 0.08, 0.00, 0.00, 0.04, 0.08, 0.06],
        turnover_penalty: 0.04,
        min: 0.82,
        max: 1.22,
    },
    atp_scale: NEUTRAL_SCALE_PROFILE,
    translation_scale: ScaleProfile {
        baseline: 1.0,
        weights: [0.10, 0.86, 0.08, 0.04, 0.18, 0.20, 0.16, 0.18],
        penalties: [0.12, 0.14, 0.05, 0.10, 0.08],
        min: 0.75,
        max: 1.45,
    },
    replication_scale: NEUTRAL_SCALE_PROFILE,
    segregation_scale: NEUTRAL_SCALE_PROFILE,
    membrane_scale: NEUTRAL_SCALE_PROFILE,
    constriction_scale: NEUTRAL_SCALE_PROFILE,
};

const REPLISOME_PROFILE: SubsystemCouplingProfile = SubsystemCouplingProfile {
    localization_rule: REPLISOME_LOCALIZATION,
    assembly_rule: REPLISOME_ASSEMBLY,
    activity_profile: REPLISOME_ACTIVITY_PROFILE,
    atp_support: SupportProfile {
        bias: 0.92,
        weights: [0.00, 0.04, 0.08, 0.00, 0.00, 0.00, 0.06, 0.06],
        turnover_penalty: 0.04,
        min: 0.82,
        max: 1.25,
    },
    translation_support: SupportProfile {
        bias: 0.92,
        weights: [0.06, 0.00, 0.00, 0.00, 0.03, 0.00, 0.04, 0.04],
        turnover_penalty: 0.04,
        min: 0.82,
        max: 1.18,
    },
    nucleotide_support: SupportProfile {
        bias: 0.96,
        weights: [0.06, 0.00, 0.08, 0.24, 0.00, 0.00, 0.14, 0.12],
        turnover_penalty: 0.06,
        min: 0.85,
        max: 1.45,
    },
    membrane_support: SupportProfile {
        bias: 0.92,
        weights: [0.00, 0.00, 0.05, 0.00, 0.00, 0.05, 0.05, 0.05],
        turnover_penalty: 0.04,
        min: 0.82,
        max: 1.20,
    },
    atp_scale: NEUTRAL_SCALE_PROFILE,
    translation_scale: NEUTRAL_SCALE_PROFILE,
    replication_scale: ScaleProfile {
        baseline: 1.0,
        weights: [0.08, 0.04, 0.90, 0.04, 0.20, 0.22, 0.16, 0.18],
        penalties: [0.08, 0.12, 0.14, 0.10, 0.08],
        min: 0.75,
        max: 1.45,
    },
    segregation_scale: ScaleProfile {
        baseline: 1.0,
        weights: [0.12, 0.02, 0.32, 0.02, 0.18, 0.20, 0.14, 0.16],
        penalties: [0.05, 0.12, 0.08, 0.08, 0.06],
        min: 0.75,
        max: 1.35,
    },
    membrane_scale: NEUTRAL_SCALE_PROFILE,
    constriction_scale: NEUTRAL_SCALE_PROFILE,
};

const SEPTUM_PROFILE: SubsystemCouplingProfile = SubsystemCouplingProfile {
    localization_rule: SEPTUM_LOCALIZATION,
    assembly_rule: SEPTUM_ASSEMBLY,
    activity_profile: SEPTUM_ACTIVITY_PROFILE,
    atp_support: SupportProfile {
        bias: 0.92,
        weights: [0.00, 0.04, 0.10, 0.00, 0.00, 0.00, 0.08, 0.08],
        turnover_penalty: 0.05,
        min: 0.82,
        max: 1.28,
    },
    translation_support: SupportProfile {
        bias: 0.92,
        weights: [0.08, 0.00, 0.04, 0.00, 0.04, 0.00, 0.08, 0.06],
        turnover_penalty: 0.05,
        min: 0.82,
        max: 1.22,
    },
    nucleotide_support: SupportProfile {
        bias: 0.92,
        weights: [0.03, 0.00, 0.00, 0.05, 0.00, 0.00, 0.05, 0.04],
        turnover_penalty: 0.04,
        min: 0.82,
        max: 1.18,
    },
    membrane_support: SupportProfile {
        bias: 0.96,
        weights: [0.00, 0.04, 0.14, 0.00, 0.00, 0.12, 0.14, 0.12],
        turnover_penalty: 0.06,
        min: 0.85,
        max: 1.45,
    },
    atp_scale: NEUTRAL_SCALE_PROFILE,
    translation_scale: ScaleProfile {
        baseline: 1.0,
        weights: [0.04, 0.38, 0.02, 0.04, 0.08, 0.10, 0.10, 0.12],
        penalties: [0.10, 0.06, 0.05, 0.08, 0.06],
        min: 0.78,
        max: 1.20,
    },
    replication_scale: NEUTRAL_SCALE_PROFILE,
    segregation_scale: NEUTRAL_SCALE_PROFILE,
    membrane_scale: ScaleProfile {
        baseline: 1.0,
        weights: [0.08, 0.02, 0.02, 0.86, 0.18, 0.18, 0.16, 0.16],
        penalties: [0.05, 0.05, 0.14, 0.10, 0.08],
        min: 0.75,
        max: 1.40,
    },
    constriction_scale: ScaleProfile {
        baseline: 1.0,
        weights: [0.10, 0.04, 0.02, 0.88, 0.22, 0.24, 0.16, 0.18],
        penalties: [0.05, 0.12, 0.14, 0.10, 0.08],
        min: 0.75,
        max: 1.45,
    },
};

fn subsystem_coupling_profile(preset: Syn3ASubsystemPreset) -> &'static SubsystemCouplingProfile {
    match preset {
        Syn3ASubsystemPreset::AtpSynthaseMembraneBand => &ATP_BAND_PROFILE,
        Syn3ASubsystemPreset::RibosomePolysomeCluster => &RIBOSOME_PROFILE,
        Syn3ASubsystemPreset::ReplisomeTrack => &REPLISOME_PROFILE,
        Syn3ASubsystemPreset::FtsZSeptumRing => &SEPTUM_PROFILE,
    }
}

const ATP_BAND_REACTIONS: [ReactionRule; 1] = [ReactionRule::new(
    "oxidative_energy_transfer",
    2,
    [
        ReactionTerm::new(TerrariumSpecies::OxygenGas, 1.0, FluxChannel::Substrate),
        ReactionTerm::new(TerrariumSpecies::Glucose, 0.35, FluxChannel::Substrate),
        EMPTY_REACTION_TERM,
        EMPTY_REACTION_TERM,
    ],
    2,
    [
        ReactionTerm::new(TerrariumSpecies::AtpFlux, 0.50, FluxChannel::Energy),
        ReactionTerm::new(TerrariumSpecies::Proton, 0.08, FluxChannel::Waste),
        EMPTY_REACTION_TERM,
        EMPTY_REACTION_TERM,
    ],
    ReactionLaw::new(0.012, [0.30, 0.18, 0.00, 0.00, 0.00, 0.24, 0.08, 0.00]),
)];

const RIBOSOME_REACTIONS: [ReactionRule; 1] = [ReactionRule::new(
    "translation_polymerization",
    3,
    [
        ReactionTerm::new(TerrariumSpecies::Glucose, 0.95, FluxChannel::Substrate),
        ReactionTerm::new(TerrariumSpecies::Ammonium, 0.72, FluxChannel::Biosynthetic),
        ReactionTerm::new(TerrariumSpecies::AtpFlux, 0.82, FluxChannel::Energy),
        EMPTY_REACTION_TERM,
    ],
    2,
    [
        ReactionTerm::new(TerrariumSpecies::CarbonDioxide, 0.20, FluxChannel::Waste),
        ReactionTerm::new(TerrariumSpecies::Proton, 0.06, FluxChannel::Waste),
        EMPTY_REACTION_TERM,
        EMPTY_REACTION_TERM,
    ],
    ReactionLaw::new(0.014, [0.34, 0.00, 0.14, 0.00, 0.00, 0.00, 0.10, 0.22]),
)];

const REPLISOME_REACTIONS: [ReactionRule; 1] = [ReactionRule::new(
    "nucleotide_polymerization",
    3,
    [
        ReactionTerm::new(TerrariumSpecies::Nitrate, 1.0, FluxChannel::Biosynthetic),
        ReactionTerm::new(TerrariumSpecies::AtpFlux, 0.72, FluxChannel::Energy),
        ReactionTerm::new(TerrariumSpecies::Glucose, 0.30, FluxChannel::Substrate),
        EMPTY_REACTION_TERM,
    ],
    2,
    [
        ReactionTerm::new(TerrariumSpecies::CarbonDioxide, 0.10, FluxChannel::Waste),
        ReactionTerm::new(TerrariumSpecies::Proton, 0.05, FluxChannel::Waste),
        EMPTY_REACTION_TERM,
        EMPTY_REACTION_TERM,
    ],
    ReactionLaw::new(0.016, [0.24, 0.08, 0.18, 0.28, 0.00, 0.00, 0.06, 0.00]),
)];

const SEPTUM_REACTIONS: [ReactionRule; 1] = [ReactionRule::new(
    "membrane_constriction_assembly",
    4,
    [
        ReactionTerm::new(
            TerrariumSpecies::Phosphorus,
            0.92,
            FluxChannel::Biosynthetic,
        ),
        ReactionTerm::new(TerrariumSpecies::AtpFlux, 0.80, FluxChannel::Energy),
        ReactionTerm::new(TerrariumSpecies::Glucose, 0.52, FluxChannel::Substrate),
        ReactionTerm::new(TerrariumSpecies::OxygenGas, 0.26, FluxChannel::Substrate),
    ],
    2,
    [
        ReactionTerm::new(TerrariumSpecies::Proton, 0.10, FluxChannel::Waste),
        ReactionTerm::new(TerrariumSpecies::CarbonDioxide, 0.08, FluxChannel::Waste),
        EMPTY_REACTION_TERM,
        EMPTY_REACTION_TERM,
    ],
    ReactionLaw::new(0.012, [0.24, 0.10, 0.18, 0.00, 0.26, 0.06, 0.06, 0.08]),
)];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalChemistryReport {
    pub atp_support: f32,
    pub translation_support: f32,
    pub nucleotide_support: f32,
    pub membrane_support: f32,
    pub crowding_penalty: f32,
    pub mean_glucose: f32,
    pub mean_oxygen: f32,
    pub mean_atp_flux: f32,
    pub mean_carbon_dioxide: f32,
}

impl Default for LocalChemistryReport {
    fn default() -> Self {
        Self {
            atp_support: 1.0,
            translation_support: 1.0,
            nucleotide_support: 1.0,
            membrane_support: 1.0,
            crowding_penalty: 1.0,
            mean_glucose: 0.0,
            mean_oxygen: 0.0,
            mean_atp_flux: 0.0,
            mean_carbon_dioxide: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalChemistrySiteReport {
    pub preset: Syn3ASubsystemPreset,
    pub site: WholeCellChemistrySite,
    pub patch_radius: usize,
    pub site_x: usize,
    pub site_y: usize,
    pub site_z: usize,
    pub localization_score: f32,
    pub atp_support: f32,
    pub translation_support: f32,
    pub nucleotide_support: f32,
    pub membrane_support: f32,
    pub crowding_penalty: f32,
    pub mean_glucose: f32,
    pub mean_oxygen: f32,
    pub mean_atp_flux: f32,
    pub mean_carbon_dioxide: f32,
    pub assembly_component_availability: f32,
    pub assembly_occupancy: f32,
    pub assembly_stability: f32,
    pub assembly_turnover: f32,
    pub substrate_draw: f32,
    pub energy_draw: f32,
    pub biosynthetic_draw: f32,
    pub byproduct_load: f32,
    pub demand_satisfaction: f32,
}

impl LocalChemistrySiteReport {
    pub fn as_report(self) -> LocalChemistryReport {
        LocalChemistryReport {
            atp_support: self.atp_support,
            translation_support: self.translation_support,
            nucleotide_support: self.nucleotide_support,
            membrane_support: self.membrane_support,
            crowding_penalty: self.crowding_penalty,
            mean_glucose: self.mean_glucose,
            mean_oxygen: self.mean_oxygen,
            mean_atp_flux: self.mean_atp_flux,
            mean_carbon_dioxide: self.mean_carbon_dioxide,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct LocalChemistryDemandSummary {
    substrate_draw: f32,
    energy_draw: f32,
    biosynthetic_draw: f32,
    byproduct_load: f32,
    demand_satisfaction: f32,
}

impl Default for LocalChemistryDemandSummary {
    fn default() -> Self {
        Self {
            substrate_draw: 0.0,
            energy_draw: 0.0,
            biosynthetic_draw: 0.0,
            byproduct_load: 0.0,
            demand_satisfaction: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SnapshotExchangeTargets {
    reactive_species: [(TerrariumSpecies, f32); 8],
}

impl SnapshotExchangeTargets {
    fn from_snapshot(snapshot: &WholeCellSnapshot) -> Self {
        Self {
            reactive_species: [
                (
                    TerrariumSpecies::Glucose,
                    (0.02 + snapshot.glucose_mm * 0.09).clamp(0.0, 1.2),
                ),
                (
                    TerrariumSpecies::OxygenGas,
                    (0.02 + snapshot.oxygen_mm * 0.10).clamp(0.0, 1.0),
                ),
                (
                    TerrariumSpecies::AtpFlux,
                    (snapshot.atp_mm * 0.14).clamp(0.0, 1.5),
                ),
                (
                    TerrariumSpecies::Ammonium,
                    (0.01 + snapshot.amino_acids_mm * 0.06).clamp(0.0, 0.8),
                ),
                (
                    TerrariumSpecies::Nitrate,
                    (0.01 + snapshot.nucleotides_mm * 0.05).clamp(0.0, 0.8),
                ),
                (
                    TerrariumSpecies::Phosphorus,
                    (0.004 + snapshot.membrane_precursors_mm * 0.025).clamp(0.0, 0.20),
                ),
                (
                    TerrariumSpecies::CarbonDioxide,
                    (0.01 + snapshot.adp_mm * 0.03 + snapshot.division_progress * 0.02)
                        .clamp(0.0, 0.8),
                ),
                (
                    TerrariumSpecies::Proton,
                    (0.002 + snapshot.division_progress * 0.015).clamp(0.0, 0.3),
                ),
            ],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalMDProbeRequest {
    pub site: WholeCellChemistrySite,
    pub n_atoms: usize,
    pub steps: usize,
    pub dt_ps: f32,
    pub box_size_angstrom: f32,
    pub temperature_k: f32,
}

impl LocalMDProbeRequest {
    pub fn new(site: WholeCellChemistrySite) -> Self {
        Self {
            site,
            ..Self::default()
        }
    }
}

impl Default for LocalMDProbeRequest {
    fn default() -> Self {
        Self {
            site: WholeCellChemistrySite::Cytosol,
            n_atoms: 32,
            steps: 32,
            dt_ps: 0.001,
            box_size_angstrom: 18.0,
            temperature_k: 310.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalMDProbeReport {
    pub site: WholeCellChemistrySite,
    pub mean_temperature: f32,
    pub mean_total_energy: f32,
    pub mean_vdw_energy: f32,
    pub mean_electrostatic_energy: f32,
    pub structural_order: f32,
    pub crowding_penalty: f32,
    pub recommended_atp_scale: f32,
    pub recommended_translation_scale: f32,
    pub recommended_replication_scale: f32,
    pub recommended_segregation_scale: f32,
    pub recommended_membrane_scale: f32,
    pub recommended_constriction_scale: f32,
}

/// Persistent per-subsystem coupling state used by the native whole-cell runtime.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WholeCellSubsystemState {
    pub preset: Syn3ASubsystemPreset,
    pub site: WholeCellChemistrySite,
    pub site_x: usize,
    pub site_y: usize,
    pub site_z: usize,
    pub localization_score: f32,
    pub structural_order: f32,
    pub crowding_penalty: f32,
    pub assembly_component_availability: f32,
    pub assembly_occupancy: f32,
    pub assembly_stability: f32,
    pub assembly_turnover: f32,
    pub substrate_draw: f32,
    pub energy_draw: f32,
    pub biosynthetic_draw: f32,
    pub byproduct_load: f32,
    pub demand_satisfaction: f32,
    pub atp_scale: f32,
    pub translation_scale: f32,
    pub replication_scale: f32,
    pub segregation_scale: f32,
    pub membrane_scale: f32,
    pub constriction_scale: f32,
    pub last_probe_step: Option<u64>,
}

impl WholeCellSubsystemState {
    pub fn new(preset: Syn3ASubsystemPreset) -> Self {
        Self {
            preset,
            site: preset.chemistry_site(),
            site_x: 0,
            site_y: 0,
            site_z: 0,
            localization_score: 0.0,
            structural_order: 1.0,
            crowding_penalty: 1.0,
            assembly_component_availability: 1.0,
            assembly_occupancy: 1.0,
            assembly_stability: 1.0,
            assembly_turnover: 0.0,
            substrate_draw: 0.0,
            energy_draw: 0.0,
            biosynthetic_draw: 0.0,
            byproduct_load: 0.0,
            demand_satisfaction: 1.0,
            atp_scale: 1.0,
            translation_scale: 1.0,
            replication_scale: 1.0,
            segregation_scale: 1.0,
            membrane_scale: 1.0,
            constriction_scale: 1.0,
            last_probe_step: None,
        }
    }

    pub fn apply_chemistry_report(&mut self, report: LocalChemistryReport) {
        self.apply_site_report(LocalChemistrySiteReport {
            preset: self.preset,
            site: self.site,
            patch_radius: 1,
            site_x: self.site_x,
            site_y: self.site_y,
            site_z: self.site_z,
            localization_score: self.localization_score,
            atp_support: report.atp_support,
            translation_support: report.translation_support,
            nucleotide_support: report.nucleotide_support,
            membrane_support: report.membrane_support,
            crowding_penalty: report.crowding_penalty,
            mean_glucose: report.mean_glucose,
            mean_oxygen: report.mean_oxygen,
            mean_atp_flux: report.mean_atp_flux,
            mean_carbon_dioxide: report.mean_carbon_dioxide,
            assembly_component_availability: 1.0,
            assembly_occupancy: 1.0,
            assembly_stability: 1.0,
            assembly_turnover: 0.0,
            substrate_draw: 0.0,
            energy_draw: 0.0,
            biosynthetic_draw: 0.0,
            byproduct_load: 0.0,
            demand_satisfaction: 1.0,
        });
    }

    pub fn apply_site_report(&mut self, report: LocalChemistrySiteReport) {
        let profile = match self.preset {
            Syn3ASubsystemPreset::AtpSynthaseMembraneBand => &ATP_BAND_PROFILE,
            Syn3ASubsystemPreset::RibosomePolysomeCluster => &RIBOSOME_PROFILE,
            Syn3ASubsystemPreset::ReplisomeTrack => &REPLISOME_PROFILE,
            Syn3ASubsystemPreset::FtsZSeptumRing => &SEPTUM_PROFILE,
        };
        self.site = report.site;
        self.site_x = report.site_x;
        self.site_y = report.site_y;
        self.site_z = report.site_z;
        self.localization_score = finite_clamped(
            report.localization_score,
            self.localization_score,
            -10.0,
            10.0,
        );
        self.assembly_component_availability = finite_clamped(
            report.assembly_component_availability,
            self.assembly_component_availability,
            0.0,
            1.0,
        );
        self.assembly_occupancy =
            finite_clamped(report.assembly_occupancy, self.assembly_occupancy, 0.0, 1.5);
        self.assembly_stability =
            finite_clamped(report.assembly_stability, self.assembly_stability, 0.0, 1.5);
        self.assembly_turnover =
            finite_clamped(report.assembly_turnover, self.assembly_turnover, 0.0, 1.5);
        self.substrate_draw = finite_clamped(report.substrate_draw, self.substrate_draw, 0.0, 4.0);
        self.energy_draw = finite_clamped(report.energy_draw, self.energy_draw, 0.0, 4.0);
        self.biosynthetic_draw =
            finite_clamped(report.biosynthetic_draw, self.biosynthetic_draw, 0.0, 4.0);
        self.byproduct_load = finite_clamped(report.byproduct_load, self.byproduct_load, 0.0, 4.0);
        self.demand_satisfaction = finite_clamped(
            report.demand_satisfaction,
            self.demand_satisfaction,
            0.35,
            1.0,
        );
        self.crowding_penalty =
            finite_clamped(report.crowding_penalty, self.crowding_penalty, 0.65, 1.0);
        let structural_target = (0.20
            + report.assembly_component_availability * 0.20
            + report.assembly_occupancy * 0.25
            + report.assembly_stability * 0.30
            + report.crowding_penalty * 0.10
            + report.demand_satisfaction * 0.08
            - report.assembly_turnover * 0.10)
            .clamp(0.2, 1.0);
        self.structural_order = finite_clamped(
            0.60 * self.structural_order + 0.40 * structural_target,
            self.structural_order,
            0.2,
            1.0,
        );

        self.atp_scale = finite_clamped(
            0.65 * self.atp_scale + 0.35 * profile.atp_scale.evaluate(report),
            self.atp_scale,
            profile.atp_scale.min,
            profile.atp_scale.max,
        );
        self.translation_scale = finite_clamped(
            0.65 * self.translation_scale + 0.35 * profile.translation_scale.evaluate(report),
            self.translation_scale,
            profile.translation_scale.min,
            profile.translation_scale.max,
        );
        self.replication_scale = finite_clamped(
            0.65 * self.replication_scale + 0.35 * profile.replication_scale.evaluate(report),
            self.replication_scale,
            profile.replication_scale.min,
            profile.replication_scale.max,
        );
        self.segregation_scale = finite_clamped(
            0.65 * self.segregation_scale + 0.35 * profile.segregation_scale.evaluate(report),
            self.segregation_scale,
            profile.segregation_scale.min,
            profile.segregation_scale.max,
        );
        self.membrane_scale = finite_clamped(
            0.65 * self.membrane_scale + 0.35 * profile.membrane_scale.evaluate(report),
            self.membrane_scale,
            profile.membrane_scale.min,
            profile.membrane_scale.max,
        );
        self.constriction_scale = finite_clamped(
            0.65 * self.constriction_scale + 0.35 * profile.constriction_scale.evaluate(report),
            self.constriction_scale,
            profile.constriction_scale.min,
            profile.constriction_scale.max,
        );
    }

    pub fn apply_probe_report(&mut self, report: LocalMDProbeReport, step_count: u64) {
        self.site = report.site;
        self.structural_order =
            finite_clamped(report.structural_order, self.structural_order, 0.2, 1.0);
        self.crowding_penalty =
            finite_clamped(report.crowding_penalty, self.crowding_penalty, 0.65, 1.0);
        self.atp_scale = finite_clamped(
            0.55 * self.atp_scale + 0.45 * report.recommended_atp_scale,
            self.atp_scale,
            0.70,
            1.45,
        );
        self.translation_scale = finite_clamped(
            0.55 * self.translation_scale + 0.45 * report.recommended_translation_scale,
            self.translation_scale,
            0.70,
            1.45,
        );
        self.replication_scale = finite_clamped(
            0.55 * self.replication_scale + 0.45 * report.recommended_replication_scale,
            self.replication_scale,
            0.70,
            1.45,
        );
        self.segregation_scale = finite_clamped(
            0.55 * self.segregation_scale + 0.45 * report.recommended_segregation_scale,
            self.segregation_scale,
            0.70,
            1.45,
        );
        self.membrane_scale = finite_clamped(
            0.55 * self.membrane_scale + 0.45 * report.recommended_membrane_scale,
            self.membrane_scale,
            0.70,
            1.45,
        );
        self.constriction_scale = finite_clamped(
            0.55 * self.constriction_scale + 0.45 * report.recommended_constriction_scale,
            self.constriction_scale,
            0.70,
            1.45,
        );
        self.last_probe_step = Some(step_count);
    }
}

pub struct WholeCellChemistryBridge {
    substrate: BatchedAtomTerrarium,
    last_report: LocalChemistryReport,
    last_site_reports: Vec<LocalChemistrySiteReport>,
    last_md_report: Option<LocalMDProbeReport>,
    exchange_targets: SnapshotExchangeTargets,
    initialized_from_snapshot: bool,
}

impl WholeCellChemistryBridge {
    pub fn new(
        x_dim: usize,
        y_dim: usize,
        z_dim: usize,
        voxel_size_au: f32,
        use_gpu: bool,
    ) -> Self {
        Self {
            substrate: BatchedAtomTerrarium::new(x_dim, y_dim, z_dim, voxel_size_au, use_gpu),
            last_report: LocalChemistryReport::default(),
            last_site_reports: Vec::new(),
            last_md_report: None,
            exchange_targets: SnapshotExchangeTargets {
                reactive_species: [
                    (TerrariumSpecies::Glucose, 0.0),
                    (TerrariumSpecies::OxygenGas, 0.0),
                    (TerrariumSpecies::AtpFlux, 0.0),
                    (TerrariumSpecies::Ammonium, 0.0),
                    (TerrariumSpecies::Nitrate, 0.0),
                    (TerrariumSpecies::Phosphorus, 0.0),
                    (TerrariumSpecies::CarbonDioxide, 0.0),
                    (TerrariumSpecies::Proton, 0.0),
                ],
            },
            initialized_from_snapshot: false,
        }
    }

    pub fn synchronize_from_snapshot(&mut self, snapshot: &WholeCellSnapshot) {
        self.exchange_targets = SnapshotExchangeTargets::from_snapshot(snapshot);
        if !self.initialized_from_snapshot {
            for (species, target) in self.exchange_targets.reactive_species {
                self.substrate.fill_species(species, target);
            }
            self.initialized_from_snapshot = true;
        }
    }

    fn apply_bulk_exchange(&mut self, dt_ms: f32) {
        if !self.initialized_from_snapshot {
            return;
        }

        let exchange_fraction = (1.0 - (-dt_ms.max(0.0) / 8.0).exp()).clamp(0.0, 0.28);
        for (species, target) in self.exchange_targets.reactive_species {
            self.substrate
                .relax_species_toward(species, target, exchange_fraction);
        }
    }

    fn subsystem_state_from_snapshot(
        snapshot: &WholeCellSnapshot,
        preset: Syn3ASubsystemPreset,
    ) -> WholeCellSubsystemState {
        snapshot
            .subsystem_states
            .iter()
            .copied()
            .find(|state| state.preset == preset)
            .unwrap_or_else(|| WholeCellSubsystemState::new(preset))
    }

    fn previous_localized_patch(
        &self,
        snapshot: Option<&WholeCellSnapshot>,
        preset: Syn3ASubsystemPreset,
    ) -> Option<LocalizedPatch> {
        let radius = subsystem_coupling_profile(preset)
            .localization_rule
            .patch_radius;
        snapshot
            .and_then(|snap| {
                let state = Self::subsystem_state_from_snapshot(snap, preset);
                if state.localization_score.is_finite()
                    && (state.localization_score.abs() > 1.0e-6
                        || state.site_x != 0
                        || state.site_y != 0
                        || state.site_z != 0)
                {
                    Some(LocalizedPatch {
                        x: state.site_x,
                        y: state.site_y,
                        z: state.site_z,
                        radius,
                        score: state.localization_score,
                    })
                } else {
                    None
                }
            })
            .or_else(|| {
                self.last_site_reports
                    .iter()
                    .find(|report| report.preset == preset)
                    .map(|report| LocalizedPatch {
                        x: report.site_x,
                        y: report.site_y,
                        z: report.site_z,
                        radius: report.patch_radius,
                        score: report.localization_score,
                    })
            })
    }

    fn resolve_localized_patch(
        &self,
        snapshot: Option<&WholeCellSnapshot>,
        preset: Syn3ASubsystemPreset,
        occupied_sites: &[LocalizedPatch],
    ) -> LocalizedPatch {
        let profile = subsystem_coupling_profile(preset);
        localize_patch(
            &self.substrate,
            profile.localization_rule,
            self.previous_localized_patch(snapshot, preset),
            occupied_sites,
        )
    }

    fn patch_species_metrics(
        &self,
        x: usize,
        y: usize,
        z: usize,
        radius: usize,
    ) -> PatchSpeciesMetrics {
        PatchSpeciesMetrics {
            mean_glucose: self.substrate.patch_mean_species(
                TerrariumSpecies::Glucose,
                x,
                y,
                z,
                radius,
            ),
            mean_oxygen: self.substrate.patch_mean_species(
                TerrariumSpecies::OxygenGas,
                x,
                y,
                z,
                radius,
            ),
            mean_atp_flux: self.substrate.patch_mean_species(
                TerrariumSpecies::AtpFlux,
                x,
                y,
                z,
                radius,
            ),
            mean_carbon_dioxide: self.substrate.patch_mean_species(
                TerrariumSpecies::CarbonDioxide,
                x,
                y,
                z,
                radius,
            ),
            mean_nitrate: self.substrate.patch_mean_species(
                TerrariumSpecies::Nitrate,
                x,
                y,
                z,
                radius,
            ),
            mean_ammonium: self.substrate.patch_mean_species(
                TerrariumSpecies::Ammonium,
                x,
                y,
                z,
                radius,
            ),
            mean_proton: self.substrate.patch_mean_species(
                TerrariumSpecies::Proton,
                x,
                y,
                z,
                radius,
            ),
            mean_phosphorus: self.substrate.patch_mean_species(
                TerrariumSpecies::Phosphorus,
                x,
                y,
                z,
                radius,
            ),
        }
    }

    fn patch_crowding_penalty(metrics: PatchSpeciesMetrics) -> f32 {
        let acidity_penalty = (1.0 / (1.0 + metrics.mean_proton * 3.8)).clamp(0.72, 1.0);
        let carbon_penalty = (1.0 / (1.0 + metrics.mean_carbon_dioxide * 1.9)).clamp(0.76, 1.0);
        (acidity_penalty * carbon_penalty).clamp(0.68, 1.0)
    }

    fn base_assembly_context(
        metrics: PatchSpeciesMetrics,
        state: WholeCellSubsystemState,
        demand: LocalChemistryDemandSummary,
        crowding_penalty: f32,
    ) -> AssemblyContext {
        let oxygen_signal = saturating_signal(metrics.mean_oxygen, 0.05);
        let carbon_signal = saturating_signal(metrics.mean_glucose, 0.05);
        let energy_signal = (0.75 * saturating_signal(metrics.mean_atp_flux, 0.04)
            + 0.25 * oxygen_signal)
            .clamp(0.0, 1.0);
        let nitrogen_signal = (0.55 * saturating_signal(metrics.mean_ammonium, 0.04)
            + 0.45 * saturating_signal(metrics.mean_nitrate, 0.04))
        .clamp(0.0, 1.0);
        let phosphorus_signal = saturating_signal(metrics.mean_phosphorus, 0.03);
        let stress_signal = (0.45 * saturating_signal(metrics.mean_carbon_dioxide, 0.04)
            + 0.30 * saturating_signal(metrics.mean_proton, 0.01)
            + 0.25 * (1.0 - crowding_penalty))
            .clamp(0.0, 1.0);
        let biosynthesis_signal = (0.30 * carbon_signal
            + 0.32 * nitrogen_signal
            + 0.14 * phosphorus_signal
            + 0.24 * energy_signal)
            .clamp(0.0, 1.0);

        AssemblyContext {
            catalyst_scale: (0.30
                + 0.18 * state.structural_order
                + 0.14 * state.atp_scale
                + 0.10 * state.translation_scale
                + 0.10 * state.replication_scale
                + 0.10 * state.membrane_scale
                + 0.08 * state.constriction_scale
                + 0.12 * energy_signal
                - 0.10 * stress_signal)
                .clamp(0.25, 1.8),
            support_scale: (0.34
                + 0.18 * oxygen_signal
                + 0.18 * carbon_signal
                + 0.16 * nitrogen_signal
                + 0.10 * phosphorus_signal
                + 0.18 * biosynthesis_signal
                + 0.10 * state.structural_order
                + 0.10 * demand.demand_satisfaction
                - 0.10 * stress_signal
                - 0.05 * demand.byproduct_load)
                .clamp(0.25, 1.6),
            demand_satisfaction: demand.demand_satisfaction,
            crowding_penalty,
            byproduct_load: demand.byproduct_load,
            substrate_pressure: demand.substrate_draw,
            energy_pressure: demand.energy_draw,
            biosynthetic_pressure: demand.biosynthetic_draw,
        }
    }

    fn local_patch_signals(
        metrics: PatchSpeciesMetrics,
        assembly: AssemblyState,
        state: WholeCellSubsystemState,
        crowding_penalty: f32,
    ) -> LocalPatchSignals {
        let oxygen_signal = saturating_signal(metrics.mean_oxygen, 0.05);
        let carbon_signal = saturating_signal(metrics.mean_glucose, 0.05);
        let energy_signal = (0.72 * saturating_signal(metrics.mean_atp_flux, 0.04)
            + 0.28 * oxygen_signal)
            .clamp(0.0, 1.0);
        let nitrogen_signal = (0.55 * saturating_signal(metrics.mean_ammonium, 0.04)
            + 0.45 * saturating_signal(metrics.mean_nitrate, 0.04))
        .clamp(0.0, 1.0);
        let phosphorus_signal = saturating_signal(metrics.mean_phosphorus, 0.03);
        let assembly_availability = saturating_signal(assembly.component_availability, 0.35);
        let assembly_occupancy = saturating_signal(assembly.occupancy, 0.55);
        let assembly_stability = saturating_signal(assembly.stability, 0.55);
        let assembly_turnover = saturating_signal(assembly.turnover, 0.30);
        let structural_signal = (0.20 * assembly_availability
            + 0.28 * assembly_occupancy
            + 0.30 * assembly_stability
            + 0.12 * state.structural_order
            + 0.10 * crowding_penalty
            - 0.18 * assembly_turnover)
            .clamp(0.0, 1.2);
        let biosynthesis_signal = (0.28 * carbon_signal
            + 0.28 * nitrogen_signal
            + 0.16 * phosphorus_signal
            + 0.16 * energy_signal
            + 0.12 * structural_signal)
            .clamp(0.0, 1.2);
        let stress_signal = (0.32 * saturating_signal(metrics.mean_carbon_dioxide, 0.04)
            + 0.24 * saturating_signal(metrics.mean_proton, 0.01)
            + 0.18 * assembly_turnover
            + 0.14 * (1.0 - crowding_penalty)
            + 0.12 * (1.0 - state.demand_satisfaction.clamp(0.0, 1.0)))
        .clamp(0.0, 1.2);

        LocalPatchSignals {
            oxygen_signal,
            carbon_signal,
            energy_signal,
            nitrogen_signal,
            phosphorus_signal,
            biosynthesis_signal,
            structural_signal,
            stress_signal,
        }
    }

    fn reaction_rules_for_preset(preset: Syn3ASubsystemPreset) -> &'static [ReactionRule] {
        match preset {
            Syn3ASubsystemPreset::AtpSynthaseMembraneBand => &ATP_BAND_REACTIONS,
            Syn3ASubsystemPreset::RibosomePolysomeCluster => &RIBOSOME_REACTIONS,
            Syn3ASubsystemPreset::ReplisomeTrack => &REPLISOME_REACTIONS,
            Syn3ASubsystemPreset::FtsZSeptumRing => &SEPTUM_REACTIONS,
        }
    }

    fn reaction_context_from_local_patch(
        preset: Syn3ASubsystemPreset,
        signals: LocalPatchSignals,
        state: WholeCellSubsystemState,
    ) -> ReactionContext {
        let profile = subsystem_coupling_profile(preset);
        let activity = profile.activity_profile.activity(signals, state);
        let catalyst_scale = profile.activity_profile.catalyst(signals, state);
        let energy_driver =
            (signals.energy_signal * (0.82 + 0.18 * state.atp_scale)).clamp(0.0, 3.0);
        let biosynthesis_driver = (signals.biosynthesis_signal
            * (0.72
                + 0.08 * state.translation_scale
                + 0.10 * state.replication_scale
                + 0.10 * state.membrane_scale))
            .clamp(0.0, 4.0);
        let replication_driver = (0.42 * signals.biosynthesis_signal
            + 0.18 * signals.energy_signal
            + 0.26 * signals.structural_signal
            + 0.14 * state.replication_scale
            - 0.14 * signals.stress_signal)
            .clamp(0.0, 3.0);
        let division_driver = (0.18 * signals.energy_signal
            + 0.24 * signals.phosphorus_signal
            + 0.24 * signals.structural_signal
            + 0.14 * state.membrane_scale
            + 0.16 * state.constriction_scale
            - 0.14 * signals.stress_signal)
            .clamp(0.0, 3.0);
        let translation_driver = (0.30 * signals.carbon_signal
            + 0.28 * signals.nitrogen_signal
            + 0.18 * signals.energy_signal
            + 0.14 * signals.structural_signal
            + 0.10 * state.translation_scale
            - 0.12 * signals.stress_signal)
            .clamp(0.0, 3.0);

        ReactionContext {
            catalyst_scale,
            drivers: [
                activity,
                energy_driver,
                biosynthesis_driver,
                replication_driver,
                division_driver,
                signals.oxygen_signal.clamp(0.0, 3.0),
                signals.carbon_signal.clamp(0.0, 3.0),
                translation_driver,
            ],
        }
    }

    fn apply_subsystem_demand(
        &mut self,
        snapshot: &WholeCellSnapshot,
        preset: Syn3ASubsystemPreset,
        patch: LocalizedPatch,
        dt_ms: f32,
    ) -> LocalChemistryDemandSummary {
        let state = Self::subsystem_state_from_snapshot(snapshot, preset);
        let demand_window = dt_ms.max(0.05);
        let metrics = self.patch_species_metrics(patch.x, patch.y, patch.z, patch.radius);
        let crowding_penalty = Self::patch_crowding_penalty(metrics);
        let assembly = evaluate_patch_assembly(
            &self.substrate,
            patch.x,
            patch.y,
            patch.z,
            patch.radius,
            subsystem_coupling_profile(preset).assembly_rule,
            Self::base_assembly_context(
                metrics,
                state,
                LocalChemistryDemandSummary::default(),
                crowding_penalty,
            ),
        );
        let context = Self::reaction_context_from_local_patch(
            preset,
            Self::local_patch_signals(metrics, assembly, state, crowding_penalty),
            state,
        );

        let mut substrate_draw = 0.0;
        let mut energy_draw = 0.0;
        let mut biosynthetic_draw = 0.0;
        let mut byproduct_load = 0.0;
        let mut removed_total = 0.0;
        let mut target_total = 0.0;

        for rule in Self::reaction_rules_for_preset(preset) {
            let flux = execute_patch_reaction(
                &mut self.substrate,
                patch.x,
                patch.y,
                patch.z,
                patch.radius,
                *rule,
                context,
                demand_window,
            );
            substrate_draw += flux.substrate_draw;
            energy_draw += flux.energy_draw;
            biosynthetic_draw += flux.biosynthetic_draw;
            byproduct_load += flux.byproduct_load;
            removed_total += flux.removed_total;
            target_total += flux.target_total;
        }

        LocalChemistryDemandSummary {
            substrate_draw,
            energy_draw,
            biosynthetic_draw,
            byproduct_load,
            demand_satisfaction: if target_total <= 1.0e-6 {
                1.0
            } else {
                (removed_total / target_total).clamp(0.0, 1.0)
            },
        }
    }

    fn site_support_report(
        &self,
        snapshot: Option<&WholeCellSnapshot>,
        preset: Syn3ASubsystemPreset,
        patch: LocalizedPatch,
        demand: LocalChemistryDemandSummary,
    ) -> LocalChemistrySiteReport {
        let profile = subsystem_coupling_profile(preset);
        let metrics = self.patch_species_metrics(patch.x, patch.y, patch.z, patch.radius);
        let crowding_penalty = Self::patch_crowding_penalty(metrics);
        let state = snapshot
            .map(|snap| Self::subsystem_state_from_snapshot(snap, preset))
            .unwrap_or_else(|| WholeCellSubsystemState::new(preset));
        let assembly_context =
            Self::base_assembly_context(metrics, state, demand, crowding_penalty);
        let assembly = evaluate_patch_assembly(
            &self.substrate,
            patch.x,
            patch.y,
            patch.z,
            patch.radius,
            profile.assembly_rule,
            assembly_context,
        );
        let atp_support = profile.atp_support.evaluate(metrics, assembly);
        let translation_support = profile.translation_support.evaluate(metrics, assembly);
        let nucleotide_support = profile.nucleotide_support.evaluate(metrics, assembly);
        let membrane_support = profile.membrane_support.evaluate(metrics, assembly);

        LocalChemistrySiteReport {
            preset,
            site: preset.chemistry_site(),
            patch_radius: patch.radius,
            site_x: patch.x,
            site_y: patch.y,
            site_z: patch.z,
            localization_score: patch.score,
            atp_support,
            translation_support,
            nucleotide_support,
            membrane_support,
            crowding_penalty,
            mean_glucose: metrics.mean_glucose,
            mean_oxygen: metrics.mean_oxygen,
            mean_atp_flux: metrics.mean_atp_flux,
            mean_carbon_dioxide: metrics.mean_carbon_dioxide,
            assembly_component_availability: assembly.component_availability,
            assembly_occupancy: assembly.occupancy,
            assembly_stability: assembly.stability,
            assembly_turnover: assembly.turnover,
            substrate_draw: demand.substrate_draw,
            energy_draw: demand.energy_draw,
            biosynthetic_draw: demand.biosynthetic_draw,
            byproduct_load: demand.byproduct_load,
            demand_satisfaction: demand.demand_satisfaction,
        }
    }

    pub fn step_with_snapshot(
        &mut self,
        dt_ms: f32,
        snapshot: Option<&WholeCellSnapshot>,
    ) -> LocalChemistryReport {
        if let Some(snapshot) = snapshot {
            self.synchronize_from_snapshot(snapshot);
        }

        let mut localized_sites = Vec::with_capacity(Syn3ASubsystemPreset::all().len());
        let mut occupied_sites = Vec::with_capacity(Syn3ASubsystemPreset::all().len());
        for preset in Syn3ASubsystemPreset::all().iter().copied() {
            let patch = self.resolve_localized_patch(snapshot, preset, &occupied_sites);
            occupied_sites.push(patch);
            localized_sites.push((preset, patch));
        }

        let mut site_demands = Vec::with_capacity(localized_sites.len());
        for (preset, patch) in &localized_sites {
            let demand = if let Some(snapshot) = snapshot {
                self.apply_subsystem_demand(snapshot, *preset, *patch, dt_ms)
            } else {
                LocalChemistryDemandSummary::default()
            };
            site_demands.push((*preset, *patch, demand));
        }

        self.substrate.step(dt_ms);
        self.apply_bulk_exchange(dt_ms);

        let mean_glucose = self.substrate.mean_species(TerrariumSpecies::Glucose);
        let mean_oxygen = self.substrate.mean_species(TerrariumSpecies::OxygenGas);
        let mean_atp_flux = self.substrate.mean_species(TerrariumSpecies::AtpFlux);
        let mean_co2 = self.substrate.mean_species(TerrariumSpecies::CarbonDioxide);
        let mean_nitrate = self.substrate.mean_species(TerrariumSpecies::Nitrate);
        let mean_ammonium = self.substrate.mean_species(TerrariumSpecies::Ammonium);
        let mean_proton = self.substrate.mean_species(TerrariumSpecies::Proton);

        let acidity_penalty = (1.0 / (1.0 + mean_proton * 3.5)).clamp(0.75, 1.0);
        let carbon_penalty = (1.0 / (1.0 + mean_co2 * 1.8)).clamp(0.78, 1.0);
        let crowding_penalty = (acidity_penalty * carbon_penalty).clamp(0.70, 1.0);

        let report = LocalChemistryReport {
            atp_support: (0.92 + mean_oxygen * 0.18 + mean_atp_flux * 0.06).clamp(0.85, 1.35),
            translation_support: (0.92 + mean_glucose * 0.16 + mean_ammonium * 0.06)
                .clamp(0.85, 1.35),
            nucleotide_support: (0.92 + mean_nitrate * 0.16 + mean_glucose * 0.04)
                .clamp(0.85, 1.35),
            membrane_support: (0.92 + mean_atp_flux * 0.10 + mean_oxygen * 0.05).clamp(0.85, 1.30),
            crowding_penalty,
            mean_glucose,
            mean_oxygen,
            mean_atp_flux,
            mean_carbon_dioxide: mean_co2,
        };

        self.last_report = report;
        self.last_site_reports = site_demands
            .into_iter()
            .map(|(preset, patch, demand)| {
                self.site_support_report(snapshot, preset, patch, demand)
            })
            .collect();
        report
    }

    pub fn step(&mut self, dt_ms: f32) -> LocalChemistryReport {
        self.step_with_snapshot(dt_ms, None)
    }

    pub fn last_report(&self) -> LocalChemistryReport {
        self.last_report
    }

    pub fn last_md_report(&self) -> Option<LocalMDProbeReport> {
        self.last_md_report
    }

    pub fn site_reports(&self) -> Vec<LocalChemistrySiteReport> {
        self.last_site_reports.clone()
    }

    pub fn run_md_probe(&mut self, request: LocalMDProbeRequest) -> LocalMDProbeReport {
        let mut md = GPUMolecularDynamics::new(request.n_atoms.max(8), "auto");
        let n_atoms = request.n_atoms.max(8);
        let box_size = request.box_size_angstrom.max(8.0);
        let center = box_size * 0.5;

        let mut positions = vec![0.0f32; n_atoms * 3];
        let mut masses = vec![12.0f32; n_atoms];
        let mut charges = vec![0.0f32; n_atoms];
        let mut sigma = vec![3.3f32; n_atoms];
        let mut epsilon = vec![0.10f32; n_atoms];

        for i in 0..n_atoms {
            let i3 = i * 3;
            let frac = i as f32 / n_atoms as f32;
            let angle = frac * std::f32::consts::TAU * 1.618;
            let radial = (0.20 + 0.45 * frac) * box_size * 0.5;
            positions[i3] = center + radial * angle.cos();
            positions[i3 + 1] = center + radial * angle.sin();
            positions[i3 + 2] = center + (((i % 5) as f32) - 2.0) * 0.8;

            match request.site {
                WholeCellChemistrySite::Cytosol => {
                    masses[i] = if i % 3 == 0 { 16.0 } else { 12.0 };
                    charges[i] = if i % 4 == 0 { -0.1 } else { 0.05 };
                    sigma[i] = 3.1 + (i % 3) as f32 * 0.15;
                    epsilon[i] = 0.08 + (i % 2) as f32 * 0.02;
                }
                WholeCellChemistrySite::AtpSynthaseBand => {
                    masses[i] = if i % 2 == 0 { 16.0 } else { 1.0 };
                    charges[i] = if i % 2 == 0 { -0.25 } else { 0.15 };
                    sigma[i] = 3.2;
                    epsilon[i] = 0.14;
                }
                WholeCellChemistrySite::RibosomeCluster => {
                    masses[i] = if i % 5 == 0 { 31.0 } else { 14.0 };
                    charges[i] = if i % 3 == 0 { -0.20 } else { 0.10 };
                    sigma[i] = 3.6;
                    epsilon[i] = 0.16;
                }
                WholeCellChemistrySite::SeptumRing => {
                    masses[i] = if i % 2 == 0 { 28.0 } else { 16.0 };
                    charges[i] = if i % 2 == 0 { 0.12 } else { -0.08 };
                    sigma[i] = 4.0;
                    epsilon[i] = 0.20;
                }
                WholeCellChemistrySite::ChromosomeTrack => {
                    masses[i] = 12.0;
                    charges[i] = if i % 3 == 0 { -0.25 } else { -0.05 };
                    sigma[i] = 3.4;
                    epsilon[i] = 0.12;
                }
            }
        }

        md.set_positions(&positions);
        md.set_masses(&masses);
        md.set_charges(&charges);
        md.set_lj_params(&sigma, &epsilon);
        md.set_box([box_size, box_size, box_size]);
        md.set_temperature(request.temperature_k);

        if matches!(
            request.site,
            WholeCellChemistrySite::RibosomeCluster
                | WholeCellChemistrySite::SeptumRing
                | WholeCellChemistrySite::ChromosomeTrack
        ) {
            for i in 0..(n_atoms - 1) {
                let (r0, k) = match request.site {
                    WholeCellChemistrySite::RibosomeCluster => (2.4, 20.0),
                    WholeCellChemistrySite::SeptumRing => (2.0, 36.0),
                    WholeCellChemistrySite::ChromosomeTrack => (1.7, 28.0),
                    _ => (2.2, 18.0),
                };
                md.add_bond(i, i + 1, r0, k);
            }
        }

        md.initialize_velocities();

        let mut temp_acc = 0.0;
        let mut total_acc = 0.0;
        let mut vdw_acc = 0.0;
        let mut elec_acc = 0.0;
        for _ in 0..request.steps.max(1) {
            let stats = md.step(request.dt_ps.max(0.0001));
            temp_acc += stats.temperature;
            total_acc += stats.total_energy;
            vdw_acc += stats.vdw_energy;
            elec_acc += stats.electrostatic_energy;
        }

        let steps_f = request.steps.max(1) as f32;
        let positions = md.positions();
        let mut mean_radius = 0.0;
        let mut mean_radius_sq = 0.0;
        for i in 0..n_atoms {
            let i3 = i * 3;
            let dx = positions[i3] - center;
            let dy = positions[i3 + 1] - center;
            let dz = positions[i3 + 2] - center;
            let radius = (dx * dx + dy * dy + dz * dz).sqrt();
            mean_radius += radius;
            mean_radius_sq += radius * radius;
        }
        mean_radius /= n_atoms as f32;
        mean_radius_sq /= n_atoms as f32;
        let variance = (mean_radius_sq - mean_radius * mean_radius).max(0.0);
        let structural_order = (1.0 / (1.0 + variance / (box_size * 0.5).max(1.0))).clamp(0.2, 1.0);

        let mean_temperature = temp_acc / steps_f;
        let mean_total_energy = total_acc / steps_f;
        let mean_vdw_energy = vdw_acc / steps_f;
        let mean_electrostatic_energy = elec_acc / steps_f;
        let crowding_penalty = (1.0
            / (1.0 + mean_vdw_energy.abs() / 150.0 + mean_total_energy.abs() / 600.0))
            .clamp(0.65, 1.0);

        let (
            recommended_atp_scale,
            recommended_translation_scale,
            recommended_replication_scale,
            recommended_segregation_scale,
            recommended_membrane_scale,
            recommended_constriction_scale,
        ) = match request.site {
            WholeCellChemistrySite::RibosomeCluster => (
                (1.0 + structural_order * 0.04) * crowding_penalty,
                (1.0 + structural_order * 0.18) * crowding_penalty,
                (1.0 + structural_order * 0.03) * crowding_penalty,
                (1.0 + structural_order * 0.02) * crowding_penalty,
                (1.0 + structural_order * 0.04) * crowding_penalty,
                (1.0 + structural_order * 0.03) * crowding_penalty,
            ),
            WholeCellChemistrySite::SeptumRing => (
                (1.00 + structural_order * 0.05) * (0.94 + crowding_penalty * 0.10),
                (1.00 + structural_order * 0.10) * (0.94 + crowding_penalty * 0.08),
                (1.00 + structural_order * 0.03) * (0.94 + crowding_penalty * 0.08),
                (1.00 + structural_order * 0.08) * (0.94 + crowding_penalty * 0.10),
                (1.02 + structural_order * 0.24) * (0.94 + crowding_penalty * 0.12),
                (1.04 + structural_order * 0.28) * (0.94 + crowding_penalty * 0.14),
            ),
            WholeCellChemistrySite::AtpSynthaseBand => (
                (1.03 + structural_order * 0.20) * (0.94 + crowding_penalty * 0.10),
                (1.00 + structural_order * 0.05) * (0.94 + crowding_penalty * 0.08),
                (1.00 + structural_order * 0.02) * (0.94 + crowding_penalty * 0.06),
                (1.00 + structural_order * 0.03) * (0.94 + crowding_penalty * 0.08),
                (1.01 + structural_order * 0.14) * (0.94 + crowding_penalty * 0.08),
                (1.00 + structural_order * 0.06) * (0.94 + crowding_penalty * 0.06),
            ),
            WholeCellChemistrySite::ChromosomeTrack => (
                (1.0 + structural_order * 0.03) * crowding_penalty,
                (1.0 + structural_order * 0.03) * crowding_penalty,
                (1.02 + structural_order * 0.26) * (0.92 + crowding_penalty * 0.16),
                (1.00 + structural_order * 0.20) * (0.92 + crowding_penalty * 0.14),
                (1.0 + structural_order * 0.02) * crowding_penalty,
                (1.0 + structural_order * 0.03) * crowding_penalty,
            ),
            WholeCellChemistrySite::Cytosol => (
                (1.0 + structural_order * 0.07) * crowding_penalty,
                (1.0 + structural_order * 0.08) * crowding_penalty,
                (1.0 + structural_order * 0.06) * crowding_penalty,
                (1.0 + structural_order * 0.05) * crowding_penalty,
                (1.0 + structural_order * 0.05) * crowding_penalty,
                (1.0 + structural_order * 0.04) * crowding_penalty,
            ),
        };

        let report = LocalMDProbeReport {
            site: request.site,
            mean_temperature,
            mean_total_energy,
            mean_vdw_energy,
            mean_electrostatic_energy,
            structural_order,
            crowding_penalty,
            recommended_atp_scale: recommended_atp_scale.clamp(0.75, 1.35),
            recommended_translation_scale: recommended_translation_scale.clamp(0.75, 1.35),
            recommended_replication_scale: recommended_replication_scale.clamp(0.75, 1.35),
            recommended_segregation_scale: recommended_segregation_scale.clamp(0.75, 1.35),
            recommended_membrane_scale: recommended_membrane_scale.clamp(0.75, 1.35),
            recommended_constriction_scale: recommended_constriction_scale.clamp(0.75, 1.35),
        };
        self.last_md_report = Some(report);
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::whole_cell::{WholeCellConfig, WholeCellSimulator};

    #[test]
    fn chemistry_bridge_generates_support_report() {
        let sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            ..WholeCellConfig::default()
        });
        let snap = sim.snapshot();
        let mut bridge = WholeCellChemistryBridge::new(12, 12, 6, 0.5, false);
        bridge.synchronize_from_snapshot(&snap);
        let report = bridge.step(0.25);

        assert!(report.atp_support > 0.0);
        assert!(report.translation_support > 0.0);
        assert!(report.nucleotide_support > 0.0);
        assert!(report.crowding_penalty > 0.0);
        let site_reports = bridge.site_reports();
        assert_eq!(site_reports.len(), Syn3ASubsystemPreset::all().len());
        assert!(site_reports.iter().all(|report| report.patch_radius > 0));
    }

    #[test]
    fn chemistry_bridge_applies_localized_substrate_demand() {
        let sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            dt_ms: 0.25,
            ..WholeCellConfig::default()
        });
        let snap = sim.snapshot();
        let mut bridge = WholeCellChemistryBridge::new(12, 12, 6, 0.5, false);
        bridge.synchronize_from_snapshot(&snap);
        bridge.step_with_snapshot(0.25, Some(&snap));

        let site_reports = bridge.site_reports();
        let ribosome = site_reports
            .iter()
            .find(|report| report.preset == Syn3ASubsystemPreset::RibosomePolysomeCluster)
            .expect("ribosome site report");
        let replisome = site_reports
            .iter()
            .find(|report| report.preset == Syn3ASubsystemPreset::ReplisomeTrack)
            .expect("replisome site report");

        assert!(ribosome.substrate_draw > 0.0);
        assert!(ribosome.energy_draw > 0.0);
        assert!(ribosome.demand_satisfaction > 0.0);
        assert!(replisome.biosynthetic_draw > 0.0);
    }

    #[test]
    fn snapshot_resync_preserves_local_depletion_memory() {
        let sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            dt_ms: 0.25,
            ..WholeCellConfig::default()
        });
        let snap = sim.snapshot();

        let mut persisted = WholeCellChemistryBridge::new(12, 12, 6, 0.5, false);
        persisted.synchronize_from_snapshot(&snap);
        persisted.step_with_snapshot(0.25, Some(&snap));
        persisted.synchronize_from_snapshot(&snap);
        persisted.step(0.05);

        let persisted_ribosome = persisted
            .site_reports()
            .into_iter()
            .find(|report| report.preset == Syn3ASubsystemPreset::RibosomePolysomeCluster)
            .expect("persisted ribosome report");

        let mut fresh = WholeCellChemistryBridge::new(12, 12, 6, 0.5, false);
        fresh.synchronize_from_snapshot(&snap);
        fresh.step(0.05);
        let fresh_ribosome = fresh
            .site_reports()
            .into_iter()
            .find(|report| report.preset == Syn3ASubsystemPreset::RibosomePolysomeCluster)
            .expect("fresh ribosome report");

        assert!(
            persisted_ribosome.mean_carbon_dioxide > fresh_ribosome.mean_carbon_dioxide,
            "persisted co2={} fresh co2={}",
            persisted_ribosome.mean_carbon_dioxide,
            fresh_ribosome.mean_carbon_dioxide
        );
        assert!(
            persisted_ribosome.crowding_penalty <= fresh_ribosome.crowding_penalty,
            "persisted crowding={} fresh crowding={}",
            persisted_ribosome.crowding_penalty,
            fresh_ribosome.crowding_penalty
        );
    }

    #[test]
    fn subsystem_state_responds_to_local_depletion_pressure() {
        let rich = LocalChemistrySiteReport {
            preset: Syn3ASubsystemPreset::RibosomePolysomeCluster,
            site: WholeCellChemistrySite::RibosomeCluster,
            patch_radius: 2,
            site_x: 4,
            site_y: 4,
            site_z: 2,
            localization_score: 1.10,
            atp_support: 1.05,
            translation_support: 1.28,
            nucleotide_support: 1.02,
            membrane_support: 1.00,
            crowding_penalty: 0.98,
            mean_glucose: 0.35,
            mean_oxygen: 0.22,
            mean_atp_flux: 0.30,
            mean_carbon_dioxide: 0.02,
            assembly_component_availability: 1.08,
            assembly_occupancy: 1.04,
            assembly_stability: 1.02,
            assembly_turnover: 0.03,
            substrate_draw: 0.04,
            energy_draw: 0.03,
            biosynthetic_draw: 0.02,
            byproduct_load: 0.02,
            demand_satisfaction: 1.0,
        };
        let depleted = LocalChemistrySiteReport {
            substrate_draw: 0.55,
            energy_draw: 0.48,
            biosynthetic_draw: 0.20,
            byproduct_load: 0.42,
            demand_satisfaction: 0.42,
            ..rich
        };

        let mut rich_state =
            WholeCellSubsystemState::new(Syn3ASubsystemPreset::RibosomePolysomeCluster);
        rich_state.apply_site_report(rich);

        let mut depleted_state =
            WholeCellSubsystemState::new(Syn3ASubsystemPreset::RibosomePolysomeCluster);
        depleted_state.apply_site_report(depleted);

        assert!(depleted_state.translation_scale < rich_state.translation_scale);
        assert!(depleted_state.demand_satisfaction < rich_state.demand_satisfaction);
        assert!(depleted_state.byproduct_load > rich_state.byproduct_load);
    }

    #[test]
    fn local_md_probe_returns_finite_metrics() {
        let mut bridge = WholeCellChemistryBridge::new(8, 8, 4, 0.5, false);
        let report = bridge.run_md_probe(LocalMDProbeRequest {
            site: WholeCellChemistrySite::RibosomeCluster,
            n_atoms: 16,
            steps: 8,
            dt_ps: 0.001,
            box_size_angstrom: 14.0,
            temperature_k: 310.0,
        });

        assert!(report.mean_temperature.is_finite());
        assert!(report.mean_total_energy.is_finite());
        assert!(report.structural_order > 0.0);
        assert!(report.crowding_penalty > 0.0);
    }

    #[test]
    fn demand_driven_reactions_shift_local_pools() {
        let sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            ..WholeCellConfig::default()
        });
        let snap = sim.snapshot();
        let mut bridge = WholeCellChemistryBridge::new(12, 12, 6, 0.5, false);
        bridge.synchronize_from_snapshot(&snap);
        let before = bridge.step(0.10);

        let after = bridge.step_with_snapshot(0.25, Some(&snap));

        assert!(after.mean_carbon_dioxide >= before.mean_carbon_dioxide);
        assert!(
            after.mean_glucose != before.mean_glucose
                || after.mean_atp_flux != before.mean_atp_flux
        );
    }

    #[test]
    fn localization_tracks_hotspots_instead_of_fixed_anchors() {
        let sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            ..WholeCellConfig::default()
        });
        let snap = sim.snapshot();
        let mut bridge = WholeCellChemistryBridge::new(12, 12, 6, 0.5, false);
        bridge.synchronize_from_snapshot(&snap);

        let bottom_z = bridge.substrate.z_dim.saturating_sub(1);
        bridge
            .substrate
            .add_hotspot(TerrariumSpecies::OxygenGas, 6, 6, bottom_z, 3.0);

        let patch = bridge.resolve_localized_patch(
            Some(&snap),
            Syn3ASubsystemPreset::AtpSynthaseMembraneBand,
            &[],
        );
        assert!(patch.z >= bridge.substrate.z_dim.saturating_sub(2));
        assert!(patch.score.is_finite());
    }

    #[test]
    fn demand_is_driven_by_local_patch_state_not_snapshot_counters() {
        let sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            dt_ms: 0.25,
            ..WholeCellConfig::default()
        });
        let snapshot = sim.snapshot();
        let mut altered_snapshot = snapshot.clone();
        altered_snapshot.active_ribosomes = 2.0;
        altered_snapshot.dnaa = 1.0;
        altered_snapshot.ftsz = 1.0;

        let mut baseline_bridge = WholeCellChemistryBridge::new(12, 12, 6, 0.5, false);
        baseline_bridge.synchronize_from_snapshot(&snapshot);
        let baseline_patch = baseline_bridge.resolve_localized_patch(
            Some(&snapshot),
            Syn3ASubsystemPreset::RibosomePolysomeCluster,
            &[],
        );
        let baseline_demand = baseline_bridge.apply_subsystem_demand(
            &snapshot,
            Syn3ASubsystemPreset::RibosomePolysomeCluster,
            baseline_patch,
            0.25,
        );

        let mut altered_bridge = WholeCellChemistryBridge::new(12, 12, 6, 0.5, false);
        altered_bridge.synchronize_from_snapshot(&altered_snapshot);
        let altered_patch = altered_bridge.resolve_localized_patch(
            Some(&altered_snapshot),
            Syn3ASubsystemPreset::RibosomePolysomeCluster,
            &[],
        );
        let altered_demand = altered_bridge.apply_subsystem_demand(
            &altered_snapshot,
            Syn3ASubsystemPreset::RibosomePolysomeCluster,
            altered_patch,
            0.25,
        );

        assert_eq!(
            (baseline_patch.x, baseline_patch.y, baseline_patch.z),
            (altered_patch.x, altered_patch.y, altered_patch.z)
        );
        assert!((baseline_demand.substrate_draw - altered_demand.substrate_draw).abs() < 1.0e-6);
        assert!((baseline_demand.energy_draw - altered_demand.energy_draw).abs() < 1.0e-6);
        assert!(
            (baseline_demand.biosynthetic_draw - altered_demand.biosynthetic_draw).abs() < 1.0e-6
        );
    }

    #[test]
    fn site_reports_capture_distinct_microdomains() {
        let sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            ..WholeCellConfig::default()
        });
        let snap = sim.snapshot();
        let mut bridge = WholeCellChemistryBridge::new(12, 12, 6, 0.5, false);
        bridge.synchronize_from_snapshot(&snap);
        bridge.step_with_snapshot(0.20, Some(&snap));

        let site_reports = bridge.site_reports();
        let atp_band = site_reports
            .iter()
            .find(|report| report.preset == Syn3ASubsystemPreset::AtpSynthaseMembraneBand)
            .expect("atp-band site report");
        let ribosome = site_reports
            .iter()
            .find(|report| report.preset == Syn3ASubsystemPreset::RibosomePolysomeCluster)
            .expect("ribosome site report");
        let unique_sites = site_reports
            .iter()
            .map(|report| (report.site_x, report.site_y, report.site_z))
            .collect::<std::collections::HashSet<_>>();

        assert!(
            atp_band.atp_support > ribosome.atp_support
                || atp_band.mean_oxygen > ribosome.mean_oxygen
        );
        assert!(ribosome.translation_support >= atp_band.translation_support);
        assert!(atp_band.demand_satisfaction > 0.0);
        assert!(ribosome.substrate_draw > 0.0);
        assert!(atp_band.localization_score.is_finite());
        assert!(unique_sites.len() > 1);
    }
}
