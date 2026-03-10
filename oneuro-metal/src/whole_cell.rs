//! Native whole-cell runtime focused on minimal bacterial cells.
//!
//! This is a coarse but performance-oriented native core: voxelized
//! intracellular RDME on a GPU-ready lattice, plus staged CME/ODE/BD/geometry
//! updates in Rust. It is meant to replace the Python skeleton as the
//! performance-critical simulation path.

use std::f32::consts::PI;

use crate::gpu;
use crate::gpu::whole_cell_rdme::{
    cpu_whole_cell_rdme, dispatch_whole_cell_rdme, IntracellularLattice, IntracellularSpecies,
};
use crate::substrate_ir::{
    ScalarBranch, ScalarContext, ScalarFactor, ScalarRule, EMPTY_SCALAR_BRANCH, EMPTY_SCALAR_FACTOR,
};
use crate::whole_cell_submodels::{
    LocalChemistryReport, LocalChemistrySiteReport, LocalMDProbeReport, LocalMDProbeRequest,
    ScheduledSubsystemProbe, Syn3ASubsystemPreset, WholeCellChemistryBridge,
    WholeCellSubsystemState,
};

#[cfg(target_os = "macos")]
use crate::gpu::GpuContext;

/// Execution backend chosen for the simulator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WholeCellBackend {
    Cpu,
    Metal,
}

impl WholeCellBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            WholeCellBackend::Cpu => "cpu",
            WholeCellBackend::Metal => "metal",
        }
    }
}

/// Optional quantum-chemistry correction profile supplied by nQPU or another
/// external chemistry backend. The runtime uses these as multiplicative
/// modifiers rather than pushing Python or quantum logic into the hot loop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WholeCellQuantumProfile {
    pub oxphos_efficiency: f32,
    pub translation_efficiency: f32,
    pub nucleotide_polymerization_efficiency: f32,
    pub membrane_synthesis_efficiency: f32,
    pub chromosome_segregation_efficiency: f32,
}

impl Default for WholeCellQuantumProfile {
    fn default() -> Self {
        Self {
            oxphos_efficiency: 1.0,
            translation_efficiency: 1.0,
            nucleotide_polymerization_efficiency: 1.0,
            membrane_synthesis_efficiency: 1.0,
            chromosome_segregation_efficiency: 1.0,
        }
    }
}

impl WholeCellQuantumProfile {
    fn normalized(self) -> Self {
        Self {
            oxphos_efficiency: self.oxphos_efficiency.clamp(0.5, 2.5),
            translation_efficiency: self.translation_efficiency.clamp(0.5, 2.5),
            nucleotide_polymerization_efficiency: self
                .nucleotide_polymerization_efficiency
                .clamp(0.5, 2.5),
            membrane_synthesis_efficiency: self.membrane_synthesis_efficiency.clamp(0.5, 2.5),
            chromosome_segregation_efficiency: self
                .chromosome_segregation_efficiency
                .clamp(0.5, 2.5),
        }
    }
}

/// Static runtime configuration for the whole-cell simulator.
#[derive(Debug, Clone)]
pub struct WholeCellConfig {
    pub x_dim: usize,
    pub y_dim: usize,
    pub z_dim: usize,
    pub voxel_size_nm: f32,
    pub dt_ms: f32,
    pub cme_interval: u64,
    pub ode_interval: u64,
    pub bd_interval: u64,
    pub geometry_interval: u64,
    pub use_gpu: bool,
}

impl Default for WholeCellConfig {
    fn default() -> Self {
        Self {
            x_dim: 24,
            y_dim: 24,
            z_dim: 12,
            voxel_size_nm: 20.0,
            dt_ms: 0.25,
            cme_interval: 4,
            ode_interval: 1,
            bd_interval: 2,
            geometry_interval: 4,
            use_gpu: true,
        }
    }
}

/// Flattened view of the current simulation state.
#[derive(Debug, Clone)]
pub struct WholeCellSnapshot {
    pub backend: WholeCellBackend,
    pub time_ms: f32,
    pub step_count: u64,
    pub atp_mm: f32,
    pub amino_acids_mm: f32,
    pub nucleotides_mm: f32,
    pub membrane_precursors_mm: f32,
    pub adp_mm: f32,
    pub glucose_mm: f32,
    pub oxygen_mm: f32,
    pub ftsz: f32,
    pub dnaa: f32,
    pub active_ribosomes: f32,
    pub active_rnap: f32,
    pub genome_bp: u32,
    pub replicated_bp: u32,
    pub chromosome_separation_nm: f32,
    pub radius_nm: f32,
    pub surface_area_nm2: f32,
    pub volume_nm3: f32,
    pub division_progress: f32,
    pub quantum_profile: WholeCellQuantumProfile,
    pub local_chemistry: Option<LocalChemistryReport>,
    pub local_chemistry_sites: Vec<LocalChemistrySiteReport>,
    pub local_md_probe: Option<LocalMDProbeReport>,
    pub subsystem_states: Vec<WholeCellSubsystemState>,
}

#[derive(Debug, Clone, Copy)]
struct WholeCellAssemblyInventory {
    atp_band_complexes: f32,
    ribosome_complexes: f32,
    rnap_complexes: f32,
    replisome_complexes: f32,
    membrane_complexes: f32,
    ftsz_polymer: f32,
    dnaa_activity: f32,
}

#[derive(Debug, Clone, Copy)]
struct WholeCellProcessFluxes {
    energy_capacity: f32,
    transcription_capacity: f32,
    translation_capacity: f32,
    replication_capacity: f32,
    segregation_capacity: f32,
    membrane_capacity: f32,
    constriction_capacity: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
enum WholeCellRuleSignal {
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
    MembranePrecursorFloor,
    AtpBandComplexes,
    RibosomeComplexes,
    RnapComplexes,
    ReplisomeComplexes,
    MembraneComplexes,
    FtszPolymer,
    DnaaActivity,
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
    DnaaSignal,
    ReplisomeAssemblySignal,
    ConstrictionSignal,
    TranscriptionDriveMix,
    TranslationDriveMix,
    BiosyntheticLoadMix,
    ConstrictionFlux,
}

impl WholeCellRuleSignal {
    const COUNT: usize = Self::ConstrictionFlux as usize + 1;
}

#[derive(Debug, Clone, Copy)]
struct WholeCellRuleContext {
    signals: [f32; WholeCellRuleSignal::COUNT],
}

impl Default for WholeCellRuleContext {
    fn default() -> Self {
        Self {
            signals: [0.0; WholeCellRuleSignal::COUNT],
        }
    }
}

impl WholeCellRuleContext {
    fn set(&mut self, signal: WholeCellRuleSignal, value: f32) {
        self.signals[signal as usize] = if value.is_finite() {
            value.max(0.0)
        } else {
            0.0
        };
    }

    fn get(self, signal: WholeCellRuleSignal) -> f32 {
        self.signals[signal as usize]
    }

    fn scalar(self) -> ScalarContext<{ WholeCellRuleSignal::COUNT }> {
        ScalarContext {
            signals: self.signals,
        }
    }
}

const fn scalar_factor(signal: WholeCellRuleSignal, bias: f32, scale: f32) -> ScalarFactor {
    ScalarFactor::new(signal as usize, bias, scale, 1.0)
}

const fn scalar_branch_1(f1: ScalarFactor, coefficient: f32) -> ScalarBranch {
    ScalarBranch::new(
        coefficient,
        1,
        [
            f1,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
        ],
    )
}

const fn scalar_branch_2(f1: ScalarFactor, f2: ScalarFactor, coefficient: f32) -> ScalarBranch {
    ScalarBranch::new(
        coefficient,
        2,
        [
            f1,
            f2,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
        ],
    )
}

const fn scalar_branch_3(
    f1: ScalarFactor,
    f2: ScalarFactor,
    f3: ScalarFactor,
    coefficient: f32,
) -> ScalarBranch {
    ScalarBranch::new(
        coefficient,
        3,
        [
            f1,
            f2,
            f3,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
        ],
    )
}

const fn scalar_branch_4(
    f1: ScalarFactor,
    f2: ScalarFactor,
    f3: ScalarFactor,
    f4: ScalarFactor,
    coefficient: f32,
) -> ScalarBranch {
    ScalarBranch::new(
        coefficient,
        4,
        [
            f1,
            f2,
            f3,
            f4,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
        ],
    )
}

const fn scalar_branch_5(
    f1: ScalarFactor,
    f2: ScalarFactor,
    f3: ScalarFactor,
    f4: ScalarFactor,
    f5: ScalarFactor,
    coefficient: f32,
) -> ScalarBranch {
    ScalarBranch::new(
        coefficient,
        5,
        [
            f1,
            f2,
            f3,
            f4,
            f5,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
        ],
    )
}

const fn scalar_branch_6(
    f1: ScalarFactor,
    f2: ScalarFactor,
    f3: ScalarFactor,
    f4: ScalarFactor,
    f5: ScalarFactor,
    f6: ScalarFactor,
    coefficient: f32,
) -> ScalarBranch {
    ScalarBranch::new(
        coefficient,
        6,
        [
            f1,
            f2,
            f3,
            f4,
            f5,
            f6,
            EMPTY_SCALAR_FACTOR,
            EMPTY_SCALAR_FACTOR,
        ],
    )
}

const fn scalar_branch_7(
    f1: ScalarFactor,
    f2: ScalarFactor,
    f3: ScalarFactor,
    f4: ScalarFactor,
    f5: ScalarFactor,
    f6: ScalarFactor,
    f7: ScalarFactor,
    coefficient: f32,
) -> ScalarBranch {
    ScalarBranch::new(
        coefficient,
        7,
        [f1, f2, f3, f4, f5, f6, f7, EMPTY_SCALAR_FACTOR],
    )
}

const fn scalar_branch_8(
    f1: ScalarFactor,
    f2: ScalarFactor,
    f3: ScalarFactor,
    f4: ScalarFactor,
    f5: ScalarFactor,
    f6: ScalarFactor,
    f7: ScalarFactor,
    f8: ScalarFactor,
    coefficient: f32,
) -> ScalarBranch {
    ScalarBranch::new(coefficient, 8, [f1, f2, f3, f4, f5, f6, f7, f8])
}

const ATP_BAND_INVENTORY_RULE: ScalarRule = ScalarRule::new(
    18.0,
    1,
    [
        scalar_branch_4(
            scalar_factor(WholeCellRuleSignal::AtpBandSignal, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::OxygenSignal, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::GlucoseSignal, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::EnergySignal, 0.70, 0.30),
            30.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    6.0,
    128.0,
);

const RIBOSOME_INVENTORY_RULE: ScalarRule = ScalarRule::new(
    18.0,
    1,
    [
        scalar_branch_4(
            scalar_factor(WholeCellRuleSignal::RibosomeSignal, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::AminoSignal, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::GlucoseSignal, 0.65, 0.35),
            scalar_factor(WholeCellRuleSignal::EnergySignal, 0.60, 0.40),
            42.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    12.0,
    256.0,
);

const RNAP_INVENTORY_RULE: ScalarRule = ScalarRule::new(
    10.0,
    1,
    [
        scalar_branch_3(
            scalar_factor(
                WholeCellRuleSignal::WeightedRibosomeReplisomeSignal,
                0.0,
                1.0,
            ),
            scalar_factor(WholeCellRuleSignal::NucleotideSignal, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::EnergySignal, 0.70, 0.30),
            20.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    8.0,
    128.0,
);

const REPLISOME_INVENTORY_RULE: ScalarRule = ScalarRule::new(
    8.0,
    1,
    [
        scalar_branch_4(
            scalar_factor(WholeCellRuleSignal::ReplisomeSignal, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::NucleotideSignal, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::EnergySignal, 0.68, 0.32),
            scalar_factor(WholeCellRuleSignal::InverseReplicatedFraction, 0.85, 0.15),
            26.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    4.0,
    160.0,
);

const MEMBRANE_INVENTORY_RULE: ScalarRule = ScalarRule::new(
    14.0,
    1,
    [
        scalar_branch_4(
            scalar_factor(WholeCellRuleSignal::WeightedAtpSeptumSignal, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::MembraneSignal, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::OxygenSignal, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::EnergySignal, 0.65, 0.35),
            26.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    6.0,
    160.0,
);

const FTSZ_POLYMER_RULE: ScalarRule = ScalarRule::new(
    28.0,
    1,
    [
        scalar_branch_4(
            scalar_factor(WholeCellRuleSignal::SeptumSignal, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::MembraneSignal, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::EnergySignal, 0.58, 0.42),
            scalar_factor(WholeCellRuleSignal::DivisionReadiness, 0.50, 0.50),
            78.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    12.0,
    320.0,
);

const DNAA_ACTIVITY_RULE: ScalarRule = ScalarRule::new(
    20.0,
    1,
    [
        scalar_branch_4(
            scalar_factor(WholeCellRuleSignal::ReplisomeSignal, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::NucleotideSignal, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::EnergySignal, 0.62, 0.38),
            scalar_factor(WholeCellRuleSignal::InverseReplicatedFraction, 0.80, 0.20),
            56.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    8.0,
    192.0,
);

const ENERGY_CAPACITY_RULE: ScalarRule = ScalarRule::new(
    0.0,
    1,
    [
        scalar_branch_6(
            scalar_factor(WholeCellRuleSignal::AtpBandComplexes, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::AtpSupport, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::AtpBandScale, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::LocalizedSupplyScale, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::OxygenSignal, 0.55, 0.45),
            scalar_factor(WholeCellRuleSignal::GlucoseSignal, 0.55, 0.45),
            1.0 / 24.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    0.40,
    2.40,
);

const TRANSCRIPTION_CAPACITY_RULE: ScalarRule = ScalarRule::new(
    0.0,
    1,
    [
        scalar_branch_4(
            scalar_factor(WholeCellRuleSignal::RnapComplexes, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::NucleotideSupport, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::LocalizedSupplyScale, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::EnergySignal, 0.55, 0.45),
            1.0 / 24.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    0.35,
    2.40,
);

const TRANSLATION_CAPACITY_RULE: ScalarRule = ScalarRule::new(
    0.0,
    1,
    [
        scalar_branch_6(
            scalar_factor(WholeCellRuleSignal::RibosomeComplexes, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::TranslationSupport, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::RibosomeTranslationScale, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::MdTranslationScale, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::LocalizedSupplyScale, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::AminoSignal, 0.55, 0.45),
            1.0 / 46.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    0.35,
    2.60,
);

const REPLICATION_CAPACITY_RULE: ScalarRule = ScalarRule::new(
    0.0,
    1,
    [
        scalar_branch_4(
            scalar_factor(WholeCellRuleSignal::ReplisomeComplexes, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::ReplisomeReplicationScale, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::NucleotideSupport, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::NucleotideSignal, 0.55, 0.45),
            1.0 / 28.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    0.30,
    2.40,
);

const SEGREGATION_CAPACITY_RULE: ScalarRule = ScalarRule::new(
    0.0,
    1,
    [
        scalar_branch_4(
            scalar_factor(WholeCellRuleSignal::ReplisomeComplexes, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::ReplisomeSegregationScale, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::QuantumSegregationEfficiency, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::EnergySignal, 0.60, 0.40),
            1.0 / 28.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    0.30,
    2.40,
);

const MEMBRANE_CAPACITY_RULE: ScalarRule = ScalarRule::new(
    0.0,
    1,
    [
        scalar_branch_5(
            scalar_factor(WholeCellRuleSignal::MembraneComplexes, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::MembraneSupport, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::MembraneAssemblyScale, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::MdMembraneScale, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::MembraneSignal, 0.55, 0.45),
            1.0 / 24.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    0.35,
    2.40,
);

const CONSTRICTION_CAPACITY_RULE: ScalarRule = ScalarRule::new(
    0.0,
    1,
    [
        scalar_branch_5(
            scalar_factor(WholeCellRuleSignal::FtszPolymer, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::FtszConstrictionScale, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::MembraneSupport, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::MdTranslationScale, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::MembraneSignal, 0.55, 0.45),
            1.0 / 90.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    0.30,
    2.60,
);

const TRANSCRIPTION_FLUX_RULE: ScalarRule = ScalarRule::new(
    0.0,
    1,
    [
        scalar_branch_7(
            scalar_factor(WholeCellRuleSignal::Dt, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::TranscriptionCapacity, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::NucleotideSignal, 0.55, 0.45),
            scalar_factor(WholeCellRuleSignal::TranscriptionDriveMix, 0.50, 1.0),
            scalar_factor(WholeCellRuleSignal::QuantumNucleotideEfficiency, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::CrowdingPenalty, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::InverseReplicatedFraction, 0.65, 0.35),
            0.060,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    0.0,
    10.0,
);

const TRANSLATION_FLUX_RULE: ScalarRule = ScalarRule::new(
    0.0,
    1,
    [
        scalar_branch_7(
            scalar_factor(WholeCellRuleSignal::Dt, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::TranslationCapacity, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::AminoSignal, 0.55, 0.45),
            scalar_factor(WholeCellRuleSignal::TranslationDriveMix, 0.55, 1.0),
            scalar_factor(WholeCellRuleSignal::QuantumTranslationEfficiency, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::CrowdingPenalty, 0.0, 1.0),
            scalar_factor(
                WholeCellRuleSignal::TranscriptionCapacityCapped16,
                0.65,
                0.35,
            ),
            0.085,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    0.0,
    10.0,
);

const ENERGY_GAIN_RULE: ScalarRule = ScalarRule::new(
    0.0,
    1,
    [
        scalar_branch_5(
            scalar_factor(WholeCellRuleSignal::Dt, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::EnergyCapacity, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::QuantumOxphosEfficiency, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::GlucoseSignal, 0.55, 0.45),
            scalar_factor(WholeCellRuleSignal::OxygenSignal, 0.55, 0.45),
            0.0155,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    0.0,
    10.0,
);

const ENERGY_COST_RULE: ScalarRule = ScalarRule::new(
    0.0,
    1,
    [
        scalar_branch_3(
            scalar_factor(WholeCellRuleSignal::Dt, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::EffectiveMetabolicLoad, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::BiosyntheticLoadMix, 0.34, 1.0),
            0.010,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    0.0,
    10.0,
);

const NUCLEOTIDE_RECHARGE_RULE: ScalarRule = ScalarRule::new(
    0.0,
    1,
    [
        scalar_branch_5(
            scalar_factor(WholeCellRuleSignal::Dt, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::EnergyCapacity, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::NucleotideSignal, 0.55, 0.45),
            scalar_factor(WholeCellRuleSignal::QuantumNucleotideEfficiency, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::GlucoseSignal, 0.55, 0.45),
            0.0032,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    0.0,
    10.0,
);

const MEMBRANE_FLUX_RULE: ScalarRule = ScalarRule::new(
    0.0,
    1,
    [
        scalar_branch_5(
            scalar_factor(WholeCellRuleSignal::Dt, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::MembraneCapacity, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::QuantumMembraneEfficiency, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::MembraneSignal, 0.55, 0.45),
            scalar_factor(WholeCellRuleSignal::EnergyCapacityCapped18, 0.55, 0.45),
            0.0028,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    0.0,
    10.0,
);

const REPLICATION_DRIVE_RULE: ScalarRule = ScalarRule::new(
    0.0,
    1,
    [
        scalar_branch_8(
            scalar_factor(WholeCellRuleSignal::Dt, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::ReplicationCapacity, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::DnaaSignal, 0.55, 0.45),
            scalar_factor(WholeCellRuleSignal::NucleotideSignal, 0.55, 0.45),
            scalar_factor(WholeCellRuleSignal::ReplisomeAssemblySignal, 0.55, 0.45),
            scalar_factor(WholeCellRuleSignal::QuantumNucleotideEfficiency, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::NucleotideSupport, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::CrowdingPenalty, 0.0, 1.0),
            18.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    0.0,
    1000.0,
);

const SEGREGATION_STEP_RULE: ScalarRule = ScalarRule::new(
    0.0,
    1,
    [
        scalar_branch_4(
            scalar_factor(WholeCellRuleSignal::Dt, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::SegregationCapacity, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::ReplisomeAssemblySignal, 0.55, 0.45),
            scalar_factor(WholeCellRuleSignal::ReplicatedFraction, 3.0, 18.0),
            1.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    0.0,
    1000.0,
);

const MEMBRANE_GROWTH_RULE: ScalarRule = ScalarRule::new(
    0.0,
    1,
    [
        scalar_branch_4(
            scalar_factor(WholeCellRuleSignal::Dt, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::MembranePrecursorFloor, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::QuantumMembraneEfficiency, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::MembraneCapacity, 0.0, 1.0),
            14.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    0.0,
    1000.0,
);

const CONSTRICTION_FLUX_RULE: ScalarRule = ScalarRule::new(
    0.0,
    1,
    [
        scalar_branch_5(
            scalar_factor(WholeCellRuleSignal::Dt, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::ConstrictionCapacity, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::ConstrictionSignal, 0.55, 0.45),
            scalar_factor(WholeCellRuleSignal::MembraneSignal, 0.55, 0.45),
            scalar_factor(WholeCellRuleSignal::ReplicatedFraction, 0.55, 0.45),
            1.0,
        ),
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
        EMPTY_SCALAR_BRANCH,
    ],
    0.0,
    1000.0,
);

const CONSTRICTION_DRIVE_RULE: ScalarRule = ScalarRule::new(
    0.0,
    3,
    [
        scalar_branch_1(scalar_factor(WholeCellRuleSignal::Dt, 0.0, 1.0), 0.002),
        scalar_branch_2(
            scalar_factor(WholeCellRuleSignal::Dt, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::ReplicatedFraction, 0.0, 1.0),
            0.012,
        ),
        scalar_branch_3(
            scalar_factor(WholeCellRuleSignal::Dt, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::ConstrictionFlux, 0.0, 1.0),
            scalar_factor(WholeCellRuleSignal::QuantumTranslationEfficiency, 0.0, 1.0),
            0.005,
        ),
        EMPTY_SCALAR_BRANCH,
    ],
    0.0,
    10.0,
);

/// Native whole-cell simulator with a Rust-owned state and scheduler.
pub struct WholeCellSimulator {
    config: WholeCellConfig,
    backend: WholeCellBackend,
    #[cfg(target_os = "macos")]
    gpu: Option<GpuContext>,
    lattice: IntracellularLattice,
    time_ms: f32,
    step_count: u64,
    atp_mm: f32,
    amino_acids_mm: f32,
    nucleotides_mm: f32,
    membrane_precursors_mm: f32,
    adp_mm: f32,
    glucose_mm: f32,
    oxygen_mm: f32,
    ftsz: f32,
    dnaa: f32,
    active_ribosomes: f32,
    active_rnap: f32,
    genome_bp: u32,
    replicated_bp: u32,
    chromosome_separation_nm: f32,
    radius_nm: f32,
    surface_area_nm2: f32,
    volume_nm3: f32,
    division_progress: f32,
    metabolic_load: f32,
    quantum_profile: WholeCellQuantumProfile,
    chemistry_bridge: Option<WholeCellChemistryBridge>,
    chemistry_report: LocalChemistryReport,
    chemistry_site_reports: Vec<LocalChemistrySiteReport>,
    last_md_probe: Option<LocalMDProbeReport>,
    scheduled_subsystem_probes: Vec<ScheduledSubsystemProbe>,
    subsystem_states: Vec<WholeCellSubsystemState>,
    md_translation_scale: f32,
    md_membrane_scale: f32,
}

impl WholeCellSimulator {
    fn finite_scale(value: f32, fallback: f32, min_value: f32, max_value: f32) -> f32 {
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

    fn subsystem_inventory_signal(state: WholeCellSubsystemState, support: f32) -> f32 {
        let occupancy = Self::saturating_signal(state.assembly_occupancy, 0.55);
        let stability = Self::saturating_signal(state.assembly_stability, 0.55);
        let turnover = Self::saturating_signal(state.assembly_turnover, 0.30);
        let stress_penalty = (0.20 * state.byproduct_load
            + 0.18 * (1.0 - state.demand_satisfaction).max(0.0)
            + 0.14 * (1.0 - state.crowding_penalty).max(0.0))
        .clamp(0.0, 1.0);
        (0.14
            + 0.18 * state.structural_order
            + 0.16 * state.assembly_component_availability
            + 0.20 * occupancy
            + 0.16 * stability
            + 0.08 * support
            + 0.10 * state.demand_satisfaction
            + 0.06 * state.crowding_penalty
            - 0.14 * turnover
            - 0.10 * stress_penalty)
            .clamp(0.15, 1.60)
    }

    fn base_rule_context(&self, dt: f32) -> WholeCellRuleContext {
        let atp_band = self.subsystem_state(Syn3ASubsystemPreset::AtpSynthaseMembraneBand);
        let ribosome = self.subsystem_state(Syn3ASubsystemPreset::RibosomePolysomeCluster);
        let replisome = self.subsystem_state(Syn3ASubsystemPreset::ReplisomeTrack);
        let septum = self.subsystem_state(Syn3ASubsystemPreset::FtsZSeptumRing);

        let atp_support = Self::finite_scale(self.chemistry_report.atp_support, 1.0, 0.70, 1.50);
        let translation_support =
            Self::finite_scale(self.chemistry_report.translation_support, 1.0, 0.70, 1.50);
        let nucleotide_support =
            Self::finite_scale(self.chemistry_report.nucleotide_support, 1.0, 0.70, 1.50);
        let membrane_support =
            Self::finite_scale(self.chemistry_report.membrane_support, 1.0, 0.70, 1.50);
        let crowding_penalty =
            Self::finite_scale(self.chemistry_report.crowding_penalty, 1.0, 0.65, 1.0);

        let atp_band_signal = Self::subsystem_inventory_signal(atp_band, atp_support);
        let ribosome_signal = Self::subsystem_inventory_signal(ribosome, translation_support);
        let replisome_signal = Self::subsystem_inventory_signal(replisome, nucleotide_support);
        let septum_signal = Self::subsystem_inventory_signal(septum, membrane_support);

        let glucose_signal = Self::saturating_signal(self.glucose_mm, 0.45);
        let oxygen_signal = Self::saturating_signal(self.oxygen_mm, 0.35);
        let amino_signal = Self::saturating_signal(self.amino_acids_mm, 0.45);
        let nucleotide_signal = Self::saturating_signal(self.nucleotides_mm, 0.35);
        let membrane_signal = Self::saturating_signal(self.membrane_precursors_mm, 0.18);
        let energy_signal = Self::saturating_signal(self.atp_mm, 0.50);
        let replicated_fraction = self.replicated_bp as f32 / self.genome_bp.max(1) as f32;
        let division_readiness = (0.35 + 0.65 * replicated_fraction).clamp(0.35, 1.0);

        let mut ctx = WholeCellRuleContext::default();
        ctx.set(WholeCellRuleSignal::Dt, dt);
        ctx.set(WholeCellRuleSignal::AtpBandSignal, atp_band_signal);
        ctx.set(WholeCellRuleSignal::RibosomeSignal, ribosome_signal);
        ctx.set(WholeCellRuleSignal::ReplisomeSignal, replisome_signal);
        ctx.set(WholeCellRuleSignal::SeptumSignal, septum_signal);
        ctx.set(
            WholeCellRuleSignal::WeightedRibosomeReplisomeSignal,
            0.55 * ribosome_signal + 0.45 * replisome_signal,
        );
        ctx.set(
            WholeCellRuleSignal::WeightedAtpSeptumSignal,
            0.55 * atp_band_signal + 0.45 * septum_signal,
        );
        ctx.set(WholeCellRuleSignal::GlucoseSignal, glucose_signal);
        ctx.set(WholeCellRuleSignal::OxygenSignal, oxygen_signal);
        ctx.set(WholeCellRuleSignal::AminoSignal, amino_signal);
        ctx.set(WholeCellRuleSignal::NucleotideSignal, nucleotide_signal);
        ctx.set(WholeCellRuleSignal::MembraneSignal, membrane_signal);
        ctx.set(WholeCellRuleSignal::EnergySignal, energy_signal);
        ctx.set(WholeCellRuleSignal::ReplicatedFraction, replicated_fraction);
        ctx.set(
            WholeCellRuleSignal::InverseReplicatedFraction,
            (1.0 - replicated_fraction).clamp(0.0, 1.0),
        );
        ctx.set(WholeCellRuleSignal::DivisionReadiness, division_readiness);
        ctx.set(
            WholeCellRuleSignal::LocalizedSupplyScale,
            self.localized_supply_scale(),
        );
        ctx.set(WholeCellRuleSignal::CrowdingPenalty, crowding_penalty);
        ctx.set(WholeCellRuleSignal::AtpSupport, atp_support);
        ctx.set(WholeCellRuleSignal::TranslationSupport, translation_support);
        ctx.set(WholeCellRuleSignal::NucleotideSupport, nucleotide_support);
        ctx.set(WholeCellRuleSignal::MembraneSupport, membrane_support);
        ctx.set(
            WholeCellRuleSignal::AtpBandScale,
            Self::finite_scale(self.atp_band_scale(), 1.0, 0.70, 1.45),
        );
        ctx.set(
            WholeCellRuleSignal::RibosomeTranslationScale,
            Self::finite_scale(self.ribosome_translation_scale(), 1.0, 0.70, 1.45),
        );
        ctx.set(
            WholeCellRuleSignal::ReplisomeReplicationScale,
            Self::finite_scale(self.replisome_replication_scale(), 1.0, 0.70, 1.45),
        );
        ctx.set(
            WholeCellRuleSignal::ReplisomeSegregationScale,
            Self::finite_scale(self.replisome_segregation_scale(), 1.0, 0.70, 1.45),
        );
        ctx.set(
            WholeCellRuleSignal::MembraneAssemblyScale,
            Self::finite_scale(self.membrane_assembly_scale(), 1.0, 0.70, 1.45),
        );
        ctx.set(
            WholeCellRuleSignal::FtszConstrictionScale,
            Self::finite_scale(self.ftsz_constriction_scale(), 1.0, 0.70, 1.45),
        );
        ctx.set(
            WholeCellRuleSignal::MdTranslationScale,
            Self::finite_scale(self.md_translation_scale(), 1.0, 0.70, 1.45),
        );
        ctx.set(
            WholeCellRuleSignal::MdMembraneScale,
            Self::finite_scale(self.md_membrane_scale(), 1.0, 0.70, 1.45),
        );
        ctx.set(
            WholeCellRuleSignal::QuantumOxphosEfficiency,
            self.quantum_profile.oxphos_efficiency,
        );
        ctx.set(
            WholeCellRuleSignal::QuantumTranslationEfficiency,
            self.quantum_profile.translation_efficiency,
        );
        ctx.set(
            WholeCellRuleSignal::QuantumNucleotideEfficiency,
            self.quantum_profile.nucleotide_polymerization_efficiency,
        );
        ctx.set(
            WholeCellRuleSignal::QuantumMembraneEfficiency,
            self.quantum_profile.membrane_synthesis_efficiency,
        );
        ctx.set(
            WholeCellRuleSignal::QuantumSegregationEfficiency,
            self.quantum_profile.chromosome_segregation_efficiency,
        );
        ctx.set(
            WholeCellRuleSignal::EffectiveMetabolicLoad,
            self.effective_metabolic_load(),
        );
        ctx.set(
            WholeCellRuleSignal::MembranePrecursorFloor,
            self.membrane_precursors_mm.max(0.1),
        );
        ctx
    }

    fn process_rule_context(
        &self,
        dt: f32,
        inventory: WholeCellAssemblyInventory,
    ) -> WholeCellRuleContext {
        let mut ctx = self.base_rule_context(dt);
        ctx.set(
            WholeCellRuleSignal::AtpBandComplexes,
            inventory.atp_band_complexes,
        );
        ctx.set(
            WholeCellRuleSignal::RibosomeComplexes,
            inventory.ribosome_complexes,
        );
        ctx.set(WholeCellRuleSignal::RnapComplexes, inventory.rnap_complexes);
        ctx.set(
            WholeCellRuleSignal::ReplisomeComplexes,
            inventory.replisome_complexes,
        );
        ctx.set(
            WholeCellRuleSignal::MembraneComplexes,
            inventory.membrane_complexes,
        );
        ctx.set(WholeCellRuleSignal::FtszPolymer, inventory.ftsz_polymer);
        ctx.set(WholeCellRuleSignal::DnaaActivity, inventory.dnaa_activity);
        ctx
    }

    fn stage_rule_context(
        &self,
        dt: f32,
        inventory: WholeCellAssemblyInventory,
        fluxes: WholeCellProcessFluxes,
    ) -> WholeCellRuleContext {
        let mut ctx = self.process_rule_context(dt, inventory);
        ctx.set(WholeCellRuleSignal::EnergyCapacity, fluxes.energy_capacity);
        ctx.set(
            WholeCellRuleSignal::EnergyCapacityCapped16,
            fluxes.energy_capacity.min(1.6),
        );
        ctx.set(
            WholeCellRuleSignal::EnergyCapacityCapped18,
            fluxes.energy_capacity.min(1.8),
        );
        ctx.set(
            WholeCellRuleSignal::TranscriptionCapacity,
            fluxes.transcription_capacity,
        );
        ctx.set(
            WholeCellRuleSignal::TranscriptionCapacityCapped16,
            fluxes.transcription_capacity.min(1.6),
        );
        ctx.set(
            WholeCellRuleSignal::TranslationCapacity,
            fluxes.translation_capacity,
        );
        ctx.set(
            WholeCellRuleSignal::ReplicationCapacity,
            fluxes.replication_capacity,
        );
        ctx.set(
            WholeCellRuleSignal::SegregationCapacity,
            fluxes.segregation_capacity,
        );
        ctx.set(
            WholeCellRuleSignal::MembraneCapacity,
            fluxes.membrane_capacity,
        );
        ctx.set(
            WholeCellRuleSignal::ConstrictionCapacity,
            fluxes.constriction_capacity,
        );
        ctx.set(
            WholeCellRuleSignal::DnaaSignal,
            Self::saturating_signal(inventory.dnaa_activity, 48.0),
        );
        ctx.set(
            WholeCellRuleSignal::ReplisomeAssemblySignal,
            Self::saturating_signal(inventory.replisome_complexes, 28.0),
        );
        ctx.set(
            WholeCellRuleSignal::ConstrictionSignal,
            Self::saturating_signal(inventory.ftsz_polymer, 90.0),
        );
        ctx.set(
            WholeCellRuleSignal::TranscriptionDriveMix,
            0.35 * fluxes.energy_capacity.min(1.6)
                + 0.15 * ctx.get(WholeCellRuleSignal::GlucoseSignal),
        );
        ctx.set(
            WholeCellRuleSignal::TranslationDriveMix,
            0.30 * fluxes.energy_capacity.min(1.8) + 0.15 * fluxes.transcription_capacity,
        );
        ctx.set(
            WholeCellRuleSignal::BiosyntheticLoadMix,
            0.28 * fluxes.translation_capacity
                + 0.16 * fluxes.transcription_capacity
                + 0.12 * fluxes.replication_capacity
                + 0.10 * fluxes.constriction_capacity,
        );
        ctx
    }

    fn assembly_inventory(&self) -> WholeCellAssemblyInventory {
        let ctx = self.base_rule_context(0.0);
        let scalar = ctx.scalar();
        WholeCellAssemblyInventory {
            atp_band_complexes: ATP_BAND_INVENTORY_RULE.evaluate(scalar),
            ribosome_complexes: RIBOSOME_INVENTORY_RULE.evaluate(scalar),
            rnap_complexes: RNAP_INVENTORY_RULE.evaluate(scalar),
            replisome_complexes: REPLISOME_INVENTORY_RULE.evaluate(scalar),
            membrane_complexes: MEMBRANE_INVENTORY_RULE.evaluate(scalar),
            ftsz_polymer: FTSZ_POLYMER_RULE.evaluate(scalar),
            dnaa_activity: DNAA_ACTIVITY_RULE.evaluate(scalar),
        }
    }

    fn process_fluxes(&self, inventory: WholeCellAssemblyInventory) -> WholeCellProcessFluxes {
        let ctx = self.process_rule_context(0.0, inventory);
        let scalar = ctx.scalar();
        WholeCellProcessFluxes {
            energy_capacity: ENERGY_CAPACITY_RULE.evaluate(scalar),
            transcription_capacity: TRANSCRIPTION_CAPACITY_RULE.evaluate(scalar),
            translation_capacity: TRANSLATION_CAPACITY_RULE.evaluate(scalar),
            replication_capacity: REPLICATION_CAPACITY_RULE.evaluate(scalar),
            segregation_capacity: SEGREGATION_CAPACITY_RULE.evaluate(scalar),
            membrane_capacity: MEMBRANE_CAPACITY_RULE.evaluate(scalar),
            constriction_capacity: CONSTRICTION_CAPACITY_RULE.evaluate(scalar),
        }
    }

    fn refresh_surrogate_pool_diagnostics(
        &mut self,
        inventory: WholeCellAssemblyInventory,
        transcription_flux: f32,
        translation_flux: f32,
        replication_flux: f32,
        membrane_flux: f32,
        constriction_flux: f32,
    ) {
        let replisome_scale =
            Self::finite_scale(self.replisome_replication_scale(), 1.0, 0.70, 1.45);
        let ftsz_translation_scale =
            Self::finite_scale(self.ftsz_translation_scale(), 1.0, 0.70, 1.45);
        self.active_rnap = (0.72 * self.active_rnap
            + 0.28 * (inventory.rnap_complexes + 8.0 * transcription_flux))
            .clamp(8.0, 256.0);
        self.active_ribosomes = (0.70 * self.active_ribosomes
            + 0.30 * (inventory.ribosome_complexes + 6.0 * translation_flux))
            .clamp(12.0, 320.0);
        self.dnaa = (0.72 * self.dnaa
            + 0.28 * (inventory.dnaa_activity + 4.0 * replication_flux * replisome_scale))
            .clamp(8.0, 256.0);
        self.ftsz = (0.70 * self.ftsz
            + 0.30
                * (inventory.ftsz_polymer
                    + 8.0 * membrane_flux
                    + 6.0 * constriction_flux
                    + 4.0 * translation_flux * ftsz_translation_scale))
            .clamp(12.0, 384.0);
    }

    fn initialize_surrogate_pool_diagnostics(&mut self) {
        let inventory = self.assembly_inventory();
        self.active_rnap = inventory.rnap_complexes.clamp(8.0, 256.0);
        self.active_ribosomes = inventory.ribosome_complexes.clamp(12.0, 320.0);
        self.dnaa = inventory.dnaa_activity.clamp(8.0, 256.0);
        self.ftsz = inventory.ftsz_polymer.clamp(12.0, 384.0);
    }

    /// Create a simulator with JCVI-syn3A-like defaults.
    pub fn new(config: WholeCellConfig) -> Self {
        let backend = if config.use_gpu && gpu::has_gpu() {
            WholeCellBackend::Metal
        } else {
            WholeCellBackend::Cpu
        };

        #[cfg(target_os = "macos")]
        let gpu = if backend == WholeCellBackend::Metal {
            GpuContext::new().ok()
        } else {
            None
        };

        let backend = {
            #[cfg(target_os = "macos")]
            {
                if backend == WholeCellBackend::Metal && gpu.is_some() {
                    WholeCellBackend::Metal
                } else {
                    WholeCellBackend::Cpu
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                WholeCellBackend::Cpu
            }
        };

        let mut lattice = IntracellularLattice::new(
            config.x_dim,
            config.y_dim,
            config.z_dim,
            config.voxel_size_nm,
        );
        lattice.fill_species(IntracellularSpecies::ATP, 1.20);
        lattice.fill_species(IntracellularSpecies::AminoAcids, 0.95);
        lattice.fill_species(IntracellularSpecies::Nucleotides, 0.80);
        lattice.fill_species(IntracellularSpecies::MembranePrecursors, 0.35);

        let radius_nm = 200.0;
        let surface_area_nm2 = 4.0 * PI * radius_nm * radius_nm;
        let volume_nm3 = 4.0 / 3.0 * PI * radius_nm.powi(3);

        let mut simulator = Self {
            config,
            backend,
            #[cfg(target_os = "macos")]
            gpu,
            lattice,
            time_ms: 0.0,
            step_count: 0,
            atp_mm: 1.20,
            amino_acids_mm: 0.95,
            nucleotides_mm: 0.80,
            membrane_precursors_mm: 0.35,
            adp_mm: 0.30,
            glucose_mm: 1.0,
            oxygen_mm: 0.85,
            ftsz: 0.0,
            dnaa: 0.0,
            active_ribosomes: 0.0,
            active_rnap: 0.0,
            genome_bp: 543_000,
            replicated_bp: 0,
            chromosome_separation_nm: 40.0,
            radius_nm,
            surface_area_nm2,
            volume_nm3,
            division_progress: 0.0,
            metabolic_load: 1.0,
            quantum_profile: WholeCellQuantumProfile::default(),
            chemistry_bridge: None,
            chemistry_report: LocalChemistryReport::default(),
            chemistry_site_reports: Vec::new(),
            last_md_probe: None,
            scheduled_subsystem_probes: Vec::new(),
            subsystem_states: Syn3ASubsystemPreset::all()
                .iter()
                .copied()
                .map(WholeCellSubsystemState::new)
                .collect(),
            md_translation_scale: 1.0,
            md_membrane_scale: 1.0,
        };
        simulator.sync_from_lattice();
        simulator.initialize_surrogate_pool_diagnostics();
        simulator
    }

    /// Step the simulator by one configured time quantum.
    pub fn step(&mut self) {
        let dt = self.config.dt_ms;
        self.update_local_chemistry(dt);
        self.rdme_stage(dt);
        if self.step_count % self.config.cme_interval == 0 {
            self.cme_stage(dt);
        }
        if self.step_count % self.config.ode_interval == 0 {
            self.ode_stage(dt);
        }
        if self.step_count % self.config.bd_interval == 0 {
            self.bd_stage(dt);
        }
        if self.step_count % self.config.geometry_interval == 0 {
            self.geometry_stage(dt);
        }
        self.time_ms += dt;
        self.step_count += 1;
    }

    /// Run the simulator for a fixed number of steps.
    pub fn run(&mut self, steps: u64) {
        for _ in 0..steps {
            self.step();
        }
    }

    /// Current backend name.
    pub fn backend(&self) -> WholeCellBackend {
        self.backend
    }

    /// Current simulation time in milliseconds.
    pub fn time_ms(&self) -> f32 {
        self.time_ms
    }

    /// Number of integration steps completed.
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Expose the current voxel dimensions.
    pub fn lattice_shape(&self) -> (usize, usize, usize) {
        (self.lattice.x_dim, self.lattice.y_dim, self.lattice.z_dim)
    }

    /// Update metabolic load. Values >1.0 increase sink terms.
    pub fn set_metabolic_load(&mut self, load: f32) {
        self.metabolic_load = load.max(0.1);
    }

    /// Update the quantum correction profile used by the coarse stage models.
    pub fn set_quantum_profile(&mut self, profile: WholeCellQuantumProfile) {
        self.quantum_profile = profile.normalized();
    }

    /// Current quantum correction profile.
    pub fn quantum_profile(&self) -> WholeCellQuantumProfile {
        self.quantum_profile
    }

    /// Enable the local chemistry lattice submodel.
    pub fn enable_local_chemistry(
        &mut self,
        x_dim: usize,
        y_dim: usize,
        z_dim: usize,
        voxel_size_au: f32,
        use_gpu: bool,
    ) {
        self.chemistry_bridge = Some(WholeCellChemistryBridge::new(
            x_dim,
            y_dim,
            z_dim,
            voxel_size_au,
            use_gpu,
        ));
    }

    /// Disable the local chemistry submodel.
    pub fn disable_local_chemistry(&mut self) {
        self.chemistry_bridge = None;
        self.chemistry_report = LocalChemistryReport::default();
        self.chemistry_site_reports.clear();
        self.last_md_probe = None;
        self.scheduled_subsystem_probes.clear();
        self.md_translation_scale = 1.0;
        self.md_membrane_scale = 1.0;
    }

    /// Latest local chemistry report, if enabled.
    pub fn local_chemistry_report(&self) -> Option<LocalChemistryReport> {
        self.chemistry_bridge
            .as_ref()
            .map(|_| self.chemistry_report)
    }

    /// Latest per-subsystem local chemistry reports, if enabled.
    pub fn local_chemistry_sites(&self) -> Vec<LocalChemistrySiteReport> {
        if self.chemistry_bridge.is_some() {
            self.chemistry_site_reports.clone()
        } else {
            Vec::new()
        }
    }

    /// Latest localized MD probe report.
    pub fn last_md_probe(&self) -> Option<LocalMDProbeReport> {
        self.last_md_probe
    }

    /// Current persistent coupling state for each Syn3A subsystem preset.
    pub fn subsystem_states(&self) -> Vec<WholeCellSubsystemState> {
        self.subsystem_states.clone()
    }

    /// Run a localized MD probe through the optional chemistry bridge.
    pub fn run_local_md_probe(
        &mut self,
        request: LocalMDProbeRequest,
    ) -> Option<LocalMDProbeReport> {
        let report = {
            let bridge = self.chemistry_bridge.as_mut()?;
            bridge.run_md_probe(request)
        };
        if let Some(preset) = Self::preset_for_site(report.site) {
            self.apply_probe_to_subsystem(preset, report);
        }
        self.last_md_probe = Some(report);
        self.md_translation_scale = report.recommended_translation_scale;
        self.md_membrane_scale = report.recommended_membrane_scale;
        Some(report)
    }

    /// Run a named Syn3A subsystem probe using the default request.
    pub fn run_syn3a_subsystem_probe(
        &mut self,
        preset: Syn3ASubsystemPreset,
    ) -> Option<LocalMDProbeReport> {
        self.run_local_md_probe(preset.default_probe_request())
    }

    /// Schedule a Syn3A subsystem probe to run periodically.
    pub fn schedule_syn3a_subsystem_probe(
        &mut self,
        preset: Syn3ASubsystemPreset,
        interval_steps: u64,
    ) {
        let interval_steps = interval_steps.max(1);
        if let Some(existing) = self
            .scheduled_subsystem_probes
            .iter_mut()
            .find(|probe| probe.preset == preset)
        {
            existing.interval_steps = interval_steps;
            return;
        }
        self.scheduled_subsystem_probes
            .push(ScheduledSubsystemProbe {
                preset,
                interval_steps,
            });
    }

    /// Clear all scheduled Syn3A subsystem probes.
    pub fn clear_syn3a_subsystem_probes(&mut self) {
        self.scheduled_subsystem_probes.clear();
        for state in &mut self.subsystem_states {
            *state = WholeCellSubsystemState::new(state.preset);
            state.apply_chemistry_report(self.chemistry_report);
        }
        self.md_translation_scale = 1.0;
        self.md_membrane_scale = 1.0;
    }

    /// Return a copy of the scheduled subsystem probes.
    pub fn scheduled_syn3a_subsystem_probes(&self) -> Vec<ScheduledSubsystemProbe> {
        self.scheduled_subsystem_probes.clone()
    }

    /// Enable the default set of Syn3A subsystem probes.
    pub fn enable_default_syn3a_subsystems(&mut self) {
        if self.chemistry_bridge.is_none() {
            self.enable_local_chemistry(12, 12, 6, 0.5, true);
        }
        self.clear_syn3a_subsystem_probes();
        for preset in Syn3ASubsystemPreset::all() {
            self.schedule_syn3a_subsystem_probe(*preset, preset.default_interval_steps());
        }
    }

    /// Mean ATP concentration across the cell.
    pub fn atp_mm(&self) -> f32 {
        self.atp_mm
    }

    /// FtsZ pool used for division ring assembly.
    pub fn ftsz(&self) -> f32 {
        self.ftsz
    }

    /// Current chromosome replication progress in base pairs.
    pub fn replicated_bp(&self) -> u32 {
        self.replicated_bp
    }

    /// Current division progress (0-1).
    pub fn division_progress(&self) -> f32 {
        self.division_progress
    }

    /// Return a copied ATP lattice channel.
    pub fn atp_lattice(&self) -> Vec<f32> {
        self.lattice.clone_species(IntracellularSpecies::ATP)
    }

    /// Seed a hotspot into a species channel.
    pub fn add_hotspot(
        &mut self,
        species: IntracellularSpecies,
        x: usize,
        y: usize,
        z: usize,
        delta: f32,
    ) {
        self.lattice.add_hotspot(species, x, y, z, delta);
        self.sync_from_lattice();
    }

    /// Snapshot the coarse state for diagnostics or bindings.
    pub fn snapshot(&self) -> WholeCellSnapshot {
        WholeCellSnapshot {
            backend: self.backend,
            time_ms: self.time_ms,
            step_count: self.step_count,
            atp_mm: self.atp_mm,
            amino_acids_mm: self.amino_acids_mm,
            nucleotides_mm: self.nucleotides_mm,
            membrane_precursors_mm: self.membrane_precursors_mm,
            adp_mm: self.adp_mm,
            glucose_mm: self.glucose_mm,
            oxygen_mm: self.oxygen_mm,
            ftsz: self.ftsz,
            dnaa: self.dnaa,
            active_ribosomes: self.active_ribosomes,
            active_rnap: self.active_rnap,
            genome_bp: self.genome_bp,
            replicated_bp: self.replicated_bp,
            chromosome_separation_nm: self.chromosome_separation_nm,
            radius_nm: self.radius_nm,
            surface_area_nm2: self.surface_area_nm2,
            volume_nm3: self.volume_nm3,
            division_progress: self.division_progress,
            quantum_profile: self.quantum_profile,
            local_chemistry: self.local_chemistry_report(),
            local_chemistry_sites: self.local_chemistry_sites(),
            local_md_probe: self.last_md_probe,
            subsystem_states: self.subsystem_states(),
        }
    }

    fn update_local_chemistry(&mut self, dt: f32) {
        let snapshot = self.snapshot();
        let scheduled_probes = self.scheduled_subsystem_probes.clone();
        let Some((chemistry_report, chemistry_site_reports, last_md_report, due_reports)) = ({
            let Some(ref mut bridge) = self.chemistry_bridge else {
                return;
            };
            let chemistry_report = bridge.step_with_snapshot((dt * 2.0).max(0.1), Some(&snapshot));
            let chemistry_site_reports = bridge.site_reports();
            let last_md_report = bridge.last_md_report();
            let mut due_reports = Vec::new();
            for scheduled in &scheduled_probes {
                if self.step_count % scheduled.interval_steps == 0 {
                    let report = bridge.run_md_probe(scheduled.preset.default_probe_request());
                    due_reports.push((scheduled.preset, report));
                }
            }
            Some((
                chemistry_report,
                chemistry_site_reports,
                last_md_report,
                due_reports,
            ))
        }) else {
            return;
        };

        self.chemistry_report = chemistry_report;
        self.chemistry_site_reports = chemistry_site_reports;
        self.refresh_subsystem_chemistry_state();
        if scheduled_probes.is_empty() {
            self.last_md_probe = last_md_report;
            return;
        }

        if due_reports.is_empty() {
            if let Some(report) = last_md_report {
                self.last_md_probe = Some(report);
            }
        } else {
            for (preset, report) in &due_reports {
                self.apply_probe_to_subsystem(*preset, *report);
            }
            let count = due_reports.len() as f32;
            self.md_translation_scale = due_reports
                .iter()
                .map(|(_, report)| report.recommended_translation_scale)
                .sum::<f32>()
                / count;
            self.md_membrane_scale = due_reports
                .iter()
                .map(|(_, report)| report.recommended_membrane_scale)
                .sum::<f32>()
                / count;
            self.last_md_probe = due_reports.last().map(|(_, report)| *report);
        }
    }

    fn md_translation_scale(&self) -> f32 {
        self.md_translation_scale
    }

    fn md_membrane_scale(&self) -> f32 {
        self.md_membrane_scale
    }

    fn preset_for_site(
        site: crate::whole_cell_submodels::WholeCellChemistrySite,
    ) -> Option<Syn3ASubsystemPreset> {
        match site {
            crate::whole_cell_submodels::WholeCellChemistrySite::AtpSynthaseBand => {
                Some(Syn3ASubsystemPreset::AtpSynthaseMembraneBand)
            }
            crate::whole_cell_submodels::WholeCellChemistrySite::RibosomeCluster => {
                Some(Syn3ASubsystemPreset::RibosomePolysomeCluster)
            }
            crate::whole_cell_submodels::WholeCellChemistrySite::ChromosomeTrack => {
                Some(Syn3ASubsystemPreset::ReplisomeTrack)
            }
            crate::whole_cell_submodels::WholeCellChemistrySite::SeptumRing => {
                Some(Syn3ASubsystemPreset::FtsZSeptumRing)
            }
            crate::whole_cell_submodels::WholeCellChemistrySite::Cytosol => None,
        }
    }

    fn subsystem_state(&self, preset: Syn3ASubsystemPreset) -> WholeCellSubsystemState {
        self.subsystem_states
            .iter()
            .copied()
            .find(|state| state.preset == preset)
            .unwrap_or_else(|| WholeCellSubsystemState::new(preset))
    }

    fn subsystem_state_mut(
        &mut self,
        preset: Syn3ASubsystemPreset,
    ) -> Option<&mut WholeCellSubsystemState> {
        self.subsystem_states
            .iter_mut()
            .find(|state| state.preset == preset)
    }

    fn refresh_subsystem_chemistry_state(&mut self) {
        for state in &mut self.subsystem_states {
            if let Some(report) = self
                .chemistry_site_reports
                .iter()
                .find(|report| report.preset == state.preset)
                .copied()
            {
                state.apply_site_report(report);
            } else {
                state.apply_chemistry_report(self.chemistry_report);
            }
        }
    }

    fn apply_probe_to_subsystem(
        &mut self,
        preset: Syn3ASubsystemPreset,
        report: LocalMDProbeReport,
    ) {
        let step_count = self.step_count;
        if let Some(state) = self.subsystem_state_mut(preset) {
            state.apply_probe_report(report, step_count);
        }
    }

    fn atp_band_scale(&self) -> f32 {
        self.subsystem_state(Syn3ASubsystemPreset::AtpSynthaseMembraneBand)
            .atp_scale
    }

    fn ribosome_translation_scale(&self) -> f32 {
        self.subsystem_state(Syn3ASubsystemPreset::RibosomePolysomeCluster)
            .translation_scale
    }

    fn replisome_replication_scale(&self) -> f32 {
        self.subsystem_state(Syn3ASubsystemPreset::ReplisomeTrack)
            .replication_scale
    }

    fn replisome_segregation_scale(&self) -> f32 {
        self.subsystem_state(Syn3ASubsystemPreset::ReplisomeTrack)
            .segregation_scale
    }

    fn ftsz_translation_scale(&self) -> f32 {
        self.subsystem_state(Syn3ASubsystemPreset::FtsZSeptumRing)
            .translation_scale
    }

    fn ftsz_constriction_scale(&self) -> f32 {
        self.subsystem_state(Syn3ASubsystemPreset::FtsZSeptumRing)
            .constriction_scale
    }

    fn membrane_assembly_scale(&self) -> f32 {
        let atp_band = self.subsystem_state(Syn3ASubsystemPreset::AtpSynthaseMembraneBand);
        let septum = self.subsystem_state(Syn3ASubsystemPreset::FtsZSeptumRing);
        (0.55 * atp_band.membrane_scale + 0.45 * septum.membrane_scale).clamp(0.70, 1.45)
    }

    fn localized_supply_scale(&self) -> f32 {
        if self.chemistry_site_reports.is_empty() {
            return 1.0;
        }
        let mean_satisfaction = self
            .chemistry_site_reports
            .iter()
            .map(|report| report.demand_satisfaction)
            .sum::<f32>()
            / self.chemistry_site_reports.len() as f32;
        Self::finite_scale(mean_satisfaction, 1.0, 0.55, 1.0)
    }

    fn localized_resource_pressure(&self) -> f32 {
        if self.chemistry_site_reports.is_empty() {
            return 0.0;
        }
        self.chemistry_site_reports
            .iter()
            .map(|report| {
                0.45 * report.substrate_draw
                    + 0.55 * report.energy_draw
                    + 0.50 * report.biosynthetic_draw
                    + 0.60 * report.byproduct_load
                    + (1.0 - report.demand_satisfaction).max(0.0) * 1.2
            })
            .sum::<f32>()
            / self.chemistry_site_reports.len() as f32
    }

    fn effective_metabolic_load(&self) -> f32 {
        let supply_scale = self.localized_supply_scale();
        let pressure = self.localized_resource_pressure();
        let local_multiplier =
            (1.0 + pressure * 0.16 + (1.0 - supply_scale).max(0.0) * 0.35).clamp(1.0, 2.2);
        self.metabolic_load.max(0.1) * local_multiplier
    }

    fn rdme_stage(&mut self, dt: f32) {
        let effective_metabolic_load = self.effective_metabolic_load();
        match self.backend {
            WholeCellBackend::Metal => {
                #[cfg(target_os = "macos")]
                {
                    if let Some(ref gpu) = self.gpu {
                        dispatch_whole_cell_rdme(
                            gpu,
                            &mut self.lattice,
                            dt,
                            effective_metabolic_load,
                        );
                    } else {
                        cpu_whole_cell_rdme(&mut self.lattice, dt, effective_metabolic_load);
                    }
                }
                #[cfg(not(target_os = "macos"))]
                {
                    cpu_whole_cell_rdme(&mut self.lattice, dt, effective_metabolic_load);
                }
            }
            WholeCellBackend::Cpu => {
                cpu_whole_cell_rdme(&mut self.lattice, dt, effective_metabolic_load)
            }
        }
        self.sync_from_lattice();
    }

    fn cme_stage(&mut self, dt: f32) {
        let inventory = self.assembly_inventory();
        let fluxes = self.process_fluxes(inventory);
        let ctx = self.stage_rule_context(dt, inventory, fluxes);
        let scalar = ctx.scalar();
        let transcription_flux = TRANSCRIPTION_FLUX_RULE.evaluate(scalar);
        let translation_flux = TRANSLATION_FLUX_RULE.evaluate(scalar);

        self.lattice
            .apply_uniform_delta(IntracellularSpecies::ATP, -0.00030 * translation_flux);
        self.lattice.apply_uniform_delta(
            IntracellularSpecies::AminoAcids,
            -0.00022 * translation_flux,
        );
        self.lattice.apply_uniform_delta(
            IntracellularSpecies::Nucleotides,
            -0.00016 * transcription_flux,
        );
        self.sync_from_lattice();
        let inventory = self.assembly_inventory();
        self.refresh_surrogate_pool_diagnostics(
            inventory,
            transcription_flux,
            translation_flux,
            0.0,
            0.0,
            0.0,
        );
    }

    fn ode_stage(&mut self, dt: f32) {
        let inventory = self.assembly_inventory();
        let fluxes = self.process_fluxes(inventory);
        let ctx = self.stage_rule_context(dt, inventory, fluxes);
        let scalar = ctx.scalar();
        let effective_metabolic_load = ctx.get(WholeCellRuleSignal::EffectiveMetabolicLoad);
        let energy_gain = ENERGY_GAIN_RULE.evaluate(scalar);
        let energy_cost = ENERGY_COST_RULE.evaluate(scalar);
        let nucleotide_recharge = NUCLEOTIDE_RECHARGE_RULE.evaluate(scalar);
        let membrane_flux = MEMBRANE_FLUX_RULE.evaluate(scalar);

        self.adp_mm = (self.adp_mm + energy_cost - 0.65 * energy_gain).clamp(0.05, 4.0);
        self.glucose_mm = (self.glucose_mm + 0.0020 * dt - 0.0012 * effective_metabolic_load * dt)
            .clamp(0.2, 3.0);
        self.oxygen_mm =
            (self.oxygen_mm + 0.0018 * dt - 0.0010 * effective_metabolic_load * dt).clamp(0.2, 2.5);

        self.lattice
            .apply_uniform_delta(IntracellularSpecies::ATP, energy_gain - energy_cost);
        self.lattice
            .apply_uniform_delta(IntracellularSpecies::Nucleotides, nucleotide_recharge);
        self.lattice
            .apply_uniform_delta(IntracellularSpecies::MembranePrecursors, membrane_flux);
        self.sync_from_lattice();
        let inventory = self.assembly_inventory();
        self.refresh_surrogate_pool_diagnostics(inventory, 0.0, 0.0, 0.0, membrane_flux, 0.0);
    }

    fn bd_stage(&mut self, dt: f32) {
        let inventory = self.assembly_inventory();
        let fluxes = self.process_fluxes(inventory);
        let ctx = self.stage_rule_context(dt, inventory, fluxes);
        let scalar = ctx.scalar();
        let replication_drive = REPLICATION_DRIVE_RULE.evaluate(scalar);
        let replication_flux = replication_drive / 18.0;
        let next_bp = self.replicated_bp as f32 + replication_drive;
        let next_bp = if next_bp.is_finite() {
            next_bp.min(self.genome_bp as f32)
        } else {
            self.replicated_bp as f32
        };
        self.replicated_bp = next_bp as u32;

        let mut segregation_ctx = ctx;
        let replicated_fraction = self.replicated_bp as f32 / self.genome_bp.max(1) as f32;
        segregation_ctx.set(WholeCellRuleSignal::ReplicatedFraction, replicated_fraction);
        segregation_ctx.set(
            WholeCellRuleSignal::InverseReplicatedFraction,
            (1.0 - replicated_fraction).clamp(0.0, 1.0),
        );
        self.chromosome_separation_nm = (self.chromosome_separation_nm
            + SEGREGATION_STEP_RULE.evaluate(segregation_ctx.scalar()))
        .min(self.radius_nm * 1.8);
        self.refresh_surrogate_pool_diagnostics(inventory, 0.0, 0.0, replication_flux, 0.0, 0.0);
    }

    fn geometry_stage(&mut self, dt: f32) {
        let inventory = self.assembly_inventory();
        let fluxes = self.process_fluxes(inventory);
        let mut ctx = self.stage_rule_context(dt, inventory, fluxes);
        let membrane_growth_nm2 = MEMBRANE_GROWTH_RULE.evaluate(ctx.scalar());

        self.surface_area_nm2 += membrane_growth_nm2;
        self.radius_nm = (self.surface_area_nm2 / (4.0 * PI)).sqrt();
        self.volume_nm3 = 4.0 / 3.0 * PI * self.radius_nm.powi(3);

        let constriction_flux = CONSTRICTION_FLUX_RULE.evaluate(ctx.scalar());
        ctx.set(WholeCellRuleSignal::ConstrictionFlux, constriction_flux);
        let constriction_drive = CONSTRICTION_DRIVE_RULE.evaluate(ctx.scalar());
        self.division_progress = (self.division_progress + constriction_drive).min(0.99);

        self.lattice.apply_uniform_delta(
            IntracellularSpecies::MembranePrecursors,
            -0.00020 * membrane_growth_nm2,
        );
        self.sync_from_lattice();
        let inventory = self.assembly_inventory();
        self.refresh_surrogate_pool_diagnostics(
            inventory,
            0.0,
            0.0,
            0.0,
            membrane_growth_nm2 * 0.001,
            constriction_flux,
        );
    }

    fn sync_from_lattice(&mut self) {
        self.atp_mm = self.lattice.mean_species(IntracellularSpecies::ATP);
        self.amino_acids_mm = self.lattice.mean_species(IntracellularSpecies::AminoAcids);
        self.nucleotides_mm = self.lattice.mean_species(IntracellularSpecies::Nucleotides);
        self.membrane_precursors_mm = self
            .lattice
            .mean_species(IntracellularSpecies::MembranePrecursors);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::whole_cell_submodels::{
        LocalChemistrySiteReport, LocalMDProbeRequest, Syn3ASubsystemPreset, WholeCellChemistrySite,
    };

    #[test]
    fn test_cpu_whole_cell_progresses_state() {
        let mut sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            dt_ms: 0.5,
            ..WholeCellConfig::default()
        });
        let start = sim.snapshot();

        sim.run(40);

        let end = sim.snapshot();
        assert_eq!(sim.backend(), WholeCellBackend::Cpu);
        assert!(end.time_ms > start.time_ms);
        assert!(end.ftsz > start.ftsz);
        assert!(end.replicated_bp > start.replicated_bp);
        assert!(end.surface_area_nm2 > start.surface_area_nm2);
        assert!(end.atp_mm > 0.0);
    }

    #[test]
    fn test_atp_hotspot_diffuses_on_cpu() {
        let mut sim = WholeCellSimulator::new(WholeCellConfig {
            x_dim: 8,
            y_dim: 8,
            z_dim: 4,
            use_gpu: false,
            dt_ms: 0.25,
            ..WholeCellConfig::default()
        });
        let center = (4, 4, 2);
        let neighbor_idx = center.2 * 8 * 8 + center.1 * 8 + (center.0 + 1);
        let center_idx = center.2 * 8 * 8 + center.1 * 8 + center.0;

        sim.add_hotspot(IntracellularSpecies::ATP, center.0, center.1, center.2, 4.0);
        let before = sim.atp_lattice();

        sim.step();

        let after = sim.atp_lattice();
        assert!(after[center_idx] < before[center_idx]);
        assert!(after[neighbor_idx] > before[neighbor_idx]);
    }

    #[test]
    fn test_initial_lattice_has_no_seeded_hotspots() {
        let sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            ..WholeCellConfig::default()
        });
        let atp = sim.atp_lattice();
        let first = atp.first().copied().expect("atp lattice");
        assert!(atp.iter().all(|value| (*value - first).abs() < 1.0e-6));
    }

    #[test]
    fn test_quantum_profile_accelerates_growth() {
        let config = WholeCellConfig {
            use_gpu: false,
            dt_ms: 0.5,
            ..WholeCellConfig::default()
        };
        let mut baseline = WholeCellSimulator::new(config.clone());
        let mut accelerated = WholeCellSimulator::new(config);
        accelerated.set_quantum_profile(WholeCellQuantumProfile {
            oxphos_efficiency: 1.60,
            translation_efficiency: 1.45,
            nucleotide_polymerization_efficiency: 1.50,
            membrane_synthesis_efficiency: 1.35,
            chromosome_segregation_efficiency: 1.30,
        });

        baseline.run(120);
        accelerated.run(120);

        let baseline_snapshot = baseline.snapshot();
        let accelerated_snapshot = accelerated.snapshot();

        assert!(accelerated_snapshot.atp_mm >= baseline_snapshot.atp_mm);
        assert!(accelerated_snapshot.ftsz > baseline_snapshot.ftsz);
        assert!(accelerated_snapshot.division_progress > baseline_snapshot.division_progress);
    }

    #[test]
    fn test_surrogate_pools_are_diagnostics_not_stage_drivers() {
        let config = WholeCellConfig {
            use_gpu: false,
            dt_ms: 0.25,
            ..WholeCellConfig::default()
        };
        let mut baseline = WholeCellSimulator::new(config.clone());
        let mut perturbed = WholeCellSimulator::new(config);

        perturbed.active_rnap = 256.0;
        perturbed.active_ribosomes = 320.0;
        perturbed.dnaa = 256.0;
        perturbed.ftsz = 384.0;

        baseline.run(16);
        perturbed.run(16);

        let baseline_snapshot = baseline.snapshot();
        let perturbed_snapshot = perturbed.snapshot();

        assert_eq!(
            perturbed_snapshot.replicated_bp,
            baseline_snapshot.replicated_bp
        );
        assert!(
            (perturbed_snapshot.division_progress - baseline_snapshot.division_progress).abs()
                < 1.0e-6
        );
        assert!(
            (perturbed_snapshot.surface_area_nm2 - baseline_snapshot.surface_area_nm2).abs()
                < 1.0e-4
        );
        assert!(perturbed_snapshot.active_rnap < 256.0);
        assert!(perturbed_snapshot.active_ribosomes < 320.0);
        assert!(perturbed_snapshot.dnaa < 256.0);
        assert!(perturbed_snapshot.ftsz < 384.0);
    }

    #[test]
    fn test_local_chemistry_bridge_updates_report_and_md_probe() {
        let mut sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            dt_ms: 0.25,
            ..WholeCellConfig::default()
        });
        sim.enable_local_chemistry(10, 10, 6, 0.5, false);

        sim.run(8);
        let chemistry = sim
            .local_chemistry_report()
            .expect("local chemistry report");
        assert!(chemistry.atp_support > 0.0);
        assert!(chemistry.translation_support > 0.0);
        assert!(chemistry.crowding_penalty > 0.0);

        let probe = sim
            .run_local_md_probe(LocalMDProbeRequest {
                site: WholeCellChemistrySite::RibosomeCluster,
                n_atoms: 16,
                steps: 8,
                dt_ps: 0.001,
                box_size_angstrom: 14.0,
                temperature_k: 310.0,
            })
            .expect("md probe");
        assert!(probe.structural_order > 0.0);
        assert!(probe.crowding_penalty > 0.0);
        assert!(sim.last_md_probe().is_some());
    }

    #[test]
    fn test_default_syn3a_subsystems_schedule_and_run() {
        let mut sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            dt_ms: 0.25,
            ..WholeCellConfig::default()
        });
        sim.enable_default_syn3a_subsystems();

        let scheduled = sim.scheduled_syn3a_subsystem_probes();
        assert_eq!(scheduled.len(), Syn3ASubsystemPreset::all().len());

        sim.run(12);

        assert!(sim.local_chemistry_report().is_some());
        assert!(sim.last_md_probe().is_some());
        assert!(sim.md_translation_scale() > 0.0);
        assert!(sim.md_membrane_scale() > 0.0);
    }

    #[test]
    fn test_subsystem_states_capture_probe_couplings() {
        let mut sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            dt_ms: 0.25,
            ..WholeCellConfig::default()
        });
        sim.enable_default_syn3a_subsystems();
        sim.run(16);

        let states = sim.subsystem_states();
        assert_eq!(states.len(), Syn3ASubsystemPreset::all().len());

        let atp_band = states
            .iter()
            .find(|state| state.preset == Syn3ASubsystemPreset::AtpSynthaseMembraneBand)
            .expect("ATP synthase state");
        assert!(atp_band.atp_scale > 0.0);
        assert!(atp_band.membrane_scale > 0.0);
        assert!(atp_band.last_probe_step.is_some());

        let replisome = states
            .iter()
            .find(|state| state.preset == Syn3ASubsystemPreset::ReplisomeTrack)
            .expect("replisome state");
        assert!(replisome.replication_scale > 0.0);
        assert!(replisome.segregation_scale > 0.0);
        assert!(replisome.last_probe_step.is_some());
    }

    #[test]
    fn test_local_chemistry_sites_are_exposed_and_site_resolved() {
        let mut sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            dt_ms: 0.25,
            ..WholeCellConfig::default()
        });
        sim.enable_default_syn3a_subsystems();
        sim.run(8);

        let site_reports = sim.local_chemistry_sites();
        assert_eq!(site_reports.len(), Syn3ASubsystemPreset::all().len());

        let atp_band = site_reports
            .iter()
            .find(|report| report.preset == Syn3ASubsystemPreset::AtpSynthaseMembraneBand)
            .expect("ATP site report");
        let replisome = site_reports
            .iter()
            .find(|report| report.preset == Syn3ASubsystemPreset::ReplisomeTrack)
            .expect("replisome site report");
        let ribosome = site_reports
            .iter()
            .find(|report| report.preset == Syn3ASubsystemPreset::RibosomePolysomeCluster)
            .expect("ribosome site report");

        assert!(atp_band.patch_radius > 0);
        assert!(atp_band.localization_score != 0.0);
        assert!(atp_band.site_z < sim.config.z_dim);
        assert!(replisome.nucleotide_support > 0.0);
        assert!(ribosome.substrate_draw > 0.0);
        assert!(replisome.biosynthetic_draw > 0.0);
        assert!(atp_band.demand_satisfaction > 0.0);
        assert!(atp_band.assembly_occupancy > 0.0);
        assert!(ribosome.assembly_stability > 0.0);
        let unique_sites = site_reports
            .iter()
            .map(|report| (report.site_x, report.site_y, report.site_z))
            .collect::<std::collections::HashSet<_>>();
        assert!(unique_sites.len() > 1);
        assert!(
            atp_band.mean_oxygen != replisome.mean_oxygen
                || atp_band.mean_atp_flux != replisome.mean_atp_flux
        );
    }

    #[test]
    fn test_localized_resource_pressure_increases_effective_metabolic_load() {
        let mut sim = WholeCellSimulator::new(WholeCellConfig {
            use_gpu: false,
            dt_ms: 0.25,
            ..WholeCellConfig::default()
        });
        sim.chemistry_site_reports = vec![LocalChemistrySiteReport {
            preset: Syn3ASubsystemPreset::RibosomePolysomeCluster,
            site: WholeCellChemistrySite::RibosomeCluster,
            patch_radius: 2,
            site_x: 4,
            site_y: 4,
            site_z: 2,
            localization_score: 0.92,
            atp_support: 0.95,
            translation_support: 0.90,
            nucleotide_support: 0.92,
            membrane_support: 0.94,
            crowding_penalty: 0.88,
            mean_glucose: 0.10,
            mean_oxygen: 0.08,
            mean_atp_flux: 0.06,
            mean_carbon_dioxide: 0.14,
            assembly_component_availability: 0.76,
            assembly_occupancy: 0.72,
            assembly_stability: 0.70,
            assembly_turnover: 0.38,
            substrate_draw: 0.60,
            energy_draw: 0.55,
            biosynthetic_draw: 0.24,
            byproduct_load: 0.42,
            demand_satisfaction: 0.46,
        }];

        assert!(sim.effective_metabolic_load() > sim.metabolic_load);
        assert!(sim.localized_supply_scale() < 1.0);
        assert!(sim.localized_resource_pressure() > 0.0);
    }

    #[test]
    fn test_replisome_probe_accelerates_replication() {
        let config = WholeCellConfig {
            use_gpu: false,
            dt_ms: 0.25,
            ..WholeCellConfig::default()
        };
        let mut baseline = WholeCellSimulator::new(config.clone());
        baseline.enable_local_chemistry(12, 12, 6, 0.5, false);

        let mut targeted = WholeCellSimulator::new(config);
        targeted.enable_local_chemistry(12, 12, 6, 0.5, false);
        targeted.schedule_syn3a_subsystem_probe(Syn3ASubsystemPreset::ReplisomeTrack, 1);

        baseline.run(12);
        targeted.run(12);

        let baseline_snapshot = baseline.snapshot();
        let targeted_snapshot = targeted.snapshot();

        assert!(
            targeted_snapshot.replicated_bp > baseline_snapshot.replicated_bp,
            "replication baseline={} targeted={}",
            baseline_snapshot.replicated_bp,
            targeted_snapshot.replicated_bp
        );
        assert!(
            targeted_snapshot.chromosome_separation_nm > baseline_snapshot.chromosome_separation_nm,
            "segregation baseline={} targeted={}",
            baseline_snapshot.chromosome_separation_nm,
            targeted_snapshot.chromosome_separation_nm
        );
    }

    #[test]
    fn test_membrane_and_septum_probes_accelerate_division() {
        let config = WholeCellConfig {
            use_gpu: false,
            dt_ms: 0.25,
            ..WholeCellConfig::default()
        };
        let mut baseline = WholeCellSimulator::new(config.clone());
        baseline.enable_local_chemistry(12, 12, 6, 0.5, false);

        let mut targeted = WholeCellSimulator::new(config);
        targeted.enable_local_chemistry(12, 12, 6, 0.5, false);
        targeted.schedule_syn3a_subsystem_probe(Syn3ASubsystemPreset::AtpSynthaseMembraneBand, 1);
        targeted.schedule_syn3a_subsystem_probe(Syn3ASubsystemPreset::RibosomePolysomeCluster, 1);
        targeted.schedule_syn3a_subsystem_probe(Syn3ASubsystemPreset::FtsZSeptumRing, 1);

        baseline.run(12);
        targeted.run(12);

        let baseline_snapshot = baseline.snapshot();
        let targeted_snapshot = targeted.snapshot();

        assert!(
            targeted_snapshot.division_progress > baseline_snapshot.division_progress,
            "division baseline={} targeted={}",
            baseline_snapshot.division_progress,
            targeted_snapshot.division_progress
        );
        assert!(
            targeted_snapshot.surface_area_nm2 > baseline_snapshot.surface_area_nm2,
            "surface baseline={} targeted={}",
            baseline_snapshot.surface_area_nm2,
            targeted_snapshot.surface_area_nm2
        );
    }
}
