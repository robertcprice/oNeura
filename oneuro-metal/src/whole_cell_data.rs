//! Data-driven whole-cell program and saved-state payloads.
//!
//! The native runtime already supports program-spec initialization and JSON
//! save/restore. This module is the serialized contract behind those flows.

use crate::whole_cell::{WholeCellConfig, WholeCellQuantumProfile};
use crate::whole_cell_submodels::{
    LocalChemistryReport, LocalChemistrySiteReport, LocalMDProbeReport, ScheduledSubsystemProbe,
    Syn3ASubsystemPreset, WholeCellSubsystemState,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::OnceLock;

const BUNDLED_SYN3A_PROGRAM_JSON: &str = include_str!("../specs/whole_cell_syn3a_reference.json");
const BUNDLED_SYN3A_ORGANISM_JSON: &str = include_str!("../specs/whole_cell_syn3a_organism.json");
pub const WHOLE_CELL_CONTRACT_VERSION: &str = "whole_cell_phase0";
pub const WHOLE_CELL_PROGRAM_SCHEMA_VERSION: u32 = 1;
pub const WHOLE_CELL_SAVED_STATE_SCHEMA_VERSION: u32 = 1;
pub const WHOLE_CELL_RUNTIME_MANIFEST_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WholeCellContractSchema {
    #[serde(default = "default_contract_version")]
    pub contract_version: String,
    #[serde(default = "default_program_schema_version")]
    pub program_schema_version: u32,
    #[serde(default = "default_saved_state_schema_version")]
    pub saved_state_schema_version: u32,
    #[serde(default = "default_runtime_manifest_schema_version")]
    pub runtime_manifest_schema_version: u32,
}

impl Default for WholeCellContractSchema {
    fn default() -> Self {
        Self {
            contract_version: default_contract_version(),
            program_schema_version: default_program_schema_version(),
            saved_state_schema_version: default_saved_state_schema_version(),
            runtime_manifest_schema_version: default_runtime_manifest_schema_version(),
        }
    }
}

impl WholeCellContractSchema {
    pub fn normalized_for_program(mut self) -> Self {
        if self.contract_version.trim().is_empty() {
            self.contract_version = default_contract_version();
        }
        self.program_schema_version = default_program_schema_version();
        if self.saved_state_schema_version == 0 {
            self.saved_state_schema_version = default_saved_state_schema_version();
        }
        if self.runtime_manifest_schema_version == 0 {
            self.runtime_manifest_schema_version = default_runtime_manifest_schema_version();
        }
        self
    }

    pub fn normalized_for_saved_state(mut self) -> Self {
        self = self.normalized_for_program();
        self.saved_state_schema_version = default_saved_state_schema_version();
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct WholeCellProvenance {
    #[serde(default)]
    pub source_dataset: Option<String>,
    #[serde(default)]
    pub organism_asset_hash: Option<String>,
    #[serde(default)]
    pub compiled_ir_hash: Option<String>,
    #[serde(default)]
    pub calibration_bundle_hash: Option<String>,
    #[serde(default)]
    pub run_manifest_hash: Option<String>,
    #[serde(default)]
    pub backend: Option<String>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WholeCellInitialLatticeSpec {
    pub atp: f32,
    pub amino_acids: f32,
    pub nucleotides: f32,
    pub membrane_precursors: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WholeCellInitialStateSpec {
    pub adp_mm: f32,
    pub glucose_mm: f32,
    pub oxygen_mm: f32,
    pub genome_bp: u32,
    pub replicated_bp: u32,
    pub chromosome_separation_nm: f32,
    pub radius_nm: f32,
    pub division_progress: f32,
    pub metabolic_load: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct WholeCellProcessWeights {
    #[serde(default)]
    pub energy: f32,
    #[serde(default)]
    pub transcription: f32,
    #[serde(default)]
    pub translation: f32,
    #[serde(default)]
    pub replication: f32,
    #[serde(default)]
    pub segregation: f32,
    #[serde(default)]
    pub membrane: f32,
    #[serde(default)]
    pub constriction: f32,
}

impl WholeCellProcessWeights {
    pub fn clamped(self) -> Self {
        Self {
            energy: self.energy.max(0.0),
            transcription: self.transcription.max(0.0),
            translation: self.translation.max(0.0),
            replication: self.replication.max(0.0),
            segregation: self.segregation.max(0.0),
            membrane: self.membrane.max(0.0),
            constriction: self.constriction.max(0.0),
        }
    }

    pub fn total(self) -> f32 {
        self.energy
            + self.transcription
            + self.translation
            + self.replication
            + self.segregation
            + self.membrane
            + self.constriction
    }

    pub fn add_weighted(&mut self, other: Self, scale: f32) {
        let other = other.clamped();
        let scale = scale.max(0.0);
        self.energy += other.energy * scale;
        self.transcription += other.transcription * scale;
        self.translation += other.translation * scale;
        self.replication += other.replication * scale;
        self.segregation += other.segregation * scale;
        self.membrane += other.membrane * scale;
        self.constriction += other.constriction * scale;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WholeCellGeometryPrior {
    pub radius_nm: f32,
    #[serde(default = "default_chromosome_radius_fraction")]
    pub chromosome_radius_fraction: f32,
    #[serde(default = "default_membrane_fraction")]
    pub membrane_fraction: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WholeCellCompositionPrior {
    #[serde(default = "default_dry_mass_fg")]
    pub dry_mass_fg: f32,
    #[serde(default = "default_gc_fraction")]
    pub gc_fraction: f32,
    #[serde(default = "default_protein_fraction")]
    pub protein_fraction: f32,
    #[serde(default = "default_rna_fraction")]
    pub rna_fraction: f32,
    #[serde(default = "default_lipid_fraction")]
    pub lipid_fraction: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellMoleculePoolSpec {
    pub species: String,
    #[serde(default)]
    pub concentration_mm: f32,
    #[serde(default)]
    pub count: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellGenomeFeature {
    pub gene: String,
    pub start_bp: u32,
    pub end_bp: u32,
    #[serde(default)]
    pub strand: i8,
    #[serde(default)]
    pub essential: bool,
    #[serde(default = "default_expression_level")]
    pub basal_expression: f32,
    #[serde(default = "default_translation_cost")]
    pub translation_cost: f32,
    #[serde(default = "default_nucleotide_cost")]
    pub nucleotide_cost: f32,
    #[serde(default)]
    pub process_weights: WholeCellProcessWeights,
    #[serde(default)]
    pub subsystem_targets: Vec<Syn3ASubsystemPreset>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellTranscriptionUnitSpec {
    pub name: String,
    #[serde(default)]
    pub genes: Vec<String>,
    #[serde(default = "default_expression_level")]
    pub basal_activity: f32,
    #[serde(default)]
    pub process_weights: WholeCellProcessWeights,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellOrganismSpec {
    pub organism: String,
    pub chromosome_length_bp: u32,
    pub origin_bp: u32,
    pub terminus_bp: u32,
    pub geometry: WholeCellGeometryPrior,
    pub composition: WholeCellCompositionPrior,
    #[serde(default)]
    pub pools: Vec<WholeCellMoleculePoolSpec>,
    #[serde(default)]
    pub genes: Vec<WholeCellGenomeFeature>,
    #[serde(default)]
    pub transcription_units: Vec<WholeCellTranscriptionUnitSpec>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WholeCellAssetClass {
    Energy,
    Translation,
    Replication,
    Segregation,
    Membrane,
    Constriction,
    QualityControl,
    Homeostasis,
    Generic,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellOperonSpec {
    pub name: String,
    #[serde(default)]
    pub genes: Vec<String>,
    pub promoter_bp: u32,
    pub terminator_bp: u32,
    pub basal_activity: f32,
    pub polycistronic: bool,
    #[serde(default)]
    pub process_weights: WholeCellProcessWeights,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellRnaProductSpec {
    pub id: String,
    pub gene: String,
    pub operon: String,
    pub length_nt: u32,
    pub basal_abundance: f32,
    pub asset_class: WholeCellAssetClass,
    #[serde(default)]
    pub process_weights: WholeCellProcessWeights,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellProteinProductSpec {
    pub id: String,
    pub gene: String,
    pub operon: String,
    pub rna_id: String,
    pub aa_length: u32,
    pub basal_abundance: f32,
    pub translation_cost: f32,
    pub nucleotide_cost: f32,
    pub asset_class: WholeCellAssetClass,
    #[serde(default)]
    pub process_weights: WholeCellProcessWeights,
    #[serde(default)]
    pub subsystem_targets: Vec<Syn3ASubsystemPreset>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WholeCellComplexComponentSpec {
    pub protein_id: String,
    pub stoichiometry: u16,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellComplexSpec {
    pub id: String,
    pub name: String,
    pub operon: String,
    #[serde(default)]
    pub components: Vec<WholeCellComplexComponentSpec>,
    pub basal_abundance: f32,
    pub asset_class: WholeCellAssetClass,
    #[serde(default)]
    pub process_weights: WholeCellProcessWeights,
    #[serde(default)]
    pub subsystem_targets: Vec<Syn3ASubsystemPreset>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellGenomeAssetPackage {
    pub organism: String,
    pub chromosome_length_bp: u32,
    pub origin_bp: u32,
    pub terminus_bp: u32,
    #[serde(default)]
    pub operons: Vec<WholeCellOperonSpec>,
    #[serde(default)]
    pub rnas: Vec<WholeCellRnaProductSpec>,
    #[serde(default)]
    pub proteins: Vec<WholeCellProteinProductSpec>,
    #[serde(default)]
    pub complexes: Vec<WholeCellComplexSpec>,
    #[serde(default)]
    pub pools: Vec<WholeCellMoleculePoolSpec>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WholeCellGenomeAssetSummary {
    pub organism: String,
    pub operon_count: usize,
    pub polycistronic_operon_count: usize,
    pub rna_count: usize,
    pub protein_count: usize,
    pub complex_count: usize,
    pub targeted_complex_count: usize,
}

impl From<&WholeCellGenomeAssetPackage> for WholeCellGenomeAssetSummary {
    fn from(package: &WholeCellGenomeAssetPackage) -> Self {
        Self {
            organism: package.organism.clone(),
            operon_count: package.operons.len(),
            polycistronic_operon_count: package
                .operons
                .iter()
                .filter(|operon| operon.polycistronic)
                .count(),
            rna_count: package.rnas.len(),
            protein_count: package.proteins.len(),
            complex_count: package.complexes.len(),
            targeted_complex_count: package
                .complexes
                .iter()
                .filter(|complex| !complex.subsystem_targets.is_empty())
                .count(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WholeCellOrganismSummary {
    pub organism: String,
    pub chromosome_length_bp: u32,
    pub gene_count: usize,
    pub transcription_unit_count: usize,
    pub pool_count: usize,
}

impl From<&WholeCellOrganismSpec> for WholeCellOrganismSummary {
    fn from(spec: &WholeCellOrganismSpec) -> Self {
        Self {
            organism: spec.organism.clone(),
            chromosome_length_bp: spec.chromosome_length_bp,
            gene_count: spec.genes.len(),
            transcription_unit_count: spec.transcription_units.len(),
            pool_count: spec.pools.len(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellOrganismProfile {
    pub organism: String,
    pub chromosome_length_bp: u32,
    pub gene_count: usize,
    pub transcription_unit_count: usize,
    pub pool_count: usize,
    pub essential_gene_fraction: f32,
    pub polycistronic_fraction: f32,
    pub coding_density: f32,
    pub mean_gene_length_bp: f32,
    pub process_scales: WholeCellProcessWeights,
    pub metabolic_burden_scale: f32,
    pub crowding_scale: f32,
    pub preferred_radius_nm: f32,
    pub chromosome_radius_fraction: f32,
    pub membrane_fraction: f32,
}

impl Default for WholeCellOrganismProfile {
    fn default() -> Self {
        Self {
            organism: "generic".to_string(),
            chromosome_length_bp: 1,
            gene_count: 0,
            transcription_unit_count: 0,
            pool_count: 0,
            essential_gene_fraction: 0.0,
            polycistronic_fraction: 0.0,
            coding_density: 0.0,
            mean_gene_length_bp: 0.0,
            process_scales: WholeCellProcessWeights {
                energy: 1.0,
                transcription: 1.0,
                translation: 1.0,
                replication: 1.0,
                segregation: 1.0,
                membrane: 1.0,
                constriction: 1.0,
            },
            metabolic_burden_scale: 1.0,
            crowding_scale: 1.0,
            preferred_radius_nm: 200.0,
            chromosome_radius_fraction: default_chromosome_radius_fraction(),
            membrane_fraction: default_membrane_fraction(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellTranscriptionUnitState {
    pub name: String,
    pub gene_count: usize,
    pub copy_gain: f32,
    pub basal_activity: f32,
    pub effective_activity: f32,
    pub support_level: f32,
    pub stress_penalty: f32,
    pub transcript_abundance: f32,
    pub protein_abundance: f32,
    pub transcript_synthesis_rate: f32,
    pub protein_synthesis_rate: f32,
    pub transcript_turnover_rate: f32,
    pub protein_turnover_rate: f32,
    pub process_drive: WholeCellProcessWeights,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WholeCellComplexAssemblyState {
    pub atp_band_complexes: f32,
    pub ribosome_complexes: f32,
    pub rnap_complexes: f32,
    pub replisome_complexes: f32,
    pub membrane_complexes: f32,
    pub ftsz_polymer: f32,
    pub dnaa_activity: f32,
    pub atp_band_target: f32,
    pub ribosome_target: f32,
    pub rnap_target: f32,
    pub replisome_target: f32,
    pub membrane_target: f32,
    pub ftsz_target: f32,
    pub dnaa_target: f32,
    pub atp_band_assembly_rate: f32,
    pub ribosome_assembly_rate: f32,
    pub rnap_assembly_rate: f32,
    pub replisome_assembly_rate: f32,
    pub membrane_assembly_rate: f32,
    pub ftsz_assembly_rate: f32,
    pub dnaa_assembly_rate: f32,
    pub atp_band_degradation_rate: f32,
    pub ribosome_degradation_rate: f32,
    pub rnap_degradation_rate: f32,
    pub replisome_degradation_rate: f32,
    pub membrane_degradation_rate: f32,
    pub ftsz_degradation_rate: f32,
    pub dnaa_degradation_rate: f32,
}

impl WholeCellComplexAssemblyState {
    pub fn total_complexes(self) -> f32 {
        self.atp_band_complexes
            + self.ribosome_complexes
            + self.rnap_complexes
            + self.replisome_complexes
            + self.membrane_complexes
            + self.ftsz_polymer
            + self.dnaa_activity
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellNamedComplexState {
    pub id: String,
    pub operon: String,
    pub asset_class: WholeCellAssetClass,
    #[serde(default)]
    pub subsystem_targets: Vec<Syn3ASubsystemPreset>,
    pub abundance: f32,
    pub target_abundance: f32,
    pub assembly_rate: f32,
    pub degradation_rate: f32,
    pub component_satisfaction: f32,
    pub structural_support: f32,
}

impl Default for WholeCellComplexAssemblyState {
    fn default() -> Self {
        Self {
            atp_band_complexes: 0.0,
            ribosome_complexes: 0.0,
            rnap_complexes: 0.0,
            replisome_complexes: 0.0,
            membrane_complexes: 0.0,
            ftsz_polymer: 0.0,
            dnaa_activity: 0.0,
            atp_band_target: 0.0,
            ribosome_target: 0.0,
            rnap_target: 0.0,
            replisome_target: 0.0,
            membrane_target: 0.0,
            ftsz_target: 0.0,
            dnaa_target: 0.0,
            atp_band_assembly_rate: 0.0,
            ribosome_assembly_rate: 0.0,
            rnap_assembly_rate: 0.0,
            replisome_assembly_rate: 0.0,
            membrane_assembly_rate: 0.0,
            ftsz_assembly_rate: 0.0,
            dnaa_assembly_rate: 0.0,
            atp_band_degradation_rate: 0.0,
            ribosome_degradation_rate: 0.0,
            rnap_degradation_rate: 0.0,
            replisome_degradation_rate: 0.0,
            membrane_degradation_rate: 0.0,
            ftsz_degradation_rate: 0.0,
            dnaa_degradation_rate: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellOrganismExpressionState {
    pub global_activity: f32,
    pub energy_support: f32,
    pub translation_support: f32,
    pub nucleotide_support: f32,
    pub membrane_support: f32,
    pub crowding_penalty: f32,
    pub metabolic_burden_scale: f32,
    pub process_scales: WholeCellProcessWeights,
    pub amino_cost_scale: f32,
    pub nucleotide_cost_scale: f32,
    pub total_transcript_abundance: f32,
    pub total_protein_abundance: f32,
    #[serde(default)]
    pub transcription_units: Vec<WholeCellTranscriptionUnitState>,
}

impl Default for WholeCellOrganismExpressionState {
    fn default() -> Self {
        Self {
            global_activity: 1.0,
            energy_support: 1.0,
            translation_support: 1.0,
            nucleotide_support: 1.0,
            membrane_support: 1.0,
            crowding_penalty: 1.0,
            metabolic_burden_scale: 1.0,
            process_scales: WholeCellProcessWeights {
                energy: 1.0,
                transcription: 1.0,
                translation: 1.0,
                replication: 1.0,
                segregation: 1.0,
                membrane: 1.0,
                constriction: 1.0,
            },
            amino_cost_scale: 1.0,
            nucleotide_cost_scale: 1.0,
            total_transcript_abundance: 0.0,
            total_protein_abundance: 0.0,
            transcription_units: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellLocalChemistrySpec {
    pub x_dim: usize,
    pub y_dim: usize,
    pub z_dim: usize,
    pub voxel_size_au: f32,
    pub use_gpu: bool,
    #[serde(default)]
    pub enable_default_syn3a_subsystems: bool,
    #[serde(default)]
    pub scheduled_subsystem_probes: Vec<ScheduledSubsystemProbe>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellProgramSpec {
    #[serde(default, alias = "name")]
    pub program_name: Option<String>,
    #[serde(default)]
    pub contract: WholeCellContractSchema,
    #[serde(default)]
    pub provenance: WholeCellProvenance,
    #[serde(default)]
    pub organism_data_ref: Option<String>,
    #[serde(default)]
    pub organism_data: Option<WholeCellOrganismSpec>,
    #[serde(default)]
    pub organism_assets: Option<WholeCellGenomeAssetPackage>,
    pub config: WholeCellConfig,
    pub initial_lattice: WholeCellInitialLatticeSpec,
    pub initial_state: WholeCellInitialStateSpec,
    #[serde(default)]
    pub quantum_profile: WholeCellQuantumProfile,
    #[serde(default)]
    pub local_chemistry: Option<WholeCellLocalChemistrySpec>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellSavedCoreState {
    pub time_ms: f32,
    pub step_count: u64,
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
    pub metabolic_load: f32,
    pub quantum_profile: WholeCellQuantumProfile,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellLatticeState {
    pub atp: Vec<f32>,
    pub amino_acids: Vec<f32>,
    pub nucleotides: Vec<f32>,
    pub membrane_precursors: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WholeCellSavedState {
    pub program_name: Option<String>,
    #[serde(default)]
    pub contract: WholeCellContractSchema,
    #[serde(default)]
    pub provenance: WholeCellProvenance,
    #[serde(default)]
    pub organism_data_ref: Option<String>,
    #[serde(default)]
    pub organism_data: Option<WholeCellOrganismSpec>,
    #[serde(default)]
    pub organism_assets: Option<WholeCellGenomeAssetPackage>,
    #[serde(default)]
    pub organism_expression: WholeCellOrganismExpressionState,
    #[serde(default)]
    pub complex_assembly: WholeCellComplexAssemblyState,
    #[serde(default)]
    pub named_complexes: Vec<WholeCellNamedComplexState>,
    pub config: WholeCellConfig,
    pub core: WholeCellSavedCoreState,
    pub lattice: WholeCellLatticeState,
    #[serde(default)]
    pub local_chemistry: Option<WholeCellLocalChemistrySpec>,
    #[serde(default)]
    pub chemistry_report: LocalChemistryReport,
    #[serde(default)]
    pub chemistry_site_reports: Vec<LocalChemistrySiteReport>,
    #[serde(default)]
    pub last_md_probe: Option<LocalMDProbeReport>,
    #[serde(default)]
    pub scheduled_subsystem_probes: Vec<ScheduledSubsystemProbe>,
    #[serde(default)]
    pub subsystem_states: Vec<WholeCellSubsystemState>,
    #[serde(default = "default_scale")]
    pub md_translation_scale: f32,
    #[serde(default = "default_scale")]
    pub md_membrane_scale: f32,
}

fn default_scale() -> f32 {
    1.0
}

fn default_contract_version() -> String {
    WHOLE_CELL_CONTRACT_VERSION.to_string()
}

fn default_program_schema_version() -> u32 {
    WHOLE_CELL_PROGRAM_SCHEMA_VERSION
}

fn default_saved_state_schema_version() -> u32 {
    WHOLE_CELL_SAVED_STATE_SCHEMA_VERSION
}

fn default_runtime_manifest_schema_version() -> u32 {
    WHOLE_CELL_RUNTIME_MANIFEST_SCHEMA_VERSION
}

fn default_expression_level() -> f32 {
    1.0
}

fn default_translation_cost() -> f32 {
    1.0
}

fn default_nucleotide_cost() -> f32 {
    1.0
}

fn default_chromosome_radius_fraction() -> f32 {
    0.55
}

fn default_membrane_fraction() -> f32 {
    0.24
}

fn default_dry_mass_fg() -> f32 {
    130.0
}

fn default_gc_fraction() -> f32 {
    0.31
}

fn default_protein_fraction() -> f32 {
    0.56
}

fn default_rna_fraction() -> f32 {
    0.22
}

fn default_lipid_fraction() -> f32 {
    0.12
}

fn stable_checksum_hex(bytes: &[u8]) -> String {
    const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET_BASIS;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{hash:016x}")
}

fn stable_json_checksum<T: Serialize>(value: &T) -> Result<String, String> {
    serde_json::to_vec(value)
        .map(|bytes| stable_checksum_hex(&bytes))
        .map_err(|error| format!("failed to serialize whole-cell contract payload: {error}"))
}

fn populate_program_contract_metadata(spec: &mut WholeCellProgramSpec) -> Result<(), String> {
    spec.contract = spec.contract.clone().normalized_for_program();
    if spec.provenance.organism_asset_hash.is_none() {
        if let Some(assets) = spec.organism_assets.as_ref() {
            spec.provenance.organism_asset_hash = Some(stable_json_checksum(assets)?);
        }
    }
    if spec.provenance.run_manifest_hash.is_none() {
        let manifest_view = (
            &spec.program_name,
            &spec.organism_data_ref,
            &spec.config,
            &spec.local_chemistry,
            &spec.contract,
        );
        spec.provenance.run_manifest_hash = Some(stable_json_checksum(&manifest_view)?);
    }
    Ok(())
}

fn populate_saved_state_contract_metadata(state: &mut WholeCellSavedState) -> Result<(), String> {
    state.contract = state.contract.clone().normalized_for_saved_state();
    if state.provenance.organism_asset_hash.is_none() {
        if let Some(assets) = state.organism_assets.as_ref() {
            state.provenance.organism_asset_hash = Some(stable_json_checksum(assets)?);
        }
    }
    if state.provenance.run_manifest_hash.is_none() {
        let manifest_view = (
            &state.program_name,
            &state.organism_data_ref,
            &state.config,
            &state.local_chemistry,
            &state.contract,
        );
        state.provenance.run_manifest_hash = Some(stable_json_checksum(&manifest_view)?);
    }
    Ok(())
}

fn normalized_share(raw: f32, total: f32) -> f32 {
    let total = total.max(1.0e-6);
    (raw.max(0.0) / total).clamp(0.0, 1.0)
}

fn derived_process_scale(
    share: f32,
    essential_fraction: f32,
    polycistronic_fraction: f32,
    bias: f32,
) -> f32 {
    (0.78 + 1.12 * share + 0.08 * essential_fraction + 0.06 * polycistronic_fraction + bias)
        .clamp(0.70, 1.45)
}

pub fn derive_organism_profile(spec: &WholeCellOrganismSpec) -> WholeCellOrganismProfile {
    let chromosome_length_bp = spec.chromosome_length_bp.max(1);
    let gene_count = spec.genes.len();
    let transcription_unit_count = spec.transcription_units.len();
    let pool_count = spec.pools.len();
    let essential_gene_fraction = if gene_count > 0 {
        spec.genes.iter().filter(|gene| gene.essential).count() as f32 / gene_count as f32
    } else {
        0.0
    };
    let polycistronic_fraction = if transcription_unit_count > 0 {
        spec.transcription_units
            .iter()
            .filter(|unit| unit.genes.len() > 1)
            .count() as f32
            / transcription_unit_count as f32
    } else {
        0.0
    };
    let total_gene_bp = spec
        .genes
        .iter()
        .map(|gene| {
            if gene.end_bp >= gene.start_bp {
                (gene.end_bp - gene.start_bp + 1) as f32
            } else {
                0.0
            }
        })
        .sum::<f32>();
    let mean_gene_length_bp = if gene_count > 0 {
        total_gene_bp / gene_count as f32
    } else {
        0.0
    };
    let coding_density = (total_gene_bp / chromosome_length_bp as f32).clamp(0.0, 1.0);

    let mut process_totals = WholeCellProcessWeights::default();
    for gene in &spec.genes {
        let scale = gene.basal_expression.max(0.0);
        process_totals.add_weighted(gene.process_weights, scale);
        process_totals.translation += gene.translation_cost.max(0.0) * 0.06 * scale;
        process_totals.replication += gene.nucleotide_cost.max(0.0) * 0.04 * scale;
    }
    for unit in &spec.transcription_units {
        process_totals.add_weighted(unit.process_weights, unit.basal_activity.max(0.0));
    }
    for pool in &spec.pools {
        let species = pool.species.to_ascii_lowercase();
        let scale = pool.concentration_mm.max(0.0) + 0.002 * pool.count.max(0.0);
        if species.contains("atp") || species.contains("oxygen") || species.contains("glucose") {
            process_totals.energy += 0.04 * scale;
        }
        if species.contains("rib") || species.contains("amino") {
            process_totals.translation += 0.03 * scale;
        }
        if species.contains("nucleotide") || species.contains("dna") {
            process_totals.replication += 0.03 * scale;
        }
        if species.contains("membrane") || species.contains("lipid") || species.contains("phosph") {
            process_totals.membrane += 0.03 * scale;
        }
    }

    let process_total = process_totals.total();
    let composition = spec.composition;
    let geometry = spec.geometry;
    let energy_bias = 0.08 * geometry.membrane_fraction + 0.05 * composition.lipid_fraction;
    let transcription_bias = 0.10 * composition.rna_fraction + 0.04 * polycistronic_fraction;
    let translation_bias = 0.08 * composition.protein_fraction + 0.04 * polycistronic_fraction;
    let replication_bias = 0.05 * (1.0 - composition.gc_fraction.clamp(0.0, 1.0));
    let segregation_bias = 0.05 * geometry.chromosome_radius_fraction.clamp(0.0, 1.0);
    let membrane_bias = 0.12 * geometry.membrane_fraction + 0.06 * composition.lipid_fraction;
    let constriction_bias = 0.10 * geometry.membrane_fraction + 0.05 * essential_gene_fraction;

    let process_scales = WholeCellProcessWeights {
        energy: derived_process_scale(
            normalized_share(process_totals.energy, process_total),
            essential_gene_fraction,
            polycistronic_fraction,
            energy_bias,
        ),
        transcription: derived_process_scale(
            normalized_share(process_totals.transcription, process_total),
            essential_gene_fraction,
            polycistronic_fraction,
            transcription_bias,
        ),
        translation: derived_process_scale(
            normalized_share(process_totals.translation, process_total),
            essential_gene_fraction,
            polycistronic_fraction,
            translation_bias,
        ),
        replication: derived_process_scale(
            normalized_share(process_totals.replication, process_total),
            essential_gene_fraction,
            polycistronic_fraction,
            replication_bias,
        ),
        segregation: derived_process_scale(
            normalized_share(process_totals.segregation, process_total),
            essential_gene_fraction,
            polycistronic_fraction,
            segregation_bias,
        ),
        membrane: derived_process_scale(
            normalized_share(process_totals.membrane, process_total),
            essential_gene_fraction,
            polycistronic_fraction,
            membrane_bias,
        ),
        constriction: derived_process_scale(
            normalized_share(process_totals.constriction, process_total),
            essential_gene_fraction,
            polycistronic_fraction,
            constriction_bias,
        ),
    };

    let metabolic_burden_scale = (0.84
        + 0.16 * coding_density
        + 0.10 * essential_gene_fraction
        + 0.08 * composition.protein_fraction
        + 0.04 * polycistronic_fraction)
        .clamp(0.85, 1.55);
    let crowding_scale = (0.88
        + 0.20 * coding_density
        + 0.12 * composition.protein_fraction
        + 0.05 * geometry.chromosome_radius_fraction)
        .clamp(0.85, 1.25);

    WholeCellOrganismProfile {
        organism: spec.organism.clone(),
        chromosome_length_bp,
        gene_count,
        transcription_unit_count,
        pool_count,
        essential_gene_fraction,
        polycistronic_fraction,
        coding_density,
        mean_gene_length_bp,
        process_scales,
        metabolic_burden_scale,
        crowding_scale,
        preferred_radius_nm: geometry.radius_nm.max(50.0),
        chromosome_radius_fraction: geometry.chromosome_radius_fraction.clamp(0.1, 0.95),
        membrane_fraction: geometry.membrane_fraction.clamp(0.05, 0.95),
    }
}

fn gene_length_bp(feature: &WholeCellGenomeFeature) -> u32 {
    if feature.end_bp >= feature.start_bp {
        feature.end_bp - feature.start_bp + 1
    } else {
        0
    }
}

fn inferred_asset_class(
    weights: WholeCellProcessWeights,
    subsystem_targets: &[Syn3ASubsystemPreset],
    name: &str,
) -> WholeCellAssetClass {
    if subsystem_targets.contains(&Syn3ASubsystemPreset::AtpSynthaseMembraneBand) {
        return WholeCellAssetClass::Energy;
    }
    if subsystem_targets.contains(&Syn3ASubsystemPreset::RibosomePolysomeCluster) {
        return WholeCellAssetClass::Translation;
    }
    if subsystem_targets.contains(&Syn3ASubsystemPreset::ReplisomeTrack) {
        return WholeCellAssetClass::Replication;
    }
    if subsystem_targets.contains(&Syn3ASubsystemPreset::FtsZSeptumRing) {
        return WholeCellAssetClass::Constriction;
    }

    let name = name.to_ascii_lowercase();
    if name.contains("chaperone") || name.contains("quality_control") {
        return WholeCellAssetClass::QualityControl;
    }
    if name.contains("transport") || name.contains("homeostasis") {
        return WholeCellAssetClass::Homeostasis;
    }

    let weights = weights.clamped();
    let ranked = [
        (weights.energy, WholeCellAssetClass::Energy),
        (weights.translation, WholeCellAssetClass::Translation),
        (weights.replication, WholeCellAssetClass::Replication),
        (weights.segregation, WholeCellAssetClass::Segregation),
        (weights.membrane, WholeCellAssetClass::Membrane),
        (weights.constriction, WholeCellAssetClass::Constriction),
        (weights.transcription, WholeCellAssetClass::Homeostasis),
    ];
    ranked
        .into_iter()
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(_, asset_class)| asset_class)
        .unwrap_or(WholeCellAssetClass::Generic)
}

fn operon_bounds(spec: &WholeCellOrganismSpec, genes: &[String]) -> (u32, u32) {
    let mut promoter_bp = u32::MAX;
    let mut terminator_bp = 0u32;
    for gene_name in genes {
        if let Some(feature) = spec.genes.iter().find(|feature| feature.gene == *gene_name) {
            promoter_bp = promoter_bp.min(feature.start_bp.min(feature.end_bp));
            terminator_bp = terminator_bp.max(feature.start_bp.max(feature.end_bp));
        }
    }
    if promoter_bp == u32::MAX {
        (0, 0)
    } else {
        (promoter_bp, terminator_bp)
    }
}

fn push_unique_subsystem_targets(
    targets: &mut Vec<Syn3ASubsystemPreset>,
    candidates: &[Syn3ASubsystemPreset],
) {
    for candidate in candidates {
        if !targets.contains(candidate) {
            targets.push(*candidate);
        }
    }
}

pub fn compile_genome_asset_package(spec: &WholeCellOrganismSpec) -> WholeCellGenomeAssetPackage {
    let mut gene_to_operon = HashMap::<String, String>::new();
    let mut operons = Vec::new();

    for transcription_unit in &spec.transcription_units {
        let (promoter_bp, terminator_bp) = operon_bounds(spec, &transcription_unit.genes);
        for gene_name in &transcription_unit.genes {
            gene_to_operon.insert(gene_name.clone(), transcription_unit.name.clone());
        }
        operons.push(WholeCellOperonSpec {
            name: transcription_unit.name.clone(),
            genes: transcription_unit.genes.clone(),
            promoter_bp,
            terminator_bp,
            basal_activity: transcription_unit.basal_activity.max(0.0),
            polycistronic: transcription_unit.genes.len() > 1,
            process_weights: transcription_unit.process_weights.clamped(),
        });
    }

    for gene in &spec.genes {
        if gene_to_operon.contains_key(&gene.gene) {
            continue;
        }
        gene_to_operon.insert(gene.gene.clone(), gene.gene.clone());
        operons.push(WholeCellOperonSpec {
            name: gene.gene.clone(),
            genes: vec![gene.gene.clone()],
            promoter_bp: gene.start_bp.min(gene.end_bp),
            terminator_bp: gene.start_bp.max(gene.end_bp),
            basal_activity: gene.basal_expression.max(0.0),
            polycistronic: false,
            process_weights: gene.process_weights.clamped(),
        });
    }

    let mut rnas = Vec::new();
    let mut proteins = Vec::new();
    for gene in &spec.genes {
        let length_nt = gene_length_bp(gene).max(1);
        let operon_name = gene_to_operon
            .get(&gene.gene)
            .cloned()
            .unwrap_or_else(|| gene.gene.clone());
        let asset_class =
            inferred_asset_class(gene.process_weights, &gene.subsystem_targets, &gene.gene);
        let rna_id = format!("{}_rna", gene.gene);
        let protein_id = format!("{}_protein", gene.gene);
        rnas.push(WholeCellRnaProductSpec {
            id: rna_id.clone(),
            gene: gene.gene.clone(),
            operon: operon_name.clone(),
            length_nt,
            basal_abundance: (4.0 + 6.0 * gene.basal_expression.max(0.05)).clamp(0.5, 256.0),
            asset_class,
            process_weights: gene.process_weights.clamped(),
        });
        proteins.push(WholeCellProteinProductSpec {
            id: protein_id,
            gene: gene.gene.clone(),
            operon: operon_name,
            rna_id,
            aa_length: (length_nt / 3).max(1),
            basal_abundance: (8.0 + 10.0 * gene.basal_expression.max(0.05)).clamp(0.5, 512.0),
            translation_cost: gene.translation_cost.max(0.0),
            nucleotide_cost: gene.nucleotide_cost.max(0.0),
            asset_class,
            process_weights: gene.process_weights.clamped(),
            subsystem_targets: gene.subsystem_targets.clone(),
        });
    }

    let mut complexes = Vec::new();
    for operon in &operons {
        let mut components = Vec::new();
        let mut process_weights = operon.process_weights;
        let mut subsystem_targets = Vec::new();
        for gene_name in &operon.genes {
            components.push(WholeCellComplexComponentSpec {
                protein_id: format!("{}_protein", gene_name),
                stoichiometry: 1,
            });
            if let Some(gene) = spec.genes.iter().find(|gene| gene.gene == *gene_name) {
                process_weights.add_weighted(gene.process_weights, 0.35);
                push_unique_subsystem_targets(&mut subsystem_targets, &gene.subsystem_targets);
            }
        }
        let asset_class = inferred_asset_class(process_weights, &subsystem_targets, &operon.name);
        complexes.push(WholeCellComplexSpec {
            id: format!("{}_complex", operon.name),
            name: format!("{} complex", operon.name),
            operon: operon.name.clone(),
            components,
            basal_abundance: (3.0
                + 7.0
                    * operon.basal_activity.max(0.05)
                    * (operon.genes.len().max(1) as f32).sqrt())
            .clamp(0.5, 256.0),
            asset_class,
            process_weights: process_weights.clamped(),
            subsystem_targets,
        });
    }

    WholeCellGenomeAssetPackage {
        organism: spec.organism.clone(),
        chromosome_length_bp: spec.chromosome_length_bp.max(1),
        origin_bp: spec.origin_bp.min(spec.chromosome_length_bp.max(1)),
        terminus_bp: spec.terminus_bp.min(spec.chromosome_length_bp.max(1)),
        operons,
        rnas,
        proteins,
        complexes,
        pools: spec.pools.clone(),
    }
}

pub fn parse_program_spec_json(spec_json: &str) -> Result<WholeCellProgramSpec, String> {
    let mut spec: WholeCellProgramSpec = serde_json::from_str(spec_json)
        .map_err(|error| format!("failed to parse program spec: {error}"))?;
    if spec.organism_data.is_none() {
        if let Some(reference) = spec.organism_data_ref.as_deref() {
            spec.organism_data = Some(resolve_bundled_organism_spec(reference)?);
        }
    }
    if spec.organism_assets.is_none() {
        if let Some(organism) = spec.organism_data.as_ref() {
            spec.organism_assets = Some(compile_genome_asset_package(organism));
        } else if let Some(reference) = spec.organism_data_ref.as_deref() {
            spec.organism_assets = Some(resolve_bundled_genome_asset_package(reference)?);
        }
    }
    populate_program_contract_metadata(&mut spec)?;
    Ok(spec)
}

pub fn bundled_syn3a_program_spec_json() -> &'static str {
    BUNDLED_SYN3A_PROGRAM_JSON
}

pub fn parse_organism_spec_json(spec_json: &str) -> Result<WholeCellOrganismSpec, String> {
    serde_json::from_str(spec_json)
        .map_err(|error| format!("failed to parse organism spec: {error}"))
}

pub fn parse_genome_asset_package_json(
    spec_json: &str,
) -> Result<WholeCellGenomeAssetPackage, String> {
    serde_json::from_str(spec_json)
        .map_err(|error| format!("failed to parse genome asset package: {error}"))
}

pub fn bundled_syn3a_organism_spec_json() -> &'static str {
    BUNDLED_SYN3A_ORGANISM_JSON
}

pub fn bundled_syn3a_organism_spec() -> Result<WholeCellOrganismSpec, String> {
    static BUNDLED_ORGANISM: OnceLock<Result<WholeCellOrganismSpec, String>> = OnceLock::new();
    BUNDLED_ORGANISM
        .get_or_init(|| parse_organism_spec_json(BUNDLED_SYN3A_ORGANISM_JSON))
        .clone()
}

pub fn bundled_syn3a_genome_asset_package_json() -> Result<&'static str, String> {
    static BUNDLED_ASSET_JSON: OnceLock<Result<String, String>> = OnceLock::new();
    match BUNDLED_ASSET_JSON.get_or_init(|| {
        bundled_syn3a_genome_asset_package().and_then(|package| {
            serde_json::to_string_pretty(&package).map_err(|error| {
                format!("failed to serialize bundled genome asset package: {error}")
            })
        })
    }) {
        Ok(json) => Ok(json.as_str()),
        Err(error) => Err(error.clone()),
    }
}

pub fn bundled_syn3a_genome_asset_package() -> Result<WholeCellGenomeAssetPackage, String> {
    static BUNDLED_ASSET_PACKAGE: OnceLock<Result<WholeCellGenomeAssetPackage, String>> =
        OnceLock::new();
    BUNDLED_ASSET_PACKAGE
        .get_or_init(|| {
            bundled_syn3a_organism_spec().map(|organism| compile_genome_asset_package(&organism))
        })
        .clone()
}

pub fn resolve_bundled_organism_spec(reference: &str) -> Result<WholeCellOrganismSpec, String> {
    match reference.trim().to_lowercase().as_str() {
        "jcvi_syn3a_reference" | "jcvi-syn3a" | "syn3a" | "syn3a_reference" => {
            bundled_syn3a_organism_spec()
        }
        _ => Err(format!("unknown bundled organism reference: {reference}")),
    }
}

pub fn resolve_bundled_genome_asset_package(
    reference: &str,
) -> Result<WholeCellGenomeAssetPackage, String> {
    match reference.trim().to_lowercase().as_str() {
        "jcvi_syn3a_reference" | "jcvi-syn3a" | "syn3a" | "syn3a_reference" => {
            bundled_syn3a_genome_asset_package()
        }
        _ => Err(format!(
            "unknown bundled genome asset package reference: {reference}"
        )),
    }
}

pub fn bundled_syn3a_program_spec() -> Result<WholeCellProgramSpec, String> {
    static BUNDLED_SPEC: OnceLock<Result<WholeCellProgramSpec, String>> = OnceLock::new();
    BUNDLED_SPEC
        .get_or_init(|| parse_program_spec_json(BUNDLED_SYN3A_PROGRAM_JSON))
        .clone()
}

pub fn parse_saved_state_json(state_json: &str) -> Result<WholeCellSavedState, String> {
    let mut state: WholeCellSavedState = serde_json::from_str(state_json)
        .map_err(|error| format!("failed to parse saved state: {error}"))?;
    populate_saved_state_contract_metadata(&mut state)?;
    Ok(state)
}

pub fn saved_state_to_json(state: &WholeCellSavedState) -> Result<String, String> {
    let mut hydrated = state.clone();
    populate_saved_state_contract_metadata(&mut hydrated)?;
    serde_json::to_string_pretty(&hydrated)
        .map_err(|error| format!("failed to serialize saved state: {error}"))
}

pub type WholeCellSeedSpec = WholeCellProgramSpec;
pub type WholeCellLocalChemistryConfig = WholeCellLocalChemistrySpec;
pub type WholeCellCheckpoint = WholeCellSavedState;

pub fn default_syn3a_seed_spec() -> Result<WholeCellProgramSpec, String> {
    bundled_syn3a_program_spec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_syn3a_program_spec_resolves_organism_data() {
        let spec = bundled_syn3a_program_spec().expect("bundled program spec");
        let organism = spec.organism_data.as_ref().expect("bundled organism data");
        let assets = spec
            .organism_assets
            .as_ref()
            .expect("bundled organism asset package");
        let profile = derive_organism_profile(organism);

        assert_eq!(
            spec.organism_data_ref.as_deref(),
            Some("jcvi_syn3a_reference")
        );
        assert_eq!(organism.organism, "JCVI-syn3A");
        assert!(profile.gene_count >= 8);
        assert!(profile.transcription_unit_count >= 4);
        assert!(profile.process_scales.translation > 0.9);
        assert!(profile.metabolic_burden_scale > 0.9);
        assert!(assets.operons.len() >= 4);
        assert_eq!(assets.rnas.len(), organism.genes.len());
        assert_eq!(assets.proteins.len(), organism.genes.len());
        assert!(assets.complexes.len() >= 4);
    }

    #[test]
    fn parse_program_spec_json_hydrates_bundled_organism_data() {
        let spec = parse_program_spec_json(
            r#"{
                "program_name": "test_program",
                "organism_data_ref": "syn3a",
                "config": {
                    "x_dim": 8,
                    "y_dim": 8,
                    "z_dim": 4,
                    "voxel_size_nm": 20.0,
                    "dt_ms": 0.25,
                    "cme_interval": 4,
                    "ode_interval": 1,
                    "bd_interval": 2,
                    "geometry_interval": 4,
                    "use_gpu": false
                },
                "initial_lattice": {
                    "atp": 1.0,
                    "amino_acids": 1.0,
                    "nucleotides": 1.0,
                    "membrane_precursors": 1.0
                },
                "initial_state": {
                    "adp_mm": 0.2,
                    "glucose_mm": 0.3,
                    "oxygen_mm": 0.4,
                    "genome_bp": 1,
                    "replicated_bp": 0,
                    "chromosome_separation_nm": 10.0,
                    "radius_nm": 100.0,
                    "division_progress": 0.0,
                    "metabolic_load": 1.0
                }
            }"#,
        )
        .expect("spec with bundled organism");

        assert!(spec.organism_data.is_some());
        assert!(spec.organism_assets.is_some());
        assert_eq!(spec.organism_data_ref.as_deref(), Some("syn3a"));
        assert_eq!(spec.contract.contract_version, WHOLE_CELL_CONTRACT_VERSION);
        assert_eq!(
            spec.contract.program_schema_version,
            WHOLE_CELL_PROGRAM_SCHEMA_VERSION
        );
        assert!(spec.provenance.organism_asset_hash.is_some());
        assert!(spec.provenance.run_manifest_hash.is_some());
    }

    #[test]
    fn bundled_syn3a_genome_asset_package_compiles_from_descriptor() {
        let organism = bundled_syn3a_organism_spec().expect("bundled organism");
        let package = bundled_syn3a_genome_asset_package().expect("bundled asset package");
        let summary = WholeCellGenomeAssetSummary::from(&package);

        assert_eq!(package.organism, organism.organism);
        assert_eq!(package.rnas.len(), organism.genes.len());
        assert_eq!(package.proteins.len(), organism.genes.len());
        assert!(package.complexes.len() >= organism.transcription_units.len());
        assert!(summary.operon_count >= organism.transcription_units.len());
        assert!(summary.protein_count >= 8);
        assert!(summary.targeted_complex_count >= 4);
    }

    #[test]
    fn bundled_syn3a_genome_asset_package_json_round_trips() {
        let package_json =
            bundled_syn3a_genome_asset_package_json().expect("bundled asset package json");
        let package = parse_genome_asset_package_json(package_json).expect("parsed asset package");

        assert_eq!(package.organism, "JCVI-syn3A");
        assert!(package.operons.iter().any(|operon| operon.polycistronic));
        assert!(package
            .complexes
            .iter()
            .any(|complex| !complex.subsystem_targets.is_empty()));
    }

    #[test]
    fn saved_state_json_hydrates_contract_defaults() {
        let spec = bundled_syn3a_program_spec().expect("bundled spec");
        let mut saved = WholeCellSavedState {
            program_name: spec.program_name.clone(),
            contract: WholeCellContractSchema::default(),
            provenance: WholeCellProvenance::default(),
            organism_data_ref: spec.organism_data_ref.clone(),
            organism_data: spec.organism_data.clone(),
            organism_assets: spec.organism_assets.clone(),
            organism_expression: WholeCellOrganismExpressionState::default(),
            complex_assembly: WholeCellComplexAssemblyState::default(),
            named_complexes: Vec::new(),
            config: spec.config.clone(),
            core: WholeCellSavedCoreState {
                time_ms: 0.0,
                step_count: 0,
                adp_mm: 0.2,
                glucose_mm: 0.3,
                oxygen_mm: 0.4,
                ftsz: 1.0,
                dnaa: 1.0,
                active_ribosomes: 1.0,
                active_rnap: 1.0,
                genome_bp: 10,
                replicated_bp: 0,
                chromosome_separation_nm: 1.0,
                radius_nm: 100.0,
                surface_area_nm2: 10.0,
                volume_nm3: 10.0,
                division_progress: 0.0,
                metabolic_load: 1.0,
                quantum_profile: WholeCellQuantumProfile::default(),
            },
            lattice: WholeCellLatticeState {
                atp: vec![1.0; spec.config.x_dim * spec.config.y_dim * spec.config.z_dim],
                amino_acids: vec![1.0; spec.config.x_dim * spec.config.y_dim * spec.config.z_dim],
                nucleotides: vec![1.0; spec.config.x_dim * spec.config.y_dim * spec.config.z_dim],
                membrane_precursors: vec![
                    1.0;
                    spec.config.x_dim * spec.config.y_dim * spec.config.z_dim
                ],
            },
            local_chemistry: None,
            chemistry_report: LocalChemistryReport::default(),
            chemistry_site_reports: Vec::new(),
            last_md_probe: None,
            scheduled_subsystem_probes: Vec::new(),
            subsystem_states: Vec::new(),
            md_translation_scale: 1.0,
            md_membrane_scale: 1.0,
        };

        let json = saved_state_to_json(&saved).expect("saved json");
        saved.contract.contract_version.clear();
        let reparsed = parse_saved_state_json(&json).expect("reparsed saved state");

        assert_eq!(
            reparsed.contract.saved_state_schema_version,
            WHOLE_CELL_SAVED_STATE_SCHEMA_VERSION
        );
        assert_eq!(
            reparsed.contract.contract_version,
            WHOLE_CELL_CONTRACT_VERSION
        );
        assert!(reparsed.provenance.organism_asset_hash.is_some());
        assert!(reparsed.provenance.run_manifest_hash.is_some());
    }
}
