//! Multi-scale zoom data model for Powers-of-Ten inspection.
//!
//! Provides structured data objects at each biological scale level — from
//! ecosystem down to individual atoms — all derived from the ab initio
//! quantum descriptor pipeline.

use crate::terrarium_web_protocol::{InspectData, InspectMetric};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Scale level enum
// ---------------------------------------------------------------------------

/// Hierarchical inspection scale for the Powers-of-Ten drill-down.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScaleLevel {
    /// Terrain, populations, atmosphere (~1 m – 10 m)
    Ecosystem,
    /// Body plan, tissues, organ systems (~1 mm – 10 cm)
    Organism,
    /// Cell types, organelles, membrane (~1 µm – 100 µm)
    Cellular,
    /// Molecular graphs, bond networks (~1 Å – 10 nm)
    Molecular,
    /// Individual atoms, electron config, quantum state (~1 pm – 1 Å)
    Atomic,
}

impl ScaleLevel {
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "ecosystem" => Some(Self::Ecosystem),
            "organism" => Some(Self::Organism),
            "cellular" => Some(Self::Cellular),
            "molecular" => Some(Self::Molecular),
            "atomic" => Some(Self::Atomic),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ecosystem => "ecosystem",
            Self::Organism => "organism",
            Self::Cellular => "cellular",
            Self::Molecular => "molecular",
            Self::Atomic => "atomic",
        }
    }
}

// ---------------------------------------------------------------------------
// Atomic-level detail
// ---------------------------------------------------------------------------

/// Per-atom render and inspection data, fully derived from `PeriodicElement`
/// and quantum descriptors.
#[derive(Debug, Clone, Serialize)]
pub struct AtomVisual {
    pub element: String,
    pub symbol: String,
    pub atomic_number: u8,
    /// 3-D position in angstroms (within the parent molecule).
    pub position: [f32; 3],
    /// Van der Waals radius in angstroms (Bondi/Mantina).
    pub vdw_radius: f32,
    /// CPK color as [R, G, B].
    pub cpk_color: [u8; 3],
    /// Formal charge on this atom.
    pub formal_charge: i8,
    /// Abbreviated electron configuration (e.g. "[He]2s2 2p2").
    pub electron_config: String,
    /// Quantum state summary from subatomic_quantum, if computed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantum_state: Option<QuantumAtomSummary>,
}

/// Summary of quantum-mechanical atom state from `subatomic_quantum`.
#[derive(Debug, Clone, Serialize)]
pub struct QuantumAtomSummary {
    pub valence_electrons: u8,
    pub ionization_proxy_ev: f32,
    pub mean_valence_orbital_energy_ev: f32,
    pub mean_effective_nuclear_charge: f32,
    pub mean_orbital_radius_angstrom: f32,
}

// ---------------------------------------------------------------------------
// Bond-level detail
// ---------------------------------------------------------------------------

/// Per-bond render data.
#[derive(Debug, Clone, Serialize)]
pub struct BondVisual {
    pub atom_i: usize,
    pub atom_j: usize,
    /// Bond order label ("single", "double", "triple", "aromatic").
    pub order: String,
    /// Inter-atomic distance in angstroms.
    pub length_angstrom: f32,
    /// Midpoint in angstroms.
    pub midpoint: [f32; 3],
}

// ---------------------------------------------------------------------------
// Molecular-level detail
// ---------------------------------------------------------------------------

/// Summary of the 7 quantum molecular descriptor fields.
#[derive(Debug, Clone, Serialize)]
pub struct QuantumMolecularDescriptorSummary {
    pub ground_state_energy_ev: f32,
    pub ground_state_energy_per_atom_ev: f32,
    pub dipole_magnitude_e_angstrom: f32,
    pub mean_abs_effective_charge: f32,
    pub charge_span: f32,
    pub mean_lda_exchange_potential_ev: f32,
    pub frontier_occupancy_fraction: f32,
}

/// Full molecular view with atom graph, bond network, and quantum descriptors.
#[derive(Debug, Clone, Serialize)]
pub struct MolecularDetail {
    pub name: String,
    pub formula: String,
    pub molecular_weight: f32,
    pub atoms: Vec<AtomVisual>,
    pub bonds: Vec<BondVisual>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantum_descriptor: Option<QuantumMolecularDescriptorSummary>,
    /// Derivation chain: atoms → molecular optics → visual color + rate scaling.
    /// Present when the species has emergent optical properties and/or metabolic rates.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub derivation_chain: Option<DerivationChain>,
    /// Pre-computed backbone ribbon mesh for molecular zoom visualization.
    /// Uses the same parallel-transport algorithm as tree branches.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backbone_mesh: Option<crate::terrarium::ribbon::RibbonMeshData>,
}

/// Traces how atomic properties propagate through the emergent pipeline
/// to produce visual colors and kinetic rates.
#[derive(Debug, Clone, Serialize)]
pub struct DerivationChain {
    /// CPK-weighted molecular color: Σ(n_i × cpk_i) / Σ(n_i)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optical: Option<OpticalDerivation>,
    /// Eyring TST rate derivation for associated metabolic pathway(s).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub rates: Vec<RateDerivation>,
}

/// How atomic composition maps to visual appearance via Beer-Lambert optics.
#[derive(Debug, Clone, Serialize)]
pub struct OpticalDerivation {
    /// Inherent molecular RGB from CPK atom colors.
    pub cpk_rgb: [f32; 3],
    /// Molar extinction coefficient (from quantum descriptor charge_span × eff_charge).
    pub molar_extinction: f32,
    /// Rayleigh scattering cross-section (from mean VDW radius).
    pub scattering_cross_section: f32,
}

/// How bond dissociation energy maps to temperature-dependent reaction rate via Eyring TST.
#[derive(Debug, Clone, Serialize)]
pub struct RateDerivation {
    /// Metabolic pathway name (e.g., "photosynthesis", "fly_glycolysis").
    pub pathway: String,
    /// Bond type (CC or CO).
    pub bond_type: String,
    /// Bond dissociation energy (eV).
    pub bond_energy_ev: f32,
    /// Enzyme catalytic efficiency (0–1).
    pub enzyme_efficiency: f32,
    /// Reference Vmax at 25°C.
    pub vmax_25: f32,
    /// Current temperature-scaled rate (if temperature provided).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_at_current_temp: Option<f32>,
    /// Literature citation for the Vmax.
    pub citation: String,
}

// ---------------------------------------------------------------------------
// Cellular-level detail
// ---------------------------------------------------------------------------

/// Information about a single organelle type within a cell.
#[derive(Debug, Clone, Serialize)]
pub struct OrganelleInfo {
    pub name: String,
    pub count: u32,
    pub function: String,
}

/// Reference to a molecule by name (used in composition lists).
#[derive(Debug, Clone, Serialize)]
pub struct MolecularRef {
    pub name: String,
    pub role: String,
}

/// Gene expression state summary.
#[derive(Debug, Clone, Serialize)]
pub struct GeneInfo {
    pub name: String,
    pub expression_level: f32,
    pub function: String,
}

/// Cell-level inspection data: organelles, membrane, gene expression, metabolism.
#[derive(Debug, Clone, Serialize)]
pub struct CellularDetail {
    pub cell_type: String,
    pub organelles: Vec<OrganelleInfo>,
    pub membrane_composition: Vec<MolecularRef>,
    pub active_genes: Vec<GeneInfo>,
    pub metabolic_state: Vec<InspectMetric>,
}

// ---------------------------------------------------------------------------
// Organism component (tissue/organ) detail
// ---------------------------------------------------------------------------

/// Tissue or organ-level component within an organism.
#[derive(Debug, Clone, Serialize)]
pub struct OrganismComponentDetail {
    pub component_name: String,
    /// "tissue", "organ", "system", "circuit"
    pub component_type: String,
    pub cell_count: u32,
    pub cell_types: Vec<String>,
    pub molecular_inventory: Vec<MolecularRef>,
    pub metrics: Vec<InspectMetric>,
}

// ---------------------------------------------------------------------------
// Top-level scale-aware inspect response
// ---------------------------------------------------------------------------

/// The full scale-aware inspection response. Extends `InspectData` with
/// structured detail at the requested scale level.
#[derive(Debug, Clone, Serialize)]
pub struct ScaleInspectResponse {
    /// Existing inspect payload (preserved for backward compatibility).
    #[serde(flatten)]
    pub base: InspectData,
    /// The resolved scale level.
    pub scale: ScaleLevel,
    /// Tissue/organ components (at organism scale).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub organism_components: Option<Vec<OrganismComponentDetail>>,
    /// Cell-level detail (at cellular scale).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cellular_detail: Option<CellularDetail>,
    /// Molecular graph detail (at molecular scale).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub molecular_detail: Option<MolecularDetail>,
    /// Atom-level detail (at atomic scale).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub atomic_detail: Option<Vec<AtomVisual>>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_level_roundtrip() {
        for level in [
            ScaleLevel::Ecosystem,
            ScaleLevel::Organism,
            ScaleLevel::Cellular,
            ScaleLevel::Molecular,
            ScaleLevel::Atomic,
        ] {
            let s = level.as_str();
            let parsed = ScaleLevel::from_str_opt(s).expect("roundtrip failed");
            assert_eq!(parsed, level, "roundtrip mismatch for {s}");

            // serde roundtrip
            let json = serde_json::to_string(&level).unwrap();
            let deser: ScaleLevel = serde_json::from_str(&json).unwrap();
            assert_eq!(deser, level, "serde roundtrip mismatch for {s}");
        }
    }

    #[test]
    fn test_scale_level_from_str_opt_invalid() {
        assert_eq!(ScaleLevel::from_str_opt("galaxy"), None);
        assert_eq!(ScaleLevel::from_str_opt(""), None);
    }

    #[test]
    fn test_atom_visual_serialization() {
        let atom = AtomVisual {
            element: "Carbon".into(),
            symbol: "C".into(),
            atomic_number: 6,
            position: [0.0, 0.0, 0.0],
            vdw_radius: 1.70,
            cpk_color: [80, 80, 80],
            formal_charge: 0,
            electron_config: "[He]2s2 2p2".into(),
            quantum_state: None,
        };
        let json = serde_json::to_string(&atom).unwrap();
        assert!(json.contains("\"atomic_number\":6"));
        assert!(json.contains("\"vdw_radius\":1.7"));
    }
}
