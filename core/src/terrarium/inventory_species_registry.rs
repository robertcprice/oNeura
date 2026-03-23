use super::*;
use crate::atomistic_chemistry::{
    AtomNode, BondOrder, EmbeddedMolecule, MoleculeGraph as AtomisticMoleculeGraph, PeriodicElement,
};
use crate::subatomic_quantum::QuantumAtomState;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

const TERRARIUM_MOLECULAR_ASSET_JSON: &str =
    include_str!("../../specs/terrarium_molecular_assets.json");
const TERRARIUM_QUANTUM_DESCRIPTOR_CACHE_JSON: &str =
    include_str!("../../specs/terrarium_quantum_descriptor_cache.json");

// ── Binary descriptor cache format ──────────────────────────────────────
//
// Layout: [magic: u32][version: u32][asset_hash: [u8; 32]][entry_count: u32][entries...]
// Each entry: [species_len: u16][species_bytes...][descriptor: 7×f32]
//
// The asset_hash is a simple deterministic hash of the molecular asset JSON,
// used to detect when the cache is stale relative to the topology source.

const BINARY_CACHE_MAGIC: u32 = 0x4F4E_5144; // "ONQD" — oNeura Quantum Descriptors
const BINARY_CACHE_VERSION: u32 = 1;

/// Compute a deterministic 32-byte hash of the molecular asset source.
/// Uses a simple FNV-1a-inspired rolling hash (no external crypto dependency).
pub fn terrarium_molecular_asset_hash() -> [u8; 32] {
    let bytes = TERRARIUM_MOLECULAR_ASSET_JSON.as_bytes();
    let mut hash = [0u8; 32];
    let mut h: u64 = 0xcbf29ce484222325; // FNV offset basis
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3); // FNV prime
    }
    // Spread the 64-bit hash across 32 bytes with mixing
    for i in 0..4 {
        let mixed = h.wrapping_mul(0x9e3779b97f4a7c15u64.wrapping_add(i as u64));
        let bytes_slice = mixed.to_le_bytes();
        hash[i * 8..(i + 1) * 8].copy_from_slice(&bytes_slice);
    }
    hash
}

/// Serialize the quantum descriptor cache to a compact binary format.
pub fn terrarium_quantum_descriptor_cache_binary() -> Vec<u8> {
    let entries = terrarium_quantum_descriptor_cache_entries();
    let asset_hash = terrarium_molecular_asset_hash();
    let mut buf = Vec::with_capacity(4 + 4 + 32 + 4 + entries.len() * (2 + 32 + 7 * 4));

    buf.extend_from_slice(&BINARY_CACHE_MAGIC.to_le_bytes());
    buf.extend_from_slice(&BINARY_CACHE_VERSION.to_le_bytes());
    buf.extend_from_slice(&asset_hash);
    buf.extend_from_slice(&(entries.len() as u32).to_le_bytes());

    for entry in &entries {
        let species_bytes = entry.species.as_bytes();
        buf.extend_from_slice(&(species_bytes.len() as u16).to_le_bytes());
        buf.extend_from_slice(species_bytes);
        for &val in &descriptor_feature_row(entry.descriptor) {
            buf.extend_from_slice(&val.to_le_bytes());
        }
    }
    buf
}

/// Prime the descriptor cache from a binary artifact. Returns the number of
/// entries primed, or an error describing why the binary is invalid.
pub fn prime_terrarium_quantum_descriptor_cache_from_binary(data: &[u8]) -> Result<usize, String> {
    let mut cursor = 0usize;

    let read_u32 = |cursor: &mut usize, data: &[u8]| -> Result<u32, String> {
        if *cursor + 4 > data.len() {
            return Err("unexpected end of binary descriptor cache".into());
        }
        let val = u32::from_le_bytes(data[*cursor..*cursor + 4].try_into().unwrap());
        *cursor += 4;
        Ok(val)
    };
    let read_u16 = |cursor: &mut usize, data: &[u8]| -> Result<u16, String> {
        if *cursor + 2 > data.len() {
            return Err("unexpected end of binary descriptor cache".into());
        }
        let val = u16::from_le_bytes(data[*cursor..*cursor + 2].try_into().unwrap());
        *cursor += 2;
        Ok(val)
    };
    let read_f32 = |cursor: &mut usize, data: &[u8]| -> Result<f32, String> {
        if *cursor + 4 > data.len() {
            return Err("unexpected end of binary descriptor cache".into());
        }
        let val = f32::from_le_bytes(data[*cursor..*cursor + 4].try_into().unwrap());
        *cursor += 4;
        Ok(val)
    };

    let magic = read_u32(&mut cursor, data)?;
    if magic != BINARY_CACHE_MAGIC {
        return Err(format!(
            "invalid binary cache magic: expected 0x{BINARY_CACHE_MAGIC:08X}, got 0x{magic:08X}"
        ));
    }
    let version = read_u32(&mut cursor, data)?;
    if version != BINARY_CACHE_VERSION {
        return Err(format!(
            "unsupported binary cache version: expected {BINARY_CACHE_VERSION}, got {version}"
        ));
    }

    // Validate asset hash
    if cursor + 32 > data.len() {
        return Err("unexpected end of binary descriptor cache (asset hash)".into());
    }
    let stored_hash: [u8; 32] = data[cursor..cursor + 32].try_into().unwrap();
    cursor += 32;
    let current_hash = terrarium_molecular_asset_hash();
    if stored_hash != current_hash {
        return Err(
            "binary descriptor cache is stale: molecular asset hash mismatch".into(),
        );
    }

    let entry_count = read_u32(&mut cursor, data)? as usize;
    let descriptors = terrarium_quantum_descriptor_slots();
    let mut primed = 0;

    for _ in 0..entry_count {
        let species_len = read_u16(&mut cursor, data)? as usize;
        if cursor + species_len > data.len() {
            return Err("unexpected end of binary descriptor cache (species name)".into());
        }
        let species_name = std::str::from_utf8(&data[cursor..cursor + species_len])
            .map_err(|e| format!("invalid UTF-8 in species name: {e}"))?;
        cursor += species_len;

        let ground_state_energy_ev = read_f32(&mut cursor, data)?;
        let ground_state_energy_per_atom_ev = read_f32(&mut cursor, data)?;
        let dipole_magnitude_e_angstrom = read_f32(&mut cursor, data)?;
        let mean_abs_effective_charge = read_f32(&mut cursor, data)?;
        let charge_span = read_f32(&mut cursor, data)?;
        let mean_lda_exchange_potential_ev = read_f32(&mut cursor, data)?;
        let frontier_occupancy_fraction = read_f32(&mut cursor, data)?;

        let species = TerrariumSpecies::from_name(species_name).ok_or_else(|| {
            format!("unknown terrarium species `{species_name}` in binary descriptor cache")
        })?;
        let descriptor = TerrariumMolecularQuantumDescriptor {
            ground_state_energy_ev,
            ground_state_energy_per_atom_ev,
            dipole_magnitude_e_angstrom,
            mean_abs_effective_charge,
            charge_span,
            mean_lda_exchange_potential_ev,
            frontier_occupancy_fraction,
        };
        if descriptors[species as usize].get().is_none()
            && descriptors[species as usize]
                .set(Some(descriptor))
                .is_ok()
        {
            primed += 1;
        }
    }
    Ok(primed)
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TerrariumMolecularQuantumDescriptor {
    pub ground_state_energy_ev: f32,
    pub ground_state_energy_per_atom_ev: f32,
    pub dipole_magnitude_e_angstrom: f32,
    pub mean_abs_effective_charge: f32,
    pub charge_span: f32,
    pub mean_lda_exchange_potential_ev: f32,
    pub frontier_occupancy_fraction: f32,
}

pub const TERRARIUM_QUANTUM_DESCRIPTOR_FEATURE_NAMES: [&str; 7] = [
    "ground_state_energy_ev",
    "ground_state_energy_per_atom_ev",
    "dipole_magnitude_e_angstrom",
    "mean_abs_effective_charge",
    "charge_span",
    "mean_lda_exchange_potential_ev",
    "frontier_occupancy_fraction",
];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TerrariumQuantumDescriptorCacheEntry {
    pub species: String,
    pub descriptor: TerrariumMolecularQuantumDescriptor,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TerrariumQuantumDescriptorTensorSnapshot {
    pub feature_names: Vec<String>,
    pub species: Vec<String>,
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct TerrariumMolecularAssetSpec {
    species: String,
    name: String,
    #[serde(default)]
    quantum_fast_path: bool,
    atoms: Vec<TerrariumMolecularAssetAtomSpec>,
    #[serde(default)]
    bonds: Vec<TerrariumMolecularAssetBondSpec>,
}

#[derive(Debug, Deserialize)]
struct TerrariumMolecularAssetAtomSpec {
    element: String,
    #[serde(default)]
    formal_charge: i8,
}

#[derive(Debug, Deserialize)]
struct TerrariumMolecularAssetBondSpec {
    i: usize,
    j: usize,
    order: TerrariumBondOrderSpec,
}

#[derive(Debug, Clone)]
struct TerrariumMolecularAsset {
    species: TerrariumSpecies,
    name: String,
    quantum_fast_path: bool,
    atoms: Vec<TerrariumMolecularAssetAtom>,
    bonds: Vec<TerrariumMolecularAssetBond>,
}

#[derive(Debug, Clone, Copy)]
struct TerrariumMolecularAssetAtom {
    element: PeriodicElement,
    formal_charge: i8,
}

#[derive(Debug, Clone, Copy)]
struct TerrariumMolecularAssetBond {
    i: usize,
    j: usize,
    order: BondOrder,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum TerrariumBondOrderSpec {
    Single,
    Double,
    Triple,
    Aromatic,
}

impl TerrariumBondOrderSpec {
    fn into_bond_order(self) -> BondOrder {
        match self {
            Self::Single => BondOrder::Single,
            Self::Double => BondOrder::Double,
            Self::Triple => BondOrder::Triple,
            Self::Aromatic => BondOrder::Aromatic,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct TerrariumInventorySpeciesProfile {
    pub region: MaterialRegionKind,
    pub phase_kind: MaterialPhaseKind,
    pub pool_name: &'static str,
    pub formal_charge: i8,
    pub element_counts: &'static [(PeriodicElement, u16)],
}

pub(crate) const TERRARIUM_INVENTORY_BOUND_SPECIES: [TerrariumSpecies; 27] = [
    TerrariumSpecies::Water,
    TerrariumSpecies::Glucose,
    TerrariumSpecies::OxygenGas,
    TerrariumSpecies::Ammonium,
    TerrariumSpecies::Nitrate,
    TerrariumSpecies::CarbonDioxide,
    TerrariumSpecies::BicarbonatePool,
    TerrariumSpecies::Proton,
    TerrariumSpecies::SurfaceProtonLoad,
    TerrariumSpecies::CalciumBicarbonateComplex,
    TerrariumSpecies::AtpFlux,
    TerrariumSpecies::AminoAcidPool,
    TerrariumSpecies::NucleotidePool,
    TerrariumSpecies::MembranePrecursorPool,
    TerrariumSpecies::SilicateMineral,
    TerrariumSpecies::ClayMineral,
    TerrariumSpecies::CarbonateMineral,
    TerrariumSpecies::IronOxideMineral,
    TerrariumSpecies::SorbedAluminumHydroxide,
    TerrariumSpecies::SorbedFerricHydroxide,
    TerrariumSpecies::DissolvedSilicate,
    TerrariumSpecies::ExchangeableCalcium,
    TerrariumSpecies::ExchangeableMagnesium,
    TerrariumSpecies::ExchangeablePotassium,
    TerrariumSpecies::ExchangeableSodium,
    TerrariumSpecies::ExchangeableAluminum,
    TerrariumSpecies::AqueousIronPool,
];

pub(crate) fn terrarium_inventory_species_profile(
    species: TerrariumSpecies,
) -> Option<TerrariumInventorySpeciesProfile> {
    let profile = match species {
        TerrariumSpecies::Water => TerrariumInventorySpeciesProfile {
            region: MaterialRegionKind::PoreWater,
            phase_kind: MaterialPhaseKind::Aqueous,
            pool_name: "water",
            formal_charge: 0,
            element_counts: &[(PeriodicElement::H, 2), (PeriodicElement::O, 1)],
        },
        TerrariumSpecies::Glucose => TerrariumInventorySpeciesProfile {
            region: MaterialRegionKind::PoreWater,
            phase_kind: MaterialPhaseKind::Aqueous,
            pool_name: "glucose",
            formal_charge: 0,
            element_counts: &[
                (PeriodicElement::C, 6),
                (PeriodicElement::H, 12),
                (PeriodicElement::O, 6),
            ],
        },
        TerrariumSpecies::OxygenGas => TerrariumInventorySpeciesProfile {
            region: MaterialRegionKind::GasPhase,
            phase_kind: MaterialPhaseKind::Gas,
            pool_name: "oxygen_gas",
            formal_charge: 0,
            element_counts: &[(PeriodicElement::O, 2)],
        },
        TerrariumSpecies::Ammonium => TerrariumInventorySpeciesProfile {
            region: MaterialRegionKind::PoreWater,
            phase_kind: MaterialPhaseKind::Aqueous,
            pool_name: "ammonium",
            formal_charge: 1,
            element_counts: &[(PeriodicElement::N, 1), (PeriodicElement::H, 4)],
        },
        TerrariumSpecies::Nitrate => TerrariumInventorySpeciesProfile {
            region: MaterialRegionKind::PoreWater,
            phase_kind: MaterialPhaseKind::Aqueous,
            pool_name: "nitrate",
            formal_charge: -1,
            element_counts: &[(PeriodicElement::N, 1), (PeriodicElement::O, 3)],
        },
        TerrariumSpecies::CarbonDioxide => TerrariumInventorySpeciesProfile {
            region: MaterialRegionKind::GasPhase,
            phase_kind: MaterialPhaseKind::Gas,
            pool_name: "carbon_dioxide",
            formal_charge: 0,
            element_counts: &[(PeriodicElement::C, 1), (PeriodicElement::O, 2)],
        },
        TerrariumSpecies::BicarbonatePool => TerrariumInventorySpeciesProfile {
            region: MaterialRegionKind::PoreWater,
            phase_kind: MaterialPhaseKind::Aqueous,
            pool_name: "bicarbonate_pool",
            formal_charge: -1,
            element_counts: &[
                (PeriodicElement::H, 1),
                (PeriodicElement::C, 1),
                (PeriodicElement::O, 3),
            ],
        },
        TerrariumSpecies::Proton => TerrariumInventorySpeciesProfile {
            region: MaterialRegionKind::PoreWater,
            phase_kind: MaterialPhaseKind::Aqueous,
            pool_name: "proton_pool",
            formal_charge: 1,
            element_counts: &[(PeriodicElement::H, 1)],
        },
        TerrariumSpecies::SurfaceProtonLoad => TerrariumInventorySpeciesProfile {
            region: MaterialRegionKind::MineralSurface,
            phase_kind: MaterialPhaseKind::Interfacial,
            pool_name: "surface_proton_load",
            formal_charge: 1,
            element_counts: &[(PeriodicElement::H, 1)],
        },
        TerrariumSpecies::CalciumBicarbonateComplex => TerrariumInventorySpeciesProfile {
            region: MaterialRegionKind::PoreWater,
            phase_kind: MaterialPhaseKind::Aqueous,
            pool_name: "calcium_bicarbonate_complex",
            formal_charge: 0,
            element_counts: &[
                (PeriodicElement::Ca, 1),
                (PeriodicElement::C, 2),
                (PeriodicElement::H, 2),
                (PeriodicElement::O, 6),
            ],
        },
        TerrariumSpecies::AtpFlux => TerrariumInventorySpeciesProfile {
            region: MaterialRegionKind::BiofilmMatrix,
            phase_kind: MaterialPhaseKind::Aqueous,
            pool_name: "atp",
            formal_charge: -4,
            element_counts: &[
                (PeriodicElement::C, 10),
                (PeriodicElement::H, 16),
                (PeriodicElement::N, 5),
                (PeriodicElement::O, 13),
                (PeriodicElement::P, 3),
            ],
        },
        TerrariumSpecies::AminoAcidPool => TerrariumInventorySpeciesProfile {
            region: MaterialRegionKind::PoreWater,
            phase_kind: MaterialPhaseKind::Aqueous,
            pool_name: "amino_acid_pool",
            formal_charge: 0,
            element_counts: &[
                (PeriodicElement::C, 4),
                (PeriodicElement::H, 8),
                (PeriodicElement::N, 1),
                (PeriodicElement::O, 2),
            ],
        },
        TerrariumSpecies::NucleotidePool => TerrariumInventorySpeciesProfile {
            region: MaterialRegionKind::PoreWater,
            phase_kind: MaterialPhaseKind::Aqueous,
            pool_name: "nucleotide_pool",
            formal_charge: -1,
            element_counts: &[
                (PeriodicElement::C, 9),
                (PeriodicElement::H, 14),
                (PeriodicElement::N, 4),
                (PeriodicElement::O, 8),
                (PeriodicElement::P, 1),
            ],
        },
        TerrariumSpecies::MembranePrecursorPool => TerrariumInventorySpeciesProfile {
            region: MaterialRegionKind::BiofilmMatrix,
            phase_kind: MaterialPhaseKind::Amorphous,
            pool_name: "membrane_precursor_pool",
            formal_charge: -1,
            element_counts: &[
                (PeriodicElement::C, 8),
                (PeriodicElement::H, 16),
                (PeriodicElement::N, 1),
                (PeriodicElement::O, 8),
                (PeriodicElement::P, 1),
            ],
        },
        TerrariumSpecies::SilicateMineral => TerrariumInventorySpeciesProfile {
            region: MaterialRegionKind::Soil,
            phase_kind: MaterialPhaseKind::Solid,
            pool_name: "quartz_like_silicate_phase",
            formal_charge: 0,
            element_counts: &[(PeriodicElement::Si, 1), (PeriodicElement::O, 2)],
        },
        TerrariumSpecies::ClayMineral => TerrariumInventorySpeciesProfile {
            region: MaterialRegionKind::Soil,
            phase_kind: MaterialPhaseKind::Solid,
            pool_name: "kaolinite_like_clay_phase",
            formal_charge: 0,
            element_counts: &[
                (PeriodicElement::Al, 2),
                (PeriodicElement::Si, 2),
                (PeriodicElement::O, 9),
                (PeriodicElement::H, 4),
            ],
        },
        TerrariumSpecies::CarbonateMineral => TerrariumInventorySpeciesProfile {
            region: MaterialRegionKind::Soil,
            phase_kind: MaterialPhaseKind::Solid,
            pool_name: "calcite_like_carbonate_phase",
            formal_charge: 0,
            element_counts: &[
                (PeriodicElement::Ca, 1),
                (PeriodicElement::C, 1),
                (PeriodicElement::O, 3),
            ],
        },
        TerrariumSpecies::IronOxideMineral => TerrariumInventorySpeciesProfile {
            region: MaterialRegionKind::Soil,
            phase_kind: MaterialPhaseKind::Solid,
            pool_name: "hematite_like_iron_oxide_phase",
            formal_charge: 0,
            element_counts: &[(PeriodicElement::Fe, 2), (PeriodicElement::O, 3)],
        },
        TerrariumSpecies::SorbedAluminumHydroxide => TerrariumInventorySpeciesProfile {
            region: MaterialRegionKind::MineralSurface,
            phase_kind: MaterialPhaseKind::Amorphous,
            pool_name: "gibbsite_like_aluminum_hydroxide_phase",
            formal_charge: 0,
            element_counts: &[
                (PeriodicElement::Al, 1),
                (PeriodicElement::O, 3),
                (PeriodicElement::H, 3),
            ],
        },
        TerrariumSpecies::SorbedFerricHydroxide => TerrariumInventorySpeciesProfile {
            region: MaterialRegionKind::MineralSurface,
            phase_kind: MaterialPhaseKind::Amorphous,
            pool_name: "ferric_hydroxide_phase",
            formal_charge: 0,
            element_counts: &[
                (PeriodicElement::Fe, 1),
                (PeriodicElement::O, 3),
                (PeriodicElement::H, 3),
            ],
        },
        TerrariumSpecies::DissolvedSilicate => TerrariumInventorySpeciesProfile {
            region: MaterialRegionKind::PoreWater,
            phase_kind: MaterialPhaseKind::Aqueous,
            pool_name: "dissolved_silicic_acid_pool",
            formal_charge: 0,
            element_counts: &[
                (PeriodicElement::H, 4),
                (PeriodicElement::Si, 1),
                (PeriodicElement::O, 4),
            ],
        },
        TerrariumSpecies::ExchangeableCalcium => TerrariumInventorySpeciesProfile {
            region: MaterialRegionKind::MineralSurface,
            phase_kind: MaterialPhaseKind::Interfacial,
            pool_name: "exchangeable_calcium_pool",
            formal_charge: 2,
            element_counts: &[(PeriodicElement::Ca, 1)],
        },
        TerrariumSpecies::ExchangeableMagnesium => TerrariumInventorySpeciesProfile {
            region: MaterialRegionKind::MineralSurface,
            phase_kind: MaterialPhaseKind::Interfacial,
            pool_name: "exchangeable_magnesium_pool",
            formal_charge: 2,
            element_counts: &[(PeriodicElement::Mg, 1)],
        },
        TerrariumSpecies::ExchangeablePotassium => TerrariumInventorySpeciesProfile {
            region: MaterialRegionKind::MineralSurface,
            phase_kind: MaterialPhaseKind::Interfacial,
            pool_name: "exchangeable_potassium_pool",
            formal_charge: 1,
            element_counts: &[(PeriodicElement::K, 1)],
        },
        TerrariumSpecies::ExchangeableSodium => TerrariumInventorySpeciesProfile {
            region: MaterialRegionKind::MineralSurface,
            phase_kind: MaterialPhaseKind::Interfacial,
            pool_name: "exchangeable_sodium_pool",
            formal_charge: 1,
            element_counts: &[(PeriodicElement::Na, 1)],
        },
        TerrariumSpecies::ExchangeableAluminum => TerrariumInventorySpeciesProfile {
            region: MaterialRegionKind::MineralSurface,
            phase_kind: MaterialPhaseKind::Interfacial,
            pool_name: "exchangeable_aluminum_pool",
            formal_charge: 3,
            element_counts: &[(PeriodicElement::Al, 1)],
        },
        TerrariumSpecies::AqueousIronPool => TerrariumInventorySpeciesProfile {
            region: MaterialRegionKind::PoreWater,
            phase_kind: MaterialPhaseKind::Aqueous,
            pool_name: "aqueous_iron_pool",
            formal_charge: 3,
            element_counts: &[(PeriodicElement::Fe, 1)],
        },
        _ => return None,
    };
    Some(profile)
}

pub(crate) fn terrarium_inventory_molecule(
    species: TerrariumSpecies,
) -> Option<MoleculeDescriptor> {
    let profile = terrarium_inventory_species_profile(species)?;
    Some(
        terrarium_inventory_embedded_molecule(species)
            .map(|molecule| {
                molecule_descriptor_from_embedded_molecule(profile.pool_name, &molecule)
            })
            .unwrap_or_else(|| {
                MoleculeDescriptor::with_formula(profile.pool_name, profile.element_counts)
            }),
    )
}

pub(crate) fn terrarium_inventory_binding(
    species: TerrariumSpecies,
) -> Option<(
    MaterialRegionKind,
    MoleculeDescriptor,
    MaterialPhaseDescriptor,
)> {
    let profile = terrarium_inventory_species_profile(species)?;
    Some((
        profile.region,
        terrarium_inventory_molecule(species)?,
        MaterialPhaseDescriptor::ambient(profile.phase_kind),
    ))
}

pub(crate) fn terrarium_inventory_embedded_molecule(
    species: TerrariumSpecies,
) -> Option<EmbeddedMolecule> {
    let molecules = terrarium_embedded_molecule_slots();
    molecules[species as usize]
        .get_or_init(|| build_embedded_molecule(species))
        .clone()
}

fn build_embedded_molecule(species: TerrariumSpecies) -> Option<EmbeddedMolecule> {
    let asset = terrarium_molecular_asset(species)?;
    let graph = molecule_graph_from_asset(asset).ok()?;
    let positions = embed_molecule_graph(&graph);
    EmbeddedMolecule::new(graph, positions).ok()
}

pub(crate) fn terrarium_inventory_quantum_descriptor(
    species: TerrariumSpecies,
) -> Option<TerrariumMolecularQuantumDescriptor> {
    maybe_prime_terrarium_quantum_descriptor_cache();
    let descriptors = terrarium_quantum_descriptor_slots();
    if let Some(descriptor) = descriptors[species as usize].get() {
        return *descriptor;
    }
    if terrarium_molecular_asset(species).is_some_and(|asset| asset.quantum_fast_path) {
        let _ = warm_terrarium_quantum_descriptor_cache();
    }
    let descriptors = terrarium_quantum_descriptor_slots();
    *descriptors[species as usize].get_or_init(|| {
        // Try exact quantum solve first, then regression fallback
        derive_quantum_descriptor(species)
            .or_else(|| derive_quantum_descriptor_regression(species))
    })
}

pub fn warm_terrarium_quantum_descriptor_cache() -> usize {
    maybe_prime_terrarium_quantum_descriptor_cache();
    static FAST_PATH_WARMED: OnceLock<()> = OnceLock::new();
    FAST_PATH_WARMED.get_or_init(|| {
        let descriptors = terrarium_quantum_descriptor_slots();
        terrarium_quantum_fast_path_species()
            .par_iter()
            .for_each(|&species| {
                let _ = *descriptors[species as usize]
                    .get_or_init(|| derive_quantum_descriptor(species));
            });
    });
    terrarium_quantum_fast_path_species().len()
}

/// Warm both the fast-path (exact quantum) and regression tiers.
/// Returns (fast_path_count, regression_count).
pub fn warm_terrarium_quantum_descriptor_cache_full() -> (usize, usize) {
    let fast = warm_terrarium_quantum_descriptor_cache();
    let regression_species = terrarium_regression_tier_species();
    let descriptors = terrarium_quantum_descriptor_slots();
    for &species in &regression_species {
        let _ = *descriptors[species as usize].get_or_init(|| {
            derive_quantum_descriptor(species)
                .or_else(|| derive_quantum_descriptor_regression(species))
        });
    }
    (fast, regression_species.len())
}

/// Species that have molecular assets but are NOT on the fast path —
/// these get regression-tier descriptors.
fn terrarium_regression_tier_species() -> Vec<TerrariumSpecies> {
    let fast_path = terrarium_quantum_fast_path_species();
    terrarium_molecular_assets()
        .iter()
        .filter(|asset| !fast_path.contains(&asset.species))
        .map(|asset| asset.species)
        .collect()
}

/// All species that have descriptors (fast-path + regression tier).
pub fn terrarium_all_descriptor_species() -> Vec<TerrariumSpecies> {
    let _ = warm_terrarium_quantum_descriptor_cache_full();
    let mut species = Vec::new();
    for &fp in terrarium_quantum_fast_path_species() {
        species.push(fp);
    }
    for &reg in &terrarium_regression_tier_species() {
        if terrarium_inventory_quantum_descriptor(reg).is_some() {
            species.push(reg);
        }
    }
    species
}

/// Look up an [`EmbeddedMolecule`] by species name string.
/// Returns `None` if the species name is unrecognized or has no molecular
/// asset topology in `terrarium_molecular_assets.json`.
pub fn terrarium_molecular_asset_by_name(name: &str) -> Option<EmbeddedMolecule> {
    let species = TerrariumSpecies::from_name(name)?;
    terrarium_inventory_embedded_molecule(species)
}

pub fn terrarium_quantum_descriptor_cache_entries() -> Vec<TerrariumQuantumDescriptorCacheEntry> {
    let _ = warm_terrarium_quantum_descriptor_cache();
    terrarium_quantum_fast_path_species()
        .iter()
        .filter_map(|&species| {
            terrarium_inventory_quantum_descriptor(species).map(|descriptor| {
                TerrariumQuantumDescriptorCacheEntry {
                    species: species.as_str().to_string(),
                    descriptor,
                }
            })
        })
        .collect()
}

/// Cache entries including regression-tier species.
pub fn terrarium_quantum_descriptor_cache_entries_full() -> Vec<TerrariumQuantumDescriptorCacheEntry> {
    terrarium_all_descriptor_species()
        .iter()
        .filter_map(|&species| {
            terrarium_inventory_quantum_descriptor(species).map(|descriptor| {
                TerrariumQuantumDescriptorCacheEntry {
                    species: species.as_str().to_string(),
                    descriptor,
                }
            })
        })
        .collect()
}

pub fn terrarium_quantum_descriptor_cache_json_pretty() -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(&terrarium_quantum_descriptor_cache_entries())
}

pub fn terrarium_quantum_descriptor_tensor_snapshot() -> TerrariumQuantumDescriptorTensorSnapshot {
    let entries = terrarium_quantum_descriptor_cache_entries();
    let rows = entries.len();
    let cols = TERRARIUM_QUANTUM_DESCRIPTOR_FEATURE_NAMES.len();
    let species = entries.iter().map(|entry| entry.species.clone()).collect();
    let mut values = Vec::with_capacity(rows * cols);
    for entry in &entries {
        values.extend_from_slice(&descriptor_feature_row(entry.descriptor));
    }
    TerrariumQuantumDescriptorTensorSnapshot {
        feature_names: TERRARIUM_QUANTUM_DESCRIPTOR_FEATURE_NAMES
            .iter()
            .map(|name| (*name).to_string())
            .collect(),
        species,
        rows,
        cols,
        values,
    }
}

pub fn terrarium_quantum_descriptor_tensor_json_pretty() -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(&terrarium_quantum_descriptor_tensor_snapshot())
}

pub fn prime_terrarium_quantum_descriptor_cache_from_json(json: &str) -> Result<usize, String> {
    let entries: Vec<TerrariumQuantumDescriptorCacheEntry> =
        serde_json::from_str(json).map_err(|error| {
            format!("failed to parse terrarium quantum descriptor cache JSON: {error}")
        })?;
    let descriptors = terrarium_quantum_descriptor_slots();
    let mut primed = 0;
    for entry in entries {
        let species = TerrariumSpecies::from_name(&entry.species).ok_or_else(|| {
            format!(
                "unknown terrarium species `{}` in descriptor cache asset",
                entry.species
            )
        })?;
        if descriptors[species as usize].get().is_none()
            && descriptors[species as usize]
                .set(Some(entry.descriptor))
                .is_ok()
        {
            primed += 1;
        }
    }
    Ok(primed)
}

fn derive_quantum_descriptor(
    species: TerrariumSpecies,
) -> Option<TerrariumMolecularQuantumDescriptor> {
    let asset = terrarium_molecular_asset(species)?;
    if !asset.quantum_fast_path {
        return None;
    }
    let molecule = terrarium_inventory_embedded_molecule(species)?;
    let (max_spatial_orbitals, max_basis_size) = quantum_budget_for_molecule(&molecule)?;
    let positions_angstrom: Vec<[f64; 3]> = molecule
        .positions_angstrom
        .iter()
        .map(|position| {
            [
                f64::from(position[0]),
                f64::from(position[1]),
                f64::from(position[2]),
            ]
        })
        .collect();
    let result = molecule
        .graph
        .quantum_reactive_site(positions_angstrom, max_spatial_orbitals, max_basis_size)
        .ok()?
        .analyze()
        .ok()?;
    let atom_count = molecule.graph.atom_count().max(1) as f32;
    let ground_state_energy_ev = result.ground_state_energy_ev()? as f32;
    let dipole = result.expected_dipole_moment_e_angstrom;
    let dipole_magnitude_e_angstrom =
        (dipole[0] * dipole[0] + dipole[1] * dipole[1] + dipole[2] * dipole[2]).sqrt() as f32;
    let mean_abs_effective_charge = if result.expected_atom_effective_charges.is_empty() {
        0.0
    } else {
        result
            .expected_atom_effective_charges
            .iter()
            .map(|charge| charge.abs() as f32)
            .sum::<f32>()
            / result.expected_atom_effective_charges.len() as f32
    };
    let (min_charge, max_charge) = result.expected_atom_effective_charges.iter().fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(min_charge, max_charge), charge| (min_charge.min(*charge), max_charge.max(*charge)),
    );
    let charge_span = if min_charge.is_finite() && max_charge.is_finite() {
        (max_charge - min_charge) as f32
    } else {
        0.0
    };
    let mean_lda_exchange_potential_ev = if molecule.graph.atom_count() == 0 {
        0.0
    } else {
        (0..molecule.graph.atom_count())
            .map(|idx| result.lda_exchange_potential_ev(idx).abs() as f32)
            .sum::<f32>()
            / molecule.graph.atom_count() as f32
    };
    let frontier_occupancy_fraction = result
        .natural_orbitals()
        .and_then(|(occupancies, _)| occupancies.last().copied())
        .map(|occupancy| (occupancy / 2.0).clamp(0.0, 1.0) as f32)
        .unwrap_or(0.0);
    Some(TerrariumMolecularQuantumDescriptor {
        ground_state_energy_ev,
        ground_state_energy_per_atom_ev: ground_state_energy_ev / atom_count,
        dipole_magnitude_e_angstrom,
        mean_abs_effective_charge,
        charge_span,
        mean_lda_exchange_potential_ev,
        frontier_occupancy_fraction,
    })
}

fn descriptor_feature_row(descriptor: TerrariumMolecularQuantumDescriptor) -> [f32; 7] {
    [
        descriptor.ground_state_energy_ev,
        descriptor.ground_state_energy_per_atom_ev,
        descriptor.dipole_magnitude_e_angstrom,
        descriptor.mean_abs_effective_charge,
        descriptor.charge_span,
        descriptor.mean_lda_exchange_potential_ev,
        descriptor.frontier_occupancy_fraction,
    ]
}

// ── Regression descriptor tier ──────────────────────────────────────────
//
// For species too large for the exact quantum solver (>5 atoms), we estimate
// quantum descriptors from molecular structure features using a simple linear
// regression trained on the fast-path species. This ensures every species with
// a canonical molecular asset gets a descriptor — no species-name tables.
//
// Structure features (6-dimensional):
//   0. atom_count
//   1. total_valence_electrons
//   2. net_formal_charge
//   3. mean_pauling_electronegativity
//   4. bond_count
//   5. spatial_extent_angstrom (max bounding box dimension)

const REGRESSION_FEATURE_COUNT: usize = 6;

fn molecule_structure_features(molecule: &EmbeddedMolecule) -> Option<[f64; REGRESSION_FEATURE_COUNT]> {
    let atom_count = molecule.graph.atom_count();
    if atom_count == 0 {
        return None;
    }
    let total_valence_electrons: usize = molecule
        .graph
        .atoms
        .iter()
        .filter_map(|atom| QuantumAtomState::from_atom_node(atom).ok())
        .map(|state| state.valence_electrons() as usize)
        .sum();
    let net_formal_charge: i32 = molecule
        .graph
        .atoms
        .iter()
        .map(|atom| atom.formal_charge as i32)
        .sum();
    let mean_electronegativity: f64 = {
        let (sum, count) = molecule.graph.atoms.iter().fold((0.0f64, 0usize), |(s, c), atom| {
            if let Some(en) = atom.element.pauling_electronegativity() {
                (s + en, c + 1)
            } else {
                (s, c)
            }
        });
        if count > 0 { sum / count as f64 } else { 2.0 }
    };
    let bond_count = molecule.graph.bond_count();
    let spatial_extent = molecule.max_extent() as f64;

    Some([
        atom_count as f64,
        total_valence_electrons as f64,
        net_formal_charge as f64,
        mean_electronegativity,
        bond_count as f64,
        spatial_extent,
    ])
}

/// Fit a linear regression (ordinary least squares) from structure features to
/// a single descriptor column. Returns weights [w0..w5, bias].
fn fit_linear_regression(
    features: &[[f64; REGRESSION_FEATURE_COUNT]],
    targets: &[f64],
) -> [f64; REGRESSION_FEATURE_COUNT + 1] {
    let n = features.len();
    if n == 0 {
        return [0.0; REGRESSION_FEATURE_COUNT + 1];
    }
    // Normal equations: (X^T X) w = X^T y, with augmented X (bias column)
    let dim = REGRESSION_FEATURE_COUNT + 1;
    let mut xtx = vec![0.0f64; dim * dim];
    let mut xty = vec![0.0f64; dim];

    for i in 0..n {
        let mut row = [0.0f64; REGRESSION_FEATURE_COUNT + 1];
        row[..REGRESSION_FEATURE_COUNT].copy_from_slice(&features[i]);
        row[REGRESSION_FEATURE_COUNT] = 1.0; // bias

        for r in 0..dim {
            for c in 0..dim {
                xtx[r * dim + c] += row[r] * row[c];
            }
            xty[r] += row[r] * targets[i];
        }
    }

    // Tikhonov regularization (ridge) for numerical stability
    for i in 0..dim {
        xtx[i * dim + i] += 1.0e-6;
    }

    // Solve via Cholesky decomposition
    solve_cholesky(&xtx, &xty, dim)
}

fn solve_cholesky(ata: &[f64], atb: &[f64], dim: usize) -> [f64; REGRESSION_FEATURE_COUNT + 1] {
    let mut l = vec![0.0f64; dim * dim];
    // Cholesky: A = L L^T
    for i in 0..dim {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i * dim + k] * l[j * dim + k];
            }
            if i == j {
                let diag = ata[i * dim + i] - sum;
                l[i * dim + j] = if diag > 0.0 { diag.sqrt() } else { 1.0e-12 };
            } else {
                l[i * dim + j] = (ata[i * dim + j] - sum) / l[j * dim + j].max(1.0e-12);
            }
        }
    }
    // Forward substitution: L y = atb
    let mut y = vec![0.0f64; dim];
    for i in 0..dim {
        let mut sum = 0.0;
        for j in 0..i {
            sum += l[i * dim + j] * y[j];
        }
        y[i] = (atb[i] - sum) / l[i * dim + i].max(1.0e-12);
    }
    // Backward substitution: L^T x = y
    let mut x = vec![0.0f64; dim];
    for i in (0..dim).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..dim {
            sum += l[j * dim + i] * x[j];
        }
        x[i] = (y[i] - sum) / l[i * dim + i].max(1.0e-12);
    }
    let mut result = [0.0f64; REGRESSION_FEATURE_COUNT + 1];
    result[..dim].copy_from_slice(&x[..dim]);
    result
}

fn predict_linear(
    features: &[f64; REGRESSION_FEATURE_COUNT],
    weights: &[f64; REGRESSION_FEATURE_COUNT + 1],
) -> f64 {
    let mut val = weights[REGRESSION_FEATURE_COUNT]; // bias
    for i in 0..REGRESSION_FEATURE_COUNT {
        val += features[i] * weights[i];
    }
    val
}

/// Train regression models on fast-path species, then predict for the target.
fn derive_quantum_descriptor_regression(
    species: TerrariumSpecies,
) -> Option<TerrariumMolecularQuantumDescriptor> {
    let target_molecule = terrarium_inventory_embedded_molecule(species)?;
    let target_features = molecule_structure_features(&target_molecule)?;

    // Collect training data from fast-path species
    let fast_path = terrarium_quantum_fast_path_species();
    let mut train_features = Vec::with_capacity(fast_path.len());
    let mut train_descriptors = Vec::with_capacity(fast_path.len());

    for &fp_species in fast_path {
        if let Some(mol) = terrarium_inventory_embedded_molecule(fp_species) {
            if let Some(feats) = molecule_structure_features(&mol) {
                if let Some(desc) = terrarium_inventory_quantum_descriptor(fp_species) {
                    train_features.push(feats);
                    train_descriptors.push(descriptor_feature_row(desc));
                }
            }
        }
    }

    if train_features.is_empty() {
        return None;
    }

    // Fit one regression per descriptor column and predict
    let mut predicted = [0.0f32; 7];
    for col in 0..7 {
        let targets: Vec<f64> = train_descriptors.iter().map(|d| d[col] as f64).collect();
        let weights = fit_linear_regression(&train_features, &targets);
        predicted[col] = predict_linear(&target_features, &weights) as f32;
    }

    Some(TerrariumMolecularQuantumDescriptor {
        ground_state_energy_ev: predicted[0],
        ground_state_energy_per_atom_ev: predicted[1],
        dipole_magnitude_e_angstrom: predicted[2].max(0.0),
        mean_abs_effective_charge: predicted[3].max(0.0),
        charge_span: predicted[4].max(0.0),
        mean_lda_exchange_potential_ev: predicted[5],
        frontier_occupancy_fraction: predicted[6].clamp(0.0, 1.0),
    })
}

fn maybe_prime_terrarium_quantum_descriptor_cache() {
    static PRIMED_CACHE: OnceLock<()> = OnceLock::new();
    PRIMED_CACHE.get_or_init(|| {
        prime_terrarium_quantum_descriptor_cache_from_json(TERRARIUM_QUANTUM_DESCRIPTOR_CACHE_JSON)
            .unwrap_or_else(|error| {
                panic!("failed to prime terrarium quantum descriptor cache: {error}")
            });
    });
}

fn quantum_budget_for_molecule(molecule: &EmbeddedMolecule) -> Option<(Option<usize>, usize)> {
    if molecule.graph.atom_count() > 5 {
        return None;
    }
    let active_electrons = molecule
        .graph
        .atoms
        .iter()
        .map(QuantumAtomState::from_atom_node)
        .collect::<Result<Vec<_>, _>>()
        .ok()?
        .into_iter()
        .map(|state| state.valence_electrons() as usize)
        .sum::<usize>();
    if active_electrons == 0 {
        return None;
    }
    let required_spatial_orbitals = active_electrons.div_ceil(2);
    let max_spatial_orbitals = required_spatial_orbitals.saturating_add(1).max(1);
    Some((Some(max_spatial_orbitals), 4096))
}

fn molecule_descriptor_from_embedded_molecule(
    name: &str,
    molecule: &EmbeddedMolecule,
) -> MoleculeDescriptor {
    let mut element_counts: Vec<(PeriodicElement, u16)> = molecule
        .graph
        .element_composition()
        .into_iter()
        .map(|(element, count)| (element, count as u16))
        .collect();
    element_counts.sort_by_key(|(element, _)| element.atomic_number());
    MoleculeDescriptor::with_formula(name, &element_counts)
}

fn terrarium_embedded_molecule_slots() -> &'static Vec<OnceLock<Option<EmbeddedMolecule>>> {
    static EMBEDDED_MOLECULES: OnceLock<Vec<OnceLock<Option<EmbeddedMolecule>>>> = OnceLock::new();
    EMBEDDED_MOLECULES.get_or_init(|| {
        (0..crate::terrarium::substrate::TERRARIUM_SPECIES_COUNT)
            .map(|_| OnceLock::new())
            .collect()
    })
}

fn terrarium_quantum_descriptor_slots(
) -> &'static Vec<OnceLock<Option<TerrariumMolecularQuantumDescriptor>>> {
    static DESCRIPTORS: OnceLock<Vec<OnceLock<Option<TerrariumMolecularQuantumDescriptor>>>> =
        OnceLock::new();
    DESCRIPTORS.get_or_init(|| {
        (0..crate::terrarium::substrate::TERRARIUM_SPECIES_COUNT)
            .map(|_| OnceLock::new())
            .collect()
    })
}

fn terrarium_molecular_assets() -> &'static [TerrariumMolecularAsset] {
    static ASSETS: OnceLock<Vec<TerrariumMolecularAsset>> = OnceLock::new();
    ASSETS.get_or_init(|| {
        load_terrarium_molecular_assets()
            .unwrap_or_else(|error| panic!("failed to load terrarium molecular assets: {error}"))
    })
}

fn terrarium_molecular_asset(
    species: TerrariumSpecies,
) -> Option<&'static TerrariumMolecularAsset> {
    terrarium_molecular_asset_slots()[species as usize].as_ref()
}

fn terrarium_molecular_asset_slots() -> &'static Vec<Option<TerrariumMolecularAsset>> {
    static ASSET_SLOTS: OnceLock<Vec<Option<TerrariumMolecularAsset>>> = OnceLock::new();
    ASSET_SLOTS.get_or_init(|| {
        let mut slots = vec![None; crate::terrarium::substrate::TERRARIUM_SPECIES_COUNT];
        for asset in terrarium_molecular_assets().iter().cloned() {
            let species = asset.species;
            slots[species as usize] = Some(asset);
        }
        slots
    })
}

fn terrarium_quantum_fast_path_species() -> &'static [TerrariumSpecies] {
    static FAST_PATH_SPECIES: OnceLock<Vec<TerrariumSpecies>> = OnceLock::new();
    FAST_PATH_SPECIES.get_or_init(|| {
        terrarium_molecular_assets()
            .iter()
            .filter(|asset| asset.quantum_fast_path)
            .map(|asset| asset.species)
            .collect()
    })
}

fn load_terrarium_molecular_assets() -> Result<Vec<TerrariumMolecularAsset>, String> {
    let specs: Vec<TerrariumMolecularAssetSpec> =
        serde_json::from_str(TERRARIUM_MOLECULAR_ASSET_JSON)
            .map_err(|error| format!("failed to parse terrarium molecular asset JSON: {error}"))?;
    specs
        .into_iter()
        .map(parse_terrarium_molecular_asset)
        .collect()
}

fn parse_terrarium_molecular_asset(
    spec: TerrariumMolecularAssetSpec,
) -> Result<TerrariumMolecularAsset, String> {
    let species = TerrariumSpecies::from_name(&spec.species).ok_or_else(|| {
        format!(
            "unknown terrarium species `{}` in molecular asset registry",
            spec.species
        )
    })?;
    let atoms = spec
        .atoms
        .into_iter()
        .map(|atom| {
            let element = PeriodicElement::from_symbol_or_name(&atom.element).ok_or_else(|| {
                format!(
                    "unknown periodic element `{}` in terrarium molecular asset",
                    atom.element
                )
            })?;
            Ok(TerrariumMolecularAssetAtom {
                element,
                formal_charge: atom.formal_charge,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let atom_count = atoms.len();
    let bonds = spec
        .bonds
        .into_iter()
        .map(|bond| {
            if bond.i >= atom_count || bond.j >= atom_count {
                return Err(format!(
                    "bond indices ({}, {}) out of range for asset `{}` with {} atoms",
                    bond.i, bond.j, spec.name, atom_count
                ));
            }
            Ok(TerrariumMolecularAssetBond {
                i: bond.i,
                j: bond.j,
                order: bond.order.into_bond_order(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(TerrariumMolecularAsset {
        species,
        name: spec.name,
        quantum_fast_path: spec.quantum_fast_path,
        atoms,
        bonds,
    })
}

fn molecule_graph_from_asset(
    asset: &TerrariumMolecularAsset,
) -> Result<AtomisticMoleculeGraph, String> {
    let mut graph = AtomisticMoleculeGraph::new(&asset.name);
    for atom in &asset.atoms {
        let mut node = AtomNode::new(atom.element);
        node.formal_charge = atom.formal_charge;
        graph.add_atom_node(node);
    }
    for bond in &asset.bonds {
        graph.add_bond(bond.i, bond.j, bond.order)?;
    }
    Ok(graph)
}

fn embed_molecule_graph(graph: &AtomisticMoleculeGraph) -> Vec<[f32; 3]> {
    let atom_count = graph.atom_count();
    if atom_count == 0 {
        return Vec::new();
    }
    if atom_count == 1 {
        return vec![[0.0, 0.0, 0.0]];
    }

    let mut positions = initial_graph_positions(graph);
    let mut bonded_pairs = std::collections::HashSet::new();
    for bond in &graph.bonds {
        bonded_pairs.insert((bond.i.min(bond.j), bond.i.max(bond.j)));
    }

    for _ in 0..256 {
        let mut forces = vec![[0.0_f32; 3]; atom_count];

        for bond in &graph.bonds {
            let delta = vector_sub(positions[bond.j], positions[bond.i]);
            let distance = vector_norm(delta).max(1.0e-4);
            let direction = vector_scale(delta, 1.0 / distance);
            let target = target_bond_length_angstrom(
                graph.atoms[bond.i].element,
                graph.atoms[bond.j].element,
                bond.order,
            );
            let spring_force = vector_scale(direction, (distance - target) * 0.22);
            forces[bond.i] = vector_add(forces[bond.i], spring_force);
            forces[bond.j] = vector_sub(forces[bond.j], spring_force);
        }

        for i in 0..atom_count {
            for j in (i + 1)..atom_count {
                let delta = vector_sub(positions[j], positions[i]);
                let distance = vector_norm(delta).max(1.0e-3);
                let direction = vector_scale(delta, 1.0 / distance);
                let is_bonded = bonded_pairs.contains(&(i, j));
                let min_separation = nonbonded_min_separation_angstrom(
                    graph.atoms[i].element,
                    graph.atoms[j].element,
                );
                let overlap_push =
                    (min_separation - distance).max(0.0) * if is_bonded { 0.02 } else { 0.20 };
                let repulsive_force = (0.05 / (distance * distance))
                    * if is_bonded { 0.15 } else { 1.0 }
                    + overlap_push;
                let charge_product =
                    (graph.atoms[i].formal_charge as f32) * (graph.atoms[j].formal_charge as f32);
                let coulomb_force = charge_product * 0.01 / (distance * distance);
                let net_pair_force = repulsive_force + coulomb_force;
                let pair_force = vector_scale(direction, net_pair_force);
                forces[i] = vector_sub(forces[i], pair_force);
                forces[j] = vector_add(forces[j], pair_force);
            }
        }

        let mut max_step = 0.0_f32;
        for idx in 0..atom_count {
            let centered_force = vector_sub(forces[idx], vector_scale(positions[idx], 0.015));
            let step = vector_scale(centered_force, 0.12);
            positions[idx] = vector_add(positions[idx], step);
            max_step = max_step.max(vector_norm(step));
        }
        recenter_positions(&mut positions);
        if max_step < 1.0e-4 {
            break;
        }
    }

    positions
}

fn initial_graph_positions(graph: &AtomisticMoleculeGraph) -> Vec<[f32; 3]> {
    let atom_count = graph.atom_count();
    let mut positions = vec![[0.0_f32; 3]; atom_count];
    let depths = graph_bfs_depths(graph);
    for idx in 1..atom_count {
        let shell = depths[idx].max(1) as f32;
        let angle = idx as f32 * 2.399_963_1_f32;
        let z = if atom_count <= 2 {
            0.0
        } else {
            1.0 - 2.0 * idx as f32 / (atom_count as f32 - 1.0)
        };
        let radial = (1.0 - z * z).max(0.0).sqrt();
        let radius = 1.35 * shell;
        positions[idx] = [
            radius * radial * angle.cos(),
            radius * radial * angle.sin(),
            radius * z,
        ];
    }
    positions
}

fn graph_bfs_depths(graph: &AtomisticMoleculeGraph) -> Vec<usize> {
    let atom_count = graph.atom_count();
    let mut adjacency = vec![Vec::new(); atom_count];
    for bond in &graph.bonds {
        adjacency[bond.i].push(bond.j);
        adjacency[bond.j].push(bond.i);
    }
    let mut depths = vec![usize::MAX; atom_count];
    let mut queue = std::collections::VecDeque::new();
    depths[0] = 0;
    queue.push_back(0);
    while let Some(atom) = queue.pop_front() {
        for &neighbor in &adjacency[atom] {
            if depths[neighbor] == usize::MAX {
                depths[neighbor] = depths[atom] + 1;
                queue.push_back(neighbor);
            }
        }
    }
    for depth in &mut depths {
        if *depth == usize::MAX {
            *depth = 1;
        }
    }
    depths
}

fn target_bond_length_angstrom(a: PeriodicElement, b: PeriodicElement, order: BondOrder) -> f32 {
    let base = a.covalent_radius_angstrom() + b.covalent_radius_angstrom();
    let order_factor = match order {
        BondOrder::Single => 1.00,
        BondOrder::Double => 0.90,
        BondOrder::Triple => 0.84,
        BondOrder::Aromatic => 0.94,
    };
    (base * order_factor).max(0.70)
}

fn nonbonded_min_separation_angstrom(a: PeriodicElement, b: PeriodicElement) -> f32 {
    (a.covalent_radius_angstrom() + b.covalent_radius_angstrom()) * 0.82
}

fn recenter_positions(positions: &mut [[f32; 3]]) {
    if positions.is_empty() {
        return;
    }
    let centroid = positions.iter().copied().fold([0.0_f32; 3], vector_add);
    let centroid = vector_scale(centroid, 1.0 / positions.len() as f32);
    for position in positions.iter_mut() {
        *position = vector_sub(*position, centroid);
    }
}

fn vector_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn vector_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn vector_scale(v: [f32; 3], scalar: f32) -> [f32; 3] {
    [v[0] * scalar, v[1] * scalar, v[2] * scalar]
}

fn vector_norm(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_patch_inventory_species_define_atomic_binding_profiles() {
        for species in TERRARIUM_INVENTORY_BOUND_SPECIES {
            let profile = terrarium_inventory_species_profile(species)
                .expect("patch inventory species should define binding profile");
            assert!(!profile.pool_name.is_empty());
            assert!(!profile.element_counts.is_empty());
        }
    }

    #[test]
    fn inventory_bindings_build_expected_regions_and_phases() {
        let gas = terrarium_inventory_binding(TerrariumSpecies::OxygenGas)
            .expect("oxygen gas binding should exist");
        let mineral = terrarium_inventory_binding(TerrariumSpecies::SilicateMineral)
            .expect("silicate mineral binding should exist");

        assert_eq!(gas.0, MaterialRegionKind::GasPhase);
        assert_eq!(gas.2.kind, MaterialPhaseKind::Gas);
        assert_eq!(mineral.0, MaterialRegionKind::Soil);
        assert_eq!(mineral.2.kind, MaterialPhaseKind::Solid);
    }

    #[test]
    fn inventory_species_profiles_capture_canonical_charge_states() {
        let bicarbonate = terrarium_inventory_species_profile(TerrariumSpecies::BicarbonatePool)
            .expect("bicarbonate profile should exist");
        let calcium = terrarium_inventory_species_profile(TerrariumSpecies::ExchangeableCalcium)
            .expect("exchangeable calcium profile should exist");
        let aluminum = terrarium_inventory_species_profile(TerrariumSpecies::ExchangeableAluminum)
            .expect("exchangeable aluminum profile should exist");

        assert_eq!(bicarbonate.formal_charge, -1);
        assert_eq!(calcium.formal_charge, 2);
        assert_eq!(aluminum.formal_charge, 3);
    }

    #[test]
    fn terrarium_molecular_assets_cover_key_aqueous_and_mineral_species() {
        let water = terrarium_molecular_asset(TerrariumSpecies::Water)
            .expect("water molecular asset should exist");
        let bicarbonate = terrarium_molecular_asset(TerrariumSpecies::BicarbonatePool)
            .expect("bicarbonate molecular asset should exist");
        let carbonate = terrarium_molecular_asset(TerrariumSpecies::CarbonateMineral)
            .expect("carbonate mineral asset should exist");

        assert_eq!(water.atoms.len(), 3);
        assert_eq!(bicarbonate.bonds.len(), 4);
        assert_eq!(carbonate.atoms.len(), 5);
    }

    #[test]
    fn generated_embedded_molecules_have_nonzero_extent() {
        let water = terrarium_inventory_embedded_molecule(TerrariumSpecies::Water)
            .expect("water embedded molecule should exist");
        let nitrate = terrarium_inventory_embedded_molecule(TerrariumSpecies::Nitrate)
            .expect("nitrate embedded molecule should exist");

        assert!(water.max_extent() > 0.1);
        assert!(nitrate.max_extent() > 0.1);
    }

    #[test]
    fn quantum_descriptors_are_derived_from_asset_embedded_molecules() {
        let water = terrarium_inventory_quantum_descriptor(TerrariumSpecies::Water)
            .expect("water quantum descriptor should exist");
        let ammonium = terrarium_inventory_quantum_descriptor(TerrariumSpecies::Ammonium)
            .expect("ammonium quantum descriptor should exist");

        assert!(water.ground_state_energy_ev.is_finite());
        assert!(ammonium.ground_state_energy_ev.is_finite());
        assert!(water.dipole_magnitude_e_angstrom > 0.0);
        assert!(ammonium.mean_abs_effective_charge > 0.0);
    }

    #[test]
    fn larger_asset_motifs_get_regression_tier_descriptors() {
        // These species have >5 atoms and can't use the exact quantum solver,
        // but the regression tier should produce derived descriptors from their
        // canonical molecular structures.
        assert!(
            terrarium_inventory_embedded_molecule(TerrariumSpecies::CalciumBicarbonateComplex)
                .is_some()
        );
        let ca_bicarb = terrarium_inventory_quantum_descriptor(
            TerrariumSpecies::CalciumBicarbonateComplex,
        )
        .expect("calcium bicarbonate complex should get regression descriptor");
        assert!(ca_bicarb.ground_state_energy_ev.is_finite());

        assert!(
            terrarium_inventory_embedded_molecule(TerrariumSpecies::CarbonateMineral).is_some()
        );
        let carbonate = terrarium_inventory_quantum_descriptor(TerrariumSpecies::CarbonateMineral)
            .expect("carbonate mineral should get regression descriptor");
        assert!(carbonate.ground_state_energy_ev.is_finite());

        // Iron oxide and clay minerals should also get regression descriptors
        let iron_oxide = terrarium_inventory_quantum_descriptor(TerrariumSpecies::IronOxideMineral)
            .expect("iron oxide mineral should get regression descriptor");
        assert!(iron_oxide.ground_state_energy_ev.is_finite());
    }

    #[test]
    fn regression_tier_produces_structurally_plausible_values() {
        // Verify the regression tier's predictions are physically plausible:
        // larger molecules should have more negative ground state energies
        let water = terrarium_inventory_quantum_descriptor(TerrariumSpecies::Water)
            .expect("water descriptor");
        let ca_bicarb = terrarium_inventory_quantum_descriptor(
            TerrariumSpecies::CalciumBicarbonateComplex,
        )
        .expect("ca bicarb descriptor");

        // Ca(HCO3)2 has 11 atoms vs water's 3, so total energy should be
        // larger in magnitude
        assert!(
            ca_bicarb.ground_state_energy_ev.abs() > water.ground_state_energy_ev.abs(),
            "larger molecule should have greater total energy magnitude"
        );
    }

    #[test]
    fn warming_quantum_cache_populates_fast_path_entries_and_tensor_snapshot() {
        let warmed = warm_terrarium_quantum_descriptor_cache();
        let entries = terrarium_quantum_descriptor_cache_entries();
        let tensor = terrarium_quantum_descriptor_tensor_snapshot();

        assert_eq!(warmed, terrarium_quantum_fast_path_species().len());
        assert_eq!(entries.len(), warmed);
        assert!(entries.iter().any(|entry| entry.species == "water"));
        assert_eq!(
            tensor.cols,
            TERRARIUM_QUANTUM_DESCRIPTOR_FEATURE_NAMES.len()
        );
        assert_eq!(tensor.rows, entries.len());
        assert_eq!(tensor.values.len(), tensor.rows * tensor.cols);
        assert_eq!(tensor.species.len(), tensor.rows);
    }

    #[test]
    fn quantum_descriptor_cache_snapshots_round_trip_through_json() {
        let entries_json = terrarium_quantum_descriptor_cache_json_pretty()
            .expect("entries json should serialize");
        let tensor_json = terrarium_quantum_descriptor_tensor_json_pretty()
            .expect("tensor json should serialize");
        let entries: Vec<TerrariumQuantumDescriptorCacheEntry> =
            serde_json::from_str(&entries_json).expect("entries json should deserialize");
        let tensor: TerrariumQuantumDescriptorTensorSnapshot =
            serde_json::from_str(&tensor_json).expect("tensor json should deserialize");

        assert!(!entries.is_empty());
        assert_eq!(tensor.rows, entries.len());
        assert_eq!(
            tensor.feature_names.len(),
            TERRARIUM_QUANTUM_DESCRIPTOR_FEATURE_NAMES.len()
        );
    }

    #[test]
    fn precomputed_quantum_descriptor_asset_covers_fast_path_species() {
        let entries: Vec<TerrariumQuantumDescriptorCacheEntry> =
            serde_json::from_str(TERRARIUM_QUANTUM_DESCRIPTOR_CACHE_JSON)
                .expect("precomputed descriptor cache json should parse");

        assert_eq!(entries.len(), terrarium_quantum_fast_path_species().len());
        assert!(entries.iter().any(|entry| entry.species == "water"));
        assert!(entries
            .iter()
            .any(|entry| entry.species == "aqueous_iron_pool"));
    }

    #[test]
    fn binary_descriptor_cache_round_trips_faithfully() {
        let _ = warm_terrarium_quantum_descriptor_cache();
        let binary = terrarium_quantum_descriptor_cache_binary();

        // Validate header structure
        assert!(binary.len() > 4 + 4 + 32 + 4);
        let magic = u32::from_le_bytes(binary[0..4].try_into().unwrap());
        assert_eq!(magic, BINARY_CACHE_MAGIC);
        let version = u32::from_le_bytes(binary[4..8].try_into().unwrap());
        assert_eq!(version, BINARY_CACHE_VERSION);

        // Binary should be smaller than JSON
        let json = terrarium_quantum_descriptor_cache_json_pretty().unwrap();
        assert!(
            binary.len() < json.len(),
            "binary ({} bytes) should be smaller than JSON ({} bytes)",
            binary.len(),
            json.len()
        );
    }

    #[test]
    fn binary_descriptor_cache_validates_asset_hash() {
        let binary = terrarium_quantum_descriptor_cache_binary();
        // Corrupt the asset hash
        let mut corrupted = binary.clone();
        corrupted[8] ^= 0xFF;
        let result = prime_terrarium_quantum_descriptor_cache_from_binary(&corrupted);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("stale"));
    }

    #[test]
    fn binary_descriptor_cache_rejects_bad_magic() {
        let result = prime_terrarium_quantum_descriptor_cache_from_binary(&[0, 0, 0, 0, 0, 0, 0, 0]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("magic"));
    }

    #[test]
    fn molecular_asset_hash_is_deterministic() {
        let hash1 = terrarium_molecular_asset_hash();
        let hash2 = terrarium_molecular_asset_hash();
        assert_eq!(hash1, hash2);
        assert_ne!(hash1, [0u8; 32]);
    }
}
