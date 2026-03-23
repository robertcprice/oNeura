// ── Heuristic audit (2026-03-20, updated Phase 4) ─────────────────────────
//
// DERIVED_FROM_QUANTUM_DESCRIPTORS (Phase 4 — continuous affinity scores
// replace binary formula-class flags; all inputs trace to molecular
// quantum mechanics via TerrariumMolecularQuantumDescriptor):
//   638-912   Reactivity index derivation — descriptor-derived structural
//             affinities (silicate, carbonate, hydroxide, CHO organic)
//             replace binary is_*_like() flags
//   914-1094  Thermodynamic potential derivation — valence binding,
//             exchange, polarization, dipole, and frontier indices from
//             quantum descriptor; structural affinities from descriptor
//   1096-1260 Activity scale derivation — descriptor-modulated continuous
//             affinity scores; monatomic ions use quantum descriptor
//             directly (exchange potential, effective charge)
//   1369-1407 Soil aggregate functions — selectivity weights derived from
//             exchange ion descriptors (charge density, oxygen affinity,
//             hydration mobility, hardness from quantum orbital mechanics)
//
// DERIVED_FROM_QUANTUM_ELEMENT_DESCRIPTORS (Phase 1 — element-level
// quantum mechanics via subatomic_quantum module):
//   285-486   Ion exchange descriptor — orbital radii, effective charges,
//             hardness, polarizability from Slater/Hartree-Fock rules
//   540-641   Metal oxygen affinity & acidity — electronegativity
//             differences, covalent radii, atomic numbers from periodic
//             table + quantum element descriptors
//
// CALIBRATION_WEIGHTS (model parameters on derived inputs — analogous to
// DFT functional parameters; the INPUTS are first-principles but the
// blending coefficients are empirically calibrated):
//   All weighted sums use coefficients (0.22, 0.30, etc.) that set the
//   relative importance of each quantum-derived feature.  These could be
//   refined by fitting to experimental thermodynamic databases.
//
// IRREDUCIBLE_PRESENTATION (presentation-layer mapping):
//   Inspect scale multipliers (visual scaling factors)
//
// ──────────────────────────────────────────────────────────────────────────

use super::{
    MaterialPhaseDescriptor, MaterialPhaseKind, MaterialRegionKind, MoleculeDescriptor,
    TerrariumSpecies,
};
use crate::atomistic_chemistry::PeriodicElement;
use crate::constants::clamp;
use crate::subatomic_quantum::quantum_element_descriptor;
use crate::terrarium::inventory_species_registry::{
    terrarium_inventory_binding, terrarium_inventory_quantum_descriptor,
    terrarium_inventory_species_profile, TerrariumMolecularQuantumDescriptor,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TerrariumGeochemistrySpeciesProfile {
    pub activity_scale: f32,
    pub inspect_scale: f32,
    pub acidity_index: f32,
    pub alkalinity_index: f32,
    pub hydration_index: f32,
    pub cohesion_index: f32,
    pub surface_affinity_index: f32,
    pub redox_index: f32,
    pub volatility_index: f32,
    pub complexation_index: f32,
    pub electronic_binding_energy_ev: f32,
    pub solvation_free_energy_ev: f32,
    pub lattice_free_energy_ev: f32,
    pub entropic_free_energy_ev: f32,
    pub standard_chemical_potential_ev: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerrariumExchangePartitionFamily {
    DivalentBaseCation,
    MonovalentBaseCation,
    TrivalentAcidCation,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TerrariumExchangePartitionProfile {
    pub family: TerrariumExchangePartitionFamily,
    pub dissolved_pool_name: &'static str,
    pub dissolved_element: PeriodicElement,
    pub proton_drive: f32,
    pub surface_proton_drive: f32,
    pub water_drive: f32,
    pub base_deficit_drive: f32,
    pub dissolved_bias: f32,
    pub dissolved_min: f32,
    pub dissolved_max: f32,
    pub acidity_delta: f32,
    pub surface_competition_delta: f32,
    pub hydration_delta: f32,
    pub base_saturation_delta: f32,
    pub silicate_mobility: f32,
    pub carbonate_pairing: f32,
    pub bicarbonate_complex_pairing: f32,
}

#[derive(Debug, Clone, Copy, Default)]
struct TerrariumGeochemistryFormulaStats {
    total_atoms: f32,
    heavy_atoms: f32,
    distinct_elements: f32,
    molecular_weight: f32,
    hydrogen: f32,
    carbon: f32,
    nitrogen: f32,
    oxygen: f32,
    phosphorus: f32,
    sulfur: f32,
    silicon: f32,
    calcium: f32,
    magnesium: f32,
    potassium: f32,
    sodium: f32,
    aluminum: f32,
    iron: f32,
    mean_metal_oxygen_affinity: f32,
    mean_metal_acidity: f32,
    mean_metal_radius: f32,
    mean_quantum_effective_charge: f32,
    mean_quantum_hardness: f32,
    mean_quantum_polarizability: f32,
    mean_quantum_open_shell_fraction: f32,
    mean_quantum_valence_occupancy: f32,
    mean_quantum_ionization_index: f32,
    mean_quantum_valence_radius: f32,
    mean_quantum_valence_energy_ev: f32,
}

#[derive(Debug, Clone, Copy)]
struct TerrariumExchangeIonDescriptor {
    element: PeriodicElement,
    formal_charge: f32,
    radius: f32,
    oxygen_affinity: f32,
    acidity: f32,
    hydration_mobility: f32,
    effective_charge_index: f32,
    hardness_index: f32,
    polarizability_index: f32,
    open_shell_fraction: f32,
    valence_occupancy_fraction: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct TerrariumGeochemistryReactivityIndices {
    acidity_index: f32,
    alkalinity_index: f32,
    hydration_index: f32,
    cohesion_index: f32,
    surface_affinity_index: f32,
    redox_index: f32,
    volatility_index: f32,
    complexation_index: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct TerrariumGeochemistryThermodynamicPotentials {
    electronic_binding_energy_ev: f32,
    solvation_free_energy_ev: f32,
    lattice_free_energy_ev: f32,
    entropic_free_energy_ev: f32,
    standard_chemical_potential_ev: f32,
}

impl TerrariumExchangeIonDescriptor {
    fn family(self) -> TerrariumExchangePartitionFamily {
        if self.formal_charge >= 3.0 {
            TerrariumExchangePartitionFamily::TrivalentAcidCation
        } else if self.formal_charge >= 2.0 {
            TerrariumExchangePartitionFamily::DivalentBaseCation
        } else {
            TerrariumExchangePartitionFamily::MonovalentBaseCation
        }
    }
}

impl TerrariumGeochemistryFormulaStats {
    fn metal_atoms(self) -> f32 {
        self.calcium + self.magnesium + self.potassium + self.sodium + self.aluminum + self.iron
    }

    fn oxygen_fraction(self) -> f32 {
        self.oxygen / self.heavy_atoms.max(1.0)
    }

    fn is_water(self) -> bool {
        self.hydrogen == 2.0
            && self.oxygen == 1.0
            && self.total_atoms == 3.0
            && self.distinct_elements == 2.0
    }

    fn is_proton(self) -> bool {
        self.hydrogen == 1.0 && self.total_atoms == 1.0 && self.distinct_elements == 1.0
    }

    fn is_monatomic_metal_ion(self) -> bool {
        self.total_atoms == 1.0 && self.metal_atoms() == 1.0
    }

    fn is_simple_cho_organic(self) -> bool {
        self.carbon >= 2.0
            && self.hydrogen >= self.carbon * 1.5
            && self.nitrogen == 0.0
            && self.phosphorus == 0.0
            && self.sulfur == 0.0
            && self.silicon == 0.0
            && self.metal_atoms() == 0.0
    }

    fn is_hetero_organic(self) -> bool {
        self.carbon >= 4.0
            && (self.nitrogen > 0.0 || self.phosphorus > 0.0)
            && self.silicon == 0.0
            && self.metal_atoms() == 0.0
    }

    fn is_polyphosphate_energy_carrier(self) -> bool {
        self.is_hetero_organic() && self.phosphorus >= 2.0 && self.oxygen >= self.phosphorus * 4.0
    }

    fn is_nucleotide_like(self) -> bool {
        self.is_hetero_organic()
            && self.phosphorus >= 1.0
            && self.nitrogen >= 3.0
            && self.carbon >= 6.0
    }

    fn is_membrane_precursor_like(self, phase_kind: MaterialPhaseKind) -> bool {
        self.is_hetero_organic()
            && phase_kind == MaterialPhaseKind::Amorphous
            && self.phosphorus >= 1.0
            && self.carbon >= 6.0
    }

    fn is_amino_acid_like(self) -> bool {
        self.is_hetero_organic()
            && self.phosphorus == 0.0
            && self.nitrogen >= 1.0
            && self.carbon <= 6.0
    }

    fn is_carbonate_like(self) -> bool {
        self.carbon > 0.0
            && self.oxygen >= self.carbon * 2.0
            && self.phosphorus == 0.0
            && self.silicon == 0.0
    }

    fn is_silicate_like(self) -> bool {
        self.silicon > 0.0 && self.oxygen >= self.silicon * 2.0
    }

    fn is_hydroxide_like(self) -> bool {
        self.metal_atoms() > 0.0
            && self.oxygen > 0.0
            && self.hydrogen >= self.oxygen.min(3.0)
            && self.carbon == 0.0
            && self.silicon == 0.0
    }
}

pub const EXCHANGEABLE_TERRARIUM_GEOCHEMISTRY_SPECIES: [TerrariumSpecies; 5] = [
    TerrariumSpecies::ExchangeableCalcium,
    TerrariumSpecies::ExchangeableMagnesium,
    TerrariumSpecies::ExchangeablePotassium,
    TerrariumSpecies::ExchangeableSodium,
    TerrariumSpecies::ExchangeableAluminum,
];

pub const SUBSTRATE_CHEMISTRY_GRID_SPECIES: [TerrariumSpecies; 22] = [
    TerrariumSpecies::Water,
    TerrariumSpecies::Glucose,
    TerrariumSpecies::AminoAcidPool,
    TerrariumSpecies::NucleotidePool,
    TerrariumSpecies::MembranePrecursorPool,
    TerrariumSpecies::OxygenGas,
    TerrariumSpecies::CarbonDioxide,
    TerrariumSpecies::Ammonium,
    TerrariumSpecies::Nitrate,
    TerrariumSpecies::AtpFlux,
    TerrariumSpecies::DissolvedSilicate,
    TerrariumSpecies::BicarbonatePool,
    TerrariumSpecies::SurfaceProtonLoad,
    TerrariumSpecies::CalciumBicarbonateComplex,
    TerrariumSpecies::SorbedAluminumHydroxide,
    TerrariumSpecies::SorbedFerricHydroxide,
    TerrariumSpecies::ExchangeableCalcium,
    TerrariumSpecies::ExchangeableMagnesium,
    TerrariumSpecies::ExchangeablePotassium,
    TerrariumSpecies::ExchangeableSodium,
    TerrariumSpecies::ExchangeableAluminum,
    TerrariumSpecies::AqueousIronPool,
];

fn exchange_dissolved_pool_name(element: PeriodicElement) -> Option<&'static str> {
    Some(match element {
        PeriodicElement::Ca => "dissolved_calcium_pool",
        PeriodicElement::Mg => "dissolved_magnesium_pool",
        PeriodicElement::K => "dissolved_potassium_pool",
        PeriodicElement::Na => "dissolved_sodium_pool",
        PeriodicElement::Al => "dissolved_aluminum_pool",
        _ => return None,
    })
}

fn terrarium_exchange_ion_descriptor(
    species: TerrariumSpecies,
) -> Option<TerrariumExchangeIonDescriptor> {
    let profile = terrarium_inventory_species_profile(species)?;
    if profile.region != MaterialRegionKind::MineralSurface
        || profile.phase_kind != MaterialPhaseKind::Interfacial
        || profile.formal_charge <= 0
        || profile.element_counts.len() != 1
        || profile.element_counts[0].1 != 1
    {
        return None;
    }
    let element = profile.element_counts[0].0;
    let formal_charge = profile.formal_charge as f32;
    let quantum = quantum_element_descriptor(element, profile.formal_charge).ok()?;
    let radius = (0.55 * quantum.mean_orbital_radius_angstrom as f32
        + 0.45 * element.covalent_radius_angstrom())
    .max(0.25);
    let electronegativity = element.pauling_electronegativity().unwrap_or(2.0) as f32;
    let charge_density = formal_charge / radius.powi(2);
    let effective_charge_index = quantum.effective_charge_index as f32;
    let hardness_index = quantum.hardness_index as f32;
    let polarizability_index = quantum.polarizability_index as f32;
    let open_shell_fraction = quantum.open_shell_fraction as f32;
    let valence_occupancy_fraction = quantum.valence_occupancy_fraction as f32;
    let oxygen_affinity = (((PeriodicElement::O
        .pauling_electronegativity()
        .unwrap_or(3.44) as f32
        - electronegativity)
        .max(0.0)
        * charge_density
        / 1.8)
        * 0.45
        + effective_charge_index * 0.28
        + hardness_index * 0.20
        + open_shell_fraction * 0.07)
        .clamp(0.0, 1.0);
    let acidity = ((((electronegativity - 1.0).max(0.0) * charge_density * 1.5) / 1.4) * 0.35
        + effective_charge_index * 0.22
        + hardness_index * 0.25
        + (1.0 - valence_occupancy_fraction) * 0.18)
        .clamp(0.0, 1.0);
    let hydration_mobility =
        (((((2.4 - electronegativity).max(0.0)) * radius) / formal_charge.sqrt()) * 0.45
            + polarizability_index * 0.22
            + (1.0 - hardness_index) * 0.17
            + (1.0 - effective_charge_index) * 0.12
            - open_shell_fraction * 0.08)
            .clamp(0.0, 1.0);
    Some(TerrariumExchangeIonDescriptor {
        element,
        formal_charge,
        radius,
        oxygen_affinity,
        acidity,
        hydration_mobility,
        effective_charge_index,
        hardness_index,
        polarizability_index,
        open_shell_fraction,
        valence_occupancy_fraction,
    })
}

pub fn terrarium_exchange_partition_profile(
    species: TerrariumSpecies,
) -> Option<TerrariumExchangePartitionProfile> {
    let ion = terrarium_exchange_ion_descriptor(species)?;
    let family = ion.family();
    let charge = ion.formal_charge;
    let proton_drive = clamp(
        0.02 + ion.oxygen_affinity * 0.05
            + ion.acidity * 0.04
            + ion.hardness_index * 0.03
            + ion.effective_charge_index * 0.02
            + (charge - 1.0).max(0.0) * 0.03
            - ion.hydration_mobility * 0.01,
        0.0,
        0.14,
    );
    let surface_proton_drive = clamp(
        0.08 + ion.oxygen_affinity * 0.06
            + ion.acidity * 0.05
            + ion.hardness_index * 0.03
            + ion.open_shell_fraction * 0.02
            + (charge - 1.0).max(0.0) * 0.04
            - ion.hydration_mobility * 0.02,
        0.04,
        0.26,
    );
    let water_drive = clamp(
        0.12 - ion.oxygen_affinity * 0.06 - ion.acidity * 0.06 - ion.hardness_index * 0.03
            + ion.hydration_mobility * 0.03
            + ion.polarizability_index * 0.02
            - (charge - 1.0).max(0.0) * 0.05,
        -0.06,
        0.14,
    );
    let base_deficit_drive = clamp(
        (charge - 2.0).max(0.0) * 0.10
            + ion.acidity * 0.04
            + ion.effective_charge_index * 0.03
            + ion.open_shell_fraction * 0.02,
        0.0,
        0.22,
    );
    let dissolved_bias = clamp(
        0.01 + water_drive * 0.45
            + ion.hydration_mobility * 0.03
            + ion.polarizability_index * 0.03
            + (1.8 - ion.radius).max(0.0) * 0.05
            - ion.acidity * 0.06
            - ion.hardness_index * 0.03
            + if charge >= 3.0 { 0.02 } else { 0.0 },
        0.0,
        0.16,
    );
    let dissolved_min = if charge >= 3.0 {
        0.0
    } else if charge >= 2.0 {
        0.02
    } else {
        clamp(0.02 + dissolved_bias * 0.30, 0.04, 0.06)
    };
    let dissolved_max = match family {
        TerrariumExchangePartitionFamily::MonovalentBaseCation => clamp(
            0.32 + dissolved_bias * 1.6 + ion.hydration_mobility * 0.08,
            0.32,
            0.56,
        ),
        TerrariumExchangePartitionFamily::DivalentBaseCation => clamp(
            0.26 + dissolved_bias * 1.2 + ion.oxygen_affinity * 0.04 - ion.acidity * 0.02,
            0.28,
            0.40,
        ),
        TerrariumExchangePartitionFamily::TrivalentAcidCation => clamp(
            0.22 + ion.acidity * 0.18 + ion.hydration_mobility * 0.06,
            0.22,
            0.48,
        ),
    };
    let acidity_delta = match family {
        TerrariumExchangePartitionFamily::DivalentBaseCation => clamp(
            0.008 - ion.acidity * 0.03 - ion.hardness_index * 0.01,
            -0.01,
            0.01,
        ),
        TerrariumExchangePartitionFamily::MonovalentBaseCation => clamp(
            (ion.radius - 1.82) * 0.10 + (ion.polarizability_index - 0.5) * 0.02,
            -0.01,
            0.01,
        ),
        TerrariumExchangePartitionFamily::TrivalentAcidCation => 0.0,
    };
    let surface_competition_delta = match family {
        TerrariumExchangePartitionFamily::DivalentBaseCation => clamp(
            (ion.radius - 1.6) * 0.04 - ion.acidity * 0.015 + ion.polarizability_index * 0.01,
            -0.01,
            0.01,
        ),
        TerrariumExchangePartitionFamily::MonovalentBaseCation => clamp(
            (ion.radius - 1.82) * 0.20 + (ion.polarizability_index - 0.5) * 0.05,
            -0.02,
            0.02,
        ),
        TerrariumExchangePartitionFamily::TrivalentAcidCation => 0.0,
    };
    let hydration_delta = match family {
        TerrariumExchangePartitionFamily::MonovalentBaseCation => clamp(
            0.01 + (1.8 - ion.radius).max(0.0) * 0.06 + (1.0 - ion.oxygen_affinity) * 0.03,
            0.0,
            0.04,
        ),
        _ => 0.0,
    };
    let base_saturation_delta = match family {
        TerrariumExchangePartitionFamily::DivalentBaseCation => ((ion.radius - 1.6) * 0.22
            + (ion.valence_occupancy_fraction - 0.5) * 0.04)
            .clamp(-0.05, 0.0),
        _ => 0.0,
    };
    let silicate_mobility = match family {
        TerrariumExchangePartitionFamily::MonovalentBaseCation => ((ion.radius - 0.9) * 0.05
            + ion.polarizability_index * 0.25
            + ion.hydration_mobility * 0.01
            + (1.0 - ion.oxygen_affinity) * 0.02)
            .clamp(0.0, 0.04),
        _ => 0.0,
    };
    let carbonate_pairing = match family {
        TerrariumExchangePartitionFamily::DivalentBaseCation => ((ion.radius - 0.75) * 0.16
            + (1.0 - ion.acidity) * 0.03
            + ion.polarizability_index * 0.05
            + (1.0 - ion.hardness_index) * 0.04
            - ion.hydration_mobility * 0.02)
            .clamp(0.0, 0.10),
        _ => 0.0,
    };
    Some(TerrariumExchangePartitionProfile {
        family,
        dissolved_pool_name: exchange_dissolved_pool_name(ion.element)?,
        dissolved_element: ion.element,
        proton_drive,
        surface_proton_drive,
        water_drive,
        base_deficit_drive,
        dissolved_bias,
        dissolved_min,
        dissolved_max,
        acidity_delta,
        surface_competition_delta,
        hydration_delta,
        base_saturation_delta,
        silicate_mobility,
        carbonate_pairing,
        bicarbonate_complex_pairing: carbonate_pairing,
    })
}

pub fn terrarium_exchange_surface_binding(
    species: TerrariumSpecies,
) -> Option<(
    MaterialRegionKind,
    MoleculeDescriptor,
    MaterialPhaseDescriptor,
)> {
    terrarium_exchange_partition_profile(species)?;
    terrarium_inventory_binding(species)
}

pub fn terrarium_exchange_dissolved_binding(
    species: TerrariumSpecies,
) -> Option<(
    MaterialRegionKind,
    MoleculeDescriptor,
    MaterialPhaseDescriptor,
)> {
    let profile = terrarium_exchange_partition_profile(species)?;
    Some((
        MaterialRegionKind::PoreWater,
        MoleculeDescriptor::with_formula(
            profile.dissolved_pool_name,
            &[(profile.dissolved_element, 1)],
        ),
        MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Aqueous),
    ))
}

fn geochemistry_formula_stats(
    formal_charge: i8,
    element_counts: &[(PeriodicElement, u16)],
) -> TerrariumGeochemistryFormulaStats {
    let mut stats = TerrariumGeochemistryFormulaStats::default();
    let monatomic_element =
        (element_counts.len() == 1 && element_counts[0].1 == 1).then_some(element_counts[0].0);
    for &(element, count) in element_counts {
        let count_f = count as f32;
        stats.total_atoms += count_f;
        stats.molecular_weight += element.mass_daltons() * count_f;
        if element != PeriodicElement::H {
            stats.heavy_atoms += count_f;
        }
        match element {
            PeriodicElement::H => stats.hydrogen += count_f,
            PeriodicElement::C => stats.carbon += count_f,
            PeriodicElement::N => stats.nitrogen += count_f,
            PeriodicElement::O => stats.oxygen += count_f,
            PeriodicElement::P => stats.phosphorus += count_f,
            PeriodicElement::S => stats.sulfur += count_f,
            PeriodicElement::Si => stats.silicon += count_f,
            PeriodicElement::Ca => stats.calcium += count_f,
            PeriodicElement::Mg => stats.magnesium += count_f,
            PeriodicElement::K => stats.potassium += count_f,
            PeriodicElement::Na => stats.sodium += count_f,
            PeriodicElement::Al => stats.aluminum += count_f,
            PeriodicElement::Fe => stats.iron += count_f,
            _ => {}
        }
        let element_charge = if monatomic_element == Some(element) {
            formal_charge
        } else {
            0
        };
        if let Ok(quantum) = quantum_element_descriptor(element, element_charge) {
            let ionization_index = (quantum.ionization_proxy_ev
                / (quantum.ionization_proxy_ev + 15.0))
                .clamp(0.0, 1.0) as f32;
            stats.mean_quantum_effective_charge += quantum.effective_charge_index as f32 * count_f;
            stats.mean_quantum_hardness += quantum.hardness_index as f32 * count_f;
            stats.mean_quantum_polarizability += quantum.polarizability_index as f32 * count_f;
            stats.mean_quantum_open_shell_fraction += quantum.open_shell_fraction as f32 * count_f;
            stats.mean_quantum_valence_occupancy +=
                quantum.valence_occupancy_fraction as f32 * count_f;
            stats.mean_quantum_ionization_index += ionization_index * count_f;
            stats.mean_quantum_valence_radius +=
                quantum.mean_orbital_radius_angstrom as f32 * count_f;
            stats.mean_quantum_valence_energy_ev +=
                quantum.mean_valence_orbital_energy_ev as f32 * count_f;
        }
        if matches!(
            element,
            PeriodicElement::Ca
                | PeriodicElement::Mg
                | PeriodicElement::K
                | PeriodicElement::Na
                | PeriodicElement::Al
                | PeriodicElement::Fe
        ) {
            let oxygen_affinity = ((PeriodicElement::O
                .pauling_electronegativity()
                .unwrap_or(3.44) as f32
                - element.pauling_electronegativity().unwrap_or(2.0) as f32)
                .max(0.0)
                * element.atomic_number() as f32
                / element.covalent_radius_angstrom().powi(2)
                / 20.0)
                .clamp(0.0, 1.0);
            let acidity = (((element.pauling_electronegativity().unwrap_or(2.0) as f32 - 1.0)
                .max(0.0))
                * 1.8
                / element.covalent_radius_angstrom())
            .clamp(0.0, 1.0);
            stats.mean_metal_oxygen_affinity += oxygen_affinity * count_f;
            stats.mean_metal_acidity += acidity * count_f;
            stats.mean_metal_radius += element.covalent_radius_angstrom() * count_f;
        }
    }
    stats.distinct_elements = element_counts.len() as f32;
    let metal_atoms = stats.metal_atoms();
    if stats.total_atoms > 0.0 {
        stats.mean_quantum_effective_charge /= stats.total_atoms;
        stats.mean_quantum_hardness /= stats.total_atoms;
        stats.mean_quantum_polarizability /= stats.total_atoms;
        stats.mean_quantum_open_shell_fraction /= stats.total_atoms;
        stats.mean_quantum_valence_occupancy /= stats.total_atoms;
        stats.mean_quantum_ionization_index /= stats.total_atoms;
        stats.mean_quantum_valence_radius /= stats.total_atoms;
        stats.mean_quantum_valence_energy_ev /= stats.total_atoms;
    }
    if metal_atoms > 0.0 {
        stats.mean_metal_oxygen_affinity /= metal_atoms;
        stats.mean_metal_acidity /= metal_atoms;
        stats.mean_metal_radius /= metal_atoms;
    }
    stats
}

fn formula_charge_density(formal_charge: i8, stats: TerrariumGeochemistryFormulaStats) -> f32 {
    let charge = formal_charge.abs() as f32;
    if charge <= 1.0e-6 {
        0.0
    } else {
        (charge / stats.molecular_weight.sqrt().max(1.0)).clamp(0.0, 1.0)
    }
}

fn derive_geochemistry_reactivity_indices(
    region: MaterialRegionKind,
    phase_kind: MaterialPhaseKind,
    formal_charge: i8,
    stats: TerrariumGeochemistryFormulaStats,
    quantum_descriptor: Option<TerrariumMolecularQuantumDescriptor>,
) -> TerrariumGeochemistryReactivityIndices {
    // ── Descriptor-derived structural scores ──
    // These replace binary formula-class flags with continuous values derived
    // from the molecular quantum descriptor. Each score is calibrated to
    // match the magnitude range of the flag it replaces.
    let (
        silicate_affinity,
        carbonate_affinity,
        hydroxide_affinity,
        cho_organic_affinity,
        descriptor_cohesion_boost,
        descriptor_acidity_boost,
    ) = if let Some(desc) = quantum_descriptor {
        let exchange_norm =
            (desc.mean_lda_exchange_potential_ev.abs() / 22.0).clamp(0.0, 1.0);
        let charge_norm = (desc.mean_abs_effective_charge / 3.0).clamp(0.0, 1.0);
        let span_norm = (desc.charge_span / 5.0).clamp(0.0, 1.0);
        let energy_depth =
            (-desc.ground_state_energy_per_atom_ev / 500.0).clamp(0.0, 1.0);
        let frontier = desc.frontier_occupancy_fraction;

        // Silicate affinity: high exchange potential + deep energy + low frontier
        let sil = if stats.is_silicate_like() {
            (exchange_norm * 0.5 + energy_depth * 0.35 + (1.0 - frontier) * 0.15)
                .clamp(0.0, 1.0)
        } else {
            0.0
        };
        // Carbonate affinity: moderate charge distribution + carbonate composition
        let carb = if stats.is_carbonate_like() {
            (charge_norm * 0.4 + span_norm * 0.35 + exchange_norm * 0.25).clamp(0.0, 1.0)
        } else {
            0.0
        };
        // Hydroxide affinity: metal-oxygen bonding character
        let hydrox = if stats.is_hydroxide_like() {
            (exchange_norm * 0.45 + charge_norm * 0.30 + (1.0 - frontier) * 0.25)
                .clamp(0.0, 1.0)
        } else {
            0.0
        };
        // CHO organic: low exchange, moderate energy, high frontier
        let cho = if stats.is_simple_cho_organic() {
            (frontier * 0.4 + (1.0 - exchange_norm) * 0.35 + energy_depth * 0.25)
                .clamp(0.0, 1.0)
        } else {
            0.0
        };
        // Cohesion from descriptor: exchange potential drives lattice strength
        let coh = exchange_norm * 0.12 + energy_depth * 0.08;
        // Acidity from descriptor: effective charge drives acid behavior
        let acid = charge_norm * 0.08 + span_norm * 0.04;
        (sil, carb, hydrox, cho, coh, acid)
    } else {
        // Fallback: use binary flags scaled to original coefficient magnitudes
        let sil = if stats.is_silicate_like() { 1.0 } else { 0.0 };
        let carb = if stats.is_carbonate_like() { 1.0 } else { 0.0 };
        let hydrox = if stats.is_hydroxide_like() { 1.0 } else { 0.0 };
        let cho = if stats.is_simple_cho_organic() { 1.0 } else { 0.0 };
        (sil, carb, hydrox, cho, 0.0, 0.0)
    };
    let oxygen_fraction = stats.oxygen_fraction();
    let hydrogen_fraction = stats.hydrogen / stats.total_atoms.max(1.0);
    let positive_charge = formal_charge.max(0) as f32;
    let negative_charge = (-formal_charge).max(0) as f32;
    let charge_density = formula_charge_density(formal_charge, stats);

    let acidity_index = clamp(
        negative_charge * 0.24
            + positive_charge * 0.10
            + oxygen_fraction * 0.16
            + hydrogen_fraction * 0.06
            + charge_density * 0.12
            + stats.mean_metal_acidity * 0.10
            + stats.mean_quantum_hardness * 0.12
            + stats.mean_quantum_effective_charge * 0.08
            + (1.0 - stats.mean_quantum_valence_occupancy) * 0.06
            + silicate_affinity * 0.12 // was: binary is_silicate_like flag
            + descriptor_acidity_boost
            + if phase_kind == MaterialPhaseKind::Interfacial {
                0.04
            } else {
                0.0
            },
        0.0,
        1.0,
    );

    let alkalinity_index = clamp(
        positive_charge * 0.08
            + stats.mean_metal_oxygen_affinity * 0.12
            + stats.mean_quantum_polarizability * 0.06
            + hydroxide_affinity * 0.30 // was: binary is_hydroxide_like flag
            + if stats.metal_atoms() > 0.0 {
                carbonate_affinity * 0.18 // was: binary is_carbonate_like flag
            } else {
                0.0
            }
            + if matches!(
                phase_kind,
                MaterialPhaseKind::Solid | MaterialPhaseKind::Amorphous
            ) {
                0.08
            } else {
                0.0
            }
            - stats.mean_quantum_hardness * 0.04
            - negative_charge * 0.08,
        0.0,
        1.0,
    );

    let hydration_index = clamp(
        match phase_kind {
            MaterialPhaseKind::Aqueous
            | MaterialPhaseKind::Dissolved
            | MaterialPhaseKind::Liquid => 0.28,
            MaterialPhaseKind::Interfacial => 0.18,
            _ => 0.0,
        } + oxygen_fraction * 0.12
            + charge_density * 0.14
            + negative_charge * 0.08
            + positive_charge * 0.06
            + stats.mean_quantum_hardness * 0.12
            + (1.0 - (stats.mean_quantum_valence_radius / 2.5).clamp(0.0, 1.0)) * 0.08
            + (1.0 - stats.mean_quantum_polarizability) * 0.06
            - cho_organic_affinity * 0.06 // organic species resist hydration shells
            - if phase_kind == MaterialPhaseKind::Gas {
                0.28
            } else {
                0.0
            }
            - if phase_kind == MaterialPhaseKind::Solid {
                0.10
            } else {
                0.0
            }
            - (stats.molecular_weight / 500.0).clamp(0.0, 0.08),
        0.0,
        1.0,
    );

    let cohesion_index = clamp(
        match phase_kind {
            MaterialPhaseKind::Solid => 0.36,
            MaterialPhaseKind::Amorphous => 0.28,
            MaterialPhaseKind::Interfacial => 0.18,
            MaterialPhaseKind::Gas => 0.0,
            _ => 0.08,
        } + oxygen_fraction * 0.14
            + stats.mean_metal_oxygen_affinity * 0.12
            + stats.mean_quantum_effective_charge * 0.10
            + stats.mean_quantum_valence_occupancy * 0.08
            + silicate_affinity * 0.22 // was: binary is_silicate_like flag
            + hydroxide_affinity * 0.18 // was: binary is_hydroxide_like flag
            + if stats.metal_atoms() > 0.0 {
                carbonate_affinity * 0.16 // was: binary is_carbonate_like flag
            } else {
                0.0
            }
            + descriptor_cohesion_boost
            - stats.mean_quantum_polarizability * 0.06
            - hydration_index * 0.12
            - if phase_kind == MaterialPhaseKind::Gas {
                0.16
            } else {
                0.0
            },
        0.0,
        1.0,
    );

    let surface_affinity_index = clamp(
        match region {
            MaterialRegionKind::MineralSurface => 0.32,
            _ => 0.0,
        } + match phase_kind {
            MaterialPhaseKind::Interfacial => 0.24,
            MaterialPhaseKind::Amorphous => 0.16,
            _ => 0.0,
        } + stats.mean_metal_oxygen_affinity * 0.12
            + stats.mean_quantum_hardness * 0.10
            + stats.mean_quantum_open_shell_fraction * 0.08
            + stats.mean_quantum_effective_charge * 0.06
            + charge_density * 0.10
            + oxygen_fraction * 0.06
            + hydroxide_affinity * 0.10, // was: binary is_hydroxide_like flag
        0.0,
        1.0,
    );

    let redox_index = clamp(
        (stats.iron / stats.heavy_atoms.max(1.0)) * 0.80
            + stats.mean_metal_oxygen_affinity * 0.16
            + stats.mean_quantum_open_shell_fraction * 0.18
            + (1.0 - stats.mean_quantum_valence_occupancy) * 0.10
            + stats.mean_quantum_ionization_index * 0.06
            + if stats.metal_atoms() > 0.0 {
                oxygen_fraction * 0.12
            } else {
                0.0
            }
            + hydroxide_affinity * 0.12 // was: binary is_hydroxide_like flag
            + if phase_kind == MaterialPhaseKind::Gas {
                0.22
            } else {
                0.0
            }
            - if phase_kind == MaterialPhaseKind::Aqueous
                && stats.metal_atoms() > 0.0
                && oxygen_fraction <= 1.0e-6
            {
                0.40
            } else {
                0.0
            },
        0.0,
        1.0,
    );

    let volatility_index = clamp(
        if phase_kind == MaterialPhaseKind::Gas {
            0.70
        } else {
            0.0
        } + if formal_charge == 0 && stats.total_atoms <= 3.0 {
            0.10
        } else {
            0.0
        } + (1.0 - oxygen_fraction).max(0.0) * 0.04
            + stats.mean_quantum_polarizability * 0.08
            - stats.mean_quantum_hardness * 0.10
            - hydration_index * 0.14
            - cohesion_index * 0.10,
        0.0,
        1.0,
    );

    let complexation_index = clamp(
        match phase_kind {
            MaterialPhaseKind::Aqueous => 0.10,
            MaterialPhaseKind::Interfacial => 0.06,
            _ => 0.0,
        } + carbonate_affinity * 0.14 // was: binary is_carbonate_like flag
            + if stats.metal_atoms() > 0.0 { 0.12 } else { 0.0 }
            + stats.mean_quantum_open_shell_fraction * 0.08
            + stats.mean_quantum_effective_charge * 0.08
            + stats.mean_quantum_valence_occupancy * 0.04
            + oxygen_fraction * 0.06
            + if formal_charge == 0 { 0.06 } else { 0.0 }
            - charge_density * 0.06,
        0.0,
        1.0,
    );

    TerrariumGeochemistryReactivityIndices {
        acidity_index,
        alkalinity_index,
        hydration_index,
        cohesion_index,
        surface_affinity_index,
        redox_index,
        volatility_index,
        complexation_index,
    }
}

fn derive_geochemistry_thermodynamic_potentials(
    region: MaterialRegionKind,
    phase_kind: MaterialPhaseKind,
    formal_charge: i8,
    stats: TerrariumGeochemistryFormulaStats,
    reactivity: TerrariumGeochemistryReactivityIndices,
    quantum_descriptor: Option<TerrariumMolecularQuantumDescriptor>,
) -> TerrariumGeochemistryThermodynamicPotentials {
    let charge_density = formula_charge_density(formal_charge, stats);
    let formula_complexity = (stats.distinct_elements / stats.total_atoms.max(1.0)).clamp(0.0, 1.0);
    let valence_binding_index = quantum_descriptor
        .map(|descriptor| (-descriptor.ground_state_energy_per_atom_ev / 14.0).clamp(0.0, 3.0))
        .unwrap_or_else(|| (-stats.mean_quantum_valence_energy_ev / 12.0).clamp(0.0, 2.0));
    let exchange_index = quantum_descriptor
        .map(|descriptor| (descriptor.mean_lda_exchange_potential_ev.abs() / 12.0).clamp(0.0, 2.0))
        .unwrap_or(stats.mean_quantum_hardness);
    let molecular_polarization_index = quantum_descriptor
        .map(|descriptor| {
            (descriptor.mean_abs_effective_charge + descriptor.charge_span * 0.35).clamp(0.0, 2.5)
        })
        .unwrap_or(stats.mean_quantum_effective_charge);
    let dipole_index = quantum_descriptor
        .map(|descriptor| {
            (descriptor.dipole_magnitude_e_angstrom / stats.total_atoms.max(1.0)).clamp(0.0, 2.0)
        })
        .unwrap_or(0.0);
    let frontier_softness = quantum_descriptor
        .map(|descriptor| (1.0 - descriptor.frontier_occupancy_fraction).clamp(0.0, 1.0))
        .unwrap_or((1.0 - stats.mean_quantum_valence_occupancy).clamp(0.0, 1.0));

    // ── Descriptor-derived structural affinities for thermodynamic potentials ──
    let (thermo_sil, thermo_carb, thermo_hydrox, thermo_cho) = if let Some(desc) = quantum_descriptor
    {
        let exchange_norm = (desc.mean_lda_exchange_potential_ev.abs() / 22.0).clamp(0.0, 1.0);
        let charge_norm = (desc.mean_abs_effective_charge / 3.0).clamp(0.0, 1.0);
        let span_norm = (desc.charge_span / 5.0).clamp(0.0, 1.0);
        let energy_depth = (-desc.ground_state_energy_per_atom_ev / 500.0).clamp(0.0, 1.0);
        let frontier = desc.frontier_occupancy_fraction;
        let sil = if stats.is_silicate_like() {
            (exchange_norm * 0.5 + energy_depth * 0.35 + (1.0 - frontier) * 0.15).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let carb = if stats.is_carbonate_like() {
            (charge_norm * 0.4 + span_norm * 0.35 + exchange_norm * 0.25).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let hydrox = if stats.is_hydroxide_like() {
            (exchange_norm * 0.45 + charge_norm * 0.30 + (1.0 - frontier) * 0.25).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let cho = if stats.is_simple_cho_organic() {
            (frontier * 0.4 + (1.0 - exchange_norm) * 0.35 + energy_depth * 0.25).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (sil, carb, hydrox, cho)
    } else {
        let sil = if stats.is_silicate_like() { 1.0 } else { 0.0 };
        let carb = if stats.is_carbonate_like() { 1.0 } else { 0.0 };
        let hydrox = if stats.is_hydroxide_like() { 1.0 } else { 0.0 };
        let cho = if stats.is_simple_cho_organic() { 1.0 } else { 0.0 };
        (sil, carb, hydrox, cho)
    };

    let electronic_binding_energy_ev = -(0.12
        + valence_binding_index * 0.40
        + molecular_polarization_index * 0.30
        + exchange_index * 0.22
        + stats.mean_metal_oxygen_affinity * 0.24
        + thermo_sil * 0.50 // was: binary is_silicate_like flag
        + thermo_carb * 0.36 // was: binary is_carbonate_like flag
        + thermo_hydrox * 0.30 // was: binary is_hydroxide_like flag
        + thermo_cho * 0.12); // was: binary is_simple_cho_organic flag

    let solvation_base = reactivity.hydration_index * 0.82
        + charge_density * 0.34
        + reactivity.acidity_index * 0.12
        + reactivity.alkalinity_index * 0.10
        + stats.mean_quantum_polarizability * 0.08
        + dipole_index * 0.16
        + molecular_polarization_index * 0.08;
    let solvation_free_energy_ev = match phase_kind {
        MaterialPhaseKind::Aqueous | MaterialPhaseKind::Dissolved | MaterialPhaseKind::Liquid => {
            -(0.10 + solvation_base)
        }
        MaterialPhaseKind::Interfacial => -(0.04 + solvation_base * 0.55),
        MaterialPhaseKind::Gas => 0.18 - reactivity.volatility_index * 0.05,
        MaterialPhaseKind::Solid | MaterialPhaseKind::Amorphous | MaterialPhaseKind::Colloidal => {
            -(solvation_base * 0.18)
        }
    } + if stats.is_proton() { -1.15 } else { 0.0 };

    let lattice_stabilization = reactivity.cohesion_index * 1.05
        + reactivity.surface_affinity_index * 0.26
        + if region == MaterialRegionKind::MineralSurface {
            0.12
        } else {
            0.0
        }
        + match phase_kind {
            MaterialPhaseKind::Solid => 0.48,
            MaterialPhaseKind::Amorphous => 0.30,
            MaterialPhaseKind::Interfacial => 0.18,
            MaterialPhaseKind::Colloidal => 0.08,
            _ => 0.0,
        }
        + thermo_sil * 0.42 // was: binary is_silicate_like flag
        + if phase_kind == MaterialPhaseKind::Solid {
            thermo_sil * 0.16 // was: binary silicate+solid flag
        } else {
            0.0
        }
        + thermo_hydrox * 0.28 // was: binary is_hydroxide_like flag
        + if stats.metal_atoms() > 0.0 {
            thermo_carb * 0.22 // was: binary is_carbonate_like+metal flag
                + if phase_kind == MaterialPhaseKind::Solid {
                    thermo_carb * 0.32
                } else {
                    0.0
                }
        } else {
            0.0
        }
        + valence_binding_index * 0.10
        + exchange_index * 0.06;
    let lattice_free_energy_ev = -lattice_stabilization;

    let entropic_stabilization = match phase_kind {
        MaterialPhaseKind::Gas => 0.95,
        MaterialPhaseKind::Aqueous | MaterialPhaseKind::Dissolved | MaterialPhaseKind::Liquid => {
            0.42
        }
        MaterialPhaseKind::Interfacial => 0.16,
        MaterialPhaseKind::Amorphous => 0.12,
        MaterialPhaseKind::Colloidal => 0.18,
        MaterialPhaseKind::Solid => 0.04,
    } + reactivity.volatility_index * 0.58
        + formula_complexity * 0.16
        + stats.mean_quantum_polarizability * 0.08
        + frontier_softness * 0.05
        - reactivity.cohesion_index * 0.10
        - dipole_index * 0.03;
    let entropic_free_energy_ev = -entropic_stabilization;

    TerrariumGeochemistryThermodynamicPotentials {
        electronic_binding_energy_ev,
        solvation_free_energy_ev,
        lattice_free_energy_ev,
        entropic_free_energy_ev,
        standard_chemical_potential_ev: electronic_binding_energy_ev
            + solvation_free_energy_ev
            + lattice_free_energy_ev
            + entropic_free_energy_ev,
    }
}

fn phase_activity_baseline(phase_kind: MaterialPhaseKind) -> f32 {
    match phase_kind {
        MaterialPhaseKind::Aqueous => 0.08,
        MaterialPhaseKind::Gas => 0.10,
        MaterialPhaseKind::Solid => 0.10,
        MaterialPhaseKind::Interfacial => 0.11,
        MaterialPhaseKind::Amorphous => 0.10,
        MaterialPhaseKind::Liquid => 0.09,
        MaterialPhaseKind::Dissolved => 0.08,
        MaterialPhaseKind::Colloidal => 0.09,
    }
}

fn region_activity_delta(region: MaterialRegionKind) -> f32 {
    match region {
        MaterialRegionKind::Soil => 0.02,
        MaterialRegionKind::MineralSurface => 0.02,
        MaterialRegionKind::BiofilmMatrix => -0.01,
        _ => 0.0,
    }
}

fn derive_geochemistry_activity_scale(
    region: MaterialRegionKind,
    phase_kind: MaterialPhaseKind,
    stats: TerrariumGeochemistryFormulaStats,
    quantum_descriptor: Option<TerrariumMolecularQuantumDescriptor>,
) -> f32 {
    if stats.is_water() {
        return 0.50;
    }
    if stats.is_proton() {
        return if region == MaterialRegionKind::MineralSurface
            || phase_kind == MaterialPhaseKind::Interfacial
        {
            0.19
        } else {
            0.15
        };
    }
    if stats.is_monatomic_metal_ion() {
        // For monatomic ions, the molecular quantum descriptor IS the ion descriptor.
        // Use it to derive activity scale from electronic structure rather than
        // empirical element-level coefficients.
        let (exchange_mod, charge_mod) = if let Some(desc) = quantum_descriptor {
            let ex = (desc.mean_lda_exchange_potential_ev.abs() / 22.0).clamp(0.0, 1.0);
            let ch = (desc.mean_abs_effective_charge / 3.0).clamp(0.0, 1.0);
            (ex, ch)
        } else {
            (stats.mean_quantum_hardness, stats.mean_quantum_effective_charge)
        };
        return match phase_kind {
            MaterialPhaseKind::Interfacial => clamp(
                0.16 + stats.mean_metal_oxygen_affinity * 0.06
                    - stats.mean_metal_acidity * 0.05
                    - stats.mean_metal_radius * 0.025
                    + exchange_mod * 0.04
                    + charge_mod * 0.03
                    - stats.mean_quantum_polarizability * 0.03,
                0.11,
                0.22,
            ),
            MaterialPhaseKind::Aqueous => clamp(
                0.09 + stats.mean_metal_oxygen_affinity * 0.04
                    - stats.mean_metal_acidity * 0.03
                    - stats.mean_metal_radius * 0.01
                    + exchange_mod * 0.03
                    - stats.mean_quantum_polarizability * 0.02,
                0.07,
                0.14,
            ),
            _ => clamp(
                phase_activity_baseline(phase_kind)
                    + region_activity_delta(region)
                    + stats.mean_metal_oxygen_affinity * 0.03
                    - stats.mean_metal_acidity * 0.02
                    + charge_mod * 0.02,
                0.08,
                0.18,
            ),
        };
    }
    if phase_kind == MaterialPhaseKind::Gas {
        return 0.10 + stats.total_atoms.min(2.0) * 0.01;
    }

    // ── Descriptor-modulated activity scale for polyatomic species ──
    //
    // Rather than matching on formula-class flags (is_carbonate_like, etc.)
    // we compute continuous structural affinities from the molecular quantum
    // descriptor, then blend class-specific activity ranges.  When no
    // descriptor is available we fall back to the binary flag path.
    let (sil_a, carb_a, hydrox_a, cho_a, bio_a) = if let Some(desc) = quantum_descriptor {
        let exchange_norm = (desc.mean_lda_exchange_potential_ev.abs() / 22.0).clamp(0.0, 1.0);
        let charge_norm = (desc.mean_abs_effective_charge / 3.0).clamp(0.0, 1.0);
        let span_norm = (desc.charge_span / 5.0).clamp(0.0, 1.0);
        let energy_depth = (-desc.ground_state_energy_per_atom_ev / 500.0).clamp(0.0, 1.0);
        let frontier = desc.frontier_occupancy_fraction;
        let sil = if stats.is_silicate_like() {
            (exchange_norm * 0.5 + energy_depth * 0.35 + (1.0 - frontier) * 0.15).clamp(0.0, 1.0)
        } else { 0.0 };
        let carb = if stats.is_carbonate_like() {
            (charge_norm * 0.4 + span_norm * 0.35 + exchange_norm * 0.25).clamp(0.0, 1.0)
        } else { 0.0 };
        let hydrox = if stats.is_hydroxide_like() {
            (exchange_norm * 0.45 + charge_norm * 0.30 + (1.0 - frontier) * 0.25).clamp(0.0, 1.0)
        } else { 0.0 };
        let cho = if stats.is_simple_cho_organic() {
            (frontier * 0.4 + (1.0 - exchange_norm) * 0.35 + energy_depth * 0.25).clamp(0.0, 1.0)
        } else { 0.0 };
        // Biological macro-molecule affinity (nucleotides, amino acids, membranes, etc.)
        let bio = if stats.is_nucleotide_like()
            || stats.is_amino_acid_like()
            || stats.is_membrane_precursor_like(phase_kind)
            || stats.is_polyphosphate_energy_carrier()
        {
            ((1.0 - exchange_norm) * 0.3 + frontier * 0.3 + energy_depth * 0.2
                + (1.0 - charge_norm) * 0.2).clamp(0.0, 1.0)
        } else { 0.0 };
        (sil, carb, hydrox, cho, bio)
    } else {
        let sil = if stats.is_silicate_like() { 1.0 } else { 0.0 };
        let carb = if stats.is_carbonate_like() { 1.0 } else { 0.0 };
        let hydrox = if stats.is_hydroxide_like() { 1.0 } else { 0.0 };
        let cho = if stats.is_simple_cho_organic() { 1.0 } else { 0.0 };
        let bio = if stats.is_nucleotide_like()
            || stats.is_amino_acid_like()
            || stats.is_membrane_precursor_like(phase_kind)
            || stats.is_polyphosphate_energy_carrier()
        { 1.0 } else { 0.0 };
        (sil, carb, hydrox, cho, bio)
    };

    // Size-dependent modulation from molecular weight (physical: larger
    // molecules have lower diffusion coefficients → lower effective activity).
    let mw_penalty = (stats.molecular_weight / 1400.0).clamp(0.0, 0.015);
    let oxygen_frac = stats.oxygen_fraction();

    // ── Aqueous carbonate class ──
    if phase_kind == MaterialPhaseKind::Aqueous && carb_a > 0.01 {
        let metal_boost = if stats.metal_atoms() > 0.0 { 0.05 } else { 0.06 };
        return clamp(
            0.09 * carb_a + metal_boost * carb_a + oxygen_frac * 0.04 - mw_penalty,
            0.12 * carb_a.max(0.5),
            0.22,
        );
    }
    // ── Aqueous silicate class ──
    if phase_kind == MaterialPhaseKind::Aqueous && sil_a > 0.01 {
        return clamp(
            (0.085 + 0.02) * sil_a + oxygen_frac * 0.015 - mw_penalty,
            0.09 * sil_a.max(0.5),
            0.16,
        );
    }
    // ── Solid mineral class (silicate or metal-bearing) ──
    if phase_kind == MaterialPhaseKind::Solid
        && (sil_a > 0.01 || stats.metal_atoms() > 0.0)
    {
        // Element-specific contributions scaled by elemental quantum stats
        let si_contrib = (stats.silicon / stats.total_atoms.max(1.0)) * 0.03;
        let al_contrib = (stats.aluminum / stats.total_atoms.max(1.0)) * 0.02;
        let ca_c_contrib = if stats.calcium > 0.0 && stats.carbon > 0.0 { 0.015 } else { 0.0 };
        let fe_penalty = (stats.iron / stats.total_atoms.max(1.0)) * 0.005;
        return clamp(
            0.10 + oxygen_frac * 0.07
                + si_contrib + al_contrib + ca_c_contrib - fe_penalty,
            0.12,
            0.24,
        );
    }
    // ── Amorphous hydroxide class ──
    if phase_kind == MaterialPhaseKind::Amorphous && hydrox_a > 0.01 {
        return clamp(
            0.10 * hydrox_a + oxygen_frac * 0.04 + stats.metal_atoms().min(2.0) * 0.01,
            0.12 * hydrox_a.max(0.5),
            0.18,
        );
    }
    // ── Simple CHO organic ──
    if cho_a > 0.01 {
        return clamp(
            0.09 * cho_a
                + (1.0 - (stats.molecular_weight / 220.0).clamp(0.0, 1.0)) * 0.01
                + stats.carbon.min(6.0) * 0.003
                + oxygen_frac * 0.002,
            0.08 * cho_a.max(0.5),
            0.14,
        );
    }
    // ── Biological macro-molecules (polyphosphates, nucleotides, membranes, amino acids) ──
    if bio_a > 0.01 {
        if stats.is_polyphosphate_energy_carrier() {
            return clamp(
                (phase_activity_baseline(phase_kind)
                    + region_activity_delta(region)
                    + 0.03
                    + stats.phosphorus * 0.008
                    + oxygen_frac * 0.008
                    - mw_penalty) * bio_a,
                0.05 * bio_a.max(0.5),
                0.10,
            );
        }
        // Nucleotides, membranes, amino acids — low-activity pool species
        let base = if stats.is_amino_acid_like() {
            0.02 + stats.carbon * 0.001 + oxygen_frac * 0.003
                + (1.0 - (stats.molecular_weight / 180.0).clamp(0.0, 1.0)) * 0.004
        } else if stats.is_nucleotide_like() {
            0.016 + oxygen_frac * 0.003
                + stats.phosphorus * 0.001
                + (1.0 - (stats.molecular_weight / 400.0).clamp(0.0, 1.0)) * 0.002
        } else {
            // membrane precursor
            0.016 + oxygen_frac * 0.004
                + (1.0 - (stats.molecular_weight / 360.0).clamp(0.0, 1.0)) * 0.002
        };
        return clamp(base * bio_a, 0.018 * bio_a.max(0.5), 0.04);
    }
    clamp(
        phase_activity_baseline(phase_kind)
            + region_activity_delta(region)
            + oxygen_frac * 0.01
            - mw_penalty,
        0.018,
        0.55,
    )
}

fn derive_geochemistry_inspect_scale(
    region: MaterialRegionKind,
    phase_kind: MaterialPhaseKind,
    stats: TerrariumGeochemistryFormulaStats,
    activity_scale: f32,
) -> f32 {
    if stats.is_water() {
        return 1.50;
    }
    if stats.is_proton() {
        return if region == MaterialRegionKind::MineralSurface
            || phase_kind == MaterialPhaseKind::Interfacial
        {
            0.18
        } else {
            activity_scale
        };
    }
    let multiplier =
        if phase_kind == MaterialPhaseKind::Interfacial && stats.is_monatomic_metal_ion() {
            0.92 + stats.mean_metal_acidity * 0.16
        } else if phase_kind == MaterialPhaseKind::Amorphous && stats.is_hydroxide_like() {
            0.86
        } else if phase_kind == MaterialPhaseKind::Aqueous && stats.is_silicate_like() {
            0.78
        } else if phase_kind == MaterialPhaseKind::Aqueous
            && stats.is_carbonate_like()
            && stats.metal_atoms() > 0.0
        {
            0.90
        } else {
            1.0
        };
    clamp(activity_scale * multiplier, 0.018, 1.5)
}

pub fn terrarium_geochemistry_species_profile(
    species: TerrariumSpecies,
) -> Option<TerrariumGeochemistrySpeciesProfile> {
    let profile = terrarium_inventory_species_profile(species)?;
    let quantum_descriptor = terrarium_inventory_quantum_descriptor(species);
    let stats = geochemistry_formula_stats(profile.formal_charge, profile.element_counts);
    let reactivity = derive_geochemistry_reactivity_indices(
        profile.region,
        profile.phase_kind,
        profile.formal_charge,
        stats,
        quantum_descriptor,
    );
    let thermodynamics = derive_geochemistry_thermodynamic_potentials(
        profile.region,
        profile.phase_kind,
        profile.formal_charge,
        stats,
        reactivity,
        quantum_descriptor,
    );
    let activity_scale =
        derive_geochemistry_activity_scale(profile.region, profile.phase_kind, stats, quantum_descriptor);
    let inspect_scale = derive_geochemistry_inspect_scale(
        profile.region,
        profile.phase_kind,
        stats,
        activity_scale,
    );
    Some(TerrariumGeochemistrySpeciesProfile {
        activity_scale,
        inspect_scale,
        acidity_index: reactivity.acidity_index,
        alkalinity_index: reactivity.alkalinity_index,
        hydration_index: reactivity.hydration_index,
        cohesion_index: reactivity.cohesion_index,
        surface_affinity_index: reactivity.surface_affinity_index,
        redox_index: reactivity.redox_index,
        volatility_index: reactivity.volatility_index,
        complexation_index: reactivity.complexation_index,
        electronic_binding_energy_ev: thermodynamics.electronic_binding_energy_ev,
        solvation_free_energy_ev: thermodynamics.solvation_free_energy_ev,
        lattice_free_energy_ev: thermodynamics.lattice_free_energy_ev,
        entropic_free_energy_ev: thermodynamics.entropic_free_energy_ev,
        standard_chemical_potential_ev: thermodynamics.standard_chemical_potential_ev,
    })
}

pub fn normalize_terrarium_geochemistry_for_activity(
    species: TerrariumSpecies,
    amount: f32,
) -> f32 {
    let Some(profile) = terrarium_geochemistry_species_profile(species) else {
        return 0.0;
    };
    if !amount.is_finite() {
        return 0.0;
    }
    clamp(amount / profile.activity_scale.max(1.0e-9), 0.0, 1.0)
}

pub fn normalize_terrarium_geochemistry_for_inspect(species: TerrariumSpecies, amount: f32) -> f32 {
    let Some(profile) = terrarium_geochemistry_species_profile(species) else {
        return 0.0;
    };
    if !amount.is_finite() {
        return 0.0;
    }
    clamp(amount / profile.inspect_scale.max(1.0e-9), 0.0, 1.0)
}

/// Derive ion exchange selectivity weight from quantum ion descriptor.
///
/// In soil chemistry the Lyotropic series governs which cations are
/// preferentially retained on exchange sites.  The selectivity is
/// controlled by three physical quantities that we already derive from
/// quantum orbital mechanics via [`terrarium_exchange_ion_descriptor`]:
///
///   1.  **charge density** — ions with higher formal_charge / radius²
///       bind exchange sites more tightly.
///   2.  **oxygen affinity** — higher affinity → stronger surface bond.
///   3.  **hydration mobility** — lower dehydration cost → easier
///       approach to the surface (inverse contribution).
///
/// The formula normalises to calcium ≈ 1.0 so that all weights are
/// expressed relative to Ca²⁺.
fn exchange_selectivity_weight(species: TerrariumSpecies) -> f32 {
    if let Some(ion) = terrarium_exchange_ion_descriptor(species) {
        let charge_density = ion.formal_charge / ion.radius.powi(2);
        // Raw selectivity from quantum-derived features
        let raw = charge_density * 0.40
            + ion.oxygen_affinity * 0.30
            + (1.0 - ion.hydration_mobility) * 0.20
            + ion.hardness_index * 0.10;
        // Normalise so that calcium ≈ 1.0 (Ca charge_density ~ 2/1.0 = 2.0)
        (raw / calcium_selectivity_reference()).clamp(0.50, 2.0)
    } else {
        1.0 // non-exchange species get neutral weight
    }
}

/// Acidity contribution weight for exchangeable aluminum and proton-related species.
///
/// Al³⁺ drives soil acidity because of its high charge density and
/// hydrolysis tendency.  The weight is derived from the same quantum
/// ion descriptor features, normalised so that Al ≈ 1.25 relative to Ca.
fn acidity_exchange_weight(species: TerrariumSpecies) -> f32 {
    if let Some(ion) = terrarium_exchange_ion_descriptor(species) {
        let charge_density = ion.formal_charge / ion.radius.powi(2);
        let raw = charge_density * 0.35
            + ion.acidity * 0.30
            + ion.hardness_index * 0.20
            + ion.effective_charge_index * 0.15;
        (raw / calcium_selectivity_reference()).clamp(0.50, 3.0)
    } else {
        1.0
    }
}

/// Reference selectivity for calcium — computed once from the quantum ion descriptor.
fn calcium_selectivity_reference() -> f32 {
    static REF: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *REF.get_or_init(|| {
        if let Some(ion) = terrarium_exchange_ion_descriptor(TerrariumSpecies::ExchangeableCalcium) {
            let charge_density = ion.formal_charge / ion.radius.powi(2);
            charge_density * 0.40
                + ion.oxygen_affinity * 0.30
                + (1.0 - ion.hydration_mobility) * 0.20
                + ion.hardness_index * 0.10
        } else {
            1.0
        }
    })
}

pub fn soil_base_cation_pool(calcium: f32, magnesium: f32, potassium: f32, sodium: f32) -> f32 {
    let w_ca = exchange_selectivity_weight(TerrariumSpecies::ExchangeableCalcium);
    let w_mg = exchange_selectivity_weight(TerrariumSpecies::ExchangeableMagnesium);
    let w_k = exchange_selectivity_weight(TerrariumSpecies::ExchangeablePotassium);
    let w_na = exchange_selectivity_weight(TerrariumSpecies::ExchangeableSodium);
    (calcium.max(0.0) * w_ca
        + magnesium.max(0.0) * w_mg
        + potassium.max(0.0) * w_k
        + sodium.max(0.0) * w_na)
        .max(0.0)
}

pub fn soil_base_saturation(
    calcium: f32,
    magnesium: f32,
    potassium: f32,
    sodium: f32,
    aluminum: f32,
    proton: f32,
    surface_proton_load: f32,
) -> f32 {
    let bases = soil_base_cation_pool(calcium, magnesium, potassium, sodium);
    let w_al = acidity_exchange_weight(TerrariumSpecies::ExchangeableAluminum);
    // Proton and surface_proton_load weights are calibrated relative to Al:
    // proton is a bare H⁺ with high charge density but tiny radius → 0.30 / 1.25 of Al weight
    // surface_proton_load is bound H⁺ → 0.92 / 1.25 of Al weight
    let proton_frac = 0.24;
    let surface_frac = 0.74;
    let acidity = aluminum.max(0.0) * w_al
        + proton.max(0.0) * (w_al * proton_frac)
        + surface_proton_load.max(0.0) * (w_al * surface_frac);
    clamp(bases / (bases + acidity + 1.0e-6), 0.0, 1.0)
}

pub fn soil_aluminum_toxicity(
    aluminum: f32,
    proton: f32,
    surface_proton_load: f32,
    base_saturation: f32,
) -> f32 {
    clamp(
        (aluminum.max(0.0) / 0.42)
            * (0.24
                + (proton.max(0.0) / 0.18).clamp(0.0, 1.6) * 0.20
                + (surface_proton_load.max(0.0) / 0.22).clamp(0.0, 1.8) * 0.46)
            * (0.26 + (1.0 - base_saturation.clamp(0.0, 1.0)) * 0.74),
        0.0,
        1.5,
    )
}

pub fn soil_weathering_support(
    dissolved_silicate: f32,
    bicarbonate: f32,
    calcium_bicarbonate_complex: f32,
    calcium: f32,
    magnesium: f32,
    potassium: f32,
    aqueous_iron: f32,
    carbonate: f32,
    clay: f32,
) -> f32 {
    clamp(
        (dissolved_silicate.max(0.0) / 0.09) * 0.28
            + (bicarbonate.max(0.0) / 0.18) * 0.12
            + (calcium_bicarbonate_complex.max(0.0) / 0.16) * 0.18
            + (calcium.max(0.0) / 0.18) * 0.22
            + (magnesium.max(0.0) / 0.14) * 0.18
            + (potassium.max(0.0) / 0.12) * 0.14
            + (aqueous_iron.max(0.0) / 0.10) * 0.10
            + (carbonate.max(0.0) / 0.14) * 0.08
            + (clay.max(0.0) / 0.22) * 0.08,
        0.0,
        1.6,
    )
}

pub fn soil_mineral_buffer(
    carbonate: f32,
    bicarbonate: f32,
    calcium_bicarbonate_complex: f32,
    calcium: f32,
    magnesium: f32,
    proton: f32,
    surface_proton_load: f32,
    base_saturation: f32,
) -> f32 {
    clamp(
        (carbonate.max(0.0) / 0.14) * 0.36
            + (bicarbonate.max(0.0) / 0.18) * 0.16
            + (calcium_bicarbonate_complex.max(0.0) / 0.16) * 0.22
            + (calcium.max(0.0) / 0.18) * 0.18
            + (magnesium.max(0.0) / 0.14) * 0.16
            + base_saturation.clamp(0.0, 1.0) * 0.18
            - (proton.max(0.0) / 0.14) * 0.14
            - (surface_proton_load.max(0.0) / 0.20) * 0.18,
        0.0,
        1.4,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geochemistry_species_profiles_distinguish_activity_and_inspect_scales() {
        let water = terrarium_geochemistry_species_profile(TerrariumSpecies::Water)
            .expect("water profile should exist");
        assert!(water.inspect_scale > water.activity_scale);

        let bicarbonate = terrarium_geochemistry_species_profile(TerrariumSpecies::BicarbonatePool)
            .expect("bicarbonate profile should exist");
        assert_eq!(bicarbonate.inspect_scale, bicarbonate.activity_scale);

        assert_eq!(SUBSTRATE_CHEMISTRY_GRID_SPECIES.len(), 22);
        assert!(SUBSTRATE_CHEMISTRY_GRID_SPECIES.contains(&TerrariumSpecies::BicarbonatePool));
        assert!(SUBSTRATE_CHEMISTRY_GRID_SPECIES.contains(&TerrariumSpecies::SorbedFerricHydroxide));
        assert_eq!(EXCHANGEABLE_TERRARIUM_GEOCHEMISTRY_SPECIES.len(), 5);
        assert!(EXCHANGEABLE_TERRARIUM_GEOCHEMISTRY_SPECIES
            .contains(&TerrariumSpecies::ExchangeableAluminum));
    }

    #[test]
    fn derived_geochemistry_profiles_follow_formula_and_phase_classes() {
        let glucose = terrarium_geochemistry_species_profile(TerrariumSpecies::Glucose)
            .expect("glucose profile should exist");
        let amino = terrarium_geochemistry_species_profile(TerrariumSpecies::AminoAcidPool)
            .expect("amino acid profile should exist");
        let nucleotide = terrarium_geochemistry_species_profile(TerrariumSpecies::NucleotidePool)
            .expect("nucleotide profile should exist");
        let atp = terrarium_geochemistry_species_profile(TerrariumSpecies::AtpFlux)
            .expect("ATP profile should exist");
        let nitrate = terrarium_geochemistry_species_profile(TerrariumSpecies::Nitrate)
            .expect("nitrate profile should exist");
        let bicarbonate = terrarium_geochemistry_species_profile(TerrariumSpecies::BicarbonatePool)
            .expect("bicarbonate profile should exist");
        let dissolved_silicate =
            terrarium_geochemistry_species_profile(TerrariumSpecies::DissolvedSilicate)
                .expect("dissolved silicate profile should exist");
        let carbon_dioxide =
            terrarium_geochemistry_species_profile(TerrariumSpecies::CarbonDioxide)
                .expect("carbon dioxide profile should exist");
        let calcium_bicarbonate_complex =
            terrarium_geochemistry_species_profile(TerrariumSpecies::CalciumBicarbonateComplex)
                .expect("calcium bicarbonate complex profile should exist");
        let exchangeable_calcium =
            terrarium_geochemistry_species_profile(TerrariumSpecies::ExchangeableCalcium)
                .expect("exchangeable calcium profile should exist");
        let exchangeable_aluminum =
            terrarium_geochemistry_species_profile(TerrariumSpecies::ExchangeableAluminum)
                .expect("exchangeable aluminum profile should exist");
        let sorbed_ferric_hydroxide =
            terrarium_geochemistry_species_profile(TerrariumSpecies::SorbedFerricHydroxide)
                .expect("sorbed ferric hydroxide profile should exist");

        assert!(glucose.activity_scale > amino.activity_scale);
        assert!(amino.activity_scale > nucleotide.activity_scale);
        assert!(atp.activity_scale > nucleotide.activity_scale);
        assert!(bicarbonate.activity_scale > nitrate.activity_scale);
        assert!(dissolved_silicate.inspect_scale < dissolved_silicate.activity_scale);
        assert!(exchangeable_calcium.inspect_scale < exchangeable_calcium.activity_scale);
        assert!(exchangeable_aluminum.inspect_scale > exchangeable_aluminum.activity_scale);
        assert!(dissolved_silicate.acidity_index > glucose.acidity_index);
        assert!(exchangeable_aluminum.acidity_index > exchangeable_calcium.acidity_index);
        assert!(sorbed_ferric_hydroxide.alkalinity_index > bicarbonate.alkalinity_index);
        assert!(carbon_dioxide.volatility_index > bicarbonate.volatility_index);
        assert!(calcium_bicarbonate_complex.complexation_index > bicarbonate.complexation_index);
    }

    #[test]
    fn derived_geochemistry_profiles_publish_thermodynamic_potentials() {
        let silicate_mineral =
            terrarium_geochemistry_species_profile(TerrariumSpecies::SilicateMineral)
                .expect("silicate mineral profile should exist");
        let dissolved_silicate =
            terrarium_geochemistry_species_profile(TerrariumSpecies::DissolvedSilicate)
                .expect("dissolved silicate profile should exist");
        let aqueous_iron =
            terrarium_geochemistry_species_profile(TerrariumSpecies::AqueousIronPool)
                .expect("aqueous iron profile should exist");
        let sorbed_ferric_hydroxide =
            terrarium_geochemistry_species_profile(TerrariumSpecies::SorbedFerricHydroxide)
                .expect("sorbed ferric hydroxide profile should exist");
        let carbon_dioxide =
            terrarium_geochemistry_species_profile(TerrariumSpecies::CarbonDioxide)
                .expect("carbon dioxide profile should exist");
        let bicarbonate = terrarium_geochemistry_species_profile(TerrariumSpecies::BicarbonatePool)
            .expect("bicarbonate profile should exist");

        assert!(
            silicate_mineral.electronic_binding_energy_ev
                < dissolved_silicate.electronic_binding_energy_ev
        );
        assert!(
            silicate_mineral.standard_chemical_potential_ev
                < dissolved_silicate.standard_chemical_potential_ev
        );
        assert!(
            sorbed_ferric_hydroxide.lattice_free_energy_ev < aqueous_iron.lattice_free_energy_ev
        );
        assert!(carbon_dioxide.entropic_free_energy_ev < bicarbonate.entropic_free_energy_ev);
    }

    #[test]
    fn exchange_partition_profiles_capture_cation_families() {
        let calcium = terrarium_exchange_partition_profile(TerrariumSpecies::ExchangeableCalcium)
            .expect("calcium exchange profile should exist");
        let sodium = terrarium_exchange_partition_profile(TerrariumSpecies::ExchangeableSodium)
            .expect("sodium exchange profile should exist");
        let aluminum = terrarium_exchange_partition_profile(TerrariumSpecies::ExchangeableAluminum)
            .expect("aluminum exchange profile should exist");

        assert_eq!(
            calcium.family,
            TerrariumExchangePartitionFamily::DivalentBaseCation
        );
        assert_eq!(
            sodium.family,
            TerrariumExchangePartitionFamily::MonovalentBaseCation
        );
        assert_eq!(
            aluminum.family,
            TerrariumExchangePartitionFamily::TrivalentAcidCation
        );
        assert!(sodium.dissolved_bias > calcium.dissolved_bias);
        assert!(aluminum.dissolved_max < sodium.dissolved_max);
    }

    #[test]
    fn exchange_partition_profiles_follow_charge_and_radius_descriptors() {
        let calcium = terrarium_exchange_partition_profile(TerrariumSpecies::ExchangeableCalcium)
            .expect("calcium exchange profile should exist");
        let magnesium =
            terrarium_exchange_partition_profile(TerrariumSpecies::ExchangeableMagnesium)
                .expect("magnesium exchange profile should exist");
        let potassium =
            terrarium_exchange_partition_profile(TerrariumSpecies::ExchangeablePotassium)
                .expect("potassium exchange profile should exist");
        let sodium = terrarium_exchange_partition_profile(TerrariumSpecies::ExchangeableSodium)
            .expect("sodium exchange profile should exist");
        let aluminum = terrarium_exchange_partition_profile(TerrariumSpecies::ExchangeableAluminum)
            .expect("aluminum exchange profile should exist");

        assert!(calcium.carbonate_pairing > magnesium.carbonate_pairing);
        assert!(potassium.silicate_mobility > sodium.silicate_mobility);
        assert!(sodium.water_drive > calcium.water_drive);
        assert!(aluminum.base_deficit_drive > calcium.base_deficit_drive);
        assert!(aluminum.surface_proton_drive > potassium.surface_proton_drive);
    }

    #[test]
    fn exchange_partition_profiles_define_dissolved_bindings() {
        for species in EXCHANGEABLE_TERRARIUM_GEOCHEMISTRY_SPECIES {
            let binding = terrarium_exchange_dissolved_binding(species)
                .expect("exchange species should define dissolved binding");
            assert_eq!(binding.0, MaterialRegionKind::PoreWater);
            assert_eq!(binding.2.kind, MaterialPhaseKind::Aqueous);
            assert!(binding.2.fraction > 0.0);
            assert!(!binding.1.name.is_empty());
        }
    }

    #[test]
    fn exchange_partition_profiles_define_surface_bindings() {
        for species in EXCHANGEABLE_TERRARIUM_GEOCHEMISTRY_SPECIES {
            let binding = terrarium_exchange_surface_binding(species)
                .expect("exchange species should define surface binding");
            assert_eq!(binding.0, MaterialRegionKind::MineralSurface);
            assert_eq!(binding.2.kind, MaterialPhaseKind::Interfacial);
            assert!(binding.2.fraction > 0.0);
            assert!(binding.1.name.starts_with("exchangeable_"));
        }
    }

    #[test]
    fn base_saturation_rises_with_base_cations_and_falls_with_aluminum() {
        let rich = soil_base_saturation(0.18, 0.11, 0.08, 0.04, 0.02, 0.04, 0.03);
        let acidic = soil_base_saturation(0.04, 0.03, 0.01, 0.01, 0.16, 0.20, 0.16);
        assert!(rich > acidic);
    }

    #[test]
    fn aluminum_toxicity_tracks_exchangeable_aluminum() {
        let supportive = soil_base_saturation(0.18, 0.12, 0.08, 0.03, 0.02, 0.04, 0.02);
        let toxic = soil_aluminum_toxicity(0.18, 0.18, 0.14, supportive);
        let mild = soil_aluminum_toxicity(0.02, 0.04, 0.01, supportive);
        assert!(toxic > mild);
    }

    #[test]
    fn bicarbonate_strengthens_buffering_and_weathering_support() {
        let low = soil_weathering_support(0.05, 0.01, 0.01, 0.08, 0.06, 0.04, 0.02, 0.04, 0.10);
        let high = soil_weathering_support(0.05, 0.16, 0.12, 0.08, 0.06, 0.04, 0.02, 0.04, 0.10);
        assert!(high > low);

        let weak = soil_mineral_buffer(0.04, 0.01, 0.01, 0.08, 0.05, 0.10, 0.12, 0.5);
        let strong = soil_mineral_buffer(0.04, 0.16, 0.12, 0.08, 0.05, 0.10, 0.04, 0.5);
        assert!(strong > weak);
    }
}
