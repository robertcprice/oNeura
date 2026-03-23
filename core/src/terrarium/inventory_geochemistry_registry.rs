// ── Heuristic audit (2026-03-20, updated Phase 4) ─────────────────────────
//
// TRANSITIVELY_DESCRIPTOR_DERIVED (inputs are now quantum-descriptor-
// derived via geochemistry.rs reactivity indices and thermodynamic
// potentials — Phase 4):
//   624-641   Stoichiometric signal weight formula — reactivity indices
//             now carry descriptor-derived structural affinities
//   829-884   Reaction profile — transport/barrier/environment all
//             consume descriptor-enriched species profiles
//   896-952   Environmental affinity — drives (proton, water, base
//             saturation, surface, alkalinity, redox) are computed
//             from deltas of descriptor-informed reactivity indices
//
// CALIBRATION_WEIGHTS (model parameters on derived inputs):
//   196-199   Thermodynamic gate sigmoid (scaling divisor 2.5)
//   647-648   Reactant/product activity weights (+0.22 / -0.18)
//   1193-1212 Basis multiplier overrides (proton 0.98, surface 0.18)
//
// IRREDUCIBLE_PRESENTATION:
//   327-333   Neighborhood sampling weights (center/orthogonal/diagonal)
//
// ──────────────────────────────────────────────────────────────────────────

use super::*;
use crate::constants::clamp;
use crate::terrarium::geochemistry::{
    normalize_terrarium_geochemistry_for_activity, terrarium_exchange_partition_profile,
    terrarium_geochemistry_species_profile, EXCHANGEABLE_TERRARIUM_GEOCHEMISTRY_SPECIES,
};
use crate::terrarium::inventory_reaction_network::{
    InventoryReactionDefinition, InventoryReactionTerm,
};
use crate::terrarium::inventory_species_registry::terrarium_inventory_species_profile;
use std::sync::OnceLock;

const TERRARIUM_STANDARD_RT_EV: f32 = 8.617_333_262_145e-5_f32 * 298.15_f32;

#[derive(Debug, Clone, Copy)]
pub(crate) enum InventoryGeochemistrySignal {
    Water,
    Proton,
    DissolvedSilicate,
    Bicarbonate,
    SurfaceProtonLoad,
    CalciumBicarbonateComplex,
    SilicateMineral,
    CarbonateMineral,
    Calcium,
    Aluminum,
    AqueousIron,
    SorbedAluminumHydroxide,
    SorbedFerricHydroxide,
    BaseSaturation,
    AlkalinityGate,
    OxygenNorm,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct InventoryGeochemistryState {
    pub water: f32,
    pub proton: f32,
    pub dissolved_silicate: f32,
    pub bicarbonate: f32,
    pub surface_proton_load: f32,
    pub calcium_bicarbonate_complex: f32,
    pub silicate_mineral: f32,
    pub carbonate_mineral: f32,
    pub calcium: f32,
    pub aluminum: f32,
    pub aqueous_iron: f32,
    pub sorbed_aluminum_hydroxide: f32,
    pub sorbed_ferric_hydroxide: f32,
    pub base_saturation: f32,
    pub alkalinity_gate: f32,
    pub oxygen_norm: f32,
}

impl InventoryGeochemistryState {
    pub(crate) fn signal(self, signal: InventoryGeochemistrySignal) -> f32 {
        match signal {
            InventoryGeochemistrySignal::Water => self.water,
            InventoryGeochemistrySignal::Proton => self.proton,
            InventoryGeochemistrySignal::DissolvedSilicate => self.dissolved_silicate,
            InventoryGeochemistrySignal::Bicarbonate => self.bicarbonate,
            InventoryGeochemistrySignal::SurfaceProtonLoad => self.surface_proton_load,
            InventoryGeochemistrySignal::CalciumBicarbonateComplex => {
                self.calcium_bicarbonate_complex
            }
            InventoryGeochemistrySignal::SilicateMineral => self.silicate_mineral,
            InventoryGeochemistrySignal::CarbonateMineral => self.carbonate_mineral,
            InventoryGeochemistrySignal::Calcium => self.calcium,
            InventoryGeochemistrySignal::Aluminum => self.aluminum,
            InventoryGeochemistrySignal::AqueousIron => self.aqueous_iron,
            InventoryGeochemistrySignal::SorbedAluminumHydroxide => self.sorbed_aluminum_hydroxide,
            InventoryGeochemistrySignal::SorbedFerricHydroxide => self.sorbed_ferric_hydroxide,
            InventoryGeochemistrySignal::BaseSaturation => self.base_saturation,
            InventoryGeochemistrySignal::AlkalinityGate => self.alkalinity_gate,
            InventoryGeochemistrySignal::OxygenNorm => self.oxygen_norm,
        }
    }

    pub(crate) fn normalized(self, signal: InventoryGeochemistrySignal) -> f32 {
        if let Some(species) = species_for_signal(signal) {
            return normalize_terrarium_geochemistry_for_activity(species, self.signal(signal));
        }
        match signal {
            InventoryGeochemistrySignal::BaseSaturation
            | InventoryGeochemistrySignal::AlkalinityGate
            | InventoryGeochemistrySignal::OxygenNorm => self.signal(signal).clamp(0.0, 1.0),
            _ => 0.0,
        }
    }
}

fn signal_for_species(species: TerrariumSpecies) -> Option<InventoryGeochemistrySignal> {
    match species {
        TerrariumSpecies::Water => Some(InventoryGeochemistrySignal::Water),
        TerrariumSpecies::Proton => Some(InventoryGeochemistrySignal::Proton),
        TerrariumSpecies::DissolvedSilicate => Some(InventoryGeochemistrySignal::DissolvedSilicate),
        TerrariumSpecies::BicarbonatePool => Some(InventoryGeochemistrySignal::Bicarbonate),
        TerrariumSpecies::SurfaceProtonLoad => Some(InventoryGeochemistrySignal::SurfaceProtonLoad),
        TerrariumSpecies::CalciumBicarbonateComplex => {
            Some(InventoryGeochemistrySignal::CalciumBicarbonateComplex)
        }
        TerrariumSpecies::SilicateMineral => Some(InventoryGeochemistrySignal::SilicateMineral),
        TerrariumSpecies::CarbonateMineral => Some(InventoryGeochemistrySignal::CarbonateMineral),
        TerrariumSpecies::ExchangeableCalcium => Some(InventoryGeochemistrySignal::Calcium),
        TerrariumSpecies::ExchangeableAluminum => Some(InventoryGeochemistrySignal::Aluminum),
        TerrariumSpecies::AqueousIronPool => Some(InventoryGeochemistrySignal::AqueousIron),
        TerrariumSpecies::SorbedAluminumHydroxide => {
            Some(InventoryGeochemistrySignal::SorbedAluminumHydroxide)
        }
        TerrariumSpecies::SorbedFerricHydroxide => {
            Some(InventoryGeochemistrySignal::SorbedFerricHydroxide)
        }
        _ => None,
    }
}

fn species_for_signal(signal: InventoryGeochemistrySignal) -> Option<TerrariumSpecies> {
    match signal {
        InventoryGeochemistrySignal::Water => Some(TerrariumSpecies::Water),
        InventoryGeochemistrySignal::Proton => Some(TerrariumSpecies::Proton),
        InventoryGeochemistrySignal::DissolvedSilicate => Some(TerrariumSpecies::DissolvedSilicate),
        InventoryGeochemistrySignal::Bicarbonate => Some(TerrariumSpecies::BicarbonatePool),
        InventoryGeochemistrySignal::SurfaceProtonLoad => Some(TerrariumSpecies::SurfaceProtonLoad),
        InventoryGeochemistrySignal::CalciumBicarbonateComplex => {
            Some(TerrariumSpecies::CalciumBicarbonateComplex)
        }
        InventoryGeochemistrySignal::SilicateMineral => Some(TerrariumSpecies::SilicateMineral),
        InventoryGeochemistrySignal::CarbonateMineral => Some(TerrariumSpecies::CarbonateMineral),
        InventoryGeochemistrySignal::Calcium => Some(TerrariumSpecies::ExchangeableCalcium),
        InventoryGeochemistrySignal::Aluminum => Some(TerrariumSpecies::ExchangeableAluminum),
        InventoryGeochemistrySignal::AqueousIron => Some(TerrariumSpecies::AqueousIronPool),
        InventoryGeochemistrySignal::SorbedAluminumHydroxide => {
            Some(TerrariumSpecies::SorbedAluminumHydroxide)
        }
        InventoryGeochemistrySignal::SorbedFerricHydroxide => {
            Some(TerrariumSpecies::SorbedFerricHydroxide)
        }
        InventoryGeochemistrySignal::BaseSaturation
        | InventoryGeochemistrySignal::AlkalinityGate
        | InventoryGeochemistrySignal::OxygenNorm => None,
    }
}

fn reaction_activity(
    state: InventoryGeochemistryState,
    terms: &'static [InventoryReactionTerm],
) -> f32 {
    let mut total = 0.0;
    let mut weight = 0.0;
    for term in terms {
        let Some(signal) = signal_for_species(term.species) else {
            continue;
        };
        let stoich = term.stoichiometry.max(0.0);
        total += state.normalized(signal) * stoich;
        weight += stoich;
    }
    if weight <= 1.0e-9 {
        0.0
    } else {
        total / weight
    }
}

fn thermodynamic_activity(state: InventoryGeochemistryState, species: TerrariumSpecies) -> f32 {
    signal_for_species(species)
        .map(|signal| state.normalized(signal).max(1.0e-4))
        .unwrap_or(1.0)
}

fn reaction_log_quotient(
    state: InventoryGeochemistryState,
    reaction: &'static InventoryReactionDefinition,
) -> f32 {
    let product_term = reaction
        .products
        .iter()
        .map(|term| thermodynamic_activity(state, term.species).ln() * term.stoichiometry.max(0.0))
        .sum::<f32>();
    let reactant_term = reaction
        .reactants
        .iter()
        .map(|term| thermodynamic_activity(state, term.species).ln() * term.stoichiometry.max(0.0))
        .sum::<f32>();
    product_term - reactant_term
}

fn reaction_delta_g_ev(
    state: InventoryGeochemistryState,
    reaction: &'static InventoryReactionDefinition,
    standard_delta_g_ev: f32,
) -> f32 {
    standard_delta_g_ev + TERRARIUM_STANDARD_RT_EV * reaction_log_quotient(state, reaction)
}

fn thermodynamic_forward_gate(delta_g_ev: f32) -> f32 {
    let scaled = (delta_g_ev / 2.5).clamp(-12.0, 12.0);
    1.0 / (1.0 + scaled.exp())
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum InventoryGeochemistryBasisOperand {
    Signal {
        signal: InventoryGeochemistrySignal,
        multiplier: f32,
    },
    Headroom {
        capacity: f32,
        signal: InventoryGeochemistrySignal,
    },
}

impl InventoryGeochemistryBasisOperand {
    pub const fn signal(signal: InventoryGeochemistrySignal, multiplier: f32) -> Self {
        Self::Signal { signal, multiplier }
    }

    pub const fn headroom(capacity: f32, signal: InventoryGeochemistrySignal) -> Self {
        Self::Headroom { capacity, signal }
    }

    pub fn eval(self, state: InventoryGeochemistryState) -> f32 {
        match self {
            InventoryGeochemistryBasisOperand::Signal { signal, multiplier } => {
                state.signal(signal) * multiplier.max(0.0)
            }
            InventoryGeochemistryBasisOperand::Headroom { capacity, signal } => {
                (capacity - state.signal(signal)).max(0.0)
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum InventoryGeochemistryBasis {
    Operand(InventoryGeochemistryBasisOperand),
    Min(
        InventoryGeochemistryBasisOperand,
        InventoryGeochemistryBasisOperand,
    ),
}

impl InventoryGeochemistryBasis {
    pub const fn operand(operand: InventoryGeochemistryBasisOperand) -> Self {
        Self::Operand(operand)
    }

    pub const fn min(
        lhs: InventoryGeochemistryBasisOperand,
        rhs: InventoryGeochemistryBasisOperand,
    ) -> Self {
        Self::Min(lhs, rhs)
    }

    pub fn eval(self, state: InventoryGeochemistryState) -> f32 {
        match self {
            InventoryGeochemistryBasis::Operand(operand) => operand.eval(state),
            InventoryGeochemistryBasis::Min(lhs, rhs) => lhs.eval(state).min(rhs.eval(state)),
        }
    }
}

fn basis_operand_from_reactant(
    term: InventoryReactionTerm,
    multiplier_override: Option<f32>,
) -> Option<InventoryGeochemistryBasisOperand> {
    let signal = signal_for_species(term.species)?;
    let multiplier = multiplier_override.unwrap_or_else(|| 1.0 / term.stoichiometry.max(1.0e-9));
    Some(InventoryGeochemistryBasisOperand::signal(
        signal, multiplier,
    ))
}

fn basis_from_reaction_reactants(
    reaction: &'static InventoryReactionDefinition,
    multiplier_overrides: &[(TerrariumSpecies, f32)],
) -> InventoryGeochemistryBasis {
    let operands = reaction
        .reactants
        .iter()
        .filter_map(|term| {
            let multiplier_override =
                multiplier_overrides
                    .iter()
                    .find_map(|(species, multiplier)| {
                        (*species == term.species).then_some(*multiplier)
                    });
            basis_operand_from_reactant(*term, multiplier_override)
        })
        .collect::<Vec<_>>();
    match operands.as_slice() {
        [operand] => InventoryGeochemistryBasis::operand(*operand),
        [lhs, rhs] => InventoryGeochemistryBasis::min(*lhs, *rhs),
        _ => panic!(
            "unsupported reactant basis shape for inventory geochemistry reaction {}",
            reaction.name
        ),
    }
}

fn basis_from_primary_reactant(
    reaction: &'static InventoryReactionDefinition,
    species: TerrariumSpecies,
) -> InventoryGeochemistryBasis {
    let term = reaction
        .reactants
        .iter()
        .find(|term| term.species == species)
        .unwrap_or_else(|| {
            panic!(
                "missing primary reactant {species:?} for inventory geochemistry reaction {}",
                reaction.name
            )
        });
    let operand = basis_operand_from_reactant(*term, None).unwrap_or_else(|| {
        panic!(
            "primary reactant {species:?} for inventory geochemistry reaction {} has no signal",
            reaction.name
        )
    });
    InventoryGeochemistryBasis::operand(operand)
}

fn basis_from_headroom(
    reaction: &'static InventoryReactionDefinition,
    species: TerrariumSpecies,
    capacity: f32,
    signal: InventoryGeochemistrySignal,
) -> InventoryGeochemistryBasis {
    let term = reaction
        .reactants
        .iter()
        .find(|term| term.species == species)
        .unwrap_or_else(|| {
            panic!(
                "missing headroom reactant {species:?} for inventory geochemistry reaction {}",
                reaction.name
            )
        });
    let operand = basis_operand_from_reactant(*term, None).unwrap_or_else(|| {
        panic!(
            "headroom reactant {species:?} for inventory geochemistry reaction {} has no signal",
            reaction.name
        )
    });
    InventoryGeochemistryBasis::min(
        operand,
        InventoryGeochemistryBasisOperand::headroom(capacity, signal),
    )
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct InventoryGeochemistryAffinity {
    pub bias: f32,
    pub min: f32,
    pub max: f32,
    pub proton: f32,
    pub water: f32,
    pub dissolved_silicate: f32,
    pub bicarbonate: f32,
    pub surface_proton_load: f32,
    pub calcium_bicarbonate_complex: f32,
    pub carbonate_mineral: f32,
    pub calcium: f32,
    pub aluminum: f32,
    pub aqueous_iron: f32,
    pub sorbed_aluminum_hydroxide: f32,
    pub sorbed_ferric_hydroxide: f32,
    pub base_saturation: f32,
    pub base_deficit: f32,
    pub alkalinity_gate: f32,
    pub oxygen_norm: f32,
    pub reactant_activity: f32,
    pub product_activity: f32,
}

impl InventoryGeochemistryAffinity {
    pub const ZERO: Self = Self {
        bias: 0.0,
        min: 0.0,
        max: 1.0,
        proton: 0.0,
        water: 0.0,
        dissolved_silicate: 0.0,
        bicarbonate: 0.0,
        surface_proton_load: 0.0,
        calcium_bicarbonate_complex: 0.0,
        carbonate_mineral: 0.0,
        calcium: 0.0,
        aluminum: 0.0,
        aqueous_iron: 0.0,
        sorbed_aluminum_hydroxide: 0.0,
        sorbed_ferric_hydroxide: 0.0,
        base_saturation: 0.0,
        base_deficit: 0.0,
        alkalinity_gate: 0.0,
        oxygen_norm: 0.0,
        reactant_activity: 0.0,
        product_activity: 0.0,
    };

    pub fn eval(
        self,
        state: InventoryGeochemistryState,
        reaction: Option<&'static InventoryReactionDefinition>,
    ) -> f32 {
        let mut total = self.bias;
        total += self.proton * state.normalized(InventoryGeochemistrySignal::Proton);
        total += self.water * state.normalized(InventoryGeochemistrySignal::Water);
        total += self.dissolved_silicate
            * state.normalized(InventoryGeochemistrySignal::DissolvedSilicate);
        total += self.bicarbonate * state.normalized(InventoryGeochemistrySignal::Bicarbonate);
        total += self.surface_proton_load
            * state.normalized(InventoryGeochemistrySignal::SurfaceProtonLoad);
        total += self.calcium_bicarbonate_complex
            * state.normalized(InventoryGeochemistrySignal::CalciumBicarbonateComplex);
        total += self.carbonate_mineral
            * state.normalized(InventoryGeochemistrySignal::CarbonateMineral);
        total += self.calcium * state.normalized(InventoryGeochemistrySignal::Calcium);
        total += self.aluminum * state.normalized(InventoryGeochemistrySignal::Aluminum);
        total += self.aqueous_iron * state.normalized(InventoryGeochemistrySignal::AqueousIron);
        total += self.sorbed_aluminum_hydroxide
            * state.normalized(InventoryGeochemistrySignal::SorbedAluminumHydroxide);
        total += self.sorbed_ferric_hydroxide
            * state.normalized(InventoryGeochemistrySignal::SorbedFerricHydroxide);
        total +=
            self.base_saturation * state.normalized(InventoryGeochemistrySignal::BaseSaturation);
        total += self.base_deficit
            * (1.0 - state.normalized(InventoryGeochemistrySignal::BaseSaturation));
        total +=
            self.alkalinity_gate * state.normalized(InventoryGeochemistrySignal::AlkalinityGate);
        total += self.oxygen_norm * state.normalized(InventoryGeochemistrySignal::OxygenNorm);
        if let Some(reaction) = reaction {
            total += self.reactant_activity * reaction_activity(state, reaction.reactants);
            total += self.product_activity * reaction_activity(state, reaction.products);
        }
        clamp(total, self.min, self.max)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct InventoryGeochemistryReactionRule {
    pub reaction: &'static InventoryReactionDefinition,
    pub basis: InventoryGeochemistryBasis,
    pub baseline_rate: f32,
    pub drive_scale: f32,
    pub standard_delta_g_ev: f32,
    pub affinity: InventoryGeochemistryAffinity,
}

impl InventoryGeochemistryReactionRule {
    pub fn proposed_extent(self, state: InventoryGeochemistryState, relaxation: f32) -> f32 {
        let thermodynamic_gate = thermodynamic_forward_gate(reaction_delta_g_ev(
            state,
            self.reaction,
            self.standard_delta_g_ev,
        ));
        self.basis.eval(state).max(0.0)
            * thermodynamic_gate
            * (self.baseline_rate.max(0.0)
                + self.affinity.eval(state, Some(self.reaction)).max(0.0) * self.drive_scale)
            * relaxation.max(0.0)
    }
}

#[derive(Debug, Clone, Copy)]
struct InventoryGeochemistryReactionProfile {
    baseline_rate: f32,
    drive_scale: f32,
    min: f32,
    max: f32,
    bias: f32,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InventoryGeochemistryReactionProcessKind {
    GasRelease,
    Mobilization,
    Immobilization,
    Association,
    Dissociation,
    SurfaceBinding,
    SurfaceRelease,
}

#[derive(Debug, Clone, Copy, Default)]
struct InventoryGeochemistryReactionDescriptor {
    solid_reactants: f32,
    solid_products: f32,
    amorphous_reactants: f32,
    amorphous_products: f32,
    interfacial_reactants: f32,
    interfacial_products: f32,
    gas_reactants: f32,
    gas_products: f32,
    proton_delta: f32,
    water_delta: f32,
    dissolved_silicate_delta: f32,
    bicarbonate_delta: f32,
    surface_proton_delta: f32,
    calcium_complex_delta: f32,
    carbonate_mineral_delta: f32,
    hydroxide_phase_delta: f32,
    ferric_hydroxide_delta: f32,
    standard_chemical_potential_delta_ev: f32,
    acidity_index_delta: f32,
    alkalinity_index_delta: f32,
    redox_index_delta: f32,
    complexation_index_delta: f32,
    reactant_hydration_index: f32,
    product_hydration_index: f32,
    reactant_cohesion_index: f32,
    product_cohesion_index: f32,
    reactant_surface_affinity_index: f32,
    product_surface_affinity_index: f32,
    reactant_volatility_index: f32,
    product_volatility_index: f32,
}

impl InventoryGeochemistryReactionDescriptor {
    fn solid_like_reactants(self) -> f32 {
        self.solid_reactants + self.amorphous_reactants
    }

    fn solid_like_products(self) -> f32 {
        self.solid_products + self.amorphous_products
    }

    fn net_solid_like_delta(self) -> f32 {
        self.solid_like_products() - self.solid_like_reactants()
    }

    fn hydration_index_delta(self) -> f32 {
        self.product_hydration_index - self.reactant_hydration_index
    }

    fn cohesion_index_delta(self) -> f32 {
        self.product_cohesion_index - self.reactant_cohesion_index
    }

    fn surface_affinity_index_delta(self) -> f32 {
        self.product_surface_affinity_index - self.reactant_surface_affinity_index
    }

    fn volatility_index_delta(self) -> f32 {
        self.product_volatility_index - self.reactant_volatility_index
    }
}

impl InventoryGeochemistryAffinity {
    fn add(self, other: Self) -> Self {
        Self {
            bias: self.bias + other.bias,
            min: self.min,
            max: self.max,
            proton: self.proton + other.proton,
            water: self.water + other.water,
            dissolved_silicate: self.dissolved_silicate + other.dissolved_silicate,
            bicarbonate: self.bicarbonate + other.bicarbonate,
            surface_proton_load: self.surface_proton_load + other.surface_proton_load,
            calcium_bicarbonate_complex: self.calcium_bicarbonate_complex
                + other.calcium_bicarbonate_complex,
            carbonate_mineral: self.carbonate_mineral + other.carbonate_mineral,
            calcium: self.calcium + other.calcium,
            aluminum: self.aluminum + other.aluminum,
            aqueous_iron: self.aqueous_iron + other.aqueous_iron,
            sorbed_aluminum_hydroxide: self.sorbed_aluminum_hydroxide
                + other.sorbed_aluminum_hydroxide,
            sorbed_ferric_hydroxide: self.sorbed_ferric_hydroxide + other.sorbed_ferric_hydroxide,
            base_saturation: self.base_saturation + other.base_saturation,
            base_deficit: self.base_deficit + other.base_deficit,
            alkalinity_gate: self.alkalinity_gate + other.alkalinity_gate,
            oxygen_norm: self.oxygen_norm + other.oxygen_norm,
            reactant_activity: self.reactant_activity + other.reactant_activity,
            product_activity: self.product_activity + other.product_activity,
        }
    }
}

fn affinity_from_signal_delta(
    signal: InventoryGeochemistrySignal,
    delta: f32,
) -> InventoryGeochemistryAffinity {
    let mut affinity = InventoryGeochemistryAffinity::ZERO;
    match signal {
        InventoryGeochemistrySignal::Water => affinity.water = delta,
        InventoryGeochemistrySignal::Proton => affinity.proton = delta,
        InventoryGeochemistrySignal::DissolvedSilicate => affinity.dissolved_silicate = delta,
        InventoryGeochemistrySignal::Bicarbonate => affinity.bicarbonate = delta,
        InventoryGeochemistrySignal::SurfaceProtonLoad => affinity.surface_proton_load = delta,
        InventoryGeochemistrySignal::CalciumBicarbonateComplex => {
            affinity.calcium_bicarbonate_complex = delta
        }
        InventoryGeochemistrySignal::SilicateMineral => {}
        InventoryGeochemistrySignal::CarbonateMineral => affinity.carbonate_mineral = delta,
        InventoryGeochemistrySignal::Calcium => affinity.calcium = delta,
        InventoryGeochemistrySignal::Aluminum => affinity.aluminum = delta,
        InventoryGeochemistrySignal::AqueousIron => affinity.aqueous_iron = delta,
        InventoryGeochemistrySignal::SorbedAluminumHydroxide => {
            affinity.sorbed_aluminum_hydroxide = delta
        }
        InventoryGeochemistrySignal::SorbedFerricHydroxide => {
            affinity.sorbed_ferric_hydroxide = delta
        }
        InventoryGeochemistrySignal::BaseSaturation => affinity.base_saturation = delta,
        InventoryGeochemistrySignal::AlkalinityGate => affinity.alkalinity_gate = delta,
        InventoryGeochemistrySignal::OxygenNorm => affinity.oxygen_norm = delta,
    }
    affinity
}

fn stoichiometric_affinity_delta(
    signal: InventoryGeochemistrySignal,
    stoich: f32,
    is_product: bool,
) -> InventoryGeochemistryAffinity {
    let magnitude = stoich.max(0.0);
    let signed = if is_product { -1.0 } else { 1.0 };
    let weight = stoichiometric_signal_weight(signal);
    affinity_from_signal_delta(signal, signed * weight * magnitude)
}

fn stoichiometric_signal_weight(signal: InventoryGeochemistrySignal) -> f32 {
    let Some(species) = species_for_signal(signal) else {
        return 0.0;
    };
    let Some(geochemistry_profile) = terrarium_geochemistry_species_profile(species) else {
        return 0.0;
    };
    let weight = 0.05
        + (0.22 / geochemistry_profile.activity_scale.max(0.06)).clamp(0.25, 3.0) * 0.05
        + geochemistry_profile.surface_affinity_index * 0.04
        + geochemistry_profile.cohesion_index * 0.04
        + geochemistry_profile.acidity_index * 0.03
        + geochemistry_profile.alkalinity_index * 0.02
        + geochemistry_profile.redox_index * 0.02
        + geochemistry_profile.complexation_index * 0.02
        + geochemistry_profile.volatility_index * 0.02;
    clamp(weight, 0.08, 0.30)
}

fn stoichiometric_reaction_affinity(
    reaction: &'static InventoryReactionDefinition,
) -> InventoryGeochemistryAffinity {
    let mut affinity = InventoryGeochemistryAffinity {
        reactant_activity: 0.22,
        product_activity: -0.18,
        ..InventoryGeochemistryAffinity::ZERO
    };
    for term in reaction.reactants {
        if let Some(signal) = signal_for_species(term.species) {
            affinity = affinity.add(stoichiometric_affinity_delta(
                signal,
                term.stoichiometry,
                false,
            ));
        }
    }
    for term in reaction.products {
        if let Some(signal) = signal_for_species(term.species) {
            affinity = affinity.add(stoichiometric_affinity_delta(
                signal,
                term.stoichiometry,
                true,
            ));
        }
    }
    affinity
}

fn accumulate_descriptor_term(
    descriptor: &mut InventoryGeochemistryReactionDescriptor,
    term: InventoryReactionTerm,
    is_product: bool,
) {
    let stoich = term.stoichiometry.max(0.0);
    let target = if is_product {
        match terrarium_inventory_species_profile(term.species) {
            Some(profile) => match profile.phase_kind {
                MaterialPhaseKind::Solid => &mut descriptor.solid_products,
                MaterialPhaseKind::Amorphous => &mut descriptor.amorphous_products,
                MaterialPhaseKind::Interfacial => &mut descriptor.interfacial_products,
                MaterialPhaseKind::Gas => &mut descriptor.gas_products,
                _ => return,
            },
            None => return,
        }
    } else {
        match terrarium_inventory_species_profile(term.species) {
            Some(profile) => match profile.phase_kind {
                MaterialPhaseKind::Solid => &mut descriptor.solid_reactants,
                MaterialPhaseKind::Amorphous => &mut descriptor.amorphous_reactants,
                MaterialPhaseKind::Interfacial => &mut descriptor.interfacial_reactants,
                MaterialPhaseKind::Gas => &mut descriptor.gas_reactants,
                _ => return,
            },
            None => return,
        }
    };
    *target += stoich;
}

fn accumulate_descriptor_signal_delta(
    descriptor: &mut InventoryGeochemistryReactionDescriptor,
    signal: InventoryGeochemistrySignal,
    delta: f32,
) {
    match signal {
        InventoryGeochemistrySignal::Water => descriptor.water_delta += delta,
        InventoryGeochemistrySignal::Proton => descriptor.proton_delta += delta,
        InventoryGeochemistrySignal::DissolvedSilicate => {
            descriptor.dissolved_silicate_delta += delta
        }
        InventoryGeochemistrySignal::Bicarbonate => descriptor.bicarbonate_delta += delta,
        InventoryGeochemistrySignal::SurfaceProtonLoad => descriptor.surface_proton_delta += delta,
        InventoryGeochemistrySignal::CalciumBicarbonateComplex => {
            descriptor.calcium_complex_delta += delta
        }
        InventoryGeochemistrySignal::CarbonateMineral => {
            descriptor.carbonate_mineral_delta += delta
        }
        InventoryGeochemistrySignal::SorbedAluminumHydroxide => {
            descriptor.hydroxide_phase_delta += delta
        }
        InventoryGeochemistrySignal::SorbedFerricHydroxide => {
            descriptor.ferric_hydroxide_delta += delta
        }
        InventoryGeochemistrySignal::SilicateMineral
        | InventoryGeochemistrySignal::Calcium
        | InventoryGeochemistrySignal::Aluminum
        | InventoryGeochemistrySignal::AqueousIron
        | InventoryGeochemistrySignal::BaseSaturation
        | InventoryGeochemistrySignal::AlkalinityGate
        | InventoryGeochemistrySignal::OxygenNorm => {}
    }
}

fn accumulate_descriptor_reactivity_indices(
    descriptor: &mut InventoryGeochemistryReactionDescriptor,
    species: TerrariumSpecies,
    stoich: f32,
    is_product: bool,
) {
    let Some(profile) = terrarium_geochemistry_species_profile(species) else {
        return;
    };
    let signed_stoich = if is_product { stoich } else { -stoich };
    descriptor.standard_chemical_potential_delta_ev +=
        signed_stoich * profile.standard_chemical_potential_ev;
    descriptor.acidity_index_delta += signed_stoich * profile.acidity_index;
    descriptor.alkalinity_index_delta += signed_stoich * profile.alkalinity_index;
    descriptor.redox_index_delta += signed_stoich * profile.redox_index;
    descriptor.complexation_index_delta += signed_stoich * profile.complexation_index;
    if is_product {
        descriptor.product_hydration_index += stoich * profile.hydration_index;
        descriptor.product_cohesion_index += stoich * profile.cohesion_index;
        descriptor.product_surface_affinity_index += stoich * profile.surface_affinity_index;
        descriptor.product_volatility_index += stoich * profile.volatility_index;
    } else {
        descriptor.reactant_hydration_index += stoich * profile.hydration_index;
        descriptor.reactant_cohesion_index += stoich * profile.cohesion_index;
        descriptor.reactant_surface_affinity_index += stoich * profile.surface_affinity_index;
        descriptor.reactant_volatility_index += stoich * profile.volatility_index;
    }
}

fn reaction_descriptor(
    reaction: &'static InventoryReactionDefinition,
) -> InventoryGeochemistryReactionDescriptor {
    let mut descriptor = InventoryGeochemistryReactionDescriptor::default();
    for term in reaction.reactants {
        accumulate_descriptor_term(&mut descriptor, *term, false);
        if let Some(signal) = signal_for_species(term.species) {
            accumulate_descriptor_signal_delta(&mut descriptor, signal, -term.stoichiometry);
        }
        accumulate_descriptor_reactivity_indices(
            &mut descriptor,
            term.species,
            term.stoichiometry.max(0.0),
            false,
        );
    }
    for term in reaction.products {
        accumulate_descriptor_term(&mut descriptor, *term, true);
        if let Some(signal) = signal_for_species(term.species) {
            accumulate_descriptor_signal_delta(&mut descriptor, signal, term.stoichiometry);
        }
        accumulate_descriptor_reactivity_indices(
            &mut descriptor,
            term.species,
            term.stoichiometry.max(0.0),
            true,
        );
    }
    descriptor
}

#[cfg(test)]
fn reaction_process_kind(
    reaction: &'static InventoryReactionDefinition,
    descriptor: InventoryGeochemistryReactionDescriptor,
) -> InventoryGeochemistryReactionProcessKind {
    if descriptor.gas_products > descriptor.gas_reactants + 1.0e-6 {
        InventoryGeochemistryReactionProcessKind::GasRelease
    } else if descriptor.surface_proton_delta > 1.0e-6 {
        InventoryGeochemistryReactionProcessKind::SurfaceBinding
    } else if descriptor.surface_proton_delta < -1.0e-6 {
        InventoryGeochemistryReactionProcessKind::SurfaceRelease
    } else if descriptor.net_solid_like_delta() < -1.0e-6 {
        InventoryGeochemistryReactionProcessKind::Mobilization
    } else if descriptor.net_solid_like_delta() > 1.0e-6 {
        InventoryGeochemistryReactionProcessKind::Immobilization
    } else if descriptor.interfacial_products > descriptor.interfacial_reactants + 1.0e-6
        && descriptor.proton_delta < -1.0e-6
    {
        InventoryGeochemistryReactionProcessKind::SurfaceBinding
    } else if descriptor.interfacial_reactants > descriptor.interfacial_products + 1.0e-6
        && descriptor.proton_delta > 1.0e-6
    {
        InventoryGeochemistryReactionProcessKind::SurfaceRelease
    } else if reaction.reactants.len() > reaction.products.len() {
        InventoryGeochemistryReactionProcessKind::Association
    } else {
        InventoryGeochemistryReactionProcessKind::Dissociation
    }
}

fn directional_drive(delta: f32, base: f32, gain: f32, cap: f32) -> f32 {
    if delta.abs() <= 1.0e-6 {
        0.0
    } else {
        -delta.signum() * (base + gain * delta.abs().min(cap))
    }
}

fn reaction_profile(
    descriptor: InventoryGeochemistryReactionDescriptor,
) -> InventoryGeochemistryReactionProfile {
    let transport_index = descriptor.product_hydration_index
        + descriptor.product_volatility_index
        + descriptor.product_surface_affinity_index * 0.5;
    let barrier_index = descriptor.reactant_cohesion_index
        + descriptor.product_cohesion_index * 0.5
        + descriptor.reactant_surface_affinity_index * 0.25;
    let mobility_gain = (descriptor.product_hydration_index + descriptor.product_volatility_index)
        - (descriptor.reactant_hydration_index + descriptor.reactant_volatility_index);
    let environment_index = descriptor.acidity_index_delta.abs()
        + descriptor.alkalinity_index_delta.abs()
        + descriptor.redox_index_delta.abs()
        + descriptor.complexation_index_delta.abs()
        + descriptor.net_solid_like_delta().abs() * 0.30
        + descriptor.standard_chemical_potential_delta_ev.abs() * 0.18;

    InventoryGeochemistryReactionProfile {
        baseline_rate: clamp(
            0.0009 + 0.0036 * transport_index / (1.0 + barrier_index),
            0.001,
            0.0055,
        ),
        drive_scale: clamp(
            0.010
                + 0.018 * (environment_index + mobility_gain.abs()) / (1.0 + barrier_index * 0.75),
            0.010,
            0.034,
        ),
        min: 0.0,
        max: clamp(
            1.0 + 0.12
                * (descriptor.product_surface_affinity_index
                    + descriptor.complexation_index_delta.abs()
                    + descriptor.product_volatility_index)
                    .min(1.4),
            1.0,
            1.2,
        ),
        bias: clamp(
            0.06 * descriptor.product_volatility_index
                + (-descriptor.standard_chemical_potential_delta_ev).max(0.0) * 0.03
                + 0.04 * mobility_gain.max(0.0) / (1.0 + barrier_index),
            0.0,
            0.12,
        ),
    }
}

fn reaction_environmental_affinity(
    descriptor: InventoryGeochemistryReactionDescriptor,
    profile: InventoryGeochemistryReactionProfile,
) -> InventoryGeochemistryAffinity {
    let mut affinity = InventoryGeochemistryAffinity {
        bias: profile.bias,
        ..InventoryGeochemistryAffinity::ZERO
    };

    let proton_stoich = directional_drive(descriptor.proton_delta, 0.04, 0.02, 3.0);
    let water_stoich = directional_drive(descriptor.water_delta, 0.025, 0.01, 3.0);
    let solid_mobilization = (-descriptor.net_solid_like_delta()).max(0.0);
    let hydration_delta = descriptor.hydration_index_delta();
    let cohesion_delta = descriptor.cohesion_index_delta();
    let surface_delta = descriptor.surface_affinity_index_delta();
    let volatility_delta = descriptor.volatility_index_delta();
    let retention_delta =
        cohesion_delta + surface_delta * 0.7 + descriptor.net_solid_like_delta() * 0.8;

    affinity.proton = clamp(
        proton_stoich
            + (descriptor.acidity_index_delta - descriptor.alkalinity_index_delta * 0.65
                + solid_mobilization * 0.52
                + (-cohesion_delta).max(0.0) * 0.24
                + hydration_delta.max(0.0) * 0.14)
                * 0.22,
        -0.26,
        0.26,
    );
    affinity.water = clamp(
        water_stoich + (hydration_delta - volatility_delta * 0.45 - cohesion_delta * 0.20) * 0.20,
        -0.18,
        0.18,
    );
    if retention_delta > 1.0e-6 {
        affinity.base_saturation = clamp(retention_delta * 0.18, 0.0, 0.22);
    } else if retention_delta < -1.0e-6 {
        affinity.base_deficit = clamp(-retention_delta * 0.20, 0.0, 0.24);
    }
    affinity.surface_proton_load = clamp(
        -surface_delta * 0.16
            - descriptor.acidity_index_delta * 0.05
            - descriptor.complexation_index_delta * 0.14,
        -0.22,
        0.22,
    );
    affinity.calcium_bicarbonate_complex =
        clamp(-descriptor.complexation_index_delta * 0.18, -0.18, 0.18);
    affinity.alkalinity_gate = clamp(
        (descriptor.alkalinity_index_delta - descriptor.acidity_index_delta * 0.55
            + cohesion_delta * 0.20
            + descriptor.complexation_index_delta * 0.25)
            * 0.22,
        -0.22,
        0.22,
    );
    affinity.oxygen_norm = clamp(
        (descriptor.redox_index_delta + cohesion_delta * 0.45 - hydration_delta * 0.35
            + descriptor.ferric_hydroxide_delta * 0.55
            + volatility_delta * 0.10)
            * 0.22,
        -0.18,
        0.18,
    );

    affinity
}

fn build_reaction_rule(
    reaction: &'static InventoryReactionDefinition,
    basis: InventoryGeochemistryBasis,
) -> InventoryGeochemistryReactionRule {
    let descriptor = reaction_descriptor(reaction);
    let standard_delta_g_ev = descriptor.standard_chemical_potential_delta_ev;
    let profile = reaction_profile(descriptor);
    let mut affinity = stoichiometric_reaction_affinity(reaction)
        .add(reaction_environmental_affinity(descriptor, profile));
    affinity.min = profile.min;
    affinity.max = profile.max;
    InventoryGeochemistryReactionRule {
        reaction,
        basis,
        baseline_rate: profile.baseline_rate,
        drive_scale: profile.drive_scale,
        standard_delta_g_ev,
        affinity,
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct InventoryGeochemistryPartitionRule {
    pub species: TerrariumSpecies,
    pub affinity: InventoryGeochemistryAffinity,
}

impl InventoryGeochemistryPartitionRule {
    pub fn target_fraction(self, state: InventoryGeochemistryState) -> f32 {
        self.affinity.eval(state, None)
    }
}

fn build_partition_rule(species: TerrariumSpecies) -> Option<InventoryGeochemistryPartitionRule> {
    let profile = terrarium_exchange_partition_profile(species)?;
    let mut affinity = InventoryGeochemistryAffinity {
        proton: profile.proton_drive,
        surface_proton_load: profile.surface_proton_drive,
        water: profile.water_drive,
        base_deficit: profile.base_deficit_drive,
        ..InventoryGeochemistryAffinity::ZERO
    };
    affinity.bias = profile.dissolved_bias;
    affinity.min = profile.dissolved_min;
    affinity.max = profile.dissolved_max;
    affinity.proton += profile.acidity_delta;
    affinity.surface_proton_load += profile.surface_competition_delta;
    affinity.water += profile.hydration_delta;
    affinity.base_saturation += profile.base_saturation_delta;
    affinity.dissolved_silicate += profile.silicate_mobility;
    affinity.carbonate_mineral -= profile.carbonate_pairing;
    affinity.calcium_bicarbonate_complex -= profile.bicarbonate_complex_pairing;
    Some(InventoryGeochemistryPartitionRule { species, affinity })
}

const SILICATE_DISSOLUTION_REACTION: InventoryReactionDefinition = InventoryReactionDefinition::new(
    "silicate_dissolution",
    &[
        InventoryReactionTerm::new(TerrariumSpecies::SilicateMineral, 1.0),
        InventoryReactionTerm::new(TerrariumSpecies::Water, 2.0),
    ],
    &[InventoryReactionTerm::new(
        TerrariumSpecies::DissolvedSilicate,
        1.0,
    )],
);

const SILICATE_PRECIPITATION_REACTION: InventoryReactionDefinition =
    InventoryReactionDefinition::new(
        "silicate_precipitation",
        &[InventoryReactionTerm::new(
            TerrariumSpecies::DissolvedSilicate,
            1.0,
        )],
        &[
            InventoryReactionTerm::new(TerrariumSpecies::SilicateMineral, 1.0),
            InventoryReactionTerm::new(TerrariumSpecies::Water, 2.0),
        ],
    );

const CARBONATE_DISSOLUTION_REACTION: InventoryReactionDefinition =
    InventoryReactionDefinition::new(
        "carbonate_dissolution",
        &[
            InventoryReactionTerm::new(TerrariumSpecies::CarbonateMineral, 1.0),
            InventoryReactionTerm::new(TerrariumSpecies::Proton, 1.0),
        ],
        &[
            InventoryReactionTerm::new(TerrariumSpecies::BicarbonatePool, 1.0),
            InventoryReactionTerm::new(TerrariumSpecies::ExchangeableCalcium, 1.0),
        ],
    );

const CARBONATE_PRECIPITATION_REACTION: InventoryReactionDefinition =
    InventoryReactionDefinition::new(
        "carbonate_precipitation",
        &[
            InventoryReactionTerm::new(TerrariumSpecies::BicarbonatePool, 1.0),
            InventoryReactionTerm::new(TerrariumSpecies::ExchangeableCalcium, 1.0),
        ],
        &[
            InventoryReactionTerm::new(TerrariumSpecies::CarbonateMineral, 1.0),
            InventoryReactionTerm::new(TerrariumSpecies::Proton, 1.0),
        ],
    );

const BICARBONATE_DEGASSING_REACTION: InventoryReactionDefinition =
    InventoryReactionDefinition::new(
        "bicarbonate_degassing",
        &[
            InventoryReactionTerm::new(TerrariumSpecies::BicarbonatePool, 1.0),
            InventoryReactionTerm::new(TerrariumSpecies::Proton, 1.0),
        ],
        &[
            InventoryReactionTerm::new(TerrariumSpecies::CarbonDioxide, 1.0),
            InventoryReactionTerm::new(TerrariumSpecies::Water, 1.0),
        ],
    );

const PROTON_SURFACE_SORPTION_REACTION: InventoryReactionDefinition =
    InventoryReactionDefinition::new(
        "proton_surface_sorption",
        &[InventoryReactionTerm::new(TerrariumSpecies::Proton, 1.0)],
        &[InventoryReactionTerm::new(
            TerrariumSpecies::SurfaceProtonLoad,
            1.0,
        )],
    );

const PROTON_SURFACE_DESORPTION_REACTION: InventoryReactionDefinition =
    InventoryReactionDefinition::new(
        "proton_surface_desorption",
        &[InventoryReactionTerm::new(
            TerrariumSpecies::SurfaceProtonLoad,
            1.0,
        )],
        &[InventoryReactionTerm::new(TerrariumSpecies::Proton, 1.0)],
    );

const CALCIUM_BICARBONATE_COMPLEXATION_REACTION: InventoryReactionDefinition =
    InventoryReactionDefinition::new(
        "calcium_bicarbonate_complexation",
        &[
            InventoryReactionTerm::new(TerrariumSpecies::ExchangeableCalcium, 1.0),
            InventoryReactionTerm::new(TerrariumSpecies::BicarbonatePool, 2.0),
        ],
        &[InventoryReactionTerm::new(
            TerrariumSpecies::CalciumBicarbonateComplex,
            1.0,
        )],
    );

const CALCIUM_BICARBONATE_DISSOCIATION_REACTION: InventoryReactionDefinition =
    InventoryReactionDefinition::new(
        "calcium_bicarbonate_dissociation",
        &[InventoryReactionTerm::new(
            TerrariumSpecies::CalciumBicarbonateComplex,
            1.0,
        )],
        &[
            InventoryReactionTerm::new(TerrariumSpecies::ExchangeableCalcium, 1.0),
            InventoryReactionTerm::new(TerrariumSpecies::BicarbonatePool, 2.0),
        ],
    );

const ALUMINUM_HYDROXIDE_PRECIPITATION_REACTION: InventoryReactionDefinition =
    InventoryReactionDefinition::new(
        "aluminum_hydroxide_precipitation",
        &[
            InventoryReactionTerm::new(TerrariumSpecies::ExchangeableAluminum, 1.0),
            InventoryReactionTerm::new(TerrariumSpecies::Water, 3.0),
        ],
        &[
            InventoryReactionTerm::new(TerrariumSpecies::SorbedAluminumHydroxide, 1.0),
            InventoryReactionTerm::new(TerrariumSpecies::Proton, 3.0),
        ],
    );

const ALUMINUM_HYDROXIDE_DISSOLUTION_REACTION: InventoryReactionDefinition =
    InventoryReactionDefinition::new(
        "aluminum_hydroxide_dissolution",
        &[
            InventoryReactionTerm::new(TerrariumSpecies::SorbedAluminumHydroxide, 1.0),
            InventoryReactionTerm::new(TerrariumSpecies::Proton, 3.0),
        ],
        &[
            InventoryReactionTerm::new(TerrariumSpecies::ExchangeableAluminum, 1.0),
            InventoryReactionTerm::new(TerrariumSpecies::Water, 3.0),
        ],
    );

const FERRIC_HYDROXIDE_PRECIPITATION_REACTION: InventoryReactionDefinition =
    InventoryReactionDefinition::new(
        "ferric_hydroxide_precipitation",
        &[
            InventoryReactionTerm::new(TerrariumSpecies::AqueousIronPool, 1.0),
            InventoryReactionTerm::new(TerrariumSpecies::Water, 3.0),
        ],
        &[
            InventoryReactionTerm::new(TerrariumSpecies::SorbedFerricHydroxide, 1.0),
            InventoryReactionTerm::new(TerrariumSpecies::Proton, 3.0),
        ],
    );

const FERRIC_HYDROXIDE_DISSOLUTION_REACTION: InventoryReactionDefinition =
    InventoryReactionDefinition::new(
        "ferric_hydroxide_dissolution",
        &[
            InventoryReactionTerm::new(TerrariumSpecies::SorbedFerricHydroxide, 1.0),
            InventoryReactionTerm::new(TerrariumSpecies::Proton, 3.0),
        ],
        &[
            InventoryReactionTerm::new(TerrariumSpecies::AqueousIronPool, 1.0),
            InventoryReactionTerm::new(TerrariumSpecies::Water, 3.0),
        ],
    );

pub(crate) fn inventory_geochemistry_reaction_rules() -> &'static [InventoryGeochemistryReactionRule]
{
    static RULES: OnceLock<Vec<InventoryGeochemistryReactionRule>> = OnceLock::new();
    RULES
        .get_or_init(|| {
            vec![
                build_reaction_rule(
                    &SILICATE_DISSOLUTION_REACTION,
                    basis_from_reaction_reactants(&SILICATE_DISSOLUTION_REACTION, &[]),
                ),
                build_reaction_rule(
                    &SILICATE_PRECIPITATION_REACTION,
                    basis_from_primary_reactant(
                        &SILICATE_PRECIPITATION_REACTION,
                        TerrariumSpecies::DissolvedSilicate,
                    ),
                ),
                build_reaction_rule(
                    &CARBONATE_DISSOLUTION_REACTION,
                    basis_from_reaction_reactants(
                        &CARBONATE_DISSOLUTION_REACTION,
                        &[(TerrariumSpecies::Proton, 0.98)],
                    ),
                ),
                build_reaction_rule(
                    &CARBONATE_PRECIPITATION_REACTION,
                    basis_from_reaction_reactants(&CARBONATE_PRECIPITATION_REACTION, &[]),
                ),
                build_reaction_rule(
                    &BICARBONATE_DEGASSING_REACTION,
                    basis_from_primary_reactant(
                        &BICARBONATE_DEGASSING_REACTION,
                        TerrariumSpecies::BicarbonatePool,
                    ),
                ),
                build_reaction_rule(
                    &PROTON_SURFACE_SORPTION_REACTION,
                    basis_from_headroom(
                        &PROTON_SURFACE_SORPTION_REACTION,
                        TerrariumSpecies::Proton,
                        0.18,
                        InventoryGeochemistrySignal::SurfaceProtonLoad,
                    ),
                ),
                build_reaction_rule(
                    &PROTON_SURFACE_DESORPTION_REACTION,
                    basis_from_primary_reactant(
                        &PROTON_SURFACE_DESORPTION_REACTION,
                        TerrariumSpecies::SurfaceProtonLoad,
                    ),
                ),
                build_reaction_rule(
                    &CALCIUM_BICARBONATE_COMPLEXATION_REACTION,
                    basis_from_reaction_reactants(&CALCIUM_BICARBONATE_COMPLEXATION_REACTION, &[]),
                ),
                build_reaction_rule(
                    &CALCIUM_BICARBONATE_DISSOCIATION_REACTION,
                    basis_from_primary_reactant(
                        &CALCIUM_BICARBONATE_DISSOCIATION_REACTION,
                        TerrariumSpecies::CalciumBicarbonateComplex,
                    ),
                ),
                build_reaction_rule(
                    &ALUMINUM_HYDROXIDE_PRECIPITATION_REACTION,
                    basis_from_reaction_reactants(&ALUMINUM_HYDROXIDE_PRECIPITATION_REACTION, &[]),
                ),
                build_reaction_rule(
                    &ALUMINUM_HYDROXIDE_DISSOLUTION_REACTION,
                    basis_from_reaction_reactants(&ALUMINUM_HYDROXIDE_DISSOLUTION_REACTION, &[]),
                ),
                build_reaction_rule(
                    &FERRIC_HYDROXIDE_PRECIPITATION_REACTION,
                    basis_from_reaction_reactants(&FERRIC_HYDROXIDE_PRECIPITATION_REACTION, &[]),
                ),
                build_reaction_rule(
                    &FERRIC_HYDROXIDE_DISSOLUTION_REACTION,
                    basis_from_reaction_reactants(&FERRIC_HYDROXIDE_DISSOLUTION_REACTION, &[]),
                ),
            ]
        })
        .as_slice()
}

pub(crate) fn inventory_geochemistry_partition_rules(
) -> &'static [InventoryGeochemistryPartitionRule] {
    static RULES: OnceLock<Vec<InventoryGeochemistryPartitionRule>> = OnceLock::new();
    RULES
        .get_or_init(|| {
            EXCHANGEABLE_TERRARIUM_GEOCHEMISTRY_SPECIES
                .iter()
                .filter_map(|&species| build_partition_rule(species))
                .collect()
        })
        .as_slice()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn neutral_partition_state() -> InventoryGeochemistryState {
        InventoryGeochemistryState {
            water: 0.0,
            proton: 0.0,
            dissolved_silicate: 0.0,
            bicarbonate: 0.0,
            surface_proton_load: 0.0,
            calcium_bicarbonate_complex: 0.0,
            silicate_mineral: 0.0,
            carbonate_mineral: 0.0,
            calcium: 0.0,
            aluminum: 0.0,
            aqueous_iron: 0.0,
            sorbed_aluminum_hydroxide: 0.0,
            sorbed_ferric_hydroxide: 0.0,
            base_saturation: 0.5,
            alkalinity_gate: 0.0,
            oxygen_norm: 0.0,
        }
    }

    #[test]
    fn partition_rules_are_generated_from_exchange_profiles() {
        let rules = inventory_geochemistry_partition_rules();
        assert_eq!(
            rules.len(),
            EXCHANGEABLE_TERRARIUM_GEOCHEMISTRY_SPECIES.len()
        );

        let calcium = rules
            .iter()
            .find(|rule| rule.species == TerrariumSpecies::ExchangeableCalcium)
            .expect("calcium partition rule should exist");
        let magnesium = rules
            .iter()
            .find(|rule| rule.species == TerrariumSpecies::ExchangeableMagnesium)
            .expect("magnesium partition rule should exist");
        let sodium = rules
            .iter()
            .find(|rule| rule.species == TerrariumSpecies::ExchangeableSodium)
            .expect("sodium partition rule should exist");
        let aluminum = rules
            .iter()
            .find(|rule| rule.species == TerrariumSpecies::ExchangeableAluminum)
            .expect("aluminum partition rule should exist");

        let neutral = neutral_partition_state();
        assert!(sodium.target_fraction(neutral) > calcium.target_fraction(neutral));
        assert!(magnesium.affinity.base_saturation < 0.0);
        assert!(calcium.affinity.carbonate_mineral < 0.0);

        let acidic = InventoryGeochemistryState {
            proton: 0.18,
            surface_proton_load: 0.18,
            base_saturation: 0.12,
            ..neutral
        };
        assert!(aluminum.target_fraction(acidic) > aluminum.target_fraction(neutral));
    }

    #[test]
    fn reaction_profiles_follow_descriptor_semantics() {
        let weathering_descriptor = reaction_descriptor(&SILICATE_DISSOLUTION_REACTION);
        let weathering_process =
            reaction_process_kind(&SILICATE_DISSOLUTION_REACTION, weathering_descriptor);
        let weathering = reaction_environmental_affinity(
            weathering_descriptor,
            reaction_profile(weathering_descriptor),
        );

        let precipitation_descriptor = reaction_descriptor(&CARBONATE_PRECIPITATION_REACTION);
        let precipitation_process =
            reaction_process_kind(&CARBONATE_PRECIPITATION_REACTION, precipitation_descriptor);
        let precipitation = reaction_environmental_affinity(
            precipitation_descriptor,
            reaction_profile(precipitation_descriptor),
        );

        let ferric_descriptor = reaction_descriptor(&FERRIC_HYDROXIDE_DISSOLUTION_REACTION);
        let ferric_process =
            reaction_process_kind(&FERRIC_HYDROXIDE_DISSOLUTION_REACTION, ferric_descriptor);
        let ferric_dissolution =
            reaction_environmental_affinity(ferric_descriptor, reaction_profile(ferric_descriptor));

        assert_eq!(
            weathering_process,
            InventoryGeochemistryReactionProcessKind::Mobilization
        );
        assert!(weathering.proton > 0.0);
        assert!(weathering.base_deficit > 0.0);
        assert_eq!(
            precipitation_process,
            InventoryGeochemistryReactionProcessKind::Immobilization
        );
        assert!(precipitation.proton < 0.0);
        assert!(precipitation.base_saturation > 0.0);
        assert_eq!(
            ferric_process,
            InventoryGeochemistryReactionProcessKind::Mobilization
        );
        assert!(ferric_dissolution.oxygen_norm < 0.0);
    }

    #[test]
    fn reaction_thermodynamics_follow_standard_state_and_mass_action() {
        let silicate_standard = reaction_descriptor(&SILICATE_DISSOLUTION_REACTION)
            .standard_chemical_potential_delta_ev;
        let precipitation_standard = reaction_descriptor(&CARBONATE_PRECIPITATION_REACTION)
            .standard_chemical_potential_delta_ev;

        let silicate_undersaturated = InventoryGeochemistryState {
            water: 1.0,
            dissolved_silicate: 0.0,
            ..neutral_partition_state()
        };
        let silicate_saturated = InventoryGeochemistryState {
            water: 1.0,
            dissolved_silicate: 0.28,
            ..neutral_partition_state()
        };
        let precipitation_reactant_rich = InventoryGeochemistryState {
            bicarbonate: 0.32,
            calcium: 0.24,
            proton: 0.04,
            ..neutral_partition_state()
        };
        let precipitation_reactant_poor = InventoryGeochemistryState {
            bicarbonate: 0.02,
            calcium: 0.03,
            proton: 0.04,
            ..neutral_partition_state()
        };

        assert!(silicate_standard.is_finite());
        assert!(precipitation_standard.is_finite());
        assert!(
            reaction_delta_g_ev(
                silicate_undersaturated,
                &SILICATE_DISSOLUTION_REACTION,
                silicate_standard,
            ) < reaction_delta_g_ev(
                silicate_saturated,
                &SILICATE_DISSOLUTION_REACTION,
                silicate_standard,
            )
        );
        assert!(
            reaction_delta_g_ev(
                precipitation_reactant_rich,
                &CARBONATE_PRECIPITATION_REACTION,
                precipitation_standard,
            ) < reaction_delta_g_ev(
                precipitation_reactant_poor,
                &CARBONATE_PRECIPITATION_REACTION,
                precipitation_standard,
            )
        );
        assert!(
            thermodynamic_forward_gate(reaction_delta_g_ev(
                precipitation_reactant_rich,
                &CARBONATE_PRECIPITATION_REACTION,
                precipitation_standard,
            )) > thermodynamic_forward_gate(reaction_delta_g_ev(
                precipitation_reactant_poor,
                &CARBONATE_PRECIPITATION_REACTION,
                precipitation_standard,
            ))
        );
    }

    #[test]
    fn stoichiometric_signal_weights_follow_species_chemistry_metadata() {
        let water = stoichiometric_signal_weight(InventoryGeochemistrySignal::Water);
        let proton = stoichiometric_signal_weight(InventoryGeochemistrySignal::Proton);
        let bicarbonate = stoichiometric_signal_weight(InventoryGeochemistrySignal::Bicarbonate);
        let dissolved_silicate =
            stoichiometric_signal_weight(InventoryGeochemistrySignal::DissolvedSilicate);
        let calcium = stoichiometric_signal_weight(InventoryGeochemistrySignal::Calcium);

        assert!(proton > water);
        assert!(bicarbonate > water);
        assert!(dissolved_silicate > water);
        assert!(calcium > water);
        assert_eq!(
            stoichiometric_signal_weight(InventoryGeochemistrySignal::BaseSaturation),
            0.0
        );
    }

    #[test]
    fn reaction_basis_derivation_respects_stoichiometry_and_exceptions() {
        let complexation =
            basis_from_reaction_reactants(&CALCIUM_BICARBONATE_COMPLEXATION_REACTION, &[]);
        let weathering = basis_from_reaction_reactants(
            &CARBONATE_DISSOLUTION_REACTION,
            &[(TerrariumSpecies::Proton, 0.98)],
        );
        let sorption = basis_from_headroom(
            &PROTON_SURFACE_SORPTION_REACTION,
            TerrariumSpecies::Proton,
            0.18,
            InventoryGeochemistrySignal::SurfaceProtonLoad,
        );
        let state = InventoryGeochemistryState {
            water: 0.0,
            proton: 0.20,
            dissolved_silicate: 0.0,
            bicarbonate: 0.30,
            surface_proton_load: 0.10,
            calcium_bicarbonate_complex: 0.0,
            silicate_mineral: 0.0,
            carbonate_mineral: 0.25,
            calcium: 0.24,
            aluminum: 0.0,
            aqueous_iron: 0.0,
            sorbed_aluminum_hydroxide: 0.0,
            sorbed_ferric_hydroxide: 0.0,
            base_saturation: 0.5,
            alkalinity_gate: 0.0,
            oxygen_norm: 0.0,
        };

        assert!((complexation.eval(state) - 0.15).abs() < 1.0e-6);
        assert!((weathering.eval(state) - 0.196).abs() < 1.0e-6);
        assert!((sorption.eval(state) - 0.08).abs() < 1.0e-6);
    }
}
