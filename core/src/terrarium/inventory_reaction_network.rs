use super::*;
use crate::terrarium::material_exchange::{
    deposit_species_to_inventory, inventory_component_amount, withdraw_species_from_inventory,
};

#[derive(Debug, Clone, Copy)]
pub(crate) struct InventoryReactionTerm {
    pub species: TerrariumSpecies,
    pub stoichiometry: f32,
}

impl InventoryReactionTerm {
    pub const fn new(species: TerrariumSpecies, stoichiometry: f32) -> Self {
        Self {
            species,
            stoichiometry,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct InventoryReactionDefinition {
    #[allow(dead_code)]
    pub name: &'static str,
    pub reactants: &'static [InventoryReactionTerm],
    pub products: &'static [InventoryReactionTerm],
}

impl InventoryReactionDefinition {
    pub const fn new(
        name: &'static str,
        reactants: &'static [InventoryReactionTerm],
        products: &'static [InventoryReactionTerm],
    ) -> Self {
        Self {
            name,
            reactants,
            products,
        }
    }
}

pub(crate) fn reaction_extent_limit(
    inventory: &RegionalMaterialInventory,
    reaction: &InventoryReactionDefinition,
) -> f32 {
    if reaction.reactants.is_empty() {
        return 0.0;
    }
    reaction
        .reactants
        .iter()
        .filter_map(|term| {
            if term.stoichiometry <= 1.0e-9 {
                None
            } else {
                Some(
                    inventory_component_amount(inventory, term.species)
                        / term.stoichiometry.max(1.0e-9),
                )
            }
        })
        .fold(f32::INFINITY, f32::min)
        .max(0.0)
}

pub(crate) fn apply_inventory_reaction(
    inventory: &mut RegionalMaterialInventory,
    reaction: &InventoryReactionDefinition,
    proposed_extent: f32,
) -> f32 {
    let extent = proposed_extent
        .max(0.0)
        .min(reaction_extent_limit(inventory, reaction));
    if extent <= 1.0e-9 {
        return 0.0;
    }
    for term in reaction.reactants {
        let amount = extent * term.stoichiometry.max(0.0);
        let removed = withdraw_species_from_inventory(inventory, term.species, amount);
        if removed + 1.0e-6 < amount {
            return 0.0;
        }
    }
    for term in reaction.products {
        let amount = extent * term.stoichiometry.max(0.0);
        if amount > 1.0e-9 {
            deposit_species_to_inventory(inventory, term.species, amount);
        }
    }
    extent
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reaction_executor_conservatively_limits_by_reactants() {
        let mut inventory = RegionalMaterialInventory::new("rxn:test".into());
        deposit_species_to_inventory(&mut inventory, TerrariumSpecies::BicarbonatePool, 0.30);
        deposit_species_to_inventory(&mut inventory, TerrariumSpecies::ExchangeableCalcium, 0.10);

        const RXN: InventoryReactionDefinition = InventoryReactionDefinition::new(
            "complex",
            &[
                InventoryReactionTerm::new(TerrariumSpecies::ExchangeableCalcium, 1.0),
                InventoryReactionTerm::new(TerrariumSpecies::BicarbonatePool, 2.0),
            ],
            &[InventoryReactionTerm::new(
                TerrariumSpecies::CalciumBicarbonateComplex,
                1.0,
            )],
        );

        let extent = apply_inventory_reaction(&mut inventory, &RXN, 0.50);
        assert!((extent - 0.10).abs() < 1.0e-5);
        assert!(
            inventory_component_amount(&inventory, TerrariumSpecies::CalciumBicarbonateComplex)
                > 0.09
        );
    }
}
