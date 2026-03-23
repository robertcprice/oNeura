use super::*;
use crate::terrarium::geochemistry::terrarium_exchange_dissolved_binding;
use crate::terrarium::inventory_geochemistry_registry::{
    inventory_geochemistry_partition_rules, inventory_geochemistry_reaction_rules,
    InventoryGeochemistryState,
};

fn canonical_binding(
    species: TerrariumSpecies,
) -> Option<(
    MaterialRegionKind,
    MoleculeDescriptor,
    MaterialPhaseDescriptor,
)> {
    crate::terrarium::material_exchange::species_inventory_binding(species)
}

fn internal_dissolved_binding(
    species: TerrariumSpecies,
) -> Option<(
    MaterialRegionKind,
    MoleculeDescriptor,
    MaterialPhaseDescriptor,
)> {
    terrarium_exchange_dissolved_binding(species)
}

pub(crate) fn inventory_species_total(
    inventory: &RegionalMaterialInventory,
    species: TerrariumSpecies,
) -> f32 {
    let Some((region, molecule, phase)) = canonical_binding(species) else {
        return 0.0;
    };
    let mut total = inventory.total_amount_for_component(
        region,
        &molecule,
        &MaterialPhaseSelector::Exact(phase),
    );
    if let Some((internal_region, internal_molecule, internal_phase)) =
        internal_dissolved_binding(species)
    {
        total += inventory.total_amount_for_component(
            internal_region,
            &internal_molecule,
            &MaterialPhaseSelector::Exact(internal_phase),
        );
    }
    total
}

pub(crate) fn withdraw_inventory_species_total(
    inventory: &mut RegionalMaterialInventory,
    species: TerrariumSpecies,
    amount: f32,
) -> f32 {
    let target = amount.max(0.0);
    if target <= 1.0e-12 {
        return 0.0;
    }
    let mut removed = 0.0;
    if let Some((internal_region, internal_molecule, _internal_phase)) =
        internal_dissolved_binding(species)
    {
        removed +=
            inventory.withdraw_component(internal_region, &internal_molecule, target as f64) as f32;
    }
    if removed + 1.0e-12 < target {
        let Some((region, molecule, _phase)) = canonical_binding(species) else {
            return removed;
        };
        removed +=
            inventory.withdraw_component(region, &molecule, (target - removed) as f64) as f32;
    }
    removed
}

fn exact_amount(
    inventory: &RegionalMaterialInventory,
    region: MaterialRegionKind,
    molecule: &MoleculeDescriptor,
    phase: MaterialPhaseDescriptor,
) -> f32 {
    inventory.total_amount_for_component(region, molecule, &MaterialPhaseSelector::Exact(phase))
}

fn transfer_component(
    inventory: &mut RegionalMaterialInventory,
    from_region: MaterialRegionKind,
    from_molecule: &MoleculeDescriptor,
    to_region: MaterialRegionKind,
    to_molecule: &MoleculeDescriptor,
    amount: f32,
    to_phase: MaterialPhaseDescriptor,
) -> f32 {
    let removed =
        inventory.withdraw_component(from_region, from_molecule, amount.max(0.0) as f64) as f32;
    if removed > 1.0e-12 {
        inventory.deposit_component(to_region, to_molecule.clone(), removed as f64, to_phase);
    }
    removed
}

fn relax_exchange_partition(
    inventory: &mut RegionalMaterialInventory,
    species: TerrariumSpecies,
    target_dissolved_fraction: f32,
    relaxation: f32,
) {
    let Some((surface_region, surface_molecule, surface_phase)) = canonical_binding(species) else {
        return;
    };
    let Some((pore_region, pore_molecule, pore_phase)) = internal_dissolved_binding(species) else {
        return;
    };
    let surface_amount = exact_amount(
        inventory,
        surface_region,
        &surface_molecule,
        surface_phase.clone(),
    );
    let dissolved_amount = exact_amount(inventory, pore_region, &pore_molecule, pore_phase.clone());
    let total = surface_amount + dissolved_amount;
    if total <= 1.0e-9 {
        return;
    }
    let target_dissolved = total * target_dissolved_fraction.clamp(0.0, 1.0);
    let next_dissolved =
        dissolved_amount + (target_dissolved - dissolved_amount) * relaxation.clamp(0.0, 1.0);
    if next_dissolved > dissolved_amount + 1.0e-9 {
        transfer_component(
            inventory,
            surface_region,
            &surface_molecule,
            pore_region,
            &pore_molecule,
            next_dissolved - dissolved_amount,
            pore_phase,
        );
    } else if dissolved_amount > next_dissolved + 1.0e-9 {
        transfer_component(
            inventory,
            pore_region,
            &pore_molecule,
            surface_region,
            &surface_molecule,
            dissolved_amount - next_dissolved,
            surface_phase,
        );
    }
}

pub(crate) fn equilibrate_inventory_geochemistry(
    inventory: &mut RegionalMaterialInventory,
    relaxation: f32,
) {
    let relaxation = relaxation.clamp(0.0, 1.0);
    if relaxation <= 1.0e-6 {
        return;
    }

    let water = inventory_species_total(inventory, TerrariumSpecies::Water);
    let proton = inventory_species_total(inventory, TerrariumSpecies::Proton);
    let dissolved_silicate =
        inventory_species_total(inventory, TerrariumSpecies::DissolvedSilicate);
    let bicarbonate = inventory_species_total(inventory, TerrariumSpecies::BicarbonatePool);
    let surface_proton_load =
        inventory_species_total(inventory, TerrariumSpecies::SurfaceProtonLoad);
    let calcium_bicarbonate_complex =
        inventory_species_total(inventory, TerrariumSpecies::CalciumBicarbonateComplex);
    let silicate_mineral = inventory_species_total(inventory, TerrariumSpecies::SilicateMineral);
    let carbonate_mineral = inventory_species_total(inventory, TerrariumSpecies::CarbonateMineral);
    let calcium = inventory_species_total(inventory, TerrariumSpecies::ExchangeableCalcium);
    let magnesium = inventory_species_total(inventory, TerrariumSpecies::ExchangeableMagnesium);
    let potassium = inventory_species_total(inventory, TerrariumSpecies::ExchangeablePotassium);
    let sodium = inventory_species_total(inventory, TerrariumSpecies::ExchangeableSodium);
    let aluminum = inventory_species_total(inventory, TerrariumSpecies::ExchangeableAluminum);
    let aqueous_iron = inventory_species_total(inventory, TerrariumSpecies::AqueousIronPool);
    let sorbed_aluminum_hydroxide =
        inventory_species_total(inventory, TerrariumSpecies::SorbedAluminumHydroxide);
    let sorbed_ferric_hydroxide =
        inventory_species_total(inventory, TerrariumSpecies::SorbedFerricHydroxide);
    let base_saturation = soil_base_saturation(
        calcium,
        magnesium,
        potassium,
        sodium,
        aluminum,
        proton,
        surface_proton_load,
    );

    let state = InventoryGeochemistryState {
        water,
        proton,
        dissolved_silicate,
        bicarbonate,
        surface_proton_load,
        calcium_bicarbonate_complex,
        silicate_mineral,
        carbonate_mineral,
        calcium,
        aluminum,
        aqueous_iron,
        sorbed_aluminum_hydroxide,
        sorbed_ferric_hydroxide,
        base_saturation,
        alkalinity_gate: (bicarbonate / (0.04 + bicarbonate)).clamp(0.0, 1.0),
        oxygen_norm: (inventory_species_total(inventory, TerrariumSpecies::OxygenGas) / 0.16)
            .clamp(0.0, 1.4),
    };

    for rule in inventory_geochemistry_reaction_rules() {
        crate::terrarium::inventory_reaction_network::apply_inventory_reaction(
            inventory,
            rule.reaction,
            rule.proposed_extent(state, relaxation),
        );
    }

    for rule in inventory_geochemistry_partition_rules() {
        relax_exchange_partition(
            inventory,
            rule.species,
            rule.target_fraction(state),
            relaxation,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silicate_hydration_weathering_moves_inventory_mass_into_dissolved_pool() {
        let mut inventory = RegionalMaterialInventory::new("geo:test".into());
        crate::terrarium::material_exchange::deposit_species_to_inventory(
            &mut inventory,
            TerrariumSpecies::SilicateMineral,
            0.40,
        );
        crate::terrarium::material_exchange::deposit_species_to_inventory(
            &mut inventory,
            TerrariumSpecies::Water,
            1.20,
        );
        crate::terrarium::material_exchange::deposit_species_to_inventory(
            &mut inventory,
            TerrariumSpecies::Proton,
            0.18,
        );

        let before_silicate =
            inventory_species_total(&inventory, TerrariumSpecies::DissolvedSilicate);
        equilibrate_inventory_geochemistry(&mut inventory, 1.0);
        let after_silicate =
            inventory_species_total(&inventory, TerrariumSpecies::DissolvedSilicate);

        assert!(after_silicate > before_silicate);
        assert!(inventory_species_total(&inventory, TerrariumSpecies::SilicateMineral) < 0.40);
    }

    #[test]
    fn inventory_species_total_includes_internal_dissolved_cation_phase() {
        let mut inventory = RegionalMaterialInventory::new("geo:cation".into());
        crate::terrarium::material_exchange::deposit_species_to_inventory(
            &mut inventory,
            TerrariumSpecies::ExchangeableCalcium,
            0.30,
        );
        crate::terrarium::material_exchange::deposit_species_to_inventory(
            &mut inventory,
            TerrariumSpecies::Water,
            1.0,
        );
        crate::terrarium::material_exchange::deposit_species_to_inventory(
            &mut inventory,
            TerrariumSpecies::Proton,
            0.16,
        );

        equilibrate_inventory_geochemistry(&mut inventory, 1.0);

        let canonical = exact_amount(
            &inventory,
            MaterialRegionKind::MineralSurface,
            &MoleculeGraph::representative_exchangeable_calcium(),
            MaterialPhaseDescriptor::ambient(MaterialPhaseKind::Interfacial),
        );
        let total = inventory_species_total(&inventory, TerrariumSpecies::ExchangeableCalcium);
        assert!(total >= canonical);
        assert!(total > 0.0);
    }

    #[test]
    fn carbonate_dissolution_generates_bicarbonate_and_calcium() {
        let mut inventory = RegionalMaterialInventory::new("geo:carbonate".into());
        crate::terrarium::material_exchange::deposit_species_to_inventory(
            &mut inventory,
            TerrariumSpecies::CarbonateMineral,
            0.36,
        );
        crate::terrarium::material_exchange::deposit_species_to_inventory(
            &mut inventory,
            TerrariumSpecies::Proton,
            0.28,
        );
        crate::terrarium::material_exchange::deposit_species_to_inventory(
            &mut inventory,
            TerrariumSpecies::Water,
            0.90,
        );

        let before_bicarbonate =
            inventory_species_total(&inventory, TerrariumSpecies::BicarbonatePool);
        let before_calcium =
            inventory_species_total(&inventory, TerrariumSpecies::ExchangeableCalcium);
        equilibrate_inventory_geochemistry(&mut inventory, 1.0);
        let after_bicarbonate =
            inventory_species_total(&inventory, TerrariumSpecies::BicarbonatePool);
        let after_calcium =
            inventory_species_total(&inventory, TerrariumSpecies::ExchangeableCalcium);

        assert!(after_bicarbonate > before_bicarbonate);
        assert!(after_calcium > before_calcium);
    }

    #[test]
    fn metal_hydroxide_sorption_sequesters_free_aluminum_and_iron() {
        let mut inventory = RegionalMaterialInventory::new("geo:hydroxide".into());
        crate::terrarium::material_exchange::deposit_species_to_inventory(
            &mut inventory,
            TerrariumSpecies::Water,
            1.20,
        );
        crate::terrarium::material_exchange::deposit_species_to_inventory(
            &mut inventory,
            TerrariumSpecies::OxygenGas,
            0.24,
        );
        crate::terrarium::material_exchange::deposit_species_to_inventory(
            &mut inventory,
            TerrariumSpecies::BicarbonatePool,
            0.26,
        );
        crate::terrarium::material_exchange::deposit_species_to_inventory(
            &mut inventory,
            TerrariumSpecies::ExchangeableAluminum,
            0.20,
        );
        crate::terrarium::material_exchange::deposit_species_to_inventory(
            &mut inventory,
            TerrariumSpecies::AqueousIronPool,
            0.20,
        );

        let before_al = inventory_species_total(&inventory, TerrariumSpecies::ExchangeableAluminum);
        let before_fe = inventory_species_total(&inventory, TerrariumSpecies::AqueousIronPool);
        equilibrate_inventory_geochemistry(&mut inventory, 1.0);
        let after_al = inventory_species_total(&inventory, TerrariumSpecies::ExchangeableAluminum);
        let after_fe = inventory_species_total(&inventory, TerrariumSpecies::AqueousIronPool);
        let sorbed_al =
            inventory_species_total(&inventory, TerrariumSpecies::SorbedAluminumHydroxide);
        let sorbed_fe =
            inventory_species_total(&inventory, TerrariumSpecies::SorbedFerricHydroxide);

        assert!(after_al < before_al);
        assert!(after_fe < before_fe);
        assert!(sorbed_al > 0.0);
        assert!(sorbed_fe > 0.0);
    }

    #[test]
    fn surface_proton_load_competes_with_exchange_sites() {
        let mut inventory = RegionalMaterialInventory::new("geo:surface-proton".into());
        crate::terrarium::material_exchange::deposit_species_to_inventory(
            &mut inventory,
            TerrariumSpecies::Water,
            1.0,
        );
        crate::terrarium::material_exchange::deposit_species_to_inventory(
            &mut inventory,
            TerrariumSpecies::Proton,
            0.24,
        );
        crate::terrarium::material_exchange::deposit_species_to_inventory(
            &mut inventory,
            TerrariumSpecies::ExchangeableCalcium,
            0.22,
        );

        let before_surface =
            inventory_species_total(&inventory, TerrariumSpecies::SurfaceProtonLoad);
        let before_calcium =
            inventory_species_total(&inventory, TerrariumSpecies::ExchangeableCalcium);
        equilibrate_inventory_geochemistry(&mut inventory, 1.0);
        let after_surface =
            inventory_species_total(&inventory, TerrariumSpecies::SurfaceProtonLoad);
        let after_calcium =
            inventory_species_total(&inventory, TerrariumSpecies::ExchangeableCalcium);

        assert!(after_surface > before_surface);
        assert!(after_calcium <= before_calcium);
    }

    #[test]
    fn calcium_bicarbonate_complex_forms_under_buffered_conditions() {
        let mut inventory = RegionalMaterialInventory::new("geo:calcium-bicarbonate".into());
        crate::terrarium::material_exchange::deposit_species_to_inventory(
            &mut inventory,
            TerrariumSpecies::Water,
            1.0,
        );
        crate::terrarium::material_exchange::deposit_species_to_inventory(
            &mut inventory,
            TerrariumSpecies::BicarbonatePool,
            0.28,
        );
        crate::terrarium::material_exchange::deposit_species_to_inventory(
            &mut inventory,
            TerrariumSpecies::ExchangeableCalcium,
            0.24,
        );
        crate::terrarium::material_exchange::deposit_species_to_inventory(
            &mut inventory,
            TerrariumSpecies::Proton,
            0.04,
        );

        let before_complex =
            inventory_species_total(&inventory, TerrariumSpecies::CalciumBicarbonateComplex);
        equilibrate_inventory_geochemistry(&mut inventory, 1.0);
        let after_complex =
            inventory_species_total(&inventory, TerrariumSpecies::CalciumBicarbonateComplex);

        assert!(after_complex > before_complex);
    }
}
