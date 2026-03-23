use super::*;
use crate::terrarium::inventory_species_registry::{
    terrarium_inventory_binding, TERRARIUM_INVENTORY_BOUND_SPECIES,
};

pub(crate) const PATCH_INVENTORY_SPECIES: [TerrariumSpecies; 27] =
    TERRARIUM_INVENTORY_BOUND_SPECIES;

pub(crate) fn species_inventory_binding(
    species: TerrariumSpecies,
) -> Option<(
    MaterialRegionKind,
    MoleculeDescriptor,
    MaterialPhaseDescriptor,
)> {
    terrarium_inventory_binding(species)
}

pub(crate) fn inventory_component_amount(
    inventory: &RegionalMaterialInventory,
    species: TerrariumSpecies,
) -> f32 {
    inventory_species_total(inventory, species)
}

pub(crate) fn deposit_species_to_inventory(
    inventory: &mut RegionalMaterialInventory,
    species: TerrariumSpecies,
    amount: f32,
) {
    let Some((region, molecule, phase)) = species_inventory_binding(species) else {
        return;
    };
    inventory.deposit_component(region, molecule, amount.max(0.0) as f64, phase);
}

pub(crate) fn withdraw_species_from_inventory(
    inventory: &mut RegionalMaterialInventory,
    species: TerrariumSpecies,
    amount: f32,
) -> f32 {
    withdraw_inventory_species_total(inventory, species, amount)
}

pub(crate) fn material_inventory_target_from_patch(
    substrate: &BatchedAtomTerrarium,
    x: usize,
    y: usize,
    z: usize,
    radius: usize,
    name: impl Into<String>,
) -> RegionalMaterialInventory {
    let mut inventory = RegionalMaterialInventory::new(name.into());
    for species in PATCH_INVENTORY_SPECIES {
        let Some((region, molecule, phase)) = species_inventory_binding(species) else {
            continue;
        };
        let sample = substrate.patch_mean_species(species, x, y, z, radius);
        if sample <= 1.0e-9 {
            continue;
        }
        let _ = inventory.set_component_amount(region, molecule, phase, sample as f64);
    }
    inventory
}

pub(crate) fn sync_inventory_species_with_patch(
    substrate: &mut BatchedAtomTerrarium,
    inventory: &mut RegionalMaterialInventory,
    species: TerrariumSpecies,
    x: usize,
    y: usize,
    z: usize,
    radius: usize,
    target_amount: f32,
    relaxation: f32,
) -> Result<f32, String> {
    let Some((region, molecule, phase)) = species_inventory_binding(species) else {
        return Ok(0.0);
    };
    let selector = MaterialPhaseSelector::Exact(phase.clone());
    let current_amount = inventory.total_amount_for_component(region, &molecule, &selector);
    let target_amount = target_amount.max(0.0);
    let desired_amount =
        current_amount + (target_amount - current_amount) * relaxation.clamp(0.0, 1.0);
    if desired_amount > current_amount + 1.0e-9 {
        let draw = desired_amount - current_amount;
        let extracted = substrate.extract_patch_species(species, x, y, z, radius, draw);
        if extracted > 1.0e-9 {
            inventory.deposit_component(region, molecule.clone(), extracted as f64, phase);
        }
    } else if current_amount > desired_amount + 1.0e-9 {
        let release = current_amount - desired_amount;
        let removed = inventory.withdraw_component(region, &molecule, release as f64) as f32;
        if removed > 1.0e-9 {
            substrate.deposit_patch_species(species, x, y, z, radius, removed);
        }
    }
    Ok(inventory.total_amount_for_component(region, &molecule, &selector))
}

pub(crate) fn sync_inventory_with_patch(
    substrate: &mut BatchedAtomTerrarium,
    inventory: &mut RegionalMaterialInventory,
    x: usize,
    y: usize,
    z: usize,
    radius: usize,
    target: &RegionalMaterialInventory,
    relaxation: f32,
) -> Result<(), String> {
    for species in PATCH_INVENTORY_SPECIES {
        let target_amount = inventory_component_amount(target, species);
        sync_inventory_species_with_patch(
            substrate,
            inventory,
            species,
            x,
            y,
            z,
            radius,
            target_amount,
            relaxation,
        )?;
    }
    equilibrate_inventory_geochemistry(inventory, relaxation);
    Ok(())
}

pub(crate) fn spill_inventory_to_patch(
    substrate: &mut BatchedAtomTerrarium,
    inventory: &mut RegionalMaterialInventory,
    x: usize,
    y: usize,
    z: usize,
    radius: usize,
) {
    for species in PATCH_INVENTORY_SPECIES {
        if inventory_component_amount(inventory, species) <= 1.0e-9 {
            continue;
        }
        let released = withdraw_species_from_inventory(inventory, species, f32::MAX);
        if released > 1.0e-9 {
            substrate.deposit_patch_species(species, x, y, z, radius, released);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sync_inventory_species_with_patch_conserves_draw_and_release() {
        let mut substrate = BatchedAtomTerrarium::new(6, 6, 3, 1.0, false);
        let x = 3usize;
        let y = 3usize;
        let z = 1usize;
        let radius = 1usize;
        substrate.add_hotspot(TerrariumSpecies::AminoAcidPool, x, y, z, 0.8);
        let mut inventory = RegionalMaterialInventory::new("test".into());
        let baseline_patch =
            substrate.patch_mean_species(TerrariumSpecies::AminoAcidPool, x, y, z, radius);

        let target = material_inventory_target_from_patch(&substrate, x, y, z, radius, "target");
        sync_inventory_with_patch(
            &mut substrate,
            &mut inventory,
            x,
            y,
            z,
            radius,
            &target,
            0.8,
        )
        .expect("sync should work");
        let inventory_after_draw =
            inventory_component_amount(&inventory, TerrariumSpecies::AminoAcidPool);
        let patch_after_draw =
            substrate.patch_mean_species(TerrariumSpecies::AminoAcidPool, x, y, z, radius);
        assert!(inventory_after_draw > 0.0);
        assert!(patch_after_draw < baseline_patch);

        deposit_species_to_inventory(&mut inventory, TerrariumSpecies::AminoAcidPool, 0.25);
        sync_inventory_with_patch(
            &mut substrate,
            &mut inventory,
            x,
            y,
            z,
            radius,
            &target,
            0.6,
        )
        .expect("release sync should work");
        let patch_after_release =
            substrate.patch_mean_species(TerrariumSpecies::AminoAcidPool, x, y, z, radius);
        assert!(patch_after_release > patch_after_draw);
    }

    #[test]
    fn spill_inventory_to_patch_returns_all_supported_species() {
        let mut substrate = BatchedAtomTerrarium::new(6, 6, 3, 1.0, false);
        let x = 2usize;
        let y = 2usize;
        let z = 1usize;
        let radius = 0usize;
        let mut inventory = RegionalMaterialInventory::new("spill".into());
        deposit_species_to_inventory(&mut inventory, TerrariumSpecies::Water, 0.18);
        deposit_species_to_inventory(&mut inventory, TerrariumSpecies::Glucose, 0.22);
        deposit_species_to_inventory(&mut inventory, TerrariumSpecies::AminoAcidPool, 0.14);

        spill_inventory_to_patch(&mut substrate, &mut inventory, x, y, z, radius);

        assert!(
            inventory.is_empty(),
            "spillback should empty local inventory"
        );
        assert!(
            substrate.patch_mean_species(TerrariumSpecies::Water, x, y, z, radius) > 0.0,
            "water should be returned to the patch"
        );
        assert!(
            substrate.patch_mean_species(TerrariumSpecies::Glucose, x, y, z, radius) > 0.0,
            "glucose should be returned to the patch"
        );
        assert!(
            substrate.patch_mean_species(TerrariumSpecies::AminoAcidPool, x, y, z, radius) > 0.0,
            "amino acid pool should be returned to the patch"
        );
    }

    #[test]
    fn spill_inventory_to_patch_returns_partitioned_exchangeable_calcium() {
        let mut substrate = BatchedAtomTerrarium::new(6, 6, 3, 1.0, false);
        let x = 2usize;
        let y = 2usize;
        let z = 1usize;
        let radius = 0usize;
        let mut inventory = RegionalMaterialInventory::new("spill:geochem".into());
        deposit_species_to_inventory(&mut inventory, TerrariumSpecies::ExchangeableCalcium, 0.24);
        deposit_species_to_inventory(&mut inventory, TerrariumSpecies::Water, 0.80);
        deposit_species_to_inventory(&mut inventory, TerrariumSpecies::Proton, 0.16);

        equilibrate_inventory_geochemistry(&mut inventory, 1.0);
        let total_before =
            inventory_component_amount(&inventory, TerrariumSpecies::ExchangeableCalcium);
        assert!(total_before > 0.0);

        spill_inventory_to_patch(&mut substrate, &mut inventory, x, y, z, radius);

        assert!(
            inventory.is_empty(),
            "spillback should empty geochemical inventory"
        );
        assert!(
            substrate.patch_mean_species(TerrariumSpecies::ExchangeableCalcium, x, y, z, radius)
                > 0.0,
            "partitioned exchangeable calcium should still return through the substrate boundary"
        );
    }

    #[test]
    fn material_inventory_target_from_patch_carries_mineral_phases() {
        let mut substrate = BatchedAtomTerrarium::new(4, 4, 3, 1.0, false);
        let x = 2usize;
        let y = 2usize;
        let z = 1usize;
        substrate.add_hotspot(TerrariumSpecies::SilicateMineral, x, y, z, 0.9);
        substrate.add_hotspot(TerrariumSpecies::ClayMineral, x, y, z, 0.5);
        substrate.add_hotspot(TerrariumSpecies::CarbonateMineral, x, y, z, 0.3);
        substrate.add_hotspot(TerrariumSpecies::IronOxideMineral, x, y, z, 0.2);

        let inventory = material_inventory_target_from_patch(&substrate, x, y, z, 1, "minerals");
        assert!(inventory_component_amount(&inventory, TerrariumSpecies::SilicateMineral) > 0.0);
        assert!(inventory_component_amount(&inventory, TerrariumSpecies::ClayMineral) > 0.0);
        assert!(inventory_component_amount(&inventory, TerrariumSpecies::CarbonateMineral) > 0.0);
        assert!(inventory_component_amount(&inventory, TerrariumSpecies::IronOxideMineral) > 0.0);
    }
}
