use super::*;
use crate::atomistic_chemistry::PeriodicElement;
use crate::terrarium::inventory_species_registry::terrarium_inventory_species_profile;
use crate::terrarium::material_exchange::{inventory_component_amount, PATCH_INVENTORY_SPECIES};

/// Reported terrarium pool totals.
///
/// These are audit/reporting values only. They do not introduce a new
/// scientific authority path.
///
/// The elemental fields use the strongest local authority available:
/// elemental substrate reservoirs when they exist, and descriptor-derived
/// stoichiometry from `terrarium_inventory_species_profile()` for molecular
/// carriers. `energy_equivalent` is a usable-fuel proxy, not joules.
#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TerrariumConservationPoolVector {
    pub hydrogen: f64,
    pub carbon: f64,
    pub nitrogen: f64,
    pub phosphorus: f64,
    pub sulfur: f64,
    pub oxygen: f64,
    pub silicon: f64,
    pub aluminum: f64,
    pub calcium: f64,
    pub magnesium: f64,
    pub potassium: f64,
    pub sodium: f64,
    pub iron: f64,
    pub water: f64,
    pub energy_equivalent: f64,
}

impl std::ops::AddAssign for TerrariumConservationPoolVector {
    fn add_assign(&mut self, rhs: Self) {
        self.hydrogen += rhs.hydrogen;
        self.carbon += rhs.carbon;
        self.nitrogen += rhs.nitrogen;
        self.phosphorus += rhs.phosphorus;
        self.sulfur += rhs.sulfur;
        self.oxygen += rhs.oxygen;
        self.silicon += rhs.silicon;
        self.aluminum += rhs.aluminum;
        self.calcium += rhs.calcium;
        self.magnesium += rhs.magnesium;
        self.potassium += rhs.potassium;
        self.sodium += rhs.sodium;
        self.iron += rhs.iron;
        self.water += rhs.water;
        self.energy_equivalent += rhs.energy_equivalent;
    }
}

impl std::ops::Add for TerrariumConservationPoolVector {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TerrariumConservationAudit {
    /// Elemental substrate reservoirs plus substrate fuel carriers.
    pub substrate: TerrariumConservationPoolVector,
    /// Explicit organism-owned chemistry pulled out of the substrate.
    pub organism_inventories: TerrariumConservationPoolVector,
    /// Explicit atmospheric chemistry fields.
    pub atmosphere_fields: TerrariumConservationPoolVector,
    /// Explicit pooled/open-water bodies.
    pub water_bodies: TerrariumConservationPoolVector,
    /// Coarse detritus / exudate fields not yet represented as explicit packets.
    pub detritus_fields: TerrariumConservationPoolVector,
    /// Coarse organism state that still carries meaningful chemistry/energy.
    pub organism_state_proxies: TerrariumConservationPoolVector,
    /// Sum of explicit terrarium chemistry carriers.
    pub explicit_domain_total: TerrariumConservationPoolVector,
    /// Best-effort aggregate including remaining coarse detritus/organism state.
    pub reported_system_total: TerrariumConservationPoolVector,
}

pub fn audit_world(world: &TerrariumWorld) -> TerrariumConservationAudit {
    let substrate = substrate_totals(&world.substrate);
    let organism_inventories = organism_inventory_totals(world);
    let atmosphere_fields = atmosphere_totals(world);
    let water_bodies = water_body_totals(world);
    let detritus_fields = detritus_totals(world);
    let organism_state_proxies = organism_state_proxy_totals(world);
    let explicit_domain_total = substrate + organism_inventories + atmosphere_fields + water_bodies;
    let reported_system_total = explicit_domain_total + detritus_fields + organism_state_proxies;

    TerrariumConservationAudit {
        substrate,
        organism_inventories,
        atmosphere_fields,
        water_bodies,
        detritus_fields,
        organism_state_proxies,
        explicit_domain_total,
        reported_system_total,
    }
}

fn substrate_totals(substrate: &BatchedAtomTerrarium) -> TerrariumConservationPoolVector {
    let mut total = TerrariumConservationPoolVector::default();
    for species_idx in 0..TERRARIUM_SPECIES_COUNT {
        let species = BatchedAtomTerrarium::species_from_index(species_idx);
        let amount = substrate
            .species_field(species)
            .iter()
            .copied()
            .map(f64::from)
            .sum::<f64>();
        if amount <= 1.0e-12 {
            continue;
        }
        total += species_conservation_totals(species, amount);
    }
    total
}

fn organism_inventory_totals(world: &TerrariumWorld) -> TerrariumConservationPoolVector {
    let mut total = TerrariumConservationPoolVector::default();

    for plant in &world.plants {
        total += inventory_totals(&plant.material_inventory);
    }
    for fruit in &world.fruits {
        total += inventory_totals(&fruit.material_inventory);
        if let Some(reproduction) = fruit.reproduction.as_ref() {
            total += inventory_totals(&reproduction.material_inventory);
        }
    }
    for seed in &world.seeds {
        total += inventory_totals(&seed.material_inventory);
    }
    for cohort in &world.explicit_microbes {
        total += inventory_totals(&cohort.material_inventory);
    }
    for fly in &world.fly_pop.flies {
        total += inventory_totals(&fly.material_inventory);
    }
    for cluster in &world.fly_pop.egg_clusters {
        total += inventory_totals(&cluster.material_inventory);
        for embryo in &cluster.embryos {
            total += inventory_totals(&embryo.material_inventory);
        }
    }

    total
}

fn atmosphere_totals(world: &TerrariumWorld) -> TerrariumConservationPoolVector {
    let water = world.humidity.iter().copied().map(f64::from).sum::<f64>();
    let carbon_dioxide = world.odorants[ATMOS_CO2_IDX]
        .iter()
        .copied()
        .map(f64::from)
        .sum::<f64>();
    let oxygen_gas = world.odorants[ATMOS_O2_IDX]
        .iter()
        .copied()
        .map(f64::from)
        .sum::<f64>();

    species_conservation_totals(TerrariumSpecies::Water, water)
        + species_conservation_totals(TerrariumSpecies::CarbonDioxide, carbon_dioxide)
        + species_conservation_totals(TerrariumSpecies::OxygenGas, oxygen_gas)
}

fn water_body_totals(world: &TerrariumWorld) -> TerrariumConservationPoolVector {
    let water = world
        .waters
        .iter()
        .filter(|water| water.alive)
        .map(|water| water.volume.max(0.0) as f64)
        .sum::<f64>();
    species_conservation_totals(TerrariumSpecies::Water, water)
}

fn detritus_totals(world: &TerrariumWorld) -> TerrariumConservationPoolVector {
    let litter_carbon = world
        .litter_carbon
        .iter()
        .copied()
        .map(f64::from)
        .sum::<f64>();
    let organic_matter = world
        .organic_matter
        .iter()
        .copied()
        .map(f64::from)
        .sum::<f64>();
    let root_exudates = world
        .root_exudates
        .iter()
        .copied()
        .map(f64::from)
        .sum::<f64>();
    TerrariumConservationPoolVector {
        carbon: litter_carbon + organic_matter + root_exudates,
        energy_equivalent: litter_carbon + root_exudates,
        ..Default::default()
    }
}

fn organism_state_proxy_totals(world: &TerrariumWorld) -> TerrariumConservationPoolVector {
    let mut total = TerrariumConservationPoolVector::default();

    for plant in &world.plants {
        total.carbon += plant.physiology.total_biomass().max(0.0) as f64;
        total.nitrogen += plant.physiology.nitrogen_buffer().max(0.0) as f64;
        total.water += plant.physiology.water_buffer().max(0.0) as f64;
        total.energy_equivalent += plant.physiology.storage_carbon().max(0.0) as f64;
    }

    for fruit in &world.fruits {
        total.carbon += fruit.source.sugar_content.max(0.0) as f64;
        total.energy_equivalent += fruit.source.sugar_content.max(0.0) as f64;
        if let Some(reproduction) = fruit.reproduction.as_ref() {
            total.carbon += reproduction.reserve_carbon.max(0.0) as f64;
            total.energy_equivalent += reproduction.reserve_carbon.max(0.0) as f64;
        }
    }

    for seed in &world.seeds {
        total.carbon += seed.reserve_carbon.max(0.0) as f64;
        total.energy_equivalent += seed.reserve_carbon.max(0.0) as f64;
    }

    for fly in &world.flies {
        total.energy_equivalent += fly.body_state().energy.max(0.0) as f64;
    }
    for metabolism in &world.fly_metabolisms {
        let trehalose_mm = metabolism.hemolymph_trehalose_mm.max(0.0) as f64;
        let atp_mm = metabolism.muscle_atp_mm.max(0.0) as f64;
        total += trehalose_elemental_contribution(trehalose_mm);
        total += atp_elemental_contribution(atp_mm);
        total.energy_equivalent += trehalose_mm + atp_mm;
    }
    for fly in &world.fly_pop.flies {
        total.energy_equivalent += fly.energy.max(0.0) as f64;
    }

    let coarse_biomass = world
        .microbial_biomass
        .iter()
        .chain(world.symbiont_biomass.iter())
        .chain(world.nitrifier_biomass.iter())
        .chain(world.denitrifier_biomass.iter())
        .copied()
        .map(f64::from)
        .sum::<f64>();
    total.carbon += coarse_biomass;
    total.energy_equivalent += coarse_biomass;

    let coarse_reserve = world
        .microbial_reserve
        .iter()
        .chain(world.nitrifier_reserve.iter())
        .chain(world.denitrifier_reserve.iter())
        .copied()
        .map(f64::from)
        .sum::<f64>();
    total.energy_equivalent += coarse_reserve;

    let earthworm_biomass = world
        .earthworm_population
        .biomass_per_voxel
        .iter()
        .copied()
        .map(f64::from)
        .sum::<f64>();
    total.carbon += earthworm_biomass;
    total.energy_equivalent += earthworm_biomass;

    let nematode_biomass = world
        .nematode_guilds
        .iter()
        .flat_map(|guild| guild.biomass_per_voxel.iter())
        .copied()
        .map(f64::from)
        .sum::<f64>();
    total.carbon += nematode_biomass;
    total.energy_equivalent += nematode_biomass;

    total
}

fn inventory_totals(inventory: &RegionalMaterialInventory) -> TerrariumConservationPoolVector {
    let mut total = TerrariumConservationPoolVector::default();
    for species in PATCH_INVENTORY_SPECIES {
        let amount = inventory_component_amount(inventory, species) as f64;
        if amount <= 1.0e-12 {
            continue;
        }
        total += species_conservation_totals(species, amount);
    }
    total
}

fn descriptor_element_totals(
    descriptor: &MoleculeDescriptor,
    amount: f64,
) -> TerrariumConservationPoolVector {
    let mut total = TerrariumConservationPoolVector::default();
    for (element, count) in &descriptor.element_counts {
        let delta = amount * *count as f64;
        accumulate_element(&mut total, *element, delta);
    }
    total
}

/// Map a [`PeriodicElement`] delta onto the corresponding
/// [`TerrariumConservationPoolVector`] field.
fn accumulate_element(
    total: &mut TerrariumConservationPoolVector,
    element: PeriodicElement,
    delta: f64,
) {
    match element {
        PeriodicElement::H => total.hydrogen += delta,
        PeriodicElement::C => total.carbon += delta,
        PeriodicElement::N => total.nitrogen += delta,
        PeriodicElement::O => total.oxygen += delta,
        PeriodicElement::P => total.phosphorus += delta,
        PeriodicElement::S => total.sulfur += delta,
        PeriodicElement::Si => total.silicon += delta,
        PeriodicElement::Al => total.aluminum += delta,
        PeriodicElement::Ca => total.calcium += delta,
        PeriodicElement::Mg => total.magnesium += delta,
        PeriodicElement::K => total.potassium += delta,
        PeriodicElement::Na => total.sodium += delta,
        PeriodicElement::Fe => total.iron += delta,
        _ => {}
    }
}

/// Compute the exact elemental contribution of `amount` molecules of a
/// [`TerrariumSpecies`], using the canonical inventory species profile
/// (descriptor-derived element counts from `terrarium_molecular_assets.json`).
///
/// Returns a zero vector if the species has no inventory profile.
pub fn species_elemental_contribution(
    species: TerrariumSpecies,
    amount: f64,
) -> TerrariumConservationPoolVector {
    species_conservation_totals(species, amount)
}

/// Elemental contribution for trehalose (C12H22O11), the primary insect
/// hemolymph sugar. Trehalose is not a distinct `TerrariumSpecies`; it is a
/// disaccharide formed by condensation of two glucose molecules (2 x C6H12O6
/// minus H2O). We derive its composition from the glucose inventory profile
/// rather than hardcoding coefficients.
///
/// Stoichiometry: 2 * glucose - H2O = C12H22O11 (12 C, 22 H, 11 O).
fn trehalose_elemental_contribution(amount: f64) -> TerrariumConservationPoolVector {
    let glucose_profile = terrarium_inventory_species_profile(TerrariumSpecies::Glucose);
    let water_profile = terrarium_inventory_species_profile(TerrariumSpecies::Water);

    match (glucose_profile, water_profile) {
        (Some(glu), Some(h2o)) => {
            // Trehalose = 2 glucose - 1 water (condensation).
            let mut total = TerrariumConservationPoolVector::default();
            for &(element, count) in glu.element_counts {
                accumulate_element(&mut total, element, amount * 2.0 * count as f64);
            }
            for &(element, count) in h2o.element_counts {
                accumulate_element(&mut total, element, -amount * count as f64);
            }
            total
        }
        _ => {
            // Fallback: hardcoded trehalose C12H22O11 (should never be
            // reached since glucose and water always have profiles).
            TerrariumConservationPoolVector {
                carbon: amount * 12.0,
                hydrogen: amount * 22.0,
                oxygen: amount * 11.0,
                ..Default::default()
            }
        }
    }
}

/// Elemental contribution for ATP (C10H16N5O13P3), the primary biological
/// energy currency. Derives its composition from the `AtpFlux` inventory
/// species profile rather than hardcoding coefficients.
fn atp_elemental_contribution(amount: f64) -> TerrariumConservationPoolVector {
    let atp_profile = terrarium_inventory_species_profile(TerrariumSpecies::AtpFlux);

    match atp_profile {
        Some(profile) => {
            let mut total = TerrariumConservationPoolVector::default();
            for &(element, count) in profile.element_counts {
                accumulate_element(&mut total, element, amount * count as f64);
            }
            total
        }
        None => {
            // Fallback: hardcoded ATP C10H16N5O13P3 (should never be reached).
            TerrariumConservationPoolVector {
                carbon: amount * 10.0,
                hydrogen: amount * 16.0,
                nitrogen: amount * 5.0,
                oxygen: amount * 13.0,
                phosphorus: amount * 3.0,
                ..Default::default()
            }
        }
    }
}

fn species_conservation_totals(
    species: TerrariumSpecies,
    amount: f64,
) -> TerrariumConservationPoolVector {
    let mut total = match species {
        TerrariumSpecies::Carbon => TerrariumConservationPoolVector {
            carbon: amount,
            ..Default::default()
        },
        TerrariumSpecies::Hydrogen => TerrariumConservationPoolVector {
            hydrogen: amount,
            ..Default::default()
        },
        TerrariumSpecies::Oxygen => TerrariumConservationPoolVector {
            oxygen: amount,
            ..Default::default()
        },
        TerrariumSpecies::Nitrogen => TerrariumConservationPoolVector {
            nitrogen: amount,
            ..Default::default()
        },
        TerrariumSpecies::Phosphorus => TerrariumConservationPoolVector {
            phosphorus: amount,
            ..Default::default()
        },
        TerrariumSpecies::Sulfur => TerrariumConservationPoolVector {
            sulfur: amount,
            ..Default::default()
        },
        TerrariumSpecies::Proton => TerrariumConservationPoolVector {
            hydrogen: amount,
            ..Default::default()
        },
        _ => {
            let Some((_, descriptor, _)) =
                crate::terrarium::material_exchange::species_inventory_binding(species)
            else {
                return TerrariumConservationPoolVector::default();
            };
            descriptor_element_totals(&descriptor, amount)
        }
    };
    match species {
        TerrariumSpecies::Water => total.water += amount,
        TerrariumSpecies::Glucose | TerrariumSpecies::AtpFlux => total.energy_equivalent += amount,
        _ => {}
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrarium::inventory_species_registry::terrarium_inventory_species_profile;

    #[test]
    fn conservation_counts_geological_elements_from_representative_minerals() {
        let quartz = species_conservation_totals(TerrariumSpecies::SilicateMineral, 2.0);
        assert!((quartz.silicon - 2.0).abs() < 1.0e-6);
        assert!((quartz.oxygen - 4.0).abs() < 1.0e-6);

        let carbonate = species_conservation_totals(TerrariumSpecies::CarbonateMineral, 1.5);
        assert!((carbonate.calcium - 1.5).abs() < 1.0e-6);
        assert!((carbonate.carbon - 1.5).abs() < 1.0e-6);
        assert!((carbonate.oxygen - 4.5).abs() < 1.0e-6);

        let iron = species_conservation_totals(TerrariumSpecies::AqueousIronPool, 0.75);
        assert!((iron.iron - 0.75).abs() < 1.0e-6);

        let bicarbonate = species_conservation_totals(TerrariumSpecies::BicarbonatePool, 2.0);
        assert!((bicarbonate.hydrogen - 2.0).abs() < 1.0e-6);
        assert!((bicarbonate.carbon - 2.0).abs() < 1.0e-6);
        assert!((bicarbonate.oxygen - 6.0).abs() < 1.0e-6);

        let aluminum_hydroxide =
            species_conservation_totals(TerrariumSpecies::SorbedAluminumHydroxide, 1.25);
        assert!((aluminum_hydroxide.aluminum - 1.25).abs() < 1.0e-6);
        assert!((aluminum_hydroxide.oxygen - 3.75).abs() < 1.0e-6);
        assert!((aluminum_hydroxide.hydrogen - 3.75).abs() < 1.0e-6);

        let ferric_hydroxide =
            species_conservation_totals(TerrariumSpecies::SorbedFerricHydroxide, 0.8);
        assert!((ferric_hydroxide.iron - 0.8).abs() < 1.0e-6);
        assert!((ferric_hydroxide.oxygen - 2.4).abs() < 1.0e-6);
        assert!((ferric_hydroxide.hydrogen - 2.4).abs() < 1.0e-6);
    }

    // ---- Descriptor-derived elemental contribution tests ----

    #[test]
    fn conservation_water_descriptor_derived_composition() {
        // Water: H2O => 2 H, 1 O
        let water = species_elemental_contribution(TerrariumSpecies::Water, 3.0);
        assert!(
            (water.hydrogen - 6.0).abs() < 1.0e-6,
            "water H: expected 6.0, got {}",
            water.hydrogen
        );
        assert!(
            (water.oxygen - 3.0).abs() < 1.0e-6,
            "water O: expected 3.0, got {}",
            water.oxygen
        );
        assert!(
            (water.water - 3.0).abs() < 1.0e-6,
            "water pool: expected 3.0, got {}",
            water.water
        );
        assert!(water.carbon.abs() < 1.0e-12);
        assert!(water.nitrogen.abs() < 1.0e-12);
    }

    #[test]
    fn conservation_glucose_descriptor_derived_composition() {
        // Glucose: C6H12O6 => 6 C, 12 H, 6 O
        let glucose = species_elemental_contribution(TerrariumSpecies::Glucose, 2.0);
        assert!(
            (glucose.carbon - 12.0).abs() < 1.0e-6,
            "glucose C: expected 12.0, got {}",
            glucose.carbon
        );
        assert!(
            (glucose.hydrogen - 24.0).abs() < 1.0e-6,
            "glucose H: expected 24.0, got {}",
            glucose.hydrogen
        );
        assert!(
            (glucose.oxygen - 12.0).abs() < 1.0e-6,
            "glucose O: expected 12.0, got {}",
            glucose.oxygen
        );
        assert!(
            (glucose.energy_equivalent - 2.0).abs() < 1.0e-6,
            "glucose energy: expected 2.0, got {}",
            glucose.energy_equivalent
        );
        assert!(glucose.nitrogen.abs() < 1.0e-12);
        assert!(glucose.phosphorus.abs() < 1.0e-12);
    }

    #[test]
    fn conservation_atp_descriptor_derived_composition() {
        // ATP: C10H16N5O13P3
        let atp = species_elemental_contribution(TerrariumSpecies::AtpFlux, 1.0);
        assert!(
            (atp.carbon - 10.0).abs() < 1.0e-6,
            "atp C: expected 10.0, got {}",
            atp.carbon
        );
        assert!(
            (atp.hydrogen - 16.0).abs() < 1.0e-6,
            "atp H: expected 16.0, got {}",
            atp.hydrogen
        );
        assert!(
            (atp.nitrogen - 5.0).abs() < 1.0e-6,
            "atp N: expected 5.0, got {}",
            atp.nitrogen
        );
        assert!(
            (atp.oxygen - 13.0).abs() < 1.0e-6,
            "atp O: expected 13.0, got {}",
            atp.oxygen
        );
        assert!(
            (atp.phosphorus - 3.0).abs() < 1.0e-6,
            "atp P: expected 3.0, got {}",
            atp.phosphorus
        );
        assert!(
            (atp.energy_equivalent - 1.0).abs() < 1.0e-6,
            "atp energy: expected 1.0, got {}",
            atp.energy_equivalent
        );
    }

    #[test]
    fn conservation_atp_helper_matches_species_profile() {
        // Verify the atp_elemental_contribution helper produces the same
        // elemental totals as species_conservation_totals minus the energy
        // and water bookkeeping (which is only added by the top-level fn).
        let from_helper = atp_elemental_contribution(2.5);
        let profile = terrarium_inventory_species_profile(TerrariumSpecies::AtpFlux)
            .expect("atp profile should exist");

        for &(element, count) in profile.element_counts {
            let expected = 2.5 * count as f64;
            let actual = match element {
                PeriodicElement::C => from_helper.carbon,
                PeriodicElement::H => from_helper.hydrogen,
                PeriodicElement::N => from_helper.nitrogen,
                PeriodicElement::O => from_helper.oxygen,
                PeriodicElement::P => from_helper.phosphorus,
                _ => 0.0,
            };
            assert!(
                (actual - expected).abs() < 1.0e-6,
                "atp element {}: expected {expected}, got {actual}",
                element.symbol()
            );
        }
    }

    #[test]
    fn conservation_trehalose_derived_from_glucose_condensation() {
        // Trehalose = 2*glucose - 1*water = C12H22O11
        let trehalose = trehalose_elemental_contribution(1.0);
        assert!(
            (trehalose.carbon - 12.0).abs() < 1.0e-6,
            "trehalose C: expected 12.0, got {}",
            trehalose.carbon
        );
        assert!(
            (trehalose.hydrogen - 22.0).abs() < 1.0e-6,
            "trehalose H: expected 22.0, got {}",
            trehalose.hydrogen
        );
        assert!(
            (trehalose.oxygen - 11.0).abs() < 1.0e-6,
            "trehalose O: expected 11.0, got {}",
            trehalose.oxygen
        );
        // Trehalose has no N, P, S, metals
        assert!(trehalose.nitrogen.abs() < 1.0e-12);
        assert!(trehalose.phosphorus.abs() < 1.0e-12);
        assert!(trehalose.sulfur.abs() < 1.0e-12);
    }

    #[test]
    fn conservation_trehalose_scaling() {
        // Verify linearity: 3 mol trehalose = 3x the single-molecule result.
        let one = trehalose_elemental_contribution(1.0);
        let three = trehalose_elemental_contribution(3.0);
        assert!((three.carbon - one.carbon * 3.0).abs() < 1.0e-6);
        assert!((three.hydrogen - one.hydrogen * 3.0).abs() < 1.0e-6);
        assert!((three.oxygen - one.oxygen * 3.0).abs() < 1.0e-6);
    }

    #[test]
    fn conservation_fly_metabolism_proxy_covers_all_elements() {
        // The old hardcoded proxy was missing hydrogen from trehalose and
        // carbon/hydrogen from ATP. Verify the new descriptor-derived path
        // accounts for all elements in both metabolites.
        let trehalose = trehalose_elemental_contribution(1.0);
        let atp = atp_elemental_contribution(1.0);

        // Trehalose must report H (was previously missing).
        assert!(
            trehalose.hydrogen > 0.0,
            "trehalose should report non-zero hydrogen"
        );
        // ATP must report C and H (were previously missing).
        assert!(
            atp.carbon > 0.0,
            "atp should report non-zero carbon"
        );
        assert!(
            atp.hydrogen > 0.0,
            "atp should report non-zero hydrogen"
        );
    }

    #[test]
    fn conservation_elemental_species_pass_through_directly() {
        // Pure elemental species should contribute exactly their amount to
        // the corresponding field and zero to all others.
        let carbon = species_elemental_contribution(TerrariumSpecies::Carbon, 5.0);
        assert!((carbon.carbon - 5.0).abs() < 1.0e-6);
        assert!(carbon.hydrogen.abs() < 1.0e-12);
        assert!(carbon.oxygen.abs() < 1.0e-12);

        let sulfur = species_elemental_contribution(TerrariumSpecies::Sulfur, 1.5);
        assert!((sulfur.sulfur - 1.5).abs() < 1.0e-6);
        assert!(sulfur.carbon.abs() < 1.0e-12);
    }

    #[test]
    fn conservation_substrate_totals_self_consistent_with_species_sum() {
        // Verify that substrate_totals produces the same result as manually
        // iterating species and calling species_conservation_totals.
        let world = TerrariumWorld::demo(42, false).expect("demo world should build");
        let computed = substrate_totals(&world.substrate);

        let mut manual = TerrariumConservationPoolVector::default();
        for species_idx in 0..TERRARIUM_SPECIES_COUNT {
            let species = BatchedAtomTerrarium::species_from_index(species_idx);
            let amount = world
                .substrate
                .species_field(species)
                .iter()
                .copied()
                .map(f64::from)
                .sum::<f64>();
            if amount > 1.0e-12 {
                manual += species_conservation_totals(species, amount);
            }
        }
        assert!(
            (computed.carbon - manual.carbon).abs() < 1.0e-6,
            "substrate C mismatch: {} vs {}",
            computed.carbon,
            manual.carbon
        );
        assert!(
            (computed.oxygen - manual.oxygen).abs() < 1.0e-6,
            "substrate O mismatch: {} vs {}",
            computed.oxygen,
            manual.oxygen
        );
        assert!(
            (computed.hydrogen - manual.hydrogen).abs() < 1.0e-6,
            "substrate H mismatch: {} vs {}",
            computed.hydrogen,
            manual.hydrogen
        );
    }

    #[test]
    fn conservation_all_bound_species_produce_nonzero_contributions() {
        // Every species in PATCH_INVENTORY_SPECIES should produce a non-zero
        // pool vector for amount = 1.0 (at least one elemental field > 0).
        for species in PATCH_INVENTORY_SPECIES {
            let contribution = species_elemental_contribution(species, 1.0);
            let sum = contribution.hydrogen
                + contribution.carbon
                + contribution.nitrogen
                + contribution.phosphorus
                + contribution.sulfur
                + contribution.oxygen
                + contribution.silicon
                + contribution.aluminum
                + contribution.calcium
                + contribution.magnesium
                + contribution.potassium
                + contribution.sodium
                + contribution.iron;
            assert!(
                sum > 0.0,
                "species {:?} should produce non-zero elemental contribution",
                species
            );
        }
    }

    #[test]
    fn conservation_audit_reported_total_is_sum_of_domains() {
        let world = TerrariumWorld::demo(42, false).expect("demo world should build");
        let audit = audit_world(&world);

        // reported_system_total = explicit_domain_total + detritus + organism_state
        // explicit_domain_total = substrate + organism_inventories + atmosphere + water
        let recomputed_explicit = audit.substrate
            + audit.organism_inventories
            + audit.atmosphere_fields
            + audit.water_bodies;
        let recomputed_total = recomputed_explicit
            + audit.detritus_fields
            + audit.organism_state_proxies;

        assert!(
            (audit.explicit_domain_total.carbon - recomputed_explicit.carbon).abs() < 1.0e-6,
            "explicit domain C mismatch"
        );
        assert!(
            (audit.reported_system_total.carbon - recomputed_total.carbon).abs() < 1.0e-6,
            "reported total C mismatch"
        );
        assert!(
            (audit.reported_system_total.oxygen - recomputed_total.oxygen).abs() < 1.0e-6,
            "reported total O mismatch"
        );
        assert!(
            (audit.reported_system_total.hydrogen - recomputed_total.hydrogen).abs() < 1.0e-6,
            "reported total H mismatch"
        );
    }
}
