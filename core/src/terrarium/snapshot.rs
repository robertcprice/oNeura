//! Terrarium Snapshot: High-performance telemetry and state capture.

use super::*;
use crate::terrarium::material_exchange::inventory_component_amount;

impl TerrariumWorld {
    /// Generate a snapshot of the current world state.
    pub fn snapshot(&self) -> TerrariumWorldSnapshot {
        let live_fruits = self
            .fruits
            .iter()
            .filter(|f| f.source.alive && f.source.sugar_content > 0.01)
            .count();
        let food_remaining: f32 = self
            .fruits
            .iter()
            .filter(|f| f.source.alive)
            .map(|f| f.source.sugar_content.max(0.0))
            .sum();
        let n_flies = self.flies.len() as f32;
        let avg_fly_energy = if self.flies.is_empty() {
            0.0
        } else {
            self.flies
                .iter()
                .map(|f| f.body_state().energy)
                .sum::<f32>()
                / n_flies
        };
        let avg_altitude = if self.flies.is_empty() {
            0.0
        } else {
            self.flies.iter().map(|f| f.body_state().z).sum::<f32>() / n_flies
        };
        let avg_fly_energy_charge = if self.fly_metabolisms.is_empty() {
            0.0
        } else {
            use crate::organism_metabolism::OrganismMetabolism;
            self.fly_metabolisms
                .iter()
                .map(|m| m.energy_charge())
                .sum::<f32>()
                / self.fly_metabolisms.len() as f32
        };
        let w = self.config.width;
        let h = self.config.height;
        let n_cells = (w * h) as f32;
        let air_layers = self.config.depth.max(1);
        let air_volume = (w * h * air_layers).max(1) as f32;
        let mean_soil_moisture = if n_cells > 0.0 {
            self.moisture.iter().sum::<f32>() / n_cells
        } else {
            0.0
        };
        let mean_litter_cover_fraction = if n_cells > 0.0 {
            self.litter_surface
                .iter()
                .map(|surface| surface.cover_fraction)
                .sum::<f32>()
                / n_cells
        } else {
            0.0
        };
        let mean_litter_surface_depth_mm = if n_cells > 0.0 {
            self.litter_surface
                .iter()
                .map(|surface| surface.support_depth_mm)
                .sum::<f32>()
                / n_cells
        } else {
            0.0
        };
        let mean_soil_glucose = self.substrate.mean_species(TerrariumSpecies::Glucose);
        let mean_soil_oxygen = self.substrate.mean_species(TerrariumSpecies::OxygenGas);
        let mean_soil_ammonium = self.substrate.mean_species(TerrariumSpecies::Ammonium);
        let mean_soil_nitrate = self.substrate.mean_species(TerrariumSpecies::Nitrate);
        let mean_soil_atp_flux = self.substrate.mean_species(TerrariumSpecies::AtpFlux);
        let mean_soil_redox = (mean_soil_oxygen
            - self.substrate.mean_species(TerrariumSpecies::CarbonDioxide)
            - self.substrate.mean_species(TerrariumSpecies::Proton) * 0.8)
            .clamp(-1.0, 1.0);
        let mean_soil_silicate_mineral = self
            .substrate
            .mean_species(TerrariumSpecies::SilicateMineral);
        let mean_soil_clay_mineral = self.substrate.mean_species(TerrariumSpecies::ClayMineral);
        let mean_soil_carbonate_mineral = self
            .substrate
            .mean_species(TerrariumSpecies::CarbonateMineral);
        let mean_soil_iron_oxide_mineral = self
            .substrate
            .mean_species(TerrariumSpecies::IronOxideMineral);
        let mean_soil_dissolved_silicate = self
            .substrate
            .mean_species(TerrariumSpecies::DissolvedSilicate);
        let mean_soil_bicarbonate = self
            .substrate
            .mean_species(TerrariumSpecies::BicarbonatePool);
        let mean_soil_surface_proton_load = self
            .substrate
            .mean_species(TerrariumSpecies::SurfaceProtonLoad);
        let mean_soil_calcium_bicarbonate_complex = self
            .substrate
            .mean_species(TerrariumSpecies::CalciumBicarbonateComplex);
        let mean_soil_sorbed_aluminum_hydroxide = self
            .substrate
            .mean_species(TerrariumSpecies::SorbedAluminumHydroxide);
        let mean_soil_sorbed_ferric_hydroxide = self
            .substrate
            .mean_species(TerrariumSpecies::SorbedFerricHydroxide);
        let mean_soil_exchangeable_calcium = self
            .substrate
            .mean_species(TerrariumSpecies::ExchangeableCalcium);
        let mean_soil_exchangeable_magnesium = self
            .substrate
            .mean_species(TerrariumSpecies::ExchangeableMagnesium);
        let mean_soil_exchangeable_potassium = self
            .substrate
            .mean_species(TerrariumSpecies::ExchangeablePotassium);
        let mean_soil_exchangeable_sodium = self
            .substrate
            .mean_species(TerrariumSpecies::ExchangeableSodium);
        let mean_soil_exchangeable_aluminum = self
            .substrate
            .mean_species(TerrariumSpecies::ExchangeableAluminum);
        let mean_soil_aqueous_iron = self
            .substrate
            .mean_species(TerrariumSpecies::AqueousIronPool);
        let owned_cells = self
            .ownership
            .iter()
            .filter(|c| !matches!(c.owner, SoilOwnershipClass::Background))
            .count();
        let owned_fraction = if n_cells > 0.0 {
            owned_cells as f32 / n_cells
        } else {
            0.0
        };
        let fly_plant_proximity_mean = if self.flies.is_empty() || self.plants.is_empty() {
            0.0
        } else {
            let mut total_dist = 0.0f32;
            for fly in &self.flies {
                let fb = fly.body_state();
                let min_dist = self
                    .plants
                    .iter()
                    .map(|p| {
                        let dx = fb.x - p.x as f32;
                        let dy = fb.y - p.y as f32;
                        (dx * dx + dy * dy).sqrt()
                    })
                    .fold(f32::MAX, f32::min);
                total_dist += min_dist;
            }
            total_dist / n_flies
        };
        let fly_altitude_mean = avg_altitude;

        let avg_wing_beat_freq = if self.flies.is_empty() {
            0.0
        } else {
            self.flies
                .iter()
                .map(|f| f.body_state().wing_beat_freq)
                .sum::<f32>()
                / n_flies
        };
        let registry_display_name = |organism_id: u64| {
            self.organism_registry
                .get(&organism_id)
                .and_then(|entry| entry.identity.display_name.clone())
        };
        let fly_stage_label = |stage: crate::drosophila_population::FlyLifeStage| match stage {
            crate::drosophila_population::FlyLifeStage::Embryo { .. } => "embryo".to_string(),
            crate::drosophila_population::FlyLifeStage::Larva { instar, .. } => {
                format!("larva-{instar}")
            }
            crate::drosophila_population::FlyLifeStage::Pupa { .. } => "pupa".to_string(),
            crate::drosophila_population::FlyLifeStage::Adult { .. } => "adult".to_string(),
        };
        let fly_sex_label = |sex: crate::drosophila_population::FlySex| match sex {
            crate::drosophila_population::FlySex::Male => "male".to_string(),
            crate::drosophila_population::FlySex::Female => "female".to_string(),
        };

        let full_plants: Vec<TerrariumPlantSnapshot> = self
            .plants
            .iter()
            .map(|p| {
                let estimated_fruit_load = p
                    .physiology
                    .fruit_count()
                    .min((p.physiology.storage_carbon().max(0.0) * 5.0).round() as u32);
                TerrariumPlantSnapshot {
                    x: p.x,
                    y: p.y,
                    organism_id: p.identity.organism_id,
                    phylo_id: p.identity.phylo_id,
                    lineage_generation: p.identity.generation,
                    parent_organism_id: p.identity.parent_organism_id,
                    co_parent_organism_id: p.identity.co_parent_organism_id,
                    display_name: p.identity.display_name.clone(),
                    taxonomy_id: p.genome.taxonomy_id,
                    common_name: crate::terrarium::plant_species::plant_common_name(
                        p.genome.taxonomy_id,
                    )
                    .to_string(),
                    scientific_name: crate::terrarium::plant_species::plant_scientific_name(
                        p.genome.taxonomy_id,
                    )
                    .to_string(),
                    growth_form: crate::terrarium::plant_species::plant_growth_form(
                        p.genome.taxonomy_id,
                    ),
                    height_mm: p.physiology.height_mm(),
                    vitality: p.cellular.vitality(),
                    storage_carbon: p.physiology.storage_carbon(),
                    fruit_load: estimated_fruit_load,
                    structure:
                        crate::terrarium::shape_projection::structure_descriptor_from_morphology(
                            &p.morphology,
                        ),
                    morphology: p
                        .morphology
                        .generate_nodes_with_context(estimated_fruit_load, p.cellular.vitality()),
                    branch_mesh: None, // computed below
                }
            })
            .collect();

        // Compute smooth ribbon branch meshes for TREES only (not grass/herbs).
        let full_plants: Vec<TerrariumPlantSnapshot> = full_plants
            .into_iter()
            .map(|mut snap| {
                let is_tree = matches!(snap.growth_form,
                    crate::botany::BotanicalGrowthForm::OrchardTree |
                    crate::botany::BotanicalGrowthForm::StoneFruitTree |
                    crate::botany::BotanicalGrowthForm::CitrusTree
                );
                if is_tree {
                    let stem_color =
                        crate::terrarium::visual_projection::plant_stem_color_u8(snap.taxonomy_id);
                    snap.branch_mesh = Some(crate::terrarium::ribbon::build_plant_branch_mesh(
                        &snap.morphology,
                        stem_color,
                        &crate::terrarium::ribbon::RibbonConfig::default(),
                    ));
                }
                snap
            })
            .collect();

        let full_fruits: Vec<TerrariumFruitSnapshot> = self
            .fruits
            .iter()
            .map(|fruit| TerrariumFruitSnapshot {
                x: fruit.source.x,
                y: fruit.source.y,
                organism_id: fruit.identity.organism_id,
                lineage_generation: fruit.identity.generation,
                parent_organism_id: fruit.identity.parent_organism_id,
                display_name: fruit.identity.display_name.clone(),
                taxonomy_id: fruit.taxonomy_id,
                common_name: crate::terrarium::plant_species::plant_common_name(fruit.taxonomy_id)
                    .to_string(),
                scientific_name: crate::terrarium::plant_species::plant_scientific_name(
                    fruit.taxonomy_id,
                )
                .to_string(),
                growth_form: crate::terrarium::plant_species::plant_growth_form(fruit.taxonomy_id),
                shape: crate::terrarium::shape_projection::fruit_shape_descriptor(
                    &fruit.source_genome,
                    fruit.radius,
                    fruit.source.ripeness,
                    fruit.source.sugar_content,
                ),
                sugar_content: fruit.source.sugar_content,
                ripeness: fruit.source.ripeness,
                radius: fruit.radius,
                attached: fruit.source.attached,
                alive: fruit.source.alive,
            })
            .collect();

        let full_seeds: Vec<TerrariumSeedSnapshot> = self
            .seeds
            .iter()
            .map(|seed| TerrariumSeedSnapshot {
                x: seed.x,
                y: seed.y,
                organism_id: seed.identity.organism_id,
                phylo_id: seed.identity.phylo_id,
                lineage_generation: seed.identity.generation,
                parent_organism_id: seed.identity.parent_organism_id,
                co_parent_organism_id: seed.identity.co_parent_organism_id,
                display_name: seed.identity.display_name.clone(),
                taxonomy_id: seed.genome.taxonomy_id,
                common_name: crate::terrarium::plant_species::plant_common_name(
                    seed.genome.taxonomy_id,
                )
                .to_string(),
                scientific_name: crate::terrarium::plant_species::plant_scientific_name(
                    seed.genome.taxonomy_id,
                )
                .to_string(),
                growth_form: crate::terrarium::plant_species::plant_growth_form(
                    seed.genome.taxonomy_id,
                ),
                shape: crate::terrarium::shape_projection::seed_shape_descriptor(
                    &seed.genome,
                    seed.reserve_carbon,
                    seed.dormancy_s,
                ),
                reserve_carbon: seed.reserve_carbon,
                dormancy_s: seed.dormancy_s,
                age_s: seed.age_s,
                burial_depth_mm: seed.microsite.burial_depth_mm,
                surface_exposure: seed.microsite.surface_exposure,
            })
            .collect();
        let full_flies: Vec<TerrariumFlySnapshot> = self
            .fly_identities
            .iter()
            .enumerate()
            .filter_map(|(index, identity)| {
                let fly = self.flies.get(index)?;
                let body = fly.body_state();
                let metabolism = self.fly_metabolisms.get(index);
                Some(TerrariumFlySnapshot {
                    organism_id: identity.organism_id,
                    display_name: identity.display_name.clone(),
                    x: body.x,
                    y: body.y,
                    z: body.z,
                    energy: body.energy,
                    hunger: metabolism.map(|m| m.hunger()).unwrap_or(0.0),
                    energy_charge: metabolism
                        .map(crate::organism_metabolism::OrganismMetabolism::energy_charge)
                        .unwrap_or(0.0),
                    is_flying: body.is_flying,
                    wing_beat_freq: body.wing_beat_freq,
                })
            })
            .collect();
        let full_fly_population: Vec<TerrariumFlyLifecycleSnapshot> = self
            .fly_pop
            .flies
            .iter()
            .map(|fly| TerrariumFlyLifecycleSnapshot {
                fly_id: fly.id,
                organism_id: fly.organism_id,
                display_name: fly.organism_id.and_then(registry_display_name),
                sex: fly_sex_label(fly.sex),
                stage: fly_stage_label(fly.stage),
                x: fly.position.0,
                y: fly.position.1,
                z: fly.position.2,
                energy: fly.energy,
                alive: fly.is_alive(),
                eggs_remaining: fly.eggs_remaining,
                mated: fly.mated,
            })
            .collect();
        let full_egg_clusters: Vec<TerrariumEggClusterSnapshot> = self
            .fly_pop
            .egg_clusters
            .iter()
            .map(|cluster| TerrariumEggClusterSnapshot {
                x: cluster.position.0,
                y: cluster.position.1,
                count: cluster.count,
                age_hours: cluster.age_hours,
                substrate_quality: cluster.substrate_quality,
            })
            .collect();
        let full_fly_embryos: Vec<TerrariumFlyEmbryoSnapshot> = self
            .fly_pop
            .iter_cluster_embryos()
            .map(|(cluster_index, _, cluster, embryo)| {
                let position = cluster.embryo_position_mm(embryo);
                TerrariumFlyEmbryoSnapshot {
                    embryo_id: embryo.id,
                    cluster_index,
                    sex: format!("{:?}", embryo.sex),
                    x: position.0,
                    y: position.1,
                    z: position.2,
                    age_hours: embryo.age_hours,
                    viability: embryo.viability,
                    water: inventory_component_amount(
                        &embryo.material_inventory,
                        TerrariumSpecies::Water,
                    ),
                    glucose: inventory_component_amount(
                        &embryo.material_inventory,
                        TerrariumSpecies::Glucose,
                    ),
                    amino_acids: inventory_component_amount(
                        &embryo.material_inventory,
                        TerrariumSpecies::AminoAcidPool,
                    ),
                    nucleotides: inventory_component_amount(
                        &embryo.material_inventory,
                        TerrariumSpecies::NucleotidePool,
                    ),
                    membrane_precursors: inventory_component_amount(
                        &embryo.material_inventory,
                        TerrariumSpecies::MembranePrecursorPool,
                    ),
                    oxygen: inventory_component_amount(
                        &embryo.material_inventory,
                        TerrariumSpecies::OxygenGas,
                    ),
                }
            })
            .collect();
        let full_explicit_microbes: Vec<TerrariumExplicitMicrobeSnapshot> = self
            .explicit_microbes
            .iter()
            .map(|cohort| TerrariumExplicitMicrobeSnapshot {
                x: cohort.x,
                y: cohort.y,
                z: cohort.z,
                guild: cohort.guild,
                represented_cells: cohort.represented_cells,
                represented_packets: cohort.represented_packets,
                genotype_id: cohort.identity.record.genotype_id,
                lineage_id: cohort.identity.record.lineage_id,
                catalog_id: cohort.identity.catalog.catalog_id,
                age_s: cohort.age_s,
                smoothed_energy: cohort.smoothed_energy,
                smoothed_stress: cohort.smoothed_stress,
            })
            .collect();

        TerrariumWorldSnapshot {
            seed_provenance: self.seed_provenance.clone(),
            conservation: crate::terrarium::conservation::audit_world(self),
            plants: self.plants.len(),
            full_plants,
            full_fruits,
            full_seeds,
            full_flies,
            full_fly_population,
            full_egg_clusters,
            full_fly_embryos,
            full_explicit_microbes,
            species_presence: self.species_presence(),
            fruits: live_fruits,
            seeds: self.seeds.len(),
            flies: self.flies.len(),
            food_remaining,
            fly_food_total: self
                .fruits
                .iter()
                .map(|f| f.source.sugar_content.max(0.0))
                .sum(),
            avg_fly_energy,
            avg_altitude,
            light: self.time_s.rem_euclid(DAY_LENGTH_S) / DAY_LENGTH_S,
            lunar_phase: self.lunar_phase(),
            moonlight: self.moonlight(),
            tidal_moisture_factor: self.tidal_moisture_factor(),
            temperature: self.temperature.iter().sum::<f32>() / air_volume,
            humidity: self.humidity.iter().sum::<f32>() / air_volume,
            mean_soil_moisture,
            mean_deep_moisture: mean(&self.deep_moisture),
            mean_litter_cover_fraction,
            mean_litter_surface_depth_mm,
            mean_microbes: self.microbial_biomass.iter().sum::<f32>() / n_cells.max(1.0),
            mean_symbionts: mean(&self.symbiont_biomass),
            mean_canopy: mean(&self.canopy_cover),
            mean_root_density: mean(&self.root_density),
            total_plant_cells: self.plants.len() as f32,
            mean_cell_vitality: 0.8,
            mean_cell_energy: 0.5,
            mean_division_pressure: 0.0,
            mean_soil_glucose,
            mean_soil_oxygen,
            mean_soil_ammonium,
            mean_soil_nitrate,
            mean_soil_redox,
            mean_soil_atp_flux,
            mean_soil_silicate_mineral,
            mean_soil_clay_mineral,
            mean_soil_carbonate_mineral,
            mean_soil_iron_oxide_mineral,
            mean_soil_dissolved_silicate,
            mean_soil_bicarbonate,
            mean_soil_surface_proton_load,
            mean_soil_calcium_bicarbonate_complex,
            mean_soil_sorbed_aluminum_hydroxide,
            mean_soil_sorbed_ferric_hydroxide,
            mean_soil_exchangeable_calcium,
            mean_soil_exchangeable_magnesium,
            mean_soil_exchangeable_potassium,
            mean_soil_exchangeable_sodium,
            mean_soil_exchangeable_aluminum,
            mean_soil_aqueous_iron,
            mean_atmospheric_co2: mean(&self.odorants[ATMOS_CO2_IDX]),
            mean_atmospheric_o2: mean(&self.odorants[ATMOS_O2_IDX]),
            ecology_event_count: self.ecology_events.len(),
            avg_fly_energy_charge,
            fly_plant_proximity_mean,
            fly_altitude_mean,
            fly_o2_gradient_correlation: 0.0,
            avg_wing_beat_freq,
            owned_fraction,
            substrate_backend: self.substrate.backend().as_str().to_string(),
            substrate_steps: self.substrate.step_count(),
            substrate_time_ms: self.substrate.time_ms(),
            time_s: self.time_s,
            atomistic_probes: self.atomistic_probes.len(),
            avg_fly_hunger: if self.fly_metabolisms.is_empty() {
                0.0
            } else {
                self.fly_metabolisms.iter().map(|m| m.hunger()).sum::<f32>()
                    / self.fly_metabolisms.len() as f32
            },
            avg_fly_trehalose_mm: if self.fly_metabolisms.is_empty() {
                0.0
            } else {
                self.fly_metabolisms
                    .iter()
                    .map(|m| m.hemolymph_trehalose_mm)
                    .sum::<f32>()
                    / self.fly_metabolisms.len() as f32
            },
            avg_fly_atp_mm: if self.fly_metabolisms.is_empty() {
                0.0
            } else {
                self.fly_metabolisms
                    .iter()
                    .map(|m| m.muscle_atp_mm)
                    .sum::<f32>()
                    / self.fly_metabolisms.len() as f32
            },
            fly_population_eggs: {
                let c = self.fly_pop.stage_census();
                c.total_eggs
            },
            fly_population_embryos: {
                let c = self.fly_pop.stage_census();
                c.embryos
            },
            fly_population_larvae: {
                let c = self.fly_pop.stage_census();
                c.larvae
            },
            fly_population_pupae: {
                let c = self.fly_pop.stage_census();
                c.pupae
            },
            fly_population_adults: {
                let c = self.fly_pop.stage_census();
                c.adults
            },
            fly_population_total: {
                let c = self.fly_pop.stage_census();
                c.total_individuals()
            },
            mean_air_pressure_kpa: mean(&self.air_pressure_kpa),
            climate: self.climate_state().cloned(),
            tracked_organisms: self.organism_registry.len(),
            named_organisms: self
                .organism_registry
                .values()
                .filter(|entry| entry.identity.display_name.is_some())
                .count(),
            ecology_events: self.ecology_events.clone(),
            cloud_cover: self.weather.cloud_cover,
            precipitation_rate_mm_h: self.weather.precipitation_rate_mm_h,
            weather_regime: format!("{:?}", self.weather.regime),
            mean_wind_speed: {
                let n = self.wind_x.len().max(1) as f32;
                (self.wind_x.iter().map(|v| v * v).sum::<f32>()
                    + self.wind_y.iter().map(|v| v * v).sum::<f32>())
                .sqrt()
                    / n.sqrt()
            },
            mean_wind_x: {
                let n = self.wind_x.len().max(1) as f32;
                self.wind_x.iter().sum::<f32>() / n
            },
            mean_wind_y: {
                let n = self.wind_y.len().max(1) as f32;
                self.wind_y.iter().sum::<f32>() / n
            },
            temperature_offset_c: self.weather.temperature_offset_c,
            lightning_flash: self.weather.lightning_flash,
            lightning_x: self.weather.lightning_x,
            lightning_y: self.weather.lightning_y,
            dew_intensity: self.weather.dew_intensity,
            mycorrhizal_connections: self.mycorrhizal_connection_count,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drosophila_population::{EggCluster, FlyEmbryoState, FlySex};
    use crate::molecular_atmosphere::WaterSourceState;
    use crate::terrarium::material_exchange::deposit_species_to_inventory;

    #[test]
    fn snapshot_exposes_full_fly_embryo_state() {
        let mut world =
            TerrariumWorld::demo(42, false).expect("demo world should build for snapshot test");
        let cs = world.config.cell_size_mm.max(1.0e-3);
        let mut clutch_inventory = RegionalMaterialInventory::new("snapshot:egg-clutch".into());
        deposit_species_to_inventory(&mut clutch_inventory, TerrariumSpecies::Water, 0.8);
        deposit_species_to_inventory(&mut clutch_inventory, TerrariumSpecies::Glucose, 0.6);
        deposit_species_to_inventory(&mut clutch_inventory, TerrariumSpecies::AminoAcidPool, 0.2);
        deposit_species_to_inventory(&mut clutch_inventory, TerrariumSpecies::NucleotidePool, 0.1);
        deposit_species_to_inventory(
            &mut clutch_inventory,
            TerrariumSpecies::MembranePrecursorPool,
            0.08,
        );
        let mut embryo_inventory = RegionalMaterialInventory::new("snapshot:embryo".into());
        deposit_species_to_inventory(&mut embryo_inventory, TerrariumSpecies::Water, 0.08);
        deposit_species_to_inventory(&mut embryo_inventory, TerrariumSpecies::Glucose, 0.05);
        deposit_species_to_inventory(
            &mut embryo_inventory,
            TerrariumSpecies::NucleotidePool,
            0.03,
        );
        let mut clutch = EggCluster {
            position: (5.0 * cs, 4.0 * cs),
            count: 0,
            age_hours: 0.0,
            substrate_quality: 0.9,
            material_inventory: clutch_inventory,
            embryos: vec![
                FlyEmbryoState {
                    id: 700,
                    sex: FlySex::Female,
                    offset_mm: (-0.08, 0.03),
                    age_hours: 6.0,
                    viability: 0.82,
                    material_inventory: embryo_inventory.clone(),
                },
                FlyEmbryoState {
                    id: 701,
                    sex: FlySex::Male,
                    offset_mm: (0.09, -0.02),
                    age_hours: 7.5,
                    viability: 0.76,
                    material_inventory: embryo_inventory,
                },
            ],
        };
        clutch.refresh_summary();
        world.fly_pop.egg_clusters.push(clutch);

        let snapshot = world.snapshot();

        assert_eq!(
            snapshot.full_fly_embryos.len() as u32,
            snapshot.fly_population_embryos,
            "snapshot should expose one explicit embryo record per lifecycle embryo"
        );
        assert_eq!(snapshot.full_fly_embryos.len(), 2);
        let embryo = &snapshot.full_fly_embryos[0];
        assert!(
            embryo.x.is_finite() && embryo.y.is_finite() && embryo.z.is_finite(),
            "embryo snapshot positions should be finite"
        );
        assert!(
            embryo.viability >= 0.0 && embryo.viability <= 1.0,
            "embryo viability should stay normalized"
        );
        assert!(
            embryo.water > 0.0 || embryo.glucose > 0.0 || embryo.nucleotides > 0.0,
            "embryo snapshot should expose embryo-owned chemistry"
        );
    }

    #[test]
    fn snapshot_exposes_finite_conservation_audit_domains() {
        let world =
            TerrariumWorld::demo(19, false).expect("demo world should build for audit snapshot");
        let audit = world.snapshot().conservation;
        let pools = [
            audit.substrate,
            audit.organism_inventories,
            audit.atmosphere_fields,
            audit.water_bodies,
            audit.detritus_fields,
            audit.organism_state_proxies,
            audit.explicit_domain_total,
            audit.reported_system_total,
        ];

        for pool in pools {
            assert!(pool.hydrogen.is_finite() && pool.hydrogen >= 0.0);
            assert!(pool.carbon.is_finite() && pool.carbon >= 0.0);
            assert!(pool.nitrogen.is_finite() && pool.nitrogen >= 0.0);
            assert!(pool.phosphorus.is_finite() && pool.phosphorus >= 0.0);
            assert!(pool.sulfur.is_finite() && pool.sulfur >= 0.0);
            assert!(pool.oxygen.is_finite() && pool.oxygen >= 0.0);
            assert!(pool.silicon.is_finite() && pool.silicon >= 0.0);
            assert!(pool.aluminum.is_finite() && pool.aluminum >= 0.0);
            assert!(pool.calcium.is_finite() && pool.calcium >= 0.0);
            assert!(pool.magnesium.is_finite() && pool.magnesium >= 0.0);
            assert!(pool.potassium.is_finite() && pool.potassium >= 0.0);
            assert!(pool.sodium.is_finite() && pool.sodium >= 0.0);
            assert!(pool.iron.is_finite() && pool.iron >= 0.0);
            assert!(pool.water.is_finite() && pool.water >= 0.0);
            assert!(pool.energy_equivalent.is_finite() && pool.energy_equivalent >= 0.0);
        }

        assert!(
            audit.explicit_domain_total.carbon > 0.0,
            "explicit domains should report some tracked carbon"
        );
        assert!(
            audit.reported_system_total.energy_equivalent > 0.0,
            "reported system total should expose usable energy carriers"
        );
    }

    #[test]
    fn conservation_audit_responds_to_explicit_domain_changes() {
        let mut world =
            TerrariumWorld::demo(23, false).expect("demo world should build for audit delta test");
        let before = world.snapshot().conservation;

        deposit_species_to_inventory(
            &mut world.plants[0].material_inventory,
            TerrariumSpecies::Glucose,
            0.75,
        );
        deposit_species_to_inventory(
            &mut world.plants[0].material_inventory,
            TerrariumSpecies::ExchangeableCalcium,
            0.50,
        );
        world
            .substrate
            .deposit_patch_species(TerrariumSpecies::Carbon, 0, 0, 0, 1, 0.50);
        world.substrate.deposit_patch_species(
            TerrariumSpecies::DissolvedSilicate,
            0,
            0,
            0,
            1,
            0.40,
        );
        world.humidity[0] += 0.20;
        world.waters.push(WaterSourceState {
            x: 0,
            y: 0,
            z: 0,
            volume: 0.35,
            evaporation_rate: 0.0,
            alive: true,
        });

        let after = world.snapshot().conservation;

        assert!(
            after.organism_inventories.energy_equivalent
                > before.organism_inventories.energy_equivalent,
            "plant inventory glucose should raise organism inventory energy-equivalent"
        );
        assert!(
            after.substrate.carbon > before.substrate.carbon,
            "substrate elemental carbon should reflect deposited carbon"
        );
        assert!(
            after.substrate.silicon > before.substrate.silicon,
            "substrate silicon should reflect dissolved silicate deposition"
        );
        assert!(
            after.organism_inventories.calcium > before.organism_inventories.calcium,
            "plant inventory calcium should raise organism inventory elemental audit"
        );
        assert!(
            after.atmosphere_fields.water > before.atmosphere_fields.water,
            "humidity field change should raise atmospheric water audit"
        );
        assert!(
            after.water_bodies.water > before.water_bodies.water,
            "new water body should raise explicit water-body audit"
        );
    }

    #[test]
    fn snapshot_exposes_weather_fields() {
        let mut world =
            TerrariumWorld::demo(77, false).expect("demo world should build for weather snapshot");
        world.weather.cloud_cover = 0.65;
        world.weather.precipitation_rate_mm_h = 3.5;
        world.weather.regime = crate::terrarium::WeatherRegime::Rain;
        // Seed some wind to verify mean_wind_speed
        for v in world.wind_x.iter_mut() {
            *v = 1.0;
        }
        for v in world.wind_y.iter_mut() {
            *v = 0.5;
        }

        let snap = world.snapshot();

        assert!(
            (snap.cloud_cover - 0.65).abs() < 1e-5,
            "snapshot cloud_cover should reflect weather state"
        );
        assert!(
            (snap.precipitation_rate_mm_h - 3.5).abs() < 1e-5,
            "snapshot precipitation should reflect weather state"
        );
        assert_eq!(
            snap.weather_regime, "Rain",
            "snapshot weather_regime should be Debug-formatted WeatherRegime"
        );
        assert!(
            snap.mean_wind_speed > 0.0,
            "snapshot mean_wind_speed should be positive when wind field is nonzero"
        );
        assert!(
            snap.mean_wind_speed.is_finite(),
            "mean_wind_speed should be finite"
        );
    }

    #[test]
    fn snapshot_exposes_wind_direction_and_weather_fields() {
        let mut world =
            TerrariumWorld::demo(44, false).expect("demo world should build for wind dir test");
        // Run a few frames so wind turbulence populates direction
        for _ in 0..10 {
            let _ = world.step_frame();
        }
        let snap = world.snapshot();
        assert!(
            snap.mean_wind_x.is_finite(),
            "mean_wind_x should be finite"
        );
        assert!(
            snap.mean_wind_y.is_finite(),
            "mean_wind_y should be finite"
        );
        assert!(
            snap.temperature_offset_c.is_finite(),
            "temperature_offset_c should be finite"
        );
        assert!(
            snap.dew_intensity.is_finite() && snap.dew_intensity >= 0.0,
            "dew_intensity should be finite and non-negative: {}",
            snap.dew_intensity
        );
        // Wind direction should have at least one nonzero component (wind turbulence is active)
        assert!(
            snap.mean_wind_x.abs() > 1e-6 || snap.mean_wind_y.abs() > 1e-6,
            "Wind direction should be nonzero after stepping: wx={}, wy={}",
            snap.mean_wind_x,
            snap.mean_wind_y
        );
    }
}
