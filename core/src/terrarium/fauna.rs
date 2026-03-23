//! Terrarium Fauna: Fly population, metabolism, and respiration.

use super::material_exchange::{
    deposit_species_to_inventory, inventory_component_amount, withdraw_species_from_inventory,
};
use super::*;
use crate::drosophila::{BodyState, DrosophilaScale, TerrariumFlyInputs};
use crate::drosophila_population::{
    Fly, FlySex, FruitResourcePatch, LinkedAdultFlyState, FLY_ENERGY_MAX,
};
use crate::fly_metabolism::FlyActivity;
use crate::organism_metabolism::OrganismMetabolism;
use crate::soil_fauna::step_soil_fauna;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct FruitContactSample {
    fruit_idx: Option<usize>,
    surface_contact: f32,
    sugar_taste: f32,
    bitter_taste: f32,
    amino_taste: f32,
}

impl TerrariumWorld {
    fn push_neural_fly_with_identity(
        &mut self,
        scale: DrosophilaScale,
        x: f32,
        y: f32,
        seed: u64,
        identity: OrganismIdentity,
    ) {
        let mut fly = DrosophilaSim::new(scale, seed);
        fly.set_body_state(x, y, 0.0, Some(0.0), None, None, None, None, None, None);
        self.flies.push(fly);
        self.fly_metabolisms.push(FlyMetabolism::default());
        self.fly_identities.push(identity);
    }

    fn linked_adult_fly_states(&self) -> Vec<LinkedAdultFlyState> {
        let cs = self.config.cell_size_mm.max(1.0e-3);
        self.fly_identities
            .iter()
            .enumerate()
            .filter_map(|(index, identity)| {
                let fly = self.flies.get(index)?;
                let body = fly.body_state();
                let energy_uj = self
                    .fly_metabolisms
                    .get(index)
                    .map(|metabolism| metabolism.energy_fraction() * FLY_ENERGY_MAX)
                    .unwrap_or(body.energy);
                Some(LinkedAdultFlyState {
                    organism_id: identity.organism_id,
                    position: (body.x * cs, body.y * cs, body.z * cs),
                    energy_uj,
                })
            })
            .collect()
    }

    fn sync_fly_identities(&mut self) {
        while self.fly_identities.len() > self.flies.len() {
            if let Some(identity) = self.fly_identities.pop() {
                self.mark_organism_dead(identity.organism_id);
            }
        }
        while self.fly_identities.len() < self.flies.len() {
            let identity = self.register_organism_identity(
                TerrariumOrganismKind::Fly,
                None,
                None,
                None,
                None,
                None,
                None,
            );
            self.fly_identities.push(identity);
        }
    }

    fn sample_fly_fruit_contact(&self, body: &BodyState) -> FruitContactSample {
        if body.is_flying || body.z > 0.2 {
            return FruitContactSample::default();
        }

        let head_reach = 0.32;
        let head_x = body.x + body.heading.cos() * head_reach;
        let head_y = body.y + body.heading.sin() * head_reach;
        let mut best = FruitContactSample::default();

        for (fruit_idx, fruit) in self.fruits.iter().enumerate() {
            if !fruit.source.alive || fruit.source.sugar_content <= 0.0 {
                continue;
            }

            let fx = fruit.source.x as f32;
            let fy = fruit.source.y as f32;
            let fruit_radius = fruit.radius.max(0.5);
            let body_contact = radial_contact_strength(body.x, body.y, fx, fy, fruit_radius + 0.58);
            let head_contact = radial_contact_strength(head_x, head_y, fx, fy, fruit_radius + 0.34);
            let surface_contact = body_contact.max(head_contact * 1.12).clamp(0.0, 1.0);
            if surface_contact <= best.surface_contact {
                continue;
            }
            let (sugar_taste, bitter_taste, amino_taste) =
                super::fruit_state::fruit_surface_taste_profile(
                    &fruit.composition,
                    surface_contact,
                );

            best = FruitContactSample {
                fruit_idx: Some(fruit_idx),
                surface_contact,
                sugar_taste,
                bitter_taste,
                amino_taste,
            };
        }

        best
    }

    fn consume_contacted_fruit(&mut self, fruit_idx: usize, requested_amount: f32) -> f32 {
        if requested_amount <= 0.0 {
            return 0.0;
        }
        let Some(fruit) = self.fruits.get_mut(fruit_idx) else {
            return 0.0;
        };
        if !fruit.source.alive || fruit.source.sugar_content <= 0.0 {
            return 0.0;
        }

        if fruit.material_inventory.is_empty() {
            deposit_species_to_inventory(
                &mut fruit.material_inventory,
                TerrariumSpecies::Glucose,
                fruit.source.sugar_content.max(0.0),
            );
            deposit_species_to_inventory(
                &mut fruit.material_inventory,
                TerrariumSpecies::Water,
                fruit.source.sugar_content.max(0.0) * 0.42,
            );
            deposit_species_to_inventory(
                &mut fruit.material_inventory,
                TerrariumSpecies::AminoAcidPool,
                fruit.source.sugar_content.max(0.0) * 0.14,
            );
        }
        let glucose_available =
            inventory_component_amount(&fruit.material_inventory, TerrariumSpecies::Glucose);
        let consumed = requested_amount
            .min(fruit.source.sugar_content)
            .min(glucose_available.max(0.0));
        let glucose_taken = withdraw_species_from_inventory(
            &mut fruit.material_inventory,
            TerrariumSpecies::Glucose,
            consumed,
        );
        let _ = withdraw_species_from_inventory(
            &mut fruit.material_inventory,
            TerrariumSpecies::Water,
            glucose_taken * 0.22,
        );
        let _ = withdraw_species_from_inventory(
            &mut fruit.material_inventory,
            TerrariumSpecies::AminoAcidPool,
            glucose_taken * 0.10,
        );
        fruit.source.sugar_content = (fruit.source.sugar_content - glucose_taken).max(0.0);
        fruit.organ.flesh_integrity = (fruit.organ.flesh_integrity - glucose_taken * 0.30).max(0.0);
        fruit.organ.seed_exposure = (fruit.organ.seed_exposure + glucose_taken * 0.18).min(1.0);
        if fruit.source.sugar_content <= 0.0 {
            fruit.source.alive = false;
        }
        glucose_taken
    }

    pub fn add_fly(&mut self, scale: DrosophilaScale, x: f32, y: f32, seed: u64) {
        let identity = self.register_organism_identity(
            TerrariumOrganismKind::Fly,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        self.push_neural_fly_with_identity(scale, x, y, seed, identity.clone());
        let sex = if seed & 1 == 0 {
            FlySex::Female
        } else {
            FlySex::Male
        };
        let adult = Fly::new_adult(
            self.fly_pop.next_id,
            sex,
            (
                x * self.config.cell_size_mm,
                y * self.config.cell_size_mm,
                0.0,
            ),
        )
        .with_organism_id(identity.organism_id);
        self.fly_pop.next_id += 1;
        self.fly_pop.add_fly(adult);
    }

    pub fn step_flies(&mut self) -> Result<(), String> {
        self.sync_fly_identities();
        if self.flies.is_empty() {
            return Ok(());
        }
        self.sensory_field.load_state(
            &self.odorants[ETHYL_ACETATE_IDX],
            &self.temperature,
            &self.wind_x,
            &self.wind_y,
            &self.wind_z,
            self.daylight(),
            &[],
        )?;

        // Capture pre-step O2 for hypoxia onset detection.
        let pre_o2: Vec<f32> = self
            .fly_metabolisms
            .iter()
            .map(|m| m.ambient_o2_fraction)
            .collect();

        // Phase 1: Pre-compute spatial O2 + altitude factor for each fly.
        let o2_values: Vec<f32> = self
            .flies
            .iter()
            .map(|fly| {
                let body = fly.body_state();
                let air_x = (body.x as usize).min(self.config.width.saturating_sub(1));
                let air_y = (body.y as usize).min(self.config.height.saturating_sub(1));
                let air_z = (body.z as usize).min(self.config.depth.max(1).saturating_sub(1));
                let local_o2 = self.sample_odorant_patch(ATMOS_O2_IDX, air_x, air_y, air_z, 1);
                let altitude_factor = (1.0 - body.z * 0.01).clamp(0.5, 1.0);
                (local_o2 * altitude_factor).max(0.0)
            })
            .collect();

        for i in 0..self.flies.len() {
            // Phase 1: Set spatial O2 on metabolism + detect hypoxia onset.
            if i < self.fly_metabolisms.len() {
                self.fly_metabolisms[i].set_ambient_o2(o2_values[i]);
                if i < pre_o2.len() && pre_o2[i] >= 0.15 && o2_values[i] < 0.15 {
                    let body = self.flies[i].body_state();
                    self.ecology_events
                        .push(EcologyTelemetryEvent::FlyHypoxiaOnset {
                            x: body.x,
                            y: body.y,
                            ambient_o2: o2_values[i],
                            altitude: body.z,
                        });
                }
                let hunger = self.fly_metabolisms[i].hunger();
                let phase = self.fly_metabolisms[i].circadian.phase_hours;
                let circadian_activity = self.fly_metabolisms[i].circadian.activity_level;
                let energy_norm = self.fly_metabolisms[i].energy_fraction() * 100.0;
                self.flies[i].set_energy(energy_norm);
                self.flies[i].set_homeostatic_inputs(hunger, phase, circadian_activity);
            }

            // Local air density calculation: derived from real molecular partial pressures
            let cell_idx = {
                let body = self.flies[i].body_state();
                let fx = body.x.round().clamp(0.0, (self.config.width - 1) as f32) as usize;
                let fy = body.y.round().clamp(0.0, (self.config.height - 1) as f32) as usize;
                fy * self.config.width + fx
            };
            let air_density = self.calculate_local_air_density(cell_idx);

            // Body step: sensory -> brain -> motor -> physics.
            let body_val = self.flies[i].body_state().clone();
            let sample = self.sensory_field.sample_fly(
                body_val.x,
                body_val.y,
                body_val.z,
                body_val.heading,
                body_val.is_flying,
            );
            let fruit_contact = self.sample_fly_fruit_contact(&body_val);

            let report = self.flies[i].body_step_terrarium_inputs(
                TerrariumFlyInputs {
                    odorant: sample.odorant,
                    left_odorant: sample.left_odorant,
                    right_odorant: sample.right_odorant,
                    left_light: sample.left_light,
                    right_light: sample.right_light,
                    temperature: sample.temperature,
                    sugar_taste: sample.sugar_taste.max(fruit_contact.sugar_taste),
                    bitter_taste: sample.bitter_taste.max(fruit_contact.bitter_taste),
                    amino_taste: sample.amino_taste.max(fruit_contact.amino_taste),
                    wind_x: sample.wind_x,
                    wind_y: sample.wind_y,
                    wind_z: sample.wind_z,
                    surface_contact: sample.food_available.max(fruit_contact.surface_contact),
                },
                air_density,
            );

            // Phase 3: Metabolic ingest + telemetry. Post-ingestive sensory
            // feedback is applied here after real resource consumption.
            let consumed_food = if report.consumed_food > 0.0 {
                fruit_contact
                    .fruit_idx
                    .map(|fruit_idx| self.consume_contacted_fruit(fruit_idx, report.consumed_food))
                    .unwrap_or(0.0)
            } else {
                0.0
            };
            if consumed_food > 0.0 {
                if i < self.fly_metabolisms.len() {
                    let hunger_signal = self.fly_metabolisms[i].hunger();
                    self.fly_metabolisms[i].ingest(consumed_food);
                    self.flies[i].register_ingestion_feedback(consumed_food, hunger_signal);
                    self.ecology_events.push(EcologyTelemetryEvent::FlyFeeding {
                        x: report.x,
                        y: report.y,
                        sugar_ingested_mg: consumed_food,
                        trehalose_mm: self.fly_metabolisms[i].hemolymph_trehalose_mm,
                    });
                }
                self.fly_food_total += consumed_food;
            }

            // Herbivore leaf grazing: flies occasionally feed on leaf tissue.
            // Grazing probability is driven by the fly's hunger (Hill kinetics on
            // metabolic need) and suppressed by the plant's defense metabolites
            // (jasmonate + salicylate deterrence via Hill repression).
            // This provides the biological trigger for the JA defense cascade
            // (Phase 5 VOC signaling).
            if !body_val.is_flying && consumed_food <= 0.0 && !self.plants.is_empty() {
                let hunger = if i < self.fly_metabolisms.len() {
                    self.fly_metabolisms[i].hunger()
                } else {
                    0.5
                };
                // Grazing drive: Hill function on hunger — hungrier flies are more
                // likely to graze. Km=0.6 means grazing ramps up steeply once
                // hunger exceeds 60%. n=2 gives cooperative sigmoidal response.
                let grazing_drive = crate::botany::physiology_bridge::hill(hunger, 0.6, 2.0);
                // Base probability per step is low: ~0.3% at max hunger.
                // This emerges from the hill output (0-1) scaled by the per-step rate.
                let graze_prob = grazing_drive * 0.003;
                let roll: f32 = self.rng.gen();
                if roll < graze_prob {
                    // Find nearest plant to fly position
                    let fly_x = body_val.x;
                    let fly_y = body_val.y;
                    let mut best_plant_idx = None;
                    let mut best_dist_sq = f32::MAX;
                    for (pidx, plant) in self.plants.iter().enumerate() {
                        let dx = fly_x - plant.x as f32;
                        let dy = fly_y - plant.y as f32;
                        let d2 = dx * dx + dy * dy;
                        // Only graze plants within ~2 cells (the fly's immediate area)
                        if d2 < 4.0 && d2 < best_dist_sq {
                            best_dist_sq = d2;
                            best_plant_idx = Some(pidx);
                        }
                    }
                    if let Some(pidx) = best_plant_idx {
                        let plant = &mut self.plants[pidx];
                        // Defense deterrence: Hill on JA+SA pool — defended plants
                        // repel grazing. Km=5.0 means ~5 molecules of combined
                        // defense metabolites provide 50% deterrence. n=2 gives
                        // cooperative switch behavior matching real herbivore
                        // avoidance of defended tissue.
                        let defense_pool = plant.metabolome.jasmonate_count
                            + plant.metabolome.salicylate_count;
                        let deterrence = crate::botany::physiology_bridge::hill(
                            defense_pool as f32, 5.0, 2.0,
                        );
                        let deterrence_roll: f32 = self.rng.gen();
                        // Fly avoids the plant proportionally to deterrence
                        if deterrence_roll >= deterrence {
                            // Successful grazing: leaf damage proportional to leaf available
                            let leaf_bm = plant.physiology.leaf_biomass();
                            // Consume a fraction of leaf biomass via Michaelis-Menten:
                            // small bites relative to available leaf area.
                            let bite_fraction = crate::botany::physiology_bridge::michaelis_menten(
                                leaf_bm, 0.3,
                            );
                            let leaf_consumed = (bite_fraction * 0.008).min(leaf_bm * 0.05);
                            if leaf_consumed > 1.0e-6 {
                                // Apply damage to plant
                                plant.physiology.apply_leaf_grazing(leaf_consumed);
                                // Mechanical damage feeds into JA_RESPONSE via
                                // EnvironmentState.mechanical_stress in the plant step.
                                plant.morphology.mechanical_damage =
                                    (plant.morphology.mechanical_damage + 0.04).min(1.0);
                                // Fly gains energy from leaf tissue (less than fruit)
                                if i < self.fly_metabolisms.len() {
                                    self.fly_metabolisms[i].ingest(leaf_consumed * 0.3);
                                }
                                self.ecology_events.push(
                                    EcologyTelemetryEvent::FlyGrazing {
                                        x: fly_x,
                                        y: fly_y,
                                        leaf_consumed,
                                        plant_deterrence: deterrence,
                                    },
                                );
                            }
                        }
                    }
                }
            }

            // Respiration: Apply O2/CO2 flux from fly body to atmosphere
            let (o2_flux, co2_flux) = self.flies[i].body_state().calculate_respiration_flux();
            let b = self.flies[i].body_state();
            let bx = b.x.round().clamp(0.0, (self.config.width - 1) as f32) as usize;
            let by = b.y.round().clamp(0.0, (self.config.height - 1) as f32) as usize;
            let bz = (b.z * 0.1)
                .round()
                .clamp(0.0, (self.config.depth - 1) as f32) as usize;

            self.exchange_atmosphere_odorant(ATMOS_O2_IDX, bx, by, bz, 1, o2_flux);
            self.exchange_atmosphere_odorant(ATMOS_CO2_IDX, bx, by, bz, 1, co2_flux);

            // Phase 5: Neural metabolic cost -- brain firing rate -> ATP demand.
            if i < self.fly_metabolisms.len() {
                let firing_rate = self.flies[i].mean_firing_rate();
                let neural_frac = (firing_rate / 50.0).clamp(0.0, 1.0);
                self.fly_metabolisms[i].set_neural_activity(neural_frac);
            }
        }
        Ok(())
    }

    pub fn step_soil_fauna(&mut self, eco_dt: f32) {
        let dt_hours = eco_dt / 3600.0;
        if dt_hours <= 0.0 {
            return;
        }
        let mut nitrifier_approx: Vec<f32> =
            self.microbial_biomass.iter().map(|b| b * 0.1).collect();
        let _result = step_soil_fauna(
            &mut self.earthworm_population,
            &mut self.nematode_guilds,
            &mut self.microbial_biomass,
            &mut nitrifier_approx,
            &mut self.organic_matter,
            &mut self.substrate,
            &self.moisture,
            &self.temperature,
            dt_hours,
            (
                self.config.width,
                self.config.height,
                self.config.depth.max(1),
            ),
        );
    }

    pub fn step_fly_population(&mut self, eco_dt: f32) {
        let dt_hours = eco_dt / 3600.0;
        if dt_hours <= 0.0 {
            return;
        }
        let mean_temp = mean(&self.temperature);
        let mean_humidity = mean(&self.humidity);
        let fruit_resources: Vec<FruitResourcePatch> = self
            .fruits
            .iter()
            .filter(|f| f.source.alive && f.source.sugar_content > 0.01)
            .map(|f| {
                let cs = self.config.cell_size_mm;
                let glucose =
                    inventory_component_amount(&f.material_inventory, TerrariumSpecies::Glucose)
                        .max(f.source.sugar_content * 0.42);
                let water =
                    inventory_component_amount(&f.material_inventory, TerrariumSpecies::Water)
                        .max(f.composition.water_fraction * (0.28 + f.radius * 0.12));
                let amino = inventory_component_amount(
                    &f.material_inventory,
                    TerrariumSpecies::AminoAcidPool,
                )
                .max(f.source.sugar_content * (0.04 + f.composition.amino_fraction * 0.22));
                let oxygen =
                    inventory_component_amount(&f.material_inventory, TerrariumSpecies::OxygenGas)
                        .max((1.0 - f.organ.rot_progress).clamp(0.0, 1.0) * 0.20);
                FruitResourcePatch {
                    position: (f.source.x as f32 * cs, f.source.y as f32 * cs, 0.0),
                    substrate_quality: clamp(
                        0.24 + glucose * 0.20 + amino * 0.14 + water * 0.10 + oxygen * 0.08
                            - f.organ.rot_progress * 0.12,
                        0.0,
                        1.4,
                    ),
                    water,
                    glucose,
                    amino_acids: amino,
                    oxygen,
                }
            })
            .collect();
        let linked_adults = self.linked_adult_fly_states();
        self.fly_pop
            .sync_linked_adults(&linked_adults, &fruit_resources);
        self.fly_pop
            .step_with_resources(dt_hours, mean_temp, &fruit_resources, mean_humidity);
        let eclosed = self.fly_pop.drain_eclosed();
        for eclosed_fly in eclosed {
            if self.flies.len() >= 12 {
                break;
            }
            let cs = self.config.cell_size_mm;
            let fx = (eclosed_fly.position.0 / cs).clamp(1.0, self.config.width as f32 - 2.0);
            let fy = (eclosed_fly.position.1 / cs).clamp(1.0, self.config.height as f32 - 2.0);
            let seed = self.rng.gen();
            let identity = eclosed_fly
                .organism_id
                .and_then(|organism_id| {
                    self.organism_registry
                        .get(&organism_id)
                        .map(|entry| entry.identity.clone())
                })
                .unwrap_or_else(|| {
                    let identity = self.register_organism_identity(
                        TerrariumOrganismKind::Fly,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    );
                    let _ = self
                        .fly_pop
                        .assign_organism_id(eclosed_fly.fly_id, identity.organism_id);
                    identity
                });
            self.push_neural_fly_with_identity(DrosophilaScale::Small, fx, fy, seed, identity);
            if let Some(m) = self.fly_metabolisms.last_mut() {
                m.fat_body_glycogen_mg = 0.009;
                m.fat_body_lipid_mg = 0.027;
                m.hemolymph_trehalose_mm = 15.0;
            }
            self.ecology_events
                .push(EcologyTelemetryEvent::FlyEclosed { x: fx, y: fy });
        }
    }

    pub fn step_fly_metabolism(&mut self, eco_dt: f32) {
        self.sync_fly_identities();
        if eco_dt <= 0.0 {
            return;
        }
        while self.fly_metabolisms.len() < self.flies.len() {
            self.fly_metabolisms.push(FlyMetabolism::default());
        }
        self.fly_metabolisms.truncate(self.flies.len());

        let mean_temp = mean(&self.temperature);
        let daylight = self.daylight();
        let moonlight = self.moonlight();
        let chrono_config = self.chronobiology_config.clone();

        for i in 0..self.flies.len() {
            let (bx, by, is_flying, speed) = {
                let body = self.flies[i].body_state();
                (body.x, body.y, body.is_flying, body.speed)
            };
            let locomotor_activity = if is_flying {
                let effort = (speed / 200.0).clamp(0.25, 1.0);
                FlyActivity::Flying(effort)
            } else if speed > 0.05 {
                let speed_frac = (speed / 2.0).clamp(0.0, 1.0);
                FlyActivity::Walking(speed_frac)
            } else {
                FlyActivity::Resting
            };

            let metabolism = &mut self.fly_metabolisms[i];
            let pre_starved =
                metabolism.hemolymph_trehalose_mm < 5.0 || metabolism.energy_charge() < 0.35;
            let pre_atp_crash = metabolism.energy_charge() < 0.5 || metabolism.muscle_atp_mm < 1.0;

            metabolism.set_activity(locomotor_activity);
            metabolism.step_circadian(eco_dt, daylight, moonlight, &chrono_config);
            metabolism.step(mean_temp, eco_dt);

            let post_hunger = metabolism.hunger();
            let post_phase = metabolism.circadian.phase_hours;
            let post_activity = metabolism.circadian.activity_level;
            let post_energy_norm = metabolism.energy_fraction() * 100.0;
            let post_energy_charge = metabolism.energy_charge();
            let post_starved = metabolism.hemolymph_trehalose_mm < 5.0 || post_energy_charge < 0.35;
            let post_atp_crash = post_energy_charge < 0.5 || metabolism.muscle_atp_mm < 1.0;

            self.flies[i].set_energy(post_energy_norm);
            self.flies[i].set_homeostatic_inputs(post_hunger, post_phase, post_activity);

            if !pre_starved && post_starved {
                self.ecology_events
                    .push(EcologyTelemetryEvent::FlyStarvationOnset {
                        x: bx,
                        y: by,
                        trehalose_mm: metabolism.hemolymph_trehalose_mm,
                        glycogen_mg: metabolism.fat_body_glycogen_mg,
                    });
            }
            if !pre_atp_crash && post_atp_crash {
                self.ecology_events
                    .push(EcologyTelemetryEvent::FlyAtpCrash {
                        x: bx,
                        y: by,
                        energy_charge: post_energy_charge,
                        trehalose_mm: metabolism.hemolymph_trehalose_mm,
                    });
            }
        }
    }
}

fn radial_contact_strength(x: f32, y: f32, center_x: f32, center_y: f32, reach: f32) -> f32 {
    let reach = reach.max(1.0e-3);
    let dx = x - center_x;
    let dy = y - center_y;
    let dist = (dx * dx + dy * dy).sqrt();
    ((reach - dist) / reach).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drosophila_population::{Fly, FlyLifeStage, FlyPopulation, FlySex};

    #[test]
    fn explicit_fruit_contact_depends_on_body_overlap() {
        let mut world = TerrariumWorld::demo(41, false).unwrap();
        world.plants.clear();
        world.seeds.clear();
        world.explicit_microbes.clear();
        world.flies.clear();
        world.waters.clear();
        world.fruits.clear();
        world.add_fruit(8, 8, 1.2, Some(1.0));
        world.add_fly(DrosophilaScale::Tiny, 8.4, 8.1, 7);
        world.flies[0].set_body_state(
            8.4,
            8.1,
            0.0,
            Some(0.0),
            Some(0.0),
            Some(false),
            Some(0.0),
            None,
            None,
            None,
        );
        let near = world.flies[0].body_state().clone();
        world.flies[0].set_body_state(
            13.0,
            8.0,
            0.0,
            Some(0.0),
            Some(0.0),
            Some(false),
            Some(0.0),
            None,
            None,
            None,
        );
        let far = world.flies[0].body_state().clone();

        let near_contact = world.sample_fly_fruit_contact(&near);
        let far_contact = world.sample_fly_fruit_contact(&far);

        assert_eq!(near_contact.fruit_idx, Some(0));
        assert!(near_contact.surface_contact > 0.2);
        assert!(near_contact.sugar_taste > 0.1);
        assert_eq!(far_contact, FruitContactSample::default());
    }

    #[test]
    fn fruit_surface_chemistry_depends_on_fruit_identity() {
        let mut world = TerrariumWorld::demo(45, false).unwrap();
        world.fruits.clear();
        world.add_fruit(8, 8, 0.9, Some(1.0));
        let contact = 0.85;

        let apple_genome =
            crate::terrarium::plant_species::genome_for_taxonomy(3750, &mut world.rng);
        world.fruits[0].taxonomy_id = apple_genome.taxonomy_id;
        world.fruits[0].source_genome = apple_genome;
        world.fruits[0].source.ripeness = 0.82;
        world.fruits[0].source.sugar_content = 0.9;
        world.fruits[0].composition = super::super::fruit_state::fruit_composition_from_parent(
            &world.fruits[0].source_genome,
            world.fruits[0].taxonomy_id,
            world.fruits[0].source.sugar_content,
            world.fruits[0].source.ripeness,
        );
        let apple_taste = super::super::fruit_state::fruit_surface_taste_profile(
            &world.fruits[0].composition,
            contact,
        );

        let lemon_genome =
            crate::terrarium::plant_species::genome_for_taxonomy(2708, &mut world.rng);
        world.fruits[0].taxonomy_id = lemon_genome.taxonomy_id;
        world.fruits[0].source_genome = lemon_genome;
        world.fruits[0].source.ripeness = 0.82;
        world.fruits[0].source.sugar_content = 0.9;
        world.fruits[0].composition = super::super::fruit_state::fruit_composition_from_parent(
            &world.fruits[0].source_genome,
            world.fruits[0].taxonomy_id,
            world.fruits[0].source.sugar_content,
            world.fruits[0].source.ripeness,
        );
        let lemon_taste = super::super::fruit_state::fruit_surface_taste_profile(
            &world.fruits[0].composition,
            contact,
        );

        assert!(
            (apple_taste.0 - lemon_taste.0).abs() > 1.0e-4
                || (apple_taste.1 - lemon_taste.1).abs() > 1.0e-4
                || (apple_taste.2 - lemon_taste.2).abs() > 1.0e-4,
            "fruit identity should change the tasted surface chemistry"
        );
        assert!(
            lemon_taste.1 > apple_taste.1,
            "citrus fruit should present a stronger bitter surface than orchard fruit"
        );
    }

    #[test]
    fn contacted_fruit_consumption_is_bounded_by_real_remaining_sugar() {
        let mut world = TerrariumWorld::demo(43, false).unwrap();
        world.fruits.clear();
        world.add_fruit(6, 6, 0.2, Some(1.0));

        let consumed = world.consume_contacted_fruit(0, 0.5);

        assert!((consumed - 0.2).abs() <= 1.0e-6);
        assert!(world.fruits[0].source.sugar_content <= 1.0e-6);
        assert!(!world.fruits[0].source.alive);
    }

    #[test]
    fn add_fly_seeds_linked_lifecycle_adult() {
        let mut world = TerrariumWorld::demo(53, false).unwrap();
        world.flies.clear();
        world.fly_metabolisms.clear();
        world.fly_identities.clear();
        world.fly_pop = FlyPopulation::new(53);

        world.add_fly(DrosophilaScale::Tiny, 5.0, 5.0, 100);

        let organism_id = world.fly_identities[0].organism_id;
        let linked = world
            .fly_pop
            .fly_by_organism_id(organism_id)
            .expect("added neural fly should seed a linked lifecycle adult");
        assert!(matches!(linked.stage, FlyLifeStage::Adult { .. }));
        assert_eq!(
            linked.position,
            (
                5.0 * world.config.cell_size_mm,
                5.0 * world.config.cell_size_mm,
                0.0
            )
        );
    }

    #[test]
    fn eclosed_population_adult_receives_identity_before_neural_spawn() {
        let mut world = TerrariumWorld::demo(59, false).unwrap();
        world.flies.clear();
        world.fly_metabolisms.clear();
        world.fly_identities.clear();
        world.fly_pop = FlyPopulation::new(59);

        let mut pupa = Fly::new_adult(999, FlySex::Female, (5.0, 5.0, 0.0));
        pupa.stage = FlyLifeStage::Pupa { age_hours: 95.0 };
        world.fly_pop.add_fly(pupa);
        world.step_fly_population(7200.0);

        let spawned_id = world.fly_identities[0].organism_id;
        assert!(
            world.fly_pop
                .flies
                .iter()
                .any(|fly| fly.organism_id == Some(spawned_id)),
            "eclosed lifecycle adult should receive the same terrarium identity as the spawned neural fly"
        );
    }
}
