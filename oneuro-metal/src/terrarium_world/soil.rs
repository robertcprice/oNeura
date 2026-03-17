use super::*;

impl TerrariumWorld {
    fn rebuild_water_mask(&mut self) {
        self.water_mask.fill(0.0);
        for water in &self.waters {
            if !water.alive {
                continue;
            }
            let amplitude = clamp(water.volume / 140.0, 0.06, 1.0);
            deposit_2d(
                &mut self.water_mask,
                self.config.width,
                self.config.height,
                water.x,
                water.y,
                2,
                amplitude,
            );
        }
    }

    pub(super) fn step_soil_fauna_phase(&mut self, eco_dt: f32) {
        let eco_dt_hours = eco_dt / 3600.0;
        let dims = (
            self.config.width,
            self.config.height,
            self.config.depth.max(1),
        );
        // Clone read-only inputs to avoid borrow conflict with &mut substrate.
        let hydration = self.substrate.hydration.clone();
        // soil_temperature not on substrate; use world temperature field sampled at surface.
        let temperature: Vec<f32> = (0..dims.0 * dims.1)
            .map(|i| {
                let x = i % dims.0;
                let y = i / dims.0;
                self.sample_temperature_at(x, y, 0)
            })
            .collect();
        let _result = step_soil_fauna(
            &mut self.earthworm_population,
            &mut self.nematode_guilds,
            &mut self.microbial_biomass,
            &mut self.nitrifier_biomass,
            &mut self.organic_matter,
            &mut self.substrate,
            &hydration,
            &temperature,
            eco_dt_hours,
            dims,
        );
    }

    pub(super) fn sync_substrate_controls(&mut self) -> Result<(), String> {
        let plane = self.config.width * self.config.height;
        let total = plane * self.config.depth.max(1);
        let mut hydration = vec![0.0f32; total];
        let mut microbes = vec![0.0f32; total];
        let mut plant_drive = vec![0.0f32; total];
        let depth = self.config.depth.max(1);
        for z in 0..depth {
            let z_frac = if depth > 1 {
                z as f32 / (depth - 1) as f32
            } else {
                0.0
            };
            for i in 0..plane {
                let gid = z * plane + i;
                hydration[gid] = clamp(
                    self.moisture[i] * (1.0 - z_frac * 0.55) + self.deep_moisture[i] * z_frac,
                    0.02,
                    1.0,
                );
                microbes[gid] = clamp(
                    self.microbial_biomass[i] * (0.65 + self.moisture[i] * 0.55)
                        + self.symbiont_biomass[i] * (0.55 + z_frac * 0.30),
                    0.02,
                    1.2,
                );
                plant_drive[gid] = clamp(
                    self.root_density[i] * (1.0 - z_frac * 0.35) * (0.35 + self.daylight() * 0.65),
                    0.0,
                    1.5,
                );
            }
        }
        self.substrate.set_hydration_field(&hydration)?;
        self.substrate.set_microbial_activity_field(&microbes)?;
        self.substrate.set_plant_drive_field(&plant_drive)?;
        Ok(())
    }

    pub(super) fn step_surface_respiration(&mut self, eco_dt: f32) {
        let depth = self.config.depth.max(1);
        for y in 0..self.config.height {
            for x in 0..self.config.width {
                let flat = idx2(self.config.width, x, y);
                let local_substrate = self.litter_carbon[flat] * 1.10
                    + self.root_exudates[flat] * 1.35
                    + self.organic_matter[flat] * 0.90;
                let moisture_factor = clamp(
                    (self.moisture[flat] + self.deep_moisture[flat] * 0.35) / 0.48,
                    0.0,
                    1.6,
                );
                let oxygen_factor = clamp(1.15 - self.deep_moisture[flat] * 0.55, 0.35, 1.1);
                let aeration_factor = clamp(
                    1.10 - self.moisture[flat] * 0.55 - self.deep_moisture[flat] * 0.45,
                    0.05,
                    1.15,
                );
                let anoxia_factor = clamp(
                    (self.deep_moisture[flat] * 0.95 + self.moisture[flat] * 0.18)
                        - oxygen_factor * 0.28,
                    0.02,
                    1.3,
                );
                let root_factor = 1.0 + self.root_density[flat] * 0.08;
                let substrate_gate = clamp(local_substrate / 0.08, 0.0, 1.35);
                let explicit_authority = self.explicit_microbe_authority[flat].clamp(0.0, 0.95);
                let explicit_activity = self.explicit_microbe_activity[flat].max(0.0);
                if explicit_authority >= EXPLICIT_OWNERSHIP_THRESHOLD {
                    continue;
                }
                let coarse_biology_factor = if explicit_authority >= EXPLICIT_OWNERSHIP_THRESHOLD {
                    0.0
                } else {
                    1.0 - explicit_authority
                };
                let decomposer_trait_factor = trait_match(
                    self.microbial_copiotroph_fraction[flat],
                    microbial_copiotroph_target(
                        substrate_gate,
                        moisture_factor,
                        oxygen_factor,
                        root_factor,
                    ),
                );
                let decomposer_packet_factor = packet_surface_factor(
                    self.microbial_cells[flat],
                    self.microbial_packets[flat],
                    MICROBIAL_PACKET_TARGET_CELLS,
                );
                let decomposer_active = self.microbial_cells[flat]
                    * (1.0 - self.microbial_dormancy[flat]).clamp(0.02, 1.0)
                    * (0.25 + 0.75 * self.microbial_vitality[flat]).clamp(0.0, 1.25)
                    * (0.55 + 0.45 * self.microbial_reserve[flat]).clamp(0.20, 1.25)
                    * decomposer_packet_factor
                    * decomposer_trait_factor
                    * 0.035
                    * coarse_biology_factor;
                let nitrifier_trait_factor = trait_match(
                    self.nitrifier_aerobic_fraction[flat],
                    nitrifier_aerobic_target(oxygen_factor, aeration_factor, anoxia_factor),
                );
                let nitrifier_packet_factor = packet_surface_factor(
                    self.nitrifier_cells[flat],
                    self.nitrifier_packets[flat],
                    NITRIFIER_PACKET_TARGET_CELLS,
                );
                let nitrifier_active = self.nitrifier_cells[flat]
                    * (1.0 - self.nitrifier_dormancy[flat]).clamp(0.02, 1.0)
                    * (0.25 + 0.75 * self.nitrifier_vitality[flat]).clamp(0.0, 1.25)
                    * (0.55 + 0.45 * self.nitrifier_reserve[flat]).clamp(0.20, 1.25)
                    * nitrifier_packet_factor
                    * nitrifier_trait_factor
                    * 0.030
                    * coarse_biology_factor;
                let denitrifier_trait_factor = trait_match(
                    self.denitrifier_anoxic_fraction[flat],
                    denitrifier_anoxic_target(
                        anoxia_factor,
                        self.deep_moisture[flat],
                        oxygen_factor,
                    ),
                );
                let denitrifier_packet_factor = packet_surface_factor(
                    self.denitrifier_cells[flat],
                    self.denitrifier_packets[flat],
                    DENITRIFIER_PACKET_TARGET_CELLS,
                );
                let denitrifier_active = self.denitrifier_cells[flat]
                    * (1.0 - self.denitrifier_dormancy[flat]).clamp(0.02, 1.0)
                    * (0.25 + 0.75 * self.denitrifier_vitality[flat]).clamp(0.0, 1.25)
                    * (0.55 + 0.45 * self.denitrifier_reserve[flat]).clamp(0.20, 1.25)
                    * denitrifier_packet_factor
                    * denitrifier_trait_factor
                    * 0.030
                    * coarse_biology_factor;
                let microbial_drive = (decomposer_active
                    + self.microbial_biomass[flat] * 0.35 * coarse_biology_factor)
                    * (0.55 + self.moisture[flat] * 0.75)
                    + explicit_activity * (0.62 + self.moisture[flat] * 0.44)
                    + self.symbiont_biomass[flat] * 0.22
                    + (nitrifier_active
                        + self.nitrifier_biomass[flat] * 0.22 * coarse_biology_factor)
                        * (0.12 + self.nitrification_potential[flat] * 0.20)
                    + (denitrifier_active
                        + self.denitrifier_biomass[flat] * 0.22 * coarse_biology_factor)
                        * (0.20 + self.denitrification_potential[flat] * 0.26)
                    + self.root_exudates[flat] * 1.35
                    + self.litter_carbon[flat] * 0.42
                    + self.organic_matter[flat] * 0.18;
                let soil_co2_flux = clamp(microbial_drive * eco_dt * 0.000035, 0.0, 0.0025);
                if soil_co2_flux <= 2.0e-6 {
                    continue;
                }
                let soil_o2_flux = -soil_co2_flux * (0.80 + self.moisture[flat] * 0.10);
                let soil_humidity_flux = clamp(
                    self.moisture[flat] * microbial_drive * eco_dt * 0.000008,
                    0.0,
                    0.0015,
                );
                self.exchange_atmosphere_flux_bundle(
                    x,
                    y,
                    0,
                    1,
                    soil_co2_flux,
                    soil_o2_flux,
                    soil_humidity_flux,
                );
            }
        }

        let fruit_fluxes = self
            .fruits
            .iter()
            .filter(|fruit| fruit.source.alive && fruit.source.sugar_content > 0.01)
            .map(|fruit| {
                let flat = idx2(
                    self.config.width,
                    fruit.source.x.min(self.config.width - 1),
                    fruit.source.y.min(self.config.height - 1),
                );
                let microbial = self.microbial_biomass[flat];
                let respiration_driver = fruit.source.sugar_content.max(0.0)
                    * (0.35 + fruit.source.ripeness.max(0.0) * 0.65)
                    * (1.0 + microbial * 6.0);
                let fruit_co2_flux = clamp(respiration_driver * eco_dt * 0.00016, 0.0, 0.004);
                let fruit_o2_flux = -fruit_co2_flux * (0.86 + microbial * 0.08);
                let fruit_humidity_flux = clamp(respiration_driver * eco_dt * 0.00005, 0.0, 0.0018);
                (
                    fruit.source.x,
                    fruit.source.y,
                    fruit.source.z.min(depth - 1),
                    fruit.radius.round().max(1.0) as usize,
                    fruit_co2_flux,
                    fruit_o2_flux,
                    fruit_humidity_flux,
                )
            })
            .collect::<Vec<_>>();
        for (x, y, z, radius, co2_flux, o2_flux, humidity_flux) in fruit_fluxes {
            self.exchange_atmosphere_flux_bundle(x, y, z, radius, co2_flux, o2_flux, humidity_flux);
        }
    }

    pub(super) fn couple_soil_atmosphere_gases(&mut self, eco_dt: f32) {
        let mut deposits = Vec::new();
        let mut extractions = Vec::new();
        let patch_radius = 1usize;

        for y in 0..self.config.height {
            for x in 0..self.config.width {
                let flat = idx2(self.config.width, x, y);
                let porosity = clamp(
                    self.soil_structure[flat] * (1.08 - self.moisture[flat] * 0.58)
                        + self.deep_moisture[flat] * 0.08,
                    0.05,
                    1.1,
                );
                let air_o2 = self.sample_odorant_patch(ATMOS_O2_IDX, x, y, 0, patch_radius);
                let air_co2 = self.sample_odorant_patch(ATMOS_CO2_IDX, x, y, 0, patch_radius);
                let soil_o2 = self.substrate.patch_mean_species(
                    TerrariumSpecies::OxygenGas,
                    x,
                    y,
                    0,
                    patch_radius,
                );
                let soil_co2 = self.substrate.patch_mean_species(
                    TerrariumSpecies::CarbonDioxide,
                    x,
                    y,
                    0,
                    patch_radius,
                );

                let eq_o2 = HENRY_O2 * air_o2;
                let eq_co2 = HENRY_CO2 * air_co2;
                let o2_flux = FICK_SURFACE_CONDUCTANCE * porosity * eco_dt * (eq_o2 - soil_o2);
                let co2_flux = FICK_SURFACE_CONDUCTANCE * porosity * eco_dt * (eq_co2 - soil_co2);

                let o2_mag = o2_flux.abs().min(0.001);
                if o2_mag > 1.0e-7 {
                    if o2_flux > 0.0 {
                        deposits.push((
                            TerrariumSpecies::OxygenGas,
                            x,
                            y,
                            0usize,
                            patch_radius,
                            o2_mag,
                        ));
                    } else {
                        extractions.push((
                            TerrariumSpecies::OxygenGas,
                            x,
                            y,
                            0usize,
                            patch_radius,
                            o2_mag,
                        ));
                    }
                }

                let co2_mag = co2_flux.abs().min(0.001);
                if co2_mag > 1.0e-7 {
                    if co2_flux > 0.0 {
                        deposits.push((
                            TerrariumSpecies::CarbonDioxide,
                            x,
                            y,
                            0usize,
                            patch_radius,
                            co2_mag,
                        ));
                    } else {
                        extractions.push((
                            TerrariumSpecies::CarbonDioxide,
                            x,
                            y,
                            0usize,
                            patch_radius,
                            co2_mag,
                        ));
                    }
                }
            }
        }

        for (species, x, y, z, radius, amount) in deposits {
            let deposited = self
                .substrate
                .deposit_patch_species(species, x, y, z, radius, amount);
            match species {
                TerrariumSpecies::OxygenGas => {
                    self.exchange_atmosphere_odorant(ATMOS_O2_IDX, x, y, z, radius, -deposited)
                }
                TerrariumSpecies::CarbonDioxide => {
                    self.exchange_atmosphere_odorant(ATMOS_CO2_IDX, x, y, z, radius, -deposited)
                }
                _ => {}
            }
        }

        for (species, x, y, z, radius, amount) in extractions {
            let removed = self
                .substrate
                .extract_patch_species(species, x, y, z, radius, amount);
            match species {
                TerrariumSpecies::OxygenGas => {
                    self.exchange_atmosphere_odorant(ATMOS_O2_IDX, x, y, z, radius, removed)
                }
                TerrariumSpecies::CarbonDioxide => {
                    self.exchange_atmosphere_odorant(ATMOS_CO2_IDX, x, y, z, radius, removed)
                }
                _ => {}
            }
        }
    }

    pub(super) fn step_broad_soil(&mut self, eco_dt: f32) -> Result<(), String> {
        self.rebuild_water_mask();
        let has_ownership = self.ownership_diagnostics.owned_fraction > 0.0;
        let pre_moisture = if has_ownership { self.moisture.clone() } else { Vec::new() };
        let pre_deep_moisture = if has_ownership { self.deep_moisture.clone() } else { Vec::new() };
        let pre_nutrients = if has_ownership { self.dissolved_nutrients.clone() } else { Vec::new() };
        let pre_nitrogen = if has_ownership { self.mineral_nitrogen.clone() } else { Vec::new() };
        let pre_shallow = if has_ownership { self.shallow_nutrients.clone() } else { Vec::new() };
        let pre_deep_minerals = if has_ownership { self.deep_minerals.clone() } else { Vec::new() };
        let pre_organic = if has_ownership { self.organic_matter.clone() } else { Vec::new() };
        let pre_litter = if has_ownership { self.litter_carbon.clone() } else { Vec::new() };
        let pre_microbes = if has_ownership { self.microbial_biomass.clone() } else { Vec::new() };
        let pre_symbionts = if has_ownership { self.symbiont_biomass.clone() } else { Vec::new() };
        let pre_exudates = if has_ownership { self.root_exudates.clone() } else { Vec::new() };

        let result = step_soil_broad_pools(
            self.config.width,
            self.config.height,
            eco_dt,
            self.daylight(),
            temp_response(
                self.sample_temperature_at(self.config.width / 2, self.config.height / 2, 0),
                24.0,
                10.0,
            ),
            &self.water_mask,
            &self.canopy_cover,
            &self.root_density,
            &self.moisture,
            &self.deep_moisture,
            &self.dissolved_nutrients,
            &self.mineral_nitrogen,
            &self.shallow_nutrients,
            &self.deep_minerals,
            &self.organic_matter,
            &self.litter_carbon,
            &self.microbial_biomass,
            &self.symbiont_biomass,
            &self.root_exudates,
            &self.soil_structure,
        )?;
        self.moisture = result.moisture;
        self.deep_moisture = result.deep_moisture;
        self.dissolved_nutrients = result.dissolved_nutrients;
        self.mineral_nitrogen = result.mineral_nitrogen;
        self.shallow_nutrients = result.shallow_nutrients;
        self.deep_minerals = result.deep_minerals;
        self.organic_matter = result.organic_matter;
        self.litter_carbon = result.litter_carbon;
        self.microbial_biomass = result.microbial_biomass;
        self.symbiont_biomass = result.symbiont_biomass;
        self.root_exudates = result.root_exudates;

        // Authority suppression: for explicitly-owned cells, blend the broad-soil
        // result back toward the pre-step value.
        if has_ownership {
            for (i, cell) in self.ownership.iter().enumerate() {
                if cell.owner.is_background() {
                    continue;
                }
                let s = cell.strength;
                macro_rules! suppress {
                    ($field:expr, $pre:expr) => {
                        $field[i] = $field[i] * (1.0 - s) + $pre[i] * s;
                    };
                }
                suppress!(self.moisture, pre_moisture);
                suppress!(self.deep_moisture, pre_deep_moisture);
                suppress!(self.dissolved_nutrients, pre_nutrients);
                suppress!(self.mineral_nitrogen, pre_nitrogen);
                suppress!(self.shallow_nutrients, pre_shallow);
                suppress!(self.deep_minerals, pre_deep_minerals);
                suppress!(self.organic_matter, pre_organic);
                suppress!(self.litter_carbon, pre_litter);
                suppress!(self.microbial_biomass, pre_microbes);
                suppress!(self.symbiont_biomass, pre_symbionts);
                suppress!(self.root_exudates, pre_exudates);
            }
        }
        Ok(())
    }

    pub(super) fn recruit_packet_populations(&mut self) {
        if self.packet_populations.len() >= GENOTYPE_PACKET_POPULATION_MAX_CELLS {
            return;
        }
        let width = self.config.width;
        let depth = self.config.depth.max(1);

        for idx in 0..self.explicit_microbes.len() {
            let cohort = &self.explicit_microbes[idx];
            let (x, y, z) = (cohort.x, cohort.y, cohort.z);

            if self
                .packet_populations
                .iter()
                .any(|pop| pop.x == x && pop.y == y)
            {
                continue;
            }
            if self.packet_populations.len() >= GENOTYPE_PACKET_POPULATION_MAX_CELLS {
                break;
            }

            let flat = idx2(width, x, y);
            let authority = self.explicit_microbe_authority[flat];
            if authority < EXPLICIT_OWNERSHIP_THRESHOLD {
                continue;
            }

            let mut pop = GenotypePacketPopulation::new(x, y, z.min(depth - 1));
            let catalog_entries = self.microbial_secondary.catalog_entries_at(flat);
            let total_cells = self.microbial_cells[flat].max(1.0);
            pop.seed_from_secondary_bank(&catalog_entries, total_cells);

            if pop.is_alive() {
                self.packet_populations.push(pop);
            }
        }
    }

    pub(super) fn step_packet_populations(&mut self, eco_dt: f32) -> Result<(), String> {
        if self.packet_populations.is_empty() {
            return Ok(());
        }

        let depth = self.config.depth.max(1);
        let width = self.config.width;
        let height = self.config.height;

        for pop_idx in 0..self.packet_populations.len() {
            let (x, y, z) = {
                let pop = &self.packet_populations[pop_idx];
                (pop.x, pop.y, pop.z.min(depth - 1))
            };

            let local_glucose =
                self.substrate
                    .patch_mean_species(TerrariumSpecies::Glucose, x, y, z, 1);
            let local_oxygen =
                self.substrate
                    .patch_mean_species(TerrariumSpecies::OxygenGas, x, y, z, 1);
            let local_co2 =
                self.substrate
                    .patch_mean_species(TerrariumSpecies::CarbonDioxide, x, y, z, 1);
            let local_stress = clamp(
                (0.3 - local_oxygen).max(0.0) * 2.0
                    + local_co2 * 0.5
                    + (0.05 - local_glucose).max(0.0) * 4.0,
                0.0,
                1.0,
            );

            // Snapshot cumulative counters before stepping
            let pre_glucose = self.packet_populations[pop_idx].total_glucose_draw();
            let pre_co2 = self.packet_populations[pop_idx].total_co2_release();
            let pre_oxygen = self.packet_populations[pop_idx].total_oxygen_draw();
            let pre_nh4 = self.packet_populations[pop_idx].total_ammonium_draw();
            let pre_proton = self.packet_populations[pop_idx].total_proton_release();

            self.packet_populations[pop_idx].step(
                eco_dt,
                local_glucose,
                local_oxygen,
                local_stress,
            );

            // Compute deltas from cumulative tracking
            let glucose_delta =
                self.packet_populations[pop_idx].total_glucose_draw() - pre_glucose;
            let co2_delta =
                self.packet_populations[pop_idx].total_co2_release() - pre_co2;
            let oxygen_delta =
                self.packet_populations[pop_idx].total_oxygen_draw() - pre_oxygen;
            let nh4_delta =
                self.packet_populations[pop_idx].total_ammonium_draw() - pre_nh4;
            let proton_delta =
                self.packet_populations[pop_idx].total_proton_release() - pre_proton;

            // --- Glucose extraction (existing) ---
            if glucose_delta > 1.0e-8 {
                let _ = self.substrate.extract_patch_species(
                    TerrariumSpecies::Glucose,
                    x,
                    y,
                    z,
                    1,
                    glucose_delta.min(0.003),
                );
            }
            // --- CO2 deposition (existing) ---
            if co2_delta > 1.0e-8 {
                let _ = self.substrate.deposit_patch_species(
                    TerrariumSpecies::CarbonDioxide,
                    x,
                    y,
                    z,
                    1,
                    co2_delta.min(0.003),
                );
            }
            // --- Oxygen consumption (was tracked but never wired) ---
            if oxygen_delta > 1.0e-8 {
                let _ = self.substrate.extract_patch_species(
                    TerrariumSpecies::OxygenGas,
                    x,
                    y,
                    z,
                    1,
                    oxygen_delta.min(0.003),
                );
            }
            // --- Ammonium draw for nitrogen metabolism (Redfield-like) ---
            if nh4_delta > 1.0e-9 {
                let _ = self.substrate.extract_patch_species(
                    TerrariumSpecies::Ammonium,
                    x,
                    y,
                    z,
                    1,
                    nh4_delta.min(0.001),
                );
            }
            // --- Proton release from metabolic acidification ---
            if proton_delta > 1.0e-9 {
                let _ = self.substrate.deposit_patch_species(
                    TerrariumSpecies::Proton,
                    x,
                    y,
                    z,
                    1,
                    proton_delta.min(0.001),
                );
            }
        }

        // --- Budding: high-activity populations spread to neighbors ---
        let mut buds: Vec<(usize, usize, usize, GenotypePacket)> = Vec::new();
        for pop in &mut self.packet_populations {
            if pop.total_cells > 10.0 && pop.age_s > 5.0 {
                if let Some((nx, ny, daughter)) = pop.try_bud(width, height) {
                    buds.push((nx, ny, pop.z, daughter));
                }
            }
        }
        // Insert buds into existing or new populations at target coordinates
        for (nx, ny, nz, packet) in buds {
            if let Some(target) = self
                .packet_populations
                .iter_mut()
                .find(|p| p.x == nx && p.y == ny)
            {
                if target.packets.len() < GENOTYPE_PACKET_MAX_PER_CELL {
                    target.packets.push(packet);
                    target.recompute_total();
                }
            } else if self.packet_populations.len() < GENOTYPE_PACKET_POPULATION_MAX_CELLS {
                let mut new_pop = GenotypePacketPopulation::new(nx, ny, nz);
                new_pop.packets.push(packet);
                new_pop.recompute_total();
                self.packet_populations.push(new_pop);
            }
        }

        self.packet_populations.retain(|pop| pop.is_alive());
        Ok(())
    }

    pub(super) fn reconcile_owned_summary_pools_from_substrate(&mut self) {
        project_owned_summary_pools(
            OwnedSummaryProjectionConfig {
                width: self.config.width,
                height: self.config.height,
                depth: self.config.depth,
                ownership_threshold: EXPLICIT_OWNERSHIP_THRESHOLD,
            },
            OwnedSummaryProjectionInputs {
                explicit_microbe_authority: &self.explicit_microbe_authority,
                ammonium: self.substrate.species_field(TerrariumSpecies::Ammonium),
                nitrate: self.substrate.species_field(TerrariumSpecies::Nitrate),
                phosphorus: self.substrate.species_field(TerrariumSpecies::Phosphorus),
                glucose: self.substrate.species_field(TerrariumSpecies::Glucose),
                carbon_dioxide: self
                    .substrate
                    .species_field(TerrariumSpecies::CarbonDioxide),
                atp_flux: self.substrate.species_field(TerrariumSpecies::AtpFlux),
            },
            OwnedSummaryProjectionOutputs {
                root_exudates: &mut self.root_exudates,
                litter_carbon: &mut self.litter_carbon,
                dissolved_nutrients: &mut self.dissolved_nutrients,
                shallow_nutrients: &mut self.shallow_nutrients,
                mineral_nitrogen: &mut self.mineral_nitrogen,
                organic_matter: &mut self.organic_matter,
            },
        )
        .expect("terrarium owned summary pools should match substrate dimensions");
    }

    /// Promote high-fitness coarse packets to explicit WholeCellSimulator cohorts.
    pub(super) fn promote_qualified_packets(&mut self) -> Result<(), String> {
        // Requires add_explicit_microbe from explicit_microbe_impl.rs (still gated).
        Ok(())
    }
}
