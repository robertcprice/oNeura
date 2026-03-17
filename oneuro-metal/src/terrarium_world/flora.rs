use super::*;

impl TerrariumWorld {
    pub(super) fn step_plants(&mut self, eco_dt: f32) -> Result<(), String> {
        if self.plants.is_empty() {
            return Ok(());
        }
        let depth = self.config.depth.max(1);
        let ammonium_surface = layer_mean_map(
            self.config.width,
            self.config.height,
            depth,
            self.substrate.species_field(TerrariumSpecies::Ammonium),
            0,
            depth.min(2),
        );
        let nitrate_surface = layer_mean_map(
            self.config.width,
            self.config.height,
            depth,
            self.substrate.species_field(TerrariumSpecies::Nitrate),
            0,
            depth.min(2),
        );
        let nitrate_deep = layer_mean_map(
            self.config.width,
            self.config.height,
            depth,
            self.substrate.species_field(TerrariumSpecies::Nitrate),
            depth / 2,
            depth,
        );
        let daylight = self.daylight();
        let mut queued_fruits = Vec::new();
        let mut queued_seeds = Vec::new();
        let mut queued_co2_fluxes: Vec<(usize, usize, usize, usize, f32)> = Vec::new();
        let mut queued_o2_fluxes: Vec<(usize, usize, usize, usize, f32)> = Vec::new();
        let mut queued_humidity_fluxes: Vec<(usize, usize, usize, usize, f32)> = Vec::new();
        let mut dead_plants = Vec::new();

        // Build per-plant asymmetric competition descriptors.
        let canopy_descriptors: Vec<CanopyDescriptor> = self
            .plants
            .iter()
            .map(|p| {
                let r = p.genome.canopy_radius_mm.max(1.0);
                let leaf = p.physiology.leaf_biomass().max(0.0);
                let lai = leaf * 200.0 / (std::f32::consts::PI * r * r * 0.01);
                CanopyDescriptor {
                    x: p.x as f32,
                    y: p.y as f32,
                    height_mm: p.physiology.height_mm(),
                    canopy_radius_mm: r,
                    lai,
                    extinction_coeff: (0.45 + p.genome.leaf_efficiency * 0.25),
                }
            })
            .collect();
        let root_descriptors: Vec<RootDescriptor> = self
            .plants
            .iter()
            .map(|p| RootDescriptor {
                x: p.x as f32,
                y: p.y as f32,
                root_depth_mm: p.genome.root_depth_bias * p.physiology.height_mm() * 0.4,
                root_radius_mm: p.genome.root_radius_mm,
                root_biomass: p.physiology.root_biomass().max(0.0),
            })
            .collect();
        let light_factors =
            compute_light_competition(&canopy_descriptors, self.config.cell_size_mm);
        let root_factors = compute_root_competition(
            &root_descriptors,
            &self.mineral_nitrogen,
            &self.shallow_nutrients,
            self.config.width,
            self.config.height,
            self.config.cell_size_mm,
        );

        for idx in 0..self.plants.len() {
            let fruit_reset_s = self.rng.gen_range(7200.0..17000.0);
            let seed_reset_s = self.rng.gen_range(12000.0..30000.0);
            let (
                x,
                y,
                genome,
                _canopy_self,
                _root_self,
                canopy_radius,
                root_radius,
                canopy_z,
                storage_signal,
                health_before,
            ) = {
                let plant = &self.plants[idx];
                (
                    plant.x,
                    plant.y,
                    plant.genome.clone(),
                    plant.canopy_amplitude(),
                    plant.root_amplitude(),
                    plant.canopy_radius_cells(),
                    plant.root_radius_cells(),
                    ((plant.physiology.height_mm() / 3.0).round() as usize).clamp(1, depth - 1),
                    plant.physiology.storage_carbon().max(0.0),
                    plant.physiology.health(),
                )
            };
            let flat = idx2(self.config.width, x, y);
            // Per-plant asymmetric competition from plant_competition module.
            let light_factor = light_factors.get(idx).copied().unwrap_or(1.0);
            let (root_n_factor, _root_p_factor) =
                root_factors.get(idx).copied().unwrap_or((1.0, 1.0));
            let canopy_comp = (1.0 - light_factor).max(0.0);
            let root_comp = (1.0 - root_n_factor).max(0.0);
            let symbionts = self.symbiont_biomass[flat];
            let deep_moisture = self.deep_moisture[flat];
            let litter = self.litter_carbon[flat];

            let oxygen = self.substrate.patch_mean_species(
                TerrariumSpecies::OxygenGas,
                x,
                y,
                1.min(depth - 1),
                root_radius,
            );
            let proton = self.substrate.patch_mean_species(
                TerrariumSpecies::Proton,
                x,
                y,
                1.min(depth - 1),
                root_radius,
            );
            let soil_nitrate_patch = self.substrate.patch_mean_species(
                TerrariumSpecies::Nitrate,
                x,
                y,
                1.min(depth - 1),
                root_radius,
            );
            let carbon_dioxide = self.substrate.patch_mean_species(
                TerrariumSpecies::CarbonDioxide,
                x,
                y,
                1.min(depth - 1),
                root_radius,
            );
            let soil_redox = clamp(
                (oxygen * 1.3 + soil_nitrate_patch * 0.7)
                    / (oxygen * 1.3
                        + soil_nitrate_patch * 0.7
                        + carbon_dioxide * 0.35
                        + proton * 0.45
                        + 1e-9),
                0.0,
                1.0,
            );
            let soil_glucose = self.substrate.patch_mean_species(
                TerrariumSpecies::Glucose,
                x,
                y,
                1.min(depth - 1),
                root_radius,
            );
            let soil_ammonium = self.substrate.patch_mean_species(
                TerrariumSpecies::Ammonium,
                x,
                y,
                1.min(depth - 1),
                root_radius,
            );
            let atp_flux = self.substrate.patch_mean_species(
                TerrariumSpecies::AtpFlux,
                x,
                y,
                1.min(depth - 1),
                root_radius,
            );
            let root_energy_gate = clamp(0.32 + atp_flux * 1800.0, 0.2, 1.2);
            let water_deficit = clamp(
                0.42 - (self.moisture[flat] + deep_moisture * 0.35),
                0.0,
                1.0,
            );
            let nitrogen_deficit = clamp(
                0.14 - (self.mineral_nitrogen[flat] + soil_ammonium + soil_nitrate_patch) * 0.32,
                0.0,
                1.0,
            );

            let (water_demand, nutrient_demand) = self.plants[idx].physiology.resource_demands(
                eco_dt,
                root_energy_gate,
                water_deficit,
                nitrogen_deficit,
            );
            let extraction = extract_root_resources_with_layers(
                self.config.width,
                self.config.height,
                x as i32,
                y as i32,
                root_radius as i32,
                water_demand,
                nutrient_demand,
                genome.root_depth_bias,
                genome.symbiosis_affinity,
                root_energy_gate,
                &self.moisture,
                &self.deep_moisture,
                &self.dissolved_nutrients,
                &self.mineral_nitrogen,
                &self.shallow_nutrients,
                &self.deep_minerals,
                &self.symbiont_biomass,
                &ammonium_surface,
                &nitrate_surface,
                &nitrate_deep,
            )?;
            self.moisture = extraction.moisture;
            self.deep_moisture = extraction.deep_moisture;
            self.dissolved_nutrients = extraction.dissolved_nutrients;
            self.mineral_nitrogen = extraction.mineral_nitrogen;
            self.shallow_nutrients = extraction.shallow_nutrients;
            self.deep_minerals = extraction.deep_minerals;

            let total_water_take = extraction.surface_water_take + extraction.deep_water_take;
            if total_water_take > 0.0 {
                let _ = self.substrate.extract_patch_species(
                    TerrariumSpecies::Water,
                    x,
                    y,
                    0,
                    root_radius,
                    total_water_take,
                );
            }
            if extraction.ammonium_take > 0.0 {
                let _ = self.substrate.extract_patch_species(
                    TerrariumSpecies::Ammonium,
                    x,
                    y,
                    0,
                    root_radius,
                    extraction.ammonium_take,
                );
            }
            let nitrate_take = extraction.rhizo_nitrate_take + extraction.deep_nitrate_take;
            if nitrate_take > 0.0 {
                let _ = self.substrate.extract_patch_species(
                    TerrariumSpecies::Nitrate,
                    x,
                    y,
                    0,
                    root_radius,
                    nitrate_take,
                );
            }

            let water_factor = if water_demand > 1.0e-6 {
                clamp(extraction.water_take / water_demand, 0.0, 1.1)
            } else {
                1.0
            };
            let nutrient_factor = if nutrient_demand > 1.0e-6 {
                clamp(extraction.nutrient_take / nutrient_demand, 0.0, 1.1)
            } else {
                1.0
            };
            let local_temp = self.sample_temperature_at(x, y, 1.min(depth - 1));
            let temp_factor = temp_response(local_temp, 24.0, 10.0);
            let local_humidity = self.sample_humidity_at(x, y, canopy_z);
            let local_light = clamp(
                daylight * (1.0 - canopy_comp * (1.18 - genome.shade_tolerance).max(0.16)),
                0.03,
                1.15,
            );
            let local_air_co2 = self.sample_odorant_patch(
                ATMOS_CO2_IDX,
                x,
                y,
                canopy_z,
                (canopy_radius.max(2) / 2).max(1),
            );
            let air_co2_factor = clamp(local_air_co2 / ATMOS_CO2_BASELINE, 0.35, 1.8);
            let stomatal_open = clamp(
                0.24 + local_light * 0.46 + water_factor * 0.28 + local_humidity * 0.22
                    - water_deficit * 0.32
                    - canopy_comp * 0.02,
                0.12,
                1.45,
            );
            let symbiosis_signal = clamp(symbionts * 6.5 * genome.symbiosis_affinity, 0.0, 1.5);
            let symbiosis_bonus = 1.0 + symbionts * 7.5 * genome.symbiosis_affinity;
            let stress_signal = clamp(
                water_deficit * 0.75
                    + nitrogen_deficit * 0.58
                    + canopy_comp * 0.018
                    + root_comp * 0.014
                    + (1.0 - soil_redox) * 0.45,
                0.0,
                1.4,
            );

            let (
                report,
                new_total_cells,
                cell_vitality,
                volatile_scale,
                seed_mass,
                updated_health,
                updated_storage,
                root_radius_after,
            ) = {
                let plant = &mut self.plants[idx];
                let cell_feedback = plant.cellular.step(
                    eco_dt,
                    local_light,
                    temp_factor,
                    extraction.water_take,
                    extraction.nutrient_take,
                    water_factor,
                    nutrient_factor,
                    symbiosis_signal,
                    stress_signal,
                    storage_signal,
                );
                let report = plant.physiology.step(
                    eco_dt,
                    extraction.water_take,
                    extraction.nutrient_take,
                    local_light,
                    temp_factor,
                    root_energy_gate,
                    symbiosis_bonus,
                    water_factor,
                    nutrient_factor,
                    canopy_comp,
                    root_comp,
                    soil_glucose,
                    air_co2_factor,
                    stomatal_open,
                    cell_feedback.photosynthetic_capacity,
                    cell_feedback.maintenance_cost,
                    cell_feedback.storage_exchange,
                    cell_feedback.division_growth,
                    cell_feedback.senescence_mass,
                    cell_feedback.energy_charge,
                    cell_feedback.vitality,
                    cell_feedback.sugar_pool,
                    cell_feedback.division_signal,
                    plant.cellular.total_cells(),
                    fruit_reset_s,
                    seed_reset_s,
                );
                (
                    report,
                    plant.cellular.total_cells(),
                    plant.cellular.vitality(),
                    plant.genome.volatile_scale,
                    plant.genome.seed_mass,
                    plant.physiology.health(),
                    plant.physiology.storage_carbon(),
                    plant.root_radius_cells(),
                )
            };

            deposit_2d(
                &mut self.root_exudates,
                self.config.width,
                self.config.height,
                x,
                y,
                root_radius_after.max(1) / 2,
                report.exudates,
            );
            deposit_2d(
                &mut self.litter_carbon,
                self.config.width,
                self.config.height,
                x,
                y,
                root_radius_after.max(1) / 2,
                report.litter,
            );
            let hotspot_radius = root_radius_after.max(1);
            self.substrate
                .add_hotspot(TerrariumSpecies::Glucose, x, y, 0, report.exudates * 12.0);
            self.substrate
                .add_hotspot(TerrariumSpecies::Ammonium, x, y, 0, report.litter * 8.0);
            self.substrate.add_hotspot(
                TerrariumSpecies::CarbonDioxide,
                x,
                y,
                1.min(depth - 1),
                report.litter * 4.0,
            );
            deposit_2d(
                &mut self.organic_matter,
                self.config.width,
                self.config.height,
                x,
                y,
                hotspot_radius / 2,
                report.litter * 0.18,
            );
            queued_co2_fluxes.push((
                x,
                y,
                canopy_z,
                (canopy_radius.max(2) / 2).max(1),
                report.co2_flux,
            ));
            queued_o2_fluxes.push((
                x,
                y,
                canopy_z,
                (canopy_radius.max(2) / 2).max(1),
                (-report.co2_flux * 1.05),
            ));
            queued_humidity_fluxes.push((
                x,
                y,
                canopy_z,
                (canopy_radius.max(2) / 2).max(1),
                report.water_vapor_flux,
            ));

            if report.spawned_fruit
                && self.fruits.len() + queued_fruits.len() < self.config.max_fruits
            {
                queued_fruits.push((x, y, report.fruit_size, volatile_scale));
            }
            if report.spawned_seed && self.seeds.len() + queued_seeds.len() < self.config.max_seeds
            {
                let dispersal = (7.0 - seed_mass * 18.0 + updated_health * 1.5)
                    .round()
                    .max(2.0) as isize;
                let dx = self.rng.gen_range(-dispersal..=dispersal);
                let dy = self.rng.gen_range(-dispersal..=dispersal);
                let sx = offset_clamped(x, dx, self.config.width);
                let sy = offset_clamped(y, dy, self.config.height);
                let dormancy =
                    self.rng.gen_range(9000.0..26000.0) * (1.18 - seed_mass * 1.5).max(0.45);
                let reserve = clamp(
                    seed_mass * self.rng.gen_range(0.85..1.25) + updated_storage.max(0.0) * 0.05,
                    0.03,
                    0.28,
                );
                let child_genome = genome.mutate(&mut self.rng);
                queued_seeds.push(TerrariumSeed {
                    x: sx as f32,
                    y: sy as f32,
                    dormancy_s: dormancy,
                    reserve_carbon: reserve,
                    age_s: 0.0,
                    genome: child_genome,
                });
            }
            if self.plants[idx].physiology.is_dead()
                || (new_total_cells < 24.0 && cell_vitality < 0.08)
            {
                dead_plants.push(idx);
            }

            let _ = (health_before, litter);
        }

        for (x, y, z, radius, flux) in queued_co2_fluxes {
            self.exchange_atmosphere_odorant(ATMOS_CO2_IDX, x, y, z, radius, flux);
        }
        for (x, y, z, radius, flux) in queued_o2_fluxes {
            self.exchange_atmosphere_odorant(ATMOS_O2_IDX, x, y, z, radius, flux);
        }
        for (x, y, z, radius, flux) in queued_humidity_fluxes {
            self.exchange_atmosphere_humidity(x, y, z, radius, flux);
        }

        dead_plants.sort_unstable();
        dead_plants.dedup();
        for idx in dead_plants.into_iter().rev() {
            let plant = self.plants.swap_remove(idx);
            let litter_return = 0.02 + plant.physiology.storage_carbon().max(0.0) * 0.20;
            deposit_2d(
                &mut self.litter_carbon,
                self.config.width,
                self.config.height,
                plant.x,
                plant.y,
                plant.root_radius_cells().max(1) / 2,
                litter_return,
            );
            deposit_2d(
                &mut self.organic_matter,
                self.config.width,
                self.config.height,
                plant.x,
                plant.y,
                plant.root_radius_cells().max(1) / 3,
                litter_return * 0.22,
            );
            self.substrate.add_hotspot(
                TerrariumSpecies::Glucose,
                plant.x,
                plant.y,
                0,
                litter_return * 10.0,
            );
            self.substrate.add_hotspot(
                TerrariumSpecies::Ammonium,
                plant.x,
                plant.y,
                0,
                litter_return * 6.0,
            );
            self.substrate.add_hotspot(
                TerrariumSpecies::CarbonDioxide,
                plant.x,
                plant.y,
                1.min(self.config.depth.max(1) - 1),
                litter_return * 3.5,
            );
        }
        for (x, y, size, volatile_scale) in queued_fruits {
            let offset = self.rng.gen_range(0..4);
            let (dx, dy) = match offset {
                0 => (2, 1),
                1 => (-2, 1),
                2 => (1, -2),
                _ => (-1, -2),
            };
            let fx = offset_clamped(x, dx, self.config.width);
            let fy = offset_clamped(y, dy, self.config.height);
            self.add_fruit(fx, fy, size, Some(volatile_scale));
        }
        self.seeds.extend(queued_seeds);
        Ok(())
    }

    pub(super) fn step_food_patches_native(&mut self, eco_dt: f32) -> Result<(), String> {
        if self.fruits.is_empty() {
            return Ok(());
        }
        let patch_remaining = self
            .fruits
            .iter()
            .map(|fruit| fruit.source.sugar_content.max(0.0))
            .collect::<Vec<_>>();
        let previous_remaining = self
            .fruits
            .iter()
            .map(|fruit| fruit.previous_remaining.max(0.0))
            .collect::<Vec<_>>();
        let deposited_all = self
            .fruits
            .iter()
            .map(|fruit| fruit.deposited_all)
            .collect::<Vec<_>>();
        let has_fruit = self
            .fruits
            .iter()
            .map(|fruit| fruit.source.alive)
            .collect::<Vec<_>>();
        let fruit_ripeness = self
            .fruits
            .iter()
            .map(|fruit| fruit.source.ripeness)
            .collect::<Vec<_>>();
        let fruit_sugar = self
            .fruits
            .iter()
            .map(|fruit| fruit.source.sugar_content)
            .collect::<Vec<_>>();
        let microbial = self
            .fruits
            .iter()
            .map(|fruit| {
                let flat = idx2(
                    self.config.width,
                    fruit.source.x.min(self.config.width - 1),
                    fruit.source.y.min(self.config.height - 1),
                );
                self.microbial_biomass[flat]
                    + self.microbial_biomass[flat] * 0.08
                    + self.microbial_biomass[flat] * 0.08
            })
            .collect::<Vec<_>>();
        let stepped = step_food_patches(
            eco_dt,
            &patch_remaining,
            &previous_remaining,
            &deposited_all,
            &has_fruit,
            &fruit_ripeness,
            &fruit_sugar,
            &microbial,
        )?;
        let mut decay_fluxes = Vec::new();
        for (idx, fruit) in self.fruits.iter_mut().enumerate() {
            let current = stepped.remaining[idx].max(0.0);
            fruit.source.sugar_content = current.min(stepped.sugar_content[idx]);
            fruit.source.alive = stepped.fruit_alive[idx];
            fruit.deposited_all = stepped.deposited_all[idx];
            let x = fruit.source.x;
            let y = fruit.source.y;
            let radius = fruit.radius.round().max(1.0) as usize;
            let detritus_total = stepped.decay_detritus[idx]
                + stepped.lost_detritus[idx]
                + stepped.final_detritus[idx];
            if stepped.decay_detritus[idx] > 0.0 {
                deposit_2d(
                    &mut self.litter_carbon,
                    self.config.width,
                    self.config.height,
                    x,
                    y,
                    radius,
                    stepped.decay_detritus[idx],
            );
            }
            if stepped.lost_detritus[idx] > 0.0 {
                deposit_2d(
                    &mut self.litter_carbon,
                    self.config.width,
                    self.config.height,
                    x,
                    y,
                    radius,
                    stepped.lost_detritus[idx],
            );
            }
            if stepped.final_detritus[idx] > 0.0 {
                deposit_2d(
                    &mut self.litter_carbon,
                    self.config.width,
                    self.config.height,
                    x,
                    y,
                    radius,
                    stepped.final_detritus[idx],
            );
            }
            if detritus_total > 0.0 {
                self.substrate.add_hotspot(
                    TerrariumSpecies::Glucose,
                    x,
                    y,
                    0,
                    detritus_total * 14.0,
                );
                self.substrate.add_hotspot(
                    TerrariumSpecies::Ammonium,
                    x,
                    y,
                    0,
                    detritus_total * 2.8,
                );
            }
            let decay_flux_driver = stepped.decay_detritus[idx]
                + stepped.lost_detritus[idx] * 0.32
                + stepped.final_detritus[idx] * 0.18;
            if decay_flux_driver > 0.0 {
                let microbial_drive = 0.65 + microbial[idx].max(0.0) * 6.0;
                let decay_co2_flux = clamp(decay_flux_driver * microbial_drive * 0.020, 0.0, 0.004);
                let decay_o2_flux = -decay_co2_flux * 0.88;
                let decay_humidity_flux = decay_co2_flux * 0.24;
                decay_fluxes.push((
                    x,
                    y,
                    fruit.source.z.min(self.config.depth.max(1) - 1),
                    radius.max(1),
                    decay_co2_flux,
                    decay_o2_flux,
                    decay_humidity_flux,
                ));
            }
            fruit.previous_remaining = current;
        }
        for (x, y, z, radius, co2_flux, o2_flux, humidity_flux) in decay_fluxes {
            self.exchange_atmosphere_odorant(ATMOS_CO2_IDX, x, y, z, radius, co2_flux);
            self.exchange_atmosphere_odorant(ATMOS_O2_IDX, x, y, z, radius, o2_flux);
            self.exchange_atmosphere_humidity(x, y, z, radius, humidity_flux);
        }
        Ok(())
    }

    pub(super) fn step_seeds_native(&mut self, eco_dt: f32) -> Result<(), String> {
        if self.seeds.is_empty() {
            return Ok(());
        }
        let mut dormancy = Vec::with_capacity(self.seeds.len());
        let mut age = Vec::with_capacity(self.seeds.len());
        let mut reserve = Vec::with_capacity(self.seeds.len());
        let mut affinity = Vec::with_capacity(self.seeds.len());
        let mut shade = Vec::with_capacity(self.seeds.len());
        let mut moisture = Vec::with_capacity(self.seeds.len());
        let mut deep_moisture = Vec::with_capacity(self.seeds.len());
        let mut nutrients = Vec::with_capacity(self.seeds.len());
        let mut symbionts = Vec::with_capacity(self.seeds.len());
        let mut canopy = Vec::with_capacity(self.seeds.len());
        let mut litter = Vec::with_capacity(self.seeds.len());
        let mut positions = Vec::with_capacity(self.seeds.len());

        for seed in &self.seeds {
            let x = seed.x.round().clamp(0.0, (self.config.width - 1) as f32) as usize;
            let y = seed.y.round().clamp(0.0, (self.config.height - 1) as f32) as usize;
            let flat = idx2(self.config.width, x, y);
            positions.push((x, y));
            dormancy.push(seed.dormancy_s);
            age.push(seed.age_s);
            reserve.push(seed.reserve_carbon);
            affinity.push(seed.genome.symbiosis_affinity);
            shade.push(seed.genome.shade_tolerance);
            moisture.push(self.moisture[flat]);
            deep_moisture.push(self.deep_moisture[flat]);
            nutrients.push(self.shallow_nutrients[flat]);
            symbionts.push(self.symbiont_biomass[flat]);
            canopy.push(self.canopy_cover[flat]);
            litter.push(self.litter_carbon[flat]);
        }

        let stepped = step_seed_bank(
            eco_dt,
            self.daylight(),
            self.plants.len(),
            self.config.max_plants,
            &dormancy,
            &age,
            &reserve,
            &affinity,
            &shade,
            &moisture,
            &deep_moisture,
            &nutrients,
            &symbionts,
            &canopy,
            &litter,
        )?;

        let mut next_bank = Vec::new();
        let mut germinations = Vec::new();
        for (idx, mut seed) in self.seeds.drain(..).enumerate() {
            seed.age_s = stepped.age_s[idx];
            seed.dormancy_s = stepped.dormancy_s[idx];
            if stepped.germinate[idx]
                && self.plants.len() + germinations.len() < self.config.max_plants
            {
                let (x, y) = positions[idx];
                let scale = stepped.seedling_scale[idx].max(0.45);
                germinations.push((x, y, seed.genome, scale));
            } else if stepped.keep[idx] {
                next_bank.push(seed);
            }
        }
        self.seeds = next_bank;
        for (x, y, genome, scale) in germinations {
            let _ = self.add_plant(x, y, Some(genome), Some(scale));
        }
        Ok(())
    }
}
