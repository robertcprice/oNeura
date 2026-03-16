// Visual biomechanics stepping for TerrariumWorld.
use super::*;

impl TerrariumWorld {
    pub(super) fn step_latent_strain_banks(&mut self, eco_dt: f32) -> Result<(), String> {
        let nitrifier_env_bias = self
            .deep_moisture
            .iter()
            .map(|value| (1.0 - value * 0.5).clamp(0.0, 1.2))
            .collect::<Vec<_>>();
        let microbial_secondary_packets = self.microbial_secondary.packets_banks();
        let microbial_secondary_trait_a = self.microbial_secondary.trait_a_banks();
        let microbial_secondary_trait_b = self.microbial_secondary.trait_b_banks();
        let microbial = step_latent_guild_banks(
            &LatentGuildState {
                total_packets: &self.microbial_packets,
                public_secondary_packets: microbial_secondary_packets,
                public_secondary_trait_a: microbial_secondary_trait_a,
                public_secondary_trait_b: microbial_secondary_trait_b,
                mutation_flux: &self.microbial_packet_mutation_flux,
                vitality: &self.microbial_vitality,
                dormancy: &self.microbial_dormancy,
                primary_trait_a: &self.microbial_strain_yield,
                primary_trait_b: &self.microbial_strain_stress_tolerance,
                weight_env_bias: &self.root_density,
                spread_env_bias: &self.moisture,
                latent_packets: [
                    &self.microbial_latent_packets[0],
                    &self.microbial_latent_packets[1],
                ],
                latent_trait_a: [
                    &self.microbial_latent_strain_yield[0],
                    &self.microbial_latent_strain_yield[1],
                ],
                latent_trait_b: [
                    &self.microbial_latent_strain_stress_tolerance[0],
                    &self.microbial_latent_strain_stress_tolerance[1],
                ],
            },
            LatentGuildConfig {
                latent_pool_base: 0.05,
                latent_pool_mutation_scale: 1.6,
                latent_pool_inactive_scale: 0.12,
                latent_pool_min: 0.02,
                latent_pool_max: 0.42,
                weight_base: 0.8,
                weight_step: 0.18,
                weight_mutation_scale: 1.4,
                weight_mutation_bank_step: 0.35,
                weight_env_scale: 0.22,
                packet_relax_rate: 0.00075,
                packet_mutation_scale: 2.0,
                packet_activity_scale: 0.35,
                spread_base: 0.06,
                spread_step: 0.0,
                spread_mutation_scale: 0.9,
                spread_mutation_bank_step: 0.25,
                spread_env_scale: 0.08,
                spread_env_center: 0.5,
                spread_min: 0.03,
                spread_max: 0.25,
                trait_relax_rate: 0.00055,
                trait_mutation_scale: 0.010,
                trait_a_polarities: [-1.0, 1.0],
                trait_b_polarities: [0.85, -0.85],
                trait_b_spread_scale: 1.0,
                trait_b_inactive_scale: 0.06,
            },
            eco_dt,
        )?;
        self.microbial_strain_yield = microbial.primary_trait_a;
        self.microbial_strain_stress_tolerance = microbial.primary_trait_b;
        self.microbial_latent_packets = microbial.latent_packets.into_iter().collect();
        self.microbial_latent_strain_yield = microbial.latent_trait_a.into_iter().collect();
        self.microbial_latent_strain_stress_tolerance =
            microbial.latent_trait_b.into_iter().collect();

        let nitrifier_secondary_packets = self.nitrifier_secondary.packets_banks();
        let nitrifier_secondary_trait_a = self.nitrifier_secondary.trait_a_banks();
        let nitrifier_secondary_trait_b = self.nitrifier_secondary.trait_b_banks();
        let nitrifier = step_latent_guild_banks(
            &LatentGuildState {
                total_packets: &self.nitrifier_packets,
                public_secondary_packets: nitrifier_secondary_packets,
                public_secondary_trait_a: nitrifier_secondary_trait_a,
                public_secondary_trait_b: nitrifier_secondary_trait_b,
                mutation_flux: &self.nitrifier_packet_mutation_flux,
                vitality: &self.nitrifier_vitality,
                dormancy: &self.nitrifier_dormancy,
                primary_trait_a: &self.nitrifier_strain_oxygen_affinity,
                primary_trait_b: &self.nitrifier_strain_ammonium_affinity,
                weight_env_bias: &nitrifier_env_bias,
                spread_env_bias: &nitrifier_env_bias,
                latent_packets: [
                    &self.nitrifier_latent_packets[0],
                    &self.nitrifier_latent_packets[1],
                ],
                latent_trait_a: [
                    &self.nitrifier_latent_strain_oxygen_affinity[0],
                    &self.nitrifier_latent_strain_oxygen_affinity[1],
                ],
                latent_trait_b: [
                    &self.nitrifier_latent_strain_ammonium_affinity[0],
                    &self.nitrifier_latent_strain_ammonium_affinity[1],
                ],
            },
            LatentGuildConfig {
                latent_pool_base: 0.04,
                latent_pool_mutation_scale: 1.4,
                latent_pool_inactive_scale: 0.10,
                latent_pool_min: 0.015,
                latent_pool_max: 0.34,
                weight_base: 0.7,
                weight_step: 0.0,
                weight_mutation_scale: 1.2,
                weight_mutation_bank_step: 0.30,
                weight_env_scale: 0.25,
                packet_relax_rate: 0.00070,
                packet_mutation_scale: 1.9,
                packet_activity_scale: 0.32,
                spread_base: 0.05,
                spread_step: 0.0,
                spread_mutation_scale: 0.8,
                spread_mutation_bank_step: 0.24,
                spread_env_scale: 0.06,
                spread_env_center: 0.0,
                spread_min: 0.03,
                spread_max: 0.22,
                trait_relax_rate: 0.00052,
                trait_mutation_scale: 0.009,
                trait_a_polarities: [-1.0, 1.0],
                trait_b_polarities: [0.70, -0.70],
                trait_b_spread_scale: 1.0,
                trait_b_inactive_scale: 0.0,
            },
            eco_dt,
        )?;
        self.nitrifier_strain_oxygen_affinity = nitrifier.primary_trait_a;
        self.nitrifier_strain_ammonium_affinity = nitrifier.primary_trait_b;
        self.nitrifier_latent_packets = nitrifier.latent_packets.into_iter().collect();
        self.nitrifier_latent_strain_oxygen_affinity =
            nitrifier.latent_trait_a.into_iter().collect();
        self.nitrifier_latent_strain_ammonium_affinity =
            nitrifier.latent_trait_b.into_iter().collect();

        let denitrifier_secondary_packets = self.denitrifier_secondary.packets_banks();
        let denitrifier_secondary_trait_a = self.denitrifier_secondary.trait_a_banks();
        let denitrifier_secondary_trait_b = self.denitrifier_secondary.trait_b_banks();
        let denitrifier = step_latent_guild_banks(
            &LatentGuildState {
                total_packets: &self.denitrifier_packets,
                public_secondary_packets: denitrifier_secondary_packets,
                public_secondary_trait_a: denitrifier_secondary_trait_a,
                public_secondary_trait_b: denitrifier_secondary_trait_b,
                mutation_flux: &self.denitrifier_packet_mutation_flux,
                vitality: &self.denitrifier_vitality,
                dormancy: &self.denitrifier_dormancy,
                primary_trait_a: &self.denitrifier_strain_anoxia_affinity,
                primary_trait_b: &self.denitrifier_strain_nitrate_affinity,
                weight_env_bias: &self.deep_moisture,
                spread_env_bias: &self.deep_moisture,
                latent_packets: [
                    &self.denitrifier_latent_packets[0],
                    &self.denitrifier_latent_packets[1],
                ],
                latent_trait_a: [
                    &self.denitrifier_latent_strain_anoxia_affinity[0],
                    &self.denitrifier_latent_strain_anoxia_affinity[1],
                ],
                latent_trait_b: [
                    &self.denitrifier_latent_strain_nitrate_affinity[0],
                    &self.denitrifier_latent_strain_nitrate_affinity[1],
                ],
            },
            LatentGuildConfig {
                latent_pool_base: 0.05,
                latent_pool_mutation_scale: 1.5,
                latent_pool_inactive_scale: 0.12,
                latent_pool_min: 0.02,
                latent_pool_max: 0.38,
                weight_base: 0.75,
                weight_step: 0.0,
                weight_mutation_scale: 1.3,
                weight_mutation_bank_step: 0.35,
                weight_env_scale: 0.28,
                packet_relax_rate: 0.00072,
                packet_mutation_scale: 1.9,
                packet_activity_scale: 0.30,
                spread_base: 0.05,
                spread_step: 0.0,
                spread_mutation_scale: 0.82,
                spread_mutation_bank_step: 0.26,
                spread_env_scale: 0.07,
                spread_env_center: 0.0,
                spread_min: 0.03,
                spread_max: 0.24,
                trait_relax_rate: 0.00054,
                trait_mutation_scale: 0.009,
                trait_a_polarities: [1.0, -1.0],
                trait_b_polarities: [-0.72, 0.72],
                trait_b_spread_scale: 1.0,
                trait_b_inactive_scale: 0.0,
            },
            eco_dt,
        )?;
        self.denitrifier_strain_anoxia_affinity = denitrifier.primary_trait_a;
        self.denitrifier_strain_nitrate_affinity = denitrifier.primary_trait_b;
        self.denitrifier_latent_packets = denitrifier.latent_packets.into_iter().collect();
        self.denitrifier_latent_strain_anoxia_affinity =
            denitrifier.latent_trait_a.into_iter().collect();
        self.denitrifier_latent_strain_nitrate_affinity =
            denitrifier.latent_trait_b.into_iter().collect();
        Ok(())
    }

    pub(super) fn step_visual_biomechanics(&mut self, dt: f32) {
        let width = self.config.width;
        let height = self.config.height;
        let depth = self.config.depth.max(1);
        if width == 0 || height == 0 || depth == 0 {
            return;
        }

        let cell_size_mm = self.config.cell_size_mm.max(1.0e-3);
        let air_pressure = &self.air_pressure_kpa;
        let air_density = &self.air_density;
        let wind_x = &self.wind_x;
        let wind_y = &self.wind_y;
        let wind_z = &self.wind_z;
        let pressure_mean = mean(air_pressure);

        for plant in &mut self.plants {
            let canopy_height_mm = plant.physiology.height_mm().max(cell_size_mm);
            let canopy_z = ((canopy_height_mm / cell_size_mm).round() as usize).min(depth - 1);
            let x = plant.x.min(width - 1);
            let y = plant.y.min(height - 1);
            let x_l = x.saturating_sub(1);
            let x_r = (x + 1).min(width - 1);
            let y_l = y.saturating_sub(1);
            let y_r = (y + 1).min(height - 1);
            let z_l = canopy_z.saturating_sub(1);
            let z_r = (canopy_z + 1).min(depth - 1);

            let i = idx3(width, height, x, y, canopy_z);
            let i_l = idx3(width, height, x_l, y, canopy_z);
            let i_r = idx3(width, height, x_r, y, canopy_z);
            let i_d = idx3(width, height, x, y_l, canopy_z);
            let i_u = idx3(width, height, x, y_r, canopy_z);
            let i_b = idx3(width, height, x, y, z_l);
            let i_f = idx3(width, height, x, y, z_r);

            let pressure_grad_x = (air_pressure[i_r] - air_pressure[i_l]) / (2.0 * cell_size_mm);
            let pressure_grad_y = (air_pressure[i_u] - air_pressure[i_d]) / (2.0 * cell_size_mm);
            let pressure_grad_z = if depth > 1 {
                (air_pressure[i_f] - air_pressure[i_b]) / (2.0 * cell_size_mm)
            } else {
                0.0
            };
            let pressure_delta =
                (air_pressure[i] - pressure_mean) / ATMOS_PRESSURE_BASELINE_KPA.max(1.0);
            let density_delta =
                (air_density[i] - ATMOS_DENSITY_BASELINE_KG_M3) / ATMOS_DENSITY_BASELINE_KG_M3;

            let vitality = plant.cellular.vitality().clamp(0.0, 1.0);
            let energy = plant.cellular.energy_charge().clamp(0.0, 1.0);
            let health = plant.physiology.health().clamp(0.0, 1.0);
            let hydration = plant.physiology.water_buffer().clamp(0.0, 1.5);
            let canopy_area_mm2 = std::f32::consts::PI
                * plant.genome.canopy_radius_mm.powi(2)
                * (0.85 + plant.physiology.leaf_biomass() * 0.28);
            let flexible_mass = (plant.physiology.leaf_biomass()
                + plant.physiology.stem_biomass() * 0.65)
                .max(0.08);
            let stiffness = (0.28
                + plant.physiology.stem_biomass() * 0.38
                + plant.physiology.root_biomass() * 0.26
                + vitality * 0.34)
                * (0.75 + hydration * 0.25);
            let damping = 0.78 + health * 0.32 + energy * 0.26;

            let force_x =
                canopy_area_mm2 * (wind_x[i] * air_density[i] * 0.010 + pressure_grad_x * 0.0032);
            let force_z =
                canopy_area_mm2 * (wind_y[i] * air_density[i] * 0.010 + pressure_grad_y * 0.0032);
            let lift_force = canopy_area_mm2
                * (wind_z[i] * air_density[i] * 0.006 - pressure_grad_z * 0.0016
                    + density_delta * 0.12
                    - pressure_delta * 0.08);

            let lateral_limit = canopy_height_mm * 0.35;
            let vertical_limit = canopy_height_mm * 0.16;
            integrate_displacement(
                &mut plant.pose.canopy_offset_mm[0],
                &mut plant.pose.canopy_velocity_mm_s[0],
                force_x,
                stiffness,
                damping,
                flexible_mass,
                dt,
                lateral_limit,
            );
            integrate_displacement(
                &mut plant.pose.canopy_offset_mm[2],
                &mut plant.pose.canopy_velocity_mm_s[2],
                force_z,
                stiffness,
                damping,
                flexible_mass,
                dt,
                lateral_limit,
            );
            integrate_displacement(
                &mut plant.pose.canopy_offset_mm[1],
                &mut plant.pose.canopy_velocity_mm_s[1],
                lift_force,
                stiffness * 0.72,
                damping * 1.10,
                flexible_mass,
                dt,
                vertical_limit,
            );
            plant.pose.stem_tilt_x_rad = clamp(
                (-plant.pose.canopy_offset_mm[2] / canopy_height_mm).atan(),
                -0.45,
                0.45,
            );
            plant.pose.stem_tilt_z_rad = clamp(
                (plant.pose.canopy_offset_mm[0] / canopy_height_mm).atan(),
                -0.45,
                0.45,
            );
        }

        for seed in &mut self.seeds {
            let x = seed.x.round().clamp(0.0, (width - 1) as f32) as usize;
            let y = seed.y.round().clamp(0.0, (height - 1) as f32) as usize;
            let x_l = x.saturating_sub(1);
            let x_r = (x + 1).min(width - 1);
            let y_l = y.saturating_sub(1);
            let y_r = (y + 1).min(height - 1);
            let z = 0usize;
            let z_r = (z + 1).min(depth - 1);

            let i = idx3(width, height, x, y, z);
            let i_l = idx3(width, height, x_l, y, z);
            let i_r = idx3(width, height, x_r, y, z);
            let i_d = idx3(width, height, x, y_l, z);
            let i_u = idx3(width, height, x, y_r, z);
            let i_f = idx3(width, height, x, y, z_r);

            let flat = idx2(width, x, y);
            let flat_l = idx2(width, x_l, y);
            let flat_r = idx2(width, x_r, y);
            let flat_d = idx2(width, x, y_l);
            let flat_u = idx2(width, x, y_r);

            let pressure_grad_x = (air_pressure[i_r] - air_pressure[i_l]) / (2.0 * cell_size_mm);
            let pressure_grad_z = (air_pressure[i_u] - air_pressure[i_d]) / (2.0 * cell_size_mm);
            let lift_driver = if depth > 1 {
                (air_pressure[i_f] - air_pressure[i]) / cell_size_mm
            } else {
                0.0
            };
            let moisture_grad_x = (self.moisture[flat_r] - self.moisture[flat_l]) / 2.0;
            let moisture_grad_z = (self.moisture[flat_u] - self.moisture[flat_d]) / 2.0;
            let nutrient_center = self.shallow_nutrients[flat] + self.symbiont_biomass[flat] * 0.35;
            let nutrient_grad_x = ((self.shallow_nutrients[flat_r]
                + self.symbiont_biomass[flat_r] * 0.35)
                - (self.shallow_nutrients[flat_l] + self.symbiont_biomass[flat_l] * 0.35))
                / 2.0;
            let nutrient_grad_z = ((self.shallow_nutrients[flat_u]
                + self.symbiont_biomass[flat_u] * 0.35)
                - (self.shallow_nutrients[flat_d] + self.symbiont_biomass[flat_d] * 0.35))
                / 2.0;

            let reserve_t = clamp(seed.cellular.reserve_carbon_equivalent() / 0.2, 0.0, 1.0);
            let dormancy_t = clamp(seed.dormancy_s / 26_000.0, 0.0, 1.0);
            let hydration_t = clamp(seed.cellular.hydration() / 1.2, 0.0, 1.0);
            let vitality_t = seed.cellular.vitality().clamp(0.0, 1.0);
            let radicle_t = (seed.cellular.last_feedback().radicle_extension / 1.5).clamp(0.0, 1.0);
            let shelter_t = clamp(self.canopy_cover[flat] / 1.4, 0.0, 1.0);
            let root_bias_t = clamp(seed.genome.root_depth_bias / 1.1, 0.0, 1.0);
            let symbiosis_t = clamp((seed.genome.symbiosis_affinity - 0.35) / 1.45, 0.0, 1.0);
            let seed_mass_t = clamp((seed.genome.seed_mass - 0.03) / 0.17, 0.0, 1.0);
            let exposure = (1.0 - shelter_t * 0.72) * (0.35 + (1.0 - dormancy_t) * 0.65);
            let seed_radius_mm = (cell_size_mm
                * (0.14
                    + reserve_t * 0.18
                    + seed_mass_t * 0.12
                    + hydration_t * 0.10
                    + vitality_t * 0.08
                    + radicle_t * 0.06))
                .max(cell_size_mm * 0.08);
            let frontal_area_mm2 = std::f32::consts::PI * seed_radius_mm * seed_radius_mm;
            let settle_force = -(0.22 + dormancy_t * 0.30 + shelter_t * 0.16) * frontal_area_mm2;
            let force_x = frontal_area_mm2
                * exposure
                * (wind_x[i] * air_density[i] * 0.0045
                    + pressure_grad_x * 0.0016
                    + moisture_grad_x * root_bias_t * 0.42
                    + nutrient_grad_x * symbiosis_t * 0.28);
            let force_z = frontal_area_mm2
                * exposure
                * (wind_y[i] * air_density[i] * 0.0045
                    + pressure_grad_z * 0.0016
                    + moisture_grad_z * root_bias_t * 0.42
                    + nutrient_grad_z * symbiosis_t * 0.28);
            let lift_force = frontal_area_mm2
                * exposure
                * (wind_z[i] * air_density[i] * 0.003 - lift_driver * 0.0008
                    + hydration_t * 0.12
                    + nutrient_center * 0.04)
                + settle_force;
            let stiffness = 0.62
                + dormancy_t * 0.30
                + shelter_t * 0.18
                + seed.cellular.last_feedback().coat_integrity * 0.18;
            let damping = 0.80 + hydration_t * 0.18 + seed_mass_t * 0.12 + vitality_t * 0.10;
            let mass = (0.04 + seed_mass_t * 0.07 + reserve_t * 0.08 + vitality_t * 0.04).max(0.04);

            integrate_displacement(
                &mut seed.pose.offset_mm[0],
                &mut seed.pose.velocity_mm_s[0],
                force_x,
                stiffness,
                damping,
                mass,
                dt,
                seed_radius_mm * 0.45,
            );
            integrate_displacement(
                &mut seed.pose.offset_mm[2],
                &mut seed.pose.velocity_mm_s[2],
                force_z,
                stiffness,
                damping,
                mass,
                dt,
                seed_radius_mm * 0.45,
            );
            integrate_displacement(
                &mut seed.pose.offset_mm[1],
                &mut seed.pose.velocity_mm_s[1],
                lift_force,
                stiffness * 1.15,
                damping * 1.08,
                mass,
                dt,
                seed_radius_mm * 0.16,
            );

            let target_pitch = clamp(
                -seed.pose.offset_mm[2] / seed_radius_mm.max(1.0e-3) * 0.18
                    - moisture_grad_z * root_bias_t * 0.24
                    - nutrient_grad_z * symbiosis_t * 0.12,
                -0.42,
                0.42,
            );
            let target_roll = clamp(
                seed.pose.offset_mm[0] / seed_radius_mm.max(1.0e-3) * 0.18
                    + moisture_grad_x * root_bias_t * 0.24
                    + nutrient_grad_x * symbiosis_t * 0.12,
                -0.42,
                0.42,
            );
            let target_yaw = if nutrient_grad_x.abs() + nutrient_grad_z.abs() > 1.0e-4 {
                clamp(nutrient_grad_z.atan2(nutrient_grad_x) * 0.22, -0.55, 0.55)
            } else {
                0.0
            };
            let relax = (dt * 4.0).clamp(0.0, 1.0);
            seed.pose.rotation_xyz_rad[0] += (target_pitch - seed.pose.rotation_xyz_rad[0]) * relax;
            seed.pose.rotation_xyz_rad[1] += (target_yaw - seed.pose.rotation_xyz_rad[1]) * relax;
            seed.pose.rotation_xyz_rad[2] += (target_roll - seed.pose.rotation_xyz_rad[2]) * relax;
        }

        for fruit in &mut self.fruits {
            let x = fruit.source.x.min(width - 1);
            let y = fruit.source.y.min(height - 1);
            let z = fruit.source.z.min(depth - 1);
            let x_l = x.saturating_sub(1);
            let x_r = (x + 1).min(width - 1);
            let y_l = y.saturating_sub(1);
            let y_r = (y + 1).min(height - 1);
            let z_l = z.saturating_sub(1);
            let z_r = (z + 1).min(depth - 1);

            let i = idx3(width, height, x, y, z);
            let i_l = idx3(width, height, x_l, y, z);
            let i_r = idx3(width, height, x_r, y, z);
            let i_d = idx3(width, height, x, y_l, z);
            let i_u = idx3(width, height, x, y_r, z);
            let i_b = idx3(width, height, x, y, z_l);
            let i_f = idx3(width, height, x, y, z_r);

            let pressure_grad_x = (air_pressure[i_r] - air_pressure[i_l]) / (2.0 * cell_size_mm);
            let pressure_grad_y = (air_pressure[i_u] - air_pressure[i_d]) / (2.0 * cell_size_mm);
            let pressure_grad_z = if depth > 1 {
                (air_pressure[i_f] - air_pressure[i_b]) / (2.0 * cell_size_mm)
            } else {
                0.0
            };
            let pressure_delta =
                (air_pressure[i] - pressure_mean) / ATMOS_PRESSURE_BASELINE_KPA.max(1.0);
            let density_delta =
                (air_density[i] - ATMOS_DENSITY_BASELINE_KG_M3) / ATMOS_DENSITY_BASELINE_KG_M3;

            let radius_mm = (fruit.radius * cell_size_mm).max(cell_size_mm * 0.5);
            let frontal_area_mm2 = std::f32::consts::PI * radius_mm * radius_mm;
            let stem_length_mm = (radius_mm * 1.8 + cell_size_mm * 2.0).max(cell_size_mm);
            let sugar = fruit.source.sugar_content.clamp(0.0, 1.5);
            let ripeness = fruit.source.ripeness.clamp(0.0, 1.0);
            let tether_mass = (0.06 + radius_mm * 0.02 + sugar * 0.22).max(0.05);
            let stiffness = 0.42 + (1.0 - ripeness) * 0.20 + sugar * 0.10;
            let damping = 0.62 + ripeness * 0.18;
            let alive_gate = if fruit.source.alive { 1.0 } else { 0.2 };
            let force_x = frontal_area_mm2
                * alive_gate
                * (wind_x[i] * air_density[i] * 0.012 + pressure_grad_x * 0.0026);
            let force_z = frontal_area_mm2
                * alive_gate
                * (wind_y[i] * air_density[i] * 0.012 + pressure_grad_y * 0.0026);
            let lift_force = frontal_area_mm2
                * alive_gate
                * (wind_z[i] * air_density[i] * 0.007 - pressure_grad_z * 0.0018
                    + density_delta * 0.08
                    - pressure_delta * 0.10);

            integrate_displacement(
                &mut fruit.pose.offset_mm[0],
                &mut fruit.pose.velocity_mm_s[0],
                force_x,
                stiffness,
                damping,
                tether_mass,
                dt,
                stem_length_mm * 0.75,
            );
            integrate_displacement(
                &mut fruit.pose.offset_mm[2],
                &mut fruit.pose.velocity_mm_s[2],
                force_z,
                stiffness,
                damping,
                tether_mass,
                dt,
                stem_length_mm * 0.75,
            );
            integrate_displacement(
                &mut fruit.pose.offset_mm[1],
                &mut fruit.pose.velocity_mm_s[1],
                lift_force,
                stiffness * 0.76,
                damping * 1.05,
                tether_mass,
                dt,
                stem_length_mm * 0.28,
            );
        }
    }
}
