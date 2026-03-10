//! Coarse whole-plant physiology for terrarium ecology.

use crate::constants::clamp;

#[derive(Debug, Clone)]
pub struct PlantStepReport {
    pub exudates: f32,
    pub litter: f32,
    pub spawned_fruit: bool,
    pub fruit_size: f32,
    pub spawned_seed: bool,
}

#[derive(Debug, Clone)]
pub struct PlantOrganismSim {
    max_height_mm: f32,
    canopy_radius_mm: f32,
    root_radius_mm: f32,
    leaf_efficiency: f32,
    root_uptake_efficiency: f32,
    water_use_efficiency: f32,
    volatile_scale: f32,
    fruiting_threshold: f32,
    litter_turnover: f32,
    shade_tolerance: f32,
    root_depth_bias: f32,
    symbiosis_affinity: f32,
    seed_mass: f32,
    leaf_biomass: f32,
    stem_biomass: f32,
    root_biomass: f32,
    storage_carbon: f32,
    water_buffer: f32,
    nitrogen_buffer: f32,
    fruit_timer_s: f32,
    seed_timer_s: f32,
    age_s: f32,
    health: f32,
    fruit_count: u32,
    height_mm: f32,
    nectar_production_rate: f32,
    odorant_geraniol: f32,
    odorant_ethyl_acetate: f32,
    odorant_emission_rate: f32,
}

impl PlantOrganismSim {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        max_height_mm: f32,
        canopy_radius_mm: f32,
        root_radius_mm: f32,
        leaf_efficiency: f32,
        root_uptake_efficiency: f32,
        water_use_efficiency: f32,
        volatile_scale: f32,
        fruiting_threshold: f32,
        litter_turnover: f32,
        shade_tolerance: f32,
        root_depth_bias: f32,
        symbiosis_affinity: f32,
        seed_mass: f32,
        leaf_biomass: f32,
        stem_biomass: f32,
        root_biomass: f32,
        storage_carbon: f32,
        water_buffer: f32,
        nitrogen_buffer: f32,
        fruit_timer_s: f32,
        seed_timer_s: f32,
    ) -> Self {
        Self {
            max_height_mm,
            canopy_radius_mm,
            root_radius_mm,
            leaf_efficiency,
            root_uptake_efficiency,
            water_use_efficiency,
            volatile_scale,
            fruiting_threshold,
            litter_turnover,
            shade_tolerance,
            root_depth_bias,
            symbiosis_affinity,
            seed_mass,
            leaf_biomass,
            stem_biomass,
            root_biomass,
            storage_carbon,
            water_buffer,
            nitrogen_buffer,
            fruit_timer_s,
            seed_timer_s,
            age_s: 0.0,
            health: 1.0,
            fruit_count: 0,
            height_mm: 6.0,
            nectar_production_rate: 0.02,
            odorant_geraniol: 0.78,
            odorant_ethyl_acetate: 0.0,
            odorant_emission_rate: 0.02,
        }
    }

    pub fn total_biomass(&self) -> f32 {
        self.leaf_biomass + self.stem_biomass + self.root_biomass + self.storage_carbon
    }

    pub fn leaf_biomass(&self) -> f32 {
        self.leaf_biomass
    }

    pub fn stem_biomass(&self) -> f32 {
        self.stem_biomass
    }

    pub fn root_biomass(&self) -> f32 {
        self.root_biomass
    }

    pub fn storage_carbon(&self) -> f32 {
        self.storage_carbon
    }

    pub fn water_buffer(&self) -> f32 {
        self.water_buffer
    }

    pub fn nitrogen_buffer(&self) -> f32 {
        self.nitrogen_buffer
    }

    pub fn fruit_timer_s(&self) -> f32 {
        self.fruit_timer_s
    }

    pub fn seed_timer_s(&self) -> f32 {
        self.seed_timer_s
    }

    pub fn age_s(&self) -> f32 {
        self.age_s
    }

    pub fn health(&self) -> f32 {
        self.health
    }

    pub fn fruit_count(&self) -> u32 {
        self.fruit_count
    }

    pub fn height_mm(&self) -> f32 {
        self.height_mm
    }

    pub fn nectar_production_rate(&self) -> f32 {
        self.nectar_production_rate
    }

    pub fn odorant_geraniol(&self) -> f32 {
        self.odorant_geraniol
    }

    pub fn odorant_ethyl_acetate(&self) -> f32 {
        self.odorant_ethyl_acetate
    }

    pub fn odorant_emission_rate(&self) -> f32 {
        self.odorant_emission_rate
    }

    pub fn resource_demands(
        &self,
        dt: f32,
        root_energy_gate: f32,
        root_water_deficit: f32,
        root_nitrogen_deficit: f32,
    ) -> (f32, f32) {
        let water_demand = (0.00082 + self.root_biomass * 0.00060 + root_water_deficit * 0.00008)
            * dt
            * self.root_uptake_efficiency
            * root_energy_gate;
        let nutrient_demand =
            (0.00034 + self.root_biomass * 0.00020 + root_nitrogen_deficit * 0.00010)
                * dt
                * self.root_uptake_efficiency
                * root_energy_gate;
        (water_demand.max(0.0), nutrient_demand.max(0.0))
    }

    pub fn is_dead(&self) -> bool {
        self.total_biomass() < 0.09 || (self.health < 0.03 && self.storage_carbon < -0.2)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        dt: f32,
        water_uptake: f32,
        nutrient_uptake: f32,
        local_light: f32,
        temp_factor: f32,
        root_pressure: f32,
        symbiosis_bonus: f32,
        water_factor: f32,
        nutrient_factor: f32,
        canopy_competition: f32,
        root_competition: f32,
        soil_glucose: f32,
        cell_photosynthetic_capacity: f32,
        cell_maintenance_cost: f32,
        cell_storage_exchange: f32,
        cell_division_growth: f32,
        cell_senescence_mass: f32,
        cell_energy_charge: f32,
        cell_vitality: f32,
        cell_sugar_pool: f32,
        cell_division_signal: f32,
        total_cells: f32,
        fruit_reset_s: f32,
        seed_reset_s: f32,
    ) -> PlantStepReport {
        self.age_s += dt;
        self.water_buffer = clamp(self.water_buffer + water_uptake, 0.0, 3.2);
        self.nitrogen_buffer = clamp(self.nitrogen_buffer + nutrient_uptake, 0.0, 2.8);

        let photosynthesis = self.leaf_biomass
            * self.leaf_efficiency
            * local_light
            * temp_factor
            * water_factor.min(1.0)
            * nutrient_factor.min(1.0)
            * symbiosis_bonus
            * root_pressure
            * cell_photosynthetic_capacity
            * 0.0023
            * dt;

        let maintenance = self.total_biomass()
            * (0.00003 + 0.00005 * temp_factor + 0.000012 * canopy_competition)
            * dt
            + cell_maintenance_cost;
        self.storage_carbon = clamp(
            self.storage_carbon + photosynthesis - maintenance + cell_storage_exchange,
            -0.28,
            6.0,
        );

        let water_used = self.water_buffer.min(
            (0.00018 + self.leaf_biomass * 0.00012) * dt / self.water_use_efficiency.max(1e-6),
        );
        let nitrogen_used = self
            .nitrogen_buffer
            .min((0.00010 + self.leaf_biomass * 0.00006) * dt / symbiosis_bonus.max(1.0));
        self.water_buffer -= water_used;
        self.nitrogen_buffer -= nitrogen_used;

        if self.storage_carbon > 0.03 {
            let allocation = self.storage_carbon.min(
                0.00042 * dt * root_pressure * (0.75 + local_light * 0.55) + cell_division_growth,
            );
            self.storage_carbon -= allocation;
            let leaf_share = clamp(
                0.37 + self.shade_tolerance * 0.05 - canopy_competition * 0.03,
                0.24,
                0.52,
            );
            let root_share = clamp(
                0.34 + self.root_depth_bias * 0.10 + root_competition * 0.03,
                0.24,
                0.50,
            );
            let stem_share = clamp(1.0 - leaf_share - root_share, 0.14, 0.36);
            let total_share = leaf_share + root_share + stem_share;
            self.leaf_biomass += allocation * (leaf_share / total_share);
            self.root_biomass += allocation * (root_share / total_share);
            self.stem_biomass += allocation * (stem_share / total_share);
        } else if self.storage_carbon < -0.03 {
            let stress = self
                .storage_carbon
                .abs()
                .min(0.00022 * dt * (1.0 + canopy_competition * 0.2));
            self.leaf_biomass = (self.leaf_biomass - stress * 0.52).max(0.04);
            self.stem_biomass = (self.stem_biomass - stress * 0.20).max(0.04);
            self.root_biomass = (self.root_biomass - stress * 0.28).max(0.04);
        }

        self.health = clamp(
            0.14 + local_light.min(1.0) * 0.24
                + water_factor.min(1.0) * 0.24
                + nutrient_factor.min(1.0) * 0.18
                + clamp(self.storage_carbon, 0.0, 0.9) * 0.40
                + clamp(symbiosis_bonus - 1.0, 0.0, 1.0) * 0.12
                + cell_vitality * 0.16
                + cell_energy_charge * 0.08
                - canopy_competition * 0.05,
            0.0,
            1.6,
        );

        self.height_mm = clamp(
            2.0 + (self.leaf_biomass * 1.6 + self.stem_biomass * 3.6) * (2.2 + self.health * 0.25)
                + total_cells * 0.0015,
            2.0,
            self.max_height_mm,
        );
        self.nectar_production_rate = clamp(
            0.02 + self.health * 0.12 + cell_sugar_pool * 0.002,
            0.0,
            0.26,
        );
        self.odorant_geraniol = clamp(0.78 + self.health * 0.16 + cell_vitality * 0.10, 0.50, 1.30);
        self.odorant_ethyl_acetate = clamp(
            self.fruit_count as f32 * 0.015 + cell_sugar_pool * 0.002 + soil_glucose * 0.08,
            0.0,
            0.16,
        );
        self.odorant_emission_rate = clamp(
            (0.003
                + self.health * 0.010
                + self.fruit_count as f32 * 0.0012
                + self.leaf_biomass * 0.003
                + cell_division_signal * 0.003)
                * self.volatile_scale,
            0.002,
            0.09,
        );

        let exudates = (self.storage_carbon + 0.22)
            .min(
                (0.00005 + self.root_biomass * 0.00006)
                    * dt
                    * (0.8 + self.symbiosis_affinity * 0.4),
            )
            .max(0.0);
        if exudates > 0.0 {
            self.storage_carbon -= exudates * 0.42;
        }

        let litter = ((self.leaf_biomass + self.root_biomass)
            * (0.00002 + (1.0 - self.health.min(1.0)) * 0.00008)
            * dt
            * self.litter_turnover)
            + cell_senescence_mass;

        self.fruit_timer_s -= dt;
        self.seed_timer_s -= dt;

        let mut spawned_fruit = false;
        let mut fruit_size = 0.0;
        if self.storage_carbon > self.fruiting_threshold
            && self.fruit_timer_s <= 0.0
            && self.health > 0.52
            && cell_sugar_pool > 2.0
        {
            spawned_fruit = true;
            fruit_size = clamp(0.45 + self.storage_carbon * 0.46, 0.35, 1.5);
            self.storage_carbon = (self.storage_carbon - 0.18).max(-0.12);
            self.fruit_count += 1;
            self.fruit_timer_s = fruit_reset_s.max(1.0);
        }

        let seed_threshold = self.fruiting_threshold + 0.12 + self.seed_mass * 0.8;
        let mut spawned_seed = false;
        if self.storage_carbon > seed_threshold
            && self.seed_timer_s <= 0.0
            && self.health > 0.58
            && cell_division_signal > 0.02
        {
            spawned_seed = true;
            self.storage_carbon = (self.storage_carbon - (0.06 + self.seed_mass * 0.55)).max(-0.12);
            self.seed_timer_s = seed_reset_s.max(1.0);
        }

        PlantStepReport {
            exudates,
            litter: litter.max(0.0),
            spawned_fruit,
            fruit_size,
            spawned_seed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PlantOrganismSim;

    #[test]
    fn plant_organism_state_stays_bounded() {
        let mut plant = PlantOrganismSim::new(
            12.0, 5.0, 4.0, 1.0, 1.0, 1.0, 1.0, 0.8, 1.0, 1.0, 0.5, 1.0, 0.08, 0.14, 0.18, 0.17,
            0.12, 0.25, 0.14, 8000.0, 14000.0,
        );
        for _ in 0..200 {
            let (_wd, _nd) = plant.resource_demands(20.0, 0.9, 0.2, 0.1);
            let _report = plant.step(
                20.0, 0.02, 0.01, 0.7, 0.95, 0.9, 1.2, 0.9, 0.8, 0.1, 0.1, 0.02, 1.0, 0.001, 0.0,
                0.0001, 0.0, 0.8, 0.7, 2.0, 0.2, 400.0, 9000.0, 15000.0,
            );
        }
        assert!(plant.leaf_biomass() >= 0.0);
        assert!(plant.root_biomass() >= 0.0);
        assert!(plant.health() >= 0.0);
        assert!(plant.height_mm() >= 2.0);
    }
}
