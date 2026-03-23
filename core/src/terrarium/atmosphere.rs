//! Terrarium Atmosphere: Molecular gas dynamics and density.

use super::*;

impl TerrariumWorld {
    /// Calculate local air density based on real molecular partial pressures and local temperature.
    /// Uses sum of partial pressures (Ideal Gas Law): rho = sum(P_i / (R_i * T))
    pub fn calculate_local_air_density(&self, cell_idx: usize) -> f32 {
        let temp_c = self.temperature[cell_idx];
        let t_kelvin = temp_c + 273.15;

        // R values (gas constants) in J/(kg*K)
        const R_N2: f32 = 296.8;
        const R_O2: f32 = 259.8;
        const R_CO2: f32 = 188.9;
        const R_H2O: f32 = 461.5;

        // 1. Fixed Nitrogen buffer (approx 78% of standard atmosphere)
        let p_n2 = 79.0;
        let rho_n2 = p_n2 / (R_N2 * t_kelvin * 0.001);

        // 2. Dynamic Oxygen
        let p_o2 = self.odorants[ATMOS_O2_IDX][cell_idx] * 101.325;
        let rho_o2 = p_o2 / (R_O2 * t_kelvin * 0.001);

        // 3. Dynamic CO2
        let p_co2 = self.odorants[ATMOS_CO2_IDX][cell_idx] * 101.325;
        let rho_co2 = p_co2 / (R_CO2 * t_kelvin * 0.001);

        // 4. Dynamic Water Vapor (Humidity)
        let p_sat = 0.61078 * ((17.27 * temp_c) / (temp_c + 237.3)).exp();
        let p_h2o = self.humidity[cell_idx] * p_sat;
        let rho_h2o = p_h2o / (R_H2O * t_kelvin * 0.001);

        rho_n2 + rho_o2 + rho_co2 + rho_h2o
    }

    pub fn sample_odorant_patch(
        &self,
        odor_idx: usize,
        x: usize,
        y: usize,
        z: usize,
        radius: usize,
    ) -> f32 {
        let field = &self.odorants[odor_idx];
        let w = self.config.width;
        let h = self.config.height;
        let d = self.config.depth.max(1);

        let mut sum = 0.0;
        let mut count = 0.0;

        let x0 = x.saturating_sub(radius);
        let x1 = (x + radius + 1).min(w);
        let y0 = y.saturating_sub(radius);
        let y1 = (y + radius + 1).min(h);

        for cy in y0..y1 {
            for cx in x0..x1 {
                sum += field[idx3(w, h, cx, cy, z.min(d - 1))];
                count += 1.0;
            }
        }
        if count > 0.0 {
            sum / count
        } else {
            0.0
        }
    }

    pub fn exchange_atmosphere_odorant(
        &mut self,
        odor_idx: usize,
        x: usize,
        y: usize,
        z: usize,
        radius: usize,
        amount: f32,
    ) {
        exchange_layer_patch(
            &mut self.odorants[odor_idx],
            self.config.width,
            self.config.height,
            self.config.depth.max(1),
            x,
            y,
            z,
            radius,
            amount,
            0.0,
            1.0,
        );
    }

    pub fn exchange_atmosphere_humidity(
        &mut self,
        x: usize,
        y: usize,
        z: usize,
        radius: usize,
        amount: f32,
    ) {
        exchange_layer_patch(
            &mut self.humidity,
            self.config.width,
            self.config.height,
            self.config.depth.max(1),
            x,
            y,
            z,
            radius,
            amount,
            0.0,
            1.0,
        );
    }

    pub fn sample_temperature_at(&self, x: usize, y: usize, z: usize) -> f32 {
        let w = self.config.width;
        let h = self.config.height;
        let d = self.config.depth.max(1);
        self.temperature[idx3(w, h, x.min(w - 1), y.min(h - 1), z.min(d - 1))]
    }

    pub fn sample_humidity_at(&self, x: usize, y: usize, z: usize) -> f32 {
        let w = self.config.width;
        let h = self.config.height;
        let d = self.config.depth.max(1);
        self.humidity[idx3(w, h, x.min(w - 1), y.min(h - 1), z.min(d - 1))]
    }

    pub fn step_atmosphere(&mut self) -> Result<(), String> {
        let mut fruit_sources: Vec<FruitSourceState> =
            self.fruits.iter().map(|f| f.source.clone()).collect();
        let plant_sources = self.plant_source_states();
        step_molecular_world_fields(
            self.config.width,
            self.config.height,
            self.config.depth.max(1),
            self.config.world_dt_s,
            self.config.cell_size_mm,
            true,
            self.daylight(),
            Some(AMMONIA_IDX),
            &mut fruit_sources,
            &plant_sources,
            &mut self.waters,
            &mut self.odorants,
            &self.odorant_params,
            &mut self.temperature,
            &mut self.humidity,
            &mut self.wind_x,
            &mut self.wind_y,
            &mut self.wind_z,
            &mut self.atmosphere_rng_state,
        )?;
        for (dst, src) in self.fruits.iter_mut().zip(fruit_sources.into_iter()) {
            dst.source = src;
        }
        Ok(())
    }
}
