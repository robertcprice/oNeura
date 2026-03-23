use super::*;

const SECONDS_PER_YEAR: f64 = 365.25 * 86_400.0;
const ATMOS_PRESSURE_BASELINE_KPA: f32 = 101.325;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TerrariumClimateDriver {
    pub scenario: ClimateScenario,
    pub climate_seed: u64,
    pub engine: ClimateEngine,
    pub last_sample: Option<crate::climate_scenarios::ClimateState>,
    pub atmosphere_relaxation: f32,
    pub soil_relaxation: f32,
    /// Year of last extreme event check (Poisson-sampled once per year).
    pub last_extreme_event_year: f64,
}

impl TerrariumClimateDriver {
    pub fn new(scenario: ClimateScenario, climate_seed: u64, start_year: Option<f64>) -> Self {
        let mut engine = ClimateEngine::new(scenario.clone(), climate_seed);
        if let Some(year) = start_year {
            engine.set_current_year(year);
        }
        let last_sample = Some(engine.state_at(engine.current_year()));
        let initial_year = engine.current_year();
        Self {
            scenario,
            climate_seed,
            engine,
            last_sample,
            atmosphere_relaxation: 0.08,
            soil_relaxation: 0.03,
            last_extreme_event_year: initial_year.floor(),
        }
    }
}

impl TerrariumWorld {
    pub fn enable_climate_driver(
        &mut self,
        scenario: ClimateScenario,
        climate_seed: u64,
        start_year: Option<f64>,
    ) {
        self.climate_driver = Some(TerrariumClimateDriver::new(
            scenario,
            climate_seed,
            start_year,
        ));
        if let Some(sample) = self
            .climate_driver
            .as_ref()
            .and_then(|driver| driver.last_sample.clone())
        {
            self.apply_climate_boundary_conditions(&sample);
        }
    }

    pub fn disable_climate_driver(&mut self) {
        self.climate_driver = None;
    }

    pub fn current_year(&self) -> Option<f64> {
        self.climate_driver
            .as_ref()
            .map(|driver| driver.engine.current_year())
    }

    pub fn climate_state(&self) -> Option<&crate::climate_scenarios::ClimateState> {
        self.climate_driver
            .as_ref()
            .and_then(|driver| driver.last_sample.as_ref())
    }

    pub(super) fn step_climate_driver(&mut self, dt_s: f32) {
        let Some(driver) = self.climate_driver.as_mut() else {
            return;
        };
        let sample = driver
            .engine
            .step((dt_s.max(0.0) as f64) / SECONDS_PER_YEAR);
        driver.last_sample = Some(sample.clone());

        // Poisson-sample extreme events once per simulated year.
        // IPCC AR5 WG2 Ch.18: extreme event frequency scales +20 % per °C warming.
        let current_year = driver.engine.current_year().floor();
        let pending_events = if current_year > driver.last_extreme_event_year {
            driver.last_extreme_event_year = current_year;
            driver.engine.extreme_events_in_year(current_year)
        } else {
            Vec::new()
        };

        self.apply_climate_boundary_conditions(&sample);

        // Apply any natural extreme events.
        for event in pending_events {
            let (etype, sev) = match &event {
                crate::climate_scenarios::ExtremeEvent::Drought { severity, .. } => {
                    ("drought", *severity as f32)
                }
                crate::climate_scenarios::ExtremeEvent::Flood { magnitude } => {
                    ("flood", *magnitude as f32)
                }
                crate::climate_scenarios::ExtremeEvent::Heatwave { .. } => ("heatwave", 0.6),
                crate::climate_scenarios::ExtremeEvent::ColdSnap { .. } => ("coldsnap", 0.6),
                crate::climate_scenarios::ExtremeEvent::Wildfire { intensity } => {
                    ("wildfire", *intensity as f32)
                }
                crate::climate_scenarios::ExtremeEvent::Hurricane { category } => {
                    ("hurricane", (*category as f32 / 5.0).clamp(0.2, 1.0))
                }
            };
            self.apply_extreme_event_manual(etype, Some(sev));
        }
    }

    fn apply_climate_boundary_conditions(
        &mut self,
        sample: &crate::climate_scenarios::ClimateState,
    ) {
        let Some(driver) = self.climate_driver.as_ref() else {
            return;
        };

        let atm_relax = driver.atmosphere_relaxation.clamp(0.0, 1.0);
        let soil_relax = driver.soil_relaxation.clamp(0.0, 1.0);
        let target_temp = sample.temperature_c as f32 + self.weather.temperature_offset_c;
        let target_humidity = sample.humidity as f32;
        let target_co2 = (sample.co2_ppm as f32 / 10_000.0).clamp(0.0, 1.0);
        let target_o2 = ATMOS_O2_BASELINE;
        let season_phase = ((sample.year.fract() as f32) * std::f32::consts::TAU)
            .rem_euclid(std::f32::consts::TAU);
        let target_wind_x =
            (sample.wind_speed_m_s as f32 * season_phase.cos() * 0.08).clamp(-2.0, 2.0);
        let target_wind_y =
            (sample.wind_speed_m_s as f32 * season_phase.sin() * 0.08).clamp(-2.0, 2.0);
        let target_pressure = (ATMOS_PRESSURE_BASELINE_KPA - (target_temp - 15.0) * 0.035
            + target_humidity * 0.22
            + (sample.wind_speed_m_s as f32).min(18.0) * 0.03
            - sample.flood_risk as f32 * 0.12)
            .clamp(94.0, 108.0);

        for ((temp, humid), ((wind_x, wind_y), wind_z)) in self
            .temperature
            .iter_mut()
            .zip(self.humidity.iter_mut())
            .zip(
                self.wind_x
                    .iter_mut()
                    .zip(self.wind_y.iter_mut())
                    .zip(self.wind_z.iter_mut()),
            )
        {
            *temp += (target_temp - *temp) * atm_relax;
            *humid = clamp(*humid + (target_humidity - *humid) * atm_relax, 0.0, 1.0);
            *wind_x += (target_wind_x - *wind_x) * atm_relax;
            *wind_y += (target_wind_y - *wind_y) * atm_relax;
            *wind_z *= 1.0 - atm_relax * 0.35;
        }

        for gas in &mut self.odorants[ATMOS_CO2_IDX] {
            *gas = clamp(*gas + (target_co2 - *gas) * atm_relax * 0.45, 0.0, 1.0);
        }
        for gas in &mut self.odorants[ATMOS_O2_IDX] {
            *gas = clamp(*gas + (target_o2 - *gas) * atm_relax * 0.18, 0.0, 1.0);
        }
        for pressure in &mut self.air_pressure_kpa {
            *pressure += (target_pressure - *pressure) * atm_relax;
        }

        let precip_t = ((sample.precipitation_mm as f32) / 1_200.0).clamp(0.0, 1.6);
        let drought_t = sample.drought_severity as f32;
        let flood_t = sample.flood_risk as f32;
        let moisture_target = clamp(
            0.08 + target_humidity * 0.16 + precip_t * 0.14 + flood_t * 0.12 - drought_t * 0.18,
            0.02,
            0.98,
        );
        let deep_target = clamp(
            moisture_target * 0.84 + precip_t * 0.10 + flood_t * 0.10,
            0.04,
            1.0,
        );
        for (surface, deep) in self.moisture.iter_mut().zip(self.deep_moisture.iter_mut()) {
            *surface = clamp(
                *surface + (moisture_target - *surface) * soil_relax,
                0.0,
                1.0,
            );
            *deep = clamp(*deep + (deep_target - *deep) * soil_relax, 0.0, 1.0);
        }
    }
}
