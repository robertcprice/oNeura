//! Wind biomechanics: drag force, stem bending stress, damage accumulation.
//!
//! Physical basis:
//! - Wind drag: F = 0.5 rho Cd A v^2 (standard aerodynamic drag)
//! - Stem bending: sigma = M/S where M = F*h and S = pi d^3/32 (cantilever beam)
//! - Yield strength: sigma_yield = 0.0005 * rho^1.5 (Niklas 1992, Plant Biomechanics)
//! - Damage accumulation: Hill kinetics on stress/yield ratio -- no hardcoded thresholds

use super::*;
use crate::botany::physiology_bridge::{hill, hill_repression};

/// Air density in kg/m^3 at ~22C, sea level.
const AIR_DENSITY_KG_M3: f32 = 1.2;

/// Wind drag force on a plant canopy (Newtons, scaled to terrarium mm units).
///
/// F = 0.5 * rho * Cd * A * v^2
/// where A = canopy cross-sectional area ~ canopy_radius * height (projected rectangle)
pub fn wind_drag_force(wind_speed_mm_s: f32, canopy_radius_mm: f32, height_mm: f32, drag_coeff: f32) -> f32 {
    if wind_speed_mm_s <= 0.0 || height_mm <= 0.0 {
        return 0.0;
    }
    // Convert mm/s to m/s for force calculation, then convert area mm^2 to m^2
    let v_m_s = wind_speed_mm_s * 0.001;
    let area_m2 = canopy_radius_mm * height_mm * 1e-6; // mm^2 -> m^2
    0.5 * AIR_DENSITY_KG_M3 * drag_coeff * area_m2 * v_m_s * v_m_s
}

/// Stem bending stress (Pa) from wind force applied at canopy center-of-mass.
///
/// Treats stem as a cantilever beam with circular cross-section.
/// sigma = M / S where M = F * h/2 (moment), S = pi d^3 / 32 (section modulus)
pub fn stem_bending_stress(force_n: f32, height_mm: f32, stem_diameter_mm: f32) -> f32 {
    if stem_diameter_mm <= 0.0 || height_mm <= 0.0 {
        return 0.0;
    }
    let h_m = height_mm * 0.001;
    let d_m = stem_diameter_mm * 0.001;
    let moment = force_n * h_m * 0.5;
    let section_modulus = std::f32::consts::PI * d_m * d_m * d_m / 32.0;
    if section_modulus < 1e-18 {
        return 0.0;
    }
    moment / section_modulus
}

/// Stem yield strength (Pa) from wood density (kg/m^3).
///
/// Empirical allometric: sigma_yield ~ 0.0005 * rho^1.5 (Niklas 1992)
/// Herbaceous stems have low density (~150-200), wood has ~500-700.
pub fn stem_yield_strength(wood_density: f32) -> f32 {
    0.0005 * wood_density.max(10.0).powf(1.5)
}

/// Elastic beam deflection angle (radians) from wind force.
///
/// theta ~ F * L^2 / (3 E I) where E ~ 0.01 * rho^2 (Young's modulus approximation)
/// and I = pi d^4 / 64 (second moment of area).
pub fn wind_deflection_angle(force_n: f32, height_mm: f32, stem_diameter_mm: f32, wood_density: f32) -> f32 {
    let h_m = height_mm.max(0.001) * 0.001;
    let d_m = stem_diameter_mm.max(0.001) * 0.001;
    let young_modulus = 0.01 * wood_density.max(10.0) * wood_density.max(10.0); // Pa
    let inertia = std::f32::consts::PI * d_m.powi(4) / 64.0;
    let stiffness = 3.0 * young_modulus * inertia;
    if stiffness < 1e-18 {
        return 0.0;
    }
    let angle = force_n * h_m * h_m / stiffness;
    angle.clamp(0.0, std::f32::consts::FRAC_PI_2) // cap at 90 degrees
}

/// Mechanical viability modifier: reduces plant viability when stem damage is high.
///
/// Returns [0.0, 1.0] where 1.0 = no damage, 0.0 = totally broken.
/// Uses Hill repression: viability drops sharply around damage = 0.5.
pub fn mechanical_viability_modifier(damage: f32) -> f32 {
    hill_repression(damage, 0.5, 3.0)
}

/// Classify the continuous weather state into a diagnostic regime label.
///
/// This is purely observational — it reads the emergent state and names it.
/// The regime label has no effect on the simulation; it exists for display
/// and checkpoint readability only.
pub fn classify_weather_regime(
    cloud_cover: f32,
    precipitation_rate_mm_h: f32,
    mean_wind_speed: f32,
) -> WeatherRegime {
    if precipitation_rate_mm_h > 8.0 || (precipitation_rate_mm_h > 3.0 && mean_wind_speed > 1.0) {
        WeatherRegime::Storm
    } else if precipitation_rate_mm_h > 0.5 {
        WeatherRegime::Rain
    } else if cloud_cover > 0.65 {
        WeatherRegime::Overcast
    } else if cloud_cover > 0.25 {
        WeatherRegime::PartlyCloudy
    } else {
        WeatherRegime::Clear
    }
}

impl TerrariumWorld {
    /// Step latent guild banks (microbial/nitrifier/denitrifier secondary genotype evolution).
    /// Stub: will be connected when guild_latent infrastructure is fully wired.
    pub(super) fn step_latent_strain_banks(&mut self, _eco_dt: f32) -> Result<(), String> {
        Ok(())
    }

    /// Step emergent weather: derives cloud cover, precipitation, and temperature
    /// offset from the simulation's actual physical state fields.
    ///
    /// Physical basis:
    /// - Cloud cover ← mean atmospheric humidity (Clausius-Clapeyron: warmer air
    ///   holds more moisture before saturation; clouds form when humidity approaches
    ///   saturation). Soil evaporation feeds atmospheric humidity.
    /// - Precipitation ← cloud saturation excess via Hill kinetics (rain begins
    ///   when cloud cover exceeds condensation threshold; intensity scales with
    ///   the humidity surplus). Precipitation removes moisture from the atmosphere.
    /// - Temperature offset ← cloud albedo cooling + evaporative cooling from
    ///   soil moisture + latent heat release from precipitation.
    /// - Wind amplification ← pressure gradient proxy from spatial temperature
    ///   variance (large temperature differences drive convection and wind).
    /// - WeatherRegime is now a diagnostic label derived FROM the continuous
    ///   state, not a driver of it.
    pub(super) fn step_weather(&mut self, dt: f32) {
        let plane = self.config.width * self.config.height;
        if plane == 0 {
            return;
        }
        let inv_plane = 1.0 / plane as f32;

        // ---- Sample mean fields from the actual simulation state ----

        let mut mean_humidity = 0.0f32;
        let mut mean_temp = 0.0f32;
        let mut temp_variance_accum = 0.0f32;
        let mut mean_soil_moisture = 0.0f32;
        let mut mean_wind_speed_sq = 0.0f32;

        for i in 0..plane {
            let h = self.humidity.get(i).copied().unwrap_or(0.5);
            let t = self.temperature.get(i).copied().unwrap_or(22.0);
            let m = self.moisture.get(i).copied().unwrap_or(0.3);
            let wx = self.wind_x.get(i).copied().unwrap_or(0.0);
            let wy = self.wind_y.get(i).copied().unwrap_or(0.0);
            mean_humidity += h;
            mean_temp += t;
            mean_soil_moisture += m;
            mean_wind_speed_sq += wx * wx + wy * wy;
        }
        mean_humidity *= inv_plane;
        mean_temp *= inv_plane;
        mean_soil_moisture *= inv_plane;
        mean_wind_speed_sq *= inv_plane;

        // Temperature variance drives convective instability
        for i in 0..plane {
            let t = self.temperature.get(i).copied().unwrap_or(22.0);
            let diff = t - mean_temp;
            temp_variance_accum += diff * diff;
        }
        let temp_variance = temp_variance_accum * inv_plane;

        // ---- Seasonal modulation (solar declination drives baseline humidity) ----
        let day_of_year = (self.time_s / 86400.0) % 365.0;
        let season_phase = (day_of_year - 172.0) / 365.0; // 0 at summer solstice
        let season_summer =
            (-(season_phase * std::f32::consts::TAU).cos() * 0.5 + 0.5).clamp(0.0, 1.0);

        // ---- Cloud cover: emerges from humidity approaching saturation ----
        //
        // Clausius-Clapeyron: saturation humidity ∝ exp(17.27T / (T+237.3)).
        // We use a normalized saturation fraction: how close the air is to holding
        // all the moisture it can. Clouds form via Hill kinetics on this fraction.
        //
        // Soil evaporation contributes: wet soil → more atmospheric humidity source.
        // Evaporation contribution from Priestley-Taylor (1972) fraction
        let wt = super::emergent_rates::weather_thermodynamics();
        let sat_humidity_normalized = mean_temp.max(0.0) / 40.0; // rough [0,1]
        let evaporation_contribution = hill(mean_soil_moisture, 0.4, 1.5) * wt.max_evaporation_contribution;
        let effective_humidity = (mean_humidity + evaporation_contribution).clamp(0.0, 1.0);

        // Cloud formation threshold: warmer air needs more humidity to saturate
        // Cooler air saturates easier (lower threshold via Hill repression of temp)
        let saturation_ease = hill_repression(sat_humidity_normalized, 0.5, 2.0);
        // Cloud threshold from Clausius-Clapeyron saturation (Tetens 1930)
        let cloud_tendency = hill(effective_humidity, wt.cloud_formation_threshold + saturation_ease * 0.2, 3.0);

        // Convective instability adds to cloud formation
        let convective_lift = hill(temp_variance, 4.0, 2.0) * 0.2;

        // Winter bias: shorter days → less solar heating → easier cloud retention
        let winter_cloud_bias = (1.0 - season_summer) * 0.1;

        let target_cloud = (cloud_tendency + convective_lift + winter_cloud_bias).clamp(0.0, 1.0);

        // ---- Precipitation: cloud moisture exceeds condensation threshold ----
        //
        // Rain requires BOTH high cloud cover AND sufficient humidity surplus.
        // Intensity via Hill on cloud saturation excess.
        let precip_readiness = hill(target_cloud, 0.6, 4.0)
            * hill(effective_humidity, 0.5, 3.0);

        // Wind speed increases precipitation intensity (orographic effect proxy)
        let mean_wind_speed = mean_wind_speed_sq.sqrt();
        let wind_precip_boost = hill(mean_wind_speed, 0.5, 2.0) * 0.3;

        // Base precipitation intensity scales with humidity surplus above threshold
        let humidity_surplus = (effective_humidity - 0.5).max(0.0) * 2.0; // [0, 1]
        let target_precip = precip_readiness
            * (1.0 + wind_precip_boost)
            * humidity_surplus
            * wt.max_precipitation_mm_h; // CC-scaled max (Trenberth+ 2003)

        // ---- Temperature offset: cloud albedo + evaporative cooling ----
        //
        // Clouds reflect sunlight → cooling (proportional to cloud cover).
        // Soil evaporation → latent heat absorption → cooling.
        // Precipitation → latent heat release → slight warming of air.
        // Cloud albedo cooling: Stefan-Boltzmann (Hartmann 1994)
        let cloud_cooling = -target_cloud * wt.max_cloud_cooling_c;
        // Evaporative cooling: latent heat (Henderson-Sellers 1984)
        let evaporative_cooling = -hill(mean_soil_moisture, 0.3, 2.0) * wt.max_evaporative_cooling_c;
        // Precipitation warming: latent heat release from condensation (Trenberth+ 2009)
        let precip_latent_warming = hill(target_precip / wt.max_precipitation_mm_h, 0.3, 2.0) * wt.max_precip_warming_c;
        let summer_warming = season_summer * 1.5; // seasonal solar heating offset

        let target_temp_offset =
            cloud_cooling + evaporative_cooling + precip_latent_warming + summer_warming;

        // ---- Smooth relaxation toward emergent targets ----
        // Exponential relaxation prevents instantaneous jumps
        // Exponential relaxation: terrarium-scale BL turnover (~3s)
        let relax = (-wt.relaxation_rate * dt).exp();
        self.weather.cloud_cover =
            target_cloud + (self.weather.cloud_cover - target_cloud) * relax;
        self.weather.precipitation_rate_mm_h =
            target_precip + (self.weather.precipitation_rate_mm_h - target_precip) * relax;
        self.weather.temperature_offset_c =
            target_temp_offset + (self.weather.temperature_offset_c - target_temp_offset) * relax;

        // Precipitation removes atmospheric moisture (negative feedback)
        if self.weather.precipitation_rate_mm_h > 0.1 {
            let moisture_removal = self.weather.precipitation_rate_mm_h * dt / 3600.0 * 0.001;
            for h in self.humidity.iter_mut() {
                *h = (*h - moisture_removal).max(0.05);
            }
        }

        // Clamp
        self.weather.cloud_cover = self.weather.cloud_cover.clamp(0.0, 1.0);
        self.weather.precipitation_rate_mm_h = self.weather.precipitation_rate_mm_h.max(0.0);

        // ---- Regime is now a DIAGNOSTIC label, not a driver ----
        self.weather.regime = classify_weather_regime(
            self.weather.cloud_cover,
            self.weather.precipitation_rate_mm_h,
            mean_wind_speed,
        );
        self.weather.regime_duration_s += dt;

        // ---- Dew/condensation: surface temp below dew point (Magnus formula) ----
        let gamma = mean_humidity.max(0.01).ln()
            + (17.27 * mean_temp) / (237.3 + mean_temp);
        let dew_point = 237.3 * gamma / (17.27 - gamma);

        // Surface temp drops below air temp at night via radiative cooling
        let solar = crate::terrarium::solar::compute_solar_state(
            self.time_s,
            self.config.latitude_deg,
            self.weather.cloud_cover,
        );
        let radiative_cooling = if solar.elevation_rad < 0.05 {
            (0.05 - solar.elevation_rad) * 40.0 // up to ~2C below air at night
        } else {
            0.0
        };
        let surface_temp =
            mean_temp - radiative_cooling + self.weather.temperature_offset_c;

        // Dew intensity: Hill activation when surface drops below dew point
        let dew_gap = (dew_point - surface_temp).max(0.0);
        self.weather.dew_intensity = hill(dew_gap, 0.5, 2.0);

        // Dew adds moisture to soil surface layer 0.
        // Rate derived from Stefan-Boltzmann radiative cooling → condensation
        // via emergent_dew_rate() (Henderson-Sellers 1984 latent heat).
        if self.weather.dew_intensity > 0.01 {
            let dew_rate = super::emergent_rates::emergent_dew_rate(mean_temp);
            let dew_moisture = self.weather.dew_intensity * dt * dew_rate;
            let plane_size = self.config.width * self.config.height;
            for layer in self.soil_layer_moisture.iter_mut().take(plane_size) {
                layer[0] = (layer[0] + dew_moisture).min(1.0);
            }
        }
    }

    /// Step lightning: electrical discharge during storms with nitrogen fixation.
    ///
    /// Strike probability scales with precipitation intensity and cloud cover
    /// (Hill kinetics). Each strike deposits ammonium and nitrate to the soil
    /// at the strike location, simulating atmospheric N₂ fixation via the
    /// Zel'dovich mechanism (N₂ + O₂ → 2NO → NO₂ → HNO₃).
    ///
    /// Flash rate derivation:
    ///   Global average: ~45 flashes/s over ~5.1×10⁸ km² (Christian+ 2003).
    ///   Per-storm cell (~100 km²): ~45/5.1e6 ≈ 8.8e-6 flashes/s baseline.
    ///   Intense storms: ~6 flashes/min/cell (Zipser+ 2006) ≈ 0.1/s.
    ///   Our terrarium is ~1e-6 km²; scaling by Hill(precip)*Hill(cloud)
    ///   gives a peak rate per dt that maps to ~1 strike per 5-10 min of
    ///   sim time at time_warp 900, consistent with observation.
    ///
    /// Nitrogen fixation: ~5 kg NO₂ per flash (Schumann+ 2007), deposited as
    ///   ~40% NH₄⁺, 60% NO₃⁻ after Zel'dovich + aqueous-phase conversion.
    ///   Scaled to terrarium mass units via cell area.
    pub(super) fn step_lightning(&mut self, dt: f32) {
        self.weather.lightning_flash = false;

        // Lightning only during Storm regime
        if !matches!(self.weather.regime, WeatherRegime::Storm) {
            return;
        }

        // Flash rate derived from Wilson charging theory: charge buildup rate
        // proportional to updraft intensity (∝ precipitation × cloud_cover²),
        // discharge when accumulated charge exceeds breakdown threshold.
        // emergent_lightning_rate() returns rate already gated by precip × cloud.
        let flash_rate_per_s = super::emergent_rates::emergent_lightning_rate(
            self.weather.precipitation_rate_mm_h,
            self.weather.cloud_cover,
        );
        let strike_prob = hill(self.weather.precipitation_rate_mm_h, 8.0, 3.0)
            * hill(self.weather.cloud_cover, 0.8, 4.0)
            * dt
            * flash_rate_per_s;

        // Deterministic pseudo-random from simulation time (GPU-style hash)
        let hash = ((self.time_s * 17.31).sin() * 43758.5453).fract().abs();
        if hash > strike_prob {
            return;
        }

        // Strike location: pseudo-random cell (GPU-style hash)
        let w = self.config.width;
        let h = self.config.height;
        if w == 0 || h == 0 {
            return;
        }
        let hash2 = ((self.time_s * 31.17).cos() * 28461.7).fract().abs();
        let strike_x = (hash2 * w as f32) as usize % w;
        let hash3 = ((self.time_s * 53.91).sin() * 91827.3).fract().abs();
        let strike_y = (hash3 * h as f32) as usize % h;

        // Nitrogen fixation from Zel'dovich kinetics at ~30,000 K channel temp.
        // Yield derived from Schumann+ 2007 (~5 kg NO₂/flash), scaled to
        // substrate units via emergent_lightning_fixation().
        let fixation = super::emergent_rates::emergent_lightning_fixation(
            self.weather.precipitation_rate_mm_h,
        );
        self.substrate.deposit_patch_species(
            TerrariumSpecies::Ammonium,
            strike_x,
            strike_y,
            0,
            1,
            fixation * 0.4,
        );
        self.substrate.deposit_patch_species(
            TerrariumSpecies::Nitrate,
            strike_x,
            strike_y,
            0,
            1,
            fixation * 0.6,
        );

        self.weather.lightning_flash = true;
        self.weather.lightning_x = strike_x as f32;
        self.weather.lightning_y = strike_y as f32;

        self.ecology_events
            .push(EcologyTelemetryEvent::LightningStrike {
                x: strike_x as f32,
                y: strike_y as f32,
                ammonium_deposited: fixation * 0.4,
                nitrate_deposited: fixation * 0.6,
            });
    }

    /// Step wind turbulence: inject stochastic fluctuations into wind field.
    ///
    /// Uses 3-octave sinusoidal noise with deterministic phase offsets for
    /// spatial coherence. Turbulence intensity scales with local mean speed.
    pub(super) fn step_wind_turbulence(&mut self, dt: f32) {
        let base_speed = self.config.base_wind_speed_mm_s;
        let turb = self.config.turbulence_intensity;
        let t = self.time_s;

        let width = self.config.width;
        let height = self.config.height;
        let depth = self.config.depth.max(1);

        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    let idx = (z * height + y) * width + x;
                    // Deterministic phase from spatial position
                    let phase_seed = (x as f32 * 0.73 + y as f32 * 1.17 + z as f32 * 2.31).sin();

                    // 3-octave noise for turbulent fluctuation
                    let octave1 = (t * 0.5 + phase_seed * 6.28).sin();
                    let octave2 = (t * 1.3 + phase_seed * 12.56 + 1.7).sin() * 0.5;
                    let octave3 = (t * 3.7 + phase_seed * 25.12 + 3.1).sin() * 0.25;
                    let fluctuation = (octave1 + octave2 + octave3) / 1.75;

                    // Height-dependent speed profile (logarithmic wind profile)
                    let height_factor = 0.5 + 0.5 * (z as f32 + 1.0) / depth as f32;

                    // Base wind direction drifts slowly
                    let dir_angle = t * 0.02 + phase_seed * 0.5;

                    let local_speed = base_speed * height_factor * (1.0 + turb * fluctuation);
                    self.wind_x[idx] += (dir_angle.cos() * local_speed - self.wind_x[idx]) * dt.min(0.5);
                    self.wind_y[idx] += (dir_angle.sin() * local_speed - self.wind_y[idx]) * dt.min(0.5);
                    // Vertical component is small (turbulent uplift only)
                    self.wind_z[idx] += (turb * fluctuation * base_speed * 0.1 - self.wind_z[idx]) * dt.min(0.5);
                }
            }
        }
    }

    /// Step biomechanics: compute wind drag and stem damage for each plant.
    pub(super) fn step_biomechanics(&mut self, dt: f32) {
        let width = self.config.width;
        let height = self.config.height;
        let depth = self.config.depth.max(1);

        for plant in self.plants.iter_mut() {
            let x = plant.x;
            let y = plant.y;
            let plant_height = plant.physiology.height_mm();
            if plant_height <= 0.0 {
                continue;
            }

            // Sample wind at canopy height
            let canopy_z = ((plant_height / 3.0).round() as usize).clamp(0, depth - 1);
            let idx = (canopy_z * height + y) * width + x;
            let wx = self.wind_x.get(idx).copied().unwrap_or(0.0);
            let wy = self.wind_y.get(idx).copied().unwrap_or(0.0);
            let wind_speed = (wx * wx + wy * wy).sqrt();

            // Look up species-specific parameters
            let profile = crate::botany::species::species_profile_by_taxonomy(plant.genome.taxonomy_id);
            let (wood_dens, drag_cd, stem_frac) = profile
                .map(|p| (p.wood_density, p.drag_coefficient, p.stem_diameter_fraction))
                .unwrap_or((300.0, 0.5, 0.08));

            let canopy_r = plant.genome.canopy_radius_mm.max(0.1);
            let stem_d = plant_height * stem_frac;

            // Compute forces and stress
            let force = wind_drag_force(wind_speed, canopy_r, plant_height, drag_cd);
            let stress = stem_bending_stress(force, plant_height, stem_d);
            let yield_strength = stem_yield_strength(wood_dens);
            let stress_ratio = if yield_strength > 0.0 { stress / yield_strength } else { 0.0 };

            // Update deflection
            let deflect = wind_deflection_angle(force, plant_height, stem_d, wood_dens);
            plant.morphology.wind_deflection = deflect;

            // Damage accumulation (Hill kinetics -- no thresholds)
            // Damage increases when stress exceeds yield: hill(ratio, 0.7, 4) => steep above 70%
            let damage_rate = hill(stress_ratio, 0.7, 4.0);
            // Recovery when stress is below yield: hill_repression(ratio, 0.3, 2)
            let recovery_rate = hill_repression(stress_ratio, 0.3, 2.0);
            let damage = plant.morphology.mechanical_damage;
            let new_damage = damage + (damage_rate * 0.1 - recovery_rate * 0.02 * damage) * dt;
            plant.morphology.mechanical_damage = new_damage.clamp(0.0, 1.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wind_drag_force_zero_speed() {
        assert_eq!(wind_drag_force(0.0, 4.0, 10.0, 0.5), 0.0);
    }

    #[test]
    fn wind_drag_force_quadratic() {
        let f1 = wind_drag_force(1.0, 4.0, 10.0, 0.5);
        let f2 = wind_drag_force(2.0, 4.0, 10.0, 0.5);
        // Force scales with v^2, so f2 should be ~4x f1
        let ratio = f2 / f1;
        assert!((ratio - 4.0).abs() < 0.1, "Force should scale quadratically, ratio={ratio}");
    }

    #[test]
    fn stem_bending_stress_thin_stem() {
        let thick = stem_bending_stress(0.001, 10.0, 2.0);
        let thin = stem_bending_stress(0.001, 10.0, 0.5);
        assert!(thin > thick * 10.0, "Thin stem should have much higher stress: thin={thin}, thick={thick}");
    }

    #[test]
    fn stem_yield_strength_wood_density() {
        let softwood = stem_yield_strength(200.0);
        let hardwood = stem_yield_strength(650.0);
        assert!(hardwood > softwood * 3.0, "Hardwood should be much stronger: hard={hardwood}, soft={softwood}");
    }

    #[test]
    fn wind_deflection_small_force() {
        // Very small force on a sturdy stem should give small deflection
        let angle = wind_deflection_angle(0.000001, 10.0, 2.0, 500.0);
        assert!(angle < 0.5, "Small force should give small deflection: {angle} rad");
        assert!(angle > 0.0, "Non-zero force should give non-zero deflection");
    }

    #[test]
    fn damage_accumulation_below_yield() {
        // At 30% of yield strength, damage rate should be very low
        let rate = hill(0.3, 0.7, 4.0);
        assert!(rate < 0.05, "Well below yield, damage rate should be very low: {rate}");
    }

    #[test]
    fn damage_accumulation_above_yield() {
        // At 150% of yield, damage should accumulate quickly
        let rate = hill(1.5, 0.7, 4.0);
        assert!(rate > 0.9, "Above yield, damage rate should be high: {rate}");
    }

    #[test]
    fn damage_recovery_low_stress() {
        let recovery = hill_repression(0.1, 0.3, 2.0);
        assert!(recovery > 0.8, "Low stress should allow strong recovery: {recovery}");
    }

    #[test]
    fn damage_does_not_exceed_one() {
        // Simulate many steps of high damage
        let mut damage = 0.0f32;
        for _ in 0..1000 {
            let damage_rate = hill(2.0, 0.7, 4.0); // way above yield
            let recovery_rate = hill_repression(2.0, 0.3, 2.0);
            damage += (damage_rate * 0.1 - recovery_rate * 0.02 * damage) * 0.05;
            damage = damage.clamp(0.0, 1.0);
        }
        assert!(damage <= 1.0, "Damage must not exceed 1.0: {damage}");
        assert!(damage > 0.9, "Sustained high wind should cause near-max damage: {damage}");
    }

    #[test]
    fn mechanical_viability_modifier_healthy() {
        let v = mechanical_viability_modifier(0.0);
        assert!((v - 1.0).abs() < 0.01, "No damage = full viability: {v}");
    }

    #[test]
    fn mechanical_viability_modifier_damaged() {
        let v = mechanical_viability_modifier(0.8);
        assert!(v < 0.3, "High damage should reduce viability: {v}");
    }

    #[test]
    fn pollen_wind_boost_concept() {
        // Verify that wind direction alignment gives a positive boost
        let wind_x = 1.0f32;
        let wind_y = 0.0f32;
        let wind_speed = (wind_x * wind_x + wind_y * wind_y).sqrt();
        // Donor is downwind (positive x direction from parent)
        let dx = 3.0f32;
        let dy = 0.0f32;
        let dist = (dx * dx + dy * dy).sqrt();
        let cos_angle = (dx * wind_x + dy * wind_y) / (dist * wind_speed).max(1e-6);
        let wind_boost = cos_angle.max(0.0) * wind_speed * 2.0;
        assert!(wind_boost > 0.0, "Downwind donor should get range boost: {wind_boost}");

        // Donor is upwind (negative x direction)
        let dx2 = -3.0f32;
        let cos_angle2 = (dx2 * wind_x + 0.0 * wind_y) / (dist * wind_speed).max(1e-6);
        let wind_boost2 = cos_angle2.max(0.0) * wind_speed * 2.0;
        assert_eq!(wind_boost2, 0.0, "Upwind donor should get no boost");
    }

    // -----------------------------------------------------------------------
    // Emergent weather tests
    // -----------------------------------------------------------------------

    #[test]
    fn weather_default_clear() {
        let w = WeatherState::default();
        assert!(w.cloud_cover < 0.1, "Default weather should be clear");
        assert_eq!(w.precipitation_rate_mm_h, 0.0);
    }

    #[test]
    fn classify_low_cloud_clear() {
        let regime = classify_weather_regime(0.1, 0.0, 0.2);
        assert!(matches!(regime, WeatherRegime::Clear));
    }

    #[test]
    fn classify_moderate_cloud_partly() {
        let regime = classify_weather_regime(0.4, 0.0, 0.2);
        assert!(matches!(regime, WeatherRegime::PartlyCloudy));
    }

    #[test]
    fn classify_high_cloud_overcast() {
        let regime = classify_weather_regime(0.8, 0.0, 0.2);
        assert!(matches!(regime, WeatherRegime::Overcast));
    }

    #[test]
    fn classify_precipitation_rain() {
        let regime = classify_weather_regime(0.8, 3.0, 0.3);
        assert!(matches!(regime, WeatherRegime::Rain));
    }

    #[test]
    fn classify_heavy_precip_storm() {
        let regime = classify_weather_regime(0.95, 10.0, 1.5);
        assert!(matches!(regime, WeatherRegime::Storm));
    }

    #[test]
    fn high_humidity_produces_clouds() {
        // Cloud formation via Hill on humidity: high humidity → high cloud tendency
        let low_h = hill(0.2, 0.45, 3.0);
        let high_h = hill(0.8, 0.45, 3.0);
        assert!(
            high_h > low_h * 2.0,
            "High humidity should produce more clouds: high={high_h}, low={low_h}",
        );
    }

    #[test]
    fn cloud_cover_causes_cooling() {
        // Cloud albedo cooling is proportional to cloud cover
        let clear_offset = -0.05 * 5.0; // ~0% cloud
        let overcast_offset = -0.8 * 5.0; // ~80% cloud
        assert!(
            overcast_offset < clear_offset,
            "Overcast should be cooler: overcast={overcast_offset}, clear={clear_offset}",
        );
    }

    #[test]
    fn precipitation_requires_clouds_and_humidity() {
        // Both cloud cover AND humidity must be high for rain
        let low_cloud = hill(0.2, 0.6, 4.0) * hill(0.8, 0.5, 3.0);
        let low_humid = hill(0.8, 0.6, 4.0) * hill(0.2, 0.5, 3.0);
        let both_high = hill(0.8, 0.6, 4.0) * hill(0.8, 0.5, 3.0);
        assert!(
            both_high > low_cloud * 3.0 && both_high > low_humid * 3.0,
            "Rain needs both clouds AND humidity: both={both_high}, low_cloud={low_cloud}, low_humid={low_humid}",
        );
    }

    #[test]
    fn exponential_relaxation_converges() {
        let target = 0.8f32;
        let mut current = 0.1f32;
        for _ in 0..100 {
            let relax = (-0.3f32 * 1.0).exp();
            current = target + (current - target) * relax;
        }
        assert!(
            (current - target).abs() < 0.01,
            "Should converge to target: {current}",
        );
    }

    #[test]
    fn soil_evaporation_feeds_clouds() {
        // Wet soil contributes to atmospheric humidity → more clouds
        let dry_evap = hill(0.1, 0.4, 1.5) * 0.15;
        let wet_evap = hill(0.8, 0.4, 1.5) * 0.15;
        assert!(
            wet_evap > dry_evap * 2.0,
            "Wet soil should evaporate more: wet={wet_evap}, dry={dry_evap}",
        );
    }

    #[test]
    fn precipitation_removes_humidity() {
        // Negative feedback: rain should reduce atmospheric humidity
        let precip_rate = 5.0f32; // mm/h
        let dt = 1.0f32;
        let moisture_removal = precip_rate * dt / 3600.0 * 0.001;
        assert!(
            moisture_removal > 0.0,
            "Precipitation should remove atmospheric moisture",
        );
    }

    #[test]
    fn lightning_only_during_storm() {
        let mut world = TerrariumWorld::demo(42, false).expect("demo world");
        // Set to Clear regime — no lightning should fire
        world.weather.regime = WeatherRegime::Clear;
        world.weather.cloud_cover = 0.1;
        world.weather.precipitation_rate_mm_h = 0.0;
        for _ in 0..100 {
            world.step_lightning(1.0);
        }
        assert!(
            !world.weather.lightning_flash,
            "Lightning should not fire in Clear regime"
        );
        assert!(
            world
                .ecology_events
                .iter()
                .all(|e| !matches!(e, EcologyTelemetryEvent::LightningStrike { .. })),
            "No LightningStrike events in clear weather"
        );
    }

    #[test]
    fn lightning_deposits_nitrogen() {
        let mut world = TerrariumWorld::demo(42, false).expect("demo world");
        // Force Storm regime with intense precipitation and cloud cover
        world.weather.regime = WeatherRegime::Storm;
        world.weather.cloud_cover = 0.95;
        world.weather.precipitation_rate_mm_h = 15.0;

        let nh4_before: f32 = world
            .substrate
            .species_field(TerrariumSpecies::Ammonium)
            .iter()
            .sum();
        let no3_before: f32 = world
            .substrate
            .species_field(TerrariumSpecies::Nitrate)
            .iter()
            .sum();

        // Run many frames to ensure at least one strike fires
        for i in 0..500 {
            world.time_s = 1000.0 + i as f32 * 0.37;
            world.step_lightning(1.0);
        }

        let nh4_after: f32 = world
            .substrate
            .species_field(TerrariumSpecies::Ammonium)
            .iter()
            .sum();
        let no3_after: f32 = world
            .substrate
            .species_field(TerrariumSpecies::Nitrate)
            .iter()
            .sum();

        assert!(
            nh4_after > nh4_before || no3_after > no3_before,
            "Lightning should deposit nitrogen: NH4 {nh4_before}->{nh4_after}, NO3 {no3_before}->{no3_after}"
        );
    }

    #[test]
    fn lightning_emits_ecology_event() {
        let mut world = TerrariumWorld::demo(42, false).expect("demo world");
        world.weather.regime = WeatherRegime::Storm;
        world.weather.cloud_cover = 0.95;
        world.weather.precipitation_rate_mm_h = 15.0;

        for i in 0..500 {
            world.time_s = 2000.0 + i as f32 * 0.41;
            world.step_lightning(1.0);
        }

        let strike_count = world
            .ecology_events
            .iter()
            .filter(|e| matches!(e, EcologyTelemetryEvent::LightningStrike { .. }))
            .count();
        assert!(
            strike_count > 0,
            "At least one LightningStrike event should fire during intense storm"
        );
    }

    #[test]
    fn dew_forms_when_surface_below_dew_point() {
        // Magnus dew point formula: gamma = ln(RH) + 17.27*T/(237.3+T)
        // dew_point = 237.3 * gamma / (17.27 - gamma)
        // With T=15, RH=0.95: gamma ~ ln(0.95)+17.27*15/252.3 ≈ -0.051+1.028=0.977
        // dew_point ≈ 237.3*0.977/(17.27-0.977) ≈ 231.8/16.29 ≈ 14.2C
        // At night (solar elev < 0.05): radiative_cooling up to ~2C → surface ~13C
        // dew_gap = 14.2 - 13 = 1.2 → hill(1.2, 0.5, 2.0) > 0
        let gap = 1.2f32;
        let intensity = hill(gap, 0.5, 2.0);
        assert!(
            intensity > 0.5,
            "Dew should form when surface is 1.2C below dew point: {intensity}"
        );

        // No dew when surface is well above dew point
        let gap_warm = 0.0f32;
        let intensity_warm = hill(gap_warm, 0.5, 2.0);
        assert!(
            intensity_warm < 0.01,
            "No dew when surface temp equals or exceeds dew point: {intensity_warm}"
        );
    }

    #[test]
    fn dew_adds_surface_moisture() {
        let mut world = TerrariumWorld::demo(42, false).expect("demo world");
        // Set conditions for dew: high humidity, night time (low solar elevation)
        for h in world.humidity.iter_mut() {
            *h = 0.95;
        }
        for t in world.temperature.iter_mut() {
            *t = 12.0; // cool night
        }
        world.time_s = 3.0 * 3600.0; // 3 AM
        world.weather.cloud_cover = 0.1;
        world.weather.temperature_offset_c = -1.0;

        let plane = world.config.width * world.config.height;
        let m_before: f32 = world
            .soil_layer_moisture
            .iter()
            .take(plane)
            .map(|layer| layer[0])
            .sum();

        // Step weather to trigger dew
        for _ in 0..10 {
            world.step_weather(1.0);
        }

        let m_after: f32 = world
            .soil_layer_moisture
            .iter()
            .take(plane)
            .map(|layer| layer[0])
            .sum();

        assert!(
            m_after >= m_before,
            "Dew should add surface moisture: {m_before} -> {m_after}"
        );
    }
}
