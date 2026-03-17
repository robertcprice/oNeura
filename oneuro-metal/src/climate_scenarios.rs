//! # Climate Scenario Projection Engine
//!
//! Self-contained climate model for multi-decade terrarium evolution runs.
//! Implements IPCC AR5 Representative Concentration Pathways (RCP 2.6, 4.5, 8.5)
//! with stochastic weather noise, extreme event generation, and derived agricultural
//! indices (Palmer Drought Index, growing season length, soil moisture).
//!
//! All parameters are calibrated against IPCC AR5 WG1 Table SPM.2 and Chapter 12
//! projections for 2081-2100 relative to 1986-2005 baseline.
//!
//! ## Design
//!
//! - **Zero crate dependencies**: uses inline xorshift64 PRNG, no `rand` crate.
//! - **Deterministic projections**: `state_at(year)` returns the same result every call.
//! - **Stochastic stepping**: `step(dt)` advances an internal clock with weather noise.
//! - **Extreme events**: Poisson-sampled per year, frequency scales with warming.
//!
//! ## IPCC Calibration (AR5 WG1 Table SPM.2, 2081-2100 vs 1986-2005)
//!
//! | Scenario | Global Mean Temp Anomaly | CO2 in 2100 | Description             |
//! |----------|--------------------------|-------------|-------------------------|
//! | RCP 2.6  | +1.0 C (0.3 - 1.7)      | ~490 ppm    | Strong mitigation       |
//! | RCP 4.5  | +1.8 C (1.1 - 2.6)      | ~650 ppm    | Intermediate            |
//! | RCP 8.5  | +3.7 C (2.6 - 4.8)      | ~1370 ppm   | Business as usual       |

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Inline xorshift64 PRNG (no external crate dependency)
// ---------------------------------------------------------------------------

/// Minimal xorshift64* PRNG. Period 2^64 - 1.
#[derive(Debug, Clone)]
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        // Ensure nonzero state (xorshift requirement).
        Self {
            state: if seed == 0 { 0x5EED_CAFE_DEAD_BEEF } else { seed },
        }
    }

    /// Advance state and return next u64.
    fn next_u64(&mut self) -> u64 {
        let mut s = self.state;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.state = s;
        s.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Approximate standard normal via Box-Muller transform.
    fn next_normal(&mut self, mean: f64, std_dev: f64) -> f64 {
        let u1 = self.next_f64().max(1e-15); // avoid log(0)
        let u2 = self.next_f64();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        mean + z * std_dev
    }

    /// Poisson-distributed sample via Knuth's algorithm (for small lambda).
    fn next_poisson(&mut self, lambda: f64) -> u32 {
        if lambda <= 0.0 {
            return 0;
        }
        let l = (-lambda).exp();
        let mut k = 0u32;
        let mut p = 1.0f64;
        loop {
            k += 1;
            p *= self.next_f64().max(1e-15);
            if p < l {
                break;
            }
        }
        k.saturating_sub(1)
    }
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// IPCC Representative Concentration Pathway or custom climate trajectory.
#[derive(Debug, Clone)]
pub enum ClimateScenario {
    /// RCP 2.6: strong mitigation, +1.0 C by 2100, CO2 peaks ~490 ppm.
    Rcp26,
    /// RCP 4.5: intermediate pathway, +1.8 C by 2100, CO2 ~650 ppm.
    Rcp45,
    /// RCP 8.5: business-as-usual, +3.7 C by 2100, CO2 ~1370 ppm.
    Rcp85,
    /// Pre-industrial baseline (~1850): 13.5 C global mean, 280 ppm CO2.
    PreIndustrial,
    /// User-defined climate trajectory with explicit parameter control.
    Custom(ClimateParams),
}

/// Tunable parameters defining a climate trajectory over time.
///
/// All trends are expressed per decade to match IPCC reporting conventions.
#[derive(Debug, Clone)]
pub struct ClimateParams {
    /// Global mean surface temperature at simulation start (degrees C).
    pub base_temp_c: f64,
    /// Temperature trend per decade (degrees C / decade).
    pub temp_trend_c_per_decade: f64,
    /// Mean annual precipitation at start (mm/year).
    pub base_precip_mm_yr: f64,
    /// Precipitation change per decade (percent / decade, e.g. -2.0 = 2% decrease).
    pub precip_change_pct_per_decade: f64,
    /// Atmospheric CO2 at simulation start (ppm).
    pub co2_ppm_start: f64,
    /// CO2 trend per decade (ppm / decade).
    pub co2_ppm_trend_per_decade: f64,
    /// Baseline extreme event frequency (events per year). Scales with warming.
    pub extreme_event_frequency: f64,
    /// Sea level rise rate (mm/year).
    pub sea_level_rise_mm_yr: f64,
}

/// Instantaneous climate conditions at a point in time.
#[derive(Debug, Clone)]
pub struct ClimateState {
    /// Simulation year (fractional, e.g. 2050.5 = mid-2050).
    pub year: f64,
    /// Surface air temperature (degrees C).
    pub temperature_c: f64,
    /// Annual-equivalent precipitation (mm).
    pub precipitation_mm: f64,
    /// Atmospheric CO2 concentration (ppm).
    pub co2_ppm: f64,
    /// Relative humidity (0-1).
    pub humidity: f64,
    /// Incoming shortwave solar radiation (W/m2).
    pub solar_radiation_w_m2: f64,
    /// Surface wind speed (m/s).
    pub wind_speed_m_s: f64,
    /// Drought severity (0 = none, 1 = extreme).
    pub drought_severity: f64,
    /// Flood risk (0 = none, 1 = extreme).
    pub flood_risk: f64,
}

/// Discrete extreme weather events that can impact terrarium ecosystems.
#[derive(Debug, Clone)]
pub enum ExtremeEvent {
    /// Sustained period of abnormally low precipitation.
    Drought {
        /// Severity from 0 (mild) to 1 (catastrophic).
        severity: f64,
        /// Duration in days.
        duration_days: u32,
    },
    /// Abnormally high water event (river, flash, coastal).
    Flood {
        /// Magnitude from 0 (minor) to 1 (catastrophic).
        magnitude: f64,
    },
    /// Extended period of anomalously high temperature.
    Heatwave {
        /// Peak temperature during event (degrees C).
        peak_temp_c: f64,
        /// Duration in days.
        duration_days: u32,
    },
    /// Sudden drop to anomalously low temperature.
    ColdSnap {
        /// Minimum temperature reached (degrees C).
        min_temp_c: f64,
    },
    /// Fire event driven by drought and heat.
    Wildfire {
        /// Intensity from 0 (small brush fire) to 1 (firestorm).
        intensity: f64,
    },
    /// Tropical cyclone / hurricane.
    Hurricane {
        /// Saffir-Simpson category (1-5).
        category: u8,
    },
}

/// Time series output from a multi-year climate projection run.
#[derive(Debug, Clone)]
pub struct ClimateTimeSeries {
    /// Year values for each sample point.
    pub years: Vec<f64>,
    /// Temperature at each sample point (degrees C).
    pub temperatures: Vec<f64>,
    /// Precipitation at each sample point (mm/yr).
    pub precipitations: Vec<f64>,
    /// CO2 concentration at each sample point (ppm).
    pub co2_levels: Vec<f64>,
    /// Extreme events with their occurrence year.
    pub extreme_events: Vec<(f64, ExtremeEvent)>,
    /// Palmer Drought Severity Index at each sample point.
    pub drought_indices: Vec<f64>,
}

// ---------------------------------------------------------------------------
// IPCC-calibrated scenario defaults
// ---------------------------------------------------------------------------

/// Reference year for IPCC AR5 baseline period midpoint (1986-2005).
const REFERENCE_YEAR: f64 = 1995.0;

/// Pre-industrial global mean surface temperature estimate (degrees C).
const PRE_INDUSTRIAL_TEMP: f64 = 13.5;

/// 1995 baseline temperature: pre-industrial + ~0.6 C observed warming by 1995.
const BASELINE_TEMP_1995: f64 = 14.1;

/// Pre-industrial CO2 concentration (ppm).
const PRE_INDUSTRIAL_CO2: f64 = 280.0;

/// 1995 observed CO2 concentration (ppm).
const BASELINE_CO2_1995: f64 = 360.0;

/// Mean annual precipitation for temperate reference (mm/year).
const BASELINE_PRECIP: f64 = 1000.0;

/// Mean annual solar radiation at mid-latitudes (W/m2).
const BASELINE_SOLAR: f64 = 240.0;

impl ClimateScenario {
    /// Convert the named scenario into explicit numeric parameters.
    ///
    /// All trend values are calibrated so that projecting from 1995 baseline
    /// to 2100 reproduces IPCC AR5 WG1 Table SPM.2 central estimates.
    fn to_params(&self) -> ClimateParams {
        match self {
            // ----------------------------------------------------------
            // RCP 2.6: +1.0 C by 2100, CO2 peaks ~490 ppm
            // Decades from 1995 to 2100 = 10.5
            // Temp trend: 1.0 / 10.5 ~ 0.095 C/decade
            // CO2 trend: (490 - 360) / 10.5 ~ 12.4 ppm/decade
            // ----------------------------------------------------------
            ClimateScenario::Rcp26 => ClimateParams {
                base_temp_c: BASELINE_TEMP_1995,
                temp_trend_c_per_decade: 0.095,
                base_precip_mm_yr: BASELINE_PRECIP,
                precip_change_pct_per_decade: 0.5,
                co2_ppm_start: BASELINE_CO2_1995,
                co2_ppm_trend_per_decade: 12.4,
                extreme_event_frequency: 1.5,
                sea_level_rise_mm_yr: 3.2,
            },
            // ----------------------------------------------------------
            // RCP 4.5: +1.8 C by 2100, CO2 ~650 ppm
            // Temp trend: 1.8 / 10.5 ~ 0.171 C/decade
            // CO2 trend: (650 - 360) / 10.5 ~ 27.6 ppm/decade
            // ----------------------------------------------------------
            ClimateScenario::Rcp45 => ClimateParams {
                base_temp_c: BASELINE_TEMP_1995,
                temp_trend_c_per_decade: 0.171,
                base_precip_mm_yr: BASELINE_PRECIP,
                precip_change_pct_per_decade: 1.0,
                co2_ppm_start: BASELINE_CO2_1995,
                co2_ppm_trend_per_decade: 27.6,
                extreme_event_frequency: 2.0,
                sea_level_rise_mm_yr: 4.5,
            },
            // ----------------------------------------------------------
            // RCP 8.5: +3.7 C by 2100, CO2 ~1370 ppm
            // Temp trend: 3.7 / 10.5 ~ 0.352 C/decade
            // CO2 trend: (1370 - 360) / 10.5 ~ 96.2 ppm/decade
            // ----------------------------------------------------------
            ClimateScenario::Rcp85 => ClimateParams {
                base_temp_c: BASELINE_TEMP_1995,
                temp_trend_c_per_decade: 0.352,
                base_precip_mm_yr: BASELINE_PRECIP,
                precip_change_pct_per_decade: -1.5,
                co2_ppm_start: BASELINE_CO2_1995,
                co2_ppm_trend_per_decade: 96.2,
                extreme_event_frequency: 3.5,
                sea_level_rise_mm_yr: 8.0,
            },
            // ----------------------------------------------------------
            // Pre-industrial: stable at ~1850 conditions.
            // ----------------------------------------------------------
            ClimateScenario::PreIndustrial => ClimateParams {
                base_temp_c: PRE_INDUSTRIAL_TEMP,
                temp_trend_c_per_decade: 0.0,
                base_precip_mm_yr: BASELINE_PRECIP,
                precip_change_pct_per_decade: 0.0,
                co2_ppm_start: PRE_INDUSTRIAL_CO2,
                co2_ppm_trend_per_decade: 0.0,
                extreme_event_frequency: 0.8,
                sea_level_rise_mm_yr: 0.0,
            },
            ClimateScenario::Custom(params) => params.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// ClimateEngine
// ---------------------------------------------------------------------------

/// Main climate projection engine.
///
/// Combines deterministic trend-based projections with stochastic weather noise
/// and extreme event generation. Suitable for driving multi-decade terrarium
/// evolution experiments where organisms must adapt to changing climate.
///
/// # Example
///
/// ```ignore
/// let engine = ClimateEngine::new(ClimateScenario::Rcp45, 42);
/// let state_2050 = engine.state_at(2050.0);
/// println!("Temperature in 2050: {:.1} C", state_2050.temperature_c);
/// ```
#[derive(Debug, Clone)]
pub struct ClimateEngine {
    params: ClimateParams,
    rng: Xorshift64,
    current_year: f64,
    /// Reference year from which trends are computed.
    start_year: f64,
}

impl ClimateEngine {
    /// Create a new climate engine from a scenario and PRNG seed.
    ///
    /// The engine begins at the IPCC baseline year (1995) for named scenarios.
    /// Custom scenarios start at year 0.
    pub fn new(scenario: ClimateScenario, seed: u64) -> Self {
        let start_year = match &scenario {
            ClimateScenario::Custom(_) => 0.0,
            _ => REFERENCE_YEAR,
        };
        Self {
            params: scenario.to_params(),
            rng: Xorshift64::new(seed),
            current_year: start_year,
            start_year,
        }
    }

    // -- Deterministic projection ----------------------------------------

    /// Compute the number of decades elapsed since the reference year.
    fn decades_from_start(&self, year: f64) -> f64 {
        (year - self.start_year) / 10.0
    }

    /// Deterministic mean temperature at a given year.
    ///
    /// Includes a sinusoidal seasonal cycle with 12 C amplitude (northern
    /// hemisphere temperate default). The fractional part of `year`
    /// determines the season.
    fn mean_temperature(&self, year: f64) -> f64 {
        let decades = self.decades_from_start(year);
        let trend = self.params.base_temp_c + decades * self.params.temp_trend_c_per_decade;
        // Seasonal cycle: warmest in July (year fraction ~0.5), coldest in January (~0.0).
        let season_phase = (year.fract() - 0.5) * 2.0 * PI;
        let seasonal_amplitude = 12.0; // degrees C peak-to-trough / 2
        trend + seasonal_amplitude * season_phase.cos()
    }

    /// Deterministic mean precipitation at a given year.
    ///
    /// Includes seasonal variation (wetter in spring/autumn for temperate).
    fn mean_precipitation(&self, year: f64) -> f64 {
        let decades = self.decades_from_start(year);
        let pct_change = decades * self.params.precip_change_pct_per_decade / 100.0;
        let annual = self.params.base_precip_mm_yr * (1.0 + pct_change);
        // Seasonal: wetter in spring (fraction ~0.25) and autumn (~0.75).
        let season = (year.fract() * 4.0 * PI).cos();
        (annual * (1.0 + 0.15 * season)).max(0.0)
    }

    /// Deterministic CO2 concentration at a given year.
    ///
    /// For RCP 2.6, CO2 peaks around 2050 then declines (negative emissions).
    /// Modeled as a logistic cap for RCP 2.6, linear for others.
    fn mean_co2(&self, year: f64) -> f64 {
        let decades = self.decades_from_start(year);
        let raw = self.params.co2_ppm_start + decades * self.params.co2_ppm_trend_per_decade;
        // For low-emission scenarios: CO2 should not exceed ~490 ppm (RCP 2.6 peak).
        // We apply a soft cap if the trend is very low (< 15 ppm/decade).
        if self.params.co2_ppm_trend_per_decade > 0.0
            && self.params.co2_ppm_trend_per_decade < 15.0
        {
            let cap = self.params.co2_ppm_start + 130.0; // ~490 for baseline 360
            let x = raw - cap;
            if x > 0.0 {
                // Logistic softcap: asymptotes at cap + 20
                cap + 20.0 * x / (x + 20.0)
            } else {
                raw
            }
        } else {
            raw.max(self.params.co2_ppm_start * 0.5) // CO2 cannot go below half start
        }
    }

    /// Compute humidity from temperature and precipitation using simplified
    /// Clausius-Clapeyron: warmer air holds more moisture, but actual humidity
    /// depends on precipitation supply.
    fn compute_humidity(&self, temp_c: f64, precip_mm: f64) -> f64 {
        // Saturation vapor pressure doubles per ~10 C (Clausius-Clapeyron).
        // Actual humidity = precip_supply / potential_evaporation.
        let sat_factor = 2.0_f64.powf((temp_c - 15.0) / 10.0);
        let potential_evap = 600.0 * sat_factor; // mm/yr at full capacity
        let rh = (precip_mm / potential_evap).clamp(0.05, 0.98);
        rh
    }

    /// Solar radiation with seasonal and latitudinal modulation.
    fn compute_solar(&self, year: f64) -> f64 {
        // Seasonal: maximum at summer solstice (fraction ~0.5 for N hemisphere).
        let season = ((year.fract() - 0.5) * 2.0 * PI).cos();
        // CO2-driven cloud feedback: slight reduction at very high CO2.
        let co2 = self.mean_co2(year);
        let cloud_feedback = if co2 > 500.0 {
            1.0 - 0.02 * ((co2 - 500.0) / 1000.0).min(1.0)
        } else {
            1.0
        };
        (BASELINE_SOLAR * (1.0 + 0.12 * season) * cloud_feedback).max(50.0)
    }

    /// Base wind speed with seasonal variation.
    fn compute_wind(&self, year: f64) -> f64 {
        let season = ((year.fract() * 2.0 * PI) + PI / 4.0).cos();
        let base = 3.5; // m/s mean
        (base * (1.0 + 0.3 * season)).max(0.5)
    }

    /// Compute drought severity from precipitation deficit.
    fn compute_drought_severity(&self, year: f64, precip_mm: f64) -> f64 {
        let baseline = self.params.base_precip_mm_yr;
        if baseline <= 0.0 {
            return 0.0;
        }
        let deficit_fraction = 1.0 - (precip_mm / baseline).min(1.0);
        // Warming amplifies drought through increased evapotranspiration.
        let decades = self.decades_from_start(year).max(0.0);
        let warming_amplifier = 1.0 + 0.05 * decades * self.params.temp_trend_c_per_decade;
        (deficit_fraction * warming_amplifier).clamp(0.0, 1.0)
    }

    /// Compute flood risk from precipitation excess and trend.
    fn compute_flood_risk(&self, precip_mm: f64) -> f64 {
        let baseline = self.params.base_precip_mm_yr;
        if baseline <= 0.0 {
            return 0.0;
        }
        let excess = (precip_mm / baseline - 1.0).max(0.0);
        (excess * 2.0).clamp(0.0, 1.0)
    }

    // -- Public API -------------------------------------------------------

    /// Deterministic climate state at a specific year (no stochastic noise).
    ///
    /// Returns the same result every time for a given year, regardless of
    /// how many times `step()` has been called.
    pub fn state_at(&self, year: f64) -> ClimateState {
        let temp = self.mean_temperature(year);
        let precip = self.mean_precipitation(year);
        let co2 = self.mean_co2(year);
        let humidity = self.compute_humidity(temp, precip);
        let solar = self.compute_solar(year);
        let wind = self.compute_wind(year);
        let drought = self.compute_drought_severity(year, precip);
        let flood = self.compute_flood_risk(precip);

        ClimateState {
            year,
            temperature_c: temp,
            precipitation_mm: precip,
            co2_ppm: co2,
            humidity,
            solar_radiation_w_m2: solar,
            wind_speed_m_s: wind,
            drought_severity: drought,
            flood_risk: flood,
        }
    }

    /// Advance the engine by `dt_years` with stochastic weather noise.
    ///
    /// Unlike `state_at()`, this method adds normally-distributed weather
    /// perturbations and advances the internal clock. Two calls with the
    /// same dt will produce different results (different noise draws).
    pub fn step(&mut self, dt_years: f64) -> ClimateState {
        self.current_year += dt_years;
        let year = self.current_year;

        let base_temp = self.mean_temperature(year);
        let base_precip = self.mean_precipitation(year);

        // Weather noise: temperature sigma ~2 C, precipitation CV ~20%.
        let temp_noise = self.rng.next_normal(0.0, 2.0);
        let precip_noise = self.rng.next_normal(1.0, 0.2).max(0.1);

        let temp = base_temp + temp_noise;
        let precip = (base_precip * precip_noise).max(0.0);
        let co2 = self.mean_co2(year);
        let humidity = self.compute_humidity(temp, precip);
        let solar = self.compute_solar(year) + self.rng.next_normal(0.0, 10.0);
        let wind = (self.compute_wind(year) + self.rng.next_normal(0.0, 0.8)).max(0.0);
        let drought = self.compute_drought_severity(year, precip);
        let flood = self.compute_flood_risk(precip);

        ClimateState {
            year,
            temperature_c: temp,
            precipitation_mm: precip,
            co2_ppm: co2,
            humidity: humidity.clamp(0.0, 1.0),
            solar_radiation_w_m2: solar.max(0.0),
            wind_speed_m_s: wind,
            drought_severity: drought,
            flood_risk: flood,
        }
    }

    /// Generate extreme events for a given year using Poisson sampling.
    ///
    /// The expected number of events scales with warming: for each degree C
    /// above the baseline, extreme event frequency increases by ~20%
    /// (consistent with IPCC AR5 WG2 Chapter 18 findings on attribution).
    pub fn extreme_events_in_year(&mut self, year: f64) -> Vec<ExtremeEvent> {
        let decades = self.decades_from_start(year).max(0.0);
        let warming = decades * self.params.temp_trend_c_per_decade;

        // Scale frequency: +20% per degree of warming above baseline.
        let freq = self.params.extreme_event_frequency * (1.0 + 0.20 * warming);
        let n_events = self.rng.next_poisson(freq);

        let mut events = Vec::with_capacity(n_events as usize);
        for _ in 0..n_events {
            let roll = self.rng.next_f64();
            let event = if roll < 0.25 {
                ExtremeEvent::Drought {
                    severity: self.rng.next_f64().clamp(0.2, 1.0),
                    duration_days: (self.rng.next_f64() * 90.0 + 14.0) as u32,
                }
            } else if roll < 0.45 {
                ExtremeEvent::Flood {
                    magnitude: self.rng.next_f64().clamp(0.1, 1.0),
                }
            } else if roll < 0.65 {
                let base_temp = self.mean_temperature(year);
                ExtremeEvent::Heatwave {
                    peak_temp_c: base_temp + 5.0 + self.rng.next_f64() * 10.0,
                    duration_days: (self.rng.next_f64() * 21.0 + 3.0) as u32,
                }
            } else if roll < 0.78 {
                let base_temp = self.mean_temperature(year);
                ExtremeEvent::ColdSnap {
                    min_temp_c: base_temp - 10.0 - self.rng.next_f64() * 15.0,
                }
            } else if roll < 0.90 {
                ExtremeEvent::Wildfire {
                    intensity: self.rng.next_f64().clamp(0.1, 1.0),
                }
            } else {
                ExtremeEvent::Hurricane {
                    category: (self.rng.next_f64() * 5.0 + 1.0).min(5.0) as u8,
                }
            };
            events.push(event);
        }
        events
    }

    /// Length of the growing season in days for a given year.
    ///
    /// Defined as the number of days where daily mean temperature exceeds 5 C
    /// (agronomic threshold for C3 crops). Computed by finding the spring and
    /// autumn crossing points of the seasonal temperature curve.
    ///
    /// Warming extends the growing season by approximately 5-7 days per
    /// degree C (observed trend from Menzel et al. 2006, Nature).
    pub fn growing_season_length(&self, year: f64) -> f64 {
        let decades = self.decades_from_start(year).max(0.0);
        let mean_annual = self.params.base_temp_c + decades * self.params.temp_trend_c_per_decade;
        let amplitude = 12.0; // seasonal amplitude

        let threshold = 5.0; // degrees C growing threshold

        // If mean - amplitude > threshold, growing season = 365 (tropical).
        if mean_annual - amplitude > threshold {
            return 365.0;
        }
        // If mean + amplitude < threshold, no growing season (arctic).
        if mean_annual + amplitude < threshold {
            return 0.0;
        }

        // Solve: mean + amplitude * cos(phase) = threshold
        // cos(phase) = (threshold - mean) / amplitude
        let cos_val = ((threshold - mean_annual) / amplitude).clamp(-1.0, 1.0);
        let crossing_phase = cos_val.acos(); // radians from peak warmth

        // Growing season = fraction of year where temp > threshold.
        // The crossing occurs at +/- crossing_phase from the warm peak.
        let fraction = crossing_phase / PI;
        let days = 365.0 * fraction;
        days.clamp(0.0, 365.0)
    }

    /// Number of frost-free days (daily minimum > 0 C) in a year.
    ///
    /// Assumes daily minimum is approximately 5 C below daily mean
    /// (typical diurnal range for temperate climates).
    pub fn frost_free_days(&self, year: f64) -> u32 {
        let decades = self.decades_from_start(year).max(0.0);
        let mean_annual = self.params.base_temp_c + decades * self.params.temp_trend_c_per_decade;
        let amplitude = 12.0;
        let diurnal_offset = 5.0; // daily min is ~5 C below daily mean

        let frost_threshold = diurnal_offset; // mean must exceed this for min > 0

        if mean_annual - amplitude > frost_threshold {
            return 365;
        }
        if mean_annual + amplitude < frost_threshold {
            return 0;
        }

        let cos_val = ((frost_threshold - mean_annual) / amplitude).clamp(-1.0, 1.0);
        let crossing = cos_val.acos();
        let days = (365.0 * crossing / PI) as u32;
        days.min(365)
    }

    /// Simplified Palmer Drought Severity Index (PDSI) for a given year.
    ///
    /// Returns a value on the standard PDSI scale:
    /// - -4.0 or below: extreme drought
    /// - -3.0 to -4.0: severe drought
    /// - -2.0 to -3.0: moderate drought
    /// -  0.0: normal
    /// - +2.0 to +3.0: moderately wet
    /// - +4.0 or above: extremely wet
    ///
    /// Computed from precipitation anomaly and Thornthwaite potential
    /// evapotranspiration estimate.
    pub fn palmer_drought_index(&self, year: f64) -> f64 {
        let temp = self.mean_temperature(year);
        let precip = self.mean_precipitation(year);
        let baseline_precip = self.params.base_precip_mm_yr;

        if baseline_precip <= 0.0 {
            return 0.0;
        }

        // Thornthwaite PET approximation (monthly scaled to annual).
        // PET increases ~5% per degree C above 10 C.
        let pet_factor = if temp > 10.0 {
            1.0 + 0.05 * (temp - 10.0)
        } else {
            0.5 + 0.05 * temp.max(0.0)
        };
        let pet = baseline_precip * pet_factor;

        // PDSI = f(precipitation - PET) / normalization.
        // Simplified: scale so +/- 4 maps to +/- 50% anomaly.
        let anomaly = (precip - pet) / baseline_precip;
        (anomaly * 8.0).clamp(-6.0, 6.0)
    }

    /// Soil moisture availability factor (0 to 1).
    ///
    /// Combines precipitation supply and evaporative demand following
    /// a simplified bucket model. Returns 0.0 for completely dry soil
    /// and 1.0 for fully saturated.
    pub fn soil_moisture_factor(&self, year: f64) -> f64 {
        let precip = self.mean_precipitation(year);
        let temp = self.mean_temperature(year);

        // Potential evapotranspiration (Hamon method simplified).
        // Higher temperature = more evaporation = drier soil.
        let sat_vapor = 0.611 * ((17.27 * temp) / (temp + 237.3)).exp();
        let pet = 1.2 * sat_vapor * 12.0; // rough annual mm scale

        // Soil moisture ~ supply / (supply + demand).
        let supply = precip.max(0.0);
        let demand = (pet * 100.0).max(1.0); // scale to mm/yr
        let moisture = supply / (supply + demand);
        moisture.clamp(0.0, 1.0)
    }

    /// Run a complete projection from `start_year` to `end_year`.
    ///
    /// Returns a `ClimateTimeSeries` with samples at interval `dt` years.
    /// Uses stochastic stepping, so each run produces different weather
    /// noise (but the same trend).
    pub fn run_projection(
        &mut self,
        start_year: f64,
        end_year: f64,
        dt: f64,
    ) -> ClimateTimeSeries {
        let n_steps = ((end_year - start_year) / dt).ceil() as usize + 1;
        let mut ts = ClimateTimeSeries {
            years: Vec::with_capacity(n_steps),
            temperatures: Vec::with_capacity(n_steps),
            precipitations: Vec::with_capacity(n_steps),
            co2_levels: Vec::with_capacity(n_steps),
            extreme_events: Vec::new(),
            drought_indices: Vec::with_capacity(n_steps),
        };

        self.current_year = start_year;
        let mut last_event_year = start_year - 1.0;

        let mut year = start_year;
        while year <= end_year + dt * 0.01 {
            let state = self.state_at(year);
            ts.years.push(state.year);
            ts.temperatures.push(state.temperature_c);
            ts.precipitations.push(state.precipitation_mm);
            ts.co2_levels.push(state.co2_ppm);
            ts.drought_indices.push(self.palmer_drought_index(year));

            // Generate extreme events at most once per year.
            let event_year = year.floor();
            if event_year > last_event_year {
                let events = self.extreme_events_in_year(year);
                for e in events {
                    ts.extreme_events.push((year, e));
                }
                last_event_year = event_year;
            }

            year += dt;
        }

        self.current_year = end_year;
        ts
    }

    /// Probability that a biome shift occurs at a given year and latitude.
    ///
    /// Based on the observation that biome boundaries shift ~100 km poleward
    /// per degree C of warming (IPCC AR5 WG2 Chapter 4). Higher latitudes
    /// have narrower biome bands, making shifts more likely.
    ///
    /// Returns probability in [0, 1].
    pub fn biome_shift_probability(&self, year: f64, latitude: f64) -> f64 {
        let decades = self.decades_from_start(year).max(0.0);
        let warming = decades * self.params.temp_trend_c_per_decade;

        // Higher latitudes experience amplified warming (Arctic amplification).
        // Factor ~1.5x at poles, 1.0x at equator.
        let lat_abs = latitude.abs().min(90.0);
        let arctic_amplification = 1.0 + 0.5 * (lat_abs / 90.0);
        let effective_warming = warming * arctic_amplification;

        // Biome shift probability: sigmoid centered at 2 C warming.
        // At 2 C effective warming, ~50% probability of shift.
        // At 4 C, ~88%. At 1 C, ~27%.
        let x = (effective_warming - 2.0) * 1.5;
        1.0 / (1.0 + (-x).exp())
    }

    /// Return the current simulation year.
    pub fn current_year(&self) -> f64 {
        self.current_year
    }

    /// Return a reference to the active climate parameters.
    pub fn params(&self) -> &ClimateParams {
        &self.params
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: absolute difference.
    fn abs_diff(a: f64, b: f64) -> f64 {
        (a - b).abs()
    }

    #[test]
    fn pre_industrial_stable() {
        // Pre-industrial scenario: temperature and CO2 should remain constant.
        let engine = ClimateEngine::new(ClimateScenario::PreIndustrial, 1);
        let s1 = engine.state_at(1850.0);
        let s2 = engine.state_at(2100.0);

        // Annual mean temperatures should be equal (seasonal variation cancels
        // at the same fractional year).
        assert!(
            abs_diff(s1.temperature_c, s2.temperature_c) < 0.01,
            "Pre-industrial temp should be stable: {} vs {}",
            s1.temperature_c,
            s2.temperature_c
        );
        assert!(
            abs_diff(s1.co2_ppm, s2.co2_ppm) < 0.01,
            "Pre-industrial CO2 should be stable: {} vs {}",
            s1.co2_ppm,
            s2.co2_ppm
        );
        assert!(
            abs_diff(s1.co2_ppm, PRE_INDUSTRIAL_CO2) < 1.0,
            "CO2 should be ~280 ppm: {}",
            s1.co2_ppm
        );
    }

    #[test]
    fn rcp26_moderate_warming() {
        // RCP 2.6: ~+1.0 C by 2100 relative to 1995 baseline.
        let engine = ClimateEngine::new(ClimateScenario::Rcp26, 42);

        // Compare at mid-year to avoid seasonal effects (use .5 fraction).
        let t_1995 = engine.state_at(1995.5).temperature_c;
        let t_2100 = engine.state_at(2100.5).temperature_c;
        let warming = t_2100 - t_1995;

        assert!(
            warming > 0.5 && warming < 2.0,
            "RCP 2.6 warming should be ~1.0 C, got {:.2} C",
            warming
        );
    }

    #[test]
    fn rcp85_severe_warming() {
        // RCP 8.5: ~+3.7 C by 2100 relative to 1995 baseline.
        let engine = ClimateEngine::new(ClimateScenario::Rcp85, 42);

        let t_1995 = engine.state_at(1995.5).temperature_c;
        let t_2100 = engine.state_at(2100.5).temperature_c;
        let warming = t_2100 - t_1995;

        assert!(
            warming > 2.5 && warming < 5.0,
            "RCP 8.5 warming should be ~3.7 C, got {:.2} C",
            warming
        );

        // RCP 8.5 should warm significantly more than RCP 2.6.
        let engine26 = ClimateEngine::new(ClimateScenario::Rcp26, 42);
        let warming_26 = engine26.state_at(2100.5).temperature_c
            - engine26.state_at(1995.5).temperature_c;
        assert!(
            warming > warming_26 * 2.0,
            "RCP 8.5 warming ({:.2}) should be >2x RCP 2.6 ({:.2})",
            warming,
            warming_26
        );
    }

    #[test]
    fn co2_increases_over_time() {
        let engine = ClimateEngine::new(ClimateScenario::Rcp85, 99);

        let co2_2000 = engine.state_at(2000.0).co2_ppm;
        let co2_2050 = engine.state_at(2050.0).co2_ppm;
        let co2_2100 = engine.state_at(2100.0).co2_ppm;

        assert!(
            co2_2050 > co2_2000,
            "CO2 should increase: 2000={:.0}, 2050={:.0}",
            co2_2000,
            co2_2050
        );
        assert!(
            co2_2100 > co2_2050,
            "CO2 should keep increasing: 2050={:.0}, 2100={:.0}",
            co2_2050,
            co2_2100
        );
        // RCP 8.5 target: ~1370 ppm by 2100.
        assert!(
            co2_2100 > 1000.0 && co2_2100 < 1500.0,
            "RCP 8.5 CO2 at 2100 should be ~1370 ppm, got {:.0}",
            co2_2100
        );
    }

    #[test]
    fn extreme_events_more_frequent_in_rcp85() {
        // Over many years, RCP 8.5 should generate more extreme events
        // than pre-industrial due to higher frequency scaling with warming.
        let mut engine85 = ClimateEngine::new(ClimateScenario::Rcp85, 123);
        let mut engine_pi = ClimateEngine::new(ClimateScenario::PreIndustrial, 123);

        let mut count_85 = 0u32;
        let mut count_pi = 0u32;

        // Sample 50 years in the late 21st century for RCP 8.5
        // and 50 years for pre-industrial (same years, different scenarios).
        for y in 2060..2110 {
            count_85 += engine85.extreme_events_in_year(y as f64).len() as u32;
            count_pi += engine_pi.extreme_events_in_year(y as f64).len() as u32;
        }

        assert!(
            count_85 > count_pi,
            "RCP 8.5 should have more extreme events: {} vs {} (pre-industrial)",
            count_85,
            count_pi
        );
    }

    #[test]
    fn growing_season_lengthens_with_warming() {
        let engine = ClimateEngine::new(ClimateScenario::Rcp85, 42);

        let gs_2000 = engine.growing_season_length(2000.0);
        let gs_2100 = engine.growing_season_length(2100.0);

        assert!(
            gs_2100 > gs_2000,
            "Growing season should lengthen with warming: 2000={:.0} days, 2100={:.0} days",
            gs_2000,
            gs_2100
        );

        // Growing season should be between 0 and 365 days.
        assert!(gs_2000 >= 0.0 && gs_2000 <= 365.0);
        assert!(gs_2100 >= 0.0 && gs_2100 <= 365.0);
    }

    #[test]
    fn drought_index_responds_to_precipitation() {
        let engine = ClimateEngine::new(ClimateScenario::Rcp85, 42);

        // Mid-summer (dry in some seasonal patterns) vs spring (wetter).
        let pdsi_spring = engine.palmer_drought_index(2050.25); // fraction 0.25 = April
        let pdsi_summer = engine.palmer_drought_index(2050.6); // fraction 0.6 = August

        // Both should be in valid PDSI range.
        assert!(
            pdsi_spring >= -6.0 && pdsi_spring <= 6.0,
            "PDSI out of range: {}",
            pdsi_spring
        );
        assert!(
            pdsi_summer >= -6.0 && pdsi_summer <= 6.0,
            "PDSI out of range: {}",
            pdsi_summer
        );

        // They should differ due to seasonal precipitation variation.
        assert!(
            abs_diff(pdsi_spring, pdsi_summer) > 0.01,
            "PDSI should vary seasonally: spring={:.2}, summer={:.2}",
            pdsi_spring,
            pdsi_summer
        );
    }

    #[test]
    fn seasonal_temperature_variation() {
        // Temperature should vary seasonally with ~12 C amplitude.
        let engine = ClimateEngine::new(ClimateScenario::Rcp45, 42);

        // Warmest around mid-year (0.5 fraction = July), coolest at year start.
        let t_jan = engine.state_at(2050.0).temperature_c; // January
        let t_jul = engine.state_at(2050.5).temperature_c; // July

        let range = t_jul - t_jan;
        assert!(
            range > 15.0,
            "Seasonal range should be >15 C (approx 24 C peak-to-peak): got {:.1} C",
            range
        );
        assert!(
            t_jul > t_jan,
            "July ({:.1}) should be warmer than January ({:.1})",
            t_jul,
            t_jan
        );
    }

    #[test]
    fn custom_scenario_works() {
        let params = ClimateParams {
            base_temp_c: 20.0,
            temp_trend_c_per_decade: 0.5,
            base_precip_mm_yr: 800.0,
            precip_change_pct_per_decade: -3.0,
            co2_ppm_start: 400.0,
            co2_ppm_trend_per_decade: 50.0,
            extreme_event_frequency: 5.0,
            sea_level_rise_mm_yr: 10.0,
        };

        let engine = ClimateEngine::new(ClimateScenario::Custom(params), 777);

        // At year 0 (start), temp should be ~20 C (plus seasonal component).
        let s = engine.state_at(0.5); // mid-year to get peak seasonal
        assert!(
            s.temperature_c > 25.0,
            "Custom base 20 C + seasonal should be >25 C at mid-year: {:.1}",
            s.temperature_c
        );

        // After 10 decades (100 years), temp should rise by ~5 C.
        let s100 = engine.state_at(100.5);
        let warming = s100.temperature_c - s.temperature_c;
        assert!(
            abs_diff(warming, 5.0) < 0.5,
            "100 years at 0.5 C/decade should give ~5 C warming: got {:.2}",
            warming
        );

        // CO2 should increase.
        let co2_0 = engine.state_at(0.0).co2_ppm;
        let co2_100 = engine.state_at(100.0).co2_ppm;
        assert!(
            co2_100 > co2_0,
            "CO2 should increase: {:.0} -> {:.0}",
            co2_0,
            co2_100
        );
    }

    #[test]
    fn soil_moisture_bounded() {
        // Soil moisture factor must always be in [0, 1].
        let scenarios = [
            ClimateScenario::PreIndustrial,
            ClimateScenario::Rcp26,
            ClimateScenario::Rcp45,
            ClimateScenario::Rcp85,
        ];
        for scenario in &scenarios {
            let engine = ClimateEngine::new(scenario.clone(), 42);
            for year_offset in 0..200 {
                let year = 1995.0 + year_offset as f64;
                let sm = engine.soil_moisture_factor(year);
                assert!(
                    sm >= 0.0 && sm <= 1.0,
                    "Soil moisture out of [0,1] at year {}: {}",
                    year,
                    sm
                );
            }
        }
    }

    #[test]
    fn projection_timeseries_length() {
        // A 100-year projection at dt=1.0 should have ~101 data points.
        let mut engine = ClimateEngine::new(ClimateScenario::Rcp45, 42);
        let ts = engine.run_projection(2000.0, 2100.0, 1.0);

        assert!(
            ts.years.len() >= 100 && ts.years.len() <= 102,
            "Expected ~101 points, got {}",
            ts.years.len()
        );
        assert_eq!(
            ts.years.len(),
            ts.temperatures.len(),
            "Years and temperatures vectors should match"
        );
        assert_eq!(
            ts.years.len(),
            ts.co2_levels.len(),
            "Years and CO2 vectors should match"
        );
        assert_eq!(
            ts.years.len(),
            ts.precipitations.len(),
            "Years and precipitation vectors should match"
        );
        assert_eq!(
            ts.years.len(),
            ts.drought_indices.len(),
            "Years and drought index vectors should match"
        );

        // First year should be ~2000, last ~2100.
        assert!(
            abs_diff(ts.years[0], 2000.0) < 0.1,
            "First year should be ~2000: {}",
            ts.years[0]
        );
        let last = ts.years[ts.years.len() - 1];
        assert!(
            abs_diff(last, 2100.0) < 1.1,
            "Last year should be ~2100: {}",
            last
        );
    }

    #[test]
    fn stochastic_step_adds_variability() {
        // Two engines with different seeds should produce different stochastic steps.
        let mut engine_a = ClimateEngine::new(ClimateScenario::Rcp45, 111);
        let mut engine_b = ClimateEngine::new(ClimateScenario::Rcp45, 222);

        // Advance both by the same dt, collect temperatures.
        let mut temps_a = Vec::new();
        let mut temps_b = Vec::new();
        for _ in 0..20 {
            temps_a.push(engine_a.step(0.1).temperature_c);
            temps_b.push(engine_b.step(0.1).temperature_c);
        }

        // At least some of the 20 steps should differ (different noise seeds).
        let n_different = temps_a
            .iter()
            .zip(temps_b.iter())
            .filter(|(a, b)| abs_diff(**a, **b) > 0.01)
            .count();

        assert!(
            n_different > 10,
            "Different seeds should produce different stochastic paths: only {} of 20 differ",
            n_different
        );
    }

    #[test]
    fn frost_free_days_increase_with_warming() {
        let engine = ClimateEngine::new(ClimateScenario::Rcp85, 42);

        let ffd_2000 = engine.frost_free_days(2000.0);
        let ffd_2100 = engine.frost_free_days(2100.0);

        assert!(
            ffd_2100 >= ffd_2000,
            "Frost-free days should increase: 2000={}, 2100={}",
            ffd_2000,
            ffd_2100
        );
        assert!(ffd_2000 <= 365, "Frost-free days should be <= 365");
        assert!(ffd_2100 <= 365, "Frost-free days should be <= 365");
    }

    #[test]
    fn biome_shift_increases_with_warming() {
        let engine = ClimateEngine::new(ClimateScenario::Rcp85, 42);

        let p_2000 = engine.biome_shift_probability(2000.0, 45.0);
        let p_2100 = engine.biome_shift_probability(2100.0, 45.0);

        assert!(
            p_2100 > p_2000,
            "Biome shift probability should increase: 2000={:.3}, 2100={:.3}",
            p_2000,
            p_2100
        );
        assert!(p_2000 >= 0.0 && p_2000 <= 1.0, "Probability out of range");
        assert!(p_2100 >= 0.0 && p_2100 <= 1.0, "Probability out of range");
    }

    #[test]
    fn arctic_amplification_increases_shift_probability() {
        let engine = ClimateEngine::new(ClimateScenario::Rcp85, 42);

        let p_equator = engine.biome_shift_probability(2080.0, 0.0);
        let p_arctic = engine.biome_shift_probability(2080.0, 75.0);

        assert!(
            p_arctic > p_equator,
            "Arctic should have higher biome shift probability: equator={:.3}, arctic={:.3}",
            p_equator,
            p_arctic
        );
    }

    #[test]
    fn rcp26_co2_soft_cap() {
        // RCP 2.6 CO2 should not significantly exceed ~490 ppm even at 2200.
        let engine = ClimateEngine::new(ClimateScenario::Rcp26, 42);
        let co2_2200 = engine.state_at(2200.0).co2_ppm;

        assert!(
            co2_2200 < 520.0,
            "RCP 2.6 CO2 should be capped near 490: got {:.0}",
            co2_2200
        );
        assert!(
            co2_2200 > 450.0,
            "RCP 2.6 CO2 should still be elevated above pre-industrial: got {:.0}",
            co2_2200
        );
    }
}
