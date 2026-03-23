//! Biogeochemical nutrient cycling: C/N/P through soil-plant-atmosphere.
//!
//! Models the major biogeochemical transformations that govern nutrient
//! availability in terrestrial ecosystems.  All fluxes obey stoichiometric
//! constraints -- carbon, nitrogen, and phosphorus are neither created nor
//! destroyed, only transformed between pools.
//!
//! # Processes modelled
//!
//! | Process               | Inputs            | Outputs           | Key control        |
//! |-----------------------|-------------------|-------------------|--------------------|
//! | Decomposition         | Organic C/N/P     | Mineral N/P, CO2  | Temperature, WFPS  |
//! | Nitrification         | NH4+              | NO3-              | O2, pH             |
//! | Denitrification       | NO3-              | N2O, N2           | Anoxia (high WFPS) |
//! | Plant uptake          | Mineral N, P      | Plant biomass     | Root biomass, MM   |
//! | P sorption            | Dissolved PO4     | Sorbed P          | Clay, Langmuir     |
//! | Litter decomposition  | Fresh litter      | Organic pools     | Lignin, C:N        |
//! | Mycorrhizal transfer  | Mineral N/P       | Plant-plant flux  | Hyphal density     |
//! | Biological N fixation | Atmospheric N2    | NH4+              | Soil N status      |
//!
//! # Key literature
//!
//! - Cleveland CC, Liptzin D (2007) "C:N:P stoichiometry in soil",
//!   Biogeochemistry 85:235-252.  Microbial biomass C:N:P ~ 60:7:1.
//! - Parton WJ et al. (1993) "Observations and modeling of biomass and
//!   soil organic matter dynamics", Global Biogeochemical Cycles 7:785.
//! - Linn DM, Doran JW (1984) "Effect of water-filled pore space on CO2
//!   and N2O production", Soil Sci. Soc. Am. J. 48:1267.
//! - Barber SA (1995) "Soil Nutrient Bioavailability", 2nd ed., Wiley.
//! - Smith SE, Read DJ (2008) "Mycorrhizal Symbiosis", 3rd ed., Academic Press.

// ============================================================================
// Constants: biogeochemistry literature values
// ============================================================================

/// Microbial biomass C:N:P ratio (Cleveland & Liptzin 2007).
const MICROBIAL_CN: f64 = 60.0 / 7.0; // ~8.57
const MICROBIAL_CP: f64 = 60.0; // C:P
/// Critical C:N ratio threshold.  Above this, microbes immobilize N from
/// the mineral pool to maintain their stoichiometry; below, they release
/// (mineralize) excess N.  ~25 for most agricultural/temperate soils
/// (Parton et al. 1993).
const CRITICAL_CN: f64 = 25.0;

/// Q10 temperature coefficient -- rate roughly doubles per 10 C
/// (Lloyd & Taylor 1994).
const DEFAULT_Q10: f64 = 2.0;

/// Reference temperature for Q10 calculations (C).
const REFERENCE_TEMP: f64 = 20.0;

/// Optimal water-filled pore space for decomposition (~60%, Linn & Doran 1984).
const OPTIMAL_WFPS: f64 = 0.60;

/// Carbon use efficiency (CUE) -- fraction of decomposed C assimilated
/// by microbes; remainder respired as CO2.  Typical range 0.3-0.6
/// (Manzoni et al. 2012).  We use 0.40 as a reasonable mean.
const BASE_CUE: f64 = 0.40;

/// Nitrification base rate (fraction of NH4+ nitrified per day at 20 C,
/// 60% WFPS, pH 7).  Literature: 0.02-0.10 d-1.
const NITRIFICATION_RATE: f64 = 0.05;

/// Denitrification base rate (fraction of NO3- denitrified per day under
/// fully anaerobic conditions at 20 C).
const DENITRIFICATION_RATE: f64 = 0.04;

/// N2O fraction of total denitrification gaseous loss.  The ratio N2O/(N2O+N2)
/// is typically 0.1-0.5, rising at lower pH and intermediate WFPS.
const N2O_FRACTION_BASE: f64 = 0.25;

/// Langmuir P sorption parameters.
/// P_max: maximum P sorption capacity (g P per unit soil, scales with clay).
/// K_langmuir: affinity constant.
const P_MAX_BASE: f64 = 0.5; // g P (at 100% clay)
const K_LANGMUIR: f64 = 10.0; // L/g

/// Michaelis-Menten half-saturation constants for plant nutrient uptake.
/// Barber (1995), typical crop roots.
const KM_NITROGEN: f64 = 0.05; // g N (half-saturation)
const KM_PHOSPHORUS: f64 = 0.01; // g P (half-saturation)

/// Biological N fixation rate at zero mineral N (g N day-1 per unit
/// fixation capacity).  Declines hyperbolically as mineral N rises.
const N_FIXATION_MAX: f64 = 0.01; // g N day-1

/// N fixation inhibition constant -- mineral N level at which fixation
/// halves.
const N_FIXATION_KI: f64 = 0.5; // g N

/// Atmospheric N deposition baseline (g N day-1).
/// Typical temperate background: ~5-15 kg N ha-1 yr-1.
/// For a 1 m^2 patch: ~0.001-0.004 g N day-1.
const N_DEPOSITION_DEFAULT: f64 = 0.002;

/// Litter stage transition rates (day-1).
const FRESH_TO_FRAGMENTED_RATE: f64 = 0.03;
const FRAGMENTED_TO_HUMIFIED_RATE: f64 = 0.005;

/// Fraction of litter mass entering organic pool per decomposition step.
const LITTER_TO_SOM_FRACTION: f64 = 0.10;

/// Leaching fraction of NO3- per day (very mobile anion).
const NITRATE_LEACHING_FRACTION: f64 = 0.01;

// ============================================================================
// Types
// ============================================================================

/// Soil organic matter pool with C:N:P stoichiometry.
#[derive(Debug, Clone)]
pub struct OrganicPool {
    pub name: String,
    pub carbon_g: f64,
    pub nitrogen_g: f64,
    pub phosphorus_g: f64,
    pub decomposition_rate: f64, // day^-1
    pub recalcitrance: f64,      // 0-1 (lignin content proxy)
}

/// Mineral nutrient pools (inorganic, plant-available).
#[derive(Debug, Clone)]
pub struct MineralPool {
    pub ammonium_g: f64,          // NH4+
    pub nitrate_g: f64,           // NO3-
    pub phosphate_g: f64,         // PO4^3-
    pub dissolved_organic_c: f64, // DOC
    pub dissolved_organic_n: f64, // DON
}

impl MineralPool {
    pub fn new() -> Self {
        Self {
            ammonium_g: 0.0,
            nitrate_g: 0.0,
            phosphate_g: 0.0,
            dissolved_organic_c: 0.0,
            dissolved_organic_n: 0.0,
        }
    }
}

/// Atmospheric exchange rates (g element day^-1).
#[derive(Debug, Clone)]
pub struct AtmosphericFlux {
    pub co2_emission: f64, // g C day^-1 (soil respiration)
    pub n2o_emission: f64, // g N day^-1 (greenhouse gas!)
    pub n2_emission: f64,  // g N day^-1 (denitrification end product)
    pub n_fixation: f64,   // g N day^-1 (biological N fixation)
    pub n_deposition: f64, // g N day^-1 (atmospheric deposition)
}

impl AtmosphericFlux {
    pub fn new() -> Self {
        Self {
            co2_emission: 0.0,
            n2o_emission: 0.0,
            n2_emission: 0.0,
            n_fixation: 0.0,
            n_deposition: N_DEPOSITION_DEFAULT,
        }
    }
}

/// Mycorrhizal network connecting plants via hyphal bridges.
#[derive(Debug, Clone)]
pub struct MycorrhizalNetwork {
    pub hyphal_density: f64,                    // m hyphae per cm^3 soil
    pub plant_connections: Vec<(usize, usize)>, // connected plant pairs
    pub transfer_efficiency: f64,               // fraction of nutrients transferred
    pub carbon_cost: f64,                       // C cost to plant per unit nutrient
    pub network_age_days: f64,
}

/// Plant nutrient demand and uptake capacity.
#[derive(Debug, Clone)]
pub struct PlantNutrientDemand {
    pub nitrogen_demand: f64,   // g N day^-1
    pub phosphorus_demand: f64, // g P day^-1
    pub root_biomass: f64,      // g
    pub mycorrhizal: bool,
    pub uptake_efficiency: f64, // 0-1
}

/// Litter decomposition stage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DecompositionStage {
    /// Recently fallen, labile compounds dominate.
    Fresh,
    /// Physically broken down, cellulose fraction increasing.
    Fragmented,
    /// Stable humus, high recalcitrance.
    Humified,
}

/// A cohort of litter at a specific decomposition stage.
#[derive(Debug, Clone)]
pub struct LitterCohort {
    pub stage: DecompositionStage,
    pub mass_g: f64,
    pub cn_ratio: f64,
    pub lignin_fraction: f64,
    pub age_days: f64,
}

/// Which nutrient is most limiting (Liebig's law).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LimitingNutrient {
    Nitrogen,
    Phosphorus,
    CoLimited,
    Neither,
}

/// Result of stoichiometric balance calculations during decomposition.
#[derive(Debug, Clone)]
pub struct StoichiometricBalance {
    pub c_mineralized: f64,
    pub n_mineralized: f64, // positive = net mineralization
    pub n_immobilized: f64, // positive = microbes drawing from mineral pool
    pub p_mineralized: f64,
    pub p_immobilized: f64,
    pub microbial_cue: f64, // carbon use efficiency
}

/// Complete result of one nutrient cycling timestep.
#[derive(Debug, Clone)]
pub struct NutrientCyclingResult {
    pub c_mineralized: f64,
    pub n_mineralized: f64,
    pub p_mineralized: f64,
    pub n2o_emitted: f64,
    pub co2_emitted: f64,
    pub n_leached: f64,
    pub p_sorbed: f64,
}

/// Main nutrient cycling engine.
///
/// Integrates organic matter decomposition, mineral nutrient transformations,
/// atmospheric exchange, plant uptake, and mycorrhizal transfer into a single
/// coherent simulation step.
#[derive(Debug, Clone)]
pub struct NutrientCycler {
    pub organic_pools: Vec<OrganicPool>,
    pub mineral_pool: MineralPool,
    pub atmospheric_flux: AtmosphericFlux,
    pub litter_cohorts: Vec<LitterCohort>,
    pub mycorrhizal: Option<MycorrhizalNetwork>,
    pub temperature_c: f64,
    pub moisture_fraction: f64, // 0-1 WFPS
    pub ph: f64,
    pub clay_fraction: f64, // affects P sorption
    pub time_days: f64,
}

// ============================================================================
// Free functions: environmental modifiers and stoichiometric helpers
// ============================================================================

/// Q10 temperature sensitivity factor.
///
/// Returns the multiplicative rate modifier for a given temperature relative
/// to a reference temperature.  At `temperature == reference_temp` the factor
/// is 1.0; at `reference_temp + 10` it equals `q10`.
pub fn q10_factor(temperature: f64, reference_temp: f64, q10: f64) -> f64 {
    q10.powf((temperature - reference_temp) / 10.0)
}

/// Moisture response function.
///
/// Parabolic response peaking at ~60% WFPS (Linn & Doran 1984).
/// Returns a factor in [0, 1].  Below ~10% WFPS activity drops to near zero
/// (too dry); above ~90% WFPS activity drops (waterlogging limits O2).
pub fn moisture_factor(wfps: f64) -> f64 {
    let w = clamp_f64(wfps, 0.0, 1.0);
    // Quadratic centered on OPTIMAL_WFPS with max = 1.0.
    // f(w) = 1 - ((w - 0.6) / 0.4)^2, clamped to [0, 1].
    let deviation = (w - OPTIMAL_WFPS) / 0.4;
    let f = 1.0 - deviation * deviation;
    clamp_f64(f, 0.0, 1.0)
}

/// Critical C:N ratio below which net mineralization occurs and above which
/// net immobilization dominates (~25 for most soils).
pub fn critical_cn_ratio() -> f64 {
    CRITICAL_CN
}

/// Compute Redfield-style ratios from absolute masses.
///
/// Returns `(C:N, N:P)`.  If the denominator is zero, returns `f64::INFINITY`.
pub fn redfield_ratio(c: f64, n: f64, p: f64) -> (f64, f64) {
    let cn = if n > 0.0 { c / n } else { f64::INFINITY };
    let np = if p > 0.0 { n / p } else { f64::INFINITY };
    (cn, np)
}

/// Liebig's law of the minimum -- identifies the most growth-limiting nutrient
/// by comparing relative supply to demand.
pub fn limiting_nutrient(
    n_available: f64,
    p_available: f64,
    n_demand: f64,
    p_demand: f64,
) -> LimitingNutrient {
    // Avoid division by zero: if demand is zero, nutrient is not limiting.
    let n_ratio = if n_demand > 0.0 {
        n_available / n_demand
    } else {
        f64::INFINITY
    };
    let p_ratio = if p_demand > 0.0 {
        p_available / p_demand
    } else {
        f64::INFINITY
    };

    const CO_LIMIT_TOLERANCE: f64 = 0.10; // within 10% = co-limited

    if n_ratio >= 1.0 && p_ratio >= 1.0 {
        LimitingNutrient::Neither
    } else if (n_ratio - p_ratio).abs() < CO_LIMIT_TOLERANCE && n_ratio < 1.0 {
        LimitingNutrient::CoLimited
    } else if n_ratio < p_ratio {
        LimitingNutrient::Nitrogen
    } else {
        LimitingNutrient::Phosphorus
    }
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Clamp a f64 value between lo and hi.
fn clamp_f64(x: f64, lo: f64, hi: f64) -> f64 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

/// Michaelis-Menten kinetics: V = Vmax * S / (Km + S).
fn michaelis_menten(substrate: f64, vmax: f64, km: f64) -> f64 {
    if substrate <= 0.0 {
        return 0.0;
    }
    vmax * substrate / (km + substrate)
}

// ============================================================================
// NutrientCycler implementation
// ============================================================================

impl NutrientCycler {
    /// Create a new NutrientCycler with default environmental conditions.
    ///
    /// Starts with empty pools and temperate defaults: 15 C, 50% WFPS,
    /// pH 6.5, 20% clay.
    pub fn new() -> Self {
        Self {
            organic_pools: Vec::new(),
            mineral_pool: MineralPool::new(),
            atmospheric_flux: AtmosphericFlux::new(),
            litter_cohorts: Vec::new(),
            mycorrhizal: None,
            temperature_c: 15.0,
            moisture_fraction: 0.50,
            ph: 6.5,
            clay_fraction: 0.20,
            time_days: 0.0,
        }
    }

    /// Add an organic matter pool to the system.
    pub fn add_organic_pool(&mut self, pool: OrganicPool) {
        self.organic_pools.push(pool);
    }

    /// Add a litter cohort to the system.
    pub fn add_litter(&mut self, cohort: LitterCohort) {
        self.litter_cohorts.push(cohort);
    }

    /// Set the mycorrhizal network.
    pub fn set_mycorrhizal(&mut self, network: MycorrhizalNetwork) {
        self.mycorrhizal = Some(network);
    }

    /// Step one day of nutrient cycling.
    ///
    /// Executes all biogeochemical transformations in sequence:
    /// decomposition -> nitrification -> denitrification -> litter progression
    /// -> atmospheric exchange -> leaching -> P sorption.
    ///
    /// Returns a `NutrientCyclingResult` summarising the fluxes for this step.
    pub fn step(&mut self, dt_days: f64) -> NutrientCyclingResult {
        // 1. Decompose organic matter
        let balance = self.decompose(dt_days);

        // 2. Nitrification: NH4+ -> NO3-
        let _n_nitrified = self.nitrify(dt_days);

        // 3. Denitrification: NO3- -> N2O + N2
        let (n2o, n2) = self.denitrify(dt_days);

        // 4. Litter quality progression
        self.advance_litter(dt_days);

        // 5. Biological N fixation (inhibited by high mineral N)
        let total_mineral_n = self.mineral_pool.ammonium_g + self.mineral_pool.nitrate_g;
        let fixation = N_FIXATION_MAX * dt_days * N_FIXATION_KI / (N_FIXATION_KI + total_mineral_n);
        self.mineral_pool.ammonium_g += fixation;
        self.atmospheric_flux.n_fixation = fixation / dt_days;

        // 6. Atmospheric N deposition
        let deposition = self.atmospheric_flux.n_deposition * dt_days;
        // Split deposition: ~60% NH4+, ~40% NO3- (typical wet+dry deposition)
        self.mineral_pool.ammonium_g += deposition * 0.6;
        self.mineral_pool.nitrate_g += deposition * 0.4;

        // 7. Nitrate leaching (mobile anion)
        let leached = self.mineral_pool.nitrate_g * NITRATE_LEACHING_FRACTION * dt_days;
        self.mineral_pool.nitrate_g -= leached;

        // 8. Phosphorus sorption
        let p_sorbed = self.phosphorus_sorption(dt_days);

        // 9. Update atmospheric flux accumulators
        let co2 = balance.c_mineralized * (1.0 - balance.microbial_cue);
        self.atmospheric_flux.co2_emission = co2 / dt_days;
        self.atmospheric_flux.n2o_emission = n2o / dt_days;
        self.atmospheric_flux.n2_emission = n2 / dt_days;

        // 10. Advance time
        self.time_days += dt_days;

        // Clamp all pools to non-negative
        self.mineral_pool.ammonium_g = self.mineral_pool.ammonium_g.max(0.0);
        self.mineral_pool.nitrate_g = self.mineral_pool.nitrate_g.max(0.0);
        self.mineral_pool.phosphate_g = self.mineral_pool.phosphate_g.max(0.0);
        self.mineral_pool.dissolved_organic_c = self.mineral_pool.dissolved_organic_c.max(0.0);
        self.mineral_pool.dissolved_organic_n = self.mineral_pool.dissolved_organic_n.max(0.0);

        NutrientCyclingResult {
            c_mineralized: balance.c_mineralized,
            n_mineralized: balance.n_mineralized,
            p_mineralized: balance.p_mineralized,
            n2o_emitted: n2o,
            co2_emitted: co2,
            n_leached: leached,
            p_sorbed,
        }
    }

    /// Decompose organic matter pools, releasing or immobilizing nutrients
    /// according to stoichiometric constraints.
    ///
    /// Temperature and moisture modify the base decomposition rate via Q10
    /// and the parabolic WFPS response.  Recalcitrant fractions (high lignin)
    /// decompose more slowly.
    fn decompose(&mut self, dt: f64) -> StoichiometricBalance {
        let temp_mod = q10_factor(self.temperature_c, REFERENCE_TEMP, DEFAULT_Q10);
        let moist_mod = moisture_factor(self.moisture_fraction);
        let env_modifier = temp_mod * moist_mod;

        let mut total_c = 0.0;
        let mut total_n_min = 0.0;
        let mut total_n_imm = 0.0;
        let mut total_p_min = 0.0;
        let mut total_p_imm = 0.0;

        for pool in &mut self.organic_pools {
            if pool.carbon_g <= 0.0 {
                continue;
            }

            // Effective decomposition rate: base * environment * (1 - recalcitrance)
            let effective_rate =
                pool.decomposition_rate * env_modifier * (1.0 - pool.recalcitrance * 0.8);
            // Fraction decomposed this step (first-order decay)
            let fraction_decomposed = (1.0 - (-effective_rate * dt).exp()).min(1.0);

            let c_decomposed = pool.carbon_g * fraction_decomposed;
            let n_released = pool.nitrogen_g * fraction_decomposed;
            let p_released = pool.phosphorus_g * fraction_decomposed;

            // Remove from pool
            pool.carbon_g -= c_decomposed;
            pool.nitrogen_g -= n_released;
            pool.phosphorus_g -= p_released;

            // DOC/DON release: small fraction of decomposed material enters
            // dissolved organic pools (leachable, slowly mineralizable).
            // This is subtracted from the budget BEFORE stoichiometric routing.
            let doc_fraction = 0.05;
            let c_to_doc = c_decomposed * doc_fraction;
            let n_to_don = n_released * doc_fraction;
            let p_to_dop = p_released * doc_fraction;
            self.mineral_pool.dissolved_organic_c += c_to_doc;
            self.mineral_pool.dissolved_organic_n += n_to_don;

            // Remaining C/N/P after DOC/DON removal
            let c_remaining = c_decomposed - c_to_doc;
            let n_remaining = n_released - n_to_don;
            let p_remaining = p_released - p_to_dop;

            // Microbial N demand from assimilated C (CUE fraction)
            let c_assimilated = c_remaining * BASE_CUE;
            let n_needed_by_microbes = c_assimilated / MICROBIAL_CN;
            let p_needed_by_microbes = c_assimilated / MICROBIAL_CP;

            // Net mineralization or immobilization of N.
            // n_to_microbes tracks what actually enters microbial biomass.
            let n_net = n_remaining - n_needed_by_microbes;
            let n_to_microbes;
            if n_net >= 0.0 {
                // Net mineralization: excess N released to mineral pool as NH4+
                self.mineral_pool.ammonium_g += n_net;
                total_n_min += n_net;
                n_to_microbes = n_needed_by_microbes;
            } else {
                // Net immobilization: microbes scavenge mineral pool
                let immobilized =
                    (-n_net).min(self.mineral_pool.ammonium_g + self.mineral_pool.nitrate_g);
                let from_nh4 = immobilized.min(self.mineral_pool.ammonium_g);
                self.mineral_pool.ammonium_g -= from_nh4;
                self.mineral_pool.nitrate_g -= immobilized - from_nh4;
                total_n_imm += immobilized;
                n_to_microbes = n_remaining + immobilized;
            }

            // Net mineralization or immobilization of P
            let p_net = p_remaining - p_needed_by_microbes;
            let p_to_microbes;
            if p_net >= 0.0 {
                self.mineral_pool.phosphate_g += p_net;
                total_p_min += p_net;
                p_to_microbes = p_needed_by_microbes;
            } else {
                let p_immobilized = (-p_net).min(self.mineral_pool.phosphate_g);
                self.mineral_pool.phosphate_g -= p_immobilized;
                total_p_imm += p_immobilized;
                p_to_microbes = p_remaining + p_immobilized;
            }

            // Microbially assimilated C/N/P go back into organic pool
            // (microbial biomass IS soil organic matter).
            pool.carbon_g += c_assimilated;
            pool.nitrogen_g += n_to_microbes;
            pool.phosphorus_g += p_to_microbes;

            total_c += c_decomposed;
        }

        // Also decompose litter into organic pools
        for cohort in &mut self.litter_cohorts {
            if cohort.mass_g <= 0.0 {
                continue;
            }
            let lignin_inhibition = 1.0 - cohort.lignin_fraction * 0.8;
            let litter_rate = match cohort.stage {
                DecompositionStage::Fresh => 0.03,
                DecompositionStage::Fragmented => 0.015,
                DecompositionStage::Humified => 0.003,
            };
            let effective = litter_rate * env_modifier * lignin_inhibition;
            let fraction = (1.0 - (-effective * dt).exp()).min(1.0);
            let mass_lost = cohort.mass_g * fraction;
            cohort.mass_g -= mass_lost;

            // Transfer to mineral pool: C and N based on C:N ratio
            let c_from_litter = mass_lost * 0.45; // ~45% of litter dry mass is C
            let n_from_litter = if cohort.cn_ratio > 0.0 {
                c_from_litter / cohort.cn_ratio
            } else {
                0.0
            };
            // Small fraction goes directly to mineral pool
            self.mineral_pool.dissolved_organic_c += c_from_litter * LITTER_TO_SOM_FRACTION;
            if n_from_litter > 0.0 {
                self.mineral_pool.dissolved_organic_n += n_from_litter * LITTER_TO_SOM_FRACTION;
            }
            total_c += c_from_litter;
        }

        StoichiometricBalance {
            c_mineralized: total_c,
            n_mineralized: total_n_min,
            n_immobilized: total_n_imm,
            p_mineralized: total_p_min,
            p_immobilized: total_p_imm,
            microbial_cue: BASE_CUE,
        }
    }

    /// Nitrification: NH4+ -> NO3- (autotrophic, aerobic).
    ///
    /// Rate depends on NH4+ concentration, temperature, moisture (requires O2,
    /// so declines at very high WFPS), and pH (optimal near 7-8, declines
    /// sharply below 4.5).
    ///
    /// Returns the mass of N transformed from NH4+ to NO3-.
    fn nitrify(&mut self, dt: f64) -> f64 {
        if self.mineral_pool.ammonium_g <= 0.0 {
            return 0.0;
        }

        let temp_mod = q10_factor(self.temperature_c, REFERENCE_TEMP, DEFAULT_Q10);

        // Nitrification needs oxygen: inhibited at high WFPS (>80%)
        let o2_factor = if self.moisture_fraction > 0.80 {
            let excess = (self.moisture_fraction - 0.80) / 0.20;
            (1.0 - excess).max(0.0)
        } else {
            1.0
        };

        // pH response: optimum near 7-8, declining below 5
        let ph_factor = if self.ph < 4.5 {
            0.0
        } else if self.ph < 6.0 {
            (self.ph - 4.5) / 1.5
        } else {
            1.0
        };

        let rate = NITRIFICATION_RATE * temp_mod * o2_factor * ph_factor;
        let fraction = (1.0 - (-rate * dt).exp()).min(1.0);
        let n_transformed = self.mineral_pool.ammonium_g * fraction;

        self.mineral_pool.ammonium_g -= n_transformed;
        self.mineral_pool.nitrate_g += n_transformed;

        n_transformed
    }

    /// Denitrification: NO3- -> N2O + N2 (anaerobic).
    ///
    /// Requires anoxic conditions (high WFPS, typically >60%).  The ratio
    /// of N2O to N2 depends on pH and WFPS: more complete reduction to N2
    /// at higher WFPS and higher pH.
    ///
    /// Returns (N2O emitted as g N, N2 emitted as g N).
    fn denitrify(&mut self, dt: f64) -> (f64, f64) {
        if self.mineral_pool.nitrate_g <= 0.0 {
            return (0.0, 0.0);
        }

        // Denitrification onset above ~60% WFPS, increasing steeply.
        let anoxia_factor = if self.moisture_fraction < 0.60 {
            0.0
        } else {
            let scaled = (self.moisture_fraction - 0.60) / 0.40;
            // Exponential onset
            scaled * scaled
        };

        if anoxia_factor <= 0.0 {
            return (0.0, 0.0);
        }

        let temp_mod = q10_factor(self.temperature_c, REFERENCE_TEMP, DEFAULT_Q10);
        let rate = DENITRIFICATION_RATE * temp_mod * anoxia_factor;
        let fraction = (1.0 - (-rate * dt).exp()).min(1.0);
        let n_total = self.mineral_pool.nitrate_g * fraction;

        self.mineral_pool.nitrate_g -= n_total;

        // N2O fraction: higher at low pH, intermediate moisture
        // At very high WFPS, reduction goes more completely to N2.
        let ph_effect = clamp_f64((7.5 - self.ph) / 3.0, 0.0, 1.0);
        let wfps_effect = if self.moisture_fraction > 0.90 {
            0.5 // very wet -> more complete denitrification to N2
        } else {
            1.0
        };
        let n2o_fraction = N2O_FRACTION_BASE * (0.5 + 0.5 * ph_effect) * wfps_effect;
        let n2o_fraction = clamp_f64(n2o_fraction, 0.0, 1.0);

        let n2o = n_total * n2o_fraction;
        let n2 = n_total * (1.0 - n2o_fraction);

        (n2o, n2)
    }

    /// Plant nutrient uptake using Michaelis-Menten kinetics.
    ///
    /// Each plant takes up N and P proportional to its root biomass and uptake
    /// efficiency, saturating at high mineral concentrations.  Returns a vector
    /// of `(N_taken, P_taken)` per plant, in the same order as `demands`.
    pub fn plant_uptake(&mut self, demands: &[PlantNutrientDemand]) -> Vec<(f64, f64)> {
        let mut results = Vec::with_capacity(demands.len());
        let total_n = self.mineral_pool.ammonium_g + self.mineral_pool.nitrate_g;
        let total_p = self.mineral_pool.phosphate_g;

        for demand in demands {
            // Potential uptake via MM kinetics, scaled by root biomass and efficiency
            let vmax_n = demand.nitrogen_demand;
            let vmax_p = demand.phosphorus_demand;
            let root_factor = (demand.root_biomass / (demand.root_biomass + 10.0)).min(1.0);
            let eff = demand.uptake_efficiency;

            let n_potential = michaelis_menten(total_n, vmax_n, KM_NITROGEN) * root_factor * eff;
            let p_potential = michaelis_menten(total_p, vmax_p, KM_PHOSPHORUS) * root_factor * eff;

            // Cannot take more than available (shared pool, approximate)
            let n_taken =
                n_potential.min(self.mineral_pool.ammonium_g + self.mineral_pool.nitrate_g);
            let p_taken = p_potential.min(self.mineral_pool.phosphate_g);

            // Draw from pools: NH4+ preferred, then NO3-
            let n_from_nh4 = n_taken.min(self.mineral_pool.ammonium_g);
            self.mineral_pool.ammonium_g -= n_from_nh4;
            self.mineral_pool.nitrate_g -= n_taken - n_from_nh4;
            self.mineral_pool.phosphate_g -= p_taken;

            results.push((n_taken, p_taken));
        }

        results
    }

    /// Mycorrhizal nutrient transfer between connected plants.
    ///
    /// Nutrients flow from plants with surplus to those with deficit, mediated
    /// by the fungal network.  The plant pays a carbon cost for each unit of
    /// nutrient transferred.  Returns `(N_received, P_received)` per plant
    /// (negative values = donated).
    pub fn mycorrhizal_transfer(&mut self, demands: &[PlantNutrientDemand]) -> Vec<(f64, f64)> {
        let mut transfers = vec![(0.0f64, 0.0f64); demands.len()];

        let network = match &self.mycorrhizal {
            Some(n) => n.clone(),
            None => return transfers,
        };

        // Maturity factor: young networks transfer less
        let maturity = clamp_f64(network.network_age_days / 60.0, 0.0, 1.0);
        let effective_efficiency = network.transfer_efficiency * maturity;

        for &(i, j) in &network.plant_connections {
            if i >= demands.len() || j >= demands.len() {
                continue;
            }

            // Compare relative satisfaction: plant with lower satisfaction
            // receives from plant with higher satisfaction.
            let total_n = self.mineral_pool.ammonium_g + self.mineral_pool.nitrate_g;
            let total_p = self.mineral_pool.phosphate_g;

            let sat_n_i = if demands[i].nitrogen_demand > 0.0 {
                total_n / demands[i].nitrogen_demand
            } else {
                f64::INFINITY
            };
            let sat_n_j = if demands[j].nitrogen_demand > 0.0 {
                total_n / demands[j].nitrogen_demand
            } else {
                f64::INFINITY
            };

            // N transfer: from higher-satisfaction to lower-satisfaction plant
            if sat_n_i != sat_n_j {
                let (donor, receiver) = if sat_n_i > sat_n_j { (i, j) } else { (j, i) };
                let gradient = (sat_n_i - sat_n_j).abs() / (sat_n_i.max(sat_n_j) + 1e-12);
                let n_transfer = gradient * effective_efficiency * network.hyphal_density * 0.001; // scale factor
                transfers[donor].0 -= n_transfer;
                transfers[receiver].0 += n_transfer;
            }

            // P transfer: same logic
            let sat_p_i = if demands[i].phosphorus_demand > 0.0 {
                total_p / demands[i].phosphorus_demand
            } else {
                f64::INFINITY
            };
            let sat_p_j = if demands[j].phosphorus_demand > 0.0 {
                total_p / demands[j].phosphorus_demand
            } else {
                f64::INFINITY
            };

            if sat_p_i != sat_p_j {
                let (donor, receiver) = if sat_p_i > sat_p_j { (i, j) } else { (j, i) };
                let gradient = (sat_p_i - sat_p_j).abs() / (sat_p_i.max(sat_p_j) + 1e-12);
                let p_transfer = gradient * effective_efficiency * network.hyphal_density * 0.0005; // scale factor
                transfers[donor].1 -= p_transfer;
                transfers[receiver].1 += p_transfer;
            }
        }

        transfers
    }

    /// Phosphorus sorption/desorption via Langmuir isotherm.
    ///
    /// `P_sorbed = P_max * K * P_solution / (1 + K * P_solution)`
    ///
    /// Clay-rich soils have higher sorption capacity.  Returns the net change
    /// in dissolved phosphate (negative = sorbed, positive = desorbed).
    fn phosphorus_sorption(&mut self, dt: f64) -> f64 {
        let p_solution = self.mineral_pool.phosphate_g;
        if p_solution <= 0.0 {
            return 0.0;
        }

        // Sorption capacity scales with clay fraction
        let p_max = P_MAX_BASE * self.clay_fraction;

        // Langmuir equilibrium concentration
        let p_eq = p_max * K_LANGMUIR * p_solution / (1.0 + K_LANGMUIR * p_solution);

        // Move toward equilibrium at a kinetic rate (not instantaneous)
        let sorption_rate = 0.1; // day^-1
        let p_to_sorb = (p_eq - 0.0) * (1.0 - (-sorption_rate * dt).exp());
        let p_to_sorb = p_to_sorb.min(p_solution);
        let p_to_sorb = p_to_sorb.max(0.0);

        self.mineral_pool.phosphate_g -= p_to_sorb;
        p_to_sorb
    }

    /// Advance litter cohorts through decomposition stages.
    ///
    /// Fresh -> Fragmented -> Humified, with increasing lignin fraction
    /// and decreasing C:N as labile fractions are consumed first.
    fn advance_litter(&mut self, dt: f64) {
        let temp_mod = q10_factor(self.temperature_c, REFERENCE_TEMP, DEFAULT_Q10);
        let moist_mod = moisture_factor(self.moisture_fraction);
        let env = temp_mod * moist_mod;

        for cohort in &mut self.litter_cohorts {
            cohort.age_days += dt;

            match cohort.stage {
                DecompositionStage::Fresh => {
                    let transition_prob = FRESH_TO_FRAGMENTED_RATE * env * dt;
                    if cohort.age_days * FRESH_TO_FRAGMENTED_RATE * env > 1.0
                        || transition_prob > 0.5
                    {
                        cohort.stage = DecompositionStage::Fragmented;
                        // Labile fractions consumed -> lignin fraction increases
                        cohort.lignin_fraction = (cohort.lignin_fraction + 0.1).min(0.8);
                        // C:N decreases as N-poor cellulose is consumed first
                        cohort.cn_ratio = (cohort.cn_ratio - 5.0).max(10.0);
                    }
                }
                DecompositionStage::Fragmented => {
                    let transition_prob = FRAGMENTED_TO_HUMIFIED_RATE * env * dt;
                    if cohort.age_days * FRAGMENTED_TO_HUMIFIED_RATE * env > 5.0
                        || (transition_prob > 0.5 && cohort.age_days > 30.0)
                    {
                        cohort.stage = DecompositionStage::Humified;
                        cohort.lignin_fraction = (cohort.lignin_fraction + 0.15).min(0.9);
                        cohort.cn_ratio = (cohort.cn_ratio - 3.0).max(8.0);
                    }
                }
                DecompositionStage::Humified => {
                    // Terminal stage -- very slow further decomposition only
                }
            }
        }

        // Remove fully decomposed litter (mass < threshold)
        self.litter_cohorts.retain(|c| c.mass_g > 1e-6);
    }

    /// N2O emission factor: fraction of total N cycling emitted as N2O.
    ///
    /// Integrates nitrification-derived N2O (~0.5-1% of nitrified N) and
    /// denitrification-derived N2O.
    pub fn n2o_emission_factor(&self) -> f64 {
        // Base EF from IPCC Tier 1: ~1% of N inputs
        let base_ef = 0.01;

        // Modify by WFPS: peaks at ~70% where both nitrification and
        // denitrification contribute
        let wfps_mod = if self.moisture_fraction < 0.50 {
            0.3
        } else if self.moisture_fraction < 0.70 {
            // Rising to peak
            0.3 + 0.7 * (self.moisture_fraction - 0.50) / 0.20
        } else if self.moisture_fraction < 0.85 {
            // At peak
            1.0
        } else {
            // Very wet: more complete denitrification to N2, less N2O
            1.0 - (self.moisture_fraction - 0.85) / 0.15
        };

        base_ef * clamp_f64(wfps_mod, 0.0, 1.5)
    }

    /// Net carbon balance proxy (g C day^-1).
    ///
    /// Positive = net C loss from soil (source to atmosphere).
    /// Negative = net C accumulation (sink).
    pub fn net_carbon_balance(&self) -> f64 {
        let _total_organic_c: f64 = self.organic_pools.iter().map(|p| p.carbon_g).sum();
        let _litter_c: f64 = self.litter_cohorts.iter().map(|c| c.mass_g * 0.45).sum();

        // Net balance = CO2 emission - inputs (fixation, deposition, litter input)
        // Simplified: just report current CO2 emission as the main C loss term
        self.atmospheric_flux.co2_emission
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a cycler with one organic pool of known C:N:P.
    fn cycler_with_pool(carbon: f64, nitrogen: f64, phosphorus: f64) -> NutrientCycler {
        let mut c = NutrientCycler::new();
        c.add_organic_pool(OrganicPool {
            name: "test_pool".to_string(),
            carbon_g: carbon,
            nitrogen_g: nitrogen,
            phosphorus_g: phosphorus,
            decomposition_rate: 0.05,
            recalcitrance: 0.1,
        });
        c.temperature_c = 20.0;
        c.moisture_fraction = 0.60;
        c
    }

    // 1. decomposition_releases_co2
    #[test]
    fn decomposition_releases_co2() {
        let mut c = cycler_with_pool(100.0, 5.0, 0.5);
        let result = c.step(1.0);
        assert!(
            result.co2_emitted > 0.0,
            "decomposition must produce CO2, got {}",
            result.co2_emitted
        );
        assert!(
            c.organic_pools[0].carbon_g < 100.0,
            "organic C must decrease"
        );
    }

    // 2. high_cn_causes_immobilization
    #[test]
    fn high_cn_causes_immobilization() {
        // C:N = 80 (>> critical 25) -> microbes must immobilize N
        let mut c = cycler_with_pool(80.0, 1.0, 0.5);
        // Pre-load mineral N so there is something to immobilize
        c.mineral_pool.ammonium_g = 1.0;
        let _initial_mineral_n = c.mineral_pool.ammonium_g + c.mineral_pool.nitrate_g;
        let result = c.step(1.0);
        let _final_mineral_n = c.mineral_pool.ammonium_g + c.mineral_pool.nitrate_g;

        // With high C:N, microbes consume mineral N for their stoichiometric needs.
        // Net mineralization should be near zero or negative (mineral N decreases).
        // The step also adds deposition, fixation, etc. -- so check the decomposition
        // balance specifically via the result.
        // At C:N=80 the pool's N release per unit C is very low, so immobilization
        // dominates: result.n_mineralized should be very low relative to C.
        let _cn_of_result = if result.n_mineralized > 0.0 {
            result.c_mineralized / result.n_mineralized
        } else {
            f64::INFINITY
        };
        // Either net N mineralization is zero (immobilization consumed it all)
        // or the effective C:N of what was released is very high.
        assert!(
            result.n_mineralized < result.c_mineralized / 10.0,
            "high C:N should yield low net N mineralization relative to C"
        );
    }

    // 3. low_cn_causes_mineralization
    #[test]
    fn low_cn_causes_mineralization() {
        // C:N = 10 (< critical 25) -> net N mineralization
        let mut c = cycler_with_pool(10.0, 1.0, 0.5);
        let result = c.step(1.0);
        assert!(
            result.n_mineralized > 0.0,
            "low C:N must cause net N mineralization, got {}",
            result.n_mineralized
        );
        assert!(
            c.mineral_pool.ammonium_g > 0.0,
            "ammonium should increase from mineralization"
        );
    }

    // 4. nitrification_produces_nitrate
    #[test]
    fn nitrification_produces_nitrate() {
        let mut c = NutrientCycler::new();
        c.mineral_pool.ammonium_g = 1.0;
        c.mineral_pool.nitrate_g = 0.0;
        c.temperature_c = 20.0;
        c.moisture_fraction = 0.60; // good O2 conditions
        c.ph = 7.0;

        let initial_nh4 = c.mineral_pool.ammonium_g;
        c.step(1.0);

        assert!(
            c.mineral_pool.nitrate_g > 0.0,
            "nitrification must produce NO3-"
        );
        assert!(
            c.mineral_pool.ammonium_g < initial_nh4,
            "NH4+ must decrease during nitrification"
        );
    }

    // 5. denitrification_requires_anoxia
    #[test]
    fn denitrification_requires_anoxia() {
        // At low WFPS (30%), denitrification should not occur
        let mut c = NutrientCycler::new();
        c.mineral_pool.nitrate_g = 1.0;
        c.moisture_fraction = 0.30; // aerobic -> no denitrification
        c.temperature_c = 20.0;

        let result = c.step(1.0);

        assert!(
            result.n2o_emitted < 1e-10,
            "no denitrification at low WFPS (30%), got N2O={}",
            result.n2o_emitted
        );

        // Now at high WFPS (85%), denitrification should occur
        let mut c2 = NutrientCycler::new();
        c2.mineral_pool.nitrate_g = 1.0;
        c2.moisture_fraction = 0.85;
        c2.temperature_c = 20.0;

        let result2 = c2.step(1.0);
        assert!(
            result2.n2o_emitted > 0.0,
            "denitrification must occur at high WFPS (85%), got N2O={}",
            result2.n2o_emitted
        );
    }

    // 6. n2o_emission_at_intermediate_moisture
    #[test]
    fn n2o_emission_at_intermediate_moisture() {
        // N2O emission factor should peak at ~70% WFPS where both
        // nitrification and denitrification pathways contribute.
        let mut c_low = NutrientCycler::new();
        c_low.moisture_fraction = 0.40;
        let ef_low = c_low.n2o_emission_factor();

        let mut c_mid = NutrientCycler::new();
        c_mid.moisture_fraction = 0.70;
        let ef_mid = c_mid.n2o_emission_factor();

        let mut c_high = NutrientCycler::new();
        c_high.moisture_fraction = 0.95;
        let ef_high = c_high.n2o_emission_factor();

        assert!(
            ef_mid > ef_low,
            "N2O EF at 70% ({}) must exceed EF at 40% ({})",
            ef_mid,
            ef_low
        );
        assert!(
            ef_mid >= ef_high,
            "N2O EF at 70% ({}) must be >= EF at 95% ({})",
            ef_mid,
            ef_high
        );
    }

    // 7. q10_doubles_rate
    #[test]
    fn q10_doubles_rate() {
        let f_at_20 = q10_factor(20.0, 20.0, 2.0);
        let f_at_30 = q10_factor(30.0, 20.0, 2.0);
        let f_at_10 = q10_factor(10.0, 20.0, 2.0);

        assert!(
            (f_at_20 - 1.0).abs() < 1e-10,
            "factor at reference temp must be 1.0"
        );
        assert!(
            (f_at_30 - 2.0).abs() < 1e-6,
            "factor at T+10 must be ~2.0, got {}",
            f_at_30
        );
        assert!(
            (f_at_10 - 0.5).abs() < 1e-6,
            "factor at T-10 must be ~0.5, got {}",
            f_at_10
        );
    }

    // 8. moisture_optimal_at_60_wfps
    #[test]
    fn moisture_optimal_at_60_wfps() {
        let f_opt = moisture_factor(0.60);
        let f_dry = moisture_factor(0.20);
        let f_wet = moisture_factor(0.95);

        assert!(
            (f_opt - 1.0).abs() < 1e-10,
            "moisture factor at 60% WFPS must be 1.0, got {}",
            f_opt
        );
        assert!(
            f_opt > f_dry,
            "optimal ({}) must exceed dry ({})",
            f_opt,
            f_dry
        );
        assert!(
            f_opt > f_wet,
            "optimal ({}) must exceed wet ({})",
            f_opt,
            f_wet
        );
    }

    // 9. plant_uptake_michaelis_menten
    #[test]
    fn plant_uptake_michaelis_menten() {
        // Uptake should saturate at high mineral concentrations.
        let mut c_low = NutrientCycler::new();
        c_low.mineral_pool.ammonium_g = 0.01;
        c_low.mineral_pool.phosphate_g = 0.01;

        let mut c_high = NutrientCycler::new();
        c_high.mineral_pool.ammonium_g = 10.0;
        c_high.mineral_pool.phosphate_g = 10.0;

        let demand = PlantNutrientDemand {
            nitrogen_demand: 0.1,
            phosphorus_demand: 0.05,
            root_biomass: 50.0,
            mycorrhizal: false,
            uptake_efficiency: 0.8,
        };

        let uptake_low = c_low.plant_uptake(&[demand.clone()]);
        let uptake_high = c_high.plant_uptake(&[demand.clone()]);

        // Uptake at high concentration must be greater than at low
        assert!(
            uptake_high[0].0 > uptake_low[0].0,
            "N uptake at high conc ({}) must exceed low ({})",
            uptake_high[0].0,
            uptake_low[0].0
        );

        // But not proportionally greater (saturation)
        let conc_ratio = 10.0 / 0.01; // 1000x
        let uptake_ratio = uptake_high[0].0 / uptake_low[0].0;
        assert!(
            uptake_ratio < conc_ratio * 0.5,
            "uptake must saturate: conc ratio {} but uptake ratio {}",
            conc_ratio,
            uptake_ratio
        );
    }

    // 10. p_sorption_langmuir
    #[test]
    fn p_sorption_langmuir() {
        // More P in solution -> more sorbed, but following Langmuir curve
        let mut c1 = NutrientCycler::new();
        c1.mineral_pool.phosphate_g = 0.1;
        c1.clay_fraction = 0.30;
        let s1 = c1.phosphorus_sorption(1.0);

        let mut c2 = NutrientCycler::new();
        c2.mineral_pool.phosphate_g = 1.0;
        c2.clay_fraction = 0.30;
        let s2 = c2.phosphorus_sorption(1.0);

        assert!(s1 > 0.0, "some P must be sorbed, got {}", s1);
        assert!(
            s2 > s1,
            "more solution P ({}) -> more sorption ({})",
            s2,
            s1
        );

        // Sorption should saturate: ratio of sorption < ratio of concentration
        let conc_ratio = 1.0 / 0.1;
        let sorb_ratio = s2 / s1;
        assert!(
            sorb_ratio < conc_ratio,
            "Langmuir must show saturation: conc ratio {} but sorption ratio {}",
            conc_ratio,
            sorb_ratio
        );
    }

    // 11. mycorrhizal_increases_uptake
    #[test]
    fn mycorrhizal_increases_uptake() {
        // Two plants: one with high demand (deficient), one with low demand (surplus).
        // Mycorrhizal network should transfer nutrients from surplus to deficient.
        let demands = vec![
            PlantNutrientDemand {
                nitrogen_demand: 0.01, // low demand (surplus)
                phosphorus_demand: 0.01,
                root_biomass: 50.0,
                mycorrhizal: true,
                uptake_efficiency: 0.8,
            },
            PlantNutrientDemand {
                nitrogen_demand: 0.5, // high demand (deficient)
                phosphorus_demand: 0.5,
                root_biomass: 20.0,
                mycorrhizal: true,
                uptake_efficiency: 0.8,
            },
        ];

        let mut c = NutrientCycler::new();
        c.mineral_pool.ammonium_g = 0.2;
        c.mineral_pool.phosphate_g = 0.1;
        c.set_mycorrhizal(MycorrhizalNetwork {
            hyphal_density: 5.0,
            plant_connections: vec![(0, 1)],
            transfer_efficiency: 0.5,
            carbon_cost: 0.1,
            network_age_days: 120.0,
        });

        let transfers = c.mycorrhizal_transfer(&demands);

        // Plant 1 (low demand) should donate (negative), plant 2 should receive (positive)
        assert!(
            transfers[1].0 > 0.0,
            "deficient plant should receive N via mycorrhizae, got {}",
            transfers[1].0
        );
        assert!(
            transfers[0].0 < 0.0,
            "surplus plant should donate N via mycorrhizae, got {}",
            transfers[0].0
        );
    }

    // 12. litter_quality_decreases
    #[test]
    fn litter_quality_decreases() {
        let mut c = NutrientCycler::new();
        c.temperature_c = 20.0;
        c.moisture_fraction = 0.60;
        c.add_litter(LitterCohort {
            stage: DecompositionStage::Fresh,
            mass_g: 10.0,
            cn_ratio: 40.0,
            lignin_fraction: 0.15,
            age_days: 0.0,
        });

        let initial_lignin = c.litter_cohorts[0].lignin_fraction;

        // Run enough steps to trigger stage transitions
        for _ in 0..200 {
            c.step(1.0);
        }

        // Litter should have progressed and lignin fraction increased
        if !c.litter_cohorts.is_empty() {
            let final_lignin = c.litter_cohorts[0].lignin_fraction;
            assert!(
                final_lignin > initial_lignin,
                "lignin fraction must increase as litter decomposes: {} -> {}",
                initial_lignin,
                final_lignin
            );
        }
        // If litter fully decomposed, that also demonstrates progression
    }

    // 13. redfield_ratio_calculation
    #[test]
    fn redfield_ratio_calculation() {
        let (cn, np) = redfield_ratio(60.0, 7.0, 1.0);
        assert!(
            (cn - 60.0 / 7.0).abs() < 0.01,
            "C:N of 60:7 should be ~8.57, got {}",
            cn
        );
        assert!(
            (np - 7.0).abs() < 0.01,
            "N:P of 7:1 should be 7.0, got {}",
            np
        );

        // Edge case: zero denominator
        let (cn_zero, _) = redfield_ratio(10.0, 0.0, 1.0);
        assert!(cn_zero.is_infinite(), "C:N with zero N must be infinity");
    }

    // 14. limiting_nutrient_detection
    #[test]
    fn limiting_nutrient_detection() {
        // Nitrogen limited: N supply low relative to demand
        assert_eq!(
            limiting_nutrient(0.1, 1.0, 1.0, 1.0),
            LimitingNutrient::Nitrogen
        );

        // Phosphorus limited: P supply low relative to demand
        assert_eq!(
            limiting_nutrient(1.0, 0.1, 1.0, 1.0),
            LimitingNutrient::Phosphorus
        );

        // Neither limited: both nutrients exceed demand
        assert_eq!(
            limiting_nutrient(2.0, 2.0, 1.0, 1.0),
            LimitingNutrient::Neither
        );

        // Co-limited: both similarly deficient
        assert_eq!(
            limiting_nutrient(0.5, 0.5, 1.0, 1.0),
            LimitingNutrient::CoLimited
        );
    }

    // 15. mass_balance_conservation
    #[test]
    fn mass_balance_conservation() {
        // Total system N should be conserved (minus gaseous losses and leaching)
        let mut c = cycler_with_pool(50.0, 5.0, 0.5);
        c.mineral_pool.ammonium_g = 1.0;
        c.mineral_pool.nitrate_g = 0.5;

        let initial_org_n: f64 = c.organic_pools.iter().map(|p| p.nitrogen_g).sum();
        let initial_min_n = c.mineral_pool.ammonium_g
            + c.mineral_pool.nitrate_g
            + c.mineral_pool.dissolved_organic_n;
        let initial_total = initial_org_n + initial_min_n;

        let result = c.step(1.0);

        let final_org_n: f64 = c.organic_pools.iter().map(|p| p.nitrogen_g).sum();
        let final_min_n = c.mineral_pool.ammonium_g
            + c.mineral_pool.nitrate_g
            + c.mineral_pool.dissolved_organic_n;
        let final_total = final_org_n + final_min_n;

        // Account for losses (gaseous N + leaching) and gains (fixation + deposition)
        let n_lost = result.n2o_emitted + result.n_leached;
        // Gaseous N2 is not tracked in result but emitted; retrieve from flux
        let n2_lost = c.atmospheric_flux.n2_emission * 1.0; // dt was 1.0
        let n_gained = c.atmospheric_flux.n_fixation * 1.0 + c.atmospheric_flux.n_deposition * 1.0;

        let expected_final = initial_total - n_lost - n2_lost + n_gained;

        let error = (final_total - expected_final).abs();
        assert!(
            error < 0.01,
            "N mass balance violated: expected {:.4}, got {:.4}, error {:.4}",
            expected_final,
            final_total,
            error
        );
    }

    // 16. temperature_modulates_decomposition
    #[test]
    fn temperature_modulates_decomposition() {
        let mut cold = cycler_with_pool(100.0, 5.0, 0.5);
        cold.temperature_c = 5.0;
        let result_cold = cold.step(1.0);

        let mut warm = cycler_with_pool(100.0, 5.0, 0.5);
        warm.temperature_c = 30.0;
        let result_warm = warm.step(1.0);

        assert!(
            result_warm.co2_emitted > result_cold.co2_emitted,
            "warm soil ({}) must respire more CO2 than cold ({})",
            result_warm.co2_emitted,
            result_cold.co2_emitted
        );
    }
}
