//! Drosophila melanogaster population dynamics with temperature-dependent lifecycle.
//!
//! Models the complete Drosophila life cycle -- embryo, three larval instars, pupa,
//! and reproducing adult -- using Sharpe-Schoolfield temperature-dependent development
//! rates. Females mate once, store sperm (Wolfner 2002), and oviposit on fermenting
//! fruit substrates (Dweck et al. 2013). Population growth emerges from the interplay
//! of food availability, temperature, and individual energy budgets.
//!
//! # Literature Sources
//!
//! - Ashburner et al. (2005) *Drosophila: A Laboratory Handbook*
//! - Schoolfield et al. (1981) *J. Theor. Biol.* 88:719-731 (temperature dependence)
//! - Wolfner (2002) *Insect Biochem. Mol. Biol.* 32:1519-1530 (sperm storage)
//! - Dweck et al. (2013) *Cell Rep.* 4:615-624 (oviposition site selection)
//! - Lehmann & Dickinson (2000) *J. Exp. Biol.* 203:1613-1624 (metabolic rates)
//! - Wigglesworth (1949) *J. Exp. Biol.* 26:150-163 (energy reserves)

use rand::prelude::*;

// ============================================================================
// Energy & Timing Constants (consistent with drosophila.rs)
// ============================================================================

/// Total energy reserves in uJ: glycogen ~500 uJ + fat ~3200 uJ (Wigglesworth 1949).
pub const FLY_ENERGY_MAX: f32 = 3700.0;

/// Minimum energy fraction required for reproduction (30% of max).
const REPRODUCTION_ENERGY_THRESHOLD_FRAC: f32 = 0.30;

/// Energy cost per egg in uJ.
/// Drosophila females allocate ~15% of metabolic budget to reproduction;
/// spread over ~400 eggs across 30 days ≈ 1.4 uJ/egg, rounded to 2 uJ.
const EGG_ENERGY_COST_UJ: f32 = 2.0;

/// Energy cost of mating in uJ (courtship + copulation, ~5 min at elevated metabolic rate).
const MATING_ENERGY_COST_UJ: f32 = 5.0;

/// Peak egg production: ~50-75 eggs/day at 25C (Ashburner 2005). We use 60 as midpoint.
const PEAK_EGGS_PER_DAY: f32 = 60.0;

/// Total lifetime eggs per female (~400, Ashburner 2005).
const LIFETIME_EGGS: u32 = 400;

/// Hours after eclosion before oviposition begins (1-2 days, use 36h midpoint).
const OVIPOSITION_DELAY_HOURS: f32 = 36.0;

/// Minimum interval between oviposition bouts (hours).
const MIN_OVIPOSITION_INTERVAL_HOURS: f32 = 2.0;

/// Eggs per cluster: 5-20 (Markow et al. 2009).
const MIN_EGGS_PER_CLUSTER: u8 = 5;
const MAX_EGGS_PER_CLUSTER: u8 = 20;

/// Maximum adult lifespan in days (~60 days under good conditions, Ashburner 2005).
const MAX_ADULT_LIFESPAN_DAYS: f32 = 60.0;

/// Proximity for mating in mm (courtship range).
const MATING_PROXIMITY_MM: f32 = 5.0;

/// Proximity to fruit for oviposition in mm.
const OVIPOSITION_PROXIMITY_MM: f32 = 8.0;

/// Larval feeding energy gain per hour in uJ (scales with substrate quality).
const LARVAL_FEED_RATE_UJ_PER_HOUR: f32 = 50.0;

/// Basal metabolic drain for all lifecycle stages in µJ/hour.
/// Scaled to the energy budget (~0.5% of reserves/hour), allowing
/// survival ~8 days without food (Ballard et al. 2008).
const BASAL_COST_UJ_PER_HOUR: f32 = 20.0;

// ============================================================================
// Temperature-Dependent Development (Sharpe-Schoolfield)
// ============================================================================

/// Reference temperature for development rate parameterisation (25 C = 298.15 K).
const T_REF_K: f32 = 298.15;

/// Universal gas constant in J/(mol*K).
const R_GAS: f32 = 8.314;

/// Lower developmental threshold (~12 C) below which rate drops to zero.
const T_LOWER_DEV_C: f32 = 12.0;

/// Upper thermal limit (~32 C) above which development ceases.
const T_UPPER_DEV_C: f32 = 32.0;

/// Enthalpy of activation for Drosophila development (Schoolfield et al. 1981).
/// ~75 kJ/mol is a commonly fitted value for Drosophila egg-to-adult.
const DELTA_H_A: f32 = 75_000.0; // J/mol

/// Stage durations at the reference temperature (25 C) in hours.
/// Ashburner et al. (2005):
/// - Embryo: ~22 h
/// - L1: ~24 h, L2: ~24 h, L3: ~48 h
/// - Pupa: ~96 h
const EMBRYO_DURATION_H_25C: f32 = 22.0;
const L1_DURATION_H_25C: f32 = 24.0;
const L2_DURATION_H_25C: f32 = 24.0;
const L3_DURATION_H_25C: f32 = 48.0;
const PUPA_DURATION_H_25C: f32 = 96.0;

/// Compute the Sharpe-Schoolfield temperature scaling factor relative to T_REF.
///
/// Returns a value in [0, 1] at T_REF, >1 if warmer (up to T_UPPER), and 0
/// outside the viable range [T_LOWER_DEV_C, T_UPPER_DEV_C].
///
/// The simplified form used here is the Arrhenius portion of Schoolfield:
///   rate(T) / rate(T_ref) = exp( ΔH_A/R * (1/T_ref - 1/T) )
/// clamped to zero outside the viable thermal window.
fn temperature_rate_factor(temp_c: f32) -> f32 {
    if temp_c <= T_LOWER_DEV_C || temp_c >= T_UPPER_DEV_C {
        return 0.0;
    }
    let t_k = temp_c + 273.15;
    let exponent = (DELTA_H_A / R_GAS) * (1.0 / T_REF_K - 1.0 / t_k);
    exponent.exp().clamp(0.0, 10.0) // cap at 10x to prevent runaway
}

/// How many "reference hours" of development elapse per real hour at given temperature.
///
/// At 25 C this returns 1.0. At 18 C it returns ~0.35 (development takes ~3x longer).
/// Below 12 C or above 32 C it returns 0.0.
fn effective_dev_hours(dt_hours: f32, temp_c: f32) -> f32 {
    dt_hours * temperature_rate_factor(temp_c)
}

// ============================================================================
// Lifecycle Stages
// ============================================================================

/// Lifecycle stage of an individual fly.
///
/// Development times are measured in "effective hours at 25 C" so that
/// temperature scaling only affects the rate at which age accumulates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FlyLifeStage {
    /// Embryo (~22h at 25C). `age_hours` counts effective development hours.
    Embryo { age_hours: f32 },
    /// Larval instar 1-3. Instar 1: 0-24h, 2: 24-48h, 3: 48-96h (effective hours).
    Larva { instar: u8, age_hours: f32 },
    /// Pupa (~96h at 25C effective hours).
    Pupa { age_hours: f32 },
    /// Adult with age in effective days (for lifespan tracking).
    Adult { age_days: f32 },
}

/// Biological sex. Females store sperm after a single mating (Wolfner 2002).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlySex {
    Male,
    Female,
}

// ============================================================================
// Individual Fly
// ============================================================================

/// A single Drosophila individual tracked through its lifecycle.
pub struct Fly {
    /// Unique identifier within the population.
    pub id: u32,
    /// Biological sex.
    pub sex: FlySex,
    /// Current lifecycle stage.
    pub stage: FlyLifeStage,
    /// Position (x, y, z) in mm.
    pub position: (f32, f32, f32),
    /// Energy in uJ.
    pub energy: f32,
    /// Whether this female has mated (males: always false).
    pub mated: bool,
    /// Remaining egg capacity (females only; 0 for males).
    pub eggs_remaining: u32,
    /// Hours since last oviposition event.
    pub time_since_last_oviposition: f32,
    /// Whether this individual is alive.
    alive: bool,
}

impl Fly {
    /// Create a new adult fly at a given position.
    pub fn new_adult(id: u32, sex: FlySex, position: (f32, f32, f32)) -> Self {
        Self {
            id,
            sex,
            stage: FlyLifeStage::Adult { age_days: 2.0 }, // past oviposition delay for immediate readiness
            position,
            energy: FLY_ENERGY_MAX * 0.8,
            mated: false,
            eggs_remaining: if sex == FlySex::Female {
                LIFETIME_EGGS
            } else {
                0
            },
            time_since_last_oviposition: OVIPOSITION_DELAY_HOURS, // ready immediately for testing
            alive: true,
        }
    }

    /// Create a new fly at embryo stage (from an egg cluster hatch).
    fn new_embryo(id: u32, sex: FlySex, position: (f32, f32, f32)) -> Self {
        Self {
            id,
            sex,
            stage: FlyLifeStage::Embryo { age_hours: 0.0 },
            position,
            energy: FLY_ENERGY_MAX * 0.1, // yolk reserves
            mated: false,
            eggs_remaining: if sex == FlySex::Female {
                LIFETIME_EGGS
            } else {
                0
            },
            time_since_last_oviposition: 0.0,
            alive: true,
        }
    }

    /// Whether this fly is an adult.
    pub fn is_adult(&self) -> bool {
        matches!(self.stage, FlyLifeStage::Adult { .. })
    }

    /// Whether this female can currently lay eggs.
    fn can_oviposit(&self) -> bool {
        self.sex == FlySex::Female
            && self.mated
            && self.eggs_remaining > 0
            && self.energy > FLY_ENERGY_MAX * REPRODUCTION_ENERGY_THRESHOLD_FRAC
            && self.time_since_last_oviposition >= MIN_OVIPOSITION_INTERVAL_HOURS
            && self.is_adult()
            && matches!(self.stage, FlyLifeStage::Adult { age_days } if age_days >= OVIPOSITION_DELAY_HOURS / 24.0)
    }
}

// ============================================================================
// Egg Cluster
// ============================================================================

/// A clutch of eggs deposited on a substrate (typically fermenting fruit).
pub struct EggCluster {
    /// Position (x, y) in mm on the substrate surface.
    pub position: (f32, f32),
    /// Number of eggs in the clutch.
    pub count: u8,
    /// Age of the clutch in effective development hours.
    pub age_hours: f32,
    /// Quality of the substrate where eggs were laid [0, 1].
    pub substrate_quality: f32,
}

// ============================================================================
// Population Manager
// ============================================================================

/// Manages the entire Drosophila population: lifecycle progression, mating,
/// oviposition, egg hatching, and mortality.
pub struct FlyPopulation {
    /// All individual flies (immature and adult).
    pub flies: Vec<Fly>,
    /// All active egg clusters.
    pub egg_clusters: Vec<EggCluster>,
    /// Next unique ID to assign.
    pub next_id: u32,
    /// RNG for stochastic processes.
    rng: StdRng,
    /// Positions of flies that eclosed (pupa → adult) this step.
    pub(crate) eclosed_positions: Vec<(f32, f32, f32)>,
}

impl FlyPopulation {
    /// Create a new empty population.
    pub fn new(seed: u64) -> Self {
        Self {
            flies: Vec::new(),
            egg_clusters: Vec::new(),
            next_id: 0,
            rng: StdRng::seed_from_u64(seed),
            eclosed_positions: Vec::new(),
        }
    }

    /// Seed the population with `n` adult flies at random positions within bounds.
    pub fn seed_adults(&mut self, n: u32, bounds: (f32, f32, f32)) {
        for _ in 0..n {
            let sex = if self.rng.gen_bool(0.5) {
                FlySex::Male
            } else {
                FlySex::Female
            };
            let pos = (
                self.rng.gen_range(0.0..bounds.0),
                self.rng.gen_range(0.0..bounds.1),
                0.0,
            );
            let fly = Fly::new_adult(self.next_id, sex, pos);
            self.next_id += 1;
            self.flies.push(fly);
        }
    }

    /// Add a specific fly to the population.
    pub fn add_fly(&mut self, fly: Fly) {
        self.flies.push(fly);
    }

    /// Drain eclosed positions (pupa → adult transitions) accumulated this step.
    pub fn drain_eclosed(&mut self) -> Vec<(f32, f32, f32)> {
        std::mem::take(&mut self.eclosed_positions)
    }

    /// Count of living flies at each stage.
    pub fn stage_census(&self) -> StageCensus {
        let mut c = StageCensus::default();
        for f in &self.flies {
            if !f.alive {
                continue;
            }
            match f.stage {
                FlyLifeStage::Embryo { .. } => c.embryos += 1,
                FlyLifeStage::Larva { .. } => c.larvae += 1,
                FlyLifeStage::Pupa { .. } => c.pupae += 1,
                FlyLifeStage::Adult { .. } => c.adults += 1,
            }
        }
        c.egg_clusters = self.egg_clusters.len() as u32;
        c.total_eggs = self.egg_clusters.iter().map(|e| e.count as u32).sum();
        c
    }

    // ========================================================================
    // Main simulation step
    // ========================================================================

    /// Advance the population by `dt_hours` at the given environmental conditions.
    ///
    /// - `temperature`: ambient temperature in degrees C
    /// - `fruit_positions`: (x, y, z) positions of available food sources
    /// - `humidity`: relative humidity [0, 1] (affects egg viability)
    pub fn step(
        &mut self,
        dt_hours: f32,
        temperature: f32,
        fruit_positions: &[(f32, f32, f32)],
        humidity: f32,
    ) {
        self.eclosed_positions.clear();
        self.advance_development(dt_hours, temperature);
        self.feed_immatures(dt_hours, temperature, fruit_positions);
        self.attempt_mating();
        self.oviposit(fruit_positions);
        self.advance_eggs(dt_hours, temperature);
        self.hatch_eggs(temperature, humidity);
        self.remove_dead();
    }

    // ========================================================================
    // Development
    // ========================================================================

    /// Advance lifecycle stages by temperature-scaled time.
    fn advance_development(&mut self, dt_hours: f32, temperature: f32) {
        let dev_dt = effective_dev_hours(dt_hours, temperature);
        let dt_days = dev_dt / 24.0;

        for fly in self.flies.iter_mut() {
            if !fly.alive {
                continue;
            }

            // Basal metabolism drains energy for all stages.
            fly.energy -= BASAL_COST_UJ_PER_HOUR * dt_hours;
            if fly.energy <= 0.0 {
                fly.alive = false;
                continue;
            }

            match fly.stage {
                FlyLifeStage::Embryo { age_hours } => {
                    let new_age = age_hours + dev_dt;
                    if new_age >= EMBRYO_DURATION_H_25C {
                        fly.stage = FlyLifeStage::Larva {
                            instar: 1,
                            age_hours: 0.0,
                        };
                    } else {
                        fly.stage = FlyLifeStage::Embryo {
                            age_hours: new_age,
                        };
                    }
                }
                FlyLifeStage::Larva { instar, age_hours } => {
                    let new_age = age_hours + dev_dt;
                    let duration = match instar {
                        1 => L1_DURATION_H_25C,
                        2 => L2_DURATION_H_25C,
                        _ => L3_DURATION_H_25C,
                    };
                    if new_age >= duration {
                        if instar < 3 {
                            fly.stage = FlyLifeStage::Larva {
                                instar: instar + 1,
                                age_hours: 0.0,
                            };
                        } else {
                            // L3 -> Pupa
                            fly.stage = FlyLifeStage::Pupa { age_hours: 0.0 };
                        }
                    } else {
                        fly.stage = FlyLifeStage::Larva {
                            instar,
                            age_hours: new_age,
                        };
                    }
                }
                FlyLifeStage::Pupa { age_hours } => {
                    let new_age = age_hours + dev_dt;
                    if new_age >= PUPA_DURATION_H_25C {
                        // Eclosion: pupa -> adult
                        fly.stage = FlyLifeStage::Adult { age_days: 0.0 };
                        fly.energy = FLY_ENERGY_MAX * 0.6; // eclosion reserves
                        self.eclosed_positions.push(fly.position);
                    } else {
                        fly.stage = FlyLifeStage::Pupa {
                            age_hours: new_age,
                        };
                    }
                }
                FlyLifeStage::Adult { age_days } => {
                    let new_age = age_days + dt_days;
                    fly.stage = FlyLifeStage::Adult { age_days: new_age };
                    fly.time_since_last_oviposition += dt_hours;

                    // Senescence: mortality increases sharply past ~40 days
                    if new_age > MAX_ADULT_LIFESPAN_DAYS {
                        fly.alive = false;
                    }
                }
            }
        }
    }

    /// Feed immature stages (larvae) from nearby fruit substrates.
    fn feed_immatures(
        &mut self,
        dt_hours: f32,
        _temperature: f32,
        fruit_positions: &[(f32, f32, f32)],
    ) {
        for fly in self.flies.iter_mut() {
            if !fly.alive {
                continue;
            }
            // Only larvae feed (embryos use yolk, pupae don't feed)
            if !matches!(fly.stage, FlyLifeStage::Larva { .. }) {
                continue;
            }
            // Check proximity to any fruit
            let near_food = fruit_positions.iter().any(|fp| {
                let dx = fly.position.0 - fp.0;
                let dy = fly.position.1 - fp.1;
                (dx * dx + dy * dy).sqrt() < OVIPOSITION_PROXIMITY_MM * 2.0
            });
            if near_food {
                fly.energy =
                    (fly.energy + LARVAL_FEED_RATE_UJ_PER_HOUR * dt_hours).min(FLY_ENERGY_MAX);
            }
        }
    }

    // ========================================================================
    // Mating
    // ========================================================================

    /// Attempt mating between nearby unmated adults with sufficient energy.
    ///
    /// Simplified courtship model: male within MATING_PROXIMITY_MM of an unmated
    /// female, both with >30% energy, results in successful mating. In reality
    /// this involves wing song (Bennet-Clark & Ewing 1969) and cVA pheromone
    /// signalling, but we collapse to an energy + proximity check.
    fn attempt_mating(&mut self) {
        // Collect indices of eligible males and females
        let males: Vec<usize> = self
            .flies
            .iter()
            .enumerate()
            .filter(|(_, f)| {
                f.alive
                    && f.sex == FlySex::Male
                    && f.is_adult()
                    && f.energy > FLY_ENERGY_MAX * REPRODUCTION_ENERGY_THRESHOLD_FRAC
            })
            .map(|(i, _)| i)
            .collect();

        let females: Vec<usize> = self
            .flies
            .iter()
            .enumerate()
            .filter(|(_, f)| {
                f.alive
                    && f.sex == FlySex::Female
                    && f.is_adult()
                    && !f.mated
                    && f.energy > FLY_ENERGY_MAX * REPRODUCTION_ENERGY_THRESHOLD_FRAC
            })
            .map(|(i, _)| i)
            .collect();

        // For each unmated female, find the closest eligible male
        let mut mated_pairs: Vec<(usize, usize)> = Vec::new();
        let mut used_males = std::collections::HashSet::new();

        for &fi in &females {
            let fp = self.flies[fi].position;
            let mut best: Option<(usize, f32)> = None;
            for &mi in &males {
                if used_males.contains(&mi) {
                    continue;
                }
                let mp = self.flies[mi].position;
                let dist = ((fp.0 - mp.0).powi(2)
                    + (fp.1 - mp.1).powi(2)
                    + (fp.2 - mp.2).powi(2))
                .sqrt();
                if dist <= MATING_PROXIMITY_MM {
                    if best.map_or(true, |(_, bd)| dist < bd) {
                        best = Some((mi, dist));
                    }
                }
            }
            if let Some((mi, _)) = best {
                mated_pairs.push((fi, mi));
                used_males.insert(mi);
            }
        }

        // Execute matings
        for (fi, mi) in mated_pairs {
            self.flies[fi].mated = true;
            self.flies[fi].energy -= MATING_ENERGY_COST_UJ;
            self.flies[mi].energy -= MATING_ENERGY_COST_UJ;
        }
    }

    // ========================================================================
    // Oviposition
    // ========================================================================

    /// Females lay eggs near fermenting fruit if conditions are met.
    ///
    /// Oviposition site selection: females strongly prefer fermenting fruit
    /// over other substrates (Dweck et al. 2013, Or56a olfactory receptor).
    fn oviposit(&mut self, fruit_positions: &[(f32, f32, f32)]) {
        if fruit_positions.is_empty() {
            return;
        }

        let mut new_clusters: Vec<EggCluster> = Vec::new();

        for fly in self.flies.iter_mut() {
            if !fly.can_oviposit() {
                continue;
            }

            // Find nearest fruit within oviposition range
            let mut closest: Option<(usize, f32)> = None;
            for (i, fp) in fruit_positions.iter().enumerate() {
                let dist = ((fly.position.0 - fp.0).powi(2)
                    + (fly.position.1 - fp.1).powi(2))
                .sqrt();
                if dist <= OVIPOSITION_PROXIMITY_MM {
                    if closest.map_or(true, |(_, cd)| dist < cd) {
                        closest = Some((i, dist));
                    }
                }
            }

            if let Some((fi, _)) = closest {
                // Determine clutch size based on energy and remaining eggs
                let max_for_energy = ((fly.energy
                    - FLY_ENERGY_MAX * REPRODUCTION_ENERGY_THRESHOLD_FRAC)
                    / EGG_ENERGY_COST_UJ) as u8;
                let max_for_rate =
                    (PEAK_EGGS_PER_DAY / 24.0 * MIN_OVIPOSITION_INTERVAL_HOURS) as u8;
                let clutch_size = max_for_energy
                    .min(max_for_rate)
                    .min(fly.eggs_remaining as u8)
                    .min(MAX_EGGS_PER_CLUSTER)
                    .max(MIN_EGGS_PER_CLUSTER.min(fly.eggs_remaining as u8));

                if clutch_size == 0 {
                    continue;
                }

                let fp = fruit_positions[fi];
                new_clusters.push(EggCluster {
                    position: (fp.0, fp.1),
                    count: clutch_size,
                    age_hours: 0.0,
                    substrate_quality: 1.0, // assume good quality fruit
                });

                fly.eggs_remaining = fly.eggs_remaining.saturating_sub(clutch_size as u32);
                fly.energy -= clutch_size as f32 * EGG_ENERGY_COST_UJ;
                fly.time_since_last_oviposition = 0.0;
            }
        }

        self.egg_clusters.extend(new_clusters);
    }

    // ========================================================================
    // Egg Development & Hatching
    // ========================================================================

    /// Advance egg cluster ages by temperature-scaled development time.
    fn advance_eggs(&mut self, dt_hours: f32, temperature: f32) {
        let dev_dt = effective_dev_hours(dt_hours, temperature);
        for cluster in self.egg_clusters.iter_mut() {
            cluster.age_hours += dev_dt;
        }
    }

    /// Hatch egg clusters that have completed embryonic development.
    ///
    /// Humidity affects viability: below 40% RH, egg desiccation kills ~50%.
    fn hatch_eggs(&mut self, _temperature: f32, humidity: f32) {
        let mut new_flies: Vec<Fly> = Vec::new();
        let survival_rate = if humidity < 0.4 { 0.5 } else { 0.95 };

        self.egg_clusters.retain(|cluster| {
            if cluster.age_hours >= EMBRYO_DURATION_H_25C {
                // Hatch: create individual larvae
                for _ in 0..cluster.count {
                    if self.rng.gen::<f32>() < survival_rate {
                        let sex = if self.rng.gen_bool(0.5) {
                            FlySex::Male
                        } else {
                            FlySex::Female
                        };
                        let fly = Fly::new_embryo(
                            self.next_id,
                            sex,
                            (cluster.position.0, cluster.position.1, 0.0),
                        );
                        // They hatch directly into L1 since the egg duration already elapsed
                        let mut larva = fly;
                        larva.stage = FlyLifeStage::Larva {
                            instar: 1,
                            age_hours: 0.0,
                        };
                        new_flies.push(larva);
                        self.next_id += 1;
                    }
                }
                false // remove hatched cluster
            } else {
                true // keep unhatched cluster
            }
        });

        self.flies.extend(new_flies);
    }

    // ========================================================================
    // Mortality
    // ========================================================================

    /// Remove dead flies from the population.
    fn remove_dead(&mut self) {
        self.flies.retain(|f| f.alive);
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /// Total count of living individuals (all stages).
    pub fn total_alive(&self) -> usize {
        self.flies.iter().filter(|f| f.alive).count()
    }

    /// Count of living adults only.
    pub fn adult_count(&self) -> usize {
        self.flies
            .iter()
            .filter(|f| f.alive && f.is_adult())
            .count()
    }

    /// Count of living females that have mated.
    pub fn mated_female_count(&self) -> usize {
        self.flies
            .iter()
            .filter(|f| f.alive && f.sex == FlySex::Female && f.mated)
            .count()
    }
}

// ============================================================================
// Census
// ============================================================================

/// Snapshot of population distribution across lifecycle stages.
#[derive(Clone, Debug, Default)]
pub struct StageCensus {
    pub embryos: u32,
    pub larvae: u32,
    pub pupae: u32,
    pub adults: u32,
    pub egg_clusters: u32,
    pub total_eggs: u32,
}

impl StageCensus {
    pub fn total_individuals(&self) -> u32 {
        self.embryos + self.larvae + self.pupae + self.adults
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a small population with specified adult males and females
    /// all co-located at origin near a fruit source.
    fn test_population(males: u32, females: u32) -> FlyPopulation {
        let mut pop = FlyPopulation::new(42);
        for _ in 0..males {
            let fly = Fly::new_adult(pop.next_id, FlySex::Male, (5.0, 5.0, 0.0));
            pop.next_id += 1;
            pop.flies.push(fly);
        }
        for _ in 0..females {
            let fly = Fly::new_adult(pop.next_id, FlySex::Female, (5.0, 5.0, 0.0));
            pop.next_id += 1;
            pop.flies.push(fly);
        }
        pop
    }

    #[test]
    fn egg_development_temperature_dependent() {
        // Eggs develop faster at 25C than at 18C.
        let mut pop_warm = FlyPopulation::new(1);
        pop_warm.egg_clusters.push(EggCluster {
            position: (5.0, 5.0),
            count: 10,
            age_hours: 0.0,
            substrate_quality: 1.0,
        });

        let mut pop_cool = FlyPopulation::new(2);
        pop_cool.egg_clusters.push(EggCluster {
            position: (5.0, 5.0),
            count: 10,
            age_hours: 0.0,
            substrate_quality: 1.0,
        });

        let _fruit = vec![(5.0_f32, 5.0_f32, 0.0_f32)];

        // Advance both by 24 real hours
        for _ in 0..24 {
            pop_warm.advance_eggs(1.0, 25.0);
            pop_cool.advance_eggs(1.0, 18.0);
        }

        // At 25C, 24 effective hours should have elapsed (rate factor = 1.0)
        assert!(
            pop_warm.egg_clusters[0].age_hours > 23.0,
            "Warm eggs should have ~24 eff hours, got {}",
            pop_warm.egg_clusters[0].age_hours
        );
        // At 18C, rate factor < 1 so fewer effective hours
        assert!(
            pop_cool.egg_clusters[0].age_hours < 20.0,
            "Cool eggs should have <20 eff hours, got {}",
            pop_cool.egg_clusters[0].age_hours
        );
        assert!(
            pop_warm.egg_clusters[0].age_hours > pop_cool.egg_clusters[0].age_hours,
            "Warm eggs should develop faster than cool eggs"
        );

        // Now hatch the warm one (should be ready after 24h at 25C since embryo takes 22h)
        pop_warm.hatch_eggs(25.0, 0.7);
        assert!(
            pop_warm.egg_clusters.is_empty(),
            "Warm eggs should have hatched"
        );
        assert!(
            !pop_warm.flies.is_empty(),
            "Hatched eggs should produce larvae"
        );

        // Cool one should NOT have hatched yet
        pop_cool.hatch_eggs(18.0, 0.7);
        assert!(
            !pop_cool.egg_clusters.is_empty(),
            "Cool eggs should not have hatched yet (only {} eff hours)",
            pop_cool.egg_clusters[0].age_hours
        );
    }

    #[test]
    fn larval_instars_progress_sequentially() {
        // Create a single L1 larva and advance it through all stages to adult.
        let mut pop = FlyPopulation::new(7);
        let mut fly = Fly::new_embryo(0, FlySex::Female, (5.0, 5.0, 0.0));
        fly.stage = FlyLifeStage::Larva {
            instar: 1,
            age_hours: 0.0,
        };
        fly.energy = FLY_ENERGY_MAX; // full energy so it doesn't starve
        pop.next_id = 1;
        pop.flies.push(fly);

        let fruit = vec![(5.0, 5.0, 0.0)];

        // Advance in 1-hour steps at 25C (rate factor = 1.0)
        // L1: 24h, L2: 24h, L3: 48h, Pupa: 96h = total 192h
        let mut stage_transitions = Vec::new();
        let mut prev_stage = format!("{:?}", pop.flies[0].stage);

        for hour in 0..250 {
            pop.advance_development(1.0, 25.0);
            pop.feed_immatures(1.0, 25.0, &fruit);

            if pop.flies.is_empty() {
                break;
            }

            let current = format!("{:?}", pop.flies[0].stage);
            if current != prev_stage {
                stage_transitions.push((hour, current.clone()));
                prev_stage = current;
            }
        }

        // Verify sequential transitions: L1 -> L2 -> L3 -> Pupa -> Adult
        assert!(
            stage_transitions.len() >= 4,
            "Expected at least 4 transitions (L1->L2->L3->Pupa->Adult), got {}: {:?}",
            stage_transitions.len(),
            stage_transitions
        );

        // Check ordering: L2 before L3, L3 before Pupa, Pupa before Adult
        let has_l2 = stage_transitions
            .iter()
            .any(|(_, s)| s.contains("instar: 2"));
        let has_l3 = stage_transitions
            .iter()
            .any(|(_, s)| s.contains("instar: 3"));
        let has_pupa = stage_transitions.iter().any(|(_, s)| s.contains("Pupa"));
        let has_adult = stage_transitions
            .iter()
            .any(|(_, s)| s.contains("Adult"));

        assert!(has_l2, "Should transition through L2");
        assert!(has_l3, "Should transition through L3");
        assert!(has_pupa, "Should transition through Pupa");
        assert!(has_adult, "Should eclose to Adult");

        // Verify order: L2 time < L3 time < Pupa time < Adult time
        let l2_time = stage_transitions
            .iter()
            .find(|(_, s)| s.contains("instar: 2"))
            .unwrap()
            .0;
        let l3_time = stage_transitions
            .iter()
            .find(|(_, s)| s.contains("instar: 3"))
            .unwrap()
            .0;
        let pupa_time = stage_transitions
            .iter()
            .find(|(_, s)| s.contains("Pupa"))
            .unwrap()
            .0;
        let adult_time = stage_transitions
            .iter()
            .find(|(_, s)| s.contains("Adult"))
            .unwrap()
            .0;

        assert!(l2_time < l3_time, "L2 before L3");
        assert!(l3_time < pupa_time, "L3 before Pupa");
        assert!(pupa_time < adult_time, "Pupa before Adult");
    }

    #[test]
    fn female_oviposition_requires_energy() {
        // A mated female with low energy should NOT lay eggs.
        let mut pop = test_population(0, 1);
        pop.flies[0].mated = true;
        pop.flies[0].energy = FLY_ENERGY_MAX * 0.1; // well below 30% threshold

        let fruit = vec![(5.0, 5.0, 0.0)];
        pop.oviposit(&fruit);

        assert!(
            pop.egg_clusters.is_empty(),
            "Low-energy female should not oviposit"
        );

        // Now give her enough energy
        pop.flies[0].energy = FLY_ENERGY_MAX * 0.8;
        pop.oviposit(&fruit);

        assert!(
            !pop.egg_clusters.is_empty(),
            "Well-fed mated female should oviposit near fruit"
        );
    }

    #[test]
    fn mating_produces_fertile_eggs() {
        // Unmated female should not lay eggs; mated female should.
        let mut pop = test_population(1, 1);

        let fruit = vec![(5.0, 5.0, 0.0)];

        // Before mating: female should not oviposit
        pop.oviposit(&fruit);
        assert!(
            pop.egg_clusters.is_empty(),
            "Unmated female should not lay eggs"
        );

        // Trigger mating (male and female are co-located)
        pop.attempt_mating();
        assert!(
            pop.flies.iter().any(|f| f.sex == FlySex::Female && f.mated),
            "Female should be mated after proximity mating"
        );

        // After mating: female should oviposit
        pop.oviposit(&fruit);
        assert!(
            !pop.egg_clusters.is_empty(),
            "Mated female should lay eggs near fruit"
        );
        assert!(pop.egg_clusters[0].count >= MIN_EGGS_PER_CLUSTER);
    }

    #[test]
    fn population_grows_with_resources() {
        // With stable food and good temperature, population should grow.
        let mut pop = test_population(3, 3);
        let fruit = vec![(5.0, 5.0, 0.0), (10.0, 10.0, 0.0)];
        let initial_count = pop.total_alive();

        // Run for ~10 simulated days in 1-hour steps at 25C
        for _ in 0..(10 * 24) {
            pop.step(1.0, 25.0, &fruit, 0.7);
        }

        let final_count = pop.total_alive();
        assert!(
            final_count > initial_count,
            "Population should grow with resources: started {}, ended {}",
            initial_count,
            final_count
        );
    }

    #[test]
    fn cold_temperature_slows_development() {
        // Below 12C, development should halt completely.
        let mut pop = FlyPopulation::new(99);
        pop.egg_clusters.push(EggCluster {
            position: (5.0, 5.0),
            count: 10,
            age_hours: 0.0,
            substrate_quality: 1.0,
        });

        // Run 100 hours at 10C (below threshold)
        for _ in 0..100 {
            pop.advance_eggs(1.0, 10.0);
        }

        assert!(
            pop.egg_clusters[0].age_hours < f32::EPSILON,
            "Eggs below 12C should have zero development, got {} eff hours",
            pop.egg_clusters[0].age_hours
        );

        // Now try at 14C -- should develop but slowly
        let mut pop2 = FlyPopulation::new(100);
        pop2.egg_clusters.push(EggCluster {
            position: (5.0, 5.0),
            count: 10,
            age_hours: 0.0,
            substrate_quality: 1.0,
        });

        for _ in 0..100 {
            pop2.advance_eggs(1.0, 14.0);
        }
        assert!(
            pop2.egg_clusters[0].age_hours > 0.0,
            "Eggs above 12C should develop"
        );
        assert!(
            pop2.egg_clusters[0].age_hours < 100.0,
            "Development at 14C should be slower than real-time"
        );
    }

    #[test]
    fn temperature_rate_factor_sanity() {
        // At 25C (reference), factor should be ~1.0
        let f25 = temperature_rate_factor(25.0);
        assert!(
            (f25 - 1.0).abs() < 0.01,
            "Rate at 25C should be ~1.0, got {}",
            f25
        );

        // Below 12C -> 0
        assert_eq!(temperature_rate_factor(10.0), 0.0);
        assert_eq!(temperature_rate_factor(12.0), 0.0); // at boundary

        // Above 32C -> 0
        assert_eq!(temperature_rate_factor(33.0), 0.0);
        assert_eq!(temperature_rate_factor(32.0), 0.0); // at boundary

        // 18C should be slower than 25C
        let f18 = temperature_rate_factor(18.0);
        assert!(f18 < f25, "18C should be slower than 25C");
        assert!(f18 > 0.1, "18C should still give some development, got {}", f18);

        // 30C should be faster than 25C
        let f30 = temperature_rate_factor(30.0);
        assert!(f30 > f25, "30C should be faster than 25C: {} vs {}", f30, f25);
    }

    #[test]
    fn unmated_female_cannot_lay() {
        let mut pop = test_population(0, 1);
        // Female is NOT mated, has plenty of energy
        pop.flies[0].energy = FLY_ENERGY_MAX;

        let fruit = vec![(5.0, 5.0, 0.0)];
        pop.oviposit(&fruit);

        assert!(
            pop.egg_clusters.is_empty(),
            "Unmated female must not oviposit"
        );
    }

    #[test]
    fn egg_count_decrements_after_oviposition() {
        let mut pop = test_population(1, 1);
        pop.attempt_mating(); // mate them
        let initial_eggs = pop.flies.iter()
            .find(|f| f.sex == FlySex::Female)
            .unwrap()
            .eggs_remaining;

        let fruit = vec![(5.0, 5.0, 0.0)];
        pop.oviposit(&fruit);

        let remaining = pop.flies.iter()
            .find(|f| f.sex == FlySex::Female)
            .unwrap()
            .eggs_remaining;

        assert!(
            remaining < initial_eggs,
            "Eggs remaining should decrease after oviposition: was {}, now {}",
            initial_eggs, remaining
        );
    }

    #[test]
    fn adult_dies_past_max_lifespan() {
        let mut pop = FlyPopulation::new(55);
        let mut fly = Fly::new_adult(0, FlySex::Male, (5.0, 5.0, 0.0));
        fly.energy = FLY_ENERGY_MAX; // plenty of energy
        pop.next_id = 1;
        pop.flies.push(fly);

        // Advance 61 days at 25C in 24h steps
        for _ in 0..61 {
            pop.advance_development(24.0, 25.0);
        }
        pop.remove_dead();

        assert!(
            pop.flies.is_empty(),
            "Adult should die after {} days",
            MAX_ADULT_LIFESPAN_DAYS
        );
    }

    #[test]
    fn stage_census_counts_correctly() {
        let mut pop = FlyPopulation::new(88);

        // Add 2 adults
        pop.flies.push(Fly::new_adult(0, FlySex::Male, (0.0, 0.0, 0.0)));
        pop.flies.push(Fly::new_adult(1, FlySex::Female, (0.0, 0.0, 0.0)));

        // Add 1 larva
        let mut larva = Fly::new_embryo(2, FlySex::Male, (0.0, 0.0, 0.0));
        larva.stage = FlyLifeStage::Larva { instar: 2, age_hours: 5.0 };
        pop.flies.push(larva);

        // Add 1 pupa
        let mut pupa = Fly::new_embryo(3, FlySex::Female, (0.0, 0.0, 0.0));
        pupa.stage = FlyLifeStage::Pupa { age_hours: 10.0 };
        pop.flies.push(pupa);

        // Add 1 egg cluster
        pop.egg_clusters.push(EggCluster {
            position: (0.0, 0.0),
            count: 12,
            age_hours: 0.0,
            substrate_quality: 1.0,
        });

        pop.next_id = 4;

        let census = pop.stage_census();
        assert_eq!(census.adults, 2);
        assert_eq!(census.larvae, 1);
        assert_eq!(census.pupae, 1);
        assert_eq!(census.embryos, 0);
        assert_eq!(census.egg_clusters, 1);
        assert_eq!(census.total_eggs, 12);
        assert_eq!(census.total_individuals(), 4);
    }
}
