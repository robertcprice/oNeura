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

use crate::constants::clamp;
use crate::terrarium::material_exchange::{
    deposit_species_to_inventory, inventory_component_amount, withdraw_species_from_inventory,
};
use crate::terrarium::{RegionalMaterialInventory, TerrariumSpecies};
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;

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
pub(crate) const LIFETIME_EGGS: u32 = 400;

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
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FlySex {
    Male,
    Female,
}

// ============================================================================
// Individual Fly
// ============================================================================

/// A single Drosophila individual tracked through its lifecycle.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Fly {
    /// Unique identifier within the population.
    pub id: u32,
    /// Shared terrarium organism identity when this fly is linked to a neural adult.
    pub organism_id: Option<u64>,
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
    /// Persistent local chemistry ledger for immature stages and reproductive state.
    pub material_inventory: RegionalMaterialInventory,
    /// Whether this individual is alive.
    alive: bool,
}

impl Fly {
    /// Create a new adult fly at a given position.
    pub fn new_adult(id: u32, sex: FlySex, position: (f32, f32, f32)) -> Self {
        Self {
            id,
            organism_id: None,
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
            material_inventory: RegionalMaterialInventory::new(format!("fly:{id}:adult")),
            alive: true,
        }
    }

    /// Create a new fly at embryo stage (from an egg cluster hatch).
    fn new_embryo(id: u32, sex: FlySex, position: (f32, f32, f32)) -> Self {
        Self {
            id,
            organism_id: None,
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
            material_inventory: RegionalMaterialInventory::new(format!("fly:{id}:embryo")),
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

    pub fn with_organism_id(mut self, organism_id: u64) -> Self {
        self.organism_id = Some(organism_id);
        self
    }

    pub fn is_alive(&self) -> bool {
        self.alive
    }
}

// ============================================================================
// Egg Cluster
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FlyEmbryoState {
    /// Stable embryo identity within the lifecycle population.
    pub id: u32,
    pub sex: FlySex,
    /// Position relative to the clutch centroid in mm.
    pub offset_mm: (f32, f32),
    /// Effective development hours accumulated by this embryo.
    pub age_hours: f32,
    /// Survival potential after chemistry and humidity stress [0, 1].
    pub viability: f32,
    /// Persistent internal chemistry for this embryo.
    pub material_inventory: RegionalMaterialInventory,
}

/// A clutch of eggs deposited on a substrate (typically fermenting fruit).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EggCluster {
    /// Position (x, y) in mm on the substrate surface.
    pub position: (f32, f32),
    /// Derived summary count kept for compatibility/reporting.
    pub count: u8,
    /// Derived mean embryonic age kept for compatibility/reporting.
    pub age_hours: f32,
    /// Quality of the substrate where eggs were laid [0, 1].
    pub substrate_quality: f32,
    /// Persistent local chemistry around the clutch.
    pub material_inventory: RegionalMaterialInventory,
    /// Explicit per-embryo state owned by this clutch.
    pub embryos: Vec<FlyEmbryoState>,
}

impl EggCluster {
    pub fn refresh_summary(&mut self) {
        self.count = self.embryos.len().min(u8::MAX as usize) as u8;
        self.age_hours = if self.embryos.is_empty() {
            0.0
        } else {
            self.embryos
                .iter()
                .map(|embryo| embryo.age_hours)
                .sum::<f32>()
                / self.embryos.len() as f32
        };
    }

    pub fn mean_embryo_viability(&self) -> f32 {
        if self.embryos.is_empty() {
            0.0
        } else {
            self.embryos
                .iter()
                .map(|embryo| embryo.viability)
                .sum::<f32>()
                / self.embryos.len() as f32
        }
    }

    pub fn embryo_position_mm(&self, embryo: &FlyEmbryoState) -> (f32, f32, f32) {
        (
            self.position.0 + embryo.offset_mm.0,
            self.position.1 + embryo.offset_mm.1,
            0.0,
        )
    }
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct FruitResourcePatch {
    pub position: (f32, f32, f32),
    pub substrate_quality: f32,
    pub water: f32,
    pub glucose: f32,
    pub amino_acids: f32,
    pub oxygen: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LinkedAdultFlyState {
    pub organism_id: u64,
    pub position: (f32, f32, f32),
    pub energy_uj: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EclosedFlyRecord {
    pub fly_id: u32,
    pub organism_id: Option<u64>,
    pub sex: FlySex,
    pub position: (f32, f32, f32),
}

fn fruit_resource_inventory_target(
    patch: &FruitResourcePatch,
    name: impl Into<String>,
) -> RegionalMaterialInventory {
    let mut inventory = RegionalMaterialInventory::new(name.into());
    deposit_species_to_inventory(&mut inventory, TerrariumSpecies::Water, patch.water);
    deposit_species_to_inventory(&mut inventory, TerrariumSpecies::Glucose, patch.glucose);
    deposit_species_to_inventory(
        &mut inventory,
        TerrariumSpecies::AminoAcidPool,
        patch.amino_acids,
    );
    deposit_species_to_inventory(&mut inventory, TerrariumSpecies::OxygenGas, patch.oxygen);
    deposit_species_to_inventory(
        &mut inventory,
        TerrariumSpecies::NucleotidePool,
        patch.amino_acids * 0.18 + patch.glucose * 0.10,
    );
    deposit_species_to_inventory(
        &mut inventory,
        TerrariumSpecies::MembranePrecursorPool,
        patch.glucose * 0.12 + patch.amino_acids * 0.08,
    );
    inventory
}

fn nearest_fruit_resource<'a>(
    resources: &'a [FruitResourcePatch],
    position: (f32, f32, f32),
    max_dist_mm: f32,
) -> Option<&'a FruitResourcePatch> {
    resources
        .iter()
        .filter_map(|resource| {
            let dx = position.0 - resource.position.0;
            let dy = position.1 - resource.position.1;
            let dist = (dx * dx + dy * dy).sqrt();
            (dist <= max_dist_mm).then_some((resource, dist))
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(resource, _)| resource)
}

// ============================================================================
// Population Manager
// ============================================================================

/// Manages the entire Drosophila population: lifecycle progression, mating,
/// oviposition, egg hatching, and mortality.
#[derive(Debug, Clone)]
pub struct FlyPopulation {
    /// All individual flies (immature and adult).
    pub flies: Vec<Fly>,
    /// All active egg clusters.
    pub egg_clusters: Vec<EggCluster>,
    /// Next unique ID to assign.
    pub next_id: u32,
    /// RNG for stochastic processes.
    rng: ChaCha12Rng,
    /// Adults that eclosed (pupa → adult) this step.
    pub(crate) eclosed_adults: Vec<EclosedFlyRecord>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FlyPopulationCheckpoint {
    pub flies: Vec<Fly>,
    pub egg_clusters: Vec<EggCluster>,
    pub next_id: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rng: Option<ChaCha12Rng>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rng_resume_seed: Option<u64>,
    pub eclosed_adults: Vec<EclosedFlyRecord>,
}

impl FlyPopulation {
    /// Create a new empty population.
    pub fn new(seed: u64) -> Self {
        Self {
            flies: Vec::new(),
            egg_clusters: Vec::new(),
            next_id: 0,
            rng: ChaCha12Rng::seed_from_u64(seed),
            eclosed_adults: Vec::new(),
        }
    }

    pub fn checkpoint(&self) -> FlyPopulationCheckpoint {
        FlyPopulationCheckpoint {
            flies: self.flies.clone(),
            egg_clusters: self.egg_clusters.clone(),
            next_id: self.next_id,
            rng: Some(self.rng.clone()),
            rng_resume_seed: None,
            eclosed_adults: self.eclosed_adults.clone(),
        }
    }

    pub fn from_checkpoint(checkpoint: FlyPopulationCheckpoint) -> Self {
        Self {
            flies: checkpoint.flies,
            egg_clusters: checkpoint.egg_clusters,
            next_id: checkpoint.next_id,
            rng: checkpoint.rng.unwrap_or_else(|| {
                ChaCha12Rng::seed_from_u64(checkpoint.rng_resume_seed.unwrap_or(0))
            }),
            eclosed_adults: checkpoint.eclosed_adults,
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

    /// Drain adults that eclosed (pupa → adult transitions) this step.
    pub fn drain_eclosed(&mut self) -> Vec<EclosedFlyRecord> {
        std::mem::take(&mut self.eclosed_adults)
    }

    pub fn fly_by_organism_id(&self, organism_id: u64) -> Option<&Fly> {
        self.flies
            .iter()
            .find(|fly| fly.organism_id == Some(organism_id))
    }

    pub fn assign_organism_id(&mut self, fly_id: u32, organism_id: u64) -> bool {
        let Some(fly) = self.flies.iter_mut().find(|fly| fly.id == fly_id) else {
            return false;
        };
        fly.organism_id = Some(organism_id);
        true
    }

    pub fn sync_linked_adults(
        &mut self,
        linked_adults: &[LinkedAdultFlyState],
        fruit_resources: &[FruitResourcePatch],
    ) {
        for fly in self.flies.iter_mut() {
            if !matches!(fly.stage, FlyLifeStage::Adult { .. }) {
                continue;
            }
            let Some(organism_id) = fly.organism_id else {
                continue;
            };
            let Some(linked) = linked_adults
                .iter()
                .find(|linked| linked.organism_id == organism_id)
            else {
                continue;
            };
            fly.position = linked.position;
            fly.energy = linked.energy_uj.clamp(0.0, FLY_ENERGY_MAX);
            if let Some(resource) = nearest_fruit_resource(
                fruit_resources,
                fly.position,
                OVIPOSITION_PROXIMITY_MM * 1.5,
            ) {
                let target = fruit_resource_inventory_target(
                    resource,
                    format!("fly:{}:adult-target", fly.id),
                );
                if fly.material_inventory.is_empty() {
                    fly.material_inventory = target.scaled(0.64);
                } else {
                    let _ = fly.material_inventory.relax_toward(&target, 0.18);
                }
            }
        }
    }

    pub fn iter_cluster_embryos(
        &self,
    ) -> impl Iterator<Item = (usize, usize, &EggCluster, &FlyEmbryoState)> + '_ {
        self.egg_clusters
            .iter()
            .enumerate()
            .flat_map(|(cluster_index, cluster)| {
                cluster
                    .embryos
                    .iter()
                    .enumerate()
                    .map(move |(embryo_index, embryo)| {
                        (cluster_index, embryo_index, cluster, embryo)
                    })
            })
    }

    pub fn embryo_at_flat_index(
        &self,
        index: usize,
    ) -> Option<(usize, usize, &EggCluster, &FlyEmbryoState)> {
        self.iter_cluster_embryos().nth(index)
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
        c.total_eggs = self
            .egg_clusters
            .iter()
            .map(|cluster| cluster.embryos.len() as u32)
            .sum();
        c.embryos += c.total_eggs;
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
        let resources = fruit_positions
            .iter()
            .map(|position| FruitResourcePatch {
                position: *position,
                substrate_quality: 1.0,
                water: 1.0,
                glucose: 1.0,
                amino_acids: 0.18,
                oxygen: 0.24,
            })
            .collect::<Vec<_>>();
        self.step_with_resources(dt_hours, temperature, &resources, humidity);
    }

    pub fn step_with_resources(
        &mut self,
        dt_hours: f32,
        temperature: f32,
        fruit_resources: &[FruitResourcePatch],
        humidity: f32,
    ) {
        self.eclosed_adults.clear();
        self.advance_development(dt_hours, temperature);
        self.feed_immatures(dt_hours, temperature, fruit_resources);
        self.attempt_mating();
        self.oviposit(fruit_resources);
        self.advance_eggs(dt_hours, temperature, fruit_resources);
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
                        fly.stage = FlyLifeStage::Embryo { age_hours: new_age };
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
                        self.eclosed_adults.push(EclosedFlyRecord {
                            fly_id: fly.id,
                            organism_id: fly.organism_id,
                            sex: fly.sex,
                            position: fly.position,
                        });
                    } else {
                        fly.stage = FlyLifeStage::Pupa { age_hours: new_age };
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
        fruit_resources: &[FruitResourcePatch],
    ) {
        for fly in self.flies.iter_mut() {
            if !fly.alive {
                continue;
            }
            // Only larvae feed (embryos use yolk, pupae don't feed)
            if !matches!(fly.stage, FlyLifeStage::Larva { .. }) {
                continue;
            }
            let Some(resource) = nearest_fruit_resource(
                fruit_resources,
                fly.position,
                OVIPOSITION_PROXIMITY_MM * 2.0,
            ) else {
                continue;
            };
            let target =
                fruit_resource_inventory_target(resource, format!("fly:{}:larva-target", fly.id));
            if fly.material_inventory.is_empty() {
                fly.material_inventory = target.scaled(0.72);
            } else {
                let _ = fly.material_inventory.relax_toward(&target, 0.20);
            }
            let quality = resource.substrate_quality.clamp(0.0, 1.4);
            let glucose_take = withdraw_species_from_inventory(
                &mut fly.material_inventory,
                TerrariumSpecies::Glucose,
                (0.08 + quality * 0.10) * dt_hours,
            );
            let amino_take = withdraw_species_from_inventory(
                &mut fly.material_inventory,
                TerrariumSpecies::AminoAcidPool,
                (0.03 + quality * 0.04) * dt_hours,
            );
            let water_take = withdraw_species_from_inventory(
                &mut fly.material_inventory,
                TerrariumSpecies::Water,
                (0.02 + quality * 0.03) * dt_hours,
            );
            let energy_gain = (glucose_take * 190.0 + amino_take * 75.0 + water_take * 10.0)
                .min(LARVAL_FEED_RATE_UJ_PER_HOUR * dt_hours * 1.6);
            fly.energy = (fly.energy + energy_gain).min(FLY_ENERGY_MAX);
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
                let dist =
                    ((fp.0 - mp.0).powi(2) + (fp.1 - mp.1).powi(2) + (fp.2 - mp.2).powi(2)).sqrt();
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

    fn embryo_offset_mm<R: Rng + ?Sized>(
        rng: &mut R,
        embryo_index: usize,
        clutch_size: usize,
    ) -> (f32, f32) {
        let columns = (clutch_size as f32).sqrt().ceil().max(1.0) as usize;
        let rows = clutch_size.div_ceil(columns);
        let row = embryo_index / columns;
        let col = embryo_index % columns;
        let base_x = (col as f32 - (columns.saturating_sub(1)) as f32 * 0.5) * 0.26;
        let base_y = (row as f32 - (rows.saturating_sub(1)) as f32 * 0.5) * 0.34;
        let jitter_x = rng.gen_range(-0.04..0.04);
        let jitter_y = rng.gen_range(-0.05..0.05);
        (base_x + jitter_x, base_y + jitter_y)
    }

    fn build_embryo_state<R: Rng + ?Sized>(
        rng: &mut R,
        next_id: &mut u32,
        cluster_inventory: &RegionalMaterialInventory,
        embryo_index: usize,
        clutch_size: usize,
        substrate_quality: f32,
    ) -> FlyEmbryoState {
        let id = *next_id;
        *next_id += 1;
        let sex = if rng.gen_bool(0.5) {
            FlySex::Male
        } else {
            FlySex::Female
        };
        let mut material_inventory =
            cluster_inventory.scaled((0.18 / clutch_size.max(1) as f64).clamp(0.012, 0.06));
        deposit_species_to_inventory(
            &mut material_inventory,
            TerrariumSpecies::Water,
            0.006 + substrate_quality * 0.004,
        );
        deposit_species_to_inventory(
            &mut material_inventory,
            TerrariumSpecies::Glucose,
            0.004 + substrate_quality * 0.003,
        );
        deposit_species_to_inventory(
            &mut material_inventory,
            TerrariumSpecies::AminoAcidPool,
            0.003 + substrate_quality * 0.003,
        );
        deposit_species_to_inventory(
            &mut material_inventory,
            TerrariumSpecies::NucleotidePool,
            0.003 + substrate_quality * 0.002,
        );
        deposit_species_to_inventory(
            &mut material_inventory,
            TerrariumSpecies::MembranePrecursorPool,
            0.002 + substrate_quality * 0.002,
        );
        FlyEmbryoState {
            id,
            sex,
            offset_mm: Self::embryo_offset_mm(rng, embryo_index, clutch_size),
            age_hours: 0.0,
            viability: clamp(
                0.64 + substrate_quality * 0.18 + rng.gen_range(-0.06..0.06),
                0.12,
                1.0,
            ),
            material_inventory,
        }
    }

    // ========================================================================
    // Oviposition
    // ========================================================================

    /// Females lay eggs near fermenting fruit if conditions are met.
    ///
    /// Oviposition site selection: females strongly prefer fermenting fruit
    /// over other substrates (Dweck et al. 2013, Or56a olfactory receptor).
    fn oviposit(&mut self, fruit_resources: &[FruitResourcePatch]) {
        if fruit_resources.is_empty() {
            return;
        }

        let mut new_clusters: Vec<EggCluster> = Vec::new();
        let mut pending_clutches: Vec<((f32, f32), f32, u8, RegionalMaterialInventory)> =
            Vec::new();

        for fly in self.flies.iter_mut() {
            if !fly.can_oviposit() {
                continue;
            }

            // Find nearest fruit within oviposition range
            let mut closest: Option<(usize, f32)> = None;
            for (i, fp) in fruit_resources.iter().enumerate() {
                let dist = ((fly.position.0 - fp.position.0).powi(2)
                    + (fly.position.1 - fp.position.1).powi(2))
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

                let fp = &fruit_resources[fi];
                let mut material_inventory = fruit_resource_inventory_target(
                    fp,
                    format!("egg-cluster:{}:{}", self.next_id, fi),
                )
                .scaled((0.10 + clutch_size as f64 * 0.02).clamp(0.10, 0.40));
                deposit_species_to_inventory(
                    &mut material_inventory,
                    TerrariumSpecies::NucleotidePool,
                    clutch_size as f32 * 0.01,
                );
                deposit_species_to_inventory(
                    &mut material_inventory,
                    TerrariumSpecies::MembranePrecursorPool,
                    clutch_size as f32 * 0.008,
                );
                pending_clutches.push((
                    (fp.position.0, fp.position.1),
                    fp.substrate_quality,
                    clutch_size,
                    material_inventory,
                ));

                fly.eggs_remaining = fly.eggs_remaining.saturating_sub(clutch_size as u32);
                fly.energy -= clutch_size as f32 * EGG_ENERGY_COST_UJ;
                fly.time_since_last_oviposition = 0.0;
            }
        }

        for (position, substrate_quality, clutch_size, material_inventory) in pending_clutches {
            let mut cluster = EggCluster {
                position,
                count: 0,
                age_hours: 0.0,
                substrate_quality,
                material_inventory,
                embryos: Vec::with_capacity(clutch_size as usize),
            };
            let embryo_target = cluster.material_inventory.clone();
            for embryo_index in 0..clutch_size as usize {
                cluster.embryos.push(Self::build_embryo_state(
                    &mut self.rng,
                    &mut self.next_id,
                    &embryo_target,
                    embryo_index,
                    clutch_size as usize,
                    substrate_quality,
                ));
            }
            cluster.refresh_summary();
            new_clusters.push(cluster);
        }

        self.egg_clusters.extend(new_clusters);
    }

    // ========================================================================
    // Egg Development & Hatching
    // ========================================================================

    /// Advance egg cluster ages by temperature-scaled development time.
    fn advance_eggs(
        &mut self,
        dt_hours: f32,
        temperature: f32,
        fruit_resources: &[FruitResourcePatch],
    ) {
        let dev_dt = effective_dev_hours(dt_hours, temperature);
        for cluster in self.egg_clusters.iter_mut() {
            if let Some(resource) = nearest_fruit_resource(
                fruit_resources,
                (cluster.position.0, cluster.position.1, 0.0),
                OVIPOSITION_PROXIMITY_MM * 1.4,
            ) {
                let target = fruit_resource_inventory_target(
                    resource,
                    format!(
                        "egg-cluster:{}:{}:target",
                        cluster.position.0, cluster.position.1
                    ),
                );
                if cluster.material_inventory.is_empty() {
                    cluster.material_inventory = target.scaled(0.78);
                } else {
                    let _ = cluster.material_inventory.relax_toward(&target, 0.18);
                }
            }
            let water =
                inventory_component_amount(&cluster.material_inventory, TerrariumSpecies::Water);
            let glucose =
                inventory_component_amount(&cluster.material_inventory, TerrariumSpecies::Glucose);
            let amino = inventory_component_amount(
                &cluster.material_inventory,
                TerrariumSpecies::AminoAcidPool,
            );
            let oxygen = inventory_component_amount(
                &cluster.material_inventory,
                TerrariumSpecies::OxygenGas,
            );
            cluster.substrate_quality = clamp(
                cluster.substrate_quality * 0.44
                    + glucose * 0.18
                    + water * 0.14
                    + amino * 0.12
                    + oxygen * 0.08,
                0.0,
                1.4,
            );
            let dev_support = clamp(
                0.28 + cluster.substrate_quality * 0.34
                    + glucose * 0.12
                    + water * 0.10
                    + amino * 0.10
                    + oxygen * 0.06,
                0.05,
                1.8,
            );
            let embryo_target = cluster
                .material_inventory
                .scaled((0.20 / cluster.embryos.len().max(1) as f64).clamp(0.014, 0.08));
            for embryo in cluster.embryos.iter_mut() {
                if embryo.material_inventory.is_empty() {
                    embryo.material_inventory = embryo_target.clone();
                } else {
                    let _ = embryo.material_inventory.relax_toward(&embryo_target, 0.16);
                }
                let embryo_water =
                    inventory_component_amount(&embryo.material_inventory, TerrariumSpecies::Water);
                let embryo_glucose = inventory_component_amount(
                    &embryo.material_inventory,
                    TerrariumSpecies::Glucose,
                );
                let embryo_amino = inventory_component_amount(
                    &embryo.material_inventory,
                    TerrariumSpecies::AminoAcidPool,
                );
                let embryo_oxygen = inventory_component_amount(
                    &embryo.material_inventory,
                    TerrariumSpecies::OxygenGas,
                );
                let embryo_nucleotide = inventory_component_amount(
                    &embryo.material_inventory,
                    TerrariumSpecies::NucleotidePool,
                );
                let embryo_membrane = inventory_component_amount(
                    &embryo.material_inventory,
                    TerrariumSpecies::MembranePrecursorPool,
                );
                let embryo_support = clamp(
                    dev_support * 0.58
                        + cluster.substrate_quality * 0.16
                        + embryo_water * 0.10
                        + embryo_glucose * 0.08
                        + embryo_amino * 0.08
                        + embryo_oxygen * 0.05
                        + embryo_nucleotide * 0.07
                        + embryo_membrane * 0.05,
                    0.03,
                    1.9,
                );
                embryo.viability = clamp(
                    embryo.viability * 0.72 + (embryo_support / 1.9).clamp(0.0, 1.0) * 0.28,
                    0.01,
                    1.0,
                );
                embryo.age_hours += dev_dt * embryo_support;
            }
            cluster.refresh_summary();
        }
    }

    /// Hatch egg clusters that have completed embryonic development.
    ///
    /// Humidity affects viability: below 40% RH, egg desiccation kills ~50%.
    fn hatch_eggs(&mut self, _temperature: f32, humidity: f32) {
        let mut new_flies: Vec<Fly> = Vec::new();
        let mut retained_clusters = Vec::with_capacity(self.egg_clusters.len());

        for mut cluster in self.egg_clusters.drain(..) {
            let water =
                inventory_component_amount(&cluster.material_inventory, TerrariumSpecies::Water);
            let glucose =
                inventory_component_amount(&cluster.material_inventory, TerrariumSpecies::Glucose);
            let oxygen = inventory_component_amount(
                &cluster.material_inventory,
                TerrariumSpecies::OxygenGas,
            );
            let humidity_factor = if humidity < 0.4 { 0.5 } else { 0.95 };
            let mut retained_embryos = Vec::with_capacity(cluster.embryos.len());
            for embryo in cluster.embryos.drain(..) {
                let embryo_water =
                    inventory_component_amount(&embryo.material_inventory, TerrariumSpecies::Water);
                let embryo_glucose = inventory_component_amount(
                    &embryo.material_inventory,
                    TerrariumSpecies::Glucose,
                );
                let embryo_oxygen = inventory_component_amount(
                    &embryo.material_inventory,
                    TerrariumSpecies::OxygenGas,
                );
                let survival_rate = clamp(
                    humidity_factor
                        * embryo.viability
                        * (0.52
                            + water * 0.10
                            + glucose * 0.06
                            + oxygen * 0.06
                            + embryo_water * 0.10
                            + embryo_glucose * 0.08
                            + embryo_oxygen * 0.08),
                    0.02,
                    0.995,
                );
                if embryo.age_hours >= EMBRYO_DURATION_H_25C {
                    if self.rng.gen::<f32>() < survival_rate {
                        let position_mm = (
                            cluster.position.0 + embryo.offset_mm.0,
                            cluster.position.1 + embryo.offset_mm.1,
                            0.0,
                        );
                        let mut larva = Fly::new_embryo(embryo.id, embryo.sex, position_mm);
                        larva.stage = FlyLifeStage::Larva {
                            instar: 1,
                            age_hours: 0.0,
                        };
                        larva.material_inventory = embryo.material_inventory;
                        new_flies.push(larva);
                    }
                } else if embryo.viability > 0.03 {
                    retained_embryos.push(embryo);
                }
            }
            cluster.embryos = retained_embryos;
            if !cluster.embryos.is_empty() {
                cluster.refresh_summary();
                retained_clusters.push(cluster);
            }
        }

        self.egg_clusters = retained_clusters;
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

    fn test_fruit_resources(positions: &[(f32, f32, f32)]) -> Vec<FruitResourcePatch> {
        positions
            .iter()
            .map(|&(x, y, z)| FruitResourcePatch {
                position: (x, y, z),
                substrate_quality: 1.0,
                water: 1.2,
                glucose: 1.1,
                amino_acids: 0.7,
                oxygen: 0.18,
            })
            .collect()
    }

    fn test_embryo(id: u32, age_hours: f32, viability: f32) -> FlyEmbryoState {
        let mut material_inventory = RegionalMaterialInventory::new(format!("embryo:test:{id}"));
        deposit_species_to_inventory(&mut material_inventory, TerrariumSpecies::Water, 0.08);
        deposit_species_to_inventory(&mut material_inventory, TerrariumSpecies::Glucose, 0.06);
        deposit_species_to_inventory(
            &mut material_inventory,
            TerrariumSpecies::AminoAcidPool,
            0.04,
        );
        deposit_species_to_inventory(
            &mut material_inventory,
            TerrariumSpecies::NucleotidePool,
            0.03,
        );
        deposit_species_to_inventory(
            &mut material_inventory,
            TerrariumSpecies::MembranePrecursorPool,
            0.02,
        );
        FlyEmbryoState {
            id,
            sex: if id % 2 == 0 {
                FlySex::Female
            } else {
                FlySex::Male
            },
            offset_mm: (
                (id % 3) as f32 * 0.08 - 0.08,
                (id % 2) as f32 * 0.09 - 0.045,
            ),
            age_hours,
            viability,
            material_inventory,
        }
    }

    fn test_egg_cluster(position: (f32, f32), count: u8, substrate_quality: f32) -> EggCluster {
        let mut cluster = EggCluster {
            position,
            count: 0,
            age_hours: 0.0,
            substrate_quality,
            material_inventory: fruit_resource_inventory_target(
                &FruitResourcePatch {
                    position: (position.0, position.1, 0.0),
                    substrate_quality,
                    water: 1.2,
                    glucose: 1.1,
                    amino_acids: 0.7,
                    oxygen: 0.18,
                },
                format!("egg:test:{:.1}:{:.1}", position.0, position.1),
            ),
            embryos: (0..count)
                .map(|i| test_embryo(i as u32, 0.0, 0.9))
                .collect(),
        };
        cluster.refresh_summary();
        cluster
    }

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
        pop_warm
            .egg_clusters
            .push(test_egg_cluster((5.0, 5.0), 10, 1.0));

        let mut pop_cool = FlyPopulation::new(2);
        pop_cool
            .egg_clusters
            .push(test_egg_cluster((5.0, 5.0), 10, 1.0));

        let fruit = test_fruit_resources(&[(5.0_f32, 5.0_f32, 0.0_f32)]);

        // Advance both by 24 real hours
        for _ in 0..24 {
            pop_warm.advance_eggs(1.0, 25.0, &fruit);
            pop_cool.advance_eggs(1.0, 18.0, &fruit);
        }

        // Warm eggs should still outpace cool eggs, but local chemistry now modulates
        // the exact age accumulation instead of assuming a pure temperature-only clock.
        assert!(
            pop_warm.egg_clusters[0].age_hours > 14.0,
            "Warm eggs should accumulate strong development support, got {}",
            pop_warm.egg_clusters[0].age_hours
        );
        // At 18C, rate factor < 1 so fewer effective hours.
        assert!(
            pop_cool.egg_clusters[0].age_hours < 20.0,
            "Cool eggs should have <20 eff hours, got {}",
            pop_cool.egg_clusters[0].age_hours
        );
        assert!(
            pop_warm.egg_clusters[0].age_hours > pop_cool.egg_clusters[0].age_hours,
            "Warm eggs should develop faster than cool eggs"
        );

        // Rich warm eggs should reach hatchability with a few more hours of chemistry-backed
        // development; cool eggs should still lag well behind.
        for _ in 0..12 {
            pop_warm.advance_eggs(1.0, 25.0, &fruit);
        }
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

        let fruit = test_fruit_resources(&[(5.0, 5.0, 0.0)]);

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
        let has_adult = stage_transitions.iter().any(|(_, s)| s.contains("Adult"));

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

        let fruit = test_fruit_resources(&[(5.0, 5.0, 0.0)]);
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

        let fruit = test_fruit_resources(&[(5.0, 5.0, 0.0)]);

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
        assert_eq!(
            pop.egg_clusters[0].embryos.len(),
            pop.egg_clusters[0].count as usize,
            "clutch summaries should be derived from explicit embryos"
        );
    }

    #[test]
    fn oviposition_materializes_explicit_embryos() {
        let mut pop = test_population(1, 1);
        pop.attempt_mating();
        let fruit = test_fruit_resources(&[(5.0, 5.0, 0.0)]);

        pop.oviposit(&fruit);

        let cluster = pop
            .egg_clusters
            .first()
            .expect("mated female should lay a clutch");
        assert!(
            cluster.embryos.len() >= MIN_EGGS_PER_CLUSTER as usize,
            "clutch should own explicit embryos"
        );
        assert!(
            cluster
                .embryos
                .iter()
                .all(|embryo| embryo.material_inventory.total_amount_moles() > 0.0),
            "each embryo should carry its own internal chemistry ledger"
        );
        let mut embryo_ids = cluster
            .embryos
            .iter()
            .map(|embryo| embryo.id)
            .collect::<Vec<_>>();
        embryo_ids.sort_unstable();
        embryo_ids.dedup();
        assert_eq!(
            embryo_ids.len(),
            cluster.embryos.len(),
            "explicit embryos should have stable unique identities"
        );
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
        pop.egg_clusters.push(test_egg_cluster((5.0, 5.0), 10, 1.0));
        let fruit = test_fruit_resources(&[(5.0, 5.0, 0.0)]);

        // Run 100 hours at 10C (below threshold)
        for _ in 0..100 {
            pop.advance_eggs(1.0, 10.0, &fruit);
        }

        assert!(
            pop.egg_clusters[0].age_hours < f32::EPSILON,
            "Eggs below 12C should have zero development, got {} eff hours",
            pop.egg_clusters[0].age_hours
        );

        // Now try at 14C -- should develop but slowly
        let mut pop2 = FlyPopulation::new(100);
        pop2.egg_clusters
            .push(test_egg_cluster((5.0, 5.0), 10, 1.0));

        for _ in 0..100 {
            pop2.advance_eggs(1.0, 14.0, &fruit);
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
        assert!(
            f18 > 0.1,
            "18C should still give some development, got {}",
            f18
        );

        // 30C should be faster than 25C
        let f30 = temperature_rate_factor(30.0);
        assert!(
            f30 > f25,
            "30C should be faster than 25C: {} vs {}",
            f30,
            f25
        );
    }

    #[test]
    fn unmated_female_cannot_lay() {
        let mut pop = test_population(0, 1);
        // Female is NOT mated, has plenty of energy
        pop.flies[0].energy = FLY_ENERGY_MAX;

        let fruit = test_fruit_resources(&[(5.0, 5.0, 0.0)]);
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
        let initial_eggs = pop
            .flies
            .iter()
            .find(|f| f.sex == FlySex::Female)
            .unwrap()
            .eggs_remaining;

        let fruit = test_fruit_resources(&[(5.0, 5.0, 0.0)]);
        pop.oviposit(&fruit);

        let remaining = pop
            .flies
            .iter()
            .find(|f| f.sex == FlySex::Female)
            .unwrap()
            .eggs_remaining;

        assert!(
            remaining < initial_eggs,
            "Eggs remaining should decrease after oviposition: was {}, now {}",
            initial_eggs,
            remaining
        );
    }

    #[test]
    fn hatching_uses_individual_embryo_state() {
        let mut pop = FlyPopulation::new(61);
        let mut cluster = test_egg_cluster((5.0, 5.0), 3, 1.0);
        cluster.embryos[0].age_hours = EMBRYO_DURATION_H_25C + 1.0;
        cluster.embryos[0].viability = 1.0;
        cluster.embryos[1].age_hours = EMBRYO_DURATION_H_25C - 4.0;
        cluster.embryos[1].viability = 0.95;
        cluster.embryos[2].age_hours = EMBRYO_DURATION_H_25C + 2.0;
        cluster.embryos[2].viability = 0.0;
        cluster.refresh_summary();
        pop.egg_clusters.push(cluster);

        pop.hatch_eggs(25.0, 0.8);

        assert_eq!(
            pop.flies.len(),
            1,
            "only viable mature embryos should hatch"
        );
        assert_eq!(
            pop.egg_clusters[0].embryos.len(),
            1,
            "immature embryos should remain in the clutch"
        );
        assert!(
            matches!(pop.flies[0].stage, FlyLifeStage::Larva { instar: 1, .. }),
            "hatched embryos should enter larval stage directly"
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
        pop.flies
            .push(Fly::new_adult(0, FlySex::Male, (0.0, 0.0, 0.0)));
        pop.flies
            .push(Fly::new_adult(1, FlySex::Female, (0.0, 0.0, 0.0)));

        // Add 1 larva
        let mut larva = Fly::new_embryo(2, FlySex::Male, (0.0, 0.0, 0.0));
        larva.stage = FlyLifeStage::Larva {
            instar: 2,
            age_hours: 5.0,
        };
        pop.flies.push(larva);

        // Add 1 pupa
        let mut pupa = Fly::new_embryo(3, FlySex::Female, (0.0, 0.0, 0.0));
        pupa.stage = FlyLifeStage::Pupa { age_hours: 10.0 };
        pop.flies.push(pupa);

        // Add 1 egg cluster
        pop.egg_clusters.push(test_egg_cluster((0.0, 0.0), 12, 1.0));

        pop.next_id = 4;

        let census = pop.stage_census();
        assert_eq!(census.adults, 2);
        assert_eq!(census.larvae, 1);
        assert_eq!(census.pupae, 1);
        assert_eq!(census.embryos, 12);
        assert_eq!(census.egg_clusters, 1);
        assert_eq!(census.total_eggs, 12);
        assert_eq!(census.total_individuals(), 16);
    }
}
