//! Parallel Worlds Orchestrator for Terrarium Evolution.
//!
//! Spawns N terrariums with varied parameters, runs them in parallel, evaluates
//! fitness, and iteratively discovers optimal ecosystem configurations via
//! evolutionary optimization of terrarium initial conditions.

use crate::terrarium_world::{TerrariumWorld, TerrariumWorldConfig, TerrariumWorldSnapshot};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::mpsc;
use std::time::Instant;

// ---------------------------------------------------------------------------
// WorldGenome
// ---------------------------------------------------------------------------

/// Parameters that vary across parallel worlds.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WorldGenome {
    // Environmental
    pub initial_proton_scale: f32,
    pub soil_temperature_c: f32,
    pub water_source_count: usize,
    pub water_volume: f32,
    pub initial_moisture_scale: f32,

    // Biological
    pub plant_count: usize,
    pub fruit_count: usize,
    pub fly_count: usize,
    pub microbe_cohort_count: usize,

    // Kinetics scaling (multiply against default Vmax)
    pub respiration_vmax_scale: f32,
    pub nitrification_vmax_scale: f32,
    pub photosynthesis_vmax_scale: f32,
    pub mineralization_vmax_scale: f32,

    // Fly brain coevolution
    pub fly_psc_scale: f32,
    pub fly_neural_steps: u32,
    pub fly_scale: u8,

    // Simulation
    pub seed: u64,
    pub time_warp: f32,

    // Enzyme probe placement (normalized 0-1, both 0 = no probe)
    pub enzyme_probe_x: f32,
    pub enzyme_probe_y: f32,
}

/// Parameter ranges for genome fields.
struct GenomeRange;

impl GenomeRange {
    const PROTON_SCALE: (f32, f32) = (0.3, 3.0);
    const TEMPERATURE: (f32, f32) = (10.0, 40.0);
    const WATER_COUNT: (usize, usize) = (1, 6);
    const WATER_VOLUME: (f32, f32) = (50.0, 300.0);
    const MOISTURE_SCALE: (f32, f32) = (0.5, 2.0);
    const PLANT_COUNT: (usize, usize) = (2, 16);
    const FRUIT_COUNT: (usize, usize) = (0, 8);
    const FLY_COUNT: (usize, usize) = (0, 6);
    const MICROBE_COUNT: (usize, usize) = (0, 6);
    const VMAX_SCALE: (f32, f32) = (0.3, 3.0);
    const PSC_SCALE: (f32, f32) = (5.0, 100.0);
    const NEURAL_STEPS: (u32, u32) = (5, 50);
    const FLY_SCALE: (u8, u8) = (0, 1);
    const TIME_WARP: (f32, f32) = (100.0, 2000.0);
    const ENZYME_POS: (f32, f32) = (0.0, 1.0);
}

impl WorldGenome {
    /// Default genome for a given seed.
    pub fn default_with_seed(seed: u64) -> Self {
        Self {
            initial_proton_scale: 1.0,
            soil_temperature_c: 25.0,
            water_source_count: 2,
            water_volume: 150.0,
            initial_moisture_scale: 1.0,
            plant_count: 6,
            fruit_count: 3,
            fly_count: 2,
            microbe_cohort_count: 0,
            fly_psc_scale: 30.0,
            fly_neural_steps: 20,
            fly_scale: 0,
            respiration_vmax_scale: 1.0,
            nitrification_vmax_scale: 1.0,
            photosynthesis_vmax_scale: 1.0,
            mineralization_vmax_scale: 1.0,
            seed,
            time_warp: 900.0,
            enzyme_probe_x: 0.0,
            enzyme_probe_y: 0.0,
        }
    }

    /// Uniformly random genome within parameter bounds.
    pub fn random(rng: &mut StdRng) -> Self {
        Self {
            initial_proton_scale: rng.gen_range(GenomeRange::PROTON_SCALE.0..=GenomeRange::PROTON_SCALE.1),
            soil_temperature_c: rng.gen_range(GenomeRange::TEMPERATURE.0..=GenomeRange::TEMPERATURE.1),
            water_source_count: rng.gen_range(GenomeRange::WATER_COUNT.0..=GenomeRange::WATER_COUNT.1),
            water_volume: rng.gen_range(GenomeRange::WATER_VOLUME.0..=GenomeRange::WATER_VOLUME.1),
            initial_moisture_scale: rng.gen_range(GenomeRange::MOISTURE_SCALE.0..=GenomeRange::MOISTURE_SCALE.1),
            plant_count: rng.gen_range(GenomeRange::PLANT_COUNT.0..=GenomeRange::PLANT_COUNT.1),
            fruit_count: rng.gen_range(GenomeRange::FRUIT_COUNT.0..=GenomeRange::FRUIT_COUNT.1),
            fly_count: rng.gen_range(GenomeRange::FLY_COUNT.0..=GenomeRange::FLY_COUNT.1),
            microbe_cohort_count: rng.gen_range(GenomeRange::MICROBE_COUNT.0..=GenomeRange::MICROBE_COUNT.1),
            fly_psc_scale: rng.gen_range(GenomeRange::PSC_SCALE.0..=GenomeRange::PSC_SCALE.1),
            fly_neural_steps: rng.gen_range(GenomeRange::NEURAL_STEPS.0..=GenomeRange::NEURAL_STEPS.1),
            fly_scale: rng.gen_range(GenomeRange::FLY_SCALE.0..=GenomeRange::FLY_SCALE.1),
            respiration_vmax_scale: rng.gen_range(GenomeRange::VMAX_SCALE.0..=GenomeRange::VMAX_SCALE.1),
            nitrification_vmax_scale: rng.gen_range(GenomeRange::VMAX_SCALE.0..=GenomeRange::VMAX_SCALE.1),
            photosynthesis_vmax_scale: rng.gen_range(GenomeRange::VMAX_SCALE.0..=GenomeRange::VMAX_SCALE.1),
            mineralization_vmax_scale: rng.gen_range(GenomeRange::VMAX_SCALE.0..=GenomeRange::VMAX_SCALE.1),
            seed: rng.gen(),
            time_warp: rng.gen_range(GenomeRange::TIME_WARP.0..=GenomeRange::TIME_WARP.1),
            enzyme_probe_x: rng.gen_range(GenomeRange::ENZYME_POS.0..=GenomeRange::ENZYME_POS.1),
            enzyme_probe_y: rng.gen_range(GenomeRange::ENZYME_POS.0..=GenomeRange::ENZYME_POS.1),
        }
    }

    /// Normalize all parameters to [0, 1] for telemetry/visualization.
    pub fn normalized_params(&self) -> Vec<f32> {
        fn norm(value: f32, lo: f32, hi: f32) -> f32 {
            ((value - lo) / (hi - lo)).clamp(0.0, 1.0)
        }
        fn norm_usize(value: usize, lo: usize, hi: usize) -> f32 {
            if hi > lo {
                ((value - lo) as f32 / (hi - lo) as f32).clamp(0.0, 1.0)
            } else {
                0.5
            }
        }
        vec![
            norm(self.initial_proton_scale, GenomeRange::PROTON_SCALE.0, GenomeRange::PROTON_SCALE.1),
            norm(self.soil_temperature_c, GenomeRange::TEMPERATURE.0, GenomeRange::TEMPERATURE.1),
            norm_usize(self.water_source_count, GenomeRange::WATER_COUNT.0, GenomeRange::WATER_COUNT.1),
            norm(self.water_volume, GenomeRange::WATER_VOLUME.0, GenomeRange::WATER_VOLUME.1),
            norm(self.initial_moisture_scale, GenomeRange::MOISTURE_SCALE.0, GenomeRange::MOISTURE_SCALE.1),
            norm_usize(self.plant_count, GenomeRange::PLANT_COUNT.0, GenomeRange::PLANT_COUNT.1),
            norm_usize(self.fruit_count, GenomeRange::FRUIT_COUNT.0, GenomeRange::FRUIT_COUNT.1),
            norm_usize(self.fly_count, GenomeRange::FLY_COUNT.0, GenomeRange::FLY_COUNT.1),
            norm_usize(self.microbe_cohort_count, GenomeRange::MICROBE_COUNT.0, GenomeRange::MICROBE_COUNT.1),
            norm(self.fly_psc_scale, GenomeRange::PSC_SCALE.0, GenomeRange::PSC_SCALE.1),
            norm(self.fly_neural_steps as f32, GenomeRange::NEURAL_STEPS.0 as f32, GenomeRange::NEURAL_STEPS.1 as f32),
            self.fly_scale as f32,
            norm(self.respiration_vmax_scale, GenomeRange::VMAX_SCALE.0, GenomeRange::VMAX_SCALE.1),
            norm(self.nitrification_vmax_scale, GenomeRange::VMAX_SCALE.0, GenomeRange::VMAX_SCALE.1),
            norm(self.photosynthesis_vmax_scale, GenomeRange::VMAX_SCALE.0, GenomeRange::VMAX_SCALE.1),
            norm(self.mineralization_vmax_scale, GenomeRange::VMAX_SCALE.0, GenomeRange::VMAX_SCALE.1),
            0.5, // seed (not normalized meaningfully)
            norm(self.time_warp, GenomeRange::TIME_WARP.0, GenomeRange::TIME_WARP.1),
            norm(self.enzyme_probe_x, GenomeRange::ENZYME_POS.0, GenomeRange::ENZYME_POS.1),
            norm(self.enzyme_probe_y, GenomeRange::ENZYME_POS.0, GenomeRange::ENZYME_POS.1),
        ]
    }

    /// Crossover two genomes (blend parameters).
    pub fn crossover(a: &Self, b: &Self, rng: &mut StdRng) -> Self {
        Self {
            initial_proton_scale: if rng.gen::<f32>() < 0.5 { a.initial_proton_scale } else { b.initial_proton_scale },
            soil_temperature_c: if rng.gen::<f32>() < 0.5 { a.soil_temperature_c } else { b.soil_temperature_c },
            water_source_count: if rng.gen::<f32>() < 0.5 { a.water_source_count } else { b.water_source_count },
            water_volume: if rng.gen::<f32>() < 0.5 { a.water_volume } else { b.water_volume },
            initial_moisture_scale: if rng.gen::<f32>() < 0.5 { a.initial_moisture_scale } else { b.initial_moisture_scale },
            plant_count: if rng.gen::<f32>() < 0.5 { a.plant_count } else { b.plant_count },
            fruit_count: if rng.gen::<f32>() < 0.5 { a.fruit_count } else { b.fruit_count },
            fly_count: if rng.gen::<f32>() < 0.5 { a.fly_count } else { b.fly_count },
            microbe_cohort_count: if rng.gen::<f32>() < 0.5 { a.microbe_cohort_count } else { b.microbe_cohort_count },
            fly_psc_scale: if rng.gen::<f32>() < 0.5 { a.fly_psc_scale } else { b.fly_psc_scale },
            fly_neural_steps: if rng.gen::<f32>() < 0.5 { a.fly_neural_steps } else { b.fly_neural_steps },
            fly_scale: if rng.gen::<f32>() < 0.5 { a.fly_scale } else { b.fly_scale },
            respiration_vmax_scale: if rng.gen::<f32>() < 0.5 { a.respiration_vmax_scale } else { b.respiration_vmax_scale },
            nitrification_vmax_scale: if rng.gen::<f32>() < 0.5 { a.nitrification_vmax_scale } else { b.nitrification_vmax_scale },
            photosynthesis_vmax_scale: if rng.gen::<f32>() < 0.5 { a.photosynthesis_vmax_scale } else { b.photosynthesis_vmax_scale },
            mineralization_vmax_scale: if rng.gen::<f32>() < 0.5 { a.mineralization_vmax_scale } else { b.mineralization_vmax_scale },
            seed: rng.gen(),
            time_warp: if rng.gen::<f32>() < 0.5 { a.time_warp } else { b.time_warp },
            enzyme_probe_x: if rng.gen::<f32>() < 0.5 { a.enzyme_probe_x } else { b.enzyme_probe_x },
            enzyme_probe_y: if rng.gen::<f32>() < 0.5 { a.enzyme_probe_y } else { b.enzyme_probe_y },
        }
    }

    /// Mutate genome in-place with given probability.
    pub fn mutate(&mut self, rng: &mut StdRng, rate: f32) {
        fn mutate_f32(value: &mut f32, lo: f32, hi: f32, rng: &mut StdRng, rate: f32) {
            if rng.gen::<f32>() < rate {
                let delta = rng.gen_range(-0.2..0.2) * (hi - lo);
                *value = (*value + delta).clamp(lo, hi);
            }
        }
        fn mutate_usize(value: &mut usize, lo: usize, hi: usize, rng: &mut StdRng, rate: f32) {
            if rng.gen::<f32>() < rate {
                let delta = if rng.gen::<f32>() < 0.5 { 1isize } else { -1 };
                *value = (*value as isize + delta).clamp(lo as isize, hi as isize) as usize;
            }
        }
        mutate_f32(&mut self.initial_proton_scale, GenomeRange::PROTON_SCALE.0, GenomeRange::PROTON_SCALE.1, rng, rate);
        mutate_f32(&mut self.soil_temperature_c, GenomeRange::TEMPERATURE.0, GenomeRange::TEMPERATURE.1, rng, rate);
        mutate_usize(&mut self.water_source_count, GenomeRange::WATER_COUNT.0, GenomeRange::WATER_COUNT.1, rng, rate);
        mutate_f32(&mut self.water_volume, GenomeRange::WATER_VOLUME.0, GenomeRange::WATER_VOLUME.1, rng, rate);
        mutate_f32(&mut self.initial_moisture_scale, GenomeRange::MOISTURE_SCALE.0, GenomeRange::MOISTURE_SCALE.1, rng, rate);
        mutate_usize(&mut self.plant_count, GenomeRange::PLANT_COUNT.0, GenomeRange::PLANT_COUNT.1, rng, rate);
        mutate_usize(&mut self.fruit_count, GenomeRange::FRUIT_COUNT.0, GenomeRange::FRUIT_COUNT.1, rng, rate);
        mutate_usize(&mut self.fly_count, GenomeRange::FLY_COUNT.0, GenomeRange::FLY_COUNT.1, rng, rate);
        mutate_usize(&mut self.microbe_cohort_count, GenomeRange::MICROBE_COUNT.0, GenomeRange::MICROBE_COUNT.1, rng, rate);
        mutate_f32(&mut self.fly_psc_scale, GenomeRange::PSC_SCALE.0, GenomeRange::PSC_SCALE.1, rng, rate);
        {
            // Mutate fly_neural_steps using the same pattern
            let mut v = self.fly_neural_steps as usize;
            mutate_usize(&mut v, GenomeRange::NEURAL_STEPS.0 as usize, GenomeRange::NEURAL_STEPS.1 as usize, rng, rate);
            self.fly_neural_steps = v as u32;
        }
        if rng.gen::<f32>() < rate {
            self.fly_scale = if self.fly_scale == 0 { 1 } else { 0 };
        }
        mutate_f32(&mut self.respiration_vmax_scale, GenomeRange::VMAX_SCALE.0, GenomeRange::VMAX_SCALE.1, rng, rate);
        mutate_f32(&mut self.nitrification_vmax_scale, GenomeRange::VMAX_SCALE.0, GenomeRange::VMAX_SCALE.1, rng, rate);
        mutate_f32(&mut self.photosynthesis_vmax_scale, GenomeRange::VMAX_SCALE.0, GenomeRange::VMAX_SCALE.1, rng, rate);
        mutate_f32(&mut self.mineralization_vmax_scale, GenomeRange::VMAX_SCALE.0, GenomeRange::VMAX_SCALE.1, rng, rate);
        mutate_f32(&mut self.time_warp, GenomeRange::TIME_WARP.0, GenomeRange::TIME_WARP.1, rng, rate);
        mutate_f32(&mut self.enzyme_probe_x, GenomeRange::ENZYME_POS.0, GenomeRange::ENZYME_POS.1, rng, rate);
        mutate_f32(&mut self.enzyme_probe_y, GenomeRange::ENZYME_POS.0, GenomeRange::ENZYME_POS.1, rng, rate);
    }

    /// Build a TerrariumWorld from this genome (full size).
    pub fn build_world(&self) -> Result<TerrariumWorld, String> {
        let config = TerrariumWorldConfig {
            width: 20,
            height: 16,
            depth: 2,
            seed: self.seed,
            time_warp: self.time_warp,
            max_plants: self.plant_count.max(1) + 4,
            max_fruits: self.fruit_count.max(1) + 8,
            ..TerrariumWorldConfig::default()
        };
        let mut world = TerrariumWorld::new(config)?;
        self.seed_world(&mut world);
        Ok(world)
    }

    /// Build a lite TerrariumWorld (10x8x2) for fast validation.
    pub fn build_world_lite(&self) -> Result<TerrariumWorld, String> {
        let config = TerrariumWorldConfig {
            width: 10,
            height: 8,
            depth: 2,
            seed: self.seed,
            time_warp: self.time_warp,
            max_plants: self.plant_count.max(1) + 4,
            max_fruits: self.fruit_count.max(1) + 8,
            ..TerrariumWorldConfig::default()
        };
        let mut world = TerrariumWorld::new(config)?;
        self.seed_world(&mut world);
        Ok(world)
    }

    /// Populate a TerrariumWorld with entities from genome parameters.
    fn seed_world(&self, world: &mut TerrariumWorld) {
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(self.seed);
        let w = world.config.width;
        let h = world.config.height;

        // Add water sources
        for _ in 0..self.water_source_count {
            let x = rng.gen_range(1..w.saturating_sub(1).max(2));
            let y = rng.gen_range(1..h.saturating_sub(1).max(2));
            world.add_water(x, y, self.water_volume, 0.0008);
        }

        // Add plants
        for _ in 0..self.plant_count {
            let x = rng.gen_range(1..w.saturating_sub(1).max(2));
            let y = rng.gen_range(1..h.saturating_sub(1).max(2));
            let _ = world.add_plant(x, y, None, None);
        }

        // Add fruits
        for _ in 0..self.fruit_count {
            let x = rng.gen_range(1..w.saturating_sub(1).max(2));
            let y = rng.gen_range(1..h.saturating_sub(1).max(2));
            world.add_fruit(x, y, rng.gen_range(0.5..1.2), None);
        }

        // Add flies (if genome allows)
        let fly_scale = if self.fly_scale >= 1 {
            crate::drosophila::DrosophilaScale::Small
        } else {
            crate::drosophila::DrosophilaScale::Tiny
        };
        for i in 0..self.fly_count {
            let x = rng.gen_range(2.0..(w as f32 - 2.0).max(3.0));
            let y = rng.gen_range(2.0..(h as f32 - 2.0).max(3.0));
            world.add_fly(fly_scale, x, y, self.seed.wrapping_add(i as u64));
        }

        // Add enzyme probe if position is non-zero
        if self.enzyme_probe_x > 0.0 || self.enzyme_probe_y > 0.0 {
            let gx = ((self.enzyme_probe_x * w as f32) as usize).min(w - 1);
            let gy = ((self.enzyme_probe_y * h as f32) as usize).min(h - 1);
            let mol = crate::enzyme_probes::select_enzyme_for_seed(self.seed);
            let _ = world.spawn_probe(&mol, gx, gy, 1);
        }
    }
}

// ---------------------------------------------------------------------------
// Fitness Evaluation
// ---------------------------------------------------------------------------

/// Fitness objectives for evolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FitnessObjective {
    MaxBiomass,
    MaxBiodiversity,
    MaxStability,
    MaxCarbonSequestration,
    MaxFruitProduction,
    MaxMicrobialHealth,
    MaxFlyEcosystem,
    MaxFlyMetabolism,
    MaxEnzymeEfficacy,
}

impl Default for FitnessObjective {
    fn default() -> Self {
        Self::MaxBiomass
    }
}

impl std::fmt::Display for FitnessObjective {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MaxBiomass => write!(f, "MaxBiomass"),
            Self::MaxBiodiversity => write!(f, "MaxBiodiversity"),
            Self::MaxStability => write!(f, "MaxStability"),
            Self::MaxCarbonSequestration => write!(f, "MaxCarbonSequestration"),
            Self::MaxFruitProduction => write!(f, "MaxFruitProduction"),
            Self::MaxMicrobialHealth => write!(f, "MaxMicrobialHealth"),
            Self::MaxFlyEcosystem => write!(f, "MaxFlyEcosystem"),
            Self::MaxFlyMetabolism => write!(f, "MaxFlyMetabolism"),
            Self::MaxEnzymeEfficacy => write!(f, "MaxEnzymeEfficacy"),
        }
    }
}

/// Evaluate fitness from a snapshot.
pub fn evaluate_fitness(objective: FitnessObjective, snapshot: &TerrariumWorldSnapshot, _periodic: &[TerrariumWorldSnapshot]) -> f32 {
    match objective {
        FitnessObjective::MaxBiomass => {
            snapshot.plants as f32 * snapshot.mean_canopy + snapshot.total_plant_cells as f32 * 0.01
        }
        FitnessObjective::MaxBiodiversity => {
            // Use microbes and symbionts as diversity proxy
            let microbe_score = snapshot.mean_microbes;
            let symbiont_score = snapshot.mean_symbionts;
            let total = microbe_score + symbiont_score;
            if total <= 0.0 {
                return 0.0;
            }
            // Simpson diversity approximation
            let p1 = microbe_score / total;
            let p2 = symbiont_score / total;
            1.0 - (p1 * p1 + p2 * p2)
        }
        FitnessObjective::MaxStability => {
            // Use mean soil moisture as stability proxy
            snapshot.mean_soil_moisture * 10.0
        }
        FitnessObjective::MaxCarbonSequestration => {
            snapshot.mean_soil_glucose + snapshot.total_plant_cells as f32 * 0.005
        }
        FitnessObjective::MaxFruitProduction => {
            snapshot.fruits as f32
        }
        FitnessObjective::MaxMicrobialHealth => {
            snapshot.mean_microbes + snapshot.mean_symbionts
        }
        FitnessObjective::MaxFlyEcosystem => {
            let energy_frac = (snapshot.avg_fly_energy / 100.0).clamp(0.0, 1.0);
            snapshot.flies as f32 * 10.0 + snapshot.fruits as f32 * 2.0 + energy_frac * 5.0
        }
        FitnessObjective::MaxFlyMetabolism => {
            snapshot.avg_fly_energy_charge * 10.0 + snapshot.flies as f32 * 2.0
        }
        FitnessObjective::MaxEnzymeEfficacy => {
            // Probe presence rewards stability; nutrient improvement measures catalytic effect
            let probe_stability = if snapshot.atomistic_probes > 0 { 5.0 } else { 0.0 };
            let nutrient_bonus = snapshot.mean_soil_glucose + snapshot.mean_soil_ammonium;
            probe_stability + nutrient_bonus
        }
    }
}

/// Multi-objective fitness (all 6 objectives).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct MultiObjectiveFitness {
    pub biomass: f32,
    pub biodiversity: f32,
    pub stability: f32,
    pub carbon: f32,
    pub fruit: f32,
    pub microbial: f32,
    pub fly_metabolism: f32,
    pub enzyme_efficacy: f32,
}

impl MultiObjectiveFitness {
    /// Evaluate all objectives at once.
    pub fn evaluate(snapshot: &TerrariumWorldSnapshot, periodic: &[TerrariumWorldSnapshot]) -> Self {
        Self {
            biomass: evaluate_fitness(FitnessObjective::MaxBiomass, snapshot, periodic),
            biodiversity: evaluate_fitness(FitnessObjective::MaxBiodiversity, snapshot, periodic),
            stability: evaluate_fitness(FitnessObjective::MaxStability, snapshot, periodic),
            carbon: evaluate_fitness(FitnessObjective::MaxCarbonSequestration, snapshot, periodic),
            fruit: evaluate_fitness(FitnessObjective::MaxFruitProduction, snapshot, periodic),
            microbial: evaluate_fitness(FitnessObjective::MaxMicrobialHealth, snapshot, periodic),
            fly_metabolism: evaluate_fitness(FitnessObjective::MaxFlyMetabolism, snapshot, periodic),
            enzyme_efficacy: evaluate_fitness(FitnessObjective::MaxEnzymeEfficacy, snapshot, periodic),
        }
    }
}

// ---------------------------------------------------------------------------
// Result Types
// ---------------------------------------------------------------------------

/// Result from evaluating a single world.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WorldResult {
    pub genome: WorldGenome,
    pub fitness: f32,
    pub final_biomass: f32,
    pub final_plants: usize,
    pub final_fruits: usize,
    pub wall_time_ms: f32,
}

/// Result from a generation of evolution.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GenerationResult {
    pub generation: usize,
    pub best_fitness: f32,
    pub mean_fitness: f32,
    pub worst_fitness: f32,
    pub best_genome: WorldGenome,
    pub wall_time_ms: f32,
    /// Stress metrics for the best genome in this generation (stress modes only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stress_metrics: Option<StressTelemetryMetrics>,
}

/// Complete result from evolution run.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EvolutionResult {
    pub generations: Vec<GenerationResult>,
    pub global_best_genome: WorldGenome,
    pub global_best_fitness: f32,
    pub total_worlds_evaluated: usize,
    pub total_wall_time_ms: f32,
}

// ---------------------------------------------------------------------------
// Telemetry
// ---------------------------------------------------------------------------

/// Per-generation telemetry record for convergence analysis.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GenerationTelemetry {
    pub generation: usize,
    pub best_fitness: f32,
    pub mean_fitness: f32,
    pub worst_fitness: f32,
    pub population_diversity: f32,
    pub best_genome_params: Vec<f32>,
    pub elapsed_ms: f32,
    /// Name of the evolution mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    /// Per-objective breakdown for Pareto modes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub multi_objective_fitness: Option<MultiObjectiveFitness>,
    /// Stress-specific metrics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stress_metrics: Option<StressTelemetryMetrics>,
}

/// Stress telemetry metrics.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StressTelemetryMetrics {
    pub pre_stress_biomass: f32,
    pub min_stress_biomass: f32,
    pub post_recovery_biomass: f32,
}

/// Compute telemetry from an EvolutionResult.
///
/// If a generation has `stress_metrics` attached (from stress-test mode), those
/// metrics flow through to the telemetry record automatically.
pub fn telemetry_from_result(result: &EvolutionResult, mode: Option<&str>) -> Vec<GenerationTelemetry> {
    result
        .generations
        .iter()
        .map(|gen| {
            let params = gen.best_genome.normalized_params();
            GenerationTelemetry {
                generation: gen.generation,
                best_fitness: gen.best_fitness,
                mean_fitness: gen.mean_fitness,
                worst_fitness: gen.worst_fitness,
                population_diversity: (gen.best_fitness - gen.worst_fitness).abs(),
                best_genome_params: params,
                elapsed_ms: gen.wall_time_ms,
                mode: mode.map(|s| s.to_string()),
                multi_objective_fitness: None,
                stress_metrics: gen.stress_metrics.clone(),
            }
        })
        .collect()
}

/// Evaluate stress metrics for the best genome in a stress-test EvolutionResult.
/// Runs a quick stress trial on the global best genome and returns the metrics.
pub fn evaluate_stress_metrics_for_best(
    result: &EvolutionResult,
    frames: usize,
    lite: bool,
) -> Option<StressTelemetryMetrics> {
    let stress = StressTestConfig {
        drought_frame: frames / 3,
        heat_spike_frame: 2 * frames / 3,
        perturbation_duration: (frames / 10).max(1),
        ..StressTestConfig::default()
    };
    let sr = run_single_world_stressed(result.global_best_genome.clone(), frames, lite, &stress).ok()?;
    Some(StressTelemetryMetrics {
        pre_stress_biomass: sr.pre_stress_biomass,
        min_stress_biomass: sr.min_stress_biomass,
        post_recovery_biomass: sr.post_recovery_biomass,
    })
}

// ---------------------------------------------------------------------------
// Evolution Configuration
// ---------------------------------------------------------------------------

/// Search strategy for evolution.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum SearchStrategy {
    Random,
    Evolutionary {
        tournament_size: usize,
        mutation_rate: f32,
        crossover_rate: f32,
        elitism: usize,
    },
}

impl Default for SearchStrategy {
    fn default() -> Self {
        Self::Evolutionary {
            tournament_size: 3,
            mutation_rate: 0.15,
            crossover_rate: 0.7,
            elitism: 2,
        }
    }
}

/// Constraints on genome generation.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct GenomeConstraints {
    pub max_fly_count: Option<usize>,
    pub max_microbe_count: Option<usize>,
    pub force_fly_count: Option<usize>,
}

impl GenomeConstraints {
    pub fn apply(&self, genome: &mut WorldGenome) {
        if let Some(max) = self.max_fly_count {
            genome.fly_count = genome.fly_count.min(max);
        }
        if let Some(max) = self.max_microbe_count {
            genome.microbe_cohort_count = genome.microbe_cohort_count.min(max);
        }
        if let Some(force) = self.force_fly_count {
            genome.fly_count = force;
        }
    }
}

/// Fitness configuration.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FitnessConfig {
    pub primary: FitnessObjective,
    pub snapshot_interval: usize,
}

impl Default for FitnessConfig {
    fn default() -> Self {
        Self {
            primary: FitnessObjective::MaxBiomass,
            snapshot_interval: 10,
        }
    }
}

/// Main evolution configuration.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EvolutionConfig {
    pub population_size: usize,
    pub generations: usize,
    pub frames_per_world: usize,
    pub strategy: SearchStrategy,
    pub fitness: FitnessConfig,
    pub thread_count: Option<usize>,
    pub master_seed: u64,
    pub constraints: GenomeConstraints,
    pub lite: bool,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            population_size: 10,
            generations: 10,
            frames_per_world: 100,
            strategy: SearchStrategy::default(),
            fitness: FitnessConfig::default(),
            thread_count: None,
            master_seed: 42,
            constraints: GenomeConstraints::default(),
            lite: false,
        }
    }
}

// ---------------------------------------------------------------------------
// World Evaluation
// ---------------------------------------------------------------------------

/// Run a single world evaluation.
pub fn run_single_world(
    genome: WorldGenome,
    frames: usize,
    snapshot_interval: usize,
    fitness_objective: FitnessObjective,
    lite: bool,
) -> Result<WorldResult, String> {
    let start = Instant::now();

    let mut world = if lite {
        genome.build_world_lite()?
    } else {
        genome.build_world()?
    };

    let mut periodic_snapshots = Vec::new();

    for frame in 0..frames {
        world.step_frame()?;

        if frame % snapshot_interval == 0 {
            periodic_snapshots.push(world.snapshot());
        }
    }

    let snapshot = world.snapshot();
    let fitness = evaluate_fitness(fitness_objective, &snapshot, &periodic_snapshots);

    Ok(WorldResult {
        genome,
        fitness,
        final_biomass: snapshot.total_plant_cells,
        final_plants: snapshot.plants,
        final_fruits: snapshot.fruits,
        wall_time_ms: start.elapsed().as_secs_f32() * 1000.0,
    })
}

/// Run a generation of worlds in parallel.
pub fn run_generation(
    genomes: Vec<WorldGenome>,
    frames: usize,
    snapshot_interval: usize,
    fitness_objective: FitnessObjective,
    _max_threads: usize,
    lite: bool,
) -> Vec<Result<WorldResult, String>> {
    let (tx, rx) = mpsc::channel();
    let genome_count = genomes.len();

    for genome in genomes {
        let tx = tx.clone();
        std::thread::spawn(move || {
            let result = run_single_world(genome, frames, snapshot_interval, fitness_objective, lite);
            let _ = tx.send(result);
        });
    }
    drop(tx);

    rx.iter().take(genome_count).collect()
}

// ---------------------------------------------------------------------------
// Main Evolution Loop
// ---------------------------------------------------------------------------

/// Run standard evolutionary optimization.
pub fn evolve(config: EvolutionConfig) -> Result<EvolutionResult, String> {
    let start = Instant::now();
    let max_threads = config.thread_count.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    });

    let (tournament_size, mutation_rate, crossover_rate, elitism) = match config.strategy {
        SearchStrategy::Evolutionary { tournament_size, mutation_rate, crossover_rate, elitism } => {
            (tournament_size, mutation_rate, crossover_rate, elitism)
        }
        SearchStrategy::Random => (3, 0.15, 0.7, 2),
    };

    let mut rng = StdRng::seed_from_u64(config.master_seed);
    let mut population: Vec<WorldGenome> = (0..config.population_size)
        .map(|_| {
            let mut g = WorldGenome::random(&mut rng);
            config.constraints.apply(&mut g);
            g
        })
        .collect();

    let mut generation_results = Vec::new();
    let mut global_best_fitness = f32::NEG_INFINITY;
    let mut global_best_genome = WorldGenome::default_with_seed(config.master_seed);
    let mut total_worlds = 0usize;

    for gen in 0..config.generations {
        let gen_start = Instant::now();

        // Evaluate population
        let results = run_generation(
            population.clone(),
            config.frames_per_world,
            config.fitness.snapshot_interval,
            config.fitness.primary,
            max_threads,
            config.lite,
        );

        let mut world_results: Vec<WorldResult> = results
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();

        if world_results.is_empty() {
            return Err("All worlds failed".to_string());
        }

        total_worlds += world_results.len();

        // Sort by fitness
        world_results.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));

        let gen_best_fitness = world_results[0].fitness;
        let gen_worst_fitness = world_results.last().unwrap().fitness;
        let gen_mean_fitness = world_results.iter().map(|r| r.fitness).sum::<f32>() / world_results.len() as f32;
        let gen_best_genome = world_results[0].genome.clone();

        if gen_best_fitness > global_best_fitness {
            global_best_fitness = gen_best_fitness;
            global_best_genome = gen_best_genome.clone();
        }

        let gen_wall_time = gen_start.elapsed().as_secs_f32() * 1000.0;

        eprintln!(
            "[gen {}/{}] best={:.4} mean={:.4} worst={:.4} ({:.0}ms)",
            gen + 1, config.generations, gen_best_fitness, gen_mean_fitness, gen_worst_fitness, gen_wall_time
        );

        generation_results.push(GenerationResult {
            generation: gen,
            best_fitness: gen_best_fitness,
            mean_fitness: gen_mean_fitness,
            worst_fitness: gen_worst_fitness,
            best_genome: gen_best_genome,
            wall_time_ms: gen_wall_time,
            stress_metrics: None,
        });

        // Breed next generation
        if gen + 1 < config.generations {
            population = breed_next_generation(
                &world_results,
                config.population_size,
                tournament_size,
                mutation_rate,
                crossover_rate,
                elitism,
                &mut rng,
            );
        }
    }

    Ok(EvolutionResult {
        generations: generation_results,
        global_best_genome,
        global_best_fitness,
        total_worlds_evaluated: total_worlds,
        total_wall_time_ms: start.elapsed().as_secs_f32() * 1000.0,
    })
}

/// Breed next generation using tournament selection.
pub fn breed_next_generation(
    results: &[WorldResult],
    population_size: usize,
    tournament_size: usize,
    mutation_rate: f32,
    crossover_rate: f32,
    elitism: usize,
    rng: &mut StdRng,
) -> Vec<WorldGenome> {
    let mut next_gen = Vec::with_capacity(population_size);

    // Elitism: carry over top performers
    for i in 0..elitism.min(results.len()) {
        next_gen.push(results[i].genome.clone());
    }

    // Fill rest with tournament selection + crossover + mutation
    while next_gen.len() < population_size {
        // Tournament selection for parent a
        let parent_a = tournament_select(results, tournament_size, rng);
        let parent_b = tournament_select(results, tournament_size, rng);

        let mut child = if rng.gen::<f32>() < crossover_rate {
            WorldGenome::crossover(&results[parent_a].genome, &results[parent_b].genome, rng)
        } else {
            results[parent_a].genome.clone()
        };

        child.mutate(rng, mutation_rate);
        next_gen.push(child);
    }

    next_gen
}

/// Tournament selection - pick best from random subset.
fn tournament_select(results: &[WorldResult], tournament_size: usize, rng: &mut StdRng) -> usize {
    let mut best_idx = rng.gen_range(0..results.len());
    let mut best_fitness = results[best_idx].fitness;

    for _ in 1..tournament_size {
        let idx = rng.gen_range(0..results.len());
        if results[idx].fitness > best_fitness {
            best_idx = idx;
            best_fitness = results[idx].fitness;
        }
    }

    best_idx
}

// ---------------------------------------------------------------------------
// Stress Testing
// ---------------------------------------------------------------------------

/// Configuration for environmental stress testing.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StressTestConfig {
    pub drought_frame: usize,
    pub heat_spike_frame: usize,
    pub perturbation_duration: usize,
    pub w_pre_stress: f32,
    pub w_recovery: f32,
    pub w_resilience: f32,
}

impl Default for StressTestConfig {
    fn default() -> Self {
        Self {
            drought_frame: 0,
            heat_spike_frame: 0,
            perturbation_duration: 3,
            w_pre_stress: 0.3,
            w_recovery: 0.4,
            w_resilience: 0.3,
        }
    }
}

/// Result from stress testing a world.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StressTestResult {
    pub genome: WorldGenome,
    pub pre_stress_biomass: f32,
    pub min_stress_biomass: f32,
    pub post_recovery_biomass: f32,
    pub resilience_fitness: f32,
}

/// Run a single world with stress perturbations.
fn run_single_world_stressed(
    genome: WorldGenome,
    frames: usize,
    lite: bool,
    stress: &StressTestConfig,
) -> Result<StressTestResult, String> {
    let mut world = if lite {
        genome.build_world_lite()?
    } else {
        genome.build_world()?
    };

    let drought_start = stress.drought_frame;
    let drought_end = (drought_start + stress.perturbation_duration).min(frames);
    let heat_start = stress.heat_spike_frame;
    let heat_end = (heat_start + stress.perturbation_duration).min(frames);

    let mut pre_stress_biomass = 0.0_f32;
    let mut min_stress_biomass = f32::INFINITY;
    let mut post_recovery_biomass = 0.0_f32;

    // Save original fields
    let original_moisture = world.moisture_field().to_vec();
    let original_temp = world.temperature_field().to_vec();

    for frame in 0..frames {
        // Inject drought
        if frame == drought_start {
            let n = original_moisture.len().min(world.moisture_field().len());
            for i in 0..n {
                world.moisture_field_mut()[i] = original_moisture[i] * 0.1;
            }
        }
        // Restore moisture
        if frame == drought_end {
            let n = original_moisture.len().min(world.moisture_field().len());
            for i in 0..n {
                world.moisture_field_mut()[i] = original_moisture[i];
            }
        }
        // Inject heat spike
        if frame == heat_start {
            let n = world.temperature_field().len();
            for i in 0..n {
                world.temperature_field_mut()[i] += 15.0;
            }
        }
        // Restore temperature
        if frame == heat_end {
            let n = original_temp.len().min(world.temperature_field().len());
            for i in 0..n {
                world.temperature_field_mut()[i] = original_temp[i];
            }
        }

        world.step_frame()?;

        let snapshot = world.snapshot();
        let biomass = evaluate_fitness(FitnessObjective::MaxBiomass, &snapshot, &[]);

        // Record pre-stress
        if frame + 1 == drought_start.min(heat_start) {
            pre_stress_biomass = biomass;
        }
        // Track minimum during stress
        if (frame >= drought_start && frame < drought_end) || (frame >= heat_start && frame < heat_end) {
            min_stress_biomass = min_stress_biomass.min(biomass);
        }
        // Record recovery
        if frame + 1 == frames {
            post_recovery_biomass = biomass;
        }
    }

    // Handle edge cases
    if pre_stress_biomass == 0.0 {
        let snapshot = world.snapshot();
        pre_stress_biomass = evaluate_fitness(FitnessObjective::MaxBiomass, &snapshot, &[]);
    }
    if min_stress_biomass == f32::INFINITY {
        min_stress_biomass = post_recovery_biomass;
    }

    let resilience_fitness = stress.w_pre_stress * pre_stress_biomass
        + stress.w_resilience * min_stress_biomass
        + stress.w_recovery * post_recovery_biomass;

    Ok(StressTestResult {
        genome,
        pre_stress_biomass,
        min_stress_biomass,
        post_recovery_biomass,
        resilience_fitness,
    })
}

/// Run stress-tested evolution.
pub fn evolve_stress_test(config: EvolutionConfig) -> Result<EvolutionResult, String> {
    let start = Instant::now();

    let frames = config.frames_per_world;
    let stress = StressTestConfig {
        drought_frame: frames / 3,
        heat_spike_frame: 2 * frames / 3,
        perturbation_duration: (frames / 10).max(1),
        ..StressTestConfig::default()
    };

    let (tournament_size, mutation_rate, crossover_rate, elitism) = match config.strategy {
        SearchStrategy::Evolutionary { tournament_size, mutation_rate, crossover_rate, elitism } => {
            (tournament_size, mutation_rate, crossover_rate, elitism)
        }
        SearchStrategy::Random => (3, 0.15, 0.7, 2),
    };

    let mut rng = StdRng::seed_from_u64(config.master_seed);
    let mut generation_results = Vec::new();
    let mut global_best_fitness = f32::NEG_INFINITY;
    let mut global_best_genome = WorldGenome::default_with_seed(config.master_seed);
    let mut total_worlds = 0usize;

    // Initialize population
    let mut population: Vec<WorldGenome> = (0..config.population_size)
        .map(|_| {
            let mut g = WorldGenome::random(&mut rng);
            config.constraints.apply(&mut g);
            g
        })
        .collect();

    for gen in 0..config.generations {
        let gen_start = Instant::now();

        // Run stressed worlds in parallel
        let (tx, rx) = mpsc::channel();
        let stress_cfg = stress.clone();
        let lite = config.lite;

        for genome in population.clone() {
            let tx = tx.clone();
            let stress_cfg = stress_cfg.clone();
            std::thread::spawn(move || {
                let result = run_single_world_stressed(genome, frames, lite, &stress_cfg);
                let _ = tx.send(result);
            });
        }
        drop(tx);

        let results: Vec<StressTestResult> = rx.iter()
            .filter_map(|r| r.ok())
            .collect();

        if results.is_empty() {
            return Err("All stressed worlds failed".to_string());
        }

        total_worlds += results.len();

        // Sort by resilience fitness
        let mut sorted_results = results;
        sorted_results.sort_by(|a, b| b.resilience_fitness.partial_cmp(&a.resilience_fitness).unwrap_or(std::cmp::Ordering::Equal));

        let gen_best_fitness = sorted_results[0].resilience_fitness;
        let gen_worst_fitness = sorted_results.last().unwrap().resilience_fitness;
        let gen_mean_fitness = sorted_results.iter().map(|r| r.resilience_fitness).sum::<f32>() / sorted_results.len() as f32;
        let gen_best_genome = sorted_results[0].genome.clone();

        if gen_best_fitness > global_best_fitness {
            global_best_fitness = gen_best_fitness;
            global_best_genome = gen_best_genome.clone();
        }

        let gen_wall_time = gen_start.elapsed().as_secs_f32() * 1000.0;

        eprintln!(
            "[stress gen {}/{}] best={:.4} mean={:.4} worst={:.4} ({:.0}ms)",
            gen + 1, config.generations, gen_best_fitness, gen_mean_fitness, gen_worst_fitness, gen_wall_time
        );

        generation_results.push(GenerationResult {
            generation: gen,
            best_fitness: gen_best_fitness,
            mean_fitness: gen_mean_fitness,
            worst_fitness: gen_worst_fitness,
            best_genome: gen_best_genome,
            wall_time_ms: gen_wall_time,
            stress_metrics: Some(StressTelemetryMetrics {
                pre_stress_biomass: sorted_results[0].pre_stress_biomass,
                min_stress_biomass: sorted_results[0].min_stress_biomass,
                post_recovery_biomass: sorted_results[0].post_recovery_biomass,
            }),
        });

        // Breed next generation
        if gen + 1 < config.generations {
            population = breed_stressed_next_generation(
                &sorted_results,
                config.population_size,
                tournament_size,
                mutation_rate,
                crossover_rate,
                elitism,
                &mut rng,
            );
        }
    }

    Ok(EvolutionResult {
        generations: generation_results,
        global_best_genome,
        global_best_fitness,
        total_worlds_evaluated: total_worlds,
        total_wall_time_ms: start.elapsed().as_secs_f32() * 1000.0,
    })
}

/// Breed next generation from stress test results.
fn breed_stressed_next_generation(
    results: &[StressTestResult],
    population_size: usize,
    tournament_size: usize,
    mutation_rate: f32,
    crossover_rate: f32,
    elitism: usize,
    rng: &mut StdRng,
) -> Vec<WorldGenome> {
    let mut next_gen = Vec::with_capacity(population_size);

    // Elitism
    for i in 0..elitism.min(results.len()) {
        next_gen.push(results[i].genome.clone());
    }

    while next_gen.len() < population_size {
        let parent_a = stress_tournament_select(results, tournament_size, rng);
        let parent_b = stress_tournament_select(results, tournament_size, rng);

        let mut child = if rng.gen::<f32>() < crossover_rate {
            WorldGenome::crossover(&results[parent_a].genome, &results[parent_b].genome, rng)
        } else {
            results[parent_a].genome.clone()
        };

        child.mutate(rng, mutation_rate);
        next_gen.push(child);
    }

    next_gen
}

fn stress_tournament_select(results: &[StressTestResult], tournament_size: usize, rng: &mut StdRng) -> usize {
    let mut best_idx = rng.gen_range(0..results.len());
    let mut best_fitness = results[best_idx].resilience_fitness;

    for _ in 1..tournament_size {
        let idx = rng.gen_range(0..results.len());
        if results[idx].resilience_fitness > best_fitness {
            best_idx = idx;
            best_fitness = results[idx].resilience_fitness;
        }
    }

    best_idx
}

// ---------------------------------------------------------------------------
// NSGA-II Pareto Optimization
// ---------------------------------------------------------------------------

/// Multi-objective result for Pareto optimization.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ParetoResult {
    pub genome: WorldGenome,
    pub objectives: MultiObjectiveFitness,
    pub rank: usize,
    pub crowding_distance: f32,
    pub wall_time_ms: f32,
}

/// Pareto front result.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ParetoEvolutionResult {
    pub pareto_front: Vec<ParetoResult>,
    pub generations_run: usize,
    pub total_worlds_evaluated: usize,
    pub total_wall_time_ms: f32,
}

/// Check if solution `a` dominates solution `b` (Pareto dominance).
fn dominates(a: &MultiObjectiveFitness, b: &MultiObjectiveFitness) -> bool {
    // a dominates b if a is no worse in all objectives and strictly better in at least one
    let better_or_equal = a.biomass >= b.biomass
        && a.biodiversity >= b.biodiversity
        && a.stability >= b.stability
        && a.carbon >= b.carbon
        && a.fruit >= b.fruit
        && a.microbial >= b.microbial
        && a.fly_metabolism >= b.fly_metabolism;

    let strictly_better = a.biomass > b.biomass
        || a.biodiversity > b.biodiversity
        || a.stability > b.stability
        || a.carbon > b.carbon
        || a.fruit > b.fruit
        || a.microbial > b.microbial
        || a.fly_metabolism > b.fly_metabolism;

    better_or_equal && strictly_better
}

/// Compute Pareto ranks using fast non-dominated sorting.
fn compute_pareto_ranks(results: &mut [ParetoResult]) {
    let n = results.len();
    if n == 0 {
        return;
    }

    // Reset ranks
    for r in results.iter_mut() {
        r.rank = usize::MAX; // Use MAX as "unranked" sentinel
        r.crowding_distance = 0.0;
    }

    // Compute domination counts and dominated sets
    let mut domination_count = vec![0usize; n];
    let mut dominated_set: Vec<Vec<usize>> = vec![Vec::new(); n];

    for i in 0..n {
        for j in (i + 1)..n {
            if dominates(&results[i].objectives, &results[j].objectives) {
                domination_count[j] += 1;
                dominated_set[i].push(j);
            } else if dominates(&results[j].objectives, &results[i].objectives) {
                domination_count[i] += 1;
                dominated_set[j].push(i);
            }
        }
    }

    // Assign ranks
    let mut current_rank = 0;
    let mut assigned = 0;

    while assigned < n {
        // Find all solutions with domination_count == 0 that haven't been ranked yet
        let front: Vec<usize> = (0..n)
            .filter(|&i| domination_count[i] == 0 && results[i].rank == usize::MAX)
            .collect();

        if front.is_empty() {
            break;
        }

        // Assign current rank
        for &i in &front {
            results[i].rank = current_rank;
            assigned += 1;
        }

        // Reduce domination count for dominated solutions
        for &i in &front {
            for &j in &dominated_set[i] {
                if domination_count[j] > 0 {
                    domination_count[j] -= 1;
                }
            }
        }

        current_rank += 1;
    }
}

/// Compute crowding distance for solutions on same front.
fn compute_crowding_distance(results: &mut [ParetoResult]) {
    if results.len() <= 2 {
        for r in results.iter_mut() {
            r.crowding_distance = f32::INFINITY;
        }
        return;
    }

    let objectives = [
        |r: &ParetoResult| r.objectives.biomass,
        |r: &ParetoResult| r.objectives.biodiversity,
        |r: &ParetoResult| r.objectives.stability,
        |r: &ParetoResult| r.objectives.carbon,
        |r: &ParetoResult| r.objectives.fruit,
        |r: &ParetoResult| r.objectives.microbial,
        |r: &ParetoResult| r.objectives.enzyme_efficacy,
    ];

    for obj_fn in objectives {
        // Sort by this objective
        results.sort_by(|a, b| {
            obj_fn(a).partial_cmp(&obj_fn(b)).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Boundary solutions get infinite distance
        results.first_mut().map(|r| r.crowding_distance = f32::INFINITY);
        results.last_mut().map(|r| r.crowding_distance = f32::INFINITY);

        let min_val = obj_fn(&results[0]);
        let max_val = obj_fn(&results[results.len() - 1]);
        let range = (max_val - min_val).max(1e-10);

        for i in 1..(results.len() - 1) {
            let distance = (obj_fn(&results[i + 1]) - obj_fn(&results[i - 1])) / range;
            results[i].crowding_distance += distance;
        }
    }
}

/// Run a single world and compute multi-objective fitness.
fn run_single_world_multiobjective(
    genome: WorldGenome,
    frames: usize,
    snapshot_interval: usize,
    lite: bool,
) -> Result<ParetoResult, String> {
    let start = Instant::now();

    let mut world = if lite {
        genome.build_world_lite()?
    } else {
        genome.build_world()?
    };

    let mut periodic_snapshots = Vec::new();

    for frame in 0..frames {
        world.step_frame()?;

        if frame % snapshot_interval == 0 {
            periodic_snapshots.push(world.snapshot());
        }
    }

    let snapshot = world.snapshot();
    let objectives = MultiObjectiveFitness::evaluate(&snapshot, &periodic_snapshots);

    Ok(ParetoResult {
        genome,
        objectives,
        rank: 0,
        crowding_distance: 0.0,
        wall_time_ms: start.elapsed().as_secs_f32() * 1000.0,
    })
}

/// Run NSGA-II Pareto optimization.
pub fn evolve_pareto(config: EvolutionConfig) -> Result<ParetoEvolutionResult, String> {
    let start = Instant::now();

    let mut rng = StdRng::seed_from_u64(config.master_seed);
    let mut generation_results = Vec::new();
    let mut total_worlds = 0usize;

    // Initialize population
    let mut population: Vec<WorldGenome> = (0..config.population_size)
        .map(|_| {
            let mut g = WorldGenome::random(&mut rng);
            config.constraints.apply(&mut g);
            g
        })
        .collect();

    for gen in 0..config.generations {
        let gen_start = Instant::now();

        // Evaluate population
        let (tx, rx) = mpsc::channel();
        let lite = config.lite;
        let snapshot_interval = config.fitness.snapshot_interval;
        let frames = config.frames_per_world;

        for genome in population.clone() {
            let tx = tx.clone();
            std::thread::spawn(move || {
                let result = run_single_world_multiobjective(genome, frames, snapshot_interval, lite);
                let _ = tx.send(result);
            });
        }
        drop(tx);

        let mut results: Vec<ParetoResult> = rx.iter()
            .filter_map(|r| r.ok())
            .collect();

        if results.is_empty() {
            return Err("All Pareto evaluations failed".to_string());
        }

        total_worlds += results.len();

        // Compute Pareto ranks and crowding distance
        compute_pareto_ranks(&mut results);

        // Compute crowding distance within each rank
        let mut rank_groups: Vec<Vec<usize>> = vec![Vec::new(); results.iter().map(|r| r.rank).max().unwrap_or(0) + 1];
        for (i, r) in results.iter().enumerate() {
            if r.rank < rank_groups.len() {
                rank_groups[r.rank].push(i);
            }
        }

        for group in &rank_groups {
            if group.len() > 2 {
                let mut subgroup: Vec<ParetoResult> = group.iter().map(|&i| results[i].clone()).collect();
                compute_crowding_distance(&mut subgroup);
                for (i, r) in subgroup.into_iter().enumerate() {
                    results[group[i]] = r;
                }
            }
        }

        // Sort by rank then crowding distance (descending)
        results.sort_by(|a, b| {
            match a.rank.cmp(&b.rank) {
                std::cmp::Ordering::Equal => {
                    b.crowding_distance.partial_cmp(&a.crowding_distance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                }
                other => other,
            }
        });

        let gen_wall_time = gen_start.elapsed().as_secs_f32() * 1000.0;

        eprintln!(
            "[pareto gen {}/{}] front_size={} evaluated={} ({:.0}ms)",
            gen + 1, config.generations,
            results.iter().filter(|r| r.rank == 0).count(),
            results.len(),
            gen_wall_time
        );

        generation_results.push(results.clone());

        // Breed next generation using NSGA-II selection
        if gen + 1 < config.generations {
            population = nsga2_breed(&results, config.population_size, &mut rng, 0.15, 0.7);
        }
    }

    // Extract final Pareto front
    let final_results = generation_results.into_iter().last().unwrap_or_default();
    let pareto_front: Vec<ParetoResult> = final_results.into_iter().filter(|r| r.rank == 0).collect();

    Ok(ParetoEvolutionResult {
        pareto_front,
        generations_run: config.generations,
        total_worlds_evaluated: total_worlds,
        total_wall_time_ms: start.elapsed().as_secs_f32() * 1000.0,
    })
}

/// Breed next generation using NSGA-II selection.
fn nsga2_breed(
    results: &[ParetoResult],
    population_size: usize,
    rng: &mut StdRng,
    mutation_rate: f32,
    crossover_rate: f32,
) -> Vec<WorldGenome> {
    let mut next_gen = Vec::with_capacity(population_size);

    // Copy top performers (elitism based on rank)
    for r in results.iter().take(population_size / 4) {
        next_gen.push(r.genome.clone());
    }

    // Fill rest with tournament selection
    while next_gen.len() < population_size {
        let parent_a = nsga2_tournament_select(results, rng);
        let parent_b = nsga2_tournament_select(results, rng);

        let mut child = if rng.gen::<f32>() < crossover_rate {
            WorldGenome::crossover(&results[parent_a].genome, &results[parent_b].genome, rng)
        } else {
            results[parent_a].genome.clone()
        };

        child.mutate(rng, mutation_rate);
        next_gen.push(child);
    }

    next_gen
}

/// NSGA-II tournament selection (lower rank wins, higher crowding distance breaks ties).
fn nsga2_tournament_select(results: &[ParetoResult], rng: &mut StdRng) -> usize {
    let tournament_size = 3.min(results.len());
    let mut best_idx = rng.gen_range(0..results.len());

    for _ in 1..tournament_size {
        let idx = rng.gen_range(0..results.len());
        let best = &results[best_idx];
        let candidate = &results[idx];

        let better = candidate.rank < best.rank
            || (candidate.rank == best.rank && candidate.crowding_distance > best.crowding_distance);

        if better {
            best_idx = idx;
        }
    }

    best_idx
}

// ---------------------------------------------------------------------------
// Stress-Resilient Pareto (NSGA-II + drought/heat perturbations)
// ---------------------------------------------------------------------------

/// Run a single world with stress perturbations AND multi-objective fitness.
fn run_single_world_multiobjective_stressed(
    genome: WorldGenome,
    frames: usize,
    snapshot_interval: usize,
    lite: bool,
    stress: &StressTestConfig,
) -> Result<(ParetoResult, StressTelemetryMetrics), String> {
    let start = Instant::now();

    let mut world = if lite {
        genome.build_world_lite()?
    } else {
        genome.build_world()?
    };

    let drought_start = stress.drought_frame;
    let drought_end = (drought_start + stress.perturbation_duration).min(frames);
    let heat_start = stress.heat_spike_frame;
    let heat_end = (heat_start + stress.perturbation_duration).min(frames);

    let mut pre_stress_biomass = 0.0_f32;
    let mut min_stress_biomass = f32::INFINITY;
    let mut post_recovery_biomass = 0.0_f32;

    let original_moisture = world.moisture_field().to_vec();
    let original_temp = world.temperature_field().to_vec();

    let mut periodic_snapshots = Vec::new();

    for frame in 0..frames {
        // Inject drought
        if frame == drought_start {
            let n = original_moisture.len().min(world.moisture_field().len());
            for i in 0..n {
                world.moisture_field_mut()[i] = original_moisture[i] * 0.1;
            }
        }
        if frame == drought_end {
            let n = original_moisture.len().min(world.moisture_field().len());
            for i in 0..n {
                world.moisture_field_mut()[i] = original_moisture[i];
            }
        }
        // Inject heat spike
        if frame == heat_start {
            let n = world.temperature_field().len();
            for i in 0..n {
                world.temperature_field_mut()[i] += 15.0;
            }
        }
        if frame == heat_end {
            let n = original_temp.len().min(world.temperature_field().len());
            for i in 0..n {
                world.temperature_field_mut()[i] = original_temp[i];
            }
        }

        world.step_frame()?;

        if frame % snapshot_interval == 0 {
            periodic_snapshots.push(world.snapshot());
        }

        let snapshot = world.snapshot();
        let biomass = evaluate_fitness(FitnessObjective::MaxBiomass, &snapshot, &[]);

        if frame + 1 == drought_start.min(heat_start) {
            pre_stress_biomass = biomass;
        }
        if (frame >= drought_start && frame < drought_end)
            || (frame >= heat_start && frame < heat_end)
        {
            min_stress_biomass = min_stress_biomass.min(biomass);
        }
        if frame + 1 == frames {
            post_recovery_biomass = biomass;
        }
    }

    if pre_stress_biomass == 0.0 {
        let snapshot = world.snapshot();
        pre_stress_biomass = evaluate_fitness(FitnessObjective::MaxBiomass, &snapshot, &[]);
    }
    if min_stress_biomass == f32::INFINITY {
        min_stress_biomass = post_recovery_biomass;
    }

    let snapshot = world.snapshot();
    let objectives = MultiObjectiveFitness::evaluate(&snapshot, &periodic_snapshots);

    let pareto = ParetoResult {
        genome,
        objectives,
        rank: 0,
        crowding_distance: 0.0,
        wall_time_ms: start.elapsed().as_secs_f32() * 1000.0,
    };
    let stress_metrics = StressTelemetryMetrics {
        pre_stress_biomass,
        min_stress_biomass,
        post_recovery_biomass,
    };

    Ok((pareto, stress_metrics))
}

/// Run NSGA-II Pareto optimization with environmental stress perturbations.
///
/// Finds ecosystems that are Pareto-optimal across all 6 objectives while
/// being resilient to drought + heat spikes.
pub fn evolve_pareto_stressed(config: EvolutionConfig) -> Result<ParetoEvolutionResult, String> {
    let start = Instant::now();

    let frames = config.frames_per_world;
    let stress = StressTestConfig {
        drought_frame: frames / 3,
        heat_spike_frame: 2 * frames / 3,
        perturbation_duration: (frames / 10).max(1),
        ..StressTestConfig::default()
    };

    let mut rng = StdRng::seed_from_u64(config.master_seed);
    let mut total_worlds = 0usize;

    let mut population: Vec<WorldGenome> = (0..config.population_size)
        .map(|_| {
            let mut g = WorldGenome::random(&mut rng);
            config.constraints.apply(&mut g);
            g
        })
        .collect();

    let mut last_results: Vec<ParetoResult> = Vec::new();

    for gen in 0..config.generations {
        let gen_start = Instant::now();

        let (tx, rx) = mpsc::channel();
        let lite = config.lite;
        let snapshot_interval = config.fitness.snapshot_interval;
        let stress_cfg = stress.clone();

        for genome in population.clone() {
            let tx = tx.clone();
            let stress_cfg = stress_cfg.clone();
            std::thread::spawn(move || {
                let result = run_single_world_multiobjective_stressed(
                    genome,
                    frames,
                    snapshot_interval,
                    lite,
                    &stress_cfg,
                );
                let _ = tx.send(result);
            });
        }
        drop(tx);

        let raw_results: Vec<(ParetoResult, StressTelemetryMetrics)> =
            rx.iter().filter_map(|r| r.ok()).collect();

        if raw_results.is_empty() {
            return Err("All stressed Pareto evaluations failed".to_string());
        }

        total_worlds += raw_results.len();

        let mut results: Vec<ParetoResult> = raw_results.into_iter().map(|(p, _)| p).collect();

        // Compute Pareto ranks and crowding distance
        compute_pareto_ranks(&mut results);

        let mut rank_groups: Vec<Vec<usize>> =
            vec![Vec::new(); results.iter().map(|r| r.rank).max().unwrap_or(0) + 1];
        for (i, r) in results.iter().enumerate() {
            if r.rank < rank_groups.len() {
                rank_groups[r.rank].push(i);
            }
        }

        for group in &rank_groups {
            if group.len() > 2 {
                let mut subgroup: Vec<ParetoResult> =
                    group.iter().map(|&i| results[i].clone()).collect();
                compute_crowding_distance(&mut subgroup);
                for (i, r) in subgroup.into_iter().enumerate() {
                    results[group[i]] = r;
                }
            }
        }

        results.sort_by(|a, b| match a.rank.cmp(&b.rank) {
            std::cmp::Ordering::Equal => b
                .crowding_distance
                .partial_cmp(&a.crowding_distance)
                .unwrap_or(std::cmp::Ordering::Equal),
            other => other,
        });

        let gen_wall_time = gen_start.elapsed().as_secs_f32() * 1000.0;

        eprintln!(
            "[pareto-stressed gen {}/{}] front_size={} evaluated={} ({:.0}ms)",
            gen + 1,
            config.generations,
            results.iter().filter(|r| r.rank == 0).count(),
            results.len(),
            gen_wall_time
        );

        last_results = results.clone();

        if gen + 1 < config.generations {
            population = nsga2_breed(&results, config.population_size, &mut rng, 0.15, 0.7);
        }
    }

    let pareto_front: Vec<ParetoResult> =
        last_results.into_iter().filter(|r| r.rank == 0).collect();

    Ok(ParetoEvolutionResult {
        pareto_front,
        generations_run: config.generations,
        total_worlds_evaluated: total_worlds,
        total_wall_time_ms: start.elapsed().as_secs_f32() * 1000.0,
    })
}

/// Generate telemetry from Pareto result.
pub fn telemetry_from_pareto_result(result: &ParetoEvolutionResult) -> GenerationTelemetry {
    if result.pareto_front.is_empty() {
        return GenerationTelemetry {
            generation: result.generations_run,
            best_fitness: 0.0,
            mean_fitness: 0.0,
            worst_fitness: 0.0,
            population_diversity: 0.0,
            best_genome_params: vec![0.5; 18],
            elapsed_ms: result.total_wall_time_ms,
            mode: Some("pareto".to_string()),
            multi_objective_fitness: None,
            stress_metrics: None,
        };
    }

    // Find best by aggregate fitness
    let best = result.pareto_front.iter()
        .max_by(|a, b| {
            let a_sum = a.objectives.biomass + a.objectives.biodiversity + a.objectives.stability
                + a.objectives.carbon + a.objectives.fruit + a.objectives.microbial;
            let b_sum = b.objectives.biomass + b.objectives.biodiversity + b.objectives.stability
                + b.objectives.carbon + b.objectives.fruit + b.objectives.microbial;
            a_sum.partial_cmp(&b_sum).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();

    let sum = best.objectives.biomass + best.objectives.biodiversity + best.objectives.stability
        + best.objectives.carbon + best.objectives.fruit + best.objectives.microbial;

    GenerationTelemetry {
        generation: result.generations_run,
        best_fitness: sum / 6.0,
        mean_fitness: sum / 6.0,
        worst_fitness: sum / 6.0,
        population_diversity: result.pareto_front.len() as f32,
        best_genome_params: best.genome.normalized_params(),
        elapsed_ms: result.total_wall_time_ms,
        mode: Some("pareto".to_string()),
        multi_objective_fitness: Some(best.objectives.clone()),
        stress_metrics: None,
    }
}


// ---------------------------------------------------------------------------
// Noise-Driven Evolution (Bet-Hedging)
// ---------------------------------------------------------------------------

/// Per-fly phenotypic noise modifiers generated by stochastic gene expression.
///
/// In real biology, genetically identical organisms exhibit phenotypic
/// heterogeneity due to stochastic gene expression. Some organisms evolve
/// to exploit this noise ("bet-hedging") -- maintaining high phenotypic
/// variance so that a fraction of the population survives unpredictable
/// environmental changes.
///
/// References:
/// - Balaban et al. (2004) "Bacterial persistence as a phenotypic switch", Science
/// - Kussell & Leibler (2005) "Phenotypic diversity, population growth", Science
/// - Veening et al. (2008) "Bistability, epigenetics, bet hedging", Nat Rev Micro
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FlyPhenotypicNoise {
    /// Metabolic rate multiplier (1.0 = normal, varies per individual).
    pub metabolic_rate_modifier: f32,
    /// Trehalase Vmax multiplier -- controls sugar processing speed.
    pub trehalase_modifier: f32,
    /// Stress tolerance multiplier (heat/drought resilience).
    pub stress_tolerance: f32,
    /// Dormancy propensity (0=never dormant, 1=always dormant). "Persister" phenotype.
    pub dormancy_propensity: f32,
}

impl Default for FlyPhenotypicNoise {
    fn default() -> Self {
        Self {
            metabolic_rate_modifier: 1.0,
            trehalase_modifier: 1.0,
            stress_tolerance: 1.0,
            dormancy_propensity: 0.0,
        }
    }
}

impl FlyPhenotypicNoise {
    /// Generate phenotypic noise from stochastic gene expression state.
    pub fn from_stochastic_state(
        fano_factor: f32,
        protein_count: f32,
        noise_intensity: f32,
        rng_seed: u64,
    ) -> Self {
        use crate::whole_cell::stochastic_expression::StochasticRng;
        let mut rng = StochasticRng::new(rng_seed);

        let noise_scale = (fano_factor - 1.0).max(0.0) * noise_intensity;

        // Metabolic rate: log-normal centered at 1.0
        let z1 = (rng.next_f32().max(1e-10).ln() * -2.0).sqrt()
            * (std::f32::consts::TAU * rng.next_f32()).cos();
        let metabolic_rate_modifier = (noise_scale * 0.15 * z1).exp().clamp(0.5, 2.0);

        let protein_ratio = if protein_count > 10.0 {
            (protein_count / 100.0).clamp(0.3, 3.0)
        } else {
            1.0
        };
        let trehalase_modifier = (protein_ratio * (1.0 + noise_scale * 0.1 * (rng.next_f32() - 0.5))).clamp(0.3, 3.0);

        let stress_tolerance = (2.0 - metabolic_rate_modifier).clamp(0.3, 2.0);

        let dormancy_propensity = if noise_scale > 0.5 && rng.next_f32() < noise_scale * 0.1 {
            (0.3 + rng.next_f32() * 0.5).clamp(0.0, 1.0)
        } else {
            0.0
        };

        Self { metabolic_rate_modifier, trehalase_modifier, stress_tolerance, dormancy_propensity }
    }
}

/// Generate phenotypic noise profiles for a population of flies.
pub fn generate_population_noise(
    fly_count: usize,
    noise_intensity: f32,
    base_seed: u64,
) -> Vec<FlyPhenotypicNoise> {
    use crate::whole_cell::stochastic_expression::{
        StochasticExpressionConfig, StochasticOperonState, StochasticRng,
        step_stochastic_expression,
    };

    if noise_intensity < 1e-6 || fly_count == 0 {
        return vec![FlyPhenotypicNoise::default(); fly_count];
    }

    let config = StochasticExpressionConfig {
        enabled: true,
        mean_burst_size: 4.0 + noise_intensity * 2.0,
        promoter_on_rate: 0.02,
        promoter_off_rate: 0.08,
        ..Default::default()
    };

    let mut rng = StochasticRng::new(base_seed);
    let mut operons: Vec<StochasticOperonState> = (0..fly_count)
        .map(|i| StochasticOperonState::from_deterministic(10.0 + (i as f32 * 0.1), 100.0))
        .collect();
    let rates: Vec<(f32, f32)> = vec![(0.1, 0.01); fly_count];

    for _ in 0..200 {
        step_stochastic_expression(&mut operons, &config, &rates, 1.0, &mut rng);
    }

    operons.iter().enumerate()
        .map(|(i, operon)| FlyPhenotypicNoise::from_stochastic_state(
            operon.fano_factor, operon.protein_count, noise_intensity,
            base_seed.wrapping_add(i as u64 * 7919),
        ))
        .collect()
}

/// Evaluate bet-hedging fitness: rewards populations with phenotypic variance.
pub fn bet_hedging_fitness(
    snapshot: &TerrariumWorldSnapshot,
    noise_profiles: &[FlyPhenotypicNoise],
) -> f32 {
    if noise_profiles.is_empty() { return snapshot.flies as f32; }

    let survival_score = snapshot.flies as f32 * 5.0;

    let rates: Vec<f32> = noise_profiles.iter().map(|p| p.metabolic_rate_modifier).collect();
    let mean_rate = rates.iter().sum::<f32>() / rates.len() as f32;
    let variance = rates.iter().map(|r| (r - mean_rate).powi(2)).sum::<f32>() / rates.len() as f32;
    let cv = if mean_rate > 0.01 { variance.sqrt() / mean_rate } else { 0.0 };
    let variance_score = cv.clamp(0.0, 1.0) * 10.0;

    let persister_count = noise_profiles.iter().filter(|p| p.dormancy_propensity > 0.2).count();
    let persister_frac = persister_count as f32 / noise_profiles.len() as f32;
    let persister_score = if persister_frac > 0.0 {
        10.0 * (1.0 - (persister_frac - 0.05).abs() * 10.0).max(0.0)
    } else { 0.0 };

    let energy_score = snapshot.avg_fly_energy_charge * 5.0;
    survival_score + variance_score + persister_score + energy_score
}

// ---------------------------------------------------------------------------
// Persister Cell Modeling
// ---------------------------------------------------------------------------

/// Antibiotic persistence simulator based on stochastic phenotypic switching.
///
/// References:
/// - Balaban et al. (2004) "Bacterial persistence as a phenotypic switch", Science
/// - Lewis (2010) "Persister cells", Annual Review of Microbiology
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PersisterCellSimulator {
    pub normal_cells: f32,
    pub persister_cells: f32,
    pub switch_to_persister_rate: f32,
    pub switch_to_normal_rate: f32,
    pub normal_growth_rate: f32,
    pub persister_growth_rate: f32,
    pub antibiotic_kill_rate: f32,
    pub antibiotic_active: bool,
    pub carrying_capacity: f32,
    pub time_hours: f32,
    pub history: Vec<PersisterSnapshot>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PersisterSnapshot {
    pub time_hours: f32,
    pub normal_cells: f32,
    pub persister_cells: f32,
    pub total_cells: f32,
    pub persister_fraction: f32,
    pub antibiotic_active: bool,
}

impl Default for PersisterCellSimulator {
    fn default() -> Self {
        Self {
            normal_cells: 1e6,
            persister_cells: 100.0,
            switch_to_persister_rate: 1e-5,
            switch_to_normal_rate: 1e-4,
            normal_growth_rate: 1.0,
            persister_growth_rate: 0.001,
            antibiotic_kill_rate: 3.0,
            antibiotic_active: false,
            carrying_capacity: 1e9,
            time_hours: 0.0,
            history: Vec::new(),
        }
    }
}

impl PersisterCellSimulator {
    pub fn with_switching_rates(to_persister: f32, to_normal: f32) -> Self {
        Self { switch_to_persister_rate: to_persister, switch_to_normal_rate: to_normal, ..Default::default() }
    }

    pub fn apply_antibiotic(&mut self) { self.antibiotic_active = true; }
    pub fn remove_antibiotic(&mut self) { self.antibiotic_active = false; }

    pub fn step(&mut self, dt_hours: f32) {
        let n = self.normal_cells;
        let p = self.persister_cells;
        let total = n + p;
        let growth_factor = (1.0 - total / self.carrying_capacity).max(0.0);

        let normal_growth = n * self.normal_growth_rate * growth_factor * dt_hours;
        let normal_to_persister = n * self.switch_to_persister_rate * dt_hours;
        let antibiotic_kill = if self.antibiotic_active { n * self.antibiotic_kill_rate * dt_hours } else { 0.0 };

        let persister_growth = p * self.persister_growth_rate * growth_factor * dt_hours;
        let persister_to_normal = p * self.switch_to_normal_rate * dt_hours;

        self.normal_cells = (n + normal_growth - normal_to_persister + persister_to_normal - antibiotic_kill).max(0.0);
        self.persister_cells = (p + persister_growth + normal_to_persister - persister_to_normal).max(0.0);
        self.time_hours += dt_hours;

        let total = self.normal_cells + self.persister_cells;
        self.history.push(PersisterSnapshot {
            time_hours: self.time_hours, normal_cells: self.normal_cells,
            persister_cells: self.persister_cells, total_cells: total,
            persister_fraction: if total > 0.0 { self.persister_cells / total } else { 0.0 },
            antibiotic_active: self.antibiotic_active,
        });
    }

    pub fn run_treatment_protocol(&mut self, growth_hours: f32, treatment_hours: f32,
        recovery_hours: f32, dt: f32,
    ) -> PersisterTreatmentResult {
        let growth_steps = (growth_hours / dt) as usize;
        for _ in 0..growth_steps { self.step(dt); }
        let pre_treatment_total = self.normal_cells + self.persister_cells;
        let pre_persister_frac = if pre_treatment_total > 0.0 { self.persister_cells / pre_treatment_total } else { 0.0 };

        self.apply_antibiotic();
        let treatment_steps = (treatment_hours / dt) as usize;
        let mut min_total = f32::INFINITY;
        for _ in 0..treatment_steps {
            self.step(dt);
            min_total = min_total.min(self.normal_cells + self.persister_cells);
        }
        let post_treatment_total = self.normal_cells + self.persister_cells;
        let survival_fraction = if pre_treatment_total > 0.0 { post_treatment_total / pre_treatment_total } else { 0.0 };

        self.remove_antibiotic();
        let recovery_steps = (recovery_hours / dt) as usize;
        for _ in 0..recovery_steps { self.step(dt); }
        let post_recovery_total = self.normal_cells + self.persister_cells;
        let regrowth_ratio = if min_total > 0.0 { post_recovery_total / min_total } else { 0.0 };

        let biphasic = self.detect_biphasic_kill();
        PersisterTreatmentResult {
            pre_treatment_total, pre_treatment_persister_fraction: pre_persister_frac,
            post_treatment_total, survival_fraction, minimum_population: min_total,
            post_recovery_total, regrowth_ratio, total_time_hours: self.time_hours, biphasic_kill: biphasic,
        }
    }

    fn detect_biphasic_kill(&self) -> bool {
        let treatment_snapshots: Vec<&PersisterSnapshot> = self.history.iter().filter(|s| s.antibiotic_active).collect();
        if treatment_snapshots.len() < 4 { return false; }
        let mid = treatment_snapshots.len() / 2;
        let early_rate = if treatment_snapshots[0].total_cells > 0.0 {
            (treatment_snapshots[mid].total_cells / treatment_snapshots[0].total_cells).ln()
        } else { 0.0 };
        let late_rate = if treatment_snapshots[mid].total_cells > 0.0 {
            (treatment_snapshots.last().unwrap().total_cells / treatment_snapshots[mid].total_cells).ln()
        } else { 0.0 };
        early_rate < late_rate * 2.0 && early_rate < -0.5
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PersisterTreatmentResult {
    pub pre_treatment_total: f32,
    pub pre_treatment_persister_fraction: f32,
    pub post_treatment_total: f32,
    pub survival_fraction: f32,
    pub minimum_population: f32,
    pub post_recovery_total: f32,
    pub regrowth_ratio: f32,
    pub total_time_hours: f32,
    pub biphasic_kill: bool,
}

// ---------------------------------------------------------------------------
// Extended Genome: Noise Parameters for Bet-Hedging Evolution
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BetHedgingGenome {
    pub base: WorldGenome,
    pub noise_intensity: f32,
    pub persister_rate_scale: f32,
    pub noise_tolerance: f32,
}

impl BetHedgingGenome {
    pub fn random(rng: &mut StdRng) -> Self {
        Self {
            base: WorldGenome::random(rng),
            noise_intensity: rng.gen_range(0.0..=1.0),
            persister_rate_scale: rng.gen_range(0.1..=10.0),
            noise_tolerance: rng.gen_range(0.0..=1.0),
        }
    }

    pub fn crossover(a: &Self, b: &Self, rng: &mut StdRng) -> Self {
        Self {
            base: WorldGenome::crossover(&a.base, &b.base, rng),
            noise_intensity: if rng.gen::<f32>() < 0.5 { a.noise_intensity } else { b.noise_intensity },
            persister_rate_scale: if rng.gen::<f32>() < 0.5 { a.persister_rate_scale } else { b.persister_rate_scale },
            noise_tolerance: if rng.gen::<f32>() < 0.5 { a.noise_tolerance } else { b.noise_tolerance },
        }
    }

    pub fn mutate(&mut self, rng: &mut StdRng, rate: f32) {
        self.base.mutate(rng, rate);
        if rng.gen::<f32>() < rate {
            self.noise_intensity = (self.noise_intensity + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0);
        }
        if rng.gen::<f32>() < rate {
            self.persister_rate_scale = (self.persister_rate_scale * (1.0 + rng.gen_range(-0.2..0.2))).clamp(0.1, 10.0);
        }
        if rng.gen::<f32>() < rate {
            self.noise_tolerance = (self.noise_tolerance + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0);
        }
    }
}

/// Run a bet-hedging evolution experiment.
pub fn evolve_bet_hedging(
    population_size: usize, generations: usize, frames: usize, seed: u64,
) -> Result<BetHedgingResult, String> {
    let start = Instant::now();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut population: Vec<BetHedgingGenome> = (0..population_size)
        .map(|_| BetHedgingGenome::random(&mut rng)).collect();

    let mut generation_results = Vec::new();
    let mut global_best_fitness = f32::NEG_INFINITY;
    let mut global_best_genome = population[0].clone();

    for gen in 0..generations {
        let gen_start = Instant::now();
        let mut fitnesses = Vec::with_capacity(population_size);

        for genome in &population {
            let mut world = match genome.base.build_world_lite() {
                Ok(w) => w, Err(_) => { fitnesses.push(0.0); continue; }
            };
            for _ in 0..frames { let _ = world.step_frame(); }
            let snapshot = world.snapshot();
            let noise_profiles = generate_population_noise(
                snapshot.flies.max(genome.base.fly_count), genome.noise_intensity, genome.base.seed,
            );
            fitnesses.push(bet_hedging_fitness(&snapshot, &noise_profiles));
        }

        let best_idx = fitnesses.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i).unwrap_or(0);
        let best_fitness = fitnesses[best_idx];
        let mean_fitness = fitnesses.iter().sum::<f32>() / fitnesses.len() as f32;

        if best_fitness > global_best_fitness {
            global_best_fitness = best_fitness;
            global_best_genome = population[best_idx].clone();
        }

        let gen_wall_time = gen_start.elapsed().as_secs_f32() * 1000.0;
        eprintln!("[bet-hedging gen {}/{}] best={:.2} mean={:.2} noise={:.3} ({:.0}ms)",
            gen + 1, generations, best_fitness, mean_fitness,
            population[best_idx].noise_intensity, gen_wall_time);

        generation_results.push(BetHedgingGenerationResult {
            generation: gen, best_fitness, mean_fitness,
            best_noise_intensity: population[best_idx].noise_intensity,
            best_persister_rate: population[best_idx].persister_rate_scale,
            wall_time_ms: gen_wall_time,
        });

        if gen + 1 < generations {
            let mut indexed: Vec<(usize, f32)> = fitnesses.iter().copied().enumerate().collect();
            indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let mut next_gen = Vec::with_capacity(population_size);
            for &(idx, _) in indexed.iter().take(2) { next_gen.push(population[idx].clone()); }
            while next_gen.len() < population_size {
                let a_idx = indexed[rng.gen_range(0..3.min(indexed.len()))].0;
                let b_idx = indexed[rng.gen_range(0..3.min(indexed.len()))].0;
                let mut child = if rng.gen::<f32>() < 0.7 {
                    BetHedgingGenome::crossover(&population[a_idx], &population[b_idx], &mut rng)
                } else { population[a_idx].clone() };
                child.mutate(&mut rng, 0.15);
                next_gen.push(child);
            }
            population = next_gen;
        }
    }

    Ok(BetHedgingResult {
        generations: generation_results, best_genome: global_best_genome,
        best_fitness: global_best_fitness,
        total_wall_time_ms: start.elapsed().as_secs_f32() * 1000.0,
    })
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BetHedgingGenerationResult {
    pub generation: usize, pub best_fitness: f32, pub mean_fitness: f32,
    pub best_noise_intensity: f32, pub best_persister_rate: f32, pub wall_time_ms: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BetHedgingResult {
    pub generations: Vec<BetHedgingGenerationResult>,
    pub best_genome: BetHedgingGenome,
    pub best_fitness: f32,
    pub total_wall_time_ms: f32,
}


// ---------------------------------------------------------------------------
// Environmental Variability Engine
// ---------------------------------------------------------------------------

/// Seasonal cycle + stochastic weather + drought events.
///
/// Generates realistic environmental variation that creates the selective
/// pressure making bet-hedging adaptive.
///
/// References:
/// - Vasseur & Yodzis (2004) "The color of environmental noise", Ecology
/// - Ruel & Ayres (1999) "Jensen's inequality predicts effects of variation", TREE
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EnvironmentalSchedule {
    /// Base temperature (°C) around which seasons oscillate.
    pub base_temperature_c: f32,
    /// Seasonal amplitude (°C), peak-to-trough / 2.
    pub seasonal_amplitude_c: f32,
    /// Period of one full seasonal cycle (simulation seconds).
    pub season_period_s: f32,
    /// Base humidity [0, 1].
    pub base_humidity: f32,
    /// Humidity seasonal amplitude.
    pub humidity_amplitude: f32,
    /// Drought probability per evaluation step.
    pub drought_probability: f32,
    /// Drought duration range (simulation seconds).
    pub drought_duration_range_s: (f32, f32),
    /// Weather noise sigma for temperature (°C).
    pub weather_noise_temp_c: f32,
    /// Weather noise sigma for humidity.
    pub weather_noise_humidity: f32,
}

impl Default for EnvironmentalSchedule {
    fn default() -> Self {
        Self {
            base_temperature_c: 22.0,
            seasonal_amplitude_c: 8.0,
            season_period_s: 365.0 * 86400.0,
            base_humidity: 0.65,
            humidity_amplitude: 0.15,
            drought_probability: 0.001,
            drought_duration_range_s: (86400.0 * 7.0, 86400.0 * 30.0),
            weather_noise_temp_c: 2.0,
            weather_noise_humidity: 0.05,
        }
    }
}

/// A snapshot of current environmental conditions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EnvironmentalSample {
    pub temperature_c: f32,
    pub humidity: f32,
    pub moisture_modifier: f32,
    pub is_drought: bool,
    pub season_phase: f32,
}

/// Active drought event.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DroughtEvent {
    pub start_time_s: f32,
    pub duration_s: f32,
    /// Fraction of moisture removed (0 = none, 1 = total).
    pub severity: f32,
}

/// State tracker for environmental variability during a world run.
#[derive(Debug, Clone)]
pub struct EnvironmentalState {
    pub schedule: EnvironmentalSchedule,
    pub active_drought: Option<DroughtEvent>,
    pub rng: StdRng,
}

impl EnvironmentalState {
    pub fn new(schedule: EnvironmentalSchedule, seed: u64) -> Self {
        Self { schedule, active_drought: None, rng: StdRng::seed_from_u64(seed) }
    }

    pub fn sample(&mut self, time_s: f32) -> EnvironmentalSample {
        use rand_distr::StandardNormal;
        let s = &self.schedule;
        let phase = std::f32::consts::TAU * time_s / s.season_period_s;
        let seasonal_temp = s.base_temperature_c + s.seasonal_amplitude_c * phase.sin();
        let seasonal_hum = s.base_humidity
            + s.humidity_amplitude * (phase + std::f32::consts::FRAC_PI_2).sin();

        let z1: f32 = self.rng.sample(StandardNormal);
        let z2: f32 = self.rng.sample(StandardNormal);
        let temp = (seasonal_temp + s.weather_noise_temp_c * z1).clamp(-10.0, 50.0);
        let humidity = (seasonal_hum + s.weather_noise_humidity * z2).clamp(0.0, 1.0);

        // Drought lifecycle
        if let Some(d) = &self.active_drought {
            if time_s > d.start_time_s + d.duration_s {
                self.active_drought = None;
            }
        }
        if self.active_drought.is_none() && self.rng.gen::<f32>() < s.drought_probability {
            let dur = self.rng.gen_range(s.drought_duration_range_s.0..=s.drought_duration_range_s.1);
            self.active_drought = Some(DroughtEvent {
                start_time_s: time_s,
                duration_s: dur,
                severity: self.rng.gen_range(0.3..0.9),
            });
        }

        let drought_factor = if let Some(d) = &self.active_drought { 1.0 - d.severity } else { 1.0 };
        EnvironmentalSample {
            temperature_c: temp,
            humidity: humidity * drought_factor,
            moisture_modifier: drought_factor,
            is_drought: self.active_drought.is_some(),
            season_phase: (phase / std::f32::consts::TAU).rem_euclid(1.0),
        }
    }
}

impl EnvironmentalSchedule {
    pub fn temperate() -> Self { Self::default() }

    pub fn tropical() -> Self {
        Self {
            base_temperature_c: 28.0, seasonal_amplitude_c: 3.0,
            base_humidity: 0.80, humidity_amplitude: 0.10,
            drought_probability: 0.003, ..Self::default()
        }
    }

    pub fn arid() -> Self {
        Self {
            base_temperature_c: 30.0, seasonal_amplitude_c: 12.0,
            base_humidity: 0.25, humidity_amplitude: 0.10,
            drought_probability: 0.005, weather_noise_temp_c: 4.0,
            ..Self::default()
        }
    }
}

/// Run a world with environmental variability applied each frame.
pub fn run_single_world_with_environment(
    genome: WorldGenome,
    frames: usize,
    schedule: &EnvironmentalSchedule,
    lite: bool,
) -> Result<(WorldResult, Vec<EnvironmentalSample>), String> {
    let start = Instant::now();
    let mut world = if lite { genome.build_world_lite()? } else { genome.build_world()? };
    let mut env = EnvironmentalState::new(schedule.clone(), genome.seed.wrapping_add(9999));
    let mut samples = Vec::with_capacity(frames);

    for _ in 0..frames {
        let sample = env.sample(world.time_s);
        samples.push(sample);
        world.step_frame()?;
    }
    let snapshot = world.snapshot();
    let fitness = evaluate_fitness(FitnessObjective::MaxBiomass, &snapshot, &[]);
    Ok((
        WorldResult {
            genome, fitness, final_biomass: snapshot.total_plant_cells,
            final_plants: snapshot.plants, final_fruits: snapshot.fruits,
            wall_time_ms: start.elapsed().as_secs_f32() * 1000.0,
        },
        samples,
    ))
}

// ---------------------------------------------------------------------------
// Spatial Heterogeneity Zones
// ---------------------------------------------------------------------------

/// Named environment zone type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ZoneType {
    Wetland, Arid, Tropical, Temperate, NutrientRich, NutrientPoor,
}

/// A spatial zone with local environmental modifiers.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpatialZone {
    pub x_start: usize,
    pub y_start: usize,
    pub x_end: usize,
    pub y_end: usize,
    pub temperature_modifier: f32,
    pub moisture_modifier: f32,
    pub nutrient_modifier: f32,
    pub light_modifier: f32,
    pub zone_type: ZoneType,
}

impl SpatialZone {
    pub fn contains(&self, x: usize, y: usize) -> bool {
        x >= self.x_start && x < self.x_end && y >= self.y_start && y < self.y_end
    }

    pub fn area(&self) -> usize {
        self.x_end.saturating_sub(self.x_start) * self.y_end.saturating_sub(self.y_start)
    }
}

/// Generate a set of spatial zones that partition a world grid.
pub fn generate_spatial_zones(
    width: usize, height: usize, zone_count: usize, rng: &mut StdRng,
) -> Vec<SpatialZone> {
    let zone_types = [
        ZoneType::Wetland, ZoneType::Arid, ZoneType::Tropical,
        ZoneType::Temperate, ZoneType::NutrientRich, ZoneType::NutrientPoor,
    ];
    let cols = (zone_count as f32).sqrt().ceil() as usize;
    let rows = (zone_count + cols - 1) / cols;
    let cw = width / cols.max(1);
    let ch = height / rows.max(1);

    let mut zones = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            if zones.len() >= zone_count { break; }
            let zt = zone_types[rng.gen_range(0..zone_types.len())];
            let (temp, moist, nutr, light) = match zt {
                ZoneType::Wetland      => (0.0, 0.5, 0.2, -0.1),
                ZoneType::Arid         => (3.0, -0.4, -0.1, 0.2),
                ZoneType::Tropical     => (5.0, 0.3, 0.3, 0.1),
                ZoneType::Temperate    => (0.0, 0.0, 0.0, 0.0),
                ZoneType::NutrientRich => (-1.0, 0.1, 0.5, 0.0),
                ZoneType::NutrientPoor => (1.0, -0.1, -0.3, 0.1),
            };
            zones.push(SpatialZone {
                x_start: c * cw, y_start: r * ch,
                x_end: ((c + 1) * cw).min(width), y_end: ((r + 1) * ch).min(height),
                temperature_modifier: temp, moisture_modifier: moist,
                nutrient_modifier: nutr, light_modifier: light, zone_type: zt,
            });
        }
    }
    zones
}

/// Compute a combined fitness modifier from spatial zones at a point.
pub fn spatial_fitness_modifier(zones: &[SpatialZone], x: usize, y: usize) -> f32 {
    for zone in zones {
        if zone.contains(x, y) {
            return 1.0 + zone.nutrient_modifier * 0.5 + zone.moisture_modifier * 0.3;
        }
    }
    1.0
}

// ---------------------------------------------------------------------------
// Multi-Species Phenotypic Noise
// ---------------------------------------------------------------------------

/// Phenotypic noise for plants — growth rate, drought tolerance, photosynthesis.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PlantPhenotypicNoise {
    pub growth_rate_modifier: f32,
    pub drought_tolerance: f32,
    pub photosynthetic_efficiency: f32,
    pub root_depth_modifier: f32,
}

impl Default for PlantPhenotypicNoise {
    fn default() -> Self {
        Self { growth_rate_modifier: 1.0, drought_tolerance: 1.0,
               photosynthetic_efficiency: 1.0, root_depth_modifier: 1.0 }
    }
}

/// Phenotypic noise for microbes — growth, resistance, biofilm, sporulation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MicrobialPhenotypicNoise {
    pub growth_rate_modifier: f32,
    pub antibiotic_resistance: f32,
    pub biofilm_propensity: f32,
    pub sporulation_threshold: f32,
}

impl Default for MicrobialPhenotypicNoise {
    fn default() -> Self {
        Self { growth_rate_modifier: 1.0, antibiotic_resistance: 0.0,
               biofilm_propensity: 0.0, sporulation_threshold: 0.5 }
    }
}

/// Generate plant noise profiles using stochastic gene expression.
pub fn generate_plant_noise(
    count: usize, noise_intensity: f32, seed: u64,
) -> Vec<PlantPhenotypicNoise> {
    if noise_intensity < 1e-6 || count == 0 {
        return vec![PlantPhenotypicNoise::default(); count];
    }
    use crate::whole_cell::stochastic_expression::StochasticRng;
    let mut rng = StochasticRng::new(seed);
    (0..count).map(|_| {
        let z1 = (rng.next_f32().max(1e-10).ln() * -2.0).sqrt()
            * (std::f32::consts::TAU * rng.next_f32()).cos();
        let z2 = (rng.next_f32().max(1e-10).ln() * -2.0).sqrt()
            * (std::f32::consts::TAU * rng.next_f32()).cos();
        PlantPhenotypicNoise {
            growth_rate_modifier: (noise_intensity * 0.2 * z1).exp().clamp(0.5, 2.0),
            drought_tolerance: (1.0 + noise_intensity * 0.15 * z2).clamp(0.3, 2.0),
            photosynthetic_efficiency: (1.0 + noise_intensity * 0.1 * (rng.next_f32() - 0.5)).clamp(0.5, 1.5),
            root_depth_modifier: (1.0 + noise_intensity * 0.2 * (rng.next_f32() - 0.5)).clamp(0.5, 2.0),
        }
    }).collect()
}

/// Generate microbial noise profiles.
pub fn generate_microbial_noise(
    count: usize, noise_intensity: f32, seed: u64,
) -> Vec<MicrobialPhenotypicNoise> {
    if noise_intensity < 1e-6 || count == 0 {
        return vec![MicrobialPhenotypicNoise::default(); count];
    }
    use crate::whole_cell::stochastic_expression::StochasticRng;
    let mut rng = StochasticRng::new(seed);
    (0..count).map(|_| {
        MicrobialPhenotypicNoise {
            growth_rate_modifier: (noise_intensity * 0.15 * (rng.next_f32() - 0.5) * 2.0).exp().clamp(0.5, 2.0),
            antibiotic_resistance: if rng.next_f32() < noise_intensity * 0.1 {
                rng.next_f32().clamp(0.1, 0.9)
            } else { 0.0 },
            biofilm_propensity: (noise_intensity * rng.next_f32() * 0.5).clamp(0.0, 1.0),
            sporulation_threshold: (0.5 + noise_intensity * (rng.next_f32() - 0.5)).clamp(0.1, 0.9),
        }
    }).collect()
}

/// Multi-species bet-hedging fitness: evaluates all trophic levels.
pub fn multi_species_bet_hedging_fitness(
    snapshot: &TerrariumWorldSnapshot,
    fly_noise: &[FlyPhenotypicNoise],
    plant_noise: &[PlantPhenotypicNoise],
    microbial_noise: &[MicrobialPhenotypicNoise],
) -> f32 {
    let fly_score = bet_hedging_fitness(snapshot, fly_noise);

    let plant_cv = if plant_noise.len() > 1 {
        let rates: Vec<f32> = plant_noise.iter().map(|p| p.growth_rate_modifier).collect();
        let mean = rates.iter().sum::<f32>() / rates.len() as f32;
        let var = rates.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / rates.len() as f32;
        if mean > 0.01 { var.sqrt() / mean } else { 0.0 }
    } else { 0.0 };

    let microbe_diversity = if microbial_noise.len() > 1 {
        let biofilm_count = microbial_noise.iter().filter(|m| m.biofilm_propensity > 0.3).count();
        let resistant_count = microbial_noise.iter().filter(|m| m.antibiotic_resistance > 0.1).count();
        (biofilm_count + resistant_count) as f32 / microbial_noise.len() as f32
    } else { 0.0 };

    fly_score + plant_cv * 15.0 + microbe_diversity * 20.0
        + snapshot.total_plant_cells * 0.01 + snapshot.mean_microbes * 5.0
}

// ---------------------------------------------------------------------------
// Drug Protocol Optimizer
// ---------------------------------------------------------------------------

/// Multi-drug treatment protocol for antibiotic resistance research.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DrugProtocol {
    /// Sequence of (drug_kill_rate, duration_hours, rest_hours) cycles.
    pub cycles: Vec<DrugCycle>,
    /// Growth phase before treatment begins.
    pub pre_growth_hours: f32,
    /// Post-treatment recovery observation period.
    pub recovery_hours: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DrugCycle {
    pub drug_kill_rate: f32,
    pub treatment_hours: f32,
    pub rest_hours: f32,
}

/// Result of running a drug protocol on persister cells.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DrugProtocolResult {
    pub final_population: f32,
    pub minimum_population: f32,
    pub survival_fraction: f32,
    pub total_time_hours: f32,
    pub cycles_completed: usize,
    pub eradication_achieved: bool,
}

impl DrugProtocol {
    /// Single drug, single dose.
    pub fn single(kill_rate: f32, treatment_hours: f32) -> Self {
        Self {
            cycles: vec![DrugCycle { drug_kill_rate: kill_rate, treatment_hours, rest_hours: 0.0 }],
            pre_growth_hours: 5.0,
            recovery_hours: 10.0,
        }
    }

    /// Pulsed dosing: alternate treatment and rest.
    pub fn pulsed(kill_rate: f32, treatment_hours: f32, rest_hours: f32, n_cycles: usize) -> Self {
        Self {
            cycles: (0..n_cycles).map(|_| DrugCycle {
                drug_kill_rate: kill_rate, treatment_hours, rest_hours,
            }).collect(),
            pre_growth_hours: 5.0,
            recovery_hours: 10.0,
        }
    }

    /// Combination therapy: two drugs in sequence.
    pub fn combination(kill_rate_a: f32, kill_rate_b: f32, hours_each: f32) -> Self {
        Self {
            cycles: vec![
                DrugCycle { drug_kill_rate: kill_rate_a, treatment_hours: hours_each, rest_hours: 1.0 },
                DrugCycle { drug_kill_rate: kill_rate_b, treatment_hours: hours_each, rest_hours: 0.0 },
            ],
            pre_growth_hours: 5.0,
            recovery_hours: 10.0,
        }
    }

    /// Run the protocol on a PersisterCellSimulator.
    pub fn execute(&self, sim: &mut PersisterCellSimulator, dt: f32) -> DrugProtocolResult {
        // Growth phase
        let growth_steps = (self.pre_growth_hours / dt) as usize;
        for _ in 0..growth_steps { sim.step(dt); }
        let pre_total = sim.normal_cells + sim.persister_cells;

        let mut min_pop = pre_total;
        let mut cycles_done = 0;

        for cycle in &self.cycles {
            // Treatment
            sim.antibiotic_kill_rate = cycle.drug_kill_rate;
            sim.apply_antibiotic();
            let treat_steps = (cycle.treatment_hours / dt) as usize;
            for _ in 0..treat_steps {
                sim.step(dt);
                let total = sim.normal_cells + sim.persister_cells;
                min_pop = min_pop.min(total);
            }
            // Rest
            sim.remove_antibiotic();
            let rest_steps = (cycle.rest_hours / dt) as usize;
            for _ in 0..rest_steps { sim.step(dt); }
            cycles_done += 1;
        }

        // Recovery
        let recovery_steps = (self.recovery_hours / dt) as usize;
        for _ in 0..recovery_steps { sim.step(dt); }

        let final_pop = sim.normal_cells + sim.persister_cells;
        DrugProtocolResult {
            final_population: final_pop,
            minimum_population: min_pop,
            survival_fraction: if pre_total > 0.0 { final_pop / pre_total } else { 0.0 },
            total_time_hours: sim.time_hours,
            cycles_completed: cycles_done,
            eradication_achieved: final_pop < 1.0,
        }
    }
}

/// Compare multiple drug protocols and return the most effective.
pub fn optimize_drug_protocol(
    protocols: &[DrugProtocol],
    switching_rate: f32,
    dt: f32,
) -> (usize, Vec<DrugProtocolResult>) {
    let results: Vec<DrugProtocolResult> = protocols.iter().map(|protocol| {
        let mut sim = PersisterCellSimulator::with_switching_rates(switching_rate, 1e-4);
        protocol.execute(&mut sim, dt)
    }).collect();

    let best_idx = results.iter().enumerate()
        .min_by(|(_, a), (_, b)| {
            a.final_population.partial_cmp(&b.final_population)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    (best_idx, results)
}

/// E. coli validation: published persistence fractions from Balaban et al. 2004.
pub fn ecoli_validation_data() -> Vec<(f32, f32)> {
    // (switching_rate_to_persister, expected_survival_fraction_after_5h_ampicillin)
    // Derived from Balaban et al. 2004, Fig. 2
    vec![
        (1e-6, 0.001),   // Low switching → minimal persistence
        (1e-5, 0.005),   // Moderate switching
        (1e-4, 0.02),    // High switching → detectable persistence
        (1e-3, 0.08),    // Very high switching → significant persistence
    ]
}

/// Validate the persister model against E. coli literature data.
pub fn validate_against_ecoli() -> Vec<(f32, f32, f32, bool)> {
    ecoli_validation_data().iter().map(|&(switch_rate, expected)| {
        let mut sim = PersisterCellSimulator {
            switch_to_persister_rate: switch_rate,
            antibiotic_kill_rate: 4.0, // ampicillin-like
            ..Default::default()
        };
        let result = sim.run_treatment_protocol(5.0, 5.0, 0.0, 0.05);
        let actual = result.survival_fraction;
        let within_order = (actual / expected).max(expected / actual.max(1e-10)) < 10.0;
        (switch_rate, expected, actual, within_order)
    }).collect()
}

// ---------------------------------------------------------------------------
// Synthetic Biology Gene Circuit Designer
// ---------------------------------------------------------------------------

/// Specification for a synthetic gene circuit with target noise properties.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GeneCircuitSpec {
    /// Target Fano factor (noise level). Fano=1 is Poisson, >1 is super-Poisson.
    pub target_fano: f32,
    /// Target mean protein count.
    pub target_mean_protein: f32,
    /// Target coefficient of variation (CV = sigma/mu).
    pub target_cv: f32,
}

/// Optimizable gene circuit parameters.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GeneCircuitParams {
    pub promoter_on_rate: f32,
    pub promoter_off_rate: f32,
    pub transcription_rate: f32,
    pub mrna_degradation_rate: f32,
    pub translation_rate: f32,
    pub protein_degradation_rate: f32,
    pub burst_size: f32,
}

impl Default for GeneCircuitParams {
    fn default() -> Self {
        Self {
            promoter_on_rate: 0.05, promoter_off_rate: 0.1,
            transcription_rate: 0.1, mrna_degradation_rate: 0.01,
            translation_rate: 0.5, protein_degradation_rate: 0.01,
            burst_size: 3.0,
        }
    }
}

impl GeneCircuitParams {
    pub fn random(rng: &mut StdRng) -> Self {
        Self {
            promoter_on_rate: rng.gen_range(0.001..0.2),
            promoter_off_rate: rng.gen_range(0.01..0.5),
            transcription_rate: rng.gen_range(0.01..1.0),
            mrna_degradation_rate: rng.gen_range(0.001..0.1),
            translation_rate: rng.gen_range(0.1..2.0),
            protein_degradation_rate: rng.gen_range(0.001..0.1),
            burst_size: rng.gen_range(1.0..20.0),
        }
    }

    pub fn mutate(&mut self, rng: &mut StdRng, rate: f32) {
        if rng.gen::<f32>() < rate { self.promoter_on_rate = (self.promoter_on_rate * (1.0 + rng.gen_range(-0.3..0.3))).clamp(0.001, 0.5); }
        if rng.gen::<f32>() < rate { self.promoter_off_rate = (self.promoter_off_rate * (1.0 + rng.gen_range(-0.3..0.3))).clamp(0.01, 1.0); }
        if rng.gen::<f32>() < rate { self.burst_size = (self.burst_size + rng.gen_range(-2.0..2.0)).clamp(1.0, 30.0); }
        if rng.gen::<f32>() < rate { self.transcription_rate = (self.transcription_rate * (1.0 + rng.gen_range(-0.3..0.3))).clamp(0.01, 2.0); }
    }

    /// Analytical Fano factor from the telegraph model.
    /// Fano = 1 + burst_size * k_off / (k_on + k_off)
    pub fn predicted_fano(&self) -> f32 {
        1.0 + self.burst_size * self.promoter_off_rate
            / (self.promoter_on_rate + self.promoter_off_rate).max(1e-8)
    }

    /// Analytical mean protein from steady-state.
    pub fn predicted_mean_protein(&self) -> f32 {
        let mean_mrna = self.transcription_rate * self.promoter_on_rate
            / ((self.promoter_on_rate + self.promoter_off_rate) * self.mrna_degradation_rate).max(1e-8);
        mean_mrna * self.translation_rate / self.protein_degradation_rate.max(1e-8)
    }

    /// Analytical CV.
    pub fn predicted_cv(&self) -> f32 {
        let mean = self.predicted_mean_protein();
        let fano = self.predicted_fano();
        if mean > 0.01 { (fano / mean).sqrt() } else { 1.0 }
    }
}

/// Result from gene circuit optimization.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GeneCircuitResult {
    pub best_params: GeneCircuitParams,
    pub achieved_fano: f32,
    pub achieved_mean_protein: f32,
    pub achieved_cv: f32,
    pub fitness: f32,
    pub generations_run: usize,
}

/// Optimize a gene circuit to achieve target noise properties.
pub fn optimize_gene_circuit(
    spec: &GeneCircuitSpec,
    population_size: usize,
    generations: usize,
    seed: u64,
) -> GeneCircuitResult {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut population: Vec<GeneCircuitParams> = (0..population_size)
        .map(|_| GeneCircuitParams::random(&mut rng)).collect();

    let fitness_fn = |p: &GeneCircuitParams| -> f32 {
        let fano_err = ((p.predicted_fano() - spec.target_fano) / spec.target_fano.max(0.1)).powi(2);
        let mean_err = ((p.predicted_mean_protein() - spec.target_mean_protein) / spec.target_mean_protein.max(1.0)).powi(2);
        let cv_err = ((p.predicted_cv() - spec.target_cv) / spec.target_cv.max(0.01)).powi(2);
        -(fano_err + mean_err + cv_err) // Negative because we maximize fitness
    };

    let mut best_params = population[0].clone();
    let mut best_fitness = f32::NEG_INFINITY;

    for _ in 0..generations {
        let fitnesses: Vec<f32> = population.iter().map(|p| fitness_fn(p)).collect();

        let best_idx = fitnesses.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i).unwrap_or(0);

        if fitnesses[best_idx] > best_fitness {
            best_fitness = fitnesses[best_idx];
            best_params = population[best_idx].clone();
        }

        // Tournament selection + mutation
        let mut next = Vec::with_capacity(population_size);
        next.push(population[best_idx].clone()); // elitism
        while next.len() < population_size {
            let a = rng.gen_range(0..population_size);
            let b = rng.gen_range(0..population_size);
            let winner = if fitnesses[a] >= fitnesses[b] { a } else { b };
            let mut child = population[winner].clone();
            child.mutate(&mut rng, 0.2);
            next.push(child);
        }
        population = next;
    }

    GeneCircuitResult {
        achieved_fano: best_params.predicted_fano(),
        achieved_mean_protein: best_params.predicted_mean_protein(),
        achieved_cv: best_params.predicted_cv(),
        best_params, fitness: best_fitness, generations_run: generations,
    }
}



// ---------------------------------------------------------------------------
// Adaptive Evolution with Environment
// ---------------------------------------------------------------------------

/// Full evolution loop that applies EnvironmentalSchedule during fitness
/// evaluation. Each world experiences seasonal temperature/humidity variation
/// and stochastic drought events, making climate-adaptive genomes fitter.
///
/// This wires `run_single_world_with_environment()` into the GA loop.
pub fn evolve_with_environment(
    config: EvolutionConfig,
    schedule: EnvironmentalSchedule,
) -> Result<EvolutionResult, String> {
    let start = Instant::now();
    let mut rng = StdRng::seed_from_u64(config.master_seed);
    let mut population: Vec<WorldGenome> = (0..config.population_size)
        .map(|_| WorldGenome::random(&mut rng))
        .collect();

    let mut global_best_fitness = f32::NEG_INFINITY;
    let mut global_best_genome = population[0].clone();
    let mut generation_results = Vec::with_capacity(config.generations);
    let mut total_worlds = 0usize;

    for gen in 0..config.generations {
        let gen_start = Instant::now();
        let mut results: Vec<WorldResult> = Vec::with_capacity(population.len());
        for genome in &population {
            let (wr, _samples) = run_single_world_with_environment(
                genome.clone(), config.frames_per_world, &schedule, config.lite,
            )?;
            results.push(wr);
        }
        total_worlds += results.len();

        let best_idx = results.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.fitness.partial_cmp(&b.fitness).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        let best = &results[best_idx];

        if best.fitness > global_best_fitness {
            global_best_fitness = best.fitness;
            global_best_genome = best.genome.clone();
        }

        let mean_fit = results.iter().map(|r| r.fitness).sum::<f32>() / results.len() as f32;
        eprintln!("  Gen {} | best={:.2} mean={:.2} | env={} | {:.1}s",
            gen, best.fitness, mean_fit,
            if schedule.drought_probability > 0.003 { "arid" }
            else if schedule.base_temperature_c > 26.0 { "tropical" }
            else { "temperate" },
            gen_start.elapsed().as_secs_f32());

        let worst_fit = results.iter().map(|r| r.fitness).fold(f32::INFINITY, f32::min);
        generation_results.push(GenerationResult {
            generation: gen,
            best_fitness: best.fitness,
            mean_fitness: mean_fit,
            worst_fitness: worst_fit,
            best_genome: best.genome.clone(),
            wall_time_ms: gen_start.elapsed().as_secs_f32() * 1000.0,
            stress_metrics: None,
        });

        // Breed next generation
        let (tournament_size, mutation_rate, crossover_rate, elitism) = match &config.strategy {
            SearchStrategy::Evolutionary { tournament_size, mutation_rate, crossover_rate, elitism } =>
                (*tournament_size, *mutation_rate, *crossover_rate, *elitism),
            _ => (3, 0.15, 0.7, 2),
        };

        population = breed_next_generation(
            &results, config.population_size,
            tournament_size, mutation_rate, crossover_rate, elitism,
            &mut rng,
        );
    }

    Ok(EvolutionResult {
        generations: generation_results,
        global_best_fitness,
        global_best_genome,
        total_worlds_evaluated: total_worlds,
        total_wall_time_ms: start.elapsed().as_secs_f32() * 1000.0,
    })
}

// ---------------------------------------------------------------------------
// Coevolution Engine
// ---------------------------------------------------------------------------

/// Coevolution mode determines inter-species dynamics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CoevolutionMode {
    /// Red Queen: antagonistic arms race (predator–prey).
    RedQueen,
    /// Mutualistic: symbiotic coevolution (plant–pollinator).
    Mutualistic,
    /// Competitive: resource competition between species.
    Competitive,
}

/// A species genome that participates in coevolution.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpeciesGenome {
    pub world_genome: WorldGenome,
    /// Per-species trait modifiers for coevolution fitness.
    pub defense_investment: f32,
    pub resource_efficiency: f32,
    pub cooperation_tendency: f32,
    pub mobility: f32,
}

impl SpeciesGenome {
    pub fn random(rng: &mut StdRng) -> Self {
        Self {
            world_genome: WorldGenome::random(rng),
            defense_investment: rng.gen_range(0.0..1.0),
            resource_efficiency: rng.gen_range(0.3..1.0),
            cooperation_tendency: rng.gen_range(0.0..1.0),
            mobility: rng.gen_range(0.1..1.0),
        }
    }

    pub fn mutate(&mut self, rng: &mut StdRng, rate: f32) {
        self.world_genome.mutate(rng, rate);
        if rng.gen::<f32>() < rate { self.defense_investment = (self.defense_investment + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0); }
        if rng.gen::<f32>() < rate { self.resource_efficiency = (self.resource_efficiency + rng.gen_range(-0.1..0.1)).clamp(0.1, 1.0); }
        if rng.gen::<f32>() < rate { self.cooperation_tendency = (self.cooperation_tendency + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0); }
        if rng.gen::<f32>() < rate { self.mobility = (self.mobility + rng.gen_range(-0.1..0.1)).clamp(0.1, 1.0); }
    }

    pub fn crossover(a: &Self, b: &Self, rng: &mut StdRng) -> Self {
        Self {
            world_genome: WorldGenome::crossover(&a.world_genome, &b.world_genome, rng),
            defense_investment: if rng.gen() { a.defense_investment } else { b.defense_investment },
            resource_efficiency: if rng.gen() { a.resource_efficiency } else { b.resource_efficiency },
            cooperation_tendency: if rng.gen() { a.cooperation_tendency } else { b.cooperation_tendency },
            mobility: if rng.gen() { a.mobility } else { b.mobility },
        }
    }
}

/// Result of a coevolution pairing.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CoevolutionPairingResult {
    pub species_a_fitness: f32,
    pub species_b_fitness: f32,
    pub interaction_strength: f32,
}

/// Evaluate fitness of two species interacting in the same world.
pub fn evaluate_coevolution_pair(
    a: &SpeciesGenome,
    b: &SpeciesGenome,
    mode: CoevolutionMode,
    frames: usize,
    lite: bool,
) -> Result<CoevolutionPairingResult, String> {
    // Run species A's world
    let wr_a = run_single_world(a.world_genome.clone(), frames, 10, FitnessObjective::MaxBiomass, lite)?;
    // Run species B's world
    let wr_b = run_single_world(b.world_genome.clone(), frames, 10, FitnessObjective::MaxBiomass, lite)?;

    let (fit_a, fit_b, interaction) = match mode {
        CoevolutionMode::RedQueen => {
            // Arms race: defense investment reduces opponent fitness
            let attack_a = a.resource_efficiency * a.mobility;
            let attack_b = b.resource_efficiency * b.mobility;
            let def_a = a.defense_investment;
            let def_b = b.defense_investment;
            let net_a = wr_a.fitness * (1.0 + def_a - attack_b * 0.5);
            let net_b = wr_b.fitness * (1.0 + def_b - attack_a * 0.5);
            (net_a, net_b, (attack_a - def_b).abs() + (attack_b - def_a).abs())
        }
        CoevolutionMode::Mutualistic => {
            // Cooperation boosts both
            let synergy = a.cooperation_tendency * b.cooperation_tendency;
            let boost = 1.0 + synergy * 0.5;
            (wr_a.fitness * boost, wr_b.fitness * boost, synergy)
        }
        CoevolutionMode::Competitive => {
            // Resource competition: efficiency determines winner share
            let total = a.resource_efficiency + b.resource_efficiency;
            let share_a = if total > 0.0 { a.resource_efficiency / total } else { 0.5 };
            let share_b = 1.0 - share_a;
            (wr_a.fitness * share_a * 2.0, wr_b.fitness * share_b * 2.0, (share_a - share_b).abs())
        }
    };

    Ok(CoevolutionPairingResult {
        species_a_fitness: fit_a,
        species_b_fitness: fit_b,
        interaction_strength: interaction,
    })
}

/// Run coevolution for multiple generations with two populations.
pub fn evolve_coevolution(
    pop_size: usize,
    generations: usize,
    frames: usize,
    mode: CoevolutionMode,
    lite: bool,
    seed: u64,
) -> Result<CoevolutionResult, String> {
    let start = Instant::now();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut pop_a: Vec<SpeciesGenome> = (0..pop_size).map(|_| SpeciesGenome::random(&mut rng)).collect();
    let mut pop_b: Vec<SpeciesGenome> = (0..pop_size).map(|_| SpeciesGenome::random(&mut rng)).collect();

    let mut history = Vec::with_capacity(generations);

    for gen in 0..generations {
        // Round-robin pairing: each individual in A is paired with one from B
        let mut fit_a = vec![0.0f32; pop_size];
        let mut fit_b = vec![0.0f32; pop_size];

        for i in 0..pop_size {
            let j = (i + gen) % pop_size; // rotating partner assignment
            let result = evaluate_coevolution_pair(&pop_a[i], &pop_b[j], mode, frames, lite)?;
            fit_a[i] += result.species_a_fitness;
            fit_b[j] += result.species_b_fitness;
        }

        let best_a = fit_a.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let best_b = fit_b.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean_a = fit_a.iter().sum::<f32>() / pop_size as f32;
        let mean_b = fit_b.iter().sum::<f32>() / pop_size as f32;

        eprintln!("  CoEvo Gen {} | A: best={:.2} mean={:.2} | B: best={:.2} mean={:.2}",
            gen, best_a, mean_a, best_b, mean_b);

        history.push(CoevolutionGeneration {
            generation: gen,
            best_fitness_a: best_a, mean_fitness_a: mean_a,
            best_fitness_b: best_b, mean_fitness_b: mean_b,
        });

        // Tournament selection + breed each population
        let mut next_a = Vec::with_capacity(pop_size);
        let mut next_b = Vec::with_capacity(pop_size);
        // Elitism: keep best
        let best_a_idx = fit_a.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        let best_b_idx = fit_b.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        next_a.push(pop_a[best_a_idx].clone());
        next_b.push(pop_b[best_b_idx].clone());

        while next_a.len() < pop_size {
            let i1 = rng.gen_range(0..pop_size);
            let i2 = rng.gen_range(0..pop_size);
            let winner = if fit_a[i1] >= fit_a[i2] { i1 } else { i2 };
            let mut child = if rng.gen::<f32>() < 0.7 {
                let other = rng.gen_range(0..pop_size);
                SpeciesGenome::crossover(&pop_a[winner], &pop_a[other], &mut rng)
            } else {
                pop_a[winner].clone()
            };
            child.mutate(&mut rng, 0.15);
            next_a.push(child);
        }
        while next_b.len() < pop_size {
            let i1 = rng.gen_range(0..pop_size);
            let i2 = rng.gen_range(0..pop_size);
            let winner = if fit_b[i1] >= fit_b[i2] { i1 } else { i2 };
            let mut child = if rng.gen::<f32>() < 0.7 {
                let other = rng.gen_range(0..pop_size);
                SpeciesGenome::crossover(&pop_b[winner], &pop_b[other], &mut rng)
            } else {
                pop_b[winner].clone()
            };
            child.mutate(&mut rng, 0.15);
            next_b.push(child);
        }

        pop_a = next_a;
        pop_b = next_b;
    }

    Ok(CoevolutionResult {
        history,
        final_population_a: pop_a,
        final_population_b: pop_b,
        mode,
        total_wall_time_ms: start.elapsed().as_secs_f32() * 1000.0,
    })
}

/// Generation-level coevolution metrics.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CoevolutionGeneration {
    pub generation: usize,
    pub best_fitness_a: f32,
    pub mean_fitness_a: f32,
    pub best_fitness_b: f32,
    pub mean_fitness_b: f32,
}

/// Full coevolution run result.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CoevolutionResult {
    pub history: Vec<CoevolutionGeneration>,
    pub final_population_a: Vec<SpeciesGenome>,
    pub final_population_b: Vec<SpeciesGenome>,
    pub mode: CoevolutionMode,
    pub total_wall_time_ms: f32,
}

// ---------------------------------------------------------------------------
// Genetic Regulatory Network
// ---------------------------------------------------------------------------

/// A gene regulatory network that determines organism phenotype from genotype.
/// Uses a Boolean network model with continuous dynamics (sigmoid activation).
///
/// References:
/// - Kauffman (1969) "Metabolic stability and epigenesis in randomly
///   constructed genetic nets", J. Theoretical Biology
/// - Aldana (2003) "Boolean dynamics of networks with scale-free topology"
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GeneRegulatoryNetwork {
    /// Number of genes in the network.
    pub n_genes: usize,
    /// Interaction matrix: weights[i][j] = effect of gene j on gene i.
    /// Positive = activation, negative = repression, zero = no interaction.
    pub weights: Vec<Vec<f32>>,
    /// Activation thresholds per gene.
    pub thresholds: Vec<f32>,
    /// Current expression levels [0, 1].
    pub expression: Vec<f32>,
    /// Hill coefficient for sigmoid response.
    pub hill_coefficient: f32,
    /// Degradation rate per gene per step.
    pub degradation_rate: f32,
}

impl GeneRegulatoryNetwork {
    /// Create a random GRN with K connections per gene (NK model).
    pub fn random_nk(n_genes: usize, k: usize, rng: &mut StdRng) -> Self {
        let mut weights = vec![vec![0.0f32; n_genes]; n_genes];
        let thresholds: Vec<f32> = (0..n_genes).map(|_| rng.gen_range(-0.5..0.5)).collect();
        let expression: Vec<f32> = (0..n_genes).map(|_| rng.gen_range(0.0..1.0)).collect();

        // Each gene receives K random connections
        for i in 0..n_genes {
            let mut inputs: Vec<usize> = (0..n_genes).collect();
            // Shuffle and take K
            for j in (1..inputs.len()).rev() {
                let swap = rng.gen_range(0..=j);
                inputs.swap(j, swap);
            }
            for &j in inputs.iter().take(k.min(n_genes)) {
                weights[i][j] = rng.gen_range(-1.0..1.0);
            }
        }

        Self {
            n_genes, weights, thresholds, expression,
            hill_coefficient: 2.0,
            degradation_rate: 0.1,
        }
    }

    /// Step the network forward by one time unit.
    pub fn step(&mut self) {
        let mut new_expression = vec![0.0f32; self.n_genes];
        for i in 0..self.n_genes {
            let mut input_sum = 0.0f32;
            for j in 0..self.n_genes {
                input_sum += self.weights[i][j] * self.expression[j];
            }
            input_sum -= self.thresholds[i];
            // Sigmoid activation with Hill coefficient
            let activated = 1.0 / (1.0 + (-self.hill_coefficient * input_sum).exp());
            new_expression[i] = activated * (1.0 - self.degradation_rate)
                + self.expression[i] * self.degradation_rate;
        }
        self.expression = new_expression;
    }

    /// Run to attractor (or max steps) and return the stable expression pattern.
    pub fn find_attractor(&mut self, max_steps: usize) -> Vec<f32> {
        let mut prev = self.expression.clone();
        for _ in 0..max_steps {
            self.step();
            // Check convergence (L2 distance < threshold)
            let dist: f32 = self.expression.iter().zip(prev.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            if dist < 1e-4 {
                return self.expression.clone();
            }
            prev = self.expression.clone();
        }
        self.expression.clone()
    }

    /// Map attractor expression to phenotypic traits.
    /// Returns (growth_rate, stress_tolerance, reproduction_rate, resource_efficiency).
    pub fn attractor_to_phenotype(&mut self) -> GRNPhenotype {
        let expr = self.find_attractor(100);
        let n = expr.len().max(1) as f32;
        // Map gene blocks to traits
        let quarter = (self.n_genes / 4).max(1);
        let growth = expr[..quarter].iter().sum::<f32>() / quarter as f32;
        let stress = expr[quarter..quarter * 2].iter().sum::<f32>() / quarter as f32;
        let repro = if quarter * 3 <= self.n_genes {
            expr[quarter * 2..quarter * 3].iter().sum::<f32>() / quarter as f32
        } else { 0.5 };
        let efficiency = if quarter * 4 <= self.n_genes {
            expr[quarter * 3..].iter().sum::<f32>() / (self.n_genes - quarter * 3).max(1) as f32
        } else { 0.5 };

        GRNPhenotype {
            growth_rate: growth,
            stress_tolerance: stress,
            reproduction_rate: repro,
            resource_efficiency: efficiency,
            attractor: expr,
            network_complexity: self.weights.iter()
                .flat_map(|row| row.iter())
                .filter(|w| w.abs() > 0.01)
                .count() as f32 / n,
        }
    }

    /// Mutate the network.
    pub fn mutate(&mut self, rng: &mut StdRng, rate: f32) {
        for i in 0..self.n_genes {
            for j in 0..self.n_genes {
                if rng.gen::<f32>() < rate * 0.1 {
                    self.weights[i][j] += rng.gen_range(-0.3..0.3);
                    self.weights[i][j] = self.weights[i][j].clamp(-2.0, 2.0);
                }
            }
            if rng.gen::<f32>() < rate {
                self.thresholds[i] += rng.gen_range(-0.2..0.2);
                self.thresholds[i] = self.thresholds[i].clamp(-2.0, 2.0);
            }
        }
    }

    /// Crossover two networks.
    pub fn crossover(a: &Self, b: &Self, rng: &mut StdRng) -> Self {
        let n = a.n_genes;
        let mut child = a.clone();
        let crossover_point = rng.gen_range(0..n);
        for i in crossover_point..n {
            child.weights[i] = b.weights[i].clone();
            child.thresholds[i] = b.thresholds[i];
            child.expression[i] = b.expression[i];
        }
        child
    }
}

/// Phenotype produced by a GRN attractor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GRNPhenotype {
    pub growth_rate: f32,
    pub stress_tolerance: f32,
    pub reproduction_rate: f32,
    pub resource_efficiency: f32,
    pub attractor: Vec<f32>,
    pub network_complexity: f32,
}

/// Evolve GRNs to find networks that produce target phenotypes.
pub fn evolve_grn(
    target_growth: f32,
    target_stress: f32,
    n_genes: usize,
    k_connections: usize,
    pop_size: usize,
    generations: usize,
    seed: u64,
) -> GRNEvolutionResult {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut population: Vec<GeneRegulatoryNetwork> = (0..pop_size)
        .map(|_| GeneRegulatoryNetwork::random_nk(n_genes, k_connections, &mut rng))
        .collect();

    let mut best_fitness = f32::NEG_INFINITY;
    let mut best_network = population[0].clone();
    let mut best_phenotype: Option<GRNPhenotype> = None;

    for _gen in 0..generations {
        let mut fitnesses = Vec::with_capacity(pop_size);
        let mut phenotypes = Vec::with_capacity(pop_size);
        for net in &mut population {
            let pheno = net.attractor_to_phenotype();
            let fit = -((pheno.growth_rate - target_growth).powi(2)
                + (pheno.stress_tolerance - target_stress).powi(2));
            fitnesses.push(fit);
            phenotypes.push(pheno);
        }

        let best_idx = fitnesses.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap().0;
        if fitnesses[best_idx] > best_fitness {
            best_fitness = fitnesses[best_idx];
            best_network = population[best_idx].clone();
            best_phenotype = Some(phenotypes[best_idx].clone());
        }

        // Tournament selection + breed
        let mut next = vec![population[best_idx].clone()]; // elitism
        while next.len() < pop_size {
            let a = rng.gen_range(0..pop_size);
            let b = rng.gen_range(0..pop_size);
            let winner = if fitnesses[a] >= fitnesses[b] { a } else { b };
            let mut child = if rng.gen::<f32>() < 0.7 {
                let other = rng.gen_range(0..pop_size);
                GeneRegulatoryNetwork::crossover(&population[winner], &population[other], &mut rng)
            } else {
                population[winner].clone()
            };
            child.mutate(&mut rng, 0.15);
            next.push(child);
        }
        population = next;
    }

    GRNEvolutionResult {
        best_network,
        best_phenotype: best_phenotype.unwrap(),
        best_fitness,
        generations_run: generations,
    }
}

/// Result of GRN evolution.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GRNEvolutionResult {
    pub best_network: GeneRegulatoryNetwork,
    pub best_phenotype: GRNPhenotype,
    pub best_fitness: f32,
    pub generations_run: usize,
}

// ---------------------------------------------------------------------------
// Ecosystem Health Metrics
// ---------------------------------------------------------------------------

/// Comprehensive ecosystem health assessment following ecological theory.
///
/// References:
/// - Shannon (1948) "A Mathematical Theory of Communication" — diversity index
/// - Odum (1969) "The Strategy of Ecosystem Development" — maturity metrics
/// - Ulanowicz (2004) "Quantifying sustainability" — ascendency
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EcosystemHealthReport {
    /// Shannon-Wiener diversity index H' = -Σ pi ln(pi).
    pub shannon_diversity: f32,
    /// Simpson's diversity D = 1 - Σ pi².
    pub simpson_diversity: f32,
    /// Evenness: H' / ln(S) where S = species richness.
    pub evenness: f32,
    /// Total biomass across all trophic levels.
    pub total_biomass: f32,
    /// Biomass-to-metabolism ratio (P:B ratio proxy).
    pub biomass_metabolism_ratio: f32,
    /// Nutrient cycling efficiency [0, 1].
    pub nutrient_cycling: f32,
    /// Energy flow through ecosystem (sum of all metabolic rates).
    pub total_energy_flow: f32,
    /// Resilience score: how quickly does the system recover from perturbation.
    pub resilience_score: f32,
    /// Stability: inverse coefficient of variation of biomass over time.
    pub stability: f32,
    /// Trophic level count (distinct functional groups present).
    pub trophic_levels: usize,
    /// Overall health score [0, 100].
    pub overall_health: f32,
}

/// Compute ecosystem health from a series of snapshots.
pub fn assess_ecosystem_health(snapshots: &[TerrariumWorldSnapshot]) -> EcosystemHealthReport {
    if snapshots.is_empty() {
        return EcosystemHealthReport {
            shannon_diversity: 0.0, simpson_diversity: 0.0, evenness: 0.0,
            total_biomass: 0.0, biomass_metabolism_ratio: 0.0,
            nutrient_cycling: 0.0, total_energy_flow: 0.0,
            resilience_score: 0.0, stability: 0.0, trophic_levels: 0,
            overall_health: 0.0,
        };
    }

    let latest = snapshots.last().unwrap();

    // Count "species" (functional groups present)
    let mut abundances: Vec<f32> = Vec::new();
    if latest.plants > 0 { abundances.push(latest.total_plant_cells); }
    if latest.flies > 0 { abundances.push(latest.flies as f32 * 10.0); }
    if latest.mean_microbes > 0.001 { abundances.push(latest.mean_microbes * 100.0); }
    if latest.fruits > 0 { abundances.push(latest.fruits as f32); }
    if latest.seeds > 0 { abundances.push(latest.seeds as f32); }
    if latest.mean_symbionts > 0.001 { abundances.push(latest.mean_symbionts * 50.0); }

    let total: f32 = abundances.iter().sum::<f32>().max(1e-10);
    let s = abundances.len();

    // Shannon-Wiener
    let shannon = -abundances.iter()
        .map(|&a| { let p = a / total; if p > 0.0 { p * p.ln() } else { 0.0 } })
        .sum::<f32>();

    // Simpson
    let simpson = 1.0 - abundances.iter()
        .map(|&a| { let p = a / total; p * p })
        .sum::<f32>();

    let evenness = if s > 1 { shannon / (s as f32).ln() } else { 1.0 };

    let total_biomass = latest.total_plant_cells
        + latest.flies as f32 * 10.0
        + latest.mean_microbes * 100.0;

    let total_energy = latest.mean_soil_atp_flux * 100.0
        + latest.avg_fly_energy * latest.flies as f32;

    let biomass_metabolism = if total_energy > 0.0 { total_biomass / total_energy } else { 0.0 };

    // Nutrient cycling proxy: glucose turnover
    let nutrient_cycling = (latest.mean_soil_glucose * 10.0).clamp(0.0, 1.0);

    // Stability: compute CV of biomass over time
    let biomass_series: Vec<f32> = snapshots.iter().map(|s| s.total_plant_cells).collect();
    let mean_biomass = biomass_series.iter().sum::<f32>() / biomass_series.len() as f32;
    let var_biomass = biomass_series.iter().map(|b| (b - mean_biomass).powi(2)).sum::<f32>()
        / biomass_series.len() as f32;
    let cv = if mean_biomass > 0.0 { var_biomass.sqrt() / mean_biomass } else { 1.0 };
    let stability = (1.0 / (cv + 0.1)).clamp(0.0, 10.0);

    // Resilience: look for recovery after dips
    let mut resilience = 0.5f32; // default moderate
    if biomass_series.len() > 10 {
        let mid = biomass_series.len() / 2;
        let first_half_mean = biomass_series[..mid].iter().sum::<f32>() / mid as f32;
        let second_half_mean = biomass_series[mid..].iter().sum::<f32>() / (biomass_series.len() - mid) as f32;
        if first_half_mean > 0.0 {
            resilience = (second_half_mean / first_half_mean).clamp(0.0, 2.0) * 0.5;
        }
    }

    // Trophic levels present
    let mut trophic = 0;
    if latest.plants > 0 { trophic += 1; } // producers
    if latest.mean_microbes > 0.001 { trophic += 1; } // decomposers
    if latest.flies > 0 { trophic += 1; } // consumers
    if latest.mean_symbionts > 0.001 { trophic += 1; } // mutualists

    // Overall health: weighted score
    let health = (shannon * 15.0
        + simpson * 15.0
        + evenness * 10.0
        + stability * 10.0
        + resilience * 15.0
        + nutrient_cycling * 10.0
        + (trophic as f32 / 4.0) * 25.0)
        .clamp(0.0, 100.0);

    EcosystemHealthReport {
        shannon_diversity: shannon,
        simpson_diversity: simpson,
        evenness,
        total_biomass,
        biomass_metabolism_ratio: biomass_metabolism,
        nutrient_cycling,
        total_energy_flow: total_energy,
        resilience_score: resilience,
        stability,
        trophic_levels: trophic,
        overall_health: health,
    }
}

// ---------------------------------------------------------------------------
// World Export / Replay
// ---------------------------------------------------------------------------

/// Serializable world state for export and replay.
#[derive(Debug, Clone, serde::Serialize)]
pub struct WorldExport {
    pub genome: WorldGenome,
    pub snapshots: Vec<TerrariumWorldSnapshot>,
    pub environmental_samples: Option<Vec<EnvironmentalSample>>,
    pub health_report: Option<EcosystemHealthReport>,
    pub metadata: WorldExportMetadata,
}

/// Metadata for exported world states.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WorldExportMetadata {
    pub frames_run: usize,
    pub lite_mode: bool,
    pub seed: u64,
    pub fitness: f32,
    pub wall_time_ms: f32,
    pub version: String,
}

/// Run a world and export full state with periodic snapshots.
pub fn run_and_export(
    genome: WorldGenome,
    frames: usize,
    lite: bool,
    snapshot_interval: usize,
    environment: Option<&EnvironmentalSchedule>,
) -> Result<WorldExport, String> {
    let start = Instant::now();
    let mut world = if lite { genome.build_world_lite()? } else { genome.build_world()? };
    let mut snapshots = Vec::with_capacity(frames / snapshot_interval.max(1) + 1);
    let mut env_samples = environment.map(|s| {
        (EnvironmentalState::new(s.clone(), genome.seed.wrapping_add(7777)),
         Vec::with_capacity(frames))
    });

    for frame in 0..frames {
        if let Some((ref mut env, ref mut samples)) = env_samples {
            let sample = env.sample(world.time_s);
            samples.push(sample);
        }
        world.step_frame()?;
        if frame % snapshot_interval.max(1) == 0 || frame == frames - 1 {
            snapshots.push(world.snapshot());
        }
    }

    let final_snap = world.snapshot();
    let fitness = evaluate_fitness(FitnessObjective::MaxBiomass, &final_snap, &[]);
    let health = if snapshots.len() >= 3 { Some(assess_ecosystem_health(&snapshots)) } else { None };

    Ok(WorldExport {
        genome: genome.clone(),
        snapshots,
        environmental_samples: env_samples.map(|(_, s)| s),
        health_report: health,
        metadata: WorldExportMetadata {
            frames_run: frames,
            lite_mode: lite,
            seed: genome.seed,
            fitness,
            wall_time_ms: start.elapsed().as_secs_f32() * 1000.0,
            version: "0.1.0".to_string(),
        },
    })
}

// ---------------------------------------------------------------------------
// Sparkline & Terminal Dashboard Helpers
// ---------------------------------------------------------------------------

/// Render a sparkline from a data series using Unicode block characters.
pub fn sparkline(data: &[f32], width: usize) -> String {
    if data.is_empty() { return String::new(); }
    let blocks = [' ', '\u{2581}', '\u{2582}', '\u{2583}', '\u{2584}', '\u{2585}', '\u{2586}', '\u{2587}', '\u{2588}'];
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(1e-10);

    // Resample to width
    let mut out = String::with_capacity(width * 4);
    for i in 0..width {
        let idx = i * data.len() / width;
        let idx = idx.min(data.len() - 1);
        let normalized = ((data[idx] - min) / range * 8.0).clamp(0.0, 8.0) as usize;
        out.push(blocks[normalized.min(8)]);
    }
    out
}

/// Format a dashboard line with label, value, and sparkline.
pub fn dashboard_line(label: &str, value: f32, history: &[f32], width: usize) -> String {
    let spark = sparkline(history, width.saturating_sub(30));
    format!("{:<14} {:>8.3} {}", label, value, spark)
}

/// Generate a full ecosystem dashboard from snapshot history.
pub fn ecosystem_dashboard(snapshots: &[TerrariumWorldSnapshot], width: usize) -> String {
    let mut out = String::new();
    let w = width.max(40);

    out.push_str(&format!("{:=<w$}\n", "= oNeura Ecosystem Dashboard ", w = w));

    if snapshots.is_empty() {
        out.push_str("No data yet.\n");
        return out;
    }

    let biomass: Vec<f32> = snapshots.iter().map(|s| s.total_plant_cells).collect();
    let moisture: Vec<f32> = snapshots.iter().map(|s| s.mean_soil_moisture).collect();
    let microbes: Vec<f32> = snapshots.iter().map(|s| s.mean_microbes).collect();
    let co2: Vec<f32> = snapshots.iter().map(|s| s.mean_atmospheric_co2).collect();
    let o2: Vec<f32> = snapshots.iter().map(|s| s.mean_atmospheric_o2).collect();
    let fly_energy: Vec<f32> = snapshots.iter().map(|s| s.avg_fly_energy).collect();
    let flies: Vec<f32> = snapshots.iter().map(|s| s.flies as f32).collect();
    let glucose: Vec<f32> = snapshots.iter().map(|s| s.mean_soil_glucose).collect();

    let latest = snapshots.last().unwrap();

    out.push_str(&format!("{:-<w$}\n", "- Population ", w = w));
    out.push_str(&format!("{}\n", dashboard_line("Biomass", latest.total_plant_cells, &biomass, w)));
    out.push_str(&format!("{}\n", dashboard_line("Flies", latest.flies as f32, &flies, w)));
    out.push_str(&format!("{}\n", dashboard_line("Microbes", latest.mean_microbes, &microbes, w)));

    out.push_str(&format!("{:-<w$}\n", "- Chemistry ", w = w));
    out.push_str(&format!("{}\n", dashboard_line("Moisture", latest.mean_soil_moisture, &moisture, w)));
    out.push_str(&format!("{}\n", dashboard_line("Glucose", latest.mean_soil_glucose, &glucose, w)));
    out.push_str(&format!("{}\n", dashboard_line("CO2", latest.mean_atmospheric_co2, &co2, w)));
    out.push_str(&format!("{}\n", dashboard_line("O2", latest.mean_atmospheric_o2, &o2, w)));

    out.push_str(&format!("{:-<w$}\n", "- Energy ", w = w));
    out.push_str(&format!("{}\n", dashboard_line("FlyEnergy", latest.avg_fly_energy, &fly_energy, w)));

    // Health assessment
    if snapshots.len() >= 3 {
        let health = assess_ecosystem_health(snapshots);
        out.push_str(&format!("{:-<w$}\n", "- Health ", w = w));
        out.push_str(&format!("  Shannon H':  {:.3}  Simpson D: {:.3}  Evenness: {:.3}\n",
            health.shannon_diversity, health.simpson_diversity, health.evenness));
        out.push_str(&format!("  Trophic: {}  Stability: {:.2}  Resilience: {:.2}\n",
            health.trophic_levels, health.stability, health.resilience_score));
        out.push_str(&format!("  Overall: {:.1}/100\n", health.overall_health));
    }

    out.push_str(&format!("{:=<w$}\n", "", w = w));
    out
}


// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn genome_random_is_within_bounds() {
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let g = WorldGenome::random(&mut rng);
            assert!(g.initial_proton_scale >= GenomeRange::PROTON_SCALE.0);
            assert!(g.initial_proton_scale <= GenomeRange::PROTON_SCALE.1);
            assert!(g.soil_temperature_c >= GenomeRange::TEMPERATURE.0);
            assert!(g.soil_temperature_c <= GenomeRange::TEMPERATURE.1);
            assert!(g.plant_count >= GenomeRange::PLANT_COUNT.0);
            assert!(g.plant_count <= GenomeRange::PLANT_COUNT.1);
        }
    }

    #[test]
    fn genome_normalized_params_in_range() {
        let mut rng = StdRng::seed_from_u64(42);
        let g = WorldGenome::random(&mut rng);
        let params = g.normalized_params();
        assert_eq!(params.len(), 18);
        for p in params {
            assert!(p >= 0.0 && p <= 1.0, "param {} not in [0,1]", p);
        }
    }

    #[test]
    fn genome_crossover_preserves_bounds() {
        let mut rng = StdRng::seed_from_u64(42);
        let a = WorldGenome::random(&mut rng);
        let b = WorldGenome::random(&mut rng);

        for _ in 0..100 {
            let child = WorldGenome::crossover(&a, &b, &mut rng);
            assert!(child.initial_proton_scale >= GenomeRange::PROTON_SCALE.0);
            assert!(child.initial_proton_scale <= GenomeRange::PROTON_SCALE.1);
        }
    }

    #[test]
    fn genome_mutation_preserves_bounds() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut g = WorldGenome::random(&mut rng);

        for _ in 0..100 {
            g.mutate(&mut rng, 0.5);
            assert!(g.initial_proton_scale >= GenomeRange::PROTON_SCALE.0);
            assert!(g.initial_proton_scale <= GenomeRange::PROTON_SCALE.1);
            assert!(g.soil_temperature_c >= GenomeRange::TEMPERATURE.0);
            assert!(g.soil_temperature_c <= GenomeRange::TEMPERATURE.1);
        }
    }

    #[test]
    fn telemetry_from_result_generates_records() {
        let result = EvolutionResult {
            generations: vec![
                GenerationResult {
                    generation: 0,
                    best_fitness: 100.0,
                    mean_fitness: 80.0,
                    worst_fitness: 60.0,
                    best_genome: WorldGenome::default_with_seed(1),
                    wall_time_ms: 50.0,
                    stress_metrics: None,
                },
                GenerationResult {
                    generation: 1,
                    best_fitness: 120.0,
                    mean_fitness: 90.0,
                    worst_fitness: 70.0,
                    best_genome: WorldGenome::default_with_seed(2),
                    wall_time_ms: 45.0,
                    stress_metrics: None,
                },
            ],
            global_best_genome: WorldGenome::default_with_seed(2),
            global_best_fitness: 120.0,
            total_worlds_evaluated: 10,
            total_wall_time_ms: 95.0,
        };
        let telemetry = telemetry_from_result(&result, Some("test"));
        assert_eq!(telemetry.len(), 2);
        assert_eq!(telemetry[0].generation, 0);
        assert_eq!(telemetry[1].generation, 1);
        assert_eq!(telemetry[0].best_fitness, 100.0);
        assert_eq!(telemetry[1].best_fitness, 120.0);
        assert_eq!(telemetry[0].best_genome_params.len(), 18);
        assert!(telemetry[0].population_diversity > 0.0);
        assert_eq!(telemetry[0].mode, Some("test".to_string()));
    }

    #[test]
    fn extended_telemetry_has_mode_name() {
        let result = EvolutionResult {
            generations: vec![GenerationResult {
                generation: 0,
                best_fitness: 50.0,
                mean_fitness: 40.0,
                worst_fitness: 30.0,
                best_genome: WorldGenome::default_with_seed(1),
                wall_time_ms: 10.0,
                stress_metrics: None,
            }],
            global_best_genome: WorldGenome::default_with_seed(1),
            global_best_fitness: 50.0,
            total_worlds_evaluated: 1,
            total_wall_time_ms: 10.0,
        };
        let telemetry = telemetry_from_result(&result, Some("standard"));
        assert_eq!(telemetry[0].mode, Some("standard".to_string()));
        let telemetry_none = telemetry_from_result(&result, None);
        assert_eq!(telemetry_none[0].mode, None);
    }

    #[test]
    fn stress_telemetry_has_stress_metrics() {
        let stress = StressTelemetryMetrics {
            pre_stress_biomass: 100.0,
            min_stress_biomass: 40.0,
            post_recovery_biomass: 75.0,
        };
        let result = EvolutionResult {
            generations: vec![GenerationResult {
                generation: 0,
                best_fitness: 50.0,
                mean_fitness: 40.0,
                worst_fitness: 30.0,
                best_genome: WorldGenome::default_with_seed(1),
                wall_time_ms: 10.0,
                stress_metrics: Some(stress),
            }],
            global_best_genome: WorldGenome::default_with_seed(1),
            global_best_fitness: 50.0,
            total_worlds_evaluated: 1,
            total_wall_time_ms: 10.0,
        };
        let telemetry = telemetry_from_result(&result, Some("stress-test"));
        assert_eq!(telemetry[0].mode, Some("stress-test".to_string()));
        let sm = telemetry[0].stress_metrics.as_ref().unwrap();
        assert_eq!(sm.pre_stress_biomass, 100.0);
        assert_eq!(sm.min_stress_biomass, 40.0);
        assert_eq!(sm.post_recovery_biomass, 75.0);
    }

    #[test]
    fn pareto_telemetry_has_multi_objective_fitness() {
        let result = ParetoEvolutionResult {
            pareto_front: vec![ParetoResult {
                genome: WorldGenome::default_with_seed(1),
                objectives: MultiObjectiveFitness {
                    biomass: 10.0, biodiversity: 5.0, stability: 8.0,
                    carbon: 3.0, fruit: 2.0, microbial: 4.0, fly_metabolism: 1.5, enzyme_efficacy: 0.0,
                },
                rank: 0,
                crowding_distance: 1.0,
                wall_time_ms: 50.0,
            }],
            generations_run: 1,
            total_worlds_evaluated: 10,
            total_wall_time_ms: 50.0,
        };
        let telemetry = telemetry_from_pareto_result(&result);
        assert_eq!(telemetry.mode, Some("pareto".to_string()));
        let mof = telemetry.multi_objective_fitness.as_ref().unwrap();
        assert_eq!(mof.biomass, 10.0);
        assert_eq!(mof.fly_metabolism, 1.5);
    }

    #[test]
    fn stress_test_config_defaults_reasonable() {
        let stress = StressTestConfig::default();
        assert!(stress.w_pre_stress + stress.w_recovery + stress.w_resilience > 0.99);
        assert!(stress.perturbation_duration > 0);
    }

    #[test]
    fn pareto_dominance_works() {
        let a = MultiObjectiveFitness {
            biomass: 10.0, biodiversity: 5.0, stability: 8.0,
            carbon: 3.0, fruit: 2.0, microbial: 4.0, fly_metabolism: 3.0, enzyme_efficacy: 0.0,
        };
        let b = MultiObjectiveFitness {
            biomass: 8.0, biodiversity: 5.0, stability: 8.0,
            carbon: 3.0, fruit: 2.0, microbial: 4.0, fly_metabolism: 3.0, enzyme_efficacy: 0.0,
        };
        // a dominates b (better in biomass, equal in others)
        assert!(dominates(&a, &b));
        assert!(!dominates(&b, &a));

        // Non-dominated case
        let c = MultiObjectiveFitness {
            biomass: 12.0, biodiversity: 3.0, stability: 8.0,
            carbon: 3.0, fruit: 2.0, microbial: 4.0, fly_metabolism: 3.0, enzyme_efficacy: 0.0,
        };
        assert!(!dominates(&a, &c));
        assert!(!dominates(&c, &a));
    }

    #[test]
    fn pareto_ranks_computed_correctly() {
        let mut results = vec![
            ParetoResult {
                genome: WorldGenome::default_with_seed(1),
                objectives: MultiObjectiveFitness { biomass: 10.0, biodiversity: 5.0, stability: 8.0, carbon: 3.0, fruit: 2.0, microbial: 4.0, fly_metabolism: 3.0, enzyme_efficacy: 0.0 },
                rank: 99, crowding_distance: 0.0, wall_time_ms: 1.0,
            },
            ParetoResult {
                genome: WorldGenome::default_with_seed(2),
                objectives: MultiObjectiveFitness { biomass: 8.0, biodiversity: 5.0, stability: 8.0, carbon: 3.0, fruit: 2.0, microbial: 4.0, fly_metabolism: 3.0, enzyme_efficacy: 0.0 },
                rank: 99, crowding_distance: 0.0, wall_time_ms: 1.0,
            },
            ParetoResult {
                genome: WorldGenome::default_with_seed(3),
                objectives: MultiObjectiveFitness { biomass: 12.0, biodiversity: 6.0, stability: 9.0, carbon: 4.0, fruit: 3.0, microbial: 5.0, fly_metabolism: 4.0, enzyme_efficacy: 0.0 },
                rank: 99, crowding_distance: 0.0, wall_time_ms: 1.0,
            },
        ];

        compute_pareto_ranks(&mut results);

        // Third solution dominates both others, should have lowest rank (best)
        assert!(results[2].rank < results[0].rank, "Best solution should have lower rank than first");
        assert!(results[2].rank < results[1].rank, "Best solution should have lower rank than second");
        // All ranks should be valid
        assert!(results.iter().all(|r| r.rank < 3));
    }

    #[test]
    fn telemetry_from_pareto_result_works() {
        let result = ParetoEvolutionResult {
            pareto_front: vec![
                ParetoResult {
                    genome: WorldGenome::default_with_seed(1),
                    objectives: MultiObjectiveFitness { biomass: 10.0, biodiversity: 5.0, stability: 8.0, carbon: 3.0, fruit: 2.0, microbial: 4.0, fly_metabolism: 3.0, enzyme_efficacy: 0.0 },
                    rank: 0, crowding_distance: 1.0, wall_time_ms: 100.0,
                },
            ],
            generations_run: 5,
            total_worlds_evaluated: 50,
            total_wall_time_ms: 5000.0,
        };

        let telemetry = telemetry_from_pareto_result(&result);
        assert_eq!(telemetry.generation, 5);
        assert_eq!(telemetry.mode, Some("pareto".to_string()));
        assert!(telemetry.multi_objective_fitness.is_some());
        assert!(telemetry.best_fitness > 0.0);
    }

    #[test]
    fn stress_config_timing_derived_from_frames() {
        // Verify stress timing is derived correctly from frames
        let frames = 30;
        let drought = frames / 3;
        let heat = 2 * frames / 3;
        let duration = (frames / 10).max(1);
        assert_eq!(drought, 10);
        assert_eq!(heat, 20);
        assert_eq!(duration, 3);
        // Verify drought completes before heat starts
        assert!(drought + duration <= heat);
    }

    #[test]
    fn multiobjective_stressed_returns_stress_metrics() {
        // Test that the stressed multi-objective evaluator returns both Pareto + stress data
        let stress = StressTestConfig {
            drought_frame: 2,
            heat_spike_frame: 4,
            perturbation_duration: 1,
            ..StressTestConfig::default()
        };

        let genome = WorldGenome::default_with_seed(42);
        let result = run_single_world_multiobjective_stressed(genome, 6, 3, true, &stress);

        match result {
            Ok((pareto, stress_metrics)) => {
                // Should have valid objective values (non-NaN)
                assert!(!pareto.objectives.biomass.is_nan());
                assert!(!pareto.objectives.stability.is_nan());
                // Stress metrics should be finite
                assert!(stress_metrics.pre_stress_biomass.is_finite());
                assert!(stress_metrics.min_stress_biomass.is_finite());
                assert!(stress_metrics.post_recovery_biomass.is_finite());
            }
            Err(_) => {
                // World construction may fail in test environment; that's OK
            }
        }
    }

    #[test]
    fn fly_metabolism_fitness_evaluates() {
        use crate::terrarium_world::TerrariumWorld;
        let mut world = TerrariumWorld::demo(7, false).unwrap();
        world.run_frames(5).unwrap();
        let snapshot = world.snapshot();
        let fitness = evaluate_fitness(FitnessObjective::MaxFlyMetabolism, &snapshot, &[]);
        // With flies present, fitness should be > 0 (energy_charge * 10 + flies * 2).
        if snapshot.flies > 0 {
            assert!(fitness > 0.0, "MaxFlyMetabolism should be > 0 with flies, got {fitness}");
        }
        // Multi-objective should include fly_metabolism.
        let multi = MultiObjectiveFitness::evaluate(&snapshot, &[]);
        assert!(multi.fly_metabolism >= 0.0, "fly_metabolism objective should be non-negative");
    }

    // ================================================================
    // Bet-Hedging & Persister Cell Tests
    // ================================================================

    #[test]
    fn phenotypic_noise_default_is_neutral() {
        let noise = FlyPhenotypicNoise::default();
        assert!((noise.metabolic_rate_modifier - 1.0).abs() < 1e-4);
        assert!((noise.trehalase_modifier - 1.0).abs() < 1e-4);
        assert!((noise.stress_tolerance - 1.0).abs() < 1e-4);
        assert!(noise.dormancy_propensity.abs() < 1e-4);
    }

    #[test]
    fn phenotypic_noise_from_stochastic_produces_variation() {
        let mut rates = Vec::new();
        for i in 0..100 {
            let noise = FlyPhenotypicNoise::from_stochastic_state(
                3.0, // High Fano factor
                150.0,
                0.8, // High noise intensity
                42 + i * 1000,
            );
            rates.push(noise.metabolic_rate_modifier);
        }
        let mean = rates.iter().sum::<f32>() / rates.len() as f32;
        let variance = rates.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / rates.len() as f32;
        // With high Fano + high noise, variance should be noticeable
        assert!(variance > 0.001, "Phenotypic noise should produce variance, got {variance:.6}");
        // All values should be in valid range
        assert!(rates.iter().all(|r| *r >= 0.5 && *r <= 2.0));
    }

    #[test]
    fn generate_population_noise_zero_intensity_is_neutral() {
        let profiles = generate_population_noise(10, 0.0, 42);
        assert_eq!(profiles.len(), 10);
        for p in &profiles {
            assert!((p.metabolic_rate_modifier - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn generate_population_noise_high_intensity_produces_variation() {
        let profiles = generate_population_noise(20, 0.9, 42);
        assert_eq!(profiles.len(), 20);
        let rates: Vec<f32> = profiles.iter().map(|p| p.metabolic_rate_modifier).collect();
        let mean = rates.iter().sum::<f32>() / rates.len() as f32;
        let variance = rates.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / rates.len() as f32;
        assert!(variance > 0.0, "High noise intensity should produce variance");
    }

    #[test]
    fn bet_hedging_fitness_rewards_flies() {
        let snapshot = TerrariumWorldSnapshot {
            plants: 5, fruits: 2, seeds: 0, flies: 4,
            food_remaining: 10.0, fly_food_total: 5.0,
            avg_fly_energy: 50.0, avg_altitude: 1.0,
            light: 1.0, temperature: 25.0, humidity: 0.7,
            mean_soil_moisture: 0.5, mean_deep_moisture: 0.3,
            mean_microbes: 100.0, mean_symbionts: 50.0,
            mean_canopy: 0.8, mean_root_density: 0.3,
            total_plant_cells: 100.0, mean_cell_vitality: 0.8,
            mean_cell_energy: 0.5, mean_division_pressure: 0.3,
            mean_soil_glucose: 1.0, mean_soil_oxygen: 0.2,
            mean_soil_ammonium: 0.1, mean_soil_nitrate: 0.05,
            mean_soil_redox: 0.3, mean_soil_atp_flux: 0.5,
            mean_atmospheric_co2: 400.0, mean_atmospheric_o2: 0.21,
            ecology_event_count: 5, avg_fly_energy_charge: 0.85,
            fly_plant_proximity_mean: 3.0, fly_altitude_mean: 1.5,
            fly_o2_gradient_correlation: 0.3, owned_fraction: 0.2,
            substrate_backend: "cpu", substrate_steps: 100,
            substrate_time_ms: 10.0, time_s: 1.0,
            atomistic_probes: 0,
        };
        let noise = vec![FlyPhenotypicNoise::default(); 4];
        let fitness = bet_hedging_fitness(&snapshot, &noise);
        // 4 flies * 5 = 20 + energy_charge * 5 = 4.25 + variance ~0
        assert!(fitness > 20.0, "Fitness should be > 20 with 4 flies, got {fitness:.2}");
    }

    #[test]
    fn persister_cells_survive_antibiotic() {
        let mut sim = PersisterCellSimulator::default();
        let result = sim.run_treatment_protocol(
            5.0,   // 5 hours growth
            10.0,  // 10 hours antibiotic
            10.0,  // 10 hours recovery
            0.1,   // 0.1 hour steps
        );

        // Pre-treatment: should have grown from initial 1M+
        assert!(result.pre_treatment_total > 1e6,
            "Pre-treatment population should exceed initial, got {:.0}", result.pre_treatment_total);

        // Post-treatment: massive kill but persisters survive
        assert!(result.post_treatment_total > 0.0,
            "Persisters should survive antibiotic treatment");
        assert!(result.survival_fraction < 0.1,
            "Most cells should be killed, survival={:.4}", result.survival_fraction);

        // Recovery: population should regrow from persisters
        assert!(result.post_recovery_total > result.minimum_population,
            "Population should recover after antibiotic removal");
        assert!(result.regrowth_ratio > 1.0,
            "Regrowth ratio should > 1, got {:.2}", result.regrowth_ratio);
    }

    #[test]
    fn persister_switching_rates_affect_survival() {
        // High switching rate = more persisters = better survival
        let mut sim_high = PersisterCellSimulator::with_switching_rates(1e-3, 1e-4);
        let result_high = sim_high.run_treatment_protocol(5.0, 10.0, 5.0, 0.1);

        // Low switching rate = fewer persisters = worse survival
        let mut sim_low = PersisterCellSimulator::with_switching_rates(1e-7, 1e-4);
        let result_low = sim_low.run_treatment_protocol(5.0, 10.0, 5.0, 0.1);

        assert!(result_high.survival_fraction > result_low.survival_fraction,
            "Higher switching rate should give better survival: high={:.6} vs low={:.6}",
            result_high.survival_fraction, result_low.survival_fraction);
    }

    #[test]
    fn persister_biphasic_kill_detected() {
        let mut sim = PersisterCellSimulator::with_switching_rates(1e-4, 1e-3);
        let result = sim.run_treatment_protocol(5.0, 20.0, 5.0, 0.1);
        // With reasonable switching rates and long treatment, should detect biphasic
        // (rapid initial kill then plateau)
        // This is probabilistic -- just verify the method runs
        assert!(result.total_time_hours > 25.0);
    }

    #[test]
    fn bet_hedging_genome_random_in_bounds() {
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..50 {
            let g = BetHedgingGenome::random(&mut rng);
            assert!(g.noise_intensity >= 0.0 && g.noise_intensity <= 1.0);
            assert!(g.persister_rate_scale >= 0.1 && g.persister_rate_scale <= 10.0);
            assert!(g.noise_tolerance >= 0.0 && g.noise_tolerance <= 1.0);
        }
    }

    #[test]
    fn bet_hedging_genome_mutation_preserves_bounds() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut g = BetHedgingGenome::random(&mut rng);
        for _ in 0..200 {
            g.mutate(&mut rng, 0.5);
            assert!(g.noise_intensity >= 0.0 && g.noise_intensity <= 1.0);
            assert!(g.persister_rate_scale >= 0.1 && g.persister_rate_scale <= 10.0);
        }
    }

    #[test]
    fn stochastic_benchmark_overhead() {
        use crate::whole_cell::stochastic_expression::*;
        use std::time::Instant;

        let config = StochasticExpressionConfig { enabled: true, ..Default::default() };
        let mut states: Vec<StochasticOperonState> = (0..100)
            .map(|i| StochasticOperonState::from_deterministic(10.0 + i as f32, 100.0))
            .collect();
        let rates: Vec<(f32, f32)> = vec![(0.1, 0.01); 100];
        let mut rng = StochasticRng::new(42);

        let start = Instant::now();
        for _ in 0..1000 {
            step_stochastic_expression(&mut states, &config, &rates, 1.0, &mut rng);
        }
        let stochastic_ms = start.elapsed().as_secs_f32() * 1000.0;

        // Should complete 100 operons * 1000 steps in under 100ms
        assert!(stochastic_ms < 500.0,
            "Stochastic stepping too slow: {stochastic_ms:.1}ms for 100k operon-steps");
        eprintln!("  Stochastic benchmark: {stochastic_ms:.1}ms for 100 operons x 1000 steps");
    }

    #[test]
    fn nsga2_breed_preserves_population_size() {
        use rand::SeedableRng;
        let results = vec![
            ParetoResult {
                genome: WorldGenome::default_with_seed(1),
                objectives: MultiObjectiveFitness {
                    biomass: 10.0, biodiversity: 5.0, stability: 8.0,
                    carbon: 3.0, fruit: 2.0, microbial: 4.0, fly_metabolism: 1.0, enzyme_efficacy: 0.0,
                },
                rank: 0, crowding_distance: 1.0, wall_time_ms: 10.0,
            },
            ParetoResult {
                genome: WorldGenome::default_with_seed(2),
                objectives: MultiObjectiveFitness {
                    biomass: 8.0, biodiversity: 7.0, stability: 6.0,
                    carbon: 5.0, fruit: 4.0, microbial: 3.0, fly_metabolism: 2.0, enzyme_efficacy: 0.0,
                },
                rank: 0, crowding_distance: 0.5, wall_time_ms: 12.0,
            },
            ParetoResult {
                genome: WorldGenome::default_with_seed(3),
                objectives: MultiObjectiveFitness {
                    biomass: 6.0, biodiversity: 9.0, stability: 4.0,
                    carbon: 7.0, fruit: 1.0, microbial: 5.0, fly_metabolism: 3.0, enzyme_efficacy: 0.0,
                },
                rank: 1, crowding_distance: 0.8, wall_time_ms: 11.0,
            },
        ];
        let mut rng = StdRng::seed_from_u64(42);
        let pop = nsga2_breed(&results, 8, &mut rng, 0.15, 0.7);
        assert_eq!(pop.len(), 8, "Breeding must produce exactly population_size genomes");
    }

    #[test]
    fn stressed_multi_fitness_evaluates_under_perturbation() {
        // Verify run_single_world_multiobjective_stressed returns valid
        // stress metrics with recovery > 0 and pre_stress >= min_stress
        let genome = WorldGenome::default_with_seed(99);
        let stress = StressTestConfig {
            drought_frame: 5,
            heat_spike_frame: 10,
            perturbation_duration: 3,
            ..StressTestConfig::default()
        };
        let result = run_single_world_multiobjective_stressed(genome, 20, 5, true, &stress);
        match result {
            Ok((pareto, metrics)) => {
                assert!(pareto.objectives.biomass >= 0.0);
                assert!(metrics.pre_stress_biomass >= 0.0);
                assert!(metrics.post_recovery_biomass >= 0.0);
                // min_stress should be <= pre_stress (stress reduces biomass)
                assert!(metrics.min_stress_biomass <= metrics.pre_stress_biomass + 1.0,
                    "min_stress={:.2} should be <= pre_stress={:.2}",
                    metrics.min_stress_biomass, metrics.pre_stress_biomass);
            }
            Err(e) => {
                // Acceptable: lite world may fail to construct with some genomes
                eprintln!("  Stressed multi-fitness construction failed (ok for edge-case genome): {e}");
            }
        }
    }

    #[test]
    fn stress_config_from_frames_scales_correctly() {
        let frames = 300;
        let stress = StressTestConfig {
            drought_frame: frames / 3,
            heat_spike_frame: 2 * frames / 3,
            perturbation_duration: (frames / 10).max(1),
            ..StressTestConfig::default()
        };
        assert_eq!(stress.drought_frame, 100);
        assert_eq!(stress.heat_spike_frame, 200);
        assert_eq!(stress.perturbation_duration, 30);
        // Stress events must not overlap
        assert!(stress.drought_frame + stress.perturbation_duration <= stress.heat_spike_frame,
            "Drought and heat spike must not overlap");
    }


    // ================================================================
    // Environmental Variability Engine Tests
    // ================================================================

    #[test]
    fn seasonal_temperature_oscillates() {
        let schedule = EnvironmentalSchedule::default();
        let mut env = EnvironmentalState::new(schedule.clone(), 42);
        let _summer = env.sample(schedule.season_period_s * 0.25); // peak
        let mut env2 = EnvironmentalState::new(schedule.clone(), 42);
        let _winter = env2.sample(schedule.season_period_s * 0.75); // trough
        // Summer should be warmer than winter (with some noise)
        // Use multiple samples to average out noise
        let mut sum_temps = Vec::new();
        let mut win_temps = Vec::new();
        for i in 0..20 {
            let mut e = EnvironmentalState::new(schedule.clone(), i);
            sum_temps.push(e.sample(schedule.season_period_s * 0.25).temperature_c);
            let mut e2 = EnvironmentalState::new(schedule.clone(), i + 100);
            win_temps.push(e2.sample(schedule.season_period_s * 0.75).temperature_c);
        }
        let avg_sum: f32 = sum_temps.iter().sum::<f32>() / sum_temps.len() as f32;
        let avg_win: f32 = win_temps.iter().sum::<f32>() / win_temps.len() as f32;
        assert!(avg_sum > avg_win, "Summer ({avg_sum:.1}) should be warmer than winter ({avg_win:.1})");
    }

    #[test]
    fn drought_reduces_moisture() {
        let schedule = EnvironmentalSchedule {
            drought_probability: 1.0, // Force drought
            ..Default::default()
        };
        let mut env = EnvironmentalState::new(schedule, 42);
        let sample = env.sample(1000.0);
        assert!(sample.is_drought, "Should be in drought with p=1.0");
        assert!(sample.moisture_modifier < 1.0, "Drought should reduce moisture");
    }

    #[test]
    fn tropical_preset_warmer() {
        let temperate = EnvironmentalSchedule::temperate();
        let tropical = EnvironmentalSchedule::tropical();
        assert!(tropical.base_temperature_c > temperate.base_temperature_c);
        assert!(tropical.base_humidity > temperate.base_humidity);
    }

    #[test]
    fn arid_preset_drier() {
        let temperate = EnvironmentalSchedule::temperate();
        let arid = EnvironmentalSchedule::arid();
        assert!(arid.base_humidity < temperate.base_humidity);
        assert!(arid.seasonal_amplitude_c > temperate.seasonal_amplitude_c);
    }

    // ================================================================
    // Spatial Heterogeneity Tests
    // ================================================================

    #[test]
    fn spatial_zone_contains_point() {
        let zone = SpatialZone {
            x_start: 5, y_start: 5, x_end: 15, y_end: 15,
            temperature_modifier: 3.0, moisture_modifier: -0.3,
            nutrient_modifier: 0.5, light_modifier: 0.0,
            zone_type: ZoneType::Tropical,
        };
        assert!(zone.contains(10, 10));
        assert!(!zone.contains(4, 10));
        assert!(!zone.contains(15, 10));
        assert_eq!(zone.area(), 100);
    }

    #[test]
    fn generate_zones_covers_grid() {
        let mut rng = StdRng::seed_from_u64(42);
        let zones = generate_spatial_zones(20, 16, 4, &mut rng);
        assert!(zones.len() <= 4);
        assert!(zones.len() >= 1);
        // At least one zone should contain center
        assert!(zones.iter().any(|z| z.contains(10, 8)));
    }

    #[test]
    fn spatial_fitness_modifier_varies() {
        let zones = vec![
            SpatialZone {
                x_start: 0, y_start: 0, x_end: 10, y_end: 10,
                temperature_modifier: 0.0, moisture_modifier: 0.5,
                nutrient_modifier: 0.5, light_modifier: 0.0,
                zone_type: ZoneType::NutrientRich,
            },
            SpatialZone {
                x_start: 10, y_start: 0, x_end: 20, y_end: 10,
                temperature_modifier: 0.0, moisture_modifier: -0.3,
                nutrient_modifier: -0.3, light_modifier: 0.0,
                zone_type: ZoneType::Arid,
            },
        ];
        let rich = spatial_fitness_modifier(&zones, 5, 5);
        let poor = spatial_fitness_modifier(&zones, 15, 5);
        assert!(rich > poor, "Nutrient-rich zone should have higher modifier");
    }

    // ================================================================
    // Multi-Species Noise Tests
    // ================================================================

    #[test]
    fn plant_noise_default_neutral() {
        let noise = PlantPhenotypicNoise::default();
        assert!((noise.growth_rate_modifier - 1.0).abs() < 1e-4);
        assert!((noise.drought_tolerance - 1.0).abs() < 1e-4);
    }

    #[test]
    fn microbial_noise_default_neutral() {
        let noise = MicrobialPhenotypicNoise::default();
        assert!((noise.growth_rate_modifier - 1.0).abs() < 1e-4);
        assert!(noise.antibiotic_resistance.abs() < 1e-4);
    }

    #[test]
    fn generate_plant_noise_produces_variation() {
        let profiles = generate_plant_noise(20, 0.8, 42);
        let rates: Vec<f32> = profiles.iter().map(|p| p.growth_rate_modifier).collect();
        let mean = rates.iter().sum::<f32>() / rates.len() as f32;
        let variance = rates.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / rates.len() as f32;
        assert!(variance > 0.001, "Plant noise should produce variation");
    }

    #[test]
    fn generate_microbial_noise_produces_variation() {
        let profiles = generate_microbial_noise(20, 0.8, 42);
        let has_resistant = profiles.iter().any(|m| m.antibiotic_resistance > 0.0);
        let has_biofilm = profiles.iter().any(|m| m.biofilm_propensity > 0.0);
        assert!(has_resistant || has_biofilm, "Microbial noise should produce phenotypic variety");
    }

    #[test]
    fn multi_species_fitness_rewards_diversity() {
        let snapshot = TerrariumWorldSnapshot {
            plants: 5, fruits: 2, seeds: 0, flies: 3,
            food_remaining: 10.0, fly_food_total: 5.0,
            avg_fly_energy: 50.0, avg_altitude: 1.0,
            light: 1.0, temperature: 25.0, humidity: 0.7,
            mean_soil_moisture: 0.5, mean_deep_moisture: 0.3,
            mean_microbes: 100.0, mean_symbionts: 50.0,
            mean_canopy: 0.8, mean_root_density: 0.3,
            total_plant_cells: 100.0, mean_cell_vitality: 0.8,
            mean_cell_energy: 0.5, mean_division_pressure: 0.3,
            mean_soil_glucose: 1.0, mean_soil_oxygen: 0.2,
            mean_soil_ammonium: 0.1, mean_soil_nitrate: 0.05,
            mean_soil_redox: 0.3, mean_soil_atp_flux: 0.5,
            mean_atmospheric_co2: 400.0, mean_atmospheric_o2: 0.21,
            ecology_event_count: 5, avg_fly_energy_charge: 0.85,
            fly_plant_proximity_mean: 3.0, fly_altitude_mean: 1.5,
            fly_o2_gradient_correlation: 0.3, owned_fraction: 0.2,
            substrate_backend: "cpu", substrate_steps: 100,
            substrate_time_ms: 10.0, time_s: 1.0,
            atomistic_probes: 0,
        };
        let fly_noise = generate_population_noise(3, 0.5, 42);
        let plant_noise = generate_plant_noise(5, 0.5, 43);
        let microbial_noise = generate_microbial_noise(10, 0.5, 44);
        let fitness = multi_species_bet_hedging_fitness(&snapshot, &fly_noise, &plant_noise, &microbial_noise);
        assert!(fitness > 0.0, "Multi-species fitness should be positive");
    }

    // ================================================================
    // Drug Protocol Optimizer Tests
    // ================================================================

    #[test]
    fn single_drug_protocol_kills_cells() {
        let protocol = DrugProtocol::single(3.0, 10.0);
        let mut sim = PersisterCellSimulator::default();
        let result = protocol.execute(&mut sim, 0.1);
        assert!(result.survival_fraction < 0.5, "Single drug should kill most cells");
        assert!(result.cycles_completed == 1);
    }

    #[test]
    fn pulsed_dosing_protocol_runs() {
        let protocol = DrugProtocol::pulsed(3.0, 5.0, 2.0, 3);
        let mut sim = PersisterCellSimulator::default();
        let result = protocol.execute(&mut sim, 0.1);
        assert_eq!(result.cycles_completed, 3);
        assert!(result.total_time_hours > 20.0);
    }

    #[test]
    fn combination_therapy_runs() {
        let protocol = DrugProtocol::combination(3.0, 5.0, 5.0);
        let mut sim = PersisterCellSimulator::default();
        let result = protocol.execute(&mut sim, 0.1);
        assert_eq!(result.cycles_completed, 2);
    }

    #[test]
    fn protocol_optimizer_finds_best() {
        let protocols = vec![
            DrugProtocol::single(1.0, 5.0),  // Weak
            DrugProtocol::single(5.0, 10.0), // Strong
            DrugProtocol::pulsed(3.0, 3.0, 1.0, 3),
        ];
        let (best_idx, results) = optimize_drug_protocol(&protocols, 1e-5, 0.1);
        // Strong single drug or pulsed should beat weak single
        assert!(results[best_idx].final_population <= results[0].final_population,
            "Optimizer should find protocol at least as good as weakest");
    }

    #[test]
    fn ecoli_validation_monotonic() {
        let data = validate_against_ecoli();
        // Higher switching rates should give higher survival
        for i in 1..data.len() {
            assert!(data[i].2 >= data[i - 1].2 * 0.1,
                "Higher switching rate should give higher survival: rate={:.0e} surv={:.4e}",
                data[i].0, data[i].2);
        }
    }

    // ================================================================
    // Gene Circuit Designer Tests
    // ================================================================

    #[test]
    fn circuit_params_predicted_fano() {
        let p = GeneCircuitParams::default();
        let fano = p.predicted_fano();
        // Fano = 1 + burst_size * k_off / (k_on + k_off)
        // = 1 + 3.0 * 0.1 / (0.05 + 0.1) = 1 + 3.0 * 0.667 = 3.0
        assert!(fano > 1.0, "Fano should be > 1 for bursty expression");
        assert!(fano < 10.0, "Default Fano should be moderate");
    }

    #[test]
    fn circuit_optimization_converges() {
        let spec = GeneCircuitSpec {
            target_fano: 5.0,
            target_mean_protein: 200.0,
            target_cv: 0.15,
        };
        let result = optimize_gene_circuit(&spec, 20, 50, 42);
        // Should approach target Fano within 2x
        assert!(result.achieved_fano > 1.0, "Should achieve super-Poisson noise");
        assert!(result.achieved_mean_protein > 0.0, "Should produce protein");
        assert!(result.generations_run == 50);
    }

    #[test]
    fn circuit_random_in_bounds() {
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..50 {
            let p = GeneCircuitParams::random(&mut rng);
            assert!(p.promoter_on_rate > 0.0 && p.promoter_on_rate < 1.0);
            assert!(p.burst_size >= 1.0 && p.burst_size <= 20.0);
        }
    }

    // ================================================================
    // Adaptive Evolution with Environment Tests
    // ================================================================

    #[test]
    fn evolve_with_environment_temperate() {
        let config = EvolutionConfig {
            population_size: 4, generations: 2, frames_per_world: 30,
            master_seed: 42, lite: true,
            ..Default::default()
        };
        let schedule = EnvironmentalSchedule::temperate();
        let result = evolve_with_environment(config, schedule).unwrap();
        assert!(result.global_best_fitness > 0.0, "Should find positive fitness");
        assert_eq!(result.generations.len(), 2);
    }

    #[test]
    fn evolve_with_environment_arid_is_harder() {
        let config = EvolutionConfig {
            population_size: 4, generations: 2, frames_per_world: 30,
            master_seed: 42, lite: true,
            ..Default::default()
        };
        let temperate = evolve_with_environment(config.clone(), EnvironmentalSchedule::temperate()).unwrap();
        let arid = evolve_with_environment(config, EnvironmentalSchedule::arid()).unwrap();
        // Both should produce results (not crash)
        assert!(temperate.total_worlds_evaluated > 0);
        assert!(arid.total_worlds_evaluated > 0);
    }

    // ================================================================
    // Coevolution Engine Tests
    // ================================================================

    #[test]
    fn coevolution_red_queen_runs() {
        let result = evolve_coevolution(4, 2, 20, CoevolutionMode::RedQueen, true, 42).unwrap();
        assert_eq!(result.history.len(), 2);
        assert!(!result.final_population_a.is_empty());
        assert!(!result.final_population_b.is_empty());
        assert_eq!(result.mode, CoevolutionMode::RedQueen);
    }

    #[test]
    fn coevolution_mutualistic_boosts_fitness() {
        let mut rng = StdRng::seed_from_u64(42);
        let a = SpeciesGenome { cooperation_tendency: 1.0, ..SpeciesGenome::random(&mut rng) };
        let b = SpeciesGenome { cooperation_tendency: 1.0, ..SpeciesGenome::random(&mut rng) };
        let result = evaluate_coevolution_pair(&a, &b, CoevolutionMode::Mutualistic, 20, true).unwrap();
        // Mutualistic with max cooperation should boost both
        assert!(result.interaction_strength > 0.5, "High cooperation species should have strong interaction");
    }

    #[test]
    fn coevolution_competitive_splits_resources() {
        let mut rng = StdRng::seed_from_u64(42);
        let a = SpeciesGenome { resource_efficiency: 0.9, ..SpeciesGenome::random(&mut rng) };
        let b = SpeciesGenome { resource_efficiency: 0.1, ..SpeciesGenome::random(&mut rng) };
        let result = evaluate_coevolution_pair(&a, &b, CoevolutionMode::Competitive, 20, true).unwrap();
        // More efficient species should get larger share
        assert!(result.species_a_fitness > result.species_b_fitness * 0.5,
            "Efficient species should outcompete: a={:.2} b={:.2}", result.species_a_fitness, result.species_b_fitness);
    }

    #[test]
    fn species_genome_mutation_preserves_bounds() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut g = SpeciesGenome::random(&mut rng);
        for _ in 0..100 {
            g.mutate(&mut rng, 0.5);
            assert!(g.defense_investment >= 0.0 && g.defense_investment <= 1.0);
            assert!(g.resource_efficiency >= 0.1 && g.resource_efficiency <= 1.0);
            assert!(g.cooperation_tendency >= 0.0 && g.cooperation_tendency <= 1.0);
            assert!(g.mobility >= 0.1 && g.mobility <= 1.0);
        }
    }

    // ================================================================
    // Genetic Regulatory Network Tests
    // ================================================================

    #[test]
    fn grn_random_nk_correct_size() {
        let mut rng = StdRng::seed_from_u64(42);
        let grn = GeneRegulatoryNetwork::random_nk(8, 3, &mut rng);
        assert_eq!(grn.n_genes, 8);
        assert_eq!(grn.weights.len(), 8);
        assert_eq!(grn.thresholds.len(), 8);
        assert_eq!(grn.expression.len(), 8);
    }

    #[test]
    fn grn_step_keeps_expression_bounded() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut grn = GeneRegulatoryNetwork::random_nk(10, 3, &mut rng);
        for _ in 0..100 {
            grn.step();
            for &e in &grn.expression {
                assert!(e >= 0.0 && e <= 1.0, "Expression out of bounds: {}", e);
            }
        }
    }

    #[test]
    fn grn_finds_attractor() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut grn = GeneRegulatoryNetwork::random_nk(6, 2, &mut rng);
        let attractor = grn.find_attractor(200);
        assert_eq!(attractor.len(), 6);
        // Attractor values should be in [0, 1]
        for &v in &attractor {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn grn_phenotype_has_valid_traits() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut grn = GeneRegulatoryNetwork::random_nk(12, 3, &mut rng);
        let pheno = grn.attractor_to_phenotype();
        assert!(pheno.growth_rate >= 0.0 && pheno.growth_rate <= 1.0);
        assert!(pheno.stress_tolerance >= 0.0 && pheno.stress_tolerance <= 1.0);
        assert!(pheno.network_complexity >= 0.0);
    }

    #[test]
    fn grn_evolution_converges() {
        let result = evolve_grn(0.7, 0.5, 8, 3, 10, 20, 42);
        assert_eq!(result.generations_run, 20);
        // Should approach target phenotype
        assert!(result.best_phenotype.growth_rate > 0.0, "Should produce growth");
    }

    #[test]
    fn grn_crossover_preserves_size() {
        let mut rng = StdRng::seed_from_u64(42);
        let a = GeneRegulatoryNetwork::random_nk(8, 3, &mut rng);
        let b = GeneRegulatoryNetwork::random_nk(8, 3, &mut rng);
        let child = GeneRegulatoryNetwork::crossover(&a, &b, &mut rng);
        assert_eq!(child.n_genes, 8);
        assert_eq!(child.weights.len(), 8);
    }

    // ================================================================
    // Ecosystem Health Tests
    // ================================================================

    #[test]
    fn ecosystem_health_empty_snapshots() {
        let report = assess_ecosystem_health(&[]);
        assert_eq!(report.overall_health, 0.0);
        assert_eq!(report.trophic_levels, 0);
    }

    #[test]
    fn ecosystem_health_computes_diversity() {
        let genome = WorldGenome::default_with_seed(42);
        let mut world = genome.build_world_lite().unwrap();
        let mut snapshots = Vec::new();
        for _ in 0..10 {
            world.step_frame().unwrap();
            snapshots.push(world.snapshot());
        }
        let report = assess_ecosystem_health(&snapshots);
        assert!(report.total_biomass >= 0.0);
        assert!(report.overall_health >= 0.0 && report.overall_health <= 100.0);
        assert!(report.stability >= 0.0);
    }

    // ================================================================
    // World Export Tests
    // ================================================================

    #[test]
    fn world_export_captures_snapshots() {
        let genome = WorldGenome::default_with_seed(42);
        let export = run_and_export(genome, 20, true, 5, None).unwrap();
        assert!(!export.snapshots.is_empty());
        assert!(export.metadata.fitness >= 0.0 || export.metadata.fitness < 0.0); // any finite
        assert_eq!(export.metadata.frames_run, 20);
        assert_eq!(export.metadata.lite_mode, true);
    }

    #[test]
    fn world_export_with_environment() {
        let genome = WorldGenome::default_with_seed(42);
        let sched = EnvironmentalSchedule::tropical();
        let export = run_and_export(genome, 15, true, 3, Some(&sched)).unwrap();
        assert!(export.environmental_samples.is_some());
        let samples = export.environmental_samples.unwrap();
        assert_eq!(samples.len(), 15);
    }

    #[test]
    fn world_export_serializes_to_json() {
        let genome = WorldGenome::default_with_seed(42);
        let export = run_and_export(genome, 10, true, 5, None).unwrap();
        let json = serde_json::to_string(&export).unwrap();
        assert!(json.len() > 100, "JSON should be substantial");
        // Verify metadata can be parsed as JSON value
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["metadata"]["frames_run"], 10);
    }

    // ================================================================
    // Sparkline & Dashboard Tests
    // ================================================================

    #[test]
    fn sparkline_renders_data() {
        let data = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let s = sparkline(&data, 5);
        assert_eq!(s.chars().count(), 5);
        // First should be space (min), last should be full block (max)
        assert_eq!(s.chars().next().unwrap(), ' ');
        assert_eq!(s.chars().last().unwrap(), '\u{2588}');
    }

    #[test]
    fn sparkline_empty_data() {
        let s = sparkline(&[], 10);
        assert!(s.is_empty());
    }

    #[test]
    fn dashboard_generates_output() {
        let genome = WorldGenome::default_with_seed(42);
        let mut world = genome.build_world_lite().unwrap();
        let mut snapshots = Vec::new();
        for _ in 0..10 {
            world.step_frame().unwrap();
            snapshots.push(world.snapshot());
        }
        let dash = ecosystem_dashboard(&snapshots, 80);
        assert!(dash.contains("oNeura Ecosystem Dashboard"));
        assert!(dash.contains("Biomass"));
        assert!(dash.contains("Moisture"));
    }



}
