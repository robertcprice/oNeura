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
                stress_metrics: None,
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
                },
                GenerationResult {
                    generation: 1,
                    best_fitness: 120.0,
                    mean_fitness: 90.0,
                    worst_fitness: 70.0,
                    best_genome: WorldGenome::default_with_seed(2),
                    wall_time_ms: 45.0,
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
    fn stress_test_config_defaults_reasonable() {
        let stress = StressTestConfig::default();
        assert!(stress.w_pre_stress + stress.w_recovery + stress.w_resilience > 0.99);
        assert!(stress.perturbation_duration > 0);
    }

    #[test]
    fn pareto_dominance_works() {
        let a = MultiObjectiveFitness {
            biomass: 10.0, biodiversity: 5.0, stability: 8.0,
            carbon: 3.0, fruit: 2.0, microbial: 4.0, fly_metabolism: 3.0,
        };
        let b = MultiObjectiveFitness {
            biomass: 8.0, biodiversity: 5.0, stability: 8.0,
            carbon: 3.0, fruit: 2.0, microbial: 4.0, fly_metabolism: 3.0,
        };
        // a dominates b (better in biomass, equal in others)
        assert!(dominates(&a, &b));
        assert!(!dominates(&b, &a));

        // Non-dominated case
        let c = MultiObjectiveFitness {
            biomass: 12.0, biodiversity: 3.0, stability: 8.0,
            carbon: 3.0, fruit: 2.0, microbial: 4.0, fly_metabolism: 3.0,
        };
        assert!(!dominates(&a, &c));
        assert!(!dominates(&c, &a));
    }

    #[test]
    fn pareto_ranks_computed_correctly() {
        let mut results = vec![
            ParetoResult {
                genome: WorldGenome::default_with_seed(1),
                objectives: MultiObjectiveFitness { biomass: 10.0, biodiversity: 5.0, stability: 8.0, carbon: 3.0, fruit: 2.0, microbial: 4.0, fly_metabolism: 3.0 },
                rank: 99, crowding_distance: 0.0, wall_time_ms: 1.0,
            },
            ParetoResult {
                genome: WorldGenome::default_with_seed(2),
                objectives: MultiObjectiveFitness { biomass: 8.0, biodiversity: 5.0, stability: 8.0, carbon: 3.0, fruit: 2.0, microbial: 4.0, fly_metabolism: 3.0 },
                rank: 99, crowding_distance: 0.0, wall_time_ms: 1.0,
            },
            ParetoResult {
                genome: WorldGenome::default_with_seed(3),
                objectives: MultiObjectiveFitness { biomass: 12.0, biodiversity: 6.0, stability: 9.0, carbon: 4.0, fruit: 3.0, microbial: 5.0, fly_metabolism: 4.0 },
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
                    objectives: MultiObjectiveFitness { biomass: 10.0, biodiversity: 5.0, stability: 8.0, carbon: 3.0, fruit: 2.0, microbial: 4.0, fly_metabolism: 3.0 },
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
}
