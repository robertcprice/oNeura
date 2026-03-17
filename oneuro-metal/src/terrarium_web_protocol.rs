//! WebSocket message protocol for the terrarium web app.
//!
//! Defines all client→server commands and server→client messages as serde enums.

use crate::terrarium_evolve::{
    EvolutionConfig, FitnessConfig, FitnessObjective, GenomeConstraints,
    SearchStrategy, WorldGenome,
};
use crate::terrarium_world::{TerrariumTopdownView, TerrariumWorldSnapshot};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Client → Server
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum ClientMsg {
    Play,
    Pause,
    Step,
    Reset {
        #[serde(default = "default_seed")]
        seed: u64,
    },
    Speed {
        fps: u32,
    },
    View {
        mode: String,
    },
    AddPlant {
        x: usize,
        y: usize,
    },
    AddFly {
        x: f32,
        y: f32,
    },
    AddFruit {
        x: usize,
        y: usize,
    },
    AddWater {
        x: usize,
        y: usize,
    },
    EvolveStart {
        config: EvolveWebConfig,
    },
    EvolveStop,
    EvolveApplyGenome {
        genome: WorldGenome,
    },
    SensitivityStart {
        config: SensitivityWebConfig,
    },
    StressStart {
        config: StressWebConfig,
    },
    EcosystemStart {
        scenario: String,
        #[serde(default = "default_seed")]
        seed: u64,
    },
    EcosystemStep,
    EcosystemRun {
        #[serde(default = "default_eco_days")]
        days: f64,
    },
}

fn default_seed() -> u64 {
    42
}

fn default_eco_days() -> f64 {
    30.0
}

/// Simplified evolution config from the browser (maps to EvolutionConfig).
#[derive(Debug, Clone, Deserialize)]
pub struct EvolveWebConfig {
    #[serde(default = "default_mode")]
    pub mode: String,
    #[serde(default = "default_population")]
    pub population: usize,
    #[serde(default = "default_generations")]
    pub generations: usize,
    #[serde(default = "default_frames")]
    pub frames: usize,
    #[serde(default = "default_fitness_str")]
    pub fitness: String,
    #[serde(default)]
    pub lite: bool,
    #[serde(default = "default_mutation_rate")]
    pub mutation_rate: f32,
    #[serde(default)]
    pub no_flies: bool,
    #[serde(default)]
    pub max_microbes: Option<usize>,
}

fn default_mode() -> String {
    "standard".into()
}
fn default_population() -> usize {
    8
}
fn default_generations() -> usize {
    5
}
fn default_frames() -> usize {
    50
}
fn default_fitness_str() -> String {
    "biomass".into()
}
fn default_mutation_rate() -> f32 {
    0.15
}

/// Config for a sensitivity sweep from the browser.
#[derive(Debug, Clone, Deserialize)]
pub struct SensitivityWebConfig {
    /// Sweep a single param (default: all 10 continuous).
    #[serde(default)]
    pub param: Option<String>,
    /// Resolution (sweep points per param). Default 5, max 20.
    #[serde(default)]
    pub resolution: Option<usize>,
    /// Frames per evaluation. Default 30.
    #[serde(default)]
    pub frames: Option<usize>,
    /// Random seed.
    #[serde(default)]
    pub seed: Option<u64>,
}

/// Config for a stress benchmark from the browser.
#[derive(Debug, Clone, Deserialize)]
pub struct StressWebConfig {
    /// Which scenarios to run (default: all 6).
    #[serde(default)]
    pub scenarios: Vec<String>,
    /// Total frames per scenario. Default 60.
    #[serde(default)]
    pub frames: Option<usize>,
    /// Random seed.
    #[serde(default)]
    pub seed: Option<u64>,
}

impl EvolveWebConfig {
    /// Convert browser config to internal EvolutionConfig.
    pub fn to_evolution_config(&self) -> EvolutionConfig {
        let fitness_objective = match self.fitness.as_str() {
            "biodiversity" => FitnessObjective::MaxBiodiversity,
            "stability" => FitnessObjective::MaxStability,
            "carbon" => FitnessObjective::MaxCarbonSequestration,
            "fruit" => FitnessObjective::MaxFruitProduction,
            "microbial" => FitnessObjective::MaxMicrobialHealth,
            "fly" => FitnessObjective::MaxFlyEcosystem,
            _ => FitnessObjective::MaxBiomass,
        };

        let mut constraints = GenomeConstraints::default();
        if self.no_flies {
            constraints.max_fly_count = Some(0);
        }
        if let Some(max) = self.max_microbes {
            constraints.max_microbe_count = Some(max);
        }

        EvolutionConfig {
            population_size: self.population,
            generations: self.generations,
            frames_per_world: self.frames,
            strategy: SearchStrategy::Evolutionary {
                tournament_size: 3,
                mutation_rate: self.mutation_rate,
                crossover_rate: 0.7,
                elitism: 2,
            },
            fitness: FitnessConfig {
                primary: fitness_objective,
                snapshot_interval: 10,
            },
            thread_count: None,
            master_seed: 42,
            constraints,
            lite: self.lite,
        }
    }
}

// ---------------------------------------------------------------------------
// Server → Client
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMsg {
    Frame(FrameData),
    Snapshot(SnapshotData),
    Entities(EntityData),
    EvolveGeneration(EvolveGenerationData),
    EvolveComplete(EvolveCompleteData),
    TournamentUpdate(crate::terrarium_web_tournament::TournamentUpdateData),
    SensitivityProgress(SensitivityProgressData),
    SensitivityComplete(SensitivityCompleteData),
    StressProgress(StressProgressData),
    StressComplete(StressCompleteData),
    EcosystemSnapshot(crate::ecosystem_integration::EcosystemSnapshot),
    EcosystemTimeSeries(crate::ecosystem_integration::EcosystemTimeSeries),
    Error {
        message: String,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct FrameData {
    pub field: Vec<f32>,
    pub width: usize,
    pub height: usize,
    pub view: String,
    pub daylight: f32,
    pub time_label: String,
    pub paused: bool,
    pub fps: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct SnapshotData {
    #[serde(flatten)]
    pub snapshot: TerrariumWorldSnapshot,
}

#[derive(Debug, Clone, Serialize)]
pub struct EntityData {
    pub plants: Vec<EntityPos>,
    pub flies: Vec<FlyEntity>,
    pub fruits: Vec<EntityPos>,
    pub waters: Vec<EntityPos>,
    pub seeds: Vec<EntityPos>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub ecology_events: Vec<EcologyEventData>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EcologyEventData {
    pub event_type: String,
    pub x: f32,
    pub y: f32,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct EntityPos {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct FlyEntity {
    pub x: f32,
    pub y: f32,
    pub heading: f32,
    pub energy_frac: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvolveGenerationData {
    pub generation: usize,
    pub best_fitness: f32,
    pub mean_fitness: f32,
    pub worst_fitness: f32,
    pub diversity: f32,
    pub best_genome: WorldGenome,
    pub mode: String,
    /// Pareto front for multi-objective modes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pareto_front: Option<Vec<ParetoFrontEntry>>,
    /// Stress metrics for stress-test modes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stress_metrics: Option<StressMetricsData>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StressMetricsData {
    pub pre_stress_biomass: f32,
    pub min_biomass: f32,
    pub recovery_biomass: f32,
    pub recovery_ratio: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct ParetoFrontEntry {
    pub biomass: f32,
    pub biodiversity: f32,
    pub stability: f32,
    pub carbon: f32,
    pub fruit: f32,
    pub microbial: f32,
    pub rank: usize,
}

impl From<&crate::terrarium_evolve::ParetoResult> for ParetoFrontEntry {
    fn from(r: &crate::terrarium_evolve::ParetoResult) -> Self {
        Self {
            biomass: r.objectives.biomass,
            biodiversity: r.objectives.biodiversity,
            stability: r.objectives.stability,
            carbon: r.objectives.carbon,
            fruit: r.objectives.fruit,
            microbial: r.objectives.microbial,
            rank: r.rank,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct EvolveCompleteData {
    pub mode: String,
    pub best_genome: WorldGenome,
    pub best_fitness: f32,
    pub total_worlds: usize,
    pub total_time_ms: f32,
}

// ---------------------------------------------------------------------------
// Sensitivity / Stress data structs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct SensitivityProgressData {
    pub parameter: String,
    pub sensitivity_index: f32,
    pub min_fitness: f32,
    pub max_fitness: f32,
    pub mean_fitness: f32,
    pub sweep_values: Vec<f32>,
    pub fitness_values: Vec<f32>,
    pub param_index: usize,
    pub total_params: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SensitivityCompleteData {
    pub total_params: usize,
    pub elapsed_ms: f64,
    pub rankings: Vec<(String, f32)>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StressProgressData {
    pub scenario: String,
    pub description: String,
    pub baseline_biomass: f32,
    pub min_biomass: f32,
    pub final_biomass: f32,
    pub recovery_ratio: f32,
    pub frames_to_min: usize,
    pub elapsed_ms: f64,
    pub scenario_index: usize,
    pub total_scenarios: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct StressCompleteData {
    pub total_scenarios: usize,
    pub elapsed_ms: f64,
}

/// Parse a view mode string into a TerrariumTopdownView.
pub fn parse_view(s: &str) -> TerrariumTopdownView {
    match s {
        "moisture" | "soil_moisture" => TerrariumTopdownView::SoilMoisture,
        "canopy" => TerrariumTopdownView::Canopy,
        "chemistry" => TerrariumTopdownView::Chemistry,
        "odor" => TerrariumTopdownView::Odor,
        "gas" | "gas_exchange" => TerrariumTopdownView::GasExchange,
        _ => TerrariumTopdownView::Terrain,
    }
}
