//! WebSocket message protocol for the terrarium web app.
//!
//! Defines all client→server commands and server→client messages as serde enums.

use crate::terrarium_evolve::{
    EvolutionConfig, FitnessConfig, FitnessObjective, GenomeConstraints, SearchStrategy,
    WorldGenome,
};
use crate::terrarium_world::OrganismRegistryEntry;
use crate::terrarium_world::{
    TerrariumAtmosphereFrame, TerrariumTopdownView, TerrariumWorldSnapshot,
};
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
        #[serde(default)]
        preset: Option<String>,
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
    /// Switch climate scenario at runtime (rcp85, rcp45, rcp26, preindustrial, none).
    SetClimate {
        scenario: String,
        #[serde(default)]
        seed: Option<u64>,
    },
    /// Adjust visual emergence blend (0.0 = legacy hardcoded, 1.0 = fully emergent).
    SetVisualBlend {
        blend: f32,
    },
    /// Manually trigger an extreme weather event in the live simulation.
    TriggerExtremeEvent {
        event_type: String,
        #[serde(default)]
        severity: Option<f32>,
    },
    /// Set time scale factor: 1.0 = base rate, 2.0 = double speed, 0.5 = half speed.
    /// Controls how many simulation steps run per rendered frame.
    SetTimeScale {
        scale: f32,
    },
    // -----------------------------------------------------------------------
    // Pharma Lab commands
    // -----------------------------------------------------------------------
    /// Enter pharma lab mode (allocates PharmaLab state).
    PharmaEnter,
    /// Exit pharma lab mode (deallocates PharmaLab state).
    PharmaExit,
    /// Add a single atom to the lab bench.
    PharmaAddAtom {
        element: String,
        position: [f32; 3],
    },
    /// Add a bond between two atoms in the same molecule.
    PharmaAddBond {
        molecule_id: u64,
        atom_a: usize,
        atom_b: usize,
        #[serde(default = "default_bond_order")]
        order: String,
    },
    /// Remove an atom from a molecule.
    PharmaRemoveAtom {
        molecule_id: u64,
        atom_idx: usize,
    },
    /// Remove a bond from a molecule.
    PharmaRemoveBond {
        molecule_id: u64,
        bond_idx: usize,
    },
    /// Remove an entire molecule.
    PharmaRemoveMolecule {
        molecule_id: u64,
    },
    /// Parse a SMILES string and add the resulting molecule.
    PharmaParseSmiles {
        smiles: String,
    },
    /// Load a molecule from the built-in library.
    PharmaLoadLibrary {
        name: String,
    },
    /// Merge two molecules (after drawing a bond between them).
    PharmaMergeMolecules {
        molecule_a: u64,
        molecule_b: u64,
    },
    /// Start MD simulation.
    PharmaMdStart,
    /// Stop MD simulation.
    PharmaMdStop,
    /// Step MD simulation by N steps.
    PharmaMdStep {
        #[serde(default = "default_md_steps")]
        steps: u32,
    },
    /// Set thermostat temperature.
    PharmaSetTemperature {
        kelvin: f32,
    },
    /// Dock a molecule against a named target.
    PharmaDock {
        ligand_id: u64,
        target: String,
    },
    /// Compute ADMET profile for a molecule.
    PharmaAdmet {
        molecule_id: u64,
    },
}

fn default_seed() -> u64 {
    42
}

fn default_eco_days() -> f64 {
    30.0
}
fn default_bond_order() -> String {
    "single".into()
}
fn default_md_steps() -> u32 {
    100
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
    SnapshotHistory(SnapshotHistoryData),
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
    Error { message: String },
}

#[derive(Debug, Clone, Serialize)]
pub struct FrameData {
    pub field: Vec<f32>,
    pub width: usize,
    pub height: usize,
    pub view: String,
    pub atmosphere: TerrariumAtmosphereFrame,
    pub terrain_surface: Vec<f32>,
    pub terrain_visuals: Vec<crate::terrarium::visual_projection::TerrariumTerrainVisualResponse>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub terrain_voxels: Vec<crate::terrarium::visual_projection::TerrariumTerrainVoxelBatch>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub terrain_cutaways: Vec<TerrainCutawayProfile>,
    pub daylight: f32,
    /// Solar elevation above horizon (radians). From compute_solar_state().
    pub sun_elevation_rad: f32,
    /// Solar azimuth (radians, 0=north clockwise). From compute_solar_state().
    pub sun_azimuth_rad: f32,
    /// Unit vector toward the sun [x, y, z]. From compute_solar_state().
    pub sun_direction: [f32; 3],
    pub time_label: String,
    pub paused: bool,
    pub fps: u32,
    /// Per-cell soil moisture [0,1] for shader DataTextures.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub moisture: Vec<f32>,
    /// Per-cell water presence [0,1] for shader DataTextures.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub water_mask: Vec<f32>,
    /// Per-cell soil structure (0=sand, 1=clay) for shader DataTextures.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub soil_structure: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SnapshotData {
    #[serde(flatten)]
    pub snapshot: TerrariumWorldSnapshot,
    pub preset: String,
    pub seed: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub field: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub view: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub atmosphere: Option<TerrariumAtmosphereFrame>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub terrain_surface: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub terrain_visuals:
        Option<Vec<crate::terrarium::visual_projection::TerrariumTerrainVisualResponse>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub terrain_voxels:
        Option<Vec<crate::terrarium::visual_projection::TerrariumTerrainVoxelBatch>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub terrain_cutaways: Option<Vec<TerrainCutawayProfile>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entities: Option<EntityData>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub daylight: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub paused: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fps: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotHistoryData {
    pub history: Vec<TerrariumSnapshotHistoryPoint>,
    pub preset: String,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganismRegistryData {
    pub entries: Vec<OrganismRegistryEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganismLineageData {
    pub organism_id: u64,
    pub lineage: Vec<OrganismRegistryEntry>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RenameOrganismRequest {
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenameOrganismResponse {
    pub organism_id: u64,
    pub entry: OrganismRegistryEntry,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrariumSnapshotHistoryPoint {
    pub time_s: f32,
    pub preset: String,
    pub seed: u64,
    pub adults: u32,
    pub eggs: u32,
    pub embryos: u32,
    pub larvae: u32,
    pub pupae: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embryo_viability: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embryo_glucose: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embryo_nucleotides: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct InspectData {
    pub kind: String,
    pub title: String,
    pub subtitle: String,
    pub preset: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub position: Option<InspectPosition>,
    pub summary: Vec<InspectMetric>,
    pub scene: Vec<InspectMetric>,
    pub cellular: Vec<InspectMetric>,
    pub molecular: Vec<InspectMetric>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scene_grid: Option<InspectGrid>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cellular_grid: Option<InspectGrid>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub molecular_grid: Option<InspectGrid>,
    pub composition: Vec<InspectComposition>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct InspectPosition {
    pub x: f32,
    pub y: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub z: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct InspectMetric {
    pub label: String,
    pub value: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fraction: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct InspectComposition {
    pub label: String,
    pub amount: f32,
    pub unit: String,
    pub category: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct InspectGrid {
    pub label: String,
    pub width: usize,
    pub height: usize,
    pub palette: String,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainCutawayProfile {
    pub x: usize,
    pub y: usize,
    pub face: String,
    pub top: f32,
    pub bottom: f32,
    pub layers: Vec<TerrainCutawayLayer>,
    pub pockets: Vec<TerrainCutawayPocket>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainCutawayLayer {
    pub material_class: String,
    pub texture_descriptor: String,
    pub y: f32,
    pub height: f32,
    pub rgb: [f32; 3],
    pub texture: f32,
    pub organic: f32,
    pub water: f32,
    pub oxygen_gas: f32,
    pub carbon_dioxide: f32,
    pub ammonium: f32,
    pub nitrate: f32,
    pub amino_acids: f32,
    pub nucleotides: f32,
    pub membrane_precursors: f32,
    pub silicate_mineral: f32,
    pub clay_mineral: f32,
    pub carbonate_mineral: f32,
    pub iron_oxide_mineral: f32,
    pub dissolved_silicate: f32,
    pub bicarbonate: f32,
    #[serde(default)]
    pub surface_proton_load: f32,
    #[serde(default)]
    pub calcium_bicarbonate_complex: f32,
    #[serde(default)]
    pub sorbed_aluminum_hydroxide: f32,
    #[serde(default)]
    pub sorbed_ferric_hydroxide: f32,
    pub exchangeable_calcium: f32,
    pub exchangeable_magnesium: f32,
    pub exchangeable_potassium: f32,
    pub exchangeable_sodium: f32,
    pub exchangeable_aluminum: f32,
    pub aqueous_iron: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainCutawayPocket {
    pub kind: String,
    pub y: f32,
    pub lateral_t: f32,
    pub scale_x: f32,
    pub scale_y: f32,
    pub scale_z: f32,
    pub rgb: [f32; 3],
    pub tilt: f32,
    pub opacity: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct EntityData {
    pub plants: Vec<EntityPos>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub full_plants: Vec<crate::terrarium_world::TerrariumPlantSnapshot>,
    pub plant_visuals: Vec<crate::terrarium::visual_projection::TerrariumPlantVisualResponse>,
    pub flies: Vec<FlyEntity>,
    pub fly_eggs: Vec<FlyEggEntity>,
    pub fly_embryos: Vec<FlyEmbryoEntity>,
    pub fly_larvae: Vec<FlyImmatureEntity>,
    pub fly_pupae: Vec<FlyImmatureEntity>,
    pub fly_visuals: Vec<crate::terrarium::visual_projection::TerrariumFlyVisualResponse>,
    pub earthworms: Vec<EarthwormMarker>,
    pub nematodes: Vec<NematodeMarker>,
    pub fruits: Vec<EntityPos>,
    pub full_fruits: Vec<crate::terrarium_world::TerrariumFruitSnapshot>,
    pub fruit_visuals: Vec<crate::terrarium::visual_projection::TerrariumFruitVisualResponse>,
    pub waters: Vec<EntityPos>,
    pub water_visuals: Vec<crate::terrarium::visual_projection::TerrariumWaterVisualResponse>,
    pub seeds: Vec<EntityPos>,
    pub full_seeds: Vec<crate::terrarium_world::TerrariumSeedSnapshot>,
    pub seed_visuals: Vec<crate::terrarium::visual_projection::TerrariumSeedVisualResponse>,
    pub soil_surface: Vec<SoilSurfaceMarker>,
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
pub struct SoilSurfaceMarker {
    pub x: f32,
    pub y: f32,
    pub visual: crate::terrarium::visual_projection::TerrariumSoilSurfaceVisualResponse,
}

#[derive(Debug, Clone, Serialize)]
pub struct EarthwormMarker {
    pub x: f32,
    pub y: f32,
    pub visual: crate::terrarium::visual_projection::TerrariumEarthwormVisualResponse,
}

#[derive(Debug, Clone, Serialize)]
pub struct NematodeMarker {
    pub x: f32,
    pub y: f32,
    pub visual: crate::terrarium::visual_projection::TerrariumNematodeVisualResponse,
}

#[derive(Debug, Clone, Serialize)]
pub struct FlyEntity {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub heading: f32,
    pub is_flying: bool,
    pub wing_beat_freq: f32,
    pub energy_frac: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct FlyEggEntity {
    pub x: f32,
    pub y: f32,
    pub count: u8,
    pub age_hours: f32,
    pub substrate_quality: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct FlyEmbryoEntity {
    pub id: u32,
    pub x: f32,
    pub y: f32,
    pub age_hours: f32,
    pub viability: f32,
    pub sex: String,
    pub cluster_index: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct FlyImmatureEntity {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub age_hours: f32,
    pub energy_frac: f32,
    pub sex: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instar: Option<u8>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_climate_deserializes() {
        let json = r#"{"cmd":"set_climate","scenario":"rcp85"}"#;
        let msg: ClientMsg = serde_json::from_str(json).expect("parse SetClimate");
        match msg {
            ClientMsg::SetClimate { scenario, seed } => {
                assert_eq!(scenario, "rcp85");
                assert!(seed.is_none());
            }
            other => panic!("Expected SetClimate, got: {other:?}"),
        }
    }

    #[test]
    fn set_visual_blend_deserializes() {
        let json = r#"{"cmd":"set_visual_blend","blend":0.75}"#;
        let msg: ClientMsg = serde_json::from_str(json).expect("parse SetVisualBlend");
        match msg {
            ClientMsg::SetVisualBlend { blend } => {
                assert!((blend - 0.75).abs() < 1e-5);
            }
            other => panic!("Expected SetVisualBlend, got: {other:?}"),
        }
    }

    #[test]
    fn trigger_extreme_event_deserializes() {
        let json = r#"{"cmd":"trigger_extreme_event","event_type":"drought","severity":0.8}"#;
        let msg: ClientMsg = serde_json::from_str(json).expect("parse TriggerExtremeEvent");
        match msg {
            ClientMsg::TriggerExtremeEvent {
                event_type,
                severity,
            } => {
                assert_eq!(event_type, "drought");
                assert!((severity.unwrap() - 0.8).abs() < 1e-5);
            }
            other => panic!("Expected TriggerExtremeEvent, got: {other:?}"),
        }
    }
}
