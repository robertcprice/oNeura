//! Native terrarium world owner.
//!
//! This is the next step past "Rust kernels under a Python shell": a reusable
//! Rust world that owns the terrarium state and advances substrate chemistry,
//! broad soil pools, plant physiology, food/seed bookkeeping, atmosphere, and
//! fly stepping without Python orchestration.

use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::StandardNormal;

use crate::constants::clamp;
use crate::drosophila::{DrosophilaScale, DrosophilaSim};
use crate::ecology_events::{step_food_patches, step_seed_bank};
use crate::ecology_fields::build_dual_radial_fields;
use crate::molecular_atmosphere::{
    odorant_channel_params, step_molecular_world_fields, FruitSourceState, OdorantChannelParams,
    PlantSourceState, WaterSourceState,
};
use crate::plant_cellular::PlantCellularStateSim;
use crate::plant_organism::PlantOrganismSim;
use crate::soil_broad::step_soil_broad_pools;
use crate::soil_uptake::extract_root_resources_with_layers;
use crate::terrarium::{BatchedAtomTerrarium, TerrariumSpecies};
use crate::terrarium_field::TerrariumSensoryField;

const DAY_LENGTH_S: f32 = 86_400.0;
const ETHYL_ACETATE_IDX: usize = 0;
const GERANIOL_IDX: usize = 1;
const AMMONIA_IDX: usize = 2;

fn idx2(width: usize, x: usize, y: usize) -> usize {
    y * width + x
}

fn idx3(width: usize, height: usize, x: usize, y: usize, z: usize) -> usize {
    (z * height + y) * width + x
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().copied().sum::<f32>() / values.len() as f32
    }
}

fn sample_normal(rng: &mut StdRng, sigma: f32) -> f32 {
    let z: f32 = rng.sample(StandardNormal);
    z * sigma
}

fn offset_clamped(value: usize, delta: isize, upper_exclusive: usize) -> usize {
    let shifted = value as isize + delta;
    shifted.clamp(0, upper_exclusive.saturating_sub(1) as isize) as usize
}

fn temp_response(temp_c: f32, optimum: f32, width: f32) -> f32 {
    let delta = (temp_c - optimum) / width.max(1e-6);
    (-delta * delta).exp()
}

fn deposit_2d(
    field: &mut [f32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    radius: usize,
    amount: f32,
) {
    if amount.abs() <= 1.0e-12 {
        return;
    }
    let radius = radius.max(1);
    let sigma = (radius as f32 * 0.72).max(0.75);
    let denom = 2.0 * sigma * sigma;
    let y0 = y.saturating_sub(radius);
    let y1 = (y + radius + 1).min(height);
    let x0 = x.saturating_sub(radius);
    let x1 = (x + radius + 1).min(width);
    let mut kernel_total = 0.0f32;

    for yy in y0..y1 {
        for xx in x0..x1 {
            let dx = xx as f32 - x as f32;
            let dy = yy as f32 - y as f32;
            kernel_total += (-(dx * dx + dy * dy) / denom).exp();
        }
    }
    if kernel_total <= 1.0e-9 {
        return;
    }
    let scale = amount / kernel_total;
    for yy in y0..y1 {
        for xx in x0..x1 {
            let dx = xx as f32 - x as f32;
            let dy = yy as f32 - y as f32;
            let kernel = (-(dx * dx + dy * dy) / denom).exp();
            field[idx2(width, xx, yy)] += kernel * scale;
        }
    }
}

fn layer_mean_map(
    width: usize,
    height: usize,
    depth: usize,
    field: &[f32],
    z0: usize,
    z1: usize,
) -> Vec<f32> {
    let plane = width * height;
    let total_z = depth.max(1);
    let start = z0.min(total_z - 1);
    let end = z1.max(start + 1).min(total_z);
    let count = (end - start) as f32;
    let mut out = vec![0.0f32; plane];
    for z in start..end {
        let slice = &field[z * plane..(z + 1) * plane];
        for (dst, src) in out.iter_mut().zip(slice.iter().copied()) {
            *dst += src;
        }
    }
    for value in &mut out {
        *value /= count;
    }
    out
}

#[derive(Debug, Clone)]
pub struct TerrariumWorldConfig {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub cell_size_mm: f32,
    pub seed: u64,
    pub use_gpu_substrate: bool,
    pub world_dt_s: f32,
    pub substrate_dt_ms: f32,
    pub substeps: usize,
    pub time_warp: f32,
    pub max_plants: usize,
    pub max_fruits: usize,
    pub max_seeds: usize,
}

impl Default for TerrariumWorldConfig {
    fn default() -> Self {
        Self {
            width: 44,
            height: 32,
            depth: 4,
            cell_size_mm: 0.5,
            seed: 7,
            use_gpu_substrate: true,
            world_dt_s: 0.05,
            substrate_dt_ms: 15.0,
            substeps: 2,
            time_warp: 900.0,
            max_plants: 28,
            max_fruits: 64,
            max_seeds: 96,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TerrariumPlantGenome {
    pub max_height_mm: f32,
    pub canopy_radius_mm: f32,
    pub root_radius_mm: f32,
    pub leaf_efficiency: f32,
    pub root_uptake_efficiency: f32,
    pub water_use_efficiency: f32,
    pub volatile_scale: f32,
    pub fruiting_threshold: f32,
    pub litter_turnover: f32,
    pub shade_tolerance: f32,
    pub root_depth_bias: f32,
    pub symbiosis_affinity: f32,
    pub seed_mass: f32,
}

impl TerrariumPlantGenome {
    pub fn sample(rng: &mut StdRng) -> Self {
        Self {
            max_height_mm: rng.gen_range(7.0..18.0),
            canopy_radius_mm: rng.gen_range(3.0..8.0),
            root_radius_mm: rng.gen_range(2.0..5.5),
            leaf_efficiency: rng.gen_range(0.7..1.4),
            root_uptake_efficiency: rng.gen_range(0.6..1.3),
            water_use_efficiency: rng.gen_range(0.6..1.25),
            volatile_scale: rng.gen_range(0.6..1.5),
            fruiting_threshold: rng.gen_range(0.45..1.2),
            litter_turnover: rng.gen_range(0.7..1.4),
            shade_tolerance: rng.gen_range(0.55..1.4),
            root_depth_bias: rng.gen_range(0.15..0.95),
            symbiosis_affinity: rng.gen_range(0.6..1.5),
            seed_mass: rng.gen_range(0.045..0.16),
        }
    }

    pub fn mutate(&self, rng: &mut StdRng) -> Self {
        Self {
            max_height_mm: clamp(self.max_height_mm + sample_normal(rng, 0.85), 6.5, 20.0),
            canopy_radius_mm: clamp(self.canopy_radius_mm + sample_normal(rng, 0.35), 2.5, 9.5),
            root_radius_mm: clamp(self.root_radius_mm + sample_normal(rng, 0.30), 1.8, 7.0),
            leaf_efficiency: clamp(self.leaf_efficiency + sample_normal(rng, 0.06), 0.55, 1.65),
            root_uptake_efficiency: clamp(
                self.root_uptake_efficiency + sample_normal(rng, 0.05),
                0.45,
                1.6,
            ),
            water_use_efficiency: clamp(
                self.water_use_efficiency + sample_normal(rng, 0.05),
                0.45,
                1.5,
            ),
            volatile_scale: clamp(self.volatile_scale + sample_normal(rng, 0.08), 0.45, 1.8),
            fruiting_threshold: clamp(
                self.fruiting_threshold + sample_normal(rng, 0.06),
                0.3,
                1.5,
            ),
            litter_turnover: clamp(self.litter_turnover + sample_normal(rng, 0.06), 0.45, 1.8),
            shade_tolerance: clamp(self.shade_tolerance + sample_normal(rng, 0.05), 0.4, 1.7),
            root_depth_bias: clamp(self.root_depth_bias + sample_normal(rng, 0.04), 0.05, 1.1),
            symbiosis_affinity: clamp(
                self.symbiosis_affinity + sample_normal(rng, 0.05),
                0.35,
                1.8,
            ),
            seed_mass: clamp(self.seed_mass + sample_normal(rng, 0.01), 0.03, 0.20),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TerrariumSeed {
    pub x: f32,
    pub y: f32,
    pub dormancy_s: f32,
    pub reserve_carbon: f32,
    pub age_s: f32,
    pub genome: TerrariumPlantGenome,
}

#[derive(Debug, Clone)]
pub struct TerrariumFruitPatch {
    pub source: FruitSourceState,
    pub radius: f32,
    pub previous_remaining: f32,
    pub deposited_all: bool,
}

#[derive(Debug, Clone)]
pub struct TerrariumPlant {
    pub x: usize,
    pub y: usize,
    pub genome: TerrariumPlantGenome,
    pub physiology: PlantOrganismSim,
    pub cellular: PlantCellularStateSim,
}

impl TerrariumPlant {
    pub fn new(
        x: usize,
        y: usize,
        genome: TerrariumPlantGenome,
        biomass_scale: f32,
        rng: &mut StdRng,
    ) -> Self {
        let fruit_timer_s = rng.gen_range(7200.0..16000.0);
        let seed_timer_s = rng.gen_range(14000.0..30000.0);
        let physiology = PlantOrganismSim::new(
            genome.max_height_mm,
            genome.canopy_radius_mm,
            genome.root_radius_mm,
            genome.leaf_efficiency,
            genome.root_uptake_efficiency,
            genome.water_use_efficiency,
            genome.volatile_scale,
            genome.fruiting_threshold,
            genome.litter_turnover,
            genome.shade_tolerance,
            genome.root_depth_bias,
            genome.symbiosis_affinity,
            genome.seed_mass,
            0.14 * biomass_scale,
            0.18 * biomass_scale,
            0.17 * biomass_scale,
            0.12 * biomass_scale,
            0.22 + genome.seed_mass * 0.4,
            0.11 + genome.seed_mass * 0.2,
            fruit_timer_s,
            seed_timer_s,
        );
        let leaf_cells = 160.0 * biomass_scale * (0.88 + genome.leaf_efficiency * 0.22);
        let stem_cells = 120.0 * biomass_scale * (0.84 + genome.max_height_mm / 24.0);
        let root_cells = 145.0 * biomass_scale * (0.78 + genome.root_depth_bias * 0.55);
        let meristem_cells = 70.0 * biomass_scale * (0.85 + genome.seed_mass * 3.8);
        let cellular = PlantCellularStateSim::new(leaf_cells, stem_cells, root_cells, meristem_cells);
        Self {
            x,
            y,
            genome,
            physiology,
            cellular,
        }
    }

    pub fn canopy_radius_cells(&self) -> usize {
        let leaf_cells = self.cellular.cluster_snapshot(crate::plant_cellular::PlantTissue::Leaf).cell_count;
        (self.genome.canopy_radius_mm + self.physiology.leaf_biomass() * 3.1 + leaf_cells * 0.0022)
            .max(2.0)
            .round() as usize
    }

    pub fn root_radius_cells(&self) -> usize {
        let root_cells = self.cellular.cluster_snapshot(crate::plant_cellular::PlantTissue::Root).cell_count;
        (self.genome.root_radius_mm + self.physiology.root_biomass() * 3.4 + root_cells * 0.0025)
            .max(2.0)
            .round() as usize
    }

    pub fn canopy_amplitude(&self) -> f32 {
        self.physiology.leaf_biomass() * (1.2 + self.physiology.height_mm() * 0.04)
    }

    pub fn root_amplitude(&self) -> f32 {
        self.physiology.root_biomass() * (1.0 + self.genome.root_depth_bias * 0.45)
    }

    pub fn source_state(&self, depth: usize) -> PlantSourceState {
        let emission_z = ((self.physiology.height_mm() / 3.0).round() as usize)
            .clamp(1, depth.saturating_sub(1).max(1));
        PlantSourceState {
            x: self.x,
            y: self.y,
            emission_z,
            odorant_emission_rate: self.physiology.odorant_emission_rate(),
            alive: !self.physiology.is_dead(),
            odorant_profile: vec![
                (GERANIOL_IDX, self.physiology.odorant_geraniol()),
                (ETHYL_ACETATE_IDX, self.physiology.odorant_ethyl_acetate()),
            ],
        }
    }
}

#[derive(Debug, Clone)]
pub struct TerrariumWorldSnapshot {
    pub plants: usize,
    pub fruits: usize,
    pub seeds: usize,
    pub flies: usize,
    pub food_remaining: f32,
    pub fly_food_total: f32,
    pub avg_fly_energy: f32,
    pub avg_altitude: f32,
    pub light: f32,
    pub temperature: f32,
    pub humidity: f32,
    pub mean_soil_moisture: f32,
    pub mean_deep_moisture: f32,
    pub mean_microbes: f32,
    pub mean_symbionts: f32,
    pub mean_canopy: f32,
    pub mean_root_density: f32,
    pub total_plant_cells: f32,
    pub mean_cell_vitality: f32,
    pub mean_cell_energy: f32,
    pub mean_division_pressure: f32,
    pub mean_soil_glucose: f32,
    pub mean_soil_oxygen: f32,
    pub mean_soil_ammonium: f32,
    pub mean_soil_nitrate: f32,
    pub mean_soil_redox: f32,
    pub mean_soil_atp_flux: f32,
    pub substrate_backend: &'static str,
    pub substrate_steps: u64,
    pub substrate_time_ms: f32,
    pub time_s: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerrariumTopdownView {
    Terrain,
    SoilMoisture,
    Canopy,
    Chemistry,
    Odor,
}

impl TerrariumTopdownView {
    pub fn label(self) -> &'static str {
        match self {
            Self::Terrain => "terrain",
            Self::SoilMoisture => "soil",
            Self::Canopy => "canopy",
            Self::Chemistry => "chemistry",
            Self::Odor => "odor",
        }
    }
}

pub struct TerrariumWorld {
    pub config: TerrariumWorldConfig,
    rng: StdRng,
    atmosphere_rng_state: u64,
    pub time_s: f32,
    odorant_params: Vec<OdorantChannelParams>,
    odorants: Vec<Vec<f32>>,
    temperature: Vec<f32>,
    humidity: Vec<f32>,
    wind_x: Vec<f32>,
    wind_y: Vec<f32>,
    wind_z: Vec<f32>,
    pub substrate: BatchedAtomTerrarium,
    pub waters: Vec<WaterSourceState>,
    pub fruits: Vec<TerrariumFruitPatch>,
    pub plants: Vec<TerrariumPlant>,
    pub seeds: Vec<TerrariumSeed>,
    pub flies: Vec<DrosophilaSim>,
    sensory_field: TerrariumSensoryField,
    canopy_cover: Vec<f32>,
    root_density: Vec<f32>,
    water_mask: Vec<f32>,
    moisture: Vec<f32>,
    deep_moisture: Vec<f32>,
    dissolved_nutrients: Vec<f32>,
    mineral_nitrogen: Vec<f32>,
    shallow_nutrients: Vec<f32>,
    deep_minerals: Vec<f32>,
    organic_matter: Vec<f32>,
    litter_carbon: Vec<f32>,
    microbial_biomass: Vec<f32>,
    symbiont_biomass: Vec<f32>,
    root_exudates: Vec<f32>,
    soil_structure: Vec<f32>,
    fly_food_total: f32,
}

impl TerrariumWorld {
    pub fn width(&self) -> usize {
        self.config.width
    }

    pub fn height(&self) -> usize {
        self.config.height
    }

    pub fn new(config: TerrariumWorldConfig) -> Result<Self, String> {
        let mut rng = StdRng::seed_from_u64(config.seed);
        let plane = config.width * config.height;
        let total = plane * config.depth.max(1);
        let substrate = BatchedAtomTerrarium::new(
            config.width,
            config.height,
            config.depth.max(1),
            config.cell_size_mm,
            config.use_gpu_substrate,
        );

        let moisture = (0..plane).map(|_| rng.gen_range(0.18..0.42)).collect::<Vec<_>>();
        let deep_moisture = (0..plane).map(|_| rng.gen_range(0.22..0.36)).collect::<Vec<_>>();
        let dissolved_nutrients = (0..plane).map(|_| rng.gen_range(0.01..0.03)).collect::<Vec<_>>();
        let mineral_nitrogen = (0..plane).map(|_| rng.gen_range(0.015..0.035)).collect::<Vec<_>>();
        let shallow_nutrients = (0..plane).map(|_| rng.gen_range(0.02..0.08)).collect::<Vec<_>>();
        let deep_minerals = (0..plane).map(|_| rng.gen_range(0.07..0.16)).collect::<Vec<_>>();
        let organic_matter = (0..plane).map(|_| rng.gen_range(0.002..0.03)).collect::<Vec<_>>();
        let litter_carbon = (0..plane).map(|_| rng.gen_range(0.006..0.02)).collect::<Vec<_>>();
        let microbial_biomass = (0..plane).map(|_| rng.gen_range(0.01..0.03)).collect::<Vec<_>>();
        let symbiont_biomass = (0..plane).map(|_| rng.gen_range(0.005..0.014)).collect::<Vec<_>>();
        let root_exudates = vec![0.0; plane];
        let soil_structure = (0..plane).map(|_| rng.gen_range(0.35..0.75)).collect::<Vec<_>>();

        Ok(Self {
            atmosphere_rng_state: config.seed ^ 0x9E37_79B9_7F4A_7C15,
            rng,
            time_s: 8.0 * 3600.0,
            odorant_params: vec![
                odorant_channel_params("ethyl_acetate").ok_or("missing ethyl acetate params")?,
                odorant_channel_params("geraniol").ok_or("missing geraniol params")?,
                odorant_channel_params("ammonia").ok_or("missing ammonia params")?,
            ],
            odorants: vec![vec![0.0; total]; 3],
            temperature: vec![22.0; total],
            humidity: vec![0.4; total],
            wind_x: vec![0.0; total],
            wind_y: vec![0.0; total],
            wind_z: vec![0.0; total],
            substrate,
            waters: Vec::new(),
            fruits: Vec::new(),
            plants: Vec::new(),
            seeds: Vec::new(),
            flies: Vec::new(),
            sensory_field: TerrariumSensoryField::new(config.width, config.height, config.depth),
            canopy_cover: vec![0.0; plane],
            root_density: vec![0.0; plane],
            water_mask: vec![0.0; plane],
            moisture,
            deep_moisture,
            dissolved_nutrients,
            mineral_nitrogen,
            shallow_nutrients,
            deep_minerals,
            organic_matter,
            litter_carbon,
            microbial_biomass,
            symbiont_biomass,
            root_exudates,
            soil_structure,
            fly_food_total: 0.0,
            config,
        })
    }

    pub fn demo(seed: u64, use_gpu_substrate: bool) -> Result<Self, String> {
        let mut world = Self::new(TerrariumWorldConfig {
            seed,
            use_gpu_substrate,
            ..TerrariumWorldConfig::default()
        })?;
        world.seed_demo_layout();
        Ok(world)
    }

    pub fn seed_demo_layout(&mut self) {
        self.waters.clear();
        self.fruits.clear();
        self.plants.clear();
        self.seeds.clear();
        self.flies.clear();
        self.fly_food_total = 0.0;
        self.time_s = 8.0 * 3600.0;

        for (x, y, volume) in [(8, 25, 180.0), (21, 14, 110.0), (35, 22, 160.0)] {
            self.add_water(x, y, volume, 0.0008);
        }

        for (x, y) in [(6, 7), (14, 11), (26, 8), (34, 15), (37, 25), (11, 24)] {
            let _ = self.add_plant(x, y, None, None);
        }

        for (x, y) in [(22, 16), (15, 11), (30, 22)] {
            self.add_fruit(x, y, 1.0, None);
        }

        self.add_fly(DrosophilaScale::Tiny, 20.0, 16.0, self.config.seed);
        self.add_fly(DrosophilaScale::Tiny, 24.0, 18.0, self.config.seed + 1);
    }

    pub fn add_water(&mut self, x: usize, y: usize, volume: f32, evaporation_rate: f32) {
        self.waters.push(WaterSourceState {
            x: x.min(self.config.width - 1),
            y: y.min(self.config.height - 1),
            z: 0,
            volume,
            evaporation_rate,
            alive: true,
        });
        deposit_2d(
            &mut self.moisture,
            self.config.width,
            self.config.height,
            x,
            y,
            2,
            0.18,
        );
        deposit_2d(
            &mut self.deep_moisture,
            self.config.width,
            self.config.height,
            x,
            y,
            2,
            0.10,
        );
    }

    pub fn add_fruit(&mut self, x: usize, y: usize, size: f32, volatile_scale: Option<f32>) {
        if self.fruits.len() >= self.config.max_fruits {
            return;
        }
        let volatile_scale = volatile_scale.unwrap_or(1.0);
        let remaining = size.clamp(0.35, 1.5);
        self.fruits.push(TerrariumFruitPatch {
            source: FruitSourceState {
                x: x.min(self.config.width - 1),
                y: y.min(self.config.height - 1),
                z: 0,
                ripeness: 0.88,
                sugar_content: remaining.min(1.0),
                odorant_emission_rate: 0.05 + size * 0.10 * volatile_scale,
                decay_rate: 0.00025 + size * 0.00003,
                alive: true,
                odorant_profile: vec![
                    (ETHYL_ACETATE_IDX, 0.90 * volatile_scale),
                    (AMMONIA_IDX, 0.02),
                ],
            },
            radius: (2.4 + size).clamp(2.0, 4.0),
            previous_remaining: remaining.min(1.0),
            deposited_all: false,
        });
    }

    pub fn add_plant(
        &mut self,
        x: usize,
        y: usize,
        genome: Option<TerrariumPlantGenome>,
        biomass_scale: Option<f32>,
    ) -> usize {
        let genome = genome.unwrap_or_else(|| TerrariumPlantGenome::sample(&mut self.rng));
        let biomass_scale = biomass_scale
            .unwrap_or_else(|| self.rng.gen_range(0.75..1.25) * (0.75 + genome.seed_mass * 4.5));
        let plant = TerrariumPlant::new(
            x.min(self.config.width - 1),
            y.min(self.config.height - 1),
            genome,
            biomass_scale,
            &mut self.rng,
        );
        self.plants.push(plant);
        self.plants.len() - 1
    }

    pub fn add_fly(&mut self, scale: DrosophilaScale, x: f32, y: f32, seed: u64) {
        let mut fly = DrosophilaSim::new(scale, seed);
        let heading = self.rng.gen_range(0.0..std::f32::consts::TAU);
        fly.set_body_state(
            x.clamp(1.0, self.config.width as f32 - 2.0),
            y.clamp(1.0, self.config.height as f32 - 2.0),
            heading,
            Some(0.0),
            Some(0.0),
            Some(false),
            Some(0.0),
            Some(95.0),
            Some(22.0),
            Some(self.time_of_day_hours()),
        );
        self.flies.push(fly);
    }

    pub fn daylight(&self) -> f32 {
        let phase = 2.0 * std::f32::consts::PI * (self.time_s / DAY_LENGTH_S - 0.25);
        phase.sin().max(0.0)
    }

    pub fn time_of_day_hours(&self) -> f32 {
        (self.time_s.rem_euclid(DAY_LENGTH_S) / 3600.0).rem_euclid(24.0)
    }

    pub fn time_label(&self) -> String {
        let seconds = self.time_s.rem_euclid(DAY_LENGTH_S);
        let hours = (seconds / 3600.0).floor() as i32;
        let minutes = ((seconds % 3600.0) / 60.0).floor() as i32;
        format!("{hours:02}:{minutes:02}")
    }

    pub fn topdown_field(&self, view: TerrariumTopdownView) -> Vec<f32> {
        match view {
            TerrariumTopdownView::Terrain => self
                .moisture
                .iter()
                .zip(self.deep_moisture.iter())
                .map(|(surface, deep)| *surface * 0.68 + *deep * 0.32)
                .collect(),
            TerrariumTopdownView::SoilMoisture => self
                .moisture
                .iter()
                .zip(self.deep_moisture.iter())
                .map(|(surface, deep)| *surface + *deep * 0.55)
                .collect(),
            TerrariumTopdownView::Canopy => self
                .canopy_cover
                .iter()
                .zip(self.root_density.iter())
                .map(|(canopy, root)| *canopy * 1.15 + *root * 0.65)
                .collect(),
            TerrariumTopdownView::Chemistry => {
                let plane = self.config.width * self.config.height;
                let depth = self.config.depth.max(1);
                let glucose = self.substrate.species_field(TerrariumSpecies::Glucose);
                let oxygen = self.substrate.species_field(TerrariumSpecies::OxygenGas);
                let ammonium = self.substrate.species_field(TerrariumSpecies::Ammonium);
                let mut field = vec![0.0f32; plane];
                for z in 0..depth {
                    let start = z * plane;
                    let end = start + plane;
                    let g = &glucose[start..end];
                    let o = &oxygen[start..end];
                    let a = &ammonium[start..end];
                    for idx in 0..plane {
                        field[idx] += g[idx] * 0.65 + o[idx] * 0.20 + a[idx] * 0.35;
                    }
                }
                for value in &mut field {
                    *value /= depth as f32;
                }
                field
            }
            TerrariumTopdownView::Odor => {
                let plane = self.config.width * self.config.height;
                let depth = self.config.depth.max(1);
                let odor = &self.odorants[ETHYL_ACETATE_IDX];
                let mut field = vec![0.0f32; plane];
                for z in 0..depth {
                    let start = z * plane;
                    for idx in 0..plane {
                        field[idx] += odor[start + idx];
                    }
                }
                for value in &mut field {
                    *value /= depth as f32;
                }
                field
            }
        }
    }

    fn rebuild_ecology_fields(&mut self) -> Result<(), String> {
        self.canopy_cover.fill(0.0);
        self.root_density.fill(0.0);
        if self.plants.is_empty() {
            return Ok(());
        }
        let mut canopy_sources = Vec::with_capacity(self.plants.len() * 5);
        let mut root_sources = Vec::with_capacity(self.plants.len() * 5);
        for plant in &self.plants {
            canopy_sources.extend([
                plant.x as f32,
                plant.y as f32,
                plant.canopy_radius_cells() as f32,
                plant.canopy_amplitude(),
                0.72,
            ]);
            root_sources.extend([
                plant.x as f32,
                plant.y as f32,
                plant.root_radius_cells() as f32,
                plant.root_amplitude(),
                0.95,
            ]);
        }
        let (canopy, root) = build_dual_radial_fields(
            self.config.width,
            self.config.height,
            &canopy_sources,
            &root_sources,
        )?;
        self.canopy_cover.copy_from_slice(&canopy);
        self.root_density.copy_from_slice(&root);
        Ok(())
    }

    fn rebuild_water_mask(&mut self) {
        self.water_mask.fill(0.0);
        for water in &self.waters {
            if !water.alive {
                continue;
            }
            let amplitude = clamp(water.volume / 140.0, 0.06, 1.0);
            deposit_2d(
                &mut self.water_mask,
                self.config.width,
                self.config.height,
                water.x,
                water.y,
                2,
                amplitude,
            );
        }
    }

    fn sync_substrate_controls(&mut self) -> Result<(), String> {
        let plane = self.config.width * self.config.height;
        let total = plane * self.config.depth.max(1);
        let mut hydration = vec![0.0f32; total];
        let mut microbes = vec![0.0f32; total];
        let mut plant_drive = vec![0.0f32; total];
        let depth = self.config.depth.max(1);
        for z in 0..depth {
            let z_frac = if depth > 1 {
                z as f32 / (depth - 1) as f32
            } else {
                0.0
            };
            for i in 0..plane {
                let gid = z * plane + i;
                hydration[gid] = clamp(
                    self.moisture[i] * (1.0 - z_frac * 0.55) + self.deep_moisture[i] * z_frac,
                    0.02,
                    1.0,
                );
                microbes[gid] = clamp(
                    self.microbial_biomass[i] * (0.65 + self.moisture[i] * 0.55)
                        + self.symbiont_biomass[i] * (0.55 + z_frac * 0.30),
                    0.02,
                    1.2,
                );
                plant_drive[gid] = clamp(
                    self.root_density[i] * (1.0 - z_frac * 0.35) * (0.35 + self.daylight() * 0.65),
                    0.0,
                    1.5,
                );
            }
        }
        self.substrate.set_hydration_field(&hydration)?;
        self.substrate.set_microbial_activity_field(&microbes)?;
        self.substrate.set_plant_drive_field(&plant_drive)?;
        Ok(())
    }

    fn sample_temperature_at(&self, x: usize, y: usize, z: usize) -> f32 {
        self.temperature[idx3(
            self.config.width,
            self.config.height,
            x.min(self.config.width - 1),
            y.min(self.config.height - 1),
            z.min(self.config.depth.max(1) - 1),
        )]
    }

    fn plant_source_states(&self) -> Vec<PlantSourceState> {
        self.plants
            .iter()
            .filter(|plant| !plant.physiology.is_dead())
            .map(|plant| plant.source_state(self.config.depth.max(1)))
            .collect()
    }

    fn fruit_food_patches(&self) -> Vec<f32> {
        let mut patches = Vec::with_capacity(self.fruits.len() * 4);
        for fruit in &self.fruits {
            if !fruit.source.alive || fruit.source.sugar_content <= 0.0 {
                continue;
            }
            patches.extend([
                fruit.source.x as f32,
                fruit.source.y as f32,
                fruit.radius.max(0.5),
                fruit.source.sugar_content.clamp(0.0, 1.0),
            ]);
        }
        patches
    }

    fn consume_nearest_fruit(&mut self, x: f32, y: f32, amount: f32) {
        if amount <= 0.0 {
            return;
        }
        let mut best_idx = None;
        let mut best_dist_sq = f32::INFINITY;
        for (idx, fruit) in self.fruits.iter().enumerate() {
            if !fruit.source.alive || fruit.source.sugar_content <= 0.0 {
                continue;
            }
            let dx = x - fruit.source.x as f32;
            let dy = y - fruit.source.y as f32;
            let dist_sq = dx * dx + dy * dy;
            if dist_sq < best_dist_sq {
                best_dist_sq = dist_sq;
                best_idx = Some(idx);
            }
        }
        if let Some(idx) = best_idx {
            let fruit = &mut self.fruits[idx];
            fruit.source.sugar_content = (fruit.source.sugar_content - amount).max(0.0);
            if fruit.source.sugar_content <= 0.0 {
                fruit.source.alive = false;
            }
        }
    }

    fn step_broad_soil(&mut self, eco_dt: f32) -> Result<(), String> {
        self.rebuild_water_mask();
        let result = step_soil_broad_pools(
            self.config.width,
            self.config.height,
            eco_dt,
            self.daylight(),
            temp_response(self.sample_temperature_at(self.config.width / 2, self.config.height / 2, 0), 24.0, 10.0),
            &self.water_mask,
            &self.canopy_cover,
            &self.root_density,
            &self.moisture,
            &self.deep_moisture,
            &self.dissolved_nutrients,
            &self.mineral_nitrogen,
            &self.shallow_nutrients,
            &self.deep_minerals,
            &self.organic_matter,
            &self.litter_carbon,
            &self.microbial_biomass,
            &self.symbiont_biomass,
            &self.root_exudates,
            &self.soil_structure,
        )?;
        self.moisture = result.moisture;
        self.deep_moisture = result.deep_moisture;
        self.dissolved_nutrients = result.dissolved_nutrients;
        self.mineral_nitrogen = result.mineral_nitrogen;
        self.shallow_nutrients = result.shallow_nutrients;
        self.deep_minerals = result.deep_minerals;
        self.organic_matter = result.organic_matter;
        self.litter_carbon = result.litter_carbon;
        self.microbial_biomass = result.microbial_biomass;
        self.symbiont_biomass = result.symbiont_biomass;
        self.root_exudates = result.root_exudates;
        Ok(())
    }

    fn step_plants(&mut self, eco_dt: f32) -> Result<(), String> {
        if self.plants.is_empty() {
            return Ok(());
        }
        let depth = self.config.depth.max(1);
        let ammonium_surface = layer_mean_map(
            self.config.width,
            self.config.height,
            depth,
            self.substrate.species_field(TerrariumSpecies::Ammonium),
            0,
            depth.min(2),
        );
        let nitrate_surface = layer_mean_map(
            self.config.width,
            self.config.height,
            depth,
            self.substrate.species_field(TerrariumSpecies::Nitrate),
            0,
            depth.min(2),
        );
        let nitrate_deep = layer_mean_map(
            self.config.width,
            self.config.height,
            depth,
            self.substrate.species_field(TerrariumSpecies::Nitrate),
            depth / 2,
            depth,
        );
        let daylight = self.daylight();
        let mut queued_fruits = Vec::new();
        let mut queued_seeds = Vec::new();
        let mut dead_plants = Vec::new();

        for idx in 0..self.plants.len() {
            let fruit_reset_s = self.rng.gen_range(7200.0..17000.0);
            let seed_reset_s = self.rng.gen_range(12000.0..30000.0);
            let (x, y, genome, canopy_self, root_self, root_radius, storage_signal, health_before) = {
                let plant = &self.plants[idx];
                (
                    plant.x,
                    plant.y,
                    plant.genome.clone(),
                    plant.canopy_amplitude(),
                    plant.root_amplitude(),
                    plant.root_radius_cells(),
                    plant.physiology.storage_carbon().max(0.0),
                    plant.physiology.health(),
                )
            };
            let flat = idx2(self.config.width, x, y);
            let canopy_comp = (self.canopy_cover[flat] - canopy_self).max(0.0);
            let root_comp = (self.root_density[flat] - root_self).max(0.0);
            let symbionts = self.symbiont_biomass[flat];
            let deep_moisture = self.deep_moisture[flat];
            let litter = self.litter_carbon[flat];

            let oxygen = self.substrate.patch_mean_species(
                TerrariumSpecies::OxygenGas,
                x,
                y,
                1.min(depth - 1),
                root_radius,
            );
            let proton = self.substrate.patch_mean_species(
                TerrariumSpecies::Proton,
                x,
                y,
                1.min(depth - 1),
                root_radius,
            );
            let soil_nitrate_patch = self.substrate.patch_mean_species(
                TerrariumSpecies::Nitrate,
                x,
                y,
                1.min(depth - 1),
                root_radius,
            );
            let carbon_dioxide = self.substrate.patch_mean_species(
                TerrariumSpecies::CarbonDioxide,
                x,
                y,
                1.min(depth - 1),
                root_radius,
            );
            let soil_redox = clamp(
                (oxygen * 1.3 + soil_nitrate_patch * 0.7)
                    / (oxygen * 1.3
                        + soil_nitrate_patch * 0.7
                        + carbon_dioxide * 0.35
                        + proton * 0.45
                        + 1e-9),
                0.0,
                1.0,
            );
            let soil_glucose = self.substrate.patch_mean_species(
                TerrariumSpecies::Glucose,
                x,
                y,
                1.min(depth - 1),
                root_radius,
            );
            let soil_ammonium = self.substrate.patch_mean_species(
                TerrariumSpecies::Ammonium,
                x,
                y,
                1.min(depth - 1),
                root_radius,
            );
            let atp_flux = self.substrate.patch_mean_species(
                TerrariumSpecies::AtpFlux,
                x,
                y,
                1.min(depth - 1),
                root_radius,
            );
            let root_energy_gate = clamp(0.32 + atp_flux * 1800.0, 0.2, 1.2);
            let water_deficit = clamp(0.42 - (self.moisture[flat] + deep_moisture * 0.35), 0.0, 1.0);
            let nitrogen_deficit = clamp(
                0.14
                    - (self.mineral_nitrogen[flat] + soil_ammonium + soil_nitrate_patch) * 0.32,
                0.0,
                1.0,
            );

            let (water_demand, nutrient_demand) = self.plants[idx]
                .physiology
                .resource_demands(eco_dt, root_energy_gate, water_deficit, nitrogen_deficit);
            let extraction = extract_root_resources_with_layers(
                self.config.width,
                self.config.height,
                x as i32,
                y as i32,
                root_radius as i32,
                water_demand,
                nutrient_demand,
                genome.root_depth_bias,
                genome.symbiosis_affinity,
                root_energy_gate,
                &self.moisture,
                &self.deep_moisture,
                &self.dissolved_nutrients,
                &self.mineral_nitrogen,
                &self.shallow_nutrients,
                &self.deep_minerals,
                &self.symbiont_biomass,
                &ammonium_surface,
                &nitrate_surface,
                &nitrate_deep,
            )?;
            self.moisture = extraction.moisture;
            self.deep_moisture = extraction.deep_moisture;
            self.dissolved_nutrients = extraction.dissolved_nutrients;
            self.mineral_nitrogen = extraction.mineral_nitrogen;
            self.shallow_nutrients = extraction.shallow_nutrients;
            self.deep_minerals = extraction.deep_minerals;

            let total_water_take = extraction.surface_water_take + extraction.deep_water_take;
            if total_water_take > 0.0 {
                let _ = self.substrate.extract_patch_species(
                    TerrariumSpecies::Water,
                    x,
                    y,
                    0,
                    root_radius,
                    total_water_take,
                );
            }
            if extraction.ammonium_take > 0.0 {
                let _ = self.substrate.extract_patch_species(
                    TerrariumSpecies::Ammonium,
                    x,
                    y,
                    0,
                    root_radius,
                    extraction.ammonium_take,
                );
            }
            let nitrate_take = extraction.rhizo_nitrate_take + extraction.deep_nitrate_take;
            if nitrate_take > 0.0 {
                let _ = self.substrate.extract_patch_species(
                    TerrariumSpecies::Nitrate,
                    x,
                    y,
                    0,
                    root_radius,
                    nitrate_take,
                );
            }

            let water_factor = if water_demand > 1.0e-6 {
                clamp(extraction.water_take / water_demand, 0.0, 1.1)
            } else {
                1.0
            };
            let nutrient_factor = if nutrient_demand > 1.0e-6 {
                clamp(extraction.nutrient_take / nutrient_demand, 0.0, 1.1)
            } else {
                1.0
            };
            let local_temp = self.sample_temperature_at(x, y, 1.min(depth - 1));
            let temp_factor = temp_response(local_temp, 24.0, 10.0);
            let local_light = clamp(
                daylight * (1.0 - canopy_comp * (1.18 - genome.shade_tolerance).max(0.16)),
                0.03,
                1.15,
            );
            let symbiosis_signal = clamp(symbionts * 6.5 * genome.symbiosis_affinity, 0.0, 1.5);
            let symbiosis_bonus = 1.0 + symbionts * 7.5 * genome.symbiosis_affinity;
            let stress_signal = clamp(
                water_deficit * 0.75
                    + nitrogen_deficit * 0.58
                    + canopy_comp * 0.018
                    + root_comp * 0.014
                    + (1.0 - soil_redox) * 0.45,
                0.0,
                1.4,
            );

            let (
                report,
                new_total_cells,
                cell_vitality,
                volatile_scale,
                seed_mass,
                updated_health,
                updated_storage,
                root_radius_after,
            ) = {
                let plant = &mut self.plants[idx];
                let cell_feedback = plant.cellular.step(
                    eco_dt,
                    local_light,
                    temp_factor,
                    extraction.water_take,
                    extraction.nutrient_take,
                    water_factor,
                    nutrient_factor,
                    symbiosis_signal,
                    stress_signal,
                    storage_signal,
                );
                let report = plant.physiology.step(
                    eco_dt,
                    extraction.water_take,
                    extraction.nutrient_take,
                    local_light,
                    temp_factor,
                    root_energy_gate,
                    symbiosis_bonus,
                    water_factor,
                    nutrient_factor,
                    canopy_comp,
                    root_comp,
                    soil_glucose,
                    cell_feedback.photosynthetic_capacity,
                    cell_feedback.maintenance_cost,
                    cell_feedback.storage_exchange,
                    cell_feedback.division_growth,
                    cell_feedback.senescence_mass,
                    cell_feedback.energy_charge,
                    cell_feedback.vitality,
                    cell_feedback.sugar_pool,
                    cell_feedback.division_signal,
                    plant.cellular.total_cells(),
                    fruit_reset_s,
                    seed_reset_s,
                );
                (
                    report,
                    plant.cellular.total_cells(),
                    plant.cellular.vitality(),
                    plant.genome.volatile_scale,
                    plant.genome.seed_mass,
                    plant.physiology.health(),
                    plant.physiology.storage_carbon(),
                    plant.root_radius_cells(),
                )
            };

            deposit_2d(
                &mut self.root_exudates,
                self.config.width,
                self.config.height,
                x,
                y,
                root_radius_after.max(1) / 2,
                report.exudates,
            );
            deposit_2d(
                &mut self.litter_carbon,
                self.config.width,
                self.config.height,
                x,
                y,
                root_radius_after.max(1) / 2,
                report.litter,
            );
            let hotspot_radius = root_radius_after.max(1);
            self.substrate.add_hotspot(
                TerrariumSpecies::Glucose,
                x,
                y,
                0,
                report.exudates * 12.0,
            );
            self.substrate.add_hotspot(
                TerrariumSpecies::Ammonium,
                x,
                y,
                0,
                report.litter * 8.0,
            );
            self.substrate.add_hotspot(
                TerrariumSpecies::CarbonDioxide,
                x,
                y,
                1.min(depth - 1),
                report.litter * 4.0,
            );
            deposit_2d(
                &mut self.organic_matter,
                self.config.width,
                self.config.height,
                x,
                y,
                hotspot_radius / 2,
                report.litter * 0.18,
            );

            if report.spawned_fruit && self.fruits.len() + queued_fruits.len() < self.config.max_fruits {
                queued_fruits.push((x, y, report.fruit_size, volatile_scale));
            }
            if report.spawned_seed && self.seeds.len() + queued_seeds.len() < self.config.max_seeds {
                let dispersal = (7.0 - seed_mass * 18.0 + updated_health * 1.5).round().max(2.0) as isize;
                let dx = self.rng.gen_range(-dispersal..=dispersal);
                let dy = self.rng.gen_range(-dispersal..=dispersal);
                let sx = offset_clamped(x, dx, self.config.width);
                let sy = offset_clamped(y, dy, self.config.height);
                let dormancy =
                    self.rng.gen_range(9000.0..26000.0) * (1.18 - seed_mass * 1.5).max(0.45);
                let reserve = clamp(
                    seed_mass * self.rng.gen_range(0.85..1.25) + updated_storage.max(0.0) * 0.05,
                    0.03,
                    0.28,
                );
                queued_seeds.push(TerrariumSeed {
                    x: sx as f32,
                    y: sy as f32,
                    dormancy_s: dormancy,
                    reserve_carbon: reserve,
                    age_s: 0.0,
                    genome: genome.mutate(&mut self.rng),
                });
            }
            if self.plants[idx].physiology.is_dead() || (new_total_cells < 24.0 && cell_vitality < 0.08) {
                dead_plants.push(idx);
            }

            let _ = (health_before, litter);
        }

        dead_plants.sort_unstable();
        dead_plants.dedup();
        for idx in dead_plants.into_iter().rev() {
            let plant = self.plants.swap_remove(idx);
            deposit_2d(
                &mut self.litter_carbon,
                self.config.width,
                self.config.height,
                plant.x,
                plant.y,
                plant.root_radius_cells().max(1) / 2,
                0.02 + plant.physiology.storage_carbon().max(0.0) * 0.20,
            );
        }
        for (x, y, size, volatile_scale) in queued_fruits {
            let offset = self.rng.gen_range(0..4);
            let (dx, dy) = match offset {
                0 => (2, 1),
                1 => (-2, 1),
                2 => (1, -2),
                _ => (-1, -2),
            };
            self.add_fruit(
                offset_clamped(x, dx, self.config.width),
                offset_clamped(y, dy, self.config.height),
                size,
                Some(volatile_scale),
            );
        }
        self.seeds.extend(queued_seeds);
        Ok(())
    }

    fn step_food_patches_native(&mut self, eco_dt: f32) -> Result<(), String> {
        if self.fruits.is_empty() {
            return Ok(());
        }
        let patch_remaining = self
            .fruits
            .iter()
            .map(|fruit| fruit.source.sugar_content.max(0.0))
            .collect::<Vec<_>>();
        let previous_remaining = self
            .fruits
            .iter()
            .map(|fruit| fruit.previous_remaining.max(0.0))
            .collect::<Vec<_>>();
        let deposited_all = self
            .fruits
            .iter()
            .map(|fruit| fruit.deposited_all)
            .collect::<Vec<_>>();
        let has_fruit = self
            .fruits
            .iter()
            .map(|fruit| fruit.source.alive)
            .collect::<Vec<_>>();
        let fruit_ripeness = self
            .fruits
            .iter()
            .map(|fruit| fruit.source.ripeness)
            .collect::<Vec<_>>();
        let fruit_sugar = self
            .fruits
            .iter()
            .map(|fruit| fruit.source.sugar_content)
            .collect::<Vec<_>>();
        let microbial = self
            .fruits
            .iter()
            .map(|fruit| {
                let flat = idx2(
                    self.config.width,
                    fruit.source.x.min(self.config.width - 1),
                    fruit.source.y.min(self.config.height - 1),
                );
                self.microbial_biomass[flat]
            })
            .collect::<Vec<_>>();
        let stepped = step_food_patches(
            eco_dt,
            &patch_remaining,
            &previous_remaining,
            &deposited_all,
            &has_fruit,
            &fruit_ripeness,
            &fruit_sugar,
            &microbial,
        )?;
        for (idx, fruit) in self.fruits.iter_mut().enumerate() {
            let current = stepped.remaining[idx].max(0.0);
            fruit.source.sugar_content = current.min(stepped.sugar_content[idx]);
            fruit.source.alive = stepped.fruit_alive[idx];
            fruit.deposited_all = stepped.deposited_all[idx];
            let x = fruit.source.x;
            let y = fruit.source.y;
            let radius = fruit.radius.round().max(1.0) as usize;
            if stepped.decay_detritus[idx] > 0.0 {
                deposit_2d(
                    &mut self.litter_carbon,
                    self.config.width,
                    self.config.height,
                    x,
                    y,
                    radius,
                    stepped.decay_detritus[idx],
                );
            }
            if stepped.lost_detritus[idx] > 0.0 {
                deposit_2d(
                    &mut self.litter_carbon,
                    self.config.width,
                    self.config.height,
                    x,
                    y,
                    radius,
                    stepped.lost_detritus[idx],
                );
            }
            if stepped.final_detritus[idx] > 0.0 {
                deposit_2d(
                    &mut self.litter_carbon,
                    self.config.width,
                    self.config.height,
                    x,
                    y,
                    radius,
                    stepped.final_detritus[idx],
                );
            }
            fruit.previous_remaining = current;
        }
        Ok(())
    }

    fn step_seeds_native(&mut self, eco_dt: f32) -> Result<(), String> {
        if self.seeds.is_empty() {
            return Ok(());
        }
        let mut dormancy = Vec::with_capacity(self.seeds.len());
        let mut age = Vec::with_capacity(self.seeds.len());
        let mut reserve = Vec::with_capacity(self.seeds.len());
        let mut affinity = Vec::with_capacity(self.seeds.len());
        let mut shade = Vec::with_capacity(self.seeds.len());
        let mut moisture = Vec::with_capacity(self.seeds.len());
        let mut deep_moisture = Vec::with_capacity(self.seeds.len());
        let mut nutrients = Vec::with_capacity(self.seeds.len());
        let mut symbionts = Vec::with_capacity(self.seeds.len());
        let mut canopy = Vec::with_capacity(self.seeds.len());
        let mut litter = Vec::with_capacity(self.seeds.len());
        let mut positions = Vec::with_capacity(self.seeds.len());

        for seed in &self.seeds {
            let x = seed.x.round().clamp(0.0, (self.config.width - 1) as f32) as usize;
            let y = seed.y.round().clamp(0.0, (self.config.height - 1) as f32) as usize;
            let flat = idx2(self.config.width, x, y);
            positions.push((x, y));
            dormancy.push(seed.dormancy_s);
            age.push(seed.age_s);
            reserve.push(seed.reserve_carbon);
            affinity.push(seed.genome.symbiosis_affinity);
            shade.push(seed.genome.shade_tolerance);
            moisture.push(self.moisture[flat]);
            deep_moisture.push(self.deep_moisture[flat]);
            nutrients.push(self.shallow_nutrients[flat]);
            symbionts.push(self.symbiont_biomass[flat]);
            canopy.push(self.canopy_cover[flat]);
            litter.push(self.litter_carbon[flat]);
        }

        let stepped = step_seed_bank(
            eco_dt,
            self.daylight(),
            self.plants.len(),
            self.config.max_plants,
            &dormancy,
            &age,
            &reserve,
            &affinity,
            &shade,
            &moisture,
            &deep_moisture,
            &nutrients,
            &symbionts,
            &canopy,
            &litter,
        )?;

        let mut next_bank = Vec::new();
        let mut germinations = Vec::new();
        for (idx, mut seed) in self.seeds.drain(..).enumerate() {
            seed.age_s = stepped.age_s[idx];
            seed.dormancy_s = stepped.dormancy_s[idx];
            if stepped.germinate[idx] && self.plants.len() < self.config.max_plants {
                let (x, y) = positions[idx];
                let scale = stepped.seedling_scale[idx].max(0.45);
                germinations.push((x, y, seed.genome, scale));
            } else if stepped.keep[idx] {
                next_bank.push(seed);
            }
        }
        self.seeds = next_bank;
        for (x, y, genome, scale) in germinations {
            let _ = self.add_plant(x, y, Some(genome), Some(scale));
        }
        Ok(())
    }

    fn step_world_fields(&mut self) -> Result<(), String> {
        let mut fruit_sources = self
            .fruits
            .iter()
            .map(|fruit| fruit.source.clone())
            .collect::<Vec<_>>();
        let plant_sources = self.plant_source_states();
        step_molecular_world_fields(
            self.config.width,
            self.config.height,
            self.config.depth.max(1),
            self.config.world_dt_s,
            self.config.cell_size_mm,
            true,
            self.daylight(),
            Some(AMMONIA_IDX),
            &mut fruit_sources,
            &plant_sources,
            &mut self.waters,
            &mut self.odorants,
            &self.odorant_params,
            &mut self.temperature,
            &mut self.humidity,
            &mut self.wind_x,
            &mut self.wind_y,
            &mut self.wind_z,
            &mut self.atmosphere_rng_state,
        )?;
        for (dst, src) in self.fruits.iter_mut().zip(fruit_sources.into_iter()) {
            dst.source = src;
        }
        Ok(())
    }

    fn step_flies(&mut self) -> Result<(), String> {
        if self.flies.is_empty() {
            return Ok(());
        }
        let food_patches = self.fruit_food_patches();
        self.sensory_field.load_state(
            &self.odorants[ETHYL_ACETATE_IDX],
            &self.temperature,
            &self.wind_x,
            &self.wind_y,
            &self.wind_z,
            self.daylight(),
            &food_patches,
        )?;
        let mut consumptions = Vec::new();
        for fly in &mut self.flies {
            let body = fly.body_state();
            let sample = self.sensory_field.sample_fly(
                body.x,
                body.y,
                body.z,
                body.heading,
                body.is_flying,
            );
            let report = fly.body_step_terrarium(
                sample.odorant,
                sample.left_light,
                sample.right_light,
                sample.temperature,
                sample.sugar_taste,
                sample.bitter_taste,
                sample.amino_taste,
                sample.wind_x,
                sample.wind_y,
                sample.wind_z,
                sample.food_available,
                0.0,
            );
            if report.consumed_food > 0.0 {
                consumptions.push((report.x, report.y, report.consumed_food));
            }
        }
        for (x, y, amount) in consumptions {
            self.fly_food_total += amount;
            self.consume_nearest_fruit(x, y, amount);
        }
        Ok(())
    }

    pub fn step_frame(&mut self) -> Result<(), String> {
        let eco_dt = self.config.world_dt_s * self.config.time_warp;
        for _ in 0..self.config.substeps.max(1) {
            self.rebuild_ecology_fields()?;
            self.step_broad_soil(eco_dt)?;
            self.sync_substrate_controls()?;
            self.substrate.step(self.config.substrate_dt_ms);
            self.step_plants(eco_dt)?;
            self.rebuild_ecology_fields()?;
            self.step_world_fields()?;
            self.step_flies()?;
            self.step_food_patches_native(eco_dt)?;
            self.step_seeds_native(eco_dt)?;
            self.time_s += eco_dt;
        }
        Ok(())
    }

    pub fn run_frames(&mut self, frames: usize) -> Result<(), String> {
        for _ in 0..frames {
            self.step_frame()?;
        }
        Ok(())
    }

    pub fn snapshot(&self) -> TerrariumWorldSnapshot {
        let live_fruits = self
            .fruits
            .iter()
            .filter(|fruit| fruit.source.alive && fruit.source.sugar_content > 0.01)
            .count();
        let food_remaining = self
            .fruits
            .iter()
            .filter(|fruit| fruit.source.alive)
            .map(|fruit| fruit.source.sugar_content.max(0.0))
            .sum::<f32>();
        let avg_fly_energy = if self.flies.is_empty() {
            0.0
        } else {
            self.flies
                .iter()
                .map(|fly| fly.body_state().energy)
                .sum::<f32>()
                / self.flies.len() as f32
        };
        let avg_altitude = if self.flies.is_empty() {
            0.0
        } else {
            self.flies
                .iter()
                .map(|fly| fly.body_state().z)
                .sum::<f32>()
                / self.flies.len() as f32
        };
        let total_plant_cells = self.plants.iter().map(|plant| plant.cellular.total_cells()).sum::<f32>();
        let mean_cell_vitality = if self.plants.is_empty() {
            0.0
        } else {
            self.plants
                .iter()
                .map(|plant| plant.cellular.vitality())
                .sum::<f32>()
                / self.plants.len() as f32
        };
        let mean_cell_energy = if self.plants.is_empty() {
            0.0
        } else {
            self.plants
                .iter()
                .map(|plant| plant.cellular.energy_charge())
                .sum::<f32>()
                / self.plants.len() as f32
        };
        let mean_division_pressure = if self.plants.is_empty() {
            0.0
        } else {
            self.plants
                .iter()
                .map(|plant| plant.cellular.division_signal())
                .sum::<f32>()
                / self.plants.len() as f32
        };

        let mean_soil_glucose = self.substrate.mean_species(TerrariumSpecies::Glucose);
        let mean_soil_oxygen = self.substrate.mean_species(TerrariumSpecies::OxygenGas);
        let mean_soil_ammonium = self.substrate.mean_species(TerrariumSpecies::Ammonium);
        let mean_soil_nitrate = self.substrate.mean_species(TerrariumSpecies::Nitrate);
        let mean_soil_redox = {
            let oxygen = mean_soil_oxygen;
            let nitrate = mean_soil_nitrate;
            let carbon_dioxide = self.substrate.mean_species(TerrariumSpecies::CarbonDioxide);
            let proton = self.substrate.mean_species(TerrariumSpecies::Proton);
            clamp(
                (oxygen * 1.3 + nitrate * 0.7)
                    / (oxygen * 1.3 + nitrate * 0.7 + carbon_dioxide * 0.35 + proton * 0.45 + 1e-9),
                0.0,
                1.0,
            )
        };

        TerrariumWorldSnapshot {
            plants: self.plants.len(),
            fruits: live_fruits,
            seeds: self.seeds.len(),
            flies: self.flies.len(),
            food_remaining,
            fly_food_total: self.fly_food_total,
            avg_fly_energy,
            avg_altitude,
            light: self.daylight(),
            temperature: mean(&self.temperature),
            humidity: mean(&self.humidity),
            mean_soil_moisture: mean(&self.moisture),
            mean_deep_moisture: mean(&self.deep_moisture),
            mean_microbes: mean(&self.microbial_biomass),
            mean_symbionts: mean(&self.symbiont_biomass),
            mean_canopy: mean(&self.canopy_cover),
            mean_root_density: mean(&self.root_density),
            total_plant_cells,
            mean_cell_vitality,
            mean_cell_energy,
            mean_division_pressure,
            mean_soil_glucose,
            mean_soil_oxygen,
            mean_soil_ammonium,
            mean_soil_nitrate,
            mean_soil_redox,
            mean_soil_atp_flux: self.substrate.mean_species(TerrariumSpecies::AtpFlux),
            substrate_backend: self.substrate.backend().as_str(),
            substrate_steps: self.substrate.step_count(),
            substrate_time_ms: self.substrate.time_ms(),
            time_s: self.time_s,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TerrariumWorld;

    #[test]
    fn native_terrarium_world_runs_and_stays_bounded() {
        let mut world = TerrariumWorld::demo(7, false).unwrap();
        world.run_frames(20).unwrap();
        let snapshot = world.snapshot();
        assert!(snapshot.plants > 0);
        assert!(snapshot.food_remaining >= 0.0);
        assert!(snapshot.light >= 0.0);
        assert!(snapshot.humidity >= 0.0);
        assert!(snapshot.mean_soil_moisture >= 0.0);
        assert!(snapshot.total_plant_cells > 0.0);
    }
}
