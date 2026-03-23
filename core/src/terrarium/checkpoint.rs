use std::path::Path;

use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TerrariumCheckpointFidelity {
    MacroEcology,
    #[serde(alias = "MacroEcologyWithFlyNeuralState")]
    MacroEcologyWithExplicitLowerScaleState,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TerrariumCheckpointLimitations {
    pub fidelity: TerrariumCheckpointFidelity,
    pub stochastic_streams_reseeded_from_resume_seeds: bool,
    pub fly_brain_state_reinitialized_on_restore: bool,
    pub explicit_microbe_count_omitted: usize,
    pub packet_population_count_omitted: usize,
    pub atomistic_probe_count_omitted: usize,
    pub ownership_map_reinitialized: bool,
    pub secondary_genotype_banks_reinitialized: bool,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct TerrariumExplicitMicrobeCheckpoint {
    pub x: usize,
    pub y: usize,
    pub z: usize,
    pub guild: u8,
    pub represented_cells: f32,
    pub represented_packets: f32,
    pub identity: TerrariumExplicitMicrobeIdentity,
    pub whole_cell: Option<crate::whole_cell_data::WholeCellCheckpoint>,
    pub simulator: crate::whole_cell_data::WholeCellCheckpoint,
    pub last_snapshot: WholeCellSnapshot,
    pub smoothed_energy: f32,
    pub smoothed_stress: f32,
    pub radius: usize,
    pub patch_radius: usize,
    pub age_steps: u64,
    pub age_s: f32,
    pub idx: u64,
    pub material_inventory: RegionalMaterialInventory,
    pub cumulative_glucose_draw: f32,
    pub cumulative_oxygen_draw: f32,
    pub cumulative_co2_release: f32,
    pub cumulative_ammonium_draw: f32,
    pub cumulative_nitrate_draw: f32,
    pub cumulative_proton_release: f32,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct TerrariumAtomisticProbeCheckpoint {
    pub id: u32,
    pub md: crate::molecular_dynamics::GPUMolecularDynamicsCheckpoint,
    pub grid_x: usize,
    pub grid_y: usize,
    pub footprint_radius: usize,
    pub n_atoms: usize,
    pub dt_fs: f32,
    pub temperature_k: f32,
    pub last_stats: crate::molecular_dynamics::MDStats,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct TerrariumWorldCheckpoint {
    pub config: TerrariumWorldConfig,
    pub time_s: f32,
    pub seed_provenance: TerrariumSeedProvenance,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rng: Option<ChaCha12Rng>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rng_resume_seed: Option<u64>,
    pub atmosphere_rng_state: u64,
    pub chronobiology_config: ChronobiologyConfig,
    pub climate_driver: Option<TerrariumClimateDriver>,
    #[serde(default)]
    pub weather: WeatherState,
    pub next_organism_id: u64,
    pub organism_registry: organism_identity::OrganismRegistry,
    pub organism_phylogeny: crate::phylogenetic_tracker::PhyloTree,
    pub odorant_params: Vec<OdorantChannelParams>,
    pub odorants: Vec<Vec<f32>>,
    pub temperature: Vec<f32>,
    pub humidity: Vec<f32>,
    pub wind_x: Vec<f32>,
    pub wind_y: Vec<f32>,
    pub wind_z: Vec<f32>,
    pub substrate: BatchedAtomTerrariumCheckpoint,
    pub waters: Vec<WaterSourceState>,
    pub fruits: Vec<TerrariumFruitPatch>,
    pub plants: Vec<TerrariumPlant>,
    pub seeds: Vec<TerrariumSeed>,
    pub fly_identities: Vec<OrganismIdentity>,
    pub flies: Vec<crate::drosophila::DrosophilaCheckpointState>,
    pub canopy_cover: Vec<f32>,
    pub root_density: Vec<f32>,
    pub water_mask: Vec<f32>,
    pub moisture: Vec<f32>,
    pub deep_moisture: Vec<f32>,
    #[serde(default)]
    pub soil_layer_moisture: [Vec<f32>; 4],
    #[serde(default)]
    pub soil_layer_nitrogen: [Vec<f32>; 4],
    pub dissolved_nutrients: Vec<f32>,
    pub mineral_nitrogen: Vec<f32>,
    pub shallow_nutrients: Vec<f32>,
    pub deep_minerals: Vec<f32>,
    pub organic_matter: Vec<f32>,
    pub litter_carbon: Vec<f32>,
    pub litter_surface: Vec<litter_surface::LitterSurfaceState>,
    pub microbial_biomass: Vec<f32>,
    pub symbiont_biomass: Vec<f32>,
    pub root_exudates: Vec<f32>,
    pub soil_structure: Vec<f32>,
    pub microbial_cells: Vec<f32>,
    pub microbial_packets: Vec<f32>,
    pub microbial_copiotroph_packets: Vec<f32>,
    pub microbial_copiotroph_fraction: Vec<f32>,
    pub microbial_strain_yield: Vec<f32>,
    pub microbial_strain_stress_tolerance: Vec<f32>,
    pub microbial_latent_packets: Vec<Vec<f32>>,
    pub microbial_latent_strain_yield: Vec<Vec<f32>>,
    pub microbial_latent_strain_stress_tolerance: Vec<Vec<f32>>,
    pub next_probe_id: u32,
    pub ownership: Vec<SoilOwnershipCell>,
    pub(crate) microbial_secondary: genotype::PublicSecondaryBanks,
    pub(crate) nitrifier_secondary: genotype::PublicSecondaryBanks,
    pub(crate) denitrifier_secondary: genotype::PublicSecondaryBanks,
    pub microbial_vitality: Vec<f32>,
    pub microbial_dormancy: Vec<f32>,
    pub microbial_reserve: Vec<f32>,
    pub microbial_packet_mutation_flux: Vec<f32>,
    pub nitrifier_cells: Vec<f32>,
    pub nitrifier_packets: Vec<f32>,
    pub nitrifier_aerobic_packets: Vec<f32>,
    pub nitrifier_aerobic_fraction: Vec<f32>,
    pub nitrifier_strain_oxygen_affinity: Vec<f32>,
    pub nitrifier_strain_ammonium_affinity: Vec<f32>,
    pub nitrifier_latent_packets: Vec<Vec<f32>>,
    pub nitrifier_latent_strain_oxygen_affinity: Vec<Vec<f32>>,
    pub nitrifier_latent_strain_ammonium_affinity: Vec<Vec<f32>>,
    pub nitrifier_vitality: Vec<f32>,
    pub nitrifier_dormancy: Vec<f32>,
    pub nitrifier_reserve: Vec<f32>,
    pub nitrifier_packet_mutation_flux: Vec<f32>,
    pub nitrification_potential: Vec<f32>,
    pub denitrifier_cells: Vec<f32>,
    pub denitrifier_packets: Vec<f32>,
    pub denitrifier_anoxic_packets: Vec<f32>,
    pub denitrifier_anoxic_fraction: Vec<f32>,
    pub denitrifier_strain_anoxia_affinity: Vec<f32>,
    pub denitrifier_strain_nitrate_affinity: Vec<f32>,
    pub denitrifier_latent_packets: Vec<Vec<f32>>,
    pub denitrifier_latent_strain_anoxia_affinity: Vec<Vec<f32>>,
    pub denitrifier_latent_strain_nitrate_affinity: Vec<Vec<f32>>,
    pub denitrifier_vitality: Vec<f32>,
    pub denitrifier_dormancy: Vec<f32>,
    pub denitrifier_reserve: Vec<f32>,
    pub denitrifier_packet_mutation_flux: Vec<f32>,
    pub denitrification_potential: Vec<f32>,
    pub explicit_microbes: Vec<TerrariumExplicitMicrobeCheckpoint>,
    pub next_microbe_idx: u64,
    pub packet_populations: Vec<packet::GenotypePacketPopulation>,
    pub atomistic_probes: Vec<TerrariumAtomisticProbeCheckpoint>,
    pub fly_food_total: f32,
    pub fly_metabolisms: Vec<FlyMetabolism>,
    pub fly_population: crate::drosophila_population::FlyPopulationCheckpoint,
    pub earthworm_population: EarthwormPopulation,
    pub nematode_guilds: Vec<NematodeGuild>,
    pub air_pressure_kpa: Vec<f32>,
    pub substep_counter: u64,
    pub snapshot: TerrariumWorldSnapshot,
    pub limitations: TerrariumCheckpointLimitations,
}

impl TerrariumCheckpointLimitations {
    fn from_world(_world: &TerrariumWorld) -> Self {
        Self {
            fidelity: TerrariumCheckpointFidelity::MacroEcologyWithExplicitLowerScaleState,
            stochastic_streams_reseeded_from_resume_seeds: false,
            fly_brain_state_reinitialized_on_restore: false,
            explicit_microbe_count_omitted: 0,
            packet_population_count_omitted: 0,
            atomistic_probe_count_omitted: 0,
            ownership_map_reinitialized: false,
            secondary_genotype_banks_reinitialized: false,
        }
    }
}

impl TerrariumExplicitMicrobe {
    fn checkpoint_state(&self) -> Result<TerrariumExplicitMicrobeCheckpoint, String> {
        Ok(TerrariumExplicitMicrobeCheckpoint {
            x: self.x,
            y: self.y,
            z: self.z,
            guild: self.guild,
            represented_cells: self.represented_cells,
            represented_packets: self.represented_packets,
            identity: self.identity.clone(),
            whole_cell: self
                .whole_cell
                .as_deref()
                .map(crate::whole_cell::WholeCellSimulator::checkpoint_state)
                .transpose()?,
            simulator: self.simulator.checkpoint_state()?,
            last_snapshot: self.last_snapshot.clone(),
            smoothed_energy: self.smoothed_energy,
            smoothed_stress: self.smoothed_stress,
            radius: self.radius,
            patch_radius: self.patch_radius,
            age_steps: self.age_steps,
            age_s: self.age_s,
            idx: self.idx,
            material_inventory: self.material_inventory.clone(),
            cumulative_glucose_draw: self.cumulative_glucose_draw,
            cumulative_oxygen_draw: self.cumulative_oxygen_draw,
            cumulative_co2_release: self.cumulative_co2_release,
            cumulative_ammonium_draw: self.cumulative_ammonium_draw,
            cumulative_nitrate_draw: self.cumulative_nitrate_draw,
            cumulative_proton_release: self.cumulative_proton_release,
        })
    }

    fn from_checkpoint_state(state: TerrariumExplicitMicrobeCheckpoint) -> Result<Self, String> {
        Ok(Self {
            x: state.x,
            y: state.y,
            z: state.z,
            guild: state.guild,
            represented_cells: state.represented_cells,
            represented_packets: state.represented_packets,
            identity: state.identity,
            whole_cell: state
                .whole_cell
                .map(crate::whole_cell::WholeCellSimulator::from_checkpoint_state)
                .transpose()?
                .map(Box::new),
            simulator: crate::whole_cell::WholeCellSimulator::from_checkpoint_state(
                state.simulator,
            )?,
            last_snapshot: state.last_snapshot,
            smoothed_energy: state.smoothed_energy,
            smoothed_stress: state.smoothed_stress,
            radius: state.radius,
            patch_radius: state.patch_radius,
            age_steps: state.age_steps,
            age_s: state.age_s,
            idx: state.idx,
            material_inventory: state.material_inventory,
            cumulative_glucose_draw: state.cumulative_glucose_draw,
            cumulative_oxygen_draw: state.cumulative_oxygen_draw,
            cumulative_co2_release: state.cumulative_co2_release,
            cumulative_ammonium_draw: state.cumulative_ammonium_draw,
            cumulative_nitrate_draw: state.cumulative_nitrate_draw,
            cumulative_proton_release: state.cumulative_proton_release,
        })
    }
}

impl AtomisticProbe {
    fn checkpoint_state(&self) -> TerrariumAtomisticProbeCheckpoint {
        TerrariumAtomisticProbeCheckpoint {
            id: self.id,
            md: self.md.checkpoint(),
            grid_x: self.grid_x,
            grid_y: self.grid_y,
            footprint_radius: self.footprint_radius,
            n_atoms: self.n_atoms,
            dt_fs: self.dt_fs,
            temperature_k: self.temperature_k,
            last_stats: self.last_stats.clone(),
        }
    }

    fn from_checkpoint_state(state: TerrariumAtomisticProbeCheckpoint) -> Result<Self, String> {
        Ok(Self {
            id: state.id,
            md: crate::molecular_dynamics::GPUMolecularDynamics::from_checkpoint(state.md)?,
            grid_x: state.grid_x,
            grid_y: state.grid_y,
            footprint_radius: state.footprint_radius,
            n_atoms: state.n_atoms,
            dt_fs: state.dt_fs,
            temperature_k: state.temperature_k,
            last_stats: state.last_stats,
        })
    }
}

impl TerrariumWorldCheckpoint {
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn from_json_str(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    pub fn save_to_path<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let json = self
            .to_json_pretty()
            .map_err(|error| format!("serialize terrarium checkpoint: {error}"))?;
        std::fs::write(path.as_ref(), json).map_err(|error| {
            format!(
                "write terrarium checkpoint {}: {error}",
                path.as_ref().display()
            )
        })
    }

    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let json = std::fs::read_to_string(path.as_ref()).map_err(|error| {
            format!(
                "read terrarium checkpoint {}: {error}",
                path.as_ref().display()
            )
        })?;
        Self::from_json_str(&json).map_err(|error| {
            format!(
                "parse terrarium checkpoint {}: {error}",
                path.as_ref().display()
            )
        })
    }
}

impl TerrariumWorld {
    pub fn checkpoint(&mut self) -> Result<TerrariumWorldCheckpoint, String> {
        Ok(TerrariumWorldCheckpoint {
            config: self.config.clone(),
            time_s: self.time_s,
            seed_provenance: self.seed_provenance.clone(),
            rng: Some(self.rng.clone()),
            rng_resume_seed: None,
            atmosphere_rng_state: self.atmosphere_rng_state,
            chronobiology_config: self.chronobiology_config.clone(),
            climate_driver: self.climate_driver.clone(),
            weather: self.weather,
            next_organism_id: self.next_organism_id,
            organism_registry: self.organism_registry.clone(),
            organism_phylogeny: self.organism_phylogeny.clone(),
            odorant_params: self.odorant_params.clone(),
            odorants: self.odorants.clone(),
            temperature: self.temperature.clone(),
            humidity: self.humidity.clone(),
            wind_x: self.wind_x.clone(),
            wind_y: self.wind_y.clone(),
            wind_z: self.wind_z.clone(),
            substrate: self.substrate.checkpoint(),
            waters: self.waters.clone(),
            fruits: self.fruits.clone(),
            plants: self.plants.clone(),
            seeds: self.seeds.clone(),
            fly_identities: self.fly_identities.clone(),
            flies: self
                .flies
                .iter_mut()
                .map(crate::drosophila::DrosophilaSim::checkpoint_state)
                .collect(),
            canopy_cover: self.canopy_cover.clone(),
            root_density: self.root_density.clone(),
            water_mask: self.water_mask.clone(),
            moisture: self.moisture.clone(),
            deep_moisture: self.deep_moisture.clone(),
            soil_layer_moisture: self.soil_layer_moisture.clone(),
            soil_layer_nitrogen: self.soil_layer_nitrogen.clone(),
            dissolved_nutrients: self.dissolved_nutrients.clone(),
            mineral_nitrogen: self.mineral_nitrogen.clone(),
            shallow_nutrients: self.shallow_nutrients.clone(),
            deep_minerals: self.deep_minerals.clone(),
            organic_matter: self.organic_matter.clone(),
            litter_carbon: self.litter_carbon.clone(),
            litter_surface: self.litter_surface.clone(),
            microbial_biomass: self.microbial_biomass.clone(),
            symbiont_biomass: self.symbiont_biomass.clone(),
            root_exudates: self.root_exudates.clone(),
            soil_structure: self.soil_structure.clone(),
            microbial_cells: self.microbial_cells.clone(),
            microbial_packets: self.microbial_packets.clone(),
            microbial_copiotroph_packets: self.microbial_copiotroph_packets.clone(),
            microbial_copiotroph_fraction: self.microbial_copiotroph_fraction.clone(),
            microbial_strain_yield: self.microbial_strain_yield.clone(),
            microbial_strain_stress_tolerance: self.microbial_strain_stress_tolerance.clone(),
            microbial_latent_packets: self.microbial_latent_packets.clone(),
            microbial_latent_strain_yield: self.microbial_latent_strain_yield.clone(),
            microbial_latent_strain_stress_tolerance: self
                .microbial_latent_strain_stress_tolerance
                .clone(),
            next_probe_id: self.next_probe_id,
            ownership: self.ownership.clone(),
            microbial_secondary: self.microbial_secondary.clone(),
            nitrifier_secondary: self.nitrifier_secondary.clone(),
            denitrifier_secondary: self.denitrifier_secondary.clone(),
            microbial_vitality: self.microbial_vitality.clone(),
            microbial_dormancy: self.microbial_dormancy.clone(),
            microbial_reserve: self.microbial_reserve.clone(),
            microbial_packet_mutation_flux: self.microbial_packet_mutation_flux.clone(),
            nitrifier_cells: self.nitrifier_cells.clone(),
            nitrifier_packets: self.nitrifier_packets.clone(),
            nitrifier_aerobic_packets: self.nitrifier_aerobic_packets.clone(),
            nitrifier_aerobic_fraction: self.nitrifier_aerobic_fraction.clone(),
            nitrifier_strain_oxygen_affinity: self.nitrifier_strain_oxygen_affinity.clone(),
            nitrifier_strain_ammonium_affinity: self.nitrifier_strain_ammonium_affinity.clone(),
            nitrifier_latent_packets: self.nitrifier_latent_packets.clone(),
            nitrifier_latent_strain_oxygen_affinity: self
                .nitrifier_latent_strain_oxygen_affinity
                .clone(),
            nitrifier_latent_strain_ammonium_affinity: self
                .nitrifier_latent_strain_ammonium_affinity
                .clone(),
            nitrifier_vitality: self.nitrifier_vitality.clone(),
            nitrifier_dormancy: self.nitrifier_dormancy.clone(),
            nitrifier_reserve: self.nitrifier_reserve.clone(),
            nitrifier_packet_mutation_flux: self.nitrifier_packet_mutation_flux.clone(),
            nitrification_potential: self.nitrification_potential.clone(),
            denitrifier_cells: self.denitrifier_cells.clone(),
            denitrifier_packets: self.denitrifier_packets.clone(),
            denitrifier_anoxic_packets: self.denitrifier_anoxic_packets.clone(),
            denitrifier_anoxic_fraction: self.denitrifier_anoxic_fraction.clone(),
            denitrifier_strain_anoxia_affinity: self.denitrifier_strain_anoxia_affinity.clone(),
            denitrifier_strain_nitrate_affinity: self.denitrifier_strain_nitrate_affinity.clone(),
            denitrifier_latent_packets: self.denitrifier_latent_packets.clone(),
            denitrifier_latent_strain_anoxia_affinity: self
                .denitrifier_latent_strain_anoxia_affinity
                .clone(),
            denitrifier_latent_strain_nitrate_affinity: self
                .denitrifier_latent_strain_nitrate_affinity
                .clone(),
            denitrifier_vitality: self.denitrifier_vitality.clone(),
            denitrifier_dormancy: self.denitrifier_dormancy.clone(),
            denitrifier_reserve: self.denitrifier_reserve.clone(),
            denitrifier_packet_mutation_flux: self.denitrifier_packet_mutation_flux.clone(),
            denitrification_potential: self.denitrification_potential.clone(),
            explicit_microbes: self
                .explicit_microbes
                .iter()
                .map(TerrariumExplicitMicrobe::checkpoint_state)
                .collect::<Result<Vec<_>, _>>()?,
            next_microbe_idx: self.next_microbe_idx,
            packet_populations: self.packet_populations.clone(),
            atomistic_probes: self
                .atomistic_probes
                .iter()
                .map(AtomisticProbe::checkpoint_state)
                .collect(),
            fly_food_total: self.fly_food_total,
            fly_metabolisms: self.fly_metabolisms.clone(),
            fly_population: self.fly_pop.checkpoint(),
            earthworm_population: self.earthworm_population.clone(),
            nematode_guilds: self.nematode_guilds.clone(),
            air_pressure_kpa: self.air_pressure_kpa.clone(),
            substep_counter: self.substep_counter,
            snapshot: self.snapshot(),
            limitations: TerrariumCheckpointLimitations::from_world(self),
        })
    }

    pub fn from_checkpoint(checkpoint: TerrariumWorldCheckpoint) -> Result<Self, String> {
        let mut world = Self::new(checkpoint.config.clone())?;
        world.config = checkpoint.config;
        let mut restored_seed_provenance = checkpoint.seed_provenance;
        restored_seed_provenance.source = TerrariumSeedSource::CheckpointRestore;
        restored_seed_provenance.source_label = format!(
            "{} -> checkpoint-restore",
            restored_seed_provenance.source_label
        );
        world.seed_provenance = restored_seed_provenance;
        world.time_s = checkpoint.time_s;
        world.rng = checkpoint
            .rng
            .unwrap_or_else(|| ChaCha12Rng::seed_from_u64(checkpoint.rng_resume_seed.unwrap_or(0)));
        world.atmosphere_rng_state = checkpoint.atmosphere_rng_state;
        world.chronobiology_config = checkpoint.chronobiology_config;
        world.climate_driver = checkpoint.climate_driver;
        world.weather = checkpoint.weather;
        world.next_organism_id = checkpoint.next_organism_id;
        world.organism_registry = checkpoint.organism_registry;
        world.organism_phylogeny = checkpoint.organism_phylogeny;
        world.odorant_params = checkpoint.odorant_params;
        world.odorants = checkpoint.odorants;
        world.temperature = checkpoint.temperature;
        world.humidity = checkpoint.humidity;
        world.wind_x = checkpoint.wind_x;
        world.wind_y = checkpoint.wind_y;
        world.wind_z = checkpoint.wind_z;
        world.substrate = BatchedAtomTerrarium::from_checkpoint(
            &checkpoint.substrate,
            world.config.use_gpu_substrate,
        );
        world.waters = checkpoint.waters;
        world.fruits = checkpoint.fruits;
        world.plants = checkpoint.plants;
        world.seeds = checkpoint.seeds;
        world.fly_identities = checkpoint.fly_identities;
        world.flies = checkpoint
            .flies
            .into_iter()
            .map(crate::drosophila::DrosophilaSim::from_checkpoint_state)
            .collect();
        world.canopy_cover = checkpoint.canopy_cover;
        world.root_density = checkpoint.root_density;
        world.water_mask = checkpoint.water_mask;
        world.moisture = checkpoint.moisture;
        world.deep_moisture = checkpoint.deep_moisture;
        world.soil_layer_moisture = checkpoint.soil_layer_moisture;
        world.soil_layer_nitrogen = checkpoint.soil_layer_nitrogen;
        world.dissolved_nutrients = checkpoint.dissolved_nutrients;
        world.mineral_nitrogen = checkpoint.mineral_nitrogen;
        world.shallow_nutrients = checkpoint.shallow_nutrients;
        world.deep_minerals = checkpoint.deep_minerals;
        world.organic_matter = checkpoint.organic_matter;
        world.litter_carbon = checkpoint.litter_carbon;
        world.litter_surface = checkpoint.litter_surface;
        world.microbial_biomass = checkpoint.microbial_biomass;
        world.symbiont_biomass = checkpoint.symbiont_biomass;
        world.root_exudates = checkpoint.root_exudates;
        world.soil_structure = checkpoint.soil_structure;
        world.microbial_cells = checkpoint.microbial_cells;
        world.microbial_packets = checkpoint.microbial_packets;
        world.microbial_copiotroph_packets = checkpoint.microbial_copiotroph_packets;
        world.microbial_copiotroph_fraction = checkpoint.microbial_copiotroph_fraction;
        world.microbial_strain_yield = checkpoint.microbial_strain_yield;
        world.microbial_strain_stress_tolerance = checkpoint.microbial_strain_stress_tolerance;
        world.microbial_latent_packets = checkpoint.microbial_latent_packets;
        world.microbial_latent_strain_yield = checkpoint.microbial_latent_strain_yield;
        world.microbial_latent_strain_stress_tolerance =
            checkpoint.microbial_latent_strain_stress_tolerance;
        world.next_probe_id = checkpoint.next_probe_id;
        world.ownership = checkpoint.ownership;
        world.microbial_secondary = checkpoint.microbial_secondary;
        world.nitrifier_secondary = checkpoint.nitrifier_secondary;
        world.denitrifier_secondary = checkpoint.denitrifier_secondary;
        world.microbial_vitality = checkpoint.microbial_vitality;
        world.microbial_dormancy = checkpoint.microbial_dormancy;
        world.microbial_reserve = checkpoint.microbial_reserve;
        world.microbial_packet_mutation_flux = checkpoint.microbial_packet_mutation_flux;
        world.nitrifier_cells = checkpoint.nitrifier_cells;
        world.nitrifier_packets = checkpoint.nitrifier_packets;
        world.nitrifier_aerobic_packets = checkpoint.nitrifier_aerobic_packets;
        world.nitrifier_aerobic_fraction = checkpoint.nitrifier_aerobic_fraction;
        world.nitrifier_strain_oxygen_affinity = checkpoint.nitrifier_strain_oxygen_affinity;
        world.nitrifier_strain_ammonium_affinity = checkpoint.nitrifier_strain_ammonium_affinity;
        world.nitrifier_latent_packets = checkpoint.nitrifier_latent_packets;
        world.nitrifier_latent_strain_oxygen_affinity =
            checkpoint.nitrifier_latent_strain_oxygen_affinity;
        world.nitrifier_latent_strain_ammonium_affinity =
            checkpoint.nitrifier_latent_strain_ammonium_affinity;
        world.nitrifier_vitality = checkpoint.nitrifier_vitality;
        world.nitrifier_dormancy = checkpoint.nitrifier_dormancy;
        world.nitrifier_reserve = checkpoint.nitrifier_reserve;
        world.nitrifier_packet_mutation_flux = checkpoint.nitrifier_packet_mutation_flux;
        world.nitrification_potential = checkpoint.nitrification_potential;
        world.denitrifier_cells = checkpoint.denitrifier_cells;
        world.denitrifier_packets = checkpoint.denitrifier_packets;
        world.denitrifier_anoxic_packets = checkpoint.denitrifier_anoxic_packets;
        world.denitrifier_anoxic_fraction = checkpoint.denitrifier_anoxic_fraction;
        world.denitrifier_strain_anoxia_affinity = checkpoint.denitrifier_strain_anoxia_affinity;
        world.denitrifier_strain_nitrate_affinity = checkpoint.denitrifier_strain_nitrate_affinity;
        world.denitrifier_latent_packets = checkpoint.denitrifier_latent_packets;
        world.denitrifier_latent_strain_anoxia_affinity =
            checkpoint.denitrifier_latent_strain_anoxia_affinity;
        world.denitrifier_latent_strain_nitrate_affinity =
            checkpoint.denitrifier_latent_strain_nitrate_affinity;
        world.denitrifier_vitality = checkpoint.denitrifier_vitality;
        world.denitrifier_dormancy = checkpoint.denitrifier_dormancy;
        world.denitrifier_reserve = checkpoint.denitrifier_reserve;
        world.denitrifier_packet_mutation_flux = checkpoint.denitrifier_packet_mutation_flux;
        world.denitrification_potential = checkpoint.denitrification_potential;
        world.explicit_microbes = checkpoint
            .explicit_microbes
            .into_iter()
            .map(TerrariumExplicitMicrobe::from_checkpoint_state)
            .collect::<Result<Vec<_>, _>>()?;
        world.next_microbe_idx = checkpoint.next_microbe_idx;
        world.packet_populations = checkpoint.packet_populations;
        world.atomistic_probes = checkpoint
            .atomistic_probes
            .into_iter()
            .map(AtomisticProbe::from_checkpoint_state)
            .collect::<Result<Vec<_>, _>>()?;
        world.fly_food_total = checkpoint.fly_food_total;
        world.fly_metabolisms = checkpoint.fly_metabolisms;
        world.fly_pop = FlyPopulation::from_checkpoint(checkpoint.fly_population);
        world.earthworm_population = checkpoint.earthworm_population;
        world.nematode_guilds = checkpoint.nematode_guilds;
        world.air_pressure_kpa = checkpoint.air_pressure_kpa;
        world.substep_counter = checkpoint.substep_counter;
        world.rebuild_explicit_microbe_fields();
        world.rebuild_ownership_diagnostics();

        Ok(world)
    }

    pub fn save_checkpoint<P: AsRef<Path>>(&mut self, path: P) -> Result<(), String> {
        self.checkpoint()?.save_to_path(path)
    }

    pub fn load_checkpoint<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let checkpoint = TerrariumWorldCheckpoint::load_from_path(path)?;
        Self::from_checkpoint(checkpoint)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn assert_snapshot_observables_close(
        expected: &TerrariumWorldSnapshot,
        observed: &TerrariumWorldSnapshot,
    ) {
        let assert_close = |label: &str, lhs: f64, rhs: f64, tol: f64| {
            // Use relative tolerance for large values, absolute for small values
            let denom = lhs.abs().max(rhs.abs()).max(1.0);
            let rel_diff = (lhs - rhs).abs() / denom;
            assert!(
                rel_diff <= tol,
                "{label} diverged after save/load replay: {lhs} vs {rhs} (rel_diff {rel_diff}, tol {tol})"
            );
        };

        assert_eq!(expected.plants, observed.plants);
        assert_eq!(expected.fruits, observed.fruits);
        assert_eq!(expected.seeds, observed.seeds);
        assert_eq!(expected.flies, observed.flies);
        assert_eq!(expected.fly_population_total, observed.fly_population_total);
        assert_eq!(expected.tracked_organisms, observed.tracked_organisms);

        assert_close(
            "time_s",
            expected.time_s as f64,
            observed.time_s as f64,
            1.0e-6,
        );
        assert_close(
            "food_remaining",
            expected.food_remaining as f64,
            observed.food_remaining as f64,
            1.0e-3,
        );
        assert_close(
            "avg_fly_energy",
            expected.avg_fly_energy as f64,
            observed.avg_fly_energy as f64,
            1.0e-3,
        );
        assert_close(
            "mean_soil_moisture",
            expected.mean_soil_moisture as f64,
            observed.mean_soil_moisture as f64,
            1.0e-3,
        );
        // Atmospheric gas tolerances widened: emergent rate engine recomputes
        // SubstrateRateTable per step from Eyring TST, and f32 accumulation
        // causes ~1e-5 relative divergence on substrate chemistry products.
        assert_close(
            "mean_atmospheric_co2",
            expected.mean_atmospheric_co2 as f64,
            observed.mean_atmospheric_co2 as f64,
            1.0e-3,
        );
        assert_close(
            "mean_atmospheric_o2",
            expected.mean_atmospheric_o2 as f64,
            observed.mean_atmospheric_o2 as f64,
            1.0e-3,
        );
        assert_close(
            "mean_air_pressure_kpa",
            expected.mean_air_pressure_kpa as f64,
            observed.mean_air_pressure_kpa as f64,
            1.0e-6,
        );
        // Carbon conservation tolerance widened to 1e-2: emergent rate engine
        // evaluates Stokes-Einstein/Arrhenius per step, and floating-point
        // accumulation over replay steps causes ~1e-5 relative divergence
        // on totals in the ~30,000 range.
        assert_close(
            "conservation.explicit_domain_total.carbon",
            expected.conservation.explicit_domain_total.carbon,
            observed.conservation.explicit_domain_total.carbon,
            1.0e-2,
        );
        assert_close(
            "conservation.explicit_domain_total.nitrogen",
            expected.conservation.explicit_domain_total.nitrogen,
            observed.conservation.explicit_domain_total.nitrogen,
            1.0e-4,
        );
        assert_close(
            "conservation.explicit_domain_total.water",
            expected.conservation.explicit_domain_total.water,
            observed.conservation.explicit_domain_total.water,
            1.0e-4,
        );
    }

    fn checkpoint_save_load_short_replay_matches_observables_for_preset(
        preset: TerrariumDemoPreset,
    ) {
        let mut world = TerrariumWorld::demo_preset(41, false, preset).unwrap();
        for _ in 0..3 {
            world.step_frame().unwrap();
        }

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be monotonic enough for test paths")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "oneura_checkpoint_{}_{}_{}.json",
            preset.cli_name(),
            std::process::id(),
            unique
        ));

        world.save_checkpoint(&path).unwrap();
        let mut restored = TerrariumWorld::load_checkpoint(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        for _ in 0..3 {
            world.step_frame().unwrap();
            restored.step_frame().unwrap();
        }

        let expected = world.snapshot();
        let observed = restored.snapshot();
        assert_snapshot_observables_close(&expected, &observed);
    }

    #[test]
    fn macro_checkpoint_roundtrip_restores_counts_and_time() {
        let mut world =
            TerrariumWorld::demo_preset(11, false, TerrariumDemoPreset::Demo).unwrap();
        for _ in 0..3 {
            world.step_frame().unwrap();
        }
        let before = world.snapshot();
        let checkpoint = world.checkpoint().unwrap();
        assert_eq!(
            checkpoint.limitations.fidelity,
            TerrariumCheckpointFidelity::MacroEcologyWithExplicitLowerScaleState
        );
        assert!(
            !checkpoint
                .limitations
                .fly_brain_state_reinitialized_on_restore
        );
        assert!(
            !checkpoint
                .limitations
                .stochastic_streams_reseeded_from_resume_seeds
        );
        let restored = TerrariumWorld::from_checkpoint(checkpoint).unwrap();
        let after = restored.snapshot();

        assert_eq!(before.seed_provenance.seed, after.seed_provenance.seed);
        assert_eq!(before.plants, after.plants);
        assert_eq!(before.fruits, after.fruits);
        assert_eq!(before.seeds, after.seeds);
        assert_eq!(before.flies, after.flies);
        assert!((before.time_s - after.time_s).abs() < 1.0e-6);
        assert!((before.mean_soil_moisture - after.mean_soil_moisture).abs() < 1.0e-6);
    }

    #[test]
    fn macro_checkpoint_json_roundtrip_serializes() {
        let mut world =
            TerrariumWorld::demo_preset(17, false, TerrariumDemoPreset::Demo).unwrap();
        world.step_frame().unwrap();

        let checkpoint = world.checkpoint().unwrap();
        let plant_count = checkpoint.plants.len();
        let fruit_count = checkpoint.fruits.len();
        let json = checkpoint.to_json_pretty().unwrap();
        let restored = TerrariumWorldCheckpoint::from_json_str(&json).unwrap();

        assert_eq!(
            restored.seed_provenance.seed,
            checkpoint.seed_provenance.seed
        );
        assert_eq!(restored.plants.len(), plant_count);
        assert_eq!(restored.fruits.len(), fruit_count);
    }

    #[test]
    fn explicit_microbe_checkpoint_roundtrip_preserves_whole_cell_state() {
        let mut world =
            TerrariumWorld::demo_preset(23, false, TerrariumDemoPreset::Demo).unwrap();
        world.explicit_microbes.clear();
        world.packet_populations.clear();
        world.explicit_microbe_authority.fill(0.0);
        world.explicit_microbe_activity.fill(0.0);
        let width = world.config.width;
        let height = world.config.height;
        let flat = (0..width * height)
            .filter(|idx| world.water_mask[*idx] < 0.40)
            .max_by(|a, b| {
                world
                    .shoreline_packet_recruitment_signal(*a)
                    .partial_cmp(&world.shoreline_packet_recruitment_signal(*b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("expected shoreline cell");
        let x = flat % width;
        let y = flat / width;
        for yy in y.saturating_sub(1)..=(y + 1).min(world.config.height.saturating_sub(1)) {
            for xx in x.saturating_sub(1)..=(x + 1).min(world.config.width.saturating_sub(1)) {
                let idx = yy * width + xx;
                world.microbial_packets[idx] = 80.0;
                world.nitrifier_packets[idx] = 20.0;
                world.denitrifier_packets[idx] = 24.0;
                world.microbial_cells[idx] = 40_000.0;
                world.nitrifier_cells[idx] = 16_000.0;
                world.denitrifier_cells[idx] = 18_000.0;
            }
        }
        world.microbial_packets[flat] = 2_450.0;
        world.nitrifier_packets[flat] = 1_280.0;
        world.denitrifier_packets[flat] = 1_310.0;
        world.microbial_cells[flat] = 1.15e6;
        world.nitrifier_cells[flat] = 5.1e5;
        world.denitrifier_cells[flat] = 5.3e5;
        world.microbial_vitality[flat] = 0.91;
        world.nitrifier_vitality[flat] = 0.84;
        world.denitrifier_vitality[flat] = 0.83;
        world.microbial_reserve[flat] = 0.83;
        world.nitrifier_reserve[flat] = 0.77;
        world.denitrifier_reserve[flat] = 0.75;
        world.microbial_packet_mutation_flux[flat] = 0.00018;
        world.nitrifier_packet_mutation_flux[flat] = 0.00015;
        world.denitrifier_packet_mutation_flux[flat] = 0.00016;
        world.nitrification_potential[flat] = 0.00011;
        world.denitrification_potential[flat] = 0.00011;
        world.moisture[flat] = 0.35;
        world.deep_moisture[flat] = 0.47;
        world.recruit_explicit_microbes_from_soil().unwrap();
        assert!(!world.explicit_microbes.is_empty());

        let checkpoint = world.explicit_microbes[0].checkpoint_state().unwrap();
        let restored = TerrariumExplicitMicrobe::from_checkpoint_state(checkpoint).unwrap();

        assert_eq!(restored.x, world.explicit_microbes[0].x);
        assert_eq!(restored.y, world.explicit_microbes[0].y);
        assert_eq!(restored.idx, world.explicit_microbes[0].idx);
        assert_eq!(
            restored.simulator.snapshot().step_count,
            world.explicit_microbes[0].simulator.snapshot().step_count
        );
        assert_eq!(
            restored.simulator.snapshot().genome_bp,
            world.explicit_microbes[0].simulator.snapshot().genome_bp
        );
        assert_eq!(
            restored.last_snapshot.atp_mm,
            world.explicit_microbes[0].last_snapshot.atp_mm
        );
        assert!(
            (restored.material_inventory.total_amount_moles()
                - world.explicit_microbes[0]
                    .material_inventory
                    .total_amount_moles())
            .abs()
                < 1.0e-12
        );
    }

    #[test]
    fn checkpoint_roundtrip_restores_microbe_packet_and_probe_layers() {
        let mut world =
            TerrariumWorld::demo_preset(29, false, TerrariumDemoPreset::Demo).unwrap();
        world.packet_populations.clear();
        world.recruit_explicit_microbes_from_soil().unwrap();
        world.recruit_packet_populations();
        if world.probe_count() == 0 {
            let mol = crate::enzyme_probes::build_tripeptide_gag();
            world.spawn_probe(&mol, 2, 2, 1).unwrap();
        }
        let before = world.snapshot();

        let checkpoint = world.checkpoint().unwrap();
        assert_eq!(checkpoint.limitations.explicit_microbe_count_omitted, 0);
        assert_eq!(checkpoint.limitations.packet_population_count_omitted, 0);
        assert_eq!(checkpoint.limitations.atomistic_probe_count_omitted, 0);
        assert!(!checkpoint.limitations.ownership_map_reinitialized);
        assert!(
            !checkpoint
                .limitations
                .secondary_genotype_banks_reinitialized
        );

        let restored = TerrariumWorld::from_checkpoint(checkpoint).unwrap();
        let after = restored.snapshot();

        assert_eq!(
            before.full_explicit_microbes.len(),
            after.full_explicit_microbes.len()
        );
        assert_eq!(before.atomistic_probes, after.atomistic_probes);
        assert_eq!(
            restored.packet_populations.len(),
            world.packet_populations.len()
        );
        assert_eq!(restored.ownership.len(), world.ownership.len());
        assert_eq!(restored.next_probe_id, world.next_probe_id);
        assert_eq!(restored.next_microbe_idx, world.next_microbe_idx);
        assert_eq!(
            restored.atomistic_probes[0].md.positions(),
            world.atomistic_probes[0].md.positions()
        );
    }

    #[test]
    fn checkpoint_roundtrip_preserves_next_stochastic_world_step() {
        let mut world =
            TerrariumWorld::demo_preset(31, false, TerrariumDemoPreset::Demo).unwrap();
        world.step_frame().unwrap();
        let checkpoint = world.checkpoint().unwrap();
        let mut restored = TerrariumWorld::from_checkpoint(checkpoint).unwrap();

        world.step_frame().unwrap();
        restored.step_frame().unwrap();

        let expected = world.snapshot();
        let observed = restored.snapshot();
        assert_eq!(expected.plants, observed.plants);
        assert_eq!(expected.fruits, observed.fruits);
        assert_eq!(expected.seeds, observed.seeds);
        assert_eq!(expected.flies, observed.flies);
        assert_eq!(
            world.fly_pop.checkpoint().flies.len(),
            restored.fly_pop.checkpoint().flies.len()
        );
        assert_eq!(
            world.fly_pop.checkpoint().egg_clusters.len(),
            restored.fly_pop.checkpoint().egg_clusters.len()
        );
        assert_eq!(
            world.fly_pop.checkpoint().next_id,
            restored.fly_pop.checkpoint().next_id
        );
    }

    #[test]
    fn checkpoint_save_load_short_replay_matches_observables_for_terrarium() {
        checkpoint_save_load_short_replay_matches_observables_for_preset(
            TerrariumDemoPreset::Demo,
        );
    }

    #[test]
    fn checkpoint_save_load_short_replay_matches_observables_for_aquarium() {
        checkpoint_save_load_short_replay_matches_observables_for_preset(
            TerrariumDemoPreset::MicroAquarium,
        );
    }
}
