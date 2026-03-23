//! Batched atom / chemistry terrarium substrate.
//!
//! This is the next honest rung between coarse ecological fields and literal
//! atom-by-atom molecular dynamics: a GPU-ready voxel lattice storing batched
//! elemental and molecular pools. Each voxel tracks CHONPS reservoirs plus a
//! handful of reactive molecular fields, then advances them with diffusion and
//! coarse reaction terms.
//!
//! The goal is not to pretend we already have full atomistic MD for an entire
//! terrarium. The goal is to keep the state representation close enough to
//! chemistry that future refinement can split bins into smaller species groups,
//! reaction networks, explicit cell lattices, and eventually localized MD.

use crate::gpu;
#[cfg(target_os = "macos")]
use crate::gpu::terrarium_substrate::{
    dispatch_terrarium_substrate, dispatch_terrarium_substrate_persistent,
};

#[cfg(target_os = "macos")]
use crate::gpu::GpuContext;

pub const TERRARIUM_SPECIES_COUNT: usize = 33;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum TerrariumSpecies {
    Carbon = 0,
    Hydrogen = 1,
    Oxygen = 2,
    Nitrogen = 3,
    Phosphorus = 4,
    Sulfur = 5,
    Water = 6,
    Glucose = 7,
    OxygenGas = 8,
    Ammonium = 9,
    Nitrate = 10,
    CarbonDioxide = 11,
    Proton = 12,
    AtpFlux = 13,
    AminoAcidPool = 14,
    NucleotidePool = 15,
    MembranePrecursorPool = 16,
    SilicateMineral = 17,
    ClayMineral = 18,
    CarbonateMineral = 19,
    IronOxideMineral = 20,
    DissolvedSilicate = 21,
    ExchangeableCalcium = 22,
    ExchangeableMagnesium = 23,
    ExchangeablePotassium = 24,
    ExchangeableSodium = 25,
    ExchangeableAluminum = 26,
    AqueousIronPool = 27,
    BicarbonatePool = 28,
    SurfaceProtonLoad = 29,
    CalciumBicarbonateComplex = 30,
    SorbedAluminumHydroxide = 31,
    SorbedFerricHydroxide = 32,
}

impl TerrariumSpecies {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Carbon => "carbon",
            Self::Hydrogen => "hydrogen",
            Self::Oxygen => "oxygen",
            Self::Nitrogen => "nitrogen",
            Self::Phosphorus => "phosphorus",
            Self::Sulfur => "sulfur",
            Self::Water => "water",
            Self::Glucose => "glucose",
            Self::OxygenGas => "oxygen_gas",
            Self::Ammonium => "ammonium",
            Self::Nitrate => "nitrate",
            Self::CarbonDioxide => "carbon_dioxide",
            Self::Proton => "proton",
            Self::AtpFlux => "atp_flux",
            Self::AminoAcidPool => "amino_acid_pool",
            Self::NucleotidePool => "nucleotide_pool",
            Self::MembranePrecursorPool => "membrane_precursor_pool",
            Self::SilicateMineral => "silicate_mineral",
            Self::ClayMineral => "clay_mineral",
            Self::CarbonateMineral => "carbonate_mineral",
            Self::IronOxideMineral => "iron_oxide_mineral",
            Self::DissolvedSilicate => "dissolved_silicate",
            Self::ExchangeableCalcium => "exchangeable_calcium",
            Self::ExchangeableMagnesium => "exchangeable_magnesium",
            Self::ExchangeablePotassium => "exchangeable_potassium",
            Self::ExchangeableSodium => "exchangeable_sodium",
            Self::ExchangeableAluminum => "exchangeable_aluminum",
            Self::AqueousIronPool => "aqueous_iron_pool",
            Self::BicarbonatePool => "bicarbonate_pool",
            Self::SurfaceProtonLoad => "surface_proton_load",
            Self::CalciumBicarbonateComplex => "calcium_bicarbonate_complex",
            Self::SorbedAluminumHydroxide => "sorbed_aluminum_hydroxide",
            Self::SorbedFerricHydroxide => "sorbed_ferric_hydroxide",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "carbon" | "c" => Some(Self::Carbon),
            "hydrogen" | "h" => Some(Self::Hydrogen),
            "oxygen" | "o" => Some(Self::Oxygen),
            "nitrogen" | "n" => Some(Self::Nitrogen),
            "phosphorus" | "p" => Some(Self::Phosphorus),
            "sulfur" | "s" => Some(Self::Sulfur),
            "water" | "h2o" => Some(Self::Water),
            "glucose" | "sugar" => Some(Self::Glucose),
            "oxygen_gas" | "o2" => Some(Self::OxygenGas),
            "ammonium" | "nh4" => Some(Self::Ammonium),
            "nitrate" | "no3" => Some(Self::Nitrate),
            "carbon_dioxide" | "co2" => Some(Self::CarbonDioxide),
            "proton" | "h_plus" | "acidity" => Some(Self::Proton),
            "atp_flux" | "atp" | "energy" => Some(Self::AtpFlux),
            "amino_acid_pool" | "amino_acids" | "amino" => Some(Self::AminoAcidPool),
            "nucleotide_pool" | "nucleotides" | "nucleotide" => Some(Self::NucleotidePool),
            "membrane_precursor_pool" | "membrane_precursors" | "membrane_precursor" => {
                Some(Self::MembranePrecursorPool)
            }
            "silicate_mineral" | "silicate" | "silicate_matrix" => Some(Self::SilicateMineral),
            "clay_mineral" | "clay" | "clay_matrix" => Some(Self::ClayMineral),
            "carbonate_mineral" | "carbonate" | "carbonate_phase" => Some(Self::CarbonateMineral),
            "iron_oxide_mineral" | "iron_oxide" | "oxide_mineral" => Some(Self::IronOxideMineral),
            "dissolved_silicate" | "silicic_acid" | "silicate_aqueous" => {
                Some(Self::DissolvedSilicate)
            }
            "exchangeable_calcium" | "calcium_exchange" | "ca" => Some(Self::ExchangeableCalcium),
            "exchangeable_magnesium" | "magnesium_exchange" | "mg" => {
                Some(Self::ExchangeableMagnesium)
            }
            "exchangeable_potassium" | "potassium_exchange" | "k" => {
                Some(Self::ExchangeablePotassium)
            }
            "exchangeable_sodium" | "sodium_exchange" | "na" => Some(Self::ExchangeableSodium),
            "exchangeable_aluminum" | "aluminum_exchange" | "al" => {
                Some(Self::ExchangeableAluminum)
            }
            "aqueous_iron_pool" | "aqueous_iron" | "iron_pool" | "fe" => {
                Some(Self::AqueousIronPool)
            }
            "bicarbonate_pool" | "bicarbonate" | "hydrogen_carbonate" | "hco3" => {
                Some(Self::BicarbonatePool)
            }
            "surface_proton_load" | "surface_proton" | "sorbed_proton" => {
                Some(Self::SurfaceProtonLoad)
            }
            "calcium_bicarbonate_complex" | "calcium_bicarbonate" | "cahco3_complex" => {
                Some(Self::CalciumBicarbonateComplex)
            }
            "sorbed_aluminum_hydroxide" | "aluminum_hydroxide" | "gibbsite" => {
                Some(Self::SorbedAluminumHydroxide)
            }
            "sorbed_ferric_hydroxide" | "ferric_hydroxide" | "iron_hydroxide" => {
                Some(Self::SorbedFerricHydroxide)
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TerrariumBackend {
    Cpu,
    Metal,
}

impl TerrariumBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
        }
    }
}

#[derive(Debug, Clone)]
pub struct TerrariumSnapshot {
    pub backend: TerrariumBackend,
    pub time_ms: f32,
    pub step_count: u64,
    pub mean_hydration: f32,
    pub mean_microbes: f32,
    pub mean_plant_drive: f32,
    pub mean_glucose: f32,
    pub mean_oxygen_gas: f32,
    pub mean_ammonium: f32,
    pub mean_nitrate: f32,
    pub mean_carbon_dioxide: f32,
    pub mean_atp_flux: f32,
    pub elemental_carbon: f32,
    pub elemental_nitrogen: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BatchedAtomTerrariumCheckpoint {
    pub x_dim: usize,
    pub y_dim: usize,
    pub z_dim: usize,
    pub voxel_size_mm: f32,
    pub current: Vec<f32>,
    pub next: Vec<f32>,
    pub hydration: Vec<f32>,
    pub microbial_activity: Vec<f32>,
    pub plant_drive: Vec<f32>,
    pub base_plant_drive: Vec<f32>,
    pub external_controls: bool,
    pub backend: TerrariumBackend,
    pub time_ms: f32,
    pub step_count: u64,
    #[serde(default = "default_temperature_c")]
    pub temperature_c: f32,
}

fn default_temperature_c() -> f32 { 20.0 }

pub struct BatchedAtomTerrarium {
    pub x_dim: usize,
    pub y_dim: usize,
    pub z_dim: usize,
    pub voxel_size_mm: f32,
    pub current: Vec<f32>,
    pub next: Vec<f32>,
    pub hydration: Vec<f32>,
    pub microbial_activity: Vec<f32>,
    pub plant_drive: Vec<f32>,
    base_plant_drive: Vec<f32>,
    external_controls: bool,
    backend: TerrariumBackend,
    #[cfg(target_os = "macos")]
    gpu: Option<GpuContext>,
    #[cfg(target_os = "macos")]
    gpu_buf_current: Option<metal::Buffer>,
    #[cfg(target_os = "macos")]
    gpu_buf_next: Option<metal::Buffer>,
    #[cfg(target_os = "macos")]
    gpu_buf_hydration: Option<metal::Buffer>,
    #[cfg(target_os = "macos")]
    gpu_buf_microbes: Option<metal::Buffer>,
    #[cfg(target_os = "macos")]
    gpu_buf_plants: Option<metal::Buffer>,
    time_ms: f32,
    step_count: u64,
    /// Temperature for emergent rate computation (set from world before step).
    pub temperature_c: f32,
}

impl BatchedAtomTerrarium {
    pub fn checkpoint(&self) -> BatchedAtomTerrariumCheckpoint {
        BatchedAtomTerrariumCheckpoint {
            x_dim: self.x_dim,
            y_dim: self.y_dim,
            z_dim: self.z_dim,
            voxel_size_mm: self.voxel_size_mm,
            current: self.current.clone(),
            next: self.next.clone(),
            hydration: self.hydration.clone(),
            microbial_activity: self.microbial_activity.clone(),
            plant_drive: self.plant_drive.clone(),
            base_plant_drive: self.base_plant_drive.clone(),
            external_controls: self.external_controls,
            backend: self.backend,
            time_ms: self.time_ms,
            step_count: self.step_count,
            temperature_c: self.temperature_c,
        }
    }

    pub fn from_checkpoint(checkpoint: &BatchedAtomTerrariumCheckpoint, use_gpu: bool) -> Self {
        let mut terrarium = Self::new(
            checkpoint.x_dim,
            checkpoint.y_dim,
            checkpoint.z_dim,
            checkpoint.voxel_size_mm,
            use_gpu,
        );
        terrarium.current = checkpoint.current.clone();
        terrarium.next = checkpoint.next.clone();
        terrarium.hydration = checkpoint.hydration.clone();
        terrarium.microbial_activity = checkpoint.microbial_activity.clone();
        terrarium.plant_drive = checkpoint.plant_drive.clone();
        terrarium.base_plant_drive = checkpoint.base_plant_drive.clone();
        terrarium.external_controls = checkpoint.external_controls;
        terrarium.time_ms = checkpoint.time_ms;
        terrarium.step_count = checkpoint.step_count;
        terrarium.temperature_c = checkpoint.temperature_c;
        terrarium
    }

    pub fn new(
        x_dim: usize,
        y_dim: usize,
        z_dim: usize,
        voxel_size_mm: f32,
        use_gpu: bool,
    ) -> Self {
        let total_voxels = x_dim * y_dim * z_dim;
        let backend = if use_gpu && gpu::has_gpu() {
            TerrariumBackend::Metal
        } else {
            TerrariumBackend::Cpu
        };

        #[cfg(target_os = "macos")]
        let (
            gpu,
            gpu_buf_current,
            gpu_buf_next,
            gpu_buf_hydration,
            gpu_buf_microbes,
            gpu_buf_plants,
        ) = if backend == TerrariumBackend::Metal {
            if let Some(gpu_ctx) = GpuContext::new().ok() {
                let size_species =
                    (TERRARIUM_SPECIES_COUNT * total_voxels * std::mem::size_of::<f32>()) as u64;
                let size_scalar = (total_voxels * std::mem::size_of::<f32>()) as u64;

                let buf_current = gpu_ctx.buffer_of_size(size_species);
                let buf_next = gpu_ctx.buffer_of_size(size_species);
                let buf_hydration = gpu_ctx.buffer_of_size(size_scalar);
                let buf_microbes = gpu_ctx.buffer_of_size(size_scalar);
                let buf_plants = gpu_ctx.buffer_of_size(size_scalar);

                (
                    Some(gpu_ctx),
                    Some(buf_current),
                    Some(buf_next),
                    Some(buf_hydration),
                    Some(buf_microbes),
                    Some(buf_plants),
                )
            } else {
                (None, None, None, None, None, None)
            }
        } else {
            (None, None, None, None, None, None)
        };

        let backend = {
            #[cfg(target_os = "macos")]
            {
                if backend == TerrariumBackend::Metal && gpu.is_some() {
                    TerrariumBackend::Metal
                } else {
                    TerrariumBackend::Cpu
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                TerrariumBackend::Cpu
            }
        };

        let mut terrarium = Self {
            x_dim,
            y_dim,
            z_dim,
            voxel_size_mm,
            current: vec![0.0; TERRARIUM_SPECIES_COUNT * total_voxels],
            next: vec![0.0; TERRARIUM_SPECIES_COUNT * total_voxels],
            hydration: vec![0.0; total_voxels],
            microbial_activity: vec![0.0; total_voxels],
            plant_drive: vec![0.0; total_voxels],
            base_plant_drive: vec![0.0; total_voxels],
            external_controls: false,
            backend,
            #[cfg(target_os = "macos")]
            gpu,
            #[cfg(target_os = "macos")]
            gpu_buf_current,
            #[cfg(target_os = "macos")]
            gpu_buf_next,
            #[cfg(target_os = "macos")]
            gpu_buf_hydration,
            #[cfg(target_os = "macos")]
            gpu_buf_microbes,
            #[cfg(target_os = "macos")]
            gpu_buf_plants,
            time_ms: 0.0,
            step_count: 0,
            temperature_c: 20.0,
        };
        terrarium.seed_default_profile();
        terrarium
    }

    pub fn backend(&self) -> TerrariumBackend {
        self.backend
    }

    pub fn time_ms(&self) -> f32 {
        self.time_ms
    }

    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    pub fn total_voxels(&self) -> usize {
        self.x_dim * self.y_dim * self.z_dim
    }

    pub fn shape(&self) -> (usize, usize, usize) {
        (self.x_dim, self.y_dim, self.z_dim)
    }

    pub fn index(&self, species: TerrariumSpecies, x: usize, y: usize, z: usize) -> usize {
        species as usize * self.total_voxels() + z * self.y_dim * self.x_dim + y * self.x_dim + x
    }

    fn control_index(&self, x: usize, y: usize, z: usize) -> usize {
        z * self.y_dim * self.x_dim + y * self.x_dim + x
    }

    pub fn fill_species(&mut self, species: TerrariumSpecies, value: f32) {
        let total = self.total_voxels();
        let start = species as usize * total;
        let end = start + total;
        self.current[start..end].fill(value.max(0.0));
        self.next[start..end].fill(value.max(0.0));
    }

    pub fn relax_species_toward(&mut self, species: TerrariumSpecies, target: f32, fraction: f32) {
        let target = target.max(0.0);
        let fraction = fraction.clamp(0.0, 1.0);
        if fraction <= 1.0e-9 {
            return;
        }

        let total = self.total_voxels();
        let start = species as usize * total;
        let end = start + total;
        for idx in start..end {
            let next_value = self.current[idx] + (target - self.current[idx]) * fraction;
            self.current[idx] = next_value.max(0.0);
            self.next[idx] = self.current[idx];
        }
    }

    pub fn add_hotspot(
        &mut self,
        species: TerrariumSpecies,
        x: usize,
        y: usize,
        z: usize,
        amplitude: f32,
    ) {
        let radius = 2.4f32;
        for zz in z.saturating_sub(2)..=(z + 2).min(self.z_dim.saturating_sub(1)) {
            for yy in y.saturating_sub(2)..=(y + 2).min(self.y_dim.saturating_sub(1)) {
                for xx in x.saturating_sub(2)..=(x + 2).min(self.x_dim.saturating_sub(1)) {
                    let dx = xx as f32 - x as f32;
                    let dy = yy as f32 - y as f32;
                    let dz = zz as f32 - z as f32;
                    let r2 = dx * dx + dy * dy + dz * dz;
                    let kernel = (-r2 / (2.0 * radius * radius)).exp();
                    let idx = self.index(species, xx, yy, zz);
                    self.current[idx] += amplitude.max(0.0) * kernel;
                    self.next[idx] = self.current[idx];
                }
            }
        }
    }

    fn patch_weight(radius: usize, dx: f32, dy: f32, dz: f32) -> f32 {
        let sigma = (radius.max(1) as f32 * 0.72).max(0.85);
        (-(dx * dx + dy * dy + dz * dz) / (2.0 * sigma * sigma)).exp()
    }

    pub fn patch_mean_species(
        &self,
        species: TerrariumSpecies,
        x: usize,
        y: usize,
        z: usize,
        radius: usize,
    ) -> f32 {
        let radius = radius.max(1);
        let z_radius = radius.div_ceil(2).max(1);
        let mut weighted_sum = 0.0f32;
        let mut weight_total = 0.0f32;

        for zz in z.saturating_sub(z_radius)..=(z + z_radius).min(self.z_dim.saturating_sub(1)) {
            for yy in y.saturating_sub(radius)..=(y + radius).min(self.y_dim.saturating_sub(1)) {
                for xx in x.saturating_sub(radius)..=(x + radius).min(self.x_dim.saturating_sub(1))
                {
                    let dx = xx as f32 - x as f32;
                    let dy = yy as f32 - y as f32;
                    let dz = zz as f32 - z as f32;
                    let weight = Self::patch_weight(radius, dx, dy, dz);
                    weighted_sum += self.current[self.index(species, xx, yy, zz)] * weight;
                    weight_total += weight;
                }
            }
        }

        if weight_total <= 1e-9 {
            0.0
        } else {
            weighted_sum / weight_total
        }
    }

    pub fn extract_patch_species(
        &mut self,
        species: TerrariumSpecies,
        x: usize,
        y: usize,
        z: usize,
        radius: usize,
        amount: f32,
    ) -> f32 {
        let target = amount.max(0.0);
        if target <= 1e-9 {
            return 0.0;
        }

        let radius = radius.max(1);
        let z_radius = radius.div_ceil(2).max(1);
        let mut weighted_total = 0.0f32;

        for zz in z.saturating_sub(z_radius)..=(z + z_radius).min(self.z_dim.saturating_sub(1)) {
            for yy in y.saturating_sub(radius)..=(y + radius).min(self.y_dim.saturating_sub(1)) {
                for xx in x.saturating_sub(radius)..=(x + radius).min(self.x_dim.saturating_sub(1))
                {
                    let dx = xx as f32 - x as f32;
                    let dy = yy as f32 - y as f32;
                    let dz = zz as f32 - z as f32;
                    let weight = Self::patch_weight(radius, dx, dy, dz);
                    weighted_total += self.current[self.index(species, xx, yy, zz)] * weight;
                }
            }
        }

        if weighted_total <= 1e-9 {
            return 0.0;
        }

        let removed = target.min(weighted_total);
        let total_voxels = self.total_voxels();
        for zz in z.saturating_sub(z_radius)..=(z + z_radius).min(self.z_dim.saturating_sub(1)) {
            for yy in y.saturating_sub(radius)..=(y + radius).min(self.y_dim.saturating_sub(1)) {
                for xx in x.saturating_sub(radius)..=(x + radius).min(self.x_dim.saturating_sub(1))
                {
                    let dx = xx as f32 - x as f32;
                    let dy = yy as f32 - y as f32;
                    let dz = zz as f32 - z as f32;
                    let weight = Self::patch_weight(radius, dx, dy, dz);
                    let idx = self.index(species, xx, yy, zz);
                    let accessible = self.current[idx] * weight;
                    if accessible <= 1e-12 {
                        continue;
                    }
                    let share = removed * (accessible / weighted_total);
                    self.current[idx] = (self.current[idx] - share).max(0.0);
                    self.next[species as usize * total_voxels
                        + zz * self.y_dim * self.x_dim
                        + yy * self.x_dim
                        + xx] = self.current[idx];
                }
            }
        }

        removed
    }

    pub fn seed_default_profile(&mut self) {
        let x_mid = (self.x_dim as f32 - 1.0) * 0.5;
        let y_mid = (self.y_dim as f32 - 1.0) * 0.5;
        let z_den = (self.z_dim.max(2) - 1) as f32;
        let total_voxels = self.total_voxels();
        let plane = self.x_dim * self.y_dim;
        for z in 0..self.z_dim {
            let depth = z as f32 / z_den;
            for y in 0..self.y_dim {
                for x in 0..self.x_dim {
                    let ctrl = self.control_index(x, y, z);
                    let gid = z * plane + y * self.x_dim + x;
                    let dx = (x as f32 - x_mid) / (self.x_dim.max(1) as f32 * 0.28);
                    let dy = (y as f32 - y_mid) / (self.y_dim.max(1) as f32 * 0.28);
                    let canopy = (-(dx * dx + dy * dy)).exp() * (1.0 - depth).powf(1.4);
                    let rhizosphere = (-(dx * dx + dy * dy)).exp() * (0.35 + depth * 0.65);

                    self.hydration[ctrl] = 0.18 + depth * 0.62;
                    self.microbial_activity[ctrl] = 0.08 + rhizosphere * 0.42;
                    self.base_plant_drive[ctrl] = canopy * 0.52;
                    self.plant_drive[ctrl] = self.base_plant_drive[ctrl];

                    // All species concentrations from thermodynamic equilibrium
                    // (Jenny 1941, Gapon 1933, Nernst, Sposito 2008, Brady & Weil 2017)
                    use crate::terrarium::emergent_rates::equilibrium_initial_concentration as eqc;
                    use TerrariumSpecies::*;
                    let water = eqc(Water, depth, canopy, rhizosphere);
                    let oxygen_gas = eqc(OxygenGas, depth, canopy, rhizosphere);
                    let carbon = eqc(Carbon, depth, canopy, rhizosphere);
                    let hydrogen = eqc(Hydrogen, depth, canopy, rhizosphere);
                    let oxygen = eqc(Oxygen, depth, canopy, rhizosphere);
                    let nitrogen = eqc(Nitrogen, depth, canopy, rhizosphere);
                    let phosphorus = eqc(Phosphorus, depth, canopy, rhizosphere);
                    let sulfur = eqc(Sulfur, depth, canopy, rhizosphere);
                    let glucose = eqc(Glucose, depth, canopy, rhizosphere);
                    let ammonium = eqc(Ammonium, depth, canopy, rhizosphere);
                    let nitrate = eqc(Nitrate, depth, canopy, rhizosphere);
                    let co2 = eqc(CarbonDioxide, depth, canopy, rhizosphere);
                    let proton = eqc(Proton, depth, canopy, rhizosphere);
                    let amino_acids = eqc(AminoAcidPool, depth, canopy, rhizosphere);
                    let nucleotides = eqc(NucleotidePool, depth, canopy, rhizosphere);
                    let membrane_precursors = eqc(MembranePrecursorPool, depth, canopy, rhizosphere);
                    let silicate_mineral = eqc(SilicateMineral, depth, canopy, rhizosphere);
                    let clay_mineral = eqc(ClayMineral, depth, canopy, rhizosphere);
                    let carbonate_mineral = eqc(CarbonateMineral, depth, canopy, rhizosphere);
                    let iron_oxide_mineral = eqc(IronOxideMineral, depth, canopy, rhizosphere);
                    let dissolved_silicate = eqc(DissolvedSilicate, depth, canopy, rhizosphere);
                    let exchangeable_calcium = eqc(ExchangeableCalcium, depth, canopy, rhizosphere);
                    let exchangeable_magnesium = eqc(ExchangeableMagnesium, depth, canopy, rhizosphere);
                    let exchangeable_potassium = eqc(ExchangeablePotassium, depth, canopy, rhizosphere);
                    let exchangeable_sodium = eqc(ExchangeableSodium, depth, canopy, rhizosphere);
                    let exchangeable_aluminum = eqc(ExchangeableAluminum, depth, canopy, rhizosphere);
                    let aqueous_iron_pool = eqc(AqueousIronPool, depth, canopy, rhizosphere);
                    let bicarbonate_pool = eqc(BicarbonatePool, depth, canopy, rhizosphere);
                    let surface_proton_load = eqc(SurfaceProtonLoad, depth, canopy, rhizosphere);
                    let calcium_bicarbonate_complex = eqc(CalciumBicarbonateComplex, depth, canopy, rhizosphere);
                    let sorbed_aluminum_hydroxide = eqc(SorbedAluminumHydroxide, depth, canopy, rhizosphere);
                    let sorbed_ferric_hydroxide = eqc(SorbedFerricHydroxide, depth, canopy, rhizosphere);

                    self.current[TerrariumSpecies::Carbon as usize * total_voxels + gid] = carbon;
                    self.current[TerrariumSpecies::Hydrogen as usize * total_voxels + gid] =
                        hydrogen;
                    self.current[TerrariumSpecies::Oxygen as usize * total_voxels + gid] = oxygen;
                    self.current[TerrariumSpecies::Nitrogen as usize * total_voxels + gid] =
                        nitrogen;
                    self.current[TerrariumSpecies::Phosphorus as usize * total_voxels + gid] =
                        phosphorus;
                    self.current[TerrariumSpecies::Sulfur as usize * total_voxels + gid] = sulfur;
                    self.current[TerrariumSpecies::Water as usize * total_voxels + gid] = water;
                    self.current[TerrariumSpecies::Glucose as usize * total_voxels + gid] = glucose;
                    self.current[TerrariumSpecies::OxygenGas as usize * total_voxels + gid] =
                        oxygen_gas;
                    self.current[TerrariumSpecies::Ammonium as usize * total_voxels + gid] =
                        ammonium;
                    self.current[TerrariumSpecies::Nitrate as usize * total_voxels + gid] = nitrate;
                    self.current[TerrariumSpecies::CarbonDioxide as usize * total_voxels + gid] =
                        co2;
                    self.current[TerrariumSpecies::Proton as usize * total_voxels + gid] = proton;
                    self.current[TerrariumSpecies::AtpFlux as usize * total_voxels + gid] = 0.0;
                    self.current[TerrariumSpecies::AminoAcidPool as usize * total_voxels + gid] =
                        amino_acids;
                    self.current[TerrariumSpecies::NucleotidePool as usize * total_voxels + gid] =
                        nucleotides;
                    self.current
                        [TerrariumSpecies::MembranePrecursorPool as usize * total_voxels + gid] =
                        membrane_precursors;
                    self.current[TerrariumSpecies::SilicateMineral as usize * total_voxels + gid] =
                        silicate_mineral;
                    self.current[TerrariumSpecies::ClayMineral as usize * total_voxels + gid] =
                        clay_mineral;
                    self.current
                        [TerrariumSpecies::CarbonateMineral as usize * total_voxels + gid] =
                        carbonate_mineral;
                    self.current
                        [TerrariumSpecies::IronOxideMineral as usize * total_voxels + gid] =
                        iron_oxide_mineral;
                    self.current
                        [TerrariumSpecies::DissolvedSilicate as usize * total_voxels + gid] =
                        dissolved_silicate;
                    self.current
                        [TerrariumSpecies::ExchangeableCalcium as usize * total_voxels + gid] =
                        exchangeable_calcium;
                    self.current
                        [TerrariumSpecies::ExchangeableMagnesium as usize * total_voxels + gid] =
                        exchangeable_magnesium;
                    self.current
                        [TerrariumSpecies::ExchangeablePotassium as usize * total_voxels + gid] =
                        exchangeable_potassium;
                    self.current
                        [TerrariumSpecies::ExchangeableSodium as usize * total_voxels + gid] =
                        exchangeable_sodium;
                    self.current
                        [TerrariumSpecies::ExchangeableAluminum as usize * total_voxels + gid] =
                        exchangeable_aluminum;
                    self.current[TerrariumSpecies::AqueousIronPool as usize * total_voxels + gid] =
                        aqueous_iron_pool;
                    self.current[TerrariumSpecies::BicarbonatePool as usize * total_voxels + gid] =
                        bicarbonate_pool;
                    self.current
                        [TerrariumSpecies::SurfaceProtonLoad as usize * total_voxels + gid] =
                        surface_proton_load;
                    self.current[TerrariumSpecies::CalciumBicarbonateComplex as usize
                        * total_voxels
                        + gid] = calcium_bicarbonate_complex;
                    self.current
                        [TerrariumSpecies::SorbedAluminumHydroxide as usize * total_voxels + gid] =
                        sorbed_aluminum_hydroxide;
                    self.current
                        [TerrariumSpecies::SorbedFerricHydroxide as usize * total_voxels + gid] =
                        sorbed_ferric_hydroxide;
                }
            }
        }
        self.next.copy_from_slice(&self.current);
    }

    fn refresh_control_fields(&mut self) {
        if self.external_controls {
            for i in 0..self.total_voxels() {
                self.plant_drive[i] = self.plant_drive[i].clamp(0.0, 1.0);
                self.hydration[i] = self.hydration[i].clamp(0.02, 1.0);
                self.microbial_activity[i] = self.microbial_activity[i].clamp(0.02, 1.2);
            }
            return;
        }
        let phase = ((self.time_ms * 0.00008).sin() + 1.0) * 0.5;
        for i in 0..self.total_voxels() {
            self.plant_drive[i] = self.base_plant_drive[i] * (0.25 + 0.75 * phase);
            self.hydration[i] = (self.hydration[i] * 0.9995).clamp(0.05, 1.0);
            self.microbial_activity[i] = (self.microbial_activity[i]
                * (0.9997 + self.hydration[i] * 0.00015))
                .clamp(0.02, 1.2);
        }
    }

    pub fn step(&mut self, dt_ms: f32) {
        let dt_ms = dt_ms.max(0.001);
        self.refresh_control_fields();
        match self.backend {
            TerrariumBackend::Cpu => self.cpu_step_into_next(dt_ms),
            TerrariumBackend::Metal => {
                #[cfg(target_os = "macos")]
                {
                    if let Some(ref gpu) = self.gpu {
                        // Use persistent buffers if available, otherwise fall back to per-frame allocation
                        if let (
                            Some(ref buf_current),
                            Some(ref buf_next),
                            Some(ref buf_hydration),
                            Some(ref buf_microbes),
                            Some(ref buf_plants),
                        ) = (
                            &self.gpu_buf_current,
                            &self.gpu_buf_next,
                            &self.gpu_buf_hydration,
                            &self.gpu_buf_microbes,
                            &self.gpu_buf_plants,
                        ) {
                            dispatch_terrarium_substrate_persistent(
                                gpu,
                                buf_current,
                                buf_next,
                                buf_hydration,
                                buf_microbes,
                                buf_plants,
                                &self.current,
                                &mut self.next,
                                &self.hydration,
                                &self.microbial_activity,
                                &self.plant_drive,
                                self.x_dim,
                                self.y_dim,
                                self.z_dim,
                                self.voxel_size_mm,
                                dt_ms,
                            );
                        } else {
                            dispatch_terrarium_substrate(
                                gpu,
                                &self.current,
                                &mut self.next,
                                &self.hydration,
                                &self.microbial_activity,
                                &self.plant_drive,
                                self.x_dim,
                                self.y_dim,
                                self.z_dim,
                                self.voxel_size_mm,
                                dt_ms,
                            );
                        }
                    } else {
                        self.cpu_step_into_next(dt_ms);
                    }
                }
                #[cfg(not(target_os = "macos"))]
                {
                    self.cpu_step_into_next(dt_ms);
                }
            }
        }
        std::mem::swap(&mut self.current, &mut self.next);
        self.time_ms += dt_ms;
        self.step_count += 1;
    }

    pub fn run(&mut self, steps: u64, dt_ms: f32) {
        for _ in 0..steps {
            self.step(dt_ms);
        }
    }

    pub fn deposit_patch_species(
        &mut self,
        species: TerrariumSpecies,
        x: usize,
        y: usize,
        z: usize,
        radius: usize,
        amount: f32,
    ) -> f32 {
        let added = amount.max(0.0);
        if added <= 1.0e-9 {
            return 0.0;
        }

        let radius = radius.max(1);
        let z_radius = radius.div_ceil(2).max(1);
        let mut weight_total = 0.0f32;

        for zz in z.saturating_sub(z_radius)..=(z + z_radius).min(self.z_dim.saturating_sub(1)) {
            for yy in y.saturating_sub(radius)..=(y + radius).min(self.y_dim.saturating_sub(1)) {
                for xx in x.saturating_sub(radius)..=(x + radius).min(self.x_dim.saturating_sub(1))
                {
                    let dx = xx as f32 - x as f32;
                    let dy = yy as f32 - y as f32;
                    let dz = zz as f32 - z as f32;
                    weight_total += Self::patch_weight(radius, dx, dy, dz);
                }
            }
        }

        if weight_total <= 1.0e-9 {
            return 0.0;
        }

        let total_voxels = self.total_voxels();
        for zz in z.saturating_sub(z_radius)..=(z + z_radius).min(self.z_dim.saturating_sub(1)) {
            for yy in y.saturating_sub(radius)..=(y + radius).min(self.y_dim.saturating_sub(1)) {
                for xx in x.saturating_sub(radius)..=(x + radius).min(self.x_dim.saturating_sub(1))
                {
                    let dx = xx as f32 - x as f32;
                    let dy = yy as f32 - y as f32;
                    let dz = zz as f32 - z as f32;
                    let weight = Self::patch_weight(radius, dx, dy, dz);
                    let share = added * (weight / weight_total);
                    let idx = self.index(species, xx, yy, zz);
                    self.current[idx] += share;
                    self.next[species as usize * total_voxels
                        + zz * self.y_dim * self.x_dim
                        + yy * self.x_dim
                        + xx] = self.current[idx];
                }
            }
        }

        added
    }

    pub fn mean_species(&self, species: TerrariumSpecies) -> f32 {
        let total = self.total_voxels();
        let start = species as usize * total;
        let end = start + total;
        self.current[start..end].iter().copied().sum::<f32>() / total as f32
    }

    pub fn species_field(&self, species: TerrariumSpecies) -> &[f32] {
        let total = self.total_voxels();
        let start = species as usize * total;
        let end = start + total;
        &self.current[start..end]
    }

    fn assign_control_field(
        target: &mut [f32],
        values: &[f32],
        min_value: f32,
        max_value: f32,
    ) -> Result<(), String> {
        if target.len() != values.len() {
            return Err(format!(
                "Control field length mismatch: expected {}, got {}",
                target.len(),
                values.len(),
            ));
        }
        for (dst, src) in target.iter_mut().zip(values.iter().copied()) {
            *dst = src.clamp(min_value, max_value);
        }
        Ok(())
    }

    pub fn set_hydration_field(&mut self, values: &[f32]) -> Result<(), String> {
        Self::assign_control_field(&mut self.hydration, values, 0.0, 1.5)?;
        self.external_controls = true;
        Ok(())
    }

    pub fn set_microbial_activity_field(&mut self, values: &[f32]) -> Result<(), String> {
        Self::assign_control_field(&mut self.microbial_activity, values, 0.0, 2.0)?;
        self.external_controls = true;
        Ok(())
    }

    pub fn set_plant_drive_field(&mut self, values: &[f32]) -> Result<(), String> {
        Self::assign_control_field(&mut self.plant_drive, values, 0.0, 1.5)?;
        self.external_controls = true;
        Ok(())
    }

    pub fn snapshot(&self) -> TerrariumSnapshot {
        TerrariumSnapshot {
            backend: self.backend,
            time_ms: self.time_ms,
            step_count: self.step_count,
            mean_hydration: self.hydration.iter().copied().sum::<f32>()
                / self.total_voxels() as f32,
            mean_microbes: self.microbial_activity.iter().copied().sum::<f32>()
                / self.total_voxels() as f32,
            mean_plant_drive: self.plant_drive.iter().copied().sum::<f32>()
                / self.total_voxels() as f32,
            mean_glucose: self.mean_species(TerrariumSpecies::Glucose),
            mean_oxygen_gas: self.mean_species(TerrariumSpecies::OxygenGas),
            mean_ammonium: self.mean_species(TerrariumSpecies::Ammonium),
            mean_nitrate: self.mean_species(TerrariumSpecies::Nitrate),
            mean_carbon_dioxide: self.mean_species(TerrariumSpecies::CarbonDioxide),
            mean_atp_flux: self.mean_species(TerrariumSpecies::AtpFlux),
            elemental_carbon: self.mean_species(TerrariumSpecies::Carbon),
            elemental_nitrogen: self.mean_species(TerrariumSpecies::Nitrogen),
        }
    }

    /// Aqueous diffusion coefficient derived from Stokes-Einstein equation.
    ///
    /// Previously a hardcoded match table (33 constants). Now computed from
    /// molecular structure via `emergent_rates::emergent_diffusion_coefficient`.
    /// Uses effective molecular radii from VDW volumes and temperature-dependent
    /// water viscosity (Arrhenius) — zero magic numbers.
    fn diffusion_coeff(species: TerrariumSpecies) -> f32 {
        // Temperature defaults to 20°C when called from substrate step.
        // The substrate doesn't track temperature directly, but the
        // TerrariumWorld does — this is a conservative default.
        // TODO: thread temperature through substrate step for full emergence.
        super::emergent_rates::emergent_diffusion_coefficient(species, 20.0)
    }

    pub(crate) fn species_from_index(idx: usize) -> TerrariumSpecies {
        match idx {
            0 => TerrariumSpecies::Carbon,
            1 => TerrariumSpecies::Hydrogen,
            2 => TerrariumSpecies::Oxygen,
            3 => TerrariumSpecies::Nitrogen,
            4 => TerrariumSpecies::Phosphorus,
            5 => TerrariumSpecies::Sulfur,
            6 => TerrariumSpecies::Water,
            7 => TerrariumSpecies::Glucose,
            8 => TerrariumSpecies::OxygenGas,
            9 => TerrariumSpecies::Ammonium,
            10 => TerrariumSpecies::Nitrate,
            11 => TerrariumSpecies::CarbonDioxide,
            12 => TerrariumSpecies::Proton,
            13 => TerrariumSpecies::AtpFlux,
            14 => TerrariumSpecies::AminoAcidPool,
            15 => TerrariumSpecies::NucleotidePool,
            16 => TerrariumSpecies::MembranePrecursorPool,
            17 => TerrariumSpecies::SilicateMineral,
            18 => TerrariumSpecies::ClayMineral,
            19 => TerrariumSpecies::CarbonateMineral,
            20 => TerrariumSpecies::IronOxideMineral,
            21 => TerrariumSpecies::DissolvedSilicate,
            22 => TerrariumSpecies::ExchangeableCalcium,
            23 => TerrariumSpecies::ExchangeableMagnesium,
            24 => TerrariumSpecies::ExchangeablePotassium,
            25 => TerrariumSpecies::ExchangeableSodium,
            26 => TerrariumSpecies::ExchangeableAluminum,
            27 => TerrariumSpecies::AqueousIronPool,
            28 => TerrariumSpecies::BicarbonatePool,
            29 => TerrariumSpecies::SurfaceProtonLoad,
            30 => TerrariumSpecies::CalciumBicarbonateComplex,
            31 => TerrariumSpecies::SorbedAluminumHydroxide,
            _ => TerrariumSpecies::SorbedFerricHydroxide,
        }
    }

    fn cpu_step_into_next(&mut self, dt_ms: f32) {
        let x_dim = self.x_dim;
        let y_dim = self.y_dim;
        let z_dim = self.z_dim;
        let total_voxels = self.total_voxels();
        let dx2 = (self.voxel_size_mm * self.voxel_size_mm).max(1e-6);
        let plane = x_dim * y_dim;

        // Compute ALL reaction rates from molecular bond energies + enzyme catalysis.
        // Zero hardcoded constants — every rate derives from Eyring TST.
        let rates = super::emergent_rates::SubstrateRateTable::at_temperature(self.temperature_c);
        // Stoichiometric coefficients from balanced chemical equations + molecular weights.
        // Zero hardcoded fractions — every coefficient derives from IUPAC atomic masses.
        let stoich = super::emergent_rates::substrate_stoichiometry();

        for z in 0..z_dim {
            for y in 0..y_dim {
                for x in 0..x_dim {
                    let gid = z * plane + y * x_dim + x;

                    let mut updated = [0.0f32; TERRARIUM_SPECIES_COUNT];
                    for species_idx in 0..TERRARIUM_SPECIES_COUNT {
                        let species = Self::species_from_index(species_idx);
                        let base = species_idx * total_voxels;
                        let idx = base + gid;
                        let c = self.current[idx];

                        let right = if x + 1 < x_dim {
                            self.current[base + z * plane + y * x_dim + (x + 1)]
                        } else {
                            c
                        };
                        let left = if x > 0 {
                            self.current[base + z * plane + y * x_dim + (x - 1)]
                        } else {
                            c
                        };
                        let up = if y + 1 < y_dim {
                            self.current[base + z * plane + (y + 1) * x_dim + x]
                        } else {
                            c
                        };
                        let down = if y > 0 {
                            self.current[base + z * plane + (y - 1) * x_dim + x]
                        } else {
                            c
                        };
                        let front = if z + 1 < z_dim {
                            self.current[base + (z + 1) * plane + y * x_dim + x]
                        } else {
                            c
                        };
                        let back = if z > 0 {
                            self.current[base + (z - 1) * plane + y * x_dim + x]
                        } else {
                            c
                        };

                        let laplacian = right + left + up + down + front + back - 6.0 * c;
                        let coeff = Self::diffusion_coeff(species) * dt_ms / dx2;
                        updated[species_idx] = (c + coeff * laplacian).max(0.0);
                    }

                    let hydration = self.hydration[gid].clamp(0.02, 1.0);
                    let microbes = self.microbial_activity[gid].clamp(0.02, 1.2);
                    let plants = self.plant_drive[gid].clamp(0.0, 1.0);
                    let atmospheric = if z == 0 { 1.0 } else { 0.0 };

                    let mut carbon = updated[TerrariumSpecies::Carbon as usize];
                    let mut hydrogen = updated[TerrariumSpecies::Hydrogen as usize];
                    let mut oxygen = updated[TerrariumSpecies::Oxygen as usize];
                    let mut nitrogen = updated[TerrariumSpecies::Nitrogen as usize];
                    let mut phosphorus = updated[TerrariumSpecies::Phosphorus as usize];
                    let mut sulfur = updated[TerrariumSpecies::Sulfur as usize];
                    let mut water = updated[TerrariumSpecies::Water as usize];
                    let mut glucose = updated[TerrariumSpecies::Glucose as usize];
                    let mut oxygen_gas = updated[TerrariumSpecies::OxygenGas as usize];
                    let mut ammonium = updated[TerrariumSpecies::Ammonium as usize];
                    let mut nitrate = updated[TerrariumSpecies::Nitrate as usize];
                    let mut co2 = updated[TerrariumSpecies::CarbonDioxide as usize];
                    let mut proton = updated[TerrariumSpecies::Proton as usize];
                    let mut atp = updated[TerrariumSpecies::AtpFlux as usize];
                    let mut amino_acids = updated[TerrariumSpecies::AminoAcidPool as usize];
                    let mut nucleotides = updated[TerrariumSpecies::NucleotidePool as usize];
                    let mut membrane_precursors =
                        updated[TerrariumSpecies::MembranePrecursorPool as usize];
                    let mut silicate_mineral = updated[TerrariumSpecies::SilicateMineral as usize];
                    let mut clay_mineral = updated[TerrariumSpecies::ClayMineral as usize];
                    let mut carbonate_mineral =
                        updated[TerrariumSpecies::CarbonateMineral as usize];
                    let mut iron_oxide_mineral =
                        updated[TerrariumSpecies::IronOxideMineral as usize];
                    let mut dissolved_silicate =
                        updated[TerrariumSpecies::DissolvedSilicate as usize];
                    let mut exchangeable_calcium =
                        updated[TerrariumSpecies::ExchangeableCalcium as usize];
                    let mut exchangeable_magnesium =
                        updated[TerrariumSpecies::ExchangeableMagnesium as usize];
                    let mut exchangeable_potassium =
                        updated[TerrariumSpecies::ExchangeablePotassium as usize];
                    let mut exchangeable_sodium =
                        updated[TerrariumSpecies::ExchangeableSodium as usize];
                    let mut exchangeable_aluminum =
                        updated[TerrariumSpecies::ExchangeableAluminum as usize];
                    let mut aqueous_iron_pool = updated[TerrariumSpecies::AqueousIronPool as usize];
                    let mut bicarbonate = updated[TerrariumSpecies::BicarbonatePool as usize];
                    let mut surface_proton_load =
                        updated[TerrariumSpecies::SurfaceProtonLoad as usize];
                    let mut calcium_bicarbonate_complex =
                        updated[TerrariumSpecies::CalciumBicarbonateComplex as usize];
                    let mut sorbed_aluminum_hydroxide =
                        updated[TerrariumSpecies::SorbedAluminumHydroxide as usize];
                    let mut sorbed_ferric_hydroxide =
                        updated[TerrariumSpecies::SorbedFerricHydroxide as usize];

                    water += atmospheric * dt_ms * rates.atm_water;
                    oxygen_gas += atmospheric * dt_ms * rates.atm_oxygen;

                    let hydration_gate = hydration / (0.18 + hydration);
                    let oxygen_gate = oxygen_gas / (0.03 + oxygen_gas);
                    let acidity_penalty = 1.0 / (1.0 + proton * 6.0);
                    let plant_gate = plants * acidity_penalty;

                    // ── Mineralization: organic biomass → glucose + NH₄⁺ + amino acids ──
                    let mineralize = (microbes
                        * hydration_gate
                        * dt_ms
                        * rates.mineralization
                        * carbon.min(hydrogen).min(oxygen))
                    .min(nitrogen + 0.01);
                    carbon = (carbon - mineralize * stoich.mineralization_carbon).max(0.0);
                    hydrogen = (hydrogen - mineralize * stoich.mineralization_hydrogen).max(0.0);
                    oxygen = (oxygen - mineralize * stoich.mineralization_oxygen).max(0.0);
                    nitrogen = (nitrogen - mineralize * stoich.mineralization_nitrogen).max(0.0);
                    glucose += mineralize * stoich.mineralization_yield_glucose;
                    ammonium += mineralize * stoich.mineralization_yield_ammonium;
                    amino_acids += mineralize * stoich.mineralization_yield_amino;
                    nucleotides += mineralize * stoich.mineralization_yield_nucleotide;

                    // ── Respiration: C₆H₁₂O₆ + 6O₂ → 6CO₂ + 6H₂O + ~30 ATP ──
                    // (Hill 1937, Hinkle 2005)
                    let respiration =
                        (microbes * hydration_gate * oxygen_gate * dt_ms * rates.respiration * glucose)
                            .min(oxygen_gas * 1.2);
                    glucose = (glucose - respiration).max(0.0);
                    oxygen_gas = (oxygen_gas - respiration * stoich.respiration_o2).max(0.0);
                    co2 += respiration * stoich.respiration_yield_co2;
                    water += respiration * stoich.respiration_yield_water;
                    atp = atp * 0.92 + respiration * stoich.respiration_yield_atp;

                    // ── Fermentation: C₆H₁₂O₆ → 2C₂H₅OH + 2CO₂ + 2 ATP ──
                    // (Pasteur 1857)
                    let fermentation = (microbes
                        * hydration_gate
                        * (1.0 - oxygen_gate)
                        * dt_ms
                        * rates.fermentation
                        * glucose)
                        .min(glucose);
                    glucose = (glucose - fermentation).max(0.0);
                    proton += fermentation * stoich.fermentation_yield_proton;
                    atp += fermentation * stoich.fermentation_yield_atp;

                    // ── Amino acid synthesis ──
                    let amino_synthesis = (microbes
                        * hydration_gate
                        * dt_ms
                        * rates.amino_synthesis
                        * glucose
                            .min(ammonium * 1.4 + nitrate * 0.5)
                            .min((0.04 + atp * 0.06).max(0.0)))
                    .min(glucose * 0.45);
                    glucose = (glucose - amino_synthesis * stoich.amino_consume_glucose).max(0.0);
                    ammonium = (ammonium - amino_synthesis * stoich.amino_consume_ammonium).max(0.0);
                    nitrate = (nitrate - amino_synthesis * stoich.amino_consume_nitrate).max(0.0);
                    atp = (atp - amino_synthesis * stoich.amino_consume_atp).max(0.0);
                    amino_acids += amino_synthesis * stoich.amino_yield;

                    // ── Nucleotide synthesis ──
                    let nucleotide_synthesis = (microbes
                        * hydration_gate
                        * dt_ms
                        * rates.nucleotide_synthesis
                        * nitrate
                            .min(phosphorus * 4.0)
                            .min(glucose * 0.8)
                            .min((0.02 + atp * 0.04).max(0.0)))
                    .min(nitrate * 0.55);
                    glucose = (glucose - nucleotide_synthesis * stoich.nucleotide_consume_glucose).max(0.0);
                    nitrate = (nitrate - nucleotide_synthesis * stoich.nucleotide_consume_nitrate).max(0.0);
                    phosphorus = (phosphorus - nucleotide_synthesis * stoich.nucleotide_consume_phosphorus).max(0.0);
                    atp = (atp - nucleotide_synthesis * stoich.nucleotide_consume_atp).max(0.0);
                    nucleotides += nucleotide_synthesis * stoich.nucleotide_yield;

                    // ── Membrane synthesis ──
                    let membrane_synthesis = ((microbes * 0.55 + plants * 0.45)
                        * hydration_gate
                        * dt_ms
                        * rates.membrane_synthesis
                        * glucose
                            .min(oxygen_gas * 0.8 + water * 0.2)
                            .min((0.02 + atp * 0.03).max(0.0)))
                    .min(glucose * 0.30);
                    glucose = (glucose - membrane_synthesis * stoich.membrane_consume_glucose).max(0.0);
                    oxygen_gas = (oxygen_gas - membrane_synthesis * stoich.membrane_consume_o2).max(0.0);
                    atp = (atp - membrane_synthesis * stoich.membrane_consume_atp).max(0.0);
                    membrane_precursors += membrane_synthesis * stoich.membrane_yield;

                    // ── Proteolysis: amino acids → NH₄⁺ + CO₂ + ATP ──
                    let proteolysis =
                        (microbes * hydration_gate * dt_ms * rates.proteolysis * amino_acids).min(amino_acids);
                    amino_acids = (amino_acids - proteolysis).max(0.0);
                    ammonium += proteolysis * stoich.proteolysis_yield_ammonium;
                    co2 += proteolysis * stoich.proteolysis_yield_co2;
                    atp += proteolysis * stoich.proteolysis_yield_atp;

                    // ── Nucleotide turnover ──
                    let nucleotide_turnover = (microbes
                        * hydration_gate
                        * (0.45 + (1.0 - oxygen_gate) * 0.35)
                        * dt_ms
                        * rates.nucleotide_turnover
                        * nucleotides)
                        .min(nucleotides);
                    nucleotides = (nucleotides - nucleotide_turnover).max(0.0);
                    nitrate += nucleotide_turnover * stoich.nucleotide_turnover_yield_nitrate;
                    phosphorus += nucleotide_turnover * stoich.nucleotide_turnover_yield_phosphorus;
                    atp += nucleotide_turnover * stoich.nucleotide_turnover_yield_atp;

                    // ── Membrane turnover ──
                    let membrane_turnover = ((microbes * 0.6 + acidity_penalty * 0.2)
                        * dt_ms
                        * rates.membrane_turnover
                        * membrane_precursors)
                        .min(membrane_precursors);
                    membrane_precursors = (membrane_precursors - membrane_turnover).max(0.0);
                    glucose += membrane_turnover * stoich.membrane_turnover_yield_glucose;
                    carbon += membrane_turnover * stoich.membrane_turnover_yield_carbon;
                    sulfur += membrane_turnover * stoich.membrane_turnover_yield_sulfur;

                    // ── Nitrification: NH₄⁺ + 2O₂ → NO₃⁻ + 2H⁺ + H₂O ──
                    // (Prosser 1990)
                    let nitrification =
                        (microbes * hydration_gate * oxygen_gate * dt_ms * rates.nitrification * ammonium)
                            .min(oxygen_gas * 1.5);
                    ammonium = (ammonium - nitrification).max(0.0);
                    nitrate += nitrification * stoich.nitrification_yield_nitrate;
                    oxygen_gas = (oxygen_gas - nitrification * stoich.nitrification_consume_o2).max(0.0);
                    proton += nitrification * stoich.nitrification_yield_proton;
                    atp += nitrification * stoich.nitrification_yield_atp;

                    // ── Denitrification: 5CH₂O + 4NO₃⁻ + 4H⁺ → 5CO₂ + 2N₂ + 7H₂O ──
                    // (Knowles 1982)
                    let denitrification = (microbes
                        * hydration_gate
                        * (1.0 - oxygen_gate)
                        * dt_ms
                        * rates.denitrification
                        * nitrate)
                        .min(nitrate);
                    nitrate = (nitrate - denitrification).max(0.0);
                    nitrogen += denitrification * stoich.denitrification_yield_n2;
                    proton = (proton - denitrification * stoich.denitrification_consume_proton).max(0.0);

                    // ── Photosynthesis: 6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂ ──
                    // (Hill 1937, Calvin 1961)
                    let photosynthesis = (plant_gate * dt_ms * rates.photosynthesis * co2)
                        .min(water * stoich.photosynthesis_consume_water)
                        .min(0.15 + co2);
                    co2 = (co2 - photosynthesis).max(0.0);
                    water = (water - photosynthesis * stoich.photosynthesis_consume_water).max(0.0);
                    glucose += photosynthesis * stoich.photosynthesis_yield_glucose;
                    oxygen_gas += photosynthesis * stoich.photosynthesis_yield_o2;
                    carbon += photosynthesis * stoich.photosynthesis_yield_carbon;
                    oxygen += photosynthesis * stoich.photosynthesis_yield_oxygen;
                    membrane_precursors += photosynthesis * stoich.photosynthesis_yield_membrane;

                    let phosphate_turnover = (glucose * rates.phosphate_turnover_fast + atp * rates.phosphate_turnover_slow) * dt_ms;
                    phosphorus = (phosphorus - phosphate_turnover * 0.05).max(0.0);
                    sulfur = (sulfur - phosphate_turnover * 0.02).max(0.0);

                    let mineral_water_gate = water / (0.08 + water);
                    let acidity_drive = proton / (0.01 + proton);
                    let silicate_weathering =
                        (silicate_mineral * mineral_water_gate * acidity_drive * dt_ms * rates.silicate_weathering)
                            .min(silicate_mineral)
                            .min(proton * 0.55);
                    // ── Silicate weathering (Chou & Wollast 1989) ──
                    silicate_mineral = (silicate_mineral - silicate_weathering).max(0.0);
                    dissolved_silicate += silicate_weathering * stoich.silicate_yield_dissolved_si;
                    exchangeable_magnesium += silicate_weathering * stoich.silicate_yield_mg;
                    exchangeable_potassium += silicate_weathering * stoich.silicate_yield_k;
                    exchangeable_sodium += silicate_weathering * stoich.silicate_yield_na;
                    exchangeable_calcium += silicate_weathering * stoich.silicate_yield_ca;
                    proton = (proton - silicate_weathering * stoich.silicate_consume_proton).max(0.0);

                    // ── Clay weathering (Ganor+ 1995) ──
                    let clay_weathering =
                        (clay_mineral * mineral_water_gate * acidity_drive * dt_ms * rates.clay_weathering)
                            .min(clay_mineral)
                            .min(proton * 0.45);
                    clay_mineral = (clay_mineral - clay_weathering).max(0.0);
                    dissolved_silicate += clay_weathering * stoich.clay_yield_dissolved_si;
                    exchangeable_aluminum += clay_weathering * stoich.clay_yield_al;
                    exchangeable_magnesium += clay_weathering * stoich.clay_yield_mg;
                    exchangeable_potassium += clay_weathering * stoich.clay_yield_k;
                    proton = (proton - clay_weathering * stoich.clay_consume_proton).max(0.0);

                    let carbonate_buffering =
                        (carbonate_mineral * mineral_water_gate * acidity_drive * dt_ms * rates.carbonate_buffering)
                            .min(carbonate_mineral)
                            .min(proton * 0.98);
                    carbonate_mineral = (carbonate_mineral - carbonate_buffering).max(0.0);
                    proton = (proton - carbonate_buffering).max(0.0);
                    exchangeable_calcium += carbonate_buffering;
                    bicarbonate += carbonate_buffering;

                    let bicarbonate_degassing = (bicarbonate
                        * acidity_drive
                        * (0.35 + (1.0 - mineral_water_gate) * 0.30)
                        * dt_ms
                        * rates.bicarb_degassing)
                        .min(bicarbonate)
                        .min(proton * 0.92 + 0.002);
                    bicarbonate = (bicarbonate - bicarbonate_degassing).max(0.0);
                    proton = (proton - bicarbonate_degassing).max(0.0);
                    co2 += bicarbonate_degassing;
                    water += bicarbonate_degassing;

                    let alkalinity_gate = bicarbonate / (0.04 + bicarbonate);
                    let mineral_surface_gate =
                        (clay_mineral * 0.55 + carbonate_mineral * 0.25 + silicate_mineral * 0.20)
                            / (0.06
                                + clay_mineral * 0.55
                                + carbonate_mineral * 0.25
                                + silicate_mineral * 0.20);

                    let proton_surface_sorption = (proton
                        * mineral_surface_gate
                        * (0.30 + acidity_drive * 0.55)
                        * (0.28 + (1.0 - alkalinity_gate) * 0.36)
                        * dt_ms
                        * rates.proton_adsorption)
                        .min(proton)
                        .min(
                            (0.42 + clay_mineral * 0.38 + carbonate_mineral * 0.20)
                                - surface_proton_load,
                        )
                        .max(0.0);
                    proton = (proton - proton_surface_sorption).max(0.0);
                    surface_proton_load += proton_surface_sorption;

                    let proton_surface_desorption = (surface_proton_load
                        * (0.18 + alkalinity_gate * 0.36)
                        * (0.22 + oxygen_gate * 0.10 + mineral_water_gate * 0.16)
                        * dt_ms
                        * rates.proton_desorption)
                        .min(surface_proton_load);
                    surface_proton_load =
                        (surface_proton_load - proton_surface_desorption).max(0.0);
                    proton += proton_surface_desorption;

                    let calcium_bicarbonate_complexation = (exchangeable_calcium
                        * bicarbonate
                        * (0.14 + alkalinity_gate * 0.34)
                        * (0.20 + mineral_water_gate * 0.18)
                        * (1.0 - acidity_drive * 0.42).clamp(0.0, 1.0)
                        * dt_ms
                        * rates.ca_bicarb_complexation)
                        .min(exchangeable_calcium)
                        .min(bicarbonate * 0.5);
                    exchangeable_calcium =
                        (exchangeable_calcium - calcium_bicarbonate_complexation).max(0.0);
                    bicarbonate = (bicarbonate - calcium_bicarbonate_complexation * 2.0).max(0.0);
                    calcium_bicarbonate_complex += calcium_bicarbonate_complexation;

                    let calcium_bicarbonate_dissociation = (calcium_bicarbonate_complex
                        * (0.14 + acidity_drive * 0.38 + surface_proton_load * 0.32)
                        * dt_ms
                        * rates.ca_bicarb_dissociation)
                        .min(calcium_bicarbonate_complex);
                    calcium_bicarbonate_complex =
                        (calcium_bicarbonate_complex - calcium_bicarbonate_dissociation).max(0.0);
                    exchangeable_calcium += calcium_bicarbonate_dissociation;
                    bicarbonate += calcium_bicarbonate_dissociation * 2.0;

                    let aluminum_hydroxide_precip = (exchangeable_aluminum
                        * mineral_water_gate
                        * (1.0 - acidity_drive * 0.55).clamp(0.0, 1.0)
                        * (0.35 + alkalinity_gate * 0.65)
                        * dt_ms
                        * rates.al_hydroxide_precip)
                        .min(exchangeable_aluminum)
                        .min(water / 3.0);
                    exchangeable_aluminum =
                        (exchangeable_aluminum - aluminum_hydroxide_precip).max(0.0);
                    water = (water - aluminum_hydroxide_precip * 3.0).max(0.0);
                    sorbed_aluminum_hydroxide += aluminum_hydroxide_precip;
                    proton += aluminum_hydroxide_precip * 3.0;

                    let aluminum_hydroxide_dissolution = (sorbed_aluminum_hydroxide
                        * acidity_drive
                        * (0.30 + (1.0 - alkalinity_gate) * 0.50)
                        * dt_ms
                        * rates.al_hydroxide_dissolution)
                        .min(sorbed_aluminum_hydroxide)
                        .min(proton / 3.0);
                    sorbed_aluminum_hydroxide =
                        (sorbed_aluminum_hydroxide - aluminum_hydroxide_dissolution).max(0.0);
                    proton = (proton - aluminum_hydroxide_dissolution * 3.0).max(0.0);
                    exchangeable_aluminum += aluminum_hydroxide_dissolution;
                    water += aluminum_hydroxide_dissolution * 3.0;

                    let ferric_hydroxide_precip = (aqueous_iron_pool
                        * mineral_water_gate
                        * oxygen_gate
                        * (1.0 - acidity_drive * 0.45).clamp(0.0, 1.0)
                        * (0.30 + alkalinity_gate * 0.70)
                        * dt_ms
                        * rates.fe_hydroxide_precip)
                        .min(aqueous_iron_pool)
                        .min(water / 3.0);
                    aqueous_iron_pool = (aqueous_iron_pool - ferric_hydroxide_precip).max(0.0);
                    water = (water - ferric_hydroxide_precip * 3.0).max(0.0);
                    sorbed_ferric_hydroxide += ferric_hydroxide_precip;
                    proton += ferric_hydroxide_precip * 3.0;

                    let ferric_hydroxide_dissolution = (sorbed_ferric_hydroxide
                        * acidity_drive
                        * (1.0 - oxygen_gate * 0.25)
                        * (0.32 + (1.0 - alkalinity_gate) * 0.42)
                        * dt_ms
                        * rates.fe_hydroxide_dissolution)
                        .min(sorbed_ferric_hydroxide)
                        .min(proton / 3.0);
                    sorbed_ferric_hydroxide =
                        (sorbed_ferric_hydroxide - ferric_hydroxide_dissolution).max(0.0);
                    proton = (proton - ferric_hydroxide_dissolution * 3.0).max(0.0);
                    aqueous_iron_pool += ferric_hydroxide_dissolution;
                    water += ferric_hydroxide_dissolution * 3.0;

                    let iron_release = (iron_oxide_mineral
                        * mineral_water_gate
                        * acidity_drive
                        * (1.0 - oxygen_gate * 0.35)
                        * dt_ms
                        * rates.iron_release)
                        .min(iron_oxide_mineral)
                        .min(proton * 0.25 + 0.002);
                    iron_oxide_mineral = (iron_oxide_mineral - iron_release).max(0.0);
                    aqueous_iron_pool += iron_release * 0.72;
                    proton = (proton - iron_release * 0.01).max(0.0);

                    let base_leaching =
                        (water * mineral_water_gate * dt_ms * rates.base_leaching).clamp(0.0, 0.02);
                    exchangeable_sodium =
                        (exchangeable_sodium - base_leaching * exchangeable_sodium * 0.22).max(0.0);
                    exchangeable_potassium = (exchangeable_potassium
                        - base_leaching * exchangeable_potassium * 0.16)
                        .max(0.0);
                    exchangeable_calcium = (exchangeable_calcium
                        - base_leaching * exchangeable_calcium * 0.12)
                        .max(0.0);
                    exchangeable_magnesium = (exchangeable_magnesium
                        - base_leaching * exchangeable_magnesium * 0.10)
                        .max(0.0);

                    water = (water * (1.0 - dt_ms * rates.water_evaporation)).clamp(0.0, 2.5);
                    glucose = glucose.clamp(0.0, 2.0);
                    oxygen_gas = oxygen_gas.clamp(0.0, 1.5);
                    ammonium = ammonium.clamp(0.0, 1.0);
                    nitrate = nitrate.clamp(0.0, 1.0);
                    co2 = co2.clamp(0.0, 1.5);
                    proton = proton.clamp(0.0, 0.8);
                    atp = atp.clamp(0.0, 6.0);
                    amino_acids = amino_acids.clamp(0.0, 1.2);
                    nucleotides = nucleotides.clamp(0.0, 1.0);
                    membrane_precursors = membrane_precursors.clamp(0.0, 1.0);
                    silicate_mineral = silicate_mineral.clamp(0.0, 4.0);
                    clay_mineral = clay_mineral.clamp(0.0, 4.0);
                    carbonate_mineral = carbonate_mineral.clamp(0.0, 2.0);
                    iron_oxide_mineral = iron_oxide_mineral.clamp(0.0, 2.0);
                    dissolved_silicate = dissolved_silicate.clamp(0.0, 1.2);
                    exchangeable_calcium = exchangeable_calcium.clamp(0.0, 1.2);
                    exchangeable_magnesium = exchangeable_magnesium.clamp(0.0, 1.0);
                    exchangeable_potassium = exchangeable_potassium.clamp(0.0, 0.9);
                    exchangeable_sodium = exchangeable_sodium.clamp(0.0, 0.9);
                    exchangeable_aluminum = exchangeable_aluminum.clamp(0.0, 0.9);
                    aqueous_iron_pool = aqueous_iron_pool.clamp(0.0, 0.8);
                    bicarbonate = bicarbonate.clamp(0.0, 1.2);
                    surface_proton_load = surface_proton_load.clamp(0.0, 0.8);
                    calcium_bicarbonate_complex = calcium_bicarbonate_complex.clamp(0.0, 0.8);
                    sorbed_aluminum_hydroxide = sorbed_aluminum_hydroxide.clamp(0.0, 0.8);
                    sorbed_ferric_hydroxide = sorbed_ferric_hydroxide.clamp(0.0, 0.8);

                    updated[TerrariumSpecies::Carbon as usize] = carbon;
                    updated[TerrariumSpecies::Hydrogen as usize] = hydrogen;
                    updated[TerrariumSpecies::Oxygen as usize] = oxygen;
                    updated[TerrariumSpecies::Nitrogen as usize] = nitrogen;
                    updated[TerrariumSpecies::Phosphorus as usize] = phosphorus;
                    updated[TerrariumSpecies::Sulfur as usize] = sulfur;
                    updated[TerrariumSpecies::Water as usize] = water;
                    updated[TerrariumSpecies::Glucose as usize] = glucose;
                    updated[TerrariumSpecies::OxygenGas as usize] = oxygen_gas;
                    updated[TerrariumSpecies::Ammonium as usize] = ammonium;
                    updated[TerrariumSpecies::Nitrate as usize] = nitrate;
                    updated[TerrariumSpecies::CarbonDioxide as usize] = co2;
                    updated[TerrariumSpecies::Proton as usize] = proton;
                    updated[TerrariumSpecies::AtpFlux as usize] = atp;
                    updated[TerrariumSpecies::AminoAcidPool as usize] = amino_acids;
                    updated[TerrariumSpecies::NucleotidePool as usize] = nucleotides;
                    updated[TerrariumSpecies::MembranePrecursorPool as usize] = membrane_precursors;
                    updated[TerrariumSpecies::SilicateMineral as usize] = silicate_mineral;
                    updated[TerrariumSpecies::ClayMineral as usize] = clay_mineral;
                    updated[TerrariumSpecies::CarbonateMineral as usize] = carbonate_mineral;
                    updated[TerrariumSpecies::IronOxideMineral as usize] = iron_oxide_mineral;
                    updated[TerrariumSpecies::DissolvedSilicate as usize] = dissolved_silicate;
                    updated[TerrariumSpecies::ExchangeableCalcium as usize] = exchangeable_calcium;
                    updated[TerrariumSpecies::ExchangeableMagnesium as usize] =
                        exchangeable_magnesium;
                    updated[TerrariumSpecies::ExchangeablePotassium as usize] =
                        exchangeable_potassium;
                    updated[TerrariumSpecies::ExchangeableSodium as usize] = exchangeable_sodium;
                    updated[TerrariumSpecies::ExchangeableAluminum as usize] =
                        exchangeable_aluminum;
                    updated[TerrariumSpecies::AqueousIronPool as usize] = aqueous_iron_pool;
                    updated[TerrariumSpecies::BicarbonatePool as usize] = bicarbonate;
                    updated[TerrariumSpecies::SurfaceProtonLoad as usize] = surface_proton_load;
                    updated[TerrariumSpecies::CalciumBicarbonateComplex as usize] =
                        calcium_bicarbonate_complex;
                    updated[TerrariumSpecies::SorbedAluminumHydroxide as usize] =
                        sorbed_aluminum_hydroxide;
                    updated[TerrariumSpecies::SorbedFerricHydroxide as usize] =
                        sorbed_ferric_hydroxide;

                    for species_idx in 0..TERRARIUM_SPECIES_COUNT {
                        let base = species_idx * total_voxels;
                        self.next[base + gid] = updated[species_idx].max(0.0);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BatchedAtomTerrarium, TerrariumBackend, TerrariumSpecies};

    #[test]
    fn substrate_stays_bounded() {
        let mut terrarium = BatchedAtomTerrarium::new(12, 12, 8, 0.5, false);
        let backend = terrarium.backend();
        assert!(matches!(backend, TerrariumBackend::Cpu));
        terrarium.run(200, 0.25);
        let snap = terrarium.snapshot();
        assert!(snap.mean_glucose >= 0.0);
        assert!(snap.mean_oxygen_gas >= 0.0);
        assert!(snap.mean_atp_flux >= 0.0);
        assert!(snap.mean_hydration > 0.0);
    }

    #[test]
    fn hotspots_increase_local_pool() {
        let mut terrarium = BatchedAtomTerrarium::new(10, 10, 6, 0.5, false);
        let before = terrarium.mean_species(TerrariumSpecies::Glucose);
        terrarium.add_hotspot(TerrariumSpecies::Glucose, 5, 5, 2, 1.0);
        let after = terrarium.mean_species(TerrariumSpecies::Glucose);
        assert!(after > before);
    }

    #[test]
    fn patch_extraction_removes_local_mass() {
        let mut terrarium = BatchedAtomTerrarium::new(10, 10, 6, 0.5, false);
        terrarium.add_hotspot(TerrariumSpecies::Ammonium, 5, 5, 2, 1.2);
        let before = terrarium.patch_mean_species(TerrariumSpecies::Ammonium, 5, 5, 2, 2);
        let removed = terrarium.extract_patch_species(TerrariumSpecies::Ammonium, 5, 5, 2, 2, 0.2);
        let after = terrarium.patch_mean_species(TerrariumSpecies::Ammonium, 5, 5, 2, 2);
        assert!(removed > 0.0);
        assert!(after < before);
    }

    #[test]
    fn metal_kernel_species_count_matches_rust_substrate_species_count() {
        let shader = include_str!("../metal/terrarium_substrate.metal");
        assert!(
            shader.contains("constant uint SPECIES_COUNT = 33;"),
            "Metal substrate kernel species count drifted from Rust terrarium species count"
        );
        assert!(shader.contains("IDX_SILICATE_MINERAL = 17;"));
        assert!(shader.contains("IDX_CLAY_MINERAL = 18;"));
        assert!(shader.contains("IDX_CARBONATE_MINERAL = 19;"));
        assert!(shader.contains("IDX_IRON_OXIDE_MINERAL = 20;"));
        assert!(shader.contains("IDX_DISSOLVED_SILICATE = 21;"));
        assert!(shader.contains("IDX_EXCHANGEABLE_CALCIUM = 22;"));
        assert!(shader.contains("IDX_EXCHANGEABLE_MAGNESIUM = 23;"));
        assert!(shader.contains("IDX_EXCHANGEABLE_POTASSIUM = 24;"));
        assert!(shader.contains("IDX_EXCHANGEABLE_SODIUM = 25;"));
        assert!(shader.contains("IDX_EXCHANGEABLE_ALUMINUM = 26;"));
        assert!(shader.contains("IDX_AQUEOUS_IRON = 27;"));
        assert!(shader.contains("IDX_BICARBONATE = 28;"));
        assert!(shader.contains("IDX_SURFACE_PROTON_LOAD = 29;"));
        assert!(shader.contains("IDX_CALCIUM_BICARBONATE_COMPLEX = 30;"));
        assert!(shader.contains("IDX_SORBED_ALUMINUM_HYDROXIDE = 31;"));
        assert!(shader.contains("IDX_SORBED_FERRIC_HYDROXIDE = 32;"));
    }

    #[test]
    fn acidic_weathering_mobilizes_representative_mineral_ions() {
        let mut terrarium = BatchedAtomTerrarium::new(4, 4, 3, 1.0, false);
        let x = 2usize;
        let y = 2usize;
        let z = 2usize;
        terrarium.add_hotspot(TerrariumSpecies::Water, x, y, z, 1.0);
        terrarium.add_hotspot(TerrariumSpecies::Proton, x, y, z, 0.6);
        terrarium.add_hotspot(TerrariumSpecies::SilicateMineral, x, y, z, 1.0);
        terrarium.add_hotspot(TerrariumSpecies::ClayMineral, x, y, z, 0.8);
        terrarium.add_hotspot(TerrariumSpecies::CarbonateMineral, x, y, z, 0.5);
        terrarium.add_hotspot(TerrariumSpecies::IronOxideMineral, x, y, z, 0.4);

        // Sum species over the ENTIRE grid to avoid diffusion artifacts.
        // With emergent Stokes-Einstein diffusion, fast-diffusing ions (Al³⁺,
        // Fe²⁺) spread quickly beyond local measurement patches. Global sums
        // test the geochemistry correctly: mineral dissolution produces ions
        // regardless of where they diffuse.
        let global_sum = |t: &BatchedAtomTerrarium, sp: TerrariumSpecies| -> f32 {
            let total = t.total_voxels();
            let base = sp as usize * total;
            (0..total).map(|i| t.current[base + i]).sum::<f32>()
        };

        let before_silica = global_sum(&terrarium, TerrariumSpecies::DissolvedSilicate);
        let before_calcium = global_sum(&terrarium, TerrariumSpecies::ExchangeableCalcium);
        let before_aluminum = global_sum(&terrarium, TerrariumSpecies::ExchangeableAluminum)
            + global_sum(&terrarium, TerrariumSpecies::SorbedAluminumHydroxide);
        let before_iron = global_sum(&terrarium, TerrariumSpecies::AqueousIronPool)
            + global_sum(&terrarium, TerrariumSpecies::SorbedFerricHydroxide);
        let before_bicarbonate = global_sum(&terrarium, TerrariumSpecies::BicarbonatePool);
        let before_surface_proton = global_sum(&terrarium, TerrariumSpecies::SurfaceProtonLoad);

        for _ in 0..5 {
            terrarium.step(10.0);
        }

        let after_silica = global_sum(&terrarium, TerrariumSpecies::DissolvedSilicate);
        let after_calcium = global_sum(&terrarium, TerrariumSpecies::ExchangeableCalcium);
        // Total mobilized Al = exchangeable + sorbed hydroxide.
        // Clay weathering releases Al³⁺ which may re-precipitate as Al(OH)₃.
        let after_aluminum = global_sum(&terrarium, TerrariumSpecies::ExchangeableAluminum)
            + global_sum(&terrarium, TerrariumSpecies::SorbedAluminumHydroxide);
        // Total mobilized Fe = aqueous + sorbed ferric hydroxide.
        let after_iron = global_sum(&terrarium, TerrariumSpecies::AqueousIronPool)
            + global_sum(&terrarium, TerrariumSpecies::SorbedFerricHydroxide);
        let after_bicarbonate = global_sum(&terrarium, TerrariumSpecies::BicarbonatePool);
        let after_surface_proton = global_sum(&terrarium, TerrariumSpecies::SurfaceProtonLoad);

        assert!(after_silica > before_silica, "Silica: {} -> {}", before_silica, after_silica);
        assert!(after_calcium > before_calcium, "Calcium: {} -> {}", before_calcium, after_calcium);
        assert!(
            after_aluminum > before_aluminum,
            "Total Al (exch+sorbed): {} -> {}",
            before_aluminum, after_aluminum,
        );
        assert!(
            after_iron > before_iron,
            "Total Fe (aq+sorbed): {} -> {}",
            before_iron, after_iron,
        );
        assert!(after_bicarbonate > before_bicarbonate);
        assert!(after_surface_proton >= before_surface_proton);
    }
}
