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
use crate::gpu::terrarium_substrate::{dispatch_terrarium_substrate, dispatch_terrarium_substrate_persistent};

#[cfg(target_os = "macos")]
use crate::gpu::GpuContext;

pub const TERRARIUM_SPECIES_COUNT: usize = 14;

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
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
}

impl BatchedAtomTerrarium {
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
        let (gpu, gpu_buf_current, gpu_buf_next, gpu_buf_hydration, gpu_buf_microbes, gpu_buf_plants) = if backend == TerrariumBackend::Metal {
            if let Some(gpu_ctx) = GpuContext::new().ok() {
                let size_species = (TERRARIUM_SPECIES_COUNT * total_voxels * std::mem::size_of::<f32>()) as u64;
                let size_scalar = (total_voxels * std::mem::size_of::<f32>()) as u64;
                
                let buf_current = gpu_ctx.buffer_of_size(size_species);
                let buf_next = gpu_ctx.buffer_of_size(size_species);
                let buf_hydration = gpu_ctx.buffer_of_size(size_scalar);
                let buf_microbes = gpu_ctx.buffer_of_size(size_scalar);
                let buf_plants = gpu_ctx.buffer_of_size(size_scalar);
                
                (Some(gpu_ctx), Some(buf_current), Some(buf_next), Some(buf_hydration), Some(buf_microbes), Some(buf_plants))
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

                    let water = self.hydration[ctrl] * 0.95;
                    let oxygen_gas = 0.10 + (1.0 - depth) * 0.22;
                    let carbon = 0.22 + depth * 0.55 + rhizosphere * 0.08;
                    let hydrogen = 0.32 + depth * 0.64;
                    let oxygen = 0.18 + (1.0 - depth) * 0.18 + water * 0.08;
                    let nitrogen = 0.05 + depth * 0.11 + rhizosphere * 0.02;
                    let phosphorus = 0.012 + depth * 0.020;
                    let sulfur = 0.006 + depth * 0.010;
                    let glucose = 0.010 + rhizosphere * 0.030;
                    let ammonium = 0.012 + depth * 0.032;
                    let nitrate = 0.018 + (1.0 - depth) * 0.034;
                    let co2 = 0.010 + depth * 0.020;
                    let proton = 0.003 + depth * 0.006;

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
                        if let (Some(ref buf_current), Some(ref buf_next), Some(ref buf_hydration), 
                                Some(ref buf_microbes), Some(ref buf_plants)) = 
                            (&self.gpu_buf_current, &self.gpu_buf_next, &self.gpu_buf_hydration, 
                             &self.gpu_buf_microbes, &self.gpu_buf_plants) {
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

    pub fn deposit_patch_species(&mut self, species: TerrariumSpecies, x: usize, y: usize, z: usize, radius: usize, amount: f32) -> f32 {
        let _ = (species, x, y, z, radius, amount); 0.0
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

    fn diffusion_coeff(species: TerrariumSpecies) -> f32 {
        match species {
            TerrariumSpecies::Carbon => 0.0010,
            TerrariumSpecies::Hydrogen => 0.0011,
            TerrariumSpecies::Oxygen => 0.0012,
            TerrariumSpecies::Nitrogen => 0.0010,
            TerrariumSpecies::Phosphorus => 0.0004,
            TerrariumSpecies::Sulfur => 0.0004,
            TerrariumSpecies::Water => 0.0032,
            TerrariumSpecies::Glucose => 0.0016,
            TerrariumSpecies::OxygenGas => 0.0048,
            TerrariumSpecies::Ammonium => 0.0020,
            TerrariumSpecies::Nitrate => 0.0018,
            TerrariumSpecies::CarbonDioxide => 0.0038,
            TerrariumSpecies::Proton => 0.0024,
            TerrariumSpecies::AtpFlux => 0.0009,
        }
    }

    fn species_from_index(idx: usize) -> TerrariumSpecies {
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
            _ => TerrariumSpecies::AtpFlux,
        }
    }

    fn cpu_step_into_next(&mut self, dt_ms: f32) {
        let x_dim = self.x_dim;
        let y_dim = self.y_dim;
        let z_dim = self.z_dim;
        let total_voxels = self.total_voxels();
        let dx2 = (self.voxel_size_mm * self.voxel_size_mm).max(1e-6);
        let plane = x_dim * y_dim;

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

                    water += atmospheric * dt_ms * 0.0008;
                    oxygen_gas += atmospheric * dt_ms * 0.0015;

                    let hydration_gate = hydration / (0.18 + hydration);
                    let oxygen_gate = oxygen_gas / (0.03 + oxygen_gas);
                    let acidity_penalty = 1.0 / (1.0 + proton * 6.0);
                    let plant_gate = plants * acidity_penalty;

                    let mineralize = (microbes
                        * hydration_gate
                        * dt_ms
                        * 0.0015
                        * carbon.min(hydrogen).min(oxygen))
                    .min(nitrogen + 0.01);
                    carbon = (carbon - mineralize * 0.60).max(0.0);
                    hydrogen = (hydrogen - mineralize * 0.84).max(0.0);
                    oxygen = (oxygen - mineralize * 0.52).max(0.0);
                    nitrogen = (nitrogen - mineralize * 0.12).max(0.0);
                    glucose += mineralize * 0.72;
                    ammonium += mineralize * 0.18;

                    let respiration =
                        (microbes * hydration_gate * oxygen_gate * dt_ms * 0.0032 * glucose)
                            .min(oxygen_gas * 1.2);
                    glucose = (glucose - respiration).max(0.0);
                    oxygen_gas = (oxygen_gas - respiration * 0.75).max(0.0);
                    co2 += respiration * 0.92;
                    water += respiration * 0.25;
                    atp = atp * 0.92 + respiration * 6.6;

                    let fermentation = (microbes
                        * hydration_gate
                        * (1.0 - oxygen_gate)
                        * dt_ms
                        * 0.0012
                        * glucose)
                        .min(glucose);
                    glucose = (glucose - fermentation).max(0.0);
                    proton += fermentation * 0.08;
                    atp += fermentation * 1.4;

                    let nitrification =
                        (microbes * hydration_gate * oxygen_gate * dt_ms * 0.0018 * ammonium)
                            .min(oxygen_gas * 1.5);
                    ammonium = (ammonium - nitrification).max(0.0);
                    nitrate += nitrification * 0.95;
                    oxygen_gas = (oxygen_gas - nitrification * 0.45).max(0.0);
                    proton += nitrification * 0.05;
                    atp += nitrification * 0.5;

                    let denitrification = (microbes
                        * hydration_gate
                        * (1.0 - oxygen_gate)
                        * dt_ms
                        * 0.0007
                        * nitrate)
                        .min(nitrate);
                    nitrate = (nitrate - denitrification).max(0.0);
                    nitrogen += denitrification * 0.65;
                    proton = (proton - denitrification * 0.03).max(0.0);

                    let photosynthesis = (plant_gate * dt_ms * 0.0022 * co2)
                        .min(water * 0.6)
                        .min(0.15 + co2);
                    co2 = (co2 - photosynthesis).max(0.0);
                    water = (water - photosynthesis * 0.45).max(0.0);
                    glucose += photosynthesis * 0.72;
                    oxygen_gas += photosynthesis * 0.84;
                    carbon += photosynthesis * 0.08;
                    oxygen += photosynthesis * 0.04;

                    let phosphate_turnover = (glucose * 0.004 + atp * 0.0006) * dt_ms;
                    phosphorus = (phosphorus - phosphate_turnover * 0.05).max(0.0);
                    sulfur = (sulfur - phosphate_turnover * 0.02).max(0.0);

                    water = (water * (1.0 - dt_ms * 0.00012)).clamp(0.0, 2.5);
                    glucose = glucose.clamp(0.0, 2.0);
                    oxygen_gas = oxygen_gas.clamp(0.0, 1.5);
                    ammonium = ammonium.clamp(0.0, 1.0);
                    nitrate = nitrate.clamp(0.0, 1.0);
                    co2 = co2.clamp(0.0, 1.5);
                    proton = proton.clamp(0.0, 0.8);
                    atp = atp.clamp(0.0, 6.0);

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
}
