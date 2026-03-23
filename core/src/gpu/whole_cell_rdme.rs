//! GPU and CPU execution for whole-cell intracellular reaction-diffusion.
//!
//! The lattice stores a small set of core intracellular species in
//! structure-of-arrays form so each species can be diffused independently
//! while keeping the memory layout GPU-friendly.

use rayon::prelude::*;
use std::ops::Range;

#[cfg(target_os = "macos")]
use super::GpuContext;

/// Core intracellular species tracked on the lattice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum IntracellularSpecies {
    ATP = 0,
    AminoAcids = 1,
    Nucleotides = 2,
    MembranePrecursors = 3,
}

impl IntracellularSpecies {
    pub const COUNT: usize = 4;

    pub fn index(self) -> usize {
        self as usize
    }

    fn diffusion_coeff_nm2_per_ms(self) -> f32 {
        match self {
            IntracellularSpecies::ATP => 60_000.0,
            IntracellularSpecies::AminoAcids => 40_000.0,
            IntracellularSpecies::Nucleotides => 28_000.0,
            IntracellularSpecies::MembranePrecursors => 10_000.0,
        }
    }

    fn basal_source_per_ms(self) -> f32 {
        match self {
            IntracellularSpecies::ATP => 0.012,
            IntracellularSpecies::AminoAcids => 0.006,
            IntracellularSpecies::Nucleotides => 0.004,
            IntracellularSpecies::MembranePrecursors => 0.002,
        }
    }

    fn basal_sink_per_ms(self) -> f32 {
        match self {
            IntracellularSpecies::ATP => 0.010,
            IntracellularSpecies::AminoAcids => 0.005,
            IntracellularSpecies::Nucleotides => 0.003,
            IntracellularSpecies::MembranePrecursors => 0.0015,
        }
    }
}

const SPECIES_ORDER: [IntracellularSpecies; IntracellularSpecies::COUNT] = [
    IntracellularSpecies::ATP,
    IntracellularSpecies::AminoAcids,
    IntracellularSpecies::Nucleotides,
    IntracellularSpecies::MembranePrecursors,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum IntracellularSpatialField {
    MembraneAdjacency = 0,
    SeptumZone = 1,
    NucleoidOccupancy = 2,
    MembraneBandZone = 3,
    PoleZone = 4,
}

impl IntracellularSpatialField {
    pub const COUNT: usize = 5;

    pub fn index(self) -> usize {
        self as usize
    }
}

#[cfg(test)]
const FIELD_ORDER: [IntracellularSpatialField; IntracellularSpatialField::COUNT] = [
    IntracellularSpatialField::MembraneAdjacency,
    IntracellularSpatialField::SeptumZone,
    IntracellularSpatialField::NucleoidOccupancy,
    IntracellularSpatialField::MembraneBandZone,
    IntracellularSpatialField::PoleZone,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum WholeCellRdmeDriveField {
    EnergySource = 0,
    AtpDemand = 1,
    AminoDemand = 2,
    NucleotideDemand = 3,
    MembraneSource = 4,
    MembraneDemand = 5,
    Crowding = 6,
}

impl WholeCellRdmeDriveField {
    pub const COUNT: usize = 7;

    pub fn index(self) -> usize {
        self as usize
    }
}

#[cfg(test)]
const DRIVE_FIELD_ORDER: [WholeCellRdmeDriveField; WholeCellRdmeDriveField::COUNT] = [
    WholeCellRdmeDriveField::EnergySource,
    WholeCellRdmeDriveField::AtpDemand,
    WholeCellRdmeDriveField::AminoDemand,
    WholeCellRdmeDriveField::NucleotideDemand,
    WholeCellRdmeDriveField::MembraneSource,
    WholeCellRdmeDriveField::MembraneDemand,
    WholeCellRdmeDriveField::Crowding,
];

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct WholeCellRdmeContext {
    pub metabolic_load: f32,
}

impl Default for WholeCellRdmeContext {
    fn default() -> Self {
        Self {
            metabolic_load: 1.0,
        }
    }
}

fn stable_rdme_substeps(voxel_size_nm: f32, dt: f32) -> (u32, f32) {
    let dx2 = voxel_size_nm * voxel_size_nm;
    let max_diffusion = SPECIES_ORDER
        .iter()
        .map(|species| species.diffusion_coeff_nm2_per_ms())
        .fold(0.0, f32::max)
        .max(1.0);
    let max_stable_dt = (0.16 * dx2 / max_diffusion).max(1.0e-4);
    let substeps = (dt / max_stable_dt).ceil().max(1.0) as u32;
    let sub_dt = dt / substeps as f32;
    (substeps.max(1), sub_dt.max(1.0e-4))
}

/// Voxelized intracellular state used by the native whole-cell runtime.
pub struct IntracellularLattice {
    pub x_dim: usize,
    pub y_dim: usize,
    pub z_dim: usize,
    pub voxel_size_nm: f32,
    pub current: Vec<f32>,
    pub next: Vec<f32>,
}

impl IntracellularLattice {
    /// Create a new zero-filled lattice.
    pub fn new(x_dim: usize, y_dim: usize, z_dim: usize, voxel_size_nm: f32) -> Self {
        let total = IntracellularSpecies::COUNT * x_dim * y_dim * z_dim;
        Self {
            x_dim,
            y_dim,
            z_dim,
            voxel_size_nm,
            current: vec![0.0; total],
            next: vec![0.0; total],
        }
    }

    /// Total voxels in a single species channel.
    pub fn total_voxels(&self) -> usize {
        self.x_dim * self.y_dim * self.z_dim
    }

    fn channel_range(&self, species: IntracellularSpecies) -> Range<usize> {
        let total_voxels = self.total_voxels();
        let start = species.index() * total_voxels;
        start..start + total_voxels
    }

    /// Set a channel to a uniform value.
    pub fn fill_species(&mut self, species: IntracellularSpecies, value: f32) {
        let range = self.channel_range(species);
        self.current[range.clone()].fill(value);
        self.next[range].fill(value);
    }

    /// Add a localized hotspot used for testing or seeding gradients.
    pub fn add_hotspot(
        &mut self,
        species: IntracellularSpecies,
        x: usize,
        y: usize,
        z: usize,
        delta: f32,
    ) {
        if x >= self.x_dim || y >= self.y_dim || z >= self.z_dim {
            return;
        }
        let idx = z * self.y_dim * self.x_dim + y * self.x_dim + x;
        let range = self.channel_range(species);
        self.current[range.start + idx] = (self.current[range.start + idx] + delta).max(0.0);
        self.next[range.start + idx] = self.current[range.start + idx];
    }

    /// Mean concentration for a species.
    pub fn mean_species(&self, species: IntracellularSpecies) -> f32 {
        let range = self.channel_range(species);
        let total = self.total_voxels() as f32;
        if total <= 0.0 {
            return 0.0;
        }
        self.current[range].iter().sum::<f32>() / total
    }

    /// Uniformly perturb a species channel.
    pub fn apply_uniform_delta(&mut self, species: IntracellularSpecies, delta: f32) {
        let range = self.channel_range(species);
        self.current[range.clone()]
            .par_iter_mut()
            .for_each(|value| *value = (*value + delta).max(0.0));
        let source = self.current[range.clone()].to_vec();
        self.next[range].copy_from_slice(&source);
    }

    pub fn apply_weighted_delta(
        &mut self,
        species: IntracellularSpecies,
        delta: f32,
        weights: &[f32],
    ) {
        let expected = self.total_voxels();
        if weights.len() != expected {
            self.apply_uniform_delta(species, delta);
            return;
        }
        let weight_sum = weights
            .iter()
            .map(|weight| weight.max(0.0))
            .sum::<f32>()
            .max(1.0e-6);
        let mean_scale = expected as f32 / weight_sum;
        let range = self.channel_range(species);
        self.current[range.clone()]
            .par_iter_mut()
            .zip(weights.par_iter())
            .for_each(|(value, weight)| {
                let local_delta = delta * weight.max(0.0) * mean_scale;
                *value = (*value + local_delta).max(0.0);
            });
        let source = self.current[range.clone()].to_vec();
        self.next[range].copy_from_slice(&source);
    }

    /// Return a copy of a channel for callers that need direct inspection.
    pub fn clone_species(&self, species: IntracellularSpecies) -> Vec<f32> {
        self.current[self.channel_range(species)].to_vec()
    }

    /// Replace a species channel with explicit values.
    pub fn set_species(
        &mut self,
        species: IntracellularSpecies,
        values: &[f32],
    ) -> Result<(), String> {
        let expected = self.total_voxels();
        if values.len() != expected {
            return Err(format!(
                "species channel length mismatch: expected {}, got {}",
                expected,
                values.len()
            ));
        }
        let range = self.channel_range(species);
        self.current[range.clone()].copy_from_slice(values);
        self.next[range].copy_from_slice(values);
        Ok(())
    }

    /// Swap read/write buffers after an RDME update.
    pub fn swap_buffers(&mut self) {
        std::mem::swap(&mut self.current, &mut self.next);
    }
}

#[derive(Debug, Clone)]
pub struct IntracellularSpatialState {
    pub x_dim: usize,
    pub y_dim: usize,
    pub z_dim: usize,
    pub fields: Vec<f32>,
}

impl IntracellularSpatialState {
    pub fn new(x_dim: usize, y_dim: usize, z_dim: usize) -> Self {
        let total = IntracellularSpatialField::COUNT * x_dim * y_dim * z_dim;
        Self {
            x_dim,
            y_dim,
            z_dim,
            fields: vec![0.0; total],
        }
    }

    pub fn total_voxels(&self) -> usize {
        self.x_dim * self.y_dim * self.z_dim
    }

    fn field_range(&self, field: IntracellularSpatialField) -> Range<usize> {
        let total_voxels = self.total_voxels();
        let start = field.index() * total_voxels;
        start..start + total_voxels
    }

    pub fn clone_field(&self, field: IntracellularSpatialField) -> Vec<f32> {
        self.fields[self.field_range(field)].to_vec()
    }

    pub fn field_slice(&self, field: IntracellularSpatialField) -> &[f32] {
        &self.fields[self.field_range(field)]
    }

    pub fn set_field(
        &mut self,
        field: IntracellularSpatialField,
        values: &[f32],
    ) -> Result<(), String> {
        let expected = self.total_voxels();
        if values.len() != expected {
            return Err(format!(
                "spatial field length mismatch: expected {}, got {}",
                expected,
                values.len()
            ));
        }
        let range = self.field_range(field);
        self.fields[range].copy_from_slice(values);
        Ok(())
    }

    pub fn weighted_mean_species(
        &self,
        lattice: &IntracellularLattice,
        species: IntracellularSpecies,
        field: IntracellularSpatialField,
    ) -> f32 {
        let species_values = &lattice.current[lattice.channel_range(species)];
        let weights = &self.fields[self.field_range(field)];
        let mut weighted_sum = 0.0;
        let mut weight_total = 0.0;
        for (&value, &weight) in species_values.iter().zip(weights.iter()) {
            let clamped = weight.max(0.0);
            weighted_sum += value * clamped;
            weight_total += clamped;
        }
        if weight_total <= 1.0e-6 {
            lattice.mean_species(species)
        } else {
            weighted_sum / weight_total
        }
    }
}

pub struct WholeCellRdmeDriveState {
    x_dim: usize,
    y_dim: usize,
    z_dim: usize,
    fields: Vec<f32>,
}

impl WholeCellRdmeDriveState {
    pub fn new(x_dim: usize, y_dim: usize, z_dim: usize) -> Self {
        let total = WholeCellRdmeDriveField::COUNT * x_dim * y_dim * z_dim;
        Self {
            x_dim,
            y_dim,
            z_dim,
            fields: vec![0.0; total],
        }
    }

    pub fn total_voxels(&self) -> usize {
        self.x_dim * self.y_dim * self.z_dim
    }

    fn field_range(&self, field: WholeCellRdmeDriveField) -> Range<usize> {
        let total_voxels = self.total_voxels();
        let start = field.index() * total_voxels;
        start..start + total_voxels
    }

    pub fn clone_field(&self, field: WholeCellRdmeDriveField) -> Vec<f32> {
        self.fields[self.field_range(field)].to_vec()
    }

    pub fn field_slice(&self, field: WholeCellRdmeDriveField) -> &[f32] {
        &self.fields[self.field_range(field)]
    }

    pub fn weighted_mean(&self, field: WholeCellRdmeDriveField, weights: &[f32]) -> f32 {
        let values = self.field_slice(field);
        if values.len() != weights.len() {
            return 0.0;
        }
        let mut weighted_sum = 0.0;
        let mut weight_total = 0.0;
        for (&value, &weight) in values.iter().zip(weights.iter()) {
            let clamped = weight.max(0.0);
            weighted_sum += value * clamped;
            weight_total += clamped;
        }
        if weight_total <= 1.0e-6 {
            0.0
        } else {
            weighted_sum / weight_total
        }
    }

    pub fn set_field(
        &mut self,
        field: WholeCellRdmeDriveField,
        values: &[f32],
    ) -> Result<(), String> {
        let expected = self.total_voxels();
        if values.len() != expected {
            return Err(format!(
                "drive field length mismatch: expected {}, got {}",
                expected,
                values.len()
            ));
        }
        let range = self.field_range(field);
        self.fields[range].copy_from_slice(values);
        Ok(())
    }
}

/// Params struct matching the Metal shader constant buffer.
#[repr(C)]
struct WholeCellRdmeParams {
    x_dim: u32,
    y_dim: u32,
    z_dim: u32,
    voxel_size_nm: f32,
    dt: f32,
    metabolic_load: f32,
}

fn local_diffusion_scale(
    species: IntracellularSpecies,
    membrane_adjacency: f32,
    septum_zone: f32,
    nucleoid_occupancy: f32,
    local_crowding: f32,
) -> f32 {
    let crowding_pressure = local_crowding.clamp(0.0, 1.6);
    let base = match species {
        IntracellularSpecies::ATP => 1.0 - 0.12 * septum_zone - 0.20 * nucleoid_occupancy,
        IntracellularSpecies::AminoAcids => 1.0 - 0.10 * septum_zone - 0.24 * nucleoid_occupancy,
        IntracellularSpecies::Nucleotides => 1.0 - 0.08 * septum_zone - 0.28 * nucleoid_occupancy,
        IntracellularSpecies::MembranePrecursors => {
            0.82 + 0.12 * membrane_adjacency - 0.14 * septum_zone
        }
    };
    (base - crowding_pressure * (0.12 + 0.22 * nucleoid_occupancy)).clamp(0.18, 1.35)
}

fn local_source_scale(
    species: IntracellularSpecies,
    membrane_adjacency: f32,
    septum_zone: f32,
    nucleoid_occupancy: f32,
    local_energy_source: f32,
    local_membrane_source: f32,
    local_crowding: f32,
) -> f32 {
    match species {
        IntracellularSpecies::ATP => {
            (0.80 + 0.14 * membrane_adjacency + 0.08 * septum_zone + 0.22 * local_energy_source
                - 0.06 * local_crowding)
                .clamp(0.25, 1.80)
        }
        IntracellularSpecies::AminoAcids => {
            (0.90 + 0.08 * (1.0 - nucleoid_occupancy) - 0.05 * local_crowding).clamp(0.30, 1.50)
        }
        IntracellularSpecies::Nucleotides => {
            (0.82 + 0.18 * nucleoid_occupancy - 0.04 * local_crowding).clamp(0.30, 1.55)
        }
        IntracellularSpecies::MembranePrecursors => (0.44
            + local_membrane_source * (0.26 * membrane_adjacency + 0.54 * septum_zone)
            - 0.04 * local_crowding)
            .clamp(0.18, 1.90),
    }
}

fn local_sink_scale(
    species: IntracellularSpecies,
    membrane_adjacency: f32,
    septum_zone: f32,
    nucleoid_occupancy: f32,
    local_atp_demand: f32,
    local_amino_demand: f32,
    local_nucleotide_demand: f32,
    local_membrane_demand: f32,
    local_crowding: f32,
    context: WholeCellRdmeContext,
) -> f32 {
    match species {
        IntracellularSpecies::ATP => (0.72
            + local_atp_demand
                * (0.24 * membrane_adjacency + 0.20 * septum_zone + 0.16 * nucleoid_occupancy)
            + 0.10 * local_crowding
            + 0.08 * context.metabolic_load.max(0.1))
        .clamp(0.20, 2.20),
        IntracellularSpecies::AminoAcids => (0.80
            + local_amino_demand * (0.28 + 0.18 * nucleoid_occupancy + 0.10 * septum_zone)
            + 0.08 * local_crowding
            + 0.05 * context.metabolic_load.max(0.1))
        .clamp(0.20, 1.90),
        IntracellularSpecies::Nucleotides => (0.70
            + local_nucleotide_demand * (0.44 * nucleoid_occupancy + 0.12 * septum_zone)
            + 0.06 * local_crowding
            + 0.05 * context.metabolic_load.max(0.1))
        .clamp(0.20, 2.20),
        IntracellularSpecies::MembranePrecursors => (0.26
            + local_membrane_demand * (0.30 * membrane_adjacency + 0.58 * septum_zone)
            + 0.08 * local_crowding
            + 0.04 * context.metabolic_load.max(0.1))
        .clamp(0.10, 2.40),
    }
}

/// Dispatch whole-cell RDME on Metal.
#[cfg(target_os = "macos")]
pub fn dispatch_whole_cell_rdme(
    gpu: &GpuContext,
    lattice: &mut IntracellularLattice,
    spatial: &IntracellularSpatialState,
    drive: &WholeCellRdmeDriveState,
    dt: f32,
    context: WholeCellRdmeContext,
) {
    let total_voxels = lattice.total_voxels() as u64;
    if total_voxels == 0 {
        return;
    }
    let (substeps, sub_dt) = stable_rdme_substeps(lattice.voxel_size_nm, dt.max(1.0e-4));
    for _ in 0..substeps {
        let buf_grid_in = gpu.buffer_from_slice(&lattice.current);
        let buf_grid_out = gpu.buffer_from_slice(&lattice.next);
        let buf_fields = gpu.buffer_from_slice(&spatial.fields);
        let buf_drive = gpu.buffer_from_slice(&drive.fields);

        let params = WholeCellRdmeParams {
            x_dim: lattice.x_dim as u32,
            y_dim: lattice.y_dim as u32,
            z_dim: lattice.z_dim as u32,
            voxel_size_nm: lattice.voxel_size_nm,
            dt: sub_dt,
            metabolic_load: context.metabolic_load,
        };
        let param_bytes = unsafe {
            std::slice::from_raw_parts(
                &params as *const WholeCellRdmeParams as *const u8,
                std::mem::size_of::<WholeCellRdmeParams>(),
            )
        };

        gpu.dispatch_1d(
            &gpu.pipelines.whole_cell_rdme,
            &[
                (&buf_grid_in, 0),
                (&buf_grid_out, 0),
                (&buf_fields, 0),
                (&buf_drive, 0),
            ],
            Some((param_bytes, 4)),
            total_voxels,
        );

        unsafe {
            let ptr = buf_grid_out.contents() as *const f32;
            std::ptr::copy_nonoverlapping(ptr, lattice.next.as_mut_ptr(), lattice.next.len());
        }

        lattice.swap_buffers();
    }
}

/// CPU fallback for non-macOS platforms.
#[cfg(not(target_os = "macos"))]
pub fn dispatch_whole_cell_rdme(
    _gpu: &super::GpuContext,
    lattice: &mut IntracellularLattice,
    spatial: &IntracellularSpatialState,
    drive: &WholeCellRdmeDriveState,
    dt: f32,
    context: WholeCellRdmeContext,
) {
    cpu_whole_cell_rdme(lattice, spatial, drive, dt, context);
}

/// CPU reference implementation of the intracellular RDME step.
pub fn cpu_whole_cell_rdme(
    lattice: &mut IntracellularLattice,
    spatial: &IntracellularSpatialState,
    drive: &WholeCellRdmeDriveState,
    dt: f32,
    context: WholeCellRdmeContext,
) {
    let x_dim = lattice.x_dim;
    let y_dim = lattice.y_dim;
    let z_dim = lattice.z_dim;
    let total_voxels = lattice.total_voxels();
    if total_voxels == 0 {
        return;
    }
    let dx2 = lattice.voxel_size_nm * lattice.voxel_size_nm;
    let (substeps, sub_dt) = stable_rdme_substeps(lattice.voxel_size_nm, dt.max(1.0e-4));

    for _ in 0..substeps {
        for species in SPECIES_ORDER {
            let range = lattice.channel_range(species);
            let current = &lattice.current[range.clone()];
            let next = &mut lattice.next[range];
            let coeff = species.diffusion_coeff_nm2_per_ms() / dx2 * sub_dt;
            let source = species.basal_source_per_ms() * sub_dt;
            let sink = species.basal_sink_per_ms() * sub_dt * context.metabolic_load.max(0.1);
            let membrane_field =
                &spatial.fields[spatial.field_range(IntracellularSpatialField::MembraneAdjacency)];
            let septum_field =
                &spatial.fields[spatial.field_range(IntracellularSpatialField::SeptumZone)];
            let nucleoid_field =
                &spatial.fields[spatial.field_range(IntracellularSpatialField::NucleoidOccupancy)];
            let energy_source_field =
                &drive.fields[drive.field_range(WholeCellRdmeDriveField::EnergySource)];
            let atp_demand_field =
                &drive.fields[drive.field_range(WholeCellRdmeDriveField::AtpDemand)];
            let amino_demand_field =
                &drive.fields[drive.field_range(WholeCellRdmeDriveField::AminoDemand)];
            let nucleotide_demand_field =
                &drive.fields[drive.field_range(WholeCellRdmeDriveField::NucleotideDemand)];
            let membrane_source_field =
                &drive.fields[drive.field_range(WholeCellRdmeDriveField::MembraneSource)];
            let membrane_demand_field =
                &drive.fields[drive.field_range(WholeCellRdmeDriveField::MembraneDemand)];
            let crowding_field =
                &drive.fields[drive.field_range(WholeCellRdmeDriveField::Crowding)];

            next.par_iter_mut().enumerate().for_each(|(gid, out)| {
                let z = gid / (y_dim * x_dim);
                let rem = gid - z * y_dim * x_dim;
                let y = rem / x_dim;
                let x = rem - y * x_dim;

                let c = current[gid];

                let right = if x + 1 < x_dim {
                    current[z * y_dim * x_dim + y * x_dim + (x + 1)]
                } else {
                    c
                };
                let left = if x > 0 {
                    current[z * y_dim * x_dim + y * x_dim + (x - 1)]
                } else {
                    c
                };
                let up = if y + 1 < y_dim {
                    current[z * y_dim * x_dim + (y + 1) * x_dim + x]
                } else {
                    c
                };
                let down = if y > 0 {
                    current[z * y_dim * x_dim + (y - 1) * x_dim + x]
                } else {
                    c
                };
                let front = if z + 1 < z_dim {
                    current[(z + 1) * y_dim * x_dim + y * x_dim + x]
                } else {
                    c
                };
                let back = if z > 0 {
                    current[(z - 1) * y_dim * x_dim + y * x_dim + x]
                } else {
                    c
                };

                let laplacian = right + left + up + down + front + back - 6.0 * c;
                let membrane_adjacency = membrane_field[gid].clamp(0.0, 1.0);
                let septum_zone = septum_field[gid].clamp(0.0, 1.0);
                let nucleoid_occupancy = nucleoid_field[gid].clamp(0.0, 1.0);
                let local_energy_source = energy_source_field[gid].clamp(0.0, 2.5);
                let local_atp_demand = atp_demand_field[gid].clamp(0.0, 2.5);
                let local_amino_demand = amino_demand_field[gid].clamp(0.0, 2.5);
                let local_nucleotide_demand = nucleotide_demand_field[gid].clamp(0.0, 2.5);
                let local_membrane_source = membrane_source_field[gid].clamp(0.0, 2.5);
                let local_membrane_demand = membrane_demand_field[gid].clamp(0.0, 2.5);
                let local_crowding = crowding_field[gid].clamp(0.0, 1.6);
                let diffusion_scale = local_diffusion_scale(
                    species,
                    membrane_adjacency,
                    septum_zone,
                    nucleoid_occupancy,
                    local_crowding,
                );
                let source_scale = local_source_scale(
                    species,
                    membrane_adjacency,
                    septum_zone,
                    nucleoid_occupancy,
                    local_energy_source,
                    local_membrane_source,
                    local_crowding,
                );
                let sink_scale = local_sink_scale(
                    species,
                    membrane_adjacency,
                    septum_zone,
                    nucleoid_occupancy,
                    local_atp_demand,
                    local_amino_demand,
                    local_nucleotide_demand,
                    local_membrane_demand,
                    local_crowding,
                    context,
                );
                let updated = c + coeff * diffusion_scale * laplacian + source * source_scale
                    - sink * sink_scale * c;
                *out = updated.max(0.0);
            });
        }
        lattice.swap_buffers();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_rdme_substeps_keep_default_dt_finite() {
        let mut lattice = IntracellularLattice::new(24, 24, 12, 20.0);
        let mut spatial = IntracellularSpatialState::new(24, 24, 12);
        let mut drive = WholeCellRdmeDriveState::new(24, 24, 12);
        lattice.fill_species(IntracellularSpecies::ATP, 1.2);
        lattice.fill_species(IntracellularSpecies::AminoAcids, 0.95);
        lattice.fill_species(IntracellularSpecies::Nucleotides, 0.80);
        lattice.fill_species(IntracellularSpecies::MembranePrecursors, 0.35);
        lattice.add_hotspot(IntracellularSpecies::ATP, 12, 12, 6, 2.5);
        for field in FIELD_ORDER {
            let values = vec![0.25; lattice.total_voxels()];
            spatial.set_field(field, &values).expect("spatial field");
        }
        for field in DRIVE_FIELD_ORDER {
            let values = vec![0.10; lattice.total_voxels()];
            drive.set_field(field, &values).expect("drive field");
        }

        for _ in 0..32 {
            cpu_whole_cell_rdme(
                &mut lattice,
                &spatial,
                &drive,
                0.25,
                WholeCellRdmeContext::default(),
            );
        }

        for species in SPECIES_ORDER {
            let values = lattice.clone_species(species);
            assert!(values.iter().all(|value| value.is_finite()));
            assert!(lattice.mean_species(species).is_finite());
        }
    }
}
