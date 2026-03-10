//! GPU Molecular Dynamics — Metal-accelerated MD simulation.
//!
//! This module provides GPU-accelerated molecular dynamics using Apple Metal
//! compute shaders. It implements:
//! - Bonded forces (bonds, angles, dihedrals)
//! - Non-bonded forces (Lennard-Jones, electrostatics)
//! - Velocity Verlet integration with Langevin thermostat
//! - Neighbor list for efficient O(N) non-bonded calculations
//!
//! Uses unified memory (StorageModeShared) for zero-copy CPU↔GPU access.

#[cfg(target_os = "macos")]
use metal::*;

/// MD GPU parameters - matches Metal shader struct.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct MDParams {
    pub n_atoms: u32,
    pub dt: f32,
    pub temperature: f32,
    pub cutoff: f32,
    pub box_x: f32,
    pub box_y: f32,
    pub box_z: f32,
    pub epsilon_lj: f32,
    pub epsilon_coulomb: f32,
}

impl Default for MDParams {
    fn default() -> Self {
        Self {
            n_atoms: 0,
            dt: 0.001,          // 1 fs timestep
            temperature: 300.0, // 300 K
            cutoff: 12.0,       // Ångströms
            box_x: 100.0,
            box_y: 100.0,
            box_z: 100.0,
            epsilon_lj: 1.0,
            epsilon_coulomb: 332.0, // kcal·Å/e²
        }
    }
}

/// Atom data for GPU - matches Metal shader struct.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct MDAtom {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub force: [f32; 3],
    pub mass: f32,
    pub charge: f32,
    pub sigma: f32,
    pub epsilon: f32,
}

impl Default for MDAtom {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
            force: [0.0, 0.0, 0.0],
            mass: 1.0,
            charge: 0.0,
            sigma: 3.4,
            epsilon: 0.1,
        }
    }
}

/// Bond parameters for GPU.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct MDBond {
    pub i: u32,
    pub j: u32,
    pub r0: f32,
    pub k: f32,
}

/// MD GPU Context - manages Metal device and pipelines.
#[cfg(target_os = "macos")]
pub struct MDGpuContext {
    device: Device,
    queue: CommandQueue,
    pipelines: MDGpuPipelines,
}

#[cfg(target_os = "macos")]
pub struct MDGpuPipelines {
    pub clear_forces: ComputePipelineState,
    pub bond_forces: ComputePipelineState,
    pub lj_forces: ComputePipelineState,
    pub coulomb_forces: ComputePipelineState,
    pub kinetic_energy: ComputePipelineState,
    pub integrate_langevin: ComputePipelineState,
    pub integrate_nve: ComputePipelineState,
    pub thermostat: ComputePipelineState,
}

#[cfg(target_os = "macos")]
impl MDGpuContext {
    /// Create new MD GPU context.
    pub fn new() -> Result<Self, String> {
        let device =
            Device::system_default().ok_or_else(|| "No Metal GPU device found".to_string())?;

        let queue = device.new_command_queue();

        // Compile shaders
        let shader_source = include_str!("../metal/md_forces.metal").to_string()
            + include_str!("../metal/md_integrate.metal")
            + include_str!("../metal/neighbor_list.metal");

        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(&shader_source, &options)
            .map_err(|e| format!("MD Metal shader compilation failed: {}", e))?;

        let pipelines = Self::create_pipelines(&device, &library)?;

        Ok(Self {
            device,
            queue,
            pipelines,
        })
    }

    fn create_pipelines(device: &Device, library: &Library) -> Result<MDGpuPipelines, String> {
        let make = |name: &str| -> Result<ComputePipelineState, String> {
            let func = library
                .get_function(name, None)
                .map_err(|e| format!("MD shader '{}': {}", name, e))?;
            device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| format!("MD pipeline '{}': {}", name, e))
        };

        Ok(MDGpuPipelines {
            clear_forces: make("clear_forces")?,
            bond_forces: make("compute_bond_forces")?,
            lj_forces: make("compute_lj_forces")?,
            coulomb_forces: make("compute_coulomb_forces")?,
            kinetic_energy: make("compute_kinetic_energy")?,
            integrate_langevin: make("integrate_langevin")?,
            integrate_nve: make("integrate_nve")?,
            thermostat: make("apply_berendsen_thermostat")?,
        })
    }

    /// Create GPU buffer from atom data.
    pub fn create_atom_buffer(&self, atoms: &[MDAtom]) -> Buffer {
        let size = (atoms.len() * std::mem::size_of::<MDAtom>()) as u64;
        let buffer = self
            .device
            .new_buffer(size, MTLResourceOptions::StorageModeShared);
        unsafe {
            let ptr = buffer.contents() as *mut MDAtom;
            std::ptr::copy_nonoverlapping(atoms.as_ptr(), ptr, atoms.len());
        }
        buffer
    }

    /// Create GPU buffer from bonds.
    pub fn create_bond_buffer(&self, bonds: &[MDBond]) -> Buffer {
        let size = (bonds.len() * std::mem::size_of::<MDBond>()) as u64;
        let buffer = self
            .device
            .new_buffer(size, MTLResourceOptions::StorageModeShared);
        unsafe {
            let ptr = buffer.contents() as *mut MDBond;
            std::ptr::copy_nonoverlapping(bonds.as_ptr(), ptr, bonds.len());
        }
        buffer
    }

    /// Clear forces on GPU.
    pub fn clear_forces(&self, atom_buffer: &Buffer, n_atoms: u32) {
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipelines.clear_forces);
        enc.set_buffer(0, Some(atom_buffer), 0);

        let threads = (n_atoms + 255) / 256;
        enc.dispatch_thread_groups(MTLSize::new(threads as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Compute bond forces on GPU.
    pub fn compute_bond_forces(&self, atom_buffer: &Buffer, bond_buffer: &Buffer, n_bonds: u32) {
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipelines.bond_forces);
        enc.set_buffer(0, Some(atom_buffer), 0);
        enc.set_buffer(1, Some(bond_buffer), 0);

        let threads = (n_bonds + 255) / 256;
        enc.dispatch_thread_groups(MTLSize::new(threads as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Integrate with Langevin thermostat.
    pub fn integrate_langevin(&self, atom_buffer: &Buffer, params: &MDParams, n_atoms: u32) {
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipelines.integrate_langevin);
        enc.set_buffer(0, Some(atom_buffer), 0);
        enc.set_bytes(
            1,
            std::mem::size_of::<MDParams>() as u64,
            params as *const _ as *const _,
        );

        let threads = (n_atoms + 255) / 256;
        enc.dispatch_thread_groups(MTLSize::new(threads as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Read atoms back from GPU.
    pub fn read_atoms(&self, buffer: &Buffer, n_atoms: usize) -> Vec<MDAtom> {
        let mut atoms = vec![MDAtom::default(); n_atoms];
        unsafe {
            let ptr = buffer.contents() as *const MDAtom;
            std::ptr::copy_nonoverlapping(ptr, atoms.as_mut_ptr(), n_atoms);
        }
        atoms
    }
}

/// CPU fallback for non-Mac platforms.
#[cfg(not(target_os = "macos"))]
pub struct MDGpuContext;

#[cfg(not(target_os = "macos"))]
impl MDGpuContext {
    pub fn new() -> Result<Self, String> {
        Ok(Self)
    }
}

/// Check if GPU acceleration is available.
pub fn has_gpu_md() -> bool {
    #[cfg(target_os = "macos")]
    {
        Device::system_default().is_some()
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}
