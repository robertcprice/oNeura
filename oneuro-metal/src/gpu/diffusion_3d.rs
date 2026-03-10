//! GPU dispatch for 3D extracellular space neurotransmitter diffusion.
//!
//! Applies a 6-neighbor Laplacian stencil (3D isotropic diffusion) to NT
//! concentrations in a voxelized extracellular space. Uses double-buffering
//! (read from current grid, write to next grid) to avoid race conditions.
//!
//! Grid memory layout (SoA per NT channel):
//!   `grid[nt_channel * Z*Y*X + z * Y*X + y * X + x]`
//!
//! Boundary condition: Neumann (zero-flux) -- missing neighbors use center value.

#[cfg(target_os = "macos")]
use super::GpuContext;

/// Params struct matching the Metal shader's `Params` constant buffer.
#[repr(C)]
struct DiffusionParams {
    x_dim: u32,
    y_dim: u32,
    z_dim: u32,
    voxel_size: f32, // mm (typically 0.01)
    dt: f32,         // ms
}

/// 3D voxel grid for extracellular space simulation.
///
/// Stores 6 NT concentrations per voxel in SoA layout. Double-buffered:
/// `current` is read from, `next` is written to, then they are swapped.
pub struct ExtracellularGrid {
    /// Grid dimensions (voxels per axis).
    pub x_dim: usize,
    pub y_dim: usize,
    pub z_dim: usize,
    /// Voxel edge length in mm.
    pub voxel_size: f32,
    /// Current grid: flat array of size 6 * x_dim * y_dim * z_dim.
    /// Layout: `current[nt * total_voxels + z * y_dim * x_dim + y * x_dim + x]`
    pub current: Vec<f32>,
    /// Next grid (write target for double-buffering).
    pub next: Vec<f32>,
}

impl ExtracellularGrid {
    /// Create a new grid with all concentrations initialized to zero.
    pub fn new(x_dim: usize, y_dim: usize, z_dim: usize, voxel_size: f32) -> Self {
        let total = 6 * x_dim * y_dim * z_dim;
        Self {
            x_dim,
            y_dim,
            z_dim,
            voxel_size,
            current: vec![0.0; total],
            next: vec![0.0; total],
        }
    }

    /// Total number of voxels (excluding NT channel dimension).
    pub fn total_voxels(&self) -> usize {
        self.x_dim * self.y_dim * self.z_dim
    }

    /// Linear index for a given (nt_channel, x, y, z) coordinate.
    pub fn index(&self, nt: usize, x: usize, y: usize, z: usize) -> usize {
        nt * self.total_voxels() + z * self.y_dim * self.x_dim + y * self.x_dim + x
    }

    /// Swap current and next buffers after a diffusion step.
    pub fn swap_buffers(&mut self) {
        std::mem::swap(&mut self.current, &mut self.next);
    }
}

/// Update extracellular diffusion on GPU (macOS/Metal).
///
/// Dispatches the `diffusion_3d` kernel with one thread per voxel. Each thread
/// processes all 6 NT channels for its voxel, applying the 3D Laplacian stencil
/// plus natural decay.
///
/// After dispatch, the caller must call `grid.swap_buffers()` to make the
/// updated values available as `grid.current`.
#[cfg(target_os = "macos")]
pub fn dispatch_diffusion_3d(gpu: &GpuContext, grid: &mut ExtracellularGrid, dt: f32) {
    let total_voxels = grid.total_voxels() as u64;
    if total_voxels == 0 {
        return;
    }

    // Source grid (read-only)
    let buf_grid_in = gpu.buffer_from_slice(&grid.current);

    // Destination grid (write-only)
    let buf_grid_out = gpu.buffer_from_slice(&grid.next);

    let params = DiffusionParams {
        x_dim: grid.x_dim as u32,
        y_dim: grid.y_dim as u32,
        z_dim: grid.z_dim as u32,
        voxel_size: grid.voxel_size,
        dt,
    };
    let param_bytes = unsafe {
        std::slice::from_raw_parts(
            &params as *const DiffusionParams as *const u8,
            std::mem::size_of::<DiffusionParams>(),
        )
    };

    gpu.dispatch_1d(
        &gpu.pipelines.diffusion_3d,
        &[
            (&buf_grid_in, 0),  // buffer(0): source grid (read-only)
            (&buf_grid_out, 0), // buffer(1): destination grid (write)
        ],
        Some((param_bytes, 2)), // buffer(2): Params
        total_voxels,
    );

    // Read back the destination grid from shared GPU memory
    let total_elems = grid.current.len();
    unsafe {
        let ptr = buf_grid_out.contents() as *const f32;
        std::ptr::copy_nonoverlapping(ptr, grid.next.as_mut_ptr(), total_elems);
    }

    // Swap buffers so grid.current contains the updated values
    grid.swap_buffers();
}

/// CPU fallback for extracellular diffusion (non-macOS platforms).
#[cfg(not(target_os = "macos"))]
pub fn dispatch_diffusion_3d(_gpu: &super::GpuContext, grid: &mut ExtracellularGrid, dt: f32) {
    cpu_diffusion_3d(grid, dt);
}

/// CPU reference implementation of 3D extracellular diffusion.
///
/// Applies the 6-neighbor Laplacian stencil with Neumann (zero-flux) boundary
/// conditions and natural decay. Reads from `grid.current`, writes to
/// `grid.next`, then swaps buffers.
pub fn cpu_diffusion_3d(grid: &mut ExtracellularGrid, dt: f32) {
    const NT_COUNT: usize = 6;
    const DIFFUSION_COEFF: f32 = 0.001; // mm^2/ms
    const NATURAL_DECAY: f32 = 0.01; // fraction per ms

    let x_dim = grid.x_dim;
    let y_dim = grid.y_dim;
    let z_dim = grid.z_dim;
    let total_voxels = grid.total_voxels();
    let dx2 = grid.voxel_size * grid.voxel_size;
    let coeff = DIFFUSION_COEFF / dx2 * dt;

    for z in 0..z_dim {
        for y in 0..y_dim {
            for x in 0..x_dim {
                let voxel_idx = z * y_dim * x_dim + y * x_dim + x;

                for nt in 0..NT_COUNT {
                    let base = nt * total_voxels;
                    let idx = base + voxel_idx;
                    let c = grid.current[idx];

                    // 6-neighbor Laplacian with Neumann boundary (use c for missing)
                    let right = if x + 1 < x_dim {
                        grid.current[base + z * y_dim * x_dim + y * x_dim + (x + 1)]
                    } else {
                        c
                    };
                    let left = if x > 0 {
                        grid.current[base + z * y_dim * x_dim + y * x_dim + (x - 1)]
                    } else {
                        c
                    };
                    let up = if y + 1 < y_dim {
                        grid.current[base + z * y_dim * x_dim + (y + 1) * x_dim + x]
                    } else {
                        c
                    };
                    let down = if y > 0 {
                        grid.current[base + z * y_dim * x_dim + (y - 1) * x_dim + x]
                    } else {
                        c
                    };
                    let front = if z + 1 < z_dim {
                        grid.current[base + (z + 1) * y_dim * x_dim + y * x_dim + x]
                    } else {
                        c
                    };
                    let back = if z > 0 {
                        grid.current[base + (z - 1) * y_dim * x_dim + y * x_dim + x]
                    } else {
                        c
                    };

                    let laplacian = right + left + up + down + front + back - 6.0 * c;
                    let mut c_new = c + coeff * laplacian;

                    // Natural decay toward zero
                    c_new *= 1.0 - NATURAL_DECAY * dt;

                    grid.next[idx] = c_new.max(0.0);
                }
            }
        }
    }

    // Swap so grid.current has the updated values
    grid.swap_buffers();
}
