//! GPU dispatch for batched terrarium substrate chemistry.

#[cfg(target_os = "macos")]
use super::GpuContext;

#[repr(C)]
struct TerrariumParams {
    x_dim: u32,
    y_dim: u32,
    z_dim: u32,
    voxel_size_mm: f32,
    dt_ms: f32,
}

#[cfg(target_os = "macos")]
pub fn dispatch_terrarium_substrate(
    gpu: &GpuContext,
    current: &[f32],
    next: &mut [f32],
    hydration: &[f32],
    microbes: &[f32],
    plant_drive: &[f32],
    x_dim: usize,
    y_dim: usize,
    z_dim: usize,
    voxel_size_mm: f32,
    dt_ms: f32,
) {
    let total_voxels = (x_dim * y_dim * z_dim) as u64;
    if total_voxels == 0 {
        return;
    }

    let buf_in = gpu.buffer_from_slice(current);
    let buf_out = gpu.buffer_from_slice(next);
    let buf_hydration = gpu.buffer_from_slice(hydration);
    let buf_microbes = gpu.buffer_from_slice(microbes);
    let buf_plants = gpu.buffer_from_slice(plant_drive);

    let params = TerrariumParams {
        x_dim: x_dim as u32,
        y_dim: y_dim as u32,
        z_dim: z_dim as u32,
        voxel_size_mm,
        dt_ms,
    };
    let param_bytes = unsafe {
        std::slice::from_raw_parts(
            &params as *const TerrariumParams as *const u8,
            std::mem::size_of::<TerrariumParams>(),
        )
    };

    gpu.dispatch_1d(
        &gpu.pipelines.terrarium_substrate,
        &[
            (&buf_in, 0),
            (&buf_out, 0),
            (&buf_hydration, 0),
            (&buf_microbes, 0),
            (&buf_plants, 0),
        ],
        Some((param_bytes, 5)),
        total_voxels,
    );

    unsafe {
        let ptr = buf_out.contents() as *const f32;
        std::ptr::copy_nonoverlapping(ptr, next.as_mut_ptr(), next.len());
    }
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_terrarium_substrate(
    _gpu: &super::GpuContext,
    _current: &[f32],
    _next: &mut [f32],
    _hydration: &[f32],
    _microbes: &[f32],
    _plant_drive: &[f32],
    _x_dim: usize,
    _y_dim: usize,
    _z_dim: usize,
    _voxel_size_mm: f32,
    _dt_ms: f32,
) {
}
