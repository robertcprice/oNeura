//! CUDA GPU infrastructure -- device init, PTX compilation, kernel cache.
//!
//! Compiles .cu kernel source at runtime via NVRTC and caches compiled modules.
//! All persistent GPU state lives in [`CudaNeuronState`] and [`CudaSynapseState`]
//! (see the [`state`] module).
//!
//! # Feature gate
//!
//! Everything meaningful in this module is behind `#[cfg(feature = "cuda")]`.
//! When the feature is disabled a thin stub [`CudaContext`] is provided whose
//! constructor always returns `Err`, so callers can branch at runtime without
//! conditional compilation at every call-site.

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaStream, LaunchConfig};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::compile_ptx;
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
pub mod state;

// ---------------------------------------------------------------------------
// CUDA-enabled implementation
// ---------------------------------------------------------------------------

/// CUDA context -- holds device handle, execution stream, and compiled kernel
/// modules.
///
/// Construction compiles all `.cu` kernel sources via NVRTC and loads the
/// resulting PTX into the device driver.  Kernel modules are then available
/// for launch through the [`CudaDevice`] handle using
/// `device.get_func("module_name", "kernel_name")`.
#[cfg(feature = "cuda")]
pub struct CudaContext {
    /// Device handle (Arc-wrapped, clone-cheap).
    pub device: Arc<CudaDevice>,
    /// Non-default stream for overlapping compute and transfer.
    pub stream: CudaStream,
}

#[cfg(feature = "cuda")]
impl CudaContext {
    /// Initialise CUDA on device 0, compile all kernels, return ready context.
    pub fn new() -> Result<Self, String> {
        Self::with_device(0)
    }

    /// Initialise a specific CUDA device by ordinal.
    pub fn with_device(ordinal: usize) -> Result<Self, String> {
        let device = CudaDevice::new(ordinal)
            .map_err(|e| format!("CUDA device {} init failed: {}", ordinal, e))?;
        let stream = device
            .fork_default_stream()
            .map_err(|e| format!("CUDA stream fork failed: {}", e))?;

        // Compile and load every kernel module.
        Self::load_kernels(&device)?;

        Ok(Self { device, stream })
    }

    /// Compile each embedded `.cu` source via NVRTC and register the module
    /// with the device driver so kernels can be launched by name.
    fn load_kernels(device: &Arc<CudaDevice>) -> Result<(), String> {
        let kernels: &[(&str, &str)] = &[
            ("hh_step", include_str!("../../cuda/hh_step.cu")),
            ("synaptic", include_str!("../../cuda/synaptic.cu")),
            ("stdp", include_str!("../../cuda/stdp.cu")),
            ("world", include_str!("../../cuda/world.cu")),
            ("sensory", include_str!("../../cuda/sensory.cu")),
            ("motor", include_str!("../../cuda/motor.cu")),
            ("fep", include_str!("../../cuda/fep.cu")),
        ];

        for (name, source) in kernels {
            let ptx =
                compile_ptx(source).map_err(|e| format!("NVRTC compile '{}': {}", name, e))?;
            device
                .load_ptx(ptx, name, &Self::kernel_fn_names(name))
                .map_err(|e| format!("Load PTX module '{}': {}", name, e))?;
        }

        Ok(())
    }

    /// Load a single PTX module from raw source string.
    ///
    /// Useful for loading user-supplied or dynamically generated kernels at
    /// runtime without recompiling the whole kernel set.
    pub fn load_ptx_module(
        &self,
        module_name: &str,
        source: &str,
        fn_names: &[&str],
    ) -> Result<(), String> {
        let ptx =
            compile_ptx(source).map_err(|e| format!("NVRTC compile '{}': {}", module_name, e))?;
        self.device
            .load_ptx(ptx, module_name, fn_names)
            .map_err(|e| format!("Load PTX '{}': {}", module_name, e))?;
        Ok(())
    }

    /// Return the `__global__` entry-point names within each compiled module.
    ///
    /// These must match the function names defined in the corresponding `.cu`
    /// source files exactly.
    fn kernel_fn_names(module: &str) -> Vec<&'static str> {
        match module {
            "hh_step" => vec!["hh_fused_step"],
            "synaptic" => vec!["synaptic_current", "nt_release"],
            "stdp" => vec!["stdp_trace_update"],
            "world" => vec!["world_diffusion", "world_food_update"],
            "sensory" => vec!["sensory_encode"],
            "motor" => vec!["motor_decode"],
            "fep" => vec!["fep_structured_stim", "fep_noise_stim"],
            _ => vec![],
        }
    }

    /// Standard 1-D launch configuration for `n` elements.
    ///
    /// Uses 256 threads per block (a common sweet-spot for occupancy on most
    /// NVIDIA architectures) and rounds the grid size up to cover all elements.
    #[inline]
    pub fn launch_cfg(n: u32) -> LaunchConfig {
        const THREADS: u32 = 256;
        let blocks = (n + THREADS - 1) / THREADS;
        LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Launch configuration with shared memory allocation.
    #[inline]
    pub fn launch_cfg_shared(n: u32, shared_bytes: u32) -> LaunchConfig {
        const THREADS: u32 = 256;
        let blocks = (n + THREADS - 1) / THREADS;
        LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: shared_bytes,
        }
    }

    /// 2-D launch configuration for grid-shaped problems (e.g. world diffusion).
    #[inline]
    pub fn launch_cfg_2d(width: u32, height: u32) -> LaunchConfig {
        const TILE: u32 = 16;
        let grid_x = (width + TILE - 1) / TILE;
        let grid_y = (height + TILE - 1) / TILE;
        LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (TILE, TILE, 1),
            shared_mem_bytes: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// CPU-only stub (feature = "cuda" disabled)
// ---------------------------------------------------------------------------

/// Stub context when the `cuda` crate feature is not compiled in.
///
/// The constructor always returns `Err` so callers can gracefully fall back
/// to Metal (macOS) or CPU scalar code.
#[cfg(not(feature = "cuda"))]
pub struct CudaContext;

#[cfg(not(feature = "cuda"))]
impl CudaContext {
    /// Always fails -- CUDA was not compiled into this build.
    pub fn new() -> Result<Self, String> {
        Err("CUDA feature not enabled at compile time".to_string())
    }
}

// ---------------------------------------------------------------------------
// Runtime capability probe
// ---------------------------------------------------------------------------

/// Returns `true` when the binary was compiled with the `cuda` feature **and**
/// a CUDA-capable device is actually present on the host.
pub fn has_cuda() -> bool {
    #[cfg(feature = "cuda")]
    {
        CudaDevice::new(0).is_ok()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}
