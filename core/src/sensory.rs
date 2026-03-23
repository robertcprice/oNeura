//! GPU sensory encoding -- bilateral olfactory, visual, and thermal.
//!
//! Maps world state to neural external_current via CUDA kernel.
//! Region indices specify which neurons are sensory: antennal lobe left/right
//! (AL_L, AL_R), optic lobe left/right, thermal sensors left/right, plus
//! 6 reserved slots for future modalities (gustatory, mechanosensory, etc.).
//!
//! # Encoding Model
//!
//! Olfactory: Each AL neuron receives current proportional to the odorant
//! concentration at the body's grid position, scaled by a receptor sensitivity
//! factor. Bilateral encoding samples left/right of the heading vector to
//! create a concentration gradient across the two antennae (stereo olfaction).
//!
//! Visual: Optic lobe neurons receive current proportional to local light
//! intensity, with left/right sampling offset by the heading direction.
//!
//! Thermal: Thermal sensor neurons encode the temperature difference from
//! a preferred temperature (Drosophila: ~25C) as signed current.

#[cfg(feature = "cuda")]
use crate::cuda::state::CudaNeuronState;
#[cfg(feature = "cuda")]
use crate::cuda::CudaContext;
#[cfg(feature = "cuda")]
use crate::world::GpuWorld;
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync};
#[cfg(feature = "cuda")]
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Region Layout
// ---------------------------------------------------------------------------

/// Number of sensory region slots. The first 6 are assigned:
///   0: AL_L   (antennal lobe left)
///   1: AL_R   (antennal lobe right)
///   2: OPTIC_L (optic lobe left)
///   3: OPTIC_R (optic lobe right)
///   4: THERM_L (thermal sensor left)
///   5: THERM_R (thermal sensor right)
///   6..11: reserved for future modalities.
#[cfg(feature = "cuda")]
pub const NUM_SENSORY_REGIONS: usize = 12;

/// Named indices into the region arrays for clarity.
#[cfg(feature = "cuda")]
pub mod region {
    pub const AL_L: usize = 0;
    pub const AL_R: usize = 1;
    pub const OPTIC_L: usize = 2;
    pub const OPTIC_R: usize = 3;
    pub const THERM_L: usize = 4;
    pub const THERM_R: usize = 5;
}

// ---------------------------------------------------------------------------
// GpuSensory
// ---------------------------------------------------------------------------

/// GPU-resident sensory encoding state.
///
/// Holds the neuron index ranges for each sensory region on the device,
/// allowing the CUDA kernel to map world state to the correct neurons
/// without any CPU-side iteration.
#[cfg(feature = "cuda")]
pub struct GpuSensory {
    /// Start index of each sensory region in the global neuron array.
    pub region_starts: CudaSlice<u32>,
    /// Number of neurons in each sensory region.
    pub region_sizes: CudaSlice<u32>,
    /// Total number of sensory neurons across all regions.
    pub total_sensory: u32,
    /// Preferred temperature for thermal encoding (degrees C). Default: 25.0.
    pub preferred_temp: f32,
    /// Olfactory gain (uA/cm^2 per concentration unit). Default: 30.0.
    pub olfactory_gain: f32,
    /// Visual gain (uA/cm^2 per light unit). Default: 20.0.
    pub visual_gain: f32,
    /// Thermal gain (uA/cm^2 per degree deviation). Default: 5.0.
    pub thermal_gain: f32,
}

#[cfg(feature = "cuda")]
impl GpuSensory {
    /// Create sensory encoding with a region layout.
    ///
    /// # Arguments
    /// * `device` - CUDA device handle.
    /// * `regions` - Array of (start_index, size) for each of the 12 region slots.
    ///   Unused regions should have size 0.
    pub fn new(
        device: &Arc<CudaDevice>,
        regions: &[(u32, u32); NUM_SENSORY_REGIONS],
    ) -> Result<Self, String> {
        let starts: Vec<u32> = regions.iter().map(|r| r.0).collect();
        let sizes: Vec<u32> = regions.iter().map(|r| r.1).collect();
        let total: u32 = sizes.iter().sum();

        Ok(Self {
            region_starts: device.htod_copy(starts).map_err(|e| format!("{}", e))?,
            region_sizes: device.htod_copy(sizes).map_err(|e| format!("{}", e))?,
            total_sensory: total,
            preferred_temp: 25.0,
            olfactory_gain: 30.0,
            visual_gain: 20.0,
            thermal_gain: 5.0,
        })
    }

    /// Create with default gains and a specified preferred temperature.
    pub fn with_preferred_temp(
        device: &Arc<CudaDevice>,
        regions: &[(u32, u32); NUM_SENSORY_REGIONS],
        preferred_temp: f32,
    ) -> Result<Self, String> {
        let mut s = Self::new(device, regions)?;
        s.preferred_temp = preferred_temp;
        Ok(s)
    }

    /// Encode sensory input from world state into neural external_current.
    ///
    /// The CUDA kernel `sensory_encode` runs one thread per sensory neuron.
    /// Each thread:
    /// 1. Determines which region it belongs to (binary search on region_starts).
    /// 2. Samples the appropriate world grid at the body's position with
    ///    bilateral offset (left regions sample left of heading, right sample right).
    /// 3. Writes the encoded current into `neurons.ext_current[neuron_idx]`.
    ///
    /// This call is additive: it does NOT zero ext_current first. The caller
    /// should call `neurons.clear_ext_current()` at the start of the step if
    /// a clean slate is desired.
    pub fn encode(
        &self,
        ctx: &CudaContext,
        neurons: &CudaNeuronState,
        world: &GpuWorld,
        body: &CudaSlice<f32>,
    ) -> Result<(), String> {
        if self.total_sensory == 0 {
            return Ok(());
        }

        let cfg = CudaContext::launch_cfg(self.total_sensory);
        let func = ctx
            .device
            .get_func("sensory", "sensory_encode")
            .ok_or("sensory_encode kernel not found")?;

        unsafe {
            func.launch(
                cfg,
                (
                    body,
                    &world.odorant_grid,
                    &world.temp_grid,
                    &world.light_grid,
                    &neurons.ext_current,
                    &self.region_starts,
                    &self.region_sizes,
                    world.height as i32,
                    world.width as i32,
                    world.n_channels as i32,
                    neurons.n as i32,
                    self.preferred_temp,
                    self.olfactory_gain,
                    self.visual_gain,
                    self.thermal_gain,
                ),
            )
        }
        .map_err(|e| format!("sensory_encode launch: {}", e))?;

        Ok(())
    }
}
