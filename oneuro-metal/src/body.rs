//! GPU-resident body state -- position, heading, speed, health.
//!
//! All 8 body parameters live in a single `CudaSlice<f32>`:
//!
//! | Index | Field       | Units              | Description                          |
//! |-------|-------------|--------------------|--------------------------------------|
//! |   0   | x           | world cells        | Horizontal position                  |
//! |   1   | y           | world cells        | Vertical position                    |
//! |   2   | heading     | radians [0, 2*pi)  | Direction of travel                  |
//! |   3   | speed       | cells/s            | Current movement speed               |
//! |   4   | hp          | [0, 100]           | Health points (food deprivation, temperature damage) |
//! |   5   | prev_temp   | degrees C          | Temperature at previous step (for gradient sensing) |
//! |   6   | prev_food   | concentration      | Food proximity at previous step      |
//! |   7   | time_of_day | hours [0, 24)      | Circadian clock for light preference |
//!
//! The compact 8-float layout allows the body to be passed as a single buffer
//! to all CUDA kernels (world, sensory, motor, FEP) without scatter/gather.

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice};
#[cfg(feature = "cuda")]
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Body State Indices
// ---------------------------------------------------------------------------

/// X-coordinate in world cells.
pub const BODY_X: usize = 0;
/// Y-coordinate in world cells.
pub const BODY_Y: usize = 1;
/// Heading in radians [0, 2*pi).
pub const BODY_HEADING: usize = 2;
/// Speed in world cells per second.
pub const BODY_SPEED: usize = 3;
/// Health points [0, 100]. Decremented by starvation and thermal stress.
pub const BODY_HP: usize = 4;
/// Temperature at the previous step (for temporal gradient detection).
pub const BODY_PREV_TEMP: usize = 5;
/// Food proximity at the previous step (for temporal gradient detection).
pub const BODY_PREV_FOOD: usize = 6;
/// Time of day in hours [0, 24) for circadian-dependent behavior.
pub const BODY_TOD: usize = 7;
/// Total number of body state elements.
pub const BODY_SIZE: usize = 8;

// ---------------------------------------------------------------------------
// GpuBody
// ---------------------------------------------------------------------------

/// GPU-resident body state for a single organism.
///
/// All body parameters are packed into a single 8-element `CudaSlice<f32>`
/// buffer that is updated in-place by the motor decoder kernel and read
/// by sensory encoding and world update kernels.
#[cfg(feature = "cuda")]
pub struct GpuBody {
    /// The 8-element body state vector on the CUDA device.
    pub state: CudaSlice<f32>,
}

#[cfg(feature = "cuda")]
impl GpuBody {
    /// Create a new body at position (x, y) with the given heading.
    ///
    /// Initializes with:
    /// - speed: 0.0 (stationary)
    /// - hp: 100.0 (full health)
    /// - prev_temp: 22.0 (ambient)
    /// - prev_food: 0.0
    /// - time_of_day: 0.0
    pub fn new(device: &Arc<CudaDevice>, x: f32, y: f32, heading: f32) -> Result<Self, String> {
        let initial = vec![x, y, heading, 0.0, 100.0, 22.0, 0.0, 0.0];
        Ok(Self {
            state: device
                .htod_copy(initial)
                .map_err(|e| format!("Body alloc: {}", e))?,
        })
    }

    /// Create a body centered in a world of the given dimensions.
    pub fn centered(device: &Arc<CudaDevice>, world_w: u32, world_h: u32) -> Result<Self, String> {
        Self::new(device, world_w as f32 / 2.0, world_h as f32 / 2.0, 0.0)
    }

    /// Download body state to CPU for result collection and logging.
    pub fn download(&self, device: &Arc<CudaDevice>) -> Result<[f32; BODY_SIZE], String> {
        let v = device
            .dtoh_sync_copy(&self.state)
            .map_err(|e| format!("D2H body: {}", e))?;
        let mut arr = [0.0f32; BODY_SIZE];
        arr.copy_from_slice(&v[..BODY_SIZE]);
        Ok(arr)
    }

    /// Upload body state from CPU (for manual position resets, etc.).
    pub fn upload(
        &mut self,
        device: &Arc<CudaDevice>,
        state: &[f32; BODY_SIZE],
    ) -> Result<(), String> {
        self.state = device
            .htod_copy(state.to_vec())
            .map_err(|e| format!("H2D body: {}", e))?;
        Ok(())
    }

    /// Get the current position as (x, y) by downloading from GPU.
    pub fn position(&self, device: &Arc<CudaDevice>) -> Result<(f32, f32), String> {
        let s = self.download(device)?;
        Ok((s[BODY_X], s[BODY_Y]))
    }

    /// Get the current heading in radians by downloading from GPU.
    pub fn heading(&self, device: &Arc<CudaDevice>) -> Result<f32, String> {
        let s = self.download(device)?;
        Ok(s[BODY_HEADING])
    }

    /// Get the current health points by downloading from GPU.
    pub fn health(&self, device: &Arc<CudaDevice>) -> Result<f32, String> {
        let s = self.download(device)?;
        Ok(s[BODY_HP])
    }

    /// Set the time of day (for circadian experiments).
    pub fn set_time_of_day(&mut self, device: &Arc<CudaDevice>, hour: f32) -> Result<(), String> {
        let mut s = self.download(device)?;
        s[BODY_TOD] = hour % 24.0;
        self.upload(device, &s)
    }

    /// Compute Euclidean distance from current position to a target point.
    /// Requires a GPU download (use sparingly in hot loops).
    pub fn distance_to(
        &self,
        device: &Arc<CudaDevice>,
        target_x: f32,
        target_y: f32,
    ) -> Result<f32, String> {
        let (x, y) = self.position(device)?;
        let dx = x - target_x;
        let dy = y - target_y;
        Ok((dx * dx + dy * dy).sqrt())
    }
}
