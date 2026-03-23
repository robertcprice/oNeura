//! GPU motor decoding -- VNC voltage readout to body physics update.
//!
//! Reads mean voltage from motor neuron populations and converts to
//! heading/speed updates via differential steering. This mirrors the
//! Drosophila ventral nerve cord (VNC) where contralateral motor neuron
//! populations drive left vs. right turning and forward vs. backward locomotion.
//!
//! # Decoding Model
//!
//! Four motor populations are defined:
//!   0: LEFT   -- drives left turning (positive heading change)
//!   1: RIGHT  -- drives right turning (negative heading change)
//!   2: FORWARD  -- drives forward motion
//!   3: BACKWARD -- drives backward motion (reversal)
//!
//! The decoder computes:
//!   turn_signal = mean_voltage(LEFT) - mean_voltage(RIGHT)
//!   speed_signal = mean_voltage(FORWARD) - mean_voltage(BACKWARD)
//!
//! These are converted to heading and speed via:
//!   heading += clamp(turn_signal * turn_gain, -max_turn_rate, max_turn_rate) * dt
//!   speed = clamp(speed_signal * speed_gain, -max_speed, max_speed)
//!
//! Note: the decoder uses a zero-threshold model (any voltage difference drives
//! action) following the DishBrain finding that zero-threshold decoding is
//! essential for breaking motor symmetry and enabling learning.

#[cfg(feature = "cuda")]
use crate::cuda::state::CudaNeuronState;
#[cfg(feature = "cuda")]
use crate::cuda::CudaContext;
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Motor Population Layout
// ---------------------------------------------------------------------------

/// Number of motor populations.
#[cfg(feature = "cuda")]
pub const NUM_MOTOR_POPS: usize = 4;

/// Named indices for the motor populations.
#[cfg(feature = "cuda")]
pub mod population {
    pub const LEFT: usize = 0;
    pub const RIGHT: usize = 1;
    pub const FORWARD: usize = 2;
    pub const BACKWARD: usize = 3;
}

// ---------------------------------------------------------------------------
// GpuMotor
// ---------------------------------------------------------------------------

/// GPU-resident motor decoder state.
///
/// Stores the neuron index ranges for each motor population on the device.
/// The decode kernel runs with a single thread (the computation is trivial)
/// and reads mean voltages from each population to update the body state.
#[cfg(feature = "cuda")]
pub struct GpuMotor {
    /// Start indices of motor populations in the global neuron array.
    pub motor_starts: CudaSlice<u32>,
    /// Number of neurons in each motor population.
    pub motor_sizes: CudaSlice<u32>,
    /// Maximum turning rate (radians per second).
    pub max_turn_rate: f32,
    /// Maximum forward/backward speed (world units per second).
    pub max_speed: f32,
    /// Gain applied to voltage difference for turning (uA/cm^2 -> rad/s).
    pub turn_gain: f32,
    /// Gain applied to voltage difference for speed (uA/cm^2 -> units/s).
    pub speed_gain: f32,
}

#[cfg(feature = "cuda")]
impl GpuMotor {
    /// Create a motor decoder with population layout and kinematic limits.
    ///
    /// # Arguments
    /// * `device` - CUDA device handle.
    /// * `populations` - (start, size) for LEFT, RIGHT, FORWARD, BACKWARD populations.
    /// * `max_turn_rate` - Maximum heading change per second (radians).
    /// * `max_speed` - Maximum movement speed (world units per second).
    pub fn new(
        device: &Arc<CudaDevice>,
        populations: &[(u32, u32); NUM_MOTOR_POPS],
        max_turn_rate: f32,
        max_speed: f32,
    ) -> Result<Self, String> {
        let starts: Vec<u32> = populations.iter().map(|p| p.0).collect();
        let sizes: Vec<u32> = populations.iter().map(|p| p.1).collect();

        Ok(Self {
            motor_starts: device.htod_copy(starts).map_err(|e| format!("{}", e))?,
            motor_sizes: device.htod_copy(sizes).map_err(|e| format!("{}", e))?,
            max_turn_rate,
            max_speed,
            // Default gains: reasonable for -65 to +40 mV voltage range.
            // A 10 mV left-right difference produces ~0.1 rad/s turning.
            turn_gain: 0.01,
            speed_gain: 0.01,
        })
    }

    /// Create with custom gain parameters.
    pub fn with_gains(
        device: &Arc<CudaDevice>,
        populations: &[(u32, u32); NUM_MOTOR_POPS],
        max_turn_rate: f32,
        max_speed: f32,
        turn_gain: f32,
        speed_gain: f32,
    ) -> Result<Self, String> {
        let mut m = Self::new(device, populations, max_turn_rate, max_speed)?;
        m.turn_gain = turn_gain;
        m.speed_gain = speed_gain;
        Ok(m)
    }

    /// Decode motor commands from neural voltages and update body state.
    ///
    /// Launches a single-thread CUDA kernel that:
    /// 1. Computes mean voltage for each of the 4 motor populations.
    /// 2. Derives turn_signal and speed_signal from voltage differences.
    /// 3. Updates body heading and position with kinematic clamping.
    /// 4. Wraps heading to [0, 2*pi) and clamps position to world bounds.
    ///
    /// # Arguments
    /// * `ctx` - CUDA context.
    /// * `neurons` - Full neuron state (voltage array accessed by motor indices).
    /// * `body` - Body state buffer (8 floats, modified in-place on GPU).
    /// * `world_w` - World width for position clamping.
    /// * `world_h` - World height for position clamping.
    /// * `dt` - Timestep in seconds.
    pub fn decode(
        &self,
        ctx: &CudaContext,
        neurons: &CudaNeuronState,
        body: &CudaSlice<f32>,
        world_w: f32,
        world_h: f32,
        dt: f32,
    ) -> Result<(), String> {
        // Motor decode is a single-thread kernel. The mean-voltage reduction
        // over each population (typically 10-50 neurons) is fast enough that
        // a parallel reduction would add overhead without benefit.
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        let func = ctx
            .device
            .get_func("motor", "motor_decode")
            .ok_or("motor_decode kernel not found")?;

        unsafe {
            func.launch(
                cfg,
                (
                    body,
                    &neurons.voltage,
                    &self.motor_starts,
                    &self.motor_sizes,
                    self.max_turn_rate,
                    self.max_speed,
                    self.turn_gain,
                    self.speed_gain,
                    world_w,
                    world_h,
                    dt,
                    neurons.n as i32,
                ),
            )
        }
        .map_err(|e| format!("motor_decode launch: {}", e))?;

        Ok(())
    }

    /// Read decoded motor signals from GPU for logging.
    ///
    /// Returns (turn_signal, speed_signal) computed from the current neuron
    /// voltages, without modifying the body state. Useful for experiment
    /// instrumentation and plotting.
    pub fn read_motor_signals(
        &self,
        ctx: &CudaContext,
        neurons: &CudaNeuronState,
    ) -> Result<(f32, f32), String> {
        let voltages = neurons.download_voltage(&ctx.device)?;
        let starts = ctx
            .device
            .dtoh_sync_copy(&self.motor_starts)
            .map_err(|e| format!("{}", e))?;
        let sizes = ctx
            .device
            .dtoh_sync_copy(&self.motor_sizes)
            .map_err(|e| format!("{}", e))?;

        let mean_v = |pop: usize| -> f32 {
            let start = starts[pop] as usize;
            let size = sizes[pop] as usize;
            if size == 0 {
                return -65.0; // resting potential if no neurons
            }
            let sum: f32 = voltages[start..start + size].iter().sum();
            sum / size as f32
        };

        let turn_signal = mean_v(population::LEFT) - mean_v(population::RIGHT);
        let speed_signal = mean_v(population::FORWARD) - mean_v(population::BACKWARD);

        Ok((turn_signal, speed_signal))
    }
}
