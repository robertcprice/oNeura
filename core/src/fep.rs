//! Free Energy Principle (FEP) stimulation protocol.
//!
//! Implements the DishBrain-style entropy contrast protocol where organisms
//! learn to minimize prediction error (free energy) through differential
//! stimulation patterns.
//!
//! # Protocol
//!
//! The FEP protocol has two modes:
//!
//! - **Structured (HIT)**: Low-entropy, predictable pulsed stimulation.
//!   5 ms on / 5 ms off square wave with position-encoded amplitude (linear
//!   ramp across the target population). The organism's sensory system can
//!   predict this pattern, producing low free energy.
//!
//! - **Noise (MISS)**: High-entropy, unpredictable random stimulation.
//!   Each step, 30% of target neurons receive random-amplitude current.
//!   The pattern is inherently unpredictable, producing high free energy.
//!
//! The organism learns to prefer the structured state (perform the task
//! correctly to receive HITs) because it minimizes prediction error. This
//! is the core mechanism from Kagan et al. (2022) "In vitro neurons learn
//! and exhibit sentience when embodied in a simulated game-world."
//!
//! # DishBrain Key Lessons Applied
//!
//! - Zero-threshold decoder (in motor.rs) breaks motor symmetry.
//! - Pulsed stimulation (5ms on/off) avoids depolarization block from
//!   sustained Na+ channel inactivation.
//! - No dopamine, no reward signal: learning is PURELY from entropy
//!   contrast + STDP.
//! - Hebbian weight nudge (not implemented here; see STDP module)
//!   accelerates FEP convergence.

#[cfg(feature = "cuda")]
use crate::cuda::state::CudaNeuronState;
#[cfg(feature = "cuda")]
use crate::cuda::CudaContext;
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync};
#[cfg(feature = "cuda")]
use std::sync::Arc;

// ---------------------------------------------------------------------------
// FEP Mode
// ---------------------------------------------------------------------------

/// Stimulation mode for the FEP protocol.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FepMode {
    /// Structured pulsed stimulation -- low entropy, predictable.
    /// 5 ms on / 5 ms off, position-encoded amplitude (linear ramp).
    Structured,
    /// Random noise -- high entropy, unpredictable.
    /// 30% of target neurons get random-amplitude current each step.
    Noise,
}

impl FepMode {
    /// Whether this mode represents a "good outcome" (HIT).
    pub fn is_hit(&self) -> bool {
        matches!(self, FepMode::Structured)
    }

    /// Shannon entropy estimate (bits) for each mode.
    ///
    /// Structured: near-zero (deterministic pattern).
    /// Noise: ~3.3 bits (30% firing probability x random amplitude).
    pub fn entropy_bits(&self) -> f32 {
        match self {
            FepMode::Structured => 0.1,
            FepMode::Noise => 3.3,
        }
    }
}

// ---------------------------------------------------------------------------
// FEP Configuration
// ---------------------------------------------------------------------------

/// Configuration for the FEP protocol timing and parameters.
#[cfg(feature = "cuda")]
#[derive(Clone, Debug)]
pub struct FepConfig {
    /// Duration of the "on" phase in simulation steps.
    /// At dt=0.1ms, 50 steps = 5ms. Default: 50.
    pub pulse_on_steps: u64,
    /// Duration of the "off" phase in simulation steps. Default: 50.
    pub pulse_off_steps: u64,
    /// Fraction of neurons stimulated per step in noise mode [0, 1]. Default: 0.3.
    pub noise_fraction: f32,
    /// Amplitude scaling for structured stimulation (uA/cm^2). Default: 30.0.
    /// This maps to psc_scale=30, which is critical for cascade propagation.
    pub structured_amplitude: f32,
    /// Amplitude scaling for noise stimulation (uA/cm^2). Default: 30.0.
    pub noise_amplitude: f32,
}

#[cfg(feature = "cuda")]
impl Default for FepConfig {
    fn default() -> Self {
        Self {
            pulse_on_steps: 50,
            pulse_off_steps: 50,
            noise_fraction: 0.3,
            structured_amplitude: 30.0,
            noise_amplitude: 30.0,
        }
    }
}

// ---------------------------------------------------------------------------
// GpuFep
// ---------------------------------------------------------------------------

/// GPU-resident FEP stimulation protocol state.
///
/// Holds the indices of stimulable neurons and the pre-computed structured
/// pattern on the CUDA device. The stimulation mode (Structured vs. Noise)
/// is selected per-step by the experiment controller based on task performance.
#[cfg(feature = "cuda")]
pub struct GpuFep {
    /// Indices of neurons that receive FEP stimulation (typically sensory relay
    /// neurons, NOT motor neurons).
    pub target_neurons: CudaSlice<u32>,
    /// Pre-computed structured stimulation pattern: position-encoded amplitude.
    /// Shape: [n_targets]. Values are in [0.5, 1.0] (linear ramp).
    pub stim_pattern: CudaSlice<f32>,
    /// Number of target neurons.
    pub n_targets: u32,
    /// Protocol configuration.
    pub config: FepConfig,
    /// Running step counter within the current stimulation cycle.
    step_in_cycle: u64,
    /// Cumulative number of structured (HIT) stimulations applied.
    pub total_hits: u64,
    /// Cumulative number of noise (MISS) stimulations applied.
    pub total_misses: u64,
}

#[cfg(feature = "cuda")]
impl GpuFep {
    /// Create a new FEP protocol with target neuron indices and default configuration.
    ///
    /// # Arguments
    /// * `device` - CUDA device handle.
    /// * `target_indices` - Indices into the global neuron array for stimulable neurons.
    /// * `amplitude` - Base stimulation amplitude (uA/cm^2).
    pub fn new(
        device: &Arc<CudaDevice>,
        target_indices: &[u32],
        amplitude: f32,
    ) -> Result<Self, String> {
        Self::with_config(
            device,
            target_indices,
            FepConfig {
                structured_amplitude: amplitude,
                noise_amplitude: amplitude,
                ..Default::default()
            },
        )
    }

    /// Create with a custom configuration.
    pub fn with_config(
        device: &Arc<CudaDevice>,
        target_indices: &[u32],
        config: FepConfig,
    ) -> Result<Self, String> {
        let n = target_indices.len();

        // Generate structured pattern: position-encoded linear ramp.
        // Neuron 0 gets 0.5 * amplitude, neuron N-1 gets 1.0 * amplitude.
        // This spatial gradient provides a predictable, low-entropy signal.
        let pattern: Vec<f32> = (0..n)
            .map(|i| 0.5 + 0.5 * (i as f32 / n.max(1) as f32))
            .collect();

        Ok(Self {
            target_neurons: device
                .htod_copy(target_indices.to_vec())
                .map_err(|e| format!("{}", e))?,
            stim_pattern: device.htod_copy(pattern).map_err(|e| format!("{}", e))?,
            n_targets: n as u32,
            config,
            step_in_cycle: 0,
            total_hits: 0,
            total_misses: 0,
        })
    }

    /// Apply FEP stimulation for one simulation step.
    ///
    /// In Structured mode, applies pulsed stimulation with 5ms on / 5ms off
    /// timing. During the "off" phase, no current is injected (returns early).
    ///
    /// In Noise mode, every step launches the noise kernel which randomly
    /// selects 30% of target neurons for random-amplitude stimulation.
    ///
    /// The step counter is used as PRNG seed for noise mode reproducibility.
    ///
    /// # Arguments
    /// * `ctx` - CUDA context.
    /// * `neurons` - Neuron state (ext_current is modified in-place).
    /// * `mode` - Whether to apply structured (HIT) or noise (MISS) stimulation.
    /// * `step` - Global simulation step (used as PRNG seed for noise).
    pub fn stimulate(
        &mut self,
        ctx: &CudaContext,
        neurons: &CudaNeuronState,
        mode: FepMode,
        step: u64,
    ) -> Result<(), String> {
        if self.n_targets == 0 {
            return Ok(());
        }

        match mode {
            FepMode::Structured => {
                self.total_hits += 1;
                // Pulsed: on for pulse_on_steps, off for pulse_off_steps.
                let cycle_len = self.config.pulse_on_steps + self.config.pulse_off_steps;
                let phase = step % cycle_len;
                if phase >= self.config.pulse_on_steps {
                    // Off phase: no stimulation.
                    return Ok(());
                }

                let cfg = CudaContext::launch_cfg(self.n_targets);
                let func = ctx
                    .device
                    .get_func("fep", "fep_structured_stim")
                    .ok_or("fep_structured_stim kernel not found")?;

                unsafe {
                    func.launch(
                        cfg,
                        (
                            &neurons.ext_current,
                            &self.target_neurons,
                            &self.stim_pattern,
                            self.config.structured_amplitude,
                            self.n_targets as i32,
                        ),
                    )
                }
                .map_err(|e| format!("fep_structured: {}", e))?;
            }
            FepMode::Noise => {
                self.total_misses += 1;

                let cfg = CudaContext::launch_cfg(self.n_targets);
                let func = ctx
                    .device
                    .get_func("fep", "fep_noise_stim")
                    .ok_or("fep_noise_stim kernel not found")?;

                unsafe {
                    func.launch(
                        cfg,
                        (
                            &neurons.ext_current,
                            &self.target_neurons,
                            self.config.noise_amplitude,
                            self.config.noise_fraction,
                            step as u32, // PRNG seed
                            self.n_targets as i32,
                        ),
                    )
                }
                .map_err(|e| format!("fep_noise: {}", e))?;
            }
        }

        Ok(())
    }

    /// Compute the FEP ratio: fraction of total stimulations that were structured (HITs).
    ///
    /// A ratio > 0.5 means the organism is performing the task correctly more
    /// often than not (receiving low-entropy stimulation). The trajectory of
    /// this ratio over time is the primary learning curve metric.
    pub fn hit_ratio(&self) -> f32 {
        let total = self.total_hits + self.total_misses;
        if total == 0 {
            return 0.5;
        }
        self.total_hits as f32 / total as f32
    }

    /// Reset the hit/miss counters (e.g. between experimental trials).
    pub fn reset_counters(&mut self) {
        self.total_hits = 0;
        self.total_misses = 0;
        self.step_in_cycle = 0;
    }

    /// Get the number of target neurons.
    pub fn n_targets(&self) -> u32 {
        self.n_targets
    }
}
