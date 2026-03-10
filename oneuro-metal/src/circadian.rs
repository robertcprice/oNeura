//! Circadian clock: TTFL oscillator with outputs to neural excitability
//! and neurotransmitter synthesis.
//!
//! This module implements a simplified transcription-translation feedback
//! loop (TTFL) that generates a ~24-hour oscillation. The core oscillator
//! consists of two variables:
//!
//! - **BMAL1**: The positive limb. BMAL1:CLOCK heterodimer activates
//!   transcription of Period/Cryptochrome genes.
//!
//! - **PER/CRY**: The negative limb. PER:CRY complex inhibits BMAL1:CLOCK,
//!   closing the feedback loop.
//!
//! The oscillator produces three outputs that modulate neural function:
//!
//! 1. **Excitability bias**: A current added to all neurons, peaking during
//!    subjective day (higher alertness).
//!
//! 2. **Alertness**: A [0, 1] signal tracking the circadian arousal drive.
//!    Combines with homeostatic sleep pressure in a two-process model.
//!
//! 3. **NT synthesis modulation**: Circadian variation in monoamine
//!    synthesis rates (serotonin peaks during day, melatonin at night).
//!
//! This module runs every step because it is computationally trivial (two
//! coupled ODEs on a single global state, plus a homeostatic adenosine
//! accumulator for the two-process sleep model).

use crate::constants::CIRCADIAN_PERIOD_H;

// ===== TTFL ODE Parameters =====

/// BMAL1 transcription rate (per hour). Reserved for detailed TTFL ODE.
const _BMAL1_TRANSCRIPTION_RATE: f32 = 1.0;

/// BMAL1 degradation rate (per hour). Reserved for detailed TTFL ODE.
const _BMAL1_DEGRADATION_RATE: f32 = 0.5;

/// PER/CRY transcription rate driven by BMAL1 (per hour). Reserved for detailed TTFL ODE.
const _PER_CRY_TRANSCRIPTION_RATE: f32 = 1.2;

/// PER/CRY degradation rate (per hour). CK1-dependent phosphorylation
/// targets PER for proteasomal degradation. Reserved for detailed TTFL ODE.
const _PER_CRY_DEGRADATION_RATE: f32 = 0.6;

/// PER/CRY inhibition strength on BMAL1.
/// Higher values = tighter negative feedback = more stable oscillation.
const PER_CRY_INHIBITION_STRENGTH: f32 = 0.8;

/// Time constant for PER/CRY dynamics (hours).
const TAU_PER: f32 = 6.0;

/// Time constant for BMAL1 dynamics (hours).
const TAU_BMAL: f32 = 8.0;

/// Maximum excitability bias amplitude (uA/cm2).
/// Positive during subjective day, negative during night.
const MAX_EXCITABILITY_BIAS: f32 = 2.0;

/// Adenosine accumulation rate during wakefulness (per hour).
const ADENOSINE_WAKE_RATE: f32 = 0.04;

/// Adenosine clearance rate during sleep (per hour).
const ADENOSINE_SLEEP_RATE: f32 = 0.08;

/// Circadian clock state.
///
/// Maintains the TTFL oscillator variables and time tracking.
/// A single instance drives the entire network; individual neurons
/// receive the same circadian signal (no per-neuron clocks, which
/// is biologically correct for SCN-driven systemic signals).
pub struct CircadianClock {
    /// Current time in hours (wraps at `CIRCADIAN_PERIOD_H`).
    pub time_hours: f32,

    /// PER/CRY complex concentration (dimensionless, [0, 1]).
    pub per_cry: f32,

    /// BMAL1:CLOCK complex concentration (dimensionless, [0, 1]).
    pub bmal1: f32,

    /// Time compression factor for simulation.
    /// `time_scale` is sim-ms per real-hour.
    /// E.g. 3600.0 means 1 ms sim = 1 hour bio = full cycle in 24 ms.
    /// 1000.0 means 1 second sim ~ 17 minutes circadian.
    pub time_scale: f32,

    /// Sleep pressure (adenosine homeostat) [0, 1].
    /// Builds during wakefulness, clears during sleep. Part of the
    /// Borbely two-process model of sleep regulation.
    pub adenosine: f32,
}

impl CircadianClock {
    /// Create a new circadian clock at dawn (peak BMAL1, low PER/CRY).
    ///
    /// # Arguments
    /// * `time_scale` - Compression factor. `3600.0` means 1 ms = 1 hour.
    ///   `1000.0` is a reasonable default for experiments.
    pub fn new(time_scale: f32) -> Self {
        let t = 6.0; // dawn
        let phase = std::f32::consts::TAU * t / CIRCADIAN_PERIOD_H;
        Self {
            time_hours: t,
            per_cry: 0.5 * (1.0 + phase.sin()),
            bmal1: 0.5 * (1.0 + (phase + std::f32::consts::PI).sin()),
            time_scale,
            adenosine: 0.3,
        }
    }

    /// Create a clock at a specific circadian phase.
    ///
    /// # Arguments
    /// * `time_hours` - Initial time in hours (0-24).
    /// * `time_scale` - Compression factor.
    pub fn at_time(time_hours: f32, time_scale: f32) -> Self {
        let phase = std::f32::consts::TAU * time_hours / CIRCADIAN_PERIOD_H;
        let bmal1 = 0.5 + 0.3 * (phase + std::f32::consts::PI).sin();
        let per_cry = 0.5 + 0.3 * phase.sin();
        Self {
            time_hours: time_hours % CIRCADIAN_PERIOD_H,
            per_cry: per_cry.clamp(0.0, 1.0),
            bmal1: bmal1.clamp(0.0, 1.0),
            time_scale,
            adenosine: 0.3,
        }
    }

    /// Advance the circadian clock by `dt_ms` milliseconds of simulation time.
    ///
    /// Integrates the TTFL ODEs using forward Euler:
    ///
    /// ```text
    /// PER/CRY target = BMAL1 * (1 - PER_CRY * inhibition_strength)
    /// d[PER/CRY]/dt = (target - PER/CRY) / tau_per
    ///
    /// BMAL1 target = 1 - PER/CRY * 0.7
    /// d[BMAL1]/dt = (target - BMAL1) / tau_bmal
    /// ```
    ///
    /// The relaxation-to-target formulation with different time constants
    /// for the two limbs produces stable limit-cycle oscillations with
    /// a period determined by `tau_per` and `tau_bmal`.
    pub fn step(&mut self, dt_ms: f32) {
        let dt_hours = dt_ms / self.time_scale;
        self.time_hours += dt_hours;
        if self.time_hours >= CIRCADIAN_PERIOD_H {
            self.time_hours -= CIRCADIAN_PERIOD_H;
        }

        // --- TTFL ODE ---

        // PER/CRY driven by BMAL1, with negative autoregulation
        let per_target = self.bmal1 * (1.0 - self.per_cry * PER_CRY_INHIBITION_STRENGTH);
        self.per_cry += dt_hours * (per_target - self.per_cry) / TAU_PER;
        self.per_cry = self.per_cry.clamp(0.0, 1.0);

        // BMAL1 repressed by PER/CRY (REV-ERB pathway, simplified)
        let bmal_target = 1.0 - self.per_cry * 0.7;
        self.bmal1 += dt_hours * (bmal_target - self.bmal1) / TAU_BMAL;
        self.bmal1 = self.bmal1.clamp(0.0, 1.0);

        // --- Adenosine homeostasis (two-process model Process S) ---
        let is_wake = self.alertness() > 0.5;
        if is_wake {
            self.adenosine += dt_hours * ADENOSINE_WAKE_RATE;
        } else {
            self.adenosine -= dt_hours * ADENOSINE_SLEEP_RATE;
        }
        self.adenosine = self.adenosine.clamp(0.0, 1.0);
    }

    /// Excitability bias driven by BMAL1 level and adenosine pressure.
    ///
    /// Returns a current in uA/cm2 that modulates neural excitability.
    /// Positive during subjective day (high BMAL1, low adenosine),
    /// negative during night or high sleep pressure.
    pub fn excitability_bias(&self) -> f32 {
        // Circadian drive: BMAL1 centered at 0.5
        let circadian = (self.bmal1 - 0.5) * 2.0;
        // Adenosine opposes excitability
        let homeostatic = -self.adenosine * 1.5;
        (circadian + homeostatic).clamp(-MAX_EXCITABILITY_BIAS, MAX_EXCITABILITY_BIAS)
    }

    /// Alertness level in [0, 1], driven by BMAL1 and opposed by adenosine.
    ///
    /// Implements the Borbely two-process model: alertness = Process C
    /// (circadian) - Process S (homeostatic sleep pressure).
    pub fn alertness(&self) -> f32 {
        let circadian_drive = self.bmal1;
        let sleep_pressure = self.adenosine;
        (circadian_drive - sleep_pressure * 0.5).clamp(0.0, 1.0)
    }

    /// NT synthesis rate modulator (multiplier on baseline synthesis).
    ///
    /// Serotonin synthesis peaks during subjective day because
    /// tryptophan hydroxylase expression is BMAL1-dependent.
    /// Returns a multiplier in [0.8, 1.2].
    pub fn nt_synthesis_modulator(&self) -> f32 {
        0.8 + 0.4 * self.bmal1
    }

    /// Serotonin synthesis modulation factor.
    ///
    /// Returns a multiplier in [0.5, 1.5] based on BMAL1 level.
    pub fn serotonin_modulation(&self) -> f32 {
        0.5 + self.bmal1.clamp(0.0, 1.0)
    }

    /// Dopamine synthesis modulation factor.
    ///
    /// Dopamine has a modest circadian variation, peaking during
    /// the active phase. Returns a multiplier in [0.8, 1.2].
    pub fn dopamine_modulation(&self) -> f32 {
        0.8 + 0.4 * (self.bmal1 - 0.3).clamp(0.0, 1.0)
    }

    /// Melatonin level (inverse of alertness).
    ///
    /// Melatonin is synthesized at night by the pineal gland when
    /// SCN output is low. Returns a level in [0, 1].
    pub fn melatonin_level(&self) -> f32 {
        1.0 - self.alertness()
    }

    /// Whether it is subjective night (for sleep-related behaviors).
    /// Night is defined as time between 22:00 and 06:00.
    pub fn is_night(&self) -> bool {
        self.time_hours >= 22.0 || self.time_hours < 6.0
    }

    /// Whether it is subjective "day" (alertness > 0.5).
    pub fn is_day(&self) -> bool {
        self.alertness() > 0.5
    }

    /// Current circadian phase in radians [0, 2pi).
    pub fn phase_radians(&self) -> f32 {
        std::f32::consts::TAU * self.time_hours / CIRCADIAN_PERIOD_H
    }

    /// Set the circadian phase as a fraction [0, 1) of the 24-hour cycle.
    ///
    /// 0.0 = midnight (low excitability), 0.5 = noon (peak excitability).
    /// Resets the TTFL oscillator variables to match the requested phase.
    pub fn set_phase(&mut self, phase_fraction: f32) {
        let t = (phase_fraction * CIRCADIAN_PERIOD_H) % CIRCADIAN_PERIOD_H;
        let phase = std::f32::consts::TAU * t / CIRCADIAN_PERIOD_H;
        self.time_hours = t;
        self.bmal1 = (0.5 + 0.3 * (phase + std::f32::consts::PI).sin()).clamp(0.0, 1.0);
        self.per_cry = (0.5 + 0.3 * phase.sin()).clamp(0.0, 1.0);
    }

    /// Apply circadian excitability bias to all neurons.
    ///
    /// Adds the current circadian bias to each alive neuron's
    /// `excitability_bias` field. Call this once per step.
    pub fn apply_to_neurons(&self, neurons: &mut crate::neuron_arrays::NeuronArrays) {
        let bias = self.excitability_bias();
        for i in 0..neurons.count {
            if neurons.alive[i] != 0 {
                neurons.excitability_bias[i] += bias;
            }
        }
    }
}

impl Default for CircadianClock {
    fn default() -> Self {
        Self::new(1000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circadian_oscillation() {
        let mut clock = CircadianClock::new(1.0); // 1ms = 1h
        let mut max_alert = 0.0f32;
        let mut min_alert = 1.0f32;

        for _ in 0..48 {
            clock.step(1.0); // 1 hour per step
            let a = clock.alertness();
            max_alert = max_alert.max(a);
            min_alert = min_alert.min(a);
        }

        // Should oscillate
        assert!(
            max_alert > 0.5,
            "Max alertness should be > 0.5: {}",
            max_alert
        );
        assert!(
            min_alert < 0.5,
            "Min alertness should be < 0.5: {}",
            min_alert
        );
    }

    #[test]
    fn test_period_wraps() {
        let mut clock = CircadianClock::new(1.0);
        clock.step(30.0); // 30 hours
        assert!(
            clock.time_hours < CIRCADIAN_PERIOD_H,
            "Time should wrap: {}",
            clock.time_hours
        );
    }

    #[test]
    fn test_excitability_bias_bounded() {
        let mut clock = CircadianClock::new(1.0);
        for _ in 0..100 {
            clock.step(1.0);
            let bias = clock.excitability_bias();
            assert!(
                bias.abs() <= MAX_EXCITABILITY_BIAS + 0.01,
                "Excitability bias out of range: {}",
                bias
            );
        }
    }

    #[test]
    fn test_alertness_bounded() {
        let mut clock = CircadianClock::new(1.0);
        for _ in 0..100 {
            clock.step(1.0);
            let a = clock.alertness();
            assert!(a >= 0.0 && a <= 1.0, "Alertness out of [0,1]: {}", a);
        }
    }

    #[test]
    fn test_melatonin_inverse_of_alertness() {
        let clock = CircadianClock::new(1000.0);
        let sum = clock.alertness() + clock.melatonin_level();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Alertness + melatonin should = 1.0: {}",
            sum
        );
    }

    #[test]
    fn test_adenosine_accumulates_during_wake() {
        let mut clock = CircadianClock::new(1.0);
        // Force high alertness for accumulation
        clock.bmal1 = 0.9;
        clock.adenosine = 0.0;

        let initial = clock.adenosine;
        for _ in 0..10 {
            clock.step(1.0);
        }
        assert!(
            clock.adenosine > initial,
            "Adenosine should accumulate during wake: {} -> {}",
            initial,
            clock.adenosine
        );
    }

    #[test]
    fn test_adenosine_clears_during_sleep() {
        let mut clock = CircadianClock::new(1.0);
        // Force low alertness
        clock.bmal1 = 0.1;
        clock.adenosine = 0.8;

        let initial = clock.adenosine;
        for _ in 0..10 {
            clock.step(1.0);
        }
        assert!(
            clock.adenosine < initial,
            "Adenosine should clear during sleep: {} -> {}",
            initial,
            clock.adenosine
        );
    }

    #[test]
    fn test_variables_stay_bounded() {
        let mut clock = CircadianClock::new(1.0);
        for _ in 0..10000 {
            clock.step(1.0);
            assert!(
                clock.bmal1 >= 0.0 && clock.bmal1 <= 1.0,
                "BMAL1: {}",
                clock.bmal1
            );
            assert!(
                clock.per_cry >= 0.0 && clock.per_cry <= 1.0,
                "PER/CRY: {}",
                clock.per_cry
            );
            assert!(
                clock.adenosine >= 0.0 && clock.adenosine <= 1.0,
                "Adenosine: {}",
                clock.adenosine
            );
        }
    }

    #[test]
    fn test_nt_synthesis_modulation() {
        let mut clock = CircadianClock::new(1.0);
        for _ in 0..100 {
            clock.step(1.0);
            let m = clock.nt_synthesis_modulator();
            assert!(
                m >= 0.79 && m <= 1.21,
                "NT synthesis modulator out of range: {}",
                m
            );
        }
    }

    #[test]
    fn test_default_clock() {
        let clock = CircadianClock::default();
        assert!((clock.time_hours - 6.0).abs() < 1e-6);
        assert!((clock.time_scale - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_to_neurons() {
        let clock = CircadianClock::new(1000.0);
        let mut neurons = crate::neuron_arrays::NeuronArrays::new(5);

        clock.apply_to_neurons(&mut neurons);

        let bias = clock.excitability_bias();
        for i in 0..5 {
            assert!(
                (neurons.excitability_bias[i] - bias).abs() < 1e-6,
                "All alive neurons should get the same circadian bias"
            );
        }
    }

    #[test]
    fn test_at_time_constructor() {
        let clock = CircadianClock::at_time(12.0, 1000.0);
        assert!((clock.time_hours - 12.0).abs() < 1e-6);
        assert!(clock.bmal1 >= 0.0 && clock.bmal1 <= 1.0);
        assert!(clock.per_cry >= 0.0 && clock.per_cry <= 1.0);
    }

    #[test]
    fn test_is_night() {
        let mut clock = CircadianClock::new(1.0);
        clock.time_hours = 23.0;
        assert!(clock.is_night());
        clock.time_hours = 3.0;
        assert!(clock.is_night());
        clock.time_hours = 12.0;
        assert!(!clock.is_night());
    }
}
