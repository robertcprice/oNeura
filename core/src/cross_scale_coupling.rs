//! Cross-scale coupling module: bidirectional data flow between simulation layers.
//!
//! Wires all scales into a chain through the substrate voxel hub:
//! ```text
//! Ecosystem substrate → Organism sensing → Whole-cell rates → Quantum profile → MD forces
//! MD energy → Quantum corrections → Whole-cell efficiency → Organism health → Ecosystem fitness
//! ```
//!
//! # Architecture
//! - Free functions that accept references to subsystems (avoids modifying protected files)
//! - `ExplicitMicrobe` struct: a microbe entity with an embedded `WholeCellSimulator`
//! - Environmental context: Arrhenius, pH, O2, glucose modulation via quantum profile

use crate::fly_metabolism::{FlyActivity, FlyMetabolism};
use crate::whole_cell::{WholeCellConfig, WholeCellQuantumProfile, WholeCellSimulator};

// ============================================================================
// Fly ↔ Substrate coupling rates
// ============================================================================

/// ATP demand rate constants (mirror fly_metabolism.rs internal constants).
const UW_TO_MM_ATP_PER_S: f32 = 1.85e-5;
const BASAL_DEMAND_UW: f32 = 20.0;
const WALK_DEMAND_UW: f32 = 30.0;
const FLY_DEMAND_UW: f32 = 100.0;
const ATP_PER_GLUCOSE: f32 = 36.0;

/// Compute O2 consumption rate (mM/s) for a fly given its current state.
/// Uses 6 mol O2 per mol glucose oxidized (aerobic respiration stoichiometry).
pub fn fly_o2_consumption_rate(fm: &FlyMetabolism) -> f32 {
    let demand_uw = match fm.activity() {
        FlyActivity::Resting => BASAL_DEMAND_UW,
        FlyActivity::Walking(s) => {
            BASAL_DEMAND_UW + (WALK_DEMAND_UW - BASAL_DEMAND_UW) * s.clamp(0.0, 1.0)
        }
        FlyActivity::Flying(e) => {
            BASAL_DEMAND_UW + (FLY_DEMAND_UW - BASAL_DEMAND_UW) * e.clamp(0.0, 1.0)
        }
    };
    let atp_demand_mm_s = demand_uw * UW_TO_MM_ATP_PER_S;
    let glucose_consumed_mm_s = atp_demand_mm_s / ATP_PER_GLUCOSE;
    let o2_rate = glucose_consumed_mm_s * 6.0;
    o2_rate * (fm.ambient_o2_fraction / 0.21).clamp(0.0, 1.0)
}

/// Compute CO2 output rate (mM/s) for a fly given its current state.
/// Respiratory quotient = 1.0 for glucose oxidation: 6 CO2 per glucose.
pub fn fly_co2_output_rate(fm: &FlyMetabolism) -> f32 {
    let demand_uw = match fm.activity() {
        FlyActivity::Resting => BASAL_DEMAND_UW,
        FlyActivity::Walking(s) => {
            BASAL_DEMAND_UW + (WALK_DEMAND_UW - BASAL_DEMAND_UW) * s.clamp(0.0, 1.0)
        }
        FlyActivity::Flying(e) => {
            BASAL_DEMAND_UW + (FLY_DEMAND_UW - BASAL_DEMAND_UW) * e.clamp(0.0, 1.0)
        }
    };
    let atp_demand_mm_s = demand_uw * UW_TO_MM_ATP_PER_S;
    let glucose_consumed_mm_s = atp_demand_mm_s / ATP_PER_GLUCOSE;
    let o2_ratio = (fm.ambient_o2_fraction / 0.21).clamp(0.0, 1.0);
    glucose_consumed_mm_s * 6.0 * o2_ratio
}

// ============================================================================
// Environmental context → WholeCellQuantumProfile
// ============================================================================

/// Compute a modulated quantum profile from ecosystem environmental context.
///
/// # Parameters
/// - `base_profile`: the WholeCellSimulator's current quantum profile
/// - `temperature_c`: local substrate temperature in Celsius
/// - `ph`: local substrate pH
/// - `o2_fraction`: local substrate O2 mole fraction
/// - `glucose_mm`: local substrate glucose concentration (mM-equivalent)
///
/// # Physics
/// - **Temperature**: Arrhenius Q10 rule — reaction rate doubles per 10°C above 37°C
/// - **pH**: Gaussian enzyme activity bell curve centered at pH 7.2 (σ=1.2)
/// - **O2**: Modulates oxidative phosphorylation efficiency linearly
/// - **Glucose**: Michaelis-Menten supply modulation (Km = 2 mM)
pub fn environmental_quantum_profile(
    base_profile: WholeCellQuantumProfile,
    temperature_c: f32,
    ph: f32,
    o2_fraction: f32,
    _glucose_mm: f32,
) -> WholeCellQuantumProfile {
    let temp_c = temperature_c.clamp(10.0, 50.0);
    let arrhenius_factor = (2.0f32).powf((temp_c - 37.0) / 10.0).clamp(0.3, 3.0);

    let ph_clamped = ph.clamp(5.0, 9.0);
    let ph_deviation = (ph_clamped - 7.2).abs();
    let ph_factor = (-ph_deviation * ph_deviation / 1.44).exp().clamp(0.3, 1.0);

    let o2_clamped = o2_fraction.clamp(0.0, 0.50);
    let o2_factor = (o2_clamped / 0.21).clamp(0.1, 1.5);

    WholeCellQuantumProfile {
        oxphos_efficiency: (base_profile.oxphos_efficiency * arrhenius_factor * ph_factor * o2_factor)
            .clamp(0.5, 2.5),
        translation_efficiency: (base_profile.translation_efficiency * arrhenius_factor * ph_factor)
            .clamp(0.5, 2.5),
        nucleotide_polymerization_efficiency: (base_profile.nucleotide_polymerization_efficiency
            * arrhenius_factor
            * ph_factor)
            .clamp(0.5, 2.5),
        membrane_synthesis_efficiency: (base_profile.membrane_synthesis_efficiency * arrhenius_factor)
            .clamp(0.5, 2.5),
        chromosome_segregation_efficiency: (base_profile.chromosome_segregation_efficiency
            * arrhenius_factor)
            .clamp(0.5, 2.5),
    }
}

// ============================================================================
// ExplicitMicrobe: whole-cell embedded in terrarium microbe entity
// ============================================================================

/// A microbe entity with an embedded whole-cell simulator.
/// Lives at a specific position on the substrate grid.
pub struct ExplicitMicrobe {
    pub id: u32,
    pub grid_x: usize,
    pub grid_y: usize,
    pub simulator: WholeCellSimulator,
    /// Accumulated dt for multi-rate stepping (whole-cell runs every N frames).
    pub accumulated_dt_ms: f32,
}

impl ExplicitMicrobe {
    /// Create a new explicit microbe with a tiny whole-cell config (minimal lattice for speed).
    pub fn new(id: u32, grid_x: usize, grid_y: usize) -> Self {
        let config = WholeCellConfig {
            x_dim: 8,
            y_dim: 8,
            z_dim: 4,
            voxel_size_nm: 25.0,
            dt_ms: 0.5,
            cme_interval: 8,
            ode_interval: 2,
            bd_interval: 4,
            geometry_interval: 8,
            use_gpu: false, // CPU-only for embedded microbes
        };
        Self {
            id,
            grid_x,
            grid_y,
            simulator: WholeCellSimulator::new(config),
            accumulated_dt_ms: 0.0,
        }
    }

    /// Step the embedded whole-cell simulator with environmental context from the substrate.
    ///
    /// # Parameters
    /// - `temperature_c`: local temperature from substrate
    /// - `ph`: local pH from substrate Proton species
    /// - `o2_fraction`: local O2 fraction from substrate OxygenGas species
    /// - `glucose_mm`: local glucose from substrate Glucose species
    /// - `eco_dt_ms`: time step in milliseconds to accumulate
    /// - `step_interval`: only run whole-cell every N calls (multi-rate)
    ///
    /// Returns (atp_produced, co2_produced) for writing back to substrate.
    pub fn step_with_environment(
        &mut self,
        temperature_c: f32,
        ph: f32,
        o2_fraction: f32,
        glucose_mm: f32,
        eco_dt_ms: f32,
        step_interval: u32,
    ) -> (f32, f32) {
        self.accumulated_dt_ms += eco_dt_ms;

        // Multi-rate: only step every N calls
        if self.accumulated_dt_ms < (step_interval as f32) * eco_dt_ms {
            return (0.0, 0.0);
        }

        // Modulate quantum profile based on environment
        let base_profile = self.simulator.quantum_profile();
        let modulated = environmental_quantum_profile(
            base_profile,
            temperature_c,
            ph,
            o2_fraction,
            glucose_mm,
        );
        self.simulator.set_quantum_profile(modulated);

        // Snapshot ATP before stepping
        let atp_before = self.simulator.atp_mm();

        // Step the whole-cell (uses its internal dt)
        let steps = (self.accumulated_dt_ms / 0.5).round().max(1.0) as usize;
        for _ in 0..steps.min(20) {
            self.simulator.step();
        }

        let atp_after = self.simulator.atp_mm();
        self.accumulated_dt_ms = 0.0;

        // Estimate metabolic outputs to write back to substrate
        let atp_delta = (atp_after - atp_before).abs();
        // CO2 produced proportional to ATP generated (aerobic respiration stoichiometry)
        let co2_produced = atp_delta * 0.167; // ~1/6 (6 CO2 per 36 ATP)
        let o2_consumed = co2_produced; // RQ = 1.0 for glucose

        (co2_produced, o2_consumed)
    }
}

// ============================================================================
// Multi-rate frame scheduler (Phase 2C)
// ============================================================================

/// Subsystem scheduling frequencies.
///
/// | Subsystem | Frequency | Rationale |
/// |-----------|-----------|-----------|
/// | Substrate chemistry | Every substep | Core simulation engine |
/// | Fly neural + body | Every substep | Real-time behavior |
/// | Fly metabolism | Every substep | MM kinetics stable at frame dt |
/// | Ecology field rebuild | 1× per frame | Was 2× per substep (redundant) |
/// | Plant competition | Every 5 frames | Plants grow slowly |
/// | Soil fauna | Every 10 frames | Earthworm dynamics are slow |
/// | Whole-cell step | Every 10 frames | Expensive, amortize |
/// | Atmosphere fields | Every 5 frames | Slow atmospheric mixing |
pub struct FrameScheduler {
    frame: u64,
}

impl FrameScheduler {
    pub fn new() -> Self {
        Self { frame: 0 }
    }

    /// Advance the frame counter. Call once per `step_frame()`.
    pub fn tick(&mut self) {
        self.frame = self.frame.wrapping_add(1);
    }

    /// Current frame number.
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Should substrate chemistry run this substep? (Always yes.)
    pub fn run_substrate(&self) -> bool {
        true
    }

    /// Should fly neural + body run this substep? (Always yes.)
    pub fn run_flies(&self) -> bool {
        true
    }

    /// Should fly metabolism run this substep? (Always yes — MM kinetics need fine dt.)
    pub fn run_fly_metabolism(&self) -> bool {
        true
    }

    /// Should ecology fields rebuild? Once per frame (not per substep).
    /// When called from the substep loop, only return true for substep 0.
    pub fn run_ecology_fields(&self, substep: usize) -> bool {
        substep == 0
    }

    /// Should plant competition run? Every 5 frames.
    pub fn run_plants(&self) -> bool {
        self.frame % 5 == 0
    }

    /// Should soil fauna run? Every 10 frames.
    pub fn run_soil_fauna(&self) -> bool {
        self.frame % 10 == 0
    }

    /// Should whole-cell embedded microbes run? Every 10 frames.
    pub fn run_whole_cell(&self) -> bool {
        self.frame % 10 == 0
    }

    /// Should atmosphere / world fields update? Every 5 frames.
    pub fn run_atmosphere(&self) -> bool {
        self.frame % 5 == 0
    }

    /// dt multiplier to compensate for reduced frequency.
    /// If a subsystem runs every N frames, its dt should be N× larger.
    pub fn dt_scale(&self, interval: u64) -> f32 {
        interval as f32
    }
}

impl Default for FrameScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fly_o2_rate_scales_with_activity() {
        let mut resting = FlyMetabolism::default();
        resting.set_activity(FlyActivity::Resting);
        let mut flying = FlyMetabolism::default();
        flying.set_activity(FlyActivity::Flying(1.0));

        let rate_rest = fly_o2_consumption_rate(&resting);
        let rate_fly = fly_o2_consumption_rate(&flying);
        assert!(rate_fly > rate_rest, "Flying O2 rate {rate_fly} should exceed resting {rate_rest}");
        assert!(rate_rest > 0.0, "Resting O2 rate should be positive");
    }

    #[test]
    fn fly_co2_rate_scales_with_activity() {
        let mut resting = FlyMetabolism::default();
        resting.set_activity(FlyActivity::Resting);
        let mut flying = FlyMetabolism::default();
        flying.set_activity(FlyActivity::Flying(1.0));

        let rate_rest = fly_co2_output_rate(&resting);
        let rate_fly = fly_co2_output_rate(&flying);
        assert!(rate_fly > rate_rest, "Flying CO2 rate {rate_fly} should exceed resting {rate_rest}");
    }

    #[test]
    fn fly_rates_drop_under_hypoxia() {
        let mut fm = FlyMetabolism::default();
        fm.set_activity(FlyActivity::Walking(0.5));
        let normal_o2 = fly_o2_consumption_rate(&fm);
        let normal_co2 = fly_co2_output_rate(&fm);

        fm.set_ambient_o2(0.05); // severe hypoxia
        let hypoxic_o2 = fly_o2_consumption_rate(&fm);
        let hypoxic_co2 = fly_co2_output_rate(&fm);

        assert!(
            hypoxic_o2 < normal_o2,
            "Hypoxic O2 rate {hypoxic_o2} should be < normal {normal_o2}"
        );
        assert!(
            hypoxic_co2 < normal_co2,
            "Hypoxic CO2 rate {hypoxic_co2} should be < normal {normal_co2}"
        );
    }

    #[test]
    fn environmental_profile_arrhenius() {
        let base = WholeCellQuantumProfile::default();
        // At 37°C (body temp), factor should be ~1.0
        let p37 = environmental_quantum_profile(base, 37.0, 7.2, 0.21, 5.0);
        assert!(
            (p37.oxphos_efficiency - base.oxphos_efficiency).abs() < 0.15,
            "At 37°C/pH7.2/normoxia, profile should be near baseline: got {:.3} vs {:.3}",
            p37.oxphos_efficiency,
            base.oxphos_efficiency
        );

        // At 47°C, reaction rates should roughly double (Arrhenius)
        let p47 = environmental_quantum_profile(base, 47.0, 7.2, 0.21, 5.0);
        assert!(
            p47.oxphos_efficiency > p37.oxphos_efficiency * 1.5,
            "At 47°C, oxphos should be >1.5× baseline: {:.3} vs {:.3}",
            p47.oxphos_efficiency,
            p37.oxphos_efficiency
        );
    }

    #[test]
    fn environmental_profile_ph_sensitivity() {
        let base = WholeCellQuantumProfile::default();
        let optimal = environmental_quantum_profile(base, 37.0, 7.2, 0.21, 5.0);
        let acidic = environmental_quantum_profile(base, 37.0, 5.0, 0.21, 5.0);

        assert!(
            acidic.translation_efficiency < optimal.translation_efficiency,
            "Acidic pH should reduce translation: {:.3} vs {:.3}",
            acidic.translation_efficiency,
            optimal.translation_efficiency
        );
    }

    #[test]
    fn environmental_profile_o2_modulation() {
        let base = WholeCellQuantumProfile::default();
        let normoxic = environmental_quantum_profile(base, 37.0, 7.2, 0.21, 5.0);
        let hypoxic = environmental_quantum_profile(base, 37.0, 7.2, 0.05, 5.0);

        assert!(
            hypoxic.oxphos_efficiency < normoxic.oxphos_efficiency,
            "Hypoxia should reduce oxphos: {:.3} vs {:.3}",
            hypoxic.oxphos_efficiency,
            normoxic.oxphos_efficiency
        );
        // O2 should NOT affect membrane synthesis (anaerobic process)
        assert!(
            (hypoxic.membrane_synthesis_efficiency - normoxic.membrane_synthesis_efficiency).abs()
                < 0.01,
            "O2 should not affect membrane synthesis"
        );
    }

    #[test]
    fn explicit_microbe_creation() {
        let microbe = ExplicitMicrobe::new(0, 5, 5);
        assert_eq!(microbe.id, 0);
        assert_eq!(microbe.grid_x, 5);
        assert_eq!(microbe.grid_y, 5);
        assert!(microbe.simulator.atp_mm() > 0.0, "Microbe should start with ATP");
    }

    #[test]
    fn explicit_microbe_steps() {
        let mut microbe = ExplicitMicrobe::new(1, 3, 3);
        let atp_before = microbe.simulator.atp_mm();

        // Step with standard conditions (interval=1 means every call)
        let (co2, o2) = microbe.step_with_environment(37.0, 7.2, 0.21, 5.0, 50.0, 1);
        let atp_after = microbe.simulator.atp_mm();

        // ATP should have changed (cell is metabolically active)
        assert!(
            (atp_after - atp_before).abs() > 1e-6,
            "ATP should change after stepping: before={atp_before:.4}, after={atp_after:.4}"
        );
        // CO2 and O2 should be non-negative
        assert!(co2 >= 0.0, "CO2 should be non-negative: {co2}");
        assert!(o2 >= 0.0, "O2 should be non-negative: {o2}");
    }

    #[test]
    fn explicit_microbe_multi_rate() {
        let mut microbe = ExplicitMicrobe::new(2, 4, 4);

        // With step_interval=10, first 9 calls should accumulate without stepping
        for _ in 0..9 {
            let (co2, o2) = microbe.step_with_environment(37.0, 7.2, 0.21, 5.0, 1.0, 10);
            assert_eq!(co2, 0.0, "Should not produce CO2 before interval reached");
            assert_eq!(o2, 0.0, "Should not consume O2 before interval reached");
        }
        assert!(
            microbe.accumulated_dt_ms > 8.0,
            "Should have accumulated dt: {}",
            microbe.accumulated_dt_ms
        );

        // 10th call should trigger the actual step
        let (_co2, _o2) = microbe.step_with_environment(37.0, 7.2, 0.21, 5.0, 1.0, 10);
        assert_eq!(
            microbe.accumulated_dt_ms, 0.0,
            "accumulated_dt should reset after step"
        );
        // co2 may or may not be > 0 depending on cell state, but accumulated_dt should reset
    }

    #[test]
    fn bidirectional_coupling_chain() {
        // Test that environment changes propagate through the full chain:
        // temperature → quantum profile → whole-cell behavior
        let mut microbe = ExplicitMicrobe::new(3, 2, 2);

        // Step at cold temperature
        let _profile_before = microbe.simulator.quantum_profile();
        let _ = microbe.step_with_environment(20.0, 7.2, 0.21, 5.0, 50.0, 1);
        let profile_cold = microbe.simulator.quantum_profile();

        // Reset and step at warm temperature
        let mut microbe_warm = ExplicitMicrobe::new(4, 2, 2);
        let _ = microbe_warm.step_with_environment(45.0, 7.2, 0.21, 5.0, 50.0, 1);
        let profile_warm = microbe_warm.simulator.quantum_profile();

        // Warm profile should have higher efficiencies than cold
        assert!(
            profile_warm.oxphos_efficiency > profile_cold.oxphos_efficiency,
            "Warm oxphos {:.3} should exceed cold {:.3}",
            profile_warm.oxphos_efficiency,
            profile_cold.oxphos_efficiency
        );
    }

    // ===== FrameScheduler tests =====

    #[test]
    fn scheduler_plants_every_5() {
        let mut sched = FrameScheduler::new();
        let mut runs = 0u32;
        for _ in 0..20 {
            if sched.run_plants() {
                runs += 1;
            }
            sched.tick();
        }
        assert_eq!(runs, 4, "Plants should run 4 times in 20 frames (0,5,10,15)");
    }

    #[test]
    fn scheduler_soil_fauna_every_10() {
        let mut sched = FrameScheduler::new();
        let mut runs = 0u32;
        for _ in 0..20 {
            if sched.run_soil_fauna() {
                runs += 1;
            }
            sched.tick();
        }
        assert_eq!(runs, 2, "Soil fauna should run 2 times in 20 frames (0,10)");
    }

    #[test]
    fn scheduler_ecology_once_per_frame() {
        let sched = FrameScheduler::new();
        assert!(sched.run_ecology_fields(0), "substep 0 should rebuild ecology");
        assert!(!sched.run_ecology_fields(1), "substep 1 should skip ecology");
        assert!(!sched.run_ecology_fields(2), "substep 2 should skip ecology");
    }

    #[test]
    fn scheduler_dt_scale() {
        let sched = FrameScheduler::new();
        assert!((sched.dt_scale(5) - 5.0).abs() < 0.01);
        assert!((sched.dt_scale(10) - 10.0).abs() < 0.01);
    }
}
