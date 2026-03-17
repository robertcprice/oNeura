//! Pharmacology engine: drug PK/PD simulation for 7 drugs plus general
//! anesthesia.
//!
//! This module implements a 1-compartment pharmacokinetic model (Bateman
//! equation for absorption and elimination) coupled with pharmacodynamic
//! effects via Hill equation dose-response curves. Each drug modulates
//! specific ion channel conductance scales or neurotransmitter concentrations
//! on the `NeuronArrays`.
//!
//! ## Supported Drugs
//!
//! | Drug        | Class              | Primary Target                        |
//! |-------------|--------------------|---------------------------------------|
//! | Fluoxetine  | SSRI               | Serotonin reuptake block              |
//! | Diazepam    | Benzodiazepine     | GABA-A conductance x5.0               |
//! | Caffeine    | Xanthine           | Adenosine block, Na_v slight increase  |
//! | Amphetamine | Psychostimulant    | DA/NE release increase                |
//! | L-DOPA      | DA precursor       | DA synthesis increase                 |
//! | Donepezil   | AChE inhibitor     | ACh concentration increase            |
//! | Ketamine    | NMDA antagonist    | NMDA conductance x0.1                 |
//!
//! ## General Anesthesia
//!
//! A composite drug effect modeling clinical general anesthesia:
//! - GABA-A conductance x8.0 (enhanced inhibition)
//! - NMDA conductance x0.05 (blocked excitation)
//! - AMPA conductance x0.4 (reduced excitation)
//! - Na_v conductance x0.5 (reduced spike generation)
//! - K_leak conductance x2.0 (membrane hyperpolarization)
//!
//! ## Implementation Note
//!
//! Drug effects are applied by modifying `conductance_scale` on
//! `NeuronArrays`, NOT by changing channel `count`. This is critical:
//! `int(1 * 0.95) = 0` for any dose, making integer count modulation
//! useless. Float `conductance_scale` provides smooth dose-response.

#[cfg(target_os = "macos")]
use crate::gpu;
#[cfg(target_os = "macos")]
use crate::gpu::state::MetalNeuronState;
use crate::neuron_arrays::NeuronArrays;
use crate::types::*;

/// Pharmacokinetic/pharmacodynamic state for a single active drug.
#[derive(Debug, Clone)]
pub struct DrugState {
    /// Drug identity.
    pub drug_type: DrugType,
    /// Administered dose in mg.
    pub dose_mg: f32,
    /// Current plasma concentration (arbitrary units).
    pub plasma_conc: f32,
    /// Time since dose administration in ms.
    pub time_since_dose: f32,
    /// Absorption rate constant (per hour).
    pub ka: f32,
    /// Elimination rate constant (per hour).
    pub ke: f32,
    /// Effective concentration 50% (plasma units).
    pub ec50: f32,
    /// Hill coefficient for dose-response.
    pub hill_n: f32,
}

impl DrugState {
    /// Create a new drug state with pre-configured PK parameters.
    pub fn new(drug_type: DrugType, dose_mg: f32) -> Self {
        let (ka, ke, ec50, hill_n) = drug_pk_params(drug_type);
        Self {
            drug_type,
            dose_mg,
            plasma_conc: 0.0,
            time_since_dose: 0.0,
            ka,
            ke,
            ec50,
            hill_n,
        }
    }

    /// Current pharmacodynamic effect magnitude [0, 1] via Hill equation.
    pub fn effect(&self) -> f32 {
        if self.plasma_conc <= 0.0 {
            return 0.0;
        }
        let cn = self.plasma_conc.powf(self.hill_n);
        let en = self.ec50.powf(self.hill_n);
        cn / (en + cn)
    }

    /// Whether this drug has been effectively eliminated (negligible plasma).
    pub fn is_eliminated(&self) -> bool {
        self.plasma_conc < 0.001 && self.time_since_dose > 1_000.0
    }
}

/// Drug PK/PD engine managing multiple concurrent drugs.
pub struct PharmacologyEngine {
    /// Currently active drugs.
    pub active_drugs: Vec<DrugState>,
}

impl PharmacologyEngine {
    /// Create an empty pharmacology engine (no drugs active).
    pub fn new() -> Self {
        Self {
            active_drugs: Vec::new(),
        }
    }

    /// Administer a drug at a given dose.
    ///
    /// If the same drug is already active, replaces it (fresh PK start).
    pub fn apply_drug(&mut self, drug: DrugType, dose_mg: f32) {
        // Remove existing instance of the same drug
        self.active_drugs.retain(|d| d.drug_type != drug);
        self.active_drugs.push(DrugState::new(drug, dose_mg));
    }

    /// Remove a specific drug (immediate clearance).
    pub fn remove_drug(&mut self, drug: DrugType) {
        self.active_drugs.retain(|d| d.drug_type != drug);
    }

    /// Advance PK for all active drugs and apply PD effects to neurons.
    ///
    /// This function:
    /// 1. Steps the Bateman PK model for each drug.
    /// 2. Computes the Hill-equation PD effect.
    /// 3. Applies conductance_scale modulations and NT changes.
    /// 4. Removes eliminated drugs.
    ///
    /// # Arguments
    /// * `neurons` - Mutable neuron arrays. Writes `conductance_scale`, `nt_conc`,
    ///   `external_current`.
    /// * `dt` - Time step in ms.
    pub fn step(&mut self, neurons: &mut NeuronArrays, dt: f32) {
        for drug in &mut self.active_drugs {
            drug.time_since_dose += dt;

            // Bateman equation: C(t) = F*D*ka / (Vd*(ka-ke)) * (e^(-ke*t) - e^(-ka*t))
            // Rate constants are in per-ms (simulation time units).
            // No time conversion needed.
            let t = drug.time_since_dose;
            if (drug.ka - drug.ke).abs() > 1e-10 {
                let f_d_ka = drug.dose_mg * drug.ka;
                let vd = 70.0; // volume of distribution (L)
                let v_ka_ke = vd * (drug.ka - drug.ke);
                drug.plasma_conc =
                    (f_d_ka / v_ka_ke) * ((-drug.ke * t).exp() - (-drug.ka * t).exp());
                drug.plasma_conc = drug.plasma_conc.max(0.0);
            } else {
                // Degenerate case (ka == ke): use limit form
                drug.plasma_conc = drug.dose_mg / 70.0 * drug.ka * t * (-drug.ke * t).exp();
                drug.plasma_conc = drug.plasma_conc.max(0.0);
            }

            // Apply PD effects
            let effect = drug.effect();
            apply_drug_effects(drug.drug_type, effect, neurons);
        }

        // Remove eliminated drugs
        self.active_drugs.retain(|d| !d.is_eliminated());
    }

    #[cfg(target_os = "macos")]
    pub fn step_gpu(&mut self, gpu: &gpu::GpuContext, neurons: &MetalNeuronState, dt: f32) {
        for drug in &mut self.active_drugs {
            drug.time_since_dose += dt;

            let t = drug.time_since_dose;
            if (drug.ka - drug.ke).abs() > 1e-10 {
                let f_d_ka = drug.dose_mg * drug.ka;
                let vd = 70.0;
                let v_ka_ke = vd * (drug.ka - drug.ke);
                drug.plasma_conc =
                    (f_d_ka / v_ka_ke) * ((-drug.ke * t).exp() - (-drug.ka * t).exp());
                drug.plasma_conc = drug.plasma_conc.max(0.0);
            } else {
                drug.plasma_conc = drug.dose_mg / 70.0 * drug.ka * t * (-drug.ke * t).exp();
                drug.plasma_conc = drug.plasma_conc.max(0.0);
            }

            let effect = drug.effect();
            gpu::pharmacology::dispatch_pharmacology(gpu, neurons, drug.drug_type as u8, effect);
        }

        self.active_drugs.retain(|d| !d.is_eliminated());
    }

    /// Apply general anesthesia (immediate, not PK-dependent).
    ///
    /// Effects (from Python oNeura implementation):
    /// - GABA-A x8.0 (enhanced inhibition)
    /// - NMDA x0.05 (near-complete block)
    /// - AMPA x0.4 (reduced fast excitation)
    /// - Na_v x0.5 (reduced AP generation)
    /// - K_leak x2.0 (membrane hyperpolarization)
    pub fn apply_anesthesia(&mut self, neurons: &mut NeuronArrays) {
        for i in 0..neurons.count {
            if neurons.alive[i] == 0 {
                continue;
            }
            let cs = &mut neurons.conductance_scale[i];
            cs[IonChannelType::GabaA.index()] = 8.0;
            cs[IonChannelType::NMDA.index()] = 0.05;
            cs[IonChannelType::AMPA.index()] = 0.4;
            cs[IonChannelType::Nav.index()] = 0.5;
            cs[IonChannelType::Kleak.index()] = 2.0;
        }
    }

    /// Remove anesthesia: reset all conductance scales to 1.0.
    pub fn remove_anesthesia(&self, neurons: &mut NeuronArrays) {
        for i in 0..neurons.count {
            if neurons.alive[i] == 0 {
                continue;
            }
            neurons.conductance_scale[i] = [1.0; IonChannelType::COUNT];
        }
    }

    /// Get the number of currently active drugs.
    pub fn n_active_drugs(&self) -> usize {
        self.active_drugs.len()
    }

    /// Check if a specific drug is currently active.
    pub fn is_active(&self, drug: DrugType) -> bool {
        self.active_drugs.iter().any(|d| d.drug_type == drug)
    }

    /// Get the current PD effect level [0, 1] for a drug, or 0 if not active.
    pub fn drug_effect_level(&self, drug: DrugType) -> f32 {
        self.active_drugs
            .iter()
            .find(|d| d.drug_type == drug)
            .map(|d| d.effect())
            .unwrap_or(0.0)
    }
}

impl Default for PharmacologyEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// PK parameters (ka, ke, ec50, hill_n) for each drug type.
///
/// Rate constants in **per-millisecond** (simulation time units).
/// These are compressed from clinical per-hour values to produce
/// meaningful drug effects within typical simulation durations
/// (1,000 - 20,000 ms). EC50 in arbitrary plasma concentration units.
///
/// Rough mapping: clinical ka in per-hour maps to sim ka that produces
/// peak plasma at ~500 - 5,000 ms depending on drug speed.
fn drug_pk_params(drug: DrugType) -> (f32, f32, f32, f32) {
    match drug {
        //                            ka       ke        ec50  hill_n
        DrugType::Fluoxetine => (0.0005, 0.00003, 0.02, 1.0), // slow onset, very slow offset
        DrugType::Diazepam => (0.002, 0.0001, 0.03, 1.5),     // fast onset
        DrugType::Caffeine => (0.003, 0.0003, 0.05, 1.0),     // fast both ways
        DrugType::Amphetamine => (0.002, 0.0001, 0.01, 1.2),  // moderate
        DrugType::LDOPA => (0.003, 0.0005, 0.03, 1.0),        // fast elimination
        DrugType::Donepezil => (0.0003, 0.00005, 0.005, 1.0), // slow, long t1/2
        DrugType::Ketamine => (0.005, 0.0005, 0.015, 1.5),    // fast onset/offset
    }
}

/// Apply pharmacodynamic effects for a drug at given effect level [0, 1].
///
/// Each drug has specific molecular targets that map to either conductance
/// scale changes (ionotropic targets) or NT concentration changes
/// (transporter/enzyme targets).
fn apply_drug_effects(drug: DrugType, effect: f32, neurons: &mut NeuronArrays) {
    if effect < 0.001 {
        return;
    }

    for i in 0..neurons.count {
        if neurons.alive[i] == 0 {
            continue;
        }

        match drug {
            DrugType::Fluoxetine => {
                // SSRI: block serotonin reuptake -> increase 5-HT concentration.
                // Set to absolute value (not multiplicative) to avoid runaway.
                let boost = 1.0 + effect * 3.0; // up to 4x baseline
                neurons.nt_conc[i][NTType::Serotonin.index()] =
                    NTType::Serotonin.resting_conc_nm() * boost;
            }

            DrugType::Diazepam => {
                // Benzodiazepine: positive allosteric modulator of GABA-A.
                // Max effect: GABA-A conductance x5.0.
                let scale = 1.0 + effect * 4.0;
                neurons.conductance_scale[i][IonChannelType::GabaA.index()] = scale;
            }

            DrugType::Caffeine => {
                // Adenosine antagonist: blocks inhibitory adenosine receptors
                // -> increased excitability. Slight Na_v upregulation.
                let nav_boost = 1.0 + effect * 0.3;
                neurons.conductance_scale[i][IonChannelType::Nav.index()] = nav_boost;
                // Direct excitatory current (modeling reduced adenosine inhibition)
                neurons.external_current[i] += effect * 2.0;
            }

            DrugType::Amphetamine => {
                // DA/NE releaser: reverse DAT and NET transporters.
                // Set to absolute values to avoid multiplicative runaway.
                let da_boost = 1.0 + effect * 5.0;
                let ne_boost = 1.0 + effect * 3.0;
                neurons.nt_conc[i][NTType::Dopamine.index()] =
                    NTType::Dopamine.resting_conc_nm() * da_boost;
                neurons.nt_conc[i][NTType::Norepinephrine.index()] =
                    NTType::Norepinephrine.resting_conc_nm() * ne_boost;
            }

            DrugType::LDOPA => {
                // DA precursor: increases DA synthesis via DOPA decarboxylase.
                let da_boost = 1.0 + effect * 4.0;
                neurons.nt_conc[i][NTType::Dopamine.index()] =
                    NTType::Dopamine.resting_conc_nm() * da_boost;
            }

            DrugType::Donepezil => {
                // AChE inhibitor: blocks ACh breakdown -> increase ACh.
                let ach_boost = 1.0 + effect * 3.0;
                neurons.nt_conc[i][NTType::Acetylcholine.index()] =
                    NTType::Acetylcholine.resting_conc_nm() * ach_boost;
            }

            DrugType::Ketamine => {
                // NMDA antagonist: block NMDA channels.
                // Max effect: NMDA conductance x0.1 (90% block).
                let nmda_scale = (1.0 - effect * 0.9).max(0.05);
                neurons.conductance_scale[i][IonChannelType::NMDA.index()] = nmda_scale;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drug_state_creation() {
        let drug = DrugState::new(DrugType::Caffeine, 100.0);
        assert!((drug.dose_mg - 100.0).abs() < 1e-6);
        assert!((drug.plasma_conc - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_bateman_pk_rises_then_falls() {
        let mut engine = PharmacologyEngine::new();
        let mut neurons = NeuronArrays::new(1);
        engine.apply_drug(DrugType::Caffeine, 200.0);

        let mut concentrations = Vec::new();
        // Need enough sim time to see full Bateman curve rise and fall.
        // Caffeine peak time: ln(ka/ke)/(ka-ke) = ln(20)/2.85 ≈ 1.05 hours
        // At 1ms/step, 60000 ms = 1 bio hour, so need ~120000 steps.
        for _ in 0..200_000 {
            engine.step(&mut neurons, 1.0);
            if !engine.active_drugs.is_empty() {
                concentrations.push(engine.active_drugs[0].plasma_conc);
            }
        }

        // Should rise then fall (Bateman curve)
        let peak_idx = concentrations
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        assert!(peak_idx > 0, "Peak should not be at time 0");
        assert!(
            peak_idx < concentrations.len() - 100,
            "Peak should not be at the end: peak at {}",
            peak_idx
        );
    }

    #[test]
    fn test_diazepam_increases_gabaa() {
        let mut engine = PharmacologyEngine::new();
        let mut neurons = NeuronArrays::new(5);

        engine.apply_drug(DrugType::Diazepam, 10.0);

        for _ in 0..5000 {
            engine.step(&mut neurons, 1.0);
        }

        let gabaa_idx = IonChannelType::GabaA.index();
        assert!(
            neurons.conductance_scale[0][gabaa_idx] > 1.0,
            "Diazepam should increase GABA-A conductance: {}",
            neurons.conductance_scale[0][gabaa_idx]
        );
    }

    #[test]
    fn test_ketamine_decreases_nmda() {
        let mut engine = PharmacologyEngine::new();
        let mut neurons = NeuronArrays::new(5);

        engine.apply_drug(DrugType::Ketamine, 50.0);

        for _ in 0..3000 {
            engine.step(&mut neurons, 1.0);
        }

        let nmda_idx = IonChannelType::NMDA.index();
        assert!(
            neurons.conductance_scale[0][nmda_idx] < 1.0,
            "Ketamine should decrease NMDA conductance: {}",
            neurons.conductance_scale[0][nmda_idx]
        );
    }

    #[test]
    fn test_anesthesia_effects() {
        let mut engine = PharmacologyEngine::new();
        let mut neurons = NeuronArrays::new(3);

        engine.apply_anesthesia(&mut neurons);

        let cs = &neurons.conductance_scale[0];
        assert!((cs[IonChannelType::GabaA.index()] - 8.0).abs() < 1e-6);
        assert!((cs[IonChannelType::NMDA.index()] - 0.05).abs() < 1e-6);
        assert!((cs[IonChannelType::AMPA.index()] - 0.4).abs() < 1e-6);
        assert!((cs[IonChannelType::Nav.index()] - 0.5).abs() < 1e-6);
        assert!((cs[IonChannelType::Kleak.index()] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_remove_anesthesia() {
        let mut engine = PharmacologyEngine::new();
        let mut neurons = NeuronArrays::new(3);

        engine.apply_anesthesia(&mut neurons);
        engine.remove_anesthesia(&mut neurons);

        for ch in 0..IonChannelType::COUNT {
            assert!(
                (neurons.conductance_scale[0][ch] - 1.0).abs() < 1e-6,
                "Channel {} should be reset to 1.0: {}",
                ch,
                neurons.conductance_scale[0][ch]
            );
        }
    }

    #[test]
    fn test_fluoxetine_increases_serotonin() {
        let mut engine = PharmacologyEngine::new();
        let mut neurons = NeuronArrays::new(3);

        engine.apply_drug(DrugType::Fluoxetine, 20.0);

        for _ in 0..10000 {
            engine.step(&mut neurons, 1.0);
        }

        let ht_idx = NTType::Serotonin.index();
        assert!(
            neurons.nt_conc[0][ht_idx] > NTType::Serotonin.resting_conc_nm(),
            "Fluoxetine should increase serotonin: {}",
            neurons.nt_conc[0][ht_idx]
        );
    }

    #[test]
    fn test_caffeine_effects() {
        let mut engine = PharmacologyEngine::new();
        let mut neurons = NeuronArrays::new(3);

        engine.apply_drug(DrugType::Caffeine, 100.0);

        for _ in 0..3000 {
            engine.step(&mut neurons, 1.0);
        }

        let nav_idx = IonChannelType::Nav.index();
        assert!(
            neurons.conductance_scale[0][nav_idx] >= 1.0,
            "Caffeine should increase Nav: {}",
            neurons.conductance_scale[0][nav_idx]
        );
    }

    #[test]
    fn test_amphetamine_increases_da_ne() {
        let mut engine = PharmacologyEngine::new();
        let mut neurons = NeuronArrays::new(3);

        engine.apply_drug(DrugType::Amphetamine, 10.0);

        for _ in 0..5000 {
            engine.step(&mut neurons, 1.0);
        }

        let da_idx = NTType::Dopamine.index();
        let ne_idx = NTType::Norepinephrine.index();
        assert!(
            neurons.nt_conc[0][da_idx] > NTType::Dopamine.resting_conc_nm(),
            "Amphetamine should increase DA: {}",
            neurons.nt_conc[0][da_idx]
        );
        assert!(
            neurons.nt_conc[0][ne_idx] > NTType::Norepinephrine.resting_conc_nm(),
            "Amphetamine should increase NE: {}",
            neurons.nt_conc[0][ne_idx]
        );
    }

    #[test]
    fn test_multiple_drugs() {
        let mut engine = PharmacologyEngine::new();
        let mut neurons = NeuronArrays::new(3);

        engine.apply_drug(DrugType::Caffeine, 100.0);
        engine.apply_drug(DrugType::Diazepam, 5.0);

        assert_eq!(engine.n_active_drugs(), 2);
        assert!(engine.is_active(DrugType::Caffeine));
        assert!(engine.is_active(DrugType::Diazepam));

        for _ in 0..3000 {
            engine.step(&mut neurons, 1.0);
        }

        let gabaa_idx = IonChannelType::GabaA.index();
        assert!(
            neurons.conductance_scale[0][gabaa_idx] > 1.0,
            "Diazepam effect should be active"
        );
    }

    #[test]
    fn test_remove_drug() {
        let mut engine = PharmacologyEngine::new();
        engine.apply_drug(DrugType::Caffeine, 100.0);
        assert!(engine.is_active(DrugType::Caffeine));

        engine.remove_drug(DrugType::Caffeine);
        assert!(!engine.is_active(DrugType::Caffeine));
        assert_eq!(engine.n_active_drugs(), 0);
    }

    #[test]
    fn test_donepezil_increases_ach() {
        let mut engine = PharmacologyEngine::new();
        let mut neurons = NeuronArrays::new(2);

        engine.apply_drug(DrugType::Donepezil, 10.0);

        for _ in 0..10000 {
            engine.step(&mut neurons, 1.0);
        }

        let ach_idx = NTType::Acetylcholine.index();
        assert!(
            neurons.nt_conc[0][ach_idx] > NTType::Acetylcholine.resting_conc_nm(),
            "Donepezil should increase ACh: {}",
            neurons.nt_conc[0][ach_idx]
        );
    }

    #[test]
    fn test_ldopa_increases_da() {
        let mut engine = PharmacologyEngine::new();
        let mut neurons = NeuronArrays::new(2);

        engine.apply_drug(DrugType::LDOPA, 250.0);

        for _ in 0..5000 {
            engine.step(&mut neurons, 1.0);
        }

        let da_idx = NTType::Dopamine.index();
        assert!(
            neurons.nt_conc[0][da_idx] > NTType::Dopamine.resting_conc_nm(),
            "L-DOPA should increase DA: {}",
            neurons.nt_conc[0][da_idx]
        );
    }

    #[test]
    fn test_dead_neurons_skipped() {
        let mut engine = PharmacologyEngine::new();
        let mut neurons = NeuronArrays::new(3);
        neurons.alive[0] = 0;

        engine.apply_anesthesia(&mut neurons);

        // Dead neuron should not be modified
        for ch in 0..IonChannelType::COUNT {
            assert!(
                (neurons.conductance_scale[0][ch] - 1.0).abs() < 1e-6,
                "Dead neuron channel {} should remain at 1.0: {}",
                ch,
                neurons.conductance_scale[0][ch]
            );
        }
        // Alive neuron should be modified
        assert!(
            (neurons.conductance_scale[1][IonChannelType::GabaA.index()] - 8.0).abs() < 1e-6,
            "Alive neuron should have anesthesia"
        );
    }

    #[test]
    fn test_effect_bounded() {
        let drug = DrugState::new(DrugType::Caffeine, 100.0);
        let e = drug.effect();
        assert!(e >= 0.0 && e <= 1.0, "Effect should be in [0,1]: {}", e);
    }

    #[test]
    fn test_drug_replace_on_readminister() {
        let mut engine = PharmacologyEngine::new();
        engine.apply_drug(DrugType::Caffeine, 100.0);
        engine.apply_drug(DrugType::Caffeine, 200.0); // should replace

        assert_eq!(engine.n_active_drugs(), 1);
        assert!((engine.active_drugs[0].dose_mg - 200.0).abs() < 1e-6);
    }
}
