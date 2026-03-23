//! Glial cell simulation: astrocytes, oligodendrocytes, and microglia.
//!
//! Glia outnumber neurons ~3:1 in the human brain and play critical roles
//! in neural function. This module models three glial types with per-neuron
//! state vectors (one glial "unit" per neuron, approximating the tight
//! neuron-glia coupling in real tissue).
//!
//! **Interval-gated**: called every `GLIA_INTERVAL` steps (10), since
//! glial dynamics are slower than neural membrane dynamics.
//!
//! ## Astrocytes
//! - **Glutamate uptake**: Clear excess glutamate from the extracellular
//!   space via EAAT transporters. Prevents excitotoxicity and shapes
//!   synaptic transmission timing.
//! - **Lactate shuttle**: When nearby neurons are metabolically stressed
//!   (low ATP), astrocytes convert glycogen to lactate and shuttle it to
//!   neurons as an alternative energy substrate.
//! - **Uptake fatigue**: Under excitotoxic conditions (very high glutamate),
//!   EAAT transporter capacity decreases.
//!
//! ## Oligodendrocytes
//! - **Myelin maintenance**: Track the myelination state of axons.
//!   Myelin integrity recovers slowly and degrades under metabolic stress.
//!
//! ## Microglia
//! - **Damage-responsive pruning**: Microglia are the brain's immune cells.
//!   They become activated by neural damage signals (ATP depletion, voltage
//!   stress) and prune weak synapses. Activation requires damage_signal > 50.0.

use crate::neuron_arrays::NeuronArrays;
use crate::synapse_arrays::SynapseArrays;
use crate::types::NTType;

// ===== Astrocyte Constants =====

/// Baseline EAAT glutamate uptake rate.
const ASTROCYTE_UPTAKE_RATE: f32 = 0.1;

/// Astrocyte lactate pool resting level (uM).
const ASTROCYTE_LACTATE_RESTING: f32 = 2000.0;

/// Maximum astrocyte lactate pool (uM).
const ASTROCYTE_LACTATE_MAX: f32 = 5000.0;

/// Lactate supply rate to neurons (fraction of pool per second).
const LACTATE_SUPPLY_RATE: f32 = 0.01;

/// Lactate-to-glucose conversion factor.
const LACTATE_TO_GLUCOSE: f32 = 0.3;

/// Lactate replenishment rate (fraction per second toward resting).
const LACTATE_REPLENISH_RATE: f32 = 0.005;

/// Uptake fatigue threshold (glutamate concentration in nM).
const UPTAKE_FATIGUE_THRESHOLD: f32 = 2000.0;

/// Uptake fatigue factor (multiplicative per interval).
const UPTAKE_FATIGUE_FACTOR: f32 = 0.99;

/// Uptake recovery rate (fraction per second toward 1.0).
const UPTAKE_RECOVERY_RATE: f32 = 0.001;

// ===== Oligodendrocyte Constants =====

/// Myelin maintenance recovery rate (per second).
const MYELIN_RECOVERY_RATE: f32 = 0.0001;

// ===== Microglia Constants =====

/// Damage signal threshold for microglia activation.
const MICROGLIA_DAMAGE_THRESHOLD: f32 = 50.0;

/// Microglia activation growth rate (per second).
const MICROGLIA_ACTIVATION_RATE: f32 = 0.01;

/// Microglia deactivation rate (per second).
const MICROGLIA_DEACTIVATION_RATE: f32 = 0.005;

/// Microglia activation threshold for synaptic pruning.
const MICROGLIA_PRUNING_ACTIVATION: f32 = 0.5;

/// Damage signal decay factor per step.
const DAMAGE_DECAY_FACTOR: f32 = 0.99;

/// ATP threshold for metabolic stress signal.
const ATP_STRESS_THRESHOLD: f32 = 2000.0;

/// Per-neuron glial state.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct GliaState {
    /// Astrocyte EAAT glutamate uptake capacity [0, 1].
    /// Decreases under excitotoxic conditions (uptake fatigue).
    pub astrocyte_uptake: Vec<f32>,

    /// Astrocyte lactate pool (uM).
    /// Astrocytes store glycogen and convert it to lactate for
    /// metabolically stressed neurons.
    pub astrocyte_lactate: Vec<f32>,

    /// Oligodendrocyte myelin integrity [0, 1].
    /// 1.0 = fully myelinated, 0.0 = demyelinated.
    pub myelin_integrity: Vec<f32>,

    /// Microglia activation level [0, 1].
    /// 0 = surveilling (resting), 1 = fully activated (phagocytic).
    pub microglia_activation: Vec<f32>,

    /// Accumulated damage signal per neuron.
    /// Drives microglia activation. Built from ATP depletion and
    /// voltage stress indicators.
    pub damage_signal: Vec<f32>,
}

impl GliaState {
    /// Create glial state for `n_neurons` neurons at resting baseline.
    pub fn new(n_neurons: usize) -> Self {
        Self {
            astrocyte_uptake: vec![1.0; n_neurons],
            astrocyte_lactate: vec![ASTROCYTE_LACTATE_RESTING; n_neurons],
            myelin_integrity: vec![1.0; n_neurons],
            microglia_activation: vec![0.0; n_neurons],
            damage_signal: vec![0.0; n_neurons],
        }
    }

    /// Expand arrays for additional neurons (e.g., after `NeuronArrays::add_neuron`).
    pub fn resize(&mut self, n_neurons: usize) {
        self.astrocyte_uptake.resize(n_neurons, 1.0);
        self.astrocyte_lactate
            .resize(n_neurons, ASTROCYTE_LACTATE_RESTING);
        self.myelin_integrity.resize(n_neurons, 1.0);
        self.microglia_activation.resize(n_neurons, 0.0);
        self.damage_signal.resize(n_neurons, 0.0);
    }

    /// Update all glial subsystems.
    ///
    /// # Arguments
    /// * `neurons` - Mutable neuron arrays. Astrocytes modify `nt_conc` (glutamate)
    ///   and `glucose`. Damage signals read `atp` and `voltage`.
    /// * `synapses` - Mutable synapse arrays. Microglia prune weak synapses.
    /// * `dt` - **Effective** time step in ms (typically `dt * GLIA_INTERVAL`).
    pub fn update(&mut self, neurons: &mut NeuronArrays, synapses: &mut SynapseArrays, dt: f32) {
        if self.astrocyte_uptake.len() < neurons.count {
            self.resize(neurons.count);
        }

        let dt_s = dt / 1000.0;

        for i in 0..neurons.count {
            if neurons.alive[i] == 0 {
                continue;
            }

            // ===== Astrocyte: Glutamate Uptake =====
            let glu_idx = NTType::Glutamate.index();
            let glu = neurons.nt_conc[i][glu_idx];
            let uptake_rate = self.astrocyte_uptake[i] * ASTROCYTE_UPTAKE_RATE;
            let glu_removed = glu * uptake_rate * dt;
            neurons.nt_conc[i][glu_idx] =
                (glu - glu_removed).max(NTType::Glutamate.resting_conc_nm());

            // ===== Astrocyte: Lactate Shuttle =====
            let lactate = &mut self.astrocyte_lactate[i];
            // Glutamate uptake generates some lactate (glutamate-glutamine cycle)
            *lactate += glu_removed * 0.5;
            // Supply lactate to neuron (converted to glucose equivalent)
            let supply = (*lactate * LACTATE_SUPPLY_RATE * dt_s).min(*lactate);
            *lactate -= supply;
            neurons.glucose[i] += supply * LACTATE_TO_GLUCOSE;
            // Replenish lactate stores from blood supply
            *lactate += (ASTROCYTE_LACTATE_RESTING - *lactate) * LACTATE_REPLENISH_RATE * dt_s;
            *lactate = lactate.clamp(0.0, ASTROCYTE_LACTATE_MAX);

            // ===== Astrocyte: Uptake Fatigue =====
            // Recovery toward baseline
            self.astrocyte_uptake[i] +=
                (1.0 - self.astrocyte_uptake[i]) * UPTAKE_RECOVERY_RATE * dt_s;
            // Fatigue under excitotoxic conditions
            if glu > UPTAKE_FATIGUE_THRESHOLD {
                self.astrocyte_uptake[i] *= UPTAKE_FATIGUE_FACTOR;
            }
            self.astrocyte_uptake[i] = self.astrocyte_uptake[i].clamp(0.1, 1.0);

            // ===== Oligodendrocyte: Myelin Maintenance =====
            let myelin = &mut self.myelin_integrity[i];
            *myelin += MYELIN_RECOVERY_RATE * dt_s;
            *myelin = myelin.clamp(0.0, 1.0);

            // ===== Microglia: Damage Detection =====
            let atp_stress = if neurons.atp[i] < ATP_STRESS_THRESHOLD {
                1.0
            } else {
                0.0
            };
            let v_stress = if neurons.voltage[i] > 0.0 || neurons.voltage[i] < -90.0 {
                1.0
            } else {
                0.0
            };
            self.damage_signal[i] += (atp_stress + v_stress) * dt_s;
            self.damage_signal[i] *= DAMAGE_DECAY_FACTOR;

            // ===== Microglia: Activation Dynamics =====
            let activation = &mut self.microglia_activation[i];
            if self.damage_signal[i] > MICROGLIA_DAMAGE_THRESHOLD {
                *activation += MICROGLIA_ACTIVATION_RATE * dt_s;
            } else {
                *activation -= MICROGLIA_DEACTIVATION_RATE * dt_s;
            }
            *activation = activation.clamp(0.0, 1.0);
        }

        // ===== Microglia: Synaptic Pruning =====
        // Activated microglia prune already-weak synapses (complement-mediated)
        for idx in 0..synapses.n_synapses {
            if !synapses.should_prune(idx) {
                continue;
            }
            let post = synapses.col_indices[idx] as usize;
            if post < self.microglia_activation.len()
                && self.microglia_activation[post] > MICROGLIA_PRUNING_ACTIVATION
            {
                synapses.weight[idx] = 0.0;
                synapses.strength[idx] = 0.0;
            }
        }
    }

    /// Get the number of neurons with activated microglia.
    pub fn n_activated_microglia(&self) -> usize {
        self.microglia_activation
            .iter()
            .filter(|&&a| a > MICROGLIA_PRUNING_ACTIVATION)
            .count()
    }

    /// Average myelination level across all neurons.
    pub fn mean_myelination(&self) -> f32 {
        if self.myelin_integrity.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.myelin_integrity.iter().sum();
        sum / self.myelin_integrity.len() as f32
    }

    /// Average astrocyte glutamate uptake capacity.
    pub fn mean_uptake_capacity(&self) -> f32 {
        if self.astrocyte_uptake.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.astrocyte_uptake.iter().sum();
        sum / self.astrocyte_uptake.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glia_state_creation() {
        let glia = GliaState::new(100);
        assert_eq!(glia.astrocyte_uptake.len(), 100);
        assert_eq!(glia.astrocyte_lactate.len(), 100);
        assert_eq!(glia.myelin_integrity.len(), 100);
        assert_eq!(glia.microglia_activation.len(), 100);
        assert_eq!(glia.damage_signal.len(), 100);
    }

    #[test]
    fn test_astrocyte_glutamate_uptake() {
        let mut glia = GliaState::new(3);
        let mut neurons = NeuronArrays::new(3);
        let mut synapses = SynapseArrays::new(3);

        // Set high glutamate on neuron 0
        let glu_idx = NTType::Glutamate.index();
        neurons.nt_conc[0][glu_idx] = 5000.0;
        let initial_glu = neurons.nt_conc[0][glu_idx];

        glia.update(&mut neurons, &mut synapses, 10.0);

        assert!(
            neurons.nt_conc[0][glu_idx] < initial_glu,
            "Astrocyte should reduce excess glutamate: {} -> {}",
            initial_glu,
            neurons.nt_conc[0][glu_idx]
        );
    }

    #[test]
    fn test_glutamate_floor_at_resting() {
        let mut glia = GliaState::new(1);
        let mut neurons = NeuronArrays::new(1);
        let mut synapses = SynapseArrays::new(1);

        // Set glutamate near resting
        let glu_idx = NTType::Glutamate.index();
        neurons.nt_conc[0][glu_idx] = NTType::Glutamate.resting_conc_nm() + 1.0;

        // Run many times
        for _ in 0..100 {
            glia.update(&mut neurons, &mut synapses, 10.0);
        }

        assert!(
            neurons.nt_conc[0][glu_idx] >= NTType::Glutamate.resting_conc_nm(),
            "Glutamate should not drop below resting: {}",
            neurons.nt_conc[0][glu_idx]
        );
    }

    #[test]
    fn test_lactate_shuttle() {
        let mut glia = GliaState::new(2);
        let mut neurons = NeuronArrays::new(2);
        let mut synapses = SynapseArrays::new(2);

        let _initial_glucose = neurons.glucose[0];

        // Run metabolism to consume some glucose, then let astrocytes help
        neurons.glucose[0] = 100.0; // low glucose
        glia.update(&mut neurons, &mut synapses, 10.0);

        assert!(
            neurons.glucose[0] > 100.0,
            "Lactate shuttle should increase glucose: {}",
            neurons.glucose[0]
        );
    }

    #[test]
    fn test_myelin_integrity_recovers() {
        let mut glia = GliaState::new(1);
        let mut neurons = NeuronArrays::new(1);
        let mut synapses = SynapseArrays::new(1);

        glia.myelin_integrity[0] = 0.5; // damaged myelin

        for _ in 0..1000 {
            glia.update(&mut neurons, &mut synapses, 10.0);
        }

        assert!(
            glia.myelin_integrity[0] > 0.5,
            "Myelin should recover: {}",
            glia.myelin_integrity[0]
        );
    }

    #[test]
    fn test_microglia_resting_without_damage() {
        let mut glia = GliaState::new(1);
        let mut neurons = NeuronArrays::new(1);
        let mut synapses = SynapseArrays::new(1);

        glia.update(&mut neurons, &mut synapses, 10.0);

        assert!(
            glia.microglia_activation[0] < 0.01,
            "Microglia should be resting without damage: {}",
            glia.microglia_activation[0]
        );
    }

    #[test]
    fn test_resize() {
        let mut glia = GliaState::new(5);
        glia.resize(10);
        assert_eq!(glia.astrocyte_uptake.len(), 10);
        assert_eq!(glia.myelin_integrity.len(), 10);
        assert_eq!(glia.microglia_activation.len(), 10);
        assert_eq!(glia.damage_signal.len(), 10);
    }

    #[test]
    fn test_uptake_fatigue_under_excitotoxicity() {
        let mut glia = GliaState::new(1);
        let mut neurons = NeuronArrays::new(1);
        let mut synapses = SynapseArrays::new(1);

        // Very high glutamate (excitotoxic)
        let glu_idx = NTType::Glutamate.index();
        neurons.nt_conc[0][glu_idx] = 10_000.0;

        let initial_uptake = glia.astrocyte_uptake[0];
        for _ in 0..100 {
            neurons.nt_conc[0][glu_idx] = 10_000.0; // keep it high
            glia.update(&mut neurons, &mut synapses, 10.0);
        }

        assert!(
            glia.astrocyte_uptake[0] < initial_uptake,
            "Uptake capacity should decrease under excitotoxicity: {} -> {}",
            initial_uptake,
            glia.astrocyte_uptake[0]
        );
    }

    #[test]
    fn test_mean_myelination() {
        let mut glia = GliaState::new(4);
        glia.myelin_integrity = vec![0.5, 0.5, 1.0, 1.0];
        assert!((glia.mean_myelination() - 0.75).abs() < 1e-6);
    }
}
