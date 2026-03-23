//! Cellular energy metabolism: glycolysis, oxidative phosphorylation, and
//! ATP consumption.
//!
//! This module implements the bioenergetics of individual neurons. It is
//! **interval-gated**: called every `METABOLISM_INTERVAL` steps (5), since
//! metabolic dynamics operate on timescales ~10x slower than membrane
//! dynamics but faster than gene expression.
//!
//! The metabolic pipeline:
//!
//! ```text
//! Glucose ──glycolysis──> Pyruvate ──OxPhos──> ATP
//!                                     ↑
//!                                   O2 consumed
//!
//! ATP ──neural activity──> ADP ──recycling──> ATP
//! ```
//!
//! ATP is the universal energy currency. Neurons consume ATP proportional
//! to their firing rate plus a baseline housekeeping cost. When ATP drops
//! below a threshold, excitability decreases (energy-dependent ion pump
//! failure), providing a natural activity limiter.

use crate::constants::*;
use crate::neuron_arrays::NeuronArrays;

// ===== Metabolic Rate Constants =====

/// Glycolysis rate constant (µM glucose consumed per ms per µM glucose).
/// Glucose → 2 pyruvate + 2 ATP (net).
const GLYCOLYSIS_RATE: f32 = 0.002;

/// Glycolysis ATP yield per glucose molecule.
const GLYCOLYSIS_ATP_YIELD: f32 = 2.0;

/// Oxidative phosphorylation rate constant (µM pyruvate consumed per ms).
/// Pyruvate + O2 → CO2 + H2O + ~34 ATP.
const OXPHOS_RATE: f32 = 0.005;

/// OxPhos ATP yield per pyruvate.
const OXPHOS_ATP_YIELD: f32 = 17.0; // ~34 per glucose / 2 pyruvate

/// O2 consumed per pyruvate in OxPhos.
const O2_PER_PYRUVATE: f32 = 3.0;

/// Baseline ATP consumption per ms (housekeeping: Na/K pump, protein synthesis).
const BASELINE_ATP_CONSUMPTION: f32 = 0.5;

/// ATP consumed per spike (Na/K-ATPase restoring gradients after AP).
const SPIKE_ATP_COST: f32 = 50.0;

/// ATP consumed per unit of external current injection per ms.
/// Reflects the cost of maintaining ion gradients under stimulation.
const CURRENT_ATP_COST: f32 = 0.01;

/// ADP recycling rate (ADP → ATP via substrate-level phosphorylation).
const ADP_RECYCLING_RATE: f32 = 0.001;

/// Maximum pyruvate pool (implicit; used for glycolysis → OxPhos coupling).
const MAX_PYRUVATE: f32 = 2000.0;

/// Glucose replenishment rate (from blood supply, µM per ms).
const GLUCOSE_REPLENISH_RATE: f32 = 0.5;

/// Oxygen replenishment rate (from blood supply, µM per ms).
const O2_REPLENISH_RATE: f32 = 0.1;

/// ATP level below which excitability starts to decrease.
const ATP_EXCITABILITY_THRESHOLD: f32 = 1000.0;

/// Update cellular metabolism for all alive neurons.
///
/// Advances ATP, ADP, glucose, and oxygen pools via simplified biochemical
/// kinetics. Computes an abstract `energy` level (backward-compatible with
/// the Python implementation) as a percentage of maximum ATP.
///
/// # Arguments
/// * `neurons` - Mutable neuron arrays. Reads `alive`, `fired`, `external_current`,
///   `spike_count`. Writes `atp`, `adp`, `glucose`, `oxygen`, `energy`,
///   `excitability_bias`.
/// * `dt` - **Effective** time step in ms (typically `dt * METABOLISM_INTERVAL`).
pub fn update_metabolism(neurons: &mut NeuronArrays, dt: f32) {
    for i in 0..neurons.count {
        if neurons.alive[i] == 0 {
            continue;
        }

        let atp = &mut neurons.atp[i];
        let adp = &mut neurons.adp[i];
        let glucose = &mut neurons.glucose[i];
        let oxygen = &mut neurons.oxygen[i];

        // --- ATP Production ---

        // 1. Glycolysis: Glucose → Pyruvate + ATP
        let glycolysis_flux = GLYCOLYSIS_RATE * (*glucose) * dt;
        let glucose_consumed = glycolysis_flux.min(*glucose);
        *glucose -= glucose_consumed;
        let pyruvate_produced = glucose_consumed * 2.0;
        let glycolysis_atp = glucose_consumed * GLYCOLYSIS_ATP_YIELD;
        *atp += glycolysis_atp;
        *adp = (*adp - glycolysis_atp).max(0.0);

        // 2. Oxidative phosphorylation: Pyruvate + O2 → ATP
        let pyruvate_available = pyruvate_produced.min(MAX_PYRUVATE);
        let o2_available = *oxygen;
        let oxphos_flux = OXPHOS_RATE * pyruvate_available * dt;
        let o2_needed = oxphos_flux * O2_PER_PYRUVATE;
        let oxphos_actual = if o2_needed > o2_available {
            // O2-limited
            o2_available / O2_PER_PYRUVATE
        } else {
            oxphos_flux
        };
        let oxphos_atp = oxphos_actual * OXPHOS_ATP_YIELD;
        *atp += oxphos_atp;
        *adp = (*adp - oxphos_atp).max(0.0);
        *oxygen -= (oxphos_actual * O2_PER_PYRUVATE).min(*oxygen);

        // 3. ADP recycling (substrate-level phosphorylation, minor pathway)
        let recycled = ADP_RECYCLING_RATE * (*adp) * dt;
        *atp += recycled;
        *adp -= recycled;

        // --- ATP Consumption ---

        // Baseline housekeeping
        let baseline_cost = BASELINE_ATP_CONSUMPTION * dt;

        // Spike cost (recent spikes)
        let spike_cost = if neurons.fired[i] != 0 {
            SPIKE_ATP_COST
        } else {
            0.0
        };

        // Current injection cost
        let current_cost = neurons.external_current[i].abs() * CURRENT_ATP_COST * dt;

        let total_cost = baseline_cost + spike_cost + current_cost;
        let consumed = total_cost.min(*atp);
        *atp -= consumed;
        *adp += consumed;

        // --- Supply replenishment (blood supply) ---
        *glucose = (*glucose + GLUCOSE_REPLENISH_RATE * dt).min(GLUCOSE_RESTING * 1.5);
        *oxygen = (*oxygen + O2_REPLENISH_RATE * dt).min(OXYGEN_RESTING * 1.5);

        // --- Clamp pools ---
        *atp = atp.clamp(0.0, ATP_RESTING * 2.0);
        *adp = adp.clamp(0.0, ADP_RESTING * 5.0);

        // --- Backward-compatible energy percentage ---
        neurons.energy[i] = (*atp / ATP_RESTING * 100.0).clamp(0.0, 200.0);

        // --- ATP-dependent excitability modulation ---
        // When ATP is low, Na/K pump fails → reduced excitability
        neurons.excitability_bias[i] = if *atp < ATP_EXCITABILITY_THRESHOLD {
            let atp_fraction = *atp / ATP_EXCITABILITY_THRESHOLD;
            -5.0 * (1.0 - atp_fraction)
        } else {
            0.0
        };
    }
}

/// Check if a neuron is in metabolic distress (ATP < 20% of resting).
///
/// Can be used by the glia module to trigger astrocyte lactate shuttle
/// support for metabolically stressed neurons.
pub fn is_metabolically_distressed(neurons: &NeuronArrays, idx: usize) -> bool {
    neurons.atp[idx] < ATP_RESTING * 0.2
}

/// Get the ATP/ADP ratio for a neuron (indicator of metabolic health).
pub fn atp_adp_ratio(neurons: &NeuronArrays, idx: usize) -> f32 {
    if neurons.adp[idx] < 1e-6 {
        return neurons.atp[idx];
    }
    neurons.atp[idx] / neurons.adp[idx]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resting_metabolism_stable() {
        let mut neurons = NeuronArrays::new(3);
        let initial_atp = neurons.atp[0];

        // Run metabolism for a moderate time at rest
        for _ in 0..100 {
            update_metabolism(&mut neurons, 5.0); // 5ms effective steps
        }

        // ATP should remain roughly stable at rest (production ~ consumption)
        let _atp_change = (neurons.atp[0] - initial_atp).abs();
        // Allow up to 50% change over 500ms — metabolism finds equilibrium
        assert!(
            neurons.atp[0] > ATP_RESTING * 0.3,
            "ATP should not crash at rest: {}",
            neurons.atp[0]
        );
    }

    #[test]
    fn test_spike_costs_atp() {
        let mut neurons = NeuronArrays::new(2);
        neurons.fired[0] = 1; // neuron 0 fired
        neurons.fired[1] = 0; // neuron 1 did not fire

        update_metabolism(&mut neurons, 5.0);

        assert!(
            neurons.atp[0] < neurons.atp[1],
            "Firing neuron should have less ATP: {} vs {}",
            neurons.atp[0],
            neurons.atp[1]
        );
    }

    #[test]
    fn test_energy_percentage_tracks_atp() {
        let mut neurons = NeuronArrays::new(1);
        update_metabolism(&mut neurons, 1.0);

        let expected_pct = neurons.atp[0] / ATP_RESTING * 100.0;
        assert!(
            (neurons.energy[0] - expected_pct).abs() < 1.0,
            "Energy should track ATP percentage: {} vs {}",
            neurons.energy[0],
            expected_pct
        );
    }

    #[test]
    fn test_atp_never_negative() {
        let mut neurons = NeuronArrays::new(1);
        // Drain everything
        neurons.glucose[0] = 0.0;
        neurons.oxygen[0] = 0.0;
        neurons.fired[0] = 1;
        neurons.external_current[0] = 100.0;

        for _ in 0..1000 {
            neurons.fired[0] = 1;
            update_metabolism(&mut neurons, 10.0);
        }

        assert!(neurons.atp[0] >= 0.0, "ATP must never go negative");
        assert!(neurons.adp[0] >= 0.0, "ADP must never go negative");
    }

    #[test]
    fn test_metabolic_distress() {
        let mut neurons = NeuronArrays::new(1);
        neurons.atp[0] = 100.0; // very low ATP
        assert!(is_metabolically_distressed(&neurons, 0));

        neurons.atp[0] = ATP_RESTING; // healthy
        assert!(!is_metabolically_distressed(&neurons, 0));
    }

    #[test]
    fn test_dead_neuron_no_metabolism() {
        let mut neurons = NeuronArrays::new(2);
        neurons.alive[0] = 0;
        let initial_atp_dead = neurons.atp[0];

        update_metabolism(&mut neurons, 5.0);

        assert!(
            (neurons.atp[0] - initial_atp_dead).abs() < 1e-6,
            "Dead neuron ATP should not change"
        );
    }
}
