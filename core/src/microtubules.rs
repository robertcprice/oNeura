//! Microtubule quantum coherence and Orch-OR (Orchestrated Objective
//! Reduction).
//!
//! This module implements a simplified model of quantum coherence in
//! tubulin dimers, based on the Penrose-Hameroff Orch-OR theory. Each
//! neuron tracks a coherence level in [0, 1] representing the fraction
//! of tubulin dimers in a quantum superposition state, and accumulates
//! "collapse events" (objective reductions) that contribute to a
//! consciousness metric.
//!
//! **Interval-gated**: called every `CYTOSKELETON_INTERVAL` steps (10),
//! since quantum coherence dynamics are modeled at a coarser timescale
//! than membrane dynamics.
//!
//! Key physics (simplified):
//!
//! - **Coherence growth**: In the absence of thermal noise, tubulin dimers
//!   can maintain quantum superposition. Lower temperature and lower neural
//!   noise (firing rate) favor coherence growth.
//!
//! - **Decoherence**: Thermal noise, neural activity, and environmental
//!   coupling cause loss of coherence. Modeled as exponential decay.
//!
//! - **Objective reduction (OR)**: When coherence exceeds a threshold, a
//!   "gravitational self-energy" criterion triggers wavefunction collapse.
//!   Each collapse is counted as an Orch-OR event, thought to be a moment
//!   of proto-consciousness.
//!
//! - **Anesthesia suppression**: General anesthetics dramatically reduce
//!   Orch-OR events, consistent with the clinical observation that
//!   anesthesia eliminates consciousness.

use crate::constants::*;
use crate::neuron_arrays::NeuronArrays;

// ===== Microtubule Constants =====

/// Base coherence growth rate per ms.
/// Represents the natural tendency of tubulin dimers to enter superposition
/// in the absence of decoherence. At optimal conditions (good Ca, good ATP),
/// growth must exceed thermal decoherence for coherence to reach collapse threshold.
const COHERENCE_GROWTH_RATE: f32 = 0.002;

/// Decoherence rate per ms from thermal noise.
/// At biological temperature (~310 K), decoherence is significant but not
/// dominant — hydrophobic pockets in tubulin shield quantum states.
const THERMAL_DECOHERENCE_RATE: f32 = 0.001;

/// Additional decoherence from neural activity (per spike per ms).
/// Action potentials disrupt quantum coherence via electromagnetic noise.
const ACTIVITY_DECOHERENCE_RATE: f32 = 0.02;

/// Coherence threshold for objective reduction (collapse).
/// When coherence exceeds this, the superposition self-collapses.
const OR_THRESHOLD: f32 = 0.6;

/// Number of tubulin dimers per microtubule (determines gravitational
/// self-energy, which sets the collapse threshold in Orch-OR theory).
/// Reserved for detailed Orch-OR gravitational self-energy computation.
const _N_TUBULIN_SUPERPOSED: u32 = 2000;

/// Coherence recovered after a collapse event.
/// Not zero, because some dimers re-enter superposition immediately.
const POST_COLLAPSE_COHERENCE: f32 = 0.1;

/// Calcium enhancement factor for coherence.
/// Moderate calcium promotes microtubule stability, supporting coherence.
/// Very high calcium (>10,000 nM) triggers depolymerization and hurts.
const CA_OPTIMAL_NM: f32 = 500.0;

/// ATP-dependent microtubule maintenance.
/// Tubulin polymerization requires GTP (proxy: ATP). Low ATP = decoherence.
const ATP_COHERENCE_THRESHOLD: f32 = 2000.0;

/// Update microtubule quantum coherence for all alive neurons.
///
/// Advances the coherence level via a simplified ODE model:
///
/// ```text
/// d[coherence]/dt = growth * (1 - coherence) * stability_factor
///                 - thermal_decoherence * coherence
///                 - activity_decoherence * coherence * recently_fired
/// ```
///
/// When coherence exceeds `OR_THRESHOLD`, an objective reduction event
/// occurs: coherence drops to `POST_COLLAPSE_COHERENCE` and the neuron's
/// `orch_or_events` counter increments.
///
/// # Arguments
/// * `neurons` - Mutable neuron arrays. Reads `alive`, `ca_cytoplasmic`,
///   `atp`, `fired`. Writes `mt_coherence`, `orch_or_events`.
/// * `dt` - **Effective** time step in ms (typically `dt * CYTOSKELETON_INTERVAL`).
pub fn update_microtubules(neurons: &mut NeuronArrays, dt: f32) {
    for i in 0..neurons.count {
        if neurons.alive[i] == 0 {
            continue;
        }

        let coherence = &mut neurons.mt_coherence[i];

        // --- Stability factors ---

        // Calcium stability: moderate Ca promotes microtubule stability,
        // extreme Ca (depolymerization) hurts coherence.
        let ca = neurons.ca_cytoplasmic[i];
        let ca_stability = if ca < CA_OPTIMAL_NM {
            // Low Ca: moderate stability
            0.5 + 0.5 * (ca / CA_OPTIMAL_NM)
        } else if ca < 10_000.0 {
            // Optimal range
            1.0
        } else {
            // Very high Ca: depolymerization → poor coherence
            (10_000.0 / ca).max(0.1)
        };

        // ATP-dependent stability: tubulin GTP cap requires energy
        let atp = neurons.atp[i];
        let atp_stability = if atp > ATP_COHERENCE_THRESHOLD {
            1.0
        } else {
            (atp / ATP_COHERENCE_THRESHOLD).max(0.1)
        };

        let stability = ca_stability * atp_stability;

        // --- Coherence growth ---
        let growth = COHERENCE_GROWTH_RATE * (1.0 - *coherence) * stability * dt;

        // --- Decoherence ---
        let thermal = THERMAL_DECOHERENCE_RATE * (*coherence) * dt;
        let activity = if neurons.fired[i] != 0 {
            ACTIVITY_DECOHERENCE_RATE * (*coherence) * dt
        } else {
            0.0
        };

        *coherence += growth - thermal - activity;
        *coherence = clamp(*coherence, 0.0, 1.0);

        // --- Objective Reduction check ---
        if *coherence >= OR_THRESHOLD {
            // Orch-OR collapse event!
            neurons.orch_or_events[i] += 1;
            *coherence = POST_COLLAPSE_COHERENCE;
        }
    }
}

/// Suppress Orch-OR via anesthetic agents.
///
/// General anesthesia is thought to suppress quantum coherence in
/// microtubules, which in Orch-OR theory explains loss of consciousness.
/// This function applies a multiplicative suppression to all neurons'
/// coherence levels.
///
/// # Arguments
/// * `neurons` - Mutable neuron arrays. Writes `mt_coherence`.
/// * `suppression_factor` - Factor in [0, 1]; e.g., 0.05 for 95% suppression.
pub fn apply_anesthesia_suppression(neurons: &mut NeuronArrays, suppression_factor: f32) {
    let factor = clamp(suppression_factor, 0.0, 1.0);
    for i in 0..neurons.count {
        if neurons.alive[i] == 0 {
            continue;
        }
        neurons.mt_coherence[i] *= factor;
    }
}

/// Get the total Orch-OR event count across all neurons.
///
/// This serves as a raw consciousness metric: more collapse events
/// per unit time = higher "proto-conscious" activity.
pub fn total_orch_or_events(neurons: &NeuronArrays) -> u64 {
    neurons.orch_or_events.iter().map(|&e| e as u64).sum()
}

/// Compute Orch-OR rate (events per second) for the network.
///
/// # Arguments
/// * `neurons` - Neuron arrays (read `orch_or_events`).
/// * `elapsed_ms` - Total simulation time elapsed in ms.
pub fn orch_or_rate_hz(neurons: &NeuronArrays, elapsed_ms: f32) -> f32 {
    if elapsed_ms <= 0.0 {
        return 0.0;
    }
    let total = total_orch_or_events(neurons) as f32;
    total / (elapsed_ms / 1000.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coherence_grows_at_rest() {
        let mut neurons = NeuronArrays::new(1);
        neurons.atp[0] = ATP_RESTING; // healthy ATP
        neurons.ca_cytoplasmic[0] = CA_OPTIMAL_NM;

        // Run for many intervals
        for _ in 0..200 {
            update_microtubules(&mut neurons, 10.0);
        }

        // Coherence should have grown from 0
        // (may have collapsed and regrown)
        assert!(
            neurons.mt_coherence[0] > 0.0 || neurons.orch_or_events[0] > 0,
            "Coherence should grow or OR events should occur"
        );
    }

    #[test]
    fn test_or_events_occur() {
        let mut neurons = NeuronArrays::new(1);
        neurons.atp[0] = ATP_RESTING;
        neurons.ca_cytoplasmic[0] = CA_OPTIMAL_NM;

        // Run for enough time to trigger at least one OR event
        for _ in 0..5000 {
            update_microtubules(&mut neurons, 10.0);
        }

        assert!(
            neurons.orch_or_events[0] > 0,
            "Should have at least one OR event after extended simulation"
        );
    }

    #[test]
    fn test_anesthesia_suppresses_coherence() {
        let mut neurons = NeuronArrays::new(3);
        for i in 0..3 {
            neurons.mt_coherence[i] = 0.5;
        }

        apply_anesthesia_suppression(&mut neurons, 0.05);

        for i in 0..3 {
            assert!(
                neurons.mt_coherence[i] < 0.03,
                "Anesthesia should suppress coherence to ~5%: {}",
                neurons.mt_coherence[i]
            );
        }
    }

    #[test]
    fn test_coherence_bounded() {
        let mut neurons = NeuronArrays::new(1);
        neurons.mt_coherence[0] = 1.5; // artificially high
        update_microtubules(&mut neurons, 10.0);
        assert!(neurons.mt_coherence[0] <= 1.0, "Coherence must be <= 1.0");
    }

    #[test]
    fn test_high_calcium_hurts_coherence() {
        let mut neurons_optimal = NeuronArrays::new(1);
        let mut neurons_toxic = NeuronArrays::new(1);
        neurons_optimal.atp[0] = ATP_RESTING;
        neurons_toxic.atp[0] = ATP_RESTING;
        neurons_optimal.ca_cytoplasmic[0] = CA_OPTIMAL_NM;
        neurons_toxic.ca_cytoplasmic[0] = 50_000.0; // depolymerization level

        for _ in 0..500 {
            update_microtubules(&mut neurons_optimal, 10.0);
            update_microtubules(&mut neurons_toxic, 10.0);
        }

        // Optimal should have more OR events (more coherence growth)
        // or higher coherence than toxic
        let optimal_activity =
            neurons_optimal.orch_or_events[0] as f32 + neurons_optimal.mt_coherence[0] * 10.0;
        let toxic_activity =
            neurons_toxic.orch_or_events[0] as f32 + neurons_toxic.mt_coherence[0] * 10.0;

        assert!(
            optimal_activity >= toxic_activity,
            "Optimal Ca should have more coherence activity: {} vs {}",
            optimal_activity,
            toxic_activity
        );
    }

    #[test]
    fn test_dead_neuron_no_coherence() {
        let mut neurons = NeuronArrays::new(1);
        neurons.alive[0] = 0;
        neurons.atp[0] = ATP_RESTING;

        for _ in 0..100 {
            update_microtubules(&mut neurons, 10.0);
        }

        assert!(
            neurons.mt_coherence[0] < 1e-6,
            "Dead neuron should have no coherence growth"
        );
    }
}
