//! Spike-Timing-Dependent Plasticity (STDP) with BCM metaplasticity,
//! receptor trafficking, synaptic tagging, and homeostatic scaling.
//!
//! This module implements the biologically realistic learning rules that
//! modify synaptic strength based on the relative timing of pre- and
//! postsynaptic spikes. It is called once per simulation step and only
//! processes synapses where the pre- OR post-neuron fired, keeping cost
//! proportional to active synapses rather than total synapse count.
//!
//! Key mechanisms:
//! - **STDP window**: 20ms asymmetric window; pre-before-post = LTP,
//!   post-before-pre = LTD.
//! - **NMDA gating**: STDP magnitude is modulated by the NMDA Mg2+ block
//!   at the postsynaptic voltage (coincidence detection).
//! - **BCM metaplasticity**: Sliding threshold (theta) tracks postsynaptic
//!   activity; above theta = LTP, below = LTD. Prevents runaway potentiation.
//! - **Receptor trafficking**: LTP inserts AMPA receptors (+1), LTD removes
//!   them (-1). This is the molecular basis of weight change.
//! - **Synaptic tagging**: Strong plasticity events set a "tag" for later
//!   protein-synthesis-dependent consolidation.
//! - **Homeostatic scaling**: Slowly scales all weights toward a target
//!   firing rate to maintain network stability.

use crate::constants::*;
use crate::neuron_arrays::NeuronArrays;
use crate::synapse_arrays::SynapseArrays;
use crate::types::NTType;

/// NMDA Mg2+ block factor at a given membrane voltage.
///
/// Returns a value in [0, 1] representing the fraction of NMDA channels
/// unblocked by Mg2+ at the given voltage. At resting potential (-65 mV),
/// ~94% are blocked. Near 0 mV, the block is almost fully relieved.
#[inline(always)]
fn nmda_mg_block(voltage: f32) -> f32 {
    1.0 / (1.0 + NMDA_MG_CONC_MM * (-0.062 * voltage).exp() / 3.57)
}

/// Exponential STDP kernel: magnitude decays exponentially with time delta.
#[inline(always)]
fn stdp_kernel(delta_t: f32, tau: f32) -> f32 {
    (-delta_t.abs() / tau).exp()
}

/// Target firing rate for homeostatic scaling, in Hz.
const TARGET_FIRING_RATE_HZ: f32 = 5.0;

/// Homeostatic scaling time constant in ms.
const HOMEOSTATIC_TAU_MS: f32 = 100_000.0;

/// BCM theta time constant in ms (sliding average).
const BCM_THETA_TAU_MS: f32 = 10_000.0;

/// Synaptic tag threshold: plasticity events larger than this get tagged.
const TAG_THRESHOLD: f32 = 0.3;

/// Tag decay time constant in ms.
const TAG_DECAY_TAU_MS: f32 = 60_000.0;

/// Update STDP, BCM, receptor trafficking, tagging, and homeostatic scaling.
///
/// This function processes all synapses where either the presynaptic or
/// postsynaptic neuron fired in the current step. For each such synapse:
///
/// 1. Compute spike-timing delta and STDP magnitude.
/// 2. Gate by NMDA Mg2+ block at postsynaptic voltage.
/// 3. Apply BCM metaplasticity: compare post activity to theta threshold.
/// 4. Traffic AMPA receptors: LTP inserts, LTD removes.
/// 5. Tag strong events for protein-synthesis consolidation.
/// 6. Update BCM theta (sliding average of postsynaptic activity).
/// 7. Recompute effective weight from receptor counts.
///
/// Homeostatic scaling runs over all synapses (not just active ones) at a
/// much slower rate.
///
/// # Arguments
/// * `neurons` - Neuron arrays (read spike_count, voltage, fired, last_fired_step).
/// * `synapses` - Mutable synapse arrays (weights, receptors, BCM state, tags).
/// * `fired_indices` - Indices of neurons that fired this step.
/// * `time` - Current simulation time in ms.
/// * `dt` - Time step in ms.
pub fn update_stdp(
    neurons: &NeuronArrays,
    synapses: &mut SynapseArrays,
    fired_indices: &[usize],
    time: f32,
    dt: f32,
) {
    // Build a quick lookup set of fired neurons for this step.
    // For small networks, a Vec<bool> is faster than a HashSet.
    let mut fired_set = vec![false; neurons.count];
    for &idx in fired_indices {
        fired_set[idx] = true;
    }

    // --- Phase 1: STDP coincidence -> eligibility accumulation ---
    // Only process synapses where pre OR post fired.
    for &pre in fired_indices {
        for syn_idx in synapses.outgoing_range(pre) {
            let post = synapses.col_indices[syn_idx] as usize;
            if neurons.alive[post] == 0 {
                continue;
            }

            // Pre fired: check for LTD (post fired BEFORE pre = depression)
            let last_post = synapses.last_post_spike[syn_idx];
            let delta_t_ltd = time - last_post;

            if delta_t_ltd > 0.0 && delta_t_ltd < STDP_WINDOW_MS {
                // Post before pre = LTD
                let raw_ltd = -STDP_LTD_RATE * stdp_kernel(delta_t_ltd, STDP_WINDOW_MS);

                // NMDA gating: LTD is weaker when NMDA is blocked (less Ca2+ influx)
                let nmda_gate = nmda_mg_block(neurons.voltage[post]);
                let ltd = raw_ltd * nmda_gate;

                // BCM modulation: if post activity is below theta, amplify LTD
                let bcm_factor = bcm_modulation(
                    synapses.post_activity_history[syn_idx],
                    synapses.bcm_theta[syn_idx],
                );
                let adjusted_ltd = ltd * (1.0 - bcm_factor).max(0.1);

                synapses.eligibility_trace[syn_idx] =
                    (synapses.eligibility_trace[syn_idx] + adjusted_ltd).clamp(-2.0, 2.0);

                // Synaptic tagging for strong LTD events
                if adjusted_ltd.abs() > TAG_THRESHOLD {
                    synapses.tagged[syn_idx] = 1;
                    synapses.tag_strength[syn_idx] = adjusted_ltd.abs();
                }
            }

            // Update last_pre_spike for this synapse (for LTP detection by post)
            // Note: presynaptic_spike() in synapse_arrays.rs already sets this,
            // but we set it here too to ensure it's current even if propagation
            // didn't call presynaptic_spike (e.g., in delayed mode).
            synapses.last_pre_spike[syn_idx] = time;
        }
    }

    // Process synapses where POST fired (for LTP: pre before post)
    for &post in fired_indices {
        // We need to find all synapses targeting this neuron.
        // CSR is organized by source, so we scan all neurons' outgoing synapses.
        // For large networks, an inverse index (incoming synapses per neuron) would
        // be more efficient, but for the molecular brain scale this is acceptable.
        for pre in 0..synapses.n_neurons {
            for syn_idx in synapses.outgoing_range(pre) {
                if synapses.col_indices[syn_idx] as usize != post {
                    continue;
                }

                // Post fired: check for LTP (pre fired BEFORE post = potentiation)
                let last_pre = synapses.last_pre_spike[syn_idx];
                let delta_t_ltp = time - last_pre;

                if delta_t_ltp > 0.0 && delta_t_ltp < STDP_WINDOW_MS {
                    // Pre before post = LTP
                    let raw_ltp = STDP_LTP_RATE * stdp_kernel(delta_t_ltp, STDP_WINDOW_MS);

                    // NMDA gating: LTP requires NMDA unblock (post depolarized)
                    let nmda_gate = nmda_mg_block(neurons.voltage[post]);
                    let ltp = raw_ltp * nmda_gate;

                    // BCM modulation: if post activity is above theta, amplify LTP
                    let bcm_factor = bcm_modulation(
                        synapses.post_activity_history[syn_idx],
                        synapses.bcm_theta[syn_idx],
                    );
                    let adjusted_ltp = ltp * bcm_factor.max(0.1);

                    synapses.eligibility_trace[syn_idx] =
                        (synapses.eligibility_trace[syn_idx] + adjusted_ltp).clamp(-2.0, 2.0);

                    // Synaptic tagging for strong LTP events
                    if adjusted_ltp.abs() > TAG_THRESHOLD {
                        synapses.tagged[syn_idx] = 1;
                        synapses.tag_strength[syn_idx] = adjusted_ltp.abs();
                    }
                }

                // Update last_post_spike time
                synapses.last_post_spike[syn_idx] = time;

                // Update BCM theta: sliding average of post activity
                let post_active = if fired_set[post] { 1.0f32 } else { 0.0f32 };
                let alpha = (dt / BCM_THETA_TAU_MS).min(1.0);
                synapses.post_activity_history[syn_idx] =
                    synapses.post_activity_history[syn_idx] * (1.0 - alpha) + post_active * alpha;
                synapses.bcm_theta[syn_idx] = synapses.post_activity_history[syn_idx]
                    * synapses.post_activity_history[syn_idx];
            }
        }
    }

    // --- Phase 2: dopamine-gated conversion of eligibility to permanent change ---
    for syn_idx in 0..synapses.n_synapses {
        if synapses.nt_type[syn_idx] == NTType::GABA.index() as u8 {
            continue;
        }
        let post = synapses.col_indices[syn_idx] as usize;
        let da_post = neurons.nt_conc[post][NTType::Dopamine.index()];
        let da_above_rest = (da_post - 20.0).max(0.0);
        let da_gain = (da_above_rest / 20.0).clamp(0.0, 10.0);
        if da_gain <= 0.1 {
            continue;
        }
        let gated_delta = synapses.eligibility_trace[syn_idx] * da_gain * 0.1;
        if gated_delta.abs() > 1.0e-6 {
            apply_receptor_change(synapses, syn_idx, gated_delta);
        }
        synapses.eligibility_trace[syn_idx] *= 0.9;
    }

    // --- Phase 3: Eligibility trace decay ---
    for syn_idx in 0..synapses.n_synapses {
        synapses.eligibility_trace[syn_idx] *= (-dt / STDP_WINDOW_MS).exp();
    }

    // --- Phase 4: Tag decay ---
    let tag_decay = (-dt / TAG_DECAY_TAU_MS).exp();
    for syn_idx in 0..synapses.n_synapses {
        if synapses.tagged[syn_idx] != 0 {
            synapses.tag_strength[syn_idx] *= tag_decay;
            if synapses.tag_strength[syn_idx] < 0.01 {
                synapses.tagged[syn_idx] = 0;
                synapses.tag_strength[syn_idx] = 0.0;
            }
        }
    }
}

/// Run homeostatic scaling over all synapses.
///
/// This is a slow process (time constant ~100s) that scales all synaptic
/// weights toward a target firing rate. Call this every ~100 steps, not
/// every step.
///
/// # Arguments
/// * `neurons` - Neuron arrays (read spike_count).
/// * `synapses` - Mutable synapse arrays (homeostatic_scale, weight).
/// * `total_time_ms` - Total simulation time elapsed, for rate computation.
/// * `dt` - Time step in ms.
pub fn update_homeostatic_scaling(
    neurons: &NeuronArrays,
    synapses: &mut SynapseArrays,
    total_time_ms: f32,
    dt: f32,
) {
    if total_time_ms <= 0.0 {
        return;
    }

    let alpha = (dt / HOMEOSTATIC_TAU_MS).min(1.0);

    for pre in 0..synapses.n_neurons {
        for syn_idx in synapses.outgoing_range(pre) {
            let post = synapses.col_indices[syn_idx] as usize;

            // Compute postsynaptic firing rate in Hz
            let post_rate_hz = (neurons.spike_count[post] as f32) / (total_time_ms / 1000.0);

            // Scale factor: > 1 if under target, < 1 if over target
            let ratio = TARGET_FIRING_RATE_HZ / (post_rate_hz + 0.1);
            let target_scale = ratio.clamp(0.5, 2.0);

            synapses.homeostatic_scale[syn_idx] =
                synapses.homeostatic_scale[syn_idx] * (1.0 - alpha) + target_scale * alpha;

            synapses.recompute_weight(syn_idx);
        }
    }
}

/// BCM modulation factor: returns a value > 1 when post activity exceeds
/// theta (favoring LTP) and < 1 when below theta (favoring LTD).
#[inline(always)]
fn bcm_modulation(post_activity: f32, theta: f32) -> f32 {
    if theta < 1e-6 {
        return 1.0;
    }
    // Bienenstock-Cooper-Munro: phi(y) = y * (y - theta)
    // Simplified to a ratio for modulating STDP magnitude.
    let ratio = post_activity / theta;
    ratio.clamp(0.0, 3.0)
}

/// Apply receptor trafficking based on plasticity magnitude.
///
/// Positive `delta` (LTP) inserts AMPA receptors; negative (LTD) removes them.
/// NMDA receptor count is also modestly affected (slower trafficking).
fn apply_receptor_change(synapses: &mut SynapseArrays, syn_idx: usize, delta: f32) {
    // AMPA trafficking: primary mechanism for synaptic weight change
    let ampa_change = (delta * 2.0).round() as i32; // scale: 0.5 LTP ~= +1 receptor
    let new_ampa = (synapses.ampa_receptors[syn_idx] as i32 + ampa_change).clamp(1, 200) as u16;
    synapses.ampa_receptors[syn_idx] = new_ampa;

    // NMDA trafficking: slower, ~10% of AMPA rate
    if delta.abs() > 0.3 {
        let nmda_change = (delta * 0.2).round() as i32;
        let new_nmda = (synapses.nmda_receptors[syn_idx] as i32 + nmda_change).clamp(1, 100) as u16;
        synapses.nmda_receptors[syn_idx] = new_nmda;
    }

    // Update effective weight from new receptor counts
    synapses.recompute_weight(syn_idx);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::NTType;

    #[test]
    fn test_nmda_mg_block_resting() {
        // At resting potential, NMDA should be mostly blocked
        let block = nmda_mg_block(-65.0);
        assert!(
            block < 0.15,
            "NMDA should be ~94% blocked at -65mV, got {}",
            block
        );
    }

    #[test]
    fn test_nmda_mg_block_depolarized() {
        // At 0 mV, block should be mostly relieved
        let block = nmda_mg_block(0.0);
        assert!(
            block > 0.6,
            "NMDA should be mostly unblocked at 0mV, got {}",
            block
        );
    }

    #[test]
    fn test_stdp_kernel_at_zero() {
        let k = stdp_kernel(0.0, 20.0);
        assert!((k - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_stdp_kernel_decays() {
        let k1 = stdp_kernel(5.0, 20.0);
        let k2 = stdp_kernel(15.0, 20.0);
        assert!(k1 > k2, "STDP kernel should decay with distance");
    }

    #[test]
    fn test_bcm_modulation_above_theta() {
        let factor = bcm_modulation(0.8, 0.4);
        assert!(factor > 1.0, "Above theta should favor LTP, got {}", factor);
    }

    #[test]
    fn test_bcm_modulation_below_theta() {
        let factor = bcm_modulation(0.2, 0.4);
        assert!(factor < 1.0, "Below theta should favor LTD, got {}", factor);
    }

    #[test]
    fn test_receptor_trafficking_ltp() {
        let edges = vec![(0u32, 1u32, NTType::Glutamate)];
        let mut synapses = SynapseArrays::from_edges(2, &edges);
        let initial_ampa = synapses.ampa_receptors[0];

        apply_receptor_change(&mut synapses, 0, 0.5);

        assert!(
            synapses.ampa_receptors[0] >= initial_ampa,
            "LTP should insert AMPA receptors"
        );
    }

    #[test]
    fn test_receptor_trafficking_ltd() {
        let edges = vec![(0u32, 1u32, NTType::Glutamate)];
        let mut synapses = SynapseArrays::from_edges(2, &edges);
        let initial_ampa = synapses.ampa_receptors[0];

        apply_receptor_change(&mut synapses, 0, -0.5);

        assert!(
            synapses.ampa_receptors[0] <= initial_ampa,
            "LTD should remove AMPA receptors"
        );
    }
}
