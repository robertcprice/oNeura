//! Spike propagation through synaptic connections.
//!
//! When a neuron fires, iterate its outgoing synapses (via CSR `row_offsets`),
//! trigger vesicle release, and inject postsynaptic current (PSC) into targets.
//! This is the serial step that bridges the GPU membrane integration and the
//! per-synapse molecular state, running once per simulation step for all fired
//! neurons.
//!
//! Performance note: we only iterate synapses of *fired* neurons, so cost is
//! proportional to `|fired| * mean_outdegree`, not `n_synapses`.

use crate::neuron_arrays::NeuronArrays;
use crate::synapse_arrays::SynapseArrays;
use crate::types::NTType;

/// Propagate spikes from fired neurons through their outgoing synapses.
///
/// For each fired neuron:
/// 1. Read its microdomain calcium level (vesicle release is Ca-dependent).
/// 2. Iterate all outgoing synapses via CSR row offsets.
/// 3. Call `presynaptic_spike()` to trigger vesicle release into the delay buffer.
/// 4. Inject PSC = `weight * psc_scale * sign` into the postsynaptic neuron's
///    `synaptic_current` accumulator for consumption on the next step.
///
/// The `sign` is -1 for GABAergic synapses (inhibitory) and +1 for all others
/// (excitatory).
///
/// # Arguments
/// * `neurons` - Mutable neuron arrays; `synaptic_current` is accumulated.
/// * `synapses` - Mutable synapse arrays; vesicle pools and pre-spike times updated.
/// * `fired_indices` - Indices of neurons that fired this step.
/// * `time` - Current simulation time in ms.
/// * `psc_scale` - Global PSC amplitude scaling (typically 30.0).
pub fn propagate_spikes(
    neurons: &mut NeuronArrays,
    synapses: &mut SynapseArrays,
    fired_indices: &[usize],
    time: f32,
    psc_scale: f32,
) {
    for &pre in fired_indices {
        let ca_level = neurons.ca_microdomain[pre];

        for syn_idx in synapses.outgoing_range(pre) {
            // Trigger vesicle release (Ca-dependent probability, fills delay buffer)
            synapses.presynaptic_spike(syn_idx, time, ca_level);

            // Identify postsynaptic target
            let post = synapses.col_indices[syn_idx] as usize;
            if neurons.alive[post] == 0 {
                continue;
            }

            // Compute PSC injection
            let nt = NTType::from_u8(synapses.nt_type[syn_idx]);
            let weight = synapses.weight[syn_idx];
            let psc = weight * psc_scale;

            // GABAergic synapses inject hyperpolarizing (negative) current;
            // all other NT types inject depolarizing (positive) current.
            let sign = match nt {
                NTType::GABA => -1.0f32,
                _ => 1.0f32,
            };

            neurons.synaptic_current[post] += sign * psc;
        }
    }
}

/// Propagate spikes with per-synapse delay awareness.
///
/// Unlike `propagate_spikes`, this variant does not inject PSC immediately
/// but instead only triggers vesicle release. The actual PSC injection
/// happens when `SynapseArrays::update_cleft()` processes the delay buffer
/// on a subsequent step. Use this when synaptic delays are meaningful.
pub fn propagate_spikes_delayed(
    neurons: &NeuronArrays,
    synapses: &mut SynapseArrays,
    fired_indices: &[usize],
    time: f32,
) {
    for &pre in fired_indices {
        let ca_level = neurons.ca_microdomain[pre];

        for syn_idx in synapses.outgoing_range(pre) {
            let post = synapses.col_indices[syn_idx] as usize;
            if neurons.alive[post] == 0 {
                continue;
            }

            synapses.presynaptic_spike(syn_idx, time, ca_level);
        }
    }
}

/// Update all active synaptic clefts and inject delayed PSC.
///
/// Call this once per step *after* `propagate_spikes_delayed` to process
/// the delay buffers, update cleft NT concentrations, and inject the
/// resulting PSC into postsynaptic neurons.
pub fn process_delayed_releases(
    neurons: &mut NeuronArrays,
    synapses: &mut SynapseArrays,
    time: f32,
    dt: f32,
    psc_scale: f32,
) {
    for syn_idx in 0..synapses.n_synapses {
        let cleft_conc = synapses.update_cleft(syn_idx, time, dt);
        if cleft_conc <= 0.0 {
            continue;
        }

        let post = synapses.col_indices[syn_idx] as usize;
        if neurons.alive[post] == 0 {
            continue;
        }

        // PSC proportional to cleft concentration (normalized by release amount)
        let nt = NTType::from_u8(synapses.nt_type[syn_idx]);
        let weight = synapses.weight[syn_idx];
        let conc_factor = (cleft_conc / 3000.0).min(1.0); // normalize to release quantum
        let psc = weight * psc_scale * conc_factor;

        let sign = match nt {
            NTType::GABA => -1.0f32,
            _ => 1.0f32,
        };

        neurons.synaptic_current[post] += sign * psc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::NTType;

    #[test]
    fn test_propagate_spikes_basic() {
        let mut neurons = NeuronArrays::new(3);
        // Neuron 0 fires
        neurons.fired[0] = 1;
        neurons.ca_microdomain[0] = 500.0;

        // Create a single synapse: 0 -> 1, glutamatergic
        let edges = vec![(0u32, 1u32, NTType::Glutamate)];
        let mut synapses = SynapseArrays::from_edges(3, &edges);

        let fired = vec![0usize];
        propagate_spikes(&mut neurons, &mut synapses, &fired, 10.0, 30.0);

        // Post-synaptic neuron 1 should have received positive current
        assert!(neurons.synaptic_current[1] > 0.0);
        // Pre-synaptic spike time should be recorded
        assert!((synapses.last_pre_spike[0] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_gaba_injects_negative_current() {
        let mut neurons = NeuronArrays::new(3);
        neurons.fired[0] = 1;
        neurons.ca_microdomain[0] = 500.0;

        let edges = vec![(0u32, 1u32, NTType::GABA)];
        let mut synapses = SynapseArrays::from_edges(3, &edges);

        let fired = vec![0usize];
        propagate_spikes(&mut neurons, &mut synapses, &fired, 10.0, 30.0);

        // GABAergic synapse should inject negative (inhibitory) current
        assert!(neurons.synaptic_current[1] < 0.0);
    }

    #[test]
    fn test_dead_neurons_skipped() {
        let mut neurons = NeuronArrays::new(3);
        neurons.fired[0] = 1;
        neurons.alive[1] = 0; // target is dead
        neurons.ca_microdomain[0] = 500.0;

        let edges = vec![(0u32, 1u32, NTType::Glutamate)];
        let mut synapses = SynapseArrays::from_edges(3, &edges);

        let fired = vec![0usize];
        propagate_spikes(&mut neurons, &mut synapses, &fired, 10.0, 30.0);

        // Dead neuron should not receive current
        assert!((neurons.synaptic_current[1]).abs() < 1e-12);
    }
}
