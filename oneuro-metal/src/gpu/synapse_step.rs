//! GPU dispatch for synapse-side dynamics.
//!
//! This covers the work that previously bounced back to CPU after membrane
//! integration: spike-triggered release, next-step synaptic current
//! accumulation, STDP, cleft decay, and `last_fired_step` updates.

#[cfg(not(target_os = "macos"))]
use crate::neuron_arrays::NeuronArrays;
#[cfg(not(target_os = "macos"))]
use crate::synapse_arrays::SynapseArrays;

#[cfg(target_os = "macos")]
use super::state::{MetalNeuronState, MetalSynapseState};
#[cfg(target_os = "macos")]
use super::GpuContext;
#[cfg(target_os = "macos")]
use metal::CommandBufferRef;

const CURRENT_SCALE: f32 = 1000.0;

#[repr(C)]
struct SynapseStepParams {
    synapse_count: u32,
    time_ms: f32,
    dt: f32,
    psc_scale: f32,
    current_scale: f32,
}

#[repr(C)]
struct CurrentCommitParams {
    neuron_count: u32,
    current_scale: f32,
}

#[repr(C)]
struct LastFiredParams {
    neuron_count: u32,
    step_index: u32,
}

#[repr(C)]
struct ClearIntParams {
    count: u32,
}

#[cfg(target_os = "macos")]
pub fn dispatch_synapse_step(
    gpu: &GpuContext,
    neurons: &MetalNeuronState,
    synapses: &MetalSynapseState,
    time_ms: f32,
    dt: f32,
    psc_scale: f32,
    step_index: u32,
) {
    let cmd = gpu.new_command_buffer();
    encode_synapse_step(
        gpu, &cmd, neurons, synapses, time_ms, dt, psc_scale, step_index,
    );
    gpu.commit_and_wait(cmd);
}

#[cfg(target_os = "macos")]
pub fn encode_synapse_step(
    gpu: &GpuContext,
    cmd: &CommandBufferRef,
    neurons: &MetalNeuronState,
    synapses: &MetalSynapseState,
    time_ms: f32,
    dt: f32,
    psc_scale: f32,
    step_index: u32,
) {
    let synapse_threads = synapses.count as u64;
    let neuron_threads = neurons.count as u64;
    if synapse_threads == 0 && neuron_threads == 0 {
        return;
    }

    let count = synapses.count;
    let neuron_count = neurons.count;

    if neuron_threads > 0 {
        let clear_params = ClearIntParams {
            count: neuron_count as u32,
        };
        let clear_bytes = unsafe {
            std::slice::from_raw_parts(
                &clear_params as *const ClearIntParams as *const u8,
                std::mem::size_of::<ClearIntParams>(),
            )
        };
        gpu.encode_1d(
            cmd,
            &gpu.pipelines.clear_i32_buffer,
            &[(&neurons.synaptic_current_accum_i32, 0)],
            Some((clear_bytes, 1)),
            neuron_threads,
        );
    }

    if synapse_threads > 0 {
        let params = SynapseStepParams {
            synapse_count: count as u32,
            time_ms,
            dt,
            psc_scale,
            current_scale: CURRENT_SCALE,
        };
        let param_bytes = unsafe {
            std::slice::from_raw_parts(
                &params as *const SynapseStepParams as *const u8,
                std::mem::size_of::<SynapseStepParams>(),
            )
        };

        gpu.encode_1d(
            cmd,
            &gpu.pipelines.synapse_step,
            &[
                (&synapses.pre_indices, 0),
                (&synapses.col_indices, 0),
                (&synapses.nt_type, 0),
                (&synapses.weight, 0),
                (&synapses.strength, 0),
                (&synapses.vesicle_rrp, 0),
                (&synapses.vesicle_recycling, 0),
                (&synapses.vesicle_reserve, 0),
                (&synapses.cleft_concentration, 0),
                (&synapses.ampa_receptors, 0),
                (&synapses.nmda_receptors, 0),
                (&synapses.gabaa_receptors, 0),
                (&synapses.last_pre_spike, 0),
                (&synapses.last_post_spike, 0),
                (&synapses.eligibility_trace, 0),
                (&synapses.bcm_theta, 0),
                (&synapses.post_activity_history, 0),
                (&synapses.tagged, 0),
                (&synapses.tag_strength, 0),
                (&synapses.homeostatic_scale, 0),
                (&neurons.fired, 0),
                (&neurons.alive, 0),
                (&neurons.voltage, 0),
                (&neurons.ca_microdomain, 0),
                (&neurons.nt_conc, 0),
                (&neurons.synaptic_current_accum_i32, 0),
            ],
            Some((param_bytes, 26)),
            synapse_threads,
        );
    }

    if neuron_threads > 0 {
        let current_params = CurrentCommitParams {
            neuron_count: neuron_count as u32,
            current_scale: CURRENT_SCALE,
        };
        let current_bytes = unsafe {
            std::slice::from_raw_parts(
                &current_params as *const CurrentCommitParams as *const u8,
                std::mem::size_of::<CurrentCommitParams>(),
            )
        };
        gpu.encode_1d(
            cmd,
            &gpu.pipelines.synaptic_current_commit,
            &[
                (&neurons.synaptic_current_accum_i32, 0),
                (&neurons.synaptic_current, 0),
            ],
            Some((current_bytes, 2)),
            neuron_threads,
        );

        let fired_params = LastFiredParams {
            neuron_count: neuron_count as u32,
            step_index,
        };
        let fired_bytes = unsafe {
            std::slice::from_raw_parts(
                &fired_params as *const LastFiredParams as *const u8,
                std::mem::size_of::<LastFiredParams>(),
            )
        };
        gpu.encode_1d(
            cmd,
            &gpu.pipelines.mark_last_fired,
            &[(&neurons.fired, 0), (&neurons.last_fired_step, 0)],
            Some((fired_bytes, 2)),
            neuron_threads,
        );
    }
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_synapse_step(
    _gpu: &super::GpuContext,
    neurons: &mut NeuronArrays,
    synapses: &mut SynapseArrays,
    time: f32,
    dt: f32,
    psc_scale: f32,
    step_index: u32,
) {
    let fired = neurons.fired_indices();
    if !fired.is_empty() {
        crate::spike_propagation::propagate_spikes(neurons, synapses, &fired, time, psc_scale);
        crate::stdp::update_stdp(neurons, synapses, &fired, time, dt);
        for &idx in &fired {
            neurons.last_fired_step[idx] = step_index;
        }
    }

    for i in 0..synapses.n_synapses {
        synapses.update_cleft(i, time, dt);
    }
}
