#[cfg(target_os = "macos")]
use super::state::{MetalGliaState, MetalNeuronState, MetalSynapseState};
#[cfg(target_os = "macos")]
use super::GpuContext;

#[repr(C)]
struct GliaNeuronParams {
    neuron_count: u32,
    dt_ms: f32,
}

#[repr(C)]
struct GliaPruneParams {
    synapse_count: u32,
}

#[cfg(target_os = "macos")]
pub fn dispatch_glia_neuron_step(
    gpu: &GpuContext,
    glia: &MetalGliaState,
    neurons: &MetalNeuronState,
    dt_ms: f32,
) {
    let n = neurons.count as u64;
    if n == 0 {
        return;
    }

    let params = GliaNeuronParams {
        neuron_count: neurons.count as u32,
        dt_ms,
    };
    let param_bytes = unsafe {
        std::slice::from_raw_parts(
            &params as *const GliaNeuronParams as *const u8,
            std::mem::size_of::<GliaNeuronParams>(),
        )
    };

    gpu.dispatch_1d(
        &gpu.pipelines.glia_neuron_step,
        &[
            (&glia.astrocyte_uptake, 0),
            (&glia.astrocyte_lactate, 0),
            (&glia.myelin_integrity, 0),
            (&glia.microglia_activation, 0),
            (&glia.damage_signal, 0),
            (&neurons.nt_conc, 0),
            (&neurons.glucose, 0),
            (&neurons.atp, 0),
            (&neurons.voltage, 0),
            (&neurons.alive, 0),
        ],
        Some((param_bytes, 10)),
        n,
    );
}

#[cfg(target_os = "macos")]
pub fn dispatch_glia_prune_synapses(
    gpu: &GpuContext,
    glia: &MetalGliaState,
    synapses: &MetalSynapseState,
) {
    let n = synapses.count as u64;
    if n == 0 {
        return;
    }

    let params = GliaPruneParams {
        synapse_count: synapses.count as u32,
    };
    let param_bytes = unsafe {
        std::slice::from_raw_parts(
            &params as *const GliaPruneParams as *const u8,
            std::mem::size_of::<GliaPruneParams>(),
        )
    };

    gpu.dispatch_1d(
        &gpu.pipelines.glia_prune_synapses,
        &[
            (&synapses.col_indices, 0),
            (&synapses.weight, 0),
            (&synapses.strength, 0),
            (&synapses.ampa_receptors, 0),
            (&synapses.nmda_receptors, 0),
            (&synapses.gabaa_receptors, 0),
            (&glia.microglia_activation, 0),
        ],
        Some((param_bytes, 7)),
        n,
    );
}
