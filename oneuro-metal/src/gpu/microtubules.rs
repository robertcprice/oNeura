use crate::neuron_arrays::NeuronArrays;

#[cfg(target_os = "macos")]
use super::state::MetalNeuronState;
#[cfg(target_os = "macos")]
use super::GpuContext;

#[repr(C)]
struct MicrotubulesParams {
    neuron_count: u32,
    dt: f32,
}

#[cfg(target_os = "macos")]
pub fn dispatch_microtubules(gpu: &GpuContext, neurons: &MetalNeuronState, dt: f32) {
    let n = neurons.count as u64;
    if n == 0 {
        return;
    }

    let params = MicrotubulesParams {
        neuron_count: neurons.count as u32,
        dt,
    };
    let param_bytes = unsafe {
        std::slice::from_raw_parts(
            &params as *const MicrotubulesParams as *const u8,
            std::mem::size_of::<MicrotubulesParams>(),
        )
    };

    gpu.dispatch_1d(
        &gpu.pipelines.microtubules,
        &[
            (&neurons.mt_coherence, 0),
            (&neurons.orch_or_events, 0),
            (&neurons.ca_cytoplasmic, 0),
            (&neurons.atp, 0),
            (&neurons.fired, 0),
            (&neurons.alive, 0),
        ],
        Some((param_bytes, 6)),
        n,
    );
}

#[cfg(not(target_os = "macos"))]
pub fn dispatch_microtubules(_gpu: &super::GpuContext, neurons: &mut NeuronArrays, dt: f32) {
    crate::microtubules::update_microtubules(neurons, dt);
}
