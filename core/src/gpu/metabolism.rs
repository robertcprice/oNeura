#[cfg(target_os = "macos")]
use super::state::MetalNeuronState;
#[cfg(target_os = "macos")]
use super::GpuContext;

#[repr(C)]
struct MetabolismParams {
    neuron_count: u32,
    dt: f32,
}

#[cfg(target_os = "macos")]
pub fn dispatch_metabolism(gpu: &GpuContext, neurons: &MetalNeuronState, dt: f32) {
    let n = neurons.count as u64;
    if n == 0 {
        return;
    }

    let params = MetabolismParams {
        neuron_count: neurons.count as u32,
        dt,
    };
    let param_bytes = unsafe {
        std::slice::from_raw_parts(
            &params as *const MetabolismParams as *const u8,
            std::mem::size_of::<MetabolismParams>(),
        )
    };

    gpu.dispatch_1d(
        &gpu.pipelines.metabolism,
        &[
            (&neurons.atp, 0),
            (&neurons.adp, 0),
            (&neurons.glucose, 0),
            (&neurons.oxygen, 0),
            (&neurons.energy, 0),
            (&neurons.excitability_bias, 0),
            (&neurons.fired, 0),
            (&neurons.external_current, 0),
            (&neurons.alive, 0),
        ],
        Some((param_bytes, 9)),
        n,
    );
}

#[cfg(not(target_os = "macos"))]
pub fn dispatch_metabolism(_gpu: &super::GpuContext, neurons: &mut NeuronArrays, dt: f32) {
    crate::metabolism::update_metabolism(neurons, dt);
}
