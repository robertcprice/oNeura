use crate::neuron_arrays::NeuronArrays;

#[cfg(target_os = "macos")]
use super::state::MetalNeuronState;
#[cfg(target_os = "macos")]
use super::GpuContext;

#[repr(C)]
struct GeneExpressionParams {
    neuron_count: u32,
    dt: f32,
}

#[cfg(target_os = "macos")]
pub fn dispatch_gene_expression(gpu: &GpuContext, neurons: &MetalNeuronState, dt: f32) {
    let n = neurons.count as u64;
    if n == 0 {
        return;
    }

    let params = GeneExpressionParams {
        neuron_count: neurons.count as u32,
        dt,
    };
    let param_bytes = unsafe {
        std::slice::from_raw_parts(
            &params as *const GeneExpressionParams as *const u8,
            std::mem::size_of::<GeneExpressionParams>(),
        )
    };

    gpu.dispatch_1d(
        &gpu.pipelines.gene_expression,
        &[
            (&neurons.ca_cytoplasmic, 0),
            (&neurons.creb_p, 0),
            (&neurons.cfos_level, 0),
            (&neurons.arc_level, 0),
            (&neurons.bdnf_level, 0),
            (&neurons.alive, 0),
        ],
        Some((param_bytes, 6)),
        n,
    );
}

#[cfg(not(target_os = "macos"))]
pub fn dispatch_gene_expression(_gpu: &super::GpuContext, neurons: &mut NeuronArrays, dt: f32) {
    crate::gene_expression::update_gene_expression(neurons, dt);
}
