#[cfg(target_os = "macos")]
use super::state::MetalNeuronState;
#[cfg(target_os = "macos")]
use super::GpuContext;

#[repr(C)]
struct PharmacologyParams {
    neuron_count: u32,
    drug_type: u32,
    effect: f32,
}

#[cfg(target_os = "macos")]
pub fn dispatch_pharmacology(
    gpu: &GpuContext,
    neurons: &MetalNeuronState,
    drug_type: u8,
    effect: f32,
) {
    let n = neurons.count as u64;
    if n == 0 || effect < 0.001 {
        return;
    }

    let params = PharmacologyParams {
        neuron_count: neurons.count as u32,
        drug_type: drug_type as u32,
        effect,
    };
    let param_bytes = unsafe {
        std::slice::from_raw_parts(
            &params as *const PharmacologyParams as *const u8,
            std::mem::size_of::<PharmacologyParams>(),
        )
    };

    gpu.dispatch_1d(
        &gpu.pipelines.pharmacology,
        &[
            (&neurons.conductance_scale, 0),
            (&neurons.nt_conc, 0),
            (&neurons.external_current, 0),
            (&neurons.alive, 0),
        ],
        Some((param_bytes, 4)),
        n,
    );
}
