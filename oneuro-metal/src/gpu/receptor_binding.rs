//! GPU dispatch for receptor binding via Hill-equation ligand activation.
//!
//! Converts ambient neurotransmitter concentrations into ionotropic receptor
//! open fractions:
//! - Glutamate -> AMPA (EC50=500 nM, n=1.3)
//! - Glutamate -> NMDA (EC50=2400 nM, n=1.5)
//! - GABA      -> GABA-A (EC50=200 nM, n=2.0)
//! - ACh       -> nAChR (EC50=30 nM, n=1.8)
//!
//! The `nt_conc` field (Vec<[f32; 6]>) is transposed from AoS to SoA before
//! dispatch: 6 contiguous float[N] arrays packed as `nt[nt_index * N + neuron]`.

#[cfg(target_os = "macos")]
use super::state::MetalNeuronState;
#[cfg(target_os = "macos")]
use super::GpuContext;
use crate::neuron_arrays::NeuronArrays;
use crate::types::NTType;
#[cfg(target_os = "macos")]
use metal::CommandBufferRef;

/// Params struct matching the Metal shader's `Params` constant buffer.
#[repr(C)]
struct ReceptorParams {
    neuron_count: u32,
}

/// Update receptor open fractions on GPU (macOS/Metal).
///
/// Dispatches the `hill_binding` kernel which computes receptor activation
/// from local NT concentrations via Hill equations. NMDA Mg2+ block is NOT
/// applied here -- it is handled in the membrane integration shader.
#[cfg(target_os = "macos")]
pub fn dispatch_receptor_binding(gpu: &GpuContext, neurons: &MetalNeuronState) {
    let cmd = gpu.new_command_buffer();
    encode_receptor_binding(gpu, &cmd, neurons);
    gpu.commit_and_wait(cmd);
}

#[cfg(target_os = "macos")]
pub fn encode_receptor_binding(
    gpu: &GpuContext,
    cmd: &CommandBufferRef,
    neurons: &MetalNeuronState,
) {
    let n = neurons.count as u64;
    if n == 0 {
        return;
    }

    let params = ReceptorParams {
        neuron_count: neurons.count as u32,
    };
    let param_bytes = unsafe {
        std::slice::from_raw_parts(
            &params as *const ReceptorParams as *const u8,
            std::mem::size_of::<ReceptorParams>(),
        )
    };

    gpu.encode_1d(
        cmd,
        &gpu.pipelines.hill_binding,
        &[
            (&neurons.nt_conc, 0),    // buffer(0): nt_conc AoS
            (&neurons.ampa_open, 0),  // buffer(1): ampa_open
            (&neurons.nmda_open, 0),  // buffer(2): nmda_open
            (&neurons.gabaa_open, 0), // buffer(3): gabaa_open
            (&neurons.nachr_open, 0), // buffer(4): nachr_open
            (&neurons.alive, 0),      // buffer(5): alive
        ],
        Some((param_bytes, 6)), // buffer(6): Params
        n,
    );
}

/// CPU fallback for receptor binding (non-macOS platforms).
#[cfg(not(target_os = "macos"))]
pub fn dispatch_receptor_binding(_gpu: &super::GpuContext, neurons: &mut NeuronArrays) {
    cpu_receptor_binding(neurons);
}

/// CPU reference implementation of receptor binding via Hill equations.
///
/// Computes open fractions for AMPA, NMDA, GABA-A, and nAChR receptors
/// from local neurotransmitter concentrations.
pub fn cpu_receptor_binding(neurons: &mut NeuronArrays) {
    use crate::constants::hill;

    // EC50 and Hill coefficients matching the Metal shader constants
    const AMPA_EC50: f32 = 500.0;
    const AMPA_HILL: f32 = 1.3;
    const NMDA_EC50: f32 = 2400.0;
    const NMDA_HILL: f32 = 1.5;
    const GABAA_EC50: f32 = 200.0;
    const GABAA_HILL: f32 = 2.0;
    const NACHR_EC50: f32 = 30.0;
    const NACHR_HILL: f32 = 1.8;
    const REST_GLU: f32 = 500.0;
    const REST_GABA: f32 = 200.0;
    const REST_ACH: f32 = 50.0;

    for i in 0..neurons.count {
        if neurons.alive[i] == 0 {
            continue;
        }

        let nt = &neurons.nt_conc[i];
        let glutamate = (nt[NTType::Glutamate.index()] - REST_GLU).max(0.0);
        let gaba = (nt[NTType::GABA.index()] - REST_GABA).max(0.0);
        let ach = (nt[NTType::Acetylcholine.index()] - REST_ACH).max(0.0);

        neurons.ampa_open[i] = hill(glutamate, AMPA_EC50, AMPA_HILL);
        neurons.nmda_open[i] = hill(glutamate, NMDA_EC50, NMDA_HILL);
        neurons.gabaa_open[i] = hill(gaba, GABAA_EC50, GABAA_HILL);
        neurons.nachr_open[i] = hill(ach, NACHR_EC50, NACHR_HILL);
    }
}
