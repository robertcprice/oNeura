//! GPU dispatch for synaptic cleft neurotransmitter dynamics.
//!
//! Applies enzymatic degradation (first-order decay with NT-specific half-life),
//! diffusion out of the cleft, and presynaptic reuptake. Operates on per-synapse
//! data (not per-neuron) from `SynapseArrays`.
//!
//! The Metal shader processes one thread per synapse. The `nt_type` field (u8)
//! is converted to f32 for the shader, which casts back to uint internally.

use crate::synapse_arrays::SynapseArrays;

#[cfg(target_os = "macos")]
use super::GpuContext;

/// Params struct matching the Metal shader's `Params` constant buffer.
#[repr(C)]
struct CleftParams {
    synapse_count: u32,
    dt: f32,
}

/// Update synaptic cleft dynamics on GPU (macOS/Metal).
///
/// Dispatches the `cleft_dynamics` kernel for all synapses. Each synapse's
/// cleft concentration decays via:
/// 1. Enzymatic degradation (NT-specific half-life -> first-order rate)
/// 2. Diffusion out of cleft (0.5 fraction/ms)
/// 3. Presynaptic reuptake (0.1 fraction/ms)
///
/// Note: This only handles the decay side of cleft dynamics. NT release from
/// vesicle fusion (driven by presynaptic spikes) is handled separately on the
/// CPU via `SynapseArrays::presynaptic_spike` and `update_cleft`, since those
/// involve the variable-length delay buffer which is not GPU-friendly.
#[cfg(target_os = "macos")]
pub fn dispatch_synapse_cleft(gpu: &GpuContext, synapses: &mut SynapseArrays, dt: f32) {
    let n = synapses.n_synapses as u64;
    if n == 0 {
        return;
    }

    let count = synapses.n_synapses;

    // Cleft concentration buffer (read/write)
    let buf_cleft = gpu.buffer_from_slice(&synapses.cleft_concentration);

    // NT type as f32 (shader expects float, casts to uint internally)
    let nt_type_f: Vec<f32> = synapses.nt_type.iter().map(|&t| t as f32).collect();
    let buf_nt_type = gpu.buffer_from_slice(&nt_type_f);

    // Per-synapse alive/active flag: synapses with cleft_concentration > 0 are active.
    // The shader checks alive < 0.5, so we set 1.0 for all valid synapses.
    // (Dead synapses would be pruned from the CSR structure entirely.)
    let alive_f: Vec<f32> = vec![1.0f32; count];
    let buf_alive = gpu.buffer_from_slice(&alive_f);

    let params = CleftParams {
        synapse_count: count as u32,
        dt,
    };
    let param_bytes = unsafe {
        std::slice::from_raw_parts(
            &params as *const CleftParams as *const u8,
            std::mem::size_of::<CleftParams>(),
        )
    };

    gpu.dispatch_1d(
        &gpu.pipelines.cleft_dynamics,
        &[
            (&buf_cleft, 0),   // buffer(0): cleft_concentration (read/write)
            (&buf_nt_type, 0), // buffer(1): nt_type as f32 (read-only)
            (&buf_alive, 0),   // buffer(2): alive flag (read-only)
        ],
        Some((param_bytes, 3)), // buffer(3): Params
        n,
    );

    // Read back modified cleft concentrations from shared GPU memory
    unsafe {
        let ptr = buf_cleft.contents() as *const f32;
        std::ptr::copy_nonoverlapping(ptr, synapses.cleft_concentration.as_mut_ptr(), count);
    }
}

/// CPU fallback for synaptic cleft dynamics (non-macOS platforms).
#[cfg(not(target_os = "macos"))]
pub fn dispatch_synapse_cleft(_gpu: &super::GpuContext, synapses: &mut SynapseArrays, dt: f32) {
    cpu_synapse_cleft(synapses, dt);
}

/// CPU reference implementation of synaptic cleft dynamics.
///
/// Applies the same enzymatic degradation, diffusion, and reuptake as the
/// Metal shader. Half-life values per NT type (ms):
/// - Dopamine: 500, Serotonin: 300, NE: 400, ACh: 2, GABA: 100, Glutamate: 10
///
/// Note: These half-life values match the Metal shader constants but differ
/// slightly from the `NTType::half_life_ms()` values in types.rs. The shader
/// values are used for consistency between GPU and CPU paths.
pub fn cpu_synapse_cleft(synapses: &mut SynapseArrays, dt: f32) {
    const LN2: f32 = 0.693_147;
    const DIFFUSION_RATE: f32 = 0.5;
    const REUPTAKE_RATE: f32 = 0.1;

    // Half-lives matching the Metal shader constant array
    const HALF_LIVES: [f32; 6] = [
        500.0, // dopamine
        300.0, // serotonin
        400.0, // norepinephrine
        2.0,   // acetylcholine
        100.0, // GABA
        10.0,  // glutamate
    ];

    for i in 0..synapses.n_synapses {
        let conc = &mut synapses.cleft_concentration[i];
        if *conc <= 0.0 {
            *conc = 0.0;
            continue;
        }

        // NT-specific enzymatic degradation
        let nt = (synapses.nt_type[i] as usize).min(5);
        let half_life = HALF_LIVES[nt];
        let decay_rate = LN2 / half_life;
        *conc -= *conc * decay_rate * dt;

        // Diffusion out of cleft
        *conc *= 1.0 - DIFFUSION_RATE * dt;

        // Presynaptic reuptake
        *conc *= 1.0 - REUPTAKE_RATE * dt;

        // Clamp non-negative
        *conc = conc.max(0.0);
    }
}
