//! GPU dispatch for Hodgkin-Huxley gating variable integration.
//!
//! Advances Na_v (m, h), K_v (n), and Ca_v (m, h) gating variables using
//! forward-Euler integration on Metal GPU. One thread per neuron, SoA layout.

use crate::neuron_arrays::NeuronArrays;

#[cfg(target_os = "macos")]
use super::state::MetalNeuronState;
#[cfg(target_os = "macos")]
use super::GpuContext;
#[cfg(target_os = "macos")]
use metal::CommandBufferRef;

/// Params struct matching the Metal shader's `Params` constant buffer.
#[repr(C)]
struct HHParams {
    neuron_count: u32,
    dt: f32,
}

/// Update HH gating variables on GPU (macOS/Metal).
///
/// Dispatches the `hh_gating` compute kernel which integrates all five gating
/// variables (nav_m, nav_h, kv_n, cav_m, cav_h) via forward-Euler with
/// clamping to [0, 1]. Voltage is read-only; gating variables are read/write.
#[cfg(target_os = "macos")]
pub fn dispatch_hh_gating(gpu: &GpuContext, neurons: &MetalNeuronState, dt: f32) {
    let cmd = gpu.new_command_buffer();
    encode_hh_gating(gpu, &cmd, neurons, dt);
    gpu.commit_and_wait(cmd);
}

#[cfg(target_os = "macos")]
pub fn encode_hh_gating(
    gpu: &GpuContext,
    cmd: &CommandBufferRef,
    neurons: &MetalNeuronState,
    dt: f32,
) {
    let n = neurons.count as u64;
    if n == 0 {
        return;
    }

    let params = HHParams {
        neuron_count: neurons.count as u32,
        dt,
    };
    let param_bytes = unsafe {
        std::slice::from_raw_parts(
            &params as *const HHParams as *const u8,
            std::mem::size_of::<HHParams>(),
        )
    };

    gpu.encode_1d(
        cmd,
        &gpu.pipelines.hh_gating,
        &[
            (&neurons.voltage, 0), // buffer(0): voltage (read-only)
            (&neurons.nav_m, 0),   // buffer(1): nav_m (read/write)
            (&neurons.nav_h, 0),   // buffer(2): nav_h (read/write)
            (&neurons.kv_n, 0),    // buffer(3): kv_n (read/write)
            (&neurons.cav_m, 0),   // buffer(4): cav_m (read/write)
            (&neurons.cav_h, 0),   // buffer(5): cav_h (read/write)
            (&neurons.alive, 0),   // buffer(6): alive mask
        ],
        Some((param_bytes, 7)), // buffer(7): Params
        n,
    );
}

/// CPU fallback for HH gating (non-macOS platforms).
#[cfg(not(target_os = "macos"))]
pub fn dispatch_hh_gating(_gpu: &super::GpuContext, neurons: &mut NeuronArrays, dt: f32) {
    cpu_hh_gating(neurons, dt);
}

/// CPU reference implementation of HH gating variable integration.
///
/// Uses the same alpha/beta rate functions as the Metal shader. This is used
/// as the fallback on non-macOS platforms and can serve as a correctness
/// reference for GPU output verification.
pub fn cpu_hh_gating(neurons: &mut NeuronArrays, dt: f32) {
    use crate::neuron_arrays::*;
    for i in 0..neurons.count {
        if neurons.alive[i] == 0 {
            continue;
        }
        let v = neurons.voltage[i];

        // Na_v activation (m)
        let am = alpha_m(v);
        let bm = beta_m(v);
        neurons.nav_m[i] += dt * (am * (1.0 - neurons.nav_m[i]) - bm * neurons.nav_m[i]);
        neurons.nav_m[i] = neurons.nav_m[i].clamp(0.0, 1.0);

        // Na_v inactivation (h)
        let ah = alpha_h(v);
        let bh = beta_h(v);
        neurons.nav_h[i] += dt * (ah * (1.0 - neurons.nav_h[i]) - bh * neurons.nav_h[i]);
        neurons.nav_h[i] = neurons.nav_h[i].clamp(0.0, 1.0);

        // K_v activation (n)
        let an = alpha_n(v);
        let bn = beta_n(v);
        neurons.kv_n[i] += dt * (an * (1.0 - neurons.kv_n[i]) - bn * neurons.kv_n[i]);
        neurons.kv_n[i] = neurons.kv_n[i].clamp(0.0, 1.0);

        // Ca_v activation (m)
        let amc = alpha_m_ca(v);
        let bmc = beta_m_ca(v);
        neurons.cav_m[i] += dt * (amc * (1.0 - neurons.cav_m[i]) - bmc * neurons.cav_m[i]);
        neurons.cav_m[i] = neurons.cav_m[i].clamp(0.0, 1.0);

        // Ca_v inactivation (h)
        let ahc = alpha_h_ca(v);
        let bhc = beta_h_ca(v);
        neurons.cav_h[i] += dt * (ahc * (1.0 - neurons.cav_h[i]) - bhc * neurons.cav_h[i]);
        neurons.cav_h[i] = neurons.cav_h[i].clamp(0.0, 1.0);
    }
}
