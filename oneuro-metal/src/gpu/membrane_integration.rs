//! GPU dispatch for membrane voltage integration via forward-Euler.
//!
//! Sums ionic currents from all 8 channel types (Na_v, K_v, K_leak, Ca_v,
//! AMPA, NMDA, GABA-A, nAChR), integrates dV/dt = (-I_ion + I_ext) / C_m,
//! and performs spike detection with refractory period enforcement.
//!
//! The `conductance_scale` array (Vec<[f32; 8]>) must be transposed from AoS
//! to SoA before dispatch: 8 contiguous float[N] arrays packed as
//! `conductance_scale[channel * N + neuron]`.

#[cfg(target_os = "macos")]
use super::state::MetalNeuronState;
#[cfg(target_os = "macos")]
use super::GpuContext;
use crate::neuron_arrays::NeuronArrays;
#[cfg(target_os = "macos")]
use metal::CommandBufferRef;

/// Params struct matching the Metal shader's `Params` constant buffer.
#[repr(C)]
struct MembraneParams {
    neuron_count: u32,
    dt: f32,
    global_bias: f32,
    membrane_capacitance_uf: f32,
    spike_threshold_mv: f32,
    refractory_period_ms: f32,
}

/// Update membrane voltage on GPU (macOS/Metal).
///
/// Dispatches the `membrane_euler` compute kernel which:
/// 1. Computes ionic current from all 8 channel types (using gating variables
///    and open fractions from prior HH/receptor-binding steps)
/// 2. Euler-integrates voltage with clamping to [-100, 60] mV
/// 3. Detects spikes (threshold crossing at -20 mV) with refractory period (2 ms)
///
/// Conductance scales are transposed from the NeuronArrays AoS format into the
/// SoA layout expected by the shader.
#[cfg(target_os = "macos")]
pub fn dispatch_membrane_integration(
    gpu: &GpuContext,
    neurons: &MetalNeuronState,
    dt: f32,
    global_bias: f32,
    membrane_capacitance_uf: f32,
    spike_threshold_mv: f32,
    refractory_period_ms: f32,
) {
    let cmd = gpu.new_command_buffer();
    encode_membrane_integration(
        gpu,
        &cmd,
        neurons,
        dt,
        global_bias,
        membrane_capacitance_uf,
        spike_threshold_mv,
        refractory_period_ms,
    );
    gpu.commit_and_wait(cmd);
}

#[cfg(target_os = "macos")]
pub fn encode_membrane_integration(
    gpu: &GpuContext,
    cmd: &CommandBufferRef,
    neurons: &MetalNeuronState,
    dt: f32,
    global_bias: f32,
    membrane_capacitance_uf: f32,
    spike_threshold_mv: f32,
    refractory_period_ms: f32,
) {
    let n = neurons.count as u64;
    if n == 0 {
        return;
    }

    let params = MembraneParams {
        neuron_count: neurons.count as u32,
        dt,
        global_bias,
        membrane_capacitance_uf,
        spike_threshold_mv,
        refractory_period_ms,
    };
    let param_bytes = unsafe {
        std::slice::from_raw_parts(
            &params as *const MembraneParams as *const u8,
            std::mem::size_of::<MembraneParams>(),
        )
    };

    gpu.encode_1d(
        cmd,
        &gpu.pipelines.membrane_euler,
        &[
            (&neurons.voltage, 0),           // buffer(0)
            (&neurons.prev_voltage, 0),      // buffer(1)
            (&neurons.fired, 0),             // buffer(2)
            (&neurons.refractory_timer, 0),  // buffer(3)
            (&neurons.spike_count, 0),       // buffer(4)
            (&neurons.nav_m, 0),             // buffer(5)
            (&neurons.nav_h, 0),             // buffer(6)
            (&neurons.kv_n, 0),              // buffer(7)
            (&neurons.cav_m, 0),             // buffer(8)
            (&neurons.cav_h, 0),             // buffer(9)
            (&neurons.ampa_open, 0),         // buffer(10)
            (&neurons.nmda_open, 0),         // buffer(11)
            (&neurons.gabaa_open, 0),        // buffer(12)
            (&neurons.nachr_open, 0),        // buffer(13)
            (&neurons.conductance_scale, 0), // buffer(14)
            (&neurons.external_current, 0),  // buffer(15)
            (&neurons.synaptic_current, 0),  // buffer(16)
            (&neurons.alive, 0),             // buffer(17)
            (&neurons.excitability_bias, 0), // buffer(18)
        ],
        Some((param_bytes, 19)), // buffer(19): Params
        n,
    );
}

/// CPU fallback for membrane integration (non-macOS platforms).
#[cfg(not(target_os = "macos"))]
pub fn dispatch_membrane_integration(
    _gpu: &super::GpuContext,
    neurons: &mut NeuronArrays,
    dt: f32,
    global_bias: f32,
    membrane_capacitance_uf: f32,
    spike_threshold_mv: f32,
    refractory_period_ms: f32,
) {
    cpu_membrane_integration(
        neurons,
        dt,
        global_bias,
        membrane_capacitance_uf,
        spike_threshold_mv,
        refractory_period_ms,
    );
}

/// CPU reference implementation of membrane voltage integration.
///
/// Computes ionic current from all 8 channel types, Euler-integrates voltage,
/// and performs spike detection with refractory period enforcement.
pub fn cpu_membrane_integration(
    neurons: &mut NeuronArrays,
    dt: f32,
    global_bias: f32,
    membrane_capacitance_uf: f32,
    spike_threshold_mv: f32,
    refractory_period_ms: f32,
) {
    use crate::constants::*;

    for i in 0..neurons.count {
        if neurons.alive[i] == 0 {
            continue;
        }

        let v = neurons.voltage[i];
        neurons.prev_voltage[i] = v;

        // Conductance scales for this neuron
        let cs = &neurons.conductance_scale[i];

        // Gating products
        let m3h = neurons.nav_m[i].powi(3) * neurons.nav_h[i]; // Na_v: m^3 h
        let n4 = neurons.kv_n[i].powi(4); // K_v:  n^4
        let m2h_ca = neurons.cav_m[i].powi(2) * neurons.cav_h[i]; // Ca_v: m^2 h

        // Voltage-gated channel currents
        let i_nav = NAV_G_MAX * cs[0] * m3h * (v - NAV_E_REV);
        let i_kv = KV_G_MAX * cs[1] * n4 * (v - KV_E_REV);
        let i_kleak = KLEAK_G_MAX * cs[2] * (v - KLEAK_E_REV);
        let i_cav = CAV_G_MAX * cs[3] * m2h_ca * (v - CAV_E_REV);

        // Mg2+ block for NMDA
        let mg_block = 1.0 / (1.0 + NMDA_MG_CONC_MM * (-0.062 * v).exp() / 3.57);

        // Ligand-gated channel currents
        let i_ampa = AMPA_G_MAX * cs[5] * neurons.ampa_open[i] * (v - AMPA_E_REV);
        let i_nmda = NMDA_G_MAX * cs[4] * neurons.nmda_open[i] * mg_block * (v - NMDA_E_REV);
        let i_gabaa = GABAA_G_MAX * cs[6] * neurons.gabaa_open[i] * (v - GABAA_E_REV);
        let i_nachr = NACHR_G_MAX * cs[7] * neurons.nachr_open[i] * (v - NACHR_E_REV);

        // Total ionic current
        let i_total = i_nav + i_kv + i_kleak + i_cav + i_ampa + i_nmda + i_gabaa + i_nachr;

        // External current
        let i_ext = neurons.external_current[i]
            + neurons.synaptic_current[i]
            + neurons.excitability_bias[i]
            + global_bias;

        // Euler integration: dV/dt = (-I_ion + I_ext) / C_m
        let dv = (-i_total + i_ext) / membrane_capacitance_uf.max(0.1) * dt;
        let v_new = (v + dv).clamp(VOLTAGE_MIN, VOLTAGE_MAX);

        // Refractory timer countdown
        let mut ref_timer = neurons.refractory_timer[i] - dt;

        // Spike detection: threshold crossing at -20 mV while not refractory
        neurons.fired[i] = 0;
        if v < spike_threshold_mv && v_new >= spike_threshold_mv && ref_timer <= 0.0 {
            neurons.fired[i] = 1;
            ref_timer = refractory_period_ms.max(dt);
            neurons.spike_count[i] += 1;
        }

        neurons.refractory_timer[i] = ref_timer.max(0.0);
        neurons.voltage[i] = v_new;
        neurons.synaptic_current[i] = 0.0;
    }
}
