//! GPU dispatch for 4-compartment calcium dynamics.
//!
//! Integrates calcium concentrations across microdomain, cytoplasmic, ER, and
//! mitochondrial compartments. Includes IP3R/RyR release from ER, SERCA pump,
//! MCU mitochondrial uptake, and plasma membrane PMCA/NCX export.

use crate::neuron_arrays::NeuronArrays;

#[cfg(target_os = "macos")]
use super::state::MetalNeuronState;
#[cfg(target_os = "macos")]
use super::GpuContext;
#[cfg(target_os = "macos")]
use metal::CommandBufferRef;

/// Params struct matching the Metal shader's `Params` constant buffer.
#[repr(C)]
struct CalciumParams {
    neuron_count: u32,
    dt: f32,
}

/// Update calcium dynamics on GPU (macOS/Metal).
///
/// Dispatches the `calcium_ode` kernel which integrates the 4-compartment
/// calcium model using forward-Euler. IP3 concentration (from the second
/// messenger system) gates IP3R release from ER.
///
/// After dispatch, reads back ca_cytoplasmic, ca_er, ca_mitochondrial,
/// and ca_microdomain from GPU shared memory.
#[cfg(target_os = "macos")]
pub fn dispatch_calcium_dynamics(gpu: &GpuContext, neurons: &MetalNeuronState, dt: f32) {
    let cmd = gpu.new_command_buffer();
    encode_calcium_dynamics(gpu, &cmd, neurons, dt);
    gpu.commit_and_wait(cmd);
}

#[cfg(target_os = "macos")]
pub fn encode_calcium_dynamics(
    gpu: &GpuContext,
    cmd: &CommandBufferRef,
    neurons: &MetalNeuronState,
    dt: f32,
) {
    let n = neurons.count as u64;
    if n == 0 {
        return;
    }

    let params = CalciumParams {
        neuron_count: neurons.count as u32,
        dt,
    };
    let param_bytes = unsafe {
        std::slice::from_raw_parts(
            &params as *const CalciumParams as *const u8,
            std::mem::size_of::<CalciumParams>(),
        )
    };

    gpu.encode_1d(
        cmd,
        &gpu.pipelines.calcium_ode,
        &[
            (&neurons.ca_cytoplasmic, 0),   // buffer(0)
            (&neurons.ca_er, 0),            // buffer(1)
            (&neurons.ca_mitochondrial, 0), // buffer(2)
            (&neurons.ca_microdomain, 0),   // buffer(3)
            (&neurons.ip3, 0),              // buffer(4)
            (&neurons.alive, 0),            // buffer(5)
        ],
        Some((param_bytes, 6)), // buffer(6): Params
        n,
    );
}

/// CPU fallback for calcium dynamics (non-macOS platforms).
#[cfg(not(target_os = "macos"))]
pub fn dispatch_calcium_dynamics(_gpu: &super::GpuContext, neurons: &mut NeuronArrays, dt: f32) {
    cpu_calcium_dynamics(neurons, dt);
}

/// CPU reference implementation of 4-compartment calcium dynamics.
///
/// Mirrors the Metal shader logic: microdomain diffusion, IP3R/RyR ER release,
/// SERCA pump, MCU mitochondrial uptake, and PMCA/NCX plasma membrane export.
pub fn cpu_calcium_dynamics(neurons: &mut NeuronArrays, dt: f32) {
    use crate::constants::*;

    for i in 0..neurons.count {
        if neurons.alive[i] == 0 {
            continue;
        }

        let mut micro = neurons.ca_microdomain[i];
        let mut cyto = neurons.ca_cytoplasmic[i];
        let mut er = neurons.ca_er[i];
        let mut mito = neurons.ca_mitochondrial[i];
        let ip3_c = neurons.ip3[i];

        // 1. Microdomain -> cytoplasm diffusion (exponential relaxation)
        let micro_diff = micro - cyto;
        let diffusion_flux = micro_diff * (1.0 - (-dt / MICRODOMAIN_DIFFUSION_TAU).exp());
        micro -= diffusion_flux;
        cyto += diffusion_flux / CA_BUFFER_CAPACITY;

        // 2. IP3R: ER -> cytoplasm
        let ip3_gate = hill(ip3_c, IP3R_K_IP3, IP3R_HILL_IP3);
        let ca_act = hill(cyto, IP3R_K_ACT, IP3R_HILL_ACT);
        let ca_inh = hill(cyto, IP3R_K_INH, IP3R_HILL_INH);
        let gradient = if er > 0.0 {
            ((er - cyto) / er).max(0.0)
        } else {
            0.0
        };
        let ip3r_flux = IP3R_VMAX * ip3_gate * ca_act * (1.0 - ca_inh) * gradient * dt;

        // 3. RyR: CICR
        let ryr_act = hill(cyto, RYR_K_ACT, RYR_HILL);
        let ryr_inh = (1.0 - hill(cyto, RYR_K_INH, RYR_HILL_INH)).max(0.0);
        let ryr_flux = RYR_VMAX * ryr_act * ryr_inh * gradient * dt;

        // 4. SERCA: cytoplasm -> ER
        let serca_flux = SERCA_VMAX * hill(cyto, SERCA_KM, SERCA_HILL) * dt;

        // 5. ER leak
        let er_leak = er * 0.001 * dt;

        // 6. Net ER exchange
        let net_er = ip3r_flux + ryr_flux - serca_flux + er_leak;
        cyto += net_er / CA_BUFFER_CAPACITY;
        er -= net_er;

        // 7. MCU: cytoplasm -> mitochondria
        let mcu_flux = MCU_VMAX * hill(cyto, MCU_KM, MCU_HILL) * dt;
        cyto -= mcu_flux / CA_BUFFER_CAPACITY;
        mito += mcu_flux;

        // 8. Mitochondrial release
        let mito_release = mito * 0.02 * dt;
        mito -= mito_release;
        cyto += mito_release / CA_BUFFER_CAPACITY;

        // 9. Plasma membrane: PMCA + NCX export, passive leak in
        let pmca = PMCA_VMAX * hill(cyto, PMCA_KM, 1.0) * dt;
        let ncx = NCX_VMAX * hill(cyto, NCX_KM, 1.0) * dt;
        let leak_in = CA_PASSIVE_LEAK_RATE * dt;
        cyto -= (pmca + ncx - leak_in) / CA_BUFFER_CAPACITY;

        // 10. Clamp all >= 0
        neurons.ca_microdomain[i] = micro.max(0.0);
        neurons.ca_cytoplasmic[i] = cyto.max(0.0);
        neurons.ca_er[i] = er.max(0.0);
        neurons.ca_mitochondrial[i] = mito.max(0.0);
    }
}
