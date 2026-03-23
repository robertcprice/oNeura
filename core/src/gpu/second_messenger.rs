//! GPU dispatch for intracellular second messenger signaling cascades.
//!
//! The most complex shader in the pipeline. Integrates:
//! - G-protein activation (Gs, Gi, Gq) from metabotropic receptor inputs
//! - cAMP / PKA pathway (adenylyl cyclase + PDE)
//! - PLC / IP3 / DAG pathway
//! - PKC (DAG + Ca2+ gated)
//! - CaMKII (Ca2+-dependent bistable switch with autophosphorylation)
//! - ERK (MAPK cross-talk from PKA, PKC, CaMKII)
//! - Phosphorylation of downstream targets (AMPA, Kv, Cav, CREB)
//!
//! G-protein inputs are derived directly in the shader from per-neuron
//! neurotransmitter concentrations, so the CPU no longer computes them on
//! every step.

use crate::neuron_arrays::NeuronArrays;
use crate::types::{NTType, ReceptorType};

#[cfg(target_os = "macos")]
use super::state::MetalNeuronState;
#[cfg(target_os = "macos")]
use super::GpuContext;
#[cfg(target_os = "macos")]
use metal::CommandBufferRef;

/// Params struct matching the Metal shader's `Params` constant buffer.
#[repr(C)]
struct SecondMessengerParams {
    neuron_count: u32,
    dt: f32,
}

#[inline(always)]
fn hill(x: f32, ec50: f32, n: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let xn = x.powf(n);
    let ec50n = ec50.powf(n);
    xn / (ec50n + xn)
}

fn compute_g_protein_inputs(neurons: &NeuronArrays) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = neurons.count;
    let mut gs_input = vec![0.0f32; n];
    let mut gi_input = vec![0.0f32; n];
    let mut gq_input = vec![0.0f32; n];

    for i in 0..n {
        if neurons.alive[i] == 0 {
            continue;
        }

        let nt = &neurons.nt_conc[i];
        let da = nt[NTType::Dopamine.index()];
        let serotonin = nt[NTType::Serotonin.index()];
        let ne = nt[NTType::Norepinephrine.index()];
        let ach = nt[NTType::Acetylcholine.index()];

        gs_input[i] = hill(da, ReceptorType::D1.ec50_nm(), ReceptorType::D1.hill_n());

        let d2_act = hill(da, ReceptorType::D2.ec50_nm(), ReceptorType::D2.hill_n());
        let ht1a_act = hill(
            serotonin,
            ReceptorType::HT1A.ec50_nm(),
            ReceptorType::HT1A.hill_n(),
        );
        let alpha2_act = hill(
            ne,
            ReceptorType::Alpha2.ec50_nm(),
            ReceptorType::Alpha2.hill_n(),
        );
        gi_input[i] = d2_act.max(ht1a_act).max(alpha2_act);

        let ht2a_act = hill(
            serotonin,
            ReceptorType::HT2A.ec50_nm(),
            ReceptorType::HT2A.hill_n(),
        );
        let m1_act = hill(
            ach,
            ReceptorType::MAChRM1.ec50_nm(),
            ReceptorType::MAChRM1.hill_n(),
        );
        let alpha1_act = hill(
            ne,
            ReceptorType::Alpha1.ec50_nm(),
            ReceptorType::Alpha1.hill_n(),
        );
        gq_input[i] = ht2a_act.max(m1_act).max(alpha1_act);
    }

    (gs_input, gi_input, gq_input)
}

/// Update second messenger cascades on GPU (macOS/Metal).
///
/// Dispatches the `second_messenger` Metal kernel for the full cascade
/// integration. Metabotropic receptor drive is derived on-GPU from `nt_conc`.
#[cfg(target_os = "macos")]
pub fn dispatch_second_messenger(gpu: &GpuContext, neurons: &MetalNeuronState, dt: f32) {
    let cmd = gpu.new_command_buffer();
    encode_second_messenger(gpu, &cmd, neurons, dt);
    gpu.commit_and_wait(cmd);
}

#[cfg(target_os = "macos")]
pub fn encode_second_messenger(
    gpu: &GpuContext,
    cmd: &CommandBufferRef,
    neurons: &MetalNeuronState,
    dt: f32,
) {
    let n = neurons.count as u64;
    if n == 0 {
        return;
    }

    let params = SecondMessengerParams {
        neuron_count: neurons.count as u32,
        dt,
    };
    let param_bytes = unsafe {
        std::slice::from_raw_parts(
            &params as *const SecondMessengerParams as *const u8,
            std::mem::size_of::<SecondMessengerParams>(),
        )
    };

    gpu.encode_1d(
        cmd,
        &gpu.pipelines.second_messenger,
        &[
            (&neurons.gs_active, 0),
            (&neurons.gi_active, 0),
            (&neurons.gq_active, 0),
            (&neurons.nt_conc, 0),
            (&neurons.camp, 0),
            (&neurons.ip3, 0),
            (&neurons.dag, 0),
            (&neurons.pka_activity, 0),
            (&neurons.pkc_activity, 0),
            (&neurons.camkii_activity, 0),
            (&neurons.erk_activity, 0),
            (&neurons.ampa_p, 0),
            (&neurons.kv_p, 0),
            (&neurons.cav_p, 0),
            (&neurons.creb_p, 0),
            (&neurons.ca_cytoplasmic, 0),
            (&neurons.ca_microdomain, 0),
            (&neurons.alive, 0),
        ],
        Some((param_bytes, 18)), // buffer(18): Params
        n,
    );
}

/// CPU fallback for second messenger cascades (non-macOS platforms).
#[cfg(not(target_os = "macos"))]
pub fn dispatch_second_messenger(_gpu: &super::GpuContext, neurons: &mut NeuronArrays, dt: f32) {
    cpu_second_messenger(neurons, dt);
}

/// CPU reference implementation of second messenger cascade integration.
///
/// Mirrors the Metal shader: G-protein relaxation, AC/PDE cAMP dynamics,
/// PLC/IP3/DAG, PKA/PKC/CaMKII/ERK kinase activation, and downstream
/// phosphorylation of AMPA, Kv, Cav, and CREB targets.
pub fn cpu_second_messenger(neurons: &mut NeuronArrays, dt: f32) {
    use crate::constants::*;

    // Compute G-protein inputs first
    let (gs_input, gi_input, gq_input) = compute_g_protein_inputs(neurons);

    // Shader constants (matching second_messenger.metal)
    const G_TAU: f32 = 20.0;
    const AC_GS_GAIN: f32 = 1.5;
    const AC_GS_EC50: f32 = 0.5;
    const GI_INH_GAIN: f32 = 0.85;
    const GI_INH_EC50: f32 = 0.4;
    const AC_BASAL_C: f32 = 0.05;
    const PDE_VMAX_C: f32 = 2.0;
    const PDE_KM_C: f32 = 500.0;
    const PKA_EC50: f32 = 200.0;
    const PKA_HILL: f32 = 1.7;
    const PLC_EC50: f32 = 0.4;
    const PLC_HILL: f32 = 1.0;
    const IP3_PROD_RATE: f32 = 1.0;
    const IP3_DECAY_RATE: f32 = 0.003;
    const IP3_BASAL_C: f32 = 10.0;
    const DAG_PROD_RATE: f32 = 1.0;
    const DAG_DECAY_RATE: f32 = 0.002;
    const DAG_BASAL_C: f32 = 5.0;
    const PKC_DAG_EC50_C: f32 = 100.0;
    const PKC_DAG_HILL: f32 = 1.5;
    const PKC_CA_EC50_C: f32 = 600.0;
    const PKC_CA_HILL: f32 = 1.0;
    const CAMKII_CA_EC50_C: f32 = 800.0;
    const CAMKII_CA_HILL_C: f32 = 4.0;
    const CAMKII_CA_THRESH_C: f32 = 500.0;
    const CAMKII_DRIVE: f32 = 0.005;
    const CAMKII_AUTOPHOS: f32 = 0.001;
    const CAMKII_DEACT: f32 = 0.0003;
    const ERK_PKA_W: f32 = 0.3;
    const ERK_PKC_W: f32 = 0.5;
    const ERK_CAMKII_W: f32 = 0.2;
    const ERK_TAU: f32 = 60.0;
    const AMPA_P_PKA_K: f32 = 0.005;
    const AMPA_P_CAMKII_K: f32 = 0.008;
    const AMPA_P_DEPHOS: f32 = 0.001;
    const KV_P_PKA_K: f32 = 0.003;
    const KV_P_DEPHOS: f32 = 0.001;
    const CAV_P_PKC_K: f32 = 0.004;
    const CAV_P_DEPHOS: f32 = 0.001;
    const CREB_P_PKA_K: f32 = 0.002;
    const CREB_P_ERK_K: f32 = 0.003;
    const CREB_P_DEPHOS: f32 = 0.0005;

    for i in 0..neurons.count {
        if neurons.alive[i] == 0 {
            continue;
        }

        // 1. G-protein relaxation
        let mut gs = neurons.gs_active[i];
        let mut gi = neurons.gi_active[i];
        let mut gq = neurons.gq_active[i];

        gs += dt * (gs_input[i] - gs) / G_TAU;
        gi += dt * (gi_input[i] - gi) / G_TAU;
        gq += dt * (gq_input[i] - gq) / G_TAU;

        gs = gs.clamp(0.0, 1.0);
        gi = gi.clamp(0.0, 1.0);
        gq = gq.clamp(0.0, 1.0);

        neurons.gs_active[i] = gs;
        neurons.gi_active[i] = gi;
        neurons.gq_active[i] = gq;

        // 2. cAMP: AC production - PDE degradation
        let mut c = neurons.camp[i];
        let ac_gs = AC_GS_GAIN * hill(gs, AC_GS_EC50, 1.0);
        let gi_inh = GI_INH_GAIN * hill(gi, GI_INH_EC50, 1.0);
        let ac_total = (AC_BASAL_C + ac_gs) * (1.0 - gi_inh);
        let pde_rate = PDE_VMAX_C * c / (PDE_KM_C + c);
        c += (ac_total - pde_rate) * dt;
        c = c.max(0.0);
        neurons.camp[i] = c;

        // 3. PKA
        let pka = hill(c, PKA_EC50, PKA_HILL);
        neurons.pka_activity[i] = pka;

        // 4. PLC / IP3 / DAG
        let plc = hill(gq, PLC_EC50, PLC_HILL);

        let mut i3 = neurons.ip3[i];
        i3 += (IP3_PROD_RATE * plc - IP3_DECAY_RATE * (i3 - IP3_BASAL_C)) * dt;
        i3 = i3.max(0.0);
        neurons.ip3[i] = i3;

        let mut d = neurons.dag[i];
        d += (DAG_PROD_RATE * plc - DAG_DECAY_RATE * (d - DAG_BASAL_C)) * dt;
        d = d.max(0.0);
        neurons.dag[i] = d;

        // 5. PKC
        let total_ca = neurons.ca_cytoplasmic[i] + neurons.ca_microdomain[i];
        let pkc =
            hill(d, PKC_DAG_EC50_C, PKC_DAG_HILL) * hill(total_ca, PKC_CA_EC50_C, PKC_CA_HILL);
        neurons.pkc_activity[i] = pkc;

        // 6. CaMKII
        let mut ck = neurons.camkii_activity[i];
        let ca_activation = hill(total_ca, CAMKII_CA_EC50_C, CAMKII_CA_HILL_C);
        let autophospho = CAMKII_AUTOPHOS * ck;
        let drive = ca_activation * CAMKII_DRIVE;
        let activation_rate = if total_ca > CAMKII_CA_THRESH_C {
            (drive + autophospho) * (1.0 - ck)
        } else {
            0.0
        };
        let deactivation_rate = CAMKII_DEACT * ck;
        ck += (activation_rate - deactivation_rate) * dt;
        ck = ck.clamp(0.0, 1.0);
        neurons.camkii_activity[i] = ck;

        // 7. ERK cross-talk
        let erk_target = ERK_PKA_W * pka + ERK_PKC_W * pkc + ERK_CAMKII_W * ck;
        let mut erk = neurons.erk_activity[i];
        erk += dt * (erk_target - erk) / ERK_TAU;
        erk = erk.clamp(0.0, 1.0);
        neurons.erk_activity[i] = erk;

        // 8. Phosphorylation targets
        let mut ap = neurons.ampa_p[i];
        ap += (AMPA_P_PKA_K * pka + AMPA_P_CAMKII_K * ck - AMPA_P_DEPHOS * ap) * dt;
        neurons.ampa_p[i] = ap.clamp(0.0, 1.0);

        let mut kp = neurons.kv_p[i];
        kp += (KV_P_PKA_K * pka - KV_P_DEPHOS * kp) * dt;
        neurons.kv_p[i] = kp.clamp(0.0, 1.0);

        let mut cp = neurons.cav_p[i];
        cp += (CAV_P_PKC_K * pkc - CAV_P_DEPHOS * cp) * dt;
        neurons.cav_p[i] = cp.clamp(0.0, 1.0);

        let mut cr = neurons.creb_p[i];
        cr += (CREB_P_PKA_K * pka + CREB_P_ERK_K * erk - CREB_P_DEPHOS * cr) * dt;
        neurons.creb_p[i] = cr.clamp(0.0, 1.0);
    }
}
