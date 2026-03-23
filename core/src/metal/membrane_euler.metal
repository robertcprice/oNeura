// ============================================================================
// membrane_euler.metal — Ionic current summation + Euler voltage integration
// ============================================================================
// Computes total ionic current from 8 channel types (Na_v, K_v, K_leak, Ca_v,
// AMPA, NMDA, GABA-A, nAChR), integrates membrane voltage via forward-Euler,
// and performs spike detection with refractory period.
// One thread per neuron.  SoA layout.
// ============================================================================

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constant float G_NAV    = 120.0f;   // mS/cm^2
constant float G_KV     = 36.0f;
constant float G_KLEAK  = 0.3f;
constant float G_CAV    = 4.4f;
constant float G_NMDA   = 0.5f;
constant float G_AMPA   = 1.0f;
constant float G_GABAA  = 1.0f;
constant float G_NACHR  = 0.8f;

constant float E_NA     = 50.0f;    // mV reversal potentials
constant float E_K      = -77.0f;
constant float E_CA     = 120.0f;
constant float E_EXC    = 0.0f;     // excitatory (AMPA, NMDA, nAChR)
constant float E_INH    = -80.0f;   // inhibitory (GABA-A)

constant float V_MIN    = -100.0f;
constant float V_MAX    = 60.0f;

// ---------------------------------------------------------------------------
// Params constant buffer
// ---------------------------------------------------------------------------
struct Params {
    uint  neuron_count;
    float dt;
    float global_bias;
    float channel_g_max[8];
    float kleak_reversal_mv;
    float membrane_capacitance_uf;
    float spike_threshold_mv;
    float refractory_period_ms;
};

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
kernel void membrane_euler(
    device       float* voltage            [[buffer(0)]],   // V_m (read/write)
    device       float* prev_voltage       [[buffer(1)]],   // previous V_m (write)
    device       uchar* fired              [[buffer(2)]],   // 1 on spike tick
    device       float* refractory_timer   [[buffer(3)]],   // countdown (ms)
    device       uint* spike_count         [[buffer(4)]],   // cumulative spike count
    device const float* nav_m              [[buffer(5)]],
    device const float* nav_h              [[buffer(6)]],
    device const float* kv_n               [[buffer(7)]],
    device const float* cav_m              [[buffer(8)]],
    device const float* cav_h              [[buffer(9)]],
    device const float* ampa_open          [[buffer(10)]],
    device const float* nmda_open          [[buffer(11)]],
    device const float* gabaa_open         [[buffer(12)]],
    device const float* nachr_open         [[buffer(13)]],
    device const float* conductance_scale  [[buffer(14)]],  // AoS: 8 floats per neuron
    device       float* external_current   [[buffer(15)]],  // consumed then cleared
    device       float* synaptic_current   [[buffer(16)]],  // consumed then cleared
    device const uchar* alive              [[buffer(17)]],
    device       float* excitability_bias  [[buffer(18)]],
    constant     Params& params            [[buffer(19)]],
    uint         gid                       [[thread_position_in_grid]]
) {
    if (gid >= params.neuron_count) return;
    if (alive[gid] == 0) return;

    float V  = voltage[gid];
    float dt = params.dt;

    // Store previous voltage for spike detection
    prev_voltage[gid] = V;

    uint base = gid * 8u;
    float cs_nav   = conductance_scale[base + 0];
    float cs_kv    = conductance_scale[base + 1];
    float cs_kleak = conductance_scale[base + 2];
    float cs_cav   = conductance_scale[base + 3];
    float cs_nmda  = conductance_scale[base + 4];
    float cs_ampa  = conductance_scale[base + 5];
    float cs_gabaa = conductance_scale[base + 6];
    float cs_nachr = conductance_scale[base + 7];

    // --- Gating products ---
    float m_nav = nav_m[gid];
    float h_nav = nav_h[gid];
    float n_kv  = kv_n[gid];
    float m_cav = cav_m[gid];
    float h_cav = cav_h[gid];

    float m3h_nav = m_nav * m_nav * m_nav * h_nav;          // Na_v: m^3 h
    float n4_kv   = n_kv * n_kv * n_kv * n_kv;              // K_v:  n^4
    float m2h_cav = m_cav * m_cav * h_cav;                   // Ca_v: m^2 h

    // --- Channel currents ---
    float I_Nav   = params.channel_g_max[0] * cs_nav   * m3h_nav * (V - E_NA);
    float I_Kv    = params.channel_g_max[1] * cs_kv    * n4_kv   * (V - E_K);
    float I_Kleak = params.channel_g_max[2] * cs_kleak           * (V - params.kleak_reversal_mv);
    float I_Cav   = params.channel_g_max[3] * cs_cav   * m2h_cav * (V - E_CA);

    // Mg^2+ block for NMDA
    float mg_block = 1.0f / (1.0f + 1.0f * exp(-0.062f * V) / 3.57f);

    float I_AMPA  = params.channel_g_max[5] * cs_ampa  * ampa_open[gid]            * (V - E_EXC);
    float I_NMDA  = params.channel_g_max[4] * cs_nmda  * nmda_open[gid] * mg_block * (V - E_EXC);
    float I_GabaA = params.channel_g_max[6] * cs_gabaa * gabaa_open[gid]           * (V - E_INH);
    float I_nAChR = params.channel_g_max[7] * cs_nachr * nachr_open[gid]           * (V - E_EXC);

    // --- Total current ---
    float I_total = I_Nav + I_Kv + I_Kleak + I_Cav
                  + I_AMPA + I_NMDA + I_GabaA + I_nAChR;

    float bias = excitability_bias[gid] + params.global_bias;
    float I_ext = external_current[gid] + synaptic_current[gid] + bias;

    // --- Euler integration ---
    float dV = (-I_total + I_ext) / max(params.membrane_capacitance_uf, 0.1f) * dt;
    float V_new = clamp(V + dV, V_MIN, V_MAX);

    // --- Refractory timer countdown ---
    float ref = refractory_timer[gid] - dt;

    // --- Spike detection ---
    // Threshold crossing: prev < -20 AND new >= -20 AND not refractory
    uchar f = 0;
    if (V < params.spike_threshold_mv && V_new >= params.spike_threshold_mv && ref <= 0.0f) {
        f = 1;
        ref = max(params.refractory_period_ms, dt);
        spike_count[gid] += 1u;
    }
    fired[gid] = f;
    refractory_timer[gid] = max(ref, 0.0f);

    // --- Write voltage ---
    voltage[gid] = V_new;
    external_current[gid] = 0.0f;
    synaptic_current[gid] = 0.0f;
}
