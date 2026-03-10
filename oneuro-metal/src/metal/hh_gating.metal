// ============================================================================
// hh_gating.metal — Hodgkin-Huxley gating variable integration
// ============================================================================
// Computes alpha/beta rate functions for Na_v (m^3 h), K_v (n^4), Ca_v (m^2 h)
// and advances gating variables via forward-Euler with clamping.
// One thread per neuron.  SoA layout: each buffer is float[N].
// ============================================================================

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Params constant buffer
// ---------------------------------------------------------------------------
struct Params {
    uint  neuron_count;
    float dt;             // integration timestep (ms)
};

// ---------------------------------------------------------------------------
// Na_v rate functions
// ---------------------------------------------------------------------------
static inline float alpha_m_nav(float V) {
    float dv = V + 40.0f;
    if (abs(dv) < 1e-6f) {
        return 1.0f;
    }
    return 0.1f * dv / (1.0f - exp(-dv / 10.0f));
}

static inline float beta_m_nav(float V) {
    return 4.0f * exp(-(V + 65.0f) / 18.0f);
}

static inline float alpha_h_nav(float V) {
    return 0.07f * exp(-(V + 65.0f) / 20.0f);
}

static inline float beta_h_nav(float V) {
    return 1.0f / (1.0f + exp(-(V + 35.0f) / 10.0f));
}

// ---------------------------------------------------------------------------
// K_v rate functions
// ---------------------------------------------------------------------------
static inline float alpha_n_kv(float V) {
    float dv = V + 55.0f;
    if (abs(dv) < 1e-6f) {
        return 0.1f;
    }
    return 0.01f * dv / (1.0f - exp(-dv / 10.0f));
}

static inline float beta_n_kv(float V) {
    return 0.125f * exp(-(V + 65.0f) / 80.0f);
}

// ---------------------------------------------------------------------------
// Ca_v rate functions
// ---------------------------------------------------------------------------
static inline float alpha_m_cav(float V) {
    float dv = V + 27.0f;
    if (abs(dv) < 1e-6f) {
        return 0.5f;
    }
    return 0.055f * dv / (1.0f - exp(-dv / 3.8f));
}

static inline float beta_m_cav(float V) {
    return 0.94f * exp(-(V + 75.0f) / 17.0f);
}

static inline float alpha_h_cav(float V) {
    return 0.000457f * exp(-(V + 13.0f) / 50.0f);
}

static inline float beta_h_cav(float V) {
    return 0.0065f / (1.0f + exp(-(V + 15.0f) / 28.0f));
}

// ---------------------------------------------------------------------------
// Forward-Euler gating update with clamping
// ---------------------------------------------------------------------------
static inline float gate_update(float gate, float alpha, float beta, float dt) {
    gate += dt * (alpha * (1.0f - gate) - beta * gate);
    return clamp(gate, 0.0f, 1.0f);
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
kernel void hh_gating(
    device const float* voltage       [[buffer(0)]],   // V_m per neuron (mV)
    device       float* nav_m         [[buffer(1)]],   // Na_v activation
    device       float* nav_h         [[buffer(2)]],   // Na_v inactivation
    device       float* kv_n          [[buffer(3)]],   // K_v activation
    device       float* cav_m         [[buffer(4)]],   // Ca_v activation
    device       float* cav_h         [[buffer(5)]],   // Ca_v inactivation
    device const uchar* alive         [[buffer(6)]],   // 1 = alive, 0 = dead
    constant     Params& params       [[buffer(7)]],
    uint         gid                  [[thread_position_in_grid]]
) {
    if (gid >= params.neuron_count) return;
    if (alive[gid] == 0) return;

    float V  = voltage[gid];
    float dt = params.dt;

    // --- Na_v (m^3 h) ---
    float am = alpha_m_nav(V);
    float bm = beta_m_nav(V);
    nav_m[gid] = gate_update(nav_m[gid], am, bm, dt);

    float ah = alpha_h_nav(V);
    float bh = beta_h_nav(V);
    nav_h[gid] = gate_update(nav_h[gid], ah, bh, dt);

    // --- K_v (n^4) ---
    float an = alpha_n_kv(V);
    float bn = beta_n_kv(V);
    kv_n[gid] = gate_update(kv_n[gid], an, bn, dt);

    // --- Ca_v (m^2 h) ---
    float amc = alpha_m_cav(V);
    float bmc = beta_m_cav(V);
    cav_m[gid] = gate_update(cav_m[gid], amc, bmc, dt);

    float ahc = alpha_h_cav(V);
    float bhc = beta_h_cav(V);
    cav_h[gid] = gate_update(cav_h[gid], ahc, bhc, dt);
}
