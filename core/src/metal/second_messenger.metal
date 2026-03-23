// ============================================================================
// second_messenger.metal — Intracellular signaling cascade integration
// ============================================================================
// G-protein activation + cAMP/PKA + PLC/IP3/DAG + PKC + CaMKII + ERK +
// phosphorylation of downstream targets (AMPA, Kv, Cav, CREB).
// One thread per neuron.  The most complex shader in the pipeline.
//
// Cascade input signals (gs_input, gi_input, gq_input) are passed per-neuron
// via dedicated buffers — Rust dispatch computes these from receptor
// activations before launching this shader.
// ============================================================================

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// G-protein time constants
constant float G_TAU = 20.0f;   // ms

// Adenylyl cyclase / cAMP
constant float AC_GS_GAIN      = 1.5f;
constant float AC_GS_EC50      = 0.5f;
constant float GI_INH_GAIN     = 0.85f;
constant float GI_INH_EC50     = 0.4f;
constant float AC_BASAL        = 0.05f;
constant float PDE_VMAX        = 2.0f;
constant float PDE_KM          = 500.0f;

// PKA
constant float PKA_EC50        = 200.0f;
constant float PKA_HILL        = 1.7f;

// NT / metabotropic receptor constants
constant uint NT_COUNT         = 6u;
constant uint NT_DOPAMINE      = 0u;
constant uint NT_SEROTONIN     = 1u;
constant uint NT_NOREPINEPHRINE = 2u;
constant uint NT_ACETYLCHOLINE = 3u;

constant float D1_EC50         = 2340.0f;
constant float D2_EC50         = 2.8f;
constant float HT1A_EC50       = 3.2f;
constant float HT2A_EC50       = 54.0f;
constant float M1_EC50         = 7900.0f;
constant float ALPHA1_EC50     = 330.0f;
constant float ALPHA2_EC50     = 56.0f;

constant float D1_HILL         = 1.0f;
constant float D2_HILL         = 1.0f;
constant float HT1A_HILL       = 1.0f;
constant float HT2A_HILL       = 1.2f;
constant float M1_HILL         = 1.0f;
constant float ALPHA1_HILL     = 1.0f;
constant float ALPHA2_HILL     = 1.0f;

// PLC / IP3 / DAG
constant float PLC_EC50        = 0.4f;
constant float PLC_HILL        = 1.0f;
constant float IP3_PROD_RATE   = 1.0f;
constant float IP3_DECAY_RATE  = 0.003f;
constant float IP3_BASAL       = 10.0f;
constant float DAG_PROD_RATE   = 1.0f;
constant float DAG_DECAY_RATE  = 0.002f;
constant float DAG_BASAL       = 5.0f;

// PKC
constant float PKC_DAG_EC50    = 100.0f;
constant float PKC_DAG_HILL    = 1.5f;
constant float PKC_CA_EC50     = 600.0f;
constant float PKC_CA_HILL     = 1.0f;

// CaMKII
constant float CAMKII_CA_EC50  = 800.0f;
constant float CAMKII_CA_HILL  = 4.0f;
constant float CAMKII_CA_THRESH = 500.0f;
constant float CAMKII_DRIVE    = 0.005f;
constant float CAMKII_AUTOPHOS = 0.001f;
constant float CAMKII_DEACT    = 0.0003f;

// ERK cross-talk weights
constant float ERK_PKA_W       = 0.3f;
constant float ERK_PKC_W       = 0.5f;
constant float ERK_CAMKII_W    = 0.2f;
constant float ERK_TAU         = 60.0f;   // ms smoothing

// Phosphorylation rate constants
constant float AMPA_P_PKA_K    = 0.005f;
constant float AMPA_P_CAMKII_K = 0.008f;
constant float AMPA_P_DEPHOS   = 0.001f;

constant float KV_P_PKA_K      = 0.003f;
constant float KV_P_DEPHOS     = 0.001f;

constant float CAV_P_PKC_K     = 0.004f;
constant float CAV_P_DEPHOS    = 0.001f;

constant float CREB_P_PKA_K    = 0.002f;
constant float CREB_P_ERK_K    = 0.003f;
constant float CREB_P_DEPHOS   = 0.0005f;

// ---------------------------------------------------------------------------
// Params
// ---------------------------------------------------------------------------
struct Params {
    uint  neuron_count;
    float dt;
};

// ---------------------------------------------------------------------------
// Hill function
// ---------------------------------------------------------------------------
static inline float hill(float x, float ec50, float n) {
    if (x <= 0.0f) return 0.0f;
    float xn    = pow(x, n);
    float ec50n = pow(ec50, n);
    return xn / (ec50n + xn);
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
kernel void second_messenger(
    // G-protein states
    device       float* gs_active          [[buffer(0)]],
    device       float* gi_active          [[buffer(1)]],
    device       float* gq_active          [[buffer(2)]],
    device const float* nt_conc            [[buffer(3)]],   // AoS: 6 floats per neuron
    // Second messengers
    device       float* camp               [[buffer(4)]],
    device       float* ip3                [[buffer(5)]],
    device       float* dag                [[buffer(6)]],
    // Kinase activities
    device       float* pka_activity       [[buffer(7)]],
    device       float* pkc_activity       [[buffer(8)]],
    device       float* camkii_activity    [[buffer(9)]],
    device       float* erk_activity       [[buffer(10)]],
    // Phosphorylation states
    device       float* ampa_p             [[buffer(11)]],
    device       float* kv_p               [[buffer(12)]],
    device       float* cav_p              [[buffer(13)]],
    device       float* creb_p             [[buffer(14)]],
    // Calcium (read-only, from calcium_ode)
    device const float* ca_cytoplasmic     [[buffer(15)]],
    device const float* ca_microdomain     [[buffer(16)]],
    // Neuron mask
    device const uchar* alive              [[buffer(17)]],
    constant     Params& params            [[buffer(18)]],
    uint         gid                       [[thread_position_in_grid]]
) {
    if (gid >= params.neuron_count) return;
    if (alive[gid] == 0) return;

    float dt = params.dt;

    uint nt_base = gid * NT_COUNT;
    float da = nt_conc[nt_base + NT_DOPAMINE];
    float serotonin = nt_conc[nt_base + NT_SEROTONIN];
    float ne = nt_conc[nt_base + NT_NOREPINEPHRINE];
    float ach = nt_conc[nt_base + NT_ACETYLCHOLINE];

    float gs_input = hill(da, D1_EC50, D1_HILL);
    float d2_act = hill(da, D2_EC50, D2_HILL);
    float ht1a_act = hill(serotonin, HT1A_EC50, HT1A_HILL);
    float alpha2_act = hill(ne, ALPHA2_EC50, ALPHA2_HILL);
    float gi_input = max(d2_act, max(ht1a_act, alpha2_act));
    float ht2a_act = hill(serotonin, HT2A_EC50, HT2A_HILL);
    float m1_act = hill(ach, M1_EC50, M1_HILL);
    float alpha1_act = hill(ne, ALPHA1_EC50, ALPHA1_HILL);
    float gq_input = max(ht2a_act, max(m1_act, alpha1_act));

    // -----------------------------------------------------------------------
    // 1. G-protein activation (first-order relaxation, tau = 20 ms)
    // -----------------------------------------------------------------------
    float gs = gs_active[gid];
    float gi = gi_active[gid];
    float gq = gq_active[gid];

    gs += dt * (gs_input - gs) / G_TAU;
    gi += dt * (gi_input - gi) / G_TAU;
    gq += dt * (gq_input - gq) / G_TAU;

    gs = clamp(gs, 0.0f, 1.0f);
    gi = clamp(gi, 0.0f, 1.0f);
    gq = clamp(gq, 0.0f, 1.0f);

    gs_active[gid] = gs;
    gi_active[gid] = gi;
    gq_active[gid] = gq;

    // -----------------------------------------------------------------------
    // 2. cAMP: AC production (Gs stimulated, Gi inhibited) - PDE degradation
    // -----------------------------------------------------------------------
    float c = camp[gid];

    float ac_gs    = AC_GS_GAIN * hill(gs, AC_GS_EC50, 1.0f);
    float gi_inh   = GI_INH_GAIN * hill(gi, GI_INH_EC50, 1.0f);
    float ac_total = (AC_BASAL + ac_gs) * (1.0f - gi_inh);
    float pde_rate = PDE_VMAX * c / (PDE_KM + c);

    c += (ac_total - pde_rate) * dt;
    c = max(0.0f, c);
    camp[gid] = c;

    // -----------------------------------------------------------------------
    // 3. PKA = hill(cAMP, 200, 1.7)
    // -----------------------------------------------------------------------
    float pka = hill(c, PKA_EC50, PKA_HILL);
    pka_activity[gid] = pka;

    // -----------------------------------------------------------------------
    // 4. PLC / IP3 / DAG
    // -----------------------------------------------------------------------
    float plc = hill(gq, PLC_EC50, PLC_HILL);

    float i3 = ip3[gid];
    i3 += (IP3_PROD_RATE * plc - IP3_DECAY_RATE * (i3 - IP3_BASAL)) * dt;
    i3 = max(0.0f, i3);
    ip3[gid] = i3;

    float d = dag[gid];
    d += (DAG_PROD_RATE * plc - DAG_DECAY_RATE * (d - DAG_BASAL)) * dt;
    d = max(0.0f, d);
    dag[gid] = d;

    // -----------------------------------------------------------------------
    // 5. PKC = hill(DAG) * hill(Ca_total)
    //    Total Ca = cytoplasmic + microdomain
    // -----------------------------------------------------------------------
    float total_ca = ca_cytoplasmic[gid] + ca_microdomain[gid];
    float pkc = hill(d, PKC_DAG_EC50, PKC_DAG_HILL)
              * hill(total_ca, PKC_CA_EC50, PKC_CA_HILL);
    pkc_activity[gid] = pkc;

    // -----------------------------------------------------------------------
    // 6. CaMKII — bistable switch with autophosphorylation
    // -----------------------------------------------------------------------
    float ck = camkii_activity[gid];

    float ca_activation = hill(total_ca, CAMKII_CA_EC50, CAMKII_CA_HILL);
    float autophospho   = CAMKII_AUTOPHOS * ck;
    float drive         = ca_activation * CAMKII_DRIVE;

    // Activation only proceeds when calcium exceeds threshold
    float activation_rate = 0.0f;
    if (total_ca > CAMKII_CA_THRESH) {
        activation_rate = (drive + autophospho) * (1.0f - ck);
    }
    float deactivation_rate = CAMKII_DEACT * ck;

    ck += (activation_rate - deactivation_rate) * dt;
    ck = clamp(ck, 0.0f, 1.0f);
    camkii_activity[gid] = ck;

    // -----------------------------------------------------------------------
    // 7. ERK — cross-talk from PKA, PKC, CaMKII (first-order smooth)
    // -----------------------------------------------------------------------
    float erk_target = ERK_PKA_W * pka + ERK_PKC_W * pkc + ERK_CAMKII_W * ck;
    float erk = erk_activity[gid];
    erk += dt * (erk_target - erk) / ERK_TAU;
    erk = clamp(erk, 0.0f, 1.0f);
    erk_activity[gid] = erk;

    // -----------------------------------------------------------------------
    // 8. Phosphorylation targets
    // -----------------------------------------------------------------------

    // AMPA phosphorylation (PKA + CaMKII drive, phosphatase removes)
    float ap = ampa_p[gid];
    ap += (AMPA_P_PKA_K * pka + AMPA_P_CAMKII_K * ck - AMPA_P_DEPHOS * ap) * dt;
    ampa_p[gid] = clamp(ap, 0.0f, 1.0f);

    // Kv phosphorylation (PKA drives)
    float kp = kv_p[gid];
    kp += (KV_P_PKA_K * pka - KV_P_DEPHOS * kp) * dt;
    kv_p[gid] = clamp(kp, 0.0f, 1.0f);

    // Cav phosphorylation (PKC drives)
    float cp = cav_p[gid];
    cp += (CAV_P_PKC_K * pkc - CAV_P_DEPHOS * cp) * dt;
    cav_p[gid] = clamp(cp, 0.0f, 1.0f);

    // CREB phosphorylation (PKA + ERK drive)
    float cr = creb_p[gid];
    cr += (CREB_P_PKA_K * pka + CREB_P_ERK_K * erk - CREB_P_DEPHOS * cr) * dt;
    creb_p[gid] = clamp(cr, 0.0f, 1.0f);
}
