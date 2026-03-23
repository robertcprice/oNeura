// ============================================================================
// calcium_ode.metal — 4-compartment Ca2+ dynamics
// ============================================================================
// Microdomain, cytoplasmic, ER, and mitochondrial calcium pools with:
//   - Microdomain -> cytoplasm diffusion (exponential relaxation)
//   - IP3R and RyR release from ER (Hill gating)
//   - SERCA pump (cytoplasm -> ER)
//   - MCU uptake (cytoplasm -> mitochondria)
//   - Mitochondrial release
//   - PMCA + NCX export, passive leak in
// One thread per neuron.  All concentrations in nM.
// ============================================================================

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constant float BUFFER_CAPACITY  = 20.0f;    // cytoplasmic buffering factor
constant float MICRO_TAU        = 0.5f;     // microdomain diffusion time constant (ms)

// IP3R
constant float IP3R_RATE        = 6000.0f;  // max IP3R flux (nM/ms)
constant float IP3_EC50         = 300.0f;
constant float IP3_HILL         = 2.0f;
constant float IP3R_CA_ACT_EC50 = 200.0f;
constant float IP3R_CA_ACT_HILL = 2.0f;
constant float IP3R_CA_INH_EC50 = 500.0f;
constant float IP3R_CA_INH_HILL = 2.0f;

// RyR
constant float RYR_RATE         = 2000.0f;
constant float RYR_ACT_EC50     = 500.0f;
constant float RYR_ACT_HILL     = 3.0f;
constant float RYR_INH_EC50     = 1500.0f;
constant float RYR_INH_HILL     = 3.0f;

// SERCA
constant float SERCA_RATE       = 1500.0f;
constant float SERCA_EC50       = 200.0f;
constant float SERCA_HILL       = 2.0f;

// ER leak
// (handled implicitly as part of net_er)

// MCU
constant float MCU_RATE         = 800.0f;
constant float MCU_EC50         = 10000.0f;
constant float MCU_HILL         = 2.5f;

// Mito release
constant float MITO_RELEASE_RATE = 0.02f;

// Plasma membrane
constant float PMCA_RATE        = 300.0f;
constant float PMCA_EC50        = 100.0f;
constant float NCX_RATE         = 1500.0f;
constant float NCX_EC50         = 1000.0f;
constant float LEAK_IN_RATE     = 210.0f;

// ---------------------------------------------------------------------------
// Params
// ---------------------------------------------------------------------------
struct Params {
    uint  neuron_count;
    float dt;
};

// ---------------------------------------------------------------------------
// Hill function: x^n / (ec50^n + x^n), returns 0 if x <= 0
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
kernel void calcium_ode(
    device       float* ca_cytoplasmic   [[buffer(0)]],
    device       float* ca_er            [[buffer(1)]],
    device       float* ca_mitochondrial [[buffer(2)]],
    device       float* ca_microdomain   [[buffer(3)]],
    device const float* ip3              [[buffer(4)]],   // IP3 concentration (nM)
    device const uchar* alive            [[buffer(5)]],
    constant     Params& params          [[buffer(6)]],
    uint         gid                     [[thread_position_in_grid]]
) {
    if (gid >= params.neuron_count) return;
    if (alive[gid] == 0) return;

    float dt    = params.dt;
    float micro = ca_microdomain[gid];
    float cyto  = ca_cytoplasmic[gid];
    float er    = ca_er[gid];
    float mito  = ca_mitochondrial[gid];
    float ip3_c = ip3[gid];

    // -----------------------------------------------------------------------
    // 1. Microdomain -> cytoplasm diffusion (exponential relaxation)
    // -----------------------------------------------------------------------
    float micro_diff     = micro - cyto;
    float diffusion_flux = micro_diff * (1.0f - exp(-dt / MICRO_TAU));
    micro -= diffusion_flux;
    cyto  += diffusion_flux / BUFFER_CAPACITY;

    // -----------------------------------------------------------------------
    // 2. IP3R: ER -> cytoplasm
    // -----------------------------------------------------------------------
    float ip3_gate = hill(ip3_c, IP3_EC50, IP3_HILL);
    float ca_act   = hill(cyto,  IP3R_CA_ACT_EC50, IP3R_CA_ACT_HILL);
    float ca_inh   = hill(cyto,  IP3R_CA_INH_EC50, IP3R_CA_INH_HILL);
    float gradient = (er > 0.0f) ? max(0.0f, (er - cyto) / er) : 0.0f;
    float ip3r_flux = IP3R_RATE * ip3_gate * ca_act * (1.0f - ca_inh) * gradient * dt;

    // -----------------------------------------------------------------------
    // 3. RyR: CICR (calcium-induced calcium release)
    // -----------------------------------------------------------------------
    float ryr_act = hill(cyto, RYR_ACT_EC50, RYR_ACT_HILL);
    float ryr_inh = 1.0f - hill(cyto, RYR_INH_EC50, RYR_INH_HILL);
    float ryr_flux = RYR_RATE * ryr_act * max(0.0f, ryr_inh) * gradient * dt;

    // -----------------------------------------------------------------------
    // 4. SERCA: cytoplasm -> ER
    // -----------------------------------------------------------------------
    float serca_flux = SERCA_RATE * hill(cyto, SERCA_EC50, SERCA_HILL) * dt;

    // -----------------------------------------------------------------------
    // 5. ER leak (small constant leak)
    // -----------------------------------------------------------------------
    float er_leak = er * 0.001f * dt;

    // -----------------------------------------------------------------------
    // 6. Net ER exchange
    // -----------------------------------------------------------------------
    float net_er = ip3r_flux + ryr_flux - serca_flux + er_leak;
    cyto += net_er / BUFFER_CAPACITY;
    er   -= net_er;

    // -----------------------------------------------------------------------
    // 7. MCU: cytoplasm -> mitochondria
    // -----------------------------------------------------------------------
    float mcu_flux = MCU_RATE * hill(cyto, MCU_EC50, MCU_HILL) * dt;
    cyto -= mcu_flux / BUFFER_CAPACITY;
    mito += mcu_flux;

    // -----------------------------------------------------------------------
    // 8. Mitochondrial release
    // -----------------------------------------------------------------------
    float mito_release = mito * MITO_RELEASE_RATE * dt;
    mito -= mito_release;
    cyto += mito_release / BUFFER_CAPACITY;

    // -----------------------------------------------------------------------
    // 9. Plasma membrane: PMCA + NCX export, passive leak in
    // -----------------------------------------------------------------------
    float pmca    = PMCA_RATE * hill(cyto, PMCA_EC50, 1.0f) * dt;
    float ncx     = NCX_RATE  * hill(cyto, NCX_EC50,  1.0f) * dt;
    float leak_in = LEAK_IN_RATE * dt;
    cyto -= (pmca + ncx - leak_in) / BUFFER_CAPACITY;

    // -----------------------------------------------------------------------
    // 10. Clamp all >= 0
    // -----------------------------------------------------------------------
    ca_microdomain[gid]   = max(0.0f, micro);
    ca_cytoplasmic[gid]   = max(0.0f, cyto);
    ca_er[gid]            = max(0.0f, er);
    ca_mitochondrial[gid] = max(0.0f, mito);
}
