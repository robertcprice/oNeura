// hh_step.cu — Fused Hodgkin-Huxley gating + membrane integration + spike detection
// One thread per neuron. Compiled at runtime via NVRTC.
//
// Constants (HH biophysics):
//   Nav:   g_max=120  E_rev=50    Kv:    g_max=36   E_rev=-77
//   Kleak: g_max=0.3  E_rev=-77   Cav:   g_max=4.4  E_rev=120
//   NMDA:  g_max=0.5  E_rev=0     AMPA:  g_max=1.0  E_rev=0
//   GabaA: g_max=1.0  E_rev=-80   NAChR: g_max=0.8  E_rev=0
//
// Ion channel index layout (SoA): cond_scale[ch * N + i]
//   0=Nav, 1=Kv, 2=Kleak, 3=Cav, 4=NMDA, 5=AMPA, 6=GabaA, 7=NAChR

// ---------------------------------------------------------------------------
// Channel constants
// ---------------------------------------------------------------------------
#define NAV_G_MAX    120.0f
#define NAV_E_REV     50.0f
#define KV_G_MAX      36.0f
#define KV_E_REV     -77.0f
#define KLEAK_G_MAX    0.3f
#define KLEAK_E_REV  -77.0f
#define CAV_G_MAX      4.4f
#define CAV_E_REV    120.0f
#define NMDA_G_MAX     0.5f
#define NMDA_E_REV     0.0f
#define AMPA_G_MAX     1.0f
#define AMPA_E_REV     0.0f
#define GABAA_G_MAX    1.0f
#define GABAA_E_REV  -80.0f
#define NACHR_G_MAX    0.8f
#define NACHR_E_REV    0.0f

#define NMDA_MG_CONC   1.0f
#define DEFAULT_C_M    1.0f
#define AP_THRESHOLD -20.0f
#define REFRACT_DUR    2.0f
#define V_MIN       -100.0f
#define V_MAX         60.0f

#define CA_SPIKE_INFLUX 50000.0f

// Channel indices in cond_scale SoA
#define CH_NAV   0
#define CH_KV    1
#define CH_KLEAK 2
#define CH_CAV   3
#define CH_NMDA  4
#define CH_AMPA  5
#define CH_GABAA 6
#define CH_NACHR 7

// ---------------------------------------------------------------------------
// Safe rate helpers — Taylor expansion near singularities
// Pattern: alpha = a*(V+V0) / (1 - exp(-(V+V0)/k))
//   Near V = -V0 the denominator -> 0, L'Hopital gives a*k.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float safe_rate(float a, float V, float V0, float k)
{
    float x = V + V0;
    float neg_x_over_k = -x / k;
    // When |x| < 1e-4 use first-order Taylor: a*k + a*x/2
    if (fabsf(x) < 1.0e-4f) {
        return a * k + a * x * 0.5f;
    }
    return a * x / (1.0f - expf(neg_x_over_k));
}

// ---------------------------------------------------------------------------
// HH gating rate functions
// ---------------------------------------------------------------------------

// Nav m-gate
__device__ __forceinline__ float alpha_m(float V)
{
    // 0.1*(V+40)/(1 - exp(-(V+40)/10)), singular at V=-40
    return safe_rate(0.1f, V, 40.0f, 10.0f);
}

__device__ __forceinline__ float beta_m(float V)
{
    return 4.0f * expf(-(V + 65.0f) / 18.0f);
}

// Nav h-gate
__device__ __forceinline__ float alpha_h(float V)
{
    return 0.07f * expf(-(V + 65.0f) / 20.0f);
}

__device__ __forceinline__ float beta_h(float V)
{
    return 1.0f / (1.0f + expf(-(V + 35.0f) / 10.0f));
}

// Kv n-gate
__device__ __forceinline__ float alpha_n(float V)
{
    // 0.01*(V+55)/(1 - exp(-(V+55)/10)), singular at V=-55
    return safe_rate(0.01f, V, 55.0f, 10.0f);
}

__device__ __forceinline__ float beta_n(float V)
{
    return 0.125f * expf(-(V + 65.0f) / 80.0f);
}

// Cav m-gate
__device__ __forceinline__ float alpha_m_ca(float V)
{
    // 0.055*(V+27)/(1 - exp(-(V+27)/3.8)), singular at V=-27
    return safe_rate(0.055f, V, 27.0f, 3.8f);
}

__device__ __forceinline__ float beta_m_ca(float V)
{
    return 0.94f * expf(-(V + 75.0f) / 17.0f);
}

// Cav h-gate
__device__ __forceinline__ float alpha_h_ca(float V)
{
    return 0.000457f * expf(-(V + 13.0f) / 50.0f);
}

__device__ __forceinline__ float beta_h_ca(float V)
{
    return 0.0065f / (1.0f + expf(-(V + 15.0f) / 28.0f));
}

// ---------------------------------------------------------------------------
// Clamp helper
// ---------------------------------------------------------------------------
__device__ __forceinline__ float clampf(float x, float lo, float hi)
{
    return fminf(fmaxf(x, lo), hi);
}

// ---------------------------------------------------------------------------
// Main kernel: hh_step
// ---------------------------------------------------------------------------
extern "C" __global__ void hh_step(
    float* __restrict__ V,
    float* __restrict__ nav_m,
    float* __restrict__ nav_h,
    float* __restrict__ kv_n,
    float* __restrict__ cav_m_gate,
    float* __restrict__ cav_h_gate,
    const float* __restrict__ ampa_open,
    const float* __restrict__ nmda_open,
    const float* __restrict__ gabaa_open,
    const float* __restrict__ nachr_open,
    const float* __restrict__ cond_scale,
    float* __restrict__ ext_current,
    float* __restrict__ refractory,
    unsigned char* __restrict__ fired,
    unsigned int* __restrict__ spike_count,
    float* __restrict__ ca_micro,
    int N,
    float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // --- Refractory check ---
    if (refractory[i] > 0.0f) {
        refractory[i] -= dt;
        fired[i] = 0;
        ext_current[i] = 0.0f;  // clear external current even during refractory
        return;
    }

    float v = V[i];
    float old_v = v;

    // --- Load gating variables ---
    float m  = nav_m[i];
    float h  = nav_h[i];
    float n  = kv_n[i];
    float mc = cav_m_gate[i];
    float hc = cav_h_gate[i];

    // --- Euler update gating variables ---
    float am  = alpha_m(v);
    float bm  = beta_m(v);
    m += dt * (am * (1.0f - m) - bm * m);
    m = clampf(m, 0.0f, 1.0f);

    float ah  = alpha_h(v);
    float bh  = beta_h(v);
    h += dt * (ah * (1.0f - h) - bh * h);
    h = clampf(h, 0.0f, 1.0f);

    float an  = alpha_n(v);
    float bn  = beta_n(v);
    n += dt * (an * (1.0f - n) - bn * n);
    n = clampf(n, 0.0f, 1.0f);

    float amc = alpha_m_ca(v);
    float bmc = beta_m_ca(v);
    mc += dt * (amc * (1.0f - mc) - bmc * mc);
    mc = clampf(mc, 0.0f, 1.0f);

    float ahc = alpha_h_ca(v);
    float bhc = beta_h_ca(v);
    hc += dt * (ahc * (1.0f - hc) - bhc * hc);
    hc = clampf(hc, 0.0f, 1.0f);

    // --- Store updated gating variables ---
    nav_m[i]      = m;
    nav_h[i]      = h;
    kv_n[i]       = n;
    cav_m_gate[i] = mc;
    cav_h_gate[i] = hc;

    // --- Load conductance scales (SoA layout: cond_scale[ch * N + i]) ---
    float cs_nav   = cond_scale[CH_NAV   * N + i];
    float cs_kv    = cond_scale[CH_KV    * N + i];
    float cs_kleak = cond_scale[CH_KLEAK * N + i];
    float cs_cav   = cond_scale[CH_CAV   * N + i];
    float cs_nmda  = cond_scale[CH_NMDA  * N + i];
    float cs_ampa  = cond_scale[CH_AMPA  * N + i];
    float cs_gabaa = cond_scale[CH_GABAA * N + i];
    float cs_nachr = cond_scale[CH_NACHR * N + i];

    // --- Compute ionic currents ---
    // I = g_max * scale * gating * (V - E_rev)

    // Nav: g = m^3 * h
    float g_nav = NAV_G_MAX * cs_nav * m * m * m * h;
    float I_nav = g_nav * (v - NAV_E_REV);

    // Kv: g = n^4
    float g_kv = KV_G_MAX * cs_kv * n * n * n * n;
    float I_kv = g_kv * (v - KV_E_REV);

    // Kleak: ohmic (no gating)
    float I_kleak = KLEAK_G_MAX * cs_kleak * (v - KLEAK_E_REV);

    // Cav: g = mc^2 * hc
    float g_cav = CAV_G_MAX * cs_cav * mc * mc * hc;
    float I_cav = g_cav * (v - CAV_E_REV);

    // NMDA: voltage-dependent Mg2+ block
    // B(V) = 1 / (1 + [Mg2+]/3.57 * exp(-0.062*V))
    float mg_block = 1.0f / (1.0f + (NMDA_MG_CONC / 3.57f) * expf(-0.062f * v));
    float I_nmda = NMDA_G_MAX * cs_nmda * nmda_open[i] * mg_block * (v - NMDA_E_REV);

    // AMPA
    float I_ampa = AMPA_G_MAX * cs_ampa * ampa_open[i] * (v - AMPA_E_REV);

    // GABA-A
    float I_gabaa = GABAA_G_MAX * cs_gabaa * gabaa_open[i] * (v - GABAA_E_REV);

    // NAChR
    float I_nachr = NACHR_G_MAX * cs_nachr * nachr_open[i] * (v - NACHR_E_REV);

    // --- Total ionic current ---
    float I_total = I_nav + I_kv + I_kleak + I_cav + I_nmda + I_ampa + I_gabaa + I_nachr;

    // --- Membrane equation: C_m * dV/dt = -I_total + I_ext ---
    float dv = (-I_total + ext_current[i]) / DEFAULT_C_M;
    v += dt * dv;
    v = clampf(v, V_MIN, V_MAX);

    // --- Spike detection: upward threshold crossing ---
    unsigned char spiked = (v >= AP_THRESHOLD && old_v < AP_THRESHOLD) ? 1 : 0;

    if (spiked) {
        refractory[i] = REFRACT_DUR;
        spike_count[i] += 1;
        ca_micro[i] += CA_SPIKE_INFLUX;
    }

    // --- Write outputs ---
    V[i] = v;
    fired[i] = spiked;
    ext_current[i] = 0.0f;  // clear for next timestep
}
