// ============================================================================
// hill_binding.metal — Receptor activation via Hill-equation ligand binding
// ============================================================================
// Converts neurotransmitter concentrations into receptor open fractions:
//   Glutamate -> AMPA (EC50=500, n=1.3)
//   Glutamate -> NMDA (EC50=2400, n=1.5)  — Mg block handled in membrane shader
//   GABA      -> GABA-A (EC50=200, n=2.0)
//   ACh       -> nAChR (EC50=30, n=1.8)
//
// One thread per neuron.
// NT concentration layout: 6 contiguous SoA arrays, each float[N].
//   Index 0: dopamine
//   Index 1: serotonin
//   Index 2: norepinephrine
//   Index 3: acetylcholine
//   Index 4: GABA
//   Index 5: glutamate
// ============================================================================

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// AMPA (glutamate)
constant float AMPA_EC50 = 500.0f;
constant float AMPA_HILL = 1.3f;

// NMDA (glutamate)
constant float NMDA_EC50 = 2400.0f;
constant float NMDA_HILL = 1.5f;

// GABA-A (GABA)
constant float GABAA_EC50 = 200.0f;
constant float GABAA_HILL = 2.0f;

// nAChR (acetylcholine)
constant float NACHR_EC50 = 30.0f;
constant float NACHR_HILL = 1.8f;

// NT index offsets
constant uint NT_DOPAMINE      = 0;
constant uint NT_SEROTONIN     = 1;
constant uint NT_NOREPINEPHRINE = 2;
constant uint NT_ACETYLCHOLINE = 3;
constant uint NT_GABA          = 4;
constant uint NT_GLUTAMATE     = 5;
constant uint NT_COUNT         = 6;

// ---------------------------------------------------------------------------
// Params
// ---------------------------------------------------------------------------
struct Params {
    uint neuron_count;
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
kernel void hill_binding(
    device const float* nt_conc      [[buffer(0)]],   // AoS: 6 floats per neuron
    device       float* ampa_open    [[buffer(1)]],
    device       float* nmda_open    [[buffer(2)]],
    device       float* gabaa_open   [[buffer(3)]],
    device       float* nachr_open   [[buffer(4)]],
    device const uchar* alive        [[buffer(5)]],
    constant     Params& params      [[buffer(6)]],
    uint         gid                 [[thread_position_in_grid]]
) {
    if (gid >= params.neuron_count) return;
    if (alive[gid] == 0) return;

    uint base = gid * NT_COUNT;
    float glutamate = nt_conc[base + NT_GLUTAMATE];
    float gaba      = nt_conc[base + NT_GABA];
    float ach       = nt_conc[base + NT_ACETYLCHOLINE];

    // Compute receptor open fractions via Hill equation
    ampa_open[gid]  = hill(glutamate, AMPA_EC50,  AMPA_HILL);
    nmda_open[gid]  = hill(glutamate, NMDA_EC50,  NMDA_HILL);
    gabaa_open[gid] = hill(gaba,      GABAA_EC50, GABAA_HILL);
    nachr_open[gid] = hill(ach,       NACHR_EC50, NACHR_HILL);
}
