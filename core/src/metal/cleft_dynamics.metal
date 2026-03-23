// ============================================================================
// cleft_dynamics.metal — Synaptic cleft neurotransmitter dynamics
// ============================================================================
// Enzymatic degradation + diffusion loss + reuptake for active synapses.
// One thread per ACTIVE SYNAPSE (not per neuron).
//
// Half-life values (ms) by NT type:
//   0 = dopamine:      500
//   1 = serotonin:     300
//   2 = norepinephrine: 400
//   3 = acetylcholine:   2  (AChE is fast)
//   4 = GABA:          100
//   5 = glutamate:      10  (rapid glutamate clearance)
// ============================================================================

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Half-lives for each NT type (ms)
// Indexed by nt_type: 0=DA, 1=5HT, 2=NE, 3=ACh, 4=GABA, 5=Glu
// ---------------------------------------------------------------------------
constant float HALF_LIVES[6] = {
    500.0f,   // dopamine
    300.0f,   // serotonin
    400.0f,   // norepinephrine
      2.0f,   // acetylcholine  (AChE extremely fast)
    100.0f,   // GABA
     10.0f    // glutamate      (rapid transporter clearance)
};

constant float LN2 = 0.693147f;

// Diffusion and reuptake rate constants (per ms)
constant float DIFFUSION_RATE = 0.5f;   // fraction lost to diffusion per ms
constant float REUPTAKE_RATE  = 0.1f;   // fraction recaptured per ms

// ---------------------------------------------------------------------------
// Params
// ---------------------------------------------------------------------------
struct Params {
    uint  synapse_count;   // number of active synapses in this dispatch
    float dt;
};

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
kernel void cleft_dynamics(
    device       float* cleft_concentration     [[buffer(0)]],  // [synapse_count]
    device const float* nt_type_float           [[buffer(1)]],  // NT type as float (cast to uint)
    device const float* alive                   [[buffer(2)]],  // per-synapse active flag
    constant     Params& params                 [[buffer(3)]],
    uint         gid                            [[thread_position_in_grid]]
) {
    if (gid >= params.synapse_count) return;
    if (alive[gid] < 0.5f) return;

    float conc = cleft_concentration[gid];
    if (conc <= 0.0f) {
        cleft_concentration[gid] = 0.0f;
        return;
    }

    float dt = params.dt;

    // Determine NT type and corresponding half-life
    uint nt = uint(nt_type_float[gid]);
    nt = min(nt, 5u);  // clamp to valid range
    float half_life = HALF_LIVES[nt];

    // Enzymatic degradation: first-order decay with ln(2)/half_life rate
    float decay_rate = LN2 / half_life;
    conc -= conc * decay_rate * dt;

    // Diffusion out of cleft
    conc *= 1.0f - DIFFUSION_RATE * dt;

    // Reuptake by presynaptic terminal
    conc *= 1.0f - REUPTAKE_RATE * dt;

    // Clamp to non-negative
    cleft_concentration[gid] = max(0.0f, conc);
}
