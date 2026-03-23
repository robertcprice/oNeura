// ============================================================================
// microtubules.metal — Orch-OR coherence dynamics
// ============================================================================

#include <metal_stdlib>
using namespace metal;

constant float COHERENCE_GROWTH_RATE = 0.002f;
constant float THERMAL_DECOHERENCE_RATE = 0.001f;
constant float ACTIVITY_DECOHERENCE_RATE = 0.02f;
constant float OR_THRESHOLD = 0.6f;
constant float POST_COLLAPSE_COHERENCE = 0.1f;
constant float CA_OPTIMAL_NM = 500.0f;
constant float ATP_COHERENCE_THRESHOLD = 2000.0f;

struct Params {
    uint neuron_count;
    float dt;
};

kernel void microtubules(
    device float* mt_coherence       [[buffer(0)]],
    device uint* orch_or_events      [[buffer(1)]],
    device const float* ca_cytoplasmic [[buffer(2)]],
    device const float* atp          [[buffer(3)]],
    device const uchar* fired        [[buffer(4)]],
    device const uchar* alive        [[buffer(5)]],
    constant Params& params          [[buffer(6)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= params.neuron_count) return;
    if (alive[gid] == 0) return;

    float dt = params.dt;
    float coherence = mt_coherence[gid];

    float ca = ca_cytoplasmic[gid];
    float ca_stability = 1.0f;
    if (ca < CA_OPTIMAL_NM) {
        ca_stability = 0.5f + 0.5f * (ca / CA_OPTIMAL_NM);
    } else if (ca >= 10000.0f) {
        ca_stability = max(10000.0f / ca, 0.1f);
    }

    float atp_v = atp[gid];
    float atp_stability = (atp_v > ATP_COHERENCE_THRESHOLD)
        ? 1.0f
        : max(atp_v / ATP_COHERENCE_THRESHOLD, 0.1f);

    float stability = ca_stability * atp_stability;
    float growth = COHERENCE_GROWTH_RATE * (1.0f - coherence) * stability * dt;
    float thermal = THERMAL_DECOHERENCE_RATE * coherence * dt;
    float activity = (fired[gid] != 0) ? (ACTIVITY_DECOHERENCE_RATE * coherence * dt) : 0.0f;

    coherence = clamp(coherence + growth - thermal - activity, 0.0f, 1.0f);

    if (coherence >= OR_THRESHOLD) {
        orch_or_events[gid] += 1u;
        coherence = POST_COLLAPSE_COHERENCE;
    }

    mt_coherence[gid] = coherence;
}
