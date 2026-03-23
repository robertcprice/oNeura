// ============================================================================
// metabolism.metal — cellular bioenergetics
// ============================================================================

#include <metal_stdlib>
using namespace metal;

constant float GLYCOLYSIS_RATE = 0.002f;
constant float GLYCOLYSIS_ATP_YIELD = 2.0f;
constant float OXPHOS_RATE = 0.005f;
constant float OXPHOS_ATP_YIELD = 17.0f;
constant float O2_PER_PYRUVATE = 3.0f;
constant float BASELINE_ATP_CONSUMPTION = 0.5f;
constant float SPIKE_ATP_COST = 50.0f;
constant float CURRENT_ATP_COST = 0.01f;
constant float ADP_RECYCLING_RATE = 0.001f;
constant float MAX_PYRUVATE = 2000.0f;
constant float GLUCOSE_REPLENISH_RATE = 0.5f;
constant float O2_REPLENISH_RATE = 0.1f;
constant float ATP_EXCITABILITY_THRESHOLD = 1000.0f;
constant float ATP_RESTING = 5000.0f;
constant float ADP_RESTING = 500.0f;
constant float GLUCOSE_RESTING = 5000.0f;
constant float OXYGEN_RESTING = 100.0f;

struct Params {
    uint neuron_count;
    float dt;
};

kernel void metabolism(
    device float* atp               [[buffer(0)]],
    device float* adp               [[buffer(1)]],
    device float* glucose           [[buffer(2)]],
    device float* oxygen            [[buffer(3)]],
    device float* energy            [[buffer(4)]],
    device float* excitability_bias [[buffer(5)]],
    device const uchar* fired       [[buffer(6)]],
    device const float* external_current [[buffer(7)]],
    device const uchar* alive       [[buffer(8)]],
    constant Params& params         [[buffer(9)]],
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= params.neuron_count) return;
    if (alive[gid] == 0) return;

    float dt = params.dt;
    float atp_v = atp[gid];
    float adp_v = adp[gid];
    float glucose_v = glucose[gid];
    float oxygen_v = oxygen[gid];

    float glycolysis_flux = GLYCOLYSIS_RATE * glucose_v * dt;
    float glucose_consumed = min(glycolysis_flux, glucose_v);
    glucose_v -= glucose_consumed;
    float pyruvate_produced = glucose_consumed * 2.0f;
    float glycolysis_atp = glucose_consumed * GLYCOLYSIS_ATP_YIELD;
    atp_v += glycolysis_atp;
    adp_v = max(adp_v - glycolysis_atp, 0.0f);

    float pyruvate_available = min(pyruvate_produced, MAX_PYRUVATE);
    float oxphos_flux = OXPHOS_RATE * pyruvate_available * dt;
    float o2_needed = oxphos_flux * O2_PER_PYRUVATE;
    float oxphos_actual = (o2_needed > oxygen_v) ? (oxygen_v / O2_PER_PYRUVATE) : oxphos_flux;
    float oxphos_atp = oxphos_actual * OXPHOS_ATP_YIELD;
    atp_v += oxphos_atp;
    adp_v = max(adp_v - oxphos_atp, 0.0f);
    oxygen_v -= min(oxphos_actual * O2_PER_PYRUVATE, oxygen_v);

    float recycled = ADP_RECYCLING_RATE * adp_v * dt;
    atp_v += recycled;
    adp_v -= recycled;

    float baseline_cost = BASELINE_ATP_CONSUMPTION * dt;
    float spike_cost = (fired[gid] != 0) ? SPIKE_ATP_COST : 0.0f;
    float current_cost = fabs(external_current[gid]) * CURRENT_ATP_COST * dt;
    float total_cost = baseline_cost + spike_cost + current_cost;
    float consumed = min(total_cost, atp_v);
    atp_v -= consumed;
    adp_v += consumed;

    glucose_v = min(glucose_v + GLUCOSE_REPLENISH_RATE * dt, GLUCOSE_RESTING * 1.5f);
    oxygen_v = min(oxygen_v + O2_REPLENISH_RATE * dt, OXYGEN_RESTING * 1.5f);

    atp_v = clamp(atp_v, 0.0f, ATP_RESTING * 2.0f);
    adp_v = clamp(adp_v, 0.0f, ADP_RESTING * 5.0f);

    atp[gid] = atp_v;
    adp[gid] = adp_v;
    glucose[gid] = glucose_v;
    oxygen[gid] = oxygen_v;
    energy[gid] = clamp((atp_v / ATP_RESTING) * 100.0f, 0.0f, 200.0f);

    if (atp_v < ATP_EXCITABILITY_THRESHOLD) {
        float atp_fraction = atp_v / ATP_EXCITABILITY_THRESHOLD;
        excitability_bias[gid] = -5.0f * (1.0f - atp_fraction);
    } else {
        excitability_bias[gid] = 0.0f;
    }
}
