// ============================================================================
// glia.metal — astrocytes, oligodendrocytes, microglia
// ============================================================================

#include <metal_stdlib>
using namespace metal;

constant float ASTROCYTE_UPTAKE_RATE = 0.1f;
constant float ASTROCYTE_LACTATE_RESTING = 2000.0f;
constant float ASTROCYTE_LACTATE_MAX = 5000.0f;
constant float LACTATE_SUPPLY_RATE = 0.01f;
constant float LACTATE_TO_GLUCOSE = 0.3f;
constant float LACTATE_REPLENISH_RATE = 0.005f;
constant float UPTAKE_FATIGUE_THRESHOLD = 2000.0f;
constant float UPTAKE_FATIGUE_FACTOR = 0.99f;
constant float UPTAKE_RECOVERY_RATE = 0.001f;
constant float MYELIN_RECOVERY_RATE = 0.0001f;
constant float MICROGLIA_DAMAGE_THRESHOLD = 50.0f;
constant float MICROGLIA_ACTIVATION_RATE = 0.01f;
constant float MICROGLIA_DEACTIVATION_RATE = 0.005f;
constant float MICROGLIA_PRUNING_ACTIVATION = 0.5f;
constant float DAMAGE_DECAY_FACTOR = 0.99f;
constant float ATP_STRESS_THRESHOLD = 2000.0f;
constant uint NT_GLU = 5u;
constant uint NT_COUNT = 6u;
constant float GLU_REST = 500.0f;

struct GliaNeuronParams {
    uint neuron_count;
    float dt_ms;
};

struct GliaPruneParams {
    uint synapse_count;
};

kernel void glia_neuron_step(
    device float* astrocyte_uptake       [[buffer(0)]],
    device float* astrocyte_lactate      [[buffer(1)]],
    device float* myelin_integrity       [[buffer(2)]],
    device float* microglia_activation   [[buffer(3)]],
    device float* damage_signal          [[buffer(4)]],
    device float* nt_conc                [[buffer(5)]],
    device float* glucose                [[buffer(6)]],
    device const float* atp              [[buffer(7)]],
    device const float* voltage          [[buffer(8)]],
    device const uchar* alive            [[buffer(9)]],
    constant GliaNeuronParams& params    [[buffer(10)]],
    uint gid                             [[thread_position_in_grid]]
) {
    if (gid >= params.neuron_count) return;
    if (alive[gid] == 0) return;

    float dt = params.dt_ms;
    float dt_s = dt / 1000.0f;
    uint nt_base = gid * NT_COUNT;

    float glu = nt_conc[nt_base + NT_GLU];
    float uptake_rate = astrocyte_uptake[gid] * ASTROCYTE_UPTAKE_RATE;
    float glu_removed = glu * uptake_rate * dt;
    nt_conc[nt_base + NT_GLU] = max(glu - glu_removed, GLU_REST);

    float lactate = astrocyte_lactate[gid];
    lactate += glu_removed * 0.5f;
    float supply = min(lactate * LACTATE_SUPPLY_RATE * dt_s, lactate);
    lactate -= supply;
    glucose[gid] += supply * LACTATE_TO_GLUCOSE;
    lactate += (ASTROCYTE_LACTATE_RESTING - lactate) * LACTATE_REPLENISH_RATE * dt_s;
    astrocyte_lactate[gid] = clamp(lactate, 0.0f, ASTROCYTE_LACTATE_MAX);

    float uptake = astrocyte_uptake[gid];
    uptake += (1.0f - uptake) * UPTAKE_RECOVERY_RATE * dt_s;
    if (glu > UPTAKE_FATIGUE_THRESHOLD) {
        uptake *= UPTAKE_FATIGUE_FACTOR;
    }
    astrocyte_uptake[gid] = clamp(uptake, 0.1f, 1.0f);

    myelin_integrity[gid] = clamp(myelin_integrity[gid] + MYELIN_RECOVERY_RATE * dt_s, 0.0f, 1.0f);

    float atp_stress = (atp[gid] < ATP_STRESS_THRESHOLD) ? 1.0f : 0.0f;
    float v = voltage[gid];
    float v_stress = (v > 0.0f || v < -90.0f) ? 1.0f : 0.0f;
    float damage = (damage_signal[gid] + (atp_stress + v_stress) * dt_s) * DAMAGE_DECAY_FACTOR;
    damage_signal[gid] = damage;

    float activation = microglia_activation[gid];
    if (damage > MICROGLIA_DAMAGE_THRESHOLD) {
        activation += MICROGLIA_ACTIVATION_RATE * dt_s;
    } else {
        activation -= MICROGLIA_DEACTIVATION_RATE * dt_s;
    }
    microglia_activation[gid] = clamp(activation, 0.0f, 1.0f);
}

kernel void glia_prune_synapses(
    device const uint* col_indices       [[buffer(0)]],
    device float* weight                 [[buffer(1)]],
    device float* strength               [[buffer(2)]],
    device const ushort* ampa_receptors  [[buffer(3)]],
    device const ushort* nmda_receptors  [[buffer(4)]],
    device const ushort* gabaa_receptors [[buffer(5)]],
    device const float* microglia_activation [[buffer(6)]],
    constant GliaPruneParams& params     [[buffer(7)]],
    uint gid                             [[thread_position_in_grid]]
) {
    if (gid >= params.synapse_count) return;

    uint post = col_indices[gid];
    uint total = uint(ampa_receptors[gid]) + uint(nmda_receptors[gid]) + uint(gabaa_receptors[gid]);
    bool weak = strength[gid] < 0.1f || total < 5u;
    if (weak && microglia_activation[post] > MICROGLIA_PRUNING_ACTIVATION) {
        weight[gid] = 0.0f;
        strength[gid] = 0.0f;
    }
}
