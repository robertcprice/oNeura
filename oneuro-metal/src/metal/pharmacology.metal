// ============================================================================
// pharmacology.metal — per-neuron pharmacodynamic effects
// ============================================================================

#include <metal_stdlib>
using namespace metal;

constant uint DRUG_FLUOXETINE = 0u;
constant uint DRUG_DIAZEPAM = 1u;
constant uint DRUG_CAFFEINE = 2u;
constant uint DRUG_AMPHETAMINE = 3u;
constant uint DRUG_LDOPA = 4u;
constant uint DRUG_DONEPEZIL = 5u;
constant uint DRUG_KETAMINE = 6u;

constant uint CH_NAV = 0u;
constant uint CH_NMDA = 4u;
constant uint CH_GABAA = 6u;

constant uint NT_DOPAMINE = 0u;
constant uint NT_SEROTONIN = 1u;
constant uint NT_NOREPINEPHRINE = 2u;
constant uint NT_ACETYLCHOLINE = 3u;

constant uint CHANNEL_COUNT = 8u;
constant uint NT_COUNT = 6u;

constant float REST_DA = 20.0f;
constant float REST_5HT = 10.0f;
constant float REST_NE = 15.0f;
constant float REST_ACH = 50.0f;

struct Params {
    uint neuron_count;
    uint drug_type;
    float effect;
};

kernel void pharmacology(
    device float* conductance_scale [[buffer(0)]],
    device float* nt_conc           [[buffer(1)]],
    device float* external_current  [[buffer(2)]],
    device const uchar* alive       [[buffer(3)]],
    constant Params& params         [[buffer(4)]],
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= params.neuron_count) return;
    if (alive[gid] == 0) return;

    float effect = params.effect;
    uint ch_base = gid * CHANNEL_COUNT;
    uint nt_base = gid * NT_COUNT;

    switch (params.drug_type) {
        case DRUG_FLUOXETINE:
            nt_conc[nt_base + NT_SEROTONIN] = REST_5HT * (1.0f + effect * 3.0f);
            break;
        case DRUG_DIAZEPAM:
            conductance_scale[ch_base + CH_GABAA] = 1.0f + effect * 4.0f;
            break;
        case DRUG_CAFFEINE:
            conductance_scale[ch_base + CH_NAV] = 1.0f + effect * 0.3f;
            external_current[gid] += effect * 2.0f;
            break;
        case DRUG_AMPHETAMINE:
            nt_conc[nt_base + NT_DOPAMINE] = REST_DA * (1.0f + effect * 5.0f);
            nt_conc[nt_base + NT_NOREPINEPHRINE] = REST_NE * (1.0f + effect * 3.0f);
            break;
        case DRUG_LDOPA:
            nt_conc[nt_base + NT_DOPAMINE] = REST_DA * (1.0f + effect * 4.0f);
            break;
        case DRUG_DONEPEZIL:
            nt_conc[nt_base + NT_ACETYLCHOLINE] = REST_ACH * (1.0f + effect * 3.0f);
            break;
        case DRUG_KETAMINE:
            conductance_scale[ch_base + CH_NMDA] = max(1.0f - effect * 0.9f, 0.05f);
            break;
        default:
            break;
    }
}
