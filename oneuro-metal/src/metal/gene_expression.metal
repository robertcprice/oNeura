// ============================================================================
// gene_expression.metal — activity-dependent transcription
// ============================================================================

#include <metal_stdlib>
using namespace metal;

constant float CFOS_INDUCTION_RATE = 0.0001f;
constant float CFOS_CA_THRESHOLD_NM = 200.0f;
constant float CFOS_DECAY_TAU_MS = 1800000.0f;
constant float ARC_INDUCTION_RATE = 0.00005f;
constant float ARC_CREB_THRESHOLD = 0.1f;
constant float ARC_DECAY_TAU_MS = 3600000.0f;
constant float BDNF_INDUCTION_RATE = 0.00001f;
constant float BDNF_CREB_THRESHOLD = 0.2f;
constant float BDNF_DECAY_TAU_MS = 43200000.0f;

struct Params {
    uint neuron_count;
    float dt;
};

kernel void gene_expression(
    device const float* ca_cytoplasmic [[buffer(0)]],
    device const float* creb_p         [[buffer(1)]],
    device float* cfos_level           [[buffer(2)]],
    device float* arc_level            [[buffer(3)]],
    device float* bdnf_level           [[buffer(4)]],
    device const uchar* alive          [[buffer(5)]],
    constant Params& params            [[buffer(6)]],
    uint gid                           [[thread_position_in_grid]]
) {
    if (gid >= params.neuron_count) return;
    if (alive[gid] == 0) return;

    float dt = params.dt;

    float ca = ca_cytoplasmic[gid];
    float cfos = cfos_level[gid];
    float ca_drive = max(ca - CFOS_CA_THRESHOLD_NM, 0.0f);
    float cfos_induction = CFOS_INDUCTION_RATE * ca_drive * (1.0f - cfos);
    float cfos_decay = cfos * dt / CFOS_DECAY_TAU_MS;
    cfos += max(cfos_induction * dt - cfos_decay, -cfos);
    cfos_level[gid] = clamp(cfos, 0.0f, 1.0f);

    float creb = creb_p[gid];
    float arc = arc_level[gid];
    float creb_drive_arc = max(creb - ARC_CREB_THRESHOLD, 0.0f);
    float arc_induction = ARC_INDUCTION_RATE * creb_drive_arc * (1.0f - arc);
    float arc_decay = arc * dt / ARC_DECAY_TAU_MS;
    arc += max(arc_induction * dt - arc_decay, -arc);
    arc_level[gid] = clamp(arc, 0.0f, 1.0f);

    float bdnf = bdnf_level[gid];
    float creb_drive_bdnf = max(creb - BDNF_CREB_THRESHOLD, 0.0f);
    float bdnf_induction = BDNF_INDUCTION_RATE * creb_drive_bdnf * (1.0f - bdnf);
    float bdnf_decay = bdnf * dt / BDNF_DECAY_TAU_MS;
    bdnf += max(bdnf_induction * dt - bdnf_decay, -bdnf);
    bdnf_level[gid] = clamp(bdnf, 0.0f, 1.0f);
}
