// ============================================================================
// synapse_step.metal — GPU-resident synapse-side dynamics
// ============================================================================
// Handles per-synapse release, cleft decay, vesicle replenishment, STDP, and
// fixed-point accumulation of next-step synaptic current.
// ============================================================================

#include <metal_stdlib>
using namespace metal;

constant float VESICLE_BASE_RELEASE_PROB = 0.3f;
constant float VESICLE_NT_PER_RELEASE_NM = 3000.0f;
constant float VESICLE_RRP_MAX = 10.0f;
constant float VESICLE_RECYCLING_MAX = 50.0f;
constant float VESICLE_RESERVE_MAX = 200.0f;
constant float VESICLE_RRP_REFILL_RATE = 0.05f;
constant float VESICLE_RECYCLING_REFILL_RATE = 0.01f;

constant float CLEFT_DIFFUSION_RATE = 0.5f;
constant float CLEFT_REUPTAKE_RATE = 0.1f;
constant float STDP_WINDOW_MS = 20.0f;
constant float STDP_LTP_RATE = 0.5f;
constant float STDP_LTD_RATE = 0.5f;
constant float NMDA_MG_CONC_MM = 1.0f;
constant float TAG_THRESHOLD = 0.3f;
constant float TAG_DECAY_TAU_MS = 60000.0f;
constant float BCM_THETA_TAU_MS = 10000.0f;
constant float AMBIENT_NT_EXPOSURE_GAIN = 0.02f;

constant uint NT_GABA = 4u;
constant uint NT_COUNT = 6u;
constant float HALF_LIVES[6] = {500.0f, 300.0f, 400.0f, 2.0f, 100.0f, 10.0f};
constant float LN2 = 0.693147f;

struct SynapseParams {
    uint synapse_count;
    float time_ms;
    float dt;
    float psc_scale;
    float current_scale;
    float nt_scale;
};

struct CurrentCommitParams {
    uint neuron_count;
    float current_scale;
};

struct NtCommitParams {
    uint slot_count;
    float nt_scale;
};

struct LastFiredParams {
    uint neuron_count;
    uint step_index;
};

struct ClearIntParams {
    uint count;
};

static inline float nmda_mg_block(float voltage) {
    return 1.0f / (1.0f + NMDA_MG_CONC_MM * exp(-0.062f * voltage) / 3.57f);
}

static inline float stdp_kernel(float delta_t, float tau) {
    return exp(-fabs(delta_t) / tau);
}

static inline float bcm_modulation(float post_activity, float theta) {
    if (theta < 1e-6f) return 1.0f;
    return clamp(post_activity / theta, 0.0f, 3.0f);
}

static inline void recompute_weight(
    device float* weight,
    device const float* strength,
    device const float* homeostatic_scale,
    device const ushort* ampa_receptors,
    device const ushort* nmda_receptors,
    device const ushort* gabaa_receptors,
    uint gid
) {
    float total = float(ampa_receptors[gid] + nmda_receptors[gid] + gabaa_receptors[gid]);
    weight[gid] = min(total / 50.0f, 2.0f) * strength[gid] * homeostatic_scale[gid];
}

static inline void apply_receptor_change(
    device ushort* ampa_receptors,
    device ushort* nmda_receptors,
    device ushort* gabaa_receptors,
    device float* weight,
    device const float* strength,
    device const float* homeostatic_scale,
    uint gid,
    float delta
) {
    int ampa_change = int(round(delta * 2.0f));
    int new_ampa = clamp(int(ampa_receptors[gid]) + ampa_change, 1, 200);
    ampa_receptors[gid] = ushort(new_ampa);

    if (fabs(delta) > 0.3f) {
        int nmda_change = int(round(delta * 0.2f));
        int new_nmda = clamp(int(nmda_receptors[gid]) + nmda_change, 1, 100);
        nmda_receptors[gid] = ushort(new_nmda);
    }

    recompute_weight(
        weight,
        strength,
        homeostatic_scale,
        ampa_receptors,
        nmda_receptors,
        gabaa_receptors,
        gid
    );
}

kernel void clear_i32_buffer(
    device int* values                  [[buffer(0)]],
    constant ClearIntParams& params     [[buffer(1)]],
    uint gid                            [[thread_position_in_grid]]
) {
    if (gid >= params.count) return;
    values[gid] = 0;
}

kernel void synapse_step(
    device const uint* pre_indices             [[buffer(0)]],
    device const uint* col_indices             [[buffer(1)]],
    device const uchar* nt_type                [[buffer(2)]],
    device float* weight                       [[buffer(3)]],
    device const float* strength               [[buffer(4)]],
    device float* vesicle_rrp                  [[buffer(5)]],
    device float* vesicle_recycling            [[buffer(6)]],
    device float* vesicle_reserve              [[buffer(7)]],
    device float* cleft_concentration          [[buffer(8)]],
    device ushort* ampa_receptors              [[buffer(9)]],
    device ushort* nmda_receptors              [[buffer(10)]],
    device ushort* gabaa_receptors             [[buffer(11)]],
    device float* last_pre_spike               [[buffer(12)]],
    device float* last_post_spike              [[buffer(13)]],
    device float* eligibility_trace            [[buffer(14)]],
    device float* bcm_theta                    [[buffer(15)]],
    device float* post_activity_history        [[buffer(16)]],
    device uchar* tagged                       [[buffer(17)]],
    device float* tag_strength                 [[buffer(18)]],
    device const float* homeostatic_scale      [[buffer(19)]],
    device const uchar* fired                  [[buffer(20)]],
    device const uchar* alive                  [[buffer(21)]],
    device const float* voltage                [[buffer(22)]],
    device const float* ca_microdomain         [[buffer(23)]],
    device const float* nt_conc                [[buffer(24)]],
    device atomic_int* nt_conc_accum           [[buffer(25)]],
    device atomic_int* synaptic_current_accum  [[buffer(26)]],
    constant SynapseParams& params             [[buffer(27)]],
    uint gid                                   [[thread_position_in_grid]]
) {
    if (gid >= params.synapse_count) return;

    uint pre = pre_indices[gid];
    uint post = col_indices[gid];
    bool pre_fired = fired[pre] != 0;
    bool post_fired = fired[post] != 0;
    bool post_alive = alive[post] != 0;

    float time_ms = params.time_ms;
    float dt = params.dt;

    if (pre_fired) {
        float ca_factor = min(ca_microdomain[pre] / 500.0f, 2.0f);
        float release_prob = VESICLE_BASE_RELEASE_PROB * ca_factor;
        float released = min(vesicle_rrp[gid] * release_prob, vesicle_rrp[gid]);
        vesicle_rrp[gid] -= released;
        cleft_concentration[gid] += released * VESICLE_NT_PER_RELEASE_NM;

        if (post_alive) {
            float sign = (nt_type[gid] == NT_GABA) ? -1.0f : 1.0f;
            int delta = int(round(sign * weight[gid] * params.psc_scale * params.current_scale));
            if (delta != 0) {
                atomic_fetch_add_explicit(
                    &synaptic_current_accum[post],
                    delta,
                    memory_order_relaxed
                );
            }

            float delta_t_ltd = time_ms - last_post_spike[gid];
            if (delta_t_ltd > 0.0f && delta_t_ltd < STDP_WINDOW_MS) {
                float raw_ltd = -STDP_LTD_RATE * stdp_kernel(delta_t_ltd, STDP_WINDOW_MS);
                float ltd = raw_ltd * nmda_mg_block(voltage[post]);
                float bcm_factor = bcm_modulation(post_activity_history[gid], bcm_theta[gid]);
                float adjusted_ltd = ltd * max(1.0f - bcm_factor, 0.1f);
                apply_receptor_change(
                    ampa_receptors,
                    nmda_receptors,
                    gabaa_receptors,
                    weight,
                    strength,
                    homeostatic_scale,
                    gid,
                    adjusted_ltd
                );
                if (fabs(adjusted_ltd) > TAG_THRESHOLD) {
                    tagged[gid] = 1;
                    tag_strength[gid] = fabs(adjusted_ltd);
                }
            }
        }

        last_pre_spike[gid] = time_ms;
    }

    if (post_fired && post_alive) {
        float delta_t_ltp = time_ms - last_pre_spike[gid];
        if (delta_t_ltp > 0.0f && delta_t_ltp < STDP_WINDOW_MS) {
            float raw_ltp = STDP_LTP_RATE * stdp_kernel(delta_t_ltp, STDP_WINDOW_MS);
            float ltp = raw_ltp * nmda_mg_block(voltage[post]);
            float bcm_factor = bcm_modulation(post_activity_history[gid], bcm_theta[gid]);
            float adjusted_ltp = ltp * max(bcm_factor, 0.1f);
            apply_receptor_change(
                ampa_receptors,
                nmda_receptors,
                gabaa_receptors,
                weight,
                strength,
                homeostatic_scale,
                gid,
                adjusted_ltp
            );
            if (fabs(adjusted_ltp) > TAG_THRESHOLD) {
                tagged[gid] = 1;
                tag_strength[gid] = fabs(adjusted_ltp);
            }
        }

        last_post_spike[gid] = time_ms;
        float alpha = min(dt / BCM_THETA_TAU_MS, 1.0f);
        float post_active = 1.0f;
        float history = post_activity_history[gid] * (1.0f - alpha) + post_active * alpha;
        post_activity_history[gid] = history;
        bcm_theta[gid] = history * history;
    }

    eligibility_trace[gid] *= exp(-dt / STDP_WINDOW_MS);

    if (tagged[gid] != 0) {
        tag_strength[gid] *= exp(-dt / TAG_DECAY_TAU_MS);
        if (tag_strength[gid] < 0.01f) {
            tagged[gid] = 0;
            tag_strength[gid] = 0.0f;
        }
    }

    float conc = cleft_concentration[gid];
    if (conc > 0.0f) {
        uint nt = min(uint(nt_type[gid]), 5u);
        float decay_rate = LN2 / HALF_LIVES[nt];
        conc -= conc * decay_rate * dt;
        conc *= 1.0f - CLEFT_DIFFUSION_RATE * dt;
        conc *= 1.0f - CLEFT_REUPTAKE_RATE * dt;
        cleft_concentration[gid] = max(0.0f, conc);
        if (post_alive) {
            float ambient_delta = conc * AMBIENT_NT_EXPOSURE_GAIN * max(dt, 0.0f);
            int nt_delta = int(round(ambient_delta * params.nt_scale));
            if (nt_delta != 0) {
                atomic_fetch_add_explicit(
                    &nt_conc_accum[post * NT_COUNT + nt],
                    nt_delta,
                    memory_order_relaxed
                );
            }
        }
    } else {
        cleft_concentration[gid] = 0.0f;
    }

    float to_recycling = max(
        min(min(VESICLE_RECYCLING_REFILL_RATE * dt, vesicle_reserve[gid]), VESICLE_RECYCLING_MAX - vesicle_recycling[gid]),
        0.0f
    );
    vesicle_reserve[gid] -= to_recycling;
    vesicle_recycling[gid] += to_recycling;

    float to_rrp = max(
        min(min(VESICLE_RRP_REFILL_RATE * dt, vesicle_recycling[gid]), VESICLE_RRP_MAX - vesicle_rrp[gid]),
        0.0f
    );
    vesicle_recycling[gid] -= to_rrp;
    vesicle_rrp[gid] += to_rrp;
}

kernel void commit_synaptic_current(
    device atomic_int* synaptic_current_accum [[buffer(0)]],
    device float* synaptic_current            [[buffer(1)]],
    constant CurrentCommitParams& params      [[buffer(2)]],
    uint gid                                  [[thread_position_in_grid]]
) {
    if (gid >= params.neuron_count) return;
    int current = atomic_exchange_explicit(
        &synaptic_current_accum[gid],
        0,
        memory_order_relaxed
    );
    synaptic_current[gid] += float(current) / params.current_scale;
}

kernel void commit_nt_concentration(
    device atomic_int* nt_conc_accum     [[buffer(0)]],
    device float* nt_conc                [[buffer(1)]],
    constant NtCommitParams& params      [[buffer(2)]],
    uint gid                             [[thread_position_in_grid]]
) {
    if (gid >= params.slot_count) return;
    int delta = atomic_exchange_explicit(
        &nt_conc_accum[gid],
        0,
        memory_order_relaxed
    );
    nt_conc[gid] += float(delta) / params.nt_scale;
}

kernel void mark_last_fired(
    device const uchar* fired            [[buffer(0)]],
    device uint* last_fired_step         [[buffer(1)]],
    constant LastFiredParams& params     [[buffer(2)]],
    uint gid                             [[thread_position_in_grid]]
) {
    if (gid >= params.neuron_count) return;
    if (fired[gid] != 0) {
        last_fired_step[gid] = params.step_index;
    }
}
