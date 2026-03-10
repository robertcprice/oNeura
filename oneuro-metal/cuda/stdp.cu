// stdp.cu — Spike-Timing-Dependent Plasticity trace update and weight modification
// One thread per synapse. Compiled at runtime via NVRTC.
//
// Implements symmetric STDP (a_minus ~ a_plus) with dopamine modulation.
// Weight domain: [0, 2] — clamped after update.
//
// CSR format: row_offsets[N+1], col_indices[S], weight[S]
//   Pre-neuron for synapse s is found via binary search of row_offsets.
//   Post-neuron for synapse s is col_indices[s].

#define WEIGHT_MIN 0.0f
#define WEIGHT_MAX 2.0f

// ---------------------------------------------------------------------------
// Helper: find pre-neuron index from synapse index via CSR row_offsets
// Binary search: find largest row r such that row_offsets[r] <= s
// ---------------------------------------------------------------------------
__device__ __forceinline__ int find_pre_neuron_stdp(
    const unsigned int* __restrict__ row_offsets,
    int s,
    int N)
{
    int lo = 0;
    int hi = N;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (row_offsets[mid] <= (unsigned int)s) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

// ---------------------------------------------------------------------------
// Clamp helper
// ---------------------------------------------------------------------------
__device__ __forceinline__ float clampf(float x, float lo, float hi)
{
    return fminf(fmaxf(x, lo), hi);
}

// ---------------------------------------------------------------------------
// Kernel: stdp_trace_update
// One thread per synapse. Updates eligibility traces and modifies weights.
//
// Algorithm per synapse s (pre -> post):
//   1. Decay traces: pre_trace *= exp(-dt/tau_pre), post_trace *= exp(-dt/tau_post)
//   2. If pre fired:  pre_trace  += 1.0
//                     weight     -= a_minus * post_trace * (1 + da_level[post])   [LTD]
//   3. If post fired: post_trace += 1.0
//                     weight     += a_plus  * pre_trace  * (1 + da_level[pre])    [LTP]
//   4. Clamp weight to [0, 2]
//
// DA modulation: dopamine concentration at the relevant neuron scales the
// plasticity magnitude. This implements reward-modulated STDP where DA
// acts as a neuromodulatory third factor.
// ---------------------------------------------------------------------------
extern "C" __global__ void stdp_trace_update(
    float* __restrict__ pre_trace,
    float* __restrict__ post_trace,
    float* __restrict__ weight,
    const unsigned int* __restrict__ col_indices,
    const unsigned int* __restrict__ row_offsets,
    const unsigned char* __restrict__ pre_fired,
    const unsigned char* __restrict__ post_fired,
    const float* __restrict__ da_level,
    float tau_pre,
    float tau_post,
    float a_plus,
    float a_minus,
    float dt,
    int N,
    int S)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= S) return;

    // Identify pre and post neurons for this synapse
    int pre  = find_pre_neuron_stdp(row_offsets, s, N);
    int post = (int)col_indices[s];

    // --- Decay eligibility traces ---
    float decay_pre  = expf(-dt / tau_pre);
    float decay_post = expf(-dt / tau_post);

    float ptr = pre_trace[s]  * decay_pre;
    float otr = post_trace[s] * decay_post;

    float w = weight[s];

    // --- Pre-synaptic spike: LTD ---
    if (pre_fired[pre]) {
        ptr += 1.0f;
        // LTD: depression proportional to post trace (how recently post fired)
        // DA modulation at post-synaptic site
        float da_mod = 1.0f + da_level[post];
        w -= a_minus * otr * da_mod;
    }

    // --- Post-synaptic spike: LTP ---
    if (post_fired[post]) {
        otr += 1.0f;
        // LTP: potentiation proportional to pre trace (how recently pre fired)
        // DA modulation at pre-synaptic site
        float da_mod = 1.0f + da_level[pre];
        w += a_plus * ptr * da_mod;
    }

    // --- Clamp weight ---
    w = clampf(w, WEIGHT_MIN, WEIGHT_MAX);

    // --- Store results ---
    pre_trace[s]  = ptr;
    post_trace[s] = otr;
    weight[s]     = w;
}
