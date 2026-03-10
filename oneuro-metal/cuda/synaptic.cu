// synaptic.cu — Synaptic current injection + neurotransmitter release
// Compiled at runtime via NVRTC.
//
// Two kernels:
//   1. synaptic_current — One thread per synapse (CSR format), atomicAdd PSC to post neuron
//   2. nt_release       — One thread per neuron, release + decay neurotransmitter concentrations
//
// NT types: 0=DA, 1=5-HT, 2=NE, 3=ACh, 4=GABA, 5=Glu

#define NT_DA    0
#define NT_5HT   1
#define NT_NE    2
#define NT_ACH   3
#define NT_GABA  4
#define NT_GLU   5
#define NUM_NT   6

// Release amount in nM upon firing
#define NT_RELEASE_NM  3000.0f

// Decay half-lives in ms for each NT type
// DA ~200ms, 5-HT ~300ms, NE ~150ms, ACh ~5ms (fast enzymatic), GABA ~50ms, Glu ~10ms
__device__ const float nt_half_life[NUM_NT] = {
    200.0f, 300.0f, 150.0f, 5.0f, 50.0f, 10.0f
};

// ---------------------------------------------------------------------------
// Helper: find pre-neuron index from synapse index via CSR row_offsets
// Uses binary search on row_offsets to find which row contains synapse s.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int find_pre_neuron(
    const unsigned int* __restrict__ row_offsets,
    int s,
    int N)
{
    // Binary search: find largest row r such that row_offsets[r] <= s
    int lo = 0;
    int hi = N;  // row_offsets has N+1 entries, valid rows are 0..N-1
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
// Kernel: synaptic_current
// One thread per synapse. Injects post-synaptic current when pre neuron fired.
// ---------------------------------------------------------------------------
extern "C" __global__ void synaptic_current(
    const unsigned int* __restrict__ row_offsets,
    const unsigned int* __restrict__ col_indices,
    const unsigned char* __restrict__ nt_type,
    const float* __restrict__ weight,
    float* __restrict__ ext_current,
    const unsigned char* __restrict__ pre_fired,
    float psc_scale,
    int N,
    int S)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= S) return;

    // Find which pre-neuron owns this synapse
    int pre = find_pre_neuron(row_offsets, s, N);

    // Only transmit if pre-neuron fired
    if (!pre_fired[pre]) return;

    // Post-synaptic target
    unsigned int post = col_indices[s];

    // Sign: GABA (type 4) is inhibitory (-1), all others excitatory (+1)
    float sign = (nt_type[s] == NT_GABA) ? -1.0f : 1.0f;

    // Post-synaptic current
    float psc = weight[s] * psc_scale * sign;

    // Atomic accumulation into post-synaptic neuron's external current
    atomicAdd(&ext_current[post], psc);
}

// ---------------------------------------------------------------------------
// Archetype-to-NT mapping
// Maps neuron archetype to its primary neurotransmitter.
// Archetype encoding (matches Python-side):
//   0 = excitatory glutamatergic -> Glu (5)
//   1 = inhibitory GABAergic     -> GABA (4)
//   2 = dopaminergic             -> DA (0)
//   3 = serotonergic             -> 5-HT (1)
//   4 = noradrenergic            -> NE (2)
//   5 = cholinergic              -> ACh (3)
//   default                      -> Glu (5)
// ---------------------------------------------------------------------------
__device__ __forceinline__ int archetype_to_nt(unsigned char arch)
{
    switch (arch) {
        case 0: return NT_GLU;
        case 1: return NT_GABA;
        case 2: return NT_DA;
        case 3: return NT_5HT;
        case 4: return NT_NE;
        case 5: return NT_ACH;
        default: return NT_GLU;
    }
}

// ---------------------------------------------------------------------------
// Kernel: nt_release
// One thread per neuron. Releases NT on spike, decays all concentrations.
// nt_conc layout: SoA — nt_conc[nt_idx * N + neuron_idx]
// ---------------------------------------------------------------------------
extern "C" __global__ void nt_release(
    float* __restrict__ nt_conc,
    const unsigned char* __restrict__ fired,
    const unsigned char* __restrict__ archetype,
    int N,
    float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // --- Release primary NT if neuron fired ---
    if (fired[i]) {
        int nt = archetype_to_nt(archetype[i]);
        nt_conc[nt * N + i] += NT_RELEASE_NM;
    }

    // --- Exponential decay of all NT concentrations ---
    for (int nt = 0; nt < NUM_NT; nt++) {
        int idx = nt * N + i;
        float conc = nt_conc[idx];
        if (conc > 0.0f) {
            // decay_factor = exp(-dt * ln(2) / half_life)
            float decay = expf(-dt * 0.693147f / nt_half_life[nt]);
            nt_conc[idx] = conc * decay;
            // Snap to zero below threshold to avoid denormals
            if (nt_conc[idx] < 0.01f) {
                nt_conc[idx] = 0.0f;
            }
        }
    }
}
