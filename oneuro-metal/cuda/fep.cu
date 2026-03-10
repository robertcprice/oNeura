// fep.cu — Free Energy Principle stimulation protocol kernels
// Compiled at runtime via NVRTC.
//
// Implements the FEP experimental paradigm where:
//   HIT  = structured pulsed stimulation (low entropy, predictable)
//   MISS = random noise stimulation (high entropy, unpredictable)
//
// The organism learns to minimize prediction error (free energy) by
// producing behaviors that result in structured (low-entropy) feedback.
//
// Two kernels:
//   1. fep_structured_stim — Deterministic patterned stimulation (HIT response)
//   2. fep_noise_stim      — Random noise stimulation (MISS response)

// ---------------------------------------------------------------------------
// Kernel: fep_structured_stim
// One thread per target neuron. Injects amplitude-scaled patterned current.
//
// This represents the low-entropy feedback the organism receives on a HIT:
// structured, predictable stimulation that the neural network can learn to predict.
// ---------------------------------------------------------------------------
extern "C" __global__ void fep_structured_stim(
    float* __restrict__ ext_current,
    const unsigned int* __restrict__ target_neurons,
    const float* __restrict__ stim_pattern,
    float amplitude,
    int T)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    unsigned int neuron_idx = target_neurons[t];
    float current = amplitude * stim_pattern[t];

    // Atomic add because ext_current may already have synaptic contributions
    atomicAdd(&ext_current[neuron_idx], current);
}

// ---------------------------------------------------------------------------
// Simple LCG (Linear Congruential Generator) for per-thread RNG
// Parameters from Numerical Recipes: a=1664525, c=1013904223
// Returns float in [0, 1)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float lcg_random(unsigned int* state)
{
    *state = (*state) * 1664525u + 1013904223u;
    // Convert to float in [0, 1) via upper bits
    return (float)(*state >> 8) / 16777216.0f;  // 2^24
}

// ---------------------------------------------------------------------------
// Kernel: fep_noise_stim
// One thread per target neuron. Injects random noise current.
//
// This represents the high-entropy feedback the organism receives on a MISS:
// random, unpredictable stimulation that cannot be modeled or anticipated.
// The contrast between structured (HIT) and random (MISS) drives FEP learning.
//
// RNG: Each thread gets a unique seed derived from the global seed + thread index.
// Uses LCG for speed (this is stimulation noise, not cryptographic randomness).
// ---------------------------------------------------------------------------
extern "C" __global__ void fep_noise_stim(
    float* __restrict__ ext_current,
    const unsigned int* __restrict__ target_neurons,
    float amplitude,
    unsigned int seed,
    int T)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    // Per-thread RNG state: combine global seed with thread index
    // Use a hash-like mixing to avoid correlated sequences across threads
    unsigned int state = seed ^ (unsigned int)t;
    state = state * 2654435761u;  // Knuth multiplicative hash
    state ^= (state >> 16);

    // Generate random value in [-0.5, 0.5)
    float rand_val = lcg_random(&state) - 0.5f;

    unsigned int neuron_idx = target_neurons[t];
    float current = amplitude * rand_val;

    atomicAdd(&ext_current[neuron_idx], current);
}
