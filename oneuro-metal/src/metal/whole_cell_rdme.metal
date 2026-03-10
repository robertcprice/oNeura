// ============================================================================
// whole_cell_rdme.metal — intracellular reaction-diffusion for whole-cell state
// ============================================================================

#include <metal_stdlib>
using namespace metal;

constant uint SPECIES_COUNT = 4;

struct Params {
    uint  x_dim;
    uint  y_dim;
    uint  z_dim;
    float voxel_size_nm;
    float dt;
    float metabolic_load;
};

inline float diffusion_coeff(uint species) {
    switch (species) {
        case 0: return 60000.0f;
        case 1: return 40000.0f;
        case 2: return 28000.0f;
        default: return 10000.0f;
    }
}

inline float basal_source(uint species) {
    switch (species) {
        case 0: return 0.012f;
        case 1: return 0.006f;
        case 2: return 0.004f;
        default: return 0.002f;
    }
}

inline float basal_sink(uint species) {
    switch (species) {
        case 0: return 0.010f;
        case 1: return 0.005f;
        case 2: return 0.003f;
        default: return 0.0015f;
    }
}

kernel void whole_cell_rdme_kernel(
    device const float* grid_in  [[buffer(0)]],
    device       float* grid_out [[buffer(1)]],
    constant     Params& params  [[buffer(2)]],
    uint         gid             [[thread_position_in_grid]]
) {
    uint X = params.x_dim;
    uint Y = params.y_dim;
    uint Z = params.z_dim;
    uint total_voxels = X * Y * Z;

    if (gid >= total_voxels) return;

    uint z = gid / (Y * X);
    uint rem = gid - z * Y * X;
    uint y = rem / X;
    uint x = rem - y * X;

    float dx2 = params.voxel_size_nm * params.voxel_size_nm;
    float load = max(params.metabolic_load, 0.1f);

    for (uint species = 0; species < SPECIES_COUNT; species++) {
        uint base = species * total_voxels;
        uint idx = base + gid;
        float c = grid_in[idx];

        float right = (x + 1 < X) ? grid_in[base + z * Y * X + y * X + (x + 1)] : c;
        float left  = (x > 0)     ? grid_in[base + z * Y * X + y * X + (x - 1)] : c;
        float up    = (y + 1 < Y) ? grid_in[base + z * Y * X + (y + 1) * X + x] : c;
        float down  = (y > 0)     ? grid_in[base + z * Y * X + (y - 1) * X + x] : c;
        float front = (z + 1 < Z) ? grid_in[base + (z + 1) * Y * X + y * X + x] : c;
        float back  = (z > 0)     ? grid_in[base + (z - 1) * Y * X + y * X + x] : c;

        float laplacian = right + left + up + down + front + back - 6.0f * c;
        float coeff = diffusion_coeff(species) / dx2 * params.dt;
        float source = basal_source(species) * params.dt;
        float sink = basal_sink(species) * params.dt * load * c;
        float updated = c + coeff * laplacian + source - sink;
        grid_out[idx] = max(0.0f, updated);
    }
}
