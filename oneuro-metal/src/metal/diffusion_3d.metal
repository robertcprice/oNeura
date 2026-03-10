// ============================================================================
// diffusion_3d.metal — 3D extracellular space diffusion
// ============================================================================
// 6-neighbor Laplacian stencil for NT diffusion through extracellular space.
// One thread per voxel.  Grid layout: [Z][Y][X] with 6 NTs per voxel.
//
// Memory layout (SoA per NT):
//   For NT channel c: grid[c * Z*Y*X + z * Y*X + y * X + x]
//   This gives better memory coalescing than interleaved [z][y][x][6].
//
// Boundary condition: Neumann (zero-flux) — missing neighbors use center value.
// ============================================================================

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constant uint  NT_COUNT       = 6;
constant float DIFFUSION_COEFF = 0.001f;   // mm^2/ms
constant float NATURAL_DECAY   = 0.01f;    // fraction per ms

// ---------------------------------------------------------------------------
// Params
// ---------------------------------------------------------------------------
struct Params {
    uint  x_dim;
    uint  y_dim;
    uint  z_dim;
    float voxel_size;   // mm (typically 0.01)
    float dt;           // ms
};

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
kernel void diffusion_3d(
    device const float* grid_in    [[buffer(0)]],    // source grid (read)
    device       float* grid_out   [[buffer(1)]],    // destination grid (write)
    constant     Params& params    [[buffer(2)]],
    uint         gid               [[thread_position_in_grid]]
) {
    uint X = params.x_dim;
    uint Y = params.y_dim;
    uint Z = params.z_dim;
    uint total_voxels = X * Y * Z;

    // Each thread handles one voxel (all 6 NTs)
    if (gid >= total_voxels) return;

    // Decompose linear index into (x, y, z)
    uint z = gid / (Y * X);
    uint rem = gid - z * Y * X;
    uint y = rem / X;
    uint x = rem - y * X;

    float dt   = params.dt;
    float dx   = params.voxel_size;
    float dx2  = dx * dx;
    float coeff = DIFFUSION_COEFF / dx2 * dt;

    // Process all 6 NT channels for this voxel
    for (uint nt = 0; nt < NT_COUNT; nt++) {
        uint base = nt * total_voxels;
        uint idx  = base + gid;

        float c = grid_in[idx];

        // 6-neighbor Laplacian with Neumann boundary (use c for missing)
        float right = (x + 1 < X) ? grid_in[base + z * Y * X + y * X + (x + 1)] : c;
        float left  = (x > 0)     ? grid_in[base + z * Y * X + y * X + (x - 1)] : c;
        float up    = (y + 1 < Y) ? grid_in[base + z * Y * X + (y + 1) * X + x] : c;
        float down  = (y > 0)     ? grid_in[base + z * Y * X + (y - 1) * X + x] : c;
        float front = (z + 1 < Z) ? grid_in[base + (z + 1) * Y * X + y * X + x] : c;
        float back  = (z > 0)     ? grid_in[base + (z - 1) * Y * X + y * X + x] : c;

        float laplacian = right + left + up + down + front + back - 6.0f * c;

        float c_new = c + coeff * laplacian;

        // Natural decay toward zero
        c_new *= 1.0f - NATURAL_DECAY * dt;

        grid_out[idx] = max(0.0f, c_new);
    }
}
