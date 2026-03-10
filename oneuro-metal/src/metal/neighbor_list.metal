// ============================================================================
// neighbor_list.metal — GPU Neighbor List Construction
// ============================================================================
// Cell-linked neighbor list for O(N) non-bonded force computation.
// Uses spatial hashing with cubic cells.
// ============================================================================

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------
struct NeighborParams {
    uint n_atoms;
    float cutoff;         // Interaction cutoff + skin
    float cell_size;     // Cell size (>= cutoff)
    uint grid_x, grid_y, grid_z;  // Grid dimensions
    uint max_neighbors;   // Max neighbors per atom
};

// Cell grid structure (built on CPU, read on GPU)
struct CellGrid {
    uint cell_start;     // Start index in atom list
    uint cell_count;    // Number of atoms in cell
};

// Neighbor list output
struct NeighborListEntry {
    uint j;              // Neighbor atom index
    float dx, dy, dz;   // displacement (for PBC)
};

// ---------------------------------------------------------------------------
// Kernel: Build cell grid (count atoms per cell)
// ---------------------------------------------------------------------------
kernel void count_atoms_per_cell(
    device float3* positions [[buffer(0)]],
    device atomic_uint* cell_counts [[buffer(1)]],
    constant NeighborParams& params [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= params.n_atoms) return;

    float3 pos = positions[id];

    // Compute cell index
    int cx = int(pos.x / params.cell_size);
    int cy = int(pos.y / params.cell_size);
    int cz = int(pos.z / params.cell_size);

    // Wrap into grid
    cx = (cx + params.grid_x) % params.grid_x;
    cy = (cy + params.grid_y) % params.grid_y;
    cz = (cz + params.grid_z) % params.grid_z;

    uint cell_idx = cx + cy * params.grid_x + cz * params.grid_x * params.grid_y;

    atomic_fetch_add_explicit(&cell_counts[cell_idx], 1, memory_order_relaxed, memory_scope_device);
}

// ---------------------------------------------------------------------------

// Kernel: Build atom-to-cell mapping
// ---------------------------------------------------------------------------
kernel void map_atoms_to_cells(
    device float3* positions [[buffer(0)]],
    device uint* atom_cells [[buffer(1)]],
    device atomic_uint* cell_offsets [[buffer(2)]],
    constant NeighborParams& params [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= params.n_atoms) return;

    float3 pos = positions[id];

    int cx = int(pos.x / params.cell_size);
    int cy = int(pos.y / params.cell_size);
    int cz = int(pos.z / params.cell_size);

    cx = (cx + params.grid_x) % params.grid_x;
    cy = (cy + params.grid_y) % params.grid_y;
    cz = (cz + params.grid_z) % params.grid_z;

    uint cell_idx = cx + cy * params.grid_x + cz * params.grid_x * params.grid_y;

    atom_cells[id] = cell_idx;

    // Get position in cell
    uint offset = atomic_fetch_add_explicit(&cell_offsets[cell_idx], 1, memory_order_relaxed, memory_scope_device);
}

// ---------------------------------------------------------------------------
// Kernel: Build neighbor list
// ---------------------------------------------------------------------------
kernel void build_neighbor_list(
    device float3* positions [[buffer(0)]],
    device uint* atom_cells [[buffer(1)]],
    device NeighborListEntry* neighbors [[buffer(2)]],
    device atomic_uint* neighbor_counts [[buffer(3)]],
    constant NeighborParams& params [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= params.n_atoms) return;

    float3 pos_i = positions[id];
    uint cell_i = atom_cells[id];

    // Decode cell coordinates
    uint cx = cell_i % params.grid_x;
    uint cy = (cell_i / params.grid_x) % params.grid_y;
    uint cz = cell_i / (params.grid_x * params.grid_y);

    uint n_neighbors = 0;

    // Check neighboring cells (27-cell neighborhood)
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {

                int nx = int(cx) + dx;
                int ny = int(cy) + dy;
                int nz = int(cz) + dz;

                // Wrap PBC
                nx = (nx + int(params.grid_x)) % int(params.grid_x);
                ny = (ny + int(params.grid_y)) % int(params.grid_y);
                nz = (nz + int(params.grid_z)) % int(params.grid_z);

                uint neighbor_cell = uint(nx) + uint(ny) * params.grid_x
                                    + uint(nz) * params.grid_x * params.grid_y;

                // In full implementation, would iterate atoms in neighbor_cell
                // and check distance < cutoff
            }
        }
    }

    neighbor_counts[id] = n_neighbors;
}

// ---------------------------------------------------------------------------
// Kernel: Update positions from neighbor list (for visualization)
// ---------------------------------------------------------------------------
kernel void extract_positions(
    device Atom* atoms [[buffer(0)]],
    device float3* positions_out [[buffer(1)]],
    device float3* velocities_out [[buffer(2)]],
    constant uint& n_atoms [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n_atoms) return;

    positions_out[id] = atoms[id].position;
    velocities_out[id] = atoms[id].velocity;
}
