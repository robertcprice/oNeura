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

inline float local_diffusion_scale(
    uint species,
    float membrane_adjacency,
    float septum_zone,
    float nucleoid_occupancy,
    float local_crowding
) {
    float crowding_pressure = clamp(local_crowding, 0.0f, 1.6f);
    float base = 1.0f;
    switch (species) {
        case 0: base = 1.0f - 0.12f * septum_zone - 0.20f * nucleoid_occupancy; break;
        case 1: base = 1.0f - 0.10f * septum_zone - 0.24f * nucleoid_occupancy; break;
        case 2: base = 1.0f - 0.08f * septum_zone - 0.28f * nucleoid_occupancy; break;
        default: base = 0.82f + 0.12f * membrane_adjacency - 0.14f * septum_zone; break;
    }
    return clamp(base - crowding_pressure * (0.12f + 0.22f * nucleoid_occupancy), 0.18f, 1.35f);
}

inline float local_source_scale(
    uint species,
    float membrane_adjacency,
    float septum_zone,
    float nucleoid_occupancy,
    float local_energy_source,
    float local_membrane_source,
    float local_crowding
) {
    switch (species) {
        case 0:
            return clamp(
                0.80f
                    + 0.14f * membrane_adjacency
                    + 0.08f * septum_zone
                    + 0.22f * local_energy_source
                    - 0.06f * local_crowding,
                0.25f,
                1.80f
            );
        case 1:
            return clamp(0.90f + 0.08f * (1.0f - nucleoid_occupancy) - 0.05f * local_crowding, 0.30f, 1.50f);
        case 2:
            return clamp(0.82f + 0.18f * nucleoid_occupancy - 0.04f * local_crowding, 0.30f, 1.55f);
        default:
            return clamp(
                0.44f
                    + local_membrane_source * (0.26f * membrane_adjacency + 0.54f * septum_zone)
                    - 0.04f * local_crowding,
                0.20f,
                1.90f
            );
    }
}

inline float local_sink_scale(
    uint species,
    float membrane_adjacency,
    float septum_zone,
    float nucleoid_occupancy,
    float local_atp_demand,
    float local_amino_demand,
    float local_nucleotide_demand,
    float local_membrane_demand,
    float local_crowding,
    constant Params& params
) {
    switch (species) {
        case 0:
            return clamp(
                0.72f
                    + local_atp_demand * (0.24f * membrane_adjacency + 0.20f * septum_zone)
                    + 0.18f * nucleoid_occupancy
                    + 0.10f * local_crowding
                    + 0.08f * max(params.metabolic_load, 0.1f),
                0.20f,
                2.20f
            );
        case 1:
            return clamp(
                0.80f
                    + local_amino_demand * (0.28f + 0.18f * nucleoid_occupancy + 0.10f * septum_zone)
                    + 0.08f * local_crowding
                    + 0.05f * max(params.metabolic_load, 0.1f),
                0.20f,
                1.90f
            );
        case 2:
            return clamp(
                0.70f
                    + local_nucleotide_demand * (0.44f * nucleoid_occupancy + 0.12f * septum_zone)
                    + 0.06f * local_crowding
                    + 0.05f * max(params.metabolic_load, 0.1f),
                0.20f,
                2.20f
            );
        default:
            return clamp(
                0.26f
                    + local_membrane_demand * (0.30f * membrane_adjacency + 0.58f * septum_zone)
                    + 0.08f * local_crowding
                    + 0.04f * max(params.metabolic_load, 0.1f),
                0.10f,
                2.40f
            );
    }
}

kernel void whole_cell_rdme_kernel(
    device const float* grid_in  [[buffer(0)]],
    device       float* grid_out [[buffer(1)]],
    device const float* field_in [[buffer(2)]],
    device const float* drive_in [[buffer(3)]],
    constant     Params& params  [[buffer(4)]],
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
    uint field_offset = total_voxels;
    float membrane_adjacency = clamp(field_in[gid], 0.0f, 1.0f);
    float septum_zone = clamp(field_in[field_offset + gid], 0.0f, 1.0f);
    float nucleoid_occupancy = clamp(field_in[2 * field_offset + gid], 0.0f, 1.0f);
    uint drive_offset = total_voxels;
    float local_energy_source = clamp(drive_in[gid], 0.0f, 2.5f);
    float local_atp_demand = clamp(drive_in[drive_offset + gid], 0.0f, 2.5f);
    float local_amino_demand = clamp(drive_in[2 * drive_offset + gid], 0.0f, 2.5f);
    float local_nucleotide_demand = clamp(drive_in[3 * drive_offset + gid], 0.0f, 2.5f);
    float local_membrane_source = clamp(drive_in[4 * drive_offset + gid], 0.0f, 2.5f);
    float local_membrane_demand = clamp(drive_in[5 * drive_offset + gid], 0.0f, 2.5f);
    float local_crowding = clamp(drive_in[6 * drive_offset + gid], 0.0f, 1.6f);

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
        float coeff = diffusion_coeff(species) / dx2 * params.dt
            * local_diffusion_scale(species, membrane_adjacency, septum_zone, nucleoid_occupancy, local_crowding);
        float source = basal_source(species) * params.dt
            * local_source_scale(
                species,
                membrane_adjacency,
                septum_zone,
                nucleoid_occupancy,
                local_energy_source,
                local_membrane_source,
                local_crowding
            );
        float sink = basal_sink(species) * params.dt * load
            * local_sink_scale(
                species,
                membrane_adjacency,
                septum_zone,
                nucleoid_occupancy,
                local_atp_demand,
                local_amino_demand,
                local_nucleotide_demand,
                local_membrane_demand,
                local_crowding,
                params
            )
            * c;
        float updated = c + coeff * laplacian + source - sink;
        grid_out[idx] = max(0.0f, updated);
    }
}
