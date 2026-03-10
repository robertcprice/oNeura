// ============================================================================
// terrarium_substrate.metal — batched atom / chemistry terrarium voxel step
// ============================================================================

#include <metal_stdlib>
using namespace metal;

constant uint SPECIES_COUNT = 14;

constant uint IDX_CARBON = 0;
constant uint IDX_HYDROGEN = 1;
constant uint IDX_OXYGEN = 2;
constant uint IDX_NITROGEN = 3;
constant uint IDX_PHOSPHORUS = 4;
constant uint IDX_SULFUR = 5;
constant uint IDX_WATER = 6;
constant uint IDX_GLUCOSE = 7;
constant uint IDX_OXYGEN_GAS = 8;
constant uint IDX_AMMONIUM = 9;
constant uint IDX_NITRATE = 10;
constant uint IDX_CO2 = 11;
constant uint IDX_PROTON = 12;
constant uint IDX_ATP = 13;

struct Params {
    uint  x_dim;
    uint  y_dim;
    uint  z_dim;
    float voxel_size_mm;
    float dt_ms;
};

inline float diffusion_coeff(uint species) {
    switch (species) {
        case IDX_CARBON: return 0.0010f;
        case IDX_HYDROGEN: return 0.0011f;
        case IDX_OXYGEN: return 0.0012f;
        case IDX_NITROGEN: return 0.0010f;
        case IDX_PHOSPHORUS: return 0.0004f;
        case IDX_SULFUR: return 0.0004f;
        case IDX_WATER: return 0.0032f;
        case IDX_GLUCOSE: return 0.0016f;
        case IDX_OXYGEN_GAS: return 0.0048f;
        case IDX_AMMONIUM: return 0.0020f;
        case IDX_NITRATE: return 0.0018f;
        case IDX_CO2: return 0.0038f;
        case IDX_PROTON: return 0.0024f;
        default: return 0.0009f;
    }
}

kernel void terrarium_substrate(
    device const float* current      [[buffer(0)]],
    device       float* next         [[buffer(1)]],
    device const float* hydration    [[buffer(2)]],
    device const float* microbes     [[buffer(3)]],
    device const float* plant_drive  [[buffer(4)]],
    constant     Params& params      [[buffer(5)]],
    uint         gid                 [[thread_position_in_grid]]
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

    float dt = params.dt_ms;
    float dx2 = max(params.voxel_size_mm * params.voxel_size_mm, 1e-6f);
    float updated[SPECIES_COUNT];

    for (uint species = 0; species < SPECIES_COUNT; species++) {
        uint base = species * total_voxels;
        uint idx = base + gid;
        float c = current[idx];

        float right = (x + 1 < X) ? current[base + z * Y * X + y * X + (x + 1)] : c;
        float left = (x > 0) ? current[base + z * Y * X + y * X + (x - 1)] : c;
        float up = (y + 1 < Y) ? current[base + z * Y * X + (y + 1) * X + x] : c;
        float down = (y > 0) ? current[base + z * Y * X + (y - 1) * X + x] : c;
        float front = (z + 1 < Z) ? current[base + (z + 1) * Y * X + y * X + x] : c;
        float back = (z > 0) ? current[base + (z - 1) * Y * X + y * X + x] : c;

        float laplacian = right + left + up + down + front + back - 6.0f * c;
        updated[species] = max(0.0f, c + diffusion_coeff(species) * dt / dx2 * laplacian);
    }

    float h = clamp(hydration[gid], 0.02f, 1.0f);
    float m = clamp(microbes[gid], 0.02f, 1.2f);
    float p = clamp(plant_drive[gid], 0.0f, 1.0f);
    float atmospheric = (z == 0) ? 1.0f : 0.0f;

    float hydration_gate = h / (0.18f + h);
    float oxygen_gate = updated[IDX_OXYGEN_GAS] / (0.03f + updated[IDX_OXYGEN_GAS]);
    float acidity_penalty = 1.0f / (1.0f + updated[IDX_PROTON] * 6.0f);
    float plant_gate = p * acidity_penalty;

    updated[IDX_WATER] += atmospheric * dt * 0.0008f;
    updated[IDX_OXYGEN_GAS] += atmospheric * dt * 0.0015f;

    float mineral_limit = min(min(updated[IDX_CARBON], updated[IDX_HYDROGEN]), updated[IDX_OXYGEN]);
    float mineralize = min(m * hydration_gate * dt * 0.0015f * mineral_limit, updated[IDX_NITROGEN] + 0.01f);
    updated[IDX_CARBON] = max(0.0f, updated[IDX_CARBON] - mineralize * 0.60f);
    updated[IDX_HYDROGEN] = max(0.0f, updated[IDX_HYDROGEN] - mineralize * 0.84f);
    updated[IDX_OXYGEN] = max(0.0f, updated[IDX_OXYGEN] - mineralize * 0.52f);
    updated[IDX_NITROGEN] = max(0.0f, updated[IDX_NITROGEN] - mineralize * 0.12f);
    updated[IDX_GLUCOSE] += mineralize * 0.72f;
    updated[IDX_AMMONIUM] += mineralize * 0.18f;

    float respiration = min(m * hydration_gate * oxygen_gate * dt * 0.0032f * updated[IDX_GLUCOSE],
                            updated[IDX_OXYGEN_GAS] * 1.2f);
    updated[IDX_GLUCOSE] = max(0.0f, updated[IDX_GLUCOSE] - respiration);
    updated[IDX_OXYGEN_GAS] = max(0.0f, updated[IDX_OXYGEN_GAS] - respiration * 0.75f);
    updated[IDX_CO2] += respiration * 0.92f;
    updated[IDX_WATER] += respiration * 0.25f;
    updated[IDX_ATP] = updated[IDX_ATP] * 0.92f + respiration * 6.6f;

    float fermentation = min(m * hydration_gate * (1.0f - oxygen_gate) * dt * 0.0012f * updated[IDX_GLUCOSE],
                             updated[IDX_GLUCOSE]);
    updated[IDX_GLUCOSE] = max(0.0f, updated[IDX_GLUCOSE] - fermentation);
    updated[IDX_PROTON] += fermentation * 0.08f;
    updated[IDX_ATP] += fermentation * 1.4f;

    float nitrification = min(m * hydration_gate * oxygen_gate * dt * 0.0018f * updated[IDX_AMMONIUM],
                              updated[IDX_OXYGEN_GAS] * 1.5f);
    updated[IDX_AMMONIUM] = max(0.0f, updated[IDX_AMMONIUM] - nitrification);
    updated[IDX_NITRATE] += nitrification * 0.95f;
    updated[IDX_OXYGEN_GAS] = max(0.0f, updated[IDX_OXYGEN_GAS] - nitrification * 0.45f);
    updated[IDX_PROTON] += nitrification * 0.05f;
    updated[IDX_ATP] += nitrification * 0.5f;

    float denitrification = min(m * hydration_gate * (1.0f - oxygen_gate) * dt * 0.0007f * updated[IDX_NITRATE],
                                updated[IDX_NITRATE]);
    updated[IDX_NITRATE] = max(0.0f, updated[IDX_NITRATE] - denitrification);
    updated[IDX_NITROGEN] += denitrification * 0.65f;
    updated[IDX_PROTON] = max(0.0f, updated[IDX_PROTON] - denitrification * 0.03f);

    float photosynthesis = min(plant_gate * dt * 0.0022f * updated[IDX_CO2],
                               min(updated[IDX_WATER] * 0.6f, 0.15f + updated[IDX_CO2]));
    updated[IDX_CO2] = max(0.0f, updated[IDX_CO2] - photosynthesis);
    updated[IDX_WATER] = max(0.0f, updated[IDX_WATER] - photosynthesis * 0.45f);
    updated[IDX_GLUCOSE] += photosynthesis * 0.72f;
    updated[IDX_OXYGEN_GAS] += photosynthesis * 0.84f;
    updated[IDX_CARBON] += photosynthesis * 0.08f;
    updated[IDX_OXYGEN] += photosynthesis * 0.04f;

    float phosphate_turnover = (updated[IDX_GLUCOSE] * 0.004f + updated[IDX_ATP] * 0.0006f) * dt;
    updated[IDX_PHOSPHORUS] = max(0.0f, updated[IDX_PHOSPHORUS] - phosphate_turnover * 0.05f);
    updated[IDX_SULFUR] = max(0.0f, updated[IDX_SULFUR] - phosphate_turnover * 0.02f);

    updated[IDX_WATER] = clamp(updated[IDX_WATER] * (1.0f - dt * 0.00012f), 0.0f, 2.5f);
    updated[IDX_GLUCOSE] = clamp(updated[IDX_GLUCOSE], 0.0f, 2.0f);
    updated[IDX_OXYGEN_GAS] = clamp(updated[IDX_OXYGEN_GAS], 0.0f, 1.5f);
    updated[IDX_AMMONIUM] = clamp(updated[IDX_AMMONIUM], 0.0f, 1.0f);
    updated[IDX_NITRATE] = clamp(updated[IDX_NITRATE], 0.0f, 1.0f);
    updated[IDX_CO2] = clamp(updated[IDX_CO2], 0.0f, 1.5f);
    updated[IDX_PROTON] = clamp(updated[IDX_PROTON], 0.0f, 0.8f);
    updated[IDX_ATP] = clamp(updated[IDX_ATP], 0.0f, 6.0f);

    for (uint species = 0; species < SPECIES_COUNT; species++) {
        next[species * total_voxels + gid] = updated[species];
    }
}
