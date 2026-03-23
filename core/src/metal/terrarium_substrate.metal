// ============================================================================
// terrarium_substrate.metal — batched atom / chemistry terrarium voxel step
// ============================================================================

#include <metal_stdlib>
using namespace metal;

constant uint SPECIES_COUNT = 33;

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
constant uint IDX_AMINO_ACIDS = 14;
constant uint IDX_NUCLEOTIDES = 15;
constant uint IDX_MEMBRANE_PRECURSORS = 16;
constant uint IDX_SILICATE_MINERAL = 17;
constant uint IDX_CLAY_MINERAL = 18;
constant uint IDX_CARBONATE_MINERAL = 19;
constant uint IDX_IRON_OXIDE_MINERAL = 20;
constant uint IDX_DISSOLVED_SILICATE = 21;
constant uint IDX_EXCHANGEABLE_CALCIUM = 22;
constant uint IDX_EXCHANGEABLE_MAGNESIUM = 23;
constant uint IDX_EXCHANGEABLE_POTASSIUM = 24;
constant uint IDX_EXCHANGEABLE_SODIUM = 25;
constant uint IDX_EXCHANGEABLE_ALUMINUM = 26;
constant uint IDX_AQUEOUS_IRON = 27;
constant uint IDX_BICARBONATE = 28;
constant uint IDX_SURFACE_PROTON_LOAD = 29;
constant uint IDX_CALCIUM_BICARBONATE_COMPLEX = 30;
constant uint IDX_SORBED_ALUMINUM_HYDROXIDE = 31;
constant uint IDX_SORBED_FERRIC_HYDROXIDE = 32;

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
        case IDX_ATP: return 0.0009f;
        case IDX_AMINO_ACIDS: return 0.0012f;
        case IDX_NUCLEOTIDES: return 0.0010f;
        case IDX_MEMBRANE_PRECURSORS: return 0.0008f;
        case IDX_SILICATE_MINERAL: return 0.00005f;
        case IDX_CLAY_MINERAL: return 0.00003f;
        case IDX_CARBONATE_MINERAL: return 0.00008f;
        case IDX_IRON_OXIDE_MINERAL: return 0.00002f;
        case IDX_DISSOLVED_SILICATE: return 0.0012f;
        case IDX_EXCHANGEABLE_CALCIUM: return 0.0006f;
        case IDX_EXCHANGEABLE_MAGNESIUM: return 0.0005f;
        case IDX_EXCHANGEABLE_POTASSIUM: return 0.0007f;
        case IDX_EXCHANGEABLE_SODIUM: return 0.0008f;
        case IDX_EXCHANGEABLE_ALUMINUM: return 0.0004f;
        case IDX_BICARBONATE: return 0.0011f;
        case IDX_SURFACE_PROTON_LOAD: return 0.0002f;
        case IDX_CALCIUM_BICARBONATE_COMPLEX: return 0.0009f;
        case IDX_SORBED_ALUMINUM_HYDROXIDE: return 0.0001f;
        case IDX_SORBED_FERRIC_HYDROXIDE: return 0.0001f;
        default: return 0.0005f;
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
    updated[IDX_AMINO_ACIDS] += mineralize * 0.10f;
    updated[IDX_NUCLEOTIDES] += mineralize * 0.04f;

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

    float amino_synthesis_limit = min(
        updated[IDX_GLUCOSE],
        min(
            updated[IDX_AMMONIUM] * 1.4f + updated[IDX_NITRATE] * 0.5f,
            max(0.0f, 0.04f + updated[IDX_ATP] * 0.06f)
        )
    );
    float amino_synthesis = min(
        m * hydration_gate * dt * 0.0010f * amino_synthesis_limit,
        updated[IDX_GLUCOSE] * 0.45f
    );
    updated[IDX_GLUCOSE] = max(0.0f, updated[IDX_GLUCOSE] - amino_synthesis * 0.55f);
    updated[IDX_AMMONIUM] = max(0.0f, updated[IDX_AMMONIUM] - amino_synthesis * 0.32f);
    updated[IDX_NITRATE] = max(0.0f, updated[IDX_NITRATE] - amino_synthesis * 0.06f);
    updated[IDX_ATP] = max(0.0f, updated[IDX_ATP] - amino_synthesis * 0.24f);
    updated[IDX_AMINO_ACIDS] += amino_synthesis * 0.86f;

    float nucleotide_synthesis_limit = min(
        updated[IDX_NITRATE],
        min(
            updated[IDX_PHOSPHORUS] * 4.0f,
            min(updated[IDX_GLUCOSE] * 0.8f, max(0.0f, 0.02f + updated[IDX_ATP] * 0.04f))
        )
    );
    float nucleotide_synthesis = min(
        m * hydration_gate * dt * 0.0007f * nucleotide_synthesis_limit,
        updated[IDX_NITRATE] * 0.55f
    );
    updated[IDX_GLUCOSE] = max(0.0f, updated[IDX_GLUCOSE] - nucleotide_synthesis * 0.20f);
    updated[IDX_NITRATE] = max(0.0f, updated[IDX_NITRATE] - nucleotide_synthesis * 0.40f);
    updated[IDX_PHOSPHORUS] = max(0.0f, updated[IDX_PHOSPHORUS] - nucleotide_synthesis * 0.06f);
    updated[IDX_ATP] = max(0.0f, updated[IDX_ATP] - nucleotide_synthesis * 0.16f);
    updated[IDX_NUCLEOTIDES] += nucleotide_synthesis * 0.80f;

    float membrane_synthesis_limit = min(
        updated[IDX_GLUCOSE],
        min(
            updated[IDX_OXYGEN_GAS] * 0.8f + updated[IDX_WATER] * 0.2f,
            max(0.0f, 0.02f + updated[IDX_ATP] * 0.03f)
        )
    );
    float membrane_synthesis = min(
        ((m * 0.55f + p * 0.45f) * hydration_gate * dt * 0.0006f * membrane_synthesis_limit),
        updated[IDX_GLUCOSE] * 0.30f
    );
    updated[IDX_GLUCOSE] = max(0.0f, updated[IDX_GLUCOSE] - membrane_synthesis * 0.42f);
    updated[IDX_OXYGEN_GAS] = max(0.0f, updated[IDX_OXYGEN_GAS] - membrane_synthesis * 0.08f);
    updated[IDX_ATP] = max(0.0f, updated[IDX_ATP] - membrane_synthesis * 0.12f);
    updated[IDX_MEMBRANE_PRECURSORS] += membrane_synthesis * 0.76f;

    float proteolysis = min(
        m * hydration_gate * dt * 0.0008f * updated[IDX_AMINO_ACIDS],
        updated[IDX_AMINO_ACIDS]
    );
    updated[IDX_AMINO_ACIDS] = max(0.0f, updated[IDX_AMINO_ACIDS] - proteolysis);
    updated[IDX_AMMONIUM] += proteolysis * 0.22f;
    updated[IDX_CO2] += proteolysis * 0.16f;
    updated[IDX_ATP] += proteolysis * 0.18f;

    float nucleotide_turnover = min(
        m * hydration_gate * (0.45f + (1.0f - oxygen_gate) * 0.35f) * dt * 0.0004f
            * updated[IDX_NUCLEOTIDES],
        updated[IDX_NUCLEOTIDES]
    );
    updated[IDX_NUCLEOTIDES] = max(0.0f, updated[IDX_NUCLEOTIDES] - nucleotide_turnover);
    updated[IDX_NITRATE] += nucleotide_turnover * 0.24f;
    updated[IDX_PHOSPHORUS] += nucleotide_turnover * 0.03f;
    updated[IDX_ATP] += nucleotide_turnover * 0.10f;

    float membrane_turnover = min(
        (m * 0.6f + acidity_penalty * 0.2f) * dt * 0.0003f * updated[IDX_MEMBRANE_PRECURSORS],
        updated[IDX_MEMBRANE_PRECURSORS]
    );
    updated[IDX_MEMBRANE_PRECURSORS] =
        max(0.0f, updated[IDX_MEMBRANE_PRECURSORS] - membrane_turnover);
    updated[IDX_GLUCOSE] += membrane_turnover * 0.12f;
    updated[IDX_CARBON] += membrane_turnover * 0.03f;
    updated[IDX_SULFUR] += membrane_turnover * 0.01f;

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
    updated[IDX_MEMBRANE_PRECURSORS] += photosynthesis * 0.04f;

    float phosphate_turnover = (updated[IDX_GLUCOSE] * 0.004f + updated[IDX_ATP] * 0.0006f) * dt;
    updated[IDX_PHOSPHORUS] = max(0.0f, updated[IDX_PHOSPHORUS] - phosphate_turnover * 0.05f);
    updated[IDX_SULFUR] = max(0.0f, updated[IDX_SULFUR] - phosphate_turnover * 0.02f);

    float mineral_water_gate = updated[IDX_WATER] / (0.08f + updated[IDX_WATER]);
    float acidity_drive = updated[IDX_PROTON] / (0.01f + updated[IDX_PROTON]);
    float silicate_weathering = min(
        min(updated[IDX_SILICATE_MINERAL] * mineral_water_gate * acidity_drive * dt * 0.00004f,
            updated[IDX_SILICATE_MINERAL]),
        updated[IDX_PROTON] * 0.55f
    );
    updated[IDX_SILICATE_MINERAL] = max(0.0f, updated[IDX_SILICATE_MINERAL] - silicate_weathering);
    updated[IDX_DISSOLVED_SILICATE] += silicate_weathering * 0.62f;
    updated[IDX_EXCHANGEABLE_MAGNESIUM] += silicate_weathering * 0.08f;
    updated[IDX_EXCHANGEABLE_POTASSIUM] += silicate_weathering * 0.06f;
    updated[IDX_EXCHANGEABLE_SODIUM] += silicate_weathering * 0.05f;
    updated[IDX_EXCHANGEABLE_CALCIUM] += silicate_weathering * 0.04f;
    updated[IDX_PROTON] = max(0.0f, updated[IDX_PROTON] - silicate_weathering * 0.04f);

    float clay_weathering = min(
        min(updated[IDX_CLAY_MINERAL] * mineral_water_gate * acidity_drive * dt * 0.00003f,
            updated[IDX_CLAY_MINERAL]),
        updated[IDX_PROTON] * 0.45f
    );
    updated[IDX_CLAY_MINERAL] = max(0.0f, updated[IDX_CLAY_MINERAL] - clay_weathering);
    updated[IDX_DISSOLVED_SILICATE] += clay_weathering * 0.34f;
    updated[IDX_EXCHANGEABLE_ALUMINUM] += clay_weathering * 0.14f;
    updated[IDX_EXCHANGEABLE_MAGNESIUM] += clay_weathering * 0.04f;
    updated[IDX_EXCHANGEABLE_POTASSIUM] += clay_weathering * 0.03f;
    updated[IDX_PROTON] = max(0.0f, updated[IDX_PROTON] - clay_weathering * 0.02f);

    float carbonate_buffering = min(
        min(updated[IDX_CARBONATE_MINERAL] * mineral_water_gate * acidity_drive * dt * 0.00008f,
            updated[IDX_CARBONATE_MINERAL]),
        updated[IDX_PROTON] * 0.98f
    );
    updated[IDX_CARBONATE_MINERAL] =
        max(0.0f, updated[IDX_CARBONATE_MINERAL] - carbonate_buffering);
    updated[IDX_PROTON] = max(0.0f, updated[IDX_PROTON] - carbonate_buffering);
    updated[IDX_EXCHANGEABLE_CALCIUM] += carbonate_buffering;
    updated[IDX_BICARBONATE] += carbonate_buffering;

    float bicarbonate_degassing = min(
        min(updated[IDX_BICARBONATE]
                * acidity_drive
                * (0.35f + (1.0f - mineral_water_gate) * 0.30f)
                * dt
                * 0.00004f,
            updated[IDX_BICARBONATE]),
        updated[IDX_PROTON] * 0.92f + 0.002f
    );
    updated[IDX_BICARBONATE] = max(0.0f, updated[IDX_BICARBONATE] - bicarbonate_degassing);
    updated[IDX_PROTON] = max(0.0f, updated[IDX_PROTON] - bicarbonate_degassing);
    updated[IDX_CO2] += bicarbonate_degassing;
    updated[IDX_WATER] += bicarbonate_degassing;

    float alkalinity_gate = updated[IDX_BICARBONATE] / (0.04f + updated[IDX_BICARBONATE]);
    float mineral_surface_gate =
        (updated[IDX_CLAY_MINERAL] * 0.55f + updated[IDX_CARBONATE_MINERAL] * 0.25f
            + updated[IDX_SILICATE_MINERAL] * 0.20f)
        / (0.06f + updated[IDX_CLAY_MINERAL] * 0.55f + updated[IDX_CARBONATE_MINERAL] * 0.25f
            + updated[IDX_SILICATE_MINERAL] * 0.20f);

    float proton_surface_sorption = max(0.0f, min(
        min(updated[IDX_PROTON]
                * mineral_surface_gate
                * (0.30f + acidity_drive * 0.55f)
                * (0.28f + (1.0f - alkalinity_gate) * 0.36f)
                * dt
                * 0.00006f,
            updated[IDX_PROTON]),
        (0.42f + updated[IDX_CLAY_MINERAL] * 0.38f + updated[IDX_CARBONATE_MINERAL] * 0.20f)
            - updated[IDX_SURFACE_PROTON_LOAD]
    ));
    updated[IDX_PROTON] = max(0.0f, updated[IDX_PROTON] - proton_surface_sorption);
    updated[IDX_SURFACE_PROTON_LOAD] += proton_surface_sorption;

    float proton_surface_desorption = min(
        updated[IDX_SURFACE_PROTON_LOAD]
            * (0.18f + alkalinity_gate * 0.36f)
            * (0.22f + oxygen_gate * 0.10f + mineral_water_gate * 0.16f)
            * dt
            * 0.00003f,
        updated[IDX_SURFACE_PROTON_LOAD]
    );
    updated[IDX_SURFACE_PROTON_LOAD] =
        max(0.0f, updated[IDX_SURFACE_PROTON_LOAD] - proton_surface_desorption);
    updated[IDX_PROTON] += proton_surface_desorption;

    float calcium_bicarbonate_complexation = min(
        min(updated[IDX_EXCHANGEABLE_CALCIUM]
                * updated[IDX_BICARBONATE]
                * (0.14f + alkalinity_gate * 0.34f)
                * (0.20f + mineral_water_gate * 0.18f)
                * clamp(1.0f - acidity_drive * 0.42f, 0.0f, 1.0f)
                * dt
                * 0.00005f,
            updated[IDX_EXCHANGEABLE_CALCIUM]),
        updated[IDX_BICARBONATE] * 0.5f
    );
    updated[IDX_EXCHANGEABLE_CALCIUM] =
        max(0.0f, updated[IDX_EXCHANGEABLE_CALCIUM] - calcium_bicarbonate_complexation);
    updated[IDX_BICARBONATE] =
        max(0.0f, updated[IDX_BICARBONATE] - calcium_bicarbonate_complexation * 2.0f);
    updated[IDX_CALCIUM_BICARBONATE_COMPLEX] += calcium_bicarbonate_complexation;

    float calcium_bicarbonate_dissociation = min(
        updated[IDX_CALCIUM_BICARBONATE_COMPLEX]
            * (0.14f + acidity_drive * 0.38f + updated[IDX_SURFACE_PROTON_LOAD] * 0.32f)
            * dt
            * 0.00004f,
        updated[IDX_CALCIUM_BICARBONATE_COMPLEX]
    );
    updated[IDX_CALCIUM_BICARBONATE_COMPLEX] =
        max(0.0f, updated[IDX_CALCIUM_BICARBONATE_COMPLEX] - calcium_bicarbonate_dissociation);
    updated[IDX_EXCHANGEABLE_CALCIUM] += calcium_bicarbonate_dissociation;
    updated[IDX_BICARBONATE] += calcium_bicarbonate_dissociation * 2.0f;

    float aluminum_hydroxide_precip = min(
        min(updated[IDX_EXCHANGEABLE_ALUMINUM]
                * mineral_water_gate
                * clamp(1.0f - acidity_drive * 0.55f, 0.0f, 1.0f)
                * (0.35f + alkalinity_gate * 0.65f)
                * dt
                * 0.00005f,
            updated[IDX_EXCHANGEABLE_ALUMINUM]),
        updated[IDX_WATER] / 3.0f
    );
    updated[IDX_EXCHANGEABLE_ALUMINUM] =
        max(0.0f, updated[IDX_EXCHANGEABLE_ALUMINUM] - aluminum_hydroxide_precip);
    updated[IDX_WATER] = max(0.0f, updated[IDX_WATER] - aluminum_hydroxide_precip * 3.0f);
    updated[IDX_SORBED_ALUMINUM_HYDROXIDE] += aluminum_hydroxide_precip;
    updated[IDX_PROTON] += aluminum_hydroxide_precip * 3.0f;

    float aluminum_hydroxide_dissolution = min(
        min(updated[IDX_SORBED_ALUMINUM_HYDROXIDE]
                * acidity_drive
                * (0.30f + (1.0f - alkalinity_gate) * 0.50f)
                * dt
                * 0.00003f,
            updated[IDX_SORBED_ALUMINUM_HYDROXIDE]),
        updated[IDX_PROTON] / 3.0f
    );
    updated[IDX_SORBED_ALUMINUM_HYDROXIDE] =
        max(0.0f, updated[IDX_SORBED_ALUMINUM_HYDROXIDE] - aluminum_hydroxide_dissolution);
    updated[IDX_PROTON] = max(0.0f, updated[IDX_PROTON] - aluminum_hydroxide_dissolution * 3.0f);
    updated[IDX_EXCHANGEABLE_ALUMINUM] += aluminum_hydroxide_dissolution;
    updated[IDX_WATER] += aluminum_hydroxide_dissolution * 3.0f;

    float ferric_hydroxide_precip = min(
        min(updated[IDX_AQUEOUS_IRON]
                * mineral_water_gate
                * oxygen_gate
                * clamp(1.0f - acidity_drive * 0.45f, 0.0f, 1.0f)
                * (0.30f + alkalinity_gate * 0.70f)
                * dt
                * 0.00005f,
            updated[IDX_AQUEOUS_IRON]),
        updated[IDX_WATER] / 3.0f
    );
    updated[IDX_AQUEOUS_IRON] = max(0.0f, updated[IDX_AQUEOUS_IRON] - ferric_hydroxide_precip);
    updated[IDX_WATER] = max(0.0f, updated[IDX_WATER] - ferric_hydroxide_precip * 3.0f);
    updated[IDX_SORBED_FERRIC_HYDROXIDE] += ferric_hydroxide_precip;
    updated[IDX_PROTON] += ferric_hydroxide_precip * 3.0f;

    float ferric_hydroxide_dissolution = min(
        min(updated[IDX_SORBED_FERRIC_HYDROXIDE]
                * acidity_drive
                * (1.0f - oxygen_gate * 0.25f)
                * (0.32f + (1.0f - alkalinity_gate) * 0.42f)
                * dt
                * 0.00003f,
            updated[IDX_SORBED_FERRIC_HYDROXIDE]),
        updated[IDX_PROTON] / 3.0f
    );
    updated[IDX_SORBED_FERRIC_HYDROXIDE] =
        max(0.0f, updated[IDX_SORBED_FERRIC_HYDROXIDE] - ferric_hydroxide_dissolution);
    updated[IDX_PROTON] = max(0.0f, updated[IDX_PROTON] - ferric_hydroxide_dissolution * 3.0f);
    updated[IDX_AQUEOUS_IRON] += ferric_hydroxide_dissolution;
    updated[IDX_WATER] += ferric_hydroxide_dissolution * 3.0f;

    float iron_release = min(
        min(updated[IDX_IRON_OXIDE_MINERAL] * mineral_water_gate * acidity_drive
                * (1.0f - oxygen_gate * 0.35f) * dt * 0.00002f,
            updated[IDX_IRON_OXIDE_MINERAL]),
        updated[IDX_PROTON] * 0.25f + 0.002f
    );
    updated[IDX_IRON_OXIDE_MINERAL] = max(0.0f, updated[IDX_IRON_OXIDE_MINERAL] - iron_release);
    updated[IDX_AQUEOUS_IRON] += iron_release * 0.72f;
    updated[IDX_PROTON] = max(0.0f, updated[IDX_PROTON] - iron_release * 0.01f);

    float base_leaching = clamp(updated[IDX_WATER] * mineral_water_gate * dt * 0.00002f, 0.0f, 0.02f);
    updated[IDX_EXCHANGEABLE_SODIUM] =
        max(0.0f, updated[IDX_EXCHANGEABLE_SODIUM] - base_leaching * updated[IDX_EXCHANGEABLE_SODIUM] * 0.22f);
    updated[IDX_EXCHANGEABLE_POTASSIUM] =
        max(0.0f, updated[IDX_EXCHANGEABLE_POTASSIUM] - base_leaching * updated[IDX_EXCHANGEABLE_POTASSIUM] * 0.16f);
    updated[IDX_EXCHANGEABLE_CALCIUM] =
        max(0.0f, updated[IDX_EXCHANGEABLE_CALCIUM] - base_leaching * updated[IDX_EXCHANGEABLE_CALCIUM] * 0.12f);
    updated[IDX_EXCHANGEABLE_MAGNESIUM] =
        max(0.0f, updated[IDX_EXCHANGEABLE_MAGNESIUM] - base_leaching * updated[IDX_EXCHANGEABLE_MAGNESIUM] * 0.10f);

    updated[IDX_WATER] = clamp(updated[IDX_WATER] * (1.0f - dt * 0.00012f), 0.0f, 2.5f);
    updated[IDX_GLUCOSE] = clamp(updated[IDX_GLUCOSE], 0.0f, 2.0f);
    updated[IDX_OXYGEN_GAS] = clamp(updated[IDX_OXYGEN_GAS], 0.0f, 1.5f);
    updated[IDX_AMMONIUM] = clamp(updated[IDX_AMMONIUM], 0.0f, 1.0f);
    updated[IDX_NITRATE] = clamp(updated[IDX_NITRATE], 0.0f, 1.0f);
    updated[IDX_CO2] = clamp(updated[IDX_CO2], 0.0f, 1.5f);
    updated[IDX_PROTON] = clamp(updated[IDX_PROTON], 0.0f, 0.8f);
    updated[IDX_ATP] = clamp(updated[IDX_ATP], 0.0f, 6.0f);
    updated[IDX_AMINO_ACIDS] = clamp(updated[IDX_AMINO_ACIDS], 0.0f, 1.2f);
    updated[IDX_NUCLEOTIDES] = clamp(updated[IDX_NUCLEOTIDES], 0.0f, 1.0f);
    updated[IDX_MEMBRANE_PRECURSORS] = clamp(updated[IDX_MEMBRANE_PRECURSORS], 0.0f, 1.0f);
    updated[IDX_SILICATE_MINERAL] = clamp(updated[IDX_SILICATE_MINERAL], 0.0f, 4.0f);
    updated[IDX_CLAY_MINERAL] = clamp(updated[IDX_CLAY_MINERAL], 0.0f, 4.0f);
    updated[IDX_CARBONATE_MINERAL] = clamp(updated[IDX_CARBONATE_MINERAL], 0.0f, 2.0f);
    updated[IDX_IRON_OXIDE_MINERAL] = clamp(updated[IDX_IRON_OXIDE_MINERAL], 0.0f, 2.0f);
    updated[IDX_DISSOLVED_SILICATE] = clamp(updated[IDX_DISSOLVED_SILICATE], 0.0f, 1.2f);
    updated[IDX_EXCHANGEABLE_CALCIUM] = clamp(updated[IDX_EXCHANGEABLE_CALCIUM], 0.0f, 1.2f);
    updated[IDX_EXCHANGEABLE_MAGNESIUM] = clamp(updated[IDX_EXCHANGEABLE_MAGNESIUM], 0.0f, 1.0f);
    updated[IDX_EXCHANGEABLE_POTASSIUM] = clamp(updated[IDX_EXCHANGEABLE_POTASSIUM], 0.0f, 0.9f);
    updated[IDX_EXCHANGEABLE_SODIUM] = clamp(updated[IDX_EXCHANGEABLE_SODIUM], 0.0f, 0.9f);
    updated[IDX_EXCHANGEABLE_ALUMINUM] = clamp(updated[IDX_EXCHANGEABLE_ALUMINUM], 0.0f, 0.9f);
    updated[IDX_AQUEOUS_IRON] = clamp(updated[IDX_AQUEOUS_IRON], 0.0f, 0.8f);
    updated[IDX_BICARBONATE] = clamp(updated[IDX_BICARBONATE], 0.0f, 1.2f);
    updated[IDX_SURFACE_PROTON_LOAD] = clamp(updated[IDX_SURFACE_PROTON_LOAD], 0.0f, 0.8f);
    updated[IDX_CALCIUM_BICARBONATE_COMPLEX] =
        clamp(updated[IDX_CALCIUM_BICARBONATE_COMPLEX], 0.0f, 0.8f);
    updated[IDX_SORBED_ALUMINUM_HYDROXIDE] = clamp(updated[IDX_SORBED_ALUMINUM_HYDROXIDE], 0.0f, 0.8f);
    updated[IDX_SORBED_FERRIC_HYDROXIDE] = clamp(updated[IDX_SORBED_FERRIC_HYDROXIDE], 0.0f, 0.8f);

    for (uint species = 0; species < SPECIES_COUNT; species++) {
        next[species * total_voxels + gid] = updated[species];
    }
}
