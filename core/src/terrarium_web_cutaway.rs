use crate::constants::clamp;
use crate::soil_broad::{
    soil_texture_absorbency, soil_texture_capillarity, soil_texture_retention,
};
use crate::substrate_coupling::{
    denitrifier_anoxic_target, microbial_copiotroph_target, nitrifier_aerobic_target,
};
use crate::terrarium::visual_projection::{
    projected_surface_relief, quantized_surface_height, sample_visual_air, sample_visual_chemistry,
    terrain_visual_response_with_chemistry, TERRAIN_VOXEL_BASE_Y, TERRAIN_VOXEL_HEIGHT,
};
use crate::terrarium::{TerrariumSpecies, TerrariumWorld};
use crate::terrarium_web_protocol::{
    TerrainCutawayLayer, TerrainCutawayPocket, TerrainCutawayProfile,
};

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

fn lerp_rgb(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
    ]
}

fn texture_descriptor(
    texture: f32,
    organic: f32,
    surface_bias: f32,
    silicate: f32,
    clay: f32,
    carbonate: f32,
    iron_oxide: f32,
    dissolved_silicate: f32,
    bicarbonate: f32,
    surface_proton_load: f32,
    calcium_bicarbonate_complex: f32,
    sorbed_aluminum_hydroxide: f32,
    sorbed_ferric_hydroxide: f32,
    exchangeable_base_cations: f32,
    exchangeable_aluminum: f32,
    aqueous_iron: f32,
) -> &'static str {
    if surface_bias > 0.72 && organic > 0.46 {
        "organic-rich surface matrix"
    } else if surface_proton_load > 0.10 && exchangeable_aluminum > 0.06 {
        "proton-loaded acidic exchange matrix"
    } else if sorbed_ferric_hydroxide > 0.08 && aqueous_iron > 0.03 {
        "ferric hydroxide-cemented matrix"
    } else if sorbed_aluminum_hydroxide > 0.08 && exchangeable_aluminum > 0.04 {
        "aluminum hydroxide-coated matrix"
    } else if exchangeable_aluminum > 0.10 && carbonate < 0.08 {
        "acid-leached aluminous matrix"
    } else if calcium_bicarbonate_complex > 0.10 && exchangeable_base_cations > 0.18 {
        "calcium-bicarbonate buffered porewater matrix"
    } else if exchangeable_base_cations > 0.22 && carbonate > 0.10 {
        "base-rich carbonate weathering matrix"
    } else if bicarbonate > 0.10 && carbonate > 0.05 {
        "bicarbonate-rich buffered porewater matrix"
    } else if dissolved_silicate > 0.14 && silicate > 0.48 {
        "silica-charged weathering matrix"
    } else if aqueous_iron > 0.06 && iron_oxide > 0.08 {
        "iron-mobilized weathering matrix"
    } else if carbonate > 0.16 && surface_bias < 0.68 {
        "carbonate-bearing mineral matrix"
    } else if iron_oxide > 0.12 {
        "oxide-stained mineral matrix"
    } else if clay > silicate * 0.68 && texture > 0.58 {
        "clay-enriched weathered matrix"
    } else if silicate > 0.52 && texture < 0.44 {
        "silicate-rich coarse matrix"
    } else if texture < 0.28 {
        "coarse mineral matrix"
    } else if texture < 0.68 {
        "mixed mineral matrix"
    } else if surface_bias < 0.26 {
        "dense weathered mineral"
    } else {
        "fine mineral matrix"
    }
}

fn nematode_density(world: &TerrariumWorld, flat: usize) -> f32 {
    world
        .nematode_guilds
        .iter()
        .map(|guild| guild.population_density.get(flat).copied().unwrap_or(0.0))
        .sum::<f32>()
}

fn push_pocket(
    pockets: &mut Vec<TerrainCutawayPocket>,
    kind: &str,
    y: f32,
    lateral_t: f32,
    scale: [f32; 3],
    rgb: [f32; 3],
    tilt: f32,
    opacity: f32,
) {
    pockets.push(TerrainCutawayPocket {
        kind: kind.to_string(),
        y,
        lateral_t,
        scale_x: scale[0],
        scale_y: scale[1],
        scale_z: scale[2],
        rgb,
        tilt,
        opacity,
    });
}

pub fn build_terrain_cutaway_profiles(
    world: &TerrariumWorld,
    atmosphere: &crate::terrarium::TerrariumAtmosphereFrame,
) -> Vec<TerrainCutawayProfile> {
    let width = world.config.width;
    let height = world.config.height;
    let depth = world.config.depth.max(1);
    let surface_relief =
        projected_surface_relief(width, height, &world.soil_structure, &world.water_mask);
    let mut profiles = Vec::new();

    for y in 0..height {
        for x in 0..width {
            let flat = y * width + x;
            let mut faces = Vec::new();
            if x == 0 {
                faces.push("west");
            }
            if x + 1 == width {
                faces.push("east");
            }
            if y == 0 {
                faces.push("north");
            }
            if y + 1 == height {
                faces.push("south");
            }
            if faces.is_empty() {
                continue;
            }

            let local_air =
                sample_visual_air(atmosphere, width, height, x as f32 + 0.5, y as f32 + 0.5);
            let chemistry = sample_visual_chemistry(world, x, y);
            let base_rgb = terrain_visual_response_with_chemistry(
                world.moisture[flat],
                world.organic_matter[flat],
                local_air,
                chemistry,
                world.config.visual_emergence_blend,
            )
            .rgb;
            let surface_top =
                quantized_surface_height(surface_relief[flat]) + TERRAIN_VOXEL_HEIGHT * 0.5;
            let bottom = TERRAIN_VOXEL_BASE_Y + TERRAIN_VOXEL_HEIGHT * 0.5;
            let column_span = (surface_top - bottom).max(TERRAIN_VOXEL_HEIGHT);
            let layer_count = ((column_span / TERRAIN_VOXEL_HEIGHT).round() as usize).max(3);
            let layer_height = column_span / layer_count as f32;
            let soil_texture = world.soil_structure[flat].clamp(0.0, 1.0);
            let organic = world.organic_matter[flat].clamp(0.0, 1.0);
            let absorbency =
                soil_texture_absorbency(world.soil_structure[flat], world.organic_matter[flat]);
            let retention =
                soil_texture_retention(world.soil_structure[flat], world.organic_matter[flat]);
            let capillarity =
                soil_texture_capillarity(world.soil_structure[flat], world.organic_matter[flat]);
            let mut layers = Vec::new();
            for layer in 0..layer_count {
                let surface_bias = if layer_count <= 1 {
                    1.0
                } else {
                    layer as f32 / (layer_count - 1) as f32
                };
                let depth_t = 1.0 - surface_bias;
                let y_center = bottom + layer as f32 * layer_height + layer_height * 0.5;
                let material_class = if layer == 0 {
                    "Bedrock"
                } else if layer + 1 == layer_count {
                    "Surface"
                } else {
                    "Subsoil"
                };
                let z_probe = if depth <= 1 {
                    0
                } else {
                    ((depth_t * (depth - 1) as f32).round() as usize).min(depth - 1)
                };
                let silicate_mineral = world.substrate.patch_mean_species(
                    TerrariumSpecies::SilicateMineral,
                    x,
                    y,
                    z_probe,
                    1,
                );
                let clay_mineral = world.substrate.patch_mean_species(
                    TerrariumSpecies::ClayMineral,
                    x,
                    y,
                    z_probe,
                    1,
                );
                let carbonate_mineral = world.substrate.patch_mean_species(
                    TerrariumSpecies::CarbonateMineral,
                    x,
                    y,
                    z_probe,
                    1,
                );
                let iron_oxide_mineral = world.substrate.patch_mean_species(
                    TerrariumSpecies::IronOxideMineral,
                    x,
                    y,
                    z_probe,
                    1,
                );
                let dissolved_silicate = world.substrate.patch_mean_species(
                    TerrariumSpecies::DissolvedSilicate,
                    x,
                    y,
                    z_probe,
                    1,
                );
                let bicarbonate = world.substrate.patch_mean_species(
                    TerrariumSpecies::BicarbonatePool,
                    x,
                    y,
                    z_probe,
                    1,
                );
                let surface_proton_load = world.substrate.patch_mean_species(
                    TerrariumSpecies::SurfaceProtonLoad,
                    x,
                    y,
                    z_probe,
                    1,
                );
                let calcium_bicarbonate_complex = world.substrate.patch_mean_species(
                    TerrariumSpecies::CalciumBicarbonateComplex,
                    x,
                    y,
                    z_probe,
                    1,
                );
                let sorbed_aluminum_hydroxide = world.substrate.patch_mean_species(
                    TerrariumSpecies::SorbedAluminumHydroxide,
                    x,
                    y,
                    z_probe,
                    1,
                );
                let sorbed_ferric_hydroxide = world.substrate.patch_mean_species(
                    TerrariumSpecies::SorbedFerricHydroxide,
                    x,
                    y,
                    z_probe,
                    1,
                );
                let exchangeable_calcium = world.substrate.patch_mean_species(
                    TerrariumSpecies::ExchangeableCalcium,
                    x,
                    y,
                    z_probe,
                    1,
                );
                let exchangeable_magnesium = world.substrate.patch_mean_species(
                    TerrariumSpecies::ExchangeableMagnesium,
                    x,
                    y,
                    z_probe,
                    1,
                );
                let exchangeable_potassium = world.substrate.patch_mean_species(
                    TerrariumSpecies::ExchangeablePotassium,
                    x,
                    y,
                    z_probe,
                    1,
                );
                let exchangeable_sodium = world.substrate.patch_mean_species(
                    TerrariumSpecies::ExchangeableSodium,
                    x,
                    y,
                    z_probe,
                    1,
                );
                let exchangeable_aluminum = world.substrate.patch_mean_species(
                    TerrariumSpecies::ExchangeableAluminum,
                    x,
                    y,
                    z_probe,
                    1,
                );
                let aqueous_iron = world.substrate.patch_mean_species(
                    TerrariumSpecies::AqueousIronPool,
                    x,
                    y,
                    z_probe,
                    1,
                );
                let exchangeable_base_cations = exchangeable_calcium
                    + exchangeable_magnesium
                    + exchangeable_potassium
                    + exchangeable_sodium;
                let texture_descriptor = texture_descriptor(
                    soil_texture,
                    organic,
                    surface_bias,
                    silicate_mineral,
                    clay_mineral,
                    carbonate_mineral,
                    iron_oxide_mineral,
                    dissolved_silicate,
                    bicarbonate,
                    surface_proton_load,
                    calcium_bicarbonate_complex,
                    sorbed_aluminum_hydroxide,
                    sorbed_ferric_hydroxide,
                    exchangeable_base_cations,
                    exchangeable_aluminum,
                    aqueous_iron,
                )
                .to_string();
                let water =
                    world
                        .substrate
                        .patch_mean_species(TerrariumSpecies::Water, x, y, z_probe, 1);
                let oxygen_gas = world.substrate.patch_mean_species(
                    TerrariumSpecies::OxygenGas,
                    x,
                    y,
                    z_probe,
                    1,
                );
                let carbon_dioxide = world.substrate.patch_mean_species(
                    TerrariumSpecies::CarbonDioxide,
                    x,
                    y,
                    z_probe,
                    1,
                );
                let ammonium = world.substrate.patch_mean_species(
                    TerrariumSpecies::Ammonium,
                    x,
                    y,
                    z_probe,
                    1,
                );
                let nitrate =
                    world
                        .substrate
                        .patch_mean_species(TerrariumSpecies::Nitrate, x, y, z_probe, 1);
                let amino_acids = world.substrate.patch_mean_species(
                    TerrariumSpecies::AminoAcidPool,
                    x,
                    y,
                    z_probe,
                    1,
                );
                let nucleotides = world.substrate.patch_mean_species(
                    TerrariumSpecies::NucleotidePool,
                    x,
                    y,
                    z_probe,
                    1,
                );
                let membrane_precursors = world.substrate.patch_mean_species(
                    TerrariumSpecies::MembranePrecursorPool,
                    x,
                    y,
                    z_probe,
                    1,
                );
                let water_t = clamp(water / 1.2, 0.0, 1.0);
                let oxygen_t = clamp(oxygen_gas / 0.22, 0.0, 1.0);
                let co2_t = clamp(carbon_dioxide / 0.18, 0.0, 1.0);
                let ammonium_t = clamp(ammonium / 0.10, 0.0, 1.0);
                let nitrate_t = clamp(nitrate / 0.10, 0.0, 1.0);
                let biosynth_t = clamp(
                    (amino_acids + nucleotides + membrane_precursors) / 0.24,
                    0.0,
                    1.0,
                );
                let silicate_t = clamp(silicate_mineral / 0.95, 0.0, 1.0);
                let clay_t = clamp(clay_mineral / 0.65, 0.0, 1.0);
                let carbonate_t = clamp(carbonate_mineral / 0.28, 0.0, 1.0);
                let iron_oxide_t = clamp(iron_oxide_mineral / 0.22, 0.0, 1.0);
                let dissolved_silicate_t = clamp(dissolved_silicate / 0.24, 0.0, 1.0);
                let bicarbonate_t = clamp(bicarbonate / 0.18, 0.0, 1.0);
                let surface_proton_t = clamp(surface_proton_load / 0.18, 0.0, 1.0);
                let calcium_bicarbonate_t = clamp(calcium_bicarbonate_complex / 0.16, 0.0, 1.0);
                let sorbed_al_t = clamp(sorbed_aluminum_hydroxide / 0.12, 0.0, 1.0);
                let sorbed_fe_t = clamp(sorbed_ferric_hydroxide / 0.12, 0.0, 1.0);
                let exchangeable_base_t = clamp(exchangeable_base_cations / 0.32, 0.0, 1.0);
                let exchangeable_al_t = clamp(exchangeable_aluminum / 0.14, 0.0, 1.0);
                let aqueous_iron_t = clamp(aqueous_iron / 0.10, 0.0, 1.0);
                let mineral_t = clamp(soil_texture * 0.72 + depth_t * 0.38, 0.0, 1.0);
                let organic_t = clamp(organic * (0.42 + surface_bias * 0.58), 0.0, 1.0);
                let matrix_base = match material_class {
                    "Bedrock" => lerp_rgb(base_rgb, [0.28, 0.26, 0.24], 0.76),
                    "Surface" => lerp_rgb(base_rgb, [0.34, 0.30, 0.22], 0.38),
                    _ => lerp_rgb(base_rgb, [0.40, 0.34, 0.28], 0.54),
                };
                let silicate_tint = lerp_rgb(matrix_base, [0.62, 0.60, 0.56], silicate_t * 0.18);
                let clay_tint = lerp_rgb(silicate_tint, [0.42, 0.34, 0.26], clay_t * 0.20);
                let carbonate_tint = lerp_rgb(clay_tint, [0.78, 0.74, 0.66], carbonate_t * 0.28);
                let oxide_tint = lerp_rgb(carbonate_tint, [0.62, 0.34, 0.22], iron_oxide_t * 0.24);
                let dissolved_tint =
                    lerp_rgb(oxide_tint, [0.58, 0.62, 0.66], dissolved_silicate_t * 0.16);
                let bicarbonate_tint =
                    lerp_rgb(dissolved_tint, [0.62, 0.70, 0.66], bicarbonate_t * 0.14);
                let proton_tint = lerp_rgb(
                    bicarbonate_tint,
                    [0.72, 0.58, 0.50],
                    surface_proton_t * 0.12,
                );
                let calcium_complex_tint = lerp_rgb(
                    proton_tint,
                    [0.72, 0.70, 0.58],
                    calcium_bicarbonate_t * 0.14,
                );
                let sorbed_al_tint =
                    lerp_rgb(calcium_complex_tint, [0.66, 0.64, 0.72], sorbed_al_t * 0.12);
                let sorbed_fe_tint =
                    lerp_rgb(sorbed_al_tint, [0.70, 0.48, 0.26], sorbed_fe_t * 0.14);
                let exchange_tint = lerp_rgb(
                    sorbed_fe_tint,
                    [0.70, 0.66, 0.56],
                    exchangeable_base_t * 0.14,
                );
                let aluminous_tint =
                    lerp_rgb(exchange_tint, [0.46, 0.42, 0.52], exchangeable_al_t * 0.10);
                let moisture_tint = lerp_rgb(oxide_tint, [0.24, 0.34, 0.46], water_t * 0.22);
                let organic_tint = lerp_rgb(
                    lerp_rgb(moisture_tint, aluminous_tint, 0.55),
                    [0.24, 0.30, 0.18],
                    organic_t * 0.24 + biosynth_t * 0.12,
                );
                let gas_tint = lerp_rgb(
                    organic_tint,
                    [0.54, 0.46, 0.26],
                    ammonium_t * 0.10
                        + nitrate_t * 0.08
                        + co2_t * 0.06
                        + exchangeable_base_t * 0.05
                        + aqueous_iron_t * 0.05,
                );
                let rgb = lerp_rgb(
                    gas_tint,
                    [0.44, 0.46, 0.48],
                    mineral_t * 0.20 - oxygen_t * 0.05 + dissolved_silicate_t * 0.04,
                );
                layers.push(TerrainCutawayLayer {
                    material_class: material_class.into(),
                    texture_descriptor,
                    y: y_center,
                    height: layer_height,
                    rgb,
                    texture: soil_texture,
                    organic,
                    water,
                    oxygen_gas,
                    carbon_dioxide,
                    ammonium,
                    nitrate,
                    amino_acids,
                    nucleotides,
                    membrane_precursors,
                    silicate_mineral,
                    clay_mineral,
                    carbonate_mineral,
                    iron_oxide_mineral,
                    dissolved_silicate,
                    bicarbonate,
                    surface_proton_load,
                    calcium_bicarbonate_complex,
                    sorbed_aluminum_hydroxide,
                    sorbed_ferric_hydroxide,
                    exchangeable_calcium,
                    exchangeable_magnesium,
                    exchangeable_potassium,
                    exchangeable_sodium,
                    exchangeable_aluminum,
                    aqueous_iron,
                });
            }

            let mut pockets = Vec::new();
            for z in 0..depth {
                let z_frac = if depth > 1 {
                    z as f32 / (depth - 1) as f32
                } else {
                    0.0
                };
                let hydration = clamp(
                    world.moisture[flat] * (1.0 - z_frac * 0.55)
                        + world.deep_moisture[flat] * z_frac,
                    0.02,
                    1.0,
                );
                let projected_microbes = clamp(
                    world.microbial_biomass[flat] * (0.65 + world.moisture[flat] * 0.55)
                        + world.symbiont_biomass[flat] * (0.55 + z_frac * 0.30),
                    0.02,
                    1.2,
                );
                let oxygen =
                    world
                        .substrate
                        .patch_mean_species(TerrariumSpecies::OxygenGas, x, y, z, 1);
                let co2 =
                    world
                        .substrate
                        .patch_mean_species(TerrariumSpecies::CarbonDioxide, x, y, z, 1);
                let glucose =
                    world
                        .substrate
                        .patch_mean_species(TerrariumSpecies::Glucose, x, y, z, 1);
                let ammonium =
                    world
                        .substrate
                        .patch_mean_species(TerrariumSpecies::Ammonium, x, y, z, 1);
                let amino =
                    world
                        .substrate
                        .patch_mean_species(TerrariumSpecies::AminoAcidPool, x, y, z, 1);
                let nucleotide = world.substrate.patch_mean_species(
                    TerrariumSpecies::NucleotidePool,
                    x,
                    y,
                    z,
                    1,
                );
                let oxygen_factor = clamp(
                    oxygen / 0.22 + (1.0 - world.deep_moisture[flat] * 0.45),
                    0.05,
                    1.2,
                );
                let aeration_factor = clamp(
                    1.10 - world.moisture[flat] * 0.55 - world.deep_moisture[flat] * 0.45,
                    0.05,
                    1.15,
                );
                let anoxia_factor = clamp(
                    (world.deep_moisture[flat] * 0.95 + world.moisture[flat] * 0.18)
                        - oxygen_factor * 0.28,
                    0.02,
                    1.3,
                );
                let root_factor = 1.0 + world.root_density[flat] * 0.08;
                let substrate_gate = clamp(
                    (world.litter_carbon[flat] * 1.10
                        + world.root_exudates[flat] * 1.35
                        + world.organic_matter[flat] * 0.90)
                        / 0.08,
                    0.0,
                    1.35,
                );
                let moisture_factor = clamp(
                    (world.moisture[flat] + world.deep_moisture[flat] * 0.35) / 0.48,
                    0.0,
                    1.6,
                );

                let copiotroph = microbial_copiotroph_target(
                    substrate_gate,
                    moisture_factor,
                    oxygen_factor,
                    root_factor,
                ) * (projected_microbes / 1.2);
                let nitrifier =
                    nitrifier_aerobic_target(oxygen_factor, aeration_factor, anoxia_factor)
                        * clamp(ammonium / 0.12, 0.0, 1.0);
                let denitrifier = denitrifier_anoxic_target(
                    anoxia_factor,
                    world.deep_moisture[flat],
                    oxygen_factor,
                ) * clamp(co2 / 0.16 + hydration * 0.35, 0.0, 1.0);

                let y_pos = surface_top - z_frac * column_span;
                let lateral_base = (0.24 + z_frac * 0.46).clamp(0.16, 0.82);
                if hydration > 0.62 {
                    push_pocket(
                        &mut pockets,
                        "water_band",
                        y_pos,
                        0.62,
                        [0.040, 0.055, 0.22],
                        [0.20, 0.42, 0.72],
                        0.04,
                        0.68,
                    );
                }
                if copiotroph > 0.32 {
                    push_pocket(
                        &mut pockets,
                        "copiotroph_band",
                        y_pos,
                        lateral_base,
                        [0.032, 0.028, 0.14],
                        lerp_rgb(
                            [0.48, 0.42, 0.26],
                            [0.70, 0.58, 0.32],
                            clamp(glucose / 0.18, 0.0, 1.0),
                        ),
                        0.16,
                        0.88,
                    );
                }
                if nitrifier > 0.30 {
                    push_pocket(
                        &mut pockets,
                        "nitrifier_band",
                        y_pos,
                        0.28 + z_frac * 0.18,
                        [0.028, 0.022, 0.16],
                        [0.74, 0.62, 0.30],
                        0.08,
                        0.90,
                    );
                }
                if denitrifier > 0.30 {
                    push_pocket(
                        &mut pockets,
                        "denitrifier_band",
                        y_pos,
                        0.70 - z_frac * 0.16,
                        [0.036, 0.028, 0.18],
                        [0.58, 0.30, 0.48],
                        0.24,
                        0.86,
                    );
                }
                if amino > 0.12 || nucleotide > 0.10 {
                    push_pocket(
                        &mut pockets,
                        "biosynthetic_pool",
                        y_pos,
                        0.46,
                        [0.022, 0.024, 0.10],
                        lerp_rgb(
                            [0.36, 0.50, 0.26],
                            [0.48, 0.62, 0.48],
                            clamp((amino + nucleotide) / 0.30, 0.0, 1.0),
                        ),
                        0.0,
                        0.84,
                    );
                }
            }

            let earthworm_density = world.earthworm_population.population_density[flat];
            let earthworm_bioturbation = world.earthworm_population.bioturbation_rate[flat];
            if earthworm_density > 6.0 || earthworm_bioturbation > 0.18 {
                push_pocket(
                    &mut pockets,
                    "earthworm_activity",
                    surface_top - column_span * (0.34 + retention * 0.18),
                    0.34,
                    [
                        0.070,
                        0.26 + clamp(earthworm_bioturbation / 4.0, 0.0, 0.22),
                        0.070,
                    ],
                    [0.48, 0.30, 0.22],
                    0.42,
                    0.96,
                );
            }

            let nematodes = nematode_density(world, flat);
            if nematodes > 0.35 {
                push_pocket(
                    &mut pockets,
                    "nematode_activity",
                    surface_top - column_span * (0.22 + capillarity * 0.16),
                    0.66,
                    [0.020, 0.018, 0.16],
                    [0.84, 0.80, 0.62],
                    0.10,
                    0.92,
                );
            }

            if world.explicit_microbe_authority[flat] > 0.18
                || world.explicit_microbe_activity[flat] > 0.18
            {
                push_pocket(
                    &mut pockets,
                    "explicit_microbe_band",
                    surface_top - column_span * (0.26 + absorbency * 0.20),
                    0.50,
                    [0.038, 0.040, 0.20],
                    [0.34, 0.64, 0.56],
                    0.18,
                    0.90,
                );
            }

            if soil_texture > 0.70 {
                push_pocket(
                    &mut pockets,
                    "stone_lens",
                    surface_top - column_span * 0.72,
                    0.56,
                    [0.05, 0.05, 0.10],
                    [0.52, 0.50, 0.48],
                    0.0,
                    0.98,
                );
            }

            for face in faces {
                profiles.push(TerrainCutawayProfile {
                    x,
                    y,
                    face: face.to_string(),
                    top: surface_top,
                    bottom,
                    layers: layers.clone(),
                    pockets: pockets.clone(),
                });
            }
        }
    }

    profiles
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrarium::presets::TerrariumDemoPreset;

    #[test]
    fn terrain_cutaway_profiles_expose_mineral_layer_fields() {
        let world = TerrariumWorld::demo_preset(23, false, TerrariumDemoPreset::MicroTerrarium)
            .expect("preset should build");
        let atmosphere = world.atmosphere_frame();
        let profiles = build_terrain_cutaway_profiles(&world, &atmosphere);
        assert!(!profiles.is_empty());

        let mut saw_silicate = false;
        let mut saw_carb_or_oxide_descriptor = false;
        let mut saw_reactive_weathering_pool = false;
        for profile in &profiles {
            for layer in &profile.layers {
                if layer.silicate_mineral > 0.0 && layer.clay_mineral > 0.0 {
                    saw_silicate = true;
                }
                if layer.dissolved_silicate > 0.0
                    || layer.exchangeable_calcium > 0.0
                    || layer.exchangeable_magnesium > 0.0
                    || layer.exchangeable_potassium > 0.0
                    || layer.exchangeable_sodium > 0.0
                    || layer.exchangeable_aluminum > 0.0
                    || layer.aqueous_iron > 0.0
                {
                    saw_reactive_weathering_pool = true;
                }
                if layer.texture_descriptor.contains("carbonate")
                    || layer.texture_descriptor.contains("oxide")
                    || layer.texture_descriptor.contains("silicate")
                    || layer.texture_descriptor.contains("clay")
                    || layer.texture_descriptor.contains("aluminous")
                    || layer.texture_descriptor.contains("base-rich")
                    || layer.texture_descriptor.contains("silica-charged")
                {
                    saw_carb_or_oxide_descriptor = true;
                }
            }
        }

        assert!(saw_silicate, "cutaway layers should carry mineral fields");
        assert!(
            saw_reactive_weathering_pool,
            "cutaway layers should expose dissolved silica and exchangeable/weathered ion pools"
        );
        assert!(
            saw_carb_or_oxide_descriptor,
            "cutaway descriptors should reflect mineralogy, not only generic dirt classes"
        );
    }
}
