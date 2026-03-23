#import bevy_pbr::mesh_view_bindings
#import bevy_pbr::mesh_bindings

#import bevy_pbr::utils
#import bevy_pbr::clustered_forward
#import bevy_pbr::lighting
#import bevy_pbr::pbr_ambient
#import bevy_pbr::shadows
#import bevy_pbr::fog
#import bevy_pbr::pbr_types
#import bevy_pbr::pbr_functions

struct TerrariumSceneMaterial {
    base_color: vec4<f32>,
    emissive: vec4<f32>,
    pbr: vec4<f32>,
    atmosphere_color: vec4<f32>,
    local_air: vec4<f32>,
    dynamics: vec4<f32>,
    flags: vec4<u32>,
};

@group(1) @binding(0)
var<uniform> material: TerrariumSceneMaterial;

struct FragmentInput {
    @builtin(front_facing) is_front: bool,
    @builtin(position) frag_coord: vec4<f32>,
    #import bevy_pbr::mesh_vertex_output
};

fn clamp01(value: f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

const TERRARIUM_SHADER_FLAG_SUBSTRATE: u32 = 1u;
const TERRARIUM_SHADER_FLAG_PLUME: u32 = 2u;
const TERRARIUM_SHADER_FLAG_FLUID: u32 = 4u;
const TERRARIUM_SHADER_FLAG_MICROBE: u32 = 8u;
const TERRARIUM_SHADER_FLAG_PLANT: u32 = 16u;
const TERRARIUM_SHADER_FLAG_SEED: u32 = 32u;
const TERRARIUM_SHADER_FLAG_FRUIT: u32 = 64u;
const TERRARIUM_SHADER_FLAG_FLY: u32 = 128u;

@fragment
fn fragment(in: FragmentInput) -> @location(0) vec4<f32> {
    let flags = material.flags.x;
    let humidity_t = clamp01(material.local_air.x);
    let pressure_t = clamp01(material.local_air.y);
    let density_t = clamp01(material.local_air.z);
    let temperature_t = clamp01(material.local_air.w);
    let activity_t = clamp01(material.dynamics.x);
    let energy_t = clamp01(material.dynamics.y);
    let stress_t = clamp01(material.dynamics.z);
    let daylight = clamp01(material.atmosphere_color.w);
    let pulse_phase =
        material.dynamics.w * 6.28318530718 + in.world_position.x * 0.11 + in.world_position.z * 0.09;
    let bio_pulse = 1.0
        + sin(pulse_phase)
            * (0.010 + humidity_t * 0.018 + pressure_t * 0.016 + activity_t * 0.030 + energy_t * 0.020);
    let atmospheric_haze =
        material.atmosphere_color.rgb
            * (0.018 + humidity_t * 0.08 + pressure_t * 0.05 + density_t * 0.05 + activity_t * 0.04);

    var output_rgb: vec3<f32> = material.base_color.rgb;
    output_rgb =
        output_rgb
            * (0.78 + daylight * 0.20 + temperature_t * 0.05 + density_t * 0.04)
            * bio_pulse
        + atmospheric_haze;

    if ((flags & TERRARIUM_SHADER_FLAG_SUBSTRATE) != 0u) {
        output_rgb =
            output_rgb * (0.92 + density_t * 0.08)
            + material.atmosphere_color.rgb * (0.04 + humidity_t * 0.04);
    }
    if ((flags & TERRARIUM_SHADER_FLAG_PLUME) != 0u) {
        output_rgb += material.atmosphere_color.rgb * (0.10 + activity_t * 0.18 + density_t * 0.08);
    }
    if ((flags & TERRARIUM_SHADER_FLAG_FLUID) != 0u) {
        output_rgb =
            output_rgb * (0.94 + humidity_t * 0.06)
            + vec3<f32>(0.02, 0.03, 0.04) * (0.4 + density_t * 0.6);
    }
    if ((flags & TERRARIUM_SHADER_FLAG_MICROBE) != 0u) {
        output_rgb += vec3<f32>(0.04, 0.03, 0.05) * (activity_t * 0.8 + energy_t * 0.4);
    }
    if ((flags & TERRARIUM_SHADER_FLAG_PLANT) != 0u) {
        output_rgb += vec3<f32>(0.01, 0.05, 0.01) * (humidity_t * 0.5 + energy_t * 0.4);
    }
    if ((flags & TERRARIUM_SHADER_FLAG_SEED) != 0u) {
        output_rgb += vec3<f32>(0.03, 0.02, 0.01) * (energy_t * 0.4 + activity_t * 0.3);
    }
    if ((flags & TERRARIUM_SHADER_FLAG_FRUIT) != 0u) {
        output_rgb += vec3<f32>(0.05, 0.02, 0.01) * (energy_t * 0.4 + humidity_t * 0.3);
    }
    if ((flags & TERRARIUM_SHADER_FLAG_FLY) != 0u) {
        output_rgb += vec3<f32>(0.02, 0.02, 0.03) * (activity_t * 0.5 + pressure_t * 0.3);
    }

    var pbr_input: PbrInput = pbr_input_new();
    pbr_input.material.base_color = vec4<f32>(output_rgb, material.base_color.a);
    pbr_input.material.reflectance = material.pbr.z;
    pbr_input.material.flags = flags;
    pbr_input.material.alpha_cutoff = material.pbr.w;
    pbr_input.material.emissive = vec4<f32>(
        material.emissive.rgb
            * (0.78 + daylight * 0.14 + humidity_t * 0.06 + energy_t * 0.10)
            + atmospheric_haze * (0.10 + temperature_t * 0.06),
        1.0,
    );
    pbr_input.material.metallic = material.pbr.x;
    pbr_input.material.perceptual_roughness = clamp(
        material.pbr.y
            - f32((flags & TERRARIUM_SHADER_FLAG_FLUID) != 0u) * (0.05 + density_t * 0.04)
            - f32((flags & TERRARIUM_SHADER_FLAG_PLUME) != 0u) * (0.08 + activity_t * 0.06)
            + stress_t * 0.08,
        0.089,
        1.0,
    );
    pbr_input.frag_coord = in.frag_coord;
    pbr_input.world_position = in.world_position;
    pbr_input.world_normal = prepare_world_normal(
        in.world_normal,
        (flags & STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT) != 0u,
        in.is_front,
    );
    pbr_input.is_orthographic = view.projection[3].w == 1.0;
    pbr_input.N = normalize(pbr_input.world_normal);
    pbr_input.V = calculate_view(in.world_position, pbr_input.is_orthographic);
    pbr_input.occlusion = clamp(1.0 - stress_t * 0.12 + density_t * 0.04, 0.72, 1.08);
    pbr_input.flags = mesh.flags;

    var shaded = pbr(pbr_input);

    if (fog.mode != FOG_MODE_OFF && (flags & STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT) != 0u) {
        shaded = apply_fog(shaded, in.world_position.xyz, view.world_position.xyz);
    }

#ifdef TONEMAP_IN_SHADER
    shaded = tone_mapping(shaded);
#ifdef DEBAND_DITHER
    var shaded_rgb = shaded.rgb;
    shaded_rgb = powsafe(shaded_rgb, 1.0 / 2.2);
    shaded_rgb = shaded_rgb + screen_space_dither(in.frag_coord.xy);
    shaded_rgb = powsafe(shaded_rgb, 2.2);
    shaded = vec4(shaded_rgb, shaded.a);
#endif
#endif

#ifdef PREMULTIPLY_ALPHA
    shaded = premultiply_alpha(flags, shaded);
#endif

    return shaded;
}
