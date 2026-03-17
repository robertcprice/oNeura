// Terrarium Bio-Pulse Shader — Bevy 0.15 ExtendedMaterial
//
// Adds living-ecosystem visual effects on top of StandardMaterial PBR:
// - Bio-pulse: subtle sinusoidal color breathing driven by organism activity
// - Atmospheric haze: humidity/density-driven fog tinting
// - Entity-type tinting: substrate, water, plant, fly, fruit each get unique modulation
// - Daylight cycle: warm/cool color shift

#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
}

#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    prepass_io::{VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}
#endif

// Custom uniforms for bio-pulse effects
struct TerrariumBio {
    // x: humidity, y: activity, z: daylight, w: time (pulse phase)
    params: vec4<f32>,
    // x: entity_flags (bitfield), y: energy, z: stress, w: unused
    entity: vec4<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var<uniform> bio: TerrariumBio;

// Entity flag bits
const FLAG_SUBSTRATE: u32 = 1u;
const FLAG_WATER: u32 = 2u;
const FLAG_PLANT: u32 = 4u;
const FLAG_FLY: u32 = 8u;
const FLAG_FRUIT: u32 = 16u;

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    let humidity = clamp(bio.params.x, 0.0, 1.0);
    let activity = clamp(bio.params.y, 0.0, 1.0);
    let daylight = clamp(bio.params.z, 0.0, 1.0);
    let time = bio.params.w;
    let flags = u32(bio.entity.x);
    let energy = clamp(bio.entity.y, 0.0, 1.0);
    let stress = clamp(bio.entity.z, 0.0, 1.0);

    // Bio-pulse: gentle sinusoidal breathing
    let pulse_phase = time * 6.28318 + in.world_position.x * 0.11 + in.world_position.z * 0.09;
    let pulse = 1.0 + sin(pulse_phase) * (0.01 + humidity * 0.018 + activity * 0.03 + energy * 0.02);

    // Atmospheric haze color (warm sunrise / cool twilight)
    let haze_color = vec3<f32>(
        0.6 + daylight * 0.3,
        0.65 + daylight * 0.15,
        0.8 - daylight * 0.1,
    );
    let haze_strength = 0.018 + humidity * 0.08 + activity * 0.04;

    // Modulate base color
    var color = pbr_input.material.base_color.rgb;
    color = color * (0.78 + daylight * 0.20) * pulse + haze_color * haze_strength;

    // Per-entity-type effects
    if ((flags & FLAG_SUBSTRATE) != 0u) {
        color += haze_color * (0.04 + humidity * 0.04);
    }
    if ((flags & FLAG_WATER) != 0u) {
        color += vec3<f32>(0.02, 0.03, 0.06) * (0.4 + humidity * 0.6);
        // Slightly reduce roughness for wet-look
        pbr_input.material.perceptual_roughness = max(pbr_input.material.perceptual_roughness - 0.08, 0.089);
    }
    if ((flags & FLAG_PLANT) != 0u) {
        color += vec3<f32>(0.01, 0.05, 0.01) * (humidity * 0.5 + energy * 0.4);
    }
    if ((flags & FLAG_FLY) != 0u) {
        color += vec3<f32>(0.02, 0.02, 0.03) * (activity * 0.5);
    }
    if ((flags & FLAG_FRUIT) != 0u) {
        color += vec3<f32>(0.05, 0.02, 0.01) * (energy * 0.4 + humidity * 0.3);
    }

    // Stress darkening
    color *= 1.0 - stress * 0.12;

    pbr_input.material.base_color = vec4<f32>(color, pbr_input.material.base_color.a);

    // Alpha discard
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

#ifdef PREPASS_PIPELINE
    let out = deferred_output(in, pbr_input);
#else
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
#endif

    return out;
}
