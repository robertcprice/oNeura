//! Rust-owned terrarium render descriptors and projection types.

use crate::plant_cellular::PlantTissue;
use crate::WholeCellChemistrySite;

// TerrariumTopdownView is canonically defined in terrarium_world.rs.
// Re-export it here so existing imports from terrarium_render still work.
pub use crate::terrarium_world::TerrariumTopdownView;

#[derive(Debug, Clone)]
pub struct TerrariumPbrMaterialRender {
    pub base_color_rgba: [f32; 4],
    pub emissive_rgb: [f32; 3],
    pub metallic: f32,
    pub perceptual_roughness: f32,
    pub reflectance: f32,
    pub alpha_blend: bool,
    pub double_sided: bool,
    pub shader_atmosphere_rgba: [f32; 4],
    pub shader_dynamics_rgba: [f32; 4],
    pub shader_flags: u32,
}

#[derive(Debug, Clone)]
pub struct TerrariumSceneMaterialRender {
    pub base_color_rgba: [f32; 4],
    pub emissive_rgba: [f32; 4],
    pub pbr_rgba: [f32; 4],
    pub atmosphere_color_rgba: [f32; 4],
    pub local_air_rgba: [f32; 4],
    pub dynamics_rgba: [f32; 4],
}

pub fn compose_scene_material_render(
    render: &TerrariumPbrMaterialRender,
    clear_color_rgb: [f32; 3],
    ambient_color_rgb: [f32; 3],
    humidity_t: f32,
    pressure_t: f32,
    temperature_t: f32,
    daylight: f32,
    time_phase: f32,
) -> TerrariumSceneMaterialRender {
    let local_humidity = render.shader_atmosphere_rgba[0].clamp(0.0, 1.0);
    let local_pressure = render.shader_atmosphere_rgba[1].clamp(0.0, 1.0);
    let local_density = render.shader_atmosphere_rgba[2].clamp(0.0, 1.0);
    let local_temperature = render.shader_atmosphere_rgba[3].clamp(0.0, 1.0);
    let activity = render.shader_dynamics_rgba[0].clamp(0.0, 1.0);
    let energy = render.shader_dynamics_rgba[1].clamp(0.0, 1.0);
    let stress = render.shader_dynamics_rgba[2].clamp(0.0, 1.0);
    let phase_bias = render.shader_dynamics_rgba[3];
    let atmosphere_rgb = [
        clear_color_rgb[0] * (0.52 + humidity_t * 0.08 + local_humidity * 0.10)
            + ambient_color_rgb[0] * (0.20 + pressure_t * 0.06 + local_pressure * 0.12)
            + local_density * 0.06
            + local_temperature * 0.04,
        clear_color_rgb[1] * (0.52 + humidity_t * 0.08 + local_humidity * 0.10)
            + ambient_color_rgb[1] * (0.20 + pressure_t * 0.06 + local_pressure * 0.12)
            + local_density * 0.08
            + energy * 0.04,
        clear_color_rgb[2] * (0.52 + humidity_t * 0.08 + local_humidity * 0.10)
            + ambient_color_rgb[2] * (0.20 + pressure_t * 0.06 + local_pressure * 0.12)
            + local_density * 0.10
            + activity * 0.03,
    ];
    TerrariumSceneMaterialRender {
        base_color_rgba: render.base_color_rgba,
        emissive_rgba: [
            render.emissive_rgb[0],
            render.emissive_rgb[1],
            render.emissive_rgb[2],
            render.base_color_rgba[3],
        ],
        pbr_rgba: [
            render.metallic,
            render.perceptual_roughness,
            render.reflectance,
            0.5,
        ],
        atmosphere_color_rgba: [
            atmosphere_rgb[0],
            atmosphere_rgb[1],
            atmosphere_rgb[2],
            daylight,
        ],
        local_air_rgba: [
            local_humidity,
            local_pressure,
            local_density,
            local_temperature.max(temperature_t),
        ],
        dynamics_rgba: [activity, energy, stress, (time_phase + phase_bias).fract()],
    }
}

#[derive(Debug, Clone)]
pub struct TerrariumLightingRender {
    pub clear_color_rgb: [f32; 3],
    pub ambient_color_rgb: [f32; 3],
    pub ambient_brightness: f32,
    pub sun_color_rgb: [f32; 3],
    pub sun_illuminance: f32,
    pub sun_translation_world: [f32; 3],
    pub sun_focus_world: [f32; 3],
    pub humidity_t: f32,
    pub pressure_t: f32,
    pub temperature_t: f32,
    pub daylight: f32,
    pub time_phase: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TerrariumMeshPrimitive {
    Cube,
    UvSphere { sectors: usize, stacks: usize },
    Cylinder { resolution: usize },
    Quad,
}

#[derive(Debug, Clone, Default)]
pub struct TerrariumTriangleMeshRender {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct TerrariumGroundTileRender {
    pub idx: usize,
    pub primitive: TerrariumMeshPrimitive,
    pub mesh: TerrariumTriangleMeshRender,
    pub translation_world: [f32; 3],
    pub scale_world: [f32; 3],
    pub terrain_norm: f32,
    pub field_norm: f32,
    pub material: TerrariumPbrMaterialRender,
}

#[derive(Debug, Clone)]
pub struct TerrariumWaterRender {
    pub render_id: u64,
    pub render_fingerprint: u64,
    pub primitive: TerrariumMeshPrimitive,
    pub mesh: TerrariumTriangleMeshRender,
    pub translation_world: [f32; 3],
    pub scale_world: [f32; 3],
    pub material: TerrariumPbrMaterialRender,
}

#[derive(Debug, Clone)]
pub struct TerrariumSubstrateVoxelRender {
    pub render_id: u64,
    pub render_fingerprint: u64,
    pub voxel: [usize; 3],
    pub primitive: TerrariumMeshPrimitive,
    pub mesh: TerrariumTriangleMeshRender,
    pub translation_world: [f32; 3],
    pub signal: f32,
    pub material: TerrariumPbrMaterialRender,
}

#[derive(Debug, Clone)]
pub struct TerrariumSubstrateBatchRender {
    pub render_id: u64,
    pub render_fingerprint: u64,
    pub mesh_cache_key: u64,
    pub material_state_key: u64,
    pub mesh: TerrariumTriangleMeshRender,
    pub material: TerrariumPbrMaterialRender,
    pub hide_on_cutaway: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TerrariumDynamicBatchKind {
    MicrobePacket,
    MicrobeSite,
    Water,
    Plume,
    PlantTissue,
    SeedPart,
    FruitPart,
    FlyPart,
}

#[derive(Debug, Clone)]
pub struct TerrariumDynamicBatchRender {
    pub render_id: u64,
    pub render_fingerprint: u64,
    pub mesh_cache_key: u64,
    pub material_state_key: u64,
    pub kind: TerrariumDynamicBatchKind,
    pub mesh: TerrariumTriangleMeshRender,
    pub material: TerrariumPbrMaterialRender,
}

#[derive(Debug, Clone)]
pub struct TerrariumExplicitMicrobePacketRender {
    pub render_id: u64,
    pub render_fingerprint: u64,
    pub translation_local: [f32; 3],
    pub mesh: TerrariumTriangleMeshRender,
    pub material: TerrariumPbrMaterialRender,
}

#[derive(Debug, Clone)]
pub struct TerrariumExplicitMicrobeSiteRender {
    pub render_id: u64,
    pub render_fingerprint: u64,
    pub site: WholeCellChemistrySite,
    pub translation_local: [f32; 3],
    pub mesh: TerrariumTriangleMeshRender,
    pub material: TerrariumPbrMaterialRender,
}

#[derive(Debug, Clone)]
pub struct TerrariumExplicitMicrobeRender {
    pub render_id: u64,
    pub body_render_fingerprint: u64,
    pub packet_population_render_fingerprint: u64,
    pub translation_world: [f32; 3],
    pub body_mesh: TerrariumTriangleMeshRender,
    pub packet_mesh: TerrariumTriangleMeshRender,
    pub packets: Vec<TerrariumExplicitMicrobePacketRender>,
    pub sites: Vec<TerrariumExplicitMicrobeSiteRender>,
    pub represented_cells: f32,
    pub represented_packets: f32,
    pub activity: f32,
    pub stress: f32,
    pub body_material: TerrariumPbrMaterialRender,
    pub packet_material: TerrariumPbrMaterialRender,
}

#[derive(Debug, Clone)]
pub struct TerrariumPlantTissueRender {
    pub render_id: u64,
    pub render_fingerprint: u64,
    pub tissue: PlantTissue,
    pub mesh: TerrariumTriangleMeshRender,
    pub translation_world: [f32; 3],
    pub rotation_xyz_rad: [f32; 3],
    pub material: TerrariumPbrMaterialRender,
}

#[derive(Debug, Clone)]
pub struct TerrariumPlantRender {
    pub render_id: u64,
    pub stem_render_fingerprint: u64,
    pub canopy_render_fingerprint: u64,
    pub stem_primitive: TerrariumMeshPrimitive,
    pub stem_mesh: TerrariumTriangleMeshRender,
    pub stem_translation_world: [f32; 3],
    pub stem_rotation_xyz_rad: [f32; 3],
    pub stem_scale_world: [f32; 3],
    pub canopy_primitive: TerrariumMeshPrimitive,
    pub canopy_mesh: TerrariumTriangleMeshRender,
    pub canopy_translation_world: [f32; 3],
    pub canopy_rotation_xyz_rad: [f32; 3],
    pub canopy_scale_world: [f32; 3],
    pub health: f32,
    pub leaf_energy: f32,
    pub tissues: Vec<TerrariumPlantTissueRender>,
    pub stem_material: TerrariumPbrMaterialRender,
    pub canopy_material: TerrariumPbrMaterialRender,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerrariumSeedPartKind {
    Coat,
    Endosperm,
    Radicle,
    CotyledonLeft,
    CotyledonRight,
}

#[derive(Debug, Clone)]
pub struct TerrariumSeedPartRender {
    pub render_id: u64,
    pub render_fingerprint: u64,
    pub kind: TerrariumSeedPartKind,
    pub mesh: TerrariumTriangleMeshRender,
    pub translation_local: [f32; 3],
    pub rotation_xyz_rad: [f32; 3],
    pub material: TerrariumPbrMaterialRender,
}

#[derive(Debug, Clone)]
pub struct TerrariumSeedRender {
    pub render_id: u64,
    pub translation_world: [f32; 3],
    pub rotation_xyz_rad: [f32; 3],
    pub parts: Vec<TerrariumSeedPartRender>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerrariumFruitPartKind {
    Skin,
    Pulp,
    Core,
    Stem,
}

#[derive(Debug, Clone)]
pub struct TerrariumFruitPartRender {
    pub render_id: u64,
    pub render_fingerprint: u64,
    pub kind: TerrariumFruitPartKind,
    pub mesh: TerrariumTriangleMeshRender,
    pub translation_local: [f32; 3],
    pub rotation_xyz_rad: [f32; 3],
    pub material: TerrariumPbrMaterialRender,
}

#[derive(Debug, Clone)]
pub struct TerrariumFruitRender {
    pub render_id: u64,
    pub translation_world: [f32; 3],
    pub ripeness: f32,
    pub sugar_content: f32,
    pub parts: Vec<TerrariumFruitPartRender>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerrariumRenderPlumeKind {
    Odor,
    GasExchange,
    Microbe,
}

#[derive(Debug, Clone)]
pub struct TerrariumPlumeRender {
    pub render_id: u64,
    pub render_fingerprint: u64,
    pub kind: TerrariumRenderPlumeKind,
    pub primitive: TerrariumMeshPrimitive,
    pub mesh: TerrariumTriangleMeshRender,
    pub translation_world: [f32; 3],
    pub rotation_xyz_rad: [f32; 3],
    pub scale_world: [f32; 3],
    pub intensity: f32,
    pub material: TerrariumPbrMaterialRender,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerrariumFlyPartKind {
    Thorax,
    Head,
    Abdomen,
    Proboscis,
    WingLeft,
    WingRight,
}

#[derive(Debug, Clone)]
pub struct TerrariumFlyPartRender {
    pub render_id: u64,
    pub render_fingerprint: u64,
    pub kind: TerrariumFlyPartKind,
    pub mesh: TerrariumTriangleMeshRender,
    pub translation_local: [f32; 3],
    pub rotation_xyz_rad: [f32; 3],
    pub material: TerrariumPbrMaterialRender,
}

#[derive(Debug, Clone)]
pub struct TerrariumFlyRender {
    pub render_id: u64,
    pub point_light_render_fingerprint: u64,
    pub translation_world: [f32; 3],
    pub body_rotation_yxz_rad: [f32; 3],
    pub parts: Vec<TerrariumFlyPartRender>,
    pub point_light_intensity: f32,
    pub point_light_color_rgb: [f32; 3],
    pub point_light_range: f32,
    pub point_light_translation_local: [f32; 3],
}

#[derive(Debug, Clone, Default)]
pub struct TerrariumDynamicRenderSnapshot {
    pub substrate_voxels: Vec<TerrariumSubstrateVoxelRender>,
    pub explicit_microbes: Vec<TerrariumExplicitMicrobeRender>,
    pub waters: Vec<TerrariumWaterRender>,
    pub plants: Vec<TerrariumPlantRender>,
    pub seeds: Vec<TerrariumSeedRender>,
    pub fruits: Vec<TerrariumFruitRender>,
    pub plumes: Vec<TerrariumPlumeRender>,
    pub flies: Vec<TerrariumFlyRender>,
}

#[derive(Debug, Clone, Copy)]
pub enum TerrariumDynamicMeshTransform {
    WorldXyz {
        translation_world: [f32; 3],
        rotation_xyz_rad: [f32; 3],
    },
    LocalXyz {
        parent_translation_world: [f32; 3],
        parent_rotation_xyz_rad: [f32; 3],
        local_translation: [f32; 3],
        local_rotation_xyz_rad: [f32; 3],
    },
    LocalYxz {
        parent_translation_world: [f32; 3],
        parent_rotation_yxz_rad: [f32; 3],
        local_translation: [f32; 3],
        local_rotation_xyz_rad: [f32; 3],
    },
}

#[derive(Debug, Clone)]
pub struct TerrariumDynamicMeshRenderDelta {
    pub render_id: u64,
    pub render_fingerprint: u64,
    pub mesh_cache_key: u64,
    pub material_state_key: u64,
    pub mesh: TerrariumTriangleMeshRender,
    pub material: TerrariumPbrMaterialRender,
    pub transform: TerrariumDynamicMeshTransform,
    pub hide_on_cutaway: bool,
}

#[derive(Debug, Clone)]
pub struct TerrariumDynamicPointLightRenderDelta {
    pub render_id: u64,
    pub render_fingerprint: u64,
    pub translation_world: [f32; 3],
    pub intensity: f32,
    pub range: f32,
    pub color_rgb: [f32; 3],
}

#[derive(Debug, Clone, Default)]
pub struct TerrariumDynamicRenderDelta {
    pub meshes: Vec<TerrariumDynamicMeshRenderDelta>,
    pub point_lights: Vec<TerrariumDynamicPointLightRenderDelta>,
    pub removed_render_ids: Vec<u64>,
}
