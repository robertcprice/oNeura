//! Snapshot-to-render projection helpers for terrarium scene deltas and batches.
#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use crate::terrarium::{
    add3, mesh_append, mesh_rotate_xyz, mesh_rotate_yxz, mesh_translate, rotate_xyz, rotate_yxz,
    terrarium_dynamic_batch_render_id, terrarium_render_child_id,
    terrarium_substrate_batch_render_id, DYNAMIC_BATCH_CHUNK_WORLD_XZ, DYNAMIC_BATCH_CHUNK_WORLD_Y,
    SUBSTRATE_BATCH_CHUNK_XY, SUBSTRATE_BATCH_CHUNK_Z, TERRARIUM_RENDER_SLOT_FLY_LIGHT,
    TERRARIUM_RENDER_SLOT_MICROBE_BODY, TERRARIUM_RENDER_SLOT_MICROBE_PACKET,
    TERRARIUM_RENDER_SLOT_PLANT_CANOPY, TERRARIUM_RENDER_SLOT_PLANT_STEM,
};
use crate::terrarium_render::{
    TerrariumDynamicBatchKind, TerrariumDynamicBatchRender, TerrariumDynamicMeshRenderDelta,
    TerrariumDynamicMeshTransform, TerrariumDynamicPointLightRenderDelta,
    TerrariumDynamicRenderDelta, TerrariumDynamicRenderSnapshot, TerrariumPbrMaterialRender,
    TerrariumSubstrateBatchRender, TerrariumTriangleMeshRender,
};

fn hash_f32(value: f32, hasher: &mut std::collections::hash_map::DefaultHasher) {
    value.to_bits().hash(hasher);
}

fn hash_vec3(value: [f32; 3], hasher: &mut std::collections::hash_map::DefaultHasher) {
    hash_f32(value[0], hasher);
    hash_f32(value[1], hasher);
    hash_f32(value[2], hasher);
}

fn hash_vec4(value: [f32; 4], hasher: &mut std::collections::hash_map::DefaultHasher) {
    hash_f32(value[0], hasher);
    hash_f32(value[1], hasher);
    hash_f32(value[2], hasher);
    hash_f32(value[3], hasher);
}

fn hash_material_render_state(
    material: &TerrariumPbrMaterialRender,
    hasher: &mut std::collections::hash_map::DefaultHasher,
) {
    hash_vec4(material.base_color_rgba, hasher);
    hash_vec3(material.emissive_rgb, hasher);
    hash_f32(material.metallic, hasher);
    hash_f32(material.perceptual_roughness, hasher);
    hash_f32(material.reflectance, hasher);
    material.alpha_blend.hash(hasher);
    material.double_sided.hash(hasher);
    hash_vec4(material.shader_atmosphere_rgba, hasher);
    hash_vec4(material.shader_dynamics_rgba, hasher);
    material.shader_flags.hash(hasher);
}

fn hash_mesh_render_state(
    mesh: &TerrariumTriangleMeshRender,
    hasher: &mut std::collections::hash_map::DefaultHasher,
) {
    mesh.positions.len().hash(hasher);
    for position in &mesh.positions {
        hash_vec3(*position, hasher);
    }
    mesh.normals.len().hash(hasher);
    for normal in &mesh.normals {
        hash_vec3(*normal, hasher);
    }
    mesh.uvs.len().hash(hasher);
    for uv in &mesh.uvs {
        hash_f32(uv[0], hasher);
        hash_f32(uv[1], hasher);
    }
    mesh.indices.hash(hasher);
}

fn visual_render_fingerprint(
    render_id: u64,
    translation_world: [f32; 3],
    rotation_xyz_rad: [f32; 3],
    mesh: &TerrariumTriangleMeshRender,
    material: &TerrariumPbrMaterialRender,
) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    render_id.hash(&mut hasher);
    hash_vec3(translation_world, &mut hasher);
    hash_vec3(rotation_xyz_rad, &mut hasher);
    hash_mesh_render_state(mesh, &mut hasher);
    hash_material_render_state(material, &mut hasher);
    hasher.finish()
}

fn point_light_render_fingerprint(
    render_id: u64,
    translation_world: [f32; 3],
    body_rotation_yxz_rad: [f32; 3],
    translation_local: [f32; 3],
    intensity: f32,
    range: f32,
    color_rgb: [f32; 3],
) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    render_id.hash(&mut hasher);
    hash_vec3(translation_world, &mut hasher);
    hash_vec3(body_rotation_yxz_rad, &mut hasher);
    hash_vec3(translation_local, &mut hasher);
    hash_f32(intensity, &mut hasher);
    hash_f32(range, &mut hasher);
    hash_vec3(color_rgb, &mut hasher);
    hasher.finish()
}

pub(crate) fn material_render_state_key(material: &TerrariumPbrMaterialRender) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    hash_material_render_state(material, &mut hasher);
    hasher.finish()
}

pub(crate) fn mesh_render_state_key(mesh: &TerrariumTriangleMeshRender) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    hash_mesh_render_state(mesh, &mut hasher);
    hasher.finish()
}

pub(crate) fn stamp_dynamic_render_fingerprints(snapshot: &mut TerrariumDynamicRenderSnapshot) {
    for voxel in &mut snapshot.substrate_voxels {
        voxel.render_fingerprint = visual_render_fingerprint(
            voxel.render_id,
            voxel.translation_world,
            [0.0, 0.0, 0.0],
            &voxel.mesh,
            &voxel.material,
        );
    }
    for microbe in &mut snapshot.explicit_microbes {
        microbe.body_render_fingerprint = visual_render_fingerprint(
            microbe.render_id,
            microbe.translation_world,
            [0.0, 0.0, 0.0],
            &microbe.body_mesh,
            &microbe.body_material,
        );
        microbe.packet_population_render_fingerprint = visual_render_fingerprint(
            microbe.render_id,
            microbe.translation_world,
            [0.0, 0.0, 0.0],
            &microbe.packet_mesh,
            &microbe.packet_material,
        );
        for packet in &mut microbe.packets {
            packet.render_fingerprint = visual_render_fingerprint(
                packet.render_id,
                [
                    microbe.translation_world[0] + packet.translation_local[0],
                    microbe.translation_world[1] + packet.translation_local[1],
                    microbe.translation_world[2] + packet.translation_local[2],
                ],
                [0.0, 0.0, 0.0],
                &packet.mesh,
                &packet.material,
            );
        }
        for site in &mut microbe.sites {
            site.render_fingerprint = visual_render_fingerprint(
                site.render_id,
                [
                    microbe.translation_world[0] + site.translation_local[0],
                    microbe.translation_world[1] + site.translation_local[1],
                    microbe.translation_world[2] + site.translation_local[2],
                ],
                [0.0, 0.0, 0.0],
                &site.mesh,
                &site.material,
            );
        }
    }
    for water in &mut snapshot.waters {
        water.render_fingerprint = visual_render_fingerprint(
            water.render_id,
            water.translation_world,
            [0.0, 0.0, 0.0],
            &water.mesh,
            &water.material,
        );
    }
    for plant in &mut snapshot.plants {
        plant.stem_render_fingerprint = visual_render_fingerprint(
            plant.render_id,
            plant.stem_translation_world,
            plant.stem_rotation_xyz_rad,
            &plant.stem_mesh,
            &plant.stem_material,
        );
        plant.canopy_render_fingerprint = visual_render_fingerprint(
            plant.render_id,
            plant.canopy_translation_world,
            plant.canopy_rotation_xyz_rad,
            &plant.canopy_mesh,
            &plant.canopy_material,
        );
        for tissue in &mut plant.tissues {
            tissue.render_fingerprint = visual_render_fingerprint(
                tissue.render_id,
                tissue.translation_world,
                tissue.rotation_xyz_rad,
                &tissue.mesh,
                &tissue.material,
            );
        }
    }
    for seed in &mut snapshot.seeds {
        for part in &mut seed.parts {
            part.render_fingerprint = visual_render_fingerprint(
                part.render_id,
                [
                    seed.translation_world[0] + part.translation_local[0],
                    seed.translation_world[1] + part.translation_local[1],
                    seed.translation_world[2] + part.translation_local[2],
                ],
                [
                    seed.rotation_xyz_rad[0] + part.rotation_xyz_rad[0],
                    seed.rotation_xyz_rad[1] + part.rotation_xyz_rad[1],
                    seed.rotation_xyz_rad[2] + part.rotation_xyz_rad[2],
                ],
                &part.mesh,
                &part.material,
            );
        }
    }
    for fruit in &mut snapshot.fruits {
        for part in &mut fruit.parts {
            part.render_fingerprint = visual_render_fingerprint(
                part.render_id,
                [
                    fruit.translation_world[0] + part.translation_local[0],
                    fruit.translation_world[1] + part.translation_local[1],
                    fruit.translation_world[2] + part.translation_local[2],
                ],
                part.rotation_xyz_rad,
                &part.mesh,
                &part.material,
            );
        }
    }
    for plume in &mut snapshot.plumes {
        plume.render_fingerprint = visual_render_fingerprint(
            plume.render_id,
            plume.translation_world,
            plume.rotation_xyz_rad,
            &plume.mesh,
            &plume.material,
        );
    }
    for fly in &mut snapshot.flies {
        for part in &mut fly.parts {
            part.render_fingerprint = visual_render_fingerprint(
                part.render_id,
                [
                    fly.translation_world[0] + part.translation_local[0],
                    fly.translation_world[1] + part.translation_local[1],
                    fly.translation_world[2] + part.translation_local[2],
                ],
                [
                    fly.body_rotation_yxz_rad[0] + part.rotation_xyz_rad[0],
                    fly.body_rotation_yxz_rad[1] + part.rotation_xyz_rad[1],
                    fly.body_rotation_yxz_rad[2] + part.rotation_xyz_rad[2],
                ],
                &part.mesh,
                &part.material,
            );
        }
        fly.point_light_render_fingerprint = point_light_render_fingerprint(
            fly.render_id,
            fly.translation_world,
            fly.body_rotation_yxz_rad,
            fly.point_light_translation_local,
            fly.point_light_intensity,
            fly.point_light_range,
            fly.point_light_color_rgb,
        );
    }
}

fn snapshot_render_fingerprints(snapshot: &TerrariumDynamicRenderSnapshot) -> HashMap<u64, u64> {
    let mut fingerprints = HashMap::new();
    for voxel in &snapshot.substrate_voxels {
        fingerprints.insert(voxel.render_id, voxel.render_fingerprint);
    }
    for microbe in &snapshot.explicit_microbes {
        fingerprints.insert(
            terrarium_render_child_id(microbe.render_id, TERRARIUM_RENDER_SLOT_MICROBE_BODY),
            microbe.body_render_fingerprint,
        );
        fingerprints.insert(
            terrarium_render_child_id(microbe.render_id, TERRARIUM_RENDER_SLOT_MICROBE_PACKET),
            microbe.packet_population_render_fingerprint,
        );
        for packet in &microbe.packets {
            fingerprints.insert(packet.render_id, packet.render_fingerprint);
        }
        for site in &microbe.sites {
            fingerprints.insert(site.render_id, site.render_fingerprint);
        }
    }
    for plant in &snapshot.plants {
        if plant.tissues.is_empty() {
            fingerprints.insert(
                terrarium_render_child_id(plant.render_id, TERRARIUM_RENDER_SLOT_PLANT_STEM),
                plant.stem_render_fingerprint,
            );
            fingerprints.insert(
                terrarium_render_child_id(plant.render_id, TERRARIUM_RENDER_SLOT_PLANT_CANOPY),
                plant.canopy_render_fingerprint,
            );
        }
        for tissue in &plant.tissues {
            fingerprints.insert(tissue.render_id, tissue.render_fingerprint);
        }
    }
    for seed in &snapshot.seeds {
        for part in &seed.parts {
            fingerprints.insert(part.render_id, part.render_fingerprint);
        }
    }
    for fruit in &snapshot.fruits {
        for part in &fruit.parts {
            fingerprints.insert(part.render_id, part.render_fingerprint);
        }
    }
    for plume in &snapshot.plumes {
        fingerprints.insert(plume.render_id, plume.render_fingerprint);
    }
    for fly in &snapshot.flies {
        for part in &fly.parts {
            fingerprints.insert(part.render_id, part.render_fingerprint);
        }
        fingerprints.insert(
            terrarium_render_child_id(fly.render_id, TERRARIUM_RENDER_SLOT_FLY_LIGHT),
            fly.point_light_render_fingerprint,
        );
    }
    fingerprints
}

fn push_dynamic_mesh_delta(
    delta: &mut TerrariumDynamicRenderDelta,
    render_id: u64,
    render_fingerprint: u64,
    mesh: &TerrariumTriangleMeshRender,
    material: &TerrariumPbrMaterialRender,
    transform: TerrariumDynamicMeshTransform,
    hide_on_cutaway: bool,
) {
    delta.meshes.push(TerrariumDynamicMeshRenderDelta {
        render_id,
        render_fingerprint,
        mesh_cache_key: mesh_render_state_key(mesh),
        material_state_key: material_render_state_key(material),
        mesh: mesh.clone(),
        material: material.clone(),
        transform,
        hide_on_cutaway,
    });
}

pub(crate) fn build_dynamic_render_delta(
    snapshot: &TerrariumDynamicRenderSnapshot,
    previous_fingerprints: &HashMap<u64, u64>,
) -> (TerrariumDynamicRenderDelta, HashMap<u64, u64>) {
    let current_fingerprints = snapshot_render_fingerprints(snapshot);
    let changed_render_ids = current_fingerprints
        .iter()
        .filter_map(|(render_id, fingerprint)| {
            if previous_fingerprints.get(render_id).copied() == Some(*fingerprint) {
                None
            } else {
                Some(*render_id)
            }
        })
        .collect::<HashSet<_>>();

    let mut delta = TerrariumDynamicRenderDelta::default();
    delta.removed_render_ids = previous_fingerprints
        .keys()
        .filter(|render_id| !current_fingerprints.contains_key(render_id))
        .copied()
        .collect();

    for voxel in &snapshot.substrate_voxels {
        if changed_render_ids.contains(&voxel.render_id) {
            push_dynamic_mesh_delta(
                &mut delta,
                voxel.render_id,
                voxel.render_fingerprint,
                &voxel.mesh,
                &voxel.material,
                TerrariumDynamicMeshTransform::WorldXyz {
                    translation_world: voxel.translation_world,
                    rotation_xyz_rad: [0.0, 0.0, 0.0],
                },
                voxel.voxel[2] == 0,
            );
        }
    }
    for microbe in &snapshot.explicit_microbes {
        let body_render_id =
            terrarium_render_child_id(microbe.render_id, TERRARIUM_RENDER_SLOT_MICROBE_BODY);
        if changed_render_ids.contains(&body_render_id) && !microbe.body_mesh.indices.is_empty() {
            push_dynamic_mesh_delta(
                &mut delta,
                body_render_id,
                microbe.body_render_fingerprint,
                &microbe.body_mesh,
                &microbe.body_material,
                TerrariumDynamicMeshTransform::WorldXyz {
                    translation_world: microbe.translation_world,
                    rotation_xyz_rad: [0.0, 0.0, 0.0],
                },
                false,
            );
        }
        let packet_render_id =
            terrarium_render_child_id(microbe.render_id, TERRARIUM_RENDER_SLOT_MICROBE_PACKET);
        if changed_render_ids.contains(&packet_render_id) && !microbe.packet_mesh.indices.is_empty()
        {
            push_dynamic_mesh_delta(
                &mut delta,
                packet_render_id,
                microbe.packet_population_render_fingerprint,
                &microbe.packet_mesh,
                &microbe.packet_material,
                TerrariumDynamicMeshTransform::WorldXyz {
                    translation_world: microbe.translation_world,
                    rotation_xyz_rad: [0.0, 0.0, 0.0],
                },
                false,
            );
        }
        for packet in &microbe.packets {
            if changed_render_ids.contains(&packet.render_id) {
                push_dynamic_mesh_delta(
                    &mut delta,
                    packet.render_id,
                    packet.render_fingerprint,
                    &packet.mesh,
                    &packet.material,
                    TerrariumDynamicMeshTransform::WorldXyz {
                        translation_world: [
                            microbe.translation_world[0] + packet.translation_local[0],
                            microbe.translation_world[1] + packet.translation_local[1],
                            microbe.translation_world[2] + packet.translation_local[2],
                        ],
                        rotation_xyz_rad: [0.0, 0.0, 0.0],
                    },
                    false,
                );
            }
        }
        for site in &microbe.sites {
            if changed_render_ids.contains(&site.render_id) {
                push_dynamic_mesh_delta(
                    &mut delta,
                    site.render_id,
                    site.render_fingerprint,
                    &site.mesh,
                    &site.material,
                    TerrariumDynamicMeshTransform::WorldXyz {
                        translation_world: [
                            microbe.translation_world[0] + site.translation_local[0],
                            microbe.translation_world[1] + site.translation_local[1],
                            microbe.translation_world[2] + site.translation_local[2],
                        ],
                        rotation_xyz_rad: [0.0, 0.0, 0.0],
                    },
                    false,
                );
            }
        }
    }
    for plant in &snapshot.plants {
        if plant.tissues.is_empty() {
            let stem_render_id =
                terrarium_render_child_id(plant.render_id, TERRARIUM_RENDER_SLOT_PLANT_STEM);
            if changed_render_ids.contains(&stem_render_id) && !plant.stem_mesh.indices.is_empty() {
                push_dynamic_mesh_delta(
                    &mut delta,
                    stem_render_id,
                    plant.stem_render_fingerprint,
                    &plant.stem_mesh,
                    &plant.stem_material,
                    TerrariumDynamicMeshTransform::WorldXyz {
                        translation_world: plant.stem_translation_world,
                        rotation_xyz_rad: plant.stem_rotation_xyz_rad,
                    },
                    false,
                );
            }
            let canopy_render_id =
                terrarium_render_child_id(plant.render_id, TERRARIUM_RENDER_SLOT_PLANT_CANOPY);
            if changed_render_ids.contains(&canopy_render_id)
                && !plant.canopy_mesh.indices.is_empty()
            {
                push_dynamic_mesh_delta(
                    &mut delta,
                    canopy_render_id,
                    plant.canopy_render_fingerprint,
                    &plant.canopy_mesh,
                    &plant.canopy_material,
                    TerrariumDynamicMeshTransform::WorldXyz {
                        translation_world: plant.canopy_translation_world,
                        rotation_xyz_rad: plant.canopy_rotation_xyz_rad,
                    },
                    false,
                );
            }
        }
        for tissue in &plant.tissues {
            if changed_render_ids.contains(&tissue.render_id) {
                push_dynamic_mesh_delta(
                    &mut delta,
                    tissue.render_id,
                    tissue.render_fingerprint,
                    &tissue.mesh,
                    &tissue.material,
                    TerrariumDynamicMeshTransform::WorldXyz {
                        translation_world: tissue.translation_world,
                        rotation_xyz_rad: tissue.rotation_xyz_rad,
                    },
                    false,
                );
            }
        }
    }
    for seed in &snapshot.seeds {
        for part in &seed.parts {
            if changed_render_ids.contains(&part.render_id) {
                push_dynamic_mesh_delta(
                    &mut delta,
                    part.render_id,
                    part.render_fingerprint,
                    &part.mesh,
                    &part.material,
                    TerrariumDynamicMeshTransform::LocalXyz {
                        parent_translation_world: seed.translation_world,
                        parent_rotation_xyz_rad: seed.rotation_xyz_rad,
                        local_translation: part.translation_local,
                        local_rotation_xyz_rad: part.rotation_xyz_rad,
                    },
                    false,
                );
            }
        }
    }
    for fruit in &snapshot.fruits {
        for part in &fruit.parts {
            if changed_render_ids.contains(&part.render_id) {
                push_dynamic_mesh_delta(
                    &mut delta,
                    part.render_id,
                    part.render_fingerprint,
                    &part.mesh,
                    &part.material,
                    TerrariumDynamicMeshTransform::WorldXyz {
                        translation_world: [
                            fruit.translation_world[0] + part.translation_local[0],
                            fruit.translation_world[1] + part.translation_local[1],
                            fruit.translation_world[2] + part.translation_local[2],
                        ],
                        rotation_xyz_rad: part.rotation_xyz_rad,
                    },
                    false,
                );
            }
        }
    }
    for plume in &snapshot.plumes {
        if changed_render_ids.contains(&plume.render_id) {
            push_dynamic_mesh_delta(
                &mut delta,
                plume.render_id,
                plume.render_fingerprint,
                &plume.mesh,
                &plume.material,
                TerrariumDynamicMeshTransform::WorldXyz {
                    translation_world: plume.translation_world,
                    rotation_xyz_rad: plume.rotation_xyz_rad,
                },
                false,
            );
        }
    }
    for fly in &snapshot.flies {
        for part in &fly.parts {
            if changed_render_ids.contains(&part.render_id) {
                push_dynamic_mesh_delta(
                    &mut delta,
                    part.render_id,
                    part.render_fingerprint,
                    &part.mesh,
                    &part.material,
                    TerrariumDynamicMeshTransform::LocalYxz {
                        parent_translation_world: fly.translation_world,
                        parent_rotation_yxz_rad: fly.body_rotation_yxz_rad,
                        local_translation: part.translation_local,
                        local_rotation_xyz_rad: part.rotation_xyz_rad,
                    },
                    false,
                );
            }
        }
        let light_render_id =
            terrarium_render_child_id(fly.render_id, TERRARIUM_RENDER_SLOT_FLY_LIGHT);
        if changed_render_ids.contains(&light_render_id) {
            delta
                .point_lights
                .push(TerrariumDynamicPointLightRenderDelta {
                    render_id: light_render_id,
                    render_fingerprint: fly.point_light_render_fingerprint,
                    translation_world: add3(
                        fly.translation_world,
                        rotate_yxz(fly.point_light_translation_local, fly.body_rotation_yxz_rad),
                    ),
                    intensity: fly.point_light_intensity,
                    range: fly.point_light_range,
                    color_rgb: fly.point_light_color_rgb,
                });
        }
    }

    (delta, current_fingerprints)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SubstrateBatchKey {
    chunk_x: usize,
    chunk_y: usize,
    chunk_z: usize,
    material_state_key: u64,
    hide_on_cutaway: bool,
}

#[derive(Debug, Clone)]
struct SubstrateBatchAccumulator {
    render_id: u64,
    material_state_key: u64,
    mesh: TerrariumTriangleMeshRender,
    material: TerrariumPbrMaterialRender,
    hide_on_cutaway: bool,
}

pub(crate) fn build_substrate_batch_renders(
    snapshot: &TerrariumDynamicRenderSnapshot,
) -> Vec<TerrariumSubstrateBatchRender> {
    let mut accumulators = HashMap::<SubstrateBatchKey, SubstrateBatchAccumulator>::new();
    for voxel in &snapshot.substrate_voxels {
        if voxel.mesh.indices.is_empty() {
            continue;
        }
        let hide_on_cutaway = voxel.voxel[2] == 0;
        let material_state_key = material_render_state_key(&voxel.material);
        let key = SubstrateBatchKey {
            chunk_x: voxel.voxel[0] / SUBSTRATE_BATCH_CHUNK_XY.max(1),
            chunk_y: voxel.voxel[1] / SUBSTRATE_BATCH_CHUNK_XY.max(1),
            chunk_z: voxel.voxel[2] / SUBSTRATE_BATCH_CHUNK_Z.max(1),
            material_state_key,
            hide_on_cutaway,
        };
        let render_id = terrarium_substrate_batch_render_id(
            key.chunk_x,
            key.chunk_y,
            key.chunk_z,
            key.material_state_key,
            key.hide_on_cutaway,
        );
        let entry = accumulators
            .entry(key)
            .or_insert_with(|| SubstrateBatchAccumulator {
                render_id,
                material_state_key,
                mesh: TerrariumTriangleMeshRender::default(),
                material: voxel.material.clone(),
                hide_on_cutaway,
            });
        let mut translated_mesh = voxel.mesh.clone();
        mesh_translate(&mut translated_mesh, voxel.translation_world);
        mesh_append(&mut entry.mesh, &translated_mesh);
    }
    let mut batches = accumulators
        .into_values()
        .filter(|batch| !batch.mesh.indices.is_empty())
        .map(|batch| {
            let mesh_cache_key = mesh_render_state_key(&batch.mesh);
            let render_fingerprint = visual_render_fingerprint(
                batch.render_id,
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                &batch.mesh,
                &batch.material,
            );
            TerrariumSubstrateBatchRender {
                render_id: batch.render_id,
                render_fingerprint,
                mesh_cache_key,
                material_state_key: batch.material_state_key,
                mesh: batch.mesh,
                material: batch.material,
                hide_on_cutaway: batch.hide_on_cutaway,
            }
        })
        .collect::<Vec<_>>();
    batches.sort_by_key(|batch| batch.render_id);
    batches
}

fn dynamic_batch_chunk_coord(value: f32, span: f32) -> i32 {
    (value / span.max(1.0e-3)).floor() as i32
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct DynamicBatchKey {
    kind: TerrariumDynamicBatchKind,
    chunk_x: i32,
    chunk_y: i32,
    chunk_z: i32,
    material_state_key: u64,
}

#[derive(Debug, Clone)]
struct DynamicBatchAccumulator {
    render_id: u64,
    kind: TerrariumDynamicBatchKind,
    material_state_key: u64,
    mesh: TerrariumTriangleMeshRender,
    material: TerrariumPbrMaterialRender,
}

fn accumulate_dynamic_batch_world_mesh(
    accumulators: &mut HashMap<DynamicBatchKey, DynamicBatchAccumulator>,
    kind: TerrariumDynamicBatchKind,
    anchor_world: [f32; 3],
    world_mesh: TerrariumTriangleMeshRender,
    material: &TerrariumPbrMaterialRender,
) {
    if world_mesh.indices.is_empty() {
        return;
    }
    let material_state_key = material_render_state_key(material);
    let key = DynamicBatchKey {
        kind,
        chunk_x: dynamic_batch_chunk_coord(anchor_world[0], DYNAMIC_BATCH_CHUNK_WORLD_XZ),
        chunk_y: dynamic_batch_chunk_coord(anchor_world[1], DYNAMIC_BATCH_CHUNK_WORLD_Y),
        chunk_z: dynamic_batch_chunk_coord(anchor_world[2], DYNAMIC_BATCH_CHUNK_WORLD_XZ),
        material_state_key,
    };
    let render_id = terrarium_dynamic_batch_render_id(
        key.kind,
        key.chunk_x,
        key.chunk_y,
        key.chunk_z,
        key.material_state_key,
    );
    let entry = accumulators
        .entry(key)
        .or_insert_with(|| DynamicBatchAccumulator {
            render_id,
            kind,
            material_state_key,
            mesh: TerrariumTriangleMeshRender::default(),
            material: material.clone(),
        });
    mesh_append(&mut entry.mesh, &world_mesh);
}

fn accumulate_dynamic_batch_mesh(
    accumulators: &mut HashMap<DynamicBatchKey, DynamicBatchAccumulator>,
    kind: TerrariumDynamicBatchKind,
    translation_world: [f32; 3],
    rotation_xyz_rad: [f32; 3],
    mesh: &TerrariumTriangleMeshRender,
    material: &TerrariumPbrMaterialRender,
) {
    if mesh.indices.is_empty() {
        return;
    }
    let mut transformed_mesh = mesh.clone();
    if rotation_xyz_rad != [0.0, 0.0, 0.0] {
        mesh_rotate_xyz(&mut transformed_mesh, rotation_xyz_rad);
    }
    mesh_translate(&mut transformed_mesh, translation_world);
    accumulate_dynamic_batch_world_mesh(
        accumulators,
        kind,
        translation_world,
        transformed_mesh,
        material,
    );
}

pub(crate) fn build_dynamic_batch_renders(
    snapshot: &TerrariumDynamicRenderSnapshot,
) -> Vec<TerrariumDynamicBatchRender> {
    let mut accumulators = HashMap::<DynamicBatchKey, DynamicBatchAccumulator>::new();
    for microbe in &snapshot.explicit_microbes {
        for packet in &microbe.packets {
            accumulate_dynamic_batch_mesh(
                &mut accumulators,
                TerrariumDynamicBatchKind::MicrobePacket,
                [
                    microbe.translation_world[0] + packet.translation_local[0],
                    microbe.translation_world[1] + packet.translation_local[1],
                    microbe.translation_world[2] + packet.translation_local[2],
                ],
                [0.0, 0.0, 0.0],
                &packet.mesh,
                &packet.material,
            );
        }
        for site in &microbe.sites {
            accumulate_dynamic_batch_mesh(
                &mut accumulators,
                TerrariumDynamicBatchKind::MicrobeSite,
                [
                    microbe.translation_world[0] + site.translation_local[0],
                    microbe.translation_world[1] + site.translation_local[1],
                    microbe.translation_world[2] + site.translation_local[2],
                ],
                [0.0, 0.0, 0.0],
                &site.mesh,
                &site.material,
            );
        }
    }
    for water in &snapshot.waters {
        accumulate_dynamic_batch_mesh(
            &mut accumulators,
            TerrariumDynamicBatchKind::Water,
            water.translation_world,
            [0.0, 0.0, 0.0],
            &water.mesh,
            &water.material,
        );
    }
    for plant in &snapshot.plants {
        for tissue in &plant.tissues {
            accumulate_dynamic_batch_mesh(
                &mut accumulators,
                TerrariumDynamicBatchKind::PlantTissue,
                tissue.translation_world,
                tissue.rotation_xyz_rad,
                &tissue.mesh,
                &tissue.material,
            );
        }
    }
    for seed in &snapshot.seeds {
        for part in &seed.parts {
            let anchor_world = add3(
                seed.translation_world,
                rotate_xyz(part.translation_local, seed.rotation_xyz_rad),
            );
            let mut transformed_mesh = part.mesh.clone();
            if part.rotation_xyz_rad != [0.0, 0.0, 0.0] {
                mesh_rotate_xyz(&mut transformed_mesh, part.rotation_xyz_rad);
            }
            mesh_translate(&mut transformed_mesh, part.translation_local);
            if seed.rotation_xyz_rad != [0.0, 0.0, 0.0] {
                mesh_rotate_xyz(&mut transformed_mesh, seed.rotation_xyz_rad);
            }
            mesh_translate(&mut transformed_mesh, seed.translation_world);
            accumulate_dynamic_batch_world_mesh(
                &mut accumulators,
                TerrariumDynamicBatchKind::SeedPart,
                anchor_world,
                transformed_mesh,
                &part.material,
            );
        }
    }
    for fruit in &snapshot.fruits {
        for part in &fruit.parts {
            accumulate_dynamic_batch_mesh(
                &mut accumulators,
                TerrariumDynamicBatchKind::FruitPart,
                [
                    fruit.translation_world[0] + part.translation_local[0],
                    fruit.translation_world[1] + part.translation_local[1],
                    fruit.translation_world[2] + part.translation_local[2],
                ],
                part.rotation_xyz_rad,
                &part.mesh,
                &part.material,
            );
        }
    }
    for plume in &snapshot.plumes {
        accumulate_dynamic_batch_mesh(
            &mut accumulators,
            TerrariumDynamicBatchKind::Plume,
            plume.translation_world,
            plume.rotation_xyz_rad,
            &plume.mesh,
            &plume.material,
        );
    }
    for fly in &snapshot.flies {
        for part in &fly.parts {
            let anchor_world = add3(
                fly.translation_world,
                rotate_yxz(part.translation_local, fly.body_rotation_yxz_rad),
            );
            let mut transformed_mesh = part.mesh.clone();
            if part.rotation_xyz_rad != [0.0, 0.0, 0.0] {
                mesh_rotate_xyz(&mut transformed_mesh, part.rotation_xyz_rad);
            }
            mesh_translate(&mut transformed_mesh, part.translation_local);
            if fly.body_rotation_yxz_rad != [0.0, 0.0, 0.0] {
                mesh_rotate_yxz(&mut transformed_mesh, fly.body_rotation_yxz_rad);
            }
            mesh_translate(&mut transformed_mesh, fly.translation_world);
            accumulate_dynamic_batch_world_mesh(
                &mut accumulators,
                TerrariumDynamicBatchKind::FlyPart,
                anchor_world,
                transformed_mesh,
                &part.material,
            );
        }
    }
    let mut batches = accumulators
        .into_values()
        .filter(|batch| !batch.mesh.indices.is_empty())
        .map(|batch| {
            let mesh_cache_key = mesh_render_state_key(&batch.mesh);
            let render_fingerprint = visual_render_fingerprint(
                batch.render_id,
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                &batch.mesh,
                &batch.material,
            );
            TerrariumDynamicBatchRender {
                render_id: batch.render_id,
                render_fingerprint,
                mesh_cache_key,
                material_state_key: batch.material_state_key,
                kind: batch.kind,
                mesh: batch.mesh,
                material: batch.material,
            }
        })
        .collect::<Vec<_>>();
    batches.sort_by_key(|batch| batch.render_id);
    batches
}
