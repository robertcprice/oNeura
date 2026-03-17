use crate::terrarium_render::{
    TerrariumDynamicMeshTransform, TerrariumDynamicRenderSnapshot, TerrariumPbrMaterialRender,
    TerrariumTriangleMeshRender,
};
use crate::terrarium_render_compat::{
    terrarium_render_child_id, terrarium_render_id_class, TERRARIUM_RENDER_SLOT_MICROBE_BODY,
    TERRARIUM_RENDER_SLOT_MICROBE_PACKET, TERRARIUM_RENDER_SLOT_PLANT_CANOPY,
    TERRARIUM_RENDER_SLOT_PLANT_STEM, TERRARIUM_SHADER_FLAG_DEBUG_OVERLAY,
    TERRARIUM_SHADER_FLAG_PLUME,
};

const RAYCAST_BVH_LEAF_SURFACES: usize = 8;

#[derive(Debug, Clone, Copy)]
pub struct TerrariumSceneRaycastHit {
    pub render_id: u64,
    pub render_fingerprint: u64,
    pub distance: f32,
    pub position_world: [f32; 3],
    pub normal_world: [f32; 3],
}

#[derive(Debug, Clone)]
pub(crate) struct TerrariumRaycastSurface {
    pub(crate) render_id: u64,
    pub(crate) render_fingerprint: u64,
    pub(crate) hide_on_cutaway: bool,
    pub(crate) solid: bool,
    pub(crate) aabb_min_world: [f32; 3],
    pub(crate) aabb_max_world: [f32; 3],
    pub(crate) world_positions: Vec<[f32; 3]>,
    pub(crate) indices: Vec<u32>,
}

#[derive(Debug, Clone)]
pub(crate) struct TerrariumRaycastBvhNode {
    pub(crate) aabb_min_world: [f32; 3],
    pub(crate) aabb_max_world: [f32; 3],
    pub(crate) left_child: Option<usize>,
    pub(crate) right_child: Option<usize>,
    pub(crate) surface_indices: Vec<usize>,
    pub(crate) has_any_surface: bool,
    pub(crate) has_cutaway_visible_surface: bool,
    pub(crate) has_solid_surface: bool,
    pub(crate) has_cutaway_visible_solid_surface: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct CachedDynamicRaycastScene {
    pub(crate) revision: u64,
    pub(crate) surfaces: Vec<TerrariumRaycastSurface>,
    pub(crate) bvh_nodes: Vec<TerrariumRaycastBvhNode>,
    pub(crate) bvh_root: Option<usize>,
}

impl Default for CachedDynamicRaycastScene {
    fn default() -> Self {
        Self {
            revision: u64::MAX,
            surfaces: Vec::new(),
            bvh_nodes: Vec::new(),
            bvh_root: None,
        }
    }
}

impl CachedDynamicRaycastScene {
    pub(crate) fn sync_with_snapshot(
        &mut self,
        revision: u64,
        snapshot: &TerrariumDynamicRenderSnapshot,
    ) {
        if self.revision == revision {
            return;
        }

        let (surfaces, bvh_nodes, bvh_root) = build_dynamic_raycast_scene(snapshot);
        self.surfaces = surfaces;
        self.bvh_nodes = bvh_nodes;
        self.bvh_root = bvh_root;
        self.revision = revision;
    }

    pub(crate) fn raycast(
        &self,
        origin_world: [f32; 3],
        direction_world: [f32; 3],
        cutaway: bool,
        solid_only: bool,
        excluded_class_mask: u16,
    ) -> Option<TerrariumSceneRaycastHit> {
        raycast_scene_internal(
            &self.surfaces,
            &self.bvh_nodes,
            self.bvh_root,
            origin_world,
            direction_world,
            cutaway,
            solid_only,
            excluded_class_mask,
        )
    }
}

fn normalize3(value: [f32; 3]) -> [f32; 3] {
    let length = (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt();
    if length <= 1.0e-6 {
        [0.0, 1.0, 0.0]
    } else {
        [value[0] / length, value[1] / length, value[2] / length]
    }
}

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn mul3(value: [f32; 3], scalar: f32) -> [f32; 3] {
    [value[0] * scalar, value[1] * scalar, value[2] * scalar]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn rotate_about_axis(value: [f32; 3], axis: [f32; 3], angle: f32) -> [f32; 3] {
    let axis = normalize3(axis);
    let cos_angle = angle.cos();
    let sin_angle = angle.sin();
    let term_parallel = mul3(axis, dot3(axis, value) * (1.0 - cos_angle));
    let term_cross = mul3(cross3(axis, value), sin_angle);
    add3(add3(mul3(value, cos_angle), term_cross), term_parallel)
}

fn rotate_xyz(value: [f32; 3], rotation_xyz_rad: [f32; 3]) -> [f32; 3] {
    let mut rotated = value;
    if rotation_xyz_rad[0].abs() > 1.0e-6 {
        rotated = rotate_about_axis(rotated, [1.0, 0.0, 0.0], rotation_xyz_rad[0]);
    }
    if rotation_xyz_rad[1].abs() > 1.0e-6 {
        rotated = rotate_about_axis(rotated, [0.0, 1.0, 0.0], rotation_xyz_rad[1]);
    }
    if rotation_xyz_rad[2].abs() > 1.0e-6 {
        rotated = rotate_about_axis(rotated, [0.0, 0.0, 1.0], rotation_xyz_rad[2]);
    }
    rotated
}

fn rotate_yxz(value: [f32; 3], rotation_yxz_rad: [f32; 3]) -> [f32; 3] {
    let mut rotated = value;
    if rotation_yxz_rad[0].abs() > 1.0e-6 {
        rotated = rotate_about_axis(rotated, [0.0, 1.0, 0.0], rotation_yxz_rad[0]);
    }
    if rotation_yxz_rad[1].abs() > 1.0e-6 {
        rotated = rotate_about_axis(rotated, [1.0, 0.0, 0.0], rotation_yxz_rad[1]);
    }
    if rotation_yxz_rad[2].abs() > 1.0e-6 {
        rotated = rotate_about_axis(rotated, [0.0, 0.0, 1.0], rotation_yxz_rad[2]);
    }
    rotated
}

fn dynamic_mesh_world_point(
    transform: TerrariumDynamicMeshTransform,
    local_point: [f32; 3],
) -> [f32; 3] {
    match transform {
        TerrariumDynamicMeshTransform::WorldXyz {
            translation_world,
            rotation_xyz_rad,
        } => add3(translation_world, rotate_xyz(local_point, rotation_xyz_rad)),
        TerrariumDynamicMeshTransform::LocalXyz {
            parent_translation_world,
            parent_rotation_xyz_rad,
            local_translation,
            local_rotation_xyz_rad,
        } => {
            let local_point = add3(
                local_translation,
                rotate_xyz(local_point, local_rotation_xyz_rad),
            );
            add3(
                parent_translation_world,
                rotate_xyz(local_point, parent_rotation_xyz_rad),
            )
        }
        TerrariumDynamicMeshTransform::LocalYxz {
            parent_translation_world,
            parent_rotation_yxz_rad,
            local_translation,
            local_rotation_xyz_rad,
        } => {
            let local_point = add3(
                local_translation,
                rotate_xyz(local_point, local_rotation_xyz_rad),
            );
            add3(
                parent_translation_world,
                rotate_yxz(local_point, parent_rotation_yxz_rad),
            )
        }
    }
}

fn build_raycast_surface_from_mesh(
    render_id: u64,
    render_fingerprint: u64,
    mesh: &TerrariumTriangleMeshRender,
    material: &TerrariumPbrMaterialRender,
    transform: TerrariumDynamicMeshTransform,
    hide_on_cutaway: bool,
) -> Option<TerrariumRaycastSurface> {
    if mesh.indices.len() < 3 || mesh.positions.is_empty() {
        return None;
    }
    let world_positions = mesh
        .positions
        .iter()
        .copied()
        .map(|position| dynamic_mesh_world_point(transform, position))
        .collect::<Vec<_>>();
    let mut aabb_min_world = [f32::INFINITY; 3];
    let mut aabb_max_world = [f32::NEG_INFINITY; 3];
    for position in &world_positions {
        for axis in 0..3 {
            aabb_min_world[axis] = aabb_min_world[axis].min(position[axis]);
            aabb_max_world[axis] = aabb_max_world[axis].max(position[axis]);
        }
    }
    Some(TerrariumRaycastSurface {
        render_id,
        render_fingerprint,
        hide_on_cutaway,
        solid: material.shader_flags
            & (TERRARIUM_SHADER_FLAG_PLUME | TERRARIUM_SHADER_FLAG_DEBUG_OVERLAY)
            == 0,
        aabb_min_world,
        aabb_max_world,
        world_positions,
        indices: mesh.indices.clone(),
    })
}

fn push_raycast_surface(
    surfaces: &mut Vec<TerrariumRaycastSurface>,
    render_id: u64,
    render_fingerprint: u64,
    mesh: &TerrariumTriangleMeshRender,
    material: &TerrariumPbrMaterialRender,
    transform: TerrariumDynamicMeshTransform,
    hide_on_cutaway: bool,
) {
    if let Some(surface) = build_raycast_surface_from_mesh(
        render_id,
        render_fingerprint,
        mesh,
        material,
        transform,
        hide_on_cutaway,
    ) {
        surfaces.push(surface);
    }
}

pub(crate) fn build_dynamic_raycast_scene(
    snapshot: &TerrariumDynamicRenderSnapshot,
) -> (
    Vec<TerrariumRaycastSurface>,
    Vec<TerrariumRaycastBvhNode>,
    Option<usize>,
) {
    let mut surfaces = Vec::new();
    for voxel in &snapshot.substrate_voxels {
        push_raycast_surface(
            &mut surfaces,
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
    for microbe in &snapshot.explicit_microbes {
        let body_render_id =
            terrarium_render_child_id(microbe.render_id, TERRARIUM_RENDER_SLOT_MICROBE_BODY);
        push_raycast_surface(
            &mut surfaces,
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
        let packet_render_id =
            terrarium_render_child_id(microbe.render_id, TERRARIUM_RENDER_SLOT_MICROBE_PACKET);
        push_raycast_surface(
            &mut surfaces,
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
        for packet in &microbe.packets {
            push_raycast_surface(
                &mut surfaces,
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
        for site in &microbe.sites {
            push_raycast_surface(
                &mut surfaces,
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
    for water in &snapshot.waters {
        push_raycast_surface(
            &mut surfaces,
            water.render_id,
            water.render_fingerprint,
            &water.mesh,
            &water.material,
            TerrariumDynamicMeshTransform::WorldXyz {
                translation_world: water.translation_world,
                rotation_xyz_rad: [0.0, 0.0, 0.0],
            },
            false,
        );
    }
    for plant in &snapshot.plants {
        if plant.tissues.is_empty() {
            let stem_render_id =
                terrarium_render_child_id(plant.render_id, TERRARIUM_RENDER_SLOT_PLANT_STEM);
            push_raycast_surface(
                &mut surfaces,
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
            let canopy_render_id =
                terrarium_render_child_id(plant.render_id, TERRARIUM_RENDER_SLOT_PLANT_CANOPY);
            push_raycast_surface(
                &mut surfaces,
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
        for tissue in &plant.tissues {
            push_raycast_surface(
                &mut surfaces,
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
    for seed in &snapshot.seeds {
        for part in &seed.parts {
            push_raycast_surface(
                &mut surfaces,
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
    for fruit in &snapshot.fruits {
        for part in &fruit.parts {
            push_raycast_surface(
                &mut surfaces,
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
    for plume in &snapshot.plumes {
        push_raycast_surface(
            &mut surfaces,
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
    for fly in &snapshot.flies {
        for part in &fly.parts {
            push_raycast_surface(
                &mut surfaces,
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
    let mut bvh_nodes = Vec::new();
    let bvh_root = if surfaces.is_empty() {
        None
    } else {
        let indices = (0..surfaces.len()).collect::<Vec<_>>();
        Some(build_raycast_bvh_recursive(
            &surfaces,
            indices,
            &mut bvh_nodes,
        ))
    };
    (surfaces, bvh_nodes, bvh_root)
}

fn ray_aabb_entry_distance(
    origin_world: [f32; 3],
    direction_world: [f32; 3],
    aabb_min_world: [f32; 3],
    aabb_max_world: [f32; 3],
    max_distance: f32,
) -> Option<f32> {
    let mut t_min = 0.0f32;
    let mut t_max = max_distance;
    for axis in 0..3 {
        if direction_world[axis].abs() <= 1.0e-6 {
            if origin_world[axis] < aabb_min_world[axis]
                || origin_world[axis] > aabb_max_world[axis]
            {
                return None;
            }
            continue;
        }
        let inv_dir = 1.0 / direction_world[axis];
        let mut t0 = (aabb_min_world[axis] - origin_world[axis]) * inv_dir;
        let mut t1 = (aabb_max_world[axis] - origin_world[axis]) * inv_dir;
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
        }
        t_min = t_min.max(t0);
        t_max = t_max.min(t1);
        if t_max < t_min {
            return None;
        }
    }
    if t_max >= 0.0 {
        Some(t_min.max(0.0))
    } else {
        None
    }
}

fn ray_intersects_triangle(
    origin_world: [f32; 3],
    direction_world: [f32; 3],
    a: [f32; 3],
    b: [f32; 3],
    c: [f32; 3],
) -> Option<(f32, [f32; 3])> {
    let edge_ab = sub3(b, a);
    let edge_ac = sub3(c, a);
    let p = cross3(direction_world, edge_ac);
    let det = dot3(edge_ab, p);
    if det.abs() <= 1.0e-6 {
        return None;
    }
    let inv_det = 1.0 / det;
    let tvec = sub3(origin_world, a);
    let u = dot3(tvec, p) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = cross3(tvec, edge_ab);
    let v = dot3(direction_world, q) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let distance = dot3(edge_ac, q) * inv_det;
    if distance <= 1.0e-4 {
        return None;
    }
    let mut normal_world = normalize3(cross3(edge_ab, edge_ac));
    if dot3(normal_world, direction_world) > 0.0 {
        normal_world = mul3(normal_world, -1.0);
    }
    Some((distance, normal_world))
}

fn aabb_centroid(aabb_min_world: [f32; 3], aabb_max_world: [f32; 3]) -> [f32; 3] {
    [
        (aabb_min_world[0] + aabb_max_world[0]) * 0.5,
        (aabb_min_world[1] + aabb_max_world[1]) * 0.5,
        (aabb_min_world[2] + aabb_max_world[2]) * 0.5,
    ]
}

fn bvh_node_matches_filter(
    node: &TerrariumRaycastBvhNode,
    cutaway: bool,
    solid_only: bool,
) -> bool {
    if solid_only {
        if cutaway {
            node.has_cutaway_visible_solid_surface
        } else {
            node.has_solid_surface
        }
    } else if cutaway {
        node.has_cutaway_visible_surface
    } else {
        node.has_any_surface
    }
}

pub(crate) fn raycast_scene_internal(
    surfaces: &[TerrariumRaycastSurface],
    bvh_nodes: &[TerrariumRaycastBvhNode],
    bvh_root: Option<usize>,
    origin_world: [f32; 3],
    direction_world: [f32; 3],
    cutaway: bool,
    solid_only: bool,
    excluded_class_mask: u16,
) -> Option<TerrariumSceneRaycastHit> {
    let direction_world = normalize3(direction_world);
    let mut best_hit: Option<TerrariumSceneRaycastHit> = None;
    let Some(root_idx) = bvh_root else {
        return None;
    };
    let mut node_stack = vec![root_idx];
    while let Some(node_idx) = node_stack.pop() {
        let node = &bvh_nodes[node_idx];
        if !bvh_node_matches_filter(node, cutaway, solid_only) {
            continue;
        }
        let max_distance = best_hit.map(|hit| hit.distance).unwrap_or(f32::INFINITY);
        let Some(_node_entry) = ray_aabb_entry_distance(
            origin_world,
            direction_world,
            node.aabb_min_world,
            node.aabb_max_world,
            max_distance,
        ) else {
            continue;
        };
        if let (Some(left_child), Some(right_child)) = (node.left_child, node.right_child) {
            let left = &bvh_nodes[left_child];
            let right = &bvh_nodes[right_child];
            let left_entry = if bvh_node_matches_filter(left, cutaway, solid_only) {
                ray_aabb_entry_distance(
                    origin_world,
                    direction_world,
                    left.aabb_min_world,
                    left.aabb_max_world,
                    max_distance,
                )
            } else {
                None
            };
            let right_entry = if bvh_node_matches_filter(right, cutaway, solid_only) {
                ray_aabb_entry_distance(
                    origin_world,
                    direction_world,
                    right.aabb_min_world,
                    right.aabb_max_world,
                    max_distance,
                )
            } else {
                None
            };
            match (left_entry, right_entry) {
                (Some(left_t), Some(right_t)) => {
                    if left_t <= right_t {
                        node_stack.push(right_child);
                        node_stack.push(left_child);
                    } else {
                        node_stack.push(left_child);
                        node_stack.push(right_child);
                    }
                }
                (Some(_), None) => node_stack.push(left_child),
                (None, Some(_)) => node_stack.push(right_child),
                (None, None) => {}
            }
            continue;
        }

        for &surface_idx in &node.surface_indices {
            let surface = &surfaces[surface_idx];
            if excluded_class_mask & (1u16 << terrarium_render_id_class(surface.render_id)) != 0 {
                continue;
            }
            if cutaway && surface.hide_on_cutaway {
                continue;
            }
            if solid_only && !surface.solid {
                continue;
            }
            let max_distance = best_hit.map(|hit| hit.distance).unwrap_or(f32::INFINITY);
            let Some(_surface_entry) = ray_aabb_entry_distance(
                origin_world,
                direction_world,
                surface.aabb_min_world,
                surface.aabb_max_world,
                max_distance,
            ) else {
                continue;
            };
            for triangle in surface.indices.chunks_exact(3) {
                let a = surface.world_positions[triangle[0] as usize];
                let b = surface.world_positions[triangle[1] as usize];
                let c = surface.world_positions[triangle[2] as usize];
                let Some((distance, normal_world)) =
                    ray_intersects_triangle(origin_world, direction_world, a, b, c)
                else {
                    continue;
                };
                if distance >= max_distance {
                    continue;
                }
                best_hit = Some(TerrariumSceneRaycastHit {
                    render_id: surface.render_id,
                    render_fingerprint: surface.render_fingerprint,
                    distance,
                    position_world: add3(origin_world, mul3(direction_world, distance)),
                    normal_world,
                });
            }
        }
    }
    best_hit
}

fn build_raycast_bvh_recursive(
    surfaces: &[TerrariumRaycastSurface],
    mut surface_indices: Vec<usize>,
    nodes: &mut Vec<TerrariumRaycastBvhNode>,
) -> usize {
    let mut aabb_min_world = [f32::INFINITY; 3];
    let mut aabb_max_world = [f32::NEG_INFINITY; 3];
    let mut centroid_min = [f32::INFINITY; 3];
    let mut centroid_max = [f32::NEG_INFINITY; 3];
    let mut has_cutaway_visible_surface = false;
    let mut has_solid_surface = false;
    let mut has_cutaway_visible_solid_surface = false;
    for &surface_idx in &surface_indices {
        let surface = &surfaces[surface_idx];
        for axis in 0..3 {
            aabb_min_world[axis] = aabb_min_world[axis].min(surface.aabb_min_world[axis]);
            aabb_max_world[axis] = aabb_max_world[axis].max(surface.aabb_max_world[axis]);
        }
        let centroid = aabb_centroid(surface.aabb_min_world, surface.aabb_max_world);
        for axis in 0..3 {
            centroid_min[axis] = centroid_min[axis].min(centroid[axis]);
            centroid_max[axis] = centroid_max[axis].max(centroid[axis]);
        }
        has_cutaway_visible_surface |= !surface.hide_on_cutaway;
        has_solid_surface |= surface.solid;
        has_cutaway_visible_solid_surface |= surface.solid && !surface.hide_on_cutaway;
    }

    let node_idx = nodes.len();
    nodes.push(TerrariumRaycastBvhNode {
        aabb_min_world,
        aabb_max_world,
        left_child: None,
        right_child: None,
        surface_indices: Vec::new(),
        has_any_surface: !surface_indices.is_empty(),
        has_cutaway_visible_surface,
        has_solid_surface,
        has_cutaway_visible_solid_surface,
    });

    if surface_indices.len() <= RAYCAST_BVH_LEAF_SURFACES {
        nodes[node_idx].surface_indices = surface_indices;
        return node_idx;
    }

    let centroid_extent = [
        centroid_max[0] - centroid_min[0],
        centroid_max[1] - centroid_min[1],
        centroid_max[2] - centroid_min[2],
    ];
    let split_axis =
        if centroid_extent[1] > centroid_extent[0] && centroid_extent[1] >= centroid_extent[2] {
            1
        } else if centroid_extent[2] > centroid_extent[0] {
            2
        } else {
            0
        };
    surface_indices.sort_by(|lhs, rhs| {
        let lhs_centroid =
            aabb_centroid(surfaces[*lhs].aabb_min_world, surfaces[*lhs].aabb_max_world);
        let rhs_centroid =
            aabb_centroid(surfaces[*rhs].aabb_min_world, surfaces[*rhs].aabb_max_world);
        lhs_centroid[split_axis].total_cmp(&rhs_centroid[split_axis])
    });
    let mid = surface_indices.len() / 2;
    if mid == 0 || mid >= surface_indices.len() {
        nodes[node_idx].surface_indices = surface_indices;
        return node_idx;
    }
    let right_indices = surface_indices.split_off(mid);
    let left_indices = surface_indices;
    let left_child = build_raycast_bvh_recursive(surfaces, left_indices, nodes);
    let right_child = build_raycast_bvh_recursive(surfaces, right_indices, nodes);
    nodes[node_idx].left_child = Some(left_child);
    nodes[node_idx].right_child = Some(right_child);
    node_idx
}
