//! Compatibility stubs for terrarium render modules.
//!
//! These functions and constants were originally expected to live in `terrarium_world.rs`,
//! but were never merged there. This module provides standalone implementations so that
//! `terrarium_render`, `terrarium_scene_query`, `terrarium_render_pipeline`, and
//! `terrarium_contact` can compile without modifying the actively-maintained terrarium_world.rs.
//!
//! Gated behind `#[cfg(feature = "terrarium_render")]`.

use std::hash::{Hash, Hasher};

use crate::terrarium_render::TerrariumTriangleMeshRender;

// ---------------------------------------------------------------------------
// Render-ID encoding helpers
// ---------------------------------------------------------------------------

/// Render-slot constants for child sub-entities within a composite render object.
pub(crate) const TERRARIUM_RENDER_SLOT_MICROBE_BODY: u8 = 1;
pub(crate) const TERRARIUM_RENDER_SLOT_MICROBE_PACKET: u8 = 2;
pub(crate) const TERRARIUM_RENDER_SLOT_PLANT_STEM: u8 = 3;
pub(crate) const TERRARIUM_RENDER_SLOT_PLANT_CANOPY: u8 = 4;
pub(crate) const TERRARIUM_RENDER_SLOT_FLY_LIGHT: u8 = 5;

/// Shader flag bits for material classification.
pub(crate) const TERRARIUM_SHADER_FLAG_PLUME: u32 = 1 << 0;
pub(crate) const TERRARIUM_SHADER_FLAG_DEBUG_OVERLAY: u32 = 1 << 1;

/// Encode a child render ID from a parent ID and a slot byte.
///
/// Layout: bits [63..8] from parent | bits [7..0] from slot.
pub(crate) fn terrarium_render_child_id(parent_id: u64, slot: u8) -> u64 {
    (parent_id & !0xFF) | (slot as u64)
}

/// Extract the render-class nibble (bits [7..4]) from a render ID.
pub(crate) fn terrarium_render_id_class(render_id: u64) -> u16 {
    ((render_id >> 4) & 0x0F) as u16
}

// ---------------------------------------------------------------------------
// Batching chunk constants
// ---------------------------------------------------------------------------

/// Spatial chunk size in X/Y voxel indices for substrate batching.
pub(crate) const SUBSTRATE_BATCH_CHUNK_XY: usize = 4;
/// Spatial chunk size in Z voxel indices for substrate batching.
pub(crate) const SUBSTRATE_BATCH_CHUNK_Z: usize = 2;

/// World-space chunk size on X/Z axes for dynamic entity batching.
pub(crate) const DYNAMIC_BATCH_CHUNK_WORLD_XZ: f32 = 2.0;
/// World-space chunk size on the Y axis for dynamic entity batching.
pub(crate) const DYNAMIC_BATCH_CHUNK_WORLD_Y: f32 = 1.0;

// ---------------------------------------------------------------------------
// Batch render-ID generators
// ---------------------------------------------------------------------------

/// Compute a deterministic render ID for a substrate batch from chunk coords + material key.
pub(crate) fn terrarium_substrate_batch_render_id(
    chunk_x: usize,
    chunk_y: usize,
    chunk_z: usize,
    material_state_key: u64,
    hide_on_cutaway: bool,
) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    0xA000_0000u64.hash(&mut hasher);
    chunk_x.hash(&mut hasher);
    chunk_y.hash(&mut hasher);
    chunk_z.hash(&mut hasher);
    material_state_key.hash(&mut hasher);
    hide_on_cutaway.hash(&mut hasher);
    hasher.finish()
}

/// Compute a deterministic render ID for a dynamic batch from kind, chunk coords, + material key.
pub(crate) fn terrarium_dynamic_batch_render_id(
    kind: crate::terrarium_render::TerrariumDynamicBatchKind,
    chunk_x: i32,
    chunk_y: i32,
    chunk_z: i32,
    material_state_key: u64,
) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    0xB000_0000u64.hash(&mut hasher);
    (kind as u32).hash(&mut hasher);
    chunk_x.hash(&mut hasher);
    chunk_y.hash(&mut hasher);
    chunk_z.hash(&mut hasher);
    material_state_key.hash(&mut hasher);
    hasher.finish()
}

// ---------------------------------------------------------------------------
// 3-D vector math helpers
// ---------------------------------------------------------------------------

/// Component-wise add.
#[inline]
pub(crate) fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Rotate a point by Euler angles in X-Y-Z order.
pub(crate) fn rotate_xyz(point: [f32; 3], angles: [f32; 3]) -> [f32; 3] {
    let (sx, cx) = angles[0].sin_cos();
    let (sy, cy) = angles[1].sin_cos();
    let (sz, cz) = angles[2].sin_cos();
    // Rx
    let y1 = cx * point[1] - sx * point[2];
    let z1 = sx * point[1] + cx * point[2];
    let x1 = point[0];
    // Ry
    let x2 = cy * x1 + sy * z1;
    let z2 = -sy * x1 + cy * z1;
    // Rz
    let x3 = cz * x2 - sz * y1;
    let y3 = sz * x2 + cz * y1;
    [x3, y3, z2]
}

/// Rotate a point by Euler angles in Y-X-Z order.
pub(crate) fn rotate_yxz(point: [f32; 3], angles: [f32; 3]) -> [f32; 3] {
    let (sy, cy) = angles[0].sin_cos();
    let (sx, cx) = angles[1].sin_cos();
    let (sz, cz) = angles[2].sin_cos();
    // Ry
    let x1 = cy * point[0] + sy * point[2];
    let z1 = -sy * point[0] + cy * point[2];
    let y1 = point[1];
    // Rx
    let y2 = cx * y1 - sx * z1;
    let z2 = sx * y1 + cx * z1;
    // Rz
    let x3 = cz * x1 - sz * y2;
    let y3 = sz * x1 + cz * y2;
    [x3, y3, z2]
}

// ---------------------------------------------------------------------------
// Mesh manipulation helpers
// ---------------------------------------------------------------------------

/// Translate all vertex positions in-place by `offset`.
pub(crate) fn mesh_translate(mesh: &mut TerrariumTriangleMeshRender, offset: [f32; 3]) {
    for p in &mut mesh.positions {
        p[0] += offset[0];
        p[1] += offset[1];
        p[2] += offset[2];
    }
}

/// Rotate all vertex positions and normals in-place using X-Y-Z Euler angles.
pub(crate) fn mesh_rotate_xyz(mesh: &mut TerrariumTriangleMeshRender, angles: [f32; 3]) {
    for p in &mut mesh.positions {
        *p = rotate_xyz(*p, angles);
    }
    for n in &mut mesh.normals {
        *n = rotate_xyz(*n, angles);
    }
}

/// Rotate all vertex positions and normals in-place using Y-X-Z Euler angles.
pub(crate) fn mesh_rotate_yxz(mesh: &mut TerrariumTriangleMeshRender, angles: [f32; 3]) {
    for p in &mut mesh.positions {
        *p = rotate_yxz(*p, angles);
    }
    for n in &mut mesh.normals {
        *n = rotate_yxz(*n, angles);
    }
}

/// Append `other` mesh data onto `target`, adjusting indices.
pub(crate) fn mesh_append(
    target: &mut TerrariumTriangleMeshRender,
    other: &TerrariumTriangleMeshRender,
) {
    let base_index = target.positions.len() as u32;
    target.positions.extend_from_slice(&other.positions);
    target.normals.extend_from_slice(&other.normals);
    target.uvs.extend_from_slice(&other.uvs);
    target.indices.extend(other.indices.iter().map(|i| i + base_index));
}

// ---------------------------------------------------------------------------
// Fly ↔ world coordinate helpers (used by terrarium_contact)
// ---------------------------------------------------------------------------

/// Convert a `BodyState` to a 3D world-space translation + heading.
///
/// The terrain grid is looked up at the fly's (x, y) coordinate to get ground height.
/// Returns (world_translation, heading).
pub(crate) fn fly_translation_world_from_body(
    config: &crate::terrarium_world::TerrariumWorldConfig,
    terrain: &[f32],
    terrain_min: f32,
    terrain_inv: f32,
    body: &crate::drosophila::BodyState,
) -> ([f32; 3], f32) {
    let cell_size = config.cell_size_mm;
    let wx = body.x * cell_size;
    let wz = body.y * cell_size;
    let ix = (body.x as usize).min(config.width.saturating_sub(1));
    let iy = (body.y as usize).min(config.height.saturating_sub(1));
    let terrain_h = terrain
        .get(iy * config.width + ix)
        .copied()
        .unwrap_or(0.0);
    let ground_world = (terrain_h - terrain_min) * terrain_inv;
    let wy = ground_world + body.z;
    ([wx, wy, wz], body.heading)
}

/// Convert a world-space translation back to body (x, y, z, is_flying).
///
/// Returns (x, y, z, is_flying_flag).
pub(crate) fn fly_body_state_from_world_translation(
    config: &crate::terrarium_world::TerrariumWorldConfig,
    terrain: &[f32],
    terrain_min: f32,
    terrain_inv: f32,
    world_pos: [f32; 3],
    is_flying: bool,
) -> (f32, f32, f32, bool) {
    let cell_size = config.cell_size_mm;
    let bx = world_pos[0] / cell_size.max(1.0e-6);
    let by = world_pos[2] / cell_size.max(1.0e-6);
    let ix = (bx as usize).min(config.width.saturating_sub(1));
    let iy = (by as usize).min(config.height.saturating_sub(1));
    let terrain_h = terrain
        .get(iy * config.width + ix)
        .copied()
        .unwrap_or(0.0);
    let ground_world = (terrain_h - terrain_min) * terrain_inv;
    let bz = world_pos[1] - ground_world;
    (bx, by, bz, is_flying)
}
