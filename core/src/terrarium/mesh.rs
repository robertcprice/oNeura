//! Pure vec3 math helpers, mesh construction primitives, and raycast delegation
//! wrappers extracted from `terrarium_world.rs`.  Every function here is
//! stateless — none touches `TerrariumWorld`.
#![allow(dead_code)]

use crate::terrarium_render::TerrariumDynamicRenderSnapshot;
use crate::terrarium_render::TerrariumTriangleMeshRender;
use crate::terrarium_scene_query::{
    TerrariumRaycastBvhNode, TerrariumRaycastSurface, TerrariumSceneRaycastHit,
};

// ---------------------------------------------------------------------------
// Vec3 math
// ---------------------------------------------------------------------------

pub(in crate::terrarium) fn normalize3(value: [f32; 3]) -> [f32; 3] {
    let length = (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt();
    if length <= 1.0e-6 {
        [0.0, 1.0, 0.0]
    } else {
        [value[0] / length, value[1] / length, value[2] / length]
    }
}

pub(crate) fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub(in crate::terrarium) fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(in crate::terrarium) fn mul3(value: [f32; 3], scalar: f32) -> [f32; 3] {
    [value[0] * scalar, value[1] * scalar, value[2] * scalar]
}

pub(in crate::terrarium) fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(in crate::terrarium) fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub(in crate::terrarium) fn rotate_about_axis(
    value: [f32; 3],
    axis: [f32; 3],
    angle: f32,
) -> [f32; 3] {
    let axis = normalize3(axis);
    let cos_angle = angle.cos();
    let sin_angle = angle.sin();
    let term_parallel = mul3(axis, dot3(axis, value) * (1.0 - cos_angle));
    let term_cross = mul3(cross3(axis, value), sin_angle);
    add3(add3(mul3(value, cos_angle), term_cross), term_parallel)
}

pub(crate) fn rotate_xyz(value: [f32; 3], rotation_xyz_rad: [f32; 3]) -> [f32; 3] {
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

pub(crate) fn rotate_yxz(value: [f32; 3], rotation_yxz_rad: [f32; 3]) -> [f32; 3] {
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

// ---------------------------------------------------------------------------
// Raycast delegation wrappers
// ---------------------------------------------------------------------------

pub(crate) fn build_dynamic_raycast_scene(
    snapshot: &TerrariumDynamicRenderSnapshot,
) -> (
    Vec<TerrariumRaycastSurface>,
    Vec<TerrariumRaycastBvhNode>,
    Option<usize>,
) {
    crate::terrarium_scene_query::build_dynamic_raycast_scene(snapshot)
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
    crate::terrarium_scene_query::raycast_scene_internal(
        surfaces,
        bvh_nodes,
        bvh_root,
        origin_world,
        direction_world,
        cutaway,
        solid_only,
        excluded_class_mask,
    )
}

// ---------------------------------------------------------------------------
// Mesh construction helpers
// ---------------------------------------------------------------------------

pub(crate) fn mesh_append(
    dst: &mut TerrariumTriangleMeshRender,
    src: &TerrariumTriangleMeshRender,
) {
    let base = dst.positions.len() as u32;
    dst.positions.extend(src.positions.iter().copied());
    dst.normals.extend(src.normals.iter().copied());
    dst.uvs.extend(src.uvs.iter().copied());
    dst.indices.extend(src.indices.iter().map(|idx| idx + base));
}

pub(crate) fn mesh_translate(mesh: &mut TerrariumTriangleMeshRender, translation: [f32; 3]) {
    for position in &mut mesh.positions {
        *position = add3(*position, translation);
    }
}

pub(crate) fn mesh_rotate_xyz(mesh: &mut TerrariumTriangleMeshRender, rotation_xyz_rad: [f32; 3]) {
    for position in &mut mesh.positions {
        *position = rotate_xyz(*position, rotation_xyz_rad);
    }
    for normal in &mut mesh.normals {
        *normal = normalize3(rotate_xyz(*normal, rotation_xyz_rad));
    }
}

pub(crate) fn mesh_rotate_yxz(mesh: &mut TerrariumTriangleMeshRender, rotation_yxz_rad: [f32; 3]) {
    for position in &mut mesh.positions {
        *position = rotate_yxz(*position, rotation_yxz_rad);
    }
    for normal in &mut mesh.normals {
        *normal = normalize3(rotate_yxz(*normal, rotation_yxz_rad));
    }
}

pub(in crate::terrarium) fn mesh_orient_from_up(
    mesh: &mut TerrariumTriangleMeshRender,
    direction: [f32; 3],
) {
    let up = [0.0, 1.0, 0.0];
    let direction = normalize3(direction);
    let dot = dot3(up, direction).clamp(-1.0, 1.0);
    if dot > 0.9999 {
        return;
    }
    if dot < -0.9999 {
        mesh_rotate_xyz(mesh, [std::f32::consts::PI, 0.0, 0.0]);
        return;
    }
    let axis = cross3(up, direction);
    let angle = dot.acos();
    for position in &mut mesh.positions {
        *position = rotate_about_axis(*position, axis, angle);
    }
    for normal in &mut mesh.normals {
        *normal = normalize3(rotate_about_axis(*normal, axis, angle));
    }
}

pub(in crate::terrarium) fn bezier3(
    a: [f32; 3],
    b: [f32; 3],
    c: [f32; 3],
    d: [f32; 3],
    t: f32,
) -> [f32; 3] {
    let ab = add3(mul3(a, 1.0 - t), mul3(b, t));
    let bc = add3(mul3(b, 1.0 - t), mul3(c, t));
    let cd = add3(mul3(c, 1.0 - t), mul3(d, t));
    let abc = add3(mul3(ab, 1.0 - t), mul3(bc, t));
    let bcd = add3(mul3(bc, 1.0 - t), mul3(cd, t));
    add3(mul3(abc, 1.0 - t), mul3(bcd, t))
}

pub(in crate::terrarium) fn mesh_push_quad(
    mesh: &mut TerrariumTriangleMeshRender,
    positions: [[f32; 3]; 4],
    normal: [f32; 3],
) {
    let base = mesh.positions.len() as u32;
    mesh.positions.extend(positions);
    mesh.normals.extend([normal; 4]);
    mesh.uvs
        .extend([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
    mesh.indices
        .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
}

pub(in crate::terrarium) fn render_cuboid_mesh(size: [f32; 3]) -> TerrariumTriangleMeshRender {
    let [hx, hy, hz] = [size[0] * 0.5, size[1] * 0.5, size[2] * 0.5];
    let mut mesh = TerrariumTriangleMeshRender::default();
    mesh_push_quad(
        &mut mesh,
        [[-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]],
        [0.0, 0.0, 1.0],
    );
    mesh_push_quad(
        &mut mesh,
        [
            [hx, -hy, -hz],
            [-hx, -hy, -hz],
            [-hx, hy, -hz],
            [hx, hy, -hz],
        ],
        [0.0, 0.0, -1.0],
    );
    mesh_push_quad(
        &mut mesh,
        [[hx, -hy, hz], [hx, -hy, -hz], [hx, hy, -hz], [hx, hy, hz]],
        [1.0, 0.0, 0.0],
    );
    mesh_push_quad(
        &mut mesh,
        [
            [-hx, -hy, -hz],
            [-hx, -hy, hz],
            [-hx, hy, hz],
            [-hx, hy, -hz],
        ],
        [-1.0, 0.0, 0.0],
    );
    mesh_push_quad(
        &mut mesh,
        [[-hx, hy, hz], [hx, hy, hz], [hx, hy, -hz], [-hx, hy, -hz]],
        [0.0, 1.0, 0.0],
    );
    mesh_push_quad(
        &mut mesh,
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, -hy, hz],
            [-hx, -hy, hz],
        ],
        [0.0, -1.0, 0.0],
    );
    mesh
}

#[allow(dead_code)] // Used by terrarium_viewer binary
pub(in crate::terrarium) fn render_quad_mesh(size: [f32; 2]) -> TerrariumTriangleMeshRender {
    let [hx, hy] = [size[0] * 0.5, size[1] * 0.5];
    let mut mesh = TerrariumTriangleMeshRender::default();
    mesh_push_quad(
        &mut mesh,
        [
            [-hx, -hy, 0.0],
            [hx, -hy, 0.0],
            [hx, hy, 0.0],
            [-hx, hy, 0.0],
        ],
        [0.0, 0.0, 1.0],
    );
    mesh
}

pub(in crate::terrarium) fn render_cylinder_mesh(
    radius: f32,
    height: f32,
    resolution: usize,
) -> TerrariumTriangleMeshRender {
    let resolution = resolution.max(3);
    let half_height = height * 0.5;
    let mut mesh = TerrariumTriangleMeshRender::default();

    for segment in 0..=resolution {
        let t = segment as f32 / resolution as f32;
        let theta = t * std::f32::consts::TAU;
        let (sin_theta, cos_theta) = theta.sin_cos();
        let normal = [cos_theta, 0.0, sin_theta];
        let x = cos_theta * radius;
        let z = sin_theta * radius;
        mesh.positions.push([x, -half_height, z]);
        mesh.normals.push(normal);
        mesh.uvs.push([t, 0.0]);
        mesh.positions.push([x, half_height, z]);
        mesh.normals.push(normal);
        mesh.uvs.push([t, 1.0]);
    }
    for segment in 0..resolution {
        let base = (segment * 2) as u32;
        mesh.indices
            .extend([base, base + 1, base + 3, base, base + 3, base + 2]);
    }

    let top_center = mesh.positions.len() as u32;
    mesh.positions.push([0.0, half_height, 0.0]);
    mesh.normals.push([0.0, 1.0, 0.0]);
    mesh.uvs.push([0.5, 0.5]);
    for segment in 0..=resolution {
        let t = segment as f32 / resolution as f32;
        let theta = t * std::f32::consts::TAU;
        let (sin_theta, cos_theta) = theta.sin_cos();
        mesh.positions
            .push([cos_theta * radius, half_height, sin_theta * radius]);
        mesh.normals.push([0.0, 1.0, 0.0]);
        mesh.uvs
            .push([cos_theta * 0.5 + 0.5, sin_theta * 0.5 + 0.5]);
    }
    for segment in 0..resolution {
        let ring = top_center + 1 + segment as u32;
        mesh.indices.extend([top_center, ring, ring + 1]);
    }

    let bottom_center = mesh.positions.len() as u32;
    mesh.positions.push([0.0, -half_height, 0.0]);
    mesh.normals.push([0.0, -1.0, 0.0]);
    mesh.uvs.push([0.5, 0.5]);
    for segment in 0..=resolution {
        let t = segment as f32 / resolution as f32;
        let theta = t * std::f32::consts::TAU;
        let (sin_theta, cos_theta) = theta.sin_cos();
        mesh.positions
            .push([cos_theta * radius, -half_height, sin_theta * radius]);
        mesh.normals.push([0.0, -1.0, 0.0]);
        mesh.uvs
            .push([cos_theta * 0.5 + 0.5, sin_theta * 0.5 + 0.5]);
    }
    for segment in 0..resolution {
        let ring = bottom_center + 1 + segment as u32;
        mesh.indices.extend([bottom_center, ring + 1, ring]);
    }

    mesh
}

pub(in crate::terrarium) fn render_ellipsoid_mesh(
    radii: [f32; 3],
    sectors: usize,
    stacks: usize,
) -> TerrariumTriangleMeshRender {
    let sectors = sectors.max(3);
    let stacks = stacks.max(2);
    let mut mesh = TerrariumTriangleMeshRender::default();
    let radii = [
        radii[0].max(1.0e-4),
        radii[1].max(1.0e-4),
        radii[2].max(1.0e-4),
    ];

    for stack in 0..=stacks {
        let v = stack as f32 / stacks as f32;
        let phi = std::f32::consts::PI * v;
        let ring = phi.sin();
        let y_unit = phi.cos();
        for sector in 0..=sectors {
            let u = sector as f32 / sectors as f32;
            let theta = u * std::f32::consts::TAU;
            let (sin_theta, cos_theta) = theta.sin_cos();
            let x_unit = ring * cos_theta;
            let z_unit = ring * sin_theta;
            mesh.positions
                .push([x_unit * radii[0], y_unit * radii[1], z_unit * radii[2]]);
            mesh.normals.push(normalize3([
                x_unit / radii[0],
                y_unit / radii[1],
                z_unit / radii[2],
            ]));
            mesh.uvs.push([u, 1.0 - v]);
        }
    }

    let ring_vertices = sectors + 1;
    for stack in 0..stacks {
        for sector in 0..sectors {
            let base = (stack * ring_vertices + sector) as u32;
            let next = base + ring_vertices as u32;
            mesh.indices
                .extend([base, next, next + 1, base, next + 1, base + 1]);
        }
    }

    mesh
}

pub(in crate::terrarium) fn render_segment_mesh(
    start: [f32; 3],
    end: [f32; 3],
    radius: f32,
    resolution: usize,
) -> TerrariumTriangleMeshRender {
    let delta = sub3(end, start);
    let length = dot3(delta, delta).sqrt().max(1.0e-4);
    let mut mesh = render_cylinder_mesh(radius, length, resolution);
    mesh_orient_from_up(&mut mesh, delta);
    mesh_translate(&mut mesh, mul3(add3(start, end), 0.5));
    mesh
}
