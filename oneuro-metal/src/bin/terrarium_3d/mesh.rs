//! Triangle/Vertex primitives, entity tags, and basic mesh builders.

use super::math::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EntityTag {
    None,
    Terrain,
    Plant(usize),
    Fly(usize),
    Water(usize),
    Fruit(usize),
    Atom(usize),
    Bond(usize),
    Metabolite(usize),
}

impl Default for EntityTag {
    fn default() -> Self { EntityTag::None }
}

#[derive(Clone)]
pub struct Vertex {
    pub pos: V3,
    pub normal: V3,
    pub color: V3,
    pub shininess: f32,
}

pub struct Triangle {
    pub v: [Vertex; 3],
    pub tag: EntityTag,
}

/// Tag all triangles in a vec with the given entity tag.
pub fn tag_all(tris: &mut [Triangle], tag: EntityTag) {
    for t in tris.iter_mut() { t.tag = tag; }
}

pub fn make_diamond(center: V3, sx: f32, sy: f32, sz: f32, color: V3, shininess: f32) -> Vec<Triangle> {
    let top    = add3(center, v3(0.0, sy, 0.0));
    let bottom = add3(center, v3(0.0, -sy, 0.0));
    let front  = add3(center, v3(0.0, 0.0, sz));
    let back   = add3(center, v3(0.0, 0.0, -sz));
    let left   = add3(center, v3(-sx, 0.0, 0.0));
    let right_ = add3(center, v3(sx, 0.0, 0.0));
    let verts = [top, bottom, front, back, left, right_];
    let faces = [
        (0,5,2),(0,2,4),(0,4,3),(0,3,5),
        (1,2,5),(1,4,2),(1,3,4),(1,5,3),
    ];
    faces.iter().map(|&(a,b,c)| {
        let n = normalize3(cross3(sub3(verts[b], verts[a]), sub3(verts[c], verts[a])));
        Triangle { v: [
            Vertex { pos: verts[a], normal: n, color, shininess },
            Vertex { pos: verts[b], normal: n, color, shininess },
            Vertex { pos: verts[c], normal: n, color, shininess },
        ], tag: EntityTag::None }
    }).collect()
}

pub fn make_billboard_quad(base: V3, width: f32, height: f32, color: V3, shininess: f32) -> Vec<Triangle> {
    let hw = width * 0.5;
    let p0 = add3(base, v3(-hw, 0.0, 0.0));
    let p1 = add3(base, v3(hw, 0.0, 0.0));
    let p2 = add3(base, v3(hw, height, 0.0));
    let p3 = add3(base, v3(-hw, height, 0.0));
    let n = [0.0, 0.0, 1.0];
    vec![
        Triangle { v: [
            Vertex { pos: p0, normal: n, color, shininess },
            Vertex { pos: p1, normal: n, color, shininess },
            Vertex { pos: p2, normal: n, color, shininess },
        ], tag: EntityTag::None },
        Triangle { v: [
            Vertex { pos: p0, normal: n, color, shininess },
            Vertex { pos: p2, normal: n, color, shininess },
            Vertex { pos: p3, normal: n, color, shininess },
        ], tag: EntityTag::None },
    ]
}

pub fn make_flat_quad(center: V3, half_size: f32, color: V3, shininess: f32) -> Vec<Triangle> {
    let p0 = add3(center, v3(-half_size, 0.0, -half_size));
    let p1 = add3(center, v3(half_size, 0.0, -half_size));
    let p2 = add3(center, v3(half_size, 0.0, half_size));
    let p3 = add3(center, v3(-half_size, 0.0, half_size));
    let n = [0.0, 1.0, 0.0];
    vec![
        Triangle { v: [
            Vertex { pos: p0, normal: n, color, shininess },
            Vertex { pos: p1, normal: n, color, shininess },
            Vertex { pos: p2, normal: n, color, shininess },
        ], tag: EntityTag::None },
        Triangle { v: [
            Vertex { pos: p0, normal: n, color, shininess },
            Vertex { pos: p2, normal: n, color, shininess },
            Vertex { pos: p3, normal: n, color, shininess },
        ], tag: EntityTag::None },
    ]
}

/// Low-poly sphere (icosphere subdivision 1) for atom rendering.
/// Returns ~80 triangles centered at pos with given radius and color.
pub fn make_sphere(center: V3, radius: f32, color: V3, shininess: f32) -> Vec<Triangle> {
    // Icosahedron base vertices
    let t = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let base_verts: Vec<V3> = vec![
        normalize3([-1.0, t, 0.0]), normalize3([1.0, t, 0.0]),
        normalize3([-1.0, -t, 0.0]), normalize3([1.0, -t, 0.0]),
        normalize3([0.0, -1.0, t]), normalize3([0.0, 1.0, t]),
        normalize3([0.0, -1.0, -t]), normalize3([0.0, 1.0, -t]),
        normalize3([t, 0.0, -1.0]), normalize3([t, 0.0, 1.0]),
        normalize3([-t, 0.0, -1.0]), normalize3([-t, 0.0, 1.0]),
    ];
    let ico_faces: [(usize,usize,usize); 20] = [
        (0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),
        (1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),
        (3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),
        (4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1),
    ];
    // Subdivide once for smoother sphere
    let mut tris = Vec::with_capacity(80);
    for &(a, b, c) in &ico_faces {
        let va = base_verts[a];
        let vb = base_verts[b];
        let vc = base_verts[c];
        let vab = normalize3(lerp3(va, vb, 0.5));
        let vbc = normalize3(lerp3(vb, vc, 0.5));
        let vca = normalize3(lerp3(vc, va, 0.5));
        for &(p0, p1, p2) in &[(va, vab, vca), (vab, vb, vbc), (vca, vbc, vc), (vab, vbc, vca)] {
            let v0 = add3(center, scale3(p0, radius));
            let v1 = add3(center, scale3(p1, radius));
            let v2 = add3(center, scale3(p2, radius));
            tris.push(Triangle {
                v: [
                    Vertex { pos: v0, normal: p0, color, shininess },
                    Vertex { pos: v1, normal: p1, color, shininess },
                    Vertex { pos: v2, normal: p2, color, shininess },
                ],
                tag: EntityTag::None,
            });
        }
    }
    tris
}

/// Cylinder (bond) between two points. Low-poly (6-sided prism).
pub fn make_cylinder(p0: V3, p1: V3, radius: f32, color: V3, shininess: f32) -> Vec<Triangle> {
    let axis = sub3(p1, p0);
    let len = len3(axis);
    if len < 1e-6 { return vec![]; }
    let dir = scale3(axis, 1.0 / len);
    // Find perpendicular vectors
    let up = if dir[1].abs() < 0.9 { [0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };
    let right = normalize3(cross3(dir, up));
    let fwd = normalize3(cross3(right, dir));
    let n_sides = 6;
    let mut bottom_ring = Vec::with_capacity(n_sides);
    let mut top_ring = Vec::with_capacity(n_sides);
    for i in 0..n_sides {
        let angle = (i as f32 / n_sides as f32) * std::f32::consts::TAU;
        let offset = add3(scale3(right, angle.cos() * radius), scale3(fwd, angle.sin() * radius));
        bottom_ring.push(add3(p0, offset));
        top_ring.push(add3(p1, offset));
    }
    let mut tris = Vec::with_capacity(n_sides * 2);
    for i in 0..n_sides {
        let j = (i + 1) % n_sides;
        let n = normalize3(cross3(sub3(top_ring[i], bottom_ring[i]), sub3(bottom_ring[j], bottom_ring[i])));
        tris.push(Triangle { v: [
            Vertex { pos: bottom_ring[i], normal: n, color, shininess },
            Vertex { pos: top_ring[i], normal: n, color, shininess },
            Vertex { pos: bottom_ring[j], normal: n, color, shininess },
        ], tag: EntityTag::None });
        tris.push(Triangle { v: [
            Vertex { pos: bottom_ring[j], normal: n, color, shininess },
            Vertex { pos: top_ring[i], normal: n, color, shininess },
            Vertex { pos: top_ring[j], normal: n, color, shininess },
        ], tag: EntityTag::None });
    }
    tris
}
