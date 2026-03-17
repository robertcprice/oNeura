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
