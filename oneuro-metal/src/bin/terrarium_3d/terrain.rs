//! Terrain heightfield mesh built from world moisture data with normal perturbation.

use oneuro_metal::TerrariumWorld;
use super::math::*;
use super::color::soil_color_v3;
use super::mesh::{Vertex, Triangle, EntityTag};
use super::{CELL_SIZE, HEIGHT_SCALE};

pub fn build_terrain_mesh(world: &TerrariumWorld) -> Vec<Triangle> {
    let gw = world.config.width;
    let gh = world.config.height;
    let moisture = world.moisture_field();
    let n_verts = gw * gh;
    let mut positions = Vec::with_capacity(n_verts);
    let mut colors = Vec::with_capacity(n_verts);
    for gy in 0..gh {
        for gx in 0..gw {
            let mi = gy * gw + gx;
            let m = if mi < moisture.len() { moisture[mi] } else { 0.3 };
            positions.push(v3(gx as f32 * CELL_SIZE, m * HEIGHT_SCALE, gy as f32 * CELL_SIZE));
            colors.push(soil_color_v3(m, m * 0.5));
        }
    }
    let mut normals = vec![[0.0f32; 3]; n_verts];
    let mut tri_indices = Vec::with_capacity((gw - 1) * (gh - 1) * 2);
    for gy in 0..(gh - 1) {
        for gx in 0..(gw - 1) {
            let i00 = gy * gw + gx;
            let i10 = i00 + 1;
            let i01 = i00 + gw;
            let i11 = i01 + 1;
            for &(a, b, c) in &[(i00, i10, i01), (i10, i11, i01)] {
                let fn_ = normalize3(cross3(sub3(positions[b], positions[a]), sub3(positions[c], positions[a])));
                for &vi in &[a, b, c] { normals[vi] = add3(normals[vi], fn_); }
            }
            tri_indices.push((i00, i10, i01));
            tri_indices.push((i10, i11, i01));
        }
    }
    for n in &mut normals { *n = normalize3(*n); }
    // Normal perturbation: add pseudo-random roughness based on grid position
    for gy in 0..gh {
        for gx in 0..gw {
            let idx = gy * gw + gx;
            normals[idx] = perturb_normal(normals[idx], gx, gy);
        }
    }
    tri_indices.iter().map(|&(a, b, c)| Triangle {
        v: [
            Vertex { pos: positions[a], normal: normals[a], color: colors[a], shininess: 8.0 },
            Vertex { pos: positions[b], normal: normals[b], color: colors[b], shininess: 8.0 },
            Vertex { pos: positions[c], normal: normals[c], color: colors[c], shininess: 8.0 },
        ],
        tag: EntityTag::Terrain,
    }).collect()
}

/// Hash-based normal perturbation for terrain visual roughness.
fn perturb_normal(n: V3, gx: usize, gy: usize) -> V3 {
    let hash = |a: usize, b: usize| -> f32 {
        let h = a.wrapping_mul(374761393) ^ b.wrapping_mul(668265263);
        let h = h.wrapping_add(h >> 13).wrapping_mul(1274126177);
        ((h & 0xFFFF) as f32 / 65535.0) - 0.5
    };
    let px = hash(gx, gy) * 0.18;
    let pz = hash(gy.wrapping_add(97), gx.wrapping_add(31)) * 0.18;
    normalize3([n[0] + px, n[1], n[2] + pz])
}
