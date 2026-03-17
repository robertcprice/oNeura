//! Terrain heightfield mesh built from world moisture data + procedural noise elevation.

use oneuro_metal::TerrariumWorld;
use super::math::*;
use super::color::soil_color_v3;
use super::mesh::{Vertex, Triangle, EntityTag};
use super::{CELL_SIZE, HEIGHT_SCALE};

/// Multi-octave value noise for procedural terrain elevation.
fn noise_hash(x: i32, y: i32) -> f32 {
    let n = x.wrapping_mul(374761393)
        .wrapping_add(y.wrapping_mul(668265263));
    let n = (n ^ (n >> 13)).wrapping_mul(1274126177);
    (n & 0x7FFF) as f32 / 32767.0
}

fn smooth_noise(x: f32, y: f32) -> f32 {
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let fx = x - ix as f32;
    let fy = y - iy as f32;
    let sx = fx * fx * (3.0 - 2.0 * fx);
    let sy = fy * fy * (3.0 - 2.0 * fy);
    let n00 = noise_hash(ix, iy);
    let n10 = noise_hash(ix + 1, iy);
    let n01 = noise_hash(ix, iy + 1);
    let n11 = noise_hash(ix + 1, iy + 1);
    let nx0 = n00 + (n10 - n00) * sx;
    let nx1 = n01 + (n11 - n01) * sx;
    nx0 + (nx1 - nx0) * sy
}

/// 4-octave fractal noise for natural-looking terrain.
fn fbm_noise(x: f32, y: f32, seed: u64) -> f32 {
    let offset = (seed % 10000) as f32;
    let mut value = 0.0f32;
    let mut amplitude = 1.0f32;
    let mut frequency = 1.0f32;
    let mut total_amp = 0.0f32;
    for _ in 0..4 {
        value += smooth_noise(x * frequency + offset, y * frequency + offset * 0.7) * amplitude;
        total_amp += amplitude;
        amplitude *= 0.45;
        frequency *= 2.2;
    }
    value / total_amp
}

/// Compute terrain height at a grid cell, combining moisture + procedural noise.
pub fn terrain_height(gx: usize, gy: usize, gw: usize, gh: usize, moisture: &[f32], seed: u64) -> f32 {
    let mi = gy.min(gh - 1) * gw + gx.min(gw - 1);
    let m = if mi < moisture.len() { moisture[mi] } else { 0.3 };
    let nx = gx as f32 / gw as f32;
    let ny = gy as f32 / gh as f32;
    let hill = fbm_noise(nx * 3.5, ny * 3.5, seed) * 0.55;
    let ridge = fbm_noise(nx * 7.0 + 100.0, ny * 7.0 + 100.0, seed.wrapping_add(42)) * 0.2;
    let detail = fbm_noise(nx * 15.0 + 200.0, ny * 15.0 + 200.0, seed.wrapping_add(99)) * 0.08;
    let moisture_depression = (1.0 - m) * 0.15;
    let elevation = hill + ridge + detail + moisture_depression;
    elevation * HEIGHT_SCALE
}

pub fn build_terrain_mesh(world: &TerrariumWorld) -> Vec<Triangle> {
    let gw = world.config.width;
    let gh = world.config.height;
    let moisture = world.moisture_field();
    let seed = world.config.width as u64 * 31 + world.config.height as u64;
    let n_verts = gw * gh;
    let mut positions = Vec::with_capacity(n_verts);
    let mut colors = Vec::with_capacity(n_verts);
    for gy in 0..gh {
        for gx in 0..gw {
            let mi = gy * gw + gx;
            let m = if mi < moisture.len() { moisture[mi] } else { 0.3 };
            let y = terrain_height(gx, gy, gw, gh, &moisture, seed);
            positions.push(v3(gx as f32 * CELL_SIZE, y, gy as f32 * CELL_SIZE));
            let height_factor = (y / HEIGHT_SCALE).clamp(0.0, 1.0);
            colors.push(soil_color_v3(m, height_factor * 0.6 + m * 0.3));
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
