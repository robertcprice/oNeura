//! Fly entity mesh generation — small animated diamond with entity tags.

use oneuro_metal::drosophila::DrosophilaSim;
use super::math::v3;
use super::mesh::{Triangle, EntityTag, make_diamond, tag_all};
use super::terrain::terrain_height;
use super::CELL_SIZE;

pub fn build_fly_meshes(flies: &[DrosophilaSim], gw: usize, gh: usize, moisture: &[f32], seed: u64, frame: usize) -> Vec<Triangle> {
    let mut tris = Vec::with_capacity(flies.len() * 8);
    for (i, fly) in flies.iter().enumerate() {
        let b = fly.body_state();
        let gx = b.x.round().clamp(0.0, (gw - 1) as f32) as usize;
        let gy = b.y.round().clamp(0.0, (gh - 1) as f32) as usize;
        let base_y = terrain_height(gx, gy, gw, gh, moisture, seed);
        let wx = b.x.clamp(0.0, (gw - 1) as f32) * CELL_SIZE;
        let wz = b.y.clamp(0.0, (gh - 1) as f32) * CELL_SIZE;
        let alt = if b.is_flying { b.z.clamp(0.0, 4.0) * 0.4 } else { 0.05 };
        let color = if b.is_flying {
            if frame % 6 < 3 { [0.95, 0.92, 0.20] } else { [0.90, 0.85, 0.15] }
        } else { [0.75, 0.65, 0.15] };
        let mut fly_tris = make_diamond(v3(wx, base_y + alt + 0.1, wz), 0.06, 0.04, 0.08, color, 16.0);
        tag_all(&mut fly_tris, EntityTag::Fly(i));
        tris.extend(fly_tris);
    }
    tris
}
