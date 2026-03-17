//! Water entity mesh generation — animated translucent blue quads with wave displacement.

use oneuro_metal::terrarium_world::TerrariumWorld;
use super::math::v3;
use super::mesh::{Triangle, EntityTag, make_flat_quad, tag_all};
use super::terrain::terrain_height;
use super::CELL_SIZE;

pub fn build_water_meshes(world: &TerrariumWorld, seed: u64, frame_idx: usize) -> Vec<Triangle> {
    let gw = world.config.width;
    let gh = world.config.height;
    let moisture = world.moisture_field();
    let mut tris = Vec::with_capacity(world.waters.len() * 2);
    for (i, water) in world.waters.iter().enumerate() {
        if water.alive {
            let gx = water.x.min(gw - 1);
            let gy = water.y.min(gh - 1);
            let base_y = terrain_height(gx, gy, gw, gh, &moisture, seed);
            let wx = gx as f32 * CELL_SIZE;
            let wz = gy as f32 * CELL_SIZE;
            let wave = (frame_idx as f32 * 0.08 + wx * 2.5 + wz * 1.8).sin() * 0.012
                     + (frame_idx as f32 * 0.12 + wx * 1.3 - wz * 2.1).cos() * 0.008;
            let mut water_tris = make_flat_quad(
                v3(wx, base_y + 0.025 + wave, wz),
                CELL_SIZE * 0.6,
                [0.20, 0.55, 0.85],
                64.0,
            );
            tag_all(&mut water_tris, EntityTag::Water(i));
            tris.extend(water_tris);
        }
    }
    tris
}
