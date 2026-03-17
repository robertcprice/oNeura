//! Fruit entity mesh generation — small ripeness-colored diamonds with entity tags.

use oneuro_metal::TerrariumFruitPatch;
use super::math::{v3, lerp3};
use super::mesh::{Triangle, EntityTag, make_diamond, tag_all};
use super::{CELL_SIZE, HEIGHT_SCALE};

pub fn build_fruit_meshes(fruits: &[TerrariumFruitPatch], gw: usize, gh: usize, moisture: &[f32]) -> Vec<Triangle> {
    let mut tris = Vec::with_capacity(fruits.len() * 8);
    for (i, fruit) in fruits.iter().enumerate() {
        if fruit.source.alive && fruit.source.sugar_content > 0.01 {
            let gx = fruit.source.x.min(gw - 1);
            let gy = fruit.source.y.min(gh - 1);
            let base_y = terrain_y(gx, gy, gw, gh, moisture);
            let wx = gx as f32 * CELL_SIZE;
            let wz = gy as f32 * CELL_SIZE;
            let ripe = fruit.source.ripeness.clamp(0.0, 1.0);
            let color = lerp3([0.39, 0.78, 0.20], [0.92, 0.32, 0.12], ripe);
            let mut fruit_tris = make_diamond(v3(wx, base_y + 0.08, wz), 0.04, 0.04, 0.04, color, 8.0);
            tag_all(&mut fruit_tris, EntityTag::Fruit(i));
            tris.extend(fruit_tris);
        }
    }
    tris
}

fn terrain_y(gx: usize, gy: usize, gw: usize, gh: usize, moisture: &[f32]) -> f32 {
    let mi = gy.min(gh - 1) * gw + gx.min(gw - 1);
    (if mi < moisture.len() { moisture[mi] } else { 0.3 }) * HEIGHT_SCALE
}
