//! Plant entity mesh generation — trunk billboard + canopy diamond with entity tags.

use oneuro_metal::TerrariumPlant;
use super::math::v3;
use super::color::plant_color_v3;
use super::mesh::{Triangle, EntityTag, make_billboard_quad, make_diamond, tag_all};
use super::terrain::terrain_height;
use super::CELL_SIZE;

pub fn build_plant_meshes(plants: &[TerrariumPlant], gw: usize, gh: usize, moisture: &[f32], seed: u64) -> Vec<Triangle> {
    let mut tris = Vec::with_capacity(plants.len() * 10);
    for (i, plant) in plants.iter().enumerate() {
        let gx = plant.x.min(gw - 1);
        let gy = plant.y.min(gh - 1);
        let base_y = terrain_height(gx, gy, gw, gh, moisture, seed);
        let wx = gx as f32 * CELL_SIZE;
        let wz = gy as f32 * CELL_SIZE;
        let plant_h = (plant.physiology.height_mm() * 0.15).clamp(0.3, 2.5);
        let canopy_r = (plant.canopy_radius_cells() as f32 * 0.12).clamp(0.15, 0.8);
        let vit = plant.cellular.vitality();
        let canopy_density = (plant.cellular.total_cells() * 0.01).clamp(0.0, 1.0);
        let mut plant_tris = make_billboard_quad(v3(wx, base_y, wz), 0.06, plant_h, [0.47, 0.31, 0.16], 4.0);
        plant_tris.extend(make_diamond(v3(wx, base_y + plant_h, wz), canopy_r, canopy_r * 0.7, canopy_r, plant_color_v3(canopy_density, vit), 8.0));
        tag_all(&mut plant_tris, EntityTag::Plant(i));
        tris.extend(plant_tris);
    }
    tris
}
