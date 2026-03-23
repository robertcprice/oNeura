use super::math::{v3, V3};
use super::mesh::Triangle;
use super::terrain::terrain_height;
use oneura_core::soil_fauna::EarthwormPopulation;

pub fn build_earthworm_meshes(
    tris: &mut Vec<Triangle>,
    population: &EarthwormPopulation,
    gw: usize,
    gh: usize,
    moisture: &[f32],
    _frame: usize,
) {
    // Earthworms are modeled as coarse density field, not individuals yet.
    // Show them as small brown dots on the surface where density is high.
    for (idx, &density) in population.population_density.iter().enumerate() {
        if density <= 0.5 {
            continue;
        }
        let gx = idx % gw;
        let gy = idx / gw;
        let base_h = terrain_height(gx, gy, gw, gh, moisture, 7);

        let mut dot = super::mesh::make_sphere(
            v3(gx as f32 * 0.5, base_h + 0.05, gy as f32 * 0.5),
            0.02,
            [0.4, 0.3, 0.2],
            1.0,
        );
        tris.extend(dot);
    }
}
