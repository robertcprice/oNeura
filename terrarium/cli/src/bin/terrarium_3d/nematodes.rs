use super::math::{v3, V3};
use super::mesh::{make_sphere, Triangle};
use super::terrain::terrain_height;
use oneura_core::soil_fauna::NematodeGuild;

pub fn build_nematode_meshes(
    tris: &mut Vec<Triangle>,
    guild: &NematodeGuild,
    gw: usize,
    gh: usize,
    moisture: &[f32],
    _frame: usize,
) {
    // Nematodes are modeled as coarse density field.
    // Show them as small translucent spheres where density is high.
    for (idx, &density) in guild.population_density.iter().enumerate() {
        if density <= 0.5 {
            continue;
        }
        let gx = idx % gw;
        let gy = idx / gw;
        let base_h = terrain_height(gx, gy, gw, gh, moisture, 7);

        let mut sphere = make_sphere(
            v3(gx as f32 * 0.5, base_h + 0.05, gy as f32 * 0.5),
            0.03,
            [0.8, 0.8, 0.7],
            0.4,
        );
        tris.extend(sphere);
    }
}

pub fn nematode_info_lines(guild: &NematodeGuild, idx: usize) -> Vec<String> {
    let mut lines = Vec::new();
    lines.push(format!("Type: {:?}", guild.kind));
    if let Some(&density) = guild.population_density.get(idx) {
        lines.push(format!("Density: {:.2} ind/g", density));
    }
    lines
}
