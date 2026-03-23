use super::math::{add3, v3, V3};
use super::mesh::{make_diamond, make_sphere, tag_all, EntityTag, Triangle};
use super::terrain::terrain_height;
use oneura_core::ant_colony::{Ant, AntColony};

const ANT_SCALE: f32 = 0.8;

pub fn build_ant_meshes(
    tris: &mut Vec<Triangle>,
    colony: &AntColony,
    gw: usize,
    gh: usize,
    moisture: &[f32],
    frame: usize,
) {
    let mut ant_tris = Vec::with_capacity(colony.ants.len() * 32);

    for (i, ant) in colony.ants.iter().enumerate() {
        let gx = ant.x_mm / 0.5;
        let gy = ant.y_mm / 0.5;
        let base_h = terrain_height(gx as usize, gy as usize, gw, gh, moisture, 7);
        let center = v3(ant.x_mm, base_h + 0.05 * ANT_SCALE, ant.y_mm);

        build_ant_body(&mut ant_tris, center, ant, i);
        build_ant_legs(&mut ant_tris, center, ant, frame, i);
        build_ant_antennae(&mut ant_tris, center, ant, i);
    }
    tris.extend(ant_tris);

    // Pheromones
    for deposit in &colony.pheromones {
        if deposit.strength <= 0.05 {
            continue;
        }
        let gx = (deposit.x_mm / 0.5) as usize;
        let gy = (deposit.y_mm / 0.5) as usize;
        if gx >= gw || gy >= gh {
            continue;
        }
        let base_h = terrain_height(gx, gy, gw, gh, moisture, 7);

        let mut p_mesh = make_sphere(
            v3(deposit.x_mm, base_h + 0.1, deposit.y_mm),
            0.1,
            [1.0, 1.0, 1.0],
            0.5,
        );
        tris.extend(p_mesh);
    }
}

fn build_ant_body(tris: &mut Vec<Triangle>, center: V3, _ant: &Ant, idx: usize) {
    let scale = ANT_SCALE;
    let body_color = [0.2, 0.15, 0.1];

    // Head
    let head_center = add3(center, v3(0.0, 0.0, 0.03 * scale));
    let mut head = make_sphere(head_center, 0.015 * scale, body_color, 4.0);
    tris.extend(head);

    // Mesosoma (thorax)
    let meso_center = center;
    let mut meso = make_diamond(
        meso_center,
        0.03 * scale,
        0.015 * scale,
        0.015 * scale,
        body_color,
        4.0,
    );
    tris.extend(meso);

    // Metasoma (gaster)
    let meta_center = add3(center, v3(0.0, 0.0, -0.04 * scale));
    let mut meta = make_sphere(meta_center, 0.025 * scale, body_color, 4.0);
    tris.extend(meta);

    // Petiole (narrow waist)
    let petiole_center = add3(center, v3(0.0, 0.0, -0.015 * scale));
    let mut petiole = make_sphere(petiole_center, 0.005 * scale, body_color, 4.0);
    tris.extend(petiole);
}

fn build_ant_legs(tris: &mut Vec<Triangle>, center: V3, _ant: &Ant, _frame: usize, _idx: usize) {
    let scale = ANT_SCALE;
    let leg_color = [0.3, 0.25, 0.2];
    let leg_angles = [
        0.0,
        std::f32::consts::FRAC_PI_3,
        std::f32::consts::FRAC_PI_3 * 2.0,
    ];

    for side in [-1.0, 1.0] {
        for i in 0..3 {
            let angle = leg_angles[i];
            let base_x = side * 0.03 * scale * angle.cos();
            let base_z = 0.02 * scale * angle.sin();

            let coxa_center = add3(center, v3(base_x, 0.0, base_z));
            let mut coxa = make_sphere(coxa_center, 0.008 * scale, leg_color, 4.0);
            tris.extend(coxa);
        }
    }
}

fn build_ant_antennae(tris: &mut Vec<Triangle>, center: V3, _ant: &Ant, _idx: usize) {
    let scale = ANT_SCALE;
    let head_top = add3(center, v3(0.0, 0.01 * scale, 0.02 * scale));

    for side in [-1.0, 1.0] {
        let scape_end = add3(head_top, v3(side * 0.03 * scale, 0.0, 0.02 * scale));
        let mut scape = make_diamond(
            scape_end,
            0.02 * scale,
            0.004 * scale,
            0.004 * scale,
            [0.2, 0.2, 0.2],
            1.0,
        );
        tris.extend(scape);
    }
}
