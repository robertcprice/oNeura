//! Fly entity mesh generation — animated body segments with wing flutter.

use super::math::{add3, v3, V3};
use super::mesh::{make_diamond, make_sphere, tag_all, EntityTag, Triangle};
use super::terrain::terrain_height;
use super::CELL_SIZE;
use oneura_core::drosophila::DrosophilaSim;

/// Scale factor for fly visibility (3x bigger for demo)
const FLY_SCALE: f32 = 3.0;

pub fn build_fly_meshes(
    flies: &[DrosophilaSim],
    gw: usize,
    gh: usize,
    moisture: &[f32],
    seed: u64,
    frame: usize,
) -> Vec<Triangle> {
    let mut tris = Vec::with_capacity(flies.len() * 24);
    for (i, fly) in flies.iter().enumerate() {
        let b = fly.body_state();
        let gx = b.x.round().clamp(0.0, (gw - 1) as f32) as usize;
        let gy = b.y.round().clamp(0.0, (gh - 1) as f32) as usize;
        let base_y = terrain_height(gx, gy, gw, gh, moisture, seed);
        let wx = b.x.clamp(0.0, (gw - 1) as f32) * CELL_SIZE;
        let wz = b.y.clamp(0.0, (gh - 1) as f32) * CELL_SIZE;
        let alt = if b.is_flying {
            b.z.clamp(0.0, 4.0) * 0.4
        } else {
            0.05
        };
        let center = v3(wx, base_y + alt + 0.1 * FLY_SCALE, wz);

        // Body color: golden when flying, dull when resting
        let body_color: V3 = if b.is_flying {
            [0.90, 0.82, 0.18]
        } else {
            [0.70, 0.60, 0.12]
        };

        // Head (small sphere) - scaled up
        let head_pos = add3(center, v3(0.0, 0.01 * FLY_SCALE, 0.03 * FLY_SCALE));
        let mut head = make_sphere(head_pos, 0.015 * FLY_SCALE, [0.65, 0.55, 0.10], 16.0);
        tag_all(&mut head, EntityTag::Fly(i));
        tris.extend(head);

        // Thorax (medium diamond) - scaled up
        let mut thorax = make_diamond(
            center,
            0.04 * FLY_SCALE,
            0.025 * FLY_SCALE,
            0.05 * FLY_SCALE,
            body_color,
            16.0,
        );
        tag_all(&mut thorax, EntityTag::Fly(i));
        tris.extend(thorax);

        // Abdomen (slightly behind + below) - scaled up
        let abd_pos = add3(center, v3(0.0, -0.008 * FLY_SCALE, -0.04 * FLY_SCALE));
        let mut abd = make_diamond(
            abd_pos,
            0.035 * FLY_SCALE,
            0.02 * FLY_SCALE,
            0.06 * FLY_SCALE,
            [
                body_color[0] * 0.85,
                body_color[1] * 0.85,
                body_color[2] * 0.85,
            ],
            16.0,
        );
        tag_all(&mut abd, EntityTag::Fly(i));
        tris.extend(abd);

        // Wing flutter animation when flying - scaled up
        if b.is_flying {
            let phase = (frame as f32 + i as f32 * 37.0) * 0.8;
            let wing_angle = phase.sin() * 0.5;
            let wing_y = 0.02 * FLY_SCALE + wing_angle.abs() * 0.02 * FLY_SCALE;
            let wing_color: V3 = [0.90, 0.92, 0.85];

            // Left wing
            let lw = add3(center, v3(-0.04 * FLY_SCALE, wing_y, 0.0));
            let mut left_wing = make_diamond(
                lw,
                0.035 * FLY_SCALE,
                0.005 * FLY_SCALE,
                0.025 * FLY_SCALE,
                wing_color,
                8.0,
            );
            tag_all(&mut left_wing, EntityTag::Fly(i));
            tris.extend(left_wing);

            // Right wing
            let rw = add3(center, v3(0.04 * FLY_SCALE, wing_y, 0.0));
            let mut right_wing = make_diamond(
                rw,
                0.035 * FLY_SCALE,
                0.005 * FLY_SCALE,
                0.025 * FLY_SCALE,
                wing_color,
                8.0,
            );
            tag_all(&mut right_wing, EntityTag::Fly(i));
            tris.extend(right_wing);
        }
    }
    tris
}
