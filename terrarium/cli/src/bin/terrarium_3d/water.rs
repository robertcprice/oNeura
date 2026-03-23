//! Water entity mesh generation — 3D water pools with depth, sitting in terrain depressions.

use super::math::v3;
use super::mesh::{EntityTag, Triangle, Vertex};
use super::terrain::terrain_height;
use super::CELL_SIZE;
use oneura_core::terrarium::TerrariumWorld;

/// Water depth below terrain surface
const WATER_DEPTH: f32 = 0.15;
/// Water color (translucent blue)
const WATER_COLOR: [f32; 3] = [0.20, 0.55, 0.85];
const WATER_COLOR_DEEP: [f32; 3] = [0.10, 0.35, 0.60];

pub fn build_water_meshes(world: &TerrariumWorld, seed: u64, frame_idx: usize) -> Vec<Triangle> {
    let gw = world.config.width;
    let gh = world.config.height;
    let moisture = world.moisture_field();
    let mut tris = Vec::with_capacity(world.waters.len() * 20);

    for (i, water) in world.waters.iter().enumerate() {
        if !water.alive {
            continue;
        }

        let gx = water.x.min(gw - 1);
        let gy = water.y.min(gh - 1);
        let base_y = terrain_height(gx, gy, gw, gh, &moisture, seed);

        // Water sits slightly below terrain to create a pool effect
        let water_surface_y = base_y - 0.02; // Slightly recessed
        let water_bottom_y = water_surface_y - WATER_DEPTH;

        let wx = gx as f32 * CELL_SIZE;
        let wz = gy as f32 * CELL_SIZE;

        // Animated wave on surface
        let wave = (frame_idx as f32 * 0.08 + wx * 2.5 + wz * 1.8).sin() * 0.015
            + (frame_idx as f32 * 0.12 + wx * 1.3 - wz * 2.1).cos() * 0.010;

        let half_size = CELL_SIZE * 0.5;
        let surface_y = water_surface_y + wave;

        // Surface (top face) with animated wave - normal pointing up
        let n_up: [f32; 3] = [0.0, 1.0, 0.0];
        let p0 = v3(wx - half_size, surface_y, wz - half_size);
        let p1 = v3(wx + half_size, surface_y, wz - half_size);
        let p2 = v3(wx + half_size, surface_y, wz + half_size);
        let p3 = v3(wx - half_size, surface_y, wz + half_size);

        tris.push(Triangle {
            v: [
                Vertex {
                    pos: p0,
                    normal: n_up,
                    color: WATER_COLOR,
                    shininess: 64.0,
                },
                Vertex {
                    pos: p1,
                    normal: n_up,
                    color: WATER_COLOR,
                    shininess: 64.0,
                },
                Vertex {
                    pos: p2,
                    normal: n_up,
                    color: WATER_COLOR,
                    shininess: 64.0,
                },
            ],
            tag: EntityTag::Water(i),
        });
        tris.push(Triangle {
            v: [
                Vertex {
                    pos: p0,
                    normal: n_up,
                    color: WATER_COLOR,
                    shininess: 64.0,
                },
                Vertex {
                    pos: p2,
                    normal: n_up,
                    color: WATER_COLOR,
                    shininess: 64.0,
                },
                Vertex {
                    pos: p3,
                    normal: n_up,
                    color: WATER_COLOR,
                    shininess: 64.0,
                },
            ],
            tag: EntityTag::Water(i),
        });

        // Bottom face - normal pointing down
        let n_down: [f32; 3] = [0.0, -1.0, 0.0];
        let b0 = v3(wx - half_size, water_bottom_y, wz - half_size);
        let b1 = v3(wx + half_size, water_bottom_y, wz - half_size);
        let b2 = v3(wx + half_size, water_bottom_y, wz + half_size);
        let b3 = v3(wx - half_size, water_bottom_y, wz + half_size);

        tris.push(Triangle {
            v: [
                Vertex {
                    pos: b0,
                    normal: n_down,
                    color: WATER_COLOR_DEEP,
                    shininess: 32.0,
                },
                Vertex {
                    pos: b2,
                    normal: n_down,
                    color: WATER_COLOR_DEEP,
                    shininess: 32.0,
                },
                Vertex {
                    pos: b1,
                    normal: n_down,
                    color: WATER_COLOR_DEEP,
                    shininess: 32.0,
                },
            ],
            tag: EntityTag::Water(i),
        });
        tris.push(Triangle {
            v: [
                Vertex {
                    pos: b0,
                    normal: n_down,
                    color: WATER_COLOR_DEEP,
                    shininess: 32.0,
                },
                Vertex {
                    pos: b3,
                    normal: n_down,
                    color: WATER_COLOR_DEEP,
                    shininess: 32.0,
                },
                Vertex {
                    pos: b2,
                    normal: n_down,
                    color: WATER_COLOR_DEEP,
                    shininess: 32.0,
                },
            ],
            tag: EntityTag::Water(i),
        });

        // Side walls
        // South wall (z-)
        let n_s: [f32; 3] = [0.0, 0.0, -1.0];
        tris.push(Triangle {
            v: [
                Vertex {
                    pos: b0,
                    normal: n_s,
                    color: WATER_COLOR_DEEP,
                    shininess: 32.0,
                },
                Vertex {
                    pos: b1,
                    normal: n_s,
                    color: WATER_COLOR_DEEP,
                    shininess: 32.0,
                },
                Vertex {
                    pos: p0,
                    normal: n_s,
                    color: WATER_COLOR,
                    shininess: 32.0,
                },
            ],
            tag: EntityTag::Water(i),
        });
        tris.push(Triangle {
            v: [
                Vertex {
                    pos: b1,
                    normal: n_s,
                    color: WATER_COLOR_DEEP,
                    shininess: 32.0,
                },
                Vertex {
                    pos: p1,
                    normal: n_s,
                    color: WATER_COLOR,
                    shininess: 32.0,
                },
                Vertex {
                    pos: p0,
                    normal: n_s,
                    color: WATER_COLOR,
                    shininess: 32.0,
                },
            ],
            tag: EntityTag::Water(i),
        });

        // North wall (z+)
        let n_n: [f32; 3] = [0.0, 0.0, 1.0];
        tris.push(Triangle {
            v: [
                Vertex {
                    pos: b3,
                    normal: n_n,
                    color: WATER_COLOR_DEEP,
                    shininess: 32.0,
                },
                Vertex {
                    pos: p3,
                    normal: n_n,
                    color: WATER_COLOR,
                    shininess: 32.0,
                },
                Vertex {
                    pos: p2,
                    normal: n_n,
                    color: WATER_COLOR,
                    shininess: 32.0,
                },
            ],
            tag: EntityTag::Water(i),
        });
        tris.push(Triangle {
            v: [
                Vertex {
                    pos: b3,
                    normal: n_n,
                    color: WATER_COLOR_DEEP,
                    shininess: 32.0,
                },
                Vertex {
                    pos: p2,
                    normal: n_n,
                    color: WATER_COLOR,
                    shininess: 32.0,
                },
                Vertex {
                    pos: b2,
                    normal: n_n,
                    color: WATER_COLOR_DEEP,
                    shininess: 32.0,
                },
            ],
            tag: EntityTag::Water(i),
        });

        // West wall (x-)
        let n_w: [f32; 3] = [-1.0, 0.0, 0.0];
        tris.push(Triangle {
            v: [
                Vertex {
                    pos: b0,
                    normal: n_w,
                    color: WATER_COLOR_DEEP,
                    shininess: 32.0,
                },
                Vertex {
                    pos: p0,
                    normal: n_w,
                    color: WATER_COLOR,
                    shininess: 32.0,
                },
                Vertex {
                    pos: p3,
                    normal: n_w,
                    color: WATER_COLOR,
                    shininess: 32.0,
                },
            ],
            tag: EntityTag::Water(i),
        });
        tris.push(Triangle {
            v: [
                Vertex {
                    pos: b0,
                    normal: n_w,
                    color: WATER_COLOR_DEEP,
                    shininess: 32.0,
                },
                Vertex {
                    pos: p3,
                    normal: n_w,
                    color: WATER_COLOR,
                    shininess: 32.0,
                },
                Vertex {
                    pos: b3,
                    normal: n_w,
                    color: WATER_COLOR_DEEP,
                    shininess: 32.0,
                },
            ],
            tag: EntityTag::Water(i),
        });

        // East wall (x+)
        let n_e: [f32; 3] = [1.0, 0.0, 0.0];
        tris.push(Triangle {
            v: [
                Vertex {
                    pos: b1,
                    normal: n_e,
                    color: WATER_COLOR_DEEP,
                    shininess: 32.0,
                },
                Vertex {
                    pos: p2,
                    normal: n_e,
                    color: WATER_COLOR,
                    shininess: 32.0,
                },
                Vertex {
                    pos: p1,
                    normal: n_e,
                    color: WATER_COLOR,
                    shininess: 32.0,
                },
            ],
            tag: EntityTag::Water(i),
        });
        tris.push(Triangle {
            v: [
                Vertex {
                    pos: b1,
                    normal: n_e,
                    color: WATER_COLOR_DEEP,
                    shininess: 32.0,
                },
                Vertex {
                    pos: b2,
                    normal: n_e,
                    color: WATER_COLOR_DEEP,
                    shininess: 32.0,
                },
                Vertex {
                    pos: p2,
                    normal: n_e,
                    color: WATER_COLOR,
                    shininess: 32.0,
                },
            ],
            tag: EntityTag::Water(i),
        });
    }
    tris
}
