use crate::constants::clamp;
use crate::drosophila::BodyState;
use crate::plant_cellular::{PlantClusterSnapshot, PlantTissue};
use crate::seed_cellular::SeedClusterSnapshot;
use super::MICROBIAL_PACKET_TARGET_CELLS;
use crate::terrarium_render::{
    TerrariumFlyPartKind, TerrariumFlyPartRender, TerrariumFruitPartKind,
    TerrariumFruitPartRender, TerrariumPbrMaterialRender, TerrariumSeedPartKind,
    TerrariumSeedPartRender, TerrariumTriangleMeshRender,
};
use crate::whole_cell_submodels::LocalChemistrySiteReport;

use super::mesh::{
    bezier3, dot3, mul3, normalize3, mesh_append, mesh_rotate_xyz, mesh_translate,
    render_cylinder_mesh, render_ellipsoid_mesh, render_segment_mesh,
    rotate_xyz,
};
use super::packet::GenotypePacket;
use super::render_utils::*;
use super::{TerrariumPlantGenome, TerrariumSeed, EXPLICIT_MICROBE_COHORT_CELLS};

pub(super) fn plant_cluster_energy_charge(snapshot: &PlantClusterSnapshot) -> f32 {
    let total =
        snapshot.chem_atp.max(0.0) + snapshot.chem_adp.max(0.0) + snapshot.chem_amp.max(0.0);
    if total <= 1.0e-6 {
        0.0
    } else {
        ((snapshot.chem_atp.max(0.0) + snapshot.chem_adp.max(0.0) * 0.5) / total).clamp(0.0, 1.0)
    }
}

pub(super) fn plant_cluster_hydration(snapshot: &PlantClusterSnapshot) -> f32 {
    let bound = snapshot.state_water.max(0.0) + snapshot.cytoplasm_water.max(0.0) * 0.4;
    (bound / (bound + 8.0)).clamp(0.0, 1.0)
}

pub(super) fn render_stateful_stem_mesh(
    height: f32,
    base_radius: f32,
    canopy_offset_world: [f32; 3],
    stem_cluster: &PlantClusterSnapshot,
) -> TerrariumTriangleMeshRender {
    let mut mesh = TerrariumTriangleMeshRender::default();
    let base = [0.0, -height * 0.5, 0.0];
    let tip = [
        canopy_offset_world[0] * 0.82,
        height * 0.5 + canopy_offset_world[1] * 0.55,
        canopy_offset_world[2] * 0.82,
    ];
    let ctrl_a = [
        canopy_offset_world[0] * 0.12,
        -height * 0.18 + canopy_offset_world[1] * 0.08,
        canopy_offset_world[2] * 0.12,
    ];
    let ctrl_b = [
        canopy_offset_world[0] * 0.58,
        height * 0.16 + canopy_offset_world[1] * 0.26,
        canopy_offset_world[2] * 0.58,
    ];
    let transport_t = stem_cluster.transcript_transport_program.clamp(0.0, 1.0);
    let vitality_t = stem_cluster.vitality.clamp(0.0, 1.0);
    let hydration_t = plant_cluster_hydration(stem_cluster);
    let segments = (4.0
        + (stem_cluster.cell_count / 90.0).sqrt().clamp(0.0, 2.0)
        + transport_t * 2.0
        + hydration_t)
        .round()
        .clamp(4.0, 8.0) as usize;
    let mut previous = base;
    for segment in 1..=segments {
        let t = segment as f32 / segments as f32;
        let next = bezier3(base, ctrl_a, ctrl_b, tip, t);
        let radius_t = 1.0 - (segment as f32 / segments as f32) * (0.34 + transport_t * 0.10);
        let radius = base_radius
            * radius_t.max(0.45)
            * (0.88 + transport_t * 0.16 + vitality_t * 0.08 + hydration_t * 0.10);
        let segment_mesh = render_segment_mesh(previous, next, radius, 12);
        mesh_append(&mut mesh, &segment_mesh);
        previous = next;
    }
    mesh
}

pub(super) fn render_stateful_canopy_mesh(
    canopy_scale_world: [f32; 3],
    genome: &TerrariumPlantGenome,
    health: f32,
    leaf_energy: f32,
    canopy_velocity_world: [f32; 3],
    leaf_cluster: &PlantClusterSnapshot,
) -> TerrariumTriangleMeshRender {
    let mut mesh = TerrariumTriangleMeshRender::default();
    let leaf_vitality = leaf_cluster.vitality.clamp(0.0, 1.0);
    let hydration_t = plant_cluster_hydration(leaf_cluster);
    let stress_t = leaf_cluster.transcript_stress_response.clamp(0.0, 1.0);
    let division_t = leaf_cluster.division_buffer.clamp(0.0, 1.0);
    let core_radii = [
        canopy_scale_world[0] * (0.26 + genome.shade_tolerance * 0.05 + leaf_vitality * 0.04),
        canopy_scale_world[1] * (0.28 + leaf_energy * 0.12 + hydration_t * 0.05),
        canopy_scale_world[2] * (0.26 + genome.leaf_efficiency * 0.04 + hydration_t * 0.04),
    ];
    let core = render_ellipsoid_mesh(core_radii, 16, 10);
    mesh_append(&mut mesh, &core);

    let lobe_count = (3.0
        + genome.leaf_efficiency * 0.8
        + genome.shade_tolerance * 0.6
        + division_t * 1.2
        + (leaf_cluster.cell_count / 140.0).sqrt().clamp(0.0, 1.5))
    .round()
    .clamp(3.0, 7.0) as usize;
    let velocity_angle = canopy_velocity_world[2].atan2(canopy_velocity_world[0]);
    let velocity_mag = (canopy_velocity_world[0] * canopy_velocity_world[0]
        + canopy_velocity_world[2] * canopy_velocity_world[2])
        .sqrt()
        .clamp(0.0, canopy_scale_world[0].max(canopy_scale_world[2]));
    for lobe in 0..lobe_count {
        let phase = velocity_angle + lobe as f32 / lobe_count as f32 * std::f32::consts::TAU;
        let (sin_phase, cos_phase) = phase.sin_cos();
        let lobe_radii = [
            canopy_scale_world[0]
                * (0.13 + health * 0.06 + genome.seed_mass * 0.18 + leaf_vitality * 0.06),
            canopy_scale_world[1] * (0.18 + leaf_energy * 0.10 + hydration_t * 0.08),
            canopy_scale_world[2]
                * (0.14 + genome.shade_tolerance * 0.07 + (1.0 - stress_t) * 0.05),
        ];
        let mut lobe_mesh = render_ellipsoid_mesh(lobe_radii, 14, 9);
        mesh_rotate_xyz(
            &mut lobe_mesh,
            [
                canopy_velocity_world[2] * 0.12,
                phase * 0.22,
                -canopy_velocity_world[0] * 0.12,
            ],
        );
        mesh_translate(
            &mut lobe_mesh,
            [
                cos_phase * canopy_scale_world[0] * 0.22
                    + canopy_velocity_world[0] * 0.12
                    + velocity_mag * cos_phase * 0.06,
                canopy_scale_world[1]
                    * (-0.10 + (lobe as f32 % 2.0) * 0.16 + hydration_t * 0.05 - stress_t * 0.04),
                sin_phase * canopy_scale_world[2] * 0.22
                    + canopy_velocity_world[2] * 0.12
                    + velocity_mag * sin_phase * 0.06,
            ],
        );
        mesh_append(&mut mesh, &lobe_mesh);
    }
    mesh
}

pub(super) fn render_stateful_plant_tissue_mesh(
    tissue: PlantTissue,
    scale_world: [f32; 3],
    snapshot: &PlantClusterSnapshot,
) -> TerrariumTriangleMeshRender {
    let vitality_t = snapshot.vitality.clamp(0.0, 1.0);
    let energy_t = plant_cluster_energy_charge(snapshot);
    let hydration_t = plant_cluster_hydration(snapshot);
    let division_t = snapshot.division_buffer.clamp(0.0, 1.0);
    let stress_t = snapshot.transcript_stress_response.clamp(0.0, 1.0);
    let transport_t = snapshot.transcript_transport_program.clamp(0.0, 1.0);
    let cell_cycle_t = snapshot.transcript_cell_cycle.clamp(0.0, 1.0);
    let mut mesh = TerrariumTriangleMeshRender::default();

    match tissue {
        PlantTissue::Leaf => {
            let core = render_ellipsoid_mesh(
                [
                    scale_world[0] * (0.70 + vitality_t * 0.18),
                    scale_world[1] * (0.55 + hydration_t * 0.18),
                    scale_world[2] * (0.70 + energy_t * 0.16),
                ],
                16,
                10,
            );
            mesh_append(&mut mesh, &core);
            let lobes = (3.0 + division_t * 2.0 + (snapshot.cell_count / 110.0).sqrt())
                .round()
                .clamp(3.0, 6.0) as usize;
            for lobe in 0..lobes {
                let phase = lobe as f32 / lobes as f32 * std::f32::consts::TAU;
                let (sin_phase, cos_phase) = phase.sin_cos();
                let mut lobe_mesh = render_ellipsoid_mesh(
                    [
                        scale_world[0] * (0.24 + hydration_t * 0.10),
                        scale_world[1] * (0.20 + vitality_t * 0.08),
                        scale_world[2] * (0.18 + (1.0 - stress_t) * 0.10),
                    ],
                    12,
                    8,
                );
                mesh_translate(
                    &mut lobe_mesh,
                    [
                        cos_phase * scale_world[0] * 0.34,
                        scale_world[1] * (-0.06 + (lobe % 2) as f32 * 0.14),
                        sin_phase * scale_world[2] * 0.34,
                    ],
                );
                mesh_append(&mut mesh, &lobe_mesh);
            }
        }
        PlantTissue::Stem => {
            let start = [0.0, -scale_world[1] * 0.5, 0.0];
            let end = [0.0, scale_world[1] * 0.5, 0.0];
            let trunk = render_segment_mesh(
                start,
                end,
                scale_world[0] * (0.32 + transport_t * 0.12 + energy_t * 0.06),
                12,
            );
            mesh_append(&mut mesh, &trunk);
            let rings = (2.0 + transport_t * 3.0).round().clamp(2.0, 5.0) as usize;
            for ring in 0..rings {
                let ring_t = ring as f32 / rings as f32;
                let mut collar = render_ellipsoid_mesh(
                    [
                        scale_world[0] * (0.36 + hydration_t * 0.10),
                        scale_world[1] * 0.06,
                        scale_world[2] * (0.36 + vitality_t * 0.08),
                    ],
                    12,
                    7,
                );
                mesh_translate(
                    &mut collar,
                    [
                        0.0,
                        -scale_world[1] * 0.36 + (scale_world[1] * 0.72) * ring_t.clamp(0.0, 1.0),
                        0.0,
                    ],
                );
                mesh_append(&mut mesh, &collar);
            }
        }
        PlantTissue::Root => {
            let anchor = [0.0, scale_world[1] * 0.12, 0.0];
            let tip = [0.0, -scale_world[1] * 0.42, 0.0];
            let trunk = render_segment_mesh(
                anchor,
                tip,
                scale_world[0] * (0.18 + hydration_t * 0.08 + transport_t * 0.06),
                12,
            );
            mesh_append(&mut mesh, &trunk);
            let branches = (3.0 + hydration_t * 2.0 + (snapshot.cell_count / 120.0).sqrt())
                .round()
                .clamp(3.0, 6.0) as usize;
            for branch in 0..branches {
                let phase = branch as f32 / branches as f32 * std::f32::consts::TAU;
                let (sin_phase, cos_phase) = phase.sin_cos();
                let start = [
                    0.0,
                    scale_world[1] * 0.06
                        + (-scale_world[1] * 0.30)
                            * (branch as f32 / branches as f32).clamp(0.0, 1.0),
                    0.0,
                ];
                let end = [
                    cos_phase * scale_world[0] * (0.45 + energy_t * 0.10),
                    -scale_world[1] * (0.26 + branch as f32 / branches as f32 * 0.32),
                    sin_phase * scale_world[2] * (0.45 + transport_t * 0.12),
                ];
                let branch_mesh = render_segment_mesh(
                    start,
                    end,
                    scale_world[0] * (0.08 + (1.0 - stress_t) * 0.03),
                    10,
                );
                mesh_append(&mut mesh, &branch_mesh);
            }
        }
        PlantTissue::Meristem => {
            let core = render_ellipsoid_mesh(
                [
                    scale_world[0] * (0.72 + division_t * 0.20),
                    scale_world[1] * (0.64 + cell_cycle_t * 0.24),
                    scale_world[2] * (0.72 + vitality_t * 0.16),
                ],
                14,
                9,
            );
            mesh_append(&mut mesh, &core);
            let buds = (2.0 + division_t * 3.0).round().clamp(2.0, 5.0) as usize;
            for bud in 0..buds {
                let phase = bud as f32 / buds as f32 * std::f32::consts::TAU;
                let (sin_phase, cos_phase) = phase.sin_cos();
                let mut bud_mesh = render_ellipsoid_mesh(
                    [
                        scale_world[0] * (0.16 + division_t * 0.08),
                        scale_world[1] * (0.18 + cell_cycle_t * 0.08),
                        scale_world[2] * (0.16 + energy_t * 0.06),
                    ],
                    10,
                    7,
                );
                mesh_translate(
                    &mut bud_mesh,
                    [
                        cos_phase * scale_world[0] * 0.32,
                        scale_world[1] * (0.20 + division_t * 0.12),
                        sin_phase * scale_world[2] * 0.32,
                    ],
                );
                mesh_append(&mut mesh, &bud_mesh);
            }
        }
    }

    mesh
}

pub(super) fn render_stateful_fruit_mesh(
    radius: f32,
    ripeness: f32,
    sugar_content: f32,
    offset_world: [f32; 3],
    velocity_world: [f32; 3],
) -> TerrariumTriangleMeshRender {
    let sectors = 20usize;
    let stacks = 12usize;
    let rx = radius * (0.44 + sugar_content * 0.08);
    let ry = radius * (0.46 + ripeness * 0.10);
    let rz = radius * (0.44 + (1.0 - ripeness) * 0.06);
    let rib_count = (5.0 + ripeness * 3.0).round().clamp(5.0, 8.0) as usize;
    let rib_amp = radius * (0.03 + sugar_content * 0.05);
    let droop = clamp(
        -velocity_world[1] * 0.004 + offset_world[1] * 0.08,
        -radius * 0.10,
        radius * 0.10,
    );
    let mut mesh = TerrariumTriangleMeshRender::default();
    for stack in 0..=stacks {
        let v = stack as f32 / stacks as f32;
        let phi = std::f32::consts::PI * v;
        let ring = phi.sin();
        let y_unit = phi.cos();
        for sector in 0..=sectors {
            let u = sector as f32 / sectors as f32;
            let theta = u * std::f32::consts::TAU;
            let (sin_theta, cos_theta) = theta.sin_cos();
            let rib = (theta * rib_count as f32).sin() * ring * rib_amp;
            let x = cos_theta * (rx + rib);
            let z = sin_theta * (rz + rib * 0.8);
            let y = y_unit * ry - ring * ring * droop;
            mesh.positions.push([x, y, z]);
            mesh.normals.push(normalize3([
                x / rx.max(1.0e-4),
                y / ry.max(1.0e-4),
                z / rz.max(1.0e-4),
            ]));
            mesh.uvs.push([u, 1.0 - v]);
        }
    }
    let ring_vertices = sectors + 1;
    for stack in 0..stacks {
        for sector in 0..sectors {
            let base = (stack * ring_vertices + sector) as u32;
            let next = base + ring_vertices as u32;
            mesh.indices
                .extend([base, next, next + 1, base, next + 1, base + 1]);
        }
    }
    mesh
}

pub(super) fn render_stateful_seed_parts(
    seed_primary: u64,
    seed: &TerrariumSeed,
    uniform_scale_world: f32,
    coat_cluster: &SeedClusterSnapshot,
    endosperm_cluster: &SeedClusterSnapshot,
    radicle_cluster: &SeedClusterSnapshot,
    cotyledon_cluster: &SeedClusterSnapshot,
    coat_material: TerrariumPbrMaterialRender,
    endosperm_material: TerrariumPbrMaterialRender,
    radicle_material: TerrariumPbrMaterialRender,
    cotyledon_material: TerrariumPbrMaterialRender,
) -> Vec<TerrariumSeedPartRender> {
    let dormancy_t = clamp(seed.dormancy_s / 26_000.0, 0.0, 1.0);
    let shade_t = clamp((seed.genome.shade_tolerance - 0.4) / 1.3, 0.0, 1.0);
    let root_bias_t = clamp(seed.genome.root_depth_bias / 1.1, 0.0, 1.0);
    let coat_cells_t = (coat_cluster.cell_count / 18.0).sqrt().clamp(0.55, 1.6);
    let endosperm_cells_t = (endosperm_cluster.cell_count / 22.0)
        .sqrt()
        .clamp(0.45, 1.8);
    let radicle_cells_t = (radicle_cluster.cell_count / 10.0).sqrt().clamp(0.35, 1.9);
    let cotyledon_cells_t = (cotyledon_cluster.cell_count / 12.0)
        .sqrt()
        .clamp(0.40, 1.8);
    let radicle_growth_t = (seed.cellular.last_feedback().radicle_extension / 1.5).clamp(0.0, 1.0);
    let cotyledon_open_t = (seed.cellular.last_feedback().cotyledon_opening / 1.5).clamp(0.0, 1.0);
    let coat_integrity_t = seed.cellular.last_feedback().coat_integrity.clamp(0.0, 1.0);
    let base_radius = (uniform_scale_world * 0.5).max(1.0e-4);
    let coat_radii = [
        base_radius * (0.72 + coat_cells_t * 0.18 + shade_t * 0.04 + coat_integrity_t * 0.06),
        base_radius
            * (0.92
                + coat_cluster.hydration.min(1.2) * 0.12
                + dormancy_t * 0.08
                + coat_cluster.vitality * 0.05),
        base_radius * (0.70 + coat_cells_t * 0.16 + coat_integrity_t * 0.10),
    ];
    let endosperm_radii = [
        coat_radii[0] * (0.62 + endosperm_cells_t * 0.10 + endosperm_cluster.energy_charge * 0.06),
        coat_radii[1] * (0.64 + endosperm_cluster.hydration.min(1.2) * 0.10),
        coat_radii[2] * (0.62 + endosperm_cluster.sugar_pool.min(2.0) * 0.06),
    ];
    let cotyledon_radii = [
        base_radius * (0.16 + cotyledon_cells_t * 0.12 + cotyledon_open_t * 0.06),
        base_radius * (0.10 + cotyledon_cluster.hydration.min(1.2) * 0.06),
        base_radius * (0.14 + cotyledon_cluster.energy_charge * 0.08 + cotyledon_open_t * 0.04),
    ];
    let radicle_radius =
        base_radius * (0.06 + radicle_cells_t * 0.05 + radicle_cluster.energy_charge * 0.03);
    let radicle_height = base_radius
        * (0.48
            + radicle_cells_t * 0.24
            + radicle_growth_t * 0.42
            + radicle_cluster.transcript_germination_program.min(1.4) * 0.10
            + root_bias_t * 0.10);

    vec![
        TerrariumSeedPartRender {
            render_id: terrarium_render_id(
                TERRARIUM_RENDER_ID_SEED,
                seed_primary,
                terrarium_seed_part_render_slot(TerrariumSeedPartKind::Coat),
            ),
            render_fingerprint: 0,
            kind: TerrariumSeedPartKind::Coat,
            mesh: render_ellipsoid_mesh(coat_radii, 16, 10),
            translation_local: [0.0, 0.0, 0.0],
            rotation_xyz_rad: [
                0.05 * (0.5 - coat_cluster.hydration.min(1.0)),
                (shade_t - 0.5) * 0.18,
                0.10 * (root_bias_t - 0.5),
            ],
            material: coat_material,
        },
        TerrariumSeedPartRender {
            render_id: terrarium_render_id(
                TERRARIUM_RENDER_ID_SEED,
                seed_primary,
                terrarium_seed_part_render_slot(TerrariumSeedPartKind::Endosperm),
            ),
            render_fingerprint: 0,
            kind: TerrariumSeedPartKind::Endosperm,
            mesh: render_ellipsoid_mesh(endosperm_radii, 14, 9),
            translation_local: [0.0, base_radius * 0.02, 0.0],
            rotation_xyz_rad: [
                endosperm_cluster.transcript_germination_program.min(1.0) * 0.05,
                0.0,
                0.0,
            ],
            material: endosperm_material,
        },
        TerrariumSeedPartRender {
            render_id: terrarium_render_id(
                TERRARIUM_RENDER_ID_SEED,
                seed_primary,
                terrarium_seed_part_render_slot(TerrariumSeedPartKind::Radicle),
            ),
            render_fingerprint: 0,
            kind: TerrariumSeedPartKind::Radicle,
            mesh: render_cylinder_mesh(radicle_radius, radicle_height, 12),
            translation_local: [
                0.0,
                -coat_radii[1] * 0.48,
                coat_radii[2] * (0.06 + radicle_growth_t * 0.12),
            ],
            rotation_xyz_rad: [
                0.12 + (0.5 - radicle_cluster.hydration.min(1.0)) * 0.10 + radicle_growth_t * 0.14,
                0.0,
                (root_bias_t - 0.5) * 0.35 + seed.pose.rotation_xyz_rad[2] * 0.12,
            ],
            material: radicle_material,
        },
        TerrariumSeedPartRender {
            render_id: terrarium_render_id(
                TERRARIUM_RENDER_ID_SEED,
                seed_primary,
                terrarium_seed_part_render_slot(TerrariumSeedPartKind::CotyledonLeft),
            ),
            render_fingerprint: 0,
            kind: TerrariumSeedPartKind::CotyledonLeft,
            mesh: render_ellipsoid_mesh(cotyledon_radii, 12, 8),
            translation_local: [
                -coat_radii[0] * 0.24,
                coat_radii[1] * (0.08 + cotyledon_open_t * 0.10),
                -coat_radii[2] * (0.04 + cotyledon_open_t * 0.10),
            ],
            rotation_xyz_rad: [
                0.12 + cotyledon_open_t * 0.18,
                -(0.10 + cotyledon_open_t * 0.16),
                -(0.08 + cotyledon_cluster.energy_charge * 0.12),
            ],
            material: cotyledon_material.clone(),
        },
        TerrariumSeedPartRender {
            render_id: terrarium_render_id(
                TERRARIUM_RENDER_ID_SEED,
                seed_primary,
                terrarium_seed_part_render_slot(TerrariumSeedPartKind::CotyledonRight),
            ),
            render_fingerprint: 0,
            kind: TerrariumSeedPartKind::CotyledonRight,
            mesh: render_ellipsoid_mesh(cotyledon_radii, 12, 8),
            translation_local: [
                coat_radii[0] * 0.24,
                coat_radii[1] * (0.08 + cotyledon_open_t * 0.10),
                -coat_radii[2] * (0.04 + cotyledon_open_t * 0.10),
            ],
            rotation_xyz_rad: [
                0.12 + cotyledon_open_t * 0.18,
                0.10 + cotyledon_open_t * 0.16,
                0.08 + cotyledon_cluster.energy_charge * 0.12,
            ],
            material: cotyledon_material,
        },
    ]
}

pub(super) fn render_stateful_fruit_parts(
    fruit_primary: u64,
    radius: f32,
    ripeness: f32,
    sugar_content: f32,
    odor_t: f32,
    microbial_t: f32,
    humidity_t: f32,
    offset_world: [f32; 3],
    velocity_world: [f32; 3],
    skin_material: TerrariumPbrMaterialRender,
    pulp_material: TerrariumPbrMaterialRender,
    core_material: TerrariumPbrMaterialRender,
    stem_material: TerrariumPbrMaterialRender,
) -> Vec<TerrariumFruitPartRender> {
    let inner_radii = [
        radius * (0.30 + sugar_content * 0.05),
        radius * (0.34 + ripeness * 0.05),
        radius * (0.30 + sugar_content * 0.04),
    ];
    let core_radii = [
        radius * (0.10 + microbial_t * 0.03),
        radius * (0.20 + (1.0 - ripeness) * 0.05),
        radius * (0.10 + odor_t * 0.03),
    ];
    let stem_height = radius * (0.34 + (1.0 - ripeness) * 0.10 + humidity_t * 0.06);
    let stem_radius = radius * (0.06 + odor_t * 0.02);
    let sway_x = clamp(velocity_world[2] * 0.05, -0.18, 0.18);
    let sway_z = clamp(-velocity_world[0] * 0.05, -0.18, 0.18);

    vec![
        TerrariumFruitPartRender {
            render_id: terrarium_render_id(
                TERRARIUM_RENDER_ID_FRUIT,
                fruit_primary,
                terrarium_fruit_part_render_slot(TerrariumFruitPartKind::Skin),
            ),
            render_fingerprint: 0,
            kind: TerrariumFruitPartKind::Skin,
            mesh: render_stateful_fruit_mesh(
                radius,
                ripeness,
                sugar_content,
                offset_world,
                velocity_world,
            ),
            translation_local: [0.0, 0.0, 0.0],
            rotation_xyz_rad: [0.0, 0.0, 0.0],
            material: skin_material,
        },
        TerrariumFruitPartRender {
            render_id: terrarium_render_id(
                TERRARIUM_RENDER_ID_FRUIT,
                fruit_primary,
                terrarium_fruit_part_render_slot(TerrariumFruitPartKind::Pulp),
            ),
            render_fingerprint: 0,
            kind: TerrariumFruitPartKind::Pulp,
            mesh: render_ellipsoid_mesh(inner_radii, 16, 10),
            translation_local: [0.0, -radius * 0.02, 0.0],
            rotation_xyz_rad: [0.0, 0.0, 0.0],
            material: pulp_material,
        },
        TerrariumFruitPartRender {
            render_id: terrarium_render_id(
                TERRARIUM_RENDER_ID_FRUIT,
                fruit_primary,
                terrarium_fruit_part_render_slot(TerrariumFruitPartKind::Core),
            ),
            render_fingerprint: 0,
            kind: TerrariumFruitPartKind::Core,
            mesh: render_ellipsoid_mesh(core_radii, 12, 8),
            translation_local: [0.0, -radius * 0.04, 0.0],
            rotation_xyz_rad: [0.08 + sway_x * 0.3, odor_t * 0.12, sway_z * 0.3],
            material: core_material,
        },
        TerrariumFruitPartRender {
            render_id: terrarium_render_id(
                TERRARIUM_RENDER_ID_FRUIT,
                fruit_primary,
                terrarium_fruit_part_render_slot(TerrariumFruitPartKind::Stem),
            ),
            render_fingerprint: 0,
            kind: TerrariumFruitPartKind::Stem,
            mesh: render_cylinder_mesh(stem_radius, stem_height, 10),
            translation_local: [0.0, radius * 0.56 + stem_height * 0.40, 0.0],
            rotation_xyz_rad: [sway_x, 0.0, sway_z],
            material: stem_material,
        },
    ]
}

pub(super) fn render_stateful_water_mesh(
    scale_world: [f32; 3],
    airflow_world: [f32; 3],
    pressure_t: f32,
    humidity_t: f32,
    volume_t: f32,
) -> TerrariumTriangleMeshRender {
    let mut mesh = render_ellipsoid_mesh(
        [
            scale_world[0] * (0.46 + humidity_t * 0.04),
            scale_world[1] * (0.42 + pressure_t * 0.08 + volume_t * 0.06),
            scale_world[2] * (0.46 + humidity_t * 0.04),
        ],
        18,
        11,
    );
    let horizontal_flow = [airflow_world[0], 0.0, airflow_world[2]];
    let flow_mag = dot3(horizontal_flow, horizontal_flow)
        .sqrt()
        .clamp(0.0, 1.8);
    let flow_angle = horizontal_flow[2].atan2(horizontal_flow[0]);
    let ripple_count = (2.0 + volume_t * 2.0 + humidity_t * 1.6)
        .round()
        .clamp(2.0, 5.0) as usize;
    for ripple in 0..ripple_count {
        let ripple_t = if ripple_count <= 1 {
            0.0
        } else {
            ripple as f32 / (ripple_count - 1) as f32
        };
        let phase = flow_angle + ripple_t * std::f32::consts::TAU * 0.42 + pressure_t * 0.55;
        let (sin_phase, cos_phase) = phase.sin_cos();
        let mut lobe = render_ellipsoid_mesh(
            [
                scale_world[0] * (0.14 + ripple_t * 0.08 + flow_mag * 0.03),
                scale_world[1] * (0.07 + humidity_t * 0.03 + volume_t * 0.02),
                scale_world[2] * (0.14 + ripple_t * 0.08 + flow_mag * 0.03),
            ],
            12,
            7,
        );
        mesh_rotate_xyz(&mut lobe, [0.0, phase * 0.12, 0.0]);
        mesh_translate(
            &mut lobe,
            [
                cos_phase * scale_world[0] * (0.10 + ripple_t * 0.18)
                    + airflow_world[0] * (0.03 + ripple_t * 0.02),
                scale_world[1] * (-0.18 + ripple_t * 0.06 + pressure_t * 0.04),
                sin_phase * scale_world[2] * (0.10 + ripple_t * 0.18)
                    + airflow_world[2] * (0.03 + ripple_t * 0.02),
            ],
        );
        mesh_append(&mut mesh, &lobe);
    }
    if flow_mag > 1.0e-4 {
        let wake = render_segment_mesh(
            [0.0, scale_world[1] * 0.02, 0.0],
            [
                horizontal_flow[0].clamp(-1.6, 1.6) * scale_world[0] * 0.18,
                scale_world[1] * (-0.04 + pressure_t * 0.05),
                horizontal_flow[2].clamp(-1.6, 1.6) * scale_world[2] * 0.18,
            ],
            scale_world[1] * (0.10 + volume_t * 0.04),
            8,
        );
        mesh_append(&mut mesh, &wake);
    }
    mesh
}

pub(super) fn render_stateful_plume_mesh(
    scale_world: [f32; 3],
    intensity: f32,
    airflow_world: [f32; 3],
    pressure_bias: f32,
    humidity_t: f32,
) -> TerrariumTriangleMeshRender {
    let mut mesh = TerrariumTriangleMeshRender::default();
    let core = render_ellipsoid_mesh(
        [
            scale_world[0] * 0.26,
            scale_world[1] * 0.34,
            scale_world[2] * 0.26,
        ],
        14,
        8,
    );
    mesh_append(&mut mesh, &core);
    let flow_mag = dot3(airflow_world, airflow_world).sqrt().clamp(0.0, 1.6);
    let flow_dir = if flow_mag <= 1.0e-6 {
        [0.0, 1.0, 0.0]
    } else {
        mul3(airflow_world, flow_mag.recip())
    };
    let tail_end = [
        flow_dir[0] * scale_world[0] * (0.16 + flow_mag * 0.24 + intensity * 0.12),
        scale_world[1] * (0.20 + intensity * 0.18 + humidity_t * 0.08)
            + flow_dir[1] * scale_world[1] * (0.10 + flow_mag * 0.12)
            - pressure_bias.clamp(-1.0, 1.0) * scale_world[1] * 0.12,
        flow_dir[2] * scale_world[2] * (0.16 + flow_mag * 0.24 + intensity * 0.12),
    ];
    let tail = render_segment_mesh(
        [0.0, -scale_world[1] * 0.04, 0.0],
        tail_end,
        scale_world[0] * (0.05 + intensity * 0.03),
        8,
    );
    mesh_append(&mut mesh, &tail);
    let puff_count = (2.0 + intensity * 3.0 + humidity_t * 1.5)
        .round()
        .clamp(2.0, 5.0) as usize;
    for layer in 0..puff_count {
        let layer_t = (layer + 1) as f32 / (puff_count + 1) as f32;
        let mut puff = render_ellipsoid_mesh(
            [
                scale_world[0] * (0.14 + intensity * 0.08 + layer_t * 0.04),
                scale_world[1] * (0.14 + intensity * 0.10 + humidity_t * 0.04),
                scale_world[2] * (0.14 + intensity * 0.08 + layer_t * 0.04),
            ],
            12,
            7,
        );
        let angle = layer_t * std::f32::consts::TAU * 0.7 + intensity * 0.4;
        mesh_rotate_xyz(
            &mut puff,
            [flow_dir[2] * 0.16, angle * 0.12, -flow_dir[0] * 0.16],
        );
        mesh_translate(
            &mut puff,
            [
                tail_end[0] * layer_t * 0.82
                    + angle.cos() * scale_world[0] * (0.08 + (1.0 - layer_t) * 0.05),
                tail_end[1] * layer_t * 0.72 + scale_world[1] * (layer_t * 0.12 - 0.05),
                tail_end[2] * layer_t * 0.82
                    + angle.sin() * scale_world[2] * (0.08 + (1.0 - layer_t) * 0.05),
            ],
        );
        mesh_append(&mut mesh, &puff);
    }
    mesh
}

#[allow(dead_code)] // Used by terrarium_viewer binary
pub(super) fn render_stateful_fly_body_mesh(
    body_scale_world: [f32; 3],
    energy_t: f32,
    air_load: f32,
) -> TerrariumTriangleMeshRender {
    let mut mesh = TerrariumTriangleMeshRender::default();
    let thorax = render_ellipsoid_mesh(
        [
            body_scale_world[0] * (0.18 + air_load * 0.06),
            body_scale_world[1] * (0.36 + air_load * 0.10),
            body_scale_world[2] * 0.22,
        ],
        18,
        12,
    );
    mesh_append(&mut mesh, &thorax);

    let mut head = render_ellipsoid_mesh(
        [
            body_scale_world[0] * 0.12,
            body_scale_world[1] * 0.18,
            body_scale_world[2] * 0.12,
        ],
        14,
        9,
    );
    mesh_translate(
        &mut head,
        [0.0, body_scale_world[1] * 0.04, body_scale_world[2] * 0.20],
    );
    mesh_append(&mut mesh, &head);

    let mut abdomen = render_ellipsoid_mesh(
        [
            body_scale_world[0] * (0.16 + (1.0 - energy_t) * 0.03),
            body_scale_world[1] * (0.24 + (1.0 - energy_t) * 0.06),
            body_scale_world[2] * (0.30 + energy_t * 0.08),
        ],
        16,
        10,
    );
    mesh_rotate_xyz(&mut abdomen, [0.08 - air_load * 0.06, 0.0, 0.0]);
    mesh_translate(
        &mut abdomen,
        [
            0.0,
            -body_scale_world[1] * 0.02,
            -body_scale_world[2] * 0.20,
        ],
    );
    mesh_append(&mut mesh, &abdomen);

    mesh
}

pub(super) fn render_stateful_fly_wing_mesh(
    wing_scale_world: [f32; 3],
    wing_stroke: f32,
    wing_twist: f32,
    air_load: f32,
) -> TerrariumTriangleMeshRender {
    let span = wing_scale_world[0].max(1.0e-4);
    let chord = wing_scale_world[1].max(1.0e-4);
    let segments = 6usize;
    let mut mesh = TerrariumTriangleMeshRender::default();
    for segment in 0..=segments {
        let t = segment as f32 / segments as f32;
        let span_x = -span * 0.5 + span * t;
        let edge_curve = (t - 0.5).abs();
        let local_chord = chord * (0.92 - edge_curve * 0.24);
        let camber = (1.0 - edge_curve * 1.2).max(0.0) * (0.010 + air_load * 0.020);
        let twist = wing_twist * (t - 0.5) * 0.7 + wing_stroke * 0.05 * (1.0 - t);
        let leading = rotate_xyz([span_x, local_chord * 0.5, camber], [0.0, twist, 0.0]);
        let trailing = rotate_xyz([span_x, -local_chord * 0.5, -camber], [0.0, twist, 0.0]);
        mesh.positions.push(leading);
        mesh.normals.push(normalize3([camber, 0.0, 1.0]));
        mesh.uvs.push([t, 0.0]);
        mesh.positions.push(trailing);
        mesh.normals.push(normalize3([camber, 0.0, 1.0]));
        mesh.uvs.push([t, 1.0]);
    }
    for segment in 0..segments {
        let base = (segment * 2) as u32;
        mesh.indices
            .extend([base, base + 1, base + 3, base, base + 3, base + 2]);
    }
    mesh
}

pub(super) fn render_stateful_proboscis_mesh(length: f32, radius: f32) -> TerrariumTriangleMeshRender {
    render_segment_mesh(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, length.max(1.0e-4)],
        radius.max(1.0e-4),
        10,
    )
}

pub(super) fn render_stateful_fly_parts(
    fly_primary: u64,
    body: &BodyState,
    body_scale_world: [f32; 3],
    wing_scale_world: [f32; 3],
    energy_t: f32,
    air_load: f32,
    body_material: TerrariumPbrMaterialRender,
    wing_material: TerrariumPbrMaterialRender,
) -> Vec<TerrariumFlyPartRender> {
    let mut parts = Vec::with_capacity(6);
    parts.push(TerrariumFlyPartRender {
        render_id: terrarium_render_id(
            TERRARIUM_RENDER_ID_FLY,
            fly_primary,
            terrarium_fly_part_render_slot(TerrariumFlyPartKind::Thorax),
        ),
        render_fingerprint: 0,
        kind: TerrariumFlyPartKind::Thorax,
        mesh: render_ellipsoid_mesh(
            [
                body_scale_world[0] * (0.18 + air_load * 0.06),
                body_scale_world[1] * (0.36 + air_load * 0.10),
                body_scale_world[2] * 0.22,
            ],
            18,
            12,
        ),
        translation_local: [0.0, 0.0, 0.0],
        rotation_xyz_rad: [0.0, 0.0, 0.0],
        material: body_material.clone(),
    });
    parts.push(TerrariumFlyPartRender {
        render_id: terrarium_render_id(
            TERRARIUM_RENDER_ID_FLY,
            fly_primary,
            terrarium_fly_part_render_slot(TerrariumFlyPartKind::Head),
        ),
        render_fingerprint: 0,
        kind: TerrariumFlyPartKind::Head,
        mesh: render_ellipsoid_mesh(
            [
                body_scale_world[0] * 0.12,
                body_scale_world[1] * 0.18,
                body_scale_world[2] * 0.12,
            ],
            14,
            9,
        ),
        translation_local: [0.0, body_scale_world[1] * 0.04, body_scale_world[2] * 0.20],
        rotation_xyz_rad: [body.pitch.clamp(-0.14, 0.14) * 0.18, 0.0, 0.0],
        material: body_material.clone(),
    });
    parts.push(TerrariumFlyPartRender {
        render_id: terrarium_render_id(
            TERRARIUM_RENDER_ID_FLY,
            fly_primary,
            terrarium_fly_part_render_slot(TerrariumFlyPartKind::Abdomen),
        ),
        render_fingerprint: 0,
        kind: TerrariumFlyPartKind::Abdomen,
        mesh: render_ellipsoid_mesh(
            [
                body_scale_world[0] * (0.16 + (1.0 - energy_t) * 0.03),
                body_scale_world[1] * (0.24 + (1.0 - energy_t) * 0.06),
                body_scale_world[2] * (0.30 + energy_t * 0.08),
            ],
            16,
            10,
        ),
        translation_local: [
            0.0,
            -body_scale_world[1] * 0.02,
            -body_scale_world[2] * 0.20,
        ],
        rotation_xyz_rad: [
            0.08 - air_load * 0.06,
            0.0,
            body.roll.clamp(-0.5, 0.5) * 0.08,
        ],
        material: body_material.clone(),
    });
    if body.proboscis_extended {
        parts.push(TerrariumFlyPartRender {
            render_id: terrarium_render_id(
                TERRARIUM_RENDER_ID_FLY,
                fly_primary,
                terrarium_fly_part_render_slot(TerrariumFlyPartKind::Proboscis),
            ),
            render_fingerprint: 0,
            kind: TerrariumFlyPartKind::Proboscis,
            mesh: render_stateful_proboscis_mesh(
                body_scale_world[2] * (0.12 + (1.0 - energy_t) * 0.08),
                body_scale_world[0] * 0.022,
            ),
            translation_local: [0.0, -body_scale_world[1] * 0.02, body_scale_world[2] * 0.30],
            rotation_xyz_rad: [0.36 + body.pitch.clamp(-0.25, 0.25) * 0.35, 0.0, 0.0],
            material: body_material.clone(),
        });
    }
    let wing_mesh = render_stateful_fly_wing_mesh(
        wing_scale_world,
        body.wing_stroke,
        body.wing_twist,
        air_load,
    );
    parts.push(TerrariumFlyPartRender {
        render_id: terrarium_render_id(
            TERRARIUM_RENDER_ID_FLY,
            fly_primary,
            terrarium_fly_part_render_slot(TerrariumFlyPartKind::WingLeft),
        ),
        render_fingerprint: 0,
        kind: TerrariumFlyPartKind::WingLeft,
        mesh: wing_mesh.clone(),
        translation_local: [-0.02, 0.03, 0.06],
        rotation_xyz_rad: [body.wing_dihedral, 0.12 + body.wing_sweep, body.wing_twist],
        material: wing_material.clone(),
    });
    parts.push(TerrariumFlyPartRender {
        render_id: terrarium_render_id(
            TERRARIUM_RENDER_ID_FLY,
            fly_primary,
            terrarium_fly_part_render_slot(TerrariumFlyPartKind::WingRight),
        ),
        render_fingerprint: 0,
        kind: TerrariumFlyPartKind::WingRight,
        mesh: wing_mesh,
        translation_local: [-0.02, 0.03, -0.06],
        rotation_xyz_rad: [
            body.wing_dihedral,
            -(0.12 + body.wing_sweep),
            -body.wing_twist,
        ],
        material: wing_material,
    });
    parts
}

pub(super) fn render_stateful_microbe_body_mesh(
    base_radius: f32,
    represented_cells: f32,
    division_progress: f32,
    energy_state: f32,
    stress_state: f32,
    gene_catabolic: f32,
    gene_scavenging: f32,
    translation_support: f32,
) -> TerrariumTriangleMeshRender {
    let mut mesh = TerrariumTriangleMeshRender::default();
    let cell_t = (represented_cells / EXPLICIT_MICROBE_COHORT_CELLS.max(1.0e-3)).clamp(0.25, 2.2);
    let core = render_ellipsoid_mesh(
        [
            base_radius * (0.88 + gene_scavenging * 0.18 + cell_t * 0.06),
            base_radius * (0.72 + energy_state * 0.18 - stress_state * 0.08),
            base_radius * (0.94 + division_progress * 0.32 + gene_catabolic * 0.12),
        ],
        16,
        10,
    );
    mesh_append(&mut mesh, &core);

    let lobe_count = (3.0 + gene_catabolic * 2.0 + translation_support * 1.5 + cell_t * 0.8)
        .round()
        .clamp(3.0, 7.0) as usize;
    for lobe in 0..lobe_count {
        let phase = lobe as f32 / lobe_count as f32 * std::f32::consts::TAU + stress_state * 0.55;
        let (sin_phase, cos_phase) = phase.sin_cos();
        let mut lobe_mesh = render_ellipsoid_mesh(
            [
                base_radius * (0.26 + energy_state * 0.06 + gene_catabolic * 0.04),
                base_radius * (0.20 + translation_support * 0.08),
                base_radius * (0.22 + gene_scavenging * 0.06),
            ],
            12,
            8,
        );
        mesh_rotate_xyz(
            &mut lobe_mesh,
            [
                cos_phase * 0.18 * translation_support,
                phase * 0.28,
                -sin_phase * 0.14 * stress_state,
            ],
        );
        mesh_translate(
            &mut lobe_mesh,
            [
                cos_phase * base_radius * (0.38 + gene_scavenging * 0.08),
                base_radius * (-0.10 + energy_state * 0.16 + (lobe as f32 % 2.0) * 0.10),
                sin_phase * base_radius * (0.40 + gene_catabolic * 0.08),
            ],
        );
        mesh_append(&mut mesh, &lobe_mesh);
    }

    if division_progress > 0.05 {
        for direction in [-1.0f32, 1.0] {
            let mut daughter = render_ellipsoid_mesh(
                [
                    base_radius * (0.34 + division_progress * 0.06),
                    base_radius * (0.24 + energy_state * 0.06),
                    base_radius * (0.30 + division_progress * 0.12),
                ],
                12,
                8,
            );
            mesh_translate(
                &mut daughter,
                [
                    0.0,
                    base_radius * 0.04 * direction,
                    direction * base_radius * (0.34 + division_progress * 0.18),
                ],
            );
            mesh_append(&mut mesh, &daughter);
        }
    }

    mesh
}

pub(super) fn render_stateful_microbe_packet_mesh(
    base_radius: f32,
    packets: &[GenotypePacket],
) -> TerrariumTriangleMeshRender {
    let mut mesh = TerrariumTriangleMeshRender::default();
    if packets.is_empty() {
        return mesh;
    }

    let packet_count = packets.len().max(1);
    for (idx, packet) in packets.iter().enumerate() {
        let packet_t =
            (packet.represented_cells / MICROBIAL_PACKET_TARGET_CELLS.max(1.0e-3)).clamp(0.15, 1.8);
        let angle =
            idx as f32 / packet_count as f32 * std::f32::consts::TAU + packet.activity * 0.42;
        let ring = base_radius * (0.82 + packet.dormancy * 0.24 + (idx % 2) as f32 * 0.08);
        let (sin_angle, cos_angle) = angle.sin_cos();
        let mut packet_mesh = render_ellipsoid_mesh(
            [
                base_radius * (0.10 + packet_t * 0.06 + packet.activity * 0.04),
                base_radius * (0.08 + packet.reserve * 0.05),
                base_radius * (0.10 + packet_t * 0.05 + (1.0 - packet.damage) * 0.03),
            ],
            10,
            7,
        );
        mesh_rotate_xyz(
            &mut packet_mesh,
            [packet.damage * 0.35, angle * 0.18, -packet.dormancy * 0.28],
        );
        mesh_translate(
            &mut packet_mesh,
            [
                cos_angle * ring,
                base_radius * (-0.18 + packet.reserve * 0.26 - packet.damage * 0.18),
                sin_angle * ring,
            ],
        );
        mesh_append(&mut mesh, &packet_mesh);
    }

    mesh
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ExplicitMicrobeSiteLocalFrame {
    pub(super) center_x: f32,
    pub(super) center_y: f32,
    pub(super) center_z: f32,
    pub(super) pitch: f32,
}

pub(super) fn explicit_microbe_site_local_frame(
    site_reports: &[LocalChemistrySiteReport],
    body_radius: f32,
) -> Option<ExplicitMicrobeSiteLocalFrame> {
    if site_reports.is_empty() {
        return None;
    }

    let mut min_x = usize::MAX;
    let mut min_y = usize::MAX;
    let mut min_z = usize::MAX;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    let mut max_z = 0usize;
    for report in site_reports {
        min_x = min_x.min(report.site_x.saturating_sub(report.patch_radius));
        min_y = min_y.min(report.site_y.saturating_sub(report.patch_radius));
        min_z = min_z.min(report.site_z.saturating_sub(report.patch_radius));
        max_x = max_x.max(report.site_x + report.patch_radius);
        max_y = max_y.max(report.site_y + report.patch_radius);
        max_z = max_z.max(report.site_z + report.patch_radius);
    }

    let center_x = (min_x + max_x) as f32 * 0.5;
    let center_y = (min_y + max_y) as f32 * 0.5;
    let center_z = (min_z + max_z) as f32 * 0.5;
    let span_x = (max_x.saturating_sub(min_x) + 1).max(1) as f32;
    let span_y = (max_y.saturating_sub(min_y) + 1).max(1) as f32;
    let span_z = (max_z.saturating_sub(min_z) + 1).max(1) as f32;
    let max_span = span_x.max(span_y).max(span_z).max(1.0);
    let pitch = (body_radius * 2.2 / (max_span + 1.0)).max(body_radius * 0.14);

    Some(ExplicitMicrobeSiteLocalFrame {
        center_x,
        center_y,
        center_z,
        pitch,
    })
}

pub(super) fn explicit_microbe_site_local_translation(
    frame: ExplicitMicrobeSiteLocalFrame,
    report: LocalChemistrySiteReport,
) -> [f32; 3] {
    [
        (report.site_x as f32 - frame.center_x) * frame.pitch,
        (report.site_z as f32 - frame.center_z) * frame.pitch,
        (report.site_y as f32 - frame.center_y) * frame.pitch,
    ]
}

pub(super) fn render_explicit_microbe_packet_mesh(
    base_radius: f32,
    packet: &GenotypePacket,
) -> TerrariumTriangleMeshRender {
    let packet_t =
        (packet.represented_cells / MICROBIAL_PACKET_TARGET_CELLS.max(1.0e-3)).clamp(0.15, 1.8);
    let activity_t = packet.activity.clamp(0.0, 1.0);
    let reserve_t = packet.reserve.clamp(0.0, 1.0);
    let damage_t = packet.damage.clamp(0.0, 1.0);
    let dormancy_t = packet.dormancy.clamp(0.0, 1.0);
    let mut mesh = render_ellipsoid_mesh(
        [
            base_radius * (0.10 + packet_t * 0.06 + activity_t * 0.04),
            base_radius * (0.08 + reserve_t * 0.05),
            base_radius * (0.10 + packet_t * 0.05 + (1.0 - damage_t) * 0.03),
        ],
        10,
        7,
    );
    mesh_rotate_xyz(
        &mut mesh,
        [
            damage_t * 0.35,
            activity_t * 0.22 + reserve_t * 0.14,
            -dormancy_t * 0.28,
        ],
    );

    if packet.qualifies_for_promotion() {
        let mut bud = render_ellipsoid_mesh(
            [
                base_radius * (0.06 + reserve_t * 0.03),
                base_radius * (0.04 + activity_t * 0.02),
                base_radius * (0.05 + activity_t * 0.03),
            ],
            8,
            6,
        );
        mesh_translate(
            &mut bud,
            [
                base_radius * 0.12,
                base_radius * (0.10 + reserve_t * 0.06),
                base_radius * 0.10,
            ],
        );
        mesh_append(&mut mesh, &bud);
    }

    mesh
}

pub(super) fn render_explicit_microbe_packet_local_translation(
    frame: ExplicitMicrobeSiteLocalFrame,
    site_reports: &[LocalChemistrySiteReport],
    packet: &GenotypePacket,
    packet_idx: usize,
) -> [f32; 3] {
    if site_reports.is_empty() {
        return [0.0, 0.0, 0.0];
    }

    let activity_t = packet.activity.clamp(0.0, 1.0);
    let dormancy_t = packet.dormancy.clamp(0.0, 1.0);
    let reserve_t = packet.reserve.clamp(0.0, 1.0);
    let damage_t = packet.damage.clamp(0.0, 1.0);
    let glucose_need = (1.0 - reserve_t) * (0.55 + activity_t * 0.45);
    let oxygen_need = activity_t * (1.0 - dormancy_t);
    let repair_need = damage_t * (0.65 + activity_t * 0.35);
    let biosynthesis_need = (1.0 - reserve_t * 0.5) * (1.0 - dormancy_t * 0.35);

    let mut weighted = [0.0; 3];
    let mut total_weight = 0.0;
    let mut fallback_translation = [0.0; 3];
    let mut best_weight = f32::NEG_INFINITY;
    for report in site_reports.iter().copied() {
        let translation = explicit_microbe_site_local_translation(frame, report);
        let support = report.atp_support.clamp(0.0, 1.0) * (0.18 + glucose_need * 0.22)
            + report.translation_support.clamp(0.0, 1.0)
                * (0.20 + activity_t * 0.18 + biosynthesis_need * 0.10)
            + report.mean_glucose.clamp(0.0, 1.0) * glucose_need * 0.18
            + report.mean_oxygen.clamp(0.0, 1.0) * oxygen_need * 0.18
            + report.assembly_component_availability.clamp(0.0, 1.0) * biosynthesis_need * 0.18
            + report.assembly_occupancy.clamp(0.0, 1.0) * 0.08
            + report.assembly_stability.clamp(0.0, 1.0) * (0.06 + repair_need * 0.12)
            + report.demand_satisfaction.clamp(0.0, 1.0) * 0.08
            + report.localization_score.clamp(0.0, 1.0) * 0.14;
        let penalty = report.crowding_penalty.max(0.0) * 0.12
            + report.byproduct_load.clamp(0.0, 1.0) * repair_need * 0.10;
        let weight = (support - penalty).max(0.02);
        if weight > best_weight {
            best_weight = weight;
            fallback_translation = translation;
        }
        weighted[0] += translation[0] * weight;
        weighted[1] += translation[1] * weight;
        weighted[2] += translation[2] * weight;
        total_weight += weight;
    }

    let mut translation = if total_weight > 1.0e-6 {
        [
            weighted[0] / total_weight,
            weighted[1] / total_weight,
            weighted[2] / total_weight,
        ]
    } else {
        fallback_translation
    };
    let packet_t = (packet.represented_cells / MICROBIAL_PACKET_TARGET_CELLS.max(1.0e-3)).sqrt();
    let phase_seed = packet.activity * 3.1
        + packet.reserve * 5.7
        + packet.damage * 7.9
        + packet.dormancy * 11.3
        + packet_idx as f32 * 0.91;
    let (sin_phase, cos_phase) = (phase_seed.fract() * std::f32::consts::TAU).sin_cos();
    let separation = frame.pitch * (0.12 + packet_t.clamp(0.15, 1.8) * 0.06);
    translation[0] += cos_phase * separation;
    translation[1] += frame.pitch * ((reserve_t - damage_t) * 0.22 - dormancy_t * 0.08);
    translation[2] += sin_phase * separation;
    translation
}
