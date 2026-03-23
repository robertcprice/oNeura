use crate::drosophila::{BodyState, DrosophilaSim};
use crate::terrarium_world::{
    fly_body_state_from_world_translation, fly_translation_world_from_body, raycast_scene_internal,
    TerrariumRaycastBvhNode, TerrariumRaycastSurface, TerrariumWorldConfig,
};

const FLY_CONTACT_COLLIDER_RADIUS_WORLD: f32 = 0.09;
const FLY_CONTACT_SURFACE_CLEARANCE_WORLD: f32 = 0.08;
const FLY_CONTACT_SWEEP_EPSILON_WORLD: f32 = 0.012;
const FLY_CONTACT_SUPPORT_CAST_HEIGHT_WORLD: f32 = 0.84;
const FLY_CONTACT_LANDING_NORMAL_Y: f32 = 0.55;
const FLY_CONTACT_IMPACT_DAMPING: f32 = 0.46;
const FLY_CONTACT_VERTICAL_DAMPING: f32 = 0.24;
const FLY_CONTACT_SURFACE_SETTLE_WORLD: f32 = 0.03;
const FLY_CONTACT_SUPPORT_COUPLING_ITERATIONS: usize = 2;
const FLY_CONTACT_SIDE_PROBE_ITERATIONS: usize = 2;
const FLY_CONTACT_SIDE_PROBE_DIRECTIONS: usize = 10;
const FLY_CONTACT_SIDE_MAX_NORMAL_Y: f32 = 0.45;
const FLY_CONTACT_SIDE_PUSH_EPSILON_WORLD: f32 = 0.003;
const FLY_CONTACT_SIDE_PROBE_HEIGHT_OFFSETS_WORLD: [f32; 3] = [
    -FLY_CONTACT_COLLIDER_RADIUS_WORLD * 0.34,
    0.0,
    FLY_CONTACT_COLLIDER_RADIUS_WORLD * 0.34,
];
const FLY_PAIRWISE_CONTACT_ITERATIONS: usize = 6;
const FLY_PAIRWISE_CONTACT_SPEED_DAMPING: f32 = 0.22;
const FLY_PAIRWISE_CONTACT_VERTICAL_DAMPING: f32 = 0.18;

fn normalize3(value: [f32; 3]) -> [f32; 3] {
    let length_sq = dot3(value, value);
    if length_sq <= 1.0e-12 {
        [0.0, 1.0, 0.0]
    } else {
        mul3(value, length_sq.sqrt().recip())
    }
}

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn mul3(value: [f32; 3], scalar: f32) -> [f32; 3] {
    [value[0] * scalar, value[1] * scalar, value[2] * scalar]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn resolve_fly_side_contacts_with_scene(
    resolved_world: &mut [f32; 3],
    collision_surfaces: &[TerrariumRaycastSurface],
    collision_nodes: &[TerrariumRaycastBvhNode],
    collision_root: Option<usize>,
    excluded_class_mask: u16,
    impact: &mut f32,
    contact_normal_world: &mut [f32; 3],
) -> bool {
    if collision_root.is_none() {
        return false;
    }

    let mut any_side_contact = false;
    for _ in 0..FLY_CONTACT_SIDE_PROBE_ITERATIONS {
        let start_world = *resolved_world;
        let mut best_push_world = [0.0, 0.0, 0.0];
        let mut best_penetration = 0.0f32;
        let target_clearance =
            FLY_CONTACT_COLLIDER_RADIUS_WORLD + FLY_CONTACT_SIDE_PUSH_EPSILON_WORLD;
        for height_offset in FLY_CONTACT_SIDE_PROBE_HEIGHT_OFFSETS_WORLD {
            let probe_center = add3(start_world, [0.0, height_offset, 0.0]);
            for dir_idx in 0..FLY_CONTACT_SIDE_PROBE_DIRECTIONS {
                let angle = dir_idx as f32 * std::f32::consts::TAU
                    / FLY_CONTACT_SIDE_PROBE_DIRECTIONS as f32;
                let (sin_angle, cos_angle) = angle.sin_cos();
                let direction_world = [cos_angle, 0.0, sin_angle];
                let Some(hit) = raycast_scene_internal(
                    collision_surfaces,
                    collision_nodes,
                    collision_root,
                    probe_center,
                    direction_world,
                    false,
                    true,
                    excluded_class_mask,
                ) else {
                    continue;
                };
                if hit.normal_world[1].abs() > FLY_CONTACT_SIDE_MAX_NORMAL_Y {
                    continue;
                }
                if hit.distance >= target_clearance {
                    continue;
                }

                let penetration = (target_clearance - hit.distance).max(0.0);
                if penetration > best_penetration {
                    best_penetration = penetration;
                    best_push_world = mul3(direction_world, penetration);
                }
                *impact = (*impact)
                    .max((penetration / target_clearance.max(1.0e-4)).clamp(0.0, 1.0) * 0.55);
                *contact_normal_world = hit.normal_world;
            }
        }

        if best_penetration <= 0.0 {
            break;
        }

        let mut push_world = best_push_world;
        let push_len = dot3(push_world, push_world).sqrt();
        let max_push = target_clearance;
        if push_len > max_push && push_len > 1.0e-6 {
            push_world = mul3(push_world, max_push / push_len);
        }
        resolved_world[0] += push_world[0];
        resolved_world[2] += push_world[2];
        any_side_contact = true;
    }

    any_side_contact
}

pub(crate) fn resolve_fly_contacts_with_scene(
    config: &TerrariumWorldConfig,
    fly: &mut DrosophilaSim,
    previous_body: &BodyState,
    terrain: &[f32],
    terrain_min: f32,
    terrain_inv: f32,
    collision_surfaces: &[TerrariumRaycastSurface],
    collision_nodes: &[TerrariumRaycastBvhNode],
    collision_root: Option<usize>,
    excluded_class_mask: u16,
) {
    if collision_root.is_none() {
        return;
    }

    let current_body = fly.body_state().clone();
    let (previous_world, _) =
        fly_translation_world_from_body(config, terrain, terrain_min, terrain_inv, previous_body);
    let (mut resolved_world, _) =
        fly_translation_world_from_body(config, terrain, terrain_min, terrain_inv, &current_body);
    let sweep_world = sub3(resolved_world, previous_world);
    let sweep_distance = dot3(sweep_world, sweep_world).sqrt();
    let mut impact = 0.0f32;
    let mut contact_normal_world = [0.0, 1.0, 0.0];

    if sweep_distance > 1.0e-5 {
        let sweep_dir = mul3(sweep_world, 1.0 / sweep_distance);
        if let Some(hit) = raycast_scene_internal(
            collision_surfaces,
            collision_nodes,
            collision_root,
            previous_world,
            sweep_dir,
            false,
            true,
            excluded_class_mask,
        ) {
            let collision_distance = sweep_distance
                + FLY_CONTACT_COLLIDER_RADIUS_WORLD
                + FLY_CONTACT_SWEEP_EPSILON_WORLD;
            if hit.distance <= collision_distance {
                impact = dot3(sweep_dir, hit.normal_world).abs().clamp(0.0, 1.0);
                contact_normal_world = hit.normal_world;
                let travel_distance = (hit.distance - FLY_CONTACT_SWEEP_EPSILON_WORLD).max(0.0);
                let travel_world = mul3(sweep_dir, travel_distance.min(sweep_distance));
                let travel_end = add3(previous_world, travel_world);
                let remaining_world = sub3(resolved_world, travel_end);
                let mut slide_world = sub3(
                    remaining_world,
                    mul3(hit.normal_world, dot3(remaining_world, hit.normal_world)),
                );
                let slide_len_sq = dot3(slide_world, slide_world);
                let contact_offset =
                    FLY_CONTACT_COLLIDER_RADIUS_WORLD + FLY_CONTACT_SWEEP_EPSILON_WORLD;
                resolved_world = add3(hit.position_world, mul3(hit.normal_world, contact_offset));
                if slide_len_sq > 1.0e-6 {
                    let slide_len = slide_len_sq.sqrt();
                    let slide_dir = mul3(slide_world, 1.0 / slide_len);
                    let slide_origin = add3(
                        resolved_world,
                        mul3(hit.normal_world, FLY_CONTACT_SWEEP_EPSILON_WORLD),
                    );
                    if let Some(slide_hit) = raycast_scene_internal(
                        collision_surfaces,
                        collision_nodes,
                        collision_root,
                        slide_origin,
                        slide_dir,
                        false,
                        true,
                        excluded_class_mask,
                    ) {
                        if slide_hit.distance
                            <= slide_len
                                + FLY_CONTACT_COLLIDER_RADIUS_WORLD
                                + FLY_CONTACT_SWEEP_EPSILON_WORLD
                        {
                            resolved_world = add3(
                                slide_hit.position_world,
                                mul3(
                                    slide_hit.normal_world,
                                    FLY_CONTACT_COLLIDER_RADIUS_WORLD
                                        + FLY_CONTACT_SWEEP_EPSILON_WORLD,
                                ),
                            );
                            contact_normal_world = slide_hit.normal_world;
                            impact = impact.max(dot3(slide_dir, slide_hit.normal_world).abs());
                            slide_world = [0.0, 0.0, 0.0];
                        }
                    }
                    resolved_world = add3(resolved_world, slide_world);
                }
            }
        }
    }

    let support_origin = add3(
        resolved_world,
        [0.0, FLY_CONTACT_SUPPORT_CAST_HEIGHT_WORLD, 0.0],
    );
    let support_hit = raycast_scene_internal(
        collision_surfaces,
        collision_nodes,
        collision_root,
        support_origin,
        [0.0, -1.0, 0.0],
        false,
        true,
        excluded_class_mask,
    );
    let mut landed = false;
    let mut support_height_world = resolved_world[1];
    if let Some(hit) = support_hit {
        contact_normal_world = hit.normal_world;
        support_height_world = hit.position_world[1] + FLY_CONTACT_SURFACE_CLEARANCE_WORLD;
        if resolved_world[1] < support_height_world {
            resolved_world[1] = support_height_world;
            impact = impact.max(0.35);
        }
        if !current_body.is_flying {
            resolved_world[1] = support_height_world;
            landed = true;
        } else if hit.normal_world[1] >= FLY_CONTACT_LANDING_NORMAL_Y
            && current_body.vertical_velocity <= 0.0
            && resolved_world[1] <= support_height_world + FLY_CONTACT_SURFACE_SETTLE_WORLD
        {
            resolved_world[1] = support_height_world;
            landed = true;
        }
    }

    for _ in 0..FLY_CONTACT_SUPPORT_COUPLING_ITERATIONS {
        let had_side_contact = resolve_fly_side_contacts_with_scene(
            &mut resolved_world,
            collision_surfaces,
            collision_nodes,
            collision_root,
            excluded_class_mask,
            &mut impact,
            &mut contact_normal_world,
        );
        let support_origin = add3(
            resolved_world,
            [0.0, FLY_CONTACT_SUPPORT_CAST_HEIGHT_WORLD, 0.0],
        );
        let mut support_adjusted = false;
        if let Some(hit) = raycast_scene_internal(
            collision_surfaces,
            collision_nodes,
            collision_root,
            support_origin,
            [0.0, -1.0, 0.0],
            false,
            true,
            excluded_class_mask,
        ) {
            contact_normal_world = hit.normal_world;
            support_height_world = hit.position_world[1] + FLY_CONTACT_SURFACE_CLEARANCE_WORLD;
            if resolved_world[1] < support_height_world {
                resolved_world[1] = support_height_world;
                impact = impact.max(0.35);
                support_adjusted = true;
            }
            if landed || !current_body.is_flying {
                if (resolved_world[1] - support_height_world).abs() > 1.0e-5 {
                    support_adjusted = true;
                }
                resolved_world[1] = support_height_world;
                landed = true;
            }
        }
        if !had_side_contact && !support_adjusted {
            break;
        }
    }

    let (resolved_x, resolved_y, mut resolved_z, _) = fly_body_state_from_world_translation(
        config,
        terrain,
        terrain_min,
        terrain_inv,
        resolved_world,
        current_body.is_flying && !landed,
    );
    if landed {
        let (_, _, support_z, _) = fly_body_state_from_world_translation(
            config,
            terrain,
            terrain_min,
            terrain_inv,
            [resolved_world[0], support_height_world, resolved_world[2]],
            false,
        );
        resolved_z = support_z;
    }

    let body = fly.body_state_mut();
    body.x = resolved_x;
    body.y = resolved_y;
    body.z = resolved_z;
    if landed {
        body.is_flying = false;
        body.vertical_velocity = 0.0;
        body.pitch *= 0.18;
        body.roll *= 0.72;
        body.speed *= 1.0 - impact * FLY_CONTACT_IMPACT_DAMPING;
    } else if impact > 0.0 {
        body.vertical_velocity *= 1.0 - impact * FLY_CONTACT_VERTICAL_DAMPING;
        body.pitch *= 1.0 - impact * 0.62;
        body.roll += contact_normal_world[0].clamp(-0.2, 0.2) * impact * 0.08;
        body.yaw_slip += contact_normal_world[2].clamp(-0.2, 0.2) * impact * 0.06;
        body.speed *= 1.0 - impact * FLY_CONTACT_IMPACT_DAMPING;
    }
}

pub(crate) fn pairwise_fly_contact_radius_world(body: &BodyState) -> f32 {
    let flight_scale = if body.is_flying { 1.08 } else { 0.94 };
    FLY_CONTACT_COLLIDER_RADIUS_WORLD * (flight_scale + body.air_load.clamp(0.0, 1.0) * 0.14)
}

pub(crate) fn resolve_pairwise_fly_contacts(
    config: &TerrariumWorldConfig,
    flies: &mut [DrosophilaSim],
    terrain: &[f32],
    terrain_min: f32,
    terrain_inv: f32,
    collision_surfaces: &[TerrariumRaycastSurface],
    collision_nodes: &[TerrariumRaycastBvhNode],
    collision_root: Option<usize>,
    excluded_class_mask: u16,
) {
    if flies.len() < 2 {
        return;
    }

    for _ in 0..FLY_PAIRWISE_CONTACT_ITERATIONS {
        let iteration_start_bodies = flies
            .iter()
            .map(|fly| fly.body_state().clone())
            .collect::<Vec<_>>();
        let mut any_overlap = false;
        for left_idx in 0..flies.len() {
            for right_idx in (left_idx + 1)..flies.len() {
                let (left_slice, right_slice) = flies.split_at_mut(right_idx);
                let left_fly = &mut left_slice[left_idx];
                let right_fly = &mut right_slice[0];
                let left_body = left_fly.body_state().clone();
                let right_body = right_fly.body_state().clone();
                let (mut left_world, _) = fly_translation_world_from_body(
                    config,
                    terrain,
                    terrain_min,
                    terrain_inv,
                    &left_body,
                );
                let (mut right_world, _) = fly_translation_world_from_body(
                    config,
                    terrain,
                    terrain_min,
                    terrain_inv,
                    &right_body,
                );
                let left_radius = pairwise_fly_contact_radius_world(&left_body);
                let right_radius = pairwise_fly_contact_radius_world(&right_body);
                let min_distance = left_radius + right_radius;
                let delta_world = sub3(right_world, left_world);
                let horizontal_delta = [delta_world[0], 0.0, delta_world[2]];
                let horizontal_distance = dot3(horizontal_delta, horizontal_delta).sqrt();
                let (separation_dir, separation_distance) = if horizontal_distance > 1.0e-4 {
                    (
                        mul3(horizontal_delta, 1.0 / horizontal_distance),
                        horizontal_distance,
                    )
                } else {
                    let fallback = normalize3([
                        left_body.heading.cos() - right_body.heading.cos(),
                        0.0,
                        left_body.heading.sin() - right_body.heading.sin(),
                    ]);
                    if fallback == [0.0, 1.0, 0.0] {
                        ([1.0, 0.0, 0.0], 0.0)
                    } else {
                        ([fallback[0], 0.0, fallback[2]], 0.0)
                    }
                };
                if separation_distance >= min_distance {
                    continue;
                }

                any_overlap = true;
                let overlap = (min_distance - separation_distance).max(0.0);
                let push_scale = overlap * 0.5 + 1.0e-3;
                left_world = sub3(left_world, mul3(separation_dir, push_scale));
                right_world = add3(right_world, mul3(separation_dir, push_scale));

                let (left_x, left_y, left_z, _) = fly_body_state_from_world_translation(
                    config,
                    terrain,
                    terrain_min,
                    terrain_inv,
                    left_world,
                    left_body.is_flying,
                );
                let (right_x, right_y, right_z, _) = fly_body_state_from_world_translation(
                    config,
                    terrain,
                    terrain_min,
                    terrain_inv,
                    right_world,
                    right_body.is_flying,
                );
                let impact_t = (overlap / min_distance.max(1.0e-4)).clamp(0.0, 1.0);
                {
                    let body = left_fly.body_state_mut();
                    body.x = left_x;
                    body.y = left_y;
                    body.z = left_z;
                    body.speed *= 1.0 - impact_t * FLY_PAIRWISE_CONTACT_SPEED_DAMPING;
                    body.vertical_velocity *=
                        1.0 - impact_t * FLY_PAIRWISE_CONTACT_VERTICAL_DAMPING;
                    body.yaw_slip -= separation_dir[0] * impact_t * 0.05;
                }
                {
                    let body = right_fly.body_state_mut();
                    body.x = right_x;
                    body.y = right_y;
                    body.z = right_z;
                    body.speed *= 1.0 - impact_t * FLY_PAIRWISE_CONTACT_SPEED_DAMPING;
                    body.vertical_velocity *=
                        1.0 - impact_t * FLY_PAIRWISE_CONTACT_VERTICAL_DAMPING;
                    body.yaw_slip += separation_dir[0] * impact_t * 0.05;
                }
            }
        }
        if !any_overlap {
            break;
        }

        for (fly, previous_body) in flies.iter_mut().zip(iteration_start_bodies.iter()) {
            resolve_fly_contacts_with_scene(
                config,
                fly,
                previous_body,
                terrain,
                terrain_min,
                terrain_inv,
                collision_surfaces,
                collision_nodes,
                collision_root,
                excluded_class_mask,
            );
        }
    }
}
