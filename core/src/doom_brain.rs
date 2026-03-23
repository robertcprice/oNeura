//! DoomBrainSim -- Doom FPS brain with embodied/disembodied modes.
//!
//! Runs a biophysical brain in a Doom-style raycasting environment. Two
//! operational modes:
//!
//! - **Embodied**: Full retina pipeline (pixel buffer -> rod/cone -> bipolar ->
//!   RGC -> V1 -> V2 -> motor). Not yet implemented; falls back to disembodied
//!   encoding with higher ray resolution.
//! - **Disembodied**: 8 raycast distances injected directly into V1 cortical
//!   populations as Weber-Fechner-scaled external current.
//!
//! # Brain Architecture
//!
//! The Doom brain consists of 11 named regions wired with biologically
//! plausible connectivity:
//!
//! | Region        | Function                          | Archetype       |
//! |---------------|-----------------------------------|-----------------|
//! | V1            | Primary visual cortex             | Pyramidal       |
//! | V2            | Secondary visual cortex           | Pyramidal       |
//! | turn_left     | Motor: left turn command           | Pyramidal       |
//! | turn_right    | Motor: right turn command          | Pyramidal       |
//! | move_forward  | Motor: forward locomotion          | Pyramidal       |
//! | shoot         | Motor: fire weapon                 | Pyramidal       |
//! | VTA           | Ventral tegmental area (DA)        | DopaminergicSN  |
//! | LC            | Locus coeruleus (NE)               | Serotonergic    |
//! | prefrontal    | Threat assessment, decision making | Pyramidal       |
//! | hippocampus   | Spatial memory                     | Pyramidal       |
//! | amygdala      | Fear/threat response               | Pyramidal       |
//!
//! # Neuromodulation
//!
//! - **Damage**: Nociceptor current to amygdala + NE burst to LC neurons.
//! - **Health pickup**: DA burst to VTA neurons.
//! - **FEP protocol**: Structured pulsed stimulation on HIT (low entropy),
//!   random 30% noise on MISS (high entropy).
//!
//! # Experiments
//!
//! 1. **Threat avoidance** -- Learn to avoid damage zones (FEP protocol).
//! 2. **Navigation** -- Explore and map the environment (grid cell coverage).
//! 3. **Combat** -- Aim and shoot at stationary targets.
//!
//! # Integration
//!
//! Uses the same `MolecularBrain` infrastructure as `RegionalBrain`, running
//! the full HH + receptor binding + STDP pipeline on Metal GPU (macOS) or
//! CPU fallback (all platforms).

use crate::network::MolecularBrain;
use crate::types::*;
use rand::prelude::*;
use rand::rngs::StdRng;

// =========================================================================
// Doom operating mode
// =========================================================================

/// Doom brain operating mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DoomMode {
    /// Full retina pipeline with pixel processing (high ray count).
    /// Currently uses Weber-Fechner encoding of 64 rays into V1.
    Embodied,
    /// Direct raycast distances (8 rays) to V1 cortical populations.
    Disembodied,
}

// =========================================================================
// Doom engine -- lightweight raycasting environment
// =========================================================================

/// Entity types in the Doom world.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
pub enum EntityType {
    /// Enemy that can damage the player.
    Enemy = 1,
    /// Health pickup that restores HP.
    HealthPack = 2,
    /// Enemy projectile in flight.
    Projectile = 3,
}

/// AI state for moving enemies.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
pub enum EnemyAI {
    /// Random walk (no player detected).
    Wander = 0,
    /// Chasing the player (within detection range).
    Chase = 1,
    /// Attacking (within attack range).
    Attack = 2,
}

/// A discrete entity in the Doom world (enemy, health pack, projectile).
#[derive(Clone, Debug)]
pub struct DoomEntity {
    pub x: f32,
    pub y: f32,
    pub hp: f32,
    pub entity_type: EntityType,
    /// Current movement angle (radians) for enemies.
    pub angle: f32,
    /// Movement speed (units/step).
    pub speed: f32,
    /// AI state for enemies.
    pub ai_state: EnemyAI,
    /// Detection range for player awareness.
    pub detect_range: f32,
    /// Attack range for firing projectiles.
    pub attack_range: f32,
    /// Cooldown timer for attacks (steps remaining).
    pub attack_cooldown: u32,
    /// Whether this entity is alive/active.
    pub alive: bool,
}

/// Wall segment defined by two endpoints.
#[derive(Clone, Debug)]
pub struct WallSegment {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

/// Lightweight Doom engine providing raycasting and game state.
///
/// Performs DDA raycasting against axis-aligned and arbitrary wall segments.
/// For fewer than 64 rays the CPU implementation is faster than a GPU kernel
/// due to the branchy, serial nature of BSP-style ray traversal.
pub struct DoomEngine {
    pub player_x: f32,
    pub player_y: f32,
    pub player_angle: f32,
    pub player_hp: f32,
    pub n_rays: u32,
    pub ray_distances: Vec<f32>,
    pub entities: Vec<DoomEntity>,
    pub walls: Vec<WallSegment>,
    /// Field of view in radians (default: 60 degrees = pi/3).
    pub fov: f32,
    /// Map bounds (width x height). Player is clamped to [0.5, bound-0.5].
    pub map_width: f32,
    pub map_height: f32,
}

impl DoomEngine {
    /// Create a new Doom engine with the specified number of rays.
    pub fn new(n_rays: u32) -> Self {
        Self {
            player_x: 5.0,
            player_y: 5.0,
            player_angle: 0.0,
            player_hp: 100.0,
            n_rays,
            ray_distances: vec![100.0; n_rays as usize],
            entities: Vec::new(),
            walls: Vec::new(),
            fov: std::f32::consts::PI / 3.0,
            map_width: 20.0,
            map_height: 20.0,
        }
    }

    /// Cast all rays from the player's position and update `ray_distances`.
    ///
    /// Each ray sweeps across the FOV, testing intersection with every wall
    /// segment. The closest hit distance is stored. A maximum distance of
    /// 100.0 units is used when no wall is hit (open space).
    pub fn raycast(&mut self) {
        let max_dist = 100.0f32;
        let half_fov = self.fov / 2.0;

        for i in 0..self.n_rays {
            let frac = i as f32 / self.n_rays.max(1) as f32;
            let angle = self.player_angle - half_fov + self.fov * frac;
            let dx = angle.cos();
            let dy = angle.sin();

            let mut min_dist = max_dist;
            for wall in &self.walls {
                let d = ray_segment_intersect(
                    self.player_x,
                    self.player_y,
                    dx,
                    dy,
                    wall.x1,
                    wall.y1,
                    wall.x2,
                    wall.y2,
                );
                if d > 0.0 && d < min_dist {
                    min_dist = d;
                }
            }
            self.ray_distances[i as usize] = min_dist;
        }
    }

    /// Add a wall segment to the map.
    pub fn add_wall(&mut self, x1: f32, y1: f32, x2: f32, y2: f32) {
        self.walls.push(WallSegment { x1, y1, x2, y2 });
    }

    /// Add a rectangular room (4 walls).
    pub fn add_room(&mut self, x: f32, y: f32, w: f32, h: f32) {
        self.add_wall(x, y, x + w, y); // bottom
        self.add_wall(x + w, y, x + w, y + h); // right
        self.add_wall(x + w, y + h, x, y + h); // top
        self.add_wall(x, y + h, x, y); // left
    }

    /// Add an entity (enemy, health pack).
    pub fn add_entity(&mut self, x: f32, y: f32, hp: f32, entity_type: EntityType) {
        self.entities.push(DoomEntity {
            x,
            y,
            hp,
            entity_type,
            angle: 0.0,
            speed: 0.0,
            ai_state: EnemyAI::Wander,
            detect_range: 8.0,
            attack_range: 5.0,
            attack_cooldown: 0,
            alive: true,
        });
    }

    /// Update player position based on motor output. Clamps to map bounds.
    pub fn update_player(&mut self, turn: f32, forward: f32, dt: f32) {
        self.player_angle += turn * dt;
        // Wrap angle to [-pi, pi)
        while self.player_angle > std::f32::consts::PI {
            self.player_angle -= 2.0 * std::f32::consts::PI;
        }
        while self.player_angle < -std::f32::consts::PI {
            self.player_angle += 2.0 * std::f32::consts::PI;
        }

        let new_x = self.player_x + self.player_angle.cos() * forward * dt;
        let new_y = self.player_y + self.player_angle.sin() * forward * dt;
        self.player_x = new_x.clamp(0.5, self.map_width - 0.5);
        self.player_y = new_y.clamp(0.5, self.map_height - 0.5);
    }

    /// Reset player to starting position and full health.
    pub fn reset_player(&mut self, x: f32, y: f32, angle: f32) {
        self.player_x = x;
        self.player_y = y;
        self.player_angle = angle;
        self.player_hp = 100.0;
    }

    /// Check if the player is within `radius` of a given point.
    pub fn player_in_radius(&self, cx: f32, cy: f32, radius: f32) -> bool {
        let dx = self.player_x - cx;
        let dy = self.player_y - cy;
        dx * dx + dy * dy < radius * radius
    }

    /// Compute the angle from the player to a world point.
    pub fn angle_to(&self, x: f32, y: f32) -> f32 {
        let dx = x - self.player_x;
        let dy = y - self.player_y;
        dy.atan2(dx)
    }

    /// Compute the angular difference between player heading and a target.
    /// Returns a value in [-pi, pi].
    pub fn angular_error_to(&self, x: f32, y: f32) -> f32 {
        let target = self.angle_to(x, y);
        let mut diff = target - self.player_angle;
        while diff > std::f32::consts::PI {
            diff -= 2.0 * std::f32::consts::PI;
        }
        while diff < -std::f32::consts::PI {
            diff += 2.0 * std::f32::consts::PI;
        }
        diff
    }

    /// Add a moving enemy with AI parameters.
    pub fn add_enemy(
        &mut self,
        x: f32,
        y: f32,
        hp: f32,
        speed: f32,
        detect_range: f32,
        attack_range: f32,
    ) {
        self.entities.push(DoomEntity {
            x,
            y,
            hp,
            entity_type: EntityType::Enemy,
            angle: 0.0,
            speed,
            ai_state: EnemyAI::Wander,
            detect_range,
            attack_range,
            attack_cooldown: 0,
            alive: true,
        });
    }

    /// Update all entity AI: wander, chase, attack, projectile movement.
    ///
    /// Enemy AI state machine:
    /// - **Wander**: Random walk. Transitions to Chase when player within detect_range.
    /// - **Chase**: Move toward player. Transitions to Attack when within attack_range.
    /// - **Attack**: Fire projectile at player, then cooldown before next shot.
    ///
    /// Returns damage dealt to player this step.
    pub fn update_entities(&mut self, rng: &mut StdRng, dt: f32) -> f32 {
        let px = self.player_x;
        let py = self.player_y;
        let mut damage = 0.0f32;

        // Collect projectile movements + spawns separately to avoid borrow issues
        let mut new_projectiles: Vec<DoomEntity> = Vec::new();

        for ent in self.entities.iter_mut() {
            if !ent.alive {
                continue;
            }

            match ent.entity_type {
                EntityType::Enemy => {
                    let dx = px - ent.x;
                    let dy = py - ent.y;
                    let dist = (dx * dx + dy * dy).sqrt();

                    // State transitions
                    ent.ai_state = if dist < ent.attack_range {
                        EnemyAI::Attack
                    } else if dist < ent.detect_range {
                        EnemyAI::Chase
                    } else {
                        EnemyAI::Wander
                    };

                    match ent.ai_state {
                        EnemyAI::Wander => {
                            // Random walk: occasionally change direction
                            if rng.gen::<f32>() < 0.05 {
                                ent.angle += rng.gen_range(-0.5f32..0.5);
                            }
                            ent.x += ent.angle.cos() * ent.speed * dt * 0.5;
                            ent.y += ent.angle.sin() * ent.speed * dt * 0.5;
                        }
                        EnemyAI::Chase => {
                            // Move toward player
                            let target_angle = dy.atan2(dx);
                            ent.angle = target_angle;
                            ent.x += ent.angle.cos() * ent.speed * dt;
                            ent.y += ent.angle.sin() * ent.speed * dt;
                        }
                        EnemyAI::Attack => {
                            // Fire projectile when cooldown expires
                            if ent.attack_cooldown == 0 {
                                let target_angle = dy.atan2(dx);
                                new_projectiles.push(DoomEntity {
                                    x: ent.x,
                                    y: ent.y,
                                    hp: 1.0,
                                    entity_type: EntityType::Projectile,
                                    angle: target_angle,
                                    speed: 3.0,                // projectile speed
                                    ai_state: EnemyAI::Wander, // unused for projectiles
                                    detect_range: 0.0,
                                    attack_range: 0.0,
                                    attack_cooldown: 0,
                                    alive: true,
                                });
                                ent.attack_cooldown = 50; // 50 steps between shots
                            }
                        }
                    }

                    if ent.attack_cooldown > 0 {
                        ent.attack_cooldown -= 1;
                    }

                    // Clamp to map
                    ent.x = ent.x.clamp(0.5, self.map_width - 0.5);
                    ent.y = ent.y.clamp(0.5, self.map_height - 0.5);
                }
                EntityType::Projectile => {
                    // Move projectile
                    ent.x += ent.angle.cos() * ent.speed * dt;
                    ent.y += ent.angle.sin() * ent.speed * dt;

                    // Check collision with player
                    let dx = px - ent.x;
                    let dy = py - ent.y;
                    if dx * dx + dy * dy < 1.0 {
                        damage += 5.0; // projectile damage
                        ent.alive = false;
                    }

                    // Remove if out of bounds
                    if ent.x < 0.0
                        || ent.x > self.map_width
                        || ent.y < 0.0
                        || ent.y > self.map_height
                    {
                        ent.alive = false;
                    }
                }
                EntityType::HealthPack => {
                    // Static: check pickup
                    let dx = px - ent.x;
                    let dy = py - ent.y;
                    if dx * dx + dy * dy < 1.5 {
                        self.player_hp = (self.player_hp + 25.0).min(100.0);
                        ent.alive = false;
                    }
                }
            }
        }

        // Apply player damage
        self.player_hp -= damage;

        // Add new projectiles
        self.entities.extend(new_projectiles);

        // Remove dead entities
        self.entities.retain(|e| e.alive);

        damage
    }

    /// Check if player shot hits any enemy. Returns true if hit.
    pub fn player_shoot(&mut self) -> bool {
        let max_range = 15.0f32;
        let dx = self.player_angle.cos();
        let dy = self.player_angle.sin();

        let mut best_dist = max_range;
        let mut best_idx: Option<usize> = None;

        for (i, ent) in self.entities.iter().enumerate() {
            if ent.entity_type != EntityType::Enemy || !ent.alive {
                continue;
            }
            // Project enemy onto ray
            let ex = ent.x - self.player_x;
            let ey = ent.y - self.player_y;
            let t = ex * dx + ey * dy;
            if t <= 0.0 || t >= best_dist {
                continue;
            }
            // Perpendicular distance
            let perp = (ex * dy - ey * dx).abs();
            if perp < 1.0 {
                // hit radius
                best_dist = t;
                best_idx = Some(i);
            }
        }

        if let Some(idx) = best_idx {
            self.entities[idx].hp -= 25.0;
            if self.entities[idx].hp <= 0.0 {
                self.entities[idx].alive = false;
            }
            true
        } else {
            false
        }
    }
}

// =========================================================================
// Dynamic Difficulty Adjustment (DDA)
// =========================================================================

/// Dynamic difficulty adjustment controller.
///
/// Monitors the brain's performance (rolling damage ratio) and adapts
/// enemy spawn rate and enemy parameters to keep the challenge level
/// in the learning sweet spot.
pub struct DDAController {
    /// Rolling window of damage events (true=took damage this step).
    damage_history: Vec<bool>,
    /// Window size for rolling average.
    window_size: usize,
    /// Current difficulty level [0.0, 2.0]. 1.0 = baseline.
    pub difficulty: f32,
    /// Steps between enemy spawns (lower = harder).
    pub spawn_interval: u32,
    /// Step counter for spawning.
    spawn_counter: u32,
    /// Target damage rate (fraction of steps with damage).
    target_damage_rate: f32,
}

impl DDAController {
    pub fn new() -> Self {
        Self {
            damage_history: Vec::with_capacity(500),
            window_size: 500,
            difficulty: 1.0,
            spawn_interval: 200,
            spawn_counter: 0,
            target_damage_rate: 0.1, // 10% of steps should have damage
        }
    }

    /// Record whether damage occurred this step and adjust difficulty.
    pub fn update(&mut self, took_damage: bool) {
        self.damage_history.push(took_damage);
        if self.damage_history.len() > self.window_size {
            self.damage_history.remove(0);
        }

        // Compute rolling damage rate
        if self.damage_history.len() >= 100 {
            let recent = &self.damage_history[self.damage_history.len().saturating_sub(100)..];
            let damage_rate = recent.iter().filter(|&&d| d).count() as f32 / recent.len() as f32;

            // Adjust difficulty toward target
            if damage_rate < self.target_damage_rate * 0.5 {
                // Too easy — increase difficulty
                self.difficulty = (self.difficulty + 0.005).min(2.0);
                self.spawn_interval = (self.spawn_interval as f32 * 0.99).max(50.0) as u32;
            } else if damage_rate > self.target_damage_rate * 1.5 {
                // Too hard — decrease difficulty
                self.difficulty = (self.difficulty - 0.005).max(0.3);
                self.spawn_interval = (self.spawn_interval as f32 * 1.01).min(500.0) as u32;
            }
        }
    }

    /// Check if it's time to spawn a new enemy. Returns true on spawn tick.
    pub fn should_spawn(&mut self) -> bool {
        self.spawn_counter += 1;
        if self.spawn_counter >= self.spawn_interval {
            self.spawn_counter = 0;
            true
        } else {
            false
        }
    }

    /// Get enemy speed scaled by difficulty.
    pub fn enemy_speed(&self) -> f32 {
        0.5 * self.difficulty
    }

    /// Get enemy HP scaled by difficulty.
    pub fn enemy_hp(&self) -> f32 {
        50.0 * self.difficulty
    }
}

/// Ray-segment intersection. Returns the distance along the ray (positive
/// if hit, negative if no intersection). Uses the standard parametric
/// line-line intersection formula.
fn ray_segment_intersect(
    ox: f32,
    oy: f32,
    dx: f32,
    dy: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
) -> f32 {
    let sx = x2 - x1;
    let sy = y2 - y1;
    let denom = dx * sy - dy * sx;
    if denom.abs() < 1e-8 {
        return -1.0;
    }
    let t = ((x1 - ox) * sy - (y1 - oy) * sx) / denom;
    let u = ((x1 - ox) * dy - (y1 - oy) * dx) / denom;
    if t > 0.0 && u >= 0.0 && u <= 1.0 {
        t
    } else {
        -1.0
    }
}

// =========================================================================
// Brain region layout
// =========================================================================

/// Neuron index ranges for each Doom brain region.
///
/// Regions are laid out contiguously: V1, V2, motor pops (4), VTA, LC,
/// prefrontal, hippocampus, amygdala. Any remaining neurons after the
/// amygdala are unassigned cortical filler that still participate in
/// dynamics via random local connectivity.
struct DoomRegionLayout {
    v1: (usize, usize), // (start, size)
    v2: (usize, usize),
    turn_left: (usize, usize),
    turn_right: (usize, usize),
    move_forward: (usize, usize),
    shoot: (usize, usize),
    vta: (usize, usize),
    lc: (usize, usize),
    prefrontal: (usize, usize),
    hippocampus: (usize, usize),
    amygdala: (usize, usize),
    /// Total neurons allocated to named regions (used by diagnostics/tests).
    #[allow(dead_code)]
    total_assigned: usize,
}

impl DoomRegionLayout {
    /// Compute region layout for `n` total neurons.
    ///
    /// Each region gets a fraction of the total neuron count, with a minimum
    /// of 4 neurons per region to ensure meaningful population dynamics.
    fn from_n(n: usize) -> Self {
        let mut offset = 0usize;
        let mut next = |frac: f32| -> (usize, usize) {
            let size = (n as f32 * frac).max(4.0) as usize;
            let start = offset;
            offset += size;
            (start, size)
        };

        let layout = Self {
            v1: next(0.15),
            v2: next(0.10),
            turn_left: next(0.05),
            turn_right: next(0.05),
            move_forward: next(0.05),
            shoot: next(0.05),
            vta: next(0.03),
            lc: next(0.03),
            prefrontal: next(0.15),
            hippocampus: next(0.12),
            amygdala: next(0.08),
            total_assigned: offset,
        };
        layout
    }

    /// Get all region (start, size) pairs with their names for iteration.
    #[allow(dead_code)]
    fn all_regions(&self) -> Vec<(&str, (usize, usize))> {
        vec![
            ("v1", self.v1),
            ("v2", self.v2),
            ("turn_left", self.turn_left),
            ("turn_right", self.turn_right),
            ("move_forward", self.move_forward),
            ("shoot", self.shoot),
            ("vta", self.vta),
            ("lc", self.lc),
            ("prefrontal", self.prefrontal),
            ("hippocampus", self.hippocampus),
            ("amygdala", self.amygdala),
        ]
    }
}

// =========================================================================
// Experiment results
// =========================================================================

/// Result of a single Doom brain experiment.
#[derive(Clone, Debug)]
pub struct DoomExperimentResult {
    /// Experiment name (e.g. "threat_avoidance", "navigation", "combat").
    pub name: String,
    /// Whether the experiment passed its success criterion.
    pub passed: bool,
    /// Name of the primary metric measured.
    pub metric_name: String,
    /// Value of the primary metric.
    pub metric_value: f64,
    /// Threshold for passing.
    pub threshold: f64,
    /// Human-readable details string.
    pub details: String,
}

// =========================================================================
// DoomBrainSim -- main simulation struct
// =========================================================================

/// Doom FPS brain simulation with a biophysical `MolecularBrain`.
///
/// Owns the game engine, brain, and region layout. The simulation loop
/// is: raycast -> inject visual current -> brain.step() -> decode motor ->
/// update player -> check game events (damage, pickups) -> apply
/// neuromodulation.
pub struct DoomBrainSim {
    /// The biophysical brain (HH + STDP + full molecular pipeline).
    pub brain: MolecularBrain,
    /// The Doom raycasting engine and game state.
    pub doom: DoomEngine,
    /// Region layout mapping neuron indices to brain areas.
    layout: DoomRegionLayout,
    /// Operating mode (embodied vs disembodied).
    pub mode: DoomMode,
    /// Total neuron count.
    pub n_neurons: usize,
    /// Simulation timestep in ms (matches brain.dt).
    pub dt: f32,
    /// PSC amplitude scaling (critical for cascade propagation).
    pub psc_scale: f32,
    /// Total simulation steps executed.
    pub step_count: u64,
    /// RNG for FEP noise stimulation.
    rng: StdRng,
    /// Nociceptor current amplitude (uA/cm^2) injected on damage.
    pub nociceptor_current: f32,
    /// NE burst current (uA/cm^2) injected to LC on damage.
    pub ne_burst_current: f32,
    /// DA burst current (uA/cm^2) injected to VTA on health pickup.
    pub da_burst_current: f32,
    /// FEP structured stimulation current (uA/cm^2).
    pub fep_structured_current: f32,
    /// FEP noise stimulation current (uA/cm^2).
    pub fep_noise_current: f32,
    /// FEP noise probability (fraction of neurons stimulated on MISS).
    pub fep_noise_prob: f32,
}

impl DoomBrainSim {
    /// Create a new Doom brain simulation.
    ///
    /// # Arguments
    /// * `n_neurons` - Total neuron count. Minimum ~100 for meaningful
    ///   dynamics; recommended 800+ for learning experiments.
    /// * `mode` - Embodied (64 rays, retina pipeline) or Disembodied (8 rays).
    /// * `seed` - RNG seed for reproducible connectivity and experiments.
    pub fn new(n_neurons: usize, mode: DoomMode, seed: u64) -> Self {
        let layout = DoomRegionLayout::from_n(n_neurons);

        // Build connectivity edges
        let edges = Self::build_connectivity(&layout, n_neurons, seed);

        // Construct brain from edges (sorts internally, builds CSR)
        let mut brain = MolecularBrain::from_edges(n_neurons, &edges);
        brain.psc_scale = 30.0;

        // Assign neuron archetypes
        Self::assign_archetypes(&mut brain, &layout);

        // Create Doom engine: 8 rays for disembodied, 64 for embodied
        let n_rays = match mode {
            DoomMode::Disembodied => 8,
            DoomMode::Embodied => 64,
        };
        let doom = DoomEngine::new(n_rays);

        Self {
            brain,
            doom,
            layout,
            mode,
            n_neurons,
            dt: 0.1,
            psc_scale: 30.0,
            step_count: 0,
            rng: StdRng::seed_from_u64(seed.wrapping_add(12345)),
            nociceptor_current: 30.0,
            ne_burst_current: 40.0,
            da_burst_current: 35.0,
            fep_structured_current: 25.0,
            fep_noise_current: 20.0,
            fep_noise_prob: 0.3,
        }
    }

    /// Build biologically plausible inter-region connectivity.
    ///
    /// Connection probabilities are scaled inversely with network size so
    /// that average in-degree stays roughly constant (~10 inputs per neuron).
    /// This prevents quadratic synapse explosion in larger networks.
    fn build_connectivity(
        layout: &DoomRegionLayout,
        n: usize,
        seed: u64,
    ) -> Vec<(u32, u32, NTType)> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut edges: Vec<(u32, u32, NTType)> = Vec::new();

        // Base probability scaled to keep ~10 inputs per neuron
        let base_prob = (10.0 / n as f32).min(0.3);

        // Closure: connect source region to destination region
        let connect = |edges: &mut Vec<(u32, u32, NTType)>,
                       rng: &mut StdRng,
                       src: (usize, usize),
                       dst: (usize, usize),
                       prob: f32,
                       nt: NTType| {
            for pre in src.0..(src.0 + src.1) {
                for post in dst.0..(dst.0 + dst.1) {
                    if pre != post && rng.gen::<f32>() < prob {
                        edges.push((pre as u32, post as u32, nt));
                    }
                }
            }
        };

        // --- Visual pathway: V1 -> V2 -> Prefrontal ---
        connect(
            &mut edges,
            &mut rng,
            layout.v1,
            layout.v2,
            base_prob * 3.0,
            NTType::Glutamate,
        );
        connect(
            &mut edges,
            &mut rng,
            layout.v2,
            layout.prefrontal,
            base_prob * 2.0,
            NTType::Glutamate,
        );

        // --- Threat pathway: V2 -> Amygdala -> LC (NE burst) ---
        connect(
            &mut edges,
            &mut rng,
            layout.v2,
            layout.amygdala,
            base_prob * 2.0,
            NTType::Glutamate,
        );
        connect(
            &mut edges,
            &mut rng,
            layout.amygdala,
            layout.lc,
            base_prob * 3.0,
            NTType::Glutamate,
        );

        // --- Reward pathway: Prefrontal -> VTA (DA) ---
        connect(
            &mut edges,
            &mut rng,
            layout.prefrontal,
            layout.vta,
            base_prob,
            NTType::Glutamate,
        );

        // --- Motor commands: Prefrontal -> Motor populations ---
        connect(
            &mut edges,
            &mut rng,
            layout.prefrontal,
            layout.turn_left,
            base_prob * 2.0,
            NTType::Glutamate,
        );
        connect(
            &mut edges,
            &mut rng,
            layout.prefrontal,
            layout.turn_right,
            base_prob * 2.0,
            NTType::Glutamate,
        );
        connect(
            &mut edges,
            &mut rng,
            layout.prefrontal,
            layout.move_forward,
            base_prob * 2.0,
            NTType::Glutamate,
        );
        connect(
            &mut edges,
            &mut rng,
            layout.prefrontal,
            layout.shoot,
            base_prob,
            NTType::Glutamate,
        );

        // --- Amygdala -> Motor (fear-driven avoidance) ---
        connect(
            &mut edges,
            &mut rng,
            layout.amygdala,
            layout.turn_left,
            base_prob * 1.5,
            NTType::Glutamate,
        );
        connect(
            &mut edges,
            &mut rng,
            layout.amygdala,
            layout.move_forward,
            base_prob * 1.5,
            NTType::Glutamate,
        );

        // --- Spatial memory: Hippocampus <-> Prefrontal ---
        connect(
            &mut edges,
            &mut rng,
            layout.hippocampus,
            layout.prefrontal,
            base_prob,
            NTType::Glutamate,
        );
        connect(
            &mut edges,
            &mut rng,
            layout.prefrontal,
            layout.hippocampus,
            base_prob,
            NTType::Glutamate,
        );

        // --- Hippocampus recurrent (spatial pattern completion) ---
        connect(
            &mut edges,
            &mut rng,
            layout.hippocampus,
            layout.hippocampus,
            base_prob * 0.8,
            NTType::Glutamate,
        );

        // --- V2 -> Hippocampus (visual spatial input) ---
        connect(
            &mut edges,
            &mut rng,
            layout.v2,
            layout.hippocampus,
            base_prob,
            NTType::Glutamate,
        );

        // --- Neuromodulatory broadcast: VTA DA -> Motor + Prefrontal ---
        connect(
            &mut edges,
            &mut rng,
            layout.vta,
            layout.prefrontal,
            base_prob,
            NTType::Dopamine,
        );
        connect(
            &mut edges,
            &mut rng,
            layout.vta,
            layout.move_forward,
            base_prob,
            NTType::Dopamine,
        );
        connect(
            &mut edges,
            &mut rng,
            layout.vta,
            layout.shoot,
            base_prob,
            NTType::Dopamine,
        );

        // --- LC NE -> arousal targets ---
        connect(
            &mut edges,
            &mut rng,
            layout.lc,
            layout.prefrontal,
            base_prob,
            NTType::Norepinephrine,
        );
        connect(
            &mut edges,
            &mut rng,
            layout.lc,
            layout.amygdala,
            base_prob,
            NTType::Norepinephrine,
        );
        connect(
            &mut edges,
            &mut rng,
            layout.lc,
            layout.v1,
            base_prob * 0.5,
            NTType::Norepinephrine,
        );
        connect(
            &mut edges,
            &mut rng,
            layout.lc,
            layout.v2,
            base_prob * 0.5,
            NTType::Norepinephrine,
        );

        // --- Local GABAergic inhibition within regions ---
        let local_prob = base_prob * 0.3;
        let add_local =
            |edges: &mut Vec<(u32, u32, NTType)>, rng: &mut StdRng, r: (usize, usize)| {
                for pre in r.0..(r.0 + r.1) {
                    for post in r.0..(r.0 + r.1) {
                        if pre != post && rng.gen::<f32>() < local_prob {
                            edges.push((pre as u32, post as u32, NTType::GABA));
                        }
                    }
                }
            };
        add_local(&mut edges, &mut rng, layout.v1);
        add_local(&mut edges, &mut rng, layout.v2);
        add_local(&mut edges, &mut rng, layout.prefrontal);
        add_local(&mut edges, &mut rng, layout.amygdala);

        // --- Motor population mutual inhibition (winner-take-all) ---
        // turn_left <-> turn_right inhibition
        connect(
            &mut edges,
            &mut rng,
            layout.turn_left,
            layout.turn_right,
            base_prob * 2.0,
            NTType::GABA,
        );
        connect(
            &mut edges,
            &mut rng,
            layout.turn_right,
            layout.turn_left,
            base_prob * 2.0,
            NTType::GABA,
        );

        edges
    }

    /// Assign neuron archetypes based on brain region identity.
    fn assign_archetypes(brain: &mut MolecularBrain, layout: &DoomRegionLayout) {
        // VTA neurons are dopaminergic
        for i in layout.vta.0..(layout.vta.0 + layout.vta.1) {
            brain.neurons.archetype[i] = NeuronArchetype::DopaminergicSN as u8;
        }
        // LC neurons are noradrenergic (Serotonergic archetype maps to NE)
        for i in layout.lc.0..(layout.lc.0 + layout.lc.1) {
            brain.neurons.archetype[i] = NeuronArchetype::Serotonergic as u8;
        }
        // Hippocampus uses granule cells for pattern separation
        let half_hipp = layout.hippocampus.1 / 2;
        for i in layout.hippocampus.0..(layout.hippocampus.0 + half_hipp) {
            brain.neurons.archetype[i] = NeuronArchetype::Granule as u8;
        }
        // All other regions default to Pyramidal (set by NeuronArrays::new)
    }

    // =====================================================================
    // Visual input injection
    // =====================================================================

    /// Inject raycast distances as external current into V1 neurons.
    ///
    /// Each ray maps to a contiguous block of V1 neurons. Current is
    /// inversely proportional to distance (Weber-Fechner encoding):
    /// closer objects produce stronger depolarization.
    fn inject_visual_input(&mut self) {
        let n_rays = self.doom.n_rays as usize;
        let v1_size = self.layout.v1.1;
        if n_rays == 0 || v1_size == 0 {
            return;
        }
        let neurons_per_ray = v1_size / n_rays;
        if neurons_per_ray == 0 {
            return;
        }

        for (i, &dist) in self.doom.ray_distances.iter().enumerate() {
            // Weber-Fechner: I = gain / (dist + 1)
            // Close objects (dist ~1) -> ~20 uA/cm^2
            // Far objects (dist ~100) -> ~0.4 uA/cm^2
            let current = 40.0 / (dist + 1.0);
            let start = self.layout.v1.0 + i * neurons_per_ray;
            for j in 0..neurons_per_ray {
                let idx = start + j;
                if idx < self.brain.neurons.count {
                    self.brain.stimulate(idx, current);
                }
            }
        }
    }

    // =====================================================================
    // Motor decoding
    // =====================================================================

    /// Compute mean voltage over a region. Returns resting potential if
    /// the region has zero neurons.
    fn region_mean_voltage(&self, region: (usize, usize)) -> f32 {
        if region.1 == 0 {
            return -65.0;
        }
        let sum: f32 = (region.0..region.0 + region.1)
            .map(|i| self.brain.neurons.voltage[i])
            .sum();
        sum / region.1 as f32
    }

    /// Count fired neurons in a region this step.
    fn region_fired_count(&self, region: (usize, usize)) -> usize {
        (region.0..region.0 + region.1)
            .filter(|&i| self.brain.neurons.fired[i] != 0)
            .count()
    }

    /// Decode motor commands from neural population voltages.
    ///
    /// Uses a zero-threshold differential decoder (following DishBrain
    /// findings): ANY voltage difference between left/right populations
    /// drives turning. This breaks motor symmetry and enables learning.
    ///
    /// Returns (turn_signal, speed, should_shoot).
    fn decode_motor(&self) -> (f32, f32, bool) {
        let left_v = self.region_mean_voltage(self.layout.turn_left);
        let right_v = self.region_mean_voltage(self.layout.turn_right);
        let forward_v = self.region_mean_voltage(self.layout.move_forward);
        let shoot_v = self.region_mean_voltage(self.layout.shoot);

        // Zero-threshold differential steering
        let turn = (right_v - left_v) * 0.01;

        // Forward speed: normalize from resting potential (-65 mV)
        // Voltage > -45 mV starts producing movement, > -20 mV is full speed
        let speed = ((forward_v + 65.0) / 45.0).clamp(0.0, 1.0);

        // Shoot when shoot population is significantly depolarized
        let should_shoot = shoot_v > -30.0;

        (turn, speed, should_shoot)
    }

    // =====================================================================
    // Neuromodulation
    // =====================================================================

    /// Apply nociceptor signal: inject current to amygdala + NE burst to LC.
    ///
    /// Called when the player takes damage. The amygdala activation drives
    /// a fear response that (via amygdala -> motor connections) triggers
    /// avoidance behavior. The LC NE burst increases global arousal and
    /// enhances STDP gain.
    fn apply_damage_signal(&mut self, damage_amount: f32) {
        let scale = (damage_amount / 10.0).min(2.0);

        // Nociceptor current to amygdala
        for i in self.layout.amygdala.0..(self.layout.amygdala.0 + self.layout.amygdala.1) {
            self.brain.stimulate(i, self.nociceptor_current * scale);
        }

        // NE burst to LC (arousal response)
        for i in self.layout.lc.0..(self.layout.lc.0 + self.layout.lc.1) {
            self.brain.stimulate(i, self.ne_burst_current * scale);
        }
    }

    /// Apply reward signal: DA burst to VTA.
    ///
    /// Called on health pickup or successful actions. The VTA DA burst
    /// modulates prefrontal and motor plasticity, reinforcing the action
    /// that led to the reward.
    fn apply_reward_signal(&mut self, reward_magnitude: f32) {
        let scale = (reward_magnitude / 10.0).min(2.0);
        for i in self.layout.vta.0..(self.layout.vta.0 + self.layout.vta.1) {
            self.brain.stimulate(i, self.da_burst_current * scale);
        }
    }

    /// Apply FEP structured stimulation (HIT protocol).
    ///
    /// On a successful avoidance (no damage), inject structured pulsed
    /// stimulation into V1 (low entropy = predictable input). This follows
    /// the Free Energy Principle: structured input is easier to predict,
    /// so the brain's prediction error (surprise) decreases.
    fn apply_fep_structured(&mut self) {
        // Pulsed: 5ms on / 5ms off pattern to avoid depolarization block
        let phase = (self.step_count % 100) as f32 * self.dt;
        let pulse_on = (phase % 10.0) < 5.0;
        if !pulse_on {
            return;
        }

        for i in self.layout.v1.0..(self.layout.v1.0 + self.layout.v1.1) {
            self.brain.stimulate(i, self.fep_structured_current);
        }
    }

    /// Apply FEP noise stimulation (MISS protocol).
    ///
    /// On damage (failed avoidance), inject random noise to 30% of V1
    /// neurons (high entropy = unpredictable input). The brain experiences
    /// high prediction error, driving plasticity to reduce future surprise.
    fn apply_fep_noise(&mut self) {
        for i in self.layout.v1.0..(self.layout.v1.0 + self.layout.v1.1) {
            if self.rng.gen::<f32>() < self.fep_noise_prob {
                self.brain.stimulate(i, self.fep_noise_current);
            }
        }
    }

    // =====================================================================
    // Simulation step
    // =====================================================================

    /// Run one complete simulation step.
    ///
    /// Pipeline:
    /// 1. Raycast the environment.
    /// 2. Inject visual input to V1 neurons.
    /// 3. Brain step (full HH + STDP + molecular pipeline).
    /// 4. Decode motor output from population voltages.
    /// 5. Update player position in the Doom engine.
    pub fn step(&mut self) {
        // 1. Raycast
        self.doom.raycast();

        // 2. Visual input injection
        self.inject_visual_input();

        // 3. Brain biophysics step
        self.brain.step();
        self.brain.sync_shadow_from_gpu();

        // 4. Motor decode
        let (turn, speed, _shoot) = self.decode_motor();

        // 5. Player physics update
        self.doom.update_player(turn, speed, self.dt);

        self.step_count += 1;
    }

    /// Run `n` simulation steps.
    pub fn run_steps(&mut self, n: u64) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Warmup: run `n` steps to stabilize baseline neural dynamics.
    ///
    /// During warmup, the brain reaches steady-state membrane potentials
    /// and the STDP traces equilibrate. Typical warmup is 2000 steps
    /// (200 ms at dt=0.1 ms).
    pub fn warmup(&mut self, n: u64) {
        self.brain.run(n);
    }

    // =====================================================================
    // Map setup helpers
    // =====================================================================

    /// Set up a simple rectangular arena with 4 walls.
    pub fn setup_arena(&mut self) {
        self.doom
            .add_room(0.0, 0.0, self.doom.map_width, self.doom.map_height);
    }

    /// Set up an arena with internal wall obstacles.
    pub fn setup_arena_with_obstacles(&mut self) {
        self.setup_arena();
        // Cross-shaped internal walls
        let w = self.doom.map_width;
        let h = self.doom.map_height;
        self.doom.add_wall(w * 0.3, h * 0.3, w * 0.3, h * 0.5);
        self.doom.add_wall(w * 0.7, h * 0.5, w * 0.7, h * 0.7);
    }

    // =====================================================================
    // Experiment 1: Threat Avoidance
    // =====================================================================

    /// Experiment 1: Threat avoidance -- learn to avoid damage zones.
    ///
    /// A damage zone is placed in the center of the arena. Each episode,
    /// the player starts at a fixed position. Damage is applied when the
    /// player enters the zone. FEP protocol: structured stim when safe
    /// (HIT), noise stim when taking damage (MISS).
    ///
    /// **Success criterion**: Second-half episodes should show less total
    /// damage than first-half episodes (>5% improvement).
    pub fn run_threat_avoidance(&mut self, n_episodes: u32) -> DoomExperimentResult {
        let steps_per_episode = 500u64;
        let damage_x = self.doom.map_width / 2.0;
        let damage_y = self.doom.map_height / 2.0;
        let damage_radius = 3.0f32;

        // Set up arena
        self.setup_arena();

        // Warmup to stabilize baseline
        self.warmup(2000);

        let mut damage_per_episode: Vec<f32> = Vec::with_capacity(n_episodes as usize);

        for _ep in 0..n_episodes {
            self.doom.reset_player(5.0, 5.0, 0.0);
            let initial_hp = self.doom.player_hp;

            for _step in 0..steps_per_episode {
                self.step();

                // Check damage zone
                let in_zone = self
                    .doom
                    .player_in_radius(damage_x, damage_y, damage_radius);

                if in_zone {
                    let damage = 0.1;
                    self.doom.player_hp -= damage;
                    self.apply_damage_signal(damage * 100.0);
                    self.apply_fep_noise();
                } else {
                    self.apply_fep_structured();
                }
            }

            let episode_damage = initial_hp - self.doom.player_hp;
            damage_per_episode.push(episode_damage.max(0.0));
        }

        // Evaluate learning: compare first half vs second half
        let half = n_episodes as usize / 2;
        let first_half_mean = if half > 0 {
            damage_per_episode[..half].iter().sum::<f32>() / half as f32
        } else {
            0.0
        };
        let second_half_mean = if n_episodes as usize - half > 0 {
            damage_per_episode[half..].iter().sum::<f32>() / (n_episodes as usize - half) as f32
        } else {
            0.0
        };

        let improvement = if first_half_mean > 0.0 {
            (first_half_mean - second_half_mean) / first_half_mean
        } else {
            0.0
        };

        DoomExperimentResult {
            name: "threat_avoidance".to_string(),
            passed: improvement > 0.05,
            metric_name: "damage_reduction_fraction".to_string(),
            metric_value: improvement as f64,
            threshold: 0.05,
            details: format!(
                "First half mean damage: {:.2}, second half: {:.2}, \
                 improvement: {:.1}% (threshold: >5%)",
                first_half_mean,
                second_half_mean,
                improvement * 100.0
            ),
        }
    }

    // =====================================================================
    // Experiment 2: Navigation
    // =====================================================================

    /// Experiment 2: Navigation -- explore the environment.
    ///
    /// Measures how many distinct grid cells (5x5 = 25 cells) the player
    /// visits during each episode. More visited cells indicate better
    /// exploratory behavior emerging from the brain dynamics.
    ///
    /// **Success criterion**: Mean areas visited > 3 of 25 cells.
    pub fn run_navigation(&mut self, n_episodes: u32) -> DoomExperimentResult {
        let steps_per_episode = 1000u64;

        // Set up arena
        self.setup_arena();

        // Warmup
        self.warmup(2000);

        let mut areas_per_episode: Vec<usize> = Vec::with_capacity(n_episodes as usize);

        for _ep in 0..n_episodes {
            self.doom
                .reset_player(self.doom.map_width / 2.0, self.doom.map_height / 2.0, 0.0);

            // Track visited grid cells (5x5 grid)
            let grid_size = 5usize;
            let cell_w = self.doom.map_width / grid_size as f32;
            let cell_h = self.doom.map_height / grid_size as f32;
            let mut visited = vec![false; grid_size * grid_size];

            for _step in 0..steps_per_episode {
                self.step();

                // Mark current grid cell as visited
                let gx = (self.doom.player_x / cell_w)
                    .floor()
                    .min(grid_size as f32 - 1.0)
                    .max(0.0) as usize;
                let gy = (self.doom.player_y / cell_h)
                    .floor()
                    .min(grid_size as f32 - 1.0)
                    .max(0.0) as usize;
                visited[gy * grid_size + gx] = true;
            }

            let n_visited = visited.iter().filter(|&&v| v).count();
            areas_per_episode.push(n_visited);
        }

        let mean_visited = if areas_per_episode.is_empty() {
            0.0
        } else {
            areas_per_episode.iter().sum::<usize>() as f64 / areas_per_episode.len() as f64
        };

        DoomExperimentResult {
            name: "navigation".to_string(),
            passed: mean_visited > 3.0,
            metric_name: "mean_areas_visited".to_string(),
            metric_value: mean_visited,
            threshold: 3.0,
            details: format!(
                "Mean areas visited: {:.1}/25 across {} episodes",
                mean_visited, n_episodes
            ),
        }
    }

    // =====================================================================
    // Experiment 3: Combat
    // =====================================================================

    /// Experiment 3: Combat -- aim and shoot at a stationary target.
    ///
    /// An enemy entity is placed at a known position. The player starts
    /// facing away from it. A "hit" is registered when the player is
    /// facing within 10 degrees of the target AND the shoot motor
    /// population is active.
    ///
    /// **Success criterion**: Mean hits per episode > 0 (any hits at all
    /// demonstrates motor-visual coordination emerging from STDP).
    pub fn run_combat(&mut self, n_episodes: u32) -> DoomExperimentResult {
        let steps_per_episode = 500u64;

        // Set up arena with one enemy
        self.setup_arena();
        let enemy_x = 15.0f32;
        let enemy_y = 10.0f32;
        self.doom
            .add_entity(enemy_x, enemy_y, 100.0, EntityType::Enemy);

        // Warmup
        self.warmup(2000);

        let mut hits_per_episode: Vec<u32> = Vec::with_capacity(n_episodes as usize);

        for _ep in 0..n_episodes {
            self.doom.reset_player(5.0, 10.0, 0.0);
            let mut hits = 0u32;

            for _step in 0..steps_per_episode {
                self.step();

                // Check if facing enemy within ~10 degrees (0.17 radians)
                let angle_err = self.doom.angular_error_to(enemy_x, enemy_y);
                let (_, _, should_shoot) = self.decode_motor();

                if should_shoot && angle_err.abs() < 0.17 {
                    hits += 1;

                    // Reward signal on hit
                    self.apply_reward_signal(10.0);
                }
            }

            hits_per_episode.push(hits);
        }

        let mean_hits = if hits_per_episode.is_empty() {
            0.0
        } else {
            hits_per_episode.iter().sum::<u32>() as f64 / hits_per_episode.len() as f64
        };

        DoomExperimentResult {
            name: "combat".to_string(),
            passed: mean_hits > 0.0,
            metric_name: "mean_hits_per_episode".to_string(),
            metric_value: mean_hits,
            threshold: 0.0,
            details: format!(
                "Mean hits: {:.1}/episode across {} episodes. \
                 Best episode: {} hits",
                mean_hits,
                n_episodes,
                hits_per_episode.iter().max().copied().unwrap_or(0),
            ),
        }
    }

    // =====================================================================
    // Run all experiments
    // =====================================================================

    /// Run all 3 experiments and return results.
    ///
    /// Each experiment creates a fresh arena but reuses the brain state
    /// (simulating continuous learning across tasks).
    pub fn run_all(&mut self, episodes_per_experiment: u32) -> Vec<DoomExperimentResult> {
        let mut results = Vec::with_capacity(3);
        results.push(self.run_threat_avoidance(episodes_per_experiment));
        results.push(self.run_navigation(episodes_per_experiment));
        results.push(self.run_combat(episodes_per_experiment));
        results
    }

    // =====================================================================
    // Diagnostics
    // =====================================================================

    /// Get a diagnostic summary of brain activity.
    pub fn diagnostics(&self) -> DoomDiagnostics {
        let total_fired: usize = (0..self.n_neurons)
            .filter(|&i| self.brain.neurons.fired[i] != 0)
            .count();

        DoomDiagnostics {
            step: self.step_count,
            player_x: self.doom.player_x,
            player_y: self.doom.player_y,
            player_angle: self.doom.player_angle,
            player_hp: self.doom.player_hp,
            total_fired,
            v1_fired: self.region_fired_count(self.layout.v1),
            v2_fired: self.region_fired_count(self.layout.v2),
            prefrontal_fired: self.region_fired_count(self.layout.prefrontal),
            motor_left_v: self.region_mean_voltage(self.layout.turn_left),
            motor_right_v: self.region_mean_voltage(self.layout.turn_right),
            motor_forward_v: self.region_mean_voltage(self.layout.move_forward),
            motor_shoot_v: self.region_mean_voltage(self.layout.shoot),
            amygdala_fired: self.region_fired_count(self.layout.amygdala),
            vta_fired: self.region_fired_count(self.layout.vta),
            lc_fired: self.region_fired_count(self.layout.lc),
        }
    }
}

/// Diagnostic snapshot of the Doom brain state.
#[derive(Clone, Debug)]
pub struct DoomDiagnostics {
    pub step: u64,
    pub player_x: f32,
    pub player_y: f32,
    pub player_angle: f32,
    pub player_hp: f32,
    pub total_fired: usize,
    pub v1_fired: usize,
    pub v2_fired: usize,
    pub prefrontal_fired: usize,
    pub motor_left_v: f32,
    pub motor_right_v: f32,
    pub motor_forward_v: f32,
    pub motor_shoot_v: f32,
    pub amygdala_fired: usize,
    pub vta_fired: usize,
    pub lc_fired: usize,
}

impl std::fmt::Display for DoomDiagnostics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Step {:6} | pos ({:.1},{:.1}) angle {:.2} HP {:.0} | \
             fired {}/{} | V1:{} V2:{} PFC:{} AMY:{} VTA:{} LC:{} | \
             motor L:{:.1} R:{:.1} F:{:.1} S:{:.1}",
            self.step,
            self.player_x,
            self.player_y,
            self.player_angle,
            self.player_hp,
            self.total_fired,
            0, // total neurons not stored in diagnostics
            self.v1_fired,
            self.v2_fired,
            self.prefrontal_fired,
            self.amygdala_fired,
            self.vta_fired,
            self.lc_fired,
            self.motor_left_v,
            self.motor_right_v,
            self.motor_forward_v,
            self.motor_shoot_v,
        )
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_doom_engine_raycast() {
        let mut doom = DoomEngine::new(8);
        doom.add_room(0.0, 0.0, 20.0, 20.0);
        doom.player_x = 10.0;
        doom.player_y = 10.0;
        doom.player_angle = 0.0;
        doom.raycast();

        // All rays should hit a wall (room is 20x20, player is centered)
        for &d in &doom.ray_distances {
            assert!(d < 100.0, "Ray should hit a wall, got dist {}", d);
            assert!(d > 0.0, "Ray distance should be positive");
        }
    }

    #[test]
    fn test_doom_engine_player_movement() {
        let mut doom = DoomEngine::new(8);
        doom.player_x = 10.0;
        doom.player_y = 10.0;
        doom.player_angle = 0.0;

        // Move forward (angle=0 means +x direction)
        doom.update_player(0.0, 5.0, 1.0);
        assert!((doom.player_x - 15.0).abs() < 0.01);
        assert!((doom.player_y - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_doom_engine_player_clamp() {
        let mut doom = DoomEngine::new(8);
        doom.player_x = 0.5;
        doom.player_y = 0.5;
        doom.player_angle = std::f32::consts::PI; // facing -x

        doom.update_player(0.0, 100.0, 1.0);
        // Should be clamped to 0.5
        assert!(doom.player_x >= 0.5);
        assert!(doom.player_y >= 0.5);
    }

    #[test]
    fn test_ray_segment_intersect() {
        // Ray from (0,0) in +x direction hitting vertical wall at x=5
        let d = ray_segment_intersect(0.0, 0.0, 1.0, 0.0, 5.0, -1.0, 5.0, 1.0);
        assert!((d - 5.0).abs() < 0.01, "Expected ~5.0, got {}", d);

        // Ray going away from the wall
        let d = ray_segment_intersect(0.0, 0.0, -1.0, 0.0, 5.0, -1.0, 5.0, 1.0);
        assert!(d < 0.0, "Expected no hit (negative), got {}", d);
    }

    #[test]
    fn test_doom_region_layout() {
        let layout = DoomRegionLayout::from_n(800);

        // Verify regions don't overlap
        let regions = layout.all_regions();
        for i in 0..regions.len() {
            let (_, (s1, sz1)) = regions[i];
            for j in (i + 1)..regions.len() {
                let (_, (s2, sz2)) = regions[j];
                let end1 = s1 + sz1;
                let end2 = s2 + sz2;
                assert!(
                    end1 <= s2 || end2 <= s1,
                    "Regions {} and {} overlap: [{},{}) vs [{},{})",
                    regions[i].0,
                    regions[j].0,
                    s1,
                    end1,
                    s2,
                    end2,
                );
            }
        }

        // Total assigned should not exceed n
        assert!(layout.total_assigned <= 800);
    }

    #[test]
    fn test_doom_brain_construction() {
        let sim = DoomBrainSim::new(200, DoomMode::Disembodied, 42);
        assert_eq!(sim.n_neurons, 200);
        assert_eq!(sim.brain.neurons.count, 200);
        assert!(sim.brain.synapses.n_synapses > 0);
    }

    #[test]
    fn test_doom_brain_step() {
        let mut sim = DoomBrainSim::new(100, DoomMode::Disembodied, 42);
        sim.setup_arena();
        sim.doom.reset_player(10.0, 10.0, 0.0);

        // Run a few steps without panic
        for _ in 0..10 {
            sim.step();
        }

        assert_eq!(sim.step_count, 10);
    }

    #[test]
    fn test_doom_brain_visual_injection() {
        let mut sim = DoomBrainSim::new(200, DoomMode::Disembodied, 42);
        sim.setup_arena();
        // Place player at (1.0, 10.0) facing left (pi = -x direction) toward
        // the wall at x=0, so rays hit it at distance ~1.
        sim.doom.reset_player(1.0, 10.0, std::f32::consts::PI);

        // Raycast near a wall
        sim.doom.raycast();
        // Close wall should produce high current
        let min_dist = sim
            .doom
            .ray_distances
            .iter()
            .cloned()
            .fold(f32::MAX, f32::min);
        assert!(
            min_dist < 5.0,
            "Should be close to a wall, min_dist={}",
            min_dist
        );

        // Inject and check that some V1 neurons received current
        sim.inject_visual_input();
        let v1_current: f32 = (sim.layout.v1.0..sim.layout.v1.0 + sim.layout.v1.1)
            .map(|i| sim.brain.neurons.external_current[i])
            .sum();
        assert!(v1_current > 0.0, "V1 should have received visual current");
    }

    #[test]
    fn test_doom_brain_diagnostics() {
        let mut sim = DoomBrainSim::new(100, DoomMode::Disembodied, 42);
        sim.setup_arena();
        sim.doom.reset_player(10.0, 10.0, 0.0);
        sim.step();

        let diag = sim.diagnostics();
        assert_eq!(diag.step, 1);
        assert!((diag.player_x - 10.0).abs() < 5.0); // player shouldn't teleport
    }

    #[test]
    fn test_doom_angular_error() {
        let mut doom = DoomEngine::new(8);
        doom.player_x = 0.0;
        doom.player_y = 0.0;
        doom.player_angle = 0.0; // facing +x

        // Target directly ahead
        let err = doom.angular_error_to(10.0, 0.0);
        assert!(err.abs() < 0.01, "Should be ~0 error, got {}", err);

        // Target at 90 degrees
        let err = doom.angular_error_to(0.0, 10.0);
        assert!(
            (err - std::f32::consts::FRAC_PI_2).abs() < 0.01,
            "Should be ~pi/2 error, got {}",
            err
        );
    }

    #[test]
    fn test_doom_fep_structured() {
        let mut sim = DoomBrainSim::new(100, DoomMode::Disembodied, 42);
        sim.setup_arena();

        // Apply structured stim and verify V1 gets current
        sim.step_count = 0; // phase should produce a pulse
        sim.apply_fep_structured();

        let v1_current: f32 = (sim.layout.v1.0..sim.layout.v1.0 + sim.layout.v1.1)
            .map(|i| sim.brain.neurons.external_current[i])
            .sum();
        assert!(v1_current > 0.0, "FEP structured should inject V1 current");
    }

    #[test]
    fn test_doom_damage_signal() {
        let mut sim = DoomBrainSim::new(200, DoomMode::Disembodied, 42);

        // Apply damage and check amygdala + LC received current
        sim.apply_damage_signal(10.0);

        let amyg_current: f32 = (sim.layout.amygdala.0
            ..sim.layout.amygdala.0 + sim.layout.amygdala.1)
            .map(|i| sim.brain.neurons.external_current[i])
            .sum();
        let lc_current: f32 = (sim.layout.lc.0..sim.layout.lc.0 + sim.layout.lc.1)
            .map(|i| sim.brain.neurons.external_current[i])
            .sum();

        assert!(
            amyg_current > 0.0,
            "Amygdala should receive nociceptor input"
        );
        assert!(lc_current > 0.0, "LC should receive NE burst");
    }

    #[test]
    fn test_doom_reward_signal() {
        let mut sim = DoomBrainSim::new(200, DoomMode::Disembodied, 42);

        sim.apply_reward_signal(10.0);

        let vta_current: f32 = (sim.layout.vta.0..sim.layout.vta.0 + sim.layout.vta.1)
            .map(|i| sim.brain.neurons.external_current[i])
            .sum();

        assert!(vta_current > 0.0, "VTA should receive DA burst");
    }

    #[test]
    fn test_doom_motor_mutual_inhibition() {
        // Verify that turn_left and turn_right have GABA connections
        let sim = DoomBrainSim::new(400, DoomMode::Disembodied, 42);

        // Check for GABAergic synapses between left/right motor populations
        let mut found_gaba_lr = false;
        let mut found_gaba_rl = false;

        let left = sim.layout.turn_left;
        let right = sim.layout.turn_right;

        for pre in left.0..(left.0 + left.1) {
            for syn_idx in sim.brain.synapses.outgoing_range(pre) {
                let post = sim.brain.synapses.col_indices[syn_idx] as usize;
                let nt = sim.brain.synapses.nt_type[syn_idx];
                if post >= right.0 && post < right.0 + right.1 && nt == NTType::GABA as u8 {
                    found_gaba_lr = true;
                }
            }
        }

        for pre in right.0..(right.0 + right.1) {
            for syn_idx in sim.brain.synapses.outgoing_range(pre) {
                let post = sim.brain.synapses.col_indices[syn_idx] as usize;
                let nt = sim.brain.synapses.nt_type[syn_idx];
                if post >= left.0 && post < left.0 + left.1 && nt == NTType::GABA as u8 {
                    found_gaba_rl = true;
                }
            }
        }

        // At 400 neurons with base_prob * 2.0, we expect at least some GABA edges
        // (probabilistic, but very likely with 400 neurons)
        assert!(
            found_gaba_lr || found_gaba_rl,
            "Should have mutual inhibition between turn_left and turn_right"
        );
    }

    #[test]
    fn test_enemy_ai_state_transitions() {
        let mut doom = DoomEngine::new(8);
        doom.add_room(0.0, 0.0, 40.0, 40.0);
        doom.map_width = 40.0;
        doom.map_height = 40.0;
        doom.player_x = 10.0;
        doom.player_y = 10.0;

        // Add enemy far away (dist=~12.7, detect=8 -> should wander)
        doom.add_enemy(19.0, 19.0, 100.0, 0.5, 8.0, 3.0);
        assert_eq!(doom.entities[0].ai_state, EnemyAI::Wander);

        let mut rng = StdRng::seed_from_u64(42);
        doom.update_entities(&mut rng, 0.1);
        // Far enemy should stay wandering (dist > detect_range)
        assert_eq!(doom.entities[0].ai_state, EnemyAI::Wander);

        // Move enemy within detect range but outside attack range (dist=6, detect=8, attack=3)
        doom.entities[0].x = 16.0;
        doom.entities[0].y = 10.0;
        doom.update_entities(&mut rng, 0.1);
        assert_eq!(doom.entities[0].ai_state, EnemyAI::Chase);

        // Move enemy within attack range (dist=2, attack=3)
        doom.entities[0].x = 12.0;
        doom.entities[0].y = 10.0;
        doom.update_entities(&mut rng, 0.1);
        assert_eq!(doom.entities[0].ai_state, EnemyAI::Attack);
    }

    #[test]
    fn test_projectile_spawning() {
        let mut doom = DoomEngine::new(8);
        doom.add_room(0.0, 0.0, 20.0, 20.0);
        doom.player_x = 10.0;
        doom.player_y = 10.0;

        // Add attacking enemy close to player
        doom.add_enemy(12.0, 10.0, 100.0, 0.5, 8.0, 5.0);
        let mut rng = StdRng::seed_from_u64(42);

        // Update should spawn a projectile
        doom.update_entities(&mut rng, 0.1);

        let projectiles: Vec<_> = doom
            .entities
            .iter()
            .filter(|e| e.entity_type == EntityType::Projectile)
            .collect();
        assert!(
            !projectiles.is_empty(),
            "Enemy should fire a projectile when attacking"
        );
    }

    #[test]
    fn test_player_shoot_hits_enemy() {
        let mut doom = DoomEngine::new(8);
        doom.player_x = 10.0;
        doom.player_y = 10.0;
        doom.player_angle = 0.0; // facing +x

        // Add enemy directly ahead
        doom.add_enemy(15.0, 10.0, 100.0, 0.5, 8.0, 5.0);

        let hit = doom.player_shoot();
        assert!(hit, "Should hit enemy directly ahead");
        assert!(doom.entities[0].hp < 100.0, "Enemy should take damage");
    }

    #[test]
    fn test_player_shoot_misses() {
        let mut doom = DoomEngine::new(8);
        doom.player_x = 10.0;
        doom.player_y = 10.0;
        doom.player_angle = 0.0; // facing +x

        // Add enemy behind player
        doom.add_enemy(5.0, 10.0, 100.0, 0.5, 8.0, 5.0);

        let hit = doom.player_shoot();
        assert!(!hit, "Should not hit enemy behind player");
    }

    #[test]
    fn test_dda_controller() {
        let mut dda = DDAController::new();
        assert!((dda.difficulty - 1.0).abs() < 0.01);

        // Simulate a period with no damage (too easy)
        for _ in 0..200 {
            dda.update(false);
        }
        assert!(
            dda.difficulty > 1.0,
            "Difficulty should increase when no damage, got {}",
            dda.difficulty
        );

        // Simulate heavy damage (too hard)
        let mut dda2 = DDAController::new();
        for _ in 0..200 {
            dda2.update(true);
        }
        assert!(
            dda2.difficulty < 1.0,
            "Difficulty should decrease with constant damage, got {}",
            dda2.difficulty
        );
    }

    #[test]
    fn test_dda_spawn_interval() {
        let mut dda = DDAController::new();
        let initial_interval = dda.spawn_interval;

        // With no damage, spawn interval should decrease (more enemies)
        for _ in 0..300 {
            dda.update(false);
        }
        assert!(
            dda.spawn_interval <= initial_interval,
            "Spawn interval should decrease when too easy: {} vs {}",
            dda.spawn_interval,
            initial_interval
        );
    }

    #[test]
    fn test_health_pack_pickup() {
        let mut doom = DoomEngine::new(8);
        doom.player_x = 10.0;
        doom.player_y = 10.0;
        doom.player_hp = 50.0;

        doom.add_entity(10.5, 10.0, 1.0, EntityType::HealthPack);
        let mut rng = StdRng::seed_from_u64(42);
        doom.update_entities(&mut rng, 0.1);

        assert!(
            doom.player_hp > 50.0,
            "Player should gain health from pickup"
        );
        assert!(doom.entities.is_empty(), "Health pack should be consumed");
    }
}
