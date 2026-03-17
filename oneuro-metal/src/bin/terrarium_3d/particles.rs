//! Simple particle system for water evaporation, pollen, and fruit sparkle effects.

use super::math::*;
use super::color::rgb_f;

pub struct Particle {
    pub pos: V3,
    pub vel: V3,
    pub color: V3,
    pub life: f32,
    pub max_life: f32,
}

pub struct ParticleSystem {
    pub particles: Vec<Particle>,
}

impl ParticleSystem {
    pub fn new() -> Self { Self { particles: Vec::new() } }

    pub fn spawn(&mut self, pos: V3, vel: V3, color: V3, life: f32) {
        if self.particles.len() < 600 {
            self.particles.push(Particle { pos, vel, color, life, max_life: life });
        }
    }

    pub fn update(&mut self, dt: f32) {
        for p in &mut self.particles {
            p.pos = add3(p.pos, scale3(p.vel, dt));
            p.vel[1] -= 0.4 * dt; // light gravity
            p.life -= dt;
        }
        self.particles.retain(|p| p.life > 0.0);
    }

    /// Spawn water evaporation mist particles.
    pub fn spawn_water_evap(&mut self, wx: f32, wy: f32, wz: f32, frame: usize, idx: usize) {
        let seed = frame.wrapping_mul(374761393) ^ idx.wrapping_mul(668265263);
        let hash = (seed & 0xFFFF) as f32 / 65535.0;
        if hash < 0.06 {
            let vx = (hash - 0.03) * 0.4;
            let vz = ((hash * 7.3) % 1.0 - 0.5) * 0.4;
            self.spawn(
                [wx, wy + 0.04, wz],
                [vx, 0.25 + hash * 0.15, vz],
                [0.70, 0.82, 0.96],
                1.8 + hash,
            );
        }
    }

    /// Spawn pollen particles from plants.
    pub fn spawn_pollen(&mut self, wx: f32, wy: f32, wz: f32, frame: usize, idx: usize) {
        let seed = frame.wrapping_mul(2246822519) ^ idx.wrapping_mul(951274213);
        let hash = (seed & 0xFFFF) as f32 / 65535.0;
        if hash < 0.02 {
            self.spawn(
                [wx, wy, wz],
                [(hash - 0.5) * 0.3, 0.08 + hash * 0.12, (hash * 3.7 % 1.0 - 0.5) * 0.3],
                [0.96, 0.93, 0.48],
                2.5 + hash * 1.5,
            );
        }
    }

    /// Spawn sparkle particles from ripe fruits.
    pub fn spawn_fruit_sparkle(&mut self, wx: f32, wy: f32, wz: f32, frame: usize, idx: usize) {
        let seed = frame.wrapping_mul(1103515245) ^ idx.wrapping_mul(12345);
        let hash = (seed & 0xFFFF) as f32 / 65535.0;
        if hash < 0.015 {
            self.spawn(
                [wx, wy + 0.05, wz],
                [(hash - 0.5) * 0.15, 0.15 + hash * 0.1, (hash * 5.1 % 1.0 - 0.5) * 0.15],
                [0.95, 0.65, 0.20],
                1.2 + hash * 0.8,
            );
        }
    }

    /// Spawn fly trail particles — glowing streaks behind flying flies.
    pub fn spawn_fly_trail(&mut self, wx: f32, wy: f32, wz: f32, heading: f32, speed: f32, frame: usize, idx: usize) {
        let seed = frame.wrapping_mul(1664525) ^ idx.wrapping_mul(1013904223);
        let hash = (seed & 0xFFFF) as f32 / 65535.0;
        if hash < 0.15 && speed > 0.5 {
            let trail_vx = -heading.cos() * speed * 0.05 + (hash - 0.5) * 0.08;
            let trail_vz = -heading.sin() * speed * 0.05 + (hash * 3.1 % 1.0 - 0.5) * 0.08;
            let brightness = (speed / 3.0).clamp(0.3, 1.0);
            self.spawn(
                [wx, wy, wz],
                [trail_vx, -0.02, trail_vz],
                [0.95 * brightness, 0.85 * brightness, 0.30 * brightness],
                0.6 + hash * 0.4,
            );
        }
    }

    /// Spawn photosynthesis energy beam (sun -> plant) — golden particles drifting down.
    pub fn spawn_photosynthesis(&mut self, wx: f32, wy: f32, wz: f32, frame: usize, idx: usize) {
        if frame % 20 != (idx * 7) % 20 { return; }
        self.spawn(
            [wx, wy + 2.0, wz],
            [0.0, -0.05, 0.0],
            [1.0, 0.95, 0.3], // golden yellow
            30.0,
        );
    }

    /// Spawn respiration CO2 (plant/fly -> upward) — gray particles drifting up.
    pub fn spawn_respiration(&mut self, wx: f32, wy: f32, wz: f32, frame: usize, idx: usize) {
        if frame % 40 != (idx * 13) % 40 { return; }
        let vx = ((idx * 37 % 100) as f32 - 50.0) * 0.0004;
        let vz = ((idx * 53 % 100) as f32 - 50.0) * 0.0004;
        self.spawn(
            [wx, wy, wz],
            [vx, 0.015, vz],
            [0.5, 0.5, 0.55], // gray CO2
            40.0,
        );
    }

    /// Render particles as 2x2 alpha-blended dots to the viewport buffer.
    pub fn render(&self, buffer: &mut [u32], bw: usize, bh: usize, mvp: &M4) {
        for p in &self.particles {
            let clip = transform4(mvp, [p.pos[0], p.pos[1], p.pos[2], 1.0]);
            if clip[3] <= 0.001 { continue; }
            let inv_w = 1.0 / clip[3];
            let sx = (clip[0] * inv_w * 0.5 + 0.5) * bw as f32;
            let sy = (1.0 - (clip[1] * inv_w * 0.5 + 0.5)) * bh as f32;
            let alpha = (p.life / p.max_life).clamp(0.0, 1.0);
            let c = scale3(p.color, alpha);
            let color = rgb_f(c[0].min(1.0), c[1].min(1.0), c[2].min(1.0));
            for dy in 0..2i32 {
                for dx in 0..2i32 {
                    let px = sx as i32 + dx;
                    let py = sy as i32 + dy;
                    if px >= 0 && px < bw as i32 && py >= 0 && py < bh as i32 {
                        buffer[py as usize * bw + px as usize] = color;
                    }
                }
            }
        }
    }
}
