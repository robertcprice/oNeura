//! Orbital camera with yaw/pitch/distance controls.

use super::math::*;
use super::{VIEWPORT_W, VIEWPORT_H, FOV_Y, NEAR, FAR};

pub struct Camera {
    pub target: V3,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
}

impl Camera {
    pub fn new() -> Self {
        Self { target: v3(11.0, 0.0, 8.0), yaw: -0.4, pitch: 0.7, distance: 30.0 }
    }

    pub fn eye(&self) -> V3 {
        let cp = self.pitch.cos();
        let sp = self.pitch.sin();
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();
        [
            self.target[0] + self.distance * cp * sy,
            self.target[1] + self.distance * sp,
            self.target[2] + self.distance * cp * cy,
        ]
    }

    pub fn right(&self) -> V3 {
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();
        normalize3([cy, 0.0, -sy])
    }

    pub fn forward_xz(&self) -> V3 {
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();
        normalize3([-sy, 0.0, -cy])
    }

    pub fn view(&self) -> M4 { look_at(self.eye(), self.target, [0.0, 1.0, 0.0]) }
    pub fn proj(&self) -> M4 { perspective(FOV_Y, VIEWPORT_W as f32 / VIEWPORT_H as f32, NEAR, FAR) }
    pub fn mvp(&self) -> M4 { mat4_mul(&self.proj(), &self.view()) }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}
