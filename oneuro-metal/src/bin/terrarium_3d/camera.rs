//! Orbital camera with yaw/pitch/distance controls and zoom level detection.

use super::math::*;
use super::{VIEWPORT_W, VIEWPORT_H, FOV_Y, NEAR, FAR};

/// Semantic zoom levels for multi-scale rendering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ZoomLevel {
    Ecosystem,   // distance > 12
    Organism,    // 3 < distance <= 12
    Cellular,    // 0.5 < distance <= 3
    Molecular,   // distance <= 0.5
}

impl ZoomLevel {
    pub fn label(&self) -> &'static str {
        match self {
            ZoomLevel::Ecosystem => "Ecosystem",
            ZoomLevel::Organism => "Organism",
            ZoomLevel::Cellular => "Cellular",
            ZoomLevel::Molecular => "Molecular",
        }
    }
}

pub struct Camera {
    pub target: V3,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub following: bool,
}

impl Camera {
    pub fn new() -> Self {
        Self { target: v3(11.0, 0.0, 8.0), yaw: -0.4, pitch: 0.7, distance: 30.0, following: false }
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

    pub fn proj(&self) -> M4 {
        // Adjust near plane based on zoom level for molecular-scale rendering
        let near = if self.distance < 0.5 { 0.001 }
                   else if self.distance < 3.0 { 0.01 }
                   else { NEAR };
        perspective(FOV_Y, VIEWPORT_W as f32 / VIEWPORT_H as f32, near, FAR)
    }

    pub fn mvp(&self) -> M4 { mat4_mul(&self.proj(), &self.view()) }

    pub fn zoom_level(&self) -> ZoomLevel {
        if self.distance > 12.0 { ZoomLevel::Ecosystem }
        else if self.distance > 3.0 { ZoomLevel::Organism }
        else if self.distance > 0.5 { ZoomLevel::Cellular }
        else { ZoomLevel::Molecular }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}
