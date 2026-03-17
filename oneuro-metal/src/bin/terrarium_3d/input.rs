//! Mouse and keyboard input handling — orbit camera, click-to-select, and scroll zoom.

use minifb::{MouseButton, MouseMode, Window};
use super::camera::Camera;
use super::math::{add3, scale3};
use super::VIEWPORT_W;

pub struct ClickEvent {
    pub x: usize,
    pub y: usize,
}

pub struct InputState {
    prev_mouse_x: f32,
    prev_mouse_y: f32,
    mouse_was_down_left: bool,
    mouse_was_down_right: bool,
    drag_distance: f32,
}

impl InputState {
    pub fn new() -> Self {
        Self {
            prev_mouse_x: 0.0, prev_mouse_y: 0.0,
            mouse_was_down_left: false, mouse_was_down_right: false,
            drag_distance: 0.0,
        }
    }

    /// Handle mouse input. Returns Some(ClickEvent) if the user clicked (not dragged) in the viewport.
    pub fn handle(&mut self, window: &Window, cam: &mut Camera) -> Option<ClickEvent> {
        let (mx, my) = window.get_mouse_pos(MouseMode::Pass).unwrap_or((0.0, 0.0));
        let left_down = window.get_mouse_down(MouseButton::Left);
        let right_down = window.get_mouse_down(MouseButton::Right);
        let mut click = None;

        if mx < VIEWPORT_W as f32 {
            let dx = mx - self.prev_mouse_x;
            let dy = my - self.prev_mouse_y;

            // Left button: orbit camera (track drag distance)
            if left_down && self.mouse_was_down_left {
                cam.yaw -= dx * 0.005;
                cam.pitch += dy * 0.005;
                cam.pitch = cam.pitch.clamp(-85.0_f32.to_radians(), 85.0_f32.to_radians());
                self.drag_distance += (dx * dx + dy * dy).sqrt();
            }

            // Right button: pan
            if right_down && self.mouse_was_down_right {
                let r = cam.right();
                let f = cam.forward_xz();
                cam.target = add3(cam.target, scale3(r, -dx * 0.03));
                cam.target = add3(cam.target, scale3(f, dy * 0.03));
            }

            // Detect click (left button released after minimal drag)
            if !left_down && self.mouse_was_down_left && self.drag_distance < 4.0 {
                let sx = mx as usize;
                let sy = my as usize;
                if sx < VIEWPORT_W {
                    click = Some(ClickEvent { x: sx, y: sy });
                }
            }

            // Reset drag tracking on press
            if left_down && !self.mouse_was_down_left {
                self.drag_distance = 0.0;
            }
        }

        if let Some((_, scroll_y)) = window.get_scroll_wheel() {
            cam.distance = (cam.distance - scroll_y * 1.5).clamp(5.0, 80.0);
        }

        self.prev_mouse_x = mx;
        self.prev_mouse_y = my;
        self.mouse_was_down_left = left_down;
        self.mouse_was_down_right = right_down;
        click
    }
}
