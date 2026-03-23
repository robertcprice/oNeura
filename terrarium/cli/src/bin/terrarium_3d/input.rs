//! Mouse and keyboard input handling for the orbit camera.

use super::camera::Camera;
use super::math::{add3, scale3};
use super::VIEWPORT_W;
use minifb::{MouseButton, MouseMode, Window};

pub struct ClickEvent {
    pub x: usize,
    pub y: usize,
}

pub struct InputState {
    prev_mouse_x: f32,
    prev_mouse_y: f32,
    mouse_was_down_left: bool,
    mouse_was_down_right: bool,
    click_start: Option<(f32, f32)>,
}

impl InputState {
    pub fn new() -> Self {
        Self {
            prev_mouse_x: 0.0,
            prev_mouse_y: 0.0,
            mouse_was_down_left: false,
            mouse_was_down_right: false,
            click_start: None,
        }
    }

    /// Handle mouse input. Returns Some(ClickEvent) if user clicked without dragging.
    pub fn handle(&mut self, window: &Window, cam: &mut Camera) -> Option<ClickEvent> {
        let (mx, my) = window.get_mouse_pos(MouseMode::Pass).unwrap_or((0.0, 0.0));
        let left_down = window.get_mouse_down(MouseButton::Left);
        let right_down = window.get_mouse_down(MouseButton::Right);
        let mut click_event = None;

        if mx < VIEWPORT_W as f32 {
            let dx = mx - self.prev_mouse_x;
            let dy = my - self.prev_mouse_y;

            // Track click start for click detection
            if left_down && !self.mouse_was_down_left {
                self.click_start = Some((mx, my));
            }
            if !left_down && self.mouse_was_down_left {
                // Mouse released — check if it was a click (small movement)
                if let Some((sx, sy)) = self.click_start {
                    let dist = ((mx - sx).powi(2) + (my - sy).powi(2)).sqrt();
                    if dist < 5.0 {
                        click_event = Some(ClickEvent {
                            x: mx as usize,
                            y: my as usize,
                        });
                    }
                }
                self.click_start = None;
            }

            if left_down && self.mouse_was_down_left {
                cam.yaw -= dx * 0.005;
                cam.pitch += dy * 0.005;
                cam.pitch = cam
                    .pitch
                    .clamp(-85.0_f32.to_radians(), 85.0_f32.to_radians());
            }
            if right_down && self.mouse_was_down_right {
                let zoom_scale = cam.distance * 0.003; // proportional to distance
                let r = cam.right();
                let f = cam.forward_xz();
                cam.target = add3(cam.target, scale3(r, -dx * zoom_scale));
                cam.target = add3(cam.target, scale3(f, dy * zoom_scale));
            }
        }

        // Scroll zoom: proportional to distance — infinite range
        if let Some((_, scroll_y)) = window.get_scroll_wheel() {
            let zoom_factor = 1.0 - scroll_y * 0.08;
            cam.distance = (cam.distance * zoom_factor).max(0.005);
        }

        self.prev_mouse_x = mx;
        self.prev_mouse_y = my;
        self.mouse_was_down_left = left_down;
        self.mouse_was_down_right = right_down;
        click_event
    }
}
