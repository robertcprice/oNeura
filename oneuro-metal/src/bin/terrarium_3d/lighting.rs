//! Sun direction, sun color, and sky gradient computation.

use super::math::*;
use super::color::rgb_f;

pub fn sun_direction(light: f32) -> V3 {
    let sun_angle = (light * std::f32::consts::PI).sin();
    let elevation = (sun_angle * 80.0_f32).to_radians().max(15.0_f32.to_radians());
    let azimuth = 30.0_f32.to_radians();
    normalize3([
        azimuth.cos() * elevation.cos(),
        elevation.sin(),
        azimuth.sin() * elevation.cos(),
    ])
}

pub fn sun_color(light: f32) -> V3 {
    let noon_c  = [1.0, 0.98, 0.92];
    let dawn_c  = [1.0, 0.65, 0.3];
    let night_c = [0.22, 0.22, 0.32];
    let sun_height = (light * std::f32::consts::PI).sin();
    if sun_height < 0.05 {
        night_c
    } else if sun_height < 0.3 {
        let t = (sun_height - 0.05) / 0.25;
        lerp3(dawn_c, noon_c, t)
    } else {
        noon_c
    }
}

/// Sky color gradient based on time of day (light = 0..1).
/// Returns (horizon_color, zenith_color) for sky rendering.
pub fn sky_gradient(light: f32) -> (V3, V3) {
    let t = light.clamp(0.0, 1.0);
    if t < 0.15 {
        // Night: deep blue-black
        ([0.02, 0.02, 0.06], [0.01, 0.01, 0.03])
    } else if t < 0.3 {
        // Dawn: orange-pink horizon, purple-blue zenith
        let s = (t - 0.15) / 0.15;
        let h = lerp3([0.02, 0.02, 0.06], [0.8, 0.4, 0.2], s);
        let z = lerp3([0.01, 0.01, 0.03], [0.3, 0.2, 0.5], s);
        (h, z)
    } else if t < 0.7 {
        // Day: bright blue sky
        let s = ((t - 0.3) / 0.4).min(1.0);
        let h = lerp3([0.8, 0.4, 0.2], [0.6, 0.75, 0.95], s);
        let z = lerp3([0.3, 0.2, 0.5], [0.3, 0.5, 0.9], s);
        (h, z)
    } else if t < 0.85 {
        // Dusk: golden-red horizon
        let s = (t - 0.7) / 0.15;
        let h = lerp3([0.6, 0.75, 0.95], [0.9, 0.4, 0.15], s);
        let z = lerp3([0.3, 0.5, 0.9], [0.2, 0.15, 0.4], s);
        (h, z)
    } else {
        // Night transition
        let s = (t - 0.85) / 0.15;
        let h = lerp3([0.9, 0.4, 0.15], [0.02, 0.02, 0.06], s);
        let z = lerp3([0.2, 0.15, 0.4], [0.01, 0.01, 0.03], s);
        (h, z)
    }
}

pub fn sky_color_at(light: f32, screen_y: f32) -> u32 {
    let sun_height = (light * std::f32::consts::PI).sin().max(0.0);
    let day_top     = [0.30, 0.50, 0.85];
    let day_horizon = [0.65, 0.78, 0.92];
    let night_top     = [0.04, 0.04, 0.10];
    let night_horizon = [0.08, 0.08, 0.16];
    let dawn_horizon  = [0.85, 0.45, 0.20];
    let top = lerp3(night_top, day_top, sun_height.min(1.0));
    let mut horiz = lerp3(night_horizon, day_horizon, sun_height.min(1.0));
    if sun_height < 0.3 && sun_height > 0.0 {
        let glow = (1.0 - sun_height / 0.3) * 0.6;
        horiz = lerp3(horiz, dawn_horizon, glow);
    }
    let c = lerp3(top, horiz, screen_y.clamp(0.0, 1.0));
    rgb_f(c[0], c[1], c[2])
}
