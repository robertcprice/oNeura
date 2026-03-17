//! Night sky rendering: moon disc and star field.
//!
//! Renders a moon based on lunar_phase (0..1) with proper illumination,
//! and a field of stars that are more visible during dark nights (new moon).

use super::color::rgb_f;

/// Star field state - positions and brightness values.
/// Generated once and reused each frame.
pub struct StarField {
    /// Star positions in spherical coordinates (theta, phi, brightness)
    /// theta: angle around horizon (0..2*PI)
    /// phi: angle from horizon up (0..PI/2, only upper hemisphere)
    /// brightness: 0..1 base brightness
    stars: Vec<(f32, f32, f32)>,
}

impl StarField {
    /// Create a new star field with the given number of stars.
    /// Uses a fixed seed for deterministic generation.
    pub fn new(count: usize, seed: u64) -> Self {
        let mut stars = Vec::with_capacity(count);
        // Simple LCG random number generator
        let mut rng_state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut rand_f32 = || -> f32 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng_state >> 33) as f32 / (1u64 << 31) as f32
        };

        for _ in 0..count {
            // Distribute stars across the sky dome (upper hemisphere only)
            let theta = rand_f32() * std::f32::consts::TAU; // Around horizon
            let phi = rand_f32() * std::f32::consts::FRAC_PI_2; // From horizon to zenith
            // Bias brightness distribution - more dim stars than bright ones
            let brightness = rand_f32().powf(2.0) * 0.8 + 0.2;
            stars.push((theta, phi, brightness));
        }
        Self { stars }
    }

    /// Get reference to star data for rendering
    pub fn stars(&self) -> &[(f32, f32, f32)] {
        &self.stars
    }
}

/// Render stars to the color buffer.
/// Stars are more visible when moonlight is low (new moon) and fade during daylight.
///
/// # Arguments
/// * `color_buf` - The framebuffer to render to
/// * `width`, `height` - Buffer dimensions
/// * `light` - Daylight factor (0..1, 0=night, 1=noon)
/// * `moonlight` - Moon illumination factor (0..1)
/// * `cam_yaw` - Camera yaw angle for proper star positioning
/// * `stars` - Slice of (theta, phi, brightness) tuples
pub fn render_stars(
    color_buf: &mut [u32],
    width: usize,
    height: usize,
    light: f32,
    moonlight: f32,
    cam_yaw: f32,
    stars: &[(f32, f32, f32)],
) {
    // Stars only visible at night (light < 0.25)
    if light > 0.25 {
        return;
    }

    // Star visibility factor:
    // - Max during new moon (moonlight ~ 0) at deep night (light ~ 0)
    // - Min during full moon or near dawn/dusk
    let night_factor = 1.0 - (light / 0.25).min(1.0);
    let moon_darkness = 1.0 - moonlight * 0.6; // Full moon dims stars by 60%
    let visibility = night_factor * moon_darkness;

    if visibility < 0.01 {
        return;
    }

    // Render each star as a small point
    for &(theta, phi, base_brightness) in stars {
        // Apply camera rotation to star position
        let adjusted_theta = theta - cam_yaw;
        let cos_t = adjusted_theta.cos();
        let cos_p = phi.cos();
        let sin_p = phi.sin();

        // Convert to screen coordinates
        // phi=0 is horizon, phi=PI/2 is zenith
        // Project onto screen: x from theta, y from phi
        let screen_x = (cos_t * cos_p * 0.5 + 0.5) * width as f32;
        let screen_y = (1.0 - sin_p) * height as f32; // Invert: phi=0 at bottom, phi=PI/2 at top

        let px = screen_x as i32;
        let py = screen_y as i32;

        // Skip if outside viewport
        if px < 0 || px >= width as i32 || py < 0 || py >= height as i32 {
            continue;
        }

        // Calculate final brightness
        let brightness = base_brightness * visibility;

        // Star color: slight blue-white tint
        let r = (0.9 + 0.1 * base_brightness) * brightness;
        let g = (0.92 + 0.08 * base_brightness) * brightness;
        let b = (1.0) * brightness;

        // Render as 1 pixel point
        let idx = py as usize * width + px as usize;
        if idx < color_buf.len() {
            // Blend with existing sky color (additive)
            let existing = color_buf[idx];
            let er = ((existing >> 16) & 0xFF) as f32 / 255.0;
            let eg = ((existing >> 8) & 0xFF) as f32 / 255.0;
            let eb = (existing & 0xFF) as f32 / 255.0;

            let nr = (er + r).min(1.0);
            let ng = (eg + g).min(1.0);
            let nb = (eb + b).min(1.0);

            color_buf[idx] = rgb_f(nr, ng, nb);
        }
    }
}

/// Render the moon as a disc with proper phase illumination.
///
/// # Arguments
/// * `color_buf` - The framebuffer to render to
/// * `width`, `height` - Buffer dimensions
/// * `light` - Daylight factor (0..1)
/// * `lunar_phase` - Moon phase (0..1, 0=new moon, 0.5=full moon)
/// * `moonlight` - Moon illumination factor for brightness
/// * `cam_yaw` - Camera yaw angle
pub fn render_moon(
    color_buf: &mut [u32],
    width: usize,
    height: usize,
    light: f32,
    lunar_phase: f32,
    moonlight: f32,
    cam_yaw: f32,
) {
    // Moon only visible at night (light < 0.3)
    if light > 0.3 {
        return;
    }

    // Moon visibility factor
    let night_factor = 1.0 - (light / 0.3).min(1.0);
    if night_factor < 0.1 {
        return;
    }

    // Moon position in sky based on phase
    // Phase 0 = new moon (moon not visible, but we position it anyway)
    // Phase 0.5 = full moon (maximum illumination)
    // Moon travels in an arc from east to west
    let moon_arc = lunar_phase * std::f32::consts::TAU; // Full cycle through the night
    let moon_altitude = (moon_arc.sin() * 0.4 + 0.5) * std::f32::consts::FRAC_PI_2; // 0 to PI/2

    // Horizontal position influenced by phase and camera
    let moon_azimuth = moon_arc * 0.5 - cam_yaw;

    // Convert to screen coordinates
    let cos_az = moon_azimuth.cos();
    let sin_az = moon_azimuth.sin();
    let cos_alt = moon_altitude.cos();
    let sin_alt = moon_altitude.sin();

    let screen_x = (cos_az * cos_alt * 0.4 + 0.5) * width as f32;
    let screen_y = (1.0 - sin_alt * 0.9) * height as f32; // Near top of screen

    // Moon size (radius in pixels)
    let moon_radius = (width as f32 * 0.025).min(24.0).max(8.0);

    // Illumination fraction (0 at new moon, 1 at full moon)
    // The phase represents the moon's orbital position, illumination is the visible lit portion
    let illumination = (1.0 - (lunar_phase * std::f32::consts::TAU).cos()) * 0.5;

    // Moon brightness based on illumination and night factor
    let brightness = illumination * moonlight * night_factor;

    // Render moon disc
    let px_center = screen_x as i32;
    let py_center = screen_y as i32;
    let radius_sq = moon_radius * moon_radius;

    for dy in -moon_radius as i32..=moon_radius as i32 {
        for dx in -moon_radius as i32..=moon_radius as i32 {
            let dist_sq = (dx * dx + dy * dy) as f32;
            if dist_sq > radius_sq {
                continue;
            }

            let px = px_center + dx;
            let py = py_center + dy;

            if px < 0 || px >= width as i32 || py < 0 || py >= height as i32 {
                continue;
            }

            // Determine if this pixel is lit based on phase
            // Simulate the terminator line (boundary between lit and dark side)
            // For a waxing/waning moon, the terminator is a vertical curve
            let normalized_x = dx as f32 / moon_radius;

            // Terminator position based on phase
            // Phase 0 (new): terminator at left edge, all dark
            // Phase 0.5 (full): terminator at edges, all lit
            // Phase 1 (new again): terminator at right edge
            let terminator_x = (lunar_phase * 2.0 - 1.0) * 1.5; // -1.5 to 1.5

            // Pixel is lit if it's on the "bright side" of the terminator
            let is_lit = normalized_x > terminator_x;

            // Soft edge at terminator
            let lit_factor = if is_lit {
                let dist_from_terminator = (normalized_x - terminator_x).max(0.0);
                (dist_from_terminator * 3.0).min(1.0)
            } else {
                let dist_from_terminator = (terminator_x - normalized_x).max(0.0);
                // Slight earthshine on the dark side (very dim)
                0.05 * (dist_from_terminator * 2.0).min(1.0)
            };

            // Moon surface color with slight texture (brighter at center)
            let center_dist = dist_sq.sqrt() / moon_radius;
            let center_bright = 1.0 - center_dist * 0.2;

            // Final moon color
            let moon_r = 1.0 * brightness * lit_factor * center_bright;
            let moon_g = 0.97 * brightness * lit_factor * center_bright;
            let moon_b = 0.90 * brightness * lit_factor * center_bright;

            let idx = py as usize * width + px as usize;
            if idx < color_buf.len() {
                // Blend with existing sky color (additive for glow effect)
                let existing = color_buf[idx];
                let er = ((existing >> 16) & 0xFF) as f32 / 255.0;
                let eg = ((existing >> 8) & 0xFF) as f32 / 255.0;
                let eb = (existing & 0xFF) as f32 / 255.0;

                let nr = (er + moon_r).min(1.0);
                let ng = (eg + moon_g).min(1.0);
                let nb = (eb + moon_b).min(1.0);

                color_buf[idx] = rgb_f(nr, ng, nb);
            }
        }
    }

    // Add subtle glow around moon when illuminated
    if brightness > 0.3 {
        let glow_radius = moon_radius * 2.0;
        let glow_brightness = brightness * 0.15;

        for dy in -glow_radius as i32..=glow_radius as i32 {
            for dx in -glow_radius as i32..=glow_radius as i32 {
                let dist_sq = (dx * dx + dy * dy) as f32;
                if dist_sq <= radius_sq || dist_sq > glow_radius * glow_radius {
                    continue;
                }

                let px = px_center + dx;
                let py = py_center + dy;

                if px < 0 || px >= width as i32 || py < 0 || py >= height as i32 {
                    continue;
                }

                let dist = dist_sq.sqrt();
                let glow_factor = (1.0 - (dist - moon_radius) / moon_radius).max(0.0);
                let glow = glow_factor * glow_brightness;

                let idx = py as usize * width + px as usize;
                if idx < color_buf.len() {
                    let existing = color_buf[idx];
                    let er = ((existing >> 16) & 0xFF) as f32 / 255.0;
                    let eg = ((existing >> 8) & 0xFF) as f32 / 255.0;
                    let eb = (existing & 0xFF) as f32 / 255.0;

                    let nr = (er + glow * 1.0).min(1.0);
                    let ng = (eg + glow * 0.98).min(1.0);
                    let nb = (eb + glow * 0.95).min(1.0);

                    color_buf[idx] = rgb_f(nr, ng, nb);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_star_field_generation() {
        let stars = StarField::new(100, 42);
        assert_eq!(stars.stars().len(), 100);

        // All stars should be in valid ranges
        for &(theta, phi, brightness) in stars.stars() {
            assert!(theta >= 0.0 && theta < std::f32::consts::TAU);
            assert!(phi >= 0.0 && phi <= std::f32::consts::FRAC_PI_2);
            assert!(brightness >= 0.2 && brightness <= 1.0);
        }
    }

    #[test]
    fn test_star_field_deterministic() {
        let stars1 = StarField::new(50, 12345);
        let stars2 = StarField::new(50, 12345);
        assert_eq!(stars1.stars(), stars2.stars());
    }

    #[test]
    fn test_stars_not_visible_at_day() {
        let stars = StarField::new(100, 42);
        let mut buf = vec![0u32; 100 * 100];
        render_stars(&mut buf, 100, 100, 0.5, 0.0, 0.0, stars.stars()); // Light = 0.5 (day)
        // Buffer should remain unchanged (all zeros)
        assert!(buf.iter().all(|&p| p == 0));
    }

    #[test]
    fn test_stars_visible_at_night() {
        let stars = StarField::new(100, 42);
        let mut buf = vec![0u32; 100 * 100];
        render_stars(&mut buf, 100, 100, 0.0, 0.0, 0.0, stars.stars()); // Light = 0 (night)
        // Some pixels should have been modified
        assert!(buf.iter().any(|&p| p != 0));
    }

    #[test]
    fn test_moon_not_visible_at_day() {
        let mut buf = vec![0u32; 100 * 100];
        render_moon(&mut buf, 100, 100, 0.5, 0.5, 1.0, 0.0); // Light = 0.5 (day)
        // Buffer should remain unchanged
        assert!(buf.iter().all(|&p| p == 0));
    }

    #[test]
    fn test_moon_visible_at_night() {
        let mut buf = vec![0u32; 100 * 100];
        render_moon(&mut buf, 100, 100, 0.0, 0.5, 1.0, 0.0); // Full moon at night
        // Some pixels should have been modified
        assert!(buf.iter().any(|&p| p != 0));
    }
}
