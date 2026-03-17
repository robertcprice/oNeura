//! Core software rasterizer — barycentric triangle fill with depth buffer, entity tags, and SSAO.
//! Multi-threaded using rayon for parallel post-processing passes.

use rayon::prelude::*;
use super::math::*;
use super::color::{rgb_f, u32_to_v3, blend};
use super::lighting::{sky_color_at, sky_gradient, moon_position};
use super::mesh::{Triangle, EntityTag};
use super::{NEAR, HEIGHT_SCALE, FOG_NEAR, FOG_FAR, CELL_SIZE, SHADOW_STEPS, SHADOW_STEP_SIZE, SHADOW_DARKEN};

pub struct Rasterizer {
    pub width: usize,
    pub height: usize,
    pub color_buf: Vec<u32>,
    z_buf: Vec<f32>,
    world_pos_buf: Vec<V3>,
    world_normal_buf: Vec<V3>,
    entity_tag_buf: Vec<EntityTag>,
}

impl Rasterizer {
    pub fn new(w: usize, h: usize) -> Self {
        Self {
            width: w, height: h,
            color_buf: vec![0; w * h],
            z_buf: vec![f32::MAX; w * h],
            world_pos_buf: vec![[0.0; 3]; w * h],
            world_normal_buf: vec![[0.0, 1.0, 0.0]; w * h],
            entity_tag_buf: vec![EntityTag::None; w * h],
        }
    }

    /// Clear all buffers with sky gradient — multi-threaded using rayon chunks.
    pub fn clear(&mut self, light: f32) {
        let (horizon, zenith) = sky_gradient(light);
        let width = self.width;
        let height = self.height;

        // Pre-compute sky colors for each row
        let row_sky_colors: Vec<u32> = (0..height)
            .map(|y| {
                let t = y as f32 / height as f32;
                let sky_c = lerp3(horizon, zenith, t);
                rgb_f(sky_c[0], sky_c[1], sky_c[2])
            })
            .collect();

        // Clear color buffer in parallel rows
        self.color_buf
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(y, row)| {
                let sky = row_sky_colors[y];
                row.fill(sky);
            });

        // Clear other buffers in parallel
        self.z_buf.par_iter_mut().for_each(|z| *z = f32::MAX);
        self.world_pos_buf.par_iter_mut().for_each(|p| *p = [0.0; 3]);
        self.world_normal_buf.par_iter_mut().for_each(|n| *n = [0.0, 1.0, 0.0]);
        self.entity_tag_buf.par_iter_mut().for_each(|t| *t = EntityTag::None);
    }

    /// Return the entity tag at the given screen pixel, for click-to-select.
    pub fn tag_at(&self, x: usize, y: usize) -> EntityTag {
        if x < self.width && y < self.height {
            self.entity_tag_buf[y * self.width + x]
        } else {
            EntityTag::None
        }
    }

    pub fn rasterize(&mut self, tris: &[Triangle], mvp: &M4, cam_eye: V3, sun_dir: V3, sun_col: V3, realistic: bool) {
        for tri in tris {
            let mut clip = [[0.0f32; 4]; 3];
            let mut screen = [[0.0f32; 3]; 3];
            let mut all_behind = true;
            for i in 0..3 {
                let p = tri.v[i].pos;
                clip[i] = transform4(mvp, [p[0], p[1], p[2], 1.0]);
                if clip[i][3] > NEAR { all_behind = false; }
            }
            if all_behind { continue; }
            for i in 0..3 {
                let w = clip[i][3].max(0.001);
                let inv_w = 1.0 / w;
                screen[i][0] = (clip[i][0] * inv_w * 0.5 + 0.5) * self.width as f32;
                screen[i][1] = (1.0 - (clip[i][1] * inv_w * 0.5 + 0.5)) * self.height as f32;
                screen[i][2] = clip[i][2] * inv_w;
            }
            let min_x = screen.iter().map(|s| s[0]).fold(f32::MAX, f32::min).max(0.0) as i32;
            let max_x = screen.iter().map(|s| s[0]).fold(f32::MIN, f32::max).min(self.width as f32 - 1.0) as i32;
            let min_y = screen.iter().map(|s| s[1]).fold(f32::MAX, f32::min).max(0.0) as i32;
            let max_y = screen.iter().map(|s| s[1]).fold(f32::MIN, f32::max).min(self.height as f32 - 1.0) as i32;
            if min_x > max_x || min_y > max_y { continue; }
            let dx01 = screen[1][0] - screen[0][0];
            let dy01 = screen[1][1] - screen[0][1];
            let dx12 = screen[2][0] - screen[1][0];
            let dy12 = screen[2][1] - screen[1][1];
            let dx20 = screen[0][0] - screen[2][0];
            let dy20 = screen[0][1] - screen[2][1];
            let area = dx01 * dy12 - dx12 * dy01;
            if area.abs() < 0.001 { continue; }
            let inv_area = 1.0 / area;
            for py in min_y..=max_y {
                let fy = py as f32 + 0.5;
                for px in min_x..=max_x {
                    let fx = px as f32 + 0.5;
                    let w0 = dx12 * (fy - screen[1][1]) - dy12 * (fx - screen[1][0]);
                    let w1 = dx20 * (fy - screen[2][1]) - dy20 * (fx - screen[2][0]);
                    let w2 = dx01 * (fy - screen[0][1]) - dy01 * (fx - screen[0][0]);
                    if area > 0.0 {
                        if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 { continue; }
                    } else {
                        if w0 > 0.0 || w1 > 0.0 || w2 > 0.0 { continue; }
                    }
                    let b0 = w0 * inv_area;
                    let b1 = w1 * inv_area;
                    let b2 = w2 * inv_area;
                    let z = screen[0][2] * b0 + screen[1][2] * b1 + screen[2][2] * b2;
                    let idx = (py as usize) * self.width + (px as usize);
                    if z >= self.z_buf[idx] { continue; }
                    self.z_buf[idx] = z;
                    let wp = add3(add3(scale3(tri.v[0].pos, b0), scale3(tri.v[1].pos, b1)), scale3(tri.v[2].pos, b2));
                    let wn = normalize3(add3(add3(scale3(tri.v[0].normal, b0), scale3(tri.v[1].normal, b1)), scale3(tri.v[2].normal, b2)));
                    self.world_pos_buf[idx] = wp;
                    self.world_normal_buf[idx] = wn;
                    self.entity_tag_buf[idx] = tri.tag;
                    let base_color = add3(add3(scale3(tri.v[0].color, b0), scale3(tri.v[1].color, b1)), scale3(tri.v[2].color, b2));
                    if !realistic {
                        self.color_buf[idx] = rgb_f(base_color[0], base_color[1], base_color[2]);
                        continue;
                    }
                    let n_dot_l = dot3(wn, sun_dir).max(0.0);
                    let shininess = tri.v[0].shininess * b0 + tri.v[1].shininess * b1 + tri.v[2].shininess * b2;
                    let ambient = 0.28;
                    let diffuse = n_dot_l * 0.65;
                    let view_dir = normalize3(sub3(cam_eye, wp));
                    let half_dir = normalize3(add3(sun_dir, view_dir));
                    let spec = dot3(wn, half_dir).max(0.0).powf(shininess) * 0.3;
                    let lit = [
                        base_color[0] * (ambient + diffuse * sun_col[0]) + spec * sun_col[0],
                        base_color[1] * (ambient + diffuse * sun_col[1]) + spec * sun_col[1],
                        base_color[2] * (ambient + diffuse * sun_col[2]) + spec * sun_col[2],
                    ];
                    self.color_buf[idx] = rgb_f(lit[0].min(1.0), lit[1].min(1.0), lit[2].min(1.0));
                }
            }
        }
    }

    /// Screen-space ambient occlusion — samples depth neighbors to darken crevices and contact areas.
    /// Multi-threaded using rayon.
    pub fn ssao_pass(&mut self) {
        let offsets: [(i32, i32); 8] = [
            (-3, 0), (3, 0), (0, -3), (0, 3),
            (-2, -2), (2, -2), (-2, 2), (2, 2),
        ];
        let npx = self.width * self.height;
        let width = self.width;
        let height = self.height;

        // Extract z_buf reference before parallel closure
        let z_buf = &self.z_buf;

        // Compute AO factors in parallel
        let ao_buf: Vec<f32> = (0..npx)
            .into_par_iter()
            .map(|idx| {
                let z_val = z_buf[idx];
                if z_val >= f32::MAX * 0.9 {
                    return 0.0;
                }
                let y = idx / width;
                let x = idx % width;
                let center_z = z_val;
                let mut occlusion = 0.0f32;
                let mut samples = 0.0f32;
                for &(dx, dy) in &offsets {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                        let ni = ny as usize * width + nx as usize;
                        let nz = z_buf[ni];
                        if nz < f32::MAX * 0.9 {
                            let diff = center_z - nz;
                            if diff > 0.0005 && diff < 0.06 {
                                occlusion += (diff / 0.06).min(1.0);
                            }
                        }
                        samples += 1.0;
                    }
                }
                if samples > 0.0 {
                    (occlusion / samples).clamp(0.0, 0.55)
                } else {
                    0.0
                }
            })
            .collect();

        // Apply AO darkening in parallel
        self.color_buf
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, pixel)| {
                let ao = ao_buf[idx];
                if ao > 0.001 {
                    let c = u32_to_v3(*pixel);
                    let factor = 1.0 - ao;
                    *pixel = rgb_f(c[0] * factor, c[1] * factor, c[2] * factor);
                }
            });
    }

    /// Shadow raycasting — parallel per-pixel heightfield march with rayon.
    pub fn shadow_pass(&mut self, moisture: &[f32], gw: usize, gh: usize, sun_dir: V3) {
        let width = self.width;
        let z_buf = &self.z_buf;
        let world_pos = &self.world_pos_buf;
        let world_normal = &self.world_normal_buf;
        let npx = width * self.height;

        // Compute shadow flags in parallel
        let shadow_flags: Vec<bool> = (0..npx).into_par_iter().map(|idx| {
            if z_buf[idx] >= f32::MAX * 0.9 { return false; }
            let wp = world_pos[idx];
            let wn = world_normal[idx];
            let origin = add3(wp, scale3(wn, 0.05));
            for step in 1..=SHADOW_STEPS {
                let t = step as f32 * SHADOW_STEP_SIZE;
                let sample = add3(origin, scale3(sun_dir, t));
                let gx_f = (sample[0] / CELL_SIZE).clamp(0.0, (gw - 1) as f32);
                let gz_f = (sample[2] / CELL_SIZE).clamp(0.0, (gh - 1) as f32);
                let x0 = gx_f.floor() as usize;
                let z0 = gz_f.floor() as usize;
                let x1 = (x0 + 1).min(gw - 1);
                let z1 = (z0 + 1).min(gh - 1);
                let fx = gx_f - x0 as f32;
                let fz = gz_f - z0 as f32;
                let m00 = moisture.get(z0 * gw + x0).copied().unwrap_or(0.3);
                let m10 = moisture.get(z0 * gw + x1).copied().unwrap_or(0.3);
                let m01 = moisture.get(z1 * gw + x0).copied().unwrap_or(0.3);
                let m11 = moisture.get(z1 * gw + x1).copied().unwrap_or(0.3);
                let terrain_h = (m00*(1.0-fx)*(1.0-fz) + m10*fx*(1.0-fz) + m01*(1.0-fx)*fz + m11*fx*fz) * HEIGHT_SCALE;
                if sample[1] < terrain_h { return true; }
                if sample[1] > HEIGHT_SCALE * 1.5 + 1.0 { return false; }
            }
            false
        }).collect();

        // Apply shadow darkening in parallel
        self.color_buf.par_iter_mut().zip(shadow_flags.par_iter()).for_each(|(c, &in_shadow)| {
            if in_shadow {
                let cv = u32_to_v3(*c);
                *c = rgb_f(cv[0] * SHADOW_DARKEN, cv[1] * SHADOW_DARKEN, cv[2] * SHADOW_DARKEN);
            }
        });
    }

    /// Distance fog — parallel per-row blending toward sky color.
    pub fn fog_pass(&mut self, cam_eye: V3, light: f32) {
        let width = self.width;
        let height = self.height;
        let z_buf = &self.z_buf;
        let world_pos = &self.world_pos_buf;

        self.color_buf.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
            let screen_y = y as f32 / height as f32;
            let sky = sky_color_at(light, screen_y);
            for x in 0..width {
                let idx = y * width + x;
                if z_buf[idx] >= f32::MAX * 0.9 { continue; }
                let wp = world_pos[idx];
                let dist = len3(sub3(wp, cam_eye));
                if dist <= FOG_NEAR { continue; }
                let fog_t = ((dist - FOG_NEAR) / (FOG_FAR - FOG_NEAR)).clamp(0.0, 1.0);
                row[x] = blend(row[x], sky, fog_t);
            }
        });
    }

    /// Draw procedural stars on the sky background during nighttime.
    /// Stars twinkle using a frame-based brightness modulation.
    pub fn draw_stars(&mut self, stars: &[(f32, f32, f32)], darkness: f32, frame: u64) {
        if darkness < 0.3 {
            return;
        }
        let w = self.width;
        let h = self.height;
        for (i, &(x_norm, y_norm, base_bright)) in stars.iter().enumerate() {
            let px = (x_norm * w as f32) as i32;
            let py = (y_norm * h as f32) as i32;
            if px < 0 || px >= w as i32 || py < 0 || py >= h as i32 {
                continue;
            }
            // Twinkle: per-star phase offset modulated by frame counter
            let twinkle_phase = (frame as f32 * 0.07 + i as f32 * 2.37).sin() * 0.5 + 0.5;
            let twinkle = 0.7 + 0.3 * twinkle_phase;
            let brightness = base_bright * darkness * twinkle;
            if brightness < 0.01 {
                continue;
            }
            let idx = py as usize * w + px as usize;
            // Additive blend with existing sky pixel
            let existing = self.color_buf[idx];
            let er = ((existing >> 16) & 0xFF) as f32 / 255.0;
            let eg = ((existing >> 8) & 0xFF) as f32 / 255.0;
            let eb = (existing & 0xFF) as f32 / 255.0;
            let nr = (er + brightness * 0.95).min(1.0);
            let ng = (eg + brightness * 0.97).min(1.0);
            let nb = (eb + brightness * 1.0).min(1.0);
            self.color_buf[idx] = rgb_f(nr, ng, nb);
        }
    }

    /// Draw a moon disc with phase-based illumination on the sky background.
    /// The terminator (shadow cutoff) shifts based on `lunar_phase`:
    /// < 0.5 waxing (right side lit), > 0.5 waning (left side lit).
    pub fn draw_moon(&mut self, lunar_phase: f32, moonlight: f32, light: f32) {
        if moonlight < 0.1 {
            return;
        }
        let darkness = (1.0 - light * 2.0).clamp(0.0, 1.0);
        if darkness < 0.2 {
            return;
        }
        let w = self.width;
        let h = self.height;
        let (azimuth, elevation) = moon_position(lunar_phase, light);

        // Map azimuth/elevation to screen pixel position
        let screen_x = ((azimuth / std::f32::consts::TAU).fract() * w as f32) as i32;
        let screen_y = ((1.0 - elevation / std::f32::consts::FRAC_PI_2) * 0.5 * h as f32) as i32;

        let radius: i32 = (w as f32 * 0.012).clamp(8.0, 12.0) as i32;
        let radius_sq = (radius * radius) as f32;
        let moon_color: V3 = [1.0, 0.98, 0.85];

        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let dist_sq = (dx * dx + dy * dy) as f32;
                if dist_sq > radius_sq {
                    continue;
                }
                let px = screen_x + dx;
                let py = screen_y + dy;
                if px < 0 || px >= w as i32 || py < 0 || py >= h as i32 {
                    continue;
                }

                // Phase-based illumination: shadow cutoff across disc
                let nx = dx as f32 / radius as f32; // -1..1 across disc
                let lit = if lunar_phase < 0.5 {
                    // Waxing: right side illuminated. Terminator moves left-to-right.
                    let cutoff = -1.0 + lunar_phase * 4.0; // -1..1 as phase goes 0..0.5
                    ((nx - cutoff) * 3.0).clamp(0.0, 1.0)
                } else {
                    // Waning: left side illuminated. Terminator moves right-to-left.
                    let cutoff = 1.0 - (lunar_phase - 0.5) * 4.0; // 1..-1 as phase goes 0.5..1
                    ((cutoff - nx) * 3.0).clamp(0.0, 1.0)
                };

                if lit < 0.01 {
                    continue;
                }

                // Slight limb darkening: dimmer at edge of disc
                let center_dist = dist_sq.sqrt() / radius as f32;
                let limb = 1.0 - center_dist * 0.25;
                let intensity = moonlight * darkness * lit * limb;

                let idx = py as usize * w + px as usize;
                let existing = self.color_buf[idx];
                let er = ((existing >> 16) & 0xFF) as f32 / 255.0;
                let eg = ((existing >> 8) & 0xFF) as f32 / 255.0;
                let eb = (existing & 0xFF) as f32 / 255.0;
                let nr = (er + moon_color[0] * intensity).min(1.0);
                let ng = (eg + moon_color[1] * intensity).min(1.0);
                let nb = (eb + moon_color[2] * intensity).min(1.0);
                self.color_buf[idx] = rgb_f(nr, ng, nb);
            }
        }
    }

    /// Draw a selection highlight outline around pixels matching the given entity tag.
    pub fn draw_selection_outline(&mut self, tag: EntityTag) {
        if tag == EntityTag::None { return; }
        let w = self.width;
        let h = self.height;
        let highlight = rgb_f(1.0, 1.0, 1.0);
        // Find border pixels: pixels with matching tag that have a non-matching neighbor
        for y in 1..(h - 1) {
            for x in 1..(w - 1) {
                let idx = y * w + x;
                if self.entity_tag_buf[idx] != tag { continue; }
                let is_edge =
                    self.entity_tag_buf[idx - 1] != tag ||
                    self.entity_tag_buf[idx + 1] != tag ||
                    self.entity_tag_buf[idx - w] != tag ||
                    self.entity_tag_buf[idx + w] != tag;
                if is_edge {
                    self.color_buf[idx] = highlight;
                }
            }
        }
    }
}
