//! Core software rasterizer — barycentric triangle fill with depth buffer, entity tags, and SSAO.

use oneuro_metal::TerrariumWorld;
use super::math::*;
use super::color::{rgb_f, u32_to_v3, blend};
use super::lighting::sky_color_at;
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

    pub fn clear(&mut self, light: f32) {
        for y in 0..self.height {
            let t = y as f32 / self.height as f32;
            let sky = sky_color_at(light, t);
            for x in 0..self.width {
                let idx = y * self.width + x;
                self.color_buf[idx] = sky;
                self.z_buf[idx] = f32::MAX;
                self.world_pos_buf[idx] = [0.0; 3];
                self.world_normal_buf[idx] = [0.0, 1.0, 0.0];
                self.entity_tag_buf[idx] = EntityTag::None;
            }
        }
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
    pub fn ssao_pass(&mut self) {
        let offsets: [(i32, i32); 8] = [
            (-3, 0), (3, 0), (0, -3), (0, 3),
            (-2, -2), (2, -2), (-2, 2), (2, 2),
        ];
        // Compute AO factor per pixel
        let npx = self.width * self.height;
        let mut ao_buf = vec![0.0f32; npx];
        for y in 0..self.height {
            for x in 0..self.width {
                let idx = y * self.width + x;
                if self.z_buf[idx] >= f32::MAX * 0.9 { continue; }
                let center_z = self.z_buf[idx];
                let mut occlusion = 0.0f32;
                let mut samples = 0.0f32;
                for &(dx, dy) in &offsets {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx >= 0 && nx < self.width as i32 && ny >= 0 && ny < self.height as i32 {
                        let ni = ny as usize * self.width + nx as usize;
                        let nz = self.z_buf[ni];
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
                    ao_buf[idx] = (occlusion / samples).clamp(0.0, 0.55);
                }
            }
        }
        // Apply AO darkening
        for idx in 0..npx {
            if ao_buf[idx] > 0.001 {
                let c = u32_to_v3(self.color_buf[idx]);
                let factor = 1.0 - ao_buf[idx];
                self.color_buf[idx] = rgb_f(c[0] * factor, c[1] * factor, c[2] * factor);
            }
        }
    }

    pub fn shadow_pass(&mut self, world: &TerrariumWorld, sun_dir: V3) {
        let gw = world.config.width;
        let gh = world.config.height;
        let moisture = world.moisture_field();
        let terrain_y_at = |wx: f32, wz: f32| -> f32 {
            let gx = (wx / CELL_SIZE).clamp(0.0, (gw - 1) as f32);
            let gz = (wz / CELL_SIZE).clamp(0.0, (gh - 1) as f32);
            let x0 = gx.floor() as usize;
            let z0 = gz.floor() as usize;
            let x1 = (x0 + 1).min(gw - 1);
            let z1 = (z0 + 1).min(gh - 1);
            let fx = gx - x0 as f32;
            let fz = gz - z0 as f32;
            let m00 = moisture.get(z0 * gw + x0).copied().unwrap_or(0.3);
            let m10 = moisture.get(z0 * gw + x1).copied().unwrap_or(0.3);
            let m01 = moisture.get(z1 * gw + x0).copied().unwrap_or(0.3);
            let m11 = moisture.get(z1 * gw + x1).copied().unwrap_or(0.3);
            (m00 * (1.0-fx)*(1.0-fz) + m10 * fx*(1.0-fz) + m01 * (1.0-fx)*fz + m11 * fx*fz) * HEIGHT_SCALE
        };
        for idx in 0..(self.width * self.height) {
            if self.z_buf[idx] >= f32::MAX * 0.9 { continue; }
            let wp = self.world_pos_buf[idx];
            let wn = self.world_normal_buf[idx];
            let origin = add3(wp, scale3(wn, 0.05));
            let mut in_shadow = false;
            for step in 1..=SHADOW_STEPS {
                let t = step as f32 * SHADOW_STEP_SIZE;
                let sample = add3(origin, scale3(sun_dir, t));
                if sample[1] < terrain_y_at(sample[0], sample[2]) { in_shadow = true; break; }
                if sample[1] > HEIGHT_SCALE * 1.5 + 1.0 { break; }
            }
            if in_shadow {
                let c = u32_to_v3(self.color_buf[idx]);
                self.color_buf[idx] = rgb_f(c[0] * SHADOW_DARKEN, c[1] * SHADOW_DARKEN, c[2] * SHADOW_DARKEN);
            }
        }
    }

    pub fn fog_pass(&mut self, cam_eye: V3, light: f32) {
        for idx in 0..(self.width * self.height) {
            if self.z_buf[idx] >= f32::MAX * 0.9 { continue; }
            let wp = self.world_pos_buf[idx];
            let dist = len3(sub3(wp, cam_eye));
            if dist <= FOG_NEAR { continue; }
            let fog_t = ((dist - FOG_NEAR) / (FOG_FAR - FOG_NEAR)).clamp(0.0, 1.0);
            let y = idx / self.width;
            let screen_y = y as f32 / self.height as f32;
            self.color_buf[idx] = blend(self.color_buf[idx], sky_color_at(light, screen_y), fog_t);
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
