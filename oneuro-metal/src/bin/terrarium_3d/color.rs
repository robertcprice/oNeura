//! Color conversion and palette utilities.

use super::math::{V3, lerp3};

pub fn rgb(r: u8, g: u8, b: u8) -> u32 {
    ((r as u32) << 16) | ((g as u32) << 8) | b as u32
}

pub fn rgb_f(r: f32, g: f32, b: f32) -> u32 {
    rgb(
        (r * 255.0).clamp(0.0, 255.0) as u8,
        (g * 255.0).clamp(0.0, 255.0) as u8,
        (b * 255.0).clamp(0.0, 255.0) as u8,
    )
}

pub fn u32_to_v3(c: u32) -> V3 {
    [
        ((c >> 16) & 0xff) as f32 / 255.0,
        ((c >> 8) & 0xff) as f32 / 255.0,
        (c & 0xff) as f32 / 255.0,
    ]
}

pub fn blend(dst: u32, src: u32, alpha: f32) -> u32 {
    let a = alpha.clamp(0.0, 1.0);
    let d = u32_to_v3(dst);
    let s = u32_to_v3(src);
    rgb_f(d[0]*(1.0-a)+s[0]*a, d[1]*(1.0-a)+s[1]*a, d[2]*(1.0-a)+s[2]*a)
}

pub fn soil_color_v3(moisture: f32, organic: f32) -> V3 {
    let dry = (180u8, 140, 80);
    let wet = (80, 60, 30);
    let rich = (60, 90, 40);
    let m = moisture.clamp(0.0, 1.0) * 2.0;
    let o = organic.clamp(0.0, 1.0) * 3.0;
    let base_r = dry.0 as f32 + (wet.0 as f32 - dry.0 as f32) * m.min(1.0);
    let base_g = dry.1 as f32 + (wet.1 as f32 - dry.1 as f32) * m.min(1.0);
    let base_b = dry.2 as f32 + (wet.2 as f32 - dry.2 as f32) * m.min(1.0);
    let r = base_r + (rich.0 as f32 - base_r) * o.min(1.0);
    let g = base_g + (rich.1 as f32 - base_g) * o.min(1.0);
    let b = base_b + (rich.2 as f32 - base_b) * o.min(1.0);
    [r / 255.0, g / 255.0, b / 255.0]
}

pub fn plant_color_v3(canopy: f32, vitality: f32) -> V3 {
    let sparse = [100.0/255.0, 160.0/255.0, 60.0/255.0];
    let dense  = [30.0/255.0, 120.0/255.0, 30.0/255.0];
    let stressed = [160.0/255.0, 150.0/255.0, 50.0/255.0];
    let c = canopy.clamp(0.0, 1.0);
    let v = (1.0 - vitality).clamp(0.0, 1.0);
    let base = lerp3(sparse, dense, c);
    lerp3(base, stressed, v)
}
