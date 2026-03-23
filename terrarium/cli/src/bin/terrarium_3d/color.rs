//! Color conversion and palette utilities, including terrain overlay heatmaps.

use super::math::{lerp3, V3};

/// Terrain visualization mode — number keys 1-8 switch between overlays.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OverlayMode {
    Default,           // 1 — soil color from moisture + organic
    Moisture,          // 2 — blue-green heatmap of soil moisture
    Temperature,       // 3 — cool blue → warm red temperature gradient
    Organic,           // 4 — dark-to-green organic matter density
    Chemistry,         // 5 — magenta-to-cyan soil chemistry activity
    Elevation,         // 6 — topographic elevation bands
    SubstrateO2,       // 7 — real O2 concentration from substrate grid
    SubstrateGlucose,  // 8 — real glucose concentration from substrate grid
    SubstrateCO2,      // 9 — real CO2 concentration from substrate grid
    SubstrateNitrogen, // 0 — real nitrogen concentration from substrate grid
}

impl OverlayMode {
    /// Cycle to next overlay mode (backtick key).
    pub fn next(&self) -> Self {
        match self {
            OverlayMode::Default => OverlayMode::Moisture,
            OverlayMode::Moisture => OverlayMode::Temperature,
            OverlayMode::Temperature => OverlayMode::Organic,
            OverlayMode::Organic => OverlayMode::Chemistry,
            OverlayMode::Chemistry => OverlayMode::Elevation,
            OverlayMode::Elevation => OverlayMode::SubstrateO2,
            OverlayMode::SubstrateO2 => OverlayMode::SubstrateGlucose,
            OverlayMode::SubstrateGlucose => OverlayMode::SubstrateCO2,
            OverlayMode::SubstrateCO2 => OverlayMode::SubstrateNitrogen,
            OverlayMode::SubstrateNitrogen => OverlayMode::Default,
        }
    }
}

impl OverlayMode {
    pub fn label(&self) -> &'static str {
        match self {
            OverlayMode::Default => "DEFAULT",
            OverlayMode::Moisture => "MOISTURE",
            OverlayMode::Temperature => "TEMPERATURE",
            OverlayMode::Organic => "ORGANIC",
            OverlayMode::Chemistry => "CHEMISTRY",
            OverlayMode::Elevation => "ELEVATION",
            OverlayMode::SubstrateO2 => "SUBSTRATE O2",
            OverlayMode::SubstrateGlucose => "SUBSTRATE GLUCOSE",
            OverlayMode::SubstrateCO2 => "SUBSTRATE CO2",
            OverlayMode::SubstrateNitrogen => "SUBSTRATE N",
        }
    }
}

/// Heatmap: blue (cold=0) → cyan → green → yellow → red (hot=1).
pub fn heatmap_v3(t: f32) -> V3 {
    let t = t.clamp(0.0, 1.0);
    if t < 0.25 {
        let s = t / 0.25;
        [0.0, s, 1.0] // blue → cyan
    } else if t < 0.5 {
        let s = (t - 0.25) / 0.25;
        [0.0, 1.0, 1.0 - s] // cyan → green
    } else if t < 0.75 {
        let s = (t - 0.5) / 0.25;
        [s, 1.0, 0.0] // green → yellow
    } else {
        let s = (t - 0.75) / 0.25;
        [1.0, 1.0 - s, 0.0] // yellow → red
    }
}

/// Elevation bands — topographic map style with contour coloring.
pub fn elevation_v3(height: f32, max_height: f32) -> V3 {
    let t = (height / max_height.max(0.01)).clamp(0.0, 1.0);
    if t < 0.3 {
        let s = t / 0.3;
        lerp3([0.15, 0.4, 0.1], [0.3, 0.55, 0.15], s)
    } else if t < 0.6 {
        let s = (t - 0.3) / 0.3;
        lerp3([0.3, 0.55, 0.15], [0.55, 0.4, 0.2], s)
    } else if t < 0.85 {
        let s = (t - 0.6) / 0.25;
        lerp3([0.55, 0.4, 0.2], [0.7, 0.65, 0.55], s)
    } else {
        let s = (t - 0.85) / 0.15;
        lerp3([0.7, 0.65, 0.55], [0.95, 0.95, 0.95], s)
    }
}

/// Chemistry overlay — magenta (low activity) → cyan (high activity).
pub fn chemistry_v3(activity: f32) -> V3 {
    let t = activity.clamp(0.0, 1.0);
    lerp3([0.4, 0.1, 0.5], [0.1, 0.8, 0.9], t)
}

/// Organic matter overlay — dark soil → vibrant green.
pub fn organic_v3(organic: f32) -> V3 {
    let t = organic.clamp(0.0, 1.0);
    lerp3([0.12, 0.08, 0.04], [0.2, 0.7, 0.15], t)
}

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
    rgb_f(
        d[0] * (1.0 - a) + s[0] * a,
        d[1] * (1.0 - a) + s[1] * a,
        d[2] * (1.0 - a) + s[2] * a,
    )
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
    let sparse = [100.0 / 255.0, 160.0 / 255.0, 60.0 / 255.0];
    let dense = [30.0 / 255.0, 120.0 / 255.0, 30.0 / 255.0];
    let stressed = [160.0 / 255.0, 150.0 / 255.0, 50.0 / 255.0];
    let c = canopy.clamp(0.0, 1.0);
    let v = (1.0 - vitality).clamp(0.0, 1.0);
    let base = lerp3(sparse, dense, c);
    lerp3(base, stressed, v)
}

/// O2 overlay: deep blue (anoxic=0) -> sky blue -> white (saturated=1).
pub fn substrate_o2_v3(o2: f32) -> V3 {
    let t = o2.clamp(0.0, 1.0);
    if t < 0.5 {
        lerp3([0.05, 0.05, 0.2], [0.2, 0.5, 0.9], t * 2.0)
    } else {
        lerp3([0.2, 0.5, 0.9], [0.9, 0.95, 1.0], (t - 0.5) * 2.0)
    }
}

/// Glucose overlay: pale yellow (depleted=0) -> amber -> deep orange (concentrated=1).
pub fn substrate_glucose_v3(gluc: f32) -> V3 {
    let t = gluc.clamp(0.0, 1.0);
    if t < 0.5 {
        lerp3([0.2, 0.18, 0.1], [0.8, 0.6, 0.1], t * 2.0)
    } else {
        lerp3([0.8, 0.6, 0.1], [1.0, 0.4, 0.0], (t - 0.5) * 2.0)
    }
}

/// CO2 overlay: green (low=0) -> yellow -> red (high=1).
pub fn substrate_co2_v3(co2: f32) -> V3 {
    let t = co2.clamp(0.0, 1.0);
    if t < 0.5 {
        lerp3([0.1, 0.3, 0.1], [0.7, 0.7, 0.1], t * 2.0)
    } else {
        lerp3([0.7, 0.7, 0.1], [0.9, 0.2, 0.1], (t - 0.5) * 2.0)
    }
}

/// Nitrogen overlay: brown (depleted=0) -> lime green (rich=1).
pub fn substrate_nitrogen_v3(n: f32) -> V3 {
    let t = n.clamp(0.0, 1.0);
    if t < 0.5 {
        lerp3([0.25, 0.15, 0.08], [0.4, 0.5, 0.15], t * 2.0)
    } else {
        lerp3([0.4, 0.5, 0.15], [0.3, 0.85, 0.2], (t - 0.5) * 2.0)
    }
}
