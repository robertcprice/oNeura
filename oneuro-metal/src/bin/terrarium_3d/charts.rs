//! Time-series data collection and sparkline chart rendering.
//!
//! Tracks multiple ecosystem metrics over time and renders inline
//! sparkline graphs in the side panel.

use oneuro_metal::TerrariumWorldSnapshot;
use std::collections::VecDeque;
use super::hud::{draw_rect, draw_text};
use super::color::rgb;
use super::{TOTAL_W, TOTAL_H, PANEL_W, VIEWPORT_W};

const MAX_HISTORY: usize = 200;

/// A single metric time series.
struct Series {
    label: &'static str,
    color: u32,
    data: VecDeque<f32>,
}

impl Series {
    fn new(label: &'static str, color: u32) -> Self {
        Self { label, color, data: VecDeque::with_capacity(MAX_HISTORY + 1) }
    }

    fn push(&mut self, value: f32) {
        self.data.push_back(value);
        if self.data.len() > MAX_HISTORY { self.data.pop_front(); }
    }

    fn min_max(&self) -> (f32, f32) {
        let mut lo = f32::MAX;
        let mut hi = f32::MIN;
        for &v in &self.data {
            if v < lo { lo = v; }
            if v > hi { hi = v; }
        }
        if lo >= hi { (lo - 1.0, hi + 1.0) } else { (lo, hi) }
    }
}

/// Tracks 6 ecosystem metrics and renders them as sparklines.
pub struct ChartPanel {
    series: Vec<Series>,
}

impl ChartPanel {
    pub fn new() -> Self {
        Self {
            series: vec![
                Series::new("Plants", rgb(40, 180, 60)),
                Series::new("Flies", rgb(230, 210, 40)),
                Series::new("Moisture", rgb(80, 156, 228)),
                Series::new("O2", rgb(120, 200, 255)),
                Series::new("CO2", rgb(200, 120, 80)),
                Series::new("Biomass", rgb(160, 220, 100)),
            ],
        }
    }

    /// Record a snapshot's data into all series.
    pub fn record(&mut self, snap: &TerrariumWorldSnapshot) {
        self.series[0].push(snap.plants as f32);
        self.series[1].push(snap.flies as f32);
        self.series[2].push(snap.mean_soil_moisture);
        self.series[3].push(snap.mean_atmospheric_o2);
        self.series[4].push(snap.mean_atmospheric_co2);
        // Approximate biomass from plant count * mean vitality
        let biomass = snap.plants as f32 * snap.mean_cell_vitality;
        self.series[5].push(biomass);
    }

    /// Draw all sparkline charts at the given position in the buffer.
    /// Returns the Y position after the last chart.
    pub fn draw(&self, buffer: &mut [u32], mut y: usize) -> usize {
        let x = VIEWPORT_W + 14;
        draw_text(buffer, TOTAL_W, TOTAL_H, x, y, "METRICS", rgb(210, 214, 220));
        y += 12;

        for series in &self.series {
            if series.data.len() < 3 { continue; }
            y = draw_sparkline(buffer, x, y, series);
            y += 2;
        }
        y
    }
}

/// Draw a single labeled sparkline chart.
fn draw_sparkline(buffer: &mut [u32], x: usize, mut y: usize, series: &Series) -> usize {
    let graph_w = (PANEL_W - 28).min(220);
    let graph_h: usize = 18;

    // Label + current value
    let current = series.data.back().copied().unwrap_or(0.0);
    let label = if current >= 100.0 {
        format!("{} {:.0}", series.label, current)
    } else if current >= 1.0 {
        format!("{} {:.1}", series.label, current)
    } else {
        format!("{} {:.3}", series.label, current)
    };
    draw_text(buffer, TOTAL_W, TOTAL_H, x, y, &label, series.color);
    y += 10;

    // Background
    draw_rect(buffer, TOTAL_W, TOTAL_H, x, y, graph_w, graph_h, rgb(16, 18, 22));

    // Plot line
    let (lo, hi) = series.min_max();
    let range = (hi - lo).max(0.001);
    let n = series.data.len();
    if n > 1 {
        for i in 0..n.min(graph_w) {
            let data_idx = if n > graph_w { i * n / graph_w } else { i };
            let val = series.data.get(data_idx).copied().unwrap_or(lo);
            let normalized = ((val - lo) / range).clamp(0.0, 1.0);
            let py = y + graph_h - 2 - (normalized * (graph_h - 3) as f32) as usize;
            let px = x + (i * graph_w / n.max(1)).min(graph_w - 1);
            if px < TOTAL_W && py < TOTAL_H {
                buffer[py * TOTAL_W + px] = series.color;
                // Thicken line slightly
                if py + 1 < TOTAL_H { buffer[(py + 1) * TOTAL_W + px] = series.color; }
            }
        }
    }

    y + graph_h
}
