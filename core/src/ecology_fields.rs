//! Native helpers for ecology spatial fields.
//!
//! These routines keep terrarium-wide spatial accumulation out of Python loops.
//! The math intentionally mirrors the Python ecology scaffolding exactly:
//! bounded windows, Gaussian radial kernels, and per-source normalization.

#[derive(Debug, Clone, Copy)]
pub struct RadialSource {
    pub x: i32,
    pub y: i32,
    pub radius: i32,
    pub amplitude: f32,
    pub sharpness: f32,
}

impl RadialSource {
    pub fn from_chunk(chunk: &[f32]) -> Result<Self, String> {
        if chunk.len() != 5 {
            return Err("radial source chunks must have 5 values".to_string());
        }
        Ok(Self {
            x: chunk[0].round() as i32,
            y: chunk[1].round() as i32,
            radius: chunk[2].round().max(0.0) as i32,
            amplitude: chunk[3],
            sharpness: chunk[4],
        })
    }
}

fn parse_sources(flat_sources: &[f32]) -> Result<Vec<RadialSource>, String> {
    if !flat_sources.len().is_multiple_of(5) {
        return Err(format!(
            "expected flattened radial sources in groups of 5, got {} values",
            flat_sources.len()
        ));
    }
    flat_sources
        .chunks_exact(5)
        .map(RadialSource::from_chunk)
        .collect()
}

fn window_bounds(
    width: usize,
    height: usize,
    x: i32,
    y: i32,
    radius: i32,
) -> (usize, usize, usize, usize) {
    let y0 = y.saturating_sub(radius).max(0) as usize;
    let y1 = (y + radius + 1).min(height as i32).max(0) as usize;
    let x0 = x.saturating_sub(radius).max(0) as usize;
    let x1 = (x + radius + 1).min(width as i32).max(0) as usize;
    (y0, y1, x0, x1)
}

fn accumulate_source(field: &mut [f32], width: usize, height: usize, source: RadialSource) {
    if source.amplitude.abs() <= 1.0e-12 {
        return;
    }
    let radius = source.radius.max(0);
    let sigma = ((radius as f32) * source.sharpness).max(0.75);
    let denom = 2.0 * sigma * sigma;
    let (y0, y1, x0, x1) = window_bounds(width, height, source.x, source.y, radius);
    if y0 >= y1 || x0 >= x1 {
        return;
    }

    let mut weights = Vec::with_capacity((y1 - y0) * (x1 - x0));
    let mut total = 0.0f32;
    for yy in y0..y1 {
        for xx in x0..x1 {
            let dy = yy as f32 - source.y as f32;
            let dx = xx as f32 - source.x as f32;
            let weight = (-(dy * dy + dx * dx) / denom).exp();
            total += weight;
            weights.push(weight);
        }
    }

    if total <= 1.0e-9 {
        return;
    }

    let scale = source.amplitude / total;
    let mut idx = 0usize;
    for yy in y0..y1 {
        let row_start = yy * width;
        for xx in x0..x1 {
            field[row_start + xx] += weights[idx] * scale;
            idx += 1;
        }
    }
}

pub fn build_radial_field(
    width: usize,
    height: usize,
    flat_sources: &[f32],
) -> Result<Vec<f32>, String> {
    let mut field = vec![0.0f32; width * height];
    for source in parse_sources(flat_sources)? {
        accumulate_source(&mut field, width, height, source);
    }
    Ok(field)
}

pub fn build_dual_radial_fields(
    width: usize,
    height: usize,
    canopy_sources: &[f32],
    root_sources: &[f32],
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let canopy = build_radial_field(width, height, canopy_sources)?;
    let root = build_radial_field(width, height, root_sources)?;
    Ok((canopy, root))
}

#[cfg(test)]
mod tests {
    use super::{build_dual_radial_fields, build_radial_field};

    #[test]
    fn radial_field_is_normalized_to_amplitude() {
        let field = build_radial_field(16, 12, &[7.0, 5.0, 3.0, 2.5, 0.72]).unwrap();
        let total: f32 = field.iter().copied().sum();
        assert!((total - 2.5).abs() < 1.0e-4);
    }

    #[test]
    fn dual_field_builder_keeps_sources_separate() {
        let (canopy, root) = build_dual_radial_fields(
            10,
            10,
            &[4.0, 4.0, 2.0, 1.0, 0.70],
            &[6.0, 6.0, 2.0, 0.5, 0.95],
        )
        .unwrap();
        let canopy_total: f32 = canopy.iter().copied().sum();
        let root_total: f32 = root.iter().copied().sum();
        assert!((canopy_total - 1.0).abs() < 1.0e-4);
        assert!((root_total - 0.5).abs() < 1.0e-4);
    }
}
