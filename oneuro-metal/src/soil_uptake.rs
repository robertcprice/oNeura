//! Native root-zone resource extraction against terrarium soil fields.

use crate::constants::clamp;

#[derive(Debug, Clone)]
pub struct SoilResourceExtraction {
    pub moisture: Vec<f32>,
    pub deep_moisture: Vec<f32>,
    pub dissolved_nutrients: Vec<f32>,
    pub mineral_nitrogen: Vec<f32>,
    pub shallow_nutrients: Vec<f32>,
    pub deep_minerals: Vec<f32>,
    pub water_take: f32,
    pub nutrient_take: f32,
    pub surface_water_take: f32,
    pub deep_water_take: f32,
    pub ammonium_take: f32,
    pub rhizo_nitrate_take: f32,
    pub deep_nitrate_take: f32,
}

fn ensure_len(name: &str, values: &[f32], total: usize) -> Result<(), String> {
    if values.len() != total {
        return Err(format!(
            "{} length mismatch: expected {}, got {}",
            name,
            total,
            values.len()
        ));
    }
    Ok(())
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

#[allow(clippy::too_many_arguments)]
pub fn extract_root_resources_with_layers(
    width: usize,
    height: usize,
    x: i32,
    y: i32,
    radius: i32,
    water_demand: f32,
    nutrient_demand: f32,
    deep_fraction: f32,
    symbiosis_factor: f32,
    root_respiration: f32,
    moisture: &[f32],
    deep_moisture: &[f32],
    dissolved_nutrients: &[f32],
    mineral_nitrogen: &[f32],
    shallow_nutrients: &[f32],
    deep_minerals: &[f32],
    symbiont_biomass: &[f32],
    ammonium_rhizo: &[f32],
    nitrate_rhizo: &[f32],
    nitrate_deep: &[f32],
) -> Result<SoilResourceExtraction, String> {
    let total = width * height;
    for (name, values) in [
        ("moisture", moisture),
        ("deep_moisture", deep_moisture),
        ("dissolved_nutrients", dissolved_nutrients),
        ("mineral_nitrogen", mineral_nitrogen),
        ("shallow_nutrients", shallow_nutrients),
        ("deep_minerals", deep_minerals),
        ("symbiont_biomass", symbiont_biomass),
        ("ammonium_rhizo", ammonium_rhizo),
        ("nitrate_rhizo", nitrate_rhizo),
        ("nitrate_deep", nitrate_deep),
    ] {
        ensure_len(name, values, total)?;
    }

    let mut moisture = moisture.to_vec();
    let mut deep_moisture = deep_moisture.to_vec();
    let mut dissolved_nutrients = dissolved_nutrients.to_vec();
    let mut mineral_nitrogen = mineral_nitrogen.to_vec();
    let mut shallow_nutrients = shallow_nutrients.to_vec();
    let mut deep_minerals = deep_minerals.to_vec();

    let radius = radius.max(1);
    let sigma = ((radius as f32) * 0.8).max(0.75);
    let denom = 2.0 * sigma * sigma;
    let deep_gate = clamp(deep_fraction, 0.0, 1.2);
    let fungal_scale = 9.0 * symbiosis_factor.max(0.1);
    let deep_mineral_scale = 0.04 + 0.16 * clamp(deep_fraction, 0.0, 1.0);
    let rhizo_nitrate_share = clamp(0.54 - deep_fraction * 0.16, 0.20, 0.58);

    let (y0, y1, x0, x1) = window_bounds(width, height, x, y, radius);
    let mut patch_indices = Vec::with_capacity((y1 - y0) * (x1 - x0));
    let mut kernel = Vec::with_capacity((y1 - y0) * (x1 - x0));
    let mut kernel_total = 0.0f32;

    for yy in y0..y1 {
        for xx in x0..x1 {
            let dy = yy as f32 - y as f32;
            let dx = xx as f32 - x as f32;
            let weight = (-(dy * dy + dx * dx) / denom).exp();
            patch_indices.push(yy * width + xx);
            kernel.push(weight);
            kernel_total += weight;
        }
    }

    if kernel_total <= 1.0e-9 {
        return Ok(SoilResourceExtraction {
            moisture,
            deep_moisture,
            dissolved_nutrients,
            mineral_nitrogen,
            shallow_nutrients,
            deep_minerals,
            water_take: 0.0,
            nutrient_take: 0.0,
            surface_water_take: 0.0,
            deep_water_take: 0.0,
            ammonium_take: 0.0,
            rhizo_nitrate_take: 0.0,
            deep_nitrate_take: 0.0,
        });
    }

    for weight in &mut kernel {
        *weight /= kernel_total;
    }

    let mut surface_total = 0.0f32;
    let mut deep_total = 0.0f32;
    let mut fungal_sum = 0.0f32;
    let mut dissolved_total = 0.0f32;
    let mut mineral_total = 0.0f32;
    let mut shallow_total = 0.0f32;
    let mut shallow_raw_total = 0.0f32;
    let mut deep_mineral_total = 0.0f32;
    let mut ammonium_total = 0.0f32;
    let mut rhizo_nitrate_total = 0.0f32;
    let mut deep_nitrate_total = 0.0f32;

    for (&idx, &w) in patch_indices.iter().zip(kernel.iter()) {
        surface_total += moisture[idx] * w;
        deep_total += deep_moisture[idx] * w * deep_gate;
        fungal_sum += symbiont_biomass[idx] * w;
        dissolved_total += dissolved_nutrients[idx] * w;
        mineral_total += mineral_nitrogen[idx] * w;
        shallow_total += shallow_nutrients[idx] * w * 0.28;
        shallow_raw_total += shallow_nutrients[idx] * w;
        deep_mineral_total += deep_minerals[idx] * w * deep_mineral_scale;
        ammonium_total += ammonium_rhizo[idx] * w;
        rhizo_nitrate_total += nitrate_rhizo[idx] * w;
        deep_nitrate_total += nitrate_deep[idx] * w;
    }

    let water_take =
        (surface_total + deep_total).min(water_demand * (0.58 + root_respiration * 0.42));
    let mut surface_water_take = 0.0f32;
    let mut deep_water_take = 0.0f32;
    if water_take > 0.0 {
        let water_total = surface_total + deep_total;
        let surface_share = if water_total > 1.0e-9 {
            surface_total / water_total
        } else {
            0.0
        };
        surface_water_take = water_take * surface_share;
        deep_water_take = water_take - surface_water_take;

        if surface_total > 1.0e-9 {
            for (&idx, &w) in patch_indices.iter().zip(kernel.iter()) {
                let avail = moisture[idx] * w;
                moisture[idx] -= surface_water_take * (avail / surface_total);
            }
        }
        if deep_total > 1.0e-9 {
            for (&idx, &w) in patch_indices.iter().zip(kernel.iter()) {
                let avail = deep_moisture[idx] * w * deep_gate;
                deep_moisture[idx] -= deep_water_take * (avail / deep_total);
            }
        }
    }

    let fungal_bonus = 1.0 + fungal_sum * fungal_scale;
    let nutrient_budget = nutrient_demand * fungal_bonus * root_respiration;
    let ammonium_scaled_total = ammonium_total * (0.55 + fungal_bonus * 0.08);
    let rhizo_nitrate_scaled_total = rhizo_nitrate_total * (0.72 + fungal_bonus * 0.12);
    let deep_nitrate_scaled_total = deep_nitrate_total * (0.72 + fungal_bonus * 0.12);
    let available = dissolved_total
        + mineral_total
        + shallow_total
        + deep_mineral_total * fungal_bonus
        + ammonium_scaled_total
        + rhizo_nitrate_scaled_total
        + deep_nitrate_scaled_total;
    let nutrient_take = available.min(nutrient_budget.max(0.0));

    let mut ammonium_take = 0.0f32;
    let mut rhizo_nitrate_take = 0.0f32;
    let mut deep_nitrate_take = 0.0f32;
    if nutrient_take > 0.0 && available > 1.0e-9 {
        let dissolved_take = nutrient_take * (dissolved_total / available);
        let mineral_take = nutrient_take * (mineral_total / available);
        let shallow_take = nutrient_take * (shallow_total / available);
        let deep_mineral_take = nutrient_take * ((deep_mineral_total * fungal_bonus) / available);
        ammonium_take = nutrient_take * (ammonium_scaled_total / available);
        let nitrate_take_total =
            nutrient_take * ((rhizo_nitrate_scaled_total + deep_nitrate_scaled_total) / available);
        rhizo_nitrate_take = nitrate_take_total * rhizo_nitrate_share;
        deep_nitrate_take = nitrate_take_total - rhizo_nitrate_take;

        if dissolved_total > 1.0e-9 {
            for (&idx, &w) in patch_indices.iter().zip(kernel.iter()) {
                let avail = dissolved_nutrients[idx] * w;
                dissolved_nutrients[idx] -= dissolved_take * (avail / dissolved_total);
            }
        }
        if mineral_total > 1.0e-9 {
            for (&idx, &w) in patch_indices.iter().zip(kernel.iter()) {
                let avail = mineral_nitrogen[idx] * w;
                mineral_nitrogen[idx] -= mineral_take * (avail / mineral_total);
            }
        }
        if shallow_raw_total > 1.0e-9 {
            for (&idx, &w) in patch_indices.iter().zip(kernel.iter()) {
                let avail = shallow_nutrients[idx] * w;
                shallow_nutrients[idx] -= shallow_take * (avail / shallow_raw_total);
            }
        }
        if deep_mineral_total > 1.0e-9 {
            for (&idx, &w) in patch_indices.iter().zip(kernel.iter()) {
                let avail = deep_minerals[idx] * w * deep_mineral_scale;
                deep_minerals[idx] -= deep_mineral_take * (avail / deep_mineral_total);
            }
        }
    }

    Ok(SoilResourceExtraction {
        moisture,
        deep_moisture,
        dissolved_nutrients,
        mineral_nitrogen,
        shallow_nutrients,
        deep_minerals,
        water_take: water_take.max(0.0),
        nutrient_take: nutrient_take.max(0.0),
        surface_water_take: surface_water_take.max(0.0),
        deep_water_take: deep_water_take.max(0.0),
        ammonium_take: ammonium_take.max(0.0),
        rhizo_nitrate_take: rhizo_nitrate_take.max(0.0),
        deep_nitrate_take: deep_nitrate_take.max(0.0),
    })
}

#[cfg(test)]
mod tests {
    use super::extract_root_resources_with_layers;

    #[test]
    fn root_extraction_stays_bounded() {
        let width = 10usize;
        let height = 8usize;
        let total = width * height;
        let result = extract_root_resources_with_layers(
            width,
            height,
            5,
            4,
            2,
            0.03,
            0.02,
            0.4,
            1.0,
            0.9,
            &vec![0.3; total],
            &vec![0.4; total],
            &vec![0.02; total],
            &vec![0.01; total],
            &vec![0.03; total],
            &vec![0.2; total],
            &vec![0.015; total],
            &vec![0.04; total],
            &vec![0.05; total],
            &vec![0.06; total],
        )
        .unwrap();

        assert!(result.water_take >= 0.0);
        assert!(result.nutrient_take >= 0.0);
        assert!(result.moisture.iter().all(|v| *v >= -1.0e-4));
        assert!(result.deep_minerals.iter().all(|v| *v >= -1.0e-4));
    }
}
