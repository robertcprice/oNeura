//! Broad soil hydrology and biogeochemistry updates for the terrarium.
//!
//! This keeps the coarse pool turnover path native while the reactive
//! chemistry substrate remains in the batched-atom terrarium lattice.

use crate::constants::clamp;

#[derive(Debug, Clone)]
pub struct SoilBroadStepResult {
    pub moisture: Vec<f32>,
    pub deep_moisture: Vec<f32>,
    pub dissolved_nutrients: Vec<f32>,
    pub mineral_nitrogen: Vec<f32>,
    pub shallow_nutrients: Vec<f32>,
    pub deep_minerals: Vec<f32>,
    pub organic_matter: Vec<f32>,
    pub litter_carbon: Vec<f32>,
    pub microbial_biomass: Vec<f32>,
    pub symbiont_biomass: Vec<f32>,
    pub root_exudates: Vec<f32>,
    pub decomposition: Vec<f32>,
    pub mineralized: Vec<f32>,
    pub litter_used: Vec<f32>,
    pub exudate_used: Vec<f32>,
    pub organic_used: Vec<f32>,
    pub microbial_turnover: Vec<f32>,
    pub sym_turnover: Vec<f32>,
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

fn diffuse2d(field: &mut [f32], width: usize, height: usize, rate: f32) {
    let mut next = vec![0.0f32; field.len()];
    for y in 0..height {
        let ym = if y == 0 { height - 1 } else { y - 1 };
        let yp = if y + 1 == height { 0 } else { y + 1 };
        for x in 0..width {
            let xm = if x == 0 { width - 1 } else { x - 1 };
            let xp = if x + 1 == width { 0 } else { x + 1 };
            let idx = y * width + x;
            next[idx] = field[idx]
                + (field[ym * width + x]
                    + field[yp * width + x]
                    + field[y * width + xm]
                    + field[y * width + xp]
                    - 4.0 * field[idx])
                    * rate;
        }
    }
    field.copy_from_slice(&next);
}

#[allow(clippy::too_many_arguments)]
pub fn step_soil_broad_pools(
    width: usize,
    height: usize,
    dt: f32,
    light: f32,
    temp_factor: f32,
    water_mask: &[f32],
    canopy_cover: &[f32],
    root_density: &[f32],
    moisture: &[f32],
    deep_moisture: &[f32],
    dissolved_nutrients: &[f32],
    mineral_nitrogen: &[f32],
    shallow_nutrients: &[f32],
    deep_minerals: &[f32],
    organic_matter: &[f32],
    litter_carbon: &[f32],
    microbial_biomass: &[f32],
    symbiont_biomass: &[f32],
    root_exudates: &[f32],
    soil_structure: &[f32],
) -> Result<SoilBroadStepResult, String> {
    let total = width * height;
    for (name, values) in [
        ("water_mask", water_mask),
        ("canopy_cover", canopy_cover),
        ("root_density", root_density),
        ("moisture", moisture),
        ("deep_moisture", deep_moisture),
        ("dissolved_nutrients", dissolved_nutrients),
        ("mineral_nitrogen", mineral_nitrogen),
        ("shallow_nutrients", shallow_nutrients),
        ("deep_minerals", deep_minerals),
        ("organic_matter", organic_matter),
        ("litter_carbon", litter_carbon),
        ("microbial_biomass", microbial_biomass),
        ("symbiont_biomass", symbiont_biomass),
        ("root_exudates", root_exudates),
        ("soil_structure", soil_structure),
    ] {
        ensure_len(name, values, total)?;
    }

    let mut moisture = moisture.to_vec();
    let mut deep_moisture = deep_moisture.to_vec();
    let mut dissolved_nutrients = dissolved_nutrients.to_vec();
    let mut mineral_nitrogen = mineral_nitrogen.to_vec();
    let mut shallow_nutrients = shallow_nutrients.to_vec();
    let mut deep_minerals = deep_minerals.to_vec();
    let mut organic_matter = organic_matter.to_vec();
    let mut litter_carbon = litter_carbon.to_vec();
    let mut microbial_biomass = microbial_biomass.to_vec();
    let mut symbiont_biomass = symbiont_biomass.to_vec();
    let mut root_exudates = root_exudates.to_vec();

    diffuse2d(&mut moisture, width, height, 0.045);
    diffuse2d(&mut deep_moisture, width, height, 0.012);
    diffuse2d(&mut dissolved_nutrients, width, height, 0.020);
    diffuse2d(&mut mineral_nitrogen, width, height, 0.015);
    diffuse2d(&mut symbiont_biomass, width, height, 0.006);

    for idx in 0..total {
        if water_mask[idx] > 0.0 {
            moisture[idx] += water_mask[idx] * (0.00016 * dt);
            deep_moisture[idx] += water_mask[idx] * (0.00008 * dt);
        }
    }

    let mut weathering = vec![0.0f32; total];
    for idx in 0..total {
        let structure = soil_structure[idx];
        let infiltration = (moisture[idx] - (0.34 + structure * 0.14)).max(0.0)
            * (0.0040 * dt)
            * (0.55 + structure);
        moisture[idx] -= infiltration;
        deep_moisture[idx] += infiltration;

        let capillary =
            (deep_moisture[idx] - 0.18).max(0.0) * (0.30 - moisture[idx]).max(0.0) * 0.012 * dt;
        deep_moisture[idx] -= capillary;
        moisture[idx] += capillary;

        let canopy_damp = 1.0 - clamp(canopy_cover[idx] * 0.40, 0.0, 0.55);
        moisture[idx] *= 1.0 - dt * 0.000018 * (0.40 + light) * canopy_damp;
        deep_moisture[idx] *= 1.0 - dt * 0.0000035;

        weathering[idx] = deep_minerals[idx] * (0.000010 * dt) * (0.45 + deep_moisture[idx]);
        deep_minerals[idx] -= weathering[idx];
        dissolved_nutrients[idx] += weathering[idx] * 0.34;
        shallow_nutrients[idx] += weathering[idx] * 0.30;
    }

    let mut decomposition = vec![0.0f32; total];
    let mut mineralized = vec![0.0f32; total];
    let mut litter_used = vec![0.0f32; total];
    let mut exudate_used = vec![0.0f32; total];
    let mut organic_used = vec![0.0f32; total];
    let mut microbial_turnover = vec![0.0f32; total];
    let mut sym_turnover = vec![0.0f32; total];

    for idx in 0..total {
        let substrate =
            litter_carbon[idx] * 1.10 + root_exudates[idx] * 1.35 + organic_matter[idx] * 0.90;
        let moisture_factor = clamp((moisture[idx] + deep_moisture[idx] * 0.35) / 0.48, 0.0, 1.6);
        let oxygen_factor = clamp(1.15 - deep_moisture[idx] * 0.55, 0.35, 1.1);
        let root_factor = 1.0 + root_density[idx] * 0.08;

        let activity = microbial_biomass[idx]
            * substrate
            * moisture_factor
            * temp_factor
            * oxygen_factor
            * root_factor;
        decomposition[idx] = activity * (0.00016 * dt);

        litter_used[idx] = litter_carbon[idx].min(decomposition[idx] * 0.43);
        exudate_used[idx] = root_exudates[idx].min(decomposition[idx] * 0.30);
        organic_used[idx] = organic_matter[idx].min(decomposition[idx] * 0.27);

        litter_carbon[idx] -= litter_used[idx];
        root_exudates[idx] -= exudate_used[idx];
        organic_matter[idx] -= organic_used[idx];

        let immobilization_demand =
            microbial_biomass[idx] * (0.000024 * dt) * (1.0 + root_density[idx] * 0.05);
        let immobilized = dissolved_nutrients[idx].min(immobilization_demand);
        dissolved_nutrients[idx] -= immobilized;

        mineralized[idx] =
            litter_used[idx] * 0.17 + exudate_used[idx] * 0.24 + organic_used[idx] * 0.10;
        let dissolved =
            litter_used[idx] * 0.21 + exudate_used[idx] * 0.15 + organic_used[idx] * 0.18;
        let humified = litter_used[idx] * 0.10 + organic_used[idx] * 0.06;

        let microbial_growth = decomposition[idx] * 0.085 + immobilized * 0.32;
        microbial_turnover[idx] = microbial_biomass[idx]
            * (0.000038 * dt + (deep_moisture[idx] - 0.85).max(0.0) * 0.00003 * dt);
        microbial_biomass[idx] += microbial_growth - microbial_turnover[idx];

        let symbiont_substrate = root_exudates[idx] * (0.65 + root_density[idx] * 0.20);
        let sym_growth = symbiont_substrate
            * (0.00022 * dt)
            * temp_factor
            * clamp(moisture[idx] / 0.35, 0.3, 1.5);
        sym_turnover[idx] = symbiont_biomass[idx] * (0.000028 * dt);
        symbiont_biomass[idx] += sym_growth - sym_turnover[idx];

        mineral_nitrogen[idx] += mineralized[idx] + sym_turnover[idx] * 0.18;
        dissolved_nutrients[idx] += dissolved + weathering[idx] * 0.24;
        shallow_nutrients[idx] += dissolved * 0.20 + mineralized[idx] * 0.36;
        litter_carbon[idx] += microbial_turnover[idx] * 0.42 + sym_turnover[idx] * 0.22;
        organic_matter[idx] += humified + microbial_turnover[idx] * 0.12;

        moisture[idx] = clamp(moisture[idx], 0.0, 1.0);
        deep_moisture[idx] = clamp(deep_moisture[idx], 0.02, 1.2);
        litter_carbon[idx] = clamp(litter_carbon[idx], 0.0, 2.0);
        root_exudates[idx] = clamp(root_exudates[idx], 0.0, 2.0);
        dissolved_nutrients[idx] = clamp(dissolved_nutrients[idx], 0.0, 2.0);
        mineral_nitrogen[idx] = clamp(mineral_nitrogen[idx], 0.0, 2.0);
        microbial_biomass[idx] = clamp(microbial_biomass[idx], 0.001, 2.0);
        symbiont_biomass[idx] = clamp(symbiont_biomass[idx], 0.001, 2.0);
        shallow_nutrients[idx] = clamp(shallow_nutrients[idx], 0.0, 2.0);
        deep_minerals[idx] = clamp(deep_minerals[idx], 0.0, 2.0);
        organic_matter[idx] = clamp(organic_matter[idx], 0.0, 2.0);
    }

    Ok(SoilBroadStepResult {
        moisture,
        deep_moisture,
        dissolved_nutrients,
        mineral_nitrogen,
        shallow_nutrients,
        deep_minerals,
        organic_matter,
        litter_carbon,
        microbial_biomass,
        symbiont_biomass,
        root_exudates,
        decomposition,
        mineralized,
        litter_used,
        exudate_used,
        organic_used,
        microbial_turnover,
        sym_turnover,
    })
}

#[cfg(test)]
mod tests {
    use super::step_soil_broad_pools;

    #[test]
    fn soil_broad_step_stays_bounded() {
        let width = 8usize;
        let height = 6usize;
        let total = width * height;
        let mask = vec![0.0f32; total];
        let canopy = vec![0.02f32; total];
        let root = vec![0.01f32; total];
        let moisture = vec![0.32f32; total];
        let deep_moisture = vec![0.40f32; total];
        let dissolved = vec![0.010f32; total];
        let mineral_n = vec![0.006f32; total];
        let shallow = vec![0.020f32; total];
        let deep_minerals = vec![0.22f32; total];
        let organic = vec![0.030f32; total];
        let litter = vec![0.028f32; total];
        let microbes = vec![0.020f32; total];
        let sym = vec![0.011f32; total];
        let exudates = vec![0.006f32; total];
        let structure = vec![0.55f32; total];

        let result = step_soil_broad_pools(
            width,
            height,
            45.0,
            0.9,
            0.95,
            &mask,
            &canopy,
            &root,
            &moisture,
            &deep_moisture,
            &dissolved,
            &mineral_n,
            &shallow,
            &deep_minerals,
            &organic,
            &litter,
            &microbes,
            &sym,
            &exudates,
            &structure,
        )
        .unwrap();

        assert!(result.moisture.iter().all(|v| *v >= 0.0 && *v <= 1.0));
        assert!(result.deep_moisture.iter().all(|v| *v >= 0.02 && *v <= 1.2));
        assert!(result
            .microbial_biomass
            .iter()
            .all(|v| *v >= 0.001 && *v <= 2.0));
        assert!(result.decomposition.iter().all(|v| *v >= 0.0));
    }
}
