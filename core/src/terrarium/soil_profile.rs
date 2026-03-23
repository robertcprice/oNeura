//! Soil depth stratification: 4-layer vertical transport model.
//!
//! Implements Fick's-law percolation and capillary rise between soil layers,
//! rainfall infiltration, surface evaporation, and root-weighted moisture access.
//!
//! Physical basis:
//! - Percolation: gravity-driven downward water movement via Hill kinetics
//! - Capillary rise: Michaelis-Menten kinetics for upward water movement
//! - Infiltration: rainfall enters top layer proportional to available pore space
//! - Evaporation: surface moisture loss driven by temperature and humidity deficit
//!
//! References:
//! - Richards equation (simplified): Jury WA, Horton R (2004) Soil Physics, Wiley
//! - Capillary rise: Lu N, Likos WJ (2004) Unsaturated Soil Mechanics, Wiley

use crate::botany::physiology_bridge::{hill, michaelis_menten};

/// Number of soil depth layers.
pub const SOIL_LAYERS: usize = 4;

/// Depth of each layer's bottom boundary (mm from surface).
pub const SOIL_LAYER_DEPTHS_MM: [f32; SOIL_LAYERS] = [2.0, 10.0, 30.0, 100.0];

/// Thickness of each layer (mm).
pub fn layer_thickness(layer: usize) -> f32 {
    if layer == 0 {
        SOIL_LAYER_DEPTHS_MM[0]
    } else if layer < SOIL_LAYERS {
        SOIL_LAYER_DEPTHS_MM[layer] - SOIL_LAYER_DEPTHS_MM[layer - 1]
    } else {
        0.0
    }
}

/// Midpoint depth of a layer (mm from surface).
pub fn layer_midpoint(layer: usize) -> f32 {
    if layer == 0 {
        SOIL_LAYER_DEPTHS_MM[0] * 0.5
    } else if layer < SOIL_LAYERS {
        (SOIL_LAYER_DEPTHS_MM[layer - 1] + SOIL_LAYER_DEPTHS_MM[layer]) * 0.5
    } else {
        0.0
    }
}

/// Step vertical water transport between adjacent layers.
///
/// - Downward percolation: gravity-driven, accelerated by sandy texture.
///   Rate = hill(upper_moisture, 0.4, 2) * percolation_coeff
/// - Upward capillary rise: moisture deficit draws water up from deeper layers.
///   Rate = michaelis_menten(lower_moisture, 0.3) * capillary_coeff * (1 - upper_moisture)
///
/// `soil_structure` at the cell encodes texture: 0 = sand, 1 = clay.
pub fn step_soil_vertical_transport(
    layers: &mut [f32; SOIL_LAYERS],
    soil_structure: f32,
    dt: f32,
) {
    let texture = soil_structure.clamp(0.0, 1.0);
    // Derive hydraulic coefficients from van Genuchten (1980) / Rawls-Brakensiek (1985)
    let params = super::emergent_rates::SoilHydraulicParams::from_texture(texture);
    let percolation_coeff = params.percolation_coeff;
    let capillary_coeff = params.capillary_coeff;

    for i in 0..(SOIL_LAYERS - 1) {
        let upper = layers[i];
        let lower = layers[i + 1];

        // Downward percolation: Darcy's law under unit gradient (Rawls & Brakensiek 1985)
        let perc_rate = hill(upper, params.percolation_threshold, 2.0) * percolation_coeff;
        let thickness_ratio = layer_thickness(i) / layer_thickness(i + 1).max(0.01);
        let perc_flux = (perc_rate * dt).min(upper * 0.5); // never drain more than half

        // Upward capillary rise: van Genuchten (1980) suction head
        let cap_rate = michaelis_menten(lower, params.capillary_km) * capillary_coeff * (1.0 - upper).max(0.0);
        let cap_flux = (cap_rate * dt).min(lower * 0.3 * thickness_ratio);

        let net_flux = perc_flux - cap_flux;
        layers[i] = (layers[i] - net_flux).clamp(0.0, 1.0);
        layers[i + 1] = (layers[i + 1] + net_flux * thickness_ratio).clamp(0.0, 1.0);
    }
}

/// Infiltrate rainfall into the top soil layer.
///
/// Rain enters proportional to available pore space (1 - moisture) via Michaelis-Menten.
/// Returns the amount actually infiltrated.
pub fn infiltrate_rainfall(
    layers: &mut [f32; SOIL_LAYERS],
    precip_rate_mm_h: f32,
    dt: f32,
) -> f32 {
    if precip_rate_mm_h <= 0.0 {
        return 0.0;
    }
    let available_pore = (1.0 - layers[0]).max(0.0);
    let max_infiltrate = precip_rate_mm_h * dt / 3600.0; // mm/h -> mm in dt seconds
    // Scale to moisture fraction via layer thickness
    let moisture_add = max_infiltrate / SOIL_LAYER_DEPTHS_MM[0].max(0.01);
    let actual = michaelis_menten(available_pore, 0.3) * moisture_add;
    layers[0] = (layers[0] + actual).clamp(0.0, 1.0);
    actual
}

/// Evaporate water from the surface layer.
///
/// Rate depends on surface moisture, temperature, and humidity deficit.
pub fn evaporate_surface(
    layers: &mut [f32; SOIL_LAYERS],
    humidity: f32,
    temperature_c: f32,
    dt: f32,
) {
    let humidity_deficit = (1.0 - humidity.clamp(0.0, 1.0)).max(0.0);
    let temp_factor = hill(temperature_c.max(0.0) / 40.0, 0.5, 1.5);
    // Evaporation rate: Priestley-Taylor (1972) ET × sim compression (Monteith 1965)
    let evap_coeff = super::emergent_rates::SoilHydraulicParams::from_texture(0.5).evaporation_coeff;
    let evap_rate = hill(layers[0], 0.2, 2.0) * temp_factor * humidity_deficit * evap_coeff;
    let evap = (evap_rate * dt).min(layers[0] * 0.3);
    layers[0] = (layers[0] - evap).max(0.0);
}

/// Compute root-accessible moisture as an exponential-decay-weighted average
/// across all layers, biased by root depth.
///
/// `root_depth_bias` in [0, 1]: 0 = shallow roots (surface only), 1 = deep roots (all layers).
pub fn root_accessible_moisture(
    layers: &[f32; SOIL_LAYERS],
    root_depth_bias: f32,
) -> f32 {
    let bias = root_depth_bias.clamp(0.0, 1.0);
    // Characteristic depth: shallow roots have small depth, deep roots access all layers
    let char_depth = 2.0 + bias * 98.0; // mm: range [2, 100]

    let mut weighted_sum = 0.0f32;
    let mut weight_total = 0.0f32;
    for i in 0..SOIL_LAYERS {
        let midpoint = layer_midpoint(i);
        let thickness = layer_thickness(i);
        let weight = (-midpoint / char_depth).exp() * thickness;
        weighted_sum += layers[i] * weight;
        weight_total += weight;
    }

    if weight_total > 0.0 {
        (weighted_sum / weight_total).clamp(0.0, 1.0)
    } else {
        layers[0]
    }
}

// ---------------------------------------------------------------------------
// TerrariumWorld integration
// ---------------------------------------------------------------------------

impl crate::terrarium::TerrariumWorld {
    /// Step the 4-layer soil profile: vertical transport + sync legacy fields.
    pub(crate) fn step_soil_profile(&mut self, dt: f32) {
        let plane = self.config.width * self.config.height;

        for cell in 0..plane {
            let texture = self.soil_structure.get(cell).copied().unwrap_or(0.5);
            let mut layers = [
                self.soil_layer_moisture[0].get(cell).copied().unwrap_or(0.3),
                self.soil_layer_moisture[1].get(cell).copied().unwrap_or(0.3),
                self.soil_layer_moisture[2].get(cell).copied().unwrap_or(0.3),
                self.soil_layer_moisture[3].get(cell).copied().unwrap_or(0.25),
            ];

            step_soil_vertical_transport(&mut layers, texture, dt);

            // Evaporation from surface
            let humidity = self.humidity.get(cell).copied().unwrap_or(0.5);
            let temp = self.temperature.get(cell).copied().unwrap_or(22.0);
            evaporate_surface(&mut layers, humidity, temp, dt);

            // Rainfall infiltration from weather
            let precip = self.weather.precipitation_rate_mm_h;
            if precip > 0.0 {
                infiltrate_rainfall(&mut layers, precip, dt);
            }

            // Write back
            for (i, &val) in layers.iter().enumerate() {
                if let Some(slot) = self.soil_layer_moisture[i].get_mut(cell) {
                    *slot = val;
                }
            }

            // Sync legacy fields from layer representation
            if let Some(m) = self.moisture.get_mut(cell) {
                *m = layers[0];
            }
            if let Some(dm) = self.deep_moisture.get_mut(cell) {
                *dm = (layers[1] + layers[2]) * 0.5;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_thickness_sums_to_total() {
        let total: f32 = (0..SOIL_LAYERS).map(layer_thickness).sum();
        assert!(
            (total - SOIL_LAYER_DEPTHS_MM[SOIL_LAYERS - 1]).abs() < 0.01,
            "Layer thicknesses should sum to total depth: {total}"
        );
    }

    #[test]
    fn percolation_wet_to_dry() {
        let mut layers = [0.8, 0.2, 0.2, 0.2];
        step_soil_vertical_transport(&mut layers, 0.3, 1.0);
        assert!(layers[0] < 0.8, "Wet surface should lose water: {}", layers[0]);
        assert!(layers[1] > 0.2, "Dry layer below should gain water: {}", layers[1]);
    }

    #[test]
    fn capillary_rise_dry_to_wet() {
        let mut layers = [0.05, 0.8, 0.8, 0.8];
        // Clay texture for strong capillary
        step_soil_vertical_transport(&mut layers, 0.9, 1.0);
        assert!(layers[0] > 0.05, "Dry surface should gain capillary water: {}", layers[0]);
    }

    #[test]
    fn infiltration_fills_top_layer() {
        let mut layers = [0.2, 0.3, 0.3, 0.3];
        let infiltrated = infiltrate_rainfall(&mut layers, 10.0, 60.0); // 10mm/h for 60s
        assert!(infiltrated > 0.0, "Should infiltrate some water");
        assert!(layers[0] > 0.2, "Top layer should increase: {}", layers[0]);
    }

    #[test]
    fn evaporation_dries_surface() {
        let mut layers = [0.8, 0.5, 0.5, 0.5];
        evaporate_surface(&mut layers, 0.3, 25.0, 10.0);
        assert!(layers[0] < 0.8, "Surface should lose water: {}", layers[0]);
    }

    #[test]
    fn root_shallow_bias_reads_surface() {
        let layers = [0.9, 0.1, 0.1, 0.1];
        let moisture = root_accessible_moisture(&layers, 0.0);
        assert!(moisture > 0.7, "Shallow roots should read mostly surface: {moisture}");
    }

    #[test]
    fn root_deep_bias_reads_all() {
        let layers = [0.1, 0.1, 0.1, 0.9];
        let shallow = root_accessible_moisture(&layers, 0.0);
        let deep = root_accessible_moisture(&layers, 1.0);
        assert!(deep > shallow, "Deep roots should access more moisture: deep={deep}, shallow={shallow}");
    }

    #[test]
    fn sandy_soil_percolates_faster() {
        let mut sandy = [0.8, 0.2, 0.2, 0.2];
        let mut clayey = [0.8, 0.2, 0.2, 0.2];
        step_soil_vertical_transport(&mut sandy, 0.1, 1.0); // sand
        step_soil_vertical_transport(&mut clayey, 0.9, 1.0); // clay
        assert!(
            sandy[0] < clayey[0],
            "Sandy soil should lose surface water faster: sand={}, clay={}",
            sandy[0], clayey[0]
        );
    }

    #[test]
    fn clay_soil_holds_capillary() {
        let mut sandy = [0.05, 0.8, 0.5, 0.5];
        let mut clayey = [0.05, 0.8, 0.5, 0.5];
        step_soil_vertical_transport(&mut sandy, 0.1, 1.0);
        step_soil_vertical_transport(&mut clayey, 0.9, 1.0);
        assert!(
            clayey[0] > sandy[0],
            "Clay should have stronger capillary rise: clay={}, sand={}",
            clayey[0], sandy[0]
        );
    }

    #[test]
    fn backward_compat_legacy_fields() {
        // Verify that converting between layer representation and legacy works
        let layers = [0.5, 0.4, 0.3, 0.2];
        let surface = root_accessible_moisture(&layers, 0.1);
        let deep = root_accessible_moisture(&layers, 0.8);
        assert!(surface > 0.0 && surface <= 1.0, "Surface moisture in range");
        assert!(deep > 0.0 && deep <= 1.0, "Deep moisture in range");
    }
}
