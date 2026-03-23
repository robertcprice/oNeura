//! Molecular Visual Phenotype: Pigment-driven plant appearance.
//!
//! Translates metabolome concentrations (chlorophyll from RbcL expression,
//! anthocyanin, carotenoid) into visual leaf/fruit RGB values. This means
//! a nitrogen-starved plant yellows because chlorophyll is low (RbcL downregulated),
//! not because we hardcode a "stressed color". Autumn colors emerge when
//! chlorophyll degrades and carotenoids become visible.
//!
//! Literature:
//! - Chlorophyll degradation during senescence: Hortensteiner, 2006, Annu Rev Plant Biol
//! - Anthocyanin biosynthesis: Winkel-Shirley, 2001, Plant Physiol 126:485
//! - Carotenoid roles in photosynthesis: Demmig-Adams & Adams, 1996, Trends Plant Sci

use super::metabolome::PlantMetabolome;
use std::collections::HashMap;

/// Molecular-derived visual state for a plant.
///
/// Every color channel is driven by actual metabolite concentrations —
/// green from chlorophyll (RbcL expression × nitrogen status),
/// red from anthocyanin, orange from carotenoid, brown from senescence.
#[derive(Debug, Clone, Copy, Default)]
pub struct MolecularVisualState {
    /// Chlorophyll density [0,1]: from RbcL expression + nitrogen availability.
    /// Drives the GREEN channel of leaf color.
    pub chlorophyll_density: f32,
    /// Anthocyanin density [0,1]: from ANTHOCYANIN_BIOSYNTHESIS + CHS expression.
    /// Drives RED/PURPLE channel of fruit and some leaf color.
    pub anthocyanin_density: f32,
    /// Carotenoid density [0,1]: from carotenoid metabolome pool.
    /// Drives ORANGE/YELLOW channel — visible when chlorophyll degrades.
    pub carotenoid_density: f32,
    /// Turgor pressure [0,1]: from metabolome water content.
    /// Drives leaf droop/wilt.
    pub turgor_pressure: f32,
    /// Senescence progress [0,1]: from energy charge depletion.
    /// Drives browning.
    pub senescence_progress: f32,
    /// Limonene/terpene level: influences citrus-like visual sheen.
    pub terpene_level: f32,
}

impl MolecularVisualState {
    /// Build visual state from metabolome + gene expression.
    pub fn from_metabolome(
        metabolome: &PlantMetabolome,
        gene_expr: &HashMap<String, f32>,
        total_biomass: f32,
    ) -> Self {
        let biomass = total_biomass.max(0.01);

        // Chlorophyll: proportional to RbcL expression (which requires light + nitrogen)
        let rbcl = gene_expr.get("RbcL").copied().unwrap_or(0.3);
        // Nitrogen status affects chlorophyll: amino acid pool is proxy for N
        let nitrogen_status = (metabolome.amino_acid_pool as f32 / (biomass * 20.0)).clamp(0.0, 1.0);
        let chlorophyll = (rbcl * 0.6 + nitrogen_status * 0.4).clamp(0.0, 1.0);

        // Anthocyanin: from metabolome pool normalized by biomass
        let anthocyanin = (metabolome.anthocyanin_count as f32 / (biomass * 5.0)).clamp(0.0, 1.0);

        // Carotenoid: from metabolome pool normalized by biomass
        let carotenoid = (metabolome.carotenoid_count as f32 / (biomass * 5.0)).clamp(0.0, 1.0);

        // Turgor: water content per unit biomass
        let turgor = (metabolome.water_count as f32 / (biomass * 200.0)).clamp(0.0, 1.0);

        // Senescence: inverse of energy charge (glucose + starch per biomass)
        let energy = ((metabolome.glucose_count + metabolome.starch_reserve) as f32)
            / (biomass * 100.0);
        let senescence = (1.0 - energy.clamp(0.0, 1.0)).max(0.0);

        // Terpene level from limonene pool
        let terpene = (metabolome.limonene_count as f32 / (biomass * 3.0)).clamp(0.0, 1.0);

        MolecularVisualState {
            chlorophyll_density: chlorophyll,
            anthocyanin_density: anthocyanin,
            carotenoid_density: carotenoid,
            turgor_pressure: turgor,
            senescence_progress: senescence,
            terpene_level: terpene,
        }
    }

    /// Compute leaf RGB from molecular pigment concentrations.
    ///
    /// Green channel = chlorophyll (high RbcL + nitrogen)
    /// When chlorophyll degrades (nitrogen stress, aging), carotenoid becomes
    /// visible → autumn yellow/orange. Anthocyanin adds red overlay (some species).
    /// Senescence adds brown.
    pub fn leaf_rgb(&self) -> [f32; 3] {
        // Base green from chlorophyll
        let green = self.chlorophyll_density * 0.55 + 0.15;

        // Red: anthocyanin + carotenoid visible when chlorophyll is low + senescence brown
        let carotenoid_visible = self.carotenoid_density * (1.0 - self.chlorophyll_density * 0.7);
        let red = self.anthocyanin_density * 0.4
            + carotenoid_visible * 0.6
            + self.senescence_progress * 0.3;

        // Blue: very low in leaves, slight anthocyanin contribution
        let blue = self.anthocyanin_density * 0.15 + 0.08;

        // Senescence desaturates toward brown
        let brown_mix = self.senescence_progress * 0.4;
        let r = (red * (1.0 - brown_mix) + 0.45 * brown_mix).clamp(0.0, 1.0);
        let g = (green * (1.0 - brown_mix * 1.2) + 0.35 * brown_mix).clamp(0.0, 1.0);
        let b = (blue * (1.0 - brown_mix) + 0.18 * brown_mix).clamp(0.0, 1.0);

        [r, g, b]
    }

    /// Compute fruit RGB from molecular pigment concentrations.
    ///
    /// Fruit color is dominated by anthocyanin (cherry/peach → red) or
    /// carotenoid (orange/lemon → orange/yellow). Chlorophyll in unripe fruit
    /// gives green undertone.
    pub fn fruit_rgb(&self) -> [f32; 3] {
        // Red: anthocyanin dominant
        let red = self.anthocyanin_density * 0.7 + self.carotenoid_density * 0.5;
        // Green: residual chlorophyll in unripe fruit
        let green = self.chlorophyll_density * 0.3 + self.carotenoid_density * 0.3;
        // Blue: trace from anthocyanin
        let blue = self.anthocyanin_density * 0.1;

        [red.clamp(0.0, 1.0), green.clamp(0.0, 1.0), blue.clamp(0.0, 1.0)]
    }

    /// Compute droop factor from turgor pressure.
    ///
    /// Low turgor (drought) → high droop → visual wilting.
    pub fn droop_from_turgor(&self) -> f32 {
        0.02 + (1.0 - self.turgor_pressure) * 0.40
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nitrogen_stress_yellows_leaves() {
        let mut gene_expr = HashMap::new();
        gene_expr.insert("RbcL".to_string(), 0.1f32); // low RbcL (N-stressed)

        let mut met = PlantMetabolome::new();
        met.amino_acid_pool = 0.5; // very low nitrogen
        met.glucose_count = 50.0;
        met.starch_reserve = 20.0;
        met.carotenoid_count = 2.0;

        let vis = MolecularVisualState::from_metabolome(&met, &gene_expr, 0.5);
        let rgb = vis.leaf_rgb();

        // Low chlorophyll → green should be lower
        assert!(vis.chlorophyll_density < 0.3, "Chlorophyll should be low under N stress");
        // Carotenoid should become visible → red/yellow shift
        // Green channel should be lower than a healthy plant
        assert!(
            rgb[1] < 0.55,
            "Green channel should be reduced under N stress, got {}",
            rgb[1]
        );
    }

    #[test]
    fn test_healthy_plant_is_green() {
        let mut gene_expr = HashMap::new();
        gene_expr.insert("RbcL".to_string(), 0.9f32);

        let mut met = PlantMetabolome::new();
        met.amino_acid_pool = 30.0;
        met.glucose_count = 200.0;
        met.starch_reserve = 100.0;

        let vis = MolecularVisualState::from_metabolome(&met, &gene_expr, 0.5);
        let rgb = vis.leaf_rgb();

        assert!(vis.chlorophyll_density > 0.6, "Chlorophyll should be high");
        assert!(
            rgb[1] > rgb[0],
            "Healthy plant should be green-dominant: R={} G={}",
            rgb[0], rgb[1]
        );
    }

    #[test]
    fn test_cherry_anthocyanin_color() {
        // Cherry: high anthocyanin → fruit should be red
        let mut met_cherry = PlantMetabolome::new();
        met_cherry.anthocyanin_count = 5.0;
        met_cherry.carotenoid_count = 0.5;

        let cherry_vis = MolecularVisualState::from_metabolome(
            &met_cherry, &HashMap::new(), 0.5
        );
        let cherry_rgb = cherry_vis.fruit_rgb();

        // Apple: low anthocyanin
        let mut met_apple = PlantMetabolome::new();
        met_apple.anthocyanin_count = 0.5;
        met_apple.carotenoid_count = 0.5;

        let apple_vis = MolecularVisualState::from_metabolome(
            &met_apple, &HashMap::new(), 0.5
        );
        let apple_rgb = apple_vis.fruit_rgb();

        assert!(
            cherry_rgb[0] > apple_rgb[0],
            "Cherry fruit should be redder than apple: cherry_R={} apple_R={}",
            cherry_rgb[0], apple_rgb[0]
        );
    }

    #[test]
    fn test_drought_increases_droop() {
        let gene_expr = HashMap::new();

        // Well-watered
        let mut met_wet = PlantMetabolome::new();
        met_wet.water_count = 2000.0;
        let vis_wet = MolecularVisualState::from_metabolome(&met_wet, &gene_expr, 0.5);

        // Drought
        let mut met_dry = PlantMetabolome::new();
        met_dry.water_count = 5.0;
        let vis_dry = MolecularVisualState::from_metabolome(&met_dry, &gene_expr, 0.5);

        assert!(
            vis_dry.droop_from_turgor() > vis_wet.droop_from_turgor(),
            "Drought should increase droop: dry={} wet={}",
            vis_dry.droop_from_turgor(), vis_wet.droop_from_turgor()
        );
        assert!(
            vis_dry.turgor_pressure < vis_wet.turgor_pressure,
            "Drought should lower turgor: dry={} wet={}",
            vis_dry.turgor_pressure, vis_wet.turgor_pressure
        );
    }

    #[test]
    fn test_senescence_browns_leaves() {
        let gene_expr = HashMap::new();

        // Dying plant: no energy
        let mut met = PlantMetabolome::new();
        met.glucose_count = 0.0;
        met.starch_reserve = 0.0;
        met.water_count = 10.0;

        let vis = MolecularVisualState::from_metabolome(&met, &gene_expr, 0.5);
        assert!(
            vis.senescence_progress > 0.8,
            "Depleted plant should show senescence, got {}",
            vis.senescence_progress
        );

        let rgb = vis.leaf_rgb();
        // Brown: red and green channels similar, green reduced
        assert!(
            rgb[1] < 0.5,
            "Senescing leaf green should be low, got {}",
            rgb[1]
        );
    }
}
