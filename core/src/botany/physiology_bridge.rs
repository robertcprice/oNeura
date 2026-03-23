//! Molecular Physiology Bridge: Concentration-Driven Behavior via Kinetic Functions.
//!
//! This module provides the kinetic functions that translate molecular concentrations
//! and gene expression levels into plant behavioral rates. Every function uses
//! Hill or Michaelis-Menten kinetics — NO explicit thresholds.
//!
//! The Hill function with n>=2 creates steep, switch-like transitions at critical
//! concentrations. This IS how biology makes decisions: not with if-statements,
//! but with cooperative binding curves.
//!
//! Literature references:
//! - Hill equation: h(s) = s^n / (Km^n + s^n) (Hill, 1910, J Physiol 40:iv-vii)
//! - Farquhar photosynthesis: Vcmax * [CO2] / (Kc*(1+[O2]/Ko) + [CO2]) (Farquhar et al., 1980)
//! - Ethylene ripening: autocatalytic Hill n=3 (Alexander & Grierson, 2002, J Exp Bot)
//! - Auxin polar transport: PIN1-mediated (Wisniewska et al., 2006, Science 312:883)

use std::collections::HashMap;

/// Hill function: the universal biological switch.
///
/// Returns a value in [0.0, 1.0] representing the fraction of activation
/// at a given concentration. At `concentration == km`, returns exactly 0.5.
///
/// - `n = 1`: hyperbolic (Michaelis-Menten)
/// - `n = 2`: sigmoidal (cooperative dimer)
/// - `n >= 3`: steep switch-like (ultrasensitive)
#[inline]
pub fn hill(concentration: f32, km: f32, n: f32) -> f32 {
    if concentration <= 0.0 {
        return 0.0;
    }
    let c_n = concentration.powf(n);
    let km_n = km.powf(n);
    c_n / (km_n + c_n)
}

/// Inverse Hill: high when concentration is LOW. Used for repression.
#[inline]
pub fn hill_repression(concentration: f32, km: f32, n: f32) -> f32 {
    1.0 - hill(concentration, km, n)
}

/// Michaelis-Menten: special case of Hill with n=1.
#[inline]
pub fn michaelis_menten(substrate: f32, km: f32) -> f32 {
    hill(substrate, km, 1.0)
}

// ---------------------------------------------------------------------------
// Photosynthesis: Farquhar-simplified
// ---------------------------------------------------------------------------

/// Molecular photosynthesis rate.
///
/// Rate scales with:
/// - RbcL expression (RuBisCO enzyme from GRN)
/// - CO2 concentration (Michaelis-Menten, Kc ≈ 270 µmol at 25°C)
/// - H2O availability (Hill n=2, threshold at ~30% of capacity)
/// - Light saturation (hyperbolic, half-sat at ~40% full sun)
/// - Leaf biomass (more leaf = more photosynthetic area)
///
/// Returns carbon fixation rate in arbitrary flux units per timestep.
pub fn molecular_photosynthesis_rate(
    rbcl_expr: f32,
    co2: f32,
    water: f32,
    light: f32,
    leaf_biomass: f32,
    dt: f32,
) -> f32 {
    // Vcmax scales linearly with RuBisCO (RbcL) expression
    let vcmax = rbcl_expr.clamp(0.0, 1.0) * 1.0; // normalized max carboxylation rate

    // CO2 limitation: Michaelis-Menten with Kc
    let co2_sat = michaelis_menten(co2.max(0.0), 0.4); // Kc=0.4 in normalized units

    // Water limitation: Hill n=2, Km=0.3 (threshold at 30% soil capacity)
    let water_sat = hill(water.max(0.0), 0.3, 2.0);

    // Light saturation: hyperbolic, half-sat at 40% full sun
    let light_sat = michaelis_menten(light.max(0.0), 0.4);

    // Combine: product of all limitations (Liebig's law in continuous form)
    let rate = vcmax * co2_sat * water_sat * light_sat * leaf_biomass.max(0.0);

    // Scale to flux units: 0.0023 is the base carbon fixation coefficient
    // (matches the existing photosynthesis scale in PlantOrganismSim)
    rate * 0.0023 * dt
}

// ---------------------------------------------------------------------------
// Fruiting: Ethylene + Sugar + FT expression
// ---------------------------------------------------------------------------

/// Fruiting drive: the molecular signal that determines when a plant fruits.
///
/// Combines three independent signals through multiplicative gating:
/// 1. Ethylene accumulation (Hill n=3, Km=50 molecules — autocatalytic threshold)
/// 2. Sugar saturation (starch + sucrose, Hill n=2, Km=20 — energy readiness)
/// 3. FT (Florigen) gene expression (linear, photoperiod-dependent)
///
/// Returns a value in [0.0, 1.0]. Fruiting occurs when this exceeds ~0.5,
/// but the threshold is INTRINSIC to the Km values, not a magic number.
pub fn fruiting_drive(ethylene: f64, starch: f64, sucrose: f64, ft_expr: f32) -> f32 {
    let eth_signal = hill(ethylene as f32, 50.0, 3.0); // steep autocatalytic
    let sugar_signal = hill((starch + sucrose) as f32, 20.0, 2.0); // energy readiness
    let ft_signal = ft_expr.clamp(0.0, 1.0); // flowering/fruiting locus

    eth_signal * sugar_signal * ft_signal
}

// ---------------------------------------------------------------------------
// Growth allocation: Auxin-driven shoot/root partitioning
// ---------------------------------------------------------------------------

/// Molecular growth allocation: returns (leaf_fraction, stem_fraction, root_fraction).
///
/// PIN1 expression drives auxin polar transport → stronger shoot (leaf) growth.
/// NRT2.1 expression (high under low nitrate) → stronger root foraging.
/// Sucrose availability gates overall allocation rate.
///
/// These fractions always sum to 1.0.
pub fn molecular_growth_allocation(
    pin1_expr: f32,
    nrt2_expr: f32,
    sucrose: f64,
) -> (f32, f32, f32) {
    // Auxin signal: PIN1 drives shoot allocation
    let auxin_drive = pin1_expr.clamp(0.0, 1.0);

    // Root foraging: NRT2.1 upregulated under nitrogen stress
    let root_drive = nrt2_expr.clamp(0.0, 1.0);

    // Sugar gate: need energy to allocate anything
    let sugar_gate = hill(sucrose as f32, 5.0, 1.0);

    // Base allocation with molecular modulation
    let leaf_raw = 0.30 + auxin_drive * 0.15 - root_drive * 0.08;
    let root_raw = 0.30 + root_drive * 0.18 - auxin_drive * 0.06;
    let stem_raw = 0.25 + sugar_gate * 0.10;

    // Normalize to sum to 1.0
    let total = (leaf_raw + root_raw + stem_raw).max(0.01);
    let leaf = (leaf_raw / total).clamp(0.15, 0.55);
    let root = (root_raw / total).clamp(0.15, 0.55);
    let stem = (1.0 - leaf - root).clamp(0.10, 0.40);

    // Re-normalize after clamping
    let total2 = leaf + root + stem;
    (leaf / total2, stem / total2, root / total2)
}

// ---------------------------------------------------------------------------
// Death: Metabolic viability
// ---------------------------------------------------------------------------

/// Metabolic viability: the plant's ability to sustain life.
///
/// When glucose AND water are depleted relative to structural biomass,
/// the plant enters metabolic crisis. Sustained viability == 0 → death cascade.
///
/// Uses product of two Hill functions — both resources must be present.
/// This replaces the hardcoded `biomass < 0.09 OR (health < 0.03 AND storage < -0.2)`.
pub fn metabolic_viability(glucose: f64, water: f64, structural_biomass: f32) -> f32 {
    // Normalize pools by biomass (per-unit-mass concentration)
    let biomass = structural_biomass.max(0.01);
    let glucose_conc = (glucose as f32) / biomass;
    let water_conc = (water as f32) / biomass;

    // Hill n=2 for both — need both resources above threshold
    let glucose_viability = hill(glucose_conc, 2.0, 2.0); // Km=2: need 2 glucose per unit biomass
    let water_viability = hill(water_conc, 5.0, 2.0); // Km=5: need 5 water per unit biomass

    // Product: both must be satisfied
    glucose_viability * water_viability
}

// ---------------------------------------------------------------------------
// Stomatal conductance: ABA-mediated closure
// ---------------------------------------------------------------------------

/// Stomatal openness: ABA closes stomata via SLAC1 anion channels.
///
/// High ABA (drought signal) → stomata close → reduced water loss but also reduced CO2 uptake.
/// Water availability also directly affects turgor-driven stomatal aperture.
pub fn stomatal_openness(aba_level: f32, water: f32) -> f32 {
    // ABA closes stomata: inverse Hill
    let aba_closure = hill_repression(aba_level, 0.4, 2.0);

    // Water opens stomata: turgor pressure
    let turgor_opening = hill(water, 0.3, 2.0);

    // Product: both conditions must be favorable
    (aba_closure * turgor_opening).clamp(0.05, 1.0)
}

/// Mechanical viability modifier: reduces plant viability when stem is damaged.
///
/// Uses Hill repression: viability drops sharply around damage = 0.5 (Km).
/// n=3 gives a steep response -- slight damage is tolerated, severe damage is lethal.
pub fn mechanical_viability_modifier(damage: f32) -> f32 {
    hill_repression(damage, 0.5, 3.0)
}

// ---------------------------------------------------------------------------
// Shade Avoidance Syndrome: R:FR phytochrome responses
// ---------------------------------------------------------------------------

/// Shade Avoidance Syndrome: elongation increase under low R:FR.
///
/// When SAS expression is high (low R:FR, canopy shade), stems elongate
/// up to 80% faster to escape the shade canopy. This is the classic
/// phytochrome-mediated shade avoidance response.
///
/// (Ballaré, 1999, Trends Plant Sci 4:97; Franklin, 2008, New Phytol 179:930)
pub fn shade_avoidance_elongation(sas_expression: f32) -> f32 {
    1.0 + hill(sas_expression, 0.5, 2.0) * 0.8
}

/// Shade Avoidance Syndrome: branching suppression under low R:FR.
///
/// High SAS reduces lateral branching by up to 60%, concentrating
/// growth resources into vertical stem elongation.
///
/// (Casal, 2012, Plant Cell Environ 35:271)
pub fn shade_avoidance_branching(sas_expression: f32) -> f32 {
    1.0 - hill(sas_expression, 0.4, 3.0) * 0.6
}

// ---------------------------------------------------------------------------
// VOC emission: molecular rate
// ---------------------------------------------------------------------------

/// VOC emission rate from metabolome, replacing hardcoded odorant formula.
///
/// Depends on metabolome VOC emission rate, volatile scale, and stomatal openness.
pub fn molecular_voc_emission(
    voc_emission_rate: f64,
    volatile_scale: f32,
    stomatal_open: f32,
) -> f32 {
    (voc_emission_rate as f32 * volatile_scale * stomatal_open.clamp(0.1, 1.0)).clamp(0.002, 0.12)
}

// ---------------------------------------------------------------------------
// Convenience: read all rates from metabolome + gene expression
// ---------------------------------------------------------------------------

/// Complete molecular state bundle for driving plant behavior.
/// Replaces the ~25 scalar arguments to PlantOrganismSim::step().
#[derive(Debug, Clone)]
pub struct MolecularDriveState {
    pub photosynthesis_rate: f32,
    pub fruiting_drive: f32,
    pub growth_allocation: (f32, f32, f32), // (leaf, stem, root)
    pub metabolic_viability: f32,
    pub stomatal_openness: f32,
    pub voc_emission_rate: f32,
    pub ethylene_level: f32,
    pub glucose_pool: f32,
    pub water_pool: f32,
    /// Shade avoidance elongation factor: 1.0 = no shade, up to 1.8 under canopy.
    /// Derived from SAS gene expression via Hill kinetics.
    pub shade_elongation_factor: f32,
    /// Shade avoidance branching factor: 1.0 = no shade, down to 0.4 under canopy.
    /// Derived from SAS gene expression via Hill kinetics.
    pub shade_branching_factor: f32,
}

/// Build the complete molecular drive state from metabolome and gene expression.
pub fn compute_molecular_drive(
    metabolome: &crate::botany::PlantMetabolome,
    gene_expr: &HashMap<String, f32>,
    light: f32,
    water_factor: f32,
    leaf_biomass: f32,
    total_biomass: f32,
    volatile_scale: f32,
    dt: f32,
    mechanical_damage: f32,
) -> MolecularDriveState {
    let rbcl = gene_expr.get("RbcL").copied().unwrap_or(0.5);
    let ft = gene_expr.get("FT").copied().unwrap_or(0.0);
    let pin1 = gene_expr.get("PIN1").copied().unwrap_or(0.5);
    let nrt2 = gene_expr.get("NRT2.1").copied().unwrap_or(0.5);
    let dreb = gene_expr.get("DREB").copied().unwrap_or(0.0);
    let sas = gene_expr.get("SAS").copied().unwrap_or(0.0);

    let co2_norm = (metabolome.co2_count as f32 / 500.0).clamp(0.0, 2.0);
    let water_norm = (metabolome.water_count as f32 / 1000.0).clamp(0.0, 2.0);

    let photo = molecular_photosynthesis_rate(
        rbcl,
        co2_norm,
        water_norm.min(water_factor),
        light,
        leaf_biomass,
        dt,
    );

    let fruit = fruiting_drive(
        metabolome.ethylene_count,
        metabolome.starch_reserve,
        metabolome.sucrose_count,
        ft,
    );

    let alloc = molecular_growth_allocation(pin1, nrt2, metabolome.sucrose_count);

    let viability = metabolic_viability(
        metabolome.glucose_count,
        metabolome.water_count,
        total_biomass,
    ) * mechanical_viability_modifier(mechanical_damage);

    // ABA correlates with DREB expression (drought response)
    let stomatal = stomatal_openness(dreb, water_factor);

    let voc = molecular_voc_emission(metabolome.voc_emission_rate, volatile_scale, stomatal);

    MolecularDriveState {
        photosynthesis_rate: photo,
        fruiting_drive: fruit,
        growth_allocation: alloc,
        metabolic_viability: viability,
        stomatal_openness: stomatal,
        voc_emission_rate: voc,
        ethylene_level: metabolome.ethylene_count as f32,
        glucose_pool: metabolome.glucose_count as f32,
        water_pool: metabolome.water_count as f32,
        shade_elongation_factor: shade_avoidance_elongation(sas),
        shade_branching_factor: shade_avoidance_branching(sas),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hill_at_km_gives_half() {
        let val = hill(0.5, 0.5, 2.0);
        assert!((val - 0.5).abs() < 0.01, "Hill at Km should be 0.5, got {val}");
    }

    #[test]
    fn test_hill_zero_concentration() {
        assert_eq!(hill(0.0, 0.5, 2.0), 0.0);
    }

    #[test]
    fn test_hill_high_concentration_saturates() {
        let val = hill(10.0, 0.5, 2.0);
        assert!(val > 0.99, "Hill at 20x Km should be near 1.0, got {val}");
    }

    #[test]
    fn test_hill_repression_inverts() {
        let act = hill(0.8, 0.5, 2.0);
        let rep = hill_repression(0.8, 0.5, 2.0);
        assert!((act + rep - 1.0).abs() < 0.001, "Hill + repression should = 1.0");
    }

    #[test]
    fn test_photosynthesis_scales_with_rbcl() {
        let high = molecular_photosynthesis_rate(1.0, 0.8, 0.8, 0.8, 0.5, 1.0);
        let low = molecular_photosynthesis_rate(0.1, 0.8, 0.8, 0.8, 0.5, 1.0);
        assert!(
            high > low * 3.0,
            "High RbcL ({high}) should give much more photosynthesis than low ({low})"
        );
    }

    #[test]
    fn test_photosynthesis_zero_in_dark() {
        let rate = molecular_photosynthesis_rate(1.0, 0.8, 0.8, 0.0, 0.5, 1.0);
        assert!(
            rate < 0.0001,
            "Photosynthesis should be ~0 in dark, got {rate}"
        );
    }

    #[test]
    fn test_fruiting_drive_needs_all_signals() {
        // Missing ethylene
        let no_eth = fruiting_drive(0.0, 100.0, 100.0, 0.9);
        assert!(no_eth < 0.05, "No ethylene → no fruiting, got {no_eth}");

        // Missing sugar
        let no_sugar = fruiting_drive(100.0, 0.0, 0.0, 0.9);
        assert!(no_sugar < 0.05, "No sugar → no fruiting, got {no_sugar}");

        // Missing FT
        let no_ft = fruiting_drive(100.0, 100.0, 100.0, 0.0);
        assert!(no_ft < 0.05, "No FT → no fruiting, got {no_ft}");

        // All present
        let full = fruiting_drive(100.0, 100.0, 100.0, 0.9);
        assert!(full > 0.5, "All signals → fruiting, got {full}");
    }

    #[test]
    fn test_growth_allocation_sums_to_one() {
        let (leaf, stem, root) = molecular_growth_allocation(0.7, 0.3, 50.0);
        let sum = leaf + stem + root;
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Allocation should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn test_nitrogen_stress_shifts_roots() {
        // High NRT2.1 (low nitrogen) → more root allocation
        let (_, _, root_stressed) = molecular_growth_allocation(0.5, 0.9, 50.0);
        let (_, _, root_normal) = molecular_growth_allocation(0.5, 0.2, 50.0);
        assert!(
            root_stressed > root_normal,
            "N-stress root ({root_stressed}) > normal root ({root_normal})"
        );
    }

    #[test]
    fn test_auxin_shifts_leaf() {
        // High PIN1 → more leaf allocation
        let (leaf_high_pin, _, _) = molecular_growth_allocation(0.9, 0.3, 50.0);
        let (leaf_low_pin, _, _) = molecular_growth_allocation(0.2, 0.3, 50.0);
        assert!(
            leaf_high_pin > leaf_low_pin,
            "High PIN1 leaf ({leaf_high_pin}) > low PIN1 leaf ({leaf_low_pin})"
        );
    }

    #[test]
    fn test_viability_healthy_plant() {
        let v = metabolic_viability(200.0, 500.0, 0.5);
        assert!(v > 0.5, "Healthy plant should have high viability, got {v}");
    }

    #[test]
    fn test_viability_dead_plant() {
        let v = metabolic_viability(0.0, 0.0, 0.5);
        assert!(v < 0.01, "Depleted plant should have zero viability, got {v}");
    }

    #[test]
    fn test_drought_death_cascade() {
        // Cut water → glucose depletes → viability drops
        let v_wet = metabolic_viability(100.0, 500.0, 0.5);
        let v_dry = metabolic_viability(100.0, 0.1, 0.5);
        assert!(
            v_wet > v_dry * 2.0,
            "Drought reduces viability: wet={v_wet} dry={v_dry}"
        );
    }

    #[test]
    fn test_stomatal_closes_with_aba() {
        let open = stomatal_openness(0.0, 0.8);
        let closed = stomatal_openness(0.9, 0.8);
        assert!(
            open > closed * 1.5,
            "ABA should close stomata: open={open}, closed={closed}"
        );
    }

    #[test]
    fn test_stomatal_closes_with_drought() {
        let wet = stomatal_openness(0.1, 0.9);
        let dry = stomatal_openness(0.1, 0.05);
        assert!(
            wet > dry * 1.5,
            "Drought should close stomata: wet={wet}, dry={dry}"
        );
    }

    #[test]
    fn test_compute_molecular_drive_smoke() {
        let mut metabolome = crate::botany::PlantMetabolome::new();
        metabolome.glucose_count = 200.0;
        metabolome.ethylene_count = 60.0;
        metabolome.starch_reserve = 30.0;
        metabolome.sucrose_count = 25.0;

        let mut gene_expr = HashMap::new();
        gene_expr.insert("RbcL".to_string(), 0.8);
        gene_expr.insert("FT".to_string(), 0.6);
        gene_expr.insert("PIN1".to_string(), 0.7);
        gene_expr.insert("NRT2.1".to_string(), 0.3);
        gene_expr.insert("DREB".to_string(), 0.1);

        let drive = compute_molecular_drive(
            &metabolome,
            &gene_expr,
            0.8,  // light
            0.9,  // water_factor
            0.4,  // leaf_biomass
            0.9,  // total_biomass
            1.0,  // volatile_scale
            20.0, // dt
            0.0,  // mechanical_damage
        );

        assert!(drive.photosynthesis_rate > 0.0, "Should photosynthesize");
        assert!(drive.fruiting_drive > 0.0, "Should have fruiting drive");
        assert!(drive.metabolic_viability > 0.5, "Should be viable");
        assert!(drive.stomatal_openness > 0.3, "Stomata should be open");
    }

    // -- Shade Avoidance Syndrome tests (Phase 4) --

    #[test]
    fn test_compute_molecular_drive_includes_shade_factors() {
        let mut metabolome = crate::botany::PlantMetabolome::new();
        metabolome.glucose_count = 200.0;
        let mut gene_expr = HashMap::new();
        gene_expr.insert("RbcL".to_string(), 0.8);
        gene_expr.insert("FT".to_string(), 0.0);
        gene_expr.insert("PIN1".to_string(), 0.5);
        gene_expr.insert("NRT2.1".to_string(), 0.5);
        gene_expr.insert("DREB".to_string(), 0.0);

        // No SAS → factors should be ~1.0
        gene_expr.insert("SAS".to_string(), 0.0);
        let drive_sun = compute_molecular_drive(
            &metabolome, &gene_expr, 0.8, 0.9, 0.4, 0.9, 1.0, 20.0, 0.0,
        );
        assert!(
            (drive_sun.shade_elongation_factor - 1.0).abs() < 0.05,
            "Full sun: elongation should be ~1.0, got {}",
            drive_sun.shade_elongation_factor,
        );
        assert!(
            (drive_sun.shade_branching_factor - 1.0).abs() < 0.05,
            "Full sun: branching should be ~1.0, got {}",
            drive_sun.shade_branching_factor,
        );

        // High SAS → elongation up, branching down
        gene_expr.insert("SAS".to_string(), 0.8);
        let drive_shade = compute_molecular_drive(
            &metabolome, &gene_expr, 0.8, 0.9, 0.4, 0.9, 1.0, 20.0, 0.0,
        );
        assert!(
            drive_shade.shade_elongation_factor > 1.4,
            "Shade: elongation should be >1.4, got {}",
            drive_shade.shade_elongation_factor,
        );
        assert!(
            drive_shade.shade_branching_factor < 0.6,
            "Shade: branching should be <0.6, got {}",
            drive_shade.shade_branching_factor,
        );
    }

    #[test]
    fn test_shade_elongation_factor() {
        let no_sas = shade_avoidance_elongation(0.0);
        let full_sas = shade_avoidance_elongation(0.8);
        assert!(
            (no_sas - 1.0).abs() < 0.01,
            "No SAS = no elongation change: {no_sas}",
        );
        assert!(
            full_sas > 1.5,
            "Full SAS should increase elongation >50%: {full_sas}",
        );
    }

    #[test]
    fn test_shade_branching_suppression() {
        let no_sas = shade_avoidance_branching(0.0);
        let full_sas = shade_avoidance_branching(0.8);
        assert!(
            (no_sas - 1.0).abs() < 0.01,
            "No SAS = no branching change: {no_sas}",
        );
        assert!(
            full_sas < 0.6,
            "Full SAS should suppress branching below 60%: {full_sas}",
        );
    }
}
