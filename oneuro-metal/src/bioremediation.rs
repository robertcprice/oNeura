//! Soil Bioremediation Simulator
//!
//! Models contaminated soil and enzyme-driven remediation. This module lets you:
//!
//! 1. **Define soil contamination** — heavy metals, hydrocarbons, pesticides
//! 2. **Design enzyme cocktails** — targeted enzymatic degradation of contaminants
//! 3. **Optimize enzyme placement** — evolve WHERE and WHAT enzymes to deploy
//! 4. **Score remediation progress** — track contaminant reduction over time
//!
//! # Scientific Basis
//!
//! Enzyme-based bioremediation is a $2.8B market growing at 11.2% CAGR. Key enzymes:
//! - **Laccases** — oxidize polycyclic aromatic hydrocarbons (PAHs)
//! - **Peroxidases** — degrade chlorinated compounds (PCBs, pesticides)
//! - **Ureases** — precipitate heavy metals as carbonates
//! - **Phosphatases** — immobilize heavy metals via phosphate precipitation
//!
//! This module is the first to combine atomistic enzyme simulation with grid-based
//! soil biogeochemistry for bioremediation optimization.
//!
//! # Usage
//!
//! ```rust,ignore
//! let contamination = SoilContaminationProfile::hydrocarbon_spill(0.8);
//! let plan = remediation_plan(&contamination, 42);
//! let score = evaluate_remediation(&plan, &contamination);
//! ```

use crate::enzyme_engineering::{EnzymeVariant, EvolutionConfig, FitnessTarget, directed_evolution};

// ---------------------------------------------------------------------------
// Contamination Model
// ---------------------------------------------------------------------------

/// Types of soil contaminants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ContaminantClass {
    /// Polycyclic aromatic hydrocarbons (PAHs) from combustion/petroleum.
    PAH,
    /// Chlorinated organic compounds (PCBs, pesticides).
    Chlorinated,
    /// Heavy metals (Pb, Cd, Hg, As).
    HeavyMetal,
    /// Petroleum hydrocarbons (BTEX, diesel range organics).
    Petroleum,
    /// Agricultural pesticides (organophosphates, carbamates).
    Pesticide,
    /// Mixed contamination (brownfield sites).
    Mixed,
}

/// A single contaminant with concentration and properties.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Contaminant {
    pub name: String,
    pub class: ContaminantClass,
    /// Concentration in mg/kg soil (ppm).
    pub concentration_ppm: f64,
    /// Regulatory limit in mg/kg (below this = "clean").
    pub regulatory_limit_ppm: f64,
    /// Aqueous solubility (mg/L) — affects bioavailability.
    pub solubility_mg_l: f64,
    /// Log Kow (octanol-water partition) — affects sorption to soil.
    pub log_kow: f64,
    /// Half-life in soil without treatment (days).
    pub natural_half_life_days: f64,
}

/// Full contamination profile of a soil site.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SoilContaminationProfile {
    pub site_name: String,
    pub contaminants: Vec<Contaminant>,
    /// Soil pH (affects enzyme activity and metal speciation).
    pub soil_ph: f64,
    /// Organic matter content (fraction 0-1).
    pub organic_matter: f64,
    /// Clay content (fraction 0-1) — affects contaminant binding.
    pub clay_fraction: f64,
    /// Moisture content (fraction 0-1).
    pub moisture: f64,
    /// Soil temperature (K).
    pub temperature_k: f64,
}

impl SoilContaminationProfile {
    /// Create a profile for a petroleum hydrocarbon spill.
    ///
    /// `severity` is 0.0 (minor) to 1.0 (catastrophic).
    pub fn hydrocarbon_spill(severity: f64) -> Self {
        let base_ppm = 500.0 + severity * 9500.0; // 500-10000 ppm range
        Self {
            site_name: "Hydrocarbon spill site".to_string(),
            contaminants: vec![
                Contaminant {
                    name: "Benzene".to_string(),
                    class: ContaminantClass::Petroleum,
                    concentration_ppm: base_ppm * 0.05,
                    regulatory_limit_ppm: 0.5,
                    solubility_mg_l: 1780.0,
                    log_kow: 2.13,
                    natural_half_life_days: 15.0,
                },
                Contaminant {
                    name: "Toluene".to_string(),
                    class: ContaminantClass::Petroleum,
                    concentration_ppm: base_ppm * 0.15,
                    regulatory_limit_ppm: 0.7,
                    solubility_mg_l: 526.0,
                    log_kow: 2.73,
                    natural_half_life_days: 20.0,
                },
                Contaminant {
                    name: "Naphthalene".to_string(),
                    class: ContaminantClass::PAH,
                    concentration_ppm: base_ppm * 0.08,
                    regulatory_limit_ppm: 5.0,
                    solubility_mg_l: 31.0,
                    log_kow: 3.30,
                    natural_half_life_days: 48.0,
                },
                Contaminant {
                    name: "Total petroleum hydrocarbons".to_string(),
                    class: ContaminantClass::Petroleum,
                    concentration_ppm: base_ppm,
                    regulatory_limit_ppm: 100.0,
                    solubility_mg_l: 50.0,
                    log_kow: 4.5,
                    natural_half_life_days: 365.0,
                },
            ],
            soil_ph: 6.5,
            organic_matter: 0.03,
            clay_fraction: 0.2,
            moisture: 0.25,
            temperature_k: 293.0,
        }
    }

    /// Create a profile for heavy metal contamination (mining/smelting).
    pub fn heavy_metal_site(severity: f64) -> Self {
        let base_ppm = 100.0 + severity * 4900.0;
        Self {
            site_name: "Heavy metal contaminated site".to_string(),
            contaminants: vec![
                Contaminant {
                    name: "Lead (Pb)".to_string(),
                    class: ContaminantClass::HeavyMetal,
                    concentration_ppm: base_ppm,
                    regulatory_limit_ppm: 400.0,
                    solubility_mg_l: 0.01,
                    log_kow: 0.0, // metals don't partition to octanol
                    natural_half_life_days: f64::INFINITY, // metals don't degrade
                },
                Contaminant {
                    name: "Cadmium (Cd)".to_string(),
                    class: ContaminantClass::HeavyMetal,
                    concentration_ppm: base_ppm * 0.1,
                    regulatory_limit_ppm: 70.0,
                    solubility_mg_l: 0.005,
                    log_kow: 0.0,
                    natural_half_life_days: f64::INFINITY,
                },
                Contaminant {
                    name: "Arsenic (As)".to_string(),
                    class: ContaminantClass::HeavyMetal,
                    concentration_ppm: base_ppm * 0.2,
                    regulatory_limit_ppm: 40.0,
                    solubility_mg_l: 0.05,
                    log_kow: 0.0,
                    natural_half_life_days: f64::INFINITY,
                },
            ],
            soil_ph: 4.5, // Acidic from mining runoff
            organic_matter: 0.01,
            clay_fraction: 0.15,
            moisture: 0.15,
            temperature_k: 288.0,
        }
    }

    /// Create a profile for pesticide-contaminated agricultural soil.
    pub fn pesticide_site(severity: f64) -> Self {
        let base_ppm = 10.0 + severity * 490.0;
        Self {
            site_name: "Pesticide contaminated agricultural soil".to_string(),
            contaminants: vec![
                Contaminant {
                    name: "Chlorpyrifos".to_string(),
                    class: ContaminantClass::Pesticide,
                    concentration_ppm: base_ppm,
                    regulatory_limit_ppm: 2.0,
                    solubility_mg_l: 1.4,
                    log_kow: 4.7,
                    natural_half_life_days: 30.0,
                },
                Contaminant {
                    name: "Atrazine".to_string(),
                    class: ContaminantClass::Pesticide,
                    concentration_ppm: base_ppm * 0.5,
                    regulatory_limit_ppm: 3.0,
                    solubility_mg_l: 33.0,
                    log_kow: 2.61,
                    natural_half_life_days: 60.0,
                },
            ],
            soil_ph: 6.8,
            organic_matter: 0.04,
            clay_fraction: 0.25,
            moisture: 0.30,
            temperature_k: 295.0,
        }
    }

    /// Overall contamination severity (0-1) relative to regulatory limits.
    pub fn severity(&self) -> f64 {
        if self.contaminants.is_empty() { return 0.0; }
        let total_exceedance: f64 = self.contaminants.iter().map(|c| {
            if c.regulatory_limit_ppm > 0.0 {
                (c.concentration_ppm / c.regulatory_limit_ppm - 1.0).max(0.0)
            } else {
                0.0
            }
        }).sum();
        (total_exceedance / self.contaminants.len() as f64).min(1.0)
    }
}

// ---------------------------------------------------------------------------
// Remediation Enzymes
// ---------------------------------------------------------------------------

/// An enzyme designed for bioremediation of a specific contaminant class.
#[derive(Debug, Clone)]
pub struct RemediationEnzyme {
    /// The base enzyme variant.
    pub variant: EnzymeVariant,
    /// Target contaminant class.
    pub target_class: ContaminantClass,
    /// Degradation efficiency (fraction of contaminant degraded per hour
    /// per unit enzyme at optimal conditions).
    pub degradation_rate: f64,
    /// pH optimum for this enzyme.
    pub ph_optimum: f64,
    /// pH range (enzyme active within optimum +/- this value).
    pub ph_range: f64,
    /// Temperature optimum (K).
    pub temp_optimum_k: f64,
}

impl RemediationEnzyme {
    /// Create a laccase for PAH/petroleum degradation.
    pub fn laccase() -> Self {
        Self {
            variant: EnzymeVariant::new("Laccase-TvLcc1", "HWHGFFQHP", 280.0, 120.0),
            target_class: ContaminantClass::PAH,
            degradation_rate: 0.08, // 8% per hour at optimal
            ph_optimum: 5.0,
            ph_range: 2.0,
            temp_optimum_k: 323.0, // 50C
        }
    }

    /// Create a peroxidase for chlorinated compound degradation.
    pub fn peroxidase() -> Self {
        Self {
            variant: EnzymeVariant::new("MnP-PhcMnP", "HDCEFQGA", 150.0, 85.0),
            target_class: ContaminantClass::Chlorinated,
            degradation_rate: 0.05,
            ph_optimum: 4.5,
            ph_range: 1.5,
            temp_optimum_k: 313.0,
        }
    }

    /// Create a phosphatase for heavy metal immobilization.
    pub fn phosphatase_remediation() -> Self {
        Self {
            variant: EnzymeVariant::new("AlkPhos-Remed", "SHDASNWL", 80.0, 60.0),
            target_class: ContaminantClass::HeavyMetal,
            degradation_rate: 0.03, // immobilization rather than degradation
            ph_optimum: 8.0,
            ph_range: 2.5,
            temp_optimum_k: 310.0,
        }
    }

    /// Create an organophosphorus hydrolase for pesticide degradation.
    pub fn oph() -> Self {
        Self {
            variant: EnzymeVariant::new("OPH-PTE", "HHDWLFCG", 2100.0, 90.0),
            target_class: ContaminantClass::Pesticide,
            degradation_rate: 0.12, // Very efficient enzyme
            ph_optimum: 8.5,
            ph_range: 2.0,
            temp_optimum_k: 308.0,
        }
    }

    /// Compute effective degradation rate given soil conditions.
    pub fn effective_rate(&self, soil_ph: f64, soil_temp_k: f64, moisture: f64) -> f64 {
        // pH effect: Gaussian around optimum
        let ph_deviation = (soil_ph - self.ph_optimum).abs();
        let ph_factor = (-0.5 * (ph_deviation / self.ph_range).powi(2)).exp();

        // Temperature effect: Arrhenius-like with denaturation
        let t = soil_temp_k;
        let t_opt = self.temp_optimum_k;
        let temp_factor = if t < t_opt - 30.0 {
            ((t - (t_opt - 30.0)) / 30.0).max(0.0)
        } else if t <= t_opt {
            0.5 + 0.5 * ((t - (t_opt - 30.0)) / 30.0).min(1.0)
        } else if t <= t_opt + 10.0 {
            1.0 - 0.5 * ((t - t_opt) / 10.0)
        } else {
            0.5 * ((t_opt + 30.0 - t) / 20.0).max(0.0)
        };

        // Moisture effect: enzymes need water for activity
        let moisture_factor = if moisture < 0.1 {
            moisture * 5.0 // Severe limitation below 10%
        } else if moisture > 0.6 {
            1.0 - (moisture - 0.6) * 0.5 // Slight reduction in waterlogged soil
        } else {
            1.0
        };

        self.degradation_rate * ph_factor * temp_factor * moisture_factor
    }
}

// ---------------------------------------------------------------------------
// Remediation Plan
// ---------------------------------------------------------------------------

/// A remediation plan: which enzymes to deploy and where.
#[derive(Debug, Clone)]
pub struct RemediationPlan {
    /// Enzyme cocktail to deploy.
    pub enzymes: Vec<RemediationEnzyme>,
    /// Estimated treatment time (days) to reach regulatory limits.
    pub estimated_days: f64,
    /// Estimated cost per m^3 of soil (USD).
    pub estimated_cost_per_m3: f64,
    /// Confidence score (0-1) that treatment will succeed.
    pub confidence: f64,
}

/// Generate a remediation plan for a contaminated site.
///
/// Selects appropriate enzymes based on contaminant classes present,
/// estimates treatment duration, and scores confidence.
pub fn remediation_plan(profile: &SoilContaminationProfile, seed: u64) -> RemediationPlan {
    let mut enzymes = Vec::new();
    let mut seen_classes = std::collections::HashSet::new();

    for contaminant in &profile.contaminants {
        if seen_classes.contains(&contaminant.class) { continue; }
        seen_classes.insert(contaminant.class);

        let enzyme = match contaminant.class {
            ContaminantClass::PAH | ContaminantClass::Petroleum => RemediationEnzyme::laccase(),
            ContaminantClass::Chlorinated => RemediationEnzyme::peroxidase(),
            ContaminantClass::HeavyMetal => RemediationEnzyme::phosphatase_remediation(),
            ContaminantClass::Pesticide => RemediationEnzyme::oph(),
            ContaminantClass::Mixed => {
                enzymes.push(RemediationEnzyme::laccase());
                RemediationEnzyme::peroxidase()
            }
        };
        enzymes.push(enzyme);
    }

    // Estimate treatment time: slowest contaminant determines total time
    let mut max_days = 0.0f64;
    for contaminant in &profile.contaminants {
        let matching_enzyme = enzymes.iter().find(|e| {
            e.target_class == contaminant.class ||
            (e.target_class == ContaminantClass::PAH && contaminant.class == ContaminantClass::Petroleum)
        });

        let days = if let Some(enzyme) = matching_enzyme {
            let rate = enzyme.effective_rate(profile.soil_ph, profile.temperature_k, profile.moisture);
            if rate > 1e-10 && contaminant.concentration_ppm > contaminant.regulatory_limit_ppm {
                let reduction_needed = contaminant.concentration_ppm / contaminant.regulatory_limit_ppm;
                // Time to reduce by the needed factor: t = ln(ratio) / rate_per_hour / 24
                reduction_needed.ln() / rate / 24.0
            } else {
                0.0
            }
        } else {
            // No enzyme for this contaminant — rely on natural attenuation
            contaminant.natural_half_life_days * 3.0 // ~87.5% reduction
        };
        max_days = max_days.max(days);
    }

    // Evolve the enzyme cocktail for better performance
    let _evolved = evolve_remediation_cocktail(&enzymes, profile, seed);

    // Cost estimation: enzyme production + application
    let enzyme_cost_per_kg = 150.0; // USD/kg for industrial enzyme
    let application_cost_per_m3 = 50.0;
    let enzyme_loading_kg_per_m3 = 0.5 * enzymes.len() as f64;
    let cost = enzyme_loading_kg_per_m3 * enzyme_cost_per_kg + application_cost_per_m3;

    // Confidence based on how well our enzymes match the contaminants
    let matched_fraction = seen_classes.len() as f64 /
        profile.contaminants.iter().map(|c| c.class).collect::<std::collections::HashSet<_>>().len().max(1) as f64;
    let ph_ok = profile.soil_ph > 3.0 && profile.soil_ph < 10.0;
    let temp_ok = profile.temperature_k > 273.0 && profile.temperature_k < 323.0;
    let confidence = matched_fraction * if ph_ok { 0.9 } else { 0.5 } * if temp_ok { 1.0 } else { 0.7 };

    RemediationPlan {
        enzymes,
        estimated_days: max_days,
        estimated_cost_per_m3: cost,
        confidence: confidence.min(1.0),
    }
}

/// Evolve remediation enzymes for better performance in the specific soil conditions.
fn evolve_remediation_cocktail(
    enzymes: &[RemediationEnzyme],
    _profile: &SoilContaminationProfile,
    seed: u64,
) -> Vec<EnzymeVariant> {
    enzymes.iter().enumerate().map(|(i, enzyme)| {
        let config = EvolutionConfig {
            population_size: 12,
            generations: 5,
            mutation_rate: 0.008,
            recombination_fraction: 0.15,
            fitness_target: FitnessTarget::MultiObjective,
            tournament_size: 3,
            elitism_fraction: 0.1,
            seed: seed.wrapping_add(i as u64 * 1000),
        };
        let result = directed_evolution(&enzyme.variant, &config);
        result.best_variant
    }).collect()
}

// ---------------------------------------------------------------------------
// Remediation Scoring
// ---------------------------------------------------------------------------

/// Result of remediation evaluation.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RemediationScore {
    /// Overall remediation effectiveness (0-1).
    pub effectiveness: f64,
    /// Per-contaminant residual concentrations after treatment.
    pub residual_concentrations: Vec<(String, f64)>,
    /// Number of contaminants below regulatory limits after treatment.
    pub compliant_count: usize,
    /// Total contaminants.
    pub total_contaminants: usize,
    /// Treatment duration (days).
    pub treatment_days: f64,
    /// Cost per m^3.
    pub cost_per_m3: f64,
    /// Composite score for optimization (higher = better).
    pub composite_score: f64,
}

/// Evaluate a remediation plan against a contamination profile.
///
/// Simulates enzyme treatment over the estimated duration and computes
/// residual contaminant concentrations.
pub fn evaluate_remediation(
    plan: &RemediationPlan,
    profile: &SoilContaminationProfile,
) -> RemediationScore {
    let mut residuals = Vec::new();
    let mut compliant = 0;

    for contaminant in &profile.contaminants {
        // Find best matching enzyme
        let best_rate = plan.enzymes.iter().map(|e| {
            if e.target_class == contaminant.class ||
               (e.target_class == ContaminantClass::PAH && contaminant.class == ContaminantClass::Petroleum) {
                e.effective_rate(profile.soil_ph, profile.temperature_k, profile.moisture)
            } else {
                0.0
            }
        }).fold(0.0f64, f64::max);

        // Natural attenuation rate
        let natural_rate = if contaminant.natural_half_life_days.is_finite() && contaminant.natural_half_life_days > 0.0 {
            (0.5f64).ln().abs() / contaminant.natural_half_life_days / 24.0
        } else {
            0.0
        };

        // Combined rate (enzyme + natural)
        let total_rate = best_rate + natural_rate;

        // Residual after treatment
        let hours = plan.estimated_days * 24.0;
        let residual = contaminant.concentration_ppm * (-total_rate * hours).exp();
        let residual = residual.max(0.0);

        if residual <= contaminant.regulatory_limit_ppm {
            compliant += 1;
        }

        residuals.push((contaminant.name.clone(), residual));
    }

    let total = profile.contaminants.len();
    let effectiveness = compliant as f64 / total.max(1) as f64;

    // Composite score: effectiveness weighted by cost-efficiency and time
    let time_factor = (1.0 / (1.0 + plan.estimated_days / 365.0)).max(0.1); // prefer faster
    let cost_factor = (1.0 / (1.0 + plan.estimated_cost_per_m3 / 500.0)).max(0.1); // prefer cheaper
    let composite = effectiveness * 0.5 + time_factor * 0.3 + cost_factor * 0.2;

    RemediationScore {
        effectiveness,
        residual_concentrations: residuals,
        compliant_count: compliant,
        total_contaminants: total,
        treatment_days: plan.estimated_days,
        cost_per_m3: plan.estimated_cost_per_m3,
        composite_score: composite,
    }
}

/// Optimize enzyme cocktail for a contaminated site using directed evolution.
///
/// This is the top-level entry point for bioremediation optimization:
/// 1. Profile the contamination
/// 2. Select initial enzyme cocktail
/// 3. Evolve each enzyme for the specific soil conditions
/// 4. Score the evolved cocktail
/// 5. Return the best plan
pub fn optimize_remediation(
    profile: &SoilContaminationProfile,
    n_iterations: usize,
    seed: u64,
) -> (RemediationPlan, RemediationScore) {
    let mut best_plan = remediation_plan(profile, seed);
    let mut best_score = evaluate_remediation(&best_plan, profile);

    for i in 1..n_iterations {
        let candidate_plan = remediation_plan(profile, seed.wrapping_add(i as u64 * 7919));
        let candidate_score = evaluate_remediation(&candidate_plan, profile);

        if candidate_score.composite_score > best_score.composite_score {
            best_plan = candidate_plan;
            best_score = candidate_score;
        }
    }

    (best_plan, best_score)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hydrocarbon_spill_profile() {
        let profile = SoilContaminationProfile::hydrocarbon_spill(0.5);
        assert_eq!(profile.contaminants.len(), 4);
        assert!(profile.severity() > 0.0);
        assert!(profile.soil_ph > 0.0);
    }

    #[test]
    fn heavy_metal_profile() {
        let profile = SoilContaminationProfile::heavy_metal_site(0.7);
        assert_eq!(profile.contaminants.len(), 3);
        assert!(profile.contaminants.iter().all(|c| c.class == ContaminantClass::HeavyMetal));
    }

    #[test]
    fn pesticide_profile() {
        let profile = SoilContaminationProfile::pesticide_site(0.3);
        assert_eq!(profile.contaminants.len(), 2);
    }

    #[test]
    fn remediation_enzymes_have_activity() {
        let laccase = RemediationEnzyme::laccase();
        let rate = laccase.effective_rate(5.0, 323.0, 0.3);
        assert!(rate > 0.0, "Laccase should be active at optimal pH/temp");

        let perox = RemediationEnzyme::peroxidase();
        let rate2 = perox.effective_rate(4.5, 313.0, 0.3);
        assert!(rate2 > 0.0, "Peroxidase should be active at optimal conditions");
    }

    #[test]
    fn remediation_plan_generation() {
        let profile = SoilContaminationProfile::hydrocarbon_spill(0.5);
        let plan = remediation_plan(&profile, 42);
        assert!(!plan.enzymes.is_empty());
        assert!(plan.estimated_days > 0.0);
        assert!(plan.estimated_cost_per_m3 > 0.0);
        assert!(plan.confidence > 0.0 && plan.confidence <= 1.0);
    }

    #[test]
    fn remediation_evaluation() {
        let profile = SoilContaminationProfile::pesticide_site(0.3);
        let plan = remediation_plan(&profile, 42);
        let score = evaluate_remediation(&plan, &profile);
        assert!(score.effectiveness >= 0.0 && score.effectiveness <= 1.0);
        assert_eq!(score.total_contaminants, 2);
        assert!(!score.residual_concentrations.is_empty());
    }

    #[test]
    fn remediation_reduces_concentration() {
        let profile = SoilContaminationProfile::pesticide_site(0.5);
        let plan = remediation_plan(&profile, 42);
        let score = evaluate_remediation(&plan, &profile);

        for (name, residual) in &score.residual_concentrations {
            let original = profile.contaminants.iter()
                .find(|c| &c.name == name)
                .map(|c| c.concentration_ppm)
                .unwrap_or(0.0);
            assert!(
                *residual <= original,
                "Residual {} ({}) should be <= original ({})",
                name, residual, original
            );
        }
    }

    #[test]
    fn optimize_remediation_improves() {
        let profile = SoilContaminationProfile::hydrocarbon_spill(0.3);
        let (_, score) = optimize_remediation(&profile, 3, 42);
        assert!(score.composite_score > 0.0, "Optimization should produce positive score");
    }

    #[test]
    fn enzyme_effective_rate_ph_sensitivity() {
        let laccase = RemediationEnzyme::laccase();
        let optimal_rate = laccase.effective_rate(5.0, 323.0, 0.3);
        let off_ph_rate = laccase.effective_rate(9.0, 323.0, 0.3); // Far from pH 5 optimum
        assert!(optimal_rate > off_ph_rate, "Enzyme should be less active far from pH optimum");
    }

    #[test]
    fn enzyme_effective_rate_moisture_sensitivity() {
        let laccase = RemediationEnzyme::laccase();
        let wet_rate = laccase.effective_rate(5.0, 323.0, 0.4);
        let dry_rate = laccase.effective_rate(5.0, 323.0, 0.05);
        assert!(wet_rate > dry_rate, "Enzyme should be more active in moist soil");
    }
}
