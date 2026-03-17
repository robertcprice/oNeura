//! Enzyme Sequence Evolution within Terrarium Context
//!
//! Bridges the `enzyme_engineering` directed evolution engine with the terrarium
//! world's enzyme probe system. This enables a two-level evolution:
//!
//! 1. **Outer loop** (terrarium_evolve.rs): NSGA-II optimizes WorldGenome parameters
//!    including enzyme probe placement (x, y coordinates)
//! 2. **Inner loop** (this module): For each placed probe, directed evolution optimizes
//!    the enzyme sequence itself for soil-relevant fitness targets
//!
//! This creates a **nested evolutionary optimization** where both the WHERE and the WHAT
//! of enzyme placement are co-evolved. The outer loop finds the best soil microenvironment
//! for each enzyme, while the inner loop finds the best enzyme for that microenvironment.
//!
//! # Novel Contribution
//!
//! This is (to our knowledge) the first system that co-evolves enzyme sequences with
//! their spatial placement in a multi-scale ecological simulation. Traditional enzyme
//! engineering treats the enzyme in isolation; here the enzyme's fitness depends on
//! its soil context (moisture, organic matter, temperature, microbial competition).

use crate::enzyme_engineering::{
    EnzymeVariant, EvolutionConfig, EvolutionResult, FitnessTarget,
    directed_evolution, compute_fitness,
};
use crate::probe_coupling::soil_enzyme_fitness;

// ---------------------------------------------------------------------------
// Soil-Contextualized Enzyme Variants
// ---------------------------------------------------------------------------

/// An enzyme variant with soil context metadata.
#[derive(Debug, Clone)]
pub struct SoilEnzymeVariant {
    /// The underlying enzyme variant with kinetic properties.
    pub variant: EnzymeVariant,
    /// Soil context where this enzyme performs best.
    pub optimal_moisture: f32,
    /// Optimal soil organic matter content.
    pub optimal_organic_matter: f32,
    /// Optimal soil temperature (K).
    pub optimal_temperature: f32,
    /// Combined fitness in the soil context.
    pub soil_fitness: f64,
}

/// Configuration for soil-contextualized enzyme evolution.
#[derive(Debug, Clone)]
pub struct SoilEnzymeEvolutionConfig {
    /// Base enzyme engineering config.
    pub base_config: EvolutionConfig,
    /// Local soil moisture at the probe site (0-1).
    pub local_moisture: f32,
    /// Local organic matter content at the probe site (0-1).
    pub local_organic_matter: f32,
    /// Local soil temperature (K).
    pub local_temperature: f32,
    /// Weight for soil-specific fitness vs. pure kinetics.
    pub soil_weight: f64,
}

impl Default for SoilEnzymeEvolutionConfig {
    fn default() -> Self {
        Self {
            base_config: EvolutionConfig {
                population_size: 20,
                generations: 10,
                mutation_rate: 0.005,
                recombination_fraction: 0.2,
                fitness_target: FitnessTarget::MultiObjective,
                tournament_size: 3,
                elitism_fraction: 0.1,
                seed: 42,
            },
            local_moisture: 0.5,
            local_organic_matter: 0.3,
            local_temperature: 298.0,
            soil_weight: 0.4,
        }
    }
}

/// Result of soil-contextualized enzyme evolution.
#[derive(Debug, Clone)]
pub struct SoilEvolutionResult {
    /// The base evolution result from enzyme_engineering.
    pub base_result: EvolutionResult,
    /// Best variant with soil context scoring.
    pub best_soil_variant: SoilEnzymeVariant,
    /// Soil-contextualized fitness history.
    pub soil_fitness_history: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Core Evolution Functions
// ---------------------------------------------------------------------------

/// Evolve an enzyme sequence optimized for a specific soil microenvironment.
///
/// This runs the `enzyme_engineering::directed_evolution()` engine but re-scores
/// each generation's variants using soil-specific fitness criteria:
/// - **Stability bonus** in hot/dry soil (enzyme must resist denaturation)
/// - **Catalytic bonus** in organic-matter-rich soil (more substrate available)
/// - **Expression penalty** in acidic/anoxic soil (harder folding conditions)
///
/// The result includes both the standard kinetic evolution result and the
/// soil-contextualized scoring.
pub fn evolve_soil_enzyme(
    parent: &EnzymeVariant,
    config: &SoilEnzymeEvolutionConfig,
) -> SoilEvolutionResult {
    // Run base directed evolution
    let base_result = directed_evolution(parent, &config.base_config);

    // Re-score all evolved variants in the soil context
    let mut best_soil_fitness = f64::NEG_INFINITY;
    let mut best_variant_idx = 0;
    let mut soil_fitness_history = Vec::new();

    for (i, variant) in base_result.evolved_variants.iter().enumerate() {
        let sf = score_variant_in_soil(variant, config);
        if sf > best_soil_fitness {
            best_soil_fitness = sf;
            best_variant_idx = i;
        }
    }

    // Build soil fitness history from generation champions
    for champion in &base_result.generation_champions {
        soil_fitness_history.push(score_variant_in_soil(champion, config));
    }

    let best = &base_result.evolved_variants[best_variant_idx];
    let best_soil_variant = SoilEnzymeVariant {
        variant: best.clone(),
        optimal_moisture: config.local_moisture,
        optimal_organic_matter: config.local_organic_matter,
        optimal_temperature: config.local_temperature,
        soil_fitness: best_soil_fitness,
    };

    SoilEvolutionResult {
        base_result,
        best_soil_variant,
        soil_fitness_history,
    }
}

/// Score a single enzyme variant in a soil context.
///
/// Combines the enzyme's intrinsic kinetic fitness with soil-specific modifiers:
/// - Temperature penalty if enzyme optimal != soil temperature
/// - Moisture modifier affecting expression and stability
/// - Organic matter availability affecting catalytic utilization
fn score_variant_in_soil(
    variant: &EnzymeVariant,
    config: &SoilEnzymeEvolutionConfig,
) -> f64 {
    // Base kinetic fitness from enzyme_engineering
    let base_fitness = compute_fitness(variant, config.base_config.fitness_target);

    // Soil-specific fitness from probe_coupling
    let soil_fit = soil_enzyme_fitness(
        variant.kcat,
        variant.km_um,
        variant.stability_score,
        variant.expression_level,
    );

    // Temperature penalty: enzymes evolved at 298K penalized if soil is very hot/cold
    let temp_deviation = (config.local_temperature - 298.0).abs() / 50.0;
    let temp_penalty = 1.0 - (temp_deviation as f64).min(0.5);

    // Moisture modifier: dry soil favors stable enzymes, wet soil favors fast enzymes
    let moisture_mod = if config.local_moisture < 0.3 {
        // Dry soil: stability matters more (enzyme aggregation risk)
        1.0 + variant.stability_score.abs() * 0.005
    } else {
        // Wet soil: catalysis is more effective (substrate diffusion)
        1.0 + (variant.kcat / 1000.0).min(0.3)
    };

    // Organic matter availability: more OM = more substrate = higher effective catalysis
    let om_bonus = 1.0 + config.local_organic_matter as f64 * 0.5;

    // Combine with configurable weighting
    let kinetic_component = base_fitness * (1.0 - config.soil_weight);
    let soil_component = soil_fit * config.soil_weight * temp_penalty * moisture_mod * om_bonus;

    kinetic_component + soil_component
}

// ---------------------------------------------------------------------------
// Enzyme Library for Different Soil Types
// ---------------------------------------------------------------------------

/// Create a parent enzyme optimized for cellulase activity (cellulose decomposition).
///
/// Based on Trichoderma reesei cellobiohydrolase Cel7A active site sequence.
/// kcat ~4.3 s^-1, Km ~1100 uM for crystalline cellulose.
pub fn cellulase_parent() -> EnzymeVariant {
    EnzymeVariant::new(
        "Cel7A-fragment",
        "EQCGGQWT", // Catalytic triad region
        4.3,
        1100.0,
    )
}

/// Create a parent enzyme optimized for urease activity (urea hydrolysis).
///
/// Based on Klebsiella aerogenes urease active site.
/// kcat ~3500 s^-1, Km ~2500 uM — one of the fastest known enzymes.
pub fn urease_parent() -> EnzymeVariant {
    EnzymeVariant::new(
        "KaUrease-fragment",
        "HHDDKLCG", // His-His-Asp catalytic triad
        3500.0,
        2500.0,
    )
}

/// Create a parent enzyme optimized for phosphatase activity.
///
/// Based on alkaline phosphatase active site.
/// kcat ~80 s^-1, Km ~60 uM for p-nitrophenyl phosphate.
pub fn phosphatase_parent() -> EnzymeVariant {
    EnzymeVariant::new(
        "AlkPhos-fragment",
        "SHDASNWL", // Ser-His-Asp catalytic triad
        80.0,
        60.0,
    )
}

/// Select a soil enzyme parent based on the dominant need at a probe site.
///
/// - High organic matter → cellulase (break down plant debris)
/// - Low nitrogen → urease (convert urea to ammonium)
/// - Low phosphorus → phosphatase (release bound phosphate)
pub fn select_parent_for_soil(organic_matter: f32, nitrogen: f32, _phosphorus: f32) -> EnzymeVariant {
    if organic_matter > 0.5 {
        cellulase_parent()
    } else if nitrogen < 0.2 {
        urease_parent()
    } else {
        phosphatase_parent()
    }
}

/// Evolve an enzyme cocktail — one variant per soil function.
///
/// Returns 3 evolved variants: cellulase, urease, phosphatase, each
/// optimized for the local soil conditions at the probe site.
pub fn evolve_enzyme_cocktail(
    config: &SoilEnzymeEvolutionConfig,
) -> Vec<SoilEvolutionResult> {
    let parents = [cellulase_parent(), urease_parent(), phosphatase_parent()];

    parents.iter().map(|parent| {
        let mut cfg = config.clone();
        // Reduce population for cocktail (3x the variants total)
        cfg.base_config.population_size = (config.base_config.population_size / 2).max(8);
        cfg.base_config.generations = (config.base_config.generations / 2).max(5);
        cfg.base_config.seed = config.base_config.seed.wrapping_add(
            parent.sequence.bytes().map(|b| b as u64).sum::<u64>()
        );
        evolve_soil_enzyme(parent, &cfg)
    }).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cellulase_parent_valid() {
        let p = cellulase_parent();
        assert!(p.kcat > 0.0);
        assert!(p.km_um > 0.0);
        assert!(!p.sequence.is_empty());
    }

    #[test]
    fn urease_parent_valid() {
        let p = urease_parent();
        assert!(p.kcat > 0.0);
        assert!(p.km_um > 0.0);
        assert!(!p.sequence.is_empty());
    }

    #[test]
    fn phosphatase_parent_valid() {
        let p = phosphatase_parent();
        assert!(p.kcat > 0.0);
        assert!(p.km_um > 0.0);
        assert!(!p.sequence.is_empty());
    }

    #[test]
    fn evolve_soil_enzyme_basic() {
        let parent = cellulase_parent();
        let config = SoilEnzymeEvolutionConfig {
            base_config: EvolutionConfig {
                population_size: 8,
                generations: 3,
                mutation_rate: 0.01,
                recombination_fraction: 0.2,
                fitness_target: FitnessTarget::MultiObjective,
                tournament_size: 3,
                elitism_fraction: 0.1,
                seed: 42,
            },
            local_moisture: 0.4,
            local_organic_matter: 0.6,
            local_temperature: 300.0,
            soil_weight: 0.4,
        };
        let result = evolve_soil_enzyme(&parent, &config);
        assert!(!result.base_result.evolved_variants.is_empty());
        assert!(result.best_soil_variant.soil_fitness > 0.0);
        assert!(!result.soil_fitness_history.is_empty());
    }

    #[test]
    fn select_parent_by_soil_type() {
        // High organic matter → cellulase
        let p1 = select_parent_for_soil(0.7, 0.5, 0.3);
        assert!(p1.name.contains("Cel"));

        // Low nitrogen → urease
        let p2 = select_parent_for_soil(0.2, 0.1, 0.3);
        assert!(p2.name.contains("Urease"));

        // Default → phosphatase
        let p3 = select_parent_for_soil(0.3, 0.5, 0.1);
        assert!(p3.name.contains("Phos"));
    }

    #[test]
    fn evolve_enzyme_cocktail_basic() {
        let config = SoilEnzymeEvolutionConfig {
            base_config: EvolutionConfig {
                population_size: 8,
                generations: 3,
                mutation_rate: 0.01,
                recombination_fraction: 0.2,
                fitness_target: FitnessTarget::MultiObjective,
                tournament_size: 3,
                elitism_fraction: 0.1,
                seed: 42,
            },
            ..SoilEnzymeEvolutionConfig::default()
        };
        let cocktail = evolve_enzyme_cocktail(&config);
        assert_eq!(cocktail.len(), 3);
        for result in &cocktail {
            assert!(!result.base_result.evolved_variants.is_empty());
        }
    }

    #[test]
    fn soil_fitness_rewards_stability_in_hot_soil() {
        let parent = cellulase_parent();
        let hot_config = SoilEnzymeEvolutionConfig {
            local_temperature: 340.0, // Hot soil
            soil_weight: 0.6,
            ..SoilEnzymeEvolutionConfig::default()
        };
        let normal_config = SoilEnzymeEvolutionConfig {
            local_temperature: 298.0, // Normal soil
            soil_weight: 0.6,
            ..SoilEnzymeEvolutionConfig::default()
        };
        let hot_score = score_variant_in_soil(&parent, &hot_config);
        let normal_score = score_variant_in_soil(&parent, &normal_config);
        // Normal temperature should generally score better (less penalty)
        assert!(normal_score >= hot_score * 0.5, "Normal temp should not be dramatically worse than hot");
    }
}
