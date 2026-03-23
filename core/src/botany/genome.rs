//! Molecular Botany: Genomic structures, species catalog, and gene regulatory network execution.
//!
//! Gene circuits use Hill-function kinetics for signal transduction.
//! Hill equation: expression = signal^n / (Km^n + signal^n)
//!
//! Literature references:
//! - Hill coefficient n=2-4 typical for cooperative binding (Alon, 2007, "An Introduction to Systems Biology")
//! - Arabidopsis flowering: FLC/FT antagonism (Michaels & Amasino, 1999, Plant Cell 11:949-956)
//! - Drought response: DREB/ABA pathway (Yamaguchi-Shinozaki & Shinozaki, 2006, Annu Rev Plant Biol)
//! - Nitrate signaling: NRT2.1 Km ~0.2 mM (Glass et al., 2002, J Exp Bot 53:855-862)
//! - RuBisCO activation by light: rapid within minutes (Portis, 2003, Photosyn Res 75:11-27)

use super::species::{botanical_species_profiles, BotanicalGrowthForm, BotanicalSpeciesProfile};
use crate::terrarium::emergent_rates::{
    literature_gene_decay, literature_gene_hill_n, literature_gene_km,
    literature_gene_max_expression,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents a specific botanical species with its unique genomic identity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BotanicalSpecies {
    pub name: String,
    pub common_name: String,
    pub taxonomy_id: u32,
    pub genome: BotanicalGenome,
}

/// The complete genomic blueprint for a plant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BotanicalGenome {
    pub chromosomes: Vec<PlantChromosome>,
    /// Maps gene names/IDs to their regulatory logic.
    pub gene_circuits: HashMap<String, PlantGeneCircuit>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlantChromosome {
    pub id: u8,
    /// Symbolic sequence representation (simplified for Phase 1).
    pub sequence: String,
}

/// Regulatory logic for a specific gene module (e.g., 'Flowering', 'DroughtResponse').
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlantGeneCircuit {
    pub gene_id: String,
    /// Environmental inputs that trigger this circuit.
    pub inputs: Vec<EnvironmentalSignal>,
    /// Current expression level [0.0, 1.0].
    pub expression: f32,
    /// Hill coefficient (cooperativity). Typical range 2-4.
    /// Higher values produce sharper switch-like responses.
    /// (Alon, 2007: n=2 for dimeric TFs, n=4 for tetrameric)
    #[serde(default = "default_hill_coefficient")]
    pub hill_n: f32,
    /// Half-maximal activation constant (Km). The signal level at which
    /// expression reaches 50% of maximum. Units depend on input signal.
    #[serde(default = "default_km")]
    pub km: f32,
    /// Maximum expression rate [0.0, 1.0]. Some genes have lower ceilings.
    #[serde(default = "default_max_expression")]
    pub max_expression: f32,
    /// Protein/mRNA decay rate (per second). Controls how fast expression
    /// drops when signal is removed. Typical mRNA half-life ~5-60 min in plants.
    #[serde(default = "default_decay_rate")]
    pub decay_rate: f32,
    /// Whether this is a repressor (true) or activator (false).
    /// Repressors use: expression = 1 - hill(signal) so expression drops with signal.
    #[serde(default)]
    pub is_repressor: bool,
}

/// Default Hill n from Alon (2007): dimeric TF = 2.0.
fn default_hill_coefficient() -> f32 {
    literature_gene_hill_n("_default") // 2.0
}
/// Default Km: generic half-maximal activation.
fn default_km() -> f32 {
    literature_gene_km("_default") // 0.50
}
/// Default max expression level.
fn default_max_expression() -> f32 {
    literature_gene_max_expression("_default") // 1.0
}
/// Default mRNA/protein decay rate: ~5.8 min half-life (Narsai+ 2007).
fn default_decay_rate() -> f32 {
    literature_gene_decay("_default") // 0.002
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EnvironmentalSignal {
    Temperature,
    Moisture,
    LightIntensity,
    Photoperiod,
    NutrientNitrogen,
    AbscisicAcid,       // Internal hormone signal
    RedFarRedRatio,     // R:FR ratio for phytochrome shade detection
    NeighborVOC,        // Defense VOC from nearby damaged plants (Phase 5)
    JasmonicAcid,       // Internal jasmonate signal (Phase 5)
    SalicylicAcid,      // Internal salicylate signal (Phase 5)
    MechanicalStress,   // Stem mechanical damage from wind (Phase 1)
}

/// Environmental state snapshot fed to gene circuits each timestep.
/// All fields use normalized [0.0, 1.0] ranges except where noted.
#[derive(Debug, Clone, Default)]
pub struct EnvironmentState {
    /// Temperature in degrees Celsius. Gene circuits normalize internally
    /// using a thermal optimum curve centered at 22 C for most plant genes.
    pub temperature_c: f32,
    /// Soil moisture fraction [0.0, 1.0] where 1.0 = field capacity.
    pub soil_moisture: f32,
    /// Photosynthetically active radiation fraction [0.0, 1.0] where 1.0 = full sun (~2000 umol/m2/s).
    pub light_intensity: f32,
    /// Day length in hours [0.0, 24.0]. Arabidopsis long-day threshold ~14 h.
    pub photoperiod_hours: f32,
    /// Soil nitrate concentration in mM. NRT2.1 Km ~0.2 mM (Glass et al., 2002).
    pub soil_nitrate: f32,
    /// Internal abscisic acid level [0.0, 1.0]. Drought signal hormone.
    pub internal_aba: f32,
    /// Red:Far-Red ratio [0.0, ~1.5] where 1.2 = full sun, <0.5 = deep shade.
    /// Phytochrome B photoequilibrium responds to this ratio.
    pub r_fr_ratio: f32,
    /// Neighbor defense VOC concentration [0.0, 1.0]. (Phase 5 placeholder)
    pub neighbor_voc: f32,
    /// Internal jasmonic acid level [0.0, 1.0]. (Phase 5 placeholder)
    pub jasmonic_acid: f32,
    /// Internal salicylic acid level [0.0, 1.0]. (Phase 5 placeholder)
    pub salicylic_acid: f32,
    /// Mechanical stem damage [0.0, 1.0]. From wind/turbulence (Phase 1).
    pub mechanical_stress: f32,
}

impl EnvironmentState {
    /// Extract the normalized [0.0, 1.0] signal value for a given EnvironmentalSignal.
    /// Temperature is normalized via a bell curve centered at 22 C (plant thermal optimum).
    /// Photoperiod is normalized to [0, 1] as hours/24.
    /// Nitrate is normalized with Km = 0.5 mM half-saturation.
    pub fn signal_value(&self, signal: EnvironmentalSignal) -> f32 {
        match signal {
            EnvironmentalSignal::Temperature => {
                // Gaussian thermal response centered at 22 C, sigma ~10 C
                // (Adapted from Yin et al., 1995, Ann Bot 76:85-92)
                let opt = 22.0f32;
                let sigma = 10.0f32;
                let diff = self.temperature_c - opt;
                (-0.5 * (diff / sigma).powi(2)).exp()
            }
            EnvironmentalSignal::Moisture => self.soil_moisture.clamp(0.0, 1.0),
            EnvironmentalSignal::LightIntensity => self.light_intensity.clamp(0.0, 1.0),
            EnvironmentalSignal::Photoperiod => {
                // Normalize: 0h = 0.0, 24h = 1.0
                (self.photoperiod_hours / 24.0).clamp(0.0, 1.0)
            }
            EnvironmentalSignal::NutrientNitrogen => {
                // Michaelis-Menten: nitrate / (nitrate + 0.5)
                // NRT2.1 half-saturation ~0.2 mM, using 0.5 as generic Km
                let km = 0.5f32;
                self.soil_nitrate.max(0.0) / (self.soil_nitrate.max(0.0) + km)
            }
            EnvironmentalSignal::AbscisicAcid => self.internal_aba.clamp(0.0, 1.0),
            EnvironmentalSignal::RedFarRedRatio => {
                // Normalize: R:FR 1.5 maps to 1.0, 0 maps to 0.0.
                // Full sun R:FR ≈ 1.2 → signal ~0.8. Deep shade R:FR ≈ 0.3 → signal ~0.2.
                (self.r_fr_ratio / 1.5).clamp(0.0, 1.0)
            }
            EnvironmentalSignal::NeighborVOC => self.neighbor_voc.clamp(0.0, 1.0),
            EnvironmentalSignal::JasmonicAcid => self.jasmonic_acid.clamp(0.0, 1.0),
            EnvironmentalSignal::SalicylicAcid => self.salicylic_acid.clamp(0.0, 1.0),
            EnvironmentalSignal::MechanicalStress => self.mechanical_stress.clamp(0.0, 1.0),
        }
    }
}

// ---------------------------------------------------------------------------
// PlantGeneCircuit constructors and GRN execution
// ---------------------------------------------------------------------------

impl PlantGeneCircuit {
    /// Create an activator gene circuit with default kinetic parameters.
    /// Hill n=2, Km=0.5, max_expression=1.0, decay=0.002/s.
    pub fn activator(gene_id: &str, inputs: Vec<EnvironmentalSignal>) -> Self {
        Self {
            gene_id: gene_id.to_string(),
            inputs,
            expression: 0.0,
            hill_n: default_hill_coefficient(),
            km: default_km(),
            max_expression: default_max_expression(),
            decay_rate: default_decay_rate(),
            is_repressor: false,
        }
    }

    /// Create a repressor gene circuit. Expression is high when signal is absent
    /// and drops as signal increases.
    pub fn repressor(gene_id: &str, inputs: Vec<EnvironmentalSignal>) -> Self {
        Self {
            gene_id: gene_id.to_string(),
            inputs,
            expression: 1.0, // repressors start active
            hill_n: default_hill_coefficient(),
            km: default_km(),
            max_expression: default_max_expression(),
            decay_rate: default_decay_rate(),
            is_repressor: true,
        }
    }

    /// Create an activator with custom kinetic parameters.
    pub fn activator_with_kinetics(
        gene_id: &str,
        inputs: Vec<EnvironmentalSignal>,
        hill_n: f32,
        km: f32,
        max_expression: f32,
        decay_rate: f32,
    ) -> Self {
        Self {
            gene_id: gene_id.to_string(),
            inputs,
            expression: 0.0,
            hill_n,
            km,
            max_expression,
            decay_rate,
            is_repressor: false,
        }
    }

    /// Create a repressor with custom kinetic parameters.
    pub fn repressor_with_kinetics(
        gene_id: &str,
        inputs: Vec<EnvironmentalSignal>,
        hill_n: f32,
        km: f32,
        max_expression: f32,
        decay_rate: f32,
    ) -> Self {
        Self {
            gene_id: gene_id.to_string(),
            inputs,
            expression: 1.0,
            hill_n,
            km,
            max_expression,
            decay_rate,
            is_repressor: true,
        }
    }

    /// Evaluate the Hill function for a given signal level.
    ///
    /// For activators: h = s^n / (Km^n + s^n)
    /// For repressors: h = 1 - s^n / (Km^n + s^n) = Km^n / (Km^n + s^n)
    ///
    /// Returns value in [0.0, max_expression].
    pub fn hill_response(&self, signal: f32) -> f32 {
        let s = signal.max(0.0);
        let km_n = self.km.powf(self.hill_n);
        let s_n = s.powf(self.hill_n);
        let denom = km_n + s_n;
        if denom < 1e-12 {
            return if self.is_repressor {
                self.max_expression
            } else {
                0.0
            };
        }
        let activation = s_n / denom;
        if self.is_repressor {
            self.max_expression * (1.0 - activation)
        } else {
            self.max_expression * activation
        }
    }

    /// Update expression level based on environmental inputs.
    ///
    /// Uses Hill-function kinetics for the target expression level,
    /// then exponential approach with decay toward that target.
    ///
    /// When multiple inputs are present, the effective signal is the
    /// geometric mean of all input signals. This models AND-gate logic
    /// where all inputs must be present for full activation.
    ///
    /// d(expression)/dt = -decay_rate * (expression - target)
    pub fn update_expression(&mut self, env: &EnvironmentState, dt: f32) {
        if self.inputs.is_empty() {
            // Constitutive expression: drift toward max_expression
            let target = self.max_expression;
            let alpha = (-self.decay_rate * dt).exp();
            self.expression = self.expression * alpha + target * (1.0 - alpha);
            self.expression = self.expression.clamp(0.0, self.max_expression);
            return;
        }

        // Compute effective signal as geometric mean of all input signals.
        // Geometric mean models AND-gate: if any input is zero, output is zero.
        let mut log_sum = 0.0f32;
        let mut count = 0u32;
        let mut any_zero = false;
        for &signal_type in &self.inputs {
            let val = env.signal_value(signal_type);
            if val < 1e-9 {
                any_zero = true;
                break;
            }
            log_sum += val.ln();
            count += 1;
        }

        let effective_signal = if any_zero || count == 0 {
            0.0
        } else {
            (log_sum / count as f32).exp()
        };

        // Compute target expression via Hill function
        let target = self.hill_response(effective_signal);

        // Exponential approach to target
        // 1st-order ODE: dx/dt = -k*(x - target)
        // Solution: x(t+dt) = target + (x - target)*e^(-k*dt)
        let alpha = (-self.decay_rate * dt).exp();
        self.expression = target + (self.expression - target) * alpha;
        self.expression = self.expression.clamp(0.0, self.max_expression);
    }
}

// ---------------------------------------------------------------------------
// BotanicalGenome GRN stepping
// ---------------------------------------------------------------------------

impl BotanicalGenome {
    /// Step all gene circuits given current environment.
    /// Each circuit independently updates its expression level.
    pub fn step_gene_regulation(&mut self, env: &EnvironmentState, dt: f32) {
        for circuit in self.gene_circuits.values_mut() {
            circuit.update_expression(env, dt);
        }
    }

    /// Get expression level of a named gene circuit, or 0.0 if not found.
    pub fn gene_expression(&self, gene_id: &str) -> f32 {
        self.gene_circuits
            .get(gene_id)
            .map(|c| c.expression)
            .unwrap_or(0.0)
    }

    /// Collect all gene expression levels into a HashMap for metabolome coupling.
    pub fn expression_snapshot(&self) -> HashMap<String, f32> {
        self.gene_circuits
            .iter()
            .map(|(k, v)| (k.clone(), v.expression))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// GenomicCatalog
// ---------------------------------------------------------------------------

/// Global catalog of known species DNA.
pub struct GenomicCatalog {
    pub species: HashMap<u32, BotanicalSpecies>,
}

impl GenomicCatalog {
    pub fn new() -> Self {
        let mut catalog = Self {
            species: HashMap::new(),
        };
        catalog.register_defaults();
        catalog
    }

    fn register_defaults(&mut self) {
        for profile in botanical_species_profiles() {
            let species = BotanicalSpecies {
                name: profile.scientific_name.to_string(),
                common_name: profile.common_name.to_string(),
                taxonomy_id: profile.taxonomy_id,
                genome: BotanicalGenome {
                    chromosomes: self.build_symbolic_chromosomes(profile),
                    gene_circuits: self.build_species_circuits(profile),
                },
            };
            self.species.insert(species.taxonomy_id, species);
        }
    }

    fn build_symbolic_chromosomes(
        &self,
        profile: &BotanicalSpeciesProfile,
    ) -> Vec<PlantChromosome> {
        let prefix = profile
            .scientific_name
            .split_whitespace()
            .map(|part| part.chars().next().unwrap_or('X').to_ascii_uppercase())
            .collect::<String>();
        (1..=profile.symbolic_chromosomes)
            .map(|id| PlantChromosome {
                id,
                sequence: format!("{prefix}_CHR{id}_SIM"),
            })
            .collect()
    }

    fn build_species_circuits(
        &self,
        profile: &BotanicalSpeciesProfile,
    ) -> HashMap<String, PlantGeneCircuit> {
        let mut circuits = HashMap::new();

        // -- Universal circuits (all species) ---------------------------------
        circuits.insert(
            "GERMINATION".to_string(),
            PlantGeneCircuit::activator(
                "GERMINATION",
                vec![
                    EnvironmentalSignal::Moisture,
                    EnvironmentalSignal::Temperature,
                ],
            ),
        );
        circuits.insert(
            "FLOWERING_TIME".to_string(),
            PlantGeneCircuit::activator(
                "FLOWERING_TIME",
                vec![
                    EnvironmentalSignal::LightIntensity,
                    EnvironmentalSignal::Photoperiod,
                ],
            ),
        );
        circuits.insert(
            "DROUGHT_RESPONSE".to_string(),
            PlantGeneCircuit::activator(
                "DROUGHT_RESPONSE",
                vec![
                    EnvironmentalSignal::Moisture,
                    EnvironmentalSignal::Temperature,
                    EnvironmentalSignal::AbscisicAcid,
                ],
            ),
        );

        // -- Growth-form-specific circuits ------------------------------------
        match profile.growth_form {
            BotanicalGrowthForm::GrassClump => {
                circuits.insert(
                    "TILLERING".to_string(),
                    PlantGeneCircuit::activator(
                        "TILLERING",
                        vec![
                            EnvironmentalSignal::LightIntensity,
                            EnvironmentalSignal::NutrientNitrogen,
                        ],
                    ),
                );
                circuits.insert(
                    "CELL_WALL_REMODELING".to_string(),
                    PlantGeneCircuit::activator(
                        "CELL_WALL_REMODELING",
                        vec![
                            EnvironmentalSignal::Temperature,
                            EnvironmentalSignal::Moisture,
                        ],
                    ),
                );
            }
            BotanicalGrowthForm::FloatingAquatic => {
                circuits.insert(
                    "FROND_BUDDING".to_string(),
                    PlantGeneCircuit::activator(
                        "FROND_BUDDING",
                        vec![
                            EnvironmentalSignal::LightIntensity,
                            EnvironmentalSignal::Moisture,
                        ],
                    ),
                );
                circuits.insert(
                    "SURFACE_FLOATATION".to_string(),
                    PlantGeneCircuit::activator(
                        "SURFACE_FLOATATION",
                        vec![
                            EnvironmentalSignal::Moisture,
                            EnvironmentalSignal::Temperature,
                        ],
                    ),
                );
            }
            BotanicalGrowthForm::SubmergedAquatic => {
                circuits.insert(
                    "AERENCHYMA_PATTERNING".to_string(),
                    PlantGeneCircuit::activator(
                        "AERENCHYMA_PATTERNING",
                        vec![
                            EnvironmentalSignal::Moisture,
                            EnvironmentalSignal::LightIntensity,
                        ],
                    ),
                );
                circuits.insert(
                    "SUBMERGED_SHOOT_ELONGATION".to_string(),
                    PlantGeneCircuit::activator(
                        "SUBMERGED_SHOOT_ELONGATION",
                        vec![
                            EnvironmentalSignal::LightIntensity,
                            EnvironmentalSignal::Photoperiod,
                        ],
                    ),
                );
            }
            BotanicalGrowthForm::RosetteHerb => {
                circuits.insert(
                    "ROSETTE_EXPANSION".to_string(),
                    PlantGeneCircuit::activator(
                        "ROSETTE_EXPANSION",
                        vec![
                            EnvironmentalSignal::LightIntensity,
                            EnvironmentalSignal::Moisture,
                        ],
                    ),
                );
            }
            BotanicalGrowthForm::OrchardTree
            | BotanicalGrowthForm::StoneFruitTree
            | BotanicalGrowthForm::CitrusTree => {
                circuits.insert(
                    "FRUIT_DEVELOPMENT".to_string(),
                    PlantGeneCircuit::activator(
                        "FRUIT_DEVELOPMENT",
                        vec![
                            EnvironmentalSignal::Temperature,
                            EnvironmentalSignal::LightIntensity,
                        ],
                    ),
                );
                circuits.insert(
                    "WOOD_FORMATION".to_string(),
                    PlantGeneCircuit::activator(
                        "WOOD_FORMATION",
                        vec![
                            EnvironmentalSignal::NutrientNitrogen,
                            EnvironmentalSignal::Temperature,
                        ],
                    ),
                );
                circuits.insert(
                    "VOLATILE_EMISSION".to_string(),
                    PlantGeneCircuit::activator(
                        "VOLATILE_EMISSION",
                        vec![
                            EnvironmentalSignal::Temperature,
                            EnvironmentalSignal::LightIntensity,
                            EnvironmentalSignal::AbscisicAcid,
                        ],
                    ),
                );
            }
        }

        if matches!(profile.growth_form, BotanicalGrowthForm::CitrusTree) {
            circuits.insert(
                "TERPENE_BIOSYNTHESIS".to_string(),
                PlantGeneCircuit::activator(
                    "TERPENE_BIOSYNTHESIS",
                    vec![
                        EnvironmentalSignal::Temperature,
                        EnvironmentalSignal::LightIntensity,
                    ],
                ),
            );
            // Citrus: high citrate synthase, high limonene synthase
            circuits.insert(
                "CITRATE_SYNTHASE".to_string(),
                PlantGeneCircuit::activator_with_kinetics(
                    "CITRATE_SYNTHASE",
                    vec![EnvironmentalSignal::Temperature],
                    2.0,
                    0.3,   // Km: active at moderate temp
                    0.9,   // high max — citrus are citrate-dominant
                    0.002,
                ),
            );
            circuits.insert(
                "LIMONENE_SYNTHASE".to_string(),
                PlantGeneCircuit::activator_with_kinetics(
                    "LIMONENE_SYNTHASE",
                    vec![
                        EnvironmentalSignal::Temperature,
                        EnvironmentalSignal::LightIntensity,
                    ],
                    2.0,
                    0.4,
                    0.85,  // high limonene in citrus
                    0.002,
                ),
            );
        }

        if matches!(profile.growth_form, BotanicalGrowthForm::StoneFruitTree) {
            circuits.insert(
                "ANTHOCYANIN_LOADING".to_string(),
                PlantGeneCircuit::activator(
                    "ANTHOCYANIN_LOADING",
                    vec![
                        EnvironmentalSignal::Temperature,
                        EnvironmentalSignal::LightIntensity,
                    ],
                ),
            );
            // Stone fruit: benzaldehyde synthase + anthocyanin biosynthesis
            circuits.insert(
                "BENZALDEHYDE_SYNTHASE".to_string(),
                PlantGeneCircuit::activator_with_kinetics(
                    "BENZALDEHYDE_SYNTHASE",
                    vec![EnvironmentalSignal::Temperature],
                    2.0,
                    0.35,
                    0.80,  // strong stone-fruit aroma
                    0.002,
                ),
            );
            circuits.insert(
                "ANTHOCYANIN_BIOSYNTHESIS".to_string(),
                PlantGeneCircuit::activator_with_kinetics(
                    "ANTHOCYANIN_BIOSYNTHESIS",
                    vec![
                        EnvironmentalSignal::LightIntensity,
                        EnvironmentalSignal::Temperature,
                    ],
                    2.0,
                    0.4,
                    0.85,  // strong anthocyanin in cherry/peach
                    0.002,
                ),
            );
            // Stone fruit: moderate malate dehydrogenase (some acidity)
            circuits.insert(
                "MALATE_DEHYDROGENASE".to_string(),
                PlantGeneCircuit::activator_with_kinetics(
                    "MALATE_DEHYDROGENASE",
                    vec![EnvironmentalSignal::Temperature],
                    2.0,
                    0.4,
                    0.6,
                    0.002,
                ),
            );
        }

        if matches!(profile.growth_form, BotanicalGrowthForm::OrchardTree) {
            circuits.insert(
                "POME_CELL_EXPANSION".to_string(),
                PlantGeneCircuit::activator(
                    "POME_CELL_EXPANSION",
                    vec![
                        EnvironmentalSignal::Temperature,
                        EnvironmentalSignal::Moisture,
                    ],
                ),
            );
            // Pome fruit (apple, pear): high malate dehydrogenase + sorbitol dehydrogenase
            circuits.insert(
                "MALATE_DEHYDROGENASE".to_string(),
                PlantGeneCircuit::activator_with_kinetics(
                    "MALATE_DEHYDROGENASE",
                    vec![EnvironmentalSignal::Temperature],
                    2.0,
                    0.35,
                    0.85,  // high malate — pome fruits are malic acid dominant
                    0.002,
                ),
            );
            circuits.insert(
                "SORBITOL_DEHYDROGENASE".to_string(),
                PlantGeneCircuit::activator_with_kinetics(
                    "SORBITOL_DEHYDROGENASE",
                    vec![
                        EnvironmentalSignal::Temperature,
                        EnvironmentalSignal::Moisture,
                    ],
                    2.0,
                    0.4,
                    0.75,  // Rosaceae transport sugar
                    0.002,
                ),
            );
        }

        // -- Core molecular circuits for ALL fruit-bearing species --
        // (Arabidopsis-specific circuits are added separately below, but ALL species
        // need the universal metabolic circuits: RbcL, FT, PIN1, NRT2.1, CHS)
        if profile.taxonomy_id != 3702 {
            // Non-Arabidopsis species: core molecular circuits from literature
            // (same citations as Arabidopsis circuits — see emergent_rates.rs)
            if !circuits.contains_key("RbcL") {
                circuits.insert(
                    "RbcL".to_string(),
                    PlantGeneCircuit::activator_with_kinetics(
                        "RbcL",
                        vec![EnvironmentalSignal::LightIntensity],
                        literature_gene_hill_n("RbcL"),
                        literature_gene_km("RbcL"),
                        literature_gene_max_expression("RbcL"),
                        literature_gene_decay("RbcL"),
                    ),
                );
            }
            if !circuits.contains_key("FT") {
                circuits.insert(
                    "FT".to_string(),
                    PlantGeneCircuit::activator_with_kinetics(
                        "FT",
                        vec![EnvironmentalSignal::Photoperiod],
                        literature_gene_hill_n("FT"),
                        literature_gene_km("FT"),
                        literature_gene_max_expression("FT"),
                        literature_gene_decay("FT"),
                    ),
                );
            }
            if !circuits.contains_key("PIN1") {
                circuits.insert(
                    "PIN1".to_string(),
                    PlantGeneCircuit::activator_with_kinetics(
                        "PIN1",
                        vec![],
                        literature_gene_hill_n("PIN1"),
                        literature_gene_km("PIN1"),
                        literature_gene_max_expression("PIN1"),
                        literature_gene_decay("PIN1"),
                    ),
                );
            }
            if !circuits.contains_key("NRT2.1") {
                circuits.insert(
                    "NRT2.1".to_string(),
                    PlantGeneCircuit::repressor_with_kinetics(
                        "NRT2.1",
                        vec![EnvironmentalSignal::NutrientNitrogen],
                        literature_gene_hill_n("NRT2.1"),
                        literature_gene_km("NRT2.1"),
                        literature_gene_max_expression("NRT2.1"),
                        literature_gene_decay("NRT2.1"),
                    ),
                );
            }
            if !circuits.contains_key("CHS") {
                circuits.insert(
                    "CHS".to_string(),
                    PlantGeneCircuit::activator_with_kinetics(
                        "CHS",
                        vec![EnvironmentalSignal::LightIntensity],
                        literature_gene_hill_n("CHS"),
                        literature_gene_km("CHS"),
                        literature_gene_max_expression("CHS"),
                        literature_gene_decay("CHS"),
                    ),
                );
            }
            if !circuits.contains_key("DREB") {
                circuits.insert(
                    "DREB".to_string(),
                    PlantGeneCircuit::activator_with_kinetics(
                        "DREB",
                        vec![EnvironmentalSignal::AbscisicAcid],
                        literature_gene_hill_n("DREB"),
                        literature_gene_km("DREB"),
                        literature_gene_max_expression("DREB"),
                        literature_gene_decay("DREB"),
                    ),
                );
            }
        }

        // -- Arabidopsis-specific molecular circuits --------------------------
        if profile.taxonomy_id == 3702 {
            circuits.insert(
                "BOLTING".to_string(),
                PlantGeneCircuit::activator(
                    "BOLTING",
                    vec![
                        EnvironmentalSignal::Photoperiod,
                        EnvironmentalSignal::Temperature,
                    ],
                ),
            );
            // Merge in the Arabidopsis-specific molecular GRN circuits
            for (id, circuit) in Self::build_arabidopsis_circuits() {
                circuits.insert(id, circuit);
            }
        }

        // -- Universal circuits (continued) -----------------------------------
        circuits.insert(
            "ROOT_FORAGING".to_string(),
            PlantGeneCircuit::activator(
                "ROOT_FORAGING",
                vec![
                    EnvironmentalSignal::NutrientNitrogen,
                    EnvironmentalSignal::Moisture,
                ],
            ),
        );
        circuits.insert(
            "STOMATAL_CONTROL".to_string(),
            PlantGeneCircuit::activator(
                "STOMATAL_CONTROL",
                vec![
                    EnvironmentalSignal::Temperature,
                    EnvironmentalSignal::Moisture,
                    EnvironmentalSignal::AbscisicAcid,
                ],
            ),
        );

        // -- Phytochrome/Shade Avoidance circuits (all species) -------------------
        // PHYB: Phytochrome B — activated by high R:FR (full sun).
        // In direct sunlight (R:FR ≈ 1.2, signal ~0.8), PHYB is active.
        // Under canopy shade (R:FR < 0.5, signal < 0.33), PHYB drops.
        // (Quail, 2002, Nat Rev Mol Cell Biol 3:85; Franklin & Whitelam, 2005)
        // PHYB/SAS: Phytochrome shade detection (Li+ 2011, Casal 2012)
        circuits.insert(
            "PHYB".to_string(),
            PlantGeneCircuit::activator_with_kinetics(
                "PHYB",
                vec![EnvironmentalSignal::RedFarRedRatio],
                literature_gene_hill_n("PHYB"),
                literature_gene_km("PHYB"),
                literature_gene_max_expression("PHYB"),
                literature_gene_decay("PHYB"),
            ),
        );
        circuits.insert(
            "SAS".to_string(),
            PlantGeneCircuit::repressor_with_kinetics(
                "SAS",
                vec![EnvironmentalSignal::RedFarRedRatio],
                literature_gene_hill_n("SAS"),
                literature_gene_km("SAS"),
                literature_gene_max_expression("SAS"),
                literature_gene_decay("SAS")
            ),
        );

        // -- Defense signaling circuits (all species, Phase 5) ---------------------
        // JA_RESPONSE: Jasmonic acid response pathway, activated by mechanical stress.
        // Wound damage from wind or herbivory triggers JA biosynthesis via the
        // octadecanoid pathway. Hill n=3 for cooperative LOX/AOS enzyme cascade.
        // (Wasternack & Hause, 2013, Ann Bot 111:1021-1058)
        // Defense circuits: Wasternack+ 2013, Vlot+ 2009, Conrath+ 2006
        circuits.insert(
            "JA_RESPONSE".to_string(),
            PlantGeneCircuit::activator_with_kinetics(
                "JA_RESPONSE",
                vec![EnvironmentalSignal::MechanicalStress],
                literature_gene_hill_n("JA_RESPONSE"),
                literature_gene_km("JA_RESPONSE"),
                literature_gene_max_expression("JA_RESPONSE"),
                literature_gene_decay("JA_RESPONSE"),
            ),
        );
        circuits.insert(
            "SA_RESPONSE".to_string(),
            PlantGeneCircuit::activator_with_kinetics(
                "SA_RESPONSE",
                vec![EnvironmentalSignal::NeighborVOC],
                literature_gene_hill_n("SA_RESPONSE"),
                literature_gene_km("SA_RESPONSE"),
                literature_gene_max_expression("SA_RESPONSE"),
                literature_gene_decay("SA_RESPONSE"),
            ),
        );
        // DEFENSE_PRIMING: AND-gate requiring BOTH neighbor VOC AND internal JA
        // (Engelberth+ 2004, PNAS 101:1781-1785; Conrath+ 2006)
        circuits.insert(
            "DEFENSE_PRIMING".to_string(),
            PlantGeneCircuit::activator_with_kinetics(
                "DEFENSE_PRIMING",
                vec![
                    EnvironmentalSignal::NeighborVOC,
                    EnvironmentalSignal::JasmonicAcid,
                ],
                literature_gene_hill_n("DEFENSE_PRIMING"),
                literature_gene_km("DEFENSE_PRIMING"),
                literature_gene_max_expression("DEFENSE_PRIMING"),
                literature_gene_decay("DEFENSE_PRIMING"),
            ),
        );

        circuits
    }

    /// Build Arabidopsis-specific molecular gene circuits (taxonomy_id 3702).
    ///
    /// These represent well-characterized pathways from the model organism:
    /// - **RbcL**: RuBisCO large subunit, primary carbon fixation enzyme.
    ///   Light-activated, rapid response. (Portis, 2003, Photosyn Res)
    /// - **FLC**: FLOWERING LOCUS C, a MADS-box repressor of flowering.
    ///   Vernalization (cold) epigenetically silences FLC. (Michaels & Amasino, 1999)
    /// - **FT**: FLORIGEN, mobile flowering signal. Long-day activated.
    ///   Photoperiod threshold ~14h. (Corbesier et al., 2007, Science 316:1030)
    /// - **DREB**: Drought Response Element Binding protein.
    ///   ABA-responsive, activated under water deficit. (Yamaguchi-Shinozaki & Shinozaki, 2006)
    /// - **NRT2.1**: High-affinity nitrate transporter, induced by low nitrate.
    ///   Km ~0.2 mM. Repressed by high nitrate. (Glass et al., 2002)
    /// - **PIN1**: Auxin efflux carrier, near-constitutive expression.
    ///   (Wisniewska et al., 2006, Science 312:883)
    /// - **CHS**: Chalcone synthase, flavonoid/anthocyanin biosynthesis.
    ///   UV/light responsive. (Kubasek et al., 1992, Plant Cell 4:1229)
    fn build_arabidopsis_circuits() -> HashMap<String, PlantGeneCircuit> {
        let mut circuits = HashMap::new();

        // All kinetic parameters from literature — see emergent_rates.rs for citations.
        // RbcL: Portis 2003, FLC: He+ 2003, FT: Corbesier+ 2007, DREB: Sakuma+ 2006,
        // NRT2.1: Glass+ 2002, PIN1: Vieten+ 2005, CHS: Winkel-Shirley 2002
        circuits.insert(
            "RbcL".to_string(),
            PlantGeneCircuit::activator_with_kinetics(
                "RbcL",
                vec![EnvironmentalSignal::LightIntensity],
                literature_gene_hill_n("RbcL"),
                literature_gene_km("RbcL"),
                literature_gene_max_expression("RbcL"),
                literature_gene_decay("RbcL"),
            ),
        );

        circuits.insert(
            "FLC".to_string(),
            PlantGeneCircuit::repressor_with_kinetics(
                "FLC",
                vec![EnvironmentalSignal::Temperature],
                literature_gene_hill_n("FLC"),
                literature_gene_km("FLC"),
                literature_gene_max_expression("FLC"),
                literature_gene_decay("FLC"),
            ),
        );

        circuits.insert(
            "FT".to_string(),
            PlantGeneCircuit::activator_with_kinetics(
                "FT",
                vec![EnvironmentalSignal::Photoperiod],
                literature_gene_hill_n("FT"),
                literature_gene_km("FT"),
                literature_gene_max_expression("FT"),
                literature_gene_decay("FT"),
            ),
        );

        circuits.insert(
            "DREB".to_string(),
            PlantGeneCircuit::activator_with_kinetics(
                "DREB",
                vec![EnvironmentalSignal::AbscisicAcid],
                literature_gene_hill_n("DREB"),
                literature_gene_km("DREB"),
                literature_gene_max_expression("DREB"),
                literature_gene_decay("DREB"),
            ),
        );

        circuits.insert(
            "NRT2.1".to_string(),
            PlantGeneCircuit::repressor_with_kinetics(
                "NRT2.1",
                vec![EnvironmentalSignal::NutrientNitrogen],
                literature_gene_hill_n("NRT2.1"),
                literature_gene_km("NRT2.1"),
                literature_gene_max_expression("NRT2.1"),
                literature_gene_decay("NRT2.1"),
            ),
        );

        circuits.insert(
            "PIN1".to_string(),
            PlantGeneCircuit::activator_with_kinetics(
                "PIN1",
                vec![], // no inputs = constitutive
                literature_gene_hill_n("PIN1"),
                literature_gene_km("PIN1"),
                literature_gene_max_expression("PIN1"),
                literature_gene_decay("PIN1"),
            ),
        );

        circuits.insert(
            "CHS".to_string(),
            PlantGeneCircuit::activator_with_kinetics(
                "CHS",
                vec![EnvironmentalSignal::LightIntensity],
                literature_gene_hill_n("CHS"),
                literature_gene_km("CHS"),
                literature_gene_max_expression("CHS"),
                literature_gene_decay("CHS"),
            ),
        );

        circuits
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gene_circuit_hill_function() {
        // Activator with n=2, Km=0.5: at signal=0.5, expression should be 50% of max.
        let circuit = PlantGeneCircuit::activator_with_kinetics(
            "test",
            vec![EnvironmentalSignal::LightIntensity],
            2.0,
            0.5,
            1.0,
            0.002,
        );
        let at_km = circuit.hill_response(0.5);
        assert!(
            (at_km - 0.5).abs() < 0.01,
            "Hill function at Km should give 50% activation, got {}",
            at_km
        );

        // At signal=0, expression should be 0
        let at_zero = circuit.hill_response(0.0);
        assert!(
            at_zero < 0.01,
            "Hill function at signal=0 should be ~0, got {}",
            at_zero
        );

        // At signal=1.0 (>>Km), expression should be near max
        let at_high = circuit.hill_response(1.0);
        assert!(
            at_high > 0.75,
            "Hill function at signal>>Km should be near max, got {}",
            at_high
        );

        // Repressor: inverted response
        let repressor = PlantGeneCircuit::repressor_with_kinetics(
            "test_rep",
            vec![EnvironmentalSignal::Moisture],
            2.0,
            0.5,
            1.0,
            0.002,
        );
        let rep_at_km = repressor.hill_response(0.5);
        assert!(
            (rep_at_km - 0.5).abs() < 0.01,
            "Repressor Hill at Km should give 50%, got {}",
            rep_at_km
        );
        let rep_at_high = repressor.hill_response(1.0);
        assert!(
            rep_at_high < 0.25,
            "Repressor at high signal should be low, got {}",
            rep_at_high
        );
    }

    #[test]
    fn test_arabidopsis_default_circuits() {
        let catalog = GenomicCatalog::new();
        let arabidopsis = catalog
            .species
            .get(&3702)
            .expect("Arabidopsis should be in catalog");

        // Arabidopsis-specific molecular circuits
        let expected_circuits = ["RbcL", "FLC", "FT", "DREB", "NRT2.1", "PIN1", "CHS"];
        for name in &expected_circuits {
            assert!(
                arabidopsis.genome.gene_circuits.contains_key(*name),
                "Arabidopsis should have circuit '{}'",
                name,
            );
        }

        // Universal circuits should also be present
        assert!(arabidopsis
            .genome
            .gene_circuits
            .contains_key("GERMINATION"));
        assert!(arabidopsis
            .genome
            .gene_circuits
            .contains_key("STOMATAL_CONTROL"));

        // FLC should be a repressor
        let flc = arabidopsis.genome.gene_circuits.get("FLC").unwrap();
        assert!(flc.is_repressor, "FLC should be a repressor");

        // PIN1 should be constitutive (no inputs)
        let pin1 = arabidopsis.genome.gene_circuits.get("PIN1").unwrap();
        assert!(
            pin1.inputs.is_empty(),
            "PIN1 should be constitutive (no inputs)"
        );
    }

    #[test]
    fn test_grn_light_response() {
        // RbcL should increase expression with light.
        let catalog = GenomicCatalog::new();
        let mut genome = catalog.species.get(&3702).unwrap().genome.clone();

        // Dark environment
        let dark = EnvironmentState {
            light_intensity: 0.0,
            temperature_c: 22.0,
            ..Default::default()
        };

        // Bright environment
        let bright = EnvironmentState {
            light_intensity: 0.9,
            temperature_c: 22.0,
            ..Default::default()
        };

        // Run dark for many steps
        for _ in 0..500 {
            genome.step_gene_regulation(&dark, 1.0);
        }
        let dark_rbcl = genome.gene_expression("RbcL");

        // Reset and run bright
        let mut genome2 = catalog.species.get(&3702).unwrap().genome.clone();
        for _ in 0..500 {
            genome2.step_gene_regulation(&bright, 1.0);
        }
        let bright_rbcl = genome2.gene_expression("RbcL");

        assert!(
            bright_rbcl > dark_rbcl + 0.3,
            "RbcL expression should be much higher in light ({}) than dark ({})",
            bright_rbcl,
            dark_rbcl
        );
    }

    #[test]
    fn test_drought_response() {
        // DREB expression should increase with ABA (drought signal).
        let catalog = GenomicCatalog::new();

        // Well-watered (low ABA)
        let mut genome_wet = catalog.species.get(&3702).unwrap().genome.clone();
        let wet_env = EnvironmentState {
            internal_aba: 0.05,
            temperature_c: 22.0,
            soil_moisture: 0.9,
            ..Default::default()
        };
        for _ in 0..500 {
            genome_wet.step_gene_regulation(&wet_env, 1.0);
        }
        let wet_dreb = genome_wet.gene_expression("DREB");

        // Drought stress (high ABA)
        let mut genome_dry = catalog.species.get(&3702).unwrap().genome.clone();
        let dry_env = EnvironmentState {
            internal_aba: 0.8,
            temperature_c: 22.0,
            soil_moisture: 0.1,
            ..Default::default()
        };
        for _ in 0..500 {
            genome_dry.step_gene_regulation(&dry_env, 1.0);
        }
        let dry_dreb = genome_dry.gene_expression("DREB");

        assert!(
            dry_dreb > wet_dreb + 0.2,
            "DREB should be higher under drought ({}) than well-watered ({})",
            dry_dreb,
            wet_dreb
        );
    }

    #[test]
    fn test_constitutive_pin1() {
        // PIN1 has no inputs, so it should drift toward max_expression.
        let catalog = GenomicCatalog::new();
        let mut genome = catalog.species.get(&3702).unwrap().genome.clone();

        let env = EnvironmentState {
            temperature_c: 5.0,   // cold
            light_intensity: 0.0, // dark
            soil_moisture: 0.1,   // dry
            ..Default::default()
        };
        for _ in 0..2000 {
            genome.step_gene_regulation(&env, 1.0);
        }
        let pin1 = genome.gene_expression("PIN1");
        assert!(
            pin1 > 0.7,
            "Constitutive PIN1 should reach near max even in harsh conditions, got {}",
            pin1
        );
    }

    #[test]
    fn test_repressor_nrt21() {
        // NRT2.1 is repressed by high nitrate.
        let catalog = GenomicCatalog::new();

        // High nitrate
        let mut genome_high_n = catalog.species.get(&3702).unwrap().genome.clone();
        let high_n_env = EnvironmentState {
            soil_nitrate: 5.0,
            temperature_c: 22.0,
            ..Default::default()
        };
        for _ in 0..500 {
            genome_high_n.step_gene_regulation(&high_n_env, 1.0);
        }
        let high_n_expr = genome_high_n.gene_expression("NRT2.1");

        // Low nitrate
        let mut genome_low_n = catalog.species.get(&3702).unwrap().genome.clone();
        let low_n_env = EnvironmentState {
            soil_nitrate: 0.01,
            temperature_c: 22.0,
            ..Default::default()
        };
        for _ in 0..500 {
            genome_low_n.step_gene_regulation(&low_n_env, 1.0);
        }
        let low_n_expr = genome_low_n.gene_expression("NRT2.1");

        assert!(
            low_n_expr > high_n_expr + 0.3,
            "NRT2.1 should be higher under low nitrate ({}) than high ({})",
            low_n_expr,
            high_n_expr
        );
    }

    #[test]
    fn test_expression_snapshot() {
        let catalog = GenomicCatalog::new();
        let genome = catalog.species.get(&3702).unwrap().genome.clone();
        let snapshot = genome.expression_snapshot();
        assert_eq!(
            snapshot.len(),
            genome.gene_circuits.len(),
            "Snapshot should have same number of entries as circuits"
        );
        for (key, &val) in &snapshot {
            assert_eq!(val, genome.gene_expression(key));
        }
    }

    #[test]
    fn test_environment_signal_normalization() {
        let env = EnvironmentState {
            temperature_c: 22.0,
            soil_moisture: 0.5,
            light_intensity: 0.8,
            photoperiod_hours: 16.0,
            soil_nitrate: 0.5,
            internal_aba: 0.3,
            r_fr_ratio: 1.2,
            ..Default::default()
        };

        // Temperature at optimum should give ~1.0
        let temp_sig = env.signal_value(EnvironmentalSignal::Temperature);
        assert!(
            temp_sig > 0.95,
            "Temperature at 22C should be near 1.0, got {}",
            temp_sig
        );

        // Temperature far from optimum should be low
        let cold_env = EnvironmentState {
            temperature_c: 0.0,
            ..Default::default()
        };
        let cold_sig = cold_env.signal_value(EnvironmentalSignal::Temperature);
        assert!(
            cold_sig < 0.15,
            "Temperature at 0C should be low, got {}",
            cold_sig
        );

        // Photoperiod normalized
        let photo_sig = env.signal_value(EnvironmentalSignal::Photoperiod);
        let expected = 16.0 / 24.0;
        assert!(
            (photo_sig - expected).abs() < 0.01,
            "Photoperiod 16h should normalize to {}, got {}",
            expected,
            photo_sig
        );

        // Nitrate at Km should give 0.5
        let nitrate_sig = env.signal_value(EnvironmentalSignal::NutrientNitrogen);
        assert!(
            (nitrate_sig - 0.5).abs() < 0.01,
            "Nitrate at Km should give 0.5, got {}",
            nitrate_sig
        );
    }

    #[test]
    fn test_non_arabidopsis_species_have_core_molecular_circuits() {
        // Apple (3750) should now have universal molecular circuits (RbcL, FT, PIN1, etc.)
        // but NOT Arabidopsis-specific FLC
        let catalog = GenomicCatalog::new();
        let apple = catalog
            .species
            .get(&3750)
            .expect("Apple should be in catalog");
        // Core molecular circuits present on all species
        assert!(
            apple.genome.gene_circuits.contains_key("RbcL"),
            "Apple should have universal RbcL circuit"
        );
        assert!(
            apple.genome.gene_circuits.contains_key("PIN1"),
            "Apple should have universal PIN1 circuit"
        );
        assert!(
            apple.genome.gene_circuits.contains_key("FT"),
            "Apple should have universal FT circuit"
        );
        assert!(
            apple.genome.gene_circuits.contains_key("NRT2.1"),
            "Apple should have universal NRT2.1 circuit"
        );
        // FLC is Arabidopsis-specific (vernalization epigenetic memory)
        assert!(
            !apple.genome.gene_circuits.contains_key("FLC"),
            "Apple should not have Arabidopsis-specific FLC circuit"
        );
        // Apple-specific (OrchardTree) enzyme circuits
        assert!(apple.genome.gene_circuits.contains_key("MALATE_DEHYDROGENASE"));
        assert!(apple.genome.gene_circuits.contains_key("SORBITOL_DEHYDROGENASE"));
        // Universal circuits still present
        assert!(apple.genome.gene_circuits.contains_key("GERMINATION"));
        assert!(apple.genome.gene_circuits.contains_key("FRUIT_DEVELOPMENT"));
    }

    #[test]
    fn test_citrus_has_citrate_and_limonene_circuits() {
        let catalog = GenomicCatalog::new();
        let orange = catalog
            .species
            .get(&2711)
            .expect("Orange should be in catalog");
        assert!(orange.genome.gene_circuits.contains_key("CITRATE_SYNTHASE"));
        assert!(orange.genome.gene_circuits.contains_key("LIMONENE_SYNTHASE"));
        // Citrus should NOT have stone-fruit circuits
        assert!(!orange.genome.gene_circuits.contains_key("BENZALDEHYDE_SYNTHASE"));
        assert!(!orange.genome.gene_circuits.contains_key("SORBITOL_DEHYDROGENASE"));
    }

    #[test]
    fn test_stone_fruit_has_benzaldehyde_and_anthocyanin() {
        let catalog = GenomicCatalog::new();
        let cherry = catalog
            .species
            .get(&42229)
            .expect("Cherry should be in catalog");
        assert!(cherry.genome.gene_circuits.contains_key("BENZALDEHYDE_SYNTHASE"));
        assert!(cherry.genome.gene_circuits.contains_key("ANTHOCYANIN_BIOSYNTHESIS"));
        // Stone fruit should NOT have citrus circuits
        assert!(!cherry.genome.gene_circuits.contains_key("CITRATE_SYNTHASE"));
        assert!(!cherry.genome.gene_circuits.contains_key("LIMONENE_SYNTHASE"));
    }

    // -- Phytochrome/SAS tests (Phase 4) --

    #[test]
    fn test_phyb_activates_in_full_sun() {
        let catalog = GenomicCatalog::new();
        let mut genome = catalog.species.get(&3702).unwrap().genome.clone();
        let env = EnvironmentState {
            r_fr_ratio: 1.2,
            light_intensity: 0.9,
            temperature_c: 22.0,
            ..Default::default()
        };
        for _ in 0..500 {
            genome.step_gene_regulation(&env, 1.0);
        }
        let phyb = genome.gene_expression("PHYB");
        assert!(phyb > 0.5, "PHYB should be active in full sun: {phyb}");
    }

    #[test]
    fn test_sas_activates_in_shade() {
        let catalog = GenomicCatalog::new();
        let mut genome = catalog.species.get(&3702).unwrap().genome.clone();
        let shade_env = EnvironmentState {
            r_fr_ratio: 0.3,
            light_intensity: 0.2,
            temperature_c: 22.0,
            ..Default::default()
        };
        for _ in 0..500 {
            genome.step_gene_regulation(&shade_env, 1.0);
        }
        let sas = genome.gene_expression("SAS");
        assert!(sas > 0.5, "SAS should activate in deep shade: {sas}");
    }

    #[test]
    fn test_all_species_have_phyb_and_sas() {
        let catalog = GenomicCatalog::new();
        for (tid, species) in &catalog.species {
            assert!(
                species.genome.gene_circuits.contains_key("PHYB"),
                "Species {} (tid={}) should have PHYB circuit",
                species.name,
                tid,
            );
            assert!(
                species.genome.gene_circuits.contains_key("SAS"),
                "Species {} (tid={}) should have SAS circuit",
                species.name,
                tid,
            );
        }
    }

    // -- Phase 5: Defense signaling / VOC gene circuit tests --

    #[test]
    fn test_ja_response_activates_with_mechanical_stress() {
        let catalog = GenomicCatalog::new();
        let mut genome = catalog.species.get(&3702).unwrap().genome.clone();
        let damaged_env = EnvironmentState {
            mechanical_stress: 0.7,
            temperature_c: 22.0,
            light_intensity: 0.5,
            ..Default::default()
        };
        for _ in 0..500 {
            genome.step_gene_regulation(&damaged_env, 1.0);
        }
        let ja = genome.gene_expression("JA_RESPONSE");
        assert!(
            ja > 0.5,
            "JA_RESPONSE should activate under mechanical stress: {ja}"
        );
    }

    #[test]
    fn test_defense_priming_requires_both_signals() {
        let catalog = GenomicCatalog::new();

        // Only neighbor VOC, no JA
        let mut g1 = catalog.species.get(&3702).unwrap().genome.clone();
        let voc_only = EnvironmentState {
            neighbor_voc: 0.8,
            jasmonic_acid: 0.0,
            temperature_c: 22.0,
            ..Default::default()
        };
        for _ in 0..500 {
            g1.step_gene_regulation(&voc_only, 1.0);
        }
        let priming_voc_only = g1.gene_expression("DEFENSE_PRIMING");

        // Both signals
        let mut g2 = catalog.species.get(&3702).unwrap().genome.clone();
        let both = EnvironmentState {
            neighbor_voc: 0.8,
            jasmonic_acid: 0.8,
            temperature_c: 22.0,
            ..Default::default()
        };
        for _ in 0..500 {
            g2.step_gene_regulation(&both, 1.0);
        }
        let priming_both = g2.gene_expression("DEFENSE_PRIMING");

        assert!(
            priming_both > priming_voc_only + 0.1,
            "DEFENSE_PRIMING needs both signals: both={priming_both}, voc_only={priming_voc_only}"
        );
    }

    #[test]
    fn test_sa_response_activates_with_neighbor_voc() {
        let catalog = GenomicCatalog::new();
        let mut genome = catalog.species.get(&3702).unwrap().genome.clone();
        let voc_env = EnvironmentState {
            neighbor_voc: 0.7,
            temperature_c: 22.0,
            ..Default::default()
        };
        for _ in 0..500 {
            genome.step_gene_regulation(&voc_env, 1.0);
        }
        let sa = genome.gene_expression("SA_RESPONSE");
        assert!(
            sa > 0.4,
            "SA_RESPONSE should activate with neighbor VOC: {sa}"
        );
    }

    #[test]
    fn test_all_species_have_defense_circuits() {
        let catalog = GenomicCatalog::new();
        for (tid, species) in &catalog.species {
            assert!(
                species.genome.gene_circuits.contains_key("JA_RESPONSE"),
                "Species {} (tid={}) should have JA_RESPONSE circuit",
                species.name,
                tid,
            );
            assert!(
                species.genome.gene_circuits.contains_key("SA_RESPONSE"),
                "Species {} (tid={}) should have SA_RESPONSE circuit",
                species.name,
                tid,
            );
            assert!(
                species.genome.gene_circuits.contains_key("DEFENSE_PRIMING"),
                "Species {} (tid={}) should have DEFENSE_PRIMING circuit",
                species.name,
                tid,
            );
        }
    }
}
