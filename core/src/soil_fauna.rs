//! Soil fauna bioturbation: earthworms and nematodes.
//!
//! This module adds the mesofauna trophic level between soil microbes and
//! plant roots.  Earthworms mix soil layers (bioturbation), enhance aeration,
//! and accelerate organic matter decomposition.  Nematodes graze on microbial
//! populations and excrete ammonium, driving a significant fraction of soil
//! nitrogen mineralization.
//!
//! # Literature
//!
//! - Edwards & Bohlen 1996, *Biology and Ecology of Earthworms*, 3rd ed.
//! - Lee 1985, *Earthworms: Their Ecology and Relationships with Soils and Land Use*.
//! - Boudreau 1997, *Diagenetic Models and Their Implementation*, Springer.
//! - Darwin 1881, *The Formation of Vegetable Mould Through the Action of Worms*.
//! - Yeates et al. 1993, *Soil Biology and Biochemistry* 25:869.
//! - Bongers & Ferris 1999, *Trends in Ecology & Evolution* 14:224.
//! - Ingham et al. 1985, *Ecological Monographs* 55:119.
//! - Lavelle 1988, *Soil Biology and Biochemistry* 20:263.
//! - Bradford 2002, *Seed Ecology, Dispersal and Germination*.

use crate::constants::clamp;
use crate::terrarium::{BatchedAtomTerrarium, TerrariumSpecies, TERRARIUM_SPECIES_COUNT};

// ── Earthworm constants (literature-sourced) ────────────────────────────
//
// Carrying capacity: 100-400 ind/m^2 (Edwards & Bohlen 1996, Table 3.2).
// We use 200 as the mid-range for a temperate loam terrarium.
const EARTHWORM_CARRYING_CAPACITY: f32 = 200.0; // ind/m^2

// Intrinsic growth rate: r ~ 0.01-0.03 d^-1 for Lumbricus terrestris
// (Lavelle 1988).  Reflects slow reproduction (cocoon → hatchling).
const EARTHWORM_R_MAX: f32 = 0.02; // d^-1

// Feeding rate: 10-30 mg dry soil / g body weight / day (Lee 1985).
// Expressed as fraction of organic matter consumed per unit biomass per day.
const EARTHWORM_FEEDING_RATE: f32 = 0.02; // g OM / g worm / day

// Bioturbation diffusivity: 0.1-10 mm^2/day (Boudreau 1997, Ch. 9).
// We set a moderate default, scaled by local population density.
const EARTHWORM_D_BIO_MAX: f32 = 5.0; // mm^2/day at carrying capacity

// Carbon assimilation efficiency from ingested organic matter.
const EARTHWORM_ASSIMILATION_EFFICIENCY: f32 = 0.15; // 15%

// Maintenance respiration: fraction of biomass respired per day.
const EARTHWORM_MAINTENANCE: f32 = 0.005; // d^-1

// Death rate: base mortality.
const EARTHWORM_DEATH_RATE: f32 = 0.003; // d^-1

// Average individual body mass (g C).  Lumbricus terrestris: 1-5 g fresh
// weight, ~15% C content (Edwards & Bohlen 1996).  Use 0.3 g C as mid-range.
const EARTHWORM_INDIVIDUAL_MASS_G_C: f32 = 0.3;

// Temperature response: Gaussian centered at 12.5 C, sigma 7 C.
// Active range approximately 5-25 C (Edwards & Bohlen 1996, Ch. 4).
const EARTHWORM_T_OPT: f32 = 12.5; // C
const EARTHWORM_T_SIGMA: f32 = 7.0; // C

// Hydration thresholds: earthworms are aerobic obligates.
// They require moist but not waterlogged soil.
const EARTHWORM_HYDRATION_MIN: f32 = 0.15; // too dry
const EARTHWORM_HYDRATION_OPT: f32 = 0.55; // optimal
const EARTHWORM_HYDRATION_MAX: f32 = 0.90; // waterlogged = anaerobic

// N:C ratio of earthworm tissue.  Excretion releases excess N as NH4+.
#[allow(dead_code)]
const EARTHWORM_TISSUE_NC_RATIO: f32 = 0.10;

// Fraction of ingested N excreted as NH4+ (mucus + urine).
const EARTHWORM_N_EXCRETION_FRACTION: f32 = 0.60;

// P:C ratio of earthworm tissue.
#[allow(dead_code)]
const EARTHWORM_TISSUE_PC_RATIO: f32 = 0.012;

// Fraction of ingested P released (cast enrichment, Lavelle 1988).
const EARTHWORM_P_RELEASE_FRACTION: f32 = 0.40;

// ── Nematode constants (literature-sourced) ─────────────────────────────
//
// Population density: 1-100 individuals/g soil (Yeates et al. 1993).
// Bacterial feeders dominate: ~60% of nematode community.
const NEMATODE_CARRYING_CAPACITY: f32 = 40.0; // ind/g soil

// Feeding rate: ~5x body weight/day for bacterial feeders (Ferris 2010).
const NEMATODE_FEEDING_RATE: f32 = 5.0; // body-weights / day

// Growth efficiency: fraction of ingested C assimilated into nematode biomass.
const NEMATODE_ASSIMILATION_EFFICIENCY: f32 = 0.25;

// Individual body mass (ug C).  Typical bacterial feeder ~0.05 ug C.
const NEMATODE_INDIVIDUAL_MASS_UG_C: f32 = 0.05;

// Growth rate: Lotka-Volterra intrinsic rate.
const NEMATODE_R_MAX: f32 = 0.10; // d^-1

// Death rate (starvation, predation).
const NEMATODE_DEATH_RATE: f32 = 0.02; // d^-1

// N mineralization: 10-30% of soil N flux comes from nematode excretion
// (Ingham et al. 1985).  NH4+ excretion per unit C consumed.
const NEMATODE_N_EXCRETION_PER_C: f32 = 0.15; // g N / g C consumed

// Temperature response: active 5-30 C, optimum ~20 C (Yeates 1981).
const NEMATODE_T_OPT: f32 = 20.0; // C
const NEMATODE_T_SIGMA: f32 = 8.0; // C

// Hydration: nematodes need water films to move.
const NEMATODE_HYDRATION_MIN: f32 = 0.10;
const NEMATODE_HYDRATION_OPT: f32 = 0.50;
const NEMATODE_HYDRATION_MAX: f32 = 0.95;

// ── Helper: Gaussian temperature response ───────────────────────────────

/// Gaussian temperature scaling factor in [0, 1].
///
/// Returns `exp(-0.5 * ((T - T_opt) / sigma)^2)`.
#[inline]
fn temperature_response(temp_c: f32, t_opt: f32, sigma: f32) -> f32 {
    let z = (temp_c - t_opt) / sigma;
    (-0.5 * z * z).exp()
}

/// Hydration scaling with optimum and boundary penalties.
///
/// Returns a factor in [0, 1] that peaks at `h_opt`, decays to zero
/// below `h_min` or above `h_max`.
#[inline]
fn hydration_response(h: f32, h_min: f32, h_opt: f32, h_max: f32) -> f32 {
    if h <= h_min || h >= h_max {
        return 0.0;
    }
    if h <= h_opt {
        (h - h_min) / (h_opt - h_min)
    } else {
        (h_max - h) / (h_max - h_opt)
    }
}

// ── Earthworm Molecular/DNA Fidelity Layer ─────────────────────────────────
//
// Lumbricus terrestris (common earthworm) molecular biology.
// Sources: Earthworm Genome Portal, Zhang et al. 2020 (Nature Comm),
// Edwards & Bohlen 1996.

/// Earthworm genome statistics.
/// Lumbricus terrestris genome assembly (Zhang et al. 2020).
pub mod earthworm_genome_stats {
    /// Total genome size in megabases.
    pub const GENOME_MB: f32 = 1060.0; // ~1.06 Gb
    /// Number of protein-coding genes.
    pub const PROTEIN_GENES: usize = 32_000;
    /// Number of chromosomes (varies by species, Lumbricus has many).
    pub const CHROMOSOME_COUNT: usize = 36;
    /// GC content percentage.
    pub const GC_CONTENT: f32 = 37.0;
    /// Genome repeat content percentage.
    pub const REPEAT_CONTENT: f32 = 58.0;
}

/// Earthworm gene function categories.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EarthwormGeneFunction {
    /// Digestive enzymes (cellulase, chitinase, protease).
    DigestiveEnzyme,
    /// Cuticle/mucus proteins.
    CuticleProtein,
    /// Hemoglobin (earthworms have blood with hemoglobin).
    Hemoglobin,
    /// Muscle and locomotion.
    Locomotion,
    /// Chloragogenous tissue (heavy metal detox, immune).
    Chloragocyte,
    /// Reproductive system.
    Reproductive,
    /// Neural/sensory.
    NeuralSensory,
    /// Stress response.
    StressResponse,
}

/// Key earthworm gene with functional annotation.
#[derive(Clone, Debug)]
pub struct EarthwormGene {
    /// Gene name/symbol.
    pub name: String,
    /// Functional description.
    pub function: String,
    /// Gene function category.
    pub category: EarthwormGeneFunction,
    /// Expression level (relative, 0-1).
    pub expression_level: f32,
}

/// Earthworm body segment structure.
#[derive(Clone, Debug, Default)]
pub struct EarthwormSegment {
    /// Segment number (1 = prostomium, 2 = peristomium, etc.).
    pub segment_id: u8,
    /// Has setae (bristles).
    pub has_setae: bool,
    /// Contains nephridia (kidney analogs).
    pub has_nephridia: bool,
    /// Muscle ring thickness.
    pub muscle_thickness: f32,
}

/// Earthworm body anatomy.
#[derive(Clone, Debug)]
pub struct EarthwormBody {
    /// Total segment count (typically ~150).
    pub segments: u8,
    /// Clitellum position (segments where it appears, e.g., 32-37).
    pub clitellum_range: (u8, u8),
    /// Body length in cm.
    pub length_cm: f32,
    /// Body diameter in mm.
    pub diameter_mm: f32,
    /// Number of setae per segment (typically 8 in Lumbricus).
    pub setae_per_segment: u8,
    /// Has dorsal pores.
    pub dorsal_pores: bool,
    /// Circulatory system: dorsal vessel pulsations per minute.
    pub heart_rate_bpm: u16,
    /// Digestive system compartments.
    pub gut_divisions: Vec<String>,
}

impl Default for EarthwormBody {
    fn default() -> Self {
        Self {
            segments: 150,
            clitellum_range: (32, 37),
            length_cm: 20.0,
            diameter_mm: 6.0,
            setae_per_segment: 8,
            dorsal_pores: true,
            heart_rate_bpm: 10, // slow, earthworms have low metabolic rate
            gut_divisions: vec![
                "buccal_cavity".into(),
                "pharynx".into(),
                "esophagus".into(),
                "crop".into(),
                "gizzard".into(),
                "intestine".into(),
            ],
        }
    }
}

/// Key earthworm genes.
pub fn earthworm_key_genes() -> Vec<EarthwormGene> {
    vec![
        // Digestive enzymes
        EarthwormGene {
            name: "LtCel".into(),
            function: "Cellulase - breaks down plant cell walls".into(),
            category: EarthwormGeneFunction::DigestiveEnzyme,
            expression_level: 0.8,
        },
        EarthwormGene {
            name: "LtChi".into(),
            function: "Chitinase - digests fungal cell walls".into(),
            category: EarthwormGeneFunction::DigestiveEnzyme,
            expression_level: 0.6,
        },
        EarthwormGene {
            name: "LtPro".into(),
            function: "Protease - protein digestion".into(),
            category: EarthwormGeneFunction::DigestiveEnzyme,
            expression_level: 0.7,
        },
        EarthwormGene {
            name: "LtAmy".into(),
            function: "Amylase - starch digestion".into(),
            category: EarthwormGeneFunction::DigestiveEnzyme,
            expression_level: 0.5,
        },
        // Hemoglobin (earthworm blood)
        EarthwormGene {
            name: "LtHbA".into(),
            function: "Hemoglobin alpha chain - oxygen transport".into(),
            category: EarthwormGeneFunction::Hemoglobin,
            expression_level: 0.9,
        },
        EarthwormGene {
            name: "LtHbB".into(),
            function: "Hemoglobin beta chain - oxygen transport".into(),
            category: EarthwormGeneFunction::Hemoglobin,
            expression_level: 0.9,
        },
        // Cuticle proteins
        EarthwormGene {
            name: "LtCut1".into(),
            function: "Cuticle collagen - protective outer layer".into(),
            category: EarthwormGeneFunction::CuticleProtein,
            expression_level: 0.7,
        },
        EarthwormGene {
            name: "LtMus1".into(),
            function: "Mucus glycoprotein - slime production".into(),
            category: EarthwormGeneFunction::CuticleProtein,
            expression_level: 0.8,
        },
        // Locomotion
        EarthwormGene {
            name: "LtMLC".into(),
            function: "Myosin light chain - circular muscle".into(),
            category: EarthwormGeneFunction::Locomotion,
            expression_level: 0.6,
        },
        EarthwormGene {
            name: "LtAct".into(),
            function: "Actin - longitudinal muscle".into(),
            category: EarthwormGeneFunction::Locomotion,
            expression_level: 0.6,
        },
        // Chloragocytes (detox)
        EarthwormGene {
            name: "LtMT".into(),
            function: "Metallothionein - heavy metal binding".into(),
            category: EarthwormGeneFunction::Chloragocyte,
            expression_level: 0.4,
        },
        EarthwormGene {
            name: "LtSOD".into(),
            function: "Superoxide dismutase - oxidative stress".into(),
            category: EarthwormGeneFunction::Chloragocyte,
            expression_level: 0.5,
        },
        EarthwormGene {
            name: "LtCAT".into(),
            function: "Catalase - peroxide detoxification".into(),
            category: EarthwormGeneFunction::Chloragocyte,
            expression_level: 0.5,
        },
        // Reproductive
        EarthwormGene {
            name: "LtVit".into(),
            function: "Vitellogenin - egg yolk protein".into(),
            category: EarthwormGeneFunction::Reproductive,
            expression_level: 0.3,
        },
        // Neural/sensory
        EarthwormGene {
            name: "LtSyn".into(),
            function: "Synaptotagmin - neurotransmitter release".into(),
            category: EarthwormGeneFunction::NeuralSensory,
            expression_level: 0.4,
        },
        EarthwormGene {
            name: "LtPho".into(),
            function: "Photoreceptor protein - light sensitivity".into(),
            category: EarthwormGeneFunction::NeuralSensory,
            expression_level: 0.3,
        },
        // Stress response
        EarthwormGene {
            name: "LtHSP70".into(),
            function: "Heat shock protein 70 - protein folding".into(),
            category: EarthwormGeneFunction::StressResponse,
            expression_level: 0.2,
        },
        EarthwormGene {
            name: "LtGST".into(),
            function: "Glutathione S-transferase - detoxification".into(),
            category: EarthwormGeneFunction::StressResponse,
            expression_level: 0.3,
        },
    ]
}

/// Earthworm molecular pathways.
pub fn earthworm_pathways() -> Vec<(&'static str, Vec<&'static str>, &'static str)> {
    vec![
        (
            "Cellulose digestion",
            vec!["LtCel", "LtAmy"],
            "Breaks down plant cellulose and starch",
        ),
        (
            "Oxygen transport",
            vec!["LtHbA", "LtHbB"],
            "Hemoglobin-based O2 transport in blood",
        ),
        (
            "Heavy metal detox",
            vec!["LtMT", "LtGST"],
            "Chloragocyte-mediated metal sequestration",
        ),
        (
            "Oxidative stress response",
            vec!["LtSOD", "LtCAT"],
            "Superoxide and peroxide neutralization",
        ),
        (
            "Peristaltic locomotion",
            vec!["LtMLC", "LtAct", "LtMus1"],
            "Circular and longitudinal muscle coordination",
        ),
        (
            "Mucus production",
            vec!["LtMus1", "LtCut1"],
            "Slime for respiration and movement",
        ),
    ]
}

// ── Nematode Molecular/DNA Fidelity Layer ───────────────────────────────────
//
// Caenorhabditis elegans is the model nematode, well-characterized.
// See celegans.rs module for full connectome and molecular detail.

/// Nematode genome statistics (C. elegans as reference).
pub mod nematode_genome_stats {
    /// Total genome size in megabases.
    pub const GENOME_MB: f32 = 100.0; // ~100 Mb
    /// Number of protein-coding genes.
    pub const PROTEIN_GENES: usize = 20_470;
    /// Number of chromosomes (6 + mitochondria).
    pub const CHROMOSOME_COUNT: usize = 6;
    /// GC content percentage.
    pub const GC_CONTENT: f32 = 35.0;
    /// Number of neurons (hermaphrodite).
    pub const NEURON_COUNT: usize = 302;
    /// Number of muscle cells.
    pub const MUSCLE_COUNT: usize = 95;
    /// Lifespan at 20C in days.
    pub const LIFESPAN_DAYS: f32 = 14.0;
}

/// Nematode feeding guild types with molecular characteristics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NematodeFeedingGuild {
    /// Bacterial feeders - most common soil nematodes.
    BacterialFeeder,
    /// Fungal feeders - stylet pierces hyphae.
    FungalFeeder,
    /// Plant parasites - stylet penetrates roots.
    PlantParasite,
    /// Predators - eat other nematodes/protozoa.
    Predator,
    /// Omnivores - feed on multiple trophic levels.
    Omnivore,
}

/// Nematode body structure summary.
#[derive(Clone, Debug, Default)]
pub struct NematodeBody {
    /// Body length in mm.
    pub length_mm: f32,
    /// Body diameter in μm.
    pub diameter_um: f32,
    /// Has stylet (for feeding).
    pub has_stylet: bool,
    /// Stylet length in μm (if present).
    pub stylet_length_um: f32,
    /// Cuticle layer count.
    pub cuticle_layers: u8,
}

impl NematodeBody {
    /// Create bacterial-feeding nematode body (e.g., C. elegans).
    pub fn bacterial_feeder() -> Self {
        Self {
            length_mm: 1.0,
            diameter_um: 80.0,
            has_stylet: false,
            stylet_length_um: 0.0,
            cuticle_layers: 4,
        }
    }

    /// Create fungal-feeding nematode body.
    pub fn fungal_feeder() -> Self {
        Self {
            length_mm: 0.8,
            diameter_um: 60.0,
            has_stylet: true,
            stylet_length_um: 15.0,
            cuticle_layers: 4,
        }
    }

    /// Create plant-parasitic nematode body.
    pub fn plant_parasite() -> Self {
        Self {
            length_mm: 0.5,
            diameter_um: 30.0,
            has_stylet: true,
            stylet_length_um: 20.0,
            cuticle_layers: 3,
        }
    }
}

// ── Earthworm population ────────────────────────────────────────────────

/// Earthworm population state on a 2D soil grid (width x height).
///
/// Biomass is stored as g C per cell, population density as individuals/m^2.
/// Bioturbation rate is the effective diffusive mixing coefficient.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EarthwormPopulation {
    /// Biomass per 2D grid cell (g C / cell).
    pub biomass_per_voxel: Vec<f32>,
    /// Population density (individuals / m^2).
    pub population_density: Vec<f32>,
    /// Bioturbation mixing rate (mm^2 / day).
    pub bioturbation_rate: Vec<f32>,
}

impl EarthwormPopulation {
    /// Create a new earthworm population on a `width x height` grid.
    ///
    /// Initial biomass is set proportional to the organic matter field,
    /// starting at ~10% carrying capacity.
    pub fn new(width: usize, height: usize, organic_matter: &[f32]) -> Self {
        let n = width * height;
        let mut biomass = vec![0.0f32; n];
        let mut density = vec![0.0f32; n];
        let bioturbation = vec![0.0f32; n];

        for i in 0..n {
            let om_signal = clamp(organic_matter.get(i).copied().unwrap_or(0.0), 0.0, 1.0);
            // Start at ~10% capacity scaled by local organic matter.
            let init_frac = 0.10 * om_signal.max(0.1);
            density[i] = EARTHWORM_CARRYING_CAPACITY * init_frac;
            biomass[i] = density[i] * EARTHWORM_INDIVIDUAL_MASS_G_C;
        }

        Self {
            biomass_per_voxel: biomass,
            population_density: density,
            bioturbation_rate: bioturbation,
        }
    }

    /// Create a zeroed population (no earthworms).
    pub fn empty(width: usize, height: usize) -> Self {
        let n = width * height;
        Self {
            biomass_per_voxel: vec![0.0; n],
            population_density: vec![0.0; n],
            bioturbation_rate: vec![0.0; n],
        }
    }
}

// ── Nematode guilds ─────────────────────────────────────────────────────

/// Nematode functional guild.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum NematodeKind {
    /// Bacterial feeders: graze on heterotrophic and nitrifying bacteria.
    BacterialFeeder,
    /// Fungal feeders: graze on saprotrophic fungi.
    FungalFeeder,
    /// Omnivores: mixed diet, lower trophic efficiency.
    Omnivore,
}

impl NematodeKind {
    /// Representative body plan for this guild.
    ///
    /// This keeps anatomical ownership with the soil-fauna model instead of
    /// repeating body templates in rendering code.
    pub fn body(self) -> NematodeBody {
        match self {
            Self::BacterialFeeder => NematodeBody::bacterial_feeder(),
            Self::FungalFeeder => NematodeBody::fungal_feeder(),
            Self::Omnivore => {
                let bacterial = NematodeBody::bacterial_feeder();
                let fungal = NematodeBody::fungal_feeder();
                NematodeBody {
                    length_mm: (bacterial.length_mm + fungal.length_mm) * 0.5,
                    diameter_um: (bacterial.diameter_um + fungal.diameter_um) * 0.5,
                    has_stylet: fungal.has_stylet,
                    stylet_length_um: fungal.stylet_length_um * 0.5,
                    cuticle_layers: bacterial.cuticle_layers.max(fungal.cuticle_layers),
                }
            }
        }
    }

    /// Feeding rate multiplier relative to the base rate.
    ///
    /// Bacterial feeders are the most active grazers; fungal feeders
    /// and omnivores feed at reduced rates (Yeates et al. 1993).
    pub fn feeding_rate_multiplier(self) -> f32 {
        match self {
            Self::BacterialFeeder => 1.0,
            Self::FungalFeeder => 0.6,
            Self::Omnivore => 0.4,
        }
    }

    /// Fraction of diet that targets heterotrophic bacteria.
    pub fn heterotroph_diet_fraction(self) -> f32 {
        match self {
            Self::BacterialFeeder => 0.80,
            Self::FungalFeeder => 0.10,
            Self::Omnivore => 0.40,
        }
    }

    /// Fraction of diet that targets nitrifying bacteria.
    pub fn nitrifier_diet_fraction(self) -> f32 {
        match self {
            Self::BacterialFeeder => 0.20,
            Self::FungalFeeder => 0.05,
            Self::Omnivore => 0.15,
        }
    }
}

/// A single nematode guild living on the same 2D grid.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NematodeGuild {
    pub kind: NematodeKind,
    /// Biomass per cell (ug C / cell).
    pub biomass_per_voxel: Vec<f32>,
    /// Population density (individuals / g soil).
    pub population_density: Vec<f32>,
}

impl NematodeGuild {
    /// Create a guild at ~5% carrying capacity.
    pub fn new(kind: NematodeKind, width: usize, height: usize) -> Self {
        let n = width * height;
        let init_frac = match kind {
            NematodeKind::BacterialFeeder => 0.05,
            NematodeKind::FungalFeeder => 0.03,
            NematodeKind::Omnivore => 0.02,
        };
        let density = NEMATODE_CARRYING_CAPACITY * init_frac;
        let biomass = density * NEMATODE_INDIVIDUAL_MASS_UG_C;
        Self {
            kind,
            biomass_per_voxel: vec![biomass; n],
            population_density: vec![density; n],
        }
    }

    /// Create an empty guild.
    pub fn empty(kind: NematodeKind, width: usize, height: usize) -> Self {
        let n = width * height;
        Self {
            kind,
            biomass_per_voxel: vec![0.0; n],
            population_density: vec![0.0; n],
        }
    }
}

// ── Step result ─────────────────────────────────────────────────────────

/// Diagnostic output from a soil fauna step.
#[derive(Debug, Clone, Default)]
pub struct SoilFaunaStepResult {
    /// Total NH4+ released by all fauna this step (summed across grid).
    pub total_nh4_released: f32,
    /// Total organic matter consumed by earthworms this step.
    pub total_om_consumed: f32,
    /// Total microbial biomass grazed by nematodes this step.
    pub total_microbial_grazed: f32,
    /// Mean bioturbation rate across grid (mm^2/day).
    pub mean_bioturbation_rate: f32,
}

// ── Integration function ────────────────────────────────────────────────

/// Advance the soil fauna by one timestep.
///
/// This couples earthworm and nematode dynamics to the microbial guilds
/// and substrate chemistry.  The 2D grid is `dims.0 x dims.1` with `dims.2`
/// providing the substrate depth (for 3D nutrient release).
///
/// # Arguments
///
/// * `earthworms` - Earthworm population state (mutable).
/// * `nematodes` - Slice of nematode guilds (mutable).
/// * `microbial_biomass` - Heterotroph biomass field, 2D (mutable, grazed).
/// * `nitrifier_biomass` - Nitrifier biomass field, 2D (mutable, grazed).
/// * `organic_matter` - Soil organic matter field, 2D (mutable, consumed by earthworms).
/// * `substrate` - 3D substrate lattice for nutrient (NH4+, P) release.
/// * `hydration` - Per-voxel hydration [0..1], 3D.
/// * `temperature` - Per-voxel soil temperature (C), 3D.
/// * `dt_hours` - Timestep in hours.
/// * `dims` - (width, height, depth) of the voxel grid.
pub fn step_soil_fauna(
    earthworms: &mut EarthwormPopulation,
    nematodes: &mut [NematodeGuild],
    microbial_biomass: &mut [f32],
    nitrifier_biomass: &mut [f32],
    organic_matter: &mut [f32],
    substrate: &mut BatchedAtomTerrarium,
    hydration: &[f32],
    temperature: &[f32],
    dt_hours: f32,
    dims: (usize, usize, usize),
) -> SoilFaunaStepResult {
    let (width, height, depth) = dims;
    let plane = width * height;
    let dt_days = dt_hours / 24.0;
    let total_voxels = substrate.total_voxels();

    let mut result = SoilFaunaStepResult::default();

    // ── Surface-layer environmental averages ────────────────────────
    // We average the top soil layer (z=0) for 2D fauna fields.
    // If the 3D grids are smaller than the 2D plane, we gracefully handle it.

    for i in 0..plane {
        // Compute environmental conditions from the surface voxel (z = 0).
        let voxel_3d = i; // z=0 layer maps directly to the 2D index
        let temp_c = temperature.get(voxel_3d).copied().unwrap_or(20.0);
        let h = hydration.get(voxel_3d).copied().unwrap_or(0.4);
        let om = organic_matter.get(i).copied().unwrap_or(0.0).max(0.0);
        let micro = microbial_biomass.get(i).copied().unwrap_or(0.0).max(0.0);
        let nitri = nitrifier_biomass.get(i).copied().unwrap_or(0.0).max(0.0);

        // ── Earthworm dynamics ──────────────────────────────────────
        let ew_temp_f = temperature_response(temp_c, EARTHWORM_T_OPT, EARTHWORM_T_SIGMA);
        let ew_hydra_f = hydration_response(
            h,
            EARTHWORM_HYDRATION_MIN,
            EARTHWORM_HYDRATION_OPT,
            EARTHWORM_HYDRATION_MAX,
        );
        let ew_env = ew_temp_f * ew_hydra_f;

        let ew_biomass = earthworms.biomass_per_voxel[i].max(0.0);
        let ew_density = earthworms.population_density[i].max(0.0);

        // Logistic growth: dN/dt = r * N * (1 - N/K) * env
        let ew_growth = EARTHWORM_R_MAX
            * ew_density
            * (1.0 - ew_density / EARTHWORM_CARRYING_CAPACITY).max(0.0)
            * ew_env
            * dt_days;

        // Feeding on organic matter: consumption proportional to biomass.
        let ew_om_demand = EARTHWORM_FEEDING_RATE * ew_biomass * ew_env * dt_days;
        let ew_om_consumed = ew_om_demand.min(om * 0.5); // never consume > 50% of local OM

        // Assimilation and respiration.
        let ew_c_assimilated = ew_om_consumed * EARTHWORM_ASSIMILATION_EFFICIENCY;
        let ew_maintenance_loss = EARTHWORM_MAINTENANCE * ew_biomass * dt_days;
        let ew_death_loss = EARTHWORM_DEATH_RATE * ew_biomass * ew_env.max(0.1) * dt_days;

        // Net biomass change.
        let ew_d_biomass = ew_c_assimilated - ew_maintenance_loss - ew_death_loss;
        let new_ew_biomass = (ew_biomass + ew_d_biomass).max(0.0);

        // Update population density from biomass.
        let new_ew_density = if EARTHWORM_INDIVIDUAL_MASS_G_C > 0.0 {
            (ew_density + ew_growth)
                .max(0.0)
                .min(EARTHWORM_CARRYING_CAPACITY * 1.2)
        } else {
            0.0
        };

        // Bioturbation rate scales with density.
        let density_fraction = new_ew_density / EARTHWORM_CARRYING_CAPACITY;
        let new_bioturbation = EARTHWORM_D_BIO_MAX * density_fraction.min(1.0) * ew_env;

        // Nutrient mineralization from earthworm activity.
        // NH4+ from excreted N (mucus + urine), sourced from ingested OM.
        let om_n_content = ew_om_consumed * 0.04; // assume 4% N in OM (C:N ~ 25:1)
        let ew_nh4_release = om_n_content * EARTHWORM_N_EXCRETION_FRACTION;
        let ew_p_release = ew_om_consumed * 0.003 * EARTHWORM_P_RELEASE_FRACTION; // ~0.3% P in OM

        // Dead biomass returns to organic matter pool.
        let ew_om_return = ew_death_loss;

        // Apply organic matter changes.
        if let Some(om_ref) = organic_matter.get_mut(i) {
            *om_ref = (*om_ref - ew_om_consumed + ew_om_return).max(0.0);
        }

        // Write updated earthworm state.
        earthworms.biomass_per_voxel[i] = new_ew_biomass;
        earthworms.population_density[i] = new_ew_density;
        earthworms.bioturbation_rate[i] = new_bioturbation;

        result.total_om_consumed += ew_om_consumed;

        // ── Nematode dynamics (per guild) ───────────────────────────
        let nem_temp_f = temperature_response(temp_c, NEMATODE_T_OPT, NEMATODE_T_SIGMA);
        let nem_hydra_f = hydration_response(
            h,
            NEMATODE_HYDRATION_MIN,
            NEMATODE_HYDRATION_OPT,
            NEMATODE_HYDRATION_MAX,
        );
        let nem_env = nem_temp_f * nem_hydra_f;

        let mut total_heterotroph_grazing = 0.0f32;
        let mut total_nitrifier_grazing = 0.0f32;
        let mut nem_nh4_release = 0.0f32;

        for guild in nematodes.iter_mut() {
            let nem_biomass = guild.biomass_per_voxel[i].max(0.0);
            let nem_density = guild.population_density[i].max(0.0);

            // Prey availability: combined microbial + nitrifier weighted by diet.
            let prey_hetero = micro * guild.kind.heterotroph_diet_fraction();
            let prey_nitri = nitri * guild.kind.nitrifier_diet_fraction();
            let prey_total = prey_hetero + prey_nitri;

            // Feeding: Lotka-Volterra functional response (Type I for simplicity).
            let feeding_multiplier = guild.kind.feeding_rate_multiplier();
            let feed_rate = NEMATODE_FEEDING_RATE * feeding_multiplier;

            // Consumption rate (ug C / day) limited by prey availability.
            let consumption_demand = feed_rate * nem_biomass * nem_env * dt_days;
            // Prey in ug C: microbial biomass is in g, nematode biomass in ug.
            // Convert microbial g -> ug for consistent units.
            let prey_ug = prey_total * 1.0e6;
            let consumption = consumption_demand.min(prey_ug * 0.3); // max 30% of local prey

            // Partition grazing between heterotrophs and nitrifiers.
            let prey_sum = prey_hetero + prey_nitri;
            let (hetero_frac, nitri_frac) = if prey_sum > 1e-12 {
                (prey_hetero / prey_sum, prey_nitri / prey_sum)
            } else {
                (0.5, 0.5)
            };
            // Convert consumption back to g for microbial fields.
            let consumption_g = consumption * 1.0e-6;
            let hetero_grazed = consumption_g * hetero_frac;
            let nitri_grazed = consumption_g * nitri_frac;
            total_heterotroph_grazing += hetero_grazed;
            total_nitrifier_grazing += nitri_grazed;

            // Growth: assimilated carbon.
            let assimilated = consumption * NEMATODE_ASSIMILATION_EFFICIENCY;
            let maintenance = 0.01 * nem_biomass * dt_days; // 1% d^-1 maintenance
            let death = NEMATODE_DEATH_RATE * nem_biomass * dt_days;

            let d_biomass = assimilated - maintenance - death;
            let new_nem_biomass = (nem_biomass + d_biomass).max(0.0);

            // Population: logistic growth modulated by prey availability.
            let prey_scale = if prey_ug > 1e-6 {
                (prey_ug / (prey_ug + nem_biomass * 2.0)).min(1.0)
            } else {
                0.0
            };
            let d_density = NEMATODE_R_MAX
                * nem_density
                * (1.0 - nem_density / NEMATODE_CARRYING_CAPACITY).max(0.0)
                * nem_env
                * prey_scale
                * dt_days;
            let new_density = (nem_density + d_density)
                .max(0.0)
                .min(NEMATODE_CARRYING_CAPACITY * 1.2);

            // N mineralization: excrete excess N from consumed microbial biomass.
            let n_excreted = consumption_g * NEMATODE_N_EXCRETION_PER_C;
            nem_nh4_release += n_excreted;

            guild.biomass_per_voxel[i] = new_nem_biomass;
            guild.population_density[i] = new_density;

            result.total_microbial_grazed += consumption_g;
        }

        // Apply grazing to microbial fields.
        if let Some(mb) = microbial_biomass.get_mut(i) {
            *mb = (*mb - total_heterotroph_grazing).max(0.0);
        }
        if let Some(nb) = nitrifier_biomass.get_mut(i) {
            *nb = (*nb - total_nitrifier_grazing).max(0.0);
        }

        // ── Nutrient release into substrate ─────────────────────────
        // Deposit NH4+ and P into the top soil layer (z=0) of the 3D substrate.
        let total_nh4 = ew_nh4_release + nem_nh4_release;
        result.total_nh4_released += total_nh4;

        // Deposit into substrate: spread across depth layers weighted toward surface.
        for z in 0..depth {
            let depth_weight = 1.0 / (1.0 + z as f32); // surface-weighted
            let voxel_idx = z * plane + i;

            // NH4+ release.
            if total_nh4 > 0.0 {
                let nh4_base = TerrariumSpecies::Ammonium as usize * total_voxels;
                if nh4_base + voxel_idx < substrate.current.len() {
                    substrate.current[nh4_base + voxel_idx] += total_nh4 * depth_weight * 0.001;
                    // scale to substrate concentration units
                }
            }

            // P release from earthworms.
            if ew_p_release > 0.0 {
                let p_base = TerrariumSpecies::Phosphorus as usize * total_voxels;
                if p_base + voxel_idx < substrate.current.len() {
                    substrate.current[p_base + voxel_idx] += ew_p_release * depth_weight * 0.001;
                }
            }
        }
    }

    // ── Bioturbation: diffusive vertical mixing ─────────────────────
    // Apply a single diffusive mixing step to homogenize substrate concentrations
    // in the vertical (z) direction.  This represents the physical churning of
    // soil by earthworm burrowing.
    //
    // For each (x,y) column, we relax each species toward the column mean
    // proportionally to the local bioturbation coefficient.
    if depth > 1 {
        for i in 0..plane {
            let d_bio = earthworms.bioturbation_rate[i];
            if d_bio < 1e-8 {
                continue;
            }
            // Mixing fraction for this timestep.
            // d_bio is in mm^2/day, dt in hours.  Diffusive mixing fraction ~
            // D * dt / dz^2.  With dz = voxel_size_mm:
            let dz = substrate.voxel_size_mm;
            let mix_frac = clamp(d_bio * (dt_hours / 24.0) / (dz * dz), 0.0, 0.3);

            for species_idx in 0..TERRARIUM_SPECIES_COUNT {
                let base = species_idx * total_voxels;

                // Compute column mean.
                let mut col_sum = 0.0f32;
                for z in 0..depth {
                    let idx = base + z * plane + i;
                    if idx < substrate.current.len() {
                        col_sum += substrate.current[idx];
                    }
                }
                let col_mean = col_sum / depth as f32;

                // Relax toward mean.
                for z in 0..depth {
                    let idx = base + z * plane + i;
                    if idx < substrate.current.len() {
                        let val = substrate.current[idx];
                        substrate.current[idx] = val + mix_frac * (col_mean - val);
                    }
                }
            }
        }
    }

    // Compute mean bioturbation rate.
    if plane > 0 {
        result.mean_bioturbation_rate =
            earthworms.bioturbation_rate.iter().sum::<f32>() / plane as f32;
    }

    result
}

// ── Default fauna initializer ───────────────────────────────────────────

/// Create a default set of nematode guilds for a `width x height` grid.
pub fn default_nematode_guilds(width: usize, height: usize) -> Vec<NematodeGuild> {
    vec![
        NematodeGuild::new(NematodeKind::BacterialFeeder, width, height),
        NematodeGuild::new(NematodeKind::FungalFeeder, width, height),
        NematodeGuild::new(NematodeKind::Omnivore, width, height),
    ]
}

// ═════════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrarium::BatchedAtomTerrarium;

    const W: usize = 4;
    const H: usize = 4;
    const D: usize = 4;
    const PLANE: usize = W * H;

    /// Helper: build a minimal substrate and return standard fields.
    fn setup() -> (
        EarthwormPopulation,
        Vec<NematodeGuild>,
        Vec<f32>, // microbial_biomass
        Vec<f32>, // nitrifier_biomass
        Vec<f32>, // organic_matter
        BatchedAtomTerrarium,
        Vec<f32>, // hydration (3D)
        Vec<f32>, // temperature (3D)
    ) {
        let substrate = BatchedAtomTerrarium::new(W, H, D, 0.5, false);
        let total_3d = W * H * D;

        let organic_matter = vec![0.05f32; PLANE];
        let earthworms = EarthwormPopulation::new(W, H, &organic_matter);
        let nematodes = default_nematode_guilds(W, H);
        let microbial_biomass = vec![0.02f32; PLANE];
        let nitrifier_biomass = vec![0.005f32; PLANE];
        let hydration = vec![0.5f32; total_3d];
        let temperature = vec![15.0f32; total_3d];

        (
            earthworms,
            nematodes,
            microbial_biomass,
            nitrifier_biomass,
            organic_matter,
            substrate,
            hydration,
            temperature,
        )
    }

    #[test]
    fn earthworm_population_grows_with_organic_matter() {
        let (mut ew, mut nem, mut micro, mut nitri, mut om, mut sub, hyd, temp) = setup();

        // Give earthworms plenty of organic matter and ideal conditions.
        for v in om.iter_mut() {
            *v = 0.20;
        }
        let initial_density: f32 = ew.population_density.iter().sum();
        assert!(initial_density > 0.0, "should start with some worms");

        // Run 100 steps of 1 hour each (~4 days).
        for _ in 0..100 {
            step_soil_fauna(
                &mut ew,
                &mut nem,
                &mut micro,
                &mut nitri,
                &mut om,
                &mut sub,
                &hyd,
                &temp,
                1.0,
                (W, H, D),
            );
        }

        let final_density: f32 = ew.population_density.iter().sum();
        assert!(
            final_density > initial_density,
            "earthworm density should increase: {initial_density} -> {final_density}"
        );
    }

    #[test]
    fn nematode_grazing_reduces_microbial_biomass() {
        let (mut ew, mut nem, mut micro, mut nitri, mut om, mut sub, hyd, temp) = setup();

        // Start with high microbial biomass and nematodes.
        for v in micro.iter_mut() {
            *v = 0.10; // high microbial biomass
        }
        // Boost nematode populations.
        for guild in nem.iter_mut() {
            for v in guild.biomass_per_voxel.iter_mut() {
                *v = 0.5; // ug C per cell
            }
            for v in guild.population_density.iter_mut() {
                *v = 20.0;
            }
        }

        let initial_micro: f32 = micro.iter().sum();
        assert!(initial_micro > 0.0);

        // Run several steps.
        for _ in 0..50 {
            step_soil_fauna(
                &mut ew,
                &mut nem,
                &mut micro,
                &mut nitri,
                &mut om,
                &mut sub,
                &hyd,
                &temp,
                1.0,
                (W, H, D),
            );
        }

        let final_micro: f32 = micro.iter().sum();
        assert!(
            final_micro < initial_micro,
            "microbial biomass should decrease from nematode grazing: {initial_micro} -> {final_micro}"
        );
    }

    #[test]
    fn bioturbation_mixes_soil_layers() {
        let (mut ew, mut nem, mut micro, mut nitri, mut om, mut sub, hyd, temp) = setup();

        // Set up a strong vertical gradient: high carbon at z=0, zero elsewhere.
        let total_voxels = sub.total_voxels();
        let c_base = TerrariumSpecies::Carbon as usize * total_voxels;
        for i in 0..PLANE {
            sub.current[c_base + i] = 1.0; // z=0 layer
            for z in 1..D {
                sub.current[c_base + z * PLANE + i] = 0.0;
            }
        }

        // Give earthworms high density so bioturbation is strong.
        for v in ew.population_density.iter_mut() {
            *v = EARTHWORM_CARRYING_CAPACITY * 0.8;
        }
        for v in ew.biomass_per_voxel.iter_mut() {
            *v = EARTHWORM_CARRYING_CAPACITY * 0.8 * EARTHWORM_INDIVIDUAL_MASS_G_C;
        }

        // Measure initial heterogeneity: variance across depths.
        let initial_variance = column_variance(&sub.current, c_base, PLANE, D, 0);

        // Run 200 steps (several days).
        for _ in 0..200 {
            step_soil_fauna(
                &mut ew,
                &mut nem,
                &mut micro,
                &mut nitri,
                &mut om,
                &mut sub,
                &hyd,
                &temp,
                1.0,
                (W, H, D),
            );
        }

        let final_variance = column_variance(&sub.current, c_base, PLANE, D, 0);
        assert!(
            final_variance < initial_variance,
            "bioturbation should reduce vertical heterogeneity: {initial_variance} -> {final_variance}"
        );
    }

    /// Compute variance of a single column (x=0, y=0) across depth.
    fn column_variance(data: &[f32], base: usize, plane: usize, depth: usize, col: usize) -> f32 {
        let values: Vec<f32> = (0..depth).map(|z| data[base + z * plane + col]).collect();
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / values.len() as f32
    }

    #[test]
    fn fauna_respects_waterlogging() {
        let (mut ew, mut nem, mut micro, mut nitri, mut om, mut sub, _, temp) = setup();

        // Waterlogged soil: hydration = 0.95 (above max threshold).
        let total_3d = W * H * D;
        let hyd_saturated = vec![0.95f32; total_3d];

        // Set initial populations.
        for v in ew.population_density.iter_mut() {
            *v = 100.0;
        }
        for v in ew.biomass_per_voxel.iter_mut() {
            *v = 100.0 * EARTHWORM_INDIVIDUAL_MASS_G_C;
        }
        for guild in nem.iter_mut() {
            for v in guild.population_density.iter_mut() {
                *v = 20.0;
            }
            for v in guild.biomass_per_voxel.iter_mut() {
                *v = 20.0 * NEMATODE_INDIVIDUAL_MASS_UG_C;
            }
        }

        let initial_ew_biomass: f32 = ew.biomass_per_voxel.iter().sum();
        let initial_nem_biomass: f32 = nem
            .iter()
            .map(|g| g.biomass_per_voxel.iter().sum::<f32>())
            .sum();

        // Run for many steps in waterlogged conditions.
        for _ in 0..200 {
            step_soil_fauna(
                &mut ew,
                &mut nem,
                &mut micro,
                &mut nitri,
                &mut om,
                &mut sub,
                &hyd_saturated,
                &temp,
                1.0,
                (W, H, D),
            );
        }

        let final_ew_biomass: f32 = ew.biomass_per_voxel.iter().sum();
        let final_nem_biomass: f32 = nem
            .iter()
            .map(|g| g.biomass_per_voxel.iter().sum::<f32>())
            .sum();

        assert!(
            final_ew_biomass < initial_ew_biomass,
            "earthworms should decline in waterlogged soil: {initial_ew_biomass} -> {final_ew_biomass}"
        );
        assert!(
            final_nem_biomass < initial_nem_biomass,
            "nematodes should decline in waterlogged soil: {initial_nem_biomass} -> {final_nem_biomass}"
        );
    }

    #[test]
    fn nutrient_mineralization_from_fauna() {
        let (mut ew, mut nem, mut micro, mut nitri, mut om, mut sub, hyd, temp) = setup();

        // Give all fauna decent populations and food.
        for v in om.iter_mut() {
            *v = 0.15;
        }
        for v in micro.iter_mut() {
            *v = 0.08;
        }
        for v in ew.population_density.iter_mut() {
            *v = 150.0;
        }
        for v in ew.biomass_per_voxel.iter_mut() {
            *v = 150.0 * EARTHWORM_INDIVIDUAL_MASS_G_C;
        }
        for guild in nem.iter_mut() {
            for v in guild.biomass_per_voxel.iter_mut() {
                *v = 1.0;
            }
            for v in guild.population_density.iter_mut() {
                *v = 30.0;
            }
        }

        // Record initial NH4+ in substrate.
        let total_voxels = sub.total_voxels();
        let nh4_base = TerrariumSpecies::Ammonium as usize * total_voxels;
        let initial_nh4: f32 = sub.current[nh4_base..nh4_base + total_voxels].iter().sum();

        let mut total_released = 0.0f32;
        for _ in 0..50 {
            let res = step_soil_fauna(
                &mut ew,
                &mut nem,
                &mut micro,
                &mut nitri,
                &mut om,
                &mut sub,
                &hyd,
                &temp,
                1.0,
                (W, H, D),
            );
            total_released += res.total_nh4_released;
        }

        let final_nh4: f32 = sub.current[nh4_base..nh4_base + total_voxels].iter().sum();

        assert!(
            total_released > 0.0,
            "fauna should release NH4+: total_released = {total_released}"
        );
        assert!(
            final_nh4 > initial_nh4,
            "substrate NH4+ should increase: {initial_nh4} -> {final_nh4}"
        );
    }

    #[test]
    fn temperature_response_function() {
        // At optimum, response should be ~1.0.
        let at_opt = temperature_response(EARTHWORM_T_OPT, EARTHWORM_T_OPT, EARTHWORM_T_SIGMA);
        assert!((at_opt - 1.0).abs() < 1e-6);

        // At 2 sigma away, response should be ~exp(-2) = 0.135.
        let two_sigma = temperature_response(
            EARTHWORM_T_OPT + 2.0 * EARTHWORM_T_SIGMA,
            EARTHWORM_T_OPT,
            EARTHWORM_T_SIGMA,
        );
        assert!((two_sigma - (-2.0f32).exp()).abs() < 1e-4);

        // At very cold temperature, response should be very small.
        let cold = temperature_response(-10.0, EARTHWORM_T_OPT, EARTHWORM_T_SIGMA);
        assert!(cold < 0.05);
    }

    #[test]
    fn hydration_response_function() {
        // Below minimum: zero.
        assert_eq!(
            hydration_response(
                0.05,
                EARTHWORM_HYDRATION_MIN,
                EARTHWORM_HYDRATION_OPT,
                EARTHWORM_HYDRATION_MAX
            ),
            0.0
        );
        // At optimum: 1.0.
        let at_opt = hydration_response(
            EARTHWORM_HYDRATION_OPT,
            EARTHWORM_HYDRATION_MIN,
            EARTHWORM_HYDRATION_OPT,
            EARTHWORM_HYDRATION_MAX,
        );
        assert!((at_opt - 1.0).abs() < 1e-6);
        // Above maximum: zero.
        assert_eq!(
            hydration_response(
                0.95,
                EARTHWORM_HYDRATION_MIN,
                EARTHWORM_HYDRATION_OPT,
                EARTHWORM_HYDRATION_MAX
            ),
            0.0
        );
    }

    #[test]
    fn empty_populations_are_stable() {
        let mut ew = EarthwormPopulation::empty(W, H);
        let mut nem = vec![NematodeGuild::empty(NematodeKind::BacterialFeeder, W, H)];
        let mut micro = vec![0.02f32; PLANE];
        let mut nitri = vec![0.005f32; PLANE];
        let mut om = vec![0.05f32; PLANE];
        let mut sub = BatchedAtomTerrarium::new(W, H, D, 0.5, false);
        let hyd = vec![0.5f32; W * H * D];
        let temp = vec![15.0f32; W * H * D];

        // Should not panic or produce NaN.
        let res = step_soil_fauna(
            &mut ew,
            &mut nem,
            &mut micro,
            &mut nitri,
            &mut om,
            &mut sub,
            &hyd,
            &temp,
            1.0,
            (W, H, D),
        );

        assert!(res.total_nh4_released >= 0.0);
        assert!(!res.total_nh4_released.is_nan());
        assert!(!res.mean_bioturbation_rate.is_nan());

        // Empty populations should remain at zero.
        assert_eq!(ew.biomass_per_voxel.iter().sum::<f32>(), 0.0);
    }

    // ========================================================================
    // Molecular/DNA Fidelity Tests
    // ========================================================================

    #[test]
    fn test_earthworm_genome_stats() {
        use super::earthworm_genome_stats::*;
        assert!((GENOME_MB - 1060.0).abs() < 50.0, "Genome ~1.06Gb");
        assert!(PROTEIN_GENES >= 30_000, "At least 30,000 genes");
        assert_eq!(CHROMOSOME_COUNT, 36, "36 chromosomes");
    }

    #[test]
    fn test_earthworm_key_genes() {
        let genes = super::earthworm_key_genes();
        assert!(genes.len() >= 15, "At least 15 key genes defined");
    }

    #[test]
    fn test_earthworm_digestive_enzymes() {
        let genes = super::earthworm_key_genes();
        let digestive: Vec<_> = genes
            .iter()
            .filter(|g| g.category == super::EarthwormGeneFunction::DigestiveEnzyme)
            .collect();
        assert!(digestive.len() >= 3, "At least 3 digestive enzymes");
        let names: Vec<_> = digestive.iter().map(|g| g.name.as_str()).collect();
        assert!(names.iter().any(|n| n.contains("Cel")), "Cellulase present");
        assert!(names.iter().any(|n| n.contains("Chi")), "Chitinase present");
    }

    #[test]
    fn test_earthworm_hemoglobin_genes() {
        let genes = super::earthworm_key_genes();
        let heme: Vec<_> = genes
            .iter()
            .filter(|g| g.category == super::EarthwormGeneFunction::Hemoglobin)
            .collect();
        assert!(heme.len() >= 2, "At least 2 hemoglobin chains");
    }

    #[test]
    fn test_earthworm_metal_detox() {
        let genes = super::earthworm_key_genes();
        let detox: Vec<_> = genes
            .iter()
            .filter(|g| g.category == super::EarthwormGeneFunction::Chloragocyte)
            .collect();
        assert!(detox.len() >= 2, "At least 2 chloragocyte genes");
    }

    #[test]
    fn test_earthworm_body_anatomy() {
        let body = super::EarthwormBody::default();
        assert!(body.segments >= 100, "At least 100 segments");
        assert!(body.length_cm > 0.0, "Positive length");
        assert!(body.gut_divisions.len() >= 5, "At least 5 gut divisions");
    }

    #[test]
    fn test_earthworm_pathways() {
        let pathways = super::earthworm_pathways();
        assert!(pathways.len() >= 5, "At least 5 molecular pathways");
    }

    #[test]
    fn test_nematode_genome_stats() {
        use super::nematode_genome_stats::*;
        assert!((GENOME_MB - 100.0).abs() < 10.0, "Genome ~100Mb");
        assert!(PROTEIN_GENES >= 20_000, "At least 20,000 genes");
        assert_eq!(CHROMOSOME_COUNT, 6, "6 chromosomes");
        assert_eq!(NEURON_COUNT, 302, "302 neurons");
    }

    #[test]
    fn test_nematode_body_types() {
        let bf = super::NematodeBody::bacterial_feeder();
        assert!(!bf.has_stylet, "Bacterial feeders lack stylet");
        assert!(bf.length_mm > 0.0, "Positive length");

        let ff = super::NematodeBody::fungal_feeder();
        assert!(ff.has_stylet, "Fungal feeders have stylet");

        let pp = super::NematodeBody::plant_parasite();
        assert!(pp.has_stylet, "Plant parasites have stylet");
        assert!(
            pp.stylet_length_um > ff.stylet_length_um,
            "Plant parasites have longer stylet"
        );
    }

    #[test]
    fn test_gene_expression_levels() {
        let genes = super::earthworm_key_genes();
        for gene in &genes {
            assert!(
                gene.expression_level >= 0.0 && gene.expression_level <= 1.0,
                "Expression level must be [0,1]"
            );
        }
    }
}
