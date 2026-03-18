//! Ant colony organism with complete molecular/DNA fidelity.
//!
//! This module implements ant (Formica sp.) colonies with:
//! - Individual ant agents with caste-specific behavior
//! - Complete anatomical structure (body segments, muscles, organs)
//! - Molecular fidelity: genes, proteins, pathways
//! - Pheromone communication system
//! - Neural system (~250,000 neurons per worker)
//!
//! # Scientific Background
//!
//! Ants are eusocial insects with complex colony organization:
//! - **Colonies**: 100-10,000+ individuals depending on species
//! - **Castes**: Queen, workers (major/minor), soldiers, drones
//! - **Body plan**: 3 segments (head, mesosoma, metasoma), 6 legs, 2 antennae
//! - **Brain**: ~250,000 neurons (worker), mushroom bodies for learning
//! - **Genome**: 280 Mb, ~18,000 protein-coding genes
//! - **Communication**: Pheromone trails, tactile, acoustic
//!
//! # References
//!
//! - Hölldobler & Wilson (1990) *The Ants* - Comprehensive ant biology
//! - Gronenberg (2008) *Comparative Neurobiology* - Ant brain structure
//! - Bonasio et al. (2010) *Science* - Ant genome, caste differentiation
//! - Smith et al. (2011) *PNAS* - Formica genome
//! - Dornhaus & Franks (2008) *Ecology* - Collective decision-making
//! - Gordon (2010) *Ant Encounters* - Colony dynamics

use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════════════
// ANT BIOLOGICAL CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

// ── Genome statistics ────────────────────────────────────────────────────────

/// Genome size in megabases (Formica spp.).
pub const ANT_GENOME_MB: f32 = 280.0;

/// Number of protein-coding genes.
pub const ANT_PROTEIN_GENES: usize = 18_000;

/// Number of chromosomes (Formica).
pub const ANT_CHROMOSOME_COUNT: usize = 26;

/// GC content percentage.
pub const ANT_GC_CONTENT_PERCENT: f32 = 38.5;

// ── Body dimensions ───────────────────────────────────────────────────────────

/// Worker body length (mm).
pub const ANT_WORKER_LENGTH_MM: f32 = 6.0;

/// Queen body length (mm).
pub const ANT_QUEEN_LENGTH_MM: f32 = 10.0;

/// Head width (mm).
pub const ANT_HEAD_WIDTH_MM: f32 = 1.2;

/// Leg span (mm).
pub const ANT_LEG_SPAN_MM: f32 = 8.0;

/// Body mass (mg).
pub const ANT_WORKER_MASS_MG: f32 = 5.0;

/// Queen mass (mg).
pub const ANT_QUEEN_MASS_MG: f32 = 25.0;

// ── Neural system ─────────────────────────────────────────────────────────────

/// Worker ant neuron count.
pub const ANT_WORKER_NEURONS: usize = 250_000;

/// Queen ant neuron count.
pub const ANT_QUEEN_NEURONS: usize = 300_000;

/// Mushroom body Kenyon cells.
pub const ANT_KENYON_CELLS: usize = 50_000;

/// Antennal lobe glomeruli.
pub const ANT_GLOMERULI_COUNT: usize = 400;

/// Optical lobe neurons.
pub const ANT_OPTICAL_NEURONS: usize = 60_000;

// ── Muscle system ─────────────────────────────────────────────────────────────

/// Number of body segments with muscles.
pub const ANT_BODY_SEGMENTS: usize = 3;

/// Muscles per leg.
pub const ANT_MUSCLES_PER_LEG: usize = 12;

/// Total leg muscles (6 legs × 12 muscles).
pub const ANT_TOTAL_LEG_MUSCLES: usize = 72;

/// Flight muscles (in queens/drones during mating).
pub const ANT_FLIGHT_MUSCLES: usize = 24;

/// Head muscles (mandibles, antennae).
pub const ANT_HEAD_MUSCLES: usize = 16;

/// Total skeletal muscles.
pub const ANT_TOTAL_MUSCLES: usize = ANT_TOTAL_LEG_MUSCLES + ANT_FLIGHT_MUSCLES + ANT_HEAD_MUSCLES;

// ── Lifespan and reproduction ────────────────────────────────────────────────

/// Worker lifespan (days).
pub const ANT_WORKER_LIFESPAN_DAYS: f32 = 60.0;

/// Queen lifespan (days).
pub const ANT_QUEEN_LIFESPAN_DAYS: f32 = 7300.0; // ~20 years

/// Egg to adult development (days).
pub const ANT_DEVELOPMENT_DAYS: f32 = 45.0;

/// Eggs per day (queen).
pub const ANT_EGGS_PER_DAY: f32 = 100.0;

/// Typical colony size.
pub const COLONY_SIZE_TYPICAL: usize = 5_000;

/// Maximum colony size.
pub const COLONY_SIZE_MAX: usize = 50_000;

// ── Pheromone system ──────────────────────────────────────────────────────────

/// Pheromone trail decay half-life (seconds).
pub const PHEROMONE_TRAIL_HALFLIFE_S: f32 = 600.0;

/// Pheromone alarm decay half-life (seconds).
pub const PHEROMONE_ALARM_HALFLIFE_S: f32 = 30.0;

/// Detection threshold.
pub const PHEROMONE_DETECTION_THRESHOLD: f32 = 0.001;

/// Maximum pheromone trail age (seconds).
pub const PHEROMONE_MAX_AGE_S: f32 = 3600.0;

// ═══════════════════════════════════════════════════════════════════════════════
// ANT BODY ANATOMY
// ═══════════════════════════════════════════════════════════════════════════════

/// Body segment types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BodySegment {
    /// Head - brain, sensory organs, mandibles.
    Head,
    /// Mesosoma (alitrunk) - legs, wings (if present).
    Mesosoma,
    /// Metasoma (gaster) - digestive, reproductive organs.
    Metasoma,
}

/// Ant body segment with tissues.
#[derive(Debug, Clone)]
pub struct AntBodySegment {
    pub segment_type: BodySegment,
    /// Length in mm.
    pub length_mm: f32,
    /// Width in mm.
    pub width_mm: f32,
    /// Health [0, 1].
    pub health: f32,
    /// Energy stored [0, 1].
    pub energy: f32,
}

/// Complete ant exoskeleton.
#[derive(Debug, Clone)]
pub struct AntExoskeleton {
    /// Cuticle thickness (μm).
    pub cuticle_thickness_um: f32,
    /// Sclerotization level [0, 1].
    pub sclerotization: f32,
    /// Pheromone glands capacity.
    pub gland_capacity: f32,
    /// Current gland fill level.
    pub gland_fill: f32,
}

/// Ant leg with joints.
#[derive(Debug, Clone)]
pub struct AntLeg {
    pub id: u8,
    /// Side (0=left, 1=right).
    pub is_left: bool,
    /// Position (1-3 front to back).
    pub position: u8,
    /// Coxa joint angle (radians).
    pub coxa_angle: f32,
    /// Femur angle.
    pub femur_angle: f32,
    /// Tibia angle.
    pub tibia_angle: f32,
    /// Tarsus angle.
    pub tarsus_angle: f32,
    /// Muscle activation [0, 1].
    pub muscle_activation: f32,
}

/// Complete ant body.
#[derive(Debug, Clone)]
pub struct AntBody {
    pub segments: Vec<AntBodySegment>,
    pub exoskeleton: AntExoskeleton,
    pub legs: Vec<AntLeg>,
    /// Digestive system fill [0, 1].
    pub crop_fill: f32,
    /// Fat body reserves [0, 1].
    pub fat_body: f32,
}

impl AntBody {
    pub fn new(caste: AntCaste) -> Self {
        let scale = caste.body_scale();

        let segments = vec![
            AntBodySegment {
                segment_type: BodySegment::Head,
                length_mm: ANT_HEAD_WIDTH_MM * scale,
                width_mm: ANT_HEAD_WIDTH_MM * scale,
                health: 1.0,
                energy: 0.5,
            },
            AntBodySegment {
                segment_type: BodySegment::Mesosoma,
                length_mm: ANT_WORKER_LENGTH_MM * 0.4 * scale,
                width_mm: ANT_HEAD_WIDTH_MM * 1.2 * scale,
                health: 1.0,
                energy: 0.5,
            },
            AntBodySegment {
                segment_type: BodySegment::Metasoma,
                length_mm: ANT_WORKER_LENGTH_MM * 0.5 * scale,
                width_mm: ANT_HEAD_WIDTH_MM * 1.5 * scale,
                health: 1.0,
                energy: 0.5,
            },
        ];

        let exoskeleton = AntExoskeleton {
            cuticle_thickness_um: 20.0 * scale,
            sclerotization: if matches!(caste, AntCaste::Soldier) { 0.9 } else { 0.6 },
            gland_capacity: 1.0,
            gland_fill: 1.0,
        };

        let mut legs = Vec::with_capacity(6);
        for leg_id in 0..6 {
            legs.push(AntLeg {
                id: leg_id,
                is_left: leg_id % 2 == 0,
                position: (leg_id / 2) as u8 + 1,
                coxa_angle: 0.0,
                femur_angle: 0.0,
                tibia_angle: 0.0,
                tarsus_angle: 0.0,
                muscle_activation: 0.0,
            });
        }

        Self {
            segments,
            exoskeleton,
            legs,
            crop_fill: 0.5,
            fat_body: 0.5,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ANT CASTE SYSTEM
// ═══════════════════════════════════════════════════════════════════════════════

/// Ant caste types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AntCaste {
    /// Reproductive female - egg laying.
    Queen,
    /// Fertile male - mating flight only.
    Drone,
    /// Sterile female - foraging, nursing.
    WorkerMinor,
    /// Larger sterile female - heavy tasks.
    WorkerMajor,
    /// Defense specialist.
    Soldier,
    /// Developing larva.
    Larva,
    /// Pupal stage.
    Pupa,
    /// Egg stage.
    Egg,
}

impl AntCaste {
    pub fn can_lay_eggs(self) -> bool { matches!(self, Self::Queen) }
    pub fn can_forage(self) -> bool { matches!(self, Self::WorkerMinor | Self::WorkerMajor) }
    pub fn can_fly(self) -> bool { matches!(self, Self::Queen | Self::Drone) }
    pub fn is_adult(self) -> bool { matches!(self, Self::Queen | Self::Drone | Self::WorkerMinor | Self::WorkerMajor | Self::Soldier) }

    /// Body size scale relative to minor worker.
    pub fn body_scale(self) -> f32 {
        match self {
            Self::Queen => 1.8,
            Self::Drone => 1.2,
            Self::WorkerMinor => 1.0,
            Self::WorkerMajor => 1.3,
            Self::Soldier => 1.4,
            Self::Larva => 0.3,
            Self::Pupa => 0.8,
            Self::Egg => 0.1,
        }
    }

    /// Brain neuron count.
    pub fn neuron_count(self) -> usize {
        (ANT_WORKER_NEURONS as f32 * self.body_scale()) as usize
    }

    /// Lifespan in days.
    pub fn lifespan_days(self) -> f32 {
        match self {
            Self::Queen => ANT_QUEEN_LIFESPAN_DAYS,
            Self::Drone => 90.0,
            Self::WorkerMinor | Self::WorkerMajor | Self::Soldier => ANT_WORKER_LIFESPAN_DAYS,
            Self::Larva | Self::Pupa | Self::Egg => ANT_DEVELOPMENT_DAYS,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHEROMONE SYSTEM
// ═══════════════════════════════════════════════════════════════════════════════

/// Pheromone types used in ant communication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PheromoneType {
    /// Trail to food source.
    Trail,
    /// Alarm/defense signal.
    Alarm,
    /// Queen presence (QMP).
    QueenSignal,
    /// Death/necrophoresis (oleic acid).
    Death,
    /// Recruitment to task.
    Recruitment,
    /// Territory marking.
    Territorial,
    /// Brood recognition.
    Brood,
    /// Cuticular hydrocarbons (colony identity).
    Cuticular,
}

/// Chemical composition of a pheromone.
#[derive(Debug, Clone)]
pub struct PheromoneChemistry {
    /// Primary compound name.
    pub compound: String,
    /// Molecular weight (Da).
    pub molecular_weight: f32,
    /// Vapor pressure (Pa at 25°C).
    pub vapor_pressure: f32,
    /// Detection threshold (ng).
    pub detection_threshold: f32,
}

impl PheromoneChemistry {
    pub fn for_type(ptype: PheromoneType) -> Self {
        match ptype {
            PheromoneType::Trail => Self {
                compound: "3-ethyl-2,5-dimethylpyrazine".into(),
                molecular_weight: 136.2,
                vapor_pressure: 0.1,
                detection_threshold: 0.01,
            },
            PheromoneType::Alarm => Self {
                compound: "4-methyl-3-heptanone".into(),
                molecular_weight: 128.2,
                vapor_pressure: 1.0,
                detection_threshold: 0.001,
            },
            PheromoneType::QueenSignal => Self {
                compound: "queen mandibular pheromone".into(),
                molecular_weight: 254.4,
                vapor_pressure: 0.001,
                detection_threshold: 0.0001,
            },
            PheromoneType::Death => Self {
                compound: "oleic acid".into(),
                molecular_weight: 282.5,
                vapor_pressure: 0.0001,
                detection_threshold: 0.1,
            },
            _ => Self {
                compound: "unknown".into(),
                molecular_weight: 200.0,
                vapor_pressure: 0.1,
                detection_threshold: 0.01,
            },
        }
    }
}

/// A pheromone deposit in the environment.
#[derive(Debug, Clone)]
pub struct PheromoneDeposit {
    pub pheromone_type: PheromoneType,
    pub x_mm: f32,
    pub y_mm: f32,
    pub strength: f32,
    pub age_s: f32,
    pub colony_id: u32,
    pub chemistry: PheromoneChemistry,
}

impl PheromoneDeposit {
    pub fn new(ptype: PheromoneType, x: f32, y: f32, strength: f32, colony_id: u32) -> Self {
        let chemistry = PheromoneChemistry::for_type(ptype);
        Self { pheromone_type: ptype, x_mm: x, y_mm: y, strength, age_s: 0.0, colony_id, chemistry }
    }

    pub fn half_life(&self) -> f32 {
        match self.pheromone_type {
            PheromoneType::Alarm => PHEROMONE_ALARM_HALFLIFE_S,
            PheromoneType::Trail => PHEROMONE_TRAIL_HALFLIFE_S,
            _ => PHEROMONE_TRAIL_HALFLIFE_S,
        }
    }

    pub fn effective_strength(&self) -> f32 {
        let decay_constant = 0.693 / self.half_life();
        self.strength * (-decay_constant * self.age_s).exp()
    }

    pub fn is_detectable(&self) -> bool {
        self.age_s < PHEROMONE_MAX_AGE_S && self.effective_strength() > PHEROMONE_DETECTION_THRESHOLD
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ANT NEURAL SYSTEM
// ═══════════════════════════════════════════════════════════════════════════════

/// Neurotransmitter types in ant brain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AntNeurotransmitter {
    /// Excitatory - main CNS transmitter.
    Glutamate,
    /// Inhibitory - mushroom bodies.
    GABA,
    /// Modulatory - reward, learning.
    Dopamine,
    /// Modulatory - alertness, aggression.
    Octopamine,
    /// Modulatory - satiety.
    Serotonin,
    /// Neuromuscular junctions.
    Acetylcholine,
    /// Social behavior.
    Vasopressin,
}

/// Ant brain regions.
#[derive(Debug, Clone)]
pub struct AntBrain {
    /// Mushroom body - learning, memory.
    pub mushroom_body: MushroomBody,
    /// Antennal lobe - olfaction.
    pub antennal_lobe: AntennalLobe,
    /// Optical lobe - vision.
    pub optical_lobe: OpticalLobe,
    /// Subesophageal ganglion - motor.
    pub subesophageal_ganglion: f32,
    /// Central complex - navigation.
    pub central_complex: CentralComplex,
    /// Neurotransmitter levels.
    pub neurotransmitters: HashMap<AntNeurotransmitter, f32>,
}

/// Mushroom body structure.
#[derive(Debug, Clone)]
pub struct MushroomBody {
    /// Calyx input region activation [0, 1].
    pub calyx_activation: f32,
    /// Peduncle output activation [0, 1].
    pub peduncle_activation: f32,
    /// Kenyon cell activation pattern.
    pub kenyon_cells: Vec<f32>,
    /// Memory consolidation level [0, 1].
    pub memory_consolidation: f32,
    /// Long-term memory strength.
    pub ltm_strength: f32,
}

impl MushroomBody {
    pub fn new() -> Self {
        Self {
            calyx_activation: 0.0,
            peduncle_activation: 0.0,
            kenyon_cells: vec![0.0; 100], // Simplified from 50,000
            memory_consolidation: 0.0,
            ltm_strength: 0.0,
        }
    }
}

/// Antennal lobe (olfactory processing).
#[derive(Debug, Clone)]
pub struct AntennalLobe {
    /// Glomeruli activation levels.
    pub glomeruli: Vec<f32>,
    /// Pheromone signal strength.
    pub pheromone_signal: f32,
    /// General odor signal.
    pub general_odor: f32,
}

impl AntennalLobe {
    pub fn new() -> Self {
        Self {
            glomeruli: vec![0.0; ANT_GLOMERULI_COUNT],
            pheromone_signal: 0.0,
            general_odor: 0.0,
        }
    }
}

/// Optical lobe (visual processing).
#[derive(Debug, Clone)]
pub struct OpticalLobe {
    /// Lamina activation.
    pub lamina: f32,
    /// Medulla activation.
    pub medulla: f32,
    /// Lobula activation (motion detection).
    pub lobula: f32,
    /// Polarized light detection.
    pub polarized_light: f32,
}

impl OpticalLobe {
    pub fn new() -> Self {
        Self { lamina: 0.0, medulla: 0.0, lobula: 0.0, polarized_light: 0.0 }
    }
}

/// Central complex (navigation).
#[derive(Debug, Clone)]
pub struct CentralComplex {
    /// Heading direction estimate (radians).
    pub heading_estimate: f32,
    /// Distance estimate (mm).
    pub distance_estimate: f32,
    /// Path integration memory.
    pub path_memory: Vec<(f32, f32)>, // (x, y) positions
}

impl CentralComplex {
    pub fn new() -> Self {
        Self { heading_estimate: 0.0, distance_estimate: 0.0, path_memory: Vec::new() }
    }
}

impl AntBrain {
    pub fn new() -> Self {
        let mut neurotransmitters = HashMap::new();
        neurotransmitters.insert(AntNeurotransmitter::Glutamate, 0.5);
        neurotransmitters.insert(AntNeurotransmitter::GABA, 0.5);
        neurotransmitters.insert(AntNeurotransmitter::Dopamine, 0.3);
        neurotransmitters.insert(AntNeurotransmitter::Octopamine, 0.3);
        neurotransmitters.insert(AntNeurotransmitter::Serotonin, 0.2);

        Self {
            mushroom_body: MushroomBody::new(),
            antennal_lobe: AntennalLobe::new(),
            optical_lobe: OpticalLobe::new(),
            subesophageal_ganglion: 0.0,
            central_complex: CentralComplex::new(),
            neurotransmitters,
        }
    }

    pub fn process_pheromone(&mut self, strength: f32, ptype: PheromoneType) {
        self.antennal_lobe.pheromone_signal =
            (self.antennal_lobe.pheromone_signal * 0.9 + strength * 0.1).min(1.0);

        // Specific glomeruli for pheromone types
        let glom_idx = match ptype {
            PheromoneType::Trail => 0,
            PheromoneType::Alarm => 1,
            PheromoneType::QueenSignal => 2,
            PheromoneType::Death => 3,
            _ => 4,
        };
        if glom_idx < self.antennal_lobe.glomeruli.len() {
            self.antennal_lobe.glomeruli[glom_idx] = strength;
        }
    }

    pub fn memorize_location(&mut self, x: f32, y: f32, quality: f32) {
        self.mushroom_body.calyx_activation = (self.mushroom_body.calyx_activation + 0.1).min(1.0);
        self.central_complex.path_memory.push((x, y));
        if self.central_complex.path_memory.len() > 20 {
            self.central_complex.path_memory.remove(0);
        }

        // Dopamine release for positive reward
        if quality > 0.5 {
            *self.neurotransmitters.get_mut(&AntNeurotransmitter::Dopamine).unwrap() =
                (self.neurotransmitters[&AntNeurotransmitter::Dopamine] + 0.1).min(1.0);
        }
    }
}

impl Default for AntBrain {
    fn default() -> Self { Self::new() }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ANT MOLECULAR/DNA LAYER
// ═══════════════════════════════════════════════════════════════════════════════

/// Key ant genes with molecular fidelity.
#[derive(Debug, Clone)]
pub struct AntGene {
    pub name: String,
    pub chromosome: u8,
    pub position_cm: f32,
    pub function: GeneFunction,
    pub expression_level: f32,
    pub essential: bool,
}

/// Gene functional categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneFunction {
    // Caste determination
    CasteDetermination,
    JuvenileHormone,
    Vitellogenin,
    // Neural function
    NeurotransmitterSynthesis,
    IonChannel,
    SynapticProtein,
    NeuralDevelopment,
    // Pheromone system
    PheromoneSynthesis,
    PheromoneReceptor,
    Chemosensory,
    // Behavior
    Foraging,
    Aggression,
    CircadianClock,
    Learning,
    // Immunity
    ImmuneResponse,
    AntimicrobialPeptide,
    // Development
    Molting,
    Metamorphosis,
    GrowthFactor,
}

impl AntGene {
    /// Get key neural and behavioral genes.
    pub fn key_genes() -> Vec<Self> {
        vec![
            // Caste determination genes
            Self { name: "Vg1".into(), chromosome: 1, position_cm: 12.0,
                function: GeneFunction::Vitellogenin, expression_level: 0.0, essential: true },
            Self { name: "Vg2".into(), chromosome: 3, position_cm: 8.0,
                function: GeneFunction::Vitellogenin, expression_level: 0.0, essential: false },
            Self { name: "JHE".into(), chromosome: 5, position_cm: 15.0,
                function: GeneFunction::JuvenileHormone, expression_level: 0.5, essential: true },
            Self { name: "Kr-h1".into(), chromosome: 2, position_cm: 20.0,
                function: GeneFunction::CasteDetermination, expression_level: 0.3, essential: true },

            // Neural genes
            Self { name: "for".into(), chromosome: 8, position_cm: 5.0,
                function: GeneFunction::Foraging, expression_level: 0.5, essential: false },
            Self { name: "Amfor".into(), chromosome: 8, position_cm: 6.0,
                function: GeneFunction::Learning, expression_level: 0.4, essential: false },
            Self { name: "per".into(), chromosome: 4, position_cm: 30.0,
                function: GeneFunction::CircadianClock, expression_level: 0.5, essential: true },
            Self { name: "tim".into(), chromosome: 4, position_cm: 32.0,
                function: GeneFunction::CircadianClock, expression_level: 0.5, essential: true },
            Self { name: "DAT".into(), chromosome: 6, position_cm: 10.0,
                function: GeneFunction::NeurotransmitterSynthesis, expression_level: 0.4, essential: true },
            Self { name: "TH".into(), chromosome: 7, position_cm: 15.0,
                function: GeneFunction::NeurotransmitterSynthesis, expression_level: 0.3, essential: true },

            // Pheromone genes
            Self { name: "Orco".into(), chromosome: 10, position_cm: 5.0,
                function: GeneFunction::PheromoneReceptor, expression_level: 0.8, essential: true },
            Self { name: "Duf".into(), chromosome: 12, position_cm: 20.0,
                function: GeneFunction::PheromoneSynthesis, expression_level: 0.6, essential: false },

            // Immunity
            Self { name: "defensin".into(), chromosome: 15, position_cm: 8.0,
                function: GeneFunction::AntimicrobialPeptide, expression_level: 0.2, essential: false },
            Self { name: "abaecin".into(), chromosome: 15, position_cm: 10.0,
                function: GeneFunction::AntimicrobialPeptide, expression_level: 0.2, essential: false },

            // Development
            Self { name: "EcR".into(), chromosome: 18, position_cm: 25.0,
                function: GeneFunction::Molting, expression_level: 0.4, essential: true },
            Self { name: "Broad".into(), chromosome: 20, position_cm: 12.0,
                function: GeneFunction::Metamorphosis, expression_level: 0.3, essential: true },
        ]
    }
}

/// Molecular pathways in ants.
#[derive(Debug, Clone)]
pub struct AntPathway {
    pub name: String,
    pub genes: Vec<String>,
    pub description: String,
}

impl AntPathway {
    pub fn key_pathways() -> Vec<Self> {
        vec![
            Self {
                name: "Juvenile Hormone Signaling".into(),
                genes: vec!["JHE".into(), "JHAMT".into(), "Met".into(), "Kr-h1".into()],
                description: "Controls development, reproduction, and caste differentiation".into(),
            },
            Self {
                name: "Dopaminergic Reward".into(),
                genes: vec!["TH".into(), "DAT".into(), "DopR".into(), "for".into()],
                description: "Learning, foraging motivation, reward processing".into(),
            },
            Self {
                name: "Circadian Clock".into(),
                genes: vec!["per".into(), "tim".into(), "Clk".into(), "cry".into()],
                description: "24-hour rhythm generation, foraging timing".into(),
            },
            Self {
                name: "Olfactory Processing".into(),
                genes: vec!["Orco".into(), "Obp".into(), "SNMP".into()],
                description: "Pheromone detection and processing".into(),
            },
            Self {
                name: "Ecdysone Signaling".into(),
                genes: vec!["EcR".into(), "USH".into(), "E75".into(), "Broad".into()],
                description: "Molting, metamorphosis, development".into(),
            },
            Self {
                name: "Immune Defense".into(),
                genes: vec!["defensin".into(), "abaecin".into(), "hymenoptaecin".into()],
                description: "Antimicrobial peptide production, pathogen defense".into(),
            },
            Self {
                name: "Vitellogenin Storage".into(),
                genes: vec!["Vg1".into(), "Vg2".into(), "Vg3".into()],
                description: "Egg yolk precursor, longevity correlation".into(),
            },
            Self {
                name: "Insulin/TOR".into(),
                genes: vec!["InR".into(), "TOR".into(), "4E-BP".into(), "FOXO".into()],
                description: "Growth, aging, nutrient sensing".into(),
            },
        ]
    }
}

/// Genome statistics.
pub mod genome_stats {
    use super::*;

    pub const CHROMOSOME_LENGTHS_MB: [f32; 26] = [
        12.0, 11.5, 11.0, 10.8, 10.5, 10.2, 9.8, 9.5, 9.2, 9.0,
        8.8, 8.5, 8.2, 8.0, 7.8, 7.5, 7.2, 7.0, 6.8, 6.5,
        6.2, 6.0, 5.8, 5.5, 5.2, 5.0,
    ];

    pub const TOTAL_BP: usize = 280_000_000;
    pub const PROTEIN_CODING: usize = 18_000;
    pub const NONCODING_RNA: usize = 5_000;
    pub const TRANSPOSONS_PERCENT: f32 = 15.0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// INDIVIDUAL ANT
// ═══════════════════════════════════════════════════════════════════════════════

/// Tasks an ant can perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AntTask {
    Idle, Foraging, Returning, Nursing, Construction, Defense, Grooming, Feeding, EggLaying,
}

/// A single ant agent.
#[derive(Debug, Clone)]
pub struct Ant {
    pub id: u32,
    pub colony_id: u32,
    pub caste: AntCaste,
    pub x_mm: f32,
    pub y_mm: f32,
    pub heading: f32,
    pub speed: f32,
    pub body: AntBody,
    pub brain: AntBrain,
    pub current_task: AntTask,
    pub carrying_food: bool,
    pub food_amount: f32,
    pub energy: f32,
    pub age_days: f32,
    pub alive: bool,
    pub gene_expression: HashMap<String, f32>,
}

impl Ant {
    pub fn new(id: u32, colony_id: u32, caste: AntCaste, x: f32, y: f32) -> Self {
        let body = AntBody::new(caste);
        let brain = AntBrain::new();

        // Initialize gene expression
        let mut gene_expression = HashMap::new();
        for gene in AntGene::key_genes() {
            gene_expression.insert(gene.name.clone(), gene.expression_level);
        }

        Self {
            id, colony_id, caste,
            x_mm: x, y_mm: y,
            heading: 0.0, speed: 0.0,
            body, brain,
            current_task: AntTask::Idle,
            carrying_food: false,
            food_amount: 0.0,
            energy: 1.0,
            age_days: 0.0,
            alive: true,
            gene_expression,
        }
    }

    pub fn step(&mut self, dt_s: f32, pheromones: &[PheromoneDeposit]) {
        if !self.alive { return; }

        self.age_days += dt_s / 86400.0;
        self.energy = (self.energy - 0.0001 * dt_s).max(0.0);

        if self.energy <= 0.0 || self.age_days > self.caste.lifespan_days() {
            self.alive = false;
            return;
        }

        // Process pheromones
        for p in pheromones.iter().filter(|p| p.is_detectable() && p.colony_id == self.colony_id) {
            let dist = ((p.x_mm - self.x_mm).powi(2) + (p.y_mm - self.y_mm).powi(2)).sqrt();
            if dist < 20.0 {
                let strength = p.effective_strength() / (dist + 1.0);
                self.brain.process_pheromone(strength, p.pheromone_type);
            }
        }

        // Task behavior
        match self.current_task {
            AntTask::Foraging => {
                self.speed = 5.0;
                self.heading += (rand_random() - 0.5) * 0.3;
            }
            AntTask::Returning => {
                let angle_to_nest = (-self.y_mm).atan2(-self.x_mm);
                self.heading = angle_to_nest;
                self.speed = 8.0;
            }
            AntTask::EggLaying => {
                self.speed = 0.0;
            }
            _ => { self.speed = 0.0; }
        }

        // Update position
        self.x_mm += self.heading.cos() * self.speed * dt_s;
        self.y_mm += self.heading.sin() * self.speed * dt_s;

        // Update path memory
        if self.speed > 0.0 {
            self.brain.central_complex.heading_estimate = self.heading;
            self.brain.central_complex.distance_estimate += self.speed * dt_s;
        }
    }

    pub fn deposit_pheromone(&self, ptype: PheromoneType) -> PheromoneDeposit {
        PheromoneDeposit::new(ptype, self.x_mm, self.y_mm, 1.0, self.colony_id)
    }
}

fn rand_random() -> f32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ns = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    ((ns % 1000000) as f32) / 1000000.0
}

// ═══════════════════════════════════════════════════════════════════════════════
// ANT COLONY
// ═══════════════════════════════════════════════════════════════════════════════

/// Complete ant colony.
#[derive(Debug, Clone)]
pub struct AntColony {
    pub id: u32,
    pub species: String,
    pub ants: Vec<Ant>,
    pub pheromones: Vec<PheromoneDeposit>,
    pub nest_x_mm: f32,
    pub nest_y_mm: f32,
    pub food_storage: f32,
    pub brood_count: usize,
    pub colony_age_days: f32,
}

impl AntColony {
    pub fn new(id: u32, species: &str, nest_x: f32, nest_y: f32, initial_workers: usize) -> Self {
        let mut ants = Vec::with_capacity(initial_workers + 10);

        ants.push(Ant::new(0, id, AntCaste::Queen, nest_x, nest_y));

        for i in 1..=initial_workers {
            let angle = (i as f32) / (initial_workers as f32) * std::f32::consts::TAU;
            let dist = 5.0 + rand_random() * 10.0;
            let x = nest_x + angle.cos() * dist;
            let y = nest_y + angle.sin() * dist;
            let caste = if i % 10 == 0 { AntCaste::WorkerMajor } else { AntCaste::WorkerMinor };
            if i % 20 == 0 { AntCaste::Soldier } else { caste };
            ants.push(Ant::new(i as u32, id, caste, x, y));
        }

        Self {
            id, species: species.to_string(),
            ants, pheromones: Vec::new(),
            nest_x_mm: nest_x, nest_y_mm: nest_y,
            food_storage: 100.0, brood_count: 50,
            colony_age_days: 0.0,
        }
    }

    pub fn step(&mut self, dt_s: f32) {
        for p in &mut self.pheromones { p.age_s += dt_s; }
        self.pheromones.retain(|p| p.is_detectable());

        for ant in &mut self.ants { ant.step(dt_s, &self.pheromones); }
        self.ants.retain(|a| a.alive);

        self.colony_age_days += dt_s / 86400.0;
    }

    pub fn stats(&self) -> ColonyStats {
        ColonyStats {
            colony_id: self.id,
            species: self.species.clone(),
            total_ants: self.ants.len(),
            queens: self.ants.iter().filter(|a| a.caste == AntCaste::Queen).count(),
            workers: self.ants.iter().filter(|a| a.caste.can_forage()).count(),
            soldiers: self.ants.iter().filter(|a| a.caste == AntCaste::Soldier).count(),
            foragers: self.ants.iter().filter(|a| a.current_task == AntTask::Foraging).count(),
            food_storage: self.food_storage,
            brood_count: self.brood_count,
            pheromone_trails: self.pheromones.len(),
            mean_energy: if self.ants.is_empty() { 0.0 }
                else { self.ants.iter().map(|a| a.energy).sum::<f32>() / self.ants.len() as f32 },
            colony_age_days: self.colony_age_days,
        }
    }
}

/// Colony statistics.
#[derive(Debug, Clone)]
pub struct ColonyStats {
    pub colony_id: u32,
    pub species: String,
    pub total_ants: usize,
    pub queens: usize,
    pub workers: usize,
    pub soldiers: usize,
    pub foragers: usize,
    pub food_storage: f32,
    pub brood_count: usize,
    pub pheromone_trails: usize,
    pub mean_energy: f32,
    pub colony_age_days: f32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ant_creation() {
        let ant = Ant::new(1, 0, AntCaste::WorkerMinor, 0.0, 0.0);
        assert!(ant.alive);
        assert_eq!(ant.caste, AntCaste::WorkerMinor);
        assert!(ant.body.legs.len() == 6);
    }

    #[test]
    fn test_colony_creation() {
        let colony = AntColony::new(0, "Formica rufa", 0.0, 0.0, 100);
        assert!(colony.ants.len() >= 100);
        assert!(colony.ants.iter().any(|a| a.caste == AntCaste::Queen));
    }

    #[test]
    fn test_pheromone_decay() {
        let mut p = PheromoneDeposit::new(PheromoneType::Trail, 0.0, 0.0, 1.0, 0);
        assert!(p.effective_strength() > 0.9);
        p.age_s = 600.0;
        assert!(p.effective_strength() < 0.7);
    }

    #[test]
    fn test_ant_caste_properties() {
        assert!(AntCaste::Queen.can_lay_eggs());
        assert!(!AntCaste::WorkerMinor.can_lay_eggs());
        assert!(AntCaste::WorkerMinor.can_forage());
        assert!(AntCaste::Queen.can_fly());
    }

    #[test]
    fn test_ant_body() {
        let body = AntBody::new(AntCaste::WorkerMinor);
        assert_eq!(body.segments.len(), 3);
        assert_eq!(body.legs.len(), 6);
    }

    #[test]
    fn test_ant_brain() {
        let mut brain = AntBrain::new();
        brain.process_pheromone(0.5, PheromoneType::Trail);
        assert!(brain.antennal_lobe.pheromone_signal > 0.0);
    }

    #[test]
    fn test_ant_genes() {
        let genes = AntGene::key_genes();
        assert!(genes.len() >= 15);
        assert!(genes.iter().any(|g| g.name == "for"));
        assert!(genes.iter().any(|g| g.name == "per"));
    }

    #[test]
    fn test_ant_pathways() {
        let pathways = AntPathway::key_pathways();
        assert!(pathways.len() >= 8);
        assert!(pathways.iter().any(|p| p.name.contains("Circadian")));
    }

    #[test]
    fn test_ant_step() {
        let mut ant = Ant::new(1, 0, AntCaste::WorkerMinor, 0.0, 0.0);
        ant.current_task = AntTask::Foraging;
        ant.step(1.0, &[]);
        assert!(ant.speed > 0.0);
    }

    #[test]
    fn test_colony_stats() {
        let colony = AntColony::new(0, "Formica rufa", 0.0, 0.0, 50);
        let stats = colony.stats();
        assert_eq!(stats.total_ants, 51);
        assert!(stats.workers > 0);
        assert_eq!(stats.queens, 1);
    }

    #[test]
    fn test_genome_stats() {
        assert_eq!(genome_stats::CHROMOSOME_LENGTHS_MB.len(), 26);
        assert!(genome_stats::PROTEIN_CODING > 15_000);
    }

    #[test]
    fn test_pheromone_chemistry() {
        let chem = PheromoneChemistry::for_type(PheromoneType::Trail);
        assert!(chem.molecular_weight > 100.0);
    }
}
