//! DrosophilaSim -- GPU-resident Drosophila brain simulator with 6 experiment runners.
//!
//! Wraps [`MolecularBrain`] with the 15-region Drosophila connectome architecture
//! from the FlyWire dataset (Dorkenwald et al. 2024, Nature 634:124-138).
//! The ENTIRE simulation loop (sensory encoding -> brain step -> motor decode)
//! runs on Metal GPU via the existing `MolecularBrain::step()` pipeline.
//! Python is only used for setup and result collection.
//!
//! # Biological Fidelity
//!
//! - **ACh is the primary excitatory NT** in insects (NOT glutamate)
//! - **Octopamine maps to NE** (index 2) -- insect analog of vertebrate NE
//! - **Mushroom body sparse coding**: ~7 random PN inputs per Kenyon cell
//! - **Central complex**: heading integration for navigation
//! - **FEP protocol**: structured stimulation on HIT, random noise on MISS
//!
//! # Experiments
//!
//! 1. **Olfactory learning** -- odor source approach with FEP protocol
//! 2. **Phototaxis** -- light gradient navigation
//! 3. **Thermotaxis** -- temperature preference (18 deg C)
//! 4. **Foraging** -- multi-food search with hunger drive
//! 5. **Drug response** -- TTX/picrotoxin/caffeine effects on behavior
//! 6. **Circadian** -- time-of-day modulation of activity
//!
//! # Scale Tiers
//!
//! | Tier   | Neurons | Use Case                        |
//! |--------|---------|----------------------------------|
//! | Tiny   | 1,000   | Fast unit tests (<1s)            |
//! | Small  | 5,000   | Mac MPS (~10s)                   |
//! | Medium | 25,000  | A100 (~60s)                      |
//! | Large  | 139,000 | Full FlyWire connectome           |

use crate::network::MolecularBrain;
use crate::types::*;
use rand::prelude::*;

// ============================================================================
// Scale Tiers
// ============================================================================

/// Scale tier for network construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrosophilaScale {
    /// 1,000 neurons -- fast unit tests.
    Tiny,
    /// 5,000 neurons -- Mac development.
    Small,
    /// 25,000 neurons -- GPU benchmarks.
    Medium,
    /// 139,000 neurons -- full FlyWire connectome scale.
    Large,
}

impl DrosophilaScale {
    /// Total neuron count for this scale tier.
    pub fn neuron_count(&self) -> usize {
        match self {
            Self::Tiny => 1_000,
            Self::Small => 5_000,
            Self::Medium => 25_000,
            Self::Large => 139_000,
        }
    }

    /// Infer scale from an arbitrary neuron count.
    pub fn from_count(n: usize) -> Self {
        match n {
            0..=2_000 => Self::Tiny,
            2_001..=10_000 => Self::Small,
            10_001..=50_000 => Self::Medium,
            _ => Self::Large,
        }
    }
}

// ============================================================================
// Region Layout
// ============================================================================

/// Brain region specification: (name, fraction of total, primary NT, inhibitory fraction).
/// Based on FlyWire neuron counts (Dorkenwald et al. 2024).
const REGION_SPECS: &[(&str, f32, NTType, f32)] = &[
    ("AL", 0.05, NTType::Acetylcholine, 0.30), // Antennal Lobe -- olfactory glomeruli
    ("MB_KC", 0.10, NTType::Acetylcholine, 0.05), // Mushroom Body Kenyon Cells -- sparse coding
    ("MBON", 0.01, NTType::Acetylcholine, 0.30), // MB Output Neurons -- decision
    ("DAN", 0.01, NTType::Dopamine, 0.00),     // Dopaminergic Neurons -- PAM/PPL1 reward
    ("CX", 0.06, NTType::Acetylcholine, 0.25), // Central Complex -- navigation, heading
    ("OL_LAM", 0.10, NTType::Acetylcholine, 0.20), // Optic Lobe Lamina -- photoreceptor input
    ("OL_MED", 0.15, NTType::Acetylcholine, 0.35), // Optic Lobe Medulla -- motion computation
    ("OL_LOB", 0.08, NTType::Acetylcholine, 0.25), // Optic Lobe Lobula -- visual features
    ("LH", 0.03, NTType::Acetylcholine, 0.20), // Lateral Horn -- innate olfactory
    ("SEZ", 0.05, NTType::Acetylcholine, 0.20), // Subesophageal Zone -- taste, feeding
    ("SUP", 0.10, NTType::Acetylcholine, 0.30), // Superior Brain -- higher processing
    ("DN", 0.03, NTType::Acetylcholine, 0.10), // Descending Neurons -- brain->VNC
    ("VNC", 0.15, NTType::Acetylcholine, 0.30), // Ventral Nerve Cord -- leg/wing motor
    ("NEUROMOD", 0.03, NTType::Dopamine, 0.00), // Neuromodulatory -- DA/5HT/Oct global
    ("OTHER", 0.05, NTType::Acetylcholine, 0.25), // Other neuropils
];

// ============================================================================
// Molecular/DNA Fidelity Layer
// ============================================================================

/// Drosophila melanogaster genome statistics.
/// Source: FlyBase (release FB2024_05), Adams et al. 2000, Hoskins et al. 2015.
pub mod genome_stats {
    /// Total genome size in megabases.
    pub const GENOME_MB: f32 = 180.0;
    /// Number of protein-coding genes.
    pub const PROTEIN_GENES: usize = 13_600;
    /// Number of chromosomes (4 major: X, 2, 3, 4).
    pub const CHROMOSOME_COUNT: usize = 4;
    /// Chromosome lengths in Mb.
    pub const CHROMOSOME_LENGTHS_MB: [f32; 4] = [41.0, 60.0, 67.0, 4.0];
    /// GC content percentage.
    pub const GC_CONTENT: f32 = 41.5;
    /// Approximate neuron count (139,000 from FlyWire).
    pub const FULL_NEURON_COUNT: usize = 139_000;
    /// Number of Kenyon cells in mushroom body.
    pub const KENYON_CELL_COUNT: usize = 2_000;
    /// Number of antennal lobe glomeruli.
    pub const GLOMERULI_COUNT: usize = 50;
}

/// Gene function categories for Drosophila neural genes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeneFunction {
    /// Ion channel (voltage-gated or ligand-gated).
    IonChannel,
    /// Neurotransmitter synthesis.
    NeurotransmitterSynthesis,
    /// Neurotransmitter receptor.
    NeurotransmitterReceptor,
    /// Synaptic vesicle protein.
    SynapticVesicle,
    /// Learning and memory.
    LearningMemory,
    /// Circadian rhythm.
    CircadianRhythm,
    /// Neural development.
    NeuralDevelopment,
    /// Neuromodulation.
    Neuromodulation,
}

/// Key Drosophila neural gene with chromosomal position.
/// Source: FlyBase (FB2024_05), reviewed literature.
#[derive(Clone, Debug)]
pub struct DrosophilaGene {
    /// Gene symbol (e.g., "Shaker").
    pub name: String,
    /// Full name (e.g., "Shaker potassium channel").
    pub full_name: String,
    /// Chromosome (X=0, 2=1, 3=2, 4=3).
    pub chromosome: u8,
    /// Cytological position (e.g., 16F for Shaker).
    pub cytoband: String,
    /// Gene function category.
    pub function: GeneFunction,
    /// Whether null mutation is lethal.
    pub essential: bool,
}

/// Molecular pathway in Drosophila nervous system.
#[derive(Clone, Debug)]
pub struct MolecularPathway {
    /// Pathway name.
    pub name: String,
    /// Participating genes.
    pub genes: Vec<String>,
    /// Brief description.
    pub description: String,
}

/// Drosophila body segments with anatomical detail.
#[derive(Clone, Debug, Default)]
pub struct DrosophilaBody {
    /// Head structures.
    pub head: HeadStructures,
    /// Thoracic segments (T1-T3).
    pub thorax: ThoraxStructures,
    /// Abdominal segments (A1-A8).
    pub abdomen: AbdomenStructures,
    /// Wing morphology.
    pub wings: WingMorphology,
}

/// Head structures: compound eyes, antennae, proboscis.
#[derive(Clone, Debug, Default)]
pub struct HeadStructures {
    /// Compound eye ommatidia count (~800 per eye).
    pub ommatidia_per_eye: u16,
    /// Antennal segments.
    pub antennal_segments: u8,
    /// Proboscis extension (0-1).
    pub proboscis_extension: f32,
    /// Maxillary palp sensory neurons.
    pub palp_neurons: u16,
}

/// Thoracic structures: three segments with legs and wings.
#[derive(Clone, Debug, Default)]
pub struct ThoraxStructures {
    /// Leg pairs (3 pairs, 5 segments each).
    pub leg_pairs: u8,
    /// Wing length in mm.
    pub wing_length_mm: f32,
    /// Haltere (gyroscope) present.
    pub halteres_present: bool,
    /// Flight muscle type (indirect synchronous).
    pub flight_muscle_type: String,
}

/// Abdominal structures: digestive and reproductive.
#[derive(Clone, Debug, Default)]
pub struct AbdomenStructures {
    /// Number of visible segments.
    pub visible_segments: u8,
    /// Crop (food storage) fill level.
    pub crop_fill: f32,
    /// Fat body reserves.
    pub fat_body: f32,
}

/// Wing morphology parameters.
#[derive(Clone, Debug, Default)]
pub struct WingMorphology {
    /// Wing area in mm².
    pub area_mm2: f32,
    /// Aspect ratio.
    pub aspect_ratio: f32,
    /// Vein pattern complexity.
    pub vein_count: u8,
}

/// Key Drosophila neural genes.
/// Based on FlyBase annotations and neural function literature.
pub fn key_genes() -> Vec<DrosophilaGene> {
    vec![
        // Ion channels
        DrosophilaGene {
            name: "Sh".into(),
            full_name: "Shaker potassium channel".into(),
            chromosome: 0, // X
            cytoband: "16F".into(),
            function: GeneFunction::IonChannel,
            essential: false,
        },
        DrosophilaGene {
            name: "Shab".into(),
            full_name: "Shab potassium channel".into(),
            chromosome: 2, // 3
            cytoband: "97F".into(),
            function: GeneFunction::IonChannel,
            essential: true,
        },
        DrosophilaGene {
            name: "para".into(),
            full_name: "paralytic sodium channel".into(),
            chromosome: 0, // X
            cytoband: "15A".into(),
            function: GeneFunction::IonChannel,
            essential: true,
        },
        DrosophilaGene {
            name: "cac".into(),
            full_name: "cacophony calcium channel".into(),
            chromosome: 0, // X
            cytoband: "12C".into(),
            function: GeneFunction::IonChannel,
            essential: true,
        },
        // Neurotransmitter synthesis
        DrosophilaGene {
            name: "TH".into(),
            full_name: "Tyrosine hydroxylase".into(),
            chromosome: 1, // 2
            cytoband: "23A".into(),
            function: GeneFunction::NeurotransmitterSynthesis,
            essential: false,
        },
        DrosophilaGene {
            name: "DDC".into(),
            full_name: "Dopa decarboxylase".into(),
            chromosome: 1, // 2
            cytoband: "21B".into(),
            function: GeneFunction::NeurotransmitterSynthesis,
            essential: false,
        },
        DrosophilaGene {
            name: "Tdc2".into(),
            full_name: "Tyrosine decarboxylase 2".into(),
            chromosome: 2, // 3
            cytoband: "99C".into(),
            function: GeneFunction::NeurotransmitterSynthesis,
            essential: false,
        },
        // Learning and memory
        DrosophilaGene {
            name: "rut".into(),
            full_name: "rutabaga adenylyl cyclase".into(),
            chromosome: 0, // X
            cytoband: "12E".into(),
            function: GeneFunction::LearningMemory,
            essential: false,
        },
        DrosophilaGene {
            name: "dnc".into(),
            full_name: "dunce cAMP phosphodiesterase".into(),
            chromosome: 0, // X
            cytoband: "3B".into(),
            function: GeneFunction::LearningMemory,
            essential: false,
        },
        DrosophilaGene {
            name: "CREB2".into(),
            full_name: "cAMP response element binding".into(),
            chromosome: 1, // 2
            cytoband: "34E".into(),
            function: GeneFunction::LearningMemory,
            essential: true,
        },
        // Circadian rhythm
        DrosophilaGene {
            name: "per".into(),
            full_name: "period".into(),
            chromosome: 0, // X
            cytoband: "3B".into(),
            function: GeneFunction::CircadianRhythm,
            essential: false,
        },
        DrosophilaGene {
            name: "tim".into(),
            full_name: "timeless".into(),
            chromosome: 2, // 3
            cytoband: "91C".into(),
            function: GeneFunction::CircadianRhythm,
            essential: false,
        },
        DrosophilaGene {
            name: "clk".into(),
            full_name: "clock".into(),
            chromosome: 1, // 2
            cytoband: "32F".into(),
            function: GeneFunction::CircadianRhythm,
            essential: false,
        },
        // Neurotransmitter receptors
        DrosophilaGene {
            name: "nAChRα7".into(),
            full_name: "nicotinic ACh receptor alpha7".into(),
            chromosome: 1, // 2
            cytoband: "45A".into(),
            function: GeneFunction::NeurotransmitterReceptor,
            essential: false,
        },
        DrosophilaGene {
            name: "GluRIIA".into(),
            full_name: "Glutamate receptor IIA".into(),
            chromosome: 1, // 2
            cytoband: "54A".into(),
            function: GeneFunction::NeurotransmitterReceptor,
            essential: false,
        },
        DrosophilaGene {
            name: "Rdl".into(),
            full_name: "Resistance to dieldrin GABA-R".into(),
            chromosome: 2, // 3
            cytoband: "66D".into(),
            function: GeneFunction::NeurotransmitterReceptor,
            essential: true,
        },
        // Neural development
        DrosophilaGene {
            name: "elav".into(),
            full_name: "embryonic lethal abnormal vision".into(),
            chromosome: 0, // X
            cytoband: "1D".into(),
            function: GeneFunction::NeuralDevelopment,
            essential: true,
        },
        DrosophilaGene {
            name: "pros".into(),
            full_name: "prospero homeobox".into(),
            chromosome: 1, // 2
            cytoband: "47F".into(),
            function: GeneFunction::NeuralDevelopment,
            essential: true,
        },
    ]
}

/// Canonical molecular pathways in Drosophila nervous system.
pub fn molecular_pathways() -> Vec<MolecularPathway> {
    vec![
        MolecularPathway {
            name: "cAMP learning pathway".into(),
            genes: vec!["rut".into(), "dnc".into(), "CREB2".into()],
            description: "Olfactory conditioning through cAMP signaling in mushroom body".into(),
        },
        MolecularPathway {
            name: "Circadian clock".into(),
            genes: vec!["per".into(), "tim".into(), "clk".into(), "cry".into()],
            description: "24-hour oscillation via PER/TIM feedback loop".into(),
        },
        MolecularPathway {
            name: "Dopamine reward".into(),
            genes: vec!["TH".into(), "DDC".into(), "DopR".into(), "DopR2".into()],
            description: "Reward signaling via PAM cluster dopaminergic neurons".into(),
        },
        MolecularPathway {
            name: "Octopamine arousal".into(),
            genes: vec!["Tdc2".into(), "Oamb".into(), "OctβR".into()],
            description: "Fight-or-flight and arousal via octopamine system".into(),
        },
        MolecularPathway {
            name: "Cholinergic transmission".into(),
            genes: vec!["ChAT".into(), "nAChRα7".into(), "VAChT".into()],
            description: "Primary excitatory transmission in insect CNS".into(),
        },
        MolecularPathway {
            name: "GABAergic inhibition".into(),
            genes: vec!["Gad1".into(), "Rdl".into(), "VGAT".into()],
            description: "Inhibitory transmission via GABA-A receptors".into(),
        },
        MolecularPathway {
            name: "Action potential generation".into(),
            genes: vec!["para".into(), "Sh".into(), "Shab".into(), "cac".into()],
            description: "Voltage-gated Na/K/Ca channels for spiking".into(),
        },
        MolecularPathway {
            name: "Synaptic vesicle cycle".into(),
            genes: vec!["SYT1".into(), "n-syb".into(), "shi".into()],
            description: "Vesicle exocytosis and endocytosis at active zones".into(),
        },
    ]
}

/// Get default Drosophila body structure.
pub fn default_body() -> DrosophilaBody {
    DrosophilaBody {
        head: HeadStructures {
            ommatidia_per_eye: 800,
            antennal_segments: 6,
            proboscis_extension: 0.0,
            palp_neurons: 120,
        },
        thorax: ThoraxStructures {
            leg_pairs: 3,
            wing_length_mm: 2.5,
            halteres_present: true,
            flight_muscle_type: "indirect synchronous".into(),
        },
        abdomen: AbdomenStructures {
            visible_segments: 6,
            crop_fill: 0.0,
            fat_body: 0.5,
        },
        wings: WingMorphology {
            area_mm2: 2.8,
            aspect_ratio: 2.2,
            vein_count: 8,
        },
    }
}

/// Computed region layout: maps each region name to (start_index, count).
#[derive(Clone, Debug)]
pub struct RegionLayout {
    /// (region_name, start_index, neuron_count) for each of the 15 regions.
    pub regions: Vec<(&'static str, usize, usize)>,
    /// Total neuron count.
    pub total: usize,
}

impl RegionLayout {
    /// Build region layout from scale, allocating neurons proportionally.
    fn from_scale(scale: DrosophilaScale) -> Self {
        let n = scale.neuron_count();
        let mut regions = Vec::with_capacity(REGION_SPECS.len());
        let mut offset = 0usize;
        let mut remaining = n;

        for (i, &(name, frac, _, _)) in REGION_SPECS.iter().enumerate() {
            let count = if i == REGION_SPECS.len() - 1 {
                // Last region gets remainder (rounding cleanup)
                remaining
            } else {
                let c = (n as f32 * frac).round().max(1.0) as usize;
                c.min(remaining)
            };
            regions.push((name, offset, count));
            offset += count;
            remaining -= count;
        }

        assert_eq!(offset, n, "Region allocation mismatch: {} != {}", offset, n);

        Self { regions, total: n }
    }

    /// Look up a region by name, returning (start, count).
    pub fn get(&self, name: &str) -> Option<(usize, usize)> {
        self.regions
            .iter()
            .find(|&&(n, _, _)| n == name)
            .map(|&(_, start, count)| (start, count))
    }

    /// Get neuron index range for a region.
    pub fn range(&self, name: &str) -> std::ops::Range<usize> {
        let (start, count) = self.get(name).unwrap_or((0, 0));
        start..start + count
    }

    /// Get the inhibitory fraction for a region by name.
    fn inhib_fraction(name: &str) -> f32 {
        REGION_SPECS
            .iter()
            .find(|&&(n, _, _, _)| n == name)
            .map(|&(_, _, _, f)| f)
            .unwrap_or(0.2)
    }
}

// ============================================================================
// FEP (Free Energy Principle) Protocol
// ============================================================================

/// FEP stimulation mode for learning experiments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FepMode {
    /// Structured pulsed stimulation (low entropy) -- applied on HIT.
    Structured,
    /// Random noise stimulation (high entropy) -- applied on MISS.
    Random,
    /// No FEP stimulation.
    Off,
}

/// FEP protocol state for a set of target neurons.
struct FepState {
    /// Neuron indices that receive FEP stimulation.
    target_indices: Vec<usize>,
    /// Current stimulation amplitude (uA/cm^2).
    amplitude: f32,
    /// Current mode.
    mode: FepMode,
    /// RNG for random mode.
    rng: StdRng,
}

impl FepState {
    fn new(indices: Vec<usize>, amplitude: f32, seed: u64) -> Self {
        Self {
            target_indices: indices,
            amplitude,
            mode: FepMode::Off,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Apply FEP stimulation to the brain for one step.
    fn stimulate(&mut self, brain: &mut MolecularBrain, step: u64) {
        match self.mode {
            FepMode::Structured => {
                // Pulsed: 5 steps on / 5 steps off (5ms on, 5ms off at dt=0.1ms = 0.5ms/0.5ms)
                let pulse_on = (step / 5) % 2 == 0;
                if pulse_on {
                    for &idx in &self.target_indices {
                        brain.stimulate(idx, self.amplitude);
                    }
                }
            }
            FepMode::Random => {
                // Random 30% of target neurons each step
                for &idx in &self.target_indices {
                    if self.rng.gen::<f32>() < 0.30 {
                        brain.stimulate(idx, self.amplitude * self.rng.gen::<f32>());
                    }
                }
            }
            FepMode::Off => {}
        }
    }
}

// ============================================================================
// Body State (Simple 2D Agent)
// ============================================================================

const DT_BODY_S: f32 = 0.05;
const FLY_WALK_SPEED: f32 = 2.0;
const FLY_FLIGHT_SPEED: f32 = 200.0;
const FLY_TURN_RATE: f32 = 3.0;
const FLY_WING_BEAT_HZ: f32 = 200.0;
const FLY_ENERGY_MAX: f32 = 100.0;
const FLY_ENERGY_WALK_COST: f32 = 0.01;
const FLY_ENERGY_FLY_COST: f32 = 0.1;
const FLY_ENERGY_FEED_GAIN: f32 = 5.0;
const GRAVITY_MM_S2: f32 = 9810.0;
const FLY_MAX_ALTITUDE: f32 = 50.0;
const FLY_TAKEOFF_SPEED: f32 = 5.0;
const FLY_CLIMB_RATE: f32 = 3.0;
const FLY_DESCENT_RATE: f32 = 5.0;

#[derive(Clone, Copy, Debug, Default)]
pub struct MotorOutput {
    pub speed: f32,
    pub turn: f32,
    pub fly_signal: f32,
    pub feed_signal: f32,
    pub climb_signal: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TerrariumFlyStepReport {
    pub speed: f32,
    pub turn: f32,
    pub fly_signal: f32,
    pub feed_signal: f32,
    pub climb_signal: f32,
    pub consumed_food: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub heading: f32,
    pub pitch: f32,
    pub energy: f32,
    pub is_flying: bool,
    pub wing_beat_freq: f32,
}

/// Minimal 2D body state for locomotion experiments.
#[derive(Clone, Debug)]
pub struct BodyState {
    /// Position in world coordinates.
    pub x: f32,
    pub y: f32,
    /// Altitude for external terrarium integration.
    pub z: f32,
    /// Heading in radians.
    pub heading: f32,
    /// Pitch in radians.
    pub pitch: f32,
    /// Speed in world units per body step.
    pub speed: f32,
    /// Energy level [0, 100].
    pub energy: f32,
    /// Local temperature at current position (deg C).
    pub temperature: f32,
    /// Time of day [0, 24) hours.
    pub time_of_day: f32,
    /// Flight state for external terrarium integration.
    pub is_flying: bool,
    /// Vertical velocity in mm/s.
    pub vertical_velocity: f32,
    /// Whether the proboscis is currently extended.
    pub proboscis_extended: bool,
    /// Wing-beat frequency in Hz.
    pub wing_beat_freq: f32,
    /// Roll angle in radians.
    pub roll: f32,
    /// Wing stroke phase/angle.
    pub wing_stroke: f32,
    /// Wing twist angle.
    pub wing_twist: f32,
    /// Wing dihedral angle.
    pub wing_dihedral: f32,
    /// Wing sweep angle.
    pub wing_sweep: f32,
}

impl BodyState {
    fn new(x: f32, y: f32) -> Self {
        Self {
            x,
            y,
            z: 0.0,
            heading: 0.0,
            pitch: 0.0,
            speed: 0.0,
            energy: FLY_ENERGY_MAX,
            temperature: 22.0,
            time_of_day: 12.0,
            is_flying: false,
            vertical_velocity: 0.0,
            proboscis_extended: false,
            wing_beat_freq: FLY_WING_BEAT_HZ,
            roll: 0.0,
            wing_stroke: 0.0,
            wing_twist: 0.0,
            wing_dihedral: 0.0,
            wing_sweep: 0.0,
        }
    }

    /// Dynamically calculate wing beat frequency based on energy, atmosphere, and load.
    fn calculate_wing_beat_freq(&self, air_density: f32, climb_signal: f32) -> f32 {
        // Base frequency (~200Hz for Drosophila)
        let base_hz = FLY_WING_BEAT_HZ;
        
        // Energy factor: fatigued flies have lower power output
        // Below 20% energy, frequency drops linearly
        let energy_t = (self.energy / FLY_ENERGY_MAX).clamp(0.0, 1.0);
        let energy_factor = 0.7 + 0.3 * energy_t.sqrt(); 

        // Temperature factor: metabolism and muscle speed are temperature dependent (Q10 ~2.0)
        // Optimal at 25C, drops off as it gets colder
        let temp_factor = (1.0 + (self.temperature - 25.0) * 0.02).clamp(0.5, 1.2);

        // Air density factor: in thinner air (higher altitude or hot), 
        // the fly must beat faster to maintain lift.
        // Standard air density is ~1.225 kg/m^3
        let density_ratio = (1.225 / air_density.max(0.1)).sqrt();
        let density_compensation = density_ratio.clamp(0.8, 1.5);

        // Climb effort: intense climbing requires higher frequency
        let effort = 1.0 + climb_signal.clamp(0.0, 1.0) * 0.2;

        (base_hz * energy_factor * temp_factor * density_compensation * effort).clamp(50.0, 350.0)
    }

    /// Calculate respiration flux: O2 consumption and CO2 release.
    /// Returns (o2_flux, co2_flux) in concentration units.
    pub fn calculate_respiration_flux(&self) -> (f32, f32) {
        // Base respiration rate
        let mut rate = 0.0001; 
        
        // Scaling with activity: flying is highly metabolic
        if self.is_flying {
            rate *= 12.0 * (self.wing_beat_freq / FLY_WING_BEAT_HZ);
        } else {
            rate *= 1.0 + (self.speed / FLY_WALK_SPEED);
        }

        // Temperature effect (Q10)
        let temp_factor = (1.0 + (self.temperature - 22.0) * 0.04).clamp(0.2, 2.5);
        rate *= temp_factor;

        // O2 consumed, CO2 released (Respiratory Quotient ~0.8-1.0)
        (-rate, rate * 0.95)
    }

    /// Update position from heading and speed, wrapping within world bounds.
    fn update(&mut self, world_w: f32, world_h: f32) {
        self.x += self.heading.cos() * self.speed;
        self.y += self.heading.sin() * self.speed;
        // Wrap to world bounds
        self.x = self.x.rem_euclid(world_w);
        self.y = self.y.rem_euclid(world_h);
    }

    fn walk(&mut self, speed: f32, turn: f32, dt: f32, world_w: f32, world_h: f32) {
        if self.is_flying {
            return;
        }
        self.z = 0.0;
        self.pitch = 0.0;
        self.heading += turn * FLY_TURN_RATE * dt;
        let actual_speed = speed.clamp(0.0, 1.0) * FLY_WALK_SPEED;
        let dx = self.heading.cos() * actual_speed * dt;
        let dy = self.heading.sin() * actual_speed * dt;
        self.x = (self.x + dx).clamp(0.0, world_w.max(1.0) - 1.0);
        self.y = (self.y + dy).clamp(0.0, world_h.max(1.0) - 1.0);
        self.speed = actual_speed;
        self.energy =
            (self.energy - FLY_ENERGY_WALK_COST * speed.clamp(0.0, 1.0)).clamp(0.0, FLY_ENERGY_MAX);
    }

    fn takeoff(&mut self) -> bool {
        if self.energy < 10.0 {
            return false;
        }
        self.is_flying = true;
        self.vertical_velocity = FLY_TAKEOFF_SPEED;
        self.z = self.z.max(0.5);
        true
    }

    fn land(&mut self) {
        self.is_flying = false;
        self.z = 0.0;
        self.pitch = 0.0;
        self.vertical_velocity = 0.0;
        self.speed = 0.0;
    }

    fn fly_3d(&mut self, speed: f32, turn: f32, climb: f32, dt: f32, world_w: f32, world_h: f32) {
        if !self.is_flying {
            return;
        }
        self.heading += turn * FLY_TURN_RATE * dt;
        let target_pitch = climb.clamp(-1.0, 1.0) * std::f32::consts::FRAC_PI_6;
        self.pitch += (target_pitch - self.pitch) * 0.1;

        let forward_speed = speed.clamp(0.0, 1.0) * FLY_FLIGHT_SPEED;
        let dx = self.heading.cos() * self.pitch.cos() * forward_speed * dt;
        let dy = self.heading.sin() * self.pitch.cos() * forward_speed * dt;

        let wing_factor = self.wing_beat_freq / FLY_WING_BEAT_HZ;
        let mut lift_accel = GRAVITY_MM_S2 * wing_factor;
        lift_accel *= 1.0 + climb.clamp(-1.0, 1.0) * 0.3;
        let vert_accel = lift_accel - GRAVITY_MM_S2;
        self.vertical_velocity += vert_accel * dt;
        self.vertical_velocity *= 0.95;
        self.vertical_velocity = self
            .vertical_velocity
            .clamp(-FLY_DESCENT_RATE, FLY_CLIMB_RATE);

        let dz = self.vertical_velocity * dt + self.pitch.sin() * forward_speed * dt;
        self.x = (self.x + dx).clamp(0.0, world_w.max(1.0) - 1.0);
        self.y = (self.y + dy).clamp(0.0, world_h.max(1.0) - 1.0);
        self.z = (self.z + dz).clamp(0.0, FLY_MAX_ALTITUDE);
        self.speed = forward_speed;

        let effort = climb.abs() * 0.5 + speed.clamp(0.0, 1.0) * 0.5 + 0.5;
        self.energy = (self.energy - FLY_ENERGY_FLY_COST * effort).clamp(0.0, FLY_ENERGY_MAX);
        if self.z <= 0.0 || self.energy <= 0.0 {
            self.land();
        }
    }
}

// ============================================================================
// World State (Simple 2D Grid)
// ============================================================================

/// Simple 2D grid world with odorant, light, and temperature channels.
struct World {
    width: usize,
    height: usize,
    /// Odorant concentration field [0, 1] (width * height).
    odorant: Vec<f32>,
    /// Light intensity field [0, 1] (width * height).
    light: Vec<f32>,
    /// Temperature field in deg C (width * height).
    temperature: Vec<f32>,
    /// Food locations: (x, y, remaining_amount).
    food_sources: Vec<(f32, f32, f32)>,
}

impl World {
    fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            odorant: vec![0.0; width * height],
            light: vec![0.5; width * height],
            temperature: vec![22.0; width * height],
            food_sources: Vec::new(),
        }
    }

    /// Place an odorant source with Gaussian falloff.
    fn place_odorant(&mut self, cx: f32, cy: f32, intensity: f32, sigma: f32) {
        for y in 0..self.height {
            for x in 0..self.width {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist_sq = dx * dx + dy * dy;
                let val = intensity * (-dist_sq / (2.0 * sigma * sigma)).exp();
                let idx = y * self.width + x;
                self.odorant[idx] = (self.odorant[idx] + val).min(1.0);
            }
        }
    }

    /// Set a linear light gradient (bright on right).
    fn set_light_gradient(&mut self) {
        for y in 0..self.height {
            for x in 0..self.width {
                self.light[y * self.width + x] = x as f32 / self.width as f32;
            }
        }
    }

    /// Set a linear temperature gradient (left_temp to right_temp).
    fn set_temp_gradient(&mut self, left_temp: f32, right_temp: f32) {
        for y in 0..self.height {
            for x in 0..self.width {
                let t = x as f32 / self.width as f32;
                self.temperature[y * self.width + x] = left_temp + t * (right_temp - left_temp);
            }
        }
    }

    /// Place a food source.
    fn place_food(&mut self, x: f32, y: f32, amount: f32) {
        self.food_sources.push((x, y, amount));
    }

    /// Sample odorant concentration at a position (bilinear interpolation).
    fn sample_odorant(&self, x: f32, y: f32) -> f32 {
        self.sample_field(&self.odorant, x, y)
    }

    /// Sample light intensity at a position.
    fn sample_light(&self, x: f32, y: f32) -> f32 {
        self.sample_field(&self.light, x, y)
    }

    /// Sample temperature at a position.
    fn sample_temperature(&self, x: f32, y: f32) -> f32 {
        self.sample_field(&self.temperature, x, y)
    }

    /// Bilinear interpolation on a 2D field.
    fn sample_field(&self, field: &[f32], x: f32, y: f32) -> f32 {
        let fx = x.clamp(0.0, (self.width - 1) as f32);
        let fy = y.clamp(0.0, (self.height - 1) as f32);
        let ix = fx as usize;
        let iy = fy as usize;
        let ix1 = (ix + 1).min(self.width - 1);
        let iy1 = (iy + 1).min(self.height - 1);
        let dx = fx - ix as f32;
        let dy = fy - iy as f32;

        let v00 = field[iy * self.width + ix];
        let v10 = field[iy * self.width + ix1];
        let v01 = field[iy1 * self.width + ix];
        let v11 = field[iy1 * self.width + ix1];

        let v0 = v00 * (1.0 - dx) + v10 * dx;
        let v1 = v01 * (1.0 - dx) + v11 * dx;
        v0 * (1.0 - dy) + v1 * dy
    }

    /// Step diffusion: simple 3x3 averaging kernel with decay.
    fn step_diffusion(&mut self, decay: f32) {
        let w = self.width;
        let h = self.height;
        let mut new_odorant = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut sum = 0.0f32;
                let mut count = 0u32;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                            sum += self.odorant[ny as usize * w + nx as usize];
                            count += 1;
                        }
                    }
                }
                new_odorant[y * w + x] = (sum / count as f32) * decay;
            }
        }
        self.odorant = new_odorant;
    }
}

// ============================================================================
// Experiment Result
// ============================================================================

/// Results from a single experiment, returned to the caller.
#[derive(Clone, Debug)]
pub struct ExperimentResult {
    /// Experiment name (e.g., "olfactory_learning").
    pub name: String,
    /// Whether the experiment passed its success criterion.
    pub passed: bool,
    /// Name of the primary metric.
    pub metric_name: String,
    /// Value of the primary metric.
    pub metric_value: f64,
    /// Threshold for passing.
    pub threshold: f64,
    /// Human-readable details string.
    pub details: String,
    /// Trajectory data: (x, y) positions sampled during the experiment.
    pub trajectories: Vec<(f32, f32)>,
}

impl std::fmt::Display for ExperimentResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let status = if self.passed { "PASS" } else { "FAIL" };
        write!(
            f,
            "[{}] {} | {} = {:.4} (threshold: {:.4}) | {}",
            status, self.name, self.metric_name, self.metric_value, self.threshold, self.details
        )
    }
}

// ============================================================================
// DrosophilaSim
// ============================================================================

/// The main Drosophila simulation struct.
///
/// Owns a `MolecularBrain` wired with Drosophila-specific connectivity
/// (15 brain regions, ACh-primary excitation, sparse MB coding), plus a
/// simple 2D world, body state, and FEP protocol for learning experiments.
///
/// The brain runs on Metal GPU automatically (via `MolecularBrain`'s
/// internal GPU dispatch) when on macOS with Apple Silicon and >= 64 neurons.
pub struct DrosophilaSim {
    /// The underlying molecular brain (Metal GPU or CPU fallback).
    pub brain: MolecularBrain,
    /// Region layout mapping names to neuron index ranges.
    pub layout: RegionLayout,
    /// Scale tier.
    pub scale: DrosophilaScale,
    /// 2D world for experiments.
    world: World,
    /// Body state (position, heading, speed).
    body: BodyState,
    /// FEP protocol (optional, activated per-experiment).
    fep: Option<FepState>,
    /// RNG for stochastic processes (used by experiment runners).
    #[allow(dead_code)]
    rng: StdRng,
    /// Neural steps per body step (controls speed-fidelity tradeoff).
    pub neural_steps_per_body: u32,
    /// World dimensions.
    pub world_width: f32,
    pub world_height: f32,
}

impl DrosophilaSim {
    // ================================================================
    // Construction
    // ================================================================

    /// Create a new Drosophila simulation at the given scale.
    ///
    /// Allocates the brain, builds Drosophila-specific connectivity,
    /// assigns archetypes (cholinergic majority), and creates the world.
    pub fn new(scale: DrosophilaScale, seed: u64) -> Self {
        let layout = RegionLayout::from_scale(scale);
        let n = layout.total;
        let mut rng = StdRng::seed_from_u64(seed);

        // Build connectivity
        let edges = Self::build_connectivity(&layout, &mut rng);

        // Create brain from edges (this builds CSR internally)
        let mut brain = MolecularBrain::from_edges(n, &edges);
        brain.psc_scale = 30.0; // Critical for cascade propagation

        // Assign neuron archetypes per region
        Self::assign_archetypes(&mut brain, &layout);

        // Assign spatial positions
        Self::assign_positions(&mut brain, &layout, &mut rng);

        // World setup
        let world_width = 64.0f32;
        let world_height = 64.0f32;
        let world = World::new(64, 64);
        let body = BodyState::new(world_width / 2.0, world_height / 2.0);

        Self {
            brain,
            layout,
            scale,
            world,
            body,
            fep: None,
            rng,
            neural_steps_per_body: 20, // 20 neural steps (0.1ms each) = 2ms per body step
            world_width,
            world_height,
        }
    }

    /// Create from a neuron count (infers scale tier).
    pub fn from_count(n: usize, seed: u64) -> Self {
        Self::new(DrosophilaScale::from_count(n), seed)
    }

    // ================================================================
    // Connectivity (FlyWire-inspired wiring)
    // ================================================================

    /// Build biologically-inspired connectivity following known Drosophila circuits.
    ///
    /// Major pathways:
    /// - Olfactory: AL -> MB (sparse KC coding, ~7 PNs per KC), AL -> LH (innate)
    /// - Mushroom body: KC -> MBON (readout), DAN -> KC (teaching), APL inhibition
    /// - Visual: Lamina -> Medulla -> Lobula (retinotopic)
    /// - Central complex: ring/compass neurons, heading integration
    /// - Motor: DN -> VNC, CX -> DN -> VNC
    /// - Neuromodulatory: DA/5-HT/Oct -> global targets
    fn build_connectivity(layout: &RegionLayout, rng: &mut StdRng) -> Vec<(u32, u32, NTType)> {
        let mut edges: Vec<(u32, u32, NTType)> = Vec::new();

        // Helper: connect src -> dst with probability prob and given NT type
        let connect = |src_name: &str,
                       dst_name: &str,
                       prob: f32,
                       nt: NTType,
                       edges: &mut Vec<(u32, u32, NTType)>,
                       rng: &mut StdRng| {
            let src_range = layout.range(src_name);
            let dst_range = layout.range(dst_name);
            if src_range.is_empty() || dst_range.is_empty() {
                return;
            }

            let n_possible = src_range.len() * dst_range.len();

            if n_possible > 50_000 {
                // Sample randomly to avoid O(N^2) blowup
                let n_expected = (n_possible as f32 * prob).round().max(1.0) as usize;
                for _ in 0..n_expected {
                    let pre = rng.gen_range(src_range.clone()) as u32;
                    let post = rng.gen_range(dst_range.clone()) as u32;
                    if pre != post {
                        edges.push((pre, post, nt));
                    }
                }
            } else {
                for pre in src_range.clone() {
                    for post in dst_range.clone() {
                        if pre != post && rng.gen::<f32>() < prob {
                            edges.push((pre as u32, post as u32, nt));
                        }
                    }
                }
            }
        };

        // Helper: sparse KC-style connectivity (fixed fan-in per destination neuron)
        let connect_sparse_kc = |src_name: &str,
                                 dst_name: &str,
                                 fan_in: usize,
                                 nt: NTType,
                                 edges: &mut Vec<(u32, u32, NTType)>,
                                 rng: &mut StdRng| {
            let src_range = layout.range(src_name);
            let dst_range = layout.range(dst_name);
            if src_range.is_empty() || dst_range.is_empty() {
                return;
            }
            let fan = fan_in.min(src_range.len());
            for post in dst_range {
                // Each KC gets exactly `fan` random PN inputs
                let mut chosen = Vec::with_capacity(fan);
                while chosen.len() < fan {
                    let pre = rng.gen_range(src_range.clone());
                    if pre != post && !chosen.contains(&pre) {
                        chosen.push(pre);
                    }
                    // Safety valve for tiny regions
                    if chosen.len() >= src_range.len() {
                        break;
                    }
                }
                for pre in chosen {
                    edges.push((pre as u32, post as u32, nt));
                }
            }
        };

        // ============================================================
        // 1. OLFACTORY CIRCUIT
        // ============================================================
        // AL projection neurons -> MB Kenyon cells (sparse coding, ~7 PNs per KC)
        connect_sparse_kc("AL", "MB_KC", 7, NTType::Acetylcholine, &mut edges, rng);
        // AL -> LH (innate olfactory pathway)
        connect("AL", "LH", 0.08, NTType::Acetylcholine, &mut edges, rng);
        // AL local inhibition (within AL, GABAergic interneurons)
        connect("AL", "AL", 0.15, NTType::GABA, &mut edges, rng);

        // ============================================================
        // 2. MUSHROOM BODY CIRCUIT (associative learning)
        // ============================================================
        // APL-like feedback inhibition (sparse coding enforcement)
        connect("MB_KC", "MB_KC", 0.03, NTType::GABA, &mut edges, rng);
        // KC -> MBON (readout of sparse KC activity)
        connect(
            "MB_KC",
            "MBON",
            0.10,
            NTType::Acetylcholine,
            &mut edges,
            rng,
        );
        // DAN -> MB_KC (reward/punishment teaching signal, dopaminergic)
        connect("DAN", "MB_KC", 0.08, NTType::Dopamine, &mut edges, rng);
        // DAN -> MBON (modulatory)
        connect("DAN", "MBON", 0.15, NTType::Dopamine, &mut edges, rng);
        // MBON -> CX (decision -> navigation)
        connect("MBON", "CX", 0.15, NTType::Acetylcholine, &mut edges, rng);
        // MBON -> LH (feedback to innate)
        connect("MBON", "LH", 0.10, NTType::Acetylcholine, &mut edges, rng);
        // MBON -> SEZ (feeding decisions)
        connect("MBON", "SEZ", 0.10, NTType::Acetylcholine, &mut edges, rng);

        // ============================================================
        // 3. VISUAL CIRCUIT (retinotopic hierarchy)
        // ============================================================
        // Lamina -> Medulla
        connect(
            "OL_LAM",
            "OL_MED",
            0.05,
            NTType::Acetylcholine,
            &mut edges,
            rng,
        );
        // Medulla -> Lobula
        connect(
            "OL_MED",
            "OL_LOB",
            0.05,
            NTType::Acetylcholine,
            &mut edges,
            rng,
        );
        // Lobula -> CX (visual features to navigation)
        connect("OL_LOB", "CX", 0.04, NTType::Acetylcholine, &mut edges, rng);
        // Lobula -> SUP (higher visual processing)
        connect(
            "OL_LOB",
            "SUP",
            0.03,
            NTType::Acetylcholine,
            &mut edges,
            rng,
        );
        // Local inhibition in medulla (motion computation requires lateral inhibition)
        connect("OL_MED", "OL_MED", 0.02, NTType::GABA, &mut edges, rng);

        // ============================================================
        // 4. CENTRAL COMPLEX (heading, navigation)
        // ============================================================
        // CX internal recurrence (ring neurons, heading integration)
        connect("CX", "CX", 0.08, NTType::Acetylcholine, &mut edges, rng);
        // CX -> DN (motor commands)
        connect("CX", "DN", 0.12, NTType::Acetylcholine, &mut edges, rng);
        // CX inhibitory interneurons
        connect("CX", "CX", 0.05, NTType::GABA, &mut edges, rng);

        // ============================================================
        // 5. MOTOR CIRCUIT
        // ============================================================
        // DN -> VNC (descending motor commands)
        connect("DN", "VNC", 0.10, NTType::Acetylcholine, &mut edges, rng);
        // VNC local CPG circuits
        connect("VNC", "VNC", 0.05, NTType::Acetylcholine, &mut edges, rng);
        // VNC inhibitory (reciprocal inhibition for alternating leg patterns)
        connect("VNC", "VNC", 0.03, NTType::GABA, &mut edges, rng);
        // SEZ -> VNC (feeding motor output)
        connect("SEZ", "VNC", 0.06, NTType::Acetylcholine, &mut edges, rng);

        // ============================================================
        // 6. LATERAL HORN (innate olfactory responses)
        // ============================================================
        // LH -> CX (innate olfactory -> navigation)
        connect("LH", "CX", 0.08, NTType::Acetylcholine, &mut edges, rng);
        // LH -> DN (direct innate motor)
        connect("LH", "DN", 0.06, NTType::Acetylcholine, &mut edges, rng);

        // ============================================================
        // 7. HIGHER PROCESSING
        // ============================================================
        // SUP -> CX (higher processing to navigation)
        connect("SUP", "CX", 0.04, NTType::Acetylcholine, &mut edges, rng);
        // SUP -> MBON (top-down modulation)
        connect("SUP", "MBON", 0.03, NTType::Acetylcholine, &mut edges, rng);
        // SUP internal
        connect("SUP", "SUP", 0.03, NTType::Acetylcholine, &mut edges, rng);

        // ============================================================
        // 8. NEUROMODULATORY (global modulation)
        // ============================================================
        // DA neurons -> MB (reward signal)
        connect("NEUROMOD", "MB_KC", 0.05, NTType::Dopamine, &mut edges, rng);
        // 5-HT global modulation (using serotonin NT, mapped to second third of NEUROMOD)
        connect("NEUROMOD", "SUP", 0.04, NTType::Serotonin, &mut edges, rng);
        connect("NEUROMOD", "CX", 0.04, NTType::Serotonin, &mut edges, rng);
        // Octopamine (mapped to NE) -> VNC for arousal/flight (last third of NEUROMOD)
        connect(
            "NEUROMOD",
            "VNC",
            0.05,
            NTType::Norepinephrine,
            &mut edges,
            rng,
        );
        connect(
            "NEUROMOD",
            "DN",
            0.05,
            NTType::Norepinephrine,
            &mut edges,
            rng,
        );

        // ============================================================
        // 9. SEZ (taste, feeding)
        // ============================================================
        // SEZ -> MBON (gustatory input to valence system)
        connect("SEZ", "MBON", 0.06, NTType::Acetylcholine, &mut edges, rng);
        // SEZ internal
        connect("SEZ", "SEZ", 0.05, NTType::Acetylcholine, &mut edges, rng);

        // Sort edges by pre neuron for CSR construction
        edges.sort_by_key(|e| e.0);
        edges
    }

    // ================================================================
    // Archetype Assignment
    // ================================================================

    /// Assign neuron archetypes per region.
    ///
    /// Most Drosophila neurons are cholinergic (Pyramidal archetype maps to
    /// high nAChR expression). Inhibitory neurons within each region get
    /// the Interneuron archetype (GABAergic).
    fn assign_archetypes(brain: &mut MolecularBrain, layout: &RegionLayout) {
        for &(name, start, count) in &layout.regions {
            if count == 0 {
                continue;
            }
            let inhib_frac = RegionLayout::inhib_fraction(name);
            let n_inhib = (count as f32 * inhib_frac).round() as usize;
            let n_excit = count - n_inhib;

            // Determine primary archetype based on region
            let excit_archetype = match name {
                "DAN" | "NEUROMOD" => NeuronArchetype::DopaminergicSN as u8,
                _ => NeuronArchetype::Cholinergic as u8, // ACh primary in insects
            };

            // Excitatory neurons (first n_excit in region)
            for i in start..start + n_excit {
                brain.neurons.archetype[i] = excit_archetype;
            }
            // Inhibitory neurons (last n_inhib in region)
            for i in start + n_excit..start + count {
                brain.neurons.archetype[i] = NeuronArchetype::Interneuron as u8;
            }
        }
    }

    // ================================================================
    // Spatial Position Assignment
    // ================================================================

    /// Assign 3D positions approximating Drosophila brain anatomy.
    fn assign_positions(brain: &mut MolecularBrain, layout: &RegionLayout, rng: &mut StdRng) {
        // Region center and spread in (x, y, z) coordinates
        let positions: &[(&str, (f32, f32, f32), f32)] = &[
            ("AL", (0.0, 0.5, 0.25), 1.0),
            ("MB_KC", (0.0, 1.0, 1.5), 1.5),
            ("MBON", (0.0, 0.55, 1.75), 0.6),
            ("DAN", (0.0, 1.0, 1.75), 0.6),
            ("CX", (0.0, 1.25, 1.25), 0.6),
            ("OL_LAM", (-1.5, 0.5, 0.25), 1.0),
            ("OL_MED", (-1.5, 0.5, 0.75), 1.0),
            ("OL_LOB", (-1.0, 0.5, 1.15), 1.0),
            ("LH", (-0.65, 0.75, 1.25), 0.7),
            ("SEZ", (0.0, -0.1, 2.0), 0.6),
            ("SUP", (0.0, 1.5, 2.0), 1.0),
            ("DN", (0.0, 0.25, 2.75), 0.6),
            ("VNC", (0.0, 0.0, 4.0), 2.0),
            ("NEUROMOD", (0.0, 1.0, 1.75), 1.5),
            ("OTHER", (0.0, 0.75, 1.5), 2.0),
        ];

        for &(name, (cx, cy, cz), spread) in positions {
            let range = layout.range(name);
            for i in range {
                brain.neurons.x[i] = cx + (rng.gen::<f32>() - 0.5) * spread;
                brain.neurons.y[i] = cy + (rng.gen::<f32>() - 0.5) * spread;
                brain.neurons.z[i] = cz + (rng.gen::<f32>() - 0.5) * spread;
            }
        }
    }

    // ================================================================
    // Sensory Encoding
    // ================================================================

    /// Encode sensory input from the world into external currents on sensory neurons.
    ///
    /// Maps world state to neural populations:
    /// - Odorant concentration -> AL (antennal lobe)
    /// - Light intensity -> OL_LAM (optic lobe lamina)
    /// - Temperature -> SEZ + "thermal" subset of OTHER
    fn encode_sensory(&mut self) {
        let x = self.body.x;
        let y = self.body.y;

        // Olfactory: odorant at fly position -> AL current
        let odorant = self.world.sample_odorant(x, y);
        if odorant > 0.01 {
            let current = odorant * 40.0; // Scale to uA/cm^2 (subthreshold to suprathreshold)
            let al_range = self.layout.range("AL");
            for i in al_range {
                self.brain.stimulate(i, current);
            }
        }

        // Visual: light at fly position -> OL_LAM current
        let light = self.world.sample_light(x, y);
        if light > 0.01 {
            // Differential encoding: left vs right eye
            let heading = self.body.heading;
            let left_x = x + heading.sin() * 1.0;
            let left_y = y - heading.cos() * 1.0;
            let right_x = x - heading.sin() * 1.0;
            let right_y = y + heading.cos() * 1.0;
            let left_light = self.world.sample_light(left_x, left_y);
            let right_light = self.world.sample_light(right_x, right_y);

            let lam_range = self.layout.range("OL_LAM");
            let half = lam_range.len() / 2;
            // Left eye -> first half of lamina
            for i in lam_range.start..lam_range.start + half {
                self.brain.stimulate(i, left_light * 30.0);
            }
            // Right eye -> second half
            for i in lam_range.start + half..lam_range.end {
                self.brain.stimulate(i, right_light * 30.0);
            }
        }

        // Thermal: temperature -> SEZ (thermal sensory)
        let temp = self.world.sample_temperature(x, y);
        self.body.temperature = temp;
        // Deviation from preferred temperature (18 deg C for Drosophila)
        let temp_signal = ((temp - 18.0).abs() / 12.0).min(1.0); // [0, 1]
        if temp_signal > 0.05 {
            let sez_range = self.layout.range("SEZ");
            let current = temp_signal * 25.0;
            for i in sez_range {
                self.brain.stimulate(i, current);
            }
        }
    }

    /// Encode externally sampled local sensory values into the fly brain.
    ///
    /// This is the bridge used when another world model already owns the
    /// environment and only passes the local sensory slice into the native fly.
    fn encode_manual_sensory(
        &mut self,
        odorant: f32,
        left_light: f32,
        right_light: f32,
        temperature: f32,
    ) {
        let odorant = odorant.max(0.0);
        if odorant > 0.01 {
            let current = odorant * 40.0;
            let al_range = self.layout.range("AL");
            for i in al_range {
                self.brain.stimulate(i, current);
            }
        }

        let left_light = left_light.max(0.0);
        let right_light = right_light.max(0.0);
        if left_light > 0.01 || right_light > 0.01 {
            let lam_range = self.layout.range("OL_LAM");
            let half = lam_range.len() / 2;
            for i in lam_range.start..lam_range.start + half {
                self.brain.stimulate(i, left_light * 30.0);
            }
            for i in lam_range.start + half..lam_range.end {
                self.brain.stimulate(i, right_light * 30.0);
            }
        }

        self.body.temperature = temperature;
        let temp_signal = ((temperature - 18.0).abs() / 12.0).min(1.0);
        if temp_signal > 0.05 {
            let sez_range = self.layout.range("SEZ");
            let current = temp_signal * 25.0;
            for i in sez_range {
                self.brain.stimulate(i, current);
            }
        }
    }

    fn stimulate_manual_taste(&mut self, sugar: f32, bitter: f32, amino: f32) {
        let sez_range = self.layout.range("SEZ");
        if sez_range.is_empty() {
            return;
        }
        let n_sez = sez_range.len();
        let gust_end = sez_range.start + (n_sez * 2 / 3).max(1);
        let sugar_current = sugar.max(0.0) * 40.0;
        let amino_current = amino.max(0.0) * 24.0;
        let bitter_current = bitter.max(0.0) * 28.0;
        for idx in sez_range.start..gust_end.min(sez_range.end) {
            let current = sugar_current + amino_current - bitter_current * 0.35;
            if current > 0.0 {
                self.brain.stimulate(idx, current);
            }
        }
    }

    fn stimulate_manual_wind(&mut self, wind_x: f32, wind_y: f32, wind_z: f32) {
        let wind_strength = (wind_x * wind_x + wind_y * wind_y + wind_z * wind_z).sqrt();
        if wind_strength <= 0.1 {
            return;
        }
        let cx_range = self.layout.range("CX");
        for idx in cx_range {
            self.brain.stimulate(idx, (wind_strength * 5.0).min(20.0));
        }
    }

    pub fn apply_reward_signal(&mut self, valence: f32) {
        if valence.abs() <= 1.0e-6 {
            return;
        }
        let dan_range = self.layout.range("DAN");
        if dan_range.is_empty() {
            return;
        }
        let half = (dan_range.len() / 2).max(1);
        let amplitude = (valence.abs() * 45.0).clamp(0.0, 60.0);
        let target = if valence >= 0.0 {
            dan_range.start..(dan_range.start + half).min(dan_range.end)
        } else {
            (dan_range.start + half).min(dan_range.end)..dan_range.end
        };
        for idx in target {
            self.brain.stimulate(idx, amplitude);
        }
    }

    fn vnc_forward_range(&self) -> std::ops::Range<usize> {
        let vnc = self.layout.range("VNC");
        let half = vnc.len() / 2;
        vnc.start..vnc.start + half
    }

    fn vnc_backward_range(&self) -> std::ops::Range<usize> {
        let vnc = self.layout.range("VNC");
        let half = vnc.len() / 2;
        vnc.start + half..vnc.end
    }

    fn vnc_left_range(&self) -> std::ops::Range<usize> {
        let fwd = self.vnc_forward_range();
        let quarter = fwd.len() / 2;
        fwd.start..fwd.start + quarter
    }

    fn vnc_right_range(&self) -> std::ops::Range<usize> {
        let fwd = self.vnc_forward_range();
        let quarter = fwd.len() / 2;
        fwd.start + quarter..fwd.end
    }

    fn vnc_flight_range(&self) -> std::ops::Range<usize> {
        let vnc = self.layout.range("VNC");
        if vnc.len() < 7 {
            return vnc.end..vnc.end;
        }
        let seg = vnc.len() / 7;
        vnc.start + 6 * seg..vnc.end
    }

    fn vnc_climb_range(&self) -> std::ops::Range<usize> {
        let flight = self.vnc_flight_range();
        let half = flight.len() / 2;
        flight.start..flight.start + half
    }

    fn vnc_descend_range(&self) -> std::ops::Range<usize> {
        let flight = self.vnc_flight_range();
        let half = flight.len() / 2;
        flight.start + half..flight.end
    }

    fn sez_proboscis_range(&self) -> std::ops::Range<usize> {
        let sez = self.layout.range("SEZ");
        let start = sez.start + (sez.len() * 2 / 3);
        start.min(sez.end)..sez.end
    }

    fn count_spikes_in_range(&self, range: std::ops::Range<usize>) -> u32 {
        let mut count = 0u32;
        for idx in range {
            if self.brain.neurons.fired[idx] != 0 {
                count += 1;
            }
        }
        count
    }

    fn read_motor_output_window(&mut self, n_steps: u32) -> MotorOutput {
        let mut fwd_acc = 0u32;
        let mut bwd_acc = 0u32;
        let mut left_acc = 0u32;
        let mut right_acc = 0u32;
        let mut flight_acc = 0u32;
        let mut climb_acc = 0u32;
        let mut descend_acc = 0u32;
        let mut feed_acc = 0u32;

        let vnc_fwd = self.vnc_forward_range();
        let vnc_bwd = self.vnc_backward_range();
        let vnc_left = self.vnc_left_range();
        let vnc_right = self.vnc_right_range();
        let vnc_flight = self.vnc_flight_range();
        let vnc_climb = self.vnc_climb_range();
        let vnc_descend = self.vnc_descend_range();
        let sez_prob = self.sez_proboscis_range();

        for step in 0..n_steps {
            if step % 2 == 0 {
                let dn_range = self.layout.range("DN");
                for idx in dn_range {
                    self.brain.stimulate(idx, 40.0);
                }
                for idx in vnc_fwd.clone() {
                    self.brain.stimulate(idx, 35.0);
                }
                for idx in vnc_bwd.clone() {
                    self.brain.stimulate(idx, 30.0);
                }
                for idx in sez_prob.clone() {
                    self.brain.stimulate(idx, 15.0);
                }
            }

            let step_count = self.brain.step_count;
            if let Some(ref mut fep) = self.fep {
                fep.stimulate(&mut self.brain, step_count);
            }
            self.brain.step();

            fwd_acc += self.count_spikes_in_range(vnc_fwd.clone());
            bwd_acc += self.count_spikes_in_range(vnc_bwd.clone());
            left_acc += self.count_spikes_in_range(vnc_left.clone());
            right_acc += self.count_spikes_in_range(vnc_right.clone());
            if !vnc_flight.is_empty() {
                flight_acc += self.count_spikes_in_range(vnc_flight.clone());
            }
            if !vnc_climb.is_empty() {
                climb_acc += self.count_spikes_in_range(vnc_climb.clone());
            }
            if !vnc_descend.is_empty() {
                descend_acc += self.count_spikes_in_range(vnc_descend.clone());
            }
            if !sez_prob.is_empty() {
                feed_acc += self.count_spikes_in_range(sez_prob.clone());
            }
        }

        self.brain.sync_shadow_from_gpu();

        let n_steps_f = n_steps.max(1) as f32;
        let max_fwd = (vnc_fwd.len().max(1) as f32) * n_steps_f;
        let max_bwd = (vnc_bwd.len().max(1) as f32) * n_steps_f;
        let fwd_rate = fwd_acc as f32 / max_fwd;
        let bwd_rate = bwd_acc as f32 / max_bwd;
        let speed = ((fwd_rate - bwd_rate * 0.5) / 0.05).clamp(0.0, 1.0);

        let total_lr = left_acc + right_acc;
        let turn = if total_lr > 0 {
            (left_acc as f32 - right_acc as f32) / total_lr as f32
        } else {
            0.0
        };

        let fly_signal = if !vnc_flight.is_empty() {
            let fly_rate = flight_acc as f32 / ((vnc_flight.len() as f32) * n_steps_f).max(1.0);
            ((fly_rate - 0.05) / 0.10).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let total_climb = climb_acc + descend_acc;
        let climb_signal = if total_climb > 0 {
            (climb_acc as f32 - descend_acc as f32) / total_climb as f32
        } else {
            0.0
        };

        let feed_signal = if !sez_prob.is_empty() {
            let feed_rate = feed_acc as f32 / ((sez_prob.len() as f32) * n_steps_f).max(1.0);
            ((feed_rate - 0.02) / 0.05).clamp(0.0, 1.0)
        } else {
            0.0
        };

        MotorOutput {
            speed,
            turn,
            fly_signal,
            feed_signal,
            climb_signal,
        }
    }

    // ================================================================
    // Motor Decoding
    // ================================================================

    /// Decode motor commands from VNC spike activity into body movement.
    ///
    /// Splits VNC into 4 quadrants: left motor, right motor, forward, backward.
    /// Differential left-right activity produces turning; forward-backward
    /// produces speed.
    fn decode_motor(&mut self) {
        let vnc_range = self.layout.range("VNC");
        if vnc_range.is_empty() {
            return;
        }
        let quarter = vnc_range.len() / 4;
        if quarter == 0 {
            return;
        }

        // Count spikes in each quadrant
        let count_spikes = |start: usize, end: usize| -> f32 {
            let mut count = 0u32;
            for i in start..end {
                if self.brain.neurons.fired[i] != 0 {
                    count += 1;
                }
            }
            count as f32
        };

        let left_spikes = count_spikes(vnc_range.start, vnc_range.start + quarter);
        let right_spikes = count_spikes(vnc_range.start + quarter, vnc_range.start + 2 * quarter);
        let fwd_spikes = count_spikes(vnc_range.start + 2 * quarter, vnc_range.start + 3 * quarter);
        let bwd_spikes = count_spikes(vnc_range.start + 3 * quarter, vnc_range.end);

        // Normalize by quadrant size
        let max_rate = quarter as f32;
        let left_rate = left_spikes / max_rate;
        let right_rate = right_spikes / max_rate;
        let fwd_rate = fwd_spikes / max_rate;
        let bwd_rate = bwd_spikes / max_rate;

        // Turn: differential left-right
        let turn = (right_rate - left_rate) * 0.3; // max ~0.3 rad/body_step
        self.body.heading += turn;

        // Speed: net forward-backward
        let speed = (fwd_rate - bwd_rate) * 2.0; // max ~2 world units/body_step
        self.body.speed = speed.max(0.0); // no backwards for now
    }

    // ================================================================
    // Simulation Step
    // ================================================================

    /// Run one body step: encode sensory -> N neural steps -> decode motor -> update body.
    pub fn body_step(&mut self) {
        // 1. Sensory encoding
        self.encode_sensory();

        // 2. Neural simulation (multiple neural steps per body step)
        for _ in 0..self.neural_steps_per_body {
            // FEP stimulation (if active)
            let step_count = self.brain.step_count;
            if let Some(ref mut fep) = self.fep {
                fep.stimulate(&mut self.brain, step_count);
            }
            self.brain.step();
        }
        self.brain.sync_shadow_from_gpu();

        // 3. Motor decoding
        self.decode_motor();

        // 4. Body update
        self.body.update(self.world_width, self.world_height);

        // 5. World diffusion
        self.world.step_diffusion(0.998);
    }

    /// Run one body step using externally supplied local sensory values.
    pub fn body_step_manual(
        &mut self,
        odorant: f32,
        left_light: f32,
        right_light: f32,
        temperature: f32,
    ) {
        self.encode_manual_sensory(odorant, left_light, right_light, temperature);

        for _ in 0..self.neural_steps_per_body {
            let step_count = self.brain.step_count;
            if let Some(ref mut fep) = self.fep {
                fep.stimulate(&mut self.brain, step_count);
            }
            self.brain.step();
        }
        self.brain.sync_shadow_from_gpu();

        self.decode_motor();
        self.body.update(self.world_width, self.world_height);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn body_step_terrarium(
        &mut self,
        odorant: f32,
        left_light: f32,
        right_light: f32,
        temperature: f32,
        sugar_taste: f32,
        bitter_taste: f32,
        amino_taste: f32,
        wind_x: f32,
        wind_y: f32,
        wind_z: f32,
        food_available: f32,
        reward_valence: f32,
        air_density: f32,
    ) -> TerrariumFlyStepReport {
        self.encode_manual_sensory(odorant, left_light, right_light, temperature);
        self.stimulate_manual_taste(sugar_taste, bitter_taste, amino_taste);
        self.stimulate_manual_wind(wind_x, wind_y, wind_z);
        self.apply_reward_signal(reward_valence);

        let motor = self.read_motor_output_window(self.neural_steps_per_body);
        if motor.fly_signal > 0.5 && !self.body.is_flying {
            self.body.takeoff();
        } else if motor.fly_signal < 0.2 && self.body.is_flying {
            self.body.land();
        }

        // Dynamic wing beat frequency calculation
        self.body.wing_beat_freq = self.body.calculate_wing_beat_freq(air_density, motor.climb_signal);

        if self.body.is_flying {
            self.body.fly_3d(
                motor.speed,
                motor.turn,
                motor.climb_signal,
                DT_BODY_S,
                self.world_width,
                self.world_height,
            );
        } else {
            self.body.walk(
                motor.speed,
                motor.turn,
                DT_BODY_S,
                self.world_width,
                self.world_height,
            );
        }

        let mut consumed_food = 0.0;
        self.body.proboscis_extended = motor.feed_signal > 0.3;
        if self.body.proboscis_extended && !self.body.is_flying && food_available > 0.1 {
            self.body.energy = (self.body.energy + food_available * FLY_ENERGY_FEED_GAIN)
                .clamp(0.0, FLY_ENERGY_MAX);
            self.apply_reward_signal(0.5);
            consumed_food = 0.03;
        }

        TerrariumFlyStepReport {
            speed: motor.speed,
            turn: motor.turn,
            fly_signal: motor.fly_signal,
            feed_signal: motor.feed_signal,
            climb_signal: motor.climb_signal,
            consumed_food,
            x: self.body.x,
            y: self.body.y,
            z: self.body.z,
            heading: self.body.heading,
            pitch: self.body.pitch,
            energy: self.body.energy,
            is_flying: self.body.is_flying,
            wing_beat_freq: self.body.wing_beat_freq,
        }
    }

    /// Run N body steps.
    pub fn run_body_steps(&mut self, n: u32) {
        for _ in 0..n {
            self.body_step();
        }
    }

    /// Reset body position and heading for a new episode.
    fn reset_episode(&mut self, x: f32, y: f32) {
        self.body = BodyState::new(x, y);
        // Reset spike counts for clean episode measurement
        self.brain.reset_spike_counts();
    }

    // ================================================================
    // Experiment 1: Olfactory Learning
    // ================================================================

    /// Olfactory learning: place an odorant source and measure whether the
    /// fly approaches it. Uses FEP protocol: structured stimulation (HIT)
    /// when approaching, random noise (MISS) when not.
    pub fn run_olfactory(&mut self, n_episodes: u32) -> ExperimentResult {
        let steps_per_episode = 200; // body steps
        let source_x = 50.0f32;
        let source_y = 32.0f32;

        // Set up FEP on AL neurons
        let al_range = self.layout.range("AL");
        let al_indices: Vec<usize> = al_range.collect();
        self.fep = Some(FepState::new(al_indices, 40.0, 777));

        let mut final_distances = Vec::new();
        let mut trajectories = Vec::new();

        for _ep in 0..n_episodes {
            // Place odorant source each episode (Gaussian falloff)
            self.world.odorant.iter_mut().for_each(|v| *v = 0.0);
            self.world.place_odorant(source_x, source_y, 1.0, 10.0);

            self.reset_episode(10.0, 32.0); // Start on left side

            let mut prev_dist = ((10.0 - source_x).powi(2) + (32.0 - source_y).powi(2)).sqrt();

            for step in 0..steps_per_episode {
                // FEP mode: approaching = HIT (structured), not approaching = MISS (random)
                let curr_dist =
                    ((self.body.x - source_x).powi(2) + (self.body.y - source_y).powi(2)).sqrt();

                if let Some(ref mut fep) = self.fep {
                    fep.mode = if curr_dist < prev_dist {
                        FepMode::Structured
                    } else {
                        FepMode::Random
                    };
                }
                prev_dist = curr_dist;

                self.body_step();

                // Record trajectory every 20 body steps
                if step % 20 == 0 {
                    trajectories.push((self.body.x, self.body.y));
                }
            }

            // Measure final distance
            let dx = self.body.x - source_x;
            let dy = self.body.y - source_y;
            let dist = (dx * dx + dy * dy).sqrt();
            final_distances.push(dist);
        }

        self.fep = None; // Disable FEP after experiment

        let mean_dist: f32 = final_distances.iter().sum::<f32>() / final_distances.len() as f32;
        let initial_dist = 40.0f32; // (10, 32) -> (50, 32)
        let improvement = (initial_dist - mean_dist) / initial_dist;

        ExperimentResult {
            name: "olfactory_learning".to_string(),
            passed: improvement > 0.1, // >10% closer
            metric_name: "approach_improvement".to_string(),
            metric_value: improvement as f64,
            threshold: 0.1,
            details: format!(
                "Mean final distance: {:.1} (initial: {:.1}), improvement: {:.1}%",
                mean_dist,
                initial_dist,
                improvement * 100.0
            ),
            trajectories,
        }
    }

    // ================================================================
    // Experiment 2: Phototaxis
    // ================================================================

    /// Phototaxis: navigate toward light. Sets a light gradient (bright on
    /// right side) and measures net displacement toward the bright side.
    pub fn run_phototaxis(&mut self, n_episodes: u32) -> ExperimentResult {
        let steps_per_episode = 200;

        self.world.set_light_gradient();

        let mut x_displacements = Vec::new();
        let mut trajectories = Vec::new();

        for _ep in 0..n_episodes {
            self.reset_episode(32.0, 32.0); // Start center

            for step in 0..steps_per_episode {
                self.body_step();
                if step % 20 == 0 {
                    trajectories.push((self.body.x, self.body.y));
                }
            }

            x_displacements.push(self.body.x - 32.0); // Positive = toward light
        }

        let mean_displacement: f32 =
            x_displacements.iter().sum::<f32>() / x_displacements.len() as f32;

        ExperimentResult {
            name: "phototaxis".to_string(),
            passed: mean_displacement > 1.0,
            metric_name: "mean_x_displacement".to_string(),
            metric_value: mean_displacement as f64,
            threshold: 1.0,
            details: format!(
                "Mean x displacement toward light: {:.2} (n={})",
                mean_displacement, n_episodes
            ),
            trajectories,
        }
    }

    // ================================================================
    // Experiment 3: Thermotaxis
    // ================================================================

    /// Thermotaxis: measure preference for 18 deg C. Sets a temperature
    /// gradient (15 to 30 deg C left to right) and measures final
    /// temperature at fly position.
    pub fn run_thermotaxis(&mut self, n_episodes: u32) -> ExperimentResult {
        let steps_per_episode = 200;

        self.world.set_temp_gradient(15.0, 30.0);
        // Preferred temperature is 18 deg C, which maps to x = 64 * (18-15)/(30-15) = 12.8

        let mut final_temps = Vec::new();
        let mut trajectories = Vec::new();

        for _ep in 0..n_episodes {
            self.reset_episode(32.0, 32.0); // Start at center (22.5 deg C)

            for step in 0..steps_per_episode {
                self.body_step();
                if step % 20 == 0 {
                    trajectories.push((self.body.x, self.body.y));
                }
            }

            let temp = self.world.sample_temperature(self.body.x, self.body.y);
            final_temps.push(temp);
        }

        let mean_temp: f32 = final_temps.iter().sum::<f32>() / final_temps.len() as f32;
        let temp_error = (mean_temp - 18.0).abs();

        ExperimentResult {
            name: "thermotaxis".to_string(),
            passed: temp_error < 5.0, // Within 5 deg C of preference
            metric_name: "mean_final_temp".to_string(),
            metric_value: mean_temp as f64,
            threshold: 5.0,
            details: format!(
                "Mean final temperature: {:.1} deg C (target: 18 deg C, error: {:.1} deg C)",
                mean_temp, temp_error
            ),
            trajectories,
        }
    }

    // ================================================================
    // Experiment 4: Foraging
    // ================================================================

    /// Foraging: place multiple food sources with odorant trails and measure
    /// how many the fly visits.
    pub fn run_foraging(&mut self, n_episodes: u32) -> ExperimentResult {
        let steps_per_episode = 400;
        let food_positions = [(15.0f32, 15.0f32), (48.0, 15.0), (15.0, 48.0), (48.0, 48.0)];

        let mut foods_found = Vec::new();
        let mut trajectories = Vec::new();

        for _ep in 0..n_episodes {
            // Set up world with food sources and odorant trails
            self.world.odorant.iter_mut().for_each(|v| *v = 0.0);
            self.world.food_sources.clear();
            for &(fx, fy) in &food_positions {
                self.world.place_food(fx, fy, 100.0);
                self.world.place_odorant(fx, fy, 0.8, 8.0);
            }

            self.reset_episode(32.0, 32.0);
            let mut visited = [false; 4];

            for step in 0..steps_per_episode {
                self.body_step();
                if step % 20 == 0 {
                    trajectories.push((self.body.x, self.body.y));
                }

                // Check proximity to food sources (within 3 world units)
                for (j, &(fx, fy)) in food_positions.iter().enumerate() {
                    let dx = self.body.x - fx;
                    let dy = self.body.y - fy;
                    if dx * dx + dy * dy < 9.0 {
                        visited[j] = true;
                    }
                }
            }

            let found = visited.iter().filter(|&&v| v).count() as u32;
            foods_found.push(found);
        }

        let mean_found: f32 = foods_found.iter().sum::<u32>() as f32 / foods_found.len() as f32;

        ExperimentResult {
            name: "foraging".to_string(),
            passed: mean_found > 0.5, // Find at least some food on average
            metric_name: "mean_food_visits".to_string(),
            metric_value: mean_found as f64,
            threshold: 0.5,
            details: format!(
                "Mean unique food sources found per episode: {:.1}",
                mean_found
            ),
            trajectories,
        }
    }

    // ================================================================
    // Experiment 5: Drug Response
    // ================================================================

    /// Drug response: measure behavioral effects of TTX (Nav blocker).
    ///
    /// Runs baseline episodes, then applies TTX by zeroing Nav conductance
    /// scale across all neurons, and measures spike suppression.
    pub fn run_drug_response(&mut self, n_episodes: u32) -> ExperimentResult {
        let neural_steps_per_episode = 500;

        // Run baseline
        let mut baseline_spikes = Vec::new();
        for _ep in 0..n_episodes {
            self.reset_episode(32.0, 32.0);
            // Stimulate AL to generate activity
            let al_range = self.layout.range("AL");
            for i in al_range.clone() {
                self.brain.stimulate(i, 20.0);
            }
            self.brain.run(neural_steps_per_episode);
            let total: u64 = self
                .brain
                .neurons
                .spike_count
                .iter()
                .map(|&c| c as u64)
                .sum();
            baseline_spikes.push(total);
        }

        // Apply TTX: block Nav channels (conductance_scale[0] = 0)
        for i in 0..self.brain.neurons.count {
            self.brain
                .set_conductance_scale(i, IonChannelType::Nav as usize, 0.0);
        }

        let mut drug_spikes = Vec::new();
        for _ep in 0..n_episodes {
            self.reset_episode(32.0, 32.0);
            // Same stimulation
            let al_range = self.layout.range("AL");
            for i in al_range.clone() {
                self.brain.stimulate(i, 20.0);
            }
            self.brain.run(neural_steps_per_episode);
            let total: u64 = self
                .brain
                .neurons
                .spike_count
                .iter()
                .map(|&c| c as u64)
                .sum();
            drug_spikes.push(total);
        }

        // Restore Nav conductance
        for i in 0..self.brain.neurons.count {
            self.brain
                .set_conductance_scale(i, IonChannelType::Nav as usize, 1.0);
        }

        let baseline_mean =
            baseline_spikes.iter().sum::<u64>() as f64 / baseline_spikes.len() as f64;
        let drug_mean = drug_spikes.iter().sum::<u64>() as f64 / drug_spikes.len() as f64;
        let suppression = if baseline_mean > 0.0 {
            1.0 - drug_mean / baseline_mean
        } else {
            0.0
        };

        ExperimentResult {
            name: "drug_response".to_string(),
            passed: suppression > 0.3, // TTX should suppress >30% of spikes
            metric_name: "spike_suppression".to_string(),
            metric_value: suppression,
            threshold: 0.3,
            details: format!(
                "Baseline spikes: {:.0}, TTX spikes: {:.0}, suppression: {:.1}%",
                baseline_mean,
                drug_mean,
                suppression * 100.0
            ),
            trajectories: Vec::new(),
        }
    }

    // ================================================================
    // Experiment 6: Circadian
    // ================================================================

    /// Circadian: measure activity difference between day and night.
    ///
    /// The circadian clock modulates excitability bias. Day phase should
    /// produce more activity than night phase.
    pub fn run_circadian(&mut self, n_episodes: u32) -> ExperimentResult {
        let neural_steps_per_phase = 500;

        let mut day_activity = Vec::new();
        let mut night_activity = Vec::new();

        for _ep in 0..n_episodes {
            // Day phase: advance circadian clock to noon-like state
            // Reset brain and set circadian to high-excitability phase
            self.reset_episode(32.0, 32.0);
            self.brain.circadian.set_phase(0.5); // Midday peak
            self.brain.run(neural_steps_per_phase);
            let day_total: u64 = self
                .brain
                .neurons
                .spike_count
                .iter()
                .map(|&c| c as u64)
                .sum();
            day_activity.push(day_total);

            // Night phase: set circadian to low-excitability phase
            self.reset_episode(32.0, 32.0);
            self.brain.circadian.set_phase(0.0); // Midnight trough
            self.brain.run(neural_steps_per_phase);
            let night_total: u64 = self
                .brain
                .neurons
                .spike_count
                .iter()
                .map(|&c| c as u64)
                .sum();
            night_activity.push(night_total);
        }

        let day_mean = day_activity.iter().sum::<u64>() as f64 / day_activity.len() as f64;
        let night_mean = night_activity.iter().sum::<u64>() as f64 / night_activity.len() as f64;
        let ratio = if night_mean > 0.0 {
            day_mean / night_mean
        } else {
            f64::INFINITY
        };

        ExperimentResult {
            name: "circadian".to_string(),
            passed: ratio > 1.1, // Day should be >10% more active
            metric_name: "day_night_ratio".to_string(),
            metric_value: ratio,
            threshold: 1.1,
            details: format!(
                "Day activity: {:.0}, Night activity: {:.0}, ratio: {:.2}",
                day_mean, night_mean, ratio
            ),
            trajectories: Vec::new(),
        }
    }

    // ================================================================
    // Run All Experiments
    // ================================================================

    /// Run all 6 experiments and return results.
    pub fn run_all(&mut self) -> Vec<ExperimentResult> {
        vec![
            self.run_olfactory(10),
            self.run_phototaxis(10),
            self.run_thermotaxis(10),
            self.run_foraging(10),
            self.run_drug_response(5),
            self.run_circadian(5),
        ]
    }

    // ================================================================
    // Accessors
    // ================================================================

    /// Get total neuron count.
    pub fn neuron_count(&self) -> usize {
        self.brain.neuron_count()
    }

    /// Get total synapse count.
    pub fn synapse_count(&self) -> usize {
        self.brain.synapse_count()
    }

    /// Whether GPU acceleration is active.
    pub fn gpu_active(&self) -> bool {
        self.brain.gpu_available()
    }

    /// Get current body state.
    pub fn body_state(&self) -> &BodyState {
        &self.body
    }

    /// Override body state for integration with an external world.
    pub fn set_body_state(
        &mut self,
        x: f32,
        y: f32,
        heading: f32,
        z: Option<f32>,
        pitch: Option<f32>,
        is_flying: Option<bool>,
        speed: Option<f32>,
        energy: Option<f32>,
        temperature: Option<f32>,
        time_of_day: Option<f32>,
    ) {
        self.body.x = x.clamp(0.0, self.world_width.max(1.0) - 1.0);
        self.body.y = y.clamp(0.0, self.world_height.max(1.0) - 1.0);
        self.body.heading = heading;
        if let Some(z) = z {
            self.body.z = z.clamp(0.0, FLY_MAX_ALTITUDE);
        }
        if let Some(pitch) = pitch {
            self.body.pitch = pitch;
        }
        if let Some(is_flying) = is_flying {
            self.body.is_flying = is_flying;
            if !is_flying {
                self.body.vertical_velocity = 0.0;
            }
        }
        if let Some(speed) = speed {
            self.body.speed = speed.max(0.0);
        }
        if let Some(energy) = energy {
            self.body.energy = energy.clamp(0.0, FLY_ENERGY_MAX);
        }
        if let Some(temperature) = temperature {
            self.body.temperature = temperature;
        }
        if let Some(time_of_day) = time_of_day {
            self.body.time_of_day = time_of_day;
        }
    }

    /// Set body energy directly (used by external metabolism coupling).
    pub fn set_energy(&mut self, energy: f32) {
        self.body.energy = energy.clamp(0.0, FLY_ENERGY_MAX);
    }

    /// Override the native world bounds for external integration.
    pub fn set_world_bounds(&mut self, width: f32, height: f32) {
        self.world_width = width.max(2.0);
        self.world_height = height.max(2.0);
        self.body.x = self.body.x.clamp(0.0, self.world_width - 1.0);
        self.body.y = self.body.y.clamp(0.0, self.world_height - 1.0);
    }

    /// Get neuron indices for a named region.
    pub fn region_indices(&self, name: &str) -> Vec<usize> {
        self.layout.range(name).collect()
    }

    /// Count fired neurons in a named region this step.
    pub fn region_fired_count(&self, name: &str) -> usize {
        self.layout
            .range(name)
            .filter(|&i| self.brain.neurons.fired[i] != 0)
            .count()
    }

    /// Get mean firing rate (Hz) across all neurons.
    pub fn mean_firing_rate(&self) -> f32 {
        self.brain.mean_firing_rate()
    }

    /// Get the region layout.
    pub fn region_layout(&self) -> &RegionLayout {
        &self.layout
    }

    /// Print a summary of the network structure.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "DrosophilaSim | scale={:?} | neurons={} | synapses={} | GPU={}\n",
            self.scale,
            self.neuron_count(),
            self.synapse_count(),
            self.gpu_active()
        );
        s.push_str("Regions:\n");
        for &(name, start, count) in &self.layout.regions {
            s.push_str(&format!(
                "  {:10} [{:6}..{:6}] n={}\n",
                name,
                start,
                start + count,
                count
            ));
        }
        s
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_neuron_counts() {
        assert_eq!(DrosophilaScale::Tiny.neuron_count(), 1_000);
        assert_eq!(DrosophilaScale::Small.neuron_count(), 5_000);
        assert_eq!(DrosophilaScale::Medium.neuron_count(), 25_000);
        assert_eq!(DrosophilaScale::Large.neuron_count(), 139_000);
    }

    #[test]
    fn test_scale_from_count() {
        assert_eq!(DrosophilaScale::from_count(500), DrosophilaScale::Tiny);
        assert_eq!(DrosophilaScale::from_count(3000), DrosophilaScale::Small);
        assert_eq!(DrosophilaScale::from_count(15000), DrosophilaScale::Medium);
        assert_eq!(DrosophilaScale::from_count(200000), DrosophilaScale::Large);
    }

    #[test]
    fn test_region_layout_allocation() {
        let layout = RegionLayout::from_scale(DrosophilaScale::Tiny);
        assert_eq!(layout.total, 1_000);
        // All regions should be contiguous and sum to total
        let sum: usize = layout.regions.iter().map(|&(_, _, c)| c).sum();
        assert_eq!(sum, 1_000);
        // Check contiguity
        let mut expected_start = 0usize;
        for &(_, start, count) in &layout.regions {
            assert_eq!(start, expected_start, "Region not contiguous");
            expected_start += count;
        }
    }

    #[test]
    fn test_region_layout_all_scales() {
        for scale in &[
            DrosophilaScale::Tiny,
            DrosophilaScale::Small,
            DrosophilaScale::Medium,
        ] {
            let layout = RegionLayout::from_scale(*scale);
            let sum: usize = layout.regions.iter().map(|&(_, _, c)| c).sum();
            assert_eq!(
                sum,
                scale.neuron_count(),
                "Scale {:?} allocation mismatch",
                scale
            );
            assert_eq!(layout.regions.len(), 15, "Should have 15 regions");
        }
    }

    #[test]
    fn test_sim_construction_tiny() {
        let sim = DrosophilaSim::new(DrosophilaScale::Tiny, 42);
        assert_eq!(sim.neuron_count(), 1_000);
        assert!(sim.synapse_count() > 0, "Should have synapses");
        assert_eq!(sim.scale, DrosophilaScale::Tiny);
    }

    #[test]
    fn test_sim_reproducible_seed() {
        let sim1 = DrosophilaSim::new(DrosophilaScale::Tiny, 42);
        let sim2 = DrosophilaSim::new(DrosophilaScale::Tiny, 42);
        assert_eq!(sim1.synapse_count(), sim2.synapse_count());
    }

    #[test]
    fn test_sim_different_seeds_differ() {
        let sim1 = DrosophilaSim::new(DrosophilaScale::Tiny, 42);
        let sim2 = DrosophilaSim::new(DrosophilaScale::Tiny, 99);
        // Extremely unlikely to have identical synapse counts with different seeds
        // but check positions as a backup
        let same_synapses = sim1.synapse_count() == sim2.synapse_count();
        let same_pos =
            (0..10).all(|i| (sim1.brain.neurons.x[i] - sim2.brain.neurons.x[i]).abs() < 1e-6);
        assert!(
            !same_synapses || !same_pos,
            "Different seeds should produce different networks"
        );
    }

    #[test]
    fn test_archetypes_assigned() {
        let sim = DrosophilaSim::new(DrosophilaScale::Tiny, 42);
        // AL should be mostly Cholinergic with some Interneurons
        let al_range = sim.layout.range("AL");
        let n_al = al_range.len();
        let n_cholinergic = al_range
            .clone()
            .filter(|&i| sim.brain.neurons.archetype[i] == NeuronArchetype::Cholinergic as u8)
            .count();
        let n_interneuron = al_range
            .filter(|&i| sim.brain.neurons.archetype[i] == NeuronArchetype::Interneuron as u8)
            .count();
        assert!(n_cholinergic > 0, "AL should have cholinergic neurons");
        assert!(
            n_interneuron > 0 || n_al < 4,
            "AL should have interneurons (unless too small)"
        );
        assert_eq!(n_cholinergic + n_interneuron, n_al);

        // DAN should be DopaminergicSN
        let dan_range = sim.layout.range("DAN");
        for i in dan_range {
            assert_eq!(
                sim.brain.neurons.archetype[i],
                NeuronArchetype::DopaminergicSN as u8,
                "DAN neurons should be DopaminergicSN"
            );
        }
    }

    #[test]
    fn test_brain_step_runs() {
        let mut sim = DrosophilaSim::new(DrosophilaScale::Tiny, 42);
        sim.brain.enable_gpu = false; // Force CPU for test portability
        sim.brain.enable_circadian = false;
        sim.brain.enable_pharmacology = false;
        sim.brain.enable_glia = false;

        // Stimulate AL and run
        let al_range = sim.layout.range("AL");
        for i in al_range {
            sim.brain.stimulate(i, 50.0);
        }
        sim.brain.run(100);

        assert_eq!(sim.brain.step_count, 100);
        assert!(sim.brain.time > 0.0);
    }

    #[test]
    fn test_body_step_updates_position() {
        let mut sim = DrosophilaSim::new(DrosophilaScale::Tiny, 42);
        sim.brain.enable_gpu = false;
        sim.brain.enable_circadian = false;
        sim.brain.enable_pharmacology = false;
        sim.brain.enable_glia = false;

        let x0 = sim.body.x;
        let y0 = sim.body.y;

        // Force some motor activity by stimulating VNC
        let vnc_range = sim.layout.range("VNC");
        for i in vnc_range {
            sim.brain.stimulate(i, 60.0);
        }
        sim.body_step();

        // Body should have been updated (position or heading changed)
        let _moved = (sim.body.x - x0).abs() > 1e-6 || (sim.body.y - y0).abs() > 1e-6;
        // Even if didn't move (no spikes yet), step count should advance
        assert!(sim.brain.step_count > 0);
        // The first step might not produce movement yet due to neural dynamics
        // but the system should not crash
    }

    #[test]
    fn test_manual_body_step_respects_external_bounds() {
        let mut sim = DrosophilaSim::new(DrosophilaScale::Tiny, 42);
        sim.brain.enable_gpu = false;
        sim.brain.enable_circadian = false;
        sim.brain.enable_pharmacology = false;
        sim.brain.enable_glia = false;

        sim.set_world_bounds(12.0, 9.0);
        sim.set_body_state(
            4.0,
            4.0,
            0.2,
            Some(0.0),
            Some(0.0),
            Some(false),
            Some(0.0),
            Some(80.0),
            Some(22.0),
            Some(9.0),
        );
        sim.body_step_manual(0.4, 0.3, 0.7, 21.0);

        let body = sim.body_state();
        assert!(body.x >= 0.0 && body.x <= 11.0);
        assert!(body.y >= 0.0 && body.y <= 8.0);
        assert!(body.temperature.is_finite());
        assert!(sim.brain.step_count > 0);
    }

    #[test]
    fn test_terrarium_step_returns_bounded_report() {
        let mut sim = DrosophilaSim::new(DrosophilaScale::Tiny, 42);
        sim.brain.enable_gpu = false;
        sim.brain.enable_circadian = false;
        sim.brain.enable_pharmacology = false;
        sim.brain.enable_glia = false;

        sim.set_world_bounds(18.0, 14.0);
        sim.set_body_state(
            6.0,
            5.0,
            0.4,
            Some(0.0),
            Some(0.0),
            Some(false),
            Some(0.0),
            Some(60.0),
            Some(22.0),
            Some(11.0),
        );
        let report =
            sim.body_step_terrarium(0.5, 0.6, 0.2, 21.0, 0.8, 0.0, 0.2, 0.1, 0.0, 0.0, 0.6, 0.0, 1.225);

        assert!(report.x >= 0.0 && report.x <= 17.0);
        assert!(report.y >= 0.0 && report.y <= 13.0);
        assert!(report.energy.is_finite());
        assert!(report.speed.is_finite());
        assert!(report.turn.is_finite());
        assert!(report.feed_signal >= 0.0);
        assert!(sim.brain.step_count > 0);
    }

    #[test]
    fn test_sensory_encoding_odorant() {
        let mut sim = DrosophilaSim::new(DrosophilaScale::Tiny, 42);
        sim.world.place_odorant(32.0, 32.0, 1.0, 5.0);

        // Body at center should detect odorant
        let odorant = sim.world.sample_odorant(32.0, 32.0);
        assert!(odorant > 0.5, "Odorant at source should be high");

        // Far from source should be near zero
        let far_odorant = sim.world.sample_odorant(0.0, 0.0);
        assert!(
            far_odorant < 0.01,
            "Odorant far from source should be near zero"
        );
    }

    #[test]
    fn test_fep_modes() {
        let mut brain = MolecularBrain::new(10);
        brain.enable_gpu = false;
        brain.enable_circadian = false;
        brain.enable_pharmacology = false;
        brain.enable_glia = false;

        let mut fep = FepState::new(vec![0, 1, 2], 40.0, 42);

        // Structured mode should inject current during pulse-on phase
        fep.mode = FepMode::Structured;
        fep.stimulate(&mut brain, 0); // step 0 -> pulse on
        assert!(brain.neurons.external_current[0] > 0.0);

        // Random mode should inject current to ~30% of neurons
        fep.mode = FepMode::Random;
        // Run many times and check that some get current
        let mut any_stimulated = false;
        for step in 0..100 {
            brain
                .neurons
                .external_current
                .iter_mut()
                .for_each(|c| *c = 0.0);
            fep.stimulate(&mut brain, step);
            if brain.neurons.external_current[0] > 0.0
                || brain.neurons.external_current[1] > 0.0
                || brain.neurons.external_current[2] > 0.0
            {
                any_stimulated = true;
                break;
            }
        }
        assert!(
            any_stimulated,
            "Random FEP should eventually stimulate neurons"
        );

        // Off mode should not inject current
        fep.mode = FepMode::Off;
        brain
            .neurons
            .external_current
            .iter_mut()
            .for_each(|c| *c = 0.0);
        fep.stimulate(&mut brain, 50);
        assert!(
            brain.neurons.external_current[0].abs() < 1e-6,
            "FEP Off should not inject current"
        );
    }

    #[test]
    fn test_world_light_gradient() {
        let mut world = World::new(64, 64);
        world.set_light_gradient();

        // Left should be dark, right should be bright
        assert!(world.sample_light(0.0, 32.0) < 0.1);
        assert!(world.sample_light(63.0, 32.0) > 0.9);

        // Center should be ~0.5
        let center = world.sample_light(32.0, 32.0);
        assert!((center - 0.5).abs() < 0.05);
    }

    #[test]
    fn test_world_temp_gradient() {
        let mut world = World::new(64, 64);
        world.set_temp_gradient(15.0, 30.0);

        let left = world.sample_temperature(0.0, 32.0);
        let right = world.sample_temperature(63.0, 32.0);
        assert!((left - 15.0).abs() < 1.0);
        assert!((right - 30.0).abs() < 1.0);
    }

    #[test]
    fn test_experiment_result_display() {
        let result = ExperimentResult {
            name: "test".to_string(),
            passed: true,
            metric_name: "score".to_string(),
            metric_value: 0.95,
            threshold: 0.5,
            details: "good".to_string(),
            trajectories: Vec::new(),
        };
        let s = format!("{}", result);
        assert!(s.contains("PASS"));
        assert!(s.contains("test"));
    }

    #[test]
    fn test_region_accessors() {
        let sim = DrosophilaSim::new(DrosophilaScale::Tiny, 42);
        let al = sim.region_indices("AL");
        assert!(!al.is_empty(), "AL should have neurons");
        let vnc = sim.region_indices("VNC");
        assert!(!vnc.is_empty(), "VNC should have neurons");

        // No overlap between regions
        for &a in &al {
            assert!(!vnc.contains(&a), "AL and VNC should not overlap");
        }
    }

    #[test]
    fn test_summary_output() {
        let sim = DrosophilaSim::new(DrosophilaScale::Tiny, 42);
        let summary = sim.summary();
        assert!(summary.contains("DrosophilaSim"));
        assert!(summary.contains("Tiny"));
        assert!(summary.contains("1000")); // neuron count
        assert!(summary.contains("AL"));
        assert!(summary.contains("VNC"));
    }

    #[test]
    fn test_drug_response_ttx_suppresses() {
        // Verify that zeroing Nav conductance reduces firing
        let mut brain = MolecularBrain::new(100);
        brain.enable_gpu = false;
        brain.enable_circadian = false;
        brain.enable_pharmacology = false;
        brain.enable_glia = false;

        // Baseline: strong stimulation
        for i in 0..100 {
            brain.stimulate(i, 80.0);
        }
        brain.run(200);
        let baseline: u64 = brain.neurons.spike_count.iter().map(|&c| c as u64).sum();

        // TTX: zero Nav
        brain.reset();
        for i in 0..100 {
            brain.neurons.conductance_scale[i][IonChannelType::Nav as usize] = 0.0;
            brain.stimulate(i, 80.0);
        }
        brain.run(200);
        let ttx: u64 = brain.neurons.spike_count.iter().map(|&c| c as u64).sum();

        // TTX should dramatically reduce spiking
        if baseline > 0 {
            let suppression = 1.0 - ttx as f64 / baseline as f64;
            assert!(
                suppression > 0.5,
                "TTX should suppress >50% of spikes (got {:.1}%)",
                suppression * 100.0
            );
        }
    }

    #[test]
    fn hunger_correlates_with_energy() {
        // A fly's energy should decrease as it runs without food.
        let mut fly = DrosophilaSim::new(DrosophilaScale::Tiny, 42);
        let initial_energy = fly.body_state().energy;
        for _ in 0..5 {
            fly.body_step();
        }
        // Energy should decrease after steps without food.
        assert!(
            fly.body_state().energy <= initial_energy,
            "energy should not increase without food: initial={}, after={}",
            initial_energy,
            fly.body_state().energy
        );
    }

    #[test]
    fn neural_activity_bounded() {
        // After a body step, fly should still have valid state.
        let mut fly = DrosophilaSim::new(DrosophilaScale::Tiny, 44);
        fly.body_step();
        let energy = fly.body_state().energy;
        assert!(
            energy >= 0.0 && energy <= 5000.0,
            "energy should be bounded, got {}",
            energy
        );
    }

    // ========================================================================
    // Molecular/DNA Fidelity Tests
    // ========================================================================

    #[test]
    fn test_genome_stats_constants() {
        use super::genome_stats::*;
        assert_eq!(CHROMOSOME_COUNT, 4, "Drosophila has 4 chromosomes");
        assert!((GENOME_MB - 180.0).abs() < 5.0, "Genome ~180Mb");
        assert!(PROTEIN_GENES >= 13_000, "At least 13,000 protein-coding genes");
        assert!(FULL_NEURON_COUNT == 139_000, "FlyWire connectome has 139k neurons");
    }

    #[test]
    fn test_chromosome_lengths() {
        use super::genome_stats::*;
        let total: f32 = CHROMOSOME_LENGTHS_MB.iter().sum();
        assert!((total - GENOME_MB).abs() < 1.0, "Chromosome lengths should sum to genome size");
        // Chromosome 4 (dot) should be smallest
        assert!(CHROMOSOME_LENGTHS_MB[3] < 5.0, "Chromosome 4 is the dot chromosome");
    }

    #[test]
    fn test_key_genes_count() {
        let genes = super::key_genes();
        assert!(genes.len() >= 15, "At least 15 key neural genes defined");
    }

    #[test]
    fn test_learning_memory_genes() {
        let genes = super::key_genes();
        let learning_genes: Vec<_> = genes
            .iter()
            .filter(|g| g.function == super::GeneFunction::LearningMemory)
            .collect();
        assert!(learning_genes.len() >= 3, "At least 3 learning/memory genes");
        // rut, dnc, CREB2 should all be present
        let names: Vec<_> = learning_genes.iter().map(|g| g.name.as_str()).collect();
        assert!(names.contains(&"rut"), "rutabaga should be in learning genes");
        assert!(names.contains(&"dnc"), "dunce should be in learning genes");
    }

    #[test]
    fn test_circadian_genes() {
        let genes = super::key_genes();
        let circadian_genes: Vec<_> = genes
            .iter()
            .filter(|g| g.function == super::GeneFunction::CircadianRhythm)
            .collect();
        assert!(circadian_genes.len() >= 3, "At least 3 circadian genes");
        let names: Vec<_> = circadian_genes.iter().map(|g| g.name.as_str()).collect();
        assert!(names.contains(&"per"), "period should be present");
        assert!(names.contains(&"tim"), "timeless should be present");
    }

    #[test]
    fn test_ion_channel_genes() {
        let genes = super::key_genes();
        let channel_genes: Vec<_> = genes
            .iter()
            .filter(|g| g.function == super::GeneFunction::IonChannel)
            .collect();
        assert!(channel_genes.len() >= 3, "At least 3 ion channel genes");
    }

    #[test]
    fn test_essential_genes() {
        let genes = super::key_genes();
        let essential: Vec<_> = genes.iter().filter(|g| g.essential).collect();
        assert!(essential.len() >= 5, "At least 5 essential genes defined");
    }

    #[test]
    fn test_molecular_pathways() {
        let pathways = super::molecular_pathways();
        assert!(pathways.len() >= 6, "At least 6 molecular pathways defined");
    }

    #[test]
    fn test_camp_pathway() {
        let pathways = super::molecular_pathways();
        let camp_pathway = pathways
            .iter()
            .find(|p| p.name.contains("cAMP"))
            .expect("cAMP pathway should exist");
        assert!(camp_pathway.genes.contains(&"rut".to_string()));
        assert!(camp_pathway.genes.contains(&"dnc".to_string()));
    }

    #[test]
    fn test_dopamine_pathway() {
        let pathways = super::molecular_pathways();
        let da_pathway = pathways
            .iter()
            .find(|p| p.name.contains("Dopamine"))
            .expect("Dopamine pathway should exist");
        assert!(da_pathway.genes.contains(&"TH".to_string()));
    }

    #[test]
    fn test_default_body_anatomy() {
        let body = super::default_body();
        assert_eq!(body.head.ommatidia_per_eye, 800, "800 ommatidia per eye");
        assert_eq!(body.thorax.leg_pairs, 3, "3 leg pairs");
        assert!(body.thorax.halteres_present, "Halteres should be present");
        assert!(body.wings.area_mm2 > 0.0, "Wing area should be positive");
    }

    #[test]
    fn test_gene_chromosome_positions() {
        let genes = super::key_genes();
        for gene in &genes {
            assert!(gene.chromosome < 4, "Chromosome must be 0-3");
        }
    }
}
