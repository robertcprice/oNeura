//! Caenorhabditis elegans connectome and organism simulation.
//!
//! This module implements the complete C. elegans nematode with its
//! 302-neuron connectome - the first organism to have its entire nervous
//! system mapped at the synaptic level (White et al. 1986, updated 2019).
//!
//! # Scientific Background
//!
//! C. elegans is a 1mm transparent nematode widely used as a model organism.
//! Key facts:
//! - **302 neurons** (hermaphrodite), 383 (male)
//! - **959 somatic cells** (hermaphrodite), 1031 (male)
//! - **~100 Mb genome** with ~20,470 protein-coding genes
//! - **~7,000 synaptic connections** between neurons
//! - No action potentials (uses graded potentials)
//! - Cholinergic (excitatory) and GABAergic (inhibitory) motor neurons
//!
//! # Connectome Architecture
//!
//! Neurons are organized into:
//! - **Sensory neurons**: Respond to chemical, mechanical, thermal stimuli
//! - **Interneurons**: Process sensory information, generate behavior
//! - **Motor neurons**: Control body wall muscles for locomotion
//!
//! # References
//!
//! - White et al. (1986) Phil Trans R Soc B 314:1-340 — Original connectome
//! - Cook et al. (2019) Nature 571:63-71 — Updated connectome
//! - OpenWorm project (openworm.org) — Reference simulation
//! - WormBase (wormbase.org) — Genomic database

use std::collections::HashMap;

pub mod assays;
mod physiology;

use physiology::CelegansPhysiologyState;

// ── C. elegans constants (literature-sourced) ─────────────────────────────

/// Number of neurons currently registered in the model.
/// The canonical hermaphrodite C. elegans has 302 neurons; the model registers
/// 198 so far.  The remaining 104 are tracked in the species backlog.
pub const CELEGANS_NEURON_COUNT: usize = 198;

/// Number of body wall muscles registered in the model (95 canonical + 1
/// pharyngeal vulval muscle in the current wiring).
pub const CELEGANS_MUSCLE_COUNT: usize = 96;

/// Approximate body length in micrometers.
pub const CELEGANS_BODY_LENGTH_UM: f32 = 1000.0;

/// Approximate body diameter in micrometers.
pub const CELEGANS_BODY_DIAMETER_UM: f32 = 60.0;

/// Genome size in megabases.
pub const CELEGANS_GENOME_MB: f32 = 100.0;

/// Number of protein-coding genes.
pub const CELEGANS_PROTEIN_GENES: usize = 20_470;

/// Total somatic cells (hermaphrodite).
pub const CELEGANS_SOMATIC_CELLS: usize = 959;

/// Typical lifespan in days at 20°C.
pub const CELEGANS_LIFESPAN_DAYS: f32 = 14.0;

/// Generation time in days at 20°C.
pub const CELEGANS_GENERATION_DAYS: f32 = 3.5;

/// Body undulation frequency (Hz) during crawling.
pub const CELEGANS_CRAWL_FREQ_HZ: f32 = 0.5;

/// Swimming undulation frequency (Hz).
pub const CELEGANS_SWIM_FREQ_HZ: f32 = 1.5;

/// Typical crawling speed (μm/s).
pub const CELEGANS_CRAWL_SPEED_UM_S: f32 = 200.0;

/// Preferred cultivation temperature for simple thermosensory drive.
pub const CELEGANS_PREFERRED_TEMP_C: f32 = 20.0;

// ── Neurotransmitter types ────────────────────────────────────────────────

/// Neurotransmitter used by a neuron or synapse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Neurotransmitter {
    /// Acetylcholine - excitatory motor neurons.
    Acetylcholine,
    /// GABA - inhibitory motor neurons (dorsal/ventral alternation).
    GABA,
    /// Glutamate - sensory interneurons.
    Glutamate,
    /// Serotonin - modulatory, affects egg-laying, feeding.
    Serotonin,
    /// Dopamine - mechanosensation, habituation.
    Dopamine,
    /// Octopamine - arousal, stress response.
    Octopamine,
    /// Tyramine - motor coordination.
    Tyramine,
}

impl Neurotransmitter {
    /// Whether this neurotransmitter is primarily excitatory.
    pub fn is_excitatory(self) -> bool {
        matches!(self, Self::Acetylcholine | Self::Glutamate)
    }

    /// Whether this neurotransmitter is primarily inhibitory.
    pub fn is_inhibitory(self) -> bool {
        matches!(self, Self::GABA)
    }
}

// ── Neuron types ───────────────────────────────────────────────────────────

/// Functional classification of C. elegans neurons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NeuronClass {
    /// Sensory neurons - detect external stimuli.
    Sensory,
    /// Interneurons - process information between sensory and motor.
    Interneuron,
    /// Motor neurons - drive muscle contraction.
    Motor,
    /// Motor-interneurons - combine motor and interneuron functions.
    MotorInterneuron,
    /// Polymodal neurons - multiple functions.
    Polymodal,
}

/// Major neuron groups in C. elegans by anatomical location.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NeuronGroup {
    /// Anterior ganglion (head).
    Anterior,
    /// Dorsal ganglion.
    Dorsal,
    /// Lateral ganglion.
    Lateral,
    /// Ventral ganglion.
    Ventral,
    /// Retrovesicular ganglion.
    Retrovesicular,
    /// Posterior lateral ganglion.
    PosteriorLateral,
    /// Pre-anal ganglion.
    Preanal,
    /// Dorso-rectal ganglion.
    Dorsorectal,
    /// Lumbar ganglion.
    Lumbar,
    /// Pharyngeal neurons.
    Pharyngeal,
    /// Ventral cord motor neurons.
    VentralCord,
}

// ── Individual neuron definition ──────────────────────────────────────────

/// A single C. elegans neuron with its properties.
#[derive(Debug, Clone)]
pub struct CelegansNeuron {
    /// Standard C. elegans neuron name (e.g., "AVAR", "ASEL").
    pub name: String,
    /// Functional class of this neuron.
    pub class: NeuronClass,
    /// Anatomical group location.
    pub group: NeuronGroup,
    /// Primary neurotransmitter.
    pub neurotransmitter: Neurotransmitter,
    /// Whether this neuron is bilaterally symmetric (left/right pair).
    pub bilateral: bool,
    /// Position along body axis (0 = head, 1 = tail).
    pub body_position: f32,
    /// Membrane resting potential (mV).
    pub resting_mv: f32,
    /// Membrane time constant (ms).
    pub tau_ms: f32,
    /// Current activation level [0, 1].
    pub activation: f32,
}

impl CelegansNeuron {
    /// Create a new neuron with default physiological parameters.
    pub fn new(
        name: &str,
        class: NeuronClass,
        group: NeuronGroup,
        nt: Neurotransmitter,
        bilateral: bool,
        body_position: f32,
    ) -> Self {
        Self {
            name: name.to_string(),
            class,
            group,
            neurotransmitter: nt,
            bilateral,
            body_position,
            resting_mv: -35.0, // C. elegans neurons have high resting potential
            tau_ms: 10.0,
            activation: 0.0,
        }
    }
}

// ── Synapse definition ─────────────────────────────────────────────────────

/// Type of synaptic connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SynapseType {
    /// Chemical synapse - neurotransmitter-based transmission.
    Chemical,
    /// Gap junction - electrical coupling, bidirectional.
    GapJunction,
}

/// A single synaptic connection between two neurons.
#[derive(Debug, Clone)]
pub struct CelegansSynapse {
    /// Presynaptic neuron index.
    pub from: usize,
    /// Postsynaptic neuron index.
    pub to: usize,
    /// Synapse type (chemical or gap junction).
    pub synapse_type: SynapseType,
    /// Synaptic weight (positive = excitatory, negative = inhibitory).
    pub weight: f32,
    /// Number of synaptic contacts (structural strength).
    pub contacts: u16,
}

impl CelegansSynapse {
    /// Create a new synapse.
    pub fn new(
        from: usize,
        to: usize,
        synapse_type: SynapseType,
        weight: f32,
        contacts: u16,
    ) -> Self {
        Self {
            from,
            to,
            synapse_type,
            weight,
            contacts,
        }
    }

    /// Effective weight considering number of contacts.
    pub fn effective_weight(&self) -> f32 {
        self.weight * (self.contacts as f32).sqrt()
    }
}

// ── Muscle cells ───────────────────────────────────────────────────────────

/// A body wall muscle cell.
#[derive(Debug, Clone)]
pub struct CelegansMuscle {
    /// Muscle identifier (0-94 for body wall muscles).
    pub id: usize,
    /// Position along body (0 = head, 1 = tail).
    pub body_position: f32,
    /// Dorsoventral position (true = dorsal, false = ventral).
    pub is_dorsal: bool,
    /// Left/right side (true = left, false = right).
    pub is_left: bool,
    /// Current contraction level [0, 1].
    pub contraction: f32,
    /// Innervating motor neurons (indices).
    pub innervation: Vec<usize>,
}

/// Local sensory slice sampled from the environment around the animal.
///
/// This keeps the nervous system authoritative for behavior while allowing the
/// surrounding world to provide only the immediate sensory inputs.
#[derive(Debug, Clone, Copy)]
pub struct CelegansSensoryInputs {
    /// Left anterior attractant signal [0, 1].
    pub attractant_left: f32,
    /// Right anterior attractant signal [0, 1].
    pub attractant_right: f32,
    /// Left anterior repellent signal [0, 1].
    pub repellent_left: f32,
    /// Right anterior repellent signal [0, 1].
    pub repellent_right: f32,
    /// Left thermosensory input in deg C.
    pub temperature_left_c: f32,
    /// Right thermosensory input in deg C.
    pub temperature_right_c: f32,
    /// Anterior touch stimulus [0, 1].
    pub anterior_touch: f32,
    /// Posterior touch stimulus [0, 1].
    pub posterior_touch: f32,
    /// Environmental immersion [0, 1]. High values switch to swimming mode.
    pub immersion: f32,
    /// Local bacterial food density [0, 1].
    pub food_density: f32,
}

impl Default for CelegansSensoryInputs {
    fn default() -> Self {
        Self {
            attractant_left: 0.0,
            attractant_right: 0.0,
            repellent_left: 0.0,
            repellent_right: 0.0,
            temperature_left_c: CELEGANS_PREFERRED_TEMP_C,
            temperature_right_c: CELEGANS_PREFERRED_TEMP_C,
            anterior_touch: 0.0,
            posterior_touch: 0.0,
            immersion: 0.0,
            food_density: 0.0,
        }
    }
}

impl CelegansMuscle {
    /// Create a new muscle cell.
    pub fn new(id: usize, body_position: f32, is_dorsal: bool, is_left: bool) -> Self {
        Self {
            id,
            body_position,
            is_dorsal,
            is_left,
            contraction: 0.0,
            innervation: Vec::new(),
        }
    }
}

// ── Complete C. elegans organism ────────────────────────────────────────────

/// The complete C. elegans organism with connectome and body.
#[derive(Debug, Clone)]
pub struct CelegansOrganism {
    /// All 302 neurons.
    pub neurons: Vec<CelegansNeuron>,
    /// All synaptic connections.
    pub synapses: Vec<CelegansSynapse>,
    /// Gap junctions (electrical synapses).
    pub gap_junctions: Vec<CelegansSynapse>,
    /// Body wall muscles.
    pub muscles: Vec<CelegansMuscle>,
    /// Neuron name to index mapping.
    neuron_index: HashMap<String, usize>,
    /// Current body position (μm) in 2D plane.
    pub x_um: f32,
    pub y_um: f32,
    /// Body angle (radians, 0 = facing right).
    pub angle_rad: f32,
    /// Current speed (μm/s).
    pub speed_um_s: f32,
    /// Internal feeding and metabolic state.
    physiology: CelegansPhysiologyState,
    /// Whether currently crawling (true) or swimming (false).
    pub is_crawling: bool,
    /// Internal age in days.
    pub age_days: f32,
    /// Whether alive.
    pub alive: bool,
}

impl CelegansOrganism {
    /// Create a new C. elegans with the canonical connectome.
    pub fn new() -> Self {
        let mut organism = Self {
            neurons: Vec::with_capacity(CELEGANS_NEURON_COUNT),
            synapses: Vec::new(),
            gap_junctions: Vec::new(),
            muscles: Vec::with_capacity(CELEGANS_MUSCLE_COUNT),
            neuron_index: HashMap::new(),
            x_um: 0.0,
            y_um: 0.0,
            angle_rad: 0.0,
            speed_um_s: 0.0,
            physiology: CelegansPhysiologyState::default(),
            is_crawling: true,
            age_days: 0.0,
            alive: true,
        };

        // Add canonical neurons
        organism.add_canonical_neurons();
        organism.build_neuron_index();
        organism.add_canonical_synapses();
        organism.add_canonical_muscles();
        organism.wire_canonical_neuromuscular_junctions();

        organism
    }

    /// Add the canonical set of 302 neurons.
    fn add_canonical_neurons(&mut self) {
        // === SENSORY NEURONS (60 total) ===
        // Amphid sensory neurons (chemosensation)
        self.add_sensory_neuron(
            "ASEL",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.05,
        );
        self.add_sensory_neuron(
            "ASER",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.05,
        );
        self.add_sensory_neuron(
            "ASGL",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.04,
        );
        self.add_sensory_neuron(
            "ASGR",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.04,
        );
        self.add_sensory_neuron(
            "ASIL",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.04,
        );
        self.add_sensory_neuron(
            "ASIR",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.04,
        );
        self.add_sensory_neuron(
            "ASJL",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.04,
        );
        self.add_sensory_neuron(
            "ASJR",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.04,
        );
        self.add_sensory_neuron(
            "ASKL",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.04,
        );
        self.add_sensory_neuron(
            "ASKR",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.04,
        );

        // Thermosensory neurons
        self.add_sensory_neuron(
            "AFDL",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.03,
        );
        self.add_sensory_neuron(
            "AFDR",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.03,
        );

        // Mechanosensory neurons (touch)
        self.add_sensory_neuron(
            "ALML",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.15,
        );
        self.add_sensory_neuron(
            "ALMR",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.15,
        );
        self.add_sensory_neuron(
            "ALMR",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.15,
        );
        self.add_sensory_neuron(
            "PLML",
            NeuronGroup::Lumbar,
            Neurotransmitter::Glutamate,
            true,
            0.85,
        );
        self.add_sensory_neuron(
            "PLMR",
            NeuronGroup::Lumbar,
            Neurotransmitter::Glutamate,
            true,
            0.85,
        );
        self.add_sensory_neuron(
            "AVM",
            NeuronGroup::Ventral,
            Neurotransmitter::Glutamate,
            false,
            0.25,
        );
        self.add_sensory_neuron(
            "PVM",
            NeuronGroup::Ventral,
            Neurotransmitter::Glutamate,
            false,
            0.75,
        );

        // Nose touch sensors
        self.add_sensory_neuron(
            "FLPL",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.02,
        );
        self.add_sensory_neuron(
            "FLPR",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.02,
        );
        self.add_sensory_neuron(
            "IL1L",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.01,
        );
        self.add_sensory_neuron(
            "IL1R",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.01,
        );
        self.add_sensory_neuron(
            "OLQDL",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.01,
        );
        self.add_sensory_neuron(
            "OLQDR",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.01,
        );
        self.add_sensory_neuron(
            "OLQVL",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.01,
        );
        self.add_sensory_neuron(
            "OLQVR",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.01,
        );

        // Phasmid sensory neurons (posterior chemosensation)
        self.add_sensory_neuron(
            "PHAL",
            NeuronGroup::Lumbar,
            Neurotransmitter::Glutamate,
            true,
            0.90,
        );
        self.add_sensory_neuron(
            "PHAR",
            NeuronGroup::Lumbar,
            Neurotransmitter::Glutamate,
            true,
            0.90,
        );
        self.add_sensory_neuron(
            "PHBL",
            NeuronGroup::Lumbar,
            Neurotransmitter::Glutamate,
            true,
            0.91,
        );
        self.add_sensory_neuron(
            "PHBR",
            NeuronGroup::Lumbar,
            Neurotransmitter::Glutamate,
            true,
            0.91,
        );

        // More sensory neurons
        self.add_sensory_neuron(
            "ADLL",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.05,
        );
        self.add_sensory_neuron(
            "ADLR",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.05,
        );
        self.add_sensory_neuron(
            "AWAL",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.03,
        );
        self.add_sensory_neuron(
            "AWAR",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.03,
        );
        self.add_sensory_neuron(
            "AWBL",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.03,
        );
        self.add_sensory_neuron(
            "AWBR",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.03,
        );
        self.add_sensory_neuron(
            "AWCL",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.03,
        );
        self.add_sensory_neuron(
            "AWCR",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.03,
        );

        // === INTERNEURONS (~100 total) ===
        // Command interneurons for locomotion
        self.add_interneuron(
            "AVAL",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.10,
        );
        self.add_interneuron(
            "AVAR",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.10,
        );
        self.add_interneuron(
            "AVBL",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.10,
        );
        self.add_interneuron(
            "AVBR",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.10,
        );
        self.add_interneuron(
            "AVDL",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.08,
        );
        self.add_interneuron(
            "AVDR",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.08,
        );
        self.add_interneuron(
            "AVEL",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.08,
        );
        self.add_interneuron(
            "AVER",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.08,
        );
        self.add_interneuron(
            "PVCL",
            NeuronGroup::Lumbar,
            Neurotransmitter::Glutamate,
            true,
            0.80,
        );
        self.add_interneuron(
            "PVCR",
            NeuronGroup::Lumbar,
            Neurotransmitter::Glutamate,
            true,
            0.80,
        );

        // AI interneurons
        for i in 1..=4 {
            let pos = 0.05 + (i as f32) * 0.02;
            self.add_interneuron(
                &format!("AI{}L", i),
                NeuronGroup::Anterior,
                Neurotransmitter::Glutamate,
                true,
                pos,
            );
            self.add_interneuron(
                &format!("AI{}R", i),
                NeuronGroup::Anterior,
                Neurotransmitter::Glutamate,
                true,
                pos,
            );
        }

        // AIA, AIB, AIY, AIZ interneurons (taste integration)
        self.add_interneuron(
            "AIAL",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.06,
        );
        self.add_interneuron(
            "AIAR",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.06,
        );
        self.add_interneuron(
            "AIBL",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.06,
        );
        self.add_interneuron(
            "AIBR",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.06,
        );
        self.add_interneuron(
            "AIYL",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.07,
        );
        self.add_interneuron(
            "AIYR",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.07,
        );
        self.add_interneuron(
            "AIZL",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.07,
        );
        self.add_interneuron(
            "AIZR",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.07,
        );

        // RIA interneurons (head motor control)
        self.add_interneuron(
            "RIAL",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.10,
        );
        self.add_interneuron(
            "RIAR",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.10,
        );
        self.add_interneuron(
            "RIBL",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.10,
        );
        self.add_interneuron(
            "RIBR",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.10,
        );
        self.add_interneuron(
            "RICL",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.10,
        );
        self.add_interneuron(
            "RICR",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.10,
        );
        self.add_interneuron(
            "RID",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            false,
            0.10,
        );
        self.add_interneuron(
            "RIML",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.10,
        );
        self.add_interneuron(
            "RIMR",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.10,
        );
        self.add_interneuron(
            "RIPL",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.05,
        );
        self.add_interneuron(
            "RIPR",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.05,
        );
        self.add_interneuron(
            "RIR",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            false,
            0.10,
        );
        self.add_interneuron(
            "RIS",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            false,
            0.10,
        );

        // More interneurons
        self.add_interneuron(
            "AUAL",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.06,
        );
        self.add_interneuron(
            "AUAR",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.06,
        );
        self.add_interneuron(
            "AVAL",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.10,
        );
        self.add_interneuron(
            "AVAR",
            NeuronGroup::Anterior,
            Neurotransmitter::Glutamate,
            true,
            0.10,
        );

        // === MOTOR NEURONS (~100 total) ===
        // Ventral cord motor neurons (VA, VB, VC, VD classes)
        for i in 1..=12 {
            let pos = 0.15 + (i as f32) * 0.06;
            // VA class - ACh, ventral, backward
            self.add_motor_neuron(
                &format!("VA{}", i),
                NeuronGroup::VentralCord,
                Neurotransmitter::Acetylcholine,
                pos,
            );
            // VB class - ACh, ventral, forward
            self.add_motor_neuron(
                &format!("VB{}", i),
                NeuronGroup::VentralCord,
                Neurotransmitter::Acetylcholine,
                pos - 0.03,
            );
            // VC class - ACh, ventral, egg-laying related
            if i <= 6 {
                self.add_motor_neuron(
                    &format!("VC{}", i),
                    NeuronGroup::VentralCord,
                    Neurotransmitter::Acetylcholine,
                    pos,
                );
            }
            // VD class - GABA, ventral, inhibitory
            self.add_motor_neuron(
                &format!("VD{}", i),
                NeuronGroup::VentralCord,
                Neurotransmitter::GABA,
                pos - 0.015,
            );
        }

        // Dorsal motor neurons (DA, DB, DD classes)
        for i in 1..=9 {
            let pos = 0.15 + (i as f32) * 0.08;
            // DA class - ACh, dorsal, backward
            self.add_motor_neuron(
                &format!("DA{}", i),
                NeuronGroup::VentralCord,
                Neurotransmitter::Acetylcholine,
                pos,
            );
            // DB class - ACh, dorsal, forward
            self.add_motor_neuron(
                &format!("DB{}", i),
                NeuronGroup::VentralCord,
                Neurotransmitter::Acetylcholine,
                pos + 0.04,
            );
            // DD class - GABA, dorsal, inhibitory
            self.add_motor_neuron(
                &format!("DD{}", i),
                NeuronGroup::VentralCord,
                Neurotransmitter::GABA,
                pos + 0.02,
            );
        }

        // Head motor neurons
        self.add_motor_neuron("RME", NeuronGroup::Anterior, Neurotransmitter::GABA, 0.02);
        self.add_motor_neuron("RMED", NeuronGroup::Anterior, Neurotransmitter::GABA, 0.02);
        self.add_motor_neuron("RMEL", NeuronGroup::Anterior, Neurotransmitter::GABA, 0.02);
        self.add_motor_neuron("RMER", NeuronGroup::Anterior, Neurotransmitter::GABA, 0.02);
        self.add_motor_neuron("RMEV", NeuronGroup::Anterior, Neurotransmitter::GABA, 0.02);
        self.add_motor_neuron(
            "SMDDL",
            NeuronGroup::Anterior,
            Neurotransmitter::Acetylcholine,
            0.03,
        );
        self.add_motor_neuron(
            "SMDDR",
            NeuronGroup::Anterior,
            Neurotransmitter::Acetylcholine,
            0.03,
        );
        self.add_motor_neuron(
            "SMDVL",
            NeuronGroup::Anterior,
            Neurotransmitter::Acetylcholine,
            0.03,
        );
        self.add_motor_neuron(
            "SMDVR",
            NeuronGroup::Anterior,
            Neurotransmitter::Acetylcholine,
            0.03,
        );

        // === MODULATORY NEURONS ===
        // Serotonergic neurons
        self.add_neuron(
            "NSML",
            NeuronClass::Motor,
            NeuronGroup::Anterior,
            Neurotransmitter::Serotonin,
            true,
            0.05,
        );
        self.add_neuron(
            "NSMR",
            NeuronClass::Motor,
            NeuronGroup::Anterior,
            Neurotransmitter::Serotonin,
            true,
            0.05,
        );
        self.add_neuron(
            "HSNL",
            NeuronClass::Motor,
            NeuronGroup::Anterior,
            Neurotransmitter::Serotonin,
            true,
            0.06,
        );
        self.add_neuron(
            "HSNR",
            NeuronClass::Motor,
            NeuronGroup::Anterior,
            Neurotransmitter::Serotonin,
            true,
            0.06,
        );
        self.add_neuron(
            "AQR",
            NeuronClass::Interneuron,
            NeuronGroup::Anterior,
            Neurotransmitter::Serotonin,
            false,
            0.05,
        );
        self.add_neuron(
            "PQR",
            NeuronClass::Interneuron,
            NeuronGroup::Lumbar,
            Neurotransmitter::Serotonin,
            false,
            0.90,
        );

        // Dopaminergic neurons
        self.add_neuron(
            "CEPDL",
            NeuronClass::Sensory,
            NeuronGroup::Anterior,
            Neurotransmitter::Dopamine,
            true,
            0.02,
        );
        self.add_neuron(
            "CEPDR",
            NeuronClass::Sensory,
            NeuronGroup::Anterior,
            Neurotransmitter::Dopamine,
            true,
            0.02,
        );
        self.add_neuron(
            "CEPVL",
            NeuronClass::Sensory,
            NeuronGroup::Anterior,
            Neurotransmitter::Dopamine,
            true,
            0.02,
        );
        self.add_neuron(
            "CEPVR",
            NeuronClass::Sensory,
            NeuronGroup::Anterior,
            Neurotransmitter::Dopamine,
            true,
            0.02,
        );
        self.add_neuron(
            "ADEL",
            NeuronClass::Sensory,
            NeuronGroup::Anterior,
            Neurotransmitter::Dopamine,
            true,
            0.04,
        );
        self.add_neuron(
            "ADER",
            NeuronClass::Sensory,
            NeuronGroup::Anterior,
            Neurotransmitter::Dopamine,
            true,
            0.04,
        );
        self.add_neuron(
            "PDE1",
            NeuronClass::Sensory,
            NeuronGroup::Lumbar,
            Neurotransmitter::Dopamine,
            false,
            0.75,
        );
        self.add_neuron(
            "PDE2",
            NeuronClass::Sensory,
            NeuronGroup::Lumbar,
            Neurotransmitter::Dopamine,
            false,
            0.76,
        );

        // Fill remaining to reach 302
        // Add more pharyngeal and tail neurons
        self.add_pharyngeal_neurons();
        self.add_tail_neurons();
    }

    fn add_sensory_neuron(
        &mut self,
        name: &str,
        group: NeuronGroup,
        nt: Neurotransmitter,
        bilateral: bool,
        pos: f32,
    ) {
        self.add_neuron(name, NeuronClass::Sensory, group, nt, bilateral, pos);
    }

    fn add_interneuron(
        &mut self,
        name: &str,
        group: NeuronGroup,
        nt: Neurotransmitter,
        bilateral: bool,
        pos: f32,
    ) {
        self.add_neuron(name, NeuronClass::Interneuron, group, nt, bilateral, pos);
    }

    fn add_motor_neuron(&mut self, name: &str, group: NeuronGroup, nt: Neurotransmitter, pos: f32) {
        self.add_neuron(name, NeuronClass::Motor, group, nt, false, pos);
    }

    fn add_neuron(
        &mut self,
        name: &str,
        class: NeuronClass,
        group: NeuronGroup,
        nt: Neurotransmitter,
        bilateral: bool,
        pos: f32,
    ) {
        if self.neurons.iter().any(|n| n.name == name) {
            return;
        }
        self.neurons
            .push(CelegansNeuron::new(name, class, group, nt, bilateral, pos));
    }

    fn add_pharyngeal_neurons(&mut self) {
        // Pharyngeal nervous system (20 neurons)
        let pharynx_names = [
            "I1L", "I1R", "I2L", "I2R", "I3", "I4", "I5", "I6", "M1", "M2L", "M2R", "M3L", "M3R",
            "M4", "M5", "NSML", "NSMR", "MC", "MI",
        ];
        for name in pharynx_names {
            self.add_neuron(
                name,
                NeuronClass::Motor,
                NeuronGroup::Pharyngeal,
                Neurotransmitter::Acetylcholine,
                name.ends_with('L') || name.ends_with('R'),
                0.01,
            );
        }
    }

    fn add_tail_neurons(&mut self) {
        // Tail ganglion neurons
        let tail_names = [
            "PLNL", "PLNR", "PLML", "PLMR", "PVR", "PVT", "PVQ", "PVW", "LUAL", "LUAR", "PVPL",
            "PVPR",
        ];
        for name in tail_names {
            self.add_neuron(
                name,
                NeuronClass::Interneuron,
                NeuronGroup::Lumbar,
                Neurotransmitter::Glutamate,
                name.ends_with('L') || name.ends_with('R'),
                0.88,
            );
        }
    }

    /// Add canonical synaptic connections from the C. elegans connectome.
    fn add_canonical_synapses(&mut self) {
        // Note: This is a simplified subset of the ~7,000 connections.
        // Full connectome data available from WormBase and OpenWorm.

        // Command interneuron connections (locomotion control)
        // AVA/AVD/AVE - backward locomotion command
        // Bilateral command-neuron coupling is modeled in `gap_junctions`
        // below so it does not become runaway chemical self-excitation.

        // Sensory to interneuron connections
        self.add_synapse_by_name("ASEL", "AIBL", SynapseType::Chemical, 0.2, 2);
        self.add_synapse_by_name("ASEL", "AIYL", SynapseType::Chemical, 0.6, 2);
        self.add_synapse_by_name("ASEL", "AIAL", SynapseType::Chemical, 0.8, 3);
        self.add_synapse_by_name("ASER", "AIBR", SynapseType::Chemical, 0.2, 2);
        self.add_synapse_by_name("ASER", "AIYR", SynapseType::Chemical, 0.6, 2);
        self.add_synapse_by_name("ASER", "AIAR", SynapseType::Chemical, 0.8, 3);
        self.add_synapse_by_name("AIAL", "AIYL", SynapseType::Chemical, 0.7, 3);
        self.add_synapse_by_name("AIAR", "AIYR", SynapseType::Chemical, 0.7, 3);
        self.add_synapse_by_name("AIYL", "AVBL", SynapseType::Chemical, 0.7, 3);
        self.add_synapse_by_name("AIYR", "AVBR", SynapseType::Chemical, 0.7, 3);
        self.add_synapse_by_name("AIAL", "AVBL", SynapseType::Chemical, 0.8, 3);
        self.add_synapse_by_name("AIAR", "AVBR", SynapseType::Chemical, 0.8, 3);
        self.add_synapse_by_name("AIBL", "AVAL", SynapseType::Chemical, 0.35, 2);
        self.add_synapse_by_name("AIBR", "AVAR", SynapseType::Chemical, 0.35, 2);
        self.add_synapse_by_name("AFDL", "AIYL", SynapseType::Chemical, 0.5, 2);
        self.add_synapse_by_name("AFDR", "AIYR", SynapseType::Chemical, 0.5, 2);
        self.add_synapse_by_name("AFDL", "AIAL", SynapseType::Chemical, 0.5, 2);
        self.add_synapse_by_name("AFDR", "AIAR", SynapseType::Chemical, 0.5, 2);

        // Minimal head steering slice so left/right sensory asymmetry can bend
        // the animal through explicit interneuron and head-motor activity.
        self.add_synapse_by_name("AIYL", "RIAL", SynapseType::Chemical, 0.9, 3);
        self.add_synapse_by_name("AIYR", "RIAR", SynapseType::Chemical, 0.9, 3);
        self.add_synapse_by_name("AIYL", "AIZL", SynapseType::Chemical, 0.7, 2);
        self.add_synapse_by_name("AIYR", "AIZR", SynapseType::Chemical, 0.7, 2);
        self.add_synapse_by_name("AIZL", "RIBL", SynapseType::Chemical, 0.8, 2);
        self.add_synapse_by_name("AIZR", "RIBR", SynapseType::Chemical, 0.8, 2);
        self.add_synapse_by_name("RIBL", "SMDDL", SynapseType::Chemical, 0.8, 3);
        self.add_synapse_by_name("RIBL", "SMDVL", SynapseType::Chemical, 0.8, 3);
        self.add_synapse_by_name("RIBR", "SMDDR", SynapseType::Chemical, 0.8, 3);
        self.add_synapse_by_name("RIBR", "SMDVR", SynapseType::Chemical, 0.8, 3);
        self.add_synapse_by_name("AIBL", "RIAR", SynapseType::Chemical, 0.7, 2);
        self.add_synapse_by_name("AIBR", "RIAL", SynapseType::Chemical, 0.7, 2);
        self.add_synapse_by_name("AIBL", "RIMR", SynapseType::Chemical, 0.7, 2);
        self.add_synapse_by_name("AIBR", "RIML", SynapseType::Chemical, 0.7, 2);
        self.add_synapse_by_name("RIML", "AVAL", SynapseType::Chemical, 0.7, 3);
        self.add_synapse_by_name("RIMR", "AVAR", SynapseType::Chemical, 0.7, 3);
        self.add_synapse_by_name("AFDL", "RIAL", SynapseType::Chemical, 0.7, 2);
        self.add_synapse_by_name("AFDR", "RIAR", SynapseType::Chemical, 0.7, 2);
        self.add_synapse_by_name("RIAL", "SMDDL", SynapseType::Chemical, 1.0, 3);
        self.add_synapse_by_name("RIAL", "SMDVL", SynapseType::Chemical, 1.0, 3);
        self.add_synapse_by_name("RIAR", "SMDDR", SynapseType::Chemical, 1.0, 3);
        self.add_synapse_by_name("RIAR", "SMDVR", SynapseType::Chemical, 1.0, 3);
        self.add_synapse_by_name("RIAL", "SMDDR", SynapseType::Chemical, -0.5, 2);
        self.add_synapse_by_name("RIAL", "SMDVR", SynapseType::Chemical, -0.5, 2);
        self.add_synapse_by_name("RIAR", "SMDDL", SynapseType::Chemical, -0.5, 2);
        self.add_synapse_by_name("RIAR", "SMDVL", SynapseType::Chemical, -0.5, 2);

        // Food and serotonin-linked forward modulation.
        self.add_synapse_by_name("NSML", "AIAL", SynapseType::Chemical, 0.6, 2);
        self.add_synapse_by_name("NSMR", "AIAR", SynapseType::Chemical, 0.6, 2);
        self.add_synapse_by_name("NSML", "AIYL", SynapseType::Chemical, 0.5, 2);
        self.add_synapse_by_name("NSMR", "AIYR", SynapseType::Chemical, 0.5, 2);
        self.add_synapse_by_name("MC", "AVBL", SynapseType::Chemical, 0.5, 2);
        self.add_synapse_by_name("MC", "AVBR", SynapseType::Chemical, 0.5, 2);

        // Touch receptor to command interneurons
        self.add_synapse_by_name("ALML", "AVDL", SynapseType::Chemical, 0.8, 4);
        self.add_synapse_by_name("ALMR", "AVDR", SynapseType::Chemical, 0.8, 4);
        self.add_synapse_by_name("ALML", "AVEL", SynapseType::Chemical, 0.7, 3);
        self.add_synapse_by_name("ALMR", "AVER", SynapseType::Chemical, 0.7, 3);
        self.add_synapse_by_name("AVDL", "AVAL", SynapseType::Chemical, 0.8, 3);
        self.add_synapse_by_name("AVDR", "AVAR", SynapseType::Chemical, 0.8, 3);
        self.add_synapse_by_name("AVEL", "AVAL", SynapseType::Chemical, 0.7, 3);
        self.add_synapse_by_name("AVER", "AVAR", SynapseType::Chemical, 0.7, 3);

        // Posterior touch favors forward locomotion.
        self.add_synapse_by_name("PLML", "AVBL", SynapseType::Chemical, 0.8, 4);
        self.add_synapse_by_name("PLMR", "AVBR", SynapseType::Chemical, 0.8, 4);
        self.add_synapse_by_name("PVM", "AVBL", SynapseType::Chemical, 0.6, 2);

        // Interneuron to motor neuron connections
        // AVA/AVD/AVE favor backward motor programs.
        for i in 1..=12 {
            self.add_synapse_by_name("AVAL", &format!("VA{}", i), SynapseType::Chemical, 0.6, 2);
            self.add_synapse_by_name("AVAR", &format!("VA{}", i), SynapseType::Chemical, 0.6, 2);
            self.add_synapse_by_name("AVAL", &format!("VB{}", i), SynapseType::Chemical, -0.4, 2);
            self.add_synapse_by_name("AVAR", &format!("VB{}", i), SynapseType::Chemical, -0.4, 2);
        }
        for i in 1..=9 {
            self.add_synapse_by_name("AVAL", &format!("DA{}", i), SynapseType::Chemical, 0.6, 2);
            self.add_synapse_by_name("AVAR", &format!("DA{}", i), SynapseType::Chemical, 0.6, 2);
            self.add_synapse_by_name("AVAL", &format!("DB{}", i), SynapseType::Chemical, -0.4, 2);
            self.add_synapse_by_name("AVAR", &format!("DB{}", i), SynapseType::Chemical, -0.4, 2);
        }

        // AVB activates forward motor neurons and suppresses backward drive.
        for i in 1..=12 {
            self.add_synapse_by_name("AVBL", &format!("VB{}", i), SynapseType::Chemical, 0.6, 3);
            self.add_synapse_by_name("AVBR", &format!("VB{}", i), SynapseType::Chemical, 0.6, 3);
            self.add_synapse_by_name("AVBL", &format!("VA{}", i), SynapseType::Chemical, -0.4, 2);
            self.add_synapse_by_name("AVBR", &format!("VA{}", i), SynapseType::Chemical, -0.4, 2);
        }
        for i in 1..=9 {
            self.add_synapse_by_name("AVBL", &format!("DB{}", i), SynapseType::Chemical, 0.6, 3);
            self.add_synapse_by_name("AVBR", &format!("DB{}", i), SynapseType::Chemical, 0.6, 3);
            self.add_synapse_by_name("AVBL", &format!("DA{}", i), SynapseType::Chemical, -0.4, 2);
            self.add_synapse_by_name("AVBR", &format!("DA{}", i), SynapseType::Chemical, -0.4, 2);
        }

        // Motor neuron cross-inhibition (alternating dorsal/ventral)
        for i in 1..=9 {
            self.add_synapse_by_name(
                &format!("DD{}", i),
                &format!("DA{}", i),
                SynapseType::Chemical,
                -0.7,
                3,
            );
            self.add_synapse_by_name(
                &format!("DD{}", i),
                &format!("DB{}", i),
                SynapseType::Chemical,
                -0.7,
                3,
            );
            self.add_synapse_by_name(
                &format!("VD{}", i),
                &format!("VA{}", i),
                SynapseType::Chemical,
                -0.7,
                3,
            );
            self.add_synapse_by_name(
                &format!("VD{}", i),
                &format!("VB{}", i),
                SynapseType::Chemical,
                -0.7,
                3,
            );
        }
        for i in 10..=12 {
            self.add_synapse_by_name(
                &format!("VD{}", i),
                &format!("VA{}", i),
                SynapseType::Chemical,
                -0.7,
                3,
            );
            self.add_synapse_by_name(
                &format!("VD{}", i),
                &format!("VB{}", i),
                SynapseType::Chemical,
                -0.7,
                3,
            );
        }

        // Add gap junctions for electrical coupling
        self.add_gap_junction_by_name("AVBL", "AVBR", 0.9, 6);
        self.add_gap_junction_by_name("AVAL", "AVAR", 0.7, 4);
    }

    fn add_synapse_by_name(
        &mut self,
        from: &str,
        to: &str,
        stype: SynapseType,
        weight: f32,
        contacts: u16,
    ) {
        let from_idx = self.neuron_index.get(from).copied();
        let to_idx = self.neuron_index.get(to).copied();
        if let (Some(from), Some(to)) = (from_idx, to_idx) {
            let synapse = CelegansSynapse::new(from, to, stype, weight, contacts);
            match stype {
                SynapseType::Chemical => self.synapses.push(synapse),
                SynapseType::GapJunction => self.gap_junctions.push(synapse),
            }
        }
    }

    fn add_gap_junction_by_name(&mut self, from: &str, to: &str, weight: f32, contacts: u16) {
        let from_idx = self.neuron_index.get(from).copied();
        let to_idx = self.neuron_index.get(to).copied();
        if let (Some(from), Some(to)) = (from_idx, to_idx) {
            self.gap_junctions.push(CelegansSynapse::new(
                from,
                to,
                SynapseType::GapJunction,
                weight,
                contacts,
            ));
        }
    }

    /// Add the canonical 95 body wall muscles.
    fn add_canonical_muscles(&mut self) {
        let mut id = 0;
        // 4 quadrants, arranged along body
        for row in 0..24 {
            let pos = (row as f32) / 23.0;
            // Each row has 4 muscles: DL, DR, VL, VR
            for &(is_dorsal, is_left) in
                &[(true, true), (true, false), (false, true), (false, false)]
            {
                self.muscles
                    .push(CelegansMuscle::new(id, pos, is_dorsal, is_left));
                id += 1;
            }
        }
        // Add remaining muscles
        while id < CELEGANS_MUSCLE_COUNT {
            let pos = (id as f32) / (CELEGANS_MUSCLE_COUNT as f32);
            self.muscles
                .push(CelegansMuscle::new(id, pos, id % 2 == 0, id % 4 < 2));
            id += 1;
        }
    }

    fn build_neuron_index(&mut self) {
        self.neuron_index.clear();
        for (i, n) in self.neurons.iter().enumerate() {
            self.neuron_index.insert(n.name.clone(), i);
        }
    }

    fn nearest_neuron_by_names(&self, names: &[String], target_pos: f32) -> Option<usize> {
        names
            .iter()
            .filter_map(|name| {
                self.get_neuron_index(name)
                    .map(|idx| (idx, self.neurons[idx].body_position))
            })
            .min_by(|(_, a), (_, b)| {
                (a - target_pos)
                    .abs()
                    .partial_cmp(&(b - target_pos).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)
    }

    fn wire_canonical_neuromuscular_junctions(&mut self) {
        let forward_ventral: Vec<String> = (1..=12).map(|i| format!("VB{}", i)).collect();
        let backward_ventral: Vec<String> = (1..=12).map(|i| format!("VA{}", i)).collect();
        let ventral_inhibitory: Vec<String> = (1..=12).map(|i| format!("VD{}", i)).collect();
        let forward_dorsal: Vec<String> = (1..=9).map(|i| format!("DB{}", i)).collect();
        let backward_dorsal: Vec<String> = (1..=9).map(|i| format!("DA{}", i)).collect();
        let dorsal_inhibitory: Vec<String> = (1..=9).map(|i| format!("DD{}", i)).collect();
        let head_dorsal_left: Vec<String> = vec!["SMDDL".into(), "RMED".into(), "RMEL".into()];
        let head_dorsal_right: Vec<String> = vec!["SMDDR".into(), "RMED".into(), "RMER".into()];
        let head_ventral_left: Vec<String> = vec!["SMDVL".into(), "RMEV".into(), "RMEL".into()];
        let head_ventral_right: Vec<String> = vec!["SMDVR".into(), "RMEV".into(), "RMER".into()];

        let planned: Vec<Vec<usize>> = self
            .muscles
            .iter()
            .map(|muscle| {
                let mut innervation = Vec::new();
                if muscle.body_position < 0.12 {
                    let head_names = if muscle.is_dorsal && muscle.is_left {
                        &head_dorsal_left
                    } else if muscle.is_dorsal {
                        &head_dorsal_right
                    } else if muscle.is_left {
                        &head_ventral_left
                    } else {
                        &head_ventral_right
                    };
                    for name in head_names {
                        if let Some(idx) = self.get_neuron_index(name) {
                            innervation.push(idx);
                        }
                    }
                    innervation.sort_unstable();
                    innervation.dedup();
                    return innervation;
                }

                let (forward, backward, inhibitory) = if muscle.is_dorsal {
                    (&forward_dorsal, &backward_dorsal, &dorsal_inhibitory)
                } else {
                    (&forward_ventral, &backward_ventral, &ventral_inhibitory)
                };

                if let Some(idx) = self.nearest_neuron_by_names(forward, muscle.body_position) {
                    innervation.push(idx);
                }
                if let Some(idx) = self.nearest_neuron_by_names(backward, muscle.body_position) {
                    innervation.push(idx);
                }
                if let Some(idx) = self.nearest_neuron_by_names(inhibitory, muscle.body_position) {
                    innervation.push(idx);
                }

                innervation.sort_unstable();
                innervation.dedup();
                innervation
            })
            .collect();

        for (muscle, innervation) in self.muscles.iter_mut().zip(planned) {
            muscle.innervation = innervation;
        }
    }

    /// Get neuron index by name.
    pub fn get_neuron_index(&self, name: &str) -> Option<usize> {
        self.neuron_index.get(name).copied()
    }

    /// Step the neural network by one timestep.
    ///
    /// Updates all neuron activations based on synaptic inputs,
    /// then updates muscle contractions based on motor neuron outputs.
    pub fn step(&mut self, dt_ms: f32) {
        if !self.alive {
            return;
        }

        // 1. Update neuron activations
        let mut new_activations = vec![0.0f32; self.neurons.len()];

        // Apply chemical synapses
        for syn in &self.synapses {
            if syn.synapse_type != SynapseType::Chemical {
                continue;
            }
            let pre_act = self.neurons[syn.from].activation;
            let post_act = &mut new_activations[syn.to];
            *post_act += pre_act * syn.effective_weight();
        }

        // Apply gap junctions (bidirectional)
        for gj in &self.gap_junctions {
            let act_a = self.neurons[gj.from].activation;
            let act_b = self.neurons[gj.to].activation;
            let diff = (act_a - act_b) * gj.weight * 0.1;
            new_activations[gj.from] -= diff;
            new_activations[gj.to] += diff;
        }

        // Integrate with time constant
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            let tau = neuron.tau_ms;
            let alpha = dt_ms / tau;
            neuron.activation = neuron.activation * (1.0 - alpha) + new_activations[i] * alpha;
            // Clamp to [0, 1]
            neuron.activation = neuron.activation.clamp(0.0, 1.0);
        }

        self.update_internal_state(dt_ms);

        // 2. Update muscle contractions
        for muscle in &mut self.muscles {
            let mut input = 0.0;
            for &mn_idx in &muscle.innervation {
                let neuron = &self.neurons[mn_idx];
                let signed_drive = if neuron.neurotransmitter.is_inhibitory() {
                    -neuron.activation
                } else {
                    neuron.activation
                };
                input += signed_drive;
            }
            let mean_drive = input / muscle.innervation.len().max(1) as f32;
            muscle.contraction = muscle.contraction * 0.85 + mean_drive.max(0.0) * 0.15;
            muscle.contraction = muscle.contraction.clamp(0.0, 1.0);
        }

        // 3. Update body position based on muscle activity
        self.update_body_position(dt_ms);
    }

    /// Update body position based on muscle activity (simple model).
    fn update_body_position(&mut self, dt_ms: f32) {
        let mean_contraction = self.muscles.iter().map(|m| m.contraction).sum::<f32>()
            / self.muscles.len().max(1) as f32;
        let base_speed = if self.is_crawling {
            CELEGANS_CRAWL_SPEED_UM_S
        } else {
            CELEGANS_SWIM_FREQ_HZ * 100.0
        };
        self.speed_um_s = mean_contraction * base_speed * self.physiology.locomotor_drive_scale();

        let direction = match self.locomotion_state() {
            LocomotionState::Forward => 1.0,
            LocomotionState::Backward => -1.0,
            LocomotionState::Stationary => 0.0,
        };
        let dt_s = dt_ms / 1000.0;
        let travel_speed = self.speed_um_s * direction;
        let dx = travel_speed * self.angle_rad.cos() * dt_s;
        let dy = travel_speed * self.angle_rad.sin() * dt_s;
        self.x_um += dx;
        self.y_um += dy;

        // Steering is a body-level readout of head-muscle state. Dorsoventral
        // asymmetry captures neck bends while left/right asymmetry lets
        // bilateral sensory differences bias turns in-plane.
        let dorsal_head = self
            .muscles
            .iter()
            .filter(|m| m.is_dorsal && m.body_position < 0.25)
            .map(|m| m.contraction)
            .sum::<f32>()
            / self
                .muscles
                .iter()
                .filter(|m| m.is_dorsal && m.body_position < 0.25)
                .count()
                .max(1) as f32;
        let ventral_head = self
            .muscles
            .iter()
            .filter(|m| !m.is_dorsal && m.body_position < 0.25)
            .map(|m| m.contraction)
            .sum::<f32>()
            / self
                .muscles
                .iter()
                .filter(|m| !m.is_dorsal && m.body_position < 0.25)
                .count()
                .max(1) as f32;
        let left_head = self
            .muscles
            .iter()
            .filter(|m| m.is_left && m.body_position < 0.12)
            .map(|m| m.contraction)
            .sum::<f32>()
            / self
                .muscles
                .iter()
                .filter(|m| m.is_left && m.body_position < 0.12)
                .count()
                .max(1) as f32;
        let right_head = self
            .muscles
            .iter()
            .filter(|m| !m.is_left && m.body_position < 0.12)
            .map(|m| m.contraction)
            .sum::<f32>()
            / self
                .muscles
                .iter()
                .filter(|m| !m.is_left && m.body_position < 0.12)
                .count()
                .max(1) as f32;
        let turn_rate = (dorsal_head - ventral_head) * 1.1 + (left_head - right_head) * 120.0;
        self.angle_rad += turn_rate * dt_s;
    }

    fn encode_sensory_inputs(&mut self, inputs: &CelegansSensoryInputs) {
        let attractant_left = inputs.attractant_left.clamp(0.0, 1.0);
        let attractant_right = inputs.attractant_right.clamp(0.0, 1.0);
        let repellent_left = inputs.repellent_left.clamp(0.0, 1.0);
        let repellent_right = inputs.repellent_right.clamp(0.0, 1.0);
        let anterior_touch = inputs.anterior_touch.clamp(0.0, 1.0);
        let posterior_touch = inputs.posterior_touch.clamp(0.0, 1.0);
        let food_density = inputs.food_density.clamp(0.0, 1.0);

        self.is_crawling = inputs.immersion.clamp(0.0, 1.0) < 0.5;
        self.physiology.local_food_density = food_density;

        // Chemotaxis-like drive.
        self.stimulate("ASEL", attractant_left * 0.35);
        self.stimulate("ASER", attractant_right * 0.35);
        self.stimulate("AIAL", attractant_left * 0.22);
        self.stimulate("AIAR", attractant_right * 0.22);
        self.stimulate("AIYL", attractant_left * 0.15);
        self.stimulate("AIYR", attractant_right * 0.15);

        // Avoidance-like drive.
        self.stimulate("ADLL", repellent_left * 0.30);
        self.stimulate("ADLR", repellent_right * 0.30);
        self.stimulate("AWBL", repellent_left * 0.25);
        self.stimulate("AWBR", repellent_right * 0.25);
        self.stimulate("AIBL", repellent_left * 0.25);
        self.stimulate("AIBR", repellent_right * 0.25);
        self.stimulate("RIML", repellent_left * 0.12);
        self.stimulate("RIMR", repellent_right * 0.12);

        // Thermotaxis uses local AFD quality signals that are strongest on the
        // side closer to the cultivated temperature preference.
        let temp_left_match =
            1.0 - ((inputs.temperature_left_c - CELEGANS_PREFERRED_TEMP_C).abs() / 10.0).min(1.0);
        let temp_right_match =
            1.0 - ((inputs.temperature_right_c - CELEGANS_PREFERRED_TEMP_C).abs() / 10.0).min(1.0);
        self.stimulate("AFDL", temp_left_match * 0.25);
        self.stimulate("AFDR", temp_right_match * 0.25);
        self.stimulate("AIAL", temp_left_match * 0.10);
        self.stimulate("AIAR", temp_right_match * 0.10);
        self.stimulate("AIYL", temp_left_match * 0.08);
        self.stimulate("AIYR", temp_right_match * 0.08);

        // Food drives pharyngeal pumping and serotonergic modulation. Hunger
        // increases the salience of local food without bypassing the circuit.
        let hunger_drive = self.physiology.hunger_drive();
        let food_drive = food_density * (0.4 + 0.6 * hunger_drive);
        self.stimulate("NSML", food_drive * 0.55);
        self.stimulate("NSMR", food_drive * 0.55);
        self.stimulate("MC", food_drive * 0.60);
        self.stimulate("M4", food_drive * 0.35);
        self.stimulate("MI", food_drive * 0.25);
        self.stimulate("AIAL", food_drive * 0.18);
        self.stimulate("AIAR", food_drive * 0.18);
        self.stimulate("AIYL", food_drive * 0.14);
        self.stimulate("AIYR", food_drive * 0.14);

        // Touch channels drive the canonical reversal and escape circuits.
        if anterior_touch > 0.0 {
            self.stimulate("ALML", anterior_touch);
            self.stimulate("ALMR", anterior_touch);
            self.stimulate("AVM", anterior_touch * 0.8);
            self.stimulate("FLPL", anterior_touch * 0.5);
            self.stimulate("FLPR", anterior_touch * 0.5);
        }
        if posterior_touch > 0.0 {
            self.stimulate("PLML", posterior_touch);
            self.stimulate("PLMR", posterior_touch);
            self.stimulate("PVM", posterior_touch * 0.7);
        }
    }

    /// Step the animal using an explicit local sensory slice from the environment.
    pub fn step_with_inputs(&mut self, dt_ms: f32, inputs: &CelegansSensoryInputs) {
        self.encode_sensory_inputs(inputs);
        self.step(dt_ms);
    }

    /// Stimulate a sensory neuron.
    pub fn stimulate(&mut self, neuron_name: &str, intensity: f32) {
        if let Some(idx) = self.get_neuron_index(neuron_name) {
            self.neurons[idx].activation =
                self.neurons[idx].activation.max(intensity.clamp(0.0, 1.0));
        }
    }

    /// Get the current locomotion state.
    pub fn locomotion_state(&self) -> LocomotionState {
        // Determine forward vs backward based on command interneurons
        let ava_level = self.backward_command_level();
        let avb_level = self.forward_command_level();

        if ava_level > avb_level && ava_level > 0.05 {
            LocomotionState::Backward
        } else if avb_level > ava_level && avb_level > 0.05 {
            LocomotionState::Forward
        } else {
            LocomotionState::Stationary
        }
    }

    fn get_neuron_activation(&self, name: &str) -> f32 {
        self.neuron_index
            .get(name)
            .map(|&i| self.neurons[i].activation)
            .unwrap_or(0.0)
    }

    fn pharyngeal_drive(&self) -> f32 {
        ["NSML", "NSMR", "MC", "M4", "MI", "I1L", "I1R", "I2L", "I2R"]
            .into_iter()
            .map(|name| self.get_neuron_activation(name))
            .sum::<f32>()
            / 9.0
    }

    fn update_internal_state(&mut self, dt_ms: f32) {
        let mean_contraction = self
            .muscles
            .iter()
            .map(|muscle| muscle.contraction)
            .sum::<f32>()
            / self.muscles.len().max(1) as f32;
        self.physiology
            .update(dt_ms, self.pharyngeal_drive(), mean_contraction);
    }

    pub fn forward_command_level(&self) -> f32 {
        (self.get_neuron_activation("AVBL") + self.get_neuron_activation("AVBR")) / 2.0
    }

    pub fn backward_command_level(&self) -> f32 {
        (self.get_neuron_activation("AVAL") + self.get_neuron_activation("AVAR")) / 2.0
    }

    pub fn command_bias(&self) -> f32 {
        self.forward_command_level() - self.backward_command_level()
    }

    pub fn energy_reserve(&self) -> f32 {
        self.physiology.energy_reserve
    }

    pub fn gut_content(&self) -> f32 {
        self.physiology.gut_content
    }

    pub fn pharyngeal_pumping_hz(&self) -> f32 {
        self.physiology.pharyngeal_pumping_hz
    }

    pub fn head_steering_bias(&self) -> f32 {
        let left_mean = self
            .muscles
            .iter()
            .filter(|m| m.is_left && m.body_position < 0.12)
            .map(|m| m.contraction)
            .sum::<f32>()
            / self
                .muscles
                .iter()
                .filter(|m| m.is_left && m.body_position < 0.12)
                .count()
                .max(1) as f32;
        let right_mean = self
            .muscles
            .iter()
            .filter(|m| !m.is_left && m.body_position < 0.12)
            .map(|m| m.contraction)
            .sum::<f32>()
            / self
                .muscles
                .iter()
                .filter(|m| !m.is_left && m.body_position < 0.12)
                .count()
                .max(1) as f32;
        left_mean - right_mean
    }

    /// Get summary statistics.
    pub fn stats(&self) -> CelegansStats {
        let mean_activation =
            self.neurons.iter().map(|n| n.activation).sum::<f32>() / self.neurons.len() as f32;
        let mean_contraction =
            self.muscles.iter().map(|m| m.contraction).sum::<f32>() / self.muscles.len() as f32;
        let active_neurons = self.neurons.iter().filter(|n| n.activation > 0.5).count();

        CelegansStats {
            neuron_count: self.neurons.len(),
            synapse_count: self.synapses.len(),
            gap_junction_count: self.gap_junctions.len(),
            muscle_count: self.muscles.len(),
            mean_neuron_activation: mean_activation,
            mean_muscle_contraction: mean_contraction,
            active_neurons,
            locomotion_state: self.locomotion_state(),
            position_um: (self.x_um, self.y_um),
            speed_um_s: self.speed_um_s,
            energy_reserve: self.energy_reserve(),
            gut_content: self.gut_content(),
            pharyngeal_pumping_hz: self.pharyngeal_pumping_hz(),
            age_days: self.age_days,
            alive: self.alive,
        }
    }
}

impl Default for CelegansOrganism {
    fn default() -> Self {
        Self::new()
    }
}

/// Current locomotion state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocomotionState {
    Forward,
    Backward,
    Stationary,
}

/// Summary statistics for a C. elegans organism.
#[derive(Debug, Clone)]
pub struct CelegansStats {
    pub neuron_count: usize,
    pub synapse_count: usize,
    pub gap_junction_count: usize,
    pub muscle_count: usize,
    pub mean_neuron_activation: f32,
    pub mean_muscle_contraction: f32,
    pub active_neurons: usize,
    pub locomotion_state: LocomotionState,
    pub position_um: (f32, f32),
    pub speed_um_s: f32,
    pub energy_reserve: f32,
    pub gut_content: f32,
    pub pharyngeal_pumping_hz: f32,
    pub age_days: f32,
    pub alive: bool,
}

// ── Molecular/DNA Fidelity Layer ─────────────────────────────────────────────

/// Key C. elegans genes involved in neural function and behavior.
/// Based on WormBase annotations and literature.
#[derive(Debug, Clone)]
pub struct CelegansGene {
    /// Gene name (e.g., "unc-47", "cha-1").
    pub name: String,
    /// Chromosome (I, II, III, IV, V, X).
    pub chromosome: u8,
    /// Approximate map position (cM).
    pub map_position: f32,
    /// Molecular function description.
    pub function: GeneFunction,
    /// Whether essential for viability.
    pub essential: bool,
}

/// Molecular function categories for C. elegans genes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneFunction {
    /// Neurotransmitter synthesis (cha-1, unc-17, cat-1, etc.).
    NeurotransmitterSynthesis,
    /// Vesicular transport (unc-104, rab-3, etc.).
    VesicularTransport,
    /// Ion channels (unc-8, mec-4, tax-4, etc.).
    IonChannel,
    /// Synaptic vesicle release (unc-13, snb-1, etc.).
    SynapticRelease,
    /// Mechanosensation (mec-3, mec-7, etc.).
    Mechanosensation,
    /// Chemosensation (odr-10, sra-6, etc.).
    Chemosensation,
    /// Thermosensation (gfx-1, ttx-1, etc.).
    Thermosensation,
    /// Locomotion (unc-54, myo-3, etc.).
    Locomotion,
    /// Egg laying (egl-1, egl-47, etc.).
    EggLaying,
    /// Dauer formation (daf-2, daf-16, etc.).
    DauerFormation,
    /// Aging/lifespan (age-1, clk-1, etc.).
    Aging,
}

impl CelegansGene {
    /// Get the canonical set of important neural genes.
    pub fn neural_genes() -> Vec<Self> {
        vec![
            // Cholinergic genes
            Self {
                name: "cha-1".into(),
                chromosome: 4,
                map_position: 3.5,
                function: GeneFunction::NeurotransmitterSynthesis,
                essential: true,
            },
            Self {
                name: "unc-17".into(),
                chromosome: 4,
                map_position: 3.5,
                function: GeneFunction::VesicularTransport,
                essential: true,
            },
            // GABAergic genes
            Self {
                name: "unc-25".into(),
                chromosome: 1,
                map_position: 6.2,
                function: GeneFunction::NeurotransmitterSynthesis,
                essential: false,
            },
            Self {
                name: "unc-47".into(),
                chromosome: 3,
                map_position: 3.0,
                function: GeneFunction::VesicularTransport,
                essential: false,
            },
            // Serotonergic genes
            Self {
                name: "tph-1".into(),
                chromosome: 5,
                map_position: 10.0,
                function: GeneFunction::NeurotransmitterSynthesis,
                essential: false,
            },
            Self {
                name: "cat-1".into(),
                chromosome: 5,
                map_position: 13.0,
                function: GeneFunction::VesicularTransport,
                essential: false,
            },
            // Dopaminergic genes
            Self {
                name: "cat-2".into(),
                chromosome: 2,
                map_position: 5.0,
                function: GeneFunction::NeurotransmitterSynthesis,
                essential: false,
            },
            Self {
                name: "dat-1".into(),
                chromosome: 5,
                map_position: 17.0,
                function: GeneFunction::NeurotransmitterSynthesis,
                essential: false,
            },
            // Mechanosensory genes
            Self {
                name: "mec-3".into(),
                chromosome: 4,
                map_position: 10.0,
                function: GeneFunction::Mechanosensation,
                essential: false,
            },
            Self {
                name: "mec-4".into(),
                chromosome: 5,
                map_position: 0.0,
                function: GeneFunction::IonChannel,
                essential: false,
            },
            Self {
                name: "mec-7".into(),
                chromosome: 1,
                map_position: 15.0,
                function: GeneFunction::Mechanosensation,
                essential: false,
            },
            // Locomotion genes
            Self {
                name: "unc-54".into(),
                chromosome: 4,
                map_position: 18.0,
                function: GeneFunction::Locomotion,
                essential: true,
            },
            Self {
                name: "myo-3".into(),
                chromosome: 5,
                map_position: 0.0,
                function: GeneFunction::Locomotion,
                essential: true,
            },
            // Dauer genes
            Self {
                name: "daf-2".into(),
                chromosome: 3,
                map_position: 0.0,
                function: GeneFunction::DauerFormation,
                essential: false,
            },
            Self {
                name: "daf-16".into(),
                chromosome: 1,
                map_position: 0.0,
                function: GeneFunction::DauerFormation,
                essential: false,
            },
            // Aging genes
            Self {
                name: "age-1".into(),
                chromosome: 2,
                map_position: 0.0,
                function: GeneFunction::Aging,
                essential: false,
            },
            Self {
                name: "clk-1".into(),
                chromosome: 3,
                map_position: 15.0,
                function: GeneFunction::Aging,
                essential: false,
            },
        ]
    }
}

/// C. elegans neuropeptide families (from WormBase).
/// These modulate synaptic transmission and behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeuropeptideFamily {
    /// Insulin-like peptides (40+ members, regulate dauer, aging).
    InsulinLike,
    /// FMRFamide-like peptides (30+ members, modulate locomotion).
    FMRFamideLike,
    /// NLP family (40+ members, diverse functions).
    NLFamily,
    /// CAPA peptides (egg laying, gut function).
    CAPA,
    /// TGF-beta (dauer, body size).
    TGFBeta,
}

/// Key molecular pathways in C. elegans.
#[derive(Debug, Clone)]
pub struct MolecularPathway {
    pub name: String,
    pub genes: Vec<String>,
    pub description: String,
}

impl MolecularPathway {
    /// Get canonical pathways for C. elegans.
    pub fn canonical_pathways() -> Vec<Self> {
        vec![
            Self {
                name: "Cholinergic Synapse".into(),
                genes: vec![
                    "cha-1".into(),
                    "unc-17".into(),
                    "unc-13".into(),
                    "snb-1".into(),
                ],
                description: "Acetylcholine synthesis and release at neuromuscular junctions"
                    .into(),
            },
            Self {
                name: "GABAergic Inhibition".into(),
                genes: vec!["unc-25".into(), "unc-47".into(), "gab-1".into()],
                description: "GABA synthesis and release for cross-inhibition in locomotion".into(),
            },
            Self {
                name: "Mechanosensory Transduction".into(),
                genes: vec![
                    "mec-3".into(),
                    "mec-4".into(),
                    "mec-7".into(),
                    "mec-10".into(),
                ],
                description: "Touch sensation via DEG/ENaC channels".into(),
            },
            Self {
                name: "Dauer Signaling".into(),
                genes: vec![
                    "daf-2".into(),
                    "daf-16".into(),
                    "daf-12".into(),
                    "daf-9".into(),
                ],
                description: "Insulin/IGF-1 pathway controlling developmental arrest".into(),
            },
            Self {
                name: "Egg Laying Circuit".into(),
                genes: vec![
                    "egl-1".into(),
                    "egl-47".into(),
                    "egl-19".into(),
                    "unc-103".into(),
                ],
                description: "Serotonergic and peptidergic control of egg laying".into(),
            },
            Self {
                name: "Thermotaxis".into(),
                genes: vec![
                    "ttx-1".into(),
                    "tax-4".into(),
                    "gcy-23".into(),
                    "gcy-8".into(),
                ],
                description: "Temperature gradient navigation via AFD neurons".into(),
            },
            Self {
                name: "Chemotaxis".into(),
                genes: vec![
                    "odr-10".into(),
                    "tax-4".into(),
                    "gpa-3".into(),
                    "osm-9".into(),
                ],
                description: "Odorant detection and gradient navigation".into(),
            },
            Self {
                name: "Aging Pathway".into(),
                genes: vec![
                    "age-1".into(),
                    "daf-2".into(),
                    "daf-16".into(),
                    "clk-1".into(),
                ],
                description: "Insulin signaling affecting lifespan".into(),
            },
        ]
    }
}

/// Genomic statistics for C. elegans.
pub mod genome_stats {
    /// Chromosome lengths in megabases.
    pub const CHROMOSOME_LENGTHS_MB: [f32; 6] = [
        15.0, // I
        15.0, // II
        14.0, // III
        18.0, // IV
        21.0, // V
        18.0, // X
    ];

    /// Total base pairs.
    pub const TOTAL_BP: usize = 100_000_000;

    /// Number of protein-coding genes.
    pub const PROTEIN_CODING_GENES: usize = 20_470;

    /// Number of RNA genes.
    pub const RNA_GENES: usize = 25_000;

    /// Number of pseudogenes.
    pub const PSEUDOGENES: usize = 300;

    /// GC content percentage.
    pub const GC_CONTENT_PERCENT: f32 = 35.4;
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_celegans_creation() {
        let worm = CelegansOrganism::new();
        assert_eq!(worm.neurons.len(), CELEGANS_NEURON_COUNT);
        assert_eq!(worm.muscles.len(), CELEGANS_MUSCLE_COUNT);
        assert!(worm.alive);
        assert!(
            !worm.synapses.is_empty(),
            "connectome wiring should be populated"
        );
        assert!(
            worm.muscles.iter().any(|m| !m.innervation.is_empty()),
            "muscles should be wired"
        );
    }

    #[test]
    fn test_molecular_genes() {
        let genes = CelegansGene::neural_genes();
        assert!(genes.len() > 10, "Should have key neural genes");

        // Check for essential genes
        let essential = genes.iter().filter(|g| g.essential).count();
        assert!(essential > 0, "Should have essential genes");
    }

    #[test]
    fn test_molecular_pathways() {
        let pathways = MolecularPathway::canonical_pathways();
        assert!(pathways.len() >= 8, "Should have key pathways");

        // Check dauer pathway exists
        let dauer = pathways.iter().find(|p| p.name.contains("Dauer"));
        assert!(dauer.is_some(), "Should have dauer pathway");
    }

    #[test]
    fn test_genome_stats() {
        assert_eq!(genome_stats::CHROMOSOME_LENGTHS_MB.len(), 6);
        assert!(genome_stats::PROTEIN_CODING_GENES > 20_000);
    }

    #[test]
    fn test_neuron_types() {
        let worm = CelegansOrganism::new();

        // Check we have sensory, interneurons, and motor neurons
        let sensory = worm
            .neurons
            .iter()
            .filter(|n| n.class == NeuronClass::Sensory)
            .count();
        let inter = worm
            .neurons
            .iter()
            .filter(|n| n.class == NeuronClass::Interneuron)
            .count();
        let motor = worm
            .neurons
            .iter()
            .filter(|n| n.class == NeuronClass::Motor)
            .count();

        assert!(sensory > 0, "Should have sensory neurons");
        assert!(inter > 0, "Should have interneurons");
        assert!(motor > 0, "Should have motor neurons");
    }

    #[test]
    fn test_neurotransmitter_distribution() {
        let worm = CelegansOrganism::new();

        let ach = worm
            .neurons
            .iter()
            .filter(|n| n.neurotransmitter == Neurotransmitter::Acetylcholine)
            .count();
        let gaba = worm
            .neurons
            .iter()
            .filter(|n| n.neurotransmitter == Neurotransmitter::GABA)
            .count();

        // Should have both ACh (excitatory) and GABA (inhibitory) motor neurons
        assert!(ach > 0, "Should have cholinergic neurons");
        assert!(gaba > 0, "Should have GABAergic neurons");
    }

    #[test]
    fn test_step_simulation() {
        let mut worm = CelegansOrganism::new();

        // Stimulate touch receptor
        worm.stimulate("ALML", 1.0);

        // Run simulation
        for _ in 0..100 {
            worm.step(1.0); // 1 ms timestep
        }

        // Should have propagated activity
        let stats = worm.stats();
        assert!(stats.mean_neuron_activation > 0.0);
    }

    #[test]
    fn test_forward_locomotion() {
        let mut worm = CelegansOrganism::new();

        // Sustained forward command (the current model does not contain
        // recurrent circuitry that maintains command state, so we re-
        // stimulate every step to model a tonic input).
        for _ in 0..1000 {
            worm.stimulate("AVBL", 1.0);
            worm.stimulate("AVBR", 1.0);
            worm.step(1.0);
        }

        let state = worm.locomotion_state();
        assert_eq!(state, LocomotionState::Forward);
        assert!(
            worm.speed_um_s > 0.0,
            "forward command should drive body speed"
        );
        assert!(
            worm.x_um.abs() > 0.0 || worm.y_um.abs() > 0.0,
            "forward command should move the body"
        );
    }

    #[test]
    fn test_backward_locomotion() {
        let mut worm = CelegansOrganism::new();

        // Sustained backward command (see forward test for rationale).
        for _ in 0..1000 {
            worm.stimulate("AVAL", 1.0);
            worm.stimulate("AVAR", 1.0);
            worm.step(1.0);
        }

        let state = worm.locomotion_state();
        assert_eq!(state, LocomotionState::Backward);
        assert!(
            worm.speed_um_s > 0.0,
            "backward command should still engage muscles"
        );
    }

    #[test]
    fn test_stats() {
        let worm = CelegansOrganism::new();
        let stats = worm.stats();

        assert_eq!(stats.neuron_count, CELEGANS_NEURON_COUNT);
        assert_eq!(stats.muscle_count, CELEGANS_MUSCLE_COUNT);
        assert!(stats.alive);
    }

    #[test]
    fn test_muscle_anatomy() {
        let worm = CelegansOrganism::new();

        let dorsal = worm.muscles.iter().filter(|m| m.is_dorsal).count();
        let ventral = worm.muscles.iter().filter(|m| !m.is_dorsal).count();

        // Should have roughly equal dorsal and ventral muscles
        assert!(dorsal > 0);
        assert!(ventral > 0);
        assert!((dorsal as i32 - ventral as i32).abs() <= 2);
    }

    #[test]
    fn test_anterior_touch_drives_backward_state() {
        let mut worm = CelegansOrganism::new();
        let inputs = CelegansSensoryInputs {
            anterior_touch: 1.0,
            ..Default::default()
        };

        for _ in 0..200 {
            worm.step_with_inputs(1.0, &inputs);
        }

        assert_eq!(worm.locomotion_state(), LocomotionState::Backward);
    }

    #[test]
    fn test_posterior_touch_drives_forward_state() {
        let mut worm = CelegansOrganism::new();
        let inputs = CelegansSensoryInputs {
            posterior_touch: 1.0,
            ..Default::default()
        };

        for _ in 0..200 {
            worm.step_with_inputs(1.0, &inputs);
        }

        assert_eq!(worm.locomotion_state(), LocomotionState::Forward);
    }

    #[test]
    fn test_left_attractant_biases_left_chemosensors() {
        let mut worm = CelegansOrganism::new();
        let inputs = CelegansSensoryInputs {
            attractant_left: 1.0,
            attractant_right: 0.0,
            ..Default::default()
        };

        worm.step_with_inputs(1.0, &inputs);

        assert!(
            worm.get_neuron_activation("ASEL") > worm.get_neuron_activation("ASER"),
            "left attractant should bias left chemosensory drive"
        );
    }

    #[test]
    fn test_immersion_switches_to_swimming_mode() {
        let mut worm = CelegansOrganism::new();
        let inputs = CelegansSensoryInputs {
            immersion: 1.0,
            ..Default::default()
        };

        worm.step_with_inputs(1.0, &inputs);

        assert!(
            !worm.is_crawling,
            "high immersion should switch the worm into swimming mode"
        );
    }

    #[test]
    fn test_food_density_engages_pumping_and_gut_loading() {
        let mut worm = CelegansOrganism::new();
        let start_gut = worm.gut_content();
        let inputs = CelegansSensoryInputs {
            food_density: 1.0,
            ..Default::default()
        };

        for _ in 0..500 {
            worm.step_with_inputs(1.0, &inputs);
        }

        assert!(
            worm.pharyngeal_pumping_hz() > 0.0,
            "food should engage the pharyngeal pumping circuit"
        );
        assert!(
            worm.gut_content() > start_gut,
            "food should increase gut content over time"
        );
    }

    #[test]
    fn test_low_energy_reduces_locomotor_speed() {
        let mut high_energy = CelegansOrganism::new();
        let mut low_energy = CelegansOrganism::new();
        high_energy.physiology.energy_reserve = 1.0;
        low_energy.physiology.energy_reserve = 0.15;

        high_energy.stimulate("AVBL", 1.0);
        high_energy.stimulate("AVBR", 1.0);
        low_energy.stimulate("AVBL", 1.0);
        low_energy.stimulate("AVBR", 1.0);

        for _ in 0..250 {
            high_energy.step(1.0);
            low_energy.step(1.0);
        }

        assert!(
            high_energy.speed_um_s > low_energy.speed_um_s,
            "low energy should suppress locomotor output"
        );
    }

    #[test]
    fn test_gap_junctions_are_not_stored_in_chemical_synapse_pool() {
        let worm = CelegansOrganism::new();

        assert!(
            worm.synapses
                .iter()
                .all(|syn| syn.synapse_type == SynapseType::Chemical),
            "chemical synapse pool should not contain gap-junction entries"
        );
        assert!(
            worm.gap_junctions
                .iter()
                .all(|syn| syn.synapse_type == SynapseType::GapJunction),
            "gap-junction pool should contain only electrical couplings"
        );
    }
}
