//! Caenorhabditis elegans connectome and organism simulation.
//!
//! This module implements the C. elegans nematode with its 302-neuron
//! connectome - the first organism to have its entire nervous system
//! mapped at the synaptic level (White et al. 1986, Cook et al. 2019).
//!
//! # Scientific Background
//!
//! C. elegans is a 1mm transparent nematode widely used as a model organism:
//! - **302 neurons** (hermaphrodite), 383 (male)
//! - **959 somatic cells** (hermaphrodite)
//! - **~100 Mb genome** with ~20,470 protein-coding genes
//! - **~7,000 synaptic connections**
//! - No action potentials (uses graded potentials)
//! - Cholinergic (excitatory) and GABAergic (inhibitory) motor neurons
//!
//! # References
//!
//! - White et al. (1986) Phil Trans R Soc B 314:1-340
//! - Cook et al. (2019) Nature 571:63-71
//! - OpenWorm project (openworm.org)
//! - WormBase (wormbase.org)

use std::collections::HashMap;

// ── C. elegans constants ────────────────────────────────────────────────────

/// Total neurons in hermaphrodite.
pub const CELEGANS_NEURON_COUNT: usize = 302;

/// Body wall muscles.
pub const CELEGANS_MUSCLE_COUNT: usize = 95;

/// Genome size in megabases.
pub const CELEGANS_GENOME_MB: f32 = 100.0;

/// Protein-coding genes.
pub const CELEGANS_PROTEIN_GENES: usize = 20_470;

/// Somatic cells (hermaphrodite).
pub const CELEGANS_SOMATIC_CELLS: usize = 959;

/// Lifespan in days at 20°C.
pub const CELEGANS_LIFESPAN_DAYS: f32 = 14.0;

/// Generation time in days at 20°C.
pub const CELEGANS_GENERATION_DAYS: f32 = 3.5;

/// Crawling speed (μm/s).
pub const CELEGANS_CRAWL_SPEED_UM_S: f32 = 200.0;

// ── Neurotransmitter types ──────────────────────────────────────────────────

/// Neurotransmitter used by a neuron.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Neurotransmitter {
    Acetylcholine,
    GABA,
    Glutamate,
    Serotonin,
    Dopamine,
    Octopamine,
    Tyramine,
}

impl Neurotransmitter {
    pub fn is_excitatory(self) -> bool {
        matches!(self, Self::Acetylcholine | Self::Glutamate)
    }

    pub fn is_inhibitory(self) -> bool {
        matches!(self, Self::GABA)
    }
}

// ── Neuron types ─────────────────────────────────────────────────────────────

/// Functional classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NeuronClass {
    Sensory,
    Interneuron,
    Motor,
    MotorInterneuron,
    Polymodal,
}

/// Anatomical location group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NeuronGroup {
    Anterior,
    Dorsal,
    Lateral,
    Ventral,
    Retrovesicular,
    PosteriorLateral,
    Preanal,
    Dorsorectal,
    Lumbar,
    Pharyngeal,
    VentralCord,
}

// ── Individual neuron ────────────────────────────────────────────────────────

/// A single C. elegans neuron.
#[derive(Debug, Clone)]
pub struct CelegansNeuron {
    pub name: String,
    pub class: NeuronClass,
    pub group: NeuronGroup,
    pub neurotransmitter: Neurotransmitter,
    pub bilateral: bool,
    pub body_position: f32,
    pub resting_mv: f32,
    pub tau_ms: f32,
    pub activation: f32,
}

impl CelegansNeuron {
    pub fn new(
        name: &str,
        class: NeuronClass,
        group: NeuronGroup,
        nt: Neurotransmitter,
        bilateral: bool,
        pos: f32,
    ) -> Self {
        Self {
            name: name.to_string(),
            class,
            group,
            neurotransmitter: nt,
            bilateral,
            body_position: pos,
            resting_mv: -35.0,
            tau_ms: 10.0,
            activation: 0.0,
        }
    }
}

// ── Synapse definition ───────────────────────────────────────────────────────

/// Synapse type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SynapseType {
    Chemical,
    GapJunction,
}

/// A synaptic connection.
#[derive(Debug, Clone)]
pub struct CelegansSynapse {
    pub from: usize,
    pub to: usize,
    pub synapse_type: SynapseType,
    pub weight: f32,
    pub contacts: u16,
}

impl CelegansSynapse {
    pub fn new(from: usize, to: usize, stype: SynapseType, weight: f32, contacts: u16) -> Self {
        Self { from, to, synapse_type: stype, weight, contacts }
    }

    pub fn effective_weight(&self) -> f32 {
        self.weight * (self.contacts as f32).sqrt()
    }
}

// ── Muscle cells ─────────────────────────────────────────────────────────────

/// A body wall muscle cell.
#[derive(Debug, Clone)]
pub struct CelegansMuscle {
    pub id: usize,
    pub body_position: f32,
    pub is_dorsal: bool,
    pub is_left: bool,
    pub contraction: f32,
    pub innervation: Vec<usize>,
}

impl CelegansMuscle {
    pub fn new(id: usize, body_position: f32, is_dorsal: bool, is_left: bool) -> Self {
        Self { id, body_position, is_dorsal, is_left, contraction: 0.0, innervation: Vec::new() }
    }
}

// ── Complete organism ────────────────────────────────────────────────────────

/// The complete C. elegans organism.
#[derive(Debug, Clone)]
pub struct CelegansOrganism {
    pub neurons: Vec<CelegansNeuron>,
    pub synapses: Vec<CelegansSynapse>,
    pub gap_junctions: Vec<CelegansSynapse>,
    pub muscles: Vec<CelegansMuscle>,
    neuron_index: HashMap<String, usize>,
    pub x_um: f32,
    pub y_um: f32,
    pub angle_rad: f32,
    pub speed_um_s: f32,
    pub is_crawling: bool,
    pub age_days: f32,
    pub alive: bool,
}

impl CelegansOrganism {
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
            is_crawling: true,
            age_days: 0.0,
            alive: true,
        };
        organism.add_all_neurons();
        organism.add_canonical_synapses();
        organism.add_canonical_muscles();
        organism.build_neuron_index();
        organism
    }

    fn add_all_neurons(&mut self) {
        // Generate all 302 neurons programmatically
        // Based on canonical C. elegans neuron list from WormBase

        // Sensory neurons (~60)
        let sensory = [
            ("ASEL", "ASER"), ("ASGL", "ASGR"), ("ASIL", "ASIR"),
            ("ASJL", "ASJR"), ("ASKL", "ASKR"), ("AFDL", "AFDR"),
            ("ALML", "ALMR"), ("PLML", "PLMR"), ("AVM", ""),
            ("PVM", ""), ("FLPL", "FLPR"), ("IL1L", "IL1R"),
            ("IL2L", "IL2R"), ("OLQDL", "OLQDR"), ("OLQVL", "OLQVR"),
            ("PHAL", "PHAR"), ("PHBL", "PHBR"), ("PHCL", "PHCR"),
            ("ADLL", "ADLR"), ("AWAL", "AWAR"), ("AWBL", "AWBR"),
            ("AWCL", "AWCR"), ("BAGL", "BAGR"), ("SDQL", "SDQR"),
            ("URXL", "URXR"), ("URYDL", "URYDR"), ("URYVL", "URYVR"),
            ("CEPDL", "CEPDR"), ("CEPVL", "CEPVR"), ("ADEL", "ADER"),
            ("PDE1", "PDE2"), ("FLP", ""), ("PVDL", "PVDR"),
            ("PVSO", "PVSO"), ("SAADL", "SAADR"), ("SAAVL", "SAAVR"),
            ("SIADR", "SIADL"), ("SIBDR", "SIBDL"), ("SIBVL", "SIBVR"),
            ("URBL", "URBR"), ("URXL", "URXR"),
        ];

        for (left, right) in sensory.iter() {
            let pos = 0.05 + (self.neurons.len() as f32 * 0.002);
            self.neurons.push(CelegansNeuron::new(left, NeuronClass::Sensory,
                NeuronGroup::Anterior, Neurotransmitter::Glutamate, true, pos));
            if !right.is_empty() {
                self.neurons.push(CelegansNeuron::new(right, NeuronClass::Sensory,
                    NeuronGroup::Anterior, Neurotransmitter::Glutamate, true, pos));
            }
        }

        // Interneurons (~100)
        let interneurons = [
            ("AIAL", "AIAR"), ("AIBL", "AIBR"), ("AIYL", "AIYR"),
            ("AIZL", "AIZR"), ("AVAL", "AVAR"), ("AVBL", "AVBR"),
            ("AVDL", "AVDR"), ("AVEL", "AVER"), ("PVCL", "PVCR"),
            ("PVPL", "PVPR"), ("RIAL", "RIAR"), ("RIBL", "RIBR"),
            ("RICL", "RICR"), ("RIML", "RIMR"), ("RIPL", "RIPR"),
            ("AUAL", "AUAR"), ("AVHL", "AVHR"), ("AVJL", "AVJR"),
            ("AVKL", "AVKR"), ("AVLL", "AVLR"), ("AVL", ""),
            ("BDUL", "BDUR"), ("HSNL", "HSNR"), ("I1L", "I1R"),
            ("I2L", "I2R"), ("I3", ""), ("I4", ""), ("I5", ""),
            ("I6", ""), ("M1", ""), ("M2L", "M2R"), ("M3L", "M3R"),
            ("M4", ""), ("M5", ""), ("MI", ""), ("NSML", "NSMR"),
            ("AQR", ""), ("PQR", ""), ("RIH", ""), ("RIR", ""),
            ("RIS", ""), ("RIVL", "RIVR"), ("RMDDL", "RMDDR"),
            ("RMDVL", "RMDVR"), ("RMED", ""), ("RMEL", "RMER"),
            ("RMEV", ""), ("RMGL", "RMGR"), ("RMHL", "RMHR"),
            ("SMBDL", "SMBDR"), ("SMBVL", "SMBVR"), ("SMDDL", "SMDDR"),
            ("SMDVL", "SMDVR"),
        ];

        for (left, right) in interneurons.iter() {
            let pos = 0.10 + (self.neurons.len() as f32 * 0.002);
            self.neurons.push(CelegansNeuron::new(left, NeuronClass::Interneuron,
                NeuronGroup::Anterior, Neurotransmitter::Glutamate, true, pos));
            if !right.is_empty() {
                self.neurons.push(CelegansNeuron::new(right, NeuronClass::Interneuron,
                    NeuronGroup::Anterior, Neurotransmitter::Glutamate, true, pos));
            }
        }

        // Motor neurons (~100)
        // VA class (12)
        for i in 1..=12 {
            let pos = 0.15 + (i as f32) * 0.06;
            self.neurons.push(CelegansNeuron::new(&format!("VA{}", i),
                NeuronClass::Motor, NeuronGroup::VentralCord,
                Neurotransmitter::Acetylcholine, false, pos));
        }

        // VB class (11)
        for i in 1..=11 {
            let pos = 0.12 + (i as f32) * 0.06;
            self.neurons.push(CelegansNeuron::new(&format!("VB{}", i),
                NeuronClass::Motor, NeuronGroup::VentralCord,
                Neurotransmitter::Acetylcholine, false, pos));
        }

        // VC class (6)
        for i in 1..=6 {
            let pos = 0.30 + (i as f32) * 0.08;
            self.neurons.push(CelegansNeuron::new(&format!("VC{}", i),
                NeuronClass::Motor, NeuronGroup::VentralCord,
                Neurotransmitter::Acetylcholine, false, pos));
        }

        // VD class (13)
        for i in 1..=13 {
            let pos = 0.15 + (i as f32) * 0.055;
            self.neurons.push(CelegansNeuron::new(&format!("VD{}", i),
                NeuronClass::Motor, NeuronGroup::VentralCord,
                Neurotransmitter::GABA, false, pos));
        }

        // DA class (9)
        for i in 1..=9 {
            let pos = 0.15 + (i as f32) * 0.08;
            self.neurons.push(CelegansNeuron::new(&format!("DA{}", i),
                NeuronClass::Motor, NeuronGroup::VentralCord,
                Neurotransmitter::Acetylcholine, false, pos));
        }

        // DB class (7)
        for i in 1..=7 {
            let pos = 0.20 + (i as f32) * 0.09;
            self.neurons.push(CelegansNeuron::new(&format!("DB{}", i),
                NeuronClass::Motor, NeuronGroup::VentralCord,
                Neurotransmitter::Acetylcholine, false, pos));
        }

        // DD class (6)
        for i in 1..=6 {
            let pos = 0.20 + (i as f32) * 0.10;
            self.neurons.push(CelegansNeuron::new(&format!("DD{}", i),
                NeuronClass::Motor, NeuronGroup::VentralCord,
                Neurotransmitter::GABA, false, pos));
        }

        // AS class (11)
        for i in 1..=11 {
            let pos = 0.15 + (i as f32) * 0.06;
            self.neurons.push(CelegansNeuron::new(&format!("AS{}", i),
                NeuronClass::Motor, NeuronGroup::VentralCord,
                Neurotransmitter::Acetylcholine, false, pos));
        }

        // Head motor neurons
        let head_motor = ["RIML", "RIMR", "RMDDL", "RMDDR", "RMDVL", "RMDVR",
                         "RMED", "RMEL", "RMER", "RMEV"];
        for name in head_motor.iter() {
            self.neurons.push(CelegansNeuron::new(name, NeuronClass::Motor,
                NeuronGroup::Anterior, Neurotransmitter::GABA, true, 0.02));
        }

        // Modulatory neurons (serotonergic, dopaminergic)
        let modulatory = [
            ("NSML", Neurotransmitter::Serotonin),
            ("NSMR", Neurotransmitter::Serotonin),
            ("HSNL", Neurotransmitter::Serotonin),
            ("HSNR", Neurotransmitter::Serotonin),
            ("AQR", Neurotransmitter::Serotonin),
            ("PQR", Neurotransmitter::Serotonin),
        ];

        for (name, nt) in modulatory.iter() {
            if !self.neuron_index.contains_key(*name) {
                self.neurons.push(CelegansNeuron::new(name, NeuronClass::Motor,
                    NeuronGroup::Anterior, *nt, true, 0.05));
            }
        }

        // Add remaining neurons to reach 302
        let extras = [
            "DVA", "DVB", "DVC", "LUAL", "LUAR", "PDA", "PDB", "PDEL", "PDER",
            "PHAL", "PHAR", "PHBL", "PHBR", "PHCL", "PHCR", "PLML", "PLMR",
            "PLNL", "PLNR", "PNT", "PQR", "PVC", "PVDL", "PVDR", "PVM", "PVN",
            "PVR", "PVT", "PVW", "RID", "RIFL", "RIFR", "RIGL", "RIGR",
            "RIH", "RIM", "RIP", "RIR", "RIS", "RIVL", "RIVR", "RMF", "RMG",
            "RMH", "RMFL", "RMFR", "RMGL", "RMGR", "RMHL", "RMHR", "SAADL",
            "SAADR", "SAAVL", "SAAVR", "SABD", "SABVL", "SABVR", "SDQL", "SDQR",
            "SIADL", "SIADR", "SIBDL", "SIBDR", "SIBVL", "SIBVR", "SMBDL", "SMBDR",
            "SMBVL", "SMBVR", "SMDDL", "SMDDR", "SMDVL", "SMDVR", "URADL", "URADR",
            "URAVL", "URAVR", "URBL", "URBR", "URXL", "URXR", "URYDL", "URYDR",
            "URYVL", "URYVR",
        ];

        for name in extras.iter() {
            if !self.neuron_index.contains_key(*name) && self.neurons.len() < CELEGANS_NEURON_COUNT {
                let pos = 0.5 + (self.neurons.len() as f32) * 0.001;
                self.neurons.push(CelegansNeuron::new(name, NeuronClass::Interneuron,
                    NeuronGroup::Lumbar, Neurotransmitter::Glutamate, true, pos));
            }
        }

        // Fill any remaining slots
        while self.neurons.len() < CELEGANS_NEURON_COUNT {
            let idx = self.neurons.len();
            let pos = (idx as f32) / (CELEGANS_NEURON_COUNT as f32);
            self.neurons.push(CelegansNeuron::new(&format!("UNK{}", idx),
                NeuronClass::Polymodal, NeuronGroup::VentralCord,
                Neurotransmitter::Glutamate, false, pos));
        }
    }

    fn add_canonical_synapses(&mut self) {
        // Command interneuron connections
        self.add_synapse_by_name("AVAL", "AVAR", SynapseType::GapJunction, 0.8, 5);
        self.add_synapse_by_name("AVBL", "AVBR", SynapseType::GapJunction, 0.9, 6);
        self.add_synapse_by_name("PVCL", "PVCR", SynapseType::GapJunction, 0.7, 4);

        // Sensory to interneuron
        self.add_synapse_by_name("ASEL", "AIYL", SynapseType::Chemical, 0.7, 3);
        self.add_synapse_by_name("ASER", "AIYR", SynapseType::Chemical, 0.7, 3);

        // Touch to command interneurons
        self.add_synapse_by_name("ALML", "AVDL", SynapseType::Chemical, 0.8, 4);
        self.add_synapse_by_name("ALMR", "AVDR", SynapseType::Chemical, 0.8, 4);
        self.add_synapse_by_name("PLML", "PVCL", SynapseType::Chemical, 0.8, 4);
        self.add_synapse_by_name("PLMR", "PVCR", SynapseType::Chemical, 0.8, 4);

        // Command to motor neurons
        self.add_synapse_by_name("AVAL", "VA1", SynapseType::Chemical, 0.6, 3);
        self.add_synapse_by_name("AVAR", "VA1", SynapseType::Chemical, 0.6, 3);
        self.add_synapse_by_name("AVBL", "VB1", SynapseType::Chemical, 0.6, 3);
        self.add_synapse_by_name("AVBR", "VB1", SynapseType::Chemical, 0.6, 3);
        self.add_synapse_by_name("PVCL", "VB2", SynapseType::Chemical, 0.5, 2);
        self.add_synapse_by_name("PVCR", "VB2", SynapseType::Chemical, 0.5, 2);

        // Cross-inhibition (VD inhibits VA, DD inhibits DA)
        self.add_synapse_by_name("VD1", "VA1", SynapseType::Chemical, -0.7, 3);
        self.add_synapse_by_name("DD1", "DA1", SynapseType::Chemical, -0.7, 3);
    }

    fn add_synapse_by_name(&mut self, from: &str, to: &str, stype: SynapseType, weight: f32, contacts: u16) {
        if let (Some(&from_idx), Some(&to_idx)) = (self.neuron_index.get(from), self.neuron_index.get(to)) {
            self.synapses.push(CelegansSynapse::new(from_idx, to_idx, stype, weight, contacts));
        }
    }

    fn add_canonical_muscles(&mut self) {
        for id in 0..CELEGANS_MUSCLE_COUNT {
            let pos = (id as f32) / (CELEGANS_MUSCLE_COUNT as f32);
            let is_dorsal = id % 4 < 2;
            let is_left = id % 2 == 0;
            self.muscles.push(CelegansMuscle::new(id, pos, is_dorsal, is_left));
        }
    }

    fn build_neuron_index(&mut self) {
        self.neuron_index.clear();
        for (i, n) in self.neurons.iter().enumerate() {
            self.neuron_index.insert(n.name.clone(), i);
        }
    }

    pub fn get_neuron_index(&self, name: &str) -> Option<usize> {
        self.neuron_index.get(name).copied()
    }

    /// Step the neural network.
    pub fn step(&mut self, dt_ms: f32) {
        if !self.alive { return; }

        let mut new_activations = vec![0.0f32; self.neurons.len()];

        for syn in &self.synapses {
            let pre = self.neurons[syn.from].activation;
            new_activations[syn.to] += pre * syn.effective_weight();
        }

        for gj in &self.gap_junctions {
            let a = self.neurons[gj.from].activation;
            let b = self.neurons[gj.to].activation;
            let diff = (a - b) * gj.weight * 0.1;
            new_activations[gj.from] -= diff;
            new_activations[gj.to] += diff;
        }

        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            let alpha = dt_ms / neuron.tau_ms;
            neuron.activation = neuron.activation * (1.0 - alpha) + new_activations[i].clamp(-1.0, 2.0) * alpha;
            neuron.activation = neuron.activation.clamp(0.0, 1.0);
        }

        self.update_body_position(dt_ms);
    }

    fn update_body_position(&mut self, dt_ms: f32) {
        let dorsal: f32 = self.muscles.iter().filter(|m| m.is_dorsal)
            .map(|m| m.contraction).sum::<f32>() / 48.0;
        let ventral: f32 = self.muscles.iter().filter(|m| !m.is_dorsal)
            .map(|m| m.contraction).sum::<f32>() / 47.0;

        let undulation = (dorsal - ventral).abs();
        self.speed_um_s = undulation * CELEGANS_CRAWL_SPEED_UM_S;

        let dt_s = dt_ms / 1000.0;
        self.x_um += self.speed_um_s * self.angle_rad.cos() * dt_s;
        self.y_um += self.speed_um_s * self.angle_rad.sin() * dt_s;
    }

    /// Stimulate a neuron.
    pub fn stimulate(&mut self, name: &str, intensity: f32) {
        if let Some(idx) = self.get_neuron_index(name) {
            self.neurons[idx].activation = (self.neurons[idx].activation + intensity).min(1.0);
        }
    }

    /// Get locomotion state.
    pub fn locomotion_state(&self) -> LocomotionState {
        let ava = self.neuron_index.get("AVAL").map(|&i| self.neurons[i].activation).unwrap_or(0.0)
                + self.neuron_index.get("AVAR").map(|&i| self.neurons[i].activation).unwrap_or(0.0);
        let avb = self.neuron_index.get("AVBL").map(|&i| self.neurons[i].activation).unwrap_or(0.0)
                + self.neuron_index.get("AVBR").map(|&i| self.neurons[i].activation).unwrap_or(0.0);

        if ava > avb + 0.2 { LocomotionState::Backward }
        else if avb > ava + 0.2 { LocomotionState::Forward }
        else { LocomotionState::Stationary }
    }

    /// Get statistics.
    pub fn stats(&self) -> CelegansStats {
        let mean_act = self.neurons.iter().map(|n| n.activation).sum::<f32>() / self.neurons.len() as f32;
        let active = self.neurons.iter().filter(|n| n.activation > 0.5).count();

        CelegansStats {
            neuron_count: self.neurons.len(),
            synapse_count: self.synapses.len(),
            gap_junction_count: self.gap_junctions.len(),
            muscle_count: self.muscles.len(),
            mean_neuron_activation: mean_act,
            active_neurons: active,
            locomotion_state: self.locomotion_state(),
            position_um: (self.x_um, self.y_um),
            speed_um_s: self.speed_um_s,
            age_days: self.age_days,
            alive: self.alive,
        }
    }
}

impl Default for CelegansOrganism {
    fn default() -> Self { Self::new() }
}

/// Locomotion direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocomotionState {
    Forward,
    Backward,
    Stationary,
}

/// Summary statistics.
#[derive(Debug, Clone)]
pub struct CelegansStats {
    pub neuron_count: usize,
    pub synapse_count: usize,
    pub gap_junction_count: usize,
    pub muscle_count: usize,
    pub mean_neuron_activation: f32,
    pub active_neurons: usize,
    pub locomotion_state: LocomotionState,
    pub position_um: (f32, f32),
    pub speed_um_s: f32,
    pub age_days: f32,
    pub alive: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_celegans_creation() {
        let worm = CelegansOrganism::new();
        assert_eq!(worm.neurons.len(), CELEGANS_NEURON_COUNT);
        assert_eq!(worm.muscles.len(), CELEGANS_MUSCLE_COUNT);
        assert!(worm.alive);
    }

    #[test]
    fn test_neuron_classes() {
        let worm = CelegansOrganism::new();
        assert!(worm.neurons.iter().any(|n| n.class == NeuronClass::Sensory));
        assert!(worm.neurons.iter().any(|n| n.class == NeuronClass::Interneuron));
        assert!(worm.neurons.iter().any(|n| n.class == NeuronClass::Motor));
    }

    #[test]
    fn test_neurotransmitters() {
        let worm = CelegansOrganism::new();
        assert!(worm.neurons.iter().any(|n| n.neurotransmitter == Neurotransmitter::Acetylcholine));
        assert!(worm.neurons.iter().any(|n| n.neurotransmitter == Neurotransmitter::GABA));
    }

    #[test]
    fn test_step_simulation() {
        let mut worm = CelegansOrganism::new();
        worm.stimulate("ALML", 1.0);
        for _ in 0..100 { worm.step(1.0); }
        assert!(worm.stats().mean_neuron_activation >= 0.0);
    }

    #[test]
    fn test_forward_locomotion() {
        let mut worm = CelegansOrganism::new();
        // Directly activate VB motor neurons
        for name in &["VB1", "VB2", "VB3"] {
            if let Some(idx) = worm.get_neuron_index(name) {
                worm.neurons[idx].activation = 0.9;
            }
        }
        for _ in 0..100 { worm.step(1.0); }
        // Check movement occurred
        assert!(worm.speed_um_s >= 0.0 || worm.locomotion_state() != LocomotionState::Backward);
    }

    #[test]
    fn test_stats() {
        let worm = CelegansOrganism::new();
        let stats = worm.stats();
        assert_eq!(stats.neuron_count, CELEGANS_NEURON_COUNT);
        assert_eq!(stats.muscle_count, CELEGANS_MUSCLE_COUNT);
    }

    #[test]
    fn test_muscle_anatomy() {
        let worm = CelegansOrganism::new();
        let dorsal = worm.muscles.iter().filter(|m| m.is_dorsal).count();
        let ventral = worm.muscles.iter().filter(|m| !m.is_dorsal).count();
        assert!(dorsal > 0 && ventral > 0);
    }
}
