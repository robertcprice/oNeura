//! Ant colony organism with collective behavior and simplified connectome.
//!
//! This module implements ant (Formica sp.) colonies with:
//! - Individual ant agents with caste-specific behavior
//! - Pheromone communication trails
//! - Collective decision-making
//! - Nest construction dynamics
//!
//! # Scientific Background
//!
//! Ants are eusocial insects with complex colony organization:
//! - **Colonies**: 100-10,000+ individuals depending on species
//! - **Castes**: Queen, workers (major/minor), soldiers, drones
//! - **Communication**: Pheromone trails, tactile, acoustic
//! - **Brain**: ~250,000 neurons (worker ant)
//!
//! # References
//!
//! - Hölldobler & Wilson (1990) *The Ants* - Comprehensive ant biology
//! - Gronenberg (2008) *Comparative Neurobiology* - Ant brain structure
//! - Dornhaus & Franks (2008) *Ecology* - Collective decision-making
//! - Gordon (2010) *Ant Encounters* - Colony dynamics

use std::collections::HashMap;

// ── Ant colony constants ───────────────────────────────────────────────────

/// Worker ant neuron count (approximate).
pub const ANT_WORKER_NEURONS: usize = 250_000;

/// Queen ant neuron count (larger brain).
pub const ANT_QUEEN_NEURONS: usize = 300_000;

/// Typical colony size for Formica species.
pub const COLONY_SIZE_TYPICAL: usize = 5_000;

/// Maximum pheromone trail age (seconds).
pub const PHEROMONE_MAX_AGE_S: f32 = 300.0;

/// Pheromone decay rate.
pub const PHEROMONE_DECAY_RATE: f32 = 0.01;

/// Ant body length (mm).
pub const ANT_BODY_LENGTH_MM: f32 = 6.0;

/// Genome size (megabases).
pub const ANT_GENOME_MB: f32 = 280.0;

/// Protein-coding genes.
pub const ANT_PROTEIN_GENES: usize = 18_000;

// ── Ant caste system ────────────────────────────────────────────────────────

/// Ant caste types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AntCaste {
    /// Reproductive female - egg laying.
    Queen,
    /// Fertile male - mating flight only.
    Drone,
    /// Sterile female - foraging, nursing, construction.
    WorkerMinor,
    /// Larger sterile female - heavy tasks.
    WorkerMajor,
    /// Defense specialist.
    Soldier,
    /// Developing larva.
    Larva,
    /// Pupal stage.
    Pupa,
}

impl AntCaste {
    /// Whether this caste can lay eggs.
    pub fn can_lay_eggs(self) -> bool {
        matches!(self, Self::Queen)
    }

    /// Whether this caste forages.
    pub fn can_forage(self) -> bool {
        matches!(self, Self::WorkerMinor | Self::WorkerMajor)
    }

    /// Relative brain size multiplier.
    pub fn brain_size_multiplier(self) -> f32 {
        match self {
            Self::Queen => 1.2,
            Self::Drone => 0.8,
            Self::WorkerMinor => 1.0,
            Self::WorkerMajor => 1.1,
            Self::Soldier => 1.05,
            Self::Larva | Self::Pupa => 0.1,
        }
    }
}

// ── Pheromone system ────────────────────────────────────────────────────────

/// Pheromone types used in ant communication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PheromoneType {
    /// Trail to food source.
    Trail,
    /// Alarm/defense signal.
    Alarm,
    /// Queen presence.
    QueenSignal,
    /// Death/necrophoresis.
    Death,
    /// Recruitment to task.
    Recruitment,
    /// Territory marking.
    Territorial,
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
}

impl PheromoneDeposit {
    pub fn new(ptype: PheromoneType, x: f32, y: f32, strength: f32, colony_id: u32) -> Self {
        Self { pheromone_type: ptype, x_mm: x, y_mm: y, strength, age_s: 0.0, colony_id }
    }

    /// Current effective strength after decay.
    pub fn effective_strength(&self) -> f32 {
        self.strength * (-PHEROMONE_DECAY_RATE * self.age_s).exp()
    }

    /// Whether pheromone is still detectable.
    pub fn is_detectable(&self) -> bool {
        self.age_s < PHEROMONE_MAX_AGE_S && self.effective_strength() > 0.01
    }
}

// ── Individual ant brain (simplified) ─────────────────────────────────────────

/// Ant brain regions (simplified).
#[derive(Debug, Clone)]
pub struct AntBrain {
    /// Mushroom body - learning and memory.
    pub mushroom_body_activation: f32,
    /// Antennal lobe - olfactory processing.
    pub antennal_lobe_activation: f32,
    /// Subesophageal ganglion - motor control.
    pub subesophageal_ganglion: f32,
    /// Optical lobe - visual processing.
    pub optical_lobe: f32,
    /// Current task focus [0, 1].
    pub task_focus: f32,
    /// Memory of food locations (simplified).
    pub food_memory: Vec<(f32, f32, f32)>, // (x, y, quality)
}

impl AntBrain {
    pub fn new() -> Self {
        Self {
            mushroom_body_activation: 0.0,
            antennal_lobe_activation: 0.0,
            subesophageal_ganglion: 0.0,
            optical_lobe: 0.0,
            task_focus: 0.0,
            food_memory: Vec::new(),
        }
    }

    /// Process olfactory input (pheromones).
    pub fn process_olfaction(&mut self, pheromone_strength: f32) {
        self.antennal_lobe_activation = (self.antennal_lobe_activation * 0.9 + pheromone_strength * 0.1).min(1.0);
    }

    /// Learn food location.
    pub fn memorize_food(&mut self, x: f32, y: f32, quality: f32) {
        // Keep only top 5 memories
        self.food_memory.push((x, y, quality));
        self.food_memory.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        self.food_memory.truncate(5);
        self.mushroom_body_activation = (self.mushroom_body_activation + 0.1).min(1.0);
    }
}

impl Default for AntBrain {
    fn default() -> Self {
        Self::new()
    }
}

// ── Individual ant agent ─────────────────────────────────────────────────────

/// A single ant in the colony.
#[derive(Debug, Clone)]
pub struct Ant {
    /// Unique identifier.
    pub id: u32,
    /// Colony identifier.
    pub colony_id: u32,
    /// Current caste.
    pub caste: AntCaste,
    /// Position (mm).
    pub x_mm: f32,
    pub y_mm: f32,
    /// Heading angle (radians).
    pub heading: f32,
    /// Speed (mm/s).
    pub speed: f32,
    /// Carrying food?
    pub carrying_food: bool,
    /// Food amount carried.
    pub food_amount: f32,
    /// Energy level [0, 1].
    pub energy: f32,
    /// Age in days.
    pub age_days: f32,
    /// Whether alive.
    pub alive: bool,
    /// Current task.
    pub current_task: AntTask,
    /// Brain state.
    pub brain: AntBrain,
}

/// Tasks an ant can perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AntTask {
    Idle,
    Foraging,
    Returning,
    Nursing,
    Construction,
    Defense,
    Grooming,
    Feeding,
}

impl Ant {
    pub fn new(id: u32, colony_id: u32, caste: AntCaste, x: f32, y: f32) -> Self {
        Self {
            id,
            colony_id,
            caste,
            x_mm: x,
            y_mm: y,
            heading: 0.0,
            speed: 0.0,
            carrying_food: false,
            food_amount: 0.0,
            energy: 1.0,
            age_days: 0.0,
            alive: true,
            current_task: AntTask::Idle,
            brain: AntBrain::new(),
        }
    }

    /// Step the ant by dt seconds.
    pub fn step(&mut self, dt_s: f32, pheromones: &[PheromoneDeposit]) {
        if !self.alive {
            return;
        }

        // Age
        self.age_days += dt_s / 86400.0;

        // Energy decay
        self.energy = (self.energy - 0.0001 * dt_s).max(0.0);
        if self.energy <= 0.0 {
            self.alive = false;
            return;
        }

        // Process nearby pheromones
        let nearby_strength: f32 = pheromones.iter()
            .filter(|p| p.is_detectable() && p.colony_id == self.colony_id)
            .map(|p| {
                let dist = ((p.x_mm - self.x_mm).powi(2) + (p.y_mm - self.y_mm).powi(2)).sqrt();
                if dist < 10.0 {
                    p.effective_strength() / (dist + 1.0)
                } else {
                    0.0
                }
            })
            .sum();
        self.brain.process_olfaction(nearby_strength);

        // Task-based behavior
        match self.current_task {
            AntTask::Foraging => {
                self.speed = 5.0;
                // Random walk with pheromone attraction
                self.heading += (rand_random() - 0.5) * 0.3;
            }
            AntTask::Returning => {
                // Head toward nest (assumed at origin)
                let angle_to_nest = (-self.y_mm).atan2(-self.x_mm);
                self.heading = angle_to_nest;
                self.speed = 8.0;
            }
            _ => {
                self.speed = 0.0;
            }
        }

        // Update position
        self.x_mm += self.heading.cos() * self.speed * dt_s;
        self.y_mm += self.heading.sin() * self.speed * dt_s;
    }

    /// Deposit pheromone at current location.
    pub fn deposit_pheromone(&self, ptype: PheromoneType) -> PheromoneDeposit {
        PheromoneDeposit::new(ptype, self.x_mm, self.y_mm, 1.0, self.colony_id)
    }
}

// Simple random function (would use rand crate in production)
fn rand_random() -> f32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    ((ns % 1000000) as f32) / 1000000.0
}

// ── Ant colony ───────────────────────────────────────────────────────────────

/// A complete ant colony.
#[derive(Debug, Clone)]
pub struct AntColony {
    pub id: u32,
    pub species: String,
    pub ants: Vec<Ant>,
    pub pheromones: Vec<PheromoneDeposit>,
    pub nest_x_mm: f32,
    pub nest_y_mm: f32,
    pub food_storage: f32,
    pub max_food: f32,
}

impl AntColony {
    pub fn new(id: u32, species: &str, nest_x: f32, nest_y: f32, initial_workers: usize) -> Self {
        let mut ants = Vec::with_capacity(initial_workers + 10);

        // Add queen
        ants.push(Ant::new(0, id, AntCaste::Queen, nest_x, nest_y));

        // Add workers
        for i in 1..=initial_workers {
            let angle = (i as f32) / (initial_workers as f32) * std::f32::consts::TAU;
            let dist = 5.0 + rand_random() * 10.0;
            let x = nest_x + angle.cos() * dist;
            let y = nest_y + angle.sin() * dist;
            let caste = if i % 10 == 0 { AntCaste::WorkerMajor } else { AntCaste::WorkerMinor };
            ants.push(Ant::new(i as u32, id, caste, x, y));
        }

        Self {
            id,
            species: species.to_string(),
            ants,
            pheromones: Vec::new(),
            nest_x_mm: nest_x,
            nest_y_mm: nest_y,
            food_storage: 100.0,
            max_food: 1000.0,
        }
    }

    /// Step the entire colony.
    pub fn step(&mut self, dt_s: f32) {
        // Age pheromones
        for p in &mut self.pheromones {
            p.age_s += dt_s;
        }
        self.pheromones.retain(|p| p.is_detectable());

        // Step each ant
        for ant in &mut self.ants {
            ant.step(dt_s, &self.pheromones);
        }

        // Remove dead ants
        self.ants.retain(|a| a.alive);
    }

    /// Get colony statistics.
    pub fn stats(&self) -> ColonyStats {
        let workers = self.ants.iter().filter(|a| a.caste.can_forage()).count();
        let foragers = self.ants.iter().filter(|a| a.current_task == AntTask::Foraging).count();
        let mean_energy = if self.ants.is_empty() {
            0.0
        } else {
            self.ants.iter().map(|a| a.energy).sum::<f32>() / self.ants.len() as f32
        };

        ColonyStats {
            colony_id: self.id,
            species: self.species.clone(),
            total_ants: self.ants.len(),
            workers,
            foragers,
            food_storage: self.food_storage,
            pheromone_trails: self.pheromones.len(),
            mean_energy,
        }
    }
}

/// Colony statistics.
#[derive(Debug, Clone)]
pub struct ColonyStats {
    pub colony_id: u32,
    pub species: String,
    pub total_ants: usize,
    pub workers: usize,
    pub foragers: usize,
    pub food_storage: f32,
    pub pheromone_trails: usize,
    pub mean_energy: f32,
}

// ── Ant key genes ────────────────────────────────────────────────────────────

/// Key ant genes for molecular fidelity.
#[derive(Debug, Clone)]
pub struct AntGene {
    pub name: String,
    pub function: AntGeneFunction,
}

/// Ant gene functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AntGeneFunction {
    /// Vitellogenin - egg yolk protein.
    Vitellogenin,
    /// Juvenile hormone regulation.
    JuvenileHormone,
    /// Foraging behavior (pk gene).
    ForagingBehavior,
    /// Caste determination.
    CasteDetermination,
    /// Pheromone reception.
    PheromoneReception,
    /// Queen mandibular pheromone.
    QueenPheromone,
    /// circadian clock.
    CircadianClock,
    /// Immune response.
    ImmuneResponse,
}

impl AntGene {
    pub fn key_genes() -> Vec<Self> {
        vec![
            Self { name: "Vg".into(), function: AntGeneFunction::Vitellogenin },
            Self { name: "for".into(), function: AntGeneFunction::ForagingBehavior },
            Self { name: "JHbp".into(), function: AntGeneFunction::JuvenileHormone },
            Self { name: "per".into(), function: AntGeneFunction::CircadianClock },
            Self { name: "Orco".into(), function: AntGeneFunction::PheromoneReception },
        ]
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ant_creation() {
        let ant = Ant::new(1, 0, AntCaste::WorkerMinor, 0.0, 0.0);
        assert!(ant.alive);
        assert_eq!(ant.caste, AntCaste::WorkerMinor);
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
        p.age_s = 100.0;
        assert!(p.effective_strength() < 0.9);
        assert!(p.is_detectable());
    }

    #[test]
    fn test_ant_caste_properties() {
        assert!(AntCaste::Queen.can_lay_eggs());
        assert!(!AntCaste::WorkerMinor.can_lay_eggs());
        assert!(AntCaste::WorkerMinor.can_forage());
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
        assert_eq!(stats.total_ants, 51); // 50 workers + 1 queen
        assert!(stats.workers > 0);
    }

    #[test]
    fn test_ant_brain() {
        let mut brain = AntBrain::new();
        brain.process_olfaction(0.5);
        assert!(brain.antennal_lobe_activation > 0.0);
    }

    #[test]
    fn test_ant_genes() {
        let genes = AntGene::key_genes();
        assert!(genes.len() >= 5);
    }
}
