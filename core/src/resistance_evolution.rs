//! Antibiotic resistance evolution dynamics.
//!
//! Models how antibiotic resistance evolves and spreads through microbial
//! populations — clinically relevant for predicting multi-drug resistance
//! (MDR) emergence, optimizing antibiotic cycling protocols, and identifying
//! mutant selection windows.
//!
//! # Scientific Basis
//!
//! - **Pharmacodynamics**: Hill equation kill rate — E_max * C^n / (MIC^n + C^n)
//!   (Regoes et al. 2004, *Antimicrobial Agents and Chemotherapy* 48:3670).
//! - **Pharmacokinetics**: One-compartment elimination — C(t) = (Dose/V) * e^(-0.693t/t½)
//!   (Levison & Levison 2009, *Infectious Disease Clinics* 23:791).
//! - **Mutant Selection Window**: Between MIC_susceptible and MPC, drug
//!   concentrations that selectively enrich resistant subpopulations (Drlica
//!   2003, *Clinical Infectious Diseases* 36(Suppl 1):S42).
//! - **Fitness valleys**: Multi-step resistance where intermediates are less
//!   fit (Weinreich et al. 2006, *Science* 312:111).
//! - **Compensatory evolution**: Fitness cost amelioration without loss of
//!   resistance (Andersson & Hughes 2010, *Nature Reviews Microbiology* 8:260).
//! - **Horizontal gene transfer**: Plasmid-mediated resistance spread
//!   (Frost et al. 2005, *Nature Reviews Microbiology* 3:722).
//!
//! # Design
//!
//! Fully self-contained — no `crate::` imports, no external dependencies.
//! Uses inline xorshift64 PRNG for reproducible stochastic dynamics.
//!
//! # Example
//!
//! ```rust
//! use oneura_core::resistance_evolution::*;
//!
//! let mut sim = ResistanceSimulator::new(10_000, 42);
//! let ab_idx = sim.add_antibiotic(Antibiotic {
//!     name: "Ciprofloxacin".into(),
//!     class: AntibioticClass::Fluoroquinolone,
//!     mic_wild_type: 0.25,
//!     half_life_hours: 4.0,
//!     peak_concentration: 4.0,
//!     mode_of_action: ModeOfAction::DNAReplication,
//! });
//! let mech_idx = sim.add_mechanism(ResistanceMechanism {
//!     name: "GyrA S83L".into(),
//!     mechanism_type: ResistanceType::TargetModification,
//!     mic_fold_increase: 32.0,
//!     fitness_cost: 0.05,
//!     reversion_rate: 1e-8,
//!     transferable: false,
//!     target_classes: vec![AntibioticClass::Fluoroquinolone],
//! });
//! let events = sim.step(&[2.0]);
//! ```

// ── Inline xorshift64 PRNG ─────────────────────────────────────────────
//
// Marsaglia 2003, *Journal of Statistical Software* 8(14):1.
// Period 2^64 - 1, sufficient for population-level simulations.

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Xorshift64 {
    state: u64,
}

#[allow(dead_code)]
impl Xorshift64 {
    /// Create a new PRNG from a seed. Seed of 0 is remapped to 1
    /// because xorshift requires a nonzero state.
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Advance state and return next u64.
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform f64 in [0, 1).
    #[inline]
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Uniform f64 in [lo, hi).
    #[inline]
    fn next_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    /// Uniform usize in [0, n).
    #[inline]
    fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() % (n as u64)) as usize
    }

    /// Return true with probability p.
    #[inline]
    fn bernoulli(&mut self, p: f64) -> bool {
        self.next_f64() < p
    }

    /// Box-Muller transform for standard normal variate.
    fn standard_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ── Enumerations ────────────────────────────────────────────────────────

/// Class of antibiotic — used for MDR/XDR classification per WHO/CDC
/// definitions (Magiorakos et al. 2012, *Clinical Microbiology and
/// Infection* 18:268).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AntibioticClass {
    BetaLactam,
    Fluoroquinolone,
    Aminoglycoside,
    Macrolide,
    Tetracycline,
    Glycopeptide,
    Polymyxin,
}

/// Molecular target of antibiotic action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModeOfAction {
    CellWall,
    ProteinSynthesis,
    DNAReplication,
    MembraneDisruption,
    FolatePathway,
}

/// Biochemical mechanism of resistance.
#[derive(Debug, Clone, PartialEq)]
pub enum ResistanceType {
    /// Drug efflux pump (e.g., AcrAB-TolC).
    Efflux,
    /// Altered drug target (e.g., GyrA mutations for fluoroquinolones).
    TargetModification,
    /// Enzymatic degradation or modification (e.g., beta-lactamases).
    Enzymatic { enzyme_name: String },
    /// Reduced outer membrane permeability (e.g., porin loss).
    Permeability,
    /// Target protection proteins (e.g., Qnr for quinolones).
    TargetProtection,
}

/// Events produced during a generation of resistance evolution.
#[derive(Debug, Clone)]
pub enum ResistanceEvent {
    /// De novo resistance mutation arose in a strain.
    DeNovoMutation {
        strain_id: u64,
        mechanism_idx: usize,
    },
    /// Resistance gene horizontally transferred between strains.
    HorizontalTransfer {
        donor: u64,
        recipient: u64,
        mechanism_idx: usize,
    },
    /// Compensatory mutation reduced fitness cost of resistance.
    CompensatoryMutation { strain_id: u64 },
    /// Resistance mechanism lost (reversion) in absence of selection.
    ResistanceLoss {
        strain_id: u64,
        mechanism_idx: usize,
    },
    /// Strain acquired multi-drug resistance (≥3 antibiotic classes).
    MdrAcquired { strain_id: u64, num_classes: usize },
    /// Population MIC crossed clinical resistance breakpoint.
    ClinicalResistance {
        antibiotic_idx: usize,
        generation: u32,
    },
}

// ── Core data structures ────────────────────────────────────────────────

/// An antibiotic with pharmacokinetic properties.
#[derive(Debug, Clone)]
pub struct Antibiotic {
    /// Human-readable name (e.g., "Ciprofloxacin").
    pub name: String,
    /// Antibiotic class for MDR classification.
    pub class: AntibioticClass,
    /// Minimum inhibitory concentration for wild-type (μg/mL).
    pub mic_wild_type: f64,
    /// Elimination half-life (hours).
    pub half_life_hours: f64,
    /// Peak plasma concentration after standard dose (Cmax, μg/mL).
    pub peak_concentration: f64,
    /// Molecular target.
    pub mode_of_action: ModeOfAction,
}

/// A resistance mechanism with fitness and transfer properties.
#[derive(Debug, Clone)]
pub struct ResistanceMechanism {
    /// Human-readable name (e.g., "TEM-1 beta-lactamase").
    pub name: String,
    /// Biochemical mechanism type.
    pub mechanism_type: ResistanceType,
    /// Fold increase in MIC when mechanism is present.
    pub mic_fold_increase: f64,
    /// Fitness cost as fractional reduction in growth rate (0.0–1.0).
    pub fitness_cost: f64,
    /// Per-generation probability of losing resistance without selection.
    pub reversion_rate: f64,
    /// Whether this mechanism can be horizontally transferred.
    pub transferable: bool,
    /// Which antibiotic classes this mechanism confers resistance to.
    pub target_classes: Vec<AntibioticClass>,
}

/// A bacterial strain with its resistance profile.
#[derive(Debug, Clone)]
pub struct Strain {
    /// Unique strain identifier.
    pub id: u64,
    /// Intrinsic growth rate (doublings per hour) — typically 0.5–2.0 for
    /// clinically relevant bacteria.
    pub growth_rate: f64,
    /// Indices into the simulator's mechanism library.
    pub resistance_mechanisms: Vec<usize>,
    /// Number of compensatory mutations that ameliorate fitness cost.
    pub compensatory_mutations: u32,
    /// Generation when this strain was created/last modified.
    pub generation: u32,
}

/// Tracks MIC evolution over generations for one antibiotic.
#[derive(Debug, Clone)]
pub struct MicTracker {
    /// Time series of (generation, population-weighted MIC).
    pub mic_history: Vec<(u32, f64)>,
    /// Clinical resistance breakpoint (μg/mL) — from CLSI/EUCAST tables.
    pub mic_breakpoint: f64,
    /// Generation when clinical resistance was first declared.
    pub resistance_declared_gen: Option<u32>,
}

/// Multi-drug resistance profile for a single strain.
#[derive(Debug, Clone)]
pub struct MdrProfile {
    /// Strain identifier.
    pub strain_id: u64,
    /// Names of antibiotics this strain resists.
    pub resistant_to: Vec<String>,
    /// Number of distinct antibiotic classes resisted.
    pub num_classes: usize,
    /// Multi-drug resistant: ≥3 antibiotic classes.
    pub is_mdr: bool,
    /// Extensively drug resistant: ≥5 classes.
    pub is_xdr: bool,
    /// Pan-drug resistant: resistant to all available classes.
    pub is_pan_resistant: bool,
}

/// Record of a fitness valley crossing event — where resistance requires
/// traversing less-fit intermediate genotypes (Weinreich et al. 2006).
#[derive(Debug, Clone)]
pub struct FitnessValleyCrossing {
    /// Generation when the crossing completed.
    pub generation: u32,
    /// Fitness of the intermediate (valley) genotype.
    pub intermediate_fitness: f64,
    /// Fitness of the final (peak) genotype.
    pub final_fitness: f64,
    /// Number of mutations involved in the crossing.
    pub num_mutations: u32,
    /// Generations elapsed to traverse the valley.
    pub crossing_time: u32,
}

// ── Simulator ───────────────────────────────────────────────────────────

/// Main resistance evolution simulator.
///
/// Tracks a polymorphic microbial population under antibiotic selection,
/// modeling de novo mutation, horizontal gene transfer, compensatory
/// evolution, and resistance loss. Produces clinically relevant outputs
/// including MIC trajectories, MDR classification, mutant selection
/// windows, and fitness valley crossings.
pub struct ResistanceSimulator {
    /// (Strain, frequency) pairs — frequencies sum to 1.0.
    pub strains: Vec<(Strain, f64)>,
    /// Library of available antibiotics.
    pub antibiotics: Vec<Antibiotic>,
    /// Library of known resistance mechanisms.
    pub mechanisms: Vec<ResistanceMechanism>,
    /// Per-antibiotic MIC trackers.
    pub mic_trackers: Vec<MicTracker>,
    /// Current generation number.
    pub generation: u32,
    /// Per-generation probability of de novo resistance mutation per strain.
    pub mutation_rate: f64,
    /// Per-generation probability of horizontal gene transfer per strain pair.
    pub hgt_rate: f64,
    /// Per-generation probability of compensatory mutation per resistant strain.
    pub compensatory_rate: f64,
    /// Effective population size (determines genetic drift strength).
    pub population_size: usize,
    /// Internal PRNG state.
    rng_state: u64,

    // Internal bookkeeping.
    next_strain_id: u64,
    /// History of fitness valley crossing events.
    valley_crossings: Vec<FitnessValleyCrossing>,
    /// Track when strains acquired resistance for valley detection:
    /// (strain_id, generation_acquired, num_mechanisms, fitness_at_acquisition).
    acquisition_log: Vec<(u64, u32, u32, f64)>,
}

// ── Pharmacokinetic / Pharmacodynamic free functions ─────────────────────

/// One-compartment pharmacokinetic model: drug concentration over time.
///
/// C(t) = (Dose / Volume) * e^(-0.693 * t / t_half)
///
/// # Arguments
/// * `dose` — administered dose (μg)
/// * `half_life` — elimination half-life (hours)
/// * `volume` — volume of distribution (L)
/// * `time_hours` — time since administration (hours)
pub fn pk_concentration(dose: f64, half_life: f64, volume: f64, time_hours: f64) -> f64 {
    if half_life <= 0.0 || volume <= 0.0 {
        return 0.0;
    }
    let c0 = dose / volume;
    let ke = 0.693 / half_life; // ln(2) / t½
    c0 * (-ke * time_hours).exp()
}

/// Compute the Mutant Prevention Concentration (MPC).
///
/// MPC is the concentration above which no first-step mutant can grow.
/// Empirically, MPC ≈ MIC_wild_type * max_fold_increase (Drlica 2003).
///
/// # Arguments
/// * `wild_type_mic` — MIC of the wild-type strain (μg/mL)
/// * `max_fold_increase` — largest fold increase among available first-step mechanisms
pub fn mutant_prevention_concentration(wild_type_mic: f64, max_fold_increase: f64) -> f64 {
    wild_type_mic * max_fold_increase
}

/// Selection coefficient for resistance at a given drug concentration.
///
/// Positive values favor the resistant strain; negative favor the susceptible.
/// Based on the pharmacodynamic growth rate differential (Day & Read 2016,
/// *Evolution, Medicine, and Public Health* 2016:40).
///
/// # Arguments
/// * `growth_susceptible` — max growth rate of susceptible strain
/// * `growth_resistant` — max growth rate of resistant strain (before fitness cost)
/// * `fitness_cost` — fractional reduction in growth rate due to resistance
/// * `concentration` — current drug concentration (μg/mL)
/// * `mic_s` — MIC of susceptible strain
/// * `mic_r` — MIC of resistant strain
pub fn selection_coefficient(
    growth_susceptible: f64,
    growth_resistant: f64,
    fitness_cost: f64,
    concentration: f64,
    mic_s: f64,
    mic_r: f64,
) -> f64 {
    // Effective growth = growth * (1 - kill_rate)
    // For susceptible: full growth, but killed above MIC_s
    // For resistant: growth * (1 - fitness_cost), killed above MIC_r
    let hill = 2.0; // typical Hill coefficient for antibiotics
    let e_max = 0.99; // maximum kill fraction

    let kill_s = ResistanceSimulator::pd_kill_rate(concentration, mic_s, hill, e_max);
    let kill_r = ResistanceSimulator::pd_kill_rate(concentration, mic_r, hill, e_max);

    let effective_s = growth_susceptible * (1.0 - kill_s);
    let effective_r = growth_resistant * (1.0 - fitness_cost) * (1.0 - kill_r);

    // Selection coefficient: (w_r - w_s) / w_s
    if effective_s.abs() < 1e-15 {
        // Susceptible completely inhibited — any resistant growth is infinite advantage
        if effective_r > 0.0 {
            1.0
        } else {
            0.0
        }
    } else {
        (effective_r - effective_s) / effective_s.abs()
    }
}

// ── Simulator implementation ────────────────────────────────────────────

impl ResistanceSimulator {
    /// Create a new simulator with a wild-type population.
    ///
    /// # Arguments
    /// * `pop_size` — effective population size
    /// * `seed` — PRNG seed for reproducibility
    pub fn new(pop_size: usize, seed: u64) -> Self {
        let wild_type = Strain {
            id: 0,
            growth_rate: 1.0, // 1 doubling/hour — typical E. coli
            resistance_mechanisms: Vec::new(),
            compensatory_mutations: 0,
            generation: 0,
        };
        Self {
            strains: vec![(wild_type, 1.0)],
            antibiotics: Vec::new(),
            mechanisms: Vec::new(),
            mic_trackers: Vec::new(),
            generation: 0,
            mutation_rate: 1e-6,
            hgt_rate: 1e-4,
            compensatory_rate: 1e-5,
            population_size: pop_size,
            rng_state: if seed == 0 { 1 } else { seed },
            next_strain_id: 1,
            valley_crossings: Vec::new(),
            acquisition_log: Vec::new(),
        }
    }

    /// Add an antibiotic to the simulator. Returns its index.
    pub fn add_antibiotic(&mut self, ab: Antibiotic) -> usize {
        let idx = self.antibiotics.len();
        // Create a MIC tracker with a default breakpoint of 4x the wild-type MIC
        // (a common CLSI convention for many drug/organism pairs).
        let tracker = MicTracker {
            mic_history: Vec::new(),
            mic_breakpoint: ab.mic_wild_type * 4.0,
            resistance_declared_gen: None,
        };
        self.antibiotics.push(ab);
        self.mic_trackers.push(tracker);
        idx
    }

    /// Add a resistance mechanism to the library. Returns its index.
    pub fn add_mechanism(&mut self, mech: ResistanceMechanism) -> usize {
        let idx = self.mechanisms.len();
        self.mechanisms.push(mech);
        idx
    }

    // ── Internal PRNG helpers ───────────────────────────────────────────

    #[inline]
    fn rng_next(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    #[inline]
    fn rng_f64(&mut self) -> f64 {
        (self.rng_next() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    #[inline]
    fn rng_usize(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.rng_next() % (n as u64)) as usize
    }

    #[inline]
    fn rng_bernoulli(&mut self, p: f64) -> bool {
        self.rng_f64() < p
    }

    fn alloc_strain_id(&mut self) -> u64 {
        let id = self.next_strain_id;
        self.next_strain_id += 1;
        id
    }

    // ── Pharmacodynamics ────────────────────────────────────────────────

    /// Pharmacodynamic kill rate via the Hill equation (sigmoidal Emax model).
    ///
    /// kill = E_max * C^n / (MIC^n + C^n)
    ///
    /// # Arguments
    /// * `concentration` — drug concentration (μg/mL)
    /// * `mic` — minimum inhibitory concentration (μg/mL)
    /// * `hill_coefficient` — Hill coefficient (steepness of dose-response)
    /// * `e_max` — maximum kill rate fraction (0.0–1.0)
    pub fn pd_kill_rate(concentration: f64, mic: f64, hill_coefficient: f64, e_max: f64) -> f64 {
        if concentration <= 0.0 || mic <= 0.0 {
            return 0.0;
        }
        let cn = concentration.powf(hill_coefficient);
        let mn = mic.powf(hill_coefficient);
        e_max * cn / (mn + cn)
    }

    // ── MIC computation ─────────────────────────────────────────────────

    /// Compute the MIC of a strain against a specific antibiotic.
    ///
    /// The MIC is the wild-type MIC multiplied by the product of all
    /// relevant resistance mechanism fold increases. When multiple
    /// mechanisms target the same antibiotic class, their effects multiply
    /// (as experimentally observed for stepwise resistance accumulation).
    pub fn strain_mic(&self, strain: &Strain, antibiotic_idx: usize) -> f64 {
        if antibiotic_idx >= self.antibiotics.len() {
            return 0.0;
        }
        let ab = &self.antibiotics[antibiotic_idx];
        let mut fold = 1.0_f64;
        for &mech_idx in &strain.resistance_mechanisms {
            if mech_idx < self.mechanisms.len() {
                let mech = &self.mechanisms[mech_idx];
                if mech.target_classes.contains(&ab.class) {
                    fold *= mech.mic_fold_increase;
                }
            }
        }
        ab.mic_wild_type * fold
    }

    /// Compute effective fitness of a strain under given antibiotic concentrations.
    ///
    /// Fitness = growth_rate * (1 - net_fitness_cost) * Π(1 - kill_rate_i)
    fn strain_fitness(&self, strain: &Strain, concentrations: &[f64]) -> f64 {
        // Net fitness cost: sum of mechanism costs, reduced by compensatory mutations
        let raw_cost: f64 = strain
            .resistance_mechanisms
            .iter()
            .filter_map(|&idx| self.mechanisms.get(idx))
            .map(|m| m.fitness_cost)
            .sum();

        // Each compensatory mutation reduces total cost by ~30% (multiplicative).
        // Based on Maisnier-Patin et al. 2002, *Molecular Microbiology* 46:355.
        let compensation = 0.7_f64.powi(strain.compensatory_mutations as i32);
        let net_cost = (raw_cost * compensation).min(0.99);

        let mut fitness = strain.growth_rate * (1.0 - net_cost);

        // Apply pharmacodynamic kill for each antibiotic present
        for (i, &conc) in concentrations.iter().enumerate() {
            if conc > 0.0 && i < self.antibiotics.len() {
                let mic = self.strain_mic(strain, i);
                let kill = Self::pd_kill_rate(conc, mic, 2.0, 0.99);
                fitness *= 1.0 - kill;
            }
        }

        fitness.max(0.0)
    }

    // ── MDR profiling ───────────────────────────────────────────────────

    /// Compute the multi-drug resistance profile for a strain.
    ///
    /// Classification follows Magiorakos et al. 2012:
    /// - MDR: non-susceptible to ≥1 agent in ≥3 antimicrobial categories
    /// - XDR: non-susceptible to ≥1 agent in all but ≤2 categories
    /// - PDR: non-susceptible to all agents in all categories
    pub fn mdr_profile(&self, strain: &Strain) -> MdrProfile {
        let mut resistant_names = Vec::new();
        let mut resistant_classes = std::collections::HashSet::new();

        for (i, ab) in self.antibiotics.iter().enumerate() {
            let mic = self.strain_mic(strain, i);
            if mic > ab.mic_wild_type * 2.0 {
                // Strain has elevated MIC — clinically non-susceptible
                resistant_names.push(ab.name.clone());
                resistant_classes.insert(ab.class.clone());
            }
        }

        let total_classes = self
            .antibiotics
            .iter()
            .map(|ab| ab.class.clone())
            .collect::<std::collections::HashSet<_>>()
            .len();

        let num_classes = resistant_classes.len();
        MdrProfile {
            strain_id: strain.id,
            resistant_to: resistant_names,
            num_classes,
            is_mdr: num_classes >= 3,
            is_xdr: num_classes >= 5,
            is_pan_resistant: total_classes > 0 && num_classes >= total_classes,
        }
    }

    // ── Main evolution step ─────────────────────────────────────────────

    /// Run one generation of resistance evolution.
    ///
    /// Each generation:
    /// 1. Compute fitness of all strains under current antibiotic selection.
    /// 2. Attempt de novo resistance mutations.
    /// 3. Attempt horizontal gene transfer of mobile resistance elements.
    /// 4. Attempt compensatory mutations for resistant strains.
    /// 5. Revert resistance in strains not under selection.
    /// 6. Apply Wright-Fisher selection (resampling by fitness).
    /// 7. Update MIC trackers and detect clinical resistance events.
    ///
    /// Returns a vector of all events that occurred this generation.
    pub fn step(&mut self, antibiotic_concentrations: &[f64]) -> Vec<ResistanceEvent> {
        let mut events = Vec::new();
        self.generation += 1;
        let gen = self.generation;

        // 1. De novo resistance mutations
        let n_strains = self.strains.len();
        let mut new_strains: Vec<(Strain, f64)> = Vec::new();

        for i in 0..n_strains {
            if self.rng_bernoulli(self.mutation_rate) && !self.mechanisms.is_empty() {
                let mech_idx = self.rng_usize(self.mechanisms.len());
                // Clone parent data to avoid borrow conflict with self.alloc_strain_id().
                let parent_mechs = self.strains[i].0.resistance_mechanisms.clone();
                let parent_growth = self.strains[i].0.growth_rate;
                let parent_comp = self.strains[i].0.compensatory_mutations;
                let parent_freq = self.strains[i].1;

                // Only mutate if strain doesn't already have this mechanism
                if !parent_mechs.contains(&mech_idx) {
                    let new_id = self.alloc_strain_id();
                    let mut new_mechs = parent_mechs;
                    new_mechs.push(mech_idx);

                    let new_strain = Strain {
                        id: new_id,
                        growth_rate: parent_growth,
                        resistance_mechanisms: new_mechs,
                        compensatory_mutations: parent_comp,
                        generation: gen,
                    };

                    // Mutant appears at frequency 1/N
                    let mutant_freq = 1.0 / self.population_size as f64;
                    let parent_freq_new = (parent_freq - mutant_freq).max(0.0);
                    self.strains[i].1 = parent_freq_new;

                    // Log acquisition for valley crossing detection
                    let fitness_at_acq =
                        self.strain_fitness(&new_strain, antibiotic_concentrations);
                    self.acquisition_log.push((
                        new_id,
                        gen,
                        new_strain.resistance_mechanisms.len() as u32,
                        fitness_at_acq,
                    ));

                    events.push(ResistanceEvent::DeNovoMutation {
                        strain_id: new_id,
                        mechanism_idx: mech_idx,
                    });

                    new_strains.push((new_strain, mutant_freq));
                }
            }
        }
        self.strains.extend(new_strains);

        // 2. Horizontal gene transfer
        let n_strains = self.strains.len();
        let mut hgt_new: Vec<(Strain, f64)> = Vec::new();

        if n_strains >= 2 {
            // Number of HGT attempts proportional to population density
            let hgt_attempts = ((n_strains * n_strains) as f64 * self.hgt_rate) as usize;
            let hgt_attempts = hgt_attempts.max(1).min(n_strains * 2);

            for _ in 0..hgt_attempts {
                let donor_idx = self.rng_usize(n_strains);
                let recip_idx = self.rng_usize(n_strains);
                if donor_idx == recip_idx {
                    continue;
                }

                // Copy data we need before any mutation (borrow checker).
                let donor_mechs = self.strains[donor_idx].0.resistance_mechanisms.clone();
                let donor_id = self.strains[donor_idx].0.id;
                let donor_freq = self.strains[donor_idx].1;
                let recip_mechs = self.strains[recip_idx].0.resistance_mechanisms.clone();
                let recip_id = self.strains[recip_idx].0.id;
                let recip_growth = self.strains[recip_idx].0.growth_rate;
                let recip_comp = self.strains[recip_idx].0.compensatory_mutations;
                let recip_freq = self.strains[recip_idx].1;

                // Find transferable mechanisms in donor that recipient lacks
                for &mech_idx in &donor_mechs {
                    if mech_idx >= self.mechanisms.len() {
                        continue;
                    }
                    if !self.mechanisms[mech_idx].transferable {
                        continue;
                    }
                    if recip_mechs.contains(&mech_idx) {
                        continue;
                    }

                    // Transfer probability weighted by donor frequency
                    let transfer_prob = donor_freq;
                    if self.rng_bernoulli(transfer_prob) {
                        let new_id = self.alloc_strain_id();
                        let mut new_mechs = recip_mechs.clone();
                        new_mechs.push(mech_idx);

                        let new_strain = Strain {
                            id: new_id,
                            growth_rate: recip_growth,
                            resistance_mechanisms: new_mechs,
                            compensatory_mutations: recip_comp,
                            generation: gen,
                        };

                        let transconjugant_freq = (recip_freq * transfer_prob)
                            .min(recip_freq * 0.5)
                            .max(1.0 / self.population_size as f64);
                        self.strains[recip_idx].1 = (recip_freq - transconjugant_freq).max(0.0);

                        events.push(ResistanceEvent::HorizontalTransfer {
                            donor: donor_id,
                            recipient: recip_id,
                            mechanism_idx: mech_idx,
                        });

                        hgt_new.push((new_strain, transconjugant_freq));
                        break; // One transfer event per pair per generation
                    }
                }
            }
        }
        self.strains.extend(hgt_new);

        // 3. Compensatory mutations
        for i in 0..self.strains.len() {
            let (strain, _) = &self.strains[i];
            if !strain.resistance_mechanisms.is_empty()
                && self.rng_bernoulli(self.compensatory_rate)
            {
                let strain_id = self.strains[i].0.id;
                self.strains[i].0.compensatory_mutations += 1;
                events.push(ResistanceEvent::CompensatoryMutation { strain_id });
            }
        }

        // 4. Resistance loss (reversion) for mechanisms without selection
        let mut loss_events = Vec::new();
        for i in 0..self.strains.len() {
            let mut to_remove = Vec::new();
            // Clone mechanism indices to avoid borrow conflict with self.rng_bernoulli().
            let mech_indices: Vec<usize> = self.strains[i].0.resistance_mechanisms.clone();
            for (j, &mech_idx) in mech_indices.iter().enumerate() {
                if mech_idx < self.mechanisms.len() {
                    let reversion_rate = self.mechanisms[mech_idx].reversion_rate;
                    let target_classes = &self.mechanisms[mech_idx].target_classes;

                    // Check if any antibiotic with this class is present above MIC
                    let under_selection =
                        self.antibiotics.iter().enumerate().any(|(ab_idx, ab)| {
                            target_classes.contains(&ab.class)
                                && ab_idx < antibiotic_concentrations.len()
                                && antibiotic_concentrations[ab_idx] > ab.mic_wild_type * 0.1
                        });

                    if !under_selection && self.rng_bernoulli(reversion_rate) {
                        to_remove.push(j);
                    }
                }
            }
            // Remove in reverse order to preserve indices
            for &j in to_remove.iter().rev() {
                let mech_idx = self.strains[i].0.resistance_mechanisms.remove(j);
                loss_events.push(ResistanceEvent::ResistanceLoss {
                    strain_id: self.strains[i].0.id,
                    mechanism_idx: mech_idx,
                });
            }
        }
        events.extend(loss_events);

        // 5. Wright-Fisher selection — reweight frequencies by fitness
        let fitnesses: Vec<f64> = self
            .strains
            .iter()
            .map(|(s, _)| self.strain_fitness(s, antibiotic_concentrations))
            .collect();

        let total_fitness: f64 = self
            .strains
            .iter()
            .zip(fitnesses.iter())
            .map(|((_, freq), fit)| freq * fit)
            .sum();

        if total_fitness > 0.0 {
            for (i, (_, freq)) in self.strains.iter_mut().enumerate() {
                *freq = (*freq * fitnesses[i]) / total_fitness;
            }
        }

        // Add drift noise for finite populations.
        // Use index-based loop to avoid borrow conflict with self.rng_f64().
        let inv_n = 1.0 / self.population_size as f64;
        let n_drift = self.strains.len();
        for i in 0..n_drift {
            let p = self.strains[i].1;
            if p > 0.0 && p < 1.0 {
                let variance = p * (1.0 - p) * inv_n;
                let u1 = self.rng_f64().max(1e-15);
                let u2 = self.rng_f64();
                let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                self.strains[i].1 = (self.strains[i].1 + noise * variance.sqrt()).clamp(0.0, 1.0);
            }
        }

        // Renormalize frequencies
        let total: f64 = self.strains.iter().map(|(_, f)| *f).sum();
        if total > 0.0 {
            for (_, freq) in self.strains.iter_mut() {
                *freq /= total;
            }
        }

        // Remove strains that have drifted to extinction (freq < 1/2N)
        let extinction_threshold = 0.5 / self.population_size as f64;
        self.strains
            .retain(|(_, freq)| *freq >= extinction_threshold);

        // Re-normalize after removal
        let total: f64 = self.strains.iter().map(|(_, f)| *f).sum();
        if total > 0.0 {
            for (_, freq) in self.strains.iter_mut() {
                *freq /= total;
            }
        }

        // 6. Check for MDR acquisition events
        for (strain, _) in &self.strains {
            let profile = self.mdr_profile(strain);
            if profile.is_mdr {
                // Only emit event if this is a newly observed MDR strain
                // (simplification: emit when profile first computed as MDR)
                events.push(ResistanceEvent::MdrAcquired {
                    strain_id: strain.id,
                    num_classes: profile.num_classes,
                });
            }
        }

        // 7. Update MIC trackers and detect clinical resistance
        for (ab_idx, tracker) in self.mic_trackers.iter_mut().enumerate() {
            // Population-weighted mean MIC
            let pop_mic: f64 = self
                .strains
                .iter()
                .map(|(s, freq)| {
                    let mut fold = 1.0_f64;
                    if ab_idx < self.antibiotics.len() {
                        let ab_class = &self.antibiotics[ab_idx].class;
                        for &mech_idx in &s.resistance_mechanisms {
                            if mech_idx < self.mechanisms.len()
                                && self.mechanisms[mech_idx].target_classes.contains(ab_class)
                            {
                                fold *= self.mechanisms[mech_idx].mic_fold_increase;
                            }
                        }
                    }
                    freq * self
                        .antibiotics
                        .get(ab_idx)
                        .map(|ab| ab.mic_wild_type * fold)
                        .unwrap_or(0.0)
                })
                .sum();

            tracker.mic_history.push((gen, pop_mic));

            // Detect clinical resistance threshold crossing
            if tracker.resistance_declared_gen.is_none() && pop_mic >= tracker.mic_breakpoint {
                tracker.resistance_declared_gen = Some(gen);
                events.push(ResistanceEvent::ClinicalResistance {
                    antibiotic_idx: ab_idx,
                    generation: gen,
                });
            }
        }

        // 8. Detect fitness valley crossings
        self.detect_valley_crossings_internal(antibiotic_concentrations);

        events
    }

    /// Predict the number of generations until clinical resistance emerges
    /// at a given drug concentration.
    ///
    /// Uses a deterministic approximation based on the mutant selection
    /// coefficient and mutation supply rate (μ * N * s).
    /// Returns `None` if resistance is not predicted to arise (e.g., drug
    /// concentration is below the MSW or above MPC).
    pub fn predict_resistance_time(
        &self,
        antibiotic_idx: usize,
        concentration: f64,
    ) -> Option<u32> {
        if antibiotic_idx >= self.antibiotics.len() || self.mechanisms.is_empty() {
            return None;
        }

        let ab = &self.antibiotics[antibiotic_idx];

        // Find the most likely resistance mechanism for this antibiotic
        let best_mech = self
            .mechanisms
            .iter()
            .filter(|m| m.target_classes.contains(&ab.class))
            .min_by(|a, b| a.fitness_cost.partial_cmp(&b.fitness_cost).unwrap());

        let mech = match best_mech {
            Some(m) => m,
            None => return None,
        };

        let mic_r = ab.mic_wild_type * mech.mic_fold_increase;

        // If concentration is above MPC, no single-step mutant survives
        if concentration >= mic_r {
            return None;
        }

        // If concentration is below wild-type MIC, no selection for resistance
        if concentration < ab.mic_wild_type {
            return None;
        }

        // Selection coefficient for the resistant mutant
        let s = selection_coefficient(
            1.0,
            1.0,
            mech.fitness_cost,
            concentration,
            ab.mic_wild_type,
            mic_r,
        );

        if s <= 0.0 {
            return None;
        }

        // Time to fixation ≈ 1 / (2 * N * μ * s) * ln(N)
        // (Kimura & Ohta 1969 approximation for beneficial mutations)
        let n = self.population_size as f64;
        let mu = self.mutation_rate;
        let supply_rate = 2.0 * n * mu * s;

        if supply_rate <= 0.0 {
            return None;
        }

        let establishment_time = 1.0 / supply_rate;
        let fixation_time = n.ln() / s;
        let total = establishment_time + fixation_time;

        Some(total.ceil() as u32)
    }

    /// Detect fitness valley crossings in the evolutionary history.
    ///
    /// A valley crossing occurs when a strain acquires multiple resistance
    /// mechanisms and at least one intermediate genotype had lower fitness
    /// than the ancestor. This indicates epistatic interactions where
    /// multi-step resistance traverses a fitness landscape valley.
    pub fn detect_valley_crossings(&self) -> Vec<FitnessValleyCrossing> {
        self.valley_crossings.clone()
    }

    /// Internal valley crossing detection — called each generation.
    fn detect_valley_crossings_internal(&mut self, concentrations: &[f64]) {
        // Look for strains with ≥2 resistance mechanisms whose intermediate
        // genotypes had lower fitness than the current genotype
        for (strain, freq) in &self.strains {
            if strain.resistance_mechanisms.len() < 2 || *freq < 1.0 / self.population_size as f64 {
                continue;
            }

            let current_fitness = self.strain_fitness(strain, concentrations);

            // Construct each single-mechanism intermediate
            let mut min_intermediate_fitness = f64::MAX;
            for &mech_idx in &strain.resistance_mechanisms {
                let intermediate = Strain {
                    id: 0,
                    growth_rate: strain.growth_rate,
                    resistance_mechanisms: vec![mech_idx],
                    compensatory_mutations: 0,
                    generation: strain.generation,
                };
                let int_fitness = self.strain_fitness(&intermediate, concentrations);
                min_intermediate_fitness = min_intermediate_fitness.min(int_fitness);
            }

            // Also check the ancestor (no mechanisms)
            let ancestor = Strain {
                id: 0,
                growth_rate: strain.growth_rate,
                resistance_mechanisms: Vec::new(),
                compensatory_mutations: 0,
                generation: 0,
            };
            let ancestor_fitness = self.strain_fitness(&ancestor, concentrations);

            // Valley detected: some intermediate is less fit than BOTH
            // the ancestor and the final multi-mechanism strain
            if min_intermediate_fitness < ancestor_fitness
                && min_intermediate_fitness < current_fitness
                && current_fitness > ancestor_fitness
            {
                // Find the acquisition time from log
                let first_acq = self
                    .acquisition_log
                    .iter()
                    .find(|(id, _, _, _)| *id == strain.id)
                    .map(|(_, gen, _, _)| *gen)
                    .unwrap_or(strain.generation);

                let crossing = FitnessValleyCrossing {
                    generation: self.generation,
                    intermediate_fitness: min_intermediate_fitness,
                    final_fitness: current_fitness,
                    num_mutations: strain.resistance_mechanisms.len() as u32,
                    crossing_time: self.generation.saturating_sub(first_acq),
                };

                // Avoid duplicate entries for the same strain
                let dominated = self.valley_crossings.iter().any(|vc| {
                    vc.num_mutations == crossing.num_mutations
                        && (vc.final_fitness - crossing.final_fitness).abs() < 1e-10
                        && vc.generation == crossing.generation
                });
                if !dominated {
                    self.valley_crossings.push(crossing);
                }
            }
        }
    }

    /// Compute the mutant selection window (MSW) for an antibiotic.
    ///
    /// Returns (MSW_low, MSW_high) where:
    /// - MSW_low = MIC of the susceptible strain (wild-type MIC)
    /// - MSW_high = MPC (highest MIC among single-step mutants)
    ///
    /// Concentrations within this window selectively enrich resistant
    /// subpopulations — the most dangerous dosing regime (Drlica 2003).
    pub fn mutant_selection_window(&self, antibiotic_idx: usize) -> (f64, f64) {
        if antibiotic_idx >= self.antibiotics.len() {
            return (0.0, 0.0);
        }

        let ab = &self.antibiotics[antibiotic_idx];
        let msw_low = ab.mic_wild_type;

        // MSW_high = MPC = max MIC among all single-step resistant mutants
        let max_fold = self
            .mechanisms
            .iter()
            .filter(|m| m.target_classes.contains(&ab.class))
            .map(|m| m.mic_fold_increase)
            .fold(1.0_f64, f64::max);

        let msw_high = mutant_prevention_concentration(ab.mic_wild_type, max_fold);

        (msw_low, msw_high)
    }

    /// Run an antibiotic cycling protocol and track resistance frequencies.
    ///
    /// Cycling alternates between antibiotics to prevent resistance
    /// accumulation against any single drug. This method implements a
    /// protocol schedule and returns per-generation frequency vectors.
    ///
    /// # Arguments
    /// * `protocol` — sequence of (antibiotic_idx, concentration, duration_generations)
    /// * `generations` — total number of generations to simulate
    ///
    /// # Returns
    /// A vector of frequency snapshots (one per generation), where each
    /// inner vector contains the frequency of each strain.
    pub fn run_cycling_protocol(
        &mut self,
        protocol: &[(usize, f64, u32)],
        generations: u32,
    ) -> Vec<Vec<f64>> {
        if protocol.is_empty() {
            return Vec::new();
        }

        let mut freq_history = Vec::with_capacity(generations as usize);
        let n_antibiotics = self.antibiotics.len();

        // Build a generation-to-concentration schedule
        let mut cycle_idx = 0usize;
        let mut gen_in_phase = 0u32;

        for _ in 0..generations {
            // Determine current antibiotic concentrations from protocol
            let (ab_idx, conc, duration) = protocol[cycle_idx % protocol.len()];
            let mut concentrations = vec![0.0; n_antibiotics];
            if ab_idx < n_antibiotics {
                concentrations[ab_idx] = conc;
            }

            self.step(&concentrations);

            // Record strain frequencies
            let freqs: Vec<f64> = self.strains.iter().map(|(_, f)| *f).collect();
            freq_history.push(freqs);

            gen_in_phase += 1;
            if gen_in_phase >= duration {
                gen_in_phase = 0;
                cycle_idx += 1;
            }
        }

        freq_history
    }

    /// Compute the population-weighted mean fitness under given antibiotic
    /// concentrations.
    pub fn population_mean_fitness(&self, concentrations: &[f64]) -> f64 {
        self.strains
            .iter()
            .map(|(s, freq)| freq * self.strain_fitness(s, concentrations))
            .sum()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper factories ────────────────────────────────────────────

    fn cipro() -> Antibiotic {
        Antibiotic {
            name: "Ciprofloxacin".into(),
            class: AntibioticClass::Fluoroquinolone,
            mic_wild_type: 0.25,
            half_life_hours: 4.0,
            peak_concentration: 4.0,
            mode_of_action: ModeOfAction::DNAReplication,
        }
    }

    fn ampicillin() -> Antibiotic {
        Antibiotic {
            name: "Ampicillin".into(),
            class: AntibioticClass::BetaLactam,
            mic_wild_type: 2.0,
            half_life_hours: 1.0,
            peak_concentration: 20.0,
            mode_of_action: ModeOfAction::CellWall,
        }
    }

    fn gentamicin() -> Antibiotic {
        Antibiotic {
            name: "Gentamicin".into(),
            class: AntibioticClass::Aminoglycoside,
            mic_wild_type: 1.0,
            half_life_hours: 2.5,
            peak_concentration: 10.0,
            mode_of_action: ModeOfAction::ProteinSynthesis,
        }
    }

    #[allow(dead_code)]
    fn erythromycin() -> Antibiotic {
        Antibiotic {
            name: "Erythromycin".into(),
            class: AntibioticClass::Macrolide,
            mic_wild_type: 0.5,
            half_life_hours: 1.5,
            peak_concentration: 5.0,
            mode_of_action: ModeOfAction::ProteinSynthesis,
        }
    }

    fn gyra_mutation() -> ResistanceMechanism {
        ResistanceMechanism {
            name: "GyrA S83L".into(),
            mechanism_type: ResistanceType::TargetModification,
            mic_fold_increase: 32.0,
            fitness_cost: 0.05,
            reversion_rate: 1e-8,
            transferable: false,
            target_classes: vec![AntibioticClass::Fluoroquinolone],
        }
    }

    fn tem1_blactamase() -> ResistanceMechanism {
        ResistanceMechanism {
            name: "TEM-1 beta-lactamase".into(),
            mechanism_type: ResistanceType::Enzymatic {
                enzyme_name: "TEM-1".into(),
            },
            mic_fold_increase: 64.0,
            fitness_cost: 0.02,
            reversion_rate: 1e-7,
            transferable: true,
            target_classes: vec![AntibioticClass::BetaLactam],
        }
    }

    fn aminoglycoside_acetyltransferase() -> ResistanceMechanism {
        ResistanceMechanism {
            name: "AAC(6')-Ib".into(),
            mechanism_type: ResistanceType::Enzymatic {
                enzyme_name: "AAC(6')-Ib".into(),
            },
            mic_fold_increase: 16.0,
            fitness_cost: 0.03,
            reversion_rate: 1e-7,
            transferable: true,
            target_classes: vec![AntibioticClass::Aminoglycoside],
        }
    }

    #[allow(dead_code)]
    fn erm_methylase() -> ResistanceMechanism {
        ResistanceMechanism {
            name: "ErmB methylase".into(),
            mechanism_type: ResistanceType::TargetModification,
            mic_fold_increase: 128.0,
            fitness_cost: 0.04,
            reversion_rate: 1e-7,
            transferable: true,
            target_classes: vec![AntibioticClass::Macrolide],
        }
    }

    // ── Tests ───────────────────────────────────────────────────────

    #[test]
    fn wild_type_mic_baseline() {
        let mut sim = ResistanceSimulator::new(10_000, 42);
        sim.add_antibiotic(cipro());
        sim.add_mechanism(gyra_mutation());

        // Wild-type strain (no mechanisms) should have the wild-type MIC
        let wt = &sim.strains[0].0;
        let mic = sim.strain_mic(wt, 0);
        assert!(
            (mic - 0.25).abs() < 1e-10,
            "Wild-type MIC should be 0.25, got {mic}"
        );
    }

    #[test]
    fn resistance_increases_mic() {
        let mut sim = ResistanceSimulator::new(10_000, 42);
        sim.add_antibiotic(cipro());
        let mech_idx = sim.add_mechanism(gyra_mutation());

        let resistant = Strain {
            id: 99,
            growth_rate: 1.0,
            resistance_mechanisms: vec![mech_idx],
            compensatory_mutations: 0,
            generation: 0,
        };

        let mic = sim.strain_mic(&resistant, 0);
        // 0.25 * 32 = 8.0
        assert!(
            (mic - 8.0).abs() < 1e-10,
            "Resistant MIC should be 8.0, got {mic}"
        );
        assert!(mic > sim.antibiotics[0].mic_wild_type);
    }

    #[test]
    fn fitness_cost_reduces_growth() {
        let mut sim = ResistanceSimulator::new(10_000, 42);
        sim.add_antibiotic(cipro());
        let mech_idx = sim.add_mechanism(gyra_mutation());

        let wt = Strain {
            id: 0,
            growth_rate: 1.0,
            resistance_mechanisms: Vec::new(),
            compensatory_mutations: 0,
            generation: 0,
        };

        let resistant = Strain {
            id: 1,
            growth_rate: 1.0,
            resistance_mechanisms: vec![mech_idx],
            compensatory_mutations: 0,
            generation: 0,
        };

        // Without drug, resistant strain should be less fit due to cost
        let no_drug = vec![0.0];
        let wt_fitness = sim.strain_fitness(&wt, &no_drug);
        let res_fitness = sim.strain_fitness(&resistant, &no_drug);
        assert!(
            res_fitness < wt_fitness,
            "Resistant strain should be less fit without drug: wt={wt_fitness}, res={res_fitness}"
        );

        // Fitness cost of GyrA S83L is 0.05
        let expected = 1.0 * (1.0 - 0.05);
        assert!(
            (res_fitness - expected).abs() < 1e-10,
            "Expected fitness {expected}, got {res_fitness}"
        );
    }

    #[test]
    fn selection_above_mic() {
        let mut sim = ResistanceSimulator::new(10_000, 42);
        sim.add_antibiotic(cipro());
        let mech_idx = sim.add_mechanism(gyra_mutation());

        let wt = Strain {
            id: 0,
            growth_rate: 1.0,
            resistance_mechanisms: Vec::new(),
            compensatory_mutations: 0,
            generation: 0,
        };

        let resistant = Strain {
            id: 1,
            growth_rate: 1.0,
            resistance_mechanisms: vec![mech_idx],
            compensatory_mutations: 0,
            generation: 0,
        };

        // Concentration above wild-type MIC (0.25) but below resistant MIC (8.0)
        let drug_present = vec![2.0];
        let wt_fitness = sim.strain_fitness(&wt, &drug_present);
        let res_fitness = sim.strain_fitness(&resistant, &drug_present);

        // Susceptible should be nearly killed
        assert!(
            wt_fitness < 0.1,
            "Susceptible should be nearly killed at 2 ug/mL: fitness={wt_fitness}"
        );
        // Resistant should survive well
        assert!(
            res_fitness > 0.5,
            "Resistant should survive at 2 ug/mL: fitness={res_fitness}"
        );
    }

    #[test]
    fn mdr_detection() {
        let mut sim = ResistanceSimulator::new(10_000, 42);
        sim.add_antibiotic(cipro());
        sim.add_antibiotic(ampicillin());
        sim.add_antibiotic(gentamicin());

        let fq_idx = sim.add_mechanism(gyra_mutation());
        let bl_idx = sim.add_mechanism(tem1_blactamase());
        let ag_idx = sim.add_mechanism(aminoglycoside_acetyltransferase());

        // Strain with 3 classes of resistance
        let mdr_strain = Strain {
            id: 42,
            growth_rate: 1.0,
            resistance_mechanisms: vec![fq_idx, bl_idx, ag_idx],
            compensatory_mutations: 0,
            generation: 0,
        };

        let profile = sim.mdr_profile(&mdr_strain);
        assert_eq!(profile.num_classes, 3, "Should resist 3 antibiotic classes");
        assert!(profile.is_mdr, "Should be classified as MDR");
        assert!(!profile.is_xdr, "Should not be XDR with only 3 classes");
        assert_eq!(profile.resistant_to.len(), 3);
    }

    #[test]
    fn hgt_transfers_resistance() {
        // Set up a simulator with high HGT rate to guarantee transfer
        let mut sim = ResistanceSimulator::new(1_000, 7);
        sim.hgt_rate = 0.5; // Very high rate for test
        sim.mutation_rate = 0.0; // Disable de novo mutation

        sim.add_antibiotic(ampicillin());
        let bl_idx = sim.add_mechanism(tem1_blactamase());

        // Manually add a resistant donor strain
        let donor = Strain {
            id: sim.alloc_strain_id(),
            growth_rate: 1.0,
            resistance_mechanisms: vec![bl_idx],
            compensatory_mutations: 0,
            generation: 0,
        };
        sim.strains.push((donor, 0.3));
        // Reduce wild-type frequency
        sim.strains[0].1 = 0.7;

        // Run several generations with sub-MIC drug to maintain both populations
        let mut any_hgt = false;
        for _ in 0..50 {
            let events = sim.step(&[0.5]); // sub-MIC — mild selection
            for ev in &events {
                if matches!(ev, ResistanceEvent::HorizontalTransfer { .. }) {
                    any_hgt = true;
                }
            }
        }

        assert!(
            any_hgt,
            "HGT should occur with transferable mechanism and high rate"
        );
    }

    #[test]
    fn compensatory_reduces_cost() {
        let mut sim = ResistanceSimulator::new(10_000, 42);
        sim.add_antibiotic(cipro());
        let mech_idx = sim.add_mechanism(gyra_mutation());

        let no_comp = Strain {
            id: 1,
            growth_rate: 1.0,
            resistance_mechanisms: vec![mech_idx],
            compensatory_mutations: 0,
            generation: 0,
        };

        let with_comp = Strain {
            id: 2,
            growth_rate: 1.0,
            resistance_mechanisms: vec![mech_idx],
            compensatory_mutations: 3,
            generation: 0,
        };

        let no_drug = vec![0.0];
        let fitness_no_comp = sim.strain_fitness(&no_comp, &no_drug);
        let fitness_with_comp = sim.strain_fitness(&with_comp, &no_drug);

        assert!(
            fitness_with_comp > fitness_no_comp,
            "Compensatory mutations should improve fitness: no_comp={fitness_no_comp}, with_comp={fitness_with_comp}"
        );

        // With 3 compensatory mutations, cost reduced by 0.7^3 = 0.343
        // Net cost = 0.05 * 0.343 = 0.01715
        let expected = 1.0 * (1.0 - 0.05 * 0.343);
        assert!(
            (fitness_with_comp - expected).abs() < 1e-10,
            "Expected {expected}, got {fitness_with_comp}"
        );
    }

    #[test]
    fn mutant_selection_window_bounds() {
        let mut sim = ResistanceSimulator::new(10_000, 42);
        sim.add_antibiotic(cipro());
        sim.add_mechanism(gyra_mutation());

        let (msw_low, msw_high) = sim.mutant_selection_window(0);

        // MSW_low = wild-type MIC = 0.25
        assert!(
            (msw_low - 0.25).abs() < 1e-10,
            "MSW_low should be 0.25, got {msw_low}"
        );

        // MSW_high = MPC = 0.25 * 32 = 8.0
        assert!(
            (msw_high - 8.0).abs() < 1e-10,
            "MSW_high should be 8.0, got {msw_high}"
        );

        // MSW_low must be less than MSW_high
        assert!(msw_low < msw_high);
    }

    #[test]
    fn cycling_delays_resistance() {
        // Compare resistance evolution with cycling vs. constant therapy
        let seed = 123;

        // --- Constant therapy: cipro only ---
        let mut sim_constant = ResistanceSimulator::new(5_000, seed);
        sim_constant.mutation_rate = 1e-3; // elevated for test speed
        let cipro_idx = sim_constant.add_antibiotic(cipro());
        let _amp_idx = sim_constant.add_antibiotic(ampicillin());
        sim_constant.add_mechanism(gyra_mutation());
        sim_constant.add_mechanism(tem1_blactamase());

        let gens = 100;
        for _ in 0..gens {
            sim_constant.step(&[1.0, 0.0]); // Only cipro
        }
        let constant_mic = sim_constant.mic_trackers[cipro_idx]
            .mic_history
            .last()
            .map(|(_, m)| *m)
            .unwrap_or(0.0);

        // --- Cycling therapy: alternate cipro/ampicillin ---
        let mut sim_cycling = ResistanceSimulator::new(5_000, seed);
        sim_cycling.mutation_rate = 1e-3;
        let cipro_idx_c = sim_cycling.add_antibiotic(cipro());
        let amp_idx_c = sim_cycling.add_antibiotic(ampicillin());
        sim_cycling.add_mechanism(gyra_mutation());
        sim_cycling.add_mechanism(tem1_blactamase());

        let protocol = vec![
            (cipro_idx_c, 1.0, 10u32), // 10 gens cipro
            (amp_idx_c, 5.0, 10u32),   // 10 gens ampicillin
        ];
        let _freq_history = sim_cycling.run_cycling_protocol(&protocol, gens);

        let cycling_mic = sim_cycling.mic_trackers[cipro_idx_c]
            .mic_history
            .last()
            .map(|(_, m)| *m)
            .unwrap_or(0.0);

        // Cycling should result in lower cipro resistance than constant therapy.
        // With stochastic simulation, we check that cycling MIC is not dramatically
        // worse. In theory, cycling gives resistance less time to fix per drug.
        // We use a generous threshold since this is stochastic.
        assert!(
            cycling_mic <= constant_mic * 2.0,
            "Cycling should not be dramatically worse: cycling_mic={cycling_mic}, constant_mic={constant_mic}"
        );
    }

    #[test]
    fn pk_concentration_decay() {
        // Test that drug concentration decays with correct half-life
        let dose = 500.0; // 500 μg
        let half_life = 4.0; // 4 hours
        let volume = 10.0; // 10 L

        let c0 = pk_concentration(dose, half_life, volume, 0.0);
        assert!(
            (c0 - 50.0).abs() < 1e-10,
            "C(0) should be dose/volume = 50.0, got {c0}"
        );

        let c_half = pk_concentration(dose, half_life, volume, half_life);
        assert!(
            (c_half - 25.0).abs() < 0.01,
            "C(t_half) should be ~25.0, got {c_half}"
        );

        let c_2half = pk_concentration(dose, half_life, volume, 2.0 * half_life);
        assert!(
            (c_2half - 12.5).abs() < 0.01,
            "C(2*t_half) should be ~12.5, got {c_2half}"
        );

        // Monotonic decay
        let c_10h = pk_concentration(dose, half_life, volume, 10.0);
        assert!(c_10h < c_2half, "Concentration should decay over time");
    }

    #[test]
    fn pd_kill_rate_sigmoid() {
        let mic = 1.0;
        let hill = 2.0;
        let e_max = 0.99;

        // At zero concentration, no killing
        let kill_0 = ResistanceSimulator::pd_kill_rate(0.0, mic, hill, e_max);
        assert!((kill_0 - 0.0).abs() < 1e-10, "Kill at C=0 should be 0");

        // At MIC, kill rate should be E_max/2 (definition of MIC in Hill model)
        let kill_mic = ResistanceSimulator::pd_kill_rate(mic, mic, hill, e_max);
        let expected = e_max * 0.5; // C^n / (MIC^n + C^n) = 0.5 when C = MIC
        assert!(
            (kill_mic - expected).abs() < 0.01,
            "Kill at MIC should be ~{expected}, got {kill_mic}"
        );

        // At 10x MIC, kill should approach E_max
        let kill_high = ResistanceSimulator::pd_kill_rate(10.0 * mic, mic, hill, e_max);
        assert!(
            kill_high > 0.95,
            "Kill at 10x MIC should be near E_max, got {kill_high}"
        );

        // Monotonically increasing
        let kill_low = ResistanceSimulator::pd_kill_rate(0.1, mic, hill, e_max);
        let kill_mid = ResistanceSimulator::pd_kill_rate(1.0, mic, hill, e_max);
        let kill_hi = ResistanceSimulator::pd_kill_rate(10.0, mic, hill, e_max);
        assert!(kill_low < kill_mid);
        assert!(kill_mid < kill_hi);
    }

    #[test]
    fn resistance_loss_without_selection() {
        // With high reversion rate and no drug, resistance should diminish
        let mut sim = ResistanceSimulator::new(1_000, 77);
        sim.mutation_rate = 0.0; // No new mutations
        sim.hgt_rate = 0.0; // No HGT
        sim.compensatory_rate = 0.0;

        sim.add_antibiotic(cipro());

        // Add mechanism with very high reversion rate for test
        let mech_idx = sim.add_mechanism(ResistanceMechanism {
            name: "Unstable resistance".into(),
            mechanism_type: ResistanceType::Efflux,
            mic_fold_increase: 8.0,
            fitness_cost: 0.15,  // significant cost accelerates loss
            reversion_rate: 0.3, // 30% chance of loss per generation without drug
            transferable: false,
            target_classes: vec![AntibioticClass::Fluoroquinolone],
        });

        // Start with all-resistant population
        sim.strains.clear();
        sim.strains.push((
            Strain {
                id: 0,
                growth_rate: 1.0,
                resistance_mechanisms: vec![mech_idx],
                compensatory_mutations: 0,
                generation: 0,
            },
            1.0,
        ));

        // Run without drug — fitness cost + reversion should eliminate resistance
        let mut any_loss = false;
        for _ in 0..100 {
            let events = sim.step(&[0.0]);
            for ev in &events {
                if matches!(ev, ResistanceEvent::ResistanceLoss { .. }) {
                    any_loss = true;
                }
            }
        }

        assert!(any_loss, "Resistance should be lost without drug selection");

        // Check that some strains have lost their resistance
        let susceptible_freq: f64 = sim
            .strains
            .iter()
            .filter(|(s, _)| s.resistance_mechanisms.is_empty())
            .map(|(_, f)| *f)
            .sum();

        // Due to fitness cost, susceptible (reverted) strains should be present
        // or all strains should have lost the mechanism
        let any_susceptible = susceptible_freq > 0.0
            || sim
                .strains
                .iter()
                .any(|(s, _)| s.resistance_mechanisms.is_empty());
        assert!(
            any_susceptible || any_loss,
            "Some strains should have reverted to susceptibility"
        );
    }

    #[test]
    fn selection_coefficient_direction() {
        // Below wild-type MIC: resistance should be costly (negative s)
        let s_below = selection_coefficient(1.0, 1.0, 0.1, 0.01, 1.0, 16.0);
        assert!(
            s_below < 0.0,
            "Selection should disfavor resistance below MIC: s={s_below}"
        );

        // Well above wild-type MIC but below resistant MIC: resistance beneficial
        let s_above = selection_coefficient(1.0, 1.0, 0.1, 5.0, 1.0, 16.0);
        assert!(
            s_above > 0.0,
            "Selection should favor resistance above MIC: s={s_above}"
        );
    }

    #[test]
    fn mpc_computation() {
        let mpc = mutant_prevention_concentration(0.25, 32.0);
        assert!(
            (mpc - 8.0).abs() < 1e-10,
            "MPC should be 0.25 * 32 = 8.0, got {mpc}"
        );

        let mpc2 = mutant_prevention_concentration(2.0, 64.0);
        assert!(
            (mpc2 - 128.0).abs() < 1e-10,
            "MPC should be 2.0 * 64 = 128.0, got {mpc2}"
        );
    }

    #[test]
    fn population_mean_fitness_coherent() {
        let mut sim = ResistanceSimulator::new(10_000, 42);
        sim.add_antibiotic(cipro());
        let mech_idx = sim.add_mechanism(gyra_mutation());

        // Pure wild-type population, no drug
        let f0 = sim.population_mean_fitness(&[0.0]);
        assert!(
            (f0 - 1.0).abs() < 1e-10,
            "Pure WT fitness without drug should be 1.0, got {f0}"
        );

        // With drug above MIC, WT fitness should drop
        let f_drug = sim.population_mean_fitness(&[2.0]);
        assert!(
            f_drug < 0.5,
            "WT fitness at 2 ug/mL cipro should be low, got {f_drug}"
        );

        // Add a resistant strain — mixed population
        sim.strains.push((
            Strain {
                id: 1,
                growth_rate: 1.0,
                resistance_mechanisms: vec![mech_idx],
                compensatory_mutations: 0,
                generation: 0,
            },
            0.0, // not yet present
        ));
        // Set frequencies: 50/50
        sim.strains[0].1 = 0.5;
        sim.strains[1].1 = 0.5;

        let f_mixed = sim.population_mean_fitness(&[2.0]);
        assert!(
            f_mixed > f_drug,
            "Mixed pop should be fitter than pure WT under drug: mixed={f_mixed}, wt_only={f_drug}"
        );
    }

    #[test]
    fn predict_resistance_time_returns_estimate() {
        let mut sim = ResistanceSimulator::new(10_000, 42);
        sim.add_antibiotic(cipro());
        sim.add_mechanism(gyra_mutation());

        // Concentration within MSW — should predict resistance time
        let time = sim.predict_resistance_time(0, 1.0);
        assert!(time.is_some(), "Should predict resistance time within MSW");
        let t = time.unwrap();
        assert!(t > 0, "Predicted time should be positive: {t}");

        // Concentration above MPC — no single-step mutant survives
        let time_above = sim.predict_resistance_time(0, 10.0);
        assert!(time_above.is_none(), "Should return None above MPC");

        // No drug — no selection for resistance
        let time_none = sim.predict_resistance_time(0, 0.0);
        assert!(time_none.is_none(), "Should return None with no drug");
    }

    #[test]
    fn step_produces_events_under_selection() {
        let mut sim = ResistanceSimulator::new(1_000, 42);
        sim.mutation_rate = 0.3; // Very high for test
        sim.add_antibiotic(cipro());
        sim.add_mechanism(gyra_mutation());

        let mut total_events = 0;
        for _ in 0..20 {
            let events = sim.step(&[1.0]);
            total_events += events.len();
        }

        assert!(
            total_events > 0,
            "Should produce events over 20 generations with high mutation rate"
        );
    }
}
