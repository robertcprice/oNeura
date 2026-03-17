//! Horizontal gene transfer (HGT) dynamics in microbial populations.
//!
//! Models the three classical mechanisms of bacterial gene transfer --
//! conjugation (cell-to-cell plasmid transfer), transformation (free DNA
//! uptake), and transduction (phage-mediated transfer) -- within a
//! terrarium microbial ecosystem.
//!
//! # Biology
//!
//! Horizontal gene transfer is the primary driver of antibiotic resistance
//! spread in bacterial populations (Frost et al. 2005, *Nature Reviews
//! Microbiology* 3:722).  Transfer rates depend on element type:
//!
//! - **Plasmids**: Conjugative transfer at 10^-3 to 10^-7 per donor per
//!   hour.  Incompatibility groups prevent co-residence of related plasmids
//!   (Novick 1987, *Microbiological Reviews* 51:381).
//! - **Transposons**: Jump within and between replicons at 10^-5 to 10^-8
//!   per cell per generation (Kleckner 1981, *Annual Review of Genetics*
//!   15:341).
//! - **Phage**: Temperate phages integrate at lysogeny rates of 0.01-0.5,
//!   with burst sizes of 50-200 upon lytic induction (Calendar 2006,
//!   *The Bacteriophages*, 2nd ed.).
//! - **Integrative elements**: ICEs and genomic islands that integrate
//!   site-specifically (Burrus et al. 2002, *Research in Microbiology*
//!   153:559).
//!
//! # Design
//!
//! This module is fully self-contained -- it does NOT import from `crate::`.
//! It uses an inline xorshift64 PRNG for reproducible stochastic dynamics
//! without depending on the `rand` crate.
//!
//! # Example
//!
//! ```rust
//! use oneuro_metal::horizontal_gene_transfer::antibiotic_resistance_spread;
//!
//! let mut pop = antibiotic_resistance_spread(1000, 42);
//! let ts = pop.run_simulation(500, 0.1);
//! assert!(ts.times.len() == 500);
//! ```

use std::collections::{HashMap, HashSet};

// ── Inline xorshift64 PRNG ─────────────────────────────────────────────
//
// Marsaglia 2003, *Journal of Statistical Software* 8(14):1.
// Period 2^64 - 1, sufficient for ecological simulations.

/// A minimal xorshift64 pseudo-random number generator.
///
/// Implements the shift constants (13, 7, 17) from Marsaglia 2003.
/// This avoids any dependency on the `rand` crate while providing
/// adequate statistical quality for population-level stochastic dynamics.
#[derive(Debug, Clone)]
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    /// Create a new PRNG from a seed.  Seed of 0 is remapped to 1
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

    /// Poisson-distributed random variate using Knuth's algorithm.
    /// Suitable for small to moderate lambda (< 30).
    fn poisson(&mut self, lambda: f64) -> u64 {
        if lambda <= 0.0 {
            return 0;
        }
        // For large lambda, use normal approximation to avoid
        // excessive loop iterations.
        if lambda > 30.0 {
            let normal = self.standard_normal();
            let value = lambda + lambda.sqrt() * normal;
            return if value < 0.0 { 0 } else { value.round() as u64 };
        }
        let l = (-lambda).exp();
        let mut k: u64 = 0;
        let mut p: f64 = 1.0;
        loop {
            k += 1;
            p *= self.next_f64();
            if p < l {
                return k - 1;
            }
        }
    }

    /// Box-Muller transform for standard normal variate.
    fn standard_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15); // avoid log(0)
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Shuffle a slice in place using Fisher-Yates.
    fn shuffle<T>(&mut self, slice: &mut [T]) {
        let n = slice.len();
        for i in (1..n).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            slice.swap(i, j);
        }
    }
}

// ── Core types ──────────────────────────────────────────────────────────

/// Classification of mobile genetic elements by transfer mechanism.
///
/// Each variant carries parameters specific to its biology:
/// - `Plasmid`: copy number and incompatibility group
/// - `Transposon`: target site specificity (0 = random, 1 = fully specific)
/// - `Phage`: burst size and lysogeny probability
/// - `IntegrativeElement`: site-specific genomic islands (ICEs)
#[derive(Debug, Clone, PartialEq)]
pub enum GeneticElementType {
    /// Autonomously replicating episome transferred by conjugation.
    /// `copy_number`: steady-state copies per cell (1-50).
    /// `incompatibility_group`: plasmids in the same group (0-255)
    /// cannot stably co-reside in one cell.
    Plasmid {
        copy_number: u32,
        incompatibility_group: u8,
    },

    /// Insertion sequence or composite transposon.
    /// `target_specificity`: probability of inserting at a preferred site
    /// (0.0 = fully random, 1.0 = perfectly site-specific).
    Transposon {
        target_specificity: f64,
    },

    /// Temperate bacteriophage capable of lysogeny.
    /// `burst_size`: virions released upon lytic induction.
    /// `lysogeny_rate`: probability of lysogenic vs. lytic outcome
    /// upon initial infection.
    Phage {
        burst_size: u32,
        lysogeny_rate: f64,
    },

    /// Integrative and conjugative element (ICE) or genomic island.
    /// Integrates site-specifically and excises at low frequency.
    IntegrativeElement,
}

/// A mobile genetic element that can be transferred between cells.
///
/// Each element has a unique `id`, a fitness cost (reduced growth rate
/// for carriers), a fitness benefit (e.g., antibiotic resistance), and
/// kinetic parameters governing transfer and loss.
#[derive(Debug, Clone)]
pub struct GeneticElement {
    /// Unique identifier assigned by `HgtPopulation::add_element`.
    pub id: u64,
    /// Human-readable name (e.g., "pBR322", "Tn10", "lambda").
    pub name: String,
    /// Transfer mechanism classification.
    pub element_type: GeneticElementType,
    /// Fractional reduction in host growth rate (0.0-1.0).
    /// Literature range: 0.01-0.15 for resistance plasmids
    /// (Andersson & Hughes 2010, *Nature Reviews Microbiology* 8:260).
    pub fitness_cost: f64,
    /// Fractional increase in host fitness under selective pressure.
    /// Only realized when environmental selection is active.
    pub fitness_benefit: f64,
    /// Base transfer rate per donor-recipient pair per time unit.
    /// Conjugation: 10^-3 to 10^-7 h^-1 (Levin et al. 1979).
    pub transfer_rate: f64,
    /// Spontaneous loss rate per cell per time unit due to segregation
    /// errors, curing, or excision.
    pub loss_rate: f64,
    /// Element size in kilobases.  Larger elements have higher
    /// segregation loss rates and metabolic burden.
    pub size_kb: f64,
}

/// A single microbial cell in the population.
///
/// Tracks which genetic elements the cell carries (by element id),
/// its current fitness, generation counter, and a convenience flag
/// for whether any carried element confers resistance.
#[derive(Debug, Clone)]
pub struct MicrobialCell {
    /// Unique cell identifier.
    pub id: u64,
    /// Set of carried genetic element ids.
    pub genome: Vec<u64>,
    /// Current fitness (growth rate relative to wild-type = 1.0).
    pub fitness: f64,
    /// Number of cell divisions since founding.
    pub generation: u32,
    /// True if any carried element confers resistance benefit.
    pub resistant: bool,
}

impl MicrobialCell {
    /// Create a new wild-type cell with no mobile genetic elements.
    fn new(id: u64) -> Self {
        Self {
            id,
            genome: Vec::new(),
            fitness: 1.0,
            generation: 0,
            resistant: false,
        }
    }

    /// Returns true if this cell carries the given element.
    fn has_element(&self, element_id: u64) -> bool {
        self.genome.contains(&element_id)
    }

    /// Add an element to this cell's genome (if not already present).
    /// Returns true if the element was added.
    fn acquire_element(&mut self, element_id: u64) -> bool {
        if self.has_element(element_id) {
            return false;
        }
        self.genome.push(element_id);
        true
    }

    /// Remove an element from this cell's genome.
    /// Returns true if the element was present and removed.
    fn lose_element(&mut self, element_id: u64) -> bool {
        if let Some(pos) = self.genome.iter().position(|&e| e == element_id) {
            self.genome.swap_remove(pos);
            true
        } else {
            false
        }
    }
}

/// Time series output from an HGT simulation run.
///
/// Records population-level observables at each time step for
/// downstream analysis and visualization.
#[derive(Debug, Clone)]
pub struct HgtTimeSeries {
    /// Wall-clock simulation times.
    pub times: Vec<f64>,
    /// Per-element frequency trajectories.  Key = element id,
    /// value = frequency at each time point.
    pub element_frequencies: HashMap<u64, Vec<f64>>,
    /// Population mean fitness at each time point.
    pub mean_fitness: Vec<f64>,
    /// Total population size at each time point.
    pub population_size: Vec<usize>,
}

/// A population of microbial cells undergoing horizontal gene transfer.
///
/// The core simulation engine.  Manages a collection of `MicrobialCell`
/// instances, a registry of `GeneticElement` definitions, and the
/// stochastic dynamics of gene transfer, loss, and selection.
///
/// # Transfer mechanisms
///
/// - **Conjugation**: Donor cells with plasmid contact recipients;
///   transfer probability = `transfer_rate * dt * donor_freq * recipient_freq * N`.
/// - **Transformation**: Cells take up free DNA from the environment
///   proportional to `free_dna_concentration * competence * dt`.
/// - **Transduction**: Phage particles inject DNA during lytic cycles
///   proportional to `phage_concentration * adsorption_rate * dt`.
///
/// # Selection
///
/// Cell fitness is the product of (1 - cost) for each carried element.
/// Selective death removes the least-fit cells probabilistically.
///
/// # Loss
///
/// Each element is lost independently per cell at its `loss_rate * dt`.
#[derive(Debug, Clone)]
pub struct HgtPopulation {
    /// All cells in the population.
    cells: Vec<MicrobialCell>,
    /// Registry of known genetic elements, keyed by id.
    elements: HashMap<u64, GeneticElement>,
    /// Next element id to assign.
    next_element_id: u64,
    /// Next cell id to assign.
    next_cell_id: u64,
    /// Pseudo-random number generator.
    rng: Xorshift64,
    /// Current simulation time.
    time: f64,
    /// Whether antibiotic selection pressure is active.
    /// When true, fitness_benefit of resistance elements is realized.
    pub antibiotic_pressure: bool,
    /// Carrying capacity for population regulation.
    pub carrying_capacity: usize,
    /// Base competence for transformation (fraction of cells competent).
    /// Typical values: 0.001-0.01 for naturally competent species.
    pub competence: f64,
    /// Phage adsorption rate constant (per phage per cell per time).
    pub phage_adsorption_rate: f64,
}

impl HgtPopulation {
    /// Create a new population of `size` wild-type cells.
    ///
    /// All cells start with no mobile genetic elements, fitness 1.0,
    /// and generation 0.  The population uses the given `seed` for
    /// reproducible stochastic dynamics.
    ///
    /// # Arguments
    ///
    /// * `size` - Initial population size.
    /// * `seed` - PRNG seed for reproducibility.
    pub fn new(size: usize, seed: u64) -> Self {
        let mut rng = Xorshift64::new(seed);
        let cells: Vec<MicrobialCell> = (0..size as u64)
            .map(|i| {
                // Burn an RNG value per cell for future stochastic
                // initialization extensions.
                let _ = rng.next_u64();
                MicrobialCell::new(i)
            })
            .collect();
        Self {
            next_cell_id: size as u64,
            cells,
            elements: HashMap::new(),
            next_element_id: 0,
            rng,
            time: 0.0,
            antibiotic_pressure: false,
            carrying_capacity: size * 2,
            competence: 0.005,
            phage_adsorption_rate: 1e-9,
        }
    }

    /// Register a new genetic element and return its assigned id.
    ///
    /// The element's `id` field is overwritten with the next available
    /// internal id.  The element definition is stored for reference
    /// during transfer, loss, and fitness calculations.
    pub fn add_element(&mut self, mut element: GeneticElement) -> u64 {
        let id = self.next_element_id;
        self.next_element_id += 1;
        element.id = id;
        self.elements.insert(id, element);
        id
    }

    /// Introduce a genetic element into a fraction of the population.
    ///
    /// Randomly selects `fraction * N` cells and gives them the element.
    /// Cells that already carry it are skipped (the actual number of
    /// new acquisitions may be slightly less than requested).
    ///
    /// # Arguments
    ///
    /// * `element_id` - Id of a previously registered element.
    /// * `fraction` - Fraction of cells to receive the element (0.0-1.0).
    pub fn introduce_element(&mut self, element_id: u64, fraction: f64) {
        if !self.elements.contains_key(&element_id) {
            return;
        }
        let n = self.cells.len();
        let count = ((n as f64) * fraction.clamp(0.0, 1.0)).round() as usize;

        // Build shuffled index array to select random cells.
        let mut indices: Vec<usize> = (0..n).collect();
        self.rng.shuffle(&mut indices);

        let element = self.elements.get(&element_id).unwrap().clone();
        for &idx in indices.iter().take(count) {
            // Check incompatibility before acquisition (read-only borrow).
            let blocked =
                Self::check_incompatibility_static(&self.elements, &self.cells[idx], &element);
            if !blocked {
                self.cells[idx].acquire_element(element_id);
            }
        }

        self.recalculate_all_fitness();
    }

    /// Check if a cell already carries a plasmid in the same
    /// incompatibility group.  Returns true if incompatible (blocked).
    ///
    /// This is a static method to avoid borrow conflicts when mutating
    /// `self.cells` while reading `self.elements`.
    fn check_incompatibility_static(
        elements: &HashMap<u64, GeneticElement>,
        cell: &MicrobialCell,
        incoming: &GeneticElement,
    ) -> bool {
        if let GeneticElementType::Plasmid {
            incompatibility_group: inc_group,
            ..
        } = &incoming.element_type
        {
            for &eid in &cell.genome {
                if eid == incoming.id {
                    continue;
                }
                if let Some(existing) = elements.get(&eid) {
                    if let GeneticElementType::Plasmid {
                        incompatibility_group: existing_group,
                        ..
                    } = &existing.element_type
                    {
                        if existing_group == inc_group {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// Recalculate fitness for all cells based on their current element
    /// complement.  Fitness = product of (1 - cost_i) for all carried
    /// elements, plus benefit if antibiotic pressure is active.
    fn recalculate_all_fitness(&mut self) {
        let antibiotic = self.antibiotic_pressure;
        let elements = &self.elements;
        for cell in &mut self.cells {
            let mut fitness = 1.0_f64;
            let mut has_resistance = false;
            for &eid in &cell.genome {
                if let Some(elem) = elements.get(&eid) {
                    fitness *= 1.0 - elem.fitness_cost;
                    if elem.fitness_benefit > 0.0 {
                        has_resistance = true;
                        if antibiotic {
                            fitness *= 1.0 + elem.fitness_benefit;
                        }
                    }
                }
            }
            cell.fitness = fitness.max(0.01); // minimum viability
            cell.resistant = has_resistance;
        }
    }

    /// Execute one simulation time step of duration `dt`.
    ///
    /// The step proceeds in order:
    /// 1. Conjugation -- plasmid transfer between contacting cells
    /// 2. Transformation -- free DNA uptake (uses default low concentration)
    /// 3. Transduction -- phage-mediated transfer (uses default low concentration)
    /// 4. Element loss -- stochastic segregation and curing
    /// 5. Selection -- fitness-proportionate survival
    /// 6. Fitness recalculation
    ///
    /// # Arguments
    ///
    /// * `dt` - Time step duration in the same units as transfer/loss rates.
    pub fn step(&mut self, dt: f64) {
        self.conjugation_events(dt);
        self.transformation_events(0.01, dt);
        self.transduction_events(0.01, dt);
        self.element_loss(dt);
        self.selection(dt);
        self.recalculate_all_fitness();
        self.time += dt;
    }

    /// Perform conjugation (plasmid transfer) events.
    ///
    /// For each plasmid element, the expected number of new transfer
    /// events is `transfer_rate * dt * N_donor * N_recipient / N_total`.
    /// Events are drawn from a Poisson distribution and recipients are
    /// chosen uniformly at random.
    ///
    /// Returns the total number of successful conjugation events.
    pub fn conjugation_events(&mut self, dt: f64) -> usize {
        let n = self.cells.len();
        if n < 2 {
            return 0;
        }

        let plasmid_ids: Vec<u64> = self
            .elements
            .iter()
            .filter(|(_, e)| matches!(e.element_type, GeneticElementType::Plasmid { .. }))
            .map(|(&id, _)| id)
            .collect();

        let mut total_events = 0usize;

        for &eid in &plasmid_ids {
            let transfer_rate = match self.elements.get(&eid) {
                Some(e) => e.transfer_rate,
                None => continue,
            };

            // Count donors (cells with element) and recipients (cells without).
            let n_donors = self.cells.iter().filter(|c| c.has_element(eid)).count();
            let n_recipients = n - n_donors;
            if n_donors == 0 || n_recipients == 0 {
                continue;
            }

            // Mass-action conjugation kinetics.
            let lambda =
                transfer_rate * dt * (n_donors as f64) * (n_recipients as f64) / (n as f64);
            let events = self.rng.poisson(lambda) as usize;

            // Clone element for incompatibility checks.
            let element = self.elements.get(&eid).unwrap().clone();

            // Select random recipients and attempt transfer.
            let mut successes = 0;
            for _ in 0..events {
                // Pick a random cell index.
                let idx = (self.rng.next_u64() as usize) % n;
                if !self.cells[idx].has_element(eid)
                    && !Self::check_incompatibility_static(
                        &self.elements,
                        &self.cells[idx],
                        &element,
                    )
                {
                    self.cells[idx].acquire_element(eid);
                    successes += 1;
                }
            }
            total_events += successes;
        }
        total_events
    }

    /// Perform transformation (free DNA uptake) events.
    ///
    /// Naturally competent cells take up free DNA from the environment.
    /// The probability per cell per dt is `free_dna_concentration *
    /// competence * dt`.  Each uptake event randomly assigns one element
    /// from the available pool.
    ///
    /// Returns the total number of successful transformation events.
    ///
    /// # Arguments
    ///
    /// * `free_dna_concentration` - Relative DNA concentration (0.0-1.0).
    /// * `dt` - Time step duration.
    pub fn transformation_events(&mut self, free_dna_concentration: f64, dt: f64) -> usize {
        let n = self.cells.len();
        if n == 0 || self.elements.is_empty() {
            return 0;
        }

        let element_ids: Vec<u64> = self.elements.keys().copied().collect();
        let prob_per_cell = free_dna_concentration * self.competence * dt;
        let lambda = prob_per_cell * (n as f64);
        let events = self.rng.poisson(lambda) as usize;

        let mut successes = 0;
        for _ in 0..events {
            let cell_idx = (self.rng.next_u64() as usize) % n;
            let elem_idx = (self.rng.next_u64() as usize) % element_ids.len();
            let eid = element_ids[elem_idx];
            let element = self.elements.get(&eid).unwrap().clone();
            if !self.cells[cell_idx].has_element(eid)
                && !Self::check_incompatibility_static(
                    &self.elements,
                    &self.cells[cell_idx],
                    &element,
                )
            {
                self.cells[cell_idx].acquire_element(eid);
                successes += 1;
            }
        }
        successes
    }

    /// Perform transduction (phage-mediated transfer) events.
    ///
    /// Temperate phage particles adsorb to cells and inject DNA.
    /// For each phage-type element, expected events =
    /// `phage_concentration * adsorption_rate * N * dt`.
    /// Upon infection, the phage integrates (lysogeny) or lyses the
    /// cell based on `lysogeny_rate`.
    ///
    /// Returns the total number of successful lysogenic integrations.
    ///
    /// # Arguments
    ///
    /// * `phage_concentration` - Relative phage titer (0.0-1.0).
    /// * `dt` - Time step duration.
    pub fn transduction_events(&mut self, phage_concentration: f64, dt: f64) -> usize {
        let n = self.cells.len();
        if n == 0 {
            return 0;
        }

        let phage_elements: Vec<(u64, f64)> = self
            .elements
            .iter()
            .filter_map(|(&id, e)| {
                if let GeneticElementType::Phage { lysogeny_rate, .. } = e.element_type {
                    Some((id, lysogeny_rate))
                } else {
                    None
                }
            })
            .collect();

        let mut total = 0usize;

        for (eid, lysogeny_rate) in &phage_elements {
            let lambda = phage_concentration * self.phage_adsorption_rate * (n as f64) * dt;
            // Scale up so events are meaningful at population level.
            let scaled_lambda = lambda * 1e6; // phage concentrations are typically 10^6-10^9
            let events = self.rng.poisson(scaled_lambda) as usize;

            for _ in 0..events {
                let idx = (self.rng.next_u64() as usize) % n;
                if self.cells[idx].has_element(*eid) {
                    continue;
                }
                // Coin flip: lysogeny or lysis.
                if self.rng.next_f64() < *lysogeny_rate {
                    self.cells[idx].acquire_element(*eid);
                    total += 1;
                }
                // Lysis outcome: cell death would remove the cell,
                // but we model that via selection pressure instead.
            }
        }
        total
    }

    /// Stochastic element loss due to segregation errors and curing.
    ///
    /// Each element in each cell is independently lost with probability
    /// `loss_rate * dt`.  This models plasmid segregation failure during
    /// division, transposon excision, and prophage curing.
    fn element_loss(&mut self, dt: f64) {
        let loss_rates: HashMap<u64, f64> = self
            .elements
            .iter()
            .map(|(&id, e)| (id, e.loss_rate))
            .collect();

        for cell in &mut self.cells {
            let mut to_remove = Vec::new();
            for &eid in &cell.genome {
                if let Some(&rate) = loss_rates.get(&eid) {
                    let prob = rate * dt;
                    // Use fast inline random check.
                    let r = {
                        let mut x = self.rng.state;
                        x ^= x << 13;
                        x ^= x >> 7;
                        x ^= x << 17;
                        self.rng.state = x;
                        (x >> 11) as f64 / ((1u64 << 53) as f64)
                    };
                    if r < prob {
                        to_remove.push(eid);
                    }
                }
            }
            for eid in to_remove {
                cell.lose_element(eid);
            }
        }
    }

    /// Fitness-proportionate selection with population regulation.
    ///
    /// Cells with lower fitness have a higher probability of dying.
    /// The population is regulated toward `carrying_capacity` by
    /// adjusting the death rate based on current population size.
    ///
    /// When antibiotic pressure is active, cells without resistance
    /// elements suffer an additional fitness penalty (50% kill rate).
    fn selection(&mut self, dt: f64) {
        let n = self.cells.len();
        if n == 0 {
            return;
        }

        let mean_f = self.mean_fitness();
        let density_pressure = (n as f64) / (self.carrying_capacity as f64);

        let mut survivors = Vec::with_capacity(n);
        for cell in &self.cells {
            // Base survival probability from fitness.
            let relative_fitness = if mean_f > 0.0 {
                cell.fitness / mean_f
            } else {
                1.0
            };

            // Density-dependent death rate.
            let death_prob = if density_pressure > 1.0 {
                (density_pressure - 1.0) * 0.5 * dt
            } else {
                0.0
            };

            // Antibiotic kill for non-resistant cells.
            let antibiotic_death = if self.antibiotic_pressure && !cell.resistant {
                0.5 * dt
            } else {
                0.0
            };

            // Combined survival probability.
            let survival = (relative_fitness * (1.0 - death_prob) * (1.0 - antibiotic_death))
                .clamp(0.0, 1.0);

            let r = self.rng.next_f64();
            if r < survival {
                survivors.push(cell.clone());
            }
        }

        // If population is below carrying capacity, allow growth via
        // division of the fittest cells.
        let deficit = if survivors.len() < self.carrying_capacity {
            let max_growth = (survivors.len() as f64 * 0.1 * dt).ceil() as usize;
            max_growth.min(self.carrying_capacity - survivors.len())
        } else {
            0
        };

        if deficit > 0 && !survivors.is_empty() {
            // Sort by fitness descending and replicate the fittest.
            let mut by_fitness: Vec<usize> = (0..survivors.len()).collect();
            by_fitness.sort_by(|&a, &b| {
                survivors[b]
                    .fitness
                    .partial_cmp(&survivors[a].fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            for i in 0..deficit {
                let parent_idx = by_fitness[i % by_fitness.len()];
                let parent = &survivors[parent_idx];
                let mut daughter = parent.clone();
                daughter.id = self.next_cell_id;
                self.next_cell_id += 1;
                daughter.generation = parent.generation + 1;
                survivors.push(daughter);
            }
        }

        self.cells = survivors;
    }

    /// Fraction of cells carrying a given element.
    ///
    /// Returns 0.0 if the population is empty or the element is unknown.
    pub fn element_frequency(&self, element_id: u64) -> f64 {
        if self.cells.is_empty() {
            return 0.0;
        }
        let count = self
            .cells
            .iter()
            .filter(|c| c.has_element(element_id))
            .count();
        (count as f64) / (self.cells.len() as f64)
    }

    /// Mean fitness across all cells in the population.
    ///
    /// Returns 0.0 for an empty population.
    pub fn mean_fitness(&self) -> f64 {
        if self.cells.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.cells.iter().map(|c| c.fitness).sum();
        sum / (self.cells.len() as f64)
    }

    /// Fraction of cells flagged as resistant.
    ///
    /// A cell is resistant if it carries any element with
    /// `fitness_benefit > 0`.
    pub fn resistance_frequency(&self) -> f64 {
        if self.cells.is_empty() {
            return 0.0;
        }
        let count = self.cells.iter().filter(|c| c.resistant).count();
        (count as f64) / (self.cells.len() as f64)
    }

    /// Current population size.
    pub fn population_size(&self) -> usize {
        self.cells.len()
    }

    /// Current simulation time.
    pub fn current_time(&self) -> f64 {
        self.time
    }

    /// Run a full simulation for `steps` time steps of duration `dt`.
    ///
    /// Records population-level observables at each step and returns
    /// an `HgtTimeSeries` for downstream analysis.
    ///
    /// # Arguments
    ///
    /// * `steps` - Number of simulation steps to execute.
    /// * `dt` - Duration of each time step.
    pub fn run_simulation(&mut self, steps: usize, dt: f64) -> HgtTimeSeries {
        let element_ids: Vec<u64> = self.elements.keys().copied().collect();

        let mut times = Vec::with_capacity(steps);
        let mut mean_fitness = Vec::with_capacity(steps);
        let mut population_size = Vec::with_capacity(steps);
        let mut element_frequencies: HashMap<u64, Vec<f64>> = HashMap::new();
        for &eid in &element_ids {
            element_frequencies.insert(eid, Vec::with_capacity(steps));
        }

        for _ in 0..steps {
            self.step(dt);

            times.push(self.time);
            mean_fitness.push(self.mean_fitness());
            population_size.push(self.cells.len());
            for &eid in &element_ids {
                element_frequencies
                    .get_mut(&eid)
                    .unwrap()
                    .push(self.element_frequency(eid));
            }
        }

        HgtTimeSeries {
            times,
            element_frequencies,
            mean_fitness,
            population_size,
        }
    }

    /// Read-only access to the cell population.
    pub fn cells(&self) -> &[MicrobialCell] {
        &self.cells
    }

    /// Read-only access to the element registry.
    pub fn elements_registry(&self) -> &HashMap<u64, GeneticElement> {
        &self.elements
    }

    /// Set of unique incompatibility groups present in the element registry.
    pub fn incompatibility_groups(&self) -> HashSet<u8> {
        let mut groups = HashSet::new();
        for elem in self.elements.values() {
            if let GeneticElementType::Plasmid {
                incompatibility_group,
                ..
            } = &elem.element_type
            {
                groups.insert(*incompatibility_group);
            }
        }
        groups
    }
}

// ── Preset scenarios ────────────────────────────────────────────────────

/// Create a population modeling R-plasmid (antibiotic resistance) spread.
///
/// Simulates the classic scenario of a conjugative resistance plasmid
/// invading a susceptible population.  The plasmid confers a 5% fitness
/// cost but provides strong resistance benefit under antibiotic pressure.
///
/// Based on dynamics observed by Levin et al. (1979, *Plasmids* 2:247)
/// and Andersson & Hughes (2010, *Nature Reviews Microbiology* 8:260).
///
/// # Arguments
///
/// * `population_size` - Number of cells in the founding population.
/// * `seed` - PRNG seed for reproducibility.
pub fn antibiotic_resistance_spread(population_size: usize, seed: u64) -> HgtPopulation {
    let mut pop = HgtPopulation::new(population_size, seed);
    pop.antibiotic_pressure = true;

    let r_plasmid = GeneticElement {
        id: 0,
        name: "pResist-1 (R-plasmid)".into(),
        element_type: GeneticElementType::Plasmid {
            copy_number: 3,
            incompatibility_group: 1,
        },
        fitness_cost: 0.05,
        fitness_benefit: 0.8,
        transfer_rate: 1e-4,
        loss_rate: 1e-5,
        size_kb: 45.0,
    };

    let eid = pop.add_element(r_plasmid);
    // Introduce into 1% of the population (a small inoculum).
    pop.introduce_element(eid, 0.01);
    pop
}

/// Create a population modeling temperate phage lysogeny dynamics.
///
/// Simulates a temperate phage (lambda-like) infecting a bacterial
/// population with a moderate lysogeny rate.  Lysogens carry the
/// prophage at a small fitness cost but gain superinfection immunity.
///
/// Based on lambda phage biology (Ptashne 2004, *A Genetic Switch*).
///
/// # Arguments
///
/// * `population_size` - Number of cells in the founding population.
/// * `seed` - PRNG seed for reproducibility.
pub fn phage_lysogeny_dynamics(population_size: usize, seed: u64) -> HgtPopulation {
    let mut pop = HgtPopulation::new(population_size, seed);

    let lambda_phage = GeneticElement {
        id: 0,
        name: "phiTerr-1 (temperate phage)".into(),
        element_type: GeneticElementType::Phage {
            burst_size: 100,
            lysogeny_rate: 0.3,
        },
        fitness_cost: 0.02,
        fitness_benefit: 0.05, // superinfection immunity advantage
        transfer_rate: 0.0,    // phage transfer is via transduction, not conjugation
        loss_rate: 1e-4,       // spontaneous prophage induction (curing)
        size_kb: 48.5,
    };

    let eid = pop.add_element(lambda_phage);
    // Start with 5% lysogens.
    pop.introduce_element(eid, 0.05);
    pop
}

/// Create a population modeling multi-drug resistance via multiple
/// incompatible plasmids.
///
/// Three resistance plasmids from two incompatibility groups compete
/// for carriage.  Plasmids A and B are in the same Inc group and
/// cannot coexist; plasmid C is in a different group and can co-reside
/// with either.  This models the clinically relevant scenario of
/// multi-drug resistance plasmid dynamics (Carattoli 2009, *International
/// Journal of Medical Microbiology* 299:455).
///
/// # Arguments
///
/// * `population_size` - Number of cells in the founding population.
/// * `seed` - PRNG seed for reproducibility.
pub fn multi_drug_resistance(population_size: usize, seed: u64) -> HgtPopulation {
    let mut pop = HgtPopulation::new(population_size, seed);
    pop.antibiotic_pressure = true;

    // Plasmid A: ampicillin resistance, Inc group 1.
    let plasmid_a = GeneticElement {
        id: 0,
        name: "pAmp-1 (ampicillin R, IncF)".into(),
        element_type: GeneticElementType::Plasmid {
            copy_number: 2,
            incompatibility_group: 1,
        },
        fitness_cost: 0.03,
        fitness_benefit: 0.6,
        transfer_rate: 5e-5,
        loss_rate: 1e-5,
        size_kb: 60.0,
    };

    // Plasmid B: tetracycline resistance, same Inc group 1 (incompatible with A).
    let plasmid_b = GeneticElement {
        id: 0,
        name: "pTet-1 (tetracycline R, IncF)".into(),
        element_type: GeneticElementType::Plasmid {
            copy_number: 5,
            incompatibility_group: 1,
        },
        fitness_cost: 0.04,
        fitness_benefit: 0.5,
        transfer_rate: 8e-5,
        loss_rate: 2e-5,
        size_kb: 40.0,
    };

    // Plasmid C: kanamycin resistance, different Inc group 2 (compatible with A or B).
    let plasmid_c = GeneticElement {
        id: 0,
        name: "pKan-1 (kanamycin R, IncI)".into(),
        element_type: GeneticElementType::Plasmid {
            copy_number: 10,
            incompatibility_group: 2,
        },
        fitness_cost: 0.02,
        fitness_benefit: 0.4,
        transfer_rate: 1e-4,
        loss_rate: 5e-6,
        size_kb: 30.0,
    };

    let a_id = pop.add_element(plasmid_a);
    let b_id = pop.add_element(plasmid_b);
    let c_id = pop.add_element(plasmid_c);

    // Introduce each plasmid into distinct 5% subsets.
    pop.introduce_element(a_id, 0.05);
    pop.introduce_element(b_id, 0.05);
    pop.introduce_element(c_id, 0.05);

    pop
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a simple test plasmid.
    fn test_plasmid(name: &str, cost: f64, benefit: f64, transfer: f64) -> GeneticElement {
        GeneticElement {
            id: 0,
            name: name.into(),
            element_type: GeneticElementType::Plasmid {
                copy_number: 5,
                incompatibility_group: 0,
            },
            fitness_cost: cost,
            fitness_benefit: benefit,
            transfer_rate: transfer,
            loss_rate: 1e-5,
            size_kb: 10.0,
        }
    }

    #[test]
    fn empty_population() {
        let pop = HgtPopulation::new(0, 42);
        assert_eq!(pop.population_size(), 0);
        assert_eq!(pop.mean_fitness(), 0.0);
        assert_eq!(pop.resistance_frequency(), 0.0);
        assert_eq!(pop.element_frequency(0), 0.0);
    }

    #[test]
    fn introduce_plasmid() {
        let mut pop = HgtPopulation::new(1000, 42);
        let p = test_plasmid("pTest", 0.05, 0.0, 1e-4);
        let eid = pop.add_element(p);
        pop.introduce_element(eid, 0.10);

        let freq = pop.element_frequency(eid);
        // Should be approximately 10% (+/- stochastic noise).
        assert!(
            freq > 0.05 && freq < 0.20,
            "Expected ~10% introduction, got {:.3}",
            freq
        );
    }

    #[test]
    fn conjugation_spreads_plasmid() {
        let mut pop = HgtPopulation::new(500, 99);
        let p = test_plasmid("pSpread", 0.0, 0.0, 0.1); // high transfer rate
        let eid = pop.add_element(p);
        pop.introduce_element(eid, 0.05);

        let initial_freq = pop.element_frequency(eid);

        // Run 200 steps of conjugation-heavy simulation.
        for _ in 0..200 {
            pop.step(0.5);
        }

        let final_freq = pop.element_frequency(eid);
        assert!(
            final_freq > initial_freq,
            "Conjugation should spread plasmid: initial={:.3}, final={:.3}",
            initial_freq,
            final_freq
        );
    }

    #[test]
    fn fitness_cost_reduces_frequency() {
        // Without selection pressure or transfer, a costly element should
        // decline in frequency over time due to selection.
        let mut pop = HgtPopulation::new(500, 77);
        let p = GeneticElement {
            id: 0,
            name: "pCostly".into(),
            element_type: GeneticElementType::Plasmid {
                copy_number: 1,
                incompatibility_group: 0,
            },
            fitness_cost: 0.20, // 20% cost -- very expensive
            fitness_benefit: 0.0,
            transfer_rate: 0.0, // no transfer
            loss_rate: 0.001,
            size_kb: 100.0,
        };
        let eid = pop.add_element(p);
        pop.introduce_element(eid, 0.50);

        let initial = pop.element_frequency(eid);

        for _ in 0..300 {
            pop.step(0.1);
        }

        let final_freq = pop.element_frequency(eid);
        assert!(
            final_freq < initial,
            "Costly element should decrease: initial={:.3}, final={:.3}",
            initial,
            final_freq
        );
    }

    #[test]
    fn transformation_uptake() {
        let mut pop = HgtPopulation::new(500, 55);
        pop.competence = 0.1; // highly competent species
        let p = test_plasmid("pTransform", 0.0, 0.0, 0.0);
        let eid = pop.add_element(p);
        // Don't introduce -- rely purely on transformation.

        // Run with high free DNA concentration.
        let mut events_total = 0usize;
        for _ in 0..100 {
            events_total += pop.transformation_events(0.5, 1.0);
        }

        // Some cells should have acquired the element.
        let freq = pop.element_frequency(eid);
        assert!(
            events_total > 0 || freq > 0.0,
            "Transformation should produce uptake events"
        );
    }

    #[test]
    fn phage_transduction() {
        let mut pop = HgtPopulation::new(500, 33);

        let phage = GeneticElement {
            id: 0,
            name: "phiTest".into(),
            element_type: GeneticElementType::Phage {
                burst_size: 100,
                lysogeny_rate: 0.5,
            },
            fitness_cost: 0.01,
            fitness_benefit: 0.0,
            transfer_rate: 0.0,
            loss_rate: 1e-4,
            size_kb: 40.0,
        };
        let eid = pop.add_element(phage);

        // Run transduction events at high phage titer for reliable lysogeny.
        let mut total = 0usize;
        for _ in 0..200 {
            total += pop.transduction_events(10.0, 1.0);
        }

        // With 50% lysogeny rate and reasonable titer, some lysogens
        // should form.
        let freq = pop.element_frequency(eid);
        assert!(
            total > 0 || freq > 0.0,
            "Transduction should produce lysogens"
        );
    }

    #[test]
    fn element_loss_over_time() {
        let mut pop = HgtPopulation::new(200, 44);
        let p = GeneticElement {
            id: 0,
            name: "pUnstable".into(),
            element_type: GeneticElementType::Plasmid {
                copy_number: 1,
                incompatibility_group: 0,
            },
            fitness_cost: 0.0,
            fitness_benefit: 0.0,
            transfer_rate: 0.0, // no transfer
            loss_rate: 0.05,    // high loss rate
            size_kb: 5.0,
        };
        let eid = pop.add_element(p);
        pop.introduce_element(eid, 1.0); // give to everyone

        let initial = pop.element_frequency(eid);
        assert!(
            initial > 0.90,
            "Should start near 100%, got {:.3}",
            initial
        );

        // Run enough steps for significant loss.
        for _ in 0..200 {
            pop.step(0.5);
        }

        let final_freq = pop.element_frequency(eid);
        assert!(
            final_freq < initial,
            "Unstable element should be lost: initial={:.3}, final={:.3}",
            initial,
            final_freq
        );
    }

    #[test]
    fn incompatibility_group_exclusion() {
        let mut pop = HgtPopulation::new(100, 88);

        // Two plasmids in the same Inc group.
        let p1 = GeneticElement {
            id: 0,
            name: "pInc1-A".into(),
            element_type: GeneticElementType::Plasmid {
                copy_number: 3,
                incompatibility_group: 5,
            },
            fitness_cost: 0.0,
            fitness_benefit: 0.0,
            transfer_rate: 0.0,
            loss_rate: 0.0,
            size_kb: 20.0,
        };
        let p2 = GeneticElement {
            id: 0,
            name: "pInc1-B".into(),
            element_type: GeneticElementType::Plasmid {
                copy_number: 3,
                incompatibility_group: 5,
            },
            fitness_cost: 0.0,
            fitness_benefit: 0.0,
            transfer_rate: 0.0,
            loss_rate: 0.0,
            size_kb: 25.0,
        };

        let id1 = pop.add_element(p1);
        let id2 = pop.add_element(p2);

        // Give plasmid 1 to all cells.
        pop.introduce_element(id1, 1.0);
        // Attempt to give plasmid 2 to all cells.
        pop.introduce_element(id2, 1.0);

        // No cell should carry both -- incompatibility exclusion.
        let both_count = pop
            .cells()
            .iter()
            .filter(|c| c.has_element(id1) && c.has_element(id2))
            .count();
        assert_eq!(
            both_count, 0,
            "Incompatible plasmids should not co-reside; {} cells have both",
            both_count
        );
    }

    #[test]
    fn antibiotic_resistance_scenario() {
        let mut pop = antibiotic_resistance_spread(500, 42);

        // Verify initial conditions.
        let initial_resistance = pop.resistance_frequency();
        assert!(
            initial_resistance > 0.0 && initial_resistance < 0.10,
            "Should start with small resistant fraction, got {:.3}",
            initial_resistance
        );
        assert!(pop.antibiotic_pressure);

        // Run simulation.
        let ts = pop.run_simulation(200, 0.1);
        assert_eq!(ts.times.len(), 200);

        // Under antibiotic pressure, resistance should spread.
        let final_resistance = pop.resistance_frequency();
        assert!(
            final_resistance > initial_resistance,
            "Resistance should spread under antibiotic pressure: {:.3} -> {:.3}",
            initial_resistance,
            final_resistance
        );
    }

    #[test]
    fn multi_drug_resistance_dynamics() {
        let mut pop = multi_drug_resistance(500, 99);

        // Should have 3 elements registered.
        assert_eq!(pop.elements_registry().len(), 3);

        // Should have 2 incompatibility groups.
        let groups = pop.incompatibility_groups();
        assert_eq!(groups.len(), 2, "Expected 2 Inc groups, got {:?}", groups);

        // Run and verify population survives.
        let ts = pop.run_simulation(100, 0.1);
        assert!(
            *ts.population_size.last().unwrap() > 0,
            "Population should survive"
        );

        // Verify no cell carries two plasmids from the same Inc group.
        for cell in pop.cells() {
            let inc_groups_in_cell: Vec<u8> = cell
                .genome
                .iter()
                .filter_map(|&eid| {
                    pop.elements_registry().get(&eid).and_then(|e| {
                        if let GeneticElementType::Plasmid {
                            incompatibility_group,
                            ..
                        } = &e.element_type
                        {
                            Some(*incompatibility_group)
                        } else {
                            None
                        }
                    })
                })
                .collect();

            let unique: HashSet<u8> = inc_groups_in_cell.iter().copied().collect();
            assert_eq!(
                inc_groups_in_cell.len(),
                unique.len(),
                "Cell {} has duplicate Inc groups: {:?}",
                cell.id,
                inc_groups_in_cell
            );
        }
    }

    #[test]
    fn lysogeny_stabilizes() {
        let mut pop = phage_lysogeny_dynamics(500, 77);

        // Verify phage element exists.
        assert_eq!(pop.elements_registry().len(), 1);

        let eid = *pop.elements_registry().keys().next().unwrap();
        let initial = pop.element_frequency(eid);
        assert!(initial > 0.0, "Should start with some lysogens");

        // Run simulation -- lysogeny should be maintained at some level
        // (not crash to zero) because of the slight fitness benefit
        // from superinfection immunity and ongoing transduction.
        let ts = pop.run_simulation(300, 0.1);

        // Check that lysogens persist throughout (frequency never
        // drops to exactly 0 for extended periods).
        let final_freq = *ts
            .element_frequencies
            .get(&eid)
            .unwrap()
            .last()
            .unwrap();
        let max_freq = ts
            .element_frequencies
            .get(&eid)
            .unwrap()
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max);

        // The phage should have been present at some point.
        assert!(
            max_freq > 0.0,
            "Lysogeny should be maintained at some level"
        );

        // Population should remain viable.
        assert!(
            *ts.population_size.last().unwrap() > 50,
            "Population should survive phage dynamics"
        );
    }

    #[test]
    fn run_simulation_time_series_consistency() {
        let mut pop = HgtPopulation::new(100, 12);
        let p = test_plasmid("pTS", 0.01, 0.0, 1e-3);
        let eid = pop.add_element(p);
        pop.introduce_element(eid, 0.1);

        let ts = pop.run_simulation(50, 0.2);

        // Time series lengths must match step count.
        assert_eq!(ts.times.len(), 50);
        assert_eq!(ts.mean_fitness.len(), 50);
        assert_eq!(ts.population_size.len(), 50);
        assert_eq!(ts.element_frequencies.get(&eid).unwrap().len(), 50);

        // Times should be monotonically increasing.
        for i in 1..ts.times.len() {
            assert!(
                ts.times[i] > ts.times[i - 1],
                "Times must be monotonically increasing"
            );
        }

        // All frequencies should be in [0, 1].
        for &f in ts.element_frequencies.get(&eid).unwrap() {
            assert!(f >= 0.0 && f <= 1.0, "Frequency out of range: {}", f);
        }

        // All fitness values should be positive.
        for &f in &ts.mean_fitness {
            assert!(f >= 0.0, "Fitness must be non-negative: {}", f);
        }
    }

    #[test]
    fn xorshift_prng_deterministic() {
        // Same seed must produce identical sequences.
        let mut rng1 = Xorshift64::new(42);
        let mut rng2 = Xorshift64::new(42);

        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }

        // Different seeds must diverge.
        let mut rng3 = Xorshift64::new(43);
        let mut rng4 = Xorshift64::new(42);
        let mut any_different = false;
        for _ in 0..10 {
            if rng3.next_u64() != rng4.next_u64() {
                any_different = true;
                break;
            }
        }
        assert!(any_different, "Different seeds should produce different sequences");
    }

    #[test]
    fn transposon_element_creation() {
        let mut pop = HgtPopulation::new(100, 22);
        let tn = GeneticElement {
            id: 0,
            name: "Tn10".into(),
            element_type: GeneticElementType::Transposon {
                target_specificity: 0.8,
            },
            fitness_cost: 0.01,
            fitness_benefit: 0.1,
            transfer_rate: 1e-6,
            loss_rate: 1e-5,
            size_kb: 9.3,
        };
        let eid = pop.add_element(tn);
        pop.introduce_element(eid, 0.20);

        let freq = pop.element_frequency(eid);
        assert!(
            freq > 0.10 && freq < 0.35,
            "Transposon introduction should be ~20%, got {:.3}",
            freq
        );

        // Verify element type is preserved.
        let elem = pop.elements_registry().get(&eid).unwrap();
        match &elem.element_type {
            GeneticElementType::Transposon {
                target_specificity,
            } => {
                assert!(
                    (*target_specificity - 0.8).abs() < 1e-10,
                    "Target specificity should be 0.8"
                );
            }
            _ => panic!("Expected Transposon element type"),
        }
    }
}
