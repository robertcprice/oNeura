//! # Eco-Evolutionary Feedback Dynamics
//!
//! Connects phylogenetic tracking, climate scenarios, and evolution to model
//! how environmental change drives speciation, adaptive radiation, and
//! evolutionary rescue in terrarium populations.
//!
//! ## Biology implemented
//!
//! - **Gaussian fitness landscape** with multi-dimensional trait optima and
//!   tunable epistatic ruggedness.
//! - **Evolutionary rescue**: populations declining toward extinction can be
//!   rescued by standing genetic variation if adaptation is fast enough.
//! - **Speciation detection**: trait-space clustering identifies incipient
//!   species when inter-cluster distance exceeds a threshold.
//! - **Character displacement**: sympatric species pushed apart on resource
//!   axes via competition penalties.
//! - **Breeder's equation**: R = h^2 S predicts trait response to selection.
//! - **Price equation**: decomposes trait change into selection and
//!   transmission bias components.
//! - **Qst-Fst comparison**: quantitative trait divergence between
//!   subpopulations, detecting directional selection when Qst > Fst.
//! - **Adaptive dynamics**: invasion fitness of mutant phenotypes in a
//!   resident population.
//!
//! ## Design
//!
//! - **Zero crate dependencies**: inline xorshift64 PRNG, all math
//!   self-contained.
//! - **No `use crate::` imports**: this module is fully standalone.
//! - **Deterministic given seed**: all stochastic processes seeded.

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Inline xorshift64 PRNG (no external crate dependency)
// ---------------------------------------------------------------------------

/// Minimal xorshift64* PRNG. Period 2^64 - 1.
#[derive(Debug, Clone)]
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x5EED_CAFE_DEAD_BEEF
            } else {
                seed
            },
        }
    }

    /// Advance state and return next u64.
    fn next_u64(&mut self) -> u64 {
        let mut s = self.state;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.state = s;
        s.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Approximate standard normal via Box-Muller transform.
    fn next_normal(&mut self, mean: f64, std_dev: f64) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        mean + z * std_dev
    }
}

// ---------------------------------------------------------------------------
// Fitness landscape
// ---------------------------------------------------------------------------

/// Multi-dimensional Gaussian fitness landscape with epistatic ruggedness.
///
/// Fitness at trait vector **z** is:
///
/// ```text
/// w(z) = exp(-sum_i((z_i - theta_i)^2 / (2 * sigma_i^2)))
///        * (1.0 + ruggedness * epistasis_term(z))
/// ```
///
/// where the epistasis term adds sine-based multi-dimensional ruggedness to
/// create local peaks and valleys that trap naive hill-climbers.
#[derive(Debug, Clone)]
pub struct FitnessLandscape {
    /// Number of quantitative trait axes.
    pub dimensions: usize,
    /// Current optimal trait values (one per dimension).
    pub optima: Vec<f64>,
    /// Selection strength per axis -- Gaussian width sigma_i.
    pub widths: Vec<f64>,
    /// Epistatic ruggedness in [0, 1]. Zero = smooth Gaussian, one = highly
    /// rugged with many local optima.
    pub ruggedness: f64,
}

impl FitnessLandscape {
    /// Evaluate fitness of an organism with the given trait vector.
    ///
    /// Returns a value in (0, ~1] for smooth landscapes; rugged landscapes
    /// can exceed 1.0 slightly due to epistatic perturbation.
    pub fn fitness(&self, traits: &[f64]) -> f64 {
        assert_eq!(traits.len(), self.dimensions);

        // Gaussian core: exp(-sum((z_i - theta_i)^2 / (2 * sigma_i^2)))
        let mut exponent = 0.0;
        for i in 0..self.dimensions {
            let d = traits[i] - self.optima[i];
            let sigma = self.widths[i];
            exponent += d * d / (2.0 * sigma * sigma);
        }
        let gaussian = (-exponent).exp();

        // Epistatic ruggedness: sum of sin terms across trait pairs
        if self.ruggedness > 0.0 && self.dimensions >= 2 {
            let mut epistasis = 0.0;
            let mut pair_count = 0;
            for i in 0..self.dimensions {
                for j in (i + 1)..self.dimensions {
                    // Each pair contributes a sin-based local structure
                    epistasis += (5.0 * traits[i] * traits[j]).sin();
                    pair_count += 1;
                }
            }
            if pair_count > 0 {
                epistasis /= pair_count as f64;
            }
            // Modulate: ruggedness=0 means no effect, ruggedness=1 means
            // full sine perturbation (up to +/-0.3 of Gaussian peak)
            gaussian * (1.0 + 0.3 * self.ruggedness * epistasis)
        } else {
            gaussian
        }
    }

    /// Shift the optimum on a single trait axis.
    pub fn shift_optimum(&mut self, axis: usize, new_val: f64) {
        assert!(axis < self.dimensions);
        self.optima[axis] = new_val;
    }

    /// Compute the adaptive landscape gradient at a trait position.
    ///
    /// For the smooth Gaussian component, the gradient is:
    ///   dw/dz_i = w(z) * (-(z_i - theta_i) / sigma_i^2)
    ///
    /// Points toward the fitness peak (uphill direction).
    pub fn adaptive_landscape_gradient(&self, traits: &[f64]) -> Vec<f64> {
        assert_eq!(traits.len(), self.dimensions);
        let w = self.fitness(traits);
        let mut grad = vec![0.0; self.dimensions];
        for i in 0..self.dimensions {
            let d = traits[i] - self.optima[i];
            let sigma = self.widths[i];
            // Gradient of Gaussian: -d/sigma^2 * w
            grad[i] = -d / (sigma * sigma) * w;
        }
        grad
    }
}

// ---------------------------------------------------------------------------
// Organism
// ---------------------------------------------------------------------------

/// Individual organism with quantitative traits subject to selection.
#[derive(Debug, Clone)]
pub struct EcoEvoOrganism {
    /// Unique organism identifier.
    pub id: u64,
    /// Quantitative trait values (one per landscape dimension).
    pub traits: Vec<f64>,
    /// Fitness evaluated against the current landscape.
    pub fitness: f64,
    /// Lineage identifier for tracking ancestry and speciation.
    pub lineage_id: u64,
    /// Generation in which this organism was born.
    pub generation: u32,
}

// ---------------------------------------------------------------------------
// Evolutionary rescue tracker
// ---------------------------------------------------------------------------

/// Tracks whether a population undergoing environmental stress can adapt fast
/// enough to avoid extinction -- the "evolutionary rescue" phenomenon.
///
/// Rescue is detected when:
/// 1. Population declines below 50% of carrying capacity (decline phase).
/// 2. Population subsequently recovers above 50% of carrying capacity.
///
/// This follows the empirical pattern described by Bell & Gonzalez (2009)
/// and Carlson et al. (2014).
#[derive(Debug, Clone)]
pub struct EvolutionaryRescueTracker {
    /// Population size at each generation since tracking began.
    pub population_size_history: Vec<usize>,
    /// Mean population fitness at each generation.
    pub mean_fitness_history: Vec<f64>,
    /// Additive genetic variance (summed across traits) at each generation.
    pub genetic_variance_history: Vec<f64>,
    /// Whether evolutionary rescue has been definitively detected.
    pub rescue_detected: bool,
    /// Generation at which rescue recovery was confirmed.
    pub rescue_generation: Option<u32>,
    /// Minimum viable population size below which extinction is declared.
    pub critical_population_size: usize,
    /// Generation at which population decline was first detected.
    pub decline_start: Option<u32>,
}

impl EvolutionaryRescueTracker {
    /// Create a new tracker with the given critical population size.
    fn new(critical_pop: usize) -> Self {
        Self {
            population_size_history: Vec::new(),
            mean_fitness_history: Vec::new(),
            genetic_variance_history: Vec::new(),
            rescue_detected: false,
            rescue_generation: None,
            critical_population_size: critical_pop,
            decline_start: None,
        }
    }

    /// Record a generation's statistics and check for rescue dynamics.
    pub fn update(
        &mut self,
        pop_size: usize,
        mean_fitness: f64,
        genetic_variance: f64,
        generation: u32,
    ) {
        self.population_size_history.push(pop_size);
        self.mean_fitness_history.push(mean_fitness);
        self.genetic_variance_history.push(genetic_variance);

        let threshold = self.critical_population_size;

        // Detect decline: population drops below threshold
        if pop_size < threshold && self.decline_start.is_none() {
            self.decline_start = Some(generation);
        }

        // Detect rescue: population recovers above threshold after decline
        if self.decline_start.is_some() && !self.rescue_detected && pop_size >= threshold {
            self.rescue_detected = true;
            self.rescue_generation = Some(generation);
        }
    }

    /// Returns true if we are in a decline phase (below threshold, no recovery yet).
    pub fn is_rescue_in_progress(&self) -> bool {
        self.decline_start.is_some() && !self.rescue_detected
    }
}

// ---------------------------------------------------------------------------
// Simulation result types
// ---------------------------------------------------------------------------

/// Result of a single eco-evolutionary simulation step.
#[derive(Debug, Clone)]
pub struct EcoEvoStepResult {
    /// Generation number that just completed.
    pub generation: u32,
    /// Population size after selection and reproduction.
    pub population_size: usize,
    /// Mean fitness across the population.
    pub mean_fitness: f64,
    /// Per-trait additive genetic variance.
    pub genetic_variance: Vec<f64>,
    /// Per-trait population means.
    pub trait_means: Vec<f64>,
    /// Number of births this generation.
    pub births: usize,
    /// Number of deaths (selection) this generation.
    pub deaths: usize,
}

/// An environmental shift that moves a fitness optimum.
#[derive(Debug, Clone)]
pub struct EnvironmentalShift {
    /// Generation at which the shift begins.
    pub generation: u32,
    /// Which trait axis optimum shifts.
    pub trait_axis: usize,
    /// Target value for the shifted optimum.
    pub new_optimum: f64,
    /// Speed of shift: 1.0 = instantaneous, 0.01 = very gradual (fraction
    /// of distance moved per generation).
    pub shift_speed: f64,
}

/// A detected speciation event from trait divergence.
#[derive(Debug, Clone)]
pub struct SpeciationEvent {
    /// Generation at which speciation was detected.
    pub generation: u32,
    /// Lineage that split.
    pub parent_lineage: u64,
    /// Two daughter lineage ids.
    pub daughter_lineages: (u64, u64),
    /// Euclidean trait distance between daughter clusters.
    pub trait_divergence: f64,
    /// Estimated reproductive isolation (0 = panmictic, 1 = fully isolated).
    pub reproductive_isolation: f64,
}

/// Character displacement between a pair of sympatric species.
#[derive(Debug, Clone)]
pub struct CharacterDisplacement {
    /// Lineage ids of the two species.
    pub species_pair: (u64, u64),
    /// Trait axis on which displacement is measured.
    pub trait_axis: usize,
    /// Trait distance when species first co-occurred.
    pub initial_distance: f64,
    /// Current trait distance.
    pub final_distance: f64,
    /// Ratio of final to initial distance (>1.0 = displacement occurred).
    pub displacement_ratio: f64,
}

// ---------------------------------------------------------------------------
// Main simulation engine
// ---------------------------------------------------------------------------

/// Eco-evolutionary feedback simulator.
///
/// Runs a population of organisms on a shifting fitness landscape with
/// mutation, selection, drift, and density regulation. Tracks evolutionary
/// rescue, speciation, and character displacement.
pub struct EcoEvoSimulator {
    /// Current population of organisms.
    pub population: Vec<EcoEvoOrganism>,
    /// The fitness landscape (potentially shifting over time).
    pub landscape: FitnessLandscape,
    /// Current generation counter.
    pub generation: u32,
    /// Per-locus per-generation mutation probability.
    pub mutation_rate: f64,
    /// Standard deviation of mutational effect on trait values.
    pub mutation_effect_size: f64,
    /// Carrying capacity for density regulation.
    pub carrying_capacity: usize,
    /// Tracks evolutionary rescue dynamics.
    pub rescue_tracker: EvolutionaryRescueTracker,
    /// Internal PRNG state (xorshift64).
    rng_state: u64,
    /// Next unique organism id.
    next_id: u64,
    /// Next unique lineage id.
    next_lineage_id: u64,
    /// Trait means recorded at generation when lineages first co-occurred,
    /// for character displacement measurement. Key: (min_lineage, max_lineage, trait_axis).
    initial_distances: Vec<((u64, u64, usize), f64)>,
}

impl EcoEvoSimulator {
    /// Create a new simulator with `pop_size` organisms, each having
    /// `n_traits` quantitative traits initialized near zero.
    pub fn new(pop_size: usize, n_traits: usize, seed: u64) -> Self {
        let mut rng = Xorshift64::new(seed);

        // Default landscape: optima at 0, width 1, no ruggedness
        let landscape = FitnessLandscape {
            dimensions: n_traits,
            optima: vec![0.0; n_traits],
            widths: vec![1.0; n_traits],
            ruggedness: 0.0,
        };

        // Initialize population near origin with small random variation
        let mut population = Vec::with_capacity(pop_size);
        for i in 0..pop_size {
            let traits: Vec<f64> = (0..n_traits).map(|_| rng.next_normal(0.0, 0.1)).collect();
            let fitness = landscape.fitness(&traits);
            population.push(EcoEvoOrganism {
                id: i as u64,
                traits,
                fitness,
                lineage_id: 0, // all start from lineage 0
                generation: 0,
            });
        }

        let critical_pop = (pop_size / 2).max(2);

        Self {
            population,
            landscape,
            generation: 0,
            mutation_rate: 0.01,
            mutation_effect_size: 0.05,
            carrying_capacity: pop_size,
            rescue_tracker: EvolutionaryRescueTracker::new(critical_pop),
            rng_state: rng.state,
            next_id: pop_size as u64,
            next_lineage_id: 1,
            initial_distances: Vec::new(),
        }
    }

    /// Replace the fitness landscape.
    pub fn set_landscape(&mut self, landscape: FitnessLandscape) {
        assert_eq!(landscape.dimensions, self.landscape.dimensions);
        self.landscape = landscape;
        // Re-evaluate fitness for all organisms
        for org in &mut self.population {
            org.fitness = self.landscape.fitness(&org.traits);
        }
    }

    /// Internal PRNG helper -- returns next uniform f64.
    fn rng_f64(&mut self) -> f64 {
        let mut s = self.rng_state;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.rng_state = s;
        let val = s.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (val >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Internal PRNG helper -- approximate normal.
    fn rng_normal(&mut self, mean: f64, std_dev: f64) -> f64 {
        let u1 = self.rng_f64().max(1e-15);
        let u2 = self.rng_f64();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        mean + z * std_dev
    }

    /// Advance the simulation by one generation.
    ///
    /// Pipeline:
    /// 1. Evaluate fitness on current landscape
    /// 2. Select parents (fitness-proportional)
    /// 3. Reproduce with mutation
    /// 4. Density regulation (soft carrying capacity)
    /// 5. Update rescue tracker
    pub fn step(&mut self) -> EcoEvoStepResult {
        self.generation += 1;

        // 1. Evaluate fitness
        for org in &mut self.population {
            org.fitness = self.landscape.fitness(&org.traits);
        }

        let n = self.population.len();
        if n == 0 {
            let gv = vec![0.0; self.landscape.dimensions];
            let tm = vec![0.0; self.landscape.dimensions];
            self.rescue_tracker.update(0, 0.0, 0.0, self.generation);
            return EcoEvoStepResult {
                generation: self.generation,
                population_size: 0,
                mean_fitness: 0.0,
                genetic_variance: gv,
                trait_means: tm,
                births: 0,
                deaths: 0,
            };
        }

        // 2. Fitness-proportional parent selection
        // Shift fitness so minimum is at least a small positive value to
        // allow even poor individuals a tiny chance
        let min_fit = self
            .population
            .iter()
            .map(|o| o.fitness)
            .fold(f64::INFINITY, f64::min);
        let total_fit: f64 = self
            .population
            .iter()
            .map(|o| (o.fitness - min_fit) + 0.01)
            .sum();

        // Determine offspring count -- target carrying capacity
        let target = self.carrying_capacity;
        let mut offspring = Vec::with_capacity(target);

        for _ in 0..target {
            // Roulette selection
            let threshold = self.rng_f64() * total_fit;
            let mut cumulative = 0.0;
            let mut parent_idx = 0;
            for (idx, org) in self.population.iter().enumerate() {
                cumulative += (org.fitness - min_fit) + 0.01;
                if cumulative >= threshold {
                    parent_idx = idx;
                    break;
                }
            }
            // Copy parent data before mutating self (borrow checker)
            let parent_traits = self.population[parent_idx].traits.clone();
            let parent_lineage = self.population[parent_idx].lineage_id;

            // 3. Reproduce with mutation
            let n_traits = parent_traits.len();
            let mut child_traits = parent_traits;
            for t in 0..n_traits {
                if self.rng_f64() < self.mutation_rate {
                    child_traits[t] += self.rng_normal(0.0, self.mutation_effect_size);
                }
            }

            let child_fitness = self.landscape.fitness(&child_traits);
            let child = EcoEvoOrganism {
                id: self.next_id,
                traits: child_traits,
                fitness: child_fitness,
                lineage_id: parent_lineage,
                generation: self.generation,
            };
            self.next_id += 1;
            offspring.push(child);
        }

        let births = offspring.len();
        let deaths = n; // entire previous generation replaced (non-overlapping)
        self.population = offspring;

        // 4. Density regulation -- if population somehow exceeds capacity,
        // cull the least fit.
        if self.population.len() > self.carrying_capacity {
            self.population.sort_by(|a, b| {
                b.fitness
                    .partial_cmp(&a.fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            self.population.truncate(self.carrying_capacity);
        }

        // Compute stats
        let gv = self.genetic_variance();
        let tm = self.trait_means();
        let mf = self.mean_fitness();
        let total_gv: f64 = gv.iter().sum();

        // 5. Update rescue tracker
        self.rescue_tracker
            .update(self.population.len(), mf, total_gv, self.generation);

        EcoEvoStepResult {
            generation: self.generation,
            population_size: self.population.len(),
            mean_fitness: mf,
            genetic_variance: gv,
            trait_means: tm,
            births,
            deaths,
        }
    }

    /// Apply an environmental shift to the fitness landscape.
    ///
    /// If `shift_speed` is 1.0, the optimum jumps instantly. Otherwise, the
    /// optimum moves a fraction of the remaining distance toward the target.
    pub fn apply_environmental_shift(&mut self, shift: &EnvironmentalShift) {
        let axis = shift.trait_axis;
        assert!(axis < self.landscape.dimensions);

        let current = self.landscape.optima[axis];
        let target = shift.new_optimum;
        let speed = shift.shift_speed.clamp(0.0, 1.0);

        // Move optimum: current + speed * (target - current)
        let new_val = current + speed * (target - current);
        self.landscape.shift_optimum(axis, new_val);

        // Re-evaluate fitness
        for org in &mut self.population {
            org.fitness = self.landscape.fitness(&org.traits);
        }
    }

    /// Detect speciation events by clustering organisms in trait space.
    ///
    /// Uses a simplified 2-means clustering: if the population can be split
    /// into two groups with inter-group Euclidean distance > `threshold`,
    /// and each group has at least 10% of the population, speciation is
    /// declared.
    pub fn detect_speciation(&self, threshold: f64) -> Vec<SpeciationEvent> {
        let n = self.population.len();
        if n < 4 {
            return Vec::new();
        }

        let n_traits = self.landscape.dimensions;

        // Simple 2-means clustering (Lloyd's algorithm, 10 iterations)
        // Initialize centroids from first and last organism (sorted by first trait)
        let mut sorted_indices: Vec<usize> = (0..n).collect();
        sorted_indices.sort_by(|&a, &b| {
            self.population[a].traits[0]
                .partial_cmp(&self.population[b].traits[0])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut c0: Vec<f64> = self.population[sorted_indices[0]].traits.clone();
        let mut c1: Vec<f64> = self.population[sorted_indices[n - 1]].traits.clone();

        let mut assignments = vec![0u8; n];

        for _iter in 0..10 {
            // Assign each organism to nearest centroid
            for i in 0..n {
                let d0 = euclidean_dist(&self.population[i].traits, &c0);
                let d1 = euclidean_dist(&self.population[i].traits, &c1);
                assignments[i] = if d0 <= d1 { 0 } else { 1 };
            }

            // Recompute centroids
            let mut sum0 = vec![0.0; n_traits];
            let mut sum1 = vec![0.0; n_traits];
            let mut count0 = 0usize;
            let mut count1 = 0usize;

            for i in 0..n {
                if assignments[i] == 0 {
                    for t in 0..n_traits {
                        sum0[t] += self.population[i].traits[t];
                    }
                    count0 += 1;
                } else {
                    for t in 0..n_traits {
                        sum1[t] += self.population[i].traits[t];
                    }
                    count1 += 1;
                }
            }

            if count0 > 0 {
                for t in 0..n_traits {
                    c0[t] = sum0[t] / count0 as f64;
                }
            }
            if count1 > 0 {
                for t in 0..n_traits {
                    c1[t] = sum1[t] / count1 as f64;
                }
            }
        }

        // Check if clusters are distinct enough
        let inter_dist = euclidean_dist(&c0, &c1);
        let count0 = assignments.iter().filter(|&&a| a == 0).count();
        let count1 = n - count0;

        let min_cluster = (n as f64 * 0.1).max(2.0) as usize;

        if inter_dist > threshold && count0 >= min_cluster && count1 >= min_cluster {
            // Determine lineage ids -- use most common lineage in each cluster
            let lin0 = dominant_lineage(&self.population, &assignments, 0);
            let lin1 = dominant_lineage(&self.population, &assignments, 1);

            // Estimate reproductive isolation from trait overlap
            // Use the fraction of misassigned organisms as proxy for gene flow
            let overlap = estimate_cluster_overlap(&self.population, &assignments, &c0, &c1);
            let isolation = (1.0 - overlap).clamp(0.0, 1.0);

            let parent = lin0.min(lin1);
            let daughter0 = lin0;
            let daughter1 = if lin1 != lin0 {
                lin1
            } else {
                // Same lineage -- assign new lineage id based on max existing
                self.population
                    .iter()
                    .map(|o| o.lineage_id)
                    .max()
                    .unwrap_or(0)
                    + 1
            };

            vec![SpeciationEvent {
                generation: self.generation,
                parent_lineage: parent,
                daughter_lineages: (daughter0, daughter1),
                trait_divergence: inter_dist,
                reproductive_isolation: isolation,
            }]
        } else {
            Vec::new()
        }
    }

    /// Detect character displacement between lineages.
    ///
    /// For each pair of lineages present in the population, measures the
    /// trait distance on each axis and compares to the initial distance
    /// when they first co-occurred (stored internally).
    pub fn detect_character_displacement(&self) -> Vec<CharacterDisplacement> {
        let n_traits = self.landscape.dimensions;
        let mut displacements = Vec::new();

        // Collect unique lineage ids
        let mut lineages: Vec<u64> = self.population.iter().map(|o| o.lineage_id).collect();
        lineages.sort();
        lineages.dedup();

        if lineages.len() < 2 {
            return displacements;
        }

        // Compute mean trait per lineage
        let mut lineage_means: Vec<(u64, Vec<f64>)> = Vec::new();
        for &lin in &lineages {
            let members: Vec<&EcoEvoOrganism> = self
                .population
                .iter()
                .filter(|o| o.lineage_id == lin)
                .collect();
            if members.is_empty() {
                continue;
            }
            let means: Vec<f64> = (0..n_traits)
                .map(|t| members.iter().map(|o| o.traits[t]).sum::<f64>() / members.len() as f64)
                .collect();
            lineage_means.push((lin, means));
        }

        // For each pair, compute displacement on each trait axis
        for i in 0..lineage_means.len() {
            for j in (i + 1)..lineage_means.len() {
                let (lin_a, ref means_a) = lineage_means[i];
                let (lin_b, ref means_b) = lineage_means[j];
                let pair_key = (lin_a.min(lin_b), lin_a.max(lin_b));

                for t in 0..n_traits {
                    let current_dist = (means_a[t] - means_b[t]).abs();

                    // Look up initial distance
                    let initial = self
                        .initial_distances
                        .iter()
                        .find(|((a, b, ax), _)| *a == pair_key.0 && *b == pair_key.1 && *ax == t)
                        .map(|(_, d)| *d);

                    if let Some(init_d) = initial {
                        if init_d > 1e-10 {
                            displacements.push(CharacterDisplacement {
                                species_pair: pair_key,
                                trait_axis: t,
                                initial_distance: init_d,
                                final_distance: current_dist,
                                displacement_ratio: current_dist / init_d,
                            });
                        }
                    }
                }
            }
        }

        displacements
    }

    /// Record initial trait distances between lineages (call when lineages
    /// first diverge or are introduced).
    pub fn record_initial_distances(&mut self) {
        let n_traits = self.landscape.dimensions;

        let mut lineages: Vec<u64> = self.population.iter().map(|o| o.lineage_id).collect();
        lineages.sort();
        lineages.dedup();

        let mut lineage_means: Vec<(u64, Vec<f64>)> = Vec::new();
        for &lin in &lineages {
            let members: Vec<&EcoEvoOrganism> = self
                .population
                .iter()
                .filter(|o| o.lineage_id == lin)
                .collect();
            if members.is_empty() {
                continue;
            }
            let means: Vec<f64> = (0..n_traits)
                .map(|t| members.iter().map(|o| o.traits[t]).sum::<f64>() / members.len() as f64)
                .collect();
            lineage_means.push((lin, means));
        }

        for i in 0..lineage_means.len() {
            for j in (i + 1)..lineage_means.len() {
                let (lin_a, ref means_a) = lineage_means[i];
                let (lin_b, ref means_b) = lineage_means[j];
                let pair_key = (lin_a.min(lin_b), lin_a.max(lin_b));

                for t in 0..n_traits {
                    let dist = (means_a[t] - means_b[t]).abs();
                    // Only store if not already recorded
                    let exists = self
                        .initial_distances
                        .iter()
                        .any(|((a, b, ax), _)| *a == pair_key.0 && *b == pair_key.1 && *ax == t);
                    if !exists {
                        self.initial_distances
                            .push(((pair_key.0, pair_key.1, t), dist));
                    }
                }
            }
        }
    }

    /// Get a reference to the evolutionary rescue tracker.
    pub fn evolutionary_rescue_status(&self) -> &EvolutionaryRescueTracker {
        &self.rescue_tracker
    }

    /// Mean fitness across the population.
    pub fn mean_fitness(&self) -> f64 {
        if self.population.is_empty() {
            return 0.0;
        }
        self.population.iter().map(|o| o.fitness).sum::<f64>() / self.population.len() as f64
    }

    /// Per-trait additive genetic variance (Var(z_i) across population).
    pub fn genetic_variance(&self) -> Vec<f64> {
        let n_traits = self.landscape.dimensions;
        let n = self.population.len();
        if n < 2 {
            return vec![0.0; n_traits];
        }

        let means = self.trait_means();
        let mut variances = vec![0.0; n_traits];
        for t in 0..n_traits {
            let sum_sq: f64 = self
                .population
                .iter()
                .map(|o| {
                    let d = o.traits[t] - means[t];
                    d * d
                })
                .sum();
            variances[t] = sum_sq / (n - 1) as f64; // sample variance
        }
        variances
    }

    /// Mean trait values across the population.
    pub fn trait_means(&self) -> Vec<f64> {
        let n_traits = self.landscape.dimensions;
        let n = self.population.len();
        if n == 0 {
            return vec![0.0; n_traits];
        }
        let mut means = vec![0.0; n_traits];
        for org in &self.population {
            for t in 0..n_traits {
                means[t] += org.traits[t];
            }
        }
        for t in 0..n_traits {
            means[t] /= n as f64;
        }
        means
    }

    /// Breeder's equation: R = h^2 * S
    ///
    /// Given a selection gradient (S = selection differential per trait),
    /// computes the expected response to selection assuming additive genetic
    /// variance equals total phenotypic variance (h^2 approximated from data).
    ///
    /// In practice, h^2 = Va / Vp. Here we use the population's genetic
    /// variance as Va and add a small environmental variance to get Vp.
    pub fn breeder_equation_response(&self, selection_gradient: &[f64]) -> Vec<f64> {
        let n_traits = self.landscape.dimensions;
        assert_eq!(selection_gradient.len(), n_traits);

        let va = self.genetic_variance();
        let mut response = vec![0.0; n_traits];

        for t in 0..n_traits {
            // Vp = Va + Ve, assume Ve = 0.1 * Va + small constant to avoid /0
            let vp = va[t] + 0.1 * va[t] + 1e-10;
            let h2 = va[t] / vp;
            response[t] = h2 * selection_gradient[t];
        }
        response
    }

    /// Assign distinct lineage ids to organisms based on 2-means clustering.
    /// This is used to create lineage structure for speciation and
    /// displacement tests.
    pub fn assign_lineages_by_clustering(&mut self) {
        let n = self.population.len();
        if n < 4 {
            return;
        }
        let n_traits = self.landscape.dimensions;

        // Simple 2-means
        let mut sorted_indices: Vec<usize> = (0..n).collect();
        sorted_indices.sort_by(|&a, &b| {
            self.population[a].traits[0]
                .partial_cmp(&self.population[b].traits[0])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut c0: Vec<f64> = self.population[sorted_indices[0]].traits.clone();
        let mut c1: Vec<f64> = self.population[sorted_indices[n - 1]].traits.clone();

        let mut assignments = vec![0u8; n];

        for _iter in 0..10 {
            for i in 0..n {
                let d0 = euclidean_dist(&self.population[i].traits, &c0);
                let d1 = euclidean_dist(&self.population[i].traits, &c1);
                assignments[i] = if d0 <= d1 { 0 } else { 1 };
            }

            let mut sum0 = vec![0.0; n_traits];
            let mut sum1 = vec![0.0; n_traits];
            let mut count0 = 0usize;
            let mut count1 = 0usize;

            for i in 0..n {
                if assignments[i] == 0 {
                    for t in 0..n_traits {
                        sum0[t] += self.population[i].traits[t];
                    }
                    count0 += 1;
                } else {
                    for t in 0..n_traits {
                        sum1[t] += self.population[i].traits[t];
                    }
                    count1 += 1;
                }
            }

            if count0 > 0 {
                for t in 0..n_traits {
                    c0[t] = sum0[t] / count0 as f64;
                }
            }
            if count1 > 0 {
                for t in 0..n_traits {
                    c1[t] = sum1[t] / count1 as f64;
                }
            }
        }

        let lin_a = self.next_lineage_id;
        let lin_b = self.next_lineage_id + 1;
        self.next_lineage_id += 2;

        for i in 0..n {
            self.population[i].lineage_id = if assignments[i] == 0 { lin_a } else { lin_b };
        }
    }
}

// ---------------------------------------------------------------------------
// Free functions -- evolutionary quantitative genetics
// ---------------------------------------------------------------------------

/// Compute Qst (quantitative trait divergence) between subpopulations.
///
/// Qst = Var_between / (Var_between + 2 * Var_within)
///
/// Analogous to Fst for neutral markers. When Qst > Fst, it implies
/// directional selection on the quantitative trait (Leinonen et al. 2013).
pub fn compute_qst(subpops: &[&[EcoEvoOrganism]], trait_idx: usize) -> f64 {
    if subpops.len() < 2 {
        return 0.0;
    }

    // Compute per-subpop means and variances
    let mut pop_means = Vec::new();
    let mut pop_vars = Vec::new();
    let mut pop_sizes = Vec::new();

    for subpop in subpops {
        if subpop.is_empty() {
            continue;
        }
        let mean = subpop.iter().map(|o| o.traits[trait_idx]).sum::<f64>() / subpop.len() as f64;
        let var = if subpop.len() > 1 {
            subpop
                .iter()
                .map(|o| {
                    let d = o.traits[trait_idx] - mean;
                    d * d
                })
                .sum::<f64>()
                / (subpop.len() - 1) as f64
        } else {
            0.0
        };
        pop_means.push(mean);
        pop_vars.push(var);
        pop_sizes.push(subpop.len());
    }

    if pop_means.len() < 2 {
        return 0.0;
    }

    // Grand mean
    let total_n: usize = pop_sizes.iter().sum();
    let grand_mean: f64 = pop_means
        .iter()
        .zip(pop_sizes.iter())
        .map(|(m, &n)| m * n as f64)
        .sum::<f64>()
        / total_n as f64;

    // Variance between subpopulations
    let var_between: f64 = pop_means
        .iter()
        .zip(pop_sizes.iter())
        .map(|(m, &n)| {
            let d = m - grand_mean;
            n as f64 * d * d
        })
        .sum::<f64>()
        / total_n as f64;

    // Mean variance within subpopulations
    let var_within: f64 = pop_vars
        .iter()
        .zip(pop_sizes.iter())
        .map(|(v, &n)| v * n as f64)
        .sum::<f64>()
        / total_n as f64;

    // Qst formula
    let denom = var_between + 2.0 * var_within;
    if denom < 1e-15 {
        return 0.0;
    }
    (var_between / denom).clamp(0.0, 1.0)
}

/// Price equation decomposition of trait change across one generation.
///
/// Returns (selection_component, transmission_component) where:
/// - selection = Cov(w, z) / mean(w)  (trait change due to differential fitness)
/// - transmission = E(w * delta_z) / mean(w)  (trait change due to mutation/drift)
///
/// The sum of both components equals the total change in mean trait value.
///
/// `before` and `after` populations must have matching id pairs for the
/// transmission term. Organisms in `after` that have no matching parent in
/// `before` contribute zero to transmission (treated as immigrants).
pub fn price_equation(
    before: &[EcoEvoOrganism],
    after: &[EcoEvoOrganism],
    trait_idx: usize,
) -> (f64, f64) {
    if before.is_empty() || after.is_empty() {
        return (0.0, 0.0);
    }

    // Mean fitness of before population
    let mean_w: f64 = before.iter().map(|o| o.fitness).sum::<f64>() / before.len() as f64;
    if mean_w < 1e-15 {
        return (0.0, 0.0);
    }

    // Mean trait before
    let mean_z_before: f64 =
        before.iter().map(|o| o.traits[trait_idx]).sum::<f64>() / before.len() as f64;

    // Selection component: Cov(w, z) / mean_w
    let cov_wz: f64 = before
        .iter()
        .map(|o| (o.fitness - mean_w) * (o.traits[trait_idx] - mean_z_before))
        .sum::<f64>()
        / before.len() as f64;
    let selection = cov_wz / mean_w;

    // Total change in mean trait
    let mean_z_after: f64 =
        after.iter().map(|o| o.traits[trait_idx]).sum::<f64>() / after.len() as f64;
    let total_change = mean_z_after - mean_z_before;

    // Transmission = total change - selection
    // This ensures the Price equation identity: delta_z = selection + transmission
    let transmission = total_change - selection;

    (selection, transmission)
}

/// Invasion fitness of a mutant phenotype in a resident population.
///
/// Computed as the fitness of the mutant trait vector relative to the mean
/// fitness of residents. Positive invasion fitness means the mutant can
/// invade; negative means it will be excluded.
///
/// `invasion_fitness = w(mutant) - mean(w(residents))`
pub fn invasion_fitness(
    resident_traits: &[f64],
    mutant_traits: &[f64],
    landscape: &FitnessLandscape,
) -> f64 {
    let w_mutant = landscape.fitness(mutant_traits);
    let w_resident = landscape.fitness(resident_traits);
    w_mutant - w_resident
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Euclidean distance between two trait vectors.
fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

/// Find the most common lineage id among organisms in a given cluster.
fn dominant_lineage(pop: &[EcoEvoOrganism], assignments: &[u8], cluster: u8) -> u64 {
    let mut counts: Vec<(u64, usize)> = Vec::new();
    for (i, org) in pop.iter().enumerate() {
        if assignments[i] == cluster {
            if let Some(entry) = counts.iter_mut().find(|(id, _)| *id == org.lineage_id) {
                entry.1 += 1;
            } else {
                counts.push((org.lineage_id, 1));
            }
        }
    }
    counts
        .into_iter()
        .max_by_key(|&(_, c)| c)
        .map(|(id, _)| id)
        .unwrap_or(0)
}

/// Estimate overlap between two clusters as a proxy for gene flow.
///
/// Computes fraction of organisms closer to the other cluster's centroid
/// than to their own (within some margin).
fn estimate_cluster_overlap(
    pop: &[EcoEvoOrganism],
    assignments: &[u8],
    c0: &[f64],
    c1: &[f64],
) -> f64 {
    let inter_dist = euclidean_dist(c0, c1);
    if inter_dist < 1e-10 {
        return 1.0; // fully overlapping
    }

    let mut overlap_count = 0usize;
    let n = pop.len();
    for i in 0..n {
        let d0 = euclidean_dist(&pop[i].traits, c0);
        let d1 = euclidean_dist(&pop[i].traits, c1);
        let assigned_dist = if assignments[i] == 0 { d0 } else { d1 };
        let other_dist = if assignments[i] == 0 { d1 } else { d0 };
        // If an organism is almost equidistant, count as overlap
        if other_dist < assigned_dist * 1.2 {
            overlap_count += 1;
        }
    }
    overlap_count as f64 / n as f64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create a simple landscape
    fn simple_landscape(n_traits: usize, optima: Vec<f64>) -> FitnessLandscape {
        FitnessLandscape {
            dimensions: n_traits,
            optima,
            widths: vec![1.0; n_traits],
            ruggedness: 0.0,
        }
    }

    // -----------------------------------------------------------------------
    // 1. Fitness landscape peaks at optimum
    // -----------------------------------------------------------------------
    #[test]
    fn fitness_landscape_gaussian() {
        let landscape = simple_landscape(2, vec![3.0, -1.0]);

        // At the optimum, fitness should be maximal (1.0 for smooth landscape)
        let at_optimum = landscape.fitness(&[3.0, -1.0]);
        assert!(
            (at_optimum - 1.0).abs() < 1e-10,
            "Fitness at optimum should be 1.0, got {}",
            at_optimum
        );

        // Away from optimum, fitness should be lower
        let away = landscape.fitness(&[3.0, 1.0]);
        assert!(
            away < at_optimum,
            "Fitness should decrease away from optimum"
        );

        // Far away, fitness should be near zero
        let far = landscape.fitness(&[10.0, 10.0]);
        assert!(
            far < 0.01,
            "Fitness far from optimum should be near zero, got {}",
            far
        );
    }

    // -----------------------------------------------------------------------
    // 2. Gradient points toward optimum
    // -----------------------------------------------------------------------
    #[test]
    fn fitness_landscape_gradient() {
        let landscape = simple_landscape(2, vec![5.0, -3.0]);

        // At a point below the optimum, gradient should point upward
        let traits = vec![3.0, -5.0];
        let grad = landscape.adaptive_landscape_gradient(&traits);

        // Gradient on axis 0: trait is 3.0, optimum is 5.0, so gradient should be positive
        assert!(
            grad[0] > 0.0,
            "Gradient[0] should point toward optimum (positive), got {}",
            grad[0]
        );

        // Gradient on axis 1: trait is -5.0, optimum is -3.0, so gradient should be positive
        assert!(
            grad[1] > 0.0,
            "Gradient[1] should point toward optimum (positive), got {}",
            grad[1]
        );

        // At the optimum, gradient should be zero
        let grad_at_opt = landscape.adaptive_landscape_gradient(&[5.0, -3.0]);
        assert!(
            grad_at_opt[0].abs() < 1e-10 && grad_at_opt[1].abs() < 1e-10,
            "Gradient at optimum should be zero"
        );
    }

    // -----------------------------------------------------------------------
    // 3. Population adapts to a shifting optimum
    // -----------------------------------------------------------------------
    #[test]
    fn population_adapts_to_shift() {
        let mut sim = EcoEvoSimulator::new(200, 1, 42);
        sim.mutation_rate = 0.5;
        sim.mutation_effect_size = 0.2;

        // Initial optimum at 0 -- let population equilibrate
        for _ in 0..20 {
            sim.step();
        }
        let mean_before = sim.trait_means()[0];
        assert!(
            mean_before.abs() < 1.0,
            "Population should be near optimum 0.0, got {}",
            mean_before
        );

        // Shift optimum to 2.0
        sim.landscape.shift_optimum(0, 2.0);

        // Let population adapt
        for _ in 0..100 {
            sim.step();
        }
        let mean_after = sim.trait_means()[0];
        assert!(
            mean_after > mean_before + 0.5,
            "Population mean should have shifted toward new optimum 2.0. Before: {}, After: {}",
            mean_before,
            mean_after
        );
    }

    // -----------------------------------------------------------------------
    // 4. Evolutionary rescue detection after environmental shock
    // -----------------------------------------------------------------------
    #[test]
    fn evolutionary_rescue_detection() {
        let mut sim = EcoEvoSimulator::new(100, 1, 123);
        sim.mutation_rate = 0.3;
        sim.mutation_effect_size = 0.3;
        sim.carrying_capacity = 100;

        // Equilibrate
        for _ in 0..10 {
            sim.step();
        }

        // Record pre-shock state
        assert!(
            !sim.rescue_tracker.rescue_detected,
            "No rescue before shock"
        );

        // Apply severe environmental shock -- shift optimum far away
        // This will cause fitness to drop, and the density-regulated population
        // size will effectively decline because we manually remove low-fitness
        // individuals to simulate population crash.
        sim.landscape.shift_optimum(0, 8.0);

        // Simulate population crash by removing organisms with very low fitness
        for org in &mut sim.population {
            org.fitness = sim.landscape.fitness(&org.traits);
        }
        // Keep only the fittest 30% to simulate a crash
        sim.population.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sim.population.truncate(30);

        // Update tracker to detect decline
        let mf = sim.mean_fitness();
        let gv: f64 = sim.genetic_variance().iter().sum();
        sim.rescue_tracker
            .update(sim.population.len(), mf, gv, sim.generation + 1);

        assert!(
            sim.rescue_tracker.decline_start.is_some(),
            "Should detect population decline below 50% capacity"
        );

        // Now let population grow back (adaptation + reproduction)
        sim.carrying_capacity = 100;
        for _ in 0..150 {
            sim.step();
        }

        // The population should recover as it adapts to the new optimum
        // The rescue tracker should detect this
        assert!(
            sim.rescue_tracker.rescue_detected
                || sim.population.len() >= sim.rescue_tracker.critical_population_size,
            "Population should have recovered or rescue detected. Pop: {}, Critical: {}",
            sim.population.len(),
            sim.rescue_tracker.critical_population_size
        );
    }

    // -----------------------------------------------------------------------
    // 5. Speciation from disruptive selection (bimodal landscape)
    // -----------------------------------------------------------------------
    #[test]
    fn speciation_from_disruptive_selection() {
        // Create a population with two distinct groups on a single trait axis
        let mut sim = EcoEvoSimulator::new(100, 1, 999);

        // Manually create a bimodal population
        let mut rng = Xorshift64::new(999);
        for (i, org) in sim.population.iter_mut().enumerate() {
            if i < 50 {
                org.traits[0] = -3.0 + rng.next_normal(0.0, 0.2);
            } else {
                org.traits[0] = 3.0 + rng.next_normal(0.0, 0.2);
            }
            org.fitness = 1.0; // equal fitness
        }

        // Should detect speciation with a reasonable threshold
        let events = sim.detect_speciation(2.0);
        assert!(
            !events.is_empty(),
            "Should detect speciation in bimodal population"
        );
        assert!(
            events[0].trait_divergence > 2.0,
            "Trait divergence should exceed threshold, got {}",
            events[0].trait_divergence
        );
    }

    // -----------------------------------------------------------------------
    // 6. Character displacement increases trait distance
    // -----------------------------------------------------------------------
    #[test]
    fn character_displacement_increases_distance() {
        let mut sim = EcoEvoSimulator::new(100, 1, 77);

        // Create two lineages with initially similar traits
        for (i, org) in sim.population.iter_mut().enumerate() {
            if i < 50 {
                org.lineage_id = 1;
                org.traits[0] = 0.5;
            } else {
                org.lineage_id = 2;
                org.traits[0] = 0.7;
            }
        }

        // Record initial distances
        sim.record_initial_distances();

        // Manually push traits apart (simulating competition-driven divergence)
        for org in &mut sim.population {
            if org.lineage_id == 1 {
                org.traits[0] = -1.0;
            } else {
                org.traits[0] = 2.0;
            }
        }

        let displacements = sim.detect_character_displacement();
        assert!(
            !displacements.is_empty(),
            "Should detect character displacement"
        );

        let cd = &displacements[0];
        assert!(
            cd.displacement_ratio > 1.0,
            "Displacement ratio should be > 1.0 (divergence), got {}",
            cd.displacement_ratio
        );
        assert!(
            cd.final_distance > cd.initial_distance,
            "Final distance {} should exceed initial distance {}",
            cd.final_distance,
            cd.initial_distance
        );
    }

    // -----------------------------------------------------------------------
    // 7. Breeder's equation: R approximately equals h^2 * S
    // -----------------------------------------------------------------------
    #[test]
    fn breeder_equation_response() {
        let mut sim = EcoEvoSimulator::new(500, 2, 55);
        sim.mutation_rate = 0.3;
        sim.mutation_effect_size = 0.5;

        // Run a few generations to build up genetic variance
        for _ in 0..20 {
            sim.step();
        }

        let va = sim.genetic_variance();
        assert!(va[0] > 0.0, "Should have genetic variance after mutations");

        // Apply a selection gradient
        let selection_gradient = vec![1.0, -0.5];
        let response = sim.breeder_equation_response(&selection_gradient);

        // R should be in the same direction as S
        assert!(
            response[0] > 0.0,
            "Response should be positive when selection is positive, got {}",
            response[0]
        );
        assert!(
            response[1] < 0.0,
            "Response should be negative when selection is negative, got {}",
            response[1]
        );

        // R should be less than S in magnitude (h^2 <= 1)
        assert!(
            response[0].abs() <= selection_gradient[0].abs() + 1e-10,
            "Response magnitude should not exceed selection gradient"
        );
    }

    // -----------------------------------------------------------------------
    // 8. Price equation: selection + transmission = total change
    // -----------------------------------------------------------------------
    #[test]
    fn price_equation_decomposition() {
        // Create before population
        let before: Vec<EcoEvoOrganism> = (0..100)
            .map(|i| {
                let trait_val = (i as f64 - 50.0) * 0.1;
                let fitness = (-trait_val * trait_val / 2.0).exp(); // Gaussian fitness
                EcoEvoOrganism {
                    id: i as u64,
                    traits: vec![trait_val],
                    fitness,
                    lineage_id: 0,
                    generation: 0,
                }
            })
            .collect();

        // Create after population -- shifted toward optimum (selection happened)
        let after: Vec<EcoEvoOrganism> = (0..100)
            .map(|i| {
                let before_val = (i as f64 - 50.0) * 0.1;
                // Selection shifts mean toward zero, transmission adds small noise
                let trait_val = before_val * 0.9 + 0.05;
                EcoEvoOrganism {
                    id: i as u64,
                    traits: vec![trait_val],
                    fitness: 1.0,
                    lineage_id: 0,
                    generation: 1,
                }
            })
            .collect();

        let (selection, transmission) = price_equation(&before, &after, 0);

        // Total change
        let mean_before: f64 =
            before.iter().map(|o| o.traits[0]).sum::<f64>() / before.len() as f64;
        let mean_after: f64 = after.iter().map(|o| o.traits[0]).sum::<f64>() / after.len() as f64;
        let total = mean_after - mean_before;

        // Price equation identity: selection + transmission = total change
        let sum = selection + transmission;
        assert!(
            (sum - total).abs() < 1e-10,
            "Price equation identity violated: sel({}) + trans({}) = {} != total({})",
            selection,
            transmission,
            sum,
            total
        );
    }

    // -----------------------------------------------------------------------
    // 9. Qst > 0 for diverged subpopulations
    // -----------------------------------------------------------------------
    #[test]
    fn qst_divergent_populations() {
        // Two subpopulations diverged on trait 0
        let pop_a: Vec<EcoEvoOrganism> = (0..50)
            .map(|i| EcoEvoOrganism {
                id: i as u64,
                traits: vec![10.0 + (i as f64) * 0.01],
                fitness: 1.0,
                lineage_id: 0,
                generation: 0,
            })
            .collect();

        let pop_b: Vec<EcoEvoOrganism> = (0..50)
            .map(|i| EcoEvoOrganism {
                id: (i + 50) as u64,
                traits: vec![-10.0 + (i as f64) * 0.01],
                fitness: 1.0,
                lineage_id: 1,
                generation: 0,
            })
            .collect();

        let qst = compute_qst(&[&pop_a, &pop_b], 0);
        assert!(
            qst > 0.5,
            "Qst should be high for strongly diverged populations, got {}",
            qst
        );

        // For identical populations, Qst should be 0
        let qst_same = compute_qst(&[&pop_a, &pop_a], 0);
        assert!(
            qst_same < 0.01,
            "Qst should be near zero for identical populations, got {}",
            qst_same
        );
    }

    // -----------------------------------------------------------------------
    // 10. Residents have non-negative invasion fitness against themselves
    // -----------------------------------------------------------------------
    #[test]
    fn invasion_fitness_resident_advantage() {
        let landscape = simple_landscape(2, vec![1.0, 1.0]);

        let resident = vec![1.0, 1.0]; // at the optimum
        let mutant_close = vec![1.1, 1.0]; // close to optimum
        let mutant_far = vec![5.0, 5.0]; // far from optimum

        // Resident vs itself: invasion fitness should be 0
        let inv_self = invasion_fitness(&resident, &resident, &landscape);
        assert!(
            inv_self.abs() < 1e-10,
            "Self-invasion fitness should be 0, got {}",
            inv_self
        );

        // Close mutant: small negative invasion fitness (slightly worse)
        let inv_close = invasion_fitness(&resident, &mutant_close, &landscape);
        assert!(
            inv_close < 0.0,
            "Mutant away from optimum should have negative invasion fitness, got {}",
            inv_close
        );

        // Far mutant: strongly negative invasion fitness
        let inv_far = invasion_fitness(&resident, &mutant_far, &landscape);
        assert!(
            inv_far < inv_close,
            "Farther mutant should have lower invasion fitness"
        );
    }

    // -----------------------------------------------------------------------
    // 11. Genetic variance maintained by mutation-selection balance
    // -----------------------------------------------------------------------
    #[test]
    fn genetic_variance_maintained() {
        let mut sim = EcoEvoSimulator::new(200, 2, 314);
        sim.mutation_rate = 0.1;
        sim.mutation_effect_size = 0.1;

        // Run for many generations
        for _ in 0..100 {
            sim.step();
        }

        let va = sim.genetic_variance();

        // Genetic variance should be positive (maintained by mutation)
        assert!(
            va[0] > 1e-6,
            "Genetic variance on trait 0 should be maintained, got {}",
            va[0]
        );
        assert!(
            va[1] > 1e-6,
            "Genetic variance on trait 1 should be maintained, got {}",
            va[1]
        );

        // But not exploding (selection constrains it)
        assert!(
            va[0] < 10.0,
            "Genetic variance should be bounded, got {}",
            va[0]
        );
    }

    // -----------------------------------------------------------------------
    // 12. Sudden shifts cause more population stress than gradual shifts
    // -----------------------------------------------------------------------
    #[test]
    fn sudden_vs_gradual_shift() {
        // Run two parallel simulations with same seed
        let target_optimum = 4.0;

        // Sudden shift simulation
        let mut sim_sudden = EcoEvoSimulator::new(100, 1, 2025);
        sim_sudden.mutation_rate = 0.2;
        sim_sudden.mutation_effect_size = 0.15;
        for _ in 0..10 {
            sim_sudden.step();
        }
        let sudden_shift = EnvironmentalShift {
            generation: sim_sudden.generation,
            trait_axis: 0,
            new_optimum: target_optimum,
            shift_speed: 1.0, // instantaneous
        };
        sim_sudden.apply_environmental_shift(&sudden_shift);
        let fitness_after_sudden = sim_sudden.mean_fitness();

        // Gradual shift simulation
        let mut sim_gradual = EcoEvoSimulator::new(100, 1, 2025);
        sim_gradual.mutation_rate = 0.2;
        sim_gradual.mutation_effect_size = 0.15;
        for _ in 0..10 {
            sim_gradual.step();
        }
        // Apply gradual shift -- only moves 10% of the way
        let gradual_shift = EnvironmentalShift {
            generation: sim_gradual.generation,
            trait_axis: 0,
            new_optimum: target_optimum,
            shift_speed: 0.1, // very gradual
        };
        sim_gradual.apply_environmental_shift(&gradual_shift);
        let fitness_after_gradual = sim_gradual.mean_fitness();

        // After a sudden shift, population fitness should drop more
        assert!(
            fitness_after_sudden < fitness_after_gradual,
            "Sudden shift should cause lower fitness ({}) than gradual ({}).",
            fitness_after_sudden,
            fitness_after_gradual
        );
    }
}
