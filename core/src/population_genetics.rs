//! Classical population genetics and evolutionary statistics.
//!
//! Provides Wright-Fisher simulation, Moran process, coalescent trees,
//! F-statistics, Tajima's D, Hardy-Weinberg tests, linkage disequilibrium,
//! effective population size estimation, and related tools for analyzing
//! evolving populations in the terrarium.
//!
//! All computations are self-contained with no external crate dependencies.
//! Random number generation uses an inline xorshift64 PRNG with binomial
//! sampling derived from the inverse-transform method.
//!
//! # Literature
//!
//! - Wright 1931, *Evolution in Mendelian Populations*, Genetics 16:97.
//! - Fisher 1930, *The Genetical Theory of Natural Selection*, Clarendon.
//! - Kimura 1962, *On the probability of fixation of mutant genes*, Genetics 47:713.
//! - Moran 1958, *Random processes in genetics*, Proc. Camb. Phil. Soc. 54:60.
//! - Kingman 1982, *The coalescent*, Stoch. Proc. Appl. 13:235.
//! - Weir & Cockerham 1984, *Estimating F-statistics*, Evolution 38:1358.
//! - Tajima 1989, *Statistical method for testing the neutral mutation hypothesis*,
//!   Genetics 123:585.
//! - Hardy 1908 / Weinberg 1908, *Mendelian proportions in a mixed population*.
//! - Lewontin 1964, *The interaction of selection and linkage*, Genetics 49:49.
//! - Nei & Tajima 1981, *Genetic drift and estimation of effective population size*,
//!   Genetics 98:625.
//! - Watterson 1975, *On the number of segregating sites*, Theor. Pop. Biol. 7:256.
//! - McDonald & Kreitman 1991, *Adaptive protein evolution at the Adh locus in
//!   Drosophila*, Nature 351:652.

// ─────────────────────────────────────────────────────────────────────────────
// Inline xorshift64 PRNG
// ─────────────────────────────────────────────────────────────────────────────

/// Advance an xorshift64 state and return the new value.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    // Ensure state is never zero (xorshift64 fixed point).
    if *state == 0 {
        *state = 0xDEAD_BEEF_CAFE_BABEu64;
    }
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Return a uniform f64 in [0, 1).
#[inline]
fn rand_f64(state: &mut u64) -> f64 {
    (xorshift64(state) >> 11) as f64 / ((1u64 << 53) as f64)
}

/// Binomial sample: number of successes in `n` trials with probability `p`.
///
/// Uses the inverse-transform method for small n and a normal approximation
/// with continuity correction for large n (threshold: n*p*(1-p) > 20).
fn binomial(state: &mut u64, n: usize, p: f64) -> usize {
    if n == 0 || p <= 0.0 {
        return 0;
    }
    if p >= 1.0 {
        return n;
    }
    let np = n as f64 * p;
    let variance = np * (1.0 - p);

    // For large n with sufficient variance, use normal approximation.
    if variance > 20.0 {
        let std_dev = variance.sqrt();
        // Box-Muller transform for a standard normal variate.
        let u1 = rand_f64(state).max(1e-15);
        let u2 = rand_f64(state);
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let x = (np + z * std_dev + 0.5).round() as i64;
        return x.max(0).min(n as i64) as usize;
    }

    // Direct simulation for small n.
    let mut successes = 0usize;
    for _ in 0..n {
        if rand_f64(state) < p {
            successes += 1;
        }
    }
    successes
}

/// Weighted random selection: returns index chosen proportional to weights.
fn weighted_choice(state: &mut u64, weights: &[f64]) -> usize {
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return 0;
    }
    let r = rand_f64(state) * total;
    let mut cumulative = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        cumulative += w;
        if r < cumulative {
            return i;
        }
    }
    weights.len() - 1
}

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// A diploid locus with allele frequencies.
#[derive(Debug, Clone)]
pub struct Locus {
    pub name: String,
    /// Frequency of each allele (must sum to 1).
    pub allele_frequencies: Vec<f64>,
    /// Position in centimorgans (for linkage disequilibrium calculations).
    pub position_cm: f64,
}

/// A subpopulation with allele frequency data.
#[derive(Debug, Clone)]
pub struct Subpopulation {
    pub name: String,
    /// Effective population size (Ne).
    pub size: usize,
    pub loci: Vec<Locus>,
    /// Per-individual fitness values.
    pub fitness_values: Vec<f64>,
}

/// Wright-Fisher population model for a single biallelic locus.
///
/// Simulates genetic drift with optional selection and mutation in a
/// diploid population of constant size.  Each generation, 2N gametes
/// are sampled from the current allele frequency, weighted by
/// genotype fitness when selection is active.
pub struct WrightFisherSim {
    /// Frequency of the focal allele (A).
    pub allele_freq: f64,
    /// Diploid population size (2N chromosomes sampled each generation).
    pub pop_size: usize,
    pub generation: u32,
    /// Selection coefficient (fitness of AA = 1+s, Aa = 1+hs, aa = 1).
    pub selection_coefficient: f64,
    /// Dominance coefficient (0 = recessive, 0.5 = additive, 1 = dominant).
    pub dominance: f64,
    /// Forward mutation rate (A -> a).
    pub mutation_rate_forward: f64,
    /// Backward mutation rate (a -> A).
    pub mutation_rate_backward: f64,
    rng_state: u64,
}

/// Moran process (continuous-time analog of Wright-Fisher).
///
/// At each step one individual is chosen to reproduce (proportional to
/// fitness) and one random individual dies, keeping population size constant.
pub struct MoranProcess {
    /// Count of each type in the population.
    pub counts: Vec<usize>,
    /// Relative fitness of each type.
    pub fitness: Vec<f64>,
    pub generation: u32,
    rng_state: u64,
}

/// A node in a coalescent genealogy.
#[derive(Debug, Clone)]
pub struct CoalescentNode {
    pub id: u64,
    /// Coalescent time (looking backward in generations).
    pub time: f64,
    /// IDs of child nodes (empty for leaf/sample nodes).
    pub children: Vec<u64>,
    /// Number of mutations placed on the branch leading to this node.
    pub mutations: u32,
}

/// Wright's F-statistics (Weir & Cockerham 1984 estimator).
#[derive(Debug, Clone, Copy)]
pub struct FStatistics {
    /// Fixation index: proportion of total heterozygosity due to
    /// between-subpopulation differentiation.
    pub fst: f64,
    /// Inbreeding coefficient within subpopulations.
    pub fis: f64,
    /// Overall inbreeding coefficient.
    pub fit: f64,
}

/// Result of Tajima's D test for departures from selective neutrality.
#[derive(Debug, Clone)]
pub struct TajimaD {
    pub d_statistic: f64,
    /// Nucleotide diversity (average pairwise differences).
    pub theta_pi: f64,
    /// Watterson's estimator from segregating sites.
    pub theta_w: f64,
    pub interpretation: SelectionSignal,
}

/// Interpretation of Tajima's D.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SelectionSignal {
    /// D approximately 0: consistent with neutral evolution.
    Neutral,
    /// D << 0: excess of rare alleles, consistent with positive (directional) selection.
    PositiveSelection,
    /// D >> 0: excess of intermediate-frequency alleles, consistent with balancing selection.
    BalancingSelection,
    /// D < 0: also consistent with a recent population bottleneck.
    RecentBottleneck,
    /// D < 0: also consistent with population expansion.
    PopulationExpansion,
}

/// Hardy-Weinberg equilibrium test result.
#[derive(Debug, Clone, Copy)]
pub struct HardyWeinbergTest {
    /// Observed heterozygosity (fraction of heterozygotes).
    pub observed_het: f64,
    /// Expected heterozygosity under HWE (2pq).
    pub expected_het: f64,
    /// Chi-squared goodness-of-fit statistic (1 d.f.).
    pub chi_squared: f64,
    /// Whether the population is in equilibrium at alpha = 0.05 (chi-sq < 3.841).
    pub in_equilibrium: bool,
    /// True if observed heterozygosity exceeds expectation (possible balancing selection).
    pub excess_het: bool,
}

/// Linkage disequilibrium between two biallelic loci.
#[derive(Debug, Clone, Copy)]
pub struct LDResult {
    /// Lewontin's D' (normalized LD coefficient).
    pub d_prime: f64,
    /// Squared correlation coefficient (r^2).
    pub r_squared: f64,
    /// Pair of locus indices being compared.
    pub locus_pair: (usize, usize),
}

/// Effective population size estimation from multiple methods.
#[derive(Debug, Clone, Copy)]
pub struct NeEstimate {
    /// Ne from the temporal method (Nei & Tajima 1981).
    pub ne_temporal: f64,
    /// Ne from linkage disequilibrium.
    pub ne_ld: f64,
    /// Ne from heterozygosity decay.
    pub ne_heterozygosity: f64,
    /// Census population count.
    pub census_n: usize,
    /// Ne/Nc ratio (typically 0.1 - 0.5).
    pub ne_nc_ratio: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Wright-Fisher implementation
// ─────────────────────────────────────────────────────────────────────────────

impl WrightFisherSim {
    /// Create a new Wright-Fisher simulation.
    ///
    /// `pop_size` is the diploid population size (number of individuals;
    /// 2N chromosomes are sampled each generation).
    pub fn new(pop_size: usize, initial_freq: f64, seed: u64) -> Self {
        Self {
            allele_freq: initial_freq.clamp(0.0, 1.0),
            pop_size: pop_size.max(1),
            generation: 0,
            selection_coefficient: 0.0,
            dominance: 0.5,
            mutation_rate_forward: 0.0,
            mutation_rate_backward: 0.0,
            rng_state: if seed == 0 {
                0xCAFE_BABE_1234_5678
            } else {
                seed
            },
        }
    }

    /// Set selection parameters.
    pub fn set_selection(&mut self, s: f64, h: f64) {
        self.selection_coefficient = s;
        self.dominance = h;
    }

    /// Set mutation rates (forward: A->a, backward: a->A).
    pub fn set_mutation(&mut self, forward: f64, backward: f64) {
        self.mutation_rate_forward = forward.max(0.0);
        self.mutation_rate_backward = backward.max(0.0);
    }

    /// Advance one generation and return the new allele frequency.
    ///
    /// 1. Apply mutation to the current frequency.
    /// 2. Compute the effective frequency after selection.
    /// 3. Binomial sample 2N gametes at the effective frequency.
    pub fn step(&mut self) -> f64 {
        let p = self.allele_freq;

        // Mutation: p' = p(1-u) + (1-p)v
        let p_after_mut =
            p * (1.0 - self.mutation_rate_forward) + (1.0 - p) * self.mutation_rate_backward;

        // Selection: compute mean fitness and frequency after selection.
        let s = self.selection_coefficient;
        let h = self.dominance;
        let q = 1.0 - p_after_mut;

        // Genotype fitnesses: AA = 1+s, Aa = 1+h*s, aa = 1.
        let w_aa = 1.0 + s;
        let w_ab = 1.0 + h * s;
        let w_bb = 1.0;

        let w_bar = p_after_mut * p_after_mut * w_aa + 2.0 * p_after_mut * q * w_ab + q * q * w_bb;

        let p_after_sel = if w_bar > 0.0 {
            (p_after_mut * p_after_mut * w_aa + p_after_mut * q * w_ab) / w_bar
        } else {
            p_after_mut
        };

        let p_clamped = p_after_sel.clamp(0.0, 1.0);

        // Genetic drift: binomial sampling of 2N gametes.
        let two_n = 2 * self.pop_size;
        let successes = binomial(&mut self.rng_state, two_n, p_clamped);
        self.allele_freq = successes as f64 / two_n as f64;
        self.generation += 1;
        self.allele_freq
    }

    /// Run until the focal allele fixes (freq = 1) or is lost (freq = 0).
    ///
    /// Returns (generations_elapsed, true_if_fixed).
    /// Safety cap at 10 * 4 * Ne generations to prevent infinite loops.
    pub fn run_until_fixation(&mut self) -> (u32, bool) {
        let max_gen = (40 * self.pop_size as u32).max(100_000);
        let start = self.generation;
        while self.allele_freq > 0.0 && self.allele_freq < 1.0 && self.generation - start < max_gen
        {
            self.step();
        }
        let fixed = self.allele_freq >= 1.0;
        (self.generation - start, fixed)
    }

    /// Expected heterozygosity: 2p(1-p).
    pub fn expected_heterozygosity(&self) -> f64 {
        2.0 * self.allele_freq * (1.0 - self.allele_freq)
    }

    /// Kimura's diffusion approximation for fixation probability of a
    /// new mutation with selection coefficient `s` in a population of
    /// effective size `ne`.
    ///
    /// P_fix = (1 - e^{-2s}) / (1 - e^{-4Ns})
    ///
    /// For neutral mutations (s = 0), P_fix = 1/(2N).
    pub fn fixation_probability(ne: usize, s: f64) -> f64 {
        if ne == 0 {
            return 0.0;
        }
        // Neutral case: avoid division by zero in the exponential formula.
        if s.abs() < 1e-12 {
            return 1.0 / (2.0 * ne as f64);
        }
        let numerator = 1.0 - (-2.0 * s).exp();
        let denominator = 1.0 - (-4.0 * ne as f64 * s).exp();
        if denominator.abs() < 1e-300 {
            // Very strong selection: fixation almost certain for beneficial,
            // almost impossible for deleterious.
            return if s > 0.0 { 1.0 } else { 0.0 };
        }
        (numerator / denominator).clamp(0.0, 1.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Moran process
// ─────────────────────────────────────────────────────────────────────────────

impl MoranProcess {
    /// Create a new Moran process.
    ///
    /// `counts[i]` is the number of individuals of type i.
    /// `fitness[i]` is the relative fitness of type i.
    pub fn new(counts: Vec<usize>, fitness: Vec<f64>, seed: u64) -> Self {
        assert_eq!(counts.len(), fitness.len(), "counts and fitness must match");
        Self {
            counts,
            fitness,
            generation: 0,
            rng_state: if seed == 0 { 0xBEEF_CAFE_0001 } else { seed },
        }
    }

    /// Perform one Moran step.
    ///
    /// One individual reproduces (fitness-proportional) and one random
    /// individual dies.  Returns `Some(i)` if type `i` has fixed.
    pub fn step(&mut self) -> Option<usize> {
        let total_n: usize = self.counts.iter().sum();
        if total_n == 0 {
            return None;
        }

        // Build fitness-weighted reproduction probabilities.
        let weights: Vec<f64> = self
            .counts
            .iter()
            .zip(self.fitness.iter())
            .map(|(&c, &f)| c as f64 * f)
            .collect();

        let reproducer = weighted_choice(&mut self.rng_state, &weights);

        // Random death (uniform across all individuals).
        let death_idx = (xorshift64(&mut self.rng_state) % total_n as u64) as usize;
        let mut cumulative = 0usize;
        let mut dying_type = 0usize;
        for (i, &c) in self.counts.iter().enumerate() {
            cumulative += c;
            if death_idx < cumulative {
                dying_type = i;
                break;
            }
        }

        // Apply birth and death.
        if self.counts[dying_type] > 0 {
            self.counts[dying_type] -= 1;
        }
        self.counts[reproducer] += 1;
        self.generation += 1;

        // Check for fixation.
        let alive_types: Vec<usize> = self
            .counts
            .iter()
            .enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(i, _)| i)
            .collect();
        if alive_types.len() == 1 {
            Some(alive_types[0])
        } else {
            None
        }
    }

    /// Run until one type fixes.
    ///
    /// Returns (steps_elapsed, index_of_fixed_type).
    pub fn run_until_fixation(&mut self) -> (u32, usize) {
        let total_n: usize = self.counts.iter().sum();
        let max_steps = (total_n as u32 * total_n as u32).max(100_000);
        for _ in 0..max_steps {
            if let Some(fixed) = self.step() {
                return (self.generation, fixed);
            }
        }
        // If not fixed within limit, return the most common type.
        let best = self
            .counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(i, _)| i)
            .unwrap_or(0);
        (self.generation, best)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Coalescent simulation
// ─────────────────────────────────────────────────────────────────────────────

/// Simulate a coalescent genealogy for `n_samples` haploid lineages.
///
/// Uses Kingman's coalescent: at each step the time to the next coalescence
/// is exponentially distributed with rate k(k-1)/2 where k is the number
/// of active lineages.  Mutations are placed on branches at rate
/// `mutation_rate` per generation per lineage.
///
/// Returns a vector of `CoalescentNode`s.  The first `n_samples` are leaf
/// nodes (time = 0); subsequent nodes are internal coalescence events.
pub fn coalescent_simulation(
    n_samples: usize,
    ne: f64,
    mutation_rate: f64,
    seed: u64,
) -> Vec<CoalescentNode> {
    if n_samples == 0 {
        return Vec::new();
    }
    let mut rng_state = if seed == 0 { 0xC0A1_E5CE_0001 } else { seed };
    let mut nodes: Vec<CoalescentNode> = Vec::new();
    let mut next_id = 0u64;

    // Create leaf nodes.
    let mut active: Vec<u64> = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        nodes.push(CoalescentNode {
            id: next_id,
            time: 0.0,
            children: Vec::new(),
            mutations: 0,
        });
        active.push(next_id);
        next_id += 1;
    }

    let mut current_time = 0.0;

    while active.len() > 1 {
        let k = active.len() as f64;
        // Rate of coalescence for k lineages: k(k-1) / (2 * 2Ne)
        // In units of 2Ne generations: rate = k(k-1)/2
        // Time is in generations: rate = k(k-1) / (4Ne)
        let rate = k * (k - 1.0) / (4.0 * ne);
        if rate <= 0.0 {
            break;
        }

        // Exponential waiting time: -ln(U) / rate
        let u = rand_f64(&mut rng_state).max(1e-15);
        let wait = -u.ln() / rate;
        current_time += wait;

        // Choose two lineages to coalesce (uniform random pair).
        let n_active = active.len();
        let idx1 = (xorshift64(&mut rng_state) % n_active as u64) as usize;
        let mut idx2 = (xorshift64(&mut rng_state) % (n_active - 1) as u64) as usize;
        if idx2 >= idx1 {
            idx2 += 1;
        }

        let child1 = active[idx1];
        let child2 = active[idx2];

        // Place mutations on each branch (Poisson process).
        // Expected mutations on branch of length `wait`: mutation_rate * wait * 2Ne
        // (since time is in generations and mu is per-generation per-site).
        let branch_length = wait;
        let expected_muts = mutation_rate * branch_length;

        // Poisson sampling via counting exponential interarrival times.
        let muts1 = poisson_sample(&mut rng_state, expected_muts);
        let muts2 = poisson_sample(&mut rng_state, expected_muts);

        // Assign mutations to children.
        // Find child1 and child2 in nodes and add mutations.
        for node in nodes.iter_mut() {
            if node.id == child1 {
                node.mutations += muts1;
            } else if node.id == child2 {
                node.mutations += muts2;
            }
        }

        // Create parent node.
        nodes.push(CoalescentNode {
            id: next_id,
            time: current_time,
            children: vec![child1, child2],
            mutations: 0,
        });

        // Update active lineages.
        // Remove the higher index first to avoid shifting.
        let (lo, hi) = if idx1 < idx2 {
            (idx1, idx2)
        } else {
            (idx2, idx1)
        };
        active.remove(hi);
        active.remove(lo);
        active.push(next_id);
        next_id += 1;
    }

    nodes
}

/// Sample from a Poisson distribution with given mean (lambda).
fn poisson_sample(state: &mut u64, lambda: f64) -> u32 {
    if lambda <= 0.0 {
        return 0;
    }
    // Knuth's algorithm for small lambda.
    if lambda < 30.0 {
        let l = (-lambda).exp();
        let mut k = 0u32;
        let mut p = 1.0f64;
        loop {
            k += 1;
            p *= rand_f64(state);
            if p <= l {
                return k - 1;
            }
            // Safety cap to prevent pathological loops.
            if k > 1000 {
                return k;
            }
        }
    } else {
        // Normal approximation for large lambda.
        let u1 = rand_f64(state).max(1e-15);
        let u2 = rand_f64(state);
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let x = (lambda + z * lambda.sqrt() + 0.5).round() as i64;
        x.max(0) as u32
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// F-statistics (Weir & Cockerham 1984)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute Wright's F-statistics across subpopulations for a given locus.
///
/// Uses the Weir & Cockerham (1984) estimator for biallelic loci:
///   Fst = (Ht - Hs) / Ht
/// where Ht is the expected heterozygosity in the total population and
/// Hs is the average expected heterozygosity within subpopulations.
///
/// `locus_idx` identifies which locus (index into `Subpopulation.loci`)
/// to analyze.  Assumes the first allele frequency at each locus is used
/// for a biallelic computation.
pub fn f_statistics(subpops: &[Subpopulation], locus_idx: usize) -> FStatistics {
    if subpops.is_empty() {
        return FStatistics {
            fst: 0.0,
            fis: 0.0,
            fit: 0.0,
        };
    }

    // Collect allele frequencies (p) for the focal allele in each subpop.
    let mut freqs: Vec<f64> = Vec::with_capacity(subpops.len());
    let mut sizes: Vec<f64> = Vec::with_capacity(subpops.len());

    for sp in subpops {
        if locus_idx < sp.loci.len() && !sp.loci[locus_idx].allele_frequencies.is_empty() {
            freqs.push(sp.loci[locus_idx].allele_frequencies[0]);
            sizes.push(sp.size as f64);
        }
    }

    if freqs.is_empty() {
        return FStatistics {
            fst: 0.0,
            fis: 0.0,
            fit: 0.0,
        };
    }

    let total_n: f64 = sizes.iter().sum();
    if total_n <= 0.0 {
        return FStatistics {
            fst: 0.0,
            fis: 0.0,
            fit: 0.0,
        };
    }

    // Weighted mean allele frequency across subpopulations.
    let p_bar: f64 = freqs
        .iter()
        .zip(sizes.iter())
        .map(|(&p, &n)| p * n)
        .sum::<f64>()
        / total_n;
    let q_bar = 1.0 - p_bar;

    // Expected heterozygosity in total population (Ht).
    let ht = 2.0 * p_bar * q_bar;

    // Average expected heterozygosity within subpopulations (Hs).
    let hs: f64 = freqs
        .iter()
        .zip(sizes.iter())
        .map(|(&p, &n)| 2.0 * p * (1.0 - p) * n)
        .sum::<f64>()
        / total_n;

    // Fst
    let fst = if ht > 1e-15 { (ht - hs) / ht } else { 0.0 };

    // For Fis, we need observed heterozygosity (Ho).
    // Without individual genotype data, approximate Fis from variance in
    // allele frequencies (Wahlund-like).
    // Fis measures inbreeding within subpopulations.  Without genotype data
    // we estimate Fis = 0 (random mating within demes).
    let fis = 0.0;

    // Fit = Fst + Fis - Fst*Fis (hierarchical relationship)
    let fit = fst + fis - fst * fis;

    FStatistics {
        fst: fst.clamp(0.0, 1.0),
        fis,
        fit: fit.clamp(0.0, 1.0),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tajima's D
// ─────────────────────────────────────────────────────────────────────────────

/// Compute Tajima's D statistic.
///
/// Tests for departures from selective neutrality by comparing two estimators
/// of the population mutation rate (theta):
///   theta_pi  = average pairwise nucleotide differences
///   theta_w   = S / a_1 (Watterson's estimator from segregating sites)
///
/// Under neutral evolution with constant population size, D ~ 0.
///
/// # Arguments
/// - `segregating_sites` (S): number of polymorphic sites.
/// - `n_sequences`: number of sampled sequences.
/// - `pairwise_differences`: average number of pairwise differences (pi).
pub fn tajimas_d(
    segregating_sites: usize,
    n_sequences: usize,
    pairwise_differences: f64,
) -> TajimaD {
    let n = n_sequences as f64;
    if n < 2.0 || segregating_sites == 0 {
        return TajimaD {
            d_statistic: 0.0,
            theta_pi: pairwise_differences,
            theta_w: 0.0,
            interpretation: SelectionSignal::Neutral,
        };
    }
    let s = segregating_sites as f64;

    // Harmonic numbers.
    let mut a1 = 0.0f64;
    let mut a2 = 0.0f64;
    for i in 1..(n_sequences) {
        let fi = i as f64;
        a1 += 1.0 / fi;
        a2 += 1.0 / (fi * fi);
    }

    let theta_w = s / a1;
    let theta_pi = pairwise_differences;

    // Variance components (Tajima 1989, equations 38-39).
    let b1 = (n + 1.0) / (3.0 * (n - 1.0));
    let b2 = 2.0 * (n * n + n + 3.0) / (9.0 * n * (n - 1.0));
    let c1 = b1 - 1.0 / a1;
    let c2 = b2 - (n + 2.0) / (a1 * n) + a2 / (a1 * a1);
    let e1 = c1 / a1;
    let e2 = c2 / (a1 * a1 + a2);

    let var_d = e1 * s + e2 * s * (s - 1.0);
    let denominator = if var_d > 0.0 { var_d.sqrt() } else { 1.0 };

    let d = (theta_pi - theta_w) / denominator;

    // Interpretation thresholds.
    let interpretation = if d < -2.0 {
        SelectionSignal::PositiveSelection
    } else if d > 2.0 {
        SelectionSignal::BalancingSelection
    } else {
        SelectionSignal::Neutral
    };

    TajimaD {
        d_statistic: d,
        theta_pi,
        theta_w,
        interpretation,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Hardy-Weinberg equilibrium
// ─────────────────────────────────────────────────────────────────────────────

/// Test for Hardy-Weinberg equilibrium in a biallelic locus.
///
/// Takes genotype counts: `[AA, Aa, aa]` and computes expected frequencies
/// under HWE (p^2, 2pq, q^2), then performs a chi-squared goodness-of-fit
/// test with 1 degree of freedom.
pub fn hardy_weinberg_test(genotype_counts: &[usize; 3]) -> HardyWeinbergTest {
    let n_aa = genotype_counts[0] as f64;
    let n_ab = genotype_counts[1] as f64;
    let n_bb = genotype_counts[2] as f64;
    let n_total = n_aa + n_ab + n_bb;

    if n_total < 1.0 {
        return HardyWeinbergTest {
            observed_het: 0.0,
            expected_het: 0.0,
            chi_squared: 0.0,
            in_equilibrium: true,
            excess_het: false,
        };
    }

    // Allele frequencies.
    let p = (2.0 * n_aa + n_ab) / (2.0 * n_total);
    let q = 1.0 - p;

    // Expected genotype counts under HWE.
    let e_aa = p * p * n_total;
    let e_ab = 2.0 * p * q * n_total;
    let e_bb = q * q * n_total;

    let observed_het = n_ab / n_total;
    let expected_het = 2.0 * p * q;

    // Chi-squared statistic.
    let chi_sq = if e_aa > 0.0 {
        (n_aa - e_aa) * (n_aa - e_aa) / e_aa
    } else {
        0.0
    } + if e_ab > 0.0 {
        (n_ab - e_ab) * (n_ab - e_ab) / e_ab
    } else {
        0.0
    } + if e_bb > 0.0 {
        (n_bb - e_bb) * (n_bb - e_bb) / e_bb
    } else {
        0.0
    };

    // Critical value at alpha = 0.05 with 1 d.f.: 3.841.
    let in_equilibrium = chi_sq < 3.841;
    let excess_het = observed_het > expected_het;

    HardyWeinbergTest {
        observed_het,
        expected_het,
        chi_squared: chi_sq,
        in_equilibrium,
        excess_het,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Linkage disequilibrium
// ─────────────────────────────────────────────────────────────────────────────

/// Compute linkage disequilibrium between two biallelic loci.
///
/// Takes haplotype frequencies: `[AB, Ab, aB, ab]` (must sum to 1).
///
/// Computes:
///   D    = f(AB) - f(A)*f(B)
///   D'   = D / D_max (Lewontin 1964)
///   r^2  = D^2 / (f(A)*f(a)*f(B)*f(b))
pub fn linkage_disequilibrium(haplotype_freqs: &[f64; 4]) -> LDResult {
    let f_ab = haplotype_freqs[0]; // AB
    let f_a_b = haplotype_freqs[1]; // Ab (A with b)
    let f_ab2 = haplotype_freqs[2]; // aB
    let _f_a_b2 = haplotype_freqs[3]; // ab

    // Marginal allele frequencies.
    let f_a = f_ab + f_a_b; // freq of A
    let f_b = f_ab + f_ab2; // freq of B
    let f_a_lower = 1.0 - f_a; // freq of a
    let f_b_lower = 1.0 - f_b; // freq of b

    // Coefficient of linkage disequilibrium.
    let d = f_ab - f_a * f_b;

    // Normalized D' (Lewontin 1964).
    let d_max = if d >= 0.0 {
        (f_a * f_b_lower).min(f_a_lower * f_b)
    } else {
        (f_a * f_b).min(f_a_lower * f_b_lower)
    };
    let d_prime = if d_max.abs() > 1e-15 {
        (d / d_max).clamp(-1.0, 1.0)
    } else {
        0.0
    };

    // Squared correlation coefficient.
    let denom = f_a * f_a_lower * f_b * f_b_lower;
    let r_squared = if denom > 1e-15 {
        (d * d / denom).min(1.0)
    } else {
        0.0
    };

    LDResult {
        d_prime: d_prime.abs(),
        r_squared,
        locus_pair: (0, 0),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Effective population size estimation
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate effective population size (Ne) from temporal allele frequency
/// change using the Nei & Tajima (1981) method.
///
/// Fc = (1/K) * sum_k [ (x_k - y_k)^2 / ((x_k + y_k)/2 - x_k * y_k) ]
/// Ne = t / (2 * (Fc - 1/(2*S)))
///
/// where x_k, y_k are allele frequencies at two time points, t is the
/// number of generations between samples, and S is the sample size.
///
/// For simplicity, assumes a large sample size (S -> inf).
pub fn ne_temporal(freq_before: &[f64], freq_after: &[f64], generations: u32) -> f64 {
    if freq_before.len() != freq_after.len() || freq_before.is_empty() || generations == 0 {
        return f64::INFINITY;
    }
    let mut fc = 0.0;
    let mut valid = 0.0;

    for (&x, &y) in freq_before.iter().zip(freq_after.iter()) {
        let mean = (x + y) / 2.0;
        let denom = mean - x * y;
        if denom.abs() > 1e-15 {
            fc += (x - y) * (x - y) / denom;
            valid += 1.0;
        }
    }

    if valid < 1.0 || fc <= 0.0 {
        return f64::INFINITY;
    }
    fc /= valid;

    let t = generations as f64;
    let ne = t / (2.0 * fc);
    ne.max(1.0)
}

/// Expected heterozygosity under mutation-drift balance.
///
/// H = 4*Ne*mu / (1 + 4*Ne*mu)
///
/// This is the infinite-alleles model equilibrium heterozygosity.
pub fn expected_het_mutation_drift(ne: f64, mutation_rate: f64) -> f64 {
    let theta = 4.0 * ne * mutation_rate;
    theta / (1.0 + theta)
}

/// Compute the Wahlund effect: reduction in observed heterozygosity
/// when distinct subpopulations are lumped together.
///
/// If subpopulations have allele frequencies p_1, p_2, ..., then
/// the heterozygosity deficit = Var(p) = E[p^2] - (E[p])^2.
///
/// Returns the heterozygosity deficit (always >= 0).
pub fn wahlund_effect(subpop_frequencies: &[f64]) -> f64 {
    if subpop_frequencies.is_empty() {
        return 0.0;
    }
    let n = subpop_frequencies.len() as f64;
    let mean_p: f64 = subpop_frequencies.iter().sum::<f64>() / n;
    let mean_p_sq: f64 = subpop_frequencies.iter().map(|&p| p * p).sum::<f64>() / n;
    let var_p = mean_p_sq - mean_p * mean_p;
    var_p.max(0.0)
}

/// Watterson's theta: population mutation rate estimated from the number
/// of segregating sites.
///
/// theta_w = S / a_1
///
/// where a_1 = sum_{i=1}^{n-1} 1/i (the (n-1)th harmonic number).
pub fn wattersons_theta(segregating_sites: usize, n_sequences: usize) -> f64 {
    if n_sequences < 2 {
        return 0.0;
    }
    let mut a1 = 0.0f64;
    for i in 1..n_sequences {
        a1 += 1.0 / i as f64;
    }
    if a1 <= 0.0 {
        return 0.0;
    }
    segregating_sites as f64 / a1
}

/// Nucleotide diversity (theta_pi): average number of pairwise differences
/// per site, normalized by the number of pairwise comparisons.
///
/// theta_pi = pi / C(n,2) = pi / (n*(n-1)/2)
///
/// where `pairwise_diffs` is the total sum of pairwise differences and
/// `n_sequences` is the number of sequences.
pub fn nucleotide_diversity(pairwise_diffs: f64, n_sequences: usize) -> f64 {
    if n_sequences < 2 {
        return 0.0;
    }
    let n = n_sequences as f64;
    let n_pairs = n * (n - 1.0) / 2.0;
    pairwise_diffs / n_pairs
}

/// McDonald-Kreitman test: compare the ratio of nonsynonymous to synonymous
/// divergence (between species) with the ratio of nonsynonymous to
/// synonymous polymorphism (within species).
///
/// Under neutrality, Dn/Ds = Pn/Ps.
///
/// The neutrality index (NI) = (Pn/Ps) / (Dn/Ds).
/// NI > 1 indicates excess nonsynonymous polymorphism (purifying selection).
/// NI < 1 indicates adaptive evolution (positive selection on fixed differences).
///
/// # Arguments
/// - `dn`: nonsynonymous divergence (fixed differences between species)
/// - `ds`: synonymous divergence
/// - `pn`: nonsynonymous polymorphism (within species)
/// - `ps`: synonymous polymorphism
///
/// Returns (neutrality_index, significant_departure).
/// Significance is based on a G-test at alpha = 0.05.
pub fn mcdonald_kreitman(dn: usize, ds: usize, pn: usize, ps: usize) -> (f64, bool) {
    let dn_f = dn as f64;
    let ds_f = ds as f64;
    let pn_f = pn as f64;
    let ps_f = ps as f64;

    // Avoid division by zero.
    if ds == 0 || ps == 0 || dn == 0 {
        return (1.0, false);
    }

    let ni = (pn_f / ps_f) / (dn_f / ds_f);

    // G-test (log-likelihood ratio test) for the 2x2 table:
    //            | Fixed | Polymorphic |
    // Nonsyn     |  Dn   |     Pn      |
    // Syn        |  Ds   |     Ps      |
    let total = dn_f + ds_f + pn_f + ps_f;
    let row1 = dn_f + pn_f;
    let row2 = ds_f + ps_f;
    let col1 = dn_f + ds_f;
    let col2 = pn_f + ps_f;

    let g_stat = 2.0
        * [
            (dn_f, row1, col1),
            (pn_f, row1, col2),
            (ds_f, row2, col1),
            (ps_f, row2, col2),
        ]
        .iter()
        .map(|&(obs, row, col)| {
            if obs > 0.0 && row > 0.0 && col > 0.0 && total > 0.0 {
                let expected = row * col / total;
                obs * (obs / expected).ln()
            } else {
                0.0
            }
        })
        .sum::<f64>();

    // Chi-squared critical value at alpha = 0.05, 1 d.f.: 3.841.
    let significant = g_stat > 3.841;

    (ni, significant)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Tolerance for floating-point comparisons.
    const EPS: f64 = 1e-6;

    /// Helper: run many Wright-Fisher replicates and return the fraction
    /// that fix the focal allele.
    #[allow(dead_code)]
    fn fixation_fraction(pop_size: usize, initial_freq: f64, s: f64, h: f64, reps: usize) -> f64 {
        let mut fixed = 0usize;
        for i in 0..reps {
            let mut sim = WrightFisherSim::new(pop_size, initial_freq, 1000 + i as u64);
            sim.set_selection(s, h);
            let (_, did_fix) = sim.run_until_fixation();
            if did_fix {
                fixed += 1;
            }
        }
        fixed as f64 / reps as f64
    }

    // ---- 1. Wright-Fisher drift ----
    #[test]
    fn wright_fisher_drift() {
        // A neutral allele at frequency 0.5 in a small population should
        // drift.  After many generations the frequency should differ from 0.5.
        // We verify that the frequency changes (i.e., drift occurs) and stays
        // in [0, 1].
        let mut sim = WrightFisherSim::new(50, 0.5, 42);
        let mut changed = false;
        for _ in 0..200 {
            let freq = sim.step();
            assert!(
                freq >= 0.0 && freq <= 1.0,
                "frequency out of range: {}",
                freq
            );
            if (freq - 0.5).abs() > 0.1 {
                changed = true;
            }
        }
        assert!(changed, "neutral allele at N=50 should drift away from 0.5");
    }

    // ---- 2. Wright-Fisher selection ----
    #[test]
    fn wright_fisher_selection() {
        // A strongly beneficial allele (s=0.1) starting at frequency 0.1
        // should increase in frequency on average.
        let mut total_freq = 0.0;
        let reps = 100;
        for seed in 0..reps {
            let mut sim = WrightFisherSim::new(200, 0.1, 100 + seed);
            sim.set_selection(0.1, 0.5);
            for _ in 0..50 {
                sim.step();
            }
            total_freq += sim.allele_freq;
        }
        let mean_freq = total_freq / reps as f64;
        // With s=0.1, additive, starting at 0.1 for 50 gen in N=200,
        // the mean should be substantially above 0.1.
        assert!(
            mean_freq > 0.3,
            "beneficial allele should increase: mean = {}",
            mean_freq
        );
    }

    // ---- 3. Fixation probability neutral ----
    #[test]
    fn fixation_probability_neutral() {
        // For a neutral mutation (s=0), P_fix = 1/(2N).
        let ne = 100;
        let p_fix = WrightFisherSim::fixation_probability(ne, 0.0);
        let expected = 1.0 / (2.0 * ne as f64);
        assert!(
            (p_fix - expected).abs() < EPS,
            "neutral P_fix = {} but expected {}",
            p_fix,
            expected
        );
    }

    // ---- 4. Fixation probability beneficial ----
    #[test]
    fn fixation_probability_beneficial() {
        // For strongly beneficial mutation with large N,
        // P_fix ~ 2s (Haldane's approximation for h=0.5 additive).
        // Kimura's formula: (1 - e^{-2s}) / (1 - e^{-4Ns})
        // For N=10000, s=0.01: P_fix should be close to 2*0.01 = 0.02.
        let ne = 10_000;
        let s = 0.01;
        let p_fix = WrightFisherSim::fixation_probability(ne, s);
        // The exact Kimura value for 4Ns=400 is essentially (1-e^{-0.02})/(1-e^{-400})
        // ~ 0.0198 / 1.0 ~ 0.0198.
        assert!(
            (p_fix - 0.0198).abs() < 0.005,
            "beneficial P_fix = {} but expected ~0.0198",
            p_fix
        );
    }

    // ---- 5. Hardy-Weinberg equilibrium ----
    #[test]
    fn hardy_weinberg_equilibrium() {
        // A population in HWE: p=0.6, N=1000.
        // Expected: AA=360, Aa=480, aa=160.
        let result = hardy_weinberg_test(&[360, 480, 160]);
        assert!(
            result.in_equilibrium,
            "HWE population should be in equilibrium, chi_sq = {}",
            result.chi_squared
        );
        assert!(
            (result.observed_het - 0.48).abs() < 0.01,
            "observed het should be ~0.48"
        );
        assert!(
            (result.expected_het - 0.48).abs() < 0.01,
            "expected het should be ~0.48"
        );
    }

    // ---- 6. Hardy-Weinberg departure ----
    #[test]
    fn hardy_weinberg_departure() {
        // Excess homozygotes (inbreeding): p=0.5 but only 10% heterozygotes.
        // N=1000: AA=450, Aa=100, aa=450.
        // Expected under HWE: AA=250, Aa=500, aa=250.
        // This should be a massive departure.
        let result = hardy_weinberg_test(&[450, 100, 450]);
        assert!(
            !result.in_equilibrium,
            "assortative mating should depart from HWE, chi_sq = {}",
            result.chi_squared
        );
        assert!(!result.excess_het, "should show deficit of heterozygotes");
    }

    // ---- 7. Fst identical pops zero ----
    #[test]
    fn fst_identical_pops_zero() {
        // Two identical subpopulations should have Fst = 0.
        let locus = Locus {
            name: "locus1".to_string(),
            allele_frequencies: vec![0.5, 0.5],
            position_cm: 0.0,
        };
        let pop_a = Subpopulation {
            name: "A".to_string(),
            size: 100,
            loci: vec![locus.clone()],
            fitness_values: vec![],
        };
        let pop_b = Subpopulation {
            name: "B".to_string(),
            size: 100,
            loci: vec![locus],
            fitness_values: vec![],
        };
        let fst = f_statistics(&[pop_a, pop_b], 0);
        assert!(
            fst.fst.abs() < EPS,
            "identical pops should have Fst=0, got {}",
            fst.fst
        );
    }

    // ---- 8. Fst divergent pops positive ----
    #[test]
    fn fst_divergent_pops_positive() {
        // Two divergent subpopulations (p=0.9 vs p=0.1) should have high Fst.
        let pop_a = Subpopulation {
            name: "A".to_string(),
            size: 100,
            loci: vec![Locus {
                name: "locus1".to_string(),
                allele_frequencies: vec![0.9, 0.1],
                position_cm: 0.0,
            }],
            fitness_values: vec![],
        };
        let pop_b = Subpopulation {
            name: "B".to_string(),
            size: 100,
            loci: vec![Locus {
                name: "locus1".to_string(),
                allele_frequencies: vec![0.1, 0.9],
                position_cm: 0.0,
            }],
            fitness_values: vec![],
        };
        let fst = f_statistics(&[pop_a, pop_b], 0);
        // Fst = (Ht - Hs) / Ht
        // p_bar = 0.5, Ht = 0.5
        // Hs = (2*0.9*0.1 + 2*0.1*0.9)/2 = 0.18
        // Fst = (0.5 - 0.18) / 0.5 = 0.64
        assert!(
            fst.fst > 0.5,
            "divergent pops should have high Fst, got {}",
            fst.fst
        );
    }

    // ---- 9. Tajima's D neutral ----
    #[test]
    fn tajimas_d_neutral() {
        // Under neutrality, theta_pi ~ theta_w, so D ~ 0.
        // For n=50 sequences with 20 segregating sites, if pairwise
        // differences match Watterson's expectation, D should be near 0.
        let n = 50;
        let s = 20;
        let theta_w = wattersons_theta(s, n);
        // If pairwise_differences = theta_w, then D = 0 exactly.
        let result = tajimas_d(s, n, theta_w);
        assert!(
            result.d_statistic.abs() < EPS,
            "D should be ~0 when theta_pi == theta_w, got {}",
            result.d_statistic
        );
        assert_eq!(result.interpretation, SelectionSignal::Neutral);
    }

    // ---- 10. LD independent loci ----
    #[test]
    fn ld_independent_loci() {
        // For two unlinked loci at linkage equilibrium:
        // f(A) = 0.6, f(B) = 0.4
        // f(AB) = 0.24, f(Ab) = 0.36, f(aB) = 0.16, f(ab) = 0.24
        let result = linkage_disequilibrium(&[0.24, 0.36, 0.16, 0.24]);
        assert!(
            result.r_squared < 0.001,
            "independent loci should have r^2 ~ 0, got {}",
            result.r_squared
        );
    }

    // ---- 11. Wahlund reduces het ----
    #[test]
    fn wahlund_reduces_het() {
        // Two subpopulations with different allele frequencies should show
        // a positive Wahlund effect (heterozygosity deficit).
        let deficit = wahlund_effect(&[0.2, 0.8]);
        // Var(p) = ((0.2-0.5)^2 + (0.8-0.5)^2)/2 = 0.09
        assert!(
            (deficit - 0.09).abs() < EPS,
            "Wahlund deficit should be 0.09, got {}",
            deficit
        );
        assert!(deficit > 0.0, "Wahlund effect should be positive");

        // Three subpopulations at p = 0.3, 0.5, 0.7: Var = 0.0267
        let deficit3 = wahlund_effect(&[0.3, 0.5, 0.7]);
        assert!(deficit3 > 0.02 && deficit3 < 0.03);

        // Identical subpopulations: no Wahlund effect.
        let deficit_zero = wahlund_effect(&[0.5, 0.5, 0.5]);
        assert!(
            deficit_zero < EPS,
            "identical pops should have zero Wahlund deficit"
        );
    }

    // ---- 12. Moran process fixation ----
    #[test]
    fn moran_process_fixation() {
        // Two types with equal fitness: one must eventually fix.
        let mut fixed_0 = 0;
        let mut fixed_1 = 0;
        let reps = 50;
        for seed in 0..reps {
            let mut moran = MoranProcess::new(vec![5, 5], vec![1.0, 1.0], 42 + seed as u64);
            let (_, which) = moran.run_until_fixation();
            if which == 0 {
                fixed_0 += 1;
            } else {
                fixed_1 += 1;
            }
            // Verify total population size is conserved.
            let total: usize = moran.counts.iter().sum();
            assert_eq!(total, 10, "Moran process should conserve population size");
        }
        // With equal fitness and equal starting counts, fixation should be
        // roughly 50/50.  Allow wide margin for stochasticity.
        assert!(fixed_0 + fixed_1 == reps);
        assert!(fixed_0 > 5, "type 0 should fix sometimes: {}", fixed_0);
        assert!(fixed_1 > 5, "type 1 should fix sometimes: {}", fixed_1);
    }

    // ---- 13. Watterson's theta calculation ----
    #[test]
    fn wattersons_theta_calculation() {
        // For n=10 sequences, S=20 segregating sites:
        // a_1 = 1 + 1/2 + 1/3 + ... + 1/9 = 2.828968...
        // theta_w = 20 / 2.828968 = 7.0732...
        let theta = wattersons_theta(20, 10);
        let a1: f64 = (1..10).map(|i| 1.0 / i as f64).sum();
        let expected = 20.0 / a1;
        assert!(
            (theta - expected).abs() < EPS,
            "theta_w = {} but expected {}",
            theta,
            expected
        );
    }

    // ---- 14. Coalescent MRCA exists ----
    #[test]
    fn coalescent_mrca_exists() {
        // With 10 samples, the coalescent should produce exactly 9 internal
        // nodes (10-1), and the root should have the largest time.
        let nodes = coalescent_simulation(10, 1000.0, 1e-4, 99);
        let n_leaves = 10;
        let n_internal = nodes.len() - n_leaves;
        assert_eq!(
            n_internal,
            n_leaves - 1,
            "should have {} internal nodes, got {}",
            n_leaves - 1,
            n_internal
        );

        // The last node should be the root (MRCA) with the largest time.
        let root = nodes.last().unwrap();
        assert!(root.time > 0.0, "MRCA time should be > 0");
        assert_eq!(root.children.len(), 2, "root should have 2 children");

        // All internal nodes should have exactly 2 children.
        for node in &nodes[n_leaves..] {
            assert_eq!(
                node.children.len(),
                2,
                "internal node {} should have 2 children",
                node.id
            );
        }

        // All leaf nodes should have time = 0.
        for node in &nodes[..n_leaves] {
            assert!(
                (node.time - 0.0).abs() < EPS,
                "leaf node {} should have time=0",
                node.id
            );
        }
    }

    // ---- Additional: nucleotide diversity ----
    #[test]
    fn nucleotide_diversity_calculation() {
        // n=5 sequences, total pairwise diffs = 20.
        // C(5,2) = 10 pairs.
        // pi = 20/10 = 2.0.
        let pi = nucleotide_diversity(20.0, 5);
        assert!((pi - 2.0).abs() < EPS, "pi should be 2.0, got {}", pi);
    }

    // ---- Additional: McDonald-Kreitman test ----
    #[test]
    fn mcdonald_kreitman_neutral() {
        // Under strict neutrality: Dn/Ds = Pn/Ps, so NI = 1.
        // Dn=10, Ds=20, Pn=10, Ps=20 => NI = (10/20)/(10/20) = 1.0.
        let (ni, sig) = mcdonald_kreitman(10, 20, 10, 20);
        assert!(
            (ni - 1.0).abs() < EPS,
            "neutral MK should have NI=1.0, got {}",
            ni
        );
        assert!(!sig, "neutral MK should not be significant");
    }

    // ---- Additional: expected_het_mutation_drift ----
    #[test]
    fn expected_het_mutation_drift_known() {
        // Ne=1000, mu=1e-4 => theta=0.4 => H = 0.4/1.4 = 0.2857...
        let h = expected_het_mutation_drift(1000.0, 1e-4);
        let expected = 0.4 / 1.4;
        assert!(
            (h - expected).abs() < EPS,
            "H should be {}, got {}",
            expected,
            h
        );
    }

    // ---- Additional: LD with perfect linkage ----
    #[test]
    fn ld_perfect_linkage() {
        // Complete linkage: only AB and ab haplotypes exist.
        // f(A) = 0.6, f(B) = 0.6
        // Haplotypes: AB=0.6, Ab=0.0, aB=0.0, ab=0.4
        // D = 0.6 - 0.6*0.6 = 0.24
        // D_max = min(0.6*0.4, 0.4*0.6) = 0.24
        // D' = 1.0
        // r^2 = 0.24^2 / (0.6*0.4*0.6*0.4) = 0.0576/0.0576 = 1.0
        let result = linkage_disequilibrium(&[0.6, 0.0, 0.0, 0.4]);
        assert!(
            (result.d_prime - 1.0).abs() < 0.01,
            "perfect linkage should have D'=1, got {}",
            result.d_prime
        );
        assert!(
            (result.r_squared - 1.0).abs() < 0.01,
            "perfect linkage should have r^2=1, got {}",
            result.r_squared
        );
    }

    // ---- Additional: ne_temporal with no change ----
    #[test]
    fn ne_temporal_no_change_is_large() {
        // If allele frequencies don't change, Ne should be very large (infinite).
        let ne = ne_temporal(&[0.3, 0.7], &[0.3, 0.7], 10);
        assert!(
            ne > 1e10 || ne == f64::INFINITY,
            "no freq change should give huge Ne, got {}",
            ne
        );
    }

    // ---- Additional: WF fixation probability deleterious ----
    #[test]
    fn fixation_probability_deleterious() {
        // Deleterious mutation (s < 0) should have very low fixation probability
        // in a large population.
        let p_fix = WrightFisherSim::fixation_probability(1000, -0.01);
        assert!(
            p_fix < 0.001,
            "deleterious mutation should have very low P_fix, got {}",
            p_fix
        );
    }
}
