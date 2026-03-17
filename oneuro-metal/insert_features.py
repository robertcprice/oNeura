#!/usr/bin/env python3
"""Atomically insert new features into terrarium_evolve.rs before the test section."""

import re

TARGET = "oneuro-metal/src/terrarium_evolve.rs"

NEW_CODE = r'''
// ---------------------------------------------------------------------------
// Adaptive Evolution with Environment
// ---------------------------------------------------------------------------

/// Full evolution loop that applies EnvironmentalSchedule during fitness
/// evaluation. Each world experiences seasonal temperature/humidity variation
/// and stochastic drought events, making climate-adaptive genomes fitter.
///
/// This wires `run_single_world_with_environment()` into the GA loop.
pub fn evolve_with_environment(
    config: EvolutionConfig,
    schedule: EnvironmentalSchedule,
) -> Result<EvolutionResult, String> {
    let start = Instant::now();
    let mut rng = StdRng::seed_from_u64(config.master_seed);
    let mut population: Vec<WorldGenome> = (0..config.population_size)
        .map(|_| WorldGenome::random(&mut rng))
        .collect();

    let mut global_best_fitness = f32::NEG_INFINITY;
    let mut global_best_genome = population[0].clone();
    let mut generation_results = Vec::with_capacity(config.generations);
    let mut total_worlds = 0usize;

    for gen in 0..config.generations {
        let gen_start = Instant::now();
        let mut results: Vec<WorldResult> = Vec::with_capacity(population.len());
        for genome in &population {
            let (wr, _samples) = run_single_world_with_environment(
                genome.clone(), config.frames_per_world, &schedule, config.lite,
            )?;
            results.push(wr);
        }
        total_worlds += results.len();

        let best_idx = results.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.fitness.partial_cmp(&b.fitness).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        let best = &results[best_idx];

        if best.fitness > global_best_fitness {
            global_best_fitness = best.fitness;
            global_best_genome = best.genome.clone();
        }

        let mean_fit = results.iter().map(|r| r.fitness).sum::<f32>() / results.len() as f32;
        eprintln!("  Gen {} | best={:.2} mean={:.2} | env={} | {:.1}s",
            gen, best.fitness, mean_fit,
            if schedule.drought_probability > 0.003 { "arid" }
            else if schedule.base_temperature_c > 26.0 { "tropical" }
            else { "temperate" },
            gen_start.elapsed().as_secs_f32());

        generation_results.push(GenerationResult {
            generation: gen,
            best_fitness: best.fitness,
            mean_fitness: mean_fit,
            best_genome: best.genome.clone(),
            world_results: results.clone(),
            wall_time_ms: gen_start.elapsed().as_secs_f32() * 1000.0,
            stress_metrics: None,
        });

        // Breed next generation
        let (tournament_size, mutation_rate, crossover_rate, elitism) = match &config.strategy {
            SearchStrategy::Evolutionary { tournament_size, mutation_rate, crossover_rate, elitism } =>
                (*tournament_size, *mutation_rate, *crossover_rate, *elitism),
            _ => (3, 0.15, 0.7, 2),
        };

        population = breed_next_generation(
            &results, &mut rng, config.population_size,
            tournament_size, mutation_rate, crossover_rate, elitism,
            &config.constraints,
        );
    }

    Ok(EvolutionResult {
        generation_results,
        global_best_fitness,
        global_best_genome,
        total_worlds_evaluated: total_worlds,
        total_wall_time_ms: start.elapsed().as_secs_f32() * 1000.0,
    })
}

// ---------------------------------------------------------------------------
// Coevolution Engine
// ---------------------------------------------------------------------------

/// Coevolution mode determines inter-species dynamics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CoevolutionMode {
    /// Red Queen: antagonistic arms race (predator–prey).
    RedQueen,
    /// Mutualistic: symbiotic coevolution (plant–pollinator).
    Mutualistic,
    /// Competitive: resource competition between species.
    Competitive,
}

/// A species genome that participates in coevolution.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpeciesGenome {
    pub world_genome: WorldGenome,
    /// Per-species trait modifiers for coevolution fitness.
    pub defense_investment: f32,
    pub resource_efficiency: f32,
    pub cooperation_tendency: f32,
    pub mobility: f32,
}

impl SpeciesGenome {
    pub fn random(rng: &mut StdRng) -> Self {
        Self {
            world_genome: WorldGenome::random(rng),
            defense_investment: rng.gen_range(0.0..1.0),
            resource_efficiency: rng.gen_range(0.3..1.0),
            cooperation_tendency: rng.gen_range(0.0..1.0),
            mobility: rng.gen_range(0.1..1.0),
        }
    }

    pub fn mutate(&mut self, rng: &mut StdRng, rate: f32) {
        self.world_genome.mutate(rng, rate);
        if rng.gen::<f32>() < rate { self.defense_investment = (self.defense_investment + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0); }
        if rng.gen::<f32>() < rate { self.resource_efficiency = (self.resource_efficiency + rng.gen_range(-0.1..0.1)).clamp(0.1, 1.0); }
        if rng.gen::<f32>() < rate { self.cooperation_tendency = (self.cooperation_tendency + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0); }
        if rng.gen::<f32>() < rate { self.mobility = (self.mobility + rng.gen_range(-0.1..0.1)).clamp(0.1, 1.0); }
    }

    pub fn crossover(a: &Self, b: &Self, rng: &mut StdRng) -> Self {
        Self {
            world_genome: WorldGenome::crossover(&a.world_genome, &b.world_genome, rng),
            defense_investment: if rng.gen() { a.defense_investment } else { b.defense_investment },
            resource_efficiency: if rng.gen() { a.resource_efficiency } else { b.resource_efficiency },
            cooperation_tendency: if rng.gen() { a.cooperation_tendency } else { b.cooperation_tendency },
            mobility: if rng.gen() { a.mobility } else { b.mobility },
        }
    }
}

/// Result of a coevolution pairing.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CoevolutionPairingResult {
    pub species_a_fitness: f32,
    pub species_b_fitness: f32,
    pub interaction_strength: f32,
}

/// Evaluate fitness of two species interacting in the same world.
pub fn evaluate_coevolution_pair(
    a: &SpeciesGenome,
    b: &SpeciesGenome,
    mode: CoevolutionMode,
    frames: usize,
    lite: bool,
) -> Result<CoevolutionPairingResult, String> {
    // Run species A's world
    let wr_a = run_single_world(a.world_genome.clone(), frames, lite, FitnessObjective::MaxBiomass, 10)?;
    // Run species B's world
    let wr_b = run_single_world(b.world_genome.clone(), frames, lite, FitnessObjective::MaxBiomass, 10)?;

    let (fit_a, fit_b, interaction) = match mode {
        CoevolutionMode::RedQueen => {
            // Arms race: defense investment reduces opponent fitness
            let attack_a = a.resource_efficiency * a.mobility;
            let attack_b = b.resource_efficiency * b.mobility;
            let def_a = a.defense_investment;
            let def_b = b.defense_investment;
            let net_a = wr_a.fitness * (1.0 + def_a - attack_b * 0.5);
            let net_b = wr_b.fitness * (1.0 + def_b - attack_a * 0.5);
            (net_a, net_b, (attack_a - def_b).abs() + (attack_b - def_a).abs())
        }
        CoevolutionMode::Mutualistic => {
            // Cooperation boosts both
            let synergy = a.cooperation_tendency * b.cooperation_tendency;
            let boost = 1.0 + synergy * 0.5;
            (wr_a.fitness * boost, wr_b.fitness * boost, synergy)
        }
        CoevolutionMode::Competitive => {
            // Resource competition: efficiency determines winner share
            let total = a.resource_efficiency + b.resource_efficiency;
            let share_a = if total > 0.0 { a.resource_efficiency / total } else { 0.5 };
            let share_b = 1.0 - share_a;
            (wr_a.fitness * share_a * 2.0, wr_b.fitness * share_b * 2.0, (share_a - share_b).abs())
        }
    };

    Ok(CoevolutionPairingResult {
        species_a_fitness: fit_a,
        species_b_fitness: fit_b,
        interaction_strength: interaction,
    })
}

/// Run coevolution for multiple generations with two populations.
pub fn evolve_coevolution(
    pop_size: usize,
    generations: usize,
    frames: usize,
    mode: CoevolutionMode,
    lite: bool,
    seed: u64,
) -> Result<CoevolutionResult, String> {
    let start = Instant::now();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut pop_a: Vec<SpeciesGenome> = (0..pop_size).map(|_| SpeciesGenome::random(&mut rng)).collect();
    let mut pop_b: Vec<SpeciesGenome> = (0..pop_size).map(|_| SpeciesGenome::random(&mut rng)).collect();

    let mut history = Vec::with_capacity(generations);

    for gen in 0..generations {
        // Round-robin pairing: each individual in A is paired with one from B
        let mut fit_a = vec![0.0f32; pop_size];
        let mut fit_b = vec![0.0f32; pop_size];

        for i in 0..pop_size {
            let j = (i + gen) % pop_size; // rotating partner assignment
            let result = evaluate_coevolution_pair(&pop_a[i], &pop_b[j], mode, frames, lite)?;
            fit_a[i] += result.species_a_fitness;
            fit_b[j] += result.species_b_fitness;
        }

        let best_a = fit_a.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let best_b = fit_b.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean_a = fit_a.iter().sum::<f32>() / pop_size as f32;
        let mean_b = fit_b.iter().sum::<f32>() / pop_size as f32;

        eprintln!("  CoEvo Gen {} | A: best={:.2} mean={:.2} | B: best={:.2} mean={:.2}",
            gen, best_a, mean_a, best_b, mean_b);

        history.push(CoevolutionGeneration {
            generation: gen,
            best_fitness_a: best_a, mean_fitness_a: mean_a,
            best_fitness_b: best_b, mean_fitness_b: mean_b,
        });

        // Tournament selection + breed each population
        let mut next_a = Vec::with_capacity(pop_size);
        let mut next_b = Vec::with_capacity(pop_size);
        // Elitism: keep best
        let best_a_idx = fit_a.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        let best_b_idx = fit_b.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        next_a.push(pop_a[best_a_idx].clone());
        next_b.push(pop_b[best_b_idx].clone());

        while next_a.len() < pop_size {
            let i1 = rng.gen_range(0..pop_size);
            let i2 = rng.gen_range(0..pop_size);
            let winner = if fit_a[i1] >= fit_a[i2] { i1 } else { i2 };
            let mut child = if rng.gen::<f32>() < 0.7 {
                let other = rng.gen_range(0..pop_size);
                SpeciesGenome::crossover(&pop_a[winner], &pop_a[other], &mut rng)
            } else {
                pop_a[winner].clone()
            };
            child.mutate(&mut rng, 0.15);
            next_a.push(child);
        }
        while next_b.len() < pop_size {
            let i1 = rng.gen_range(0..pop_size);
            let i2 = rng.gen_range(0..pop_size);
            let winner = if fit_b[i1] >= fit_b[i2] { i1 } else { i2 };
            let mut child = if rng.gen::<f32>() < 0.7 {
                let other = rng.gen_range(0..pop_size);
                SpeciesGenome::crossover(&pop_b[winner], &pop_b[other], &mut rng)
            } else {
                pop_b[winner].clone()
            };
            child.mutate(&mut rng, 0.15);
            next_b.push(child);
        }

        pop_a = next_a;
        pop_b = next_b;
    }

    Ok(CoevolutionResult {
        history,
        final_population_a: pop_a,
        final_population_b: pop_b,
        mode,
        total_wall_time_ms: start.elapsed().as_secs_f32() * 1000.0,
    })
}

/// Generation-level coevolution metrics.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CoevolutionGeneration {
    pub generation: usize,
    pub best_fitness_a: f32,
    pub mean_fitness_a: f32,
    pub best_fitness_b: f32,
    pub mean_fitness_b: f32,
}

/// Full coevolution run result.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CoevolutionResult {
    pub history: Vec<CoevolutionGeneration>,
    pub final_population_a: Vec<SpeciesGenome>,
    pub final_population_b: Vec<SpeciesGenome>,
    pub mode: CoevolutionMode,
    pub total_wall_time_ms: f32,
}

// ---------------------------------------------------------------------------
// Genetic Regulatory Network
// ---------------------------------------------------------------------------

/// A gene regulatory network that determines organism phenotype from genotype.
/// Uses a Boolean network model with continuous dynamics (sigmoid activation).
///
/// References:
/// - Kauffman (1969) "Metabolic stability and epigenesis in randomly
///   constructed genetic nets", J. Theoretical Biology
/// - Aldana (2003) "Boolean dynamics of networks with scale-free topology"
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GeneRegulatoryNetwork {
    /// Number of genes in the network.
    pub n_genes: usize,
    /// Interaction matrix: weights[i][j] = effect of gene j on gene i.
    /// Positive = activation, negative = repression, zero = no interaction.
    pub weights: Vec<Vec<f32>>,
    /// Activation thresholds per gene.
    pub thresholds: Vec<f32>,
    /// Current expression levels [0, 1].
    pub expression: Vec<f32>,
    /// Hill coefficient for sigmoid response.
    pub hill_coefficient: f32,
    /// Degradation rate per gene per step.
    pub degradation_rate: f32,
}

impl GeneRegulatoryNetwork {
    /// Create a random GRN with K connections per gene (NK model).
    pub fn random_nk(n_genes: usize, k: usize, rng: &mut StdRng) -> Self {
        let mut weights = vec![vec![0.0f32; n_genes]; n_genes];
        let thresholds: Vec<f32> = (0..n_genes).map(|_| rng.gen_range(-0.5..0.5)).collect();
        let expression: Vec<f32> = (0..n_genes).map(|_| rng.gen_range(0.0..1.0)).collect();

        // Each gene receives K random connections
        for i in 0..n_genes {
            let mut inputs: Vec<usize> = (0..n_genes).collect();
            // Shuffle and take K
            for j in (1..inputs.len()).rev() {
                let swap = rng.gen_range(0..=j);
                inputs.swap(j, swap);
            }
            for &j in inputs.iter().take(k.min(n_genes)) {
                weights[i][j] = rng.gen_range(-1.0..1.0);
            }
        }

        Self {
            n_genes, weights, thresholds, expression,
            hill_coefficient: 2.0,
            degradation_rate: 0.1,
        }
    }

    /// Step the network forward by one time unit.
    pub fn step(&mut self) {
        let mut new_expression = vec![0.0f32; self.n_genes];
        for i in 0..self.n_genes {
            let mut input_sum = 0.0f32;
            for j in 0..self.n_genes {
                input_sum += self.weights[i][j] * self.expression[j];
            }
            input_sum -= self.thresholds[i];
            // Sigmoid activation with Hill coefficient
            let activated = 1.0 / (1.0 + (-self.hill_coefficient * input_sum).exp());
            new_expression[i] = activated * (1.0 - self.degradation_rate)
                + self.expression[i] * self.degradation_rate;
        }
        self.expression = new_expression;
    }

    /// Run to attractor (or max steps) and return the stable expression pattern.
    pub fn find_attractor(&mut self, max_steps: usize) -> Vec<f32> {
        let mut prev = self.expression.clone();
        for _ in 0..max_steps {
            self.step();
            // Check convergence (L2 distance < threshold)
            let dist: f32 = self.expression.iter().zip(prev.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            if dist < 1e-4 {
                return self.expression.clone();
            }
            prev = self.expression.clone();
        }
        self.expression.clone()
    }

    /// Map attractor expression to phenotypic traits.
    /// Returns (growth_rate, stress_tolerance, reproduction_rate, resource_efficiency).
    pub fn attractor_to_phenotype(&mut self) -> GRNPhenotype {
        let expr = self.find_attractor(100);
        let n = expr.len().max(1) as f32;
        // Map gene blocks to traits
        let quarter = (self.n_genes / 4).max(1);
        let growth = expr[..quarter].iter().sum::<f32>() / quarter as f32;
        let stress = expr[quarter..quarter * 2].iter().sum::<f32>() / quarter as f32;
        let repro = if quarter * 3 <= self.n_genes {
            expr[quarter * 2..quarter * 3].iter().sum::<f32>() / quarter as f32
        } else { 0.5 };
        let efficiency = if quarter * 4 <= self.n_genes {
            expr[quarter * 3..].iter().sum::<f32>() / (self.n_genes - quarter * 3).max(1) as f32
        } else { 0.5 };

        GRNPhenotype {
            growth_rate: growth,
            stress_tolerance: stress,
            reproduction_rate: repro,
            resource_efficiency: efficiency,
            attractor: expr,
            network_complexity: self.weights.iter()
                .flat_map(|row| row.iter())
                .filter(|w| w.abs() > 0.01)
                .count() as f32 / n,
        }
    }

    /// Mutate the network.
    pub fn mutate(&mut self, rng: &mut StdRng, rate: f32) {
        for i in 0..self.n_genes {
            for j in 0..self.n_genes {
                if rng.gen::<f32>() < rate * 0.1 {
                    self.weights[i][j] += rng.gen_range(-0.3..0.3);
                    self.weights[i][j] = self.weights[i][j].clamp(-2.0, 2.0);
                }
            }
            if rng.gen::<f32>() < rate {
                self.thresholds[i] += rng.gen_range(-0.2..0.2);
                self.thresholds[i] = self.thresholds[i].clamp(-2.0, 2.0);
            }
        }
    }

    /// Crossover two networks.
    pub fn crossover(a: &Self, b: &Self, rng: &mut StdRng) -> Self {
        let n = a.n_genes;
        let mut child = a.clone();
        let crossover_point = rng.gen_range(0..n);
        for i in crossover_point..n {
            child.weights[i] = b.weights[i].clone();
            child.thresholds[i] = b.thresholds[i];
            child.expression[i] = b.expression[i];
        }
        child
    }
}

/// Phenotype produced by a GRN attractor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GRNPhenotype {
    pub growth_rate: f32,
    pub stress_tolerance: f32,
    pub reproduction_rate: f32,
    pub resource_efficiency: f32,
    pub attractor: Vec<f32>,
    pub network_complexity: f32,
}

/// Evolve GRNs to find networks that produce target phenotypes.
pub fn evolve_grn(
    target_growth: f32,
    target_stress: f32,
    n_genes: usize,
    k_connections: usize,
    pop_size: usize,
    generations: usize,
    seed: u64,
) -> GRNEvolutionResult {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut population: Vec<GeneRegulatoryNetwork> = (0..pop_size)
        .map(|_| GeneRegulatoryNetwork::random_nk(n_genes, k_connections, &mut rng))
        .collect();

    let mut best_fitness = f32::NEG_INFINITY;
    let mut best_network = population[0].clone();
    let mut best_phenotype: Option<GRNPhenotype> = None;

    for _gen in 0..generations {
        let mut fitnesses = Vec::with_capacity(pop_size);
        let mut phenotypes = Vec::with_capacity(pop_size);
        for net in &mut population {
            let pheno = net.attractor_to_phenotype();
            let fit = -((pheno.growth_rate - target_growth).powi(2)
                + (pheno.stress_tolerance - target_stress).powi(2));
            fitnesses.push(fit);
            phenotypes.push(pheno);
        }

        let best_idx = fitnesses.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap().0;
        if fitnesses[best_idx] > best_fitness {
            best_fitness = fitnesses[best_idx];
            best_network = population[best_idx].clone();
            best_phenotype = Some(phenotypes[best_idx].clone());
        }

        // Tournament selection + breed
        let mut next = vec![population[best_idx].clone()]; // elitism
        while next.len() < pop_size {
            let a = rng.gen_range(0..pop_size);
            let b = rng.gen_range(0..pop_size);
            let winner = if fitnesses[a] >= fitnesses[b] { a } else { b };
            let mut child = if rng.gen::<f32>() < 0.7 {
                let other = rng.gen_range(0..pop_size);
                GeneRegulatoryNetwork::crossover(&population[winner], &population[other], &mut rng)
            } else {
                population[winner].clone()
            };
            child.mutate(&mut rng, 0.15);
            next.push(child);
        }
        population = next;
    }

    GRNEvolutionResult {
        best_network,
        best_phenotype: best_phenotype.unwrap(),
        best_fitness,
        generations_run: generations,
    }
}

/// Result of GRN evolution.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GRNEvolutionResult {
    pub best_network: GeneRegulatoryNetwork,
    pub best_phenotype: GRNPhenotype,
    pub best_fitness: f32,
    pub generations_run: usize,
}

// ---------------------------------------------------------------------------
// Ecosystem Health Metrics
// ---------------------------------------------------------------------------

/// Comprehensive ecosystem health assessment following ecological theory.
///
/// References:
/// - Shannon (1948) "A Mathematical Theory of Communication" — diversity index
/// - Odum (1969) "The Strategy of Ecosystem Development" — maturity metrics
/// - Ulanowicz (2004) "Quantifying sustainability" — ascendency
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EcosystemHealthReport {
    /// Shannon-Wiener diversity index H' = -Σ pi ln(pi).
    pub shannon_diversity: f32,
    /// Simpson's diversity D = 1 - Σ pi².
    pub simpson_diversity: f32,
    /// Evenness: H' / ln(S) where S = species richness.
    pub evenness: f32,
    /// Total biomass across all trophic levels.
    pub total_biomass: f32,
    /// Biomass-to-metabolism ratio (P:B ratio proxy).
    pub biomass_metabolism_ratio: f32,
    /// Nutrient cycling efficiency [0, 1].
    pub nutrient_cycling: f32,
    /// Energy flow through ecosystem (sum of all metabolic rates).
    pub total_energy_flow: f32,
    /// Resilience score: how quickly does the system recover from perturbation.
    pub resilience_score: f32,
    /// Stability: inverse coefficient of variation of biomass over time.
    pub stability: f32,
    /// Trophic level count (distinct functional groups present).
    pub trophic_levels: usize,
    /// Overall health score [0, 100].
    pub overall_health: f32,
}

/// Compute ecosystem health from a series of snapshots.
pub fn assess_ecosystem_health(snapshots: &[TerrariumWorldSnapshot]) -> EcosystemHealthReport {
    if snapshots.is_empty() {
        return EcosystemHealthReport {
            shannon_diversity: 0.0, simpson_diversity: 0.0, evenness: 0.0,
            total_biomass: 0.0, biomass_metabolism_ratio: 0.0,
            nutrient_cycling: 0.0, total_energy_flow: 0.0,
            resilience_score: 0.0, stability: 0.0, trophic_levels: 0,
            overall_health: 0.0,
        };
    }

    let latest = snapshots.last().unwrap();

    // Count "species" (functional groups present)
    let mut abundances: Vec<f32> = Vec::new();
    if latest.plants > 0 { abundances.push(latest.total_plant_cells); }
    if latest.flies > 0 { abundances.push(latest.flies as f32 * 10.0); }
    if latest.mean_microbes > 0.001 { abundances.push(latest.mean_microbes * 100.0); }
    if latest.fruits > 0 { abundances.push(latest.fruits as f32); }
    if latest.seeds > 0 { abundances.push(latest.seeds as f32); }
    if latest.mean_symbionts > 0.001 { abundances.push(latest.mean_symbionts * 50.0); }

    let total: f32 = abundances.iter().sum::<f32>().max(1e-10);
    let s = abundances.len();

    // Shannon-Wiener
    let shannon = -abundances.iter()
        .map(|&a| { let p = a / total; if p > 0.0 { p * p.ln() } else { 0.0 } })
        .sum::<f32>();

    // Simpson
    let simpson = 1.0 - abundances.iter()
        .map(|&a| { let p = a / total; p * p })
        .sum::<f32>();

    let evenness = if s > 1 { shannon / (s as f32).ln() } else { 1.0 };

    let total_biomass = latest.total_plant_cells
        + latest.flies as f32 * 10.0
        + latest.mean_microbes * 100.0;

    let total_energy = latest.mean_soil_atp_flux * 100.0
        + latest.avg_fly_energy * latest.flies as f32;

    let biomass_metabolism = if total_energy > 0.0 { total_biomass / total_energy } else { 0.0 };

    // Nutrient cycling proxy: glucose turnover
    let nutrient_cycling = (latest.mean_soil_glucose * 10.0).clamp(0.0, 1.0);

    // Stability: compute CV of biomass over time
    let biomass_series: Vec<f32> = snapshots.iter().map(|s| s.total_plant_cells).collect();
    let mean_biomass = biomass_series.iter().sum::<f32>() / biomass_series.len() as f32;
    let var_biomass = biomass_series.iter().map(|b| (b - mean_biomass).powi(2)).sum::<f32>()
        / biomass_series.len() as f32;
    let cv = if mean_biomass > 0.0 { var_biomass.sqrt() / mean_biomass } else { 1.0 };
    let stability = (1.0 / (cv + 0.1)).clamp(0.0, 10.0);

    // Resilience: look for recovery after dips
    let mut resilience = 0.5f32; // default moderate
    if biomass_series.len() > 10 {
        let mid = biomass_series.len() / 2;
        let first_half_mean = biomass_series[..mid].iter().sum::<f32>() / mid as f32;
        let second_half_mean = biomass_series[mid..].iter().sum::<f32>() / (biomass_series.len() - mid) as f32;
        if first_half_mean > 0.0 {
            resilience = (second_half_mean / first_half_mean).clamp(0.0, 2.0) * 0.5;
        }
    }

    // Trophic levels present
    let mut trophic = 0;
    if latest.plants > 0 { trophic += 1; } // producers
    if latest.mean_microbes > 0.001 { trophic += 1; } // decomposers
    if latest.flies > 0 { trophic += 1; } // consumers
    if latest.mean_symbionts > 0.001 { trophic += 1; } // mutualists

    // Overall health: weighted score
    let health = (shannon * 15.0
        + simpson * 15.0
        + evenness * 10.0
        + stability * 10.0
        + resilience * 15.0
        + nutrient_cycling * 10.0
        + (trophic as f32 / 4.0) * 25.0)
        .clamp(0.0, 100.0);

    EcosystemHealthReport {
        shannon_diversity: shannon,
        simpson_diversity: simpson,
        evenness,
        total_biomass,
        biomass_metabolism_ratio: biomass_metabolism,
        nutrient_cycling,
        total_energy_flow: total_energy,
        resilience_score: resilience,
        stability,
        trophic_levels: trophic,
        overall_health: health,
    }
}

// ---------------------------------------------------------------------------
// World Export / Replay
// ---------------------------------------------------------------------------

/// Serializable world state for export and replay.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WorldExport {
    pub genome: WorldGenome,
    pub snapshots: Vec<TerrariumWorldSnapshot>,
    pub environmental_samples: Option<Vec<EnvironmentalSample>>,
    pub health_report: Option<EcosystemHealthReport>,
    pub metadata: WorldExportMetadata,
}

/// Metadata for exported world states.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WorldExportMetadata {
    pub frames_run: usize,
    pub lite_mode: bool,
    pub seed: u64,
    pub fitness: f32,
    pub wall_time_ms: f32,
    pub version: String,
}

/// Run a world and export full state with periodic snapshots.
pub fn run_and_export(
    genome: WorldGenome,
    frames: usize,
    lite: bool,
    snapshot_interval: usize,
    environment: Option<&EnvironmentalSchedule>,
) -> Result<WorldExport, String> {
    let start = Instant::now();
    let mut world = if lite { genome.build_world_lite()? } else { genome.build_world()? };
    let mut snapshots = Vec::with_capacity(frames / snapshot_interval.max(1) + 1);
    let mut env_samples = environment.map(|s| {
        (EnvironmentalState::new(s.clone(), genome.seed.wrapping_add(7777)),
         Vec::with_capacity(frames))
    });

    for frame in 0..frames {
        if let Some((ref mut env, ref mut samples)) = env_samples {
            let sample = env.sample(world.time_s);
            samples.push(sample);
        }
        world.step_frame()?;
        if frame % snapshot_interval.max(1) == 0 || frame == frames - 1 {
            snapshots.push(world.snapshot());
        }
    }

    let final_snap = world.snapshot();
    let fitness = evaluate_fitness(FitnessObjective::MaxBiomass, &final_snap, &[]);
    let health = if snapshots.len() >= 3 { Some(assess_ecosystem_health(&snapshots)) } else { None };

    Ok(WorldExport {
        genome: genome.clone(),
        snapshots,
        environmental_samples: env_samples.map(|(_, s)| s),
        health_report: health,
        metadata: WorldExportMetadata {
            frames_run: frames,
            lite_mode: lite,
            seed: genome.seed,
            fitness,
            wall_time_ms: start.elapsed().as_secs_f32() * 1000.0,
            version: "0.1.0".to_string(),
        },
    })
}

// ---------------------------------------------------------------------------
// Sparkline & Terminal Dashboard Helpers
// ---------------------------------------------------------------------------

/// Render a sparkline from a data series using Unicode block characters.
pub fn sparkline(data: &[f32], width: usize) -> String {
    if data.is_empty() { return String::new(); }
    let blocks = [' ', '\u{2581}', '\u{2582}', '\u{2583}', '\u{2584}', '\u{2585}', '\u{2586}', '\u{2587}', '\u{2588}'];
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(1e-10);

    // Resample to width
    let mut out = String::with_capacity(width * 4);
    for i in 0..width {
        let idx = i * data.len() / width;
        let idx = idx.min(data.len() - 1);
        let normalized = ((data[idx] - min) / range * 8.0).clamp(0.0, 8.0) as usize;
        out.push(blocks[normalized.min(8)]);
    }
    out
}

/// Format a dashboard line with label, value, and sparkline.
pub fn dashboard_line(label: &str, value: f32, history: &[f32], width: usize) -> String {
    let spark = sparkline(history, width.saturating_sub(30));
    format!("{:<14} {:>8.3} {}", label, value, spark)
}

/// Generate a full ecosystem dashboard from snapshot history.
pub fn ecosystem_dashboard(snapshots: &[TerrariumWorldSnapshot], width: usize) -> String {
    let mut out = String::new();
    let w = width.max(40);

    out.push_str(&format!("{:=<w$}\n", "= oNeura Ecosystem Dashboard ", w = w));

    if snapshots.is_empty() {
        out.push_str("No data yet.\n");
        return out;
    }

    let biomass: Vec<f32> = snapshots.iter().map(|s| s.total_plant_cells).collect();
    let moisture: Vec<f32> = snapshots.iter().map(|s| s.mean_soil_moisture).collect();
    let microbes: Vec<f32> = snapshots.iter().map(|s| s.mean_microbes).collect();
    let co2: Vec<f32> = snapshots.iter().map(|s| s.mean_atmospheric_co2).collect();
    let o2: Vec<f32> = snapshots.iter().map(|s| s.mean_atmospheric_o2).collect();
    let fly_energy: Vec<f32> = snapshots.iter().map(|s| s.avg_fly_energy).collect();
    let flies: Vec<f32> = snapshots.iter().map(|s| s.flies as f32).collect();
    let glucose: Vec<f32> = snapshots.iter().map(|s| s.mean_soil_glucose).collect();

    let latest = snapshots.last().unwrap();

    out.push_str(&format!("{:-<w$}\n", "- Population ", w = w));
    out.push_str(&format!("{}\n", dashboard_line("Biomass", latest.total_plant_cells, &biomass, w)));
    out.push_str(&format!("{}\n", dashboard_line("Flies", latest.flies as f32, &flies, w)));
    out.push_str(&format!("{}\n", dashboard_line("Microbes", latest.mean_microbes, &microbes, w)));

    out.push_str(&format!("{:-<w$}\n", "- Chemistry ", w = w));
    out.push_str(&format!("{}\n", dashboard_line("Moisture", latest.mean_soil_moisture, &moisture, w)));
    out.push_str(&format!("{}\n", dashboard_line("Glucose", latest.mean_soil_glucose, &glucose, w)));
    out.push_str(&format!("{}\n", dashboard_line("CO2", latest.mean_atmospheric_co2, &co2, w)));
    out.push_str(&format!("{}\n", dashboard_line("O2", latest.mean_atmospheric_o2, &o2, w)));

    out.push_str(&format!("{:-<w$}\n", "- Energy ", w = w));
    out.push_str(&format!("{}\n", dashboard_line("FlyEnergy", latest.avg_fly_energy, &fly_energy, w)));

    // Health assessment
    if snapshots.len() >= 3 {
        let health = assess_ecosystem_health(snapshots);
        out.push_str(&format!("{:-<w$}\n", "- Health ", w = w));
        out.push_str(&format!("  Shannon H':  {:.3}  Simpson D: {:.3}  Evenness: {:.3}\n",
            health.shannon_diversity, health.simpson_diversity, health.evenness));
        out.push_str(&format!("  Trophic: {}  Stability: {:.2}  Resilience: {:.2}\n",
            health.trophic_levels, health.stability, health.resilience_score));
        out.push_str(&format!("  Overall: {:.1}/100\n", health.overall_health));
    }

    out.push_str(&format!("{:=<w$}\n", "", w = w));
    out
}

'''

# Read original file
with open(TARGET) as f:
    content = f.read()

# Find the insertion point: just before the test section marker
marker = "// ---------------------------------------------------------------------------\n// Tests\n// ---------------------------------------------------------------------------"
idx = content.find(marker)
if idx < 0:
    print("ERROR: Could not find test section marker")
    exit(1)

# Insert before tests
new_content = content[:idx] + NEW_CODE + "\n" + content[idx:]

# Atomic write
import tempfile, os
with tempfile.NamedTemporaryFile(mode='w', dir=os.path.dirname(TARGET), delete=False, suffix='.rs') as tmp:
    tmp.write(new_content)
    tmp_path = tmp.name
os.replace(tmp_path, TARGET)

print(f"Inserted {len(NEW_CODE)} chars of new feature code before tests section")
print(f"New file size: {len(new_content)} chars")
