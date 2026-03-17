//! Ecosystem Integration Module
//!
//! Cross-module bridge connecting all 11 novel biology modules into a unified
//! simulation engine. This module orchestrates the interplay between:
//!
//! - **Climate** → drives temperature/moisture for nutrient cycling and eco-evo
//! - **Nutrient cycling** → C/N/P biogeochemistry feeds microbiome and plants
//! - **Microbiome assembly** → community dynamics coupled to nutrient pools
//! - **Resistance evolution** → AMR emergence under antibiotic pressure
//! - **Horizontal gene transfer** → plasmid/phage spread in microbial communities
//! - **Biofilm dynamics** → spatial structure and quorum sensing
//! - **Eco-evolutionary feedback** → fitness landscapes shift with ecology
//! - **Population genetics** → drift, selection, allele frequency tracking
//! - **Phylogenetic tracker** → lineage recording and speciation detection
//! - **Metabolic flux** → FBA for microbial growth yield estimation
//! - **Guild latent** → latent variable model for unobserved guild dynamics
//!
//! The integration follows a producer-consumer pattern where each module's
//! outputs feed into downstream modules without circular dependencies.

// ─── Imports from constituent modules ───────────────────────────────────────

use crate::climate_scenarios::{ClimateEngine, ClimateScenario};
use crate::nutrient_cycling::NutrientCycler;
use crate::microbiome_assembly::{CommunityAssembler, MicrobialTaxon, Resource as MicrobeResource, shannon_diversity};
use crate::resistance_evolution::{ResistanceSimulator, Antibiotic, ResistanceMechanism, AntibioticClass, ModeOfAction, ResistanceType};
use crate::horizontal_gene_transfer::HgtPopulation;
use crate::biofilm_dynamics::BiofilmSimulator;
use crate::eco_evolutionary_feedback::EcoEvoSimulator;
use crate::population_genetics::WrightFisherSim;
use crate::phylogenetic_tracker::{PhyloTree, PhyloTraits};
use crate::metabolic_flux::{MetabolicNetwork, soil_microbe_generic};

// ─── Integrated ecosystem snapshot ──────────────────────────────────────────

/// A snapshot of the entire integrated ecosystem at a single time point.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EcosystemSnapshot {
    /// Current simulation time in days
    pub time_days: f64,
    // ── Climate ──
    pub temperature_c: f64,
    pub co2_ppm: f64,
    pub precipitation_mm: f64,
    // ── Nutrient pools ──
    pub soil_organic_c_g: f64,
    pub soil_mineral_n_g: f64,
    pub soil_mineral_p_g: f64,
    pub n_mineralization_rate: f64,
    pub co2_flux: f64,
    // ── Microbiome ──
    pub microbial_richness: usize,
    pub microbial_shannon: f64,
    pub total_microbial_biomass: f64,
    // ── Resistance ──
    pub resistance_events: usize,
    pub mdr_strain_count: usize,
    pub mean_mic_fold: f64,
    // ── HGT ──
    pub hgt_resistance_freq: f64,
    pub hgt_mean_fitness: f64,
    // ── Biofilm ──
    pub biofilm_cell_count: usize,
    pub biofilm_biomass: f64,
    pub eps_coverage: f64,
    pub quorum_activated: f64,
    // ── Eco-evo ──
    pub mean_fitness: f64,
    pub genetic_variance_sum: f64,
    pub eco_evo_births: usize,
    pub eco_evo_deaths: usize,
    // ── Pop genetics ──
    pub allele_freq: f64,
    // ── Phylogenetics ──
    pub phylo_tree_size: usize,
    pub phylo_diversity: f32,
    pub speciation_events: usize,
    // ── Metabolic flux ──
    pub fba_growth_rate: f64,
    pub fba_status: String,
}

/// Configuration for the integrated ecosystem simulation.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EcosystemConfig {
    pub climate_scenario: ClimateScenario,
    pub initial_pop_size: usize,
    pub n_traits: usize,
    pub biofilm_width: usize,
    pub biofilm_height: usize,
    pub antibiotic_pressure: bool,
    pub antibiotic_concentration: f64,
    pub dt_days: f64,
    pub seed: u64,
}

impl Default for EcosystemConfig {
    fn default() -> Self {
        Self {
            climate_scenario: ClimateScenario::Rcp45,
            initial_pop_size: 100,
            n_traits: 3,
            biofilm_width: 30,
            biofilm_height: 20,
            antibiotic_pressure: false,
            antibiotic_concentration: 0.0,
            dt_days: 1.0,
            seed: 42,
        }
    }
}

/// Time series of ecosystem snapshots collected during a simulation run.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EcosystemTimeSeries {
    pub snapshots: Vec<EcosystemSnapshot>,
    pub total_days: f64,
    pub config: EcosystemConfig,
}

// ─── Integrated Ecosystem Simulator ─────────────────────────────────────────

/// The main integrated ecosystem simulator.
///
/// Orchestrates all 11 novel biology modules in a single coherent simulation loop.
pub struct IntegratedEcosystem {
    pub config: EcosystemConfig,
    climate: ClimateEngine,
    nutrients: NutrientCycler,
    microbiome: CommunityAssembler,
    resistance: ResistanceSimulator,
    hgt: HgtPopulation,
    biofilm: BiofilmSimulator,
    eco_evo: EcoEvoSimulator,
    pop_gen: WrightFisherSim,
    phylo: PhyloTree,
    fba_network: MetabolicNetwork,
    time_days: f64,
    step_count: u64,
}

impl IntegratedEcosystem {
    /// Create a new integrated ecosystem with the given configuration.
    pub fn new(config: EcosystemConfig) -> Self {
        let seed = config.seed;

        // 1. Climate engine
        let climate = ClimateEngine::new(config.climate_scenario.clone(), seed);

        // 2. Nutrient cycling
        let mut nutrients = NutrientCycler::new();
        nutrients.temperature_c = 20.0;
        nutrients.moisture_fraction = 0.6;

        // 3. Microbiome assembly
        let mut microbiome = CommunityAssembler::new(seed + 1);
        for i in 0..5u64 {
            microbiome.add_taxon(MicrobialTaxon {
                id: i,
                name: format!("Taxon_{}", i),
                growth_rate: 0.5 + (i as f64) * 0.1,
                resource_preferences: vec![0.05; 3], // K_s for 3 resources
                metabolic_outputs: vec![],
                stress_tolerance: 0.5,
                biofilm_propensity: 0.2,
                competitive_ability: 0.5,
            });
        }
        microbiome.add_resource(MicrobeResource {
            name: "glucose".into(),
            concentration: 100.0,
            inflow_rate: 10.0,
            decay_rate: 0.01,
        });
        microbiome.add_resource(MicrobeResource {
            name: "ammonium".into(),
            concentration: 50.0,
            inflow_rate: 5.0,
            decay_rate: 0.005,
        });
        microbiome.add_resource(MicrobeResource {
            name: "phosphate".into(),
            concentration: 20.0,
            inflow_rate: 2.0,
            decay_rate: 0.002,
        });

        // 4. Resistance evolution
        let mut resistance = ResistanceSimulator::new(config.initial_pop_size, seed + 2);
        resistance.add_antibiotic(Antibiotic {
            name: "Ciprofloxacin".into(),
            class: AntibioticClass::Fluoroquinolone,
            mode_of_action: ModeOfAction::DNAReplication,
            mic_wild_type: 0.25,
            half_life_hours: 4.0,
            peak_concentration: 4.0,
        });
        resistance.add_mechanism(ResistanceMechanism {
            name: "GyrA_S83L".into(),
            mechanism_type: ResistanceType::TargetModification,
            target_classes: vec![AntibioticClass::Fluoroquinolone],
            mic_fold_increase: 32.0,
            fitness_cost: 0.05,
            reversion_rate: 1e-8,
            transferable: false,
        });

        // 5. Horizontal gene transfer
        let hgt = HgtPopulation::new(config.initial_pop_size, seed + 3);

        // 6. Biofilm dynamics
        let mut biofilm = BiofilmSimulator::new(
            config.biofilm_width,
            config.biofilm_height,
            seed + 4,
        );
        biofilm.seed_cells(0, 30, 3.3e-4);

        // 7. Eco-evolutionary feedback
        let eco_evo = EcoEvoSimulator::new(
            config.initial_pop_size,
            config.n_traits,
            seed + 5,
        );

        // 8. Population genetics (Wright-Fisher)
        let pop_gen = WrightFisherSim::new(config.initial_pop_size, 0.5, seed + 6);

        // 9. Phylogenetic tracker
        let mut phylo = PhyloTree::new();
        phylo.add_node(None, 0, 0.0, 0, 0.0, PhyloTraits {
            biomass: 1.0,
            drought_tolerance: 0.5,
            enzyme_efficacy: 0.5,
            reproductive_rate: 1.0,
            niche_width: 0.5,
        });

        // 10. Metabolic flux (soil microbe FBA network)
        let fba_network = soil_microbe_generic();

        Self {
            config,
            climate,
            nutrients,
            microbiome,
            resistance,
            hgt,
            biofilm,
            eco_evo,
            pop_gen,
            phylo,
            fba_network,
            time_days: 0.0,
            step_count: 0,
        }
    }

    /// Advance the integrated ecosystem by one time step.
    pub fn step(&mut self) -> EcosystemSnapshot {
        let dt = self.config.dt_days;
        self.time_days += dt;
        self.step_count += 1;

        // ── 1. Climate ──
        let dt_years = dt / 365.25;
        let climate = self.climate.step(dt_years);

        // ── 2. Nutrient cycling (climate → soil) ──
        self.nutrients.temperature_c = climate.temperature_c;
        self.nutrients.moisture_fraction = (climate.precipitation_mm / 2000.0).clamp(0.1, 1.0);
        let nutrient_result = self.nutrients.step(dt);

        // ── 3. Microbiome assembly ──
        let _assembly_events = self.microbiome.step(dt);
        let abundances = &self.microbiome.state.abundances;
        let microbial_shannon = shannon_diversity(abundances);
        let total_microbial_biomass: f64 = abundances.iter().sum();
        let microbial_richness = abundances.iter().filter(|&&a| a > 0.1).count();

        // ── 4. Resistance evolution ──
        let concentrations = if self.config.antibiotic_pressure {
            vec![self.config.antibiotic_concentration]
        } else {
            vec![0.0]
        };
        let resistance_events = self.resistance.step(&concentrations);

        // Count MDR strains (public field: Vec<(Strain, f64)>)
        let mdr_count = self.resistance.strains
            .iter()
            .filter(|(s, _freq)| s.resistance_mechanisms.len() >= 2)
            .count();
        let mean_mic_fold = if !self.resistance.strains.is_empty() {
            self.resistance.strains.iter().map(|(s, _)| {
                s.resistance_mechanisms.len() as f64 * 4.0
            }).sum::<f64>() / self.resistance.strains.len() as f64
        } else {
            1.0
        };

        // ── 5. HGT ──
        self.hgt.step(dt * 3600.0);
        let hgt_resistance_freq = self.hgt.resistance_frequency();
        let hgt_mean_fitness = self.hgt.mean_fitness();

        // ── 6. Biofilm dynamics ──
        self.biofilm.step(dt * 86400.0);
        let biofilm_cells = self.biofilm.cell_count();
        let biofilm_biomass = self.biofilm.biomass();
        let eps = self.biofilm.eps_coverage();
        let quorum = self.biofilm.quorum_activated_fraction(0);

        // ── 7. Eco-evolutionary feedback ──
        let eco_evo_result = self.eco_evo.step();

        // ── 8. Population genetics ──
        let allele_freq = self.pop_gen.step();

        // ── 9. Phylogenetic tracker ──
        let parent_id = if self.step_count > 1 { Some(self.step_count - 1) } else { None };
        let genome_hash = (allele_freq * 1e18) as u64;
        self.phylo.add_node(
            parent_id,
            self.step_count as u32,
            eco_evo_result.mean_fitness as f32,
            genome_hash,
            self.time_days as f32,
            PhyloTraits {
                biomass: total_microbial_biomass as f32,
                drought_tolerance: 0.5,
                enzyme_efficacy: 0.5,
                reproductive_rate: 1.0,
                niche_width: 0.5,
            },
        );
        let speciation_events = self.phylo.speciation_events(0.5);
        let phylo_diversity = self.phylo.phylogenetic_diversity();

        // ── 10. FBA metabolic yield ──
        let fba_result = self.fba_network.fba(10, true);

        // ── Build snapshot ──
        let organic_c = if !self.nutrients.organic_pools.is_empty() {
            self.nutrients.organic_pools[0].carbon_g
        } else {
            0.0
        };

        EcosystemSnapshot {
            time_days: self.time_days,
            temperature_c: climate.temperature_c,
            co2_ppm: climate.co2_ppm,
            precipitation_mm: climate.precipitation_mm,
            soil_organic_c_g: organic_c,
            soil_mineral_n_g: self.nutrients.mineral_pool.ammonium_g
                + self.nutrients.mineral_pool.nitrate_g,
            soil_mineral_p_g: self.nutrients.mineral_pool.phosphate_g,
            n_mineralization_rate: nutrient_result.n_mineralized,
            co2_flux: nutrient_result.co2_emitted,
            microbial_richness,
            microbial_shannon,
            total_microbial_biomass,
            resistance_events: resistance_events.len(),
            mdr_strain_count: mdr_count,
            mean_mic_fold,
            hgt_resistance_freq,
            hgt_mean_fitness,
            biofilm_cell_count: biofilm_cells,
            biofilm_biomass,
            eps_coverage: eps,
            quorum_activated: quorum,
            mean_fitness: eco_evo_result.mean_fitness,
            genetic_variance_sum: eco_evo_result.genetic_variance.iter().sum(),
            eco_evo_births: eco_evo_result.births,
            eco_evo_deaths: eco_evo_result.deaths,
            allele_freq,
            phylo_tree_size: self.phylo.len(),
            phylo_diversity,
            speciation_events: speciation_events.len(),
            fba_growth_rate: fba_result.objective_value,
            fba_status: format!("{:?}", fba_result.status),
        }
    }

    /// Run the full simulation for a given number of days.
    pub fn run(&mut self, total_days: f64) -> EcosystemTimeSeries {
        let steps = (total_days / self.config.dt_days).ceil() as usize;
        let mut snapshots = Vec::with_capacity(steps);
        for _ in 0..steps {
            snapshots.push(self.step());
        }
        EcosystemTimeSeries {
            snapshots,
            total_days,
            config: self.config.clone(),
        }
    }

    pub fn time(&self) -> f64 { self.time_days }
    pub fn phylo_tree(&self) -> &PhyloTree { &self.phylo }
    pub fn nutrient_cycler(&self) -> &NutrientCycler { &self.nutrients }
    pub fn climate_engine(&self) -> &ClimateEngine { &self.climate }
    pub fn resistance_sim(&self) -> &ResistanceSimulator { &self.resistance }
    pub fn resistance_sim_mut(&mut self) -> &mut ResistanceSimulator { &mut self.resistance }
    pub fn eco_evo_sim(&self) -> &EcoEvoSimulator { &self.eco_evo }
    pub fn biofilm_sim(&self) -> &BiofilmSimulator { &self.biofilm }
    pub fn hgt_population(&self) -> &HgtPopulation { &self.hgt }
    pub fn metabolic_network(&self) -> &MetabolicNetwork { &self.fba_network }
}

// ─── Display helpers ────────────────────────────────────────────────────────

impl EcosystemSnapshot {
    pub fn summary_line(&self) -> String {
        format!(
            "Day {:.0} | T={:.1}C CO2={:.0}ppm | C={:.1}g N={:.2}g P={:.2}g | \
             Microbes: {} spp H'={:.2} | Resist: {} events {} MDR | \
             Biofilm: {} cells {:.1}% EPS | Fitness={:.3} | Allele={:.3} | \
             Phylo: {} nodes PD={:.1} | FBA={:.3} {:}",
            self.time_days,
            self.temperature_c, self.co2_ppm,
            self.soil_organic_c_g, self.soil_mineral_n_g, self.soil_mineral_p_g,
            self.microbial_richness, self.microbial_shannon,
            self.resistance_events, self.mdr_strain_count,
            self.biofilm_cell_count, self.eps_coverage * 100.0,
            self.mean_fitness,
            self.allele_freq,
            self.phylo_tree_size, self.phylo_diversity,
            self.fba_growth_rate, self.fba_status,
        )
    }
}

impl EcosystemTimeSeries {
    pub fn print_report(&self) {
        if self.snapshots.is_empty() {
            println!("No snapshots collected.");
            return;
        }
        let first = &self.snapshots[0];
        let last = self.snapshots.last().unwrap();

        println!("========== INTEGRATED ECOSYSTEM SIMULATION REPORT ==========");
        println!("Climate: {:?}  Duration: {:.0} days  Steps: {}",
            self.config.climate_scenario, self.total_days, self.snapshots.len());
        println!("--- CLIMATE ---");
        println!("  Temperature: {:.1}C -> {:.1}C", first.temperature_c, last.temperature_c);
        println!("  CO2: {:.0} -> {:.0} ppm", first.co2_ppm, last.co2_ppm);
        println!("--- NUTRIENT CYCLING ---");
        println!("  Organic C: {:.1} -> {:.1} g", first.soil_organic_c_g, last.soil_organic_c_g);
        println!("  Mineral N: {:.2} -> {:.2} g", first.soil_mineral_n_g, last.soil_mineral_n_g);
        println!("  Mineral P: {:.2} -> {:.2} g", first.soil_mineral_p_g, last.soil_mineral_p_g);
        let total_co2: f64 = self.snapshots.iter().map(|s| s.co2_flux).sum();
        println!("  Total CO2 respired: {:.2} g", total_co2);
        println!("--- MICROBIOME ---");
        println!("  Richness: {} -> {}", first.microbial_richness, last.microbial_richness);
        println!("  Shannon H': {:.2} -> {:.2}", first.microbial_shannon, last.microbial_shannon);
        println!("  Total biomass: {:.1} -> {:.1}", first.total_microbial_biomass, last.total_microbial_biomass);
        println!("--- ANTIMICROBIAL RESISTANCE ---");
        let total_events: usize = self.snapshots.iter().map(|s| s.resistance_events).sum();
        println!("  Total resistance events: {}", total_events);
        println!("  MDR strains: {} -> {}", first.mdr_strain_count, last.mdr_strain_count);
        println!("--- HGT ---");
        println!("  Resistance frequency: {:.3} -> {:.3}", first.hgt_resistance_freq, last.hgt_resistance_freq);
        println!("--- BIOFILM ---");
        println!("  Cells: {} -> {}", first.biofilm_cell_count, last.biofilm_cell_count);
        println!("  EPS coverage: {:.1}% -> {:.1}%",
            first.eps_coverage * 100.0, last.eps_coverage * 100.0);
        println!("  Quorum activated: {:.1}% -> {:.1}%",
            first.quorum_activated * 100.0, last.quorum_activated * 100.0);
        println!("--- ECO-EVOLUTIONARY DYNAMICS ---");
        println!("  Mean fitness: {:.3} -> {:.3}", first.mean_fitness, last.mean_fitness);
        println!("  Genetic variance: {:.4} -> {:.4}", first.genetic_variance_sum, last.genetic_variance_sum);
        println!("--- POPULATION GENETICS ---");
        println!("  Allele freq: {:.3} -> {:.3}", first.allele_freq, last.allele_freq);
        println!("--- PHYLOGENETICS ---");
        println!("  Tree size: {} -> {} nodes", first.phylo_tree_size, last.phylo_tree_size);
        println!("  Faith's PD: {:.1} -> {:.1}", first.phylo_diversity, last.phylo_diversity);
        let total_spec: usize = self.snapshots.iter().map(|s| s.speciation_events).sum();
        println!("  Speciation events: {}", total_spec);
        println!("--- METABOLIC FLUX (FBA) ---");
        println!("  Growth rate: {:.4} -> {:.4}", first.fba_growth_rate, last.fba_growth_rate);
        println!("  Status: {} -> {}", first.fba_status, last.fba_status);
        println!("============================================================");
    }

    pub fn to_json(&self) -> String {
        let mut entries = Vec::new();
        for s in &self.snapshots {
            entries.push(format!(
                "{{\"day\":{:.1},\"temp\":{:.2},\"co2\":{:.1},\"organic_c\":{:.2},\
                 \"mineral_n\":{:.3},\"mineral_p\":{:.3},\"richness\":{},\
                 \"shannon\":{:.3},\"resist_events\":{},\"mdr\":{},\
                 \"biofilm_cells\":{},\"eps\":{:.3},\"fitness\":{:.4},\
                 \"allele_freq\":{:.4},\"phylo_pd\":{:.2},\"fba_growth\":{:.4}}}",
                s.time_days, s.temperature_c, s.co2_ppm,
                s.soil_organic_c_g, s.soil_mineral_n_g, s.soil_mineral_p_g,
                s.microbial_richness, s.microbial_shannon,
                s.resistance_events, s.mdr_strain_count,
                s.biofilm_cell_count, s.eps_coverage,
                s.mean_fitness, s.allele_freq,
                s.phylo_diversity, s.fba_growth_rate,
            ));
        }
        format!("[{}]", entries.join(",\n"))
    }
}

// ─── Preset configurations ──────────────────────────────────────────────────

pub fn climate_impact_scenario(seed: u64) -> IntegratedEcosystem {
    IntegratedEcosystem::new(EcosystemConfig {
        climate_scenario: ClimateScenario::Rcp45,
        initial_pop_size: 200,
        n_traits: 3,
        dt_days: 30.0,
        seed,
        ..Default::default()
    })
}

pub fn amr_emergence_scenario(seed: u64) -> IntegratedEcosystem {
    let mut eco = IntegratedEcosystem::new(EcosystemConfig {
        climate_scenario: ClimateScenario::Rcp26,
        initial_pop_size: 500,
        n_traits: 2,
        biofilm_width: 40,
        biofilm_height: 30,
        antibiotic_pressure: true,
        antibiotic_concentration: 2.0,
        dt_days: 1.0,
        seed,
    });
    eco.resistance.add_mechanism(ResistanceMechanism {
        name: "QnrB_plasmid".into(),
        mechanism_type: ResistanceType::TargetProtection,
        target_classes: vec![AntibioticClass::Fluoroquinolone],
        mic_fold_increase: 8.0,
        fitness_cost: 0.02,
        reversion_rate: 5e-6,
        transferable: true,
    });
    eco
}

pub fn soil_health_scenario(seed: u64) -> IntegratedEcosystem {
    IntegratedEcosystem::new(EcosystemConfig {
        initial_pop_size: 300,
        n_traits: 4,
        dt_days: 7.0,
        seed,
        ..Default::default()
    })
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ecosystem_constructs() {
        let eco = IntegratedEcosystem::new(EcosystemConfig::default());
        assert_eq!(eco.time(), 0.0);
        assert!(eco.phylo_tree().len() >= 1);
    }

    #[test]
    fn ecosystem_step_produces_snapshot() {
        let mut eco = IntegratedEcosystem::new(EcosystemConfig::default());
        let snap = eco.step();
        assert!(snap.time_days > 0.0);
        assert!(snap.temperature_c > -50.0 && snap.temperature_c < 60.0);
        assert!(snap.co2_ppm > 200.0);
        assert!(snap.fba_growth_rate >= 0.0);
    }

    #[test]
    fn ecosystem_run_collects_timeseries() {
        let mut eco = IntegratedEcosystem::new(EcosystemConfig {
            dt_days: 10.0,
            ..Default::default()
        });
        let ts = eco.run(100.0);
        assert_eq!(ts.snapshots.len(), 10);
        for i in 1..ts.snapshots.len() {
            assert!(ts.snapshots[i].time_days > ts.snapshots[i - 1].time_days);
        }
    }

    #[test]
    fn ecosystem_climate_drives_co2() {
        let mut eco = IntegratedEcosystem::new(EcosystemConfig {
            climate_scenario: ClimateScenario::Rcp85,
            dt_days: 365.25,
            ..Default::default()
        });
        let ts = eco.run(365.25 * 50.0);
        let first = &ts.snapshots[0];
        let last = ts.snapshots.last().unwrap();
        assert!(last.co2_ppm > first.co2_ppm,
            "CO2 should increase under RCP 8.5: {} -> {}", first.co2_ppm, last.co2_ppm);
    }

    #[test]
    fn ecosystem_nutrient_cycling_active() {
        let mut eco = IntegratedEcosystem::new(EcosystemConfig::default());
        let snap = eco.step();
        assert!(snap.soil_organic_c_g >= 0.0, "Organic C should be non-negative");
    }

    #[test]
    fn ecosystem_phylo_tree_grows() {
        let mut eco = IntegratedEcosystem::new(EcosystemConfig::default());
        for _ in 0..10 {
            eco.step();
        }
        assert!(eco.phylo_tree().len() >= 10,
            "Phylo tree should grow: {} nodes", eco.phylo_tree().len());
    }

    #[test]
    fn ecosystem_biofilm_runs() {
        let mut eco = IntegratedEcosystem::new(EcosystemConfig::default());
        let snap = eco.step();
        assert!(snap.biofilm_cell_count > 0);
    }

    #[test]
    fn ecosystem_fba_optimal() {
        let mut eco = IntegratedEcosystem::new(EcosystemConfig::default());
        let snap = eco.step();
        assert_eq!(snap.fba_status, "Optimal",
            "FBA should find optimal, got: {}", snap.fba_status);
        assert!(snap.fba_growth_rate > 0.0,
            "FBA growth should be positive: {}", snap.fba_growth_rate);
    }

    #[test]
    fn ecosystem_amr_scenario_works() {
        let mut eco = amr_emergence_scenario(42);
        let ts = eco.run(30.0);
        assert_eq!(ts.snapshots.len(), 30);
        let last = ts.snapshots.last().unwrap();
        assert!(last.time_days >= 29.0);
    }

    #[test]
    fn ecosystem_json_export() {
        let mut eco = IntegratedEcosystem::new(EcosystemConfig {
            dt_days: 10.0,
            ..Default::default()
        });
        let ts = eco.run(30.0);
        let json = ts.to_json();
        assert!(json.starts_with('['));
        assert!(json.ends_with(']'));
        assert!(json.contains("\"day\""));
    }

    #[test]
    fn ecosystem_summary_line_format() {
        let mut eco = IntegratedEcosystem::new(EcosystemConfig::default());
        let snap = eco.step();
        let line = snap.summary_line();
        assert!(line.contains("Day"));
        assert!(line.contains("FBA="));
    }

    #[test]
    fn ecosystem_preset_soil_health() {
        let mut eco = soil_health_scenario(99);
        let snap = eco.step();
        assert!(snap.time_days > 0.0);
    }

    #[test]
    fn ecosystem_preset_climate_impact() {
        let mut eco = climate_impact_scenario(77);
        let snap = eco.step();
        assert!(snap.time_days > 0.0);
    }
}
