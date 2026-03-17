//! Soil Nutrient Cycling Under Climate Change
//!
//! Demonstrates how warming, altered precipitation, and rising CO2 reshape
//! soil biogeochemistry and microbial community composition over 50+ years.
//!
//! Wires three modules together:
//! - `nutrient_cycling`    -- C/N/P decomposition, nitrification, denitrification, leaching
//! - `climate_scenarios`   -- IPCC RCP 2.6/4.5/8.5 multi-decade projections
//! - `microbiome_assembly` -- Monod-kinetics microbial community dynamics
//!
//! Usage:
//!   soil_nutrient_demo [options]
//!     --climate <RCP>      Climate scenario: rcp26, rcp45, rcp85 (default: rcp45)
//!     --years <N>          Years to simulate (default: 50)
//!     --soil <TYPE>        Soil type: clay, loam, sand (default: loam)
//!     --seed <N>           Random seed (default: 42)
//!     --help               Show help

use oneuro_metal::climate_scenarios::{ClimateEngine, ClimateScenario};
use oneuro_metal::microbiome_assembly::{
    CommunityAssembler, CrossFeedingLink, MicrobialTaxon, Resource,
};
use oneuro_metal::nutrient_cycling::{
    limiting_nutrient, DecompositionStage, LitterCohort, NutrientCycler, OrganicPool,
};

// ---------------------------------------------------------------------------
// Soil presets: clay fraction, initial pools, pH
// ---------------------------------------------------------------------------

struct SoilPreset {
    name: &'static str,
    clay_fraction: f64,
    ph: f64,
    /// Initial soil organic C (g per m^2 top 30 cm, ~representative patches)
    humus_c: f64,
    humus_n: f64,
    humus_p: f64,
    litter_mass: f64,
    litter_cn: f64,
    /// Initial mineral N (NH4+ + NO3-), g
    mineral_n: f64,
    /// Initial mineral P (PO4), g
    mineral_p: f64,
}

fn soil_preset(name: &str) -> SoilPreset {
    match name {
        "clay" => SoilPreset {
            name: "Clay",
            clay_fraction: 0.55,
            ph: 7.0,
            humus_c: 15.0,
            humus_n: 1.20,
            humus_p: 0.15,
            litter_mass: 2.0,
            litter_cn: 20.0,
            mineral_n: 0.80,
            mineral_p: 0.10,
        },
        "sand" => SoilPreset {
            name: "Sand",
            clay_fraction: 0.08,
            ph: 5.8,
            humus_c: 8.0,
            humus_n: 0.50,
            humus_p: 0.06,
            litter_mass: 1.5,
            litter_cn: 30.0,
            mineral_n: 0.30,
            mineral_p: 0.04,
        },
        _ => SoilPreset {
            // loam default
            name: "Loam",
            clay_fraction: 0.25,
            ph: 6.5,
            humus_c: 12.5,
            humus_n: 0.85,
            humus_p: 0.10,
            litter_mass: 1.8,
            litter_cn: 25.0,
            mineral_n: 0.50,
            mineral_p: 0.08,
        },
    }
}

// ---------------------------------------------------------------------------
// Microbial community setup: 5 functional guilds
// ---------------------------------------------------------------------------

/// Build the microbial community assembler with 5 soil functional guilds.
///
/// Resources:
///   0 = organic C (from decomposition)
///   1 = ammonium (NH4+)
///   2 = nitrate (NO3-)
///   3 = phosphate (PO4)
fn build_community(seed: u64) -> CommunityAssembler {
    let mut assembler = CommunityAssembler::new(seed);

    // Resources
    assembler.add_resource(Resource {
        name: "Organic C".into(),
        concentration: 100.0,
        inflow_rate: 5.0,
        decay_rate: 0.01,
    });
    assembler.add_resource(Resource {
        name: "Ammonium".into(),
        concentration: 50.0,
        inflow_rate: 2.0,
        decay_rate: 0.005,
    });
    assembler.add_resource(Resource {
        name: "Nitrate".into(),
        concentration: 30.0,
        inflow_rate: 1.0,
        decay_rate: 0.005,
    });
    assembler.add_resource(Resource {
        name: "Phosphate".into(),
        concentration: 20.0,
        inflow_rate: 0.5,
        decay_rate: 0.002,
    });

    // Guild 0: Decomposers (heterotrophs, break down organic C)
    let decomposers = MicrobialTaxon {
        id: 0,
        name: "Decomposers".into(),
        growth_rate: 0.8,
        resource_preferences: vec![0.5, 0.1, 0.0, 0.05], // primary: organic C
        metabolic_outputs: vec![0.0, 0.3, 0.0, 0.1],     // release NH4+, PO4
        stress_tolerance: 0.6,
        biofilm_propensity: 0.3,
        competitive_ability: 0.7,
    };
    assembler.add_taxon(decomposers);

    // Guild 1: Nitrifiers (NH4+ -> NO3-)
    let nitrifiers = MicrobialTaxon {
        id: 1,
        name: "Nitrifiers".into(),
        growth_rate: 0.3,
        resource_preferences: vec![0.05, 0.8, 0.0, 0.02], // primary: NH4+
        metabolic_outputs: vec![0.0, 0.0, 0.5, 0.0],      // produce NO3-
        stress_tolerance: 0.3,                            // moisture-sensitive
        biofilm_propensity: 0.2,
        competitive_ability: 0.5,
    };
    assembler.add_taxon(nitrifiers);

    // Guild 2: Denitrifiers (NO3- -> N2/N2O, anaerobic microsites)
    let denitrifiers = MicrobialTaxon {
        id: 2,
        name: "Denitrifiers".into(),
        growth_rate: 0.5,
        resource_preferences: vec![0.3, 0.0, 0.6, 0.02], // primary: NO3- + organic C
        metabolic_outputs: vec![0.0, 0.0, 0.0, 0.0],     // gaseous loss (modeled externally)
        stress_tolerance: 0.7,                           // tolerant (anaerobic adapted)
        biofilm_propensity: 0.5,
        competitive_ability: 0.4,
    };
    assembler.add_taxon(denitrifiers);

    // Guild 3: N-fixers (atmospheric N2 -> NH4+)
    let n_fixers = MicrobialTaxon {
        id: 3,
        name: "N-fixers".into(),
        growth_rate: 0.2,
        resource_preferences: vec![0.2, 0.0, 0.0, 0.4], // need P, some C
        metabolic_outputs: vec![0.0, 0.4, 0.0, 0.0],    // produce NH4+
        stress_tolerance: 0.4,
        biofilm_propensity: 0.1,
        competitive_ability: 0.3,
    };
    assembler.add_taxon(n_fixers);

    // Guild 4: Mycorrhizal fungi
    let mycorrhizal = MicrobialTaxon {
        id: 4,
        name: "Mycorrhizal".into(),
        growth_rate: 0.15,
        resource_preferences: vec![0.3, 0.05, 0.0, 0.3], // organic C + PO4
        metabolic_outputs: vec![0.0, 0.05, 0.0, 0.2],    // mobilize P
        stress_tolerance: 0.5,
        biofilm_propensity: 0.8, // hyphal networks
        competitive_ability: 0.6,
    };
    assembler.add_taxon(mycorrhizal);

    // Cross-feeding links
    // Decomposers produce NH4+ (resource 1) -> Nitrifiers consume it
    assembler.add_cross_feeding(CrossFeedingLink {
        producer: 0,
        consumer: 1,
        metabolite_idx: 1,
        transfer_efficiency: 0.7,
    });
    // Nitrifiers produce NO3- (resource 2) -> Denitrifiers consume it
    assembler.add_cross_feeding(CrossFeedingLink {
        producer: 1,
        consumer: 2,
        metabolite_idx: 2,
        transfer_efficiency: 0.6,
    });
    // N-fixers produce NH4+ (resource 1) -> Decomposers can use it
    assembler.add_cross_feeding(CrossFeedingLink {
        producer: 3,
        consumer: 0,
        metabolite_idx: 1,
        transfer_efficiency: 0.5,
    });

    // Inoculate all guilds with starting abundances
    for id in 0..5u64 {
        assembler.immigrate(id, 50.0);
    }

    assembler
}

// ---------------------------------------------------------------------------
// Annual record
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct AnnualRecord {
    year: f64,
    temp_c: f64,
    precip_mm: f64,
    co2_ppm: f64,
    soil_c_kg: f64,
    soil_n_g: f64,
    cumulative_n2o_mg: f64,
    shannon: f64,
    limiting: String,
    abundances: Vec<f64>, // per-guild
    extreme_events: usize,
}

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

fn run_simulation(
    scenario: ClimateScenario,
    years: usize,
    soil_name: &str,
    seed: u64,
) -> Vec<AnnualRecord> {
    let preset = soil_preset(soil_name);
    let start_year = 2025.0;

    // ---- NutrientCycler setup ----
    let mut cycler = NutrientCycler::new();
    cycler.temperature_c = 15.0;
    cycler.moisture_fraction = 0.50;
    cycler.ph = preset.ph;
    cycler.clay_fraction = preset.clay_fraction;

    // Humus pool (stable SOM)
    cycler.add_organic_pool(OrganicPool {
        name: "Humus".into(),
        carbon_g: preset.humus_c,
        nitrogen_g: preset.humus_n,
        phosphorus_g: preset.humus_p,
        decomposition_rate: 0.0001, // very slow, stable
        recalcitrance: 0.7,
    });

    // Microbial biomass pool (fast turnover)
    cycler.add_organic_pool(OrganicPool {
        name: "Microbial biomass".into(),
        carbon_g: preset.humus_c * 0.03,            // ~3% of SOC
        nitrogen_g: preset.humus_c * 0.03 / 8.57,   // C:N ~8.57
        phosphorus_g: preset.humus_c * 0.03 / 60.0, // C:P ~60
        decomposition_rate: 0.02,                   // fast turnover
        recalcitrance: 0.05,
    });

    // Fresh litter
    cycler.add_litter(LitterCohort {
        stage: DecompositionStage::Fresh,
        mass_g: preset.litter_mass,
        cn_ratio: preset.litter_cn,
        lignin_fraction: 0.15,
        age_days: 0.0,
    });

    // Set initial mineral pools
    cycler.mineral_pool.ammonium_g = preset.mineral_n * 0.4;
    cycler.mineral_pool.nitrate_g = preset.mineral_n * 0.6;
    cycler.mineral_pool.phosphate_g = preset.mineral_p;

    // ---- Climate engine ----
    let mut climate = ClimateEngine::new(scenario, seed);

    // ---- Microbial community ----
    let mut community = build_community(seed);

    // ---- Annual simulation loop ----
    let mut records = Vec::with_capacity(years + 1);
    let mut cumulative_n2o_mg = 0.0;

    for yr_offset in 0..=years {
        let sim_year = start_year + yr_offset as f64;

        // Get annual climate state (deterministic for table consistency)
        let climate_state = climate.state_at(sim_year);
        let soil_moisture = climate.soil_moisture_factor(sim_year);

        // Extreme events this year
        let extremes = climate.extreme_events_in_year(sim_year);
        let n_extremes = extremes.len();

        // Update cycler environmental drivers from climate
        cycler.temperature_c = climate_state.temperature_c;
        // Convert soil moisture factor (0-1) to WFPS
        cycler.moisture_fraction = soil_moisture.clamp(0.05, 0.95);

        // Step nutrient cycling 365 days (daily steps)
        let mut annual_n2o = 0.0;
        let mut annual_co2 = 0.0;
        for _day in 0..365 {
            let result = cycler.step(1.0);
            annual_n2o += result.n2o_emitted;
            annual_co2 += result.co2_emitted;
        }

        // Convert N2O from g to mg for the table
        cumulative_n2o_mg += annual_n2o * 1000.0;

        // Feed mineral nutrients into microbiome resource pools.
        // Resource 0 (organic C): proportional to CO2 emitted (decomposition proxy)
        // Resource 1 (NH4+): from mineral pool ammonium
        // Resource 2 (NO3-): from mineral pool nitrate
        // Resource 3 (PO4): from mineral pool phosphate
        let c_supply = annual_co2.max(0.001) * 10.0; // scale up for community dynamics
        let nh4_supply = cycler.mineral_pool.ammonium_g.max(0.001) * 5.0;
        let no3_supply = cycler.mineral_pool.nitrate_g.max(0.001) * 5.0;
        let po4_supply = cycler.mineral_pool.phosphate_g.max(0.001) * 5.0;

        // Set resource concentrations proportional to nutrient availability
        if community.state.resources.len() >= 4 {
            community.state.resources[0] = c_supply;
            community.state.resources[1] = nh4_supply;
            community.state.resources[2] = no3_supply;
            community.state.resources[3] = po4_supply;
        }

        // Apply climate stress to microbial community.
        // Warming boosts decomposers but stresses moisture-sensitive taxa.
        let warming_anomaly = (climate_state.temperature_c - 15.0).max(0.0);
        let drought_stress = climate_state.drought_severity;

        // Perturb moisture-sensitive nitrifiers under drought
        if drought_stress > 0.3 {
            community.perturb_taxon(1, drought_stress * 0.1); // nitrifiers
            community.perturb_taxon(4, drought_stress * 0.05); // mycorrhizal
        }

        // Warming slightly boosts decomposer resources
        if warming_anomaly > 1.0 && community.state.resources.len() > 0 {
            community.perturb_resource(0, 1.0 + warming_anomaly * 0.02);
        }

        // Apply extreme event impacts
        for event in &extremes {
            match event {
                oneuro_metal::climate_scenarios::ExtremeEvent::Drought { severity, .. } => {
                    // Severe drought kills moisture-sensitive taxa
                    community.perturb_taxon(1, severity * 0.2);
                    community.perturb_taxon(3, severity * 0.15);
                    community.perturb_taxon(4, severity * 0.15);
                }
                oneuro_metal::climate_scenarios::ExtremeEvent::Heatwave { .. } => {
                    // Heatwave boosts decomposers, stresses others
                    community.perturb_resource(0, 1.2);
                    community.perturb_taxon(1, 0.1);
                }
                oneuro_metal::climate_scenarios::ExtremeEvent::Flood { magnitude } => {
                    // Flooding creates anaerobic conditions: boosts denitrifiers
                    community.perturb_resource(2, 1.0 + magnitude * 0.5);
                }
                _ => {}
            }
        }

        // Step microbial community (simulate ~1 year in hourly steps)
        // 365 days * 24 hours = 8760 hours; step in chunks for performance
        let hours_per_year = 8760.0;
        let dt_hours = 24.0; // daily resolution for community dynamics
        let steps = (hours_per_year / dt_hours) as usize;
        for _ in 0..steps {
            community.step(dt_hours);
        }

        // Collect metrics
        let diversity = community.diversity();
        let total_soil_c: f64 = cycler.organic_pools.iter().map(|p| p.carbon_g).sum::<f64>()
            + cycler
                .litter_cohorts
                .iter()
                .map(|c| c.mass_g * 0.45)
                .sum::<f64>();
        let total_soil_n = cycler.mineral_pool.ammonium_g
            + cycler.mineral_pool.nitrate_g
            + cycler
                .organic_pools
                .iter()
                .map(|p| p.nitrogen_g)
                .sum::<f64>();

        // Determine limiting nutrient
        let n_avail = cycler.mineral_pool.ammonium_g + cycler.mineral_pool.nitrate_g;
        let p_avail = cycler.mineral_pool.phosphate_g;
        // Use reasonable demand estimates
        let limit = limiting_nutrient(n_avail, p_avail, 0.05, 0.01);
        let limit_str = match limit {
            oneuro_metal::nutrient_cycling::LimitingNutrient::Nitrogen => "N",
            oneuro_metal::nutrient_cycling::LimitingNutrient::Phosphorus => "P",
            oneuro_metal::nutrient_cycling::LimitingNutrient::CoLimited => "Co-limited",
            oneuro_metal::nutrient_cycling::LimitingNutrient::Neither => "Neither",
        };

        // Collect guild abundances
        let abundances: Vec<f64> = community.state.abundances.clone();

        // Add fresh litter each year (annual leaf fall)
        cycler.add_litter(LitterCohort {
            stage: DecompositionStage::Fresh,
            mass_g: preset.litter_mass * 0.8,
            cn_ratio: preset.litter_cn,
            lignin_fraction: 0.15,
            age_days: 0.0,
        });

        records.push(AnnualRecord {
            year: sim_year,
            temp_c: climate_state.temperature_c,
            precip_mm: climate_state.precipitation_mm,
            co2_ppm: climate_state.co2_ppm,
            soil_c_kg: total_soil_c,
            soil_n_g: total_soil_n,
            cumulative_n2o_mg,
            shannon: diversity.shannon,
            limiting: limit_str.to_string(),
            abundances,
            extreme_events: n_extremes,
        });
    }

    records
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

fn scenario_label(name: &str) -> &str {
    match name {
        "rcp26" => "RCP 2.6",
        "rcp45" => "RCP 4.5",
        "rcp85" => "RCP 8.5",
        _ => "RCP 4.5",
    }
}

fn print_results(records: &[AnnualRecord], climate_name: &str, soil_name: &str) {
    let n = records.len();
    if n == 0 {
        eprintln!("No records to display.");
        return;
    }

    eprintln!();
    eprintln!("=== Soil Nutrient Cycling Under Climate Change ===");
    eprintln!(
        "Scenario: {} | Soil: {} | Duration: {} years",
        scenario_label(climate_name),
        soil_name,
        n.saturating_sub(1)
    );
    eprintln!();

    // Header
    eprintln!(
        "{:>6} | {:>7} | {:>8} | {:>8} | {:>10} | {:>9} | {:>9} | {:>7} | {:>3} | {}",
        "Year",
        "Temp(C)",
        "Rain(mm)",
        "CO2(ppm)",
        "Soil C(kg)",
        "Soil N(g)",
        "N2O(mg)",
        "Shannon",
        "Evt",
        "Limiting"
    );
    eprintln!("{}", "-".repeat(105));

    // Print rows at selected intervals for readability
    let intervals: Vec<usize> = {
        let mut v = vec![0]; // year 0
        for i in 1..n {
            let yr_offset = i;
            // Show every 5 years, plus first and last
            if yr_offset % 5 == 0 || i == n - 1 {
                v.push(i);
            }
        }
        v
    };

    for &idx in &intervals {
        let r = &records[idx];
        eprintln!(
            "{:>6.0} | {:>7.1} | {:>8.0} | {:>8.0} | {:>10.3} | {:>9.1} | {:>9.1} | {:>7.2} | {:>3} | {}",
            r.year, r.temp_c, r.precip_mm, r.co2_ppm,
            r.soil_c_kg, r.soil_n_g * 1000.0, r.cumulative_n2o_mg,
            r.shannon, r.extreme_events, r.limiting
        );
    }

    // ---- Summary ----
    let first = &records[0];
    let last = &records[n - 1];

    let c_lost = first.soil_c_kg - last.soil_c_kg;
    let c_pct = if first.soil_c_kg > 0.0 {
        c_lost / first.soil_c_kg * 100.0
    } else {
        0.0
    };

    let shannon_change_pct = if first.shannon > 0.0 {
        (last.shannon - first.shannon) / first.shannon * 100.0
    } else {
        0.0
    };

    // Find year where limitation shifts
    let mut shift_year: Option<f64> = None;
    if n > 1 {
        let initial_limit = &records[0].limiting;
        for r in &records[1..] {
            if r.limiting != *initial_limit && shift_year.is_none() {
                shift_year = Some(r.year);
            }
        }
    }

    // Total extreme events
    let total_extremes: usize = records.iter().map(|r| r.extreme_events).sum();

    eprintln!();
    eprintln!("=== Summary ===");
    eprintln!(
        "Total soil C change: {:.3} kg ({}{:.1}%)",
        c_lost,
        if c_lost >= 0.0 { "-" } else { "+" },
        c_pct.abs()
    );
    eprintln!(
        "Cumulative N2O emissions: {:.1} mg N",
        last.cumulative_n2o_mg
    );
    eprintln!(
        "Microbial diversity change: {}{:.1}% (Shannon {:.2} -> {:.2})",
        if shannon_change_pct >= 0.0 { "+" } else { "" },
        shannon_change_pct,
        first.shannon,
        last.shannon
    );
    if let Some(yr) = shift_year {
        eprintln!(
            "Nutrient limitation shifted: {} -> {} (year {:.0})",
            first.limiting, last.limiting, yr
        );
    } else {
        eprintln!(
            "Nutrient limitation: {} (stable throughout)",
            first.limiting
        );
    }
    eprintln!(
        "Total extreme events over {} years: {}",
        n.saturating_sub(1),
        total_extremes
    );
    eprintln!(
        "Temperature change: {:.1} C -> {:.1} C ({:+.1} C)",
        first.temp_c,
        last.temp_c,
        last.temp_c - first.temp_c
    );

    // ---- Microbial Community Shift ----
    eprintln!();
    eprintln!("=== Microbial Community Shift ===");

    let guild_names = [
        "Decomposers",
        "Nitrifiers",
        "Denitrifiers",
        "N-fixers",
        "Mycorrhizal",
    ];

    // Compute relative abundances
    let first_total: f64 = first.abundances.iter().sum::<f64>().max(1e-12);
    let last_total: f64 = last.abundances.iter().sum::<f64>().max(1e-12);

    eprintln!(
        "{:<14} | {:>8} | {:>8} | {:>10} | {}",
        "Taxa", "Year 0", "Final", "Change", "Driver"
    );
    eprintln!("{}", "-".repeat(65));

    let drivers = [
        "warming boost",
        "moisture stress",
        "anaerobic microsites",
        "P limitation",
        "drought",
    ];

    for (i, name) in guild_names.iter().enumerate() {
        let pct_first = if i < first.abundances.len() {
            first.abundances[i] / first_total * 100.0
        } else {
            0.0
        };
        let pct_last = if i < last.abundances.len() {
            last.abundances[i] / last_total * 100.0
        } else {
            0.0
        };
        let change = pct_last - pct_first;
        let driver = if i < drivers.len() { drivers[i] } else { "" };

        eprintln!(
            "{:<14} | {:>7.1}% | {:>7.1}% | {:>+9.1}% | {}",
            name, pct_first, pct_last, change, driver
        );
    }

    eprintln!();
    eprintln!(
        "Total community biomass: {:.1} -> {:.1} ({:+.1}%)",
        first_total,
        last_total,
        (last_total - first_total) / first_total * 100.0
    );
}

// ---------------------------------------------------------------------------
// CLI and main
// ---------------------------------------------------------------------------

fn print_help() {
    eprintln!("soil_nutrient_demo -- Soil nutrient cycling under climate change");
    eprintln!();
    eprintln!("Simulates C/N/P biogeochemistry coupled with microbial community");
    eprintln!("dynamics under IPCC climate scenarios (RCP 2.6, 4.5, 8.5).");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --climate <RCP>    Climate scenario: rcp26, rcp45, rcp85 (default: rcp45)");
    eprintln!("  --years <N>        Years to simulate (default: 50)");
    eprintln!("  --soil <TYPE>      Soil type: clay, loam, sand (default: loam)");
    eprintln!("  --seed <N>         Random seed (default: 42)");
    eprintln!("  --help             Show this help");
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  soil_nutrient_demo");
    eprintln!("  soil_nutrient_demo --climate rcp85 --years 75 --soil clay");
    eprintln!("  soil_nutrient_demo --climate rcp26 --soil sand --seed 123");
}

fn main() {
    let mut climate_name = "rcp45".to_string();
    let mut years: usize = 50;
    let mut soil_name = "loam".to_string();
    let mut seed: u64 = 42;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--climate" => {
                climate_name = args
                    .get(i + 1)
                    .cloned()
                    .unwrap_or_else(|| climate_name.clone());
                i += 1;
            }
            "--years" => {
                years = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(years);
                i += 1;
            }
            "--soil" => {
                soil_name = args
                    .get(i + 1)
                    .cloned()
                    .unwrap_or_else(|| soil_name.clone());
                i += 1;
            }
            "--seed" => {
                seed = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(seed);
                i += 1;
            }
            "--help" | "-h" => {
                print_help();
                return;
            }
            other => {
                eprintln!("Unknown option: {}", other);
                eprintln!("Use --help for usage information.");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let scenario = match climate_name.as_str() {
        "rcp26" => ClimateScenario::Rcp26,
        "rcp45" => ClimateScenario::Rcp45,
        "rcp85" => ClimateScenario::Rcp85,
        other => {
            eprintln!(
                "Unknown climate scenario '{}'. Use rcp26, rcp45, or rcp85.",
                other
            );
            std::process::exit(1);
        }
    };

    let records = run_simulation(scenario, years, &soil_name, seed);
    print_results(&records, &climate_name, &soil_preset(&soil_name).name);
}
