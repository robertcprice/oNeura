//! # Evolution Lab: Eco-Evolutionary Dynamics Demo
//!
//! Wires `eco_evolutionary_feedback`, `population_genetics`, and
//! `phylogenetic_tracker` into an interactive evolutionary dynamics
//! demonstration with four simulation modes.
//!
//! ## Modes
//!
//! - **adaptation**: Gradual environmental shift, track trait evolution toward a
//!   new fitness optimum.
//! - **rescue**: Lethal environmental shift at a configurable generation, track
//!   whether the population survives via evolutionary rescue.
//! - **speciation**: Disruptive selection with two fitness peaks, track lineage
//!   divergence and cluster assignments.
//! - **drift**: Neutral Wright-Fisher alongside eco-evo, compare allele
//!   frequency trajectories to neutral expectations.
//!
//! ## Usage
//!
//! ```text
//! cargo build --no-default-features --bin evolution_lab
//! ./target/debug/evolution_lab --mode adaptation --pop-size 500 --generations 200
//! ```

use oneuro_metal::eco_evolutionary_feedback::{
    EcoEvoSimulator, EnvironmentalShift, FitnessLandscape,
    price_equation,
};
use oneuro_metal::phylogenetic_tracker::{PhyloTraits, PhyloTree};
use oneuro_metal::population_genetics::WrightFisherSim;

// ─────────────────────────────────────────────────────────────────────────────
// CLI parsing
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Config {
    mode: Mode,
    pop_size: usize,
    generations: usize,
    n_traits: usize,
    shift_type: ShiftType,
    shift_gen: u32,
    seed: u64,
    export_newick: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Mode {
    Adaptation,
    Rescue,
    Speciation,
    Drift,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ShiftType {
    None,
    Gradual,
    Sudden,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            mode: Mode::Adaptation,
            pop_size: 500,
            generations: 200,
            n_traits: 3,
            shift_type: ShiftType::Gradual,
            shift_gen: 50,
            seed: 42,
            export_newick: false,
        }
    }
}

fn print_help() {
    eprintln!(
        "\
evolution_lab -- Eco-Evolutionary Dynamics Demo

USAGE:
    evolution_lab [OPTIONS]

OPTIONS:
    --mode <MODE>        Mode: adaptation, rescue, speciation, drift (default: adaptation)
    --pop-size <N>       Population size (default: 500)
    --generations <N>    Generations to simulate (default: 200)
    --traits <N>         Number of quantitative traits (default: 3)
    --shift <TYPE>       Environmental shift: none, gradual, sudden (default: gradual)
    --shift-gen <N>      Generation when shift occurs (default: 50)
    --seed <N>           Random seed (default: 42)
    --newick             Export Newick tree at end
    --help               Show this help

MODES:
    adaptation   Gradual environmental shift, track trait adaptation toward new optimum
    rescue       Lethal shift at shift-gen, track population crash and recovery
    speciation   Disruptive selection with two fitness peaks, track lineage divergence
    drift        Neutral evolution, compare to Wright-Fisher expectations"
    );
}

fn parse_args() -> Result<Config, String> {
    let args: Vec<String> = std::env::args().collect();
    let mut cfg = Config::default();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            "--mode" => {
                i += 1;
                if i >= args.len() {
                    return Err("--mode requires a value".into());
                }
                cfg.mode = match args[i].as_str() {
                    "adaptation" => Mode::Adaptation,
                    "rescue" => Mode::Rescue,
                    "speciation" => Mode::Speciation,
                    "drift" => Mode::Drift,
                    other => return Err(format!("Unknown mode: {other}")),
                };
            }
            "--pop-size" => {
                i += 1;
                if i >= args.len() {
                    return Err("--pop-size requires a value".into());
                }
                cfg.pop_size = args[i]
                    .parse()
                    .map_err(|_| format!("Invalid pop-size: {}", args[i]))?;
            }
            "--generations" => {
                i += 1;
                if i >= args.len() {
                    return Err("--generations requires a value".into());
                }
                cfg.generations = args[i]
                    .parse()
                    .map_err(|_| format!("Invalid generations: {}", args[i]))?;
            }
            "--traits" => {
                i += 1;
                if i >= args.len() {
                    return Err("--traits requires a value".into());
                }
                cfg.n_traits = args[i]
                    .parse()
                    .map_err(|_| format!("Invalid traits: {}", args[i]))?;
            }
            "--shift" => {
                i += 1;
                if i >= args.len() {
                    return Err("--shift requires a value".into());
                }
                cfg.shift_type = match args[i].as_str() {
                    "none" => ShiftType::None,
                    "gradual" => ShiftType::Gradual,
                    "sudden" => ShiftType::Sudden,
                    other => return Err(format!("Unknown shift type: {other}")),
                };
            }
            "--shift-gen" => {
                i += 1;
                if i >= args.len() {
                    return Err("--shift-gen requires a value".into());
                }
                cfg.shift_gen = args[i]
                    .parse()
                    .map_err(|_| format!("Invalid shift-gen: {}", args[i]))?;
            }
            "--seed" => {
                i += 1;
                if i >= args.len() {
                    return Err("--seed requires a value".into());
                }
                cfg.seed = args[i]
                    .parse()
                    .map_err(|_| format!("Invalid seed: {}", args[i]))?;
            }
            "--newick" => {
                cfg.export_newick = true;
            }
            other => {
                return Err(format!("Unknown argument: {other}"));
            }
        }
        i += 1;
    }

    // Validate
    if cfg.pop_size < 4 {
        return Err("Population size must be >= 4".into());
    }
    if cfg.generations == 0 {
        return Err("Generations must be > 0".into());
    }
    if cfg.n_traits == 0 {
        return Err("Traits must be > 0".into());
    }

    Ok(cfg)
}

// ─────────────────────────────────────────────────────────────────────────────
// ANSI helpers
// ─────────────────────────────────────────────────────────────────────────────

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const RED: &str = "\x1b[31m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const CYAN: &str = "\x1b[36m";
const MAGENTA: &str = "\x1b[35m";
const WHITE: &str = "\x1b[37m";
const BG_RED: &str = "\x1b[41m";
const BG_GREEN: &str = "\x1b[42m";

// ─────────────────────────────────────────────────────────────────────────────
// Formatting helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Format trait means and standard deviations as "[0.50+/-0.30, ...]".
fn format_traits(means: &[f64], variances: &[f64]) -> String {
    let max_display = means.len().min(4);
    let parts: Vec<String> = (0..max_display)
        .map(|i| {
            let sd = if i < variances.len() {
                variances[i].sqrt()
            } else {
                0.0
            };
            format!("{:.2}+/-{:.2}", means[i], sd)
        })
        .collect();
    let suffix = if means.len() > 4 { ", ..." } else { "" };
    format!("[{}{}]", parts.join(", "), suffix)
}

/// Display a float compactly.
fn fmt_f(val: f64, width: usize, prec: usize) -> String {
    format!("{:>width$.prec$}", val, width = width, prec = prec)
}

// ─────────────────────────────────────────────────────────────────────────────
// Inline xorshift64 for genome hash generation (same algorithm as modules)
// ─────────────────────────────────────────────────────────────────────────────

fn xorshift64(state: &mut u64) -> u64 {
    if *state == 0 {
        *state = 0xDEAD_BEEF_CAFE_BABE;
    }
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ─────────────────────────────────────────────────────────────────────────────
// Core simulation
// ─────────────────────────────────────────────────────────────────────────────

fn run_simulation(cfg: &Config) {
    let mode_str = match cfg.mode {
        Mode::Adaptation => "adaptation",
        Mode::Rescue => "rescue",
        Mode::Speciation => "speciation",
        Mode::Drift => "drift",
    };
    let shift_str = match cfg.shift_type {
        ShiftType::None => "none",
        ShiftType::Gradual => "gradual",
        ShiftType::Sudden => "sudden",
    };

    // Header
    eprintln!();
    eprintln!(
        "{BOLD}{CYAN}=== Evolution Lab: Eco-Evolutionary Dynamics ==={RESET}"
    );
    eprintln!(
        "Mode: {BOLD}{}{RESET} | Pop: {BOLD}{}{RESET} | Traits: {BOLD}{}{RESET} | \
         Shift: {BOLD}{} at gen {}{RESET} | Seed: {}",
        mode_str, cfg.pop_size, cfg.n_traits, shift_str, cfg.shift_gen, cfg.seed
    );
    eprintln!();

    // Set up the eco-evo simulator
    let mut sim = EcoEvoSimulator::new(cfg.pop_size, cfg.n_traits, cfg.seed);

    // Configure landscape based on mode
    match cfg.mode {
        Mode::Speciation => {
            // Disruptive selection: wide landscape so bimodal peaks emerge
            // We use ruggedness to create a fitness valley in the middle
            let landscape = FitnessLandscape {
                dimensions: cfg.n_traits,
                optima: vec![0.0; cfg.n_traits],
                widths: vec![2.0; cfg.n_traits],
                ruggedness: 0.8,
            };
            sim.set_landscape(landscape);
            // Higher mutation rate to encourage divergence
            sim.mutation_rate = 0.05;
            sim.mutation_effect_size = 0.15;
        }
        Mode::Drift => {
            // Neutral landscape: very wide Gaussian, everything is about equal fitness
            let landscape = FitnessLandscape {
                dimensions: cfg.n_traits,
                optima: vec![0.0; cfg.n_traits],
                widths: vec![100.0; cfg.n_traits],
                ruggedness: 0.0,
            };
            sim.set_landscape(landscape);
            sim.mutation_rate = 0.01;
            sim.mutation_effect_size = 0.05;
        }
        Mode::Rescue => {
            // Tight landscape: organisms must be near optimum to survive
            let landscape = FitnessLandscape {
                dimensions: cfg.n_traits,
                optima: vec![0.0; cfg.n_traits],
                widths: vec![0.5; cfg.n_traits],
                ruggedness: 0.0,
            };
            sim.set_landscape(landscape);
            sim.mutation_rate = 0.03;
            sim.mutation_effect_size = 0.08;
        }
        Mode::Adaptation => {
            // Default landscape with moderate selection
            sim.mutation_rate = 0.02;
            sim.mutation_effect_size = 0.06;
        }
    }

    // Environmental shifts: build the shifts for each trait axis
    let target_optimum: Vec<f64> = match cfg.mode {
        Mode::Rescue => vec![1.5; cfg.n_traits], // large lethal shift
        Mode::Adaptation => {
            // Moderate shift to 0.8 on each axis
            (0..cfg.n_traits).map(|_| 0.8).collect()
        }
        _ => vec![0.0; cfg.n_traits], // No shift for speciation/drift
    };

    let shift_speed = match cfg.shift_type {
        ShiftType::None => 0.0,
        ShiftType::Gradual => 0.05,
        ShiftType::Sudden => 1.0,
    };

    let shifts: Vec<EnvironmentalShift> = (0..cfg.n_traits)
        .map(|axis| EnvironmentalShift {
            generation: cfg.shift_gen,
            trait_axis: axis,
            new_optimum: target_optimum[axis],
            shift_speed,
        })
        .collect();

    // Set up phylogenetic tree
    let mut tree = PhyloTree::new();
    let mut hash_rng = cfg.seed.wrapping_add(0xBEEF);

    // Register initial population in the tree
    let mut org_to_node: Vec<(u64, u64)> = Vec::new(); // (organism_id, phylo_node_id)
    for org in &sim.population {
        let genome_hash = xorshift64(&mut hash_rng);
        let traits = PhyloTraits {
            biomass: 1.0,
            drought_tolerance: 0.5,
            enzyme_efficacy: 0.5,
            reproductive_rate: org.fitness as f32,
            niche_width: 0.5,
        };
        let node_id = tree.add_node(None, 0, org.fitness as f32, genome_hash, 0.0, traits);
        org_to_node.push((org.id, node_id));
    }

    // Wright-Fisher comparison for drift mode
    let mut wf_sim = if cfg.mode == Mode::Drift {
        let mut wf = WrightFisherSim::new(cfg.pop_size, 0.5, cfg.seed.wrapping_add(1));
        wf.set_selection(0.0, 0.5); // neutral
        wf.set_mutation(1e-4, 1e-4);
        Some(wf)
    } else {
        None
    };

    // History for Price equation (need previous generation snapshot)
    let mut prev_population_snapshot = sim.population.clone();

    // Allele frequency history for drift mode (tracking first trait as proxy)
    let mut wf_freq_history: Vec<f64> = Vec::new();
    let mut eco_freq_history: Vec<f64> = Vec::new();

    // Speciation tracking
    let mut speciation_count = 0u32;

    // Table header
    eprintln!(
        "{BOLD}{DIM} Gen  | Mean Fit | Gen Var | Pop Size | Traits [mean+/-sd]          | Events{RESET}"
    );
    eprintln!(
        "{DIM}------|----------|---------|----------|-----------------------------|--------------------------{RESET}"
    );

    // Report interval: scale with generation count
    let report_interval = match cfg.generations {
        0..=50 => 1,
        51..=200 => 5,
        201..=500 => 10,
        _ => 25,
    };

    // Accumulate Price equation components
    let mut total_selection = 0.0_f64;
    let mut total_transmission = 0.0_f64;
    let mut price_samples = 0u32;

    // ── Main loop ──────────────────────────────────────────────────────────
    for gen in 0..cfg.generations {
        // Apply environmental shifts
        if gen as u32 >= cfg.shift_gen && cfg.shift_type != ShiftType::None {
            match cfg.mode {
                Mode::Adaptation | Mode::Rescue => {
                    for shift in &shifts {
                        sim.apply_environmental_shift(shift);
                    }
                }
                Mode::Speciation => {
                    // Disruptive selection: shift two peaks apart
                    if gen as u32 == cfg.shift_gen {
                        // Create bimodal landscape by shifting optimum off-center
                        // and increasing ruggedness
                        let landscape = FitnessLandscape {
                            dimensions: cfg.n_traits,
                            optima: vec![0.5; cfg.n_traits],
                            widths: vec![1.5; cfg.n_traits],
                            ruggedness: 0.95,
                        };
                        sim.set_landscape(landscape);
                    }
                }
                Mode::Drift => {} // no shift
            }
        }

        // Step the eco-evo simulation
        let result = sim.step();

        // Compute Price equation for this generation
        if !prev_population_snapshot.is_empty() && !sim.population.is_empty() {
            for t in 0..cfg.n_traits.min(1) {
                let (sel, trans) = price_equation(
                    &prev_population_snapshot,
                    &sim.population,
                    t,
                );
                total_selection += sel;
                total_transmission += trans;
                price_samples += 1;
            }
        }
        prev_population_snapshot = sim.population.clone();

        // Step Wright-Fisher (drift mode)
        if let Some(ref mut wf) = wf_sim {
            let freq = wf.step();
            wf_freq_history.push(freq);
            // Use mean of first trait as eco-evo "allele frequency" proxy
            let eco_freq = if !sim.population.is_empty() {
                let mean_t0: f64 = sim.population.iter().map(|o| o.traits[0]).sum::<f64>()
                    / sim.population.len() as f64;
                // Normalize to [0,1] range: sigmoid transform
                1.0 / (1.0 + (-mean_t0 * 5.0).exp())
            } else {
                0.5
            };
            eco_freq_history.push(eco_freq);
        }

        // Register new organisms in phylogenetic tree (sample to keep tree manageable)
        let sample_rate = if cfg.pop_size > 200 {
            200.0 / cfg.pop_size as f64
        } else {
            1.0
        };
        let mut new_org_to_node: Vec<(u64, u64)> = Vec::new();
        for org in &sim.population {
            let should_sample = {
                let mut h = hash_rng;
                let r = xorshift64(&mut h) as f64 / u64::MAX as f64;
                hash_rng = h;
                r < sample_rate
            };
            if should_sample {
                let genome_hash = xorshift64(&mut hash_rng);
                // Find parent node (use lineage_id matching -- first match from prev gen)
                let parent_node = org_to_node
                    .iter()
                    .find(|(_, _)| true) // fallback: first available
                    .map(|(_, nid)| *nid);
                let traits = PhyloTraits {
                    biomass: 1.0,
                    drought_tolerance: 0.5,
                    enzyme_efficacy: 0.5,
                    reproductive_rate: org.fitness as f32,
                    niche_width: 0.5,
                };
                let node_id = tree.add_node(
                    parent_node,
                    result.generation,
                    org.fitness as f32,
                    genome_hash,
                    result.generation as f32,
                    traits,
                );
                new_org_to_node.push((org.id, node_id));
            }
        }

        // Mark old generation as dead in phylo tree
        for &(_, nid) in &org_to_node {
            tree.mark_dead(nid, result.generation as f32);
        }
        org_to_node = new_org_to_node;

        // Detect speciation (speciation mode)
        if cfg.mode == Mode::Speciation && gen as u32 > cfg.shift_gen {
            let events = sim.detect_speciation(1.0);
            speciation_count += events.len() as u32;
            if !events.is_empty() && gen % report_interval == 0 {
                sim.assign_lineages_by_clustering();
            }
        }

        // Report
        if gen % report_interval == 0 || gen == cfg.generations - 1 {
            let total_gv: f64 = result.genetic_variance.iter().sum();
            let trait_str = format_traits(&result.trait_means, &result.genetic_variance);

            // Build event string
            let event = build_event_string(cfg, gen, &result, &sim);

            // Color fitness based on value
            let fit_color = if result.mean_fitness > 0.85 {
                GREEN
            } else if result.mean_fitness > 0.5 {
                YELLOW
            } else {
                RED
            };

            // Color pop size for rescue mode
            let pop_color = if cfg.mode == Mode::Rescue {
                if result.population_size < cfg.pop_size / 2 {
                    RED
                } else if result.population_size < cfg.pop_size * 3 / 4 {
                    YELLOW
                } else {
                    GREEN
                }
            } else {
                WHITE
            };

            eprintln!(
                " {BOLD}{:>4}{RESET} | {fit_color}{}{RESET} | {} | {pop_color}{:>8}{RESET} | {} | {}",
                gen,
                fmt_f(result.mean_fitness, 7, 4),
                fmt_f(total_gv, 7, 4),
                result.population_size,
                trait_str,
                event,
            );
        }

        // Check for extinction
        if sim.population.is_empty() {
            eprintln!();
            eprintln!(
                "{BOLD}{BG_RED} EXTINCTION at generation {} {RESET}",
                gen
            );
            eprintln!(
                "Population could not survive the environmental shift."
            );
            break;
        }
    }

    eprintln!();

    // ── Population Genetics Summary ────────────────────────────────────────
    eprintln!(
        "{BOLD}{CYAN}=== Population Genetics Summary ==={RESET}"
    );

    // Price equation
    let avg_sel = if price_samples > 0 {
        total_selection / price_samples as f64
    } else {
        0.0
    };
    let avg_trans = if price_samples > 0 {
        total_transmission / price_samples as f64
    } else {
        0.0
    };
    eprintln!(
        "Price equation (avg/gen): Selection = {GREEN}{:.4}{RESET}, Transmission = {YELLOW}{:.4}{RESET}",
        avg_sel, avg_trans
    );

    // Breeder's equation
    if !sim.population.is_empty() {
        let selection_gradient: Vec<f64> = sim
            .trait_means()
            .iter()
            .map(|m| {
                // Selection differential: distance from current mean to optimum
                let opt = sim.landscape.optima.get(0).copied().unwrap_or(0.0);
                opt - m
            })
            .collect();
        let response = sim.breeder_equation_response(&selection_gradient);
        let gv = sim.genetic_variance();
        let total_va: f64 = gv.iter().sum();
        let total_vp = total_va + 0.1 * total_va + 1e-10;
        let h2 = total_va / total_vp;
        let s_mag: f64 = selection_gradient
            .iter()
            .map(|s| s * s)
            .sum::<f64>()
            .sqrt();
        let r_mag: f64 = response.iter().map(|r| r * r).sum::<f64>().sqrt();

        eprintln!(
            "Breeder's equation: h2 = {:.3}, S = {:.3}, R = {:.3}",
            h2, s_mag, r_mag
        );

        // Genetic variance status
        let var_maintained = total_va > 0.001;
        let var_status = if var_maintained {
            format!("{GREEN}Yes{RESET} (mutation-selection balance)")
        } else {
            format!("{RED}No{RESET} (depleted by selection)")
        };
        eprintln!("Genetic variance maintained: {}", var_status);
    }

    // Effective population size (rough estimate from rescue tracker)
    let rescue = sim.evolutionary_rescue_status();
    if rescue.population_size_history.len() >= 2 {
        // Harmonic mean of population sizes as rough Ne proxy
        let harmonic_sum: f64 = rescue
            .population_size_history
            .iter()
            .filter(|&&s| s > 0)
            .map(|&s| 1.0 / s as f64)
            .sum();
        let n_valid = rescue
            .population_size_history
            .iter()
            .filter(|&&s| s > 0)
            .count();
        let ne = if harmonic_sum > 0.0 && n_valid > 0 {
            n_valid as f64 / harmonic_sum
        } else {
            cfg.pop_size as f64
        };
        let ne_ratio = ne / cfg.pop_size as f64;
        eprintln!(
            "Effective population size: Ne = {BOLD}{:.0}{RESET} (Ne/N = {:.2})",
            ne, ne_ratio
        );
    }

    // Rescue mode specific output
    if cfg.mode == Mode::Rescue {
        eprintln!();
        eprintln!(
            "{BOLD}{CYAN}=== Evolutionary Rescue Analysis ==={RESET}"
        );
        if rescue.rescue_detected {
            eprintln!(
                "{BG_GREEN}{BOLD} RESCUE SUCCESSFUL {RESET} at generation {}",
                rescue.rescue_generation.unwrap_or(0)
            );
            if let Some(decline_gen) = rescue.decline_start {
                let rescue_gen = rescue.rescue_generation.unwrap_or(decline_gen);
                let lag = rescue_gen.saturating_sub(decline_gen);
                eprintln!("  Decline began: gen {}", decline_gen);
                eprintln!("  Recovery at:   gen {}", rescue_gen);
                eprintln!("  Rescue lag:    {} generations", lag);
            }
        } else if rescue.is_rescue_in_progress() {
            eprintln!(
                "{BG_RED}{BOLD} RESCUE IN PROGRESS {RESET} -- population still declining"
            );
            if let Some(dg) = rescue.decline_start {
                eprintln!("  Decline began: gen {}", dg);
            }
        } else if sim.population.is_empty() {
            eprintln!(
                "{BG_RED}{BOLD} RESCUE FAILED -- EXTINCTION {RESET}"
            );
        } else {
            eprintln!("No population decline detected (shift may have been too mild).");
        }

        // Show genetic variance trajectory
        if rescue.genetic_variance_history.len() > 3 {
            let early_gv: f64 = rescue.genetic_variance_history[..3].iter().sum::<f64>() / 3.0;
            let n = rescue.genetic_variance_history.len();
            let late_gv: f64 = rescue.genetic_variance_history[n - 3..].iter().sum::<f64>() / 3.0;
            let ratio = if early_gv > 1e-10 {
                late_gv / early_gv
            } else {
                0.0
            };
            eprintln!(
                "  Genetic variance: early={:.4}, late={:.4} (ratio={:.2}x)",
                early_gv, late_gv, ratio
            );
        }
    }

    // Drift mode specific output
    if cfg.mode == Mode::Drift {
        eprintln!();
        eprintln!(
            "{BOLD}{CYAN}=== Drift Analysis: Wright-Fisher Comparison ==={RESET}"
        );
        if let Some(ref wf) = wf_sim {
            eprintln!(
                "Wright-Fisher final allele freq: {:.4}",
                wf.allele_freq
            );
            eprintln!(
                "Expected heterozygosity (WF):    {:.4}",
                wf.expected_heterozygosity()
            );

            // Compare variance of allele frequency changes
            if wf_freq_history.len() > 10 && eco_freq_history.len() > 10 {
                let wf_var = compute_variance(&wf_freq_history);
                let eco_var = compute_variance(&eco_freq_history);
                eprintln!(
                    "WF allele freq variance:         {:.6}",
                    wf_var
                );
                eprintln!(
                    "Eco-evo trait freq variance:      {:.6}",
                    eco_var
                );
                let ratio = if wf_var > 1e-10 {
                    eco_var / wf_var
                } else {
                    0.0
                };
                let interp = if ratio > 2.0 {
                    format!("{RED}Eco-evo has excess variance (selection/structure){RESET}")
                } else if ratio < 0.5 {
                    format!("{GREEN}Eco-evo has reduced variance (stabilizing selection){RESET}")
                } else {
                    format!("{YELLOW}Consistent with near-neutral dynamics{RESET}")
                };
                eprintln!("Variance ratio (eco/WF):         {:.3} -- {}", ratio, interp);
            }

            // Fixation probability comparison
            let p_fix_neutral = WrightFisherSim::fixation_probability(cfg.pop_size, 0.0);
            let p_fix_beneficial = WrightFisherSim::fixation_probability(cfg.pop_size, 0.01);
            eprintln!();
            eprintln!(
                "Kimura fixation probability:  neutral = {:.6}, s=0.01 = {:.4}",
                p_fix_neutral, p_fix_beneficial
            );
        }
    }

    // Speciation mode specific output
    if cfg.mode == Mode::Speciation {
        eprintln!();
        eprintln!(
            "{BOLD}{CYAN}=== Speciation Analysis ==={RESET}"
        );

        // Final speciation check
        let final_events = sim.detect_speciation(0.8);
        let displacement = sim.detect_character_displacement();

        eprintln!(
            "Speciation events detected:    {}",
            speciation_count + final_events.len() as u32
        );
        for (i, ev) in final_events.iter().enumerate() {
            eprintln!(
                "  Event {}: gen {}, divergence = {:.3}, isolation = {:.3}, lineages ({}, {})",
                i + 1,
                ev.generation,
                ev.trait_divergence,
                ev.reproductive_isolation,
                ev.daughter_lineages.0,
                ev.daughter_lineages.1,
            );
        }

        if !displacement.is_empty() {
            eprintln!();
            eprintln!(
                "{BOLD}Character Displacement:{RESET}"
            );
            for cd in &displacement {
                let status = if cd.displacement_ratio > 1.2 {
                    format!("{GREEN}Displacement detected{RESET}")
                } else if cd.displacement_ratio > 1.0 {
                    format!("{YELLOW}Weak displacement{RESET}")
                } else {
                    format!("{DIM}No displacement{RESET}")
                };
                eprintln!(
                    "  Species ({}, {}), trait {}: initial={:.3}, final={:.3}, ratio={:.2} -- {}",
                    cd.species_pair.0,
                    cd.species_pair.1,
                    cd.trait_axis,
                    cd.initial_distance,
                    cd.final_distance,
                    cd.displacement_ratio,
                    status,
                );
            }
        }

        // Cluster analysis
        sim.assign_lineages_by_clustering();
        let mut lineage_counts: Vec<(u64, usize)> = Vec::new();
        for org in &sim.population {
            if let Some(entry) = lineage_counts.iter_mut().find(|(id, _)| *id == org.lineage_id) {
                entry.1 += 1;
            } else {
                lineage_counts.push((org.lineage_id, 1));
            }
        }
        lineage_counts.sort_by(|a, b| b.1.cmp(&a.1));

        eprintln!();
        eprintln!("{BOLD}Cluster Assignments:{RESET}");
        for (lin_id, count) in &lineage_counts {
            let pct = 100.0 * *count as f64 / sim.population.len() as f64;
            let bar_len = (pct / 2.0) as usize;
            let bar: String = std::iter::repeat('#').take(bar_len).collect();
            eprintln!(
                "  Lineage {:>3}: {:>4} organisms ({:>5.1}%) {CYAN}{}{RESET}",
                lin_id, count, pct, bar
            );
        }
    }

    // ── Phylogenetic Summary ───────────────────────────────────────────────
    eprintln!();
    eprintln!(
        "{BOLD}{CYAN}=== Phylogenetic Summary ==={RESET}"
    );
    let living = tree.living_nodes();
    let phylo_spec_events = tree.speciation_events(0.3);
    let faith_pd = tree.phylogenetic_diversity();

    eprintln!(
        "Total lineages:     {BOLD}{}{RESET}",
        tree.len()
    );
    eprintln!(
        "Living lineages:    {BOLD}{}{RESET}",
        living.len()
    );
    eprintln!(
        "Tree depth:         {} generations",
        tree.tree_depth()
    );
    eprintln!(
        "Mean branch length: {:.2}",
        tree.mean_branch_length()
    );
    eprintln!(
        "Speciation events:  {} (hash divergence > 0.3)",
        phylo_spec_events.len()
    );
    eprintln!("Faith's PD:         {:.1}", faith_pd);

    // Diversity over time (sample 5 time points)
    if cfg.generations > 5 {
        let step = cfg.generations as f32 / 5.0;
        let bins: Vec<f32> = (0..5).map(|i| (i as f32 + 1.0) * step).collect();
        let diversity = tree.diversity_over_time(&bins);
        eprint!("Diversity trend:    ");
        for (t, count) in &diversity {
            eprint!("t={:.0}:{} ", t, count);
        }
        eprintln!();
    }

    // Newick export
    if cfg.export_newick {
        eprintln!();
        eprintln!(
            "{BOLD}{CYAN}=== Newick Tree ==={RESET}"
        );
        let newick = tree.to_newick();
        if newick.len() > 2000 {
            eprintln!(
                "{}... (truncated, {} total chars)",
                &newick[..2000],
                newick.len()
            );
        } else {
            eprintln!("{}", newick);
        }
    }

    // ── Final summary ──────────────────────────────────────────────────────
    eprintln!();
    eprintln!(
        "{BOLD}{MAGENTA}=== Simulation Complete ==={RESET}"
    );
    eprintln!(
        "Ran {} generations of {} mode with {} organisms and {} traits.",
        cfg.generations, mode_str, cfg.pop_size, cfg.n_traits
    );
    if !sim.population.is_empty() {
        eprintln!(
            "Final population: {} organisms, mean fitness: {:.4}",
            sim.population.len(),
            sim.mean_fitness()
        );
    }
    eprintln!();
}

/// Build a context-appropriate event string for the current generation.
fn build_event_string(
    cfg: &Config,
    gen: usize,
    result: &oneuro_metal::eco_evolutionary_feedback::EcoEvoStepResult,
    sim: &EcoEvoSimulator,
) -> String {
    let mut events: Vec<String> = Vec::new();

    // Environmental shift announcement
    if gen as u32 == cfg.shift_gen && cfg.shift_type != ShiftType::None {
        match cfg.mode {
            Mode::Rescue => {
                events.push(format!(
                    "{BG_RED}{BOLD} LETHAL SHIFT! {RESET} Optimum -> [1.5, ...]"
                ));
            }
            Mode::Adaptation => {
                events.push(format!(
                    "{YELLOW}SHIFT!{RESET} Optimum -> [0.8, ...]"
                ));
            }
            Mode::Speciation => {
                events.push(format!(
                    "{MAGENTA}Disruptive selection activated{RESET}"
                ));
            }
            Mode::Drift => {}
        }
    }

    // Rescue mode events
    if cfg.mode == Mode::Rescue {
        let rescue = sim.evolutionary_rescue_status();
        if rescue.is_rescue_in_progress() {
            let pop_pct = 100.0 * result.population_size as f64 / cfg.pop_size as f64;
            events.push(format!("{RED}Declining ({:.0}%){RESET}", pop_pct));
        }
        if rescue.rescue_detected && rescue.rescue_generation == Some(result.generation) {
            events.push(format!("{BG_GREEN}{BOLD} RESCUED! {RESET}"));
        }
    }

    // Fitness-based events
    if result.mean_fitness > 0.90 && gen > cfg.shift_gen as usize + 10 {
        events.push(format!("{GREEN}Near optimum{RESET}"));
    } else if result.mean_fitness > 0.80 && gen > cfg.shift_gen as usize + 5 {
        events.push(format!("{GREEN}Adapting...{RESET}"));
    } else if gen < cfg.shift_gen as usize && result.mean_fitness > 0.80 {
        events.push(format!("{DIM}Stabilizing selection{RESET}"));
    }

    // Speciation events
    if cfg.mode == Mode::Speciation && gen as u32 > cfg.shift_gen {
        let spec_events = sim.detect_speciation(1.0);
        if !spec_events.is_empty() {
            events.push(format!(
                "{MAGENTA}Speciation! ({}){RESET}",
                spec_events.len()
            ));
        }
    }

    // Low genetic variance warning
    let total_gv: f64 = result.genetic_variance.iter().sum();
    if total_gv < 0.001 && gen > 10 {
        events.push(format!("{RED}Low genetic variance{RESET}"));
    }

    if events.is_empty() {
        String::new()
    } else {
        events.join(", ")
    }
}

/// Compute variance of a slice of f64 values.
fn compute_variance(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    data.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

fn main() {
    let cfg = match parse_args() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: {e}");
            eprintln!("Run with --help for usage information.");
            std::process::exit(1);
        }
    };
    run_simulation(&cfg);
}
