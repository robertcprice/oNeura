//! Antimicrobial Resistance Simulator
//!
//! Interactive terminal demo wiring together resistance_evolution,
//! horizontal_gene_transfer, and biofilm_dynamics into a single AMR
//! simulation with rich ANSI-colored output.
//!
//! # Usage
//!
//! ```bash
//! cargo build --no-default-features --bin amr_simulator
//! ./target/debug/amr_simulator --protocol cycling --population 1000 --generations 200
//! ```

#[allow(dead_code)]
mod ansi {
    pub const RESET: &str = "\x1b[0m";
    pub const BOLD: &str = "\x1b[1m";
    pub const DIM: &str = "\x1b[2m";
    pub const RED: &str = "\x1b[31m";
    pub const GREEN: &str = "\x1b[32m";
    pub const YELLOW: &str = "\x1b[33m";
    pub const BLUE: &str = "\x1b[34m";
    pub const MAGENTA: &str = "\x1b[35m";
    pub const CYAN: &str = "\x1b[36m";
    pub const WHITE: &str = "\x1b[37m";
    pub const BG_RED: &str = "\x1b[41m";
    pub const BG_GREEN: &str = "\x1b[42m";
    pub const BG_YELLOW: &str = "\x1b[43m";
}
use ansi::*;

use oneuro_metal::resistance_evolution::{
    Antibiotic, AntibioticClass, ModeOfAction, ResistanceEvent, ResistanceMechanism,
    ResistanceSimulator, ResistanceType,
};
use oneuro_metal::horizontal_gene_transfer::{
    GeneticElement, GeneticElementType, HgtPopulation,
};
use oneuro_metal::biofilm_dynamics::{self, BiofilmSimulator};

// ── CLI argument parsing ─────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Config {
    protocol: Protocol,
    population: usize,
    generations: u32,
    antibiotics: usize,
    hgt_rate: f64,
    biofilm: bool,
    seed: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Protocol {
    Cycling,
    Combination,
    Escalation,
}

impl Protocol {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "cycling" => Some(Protocol::Cycling),
            "combination" => Some(Protocol::Combination),
            "escalation" => Some(Protocol::Escalation),
            _ => None,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Protocol::Cycling => "cycling",
            Protocol::Combination => "combination",
            Protocol::Escalation => "escalation",
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            protocol: Protocol::Cycling,
            population: 1000,
            generations: 200,
            antibiotics: 3,
            hgt_rate: 1e-4,
            biofilm: false,
            seed: 42,
        }
    }
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let mut config = Config::default();
    let mut i = 1;

    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            "--protocol" => {
                i += 1;
                if i < args.len() {
                    config.protocol = Protocol::from_str(&args[i]).unwrap_or_else(|| {
                        eprintln!(
                            "{}Error{}: Unknown protocol '{}'. Use cycling, combination, or escalation.",
                            RED, RESET, args[i]
                        );
                        std::process::exit(1);
                    });
                }
            }
            "--population" => {
                i += 1;
                if i < args.len() {
                    config.population = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("{}Error{}: Invalid population size.", RED, RESET);
                        std::process::exit(1);
                    });
                }
            }
            "--generations" => {
                i += 1;
                if i < args.len() {
                    config.generations = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("{}Error{}: Invalid generation count.", RED, RESET);
                        std::process::exit(1);
                    });
                }
            }
            "--antibiotics" => {
                i += 1;
                if i < args.len() {
                    config.antibiotics = args[i].parse::<usize>().unwrap_or_else(|_| {
                        eprintln!("{}Error{}: Invalid antibiotic count.", RED, RESET);
                        std::process::exit(1);
                    });
                    if config.antibiotics < 1 || config.antibiotics > 7 {
                        eprintln!("{}Error{}: Antibiotics must be 1-7.", RED, RESET);
                        std::process::exit(1);
                    }
                }
            }
            "--hgt-rate" => {
                i += 1;
                if i < args.len() {
                    config.hgt_rate = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("{}Error{}: Invalid HGT rate.", RED, RESET);
                        std::process::exit(1);
                    });
                }
            }
            "--biofilm" => {
                config.biofilm = true;
            }
            "--seed" => {
                i += 1;
                if i < args.len() {
                    config.seed = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("{}Error{}: Invalid seed.", RED, RESET);
                        std::process::exit(1);
                    });
                }
            }
            other => {
                eprintln!("{}Warning{}: Unknown argument '{}', ignoring.", YELLOW, RESET, other);
            }
        }
        i += 1;
    }

    config
}

fn print_help() {
    eprintln!(
        "\
{BOLD}{CYAN}Antimicrobial Resistance Simulator{RESET}
{DIM}Models resistance evolution, horizontal gene transfer, and biofilm dynamics{RESET}

{BOLD}USAGE:{RESET}
    amr_simulator [OPTIONS]

{BOLD}OPTIONS:{RESET}
    --protocol <TYPE>    Treatment protocol: cycling, combination, escalation (default: cycling)
    --population <N>     Bacterial population size (default: 1000)
    --generations <N>    Generations to simulate (default: 200)
    --antibiotics <N>    Number of antibiotic classes, 1-7 (default: 3)
    --hgt-rate <F>       Horizontal gene transfer rate (default: 1e-4)
    --biofilm            Enable parallel biofilm simulation
    --seed <N>           Random seed for reproducibility (default: 42)
    --help               Show this help message

{BOLD}PROTOCOLS:{RESET}
    {GREEN}cycling{RESET}       Rotate drugs every 20 generations to limit single-drug resistance
    {YELLOW}combination{RESET}   All drugs simultaneously at reduced dose (50%)
    {RED}escalation{RESET}    Start at 50% dose, escalate to 200% when resistance detected

{BOLD}EXAMPLES:{RESET}
    amr_simulator --protocol cycling --generations 300 --biofilm
    amr_simulator --protocol combination --population 5000 --hgt-rate 1e-3
    amr_simulator --protocol escalation --antibiotics 5 --seed 123
"
    );
}

// ── Antibiotic and mechanism definitions ────────────────────────────

/// Returns (Antibiotic, ResistanceMechanism, clinical_breakpoint) tuples for
/// up to 7 antibiotic classes. Data from EUCAST/CLSI breakpoint tables.
fn antibiotic_library() -> Vec<(Antibiotic, ResistanceMechanism, f64)> {
    vec![
        (
            Antibiotic {
                name: "Ciprofloxacin".into(),
                class: AntibioticClass::Fluoroquinolone,
                mic_wild_type: 0.25,
                half_life_hours: 4.0,
                peak_concentration: 4.0,
                mode_of_action: ModeOfAction::DNAReplication,
            },
            ResistanceMechanism {
                name: "GyrA S83L".into(),
                mechanism_type: ResistanceType::TargetModification,
                mic_fold_increase: 32.0,
                fitness_cost: 0.05,
                reversion_rate: 1e-8,
                transferable: false,
                target_classes: vec![AntibioticClass::Fluoroquinolone],
            },
            1.0, // EUCAST breakpoint for Enterobacterales
        ),
        (
            Antibiotic {
                name: "Ampicillin".into(),
                class: AntibioticClass::BetaLactam,
                mic_wild_type: 2.0,
                half_life_hours: 1.0,
                peak_concentration: 20.0,
                mode_of_action: ModeOfAction::CellWall,
            },
            ResistanceMechanism {
                name: "TEM-1 beta-lactamase".into(),
                mechanism_type: ResistanceType::Enzymatic {
                    enzyme_name: "TEM-1".into(),
                },
                mic_fold_increase: 64.0,
                fitness_cost: 0.03,
                reversion_rate: 1e-7,
                transferable: true,
                target_classes: vec![AntibioticClass::BetaLactam],
            },
            8.0, // EUCAST breakpoint
        ),
        (
            Antibiotic {
                name: "Gentamicin".into(),
                class: AntibioticClass::Aminoglycoside,
                mic_wild_type: 1.0,
                half_life_hours: 2.5,
                peak_concentration: 10.0,
                mode_of_action: ModeOfAction::ProteinSynthesis,
            },
            ResistanceMechanism {
                name: "AAC(6')-Ib".into(),
                mechanism_type: ResistanceType::Enzymatic {
                    enzyme_name: "AAC(6')-Ib acetyltransferase".into(),
                },
                mic_fold_increase: 16.0,
                fitness_cost: 0.04,
                reversion_rate: 1e-7,
                transferable: true,
                target_classes: vec![AntibioticClass::Aminoglycoside],
            },
            4.0, // EUCAST breakpoint
        ),
        (
            Antibiotic {
                name: "Erythromycin".into(),
                class: AntibioticClass::Macrolide,
                mic_wild_type: 0.5,
                half_life_hours: 1.5,
                peak_concentration: 5.0,
                mode_of_action: ModeOfAction::ProteinSynthesis,
            },
            ResistanceMechanism {
                name: "ErmB methylase".into(),
                mechanism_type: ResistanceType::TargetModification,
                mic_fold_increase: 128.0,
                fitness_cost: 0.06,
                reversion_rate: 1e-8,
                transferable: true,
                target_classes: vec![AntibioticClass::Macrolide],
            },
            2.0,
        ),
        (
            Antibiotic {
                name: "Tetracycline".into(),
                class: AntibioticClass::Tetracycline,
                mic_wild_type: 1.0,
                half_life_hours: 8.0,
                peak_concentration: 4.0,
                mode_of_action: ModeOfAction::ProteinSynthesis,
            },
            ResistanceMechanism {
                name: "TetA efflux pump".into(),
                mechanism_type: ResistanceType::Efflux,
                mic_fold_increase: 16.0,
                fitness_cost: 0.07,
                reversion_rate: 1e-7,
                transferable: true,
                target_classes: vec![AntibioticClass::Tetracycline],
            },
            4.0,
        ),
        (
            Antibiotic {
                name: "Vancomycin".into(),
                class: AntibioticClass::Glycopeptide,
                mic_wild_type: 1.0,
                half_life_hours: 6.0,
                peak_concentration: 30.0,
                mode_of_action: ModeOfAction::CellWall,
            },
            ResistanceMechanism {
                name: "VanA operon".into(),
                mechanism_type: ResistanceType::TargetModification,
                mic_fold_increase: 256.0,
                fitness_cost: 0.10,
                reversion_rate: 1e-9,
                transferable: true,
                target_classes: vec![AntibioticClass::Glycopeptide],
            },
            4.0,
        ),
        (
            Antibiotic {
                name: "Colistin".into(),
                class: AntibioticClass::Polymyxin,
                mic_wild_type: 0.5,
                half_life_hours: 3.0,
                peak_concentration: 2.0,
                mode_of_action: ModeOfAction::MembraneDisruption,
            },
            ResistanceMechanism {
                name: "MCR-1 phosphoethanolamine transferase".into(),
                mechanism_type: ResistanceType::TargetModification,
                mic_fold_increase: 8.0,
                fitness_cost: 0.08,
                reversion_rate: 1e-8,
                transferable: true,
                target_classes: vec![AntibioticClass::Polymyxin],
            },
            2.0,
        ),
    ]
}

// ── MIC color formatting ──────────────────────────────────────────────

/// Color-code a MIC value relative to the clinical breakpoint:
///   green  = susceptible (MIC <= breakpoint)
///   yellow = intermediate (MIC <= 4x breakpoint)
///   red    = resistant (MIC > 4x breakpoint)
fn format_mic(mic: f64, breakpoint: f64) -> String {
    let color = if mic <= breakpoint {
        GREEN
    } else if mic <= breakpoint * 4.0 {
        YELLOW
    } else {
        RED
    };
    format!("{}{:>8.2}{}", color, mic, RESET)
}

/// Format a percentage with color thresholds.
fn format_pct(pct: f64) -> String {
    // Clamp to zero to avoid displaying "-0%"
    let pct = if pct.abs() < 0.01 { 0.0 } else { pct };
    let color = if pct < 5.0 {
        GREEN
    } else if pct < 25.0 {
        YELLOW
    } else {
        RED
    };
    format!("{}{:>4.0}%{}", color, pct, RESET)
}

/// Format a fitness value.
fn format_fitness(f: f64) -> String {
    let color = if f > 0.9 {
        GREEN
    } else if f > 0.7 {
        YELLOW
    } else {
        RED
    };
    format!("{}{:>6.3}{}", color, f, RESET)
}

// ── Event formatting ──────────────────────────────────────────────────

/// Convert resistance events to a compact, colored string.
fn format_events(events: &[ResistanceEvent], ab_names: &[String]) -> String {
    if events.is_empty() {
        return String::new();
    }

    let mut parts: Vec<String> = Vec::new();

    for ev in events {
        let s = match ev {
            ResistanceEvent::DeNovoMutation { mechanism_idx, .. } => {
                format!("{}De novo: mech#{}{}", CYAN, mechanism_idx, RESET)
            }
            ResistanceEvent::HorizontalTransfer { mechanism_idx, .. } => {
                format!("{}HGT: mech#{}{}", MAGENTA, mechanism_idx, RESET)
            }
            ResistanceEvent::CompensatoryMutation { .. } => {
                format!("{}Compensatory{}", BLUE, RESET)
            }
            ResistanceEvent::ResistanceLoss { mechanism_idx, .. } => {
                format!("{}Loss: mech#{}{}", DIM, mechanism_idx, RESET)
            }
            ResistanceEvent::MdrAcquired { num_classes, .. } => {
                format!(
                    "{}{}MDR ({}cl)!{}",
                    BG_RED, WHITE, num_classes, RESET
                )
            }
            ResistanceEvent::ClinicalResistance {
                antibiotic_idx, ..
            } => {
                let name = ab_names
                    .get(*antibiotic_idx)
                    .map(|s| s.as_str())
                    .unwrap_or("?");
                format!(
                    "{}{}R: {}!{}",
                    BG_YELLOW, RED, name, RESET
                )
            }
        };
        parts.push(s);
    }

    // Truncate to at most 3 events for table readability
    if parts.len() > 3 {
        let n = parts.len();
        parts.truncate(3);
        parts.push(format!("{}+{} more{}", DIM, n - 3, RESET));
    }

    parts.join(", ")
}

// ── HGT simulation ───────────────────────────────────────────────────

/// Run a parallel HGT simulation and report plasmid spread dynamics.
fn run_hgt_simulation(config: &Config, transferable_mechs: &[(&str, f64)]) {
    eprintln!();
    eprintln!(
        "{}{}=== Horizontal Gene Transfer Dynamics ==={}", BOLD, MAGENTA, RESET
    );

    let mut hgt_pop = HgtPopulation::new(config.population, config.seed + 100);
    hgt_pop.antibiotic_pressure = true;

    // Add plasmid-borne resistance elements for each transferable mechanism
    let mut element_ids: Vec<(u64, String)> = Vec::new();
    for (name, transfer_rate) in transferable_mechs {
        let eid = hgt_pop.add_element(GeneticElement {
            id: 0,
            name: format!("p{}", name),
            element_type: GeneticElementType::Plasmid {
                copy_number: 5,
                incompatibility_group: element_ids.len() as u8,
            },
            fitness_cost: 0.03,
            fitness_benefit: 0.15,
            transfer_rate: *transfer_rate,
            loss_rate: 1e-6,
            size_kb: 80.0,
        });
        // Introduce at low frequency (1% of population)
        hgt_pop.introduce_element(eid, 0.01);
        element_ids.push((eid, name.to_string()));
    }

    // Run 500 steps at dt=0.1 (represents ~50 time units)
    let hgt_steps = 500;
    let dt = 0.1;
    let ts = hgt_pop.run_simulation(hgt_steps, dt);

    eprintln!(
        "  Population: {} | Steps: {} | dt: {}",
        config.population, hgt_steps, dt
    );
    eprintln!();

    // Report element frequency trajectories
    eprintln!(
        "  {:<20} {:>10} {:>10} {:>10}",
        "Plasmid", "Initial", "Mid", "Final"
    );
    eprintln!("  {}", "-".repeat(55));

    for (eid, name) in &element_ids {
        if let Some(freqs) = ts.element_frequencies.get(eid) {
            let initial = freqs.first().copied().unwrap_or(0.0);
            let mid = if freqs.len() > 1 {
                freqs[freqs.len() / 2]
            } else {
                initial
            };
            let final_f = freqs.last().copied().unwrap_or(0.0);

            let color_final = if final_f > 0.5 {
                RED
            } else if final_f > 0.1 {
                YELLOW
            } else {
                GREEN
            };

            eprintln!(
                "  {:<20} {:>9.1}% {:>9.1}% {}{:>9.1}%{}",
                format!("p{}", name),
                initial * 100.0,
                mid * 100.0,
                color_final,
                final_f * 100.0,
                RESET
            );
        }
    }

    // Final population stats
    let final_fitness = ts.mean_fitness.last().copied().unwrap_or(1.0);
    let final_size = ts.population_size.last().copied().unwrap_or(0);
    let resistance_freq = hgt_pop.resistance_frequency();

    eprintln!();
    eprintln!(
        "  Final population: {} | Mean fitness: {:.3} | Resistance freq: {:.1}%",
        final_size,
        final_fitness,
        resistance_freq * 100.0
    );
}

// ── Biofilm simulation ───────────────────────────────────────────────

/// Run a biofilm simulation showing tolerance amplification.
fn run_biofilm_simulation(config: &Config) {
    eprintln!();
    eprintln!(
        "{}{}=== Biofilm Impact on Antibiotic Tolerance ==={}", BOLD, GREEN, RESET
    );
    eprintln!();

    // Use the Pseudomonas preset -- clinically relevant biofilm former
    let mut biofilm = biofilm_dynamics::pseudomonas_biofilm(config.seed + 200);

    // Phase 1: Let biofilm mature (grow for 200 steps)
    let maturation_steps = 200;
    let dt = 10.0; // seconds per step
    let ts = biofilm.run_simulation(maturation_steps, dt);

    let mature_biomass = ts.biomass.last().copied().unwrap_or(0.0);
    let mature_eps = ts.eps_coverage.last().copied().unwrap_or(0.0);
    let mushrooms = biofilm.mushroom_structure_count();

    eprintln!(
        "  {}Biofilm maturation ({} steps x {}s):{}", BOLD, maturation_steps, dt, RESET
    );
    eprintln!(
        "  Cells: {:.0} | EPS coverage: {:.1}% | Mushroom structures: {}",
        mature_biomass,
        mature_eps * 100.0,
        mushrooms
    );

    // Phase 2: Quorum sensing status
    let qa_final = ts.quorum_activation.last().copied().unwrap_or(0.0);
    let qa_color = if qa_final > 0.5 { GREEN } else { YELLOW };
    eprintln!(
        "  Quorum activation: {}{:.1}%{}",
        qa_color,
        qa_final * 100.0,
        RESET
    );

    // Phase 3: Antibiotic challenge -- compare planktonic vs biofilm
    // For planktonic comparison, create a fresh population with no biofilm
    let mut planktonic = BiofilmSimulator::new(50, 30, config.seed + 300);
    planktonic.seed_cells(0, 50, 3.3e-4);
    // Grow planktonic cells briefly (no EPS buildup)
    for _ in 0..10 {
        planktonic.step(dt);
    }

    let test_concentrations = [1.0, 4.0, 16.0, 64.0, 256.0];
    let challenge_dt = 60.0; // 1-minute challenge

    eprintln!();
    eprintln!(
        "  {:<15} {:>12} {:>12} {:>12}",
        "Conc (ug/mL)", "Planktonic", "Biofilm", "Tolerance"
    );
    eprintln!("  {}", "-".repeat(55));

    for &conc in &test_concentrations {
        // Clone both to avoid depleting populations
        let mut p_clone = planktonic.clone();
        let mut b_clone = biofilm.clone();

        let p_before = p_clone.cell_count();
        let b_before = b_clone.cell_count();

        let p_killed = p_clone.antibiotic_challenge(conc, challenge_dt);
        let b_killed = b_clone.antibiotic_challenge(conc, challenge_dt);

        let p_survival = if p_before > 0 {
            ((p_before - p_killed) as f64 / p_before as f64) * 100.0
        } else {
            0.0
        };
        let b_survival = if b_before > 0 {
            ((b_before - b_killed) as f64 / b_before as f64) * 100.0
        } else {
            0.0
        };

        let tolerance_ratio = if p_survival > 0.1 {
            b_survival / p_survival
        } else if b_survival > 0.1 {
            999.0 // effectively infinite
        } else {
            1.0
        };

        let tol_color = if tolerance_ratio > 10.0 {
            RED
        } else if tolerance_ratio > 2.0 {
            YELLOW
        } else {
            GREEN
        };

        eprintln!(
            "  {:>10.1}     {:>9.1}%  {:>9.1}%  {}{:>8.1}x{}",
            conc, p_survival, b_survival, tol_color, tolerance_ratio, RESET
        );
    }

    // Summary
    let species_dist = biofilm.species_distribution();
    let species_str: Vec<String> = species_dist
        .iter()
        .map(|(sp, cnt)| format!("sp{}: {}", sp, cnt))
        .collect();

    eprintln!();
    eprintln!(
        "  Species: {} | Total biomass: {:.0} cells",
        species_str.join(", "),
        biofilm.biomass()
    );
    eprintln!(
        "  {}Biofilm tolerance is the primary barrier to antibiotic eradication{}",
        DIM, RESET
    );
}

// ── Main simulation loop ─────────────────────────────────────────────

fn main() {
    let config = parse_args();

    // ── Header ────────────────────────────────────────────────────────
    eprintln!();
    eprintln!(
        "{}{} ============================================== {}",
        BOLD, CYAN, RESET
    );
    eprintln!(
        "{}{}   Antimicrobial Resistance Simulator          {}",
        BOLD, CYAN, RESET
    );
    eprintln!(
        "{}{} ============================================== {}",
        BOLD, CYAN, RESET
    );
    eprintln!();
    eprintln!(
        "  Protocol: {}{}{} | Population: {}{}{} | Generations: {}{}{}",
        BOLD, config.protocol.name(), RESET,
        BOLD, config.population, RESET,
        BOLD, config.generations, RESET,
    );
    eprintln!(
        "  Antibiotics: {}{}{} | HGT rate: {}{:.0e}{} | Biofilm: {}{}{} | Seed: {}",
        BOLD, config.antibiotics, RESET,
        BOLD, config.hgt_rate, RESET,
        BOLD,
        if config.biofilm { "ON" } else { "OFF" },
        RESET,
        config.seed,
    );
    eprintln!();

    // ── Setup ─────────────────────────────────────────────────────────

    let library = antibiotic_library();
    let n_abs = config.antibiotics.min(library.len());
    let selected: Vec<_> = library.into_iter().take(n_abs).collect();

    let mut sim = ResistanceSimulator::new(config.population, config.seed);
    sim.hgt_rate = config.hgt_rate;

    // Clinical breakpoints for each antibiotic (for coloring)
    let mut breakpoints: Vec<f64> = Vec::new();
    let mut ab_names: Vec<String> = Vec::new();
    for (ab, mech, bp) in &selected {
        let _ab_idx = sim.add_antibiotic(ab.clone());
        let _mech_idx = sim.add_mechanism(mech.clone());
        breakpoints.push(*bp);
        ab_names.push(ab.name.clone());
    }

    // Collect transferable mechanism names for HGT simulation
    // (done separately to avoid borrow issues with selected)
    let transferable_info: Vec<(String, f64)> = selected
        .iter()
        .filter(|(_, mech, _)| mech.transferable)
        .map(|(_, mech, _)| (mech.name.clone(), config.hgt_rate))
        .collect();

    // ── Print table header ────────────────────────────────────────────

    // Build dynamic header based on number of antibiotics
    let mut header = format!(" {:<4} |", "Gen");
    let mut separator = format!("------|");
    for name in &ab_names {
        // Truncate to 8 chars
        let short: String = name.chars().take(8).collect();
        header.push_str(&format!(" {:>8} |", short));
        separator.push_str("----------|");
    }
    header.push_str(&format!(
        " {:>6} | {:>4} | {:>7} | {}",
        "Drug", "MDR%", "Fitness", "Events"
    ));
    separator.push_str("--------|------|---------|--------");

    eprintln!("{}{}{}", BOLD, header, RESET);
    eprintln!("{}", separator);

    // ── Simulation loop ───────────────────────────────────────────────

    // Track which drugs are active each generation and clinical resistance milestones
    let mut clinical_resistance_gen: Vec<Option<u32>> = vec![None; n_abs];
    let mut mdr_emergence_gen: Option<u32> = None;
    let mut xdr_detected = false;

    // Build protocol schedule
    let cycle_duration = 20u32; // generations per drug in cycling
    let base_concentrations: Vec<f64> = selected
        .iter()
        .map(|(ab, _, _)| ab.peak_concentration * 0.5) // therapeutic trough ~50% of Cmax
        .collect();

    // Escalation state
    let mut escalation_multiplier = 0.5_f64;
    let mut escalation_drug_idx = 0usize;

    for gen in 0..config.generations {
        // ── Determine concentrations for this generation ─────────────

        let mut concentrations = vec![0.0_f64; n_abs];
        let active_drug_name: String;

        match config.protocol {
            Protocol::Cycling => {
                // Rotate drugs every cycle_duration generations
                let phase = (gen / cycle_duration) as usize % n_abs;
                concentrations[phase] = base_concentrations[phase];
                active_drug_name = ab_names[phase].chars().take(6).collect();
            }
            Protocol::Combination => {
                // All drugs at 50% dose simultaneously
                for i in 0..n_abs {
                    concentrations[i] = base_concentrations[i] * 0.5;
                }
                active_drug_name = "ALL".into();
            }
            Protocol::Escalation => {
                // Start with one drug at 50%, escalate on resistance detection
                concentrations[escalation_drug_idx] =
                    base_concentrations[escalation_drug_idx] * escalation_multiplier;
                active_drug_name = format!(
                    "{}x{:.1}",
                    ab_names[escalation_drug_idx].chars().take(4).collect::<String>(),
                    escalation_multiplier
                );
            }
        }

        // ── Step the simulation ───────────────────────────────────────

        let events = sim.step(&concentrations);

        // ── Compute observables ───────────────────────────────────────

        // Population-weighted MIC for each antibiotic
        let mics: Vec<f64> = (0..n_abs)
            .map(|ab_idx| {
                sim.strains
                    .iter()
                    .map(|(strain, freq)| freq * sim.strain_mic(strain, ab_idx))
                    .sum()
            })
            .collect();

        // MDR percentage: fraction of population that is MDR
        let mdr_fraction: f64 = sim
            .strains
            .iter()
            .filter(|(strain, _)| sim.mdr_profile(strain).is_mdr)
            .map(|(_, freq)| freq)
            .sum();
        let mdr_pct = mdr_fraction * 100.0;

        // Mean population fitness
        let mean_fitness = sim.population_mean_fitness(&concentrations);

        // Track clinical resistance milestones
        for i in 0..n_abs {
            if clinical_resistance_gen[i].is_none() && mics[i] > breakpoints[i] * 4.0 {
                clinical_resistance_gen[i] = Some(gen);
            }
        }
        if mdr_emergence_gen.is_none() && mdr_pct > 1.0 {
            mdr_emergence_gen = Some(gen);
        }

        // Check for XDR
        let xdr_now: bool = sim
            .strains
            .iter()
            .any(|(strain, freq)| *freq > 0.001 && sim.mdr_profile(strain).is_xdr);
        if xdr_now && !xdr_detected {
            xdr_detected = true;
        }

        // Escalation protocol logic: increase dose or switch drug when
        // the current drug's MIC crosses the breakpoint
        if config.protocol == Protocol::Escalation && gen > 0 && gen % 10 == 0 {
            let current_mic = mics[escalation_drug_idx];
            let current_bp = breakpoints[escalation_drug_idx];
            if current_mic > current_bp {
                if escalation_multiplier < 2.0 {
                    escalation_multiplier += 0.25;
                } else {
                    // Switch to next drug
                    escalation_drug_idx = (escalation_drug_idx + 1) % n_abs;
                    escalation_multiplier = 0.5;
                }
            }
        }

        // ── Format events ────────────────────────────────────────────

        let event_str = format_events(&events, &ab_names);

        // Only print at key generations (every 10, or when events occur,
        // or first/last generation)
        let print_row = gen == 0
            || gen == config.generations - 1
            || gen % 10 == 0
            || !events.is_empty();

        if print_row {
            let mut row = format!(" {:>4} |", gen);
            for i in 0..n_abs {
                row.push_str(&format!(" {} |", format_mic(mics[i], breakpoints[i])));
            }
            row.push_str(&format!(
                " {:>6} | {} | {} | {}",
                active_drug_name,
                format_pct(mdr_pct),
                format_fitness(mean_fitness),
                event_str,
            ));
            eprintln!("{}", row);
        }
    }

    // ── Results summary ───────────────────────────────────────────────

    eprintln!();
    eprintln!(
        "{}{}=== Results ==={}", BOLD, CYAN, RESET
    );

    // Time to clinical resistance for each antibiotic
    for i in 0..n_abs {
        let gen_str = match clinical_resistance_gen[i] {
            Some(g) => format!("{}Generation {}{}", RED, g, RESET),
            None => format!("{}Not reached{}", GREEN, RESET),
        };
        eprintln!("  Time to clinical resistance ({}): {}", ab_names[i], gen_str);
    }

    // MDR emergence
    match mdr_emergence_gen {
        Some(g) => eprintln!(
            "  MDR emergence (>1% prevalence): {}Generation {}{}",
            RED, g, RESET
        ),
        None => eprintln!(
            "  MDR emergence: {}Not detected{}",
            GREEN, RESET
        ),
    }

    if xdr_detected {
        eprintln!(
            "  {}{}XDR strain detected!{}",
            BG_RED, WHITE, RESET
        );
    }

    // Mutant selection windows
    eprintln!();
    eprintln!(
        "{}Mutant Selection Windows:{}", BOLD, RESET
    );
    for i in 0..n_abs {
        let (msw_lo, msw_hi) = sim.mutant_selection_window(i);
        eprintln!(
            "  {}: [{:.2}, {:.2}] ug/mL",
            ab_names[i], msw_lo, msw_hi
        );
    }

    // Fitness valley crossings
    let valleys = sim.detect_valley_crossings();
    if !valleys.is_empty() {
        eprintln!();
        eprintln!(
            "{}Fitness Valley Crossings:{}", BOLD, RESET
        );
        for v in &valleys {
            eprintln!(
                "  Gen {}: {} mutations, intermediate fitness {:.3} -> final {:.3} ({} gens to cross)",
                v.generation,
                v.num_mutations,
                v.intermediate_fitness,
                v.final_fitness,
                v.crossing_time,
            );
        }
    }

    // Predicted resistance times (analytical)
    eprintln!();
    eprintln!(
        "{}Analytical Resistance Predictions (deterministic approx.):{}", BOLD, RESET
    );
    for i in 0..n_abs {
        let conc = base_concentrations[i];
        match sim.predict_resistance_time(i, conc) {
            Some(t) => eprintln!(
                "  {} at {:.1} ug/mL: ~{} generations",
                ab_names[i], conc, t
            ),
            None => eprintln!(
                "  {} at {:.1} ug/mL: resistance not predicted at this concentration",
                ab_names[i], conc
            ),
        }
    }

    // ── HGT simulation ────────────────────────────────────────────────

    // Build the transferable mechanism references from the stored info
    let transferable_refs: Vec<(&str, f64)> = transferable_info
        .iter()
        .map(|(name, rate)| (name.as_str(), *rate))
        .collect();

    if !transferable_refs.is_empty() {
        run_hgt_simulation(&config, &transferable_refs);
    }

    // ── Biofilm simulation ────────────────────────────────────────────

    if config.biofilm {
        run_biofilm_simulation(&config);
    }

    // ── Final footer ──────────────────────────────────────────────────

    eprintln!();
    eprintln!(
        "{}{} ============================================== {}",
        BOLD, DIM, RESET
    );
    eprintln!(
        "{}  Simulation complete. {} generations, {} antibiotics.  {}",
        DIM, config.generations, n_abs, RESET
    );
    eprintln!(
        "{}  Reference: Regoes et al. 2004, Drlica 2003, Frost et al. 2005  {}",
        DIM, RESET
    );
    eprintln!(
        "{}{} ============================================== {}",
        BOLD, DIM, RESET
    );
    eprintln!();
}
