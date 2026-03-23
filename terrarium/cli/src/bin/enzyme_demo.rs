//! Multiscale Enzyme-in-Soil Demo
//!
//! Demonstrates placing a real atomistic enzyme (tripeptide Gly-Ala-Gly)
//! into the terrarium soil grid as an AtomisticProbe. The enzyme's MD
//! dynamics run alongside the ecology simulation, and evolution optimizes
//! enzyme placement for maximum ecosystem benefit.
//!
//! Usage:
//!   enzyme_demo --population 8 --generations 5 --frames 100 --lite

use oneura_core::atomistic_chemistry::{
    BondOrder, EmbeddedMolecule, MoleculeGraph, PeriodicElement,
};
use oneura_core::terrarium::{
    evolve, telemetry_from_result, EvolutionConfig, FitnessConfig, FitnessObjective,
    GenomeConstraints, SearchStrategy, WorldGenome,
};

/// Build a tripeptide Gly-Ala-Gly as an EmbeddedMolecule.
///
/// This is a minimal 27-atom peptide that demonstrates the full
/// atomistic pipeline: element types, covalent bonds, and 3D coordinates.
fn build_tripeptide_gag() -> EmbeddedMolecule {
    let mut g = MoleculeGraph::new("Gly-Ala-Gly");

    // Glycine-1: N-CA-C(=O)
    let n1 = g.add_element(PeriodicElement::N); // 0
    let ca1 = g.add_element(PeriodicElement::C); // 1
    let c1 = g.add_element(PeriodicElement::C); // 2
    let o1 = g.add_element(PeriodicElement::O); // 3
    let h1a = g.add_element(PeriodicElement::H); // 4 NH2
    let h1b = g.add_element(PeriodicElement::H); // 5 NH2
    let h1c = g.add_element(PeriodicElement::H); // 6 CA-H
    let h1d = g.add_element(PeriodicElement::H); // 7 CA-H

    // Alanine: N-CA(CH3)-C(=O)
    let n2 = g.add_element(PeriodicElement::N); // 8
    let ca2 = g.add_element(PeriodicElement::C); // 9
    let c2 = g.add_element(PeriodicElement::C); // 10
    let o2 = g.add_element(PeriodicElement::O); // 11
    let cb2 = g.add_element(PeriodicElement::C); // 12 CH3
    let h2a = g.add_element(PeriodicElement::H); // 13 NH
    let h2b = g.add_element(PeriodicElement::H); // 14 CA-H
    let h2c = g.add_element(PeriodicElement::H); // 15 CB-H
    let h2d = g.add_element(PeriodicElement::H); // 16 CB-H
    let h2e = g.add_element(PeriodicElement::H); // 17 CB-H

    // Glycine-2: N-CA-C(=O)-OH
    let n3 = g.add_element(PeriodicElement::N); // 18
    let ca3 = g.add_element(PeriodicElement::C); // 19
    let c3 = g.add_element(PeriodicElement::C); // 20
    let o3 = g.add_element(PeriodicElement::O); // 21
    let oh3 = g.add_element(PeriodicElement::O); // 22 COOH oxygen
    let h3a = g.add_element(PeriodicElement::H); // 23 NH
    let h3b = g.add_element(PeriodicElement::H); // 24 CA-H
    let h3c = g.add_element(PeriodicElement::H); // 25 CA-H
    let h3d = g.add_element(PeriodicElement::H); // 26 OH

    // Gly-1 bonds
    g.add_bond(n1, ca1, BondOrder::Single).unwrap();
    g.add_bond(ca1, c1, BondOrder::Single).unwrap();
    g.add_bond(c1, o1, BondOrder::Double).unwrap();
    g.add_bond(n1, h1a, BondOrder::Single).unwrap();
    g.add_bond(n1, h1b, BondOrder::Single).unwrap();
    g.add_bond(ca1, h1c, BondOrder::Single).unwrap();
    g.add_bond(ca1, h1d, BondOrder::Single).unwrap();

    // Peptide bond Gly1-Ala
    g.add_bond(c1, n2, BondOrder::Single).unwrap();

    // Ala bonds
    g.add_bond(n2, ca2, BondOrder::Single).unwrap();
    g.add_bond(ca2, c2, BondOrder::Single).unwrap();
    g.add_bond(c2, o2, BondOrder::Double).unwrap();
    g.add_bond(ca2, cb2, BondOrder::Single).unwrap();
    g.add_bond(n2, h2a, BondOrder::Single).unwrap();
    g.add_bond(ca2, h2b, BondOrder::Single).unwrap();
    g.add_bond(cb2, h2c, BondOrder::Single).unwrap();
    g.add_bond(cb2, h2d, BondOrder::Single).unwrap();
    g.add_bond(cb2, h2e, BondOrder::Single).unwrap();

    // Peptide bond Ala-Gly2
    g.add_bond(c2, n3, BondOrder::Single).unwrap();

    // Gly-2 bonds
    g.add_bond(n3, ca3, BondOrder::Single).unwrap();
    g.add_bond(ca3, c3, BondOrder::Single).unwrap();
    g.add_bond(c3, o3, BondOrder::Double).unwrap();
    g.add_bond(c3, oh3, BondOrder::Single).unwrap(); // COOH
    g.add_bond(n3, h3a, BondOrder::Single).unwrap();
    g.add_bond(ca3, h3b, BondOrder::Single).unwrap();
    g.add_bond(ca3, h3c, BondOrder::Single).unwrap();
    g.add_bond(oh3, h3d, BondOrder::Single).unwrap();

    // Approximate 3D coordinates (angstroms, backbone extended)
    let positions: Vec<[f32; 3]> = vec![
        // Gly-1
        [-3.0, 0.0, 0.0],   // N1
        [-1.5, 0.0, 0.0],   // CA1
        [-0.7, 1.2, 0.0],   // C1
        [-1.2, 2.3, 0.0],   // O1
        [-3.5, 0.8, 0.0],   // H1a
        [-3.5, -0.8, 0.0],  // H1b
        [-1.2, -0.5, 0.9],  // H1c
        [-1.2, -0.5, -0.9], // H1d
        // Ala
        [0.6, 1.1, 0.0],  // N2
        [1.5, 2.2, 0.0],  // CA2
        [2.9, 1.7, 0.0],  // C2
        [3.3, 0.5, 0.0],  // O2
        [1.3, 3.0, 1.2],  // CB2
        [0.9, 0.2, 0.0],  // H2a
        [1.3, 2.8, -0.9], // H2b
        [0.3, 3.4, 1.2],  // H2c
        [2.0, 3.8, 1.2],  // H2d
        [1.4, 2.5, 2.1],  // H2e
        // Gly-2
        [3.7, 2.7, 0.0],  // N3
        [5.1, 2.5, 0.0],  // CA3
        [5.9, 3.7, 0.0],  // C3
        [5.5, 4.8, 0.0],  // O3
        [7.1, 3.5, 0.0],  // OH
        [3.3, 3.6, 0.0],  // H3a
        [5.4, 2.0, 0.9],  // H3b
        [5.4, 2.0, -0.9], // H3c
        [7.5, 4.4, 0.0],  // H3d
    ];

    EmbeddedMolecule::new(g, positions).unwrap()
}

fn main() {
    let mut population: usize = 8;
    let mut generations: usize = 5;
    let mut frames: usize = 100;
    let mut lite = false;
    let mut seed: u64 = 42;
    let mut telemetry_path: Option<String> = None;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--population" => {
                population = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(population);
                i += 1;
            }
            "--generations" => {
                generations = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(generations);
                i += 1;
            }
            "--frames" => {
                frames = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(frames);
                i += 1;
            }
            "--lite" => {
                lite = true;
            }
            "--seed" => {
                seed = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(seed);
                i += 1;
            }
            "--telemetry" => {
                telemetry_path = args.get(i + 1).cloned();
                i += 1;
            }
            "--help" | "-h" => {
                eprintln!("Enzyme-in-Soil Demo: evolve worlds with atomistic enzyme probes");
                eprintln!("  --population <N>   Population size (default: 8)");
                eprintln!("  --generations <N>  Generations (default: 5)");
                eprintln!("  --frames <N>       Frames per world (default: 100)");
                eprintln!("  --lite             Use lite 10x8 worlds");
                eprintln!("  --seed <N>         Random seed (default: 42)");
                eprintln!("  --telemetry <PATH> Output telemetry JSON");
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown arg: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    eprintln!("=== Enzyme-in-Soil Demo ===");
    eprintln!("Building tripeptide Gly-Ala-Gly ({} atoms)...", 27);

    // Verify molecule builds correctly
    let mol = build_tripeptide_gag();
    eprintln!(
        "Molecule: {} ({} atoms, {} bonds)",
        mol.graph.name,
        mol.graph.atom_count(),
        mol.graph.bond_count()
    );

    // Run evolution with enzyme fitness
    let config = EvolutionConfig {
        population_size: population,
        generations,
        frames_per_world: frames,
        master_seed: seed,
        lite,
        constraints: GenomeConstraints {
            max_fly_count: Some(2),
            max_microbe_count: Some(3),
            force_fly_count: None,
        },
        fitness: FitnessConfig {
            primary: FitnessObjective::MaxEnzymeEfficacy,
            snapshot_interval: 10,
        },
        strategy: SearchStrategy::Evolutionary {
            tournament_size: 3,
            mutation_rate: 0.15,
            crossover_rate: 0.7,
            elitism: 2,
        },
        thread_count: None,
    };

    eprintln!(
        "Running evolution: {} pop x {} gen x {} frames (lite={})...",
        population, generations, frames, lite
    );

    match evolve(config) {
        Ok(result) => {
            eprintln!("\n=== Enzyme Evolution Complete ===");
            eprintln!("Best fitness: {:.4}", result.global_best_fitness);
            eprintln!(
                "Best genome enzyme position: ({:.2}, {:.2})",
                result.global_best_genome.enzyme_probe_x, result.global_best_genome.enzyme_probe_y
            );
            eprintln!("Total worlds evaluated: {}", result.total_worlds_evaluated);
            eprintln!("Total time: {:.2}s", result.total_wall_time_ms / 1000.0);

            // Demonstrate spawning the probe in a fresh world
            eprintln!("\n--- Spawning enzyme probe in best world ---");
            let best = &result.global_best_genome;
            let build_fn = if lite {
                WorldGenome::build_world_lite
            } else {
                WorldGenome::build_world
            };
            match build_fn(best) {
                Ok(mut world) => {
                    let w = world.config.width;
                    let h = world.config.height;
                    let gx = ((best.enzyme_probe_x * w as f32) as usize).min(w - 1);
                    let gy = ((best.enzyme_probe_y * h as f32) as usize).min(h - 1);
                    match world.spawn_probe(&mol, gx, gy, 1) {
                        Ok(id) => {
                            eprintln!(
                                "Probe spawned: id={}, grid=({},{}), atoms={}",
                                id,
                                gx,
                                gy,
                                mol.graph.atom_count()
                            );
                            // Run a few frames with probe active
                            for _ in 0..10 {
                                let _ = world.step_frame();
                            }
                            let snap = world.snapshot();
                            eprintln!("After 10 frames with probe:");
                            eprintln!("  Probes: {}", snap.atomistic_probes);
                            eprintln!("  Soil glucose: {:.4}", snap.mean_soil_glucose);
                            eprintln!("  Soil ammonium: {:.4}", snap.mean_soil_ammonium);
                            eprintln!("  Cell energy: {:.4}", snap.mean_cell_energy);
                        }
                        Err(e) => eprintln!("Failed to spawn probe: {}", e),
                    }
                }
                Err(e) => eprintln!("Failed to build world: {}", e),
            }

            // Export telemetry
            if let Some(ref path) = telemetry_path {
                let telemetry = telemetry_from_result(&result, Some("enzyme_demo"));
                let json = serde_json::to_string_pretty(&telemetry).unwrap_or_default();
                if let Err(e) = std::fs::write(path, &json) {
                    eprintln!("Error writing telemetry: {}", e);
                } else {
                    eprintln!("Telemetry written to: {}", path);
                }
            }
        }
        Err(e) => {
            eprintln!("Evolution failed: {}", e);
            std::process::exit(1);
        }
    }
}
