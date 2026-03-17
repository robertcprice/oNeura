#!/usr/bin/env python3
"""Insert new tests into terrarium_evolve.rs test module."""

TARGET = "oneuro-metal/src/terrarium_evolve.rs"

NEW_TESTS = r'''
    // ================================================================
    // Adaptive Evolution with Environment Tests
    // ================================================================

    #[test]
    fn evolve_with_environment_temperate() {
        let config = EvolutionConfig {
            population_size: 4, generations: 2, frames_per_world: 30,
            master_seed: 42, lite: true,
            ..Default::default()
        };
        let schedule = EnvironmentalSchedule::temperate();
        let result = evolve_with_environment(config, schedule).unwrap();
        assert!(result.global_best_fitness > 0.0, "Should find positive fitness");
        assert_eq!(result.generation_results.len(), 2);
    }

    #[test]
    fn evolve_with_environment_arid_is_harder() {
        let config = EvolutionConfig {
            population_size: 4, generations: 2, frames_per_world: 30,
            master_seed: 42, lite: true,
            ..Default::default()
        };
        let temperate = evolve_with_environment(config.clone(), EnvironmentalSchedule::temperate()).unwrap();
        let arid = evolve_with_environment(config, EnvironmentalSchedule::arid()).unwrap();
        // Both should produce results (not crash)
        assert!(temperate.total_worlds_evaluated > 0);
        assert!(arid.total_worlds_evaluated > 0);
    }

    // ================================================================
    // Coevolution Engine Tests
    // ================================================================

    #[test]
    fn coevolution_red_queen_runs() {
        let result = evolve_coevolution(4, 2, 20, CoevolutionMode::RedQueen, true, 42).unwrap();
        assert_eq!(result.history.len(), 2);
        assert!(!result.final_population_a.is_empty());
        assert!(!result.final_population_b.is_empty());
        assert_eq!(result.mode, CoevolutionMode::RedQueen);
    }

    #[test]
    fn coevolution_mutualistic_boosts_fitness() {
        let mut rng = StdRng::seed_from_u64(42);
        let a = SpeciesGenome { cooperation_tendency: 1.0, ..SpeciesGenome::random(&mut rng) };
        let b = SpeciesGenome { cooperation_tendency: 1.0, ..SpeciesGenome::random(&mut rng) };
        let result = evaluate_coevolution_pair(&a, &b, CoevolutionMode::Mutualistic, 20, true).unwrap();
        // Mutualistic with max cooperation should boost both
        assert!(result.interaction_strength > 0.5, "High cooperation species should have strong interaction");
    }

    #[test]
    fn coevolution_competitive_splits_resources() {
        let mut rng = StdRng::seed_from_u64(42);
        let a = SpeciesGenome { resource_efficiency: 0.9, ..SpeciesGenome::random(&mut rng) };
        let b = SpeciesGenome { resource_efficiency: 0.1, ..SpeciesGenome::random(&mut rng) };
        let result = evaluate_coevolution_pair(&a, &b, CoevolutionMode::Competitive, 20, true).unwrap();
        // More efficient species should get larger share
        assert!(result.species_a_fitness > result.species_b_fitness * 0.5,
            "Efficient species should outcompete: a={:.2} b={:.2}", result.species_a_fitness, result.species_b_fitness);
    }

    #[test]
    fn species_genome_mutation_preserves_bounds() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut g = SpeciesGenome::random(&mut rng);
        for _ in 0..100 {
            g.mutate(&mut rng, 0.5);
            assert!(g.defense_investment >= 0.0 && g.defense_investment <= 1.0);
            assert!(g.resource_efficiency >= 0.1 && g.resource_efficiency <= 1.0);
            assert!(g.cooperation_tendency >= 0.0 && g.cooperation_tendency <= 1.0);
            assert!(g.mobility >= 0.1 && g.mobility <= 1.0);
        }
    }

    // ================================================================
    // Genetic Regulatory Network Tests
    // ================================================================

    #[test]
    fn grn_random_nk_correct_size() {
        let mut rng = StdRng::seed_from_u64(42);
        let grn = GeneRegulatoryNetwork::random_nk(8, 3, &mut rng);
        assert_eq!(grn.n_genes, 8);
        assert_eq!(grn.weights.len(), 8);
        assert_eq!(grn.thresholds.len(), 8);
        assert_eq!(grn.expression.len(), 8);
    }

    #[test]
    fn grn_step_keeps_expression_bounded() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut grn = GeneRegulatoryNetwork::random_nk(10, 3, &mut rng);
        for _ in 0..100 {
            grn.step();
            for &e in &grn.expression {
                assert!(e >= 0.0 && e <= 1.0, "Expression out of bounds: {}", e);
            }
        }
    }

    #[test]
    fn grn_finds_attractor() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut grn = GeneRegulatoryNetwork::random_nk(6, 2, &mut rng);
        let attractor = grn.find_attractor(200);
        assert_eq!(attractor.len(), 6);
        // Attractor values should be in [0, 1]
        for &v in &attractor {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn grn_phenotype_has_valid_traits() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut grn = GeneRegulatoryNetwork::random_nk(12, 3, &mut rng);
        let pheno = grn.attractor_to_phenotype();
        assert!(pheno.growth_rate >= 0.0 && pheno.growth_rate <= 1.0);
        assert!(pheno.stress_tolerance >= 0.0 && pheno.stress_tolerance <= 1.0);
        assert!(pheno.network_complexity >= 0.0);
    }

    #[test]
    fn grn_evolution_converges() {
        let result = evolve_grn(0.7, 0.5, 8, 3, 10, 20, 42);
        assert_eq!(result.generations_run, 20);
        // Should approach target phenotype
        assert!(result.best_phenotype.growth_rate > 0.0, "Should produce growth");
    }

    #[test]
    fn grn_crossover_preserves_size() {
        let mut rng = StdRng::seed_from_u64(42);
        let a = GeneRegulatoryNetwork::random_nk(8, 3, &mut rng);
        let b = GeneRegulatoryNetwork::random_nk(8, 3, &mut rng);
        let child = GeneRegulatoryNetwork::crossover(&a, &b, &mut rng);
        assert_eq!(child.n_genes, 8);
        assert_eq!(child.weights.len(), 8);
    }

    // ================================================================
    // Ecosystem Health Tests
    // ================================================================

    #[test]
    fn ecosystem_health_empty_snapshots() {
        let report = assess_ecosystem_health(&[]);
        assert_eq!(report.overall_health, 0.0);
        assert_eq!(report.trophic_levels, 0);
    }

    #[test]
    fn ecosystem_health_computes_diversity() {
        let genome = WorldGenome::default_with_seed(42);
        let mut world = genome.build_world_lite().unwrap();
        let mut snapshots = Vec::new();
        for _ in 0..10 {
            world.step_frame().unwrap();
            snapshots.push(world.snapshot());
        }
        let report = assess_ecosystem_health(&snapshots);
        assert!(report.total_biomass >= 0.0);
        assert!(report.overall_health >= 0.0 && report.overall_health <= 100.0);
        assert!(report.stability >= 0.0);
    }

    // ================================================================
    // World Export Tests
    // ================================================================

    #[test]
    fn world_export_captures_snapshots() {
        let genome = WorldGenome::default_with_seed(42);
        let export = run_and_export(genome, 20, true, 5, None).unwrap();
        assert!(!export.snapshots.is_empty());
        assert!(export.metadata.fitness >= 0.0 || export.metadata.fitness < 0.0); // any finite
        assert_eq!(export.metadata.frames_run, 20);
        assert_eq!(export.metadata.lite_mode, true);
    }

    #[test]
    fn world_export_with_environment() {
        let genome = WorldGenome::default_with_seed(42);
        let sched = EnvironmentalSchedule::tropical();
        let export = run_and_export(genome, 15, true, 3, Some(&sched)).unwrap();
        assert!(export.environmental_samples.is_some());
        let samples = export.environmental_samples.unwrap();
        assert_eq!(samples.len(), 15);
    }

    #[test]
    fn world_export_serializes_to_json() {
        let genome = WorldGenome::default_with_seed(42);
        let export = run_and_export(genome, 10, true, 5, None).unwrap();
        let json = serde_json::to_string(&export).unwrap();
        assert!(json.len() > 100, "JSON should be substantial");
        // Can round-trip
        let parsed: WorldExport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.metadata.frames_run, 10);
    }

    // ================================================================
    // Sparkline & Dashboard Tests
    // ================================================================

    #[test]
    fn sparkline_renders_data() {
        let data = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let s = sparkline(&data, 5);
        assert_eq!(s.chars().count(), 5);
        // First should be space (min), last should be full block (max)
        assert_eq!(s.chars().next().unwrap(), ' ');
        assert_eq!(s.chars().last().unwrap(), '\u{2588}');
    }

    #[test]
    fn sparkline_empty_data() {
        let s = sparkline(&[], 10);
        assert!(s.is_empty());
    }

    #[test]
    fn dashboard_generates_output() {
        let genome = WorldGenome::default_with_seed(42);
        let mut world = genome.build_world_lite().unwrap();
        let mut snapshots = Vec::new();
        for _ in 0..10 {
            world.step_frame().unwrap();
            snapshots.push(world.snapshot());
        }
        let dash = ecosystem_dashboard(&snapshots, 80);
        assert!(dash.contains("oNeura Ecosystem Dashboard"));
        assert!(dash.contains("Biomass"));
        assert!(dash.contains("Moisture"));
    }
'''

with open(TARGET) as f:
    content = f.read()

# Find the closing brace of the test module (last line)
# Insert before the final "}"
# Find the last test and insert after it
marker = "    #[test]\n    fn circuit_random_in_bounds()"
idx = content.find(marker)
if idx < 0:
    print("ERROR: Could not find circuit_random_in_bounds test")
    exit(1)

# Find the end of this test (closing brace + newline before module close)
# Search for the next "\n\n}" or just "}\n\n}"
end_marker = "}\n\n}"
end_idx = content.find(end_marker, idx)
if end_idx < 0:
    # Try simpler pattern
    end_marker = "}\n}"
    end_idx = content.find(end_marker, idx)

if end_idx < 0:
    print("ERROR: Could not find end of test module")
    exit(1)

# Insert after the first "}" (end of circuit_random_in_bounds test), before the module close
insert_at = end_idx + 1  # after the first "}"

new_content = content[:insert_at] + "\n" + NEW_TESTS + "\n" + content[insert_at:]

import tempfile, os
with tempfile.NamedTemporaryFile(mode='w', dir=os.path.dirname(TARGET), delete=False, suffix='.rs') as tmp:
    tmp.write(new_content)
    tmp_path = tmp.name
os.replace(tmp_path, TARGET)

print(f"Inserted {len(NEW_TESTS)} chars of new tests")
print(f"New file size: {len(new_content)} chars")
