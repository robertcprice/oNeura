// Standalone Drug Protocol Optimizer CLI
//
// Compares antibiotic treatment protocols against bacterial persister cells,
// validated against Balaban et al. 2004 E. coli persistence data.
//
// Usage:
//   drug_optimizer --mode compare --switching-rate 1e-5
//   drug_optimizer --mode validate
//   drug_optimizer --mode optimize --kill-rate 3.0 --cycles 4
//   drug_optimizer --mode scan --output results.json

use oneura_core::terrarium::{
    optimize_drug_protocol, validate_against_ecoli, DrugProtocol, DrugProtocolResult,
    PersisterCellSimulator,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args
        .iter()
        .position(|a| a == "--mode")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("compare");

    match mode {
        "compare" => run_compare(&args),
        "validate" => run_validate(),
        "optimize" => run_optimize(&args),
        "scan" => run_scan(&args),
        "help" | "--help" | "-h" => print_help(),
        _ => {
            eprintln!(
                "Unknown mode '{}'. Use --mode compare|validate|optimize|scan",
                mode
            );
            std::process::exit(1);
        }
    }
}

fn print_help() {
    println!("oNeura Drug Protocol Optimizer");
    println!("==============================");
    println!();
    println!("Compares antibiotic treatment strategies against bacterial persister cells.");
    println!("Based on the Balaban persister switching model (Balaban et al. 2004).");
    println!();
    println!("MODES:");
    println!("  --mode compare    Compare single, pulsed, and combination protocols (default)");
    println!("  --mode validate   Validate model against published E. coli persistence data");
    println!("  --mode optimize   Find optimal pulsed dosing schedule");
    println!("  --mode scan       Scan switching rate × kill rate parameter space");
    println!();
    println!("OPTIONS:");
    println!("  --switching-rate <f32>   Persister switching rate (default: 1e-5)");
    println!("  --kill-rate <f32>        Antibiotic kill rate (default: 3.0)");
    println!("  --cycles <usize>         Number of pulsed cycles to try (default: 5)");
    println!("  --dt <f32>               Simulation timestep in hours (default: 0.05)");
    println!("  --output <path>          Write JSON results to file");
    println!("  --json                   Output results as JSON to stdout");
}

fn parse_f32(args: &[String], flag: &str, default: f32) -> f32 {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn parse_usize(args: &[String], flag: &str, default: usize) -> usize {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn run_compare(args: &[String]) {
    let switching_rate = parse_f32(args, "--switching-rate", 1e-5);
    let kill_rate = parse_f32(args, "--kill-rate", 3.0);
    let dt = parse_f32(args, "--dt", 0.05);
    let json_output = has_flag(args, "--json");

    println!("Drug Protocol Comparison");
    println!("========================");
    println!("Switching rate: {:.2e}", switching_rate);
    println!("Kill rate: {:.1}", kill_rate);
    println!();

    let protocols = vec![
        ("Single dose (10h)", DrugProtocol::single(kill_rate, 10.0)),
        (
            "Pulsed (3h on / 2h off x3)",
            DrugProtocol::pulsed(kill_rate, 3.0, 2.0, 3),
        ),
        (
            "Pulsed (5h on / 1h off x5)",
            DrugProtocol::pulsed(kill_rate, 5.0, 1.0, 5),
        ),
        (
            "Combination (A+B, 5h each)",
            DrugProtocol::combination(kill_rate, kill_rate * 1.5, 5.0),
        ),
        (
            "High-dose short (2x kill, 5h)",
            DrugProtocol::single(kill_rate * 2.0, 5.0),
        ),
    ];

    let protocol_list: Vec<DrugProtocol> = protocols
        .iter()
        .map(|(_, p): &(&str, DrugProtocol)| p.clone())
        .collect();
    let (best_idx, results) = optimize_drug_protocol(&protocol_list, switching_rate, dt);

    if json_output {
        let output: Vec<serde_json::Value> = protocols
            .iter()
            .zip(results.iter())
            .enumerate()
            .map(|(i, ((name, _), result))| {
                serde_json::json!({
                    "name": name,
                    "is_best": i == best_idx,
                    "final_population": result.final_population,
                    "survival_fraction": result.survival_fraction,
                    "minimum_population": result.minimum_population,
                    "eradication_achieved": result.eradication_achieved,
                    "total_time_hours": result.total_time_hours,
                    "cycles_completed": result.cycles_completed,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&output).unwrap());
        return;
    }

    println!(
        "{:<35} {:>12} {:>12} {:>10} {:>8}",
        "Protocol", "Final Pop", "Survival%", "Min Pop", "Eradic"
    );
    println!("{}", "-".repeat(80));
    for (i, ((name, _), result)) in protocols.iter().zip(results.iter()).enumerate() {
        let marker = if i == best_idx { " << BEST" } else { "" };
        println!(
            "{:<35} {:>12.1} {:>11.4}% {:>10.1} {:>8}{}",
            name,
            result.final_population,
            result.survival_fraction * 100.0,
            result.minimum_population,
            if result.eradication_achieved {
                "YES"
            } else {
                "no"
            },
            marker,
        );
    }

    println!();
    println!("Best protocol: {}", protocols[best_idx].0);

    if let Some(output_path) = args
        .iter()
        .position(|a| a == "--output")
        .and_then(|i| args.get(i + 1))
    {
        let output: Vec<serde_json::Value> = protocols
            .iter()
            .zip(results.iter())
            .enumerate()
            .map(|(i, ((name, _), result))| {
                serde_json::json!({
                    "name": name,
                    "is_best": i == best_idx,
                    "final_population": result.final_population,
                    "survival_fraction": result.survival_fraction,
                    "eradication_achieved": result.eradication_achieved,
                })
            })
            .collect();
        std::fs::write(output_path, serde_json::to_string_pretty(&output).unwrap())
            .expect("Failed to write output file");
        println!("Results written to {}", output_path);
    }
}

fn run_validate() {
    println!("E. coli Persistence Model Validation");
    println!("=====================================");
    println!("Reference: Balaban et al. 2004, Science 305:1622-1625");
    println!();
    println!(
        "{:<20} {:>12} {:>12} {:>10}",
        "Switch Rate", "Expected", "Simulated", "Valid?"
    );
    println!("{}", "-".repeat(58));

    let results = validate_against_ecoli();
    let mut all_valid = true;
    for (switch_rate, expected, actual, within_order) in &results {
        println!(
            "{:<20.2e} {:>12.4} {:>12.4} {:>10}",
            switch_rate,
            expected,
            actual,
            if *within_order { "YES" } else { "FAIL" },
        );
        if !within_order {
            all_valid = false;
        }
    }

    println!();
    if all_valid {
        println!("VALIDATION PASSED: All simulated values within 1 order of magnitude of published data.");
    } else {
        println!("VALIDATION PARTIAL: Some values outside expected range.");
    }

    // Show monotonicity check
    println!();
    println!("Monotonicity check (higher switching rate -> higher survival):");
    let mut prev_actual = 0.0;
    let mut monotonic = true;
    for (switch_rate, _, actual, _) in &results {
        let ok = *actual >= prev_actual;
        if !ok {
            monotonic = false;
        }
        println!(
            "  {:.2e} -> survival {:.6} {}",
            switch_rate,
            actual,
            if ok { "OK" } else { "VIOLATED" }
        );
        prev_actual = *actual;
    }
    println!(
        "Monotonicity: {}",
        if monotonic { "PASSED" } else { "FAILED" }
    );
}

fn run_optimize(args: &[String]) {
    let switching_rate = parse_f32(args, "--switching-rate", 1e-5);
    let kill_rate = parse_f32(args, "--kill-rate", 3.0);
    let max_cycles = parse_usize(args, "--cycles", 5);
    let dt = parse_f32(args, "--dt", 0.05);

    println!("Pulsed Dosing Optimization");
    println!("==========================");
    println!("Switching rate: {:.2e}", switching_rate);
    println!("Kill rate: {:.1}", kill_rate);
    println!("Max cycles: {}", max_cycles);
    println!();

    // Search over treatment_hours × rest_hours × n_cycles
    let treatment_hours_range = [1.0, 2.0, 3.0, 5.0, 8.0];
    let rest_hours_range = [0.5, 1.0, 2.0, 3.0, 5.0];

    let mut best_protocol: Option<(String, DrugProtocolResult)> = None;

    println!(
        "{:<40} {:>12} {:>12} {:>8}",
        "Protocol", "Final Pop", "Survival%", "Eradic"
    );
    println!("{}", "-".repeat(75));

    for &treatment_h in &treatment_hours_range {
        for &rest_h in &rest_hours_range {
            for n_cycles in 1..=max_cycles {
                let protocol = DrugProtocol::pulsed(kill_rate, treatment_h, rest_h, n_cycles);
                let mut sim = PersisterCellSimulator::with_switching_rates(switching_rate, 1e-4);
                let result = protocol.execute(&mut sim, dt);

                let name = format!("{:.0}h on / {:.1}h off x{}", treatment_h, rest_h, n_cycles);

                let is_better = best_protocol.as_ref().map_or(true, |(_, best)| {
                    result.final_population < best.final_population
                });

                if is_better || result.eradication_achieved {
                    println!(
                        "{:<40} {:>12.1} {:>11.4}% {:>8}",
                        name,
                        result.final_population,
                        result.survival_fraction * 100.0,
                        if result.eradication_achieved {
                            "YES"
                        } else {
                            "no"
                        },
                    );
                }

                if is_better {
                    best_protocol = Some((name, result));
                }
            }
        }
    }

    if let Some((name, result)) = &best_protocol {
        println!();
        println!(
            "OPTIMAL: {} (survival {:.6}%)",
            name,
            result.survival_fraction * 100.0
        );
    }
}

fn run_scan(args: &[String]) {
    let dt = parse_f32(args, "--dt", 0.05);
    let json_output = has_flag(args, "--json");

    let switching_rates = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3];
    let kill_rates = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0];

    if !json_output {
        println!("Parameter Space Scan: Switching Rate x Kill Rate");
        println!("=================================================");
        println!();
        println!("Survival fraction (%) for single 10h treatment:");
        println!();
        print!("{:<12}", "Switch\\Kill");
        for &kr in &kill_rates {
            print!("{:>10.1}", kr);
        }
        println!();
        println!("{}", "-".repeat(12 + kill_rates.len() * 10));
    }

    let mut scan_results = Vec::new();

    for &sr in &switching_rates {
        if !json_output {
            print!("{:<12.2e}", sr);
        }
        for &kr in &kill_rates {
            let protocol = DrugProtocol::single(kr, 10.0);
            let mut sim = PersisterCellSimulator::with_switching_rates(sr, 1e-4);
            let result = protocol.execute(&mut sim, dt);
            if !json_output {
                print!("{:>9.4}%", result.survival_fraction * 100.0);
            }
            scan_results.push(serde_json::json!({
                "switching_rate": sr,
                "kill_rate": kr,
                "survival_fraction": result.survival_fraction,
                "final_population": result.final_population,
                "eradication": result.eradication_achieved,
            }));
        }
        if !json_output {
            println!();
        }
    }

    if json_output {
        println!("{}", serde_json::to_string_pretty(&scan_results).unwrap());
    }

    if let Some(output_path) = args
        .iter()
        .position(|a| a == "--output")
        .and_then(|i| args.get(i + 1))
    {
        std::fs::write(
            output_path,
            serde_json::to_string_pretty(&scan_results).unwrap(),
        )
        .expect("Failed to write output file");
        if !json_output {
            println!();
            println!("Results written to {}", output_path);
        }
    }
}
