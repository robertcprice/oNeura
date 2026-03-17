// Standalone Gene Circuit Noise Designer CLI
//
// Designs synthetic gene circuits with specific noise properties using
// evolutionary optimization over the telegraph promoter model.
//
// Usage:
//   gene_circuit --target-fano 5.0 --target-mean 100.0 --target-cv 0.15
//   gene_circuit --mode sweep --output results.json
//   gene_circuit --mode analyze --on-rate 0.05 --off-rate 0.1 --burst 3.0

use oneuro_metal::terrarium_evolve::{
    GeneCircuitSpec, GeneCircuitParams, GeneCircuitResult, optimize_gene_circuit,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.iter().position(|a| a == "--mode")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("design");

    match mode {
        "design" => run_design(&args),
        "analyze" => run_analyze(&args),
        "sweep" => run_sweep(&args),
        "help" | "--help" | "-h" => print_help(),
        _ => {
            eprintln!("Unknown mode '{}'. Use --mode design|analyze|sweep", mode);
            std::process::exit(1);
        }
    }
}

fn print_help() {
    println!("oNeura Gene Circuit Noise Designer");
    println!("===================================");
    println!();
    println!("Designs synthetic gene circuits with target noise properties.");
    println!("Uses the telegraph promoter model (Raj & van Oudenaarden 2008).");
    println!();
    println!("MODES:");
    println!("  --mode design     Optimize circuit for target noise (default)");
    println!("  --mode analyze    Compute noise properties of given parameters");
    println!("  --mode sweep      Sweep parameter space and report Fano landscape");
    println!();
    println!("DESIGN OPTIONS:");
    println!("  --target-fano <f32>     Target Fano factor (default: 5.0)");
    println!("  --target-mean <f32>     Target mean protein count (default: 100.0)");
    println!("  --target-cv <f32>       Target coefficient of variation (default: 0.15)");
    println!("  --population <usize>    Evolution population size (default: 100)");
    println!("  --generations <usize>   Evolution generations (default: 200)");
    println!("  --seed <u64>            Random seed (default: 42)");
    println!();
    println!("ANALYZE OPTIONS:");
    println!("  --on-rate <f32>         Promoter ON rate (default: 0.05)");
    println!("  --off-rate <f32>        Promoter OFF rate (default: 0.1)");
    println!("  --burst <f32>           Burst size (default: 3.0)");
    println!("  --txn-rate <f32>        Transcription rate (default: 0.1)");
    println!("  --mrna-deg <f32>        mRNA degradation rate (default: 0.01)");
    println!("  --tln-rate <f32>        Translation rate (default: 0.5)");
    println!("  --prot-deg <f32>        Protein degradation rate (default: 0.01)");
    println!();
    println!("COMMON OPTIONS:");
    println!("  --output <path>         Write JSON results to file");
    println!("  --json                  Output results as JSON to stdout");
    println!();
    println!("THEORY:");
    println!("  Telegraph model: promoter switches between ON and OFF states.");
    println!("  Fano factor = 1 + burst_size * k_off / (k_on + k_off)");
    println!("  Mean protein = (txn_rate * k_on / ((k_on + k_off) * mrna_deg)) * (tln_rate / prot_deg)");
    println!("  CV = sqrt(Fano / mean)");
}

fn parse_f32(args: &[String], flag: &str, default: f32) -> f32 {
    args.iter().position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn parse_usize(args: &[String], flag: &str, default: usize) -> usize {
    args.iter().position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn parse_u64(args: &[String], flag: &str, default: u64) -> u64 {
    args.iter().position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn run_design(args: &[String]) {
    let target_fano = parse_f32(args, "--target-fano", 5.0);
    let target_mean = parse_f32(args, "--target-mean", 100.0);
    let target_cv = parse_f32(args, "--target-cv", 0.15);
    let population = parse_usize(args, "--population", 100);
    let generations = parse_usize(args, "--generations", 200);
    let seed = parse_u64(args, "--seed", 42);
    let json_output = has_flag(args, "--json");

    let spec = GeneCircuitSpec {
        target_fano,
        target_mean_protein: target_mean,
        target_cv,
    };

    let result = optimize_gene_circuit(&spec, population, generations, seed);

    if json_output {
        let output = serde_json::json!({
            "target": { "fano": target_fano, "mean_protein": target_mean, "cv": target_cv },
            "achieved": {
                "fano": result.achieved_fano,
                "mean_protein": result.achieved_mean_protein,
                "cv": result.achieved_cv,
            },
            "fitness": result.fitness,
            "parameters": {
                "promoter_on_rate": result.best_params.promoter_on_rate,
                "promoter_off_rate": result.best_params.promoter_off_rate,
                "transcription_rate": result.best_params.transcription_rate,
                "mrna_degradation_rate": result.best_params.mrna_degradation_rate,
                "translation_rate": result.best_params.translation_rate,
                "protein_degradation_rate": result.best_params.protein_degradation_rate,
                "burst_size": result.best_params.burst_size,
            },
            "generations_run": result.generations_run,
        });
        println!("{}", serde_json::to_string_pretty(&output).unwrap());
    } else {
        println!("Gene Circuit Noise Designer");
        println!("===========================");
        println!();
        println!("Target:  Fano={:.2}  Mean={:.1}  CV={:.4}", target_fano, target_mean, target_cv);
        println!("Search:  pop={}  gen={}  seed={}", population, generations, seed);
        println!();
        println!("OPTIMIZED PARAMETERS:");
        println!("  Promoter ON rate:        {:.6}", result.best_params.promoter_on_rate);
        println!("  Promoter OFF rate:       {:.6}", result.best_params.promoter_off_rate);
        println!("  Transcription rate:      {:.6}", result.best_params.transcription_rate);
        println!("  mRNA degradation rate:   {:.6}", result.best_params.mrna_degradation_rate);
        println!("  Translation rate:        {:.6}", result.best_params.translation_rate);
        println!("  Protein degradation:     {:.6}", result.best_params.protein_degradation_rate);
        println!("  Burst size:              {:.2}", result.best_params.burst_size);
        println!();
        println!("ACHIEVED NOISE:");
        println!("  Fano factor:  {:.4}  (target: {:.2}, error: {:.2}%)",
            result.achieved_fano, target_fano,
            ((result.achieved_fano - target_fano) / target_fano * 100.0).abs());
        println!("  Mean protein: {:.2}  (target: {:.1}, error: {:.2}%)",
            result.achieved_mean_protein, target_mean,
            ((result.achieved_mean_protein - target_mean) / target_mean * 100.0).abs());
        println!("  CV:           {:.6}  (target: {:.4}, error: {:.2}%)",
            result.achieved_cv, target_cv,
            ((result.achieved_cv - target_cv) / target_cv * 100.0).abs());
        println!();
        println!("FITNESS: {:.6}", result.fitness);

        // Regime classification
        let duty_cycle = result.best_params.promoter_on_rate
            / (result.best_params.promoter_on_rate + result.best_params.promoter_off_rate);
        println!();
        println!("REGIME ANALYSIS:");
        println!("  Duty cycle (ON fraction): {:.2}%", duty_cycle * 100.0);
        if duty_cycle > 0.8 {
            println!("  Classification: CONSTITUTIVE-LIKE (mostly ON)");
        } else if duty_cycle < 0.2 {
            println!("  Classification: BURSTY (mostly OFF, rare bursts)");
        } else {
            println!("  Classification: SWITCHING (balanced ON/OFF)");
        }
        let burst_freq = result.best_params.promoter_on_rate;
        let burst_dur = 1.0 / result.best_params.promoter_off_rate;
        println!("  Burst frequency: {:.4} /time", burst_freq);
        println!("  Mean burst duration: {:.2} time units", burst_dur);
        println!("  Proteins per burst: ~{:.0}", result.best_params.burst_size * burst_dur * result.best_params.transcription_rate);
    }

    if let Some(output_path) = args.iter().position(|a| a == "--output").and_then(|i| args.get(i + 1)) {
        let output = serde_json::json!({
            "target": { "fano": target_fano, "mean_protein": target_mean, "cv": target_cv },
            "achieved": { "fano": result.achieved_fano, "mean_protein": result.achieved_mean_protein, "cv": result.achieved_cv },
            "parameters": {
                "promoter_on_rate": result.best_params.promoter_on_rate,
                "promoter_off_rate": result.best_params.promoter_off_rate,
                "burst_size": result.best_params.burst_size,
            },
        });
        std::fs::write(output_path, serde_json::to_string_pretty(&output).unwrap())
            .expect("Failed to write output file");
        println!("Results written to {}", output_path);
    }
}

fn run_analyze(args: &[String]) {
    let params = GeneCircuitParams {
        promoter_on_rate: parse_f32(args, "--on-rate", 0.05),
        promoter_off_rate: parse_f32(args, "--off-rate", 0.1),
        transcription_rate: parse_f32(args, "--txn-rate", 0.1),
        mrna_degradation_rate: parse_f32(args, "--mrna-deg", 0.01),
        translation_rate: parse_f32(args, "--tln-rate", 0.5),
        protein_degradation_rate: parse_f32(args, "--prot-deg", 0.01),
        burst_size: parse_f32(args, "--burst", 3.0),
    };

    let json_output = has_flag(args, "--json");

    let fano = params.predicted_fano();
    let mean = params.predicted_mean_protein();
    let cv = params.predicted_cv();
    let duty_cycle = params.promoter_on_rate / (params.promoter_on_rate + params.promoter_off_rate);

    if json_output {
        println!("{}", serde_json::to_string_pretty(&serde_json::json!({
            "parameters": {
                "promoter_on_rate": params.promoter_on_rate,
                "promoter_off_rate": params.promoter_off_rate,
                "burst_size": params.burst_size,
            },
            "predictions": { "fano": fano, "mean_protein": mean, "cv": cv, "duty_cycle": duty_cycle },
        })).unwrap());
    } else {
        println!("Gene Circuit Analysis");
        println!("=====================");
        println!();
        println!("INPUT PARAMETERS:");
        println!("  k_on  = {:.6}    k_off = {:.6}", params.promoter_on_rate, params.promoter_off_rate);
        println!("  txn   = {:.6}    mRNA deg = {:.6}", params.transcription_rate, params.mrna_degradation_rate);
        println!("  tln   = {:.6}    prot deg = {:.6}", params.translation_rate, params.protein_degradation_rate);
        println!("  burst = {:.2}", params.burst_size);
        println!();
        println!("PREDICTED NOISE PROPERTIES:");
        println!("  Fano factor:     {:.4}", fano);
        println!("  Mean protein:    {:.2}", mean);
        println!("  CV:              {:.6}", cv);
        println!("  Variance:        {:.2}", fano * mean);
        println!("  Std deviation:   {:.2}", (fano * mean).sqrt());
        println!();
        println!("PROMOTER REGIME:");
        println!("  Duty cycle:      {:.1}%", duty_cycle * 100.0);
        if fano < 1.2 {
            println!("  Noise level:     POISSON-LIKE (minimal noise)");
        } else if fano < 3.0 {
            println!("  Noise level:     MODERATE super-Poisson");
        } else if fano < 10.0 {
            println!("  Noise level:     HIGH super-Poisson (bursty)");
        } else {
            println!("  Noise level:     VERY HIGH (extreme bursting)");
        }
    }
}

fn run_sweep(args: &[String]) {
    let population = parse_usize(args, "--population", 50);
    let generations = parse_usize(args, "--generations", 100);
    let json_output = has_flag(args, "--json");

    let fano_targets = [1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0];
    let mean_targets = [10.0, 50.0, 100.0, 500.0, 1000.0];

    if !json_output {
        println!("Gene Circuit Noise Landscape Sweep");
        println!("===================================");
        println!("Optimizing circuits across Fano x Mean protein space");
        println!();
        print!("{:<12}", "Fano\\Mean");
        for &m in &mean_targets {
            print!("{:>10.0}", m);
        }
        println!("{:>12}", "Best CV");
        println!("{}", "-".repeat(12 + mean_targets.len() * 10 + 12));
    }

    let mut sweep_results = Vec::new();
    let mut seed: u64 = 1;

    for &target_fano in &fano_targets {
        if !json_output {
            print!("{:<12.1}", target_fano);
        }
        let mut best_cv_for_fano = f32::INFINITY;

        for &target_mean in &mean_targets {
            let spec = GeneCircuitSpec {
                target_fano,
                target_mean_protein: target_mean,
                target_cv: (target_fano / target_mean).sqrt(),
            };
            let result = optimize_gene_circuit(&spec, population, generations, seed);
            seed += 1;

            let fano_error = ((result.achieved_fano - target_fano) / target_fano).abs();
            if !json_output {
                if fano_error < 0.1 {
                    print!("{:>10.4}", result.achieved_cv);
                } else {
                    print!("{:>10}", "---");
                }
            }
            if result.achieved_cv < best_cv_for_fano {
                best_cv_for_fano = result.achieved_cv;
            }

            sweep_results.push(serde_json::json!({
                "target_fano": target_fano,
                "target_mean": target_mean,
                "achieved_fano": result.achieved_fano,
                "achieved_mean": result.achieved_mean_protein,
                "achieved_cv": result.achieved_cv,
                "fitness": result.fitness,
                "burst_size": result.best_params.burst_size,
                "on_rate": result.best_params.promoter_on_rate,
                "off_rate": result.best_params.promoter_off_rate,
            }));
        }

        if !json_output {
            println!("{:>12.6}", best_cv_for_fano);
        }
    }

    if json_output {
        println!("{}", serde_json::to_string_pretty(&sweep_results).unwrap());
    }

    if let Some(output_path) = args.iter().position(|a| a == "--output").and_then(|i| args.get(i + 1)) {
        std::fs::write(output_path, serde_json::to_string_pretty(&sweep_results).unwrap())
            .expect("Failed to write output file");
        if !json_output {
            println!();
            println!("Results written to {}", output_path);
        }
    }
}
