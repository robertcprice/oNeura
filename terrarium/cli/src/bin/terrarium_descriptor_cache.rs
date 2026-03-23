use oneura_core::terrarium::{
    terrarium_molecular_asset_hash, terrarium_quantum_descriptor_cache_binary,
    terrarium_quantum_descriptor_cache_json_pretty,
    terrarium_quantum_descriptor_tensor_json_pretty, warm_terrarium_quantum_descriptor_cache,
};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputMode {
    Summary,
    EntriesJson,
    TensorJson,
    Binary,
}

#[derive(Debug, Clone)]
struct Args {
    mode: OutputMode,
    output: Option<PathBuf>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            mode: OutputMode::Summary,
            output: None,
        }
    }
}

fn parse_args() -> Args {
    let argv: Vec<String> = env::args().collect();
    let mut args = Args::default();
    let mut idx = 1;
    while idx < argv.len() {
        match argv[idx].as_str() {
            "--entries-json" => args.mode = OutputMode::EntriesJson,
            "--tensor-json" => args.mode = OutputMode::TensorJson,
            "--binary" => args.mode = OutputMode::Binary,
            "--output" | "-o" => {
                if let Some(path) = argv.get(idx + 1) {
                    args.output = Some(PathBuf::from(path));
                    idx += 1;
                } else {
                    eprintln!("missing value for --output");
                    std::process::exit(1);
                }
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown argument: {other}");
                print_help();
                std::process::exit(1);
            }
        }
        idx += 1;
    }
    if args.output.is_some() && args.mode == OutputMode::Summary {
        args.mode = OutputMode::EntriesJson;
    }
    args
}

fn print_help() {
    println!("Terrarium quantum descriptor cache materializer");
    println!();
    println!(
        "Usage: terrarium_descriptor_cache [--entries-json|--tensor-json|--binary] [--output PATH]"
    );
    println!();
    println!("Options:");
    println!("  --entries-json    Emit per-species descriptor entries as pretty JSON");
    println!("  --tensor-json     Emit a row-major tensor snapshot for accelerator ingestion");
    println!("  --binary          Emit a compact binary cache with asset hash validation");
    println!("  --output, -o      Write the selected output to a file");
    println!("  --help, -h        Show this help");
    println!();
    println!("Without a format flag, the command warms the terrarium fast-path descriptor cache");
    println!("and prints a short summary.");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let start = Instant::now();
    let warmed = warm_terrarium_quantum_descriptor_cache();
    let elapsed = start.elapsed();

    match args.mode {
        OutputMode::Summary => {
            let hash = terrarium_molecular_asset_hash();
            println!(
                "warmed {} terrarium quantum descriptors in {:.2?}",
                warmed, elapsed
            );
            println!(
                "molecular asset hash: {:02x}{:02x}{:02x}{:02x}...",
                hash[0], hash[1], hash[2], hash[3]
            );
            println!("use --entries-json, --tensor-json, or --binary to emit a cache artifact");
        }
        OutputMode::EntriesJson => {
            let json = terrarium_quantum_descriptor_cache_json_pretty()?;
            if let Some(path) = args.output {
                fs::write(&path, json.as_bytes())?;
                println!(
                    "warmed {} terrarium quantum descriptors in {:.2?} and wrote {} ({} bytes)",
                    warmed,
                    elapsed,
                    path.display(),
                    json.len()
                );
            } else {
                print!("{json}");
            }
        }
        OutputMode::TensorJson => {
            let json = terrarium_quantum_descriptor_tensor_json_pretty()?;
            if let Some(path) = args.output {
                fs::write(&path, json.as_bytes())?;
                println!(
                    "warmed {} terrarium quantum descriptors in {:.2?} and wrote {} ({} bytes)",
                    warmed,
                    elapsed,
                    path.display(),
                    json.len()
                );
            } else {
                print!("{json}");
            }
        }
        OutputMode::Binary => {
            let binary = terrarium_quantum_descriptor_cache_binary();
            if let Some(path) = args.output {
                fs::write(&path, &binary)?;
                println!(
                    "warmed {} terrarium quantum descriptors in {:.2?} and wrote {} ({} bytes, binary)",
                    warmed,
                    elapsed,
                    path.display(),
                    binary.len()
                );
            } else {
                // Binary to stdout — write raw bytes
                use std::io::Write;
                std::io::stdout().write_all(&binary)?;
            }
        }
    }

    Ok(())
}
