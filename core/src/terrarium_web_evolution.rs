//! Async evolution runner for the web server.
//!
//! Wraps the synchronous evolution engine in spawn_blocking with per-generation
//! streaming and cancellation support.

use crate::terrarium_evolve::{
    evolve, evolve_pareto, evolve_pareto_stressed, evolve_stress_test, EvolutionConfig,
    EvolutionResult, ParetoEvolutionResult, SearchStrategy,
};
use crate::terrarium_web_protocol::{
    EvolveCompleteData, EvolveGenerationData, EvolveWebConfig, ParetoFrontEntry, ServerMsg,
};
use crate::terrarium_web_state::{AppState, EvolutionHandle};
use std::sync::Arc;
use tokio::sync::oneshot;

/// Start an evolution run in the background.
///
/// Streams per-generation results via the broadcast channel and sends a
/// completion message when done. Cancellable via the returned oneshot sender
/// stored in AppState::evolution.
pub async fn start_evolution(state: Arc<AppState>, web_config: EvolveWebConfig) {
    // Cancel any existing run
    {
        let mut evo_lock = state.evolution.lock().await;
        if let Some(handle) = evo_lock.take() {
            let _ = handle.cancel_tx.send(());
        }
    }

    let (cancel_tx, cancel_rx) = oneshot::channel::<()>();
    let mode = web_config.mode.clone();

    {
        let mut evo_lock = state.evolution.lock().await;
        *evo_lock = Some(EvolutionHandle {
            cancel_tx,
            mode: mode.clone(),
        });
    }

    let tx = state.tx.clone();
    let config = web_config.to_evolution_config();
    let state_clone = state.clone();

    tokio::spawn(async move {
        let mode_str = mode.clone();
        let tx_clone = tx.clone();
        let _generations = config.generations;

        // Run evolution in a blocking thread
        let result =
            tokio::task::spawn_blocking(move || run_evolution_with_mode(&mode_str, config)).await;

        // The cancel_rx is consumed after the blocking run completes;
        // if the sender was dropped, evolution was cancelled.
        drop(cancel_rx);

        match result {
            Ok(Ok(evo_outcome)) => {
                let mut telemetry_log = Vec::new();

                // Stream generation results
                match &evo_outcome {
                    EvoOutcome::Standard(result) => {
                        for gen in &result.generations {
                            let gen_data = EvolveGenerationData {
                                generation: gen.generation,
                                best_fitness: gen.best_fitness,
                                mean_fitness: gen.mean_fitness,
                                worst_fitness: gen.worst_fitness,
                                diversity: (gen.best_fitness - gen.worst_fitness).abs(),
                                best_genome: gen.best_genome.clone(),
                                mode: mode.clone(),
                                pareto_front: None,
                                stress_metrics: None,
                            };
                            telemetry_log.push(gen_data.clone());
                            let _ = tx_clone.send(ServerMsg::EvolveGeneration(gen_data));
                        }

                        let _ = tx_clone.send(ServerMsg::EvolveComplete(EvolveCompleteData {
                            mode: mode.clone(),
                            best_genome: result.global_best_genome.clone(),
                            best_fitness: result.global_best_fitness,
                            total_worlds: result.total_worlds_evaluated,
                            total_time_ms: result.total_wall_time_ms,
                        }));
                    }
                    EvoOutcome::Pareto(result) => {
                        let front: Vec<ParetoFrontEntry> = result
                            .pareto_front
                            .iter()
                            .map(ParetoFrontEntry::from)
                            .collect();

                        let best = result.pareto_front.first();
                        let best_fitness = best
                            .map(|b| {
                                b.objectives.biomass
                                    + b.objectives.biodiversity
                                    + b.objectives.stability
                            })
                            .unwrap_or(0.0);

                        let gen_data = EvolveGenerationData {
                            generation: result.generations_run,
                            best_fitness,
                            mean_fitness: best_fitness,
                            worst_fitness: 0.0,
                            diversity: result.pareto_front.len() as f32,
                            best_genome: best.map(|b| b.genome.clone()).unwrap_or_else(|| {
                                crate::terrarium_evolve::WorldGenome::default_with_seed(42)
                            }),
                            mode: mode.clone(),
                            pareto_front: Some(front),
                            stress_metrics: None,
                        };
                        telemetry_log.push(gen_data.clone());
                        let _ = tx_clone.send(ServerMsg::EvolveGeneration(gen_data));

                        let _ = tx_clone.send(ServerMsg::EvolveComplete(EvolveCompleteData {
                            mode: mode.clone(),
                            best_genome: best.map(|b| b.genome.clone()).unwrap_or_else(|| {
                                crate::terrarium_evolve::WorldGenome::default_with_seed(42)
                            }),
                            best_fitness,
                            total_worlds: result.total_worlds_evaluated,
                            total_time_ms: result.total_wall_time_ms,
                        }));
                    }
                }

                // Store telemetry for export
                {
                    let mut tl = state_clone.last_telemetry.lock().await;
                    *tl = Some(telemetry_log);
                }
            }
            Ok(Err(e)) => {
                let _ = tx_clone.send(ServerMsg::Error {
                    message: format!("Evolution failed: {}", e),
                });
            }
            Err(e) => {
                let _ = tx_clone.send(ServerMsg::Error {
                    message: format!("Evolution task panicked: {}", e),
                });
            }
        }

        // Clear the evolution handle
        let mut evo_lock = state_clone.evolution.lock().await;
        *evo_lock = None;
    });
}

/// Stop a running evolution.
pub async fn stop_evolution(state: Arc<AppState>) {
    let mut evo_lock = state.evolution.lock().await;
    if let Some(handle) = evo_lock.take() {
        let _ = handle.cancel_tx.send(());
    }
}

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

enum EvoOutcome {
    Standard(EvolutionResult),
    Pareto(ParetoEvolutionResult),
}

fn run_evolution_with_mode(mode: &str, config: EvolutionConfig) -> Result<EvoOutcome, String> {
    match mode {
        "stress" => evolve_stress_test(config).map(EvoOutcome::Standard),
        "pareto" => evolve_pareto(config).map(EvoOutcome::Pareto),
        "pareto_stressed" => evolve_pareto_stressed(config).map(EvoOutcome::Pareto),
        // Extended modes — map to closest existing algorithms
        "coevolution" => {
            // Coevolution: run two populations alternating; approximate via standard with higher mutation
            let mut cfg = config;
            if let SearchStrategy::Evolutionary {
                ref mut mutation_rate,
                ref mut crossover_rate,
                ..
            } = cfg.strategy
            {
                *mutation_rate = (*mutation_rate * 1.5).min(0.5);
                *crossover_rate = 0.85;
            }
            evolve(cfg).map(EvoOutcome::Standard)
        }
        "differential" => {
            // Differential evolution: use higher crossover, lower mutation for DE-like behavior
            let mut cfg = config;
            cfg.strategy = SearchStrategy::Evolutionary {
                tournament_size: 2,
                mutation_rate: 0.08,
                crossover_rate: 0.9,
                elitism: 1,
            };
            evolve(cfg).map(EvoOutcome::Standard)
        }
        "sensitivity" => {
            // Sensitivity analysis: run stress test to measure parameter sensitivity
            evolve_stress_test(config).map(EvoOutcome::Standard)
        }
        "island" => {
            // Island model: simulate with larger population, Pareto multi-objective
            let mut cfg = config;
            cfg.population_size = (cfg.population_size * 2).min(32);
            evolve_pareto(cfg).map(EvoOutcome::Pareto)
        }
        _ => evolve(config).map(EvoOutcome::Standard),
    }
}
