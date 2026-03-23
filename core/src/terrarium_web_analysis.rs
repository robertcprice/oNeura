//! Async sensitivity and stress analysis runners for the web server.
//!
//! Streams per-parameter (sensitivity) or per-scenario (stress) results via
//! the broadcast channel, mirroring the pattern from terrarium_web_evolution.

use crate::terrarium_evolve::WorldGenome;
use crate::terrarium_web_protocol::{
    SensitivityCompleteData, SensitivityProgressData, SensitivityWebConfig, ServerMsg,
    StressCompleteData, StressProgressData, StressWebConfig,
};
use crate::terrarium_web_state::AppState;
use std::sync::Arc;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Sensitivity sweep
// ---------------------------------------------------------------------------

/// Continuous parameter ranges for OAT sweep.
fn param_range(name: &str) -> (f32, f32) {
    match name {
        "proton_scale" => (0.3, 3.0),
        "temperature_c" => (10.0, 40.0),
        "water_volume" => (50.0, 300.0),
        "moisture_scale" => (0.5, 2.0),
        "respiration_vmax" => (0.3, 3.0),
        "nitrification_vmax" => (0.3, 3.0),
        "photosynthesis_vmax" => (0.3, 3.0),
        "mineralization_vmax" => (0.3, 3.0),
        "time_warp" => (100.0, 2000.0),
        "psc_scale" => (5.0, 100.0),
        _ => (0.0, 1.0),
    }
}

fn set_genome_param(genome: &mut WorldGenome, name: &str, val: f32) {
    match name {
        "proton_scale" => genome.initial_proton_scale = val,
        "temperature_c" => genome.soil_temperature_c = val,
        "water_volume" => genome.water_volume = val,
        "moisture_scale" => genome.initial_moisture_scale = val,
        "respiration_vmax" => genome.respiration_vmax_scale = val,
        "nitrification_vmax" => genome.nitrification_vmax_scale = val,
        "photosynthesis_vmax" => genome.photosynthesis_vmax_scale = val,
        "mineralization_vmax" => genome.mineralization_vmax_scale = val,
        "time_warp" => genome.time_warp = val,
        "psc_scale" => genome.fly_psc_scale = val,
        _ => {}
    }
}

const SWEEPABLE: &[&str] = &[
    "proton_scale",
    "temperature_c",
    "water_volume",
    "moisture_scale",
    "respiration_vmax",
    "nitrification_vmax",
    "photosynthesis_vmax",
    "mineralization_vmax",
    "time_warp",
    "psc_scale",
];

fn evaluate_fitness(genome: &WorldGenome, frames: usize) -> f32 {
    match genome.build_world_lite() {
        Ok(mut world) => {
            for _ in 0..frames {
                let _ = world.step_frame();
            }
            let snap = world.snapshot();
            snap.total_plant_cells + snap.food_remaining
        }
        Err(_) => 0.0,
    }
}

/// Start a sensitivity sweep in the background, streaming per-param results.
pub async fn start_sensitivity(state: Arc<AppState>, config: SensitivityWebConfig) {
    let tx = state.tx.clone();

    tokio::spawn(async move {
        let tx_clone = tx.clone();

        let result = tokio::task::spawn_blocking(move || {
            let start = Instant::now();
            let resolution = config.resolution.unwrap_or(5).max(2).min(20);
            let frames = config.frames.unwrap_or(30).max(10);
            let seed = config.seed.unwrap_or(42);

            let params: Vec<&str> = if let Some(ref p) = config.param {
                let target: &str = p.as_str();
                SWEEPABLE
                    .iter()
                    .filter(|&&n| n == target)
                    .cloned()
                    .collect()
            } else {
                SWEEPABLE.to_vec()
            };

            let total_params = params.len();
            let mut all_results = Vec::new();

            for (idx, name) in params.iter().enumerate() {
                let (lo, hi) = param_range(name);
                let mut sweep_values = Vec::with_capacity(resolution);
                let mut fitness_values = Vec::with_capacity(resolution);

                for i in 0..resolution {
                    let t = i as f32 / (resolution - 1).max(1) as f32;
                    let val = lo + t * (hi - lo);
                    sweep_values.push(val);
                    let mut genome = WorldGenome::default_with_seed(seed);
                    set_genome_param(&mut genome, name, val);
                    fitness_values.push(evaluate_fitness(&genome, frames));
                }

                let min_f = fitness_values.iter().cloned().fold(f32::INFINITY, f32::min);
                let max_f = fitness_values
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                let mean_f: f32 =
                    fitness_values.iter().sum::<f32>() / fitness_values.len().max(1) as f32;
                let si = if mean_f.abs() > 1e-10 {
                    (max_f - min_f) / mean_f
                } else {
                    0.0
                };

                let progress = SensitivityProgressData {
                    parameter: name.to_string(),
                    sensitivity_index: si,
                    min_fitness: min_f,
                    max_fitness: max_f,
                    mean_fitness: mean_f,
                    sweep_values: sweep_values.clone(),
                    fitness_values: fitness_values.clone(),
                    param_index: idx,
                    total_params,
                };
                all_results.push(progress.clone());
                let _ = tx_clone.send(ServerMsg::SensitivityProgress(progress));
            }

            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

            // Sort by SI descending for the completion message
            all_results.sort_by(|a, b| {
                b.sensitivity_index
                    .partial_cmp(&a.sensitivity_index)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let rankings: Vec<(String, f32)> = all_results
                .iter()
                .map(|r| (r.parameter.clone(), r.sensitivity_index))
                .collect();

            let _ = tx_clone.send(ServerMsg::SensitivityComplete(SensitivityCompleteData {
                total_params,
                elapsed_ms,
                rankings,
            }));
        })
        .await;

        if let Err(e) = result {
            let _ = tx.send(ServerMsg::Error {
                message: format!("Sensitivity analysis panicked: {}", e),
            });
        }
    });
}

// ---------------------------------------------------------------------------
// Stress benchmark
// ---------------------------------------------------------------------------

const SCENARIOS: &[&str] = &[
    "drought",
    "heat_spike",
    "nutrient_depletion",
    "population_crash",
    "invasive_species",
    "combined",
];

fn apply_stress(world: &mut crate::terrarium_world::TerrariumWorld, scenario: &str) {
    match scenario {
        "drought" => {
            for h in world.substrate.hydration.iter_mut() {
                *h = 0.0;
            }
            for ws in &mut world.waters {
                ws.volume = 0.0;
            }
        }
        "heat_spike" => {
            let n_species = 14;
            let n_voxels = world.substrate.current.len() / n_species;
            for v in 0..n_voxels {
                let idx = v * n_species + 12;
                if idx < world.substrate.current.len() {
                    world.substrate.current[idx] *= 10.0;
                }
            }
        }
        "nutrient_depletion" => {
            for val in world.substrate.current.iter_mut() {
                *val *= 0.01;
            }
        }
        "population_crash" => {
            let keep = (world.flies.len() as f32 * 0.2).ceil() as usize;
            world.flies.truncate(keep);
        }
        "invasive_species" => {
            use crate::{FruitSourceState, TerrariumFruitPatch};
            use rand::{Rng, SeedableRng};
            let mut rng = rand::rngs::StdRng::seed_from_u64(999);
            let w = world.config.width;
            let h = world.config.height;
            for _ in 0..10 {
                if world.fruits.len() < world.config.max_fruits {
                    let source = FruitSourceState {
                        x: rng.gen_range(0..w),
                        y: rng.gen_range(0..h),
                        z: 0,
                        attached: false,
                        ripeness: 1.0,
                        sugar_content: 5.0,
                        odorant_emission_rate: 0.01,
                        decay_rate: 0.001,
                        alive: true,
                        odorant_profile: vec![],
                    };
                    let source_genome =
                        crate::terrarium::plant_species::sample_named_plant_genome(&mut rng);
                    let taxonomy_id = source_genome.taxonomy_id;
                    let composition = crate::terrarium::fruit_state::fruit_composition_from_parent(
                        &source_genome,
                        taxonomy_id,
                        source.sugar_content,
                        source.ripeness,
                    );
                    let identity = world.register_organism_identity(
                        crate::terrarium::TerrariumOrganismKind::Fruit,
                        Some(taxonomy_id),
                        None,
                        None,
                        None,
                        None,
                        None,
                    );
                    world.fruits.push(TerrariumFruitPatch {
                        identity,
                        source,
                        taxonomy_id,
                        source_genome,
                        composition,
                        development: None,
                        organ: crate::terrarium::fruit_state::detached_fruit_organ_state(1.0),
                        reproduction: None,
                        radius: 1.5,
                        previous_remaining: 5.0,
                        deposited_all: false,
                        material_inventory: crate::terrarium::RegionalMaterialInventory::new(
                            "fruit:web-analysis".into(),
                        ),
                    });
                }
            }
        }
        "combined" => {
            apply_stress(world, "drought");
            apply_stress(world, "heat_spike");
        }
        _ => {}
    }
}

fn scenario_description(name: &str) -> &'static str {
    match name {
        "drought" => "Moisture -> 0 at frame N/3",
        "heat_spike" => "Temperature +15C at frame 2N/3",
        "nutrient_depletion" => "Dissolved nutrients -> 0 at frame N/3",
        "population_crash" => "Remove 80% of flies at frame N/3",
        "invasive_species" => "Add 10 extra plants at frame N/2",
        "combined" => "Drought + Heat spike simultaneously at frame N/3",
        _ => "Unknown scenario",
    }
}

/// Start a stress benchmark in the background, streaming per-scenario results.
pub async fn start_stress(state: Arc<AppState>, config: StressWebConfig) {
    let tx = state.tx.clone();

    tokio::spawn(async move {
        let tx_clone = tx.clone();

        let result = tokio::task::spawn_blocking(move || {
            let start = Instant::now();
            let frames = config.frames.unwrap_or(60).max(20);
            let seed = config.seed.unwrap_or(42);

            let scenarios: Vec<&str> = if config.scenarios.is_empty() {
                SCENARIOS.to_vec()
            } else {
                config
                    .scenarios
                    .iter()
                    .filter_map(|s: &String| {
                        let target: &str = s.as_str();
                        SCENARIOS.iter().find(|&&sc| sc == target).cloned()
                    })
                    .collect()
            };

            let total_scenarios = scenarios.len();

            for (idx, scenario) in scenarios.iter().enumerate() {
                let sc_start = Instant::now();
                let genome = WorldGenome::default_with_seed(seed);
                let mut world = genome.build_world_lite().expect("Failed to build world");
                let stress_frame = frames / 3;

                // Warmup
                for _ in 0..stress_frame {
                    let _ = world.step_frame();
                }
                let pre_snap = world.snapshot();
                let baseline_biomass = pre_snap.total_plant_cells + pre_snap.food_remaining;

                // Apply stress
                apply_stress(&mut world, scenario);

                // Post-stress measurement
                let mut min_biomass = f32::INFINITY;
                let mut min_frame = stress_frame;
                let mut final_biomass = 0.0f32;
                for f in stress_frame..frames {
                    let _ = world.step_frame();
                    let snap = world.snapshot();
                    let bm = snap.total_plant_cells + snap.food_remaining;
                    if bm < min_biomass {
                        min_biomass = bm;
                        min_frame = f;
                    }
                    final_biomass = bm;
                }
                if min_biomass == f32::INFINITY {
                    min_biomass = final_biomass;
                }
                let recovery_ratio = if baseline_biomass > 1e-6 {
                    final_biomass / baseline_biomass
                } else {
                    0.0
                };

                let progress = StressProgressData {
                    scenario: scenario.to_string(),
                    description: scenario_description(scenario).to_string(),
                    baseline_biomass,
                    min_biomass,
                    final_biomass,
                    recovery_ratio,
                    frames_to_min: min_frame.saturating_sub(stress_frame),
                    elapsed_ms: sc_start.elapsed().as_secs_f64() * 1000.0,
                    scenario_index: idx,
                    total_scenarios,
                };
                let _ = tx_clone.send(ServerMsg::StressProgress(progress));
            }

            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            let _ = tx_clone.send(ServerMsg::StressComplete(StressCompleteData {
                total_scenarios,
                elapsed_ms,
            }));
        })
        .await;

        if let Err(e) = result {
            let _ = tx.send(ServerMsg::Error {
                message: format!("Stress benchmark panicked: {}", e),
            });
        }
    });
}
