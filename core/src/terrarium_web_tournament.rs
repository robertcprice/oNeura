//! Genome tournament / citizen-science leaderboard for the terrarium web app.

use crate::terrarium_evolve::WorldGenome;
use crate::terrarium_web_protocol::ServerMsg;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{broadcast, Mutex};

/// A single tournament entry (submitted genome + evaluation result).
#[derive(Debug, Clone, Serialize)]
pub struct TournamentEntry {
    pub id: u32,
    pub name: String,
    pub genome: WorldGenome,
    pub submitted_at_ms: u64,
    pub evaluated: bool,
    pub fitness: Option<TournamentFitness>,
    pub rank: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TournamentFitness {
    pub biomass: f32,
    pub biodiversity: f32,
    pub stability: f32,
    pub carbon: f32,
    pub fruit: f32,
    pub microbial: f32,
}

/// Evaluation config for fair comparison (all entries use same parameters).
#[derive(Debug, Clone)]
pub struct TournamentEvalConfig {
    pub seed: u64,
    pub frames: usize,
    pub lite: bool,
}

impl Default for TournamentEvalConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            frames: 100,
            lite: true,
        }
    }
}

/// Tournament state: entries + evaluation config.
pub struct TournamentState {
    pub entries: Vec<TournamentEntry>,
    next_id: u32,
    pub eval_config: TournamentEvalConfig,
}

impl TournamentState {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            next_id: 1,
            eval_config: TournamentEvalConfig::default(),
        }
    }
}

/// Submit a genome to the tournament. Spawns async evaluation.
pub async fn submit_genome(
    tournament: Arc<Mutex<TournamentState>>,
    tx: broadcast::Sender<ServerMsg>,
    name: String,
    genome: WorldGenome,
) -> u32 {
    let id;
    let eval_config;
    {
        let mut state = tournament.lock().await;
        id = state.next_id;
        state.next_id += 1;
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        state.entries.push(TournamentEntry {
            id,
            name: name.clone(),
            genome: genome.clone(),
            submitted_at_ms: now_ms,
            evaluated: false,
            fitness: None,
            rank: None,
        });
        eval_config = state.eval_config.clone();
    }

    // Spawn evaluation in background
    let tournament_clone = tournament.clone();
    tokio::task::spawn_blocking(move || evaluate_genome_sync(id, genome, &eval_config))
        .await
        .ok()
        .and_then(|result| result)
        .map(|fitness| {
            let tournament_clone2 = tournament_clone.clone();
            let tx_clone = tx.clone();
            tokio::spawn(async move {
                {
                    let mut state = tournament_clone2.lock().await;
                    if let Some(entry) = state.entries.iter_mut().find(|e| e.id == id) {
                        entry.fitness = Some(fitness);
                        entry.evaluated = true;
                    }
                    recompute_ranks(&mut state.entries);
                }
                // Broadcast update
                let _ = tx_clone.send(ServerMsg::TournamentUpdate(
                    build_leaderboard_data(&tournament_clone2, "biomass").await,
                ));
            });
        });

    id
}

/// Evaluate a genome synchronously (runs on blocking thread).
fn evaluate_genome_sync(
    _id: u32,
    genome: WorldGenome,
    eval_config: &TournamentEvalConfig,
) -> Option<TournamentFitness> {
    // Build and run a world for the configured number of frames
    let mut world = genome.build_world().ok()?;
    for _ in 0..eval_config.frames {
        let _ = world.step_frame();
    }
    let snapshot = world.snapshot();

    Some(TournamentFitness {
        biomass: snapshot.plants as f32 * 10.0
            + snapshot.mean_canopy * 100.0
            + snapshot.total_plant_cells,
        biodiversity: (snapshot.plants as f32).sqrt()
            + (snapshot.mean_microbes / 50.0).min(1.0)
            + (snapshot.flies as f32 * 5.0),
        stability: snapshot.mean_cell_vitality * 100.0,
        carbon: snapshot.mean_atmospheric_co2 * 1000.0,
        fruit: snapshot.fruits as f32 * 20.0 + snapshot.food_remaining,
        microbial: snapshot.mean_microbes + snapshot.mean_symbionts,
    })
}

/// Recompute NSGA-II style Pareto ranks across all evaluated entries.
fn recompute_ranks(entries: &mut [TournamentEntry]) {
    let evaluated: Vec<usize> = entries
        .iter()
        .enumerate()
        .filter(|(_, e)| e.evaluated && e.fitness.is_some())
        .map(|(i, _)| i)
        .collect();

    if evaluated.is_empty() {
        return;
    }

    // Simple Pareto dominance ranking
    let n = evaluated.len();
    let mut ranks = vec![0usize; n];
    let mut assigned = vec![false; n];

    let mut current_rank = 0;
    let mut remaining = n;

    while remaining > 0 {
        // Find non-dominated set
        let mut front = Vec::new();
        for i in 0..n {
            if assigned[i] {
                continue;
            }
            let fi = entries[evaluated[i]].fitness.as_ref().unwrap();
            let dominated = (0..n).any(|j| {
                if i == j || assigned[j] {
                    return false;
                }
                let fj = entries[evaluated[j]].fitness.as_ref().unwrap();
                dominates(fj, fi)
            });
            if !dominated {
                front.push(i);
            }
        }

        if front.is_empty() {
            // Safety: assign remaining to current rank
            for i in 0..n {
                if !assigned[i] {
                    ranks[i] = current_rank;
                    assigned[i] = true;
                }
            }
            break;
        }

        for &idx in &front {
            ranks[idx] = current_rank;
            assigned[idx] = true;
            remaining -= 1;
        }
        current_rank += 1;
    }

    // Apply ranks back
    for (local_idx, &global_idx) in evaluated.iter().enumerate() {
        entries[global_idx].rank = Some(ranks[local_idx]);
    }
}

/// Returns true if `a` Pareto-dominates `b` (all >= and at least one >).
fn dominates(a: &TournamentFitness, b: &TournamentFitness) -> bool {
    let objs_a = [
        a.biomass,
        a.biodiversity,
        a.stability,
        a.carbon,
        a.fruit,
        a.microbial,
    ];
    let objs_b = [
        b.biomass,
        b.biodiversity,
        b.stability,
        b.carbon,
        b.fruit,
        b.microbial,
    ];

    let mut all_ge = true;
    let mut any_gt = false;
    for i in 0..6 {
        if objs_a[i] < objs_b[i] {
            all_ge = false;
            break;
        }
        if objs_a[i] > objs_b[i] {
            any_gt = true;
        }
    }
    all_ge && any_gt
}

/// Build leaderboard data for the frontend.
pub async fn build_leaderboard_data(
    tournament: &Arc<Mutex<TournamentState>>,
    sort_by: &str,
) -> TournamentUpdateData {
    let state = tournament.lock().await;
    let mut entries: Vec<TournamentLeaderboardEntry> = state
        .entries
        .iter()
        .map(|e| TournamentLeaderboardEntry {
            id: e.id,
            name: e.name.clone(),
            fitness: e.fitness.clone(),
            rank: e.rank,
            evaluated: e.evaluated,
        })
        .collect();

    // Sort by requested field
    entries.sort_by(|a, b| {
        let fa = a.fitness.as_ref();
        let fb = b.fitness.as_ref();
        let va = fa
            .map(|f| match sort_by {
                "biodiversity" => f.biodiversity,
                "stability" => f.stability,
                "carbon" => f.carbon,
                "fruit" => f.fruit,
                "microbial" => f.microbial,
                _ => f.biomass,
            })
            .unwrap_or(0.0);
        let vb = fb
            .map(|f| match sort_by {
                "biodiversity" => f.biodiversity,
                "stability" => f.stability,
                "carbon" => f.carbon,
                "fruit" => f.fruit,
                "microbial" => f.microbial,
                _ => f.biomass,
            })
            .unwrap_or(0.0);
        vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
    });

    TournamentUpdateData { entries }
}

/// Get a genome by ID.
pub async fn get_genome(tournament: &Arc<Mutex<TournamentState>>, id: u32) -> Option<WorldGenome> {
    let state = tournament.lock().await;
    state
        .entries
        .iter()
        .find(|e| e.id == id)
        .map(|e| e.genome.clone())
}

/// Delete an entry by ID.
pub async fn delete_entry(tournament: &Arc<Mutex<TournamentState>>, id: u32) -> bool {
    let mut state = tournament.lock().await;
    let before = state.entries.len();
    state.entries.retain(|e| e.id != id);
    let removed = state.entries.len() < before;
    if removed {
        recompute_ranks(&mut state.entries);
    }
    removed
}

// ---------------------------------------------------------------------------
// Protocol types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct TournamentUpdateData {
    pub entries: Vec<TournamentLeaderboardEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TournamentLeaderboardEntry {
    pub id: u32,
    pub name: String,
    pub fitness: Option<TournamentFitness>,
    pub rank: Option<usize>,
    pub evaluated: bool,
}

/// Request body for submitting a genome.
#[derive(Debug, Deserialize)]
pub struct TournamentSubmitRequest {
    pub name: String,
    pub genome: WorldGenome,
}
