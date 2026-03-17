//! Shared application state for the terrarium web server.

use crate::terrarium_web_auth::AuthState;
use crate::terrarium_web_protocol::{
    EcologyEventData, EvolveGenerationData, EntityData, EntityPos, FlyEntity, ServerMsg,
};
use crate::terrarium_web_tournament::TournamentState;
use crate::terrarium_world::{EcologyTelemetryEvent, TerrariumTopdownView, TerrariumWorld, TerrariumWorldConfig};
use std::sync::Arc;
use tokio::sync::{broadcast, Mutex, RwLock};

/// Simulation parameters adjustable at runtime.
pub struct SimParams {
    pub paused: bool,
    pub target_fps: u32,
    pub view: TerrariumTopdownView,
    pub seed: u64,
}

impl Default for SimParams {
    fn default() -> Self {
        Self {
            paused: false,
            target_fps: 10,
            view: TerrariumTopdownView::Terrain,
            seed: 42,
        }
    }
}

/// Handle for a running evolution task (allows cancellation).
pub struct EvolutionHandle {
    pub cancel_tx: tokio::sync::oneshot::Sender<()>,
    pub mode: String,
}

/// Shared application state.
pub struct AppState {
    pub world: Mutex<TerrariumWorld>,
    pub params: RwLock<SimParams>,
    pub tx: broadcast::Sender<ServerMsg>,
    pub evolution: Mutex<Option<EvolutionHandle>>,
    pub frame_count: Mutex<u64>,
    /// Last evolution telemetry for export.
    pub last_telemetry: Mutex<Option<Vec<EvolveGenerationData>>>,
    /// Tournament state.
    pub tournament: Arc<Mutex<TournamentState>>,
    /// Authentication state.
    pub auth: Mutex<AuthState>,
}

impl AppState {
    /// Create new application state from a seed.
    pub fn new(seed: u64, broadcast_capacity: usize, require_auth: bool) -> Result<Arc<Self>, String> {
        let config = TerrariumWorldConfig {
            width: 20,
            height: 16,
            depth: 2,
            seed,
            time_warp: 900.0,
            max_plants: 20,
            max_fruits: 16,
            ..TerrariumWorldConfig::default()
        };

        let mut world = TerrariumWorld::new(config)?;

        // Seed default entities
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // 3 water sources
        for _ in 0..3 {
            let x = rng.gen_range(1..19);
            let y = rng.gen_range(1..15);
            world.add_water(x, y, 150.0, 0.0008);
        }

        // 6 plants
        for _ in 0..6 {
            let x = rng.gen_range(1..19);
            let y = rng.gen_range(1..15);
            let _ = world.add_plant(x, y, None, None);
        }

        // 3 fruits
        for _ in 0..3 {
            let x = rng.gen_range(1..19);
            let y = rng.gen_range(1..15);
            world.add_fruit(x, y, 0.8, None);
        }

        // 2 flies
        for i in 0..2 {
            let x = rng.gen_range(2.0..18.0_f32);
            let y = rng.gen_range(2.0..14.0_f32);
            world.add_fly(
                crate::drosophila::DrosophilaScale::Tiny,
                x,
                y,
                seed.wrapping_add(i as u64),
            );
        }

        let (tx, _) = broadcast::channel(broadcast_capacity);

        Ok(Arc::new(Self {
            world: Mutex::new(world),
            params: RwLock::new(SimParams {
                seed,
                ..Default::default()
            }),
            tx,
            evolution: Mutex::new(None),
            frame_count: Mutex::new(0),
            last_telemetry: Mutex::new(None),
            tournament: Arc::new(Mutex::new(TournamentState::new())),
            auth: Mutex::new(AuthState::new(require_auth)),
        }))
    }

    /// Extract entity positions from the world (call while holding world lock).
    pub fn extract_entities(world: &TerrariumWorld) -> EntityData {
        let energy_charges = world.fly_energy_charges();

        let plants: Vec<EntityPos> = world
            .plants
            .iter()
            .map(|p| EntityPos {
                x: p.x as f32 + 0.5,
                y: p.y as f32 + 0.5,
            })
            .collect();

        let flies: Vec<FlyEntity> = world
            .flies
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let body = f.body_state();
                FlyEntity {
                    x: body.x,
                    y: body.y,
                    heading: body.heading,
                    energy_frac: energy_charges.get(i).copied().unwrap_or(0.5),
                }
            })
            .collect();

        let fruits: Vec<EntityPos> = world
            .fruits
            .iter()
            .map(|f| EntityPos {
                x: f.source.x as f32 + 0.5,
                y: f.source.y as f32 + 0.5,
            })
            .collect();

        let waters: Vec<EntityPos> = world
            .waters
            .iter()
            .map(|w| EntityPos {
                x: w.x as f32 + 0.5,
                y: w.y as f32 + 0.5,
            })
            .collect();

        let seeds: Vec<EntityPos> = world
            .seeds
            .iter()
            .map(|s| EntityPos {
                x: s.x as f32 + 0.5,
                y: s.y as f32 + 0.5,
            })
            .collect();

        let ecology_events: Vec<EcologyEventData> = world
            .recent_ecology_events()
            .iter()
            .map(|evt| match evt {
                EcologyTelemetryEvent::FlyAtpCrash { x, y, energy_charge, .. } => EcologyEventData {
                    event_type: "atp_crash".into(), x: *x, y: *y,
                    detail: format!("EC={:.3}", energy_charge),
                },
                EcologyTelemetryEvent::FlyStarvationOnset { x, y, trehalose_mm, .. } => EcologyEventData {
                    event_type: "starvation".into(), x: *x, y: *y,
                    detail: format!("Tre={:.2}mM", trehalose_mm),
                },
                EcologyTelemetryEvent::FlyFeeding { x, y, sugar_ingested_mg, .. } => EcologyEventData {
                    event_type: "feeding".into(), x: *x, y: *y,
                    detail: format!("+{:.2}mg", sugar_ingested_mg),
                },
                EcologyTelemetryEvent::FlyEclosed { x, y } => EcologyEventData {
                    event_type: "eclosed".into(), x: *x, y: *y,
                    detail: "New adult".into(),
                },
                EcologyTelemetryEvent::FlyHypoxiaOnset { x, y, ambient_o2, altitude } => EcologyEventData {
                    event_type: "hypoxia".into(), x: *x, y: *y,
                    detail: format!("O2={:.1}% alt={:.1}", ambient_o2 * 100.0, altitude),
                },
            })
            .collect();

        EntityData {
            plants,
            flies,
            fruits,
            waters,
            seeds,
            ecology_events,
        }
    }
}
