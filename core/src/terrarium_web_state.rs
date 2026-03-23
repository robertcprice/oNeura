//! Shared application state for the terrarium web server.

use crate::botany::visual_phenotype::MolecularVisualState;
use crate::drosophila_population::FlyLifeStage;
use crate::ecosystem_integration::IntegratedEcosystem;
use crate::organism_metabolism::OrganismMetabolism;
use crate::terrarium::visual_projection::{
    fly_visual_response, plant_visual_response_with_molecular_state, sample_visual_air,
    sample_visual_chemistry, water_visual_response_with_chemistry,
};
use crate::terrarium::TerrariumDemoPreset;
use crate::terrarium_web_auth::AuthState;
use crate::terrarium_web_protocol::{
    EarthwormMarker, EcologyEventData, EntityData, EntityPos, EvolveGenerationData, FlyEggEntity,
    FlyEmbryoEntity, FlyEntity, FlyImmatureEntity, NematodeMarker, ServerMsg, SnapshotHistoryData,
    SoilSurfaceMarker, TerrariumSnapshotHistoryPoint,
};
use crate::terrarium_web_tournament::TournamentState;
use crate::terrarium_world::{EcologyTelemetryEvent, TerrariumTopdownView, TerrariumWorld};
use std::sync::Arc;
use tokio::sync::{broadcast, Mutex, RwLock};

/// Simulation parameters adjustable at runtime.
pub struct SimParams {
    pub paused: bool,
    pub target_fps: u32,
    pub view: TerrariumTopdownView,
    pub seed: u64,
    pub preset: TerrariumDemoPreset,
    /// Number of simulation steps per rendered frame (time scale).
    /// 1 = base rate, 4 = 4x speed, etc.
    pub steps_per_frame: u32,
}

impl Default for SimParams {
    fn default() -> Self {
        Self {
            paused: false,
            target_fps: 16,
            view: TerrariumTopdownView::Terrain,
            seed: 42,
            preset: TerrariumDemoPreset::MicroTerrarium,
            steps_per_frame: 1,
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
    pub snapshot_history: Mutex<Vec<TerrariumSnapshotHistoryPoint>>,
    /// Last evolution telemetry for export.
    pub last_telemetry: Mutex<Option<Vec<EvolveGenerationData>>>,
    /// Tournament state.
    pub tournament: Arc<Mutex<TournamentState>>,
    /// Authentication state.
    pub auth: Mutex<AuthState>,
    /// Integrated ecosystem simulation (11-module orchestrator).
    pub ecosystem: Mutex<Option<IntegratedEcosystem>>,
}

impl AppState {
    /// Create new application state from a seed.
    pub fn new(
        seed: u64,
        broadcast_capacity: usize,
        require_auth: bool,
    ) -> Result<Arc<Self>, String> {
        Self::new_with_preset(
            seed,
            broadcast_capacity,
            require_auth,
            TerrariumDemoPreset::MicroTerrarium,
        )
    }

    pub fn new_with_preset(
        seed: u64,
        broadcast_capacity: usize,
        require_auth: bool,
        preset: TerrariumDemoPreset,
    ) -> Result<Arc<Self>, String> {
        let world = Self::build_world(seed, preset)?;
        Ok(Self::new_from_world(
            world,
            broadcast_capacity,
            require_auth,
            preset,
        ))
    }

    pub fn new_from_world(
        world: TerrariumWorld,
        broadcast_capacity: usize,
        require_auth: bool,
        preset: TerrariumDemoPreset,
    ) -> Arc<Self> {
        let initial_snapshot = world.snapshot();
        let seed = world.seed_provenance().seed;
        let initial_history = vec![snapshot_history_point(
            &initial_snapshot,
            seed,
            preset.cli_name(),
        )];

        let (tx, _) = broadcast::channel(broadcast_capacity);

        Arc::new(Self {
            world: Mutex::new(world),
            params: RwLock::new(SimParams {
                seed,
                preset,
                ..Default::default()
            }),
            tx,
            evolution: Mutex::new(None),
            frame_count: Mutex::new(0),
            snapshot_history: Mutex::new(initial_history),
            last_telemetry: Mutex::new(None),
            tournament: Arc::new(Mutex::new(TournamentState::new())),
            auth: Mutex::new(AuthState::new(require_auth)),
            ecosystem: Mutex::new(None),
        })
    }

    pub fn build_world(seed: u64, preset: TerrariumDemoPreset) -> Result<TerrariumWorld, String> {
        TerrariumWorld::demo_preset(seed, true, preset)
    }

    pub async fn install_world(
        &self,
        world: TerrariumWorld,
        preset_hint: Option<TerrariumDemoPreset>,
        paused: bool,
    ) {
        let initial_snapshot = world.snapshot();
        let seed = world.seed_provenance().seed;
        let inferred_preset = preset_hint
            .or_else(|| TerrariumDemoPreset::infer_from_config(&world.config))
            .unwrap_or(TerrariumDemoPreset::Demo);
        let initial_history = vec![snapshot_history_point(
            &initial_snapshot,
            seed,
            inferred_preset.cli_name(),
        )];

        {
            let mut world_lock = self.world.lock().await;
            *world_lock = world;
        }

        {
            let mut params = self.params.write().await;
            params.seed = seed;
            params.preset = inferred_preset;
            params.paused = paused;
        }

        {
            let mut frame_count = self.frame_count.lock().await;
            *frame_count = 0;
        }

        {
            let mut history = self.snapshot_history.lock().await;
            *history = initial_history;
        }

        {
            let mut telemetry = self.last_telemetry.lock().await;
            *telemetry = None;
        }

        {
            let mut evolution = self.evolution.lock().await;
            if let Some(handle) = evolution.take() {
                let _ = handle.cancel_tx.send(());
            }
        }

        {
            let mut ecosystem = self.ecosystem.lock().await;
            *ecosystem = None;
        }
    }

    pub async fn record_snapshot_history(
        &self,
        snapshot: &crate::terrarium_world::TerrariumWorldSnapshot,
        seed: u64,
        preset: &str,
    ) -> SnapshotHistoryData {
        let point = snapshot_history_point(snapshot, seed, preset);
        let mut history = self.snapshot_history.lock().await;
        push_snapshot_history_point(&mut history, point);
        SnapshotHistoryData {
            history: history.clone(),
            preset: preset.to_string(),
            seed,
        }
    }

    pub async fn snapshot_history_data(&self) -> SnapshotHistoryData {
        let params = self.params.read().await;
        let history = self.snapshot_history.lock().await.clone();
        SnapshotHistoryData {
            history,
            preset: params.preset.cli_name().to_string(),
            seed: params.seed,
        }
    }

    /// Extract entity positions from the world (call while holding world lock).
    pub fn extract_entities(world: &TerrariumWorld) -> EntityData {
        let atmosphere = world.atmosphere_frame();
        let width = world.config.width;
        let height = world.config.height;
        let time_s = world.time_s;
        let energy_charges: Vec<f32> = world
            .fly_metabolisms
            .iter()
            .map(|metabolism| metabolism.energy_charge())
            .collect();

        let plants: Vec<EntityPos> = world
            .plants
            .iter()
            .map(|p| EntityPos {
                x: p.x as f32 + 0.5,
                y: p.y as f32 + 0.5,
            })
            .collect();

        let full_plants: Vec<crate::terrarium_world::TerrariumPlantSnapshot> = world
            .plants
            .iter()
            .map(|p| {
                let estimated_fruit_load = p
                    .physiology
                    .fruit_count()
                    .min((p.physiology.storage_carbon().max(0.0) * 5.0).round() as u32);
                crate::terrarium_world::TerrariumPlantSnapshot {
                    x: p.x,
                    y: p.y,
                    organism_id: p.identity.organism_id,
                    phylo_id: p.identity.phylo_id,
                    lineage_generation: p.identity.generation,
                    parent_organism_id: p.identity.parent_organism_id,
                    co_parent_organism_id: p.identity.co_parent_organism_id,
                    display_name: p.identity.display_name.clone(),
                    taxonomy_id: p.genome.taxonomy_id,
                    common_name: crate::terrarium::plant_species::plant_common_name(
                        p.genome.taxonomy_id,
                    )
                    .to_string(),
                    scientific_name: crate::terrarium::plant_species::plant_scientific_name(
                        p.genome.taxonomy_id,
                    )
                    .to_string(),
                    growth_form: crate::terrarium::plant_species::plant_growth_form(
                        p.genome.taxonomy_id,
                    ),
                    height_mm: p.physiology.height_mm(),
                    vitality: p.cellular.vitality(),
                    storage_carbon: p.physiology.storage_carbon(),
                    fruit_load: estimated_fruit_load,
                    structure:
                        crate::terrarium::shape_projection::structure_descriptor_from_morphology(
                            &p.morphology,
                        ),
                    morphology: p.morphology.generate_nodes_with_context(
                        estimated_fruit_load,
                        p.cellular.vitality(),
                    ),
                    branch_mesh: None,
                }
            })
            .collect();

        let plant_visuals = world
            .plants
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let local_air = sample_visual_air(
                    &atmosphere,
                    width,
                    height,
                    p.x as f32 + 0.5,
                    p.y as f32 + 0.5,
                );
                let gene_snapshot = p.botanical_genome.expression_snapshot();
                let molecular_visual = MolecularVisualState::from_metabolome(
                    &p.metabolome,
                    &gene_snapshot,
                    p.physiology.leaf_biomass() + p.physiology.stem_biomass(),
                );
                plant_visual_response_with_molecular_state(
                    p.genome.taxonomy_id,
                    local_air,
                    p.cellular.vitality(),
                    (p.cellular.total_cells() * 0.01).clamp(0.0, 1.0),
                    time_s,
                    i as f32,
                    Some(molecular_visual),
                )
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
                    z: body.z,
                    heading: body.heading,
                    is_flying: body.is_flying,
                    wing_beat_freq: body.wing_beat_freq,
                    energy_frac: energy_charges.get(i).copied().unwrap_or(0.5),
                }
            })
            .collect();

        let cell_size_mm = world.config.cell_size_mm.max(1.0e-3);
        let fly_eggs: Vec<FlyEggEntity> = world
            .fly_population()
            .egg_clusters
            .iter()
            .map(|cluster| FlyEggEntity {
                x: cluster.position.0 / cell_size_mm,
                y: cluster.position.1 / cell_size_mm,
                count: cluster.count,
                age_hours: cluster.age_hours,
                substrate_quality: cluster.substrate_quality,
            })
            .collect();

        let fly_embryos: Vec<FlyEmbryoEntity> = world
            .fly_population()
            .iter_cluster_embryos()
            .map(|(cluster_index, _, cluster, embryo)| {
                let position = cluster.embryo_position_mm(embryo);
                FlyEmbryoEntity {
                    id: embryo.id,
                    x: position.0 / cell_size_mm,
                    y: position.1 / cell_size_mm,
                    age_hours: embryo.age_hours,
                    viability: embryo.viability,
                    sex: format!("{:?}", embryo.sex),
                    cluster_index,
                }
            })
            .collect();

        let fly_larvae: Vec<FlyImmatureEntity> = world
            .fly_population()
            .flies
            .iter()
            .filter_map(|fly| match fly.stage {
                FlyLifeStage::Larva { instar, age_hours } => Some(FlyImmatureEntity {
                    x: fly.position.0 / cell_size_mm,
                    y: fly.position.1 / cell_size_mm,
                    z: fly.position.2 / cell_size_mm,
                    age_hours,
                    energy_frac: (fly.energy / crate::drosophila_population::FLY_ENERGY_MAX)
                        .clamp(0.0, 1.0),
                    sex: format!("{:?}", fly.sex),
                    instar: Some(instar),
                }),
                _ => None,
            })
            .collect();

        let fly_pupae: Vec<FlyImmatureEntity> = world
            .fly_population()
            .flies
            .iter()
            .filter_map(|fly| match fly.stage {
                FlyLifeStage::Pupa { age_hours } => Some(FlyImmatureEntity {
                    x: fly.position.0 / cell_size_mm,
                    y: fly.position.1 / cell_size_mm,
                    z: fly.position.2 / cell_size_mm,
                    age_hours,
                    energy_frac: (fly.energy / crate::drosophila_population::FLY_ENERGY_MAX)
                        .clamp(0.0, 1.0),
                    sex: format!("{:?}", fly.sex),
                    instar: None,
                }),
                _ => None,
            })
            .collect();

        let fly_visuals = world
            .flies
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let body = f.body_state();
                let local_air = sample_visual_air(&atmosphere, width, height, body.x, body.y);
                fly_visual_response(
                    local_air,
                    body,
                    energy_charges.get(i).copied().unwrap_or(0.5),
                    time_s,
                    i as f32,
                )
            })
            .collect();

        let earthworms = world
            .earthworm_visual_markers()
            .into_iter()
            .map(|(x, y, visual)| EarthwormMarker {
                x: x as f32 + 0.5,
                y: y as f32 + 0.5,
                visual,
            })
            .collect();

        let nematodes = world
            .nematode_visual_markers()
            .into_iter()
            .map(|(x, y, visual)| NematodeMarker {
                x: x as f32 + 0.5,
                y: y as f32 + 0.5,
                visual,
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

        let full_fruits: Vec<crate::terrarium_world::TerrariumFruitSnapshot> = world
            .fruits
            .iter()
            .map(|fruit| crate::terrarium_world::TerrariumFruitSnapshot {
                x: fruit.source.x,
                y: fruit.source.y,
                organism_id: fruit.identity.organism_id,
                lineage_generation: fruit.identity.generation,
                parent_organism_id: fruit.identity.parent_organism_id,
                display_name: fruit.identity.display_name.clone(),
                taxonomy_id: fruit.taxonomy_id,
                common_name: crate::terrarium::plant_species::plant_common_name(fruit.taxonomy_id)
                    .to_string(),
                scientific_name: crate::terrarium::plant_species::plant_scientific_name(
                    fruit.taxonomy_id,
                )
                .to_string(),
                growth_form: crate::terrarium::plant_species::plant_growth_form(fruit.taxonomy_id),
                shape: crate::terrarium::shape_projection::fruit_shape_descriptor(
                    &fruit.source_genome,
                    fruit.radius,
                    fruit.source.ripeness,
                    fruit.source.sugar_content,
                ),
                sugar_content: fruit.source.sugar_content,
                ripeness: fruit.source.ripeness,
                radius: fruit.radius,
                attached: fruit.source.attached,
                alive: fruit.source.alive,
            })
            .collect();

        let fruit_visuals = world.fruit_visuals();

        let waters: Vec<EntityPos> = world
            .waters
            .iter()
            .map(|w| EntityPos {
                x: w.x as f32 + 0.5,
                y: w.y as f32 + 0.5,
            })
            .collect();

        let water_cycle = crate::terrarium::visual_projection::TerrariumWaterCycleInputs {
            lunar_phase: world.lunar_phase(),
            moonlight: world.moonlight(),
            tidal_moisture_factor: world.tidal_moisture_factor(),
        };
        let water_visuals = world
            .waters
            .iter()
            .enumerate()
            .map(|(i, w)| {
                let local_air = sample_visual_air(
                    &atmosphere,
                    width,
                    height,
                    w.x as f32 + 0.5,
                    w.y as f32 + 0.5,
                );
                let chemistry = sample_visual_chemistry(&world, w.x, w.y);
                water_visual_response_with_chemistry(
                    local_air,
                    w.volume,
                    time_s,
                    i as f32,
                    water_cycle,
                    chemistry,
                )
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

        let full_seeds: Vec<crate::terrarium_world::TerrariumSeedSnapshot> = world
            .seeds
            .iter()
            .map(|seed| crate::terrarium_world::TerrariumSeedSnapshot {
                x: seed.x,
                y: seed.y,
                organism_id: seed.identity.organism_id,
                phylo_id: seed.identity.phylo_id,
                lineage_generation: seed.identity.generation,
                parent_organism_id: seed.identity.parent_organism_id,
                co_parent_organism_id: seed.identity.co_parent_organism_id,
                display_name: seed.identity.display_name.clone(),
                taxonomy_id: seed.genome.taxonomy_id,
                common_name: crate::terrarium::plant_species::plant_common_name(
                    seed.genome.taxonomy_id,
                )
                .to_string(),
                scientific_name: crate::terrarium::plant_species::plant_scientific_name(
                    seed.genome.taxonomy_id,
                )
                .to_string(),
                growth_form: crate::terrarium::plant_species::plant_growth_form(
                    seed.genome.taxonomy_id,
                ),
                shape: crate::terrarium::shape_projection::seed_shape_descriptor(
                    &seed.genome,
                    seed.reserve_carbon,
                    seed.dormancy_s,
                ),
                reserve_carbon: seed.reserve_carbon,
                dormancy_s: seed.dormancy_s,
                age_s: seed.age_s,
                burial_depth_mm: seed.microsite.burial_depth_mm,
                surface_exposure: seed.microsite.surface_exposure,
            })
            .collect();

        let seed_visuals = world.seed_visuals();

        let soil_surface = world
            .soil_surface_markers()
            .into_iter()
            .map(|(x, y, visual)| SoilSurfaceMarker {
                x: x as f32 + 0.5,
                y: y as f32 + 0.5,
                visual,
            })
            .collect();

        let ecology_events: Vec<EcologyEventData> = world
            .recent_ecology_events()
            .iter()
            .map(|evt| match evt {
                EcologyTelemetryEvent::FlyAtpCrash {
                    x,
                    y,
                    energy_charge,
                    ..
                } => EcologyEventData {
                    event_type: "atp_crash".into(),
                    x: *x,
                    y: *y,
                    detail: format!("EC={:.3}", energy_charge),
                },
                EcologyTelemetryEvent::FlyStarvationOnset {
                    x, y, trehalose_mm, ..
                } => EcologyEventData {
                    event_type: "starvation".into(),
                    x: *x,
                    y: *y,
                    detail: format!("Tre={:.2}mM", trehalose_mm),
                },
                EcologyTelemetryEvent::FlyFeeding {
                    x,
                    y,
                    sugar_ingested_mg,
                    ..
                } => EcologyEventData {
                    event_type: "feeding".into(),
                    x: *x,
                    y: *y,
                    detail: format!("+{:.2}mg", sugar_ingested_mg),
                },
                EcologyTelemetryEvent::FlyEclosed { x, y } => EcologyEventData {
                    event_type: "eclosed".into(),
                    x: *x,
                    y: *y,
                    detail: "New adult".into(),
                },
                EcologyTelemetryEvent::FlyHypoxiaOnset {
                    x,
                    y,
                    ambient_o2,
                    altitude,
                } => EcologyEventData {
                    event_type: "hypoxia".into(),
                    x: *x,
                    y: *y,
                    detail: format!("O2={:.1}% alt={:.1}", ambient_o2 * 100.0, altitude),
                },
                EcologyTelemetryEvent::ExplicitPromotion {
                    x,
                    y,
                    guild,
                    represented_cells,
                    ..
                } => EcologyEventData {
                    event_type: "promotion".into(),
                    x: *x as f32,
                    y: *y as f32,
                    detail: format!("guild={} cells={:.1}", guild, represented_cells),
                },
                EcologyTelemetryEvent::ExplicitDemotion {
                    x,
                    y,
                    represented_cells,
                    atp_mm,
                    ..
                } => EcologyEventData {
                    event_type: "demotion".into(),
                    x: *x as f32,
                    y: *y as f32,
                    detail: format!("cells={:.1} ATP={:.2}mM", represented_cells, atp_mm),
                },
                EcologyTelemetryEvent::ExplicitDeath {
                    x,
                    y,
                    reason,
                    represented_cells,
                    ..
                } => EcologyEventData {
                    event_type: "death".into(),
                    x: *x as f32,
                    y: *y as f32,
                    detail: format!("{} cells={:.1}", reason, represented_cells),
                },
                EcologyTelemetryEvent::CellDivision {
                    x,
                    y,
                    parent_represented_cells,
                    daughter_represented_cells,
                    ..
                } => EcologyEventData {
                    event_type: "division".into(),
                    x: *x as f32,
                    y: *y as f32,
                    detail: format!(
                        "parent={:.1}→daughter={:.1}",
                        parent_represented_cells, daughter_represented_cells
                    ),
                },
                EcologyTelemetryEvent::CellDivisionDaughter {
                    x,
                    y,
                    represented_cells,
                    ..
                } => EcologyEventData {
                    event_type: "division_daughter".into(),
                    x: *x as f32,
                    y: *y as f32,
                    detail: format!("cells={:.1}", represented_cells),
                },
                EcologyTelemetryEvent::PacketPopulationSeed { x, y } => EcologyEventData {
                    event_type: "packet_seed".into(),
                    x: *x as f32,
                    y: *y as f32,
                    detail: "Packet seeded".into(),
                },
                EcologyTelemetryEvent::PacketPromotion {
                    x,
                    y,
                    activity,
                    represented_cells,
                    ..
                } => EcologyEventData {
                    event_type: "packet_promotion".into(),
                    x: *x as f32,
                    y: *y as f32,
                    detail: format!("activity={:.2} cells={:.1}", activity, represented_cells),
                },
                EcologyTelemetryEvent::FlyGrazing {
                    x,
                    y,
                    leaf_consumed,
                    plant_deterrence,
                } => EcologyEventData {
                    event_type: "grazing".into(),
                    x: *x,
                    y: *y,
                    detail: format!("leaf={:.4} det={:.2}", leaf_consumed, plant_deterrence),
                },
                EcologyTelemetryEvent::LightningStrike {
                    x,
                    y,
                    ammonium_deposited,
                    nitrate_deposited,
                } => EcologyEventData {
                    event_type: "lightning".into(),
                    x: *x,
                    y: *y,
                    detail: format!(
                        "NH4={:.3} NO3={:.3}",
                        ammonium_deposited, nitrate_deposited
                    ),
                },
                EcologyTelemetryEvent::ExtremeEventOnset {
                    event_type,
                    severity,
                } => EcologyEventData {
                    event_type: format!("extreme_{event_type}"),
                    x: world.config.width as f32 / 2.0,
                    y: world.config.height as f32 / 2.0,
                    detail: format!("{event_type} sev={severity:.2}"),
                },
            })
            .collect();

        EntityData {
            plants,
            full_plants, // needed by browser for tree morphology rendering
            plant_visuals,
            flies,
            fly_eggs,
            fly_embryos,
            fly_larvae,
            fly_pupae,
            fly_visuals,
            earthworms,
            nematodes,
            fruits,
            full_fruits,
            fruit_visuals,
            waters,
            water_visuals,
            seeds,
            full_seeds,
            seed_visuals,
            soil_surface,
            ecology_events,
        }
    }
}

const SNAPSHOT_HISTORY_LIMIT: usize = 180;

fn mean_embryo_component<F>(
    embryos: &[crate::terrarium_world::TerrariumFlyEmbryoSnapshot],
    sample: F,
) -> Option<f32>
where
    F: Fn(&crate::terrarium_world::TerrariumFlyEmbryoSnapshot) -> f32,
{
    if embryos.is_empty() {
        return None;
    }
    Some(embryos.iter().map(sample).sum::<f32>() / embryos.len() as f32)
}

pub fn snapshot_history_point(
    snapshot: &crate::terrarium_world::TerrariumWorldSnapshot,
    seed: u64,
    preset: &str,
) -> TerrariumSnapshotHistoryPoint {
    let adults = snapshot
        .full_fly_population
        .iter()
        .filter(|fly| fly.stage.eq_ignore_ascii_case("adult"))
        .count() as u32;
    TerrariumSnapshotHistoryPoint {
        time_s: snapshot.time_s,
        preset: preset.to_string(),
        seed,
        adults: adults.max(snapshot.flies as u32),
        eggs: snapshot.fly_population_eggs,
        embryos: snapshot.fly_population_embryos,
        larvae: snapshot.fly_population_larvae,
        pupae: snapshot.fly_population_pupae,
        embryo_viability: mean_embryo_component(&snapshot.full_fly_embryos, |embryo| {
            embryo.viability
        }),
        embryo_glucose: mean_embryo_component(&snapshot.full_fly_embryos, |embryo| embryo.glucose),
        embryo_nucleotides: mean_embryo_component(&snapshot.full_fly_embryos, |embryo| {
            embryo.nucleotides
        }),
    }
}

pub fn push_snapshot_history_point(
    history: &mut Vec<TerrariumSnapshotHistoryPoint>,
    point: TerrariumSnapshotHistoryPoint,
) {
    if let Some(last) = history.last() {
        if last.seed != point.seed
            || last.preset != point.preset
            || point.time_s + 1.0e-6 < last.time_s
        {
            history.clear();
        }
    }
    if let Some(last) = history.last_mut() {
        if last.seed == point.seed
            && last.preset == point.preset
            && (last.time_s - point.time_s).abs() < 1.0e-6
        {
            *last = point;
            return;
        }
    }
    history.push(point);
    if history.len() > SNAPSHOT_HISTORY_LIMIT {
        let overflow = history.len() - SNAPSHOT_HISTORY_LIMIT;
        history.drain(0..overflow);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snapshot(
        time_s: f32,
        flies: usize,
        embryos: u32,
        viability: f32,
    ) -> crate::terrarium_world::TerrariumWorldSnapshot {
        let mut snapshot = crate::terrarium_world::TerrariumWorldSnapshot::default();
        snapshot.time_s = time_s;
        snapshot.flies = flies;
        snapshot.fly_population_eggs = embryos;
        snapshot.fly_population_embryos = embryos;
        snapshot.full_fly_embryos = (0..embryos)
            .map(|id| crate::terrarium_world::TerrariumFlyEmbryoSnapshot {
                embryo_id: id,
                cluster_index: 0,
                sex: "Female".into(),
                x: 0.0,
                y: 0.0,
                z: 0.0,
                age_hours: time_s,
                viability,
                water: 0.1,
                glucose: 0.05,
                amino_acids: 0.02,
                nucleotides: 0.01,
                membrane_precursors: 0.01,
                oxygen: 0.03,
            })
            .collect();
        snapshot
    }

    #[test]
    fn push_snapshot_history_replaces_duplicate_timestamp() {
        let mut history = Vec::new();
        push_snapshot_history_point(
            &mut history,
            snapshot_history_point(&make_snapshot(1.0, 2, 3, 0.7), 42, "demo"),
        );
        push_snapshot_history_point(
            &mut history,
            snapshot_history_point(&make_snapshot(1.0, 2, 5, 0.8), 42, "demo"),
        );
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].embryos, 5);
        assert_eq!(history[0].eggs, 5);
        assert_eq!(history[0].embryo_viability, Some(0.8));
    }

    #[test]
    fn push_snapshot_history_resets_on_seed_or_time_rewind() {
        let mut history = Vec::new();
        push_snapshot_history_point(
            &mut history,
            snapshot_history_point(&make_snapshot(2.0, 2, 3, 0.7), 42, "demo"),
        );
        push_snapshot_history_point(
            &mut history,
            snapshot_history_point(&make_snapshot(3.0, 2, 4, 0.7), 42, "demo"),
        );
        push_snapshot_history_point(
            &mut history,
            snapshot_history_point(&make_snapshot(0.5, 1, 1, 0.5), 43, "terrarium"),
        );
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].seed, 43);
        assert_eq!(history[0].preset, "terrarium");
        assert_eq!(history[0].time_s, 0.5);
    }

    #[test]
    fn push_snapshot_history_caps_length() {
        let mut history = Vec::new();
        for index in 0..(SNAPSHOT_HISTORY_LIMIT + 12) {
            push_snapshot_history_point(
                &mut history,
                snapshot_history_point(&make_snapshot(index as f32, 1, 1, 0.6), 42, "demo"),
            );
        }
        assert_eq!(history.len(), SNAPSHOT_HISTORY_LIMIT);
        assert!((history[0].time_s - 12.0).abs() < 1.0e-6);
        assert!(
            (history.last().unwrap().time_s - (SNAPSHOT_HISTORY_LIMIT + 11) as f32).abs() < 1.0e-6
        );
    }
}
