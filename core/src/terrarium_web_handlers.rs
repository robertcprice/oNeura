//! HTTP and WebSocket handlers for the terrarium web server.

use crate::ecosystem_integration;
use crate::terrarium::archive::TerrariumWorldArchive;
use crate::terrarium::checkpoint::TerrariumWorldCheckpoint;
use crate::terrarium::TerrariumDemoPreset;
use crate::terrarium_web_analysis::{start_sensitivity, start_stress};
use crate::terrarium_web_annotations::all_annotations;
use crate::terrarium_web_cutaway::build_terrain_cutaway_profiles;
use crate::terrarium_web_evolution::{start_evolution, stop_evolution};
use crate::terrarium_web_inspect::{build_scale_inspect_data, InspectQuery};
use crate::terrarium_web_protocol::{
    parse_view, ClientMsg, FrameData, OrganismLineageData, OrganismRegistryData,
    PharmaAdmetResultData, PharmaDockingResultData,
    RenameOrganismRequest, RenameOrganismResponse, ServerMsg, SnapshotData,
};
use crate::terrarium_web_state::AppState;
use crate::terrarium_web_tournament::{
    build_leaderboard_data, delete_entry, get_genome, submit_genome, TournamentSubmitRequest,
};
use axum::body::Bytes;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{Html, IntoResponse};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use std::sync::Arc;

/// Embedded HTML frontend.
const INDEX_HTML: &str = include_str!("../../terrarium/web/terrarium.html");
const THREE_JS: &str = include_str!("../../terrarium/web/vendor/three.min.js");
const ORBIT_CONTROLS_JS: &str = include_str!("../../terrarium/web/vendor/OrbitControls.js");
const TERRAIN_RENDERER_JS: &str = include_str!("../../terrarium/web/js/terrain_renderer.js");
const PLANT_RENDERER_JS: &str = include_str!("../../terrarium/web/js/plant_renderer.js");
const SCALE_MANAGER_JS: &str = include_str!("../../terrarium/web/js/scale_manager.js");
const MOLECULAR_RENDERER_JS: &str = include_str!("../../terrarium/web/js/molecular_renderer.js");
const ORGANISM_RENDERER_JS: &str = include_str!("../../terrarium/web/js/organism_renderer.js");
const CELLULAR_RENDERER_JS: &str = include_str!("../../terrarium/web/js/cellular_renderer.js");
const SSAO_EFFECT_JS: &str = include_str!("../../terrarium/web/js/ssao_effect.js");
const CUTAWAY_CONTROLLER_JS: &str = include_str!("../../terrarium/web/js/cutaway_controller.js");
const DENSITY_VOLUME_JS: &str = include_str!("../../terrarium/web/js/density_volume.js");
const MICROCOSM_HTML: &str = include_str!("../../terrarium/web/microcosm.html");
const MICROCOSM_JS: &str = include_str!("../../terrarium/web/js/microcosm.js");
const PHARMA_LAB_JS: &str = include_str!("../../terrarium/web/js/pharma_lab.js");
const PERIODIC_TABLE_JS: &str = include_str!("../../terrarium/web/js/periodic_table.js");
const MOLECULE_BUILDER_JS: &str = include_str!("../../terrarium/web/js/molecule_builder.js");
const REACTION_ANIMATOR_JS: &str = include_str!("../../terrarium/web/js/reaction_animator.js");
const ADMET_DASHBOARD_JS: &str = include_str!("../../terrarium/web/js/admet_dashboard.js");

/// GET / — serve the embedded HTML page.
pub async fn index_handler() -> Html<&'static str> {
    Html(INDEX_HTML)
}

/// GET /microcosm — serve the minimal soil-focused demo.
pub async fn microcosm_handler() -> Html<&'static str> {
    Html(MICROCOSM_HTML)
}

pub async fn microcosm_js_handler() -> impl IntoResponse {
    (
        [
            (header::CONTENT_TYPE, "application/javascript; charset=utf-8"),
            (header::CACHE_CONTROL, "public, max-age=3600"),
        ],
        MICROCOSM_JS,
    )
}

pub async fn three_js_handler() -> impl IntoResponse {
    (
        [
            (header::CONTENT_TYPE, "application/javascript; charset=utf-8"),
            (header::CACHE_CONTROL, "public, max-age=86400, immutable"),
        ],
        THREE_JS,
    )
}

pub async fn orbit_controls_handler() -> impl IntoResponse {
    (
        [
            (header::CONTENT_TYPE, "application/javascript; charset=utf-8"),
            (header::CACHE_CONTROL, "public, max-age=86400, immutable"),
        ],
        ORBIT_CONTROLS_JS,
    )
}

pub async fn terrain_renderer_handler() -> impl IntoResponse {
    (
        [
            (header::CONTENT_TYPE, "application/javascript; charset=utf-8"),
            (header::CACHE_CONTROL, "public, max-age=3600"),
        ],
        TERRAIN_RENDERER_JS,
    )
}

pub async fn plant_renderer_handler() -> impl IntoResponse {
    (
        [
            (header::CONTENT_TYPE, "application/javascript; charset=utf-8"),
            (header::CACHE_CONTROL, "public, max-age=3600"),
        ],
        PLANT_RENDERER_JS,
    )
}

pub async fn scale_manager_handler() -> impl IntoResponse {
    (
        [
            (header::CONTENT_TYPE, "application/javascript; charset=utf-8"),
            (header::CACHE_CONTROL, "public, max-age=3600"),
        ],
        SCALE_MANAGER_JS,
    )
}

pub async fn molecular_renderer_handler() -> impl IntoResponse {
    (
        [
            (header::CONTENT_TYPE, "application/javascript; charset=utf-8"),
            (header::CACHE_CONTROL, "public, max-age=3600"),
        ],
        MOLECULAR_RENDERER_JS,
    )
}

pub async fn organism_renderer_handler() -> impl IntoResponse {
    (
        [
            (header::CONTENT_TYPE, "application/javascript; charset=utf-8"),
            (header::CACHE_CONTROL, "public, max-age=3600"),
        ],
        ORGANISM_RENDERER_JS,
    )
}

pub async fn cellular_renderer_handler() -> impl IntoResponse {
    (
        [
            (header::CONTENT_TYPE, "application/javascript; charset=utf-8"),
            (header::CACHE_CONTROL, "public, max-age=3600"),
        ],
        CELLULAR_RENDERER_JS,
    )
}

pub async fn ssao_effect_handler() -> impl IntoResponse {
    (
        [
            (header::CONTENT_TYPE, "application/javascript; charset=utf-8"),
            (header::CACHE_CONTROL, "public, max-age=3600"),
        ],
        SSAO_EFFECT_JS,
    )
}

pub async fn cutaway_controller_handler() -> impl IntoResponse {
    (
        [
            (header::CONTENT_TYPE, "application/javascript; charset=utf-8"),
            (header::CACHE_CONTROL, "public, max-age=3600"),
        ],
        CUTAWAY_CONTROLLER_JS,
    )
}

pub async fn density_volume_handler() -> impl IntoResponse {
    (
        [
            (header::CONTENT_TYPE, "application/javascript; charset=utf-8"),
            (header::CACHE_CONTROL, "public, max-age=3600"),
        ],
        DENSITY_VOLUME_JS,
    )
}

// ---------------------------------------------------------------------------
// Pharma Lab JS handlers
// ---------------------------------------------------------------------------

pub async fn pharma_lab_handler() -> impl IntoResponse {
    ([
        (header::CONTENT_TYPE, "application/javascript; charset=utf-8"),
        (header::CACHE_CONTROL, "public, max-age=3600"),
    ], PHARMA_LAB_JS)
}
pub async fn periodic_table_handler() -> impl IntoResponse {
    ([
        (header::CONTENT_TYPE, "application/javascript; charset=utf-8"),
        (header::CACHE_CONTROL, "public, max-age=3600"),
    ], PERIODIC_TABLE_JS)
}
pub async fn molecule_builder_handler() -> impl IntoResponse {
    ([
        (header::CONTENT_TYPE, "application/javascript; charset=utf-8"),
        (header::CACHE_CONTROL, "public, max-age=3600"),
    ], MOLECULE_BUILDER_JS)
}
pub async fn reaction_animator_handler() -> impl IntoResponse {
    ([
        (header::CONTENT_TYPE, "application/javascript; charset=utf-8"),
        (header::CACHE_CONTROL, "public, max-age=3600"),
    ], REACTION_ANIMATOR_JS)
}
pub async fn admet_dashboard_handler() -> impl IntoResponse {
    ([
        (header::CONTENT_TYPE, "application/javascript; charset=utf-8"),
        (header::CACHE_CONTROL, "public, max-age=3600"),
    ], ADMET_DASHBOARD_JS)
}

/// GET /api/pharma/elements — full periodic table as JSON.
pub async fn pharma_elements_handler() -> impl IntoResponse {
    use crate::atomistic_chemistry::PeriodicElement;
    let elements: Vec<serde_json::Value> = PeriodicElement::all().map(|e| {
        serde_json::json!({
            "z": e.atomic_number(),
            "symbol": e.symbol(),
            "name": e.name(),
            "mass": e.mass_daltons(),
            "covalentRadius": e.covalent_radius_angstrom(),
            "vdwRadius": e.van_der_waals_radius_angstrom(),
            "cpkColor": e.cpk_color_rgb(),
            "electronegativity": e.pauling_electronegativity(),
            "electronConfig": e.electron_configuration_short(),
        })
    }).collect();
    (
        [(header::CONTENT_TYPE, "application/json; charset=utf-8"),
         (header::CACHE_CONTROL, "public, max-age=86400, immutable")],
        serde_json::to_string(&elements).unwrap_or_default(),
    )
}

/// GET /api/pharma/library — molecule library list.
pub async fn pharma_library_handler() -> impl IntoResponse {
    let entries = crate::pharma_lab::library::all_library_molecules();
    (
        [(header::CONTENT_TYPE, "application/json; charset=utf-8"),
         (header::CACHE_CONTROL, "public, max-age=3600")],
        serde_json::to_string(entries).unwrap_or_default(),
    )
}

/// GET /api/snapshot — return the latest world snapshot as JSON.
pub async fn snapshot_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let world = state.world.lock().await;
    let snapshot = world.snapshot();
    let daylight = snapshot.light;
    let time_label = world.time_label();
    let params = state.params.read().await;
    let field = world.topdown_field(params.view);
    let atmosphere = world.atmosphere_frame();
    let (terrain_surface, terrain_visuals, terrain_voxels, terrain_cutaways) =
        build_terrain_projection(&world, &atmosphere, params.view);
    let entities = AppState::extract_entities(&world);
    axum::Json(SnapshotData {
        snapshot,
        preset: params.preset.cli_name().to_string(),
        seed: params.seed,
        field: Some(field),
        width: Some(world.config.width),
        height: Some(world.config.height),
        view: Some(params.view.label().to_string()),
        atmosphere: Some(atmosphere),
        terrain_surface: Some(terrain_surface),
        terrain_visuals: Some(terrain_visuals),
        terrain_voxels: Some(terrain_voxels),
        terrain_cutaways: Some(terrain_cutaways),
        entities: Some(entities),
        daylight: Some(daylight),
        time_label: Some(time_label),
        paused: Some(params.paused),
        fps: Some(params.target_fps),
    })
}

/// GET /api/snapshot_history — return the recent authoritative lifecycle time series.
pub async fn snapshot_history_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    axum::Json(state.snapshot_history_data().await)
}

/// GET /api/inspect — return authoritative detail for a selected world object or cell.
pub async fn inspect_handler(
    State(state): State<Arc<AppState>>,
    Query(query): Query<InspectQuery>,
) -> impl IntoResponse {
    let world = state.world.lock().await;
    let params = state.params.read().await;
    match build_scale_inspect_data(&world, params.preset, &query) {
        Ok(data) => axum::Json(data).into_response(),
        Err(message) => (
            StatusCode::BAD_REQUEST,
            axum::Json(serde_json::json!({ "error": message })),
        )
            .into_response(),
    }
}

/// GET /api/organisms — return the tracked organism registry for the live world.
pub async fn organisms_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let world = state.world.lock().await;
    let entries = world.organism_registry.values().cloned().collect();
    axum::Json(OrganismRegistryData { entries })
}

/// GET /api/organisms/:id/lineage — return the registry lineage chain for one organism.
pub async fn organism_lineage_handler(
    State(state): State<Arc<AppState>>,
    Path(organism_id): Path<u64>,
) -> impl IntoResponse {
    let world = state.world.lock().await;
    if world.organism_registry_entry(organism_id).is_none() {
        return (
            StatusCode::NOT_FOUND,
            axum::Json(serde_json::json!({"error": format!("Unknown organism id: {organism_id}")})),
        )
            .into_response();
    }
    axum::Json(OrganismLineageData {
        organism_id,
        lineage: world.organism_lineage(organism_id),
    })
    .into_response()
}

/// POST /api/organisms/:id/name — update an organism display name.
pub async fn organism_rename_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(organism_id): Path<u64>,
    axum::Json(body): axum::Json<RenameOrganismRequest>,
) -> impl IntoResponse {
    {
        let auth = state.auth.lock().await;
        if !auth.check_auth(headers.get("authorization").and_then(|h| h.to_str().ok())) {
            return (
                StatusCode::UNAUTHORIZED,
                axum::Json(serde_json::json!({"error": "Unauthorized"})),
            )
                .into_response();
        }
    }

    let trimmed = body.name.trim();
    if trimmed.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            axum::Json(serde_json::json!({"error": "Organism name cannot be empty"})),
        )
            .into_response();
    }

    let updated_entry = {
        let mut world = state.world.lock().await;
        if !world.set_organism_name(organism_id, trimmed.to_string()) {
            return (
                StatusCode::NOT_FOUND,
                axum::Json(
                    serde_json::json!({"error": format!("Unknown organism id: {organism_id}")}),
                ),
            )
                .into_response();
        }
        world.organism_registry_entry(organism_id).cloned()
    };

    broadcast_current_state(&state).await;

    match updated_entry {
        Some(entry) => axum::Json(RenameOrganismResponse { organism_id, entry }).into_response(),
        None => (
            StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(serde_json::json!({"error": "Organism rename succeeded but registry entry was not found afterward"})),
        )
            .into_response(),
    }
}

// ---------------------------------------------------------------------------
// Tournament endpoints
// ---------------------------------------------------------------------------

/// POST /api/tournament/submit
pub async fn tournament_submit(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(body): axum::Json<TournamentSubmitRequest>,
) -> impl IntoResponse {
    // Auth check for mutations
    {
        let auth = state.auth.lock().await;
        if !auth.check_auth(headers.get("authorization").and_then(|h| h.to_str().ok())) {
            return (
                StatusCode::UNAUTHORIZED,
                axum::Json(serde_json::json!({"error": "Unauthorized"})),
            )
                .into_response();
        }
    }

    let id = submit_genome(
        state.tournament.clone(),
        state.tx.clone(),
        body.name,
        body.genome,
    )
    .await;

    axum::Json(serde_json::json!({"id": id, "status": "evaluating"})).into_response()
}

#[derive(Deserialize)]
pub struct LeaderboardQuery {
    #[serde(default = "default_sort")]
    sort: String,
}
fn default_sort() -> String {
    "biomass".into()
}

/// GET /api/tournament/leaderboard
pub async fn tournament_leaderboard(
    State(state): State<Arc<AppState>>,
    Query(query): Query<LeaderboardQuery>,
) -> impl IntoResponse {
    let data = build_leaderboard_data(&state.tournament, &query.sort).await;
    axum::Json(data)
}

/// GET /api/tournament/genome/:id
pub async fn tournament_genome(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u32>,
) -> impl IntoResponse {
    match get_genome(&state.tournament, id).await {
        Some(genome) => axum::Json(serde_json::json!(genome)).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            axum::Json(serde_json::json!({"error": "Not found"})),
        )
            .into_response(),
    }
}

/// DELETE /api/tournament/:id
pub async fn tournament_delete(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(id): Path<u32>,
) -> impl IntoResponse {
    {
        let auth = state.auth.lock().await;
        if !auth.check_auth(headers.get("authorization").and_then(|h| h.to_str().ok())) {
            return (StatusCode::UNAUTHORIZED, "Unauthorized").into_response();
        }
    }

    if delete_entry(&state.tournament, id).await {
        (StatusCode::OK, "Deleted").into_response()
    } else {
        (StatusCode::NOT_FOUND, "Not found").into_response()
    }
}

// ---------------------------------------------------------------------------
// Annotations endpoint
// ---------------------------------------------------------------------------

/// GET /api/annotations
pub async fn annotations_handler() -> impl IntoResponse {
    axum::Json(all_annotations())
}

// ---------------------------------------------------------------------------
// Export endpoint
// ---------------------------------------------------------------------------

/// GET /api/archive — return the current terrarium archive record.
pub async fn archive_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let world = state.world.lock().await;
    axum::Json(TerrariumWorldArchive {
        config: world.config.clone(),
        time_s: world.time_s,
        seed_provenance: world.seed_provenance().clone(),
        climate_driver: world.climate_driver.clone(),
        snapshot: world.snapshot(),
        organism_registry: world.organism_registry.clone(),
        organism_phylogeny: world.organism_phylogeny.clone(),
    })
}

/// GET /api/checkpoint — return the current terrarium checkpoint record.
pub async fn checkpoint_handler(
    State(state): State<Arc<AppState>>,
) -> Result<axum::Json<TerrariumWorldCheckpoint>, (StatusCode, axum::Json<serde_json::Value>)> {
    let mut world = state.world.lock().await;
    world.checkpoint().map(axum::Json).map_err(|error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(serde_json::json!({
                "error": format!("Could not build terrarium checkpoint: {error}")
            })),
        )
    })
}

/// POST /api/import/checkpoint — replace the live world from a checkpoint JSON or bundle.
pub async fn import_checkpoint_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    {
        let auth = state.auth.lock().await;
        if !auth.check_auth(headers.get("authorization").and_then(|h| h.to_str().ok())) {
            return (
                StatusCode::UNAUTHORIZED,
                axum::Json(serde_json::json!({"error": "Unauthorized"})),
            )
                .into_response();
        }
    }

    {
        let mut params = state.params.write().await;
        params.paused = true;
    }

    let value: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(value) => value,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                axum::Json(serde_json::json!({
                    "error": format!("Could not parse checkpoint JSON: {error}")
                })),
            )
                .into_response()
        }
    };

    let checkpoint_value = value
        .get("simulation")
        .and_then(|simulation| simulation.get("checkpoint"))
        .cloned()
        .unwrap_or(value);

    let checkpoint_json =
        match serde_json::to_string(&checkpoint_value) {
            Ok(json) => json,
            Err(error) => return (
                StatusCode::BAD_REQUEST,
                axum::Json(serde_json::json!({
                    "error": format!("Could not serialize extracted checkpoint payload: {error}")
                })),
            )
                .into_response(),
        };

    let checkpoint =
        match TerrariumWorldCheckpoint::from_json_str(&checkpoint_json) {
            Ok(checkpoint) => checkpoint,
            Err(error) => return (
                StatusCode::BAD_REQUEST,
                axum::Json(serde_json::json!({
                    "error": format!("Input did not contain a valid terrarium checkpoint: {error}")
                })),
            )
                .into_response(),
        };

    let world = match crate::terrarium::TerrariumWorld::from_checkpoint(checkpoint) {
        Ok(world) => world,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                axum::Json(serde_json::json!({
                    "error": format!("Could not restore terrarium checkpoint: {error}")
                })),
            )
                .into_response()
        }
    };

    let preset = TerrariumDemoPreset::infer_from_config(&world.config);
    let seed = world.seed_provenance().seed;
    let tracked = world.snapshot().tracked_organisms;
    state.install_world(world, preset, true).await;
    broadcast_current_state(&state).await;

    axum::Json(serde_json::json!({
        "status": "ok",
        "seed": seed,
        "preset": preset.unwrap_or(TerrariumDemoPreset::Demo).cli_name(),
        "tracked_organisms": tracked,
    }))
    .into_response()
}

/// GET /api/export/bundle
pub async fn export_bundle(
    State(state): State<Arc<AppState>>,
) -> Result<axum::Json<serde_json::Value>, (StatusCode, axum::Json<serde_json::Value>)> {
    let mut world = state.world.lock().await;
    let snapshot = world.snapshot();
    let archive = world.archive();
    let checkpoint = world.checkpoint().map_err(|error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(serde_json::json!({
                "error": format!("Could not build terrarium export bundle checkpoint: {error}")
            })),
        )
    })?;
    let params = state.params.read().await;
    let telemetry = state.last_telemetry.lock().await;
    let snapshot_history = state.snapshot_history.lock().await.clone();
    let leaderboard = build_leaderboard_data(&state.tournament, "biomass").await;

    let bundle = serde_json::json!({
        "version": "1.0",
        "exported_at": chrono_now_iso(),
        "simulation": {
            "config": {
                "width": world.config.width,
                "height": world.config.height,
                "depth": world.config.depth,
                "seed": world.config.seed,
                "time_warp": world.config.time_warp,
            },
            "snapshot": snapshot,
            "snapshot_history": snapshot_history,
            "archive": archive,
            "checkpoint": checkpoint,
            "seed": params.seed,
        },
        "evolution_telemetry": *telemetry,
        "tournament_leaderboard": leaderboard.entries,
    });

    Ok(axum::Json(bundle))
}

fn chrono_now_iso() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{}", now)
}

// ---------------------------------------------------------------------------
// Auth endpoint
// ---------------------------------------------------------------------------

/// GET /api/auth/token
pub async fn auth_token(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut auth = state.auth.lock().await;
    let response = auth.generate_token();
    axum::Json(response)
}

// ---------------------------------------------------------------------------
// Ecosystem integration endpoints
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct EcosystemStartQuery {
    #[serde(default = "default_scenario")]
    scenario: String,
    #[serde(default = "default_eco_seed")]
    seed: u64,
}
fn default_scenario() -> String {
    "climate".into()
}
fn default_eco_seed() -> u64 {
    42
}

/// GET /api/ecosystem/snapshot — return the current ecosystem snapshot.
pub async fn ecosystem_snapshot_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let eco = state.ecosystem.lock().await;
    match eco.as_ref() {
        Some(ecosystem) => {
            let snapshot = serde_json::json!({
                "active": true,
                "time_days": ecosystem.time(),
                "config": ecosystem.config,
            });
            axum::Json(snapshot).into_response()
        }
        None => axum::Json(serde_json::json!({"active": false})).into_response(),
    }
}

/// POST /api/ecosystem/start — start an integrated ecosystem scenario.
pub async fn ecosystem_start_handler(
    State(state): State<Arc<AppState>>,
    Query(query): Query<EcosystemStartQuery>,
) -> impl IntoResponse {
    let ecosystem = match query.scenario.as_str() {
        "climate" | "climate_impact" => ecosystem_integration::climate_impact_scenario(query.seed),
        "amr" | "amr_emergence" => ecosystem_integration::amr_emergence_scenario(query.seed),
        "soil" | "soil_health" => ecosystem_integration::soil_health_scenario(query.seed),
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                axum::Json(serde_json::json!({"error": format!("Unknown scenario: {}. Use: climate, amr, soil", query.scenario)})),
            ).into_response();
        }
    };

    let config_json = serde_json::json!(ecosystem.config);
    let mut eco = state.ecosystem.lock().await;
    *eco = Some(ecosystem);

    axum::Json(serde_json::json!({
        "status": "started",
        "scenario": query.scenario,
        "seed": query.seed,
        "config": config_json,
    }))
    .into_response()
}

/// POST /api/ecosystem/step — advance the ecosystem by one time step.
pub async fn ecosystem_step_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut eco = state.ecosystem.lock().await;
    match eco.as_mut() {
        Some(ecosystem) => {
            let snapshot = ecosystem.step();
            axum::Json(serde_json::json!({
                "status": "stepped",
                "snapshot": snapshot,
            })).into_response()
        }
        None => (
            StatusCode::BAD_REQUEST,
            axum::Json(serde_json::json!({"error": "No ecosystem active. POST /api/ecosystem/start first."})),
        ).into_response(),
    }
}

#[derive(Deserialize)]
pub struct EcosystemRunQuery {
    #[serde(default = "default_run_days")]
    days: f64,
}
fn default_run_days() -> f64 {
    30.0
}

/// POST /api/ecosystem/run — run ecosystem for N days, return full time series.
pub async fn ecosystem_run_handler(
    State(state): State<Arc<AppState>>,
    Query(query): Query<EcosystemRunQuery>,
) -> impl IntoResponse {
    let mut eco = state.ecosystem.lock().await;
    match eco.as_mut() {
        Some(ecosystem) => {
            let timeseries = ecosystem.run(query.days);
            axum::Json(serde_json::json!({
                "status": "completed",
                "days": query.days,
                "snapshots": timeseries.snapshots.len(),
                "timeseries": timeseries,
            })).into_response()
        }
        None => (
            StatusCode::BAD_REQUEST,
            axum::Json(serde_json::json!({"error": "No ecosystem active. POST /api/ecosystem/start first."})),
        ).into_response(),
    }
}

// ---------------------------------------------------------------------------
// WebSocket handler
// ---------------------------------------------------------------------------

/// WS /ws — WebSocket handler for real-time frame streaming + commands.
pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.max_frame_size(16 * 1024 * 1024) // 16 MB — terrain visuals can be large
        .max_message_size(16 * 1024 * 1024)
        .on_upgrade(move |socket| handle_ws(socket, state))
}

async fn handle_ws(socket: WebSocket, state: Arc<AppState>) {
    eprintln!("[ws] Client connected");
    let (mut ws_tx, mut ws_rx) = socket.split();

    // Subscribe to broadcast channel
    let mut broadcast_rx = state.tx.subscribe();

    // Spawn a task to forward broadcast messages to this WebSocket client
    let forward_task = tokio::spawn(async move {
        loop {
            match broadcast_rx.recv().await {
                Ok(msg) => {
                    // Encode Frame messages as binary for performance.
                    // Other messages sent as plain JSON text (client DecompressionStream
                    // not reliably supported in all browsers).
                    let ws_msg = match &msg {
                        ServerMsg::Frame(frame) => encode_binary_frame(frame),
                        _ => {
                            if let Ok(json) = serde_json::to_string(&msg) {
                                Message::Text(json.into())
                            } else {
                                continue;
                            }
                        }
                    };
                    let msg_len = match &ws_msg {
                        Message::Binary(b) => b.len(),
                        Message::Text(t) => t.len(),
                        _ => 0,
                    };
                    if ws_tx.send(ws_msg).await.is_err() {
                        eprintln!("[ws] Send failed (msg_len={msg_len})");
                        break;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    eprintln!("[ws] client lagged, skipped {} messages", n);
                }
                Err(_) => break,
            }
        }
    });

    broadcast_current_state(&state).await;

    // Process incoming commands from this client
    while let Some(Ok(msg)) = ws_rx.next().await {
        if let Message::Text(text) = msg {
            match serde_json::from_str::<ClientMsg>(&text) {
                Ok(cmd) => handle_command(cmd, &state).await,
                Err(e) => {
                    eprintln!("[ws] bad command: {}: {}", text, e);
                }
            }
        }
    }

    eprintln!("[ws] Client disconnected");
    forward_task.abort();
}

/// Encode a Frame as a binary WebSocket message.
/// Format: header [0x46, 0x52, width_u8, height_u8] + width*height u16 big-endian + '\0' + JSON metadata
fn encode_binary_frame(frame: &FrameData) -> Message {
    let w = frame.width as u8;
    let h = frame.height as u8;
    let field_len = frame.field.len();

    // Find min/max for normalization
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for &v in &frame.field {
        if v < mn {
            mn = v;
        }
        if v > mx {
            mx = v;
        }
    }
    let range = if (mx - mn).abs() < 1e-12 {
        1.0
    } else {
        mx - mn
    };

    // Header (4 bytes) + field (2 bytes each) + null separator (1 byte) + JSON metadata
    // Exclude terrain_voxels and terrain_cutaways from per-frame binary messages
    // to keep under browser WebSocket size limits (~1 MB in Safari).
    // These large 3D mesh structures are sent in the Snapshot message on intervals.
    let meta = serde_json::json!({
        "view": frame.view,
        "atmosphere": frame.atmosphere,
        "terrain_surface": frame.terrain_surface,
        "terrain_visuals": frame.terrain_visuals,
        "daylight": frame.daylight,
        "sun_elevation_rad": frame.sun_elevation_rad,
        "sun_azimuth_rad": frame.sun_azimuth_rad,
        "sun_direction": frame.sun_direction,
        "time_label": frame.time_label,
        "paused": frame.paused,
        "fps": frame.fps,
        "mn": mn,
        "mx": mx,
        "moisture": frame.moisture,
        "water_mask": frame.water_mask,
        "soil_structure": frame.soil_structure,
    });
    let meta_bytes = meta.to_string().into_bytes();

    let total = 4 + field_len * 2 + 1 + meta_bytes.len();
    let mut buf = Vec::with_capacity(total);

    // Header: magic bytes 'FR' + dimensions
    buf.push(0x46); // 'F'
    buf.push(0x52); // 'R'
    buf.push(w);
    buf.push(h);

    // Field data as u16 big-endian (normalized 0-65535)
    for &v in &frame.field {
        let norm = ((v - mn) / range).clamp(0.0, 1.0);
        let u16_val = (norm * 65535.0) as u16;
        buf.push((u16_val >> 8) as u8);
        buf.push((u16_val & 0xFF) as u8);
    }

    // Null separator
    buf.push(0);

    // JSON metadata (plain — gzip not reliably supported in all browser WS clients)
    buf.extend_from_slice(&meta_bytes);

    Message::Binary(buf.into())
}

/// Gzip-compress raw bytes using flate2.
fn gzip_bytes(data: &[u8]) -> Vec<u8> {
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;
    let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
    let _ = encoder.write_all(data);
    encoder.finish().unwrap_or_else(|_| data.to_vec())
}

/// Gzip-compress a JSON string and wrap as a binary WS message.
/// Header: [0x47, 0x5A] ("GZ") followed by gzipped JSON.
fn gzip_ws_json(json: &str) -> Message {
    let compressed = gzip_bytes(json.as_bytes());
    let mut buf = Vec::with_capacity(2 + compressed.len());
    buf.push(0x47); // 'G'
    buf.push(0x5A); // 'Z'
    buf.extend_from_slice(&compressed);
    Message::Binary(buf.into())
}

fn build_terrain_visuals(
    world: &crate::terrarium_world::TerrariumWorld,
    atmosphere: &crate::terrarium::TerrariumAtmosphereFrame,
) -> Vec<crate::terrarium::visual_projection::TerrariumTerrainVisualResponse> {
    let width = world.config.width;
    let height = world.config.height;
    let mut visuals = Vec::with_capacity(width * height);
    for idx in 0..(width * height) {
        let x = (idx % width) as f32 + 0.5;
        let y = (idx / width) as f32 + 0.5;
        let local_air =
            crate::terrarium::visual_projection::sample_visual_air(atmosphere, width, height, x, y);
        let chemistry = crate::terrarium::visual_projection::sample_visual_chemistry(
            world,
            idx % width,
            idx / width,
        );
        visuals.push(
            crate::terrarium::visual_projection::terrain_visual_response_with_chemistry(
                world.moisture[idx],
                world.organic_matter[idx],
                local_air,
                chemistry,
                world.config.visual_emergence_blend,
            ),
        );
    }
    visuals
}

fn build_terrain_projection(
    world: &crate::terrarium_world::TerrariumWorld,
    atmosphere: &crate::terrarium::TerrariumAtmosphereFrame,
    view: crate::terrarium::TerrariumTopdownView,
) -> (
    Vec<f32>,
    Vec<crate::terrarium::visual_projection::TerrariumTerrainVisualResponse>,
    Vec<crate::terrarium::visual_projection::TerrariumTerrainVoxelBatch>,
    Vec<crate::terrarium_web_protocol::TerrainCutawayProfile>,
) {
    let terrain_view = matches!(view, crate::terrarium::TerrariumTopdownView::Terrain);
    let visuals = if terrain_view {
        build_terrain_visuals(world, atmosphere)
    } else {
        Vec::new()
    };
    let (surface, voxels, cutaways) = if terrain_view {
        let (surface, voxels) =
            crate::terrarium::visual_projection::terrain_voxel_batches(world, atmosphere);
        let cutaways = build_terrain_cutaway_profiles(world, atmosphere);
        (surface, voxels, cutaways)
    } else {
        (world.soil_structure.clone(), Vec::new(), Vec::new())
    };
    (surface, visuals, voxels, cutaways)
}

async fn broadcast_current_state(state: &Arc<AppState>) {
    let world = state.world.lock().await;
    let params = state.params.read().await;
    let field = world.topdown_field(params.view);
    let atmosphere = world.atmosphere_frame();
    // Initial connection: send FULL terrain data including voxels and cutaways
    // so the browser can build the complete 3D terrain with depth layers.
    let (terrain_surface, terrain_visuals, terrain_voxels, terrain_cutaways) =
        build_terrain_projection(&world, &atmosphere, params.view);
    let snapshot = world.snapshot();
    let entities = AppState::extract_entities(&world);
    let time_label = world.time_label();
    let snapshot_history = state
        .record_snapshot_history(&snapshot, params.seed, params.preset.cli_name())
        .await;

    // Send frame with terrain data + shader DataTexture arrays
    let plane = world.config.width * world.config.height;
    let solar = world.solar_state();
    let _ = state.tx.send(ServerMsg::Frame(FrameData {
        field,
        width: world.config.width,
        height: world.config.height,
        view: params.view.label().to_string(),
        atmosphere,
        terrain_surface: terrain_surface.clone(),
        terrain_visuals,
        terrain_voxels: Vec::new(),
        terrain_cutaways: Vec::new(),
        daylight: snapshot.light,
        sun_elevation_rad: solar.elevation_rad,
        sun_azimuth_rad: solar.azimuth_rad,
        sun_direction: solar.direction,
        time_label,
        paused: params.paused,
        fps: params.target_fps,
        moisture: world.moisture[..plane].to_vec(),
        water_mask: world.water_mask[..plane].to_vec(),
        soil_structure: world.soil_structure[..plane].to_vec(),
    }));
    let _ = state.tx.send(ServerMsg::Snapshot(SnapshotData {
        snapshot,
        preset: params.preset.cli_name().to_string(),
        seed: params.seed,
        field: None,
        width: None,
        height: None,
        view: None,
        atmosphere: None,
        terrain_surface: None,
        terrain_visuals: None,
        terrain_voxels: None,
        terrain_cutaways: None,
        entities: None,
        daylight: None,
        time_label: None,
        paused: None,
        fps: None,
    }));
    let _ = state.tx.send(ServerMsg::SnapshotHistory(snapshot_history));
    let _ = state.tx.send(ServerMsg::Entities(entities));
}

async fn handle_command(cmd: ClientMsg, state: &Arc<AppState>) {
    match cmd {
        ClientMsg::Play => {
            let mut params = state.params.write().await;
            params.paused = false;
        }
        ClientMsg::Pause => {
            let mut params = state.params.write().await;
            params.paused = true;
        }
        ClientMsg::Step => {
            let mut world = state.world.lock().await;
            let _ = world.step_frame();

            let params = state.params.read().await;
            let field = world.topdown_field(params.view);
            let atmosphere = world.atmosphere_frame();
            let (terrain_surface, terrain_visuals, terrain_voxels, terrain_cutaways) =
                build_terrain_projection(&world, &atmosphere, params.view);
            let w = world.config.width;
            let h = world.config.height;
            let snapshot = world.snapshot();
            let entities = AppState::extract_entities(&world);
            let time_label = world.time_label();
            let snapshot_history = state
                .record_snapshot_history(&snapshot, params.seed, params.preset.cli_name())
                .await;

            let plane = w * h;
            let solar = world.solar_state();
            let _ = state.tx.send(ServerMsg::Frame(FrameData {
                field,
                width: w,
                height: h,
                view: params.view.label().to_string(),
                atmosphere,
                terrain_surface,
                terrain_visuals,
                terrain_voxels,
                terrain_cutaways,
                daylight: snapshot.light,
                sun_elevation_rad: solar.elevation_rad,
                sun_azimuth_rad: solar.azimuth_rad,
                sun_direction: solar.direction,
                time_label,
                paused: true,
                fps: params.target_fps,
                moisture: world.moisture[..plane].to_vec(),
                water_mask: world.water_mask[..plane].to_vec(),
                soil_structure: world.soil_structure[..plane].to_vec(),
            }));
            let _ = state.tx.send(ServerMsg::Snapshot(SnapshotData {
                snapshot,
                preset: params.preset.cli_name().to_string(),
                seed: params.seed,
                field: None,
                width: None,
                height: None,
                view: None,
                atmosphere: None,
                terrain_surface: None,
                terrain_visuals: None,
                terrain_voxels: None,
                terrain_cutaways: None,
                entities: None,
                daylight: None,
                time_label: None,
                paused: None,
                fps: None,
            }));
            let _ = state.tx.send(ServerMsg::SnapshotHistory(snapshot_history));
            let _ = state.tx.send(ServerMsg::Entities(entities));
        }
        ClientMsg::Reset { seed, preset } => {
            let next_preset = if let Some(name) = preset {
                match TerrariumDemoPreset::parse(&name) {
                    Some(preset) => preset,
                    None => {
                        let _ = state.tx.send(ServerMsg::Error {
                            message: format!("Unknown preset: {name}"),
                        });
                        return;
                    }
                }
            } else {
                state.params.read().await.preset
            };

            if let Ok(new_world) = AppState::build_world(seed, next_preset) {
                state
                    .install_world(new_world, Some(next_preset), true)
                    .await;
                broadcast_current_state(state).await;
            }
        }
        ClientMsg::Speed { fps } => {
            let mut params = state.params.write().await;
            params.target_fps = fps.clamp(1, 60);
        }
        ClientMsg::View { mode } => {
            let mut params = state.params.write().await;
            params.view = parse_view(&mode);
            let should_refresh = params.paused;
            drop(params);
            if should_refresh {
                broadcast_current_state(state).await;
            }
        }
        ClientMsg::AddPlant { x, y } => {
            let mut world = state.world.lock().await;
            let _ = world.add_plant(x, y, None, None);
            drop(world);
            if state.params.read().await.paused {
                broadcast_current_state(state).await;
            }
        }
        ClientMsg::AddFly { x, y } => {
            let mut world = state.world.lock().await;
            let seed = {
                let params = state.params.read().await;
                params.seed
            };
            let fly_count = world.flies.len() as u64;
            world.add_fly(
                crate::drosophila::DrosophilaScale::Tiny,
                x,
                y,
                seed.wrapping_add(fly_count + 100),
            );
            drop(world);
            if state.params.read().await.paused {
                broadcast_current_state(state).await;
            }
        }
        ClientMsg::AddFruit { x, y } => {
            let mut world = state.world.lock().await;
            world.add_fruit(x, y, 0.8, None);
            drop(world);
            if state.params.read().await.paused {
                broadcast_current_state(state).await;
            }
        }
        ClientMsg::AddWater { x, y } => {
            let mut world = state.world.lock().await;
            world.add_water(x, y, 150.0, 0.0008);
            drop(world);
            if state.params.read().await.paused {
                broadcast_current_state(state).await;
            }
        }
        ClientMsg::EvolveStart { config } => {
            start_evolution(state.clone(), config).await;
        }
        ClientMsg::EvolveStop => {
            stop_evolution(state.clone()).await;
        }
        ClientMsg::EvolveApplyGenome { genome } => match genome.build_world() {
            Ok(new_world) => {
                state.install_world(new_world, None, true).await;
            }
            Err(e) => {
                let _ = state.tx.send(ServerMsg::Error {
                    message: format!("Failed to apply genome: {}", e),
                });
            }
        },
        ClientMsg::SensitivityStart { config } => {
            start_sensitivity(state.clone(), config).await;
        }
        ClientMsg::StressStart { config } => {
            start_stress(state.clone(), config).await;
        }
        ClientMsg::EcosystemStart { scenario, seed } => {
            let ecosystem = match scenario.as_str() {
                "amr" | "amr_emergence" => ecosystem_integration::amr_emergence_scenario(seed),
                "soil" | "soil_health" => ecosystem_integration::soil_health_scenario(seed),
                _ => ecosystem_integration::climate_impact_scenario(seed),
            };
            let mut eco = state.ecosystem.lock().await;
            *eco = Some(ecosystem);
        }
        ClientMsg::EcosystemStep => {
            let mut eco = state.ecosystem.lock().await;
            if let Some(ecosystem) = eco.as_mut() {
                let snapshot = ecosystem.step();
                let _ = state.tx.send(ServerMsg::EcosystemSnapshot(snapshot));
            }
        }
        ClientMsg::EcosystemRun { days } => {
            let mut eco = state.ecosystem.lock().await;
            if let Some(ecosystem) = eco.as_mut() {
                let timeseries = ecosystem.run(days);
                let _ = state.tx.send(ServerMsg::EcosystemTimeSeries(timeseries));
            }
        }
        ClientMsg::SetClimate { scenario, seed } => {
            let mut world = state.world.lock().await;
            let climate_seed = seed.unwrap_or(world.seed_provenance().seed);
            match scenario.as_str() {
                "rcp85" => world.enable_climate_driver(
                    crate::climate_scenarios::ClimateScenario::Rcp85,
                    climate_seed,
                    None,
                ),
                "rcp45" => world.enable_climate_driver(
                    crate::climate_scenarios::ClimateScenario::Rcp45,
                    climate_seed,
                    None,
                ),
                "rcp26" => world.enable_climate_driver(
                    crate::climate_scenarios::ClimateScenario::Rcp26,
                    climate_seed,
                    None,
                ),
                "preindustrial" => world.enable_climate_driver(
                    crate::climate_scenarios::ClimateScenario::PreIndustrial,
                    climate_seed,
                    None,
                ),
                "none" | "" => world.disable_climate_driver(),
                other => {
                    let _ = state.tx.send(ServerMsg::Error {
                        message: format!("Unknown climate scenario: {other}"),
                    });
                    return;
                }
            }
            drop(world);
            broadcast_current_state(state).await;
        }
        ClientMsg::SetVisualBlend { blend } => {
            let mut world = state.world.lock().await;
            world.config.visual_emergence_blend = blend.clamp(0.0, 1.0);
            drop(world);
            broadcast_current_state(state).await;
        }
        ClientMsg::TriggerExtremeEvent {
            event_type,
            severity,
        } => {
            let mut world = state.world.lock().await;
            world.apply_extreme_event_manual(&event_type, severity);
            drop(world);
            broadcast_current_state(state).await;
        }
        ClientMsg::SetTimeScale { scale } => {
            let clamped = scale.clamp(0.25, 32.0);
            let mut params = state.params.write().await;
            params.steps_per_frame = clamped.round().max(1.0) as u32;
        }
        // ---------------------------------------------------------------
        // Pharma Lab commands
        // ---------------------------------------------------------------
        ClientMsg::PharmaEnter => {
            let mut lab = state.pharma_lab.lock().await;
            if lab.is_none() {
                *lab = Some(crate::pharma_lab::PharmaLab::new(
                    crate::pharma_lab::LabConfig::default(),
                ));
            }
            let snap = lab.as_ref().unwrap().snapshot();
            let _ = state.tx.send(ServerMsg::PharmaLabState(snap));
        }
        ClientMsg::PharmaExit => {
            let mut lab = state.pharma_lab.lock().await;
            *lab = None;
        }
        ClientMsg::PharmaAddAtom { element, position } => {
            if let Some(elem) = crate::atomistic_chemistry::PeriodicElement::from_symbol_or_name(&element) {
                let mut lab = state.pharma_lab.lock().await;
                if let Some(lab) = lab.as_mut() {
                    lab.add_atom(elem, position);
                    let snap = lab.snapshot();
                    let frame = lab.build_frame(&[]);
                    let _ = state.tx.send(ServerMsg::PharmaLabState(snap));
                    let _ = state.tx.send(ServerMsg::PharmaLabFrame(frame));
                }
            } else {
                let _ = state.tx.send(ServerMsg::PharmaError {
                    message: format!("unknown element: {}", element),
                });
            }
        }
        ClientMsg::PharmaAddBond { molecule_id, atom_a, atom_b, order } => {
            let bond_order = match order.to_lowercase().as_str() {
                "single" | "s" => crate::atomistic_chemistry::BondOrder::Single,
                "double" | "d" => crate::atomistic_chemistry::BondOrder::Double,
                "triple" | "t" => crate::atomistic_chemistry::BondOrder::Triple,
                "aromatic" | "a" => crate::atomistic_chemistry::BondOrder::Aromatic,
                _ => crate::atomistic_chemistry::BondOrder::Single,
            };
            let mut lab = state.pharma_lab.lock().await;
            if let Some(lab) = lab.as_mut() {
                if let Err(e) = lab.add_bond(molecule_id, atom_a, atom_b, bond_order) {
                    let _ = state.tx.send(ServerMsg::PharmaError { message: e });
                } else {
                    let snap = lab.snapshot();
                    let frame = lab.build_frame(&[]);
                    let _ = state.tx.send(ServerMsg::PharmaLabState(snap));
                    let _ = state.tx.send(ServerMsg::PharmaLabFrame(frame));
                }
            }
        }
        ClientMsg::PharmaRemoveAtom { molecule_id, atom_idx } => {
            let mut lab = state.pharma_lab.lock().await;
            if let Some(lab) = lab.as_mut() {
                let _ = lab.remove_atom(molecule_id, atom_idx);
                let snap = lab.snapshot();
                let frame = lab.build_frame(&[]);
                let _ = state.tx.send(ServerMsg::PharmaLabState(snap));
                let _ = state.tx.send(ServerMsg::PharmaLabFrame(frame));
            }
        }
        ClientMsg::PharmaRemoveBond { molecule_id, bond_idx } => {
            let mut lab = state.pharma_lab.lock().await;
            if let Some(lab) = lab.as_mut() {
                let _ = lab.remove_bond(molecule_id, bond_idx);
                let snap = lab.snapshot();
                let frame = lab.build_frame(&[]);
                let _ = state.tx.send(ServerMsg::PharmaLabState(snap));
                let _ = state.tx.send(ServerMsg::PharmaLabFrame(frame));
            }
        }
        ClientMsg::PharmaRemoveMolecule { molecule_id } => {
            let mut lab = state.pharma_lab.lock().await;
            if let Some(lab) = lab.as_mut() {
                let _ = lab.remove_molecule(molecule_id);
                let snap = lab.snapshot();
                let _ = state.tx.send(ServerMsg::PharmaLabState(snap));
            }
        }
        ClientMsg::PharmaParseSmiles { smiles } => {
            let mut lab = state.pharma_lab.lock().await;
            if let Some(lab) = lab.as_mut() {
                match lab.add_molecule_from_smiles(&smiles) {
                    Ok(_id) => {
                        let snap = lab.snapshot();
                        let frame = lab.build_frame(&[]);
                        let _ = state.tx.send(ServerMsg::PharmaLabState(snap));
                        let _ = state.tx.send(ServerMsg::PharmaLabFrame(frame));
                    }
                    Err(e) => {
                        let _ = state.tx.send(ServerMsg::PharmaError { message: e });
                    }
                }
            }
        }
        ClientMsg::PharmaLoadLibrary { name } => {
            let mut lab = state.pharma_lab.lock().await;
            if let Some(lab) = lab.as_mut() {
                match lab.add_library_molecule(&name) {
                    Ok(_id) => {
                        let snap = lab.snapshot();
                        let frame = lab.build_frame(&[]);
                        let _ = state.tx.send(ServerMsg::PharmaLabState(snap));
                        let _ = state.tx.send(ServerMsg::PharmaLabFrame(frame));
                    }
                    Err(e) => {
                        let _ = state.tx.send(ServerMsg::PharmaError { message: e });
                    }
                }
            }
        }
        ClientMsg::PharmaMergeMolecules { molecule_a, molecule_b } => {
            let mut lab = state.pharma_lab.lock().await;
            if let Some(lab) = lab.as_mut() {
                let _ = lab.merge_molecules(molecule_a, molecule_b);
                let snap = lab.snapshot();
                let frame = lab.build_frame(&[]);
                let _ = state.tx.send(ServerMsg::PharmaLabState(snap));
                let _ = state.tx.send(ServerMsg::PharmaLabFrame(frame));
            }
        }
        ClientMsg::PharmaMdStart => {
            let mut lab = state.pharma_lab.lock().await;
            if let Some(lab) = lab.as_mut() {
                lab.set_md_running(true);
                let snap = lab.snapshot();
                let _ = state.tx.send(ServerMsg::PharmaLabState(snap));
            }
        }
        ClientMsg::PharmaMdStop => {
            let mut lab = state.pharma_lab.lock().await;
            if let Some(lab) = lab.as_mut() {
                lab.set_md_running(false);
                let snap = lab.snapshot();
                let _ = state.tx.send(ServerMsg::PharmaLabState(snap));
            }
        }
        ClientMsg::PharmaMdStep { steps } => {
            let mut lab = state.pharma_lab.lock().await;
            if let Some(lab) = lab.as_mut() {
                for mol in &mut lab.molecules { mol.locked = false; }
                let _stats = lab.step_md(steps);
                let reactions = lab.check_proximity_reactions();
                let frame = lab.build_frame(&reactions);
                let _ = state.tx.send(ServerMsg::PharmaLabFrame(frame));
                for event in reactions {
                    let _ = state.tx.send(ServerMsg::PharmaReactionEvent(event));
                }
            }
        }
        ClientMsg::PharmaSetTemperature { kelvin } => {
            let mut lab = state.pharma_lab.lock().await;
            if let Some(lab) = lab.as_mut() {
                lab.set_temperature(kelvin.clamp(10.0, 5000.0));
            }
        }
        ClientMsg::PharmaDock { ligand_id, target } => {
            let lab = state.pharma_lab.lock().await;
            if let Some(lab) = lab.as_ref() {
                match lab.dock_ligand(ligand_id, &target) {
                    Ok(result) => {
                        let _ = state.tx.send(ServerMsg::PharmaDockingResult(
                            PharmaDockingResultData {
                                ligand_name: result.candidate.name.clone(),
                                target_name: target,
                                binding_energy_kcal: result.binding_energy_kcal,
                                contacts: result.contacts,
                                pharmacophore_match: result.pharmacophore_match_score,
                                ligand_efficiency: result.ligand_efficiency,
                                vdw_energy: result.vdw_energy,
                                electrostatic_energy: result.electrostatic_energy,
                                desolvation_penalty: result.desolvation_penalty,
                                entropy_penalty: result.entropy_penalty,
                            },
                        ));
                    }
                    Err(e) => {
                        let _ = state.tx.send(ServerMsg::PharmaError { message: e });
                    }
                }
            }
        }
        ClientMsg::PharmaAdmet { molecule_id } => {
            let lab = state.pharma_lab.lock().await;
            if let Some(lab) = lab.as_ref() {
                match lab.compute_admet(molecule_id) {
                    Ok(profile) => {
                        let _ = state.tx.send(ServerMsg::PharmaAdmetResult(
                            PharmaAdmetResultData {
                                molecule_name: profile.candidate_name.clone(),
                                absorption: profile.absorption,
                                distribution_vd: profile.distribution_vd,
                                metabolic_stability: profile.metabolic_stability,
                                herg_risk: profile.herg_risk,
                                drug_likeness: profile.drug_likeness,
                                lipinski_violations: profile.lipinski_violations,
                                bbb_permeability: profile.bbb_permeability,
                                hepatotoxicity_risk: profile.hepatotoxicity_risk,
                                plasma_protein_binding: profile.plasma_protein_binding,
                            },
                        ));
                    }
                    Err(e) => {
                        let _ = state.tx.send(ServerMsg::PharmaError { message: e });
                    }
                }
            }
        }
    }
}

/// Pharma lab MD frame loop. Runs as a tokio task, sends frames at ~30fps when MD is active.
pub async fn pharma_frame_loop(state: Arc<AppState>) {
    loop {
        let should_step = {
            let lab = state.pharma_lab.lock().await;
            lab.as_ref().map_or(false, |l| l.is_md_running())
        };
        if should_step {
            let mut lab_guard = state.pharma_lab.lock().await;
            if let Some(lab) = lab_guard.as_mut() {
                let _stats = lab.step_md(10);
                let reactions = lab.check_proximity_reactions();
                let frame = lab.build_frame(&reactions);
                let _ = state.tx.send(ServerMsg::PharmaLabFrame(frame));
                for event in reactions {
                    let _ = state.tx.send(ServerMsg::PharmaReactionEvent(event));
                }
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(33)).await;
    }
}

/// The main simulation frame loop. Runs as a tokio task.
pub async fn frame_loop(state: Arc<AppState>) {
    loop {
        let (paused, target_fps, view, steps_per_frame) = {
            let params = state.params.read().await;
            (params.paused, params.target_fps, params.view, params.steps_per_frame)
        };

        let interval = std::time::Duration::from_millis(1000 / target_fps.max(1) as u64);

        if !paused {
            let mut world = state.world.lock().await;
            // Run multiple simulation steps per rendered frame for time acceleration
            for _ in 0..steps_per_frame.max(1) {
                let _ = world.step_frame();
            }

            let field = world.topdown_field(view);
            let atmosphere = world.atmosphere_frame();
            let terrain_visuals = build_terrain_visuals(&world, &atmosphere);
            let w = world.config.width;
            let h = world.config.height;
            let snapshot = world.snapshot();
            let params = state.params.read().await;
            let snapshot_history = state
                .record_snapshot_history(&snapshot, params.seed, params.preset.cli_name())
                .await;

            let mut fc = state.frame_count.lock().await;
            *fc += 1;
            let frame_num = *fc;

            let time_label = world.time_label();
            let daylight = snapshot.light;
            let solar = world.solar_state();

            // Per-frame: field + terrain_visuals + shader DataTexture arrays.
            let plane = w * h;
            let _ = state.tx.send(ServerMsg::Frame(FrameData {
                field,
                width: w,
                height: h,
                view: view.label().to_string(),
                atmosphere,
                terrain_surface: world.soil_structure.clone(),
                terrain_visuals,
                terrain_voxels: Vec::new(),
                terrain_cutaways: Vec::new(),
                daylight,
                sun_elevation_rad: solar.elevation_rad,
                sun_azimuth_rad: solar.azimuth_rad,
                sun_direction: solar.direction,
                time_label,
                paused: false,
                fps: target_fps,
                moisture: world.moisture[..plane].to_vec(),
                water_mask: world.water_mask[..plane].to_vec(),
                soil_structure: world.soil_structure[..plane].to_vec(),
            }));

            // Send entities every 2nd frame for smooth sync, snapshots less often
            let snapshot_interval = if state.tx.receiver_count() > 4 { 6 } else { 3 };
            // Entities every other frame for smooth visual sync
            let entity_interval = 2;
            // Send entities frequently for smooth visual sync
            if frame_num % entity_interval == 0 {
                let entities = AppState::extract_entities(&world);
                let _ = state.tx.send(ServerMsg::Entities(entities));
            }
            // Send full snapshot less frequently (expensive)
            if frame_num % snapshot_interval == 0 {
                let _ = state.tx.send(ServerMsg::Snapshot(SnapshotData {
                    snapshot,
                    preset: params.preset.cli_name().to_string(),
                    seed: params.seed,
                    field: None,
                    width: None,
                    height: None,
                    view: None,
                    atmosphere: None,
                    terrain_surface: None,
                    terrain_visuals: None,
                    terrain_voxels: None,
                    terrain_cutaways: None,
                    entities: None,
                    daylight: None,
                    time_label: None,
                    paused: None,
                    fps: None,
                }));
                let _ = state.tx.send(ServerMsg::SnapshotHistory(snapshot_history));
            }
        }

        tokio::time::sleep(interval).await;
    }
}
