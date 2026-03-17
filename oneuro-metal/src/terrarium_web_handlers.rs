//! HTTP and WebSocket handlers for the terrarium web server.

use crate::terrarium_web_annotations::all_annotations;
use crate::terrarium_web_evolution::{start_evolution, stop_evolution};
use crate::terrarium_web_protocol::{
    parse_view, ClientMsg, FrameData, ServerMsg, SnapshotData,
};
use crate::terrarium_web_state::AppState;
use crate::terrarium_web_tournament::{
    build_leaderboard_data, delete_entry, get_genome, submit_genome, TournamentSubmitRequest,
};
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{Html, IntoResponse};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use std::sync::Arc;

/// Embedded HTML frontend.
const INDEX_HTML: &str = include_str!("../web/terrarium.html");

/// GET / — serve the embedded HTML page.
pub async fn index_handler() -> Html<&'static str> {
    Html(INDEX_HTML)
}

/// GET /api/snapshot — return the latest world snapshot as JSON.
pub async fn snapshot_handler(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let world = state.world.lock().await;
    let snapshot = world.snapshot();
    axum::Json(snapshot)
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
            return (StatusCode::UNAUTHORIZED, axum::Json(serde_json::json!({"error": "Unauthorized"}))).into_response();
        }
    }

    let id = submit_genome(
        state.tournament.clone(),
        state.tx.clone(),
        body.name,
        body.genome,
    ).await;

    axum::Json(serde_json::json!({"id": id, "status": "evaluating"})).into_response()
}

#[derive(Deserialize)]
pub struct LeaderboardQuery {
    #[serde(default = "default_sort")]
    sort: String,
}
fn default_sort() -> String { "biomass".into() }

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
        None => (StatusCode::NOT_FOUND, axum::Json(serde_json::json!({"error": "Not found"}))).into_response(),
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

/// GET /api/export/bundle
pub async fn export_bundle(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let world = state.world.lock().await;
    let snapshot = world.snapshot();
    let params = state.params.read().await;
    let telemetry = state.last_telemetry.lock().await;
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
            "seed": params.seed,
        },
        "evolution_telemetry": *telemetry,
        "tournament_leaderboard": leaderboard.entries,
    });

    axum::Json(bundle)
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
pub async fn auth_token(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let mut auth = state.auth.lock().await;
    let response = auth.generate_token();
    axum::Json(response)
}

// ---------------------------------------------------------------------------
// WebSocket handler
// ---------------------------------------------------------------------------

/// WS /ws — WebSocket handler for real-time frame streaming + commands.
pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws(socket, state))
}

async fn handle_ws(socket: WebSocket, state: Arc<AppState>) {
    let (mut ws_tx, mut ws_rx) = socket.split();

    // Subscribe to broadcast channel
    let mut broadcast_rx = state.tx.subscribe();

    // Spawn a task to forward broadcast messages to this WebSocket client
    let forward_task = tokio::spawn(async move {
        loop {
            match broadcast_rx.recv().await {
                Ok(msg) => {
                    // Encode Frame messages as binary for performance
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
                    if ws_tx.send(ws_msg).await.is_err() {
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
        if v < mn { mn = v; }
        if v > mx { mx = v; }
    }
    let range = if (mx - mn).abs() < 1e-12 { 1.0 } else { mx - mn };

    // Header (4 bytes) + field (2 bytes each) + null separator (1 byte) + JSON metadata
    let meta = serde_json::json!({
        "view": frame.view,
        "daylight": frame.daylight,
        "time_label": frame.time_label,
        "paused": frame.paused,
        "fps": frame.fps,
        "mn": mn,
        "mx": mx,
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

    // JSON metadata
    buf.extend_from_slice(&meta_bytes);

    Message::Binary(buf.into())
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
            let w = world.config.width;
            let h = world.config.height;
            let snapshot = world.snapshot();
            let entities = AppState::extract_entities(&world);
            let time_label = format!("{:.1}s", snapshot.time_s);

            let _ = state.tx.send(ServerMsg::Frame(FrameData {
                field,
                width: w,
                height: h,
                view: params.view.label().to_string(),
                daylight: snapshot.light,
                time_label,
                paused: true,
                fps: params.target_fps,
            }));
            let _ = state.tx.send(ServerMsg::Snapshot(SnapshotData { snapshot }));
            let _ = state.tx.send(ServerMsg::Entities(entities));
        }
        ClientMsg::Reset { seed } => {
            let config = crate::terrarium_world::TerrariumWorldConfig {
                width: 20,
                height: 16,
                depth: 2,
                seed,
                time_warp: 900.0,
                max_plants: 20,
                max_fruits: 16,
                ..crate::terrarium_world::TerrariumWorldConfig::default()
            };

            if let Ok(new_world) = crate::terrarium_world::TerrariumWorld::new(config) {
                let mut world = state.world.lock().await;
                *world = new_world;

                use rand::{Rng, SeedableRng};
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                for _ in 0..3 {
                    let x = rng.gen_range(1..19);
                    let y = rng.gen_range(1..15);
                    world.add_water(x, y, 150.0, 0.0008);
                }
                for _ in 0..6 {
                    let x = rng.gen_range(1..19);
                    let y = rng.gen_range(1..15);
                    let _ = world.add_plant(x, y, None, None);
                }
                for _ in 0..3 {
                    let x = rng.gen_range(1..19);
                    let y = rng.gen_range(1..15);
                    world.add_fruit(x, y, 0.8, None);
                }
                for i in 0..2u64 {
                    let x = rng.gen_range(2.0..18.0_f32);
                    let y = rng.gen_range(2.0..14.0_f32);
                    world.add_fly(
                        crate::drosophila::DrosophilaScale::Tiny,
                        x,
                        y,
                        seed.wrapping_add(i),
                    );
                }

                let mut params = state.params.write().await;
                params.seed = seed;
                params.paused = true;
            }
        }
        ClientMsg::Speed { fps } => {
            let mut params = state.params.write().await;
            params.target_fps = fps.clamp(1, 60);
        }
        ClientMsg::View { mode } => {
            let mut params = state.params.write().await;
            params.view = parse_view(&mode);
        }
        ClientMsg::AddPlant { x, y } => {
            let mut world = state.world.lock().await;
            let _ = world.add_plant(x, y, None, None);
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
        }
        ClientMsg::AddFruit { x, y } => {
            let mut world = state.world.lock().await;
            world.add_fruit(x, y, 0.8, None);
        }
        ClientMsg::AddWater { x, y } => {
            let mut world = state.world.lock().await;
            world.add_water(x, y, 150.0, 0.0008);
        }
        ClientMsg::EvolveStart { config } => {
            start_evolution(state.clone(), config).await;
        }
        ClientMsg::EvolveStop => {
            stop_evolution(state.clone()).await;
        }
        ClientMsg::EvolveApplyGenome { genome } => {
            match genome.build_world() {
                Ok(new_world) => {
                    let mut world = state.world.lock().await;
                    *world = new_world;
                    let mut params = state.params.write().await;
                    params.paused = true;
                }
                Err(e) => {
                    let _ = state.tx.send(ServerMsg::Error {
                        message: format!("Failed to apply genome: {}", e),
                    });
                }
            }
        }
    }
}

/// The main simulation frame loop. Runs as a tokio task.
pub async fn frame_loop(state: Arc<AppState>) {
    loop {
        let (paused, target_fps, view) = {
            let params = state.params.read().await;
            (params.paused, params.target_fps, params.view)
        };

        let interval = std::time::Duration::from_millis(1000 / target_fps.max(1) as u64);

        if !paused {
            let mut world = state.world.lock().await;
            let _ = world.step_frame();

            let field = world.topdown_field(view);
            let w = world.config.width;
            let h = world.config.height;
            let snapshot = world.snapshot();

            let mut fc = state.frame_count.lock().await;
            *fc += 1;
            let frame_num = *fc;

            let time_label = format!("{:.1}s", snapshot.time_s);
            let daylight = snapshot.light;

            let _ = state.tx.send(ServerMsg::Frame(FrameData {
                field,
                width: w,
                height: h,
                view: view.label().to_string(),
                daylight,
                time_label,
                paused: false,
                fps: target_fps,
            }));

            // Smart throttling: send snapshots less frequently when many clients connected
            let snapshot_interval = if state.tx.receiver_count() > 4 { 10 } else { 5 };
            if frame_num % snapshot_interval == 0 {
                let entities = AppState::extract_entities(&world);
                let _ = state
                    .tx
                    .send(ServerMsg::Snapshot(SnapshotData { snapshot }));
                let _ = state.tx.send(ServerMsg::Entities(entities));
            }
        }

        tokio::time::sleep(interval).await;
    }
}
