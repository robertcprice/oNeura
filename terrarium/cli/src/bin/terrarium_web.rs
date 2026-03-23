//! Interactive Terrarium Web App — serves an embedded HTML/JS frontend with
//! real-time WebSocket streaming of simulation state.
//!
//! Usage:
//!   terrarium_web [--port PORT] [--seed SEED] [--fps FPS] [--require-auth]
//!
//! Opens a browser-ready interface at http://localhost:<PORT> with:
//!   - Real-time topdown field rendering (6 view modes)
//!   - Interactive entity placement (plants, flies, fruits, water)
//!   - Live stats dashboard (37 snapshot fields)
//!   - Evolution control (8 modes including Pareto, stress, island, DE)
//!   - Tournament / citizen-science leaderboard
//!   - Educational annotations with interactive sliders
//!   - Research export bundle
//!   - Optional bearer-token authentication

use oneura_core::terrarium::TerrariumDemoPreset;
use oneura_core::terrarium_web_handlers::{
    annotations_handler, archive_handler, auth_token, checkpoint_handler, ecosystem_run_handler,
    ecosystem_snapshot_handler, ecosystem_start_handler, ecosystem_step_handler, export_bundle,
    frame_loop, import_checkpoint_handler, index_handler, inspect_handler, orbit_controls_handler,
    organism_lineage_handler, organism_rename_handler, organisms_handler, snapshot_handler,
    snapshot_history_handler, terrain_renderer_handler, plant_renderer_handler, scale_manager_handler, molecular_renderer_handler, organism_renderer_handler, cellular_renderer_handler, ssao_effect_handler, cutaway_controller_handler, density_volume_handler, three_js_handler, microcosm_handler, microcosm_js_handler, tournament_delete,
    tournament_genome, tournament_leaderboard, tournament_submit, ws_handler,
};
use oneura_core::terrarium_web_state::AppState;
use tower_http::compression::CompressionLayer;
use tower_http::cors::{Any, CorsLayer};

#[tokio::main]
async fn main() {
    let mut port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8420);
    let mut seed: u64 = std::env::var("SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let mut fps: u32 = std::env::var("FPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let mut require_auth = std::env::var("REQUIRE_AUTH")
        .ok()
        .map(|s| matches!(s.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false);
    let mut preset = std::env::var("WORLD_PRESET")
        .ok()
        .and_then(|s| TerrariumDemoPreset::parse(&s))
        .unwrap_or(TerrariumDemoPreset::MicroTerrarium);
    let mut climate_scenario: Option<String> = std::env::var("CLIMATE").ok();

    // Simple arg parsing
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--port" => {
                i += 1;
                port = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(port);
            }
            "--seed" => {
                i += 1;
                seed = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(seed);
            }
            "--fps" => {
                i += 1;
                fps = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(fps);
            }
            "--require-auth" => {
                require_auth = true;
            }
            "--preset" => {
                i += 1;
                preset = args
                    .get(i)
                    .and_then(|s| TerrariumDemoPreset::parse(s))
                    .unwrap_or(preset);
            }
            "--climate" => {
                i += 1;
                climate_scenario = args.get(i).cloned();
            }
            "--help" | "-h" => {
                eprintln!(
                    "Usage: terrarium_web [--port PORT] [--seed SEED] [--fps FPS] [--preset NAME] [--climate SCENARIO] [--require-auth]"
                );
                eprintln!("  --port PORT         HTTP port (default: 8420)");
                eprintln!("  --seed SEED         World seed (default: 42)");
                eprintln!("  --fps FPS           Target frames per second (default: 16)");
                eprintln!(
                    "  --preset NAME       Demo preset: demo | terrarium | aquarium (default: terrarium)"
                );
                eprintln!("  --climate SCENARIO  Climate scenario: rcp85 | rcp45 | rcp26 | preindustrial");
                eprintln!("  --require-auth      Require bearer tokens for mutation endpoints");
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    eprintln!(
        "Constructing world (seed={}, preset={})...",
        seed,
        preset.cli_name()
    );
    let state = match AppState::new_with_preset(seed, 64, require_auth, preset) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to create world: {}", e);
            std::process::exit(1);
        }
    };

    // Enable climate driver if requested
    if let Some(ref scenario_name) = climate_scenario {
        let mut world = state.world.lock().await;
        let scenario = match scenario_name.as_str() {
            "rcp85" => oneura_core::climate_scenarios::ClimateScenario::Rcp85,
            "rcp45" => oneura_core::climate_scenarios::ClimateScenario::Rcp45,
            "rcp26" => oneura_core::climate_scenarios::ClimateScenario::Rcp26,
            "preindustrial" => oneura_core::climate_scenarios::ClimateScenario::PreIndustrial,
            other => {
                eprintln!(
                    "Unknown climate scenario: '{}'. Use: rcp85, rcp45, rcp26, preindustrial",
                    other
                );
                std::process::exit(1);
            }
        };
        world.enable_climate_driver(scenario, seed, None);
        eprintln!("Climate driver enabled: {}", scenario_name);
    }

    // Set initial FPS
    {
        let mut params = state.params.write().await;
        params.target_fps = fps;
    }

    // Spawn the simulation frame loop
    let loop_state = state.clone();
    tokio::spawn(async move {
        frame_loop(loop_state).await;
    });

    // Build axum router
    let app = axum::Router::new()
        .route("/", axum::routing::get(index_handler))
        .route("/vendor/three.min.js", axum::routing::get(three_js_handler))
        .route(
            "/vendor/OrbitControls.js",
            axum::routing::get(orbit_controls_handler),
        )
        .route("/js/terrain_renderer.js", axum::routing::get(terrain_renderer_handler))
        .route("/js/plant_renderer.js", axum::routing::get(plant_renderer_handler))
        .route("/js/scale_manager.js", axum::routing::get(scale_manager_handler))
        .route("/js/molecular_renderer.js", axum::routing::get(molecular_renderer_handler))
        .route("/js/organism_renderer.js", axum::routing::get(organism_renderer_handler))
        .route("/js/cellular_renderer.js", axum::routing::get(cellular_renderer_handler))
        .route("/js/ssao_effect.js", axum::routing::get(ssao_effect_handler))
        .route("/js/cutaway_controller.js", axum::routing::get(cutaway_controller_handler))
        .route("/js/density_volume.js", axum::routing::get(density_volume_handler))
        .route("/microcosm", axum::routing::get(microcosm_handler))
        .route("/js/microcosm.js", axum::routing::get(microcosm_js_handler))
        .route("/healthz", axum::routing::get(|| async { "ok" }))
        .route("/ws", axum::routing::get(ws_handler))
        .route("/api/snapshot", axum::routing::get(snapshot_handler))
        .route(
            "/api/snapshot_history",
            axum::routing::get(snapshot_history_handler),
        )
        .route("/api/inspect", axum::routing::get(inspect_handler))
        .route("/api/organisms", axum::routing::get(organisms_handler))
        .route(
            "/api/organisms/{id}/lineage",
            axum::routing::get(organism_lineage_handler),
        )
        .route(
            "/api/organisms/{id}/name",
            axum::routing::post(organism_rename_handler),
        )
        .route("/api/archive", axum::routing::get(archive_handler))
        .route("/api/checkpoint", axum::routing::get(checkpoint_handler))
        .route(
            "/api/import/checkpoint",
            axum::routing::post(import_checkpoint_handler)
                .layer(axum::extract::DefaultBodyLimit::max(128 * 1024 * 1024)),
        )
        .route(
            "/api/tournament/submit",
            axum::routing::post(tournament_submit),
        )
        .route(
            "/api/tournament/leaderboard",
            axum::routing::get(tournament_leaderboard),
        )
        .route(
            "/api/tournament/genome/{id}",
            axum::routing::get(tournament_genome),
        )
        .route(
            "/api/tournament/{id}",
            axum::routing::delete(tournament_delete),
        )
        .route("/api/annotations", axum::routing::get(annotations_handler))
        .route("/api/export/bundle", axum::routing::get(export_bundle))
        .route("/api/auth/token", axum::routing::get(auth_token))
        .route(
            "/api/ecosystem/snapshot",
            axum::routing::get(ecosystem_snapshot_handler),
        )
        .route(
            "/api/ecosystem/start",
            axum::routing::post(ecosystem_start_handler),
        )
        .route(
            "/api/ecosystem/step",
            axum::routing::post(ecosystem_step_handler),
        )
        .route(
            "/api/ecosystem/run",
            axum::routing::post(ecosystem_run_handler),
        )
        .layer(CompressionLayer::new().gzip(true))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port);
    let climate_label = climate_scenario.as_deref().unwrap_or("none");
    eprintln!(
        "Terrarium web server listening on http://localhost:{} (preset={}, climate={})",
        port,
        preset.cli_name(),
        climate_label
    );
    if require_auth {
        eprintln!("  Authentication required for mutations (GET /api/auth/token to get a token)");
    }

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("Failed to bind");

    axum::serve(listener, app).await.expect("Server error");
}
