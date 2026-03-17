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

use oneuro_metal::terrarium_web_handlers::{
    annotations_handler, auth_token, export_bundle, frame_loop, index_handler,
    snapshot_handler, tournament_delete, tournament_genome, tournament_leaderboard,
    tournament_submit, ws_handler,
};
use oneuro_metal::terrarium_web_state::AppState;

#[tokio::main]
async fn main() {
    let mut port: u16 = 8420;
    let mut seed: u64 = 42;
    let mut fps: u32 = 10;
    let mut require_auth = false;

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
            "--help" | "-h" => {
                eprintln!("Usage: terrarium_web [--port PORT] [--seed SEED] [--fps FPS] [--require-auth]");
                eprintln!("  --port PORT      HTTP port (default: 8420)");
                eprintln!("  --seed SEED      World seed (default: 42)");
                eprintln!("  --fps FPS        Target frames per second (default: 10)");
                eprintln!("  --require-auth   Require bearer tokens for mutation endpoints");
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    eprintln!("Constructing world (seed={})...", seed);
    let state = match AppState::new(seed, 64, require_auth) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to create world: {}", e);
            std::process::exit(1);
        }
    };

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
        .route("/ws", axum::routing::get(ws_handler))
        .route("/api/snapshot", axum::routing::get(snapshot_handler))
        .route("/api/tournament/submit", axum::routing::post(tournament_submit))
        .route("/api/tournament/leaderboard", axum::routing::get(tournament_leaderboard))
        .route("/api/tournament/genome/{id}", axum::routing::get(tournament_genome))
        .route("/api/tournament/{id}", axum::routing::delete(tournament_delete))
        .route("/api/annotations", axum::routing::get(annotations_handler))
        .route("/api/export/bundle", axum::routing::get(export_bundle))
        .route("/api/auth/token", axum::routing::get(auth_token))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port);
    eprintln!("Terrarium web server listening on http://localhost:{}", port);
    if require_auth {
        eprintln!("  Authentication required for mutations (GET /api/auth/token to get a token)");
    }

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("Failed to bind");

    axum::serve(listener, app)
        .await
        .expect("Server error");
}
