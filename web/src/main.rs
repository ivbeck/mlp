//! mlp console: a local web ui that trains `multilayer-perceptron` models and streams
//! every epoch to the browser

mod dataset;
mod run;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use axum::{
    body::Bytes,
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Path, Query, State,
    },
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use serde::Deserialize;
use serde_json::json;
use tokio::sync::broadcast::error::RecvError;
use tracing::{info, warn};

use crate::dataset::StoredDataset;
use crate::run::{lock, Command, Event, RunConfig, RunHandle};

const EXAMPLE_CSV: &str = include_str!("../../example_data.csv");
/// Largest upload accepted, the whole file is held in memory
const MAX_UPLOAD: usize = 64 * 1024 * 1024;

#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
    /// Address to listen on. Keep it on loopback unless you trust the network,
    /// there is no login
    #[arg(short, long, default_value = "127.0.0.1:7878")]
    bind: String,
}

#[derive(Debug, Default)]
struct App {
    datasets: Mutex<HashMap<String, StoredDataset>>,
    runs: Mutex<HashMap<String, RunHandle>>,
    next_id: Mutex<HashMap<&'static str, u64>>,
}

type Shared = Arc<App>;

impl App {
    fn id(&self, prefix: &'static str) -> String {
        let mut ids = lock(&self.next_id);
        let n = ids.entry(prefix).or_insert(0);
        *n += 1;
        format!("{prefix}{n}")
    }
}

struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "error": self.1 }))).into_response()
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> Self {
        Self(StatusCode::UNPROCESSABLE_ENTITY, format!("{e:#}"))
    }
}

fn not_found(what: &str) -> ApiError {
    ApiError(StatusCode::NOT_FOUND, format!("{what} not found"))
}

#[derive(Debug, Deserialize)]
struct UploadQuery {
    name: Option<String>,
    has_header: Option<bool>,
}

async fn upload(
    State(app): State<Shared>,
    Query(q): Query<UploadQuery>,
    body: Bytes,
) -> Result<impl IntoResponse, ApiError> {
    let id = app.id("d");
    let name = q.name.unwrap_or_else(|| "upload.csv".into());
    let stored = tokio::task::spawn_blocking(move || {
        dataset::parse(id, name, body.to_vec(), q.has_header, false)
    })
    .await
    .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))??;
    let info = stored.info.clone();
    info!(id = %info.id, name = %info.name, rows = info.rows, "dataset loaded");
    lock(&app.datasets).insert(info.id.clone(), stored);
    Ok(Json(info))
}

async fn example(State(app): State<Shared>) -> Result<impl IntoResponse, ApiError> {
    let id = app.id("d");
    let stored = dataset::parse(
        id,
        "example_data.csv".into(),
        EXAMPLE_CSV.as_bytes().to_vec(),
        Some(true),
        true,
    )?;
    let info = stored.info.clone();
    lock(&app.datasets).insert(info.id.clone(), stored);
    Ok(Json(info))
}

async fn get_dataset(
    State(app): State<Shared>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    lock(&app.datasets)
        .get(&id)
        .map(|d| Json(d.info.clone()))
        .ok_or_else(|| not_found("Dataset"))
}

async fn start_run(
    State(app): State<Shared>,
    Json(config): Json<RunConfig>,
) -> Result<impl IntoResponse, ApiError> {
    let stored = lock(&app.datasets)
        .get(&config.dataset_id)
        .cloned()
        .ok_or_else(|| not_found("Dataset"))?;
    let id = app.id("r");
    let handle = tokio::task::spawn_blocking(move || run::start(id, config, &stored))
        .await
        .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))??;
    let summary = lock(&handle.shared).summary.clone();
    info!(id = %summary.id, arch = ?summary.architecture, "run started");
    lock(&app.runs).insert(summary.id.clone(), handle);
    Ok(Json(summary))
}

async fn list_runs(State(app): State<Shared>) -> impl IntoResponse {
    let runs = lock(&app.runs);
    let mut out: Vec<_> = runs
        .values()
        .map(|h| {
            let s = lock(&h.shared);
            json!({ "summary": s.summary, "history": s.history })
        })
        .collect();
    drop(runs);
    out.sort_by_key(|v| v["summary"]["started_at_ms"].as_u64().unwrap_or(0));
    Json(out)
}

async fn control(
    State(app): State<Shared>,
    Path(id): Path<String>,
    Json(cmd): Json<serde_json::Value>,
) -> Result<impl IntoResponse, ApiError> {
    let cmd: Command = serde_json::from_value(cmd["cmd"].clone())
        .map_err(|e| ApiError(StatusCode::BAD_REQUEST, e.to_string()))?;
    let runs = lock(&app.runs);
    let handle = runs.get(&id).ok_or_else(|| not_found("Run"))?;
    if lock(&handle.shared).summary.state.is_over() {
        return Err(ApiError(StatusCode::CONFLICT, "Run is over".into()));
    }
    handle
        .commands
        .send(cmd)
        .map_err(|_| ApiError(StatusCode::CONFLICT, "Run is over".into()))?;
    drop(runs);
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Debug, Deserialize)]
struct PlaneBody {
    x: usize,
    y: usize,
}

async fn set_plane(
    State(app): State<Shared>,
    Path(id): Path<String>,
    Json(body): Json<PlaneBody>,
) -> Result<impl IntoResponse, ApiError> {
    let handle = lock(&app.runs)
        .get(&id)
        .cloned()
        .ok_or_else(|| not_found("Run"))?;
    let snap = tokio::task::spawn_blocking(move || handle.set_plane(body.x, body.y))
    .await
    .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))??;
    Ok(Json(snap))
}

#[derive(Debug, Deserialize)]
struct PredictBody {
    /// Raw feature values in file units, target column left out
    inputs: Vec<f64>,
}

async fn predict(
    State(app): State<Shared>,
    Path(id): Path<String>,
    Json(body): Json<PredictBody>,
) -> Result<impl IntoResponse, ApiError> {
    let model = {
        let runs = lock(&app.runs);
        let handle = runs.get(&id).ok_or_else(|| not_found("Run"))?;
        Arc::clone(&handle.model)
    };
    let out = tokio::task::spawn_blocking(move || {
        let mut m = lock(&model);
        let want = m.mlp.layers().first().map_or(0, multilayer_perceptron::layer::Layer::input_size);
        if body.inputs.len() != want {
            return Err(ApiError(
                StatusCode::BAD_REQUEST,
                format!("Expected {want} inputs, got {}", body.inputs.len()),
            ));
        }
        let input = m.to_net_input(&body.inputs);
        let layers: Vec<Vec<f64>> = m
            .mlp
            .forward_trace(&input)
            .into_iter()
            .map(|a| a.to_vec())
            .collect();
        let output = layers.last().and_then(|l| l.first()).copied();
        Ok(json!({ "input": input.to_vec(), "layers": layers, "output": output }))
    })
    .await
    .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))??;
    Ok(Json(out))
}

async fn stream(
    State(app): State<Shared>,
    Path(id): Path<String>,
    ws: WebSocketUpgrade,
) -> Result<Response, ApiError> {
    let (rx, hello) = {
        let runs = lock(&app.runs);
        let handle = runs.get(&id).ok_or_else(|| not_found("Run"))?;
        // Subscribe before reading history so no epoch falls in between
        let rx = handle.events.subscribe();
        let s = lock(&handle.shared);
        let hello = json!({
            "type": "hello",
            "summary": s.summary,
            "history": s.history,
            "snapshot": s.last_snapshot,
        });
        drop(s);
        (rx, hello)
    };
    Ok(ws.on_upgrade(move |socket| pump(socket, rx, hello)))
}

async fn pump(
    mut socket: WebSocket,
    mut rx: tokio::sync::broadcast::Receiver<Event>,
    hello: serde_json::Value,
) {
    let last_epoch = hello["history"]
        .as_array()
        .and_then(|h| h.last())
        .and_then(|r| r["epoch"].as_u64())
        .unwrap_or(0);
    if socket
        .send(Message::Text(hello.to_string().into()))
        .await
        .is_err()
    {
        return;
    }
    loop {
        let event = match rx.recv().await {
            Ok(e) => e,
            Err(RecvError::Lagged(n)) => {
                warn!(skipped = n, "slow websocket client");
                if socket
                    .send(Message::Text(
                        json!({ "type": "lagged" }).to_string().into(),
                    ))
                    .await
                    .is_err()
                {
                    return;
                }
                continue;
            }
            Err(RecvError::Closed) => return,
        };
        // Epochs already in the hello history may also sit in the channel
        if let Event::Epoch(r) = &event {
            if r.epoch as u64 <= last_epoch {
                continue;
            }
        }
        let Ok(text) = serde_json::to_string(&event) else {
            continue;
        };
        if socket.send(Message::Text(text.into())).await.is_err() {
            return;
        }
    }
}

/// The built frontend (`npm run build` in web/ui). Debug builds read it from disk,
/// release builds carry it inside the binary.
#[derive(rust_embed::Embed)]
#[folder = "ui/dist/"]
struct Ui;

async fn ui(uri: axum::http::Uri) -> Response {
    let path = uri.path().trim_start_matches('/');
    let (path, file) = match Ui::get(path) {
        Some(f) if !path.is_empty() => (path, f),
        // Unknown paths get the app shell, the ui routes on the client
        _ => match Ui::get("index.html") {
            Some(f) => ("index.html", f),
            None => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Frontend not built. Run `npm ci && npm run build` in web/ui, then reload.",
                )
                    .into_response()
            }
        },
    };
    let mime = mime_guess::from_path(path).first_or_octet_stream();
    let cache = if path.starts_with("assets/") {
        "public, max-age=31536000, immutable"
    } else {
        "no-cache"
    };
    (
        [
            (header::CONTENT_TYPE, mime.as_ref().to_string()),
            (header::CACHE_CONTROL, cache.to_string()),
        ],
        file.data,
    )
        .into_response()
}

// tokio::main expands to an expect
#[allow(clippy::unwrap_in_result)]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    let args = Args::parse();
    let app: Shared = Arc::new(App::default());

    let router = Router::new()
        .route("/api/datasets", post(upload))
        .route("/api/datasets/example", post(example))
        .route("/api/datasets/{id}", get(get_dataset))
        .route("/api/runs", get(list_runs).post(start_run))
        .route("/api/runs/{id}/control", post(control))
        .route("/api/runs/{id}/predict", post(predict))
        .route("/api/runs/{id}/plane", post(set_plane))
        .route("/api/runs/{id}/ws", get(stream))
        .fallback(ui)
        .layer(axum::extract::DefaultBodyLimit::max(MAX_UPLOAD))
        .with_state(app);

    let listener = tokio::net::TcpListener::bind(&args.bind).await?;
    info!("mlp console on http://{}", listener.local_addr()?);
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
