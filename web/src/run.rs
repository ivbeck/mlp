//! One training run: the model lives on its own thread, the ui steers it with
//! commands and listens to a broadcast of epoch events

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use multilayer_perceptron::{
    activation::{ActivationFunction, LeakyReLU, Linear, ReLU, Sigmoid, Tanh},
    data::Dataset,
    loss::{BinaryCrossEntropy, HuberLoss, LossFunction, MeanAbsoluteError, MeanSquaredError},
    mlp::MLPBuilder,
    optimizer::{Adam, Optimizer, RMSprop, SGDMomentum, SGD},
    MultilayerPerceptron,
};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc};

use crate::dataset::Kind;
use crate::encode::{Cell, Encoder, InputInfo, TargetSpec};

/// Cells per side of the prediction field
const FIELD_RES: usize = 48;
/// Above this many parameters weight snapshots are not streamed
const MAX_STREAMED_PARAMS: usize = 20_000;
/// Fastest the ui gets a snapshot (weights + field), epochs still stream every time
const SNAPSHOT_EVERY: Duration = Duration::from_millis(90);

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LayerSpec {
    pub size: usize,
    pub activation: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct RunConfig {
    pub dataset_id: String,
    pub target: usize,
    pub normalize: bool,
    pub hidden: Vec<LayerSpec>,
    pub output_activation: String,
    pub loss: String,
    pub optimizer: String,
    pub learning_rate: f64,
    #[serde(default = "default_momentum")]
    pub momentum: f64,
    pub epochs: usize,
    pub batch_size: usize,
    pub validation_split: f64,
    pub shuffle: bool,
    pub early_stopping_patience: Option<usize>,
    /// Shuffle the rows once before the tail split, the CLI never does this
    #[serde(default)]
    pub shuffle_before_split: bool,
    /// Positions in the run's input columns (numeric ones) for the field plane
    #[serde(default)]
    pub plane: Option<(usize, usize)>,
    /// Column kinds chosen in the ui, by column index; missing ones keep the detected kind
    #[serde(default)]
    pub kinds: HashMap<usize, Kind>,
    /// Which of a two-valued target's values counts as 1
    #[serde(default)]
    pub target_positive: Option<String>,
    /// Epochs per second cap, 0 = as fast as the machine goes. Changes wall clock only.
    #[serde(default)]
    pub pace: f64,
    #[serde(default)]
    pub label: String,
}

const fn default_momentum() -> f64 {
    0.9
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RunState {
    Running,
    Paused,
    Finished,
    EarlyStopped,
    Stopped,
    Failed,
}

impl RunState {
    pub const fn is_over(self) -> bool {
        matches!(
            self,
            Self::Finished | Self::EarlyStopped | Self::Stopped | Self::Failed
        )
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct EpochRecord {
    pub epoch: usize,
    pub train_loss: f64,
    pub val_loss: Option<f64>,
    /// Accuracy at threshold 0.5, only for 0/1 targets
    pub train_acc: Option<f64>,
    pub val_acc: Option<f64>,
    pub best: bool,
    pub since_best: usize,
    pub ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Snapshot {
    pub epoch: usize,
    /// Per layer, weights row major (`input_size` x `output_size`), None when the net is too big
    pub weights: Option<Vec<Vec<f64>>>,
    pub biases: Option<Vec<Vec<f64>>>,
    pub field: Option<Field>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Field {
    pub x: usize,
    pub y: usize,
    /// Raw (unnormalized) feature ranges covered
    pub x_range: (f64, f64),
    pub y_range: (f64, f64),
    pub res: usize,
    /// Row major from (`x_min`, `y_min`), network output
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    Epoch(EpochRecord),
    Snapshot(Snapshot),
    State {
        state: RunState,
        reason: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Command {
    Pause,
    Resume,
    Step,
    Stop,
}

#[derive(Debug, Clone, Serialize)]
pub struct Note {
    /// "info" or "warn"
    pub level: &'static str,
    pub text: String,
}

impl Note {
    fn info(text: impl Into<String>) -> Self {
        Self { level: "info", text: text.into() }
    }
    fn warn(text: impl Into<String>) -> Self {
        Self { level: "warn", text: text.into() }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RunSummary {
    pub id: String,
    pub config: RunConfig,
    pub architecture: Vec<usize>,
    pub parameters: usize,
    pub dataset_name: String,
    pub features: Vec<String>,
    pub target_name: String,
    pub binary_target: bool,
    pub train_rows: usize,
    pub val_rows: usize,
    pub state: RunState,
    pub reason: Option<String>,
    pub notes: Vec<Note>,
    /// Source columns feeding the network, in encoding order
    pub inputs: Vec<InputInfo>,
    pub target: TargetSpec,
    pub started_at_ms: u128,
}

/// What the http side shares with the thread
#[derive(Debug)]
pub struct Shared {
    pub summary: RunSummary,
    pub history: Vec<EpochRecord>,
    pub last_snapshot: Option<Snapshot>,
}

#[derive(Debug, Clone)]
pub struct RunHandle {
    pub shared: Arc<Mutex<Shared>>,
    pub model: Arc<Mutex<Model>>,
    pub events: broadcast::Sender<Event>,
    pub commands: mpsc::UnboundedSender<Command>,
    pub plane: Arc<FieldPlane>,
}

impl RunHandle {
    /// Point the field at two other features and return a fresh snapshot right away
    ///
    /// # Errors
    ///
    /// Indices out of range or equal
    pub fn set_plane(&self, x: usize, y: usize) -> Result<Snapshot> {
        let numeric = |i: usize| self.plane.ranges.get(i).copied().flatten().is_some();
        if x == y || !numeric(x) || !numeric(y) {
            anyhow::bail!("The chart needs two different number columns");
        }
        *lock(&self.plane.plane) = Some((x, y));
        let mut m = lock(&self.model);
        let epoch = lock(&self.shared).history.last().map_or(0, |r| r.epoch);
        let snap = snapshot(&mut m, epoch, &self.plane);
        drop(m);
        lock(&self.shared).last_snapshot = Some(snap.clone());
        Ok(snap)
    }
}

/// The trained network plus what is needed to feed it raw values
#[derive(Debug)]
pub struct Model {
    pub mlp: MultilayerPerceptron,
    pub encoder: Arc<Encoder>,
    pub mean: Option<Array1<f64>>,
    pub std: Option<Array1<f64>>,
}

impl Model {
    /// One raw value per input column, encoded and scaled like the training rows
    pub fn to_net_input(&self, cells: &[Cell]) -> Array1<f64> {
        let x = Array1::from_vec(self.encoder.encode(cells));
        match (&self.mean, &self.std) {
            (Some(m), Some(s)) => (&x - m) / s,
            _ => x,
        }
    }
}

pub fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn activation(name: &str) -> Result<Arc<dyn ActivationFunction>> {
    Ok(match name {
        "relu" => Arc::new(ReLU),
        "leaky_relu" => Arc::new(LeakyReLU::default()),
        "sigmoid" => Arc::new(Sigmoid),
        "tanh" => Arc::new(Tanh),
        "linear" => Arc::new(Linear),
        other => anyhow::bail!("Unknown activation function: {other}"),
    })
}

fn loss(name: &str) -> Result<Arc<dyn LossFunction>> {
    Ok(match name {
        "mse" => Arc::new(MeanSquaredError),
        "mae" => Arc::new(MeanAbsoluteError),
        "bce" => Arc::new(BinaryCrossEntropy::default()),
        "huber" => Arc::new(HuberLoss::default()),
        other => anyhow::bail!("Unknown loss function: {other}"),
    })
}

fn optimizer(name: &str, lr: f64, momentum: f64) -> Result<Box<dyn Optimizer>> {
    Ok(match name {
        "sgd" => Box::new(SGD::new(lr)),
        "momentum" => Box::new(SGDMomentum::new(lr, momentum)),
        "adam" => Box::new(Adam::new(lr)),
        "rmsprop" => Box::new(RMSprop::new(lr)),
        other => anyhow::bail!("Unknown optimizer: {other}"),
    })
}

#[allow(clippy::cast_precision_loss)]
fn accuracy(mlp: &mut MultilayerPerceptron, data: &Dataset) -> f64 {
    let n = data.num_samples();
    if n == 0 {
        return 0.0;
    }
    let hits = (0..n)
        .filter(|&i| {
            let out = mlp.predict(&data.features.row(i).to_owned());
            (out[0] >= 0.5) == (data.labels[i] >= 0.5)
        })
        .count();
    hits as f64 / n as f64
}

struct Prepared {
    train: Dataset,
    val: Option<Dataset>,
    model: Model,
    summary: RunSummary,
}

#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
fn prepare(
    id: String,
    config: RunConfig,
    stored: &crate::dataset::StoredDataset,
) -> Result<Prepared> {
    let info = &stored.info;
    if config.target >= info.columns.len() {
        anyhow::bail!("Target column {} does not exist", config.target);
    }
    if config.epochs == 0 || config.batch_size == 0 {
        anyhow::bail!("Epochs and batch size must be at least 1");
    }
    if !(0.0..1.0).contains(&config.validation_split) {
        anyhow::bail!("Validation split must be in [0, 1)");
    }
    let mut notes = Vec::new();
    let target_col = &info.columns[config.target];
    let encoder = Encoder::build(
        info,
        &stored.rows,
        config.target,
        &config.kinds,
        config.target_positive.as_deref(),
    )?;
    let (mut dataset, dropped) = encoder.dataset(&stored.rows)?;
    let features = encoder.names.clone();
    let binary = matches!(encoder.target, TargetSpec::Binary { .. });
    if dropped > 0 {
        notes.push(Note::info(format!(
            "{dropped} rows have no {} and were left out.",
            target_col.name
        )));
    }
    let wide: Vec<String> = encoder
        .inputs
        .iter()
        .filter(|i| i.width > 1)
        .map(|i| format!("{} → {}", i.name, i.width))
        .collect();
    if !wide.is_empty() {
        notes.push(Note::info(format!(
            "{} columns became {} network inputs ({}).",
            encoder.inputs.len(),
            features.len(),
            wide.join(", ")
        )));
    }
    if let TargetSpec::Binary { one, zero } = &encoder.target {
        notes.push(Note::info(format!(
            "{} = {one} is class 1, {zero} is class 0.",
            target_col.name
        )));
    }

    if target_col.constant {
        notes.push(Note::warn(format!(
            "{} never changes, there is nothing to learn from it.",
            target_col.name
        )));
    }
    if config.shuffle_before_split {
        dataset = dataset.shuffle();
        notes.push(Note::info("Rows shuffled once before the validation split."));
    }

    let (mean, std) = if config.normalize {
        let (_, mean, mut std) = dataset.normalize();
        let constant: Vec<&str> = std
            .iter()
            .zip(&features)
            .filter(|(s, _)| **s == 0.0)
            .map(|(_, n)| n.as_str())
            .collect();
        if !constant.is_empty() {
            notes.push(Note::warn(format!(
                "Constant column(s) {} left unscaled; the crate's normalize would divide by zero there.",
                constant.join(", ")
            )));
            std.mapv_inplace(|s| if s == 0.0 { 1.0 } else { s });
        }
        dataset = dataset.apply_normalization(&mean, &std);
        (Some(mean), Some(std))
    } else {
        (None, None)
    };

    let (train, val) = if config.validation_split > 0.0 {
        let (t, v) = dataset.train_validation_split(config.validation_split);
        if v.num_samples() == 0 {
            (t, None)
        } else {
            (t, Some(v))
        }
    } else {
        (dataset, None)
    };
    if train.num_samples() == 0 {
        anyhow::bail!("Validation split leaves no rows to train on");
    }
    if let Some(v) = &val {
        if binary {
            let ones = v.labels.iter().filter(|&&l| l >= 0.5).count();
            if ones == 0 || ones == v.num_samples() {
                notes.push(Note::warn(if config.shuffle_before_split {
                    "Validation rows hold only one class, so validation accuracy says little."
                } else {
                    "Validation rows hold only one class. The split takes the last rows in file order; shuffle before the split."
                }));
            }
        }
    }

    let mut builder = MLPBuilder::new().input_size(train.num_features());
    for layer in &config.hidden {
        if layer.size == 0 {
            anyhow::bail!("Layer sizes must be at least 1");
        }
        builder = builder.layer(layer.size, activation(&layer.activation)?);
    }
    let mlp = builder
        .layer(1, activation(&config.output_activation)?)
        .loss_function(loss(&config.loss)?)
        .optimizer(optimizer(
            &config.optimizer,
            config.learning_rate,
            config.momentum,
        )?)
        .build()?;

    let summary = RunSummary {
        id,
        architecture: mlp.architecture(),
        parameters: mlp.num_parameters(),
        dataset_name: info.name.clone(),
        features,
        target_name: target_col.name.clone(),
        binary_target: binary,
        train_rows: train.num_samples(),
        val_rows: val.as_ref().map_or(0, Dataset::num_samples),
        state: RunState::Running,
        reason: None,
        notes,
        inputs: encoder.inputs.clone(),
        target: encoder.target.clone(),
        started_at_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis()),
        config,
    };
    Ok(Prepared {
        train,
        val,
        model: Model {
            mlp,
            encoder: Arc::new(encoder),
            mean,
            std,
        },
        summary,
    })
}

#[allow(clippy::cast_precision_loss)]
fn snapshot(model: &mut Model, epoch: usize, p: &FieldPlane) -> Snapshot {
    let (weights, biases) = if model.mlp.num_parameters() <= MAX_STREAMED_PARAMS {
        let layers = model.mlp.layers();
        (
            Some(
                layers
                    .iter()
                    .map(|l| l.weights.iter().copied().collect())
                    .collect(),
            ),
            Some(layers.iter().map(|l| l.biases.to_vec()).collect()),
        )
    } else {
        (None, None)
    };
    let plane = *lock(&p.plane);
    let field = plane.and_then(|(x, y)| {
        let (x0, x1) = pad(p.ranges.get(x).copied().flatten()?);
        let (y0, y1) = pad(p.ranges.get(y).copied().flatten()?);
        let mut raw = p.defaults.clone();
        let mut values = Vec::with_capacity(FIELD_RES * FIELD_RES);
        let step = (FIELD_RES - 1) as f64;
        for j in 0..FIELD_RES {
            raw[y] = Cell::Num((y1 - y0).mul_add(j as f64 / step, y0));
            for i in 0..FIELD_RES {
                raw[x] = Cell::Num((x1 - x0).mul_add(i as f64 / step, x0));
                let input = model.to_net_input(&raw);
                values.push(model.mlp.predict(&input)[0]);
            }
        }
        Some(Field {
            x,
            y,
            x_range: (x0, x1),
            y_range: (y0, y1),
            res: FIELD_RES,
            values,
        })
    });
    Snapshot {
        epoch,
        weights,
        biases,
        field,
    }
}

fn pad((lo, hi): (f64, f64)) -> (f64, f64) {
    let span = if hi > lo { hi - lo } else { 1.0 };
    (span.mul_add(-0.08, lo), span.mul_add(0.08, hi))
}

#[derive(Debug)]
pub struct FieldPlane {
    plane: Mutex<Option<(usize, usize)>>,
    /// Raw min/max per input column, None for the ones that are not numbers
    ranges: Vec<Option<(f64, f64)>>,
    /// Where every input sits when it is not on the chart: mean or most common value
    defaults: Vec<Cell>,
}

/// Validate the config, build the model and start the thread
///
/// # Errors
///
/// Bad config, or a dataset the crate refuses
pub fn start(
    id: String,
    config: RunConfig,
    stored: &crate::dataset::StoredDataset,
) -> Result<RunHandle> {
    let p = prepare(id, config, stored)?;
    let inputs = &p.summary.inputs;
    let ranges: Vec<Option<(f64, f64)>> = inputs
        .iter()
        .map(|i| match (i.kind, i.min, i.max) {
            (Kind::Numeric, Some(lo), Some(hi)) => Some((lo, hi)),
            _ => None,
        })
        .collect();
    let numeric: Vec<usize> = (0..ranges.len()).filter(|&i| ranges[i].is_some()).collect();
    let plane = p
        .summary
        .config
        .plane
        .filter(|(x, y)| x != y && numeric.contains(x) && numeric.contains(y))
        .or_else(|| (numeric.len() >= 2).then(|| (numeric[0], numeric[1])));
    let field_plane = Arc::new(FieldPlane {
        plane: Mutex::new(plane),
        ranges,
        defaults: inputs.iter().map(|i| i.default.clone()).collect(),
    });
    let (events, _) = broadcast::channel(4096);
    let (commands, rx) = mpsc::unbounded_channel();
    let shared = Arc::new(Mutex::new(Shared {
        summary: p.summary,
        history: Vec::new(),
        last_snapshot: None,
    }));
    let model = Arc::new(Mutex::new(p.model));

    {
        let mut m = lock(&model);
        let first = snapshot(&mut m, 0, &field_plane);
        lock(&shared).last_snapshot = Some(first);
    }

    let handle = RunHandle {
        shared: Arc::clone(&shared),
        model: Arc::clone(&model),
        events: events.clone(),
        commands,
        plane: Arc::clone(&field_plane),
    };
    std::thread::Builder::new()
        .name("mlp-run".into())
        .spawn(move || {
            let result = train_loop(
                &shared,
                &model,
                &events,
                rx,
                &p.train,
                p.val.as_ref(),
                &field_plane,
            );
            let (state, reason) = match result {
                Ok(end) => end,
                Err(e) => (RunState::Failed, Some(format!("{e:#}"))),
            };
            let mut s = lock(&shared);
            s.summary.state = state;
            s.summary.reason.clone_from(&reason);
            drop(s);
            let _ = events.send(Event::State { state, reason });
        })
        .context("Failed to spawn training thread")?;
    Ok(handle)
}

#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
fn train_loop(
    shared: &Mutex<Shared>,
    model: &Mutex<Model>,
    events: &broadcast::Sender<Event>,
    mut rx: mpsc::UnboundedReceiver<Command>,
    train: &Dataset,
    val: Option<&Dataset>,
    plane: &FieldPlane,
) -> Result<(RunState, Option<String>)> {
    let config = lock(shared).summary.config.clone();
    let binary = lock(shared).summary.binary_target;
    let mut paused = false;
    let mut steps_left = 0usize;
    let mut best = f64::INFINITY;
    let mut since_best = 0usize;
    let mut last_snap: Option<Instant> = None;
    let set_state = |state: RunState| {
        lock(shared).summary.state = state;
        let _ = events.send(Event::State {
            state,
            reason: None,
        });
    };

    for epoch in 1..=config.epochs {
        // Commands between epochs, blocking while paused
        loop {
            let cmd = if paused && steps_left == 0 {
                match rx.blocking_recv() {
                    Some(c) => Some(c),
                    None => return Ok((RunState::Stopped, Some("Console closed".into()))),
                }
            } else {
                rx.try_recv().ok()
            };
            match cmd {
                Some(Command::Stop) => {
                    return Ok((
                        RunState::Stopped,
                        Some(format!("Stopped before epoch {epoch}")),
                    ))
                }
                Some(Command::Pause) => {
                    if !paused {
                        paused = true;
                        set_state(RunState::Paused);
                    }
                }
                Some(Command::Resume) => {
                    if paused {
                        paused = false;
                        steps_left = 0;
                        set_state(RunState::Running);
                    }
                }
                Some(Command::Step) => {
                    if !paused {
                        paused = true;
                        set_state(RunState::Paused);
                    }
                    steps_left += 1;
                }
                None => break,
            }
        }
        if paused {
            steps_left = steps_left.saturating_sub(1);
        }

        let started = Instant::now();
        let mut m = lock(model);
        let train_loss = m
            .mlp
            .train_epoch(train, config.batch_size, config.shuffle)?;
        let val_loss = match val {
            Some(v) => Some(m.mlp.evaluate(v)?),
            None => None,
        };
        let (train_acc, val_acc) = if binary {
            (
                Some(accuracy(&mut m.mlp, train)),
                val.map(|v| accuracy(&mut m.mlp, v)),
            )
        } else {
            (None, None)
        };
        let watched = val_loss.unwrap_or(train_loss);
        let improved = watched < best;
        if improved {
            best = watched;
            since_best = 0;
        } else {
            since_best += 1;
        }
        let record = EpochRecord {
            epoch,
            train_loss,
            val_loss,
            train_acc,
            val_acc,
            best: improved,
            since_best,
            ms: started.elapsed().as_secs_f64() * 1000.0,
        };
        let snap = (last_snap.is_none_or(|t| t.elapsed() >= SNAPSHOT_EVERY) || epoch == config.epochs || paused)
            .then(|| snapshot(&mut m, epoch, plane));
        drop(m);

        {
            let mut s = lock(shared);
            s.history.push(record.clone());
            if let Some(snap) = &snap {
                s.last_snapshot = Some(snap.clone());
            }
        }
        let _ = events.send(Event::Epoch(record));
        if let Some(snap) = snap {
            last_snap = Some(Instant::now());
            let _ = events.send(Event::Snapshot(snap));
        }

        if !train_loss.is_finite() || val_loss.is_some_and(|v| !v.is_finite()) {
            return Ok((
                RunState::Failed,
                Some(format!(
                    "Loss is not a number at epoch {epoch}; lower the learning rate or normalize"
                )),
            ));
        }
        // Same rule as MultilayerPerceptron::train: only validation loss stops early
        if let (Some(patience), Some(_)) = (config.early_stopping_patience, val_loss) {
            if since_best >= patience {
                let mut m = lock(model);
                let snap = snapshot(&mut m, epoch, plane);
                drop(m);
                lock(shared).last_snapshot = Some(snap.clone());
                let _ = events.send(Event::Snapshot(snap));
                return Ok((
                    RunState::EarlyStopped,
                    Some(format!(
                        "No validation improvement for {patience} epochs, stopped at {epoch}"
                    )),
                ));
            }
        }
        if config.pace > 0.0 && !paused {
            let budget = Duration::from_secs_f64(1.0 / config.pace);
            if let Some(rest) = budget.checked_sub(started.elapsed()) {
                std::thread::sleep(rest);
            }
        }
    }
    Ok((RunState::Finished, None))
}
