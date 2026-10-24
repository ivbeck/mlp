# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Local training console. An axum server inside this repo (`web/`, own crate depending on the
`multilayer-perceptron` lib by path) runs training natively and streams every epoch to the
browser over a WebSocket. The browser UI is served by that same binary. Frontend framework:
asked in the direction round, see the surface brief.

## Users

People who have their own tabular data (a CSV of numbers, one target column) and want to
train a small multi-layer perceptron on it without learning the CLI flags. They run the
console on their own machine, load their file, shape a network, watch it train and then
probe the result. Comfortable with the idea of features, targets and loss; not necessarily
Rust users.

## Product Purpose

Turn a numeric CSV into a trained MLP you can watch learn and then interrogate. Success: a
user goes from "here is my file" to "this net predicts my target this well, and I can see
why it got there" in one sitting, without touching a terminal after starting the server.

## Positioning

The training is the real `multilayer-perceptron` Rust crate running natively on the user's
machine, not a JS re-implementation; data never leaves the machine. Every number on screen
is what the crate computed.

## Operating Context

- Desktop browser on the same machine as the server (localhost). Long-ish sessions while
  runs train; the screen is watched, not glanced at.
- Data arrives as CSV files, optional header, one target column. The CLI takes numbers
  only; the console also reads text categories, yes/no columns and ISO dates and encodes
  them itself (one-hot, top 30 values per column, missing as its own value or mean fill). `example_data.csv` (20 rows, 2 features, binary target) is the
  bundled sample.
- CLI parity: every CLI option (`--architecture`, `--hidden-activation`,
  `--output-activation`, `--loss`, `--optimizer`, `--learning-rate`, `--epochs`,
  `--batch-size`, `--validation-split`, `--no-shuffle`, `--early-stopping-patience`,
  `--normalize`) has a control in the UI.

## Capabilities and Constraints

- Activations: ReLU, Sigmoid, Tanh, Linear, LeakyReLU, Softmax. Losses: MSE, MAE, BCE,
  categorical CE, Huber. Optimizers: SGD, SGD+momentum, Adam, RMSprop.
- Training in the lib takes a single scalar target per sample (`Array1::from_elem(1, label)`),
  so the output layer is effectively one neuron. Multi-class training is not supported by the
  lib today; the UI must not pretend it is.
- Train/validation split is a plain tail split (no shuffle before split). Early stopping on
  validation loss. Metrics recorded: train and validation loss per epoch; accuracy vectors
  exist but are never filled.
- `normalize` divides by the feature std, a constant column gives NaN.
- Small lib changes are allowed for the UI (accessors such as `layers()`, a per-epoch step),
  kept within the crate's strict clippy config (pedantic/nursery deny, no print).

## Evidence on Hand

- `example_data.csv`. No real user datasets, screenshots, benchmarks or testimonials exist;
  do not invent any. Demo data shown in the UI is labelled as sample data.

## Product Principles

1. The numbers are the crate's. Nothing shown is estimated or smoothed without saying so.
2. Watching training is the core moment; it must be live, legible and interruptible.
3. Every CLI option has a home, but defaults mean a first run needs only a file and a target.
4. Say what the lib can't do (multi-class, NaN from constant columns) before the user hits it.

## Accessibility & Inclusion

No product-specific standard established. Keyboard operable controls and charts that do not
rely on colour alone.
