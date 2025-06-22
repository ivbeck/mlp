# Multi-layer Perceptron in Rust

A flexible implementation of multi-layer perceptrons (MLPs) for training on preprocessed data, built in Rust with full Nix support.

## Features

- **Flexible Architecture**: Define arbitrary network architectures with any number of layers
- **Multiple Activation Functions**: ReLU, Sigmoid, Tanh, Linear, LeakyReLU, Softmax
- **Various Loss Functions**: Mean Squared Error, Binary Cross-Entropy, Categorical Cross-Entropy, Huber Loss
- **Advanced Optimizers**: SGD, SGD with Momentum, Adam, RMSprop
- **Data Handling**: CSV loading, normalization, train/validation splits, mini-batch processing
- **Training Features**: Early stopping, learning rate scheduling, progress monitoring
- **CLI Interface**: Easy-to-use command-line interface for training models

## Quick Start

### Using Nix (Recommended)

```bash
# Enter development shell
nix develop

# Build the project
cargo build --release

# Train on example data
cargo run -- --data example_data.csv --target-column 2 --has-header \
  --architecture "10,5,1" --epochs 50 --learning-rate 0.01
```

### Using Cargo

```bash
# Install dependencies and build
cargo build --release

# Train on example data
./target/release/multilayer-perceptron \
  --data example_data.csv \
  --target-column 2 \
  --has-header \
  --architecture "10,5,1" \
  --epochs 50 \
  --learning-rate 0.01
```

## CLI Usage

### Basic Training

```bash
cargo run -- --data your_data.csv --target-column 0 --has-header
```

### Advanced Configuration

```bash
cargo run -- \
  --data dataset.csv \
  --target-column 3 \
  --has-header \
  --architecture "64,32,16,1" \
  --hidden-activation relu \
  --output-activation sigmoid \
  --loss binary_crossentropy \
  --optimizer adam \
  --learning-rate 0.001 \
  --epochs 200 \
  --batch-size 64 \
  --validation-split 0.2 \
  --early-stopping-patience 10 \
  --normalize \
  --verbose
```

### Command Line Options

- `--data`: Path to CSV file containing the dataset
- `--target-column`: Column index for target variable (0-based)
- `--has-header`: Whether CSV file has header row
- `--architecture`: Network architecture (comma-separated layer sizes)
- `--hidden-activation`: Activation function for hidden layers (relu, sigmoid, tanh)
- `--output-activation`: Activation function for output layer (relu, sigmoid, tanh, linear)
- `--loss`: Loss function (mse, binary_crossentropy)
- `--optimizer`: Optimizer (sgd, adam)
- `--learning-rate`: Learning rate for optimization
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for mini-batch gradient descent
- `--validation-split`: Fraction of data to use for validation
- `--no-shuffle`: Disable data shuffling between epochs
- `--early-stopping-patience`: Stop training if no improvement for N epochs
- `--normalize`: Normalize input features to zero mean and unit variance
- `--verbose`: Enable verbose logging

## Library Usage

You can also use this as a library in your Rust projects:

```rust
use multilayer_perceptron::{
    activation::{ReLU, Sigmoid},
    data::{Dataset, TrainingConfig},
    loss::MeanSquaredError,
    mlp::MLPBuilder,
    optimizer::Adam,
};
use std::sync::Arc;

// Load your data
let dataset = Dataset::from_csv("data.csv", 0, true)?;

// Create MLP
let mut mlp = MLPBuilder::new()
    .input_size(dataset.num_features())
    .layer(64, Arc::new(ReLU))
    .layer(32, Arc::new(ReLU))
    .layer(1, Arc::new(Sigmoid))
    .loss_function(Arc::new(MeanSquaredError))
    .optimizer(Box::new(Adam::new(0.001)))
    .build()?;

// Configure training
let config = TrainingConfig {
    epochs: 100,
    batch_size: 32,
    learning_rate: 0.001,
    validation_split: 0.2,
    ..Default::default()
};

// Train the model
let metrics = mlp.train(&dataset, &config)?;

// Make predictions
let prediction = mlp.predict(&input_vector);
```

## Supported Formats

### CSV Data Format

The program expects CSV data with:

- Numerical values only
- Optional header row
- One column designated as the target variable
- All other columns treated as features

Example:

```csv
feature1,feature2,feature3,target
0.1,0.2,0.3,0
0.4,0.5,0.6,1
...
```

### Network Architecture

Specify layer sizes as comma-separated values:

- `"10,1"`: Input → 10 neurons → 1 output
- `"64,32,16,1"`: Input → 64 → 32 → 16 → 1 output
- `"100,50,25,10,2"`: Input → 100 → 50 → 25 → 10 → 2 outputs

## Examples

### Binary Classification

```bash
cargo run -- \
  --data binary_classification.csv \
  --target-column 2 \
  --has-header \
  --architecture "10,5,1" \
  --output-activation sigmoid \
  --loss binary_crossentropy \
  --normalize
```

### Regression

```bash
cargo run -- \
  --data regression_data.csv \
  --target-column 0 \
  --architecture "20,10,1" \
  --output-activation linear \
  --loss mse \
  --optimizer adam \
  --learning-rate 0.001
```

### Multi-class Classification

```bash
cargo run -- \
  --data multiclass_data.csv \
  --target-column 4 \
  --architecture "50,25,10" \
  --output-activation softmax \
  --loss categorical_crossentropy
```

## Development

This project uses:

- **Nix**: For reproducible development environment
- **Just**: For task automation
- **Pre-commit**: For code quality checks
- **Rust**: With strict linting rules

### Development Commands

```bash
# Format and check code
just pre-commit-all

# Run with auto-recompilation
just watch --data example_data.csv --has-header --target-column 2

# Run tests
cargo test

# Build optimized release
cargo build --release
```

## Performance Tips

1. **Normalize your data** using `--normalize` for better convergence
2. **Use appropriate batch sizes** (32-128 for most datasets)
3. **Enable early stopping** to prevent overfitting
4. **Start with Adam optimizer** for most problems
5. **Use ReLU activations** for hidden layers in most cases

## Architecture Guidelines

- **Input layer**: Size = number of features in your data
- **Hidden layers**: Start with 2-3 layers, sizes decreasing by ~50% each layer
- **Output layer**:
  - 1 neuron for regression or binary classification
  - N neurons for N-class classification
- **Activation functions**:
  - Hidden: ReLU (general purpose), Tanh (when outputs need to be bounded)
  - Output: Linear (regression), Sigmoid (binary classification), Softmax (multi-class)

## License

MIT License - see LICENSE file for details.
