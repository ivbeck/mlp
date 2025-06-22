use anyhow::Result;
use clap::Parser;
use multilayer_perceptron::{
    activation::{ReLU, Sigmoid, Tanh},
    data::{Dataset, TrainingConfig},
    loss::{BinaryCrossEntropy, MeanSquaredError},
    mlp::MLPBuilder,
    optimizer::{Adam, SGD},
};
use std::sync::Arc;
use tracing::{error, info};

#[derive(Parser, Debug)]
#[clap(author, version, about)]
#[allow(clippy::struct_excessive_bools)]
/// Multi-layer Perceptron trainer for preprocessed data
struct Args {
    /// Path to the CSV file containing the dataset
    #[arg(short = 'd', long)]
    data: String,

    /// Column index for the target variable (0-based)
    #[arg(short = 't', long, default_value = "0")]
    target_column: usize,

    /// Whether the CSV file has a header row
    #[arg(long)]
    has_header: bool,

    /// Network architecture (comma-separated layer sizes, e.g., "64,32,1")
    #[arg(short = 'a', long, default_value = "10,5,1")]
    architecture: String,

    /// Activation function for hidden layers (relu, sigmoid, tanh)
    #[arg(long, default_value = "relu")]
    hidden_activation: String,

    /// Activation function for output layer (relu, sigmoid, tanh, linear)
    #[arg(long, default_value = "sigmoid")]
    output_activation: String,

    /// Loss function (mse, `binary_crossentropy`)
    #[arg(short = 'l', long, default_value = "mse")]
    loss: String,

    /// Optimizer (sgd, adam)
    #[arg(short = 'o', long, default_value = "adam")]
    optimizer: String,

    /// Learning rate
    #[arg(long, default_value = "0.001")]
    learning_rate: f64,

    /// Number of training epochs
    #[arg(short = 'e', long, default_value = "100")]
    epochs: usize,

    /// Batch size for mini-batch gradient descent
    #[arg(short = 'b', long, default_value = "32")]
    batch_size: usize,

    /// Validation split ratio (0.0 to 1.0)
    #[arg(long, default_value = "0.2")]
    validation_split: f64,

    /// Whether to shuffle data between epochs
    #[arg(long)]
    no_shuffle: bool,

    /// Early stopping patience (epochs without improvement)
    #[arg(long)]
    early_stopping_patience: Option<usize>,

    /// Normalize input features
    #[arg(long)]
    normalize: bool,

    /// whether to be verbose
    #[arg(short = 'v')]
    verbose: bool,
}

fn parse_architecture(arch_str: &str) -> Result<Vec<usize>> {
    arch_str
        .split(',')
        .map(|s| s.trim().parse::<usize>().map_err(anyhow::Error::from))
        .collect()
}

fn create_activation(
    name: &str,
) -> Result<Arc<dyn multilayer_perceptron::activation::ActivationFunction>> {
    match name.to_lowercase().as_str() {
        "relu" => Ok(Arc::new(ReLU)),
        "sigmoid" => Ok(Arc::new(Sigmoid)),
        "tanh" => Ok(Arc::new(Tanh)),
        "linear" => Ok(Arc::new(multilayer_perceptron::activation::Linear)),
        _ => anyhow::bail!("Unknown activation function: {}", name),
    }
}

fn create_loss(name: &str) -> Result<Arc<dyn multilayer_perceptron::loss::LossFunction>> {
    match name.to_lowercase().as_str() {
        "mse" | "mean_squared_error" => Ok(Arc::new(MeanSquaredError)),
        "binary_crossentropy" | "bce" => Ok(Arc::new(BinaryCrossEntropy::default())),
        _ => anyhow::bail!("Unknown loss function: {}", name),
    }
}

fn create_optimizer(
    name: &str,
    learning_rate: f64,
) -> Result<Box<dyn multilayer_perceptron::optimizer::Optimizer>> {
    match name.to_lowercase().as_str() {
        "sgd" => Ok(Box::new(SGD::new(learning_rate))),
        "adam" => Ok(Box::new(Adam::new(learning_rate))),
        _ => anyhow::bail!("Unknown optimizer: {}", name),
    }
}

#[allow(clippy::unused_async)]
async fn run_training(args: Args) -> Result<()> {
    info!("Loading dataset from: {}", args.data);

    // Load dataset
    let mut dataset = Dataset::from_csv(&args.data, args.target_column, args.has_header)?;
    info!(
        "Loaded dataset with {} samples and {} features",
        dataset.num_samples(),
        dataset.num_features()
    );

    // Normalize if requested
    let (_mean, _std) = if args.normalize {
        info!("Normalizing features...");
        let (normalized_dataset, mean, std) = dataset.normalize();
        dataset = normalized_dataset;
        (Some(mean), Some(std))
    } else {
        (None, None)
    };

    // Parse architecture
    let layer_sizes = parse_architecture(&args.architecture)?;
    info!("Network architecture: {:?}", layer_sizes);

    // Set input size based on dataset
    let mut full_architecture = vec![dataset.num_features()];
    full_architecture.extend(layer_sizes);

    // Create activations
    let hidden_activation = create_activation(&args.hidden_activation)?;
    let output_activation = create_activation(&args.output_activation)?;

    let mut activations = Vec::new();
    for _ in 0..full_architecture.len() - 2 {
        activations.push(Arc::clone(&hidden_activation));
    }
    activations.push(output_activation);

    info!("Hidden activation: {}", args.hidden_activation);
    info!("Output activation: {}", args.output_activation);

    // Create loss function and optimizer
    let loss_function = create_loss(&args.loss)?;
    let optimizer = create_optimizer(&args.optimizer, args.learning_rate)?;

    info!("Loss function: {}", args.loss);
    info!("Optimizer: {} (lr={})", args.optimizer, args.learning_rate);

    // Build MLP
    let mut mlp = MLPBuilder::new().input_size(full_architecture[0]);

    for i in 1..full_architecture.len() {
        mlp = mlp.layer(full_architecture[i], Arc::clone(&activations[i - 1]));
    }

    let mut mlp = mlp
        .loss_function(loss_function)
        .optimizer(optimizer)
        .build()?;

    info!("Created MLP with {} parameters", mlp.num_parameters());

    // Training configuration
    let config = TrainingConfig {
        learning_rate: args.learning_rate,
        epochs: args.epochs,
        batch_size: args.batch_size,
        shuffle: !args.no_shuffle,
        validation_split: args.validation_split,
        early_stopping_patience: args.early_stopping_patience,
    };

    info!("Training configuration: {:?}", config);

    // Train the model
    info!("Starting training...");
    let metrics = mlp.train(&dataset, &config)?;

    // Print final results
    if let Some(final_train_loss) = metrics.latest_train_loss() {
        info!("Final training loss: {:.6}", final_train_loss);
    }

    if let Some(final_val_loss) = metrics.latest_val_loss() {
        info!("Final validation loss: {:.6}", final_val_loss);
    }

    info!("Training completed successfully!");

    // Example prediction on first sample
    if dataset.num_samples() > 0 {
        let sample_input = dataset.features.row(0).to_owned();
        let prediction = mlp.predict(&sample_input);
        let actual = dataset.labels[0];
        info!(
            "Example prediction: {:.6}, Actual: {:.6}",
            prediction[0], actual
        );
    }

    Ok(())
}

#[tokio::main]
async fn main() {
    // Initialize tracing
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("Failed to set tracing subscriber");

    let args = Args::parse();

    if args.verbose {
        info!("Verbose mode is enabled");
    }

    if let Err(e) = run_training(args).await {
        error!("Training failed: {}", e);
        std::process::exit(1);
    }
}
