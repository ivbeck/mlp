//! Multi-layer Perceptron implementation

use crate::{
    activation::ActivationFunction,
    data::{Dataset, TrainingConfig},
    layer::Layer,
    loss::LossFunction,
    optimizer::Optimizer,
};
use anyhow::{Context, Result};
use ndarray::Array1;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Training metrics for monitoring progress
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    /// Training loss per epoch
    pub train_losses: Vec<f64>,
    /// Validation loss per epoch (if validation data is provided)
    pub val_losses: Vec<f64>,
    /// Training accuracy per epoch (for classification tasks)
    pub train_accuracies: Vec<f64>,
    /// Validation accuracy per epoch (for classification tasks)
    pub val_accuracies: Vec<f64>,
}

impl TrainingMetrics {
    /// Create new empty training metrics
    #[must_use]
    pub const fn new() -> Self {
        Self {
            train_losses: Vec::new(),
            val_losses: Vec::new(),
            train_accuracies: Vec::new(),
            val_accuracies: Vec::new(),
        }
    }

    /// Get the latest training loss
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn latest_train_loss(&self) -> Option<f64> {
        self.train_losses.last().copied()
    }

    /// Get the latest validation loss
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn latest_val_loss(&self) -> Option<f64> {
        self.val_losses.last().copied()
    }

    /// Check if validation loss has improved in the last n epochs
    ///
    /// # Panics
    ///
    /// Panics if called when `val_losses` is empty
    #[must_use]
    pub fn has_val_loss_improved(&self, patience: usize) -> bool {
        if self.val_losses.len() < patience + 1 {
            return true; // Not enough data to determine
        }

        let recent_losses = &self.val_losses[self.val_losses.len() - patience..];
        let best_recent = recent_losses
            .iter()
            .fold(f64::INFINITY, |acc, &loss| acc.min(loss));

        let current_loss = self.val_losses.last().copied().unwrap_or(f64::INFINITY);
        current_loss <= best_recent
    }
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-layer Perceptron neural network
#[derive(Debug)]
pub struct MultilayerPerceptron {
    /// Layers of the network
    layers: Vec<Layer>,
    /// Loss function
    loss_function: Arc<dyn LossFunction>,
    /// Optimizer
    optimizer: Box<dyn Optimizer>,
    /// Last network output (stored for loss computation)
    last_output: Option<Array1<f64>>,
}

impl MultilayerPerceptron {
    /// Create a new MLP with the specified architecture
    ///
    /// # Errors
    ///
    /// Returns an error if the layer sizes are invalid (empty or contain zeros)
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(
        layer_sizes: &[usize],
        activations: Vec<Arc<dyn ActivationFunction>>,
        loss_function: Arc<dyn LossFunction>,
        optimizer: Box<dyn Optimizer>,
    ) -> Result<Self> {
        if layer_sizes.len() < 2 {
            anyhow::bail!("Network must have at least 2 layers (input and output)");
        }

        if layer_sizes.iter().any(|&size| size == 0) {
            anyhow::bail!("All layer sizes must be greater than 0");
        }

        if activations.len() != layer_sizes.len() - 1 {
            anyhow::bail!(
                "Number of activations ({}) must equal number of hidden + output layers ({})",
                activations.len(),
                layer_sizes.len() - 1
            );
        }

        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            let layer = Layer::new(
                layer_sizes[i],
                layer_sizes[i + 1],
                Arc::clone(&activations[i]),
            );
            layers.push(layer);
        }

        Ok(Self {
            layers,
            loss_function,
            optimizer,
            last_output: None,
        })
    }

    /// Forward pass through the network
    pub fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
        let mut current_input = input.clone();

        for layer in &mut self.layers {
            current_input = layer.forward(&current_input);
        }

        // Store the output for backward pass
        self.last_output = Some(current_input.clone());
        current_input
    }

    /// Backward pass through the network
    ///
    /// # Errors
    ///
    /// Returns an error if forward pass hasn't been called first
    fn backward(&mut self, target: &Array1<f64>) -> Result<f64> {
        let output = self
            .last_output
            .as_ref()
            .context("Forward pass must be called before backward pass")?;

        let mut current_gradient = self.loss_function.compute_gradient(output, target);
        let loss = self.loss_function.compute_loss(output, target);

        // Backward pass through layers (in reverse order)
        let mut gradients = Vec::new();
        #[allow(unused_variables)]
        for (layer_idx, layer) in self.layers.iter().rev().enumerate() {
            let (input_gradient, weight_gradient, bias_gradient) =
                layer.backward(&current_gradient);
            gradients.push((weight_gradient, bias_gradient));
            current_gradient = input_gradient;
        }

        // Reverse gradients to match layer order
        gradients.reverse();

        // Update parameters using optimizer
        for (layer_idx, (layer, (weight_grad, bias_grad))) in
            self.layers.iter_mut().zip(gradients.iter()).enumerate()
        {
            let (new_weights, new_biases) = self.optimizer.update(
                layer_idx,
                &layer.weights,
                &layer.biases,
                weight_grad,
                bias_grad,
            );
            layer.weights = new_weights;
            layer.biases = new_biases;
        }

        Ok(loss)
    }

    /// Train the network on the provided dataset
    ///
    /// # Errors
    ///
    /// Returns an error if the dataset is invalid or training fails
    #[allow(clippy::cast_precision_loss)]
    pub fn train(&mut self, dataset: &Dataset, config: &TrainingConfig) -> Result<TrainingMetrics> {
        info!("Starting training with {} epochs", config.epochs);

        let mut metrics = TrainingMetrics::new();
        let mut best_val_loss = f64::INFINITY;
        let mut epochs_without_improvement = 0;

        // Split dataset if validation is requested
        let (train_data, val_data) = if config.validation_split > 0.0 {
            let (train, val) = dataset.train_validation_split(config.validation_split);
            (train, Some(val))
        } else {
            (dataset.clone(), None)
        };

        for epoch in 0..config.epochs {
            debug!("Starting epoch {}/{}", epoch + 1, config.epochs);

            let epoch_loss = self.train_epoch(&train_data, config.batch_size, config.shuffle)?;
            metrics.train_losses.push(epoch_loss);

            // Validate if validation data is available
            if let Some(val_data) = &val_data {
                let val_loss = self.evaluate(val_data)?;
                metrics.val_losses.push(val_loss);

                // Early stopping check
                if let Some(patience) = config.early_stopping_patience {
                    if val_loss < best_val_loss {
                        best_val_loss = val_loss;
                        epochs_without_improvement = 0;
                    } else {
                        epochs_without_improvement += 1;
                        if epochs_without_improvement >= patience {
                            warn!(
                                "Early stopping at epoch {} (no improvement for {} epochs)",
                                epoch + 1,
                                patience
                            );
                            break;
                        }
                    }
                }

                info!(
                    "Epoch {}/{}: train_loss={:.6}, val_loss={:.6}",
                    epoch + 1,
                    config.epochs,
                    epoch_loss,
                    val_loss
                );
            } else {
                info!(
                    "Epoch {}/{}: train_loss={:.6}",
                    epoch + 1,
                    config.epochs,
                    epoch_loss
                );
            }
        }

        info!("Training completed");
        Ok(metrics)
    }

    /// Run one epoch of mini-batch training over `data` and return the mean batch loss
    ///
    /// This is what [`Self::train`] does per epoch, exposed so callers can drive
    /// training step by step (pause, resume, stream progress).
    ///
    /// # Errors
    ///
    /// Returns an error if a backward pass fails
    #[allow(clippy::cast_precision_loss)]
    pub fn train_epoch(&mut self, data: &Dataset, batch_size: usize, shuffle: bool) -> Result<f64> {
        let epoch_data = if shuffle {
            data.shuffle()
        } else {
            data.clone()
        };

        let batches = epoch_data.create_batches(batch_size.max(1));
        if batches.is_empty() {
            anyhow::bail!("No training samples");
        }
        let mut epoch_loss = 0.0;

        for (batch_idx, batch) in batches.iter().enumerate() {
            let mut batch_loss = 0.0;

            for sample_idx in 0..batch.num_samples() {
                let input = batch.features.row(sample_idx).to_owned();
                let target = Array1::from_elem(1, batch.labels[sample_idx]);

                // Forward and backward pass
                self.forward(&input);
                let loss = self.backward(&target)?;
                batch_loss += loss;
            }

            batch_loss /= batch.num_samples() as f64;
            epoch_loss += batch_loss;

            debug!(
                "Batch {}/{} loss: {:.6}",
                batch_idx + 1,
                batches.len(),
                batch_loss
            );
        }

        Ok(epoch_loss / batches.len() as f64)
    }

    /// Forward pass that returns the output of every layer, last one is the network output
    pub fn forward_trace(&mut self, input: &Array1<f64>) -> Vec<Array1<f64>> {
        let mut outputs = Vec::with_capacity(self.layers.len());
        let mut current_input = input.clone();

        for layer in &mut self.layers {
            current_input = layer.forward(&current_input);
            outputs.push(current_input.clone());
        }

        self.last_output = Some(current_input);
        outputs
    }

    /// The layers of the network, input side first
    #[must_use]
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    /// Evaluate the network on a dataset
    ///
    /// # Errors
    ///
    /// Returns an error if evaluation fails
    #[allow(clippy::cast_precision_loss)]
    pub fn evaluate(&mut self, dataset: &Dataset) -> Result<f64> {
        let mut total_loss = 0.0;
        let num_samples = dataset.num_samples();

        for sample_idx in 0..num_samples {
            let input = dataset.features.row(sample_idx).to_owned();
            let target = Array1::from_elem(1, dataset.labels[sample_idx]);

            let output = self.forward(&input);
            let loss = self.loss_function.compute_loss(&output, &target);
            total_loss += loss;
        }

        Ok(total_loss / num_samples as f64)
    }

    /// Make predictions on new data
    pub fn predict(&mut self, inputs: &Array1<f64>) -> Array1<f64> {
        self.forward(inputs)
    }

    /// Get the number of parameters in the network
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        self.layers.iter().map(Layer::num_parameters).sum()
    }

    /// Reset the network (clear cached values)
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
        self.optimizer.reset();
        self.last_output = None;
    }

    /// Get network architecture as a vector of layer sizes
    #[must_use]
    pub fn architecture(&self) -> Vec<usize> {
        let mut sizes = vec![self.layers[0].input_size()];
        for layer in &self.layers {
            sizes.push(layer.output_size());
        }
        sizes
    }
}

/// Builder for creating MLPs with a fluent API
#[derive(Debug)]
pub struct MLPBuilder {
    layer_sizes: Vec<usize>,
    activations: Vec<Arc<dyn ActivationFunction>>,
    loss_function: Option<Arc<dyn LossFunction>>,
    optimizer: Option<Box<dyn Optimizer>>,
}

impl MLPBuilder {
    /// Create a new MLP builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            layer_sizes: Vec::new(),
            activations: Vec::new(),
            loss_function: None,
            optimizer: None,
        }
    }

    /// Add a layer to the network
    #[must_use]
    pub fn layer(mut self, size: usize, activation: Arc<dyn ActivationFunction>) -> Self {
        if self.layer_sizes.is_empty() {
            // This is the input layer
            self.layer_sizes.push(size);
        }
        self.layer_sizes.push(size);
        self.activations.push(activation);
        self
    }

    /// Set the input size (optional - can be inferred from first layer)
    #[must_use]
    pub fn input_size(mut self, size: usize) -> Self {
        if self.layer_sizes.is_empty() {
            self.layer_sizes.push(size);
        } else {
            self.layer_sizes[0] = size;
        }
        self
    }

    /// Set the loss function
    #[must_use]
    pub fn loss_function(mut self, loss: Arc<dyn LossFunction>) -> Self {
        self.loss_function = Some(loss);
        self
    }

    /// Set the optimizer
    #[must_use]
    pub fn optimizer(mut self, optimizer: Box<dyn Optimizer>) -> Self {
        self.optimizer = Some(optimizer);
        self
    }

    /// Build the MLP
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid
    pub fn build(self) -> Result<MultilayerPerceptron> {
        let loss_function = self
            .loss_function
            .context("Loss function must be specified")?;
        let optimizer = self.optimizer.context("Optimizer must be specified")?;

        if self.layer_sizes.len() < 2 {
            anyhow::bail!("At least input and output layer sizes must be specified");
        }

        MultilayerPerceptron::new(
            &self.layer_sizes,
            self.activations,
            loss_function,
            optimizer,
        )
    }
}

impl Default for MLPBuilder {
    fn default() -> Self {
        Self::new()
    }
}
