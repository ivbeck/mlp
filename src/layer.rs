//! Neural network layer implementation

use crate::activation::ActivationFunction;
use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use std::sync::Arc;

/// A single layer in the neural network
#[derive(Debug)]
pub struct Layer {
    /// Weight matrix (`input_size` x `output_size`)
    pub weights: Array2<f64>,
    /// Bias vector (`output_size`)
    pub biases: Array1<f64>,
    /// Activation function for this layer
    pub activation: Arc<dyn ActivationFunction>,
    /// Last input to this layer (stored for backpropagation)
    last_input: Option<Array1<f64>>,
    /// Last weighted sum before activation (stored for backpropagation)
    last_weighted_sum: Option<Array1<f64>>,
}

impl Layer {
    /// Create a new layer with random weights and zero biases
    ///
    /// # Panics
    ///
    /// Panics if `input_size` o`output_size`ze is zero
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation: Arc<dyn ActivationFunction>,
    ) -> Self {
        assert!(input_size > 0, "Input size must be greater than 0");
        assert!(output_size > 0, "Output size must be greater than 0");

        let mut rng = rand::thread_rng();

        // Xavier/Glorot initialization
        let limit = (6.0 / (input_size + output_size) as f64).sqrt();
        let weights =
            Array2::from_shape_fn((input_size, output_size), |_| rng.gen_range(-limit..limit));

        let biases = Array1::zeros(output_size);

        Self {
            weights,
            biases,
            activation,
            last_input: None,
            last_weighted_sum: None,
        }
    }

    /// Forward pass through the layer
    pub fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
        // Store input for backpropagation
        self.last_input = Some(input.clone());

        // Compute weighted sum: input * weights + biases
        let weighted_sum = input.dot(&self.weights) + &self.biases;

        // Store weighted sum for backpropagation
        self.last_weighted_sum = Some(weighted_sum.clone());

        // Apply activation function
        self.activation.activate(&weighted_sum)
    }

    /// Backward pass through the layer
    /// Returns the gradient with respect to the input
    ///
    /// # Panics
    ///
    /// Panics if `forward` has not been called before this method
    #[must_use]
    pub fn backward(
        &self,
        output_gradient: &Array1<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<f64>) {
        let last_input = self
            .last_input
            .as_ref()
            .expect("Forward pass must be called before backward pass");
        let last_weighted_sum = self
            .last_weighted_sum
            .as_ref()
            .expect("Forward pass must be called before backward pass");

        // Compute activation derivative
        let activation_derivative = self.activation.derivative(last_weighted_sum);

        // Compute gradient with respect to weighted sum
        let weighted_sum_gradient = output_gradient * &activation_derivative;

        // Gradient with respect to biases is just the weighted sum gradient
        let bias_gradient = weighted_sum_gradient.clone();

        // Compute gradient with respect to input
        let input_gradient = self.weights.dot(&weighted_sum_gradient);

        // Compute gradient with respect to weights
        let weight_gradient = last_input
            .clone()
            .insert_axis(Axis(1))
            .dot(&weighted_sum_gradient.insert_axis(Axis(0)));

        (input_gradient, weight_gradient, bias_gradient)
    }

    /// Update weights and biases using gradients
    pub fn update_parameters(
        &mut self,
        weight_gradient: &Array2<f64>,
        bias_gradient: &Array1<f64>,
        learning_rate: f64,
    ) {
        self.weights = &self.weights - &(weight_gradient * learning_rate);
        self.biases = &self.biases - &(bias_gradient * learning_rate);
    }

    /// Get the number of input neurons
    #[must_use]
    pub fn input_size(&self) -> usize {
        self.weights.nrows()
    }

    /// Get the number of output neurons
    #[must_use]
    pub fn output_size(&self) -> usize {
        self.weights.ncols()
    }

    /// Get the total number of parameters (weights + biases)
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        self.weights.len() + self.biases.len()
    }

    /// Reset the stored activations (useful for new forward pass)
    pub fn reset(&mut self) {
        self.last_input = None;
        self.last_weighted_sum = None;
    }
}

/// Builder for creating layers with specific configurations
#[derive(Debug)]
pub struct LayerBuilder {
    input_size: usize,
    output_size: usize,
    activation: Option<Arc<dyn ActivationFunction>>,
}

impl LayerBuilder {
    /// Create a new layer builder
    #[must_use]
    pub const fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            input_size,
            output_size,
            activation: None,
        }
    }

    /// Set the activation function
    #[must_use]
    pub fn activation(mut self, activation: Arc<dyn ActivationFunction>) -> Self {
        self.activation = Some(activation);
        self
    }

    /// Build the layer
    ///
    /// # Panics
    ///
    /// Panics if no activation function was set
    #[must_use]
    pub fn build(self) -> Layer {
        let activation = self.activation.expect("Activation function must be set");

        Layer::new(self.input_size, self.output_size, activation)
    }
}
