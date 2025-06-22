//! Optimization algorithms for neural network training

use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Trait for optimization algorithms
pub trait Optimizer: Send + Sync + std::fmt::Debug {
    /// Update parameters using gradients
    fn update(
        &mut self,
        layer_id: usize,
        weights: &Array2<f64>,
        biases: &Array1<f64>,
        weight_gradients: &Array2<f64>,
        bias_gradients: &Array1<f64>,
    ) -> (Array2<f64>, Array1<f64>);

    /// Reset optimizer state (useful when starting new training)
    fn reset(&mut self);
}

/// Stochastic Gradient Descent optimizer
#[derive(Debug, Clone, Copy)]
pub struct SGD {
    /// Learning rate
    pub learning_rate: f64,
}

impl SGD {
    /// Create a new SGD optimizer
    #[must_use]
    pub const fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for SGD {
    fn update(
        &mut self,
        _layer_id: usize,
        weights: &Array2<f64>,
        biases: &Array1<f64>,
        weight_gradients: &Array2<f64>,
        bias_gradients: &Array1<f64>,
    ) -> (Array2<f64>, Array1<f64>) {
        let new_weights = weights - &(weight_gradients * self.learning_rate);
        let new_biases = biases - &(bias_gradients * self.learning_rate);
        (new_weights, new_biases)
    }

    fn reset(&mut self) {
        // SGD has no internal state to reset
    }
}

/// SGD with Momentum optimizer
#[derive(Debug, Clone)]
pub struct SGDMomentum {
    /// Learning rate
    pub learning_rate: f64,
    /// Momentum coefficient
    pub momentum: f64,
    /// Velocity for weights (per layer)
    weight_velocities: HashMap<usize, Array2<f64>>,
    /// Velocity for biases (per layer)
    bias_velocities: HashMap<usize, Array1<f64>>,
}

impl SGDMomentum {
    /// Create a new SGD with Momentum optimizer
    #[must_use]
    pub fn new(learning_rate: f64, momentum: f64) -> Self {
        Self {
            learning_rate,
            momentum,
            weight_velocities: HashMap::new(),
            bias_velocities: HashMap::new(),
        }
    }
}

impl Optimizer for SGDMomentum {
    fn update(
        &mut self,
        layer_id: usize,
        weights: &Array2<f64>,
        biases: &Array1<f64>,
        weight_gradients: &Array2<f64>,
        bias_gradients: &Array1<f64>,
    ) -> (Array2<f64>, Array1<f64>) {
        // Initialize velocities if not present
        let weight_velocity = self
            .weight_velocities
            .entry(layer_id)
            .or_insert_with(|| Array2::zeros(weights.raw_dim()));
        let bias_velocity = self
            .bias_velocities
            .entry(layer_id)
            .or_insert_with(|| Array1::zeros(biases.raw_dim()));

        // Update velocities
        *weight_velocity =
            &*weight_velocity * self.momentum + weight_gradients * self.learning_rate;
        *bias_velocity = &*bias_velocity * self.momentum + bias_gradients * self.learning_rate;

        // Update parameters
        let new_weights = weights - &*weight_velocity;
        let new_biases = biases - &*bias_velocity;

        (new_weights, new_biases)
    }

    fn reset(&mut self) {
        self.weight_velocities.clear();
        self.bias_velocities.clear();
    }
}

/// Adam optimizer
#[derive(Debug, Clone)]
pub struct Adam {
    /// Learning rate
    pub learning_rate: f64,
    /// Exponential decay rate for first moment estimates
    pub beta1: f64,
    /// Exponential decay rate for second moment estimates
    pub beta2: f64,
    /// Small constant for numerical stability
    pub epsilon: f64,
    /// First moment estimates for weights (per layer)
    weight_m: HashMap<usize, Array2<f64>>,
    /// Second moment estimates for weights (per layer)
    weight_v: HashMap<usize, Array2<f64>>,
    /// First moment estimates for biases (per layer)
    bias_m: HashMap<usize, Array1<f64>>,
    /// Second moment estimates for biases (per layer)
    bias_v: HashMap<usize, Array1<f64>>,
    /// Time step counter
    t: usize,
}

impl Adam {
    /// Create a new Adam optimizer
    #[must_use]
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_m: HashMap::new(),
            weight_v: HashMap::new(),
            bias_m: HashMap::new(),
            bias_v: HashMap::new(),
            t: 0,
        }
    }

    /// Create a new Adam optimizer with custom parameters
    #[must_use]
    pub fn new_with_params(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_m: HashMap::new(),
            weight_v: HashMap::new(),
            bias_m: HashMap::new(),
            bias_v: HashMap::new(),
            t: 0,
        }
    }
}

impl Optimizer for Adam {
    #[allow(
        clippy::similar_names,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap
    )]
    fn update(
        &mut self,
        layer_id: usize,
        weights: &Array2<f64>,
        biases: &Array1<f64>,
        weight_gradients: &Array2<f64>,
        bias_gradients: &Array1<f64>,
    ) -> (Array2<f64>, Array1<f64>) {
        self.t += 1;

        // Initialize moment estimates if not present
        let weight_m = self
            .weight_m
            .entry(layer_id)
            .or_insert_with(|| Array2::zeros(weights.raw_dim()));
        let weight_v = self
            .weight_v
            .entry(layer_id)
            .or_insert_with(|| Array2::zeros(weights.raw_dim()));
        let bias_m = self
            .bias_m
            .entry(layer_id)
            .or_insert_with(|| Array1::zeros(biases.raw_dim()));
        let bias_v = self
            .bias_v
            .entry(layer_id)
            .or_insert_with(|| Array1::zeros(biases.raw_dim()));

        // Update first moment estimates
        *weight_m = &*weight_m * self.beta1 + weight_gradients * (1.0 - self.beta1);
        *bias_m = &*bias_m * self.beta1 + bias_gradients * (1.0 - self.beta1);

        // Update second moment estimates
        *weight_v =
            &*weight_v * self.beta2 + &(weight_gradients * weight_gradients) * (1.0 - self.beta2);
        *bias_v = &*bias_v * self.beta2 + &(bias_gradients * bias_gradients) * (1.0 - self.beta2);

        // Bias correction
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        let weight_m_corrected = &*weight_m / bias_correction1;
        let weight_v_corrected = &*weight_v / bias_correction2;
        let bias_m_corrected = &*bias_m / bias_correction1;
        let bias_v_corrected = &*bias_v / bias_correction2;

        // Update parameters
        let weight_update =
            &weight_m_corrected / &(weight_v_corrected.mapv(f64::sqrt) + self.epsilon);
        let bias_update = &bias_m_corrected / &(bias_v_corrected.mapv(f64::sqrt) + self.epsilon);

        let new_weights = weights - &(weight_update * self.learning_rate);
        let new_biases = biases - &(bias_update * self.learning_rate);

        (new_weights, new_biases)
    }

    fn reset(&mut self) {
        self.weight_m.clear();
        self.weight_v.clear();
        self.bias_m.clear();
        self.bias_v.clear();
        self.t = 0;
    }
}

/// `RMSprop` optimizer
#[derive(Debug, Clone)]
pub struct RMSprop {
    /// Learning rate
    pub learning_rate: f64,
    /// Exponential decay rate
    pub rho: f64,
    /// Small constant for numerical stability
    pub epsilon: f64,
    /// Accumulator for weights (per layer)
    weight_accumulator: HashMap<usize, Array2<f64>>,
    /// Accumulator for biases (per layer)
    bias_accumulator: HashMap<usize, Array1<f64>>,
}

impl RMSprop {
    /// Create a new `RMSprop` optimizer
    #[must_use]
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            rho: 0.9,
            epsilon: 1e-8,
            weight_accumulator: HashMap::new(),
            bias_accumulator: HashMap::new(),
        }
    }

    /// Create a new `RMSprop` optimizer with custom parameters
    #[must_use]
    pub fn new_with_params(learning_rate: f64, rho: f64, epsilon: f64) -> Self {
        Self {
            learning_rate,
            rho,
            epsilon,
            weight_accumulator: HashMap::new(),
            bias_accumulator: HashMap::new(),
        }
    }
}

impl Optimizer for RMSprop {
    fn update(
        &mut self,
        layer_id: usize,
        weights: &Array2<f64>,
        biases: &Array1<f64>,
        weight_gradients: &Array2<f64>,
        bias_gradients: &Array1<f64>,
    ) -> (Array2<f64>, Array1<f64>) {
        // Initialize accumulators if not present
        let weight_acc = self
            .weight_accumulator
            .entry(layer_id)
            .or_insert_with(|| Array2::zeros(weights.raw_dim()));
        let bias_acc = self
            .bias_accumulator
            .entry(layer_id)
            .or_insert_with(|| Array1::zeros(biases.raw_dim()));

        // Update accumulators
        *weight_acc =
            &*weight_acc * self.rho + &(weight_gradients * weight_gradients) * (1.0 - self.rho);
        *bias_acc = &*bias_acc * self.rho + &(bias_gradients * bias_gradients) * (1.0 - self.rho);

        // Update parameters
        let weight_update = weight_gradients / &(weight_acc.mapv(f64::sqrt) + self.epsilon);
        let bias_update = bias_gradients / &(bias_acc.mapv(f64::sqrt) + self.epsilon);

        let new_weights = weights - &(weight_update * self.learning_rate);
        let new_biases = biases - &(bias_update * self.learning_rate);

        (new_weights, new_biases)
    }

    fn reset(&mut self) {
        self.weight_accumulator.clear();
        self.bias_accumulator.clear();
    }
}
