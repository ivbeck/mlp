//! Activation functions for neural networks

use ndarray::Array1;

/// Trait for activation functions
pub trait ActivationFunction: Send + Sync + std::fmt::Debug {
    /// Apply the activation function to the input
    fn activate(&self, x: &Array1<f64>) -> Array1<f64>;

    /// Compute the derivative of the activation function
    fn derivative(&self, x: &Array1<f64>) -> Array1<f64>;
}

/// Sigmoid activation function
#[derive(Debug, Clone, Copy)]
pub struct Sigmoid;

impl ActivationFunction for Sigmoid {
    fn activate(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }

    fn derivative(&self, x: &Array1<f64>) -> Array1<f64> {
        let sigmoid = self.activate(x);
        &sigmoid * &sigmoid.mapv(|v| 1.0 - v)
    }
}

/// `ReLU` activation function
#[derive(Debug, Clone, Copy)]
pub struct ReLU;

impl ActivationFunction for ReLU {
    fn activate(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| v.max(0.0))
    }

    fn derivative(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
    }
}

/// Tanh activation function
#[derive(Debug, Clone, Copy)]
pub struct Tanh;

impl ActivationFunction for Tanh {
    fn activate(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(f64::tanh)
    }

    fn derivative(&self, x: &Array1<f64>) -> Array1<f64> {
        let tanh = self.activate(x);
        tanh.mapv(|v| v.mul_add(-v, 1.0))
    }
}

/// Linear activation function (identity)
#[derive(Debug, Clone, Copy)]
pub struct Linear;

impl ActivationFunction for Linear {
    fn activate(&self, x: &Array1<f64>) -> Array1<f64> {
        x.clone()
    }

    fn derivative(&self, x: &Array1<f64>) -> Array1<f64> {
        Array1::ones(x.len())
    }
}

/// Leaky `ReLU` activation function
#[derive(Debug, Clone, Copy)]
pub struct LeakyReLU {
    /// The slope for negative values
    pub alpha: f64,
}

impl LeakyReLU {
    /// Create a new `LeakyReLU` with the given alpha parameter
    #[must_use]
    pub const fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self::new(0.01)
    }
}

impl ActivationFunction for LeakyReLU {
    fn activate(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| if v > 0.0 { v } else { self.alpha * v })
    }

    fn derivative(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| if v > 0.0 { 1.0 } else { self.alpha })
    }
}

/// Softmax activation function (typically used for output layer)
#[derive(Debug, Clone, Copy)]
pub struct Softmax;

impl ActivationFunction for Softmax {
    fn activate(&self, x: &Array1<f64>) -> Array1<f64> {
        let max_val = x.fold(f64::NEG_INFINITY, |acc, &val| acc.max(val));
        let shifted = x.mapv(|v| v - max_val);
        let exp_values = shifted.mapv(f64::exp);
        let sum = exp_values.sum();
        exp_values.mapv(|v| v / sum)
    }

    fn derivative(&self, x: &Array1<f64>) -> Array1<f64> {
        let softmax = self.activate(x);
        // For softmax, the derivative is more complex and typically handled differently
        // in the context of cross-entropy loss. This is a simplified version.
        &softmax * &softmax.mapv(|v| 1.0 - v)
    }
}
