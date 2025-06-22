//! Loss functions for neural network training

use ndarray::Array1;

/// Trait for loss functions
pub trait LossFunction: Send + Sync + std::fmt::Debug {
    /// Compute the loss given predictions and targets
    fn compute_loss(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> f64;

    /// Compute the gradient of the loss with respect to predictions
    fn compute_gradient(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> Array1<f64>;
}

/// Mean Squared Error loss function
#[derive(Debug, Clone, Copy)]
pub struct MeanSquaredError;

impl LossFunction for MeanSquaredError {
    fn compute_loss(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
        let diff = predictions - targets;
        let squared_diff = &diff * &diff;
        squared_diff.mean().unwrap_or(0.0)
    }

    #[allow(clippy::cast_precision_loss)]
    fn compute_gradient(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> Array1<f64> {
        let n = predictions.len() as f64;
        (predictions - targets) * (2.0 / n)
    }
}

/// Mean Absolute Error loss function
#[derive(Debug, Clone, Copy)]
pub struct MeanAbsoluteError;

impl LossFunction for MeanAbsoluteError {
    fn compute_loss(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
        let diff = predictions - targets;
        diff.mapv(f64::abs).mean().unwrap_or(0.0)
    }

    #[allow(clippy::cast_precision_loss)]
    fn compute_gradient(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> Array1<f64> {
        let diff = predictions - targets;
        let n = predictions.len() as f64;
        diff.mapv(f64::signum) / n
    }
}

/// Binary Cross-Entropy loss function
#[derive(Debug, Clone, Copy)]
pub struct BinaryCrossEntropy {
    /// Small epsilon value to prevent log(0)
    pub epsilon: f64,
}

impl Default for BinaryCrossEntropy {
    fn default() -> Self {
        Self { epsilon: 1e-15 }
    }
}

impl BinaryCrossEntropy {
    /// Create a new Binary Cross-Entropy loss with custom epsilon
    #[must_use]
    pub const fn new(epsilon: f64) -> Self {
        Self { epsilon }
    }
}

impl LossFunction for BinaryCrossEntropy {
    #[allow(clippy::cast_precision_loss)]
    fn compute_loss(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
        let clipped_predictions = predictions.mapv(|p| p.max(self.epsilon).min(1.0 - self.epsilon));
        let loss_terms = targets
            .iter()
            .zip(clipped_predictions.iter())
            .map(|(&t, &p)| -t.mul_add(p.ln(), (1.0 - t) * (1.0 - p).ln()))
            .collect::<Vec<_>>();

        loss_terms.iter().sum::<f64>() / loss_terms.len() as f64
    }

    #[allow(clippy::cast_precision_loss)]
    fn compute_gradient(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> Array1<f64> {
        let clipped_predictions = predictions.mapv(|p| p.max(self.epsilon).min(1.0 - self.epsilon));
        let n = predictions.len() as f64;

        let gradient = targets
            .iter()
            .zip(clipped_predictions.iter())
            .map(|(&t, &p)| (p - t) / (p * (1.0 - p)))
            .collect::<Vec<_>>();

        Array1::from_vec(gradient) / n
    }
}

/// Categorical Cross-Entropy loss function
#[derive(Debug, Clone, Copy)]
pub struct CategoricalCrossEntropy {
    /// Small epsilon value to prevent log(0)
    pub epsilon: f64,
}

impl Default for CategoricalCrossEntropy {
    fn default() -> Self {
        Self { epsilon: 1e-15 }
    }
}

impl CategoricalCrossEntropy {
    /// Create a new Categorical Cross-Entropy loss with custom epsilon
    #[must_use]
    pub const fn new(epsilon: f64) -> Self {
        Self { epsilon }
    }
}

impl LossFunction for CategoricalCrossEntropy {
    fn compute_loss(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
        let clipped_predictions = predictions.mapv(|p| p.max(self.epsilon));
        let loss_terms = targets
            .iter()
            .zip(clipped_predictions.iter())
            .map(|(&t, &p)| -t * p.ln())
            .collect::<Vec<_>>();

        loss_terms.iter().sum::<f64>()
    }

    fn compute_gradient(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> Array1<f64> {
        let clipped_predictions = predictions.mapv(|p| p.max(self.epsilon));

        let gradient = targets
            .iter()
            .zip(clipped_predictions.iter())
            .map(|(&t, &p)| -t / p)
            .collect::<Vec<_>>();

        Array1::from_vec(gradient)
    }
}

/// Huber loss function (robust to outliers)
#[derive(Debug, Clone, Copy)]
pub struct HuberLoss {
    /// Delta parameter for Huber loss
    pub delta: f64,
}

impl Default for HuberLoss {
    fn default() -> Self {
        Self { delta: 1.0 }
    }
}

impl HuberLoss {
    /// Create a new Huber loss with custom delta
    #[must_use]
    pub const fn new(delta: f64) -> Self {
        Self { delta }
    }
}

impl LossFunction for HuberLoss {
    fn compute_loss(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
        let diff = predictions - targets;
        let abs_diff = diff.mapv(f64::abs);

        let loss_terms = abs_diff
            .iter()
            .map(|&a| {
                if a <= self.delta {
                    0.5 * a * a
                } else {
                    self.delta * 0.5f64.mul_add(-self.delta, a)
                }
            })
            .collect::<Vec<_>>();

        #[allow(clippy::cast_precision_loss)]
        {
            loss_terms.iter().sum::<f64>() / loss_terms.len() as f64
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn compute_gradient(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> Array1<f64> {
        let diff = predictions - targets;
        let n = predictions.len() as f64;

        let gradient = diff
            .iter()
            .map(|&d| {
                if d.abs() <= self.delta {
                    d
                } else {
                    self.delta * d.signum()
                }
            })
            .collect::<Vec<_>>();

        Array1::from_vec(gradient) / n
    }
}
