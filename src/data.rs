//! Data handling utilities for the MLP

use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Represents a dataset with features and labels
#[derive(Debug, Clone)]
pub struct Dataset {
    /// Feature matrix (samples x features)
    pub features: Array2<f64>,
    /// Label vector for regression or class indices for classification
    pub labels: Array1<f64>,
}

/// Training configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate for gradient descent
    pub learning_rate: f64,
    /// Number of training epochs
    pub epochs: usize,
    /// Batch size for mini-batch gradient descent
    pub batch_size: usize,
    /// Whether to shuffle data between epochs
    pub shuffle: bool,
    /// Validation split ratio (0.0 to 1.0)
    pub validation_split: f64,
    /// Early stopping patience (stop if no improvement for this many epochs)
    pub early_stopping_patience: Option<usize>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            epochs: 100,
            batch_size: 32,
            shuffle: true,
            validation_split: 0.2,
            early_stopping_patience: Some(10),
        }
    }
}

impl Dataset {
    /// Create a new dataset from features and labels
    #[must_use]
    pub const fn new(features: Array2<f64>, labels: Array1<f64>) -> Self {
        Self { features, labels }
    }

    /// Load dataset from CSV file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed
    pub fn from_csv<P: AsRef<Path>>(
        path: P,
        target_column: usize,
        has_header: bool,
    ) -> Result<Self> {
        let file = std::fs::File::open(path.as_ref()).context("Failed to open CSV file")?;
        Self::from_csv_reader(file, target_column, has_header)
    }

    /// Load dataset from anything that reads CSV text (an upload, a byte slice)
    ///
    /// # Errors
    ///
    /// Returns an error if the data cannot be read or parsed
    pub fn from_csv_reader<R: std::io::Read>(
        source: R,
        target_column: usize,
        has_header: bool,
    ) -> Result<Self> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(has_header)
            .from_reader(source);

        let mut all_data: Vec<Vec<f64>> = Vec::new();

        for (row_index, result) in reader.records().enumerate() {
            let record = result.context(format!("Failed to read row {row_index}"))?;

            let mut row_data: Vec<f64> = Vec::new();
            for (col_index, field) in record.iter().enumerate() {
                let value: f64 = field.parse().context(format!(
                    "Failed to parse value '{field}' at row {row_index}, column {col_index}"
                ))?;
                row_data.push(value);
            }
            all_data.push(row_data);
        }

        if all_data.is_empty() {
            anyhow::bail!("No data found in CSV file");
        }

        let num_cols = all_data[0].len();
        if target_column >= num_cols {
            anyhow::bail!(
                "Target column index {target_column} is out of range (max: {}))",
                num_cols - 1
            );
        }

        // Separate features and labels
        let mut features_data = Vec::new();
        let mut labels_data = Vec::new();

        for row in all_data {
            let mut features_row = Vec::new();
            for (col_index, value) in row.into_iter().enumerate() {
                if col_index == target_column {
                    labels_data.push(value);
                } else {
                    features_row.push(value);
                }
            }
            features_data.push(features_row);
        }

        let num_samples = features_data.len();
        let num_features = features_data[0].len();

        // Convert to ndarray
        let features = Array2::from_shape_vec(
            (num_samples, num_features),
            features_data.into_iter().flatten().collect(),
        )
        .context("Failed to create features array")?;

        let labels = Array1::from_vec(labels_data);

        Ok(Self::new(features, labels))
    }

    /// Get the number of samples in the dataset
    #[must_use]
    pub fn num_samples(&self) -> usize {
        self.features.nrows()
    }

    /// Get the number of features in the dataset
    #[must_use]
    pub fn num_features(&self) -> usize {
        self.features.ncols()
    }

    /// Split the dataset into training and validation sets
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    pub fn train_validation_split(&self, validation_ratio: f64) -> (Self, Self) {
        let num_samples = self.num_samples();
        let num_val = (num_samples as f64 * validation_ratio).round() as usize;
        let num_train = num_samples - num_val;

        let train_features = self.features.slice(s![..num_train, ..]).to_owned();
        let train_labels = self.labels.slice(s![..num_train]).to_owned();

        let val_features = self.features.slice(s![num_train.., ..]).to_owned();
        let val_labels = self.labels.slice(s![num_train..]).to_owned();

        (
            Self::new(train_features, train_labels),
            Self::new(val_features, val_labels),
        )
    }

    /// Normalize features to have zero mean and unit variance
    ///
    /// # Panics
    ///
    /// If the dataset has no samples or features
    #[must_use]
    pub fn normalize(&self) -> (Self, Array1<f64>, Array1<f64>) {
        let mean = self.features.mean_axis(ndarray::Axis(0)).unwrap();
        let std = self.features.std_axis(ndarray::Axis(0), 0.0);

        let normalized_features = (&self.features - &mean) / &std;

        (
            Self::new(normalized_features, self.labels.clone()),
            mean,
            std,
        )
    }

    /// Apply normalization using provided mean and standard deviation
    #[must_use]
    pub fn apply_normalization(&self, mean: &Array1<f64>, std: &Array1<f64>) -> Self {
        let normalized_features = (&self.features - mean) / std;
        Self::new(normalized_features, self.labels.clone())
    }

    /// Shuffle the dataset
    #[must_use]
    pub fn shuffle(&self) -> Self {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        let mut indices: Vec<usize> = (0..self.num_samples()).collect();
        indices.shuffle(&mut thread_rng());

        let mut shuffled_features = Array2::zeros((self.num_samples(), self.num_features()));
        let mut shuffled_labels = Array1::zeros(self.num_samples());

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            shuffled_features
                .row_mut(new_idx)
                .assign(&self.features.row(old_idx));
            shuffled_labels[new_idx] = self.labels[old_idx];
        }

        Self::new(shuffled_features, shuffled_labels)
    }

    /// Create mini-batches from the dataset
    #[must_use]
    pub fn create_batches(&self, batch_size: usize) -> Vec<Self> {
        let num_samples = self.num_samples();
        let mut batches = Vec::new();

        for start_idx in (0..num_samples).step_by(batch_size) {
            let end_idx = (start_idx + batch_size).min(num_samples);

            let batch_features = self.features.slice(s![start_idx..end_idx, ..]).to_owned();
            let batch_labels = self.labels.slice(s![start_idx..end_idx]).to_owned();

            batches.push(Self::new(batch_features, batch_labels));
        }

        batches
    }
}

// Import slice syntax
use ndarray::s;
