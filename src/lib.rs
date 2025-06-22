//! Multi-layer Perceptron Library
//!
//! This library provides a flexible implementation of multi-layer perceptrons
//! for training on preprocessed data.

pub mod activation;
pub mod data;
pub mod layer;
pub mod loss;
pub mod mlp;
pub mod optimizer;

pub use mlp::MultilayerPerceptron;
