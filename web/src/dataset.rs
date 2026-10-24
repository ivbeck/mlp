//! Uploaded CSV files: kept as raw bytes, so every run parses them with the crate's own
//! `Dataset::from_csv_reader`, plus a summary the ui draws from

use anyhow::{Context, Result};
use multilayer_perceptron::data::Dataset;
use serde::Serialize;

/// Rows sent to the browser for scatter plots, the rest stays server side
const SAMPLE_ROWS: usize = 600;
/// More distinct values than this and a column counts as continous
const MAX_DISTINCT: usize = 12;

#[derive(Debug, Clone)]
pub struct StoredDataset {
    pub info: DatasetInfo,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DatasetInfo {
    pub id: String,
    pub name: String,
    pub has_header: bool,
    pub rows: usize,
    pub columns: Vec<ColumnInfo>,
    /// First `SAMPLE_ROWS` rows, all columns, in file order
    pub sample: Vec<Vec<f64>>,
    pub size_bytes: usize,
    /// True for the bundled example file
    pub is_sample: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ColumnInfo {
    pub index: usize,
    pub name: String,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    /// Population std, the same one `Dataset::normalize` divides by
    pub std: f64,
    pub constant: bool,
    /// Sorted distinct values when there are few of them, else empty
    pub distinct: Vec<f64>,
    /// Exactly the values 0 and 1 in this column
    pub binary: bool,
}

/// A header row is assumed when any field of the first record is not a number
fn sniff_header(bytes: &[u8]) -> bool {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(bytes);
    reader.records().next().is_some_and(|first| {
        first.is_ok_and(|rec| rec.iter().any(|f| f.trim().parse::<f64>().is_err()))
    })
}

/// Parse an upload. `has_header` of `None` means guess.
///
/// # Errors
///
/// The crate's own parse error when the file is not all numbers, or an empty file
#[allow(clippy::cast_precision_loss)]
pub fn parse(
    id: String,
    name: String,
    bytes: Vec<u8>,
    has_header: Option<bool>,
    is_sample: bool,
) -> Result<StoredDataset> {
    let has_header = has_header.unwrap_or_else(|| sniff_header(&bytes));

    // Same parser the runs use, so a file that loads here trains there
    Dataset::from_csv_reader(bytes.as_slice(), 0, has_header)?;

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(has_header)
        .from_reader(bytes.as_slice());
    let names: Vec<String> = if has_header {
        reader
            .headers()
            .context("Failed to read header row")?
            .iter()
            .map(|h| h.trim().to_string())
            .collect()
    } else {
        Vec::new()
    };

    let mut rows: Vec<Vec<f64>> = Vec::new();
    for record in reader.records() {
        let record = record.context("Failed to read row")?;
        let row = record
            .iter()
            .map(|f| f.trim().parse::<f64>().unwrap_or(f64::NAN))
            .collect();
        rows.push(row);
    }
    let width = rows.first().map_or(0, Vec::len);
    if rows.iter().any(|r| r.len() != width) {
        anyhow::bail!("Rows have different numbers of columns");
    }
    if width < 2 {
        anyhow::bail!("Need at least one feature column and one target column");
    }

    let n = rows.len() as f64;
    let columns = (0..width)
        .map(|c| {
            let values: Vec<f64> = rows.iter().map(|r| r[c]).collect();
            let min = values.iter().copied().fold(f64::INFINITY, f64::min);
            let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mean = values.iter().sum::<f64>() / n;
            let std = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
            let mut distinct: Vec<f64> = Vec::new();
            for &v in &values {
                if !distinct.contains(&v) {
                    distinct.push(v);
                    if distinct.len() > MAX_DISTINCT {
                        break;
                    }
                }
            }
            if distinct.len() > MAX_DISTINCT {
                distinct.clear();
            } else {
                distinct.sort_by(f64::total_cmp);
            }
            // Both classes present, an all-ones column is constant, not a label
            let binary = distinct.len() == 2 && distinct[0] == 0.0 && (distinct[1] - 1.0).abs() < f64::EPSILON;
            ColumnInfo {
                index: c,
                name: names
                    .get(c)
                    .filter(|s| !s.is_empty())
                    .cloned()
                    .unwrap_or_else(|| format!("col {c}")),
                min,
                max,
                mean,
                std,
                constant: std == 0.0,
                distinct,
                binary,
            }
        })
        .collect();

    let info = DatasetInfo {
        id,
        name,
        has_header,
        rows: rows.len(),
        columns,
        sample: rows.into_iter().take(SAMPLE_ROWS).collect(),
        size_bytes: bytes.len(),
        is_sample,
    };
    Ok(StoredDataset { info, bytes })
}
