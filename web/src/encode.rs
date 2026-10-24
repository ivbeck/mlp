//! Turns typed text columns into the plain number matrix the crate trains on. Everything
//! here is the console's own preprocessing, the crate only ever sees the encoded numbers.

use std::collections::HashMap;
use std::f64::consts::TAU;

use anyhow::Result;
use multilayer_perceptron::data::Dataset;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::dataset::{is_missing, parse_date, ColumnInfo, DatasetInfo, Kind};

/// One-hot columns per category column, the rest share an "other" slot
pub const MAX_ONE_HOT: usize = 30;
const MISSING: &str = "(missing)";
const OTHER: &str = "(other)";

/// One raw value for one input column: a number from a chart or a field, or a cell
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Cell {
    Num(f64),
    Text(String),
}

impl Cell {
    fn text(&self) -> String {
        match self {
            Self::Num(v) => v.to_string(),
            Self::Text(s) => s.clone(),
        }
    }
    fn num(&self) -> Option<f64> {
        match self {
            Self::Num(v) => v.is_finite().then_some(*v),
            Self::Text(s) => (!is_missing(s)).then(|| s.parse().ok()).flatten(),
        }
    }
}

#[derive(Debug, Clone)]
enum Enc {
    /// Missing cells take the mean, plus a 0/1 column saying so when the data has any
    Numeric {
        mean: f64,
        flag: bool,
    },
    Binary {
        one: String,
    },
    /// One-hot over the listed values, missing counts as its own value
    Category {
        cats: Vec<String>,
        other: bool,
    },
    /// Year, then month and weekday (and hour) as points on a circle so December sits
    /// next to January
    Date {
        time: bool,
    },
}

/// One source column that feeds the network, what the ui shows for it in Probe
#[derive(Debug, Clone, Serialize)]
pub struct InputInfo {
    pub index: usize,
    pub name: String,
    pub kind: Kind,
    /// Values offered in Probe for binary and category inputs
    pub options: Vec<String>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    /// Mean for numbers, most common value otherwise; held there when not on the chart
    pub default: Cell,
    /// How many network inputs this column becomes
    pub width: usize,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TargetSpec {
    Binary { one: String, zero: String },
    Numeric,
}

#[derive(Debug, Clone)]
pub struct Encoder {
    plans: Vec<(usize, Enc)>,
    pub inputs: Vec<InputInfo>,
    /// One name per network input, `workclass = Private` style
    pub names: Vec<String>,
    pub target: TargetSpec,
    pub target_index: usize,
}

fn effective(col: &ColumnInfo, overrides: &HashMap<usize, Kind>) -> Kind {
    overrides
        .get(&col.index)
        .copied()
        .filter(|k| col.allowed.contains(k))
        .unwrap_or(col.kind)
}

fn date_parts(cell: &str) -> Option<[f64; 7]> {
    let (year, month, day, hour) = parse_date(cell)?;
    // Sakamoto's weekday, 0 = Sunday
    let offsets = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let yy = if month < 3 { year - 1 } else { year };
    let weekday = (yy + yy / 4 - yy / 100 + yy / 400 + offsets[(month - 1) as usize] + i32::try_from(day).unwrap_or(1))
        .rem_euclid(7);
    let mo = TAU * f64::from(month - 1) / 12.0;
    let wk = TAU * f64::from(weekday) / 7.0;
    let hr = TAU * f64::from(hour.unwrap_or(0)) / 24.0;
    Some([f64::from(year), mo.sin(), mo.cos(), wk.sin(), wk.cos(), hr.sin(), hr.cos()])
}

impl Encoder {
    /// Plan the encoding of every used column against the whole file
    ///
    /// # Errors
    ///
    /// A target that is neither numbers nor two values, or nothing left to learn from
    #[allow(clippy::too_many_lines)]
    pub fn build(
        info: &DatasetInfo,
        rows: &[Vec<String>],
        target: usize,
        overrides: &HashMap<usize, Kind>,
        positive: Option<&str>,
    ) -> Result<Self> {
        let tcol = info
            .columns
            .get(target)
            .ok_or_else(|| anyhow::anyhow!("Target column {target} does not exist"))?;
        let target_spec = if tcol.binary || (tcol.distinct == 2 && tcol.min.is_none()) {
            let mut vals: Vec<&str> = tcol.categories.iter().map(|(v, _)| v.as_str()).collect();
            vals.sort_unstable();
            // 1 is "1", "true" or "yes" when there is one, else the rarer value
            let natural = vals
                .iter()
                .find(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes" | "y"))
                .copied();
            let rarer = tcol.categories.last().map(|(v, _)| v.as_str());
            let one = positive
                .filter(|p| vals.contains(p))
                .or(natural)
                .or(rarer)
                .unwrap_or_default()
                .to_string();
            let zero = vals
                .iter()
                .find(|v| **v != one)
                .copied()
                .unwrap_or_default()
                .to_string();
            TargetSpec::Binary { one, zero }
        } else if tcol.min.is_some() {
            TargetSpec::Numeric
        } else {
            anyhow::bail!(
                "{} has {} different values. The crate trains a single output, so it can predict a number or one of two classes, not {} classes.",
                tcol.name,
                tcol.distinct,
                tcol.distinct
            );
        };

        let mut plans = Vec::new();
        let mut inputs = Vec::new();
        let mut names = Vec::new();
        for col in info.columns.iter().filter(|c| c.index != target) {
            let kind = effective(col, overrides);
            let cells = || rows.iter().map(|r| r[col.index].as_str());
            let start = names.len();
            let (enc, options, default) = match kind {
                Kind::Ignore => continue,
                Kind::Numeric => {
                    let mean = col.mean.unwrap_or(0.0);
                    names.push(col.name.clone());
                    let flag = col.missing > 0;
                    if flag {
                        names.push(format!("{} missing", col.name));
                    }
                    (Enc::Numeric { mean, flag }, Vec::new(), Cell::Num(mean))
                }
                Kind::Binary => {
                    let mut vals: Vec<String> =
                        col.categories.iter().map(|(v, _)| v.clone()).collect();
                    vals.sort();
                    let one = vals
                        .iter()
                        .find(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes" | "y"))
                        .cloned()
                        .or_else(|| vals.last().cloned())
                        .unwrap_or_default();
                    names.push(format!("{} = {one}", col.name));
                    let mode = col
                        .categories
                        .first()
                        .map_or_else(String::new, |(v, _)| v.clone());
                    (Enc::Binary { one }, vals, Cell::Text(mode))
                }
                Kind::Category => {
                    let mut cats: Vec<String> = col
                        .categories
                        .iter()
                        .take(MAX_ONE_HOT)
                        .map(|(v, _)| v.clone())
                        .collect();
                    if col.missing > 0 {
                        cats.push(MISSING.into());
                    }
                    let other = col.distinct > MAX_ONE_HOT;
                    for c in &cats {
                        names.push(format!("{} = {c}", col.name));
                    }
                    if other {
                        names.push(format!("{} = {OTHER}", col.name));
                    }
                    let mode = cats.first().cloned().unwrap_or_default();
                    let mut options = cats.clone();
                    if other {
                        options.push(OTHER.into());
                    }
                    (Enc::Category { cats, other }, options, Cell::Text(mode))
                }
                Kind::Date => {
                    let time = cells().any(|c| c.contains(':'));
                    names.extend(
                        [
                            "year",
                            "month sin",
                            "month cos",
                            "weekday sin",
                            "weekday cos",
                        ]
                        .iter()
                        .map(|p| format!("{} {p}", col.name)),
                    );
                    if time {
                        names.push(format!("{} hour sin", col.name));
                        names.push(format!("{} hour cos", col.name));
                    }
                    let first = cells()
                        .find(|c| parse_date(c).is_some())
                        .unwrap_or_default()
                        .to_string();
                    (Enc::Date { time }, Vec::new(), Cell::Text(first))
                }
            };
            inputs.push(InputInfo {
                index: col.index,
                name: col.name.clone(),
                kind,
                options,
                min: col.min,
                max: col.max,
                default,
                width: names.len() - start,
            });
            plans.push((col.index, enc));
        }
        if names.is_empty() {
            anyhow::bail!("Every feature column is ignored, there is nothing to learn from");
        }
        Ok(Self {
            plans,
            inputs,
            names,
            target: target_spec,
            target_index: target,
        })
    }

    fn push(enc: &Enc, cell: &Cell, out: &mut Vec<f64>) {
        match enc {
            Enc::Numeric { mean, flag } => {
                let v = cell.num();
                out.push(v.unwrap_or(*mean));
                if *flag {
                    out.push(if v.is_some() { 0.0 } else { 1.0 });
                }
            }
            Enc::Binary { one } => out.push(if cell.text() == *one { 1.0 } else { 0.0 }),
            Enc::Category { cats, other } => {
                let t = cell.text();
                let key = if is_missing(&t) {
                    MISSING.to_string()
                } else {
                    t
                };
                let hit = cats.iter().position(|c| *c == key);
                out.extend(
                    cats.iter()
                        .enumerate()
                        .map(|(i, _)| if Some(i) == hit { 1.0 } else { 0.0 }),
                );
                if *other {
                    out.push(if hit.is_none() { 1.0 } else { 0.0 });
                }
            }
            Enc::Date { time } => {
                let p = date_parts(&cell.text()).unwrap_or([0.0; 7]);
                out.extend_from_slice(&p[..if *time { 7 } else { 5 }]);
            }
        }
    }

    /// Network input for one value per input column, in `inputs` order
    pub fn encode(&self, cells: &[Cell]) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.names.len());
        for ((_, enc), cell) in self.plans.iter().zip(cells) {
            Self::push(enc, cell, &mut out);
        }
        out
    }

    fn label(&self, s: &str) -> Option<f64> {
        if is_missing(s) {
            return None;
        }
        match &self.target {
            TargetSpec::Binary { one, .. } => Some(if s == one { 1.0 } else { 0.0 }),
            TargetSpec::Numeric => s.parse().ok(),
        }
    }

    /// The whole file as the crate's dataset, plus how many rows had no target
    ///
    /// # Errors
    ///
    /// No rows with a target value
    pub fn dataset(&self, rows: &[Vec<String>]) -> Result<(Dataset, usize)> {
        let mut feats = Vec::with_capacity(rows.len() * self.names.len());
        let mut labels = Vec::with_capacity(rows.len());
        let mut dropped = 0;
        for row in rows {
            let Some(y) = self.label(&row[self.target_index]) else {
                dropped += 1;
                continue;
            };
            feats.extend(self.encode(&self.row_cells(row)));
            labels.push(y);
        }
        if labels.is_empty() {
            anyhow::bail!("No row has a value in the target column");
        }
        let features = Array2::from_shape_vec((labels.len(), self.names.len()), feats)?;
        Ok((Dataset::new(features, Array1::from_vec(labels)), dropped))
    }

    /// The cells of one file row, in `inputs` order
    pub fn row_cells(&self, row: &[String]) -> Vec<Cell> {
        self.plans
            .iter()
            .map(|(i, _)| Cell::Text(row[*i].clone()))
            .collect()
    }
}
