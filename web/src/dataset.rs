//! Uploaded CSV files: kept as text cells, typed per column (number, two values, category,
//! date), so a run can encode them into the numbers the crate trains on

use std::collections::HashMap;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Rows sent to the browser for scatter plots and the probe's row picker
const SAMPLE_ROWS: usize = 600;
/// Categories listed per column in the summary
const MAX_LISTED: usize = 60;
/// A text column with more distinct values than this is taken for an id or free text
const MAX_AUTO_CATEGORIES: usize = 60;

/// Cells that mean "no value" in the files people export
pub fn is_missing(s: &str) -> bool {
    matches!(
        s,
        "" | "?"
            | "NA"
            | "N/A"
            | "na"
            | "n/a"
            | "nan"
            | "NaN"
            | "NAN"
            | "null"
            | "NULL"
            | "None"
            | "none"
            | "-"
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Numeric,
    Binary,
    Category,
    Date,
    Ignore,
}

#[derive(Debug, Clone)]
pub struct StoredDataset {
    pub info: DatasetInfo,
    /// Every row, trimmed cells, in file order
    pub rows: Vec<Vec<String>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DatasetInfo {
    pub id: String,
    pub name: String,
    pub has_header: bool,
    pub rows: usize,
    pub columns: Vec<ColumnInfo>,
    /// First `SAMPLE_ROWS` rows as text, in file order
    pub sample: Vec<Vec<String>>,
    pub size_bytes: usize,
    /// True for the bundled example file
    pub is_sample: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ColumnInfo {
    pub index: usize,
    pub name: String,
    /// What the console makes of the column unless told otherwise
    pub kind: Kind,
    /// Kinds the column can be used as
    pub allowed: Vec<Kind>,
    /// Why the detected kind is what it is, when that is not obvious
    pub reason: Option<String>,
    pub missing: usize,
    pub distinct: usize,
    /// Stats over the cells that read as numbers, None for text columns
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub mean: Option<f64>,
    /// Population std, the same one `Dataset::normalize` divides by
    pub std: Option<f64>,
    pub constant: bool,
    /// Most frequent values first, with counts, at most `MAX_LISTED`
    pub categories: Vec<(String, usize)>,
    /// Exactly two distinct values and nothing missing
    pub binary: bool,
}

/// Year, month, day and optional hour from `YYYY-MM-DD[ T]hh:mm[:ss]`
pub fn parse_date(cell: &str) -> Option<(i32, u32, u32, Option<u32>)> {
    let (date, time) = cell
        .split_once(['T', ' '])
        .map_or((cell, None), |(d, t)| (d, Some(t)));
    let mut parts = date.split('-');
    let year: i32 = parts.next()?.parse().ok()?;
    let month: u32 = parts.next()?.parse().ok()?;
    let day: u32 = parts.next()?.parse().ok()?;
    if parts.next().is_some()
        || !(1000..=9999).contains(&year)
        || !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
    {
        return None;
    }
    let hour = match time {
        Some(clock) => {
            let hour: u32 = clock.split(':').next()?.parse().ok()?;
            if hour >= 24 {
                return None;
            }
            Some(hour)
        }
        None => None,
    };
    Some((year, month, day, hour))
}

/// A header row is assumed when some column reads as numbers below a text first cell,
/// or every first cell is text that does not come back further down
fn sniff_header(records: &[Vec<String>]) -> bool {
    let Some(first) = records.first() else {
        return false;
    };
    let rest = &records[1..records.len().min(400)];
    if rest.is_empty() {
        return first.iter().any(|f| f.parse::<f64>().is_err());
    }
    let numeric_below = first.iter().enumerate().any(|(c, f)| {
        f.parse::<f64>().is_err()
            && !is_missing(f)
            && rest.iter().all(|r| {
                r.get(c)
                    .is_some_and(|v| is_missing(v) || v.parse::<f64>().is_ok())
            })
    });
    let fresh_text = first.iter().enumerate().all(|(c, f)| {
        f.parse::<f64>().is_err() && !is_missing(f) && rest.iter().all(|r| r.get(c) != Some(f))
    });
    numeric_below || fresh_text
}

#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
fn describe(index: usize, name: String, cells: &[&str]) -> ColumnInfo {
    let present: Vec<&str> = cells.iter().copied().filter(|c| !is_missing(c)).collect();
    let missing = cells.len() - present.len();

    let mut counts: HashMap<&str, usize> = HashMap::new();
    for c in &present {
        *counts.entry(c).or_insert(0) += 1;
    }
    let distinct = counts.len();
    let mut categories: Vec<(String, usize)> =
        counts.iter().map(|(k, v)| ((*k).to_string(), *v)).collect();
    categories.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    categories.truncate(MAX_LISTED);

    let numbers: Vec<f64> = present
        .iter()
        .filter_map(|c| c.parse::<f64>().ok())
        .collect();
    let all_numeric = !present.is_empty() && numbers.len() == present.len();
    let (min, max, mean, std) = if all_numeric {
        let n = numbers.len() as f64;
        let mean = numbers.iter().sum::<f64>() / n;
        (
            Some(numbers.iter().copied().fold(f64::INFINITY, f64::min)),
            Some(numbers.iter().copied().fold(f64::NEG_INFINITY, f64::max)),
            Some(mean),
            Some((numbers.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt()),
        )
    } else {
        (None, None, None, None)
    };
    let all_dates =
        !present.is_empty() && !all_numeric && present.iter().all(|c| parse_date(c).is_some());
    let binary = distinct == 2 && missing == 0;

    let mut allowed = Vec::new();
    if all_numeric {
        allowed.push(Kind::Numeric);
    }
    if binary {
        allowed.push(Kind::Binary);
    }
    if (2..=2000).contains(&distinct) {
        allowed.push(Kind::Category);
    }
    if all_dates {
        allowed.push(Kind::Date);
    }
    allowed.push(Kind::Ignore);

    let lower = name.to_lowercase();
    let id_like = distinct == cells.len() && cells.len() > 20;
    let (kind, reason) = if distinct <= 1 {
        (Kind::Ignore, Some("never changes".to_string()))
    } else if all_numeric {
        if binary
            && numbers
                .iter()
                .all(|&v| v == 0.0 || (v - 1.0).abs() < f64::EPSILON)
        {
            (Kind::Binary, None)
        } else if id_like && (lower == "id" || lower.ends_with("_id") || lower == "index") {
            (
                Kind::Ignore,
                Some("a unique number per row, looks like an id".to_string()),
            )
        } else {
            (Kind::Numeric, None)
        }
    } else if all_dates {
        (Kind::Date, None)
    } else if binary {
        (Kind::Binary, None)
    } else if distinct <= MAX_AUTO_CATEGORIES {
        (Kind::Category, None)
    } else if id_like {
        (
            Kind::Ignore,
            Some("a different value in every row, looks like an id".to_string()),
        )
    } else {
        (
            Kind::Ignore,
            Some(format!(
                "{distinct} distinct values, too many to use as categories by default"
            )),
        )
    };

    ColumnInfo {
        index,
        name,
        kind,
        allowed,
        reason,
        missing,
        distinct,
        min,
        max,
        mean,
        std,
        constant: distinct <= 1,
        categories,
        binary,
    }
}

/// A text column whose values map one to one onto a number column (education and
/// `education_num`) says the same thing twice. Feeding both lets the chart and the probe
/// ask for combinations that never occur, so the text one is ignored by default.
fn mark_duplicates(columns: &mut [ColumnInfo], records: &[Vec<String>]) {
    let numeric: Vec<usize> = columns
        .iter()
        .filter(|c| c.kind == Kind::Numeric)
        .map(|c| c.index)
        .collect();
    for c in 0..columns.len() {
        if columns[c].kind != Kind::Category {
            continue;
        }
        let twin = numeric.iter().copied().find(|&n| {
            columns[n].distinct == columns[c].distinct && {
                let mut fwd: HashMap<&str, &str> = HashMap::new();
                let mut back: HashMap<&str, &str> = HashMap::new();
                records.iter().all(|r| {
                    let (t, v) = (r[c].as_str(), r[n].as_str());
                    *fwd.entry(t).or_insert(v) == v && *back.entry(v).or_insert(t) == t
                })
            }
        });
        if let Some(n) = twin {
            let other = columns[n].name.clone();
            columns[c].kind = Kind::Ignore;
            columns[c].reason = Some(format!("same information as {other}, one value for one value"));
        }
    }
}

/// Parse an upload. `has_header` of `None` means guess.
///
/// # Errors
///
/// Unreadable CSV, ragged rows, or fewer than two columns
pub fn parse(
    id: String,
    name: String,
    bytes: &[u8],
    has_header: Option<bool>,
    is_sample: bool,
) -> Result<StoredDataset> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .trim(csv::Trim::All)
        .flexible(true)
        .from_reader(bytes);
    let mut records: Vec<Vec<String>> = Vec::new();
    for (n, rec) in reader.records().enumerate() {
        let rec = rec.with_context(|| format!("Failed to read row {n}"))?;
        if rec.iter().all(str::is_empty) {
            continue;
        }
        records.push(rec.iter().map(str::to_string).collect());
    }
    if records.is_empty() {
        anyhow::bail!("No data found in the file");
    }
    let has_header = has_header.unwrap_or_else(|| sniff_header(&records));
    let names: Vec<String> = if has_header {
        records.remove(0)
    } else {
        Vec::new()
    };
    let width = records.first().map_or(names.len(), Vec::len);
    if let Some((n, r)) = records.iter().enumerate().find(|(_, r)| r.len() != width) {
        anyhow::bail!(
            "Row {} has {} fields, the first row has {width}",
            n + usize::from(has_header) + 1,
            r.len()
        );
    }
    if width < 2 {
        anyhow::bail!("Need at least one feature column and one target column");
    }
    if records.is_empty() {
        anyhow::bail!("The file has a header but no rows");
    }

    let mut columns: Vec<ColumnInfo> = (0..width)
        .map(|c| {
            let cells: Vec<&str> = records.iter().map(|r| r[c].as_str()).collect();
            let name = names
                .get(c)
                .filter(|s| !s.is_empty())
                .cloned()
                .unwrap_or_else(|| format!("col {c}"));
            describe(c, name, &cells)
        })
        .collect();
    mark_duplicates(&mut columns, &records);

    let info = DatasetInfo {
        id,
        name,
        has_header,
        rows: records.len(),
        columns,
        sample: records.iter().take(SAMPLE_ROWS).cloned().collect(),
        size_bytes: bytes.len(),
        is_sample,
    };
    Ok(StoredDataset {
        info,
        rows: records,
    })
}
