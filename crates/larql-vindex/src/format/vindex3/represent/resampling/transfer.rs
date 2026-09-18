//! **A3 — selection to heldout transfer.**
//!
//! A band frozen on one bank, applied to another's values. Two
//! readings, never conflated: the RELATIVE rule (each bank against its
//! own full value — does the sampling behaviour transfer?) and the
//! ABSOLUTE rule (confounded by the between-set difference by
//! construction, reported so nobody mistakes one for the other).

use serde::Serialize;

use super::summary::{relative, Band};

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Coverage {
    pub sequences: usize,
    pub samples: usize,
    pub nominal: f64,
    /// Fraction of the target's relative values inside the source's
    /// relative band.
    pub relative_coverage: f64,
    /// Fraction of the target's absolute values inside the source's
    /// absolute band.
    pub absolute_coverage: f64,
    pub source_band: Band,
    pub source_absolute_band: Band,
}

fn fraction_inside(band: &Band, values: &[f64]) -> f64 {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    finite.iter().filter(|v| band.contains(**v)).count() as f64 / finite.len() as f64
}

/// Coverage of `target` values (with `target_full`) by bands frozen on
/// `source` values (with `source_full`).
pub fn coverage(
    sequences: usize,
    source: &[f64],
    source_full: f64,
    target: &[f64],
    target_full: f64,
) -> Coverage {
    let source_band = Band::of(&relative(source, source_full));
    let source_absolute_band = Band::of(source);
    Coverage {
        sequences,
        samples: target.len(),
        nominal: super::BAND_HIGH - super::BAND_LOW,
        relative_coverage: fraction_inside(&source_band, &relative(target, target_full)),
        absolute_coverage: fraction_inside(&source_absolute_band, target),
        source_band,
        source_absolute_band,
    }
}
