//! V3-OBS-1's stats observer: the cheap consumer of the carrier tap.
//!
//! Computes, while a write's vectors are still borrowed, only the
//! quantities the freeze (`docs/v3-obs-1-carrier-observation.md`, P5)
//! admits as observation statistics: the carrier's norm, the delta's
//! norm, a fixed low-dimensional projection whose basis carries its own
//! identity, and — when configured — raw dot products against a small
//! predeclared set of output-head rows. Nothing here runs a head pass:
//! per-layer top-k is a logit lens, one full head matmul per layer per
//! token, and belongs to an on-demand or replay consumer.
//!
//! This is the TREATMENT arm of the capture-cost protocol; the control
//! is [`NoopObserver`](super::observe::NoopObserver). Its cost is
//! measured, never described.

use std::fmt::Write as _;

use sha2::{Digest, Sha256};

use super::observe::{CarrierWriteRecord, StepEvent, StepObserver, SublayerSite};

/// The method every `norm` and `delta_norm` in a [`WriteStats`] was
/// computed by: Euclidean length, accumulated in f64.
pub const NORM_METHOD: &str = "l2-v1";
/// The probe method: raw carrier · head-row dot product — no final
/// norm, no softmax. A DIFFERENCE between two probe values is
/// meaningful; a probability is not derivable from them.
pub const PROBE_METHOD: &str = "dot-raw-v1";
/// The provider recorded on a basis built by [`FixedBasis::seeded`].
pub const SEEDED_BASIS_PROVIDER: &str = "seeded-orthonormal-v1";
/// The provider recorded on a basis built by [`FixedBasis::from_rows`]
/// when the caller names none.
pub const SUPPLIED_BASIS_PROVIDER: &str = "supplied-rows-v1";

/// Knuth's MMIX linear congruential constants — the generator behind
/// [`FixedBasis::seeded`], named so the basis is reproducible from its
/// seed on any machine.
const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
const LCG_INCREMENT: u64 = 1_442_695_040_888_963_407;
/// Bits discarded from each LCG state before it becomes a value: the
/// low bits of an LCG are the weak ones.
const LCG_DISCARD_BITS: u32 = 33;
/// Width of the value range after the discard (`2^31`).
const LCG_RANGE: f64 = (1u64 << 31) as f64;
/// A row whose norm falls below this after Gram–Schmidt is linearly
/// dependent on its predecessors and is refused rather than normalised
/// into noise.
const DEPENDENT_ROW_FLOOR: f64 = 1e-9;

/// A fixed low-dimensional projection basis with its own identity.
///
/// The executor never interprets the axes. Meaning — relation, entity,
/// an answer direction — belongs to whoever registered the basis; what
/// travels with every coordinate is enough to say WHICH basis produced
/// it: provider, id, content hash, dimensionality and source width.
#[derive(Debug, Clone)]
pub struct FixedBasis {
    provider: String,
    id: String,
    hidden: usize,
    rows: Vec<Vec<f32>>,
    hash: [u8; 32],
}

/// What identifies a basis on every coordinate it produces.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct BasisIdentity {
    pub provider: String,
    pub id: String,
    /// SHA-256 over the rows' exact f32 little-endian bytes, prefixed
    /// by the width and the row count, as lowercase hex.
    pub hash_hex: String,
    pub dims: usize,
    pub hidden: usize,
}

impl FixedBasis {
    /// A basis from caller-supplied rows (a registered reader, an
    /// answer-direction set). Every row must be `hidden` wide and there
    /// must be at least one; the hash covers the rows' exact bytes in
    /// order, so two callers holding the same rows get the same hash
    /// whatever they named them.
    pub fn from_rows(
        provider: Option<&str>,
        id: &str,
        hidden: usize,
        rows: Vec<Vec<f32>>,
    ) -> Result<Self, String> {
        if hidden == 0 {
            return Err("a basis needs a non-zero source width".to_string());
        }
        if rows.is_empty() {
            return Err("a basis needs at least one row".to_string());
        }
        if let Some((index, row)) = rows.iter().enumerate().find(|(_, r)| r.len() != hidden) {
            return Err(format!(
                "basis row {index} is {} wide; the source width is {hidden}",
                row.len()
            ));
        }
        let hash = content_hash(hidden, &rows);
        Ok(Self {
            provider: provider.unwrap_or(SUPPLIED_BASIS_PROVIDER).to_string(),
            id: id.to_string(),
            hidden,
            rows,
            hash,
        })
    }

    /// A deterministic pseudo-random orthonormal basis: `dims` rows of
    /// width `hidden`, Gram–Schmidt orthonormalised in f64, generated
    /// from `seed`. Reproducible across runs and machines, so two runs
    /// projected with the same seed share a coordinate system by
    /// construction — which the hash then proves rather than asserts.
    pub fn seeded(hidden: usize, dims: usize, seed: u64) -> Result<Self, String> {
        if dims == 0 || dims > hidden {
            return Err(format!(
                "a seeded basis needs 1..={hidden} dimensions, not {dims}"
            ));
        }
        let mut state = seed;
        let mut next = move || {
            state = state
                .wrapping_mul(LCG_MULTIPLIER)
                .wrapping_add(LCG_INCREMENT);
            (state >> LCG_DISCARD_BITS) as f64 / LCG_RANGE - 0.5
        };
        let mut rows: Vec<Vec<f64>> = Vec::with_capacity(dims);
        while rows.len() < dims {
            let mut row: Vec<f64> = (0..hidden).map(|_| next()).collect();
            for previous in &rows {
                let dot: f64 = row.iter().zip(previous).map(|(a, b)| a * b).sum();
                for (r, p) in row.iter_mut().zip(previous) {
                    *r -= dot * p;
                }
            }
            let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm < DEPENDENT_ROW_FLOOR {
                // A draw that landed in the span of its predecessors:
                // draw again rather than divide by nothing. With a
                // continuous generator this is a measure-zero event,
                // but the loop must terminate on it, not panic.
                continue;
            }
            rows.push(row.into_iter().map(|v| v / norm).collect());
        }
        let rows: Vec<Vec<f32>> = rows
            .into_iter()
            .map(|row| row.into_iter().map(|v| v as f32).collect())
            .collect();
        let hash = content_hash(hidden, &rows);
        Ok(Self {
            provider: SEEDED_BASIS_PROVIDER.to_string(),
            id: format!("seed-{seed}-{dims}x{hidden}"),
            hidden,
            rows,
            hash,
        })
    }

    pub fn identity(&self) -> BasisIdentity {
        BasisIdentity {
            provider: self.provider.clone(),
            id: self.id.clone(),
            hash_hex: hex(&self.hash),
            dims: self.rows.len(),
            hidden: self.hidden,
        }
    }

    pub fn dims(&self) -> usize {
        self.rows.len()
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// The rows, for a consumer that wants to check orthonormality or
    /// re-derive the hash for itself.
    pub fn rows(&self) -> &[Vec<f32>] {
        &self.rows
    }

    /// Project a `hidden`-wide vector: `dims` dot products, accumulated
    /// in f64, returned at the coordinates' declared precision.
    pub fn project(&self, v: &[f32]) -> Vec<f32> {
        debug_assert_eq!(
            v.len(),
            self.hidden,
            "projected a vector of the wrong width"
        );
        self.rows.iter().map(|row| dot(row, v) as f32).collect()
    }
}

/// A small, predeclared set of output-head rows and the tokens they
/// belong to. A raw dot product of the carrier against each row is the
/// selected-token statistic V3-OBS-1 admits; it is NOT a logit lens,
/// because no final norm and no full head run.
#[derive(Debug, Clone)]
pub struct HeadProbe {
    tokens: Vec<u32>,
    rows: Vec<Vec<f32>>,
}

impl HeadProbe {
    /// `tokens[i]`'s head row is `rows[i]`; every row is `hidden` wide.
    pub fn new(tokens: Vec<u32>, rows: Vec<Vec<f32>>, hidden: usize) -> Result<Self, String> {
        if tokens.len() != rows.len() {
            return Err(format!(
                "{} probe tokens but {} head rows",
                tokens.len(),
                rows.len()
            ));
        }
        if let Some((index, row)) = rows.iter().enumerate().find(|(_, r)| r.len() != hidden) {
            return Err(format!(
                "probe row {index} (token {}) is {} wide; the carrier is {hidden}",
                tokens[index],
                row.len()
            ));
        }
        Ok(Self { tokens, rows })
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    fn dots(&self, v: &[f32]) -> Vec<f32> {
        self.rows.iter().map(|row| dot(row, v) as f32).collect()
    }
}

/// One write's statistics, in the write's coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct WriteStats {
    pub layer: usize,
    pub site: SublayerSite,
    pub position: usize,
    /// `‖after‖₂` by [`NORM_METHOD`].
    pub norm: f64,
    /// `‖delta‖₂` by [`NORM_METHOD`].
    pub delta_norm: f64,
    /// Carried through from the record: the scale the layer applies to
    /// `after` before its boundary, where the program has one.
    pub layer_scale: Option<f32>,
    /// `after` in the observer's basis, one coordinate per basis row.
    pub projection: Vec<f32>,
    /// One value per probe token by [`PROBE_METHOD`], in the probe's
    /// token order; empty when no probe is configured.
    pub probe: Vec<f32>,
}

/// The stats consumer: one [`WriteStats`] row per single-stream carrier
/// write, computed at the tap.
#[derive(Debug)]
pub struct StatsObserver {
    basis: FixedBasis,
    probe: Option<HeadProbe>,
    pub rows: Vec<WriteStats>,
    /// Structural events seen — counted, not kept, because the cost arm
    /// must not pay for storage the control arm does not.
    pub events: usize,
}

impl StatsObserver {
    pub fn new(basis: FixedBasis, probe: Option<HeadProbe>) -> Self {
        Self {
            basis,
            probe,
            rows: Vec::new(),
            events: 0,
        }
    }

    pub fn basis(&self) -> &FixedBasis {
        &self.basis
    }

    pub fn probe(&self) -> Option<&HeadProbe> {
        self.probe.as_ref()
    }

    /// Hand over the rows collected so far and start again.
    pub fn take_rows(&mut self) -> Vec<WriteStats> {
        std::mem::take(&mut self.rows)
    }
}

impl StepObserver for StatsObserver {
    fn event(&mut self, _event: StepEvent) {
        self.events += 1;
    }

    fn carrier_write(&mut self, record: CarrierWriteRecord<'_>) {
        self.rows.push(WriteStats {
            layer: record.layer,
            site: record.site,
            position: record.position,
            norm: l2(record.after),
            delta_norm: l2(record.delta),
            layer_scale: record.layer_scale,
            projection: self.basis.project(record.after),
            probe: self
                .probe
                .as_ref()
                .map(|probe| probe.dots(record.after))
                .unwrap_or_default(),
        });
    }
}

fn l2(v: &[f32]) -> f64 {
    v.iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt()
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum()
}

fn content_hash(hidden: usize, rows: &[Vec<f32>]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update((hidden as u64).to_le_bytes());
    hasher.update((rows.len() as u64).to_le_bytes());
    for row in rows {
        for value in row {
            hasher.update(value.to_le_bytes());
        }
    }
    hasher.finalize().into()
}

fn hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}
