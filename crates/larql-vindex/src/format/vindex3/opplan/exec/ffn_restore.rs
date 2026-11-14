//! Clean-run restoration at an FFN site (SG-EDIT-1 I5, item A).
//!
//! At one position of one layer write `r = a + δ`, where `a` is the
//! carrier entering the FFN's residual add (the attention site's `after`)
//! and `δ` is the executed FFN write: after the FFN's post-norm and
//! residual-delta scale, as the carrier received it. Subscript `c` is the
//! clean run and `k` the corrupted one. Two restorations exist, and they
//! mean different things:
//!
//! * **FFN-write restoration** puts the clean FFN write on the corrupted
//!   carrier: `Add(δ_c − δ_k)` at `(layer, Ffn, p)`, expected state
//!   `a_k + δ_c`. It is NOT `r_c` unless `a_k = a_c`, which a corruption
//!   upstream of the layer does not give.
//! * **Full-carrier restoration** puts the clean carrier back:
//!   `Replace(r_c)` at `(layer, Ffn, p)`, expected state `r_c` bit for bit.
//!
//! Each `δ` is read from the run's own carriers as `after(Ffn) −
//! after(Attention)`. An FFN-site intervention at layer `ℓ` cannot change
//! anything at or before `(ℓ, Attention)` at any position, so the
//! corrupted run's `a_k` and `δ_k`, captured once without intervention,
//! stay the values the intervened run sees. The restoration drivers below
//! take them from two [`CarrierCapture`]s, one per run.
//!
//! Every vector is a [`VectorProvenance::Literal`]: the derivation lives on
//! the returned [`RestorationRecord`], beside each vector's hash.

use super::intervene::{
    vector_sha256, Address, CarrierCapture, Intervention, InterventionKind, InterventionPlan,
    VectorProvenance,
};
use super::observe::SublayerSite;
use crate::error::VindexError;

/// The unit roundoff of `f32`, round to nearest: half its epsilon.
const F32_UNIT_ROUNDOFF: f64 = f32::EPSILON as f64 / 2.0;

/// The f32 roundings between the captured carriers and the restored one:
/// `δ_c`, `δ_k`, their difference, and the residual add that lands it.
const RESTORATION_ROUNDINGS: f64 = 4.0;

/// Which restoration a vector implements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Restoration {
    /// `Add(δ_c − δ_k)`; expected state `a_k + δ_c`.
    FfnWrite,
    /// `Replace(r_c)`; expected state `r_c`.
    FullCarrier,
}

/// One restored position: what was patched in and where it came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestoredPosition {
    pub position: usize,
    pub vector_sha256: String,
}

/// The derivation behind a restoration plan. A receipt carries this; the
/// plan's own provenance only says "literal".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestorationRecord {
    pub restoration: Restoration,
    pub layer: usize,
    pub clean_run: String,
    pub corrupt_run: Option<String>,
    pub positions: Vec<RestoredPosition>,
}

/// The carrier addresses a capture must declare so the drivers can read
/// both runs at `layer` over `positions`.
pub fn capture_addresses(layer: usize, positions: &[usize]) -> Vec<(usize, SublayerSite, usize)> {
    positions
        .iter()
        .flat_map(|&p| {
            [
                (layer, SublayerSite::Attention, p),
                (layer, SublayerSite::Ffn, p),
            ]
        })
        .collect()
}

fn carrier<'c>(
    capture: &'c CarrierCapture,
    run: &str,
    layer: usize,
    site: SublayerSite,
    position: usize,
) -> Result<&'c [f32], VindexError> {
    capture.get(layer, site, position).ok_or_else(|| {
        VindexError::Parse(format!(
            "restoration needs run `{run}`'s carrier at layer {layer} {site:?} position \
             {position}, which was not captured"
        ))
    })
}

/// The executed FFN write `after(Ffn) − after(Attention)` at one position,
/// in f32, as the restoration applies it.
pub fn ffn_write(
    capture: &CarrierCapture,
    run: &str,
    layer: usize,
    position: usize,
) -> Result<Vec<f32>, VindexError> {
    let entering = carrier(capture, run, layer, SublayerSite::Attention, position)?;
    let written = carrier(capture, run, layer, SublayerSite::Ffn, position)?;
    Ok(written.iter().zip(entering).map(|(r, a)| r - a).collect())
}

fn require_positions(positions: &[usize]) -> Result<(), VindexError> {
    if positions.is_empty() {
        return Err(VindexError::Parse(
            "a restoration names no position".to_string(),
        ));
    }
    Ok(())
}

/// FFN-write restoration: one `Add(δ_c − δ_k)` per position at
/// `(layer, Ffn, p)`. Each position carries its own vector, so each is its
/// own intervention.
pub fn ffn_write_restoration(
    clean: &CarrierCapture,
    clean_run: &str,
    corrupt: &CarrierCapture,
    corrupt_run: &str,
    layer: usize,
    positions: &[usize],
) -> Result<(InterventionPlan, RestorationRecord), VindexError> {
    require_positions(positions)?;
    let mut plan = InterventionPlan::none();
    let mut restored = Vec::with_capacity(positions.len());
    for &position in positions {
        let clean_write = ffn_write(clean, clean_run, layer, position)?;
        let corrupt_write = ffn_write(corrupt, corrupt_run, layer, position)?;
        let difference: Vec<f32> = clean_write
            .iter()
            .zip(&corrupt_write)
            .map(|(c, k)| c - k)
            .collect();
        let sha256 = vector_sha256(&difference);
        let intervention = Intervention::add(
            Address::new(layer, SublayerSite::Ffn, [position])?,
            difference,
            VectorProvenance::Literal {
                sha256: sha256.clone(),
            },
        )?;
        plan = plan.with(intervention)?;
        restored.push(RestoredPosition {
            position,
            vector_sha256: sha256,
        });
    }
    Ok((
        plan,
        RestorationRecord {
            restoration: Restoration::FfnWrite,
            layer,
            clean_run: clean_run.to_string(),
            corrupt_run: Some(corrupt_run.to_string()),
            positions: restored,
        },
    ))
}

/// Full-carrier restoration: one captured `Replace(r_c)` per position at
/// `(layer, Ffn, p)`.
pub fn full_carrier_restoration(
    clean: &CarrierCapture,
    clean_run: &str,
    layer: usize,
    positions: &[usize],
) -> Result<(InterventionPlan, RestorationRecord), VindexError> {
    require_positions(positions)?;
    let mut plan = InterventionPlan::none();
    let mut restored = Vec::with_capacity(positions.len());
    for &position in positions {
        let from = (layer, SublayerSite::Ffn, position);
        let intervention = clean.intervention(
            clean_run,
            from,
            InterventionKind::Replace,
            Address::new(layer, SublayerSite::Ffn, [position])?,
        )?;
        let sha256 = intervention
            .vector_sha256()
            .expect("a replace carries its vector's hash")
            .to_string();
        plan = plan.with(intervention)?;
        restored.push(RestoredPosition {
            position,
            vector_sha256: sha256,
        });
    }
    Ok((
        plan,
        RestorationRecord {
            restoration: Restoration::FullCarrier,
            layer,
            clean_run: clean_run.to_string(),
            corrupt_run: None,
            positions: restored,
        },
    ))
}

/// The FFN-write contract at one position: how far the restored carrier
/// is from the exact `a_k + δ_c`, as a multiple of the f32 rounding bound
/// for the four roundings between the captures and the carrier. A value
/// at most `1.0` meets the contract.
///
/// The bound is elementwise: `RESTORATION_ROUNDINGS · u · (|r_c| + |a_c| +
/// |r_k| + |a_k| + |restored|)`, which dominates the first-order error of
/// computing both writes, their difference and the residual add in f32.
/// Elements where the bound is zero must match exactly.
pub fn ffn_write_contract_ratio(
    clean: &CarrierCapture,
    clean_run: &str,
    corrupt: &CarrierCapture,
    corrupt_run: &str,
    layer: usize,
    position: usize,
    restored: &[f32],
) -> Result<f64, VindexError> {
    let a_c = carrier(clean, clean_run, layer, SublayerSite::Attention, position)?;
    let r_c = carrier(clean, clean_run, layer, SublayerSite::Ffn, position)?;
    let a_k = carrier(
        corrupt,
        corrupt_run,
        layer,
        SublayerSite::Attention,
        position,
    )?;
    let r_k = carrier(corrupt, corrupt_run, layer, SublayerSite::Ffn, position)?;
    if restored.len() != a_k.len() {
        return Err(VindexError::Parse(format!(
            "restored carrier has width {} but the captured carriers have width {}",
            restored.len(),
            a_k.len()
        )));
    }
    let mut worst = 0.0f64;
    for i in 0..restored.len() {
        let expected = f64::from(a_k[i]) + (f64::from(r_c[i]) - f64::from(a_c[i]));
        let error = (f64::from(restored[i]) - expected).abs();
        let magnitude = f64::from(r_c[i]).abs()
            + f64::from(a_c[i]).abs()
            + f64::from(r_k[i]).abs()
            + f64::from(a_k[i]).abs()
            + f64::from(restored[i]).abs();
        let bound = RESTORATION_ROUNDINGS * F32_UNIT_ROUNDOFF * magnitude;
        let ratio = if bound == 0.0 {
            if error == 0.0 {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            error / bound
        };
        worst = worst.max(ratio);
    }
    Ok(worst)
}
