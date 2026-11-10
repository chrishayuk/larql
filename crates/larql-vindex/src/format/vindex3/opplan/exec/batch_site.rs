//! Batch-site entry and exit around a layer.

use self::attention_residual::{BoundaryPhase, History};
use crate::error::VindexError;
use address::CarrierAddress;
use backend::PlanBackend;
use hyper_connection::{Bundle, Mutation, SinkhornSplit, SiteReduction};
use larql_models::config::HyperConnection;
use observe::{CarrierForm, CarrierTransition, HcSite, HistoryWriteMode};
use prepared::{PreparedAttnResSite, PreparedHcSite};
use std::borrow::Cow;
use weights::LoadedWeight;

#[allow(unused_imports)]
use super::*;

/// What entering one site produced for the whole batch: the `[hidden]`
/// vector each position's branch sees, and the reductions the update
/// needs afterwards — `None` on rows, where the update is the residual
/// add. The inputs BORROW a row plane (the branch reads the residual
/// itself) and are owned on a bundle plane (the branch reads what the
/// site reduced).
pub(super) struct BatchSiteEntry<'a> {
    pub(super) branch_inputs: Cow<'a, [Vec<f32>]>,
    pub(super) reductions: Option<Vec<SiteReduction>>,
    /// One reduction per position on a history plane, present exactly
    /// when the site reduced. `None` where the reference does not reduce
    /// — layer 0's attention site — and the branch input is then each
    /// position's own prefix.
    ///
    /// The "did it reduce" decision is per LAYER, not per position:
    /// every position snapshots at the same layers, so their histories
    /// always hold the same COUNT. What differs between positions is the
    /// snapshots' values, which is exactly what the swap and broadcast
    /// controls attack.
    pub(super) attn_res: Option<AttnResBatchEntry>,
}

/// What one attention-residual site read on the way in, per position.
///
/// The state is captured HERE, at entry, and carried to the record
/// emitted on the way out — the same two-point capture the decode
/// traversal performs. It exists so that a batch record and a decode
/// record describe the same moment of the same site, which is the
/// premise A7 rests on.
pub(super) struct AttnResBatchEntry {
    pub(super) reductions: Vec<attention_residual::Reduction>,
    pub(super) prefixes_before: Vec<Vec<f32>>,
    pub(super) snapshot_counts_before: Vec<usize>,
}

/// Which site a batch record belongs to, and under which control.
#[derive(Clone, Copy)]
pub(super) struct BatchSiteContext {
    pub(super) layer: usize,
    pub(super) site: HcSite,
    pub(super) mutation: Mutation,
    /// Carried onto the site's [`CarrierWritePlane`]; the executor
    /// applies it after the FFN site returns.
    pub(super) layer_scale: Option<f32>,
    /// The provider's position when the traversal began: batch row `i`
    /// is absolute position `base + i` (RESIDUAL-BUS-2 A2).
    pub(super) base: usize,
}

/// Which site of which layer is being entered, and the operands that
/// describe it under each topology that has any.
///
/// One value rather than four parameters because they are one fact: a
/// site is identified by its layer and its half of the layer, and the
/// two operand slots are the SAME site seen by two topologies, at most
/// one of which a component declares. Passing them separately let a
/// caller name one site in `which` and hand over another's operands.
pub(super) struct BatchSite<'p> {
    pub(super) layer: usize,
    pub(super) which: HcSite,
    pub(super) hyper_connection: Option<&'p PreparedHcSite>,
    pub(super) attention_residual: Option<&'p PreparedAttnResSite>,
}

/// Enter one site for every position. Stages one to three run per
/// position, in parallel; the carrier, the layer's site operands and the
/// topology must agree three ways, which preparation guarantees.
pub(super) fn enter_batch_site<'a>(
    h: &'a Plane,
    at: BatchSite<'_>,
    topology: Option<HyperConnection>,
    norm_eps: f64,
    mutation: Mutation,
) -> Result<BatchSiteEntry<'a>, VindexError> {
    let BatchSite {
        layer,
        which,
        hyper_connection: site,
        attention_residual: attn_res,
    } = at;
    // The attention-residual arm first: each position reduces from its
    // OWN history, in parallel, and the ordinary operator downstream
    // still runs once over the resulting `[positions, hidden]`. That
    // split is the invariant — batching vectorises the branch, never the
    // state.
    if let Plane::Histories(histories) = h {
        let Some(site) = attn_res else {
            return Err(VindexError::Parse(format!(
                "layer {layer} carries a residual-history plane and no attention-residual \
                 sites; preparation should have refused the image"
            )));
        };
        let guarded = match which {
            HcSite::Attention => {
                histories.first().is_some_and(|h| h.snapshot_count() > 0)
                    || mutation == Mutation::AttnResLayer0AttentionSiteRuns
            }
            HcSite::Ffn => match mutation {
                Mutation::AttnResMlpSiteGuardedOnNonEmpty => {
                    histories.first().is_some_and(|h| h.snapshot_count() > 0)
                }
                Mutation::AttnResMlpSiteSkippedAtLayer0 if layer == 0 => false,
                _ => true,
            },
        };
        if !guarded {
            let prefixes: Result<Vec<Vec<f32>>, VindexError> = histories
                .iter()
                .map(|history| {
                    history.prefix().map(<[f32]>::to_vec).ok_or_else(|| {
                        VindexError::Parse(format!(
                            "layer {layer} entered a site with no prefix at some position"
                        ))
                    })
                })
                .collect();
            return Ok(BatchSiteEntry {
                branch_inputs: Cow::Owned(prefixes?),
                reductions: None,
                attn_res: None,
            });
        }
        // Captured BEFORE the reduction and before this layer's boundary
        // event, so the record describes the state the site was entered
        // on rather than the state it left behind.
        let prefixes_before: Vec<Vec<f32>> = histories
            .iter()
            .map(|history| history.prefix().unwrap_or_default().to_vec())
            .collect();
        let snapshot_counts_before: Vec<usize> =
            histories.iter().map(History::snapshot_count).collect();
        let pair = site.pair();
        let mut reductions: Vec<attention_residual::Reduction> = histories
            .par_iter()
            .map(|history| attention_residual::reduce(history, pair, norm_eps, mutation))
            .collect::<Result<_, _>>()?;
        if mutation == Mutation::AttnResHistoryFromPositionZero {
            // The BROADCAST control: position 0's reduction applied to
            // every row — what a traversal that built one shared history
            // would produce. Invisible at batch size one, which is why
            // the witness runs three distinguishable positions.
            if let Some(first) = reductions.first().cloned() {
                for reduction in reductions.iter_mut() {
                    *reduction = first.clone();
                }
            }
        }
        let branch_inputs = reductions.iter().map(|r| r.mixed.clone()).collect();
        return Ok(BatchSiteEntry {
            branch_inputs: Cow::Owned(branch_inputs),
            reductions: None,
            attn_res: Some(AttnResBatchEntry {
                reductions,
                prefixes_before,
                snapshot_counts_before,
            }),
        });
    }
    match (h, site, topology) {
        (Plane::Rows(rows), None, None) => Ok(BatchSiteEntry {
            branch_inputs: Cow::Borrowed(rows),
            reductions: None,
            attn_res: None,
        }),
        (Plane::Bundles(bundles), Some(site), Some(hc)) => {
            if mutation == Mutation::BypassComposition {
                // The control: stream 0 stands in for every bundle and
                // no split exists to report.
                return Ok(BatchSiteEntry {
                    branch_inputs: Cow::Owned(
                        bundles.iter().map(|x| x.stream(0).to_vec()).collect(),
                    ),
                    reductions: None,
                    attn_res: None,
                });
            }
            let weights = site.weights();
            let mut reductions: Vec<SiteReduction> = bundles
                .par_iter()
                .map(|x| hyper_connection::reduce(x, &weights, hc, norm_eps, mutation))
                .collect();
            if mutation == Mutation::SplitFromPositionZero {
                // The control: one position's state applied to every row
                // — invisible at batch size one, which is why the witness
                // runs three distinguishable positions.
                if let Some(first) = reductions.first().cloned() {
                    for reduction in reductions.iter_mut() {
                        *reduction = first.clone();
                    }
                }
            }
            let branch_inputs = reductions.iter().map(|r| r.reduced.clone()).collect();
            Ok(BatchSiteEntry {
                branch_inputs: Cow::Owned(branch_inputs),
                reductions: Some(reductions),
                attn_res: None,
            })
        }
        _ => Err(VindexError::Parse(
            "the residual plane, the layer's hyper-connection sites and the declared topology \
             disagree; preparation should have refused the image"
                .to_string(),
        )),
    }
}

/// Name one transition at every position of the batch, in position order,
/// at absolute positions `base..base + positions` (RESIDUAL-BUS-2 A1, A2).
pub(super) fn each_position(
    sink: &mut dyn FnMut(PlaneEvent) -> Result<(), VindexError>,
    layer: usize,
    base: usize,
    form: CarrierForm,
    positions: usize,
    transition: CarrierTransition,
) -> Result<(), VindexError> {
    for row in 0..positions {
        sink(PlaneEvent::Transition {
            address: CarrierAddress::of(base + row, layer, form, &transition),
            transition,
        })?;
    }
    Ok(())
}

/// The block-boundary event across every position — the batch form of
/// the decode traversal's third contract point.
///
/// Offered at each of the three points a snapshot could be taken; does
/// something only at the point this run's control selects, the
/// reference's being [`BoundaryPhase::AfterAttentionReduce`]. The prefix
/// RESET always happens at the reference's point whatever the snapshot's
/// phase, because the reference resets in the same statement it appends.
///
/// **Every position appends its OWN entering state.** A traversal that
/// appended one position's vector to every history would produce a
/// plausible plane and a wrong model; that is what the broadcast control
/// exists to catch, and why the values are recorded per position here
/// rather than as one shared vector.
#[allow(clippy::too_many_arguments)]
pub(super) fn batch_boundary_event(
    h: &mut Plane,
    phase: BoundaryPhase,
    entering_prefixes: &[Vec<f32>],
    mixed: &[Vec<f32>],
    layer: usize,
    base: usize,
    mutation: Mutation,
    sink: &mut dyn FnMut(PlaneEvent) -> Result<(), VindexError>,
) -> Result<(), VindexError> {
    let Plane::Histories(histories) = h else {
        return Ok(());
    };
    let snapshot_phase = match mutation {
        Mutation::AttnResSiteOverNewSnapshots => BoundaryPhase::BeforeAttentionReduce,
        Mutation::AttnResSnapshotAfterAttention => BoundaryPhase::AfterAttentionBranch,
        _ => BoundaryPhase::AfterAttentionReduce,
    };
    if phase == snapshot_phase {
        let before = histories.first().map_or(0, |h| h.snapshot_count());
        let mut values: Vec<Vec<f32>> = Vec::with_capacity(histories.len());
        for (position, history) in histories.iter_mut().enumerate() {
            let entering = entering_prefixes
                .get(position)
                .cloned()
                .unwrap_or_else(|| history.prefix().unwrap_or_default().to_vec());
            let value = match (mutation, phase) {
                (Mutation::AttnResSnapshotIsMixedVector, _) => {
                    mixed.get(position).cloned().unwrap_or(entering)
                }
                (_, BoundaryPhase::AfterAttentionBranch) => {
                    history.prefix().map(<[f32]>::to_vec).unwrap_or(entering)
                }
                _ => entering,
            };
            history.push_snapshot(value.clone());
            values.push(value);
        }
        each_position(
            sink,
            layer,
            base,
            CarrierForm::History,
            histories.len(),
            CarrierTransition::HistorySnapshot,
        )?;
        let after = histories.first().map_or(0, |h| h.snapshot_count());
        sink(PlaneEvent::AttentionResidualBoundary(
            AttnResBoundaryPlane {
                layer,
                snapshots_before: before,
                snapshots_after: after,
                values: &values,
                entering_prefixes,
            },
        ))?;
    }
    if phase == BoundaryPhase::AfterAttentionReduce {
        for history in histories.iter_mut() {
            history.reset_prefix();
        }
        each_position(
            sink,
            layer,
            base,
            CarrierForm::History,
            histories.len(),
            CarrierTransition::HistoryReset,
        )?;
    }
    Ok(())
}

/// Leave one site for every position: fold each position's `[hidden]`
/// delta back into the carrier. On rows that is the residual add; on
/// bundles it is stage five per position — one operation, not an add —
/// and the sink sees every position's state the moment it exists.
pub(super) fn leave_batch_site<B: PlanBackend + ?Sized>(
    backend: &B,
    h: &mut Plane,
    deltas: Vec<Vec<f32>>,
    reductions: Option<Vec<SiteReduction>>,
    attn_res: Option<AttnResBatchEntry>,
    context: BatchSiteContext,
    sink: &mut dyn FnMut(PlaneEvent) -> Result<(), VindexError>,
) -> Result<(), VindexError> {
    // Each branch row updates ONLY its originating history. The delta at
    // index i is the branch's output for position i, and it goes into
    // position i's own state — never into a shared one, and never into
    // another position's.
    if let Plane::Histories(histories) = &mut *h {
        if context.mutation == Mutation::AttnResSwapPositionHistories && histories.len() >= 2 {
            // The SWAP control: positions 0 and 1 exchange histories
            // between the reduction and the update, so each write lands
            // in the other position's state. Catches loss of positional
            // identity, which the broadcast control cannot see.
            histories.swap(0, 1);
        }
        // The WRITE-OFFSET control: send each branch row into the NEXT
        // position's history, dropping the last. Distinct from the swap,
        // which is an involution on two rows and leaves the multiset of
        // (history, delta) pairings intact at every other position; this
        // one misaligns the whole plane by one and leaves position 0
        // unwritten, which is the shape an off-by-one in a batched write
        // actually takes.
        let deltas: Vec<Vec<f32>> =
            if context.mutation == Mutation::AttnResWriteOffsetByOne && histories.len() >= 2 {
                let mut shifted = vec![vec![0.0; deltas.first().map_or(0, Vec::len)]];
                shifted.extend(deltas.iter().take(deltas.len() - 1).cloned());
                shifted
            } else {
                deltas
            };
        for (position, (history, delta)) in histories.iter_mut().zip(&deltas).enumerate() {
            let mode = if history.prefix().is_some() {
                HistoryWriteMode::Add
            } else {
                HistoryWriteMode::Replace
            };
            history.write(delta);
            let transition = CarrierTransition::HistoryWrite {
                site: context.site,
                mode,
            };
            sink(PlaneEvent::Transition {
                address: CarrierAddress::of(
                    context.base + position,
                    context.layer,
                    CarrierForm::History,
                    &transition,
                ),
                transition,
            })?;
        }
        if let Some(entry) = attn_res {
            sink(PlaneEvent::AttentionResidualSite(AttnResSitePlane {
                layer: context.layer,
                site: context.site,
                reductions: &entry.reductions,
                prefixes_before: &entry.prefixes_before,
                snapshot_counts_before: &entry.snapshot_counts_before,
                branch_outputs: &deltas,
                histories_out: histories,
            }))?;
        }
        return Ok(());
    }
    match (h, reductions) {
        (Plane::Rows(rows), None) => {
            rows.par_iter_mut()
                .zip(deltas.par_iter())
                .for_each(|(row, delta)| backend.residual_add(row, delta));
            each_position(
                sink,
                context.layer,
                context.base,
                CarrierForm::Single,
                rows.len(),
                CarrierTransition::Add { site: context.site },
            )?;
            sink(PlaneEvent::CarrierWrite(CarrierWritePlane {
                layer: context.layer,
                site: context.site,
                base: context.base,
                deltas: &deltas,
                after: rows,
                layer_scale: context.layer_scale,
            }))
        }
        (Plane::Bundles(bundles), Some(reductions)) => {
            if context.mutation == Mutation::SwapPositionsBeforeUpdate && bundles.len() >= 2 {
                // The control: positions 0 and 1 exchange their bundles
                // between the reduction and the update, so the update
                // carries the wrong position's state forward.
                bundles.swap(0, 1);
            }
            let (splits, reduced): (Vec<SinkhornSplit>, Vec<Vec<f32>>) =
                reductions.into_iter().map(|r| (r.split, r.reduced)).unzip();
            let next: Vec<Bundle> = bundles
                .par_iter()
                .zip(deltas.par_iter())
                .zip(splits.par_iter())
                .map(|((x, delta), split)| {
                    hyper_connection::update(x, delta, split, context.mutation)
                })
                .collect();
            *bundles = next;
            each_position(
                sink,
                context.layer,
                context.base,
                CarrierForm::Bundle,
                bundles.len(),
                CarrierTransition::HcUpdate { site: context.site },
            )?;
            sink(PlaneEvent::HyperConnectionSite(HcSitePlane {
                layer: context.layer,
                site: context.site,
                splits: &splits,
                reduced: &reduced,
                branch_outputs: &deltas,
                bundles_out: bundles,
            }))
        }
        (Plane::Bundles(bundles), None) => {
            // The bypass control: the delta lands in stream 0 alone and
            // no record is emitted, exactly what a traversal that never
            // ran the topology would do.
            debug_assert_eq!(context.mutation, Mutation::BypassComposition);
            bundles
                .par_iter_mut()
                .zip(deltas.par_iter())
                .for_each(|(x, delta)| backend.residual_add(x.stream_mut(0), delta));
            each_position(
                sink,
                context.layer,
                context.base,
                CarrierForm::Bundle,
                bundles.len(),
                CarrierTransition::Add { site: context.site },
            )
        }
        (Plane::Rows(_), Some(_)) => Err(VindexError::Parse(
            "a row plane received site reductions; preparation should have refused the image"
                .to_string(),
        )),
        // A history plane returns above; reaching here means its entry
        // was lost on the way in.
        (Plane::Histories(_), _) => Err(VindexError::Parse(
            "an attention-residual plane left a site through the bundle path; the traversal \
             handles it before this match"
                .to_string(),
        )),
    }
}

/// Whether the FFN runs as ONE multi-position call (CPU-7C2) or as the
/// pre-C2 parallel loop over positions.
///
/// A CPU-7C2 arm switch, and it exists so arm B and arms C/E/D live in one
/// binary. The alternative — comparing against CPU-7C1's banked
/// `B/(2A) = 1.422` — would anchor a gate on a number measured in another
/// run, on another build, which is exactly the defect that cost CPU-5 its
/// Bank 2.
///
/// Default ON: the multi-position shape is the one that respects this
/// module's own ownership rule, and a default that did not would make
/// every other measurement in the repo a measurement of the defect.
/// Only `0` and `off` select the legacy shape.
pub const MULTI_POSITION_FFN_ENV: &str = "LARQL_FFN_MULTI_POSITION";

pub(super) static FFN_SHAPE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

pub fn multi_position_ffn() -> bool {
    match FFN_SHAPE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let on = !matches!(
                std::env::var(MULTI_POSITION_FFN_ENV)
                    .ok()
                    .as_deref()
                    .map(str::trim),
                Some("0") | Some("off")
            );
            FFN_SHAPE.store(if on { 2 } else { 1 }, std::sync::atomic::Ordering::Relaxed);
            on
        }
    }
}

/// Select the FFN shape explicitly, for a harness running both arms in one
/// process. Not for production code, for the reason [`multi_position_ffn`]
/// gives.
pub fn set_multi_position_ffn(on: bool) {
    FFN_SHAPE.store(if on { 2 } else { 1 }, std::sync::atomic::Ordering::Relaxed);
}

/// One layer's attention operands, loaded once in the backend's
/// declared format. Owned by whichever traversal is running — the batch
/// path loads them per forward, the decode session keeps them for its
/// lifetime — so both paths resolve operands through exactly one place.
/// An attention layer's projection biases. Q/K/V always travel together;
/// the output bias is present under `attention_bias` and absent under
/// `qkv_bias`, whose output projection is unbiased.
pub(in super::super) struct AttentionBiases {
    pub(super) q: Vec<f32>,
    pub(super) k: Vec<f32>,
    pub(super) v: Vec<f32>,
    pub(super) o: Option<Vec<f32>>,
}

pub(in super::super) struct AttentionOperands {
    pub(super) w_q: LoadedWeight,
    pub(super) w_k: LoadedWeight,
    pub(super) w_v: LoadedWeight,
    pub(super) w_o: LoadedWeight,
    pub(super) qk_weights: Option<(Vec<f32>, Vec<f32>)>,
    pub(super) gate: Option<LoadedWeight>,
    /// Projection biases, f32 (elementwise glue, not matrix traffic).
    pub(super) biases: Option<AttentionBiases>,
    /// Sink logits, f32.
    pub(super) sinks: Option<Vec<f32>>,
    /// The run's RoPE construction (PREC-DIAG-1 E0b), carried into every
    /// call; set through `PreparedOperands::set_rope_angles`.
    pub(super) rope_angles: super::rope_angles::RopeAngles,
}
