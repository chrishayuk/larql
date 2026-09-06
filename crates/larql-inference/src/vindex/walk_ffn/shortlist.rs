//! Two-stage shortlist-then-rerank selection (2026-07-30 review, item
//! 19; ROADMAP "Vindex + WalkFFN review" item 19).
//!
//! The joint selectors' rerank criteria have existed since the K=200
//! selection experiments (`FeatureSelector`, `selector.rs`), but every
//! joint call paid a full O(N·d) `gate_scores_batch` projection — and
//! the up-score criteria a second full up projection — to score all N
//! features before keeping K. This module adds the missing *shortlist*
//! structure:
//!
//! - **Stage 1** — top-M candidates by cheap gate score, through the
//!   production `gate_walk` → `gate_knn_q4` → `gate_knn` chain
//!   ([`WalkFfn::production_gate_chain`]) — the one projection the
//!   production walk already pays. No new projection code.
//! - **Stage 2** — the configured criterion evaluated for ONLY those M
//!   candidates: per-candidate up dots via the per-row `ffn_row_dot`
//!   primitive and row norms from the selector's lazy caches — O(M·d),
//!   never a full projection — then a full rerank to the final top-K.
//!
//! The audit's exact formula: shortlist by gate, rerank by
//! `|φ(g·x)(u·x)|·‖d‖`. The criterion formulas themselves are
//! single-sourced in [`criterion_weight`], shared with the
//! full-projection `joint_gate_knn` so the two paths cannot drift.
//!
//! Hits keep the `(feat_idx, raw_gate_score)` contract, ordered by the
//! FINAL rerank (descending criterion weight, feature index on ties) —
//! the runtime trace's `rank` field therefore reports rerank order.
//! A layer where the shortlist cannot run (M < K, `Random` selector,
//! or missing stage-2 inputs) declines to single-stage selection
//! observably: a [`PATH_SHORTLIST_DECLINED`] dispatch-trace entry plus
//! [`WalkFfn::shortlist_decline_count`] — the `selector:fallback`
//! precedent (M10), never silence.

use ndarray::Array1;

use super::helpers::selection_weight_cmp_desc;
use super::WalkFfn;
use crate::vindex::walk_config::FeatureSelector;

/// Dispatch-trace label emitted when a configured shortlist declines
/// to single-stage selection (M < K, unscored selector, or stage-2
/// inputs unavailable). Counted by [`WalkFfn::shortlist_decline_count`].
pub(super) const PATH_SHORTLIST_DECLINED: &str = "shortlist:declined";

/// `GateIndex::ffn_row_dot` component index for the up projection
/// (the trait convention: 0 = gate, 1 = up, 2 = down).
const FFN_COMPONENT_UP: usize = 1;

/// Which per-feature inputs a rerank criterion consumes beyond the
/// gate score stage 1 already computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CriterionInputs {
    /// `u·x` — prompt-conditional up score (per-row dot in stage 2).
    pub(super) up_score: bool,
    /// `‖up_row‖` — static, from the lazy per-layer norm cache.
    pub(super) up_norm: bool,
    /// `‖down_row‖` — static, from the lazy per-layer norm cache.
    pub(super) down_norm: bool,
}

/// The inputs `kind` needs, or `None` for [`FeatureSelector::Random`]
/// — the control selector has no rerank criterion, so two-stage
/// selection cannot apply to it.
pub(super) fn criterion_inputs(kind: FeatureSelector) -> Option<CriterionInputs> {
    match kind {
        FeatureSelector::GateOnly => Some(CriterionInputs {
            up_score: false,
            up_norm: false,
            down_norm: false,
        }),
        FeatureSelector::GateXDownNorm => Some(CriterionInputs {
            up_score: false,
            up_norm: false,
            down_norm: true,
        }),
        FeatureSelector::GateXUpDownNorm => Some(CriterionInputs {
            up_score: false,
            up_norm: true,
            down_norm: true,
        }),
        FeatureSelector::GateXUpScore => Some(CriterionInputs {
            up_score: true,
            up_norm: false,
            down_norm: false,
        }),
        FeatureSelector::ActXUpScoreXDownNorm => Some(CriterionInputs {
            up_score: true,
            up_norm: false,
            down_norm: true,
        }),
        FeatureSelector::Random => None,
    }
}

/// The per-feature selection weight for a scored criterion — THE
/// single source of the [`FeatureSelector`] formulas, shared by the
/// full-projection `joint_gate_knn` and the two-stage rerank so the
/// two paths rank by identical math. Inputs the criterion does not
/// consume (per [`criterion_inputs`]) are ignored.
///
/// Panics on [`FeatureSelector::Random`]: it has no weight formula —
/// callers must gate on `criterion_inputs(kind).is_some()` first.
pub(super) fn criterion_weight(
    kind: FeatureSelector,
    use_gelu: bool,
    gate: f32,
    up_score: f32,
    up_norm: f32,
    down_norm: f32,
) -> f32 {
    match kind {
        FeatureSelector::GateOnly => gate.abs(),
        FeatureSelector::GateXDownNorm => gate.abs() * down_norm,
        FeatureSelector::GateXUpDownNorm => gate.abs() * down_norm * up_norm,
        FeatureSelector::GateXUpScore => gate.abs() * up_score.abs(),
        FeatureSelector::ActXUpScoreXDownNorm => {
            let activated = if use_gelu {
                crate::ffn::gelu_tanh(gate)
            } else {
                gate * crate::ffn::sigmoid(gate)
            };
            (activated * up_score).abs() * down_norm
        }
        FeatureSelector::Random => {
            unreachable!("Random has no rerank criterion (criterion_inputs returns None)")
        }
    }
}

/// Deterministic final selection order: criterion weight descending
/// (NaN panics via [`selection_weight_cmp_desc`]), feature index
/// ascending on exact weight ties. Shared by `joint_gate_knn`'s final
/// top-K and the stage-2 rerank so both report the same rank order.
pub(super) fn rerank_cmp(a: &(usize, f32, f32), b: &(usize, f32, f32)) -> std::cmp::Ordering {
    selection_weight_cmp_desc(a.2, b.2).then_with(|| a.0.cmp(&b.0))
}

impl<'a> WalkFfn<'a> {
    /// Whether the selection criteria activate with GELU (vs SiLU) —
    /// the same arch check the sparse walk applies to its activations.
    pub(super) fn selector_use_gelu(&self) -> bool {
        self.weights.arch.gate_up_is_gelu_tanh()
    }

    /// The production gate-selection chain — exact `gate_walk` first,
    /// then backend `gate_knn_q4`, then `gate_knn` (see the HNSW note
    /// in `sparse_route.rs`). Single source for the `GateOnly` route,
    /// the joint selectors' fallback, and the stage-1 shortlist.
    pub(super) fn production_gate_chain(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        self.index
            .gate_walk(layer, residual, top_k)
            .or_else(|| {
                self.backend
                    .and_then(|be| self.index.gate_knn_q4(layer, residual, top_k, be))
            })
            .unwrap_or_else(|| self.index.gate_knn(layer, residual, top_k))
    }

    /// Two-stage selection: stage-1 shortlist (top-M by gate, the
    /// production chain), stage-2 rerank (criterion for the M
    /// candidates only, O(M·d)) — see the module doc. Declines to
    /// [`WalkFfn::single_stage_hits`] with the observable
    /// [`PATH_SHORTLIST_DECLINED`] signal when it cannot run.
    pub(super) fn shortlist_rerank(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        x_slice: &[f32],
        top_k: usize,
        shortlist_m: usize,
        kind: FeatureSelector,
    ) -> Vec<(usize, f32)> {
        match self.try_shortlist_rerank(layer, residual, x_slice, top_k, shortlist_m, kind) {
            Some(hits) => hits,
            None => {
                // Observable decline — the selector:fallback precedent
                // (M10): a configured two-stage selection silently
                // running single-stage would falsify the cost model
                // the config claims.
                self.trace_path(layer, PATH_SHORTLIST_DECLINED);
                self.shortlist_declines
                    .set(self.shortlist_declines.get() + 1);
                self.single_stage_hits(layer, residual, top_k, kind)
            }
        }
    }

    /// The fallible two-stage body. `None` = decline (M < K, unscored
    /// selector, or a stage-2 input unavailable); the wrapper handles
    /// the signal + single-stage fallback.
    fn try_shortlist_rerank(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        x_slice: &[f32],
        top_k: usize,
        shortlist_m: usize,
        kind: FeatureSelector,
    ) -> Option<Vec<(usize, f32)>> {
        let needs = criterion_inputs(kind)?;
        // M must cover K: reranking M < K candidates could never fill
        // the requested top-K honestly.
        if shortlist_m < top_k {
            return None;
        }

        // ── Stage 1 — top-M by cheap gate score (the ONE projection). ──
        let candidates = self.production_gate_chain(layer, residual, shortlist_m);
        if candidates.is_empty() {
            return Some(Vec::new());
        }

        // ── Stage 2 — criterion for the M candidates only, O(M·d). ──
        let down_norms = if needs.down_norm {
            Some(self.down_row_norms(layer)?)
        } else {
            None
        };
        let up_norms = if needs.up_norm {
            Some(self.up_row_norms(layer)?)
        } else {
            None
        };
        let use_gelu = self.selector_use_gelu();
        let norm_for = |norms: &Option<std::sync::Arc<Vec<f32>>>, feat: usize| {
            norms
                .as_ref()
                .map(|n| n.get(feat).copied().unwrap_or(0.0))
                .unwrap_or(0.0)
        };
        let mut weighted: Vec<(usize, f32, f32)> = Vec::with_capacity(candidates.len());
        for (feat, gate) in candidates {
            let up_score = if needs.up_score {
                // Per-row fused dot — O(d) per candidate, dispatched
                // FP4 → native → Q4K by the unified row access.
                self.index
                    .ffn_row_dot(layer, FFN_COMPONENT_UP, feat, x_slice)?
            } else {
                0.0
            };
            let weight = criterion_weight(
                kind,
                use_gelu,
                gate,
                up_score,
                norm_for(&up_norms, feat),
                norm_for(&down_norms, feat),
            );
            weighted.push((feat, gate, weight));
        }

        // Final rerank — full deterministic order (the trace's rank
        // field reports this order).
        weighted.sort_unstable_by(rerank_cmp);
        weighted.truncate(top_k.min(weighted.len()));
        Some(weighted.into_iter().map(|(f, g, _)| (f, g)).collect())
    }
}
