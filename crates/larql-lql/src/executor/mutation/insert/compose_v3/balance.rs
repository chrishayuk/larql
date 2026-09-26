//! Phase 3 of the V3 compose install: V2's post-install adjustment
//! passes (`insert/balance.rs`), control flow ported literally onto the
//! operand-source seam.
//!
//! The probe is the only substrate difference: where V2 re-runs the
//! canonical prompt through its walk-FFN forward over the patch
//! overlay, V3 runs the composed program — the overlay's operand
//! edits derived fresh each probe, since balance rescales the down
//! column between iterations. Every decision rule, constant, and the
//! snapshot/rollback contract is V2's, unchanged:
//!
//! - amplify (×UP_SCALE) below PROB_FLOOR, shrink (×DOWN_SCALE) above
//!   PROB_CEILING, stop inside the band;
//! - snapshot the best-probability down state ONLY while amplifying,
//!   roll back to it on saturation (MAX_STALE stale iterations);
//! - cross-fact: replay up to MAX_PRIORS_CHECKED prior installs'
//!   canonical prompts; while any prior's target sits below
//!   PRIOR_FLOOR, shrink THIS install ×0.7, capped at CROSS_ITERS.

use crate::error::LqlError;
use crate::executor::helpers::{target_prefix, TARGET_PREFIX_CHARS};
use crate::executor::tuning::{
    canonical_prompt, BALANCE_ITERS, BALANCE_PROBE_TOP_K, CROSS_ITERS, DOWN_SCALE,
    MAX_PRIORS_CHECKED, MAX_STALE, PRIOR_FLOOR, PROB_CEILING, PROB_FLOOR, UP_SCALE,
};
use crate::executor::vindex3::{
    compose_overrides, encode_v3_prompt, top_k_probs, v3_continuation, V3Runtime,
};
use crate::executor::{Backend, Session};
use larql_vindex::format::vindex3::knowledge::KnowledgeOverlay;
use larql_vindex::tokenizers::Tokenizer;

/// The probe: the target token's probability on `prompt_ids` through
/// the COMPOSED program — V2's statistic (top BALANCE_PROBE_TOP_K
/// decoded predictions, matched by containment or 3-char prefix).
pub(crate) fn probe_target_prob(
    runtime: &V3Runtime,
    tokenizer: &Tokenizer,
    overlay: &KnowledgeOverlay,
    prompt_ids: &[u32],
    target: &str,
) -> Result<f64, LqlError> {
    let continuation = v3_continuation(runtime)?;
    let output = match compose_overrides(runtime, overlay)? {
        Some(overrides) => {
            runtime
                .execute_streaming_overlaid(prompt_ids, &overrides, &continuation, &mut |_| Ok(()))
        }
        None => runtime.execute_streaming(prompt_ids, &continuation, &mut |_| Ok(())),
    }
    .map_err(|e| LqlError::exec("balance: v3 probe failed", e))?;
    let logits = output
        .logits
        .ok_or_else(|| LqlError::Execution("balance: no output head".into()))?;

    let prefix = target_prefix(target, TARGET_PREFIX_CHARS);
    Ok(top_k_probs(&logits, BALANCE_PROBE_TOP_K)
        .into_iter()
        .find(|(id, _)| {
            let tok = tokenizer.decode(&[*id], false).unwrap_or_default();
            tok.contains(target) || tok.starts_with(prefix)
        })
        .map(|(_, prob)| prob as f64)
        .unwrap_or(0.0))
}

impl Session {
    /// V2's `balance_installed`, single-slot form (the V3 install is
    /// single-layer): greedy amplify/shrink on the slot's down column
    /// until the canonical-prompt target probability lands in
    /// [PROB_FLOOR, PROB_CEILING], with amplify-only snapshot/rollback.
    pub(super) fn balance_installed_v3(
        &mut self,
        layer: usize,
        feature: usize,
        entity: &str,
        relation: &str,
        target: &str,
    ) -> Result<(), LqlError> {
        let prompt = canonical_prompt(relation, entity);

        let mut best_prob: f64 = 0.0;
        let mut best_down: Option<Vec<f32>> = None;
        let mut stale_iters = 0usize;

        let bos = self.v3_bos_token();
        for _iter in 0..BALANCE_ITERS {
            let target_prob = {
                let Backend::Vindex3 {
                    runtime,
                    tokenizer,
                    overlay,
                    ..
                } = &self.backend
                else {
                    unreachable!("caller matched the backend");
                };
                let tokenizer = tokenizer.as_ref().expect("compose install had a tokenizer");
                let prompt_ids = encode_v3_prompt(tokenizer, prompt.as_str(), bos)?;
                probe_target_prob(runtime, tokenizer, overlay, &prompt_ids, target)?
            };

            if (PROB_FLOOR..=PROB_CEILING).contains(&target_prob) {
                best_down = None;
                break;
            }

            let amplify_mode = target_prob < PROB_FLOOR;
            if amplify_mode {
                if target_prob > best_prob {
                    best_prob = target_prob;
                    best_down = self.overlay_down_at(layer, feature);
                    stale_iters = 0;
                } else {
                    stale_iters += 1;
                }
                if stale_iters >= MAX_STALE {
                    break;
                }
            }

            let scale: f32 = if amplify_mode { UP_SCALE } else { DOWN_SCALE };
            self.scale_overlay_down(layer, feature, scale);
        }

        if let Some(best) = best_down {
            let Backend::Vindex3 { overlay, .. } = &mut self.backend else {
                unreachable!("caller matched the backend");
            };
            overlay.set_down_vector(layer, feature, best);
        }
        Ok(())
    }

    /// V2's `cross_fact_regression_check`: while any prior install's
    /// canonical target sits below PRIOR_FLOOR, shrink THIS install's
    /// down column ×0.7, capped at CROSS_ITERS.
    pub(super) fn cross_fact_regression_check_v3(
        &mut self,
        layer: usize,
        feature: usize,
    ) -> Result<(), LqlError> {
        let bos = self.v3_bos_token();
        if self.installed_edges.is_empty() {
            return Ok(());
        }

        for _iter in 0..CROSS_ITERS {
            let any_regressed = {
                let Backend::Vindex3 {
                    runtime,
                    tokenizer,
                    overlay,
                    ..
                } = &self.backend
                else {
                    unreachable!("caller matched the backend");
                };
                let tokenizer = tokenizer.as_ref().expect("compose install had a tokenizer");
                let priors: Vec<_> = self
                    .installed_edges
                    .iter()
                    .rev()
                    .take(MAX_PRIORS_CHECKED)
                    .collect();
                let mut regressed = false;
                for fact in priors {
                    let fact_ids =
                        encode_v3_prompt(tokenizer, fact.canonical_prompt.as_str(), bos)?;
                    let p =
                        probe_target_prob(runtime, tokenizer, overlay, &fact_ids, &fact.target)?;
                    if p < PRIOR_FLOOR {
                        regressed = true;
                        break;
                    }
                }
                regressed
            };
            if !any_regressed {
                break;
            }
            // V2's literal shrink factor for the cross pass.
            self.scale_overlay_down(layer, feature, 0.7_f32);
        }
        Ok(())
    }

    pub(crate) fn overlay_down_at(&self, layer: usize, feature: usize) -> Option<Vec<f32>> {
        let Backend::Vindex3 { overlay, .. } = &self.backend else {
            unreachable!("caller matched the backend");
        };
        overlay
            .down_override_at(layer, feature)
            .map(<[f32]>::to_vec)
    }

    pub(crate) fn scale_overlay_down(&mut self, layer: usize, feature: usize, scale: f32) {
        let Backend::Vindex3 { overlay, .. } = &mut self.backend else {
            unreachable!("caller matched the backend");
        };
        if let Some(down) = overlay.down_override_at(layer, feature) {
            let scaled: Vec<f32> = down.iter().map(|v| v * scale).collect();
            overlay.set_down_vector(layer, feature, scaled);
        }
    }
}
