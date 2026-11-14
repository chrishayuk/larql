//! Phase 3 of `INSERT INTO EDGES` (Compose mode): post-install
//! adjustment passes.
//!
//!   - `balance_installed`: greedy per-fact loop that scales each
//!     installed down_col to land the target token at a reasonable
//!     probability on the canonical prompt (PROB_FLOOR..PROB_CEILING).
//!     Rolls back to the best snapshot if amplification saturates.
//!
//!   - `cross_fact_regression_check`: after local balance, verify the
//!     newly-strengthened down_col hasn't hijacked any prior install's
//!     template-matched prompt. Shrinks THIS install × 0.7 per pass
//!     until priors recover, capped at CROSS_ITERS.

use crate::error::LqlError;
use crate::executor::helpers::target_token_matches;
use crate::executor::tuning::{
    canonical_prompt, BALANCE_ITERS, BALANCE_PROBE_TOP_K, CROSS_ITERS, DOWN_SCALE,
    MAX_PRIORS_CHECKED, MAX_STALE, PRIOR_FLOOR, PROB_CEILING, PROB_FLOOR, UP_SCALE,
};
use crate::executor::Session;

use super::compose::InstalledSlot;

impl Session {
    /// Greedy amplify/shrink on the freshly installed slots until the
    /// target token's canonical-prompt probability lands in
    /// [PROB_FLOOR, PROB_CEILING]. Snapshots and rolls back on amplify
    /// saturation (residual blow-up in late layers).
    ///
    /// No-op when `installed` is empty.
    pub(super) fn balance_installed(
        &mut self,
        installed: &[InstalledSlot],
        entity: &str,
        relation: &str,
        target: &str,
    ) -> Result<(), LqlError> {
        if installed.is_empty() {
            return Ok(());
        }

        // All tuning constants — including the [PROB_FLOOR, PROB_CEILING]
        // band and the per-iter scales — live in `executor::tuning` so
        // every magic number has a single home with its measurement-link
        // comment.

        let (path, config, _patched) = self.require_vindex()?;
        let mut cb = larql_vindex::SilentLoadCallbacks;
        let weights = larql_vindex::load_model_weights(path, &mut cb)
            .map_err(|e| LqlError::exec("balance: load weights", e))?;
        let tokenizer = larql_vindex::load_vindex_tokenizer(path)
            .map_err(|e| LqlError::exec("balance: load tokenizer", e))?;

        let prompt = canonical_prompt(relation, entity);
        let prompt_ids =
            crate::executor::query::encode_vindex_prompt(config, &tokenizer, prompt.as_str())?;

        // Snapshot/restore applies only to the AMPLIFY path: when
        // UP_SCALE saturates (residual blow-up, softmax collapse in
        // late layers), we roll back to the iteration that produced
        // the highest target_prob before regression. DOWN scaling
        // is monotonic — each iter strictly reduces target_prob
        // toward the ceiling — so no snapshot/restore for that case
        // (rolling back "best prob" would undo the correction).
        let mut best_prob: f64 = 0.0;
        let mut best_down: Option<Vec<Vec<f32>>> = None;
        let mut stale_iters = 0usize;

        for _iter in 0..BALANCE_ITERS {
            let (_, _, patched) = self.require_vindex()?;
            let walk_ffn =
                larql_inference::vindex::WalkFfn::new_unlimited_with_trace(&weights, patched);
            let result = larql_inference::predict_with_ffn(
                &weights,
                &tokenizer,
                &prompt_ids,
                BALANCE_PROBE_TOP_K,
                &walk_ffn,
            );

            let target_prob: f64 = result
                .predictions
                .iter()
                .find(|(tok, _)| target_token_matches(tok, target))
                .map(|(_, prob)| *prob)
                .unwrap_or(0.0);

            // Converged inside band — keep current state.
            if (PROB_FLOOR..=PROB_CEILING).contains(&target_prob) {
                best_down = None;
                break;
            }

            let amplify_mode = target_prob < PROB_FLOOR;

            // Snapshot only during amplify — track the best pre-saturation
            // state so we can roll back if UP_SCALE blows up. Don't
            // snapshot during DOWN scaling (a DOWN step's "lower prob"
            // is the improvement, not a regression to roll back from).
            if amplify_mode {
                if target_prob > best_prob {
                    best_prob = target_prob;
                    let snap: Vec<Vec<f32>> = installed
                        .iter()
                        .filter_map(|slot| {
                            let (_, _, p) = self.require_vindex().ok()?;
                            p.down_override_at(slot.layer, slot.feature)
                                .map(|v| v.to_vec())
                        })
                        .collect();
                    best_down = Some(snap);
                    stale_iters = 0;
                } else {
                    stale_iters += 1;
                }
                // Saturation — amplification stopped improving target
                if stale_iters >= MAX_STALE {
                    break;
                }
            }

            let scale: f32 = if amplify_mode { UP_SCALE } else { DOWN_SCALE };

            let (_, _, patched_mut) = self.require_patched_mut()?;
            for slot in installed {
                if let Some(down) = patched_mut.down_override_at(slot.layer, slot.feature) {
                    let scaled: Vec<f32> = down.iter().map(|v| v * scale).collect();
                    patched_mut.set_down_vector(slot.layer, slot.feature, scaled);
                }
            }
        }

        // Roll back to best snapshot only if saturation happened
        // during amplification. Empty best_down means we either
        // converged or were down-scaling — in both cases the
        // current overlay state is correct.
        if let Some(best) = best_down {
            let (_, _, patched_mut) = self.require_patched_mut()?;
            for (slot, down) in installed.iter().zip(best.iter()) {
                patched_mut.set_down_vector(slot.layer, slot.feature, down.clone());
            }
        }

        Ok(())
    }

    /// Check that the newly-installed slots haven't hijacked any prior
    /// install's canonical prompt. If any prior fact's target prob
    /// drops below `PRIOR_FLOOR`, shrink THIS install × 0.7 and retry,
    /// capped at `CROSS_ITERS`. No-op when `installed` is empty or the
    /// session has no prior compose installs.
    pub(super) fn cross_fact_regression_check(
        &mut self,
        installed: &[InstalledSlot],
    ) -> Result<(), LqlError> {
        // Local balance brought THIS fact's target into band on
        // THIS fact's canonical. But the newly-strengthened down
        // vector can have template overlap that hijacks prior
        // installs (observed at N=10: one install's "H" token
        // fired on every "The capital of X is" prompt, overriding
        // native Paris/Berlin/Rome).
        //
        // For each prior install, INFER its canonical and verify
        // its target is still above the retrieval floor. If any
        // prior regressed, shrink THIS install's down_col AND
        // verify OUR own target is still retrievable. Stop if
        // shrinking would drop our own target below the floor
        // (fixed-point: both constraints can't be satisfied;
        // accept the state with best joint coverage).
        //
        // CROSS_ITERS, PRIOR_FLOOR, MAX_PRIORS_CHECKED live in
        // `executor::tuning` — see the regression-pass section of
        // that module for the empirical justification.

        if installed.is_empty() || self.installed_edges.is_empty() {
            return Ok(());
        }

        let (path, config, _patched) = self.require_vindex()?;
        let path = path.to_path_buf();
        let config = config.clone();
        let mut cb = larql_vindex::SilentLoadCallbacks;
        let weights = larql_vindex::load_model_weights(&path, &mut cb)
            .map_err(|e| LqlError::exec("cross-balance: load weights", e))?;
        let tokenizer = larql_vindex::load_vindex_tokenizer(&path)
            .map_err(|e| LqlError::exec("cross-balance: load tokenizer", e))?;

        for _iter in 0..CROSS_ITERS {
            let mut any_regressed = false;
            let priors_to_check: Vec<_> = self
                .installed_edges
                .iter()
                .rev()
                .take(MAX_PRIORS_CHECKED)
                .cloned()
                .collect();
            for fact in &priors_to_check {
                let fact_ids = crate::executor::query::encode_vindex_prompt(
                    &config,
                    &tokenizer,
                    fact.canonical_prompt.as_str(),
                )?;
                let (_, _, patched) = self.require_vindex()?;
                let walk =
                    larql_inference::vindex::WalkFfn::new_unlimited_with_trace(&weights, patched);
                let r = larql_inference::predict_with_ffn(
                    &weights,
                    &tokenizer,
                    &fact_ids,
                    BALANCE_PROBE_TOP_K,
                    &walk,
                );
                let p: f64 = r
                    .predictions
                    .iter()
                    .find(|(tok, _)| target_token_matches(tok, &fact.target))
                    .map(|(_, p)| *p)
                    .unwrap_or(0.0);
                if p < PRIOR_FLOOR {
                    any_regressed = true;
                    break;
                }
            }
            if !any_regressed {
                break;
            }

            let (_, _, patched_mut) = self.require_patched_mut()?;
            for slot in installed {
                if let Some(down) = patched_mut.down_override_at(slot.layer, slot.feature) {
                    let scaled: Vec<f32> = down.iter().map(|v| v * 0.7_f32).collect();
                    patched_mut.set_down_vector(slot.layer, slot.feature, scaled);
                }
            }
        }

        Ok(())
    }
}
