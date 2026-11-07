//! `REBALANCE` — global fixed-point rebalance over compose installs.
//!
//! Per-INSERT balance is greedy: it scales THIS install's down_col
//! to meet THIS fact's canonical probability target. That works for
//! N=1 but breaks at N>5 because later installs hijack template-
//! matched siblings that earlier installs' local balance already
//! accepted.
//!
//! Global rebalance runs a fixed-point loop over every registered
//! compose-mode install:
//!
//!   for iter in 0..max_iters:
//!       for fact in installed_edges:
//!           prob = INFER(fact.canonical); extract target_prob
//!           if prob > ceiling: scale down_col(fact) × 0.85
//!           elif prob < floor:  scale down_col(fact) × 1.15
//!       if no fact was scaled this iter: converged, break
//!
//! Smaller scale factors than per-INSERT (0.85 / 1.15 vs 0.7 / 1.6)
//! to dampen oscillation between competing template-shared facts.

use crate::error::LqlError;
use crate::executor::helpers::target_token_matches;
use crate::executor::tuning::{
    REBALANCE_CEILING_DEFAULT, REBALANCE_DOWN_SCALE, REBALANCE_FLOOR_DEFAULT,
    REBALANCE_MAX_ITERS_DEFAULT, REBALANCE_PROBE_TOP_K, REBALANCE_UP_SCALE,
};
use crate::executor::Session;

impl Session {
    pub(crate) fn exec_rebalance(
        &mut self,
        max_iters: Option<u32>,
        floor: Option<f32>,
        ceiling: Option<f32>,
    ) -> Result<Vec<String>, LqlError> {
        let max_iters = max_iters.unwrap_or(REBALANCE_MAX_ITERS_DEFAULT) as usize;
        let floor = floor.unwrap_or(REBALANCE_FLOOR_DEFAULT) as f64;
        let ceiling = ceiling.unwrap_or(REBALANCE_CEILING_DEFAULT) as f64;

        if self.installed_edges.is_empty() {
            return Ok(vec![
                "Rebalance: no compose-mode installs to rebalance (KNN installs don't need it)"
                    .into(),
            ]);
        }

        if matches!(self.backend, crate::executor::Backend::Vindex3 { .. }) {
            return self.exec_rebalance_v3(max_iters, floor, ceiling);
        }

        let n_facts = self.installed_edges.len();
        let (path, config, _patched) = self.require_vindex()?;
        let path = path.to_path_buf();
        let config = config.clone();
        let mut cb = larql_vindex::SilentLoadCallbacks;
        let weights = larql_vindex::load_model_weights(&path, &mut cb)
            .map_err(|e| LqlError::exec("rebalance: load weights", e))?;
        let tokenizer = larql_vindex::load_vindex_tokenizer(&path)
            .map_err(|e| LqlError::exec("rebalance: load tokenizer", e))?;

        // DOWN_SCALE / UP_SCALE / PROBE_TOP_K live in `executor::tuning`
        // (REBALANCE_DOWN_SCALE / REBALANCE_UP_SCALE / REBALANCE_PROBE_TOP_K).

        let mut iters_run = 0usize;
        let mut final_probs: Vec<f64> = vec![0.0; n_facts];

        for iter in 0..max_iters {
            iters_run = iter + 1;
            let mut any_changed = false;
            let facts_snapshot = self.installed_edges.clone();

            for (i, fact) in facts_snapshot.iter().enumerate() {
                let ids = crate::executor::query::encode_vindex_prompt(
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
                    &ids,
                    REBALANCE_PROBE_TOP_K,
                    &walk,
                );

                let prob: f64 = r
                    .predictions
                    .iter()
                    .find(|(tok, _)| target_token_matches(tok, &fact.target))
                    .map(|(_, p)| *p)
                    .unwrap_or(0.0);
                final_probs[i] = prob;

                let scale: Option<f32> = if prob > ceiling {
                    Some(REBALANCE_DOWN_SCALE)
                } else if prob < floor {
                    Some(REBALANCE_UP_SCALE)
                } else {
                    None
                };

                if let Some(scale) = scale {
                    let (_, _, patched_mut) = self.require_patched_mut()?;
                    if let Some(down) = patched_mut.down_override_at(fact.layer, fact.feature) {
                        let scaled: Vec<f32> = down.iter().map(|v| v * scale).collect();
                        patched_mut.set_down_vector(fact.layer, fact.feature, scaled);
                        any_changed = true;
                    }
                }
            }

            if !any_changed {
                break;
            }
        }

        let slots: Vec<(usize, usize)> = self
            .installed_edges
            .iter()
            .map(|fact| (fact.layer, fact.feature))
            .collect();
        self.refresh_recorded_patch_ops_for_slots(&slots)?;

        // Summary
        let mut in_band = 0usize;
        let mut below = 0usize;
        let mut above = 0usize;
        for &p in &final_probs {
            if p < floor {
                below += 1;
            } else if p > ceiling {
                above += 1;
            } else {
                in_band += 1;
            }
        }
        let mut out = Vec::new();
        out.push(format!(
            "Rebalance: {n_facts} compose installs, {iters_run} iterations",
        ));
        out.push(format!(
            "  band [{floor:.2}, {ceiling:.2}]: {in_band} in band, {below} below (amplifying), {above} above (shrinking)"
        ));
        out.push(format!(
            "  {}",
            if below == 0 && above == 0 {
                "all converged in band"
            } else {
                "saturated (some facts hit oscillation limit — template-competition at this layer)"
            }
        ));
        Ok(out)
    }
}

impl Session {
    /// The V3 arm of REBALANCE: V2's fixed-point loop verbatim, the
    /// probe running the composed program through the operand-source
    /// seam (same probe the per-INSERT balance uses).
    fn exec_rebalance_v3(
        &mut self,
        max_iters: usize,
        floor: f64,
        ceiling: f64,
    ) -> Result<Vec<String>, LqlError> {
        let bos = self.v3_bos_token();
        use crate::executor::mutation::insert::probe_target_prob;
        use crate::executor::vindex3::encode_raw_prompt;
        use crate::executor::Backend;

        let n_facts = self.installed_edges.len();
        let mut iters_run = 0usize;
        let mut final_probs: Vec<f64> = vec![0.0; n_facts];

        for iter in 0..max_iters {
            iters_run = iter + 1;
            let mut any_changed = false;
            let facts_snapshot = self.installed_edges.clone();

            for (i, fact) in facts_snapshot.iter().enumerate() {
                let prob = {
                    let Backend::Vindex3 {
                        runtime,
                        tokenizer,
                        overlay,
                        ..
                    } = &self.backend
                    else {
                        unreachable!("dispatch matched the backend");
                    };
                    let tokenizer = tokenizer.as_ref().expect("compose install had a tokenizer");
                    let ids = encode_raw_prompt(tokenizer, fact.canonical_prompt.as_str(), bos)?;
                    probe_target_prob(runtime, tokenizer, overlay, &ids, &fact.target)?
                };
                final_probs[i] = prob;

                let scale: Option<f32> = if prob > ceiling {
                    Some(REBALANCE_DOWN_SCALE)
                } else if prob < floor {
                    Some(REBALANCE_UP_SCALE)
                } else {
                    None
                };
                if let Some(scale) = scale {
                    if self.overlay_down_at(fact.layer, fact.feature).is_some() {
                        self.scale_overlay_down(fact.layer, fact.feature, scale);
                        any_changed = true;
                    }
                }
            }
            if !any_changed {
                break;
            }
        }

        // Recorded Insert ops must carry the rebalanced vectors, as on
        // V2 (`refresh_recorded_patch_ops_for_slots`).
        let slots: Vec<(usize, usize)> = self
            .installed_edges
            .iter()
            .map(|fact| (fact.layer, fact.feature))
            .collect();
        self.refresh_recorded_patch_ops_for_slots_v3(&slots);

        let mut in_band = 0usize;
        let mut below = 0usize;
        let mut above = 0usize;
        for &p in &final_probs {
            if p < floor {
                below += 1;
            } else if p > ceiling {
                above += 1;
            } else {
                in_band += 1;
            }
        }
        let mut out = Vec::new();
        out.push(format!(
            "Rebalance: {n_facts} compose installs, {iters_run} iterations",
        ));
        out.push(format!(
            "  band [{floor:.2}, {ceiling:.2}]: {in_band} in band, {below} below (amplifying), {above} above (shrinking)"
        ));
        out.push(format!(
            "  {}",
            if below == 0 && above == 0 {
                "all converged in band"
            } else {
                "saturated (some facts hit oscillation limit — template-competition at this layer)"
            }
        ));
        Ok(out)
    }

    /// V2's post-hoc op refresh, over the overlay: rewrite recorded
    /// `Insert` ops' vectors from the slots' final overlay state.
    fn refresh_recorded_patch_ops_for_slots_v3(&mut self, slots: &[(usize, usize)]) {
        if slots.is_empty() || self.patch_recording.is_none() {
            return;
        }
        let snapshots: std::collections::HashMap<(usize, usize), (String, String, String)> = {
            let crate::executor::Backend::Vindex3 { overlay, .. } = &self.backend else {
                unreachable!("dispatch matched the backend");
            };
            let b64 = larql_vindex::patch::core::encode_gate_vector;
            slots
                .iter()
                .filter_map(|&(layer, feature)| {
                    Some((
                        (layer, feature),
                        (
                            b64(overlay.gate_override_at(layer, feature)?),
                            b64(overlay.up_override_at(layer, feature)?),
                            b64(overlay.down_override_at(layer, feature)?),
                        ),
                    ))
                })
                .collect()
        };
        let Some(recording) = self.patch_recording.as_mut() else {
            return;
        };
        for op in &mut recording.operations {
            if let larql_vindex::PatchOp::Insert {
                layer,
                feature,
                gate_vector_b64,
                up_vector_b64,
                down_vector_b64,
                ..
            } = op
            {
                if let Some((gate, up, down)) = snapshots.get(&(*layer, *feature)) {
                    *gate_vector_b64 = Some(gate.clone());
                    *up_vector_b64 = Some(up.clone());
                    *down_vector_b64 = Some(down.clone());
                }
            }
        }
    }
}
