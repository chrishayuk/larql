//! The VINDEX3 arm of `INSERT … MODE COMPOSE` (V3-LQL-3B compose).
//!
//! Same install formula as the V2 arm in `compose.rs`
//! (`install_compiled_slot`, validated on Gemma 3 4B):
//!
//! ```text
//! gate[slot]   = unit(residual) · g_ref · GATE_SCALE
//! up[slot]     = unit(residual) · u_ref
//! down[:,slot] = unit(target_embed) · d_ref · alpha_mul
//! ```
//!
//! with the layer-median reference norms sampled the same way (first
//! `min(n, 100)` features, L2, median, 1.0 fallback) — but every input
//! resolved through V3's own authorities: the residual from the plan's
//! execution taps, the target embedding from role `embedding`, the
//! norms from the plan's FFN operands via the runtime's resolver, and
//! the written vectors landing in the [`KnowledgeOverlay`], where
//! browse merges them into its scan and execution observes them
//! through the operand-source seam.
//!
//! The full V2 pipeline is ported, control flow moved onto the seam
//! and nothing else changed:
//!
//! 1. plan — install layer, free slot (V2's rule), target embedding
//!    from role `embedding`;
//! 2. capture — canonical residual + layer decoys on the **clean
//!    base** program (V2's contract: prior installs must not
//!    contaminate the capture; see `capture.rs`);
//! 3. install — the slot formula above, then the batch refine:
//!    every gate at the layer rebuilt from the session's RAW captured
//!    residuals + decoys via the shared `refine_gates` (never from
//!    refined state — V2's idempotence rule, the fix for its
//!    compound-drift bug);
//! 4. balance + cross-fact regression — V2's decision rules verbatim,
//!    probing through the composed V3 forward (see `balance.rs`);
//! 5. record — the PatchOp carries the FINAL post-refine/post-balance
//!    vectors read back from the overlay.
//!
//! [`KnowledgeOverlay`]: larql_vindex::format::vindex3::knowledge::KnowledgeOverlay

mod balance;
mod capture;

pub(crate) use balance::probe_target_prob;
#[cfg(test)]
mod tests;

use larql_inference::vindex3::Vindex3Runtime;
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
use larql_vindex::format::vindex3::opplan::LayerFfn;

use super::compose::{median_or, unit_vector};
use super::DEFAULT_INSERT_CONFIDENCE;
use crate::error::LqlError;
use crate::executor::tuning::{DEFAULT_INSERT_ALPHA_MUL, GATE_SCALE};
use crate::executor::{Backend, Session};

/// How many features the layer-median norm statistic samples — the V2
/// arm's `compute_layer_median_norms(_, _, 100)`.
const NORM_SAMPLE_SIZE: usize = 100;

/// The three layer-typical norms the install matches (g_ref / u_ref /
/// d_ref) — V2's rule, ported for IDENTICAL behaviour:
///
/// V2 samples its **browse feature stores** (`gate_vectors.bin`,
/// `up_features.bin`, `down_features.bin`), and current extractions
/// only produce the gate store — `up_features.bin`/`down_features.bin`
/// do not exist on a `build_vindex` output, so V2's up/down medians
/// ALWAYS take the `median_or` fallback of 1.0. That fallback is what
/// the validated install behaviour ran with, so the port reproduces
/// it exactly: a real gate median, unit up/down references. Computing
/// real up/down medians from the plan's operands is a strictly-later
/// improvement, to be made on BOTH arms after parity is banked (the
/// V2→V3 compose parity gate pinned this — its stage-3 magnitude
/// check caught the first version of this function computing real
/// medians V2 never sees).
fn layer_median_norms(
    runtime: &Vindex3Runtime<ProductionBackend>,
    layer: usize,
) -> Result<(f32, f32, f32), LqlError> {
    let Some(LayerFfn::Dense(ffn)) = &runtime.plan().layers[layer].ffn else {
        return Err(LqlError::Execution(format!(
            "layer {layer} is routed — compose installs on MoE layers are a later role rung"
        )));
    };
    let operands = runtime.operands();
    let gate_ref = ffn.gate.as_ref().unwrap_or(&ffn.up);
    let gate = operands
        .load(gate_ref)
        .map_err(|e| LqlError::exec("failed to load FFN operand", e))?;

    let features = gate_ref.shape[0];
    let hidden = gate_ref.shape[1];
    let sample = features.min(NORM_SAMPLE_SIZE);
    let mut gate_norms: Vec<f32> = (0..sample)
        .filter_map(|i| {
            let row = &gate[i * hidden..(i + 1) * hidden];
            let n: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            (n.is_finite() && n > 0.0).then_some(n)
        })
        .collect();

    Ok((median_or(&mut gate_norms, 1.0), 1.0, 1.0))
}

impl Session {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn exec_insert_compose_v3(
        &mut self,
        entity: &str,
        relation: &str,
        target: &str,
        layer_hint: Option<u32>,
        confidence: Option<f32>,
        alpha_override: Option<f32>,
    ) -> Result<Vec<String>, LqlError> {
        let bos = self.v3_bos_token();
        let alpha_mul = alpha_override.unwrap_or(DEFAULT_INSERT_ALPHA_MUL);
        let c_score = confidence.unwrap_or(DEFAULT_INSERT_CONFIDENCE);

        let (layer, feature, target_id, gate_vec, up_vec, down_vec);
        let (pending_decoys, raw_residual);
        let (g_ref, u_ref);
        {
            let Backend::Vindex3 {
                runtime,
                tokenizer,
                knowledge,
                overlay,
                ..
            } = &self.backend
            else {
                unreachable!("caller matched the backend");
            };
            let tokenizer = tokenizer.as_ref().ok_or_else(|| {
                LqlError::Execution(
                    "INSERT needs a tokenizer (the canonical prompt and the target must \
                     tokenize) and this container carries no tokenizer.json"
                        .into(),
                )
            })?;
            let view = knowledge.as_ref().ok_or_else(|| {
                LqlError::Execution(
                    "a compose install needs the browse view (free-slot search reads the \
                     feature space) and this container carries no tokenizer.json"
                        .into(),
                )
            })?;

            // ── Plan: layer, slot, target embedding ──
            let num_layers = runtime.plan().layers.len();
            layer = match layer_hint {
                Some(l) => (l as usize).min(num_layers.saturating_sub(1)),
                None => num_layers.saturating_sub(2),
            };
            feature = overlay.find_free_feature(view, layer).ok_or_else(|| {
                LqlError::Execution(format!("no free feature slot at layer {layer}"))
            })?;

            let spaced_target = format!(" {target}");
            let target_encoding = tokenizer
                .encode(spaced_target.as_str(), false)
                .map_err(|e| LqlError::exec("tokenize error", e))?;
            target_id = target_encoding.get_ids().first().copied().unwrap_or(0);
            let (embed, embed_scale) = view.embedding();
            let row = embed.row((target_id as usize).min(embed.shape()[0].saturating_sub(1)));
            let target_embed: Vec<f32> = row.iter().map(|v| v * embed_scale).collect();

            // ── Capture: the canonical prompt's residual on the CLEAN
            // BASE program (V2's contract — prior installs' slots must
            // not fire during capture), plus this layer's decoys when
            // the session cache does not hold them yet ──
            let prompt = crate::executor::tuning::canonical_prompt(relation, entity);
            let residual = crate::executor::vindex3::capture_layer_residual(
                runtime,
                tokenizer,
                prompt.as_str(),
                layer,
                None,
                bos,
            )?;
            pending_decoys = capture::capture_layer_decoys(
                self,
                runtime,
                tokenizer,
                view.vocab_size(),
                entity,
                relation,
                layer,
            )?;
            raw_residual = residual.clone();
            let gate_dir = unit_vector(&residual);

            // ── Synthesis: the validated install formula ──
            let (g, u, d_ref) = layer_median_norms(runtime, layer)?;
            g_ref = g;
            u_ref = u;
            gate_vec = gate_dir
                .iter()
                .map(|v| v * g_ref * GATE_SCALE)
                .collect::<Vec<f32>>();
            up_vec = gate_dir.iter().map(|v| v * u_ref).collect::<Vec<f32>>();
            let target_unit = unit_vector(&target_embed);
            down_vec = target_unit
                .iter()
                .map(|v| v * d_ref * alpha_mul)
                .collect::<Vec<f32>>();
        }

        // Commit the decoys and this fact's RAW residual to the session
        // caches — the refine pass reads the raw snapshot, never
        // refined state (V2's idempotence rule).
        if let Some(decoys) = pending_decoys {
            self.decoy_residual_cache.insert(layer, decoys);
        }
        self.raw_install_residuals.insert(
            (layer, feature),
            larql_vindex::ndarray::Array1::from_vec(raw_residual),
        );

        // ── Install: overlay state browse merges and execution
        // observes through the operand-source seam ──
        let meta = larql_vindex::FeatureMeta {
            top_token: target.to_string(),
            top_token_id: target_id,
            c_score,
            top_k: vec![larql_models::TopKEntry {
                token: target.to_string(),
                token_id: target_id,
                logit: c_score,
            }],
        };
        {
            let Backend::Vindex3 { overlay, .. } = &mut self.backend else {
                unreachable!("caller matched the backend");
            };
            overlay.insert_feature(layer, feature, gate_vec, meta);
            overlay.set_up_vector(layer, feature, up_vec);
            overlay.set_down_vector(layer, feature, down_vec);
        }

        // ── Batch refine from raw captures (V2's `refine_layer_from_raw`):
        // rebuild EVERY gate + up at this layer from the raw residual
        // snapshot + decoys — idempotent by construction ──
        self.refine_layer_from_raw_v3(layer, g_ref, u_ref);

        // ── Balance + cross-fact regression (V2's decision rules,
        // composed-forward probes) ──
        self.balance_installed_v3(layer, feature, entity, relation, target)?;
        self.cross_fact_regression_check_v3(layer, feature)?;

        // Register THIS fact for future cross-balance passes.
        self.installed_edges.push(crate::executor::InstalledEdge {
            layer,
            feature,
            canonical_prompt: crate::executor::tuning::canonical_prompt(relation, entity),
            target: target.to_string(),
            target_id,
        });

        // ── Record: the patch op carries the FINAL post-refine /
        // post-balance vectors, read back from the overlay ──
        let (final_gate, final_up, final_down) = {
            let Backend::Vindex3 { overlay, .. } = &self.backend else {
                unreachable!("caller matched the backend");
            };
            (
                overlay
                    .gate_override_at(layer, feature)
                    .map(<[f32]>::to_vec)
                    .unwrap_or_default(),
                overlay
                    .up_override_at(layer, feature)
                    .map(<[f32]>::to_vec)
                    .unwrap_or_default(),
                overlay
                    .down_override_at(layer, feature)
                    .map(<[f32]>::to_vec)
                    .unwrap_or_default(),
            )
        };
        if let Some(ref mut recording) = self.patch_recording {
            let b64 = larql_vindex::patch::core::encode_gate_vector;
            recording.operations.push(larql_vindex::PatchOp::Insert {
                layer,
                feature,
                relation: Some(relation.to_string()),
                entity: entity.to_string(),
                target: target.to_string(),
                confidence: Some(c_score),
                gate_vector_b64: Some(b64(&final_gate)),
                up_vector_b64: Some(b64(&final_up)),
                down_vector_b64: Some(b64(&final_down)),
                down_meta: Some(larql_vindex::patch::core::PatchDownMeta {
                    top_token: target.to_string(),
                    top_token_id: target_id,
                    c_score,
                }),
            });
        }

        Ok(vec![
            format!(
                "Inserted: {} —[{}]→ {} at L{}/F{} (compose overlay)",
                entity, relation, target, layer, feature,
            ),
            "  mode: COMPOSE — FFN slot install (VINDEX3 operand-source seam); \
             refine + balance + cross-fact passes applied (V2 pipeline)"
                .into(),
            format!("  alpha {alpha_mul}, gate scale {GATE_SCALE}"),
        ])
    }

    /// V2's `refine_layer_from_raw`, on the overlay: rebuild every
    /// composed gate + up at `layer` from the session's RAW residual
    /// snapshot + cached decoys via the shared `refine_gates`
    /// (modified Gram-Schmidt). Reading raw captures — never refined
    /// state — is what makes repeated refines idempotent.
    pub(crate) fn refine_layer_from_raw_v3(&mut self, layer: usize, g_ref: f32, u_ref: f32) {
        let inputs: Vec<larql_vindex::RefineInput> = self
            .raw_install_residuals
            .iter()
            .filter(|((l, _), _)| *l == layer)
            .map(|((l, f), r)| larql_vindex::RefineInput {
                layer: *l,
                feature: *f,
                gate: r.clone(),
            })
            .collect();
        let decoys = self.decoy_residual_cache.get(&layer).cloned();
        let layer_decoys = decoys.as_deref().unwrap_or(&[]);

        if !super::compose::should_refine(inputs.len(), layer_decoys.len()) {
            return;
        }

        let result = larql_vindex::refine_gates(&inputs, layer_decoys);
        let Backend::Vindex3 { overlay, .. } = &mut self.backend else {
            unreachable!("caller matched the backend");
        };
        for refined in result.gates {
            let refined_vec: Vec<f32> = refined.gate.into_raw_vec_and_offset().0;
            let dir = unit_vector(&refined_vec);
            let new_gate: Vec<f32> = dir.iter().map(|v| v * g_ref * GATE_SCALE).collect();
            let new_up: Vec<f32> = dir.iter().map(|v| v * u_ref).collect();
            overlay.set_gate_vector(refined.layer, refined.feature, new_gate);
            overlay.set_up_vector(refined.layer, refined.feature, new_up);
        }
    }
}
