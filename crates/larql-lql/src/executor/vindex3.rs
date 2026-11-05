//! VINDEX3 statement execution (LQL-1: USE / SHOW / INFER).
//!
//! The invariant this module exists to keep: **LQL binds a model once,
//! then operates on the runtime's declared facts and capabilities.**
//! No statement here reconstructs architecture from weights or family
//! metadata — every fact printed by `STATS` / `SHOW LAYERS` is read
//! from the container's executable plan, and `INFER` runs the same
//! runtime seam `larql-server` serves
//! (`prefill_into → session_with_kv → continue_session`), so a fourth
//! entry point joins the equality chain the SERVE-1 gates established.

use larql_inference::layer_graph::generate::detok::Detokenizer;
use larql_inference::vindex3::{continue_session, Vindex3Runtime};
use larql_inference::{EosConfig, SamplingConfig};
use larql_vindex::format::vindex3::opplan::exec::continuation::{
    plan_continuation_geometry, LayerContinuationGeometry,
};
use larql_vindex::format::vindex3::opplan::exec::lowering::SharedProvider;
use larql_vindex::format::vindex3::opplan::LayerAttention;

use crate::error::LqlError;
use crate::executor::{Backend, Session};

mod overrides;
mod session;
mod weights;
pub(crate) use overrides::*;
pub(crate) use session::*;

/// The statements a VINDEX3 binding serves today. Everything else gets
/// [`unsupported`] — a capability refusal, not a format apology.
/// The capability string is part of the contract, not decoration: a
/// statement that executes on a V3 binding must appear here, or the
/// binding is understating what it serves. EXTRACT does execute (the
/// dispatch does not gate on the backend) and now says so.
pub(crate) const SUPPORTED: &str = "SELECT, DESCRIBE, WALK, EXPLAIN WALK, \
     SHOW RELATIONS/LAYERS/FEATURES/ENTITIES/PATCHES/MODELS, \
     SHOW COMPONENTS/REPRESENTATIONS/PROVENANCE/AUTHORITY, INFER [TOP n] [GENERATE n], \
     EXPLAIN INFER, TRACE, STATS, USE, EXTRACT [FORMAT VINDEX2|VINDEX3], \
     INSERT [MODE KNN|COMPOSE], DELETE, UPDATE, UPDATE WEIGHTS, MERGE, \
     BEGIN/SAVE/APPLY/REMOVE PATCH, COMPILE [CURRENT] INTO VINDEX, DIFF [PHYSICAL], COMPACT INTO VINDEX";

/// Component id a container's text stack is bound under.
pub(crate) const V3_COMPONENT: &str = "target";

/// The served realisation, resolved from the shipped registry by
/// identity rather than constructed here (LOWERING-PLUGIN-1, L3).
pub(crate) type V3Runtime = Vindex3Runtime<SharedProvider>;

/// The capability refusal for statements a V3 binding does not serve.
pub(crate) fn unsupported(what: &str) -> LqlError {
    LqlError::Execution(format!(
        "{what} is not supported on a VINDEX3 container yet. \
         Supported: {SUPPORTED}."
    ))
}

impl Session {
    /// The BOS token id the bound V3 container declares, resolved once
    /// at bind. `None` on any other backend — callers are already in a
    /// V3 arm when they ask.
    pub(crate) fn v3_bos_token(&self) -> Option<u32> {
        match &self.backend {
            Backend::Vindex3 { bos_token, .. } => *bos_token,
            _ => None,
        }
    }

    /// `INFER` on a V3 binding. Without `GENERATE`: classic single-step
    /// top-k next-token prediction, priced from the batch-prefill
    /// logits. With `GENERATE n`: greedy autoregressive continuation
    /// through the proven runtime stack.
    pub(crate) fn exec_v3_infer(
        &self,
        prompt: &str,
        top_k: usize,
        generate: Option<u32>,
    ) -> Result<Vec<String>, LqlError> {
        let Backend::Vindex3 {
            runtime,
            tokenizer,
            overlay,
            bos_token,
            ..
        } = &self.backend
        else {
            unreachable!("caller matched the backend");
        };
        let tokenizer = tokenizer.as_ref().ok_or_else(|| {
            LqlError::Execution(
                "INFER needs a tokenizer and this container carries no tokenizer.json — \
                 token-id capability only"
                    .into(),
            )
        })?;
        let prompt_ids = encode_v3_prompt(tokenizer, prompt, *bos_token)?;

        let start = std::time::Instant::now();

        match generate {
            None => {
                // One observed pass over the same traversal generation
                // runs: logits for the display, per-layer residual taps
                // for the KnnStore override (only captured while the
                // overlay holds entries).
                let capture_residuals = !overlay.knn_store.is_empty();
                let mut residuals: Vec<(usize, Vec<f32>)> = Vec::new();
                let mut sink = |event: larql_inference::vindex3::PlaneEvent| {
                    if capture_residuals {
                        if let larql_inference::vindex3::PlaneEvent::Layer { index, trace } = event
                        {
                            // Normed FFN inputs — the same tap the
                            // stored keys were captured from.
                            if let Some(last) = trace.ffn_input.last() {
                                residuals.push((index, last.clone()));
                            }
                        }
                    }
                    Ok(())
                };
                // Compose edits reach execution through the operand-
                // source seam; without them this is the plain pass,
                // bit for bit.
                let output = match compose_overrides(runtime, overlay)? {
                    Some(overrides) => runtime
                        .execute_streaming_overlaid(
                            &prompt_ids,
                            &overrides,
                            &v3_continuation(runtime)?,
                            &mut sink,
                        )
                        .map_err(|e| LqlError::exec("v3 prefill failed", e))?,
                    None => runtime
                        .execute_streaming(&prompt_ids, &v3_continuation(runtime)?, &mut sink)
                        .map_err(|e| LqlError::exec("v3 prefill failed", e))?,
                };
                let logits = output.logits.ok_or_else(|| {
                    LqlError::Execution("the component carries no output head".into())
                })?;
                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

                let scored = top_k_probs(&logits, top_k);
                // The shared post-logits resolution rule — same
                // first-stored-layer, fixed-threshold gate V2 runs.
                let raw: Vec<(String, f64)> = scored
                    .iter()
                    .map(|(id, prob)| {
                        (
                            tokenizer.decode(&[*id], false).unwrap_or_default(),
                            *prob as f64,
                        )
                    })
                    .collect();
                let (_, knn_override) = larql_inference::apply_knn_override(
                    raw,
                    &residuals,
                    Some(&overlay.knn_store),
                    top_k,
                );

                let mut out = Vec::new();
                out.push("Predictions (VINDEX3 program):".into());
                match &knn_override {
                    Some(ovr) => {
                        let model_top1 = scored.first().map(|(id, prob)| {
                            (
                                tokenizer.decode(&[*id], false).unwrap_or_default(),
                                *prob as f64,
                            )
                        });
                        out.push(format!(
                            "   1. {:20} (100.00%, {})",
                            ovr.token,
                            crate::executor::helpers::format_knn_override_summary(
                                ovr,
                                model_top1.as_ref(),
                            ),
                        ));
                        for (i, (id, prob)) in
                            scored.iter().take(top_k.saturating_sub(1)).enumerate()
                        {
                            let token = tokenizer.decode(&[*id], false).unwrap_or_default();
                            out.push(format!(
                                "  {:2}. {:20} ({:.2}%)  [id {}]",
                                i + 2,
                                token,
                                prob * 100.0,
                                id,
                            ));
                        }
                    }
                    None => {
                        for (i, (id, prob)) in scored.iter().enumerate() {
                            let token = tokenizer.decode(&[*id], false).unwrap_or_default();
                            out.push(format!(
                                "  {:2}. {:20} ({:.2}%)  [id {}]",
                                i + 1,
                                token,
                                prob * 100.0,
                                id,
                            ));
                        }
                    }
                }
                out.push(format!("  {:.0}ms", elapsed_ms));
                if knn_override.is_some() {
                    out.push(
                        "  note: KNN override is a post-logits retrieval sidecar, not an \
                         FFN/residual edit."
                            .into(),
                    );
                }
                Ok(out)
            }
            Some(n) => {
                let mut kv = v3_continuation(runtime)?.build();
                let overrides = compose_overrides(runtime, overlay)?;
                let prefill_logits = match &overrides {
                    Some(ov) => runtime.prefill_into_overlaid(&prompt_ids, ov, &mut *kv),
                    None => runtime.prefill_into(&prompt_ids, &mut *kv),
                }
                .map_err(|e| LqlError::exec("v3 prefill failed", e))?;
                let mut session = match &overrides {
                    Some(ov) => runtime.session_with_kv_overlaid(&mut *kv, ov),
                    None => runtime.session_with_kv(&mut *kv),
                }
                .map_err(|e| LqlError::exec("v3 session failed", e))?;
                let mut detok = Detokenizer::new(tokenizer);
                detok.seed(&prompt_ids);
                let mut text = String::new();
                let result = continue_session(
                    &mut session,
                    prefill_logits,
                    n as usize,
                    SamplingConfig::greedy(),
                    &EosConfig::builtin(),
                    |id| text.push_str(&detok.push(id)),
                )
                .map_err(|e| LqlError::exec("v3 decode failed", e))?;
                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

                let ids: Vec<String> = result.tokens.iter().map(u32::to_string).collect();
                Ok(vec![
                    format!("Generated ({} tokens, greedy):", result.tokens.len()),
                    format!("  ids:  {}", ids.join(",")),
                    format!("  text: {}", text),
                    format!(
                        "  prompt {} tokens, continued from position {}",
                        result.prompt_len, result.prompt_len,
                    ),
                    format!("  {:.0}ms", elapsed_ms),
                ])
            }
        }
    }

    /// `STATS` on a V3 binding: the container's own authority, not a
    /// reconstruction — generation, component, closure, and geometry as
    /// the executable plan declares them.
    pub(crate) fn exec_v3_stats(&self) -> Result<Vec<String>, LqlError> {
        let Backend::Vindex3 {
            path,
            runtime,
            tokenizer,
            ..
        } = &self.backend
        else {
            unreachable!("caller matched the backend");
        };
        let plan = runtime.plan();
        // Read the layers themselves, not `plan_kv_geometry` — that
        // adapter refuses (formerly: panicked, drill F8) on any model
        // carrying recurrent continuation, and a pure-SSM binding is
        // exactly such a model. A recurrence has no KV row to report;
        // the honest line counts it as what it is.
        let mut sliding = 0usize;
        let mut full = 0usize;
        let mut kv_dims: std::collections::BTreeSet<usize> = Default::default();
        for layer in &plan.layers {
            if let Some(op) = layer.attention.softmax() {
                if op.window.is_some() {
                    sliding += 1;
                } else {
                    full += 1;
                }
                kv_dims.insert(op.num_kv_heads * op.head_dim);
            }
        }
        // **The continuation summary comes from the continuation
        // planner**, never from "which layers are not softmax". That
        // shortcut was true while a recurrence was the only alternative
        // to rows, and became a false claim the moment MLA executed: its
        // layers are not recurrent, and its cache is not constant in
        // sequence length. Read off the wrong axis, this line reported a
        // 27-layer Kimi stack as "recurrent state only … constant in
        // sequence length" while 7 of those layers grew a cache with
        // every position. A summary is exactly where such a claim goes
        // unnoticed, so it is derived from the same seam the executor
        // allocates from.
        let regions = plan_continuation_geometry(plan);
        let (mut recurrent, mut latent, mut state_elements) = (0usize, 0usize, 0usize);
        let mut latent_widths: std::collections::BTreeSet<usize> = Default::default();
        if let Ok(regions) = &regions {
            for region in regions {
                match region {
                    LayerContinuationGeometry::Recurrent(r)
                    | LayerContinuationGeometry::KvAndRecurrent { recurrent: r, .. } => {
                        recurrent += 1;
                        state_elements += r.elements();
                    }
                    LayerContinuationGeometry::LatentKv(l) => {
                        latent += 1;
                        latent_widths.insert(l.width);
                    }
                    LayerContinuationGeometry::Kv(_) | LayerContinuationGeometry::Stateless => {}
                }
            }
        }
        let kv_dims: Vec<String> = kv_dims.iter().map(usize::to_string).collect();
        let mut spans = vec![format!("{sliding} sliding"), format!("{full} full")];
        if latent > 0 {
            spans.push(format!("{latent} latent-cache"));
        }
        if recurrent > 0 {
            spans.push(format!("{recurrent} recurrent"));
        }
        let attention_line = format!(
            "Attention:       {} (windows from the plan)",
            spans.join(" / ")
        );
        // One clause per region the program actually declares — nothing
        // asserted about a region that is absent, and the growth
        // behaviour stated per clause rather than for the stack as a
        // whole, which is what made the old line wrong on a hybrid.
        let continuation_line = match &regions {
            Err(refusal) => format!("Continuation:    cannot be sized: {refusal}"),
            Ok(_) => {
                let mut parts: Vec<String> = Vec::new();
                if !kv_dims.is_empty() {
                    parts.push(format!(
                        "KV rows (kv_dim {}), growing with the prefix",
                        kv_dims.join(", ")
                    ));
                }
                if latent > 0 {
                    let widths: Vec<String> = latent_widths.iter().map(usize::to_string).collect();
                    parts.push(format!(
                        "latent cache on {latent} layer(s) ({} elements per position), growing \
                         with the prefix",
                        widths.join(", ")
                    ));
                }
                if recurrent > 0 {
                    parts.push(format!(
                        "recurrent state on {recurrent} layer(s) ({state_elements} elements), \
                         constant in sequence length"
                    ));
                }
                if parts.is_empty() {
                    "Continuation:    nothing survives a step".to_string()
                } else {
                    format!("Continuation:    {}", parts.join("; "))
                }
            }
        };

        Ok(vec![
            format!("Model:           {} (VINDEX3)", runtime.model_name()),
            format!("Path:            {}", path.display()),
            "Generation:      3".into(),
            format!("Component:       {V3_COMPONENT}"),
            "Execution:       closed (operand-verified executable plan)".into(),
            format!("Layers:          {}", plan.layers.len()),
            attention_line,
            continuation_line,
            format!(
                "Output head:     {}",
                if plan.output.is_some() {
                    "present"
                } else {
                    "absent"
                }
            ),
            format!(
                "Tokenizer:       {}",
                if tokenizer.is_some() {
                    "present"
                } else {
                    "absent (token-id capability only)"
                }
            ),
            format!("Capabilities:    {SUPPORTED}"),
        ])
    }

    /// `SHOW LAYERS` on a V3 binding: per-layer attention facts, read
    /// off the executable plan.
    pub(crate) fn exec_v3_show_layers(&self) -> Result<Vec<String>, LqlError> {
        let Backend::Vindex3 { runtime, .. } = &self.backend else {
            unreachable!("caller matched the backend");
        };
        let plan = runtime.plan();
        let mut out = Vec::new();
        out.push(format!(
            "{:<8} {:<10} {:>8} {:>10} {:>10} {:>8}",
            "Layer", "Attention", "Window", "Q heads", "KV heads", "Head dim"
        ));
        out.push("-".repeat(60));
        for (index, layer) in plan.layers.iter().enumerate() {
            // A linear-attention layer has no window and no softmax head
            // geometry to show. The columns read `-` rather than borrowing
            // the recurrence's head counts, which describe a fixed-size
            // state and not retained positions.
            let dash = "-".to_string();
            let (kind, window, q_heads, kv_heads, head_dim) = match layer.attention.softmax() {
                Some(op) => (
                    if op.window.is_some() {
                        "sliding"
                    } else {
                        "full"
                    },
                    op.window
                        .map(|w| w.to_string())
                        .unwrap_or_else(|| dash.clone()),
                    op.num_q_heads.to_string(),
                    op.num_kv_heads.to_string(),
                    op.head_dim.to_string(),
                ),
                // Name the operator, not the coarse family: "linear"
                // said only what the layer is NOT. The columns still read
                // `-` — a recurrence's head counts describe a fixed-size
                // state, not retained positions.
                None => (
                    match &layer.attention {
                        LayerAttention::GatedDelta(_) => "gated-delta",
                        LayerAttention::Kda(_) => "kda",
                        LayerAttention::Mamba2(_) => "mamba2",
                        LayerAttention::Mla(_) => "mla",
                        LayerAttention::ConvQkv(_) => "conv-qkv",
                        LayerAttention::Softmax(_) => unreachable!("softmax handled above"),
                    },
                    dash.clone(),
                    dash.clone(),
                    dash.clone(),
                    dash.clone(),
                ),
            };
            out.push(format!(
                "{:<8} {:<10} {:>8} {:>10} {:>10} {:>8}",
                index, kind, window, q_heads, kv_heads, head_dim,
            ));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests;
