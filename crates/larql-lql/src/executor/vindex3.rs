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

use larql_inference::vindex3::Vindex3Runtime;
use larql_vindex::format::vindex3::opplan::exec::continuation::{
    plan_continuation_geometry, LayerContinuationGeometry,
};
use larql_vindex::format::vindex3::opplan::exec::lowering::SharedProvider;
use larql_vindex::format::vindex3::opplan::LayerAttention;

use crate::error::LqlError;
use crate::executor::{Backend, Session};

mod capture;
mod export_weight;
mod generate;
mod live;
mod low_rank;
mod npy;
mod npy_output;
mod overrides;
mod session;
mod tensors;
mod weights;
pub(crate) use capture::{AppliedPatchFile, CaptureRequest};
pub use capture::{CaptureSpec, Captured};
pub use export_weight::WeightRead;
pub(crate) use live::{live_weight_lines, nothing_in_effect_line};
pub use low_rank::{operand_bits_from_values, operand_origin, OperandValues, StagedLowRank};
pub use npy::{npy_file_bytes, NpyDescr};
pub(crate) use overrides::*;
pub(crate) use session::*;
pub use weights::{DiscardedWeightEdits, EditedElement, StagedWeightDelta};

/// The statements a VINDEX3 binding serves today. Everything else gets
/// [`unsupported`] — a capability refusal, not a format apology.
/// The capability string is part of the contract, not decoration, so it is
/// not kept here: it is every statement of the help catalogue whose page
/// does not require a VINDEX2 vindex. A statement that executes on a V3
/// binding is therefore listed exactly when `HELP` documents it as served,
/// and `tests/help/banner.rs` runs the catalogue's examples on a container
/// to check that the pages and the executor agree.
pub(crate) fn supported() -> String {
    crate::help::render::vindex3_supported()
}

/// How a representation source is spelled, and what it means for the weights a run computes with. The words are
/// the ones `larql vindex3 exec --representation-source` takes.
pub(crate) fn describe_source(
    source: larql_vindex::format::vindex3::opplan::exec::operands::RepresentationSource,
) -> &'static str {
    use larql_vindex::format::vindex3::opplan::exec::operands::RepresentationSource;
    match source {
        RepresentationSource::Stored => {
            "stored (the stored weights execute; nothing is manufactured at load)"
        }
        RepresentationSource::Auto => {
            "auto (the loader may quantise large matrices at load: faster, and not the stored arithmetic)"
        }
        RepresentationSource::Transient => {
            "transient (representations are quantised at load, ignoring any compiled pack)"
        }
    }
}

/// Component id a container's text stack is bound under.
pub(crate) const V3_COMPONENT: &str = "target";

/// The served realisation, resolved from the shipped registry by
/// identity rather than constructed here (LOWERING-PLUGIN-1, L3).
pub(crate) type V3Runtime = Vindex3Runtime<SharedProvider>;

/// The capability refusal for statements a V3 binding does not serve.
pub(crate) fn unsupported(what: &str) -> LqlError {
    LqlError::Execution(format!(
        "{what} is not supported on a VINDEX3 container yet. \
         Supported: {}. Run HELP; for the syntax of each.",
        supported()
    ))
}

/// One ranked next token.
#[derive(Debug, Clone, PartialEq)]
pub struct Prediction {
    pub token_id: u32,
    /// The token decoded alone, special tokens kept.
    pub token: String,
    /// Its softmax probability over the whole vocabulary.
    pub probability: f32,
}

/// What `INFER … TOP <k>` computes on a VINDEX3 binding.
#[derive(Debug, Clone)]
pub struct Predictions {
    /// The prompt as INFER's raw-prompt encoder encodes it.
    pub prompt_ids: Vec<u32>,
    /// The top `k` tokens, most probable first.
    pub top: Vec<Prediction>,
    /// The session KNN store's answer, where it overrides the model's top token.
    pub knn_override: Option<larql_inference::KnnOverride>,
    pub elapsed_ms: f64,
}

/// What `INFER … GENERATE <n>` computes on a VINDEX3 binding.
#[derive(Debug, Clone)]
pub struct Generated {
    /// The prompt as INFER's raw-prompt encoder encodes it.
    pub prompt_ids: Vec<u32>,
    /// The prompt length prefill consumed, in tokens.
    pub prompt_len: usize,
    /// The generated ids, in order; a stop token is never included.
    pub token_ids: Vec<u32>,
    /// Their text, detokenized as generation streamed it.
    pub text: String,
    pub elapsed_ms: f64,
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

    /// The bound container's runtime, tokenizer, overlay and declared BOS, for a statement that runs the
    /// program over text: refused without a VINDEX3 binding, or without a tokenizer.
    fn v3_text_program(
        &self,
        statement: &str,
    ) -> Result<
        (
            &V3Runtime,
            &larql_vindex::tokenizers::Tokenizer,
            &larql_vindex::format::vindex3::knowledge::KnowledgeOverlay,
            Option<u32>,
        ),
        LqlError,
    > {
        let (runtime, tokenizer, overlay, bos_token) = match &self.backend {
            Backend::Vindex3 {
                runtime,
                tokenizer,
                overlay,
                bos_token,
                ..
            } => (runtime, tokenizer, overlay, *bos_token),
            Backend::None => return Err(LqlError::NoBackend),
            _ => {
                return Err(LqlError::Execution(format!(
                    "{statement} here runs a VINDEX3 container's program, and this session is not bound to one"
                )))
            }
        };
        let tokenizer = tokenizer.as_ref().ok_or_else(|| {
            LqlError::Execution(format!(
                "{statement} needs a tokenizer and this container carries no tokenizer.json — \
                 token-id capability only"
            ))
        })?;
        Ok((runtime, tokenizer, overlay, bos_token))
    }

    /// The token ids `text` encodes to, as the bound VINDEX3 container's statements encode it. With
    /// `special_tokens`, this is INFER's raw-prompt encoder (the encoding of every `INFER` and `CAPTURE` prompt):
    /// the tokenizer's own special tokens, plus the container's declared BOS where the tokenizer did not add it;
    /// no chat template. Without, the text's tokens alone: how `CAPTURE … CONTINUE "<text>"` encodes its text.
    pub fn tokenize(&self, text: &str, special_tokens: bool) -> Result<Vec<u32>, LqlError> {
        let (_, tokenizer, _, bos) = self.v3_text_program("TOKENIZE")?;
        if special_tokens {
            return Ok(encode_raw_prompt(tokenizer, text, bos)?);
        }
        Ok(tokenizer
            .encode(text, false)
            .map_err(|e| LqlError::exec("tokenize", e))?
            .get_ids()
            .to_vec())
    }

    /// The text `ids` decode to, special tokens kept: the decoding `CAPTURE` records for each position's token.
    pub fn decode(&self, ids: &[u32]) -> Result<String, LqlError> {
        let (_, tokenizer, _, _) = self.v3_text_program("DECODE")?;
        tokenizer
            .decode(ids, false)
            .map_err(|e| LqlError::exec("decode", e))
    }

    /// The typed operation behind `INFER "<prompt>" TOP <k>` on a VINDEX3 binding: the next-token
    /// distribution's top `top_k`, priced from the batch-prefill logits, and the KNN override where the
    /// session's KNN store answers.
    pub fn predict(&self, prompt: &str, top_k: usize) -> Result<Predictions, LqlError> {
        let (runtime, tokenizer, overlay, bos_token) = self.v3_text_program("INFER")?;
        let prompt_ids = encode_raw_prompt(tokenizer, prompt, bos_token)?;
        let start = std::time::Instant::now();
        // One observed pass over the same traversal generation
        // runs: logits for the display, per-layer residual taps
        // for the KnnStore override (only captured while the
        // overlay holds entries).
        let capture_residuals = !overlay.knn_store.is_empty();
        let mut residuals: Vec<(usize, Vec<f32>)> = Vec::new();
        let mut sink = |event: larql_inference::vindex3::PlaneEvent| {
            if capture_residuals {
                if let larql_inference::vindex3::PlaneEvent::Layer { index, trace } = event {
                    // Normed FFN inputs — the same tap the
                    // stored keys were captured from.
                    if let Some(last) = trace.ffn_input.last() {
                        residuals.push((index, last.clone()));
                    }
                }
            }
            Ok(())
        };
        // Compose edits and pending stored-weight ops reach
        // execution through the operand-source seam; without
        // either this is the plain pass, bit for bit.
        let overrides = execution_overrides(runtime, overlay)?;
        let output = match &overrides {
            Some(overrides) => runtime
                .execute_streaming_overlaid(
                    &prompt_ids,
                    overrides,
                    &v3_continuation(runtime)?,
                    &mut sink,
                )
                .map_err(|e| LqlError::exec("v3 prefill failed", e))?,
            None => runtime
                .execute_streaming(&prompt_ids, &v3_continuation(runtime)?, &mut sink)
                .map_err(|e| LqlError::exec("v3 prefill failed", e))?,
        };
        confirm_patched_weights_ran(overrides.as_ref())?;
        let logits = output
            .logits
            .ok_or_else(|| LqlError::Execution("the component carries no output head".into()))?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        let top: Vec<Prediction> = top_k_probs(&logits, top_k)
            .into_iter()
            .map(|(token_id, probability)| Prediction {
                token_id,
                token: tokenizer.decode(&[token_id], false).unwrap_or_default(),
                probability,
            })
            .collect();
        // The shared post-logits resolution rule — same
        // first-stored-layer, fixed-threshold gate V2 runs.
        let raw: Vec<(String, f64)> = top
            .iter()
            .map(|p| (p.token.clone(), p.probability as f64))
            .collect();
        let (_, knn_override) =
            larql_inference::apply_knn_override(raw, &residuals, Some(&overlay.knn_store), top_k);
        Ok(Predictions {
            prompt_ids,
            top,
            knn_override,
            elapsed_ms,
        })
    }

    /// The typed operation behind `INFER "<prompt>" GENERATE <n>` on a VINDEX3 binding: greedy continuation
    /// for up to `max_new` tokens, stopping before a built-in stop token.
    pub fn generate(&self, prompt: &str, max_new: usize) -> Result<Generated, LqlError> {
        let (runtime, tokenizer, overlay, bos_token) = self.v3_text_program("INFER")?;
        let prompt_ids = encode_raw_prompt(tokenizer, prompt, bos_token)?;
        let start = std::time::Instant::now();
        let (result, text) =
            generate::greedy_generate(runtime, overlay, tokenizer, &prompt_ids, max_new)?;
        Ok(Generated {
            prompt_ids,
            prompt_len: result.prompt_len,
            token_ids: result.tokens,
            text,
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
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
        match generate {
            None => {
                let predicted = self.predict(prompt, top_k)?;
                let mut out = Vec::new();
                out.push("Predictions (VINDEX3 program):".into());
                match &predicted.knn_override {
                    Some(ovr) => {
                        let model_top1 = predicted
                            .top
                            .first()
                            .map(|p| (p.token.clone(), p.probability as f64));
                        out.push(format!(
                            "   1. {:20} (100.00%, {})",
                            ovr.token,
                            crate::executor::helpers::format_knn_override_summary(
                                ovr,
                                model_top1.as_ref(),
                            ),
                        ));
                        for (i, p) in predicted
                            .top
                            .iter()
                            .take(top_k.saturating_sub(1))
                            .enumerate()
                        {
                            out.push(format!(
                                "  {:2}. {:20} ({:.2}%)  [id {}]",
                                i + 2,
                                p.token,
                                p.probability * 100.0,
                                p.token_id,
                            ));
                        }
                    }
                    None => {
                        for (i, p) in predicted.top.iter().enumerate() {
                            out.push(format!(
                                "  {:2}. {:20} ({:.2}%)  [id {}]",
                                i + 1,
                                p.token,
                                p.probability * 100.0,
                                p.token_id,
                            ));
                        }
                    }
                }
                out.push(format!("  {:.0}ms", predicted.elapsed_ms));
                if predicted.knn_override.is_some() {
                    out.push(
                        "  note: KNN override is a post-logits retrieval sidecar, not an \
                         FFN/residual edit."
                            .into(),
                    );
                }
                Ok(out)
            }
            Some(n) => {
                let generated = self.generate(prompt, n as usize)?;
                let ids: Vec<String> = generated.token_ids.iter().map(u32::to_string).collect();
                Ok(vec![
                    format!("Generated ({} tokens, greedy):", generated.token_ids.len()),
                    format!("  ids:  {}", ids.join(",")),
                    format!("  text: {}", generated.text),
                    format!(
                        "  prompt {} tokens, continued from position {}",
                        generated.prompt_len, generated.prompt_len,
                    ),
                    format!("  {:.0}ms", generated.elapsed_ms),
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
            format!("Capabilities:    {}", supported()),
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
