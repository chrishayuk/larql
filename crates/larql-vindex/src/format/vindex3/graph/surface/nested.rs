//! Execution surfaces from a nested component topology.

use larql_models::config::{
    Activation, FfnType, NormSpec, NormType, ParameterFreeQkNorm, QkNormScope,
};
use larql_models::inventory::components::ComponentTopology;

#[allow(unused_imports)]
use super::*;

/// Build the surface for a perception component from its nested config
/// reading plus tensor evidence.
///
/// A nested component has no detection trait to resolve through, so the
/// judged derivations live here, in one place: MHA (kv = q) unless
/// declared otherwise, `head_dim = hidden/heads` when divisible, canonical
/// 1/√d scaling, norm kind from which epsilon spelling the config
/// declares, and FFN gating from whether gate tensors exist under the
/// tower (`has_gate_tensors` — evidence, not a family fact).
pub fn surface_from_nested(
    nested: &ComponentTopology,
    has_gate_tensors: bool,
) -> Result<ExecutionSurface, Vec<String>> {
    let mut missing = Vec::new();
    // `None` stays distinct from a declared value: it can only answer
    // `head_dim` by derivation, and an absent width derives nothing.
    let hidden = nested.hidden_size.filter(|&h| h > 0);
    let heads = match nested.num_attention_heads {
        Some(h) if h > 0 => h,
        _ => {
            missing.push("num_attention_heads".to_string());
            0
        }
    };
    let head_dim = match nested.head_dim {
        Some(d) => d,
        None if heads > 0 && hidden.is_some_and(|h| h.is_multiple_of(heads)) => {
            hidden.unwrap_or_default() / heads
        }
        None => {
            // Two different defects reach here and must not read alike.
            // `heads == 0` is the sentinel for "no readable head count",
            // so formatting it into the arithmetic produced Qwen3.8's
            // "hidden 1152 not divisible by 0 heads" — a nonsense sum
            // standing in for an unread config spelling. A genuine
            // indivisibility is a different fact and keeps its own words.
            missing.push(match hidden {
                _ if heads == 0 => {
                    "head_dim (no readable attention-head count to derive it from)".to_string()
                }
                None => "head_dim (no readable hidden_size to derive it from)".to_string(),
                Some(h) => format!("head_dim (hidden {h} not divisible by {heads} heads)"),
            });
            0
        }
    };
    let intermediate_size = match nested.intermediate_size {
        Some(i) => i,
        None => {
            missing.push("intermediate_size".to_string());
            0
        }
    };
    let activation = match nested
        .hidden_act
        .as_deref()
        .and_then(Activation::from_hf_name)
    {
        Some(a) => a,
        None => {
            missing.push(format!(
                "hidden_act (declared {:?}, judged mapping required)",
                nested.hidden_act
            ));
            Activation::Gelu
        }
    };
    // The epsilon spelling the config declares names the norm kind; a
    // component declaring neither has no norm surface to persist.
    //
    // The message says both halves on purpose. A bare `norm_eps` reads
    // as one absent number a reader might reasonably supply, when what
    // is actually absent is the *kind* as well: the spelling is the
    // only evidence of whether this tower runs LayerNorm or RMSNorm,
    // so a config carrying neither has said nothing about its norm at
    // all. Qwen3.8's `vision_config` is exactly this — depth, heads,
    // widths and activation all declared, no epsilon key of either
    // spelling — and the honest answer is to refuse the surface rather
    // than pick an epsilon and a kind on the checkpoint's behalf.
    let (kind, eps) = match (nested.norm_kind, nested.norm_eps) {
        (Some(kind), Some(eps)) => (kind, eps),
        _ => {
            missing.push(
                "norm_eps (declares neither `layer_norm_eps` nor `rms_norm_eps`, so the \
                 norm kind is undeclared too — this build will not choose one)"
                    .to_string(),
            );
            (NormType::LayerNorm, 0.0)
        }
    };
    if !missing.is_empty() {
        return Err(missing);
    }
    Ok(ExecutionSurface {
        // A perception tower declares no sequence extent of its own; the
        // absence is the fact, not a zero.
        context_length: None,
        // No judged perception tower declares a linear-attention
        // recurrence, Multi-Latent Attention, or an SSM mixer.
        linear_attention: None,
        kda: None,
        kda_gate_lower_bound: None,
        kda_gate_form: None,
        kda_use_full_rank_gate: None,
        mla: None,
        mamba2: None,
        conv_qkv: None,
        residual_in_fp32: None,
        attention: Some(AttentionSurface {
            num_q_heads: heads,
            num_kv_heads: nested.num_key_value_heads.unwrap_or(heads),
            head_dim,
            // No perception tower has declared a query-scale operation.
            // `None` says that; `Some(1.0)` would claim the source
            // specifies a multiply by one.
            query_scale: None,
            score_scale: (head_dim as f64).powf(-0.5),
            logit_softcapping: None,
            qk_norm_scope: QkNormScope::PerHead,
            qk_norm_weight_offset: 0.0,
            parameter_free_qk_norm: ParameterFreeQkNorm::default(),
            output_gate: None,
            sinks: None,
            // What the tower declares about its projection biases, when
            // it declares anything (Gemma 4 vision: `false`); the loader's
            // tensor-presence check answers otherwise, as for text.
            attention_bias: nested.tower.attention_bias,
            // Vision towers declare only the all-four form.
            qkv_bias: None,
        }),
        ffn: Some(FfnSurface {
            // A perception tower's width is required above (its absence
            // is already a refusal), so it is always present here.
            intermediate_size: Some(intermediate_size),
            // A nested component declares its own per-layer widths, or none.
            intermediate_size_by_layer: nested.ffn_intermediate_size_by_layer.clone(),
            activation,
            ffn_type: if has_gate_tensors {
                FfnType::Gated
            } else {
                FfnType::Standard
            },
            // Nested towers declare no gate policy; plain gating is the
            // fact, not a fallback.
            gate_policy: larql_models::ExpertGatePolicy::Gated,
            moe: None,
            // A perception tower's operator set is not executed here;
            // its biases are unjudged.
            bias: None,
        }),
        norm: NormSurface {
            pre: NormSpec {
                kind,
                eps,
                weight_offset: 0.0,
            },
            // Unjudged, not "the same as `pre`". Perception towers have
            // no four-norm placement here, so nothing consumes it — and
            // claiming an equivalence nobody established is the
            // inherited-default failure this shape exists to prevent.
            post: None,
            final_norm: NormSpec {
                kind,
                eps,
                weight_offset: 0.0,
            },
            // Perception towers keep their own norm topology; placement
            // is a decoder-stack concept until the perception op set (5d)
            // defines its own.
            placement: None,
            bias: None,
        },
        head: None,
        // No perception tower has declared a residual-scale operation.
        residual_scale: None,
        // Nor a multi-stream residual: every judged tower adds its
        // sublayer outputs into one vector.
        residual_topology: larql_models::config::ResidualTopology::SingleStream,
    })
}

/// `serde` default for [`MlaSurface::query`]: the form every container
/// written before it was recorded declared.
pub(super) fn direct_query_form() -> larql_models::config::MlaQueryForm {
    larql_models::config::MlaQueryForm::Direct
}
