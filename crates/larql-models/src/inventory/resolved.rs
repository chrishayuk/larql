//! The resolved half of the inventory: what this build's detection would
//! actually run for the checkpoint.
//!
//! Everything here goes through the public detection surface
//! ([`crate::detect_from_json`], the registry) — no re-implementation of any
//! resolution rule. The point is to report the serving path's own answers,
//! so a wrong answer (a generic fallback running full attention everywhere,
//! a defaulted rope base) appears *as the serving path would produce it*,
//! next to the declared facts it disagrees with.

use serde_json::Value;

use crate::detect::{detect_from_json, find_architecture};

use super::report::{
    AttentionSummary, Detection, Identity, LayerPolicy, MlaExecution, MoeExecution,
    ResolvedExecution, ResolvedTopology,
};
use crate::config::ModelArchitecture;

/// Attention-kind labels for [`Detection::attention_kind`].
const ATTENTION_SLIDING: &str = "sliding";
const ATTENTION_FULL: &str = "full";

/// Read the checkpoint's identity facts straight from the config value.
pub fn read_identity(config: &Value) -> Identity {
    let text_config = config.get("text_config").unwrap_or(config);
    // The mamba_ssm-native spelling of the same fact: no `model_type`
    // key exists, and `ssm_cfg.layer: "Mamba2"` is that package's
    // identity declaration — the same judgment the parser makes, made
    // here so detection and identity cannot disagree about it.
    let model_type = text_config["model_type"]
        .as_str()
        .or_else(|| config["model_type"].as_str())
        .or_else(|| {
            (text_config["ssm_cfg"]["layer"].as_str() == Some("Mamba2")).then_some("mamba2")
        })
        .unwrap_or("")
        .to_string();
    let architectures = config["architectures"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let dtype = config["dtype"]
        .as_str()
        .or_else(|| config["torch_dtype"].as_str())
        .map(str::to_string);
    let transformers_version = config["transformers_version"].as_str().map(str::to_string);
    // Nested component configs: any top-level object value whose key ends in
    // `_config` (`text_config`, `vision_config`, `language_config`, …).
    let components = config
        .as_object()
        .map(|map| {
            map.iter()
                .filter(|(k, v)| v.is_object() && k.ends_with("_config"))
                .map(|(k, _)| k.clone())
                .collect()
        })
        .unwrap_or_default();
    Identity {
        model_type,
        architectures,
        dtype,
        transformers_version,
        components,
    }
}

/// Run detection and describe what came back.
pub fn resolve(config: &Value, identity: &Identity) -> (Detection, ResolvedTopology) {
    resolve_with_tensor_evidence(config, identity, &[])
}

/// [`resolve`], with the tensor estate available as evidence for the one
/// judgment a config alone cannot make: an `attention_layers_idx` set
/// that fits both index bases (see
/// [`disambiguate_attention_set_by_mixer_shape`]). Everything else is
/// identical — tensor evidence never *overrides* a declaration, it only
/// settles a declared ambiguity, and the settlement is recorded in the
/// resolution's provenance.
pub fn resolve_with_tensor_evidence(
    config: &Value,
    identity: &Identity,
    tensors: &[super::report::TensorFact],
) -> (Detection, ResolvedTopology) {
    let arch = detect_from_json(config);
    let registry_entry = find_architecture(&identity.model_type);
    let validation_errors = match arch.validate() {
        Ok(()) => Vec::new(),
        Err(errors) => errors.iter().map(|e| format!("{e:?}")).collect(),
    };
    let detection = Detection {
        family: arch.family().to_string(),
        generic_fallback: registry_entry.is_none(),
        // `AttentionKind` serialises to its lowercase tag; reuse that rather
        // than inventing a second spelling here.
        attention_kind: registry_entry.and_then(|e| {
            serde_json::to_value(e.attention_kind)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string))
        }),
        validation_errors,
    };

    let cfg = arch.config();
    let disambiguated = disambiguate_attention_set_by_mixer_shape(config, cfg, tensors);
    // The checkpoint's per-layer declaration in ONE vocabulary, whichever
    // spelling it used. `layer_types` wins when present — it is the direct
    // form, and GLM-5.3-Flash writes both — otherwise the index-set form
    // (`linear_attn_config`) answers, which is the only form Kimi Linear
    // writes. Without this fallback Kimi declares nothing per layer and
    // every one of its 20 recurrent layers falls to the resolved boolean's
    // default: a 27-layer full-attention tower for a hybrid model
    // (`docs/glm5-flash-funnel.md` §4.5).
    // The declared per-layer topology, canonicalised. One resolution
    // covers every spelling (`layer_types`, the two-set `linear_attn_config`
    // form, Inkling's `local_layer_ids` with an implied complement), so no
    // consumer below needs to know which one the checkpoint used.
    //
    // An UNRESOLVED declaration is not an absent one: every layer is
    // marked with a spelling outside the executable vocabulary, so the
    // plan blocks instead of letting the resolved boolean answer for a
    // topology the checkpoint actually stated.
    // A family whose `model_type` is itself the whole-stack declaration
    // (pure SSM — no interleave exists to state) answers when no
    // interleave was declared. The interleave stays authoritative when
    // both exist: an explicit per-layer statement outranks a uniform one.
    // The uniform answer only speaks for a stack that declared NO
    // interleave. A declared-but-unresolved one (OuteAI's ambiguous-base
    // `attention_layers_idx` is the live case) must stay unresolved:
    // letting the family's uniform kind answer for it would silently
    // re-declare the very layers the interleave singles out.
    let declared_kinds: Option<Vec<crate::config::LayerKind>> = cfg
        .linear_attn_interleave
        .resolved()
        .or(disambiguated.as_ref())
        .map(|r| r.layers.clone())
        .or_else(|| {
            matches!(
                cfg.linear_attn_interleave,
                crate::config::DeclaredInterleave::Absent
            )
            .then(|| {
                arch.declared_uniform_layer_kind()
                    .map(|kind| vec![kind; cfg.num_layers])
            })
            .flatten()
        });
    let declared_spans: Option<Vec<String>> = declared_kinds
        .as_ref()
        .map(|kinds| kinds.iter().map(layer_kind_spelling).collect())
        .or_else(|| {
            cfg.linear_attn_interleave.error().map(|_| {
                vec![crate::config::LAYER_TYPE_UNRESOLVED_INTERLEAVE.to_string(); cfg.num_layers]
            })
        });
    let layers: Vec<LayerPolicy> = (0..cfg.num_layers)
        .map(|layer| {
            let sliding = arch.is_sliding_window_layer(layer);
            LayerPolicy {
                layer,
                attention: if sliding {
                    ATTENTION_SLIDING
                } else {
                    ATTENTION_FULL
                }
                .to_string(),
                declared_span: declared_spans
                    .as_ref()
                    .and_then(|types| types.get(layer))
                    .cloned(),
                declared_kind: declared_kinds.as_ref().and_then(|k| k.get(layer)).cloned(),
                window: if sliding {
                    arch.sliding_window_size()
                } else {
                    None
                },
                // On a hybrid, a resolved Full layer rotates as the
                // conv-QKV block declares — the leading `rope_emb_dim`
                // dims of each head, frequencies over the rotary width
                // (transcribed from the reference's GPTNeoX-style
                // partial-rotary application). The family's per-layer
                // answer (None — recurrence carries position) still
                // speaks for every mixer layer.
                position: match (cfg.conv_qkv_attn, declared_kinds.as_ref()) {
                    (Some(attn), Some(kinds))
                        if matches!(kinds.get(layer), Some(crate::config::LayerKind::Full)) =>
                    {
                        crate::config::PositionPolicy::PartialRope {
                            theta: attn.rope_theta,
                            rotary_fraction: attn.rotary_dim as f64 / attn.head_dim as f64,
                            basis: crate::config::RotaryFrequencyBasis::RotaryWidth,
                        }
                    }
                    _ => arch.position_policy_for_layer(layer),
                },
                head_dim: arch.head_dim_for_layer(layer),
                num_kv_heads: arch.num_kv_heads_for_layer(layer),
                v_from_k: arch.v_shares_k(layer),
                expert_bank: expert_bank_prefix(arch.as_ref(), layer),
            }
        })
        .collect();
    let sliding_layers = layers
        .iter()
        .filter(|l| l.attention == ATTENTION_SLIDING)
        .count();
    // Every semantic decision the serving path would make, resolved once
    // and recorded — the executor downstream reads, never defaults.
    //
    // Absence stays absence here. An identity default (`query_scale` 1.0,
    // `output_multiplier` 1.0) is numerically plausible but semantically
    // indistinguishable from a real declaration, so an ingestion
    // regression would produce a fully executable *wrong* program rather
    // than a loud one. Only a judgment may turn absence into an operation.
    let execution = ResolvedExecution {
        query_scale: arch.qk_scale_factor(),
        score_scale: arch.attention_scale(),
        attn_logit_softcapping: arch.attn_logit_softcapping(),
        qk_norm_scope: arch.qk_norm_scope(),
        qk_norm_weight_offset: arch.qk_norm_weight_offset(),
        parameter_free_qk_norm: {
            let mut norms = arch.parameter_free_qk_norm();
            norms.v = arch.has_v_norm();
            norms
        },
        attention_output_gate: arch.attention_output_gate(),
        attention_sinks: arch.attention_sinks(),
        attention_bias: arch.attention_bias(),
        moe: arch.is_moe().then(|| MoeExecution {
            branch_scale: cfg.routed_scaling_factor,
            dense_prefix_layers: cfg.first_k_dense_replace,
            experts: arch.num_experts(),
            top_k: arch.num_experts_per_token(),
            expert_intermediate_size: arch.moe_intermediate_size(),
            router_kind: arch.moe_router_kind(),
            routing_policy: arch.expert_routing_policy(),
            router_bias: arch.moe_router_bias_key(0).is_some(),
            expert_format: arch.expert_format(),
            gate_up_layout: arch.gate_up_layout(),
            shared_experts: arch.num_shared_experts(),
            hybrid: arch.is_hybrid_moe(),
        }),
        // `uses_mla()` alone decides the LAYER'S OPERATOR (every
        // `LayerKind::Full` layer, in `graph::build::operator_and_span`);
        // this field answers a SEPARATE question — whether the geometry
        // to check its operands against fully resolved. A family that
        // declares MLA but omits one of the four dimensions gets `None`
        // here while its layers still classify as MLA, the same "declared
        // ≠ geometry resolved" split KDA's `Option<KdaGeometry>` already
        // makes — never a defaulted dimension standing in for a fact the
        // checkpoint never stated.
        mla: arch.uses_mla().then_some(()).and_then(|()| {
            let qk_nope_head_dim = arch.mla_qk_nope_head_dim()?;
            let qk_rope_head_dim = arch.mla_qk_rope_head_dim()?;
            let v_head_dim = arch.mla_v_head_dim()?;
            Some(MlaExecution {
                num_heads: cfg.num_q_heads,
                kv_lora_rank: arch.kv_lora_rank(),
                qk_nope_head_dim,
                qk_rope_head_dim,
                v_head_dim,
            })
        }),
        activation: arch.activation(),
        ffn_type: arch.ffn_type(),
        gate_policy: arch.expert_gate_policy(),
        norm_pre: arch.pre_norm_spec(),
        norm_post: arch.post_norm_spec(),
        norm_final: arch.final_norm_spec(),
        embedding_norm: arch.embedding_norm(),
        post_norms: arch.has_post_norms(),
        embed_scale: arch.embed_scale(),
        output_multiplier: arch.logit_scale(),
        final_logit_softcapping: arch.final_logit_softcapping(),
        residual_scale: arch.residual_scale(),
        residual_in_fp32: cfg.residual_in_fp32,
        head_reuses_embedding: arch.output_head_reuses_embedding(),
    };
    let topology = ResolvedTopology {
        num_layers: cfg.num_layers,
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        num_q_heads: cfg.num_q_heads,
        num_kv_heads: cfg.num_kv_heads,
        head_dim: cfg.head_dim,
        vocab_size: cfg.vocab_size,
        sliding_window: cfg.sliding_window,
        attention: AttentionSummary {
            sliding_layers,
            full_layers: cfg.num_layers - sliding_layers,
        },
        layers,
        execution: Some(execution),
        // Present only when the model declares a complete recurrence.
        // A partial declaration resolves to `None` rather than being
        // completed with defaults — see `LinearAttentionTopology::from_config`.
        linear_attention: crate::inventory::report::LinearAttentionTopology::from_config(cfg),
        kda: cfg.kda_geometry,
        kda_gate_lower_bound: cfg.kda_gate_lower_bound,
        mamba2: cfg.mamba2_geometry,
        mamba2_provenance: cfg.mamba2_provenance.clone(),
        conv_qkv_attn: cfg.conv_qkv_attn,
        conv_qkv_provenance: cfg.conv_qkv_provenance.clone(),
    };
    (detection, topology)
}

/// The architecture-relative prefix of a layer's packed expert bank: the
/// parent of the family's fused `gate_up` operand key. Asked of the arch,
/// never inferred from a substring — the family names its own operands.
/// [`bind_expert_banks`] resolves it to the source spelling once the
/// tensor names are known.
fn expert_bank_prefix(arch: &dyn ModelArchitecture, layer: usize) -> Option<String> {
    if let Some(key) = arch
        .packed_gate_up_blocks_key(layer)
        .or_else(|| arch.packed_experts_gate_up_key(layer))
    {
        return key.rsplit_once('.').map(|(parent, _)| parent.to_string());
    }
    per_expert_bank_prefix(arch, layer)
}

/// The common ancestor of every expert's own gate/up/down tensors, for a
/// checkpoint that ships them as `experts` wholly separate tensors rather
/// than one packed bank (`ExpertFormat::PerExpert` — Kimi Linear, Mixtral,
/// DeepSeek). Without this, [`bind_expert_banks`] never carves a
/// per-expert layer's bytes out of the decoder stack — they are real
/// bytes sitting in the wrong object, not a missing fact.
///
/// Derived from evidence, never a fixed count of path segments to strip:
/// the architecture's own key for expert 0 and expert 1 diverge at
/// exactly the expert-index segment (`"…experts.0.w1.weight"` vs
/// `"…experts.1.w1.weight"`), so their longest common BYTE prefix, proven
/// to end at a `.` boundary, is the parent every expert's operands
/// share — a substring collision (`experts.1` inside `experts.10`) cannot
/// produce a false match here because the two probe keys are fixed at
/// indices 0 and 1, which can never be one a prefix of the other.
fn per_expert_bank_prefix(arch: &dyn ModelArchitecture, layer: usize) -> Option<String> {
    if arch.expert_format() != crate::config::ExpertFormat::PerExpert || arch.num_experts() < 2 {
        return None;
    }
    let first = arch.expert_ffn_gate_key(layer, 0)?;
    let second = arch.expert_ffn_gate_key(layer, 1)?;
    let common = first
        .bytes()
        .zip(second.bytes())
        .take_while(|(a, b)| a == b)
        .count();
    (common > 0 && first.as_bytes()[common - 1] == b'.').then(|| first[..common - 1].to_string())
}

/// Resolve each layer's arch-relative expert-bank prefix to the source
/// name the checkpoint actually spells (`layers.3.mlp.experts` →
/// `model.layers.3.mlp.experts`): the tensor whose name ends with the
/// arch prefix at a segment boundary names it. A bank the arch declares
/// but no tensor spells resolves to `None` — the layer is then not
/// routed by evidence, and closure says so.
pub fn bind_expert_banks(topology: &mut ResolvedTopology, tensors: &[super::report::TensorFact]) {
    for layer in &mut topology.layers {
        let Some(relative) = layer.expert_bank.take() else {
            continue;
        };
        let dotted = format!("{relative}.");
        layer.expert_bank = tensors
            .iter()
            .filter_map(|t| {
                // `…{relative}.{leaf}` at a segment boundary: the source
                // prefix is everything before `{relative}.{leaf}` plus
                // `{relative}` itself.
                let at = t.name.find(&dotted)?;
                let boundary_ok = at == 0 || t.name.as_bytes()[at - 1] == b'.';
                boundary_ok.then(|| t.name[..at + relative.len()].to_string())
            })
            .next();
    }
}

/// The `layer_types` spelling one canonical kind corresponds to.
///
/// The bridge between the canonical vocabulary and the one every existing
/// consumer already speaks, so a new spelling reaches them without each
/// growing a second code path.
fn layer_kind_spelling(kind: &crate::config::LayerKind) -> String {
    use crate::config::LayerKind;
    match kind {
        LayerKind::Full => crate::config::LAYER_TYPE_FULL_ATTENTION,
        LayerKind::Sliding { .. } => crate::config::LAYER_TYPE_SLIDING_ATTENTION,
        LayerKind::Recurrent(_) => crate::config::LAYER_TYPE_LINEAR_ATTENTION,
        // Carried verbatim, so the report shows what the checkpoint
        // actually said and the layer fails to round-trip to anything
        // executable — which is what makes it block.
        LayerKind::Unexpressed { declared } => return declared.clone(),
    }
    .to_string()
}

/// J5 — settle a declared index-base ambiguity from the tensor estate.
///
/// OuteAI's `attention_layers_idx: [6,12,18,24]` over 32 layers places
/// validly under BOTH index bases, so the interleave resolver honestly
/// answers [`InterleaveError::AmbiguousBase`]
/// (`crate::config::InterleaveError`) — the declaration does not
/// determine its own reading. The tensor estate does: an attention
/// mixer's fused-QKV `in_proj` row count
/// ([`ConvQkvAttnGeometry::qkv_rows`](crate::config::ConvQkvAttnGeometry::qkv_rows))
/// differs from a Mamba2 mixer's
/// ([`Mamba2Geometry::in_proj_rows`](crate::config::Mamba2Geometry::in_proj_rows)),
/// and every layer carries exactly one of the two.
///
/// Each base is attempted against every layer's observed `mixer.in_proj`
/// rows; exactly one consistent base resolves, with the evidence recorded
/// in the provenance's sources. Zero or two consistent bases (or absent
/// geometry, or missing tensors) leave the declaration unresolved — this
/// pass settles ambiguity, it never invents an answer.
fn disambiguate_attention_set_by_mixer_shape(
    config: &Value,
    cfg: &crate::config::ModelConfig,
    tensors: &[super::report::TensorFact],
) -> Option<crate::config::ResolvedInterleave> {
    use crate::config::{
        InterleaveEncoding, InterleaveError, InterleaveProvenance, LayerIndexBase, LayerKind,
        RecurrenceFamily,
    };
    if !matches!(
        cfg.linear_attn_interleave.error(),
        Some(InterleaveError::AmbiguousBase { .. })
    ) {
        return None;
    }
    let mamba2 = cfg.mamba2_geometry?;
    let conv_qkv = cfg.conv_qkv_attn?;
    let text_config = config.get("text_config").unwrap_or(config);
    let (key, set) = ["attention_layers_idx", "attn_layer_idx"]
        .into_iter()
        .find_map(|key| {
            let indices: Vec<i64> = text_config
                .get(key)?
                .as_array()?
                .iter()
                .filter_map(Value::as_i64)
                .collect();
            (!indices.is_empty()).then_some((key, indices))
        })?;
    // Observed `mixer.in_proj` rows per layer, matched on the exact
    // dotted path so `layers.6` never matches `layers.16`.
    let in_proj_rows = |layer: usize| -> Option<usize> {
        let suffix = format!("layers.{layer}.mixer.in_proj.weight");
        tensors
            .iter()
            .find(|t| {
                t.name.ends_with(&suffix)
                    && (t.name.len() == suffix.len()
                        || t.name.as_bytes()[t.name.len() - suffix.len() - 1] == b'.')
            })
            .and_then(|t| t.shape.first().copied())
    };
    let attention_rows = conv_qkv.qkv_rows();
    let mamba_rows = mamba2.in_proj_rows(cfg.hidden_size);
    if attention_rows == mamba_rows {
        // The two mixers are indistinguishable by this evidence.
        return None;
    }
    let consistent = |base: LayerIndexBase| -> bool {
        let full: Vec<usize> = set
            .iter()
            .filter_map(|declared| usize::try_from(declared - base.offset()).ok())
            .collect();
        if full.len() != set.len() || full.iter().any(|l| *l >= cfg.num_layers) {
            return false;
        }
        (0..cfg.num_layers).all(|layer| {
            let expected = if full.contains(&layer) {
                attention_rows
            } else {
                mamba_rows
            };
            in_proj_rows(layer) == Some(expected)
        })
    };
    let mut bases = LayerIndexBase::ALL.into_iter().filter(|b| consistent(*b));
    let (base, none) = (bases.next()?, bases.next());
    if none.is_some() {
        return None;
    }
    let layers = (0..cfg.num_layers)
        .map(|layer| {
            let declared = layer as i64 + base.offset();
            if set.contains(&declared) {
                LayerKind::Full
            } else {
                // Identified, not inferred from the key name: the base
                // proof above verified every complement layer's
                // `mixer.in_proj` rows against the DECLARED Mamba2
                // geometry — operand evidence of the mixer itself, the
                // same evidence class the spelling reader lacks.
                LayerKind::Recurrent(RecurrenceFamily::Mamba2)
            }
        })
        .collect();
    Some(crate::config::ResolvedInterleave {
        layer_count: cfg.num_layers,
        provenance: InterleaveProvenance {
            sources: vec![
                key.to_string(),
                format!(
                    "tensor-evidence: layers.N.mixer.in_proj rows \
                     ({attention_rows} attention vs {mamba_rows} mamba2) prove the base"
                ),
            ],
            encoding: InterleaveEncoding::ExplicitSetWithComplement,
            resolved_base: Some(base),
            scope: "target.decoder_stack".to_string(),
        },
        layers,
    })
}
