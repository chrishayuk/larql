//! Linear-attention and MLA surfaces, and surfaces from resolved facts.

use super::super::object::LogicalObject;
use larql_models::config::{Activation, AttentionGateSpec};
use larql_models::inventory::ArchitectureInventory;
use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use super::*;

/// What the Gated DeltaNet operator reads.
///
/// Mirrors [`LinearAttentionTopology`](larql_models::inventory::report::LinearAttentionTopology)
/// rather than reusing it, for the same reason [`AttentionSurface`] does not
/// reuse the resolved topology: the surface is the executor's contract and
/// may diverge from the architectural record. It does not, today.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinearAttentionSurface {
    /// Hk — query/key-side head count (16 on Qwen3.8).
    pub key_heads: usize,
    /// Dk (128).
    pub key_head_dim: usize,
    /// Hv — value-side head count (48). Distinct from [`Self::key_heads`]
    /// on purpose; no single head count describes this operator.
    pub value_heads: usize,
    /// Dv (128).
    pub value_head_dim: usize,
    /// Depthwise causal convolution width over the fused q|k|v channels (4).
    pub conv_kernel: usize,
    /// The precision the recurrence keeps its state at. Consumed: the
    /// reference operator allocates and accumulates its state at this
    /// precision rather than the model's bulk dtype.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_dtype: Option<larql_models::inventory::report::RecurrentStateDtype>,
}

impl LinearAttentionSurface {
    /// `2·Hk·Dk + Hv·Dv` — the fused projection's row count, and the
    /// channel count the depthwise convolution runs over. Derived so it
    /// cannot drift from the head counts.
    pub fn qkv_channels(self) -> usize {
        self.key_heads * self.key_head_dim * 2 + self.value_heads * self.value_head_dim
    }

    /// `Hv·Dv` — the value/gate width.
    pub fn value_width(self) -> usize {
        self.value_heads * self.value_head_dim
    }
}

/// What the Multi-Latent Attention operator reads.
///
/// Mirrors [`MlaExecution`](larql_models::inventory::report::MlaExecution)
/// rather than reusing it, for the same reason [`LinearAttentionSurface`]
/// does not reuse `LinearAttentionTopology`: the surface is the executor's
/// contract, and may diverge from the architectural record. It does not,
/// today.
// No `Eq`: `kv_a_norm_eps` is a float, and the operator's own epsilon
// is exactly the kind of fact whose equality is approximate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MlaSurface {
    /// Query/output head count — the decompressed K/V side always
    /// produces this many heads' worth of output.
    pub num_heads: usize,
    /// Compressed KV latent width.
    pub kv_lora_rank: usize,
    /// Non-RoPE portion of the query/key head width.
    pub qk_nope_head_dim: usize,
    /// RoPE portion of the query/key head width, one SHARED projection
    /// (MQA-style) across every head.
    pub qk_rope_head_dim: usize,
    /// Value head width — independent of the query/key head width.
    pub v_head_dim: usize,
    /// Epsilon of `kv_a_layernorm`, the latent norm applied between the
    /// compressed cache and its decompression — the FAMILY'S OWN value,
    /// which on Kimi Linear is `KimiRMSNorm`'s class default `1e-6` and
    /// not the layer's `rms_norm_eps` (`1e-5`).
    ///
    /// Carried on the surface because it is a per-operator norm site the
    /// component-level norm surface cannot speak for: the drill's F6, the
    /// one judged semantic the container could not carry, which lived as
    /// a constant inside a family-shaped executor and so could not
    /// survive deleting the checkpoint.
    ///
    /// `None` = unjudged for this family; the operator refuses rather
    /// than borrowing the layer eps. Absent on containers written before
    /// it was recorded, which is the same state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_a_norm_eps: Option<f64>,
    /// Which query these layers build: one dense `q_proj`, or Kimi-K3's
    /// `q_a_proj` -> `q_a_layernorm` -> `q_b_proj` under a declared
    /// `q_lora_rank`.
    ///
    /// A DECLARED fact, resolved from `q_lora_rank`'s presence and never
    /// from the operand estate — `q_proj` and `q_b_proj` have the same
    /// row count on K3 (`Hq*q_head_dim`, 18432) and differ only in their
    /// columns, so an estate-derived form would be decided by the very
    /// thing the form decides. Closure holds the shipped operands to
    /// this from both sides.
    ///
    /// Defaults to `Direct` on containers written before it was
    /// recorded, which is what those checkpoints declared.
    #[serde(default = "direct_query_form")]
    pub query: larql_models::config::MlaQueryForm,
    /// The output gate the checkpoint declares on its MLA layers
    /// (`mla_use_output_gate: true`): `sigmoid(g_proj(x)) ⊙ attn_value`
    /// before `o_proj`, the same generic operation
    /// [`AttentionSurface::output_gate`] carries for the softmax family, at
    /// width `Hq·v_head_dim`. `None` = no gate (undeclared, or declared
    /// `false`; the reference's default is none). Absent on containers
    /// written before it was recorded, which is the same state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_gate: Option<AttentionGateSpec>,
}

impl MlaSurface {
    /// `qk_nope_head_dim + qk_rope_head_dim` — one query/key head's full
    /// width, and the row width `self_attn.q_proj.weight` is fused at
    /// (`num_heads · q_head_dim`).
    pub fn q_head_dim(self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
}

/// What the Mamba2/SSD mixer reads.
///
/// The geometry is reused from the architectural record directly (the
/// same way [`ExecutionSurface::kda`] reuses
/// [`KdaGeometry`](larql_models::config::KdaGeometry)) — every field is
/// something the operator reads, and the struct already refuses partial
/// declarations at the parse boundary. The activation sits beside it
/// because a mixer-only component has no FFN surface to carry
/// `hidden_act`, and the mixer genuinely consumes it (the conv branch and
/// the output gate both apply it).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Mamba2Surface {
    pub geometry: larql_models::config::Mamba2Geometry,
    /// The mixer's nonlinearity (`hidden_act`, SiLU on every judged
    /// checkpoint — read, never assumed).
    pub activation: Activation,
}

/// Build the surface for a text-path component (target/drafter) from its
/// inventory's resolution. Returns the missing source facts when the
/// surface cannot be completed — the caller turns those into blocking
/// findings, never into defaults.
pub fn surface_from_resolved(
    inventory: &ArchitectureInventory,
) -> Result<ExecutionSurface, Vec<String>> {
    let resolved = &inventory.resolved;
    let Some(execution) = &resolved.execution else {
        return Err(vec![
            "resolved.execution (pre-v3 inventory — re-run inspect-hf)".to_string(),
        ]);
    };
    // Presence follows the program (schema 6). A layer that declares no
    // per-layer kind is an attention-class layer — that is what absence
    // means on every judged transformer config — so the surface is
    // present unless EVERY layer declares a recurrence. The FFN follows
    // the declared width, and a wholly-routed family declares its width
    // in the ROUTED spelling: `Qwen3_5MoeTextConfig` has no
    // `intermediate_size` field at all (it is `@strict`, and every layer
    // is a `Qwen3_5MoeSparseMoeBlock`), so reading only the dense
    // spelling graded a 397B MoE as having no FFN op and sent both
    // `hidden_act` and `num_experts_per_tok` to "no built component
    // answered the probe". A mixer-only stack declares NEITHER width and
    // still has no FFN — writing one there is the F1 fabrication one op
    // over.
    let attends = resolved.layers.is_empty()
        || resolved.layers.iter().any(|l| {
            !matches!(
                l.declared_kind,
                Some(larql_models::config::LayerKind::Recurrent(_))
            )
        });
    let dense_ffn_width = (resolved.intermediate_size > 0).then_some(resolved.intermediate_size);
    let routed_ffn_width = execution
        .moe
        .filter(|m| m.expert_intermediate_size > 0)
        .map(|m| m.expert_intermediate_size);
    let has_ffn = dense_ffn_width.is_some() || routed_ffn_width.is_some();
    // The surface carries the component's declared head geometry; a
    // family that varies it by layer (Gemma 4's global layers) records
    // each layer's geometry on its `AttentionLayerPolicy`, and the op
    // plan reads the layer's, so nothing here is averaged away.
    Ok(ExecutionSurface {
        // The declared extent of the programme, transcribed from the
        // judged execution semantic. Not read from the tokenizer's
        // `model_max_length`, which is a serving bound on a different
        // component and would be a second authority for one fact.
        context_length: resolved.context_length.map(|v| v as u64),
        // Carried from the architectural record, not re-derived. `None`
        // when the model declares no recurrence — every layer attends by
        // softmax and the operator is never reached.
        linear_attention: resolved.linear_attention.map(|t| LinearAttentionSurface {
            key_heads: t.key_heads,
            key_head_dim: t.key_head_dim,
            value_heads: t.value_heads,
            value_head_dim: t.value_head_dim,
            conv_kernel: t.conv_kernel,
            state_dtype: t.state_dtype,
        }),
        kda: resolved.kda,
        kda_gate_lower_bound: resolved.kda_gate_lower_bound,
        kda_gate_form: resolved.kda_gate_form,
        kda_use_full_rank_gate: resolved.kda_use_full_rank_gate,
        mamba2: resolved.mamba2.map(|geometry| Mamba2Surface {
            geometry,
            activation: execution.activation,
        }),
        conv_qkv: resolved.conv_qkv_attn,
        residual_in_fp32: execution.residual_in_fp32,
        // Absent means this build has not judged what the checkpoint
        // declares. Refusing is still the point — the alternative is a
        // four-stream model quietly served as a one-stream one — but the
        // reason must NOT say the declaration is incomplete.
        //
        // A header census of Hy4-preview read its surface directly:
        // `hc_pre.hc_fn` is `[2 * hc, hc * d]` with a two-entry scale and
        // no combination block, the config carries `hc_magnitude` and no
        // `hc_sinkhorn_iters`. That is a COMPLETE declaration of a
        // Sinkhorn-free hyper-connection, not a half-written Sinkhorn
        // one. Calling it partial sends the next reader to finish a
        // declaration nothing is missing from.
        //
        // The reason is READ, not written here. Since K3-ATTNRES-1 there
        // are two ways to resolve to nothing — a partial Sinkhorn
        // declaration, and a checkpoint declaring two whole topologies at
        // once — and they send a reader to opposite places. A single
        // hardcoded sentence told the second case to go and find a
        // missing iteration count. The architecture decided it and its
        // words travel with the absence.
        residual_topology: match execution.residual_topology {
            Some(topology) => topology,
            None => {
                return Err(vec![format!(
                    "residual topology ({})",
                    execution.residual_topology_refusal.as_deref().unwrap_or(
                        "the declaration resolved to no judged topology, and this inventory \
                         predates the field that carries why — re-run inspect-hf"
                    )
                )])
            }
        },
        mla: execution.mla.map(|m| MlaSurface {
            num_heads: m.num_heads,
            kv_lora_rank: m.kv_lora_rank,
            qk_nope_head_dim: m.qk_nope_head_dim,
            qk_rope_head_dim: m.qk_rope_head_dim,
            v_head_dim: m.v_head_dim,
            kv_a_norm_eps: m.kv_a_norm_eps,
            query: m.query,
            output_gate: m.output_gate,
        }),
        attention: attends.then_some(AttentionSurface {
            num_q_heads: resolved.num_q_heads,
            num_kv_heads: resolved.num_kv_heads,
            head_dim: resolved.head_dim,
            query_scale: execution.query_scale,
            score_scale: execution.score_scale,
            logit_softcapping: execution.attn_logit_softcapping,
            qk_norm_scope: execution.qk_norm_scope,
            qk_norm_weight_offset: execution.qk_norm_weight_offset,
            parameter_free_qk_norm: execution.parameter_free_qk_norm,
            // Judged per model; never inferred from operand presence.
            output_gate: execution.attention_output_gate,
            sinks: execution.attention_sinks,
            attention_bias: execution.attention_bias,
            qkv_bias: execution.qkv_bias,
        }),
        ffn: has_ffn.then(|| FfnSurface {
            intermediate_size: dense_ffn_width,
            intermediate_size_by_layer: resolved.ffn_intermediate_size_by_layer.clone(),
            activation: execution.activation,
            ffn_type: execution.ffn_type,
            gate_policy: execution.gate_policy,
            moe: execution.moe.map(|m| MoeSurface {
                branch_scale: m.branch_scale,
                dense_prefix_layers: m.dense_prefix_layers,
                experts: m.experts,
                top_k: m.top_k,
                expert_intermediate_size: m.expert_intermediate_size,
                router_kind: m.router_kind,
                routing_policy: m.routing_policy,
                router_bias: m.router_bias,
                expert_format: m.expert_format,
                gate_up_layout: m.gate_up_layout,
                shared_experts: m.shared_experts,
                shared_expert_intermediate_size: m.shared_expert_intermediate_size,
                shared_expert_gate: m.shared_expert_gate,
                hybrid: m.hybrid,
                latent: match m.routed_expert_form {
                    larql_models::config::RoutedExpertForm::Uniform => None,
                    larql_models::config::RoutedExpertForm::Latent { width, norm } => {
                        Some(MoeLatent {
                            width,
                            norm: norm.map(|n| LatentNorm { eps: n.eps }),
                        })
                    }
                },
            }),
            bias: execution.ffn_bias,
        }),
        norm: NormSurface {
            pre: execution.norm_pre,
            post: execution.norm_post,
            final_norm: execution.norm_final,
            // From operand evidence via `attach_stack_evidence`, once the
            // builder knows the component's stack object.
            placement: None,
            bias: execution.norm_bias,
        },
        // Attached by the builder once it knows the component's objects.
        head: None,
        residual_scale: execution.residual_scale,
    })
}

/// Attach the facts only the stack's operand estate can state: norm
/// placement from the norm-role evidence across the stack's bindings.
/// Shared by the builder and the G4 re-derivation, so the two can never
/// judge the same bytes differently.
pub fn attach_stack_evidence(
    surface: &mut ExecutionSurface,
    inventory: &ArchitectureInventory,
    stack: &LogicalObject,
) -> Result<(), Vec<String>> {
    let relative: Vec<String> = stack
        .source_bindings
        .iter()
        .flat_map(|binding| {
            inventory
                .tensors
                .tensors
                .iter()
                .filter_map(|t| t.name.strip_prefix(&binding.tensor_prefix))
                .map(|rest| rest.trim_start_matches('.').to_string())
        })
        .collect();
    // A mixer-only program (every layer declared a Mamba2 recurrence)
    // reads its own placement evidence: one pre-mixer norm per layer, no
    // attention/FFN wrap norms. The choice is made from the DECLARED
    // program, so a transformer stack that lost its norms still fails the
    // transformer evidence rather than sliding into the mixer's.
    // A hybrid's attention layers carry the same single pre-mixer norm
    // (the mamba_ssm lineage wraps EVERY block, mixer or attention, in
    // one `norm.weight`), so a Full layer counts as mixer-normed exactly
    // when the conv-QKV block is declared — a transformer's Full layer
    // still reads the transformer evidence.
    let mixer_only = !inventory.resolved.layers.is_empty()
        && inventory
            .resolved
            .layers
            .iter()
            .all(|l| match l.declared_kind {
                Some(larql_models::config::LayerKind::Recurrent(
                    larql_models::config::RecurrenceFamily::Mamba2,
                )) => true,
                Some(larql_models::config::LayerKind::Full) => {
                    inventory.resolved.conv_qkv_attn.is_some()
                }
                _ => false,
            });
    let evidence = if mixer_only {
        super::super::roles::mixer_norm_placement_evidence(relative.iter().map(String::as_str))
    } else {
        super::super::roles::norm_placement_evidence(relative.iter().map(String::as_str))
    };
    match evidence {
        Ok(placement) => {
            surface.norm.placement = Some(placement);
            // A post-norm stack's ONLY norm sites are the post ones, and
            // the component declares exactly one epsilon. So the declared
            // epsilon is theirs.
            //
            // This is not the four-norm case wearing a different hat, and
            // the difference is why it is safe here and refused there. A
            // four-norm stack HAS both sites and they can genuinely
            // differ — Muse-Glimmer's are 1e-5 pre and 1e-8 post — so
            // filling `post` from `pre` there would be inheriting one
            // site's judged value into another's, which is the
            // executable-but-unfounded failure. Here there is no second
            // site to disagree with: reading the declaration as belonging
            // to the pre sites would give an epsilon to norms this stack
            // does not have and none to the norms it does.
            if placement == super::super::roles::NormPlacement::PostOnly
                && surface.norm.post.is_none()
            {
                surface.norm.post = Some(surface.norm.pre);
            }
            Ok(())
        }
        Err(reason) => Err(vec![format!("norm placement ({reason})")]),
    }
}

/// The head surface for a component that owns embedding/output-head
/// objects, from the same resolution.
pub fn head_from_resolved(inventory: &ArchitectureInventory) -> Result<HeadSurface, Vec<String>> {
    let resolved = &inventory.resolved;
    let mut missing = Vec::new();
    let Some(execution) = &resolved.execution else {
        return Err(vec![
            "resolved.execution (pre-v3 inventory — re-run inspect-hf)".to_string(),
        ]);
    };
    let Some(vocab_size) = resolved.vocab_size else {
        missing.push("vocab_size".to_string());
        return Err(missing);
    };
    // The mamba_ssm lineage pads the embedding rows up to a declared
    // multiple, and the tied head genuinely EMITS the padded width — the
    // reference's own logits are that wide. The surface carries what the
    // head does; the declared vocab remains on the resolution as the
    // meaningful prefix.
    let vocab_size = match resolved.pad_vocab_size_multiple {
        Some(multiple) if multiple > 0 => vocab_size.div_ceil(multiple) * multiple,
        _ => vocab_size,
    };
    Ok(HeadSurface {
        vocab_size,
        embedding_norm: execution.embedding_norm,
        embed_scale: execution.embed_scale,
        output_multiplier: execution.output_multiplier,
        final_logit_softcapping: execution.final_logit_softcapping,
        head_reuses_embedding: execution.head_reuses_embedding,
        input_position: execution.input_position,
    })
}
