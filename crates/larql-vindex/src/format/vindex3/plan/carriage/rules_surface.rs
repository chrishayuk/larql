//! Interleave, norm, FFN, scaling and Gemma 4 carriage rules.

use super::*;

pub(super) const SURFACE_RULES: &[CarriageRule] = &[
    // ── The interleave, in the two-set spelling, and the KDA conv ────
    CarriageRule {
        leaf: "kda_layers",
        reaches: Carriage::Lowered,
        site: "Component.attention[].{operator,span} → LayerAttention::{Kda,GatedDelta,Softmax}",
        probe: Some(probe_recurrent_layer_set),
    },
    CarriageRule {
        leaf: "full_attn_layers",
        reaches: Carriage::Lowered,
        site: "Component.attention[].{operator,span} → LayerAttention::{Kda,GatedDelta,Softmax}",
        probe: Some(probe_softmax_layer_set),
    },
    CarriageRule {
        leaf: "gate_lower_bound",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.kda_gate_lower_bound → KdaOp.gate_lower_bound",
        probe: Some(probe_kda_gate_lower_bound),
    },
    CarriageRule {
        leaf: "short_conv_kernel_size",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.kda.conv_kernel → KdaOp.conv_kernel",
        probe: Some(probe_kda_conv_kernel),
    },
    // The KDA output gate's FORM (K3-REP-GATE-1). Lowered: the op carries
    // the form as a type, the executor projects the gate from whichever
    // operand the form names, and closure holds the shipped operands to
    // the declaration from both sides.
    CarriageRule {
        leaf: "use_full_rank_gate",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.kda_use_full_rank_gate → KdaOp.output_gate (KdaOutputGate::{LowRank,FullRank}) → exec::kda output-gate projection",
        probe: Some(probe_kda_use_full_rank_gate),
    },
    // A rescale of the whole routed branch, which this schema's MoE
    // surface has no field for. Refuses — and refusing for a stated reason
    // is the point of reading it: a key nothing reads blocks with no
    // account of why.
    CarriageRule {
        leaf: "routed_scaling_factor",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.ffn.moe.branch_scale",
        probe: Some(probe_moe_branch_scale),
    },
    // How many leading layers are dense. The op plan decides each layer's
    // FFN kind from operand evidence, but no field on the graph states the
    // prefix, so the declaration is not carried.
    CarriageRule {
        leaf: "first_k_dense_replace",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.ffn.moe.dense_prefix_layers",
        probe: Some(probe_moe_dense_prefix),
    },
    // Kimi Linear declares it true while carrying `qk_rope_head_dim: 64`,
    // so what it asserts about the rotary is not yet judged. Unjudged is
    // the honest verdict, and it blocks.
    CarriageRule {
        leaf: "mla_use_nope",
        reaches: Carriage::Represented,
        site: "Component.attention[].position → PositionPolicy::None",
        probe: Some(probe_mla_nope),
    },
    CarriageRule {
        leaf: "model_max_length",
        reaches: Carriage::Parsed,
        site: "no schema field — a KV-allocation bound, read by no generic op",
        probe: None,
    },
    CarriageRule {
        leaf: "num_nextn_predict_layers",
        reaches: Carriage::Represented,
        site: "no schema field — this schema has no multi-token-prediction object",
        // Zero declared layers is no MTP head, which this schema
        // represents exactly by carrying none. Any positive count is a
        // sub-stack it cannot state, and refuses.
        probe: Some(probe_absent_when_zero),
    },
    // The stride spelling of the per-layer span table ("every Nth layer
    // attends fully", Gemma 2/3). Read back as the period the BUILT table
    // shows — derived from the spans, never echoed from the declaration —
    // so a period the resolution did not honour mismatches, and a table
    // with no single period (or a non-integer spelling such as
    // EXAONE-4's `"LLLG"`) still refuses.
    CarriageRule {
        leaf: "sliding_window_pattern",
        reaches: Carriage::Lowered,
        site: "Component.attention[].span, read back as the period of its full-attention layers → AttentionOp.window",
        probe: Some(probe_full_attention_period),
    },
    // Gemma 3's sliding layers' RoPE base: the flat spelling of
    // `rope_parameters.sliding_attention.rope_theta`
    // (`Gemma3TextConfig.convert_rope_params_to_dict`). Scoped to the
    // sliding span by `plan::rope_scope`, and answered from those layers'
    // carried position policy exactly as the per-section spelling is.
    CarriageRule {
        leaf: "rope_local_base_freq",
        reaches: Carriage::Lowered,
        site: "Component.attention[].position on the sliding span (PositionPolicy::Rope) → AttentionOp.position",
        probe: Some(probe_rope_theta),
    },
    // ── Norms ───────────────────────────────────────────────────────
    CarriageRule {
        leaf: "rms_norm_eps",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.norm.pre.eps → NormOp.eps",
        probe: Some(probe_pre_norm_eps),
    },
    // LFM2 spells the same fact `norm_eps`. Its separate
    // `block_norm_eps` is NOT this fact and has no rule, so it keeps
    // refusing until something judges the FFN blocks it names.
    CarriageRule {
        leaf: "norm_eps",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.norm.pre.eps → NormOp.eps",
        probe: Some(probe_pre_norm_eps),
    },
    CarriageRule {
        leaf: "layer_norm_eps",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.norm.pre.eps → NormOp.eps",
        probe: Some(probe_pre_norm_eps),
    },
    CarriageRule {
        leaf: "norm_epsilon",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.norm.pre.eps → NormOp.eps",
        probe: Some(probe_pre_norm_eps),
    },
    CarriageRule {
        leaf: "layer_norm_epsilon",
        reaches: Carriage::Lowered,
        // GPT-2's spelling; `detect/parser.rs:292` folds it into the same
        // `norm_eps` read as its three siblings above.
        site: "ExecutionSurface.norm.pre.eps → NormOp.eps",
        probe: Some(probe_pre_norm_eps),
    },
    CarriageRule {
        leaf: "post_norm_eps",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.norm.post.eps → NormOp.eps at the post sites",
        probe: Some(probe_post_norm_eps),
    },
    // ── FFN ─────────────────────────────────────────────────────────
    CarriageRule {
        leaf: "hidden_act",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.ffn.activation → FfnOp.activation",
        probe: Some(probe_activation),
    },
    CarriageRule {
        leaf: "hidden_activation",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.ffn.activation → FfnOp.activation",
        probe: Some(probe_activation),
    },
    // Falcon's one-word FFN shape: `swiglu` is gated + SiLU, `geglu` is
    // gated + GELU, a plain nonlinearity name is the ungated shape. Two
    // surface facts answer for one declared word.
    CarriageRule {
        leaf: "activation",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.ffn.{ffn_type, activation} → FfnOp — the shape the word names",
        probe: Some(probe_ffn_shape_name),
    },
    // E30 static shards: a derived checkpoint declares each layer's dense
    // FFN width. Lowered: the planner shapes every layer's gate/up/down
    // against it and states it on that layer's `FfnOp`, which the CPU
    // paths and the Metal lowering read as the layer's intermediate width.
    CarriageRule {
        leaf: "larql_ffn_intermediate_size_by_layer",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.ffn.intermediate_size_by_layer → FfnOp.intermediate_size, per layer",
        probe: Some(probe_ffn_width_by_layer),
    },
    CarriageRule {
        leaf: "swiglu_limit",
        reaches: Carriage::Represented,
        // GPT-OSS's clamped GLU: `gate.min(limit)`, `up.clamp(±limit)`,
        // `(up + 1) * gate * sigmoid(alpha * gate)`. Carried as a gate
        // *policy* rather than an activation variant, and judged here by
        // the limit it carries. Represented, not lowered: the interpreter
        // and the lowering refuse a ClampedGlu FFN until A-9.3/A-9.4.
        site: "ExecutionSurface.ffn.gate_policy (ExpertGatePolicy::ClampedGlu.limit) → FfnOp.gate_policy",
        probe: Some(probe_swiglu_limit),
    },
    // Kimi-K3's SiTU-GLU softcaps. Parameters of the combine that
    // `hidden_act: "situ"` names — carried as a gate POLICY, for the same
    // reason `swiglu_limit` is: the bound changes the model, not the
    // nonlinearity. Lowered rather than represented, because unlike
    // ClampedGlu both the interpreter and the Metal lowering execute this
    // one, and a fact's claimed carriage must be the carriage witnessed.
    CarriageRule {
        leaf: "activation_situ_beta",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.ffn.gate_policy (ExpertGatePolicy::SituGlu.beta) → FfnOp.gate_policy",
        probe: Some(probe_situ_beta),
    },
    CarriageRule {
        leaf: "activation_situ_linear_beta",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.ffn.gate_policy (ExpertGatePolicy::SituGlu.linear_beta) → FfnOp.gate_policy",
        probe: Some(probe_situ_linear_beta),
    },
    // ── Attention/output scaling ────────────────────────────────────
    CarriageRule {
        leaf: "qk_scale_factor",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.attention.query_scale → AttentionOp.query_scale",
        probe: Some(probe_query_scale),
    },
    CarriageRule {
        leaf: "query_pre_attn_scalar",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.attention.score_scale → AttentionOp.score_scale",
        probe: Some(probe_score_scale),
    },
    CarriageRule {
        leaf: "attn_logit_softcapping",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.attention.logit_softcapping → AttentionOp.logit_softcapping",
        probe: Some(probe_attn_softcap),
    },
    CarriageRule {
        leaf: "final_logit_softcapping",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.head.final_logit_softcapping → OutputOp.softcapping",
        probe: Some(probe_final_softcap),
    },
    CarriageRule {
        leaf: "output_multiplier",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.head.output_multiplier → OutputOp.multiplier",
        probe: Some(probe_output_multiplier),
    },
    CarriageRule {
        leaf: "embedding_multiplier",
        reaches: Carriage::Lowered,
        // Granite's embedding-scale operation, wired through
        // `GraniteArch::embed_scale()` (`config/architecture.rs`) into
        // `HeadSurface.embed_scale` and on into `EmbeddingOp.scale`
        // (`opplan/build.rs`).
        site: "ExecutionSurface.head.embed_scale → EmbeddingOp.scale",
        probe: Some(probe_embed_scale),
    },
    CarriageRule {
        leaf: "attention_multiplier",
        reaches: Carriage::Lowered,
        // NOT `qk_scale_factor`/`query_scale` — Granite's attention_multiplier
        // *replaces* the standard 1/sqrt(head_dim) score scale rather than
        // multiplying on top of it (every legacy-path call site treats it
        // that way, and the declared value — 1/head_dim — confirms it
        // numerically). `ModelArchitecture::attention_scale`'s default
        // resolves it into `score_scale` accordingly.
        site: "ExecutionSurface.attention.score_scale → AttentionOp.score_scale",
        probe: Some(probe_score_scale),
    },
    CarriageRule {
        leaf: "logits_scaling",
        reaches: Carriage::Lowered,
        // Granite's spelling, and NOT a synonym: `logits_scaling` is a
        // divisor (`logits / d`) where `output_multiplier` is a multiplier.
        // Scaling does commute through the linear head, so the two describe
        // the same operation — but only once the divisor is inverted, which
        // `ModelArchitecture::logit_scale` does. The container therefore
        // carries `1/d`, and this probe inverts it back to compare against
        // the declared leaf.
        site: "ExecutionSurface.head.output_multiplier → OutputOp.multiplier (as 1/d)",
        probe: Some(probe_logits_scaling),
    },
    CarriageRule {
        leaf: "residual_multiplier",
        reaches: Carriage::Lowered,
        // Granite's residual-stream scale: the sublayer's own output
        // (attention or FFN) is multiplied by this before its residual
        // add, at both sites — no other family in this registry scales
        // the residual stream, so this is new schema (A-11.3), not a
        // second spelling of an existing field.
        site: "ExecutionSurface.residual_scale → LayerPlan.residual_scale",
        probe: Some(probe_residual_scale),
    },
    CarriageRule {
        leaf: "norm_topk_prob",
        reaches: Carriage::Represented,
        // Whether router weights are renormalised after top-k selection.
        // The cross-check this rule once said it lacked now exists: the
        // routing policy IS this flag, and `moe_renormalize` is the same
        // fact in Kimi Linear's spelling. See `probe_moe_routing_policy`
        // for why it reports rather than compares.
        site: "ExecutionSurface.ffn.moe.routing_policy",
        probe: Some(probe_moe_routing_policy),
    },
    CarriageRule {
        leaf: "num_experts_per_tok",
        reaches: Carriage::Lowered,
        // The canonical HF spelling of routing width — same underlying
        // resolved value as `top_k_experts`: `ModelArchitecture::num_experts_per_token()`
        // already bridges both spellings per family (GPT-OSS reads
        // `num_experts_per_token` directly; Gemma 4 tries `top_k_experts`
        // first, falling back to `num_experts_per_token` — confirmed by
        // reading both overrides), so the same probe answers both.
        site: "ExecutionSurface.ffn.moe.top_k → RoutedFfnOp routing",
        probe: Some(probe_moe_top_k),
    },
    CarriageRule {
        leaf: "num_experts_per_token",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.ffn.moe.top_k → RoutedFfnOp routing",
        probe: Some(probe_moe_top_k),
    },
    // ── Facts that stop at the parser, reviewed ─────────────────────
    CarriageRule {
        leaf: "attention_bias",
        reaches: Carriage::Represented,
        // A-9.1: the surface states it, and operand closure enforces it
        // both ways — `true` requires all four bias operands, anything
        // else refuses any bias operand it finds — so the boolean and the
        // operand evidence cannot drift apart. The executors add the four
        // biases; the Metal lowering refuses them until A-9.4.
        site: "ExecutionSurface.attention.attention_bias → AttentionOp.{q,k,v,o}_bias (closure-paired)",
        probe: Some(probe_attention_bias),
    },
    CarriageRule {
        leaf: "qkv_bias",
        reaches: Carriage::Represented,
        // The Q/K/V-only form (Qwen2). Closure holds it both ways: `true`
        // requires the three Q/K/V bias operands and refuses an output
        // bias; declaring it beside `attention_bias: true` is refused as a
        // contradiction rather than resolved in favour of either.
        site: "ExecutionSurface.attention.qkv_bias → AttentionOp.{q,k,v}_bias, no o_bias (closure-paired)",
        probe: Some(probe_qkv_bias),
    },
    CarriageRule {
        leaf: "num_kv_shared_layers",
        reaches: Carriage::Represented,
        // Gemma 4 E2B/E4B: the last N layers read the KV state of the last
        // non-shared layer of their type instead of projecting their own —
        // attention reading ANOTHER op's state, a cross-layer dependency
        // the graph does not represent (V3-F0's open ontology question,
        // scored by that witness). The table represents "no layer shares"
        // and nothing else, so `0` agrees and any other count is dropped
        // at the boundary and blocks — refused, never mis-served as
        // per-layer projections.
        site: "Component.attention[] — no KV-sharing relationship exists; only 0 is representable",
        probe: Some(probe_kv_shared_layers),
    },
    // ── Gemma 4 (V3-F0 witness 3) ──────────────────────────────────
    CarriageRule {
        leaf: "attention_k_eq_v",
        reaches: Carriage::Represented,
        site: "Component.attention[].v_from_k → AttentionOp.v_from_k (closure-paired: no V operand on such a layer)",
        probe: Some(probe_k_eq_v),
    },
    CarriageRule {
        leaf: "enable_moe_block",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.ffn.moe (Some = a routed block is judged) → LayerFfn::Routed / hybrid",
        probe: Some(probe_moe_enabled),
    },
    CarriageRule {
        leaf: "top_k_experts",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.ffn.moe.top_k → RoutedFfnOp routing",
        probe: Some(probe_moe_top_k),
    },
    CarriageRule {
        leaf: "global_head_dim",
        reaches: Carriage::Lowered,
        site: "Component.attention[].geometry.head_dim on the full layers → AttentionOp.head_dim",
        probe: Some(probe_full_layer_head_dim),
    },
    CarriageRule {
        leaf: "num_global_key_value_heads",
        reaches: Carriage::Lowered,
        site: "Component.attention[].geometry.num_kv_heads on the full layers → AttentionOp.num_kv_heads",
        probe: Some(probe_full_layer_kv_heads),
    },
    CarriageRule {
        leaf: "hidden_size_per_layer_input",
        reaches: Carriage::Represented,
        // Per-layer-input embeddings (Gemma 3n/4 E2B): a second embedding
        // table gated into every layer. No object or op exists for it; the
        // graph represents its ABSENCE only, so `0` agrees and any width
        // is dropped at the boundary and blocks.
        site: "no schema field — the graph represents PLE as absent; only 0 is representable",
        probe: Some(probe_zero),
    },
    CarriageRule {
        leaf: "use_double_wide_mlp",
        reaches: Carriage::Represented,
        // Doubles the MLP width on KV-shared layers; no KV-shared layer is
        // representable (see `num_kv_shared_layers`), so only `false` is.
        site: "no schema field — only `false` is representable",
        probe: Some(probe_false),
    },
    CarriageRule {
        leaf: "use_clipped_linears",
        reaches: Carriage::Represented,
        // A tower option that clips projection outputs; no op carries a
        // clip, so only `false` is representable.
        site: "no schema field on the tower surface — only `false` is representable",
        probe: Some(probe_false),
    },
    CarriageRule {
        leaf: "mlp_bias",
        reaches: Carriage::Parsed,
        // Same argument as `attention_bias` immediately above: VINDEX3 has
        // no `mlp_bias` field, and operand closure over the FFN's actual
        // bias tensors (or their absence) is the real gate. Granite 4.1
        // declares `false` on 3B/8B/30B, which agrees trivially; a
        // checkpoint declaring `true` blocks at G5b if the projections
        // don't carry bias operands, not here.
        site: "no schema field — carried instead as operand evidence, gated by G5b closure",
        probe: None,
    },
    CarriageRule {
        leaf: "max_position_embeddings",
        reaches: Carriage::Parsed,
        // A serving/KV-allocation bound, not a forward-pass semantic: no
        // op reads it, and two checkpoints differing only here compute
        // identical logits for any prompt both can hold. Recorded so the
        // absence is a judgement on the report rather than a silence.
        site: "no schema field — a KV-allocation bound, read by no generic op",
        probe: None,
    },
];
