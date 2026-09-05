//! Semantic-class registry behavior.

use crate::format::vindex3::plan::semantics::{classify_key, component_of, leaf_of};
use crate::format::vindex3::plan::SemanticClass;
use larql_models::inventory::config_keys::CONSUMED_LEAF_KEYS;

#[test]
fn dangerous_scalars_grade_execution_semantic() {
    for key in [
        "layer_rope_theta",
        "qk_scale_factor",
        "output_multiplier",
        "post_norm_eps",
        "hidden_activation",
        "attention_bias",
    ] {
        assert_eq!(classify_key(key), SemanticClass::ExecutionSemantic, "{key}");
    }
}

#[test]
fn shape_keys_grade_tensor_semantic() {
    for key in ["hidden_size", "patch_size", "out_hidden_size", "head_dim"] {
        assert_eq!(classify_key(key), SemanticClass::TensorSemantic, "{key}");
    }
}

#[test]
fn interface_keys_grade_interface_semantic() {
    for key in [
        "target_layer_ids",
        "block_size",
        "mask_token_id",
        "image_token_id",
    ] {
        assert_eq!(classify_key(key), SemanticClass::InterfaceSemantic, "{key}");
    }
}

#[test]
fn unseen_keys_grade_unknown_and_therefore_block() {
    let class = classify_key("some_future_field_nobody_reviewed");
    assert_eq!(class, SemanticClass::Unknown);
    assert!(class.is_critical());
}

#[test]
fn metadata_keys_do_not_block() {
    let class = classify_key("model_type");
    assert_eq!(class, SemanticClass::MetadataOnly);
    assert!(!class.is_critical());
}

#[test]
fn component_attribution_follows_config_nesting() {
    assert_eq!(component_of("text_config.qk_scale_factor"), "text");
    assert_eq!(component_of("vision_config.patch_size"), "vision");
    assert_eq!(component_of("image_token_id"), "root");
    // A bare leaf ending in `_config` is a root key, not a section.
    assert_eq!(component_of("is_llama_config"), "root");
    assert_eq!(
        component_of("vision_config.rope_parameters.rope_theta"),
        "vision"
    );
}

/// The census: every leaf name a real parser reads
/// ([`CONSUMED_LEAF_KEYS`]) must be judged by this registry — placed in
/// some bucket, `Unknown` refused. `CONSUMED_LEAF_KEYS` is *the* list of
/// key names any `ModelConfig` parser reads, kept honest by its own sync
/// test against the parser source (`larql-models`' `tests/config_keys.rs`);
/// this test closes the other side, so a leaf cannot be `consumed` and
/// *unjudged* at once — a fact `Consumed`-but-`Unknown` used to grade
/// `Representable` in `carriage_finding` (`plan/mod.rs`), not
/// `Unrepresented`, so a key nobody had reviewed sailed through the plan
/// exactly like an unread one is supposed to be caught doing.
///
/// A-11 (2026-08-18) is the origin: Granite's four scaling multipliers
/// were consumed and silently `representable`/`unknown`. Auditing the
/// full 80-key list surfaced 41 more in the same state — GPT-OSS's other
/// YaRN leaves, MoE/MLA operand counts, four GPT-2 shape aliases, a
/// fourth norm-epsilon spelling. All 41 are now bucketed (`semantics.rs`);
/// this test is what keeps the count at zero from here.
#[test]
fn every_consumed_leaf_key_is_judged() {
    let unjudged: Vec<&str> = CONSUMED_LEAF_KEYS
        .iter()
        .copied()
        .filter(|key| classify_key(key) == SemanticClass::Unknown)
        .collect();
    assert!(
        unjudged.is_empty(),
        "consumed but never classified — add each to a bucket in \
         plan/semantics.rs before it can be silently `representable`: \
         {unjudged:?}"
    );
}

#[test]
fn leaf_extraction() {
    assert_eq!(
        leaf_of("text_config.rope_parameters.rope_theta"),
        "rope_theta"
    );
    assert_eq!(leaf_of("block_size"), "block_size");
}

/// **A named absent component is not a normalisation gap.**
///
/// The census's own falsification test. Almost everything it has found so
/// far has been normalisation — an alias, a disabled flag, a checked
/// default — and a taxonomy able to express only those would score
/// GLM-5.3-Flash's sparse indexer as more of the same. It has to say
/// instead that this is architecture work nobody has done.
#[test]
fn a_key_configuring_an_absent_component_grades_unsupported_and_blocks() {
    for key in [
        "index_head_dim",
        "index_topk",
        "indexer_types",
        "indexer_rope_interleave",
    ] {
        assert_eq!(
            classify_key(key),
            SemanticClass::UnsupportedComponent,
            "{key}"
        );
    }

    // It must still block. The class changes what the report SAYS, never
    // how much it permits: an unimplemented component is exactly as
    // disqualifying as an unexamined key.
    assert!(SemanticClass::UnsupportedComponent.is_critical());

    // The Sinkhorn hyper-connection keys sat in this list from wave 7 to
    // wave 18. Wave 19 executed the topology they configure on both
    // traversals, so they grade as execution semantics carried to the
    // component's residual topology — and `mhc`, which no reference
    // explains, stays exactly where it was.
    for key in ["hc_mult", "hc_sinkhorn_iters", "hc_eps"] {
        assert_eq!(classify_key(key), SemanticClass::ExecutionSemantic, "{key}");
    }
    assert_eq!(classify_key("mhc"), SemanticClass::Unknown);
}

/// The indexer's rotary pairing belongs to the indexer.
///
/// A regex over `rope` swept `indexer_rope_interleave` into the general
/// RoPE cluster, where acting on it would have applied an interleaved
/// pairing to the whole model — wrong component and wrong operator, and
/// live rather than latent, because GLM declares `true`.
#[test]
fn the_indexers_rotary_pairing_is_owned_by_the_indexer() {
    use crate::format::vindex3::plan::semantics::unsupported_component;

    assert_eq!(
        unsupported_component("indexer_rope_interleave"),
        unsupported_component("index_topk"),
        "the indexer's own rope belongs to the indexer, not to general RoPE handling"
    );
    assert_ne!(
        classify_key("indexer_rope_interleave"),
        classify_key("rope_interleaved"),
        "the model's rotary pairing and the indexer's are different facts"
    );
}

/// **The table may not become the convenient bucket.**
///
/// `mhc: true` sits directly beside `hc_eps`, `hc_mult` and
/// `hc_sinkhorn_iters` in GLM's config and is very likely part of the
/// same component. It is a bare boolean whose expansion cannot be
/// checked — `glm5_next` is not in transformers 5.5.0 and the repo ships
/// no modeling code — so it stays `unknown`.
///
/// This arm is what stops the registry absorbing everything nearby and
/// reporting a tidier estimate than the evidence supports.
#[test]
fn a_neighbouring_key_with_no_evidence_stays_unknown() {
    assert_eq!(classify_key("mhc"), SemanticClass::Unknown);
    for key in ["qk_head_dim", "topk_method", "moe_router_dtype"] {
        assert_eq!(classify_key(key), SemanticClass::Unknown, "{key}");
    }
}

#[test]
fn decoding_policy_is_recognised_without_swallowing_routing_or_geometry() {
    use crate::format::vindex3::plan::semantics::classify_key;
    // Preserved and classified, not blocking.
    for leaf in [
        "do_sample",
        "top_k",
        "bad_words_ids",
        "num_beams",
        "max_length",
    ] {
        assert_eq!(
            classify_key(leaf),
            SemanticClass::GenerationPolicy,
            "{leaf} is decoding policy"
        );
    }
    // The control the table's exact-name contract exists for. `top_k` is a
    // sampling cutoff; MoE's routing top-k is spelled differently and is
    // execution-semantic. A substring rule would have taken both.
    assert_ne!(
        classify_key("num_experts_per_tok"),
        SemanticClass::GenerationPolicy
    );
    assert!(classify_key("num_experts_per_tok").is_critical());
    // Dropout parameterises a path inference never runs, at any rate.
    for leaf in ["attn_pdrop", "attention_dropout", "input_jitter_noise"] {
        assert_eq!(classify_key(leaf), SemanticClass::TrainingOnly, "{leaf}");
    }
}

#[test]
fn pretraining_tp_is_judged_by_value_because_the_forward_pass_reads_it() {
    use crate::format::vindex3::plan::semantics::classify_key_at;
    // At 1 — every checkpoint in the conformance corpus — a no-op.
    assert_eq!(
        classify_key_at("pretraining_tp", &serde_json::json!(1)),
        SemanticClass::TrainingOnly
    );
    // Above 1, HF Llama slices every projection into `tp` shards and sums
    // them. Same key, different fact, and the name-only reading would have
    // silenced this arm.
    assert_eq!(
        classify_key_at("pretraining_tp", &serde_json::json!(2)),
        SemanticClass::ExecutionSemantic
    );
    assert!(classify_key_at("pretraining_tp", &serde_json::json!(2)).is_critical());
    // A key outside the value table answers the same either way.
    assert_eq!(
        classify_key_at("do_sample", &serde_json::json!(true)),
        classify_key("do_sample")
    );
}

#[test]
fn one_idea_collects_its_many_spellings_without_taking_a_neighbour() {
    use crate::format::vindex3::plan::semantics::{cluster_for, SemanticCluster};
    // The claim the census rests on: forty rope spellings, one idea.
    for subject in [
        "rope_theta",
        "rope_scaling.rope_type",
        "text_config.rope_parameters.mscale_all_dim",
        "rope_scaling.low_freq_factor",
        "partial_rotary_factor",
    ] {
        assert_eq!(
            cluster_for(subject),
            SemanticCluster::PositionRope,
            "{subject}"
        );
    }
    // ...and the neighbours it must not swallow. `mrope_section` is its own
    // operator, and a sampling `top_k` is not MoE routing — the pair a
    // substring rule would have merged.
    assert_eq!(
        cluster_for("text_config.rope_parameters.mrope_section"),
        SemanticCluster::PositionMrope
    );
    assert_eq!(
        cluster_for("num_experts_per_tok"),
        SemanticCluster::MoeRouting
    );
    assert_eq!(cluster_for("top_k"), SemanticCluster::InertGenerationPolicy);
}

#[test]
fn a_nested_section_owns_its_keys_whatever_the_leaf_is_called() {
    use crate::format::vindex3::plan::semantics::{cluster_for, SemanticCluster};
    // `…quantization_config.…weights.type` ends in `type`, which the rope
    // table claims. Path ownership is asked first, so the section wins.
    assert_eq!(
        cluster_for("type"),
        SemanticCluster::PositionRope,
        "a bare `type` is the rope spelling"
    );
    assert_eq!(
        cluster_for("quantization_config.config_groups.group_0.weights.type"),
        SemanticCluster::RepresentationQuantization
    );
}

#[test]
fn an_unjudged_subject_stays_unclustered_rather_than_joining_a_neighbour() {
    use crate::format::vindex3::plan::semantics::{cluster_for, SemanticCluster};
    // The control. `Unclustered` is a finding about the taxonomy; growing
    // the buckets by pattern-matching would hide exactly the subjects that
    // still need a judgement.
    assert_eq!(
        cluster_for("some_field_nobody_has_reviewed"),
        SemanticCluster::Unclustered
    );
}

#[test]
fn the_expert_schedule_is_inert_only_at_the_uniform_stack() {
    use crate::format::vindex3::plan::semantics::classify_key_at;
    // Stride 1, no exceptions: every layer routes, which is the uniform
    // stack the graph already builds.
    assert_eq!(
        classify_key_at("decoder_sparse_step", &serde_json::json!(1)),
        SemanticClass::TrainingOnly
    );
    assert_eq!(
        classify_key_at("mlp_only_layers", &serde_json::json!([])),
        SemanticClass::TrainingOnly
    );
    // The controls. A stride of 2 makes half the tower dense and a
    // non-empty exception list carves out named layers — real per-layer
    // topology, not expressible, and it must keep blocking. Without
    // these, the fix reads as "stop looking at the expert schedule".
    assert!(classify_key_at("decoder_sparse_step", &serde_json::json!(2)).is_critical());
    assert!(classify_key_at("mlp_only_layers", &serde_json::json!([0, 1])).is_critical());
}
