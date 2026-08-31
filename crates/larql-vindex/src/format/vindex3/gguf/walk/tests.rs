//! The walk is tested against Qwen3.8's real inventory — 851 primary-text
//! physical tensors in the shape the hero container actually carries,
//! read from its segment headers rather than imagined.

use super::*;

const LAYERS: usize = 64;
/// Attending layers sit at N ≡ 3 (mod 4): sixteen of sixty-four.
fn attends(l: usize) -> bool {
    l % 4 == 3
}

/// Qwen3.8-27B's primary-text surface, exactly as the container holds
/// it: 848 in the decoder stack plus embedding, final norm and an
/// untied output head.
fn qwen_sources() -> Vec<SourceTensor> {
    let mut v = Vec::new();
    let mut push = |object: &str, name: String, role: &str, layer: Option<usize>| {
        v.push(SourceTensor {
            object: object.into(),
            name,
            role: role.into(),
            layer,
            representation: RepresentationKind::Bf16,
        })
    };
    for l in 0..LAYERS {
        let s = "target.decoder_stack";
        // Trunk norms — both on EVERY layer, recurrent ones included.
        push(
            s,
            format!("{l}.input_layernorm.weight"),
            "input layer norm",
            Some(l),
        );
        push(
            s,
            format!("{l}.post_attention_layernorm.weight"),
            "post-attention layer norm",
            Some(l),
        );
        // Dense FFN, every layer.
        push(s, format!("{l}.mlp.gate_proj.weight"), "ffn gate", Some(l));
        push(s, format!("{l}.mlp.up_proj.weight"), "ffn up", Some(l));
        push(s, format!("{l}.mlp.down_proj.weight"), "ffn down", Some(l));
        if attends(l) {
            push(s, format!("{l}.self_attn.q_proj.weight"), "query", Some(l));
            push(s, format!("{l}.self_attn.k_proj.weight"), "key", Some(l));
            push(s, format!("{l}.self_attn.v_proj.weight"), "value", Some(l));
            push(s, format!("{l}.self_attn.o_proj.weight"), "output", Some(l));
            push(
                s,
                format!("{l}.self_attn.q_norm.weight"),
                "attention q norm",
                Some(l),
            );
            push(
                s,
                format!("{l}.self_attn.k_norm.weight"),
                "attention k norm",
                Some(l),
            );
        } else {
            push(
                s,
                format!("{l}.linear_attn.in_proj_qkv.weight"),
                "fused recurrent q|k|v",
                Some(l),
            );
            push(
                s,
                format!("{l}.linear_attn.in_proj_z.weight"),
                "output-gate projection",
                Some(l),
            );
            push(
                s,
                format!("{l}.linear_attn.in_proj_a.weight"),
                "decay projection",
                Some(l),
            );
            push(
                s,
                format!("{l}.linear_attn.in_proj_b.weight"),
                "write-strength projection",
                Some(l),
            );
            push(
                s,
                format!("{l}.linear_attn.conv1d.weight"),
                "causal conv over q|k|v",
                Some(l),
            );
            push(s, format!("{l}.linear_attn.A_log"), "log decay", Some(l));
            push(
                s,
                format!("{l}.linear_attn.dt_bias"),
                "timestep bias",
                Some(l),
            );
            push(
                s,
                format!("{l}.linear_attn.norm.weight"),
                "gated norm",
                Some(l),
            );
            push(
                s,
                format!("{l}.linear_attn.out_proj.weight"),
                "output projection",
                Some(l),
            );
        }
    }
    push(
        "target.embedding",
        "embed_tokens.weight".into(),
        "embedding",
        None,
    );
    push(
        "target.final_norm",
        "norm.weight".into(),
        "final norm",
        None,
    );
    push(
        "target.output_head",
        "lm_head.weight".into(),
        "output head",
        None,
    );
    v
}

fn vision_excluded() -> Vec<Exclusion> {
    vec![Exclusion {
        object: "vision.perception_tower".into(),
        count: 333,
        reason: "export surface = primary_text",
    }]
}

/// The head is untied, so it must be produced. Derived from the graph's
/// `head_reuses_embedding = false`, not from what the walk happened to
/// plan.
fn required() -> Vec<(&'static str, &'static str)> {
    vec![
        ("token_embd.weight", "every model needs an embedding"),
        ("output_norm.weight", "final norm before the head"),
        ("output.weight", "graph says head_reuses_embedding = false"),
    ]
}

#[test]
fn the_whole_primary_text_surface_is_accounted_for() {
    let sources = qwen_sources();
    assert_eq!(
        sources.len(),
        851,
        "848 decoder + embedding + final norm + head"
    );

    let (plans, ledger) = walk_primary_text(&sources, vision_excluded(), &required());

    assert_eq!(ledger.errors, vec![], "no unplanned, duplicate or missing");
    assert_eq!(ledger.source_total, 851);
    assert_eq!(
        ledger.accounted, 851,
        "every physical tensor traced to a target"
    );
    assert_eq!(plans.len(), 851);
    assert!(ledger.ready());

    assert_eq!(ledger.source_by_object["target.decoder_stack"], 848);
    assert_eq!(ledger.source_by_object["target.embedding"], 1);
    assert_eq!(ledger.source_by_object["target.final_norm"], 1);
    assert_eq!(ledger.source_by_object["target.output_head"], 1);

    // Vision is excluded by surface, and says so.
    assert_eq!(ledger.excluded.len(), 1);
    assert_eq!(ledger.excluded[0].count, 333);
    assert_eq!(ledger.excluded[0].reason, "export surface = primary_text");
}

/// **The applicability trap.** The name says "post attention"; the
/// container carries one per layer, recurrent layers included. Keying
/// this off layer kind would silently drop forty-eight tensors, and
/// llama.cpp would find no norm where it expects one.
#[test]
fn post_attention_norm_is_a_trunk_norm_not_an_attention_layer_only_norm() {
    let (plans, _) = walk_primary_text(&qwen_sources(), vec![], &[]);
    let count = plans
        .iter()
        .filter(|p| p.target_name.ends_with(".post_attention_norm.weight"))
        .count();
    assert_eq!(
        count, 64,
        "one per layer across the whole stack, not one per attending layer"
    );
    let attn_norm = plans
        .iter()
        .filter(|p| p.target_name.ends_with(".attn_norm.weight"))
        .count();
    assert_eq!(attn_norm, 64);
    // And the Q/K norms genuinely are attention-only.
    for suffix in [".attn_q_norm.weight", ".attn_k_norm.weight"] {
        assert_eq!(
            plans
                .iter()
                .filter(|p| p.target_name.ends_with(suffix))
                .count(),
            16,
            "{suffix} exists only on attending layers"
        );
    }
}

/// **The silent-fallback trap.** llama.cpp's qwen35 loader treats
/// `output.weight` as optional and ties the embedding when it is
/// missing. That runs, and produces plausible text, and is not the model
/// that was exported. Permissiveness in the target is not permission
/// here.
#[test]
fn untied_output_head_is_required_even_when_the_target_runtime_can_fallback() {
    let mut sources = qwen_sources();
    sources.retain(|t| t.role != "output head");
    assert_eq!(sources.len(), 850);

    let (_, ledger) = walk_primary_text(&sources, vision_excluded(), &required());
    assert!(
        ledger.errors.iter().any(|e| matches!(
            e,
            WalkError::MissingRequired { target, .. } if target == "output.weight"
        )),
        "an untied head that was not produced must refuse: {:?}",
        ledger.errors
    );
    assert!(!ledger.ready());
}

/// A role with no target is named rather than skipped — the state the
/// four layer-norm families were in before the real inventory found
/// them.
#[test]
fn an_unmapped_role_is_reported_not_silently_dropped() {
    let mut sources = qwen_sources();
    sources.push(SourceTensor {
        object: "target.decoder_stack".into(),
        name: "0.something.new.weight".into(),
        role: "a role nothing maps".into(),
        layer: Some(0),
        representation: RepresentationKind::Bf16,
    });
    let (_, ledger) = walk_primary_text(&sources, vec![], &[]);
    assert!(ledger.errors.iter().any(|e| matches!(
        e,
        WalkError::Unplanned { role, .. } if role == "a role nothing maps"
    )));
    assert_eq!(
        ledger.accounted, 851,
        "the unplanned one is not counted as done"
    );
    assert_eq!(ledger.source_total, 852);
    assert!(!ledger.ready(), "accounted < source_total");
}

/// Two sources claiming one target would leave whichever was written
/// last, silently.
#[test]
fn two_sources_claiming_one_target_name_is_an_error() {
    let mut sources = qwen_sources();
    sources.push(SourceTensor {
        object: "target.decoder_stack".into(),
        name: "0.mlp.down_proj.weight.duplicate".into(),
        role: "ffn down".into(),
        layer: Some(0),
        representation: RepresentationKind::Bf16,
    });
    let (_, ledger) = walk_primary_text(&sources, vec![], &[]);
    assert!(ledger.errors.iter().any(|e| matches!(
        e,
        WalkError::DuplicateTarget { target, .. } if target == "blk.0.ffn_down.weight"
    )));
}

/// NVFP4 adds target tensors rather than mapping one to one, so the two
/// counts differ in both directions and neither is wrong.
#[test]
fn nvfp4_sources_generate_sibling_scale_tensors() {
    let mut sources = qwen_sources();
    for t in sources.iter_mut() {
        if t.role == "ffn down" {
            t.representation = RepresentationKind::Nvfp4;
        }
    }
    let (plans, ledger) = walk_primary_text(&sources, vec![], &[]);
    assert_eq!(ledger.generated_scale_tensors, 64, "one per NVFP4 tensor");
    assert_eq!(
        ledger.target_total,
        851 + 64,
        "targets exceed sources by the scale siblings"
    );
    let scaled = plans
        .iter()
        .find(|p| p.target_name == "blk.0.ffn_down.weight")
        .unwrap();
    assert_eq!(scaled.scale_tensor.as_deref(), Some("blk.0.ffn_down.scale"));
}
