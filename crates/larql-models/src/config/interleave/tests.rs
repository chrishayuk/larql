//! Every spelling in the wild resolves to the same canonical topology, and
//! every malformed one blocks.
//!
//! The positive fixtures are the four real declarations plus Inkling's MTP
//! sub-scope. The negative fixtures are the six ways a declaration can fail
//! to determine one kind per layer — each of which, before this module,
//! would have fallen through to full attention.

use super::*;

const KIMI_LAYERS: usize = 27;
const GLM_LAYERS: usize = 45;
const INKLING_LAYERS: usize = 42;
const INKLING_MTP_LAYERS: usize = 8;
const INKLING_WINDOW: usize = 512;

fn resolved(
    config: &serde_json::Value,
    scope: InterleaveScope,
    layers: usize,
    window: Option<usize>,
) -> ResolvedInterleave {
    match read_declared_interleave(config, scope, layers, window) {
        DeclaredInterleave::Resolved(r) => *r,
        other => panic!("expected a resolution, got {other:?}"),
    }
}

fn error(config: &serde_json::Value, scope: InterleaveScope, layers: usize) -> InterleaveError {
    match read_declared_interleave(config, scope, layers, None) {
        DeclaredInterleave::Unresolved(e) => e,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

fn is_recurrent(k: &LayerKind) -> bool {
    matches!(k, LayerKind::Recurrent(_))
}
fn is_sliding(k: &LayerKind) -> bool {
    matches!(k, LayerKind::Sliding { .. })
}
fn is_full(k: &LayerKind) -> bool {
    matches!(k, LayerKind::Full)
}

/// **GLM-5.3-Flash** — two zero-based sets, `L L L S` from layer 0.
#[test]
fn glm_resolves_34_kda_and_11_full_zero_based() {
    let full: Vec<i64> = (0..GLM_LAYERS as i64).filter(|i| i % 4 == 3).collect();
    let kda: Vec<i64> = (0..GLM_LAYERS as i64).filter(|i| i % 4 != 3).collect();
    let config = serde_json::json!({
        "linear_attn_config": { "kda_layers": kda, "full_attn_layers": full }
    });
    let r = resolved(&config, InterleaveScope::DecoderStack, GLM_LAYERS, None);
    assert_eq!(r.count(is_recurrent), 34);
    assert_eq!(r.count(is_full), 11);
    assert_eq!(r.provenance.resolved_base, Some(LayerIndexBase::Zero));
    assert_eq!(r.provenance.encoding, InterleaveEncoding::PartitionSets);
    // Layer 0 is recurrent and stays layer 0 — the boundary that decides
    // the base.
    assert_eq!(r.layers[0], LayerKind::Recurrent(RecurrenceFamily::Kda));
}

/// **Kimi Linear** — the same two keys, one-based.
#[test]
fn kimi_resolves_20_kda_and_7_full_one_based() {
    let full: Vec<i64> = vec![4, 8, 12, 16, 20, 24, 27];
    let kda: Vec<i64> = (1..=KIMI_LAYERS as i64)
        .filter(|i| !full.contains(i))
        .collect();
    let config = serde_json::json!({
        "linear_attn_config": { "kda_layers": kda, "full_attn_layers": full }
    });
    let r = resolved(&config, InterleaveScope::DecoderStack, KIMI_LAYERS, None);
    assert_eq!(r.count(is_recurrent), 20);
    assert_eq!(r.count(is_full), 7);
    assert_eq!(r.provenance.resolved_base, Some(LayerIndexBase::One));
    // Declared 27 against 27 layers becomes layer 26 — out of range
    // zero-based, which is what proves the base.
    assert_eq!(r.layers[26], LayerKind::Full);
    assert_eq!(r.layers[0], LayerKind::Recurrent(RecurrenceFamily::Kda));
}

/// **Inkling-Small** — one zero-based set, complement implied, and the
/// window travels with the kind.
#[test]
fn inkling_resolves_35_sliding_and_7_global_from_one_set() {
    let global = [5i64, 11, 17, 23, 29, 35, 41];
    let local: Vec<i64> = (0..INKLING_LAYERS as i64)
        .filter(|i| !global.contains(i))
        .collect();
    assert_eq!(local.len(), 35);
    let config = serde_json::json!({ "local_layer_ids": local });

    let r = resolved(
        &config,
        InterleaveScope::DecoderStack,
        INKLING_LAYERS,
        Some(INKLING_WINDOW),
    );
    assert_eq!(r.count(is_sliding), 35, "the whole point: not 0");
    assert_eq!(r.count(is_full), 7);
    assert_eq!(
        r.provenance.encoding,
        InterleaveEncoding::ExplicitSetWithComplement
    );
    assert_eq!(r.provenance.resolved_base, Some(LayerIndexBase::Zero));
    assert_eq!(r.provenance.sources, vec!["local_layer_ids".to_string()]);

    // The window is carried, not defaulted — a KV planner reading this
    // sizes 512, not the whole 1,048,576-token prefix.
    assert_eq!(
        r.layers[0],
        LayerKind::Sliding {
            window: Some(INKLING_WINDOW)
        }
    );
    for g in global {
        assert_eq!(r.layers[g as usize], LayerKind::Full, "layer {g}");
    }
}

/// An undeclared window stays absent rather than becoming a number.
#[test]
fn an_undeclared_window_is_absent_not_defaulted() {
    let config = serde_json::json!({ "local_layer_ids": [0, 1] });
    let r = resolved(&config, InterleaveScope::DecoderStack, 4, None);
    assert_eq!(r.layers[0], LayerKind::Sliding { window: None });
}

/// **Inkling-Small MTP** — the same key again, for a different layer
/// space. Resolving it against the decoder's 42 layers would be wrong;
/// the scope is what keeps them apart.
#[test]
fn the_mtp_sub_stack_resolves_independently_of_the_decoder() {
    let config = serde_json::json!({
        "local_layer_ids": (0..42).filter(|i| ![5, 11, 17, 23, 29, 35, 41].contains(i)).collect::<Vec<i32>>(),
        "mtp_config": { "local_layer_ids": [0, 2, 4, 5, 6, 7] },
    });
    let mtp = resolved(
        &config,
        InterleaveScope::MtpStack,
        INKLING_MTP_LAYERS,
        Some(INKLING_WINDOW),
    );
    assert_eq!(mtp.layer_count, INKLING_MTP_LAYERS);
    assert_eq!(mtp.count(is_sliding), 6);
    assert_eq!(mtp.count(is_full), 2);
    assert_eq!(mtp.layers[1], LayerKind::Full);
    assert_eq!(mtp.layers[3], LayerKind::Full);
    assert_eq!(mtp.provenance.scope, "target.mtp_stack");
    assert_eq!(
        mtp.provenance.sources,
        vec!["mtp_config.local_layer_ids".to_string()]
    );

    // And the decoder scope, read from the same config, is a different
    // resolution — 42 layers, not 8.
    let decoder = resolved(
        &config,
        InterleaveScope::DecoderStack,
        INKLING_LAYERS,
        Some(INKLING_WINDOW),
    );
    assert_eq!(decoder.layer_count, INKLING_LAYERS);
    assert_ne!(decoder.layers.len(), mtp.layers.len());
}

/// **Qwen3.8** — a per-layer array, no base to prove.
#[test]
fn a_per_layer_array_resolves_without_a_base() {
    let entries: Vec<&str> = (0..8)
        .map(|i| {
            if i % 4 == 3 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect();
    let config = serde_json::json!({ "layer_types": entries });
    let r = resolved(&config, InterleaveScope::DecoderStack, 8, None);
    assert_eq!(r.count(is_recurrent), 6);
    assert_eq!(r.count(is_full), 2);
    assert_eq!(r.provenance.encoding, InterleaveEncoding::PerLayerArray);
    assert_eq!(
        r.provenance.resolved_base, None,
        "position IS the index; there is no base to prove"
    );
    // The array names a recurrence but not which one.
    assert_eq!(
        r.layers[0],
        LayerKind::Recurrent(RecurrenceFamily::Unidentified)
    );
}

/// A sliding entry in the array carries the window too.
#[test]
fn a_sliding_array_entry_carries_the_window() {
    let config = serde_json::json!({
        "layer_types": ["sliding_attention", "full_attention"]
    });
    let r = resolved(&config, InterleaveScope::DecoderStack, 2, Some(128));
    assert_eq!(r.layers[0], LayerKind::Sliding { window: Some(128) });
    assert_eq!(r.layers[1], LayerKind::Full);
}

// ── the negative arm ────────────────────────────────────────────────
//
// Six ways a declaration fails to determine one kind per layer. Before
// this module every one of them read as "declared nothing", and the
// caller's default answered.

#[test]
fn an_absent_declaration_is_absent_not_an_error() {
    let config = serde_json::json!({ "hidden_size": 4096 });
    assert_eq!(
        read_declared_interleave(&config, InterleaveScope::DecoderStack, 8, None),
        DeclaredInterleave::Absent
    );
}

#[test]
fn two_declarations_naming_one_layer_block() {
    let config = serde_json::json!({
        "linear_attn_config": { "kda_layers": [0, 1, 2], "full_attn_layers": [2, 3] }
    });
    assert_eq!(
        error(&config, InterleaveScope::DecoderStack, 4),
        InterleaveError::Overlap { layer: 2 }
    );
}

#[test]
fn a_layer_no_declaration_names_blocks() {
    let config = serde_json::json!({
        "linear_attn_config": { "kda_layers": [0, 1], "full_attn_layers": [2] }
    });
    assert_eq!(
        error(&config, InterleaveScope::DecoderStack, 4),
        InterleaveError::Uncovered { layer: 3 }
    );
}

#[test]
fn an_index_outside_the_scope_blocks() {
    let config = serde_json::json!({
        "linear_attn_config": { "kda_layers": [0, 1], "full_attn_layers": [2, 99] }
    });
    assert_eq!(
        error(&config, InterleaveScope::DecoderStack, 4),
        InterleaveError::NoConsistentBase {
            declared_indices: 4,
            layer_count: 4,
        }
    );
}

/// A bare set that reads validly under **both** bases proves neither.
///
/// Only reachable with an implied complement: `{1,2,3}` in a 5-layer scope
/// is layers 1–3 zero-based and layers 0–2 one-based, and both leave a
/// well-formed complement. A partition cannot do this — one of `0..n` and
/// `1..=n` contains 0 and the other does not.
#[test]
fn a_declaration_valid_under_both_bases_blocks() {
    let config = serde_json::json!({ "local_layer_ids": [1, 2, 3] });
    assert_eq!(
        error(&config, InterleaveScope::DecoderStack, 5),
        InterleaveError::AmbiguousBase { layer_count: 5 }
    );
}

/// An entry with no kind is unexpressed **for that layer**, and the layers
/// beside it still resolve.
///
/// This is GLM-5.3-Flash's shape: 34 recurrent entries this build reads
/// and 11 `deepseek_sparse_attention` entries it does not. Failing the
/// whole array would report 45 unexpressed and hide the 34 that are
/// understood — worse information, from a stricter-looking rule.
#[test]
fn an_unknown_array_entry_is_unexpressed_for_its_layer_only() {
    let config = serde_json::json!({
        "layer_types": ["linear_attention", "deepseek_sparse_attention", "full_attention"]
    });
    let r = resolved(&config, InterleaveScope::DecoderStack, 3, None);
    assert_eq!(
        r.layers[0],
        LayerKind::Recurrent(RecurrenceFamily::Unidentified)
    );
    assert_eq!(
        r.layers[1],
        LayerKind::Unexpressed {
            declared: "deepseek_sparse_attention".to_string()
        },
        "the declaration is carried verbatim, not erased"
    );
    assert_eq!(r.layers[2], LayerKind::Full);
    assert_eq!(r.count(is_recurrent), 1, "the readable layers still read");
}

#[test]
fn an_array_of_the_wrong_length_blocks() {
    let config = serde_json::json!({ "layer_types": vec!["full_attention"; 3] });
    assert_eq!(
        error(&config, InterleaveScope::DecoderStack, 4),
        InterleaveError::LengthMismatch {
            declared: 3,
            layer_count: 4,
        }
    );
}

/// Two complements determine nothing, so they block rather than letting
/// evaluation order pick a winner.
#[test]
fn two_complements_block() {
    let declarations = [
        Declaration {
            kind: LayerKind::Full,
            membership: Membership::ExplicitSet(vec![0]),
        },
        Declaration {
            kind: LayerKind::Sliding { window: None },
            membership: Membership::Complement,
        },
        Declaration {
            kind: LayerKind::Recurrent(RecurrenceFamily::Kda),
            membership: Membership::Complement,
        },
    ];
    assert_eq!(
        resolve_declarations("scope", vec![], &declarations, 4).unwrap_err(),
        InterleaveError::MultipleComplements
    );
}

/// Every resolution records where it came from and how it was read — the
/// chain that lets two checkpoints be shown to declare one concept
/// differently.
#[test]
fn every_resolution_carries_its_provenance() {
    let kimi = serde_json::json!({
        "linear_attn_config": { "kda_layers": [1, 2, 3], "full_attn_layers": [4] }
    });
    let r = resolved(&kimi, InterleaveScope::DecoderStack, 4, None);
    assert_eq!(
        r.provenance.sources,
        vec![
            "linear_attn_config.kda_layers".to_string(),
            "linear_attn_config.full_attn_layers".to_string(),
        ]
    );
    assert_eq!(r.provenance.encoding, InterleaveEncoding::PartitionSets);
    assert_eq!(r.provenance.resolved_base, Some(LayerIndexBase::One));
    assert_eq!(r.provenance.scope, "target.decoder_stack");
}
