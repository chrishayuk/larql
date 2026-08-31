//! The table is pinned against Qwen3.8's real graph values, and every
//! entry declares what produced it — so a literal creeping in is
//! visible as `derived_from: "target constant"` on something that is
//! not a target constant.

use super::*;

/// Qwen3.8's cadence: three recurrent, one attending, sixteen times.
fn qwen_attending() -> Vec<usize> {
    (0..64).filter(|i| i % 4 == 3).collect()
}

#[test]
fn the_cadence_yields_the_interval_rather_than_assuming_four() {
    assert_eq!(full_attention_interval(&qwen_attending(), 64), Ok(4));
    // A different real cadence gives a different answer, which is the
    // whole point of deriving it.
    let every_third: Vec<usize> = (0..63).filter(|i| i % 3 == 2).collect();
    assert_eq!(full_attention_interval(&every_third, 63), Ok(3));
}

/// A stack whose attending layers do not repeat has no interval that
/// describes it. Taking the converter's default of 4 would export a
/// different layer programme than the model has.
#[test]
fn an_irregular_cadence_refuses_rather_than_defaulting() {
    let irregular = vec![3usize, 7, 12, 15];
    assert!(matches!(
        full_attention_interval(&irregular, 16),
        Err(MetadataError::IrregularAttentionCadence { .. })
    ));
    // Regular but stopping early: the interval would describe a prefix
    // and lie about the tail.
    let prefix_only = vec![3usize, 7, 11];
    assert!(matches!(
        full_attention_interval(&prefix_only, 64),
        Err(MetadataError::IrregularAttentionCadence { .. })
    ));
}

/// The graph declares three sections; llama.cpp's array has four slots.
/// The trailing zero is the target's spelling, not a fourth section.
#[test]
fn rope_sections_pad_to_the_targets_width() {
    assert_eq!(rope_sections(&[11, 11, 10]).unwrap(), vec![11, 11, 10, 0]);
    assert_eq!(
        rope_sections(&[11, 11, 10, 0]).unwrap(),
        vec![11, 11, 10, 0]
    );
    assert!(
        rope_sections(&[]).is_err(),
        "no declared sections is not a pad-to-four"
    );
    assert!(
        rope_sections(&[1, 2, 3, 4, 5]).is_err(),
        "more sections than slots"
    );
}

/// **The namespace trap.** These two constants sit beside each other and
/// mean different things; swapping them produces a file that loads and
/// misreads.
#[test]
fn the_file_type_is_not_the_ggml_tensor_type() {
    assert_eq!(FILE_TYPE_MOSTLY_NVFP4, 39, "LLAMA_FTYPE_MOSTLY_NVFP4");
    assert_eq!(
        larql_models::quant::nvfp4_ggml::TYPE_NVFP4,
        40,
        "GGML_TYPE_NVFP4 — a different enumeration entirely"
    );
    assert_ne!(
        FILE_TYPE_MOSTLY_NVFP4,
        larql_models::quant::nvfp4_ggml::TYPE_NVFP4,
        "adjacent numbers, unrelated meanings"
    );
}

/// Only the four `general.*` keys may be literals. Everything else must
/// name a graph fact, which is what keeps model knowledge off the
/// target side.
#[test]
fn nothing_but_target_constants_is_hardcoded() {
    use crate::format::vindex3::gguf::preflight::tests_support::qwen_shaped_surface;
    let surface = qwen_shaped_surface();
    let table = qwen35_metadata(
        &surface,
        64,
        5120,
        10_000_000.0,
        &[11, 11, 10],
        0.25,
        &qwen_attending(),
    )
    .expect("a complete surface yields a complete table");

    let constants: Vec<&str> = table
        .iter()
        .filter(|m| m.derived_from == "target constant")
        .map(|m| m.key.as_str())
        .collect();
    assert_eq!(
        constants,
        vec![
            "general.type",
            "general.architecture",
            "general.quantization_version",
            "general.file_type"
        ],
        "only facts about llama.cpp may be literals"
    );
    for m in &table {
        assert!(
            !m.derived_from.is_empty(),
            "`{}` does not say where it came from",
            m.key
        );
    }
}

/// Pinned against the hero container's actual values.
#[test]
fn the_table_matches_qwens_real_graph() {
    use crate::format::vindex3::gguf::preflight::tests_support::qwen_shaped_surface;
    let table = qwen35_metadata(
        &qwen_shaped_surface(),
        64,
        5120,
        10_000_000.0,
        &[11, 11, 10],
        0.25,
        &qwen_attending(),
    )
    .unwrap();
    let get = |k: &str| table.iter().find(|m| m.key == k).map(|m| m.value.clone());

    assert_eq!(get("qwen35.block_count"), Some(MetaValue::U32(64)));
    assert_eq!(get("qwen35.context_length"), Some(MetaValue::U32(262_144)));
    assert_eq!(get("qwen35.embedding_length"), Some(MetaValue::U32(5120)));
    assert_eq!(
        get("qwen35.feed_forward_length"),
        Some(MetaValue::U32(17408))
    );
    assert_eq!(get("qwen35.attention.head_count"), Some(MetaValue::U32(24)));
    assert_eq!(
        get("qwen35.attention.head_count_kv"),
        Some(MetaValue::U32(4))
    );
    assert_eq!(
        get("qwen35.attention.key_length"),
        Some(MetaValue::U32(256))
    );
    assert_eq!(get("qwen35.rope.dimension_count"), Some(MetaValue::U32(64)));
    assert_eq!(
        get("qwen35.rope.dimension_sections"),
        Some(MetaValue::ArrU32(vec![11, 11, 10, 0]))
    );
    assert_eq!(get("qwen35.ssm.conv_kernel"), Some(MetaValue::U32(4)));
    assert_eq!(
        get("qwen35.ssm.inner_size"),
        Some(MetaValue::U32(6144)),
        "48 x 128"
    );
    assert_eq!(get("qwen35.ssm.state_size"), Some(MetaValue::U32(128)));
    assert_eq!(get("qwen35.ssm.time_step_rank"), Some(MetaValue::U32(48)));
    assert_eq!(get("qwen35.ssm.group_count"), Some(MetaValue::U32(16)));
    assert_eq!(
        get("qwen35.full_attention_interval"),
        Some(MetaValue::U32(4))
    );
}

/// A surface missing a fact refuses rather than emitting a partial
/// table — the same defect preflight names, caught again at the point of
/// use.
#[test]
fn a_missing_fact_refuses_the_whole_table() {
    use crate::format::vindex3::gguf::preflight::tests_support::qwen_shaped_surface;
    let mut surface = qwen_shaped_surface();
    surface.context_length = None;
    assert_eq!(
        qwen35_metadata(
            &surface,
            64,
            5120,
            1.0,
            &[11, 11, 10],
            0.25,
            &qwen_attending()
        ),
        Err(MetadataError::Missing("execution.context_length"))
    );
}
