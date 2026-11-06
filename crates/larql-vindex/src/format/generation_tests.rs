//! Colocated tests for `generation` — schema-to-generation dispatch.
//!
//! The mapping is **many-to-one**, and an earlier version that assumed a
//! bijection refused every legacy-schema index in existence. These tests pin
//! the table rather than the arithmetic, so a future "simplification" back to
//! `version == generation` fails loudly.

use super::generation::{
    admit_extraction_generation, detect_generation, generation_for_schema, schema_range_label,
    supported_schema_summary, ContainerGeneration, GenerationRequest, IndexSchemaVersion,
    ALL_GENERATIONS, DEFAULT_EXTRACTION_GENERATION, V2_CURRENT_SCHEMA, V2_MIN_SCHEMA,
    V3_CURRENT_SCHEMA,
};
use crate::format::filenames::INDEX_JSON;
use crate::VindexError;

fn schema(v: u32) -> IndexSchemaVersion {
    IndexSchemaVersion(v)
}

// ── The pinned table ───────────────────────────────────────────────────────

#[test]
fn schema_one_is_the_shipped_generation_with_defaults() {
    assert_eq!(
        generation_for_schema(schema(1)).unwrap(),
        ContainerGeneration::V2
    );
}

#[test]
fn schema_two_is_the_shipped_generation() {
    assert_eq!(
        generation_for_schema(schema(2)).unwrap(),
        ContainerGeneration::V2
    );
}

#[test]
fn schema_three_is_the_successor() {
    assert_eq!(
        generation_for_schema(schema(3)).unwrap(),
        ContainerGeneration::V3
    );
}

#[test]
fn schema_zero_is_unsupported_and_names_the_supported_sets() {
    let err = generation_for_schema(schema(0)).unwrap_err();
    let text = err.to_string();
    assert!(text.contains("1-2 (VINDEX2)"), "{text}");
    assert!(text.contains("3-4 (VINDEX3)"), "{text}");
}

/// Schema 4 is VINDEX3's current write target since `RegionSchema` gained
/// its stored-layout declaration; 5 is the first beyond the frontier.
#[test]
fn a_schema_past_the_frontier_is_unsupported_and_names_the_supported_sets() {
    let err = generation_for_schema(schema(5)).unwrap_err();
    let text = err.to_string();
    assert!(text.contains('5'), "{text}");
    assert!(text.contains("VINDEX2"), "{text}");
    assert!(text.contains("VINDEX3"), "{text}");
}

/// The point of the bump: a binary that predates the layout field supports
/// `3..=3` and therefore refuses a schema-4 container outright, rather than
/// parsing it and ignoring a field that changes what the bytes mean.
#[test]
fn schema_four_is_read_by_the_successor_and_is_what_it_writes() {
    let v3 = ContainerGeneration::V3;
    assert_eq!(v3.current_schema_version(), schema(4));
    assert!(
        v3.reads_schema(schema(3)),
        "legacy containers stay readable"
    );
    assert!(v3.reads_schema(schema(4)));
}

// ── Schema and generation are different things ─────────────────────────────

#[test]
fn a_generation_spans_more_schemas_than_it_writes() {
    // The model error the E0 regression exposed, as an assertion.
    let v2 = ContainerGeneration::V2;
    assert_eq!(v2.current_schema_version(), schema(V2_CURRENT_SCHEMA));
    assert!(v2.reads_schema(schema(V2_MIN_SCHEMA)));
    assert!(v2.reads_schema(schema(V2_CURRENT_SCHEMA)));
    assert_ne!(
        v2.supported_schema_versions().count(),
        1,
        "VINDEX2 reads more than one schema; a bijection would be wrong"
    );
}

/// Both generations now span more schemas than they write.
///
/// This replaces `the_successor_currently_reads_exactly_one_schema`, whose
/// premise the `RegionLayout` bump deliberately ended: VINDEX3 reads 3..=4
/// exactly as VINDEX2 reads 1..=2. "Spans a range" is the steady state, and
/// a singleton is only what a generation looks like before its first
/// meaning-changing addition.
#[test]
fn the_successor_spans_a_range_like_its_predecessor() {
    let v3 = ContainerGeneration::V3;
    assert_eq!(v3.current_schema_version(), schema(V3_CURRENT_SCHEMA));
    assert!(
        v3.supported_schema_versions().count() > 1,
        "VINDEX3 reads legacy schema 3 alongside the schema 4 it writes"
    );
}

#[test]
fn no_two_generations_claim_the_same_schema() {
    // Overlap would make dispatch ambiguous and the "sole discriminator"
    // property false.
    for a in ALL_GENERATIONS {
        for b in ALL_GENERATIONS {
            if a == b {
                continue;
            }
            let overlap = a
                .supported_schema_versions()
                .any(|v| b.reads_schema(schema(v)));
            assert!(!overlap, "{} and {} overlap", a.name(), b.name());
        }
    }
}

#[test]
fn every_generation_reads_the_schema_it_writes() {
    for g in ALL_GENERATIONS {
        assert!(g.reads_schema(g.current_schema_version()), "{}", g.name());
    }
}

/// The LYRW version identifies the **generation**, not the schema.
///
/// This replaces `the_lyrw_version_trails_the_current_schema_by_one`, which
/// held only by coincidence: VINDEX2 was LYRW 1 / schema 2 and VINDEX3 was
/// LYRW 2 / schema 3, so "lyrw + 1 == current_schema" and
/// "lyrw + 1 == generation number" were indistinguishable. The `RegionLayout`
/// bump separates them — VINDEX3 is LYRW 2 writing schema 4 — and the
/// relation that actually matters is the one `container_generation_for` uses
/// to tell a caller which loader to reach for.
#[test]
fn the_lyrw_version_and_the_schema_are_no_longer_locked_in_step() {
    // The relation that survives — LYRW version identifies the generation —
    // is asserted by `lyrw_versions_map_back_to_their_generation` below and
    // is not repeated here. What this pins is the *separation*: VINDEX3 is
    // LYRW 2 writing schema 4, so a reader must not derive either from the
    // other.
    let v3 = ContainerGeneration::V3;
    assert_ne!(
        v3.lyrw_format_version() + 1,
        v3.current_schema_version().get(),
        "schema and LYRW version must no longer be locked in step"
    );
    // The LYRW container format itself did not change — only what a schema
    // field means — which is why this is a schema bump and not a new
    // generation.
    assert_eq!(v3.lyrw_format_version(), 2);
}

#[test]
fn lyrw_versions_map_back_to_their_generation() {
    for g in ALL_GENERATIONS {
        assert_eq!(
            ContainerGeneration::from_lyrw_format_version(g.lyrw_format_version()).unwrap(),
            g
        );
    }
    assert!(ContainerGeneration::from_lyrw_format_version(9).is_err());
}

// ── Unified dispatch versus direct-loader refusal ──────────────────────────

#[test]
fn unified_dispatch_routes_a_legacy_schema_to_the_shipped_generation() {
    assert_eq!(
        generation_for_schema(schema(1)).unwrap(),
        ContainerGeneration::V2
    );
}

#[test]
fn the_successor_loader_refuses_a_legacy_schema_by_name() {
    // Dispatch accepts it; the wrong loader must not.
    let found = generation_for_schema(schema(1)).unwrap();
    let err = found.require(ContainerGeneration::V3).unwrap_err();
    let text = err.to_string();
    assert!(text.contains("VINDEX2"), "{text}");
    assert!(text.contains("VINDEX3"), "{text}");
}

#[test]
fn a_matching_loader_accepts() {
    assert!(ContainerGeneration::V2
        .require(ContainerGeneration::V2)
        .is_ok());
}

#[test]
fn the_summary_lists_every_generations_range() {
    let s = supported_schema_summary();
    assert!(s.contains("1-2 (VINDEX2)"), "{s}");
    assert!(s.contains("3-4 (VINDEX3)"), "{s}");
}

/// Both renderings, tested directly.
///
/// This used to rely on VINDEX3 happening to span one schema, so the
/// singleton branch lost its only cover the moment that stopped being
/// true — the same "an invariant that held by coincidence" shape as the
/// LYRW-version test above. A generation that has not yet made a
/// meaning-changing addition still renders as a singleton, so the branch
/// is live and now covered regardless of what the real generations span.
#[test]
fn a_range_and_a_singleton_render_differently() {
    assert_eq!(schema_range_label(3..=4, "VINDEX3"), "3-4 (VINDEX3)");
    assert_eq!(schema_range_label(7..=7, "VINDEX8"), "7 (VINDEX8)");
}

// ── Detection from disk ────────────────────────────────────────────────────

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir()
        .join("vindex-generation-tests")
        .join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn detection_reads_only_the_version_field() {
    // A VINDEX3 index.json carries keys the shipped config struct has never
    // seen; detection must report the generation rather than fail on shape.
    let dir = temp_dir("rich-v3");
    std::fs::write(
        dir.join(INDEX_JSON),
        r#"{"version": 3, "profiles": ["exact"], "segments": {"routed/layer_0": 2}}"#,
    )
    .unwrap();
    assert_eq!(detect_generation(&dir).unwrap(), ContainerGeneration::V3);
}

#[test]
fn a_missing_version_field_is_refused_naming_what_is_supported() {
    let dir = temp_dir("no-version");
    std::fs::write(dir.join(INDEX_JSON), r#"{"model": "x"}"#).unwrap();
    let err = detect_generation(&dir).unwrap_err();
    let text = err.to_string();
    assert!(text.contains("no version field"), "{text}");
    assert!(text.contains("VINDEX2"), "{text}");
}

#[test]
fn a_missing_index_json_is_io_not_a_generation_verdict() {
    let dir = temp_dir("absent");
    assert!(matches!(detect_generation(&dir), Err(VindexError::Io(_))));
}

#[test]
fn malformed_json_is_a_parse_error() {
    let dir = temp_dir("malformed");
    std::fs::write(dir.join(INDEX_JSON), "{not json").unwrap();
    assert!(matches!(
        detect_generation(&dir),
        Err(VindexError::Parse(_))
    ));
}

#[test]
fn a_schema_version_displays_as_its_number() {
    assert_eq!(schema(3).to_string(), "3");
    assert_eq!(schema(3).get(), 3);
}

/// **The default-flip gate's pinned test.** A fresh extraction with no
/// expressed preference writes VINDEX3: the M4 flip, decided 2026-11-06.
/// Changing `DEFAULT_EXTRACTION_GENERATION` again means changing this
/// test in the same commit — that pair is the explicit decision.
#[test]
fn auto_extraction_resolves_to_v3_after_the_m4_flip() {
    assert_eq!(
        admit_extraction_generation(GenerationRequest::Auto),
        ContainerGeneration::V3,
    );
    assert_eq!(DEFAULT_EXTRACTION_GENERATION, ContainerGeneration::V3);
}

/// Explicit requests pass through admission untouched, in both
/// directions — the escape hatch works the same before and after the
/// flip.
#[test]
fn explicit_generation_requests_are_never_overridden() {
    assert_eq!(
        admit_extraction_generation(GenerationRequest::Explicit(ContainerGeneration::V2)),
        ContainerGeneration::V2,
    );
    assert_eq!(
        admit_extraction_generation(GenerationRequest::Explicit(ContainerGeneration::V3)),
        ContainerGeneration::V3,
    );
}

/// The listing fact source answers for BOTH generations — the enabler
/// for "no V3 artifact silently disappears from a consumer surface".
#[test]
fn summarize_container_reads_either_generation() {
    use super::generation::summarize_container;

    let v2 = temp_dir("summary-v2");
    std::fs::write(
        v2.join(INDEX_JSON),
        r#"{"version":2,"model":"gemma-3-4b","num_layers":34,"hidden_size":2560}"#,
    )
    .unwrap();
    let s = summarize_container(&v2).unwrap();
    assert_eq!(s.generation, ContainerGeneration::V2);
    assert_eq!(s.model, "gemma-3-4b");
    assert_eq!(s.num_layers, 34);
    assert_eq!(s.hidden_size, 2560);

    let v3 = temp_dir("summary-v3");
    std::fs::write(
        v3.join(INDEX_JSON),
        r#"{"version":4,"model":"granite-4.1-3b","num_layers":40,"hidden_size":2560}"#,
    )
    .unwrap();
    let s = summarize_container(&v3).unwrap();
    assert_eq!(s.generation, ContainerGeneration::V3);
    assert_eq!(s.model, "granite-4.1-3b");
    assert_eq!(s.num_layers, 40);
    assert_eq!(s.hidden_size, 2560);
}

/// A container missing identity fields is listed with them blank —
/// never hidden, and never a hard failure.
#[test]
fn summarize_container_tolerates_missing_identity_fields() {
    use super::generation::summarize_container;

    let dir = temp_dir("summary-sparse");
    std::fs::write(dir.join(INDEX_JSON), r#"{"version":4}"#).unwrap();
    let s = summarize_container(&dir).unwrap();
    assert_eq!(s.generation, ContainerGeneration::V3);
    assert!(s.model.is_empty());
    assert_eq!(s.num_layers, 0);
    assert_eq!(s.hidden_size, 0);
}

/// Each generation's conventional directory suffix strips to the bare
/// name; anything else — including a bare suffix — is left alone.
#[test]
fn container_dir_suffixes_strip_to_the_model_name() {
    use super::generation::strip_container_dir_suffix;

    assert_eq!(ContainerGeneration::V2.dir_suffix(), ".vindex");
    assert_eq!(ContainerGeneration::V3.dir_suffix(), ".vindex3");
    assert_eq!(
        strip_container_dir_suffix("gemma3-4b-f16.vindex"),
        "gemma3-4b-f16"
    );
    assert_eq!(
        strip_container_dir_suffix("qwen3-0.6b.vindex3"),
        "qwen3-0.6b"
    );
    assert_eq!(strip_container_dir_suffix("plain-dir"), "plain-dir");
    assert_eq!(strip_container_dir_suffix("model.vindex2"), "model.vindex2");
    assert_eq!(strip_container_dir_suffix(".vindex3"), ".vindex3");
    assert_eq!(strip_container_dir_suffix(".vindex"), ".vindex");
}
