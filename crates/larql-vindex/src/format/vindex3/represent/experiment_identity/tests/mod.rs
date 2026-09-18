//! A declaration decides; the environment only supplies the candidate.
//! Each test names one way the REAL-EVIDENCE-1 smoke could have been
//! caught, and one way a false refusal must not happen.

use std::collections::BTreeMap;

use super::super::observation_stream::RuntimeScope;
use super::{ByteFootprint, DeclaredIdentity, EXPECT_IDENTITY_ENV};

const FLAGSHIP_RUN: &str = "kda-q8-l20-21-22-24-25-x-kimi-map-l20-26q80";
const FLAGSHIP_CANDIDATE: &str = "kimi-map-l20-26q80";
const FLAGSHIP_BF16: usize = 377_487_360;
const FLAGSHIP_Q8: usize = 200_540_160;

/// The declaration as the programme writes it: no `raw_env`, layers in
/// whatever order a human typed them.
const DECLARATION: &str = r#"{
  "run": "kda-q8-l20-21-22-24-25-x-kimi-map-l20-26q80",
  "expert_candidate": "kimi-map-l20-26q80",
  "scope": {
    "kda_q8_layers": [25, 20, 21, 24, 22],
    "mla_q8_layers": [],
    "shared_q8_layers": [],
    "lm_head_q8": false
  },
  "bytes": { "bf16": 377487360, "q8_0": 200540160 },
  "provenance": { "historical_report": "kimi_flagship-selection-8192_report.json" }
}"#;

fn declared() -> DeclaredIdentity {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("identity.json");
    std::fs::write(&path, DECLARATION).unwrap();
    DeclaredIdentity::load(&path).unwrap()
}

fn resolved_flagship(raw: BTreeMap<String, String>) -> RuntimeScope {
    RuntimeScope::resolved(vec![20, 21, 22, 24, 25], vec![], vec![], false, raw)
}

#[test]
fn a_declaration_needs_no_raw_env_and_is_normalised_on_load() {
    let d = declared();
    assert_eq!(d.scope.kda_q8_layers, vec![20, 21, 22, 24, 25]);
    assert!(d.scope.raw_env.is_empty());
    assert_eq!(d.run, FLAGSHIP_RUN);
    assert_eq!(d.expert_candidate.as_deref(), Some(FLAGSHIP_CANDIDATE));
    assert_eq!(
        d.bytes,
        ByteFootprint {
            bf16: FLAGSHIP_BF16,
            q8_0: FLAGSHIP_Q8
        }
    );
    assert!(d.provenance.is_some(), "provenance travels, uncompared");
}

#[test]
fn the_historical_arm_passes_both_stages_whatever_the_env_spelling() {
    let d = declared();
    let mut raw = BTreeMap::new();
    raw.insert(
        "LARQL_KDA_Q8_LAYER".to_string(),
        "25,24,22,21,20".to_string(),
    );
    d.check_transition(
        FLAGSHIP_RUN,
        Some(FLAGSHIP_CANDIDATE),
        &resolved_flagship(raw),
    )
    .expect("the same transition in a different spelling is the same experiment");
    d.check_bytes(FLAGSHIP_BF16, FLAGSHIP_Q8).unwrap();
}

/// THE INCIDENT. A shell still carrying the previous arm's MLA and head
/// flags resolves to a different run, a different scope and a different
/// footprint — and every one of those must be named, before any layer
/// is loaded.
#[test]
fn the_wrong_intervention_is_refused_at_stage_one_naming_every_difference() {
    let d = declared();
    let scope = RuntimeScope::resolved(
        vec![20, 21, 22, 24, 25],
        vec![23, 26],
        vec![],
        true,
        BTreeMap::new(),
    );
    let refusal = d
        .check_transition(
            "kda-q8-l20-21-22-24-25-x-kimi-map-l20-26q80-mla23-26-headq8",
            Some(FLAGSHIP_CANDIDATE),
            &scope,
        )
        .expect_err("a stale environment must be refused");
    let text = refusal.to_string();
    assert_eq!(refusal.differences.len(), 2, "{text}");
    assert!(text.contains("run: declared"), "{text}");
    assert!(text.contains("lm_head_q8 true"), "{text}");
    assert!(text.contains("mla [23, 26]"), "{text}");
    assert!(text.contains("2 field(s) differ"), "{text}");
}

#[test]
fn the_wrong_footprint_is_refused_at_stage_two() {
    let d = declared();
    let refusal = d
        .check_bytes(493_944_832, 262_408_192)
        .expect_err("the smoke's footprint is not the flagship's");
    assert_eq!(refusal.differences.len(), 2);
    assert!(refusal.differences[0].starts_with("bytes.bf16: declared 377487360"));
    assert!(refusal.differences[1].starts_with("bytes.q8_0: declared 200540160"));
}

#[test]
fn a_missing_overlay_is_a_difference_not_a_default() {
    let d = declared();
    let refusal = d
        .check_transition(
            "kda-q8-l20-21-22-24-25",
            None,
            &resolved_flagship(BTreeMap::new()),
        )
        .expect_err("no overlay is a different experiment");
    assert!(refusal
        .differences
        .iter()
        .any(|x| x.starts_with("expert_candidate: declared Some(")));
}

#[test]
fn an_unreadable_or_malformed_declaration_is_an_error_not_a_pass() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("absent.json");
    assert!(DeclaredIdentity::load(&missing)
        .unwrap_err()
        .contains("absent.json"));
    let bad = dir.path().join("bad.json");
    std::fs::write(&bad, b"{\"run\": 1}").unwrap();
    assert!(DeclaredIdentity::load(&bad)
        .unwrap_err()
        .contains("bad.json"));
}

#[test]
fn the_env_name_is_the_one_the_runner_reads() {
    assert_eq!(EXPECT_IDENTITY_ENV, "LARQL_Q2A_EXPECT_IDENTITY");
}
