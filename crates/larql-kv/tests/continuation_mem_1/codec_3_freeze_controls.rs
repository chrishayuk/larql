//! CONTINUATION-CODEC-3 W1: controls for the freeze's authority, the
//! bound verdict and the new guards. Hand-built vectors and fixtures only —
//! no model arm runs here.

use std::collections::BTreeMap;

use larql_kv::CodecRecentKvState;
use larql_vindex::format::vindex3::represent::measure::plan::metrics::PositionMetrics;

use super::codec_2::{BANK_LEN, DECODE, RUNGS};
use super::codec_3_freeze::{
    adjudicate_bound, bank3, freeze, spec, Binding, Provider, ARMS, CODEC_BITS,
};
use super::codec_3_freeze_arms::{codec_residency_guard, mixed_residency_guard, trace_guard};
use super::*;

/// A scored position against R.
fn pos(position: usize, kl: f64) -> PositionMetrics {
    PositionMetrics {
        sample: 0,
        position,
        category: "fixture".into(),
        kl,
        top1_agree: true,
        top5_overlap: 5,
        delta_nll: None,
        reference_margin: 0.95,
        reference_entropy: 0.0,
        max_abs_delta: 0.0,
        mean_abs_delta: 0.0,
    }
}

/// The frozen ladder's shape: one stratum per rung, DECODE positions each,
/// every position confident (so B4 has support), KL `kl`.
fn strata(kl: f64) -> Vec<Vec<PositionMetrics>> {
    RUNGS
        .iter()
        .map(|&n| (0..DECODE).map(|i| pos(n - DECODE + i, kl)).collect())
        .collect()
}

/// C, and a candidate that matches C (passes every rule) or adds damage
/// far past ε (fails B2 and B3).
const C_KL: f64 = 0.1;
const FAILING_EXTRA_KL: f64 = 1.0;

fn b(f: &mut Value) -> &mut Value {
    &mut f["acceptance"]["rule_binding"]
}

fn verdict_of(v: &Value) -> &str {
    v["verdict"].as_str().unwrap_or_default()
}

#[test]
fn the_binding_is_read_from_the_freeze_as_frozen() {
    let _serial = serial();
    let binding = Binding::from_freeze(&freeze()).unwrap();
    assert_eq!(
        binding,
        Binding {
            inherited_symbol: "CH".into(),
            primary: "CH-recent256".into(),
            secondary_report_only: "CH-recent128".into(),
            control_report_only: "CH".into(),
        }
    );
}

/// The arm table is the freeze's: the same names as its compressed arms,
/// and each runs what the freeze's description says.
#[test]
fn the_arm_table_is_the_freezes() {
    let _serial = serial();
    let freeze = freeze();
    let declared: Vec<&String> = freeze["acceptance"]["arms"]
        .as_object()
        .unwrap()
        .keys()
        .filter(|k| !["R", "NULL", "C", "dropped"].contains(&k.as_str()))
        .collect();
    let mut table: Vec<&str> = ARMS.iter().map(|a| a.name).collect();
    let mut declared: Vec<&str> = declared.iter().map(|s| s.as_str()).collect();
    table.sort_unstable();
    declared.sort_unstable();
    assert_eq!(table, declared);
    for a in ARMS {
        let text = freeze["acceptance"]["arms"][a.name].as_str().unwrap();
        let runs = match a.provider {
            Provider::Codec => format!("codec/v1 bits {CODEC_BITS}"),
            Provider::Recent { window } => {
                format!("codec-recent/v1, bits {CODEC_BITS}, exact_recent_k {window}")
            }
        };
        assert!(
            text.contains(&runs),
            "{}: `{text}` does not say `{runs}`",
            a.name
        );
    }
}

#[test]
fn a_malformed_binding_is_refused() {
    let _serial = serial();
    let base = freeze();
    let refused = |edit: &dyn Fn(&mut Value)| {
        let mut f = base.clone();
        edit(&mut f);
        Binding::from_freeze(&f).unwrap_err()
    };
    let missing = refused(&|f| {
        b(f).as_object_mut().unwrap().remove("primary");
    });
    assert!(missing.contains("primary"), "{missing}");
    let undeclared = refused(&|f| b(f)["primary"] = json!("CH-recent64"));
    assert!(undeclared.contains("not a declared"), "{undeclared}");
    let symbol = refused(&|f| b(f)["inherited_symbol"] = json!("H"));
    assert!(symbol.contains("inherited symbol"), "{symbol}");
    let duplicate = refused(&|f| b(f)["secondary_report_only"] = json!("CH-recent256"));
    assert!(duplicate.contains("distinct"), "{duplicate}");
    assert!(spec("CH-recent64").is_none());
}

#[test]
fn bank_3_is_the_one_the_freeze_names() {
    let _serial = serial();
    let freeze = freeze();
    let ids = bank3(&freeze).unwrap();
    assert_eq!(ids.len(), BANK_LEN);
    assert_eq!(ids[0], 2, "position 0 is <bos>");
    let mut other = freeze.clone();
    other["bank"]["ids_sha256_u32le"] = json!("5120445c411424f8");
    assert!(bank3(&other).unwrap_err().contains("names bank"));
}

fn bound(primary_kl: f64, secondary_kl: f64) -> Value {
    let binding = Binding::from_freeze(&freeze()).unwrap();
    let arms: BTreeMap<String, Vec<Vec<PositionMetrics>>> = [
        (binding.primary.clone(), strata(primary_kl)),
        (binding.secondary_report_only.clone(), strata(secondary_kl)),
        (binding.control_report_only.clone(), strata(C_KL)),
    ]
    .into_iter()
    .collect();
    adjudicate_bound(&binding, &strata(C_KL), &arms)
}

/// Secondary isolation, both directions: the report-only arm can neither
/// rescue a failed primary nor poison a passing one.
#[test]
fn the_verdict_is_the_primarys_alone_in_both_directions() {
    let _serial = serial();
    let rescue = bound(C_KL + FAILING_EXTRA_KL, C_KL);
    assert_eq!(verdict_of(&rescue["verdict"]), "NotAcceptable");
    assert_eq!(
        verdict_of(&rescue["reported_not_adjudicated"]["secondary"]),
        "Acceptable"
    );

    let poison = bound(C_KL, C_KL + FAILING_EXTRA_KL);
    assert_eq!(verdict_of(&poison["verdict"]), "Acceptable");
    assert_eq!(
        verdict_of(&poison["reported_not_adjudicated"]["secondary"]),
        "NotAcceptable"
    );
    assert_eq!(poison["rule_binding"]["primary"], "CH-recent256");
}

#[test]
fn the_trace_and_residency_guards_stop_on_a_fault() {
    let _serial = serial();
    let ch = json!(["1", 10]);
    trace_guard("CH-recent256", 2_048, &ch, &json!(["1", 10])).unwrap();
    let moved = trace_guard("CH-recent256", 2_048, &ch, &json!(["2", 10])).unwrap_err();
    assert!(moved.contains("provider fault"), "{moved}");
    assert!(codec_residency_guard("CH", 2_048, None).is_err());
    assert!(mixed_residency_guard("CH-recent256", 2_048, None).is_err());
}

/// Executed: a provider whose window is wider than the one it is measured
/// against misses its declaration, and the guard stops on it.
#[test]
fn the_mixed_residency_guard_stops_on_an_undeclared_window() {
    let _serial = serial();
    let subject = subjects::fixture(miniature_glimmer, "codec3-w1-residency");
    let backend = ReferenceBackend::new();
    let ops = subject.prepare(&backend);
    let journey = Journey {
        prefill: G_TOKENS.to_vec(),
        resume: vec![5, 9, 13],
        decode: vec![1, 2, 3, 4],
    };
    let mut kv = Measured::new(CodecRecentKvState::new(CODEC_BITS, 4));
    subjects::run(&subject, &ops, &backend, &mut kv, &journey);
    mixed_residency_guard("CH-recent4", 12, kv.mixed_residency(4)).unwrap();
    let stop = mixed_residency_guard("CH-recent2", 12, kv.mixed_residency(2)).unwrap_err();
    assert!(stop.contains("residency regression"), "{stop}");
}
