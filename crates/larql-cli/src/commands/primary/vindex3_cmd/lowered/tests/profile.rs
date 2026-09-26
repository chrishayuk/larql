//! The `--profile` ledger renders what it was given: per-stage ms, the
//! byte floor per class, the sampling distortion line, and a loud
//! overflow warning.

use std::collections::BTreeMap;

use larql_compute_metal::lowering::profile::{Stage, StageProfile};

use super::super::profile::{StageBytes, StageLedger};

fn token(ms: &[(Stage, f64)], span_ms: f64, overflowed: usize) -> StageProfile {
    StageProfile {
        stage_ns: ms.iter().map(|(s, v)| (*s, (v * 1e6) as u64)).collect(),
        stage_runs: ms.iter().map(|(s, _)| (*s, 1u32)).collect(),
        span_ns: (span_ms * 1e6) as u64,
        overflowed,
    }
}

#[test]
fn empty_ledger_says_so() {
    let ledger = StageLedger::default();
    assert_eq!(
        ledger.render(),
        vec!["profile: no tokens recorded".to_string()]
    );
}

#[test]
fn render_prices_byte_classes_and_averages_over_tokens() {
    let mut ledger = StageLedger {
        bytes: StageBytes {
            attn_proj: 367_000_000, // exactly 1.00 ms at the 367 GB/s ceiling
            experts: 734_000_000,   // 2.00 ms floor
            ..Default::default()
        },
        ..Default::default()
    };
    // Two tokens: attn.proj 2 ms, experts 4 ms each; span 7 ms; GPU 7.5.
    let t = token(&[(Stage::AttnProj, 2.0), (Stage::Experts, 4.0)], 7.0, 0);
    ledger.record(&t, 7.5);
    ledger.record(&t, 7.5);
    let lines = ledger.render();
    let text = lines.join("\n");
    assert!(text.contains("over 2 token(s)"), "{text}");
    // attn.proj: 2.000 ms, 367 MB, 184 GB/s (2× the floor), floor 1.00.
    let proj = lines
        .iter()
        .find(|l| l.contains("attn.proj"))
        .expect("attn.proj row");
    assert!(proj.contains("2.000"), "{proj}");
    assert!(proj.contains("367.0"), "{proj}");
    assert!(proj.contains("184"), "{proj}");
    assert!(proj.contains("1.00"), "{proj}");
    // experts: 4 ms over a 2 ms floor.
    let experts = lines
        .iter()
        .find(|l| l.contains("ffn.experts"))
        .expect("experts row");
    assert!(
        experts.contains("4.000") && experts.contains("2.00"),
        "{experts}"
    );
    // A stage with no byte class prints dashes.
    let mut no_bytes = StageLedger::default();
    no_bytes.record(&token(&[(Stage::AttnNorm, 0.5)], 0.5, 0), 0.5);
    let norm = no_bytes
        .render()
        .into_iter()
        .find(|l| l.contains("attn.norm"))
        .expect("norm row");
    assert!(norm.contains(" - "), "{norm}");
    // Totals: attributed 6 ms, floor 3 ms, residual 3 ms; gaps 1 ms.
    assert!(text.contains("attributed       6.000"), "{text}");
    assert!(text.contains("residual over byte floor: 3.000"), "{text}");
    assert!(text.contains("gaps between stages 1.000"), "{text}");
    assert!(text.contains("command-buffer GPU span 7.500"), "{text}");
    assert!(!text.contains("WARNING"), "{text}");
}

#[test]
fn render_warns_on_overflow() {
    let mut ledger = StageLedger::default();
    ledger.record(&token(&[(Stage::Head, 1.0)], 1.0, 3), 1.0);
    let text = ledger.render().join("\n");
    assert!(text.contains("WARNING: 3 stage run(s)"), "{text}");
}

#[test]
fn stage_bytes_total_and_class_mapping() {
    let b = StageBytes {
        attn_proj: 1,
        attn_out: 2,
        dense_ffn: 4,
        dense_gate_up: 3,
        dense_down: 1,
        experts: 8,
        head: 16,
    };
    // The gate/up and down parts partition `dense_ffn`; the total counts
    // the class once.
    assert_eq!(b.total(), 31);
    let _ = BTreeMap::<Stage, u64>::new();
}

/// When the lowering ran the projections as their own stages, their bytes
/// are priced there and the glue stages that used to carry them print
/// dashes — no byte is priced twice.
#[test]
fn split_projection_stages_take_the_bytes_from_their_glue() {
    let mut ledger = StageLedger {
        bytes: StageBytes {
            attn_out: 367_000_000,
            dense_ffn: 1_101_000_000,
            dense_gate_up: 734_000_000,
            dense_down: 367_000_000,
            ..Default::default()
        },
        ..Default::default()
    };
    let t = token(
        &[
            (Stage::AttnOut, 0.1),
            (Stage::AttnOProj, 2.0),
            (Stage::DenseFfn, 0.2),
            (Stage::FfnGateUp, 4.0),
            (Stage::FfnDown, 1.0),
        ],
        7.3,
        0,
    );
    ledger.record(&t, 7.3);
    let lines = ledger.render();
    let row = |label: &str| {
        lines
            .iter()
            .find(|l| l.trim_start().starts_with(label))
            .unwrap_or_else(|| panic!("{label} row in {lines:?}"))
            .clone()
    };
    // attn.o_proj: 367 MB over 2 ms = 184 GB/s; its glue prints dashes.
    assert!(row("attn.o_proj").contains("184"), "{lines:?}");
    assert!(row("attn.out ").contains(" - "), "{lines:?}");
    // ffn.gate_up: 734 MB over 4 ms = 184 GB/s; ffn.down 367 MB over 1 ms.
    assert!(row("ffn.gate_up").contains("734.0"), "{lines:?}");
    assert!(row("ffn.gate_up").contains("184"), "{lines:?}");
    assert!(row("ffn.down").contains("367"), "{lines:?}");
    assert!(row("ffn.dense").contains(" - "), "{lines:?}");
}

/// Without the split stages the lumped classes keep their bytes.
#[test]
fn lumped_stages_keep_their_bytes_without_the_split() {
    let mut ledger = StageLedger {
        bytes: StageBytes {
            attn_out: 367_000_000,
            dense_ffn: 1_101_000_000,
            dense_gate_up: 734_000_000,
            dense_down: 367_000_000,
            ..Default::default()
        },
        ..Default::default()
    };
    ledger.record(
        &token(&[(Stage::AttnOut, 1.0), (Stage::DenseFfn, 3.0)], 4.0, 0),
        4.0,
    );
    let lines = ledger.render();
    let find = |label: &str| {
        lines
            .iter()
            .find(|l| l.trim_start().starts_with(label))
            .unwrap_or_else(|| panic!("{label} row in {lines:?}"))
            .clone()
    };
    assert!(find("attn.out").contains("367.0"), "{lines:?}");
    assert!(find("ffn.dense").contains("1101.0"), "{lines:?}");
}
