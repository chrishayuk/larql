//! **The teacher-forced runner, as a witness rather than an
//! implementation.**
//!
//! Every validity condition this file used to assert now lives in
//! `represent::measure::teacher_forced` as a typed refusal — see that
//! module's conservation inventory for where each one went. What
//! remains here is the claim a test is entitled to make:
//!
//! > this known experiment should qualify.
//!
//! plus two checks that were never about the measurement's validity and
//! would have muddled ownership by moving: the sub-4096 refusal is a
//! claim about [`QualityGate::evaluate`], and the operand-removal
//! control is a claim about `verify_complete`.
//!
//! ```text
//! LARQL_KIMI_VINDEX3=~/chris-models/Kimi-Linear-48B-A3B-Instruct.aligned.vindex3 \
//! LARQL_KIMI_Q6_CANDIDATE=/tmp/kimi-q80-l25.vindex3 \
//! LARQL_KIMI_QUALITY_BANK=/tmp/kimi_quality_bank \
//! LARQL_Q2A_SEQUENCES=8 LARQL_Q2A_LABEL=q80-l25 \
//!   cargo test -p larql-vindex --features gpu --release --lib q2a_teacher_forced -- --nocapture
//! ```

use crate::format::vindex3::opplan::exec::kimi_source::{verify_complete, CandidateOverlay};
use crate::format::vindex3::represent::measure::teacher_forced::measure_teacher_forced;
use crate::format::vindex3::represent::measure::{
    TeacherForcedRequest, BANK_ENV, CANDIDATE_ENV, SOURCE_ENV,
};
use crate::format::vindex3::represent::quality::Criterion;

#[test]
fn the_teacher_forced_measurement_qualifies_and_the_gate_refuses_on_positions() {
    let Some(request) = TeacherForcedRequest::from_env() else {
        eprintln!("skipped: set {SOURCE_ENV}, {CANDIDATE_ENV} and {BANK_ENV}");
        return;
    };
    let receipt = measure_teacher_forced(&request).expect("the measurement is admissible");

    // The whole of this test's claim about validity: the production
    // procedure checked everything, and says so.
    assert!(
        receipt.qualifies(),
        "the run skipped a validity condition: {:?}",
        receipt.verified
    );
    assert_eq!(
        receipt.gate.id, request.gate,
        "judged by the gate requested"
    );
    assert_eq!(receipt.verified.gate_evaluated, request.gate);
    eprintln!(
        "[q2a] {} positions, gate {}, verdict {}; report at {}",
        receipt.verified.positions,
        receipt.gate.id,
        if receipt.verdict_passed {
            "PASS"
        } else {
            "FAIL"
        },
        receipt.report_path
    );

    // ── A claim about the GATE, not about the measurement. ──
    if receipt.bank.positions < 4096 {
        assert!(
            !receipt.verdict_passed,
            "a sub-4096-position bank can never pass this gate"
        );
        assert!(
            receipt
                .verdict_failures
                .iter()
                .any(|f| f.starts_with(Criterion::Positions.name()) && f.contains("< 4096")),
            "the refusal must name the positions criterion: {:?}",
            receipt.verdict_failures
        );
        assert!(
            !receipt
                .verdict_failures
                .iter()
                .any(|f| f.starts_with(Criterion::CoveredMass.name())),
            "this bank's truncation must be wide enough to judge: {:?}",
            receipt.verdict_failures
        );
    }

    // ── A claim about `verify_complete`, not about this run: removing
    //    one compiled operand a candidate route consumed must refuse at
    //    load rather than fall back to source bytes. ──
    let overlay = CandidateOverlay::open(
        &request.candidate,
        &request.source,
        &crate::format::vindex3::opplan::exec::kimi_source::KimiSourceModel::open(&request.source)
            .expect("source opens")
            .geometry,
    )
    .expect("candidate overlay opens");
    let geometry =
        crate::format::vindex3::opplan::exec::kimi_source::KimiSourceModel::open(&request.source)
            .expect("source opens")
            .geometry;
    let (key, tensor) = overlay
        .index
        .ledger
        .sealed
        .iter()
        .next()
        .map(|(k, seal)| (k.clone(), seal.tensor.clone()))
        .expect("a compiled overlay seals at least one operand");
    let mut mutated = overlay.index.clone();
    mutated.ledger.sealed.remove(&key);
    let refusal = verify_complete(&mutated, &geometry).expect_err(
        "an incomplete compiled bank must refuse at load, never fall back to source bytes",
    );
    assert!(
        format!("{refusal}").contains(&tensor),
        "the refusal must name the missing operand: {refusal}"
    );
    verify_complete(&overlay.index, &geometry).expect("control: the untouched ledger verifies");
}
