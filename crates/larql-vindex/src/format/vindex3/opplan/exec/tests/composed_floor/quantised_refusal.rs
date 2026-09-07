//! The witness that the evidence floors are ENFORCED, on a real
//! quantised container, under a budget that asks nothing.
//!
//! `ResidencyBudget::UNBOUNDED` is the case that used to slip through
//! entirely. The floor was consulted in one place — the search for a
//! shallower extent under preparation pressure — so with no pressure
//! there was nothing to consult it, and a caller could declare
//! `CertifiedExact`, plan a Q4_K model, and be refused nothing. The
//! requirement was real and no code ever asked it.
//!
//! Q4_K is the honest subject: its error is a property of the tensor
//! that was fitted, not of the scheme, so it declares no source-fidelity
//! bound at all. Under the settled referent — decoded values against the
//! typed logical source tensor — that is not a small error, it is NO
//! CLAIM, and the two evidence floors must say so.

use super::vocabulary::UNREACHABLE_BOUND;
use crate::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::accounting::{RepresentationFloor, ResidencyBudget};
use crate::format::vindex3::opplan::exec::operands::{
    OperandSource, OperandStore, RepresentationSource,
};
use crate::format::vindex3::opplan::exec::prepared::{select_realizations_within, ExecutionSlice};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::plan_component_ops;
use crate::format::vindex3::represent::kquant::Q4_K;
use crate::format::vindex3::represent::{compile_representation, policy, RepresentSpec};

/// Plan the dense fixture compiled to Q4_K under `floor`, asking the
/// budget for nothing else.
fn plan_q4_under(floor: RepresentationFloor) -> Result<usize, String> {
    let tmp = tempfile::tempdir().unwrap();
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let src = tmp.path().join("src.vindex3");
    let out = tmp.path().join("q4.vindex3");
    encode_fixture_container(dense_f32_model, &checkpoint, &src, "target");
    compile_representation(
        &src,
        &out,
        &RepresentSpec {
            encoding: Q4_K.name.to_string(),
            objects: Vec::new(),
            roles: policy::RolePolicy::default(),
            deployment: false,
            protect: policy::Protections::default(),
        },
    )
    .expect("Q4_K compiles the fixture");

    let inspection = inspect_container(&out, false).unwrap();
    let plan = plan_component_ops(&inspection, &out, "target")
        .unwrap()
        .plan
        .unwrap();
    // FOR the Q4_K representation: opening the container plainly resolves
    // the f32 source it was compiled from, and would test nothing.
    let store = OperandStore::open_for(
        &out,
        &inspection,
        Some(Q4_K.name),
        RepresentationSource::Stored,
    )
    .expect("the compiled pack binds");
    select_realizations_within(
        &plan,
        OperandSource::from(&store),
        &ProductionBackend::new(),
        &ExecutionSlice::Full,
        &ResidencyBudget::UNBOUNDED.with_fidelity(floor),
    )
    .map(|records| records.len())
    .map_err(|e| e.to_string())
}

/// The structural floor proceeds: every pin is on the whole artifact,
/// which is all it ever asked. It makes no source-fidelity claim, and
/// that is precisely why it can be the default.
#[test]
fn terminal_extent_plans_a_quantised_model_without_claiming_anything_about_it() {
    let planned = plan_q4_under(RepresentationFloor::TerminalExtent)
        .expect("the complete stored representation is exactly what a Q4_K pin holds");
    assert!(planned > 0, "the fixture planned operands");
}

/// **The witness.** The same container, the same unbounded budget, and a
/// requirement Q4_K cannot answer.
#[test]
fn certified_exact_refuses_an_unbounded_quantised_plan() {
    let refusal = plan_q4_under(RepresentationFloor::CertifiedExact)
        .expect_err("Q4_K states no source-fidelity bound, so it cannot be certified exact");
    assert!(
        refusal.contains("fidelity requirement"),
        "the refusal names the requirement: {refusal}"
    );
    assert!(
        refusal.contains("states no source-fidelity bound"),
        "and why the operand cannot meet it: {refusal}"
    );
    assert!(
        refusal.contains(Q4_K.name),
        "and which representation is at fault: {refusal}"
    );
    assert!(
        refusal.contains("verified attestation"),
        "and the one thing that could supply it: {refusal}"
    );
}

/// And no bound is loose enough to buy a claim that was never made — an
/// absent certificate is not a large radius, it is no radius.
#[test]
fn within_refuses_a_quantised_plan_at_any_bound() {
    let refusal = plan_q4_under(RepresentationFloor::Within(UNREACHABLE_BOUND))
        .expect_err("an absent claim cannot satisfy a bound, however generous");
    assert!(
        refusal.contains("states no source-fidelity bound"),
        "{refusal}"
    );
}
