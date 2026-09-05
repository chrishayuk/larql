//! QW-3.5C: the fused query/gate projection, end to end.
//!
//! Qwen3.8's `q_proj` carries `2 · num_heads · head_dim` rows — query and
//! gate interleaved per head — and `o_proj` is still sized by the plain
//! attention width. Two things have to hold for that to be represented
//! rather than merely tolerated:
//!
//! 1. a container whose projection is double-width must PLAN, ENCODE and
//!    CLOSE, without a separate gate operand that does not exist;
//! 2. a container claiming the gate while shipping an ordinary-width
//!    projection must be REFUSED — otherwise carriage is just believing
//!    `attn_output_gate: true`, and the tensor geometry witnesses nothing.
//!
//! The layout/order mutation table (`ContiguousHalves`, `SwapPerHeadQGate`,
//! `GateGetsQNorm`, `GateGetsRoPe`, `SiluGate`, `GateAfterOProj`, `NoGate`)
//! is NOT here yet, and until it is, this file proves geometry and closure
//! only — not that the runtime reads those 128 rows as
//! `[q_h0 | gate_h0 | q_h1 | …]` rather than the equally well-shaped
//! `[all q | all gate]`. Said plainly so the coverage is not overread.

use crate::format::vindex3::encode::encode_system_unenforced as encode_system;
use crate::format::vindex3::fixtures::{gated_q_f32_model, DENSE_HEAD_DIM, DENSE_Q_HEADS};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::plan_component_ops;
use crate::format::vindex3::plan::plan_system;
use larql_models::config::GateSource;

/// A double-width projection plans, encodes and closes — and the gate op
/// names the query projection as its own operand.
#[test]
fn a_fused_query_projection_plans_encodes_and_closes() {
    let dir = tempfile::tempdir().unwrap();
    gated_q_f32_model(dir.path());
    let inventory = larql_models::inventory::build_inventory(dir.path()).unwrap();

    let plan = plan_system(&[("gated".to_string(), inventory.clone())]);
    let blocking: Vec<String> = plan
        .artifacts
        .iter()
        .flat_map(|a| &a.findings)
        .filter(|f| f.blocks())
        .map(|f| format!("{}: {}", f.subject, f.detail))
        .collect();
    assert!(plan.admissible, "blocking: {blocking:#?}");

    let container = tempfile::tempdir().unwrap();
    encode_system(&[("gated".to_string(), inventory)], container.path()).unwrap();
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    assert!(outcome.closed(), "defects: {:#?}", outcome.defects);

    let op = outcome.plan.unwrap().layers[0]
        .attention
        .softmax()
        .expect("a softmax layer")
        .output_gate
        .clone()
        .expect("the layer carries a gate");
    assert_eq!(op.spec.source, GateSource::FusedQueryProjection);
    // No separate gate tensor exists; the op names the query projection
    // and reads one matrix for both roles.
    assert!(
        op.projection.tensor.ends_with("q_proj.weight"),
        "{:?}",
        op.projection
    );
    assert_eq!(
        op.projection.shape,
        vec![DENSE_Q_HEADS * DENSE_HEAD_DIM * 2, 64],
        "the gate's operand is the DOUBLE-width projection"
    );
}

/// **The negative control that stops carriage from merely believing the
/// config.**
///
/// Same config — still `attn_output_gate: true` — but the stored
/// projection is ordinary width. The gate has no rows to live in, and
/// closure must say so. Without this, `attn_output_gate` would be
/// "carried" on a checkpoint that cannot possibly execute a gate.
#[test]
fn a_declared_gate_without_the_rows_to_hold_it_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    gated_q_f32_model(dir.path());
    // Halve the projection back to the ungated width, leaving the
    // declaration intact.
    crate::format::vindex3::fixtures::shrink_q_proj_to_ungated_width(dir.path());

    let inventory = larql_models::inventory::build_inventory(dir.path()).unwrap();
    let container = tempfile::tempdir().unwrap();
    let encoded = encode_system(&[("gated".to_string(), inventory)], container.path());
    // Encode ACCEPTS this container — the defect is not a malformed
    // file, it is a semantic one — so the refusal must come from operand
    // closure. Asserted rather than tolerated: an `if let Err(_) =
    // encoded { return }` escape hatch here would let the test pass
    // without ever reaching the assertions below.
    encoded.expect("encode accepts the container; the gate defect is semantic, caught at closure");
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    assert!(
        !outcome.closed(),
        "a gate declared over an ungated-width projection must not close"
    );
    assert!(
        outcome
            .defects
            .iter()
            .any(|d| format!("{d:?}").contains("q_proj")),
        "the defect must name the projection: {:#?}",
        outcome.defects
    );
}

/// Every semantic mutation of the fused gate path, measured against the
/// shipped implementation.
///
/// Mutations are threaded through `ReferenceBackend::attention_mutated`,
/// which is the same code `PlanBackend::attention` runs — the trait
/// method is a one-line wrapper passing `GateMutation::None`. Nothing
/// here re-implements the operator, so an arm that fails to move is a
/// statement about the shipped path and not about a copy of it.
///
/// No magnitude is pre-registered. The requirement is only that each arm
/// leaves the control's neighbourhood on at least one of rel-RMS or
/// cosine, which is what "the executor can tell these semantics apart"
/// actually means.
mod mutation_table {
    use super::*;
    use crate::format::vindex3::opplan::exec::kernels::GateMutation;
    use crate::format::vindex3::opplan::exec::operands::OperandStore;
    use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
    use crate::format::vindex3::opplan::exec::AttentionOperands;

    /// Control plus every arm, run through one gated layer.
    fn outputs_for(mutation: GateMutation) -> Vec<f32> {
        let dir = tempfile::tempdir().unwrap();
        gated_q_f32_model(dir.path());
        let inventory = larql_models::inventory::build_inventory(dir.path()).unwrap();
        let container = tempfile::tempdir().unwrap();
        encode_system(&[("gated".to_string(), inventory)], container.path()).unwrap();
        let inspection = inspect_container(container.path(), false).unwrap();
        let plan = plan_component_ops(&inspection, container.path(), "target")
            .unwrap()
            .plan
            .unwrap();
        let store = OperandStore::open(container.path(), &inspection).unwrap();
        let layer = &plan.layers[0];
        let op = layer.attention.softmax().unwrap();

        // Eight positions, not three: the rotary arms are position-driven
        // and a 3-token batch (positions 0,1,2 — one of which does not
        // rotate at all) makes GateGetsRoPe far quieter than it should be.
        // A control that is barely above its own floor is a weak control.
        let hidden = crate::format::vindex3::fixtures::DENSE_HIDDEN;
        let inputs: Vec<Vec<f32>> = (0..8)
            .map(|p| crate::format::vindex3::fixtures::lcg_values(hidden, 7000 + p as u64))
            .collect();
        // Operands and the call are built by the INTERPRETER's own
        // loader and `AttentionOperands::call`, not by a transcription
        // here: a test that assembled its own call could disagree with
        // the shipped one about which tensor is which and never notice.
        let operands = AttentionOperands::load(
            op,
            (&store).into(),
            &|_: &crate::format::vindex3::opplan::OperandRef| {
                Ok(crate::format::vindex3::opplan::exec::backend::WeightFormat::F32)
            },
        )
        .unwrap();
        let call = operands.call(op, &inputs, layer.declared_norm_eps, hidden);
        ReferenceBackend
            .attention_mutated(call, mutation)
            .unwrap()
            .outputs
            .concat()
    }

    fn rel_rms(a: &[f32], b: &[f32]) -> f32 {
        let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
        let den: f32 = b.iter().map(|y| y * y).sum();
        (num / den.max(f32::MIN_POSITIVE)).sqrt()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|y| y * y).sum::<f32>().sqrt();
        dot / (na * nb).max(f32::MIN_POSITIVE)
    }

    #[test]
    fn every_semantic_mutation_of_the_fused_gate_is_distinguishable() {
        let control = outputs_for(GateMutation::None);
        assert!(
            control.iter().any(|v| v.abs() > 1e-6),
            "an all-zero control cannot discriminate anything"
        );

        let arms = [
            GateMutation::ContiguousHalves,
            GateMutation::SwapPerHeadQGate,
            GateMutation::GateGetsQNorm,
            GateMutation::GateGetsRoPe,
            GateMutation::SiluGate,
            GateMutation::NoGate,
            GateMutation::GateAfterOProj,
        ];
        println!("\n  mutation              rel_rms        cosine");
        println!("  ------------------------------------------------");
        println!(
            "  {:<20}  {:>10.3e}  {:>10.7}",
            "None (control)", 0.0, 1.0_f32
        );
        for arm in arms {
            let got = outputs_for(arm);
            let (r, c) = (rel_rms(&got, &control), cosine(&got, &control));
            println!("  {:<20}  {:>10.3e}  {:>10.7}", format!("{arm:?}"), r, c);
            assert!(
                r > 1e-4 || c < 0.9999,
                "{arm:?} left the shipped output unchanged (rel_rms {r:e}, cos {c:.7}) — \
                 the executor cannot tell this semantics apart, so the gate's meaning is \
                 not actually pinned"
            );
        }
    }

    /// **The headline.** A contiguous split keeps every dimension valid
    /// and every closure check satisfied; only the values move. If this
    /// arm is quiet, the executor is not reading the checkpoint's row
    /// semantics — it is merely reading rows.
    #[test]
    fn a_contiguous_split_is_loudly_wrong() {
        let control = outputs_for(GateMutation::None);
        let contiguous = outputs_for(GateMutation::ContiguousHalves);
        assert_eq!(
            control.len(),
            contiguous.len(),
            "the wrong layout must stay perfectly well-shaped — that is what makes it dangerous"
        );
        let r = rel_rms(&contiguous, &control);
        assert!(
            r > 1e-2,
            "contiguous halves must be loud, not marginal: {r:e}"
        );
    }

    /// **Why the table needs two metrics.**
    ///
    /// `NoGate` scales every output by a positive factor, so its cosine
    /// against the control is ~1.0 while its rel-RMS is ~1.0 — a
    /// cosine-only gate would wave it through. `GateGetsRoPe` and
    /// `GateAfterOProj` are the opposite shape: quiet in rel-RMS terms
    /// and only a few parts in ten thousand off in cosine.
    ///
    /// This is the same finding the Gated DeltaNet table produced with
    /// `ScaleBeforeNorm` (output 10x wrong, cosine exactly 1.0000000), and
    /// it is asserted rather than remarked on so that a future
    /// simplification to one metric fails here.
    #[test]
    fn neither_metric_alone_would_catch_every_arm() {
        let control = outputs_for(GateMutation::None);
        let no_gate = outputs_for(GateMutation::NoGate);
        assert!(
            cosine(&no_gate, &control) > 0.999,
            "NoGate is a pure positive rescale; if cosine ever separates it, this \
             justification is stale"
        );
        assert!(
            rel_rms(&no_gate, &control) > 0.1,
            "...and rel_rms is what actually catches it"
        );

        let after = outputs_for(GateMutation::GateAfterOProj);
        assert!(
            rel_rms(&after, &control) < 0.05,
            "GateAfterOProj is quiet in rel_rms; cosine is what carries it"
        );
        assert!(cosine(&after, &control) < 0.99999);
    }
}
