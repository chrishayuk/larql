//! **Wave 19b — the batch traversal carries the bundle, per position,
//! and the witness proves it against the decode traversal 19a proved.**
//!
//! The batch carrier is `[positions, streams, hidden]`: one bundle per
//! position. Stages one to three run per position, the ordinary
//! operator runs ONCE over `[positions, hidden]`, and stage five runs
//! per position. The witness reads the site's state for every position
//! from the streaming sink and holds it to the same assertions the
//! decode witness holds — A1 against the oracle, A2 structurally, A4 on
//! the hybrid, A5 on width — plus A7: the batch and decode traversals
//! agree to the bit, record by record and logit by logit. The public
//! refusal still stands; every image here is prepared through the seam.
//!
//! # No single-position shortcut
//!
//! The oracle's three positions carry DISTINCT states, so a defect that
//! applies one position's state to every row is visible: the frozen
//! control (e) fails A1 at positions 1 and 2 while passing at 0, and a
//! swap of two positions' bundles before the update fails A2 and A7.

use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::hyper_connection::{head_reduce, Bundle, Mutation};
use crate::format::vindex3::opplan::exec::kv::RowKvState;
use crate::format::vindex3::opplan::exec::observe::HcSite;
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::exec::{
    execute_prepared_streaming, execute_prepared_streaming_mutated, FinalOutput, FinalState,
    LayerTrace, Plane, PlaneEvent, ResumePoint,
};

use super::wave19_hc_decode::{
    a1_at, a1_foreign_stages, a2_expansion, a4_ffn_branch, a5_width, assert_all_hold,
    assert_caught_by, layer_range, prepare, run_from_oracle, run_from_tokens, store, Record, Run,
    Witness, SITES,
};
use super::wave19_hc_substrate::{
    self as substrate, Oracle, Substrate, Variant, HIDDEN, LAYERS, NORM_EPS, POSITIONS, STREAMS,
    VOCAB,
};

/// A batch run's observations and outputs.
struct BatchRun {
    witness: Witness,
    layers: Vec<(usize, LayerTrace)>,
    embedded: Option<Plane>,
    out: FinalOutput,
}

/// Collect a batch traversal's events into the decode witness's record
/// shape, one record per position per site.
fn collect(
    sub: &Substrate,
    ops: &PreparedOperands,
    tokens: &[u32],
    resume: Option<ResumePoint>,
    mutation: Mutation,
) -> BatchRun {
    let backend = ReferenceBackend::new();
    let mut witness = Witness::default();
    let mut layers = Vec::new();
    let mut embedded = None;
    let out = execute_prepared_streaming_mutated(
        &sub.plan,
        ops,
        tokens,
        &backend,
        resume,
        &mut |event| {
            match event {
                PlaneEvent::Embedded(plane) => embedded = Some(plane.clone()),
                PlaneEvent::Layer { index, trace } => layers.push((index, trace)),
                PlaneEvent::HyperConnectionSite(site) => {
                    for position in 0..site.splits.len() {
                        witness.records.push(Record {
                            layer: site.layer,
                            site: site.site,
                            position,
                            split: site.splits[position].clone(),
                            reduced: site.reduced[position].clone(),
                            branch_output: site.branch_outputs[position].clone(),
                            bundle_out: site.bundles_out[position].clone(),
                        });
                    }
                }
                // This wave's witness reads hyper-connection sites; the
                // attention-residual events belong to K3-ATTNRES-1's own.
                PlaneEvent::AttentionResidualSite(_) | PlaneEvent::AttentionResidualBoundary(_) => {
                }
            }
            Ok(())
        },
        mutation,
    )
    .expect("the batch traversal runs");
    BatchRun {
        witness,
        layers,
        embedded,
        out,
    }
}

/// The oracle's three positions entered at layer 0 as bundles — the
/// layer-range contract, in the batch's own form.
fn run_batch_from_oracle(sub: &Substrate, slice: ExecutionSlice, mutation: Mutation) -> BatchRun {
    let ops = prepare(sub, slice).expect("the seam prepares the substrate");
    let oracle = Oracle::load();
    let resume = ResumePoint {
        next_layer: 0,
        hidden: Plane::Bundles((0..POSITIONS).map(|p| oracle.input(p)).collect()),
    };
    // Token ids are not read under a resume point; the count is.
    collect(sub, &ops, &[0; POSITIONS], Some(resume), mutation)
}

/// A7. Record by record, the batch traversal's site state equals the
/// decode traversal's, to the bit.
fn a7_parity(batch: &Witness, decode: &Witness) -> Result<(), String> {
    if batch.records.len() != decode.records.len() {
        return Err(format!(
            "A7: {} batch records against {} decode records",
            batch.records.len(),
            decode.records.len()
        ));
    }
    for record in &batch.records {
        let what = format!(
            "A7 layer {} {:?} position {}",
            record.layer, record.site, record.position
        );
        let other = decode
            .record(record.layer, record.site, record.position)
            .ok_or_else(|| format!("{what}: no decode record"))?;
        let same = record.split == other.split
            && record.reduced == other.reduced
            && record.branch_output == other.branch_output
            && record.bundle_out == other.bundle_out;
        if !same {
            return Err(format!("{what}: the batch and decode traversals disagree"));
        }
    }
    Ok(())
}

/// The batch chain on the headless layer range, with the decode run as
/// A7's reference.
fn witness_batch(variant: Variant, mutation: Mutation) -> Vec<(&'static str, Result<(), String>)> {
    let sub = substrate::build(variant);
    let oracle = Oracle::load();
    let batch = run_batch_from_oracle(&sub, layer_range(), mutation);
    let decode = run_from_oracle(&sub, layer_range(), Mutation::None);
    let ops = prepare(&sub, layer_range()).unwrap();
    vec![
        ("A1", a1_foreign_stages(&batch.witness, &oracle)),
        ("A2", a2_expansion(&batch.witness, &oracle)),
        ("A4", a4_ffn_branch(&batch.witness, &sub, &ops)),
        ("A5", a5_width(&batch.witness)),
        ("A7", a7_parity(&batch.witness, &decode.witness)),
    ]
}

// ── The positive witness ──

#[test]
fn the_batch_traversal_runs_the_bundle_per_position_and_agrees_with_decode() {
    assert_all_hold(&witness_batch(Variant::Headless, Mutation::None));
}

#[test]
fn the_batch_hybrid_ffn_receives_the_reduced_vector() {
    assert_all_hold(&witness_batch(Variant::Hybrid, Mutation::None));
}

/// The three positions are distinguishable: a witness at batch size
/// three is a real multi-position check, not three copies of one.
#[test]
fn the_positions_carry_distinct_state() {
    let sub = substrate::build(Variant::Headless);
    let batch = run_batch_from_oracle(&sub, layer_range(), Mutation::None);
    for layer in 0..LAYERS {
        for site in [HcSite::Attention, HcSite::Ffn] {
            let reduced: Vec<&Vec<f32>> = (0..POSITIONS)
                .map(|p| &batch.witness.record(layer, site, p).unwrap().reduced)
                .collect();
            assert_ne!(reduced[0], reduced[1]);
            assert_ne!(reduced[1], reduced[2]);
            assert_ne!(reduced[0], reduced[2]);
        }
    }
}

/// The layer planes are bundle planes, one bundle per position, and the
/// FFN input stays `[hidden]` rows.
#[test]
fn the_layer_planes_are_bundles_and_the_ffn_input_is_rows() {
    let sub = substrate::build(Variant::Headless);
    let batch = run_batch_from_oracle(&sub, layer_range(), Mutation::None);
    assert_eq!(batch.layers.len(), LAYERS);
    for (index, trace) in &batch.layers {
        for plane in [&trace.post_attention, &trace.post_layer] {
            let bundles = plane.bundles().expect("a bundle plane");
            assert_eq!(bundles.len(), POSITIONS);
            assert!(bundles
                .iter()
                .all(|b| b.streams() == STREAMS && b.hidden() == HIDDEN));
            assert!(plane.try_rows().is_err(), "a bundle plane has no rows");
        }
        assert_eq!(trace.ffn_input.len(), POSITIONS);
        assert!(trace.ffn_input.iter().all(|row| row.len() == HIDDEN));
        // The plane after the FFN site IS the site's bundle_out.
        for (position, after) in trace.post_layer.bundles().unwrap().iter().enumerate() {
            let record = batch.witness.record(*index, HcSite::Ffn, position).unwrap();
            assert_eq!(*after, record.bundle_out);
        }
    }
}

/// Headless, layer range: the run ends on the last position's bundle
/// and produces no logits.
#[test]
fn a_headless_layer_range_ends_on_a_bundle() {
    let sub = substrate::build(Variant::Headless);
    let batch = run_batch_from_oracle(&sub, layer_range(), Mutation::None);
    assert!(batch.out.logits.is_none());
    let last = batch
        .witness
        .record(LAYERS - 1, HcSite::Ffn, POSITIONS - 1)
        .unwrap();
    assert_eq!(batch.out.exit, FinalState::Bundle(last.bundle_out.clone()));
    assert!(batch.out.exit.try_hidden().is_err());
}

/// Head-bearing, whole stack, from the oracle's state: the exit reduces
/// through the head, and the logits are the decode step's logits at the
/// last position, to the bit.
#[test]
fn a_head_bearing_stack_reduces_through_the_head_and_matches_decode() {
    let sub = substrate::build(Variant::HeadBearing);
    let oracle = Oracle::load();
    let batch = run_batch_from_oracle(&sub, ExecutionSlice::Full, Mutation::None);
    let decode = run_from_oracle(&sub, ExecutionSlice::Full, Mutation::None);
    assert_all_hold(&[("A7", a7_parity(&batch.witness, &decode.witness))]);
    let last = &decode.steps[POSITIONS - 1];
    assert_eq!(batch.out.logits, last.logits);
    // The decode witness reports the head's reduction BEFORE the final
    // norm; the batch exit is the final-normed vector the head read.
    // One is the other through the image's own final norm, and the
    // reduction is `head_reduce` over the last bundle with the oracle's
    // head weights.
    let head = oracle.head();
    let bundle = last.bundle.as_ref().unwrap();
    let reduced = head_reduce(
        bundle.as_flat(),
        STREAMS,
        HIDDEN,
        &head.weights(),
        NORM_EPS,
        oracle.topology().sinkhorn_eps,
    );
    assert_eq!(last.exit.as_deref().unwrap(), reduced.as_slice());
    let ops = prepare(&sub, ExecutionSlice::Full).unwrap();
    let final_norm = ops
        .final_norm()
        .expect("a whole-stack image carries the final norm");
    let expected = final_norm.apply(&ReferenceBackend::new(), &reduced);
    assert_eq!(batch.out.exit.hidden(), expected.as_slice());
    assert_eq!(batch.out.logits.as_ref().unwrap().len(), VOCAB);
}

/// From tokens: the embedding enters replicated, and the whole batch
/// traversal agrees with the decode steps, record by record and at the
/// logits.
#[test]
fn from_tokens_the_batch_and_decode_traversals_agree_to_the_bit() {
    let sub = substrate::build(Variant::HeadBearing);
    let tokens = [2u32, 5, 1];
    let ops = prepare(&sub, ExecutionSlice::Full).unwrap();
    let batch = collect(&sub, &ops, &tokens, None, Mutation::None);
    let decode = run_from_tokens(&sub, &tokens, Mutation::None);
    assert_all_hold(&[("A7", a7_parity(&batch.witness, &decode.witness))]);
    assert_eq!(batch.out.logits, decode.steps[tokens.len() - 1].logits);
    let embedded = batch.embedded.expect("a fresh run emits plane 000");
    let bundles = embedded.bundles().expect("the embedding enters as bundles");
    for (position, &token) in tokens.iter().enumerate() {
        let row = substrate::llama_embedding_row(token);
        for stream in 0..STREAMS {
            assert_eq!(bundles[position].stream(stream), row.as_slice());
        }
    }
}

/// `step_many` is the batch traversal in decode clothing: on a
/// hyper-connected component it now runs, and is indistinguishable from
/// stepping once per token.
#[test]
fn step_many_matches_stepping_on_a_hyper_connected_component() {
    let sub = substrate::build(Variant::HeadBearing);
    let ops = prepare(&sub, ExecutionSlice::Full).unwrap();
    let backend = ReferenceBackend::new();
    let tokens = [4u32, 1, 6];
    let mut kv = RowKvState::default();
    let mut many = DecodeSession::over_prepared(&sub.plan, &ops, &backend, &mut kv).unwrap();
    let batched = many.step_many(&tokens).unwrap().logits.unwrap();
    let mut kv = RowKvState::default();
    let mut one = DecodeSession::over_prepared(&sub.plan, &ops, &backend, &mut kv).unwrap();
    let mut last = None;
    for &token in &tokens {
        last = one.step(token).unwrap().logits;
    }
    assert_eq!(batched, last.unwrap());
}

// ── The controls ──

#[test]
fn batch_mutant_a_bypassing_the_composition_is_caught_by_a1_and_a5() {
    assert_caught_by(
        &witness_batch(Variant::Headless, Mutation::BypassComposition),
        &["A1", "A5"],
    );
}

#[test]
fn batch_mutant_b_one_sinkhorn_iteration_is_caught_by_a1() {
    let results = witness_batch(Variant::Headless, Mutation::SingleIteration);
    assert_caught_by(&results, &["A1", "A7"]);
    assert_all_hold(&results[1..2]);
}

#[test]
fn batch_mutant_b_a_transposed_combination_is_caught_by_a2_and_not_a1() {
    let results = witness_batch(Variant::Headless, Mutation::TransposedCombination);
    assert_caught_by(&results, &["A2", "A7"]);
    assert_all_hold(&results[..1]);
}

/// The batch path has no attention-input tap; the pre-norm control is
/// caught by parity with the decode traversal, which A3 already pinned.
#[test]
fn batch_mutant_c_pre_norm_on_a_stream_is_caught_by_a7_alone() {
    let results = witness_batch(Variant::Headless, Mutation::PreNormOnStreamZero);
    assert_caught_by(&results, &["A7"]);
    assert_all_hold(&results[..2]);
}

#[test]
fn batch_mutant_d_hybrid_residual_from_a_stream_is_caught_by_a4() {
    let results = witness_batch(Variant::Hybrid, Mutation::HybridResidualFromStreamZero);
    assert_caught_by(&results, &["A4", "A7"]);
    assert_all_hold(&results[..1]);
}

/// (e) One position's state applied to every row: invisible at position
/// 0, caught at positions 1 and 2 — which is why the witness runs three.
#[test]
fn batch_mutant_e_one_positions_split_on_every_row_is_caught_after_position_zero() {
    let sub = substrate::build(Variant::Headless);
    let oracle = Oracle::load();
    let batch = run_batch_from_oracle(&sub, layer_range(), Mutation::SplitFromPositionZero);
    let decode = run_from_oracle(&sub, layer_range(), Mutation::None);
    assert!(
        a1_at(&batch.witness, &oracle, 0).is_ok(),
        "position 0 is its own state"
    );
    assert!(a1_at(&batch.witness, &oracle, 1).is_err());
    assert!(a1_at(&batch.witness, &oracle, 2).is_err());
    assert!(a7_parity(&batch.witness, &decode.witness).is_err());
}

/// Swapping two positions' bundles between the reduction and the update
/// leaves every split correct and every expansion wrong: A2 and A7 see
/// it, A1 does not. A witness blind to per-position state would pass.
#[test]
fn batch_swapping_positions_before_the_update_is_caught_by_a2_and_a7_and_not_a1() {
    let results = witness_batch(Variant::Headless, Mutation::SwapPositionsBeforeUpdate);
    assert_caught_by(&results, &["A2", "A7"]);
    assert_all_hold(&results[..1]);
}

// ── The resume contract ──

#[test]
fn a_row_plane_cannot_resume_a_hyper_connected_component() {
    let sub = substrate::build(Variant::Headless);
    let ops = prepare(&sub, layer_range()).unwrap();
    let backend = ReferenceBackend::new();
    let rows = Plane::Rows(vec![vec![0.0; HIDDEN]; POSITIONS]);
    let err = execute_prepared_streaming(
        &sub.plan,
        &ops,
        &[0; POSITIONS],
        &backend,
        Some(ResumePoint {
            next_layer: 0,
            hidden: rows,
        }),
        &mut |_| Ok(()),
    )
    .err()
    .map(|e| e.to_string())
    .expect("rows have no reading on a bundle carrier");
    assert!(err.contains("carries bundles, not rows"), "{err}");
}

#[test]
fn a_bundle_plane_cannot_resume_a_single_stream_component() {
    let sub = substrate::single_stream_sibling();
    let store = store(&sub);
    let backend = ReferenceBackend::new();
    let ops = PreparedOperands::load(&sub.plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let bundles = Plane::Bundles(vec![Bundle::replicate(&[0.0; HIDDEN], STREAMS); POSITIONS]);
    let err = execute_prepared_streaming(
        &sub.plan,
        &ops,
        &[0; POSITIONS],
        &backend,
        Some(ResumePoint {
            next_layer: 0,
            hidden: bundles,
        }),
        &mut |_| Ok(()),
    )
    .err()
    .map(|e| e.to_string())
    .expect("bundles have no reading on one stream");
    assert!(err.contains("carries rows, not bundles"), "{err}");
}

#[test]
fn resume_bundles_of_the_wrong_shape_are_refused() {
    let sub = substrate::build(Variant::Headless);
    let ops = prepare(&sub, layer_range()).unwrap();
    let backend = ReferenceBackend::new();
    let narrow = Plane::Bundles(vec![
        Bundle::replicate(&[0.0; HIDDEN - 1], STREAMS);
        POSITIONS
    ]);
    let err = execute_prepared_streaming(
        &sub.plan,
        &ops,
        &[0; POSITIONS],
        &backend,
        Some(ResumePoint {
            next_layer: 0,
            hidden: narrow,
        }),
        &mut |_| Ok(()),
    )
    .err()
    .map(|e| e.to_string())
    .expect("the bundle must match the component");
    assert!(err.contains("resume bundles do not match"), "{err}");
}

/// The decode witness's run shape is reused here; this pins that a
/// decode `Run` really carries one step per position.
#[test]
fn the_decode_reference_carries_one_step_per_position() {
    let sub = substrate::build(Variant::Headless);
    let decode: Run = run_from_oracle(&sub, layer_range(), Mutation::None);
    assert_eq!(decode.steps.len(), POSITIONS);
    assert_eq!(decode.witness.records.len(), LAYERS * SITES * POSITIONS);
}
