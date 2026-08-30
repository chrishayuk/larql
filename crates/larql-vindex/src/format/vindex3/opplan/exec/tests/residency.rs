//! Operand residency: prepared once, read by both traversals.
//!
//! The claim these gates defend is architectural, not numerical — "a
//! served model's operands are lowered once, and every request reads
//! that image". A claim like that regresses quietly if it is only ever
//! checked with a stopwatch, so the gates below read
//! [`OperandStore::load_count`] and assert the *shape* directly.
//!
//! The corresponding numbers, measured on a real 3 B container: a warm
//! 5-token / 1-token request cost 7.44 s before this, of which 3.83 s
//! was batch prefill loading the model and 3.27 s was the decode
//! session loading it again — against 0.13 s of decode arithmetic.

use crate::format::vindex3::fixtures::{encode_fixture_container, miniature_glimmer};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::kv::{KvState, RowKvState};
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::{
    execute_plan_streaming, execute_prepared_streaming, prefill_plan, prefill_prepared,
};
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};

/// A few vocabulary-covered ids; the fixture's vocab is tiny.
const TOKENS: [u32; 4] = [1, 2, 3, 4];

fn fixture() -> (tempfile::TempDir, ComponentOpPlan, OperandStore) {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "residency-fixture",
    );
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    (container, plan, store)
}

/// One request as the serve path runs it: batch-prefill the prompt into
/// the caller's KV, then open a decode session over the same KV and
/// step once. Returns the decoded logits.
fn one_request(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    backend: &ProductionBackend,
    prompt: &[u32],
) -> Vec<f32> {
    let mut kv = RowKvState::default();
    let out = prefill_prepared(plan, ops, prompt, backend, &mut kv).unwrap();
    assert!(out.logits.is_some(), "the fixture carries a head");
    let mut session = DecodeSession::over_prepared(plan, ops, backend, &mut kv).unwrap();
    session
        .step(prompt[prompt.len() - 1])
        .unwrap()
        .logits
        .unwrap()
}

/// **Gate 1 — preparation once.** Serving N requests over a prepared
/// image must read the operand store exactly zero further times. This
/// is the whole rung, expressed without a clock.
#[test]
fn serving_requests_over_a_prepared_image_never_touches_the_store() {
    let (_container, plan, store) = fixture();
    let backend = ProductionBackend::new();

    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let after_prepare = store.load_count();
    assert!(
        after_prepare > 0,
        "preparation must actually load the operands"
    );

    for _ in 0..5 {
        one_request(&plan, &ops, &backend, &TOKENS);
    }
    assert_eq!(
        store.load_count(),
        after_prepare,
        "requests must read the prepared image, never the store"
    );
}

/// The counterfactual that makes Gate 1 mean something: the unprepared
/// entry point *does* load, every single call. Without this the gate
/// above would also pass against a store that had been emptied.
#[test]
fn the_unprepared_entry_point_loads_the_model_on_every_call() {
    let (_container, plan, store) = fixture();
    let backend = ProductionBackend::new();

    let mut kv = RowKvState::default();
    prefill_plan(&plan, &store, &TOKENS, &backend, &mut kv).unwrap();
    let after_first = store.load_count();
    assert!(after_first > 0);

    let mut kv = RowKvState::default();
    prefill_plan(&plan, &store, &TOKENS, &backend, &mut kv).unwrap();
    assert!(
        store.load_count() > after_first,
        "the store-taking form prepares per call — that is the cost the serve path used to pay"
    );
}

/// **Gate 2 — request parity.** The resident path and the load-per-call
/// path must produce identical logits and identical KV contents. Same
/// program, same arithmetic; residency is a lifetime change, never a
/// numerical one.
#[test]
fn prepared_and_unprepared_paths_agree_bit_for_bit() {
    let (_container, plan, store) = fixture();
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();

    let mut kv_cold = RowKvState::default();
    let cold = prefill_plan(&plan, &store, &TOKENS, &backend, &mut kv_cold).unwrap();

    let mut kv_warm = RowKvState::default();
    let warm = prefill_prepared(&plan, &ops, &TOKENS, &backend, &mut kv_warm).unwrap();

    assert_eq!(cold.logits, warm.logits, "prefill logits must be identical");
    assert_eq!(
        cold.final_hidden, warm.final_hidden,
        "final hidden state must be identical"
    );
    assert_eq!(kv_cold.position(), kv_warm.position());
    for layer in 0..plan.layers.len() {
        assert_eq!(
            kv_cold.keys(layer),
            kv_warm.keys(layer),
            "layer {layer} keys must be identical"
        );
        assert_eq!(
            kv_cold.values(layer),
            kv_warm.values(layer),
            "layer {layer} values must be identical"
        );
    }
}

/// **Gate 3 — continuation parity.** Resuming a prepared session from a
/// populated provider must land on the same logits as a session that
/// prefilled the whole prompt itself. This is the property N1's
/// bit-identical resumption rests on, now over shared operands.
///
/// Since V3-SERVE-2 this gate straddles both attention realisations,
/// which is what makes it load-bearing: the whole-prompt arm starts at
/// position 0 and takes the **batched** branch, while the split arm's
/// second chunk extends a populated provider and takes the
/// **per-position** branch. They must still agree bit-for-bit — a
/// batched pass that populated K/V differently from the stepped one
/// would show up here as a resumed continuation that diverges.
#[test]
fn resuming_over_a_prepared_image_matches_a_full_prefill() {
    let (_container, plan, store) = fixture();
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();

    // Whole prompt in one go.
    let mut kv_full = RowKvState::default();
    prefill_prepared(&plan, &ops, &TOKENS, &backend, &mut kv_full).unwrap();
    let full = DecodeSession::over_prepared(&plan, &ops, &backend, &mut kv_full)
        .unwrap()
        .step(TOKENS[3])
        .unwrap()
        .logits
        .unwrap();

    // Same prompt, prefilled in two chunks against one provider.
    let mut kv_split = RowKvState::default();
    prefill_prepared(&plan, &ops, &TOKENS[..2], &backend, &mut kv_split).unwrap();
    prefill_prepared(&plan, &ops, &TOKENS[2..], &backend, &mut kv_split).unwrap();
    let split = DecodeSession::over_prepared(&plan, &ops, &backend, &mut kv_split)
        .unwrap()
        .step(TOKENS[3])
        .unwrap()
        .logits
        .unwrap();

    assert_eq!(full, split, "a resumed continuation must be bit-identical");
}

/// **Gate 4 — session isolation.** Two sessions over one prepared image
/// hold independent continuation state: sharing immutable operands must
/// not make them share anything mutable. Run interleaved so a shared
/// buffer would show up as cross-talk.
#[test]
fn concurrent_sessions_share_operands_but_not_continuation_state() {
    let (_container, plan, store) = fixture();
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();

    // Reference: each prompt run entirely on its own.
    let alone_a = one_request(&plan, &ops, &backend, &TOKENS[..2]);
    let alone_b = one_request(&plan, &ops, &backend, &TOKENS);

    // Interleaved: two live providers, stepped alternately.
    let mut kv_a = RowKvState::default();
    let mut kv_b = RowKvState::default();
    prefill_prepared(&plan, &ops, &TOKENS[..2], &backend, &mut kv_a).unwrap();
    prefill_prepared(&plan, &ops, &TOKENS, &backend, &mut kv_b).unwrap();
    let inter_a = DecodeSession::over_prepared(&plan, &ops, &backend, &mut kv_a)
        .unwrap()
        .step(TOKENS[1])
        .unwrap()
        .logits
        .unwrap();
    let inter_b = DecodeSession::over_prepared(&plan, &ops, &backend, &mut kv_b)
        .unwrap()
        .step(TOKENS[3])
        .unwrap()
        .logits
        .unwrap();

    assert_eq!(alone_a, inter_a, "session A disturbed by a concurrent B");
    assert_eq!(alone_b, inter_b, "session B disturbed by a concurrent A");
    assert_ne!(
        alone_a, alone_b,
        "the two prompts must differ, or the test proves nothing"
    );
}

/// A slice prepares strictly less than the whole model — the seam the
/// decoupled surfaces (layer-range shards, attention-only nodes, expert
/// servers) grow from.
#[test]
fn a_layer_range_slice_prepares_only_its_own_layers() {
    let (_container, plan, store) = fixture();
    let backend = ProductionBackend::new();
    assert!(
        plan.layers.len() >= 2,
        "the fixture needs at least two layers to slice"
    );

    let before_full = store.load_count();
    let full = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let full_loads = store.load_count() - before_full;

    let before_slice = store.load_count();
    let sliced = PreparedOperands::load(
        &plan,
        &store,
        &backend,
        ExecutionSlice::LayerRange { start: 0, end: 1 },
    )
    .unwrap();
    let slice_loads = store.load_count() - before_slice;

    assert_eq!(full.layer_count(), plan.layers.len());
    assert_eq!(sliced.layer_count(), 1);
    assert!(
        slice_loads < full_loads,
        "a one-layer slice must load less than the whole model ({slice_loads} vs {full_loads})"
    );
    // A slice that does not carry the stack's ends carries neither the
    // embedding table nor the head — it consumes hidden states.
    assert!(full.has_output());
    assert!(!sliced.has_output());
    assert!(full.embed_table().is_some());
    assert!(sliced.embed_table().is_none());
}

/// A slice the plan cannot satisfy is refused, not silently truncated.
/// Serving "as much as exists" is how a shard ends up quietly executing
/// the wrong submodel.
#[test]
fn an_out_of_range_slice_is_refused() {
    let (_container, plan, store) = fixture();
    let backend = ProductionBackend::new();
    let past_end = plan.layers.len() + 1;

    let refuse = |slice, why: &str| match PreparedOperands::load(&plan, &store, &backend, slice) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("{why}"),
    };

    let err = refuse(
        ExecutionSlice::LayerRange {
            start: 0,
            end: past_end,
        },
        "a slice past the stack must be refused",
    );
    assert!(err.contains("outside component"), "{err}");

    let err = refuse(
        ExecutionSlice::LayerRange { start: 1, end: 1 },
        "an empty slice must be refused",
    );
    assert!(err.contains("empty"), "{err}");
}

/// The batch traversal must agree between its store-taking and prepared
/// forms too — prefill is not the only caller, and a `dump-layers` run
/// through the prepared image has to see the same planes.
#[test]
fn batch_traversal_agrees_between_prepared_and_unprepared_forms() {
    let (_container, plan, store) = fixture();
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();

    let mut cold_layers = Vec::new();
    let cold = execute_plan_streaming(&plan, &store, &TOKENS, &backend, None, &mut |event| {
        if let crate::format::vindex3::opplan::exec::PlaneEvent::Layer { index, trace } = event {
            cold_layers.push((index, trace.post_layer.clone()));
        }
        Ok(())
    })
    .unwrap();

    let mut warm_layers = Vec::new();
    let warm = execute_prepared_streaming(&plan, &ops, &TOKENS, &backend, None, &mut |event| {
        if let crate::format::vindex3::opplan::exec::PlaneEvent::Layer { index, trace } = event {
            warm_layers.push((index, trace.post_layer.clone()));
        }
        Ok(())
    })
    .unwrap();

    assert_eq!(cold.logits, warm.logits);
    assert_eq!(cold.final_hidden, warm.final_hidden);
    assert_eq!(cold_layers, warm_layers, "every layer plane must match");
}

/// Feeding token ids to an image that was prepared without the stack's
/// input end is refused, not guessed at. A layer-range slice consumes
/// hidden states; there is no embedding table to look an id up in.
#[test]
fn executing_token_ids_against_a_sliced_image_is_refused() {
    let (_container, plan, store) = fixture();
    let backend = ProductionBackend::new();
    let sliced = PreparedOperands::load(
        &plan,
        &store,
        &backend,
        ExecutionSlice::LayerRange { start: 0, end: 1 },
    )
    .unwrap();

    let mut kv = RowKvState::default();
    let msg = match prefill_prepared(&plan, &sliced, &TOKENS, &backend, &mut kv) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("a sliced image must refuse token ids"),
    };
    assert!(msg.contains("no embedding table"), "{msg}");

    // The decode session refuses the same thing for the same reason.
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(&plan, &sliced, &backend, &mut kv).unwrap();
    let msg = match session.step(TOKENS[0]) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("a sliced session must refuse token ids"),
    };
    assert!(msg.contains("no embedding table"), "{msg}");
}

/// Residency and the overlay seam compose, in that order: a prepared
/// image is the **effective** operands for the source it was prepared
/// from.
///
/// This is the property the two seams have to agree on. Preparing from
/// an overlaid source must bake the overlay into the resident image —
/// otherwise residency would silently serve base weights to a session
/// that asked for edited ones — and preparing from the bare store must
/// stay bit-identical to before.
#[test]
fn preparing_through_an_overlaid_source_bakes_the_overlay_in() {
    use crate::format::vindex3::opplan::exec::operands::{
        OperandEdit, OperandOverrides, OperandSource,
    };
    use crate::format::vindex3::opplan::LayerFfn;

    let (_container, plan, store) = fixture();
    let backend = ProductionBackend::new();

    let gate = match &plan.layers[0].ffn {
        Some(LayerFfn::Dense(op)) => op.gate.clone().expect("the miniature FFN is gated"),
        other => panic!("layer 0 should be dense, got {other:?}"),
    };
    let cols = gate.shape[1];
    let mut overrides = OperandOverrides::new();
    overrides.push(
        &gate,
        OperandEdit::Row {
            index: 0,
            values: vec![3.0; cols],
        },
    );

    let base_image = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let edited_image = PreparedOperands::load(
        &plan,
        OperandSource::overlaid(&store, &overrides),
        &backend,
        ExecutionSlice::Full,
    )
    .unwrap();

    let base = one_request(&plan, &base_image, &backend, &TOKENS);
    let edited = one_request(&plan, &edited_image, &backend, &TOKENS);
    assert_ne!(
        base, edited,
        "an overlay edit must reach execution through the prepared image"
    );

    // And an empty overlay prepares to the same image as the bare store.
    let empty = OperandOverrides::new();
    let neutral = PreparedOperands::load(
        &plan,
        OperandSource::overlaid(&store, &empty),
        &backend,
        ExecutionSlice::Full,
    )
    .unwrap();
    assert_eq!(
        base,
        one_request(&plan, &neutral, &backend, &TOKENS),
        "an empty overlay must prepare bit-identically to the bare store"
    );
}

/// **The staleness invariant.** A prepared image is a *compiled
/// derivative* of an effective operand source. Once compose mutation
/// can change that source under a long-lived image, "still valid?" has
/// to be answerable — otherwise residency quietly becomes a second
/// authority for what the model means, executing the pre-edit model
/// forever.
///
/// The dangerous shape this pins:
///
/// ```text
/// prepare(base + overlay A) -> image A
/// overlay A mutates to B
/// request reuses image A        <- must be refusable
/// ```
#[test]
fn a_prepared_image_cannot_masquerade_as_current_after_the_overlay_moves() {
    use crate::format::vindex3::opplan::exec::operands::{
        OperandEdit, OperandOverrides, OperandSource,
    };
    use crate::format::vindex3::opplan::LayerFfn;

    let (_container, plan, store) = fixture();
    let backend = ProductionBackend::new();
    let gate = match &plan.layers[0].ffn {
        Some(LayerFfn::Dense(op)) => op.gate.clone().expect("the miniature FFN is gated"),
        other => panic!("layer 0 should be dense, got {other:?}"),
    };
    let cols = gate.shape[1];

    let mut overrides = OperandOverrides::new();
    overrides.push(
        &gate,
        OperandEdit::Row {
            index: 0,
            values: vec![3.0; cols],
        },
    );
    let image = PreparedOperands::load(
        &plan,
        OperandSource::overlaid(&store, &overrides),
        &backend,
        ExecutionSlice::Full,
    )
    .unwrap();
    assert!(
        image.is_current_for(&OperandSource::overlaid(&store, &overrides)),
        "an untouched overlay must leave its image current"
    );

    // The overlay moves under the image.
    overrides.push(
        &gate,
        OperandEdit::Row {
            index: 1,
            values: vec![5.0; cols],
        },
    );
    let moved = OperandSource::overlaid(&store, &overrides);
    assert!(
        !image.is_current_for(&moved),
        "an image compiled before the edit must not pass as current"
    );
    let err = image
        .ensure_current_for(&moved)
        .expect_err("and the refusal must be available as an error");
    assert!(
        err.to_string().contains("different effective operand"),
        "{err}"
    );

    // And it is genuinely stale, not merely flagged: re-preparing from
    // the moved source computes something else.
    let fresh = PreparedOperands::load(&plan, moved, &backend, ExecutionSlice::Full).unwrap();
    assert_ne!(
        one_request(&plan, &image, &backend, &TOKENS),
        one_request(&plan, &fresh, &backend, &TOKENS),
        "the stale image really does execute the pre-edit model"
    );
}

/// Conservative in the right direction: a stale image may be judged
/// stale twice over, but is never judged valid. Two things follow from
/// that, and one thing deliberately does not.
#[test]
fn staleness_errs_toward_re_preparation_never_toward_reuse() {
    use crate::format::vindex3::opplan::exec::operands::{
        OperandEdit, OperandOverrides, OperandSource,
    };
    use crate::format::vindex3::opplan::LayerFfn;

    let (_container, plan, store) = fixture();
    let backend = ProductionBackend::new();
    let gate = match &plan.layers[0].ffn {
        Some(LayerFfn::Dense(op)) => op.gate.clone().expect("the miniature FFN is gated"),
        other => panic!("layer 0 should be dense, got {other:?}"),
    };
    let cols = gate.shape[1];
    let edit = |index| OperandEdit::Row {
        index,
        values: vec![3.0; cols],
    };

    let mut overrides = OperandOverrides::new();
    overrides.push(&gate, edit(0));
    let image = PreparedOperands::load(
        &plan,
        OperandSource::overlaid(&store, &overrides),
        &backend,
        ExecutionSlice::Full,
    )
    .unwrap();

    // A clone holds the same edits *now*, but the two sets diverge
    // independently from here — so it takes a fresh identity and the
    // image does not claim to describe it.
    let twin = overrides.clone();
    assert!(
        !image.is_current_for(&OperandSource::overlaid(&store, &twin)),
        "a clone diverges independently, so it gets its own identity"
    );

    // Reverting is still a mutation: content may return to where it
    // was, the generation does not. Costs one re-preparation; buys the
    // guarantee that the check never has to reason about content.
    overrides.push(&gate, edit(1));
    assert!(!image.is_current_for(&OperandSource::overlaid(&store, &overrides)));

    // What is deliberately NOT a difference: an EMPTY overlay is the
    // bare store — `OperandSource::overlaid` drops it — so an image
    // prepared from one is genuinely current for the other. The stamp
    // tracks effective sources, not the syntax used to build them.
    let bare = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let empty = OperandOverrides::new();
    assert!(bare.is_current_for(&OperandSource::from(&store)));
    assert!(
        bare.is_current_for(&OperandSource::overlaid(&store, &empty)),
        "an empty overlay is the bare store, so the image describes both"
    );
}
