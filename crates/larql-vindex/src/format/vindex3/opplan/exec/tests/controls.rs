//! Stage C: causal authority of the persisted IR (V3-G5b-2c).
//!
//! Three controls, three distinct semantic pathways, one rule: each
//! mutation touches **only the container's persisted graph** — never
//! oracle code, never executor code. Together they prove the IR's facts
//! are load-bearing:
//!
//! ```text
//! C1  query_scale 3.87 → 3.5      scalar semantics are authoritative
//! C2  layer 1 None → Rope         per-layer topology is authoritative
//! C3  remove the gate judgment    operation semantics are authoritative
//!                                 (fail-closed: refusal, not drift)
//! ```
//!
//! A hidden default is precisely a fact whose mutation changes nothing;
//! these tests are the search for hidden defaults.

use super::golden::{executor_trace_from, golden_forward, max_abs, miniature_glimmer, G_LAYERS};
use crate::format::vindex3::encode::{encode_system, SYSTEM_GRAPH_JSON};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::{plan_component_ops, ClosureDefect};

/// Divergence threshold: well above fp noise (~5e-8 measured), well
/// below any semantic effect.
const NOISE_CEILING: f32 = 1e-5;

/// Encode the miniature fixture and hand back the container dir.
fn encoded_container(dir: &std::path::Path) -> tempfile::TempDir {
    let inventory = larql_models::inventory::build_inventory(dir).unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_system(&[("mini-glimmer".to_string(), inventory)], container.path()).unwrap();
    container
}

/// Edit one component's persisted graph JSON in place.
fn mutate_graph(container: &std::path::Path, mutate: impl FnOnce(&mut serde_json::Value)) {
    let path = container.join(SYSTEM_GRAPH_JSON);
    let mut graph: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let target = graph["components"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|c| c["id"] == "target")
        .unwrap();
    mutate(target);
    std::fs::write(&path, graph.to_string()).unwrap();
}

/// C1 — scalar authority: a mutated persisted `query_scale` must change
/// computation from the first layer onward. The executor and oracle are
/// untouched; only the IR moved.
#[test]
fn c1_query_scale_mutation_changes_computation() {
    let dir = tempfile::tempdir().unwrap();
    miniature_glimmer(dir.path());
    let golden = golden_forward(dir.path());
    let container = encoded_container(dir.path());
    mutate_graph(container.path(), |target| {
        target["execution"]["attention"]["query_scale"] = serde_json::json!(3.5);
    });
    let executed = executor_trace_from(container.path());

    let layer0 = max_abs(
        executed.layers[0].post_attention.rows(),
        &golden.layers[0].post_attention,
    );
    assert!(
        layer0 > NOISE_CEILING,
        "query_scale must be causally load-bearing from layer 0 (diff {layer0:e})"
    );
}

/// C2 — layer-policy authority: flipping layer 1 from NoPE to RoPE must
/// leave layer 0 *identical* to golden and diverge exactly at layer 1 —
/// the location of first divergence is predictable from the mutation.
#[test]
fn c2_position_policy_mutation_diverges_exactly_at_its_layer() {
    let dir = tempfile::tempdir().unwrap();
    miniature_glimmer(dir.path());
    let golden = golden_forward(dir.path());
    let container = encoded_container(dir.path());
    mutate_graph(container.path(), |target| {
        target["attention"][1]["position"] =
            serde_json::json!({ "kind": "rope", "theta": 500000.0 });
    });
    let executed = executor_trace_from(container.path());

    let layer0 = max_abs(
        executed.layers[0].post_layer.rows(),
        &golden.layers[0].post_layer,
    );
    assert!(
        layer0 < NOISE_CEILING,
        "layer 0 precedes the mutation and must match golden (diff {layer0:e})"
    );
    let layer1 = max_abs(
        executed.layers[1].post_attention.rows(),
        &golden.layers[1].post_attention,
    );
    assert!(
        layer1 > NOISE_CEILING,
        "layer 1 carries the mutation and must diverge (diff {layer1:e})"
    );
    assert_eq!(G_LAYERS, 2);
}

/// C3 — operation-semantic authority: removing the judged gate semantics
/// from the persisted surface must REFUSE at operand closure (the gate
/// operand still exists), naming the primitive — fail-closed all the way
/// into execution, never a silently ungated forward.
#[test]
fn c3_removing_gate_judgment_refuses_execution() {
    let dir = tempfile::tempdir().unwrap();
    miniature_glimmer(dir.path());
    let container = encoded_container(dir.path());
    mutate_graph(container.path(), |target| {
        target["execution"]["attention"]
            .as_object_mut()
            .unwrap()
            .remove("output_gate");
    });

    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    assert!(
        outcome.plan.is_none(),
        "an unjudged gate operand must not plan"
    );
    let named = outcome.defects.iter().any(|d| {
        matches!(
            d,
            ClosureDefect::OperandImpliesAbsentOp { required_primitive, .. }
                if required_primitive.contains("attention output gate")
        )
    });
    assert!(
        named,
        "the refusal must name the primitive: {:?}",
        outcome.defects
    );
}

/// C4: the sliding window is load-bearing, and binds exactly where the
/// geometry says it does.
///
/// `golden ↔ reference` agreement on a masked layer proves nothing
/// unless the mask actually excludes something — two implementations
/// that both attend to everything agree trivially. With `G_WINDOW` 3
/// over 5 positions the exclusion starts at position 3 (which sees keys
/// 1..=3 rather than 0..=3), so widening the span to `full` must leave
/// positions 0..=2 untouched and move positions 3..=4.
///
/// Position-resolved rather than layer-resolved on purpose: a control
/// that only said "layer 0 moved" would also pass if the window were
/// wrong in a way that changed every position.
#[test]
fn c4_sliding_window_binds_exactly_from_the_first_excluded_position() {
    // The first position whose span the window truncates:
    // `start = (position + 1) - G_WINDOW > 0` first holds at G_WINDOW.
    const FIRST_MASKED_POSITION: usize = super::golden::G_WINDOW;

    let dir = tempfile::tempdir().unwrap();
    miniature_glimmer(dir.path());
    let golden = golden_forward(dir.path());
    let container = encoded_container(dir.path());
    mutate_graph(container.path(), |target| {
        // Widen layer 0's span; every other judged fact is untouched.
        target["attention"][0]["span"] = serde_json::json!("full");
        target["attention"][0]["window"] = serde_json::Value::Null;
    });
    let executed = executor_trace_from(container.path());

    let masked = &golden.layers[0].post_attention;
    let widened = executed.layers[0].post_attention.rows();
    assert!(
        masked.len() > FIRST_MASKED_POSITION,
        "fixture must run past the first masked position to test the window"
    );
    for position in 0..FIRST_MASKED_POSITION {
        let diff = max_abs(&widened[position..=position], &masked[position..=position]);
        assert!(
            diff < NOISE_CEILING,
            "position {position} precedes the window's first exclusion and must not move \
             (diff {diff:e})"
        );
    }
    for position in FIRST_MASKED_POSITION..masked.len() {
        let diff = max_abs(&widened[position..=position], &masked[position..=position]);
        assert!(
            diff > NOISE_CEILING,
            "position {position} is masked under the declared window, so widening the span \
             must change it — the window is not load-bearing (diff {diff:e})"
        );
    }
}

/// C5 — the judged embedding normalisation is load-bearing.
///
/// This is the control the upstream oracle earned. The op is weightless,
/// so no operand evidences it and no closure check can infer it: the only
/// thing standing between "the container states it" and "the executor
/// silently omits it" is that removing the statement changes the numbers.
///
/// Divergence is asserted from the *embedding* onward, not merely from
/// layer 0 — a missing normalisation is a pure per-row rescale, and the
/// real upstream diff showed cosine similarity of exactly 1.0 at that
/// plane while `max_abs` was 42. A control that only compared directions
/// would have scored this bug as agreement.
#[test]
fn c5_removing_the_embedding_norm_changes_computation() {
    let dir = tempfile::tempdir().unwrap();
    miniature_glimmer(dir.path());
    let golden = golden_forward(dir.path());
    let container = encoded_container(dir.path());

    let intact = executor_trace_from(container.path());
    let layer0 = max_abs(
        intact.layers[0].post_attention.rows(),
        &golden.layers[0].post_attention,
    );
    assert!(
        layer0 < NOISE_CEILING,
        "the intact container must match golden before the mutation means anything ({layer0:e})"
    );

    mutate_graph(container.path(), |target| {
        target["execution"]["head"]["embedding_norm"] = serde_json::Value::Null;
    });
    let stripped = executor_trace_from(container.path());
    let diverged = max_abs(
        stripped.layers[0].post_attention.rows(),
        &golden.layers[0].post_attention,
    );
    assert!(
        diverged > NOISE_CEILING,
        "removing the judged embedding norm changed nothing — it is not load-bearing \
         ({diverged:e})"
    );
}

/// C6 — the layer norms' affine convention is authoritative.
///
/// Glimmer's layer norms are centred (`normed * (1 + w)`); flattening the
/// offset to 0 changes what every layer computes. The embedding is
/// asserted *unmoved* so the divergence is attributable to the norm
/// rather than to something earlier in the program.
#[test]
fn c6_layer_norm_weight_offset_is_authoritative() {
    let dir = tempfile::tempdir().unwrap();
    miniature_glimmer(dir.path());
    let golden = golden_forward(dir.path());
    let container = encoded_container(dir.path());

    let intact = executor_trace_from(container.path());
    let before = max_abs(
        intact.layers[0].post_attention.rows(),
        &golden.layers[0].post_attention,
    );
    assert!(
        before < NOISE_CEILING,
        "intact must match golden ({before:e})"
    );

    mutate_graph(container.path(), |target| {
        target["execution"]["norm"]["pre"]["weight_offset"] = serde_json::json!(0.0);
    });
    let flattened = executor_trace_from(container.path());

    // The embedding precedes every layer norm and must not move.
    let embed = max_abs(flattened.embedded.rows(), intact.embedded.rows());
    assert!(
        embed < NOISE_CEILING,
        "the embedding precedes the mutated norm and must not move ({embed:e})"
    );
    let after = max_abs(
        flattened.layers[0].post_attention.rows(),
        &golden.layers[0].post_attention,
    );
    assert!(
        after > NOISE_CEILING,
        "flattening the centred-norm offset changed nothing — it is not load-bearing \
         ({after:e})"
    );
}

/// C7 — norm facts are **site-local**, not model-scope.
///
/// The final norm's offset is a different fact from the layer norms'.
/// Mutating it must move the final norm and the logits while leaving
/// *every* layer trace untouched. This is the control that would have
/// caught the bug the real table found: a single model-scope offset
/// cannot satisfy both sites, and only a per-site check notices.
#[test]
fn c7_final_norm_offset_is_site_local() {
    let dir = tempfile::tempdir().unwrap();
    miniature_glimmer(dir.path());
    let container = encoded_container(dir.path());
    let intact = executor_trace_from(container.path());

    mutate_graph(container.path(), |target| {
        target["execution"]["norm"]["final_norm"]["weight_offset"] = serde_json::json!(1.0);
    });
    let mutated = executor_trace_from(container.path());

    for (layer, (a, b)) in intact.layers.iter().zip(&mutated.layers).enumerate() {
        let attn = max_abs(a.post_attention.rows(), b.post_attention.rows());
        let post = max_abs(a.post_layer.rows(), b.post_layer.rows());
        assert!(
            attn < NOISE_CEILING && post < NOISE_CEILING,
            "layer {layer} moved, but the final norm runs after every layer \
             (attn {attn:e}, post {post:e})"
        );
    }
    let final_gap = max_abs(
        &[mutated.final_hidden().to_vec()],
        &[intact.final_hidden().to_vec()],
    );
    assert!(
        final_gap > NOISE_CEILING,
        "the final norm did not move under its own offset ({final_gap:e})"
    );
    let logits_gap = max_abs(
        std::slice::from_ref(mutated.logits.as_ref().unwrap()),
        std::slice::from_ref(intact.logits.as_ref().unwrap()),
    );
    assert!(
        logits_gap > NOISE_CEILING,
        "logits did not move under the final norm's offset ({logits_gap:e})"
    );
}
