//! CPU-2C: one physical projection, two consumers, computed once.
//!
//! Qwen3.8's output gate is the other half of its query projection over
//! the same activation, and the backend projected `12288 x 5120` twice
//! per softmax layer to collect the halves separately — 2.01 GB/token of
//! traffic and 2.01 GB of residency for a vector it had already computed
//! and thrown away.
//!
//! The sharing is licensed by the JUDGED gate source, not by the two
//! operands happening to resolve to the same tensor. So the controls here
//! are about when it applies and when it must not:
//!
//! ```text
//! same operand + same activation   ->  one projection   (FusedQueryProjection)
//! its own operand                  ->  two projections  (AttentionInput)
//! sharing withheld                 ->  two projections, still correct
//! ```
//!
//! The last one matters most: the fallback has to stay right, because the
//! reference backend deliberately never shares — it is the literal
//! transcription, and "the two agree" is the whole reason this
//! optimisation is allowed.

use larql_models::config::{
    AttentionGateSpec, GateActivation, GateCombine, GatePlacement, GateSource, ParameterFreeQkNorm,
    PositionPolicy,
};

use super::super::backend::{AttentionCall, AttentionStepCall, GateCall, PlanBackend, WeightSlice};
use super::super::cpu::thread_projection_calls;
use super::super::kv_view::KvView;
use super::super::production::ProductionBackend;
use crate::format::vindex3::fixtures::lcg_values;
use crate::format::vindex3::graph::policy::AttentionSpan;

const HEADS: usize = 2;
const HEAD_DIM: usize = 4;
const HIDDEN: usize = HEADS * HEAD_DIM;
const Q_ROWS: usize = HEADS * HEAD_DIM;

fn spec(source: GateSource) -> AttentionGateSpec {
    AttentionGateSpec {
        source,
        activation: GateActivation::Sigmoid,
        combine: GateCombine::ElementwiseMultiply,
        placement: GatePlacement::AfterAggregationBeforeOutputProjection,
    }
}

/// An attention call over tiny f32 weights, gated as asked.
///
/// `w_q` is double width for a fused gate — that IS the geometry the
/// source asserts — and ordinary width otherwise.
struct Fixture {
    inputs: Vec<Vec<f32>>,
    w_q: Vec<f32>,
    w_kv: Vec<f32>,
    w_o: Vec<f32>,
    gate_weight: Vec<f32>,
}

impl Fixture {
    fn new(fused: bool) -> Self {
        let q_out = if fused { Q_ROWS * 2 } else { Q_ROWS };
        Self {
            inputs: vec![lcg_values(HIDDEN, 3)],
            w_q: lcg_values(q_out * HIDDEN, 11),
            w_kv: lcg_values(Q_ROWS * HIDDEN, 12),
            w_o: lcg_values(HIDDEN * Q_ROWS, 13),
            gate_weight: lcg_values(Q_ROWS * HIDDEN, 14),
        }
    }

    fn call(&self, source: Option<GateSource>) -> AttentionCall<'_> {
        AttentionCall {
            inputs: &self.inputs,
            w_q: WeightSlice::F32(&self.w_q),
            w_k: WeightSlice::F32(&self.w_kv),
            w_v: WeightSlice::F32(&self.w_kv),
            w_o: WeightSlice::F32(&self.w_o),
            hidden: HIDDEN,
            head_dim: HEAD_DIM,
            num_q_heads: HEADS,
            num_kv_heads: HEADS,
            qk_norm: None,
            parameter_free_qk_norm: ParameterFreeQkNorm::default(),
            qk_norm_eps: 1e-6,
            query_scale: None,
            position: PositionPolicy::None,
            score_scale: 1.0 / (HEAD_DIM as f64).sqrt(),
            logit_softcapping: None,
            span: AttentionSpan::Full,
            window: None,
            gate: source.map(|source| GateCall {
                spec: spec(source),
                // A fused gate reads the QUERY operand — that is what the
                // source means, and what the call builder hands it.
                weight: match source {
                    GateSource::FusedQueryProjection => WeightSlice::F32(&self.w_q),
                    GateSource::AttentionInput => WeightSlice::F32(&self.gate_weight),
                },
            }),
            bias: None,
            sinks: None,
        }
    }
}

/// Projections THIS THREAD issued while `f` was running.
///
/// Per thread, not from the process ledger: the suite runs in parallel
/// and other tests project constantly, so an exact count against a shared
/// counter would be measuring them. Single-threaded runs would pass
/// either way, which is exactly how a test like this passes for the wrong
/// reason.
fn projections_during(f: impl FnOnce()) -> u64 {
    let before = thread_projection_calls();
    f();
    thread_projection_calls() - before
}

fn step(call: AttentionCall<'_>) -> Vec<f32> {
    ProductionBackend::new()
        .attention_step(AttentionStepCall::new(call, 0, KvView::empty()).unwrap())
        .expect("the tiny fixture attends")
        .output
}

/// **The claim.** A fused gate costs ONE query projection, not two.
///
/// Counted through the executor's own ledger rather than timed: the claim
/// is about how many times the operand is read, and a timing test would
/// pass on a fast machine whatever the answer.
#[test]
fn a_fused_gate_reads_the_query_operand_once() {
    let f = Fixture::new(true);
    let fused = projections_during(|| {
        step(f.call(Some(GateSource::FusedQueryProjection)));
    });

    // The same layer with no gate at all: q, k, v, o.
    let plain = Fixture::new(false);
    let ungated = projections_during(|| {
        step(plain.call(None));
    });

    assert_eq!(
        fused, ungated,
        "a fused gate must add no projection — its values come out of the query product, and \
         {fused} calls against {ungated} means the operand was read again"
    );
}

/// A gate with its OWN operand still costs its own projection.
///
/// The control that stops the optimisation from being "skip the gate
/// projection": an `AttentionInput` gate is a different matrix over a
/// different activation and sharing would be simply wrong.
#[test]
fn a_gate_with_its_own_operand_is_projected_separately() {
    let f = Fixture::new(false);
    let gated = projections_during(|| {
        step(f.call(Some(GateSource::AttentionInput)));
    });
    let ungated = projections_during(|| {
        step(f.call(None));
    });
    assert_eq!(
        gated,
        ungated + 1,
        "a gate with its own operand must cost one more projection"
    );
}

/// **Withholding the sharing changes the call count and nothing else.**
///
/// This drives `attend_position` twice over the SAME projected position:
/// once handed the gate half, once made to fetch it itself. That second
/// path is the one the reference backend takes and the one a future
/// caller reaching the seam another way would take, so it has to stay
/// right — "the two agree" is the entire licence for the optimisation.
///
/// Bit-identical, not merely close: both halves come from the same
/// deterministic product over the same activation, so recomputing it
/// reproduces every element exactly. A tolerance here would hide a real
/// difference behind a threshold nobody chose.
#[test]
fn withholding_the_shared_half_costs_a_projection_and_changes_no_value() {
    let f = Fixture::new(true);
    let call = f.call(Some(GateSource::FusedQueryProjection));
    let projected =
        ProductionBackend::project_position(&call, 0, &f.inputs[0]).expect("the fixture projects");
    let (q, k, v) = &projected.qkv;
    let gate = projected
        .gate
        .as_deref()
        .expect("a fused gate shares its product");

    let attend = |handed: Option<&[f32]>| {
        ProductionBackend::attend_position(
            &call,
            0,
            q,
            |_| k.as_slice(),
            |_| v.as_slice(),
            &f.inputs[0],
            handed,
        )
        .expect("the fixture attends")
    };

    let mut shared = Vec::new();
    let shared_calls = projections_during(|| shared = attend(Some(gate)));
    let mut refetched = Vec::new();
    let refetched_calls = projections_during(|| refetched = attend(None));

    assert_eq!(
        refetched_calls,
        shared_calls + 1,
        "withholding the shared half must cost exactly one more projection"
    );
    assert_eq!(
        shared, refetched,
        "sharing the query product and re-projecting it must agree EXACTLY — they are the same \
         arithmetic over the same activation, so any difference at all is a real one"
    );
}
