//! Backend-seam gates (V3-G5b-3b): the interpreter owns meaning, the
//! backend owns arithmetic.
//!
//! Two properties, and the second is what stops the first from being
//! vacuous:
//!
//! 1. **Swapping the backend does not change which operations run.** The
//!    op sequence is a function of the plan alone.
//! 2. **The backend really is doing the arithmetic.** A backend whose
//!    numbers differ must change the result — otherwise "both backends
//!    agree" could just mean the seam is bypassed and the interpreter is
//!    still computing everything itself.
//!
//! Without (2), a seam that silently ignored its backend would pass (1)
//! perfectly.

use std::path::Path;
use std::sync::Mutex;

use super::golden::{executor_trace_from, max_abs, miniature_glimmer, G_TOKENS};
use crate::error::VindexError;
use crate::format::vindex3::encode::encode_system;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::{
    AttentionCall, AttentionStepCall, AttentionStepOut, FfnCall, NormCall, PlanBackend,
    ProjectCall, RoutedFfnCall, WeightSlice,
};
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::exec::{execute_plan, ExecutionTrace};
use crate::format::vindex3::opplan::plan_component_ops;

/// A perturbation far above f32 noise but far below "obviously broken",
/// so a result that ignores it cannot be excused as rounding.
const PERTURBATION: f32 = 1.001;

/// Records the operation sequence while delegating every computation to
/// the reference backend, so the recording is numerically transparent.
struct RecordingBackend {
    inner: ReferenceBackend,
    ops: Mutex<Vec<&'static str>>,
}

impl RecordingBackend {
    fn new() -> Self {
        Self {
            inner: ReferenceBackend::new(),
            ops: Mutex::new(Vec::new()),
        }
    }

    fn record(&self, op: &'static str) {
        self.ops.lock().unwrap().push(op);
    }
}

impl PlanBackend for RecordingBackend {
    fn name(&self) -> &str {
        "recording"
    }

    fn embed(&self, table: &[f32], hidden: usize, token: u32, scale: Option<f32>) -> Vec<f32> {
        self.record("embed");
        self.inner.embed(table, hidden, token, scale)
    }

    fn norm(&self, call: NormCall<'_>) -> Vec<f32> {
        self.record("norm");
        self.inner.norm(call)
    }

    fn project(&self, call: ProjectCall<'_>) -> Result<Vec<f32>, VindexError> {
        self.record("project");
        self.inner.project(call)
    }

    fn attention(&self, call: AttentionCall<'_>) -> Result<Vec<Vec<f32>>, VindexError> {
        self.record("attention");
        self.inner.attention(call)
    }

    fn attention_step(&self, call: AttentionStepCall<'_>) -> Result<AttentionStepOut, VindexError> {
        self.record("attention_step");
        self.inner.attention_step(call)
    }

    fn ffn(&self, call: FfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        self.record("ffn");
        self.inner.ffn(call)
    }

    fn routed_ffn(&self, call: RoutedFfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        self.record("routed_ffn");
        self.inner.routed_ffn(call)
    }

    fn output_head(
        &self,
        projection: WeightSlice<'_>,
        vocab: usize,
        hidden: usize,
        x: &[f32],
        multiplier: Option<f64>,
        softcapping: Option<f32>,
    ) -> Result<Vec<f32>, VindexError> {
        self.record("output_head");
        self.inner
            .output_head(projection, vocab, hidden, x, multiplier, softcapping)
    }

    fn residual_add(&self, acc: &mut [f32], delta: &[f32]) {
        self.record("residual_add");
        self.inner.residual_add(acc, delta)
    }
}

/// Reference arithmetic with every normalisation scaled by
/// [`PERTURBATION`] — a backend that computes something genuinely
/// different, used only to prove the seam carries arithmetic at all.
struct PerturbedBackend(ReferenceBackend);

impl PlanBackend for PerturbedBackend {
    fn name(&self) -> &str {
        "perturbed"
    }

    fn embed(&self, table: &[f32], hidden: usize, token: u32, scale: Option<f32>) -> Vec<f32> {
        self.0.embed(table, hidden, token, scale)
    }

    fn norm(&self, call: NormCall<'_>) -> Vec<f32> {
        self.0
            .norm(call)
            .into_iter()
            .map(|v| v * PERTURBATION)
            .collect()
    }

    fn project(&self, call: ProjectCall<'_>) -> Result<Vec<f32>, VindexError> {
        self.0.project(call)
    }

    fn attention(&self, call: AttentionCall<'_>) -> Result<Vec<Vec<f32>>, VindexError> {
        self.0.attention(call)
    }

    fn attention_step(&self, call: AttentionStepCall<'_>) -> Result<AttentionStepOut, VindexError> {
        self.0.attention_step(call)
    }

    fn ffn(&self, call: FfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        self.0.ffn(call)
    }

    fn routed_ffn(&self, call: RoutedFfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        self.0.routed_ffn(call)
    }

    fn output_head(
        &self,
        projection: WeightSlice<'_>,
        vocab: usize,
        hidden: usize,
        x: &[f32],
        multiplier: Option<f64>,
        softcapping: Option<f32>,
    ) -> Result<Vec<f32>, VindexError> {
        self.0
            .output_head(projection, vocab, hidden, x, multiplier, softcapping)
    }

    fn residual_add(&self, acc: &mut [f32], delta: &[f32]) {
        self.0.residual_add(acc, delta)
    }
}

/// Encode the miniature Glimmer fixture once and run `backend` over it.
fn run_on<B: PlanBackend>(container: &Path, backend: &B) -> ExecutionTrace {
    let inspection = inspect_container(container, false).unwrap();
    let outcome = plan_component_ops(&inspection, container, "target").unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    let store = OperandStore::open(container, &inspection).unwrap();
    execute_plan(&outcome.plan.unwrap(), &store, &G_TOKENS, backend).unwrap()
}

fn encoded_miniature() -> tempfile::TempDir {
    let source = tempfile::tempdir().unwrap();
    miniature_glimmer(source.path());
    let inventory = larql_models::inventory::build_inventory(source.path()).unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_system(&[("mini-glimmer".to_string(), inventory)], container.path()).unwrap();
    container
}

/// A backend that delegates every computation reproduces the reference
/// trace exactly — the interpreter routes all arithmetic through the
/// seam, and routing it does not perturb it.
#[test]
fn a_delegating_backend_reproduces_the_reference_trace_bit_for_bit() {
    let container = encoded_miniature();
    let reference = executor_trace_from(container.path());
    let recorded = run_on(container.path(), &RecordingBackend::new());

    for (layer, (a, b)) in reference.layers.iter().zip(&recorded.layers).enumerate() {
        assert_eq!(
            max_abs(&a.post_attention, &b.post_attention),
            0.0,
            "layer {layer} post_attention differs across backends"
        );
        assert_eq!(
            max_abs(&a.post_layer, &b.post_layer),
            0.0,
            "layer {layer} post_layer differs across backends"
        );
    }
    assert_eq!(reference.logits, recorded.logits);
}

/// The operation sequence is a function of the plan, not of the backend.
///
/// Asserted against the fixture's known shape rather than a golden blob,
/// so the test says what it means: one attention per layer, one output
/// head, and two residual writes per layer per position.
#[test]
fn the_operation_sequence_is_determined_by_the_plan() {
    let container = encoded_miniature();
    let backend = RecordingBackend::new();
    let _ = run_on(container.path(), &backend);
    let ops = backend.ops.lock().unwrap();

    let layers = super::golden::G_LAYERS;
    let positions = G_TOKENS.len();
    assert_eq!(
        ops.iter().filter(|o| **o == "attention").count(),
        layers,
        "one attention op per layer"
    );
    assert_eq!(
        ops.iter().filter(|o| **o == "ffn").count(),
        layers * positions,
        "one FFN per layer per position"
    );
    assert_eq!(
        ops.iter().filter(|o| **o == "embed").count(),
        positions,
        "one embedding lookup per token"
    );
    assert_eq!(
        ops.iter().filter(|o| **o == "output_head").count(),
        1,
        "exactly one output head"
    );
    // Four-norm placement: attention and FFN each write a residual.
    assert_eq!(
        ops.iter().filter(|o| **o == "residual_add").count(),
        layers * positions * 2,
        "two residual writes per layer per position"
    );
    // The first thing any plan does is embed, and the last is the head.
    assert_eq!(ops.first().copied(), Some("embed"));
    assert_eq!(ops.last().copied(), Some("output_head"));
}

/// **The non-degeneracy control.** A backend whose arithmetic differs
/// must change the result.
///
/// If this ever passes-by-agreeing, the seam is decorative: the
/// interpreter would be computing the numbers itself and ignoring the
/// backend, and every cross-backend parity claim built on it would be
/// vacuous.
#[test]
fn a_backend_with_different_arithmetic_changes_the_result() {
    let container = encoded_miniature();
    let reference = executor_trace_from(container.path());
    let perturbed = run_on(container.path(), &PerturbedBackend(ReferenceBackend::new()));

    let divergence = max_abs(
        &reference.layers[0].post_attention,
        &perturbed.layers[0].post_attention,
    );
    assert!(
        divergence > 0.0,
        "perturbing the backend's norm changed nothing — the seam is not carrying arithmetic"
    );
    assert_ne!(
        reference.logits, perturbed.logits,
        "perturbing the backend's norm left the logits identical"
    );
}

// ── The three-way triangle (V3-G5b-3b) ──

/// Tolerance between two independent f32 realisations of the same
/// program on 12-wide state: reassociation and kernel formulation only
/// (BLAS sgemv vs scalar loop, divide-by-rms vs multiply-by-reciprocal).
const TRIANGLE_TOLERANCE: f32 = 1e-5;

/// `golden ↔ reference ↔ production` on the miniature Glimmer fixture.
///
/// The fixture's window (`G_WINDOW` 3 over 5 positions) genuinely
/// truncates from position 3 — see `controls::c4` — so this closes
/// sliding-window masking too, not just the unmasked path.
#[test]
fn the_triangle_closes_on_the_miniature_fixture() {
    let source = tempfile::tempdir().unwrap();
    miniature_glimmer(source.path());
    let golden = super::golden::golden_forward(source.path());

    let inventory = larql_models::inventory::build_inventory(source.path()).unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_system(&[("mini-glimmer".to_string(), inventory)], container.path()).unwrap();

    let reference = executor_trace_from(container.path());
    let production = run_on(container.path(), &ProductionBackend::new());

    for layer in 0..super::golden::G_LAYERS {
        for (name, g, r, p) in [
            (
                "post_attention",
                &golden.layers[layer].post_attention,
                &reference.layers[layer].post_attention,
                &production.layers[layer].post_attention,
            ),
            (
                "post_layer",
                &golden.layers[layer].post_layer,
                &reference.layers[layer].post_layer,
                &production.layers[layer].post_layer,
            ),
        ] {
            let vs_golden = max_abs(p, g);
            let vs_reference = max_abs(p, r);
            assert!(
                vs_golden < TRIANGLE_TOLERANCE,
                "layer {layer} {name}: production diverges from the independent golden \
                 oracle ({vs_golden:e})"
            );
            assert!(
                vs_reference < TRIANGLE_TOLERANCE,
                "layer {layer} {name}: production diverges from the reference backend \
                 ({vs_reference:e})"
            );
        }
    }

    let reference_logits = reference.logits.expect("plan carries an output head");
    let production_logits = production.logits.expect("plan carries an output head");
    let logit_gap = max_abs(
        std::slice::from_ref(&production_logits),
        std::slice::from_ref(&reference_logits),
    );
    assert!(
        logit_gap < TRIANGLE_TOLERANCE,
        "logits diverge between backends ({logit_gap:e})"
    );
    let golden_gap = max_abs(
        std::slice::from_ref(&production_logits),
        std::slice::from_ref(&golden.logits),
    );
    assert!(
        golden_gap < TRIANGLE_TOLERANCE,
        "logits diverge from the golden oracle ({golden_gap:e})"
    );

    // Non-degeneracy: two genuinely independent realisations must not
    // agree bit-exactly. If they do, they are sharing arithmetic, and
    // the agreement above proves nothing about either.
    assert_ne!(
        production_logits, reference_logits,
        "production and reference logits are bit-identical — the backends are sharing \
         arithmetic rather than independently realising the same program"
    );
}
