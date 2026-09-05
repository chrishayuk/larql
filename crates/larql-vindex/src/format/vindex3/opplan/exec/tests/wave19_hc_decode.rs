//! **Wave 19a — the decode traversal carries the hyper-connection bundle,
//! and an intermediate-state witness proves it.**
//!
//! The claim, kept to what the freeze allows: on a hyper-connected
//! component the decode step's residual is a bundle; each site reduces it
//! to the `[hidden]` vector the ordinary operator consumes and expands
//! the operator's output back; the observer sees state that cannot exist
//! on the single-stream path. The public refusal at
//! `PreparedOperands::load` stands throughout — every image here is
//! prepared through the test-only witness seam.
//!
//! # The assertions (the freeze's A1–A6, decode half)
//!
//! ```text
//! A1  foreign      at layer 0's attention site, fed the ORACLE's own
//!                  state, the tapped split and reduced vector equal the
//!                  oracle's stage outputs, all three positions
//! A2  structural   bundle_out == expand(b, x_in, split), and it is NOT
//!                  x_in + b
//! A3  branch in    the attention-input tap == pre_attention(reduced)
//! A4  FFN site     b_ffn == the FFN recomputed from the reduced vector
//!                  (hybrid: this pins the router's raw residual)
//! A5  width        every record is streams x hidden, one per site per
//!                  layer per position
//! A6  headless     Full on the headless variant refuses at preparation
//!                  with the head reason; LayerRange runs; HeadBearing
//!                  Full reduces through the head and produces logits
//! ```
//!
//! # The controls (mutants a–d)
//!
//! Each perturbs the real traversal or the real composition, and each
//! must be caught by the assertion the freeze names — a witness that
//! could not fail would prove nothing. The unmutated traversal passes
//! every assertion first, so "caught" is a difference, not a default.

use crate::format::vindex3::opplan::exec::decode::{DecodeSession, StepRun};
use crate::format::vindex3::opplan::exec::hyper_connection::{
    expand_streams, head_reduce, Bundle, Mutation, SinkhornSplit,
};
use crate::format::vindex3::opplan::exec::kv::RowKvState;
use crate::format::vindex3::opplan::exec::observe::{
    HcSite, HcSiteRecord, InputSite, NoopObserver, StepEvent, StepObserver,
};
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::exec::{execute_prepared_streaming, execute_text};

use super::wave19_hc_substrate::{
    self as substrate, Oracle, Substrate, Variant, HIDDEN, LAYERS, MIX_ROWS, NORM_EPS, POSITIONS,
    STREAMS, VOCAB,
};

/// The wave-17 stage tolerance: f32 transcription noise between torch's
/// vectorised accumulation and scalar loops, at values around 20.
const TOLERANCE: f32 = 5e-5;
/// The smallest disagreement a control must produce to count as one —
/// well above [`TOLERANCE`], so "differs" cannot be transcription noise.
const CONTROL_FLOOR: f32 = 1e-4;
/// Sites per layer.
const SITES: usize = 2;

/// One site's record, owned.
struct Record {
    layer: usize,
    site: HcSite,
    position: usize,
    split: SinkhornSplit,
    reduced: Vec<f32>,
    branch_output: Vec<f32>,
    bundle_out: Bundle,
}

/// The observer: records every site and the attention-input tap.
#[derive(Default)]
struct Witness {
    records: Vec<Record>,
    attention_inputs: Vec<(usize, Vec<f32>)>,
    events: Vec<StepEvent>,
}

impl StepObserver for Witness {
    fn event(&mut self, event: StepEvent) {
        self.events.push(event);
    }

    fn operand_input(&mut self, layer: usize, site: InputSite, values: &[f32]) {
        if site == InputSite::Attention {
            self.attention_inputs.push((layer, values.to_vec()));
        }
    }

    fn hyper_connection_site(&mut self, record: HcSiteRecord<'_>) {
        self.records.push(Record {
            layer: record.layer,
            site: record.site,
            position: record.position,
            split: record.split.clone(),
            reduced: record.reduced.to_vec(),
            branch_output: record.branch_output.to_vec(),
            bundle_out: record.bundle_out.clone(),
        });
    }
}

impl Witness {
    fn record(&self, layer: usize, site: HcSite, position: usize) -> Option<&Record> {
        self.records
            .iter()
            .find(|r| r.layer == layer && r.site == site && r.position == position)
    }

    fn attention_input(&self, layer: usize, position: usize) -> Option<&[f32]> {
        self.attention_inputs
            .iter()
            .filter(|(l, _)| *l == layer)
            .nth(position)
            .map(|(_, v)| v.as_slice())
    }
}

/// A decode run's observations and outputs.
struct Run {
    witness: Witness,
    steps: Vec<StepRun>,
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "comparing different shapes");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn close(actual: &[f32], expected: &[f32], what: &str) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{what}: {} values against {}",
            actual.len(),
            expected.len()
        ));
    }
    let diff = max_abs_diff(actual, expected);
    if diff <= TOLERANCE {
        Ok(())
    } else {
        Err(format!("{what}: max |diff| {diff:e} exceeds {TOLERANCE:e}"))
    }
}

fn store(sub: &Substrate) -> OperandStore {
    OperandStore::open(sub.container.path(), &sub.inspection).unwrap()
}

/// Prepare through the witness seam.
fn prepare(sub: &Substrate, slice: ExecutionSlice) -> Result<PreparedOperands, String> {
    let store = store(sub);
    PreparedOperands::load_for_hyper_connection_witness(
        &sub.plan,
        &store,
        &ReferenceBackend::new(),
        slice,
    )
    .map_err(|e| e.to_string())
}

/// The seam's refusal text, for the negative preparations.
fn prepare_err(sub: &Substrate, slice: ExecutionSlice) -> String {
    match prepare(sub, slice) {
        Ok(_) => panic!("preparation succeeded where it must refuse"),
        Err(err) => err,
    }
}

/// The whole layer range of the substrate: the headless shape.
fn layer_range() -> ExecutionSlice {
    ExecutionSlice::LayerRange {
        start: 0,
        end: LAYERS,
    }
}

/// Run the oracle's three positions through the stack, each entering
/// layer 0 as the oracle's own bundle, under `mutation`.
fn run_from_oracle(sub: &Substrate, slice: ExecutionSlice, mutation: Mutation) -> Run {
    let ops = prepare(sub, slice).expect("the seam prepares the substrate");
    let backend = ReferenceBackend::new();
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(&sub.plan, &ops, &backend, &mut kv).unwrap();
    let oracle = Oracle::load();
    let mut witness = Witness::default();
    let mut steps = Vec::new();
    for position in 0..POSITIONS {
        let step = session
            .step_from_bundle(oracle.input(position), &mut witness, mutation)
            .expect("the step runs");
        steps.push(step);
    }
    Run { witness, steps }
}

/// Run `tokens` through a whole-stack image from the embedding, under
/// `mutation`.
fn run_from_tokens(sub: &Substrate, tokens: &[u32], mutation: Mutation) -> Run {
    let ops = prepare(sub, ExecutionSlice::Full).expect("the seam prepares the substrate");
    let backend = ReferenceBackend::new();
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(&sub.plan, &ops, &backend, &mut kv).unwrap();
    let mut witness = Witness::default();
    let mut steps = Vec::new();
    for &token in tokens {
        steps.push(
            session
                .step_mutated(token, &mut witness, mutation)
                .expect("the step runs"),
        );
    }
    Run { witness, steps }
}

// ── The assertions ──

/// A1. The foreign reference: at layer 0's attention site, fed the
/// oracle's state, the split and the reduced vector are the oracle's.
fn a1_foreign_stages(run: &Run, oracle: &Oracle) -> Result<(), String> {
    for position in 0..POSITIONS {
        let record = run
            .witness
            .record(0, HcSite::Attention, position)
            .ok_or_else(|| format!("A1: no record at layer 0 attention, position {position}"))?;
        close(
            &record.split.pre,
            &oracle.stage("sinkhorn_pre", position),
            &format!("A1 position {position} pre"),
        )?;
        close(
            &record.split.post,
            &oracle.stage("sinkhorn_post", position),
            &format!("A1 position {position} post"),
        )?;
        close(
            &record.split.comb,
            &oracle.stage("sinkhorn_comb", position),
            &format!("A1 position {position} comb"),
        )?;
        close(
            &record.reduced,
            &oracle.stage("reduced", position),
            &format!("A1 position {position} reduced"),
        )?;
    }
    Ok(())
}

/// The bundle that entered each record's site, in emission order: the
/// oracle's state at layer 0's attention site, and the previous
/// record's output everywhere else.
fn entering_bundles<'a>(run: &'a Run, oracle: &Oracle) -> Vec<(&'a Record, Bundle)> {
    let mut out = Vec::new();
    let mut previous: Option<&Bundle> = None;
    for record in &run.witness.records {
        let entering = if record.layer == 0 && record.site == HcSite::Attention {
            oracle.input(record.position)
        } else {
            previous
                .expect("every site after the first follows a record")
                .clone()
        };
        out.push((record, entering));
        previous = Some(&record.bundle_out);
    }
    out
}

/// A2. Stage five ran, and it is not a residual add.
fn a2_expansion(run: &Run, oracle: &Oracle) -> Result<(), String> {
    for (record, x) in entering_bundles(run, oracle) {
        let what = format!(
            "A2 layer {} {:?} position {}",
            record.layer, record.site, record.position
        );
        let expected = expand_streams(
            &record.branch_output,
            x.as_flat(),
            &record.split,
            STREAMS,
            HIDDEN,
        );
        close(record.bundle_out.as_flat(), &expected, &what)?;
        // The single-stream form: every stream gets `x + b`.
        let added: Vec<f32> = (0..STREAMS)
            .flat_map(|j| {
                x.stream(j)
                    .iter()
                    .zip(&record.branch_output)
                    .map(|(r, b)| r + b)
                    .collect::<Vec<_>>()
            })
            .collect();
        let diff = max_abs_diff(record.bundle_out.as_flat(), &added);
        if diff < CONTROL_FLOOR {
            return Err(format!(
                "{what}: the bundle is within {diff:e} of a residual add"
            ));
        }
    }
    Ok(())
}

/// A3. The attention input is the pre-norm of the REDUCED vector.
fn a3_branch_input(run: &Run, ops: &PreparedOperands) -> Result<(), String> {
    let backend = ReferenceBackend::new();
    for layer in 0..LAYERS {
        let norm = ops.layers()[layer]
            .pre_attention
            .as_ref()
            .ok_or("A3: the substrate has a pre-attention norm")?;
        for position in 0..POSITIONS {
            let record = run
                .witness
                .record(layer, HcSite::Attention, position)
                .ok_or_else(|| format!("A3: no record at layer {layer} position {position}"))?;
            let tapped = run
                .witness
                .attention_input(layer, position)
                .ok_or_else(|| format!("A3: no attention-input tap at layer {layer}"))?;
            close(
                tapped,
                &norm.apply(&backend, &record.reduced),
                &format!("A3 layer {layer} position {position}"),
            )?;
        }
    }
    Ok(())
}

/// A4. The FFN site's branch output is the FFN recomputed from the
/// reduced vector — residual argument included.
fn a4_ffn_branch(run: &Run, sub: &Substrate, ops: &PreparedOperands) -> Result<(), String> {
    let backend = ReferenceBackend::new();
    for layer in 0..LAYERS {
        let prepared = &ops.layers()[layer];
        let plan_layer = &sub.plan.layers[layer];
        let (ffn, ffn_op) = match (&prepared.ffn, &plan_layer.ffn) {
            (Some(ffn), Some(op)) => (ffn, op),
            _ => return Err(format!("A4: layer {layer} carries no FFN")),
        };
        for position in 0..POSITIONS {
            let record = run
                .witness
                .record(layer, HcSite::Ffn, position)
                .ok_or_else(|| format!("A4: no FFN record at layer {layer} position {position}"))?;
            let v = &record.reduced;
            let normed = match &prepared.pre_ffn {
                Some(norm) => norm.apply(&backend, v),
                None => v.clone(),
            };
            let out = ffn
                .apply_from_residual(ffn_op, &backend, v, &normed, HIDDEN)
                .map_err(|e| e.to_string())?;
            let mut out = match &prepared.post_ffn {
                Some(norm) => norm.apply(&backend, &out),
                None => out,
            };
            crate::format::vindex3::opplan::exec::scale_residual_delta(
                plan_layer.residual_scale,
                &mut out,
            );
            close(
                &record.branch_output,
                &out,
                &format!("A4 layer {layer} position {position}"),
            )?;
        }
    }
    Ok(())
}

/// A5. One record per site per layer per position, every one a
/// `streams x hidden` bundle with `[hidden]` vectors beside it.
fn a5_width(run: &Run) -> Result<(), String> {
    let expected = LAYERS * SITES * POSITIONS;
    if run.witness.records.len() != expected {
        return Err(format!(
            "A5: {} records, expected {expected}",
            run.witness.records.len()
        ));
    }
    for record in &run.witness.records {
        let what = format!(
            "A5 layer {} {:?} position {}",
            record.layer, record.site, record.position
        );
        if record.bundle_out.streams() != STREAMS || record.bundle_out.hidden() != HIDDEN {
            return Err(format!(
                "{what}: bundle is {} x {}",
                record.bundle_out.streams(),
                record.bundle_out.hidden()
            ));
        }
        if record.reduced.len() != HIDDEN || record.branch_output.len() != HIDDEN {
            return Err(format!("{what}: the branch vectors are not [hidden]"));
        }
        if record.split.pre.len() != STREAMS
            || record.split.post.len() != STREAMS
            || record.split.comb.len() != STREAMS * STREAMS
        {
            return Err(format!("{what}: the split is not [streams]/[streams^2]"));
        }
    }
    Ok(())
}

/// The whole decode chain on a headless, layer-range run.
fn witness_headless(mutation: Mutation) -> Vec<(&'static str, Result<(), String>)> {
    let sub = substrate::build(Variant::Headless);
    let oracle = Oracle::load();
    let run = run_from_oracle(&sub, layer_range(), mutation);
    let ops = prepare(&sub, layer_range()).unwrap();
    vec![
        ("A1", a1_foreign_stages(&run, &oracle)),
        ("A2", a2_expansion(&run, &oracle)),
        ("A3", a3_branch_input(&run, &ops)),
        ("A4", a4_ffn_branch(&run, &sub, &ops)),
        ("A5", a5_width(&run)),
    ]
}

/// The hybrid chain: A4 on the estate whose router reads the residual.
fn witness_hybrid(mutation: Mutation) -> Vec<(&'static str, Result<(), String>)> {
    let sub = substrate::build(Variant::Hybrid);
    let oracle = Oracle::load();
    let run = run_from_oracle(&sub, layer_range(), mutation);
    let ops = prepare(&sub, layer_range()).unwrap();
    vec![
        ("A1", a1_foreign_stages(&run, &oracle)),
        ("A2", a2_expansion(&run, &oracle)),
        ("A4", a4_ffn_branch(&run, &sub, &ops)),
        ("A5", a5_width(&run)),
    ]
}

fn assert_all_hold(results: &[(&str, Result<(), String>)]) {
    for (name, result) in results {
        assert!(result.is_ok(), "{name} failed: {:?}", result);
    }
}

fn assert_caught_by(results: &[(&str, Result<(), String>)], named: &[&str]) {
    for name in named {
        let (_, result) = results
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("{name} is not in the chain"));
        assert!(result.is_err(), "{name} did not catch the mutant");
    }
}

// ── The positive witness ──

#[test]
fn the_decode_traversal_runs_the_bundle_and_the_witness_holds() {
    assert_all_hold(&witness_headless(Mutation::None));
}

#[test]
fn the_hybrid_ffn_receives_the_reduced_vector() {
    assert_all_hold(&witness_hybrid(Mutation::None));
}

/// The embedding enters replicated: at layer 0's attention site the
/// reduced vector is `(Σ pre) · e` for the row the token embeds to.
#[test]
fn the_embedding_enters_every_stream_and_the_head_reduces_the_exit() {
    let sub = substrate::build(Variant::HeadBearing);
    let oracle = Oracle::load();
    let tokens = [2u32, 5, 1];
    let run = run_from_tokens(&sub, &tokens, Mutation::None);
    for (position, &token) in tokens.iter().enumerate() {
        let record = run.witness.record(0, HcSite::Attention, position).unwrap();
        let e = substrate::llama_embedding_row(token);
        let weight: f32 = record.split.pre.iter().sum();
        let expected: Vec<f32> = e.iter().map(|v| v * weight).collect();
        close(&record.reduced, &expected, "replicated embedding").unwrap();
    }
    // The exit: the bundle after the last layer reduces through the
    // head's own operation (the oracle's head weights), then the final
    // norm and the output head run as they always have.
    let head = oracle.head();
    let hc = oracle.topology();
    for step in &run.steps {
        let bundle = step.bundle.as_ref().expect("a bundle left the stack");
        let expected = head_reduce(
            bundle.as_flat(),
            STREAMS,
            HIDDEN,
            &head.weights(),
            NORM_EPS,
            hc.sinkhorn_eps,
        );
        close(step.exit.as_deref().unwrap(), &expected, "head reduction").unwrap();
        assert_eq!(step.logits.as_ref().unwrap().len(), VOCAB);
    }
    assert!(run
        .witness
        .events
        .iter()
        .any(|e| matches!(e, StepEvent::Logits { vocab } if *vocab == VOCAB)));
}

/// Observation cannot fork the semantics: the unobserved step and the
/// witnessed step produce the same logits, bit for bit.
#[test]
fn the_witnessed_step_is_the_unobserved_step() {
    let sub = substrate::build(Variant::HeadBearing);
    let ops = prepare(&sub, ExecutionSlice::Full).unwrap();
    let backend = ReferenceBackend::new();
    let tokens = [3u32, 0, 6];
    let mut kv = RowKvState::default();
    let mut plain = DecodeSession::over_prepared(&sub.plan, &ops, &backend, &mut kv).unwrap();
    let unobserved: Vec<Vec<f32>> = tokens
        .iter()
        .map(|&t| {
            plain
                .step_observed(t, &mut NoopObserver)
                .unwrap()
                .logits
                .unwrap()
        })
        .collect();
    let mut kv = RowKvState::default();
    let mut seen = DecodeSession::over_prepared(&sub.plan, &ops, &backend, &mut kv).unwrap();
    let mut witness = Witness::default();
    let observed: Vec<Vec<f32>> = tokens
        .iter()
        .map(|&t| seen.step_observed(t, &mut witness).unwrap().logits.unwrap())
        .collect();
    assert_eq!(unobserved, observed);
    assert_eq!(witness.records.len(), LAYERS * SITES * tokens.len());
}

// ── The controls ──

#[test]
fn mutant_a_bypassing_the_composition_is_caught_by_a1_and_a5() {
    assert_caught_by(
        &witness_headless(Mutation::BypassComposition),
        &["A1", "A5"],
    );
}

#[test]
fn mutant_b_one_sinkhorn_iteration_is_caught_by_a1() {
    let results = witness_headless(Mutation::SingleIteration);
    assert_caught_by(&results, &["A1"]);
    // The defect is IN the split, so the expansion is consistent with
    // the (wrong) split it reports: A2 must not be the one that catches
    // it, or the assertions are not measuring what they claim.
    assert_all_hold(&results[1..2]);
}

#[test]
fn mutant_b_a_uniform_reduction_is_caught_by_a1() {
    assert_caught_by(&witness_headless(Mutation::UniformReduction), &["A1"]);
}

#[test]
fn mutant_b_a_transposed_combination_is_caught_by_a2_and_not_a1() {
    let results = witness_headless(Mutation::TransposedCombination);
    assert_caught_by(&results, &["A2"]);
    // The split reported is the correct one — only the expansion lies.
    assert_all_hold(&results[..1]);
}

#[test]
fn mutant_c_pre_norm_on_a_stream_is_caught_by_a3_alone() {
    let results = witness_headless(Mutation::PreNormOnStreamZero);
    assert_caught_by(&results, &["A3"]);
    // The split, the reduction and the expansion are all correct; only
    // what the operator SAW is wrong, and only A3 looks there.
    assert_all_hold(&results[..1]);
}

#[test]
fn mutant_c_pre_norm_on_the_stream_mean_is_caught_by_a3() {
    assert_caught_by(&witness_headless(Mutation::PreNormOnStreamMean), &["A3"]);
}

#[test]
fn mutant_d_hybrid_residual_from_a_stream_is_caught_by_a4() {
    let results = witness_hybrid(Mutation::HybridResidualFromStreamZero);
    assert_caught_by(&results, &["A4"]);
    // The attention site is untouched by an FFN-site defect.
    assert_all_hold(&results[..1]);
}

#[test]
fn mutant_d_hybrid_residual_from_the_stream_mean_is_caught_by_a4() {
    assert_caught_by(
        &witness_hybrid(Mutation::HybridResidualFromStreamMean),
        &["A4"],
    );
}

/// Mutant (d) has nothing to bite on a dense FFN: the dense path reads
/// only its normed input, so the residual argument is inert there. The
/// hybrid estate is what makes the defect observable — recorded so the
/// choice of substrate is a measured fact, not an assumption.
#[test]
fn mutant_d_is_invisible_on_the_dense_estate() {
    assert_all_hold(&witness_headless(Mutation::HybridResidualFromStreamZero));
}

// ── The refusals, unchanged and new ──

/// P1. The public loader still refuses every hyper-connected variant
/// with the topology reason, and so does public execution.
#[test]
fn the_public_loader_still_refuses_every_variant() {
    for variant in [Variant::Headless, Variant::HeadBearing, Variant::Hybrid] {
        let sub = substrate::build(variant);
        let store = store(&sub);
        let err = PreparedOperands::load(
            &sub.plan,
            &store,
            &ReferenceBackend::new(),
            ExecutionSlice::Full,
        )
        .err()
        .map(|e| e.to_string())
        .unwrap_or_else(|| {
            panic!("{variant:?}: the public loader prepared a hyper-connected plan")
        });
        assert!(err.contains("cannot execute it"), "{variant:?}: {err}");
        assert!(err.contains("traversal"), "{variant:?}: {err}");
        let err = execute_text(&sub.plan, &store, &[1, 2])
            .err()
            .map(|e| e.to_string())
            .unwrap_or_else(|| panic!("{variant:?}: public execution ran"));
        assert!(err.contains("cannot execute it"), "{variant:?}: {err}");
    }
}

/// The seam is for hyper-connected plans only: the single-stream sibling
/// is sent back to the public path.
#[test]
fn the_seam_refuses_a_single_stream_plan() {
    let sub = substrate::single_stream_sibling();
    let err = prepare_err(&sub, ExecutionSlice::Full);
    assert!(err.contains("single residual stream"), "{err}");
    assert!(err.contains("public path"), "{err}");
}

/// A6. Headless: a whole-stack image refuses at preparation, naming the
/// head; a layer-range image prepares and runs, and produces no logits.
#[test]
fn a_headless_component_prepares_as_a_layer_range_and_not_as_a_whole_stack() {
    let sub = substrate::build(Variant::Headless);
    let err = prepare_err(&sub, ExecutionSlice::Full);
    assert!(err.contains("hyper_connection_head"), "{err}");
    assert!(err.contains("layer-range"), "{err}");
    assert!(
        !err.contains("traversal"),
        "the head reason is not the topology reason: {err}"
    );
    let run = run_from_oracle(&sub, layer_range(), Mutation::None);
    for step in &run.steps {
        assert!(step.logits.is_none(), "a layer range produces no logits");
        assert!(step.exit.is_none(), "a layer range reduces nothing");
        assert!(
            step.bundle.is_some(),
            "the bundle is the layer range's output"
        );
    }
}

/// A layer scale under the topology is unjudged and refused by name.
#[test]
fn a_layer_scale_under_the_topology_is_refused_at_preparation() {
    let sub = substrate::build(Variant::HybridWithLayerScale);
    let err = prepare_err(&sub, layer_range());
    assert!(err.contains("layer scale"), "{err}");
    assert!(err.contains("unjudged"), "{err}");
}

/// The batch traversal refuses an image that carries the bundle: the
/// seam cannot make the batch path look supported before 19b.
#[test]
fn the_batch_traversal_refuses_a_hyper_connected_image() {
    let sub = substrate::build(Variant::HeadBearing);
    let ops = prepare(&sub, ExecutionSlice::Full).unwrap();
    let backend = ReferenceBackend::new();
    let err = execute_prepared_streaming(&sub.plan, &ops, &[1, 2], &backend, None, &mut |_| Ok(()))
        .err()
        .map(|e| e.to_string())
        .expect("the batch traversal refuses");
    assert!(err.contains("batch traversal"), "{err}");
    assert!(err.contains("19b"), "{err}");
    // `step_many` is the batch traversal in decode clothing.
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(&sub.plan, &ops, &backend, &mut kv).unwrap();
    let err = session
        .step_many(&[1, 2])
        .err()
        .map(|e| e.to_string())
        .unwrap();
    assert!(err.contains("batch traversal"), "{err}");
}

/// P4. The single stream is untouched: on the sibling with no topology,
/// the decode step's logits equal the batch traversal's, bit for bit —
/// and the batch path did not change in this wave.
#[test]
fn the_single_stream_sibling_decodes_exactly_as_it_batches() {
    let sub = substrate::single_stream_sibling();
    let store = store(&sub);
    let backend = ReferenceBackend::new();
    let tokens = [4u32, 1, 6, 2];
    let batch = execute_text(&sub.plan, &store, &tokens).unwrap();
    let mut session = DecodeSession::new(&sub.plan, &store, &backend).unwrap();
    let mut witness = Witness::default();
    let mut last = None;
    for &token in &tokens {
        last = session.step_observed(token, &mut witness).unwrap().logits;
    }
    assert_eq!(last.unwrap(), batch.logits.unwrap());
    assert!(
        witness.records.is_empty(),
        "a single-stream step emits no hyper-connection record"
    );
    assert!(!store_carries_hc(&sub));
}

fn store_carries_hc(sub: &Substrate) -> bool {
    let store = store(sub);
    PreparedOperands::load(
        &sub.plan,
        &store,
        &ReferenceBackend::new(),
        ExecutionSlice::Full,
    )
    .unwrap()
    .carries_hyper_connection()
}

/// The residency census counts the sites and the head as glue, by the
/// decision that made the mix projection f32.
#[test]
fn the_sites_and_the_head_are_counted_as_glue() {
    let headless = prepare(&substrate::build(Variant::Headless), layer_range()).unwrap();
    let head_bearing = prepare(
        &substrate::build(Variant::HeadBearing),
        ExecutionSlice::Full,
    )
    .unwrap();
    let site_bytes =
        LAYERS * SITES * (MIX_ROWS * STREAMS * HIDDEN + MIX_ROWS + 3) * std::mem::size_of::<f32>();
    let head_bytes = (STREAMS * STREAMS * HIDDEN + STREAMS) * std::mem::size_of::<f32>();
    let norms = |ops: &PreparedOperands| -> usize { ops.residency_census().glue.widened_f32 };
    // Headless: the layer norms plus the sites (a layer range carries no
    // final norm). Head-bearing: the same, plus the final norm and the
    // head's two vectors.
    let headless_glue = norms(&headless);
    let head_bearing_glue = norms(&head_bearing);
    assert!(
        headless_glue >= site_bytes,
        "{headless_glue} < {site_bytes}"
    );
    assert_eq!(
        head_bearing_glue - headless_glue,
        head_bytes + HIDDEN * std::mem::size_of::<f32>()
    );
}

// ── The loader's own refusals, on doctored plans ──
//
// Closure never produces these plans; the loader still refuses them by
// name rather than trusting the builder, because an image whose sites,
// topology and head disagree would run a wrong model fluently.

fn head_bearing_plan() -> (Substrate, OperandStore) {
    let sub = substrate::build(Variant::HeadBearing);
    let store = store(&sub);
    (sub, store)
}

fn seam_err(
    plan: &crate::format::vindex3::opplan::ComponentOpPlan,
    store: &OperandStore,
) -> String {
    match PreparedOperands::load_for_hyper_connection_witness(
        plan,
        store,
        &ReferenceBackend::new(),
        ExecutionSlice::Full,
    ) {
        Ok(_) => panic!("preparation succeeded where it must refuse"),
        Err(err) => err.to_string(),
    }
}

#[test]
fn a_layer_without_sites_under_the_topology_is_refused() {
    let (sub, store) = head_bearing_plan();
    let mut plan = sub.plan.clone();
    plan.layers[1].hyper_connection = None;
    let err = seam_err(&plan, &store);
    assert!(err.contains("layer 1 carries no"), "{err}");
}

#[test]
fn sites_on_a_single_stream_component_are_refused_by_the_public_loader() {
    let (sub, store) = head_bearing_plan();
    let mut plan = sub.plan.clone();
    plan.residual_topology = larql_models::config::ResidualTopology::SingleStream;
    let err = PreparedOperands::load(
        &plan,
        &store,
        &ReferenceBackend::new(),
        ExecutionSlice::Full,
    )
    .err()
    .map(|e| e.to_string())
    .expect("a single-stream plan with sites is refused");
    assert!(err.contains("single"), "{err}");
    assert!(err.contains("never produces"), "{err}");
}

#[test]
fn a_head_at_a_sites_geometry_is_refused() {
    let (sub, store) = head_bearing_plan();
    let site = sub.plan.layers[0].hyper_connection.clone().unwrap();
    let mut plan = sub.plan.clone();
    plan.hyper_connection_head.as_mut().unwrap().reduce_fn = site.attention.mix_fn.clone();
    let err = seam_err(&plan, &store);
    assert!(err.contains("head's geometry"), "{err}");

    let mut plan = sub.plan.clone();
    plan.hyper_connection_head.as_mut().unwrap().scale = site.attention.scale.clone();
    let err = seam_err(&plan, &store);
    assert!(err.contains("scale holds 3 values"), "{err}");
}

#[test]
fn a_site_at_the_heads_geometry_is_refused() {
    let (sub, store) = head_bearing_plan();
    let head = sub.plan.hyper_connection_head.clone().unwrap();
    let mut plan = sub.plan.clone();
    plan.layers[0].hyper_connection.as_mut().unwrap().ffn.mix_fn = head.reduce_fn;
    let err = seam_err(&plan, &store);
    assert!(err.contains("layer 0 ffn site: mix_fn holds"), "{err}");
}

#[test]
fn layers_that_disagree_on_norm_eps_leave_the_head_no_epsilon() {
    let (sub, store) = head_bearing_plan();
    let mut plan = sub.plan.clone();
    plan.layers[1].declared_norm_eps = 2.0 * NORM_EPS;
    let err = seam_err(&plan, &store);
    assert!(err.contains("one component value"), "{err}");
}

// ── The bundle entry's own refusals ──

#[test]
fn a_bundle_cannot_enter_a_single_stream_session() {
    let sub = substrate::single_stream_sibling();
    let store = store(&sub);
    let backend = ReferenceBackend::new();
    let mut session = DecodeSession::new(&sub.plan, &store, &backend).unwrap();
    let err = session
        .step_from_bundle(Oracle::load().input(0), &mut NoopObserver, Mutation::None)
        .err()
        .map(|e| e.to_string())
        .expect("a bundle has no meaning on one stream");
    assert!(err.contains("single-stream component"), "{err}");
}

#[test]
fn a_bundle_of_the_wrong_shape_is_refused_at_entry() {
    let sub = substrate::build(Variant::HeadBearing);
    let ops = prepare(&sub, ExecutionSlice::Full).unwrap();
    let backend = ReferenceBackend::new();
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(&sub.plan, &ops, &backend, &mut kv).unwrap();
    let narrow = Bundle::replicate(&[0.0; HIDDEN - 1], STREAMS);
    let err = session
        .step_from_bundle(narrow, &mut NoopObserver, Mutation::None)
        .err()
        .map(|e| e.to_string())
        .expect("the entering bundle must match the component");
    assert!(err.contains("entering bundle"), "{err}");
}
