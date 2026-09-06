//! **K3-ATTNRES-1 2a — the decode traversal carries a residual history.**
//!
//! Frozen in `docs/arch-conformance/forecasts/k3-attnres-1-traverse.json`
//! against the oracle commit `ec7da08d`, and scored here. The claim is
//! narrow: the decode step carries an explicit prefix-plus-snapshots
//! state, reproduces the oracle's per-site probabilities and mixed
//! vectors, emits the schedule the reference spells, and is caught by
//! every named ordering mutation. Nothing lifts — the public loader
//! still refuses the topology, and the batch path refuses it by name.
//!
//! # Two kinds of evidence, and they answer different questions
//!
//! ```text
//! FOREIGN     each site replayed from the ORACLE's own recorded
//!             entering state, its probabilities and mixed vector
//!             compared against the oracle's. This is the arithmetic,
//!             and its reference is a torch transcription of Kimi-K3's
//!             own file.
//!
//! STRUCTURAL  a real decode run over a synthetic stack, its record
//!             stream compared against the oracle's SCHEDULE — which
//!             layers reduce, over how many candidates, and where the
//!             boundary events fall relative to the two sites.
//! ```
//!
//! The full-stack vectors are deliberately NOT compared against the
//! oracle's. The oracle's sublayer is a stand-in (pre-norm, linear,
//! tanh) and this substrate's operators are real attention and a real
//! FFN, so a final-vector comparison would be measuring the branch
//! rather than the topology. What crosses that boundary is the
//! per-site arithmetic and the schedule; the controls are scored
//! against this substrate's own reference run.
//!
//! # Two properties are proven by SHAPE, and can be proven no other way
//!
//! The oracle measured `layer0_attention_site_runs` and
//! `mlp_site_guarded_on_nonempty` at a divergence of EXACTLY zero.
//! Softmax over one candidate is the identity, so layer 0's skipped
//! attention site computes what a regularised always-run site computes;
//! and the mlp site's guard never fires, because no site in this
//! schedule ever sees an empty snapshot set. No value comparison at any
//! geometry can catch either. They are caught by which records exist,
//! and if this file ever reports them caught by a value assertion, that
//! assertion is broken.

use serde_json::Value;

use super::super::attention_residual::{self, History};
use super::super::decode::DecodeSession;
use super::super::hyper_connection::Mutation;
use super::super::kv::RowKvState;
use super::super::observe::{
    AttnResBoundaryRecord, AttnResSiteRecord, HcSite, NoopObserver, StepEvent, StepObserver,
};
use super::super::operands::OperandStore;
use super::super::prepared::{ExecutionSlice, PreparedOperands};
use super::super::reference::ReferenceBackend;
use crate::format::vindex3::encode::encode_graph;
use crate::format::vindex3::fixtures::ShardBuilder;
use crate::format::vindex3::inspect::{inspect_container, SystemInspection};
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};
use crate::format::vindex3::plan::plan_system;

/// The oracle's own geometry, so the schedule under test is the schedule
/// it exported: boundaries at 0, 3 and 6, and a four-candidate exit.
const HIDDEN: usize = 5;
const LAYERS: usize = 7;
const BLOCK: usize = 3;
const POSITIONS: usize = 3;
const VOCAB: usize = 7;
const NORM_EPS: f64 = 1e-5;

// Ordinary operator geometry, deliberately not derived from `HIDDEN`, so
// no shape coincidence can let a transposed operand pass.
const HEADS: usize = 1;
const HEAD_DIM: usize = 4;
const INTER: usize = 4;

/// f32 storage throughout and a stand-in-free comparison, so this is
/// transcription noise and nothing else.
const TOLERANCE: f32 = 5e-5;

/// The floor and ceiling the oracle ships. A substrate outside this band
/// has a saturated or starved softmax, on which every candidate-set
/// control is invisible — the failure the oracle demonstrated on itself.
const MIN_PROB: f32 = 5e-3;
const MAX_PROB: f32 = 0.98;

const ORACLE: &str = include_str!("attn_res_oracle.json");

// ── The oracle, read ────────────────────────────────────────────────

struct Oracle {
    doc: Value,
}

impl Oracle {
    fn load() -> Self {
        Self {
            doc: serde_json::from_str(ORACLE).expect("the oracle export parses"),
        }
    }

    fn floats(&self, pointer: &str) -> Vec<f32> {
        self.doc
            .pointer(pointer)
            .unwrap_or_else(|| panic!("the oracle has no {pointer}"))
            .as_array()
            .expect("an array")
            .iter()
            .map(|v| v.as_f64().expect("a number") as f32)
            .collect()
    }

    fn count(&self, pointer: &str) -> usize {
        self.doc
            .pointer(pointer)
            .unwrap_or_else(|| panic!("the oracle has no {pointer}"))
            .as_u64()
            .expect("a count") as usize
    }

    fn ran(&self, pointer: &str) -> bool {
        self.doc
            .pointer(pointer)
            .unwrap_or_else(|| panic!("the oracle has no {pointer}"))
            .as_bool()
            .expect("a flag")
    }

    /// One position's slice of a flattened `[positions, width]` field.
    fn row(&self, pointer: &str, position: usize, width: usize) -> Vec<f32> {
        let flat = self.floats(pointer);
        assert_eq!(
            flat.len(),
            POSITIONS * width,
            "{pointer} is not [3, {width}]"
        );
        flat[position * width..(position + 1) * width].to_vec()
    }

    fn site_pair(&self, layer: usize, site: HcSite) -> (Vec<f32>, Vec<f32>) {
        let (norm, proj) = match site {
            HcSite::Attention => ("attn_res_norm", "attn_res_proj"),
            HcSite::Ffn => ("mlp_res_norm", "mlp_res_proj"),
        };
        (
            self.floats(&format!("/weights/layers/{layer}/{norm}")),
            self.floats(&format!("/weights/layers/{layer}/{proj}")),
        )
    }

    fn exit_pair(&self) -> (Vec<f32>, Vec<f32>) {
        (
            self.floats("/weights/exit/norm"),
            self.floats("/weights/exit/proj"),
        )
    }

    /// The snapshot values taken at boundary layers STRICTLY BEFORE
    /// `layer`, oldest first — the set the attention site of `layer`
    /// reduces over. Reconstructed from the recorded events rather than
    /// recomputed, so the test reads the oracle rather than re-deriving
    /// it.
    fn snapshots_before(&self, layer: usize, position: usize) -> Vec<Vec<f32>> {
        (0..layer)
            .filter(|l| {
                self.doc
                    .pointer(&format!("/witness/{l}/snapshot_event/taken"))
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
            })
            .map(|l| {
                self.row(
                    &format!("/witness/{l}/snapshot_event/value"),
                    position,
                    HIDDEN,
                )
            })
            .collect()
    }

    /// The set the MLP site of `layer` reduces over: the above, plus this
    /// layer's own snapshot when it is a boundary.
    fn snapshots_through(&self, layer: usize, position: usize) -> Vec<Vec<f32>> {
        let mut snaps = self.snapshots_before(layer, position);
        if self
            .doc
            .pointer(&format!("/witness/{layer}/snapshot_event/taken"))
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            snaps.push(self.row(
                &format!("/witness/{layer}/snapshot_event/value"),
                position,
                HIDDEN,
            ));
        }
        snaps
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "comparing different shapes");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn close(actual: &[f32], expected: &[f32], what: &str) {
    let diff = max_abs_diff(actual, expected);
    assert!(
        diff <= TOLERANCE,
        "{what}: max |diff| {diff:e} exceeds {TOLERANCE:e}"
    );
}

// ── The witness ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
struct SiteRow {
    layer: usize,
    site: HcSite,
    candidate_count: usize,
    snapshot_count_before: usize,
    probs: Vec<f32>,
    mixed: Vec<f32>,
    prefix_before: Vec<f32>,
    prefix_after: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
struct BoundaryRow {
    layer: usize,
    snapshots_before: usize,
    snapshots_after: usize,
    value: Vec<f32>,
    entering_prefix: Vec<f32>,
}

/// One entry per observation, IN EMISSION ORDER — the ordering claim of
/// the topology is a claim about this sequence.
#[derive(Debug, Clone, PartialEq)]
enum Event {
    Site(SiteRow),
    Boundary(BoundaryRow),
}

#[derive(Default)]
struct Witness {
    events: Vec<Event>,
}

impl StepObserver for Witness {
    fn event(&mut self, _event: StepEvent) {}

    fn attention_residual_site(&mut self, r: AttnResSiteRecord<'_>) {
        self.events.push(Event::Site(SiteRow {
            layer: r.layer,
            site: r.site,
            candidate_count: r.candidate_count,
            snapshot_count_before: r.snapshot_count_before,
            probs: r.probs.to_vec(),
            mixed: r.mixed_vector.to_vec(),
            prefix_before: r.prefix_before.to_vec(),
            prefix_after: r.prefix_after.to_vec(),
        }));
    }

    fn attention_residual_boundary(&mut self, r: AttnResBoundaryRecord<'_>) {
        self.events.push(Event::Boundary(BoundaryRow {
            layer: r.layer,
            snapshots_before: r.snapshots_before,
            snapshots_after: r.snapshots_after,
            value: r.value.to_vec(),
            entering_prefix: r.entering_prefix.to_vec(),
        }));
    }
}

impl Witness {
    fn sites(&self) -> Vec<&SiteRow> {
        self.events
            .iter()
            .filter_map(|e| match e {
                Event::Site(s) => Some(s),
                Event::Boundary(_) => None,
            })
            .collect()
    }

    fn boundaries(&self) -> Vec<&BoundaryRow> {
        self.events
            .iter()
            .filter_map(|e| match e {
                Event::Boundary(b) => Some(b),
                Event::Site(_) => None,
            })
            .collect()
    }

    fn site(&self, layer: usize, site: HcSite) -> Option<&SiteRow> {
        self.sites()
            .into_iter()
            .find(|s| s.layer == layer && s.site == site)
    }
}

// ── The substrate ───────────────────────────────────────────────────

struct Substrate {
    _source: tempfile::TempDir,
    container: tempfile::TempDir,
    inspection: SystemInspection,
    plan: ComponentOpPlan,
}

/// Deterministic small values, distinct per seed — the ordinary
/// operators' weights, which the topology does not care about beyond
/// their producing distinguishable candidates.
fn values(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 0.4
        })
        .collect()
}

fn norm_weights(len: usize, seed: u64) -> Vec<f32> {
    values(len, seed).iter().map(|v| 1.0 + v).collect()
}

/// A dense softmax stack that DECLARES the period and ships the four
/// site operands on every layer plus the exit pair — the oracle's own
/// pairs, at f32, so the reduction runs on exactly the values the oracle
/// ran on.
fn substrate() -> Substrate {
    let oracle = Oracle::load();
    let source = tempfile::tempdir().unwrap();
    std::fs::write(
        source.path().join("config.json"),
        serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "torch_dtype": "float32",
            "model_type": "llama",
            "hidden_size": HIDDEN,
            "num_hidden_layers": LAYERS,
            "intermediate_size": INTER,
            "num_attention_heads": HEADS,
            "num_key_value_heads": HEADS,
            "head_dim": HEAD_DIM,
            "vocab_size": VOCAB,
            "rms_norm_eps": NORM_EPS,
            "rope_theta": 10000.0,
            "attn_res_block_size": BLOCK
        })
        .to_string(),
    )
    .unwrap();

    let rows = HEADS * HEAD_DIM;
    let mut shard = ShardBuilder::new();
    shard.push(
        "model.embed_tokens.weight",
        &[VOCAB, HIDDEN],
        &values(VOCAB * HIDDEN, 1),
    );
    shard.push("model.norm.weight", &[HIDDEN], &norm_weights(HIDDEN, 2));
    shard.push(
        "lm_head.weight",
        &[VOCAB, HIDDEN],
        &values(VOCAB * HIDDEN, 3),
    );
    let (exit_norm, exit_proj) = oracle.exit_pair();
    shard.push("model.output_attn_res_norm.weight", &[HIDDEN], &exit_norm);
    shard.push(
        "model.output_attn_res_proj.weight",
        &[1, HIDDEN],
        &exit_proj,
    );
    for layer in 0..LAYERS {
        let seed = 100 + layer as u64 * 10;
        let p = format!("model.layers.{layer}");
        for (leaf, shape, vals) in [
            (
                "self_attn.q_proj.weight",
                vec![rows, HIDDEN],
                values(rows * HIDDEN, seed),
            ),
            (
                "self_attn.k_proj.weight",
                vec![rows, HIDDEN],
                values(rows * HIDDEN, seed + 1),
            ),
            (
                "self_attn.v_proj.weight",
                vec![rows, HIDDEN],
                values(rows * HIDDEN, seed + 2),
            ),
            (
                "self_attn.o_proj.weight",
                vec![HIDDEN, rows],
                values(HIDDEN * rows, seed + 3),
            ),
            (
                "input_layernorm.weight",
                vec![HIDDEN],
                norm_weights(HIDDEN, seed + 4),
            ),
            (
                "post_attention_layernorm.weight",
                vec![HIDDEN],
                norm_weights(HIDDEN, seed + 5),
            ),
            (
                "mlp.gate_proj.weight",
                vec![INTER, HIDDEN],
                values(INTER * HIDDEN, seed + 6),
            ),
            (
                "mlp.up_proj.weight",
                vec![INTER, HIDDEN],
                values(INTER * HIDDEN, seed + 7),
            ),
            (
                "mlp.down_proj.weight",
                vec![HIDDEN, INTER],
                values(HIDDEN * INTER, seed + 8),
            ),
        ] {
            shard.push(&format!("{p}.{leaf}"), &shape, &vals);
        }
        // The oracle's own site pairs, at the layer they belong to.
        for (site, norm_leaf, proj_leaf) in [
            (
                HcSite::Attention,
                "self_attention_res_norm.weight",
                "self_attention_res_proj.weight",
            ),
            (HcSite::Ffn, "mlp_res_norm.weight", "mlp_res_proj.weight"),
        ] {
            let (norm, proj) = oracle.site_pair(layer, site);
            shard.push(&format!("{p}.{norm_leaf}"), &[HIDDEN], &norm);
            shard.push(&format!("{p}.{proj_leaf}"), &[1, HIDDEN], &proj);
        }
    }
    shard.write(source.path());

    let inventory = larql_models::inventory::build_inventory(source.path()).unwrap();
    let named = vec![("attn-res-substrate".to_string(), inventory)];
    let container = tempfile::tempdir().unwrap();
    let system = plan_system(&named);
    encode_graph(&system.graph, &named, container.path()).unwrap();
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    let plan = outcome
        .plan
        .unwrap_or_else(|| panic!("the substrate closes: {:?}", outcome.defects));
    Substrate {
        _source: source,
        container,
        inspection,
        plan,
    }
}

/// Prepare through the 2a WITNESS SEAM — the public loader still
/// refuses, and a test that reached the traversal through it would be
/// proving the refusal had already lifted.
fn prepare(sub: &Substrate) -> (OperandStore, PreparedOperands) {
    let store = OperandStore::open(sub.container.path(), &sub.inspection).unwrap();
    let ops = PreparedOperands::load_for_attention_residual_witness(
        &sub.plan,
        &store,
        &ReferenceBackend::new(),
        ExecutionSlice::Full,
    )
    .expect("the witness seam prepares an attention-residual plan");
    (store, ops)
}

struct Run {
    witness: Witness,
    exit: Vec<f32>,
}

/// One decode step over the substrate under `mutation`, with everything
/// it observed.
fn run(sub: &Substrate, mutation: Mutation) -> Run {
    let (_store, ops) = prepare(sub);
    let backend = ReferenceBackend::new();
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(&sub.plan, &ops, &backend, &mut kv).unwrap();
    let mut witness = Witness::default();
    let step = session
        .step_mutated(1, &mut witness, mutation)
        .expect("the decode step runs the topology");
    Run {
        witness,
        exit: step.exit.expect("a whole-stack image reduces at the exit"),
    }
}

// ── A1: the foreign comparison ──────────────────────────────────────

/// **The arithmetic, against a reference this build did not write.**
///
/// Every site the oracle recorded is replayed from the oracle's OWN
/// entering state — the layer's prefix and the snapshot set
/// reconstructed from its recorded boundary events — and the
/// probabilities and mixed vector are compared. Three positions, both
/// sites, seven layers, plus the exit.
///
/// This is what makes the module a transcription rather than an
/// invention: nothing here is compared against another Rust function.
#[test]
fn every_site_reproduces_the_oracles_probabilities_and_mixed_vector() {
    let oracle = Oracle::load();
    let mut checked = 0;
    for layer in 0..LAYERS {
        for (site, key) in [
            (HcSite::Attention, "attention_site"),
            (HcSite::Ffn, "mlp_site"),
        ] {
            if !oracle.ran(&format!("/witness/{layer}/{key}/ran")) {
                continue;
            }
            let candidates = oracle.count(&format!("/witness/{layer}/{key}/candidate_count"));
            let (norm, proj) = oracle.site_pair(layer, site);
            for position in 0..POSITIONS {
                // The entering state, from the oracle. The attention site
                // enters on the layer's own prefix; the mlp site enters on
                // the post-attention prefix the oracle recorded for it.
                let (prefix, snapshots) = match site {
                    HcSite::Attention => (
                        oracle.row(&format!("/witness/{layer}/prefix_in"), position, HIDDEN),
                        oracle.snapshots_before(layer, position),
                    ),
                    HcSite::Ffn => (
                        oracle.row(
                            &format!("/witness/{layer}/{key}/prefix_in"),
                            position,
                            HIDDEN,
                        ),
                        oracle.snapshots_through(layer, position),
                    ),
                };
                let mut history = History::new(prefix);
                for snapshot in snapshots {
                    history.push_snapshot(snapshot);
                }
                assert_eq!(
                    history.candidate_count(),
                    candidates,
                    "layer {layer} {key} position {position}: reconstructed candidate count"
                );
                let reduction = attention_residual::reduce(
                    &history,
                    attention_residual::SitePair {
                        norm: &norm,
                        proj: &proj,
                    },
                    NORM_EPS,
                    Mutation::None,
                )
                .expect("the reduction runs");
                close(
                    &reduction.probs,
                    &oracle.row(
                        &format!("/witness/{layer}/{key}/softmax_probs"),
                        position,
                        candidates,
                    ),
                    &format!("layer {layer} {key} position {position} probs"),
                );
                close(
                    &reduction.mixed,
                    &oracle.row(
                        &format!("/witness/{layer}/{key}/mixed_vector"),
                        position,
                        HIDDEN,
                    ),
                    &format!("layer {layer} {key} position {position} mixed"),
                );
                checked += 1;
            }
        }
    }
    // 13 reducing sites x 3 positions: layer 0 contributes its mlp site
    // alone, every other layer both.
    assert_eq!(checked, 13 * POSITIONS, "sites compared");

    // The exit, the same way.
    let (norm, proj) = oracle.exit_pair();
    let candidates = oracle.count("/exit/candidate_count");
    for position in 0..POSITIONS {
        let mut history = History::new(oracle.row("/exit/prefix_in", position, HIDDEN));
        for snapshot in oracle.snapshots_through(LAYERS - 1, position) {
            history.push_snapshot(snapshot);
        }
        assert_eq!(history.candidate_count(), candidates);
        let reduction = attention_residual::reduce(
            &history,
            attention_residual::SitePair {
                norm: &norm,
                proj: &proj,
            },
            NORM_EPS,
            Mutation::None,
        )
        .unwrap();
        close(
            &reduction.probs,
            &oracle.row("/exit/softmax_probs", position, candidates),
            "exit probs",
        );
        close(
            &reduction.mixed,
            &oracle.row("/exit/mixed_vector", position, HIDDEN),
            "exit mixed",
        );
    }
}

// ── A4: the schedule, structurally ──────────────────────────────────

/// **The schedule the reference spells, read off a real decode run.**
///
/// Every assertion here is about which records exist and what they
/// count, in emission order — the plane on which the two zero-delta
/// properties live.
#[test]
fn the_decode_traversal_emits_the_oracles_schedule() {
    let sub = substrate();
    let run = run(&sub, Mutation::None);
    let oracle = Oracle::load();

    // Layer 0 emits NO attention-site record. Not a record over one
    // candidate — none at all, because the reference's guard finds an
    // empty snapshot set and does not reduce.
    assert!(
        run.witness.site(0, HcSite::Attention).is_none(),
        "layer 0 must emit no attention-site record: {:?}",
        run.witness.sites()
    );
    // ...and every later layer does.
    for layer in 1..LAYERS {
        assert!(
            run.witness.site(layer, HcSite::Attention).is_some(),
            "layer {layer} attention site"
        );
    }
    // The mlp site is unconditional — every layer, layer 0 included.
    for layer in 0..LAYERS {
        assert!(
            run.witness.site(layer, HcSite::Ffn).is_some(),
            "layer {layer} mlp site"
        );
    }

    // Layer 0's mlp site sees TWO candidates: the snapshot the boundary
    // event has already taken, and the prefix. Never one — the oracle
    // falsified that reading of the reference, and this is the assertion
    // that keeps it falsified.
    assert_eq!(
        run.witness.site(0, HcSite::Ffn).unwrap().candidate_count,
        2,
        "layer 0's mlp site mixes the snapshot and the prefix"
    );

    // Every candidate count, against the oracle's own schedule.
    for layer in 0..LAYERS {
        for (site, key) in [
            (HcSite::Attention, "attention_site"),
            (HcSite::Ffn, "mlp_site"),
        ] {
            let expected = oracle.count(&format!("/witness/{layer}/{key}/candidate_count"));
            match run.witness.site(layer, site) {
                Some(row) => assert_eq!(
                    row.candidate_count, expected,
                    "layer {layer} {key} candidate count"
                ),
                None => assert_eq!(expected, 0, "layer {layer} {key} was expected to reduce"),
            }
        }
    }

    // The ordering claim, per boundary: the ATTENTION site read the set
    // the event had not yet extended, and the MLP site read the one it
    // had.
    let boundaries = run.witness.boundaries();
    assert_eq!(
        boundaries.iter().map(|b| b.layer).collect::<Vec<_>>(),
        vec![0, 3, 6],
        "a boundary at every layer where layer % block == 0"
    );
    for boundary in &boundaries {
        assert_eq!(boundary.snapshots_after, boundary.snapshots_before + 1);
        // The snapshot is the ENTERING prefix state.
        assert_eq!(
            boundary.value, boundary.entering_prefix,
            "layer {} snapshots the entering state",
            boundary.layer
        );
        if let Some(attention) = run.witness.site(boundary.layer, HcSite::Attention) {
            assert_eq!(
                attention.snapshot_count_before, boundary.snapshots_before,
                "layer {} attention reads the OLD set",
                boundary.layer
            );
        }
        let mlp = run.witness.site(boundary.layer, HcSite::Ffn).unwrap();
        assert_eq!(
            mlp.snapshot_count_before, boundary.snapshots_after,
            "layer {} mlp reads the EXTENDED set",
            boundary.layer
        );
    }

    // Layer 3 spelled out, because it is the case the whole ordering
    // claim rests on and a count table can hide.
    let l3_attn = run.witness.site(3, HcSite::Attention).unwrap();
    assert_eq!(
        (l3_attn.snapshot_count_before, l3_attn.candidate_count),
        (1, 2)
    );
    let l3_boundary = boundaries.iter().find(|b| b.layer == 3).unwrap();
    assert_eq!(
        (l3_boundary.snapshots_before, l3_boundary.snapshots_after),
        (1, 2)
    );
    let l3_mlp = run.witness.site(3, HcSite::Ffn).unwrap();
    assert_eq!(
        (l3_mlp.snapshot_count_before, l3_mlp.candidate_count),
        (2, 3)
    );

    // A2: the write is an ADD, except where a boundary reset the prefix
    // — there the attention branch's output BECOMES the prefix.
    for row in run.witness.sites() {
        if row.layer >= LAYERS {
            continue; // the exit's pseudo-record
        }
        let boundary_reset = row.site == HcSite::Attention
            && attention_residual::is_block_boundary(row.layer, BLOCK);
        if !boundary_reset {
            let delta: Vec<f32> = row
                .prefix_after
                .iter()
                .zip(&row.prefix_before)
                .map(|(a, b)| a - b)
                .collect();
            assert!(
                delta.iter().any(|d| d.abs() > 1e-6),
                "layer {} {:?}: the branch contributed nothing",
                row.layer,
                row.site
            );
        }
    }

    // A5: the exit ran, over every snapshot plus the prefix.
    let exit = run
        .witness
        .sites()
        .into_iter()
        .find(|s| s.layer == LAYERS)
        .expect("the exit emits a record");
    assert_eq!(exit.candidate_count, oracle.count("/exit/candidate_count"));
    assert_eq!(exit.snapshot_count_before, oracle.count("/exit/snapshots"));
    assert_eq!(
        run.exit, exit.mixed,
        "the step's exit IS the exit reduction"
    );

    // A6: the instrument can see. A substrate whose probabilities have
    // saturated proves nothing, and would report every control below as
    // passing while rejecting none of them.
    for row in run.witness.sites() {
        let max = row.probs.iter().copied().fold(f32::MIN, f32::max);
        let min = row.probs.iter().copied().fold(f32::MAX, f32::min);
        assert!(
            max <= MAX_PROB && min >= MIN_PROB,
            "layer {} {:?}: probabilities {:?} outside the oracle's band — the substrate \
             cannot see the candidate-set controls",
            row.layer,
            row.site,
            row.probs
        );
    }
}

// ── The rejecting controls ──────────────────────────────────────────

/// Every mutation the oracle measured above zero must move this
/// substrate's exit vector too.
#[test]
fn every_value_visible_mutation_is_caught() {
    let sub = substrate();
    let reference = run(&sub, Mutation::None);
    for mutation in [
        Mutation::AttnResSiteOverNewSnapshots,
        Mutation::AttnResSnapshotIsMixedVector,
        Mutation::AttnResSnapshotAfterAttention,
        Mutation::AttnResMlpSiteSkippedAtLayer0,
        Mutation::AttnResMixOverNormalisedCandidates,
        Mutation::AttnResScoreWithoutRmsNorm,
        Mutation::AttnResExitSkipped,
        Mutation::AttnResExitUsesALayerPair,
    ] {
        let mutated = run(&sub, mutation);
        let diff = max_abs_diff(&mutated.exit, &reference.exit);
        assert!(
            diff > 1e-3,
            "{mutation:?} left the exit vector unchanged (max |diff| {diff:e}); the oracle \
             measured this defect above zero, so a substrate that cannot see it is not a witness"
        );
    }
}

/// **The two the oracle proved unreachable by value, caught by shape.**
///
/// Both must leave the exit vector bit-identical — that is the oracle's
/// measurement, reproduced here rather than taken on trust — and both
/// must change the record stream. A run that reported either as caught
/// by a value comparison would have a broken assertion, and this test
/// asserts the zero as firmly as it asserts the structural difference.
#[test]
fn the_two_numerically_inert_mutations_are_caught_by_the_witness_alone() {
    let sub = substrate();
    let reference = run(&sub, Mutation::None);

    // Layer 0's attention site, run instead of skipped. Softmax over one
    // candidate is the identity, so nothing moves.
    let regularised = run(&sub, Mutation::AttnResLayer0AttentionSiteRuns);
    assert_eq!(
        max_abs_diff(&regularised.exit, &reference.exit),
        0.0,
        "the oracle measured this at exactly 0.0; a non-zero here means the traversal \
         changed something else as well"
    );
    let extra = regularised.witness.site(0, HcSite::Attention).expect(
        "the regularised traversal emits a layer-0 attention record where the reference \
         emits none — the only observable difference",
    );
    assert_eq!(
        extra.candidate_count, 1,
        "it reduces over the prefix alone, which is why it is invisible"
    );
    assert!(reference.witness.site(0, HcSite::Attention).is_none());

    // The mlp site given the attention site's guard. The guard never
    // fires, because no site in this schedule sees an empty set.
    let guarded = run(&sub, Mutation::AttnResMlpSiteGuardedOnNonEmpty);
    assert_eq!(
        max_abs_diff(&guarded.exit, &reference.exit),
        0.0,
        "the oracle measured this at exactly 0.0"
    );
    assert_eq!(
        guarded.witness.sites().len(),
        reference.witness.sites().len(),
        "the guard fires nowhere: every mlp site still reduces"
    );
}

// ── What has NOT lifted ─────────────────────────────────────────────

/// 2a proves the decode traversal and lifts nothing. The public loader
/// still refuses the topology by name, and the batch path refuses it
/// too — so the seam that proves the traversal cannot make either look
/// supported.
#[test]
fn the_public_loader_and_the_batch_path_still_refuse() {
    let sub = substrate();
    let store = OperandStore::open(sub.container.path(), &sub.inspection).unwrap();

    let public = match PreparedOperands::load(
        &sub.plan,
        &store,
        &ReferenceBackend::new(),
        ExecutionSlice::Full,
    ) {
        Ok(_) => panic!("the public loader must still refuse an attention-residual plan"),
        Err(err) => err.to_string(),
    };
    assert!(public.contains("cannot execute it"), "{public}");
    assert!(public.contains("traversal"), "{public}");

    // The batch path, reached the ONLY way it can be while the public
    // loader still refuses: through an image prepared by the 2a witness
    // seam. That is precisely the risk the freeze names — the seam
    // produces a `PreparedOperands` the batch traversal would otherwise
    // consume happily, running a component with a snapshot history over
    // a plane of plain `[hidden]` rows and dropping every snapshot and
    // every boundary event silently.
    //
    // Going through `execute_text` here would prove nothing: it prepares
    // through the PUBLIC loader, which refuses first, so the batch guard
    // would never be reached and a test asserting on its message would
    // be asserting on the topology refusal wearing a different name.
    let (_store, ops) = prepare(&sub);
    let batch = match super::super::execute_prepared_streaming(
        &sub.plan,
        &ops,
        &[1, 2, 3],
        &ReferenceBackend::new(),
        None,
        &mut |_| Ok(()),
    ) {
        Ok(_) => panic!("the batch traversal must refuse an attention-residual image until 2b"),
        Err(err) => err.to_string(),
    };
    assert!(batch.contains("no snapshot history"), "{batch}");
    assert!(batch.contains("2b"), "{batch}");
}

/// **Observation is optional, and the traversal must not depend on being
/// watched.**
///
/// `StepObserver`'s two attention-residual hooks have default no-op
/// bodies, so an observer that ignores them is a supported caller — and
/// the production path is exactly that caller. This runs the same step
/// under `NoopObserver`, which overrides neither, and requires the exit
/// vector to be bit-identical to the observed run's.
///
/// Not coverage padding: the site record borrows the reduction's own
/// `probs` and `mixed`, so a traversal that computed anything inside the
/// observer call — or skipped work when nobody was listening — would
/// diverge here and nowhere else. It is also the only test in this file
/// that executes those default bodies, which is how the gap was found.
#[test]
fn the_traversal_runs_identically_with_an_unobserving_observer() {
    let sub = substrate();
    let observed = run(&sub, Mutation::None);

    let (_store, ops) = prepare(&sub);
    let backend = ReferenceBackend::new();
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(&sub.plan, &ops, &backend, &mut kv).unwrap();
    let unobserved = session
        .step_mutated(1, &mut NoopObserver, Mutation::None)
        .expect("an unobserved step runs the topology")
        .exit
        .expect("a whole-stack image reduces at the exit");

    assert_eq!(
        unobserved, observed.exit,
        "the exit differs when nobody is observing; the traversal is doing work inside its \
         observation points"
    );
}
