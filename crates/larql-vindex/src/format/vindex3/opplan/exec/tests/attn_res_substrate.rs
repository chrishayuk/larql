//! **The K3-ATTNRES-1 substrate, and the oracle it is built from.**
//!
//! Shared by the decode witness (2a) and the batch witness (2b) so that
//! the two transitions are scored against ONE fixture. A second copy of
//! this file would let the two paths agree with each other while both
//! drifted from the oracle, which is the exact failure A7 exists to
//! rule out.
//!
//! Nothing here asserts anything. It reads the oracle export, builds a
//! synthetic stack carrying the oracle's own site pairs, and prepares it
//! through the witness seam; every claim about what those produce lives
//! in the two witness modules.

use serde_json::Value;

use super::super::observe::HcSite;
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
pub(super) const HIDDEN: usize = 5;
pub(super) const LAYERS: usize = 7;
pub(super) const BLOCK: usize = 3;
pub(super) const POSITIONS: usize = 3;
pub(super) const VOCAB: usize = 7;
pub(super) const NORM_EPS: f64 = 1e-5;

// Ordinary operator geometry, deliberately not derived from `HIDDEN`, so
// no shape coincidence can let a transposed operand pass.
pub(super) const HEADS: usize = 1;
pub(super) const HEAD_DIM: usize = 4;
pub(super) const INTER: usize = 4;

/// f32 storage throughout and a stand-in-free comparison, so this is
/// transcription noise and nothing else.
pub(super) const TOLERANCE: f32 = 5e-5;

/// The floor and ceiling the oracle ships. A substrate outside this band
/// has a saturated or starved softmax, on which every candidate-set
/// control is invisible — the failure the oracle demonstrated on itself.
pub(super) const MIN_PROB: f32 = 5e-3;
pub(super) const MAX_PROB: f32 = 0.98;

pub(super) const ORACLE: &str = include_str!("attn_res_oracle.json");

// ── The oracle, read ────────────────────────────────────────────────

pub(super) struct Oracle {
    doc: Value,
}

impl Oracle {
    pub(super) fn load() -> Self {
        Self {
            doc: serde_json::from_str(ORACLE).expect("the oracle export parses"),
        }
    }

    pub(super) fn floats(&self, pointer: &str) -> Vec<f32> {
        self.doc
            .pointer(pointer)
            .unwrap_or_else(|| panic!("the oracle has no {pointer}"))
            .as_array()
            .expect("an array")
            .iter()
            .map(|v| v.as_f64().expect("a number") as f32)
            .collect()
    }

    pub(super) fn count(&self, pointer: &str) -> usize {
        self.doc
            .pointer(pointer)
            .unwrap_or_else(|| panic!("the oracle has no {pointer}"))
            .as_u64()
            .expect("a count") as usize
    }

    pub(super) fn ran(&self, pointer: &str) -> bool {
        self.doc
            .pointer(pointer)
            .unwrap_or_else(|| panic!("the oracle has no {pointer}"))
            .as_bool()
            .expect("a flag")
    }

    /// One position's slice of a flattened `[positions, width]` field.
    pub(super) fn row(&self, pointer: &str, position: usize, width: usize) -> Vec<f32> {
        let flat = self.floats(pointer);
        assert_eq!(
            flat.len(),
            POSITIONS * width,
            "{pointer} is not [3, {width}]"
        );
        flat[position * width..(position + 1) * width].to_vec()
    }

    pub(super) fn site_pair(&self, layer: usize, site: HcSite) -> (Vec<f32>, Vec<f32>) {
        let (norm, proj) = match site {
            HcSite::Attention => ("attn_res_norm", "attn_res_proj"),
            HcSite::Ffn => ("mlp_res_norm", "mlp_res_proj"),
        };
        (
            self.floats(&format!("/weights/layers/{layer}/{norm}")),
            self.floats(&format!("/weights/layers/{layer}/{proj}")),
        )
    }

    pub(super) fn exit_pair(&self) -> (Vec<f32>, Vec<f32>) {
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
    pub(super) fn snapshots_before(&self, layer: usize, position: usize) -> Vec<Vec<f32>> {
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
    pub(super) fn snapshots_through(&self, layer: usize, position: usize) -> Vec<Vec<f32>> {
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

pub(super) fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "comparing different shapes");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

pub(super) fn close(actual: &[f32], expected: &[f32], what: &str) {
    let diff = max_abs_diff(actual, expected);
    assert!(
        diff <= TOLERANCE,
        "{what}: max |diff| {diff:e} exceeds {TOLERANCE:e}"
    );
}

// ── The substrate ───────────────────────────────────────────────────

pub(super) struct Substrate {
    pub(super) _source: tempfile::TempDir,
    pub(super) container: tempfile::TempDir,
    pub(super) inspection: SystemInspection,
    pub(super) plan: ComponentOpPlan,
}

/// Deterministic small values, distinct per seed — the ordinary
/// operators' weights, which the topology does not care about beyond
/// their producing distinguishable candidates.
pub(super) fn values(len: usize, seed: u64) -> Vec<f32> {
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

pub(super) fn norm_weights(len: usize, seed: u64) -> Vec<f32> {
    values(len, seed).iter().map(|v| 1.0 + v).collect()
}

/// A dense softmax stack that DECLARES the period and ships the four
/// site operands on every layer plus the exit pair — the oracle's own
/// pairs, at f32, so the reduction runs on exactly the values the oracle
/// ran on.
pub(super) fn substrate() -> Substrate {
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
pub(super) fn prepare(sub: &Substrate) -> (OperandStore, PreparedOperands) {
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
