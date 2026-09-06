//! A derived static-shard container declares a per-layer dense-FFN width
//! and stores physically narrower gate/up/down tensors for the layers it
//! names (E30). The claim under test is not "a narrower FFN runs" but
//! "the plan reads the DECLARED width, checks the tensors against it, and
//! the executor produces the arithmetic those tensors define".
//!
//! The exact arm: two checkpoints with identical weights except that in
//! one, layer 1's gate rows OUTSIDE the retained set are zeroed (so those
//! channels contribute exactly `activation(0) * up = 0`), and in the other
//! those rows are physically REMOVED and the width declared. Removing a
//! run of exact zeros from a sum leaves every partial sum bit-identical,
//! so the two containers must produce bit-identical logits on the
//! reference backend. A build that ignored the declaration would either
//! refuse the narrow tensors or read them at the wrong stride; neither
//! passes.

use serial_test::serial;
use std::path::Path;

use crate::format::vindex3::encode::encode_system;
use crate::format::vindex3::fixtures::{
    lcg_values, norm_values, ShardBuilder, DENSE_HEAD_DIM, DENSE_HIDDEN, DENSE_INTERMEDIATE,
    DENSE_KV_HEADS, DENSE_LAYERS, DENSE_Q_HEADS, DENSE_VOCAB,
};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::{plan_component_ops, LayerFfn, OpPlanOutcome};

const TOKEN: u32 = 7;
/// The layer whose FFN is narrowed.
const NARROW_LAYER: usize = 1;
/// The retained channels: every third one, so the retained set is neither
/// a prefix nor contiguous — a stride that a prefix-only slicer gets wrong.
const RETAIN_STRIDE: usize = 3;
/// The key the derived checkpoint declares its widths under.
const WIDTH_KEY: &str = "larql_ffn_intermediate_size_by_layer";

fn retained() -> Vec<usize> {
    (0..DENSE_INTERMEDIATE).step_by(RETAIN_STRIDE).collect()
}

/// How the narrow layer is written.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Estate {
    /// Full-width tensors; gate rows outside the retained set are ZERO.
    ZeroedFull,
    /// Only the retained rows/columns, width declared per layer.
    SlicedDeclared,
    /// Only the retained rows/columns, but the declaration says full width.
    SlicedUndeclared,
    /// Full-width tensors with a declaration covering the wrong number of
    /// layers.
    BadDeclarationLength,
}

fn model(dir: &Path, estate: Estate) {
    let keep = retained();
    let k = keep.len();
    let mut config = serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "torch_dtype": "float32",
        "model_type": "llama",
        "tie_word_embeddings": false,
        "hidden_size": DENSE_HIDDEN,
        "num_hidden_layers": DENSE_LAYERS,
        "intermediate_size": DENSE_INTERMEDIATE,
        "num_attention_heads": DENSE_Q_HEADS,
        "num_key_value_heads": DENSE_KV_HEADS,
        "head_dim": DENSE_HEAD_DIM,
        "vocab_size": DENSE_VOCAB,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0
    });
    match estate {
        Estate::SlicedDeclared => {
            let widths: Vec<usize> = (0..DENSE_LAYERS)
                .map(|l| {
                    if l == NARROW_LAYER {
                        k
                    } else {
                        DENSE_INTERMEDIATE
                    }
                })
                .collect();
            config[WIDTH_KEY] = serde_json::json!(widths);
        }
        Estate::BadDeclarationLength => {
            config[WIDTH_KEY] = serde_json::json!(vec![DENSE_INTERMEDIATE]);
        }
        Estate::ZeroedFull | Estate::SlicedUndeclared => {}
    }
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();

    let q_rows = DENSE_Q_HEADS * DENSE_HEAD_DIM;
    let kv_rows = DENSE_KV_HEADS * DENSE_HEAD_DIM;
    let mut shard = ShardBuilder::new();
    shard.push(
        "model.embed_tokens.weight",
        &[DENSE_VOCAB, DENSE_HIDDEN],
        &lcg_values(DENSE_VOCAB * DENSE_HIDDEN, 1),
    );
    shard.push(
        "model.norm.weight",
        &[DENSE_HIDDEN],
        &norm_values(DENSE_HIDDEN, 2),
    );
    shard.push(
        "lm_head.weight",
        &[DENSE_VOCAB, DENSE_HIDDEN],
        &lcg_values(DENSE_VOCAB * DENSE_HIDDEN, 3),
    );
    for layer in 0..DENSE_LAYERS {
        let seed = 100 + layer as u64 * 10;
        let prefix = format!("model.layers.{layer}");
        for (leaf, shape, s) in [
            ("self_attn.q_proj.weight", vec![q_rows, DENSE_HIDDEN], seed),
            (
                "self_attn.k_proj.weight",
                vec![kv_rows, DENSE_HIDDEN],
                seed + 1,
            ),
            (
                "self_attn.v_proj.weight",
                vec![kv_rows, DENSE_HIDDEN],
                seed + 2,
            ),
            (
                "self_attn.o_proj.weight",
                vec![DENSE_HIDDEN, q_rows],
                seed + 3,
            ),
        ] {
            let n = shape.iter().product();
            shard.push(&format!("{prefix}.{leaf}"), &shape, &lcg_values(n, s));
        }
        shard.push(
            &format!("{prefix}.input_layernorm.weight"),
            &[DENSE_HIDDEN],
            &norm_values(DENSE_HIDDEN, seed + 4),
        );
        shard.push(
            &format!("{prefix}.post_attention_layernorm.weight"),
            &[DENSE_HIDDEN],
            &norm_values(DENSE_HIDDEN, seed + 5),
        );
        let gate = lcg_values(DENSE_INTERMEDIATE * DENSE_HIDDEN, seed + 6);
        let up = lcg_values(DENSE_INTERMEDIATE * DENSE_HIDDEN, seed + 7);
        let down = lcg_values(DENSE_HIDDEN * DENSE_INTERMEDIATE, seed + 8);
        let narrow = layer == NARROW_LAYER && estate != Estate::BadDeclarationLength;
        match (narrow, estate) {
            (true, Estate::ZeroedFull) => {
                // Same shapes, but every non-retained gate row is zero:
                // `silu(0) * up = 0`, so those channels add exact zeros.
                let mut gate_z = gate.clone();
                for row in 0..DENSE_INTERMEDIATE {
                    if !keep.contains(&row) {
                        gate_z[row * DENSE_HIDDEN..(row + 1) * DENSE_HIDDEN].fill(0.0);
                    }
                }
                shard.push(
                    &format!("{prefix}.mlp.gate_proj.weight"),
                    &[DENSE_INTERMEDIATE, DENSE_HIDDEN],
                    &gate_z,
                );
                shard.push(
                    &format!("{prefix}.mlp.up_proj.weight"),
                    &[DENSE_INTERMEDIATE, DENSE_HIDDEN],
                    &up,
                );
                shard.push(
                    &format!("{prefix}.mlp.down_proj.weight"),
                    &[DENSE_HIDDEN, DENSE_INTERMEDIATE],
                    &down,
                );
            }
            (true, Estate::SlicedDeclared | Estate::SlicedUndeclared) => {
                // Physically narrower: retained ROWS of gate/up, retained
                // COLUMNS of down, in the retained order.
                let rows = |w: &[f32]| -> Vec<f32> {
                    keep.iter()
                        .flat_map(|&r| w[r * DENSE_HIDDEN..(r + 1) * DENSE_HIDDEN].iter().copied())
                        .collect()
                };
                let mut cols: Vec<f32> = Vec::with_capacity(DENSE_HIDDEN * k);
                for h in 0..DENSE_HIDDEN {
                    for &c in &keep {
                        cols.push(down[h * DENSE_INTERMEDIATE + c]);
                    }
                }
                shard.push(
                    &format!("{prefix}.mlp.gate_proj.weight"),
                    &[k, DENSE_HIDDEN],
                    &rows(&gate),
                );
                shard.push(
                    &format!("{prefix}.mlp.up_proj.weight"),
                    &[k, DENSE_HIDDEN],
                    &rows(&up),
                );
                shard.push(
                    &format!("{prefix}.mlp.down_proj.weight"),
                    &[DENSE_HIDDEN, k],
                    &cols,
                );
            }
            _ => {
                shard.push(
                    &format!("{prefix}.mlp.gate_proj.weight"),
                    &[DENSE_INTERMEDIATE, DENSE_HIDDEN],
                    &gate,
                );
                shard.push(
                    &format!("{prefix}.mlp.up_proj.weight"),
                    &[DENSE_INTERMEDIATE, DENSE_HIDDEN],
                    &up,
                );
                shard.push(
                    &format!("{prefix}.mlp.down_proj.weight"),
                    &[DENSE_HIDDEN, DENSE_INTERMEDIATE],
                    &down,
                );
            }
        }
    }
    shard.write(dir);
}

/// Encode the fixture; the encoder runs the plan closure and REFUSES a
/// container whose operands do not close, so a refusal surfaces here as
/// the encode error, itemised — the same text `larql vindex3 encode` prints.
fn encode_of(estate: Estate) -> Result<tempfile::TempDir, String> {
    let src = tempfile::tempdir().unwrap();
    model(src.path(), estate);
    let inventory =
        larql_models::inventory::build_inventory(src.path()).expect("fixture inventory");
    let container = tempfile::tempdir().unwrap();
    encode_system(
        &[("target-artifact".to_string(), inventory)],
        container.path(),
    )
    .map(|_| container)
    .map_err(|e| e.to_string())
}

fn outcome_of(
    estate: Estate,
) -> (
    tempfile::TempDir,
    OpPlanOutcome,
    crate::format::vindex3::inspect::SystemInspection,
) {
    let container = encode_of(estate).unwrap_or_else(|e| panic!("{estate:?} must encode: {e}"));
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    (container, outcome, inspection)
}

fn logits_of(estate: Estate) -> (Vec<f32>, Vec<usize>) {
    let (container, outcome, inspection) = outcome_of(estate);
    let plan = outcome
        .plan
        .unwrap_or_else(|| panic!("{estate:?} must lower: {:?}", outcome.defects));
    let widths: Vec<usize> = plan
        .layers
        .iter()
        .map(|l| match &l.ffn {
            Some(LayerFfn::Dense(op)) => op.intermediate_size,
            other => panic!("fixture layers are dense, found {other:?}"),
        })
        .collect();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    let backend = ReferenceBackend::new();
    let mut session = DecodeSession::new(&plan, &store, &backend).unwrap();
    let out = session.step(TOKEN).unwrap();
    (out.logits.expect("the fixture carries a head"), widths)
}

/// **The declaration reaches the op, per layer.** The plan states layer
/// 1's width as the DECLARED narrow value and layer 0's as the dense one.
#[test]
#[serial]
fn the_plan_states_each_layers_declared_width() {
    let (_, widths) = logits_of(Estate::SlicedDeclared);
    assert_eq!(widths, vec![DENSE_INTERMEDIATE, retained().len()]);
}

/// **Exact arm.** Removing zero-contribution channels leaves every partial
/// sum untouched, so the sliced-and-declared container reproduces the
/// zeroed full-width container bit for bit.
#[test]
#[serial]
fn a_sliced_declared_layer_reproduces_the_zeroed_full_layer_bit_for_bit() {
    let (zeroed, _) = logits_of(Estate::ZeroedFull);
    let (sliced, _) = logits_of(Estate::SlicedDeclared);
    assert_eq!(zeroed.len(), sliced.len());
    let max_abs = zeroed
        .iter()
        .zip(&sliced)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(max_abs, 0.0, "sliced vs zeroed logits differ by {max_abs}");
    // And the fixture is not degenerate: the logits carry signal.
    assert!(zeroed.iter().any(|v| v.abs() > 1e-3));
}

/// **Checked, not assumed.** Narrow tensors under a full-width
/// declaration are refused — by the ENCODER, whose closure check names
/// the layer's three FFN tensors with the expected and stored shapes.
#[test]
#[serial]
fn narrow_tensors_under_a_full_width_declaration_are_refused() {
    let err =
        encode_of(Estate::SlicedUndeclared).expect_err("an undeclared narrow FFN must not encode");
    assert!(err.contains("GeometryMismatch"), "{err}");
    for leaf in ["gate_proj", "up_proj", "down_proj"] {
        assert!(
            err.contains(&format!("{NARROW_LAYER}.mlp.{leaf}.weight")),
            "the refusal must name layer {NARROW_LAYER}'s {leaf}: {err}"
        );
    }
    let k = retained().len();
    assert!(
        err.contains(&format!("actual: [{k}, {DENSE_HIDDEN}]")),
        "{err}"
    );
    assert!(
        err.contains(&format!("expected: [{DENSE_INTERMEDIATE}, {DENSE_HIDDEN}]")),
        "{err}"
    );
}

/// **A declaration nobody can hold is refused before any layer is
/// shaped against it** — the encoder names the declaration itself, not a
/// tensor, because no tensor is wrong.
#[test]
#[serial]
fn a_declaration_covering_the_wrong_number_of_layers_is_refused() {
    let err =
        encode_of(Estate::BadDeclarationLength).expect_err("a short declaration must not encode");
    assert!(err.contains("FfnWidthDeclaration"), "{err}");
    assert!(
        err.contains(&format!(
            "declares 1 per-layer FFN widths for a {DENSE_LAYERS}-layer component"
        )),
        "{err}"
    );
    assert!(
        !err.contains("GeometryMismatch"),
        "no tensor is wrong here: {err}"
    );
}
