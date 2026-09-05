//! THE Stage A gate: container-driven naive execution ≡ checkpoint-driven
//! production execution, layer by layer, on the dense F32 fixture.

use larql_compute::forward::hooks::RecordHook;

use super::{dense_f32_model, HIDDEN, LAYERS, VOCAB};
use crate::format::vindex3::encode::encode_system;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::{execute_text, ExecutionTrace};
use crate::format::vindex3::opplan::plan_component_ops;

/// Tolerance for naive-loop vs BLAS f32 on a 64-wide model: differences
/// are reassociation-only — observed ~7e-8 hidden / ~2.4e-7 logits — so
/// the bound sits ~100× above the measured noise floor and far below
/// anything a semantic defect could produce.
const HIDDEN_TOLERANCE: f32 = 1e-5;
const LOGIT_TOLERANCE: f32 = 3e-5;

const TOKENS: [u32; 6] = [3, 17, 42, 99, 7, 63];

struct OracleTrace {
    post_attention: Vec<Vec<Vec<f32>>>,
    post_layer: Vec<Vec<Vec<f32>>>,
    logits: Vec<f32>,
}

/// The production forward, checkpoint-driven, with hook taps.
fn oracle(dir: &std::path::Path) -> OracleTrace {
    let weights = larql_models::load_model_dir(dir).unwrap();
    let view = larql_models::WeightsView::dense(&weights);
    let ffn = larql_compute::ffn::WeightFfn { weights: &weights };
    let mut hook = RecordHook::for_layers(0..LAYERS);
    let mut h = larql_compute::forward::embed_tokens_pub(&weights, &TOKENS);
    for layer in 0..LAYERS {
        let (h_next, _, _, _) = larql_compute::forward::layer::run_layer_with_capture_hooked(
            view, &h, layer, &ffn, false, false, None, None, &mut hook,
        )
        .unwrap();
        h = h_next;
    }
    let raw = larql_compute::forward::predict::forward_raw_logits(view, &TOKENS, None);
    let to_rows = |a: &ndarray::Array2<f32>| -> Vec<Vec<f32>> {
        a.outer_iter().map(|row| row.to_vec()).collect()
    };
    OracleTrace {
        post_attention: (0..LAYERS)
            .map(|l| to_rows(&hook.post_attention[&l]))
            .collect(),
        post_layer: (0..LAYERS).map(|l| to_rows(&hook.post_layer[&l])).collect(),
        logits: raw.logits.to_vec(),
    }
}

/// The plan executor, container-driven.
fn plan_execution(dir: &std::path::Path) -> ExecutionTrace {
    let inventory = larql_models::inventory::build_inventory(dir).unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_system(
        &[("dense-artifact".to_string(), inventory)],
        container.path(),
    )
    .unwrap();
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    assert!(outcome.closed(), "closure must hold: {:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    execute_text(&plan, &store, &TOKENS).unwrap()
}

fn max_abs_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    a.iter()
        .zip(b)
        .flat_map(|(ra, rb)| ra.iter().zip(rb).map(|(x, y)| (x - y).abs()))
        .fold(0.0, f32::max)
}

/// THE gate: every layer's post-attention and post-layer states agree,
/// then the logits — the whole program, localised per layer on failure.
#[test]
fn container_execution_matches_the_checkpoint_forward_layer_by_layer() {
    let dir = tempfile::tempdir().unwrap();
    dense_f32_model(dir.path());
    let oracle = oracle(dir.path());
    let executed = plan_execution(dir.path());

    assert_eq!(executed.layers.len(), LAYERS);
    for layer in 0..LAYERS {
        let attn = max_abs_diff(
            executed.layers[layer].post_attention.rows(),
            &oracle.post_attention[layer],
        );
        assert!(
            attn < HIDDEN_TOLERANCE,
            "layer {layer} post_attention diverges: max_abs {attn}"
        );
        let post = max_abs_diff(
            executed.layers[layer].post_layer.rows(),
            &oracle.post_layer[layer],
        );
        assert!(
            post < HIDDEN_TOLERANCE,
            "layer {layer} post_layer diverges: max_abs {post}"
        );
    }

    assert_eq!(executed.final_hidden().len(), HIDDEN);
    let logits = executed.logits.expect("plan carries an output op");
    assert_eq!(logits.len(), VOCAB);
    let worst = logits
        .iter()
        .zip(&oracle.logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < LOGIT_TOLERANCE, "logits diverge: max_abs {worst}");

    // Instrument check: the two sides share no arithmetic (naive loops
    // vs BLAS), so bit-identical agreement everywhere would mean the
    // comparison is accidentally degenerate, not that execution is
    // perfect. Reassociation must show up somewhere.
    let last = LAYERS - 1;
    let residual_diff = max_abs_diff(
        executed.layers[last].post_layer.rows(),
        &oracle.post_layer[last],
    );
    assert!(
        residual_diff > 0.0 || worst > 0.0,
        "independent arithmetic cannot agree bit-for-bit across a whole forward"
    );
    eprintln!("parity margins: last-layer max_abs {residual_diff:e}, logits max_abs {worst:e}");
}

/// The negative control the parity claim needs: an instrument that
/// cannot fail on a known-different input proves nothing. Flip one
/// stored weight byte in the container — parity must break.
#[test]
fn parity_fails_on_a_corrupted_operand() {
    let dir = tempfile::tempdir().unwrap();
    dense_f32_model(dir.path());
    let oracle = oracle(dir.path());

    let inventory = larql_models::inventory::build_inventory(dir.path()).unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_system(
        &[("dense-artifact".to_string(), inventory)],
        container.path(),
    )
    .unwrap();
    // Corrupt one value inside a row a test token actually reads:
    // sign-bit flip on the first component of token 3's embedding.
    let victim = container
        .path()
        .join("segments")
        .join("target.embedding.bin");
    let (_, payload_start) =
        crate::format::vindex3::encode::segment::read_segment_header(&victim).unwrap();
    let mut bytes = std::fs::read(&victim).unwrap();
    let float_size = std::mem::size_of::<f32>();
    let target_byte = payload_start as usize + 3 * HIDDEN * float_size + (float_size - 1);
    bytes[target_byte] ^= 0x80;
    std::fs::write(&victim, bytes).unwrap();

    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    let plan = outcome.plan.unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    let executed = execute_text(&plan, &store, &TOKENS).unwrap();

    let diff = max_abs_diff(
        executed.layers[LAYERS - 1].post_layer.rows(),
        &oracle.post_layer[LAYERS - 1],
    );
    assert!(
        diff >= HIDDEN_TOLERANCE,
        "a corrupted operand must break parity (diff {diff})"
    );
}
