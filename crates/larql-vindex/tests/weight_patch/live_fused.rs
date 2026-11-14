//! The fused BF16 kernel over a patched tensor.
//!
//! The production CPU policy widens a matrix whose f32 image fits the performance cluster's L2 and keeps a
//! larger one as its stored BF16 bytes, run in place by the fused kernel. The other live tests use a fixture
//! every matrix of which widens, so they reach the stored-BF16 residency only at the loader. This one builds a
//! model whose FFN matrices are too large to widen and executes it end to end: the patched tensor must be
//! pinned to the stored-BF16 residency, exactly as its compiled twin is, and compute the same bits through it.

use std::path::Path;

use larql_vindex::format::vindex3::fixtures::{
    encode_fixture_container, lcg_values, norm_values, ShardBuilder, DENSE_HEAD_DIM, DENSE_HIDDEN,
    DENSE_KV_HEADS, DENSE_Q_HEADS, DENSE_VOCAB,
};
use larql_vindex::format::vindex3::opplan::exec::backend::WeightFormat;
use larql_vindex::format::vindex3::opplan::exec::operands::OperandSource;
use larql_vindex::format::vindex3::opplan::exec::prepared::{select_realizations, ExecutionSlice};
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
use larql_vindex::format::vindex3::opplan::exec::realization::RealizationRecord;
use larql_vindex::format::vindex3::weight_patch::{record_op, WeightEdit, WeightOp};

use crate::fixture::Source;
use crate::live_support::*;

/// One layer is enough: the point is one matrix the policy does not widen.
const LAYERS: usize = 1;
/// `[64, 65536]`: 16 MiB as f32, the default performance-cluster L2 and the largest any Apple performance
/// cluster reports, so the policy keeps it as stored BF16 (8 MiB, below the size at which it would stream).
const INTERMEDIATE: usize = 1 << 16;
/// The single layer's down projection.
const BIG_TENSOR: &str = "0.mlp.down_proj.weight";
const BF16_DTYPE: &str = "BF16";
/// Elements of row [`EDIT_ROW`] the patch edits.
const EDIT_COLUMNS: [usize; 3] = [7, 4096, 65_000];

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn write_wide_ffn_checkpoint(dir: &Path) {
    std::fs::write(
        dir.join("config.json"),
        serde_json::json!({
            "architectures": ["LlamaForCausalLM"], "torch_dtype": "bfloat16", "model_type": "llama",
            "tie_word_embeddings": false, "hidden_size": DENSE_HIDDEN, "num_hidden_layers": LAYERS,
            "intermediate_size": INTERMEDIATE, "num_attention_heads": DENSE_Q_HEADS,
            "num_key_value_heads": DENSE_KV_HEADS, "head_dim": DENSE_HEAD_DIM, "vocab_size": DENSE_VOCAB,
            "rms_norm_eps": 1e-5, "rope_theta": 10000.0
        })
        .to_string(),
    )
    .unwrap();
    let (q, kv) = (
        DENSE_Q_HEADS * DENSE_HEAD_DIM,
        DENSE_KV_HEADS * DENSE_HEAD_DIM,
    );
    let mut shard = ShardBuilder::new();
    let mut push = |name: &str, shape: &[usize], values: Vec<f32>| {
        shard.push_bytes(name, BF16_DTYPE, shape, &bf16_bytes(&values));
    };
    let matrix = |rows: usize, cols: usize, seed: u64| lcg_values(rows * cols, seed);
    push(
        "model.embed_tokens.weight",
        &[DENSE_VOCAB, DENSE_HIDDEN],
        matrix(DENSE_VOCAB, DENSE_HIDDEN, 1),
    );
    push(
        "model.norm.weight",
        &[DENSE_HIDDEN],
        norm_values(DENSE_HIDDEN, 2),
    );
    push(
        "lm_head.weight",
        &[DENSE_VOCAB, DENSE_HIDDEN],
        matrix(DENSE_VOCAB, DENSE_HIDDEN, 3),
    );
    let p = "model.layers.0";
    for (suffix, rows, cols, seed) in [
        ("self_attn.q_proj.weight", q, DENSE_HIDDEN, 100),
        ("self_attn.k_proj.weight", kv, DENSE_HIDDEN, 101),
        ("self_attn.v_proj.weight", kv, DENSE_HIDDEN, 102),
        ("self_attn.o_proj.weight", DENSE_HIDDEN, q, 103),
        ("mlp.gate_proj.weight", INTERMEDIATE, DENSE_HIDDEN, 106),
        ("mlp.up_proj.weight", INTERMEDIATE, DENSE_HIDDEN, 107),
        ("mlp.down_proj.weight", DENSE_HIDDEN, INTERMEDIATE, 108),
    ] {
        push(
            &format!("{p}.{suffix}"),
            &[rows, cols],
            matrix(rows, cols, seed),
        );
    }
    for (suffix, seed) in [
        ("input_layernorm.weight", 104),
        ("post_attention_layernorm.weight", 105),
    ] {
        push(
            &format!("{p}.{suffix}"),
            &[DENSE_HIDDEN],
            norm_values(DENSE_HIDDEN, seed),
        );
    }
    shard.write(dir);
}

/// The resident form the big tensor is pinned to.
fn pinned(records: &[RealizationRecord]) -> WeightFormat {
    records
        .iter()
        .find(|r| r.planned.operand.tensor == BIG_TENSOR)
        .expect("the plan projects through the big tensor")
        .selection
        .realization
        .format()
}

#[test]
fn a_patched_tensor_too_large_to_widen_runs_the_fused_bf16_kernel_as_its_compiled_twin_does() {
    let dir = tempfile::tempdir().unwrap();
    let checkpoint = dir.path().join("checkpoint");
    std::fs::create_dir(&checkpoint).unwrap();
    let container = dir.path().join("wide.vindex3");
    encode_fixture_container(write_wide_ffn_checkpoint, &checkpoint, &container, MODEL);
    let src = Source { dir, container };

    let edits: Vec<WeightEdit> = EDIT_COLUMNS
        .iter()
        .map(|&col| WeightEdit {
            row: EDIT_ROW,
            col,
            delta: EDIT_DELTA,
        })
        .collect();
    let staged = WeightOp::Delta(record_op(&src.container, BIG_TENSOR, &edits).unwrap());
    let compiled = compile(&src, &[staged.as_ref()], "wide-compiled.vindex3");
    let bound = bind(&src.container);
    let fresh = bind(&compiled);
    let overlay = session_with(&bound, vec![staged]);
    let overrides = overlay.live_weight_overrides().unwrap().unwrap();

    // The pin: stored BF16 for the patched tensor, in the session and in the compiled container alike.
    let backend = ProductionBackend::new();
    let live_records = select_realizations(
        &bound.plan,
        OperandSource::overlaid(&bound.store, &overrides),
        &backend,
        &ExecutionSlice::Full,
    )
    .unwrap();
    let stored_records = select_realizations(
        &fresh.plan,
        (&fresh.store).into(),
        &backend,
        &ExecutionSlice::Full,
    )
    .unwrap();
    assert_eq!(live_records, stored_records);
    assert_eq!(
        pinned(&live_records),
        WeightFormat::Bf16,
        "the patched tensor must be kept as stored BF16 for this test to reach the fused kernel"
    );

    let base = bound.compute(None);
    let live = bound.compute(Some(&overrides));
    overrides.ensure_patched_read().unwrap();
    assert_ne!(live.logits, base.logits, "the edit changes the logits");
    assert_eq!(
        live,
        fresh.compute(None),
        "and they are the compiled container's, bit for bit"
    );
}
