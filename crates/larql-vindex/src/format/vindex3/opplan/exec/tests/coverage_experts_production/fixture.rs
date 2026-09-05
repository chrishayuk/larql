//! The routed miniature, its carrier variant, and the load/compare
//! helpers shared by the expert-bank and production-backend gates.

use crate::format::vindex3::opplan::OperandRef;
use std::path::Path;

use larql_models::config::ExpertFormat;

use super::super::device::LoopDevice;
use super::super::{lcg_values, norm_values, ShardBuilder};
use crate::format::vindex3::encode::encode_system_unenforced as encode_system;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::{WeightFormat, WeightFormats, WeightSlice};
use crate::format::vindex3::opplan::exec::device::DevicePlanBackend;
use crate::format::vindex3::opplan::exec::experts::FfnOperands;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::weights::{quantize_mxfp4, LoadedWeight};
use crate::format::vindex3::opplan::{
    plan_component_ops, ComponentOpPlan, ExpertBank, LayerFfn, RoutedFfnOp,
};

// ── Routed miniature geometry: MXFP4 needs k ≡ 0 (mod 32) on both projections ──
pub(super) const HIDDEN: usize = 32;
pub(super) const INTER: usize = 32;
pub(super) const EXPERTS: usize = 4;
pub(super) const TOP_K: usize = 2;
pub(super) const Q_HEADS: usize = 2;
pub(super) const KV_HEADS: usize = 1;
pub(super) const HEAD_DIM: usize = 16;
pub(super) const VOCAB: usize = 29;
pub(super) const LAYERS: usize = 2;
pub(super) const WINDOW: usize = 4;
/// GPT-OSS's published clamp.
pub(super) const SWIGLU_LIMIT: f32 = 7.0;
/// Gate and up share one fused operand.
pub(super) const FUSED_BRANCHES: usize = larql_models::quant::mxfp4::FUSED_HALVES;
/// MXFP4 group geometry, as the packer lays it out.
pub(super) const MXFP4_GROUP_ELEMS: usize = larql_models::quant::mxfp4::MXFP4_GROUP_ELEMS;
pub(super) const MXFP4_GROUP_BYTES: usize = larql_models::quant::mxfp4::MXFP4_GROUP_BYTES;
/// Bank-relative suffixes of the classified MXFP4 streams and of the
/// unclassified BF16 copies (which land in the bank object because they
/// share its prefix, but block closure — so they are only ever reached by
/// an op edited to point at them).
pub(super) const GATE_UP_BLOCKS: &str = "mlp.experts.gate_up_proj_blocks";
pub(super) const DOWN_BLOCKS: &str = "mlp.experts.down_proj_blocks";
pub(super) const GATE_UP_BF16: &str = "mlp.experts.gate_up_proj_bf16";
pub(super) const DOWN_BF16: &str = "mlp.experts.down_proj_bf16";
pub(super) const BLOCKS_SUFFIX: &str = "_blocks";
pub(super) const SCALES_SUFFIX: &str = "_scales";
pub(super) const BF16_SUFFIX: &str = "_bf16";
pub(super) const DTYPE_U8: &str = "U8";
pub(super) const DTYPE_BF16: &str = "BF16";
/// Seed of the normalised FFN input the routed tests apply.
pub(super) const INPUT_SEED: u64 = 9_001;

/// Loop-vs-BLAS reassociation on 32-wide operands, and the bf16 → f16 /
/// e2m1 → f16 conversions are exact, so f16 realisations sit here too.
pub(super) const NOISE_CEILING: f32 = 1e-5;
/// A hand-built call's epsilon.
pub(super) const EPS: f64 = 1e-5;

/// Round-to-nearest-even bf16 of `v`, back as the f32 it denotes.
pub(super) fn bf16_round(v: f32) -> f32 {
    f32::from_bits(u32::from(bf16_bits(v)) << 16)
}

/// The bf16 bit pattern nearest `v` (ties to even).
pub(super) fn bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let rounding = 0x7FFF + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding)) >> 16) as u16
}

/// The GPT-OSS miniature with bf16-representable source values. With
/// `bf16_copies`, each layer's expert bank additionally carries an
/// unquantised BF16 copy of both projections under unclassified names.
pub(super) fn miniature_gpt_oss(dir: &Path, bf16_copies: bool) {
    let config = serde_json::json!({
        "architectures": ["GptOssForCausalLM"],
        "model_type": "gpt_oss",
        "torch_dtype": "float32",
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "intermediate_size": INTER,
        "num_attention_heads": Q_HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "num_local_experts": EXPERTS,
        "num_experts_per_tok": TOP_K,
        "sliding_window": WINDOW,
        "layer_types": ["sliding_attention", "full_attention"],
        "rope_theta": 10000.0,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 32.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "original_max_position_embeddings": 4096,
            "truncate": false
        },
        "rms_norm_eps": 1e-5,
        "attention_bias": true,
        "swiglu_limit": SWIGLU_LIMIT,
        "tie_word_embeddings": false,
        "quantization_config": {
            "quant_method": "mxfp4",
            "modules_to_not_convert": [
                "model.layers.*.self_attn",
                "model.layers.*.mlp.router",
                "model.embed_tokens",
                "lm_head"
            ]
        }
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();

    let q_rows = Q_HEADS * HEAD_DIM;
    let kv_rows = KV_HEADS * HEAD_DIM;
    let mut shard = ShardBuilder::new();
    shard.push(
        "model.embed_tokens.weight",
        &[VOCAB, HIDDEN],
        &lcg_values(VOCAB * HIDDEN, 1),
    );
    shard.push("model.norm.weight", &[HIDDEN], &norm_values(HIDDEN, 2));
    shard.push(
        "lm_head.weight",
        &[VOCAB, HIDDEN],
        &lcg_values(VOCAB * HIDDEN, 3),
    );
    for layer in 0..LAYERS {
        let seed = 700 + layer as u64 * 40;
        let prefix = format!("model.layers.{layer}");
        let scaled = |n: usize, s: u64, g: f32| -> Vec<f32> {
            lcg_values(n, s).into_iter().map(|v| v * g).collect()
        };
        for (suffix, shape, values) in [
            (
                "self_attn.q_proj.weight",
                vec![q_rows, HIDDEN],
                scaled(q_rows * HIDDEN, seed, 1.0),
            ),
            (
                "self_attn.k_proj.weight",
                vec![kv_rows, HIDDEN],
                scaled(kv_rows * HIDDEN, seed + 1, 1.0),
            ),
            (
                "self_attn.v_proj.weight",
                vec![kv_rows, HIDDEN],
                scaled(kv_rows * HIDDEN, seed + 2, 1.0),
            ),
            (
                "self_attn.o_proj.weight",
                vec![HIDDEN, q_rows],
                scaled(HIDDEN * q_rows, seed + 3, 1.0),
            ),
            (
                "self_attn.q_proj.bias",
                vec![q_rows],
                scaled(q_rows, seed + 4, 2.0),
            ),
            (
                "self_attn.k_proj.bias",
                vec![kv_rows],
                scaled(kv_rows, seed + 5, 2.0),
            ),
            (
                "self_attn.v_proj.bias",
                vec![kv_rows],
                scaled(kv_rows, seed + 6, 2.0),
            ),
            (
                "self_attn.o_proj.bias",
                vec![HIDDEN],
                scaled(HIDDEN, seed + 7, 2.0),
            ),
            (
                "self_attn.sinks",
                vec![Q_HEADS],
                scaled(Q_HEADS, seed + 8, 4.0),
            ),
            (
                "input_layernorm.weight",
                vec![HIDDEN],
                norm_values(HIDDEN, seed + 9),
            ),
            (
                "post_attention_layernorm.weight",
                vec![HIDDEN],
                norm_values(HIDDEN, seed + 10),
            ),
            (
                "mlp.router.weight",
                vec![EXPERTS, HIDDEN],
                scaled(EXPERTS * HIDDEN, seed + 11, 4.0),
            ),
            (
                "mlp.router.bias",
                vec![EXPERTS],
                scaled(EXPERTS, seed + 12, 1.0),
            ),
            (
                "mlp.experts.gate_up_proj_bias",
                vec![EXPERTS, FUSED_BRANCHES * INTER],
                scaled(EXPERTS * FUSED_BRANCHES * INTER, seed + 13, 1.0),
            ),
            (
                "mlp.experts.down_proj_bias",
                vec![EXPERTS, HIDDEN],
                scaled(EXPERTS * HIDDEN, seed + 14, 1.0),
            ),
        ] {
            shard.push(&format!("{prefix}.{suffix}"), &shape, &values);
        }
        // Packed MXFP4 experts from bf16-representable sources, so a BF16
        // copy of the same bank requantises to these exact bytes.
        for (suffix, bf16_suffix, rows, k, s) in [
            (
                GATE_UP_BLOCKS,
                GATE_UP_BF16,
                FUSED_BRANCHES * INTER,
                HIDDEN,
                seed + 15,
            ),
            (DOWN_BLOCKS, DOWN_BF16, HIDDEN, INTER, seed + 16),
        ] {
            let groups = k / MXFP4_GROUP_ELEMS;
            let mut blocks = Vec::new();
            let mut scales = Vec::new();
            let mut bf16 = Vec::new();
            for e in 0..EXPERTS {
                let values: Vec<f32> = scaled(rows * k, s + e as u64 * 100, 2.0)
                    .into_iter()
                    .map(bf16_round)
                    .collect();
                let LoadedWeight::Mxfp4 { packed, scales: sc } =
                    quantize_mxfp4(&values, rows, k, suffix).unwrap()
                else {
                    unreachable!()
                };
                blocks.extend_from_slice(&packed.as_slice()[..rows * groups * MXFP4_GROUP_BYTES]);
                scales.extend_from_slice(&sc.as_slice()[..rows * groups]);
                bf16.extend(values.iter().flat_map(|v| bf16_bits(*v).to_le_bytes()));
            }
            shard.push_bytes(
                &format!("{prefix}.{suffix}"),
                DTYPE_U8,
                &[EXPERTS, rows, groups, MXFP4_GROUP_BYTES],
                &blocks,
            );
            shard.push_bytes(
                &format!("{prefix}.{}", suffix.replace(BLOCKS_SUFFIX, SCALES_SUFFIX)),
                DTYPE_U8,
                &[EXPERTS, rows, groups],
                &scales,
            );
            if bf16_copies {
                shard.push_bytes(
                    &format!("{prefix}.{bf16_suffix}"),
                    DTYPE_BF16,
                    &[EXPERTS, rows, k],
                    &bf16,
                );
            }
        }
    }
    shard.write(dir);
}

/// Encode `dir` into a fresh container.
pub(super) fn encoded(dir: &Path, artifact: &str) -> tempfile::TempDir {
    let inventory = larql_models::inventory::build_inventory(dir).unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_system(&[(artifact.to_string(), inventory)], container.path()).unwrap();
    container
}

/// A closed plan and the store over the same container.
pub(super) fn closed_plan(container: &Path) -> (ComponentOpPlan, OperandStore) {
    let inspection = inspect_container(container, false).unwrap();
    let outcome = plan_component_ops(&inspection, container, "target").unwrap();
    assert!(outcome.closed(), "{:?}", outcome.defects);
    let store = OperandStore::open(container, &inspection).unwrap();
    (outcome.plan.unwrap(), store)
}

/// The routed miniature: its closed plan, its store, and layer 0's op.
pub(super) struct RoutedFixture {
    _dir: tempfile::TempDir,
    _container: tempfile::TempDir,
    pub(super) store: OperandStore,
    pub(super) op: RoutedFfnOp,
    pub(super) plan: ComponentOpPlan,
}

pub(super) fn routed_fixture() -> RoutedFixture {
    let dir = tempfile::tempdir().unwrap();
    miniature_gpt_oss(dir.path(), false);
    let container = encoded(dir.path(), "mini-gpt-oss");
    let (plan, store) = closed_plan(container.path());
    let op = plan.layers[0]
        .ffn
        .as_ref()
        .expect("layer 0 has an FFN")
        .routed()
        .expect("layer 0 is routed")
        .clone();
    RoutedFixture {
        _dir: dir,
        _container: container,
        store,
        op,
        plan,
    }
}

/// The store of a container whose bank also carries the BF16 copies —
/// not closable (the copies are unclassified), so the op comes from
/// [`routed_fixture`] and is edited to point at them.
pub(super) fn bf16_carrier_store() -> (tempfile::TempDir, tempfile::TempDir, OperandStore) {
    let dir = tempfile::tempdir().unwrap();
    miniature_gpt_oss(dir.path(), true);
    let container = encoded(dir.path(), "mini-gpt-oss");
    let inspection = inspect_container(container.path(), false).unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    (dir, container, store)
}

/// `op` re-pointed at the BF16 copies, declared `PackedBF16`.
pub(super) fn bf16_op(op: &RoutedFfnOp) -> RoutedFfnOp {
    let mut op = op.clone();
    op.expert_format = ExpertFormat::PackedBF16;
    let ExpertBank::Packed { gate_up, down } = &mut op.bank else {
        panic!("fixture builds a packed bank");
    };
    for projection in [gate_up, down] {
        projection.weights.tensor = projection
            .weights
            .tensor
            .replace(BLOCKS_SUFFIX, BF16_SUFFIX);
        projection.scales = None;
    }
    op
}

pub(super) fn routed(op: &RoutedFfnOp) -> LayerFfn {
    LayerFfn::Routed(Box::new(op.clone()))
}

pub(super) fn load(op: &RoutedFfnOp, store: &OperandStore, format: WeightFormat) -> FfnOperands {
    FfnOperands::load(
        &routed(op),
        store.into(),
        &|_: &OperandRef| Ok(format),
        format,
    )
    .unwrap()
}

pub(super) fn load_err(op: &RoutedFfnOp, store: &OperandStore, format: WeightFormat) -> String {
    FfnOperands::load(
        &routed(op),
        store.into(),
        &|_: &OperandRef| Ok(format),
        format,
    )
    .err()
    .expect("loading must refuse")
    .to_string()
}

/// A device backend that binds the FFN class in `format` and everything
/// else in f32 (the loop device has f32, f16 and MXFP4 gemvs).
pub(super) fn loop_device(format: WeightFormat) -> DevicePlanBackend<LoopDevice> {
    DevicePlanBackend::with_formats(
        LoopDevice,
        "loop-device-coverage",
        WeightFormats {
            attention: WeightFormat::F32,
            ffn: format,
            head: WeightFormat::F32,
        },
    )
}

pub(super) fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// The variant name of every matrix the operands expose.
pub(super) fn slice_kinds(operands: &FfnOperands) -> Vec<&'static str> {
    operands.weight_slices().iter().map(slice_kind).collect()
}

/// The two MXFP4 streams of every matrix, for byte comparison.
pub(super) fn mxfp4_bytes(operands: &FfnOperands) -> Vec<(Vec<u8>, Vec<u8>)> {
    operands
        .weight_slices()
        .iter()
        .map(|s| match s {
            WeightSlice::Mxfp4 { packed, scales } => (packed.to_vec(), scales.to_vec()),
            other => panic!("expected an MXFP4 slice, got {}", slice_kind(other)),
        })
        .collect()
}

/// The representation's name, for a panic message.
///
/// Delegates rather than matching again: this used to be a second copy of
/// the same table, and a copy is one variant away from disagreeing with
/// the one the refusals print.
pub(super) fn slice_kind(slice: &WeightSlice<'_>) -> &'static str {
    slice.representation()
}
