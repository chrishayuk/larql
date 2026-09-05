//! The device backend's seam, tested without any device.
//!
//! `DevicePlanBackend` is generic over `larql-compute`'s `MatMul`, so
//! its routing and fail-closed contract are testable with local trait
//! implementors: one whose gemv is a plain loop (parity against the
//! reference backend), one that widens f16 weights and loops (the f16
//! residency path's arithmetic), and one with no gemv kernel at all
//! (every matmul path must refuse, not fall back). Real-device parity
//! for `--backend metal` runs at the CLI layer, where the concrete
//! Metal backend is injected.

use larql_compute::backend::MatMul;
use larql_compute::cpu::ops::q4_common::f16_to_f32;
use ndarray::{Array2, ArrayView2};

use super::golden::{miniature_glimmer, G_TOKENS};
use super::{lcg_values, norm_values, ShardBuilder};
use crate::format::vindex3::encode::encode_system;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::WeightFormat;
use crate::format::vindex3::opplan::exec::device::DevicePlanBackend;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::exec::{execute_plan, ExecutionTrace};
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};
use std::path::Path;

/// Above loop-vs-loop reassociation noise on the miniature's shapes,
/// far below any semantic effect.
const NOISE_CEILING: f32 = 1e-5;

/// The f16 realisation's tolerance. The miniature fixture stores f32
/// tensors, so its f16 load path rounds to nearest (2⁻¹¹ relative per
/// weight) and the error compounds through norms, softmax and two
/// layers to ~1% — unlike the real container's bf16 payload, which
/// converts exactly (pinned by the unit tests in `weights.rs`). This
/// test is a plumbing gate (layout, padding, dtype length — which fail
/// as garbage, orders of magnitude past this ceiling); the bit-exact
/// f16 decode-vs-batch parity test and the real-model table carry the
/// precision claims.
const F16_CEILING: f32 = 0.05;

/// A "device" whose gemv is a plain in-order loop. `pub(super)` so the
/// decode-parity tests can drive the same device through a session.
pub(super) struct LoopDevice;

impl MatMul for LoopDevice {
    fn matmul(&self, _a: ArrayView2<f32>, _b: ArrayView2<f32>) -> Array2<f32> {
        unimplemented!("the plan backend only dispatches gemv")
    }

    fn matmul_transb(&self, _a: ArrayView2<f32>, _b: ArrayView2<f32>) -> Array2<f32> {
        unimplemented!("the plan backend only dispatches gemv")
    }

    fn f32_gemv_force(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<Vec<f32>> {
        let (n, k) = (w.shape()[0], w.shape()[1]);
        if x.len() != k {
            return None;
        }
        Some(
            (0..n)
                .map(|row| (0..k).map(|col| w[[row, col]] * x[col]).sum())
                .collect(),
        )
    }

    fn f16_gemv_force(&self, w_f16: &[u8], x: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
        if w_f16.len() < n * k * 2 || x.len() != k {
            return None;
        }
        Some(
            (0..n)
                .map(|row| {
                    (0..k)
                        .map(|col| {
                            let at = (row * k + col) * 2;
                            let bits = u16::from_le_bytes([w_f16[at], w_f16[at + 1]]);
                            f16_to_f32(bits) * x[col]
                        })
                        .sum()
                })
                .collect(),
        )
    }

    fn mxfp4_gemv(
        &self,
        packed: &[u8],
        scales: &[u8],
        x: &[f32],
        n: usize,
        k: usize,
    ) -> Option<Vec<f32>> {
        if !k.is_multiple_of(32) || x.len() < k {
            return None;
        }
        if packed.len() < n * (k / 32) * 16 || scales.len() < n * (k / 32) {
            return None;
        }
        Some(Self::mxfp4_loop(packed, scales, x, n, k))
    }
}

impl LoopDevice {
    /// CPU MXFP4 gemv: table + e8m0 decode, plain in-order loop — the
    /// independent arithmetic the device seam's MXFP4 routing is
    /// tested against.
    fn mxfp4_loop(packed: &[u8], scales: &[u8], x: &[f32], n: usize, k: usize) -> Vec<f32> {
        use larql_models::quant::mxfp4::{e8m0_to_f32, MXFP4_TABLE};
        let groups = k / 32;
        (0..n)
            .map(|row| {
                let mut acc = 0.0f32;
                for g in 0..groups {
                    let scale = e8m0_to_f32(scales[row * groups + g]);
                    let bytes = &packed[(row * groups + g) * 16..][..16];
                    for (b, &byte) in bytes.iter().enumerate() {
                        let e0 = g * 32 + 2 * b;
                        acc += MXFP4_TABLE[(byte & 0x0F) as usize] * scale * x[e0];
                        acc += MXFP4_TABLE[(byte >> 4) as usize] * scale * x[e0 + 1];
                    }
                }
                acc
            })
            .collect()
    }
}

/// A device with no gemv kernel — the trait default `None`.
struct KernellessDevice;

impl MatMul for KernellessDevice {
    fn matmul(&self, _a: ArrayView2<f32>, _b: ArrayView2<f32>) -> Array2<f32> {
        unimplemented!()
    }

    fn matmul_transb(&self, _a: ArrayView2<f32>, _b: ArrayView2<f32>) -> Array2<f32> {
        unimplemented!()
    }
}

/// A 32-aligned miniature Glimmer: same judged family and tensor
/// estate as `miniature_glimmer`, with every matrix K dimension a
/// multiple of the MXFP4 32-element group (the real model's are; the
/// deliberately awkward hidden-12 fixture correctly refuses to
/// quantise).
pub(super) fn aligned_glimmer(dir: &Path) {
    const HIDDEN: usize = 64;
    const Q_HEADS: usize = 2;
    const KV_HEADS: usize = 1;
    const HEAD_DIM: usize = 32;
    const FFN: usize = 96;
    const VOCAB: usize = 32;
    std::fs::write(
        dir.join("config.json"),
        serde_json::json!({
            "architectures": ["MuseGlimmerForConditionalGeneration"],
            "torch_dtype": "float32",
            "model_type": "muse_glimmer_text",
            "hidden_size": HIDDEN,
            "num_hidden_layers": 2,
            "intermediate_size": FFN,
            "num_attention_heads": Q_HEADS,
            "num_key_value_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "vocab_size": VOCAB,
            "sliding_window": 3,
            "rms_norm_eps": 1e-5,
            "rope_parameters": { "rope_theta": 500000.0, "rope_type": "default" },
            "layer_types": ["sliding_attention", "full_attention"],
            "layer_rope_theta": [500000.0, 0.0],
            "qk_scale_factor": 3.87,
            "output_multiplier": 0.196,
            "post_norm_eps": 1e-8,
            "attn_logit_softcapping": 50.0,
            "final_logit_softcapping": 20.0
        })
        .to_string(),
    )
    .unwrap();
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
    for layer in 0..2usize {
        let seed = 700 + layer as u64 * 30;
        let prefix = format!("model.layers.{layer}");
        for (suffix, shape, values) in [
            (
                "self_attn.q_proj.weight",
                vec![q_rows, HIDDEN],
                lcg_values(q_rows * HIDDEN, seed),
            ),
            (
                "self_attn.k_proj.weight",
                vec![kv_rows, HIDDEN],
                lcg_values(kv_rows * HIDDEN, seed + 1),
            ),
            (
                "self_attn.v_proj.weight",
                vec![kv_rows, HIDDEN],
                lcg_values(kv_rows * HIDDEN, seed + 2),
            ),
            (
                "self_attn.o_proj.weight",
                vec![HIDDEN, q_rows],
                lcg_values(HIDDEN * q_rows, seed + 3),
            ),
            (
                "self_attn.gate_proj.weight",
                vec![q_rows, HIDDEN],
                lcg_values(q_rows * HIDDEN, seed + 4),
            ),
            (
                "input_layernorm.weight",
                vec![HIDDEN],
                norm_values(HIDDEN, seed + 5),
            ),
            (
                "post_attention_layernorm.weight",
                vec![HIDDEN],
                norm_values(HIDDEN, seed + 6),
            ),
            (
                "pre_feedforward_layernorm.weight",
                vec![HIDDEN],
                norm_values(HIDDEN, seed + 7),
            ),
            (
                "post_feedforward_layernorm.weight",
                vec![HIDDEN],
                norm_values(HIDDEN, seed + 8),
            ),
            (
                "mlp.gate_proj.weight",
                vec![FFN, HIDDEN],
                lcg_values(FFN * HIDDEN, seed + 9),
            ),
            (
                "mlp.up_proj.weight",
                vec![FFN, HIDDEN],
                lcg_values(FFN * HIDDEN, seed + 10),
            ),
            (
                "mlp.down_proj.weight",
                vec![HIDDEN, FFN],
                lcg_values(HIDDEN * FFN, seed + 11),
            ),
        ] {
            shard.push(&format!("{prefix}.{suffix}"), &shape, &values);
        }
    }
    shard.write(dir);
}

/// Encode `aligned_glimmer` and open its plan + store.
pub(super) fn aligned_fixture() -> (tempfile::TempDir, ComponentOpPlan, OperandStore) {
    let dir = tempfile::tempdir().unwrap();
    aligned_glimmer(dir.path());
    let inventory = larql_models::inventory::build_inventory(dir.path()).unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_system(
        &[("mini-glimmer-aligned".to_string(), inventory)],
        container.path(),
    )
    .unwrap();
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    (container, plan, store)
}

fn fixture() -> (tempfile::TempDir, ComponentOpPlan, OperandStore) {
    let dir = tempfile::tempdir().unwrap();
    miniature_glimmer(dir.path());
    let inventory = larql_models::inventory::build_inventory(dir.path()).unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_system(&[("mini-glimmer".to_string(), inventory)], container.path()).unwrap();
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    (container, plan, store)
}

fn max_abs(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    a.iter()
        .zip(b)
        .flat_map(|(ra, rb)| ra.iter().zip(rb).map(|(x, y)| (x - y).abs()))
        .fold(0.0, f32::max)
}

fn assert_traces_agree(a: &ExecutionTrace, b: &ExecutionTrace, ceiling: f32, label: &str) {
    for (index, (da, db)) in a.layers.iter().zip(&b.layers).enumerate() {
        let delta = max_abs(da.post_layer.rows(), db.post_layer.rows());
        assert!(delta < ceiling, "{label}: layer {index} max_abs {delta}");
    }
    let logits_delta = max_abs(
        std::slice::from_ref(a.logits.as_ref().unwrap()),
        std::slice::from_ref(b.logits.as_ref().unwrap()),
    );
    assert!(
        logits_delta < ceiling,
        "{label}: logits max_abs {logits_delta}"
    );
}

#[test]
fn a_device_backend_matches_the_reference_layer_by_layer() {
    let (_c, plan, store) = fixture();
    let device = DevicePlanBackend::new(LoopDevice, "loop-device-test", WeightFormat::F32);
    let on_device = execute_plan(&plan, &store, &G_TOKENS, &device).unwrap();
    let on_reference = execute_plan(&plan, &store, &G_TOKENS, &ReferenceBackend::new()).unwrap();
    assert_traces_agree(
        &on_device,
        &on_reference,
        NOISE_CEILING,
        "f32 device vs reference",
    );
}

/// The f16 residency path: the interpreter loads f16 operands for a
/// backend that declares them, and the arithmetic stays within the
/// documented conversion tolerance of the f32 reference.
#[test]
fn an_f16_device_backend_matches_the_reference_within_the_conversion_floor() {
    let (_c, plan, store) = fixture();
    let device = DevicePlanBackend::new(LoopDevice, "loop-device-f16-test", WeightFormat::F16);
    let on_device = execute_plan(&plan, &store, &G_TOKENS, &device).unwrap();
    let on_reference = execute_plan(&plan, &store, &G_TOKENS, &ReferenceBackend::new()).unwrap();
    assert_traces_agree(
        &on_device,
        &on_reference,
        F16_CEILING,
        "f16 device vs reference",
    );
}

#[test]
fn a_kernelless_device_fails_closed_naming_the_shape() {
    let (_c, plan, store) = fixture();
    let device = DevicePlanBackend::new(KernellessDevice, "kernelless-test", WeightFormat::F32);
    let err = execute_plan(&plan, &store, &G_TOKENS, &device).unwrap_err();
    assert!(
        err.to_string().contains("f32_gemv") && err.to_string().contains("refused"),
        "{err}"
    );
}

/// The MXFP4 realisation runs the plan end to end and stays loosely in
/// the reference's neighbourhood — 4-bit weights are a coarse
/// realisation, so this is a routing/plumbing gate; the bit-exact
/// decode-vs-batch parity and the real-model table carry the numeric
/// claims.
#[test]
fn an_mxfp4_device_backend_executes_the_plan_end_to_end() {
    let (_c, plan, store) = aligned_fixture();
    let device = DevicePlanBackend::new(LoopDevice, "loop-device-mxfp4-test", WeightFormat::Mxfp4);
    let on_device = execute_plan(&plan, &store, &G_TOKENS, &device).unwrap();
    let on_reference = execute_plan(&plan, &store, &G_TOKENS, &ReferenceBackend::new()).unwrap();
    // Direction only: 4-bit quantisation of the miniature's ±0.05
    // weights is very coarse; the argmax agreeing would already be
    // luck. Assert the outputs are finite and correlated, not close.
    let logits = on_device.logits.as_ref().unwrap();
    assert!(logits.iter().all(|v| v.is_finite()));
    let reference = on_reference.logits.as_ref().unwrap();
    let dot: f32 = logits.iter().zip(reference).map(|(a, b)| a * b).sum();
    let na: f32 = logits.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = reference.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        dot / (na * nb) > 0.5,
        "mxfp4 logits decorrelated from reference: cos {}",
        dot / (na * nb)
    );
}

/// An mxfp4-declaring backend on a device with no MXFP4 kernel must
/// refuse, never dequantise and take another path.
#[test]
fn a_kernelless_device_fails_closed_for_mxfp4_too() {
    let (_c, plan, store) = aligned_fixture();
    let device = DevicePlanBackend::new(KernellessDevice, "kernelless-mxfp4", WeightFormat::Mxfp4);
    let err = execute_plan(&plan, &store, &G_TOKENS, &device).unwrap_err();
    assert!(
        err.to_string().contains("mxfp4_gemv") && err.to_string().contains("refused"),
        "{err}"
    );
}

/// An f16-declaring backend on a device with no f16 kernel must refuse,
/// never quietly widen and take the f32 path.
#[test]
fn a_kernelless_device_fails_closed_for_f16_too() {
    let (_c, plan, store) = fixture();
    let device = DevicePlanBackend::new(KernellessDevice, "kernelless-f16", WeightFormat::F16);
    let err = execute_plan(&plan, &store, &G_TOKENS, &device).unwrap_err();
    assert!(
        err.to_string().contains("f16_gemv") && err.to_string().contains("refused"),
        "{err}"
    );
}
